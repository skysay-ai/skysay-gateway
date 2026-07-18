# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Room-side synchronized evidence recorder for the LiveKit media path."""
from __future__ import annotations

import json
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

try:
    from .evidence_contract import semantic_response_audio_policy
except ImportError:  # copied into /app in the pinned sandbox image
    from evidence_contract import semantic_response_audio_policy


Track = Literal["caller", "agent"]


@dataclass(frozen=True)
class SpeechInterval:
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptItem:
    role: Track
    text: str
    at: float


class SynchronizedEvidenceRecorder:
    """Collects two real-time PCM tracks on one monotonic timeline.

    A separate LiveKit room participant feeds this recorder, so agent audio is
    captured after playout/interruptions rather than when TTS merely generated
    it.  Both tracks are padded to the same duration and per-turn clips are
    sliced from observed caller/agent speaking intervals.
    """

    def __init__(
        self,
        root: Path,
        *,
        sample_rate: int = 8_000,
        epoch: float | None = None,
        transcript_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.sample_rate = sample_rate
        self.epoch = time.monotonic() if epoch is None else float(epoch)
        self._tracks: dict[Track, bytearray] = {
            "caller": bytearray(),
            "agent": bytearray(),
        }
        self._active: dict[Track, float | None] = {"caller": None, "agent": None}
        self._intervals: dict[Track, list[SpeechInterval]] = {
            "caller": [],
            "agent": [],
        }
        self._texts: list[TranscriptItem] = []
        self._transcript_metadata = dict(transcript_metadata or {})
        if {"turns", "summary", "semantic_response_audio_policy"} & set(
            self._transcript_metadata
        ):
            raise ValueError("transcript metadata cannot replace PR #79 evidence fields")
        self._finalized = False

    def append_pcm(
        self,
        track: Track,
        pcm16: bytes,
        *,
        received_at: float | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append after evidence finalization")
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("PCM must be non-empty signed 16-bit mono")
        received = time.monotonic() if received_at is None else float(received_at)
        samples = len(pcm16) // 2
        end_sample = max(0, round((received - self.epoch) * self.sample_rate))
        start_sample = max(0, end_sample - samples)
        start_byte = start_sample * 2
        end_byte = start_byte + len(pcm16)
        target = self._tracks[track]
        if len(target) < end_byte:
            target.extend(b"\x00" * (end_byte - len(target)))
        target[start_byte:end_byte] = pcm16

    def mark_state(
        self,
        track: Track,
        old_state: str,
        new_state: str,
        *,
        at: float | None = None,
    ) -> None:
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        when = time.monotonic() if at is None else float(at)
        if new_state == "speaking" and self._active[track] is None:
            self._active[track] = when
        if old_state == "speaking" and new_state != "speaking":
            started = self._active[track]
            if started is not None and when > started:
                self._intervals[track].append(SpeechInterval(started, when))
            self._active[track] = None

    def add_text(
        self, role: Track, text: str, *, at: float | None = None
    ) -> None:
        normalized = " ".join(str(text or "").split())
        if normalized:
            self._texts.append(
                TranscriptItem(
                    role=role,
                    text=normalized,
                    at=time.monotonic() if at is None else float(at),
                )
            )

    def finalize(self, *, ended_at: float | None = None) -> dict[str, object]:
        if self._finalized:
            return self._summary()
        self._finalized = True
        ended = time.monotonic() if ended_at is None else float(ended_at)
        for track, started in self._active.items():
            if started is not None and ended > started:
                self._intervals[track].append(SpeechInterval(started, ended))
                self._active[track] = None
        self.root.mkdir(parents=True, exist_ok=True)
        maximum = max(len(value) for value in self._tracks.values())
        if maximum == 0:
            raise RuntimeError("no room audio was captured")
        for value in self._tracks.values():
            value.extend(b"\x00" * (maximum - len(value)))
        caller = bytes(self._tracks["caller"])
        agent = bytes(self._tracks["agent"])
        self._write_wav(self.root / "caller.wav", caller)
        self._write_wav(self.root / "agent.wav", agent)
        self._write_wav(self.root / "mixed.wav", self._mix(caller, agent))

        user_items = [item for item in self._texts if item.role == "caller"]
        agent_items = [item for item in self._texts if item.role == "agent"]
        turns: list[dict[str, object]] = []
        used_agent_intervals: set[int] = set()
        for index, (user_item, caller_interval) in enumerate(
            zip(user_items, self._intervals["caller"], strict=False), start=1
        ):
            response_entry = next(
                (
                    (position, interval)
                    for position, interval in enumerate(self._intervals["agent"])
                    if position not in used_agent_intervals
                    and interval.start >= caller_interval.end
                ),
                None,
            )
            if response_entry is None or index > len(agent_items):
                continue
            response_position, response_interval = response_entry
            used_agent_intervals.add(response_position)
            agent_item = agent_items[index - 1]
            caller_name = f"turn-{index:03d}-caller.wav"
            agent_name = f"turn-{index:03d}-agent.wav"
            self._write_wav(
                self.root / caller_name,
                self._slice(caller, caller_interval),
            )
            self._write_wav(
                self.root / agent_name,
                self._slice(agent, response_interval),
            )
            interrupted = any(
                interval.start < caller_interval.start < interval.end
                for interval in self._intervals["agent"]
            )
            turns.append(
                {
                    "index": index,
                    "user": user_item.text,
                    "assistant": agent_item.text,
                    "assistant_spoken": agent_item.text,
                    "semantic": True,
                    "interrupted": interrupted,
                    "start_ms": self._relative_ms(caller_interval.start),
                    "end_ms": self._relative_ms(response_interval.end),
                    "caller_audio": caller_name,
                    "agent_audio": agent_name,
                }
            )
        if not turns:
            raise RuntimeError("no semantic caller/agent turn could be linked")
        (self.root / "transcript.json").write_text(
            json.dumps(
                {
                    **self._transcript_metadata,
                    "measurement_source": "livekit_room_sidecar_synchronized_tracks",
                    "semantic_response_audio_policy": semantic_response_audio_policy(),
                    "summary": {"turn_count": len(turns)},
                    "turns": turns,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for path in self.root.iterdir():
            if path.is_file():
                path.chmod(0o600)
        return self._summary()

    def _slice(self, payload: bytes, interval: SpeechInterval) -> bytes:
        start = max(0, round((interval.start - self.epoch) * self.sample_rate)) * 2
        end = max(start + 2, round((interval.end - self.epoch) * self.sample_rate) * 2)
        return payload[start:end]

    def _relative_ms(self, value: float) -> int:
        return max(0, round((value - self.epoch) * 1_000))

    def _write_wav(self, path: Path, payload: bytes) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(payload)

    @staticmethod
    def _mix(caller: bytes, agent: bytes) -> bytes:
        left = struct.unpack(f"<{len(caller) // 2}h", caller)
        right = struct.unpack(f"<{len(agent) // 2}h", agent)
        mixed = (
            max(-32768, min(32767, caller_sample + agent_sample))
            for caller_sample, agent_sample in zip(left, right, strict=True)
        )
        return struct.pack(f"<{len(left)}h", *mixed)

    def _summary(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "sample_rate": self.sample_rate,
            "semantic_turns": len(list(self.root.glob("turn-*-caller.wav"))),
            "tracks_synchronized": True,
        }

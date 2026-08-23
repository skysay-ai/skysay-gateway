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

# How far past a response interval's end a terminal message may be stamped
# and still count as that interval's speech.  ``add_terminal_message`` fires
# right after the goodbye's completed playout while the interval end comes
# from the agent state-change event, so the two observations of the same
# playout end can land in either order; a real production call showed 3ms.
# Goodbyes spoken as separate, later speech land 8-60s past the final turn's
# interval, far outside this window.
_TERMINAL_ATTRIBUTION_SLACK_SECONDS = 2.0


@dataclass(frozen=True)
class SpeechInterval:
    start: float
    end: float


@dataclass(frozen=True)
class TranscriptItem:
    role: Track
    text: str
    at: float


class NoRoomAudioCaptured(RuntimeError):
    """The room ended before either synchronized track received audio."""


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
        # A LiveKit AudioStream can deliver several already-buffered frames in
        # one event-loop turn. Callback time anchors a subscription, but it is
        # not a media timestamp for every frame drained from that subscription.
        self._stream_end_samples: dict[tuple[Track, object], int] = {}
        self._active: dict[Track, float | None] = {"caller": None, "agent": None}
        self._intervals: dict[Track, list[SpeechInterval]] = {
            "caller": [],
            "agent": [],
        }
        self._texts: list[TranscriptItem] = []
        self._terminal_messages: list[TranscriptItem] = []
        self._termination_started_at: float | None = None
        self._turns: list[dict[str, object]] = []
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

    def append_stream_pcm(
        self,
        track: Track,
        pcm16: bytes,
        *,
        stream_id: object,
        received_at: float | None = None,
    ) -> None:
        """Append ordered PCM without turning scheduler jitter into bad audio.

        Observation time places each frame on the shared timeline while the
        stream cursor prevents already-buffered frames from overwriting one
        another. If a participant stops emitting frames, observation time can
        still advance the cursor and preserve that real silence. A republished
        track receives a new ``stream_id`` and is anchored again.
        """

        if self._finalized:
            raise RuntimeError("cannot append after evidence finalization")
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("PCM must be non-empty signed 16-bit mono")
        try:
            key = (track, stream_id)
            previous_end = self._stream_end_samples.get(key)
        except TypeError as exc:
            raise ValueError("stream_id must be hashable") from exc
        samples = len(pcm16) // 2
        received = time.monotonic() if received_at is None else float(received_at)
        observed_end = max(0, round((received - self.epoch) * self.sample_rate))
        observed_start = max(0, observed_end - samples)
        start_sample = (
            observed_start
            if previous_end is None
            else max(previous_end, observed_start)
        )
        end_sample = start_sample + samples
        start_byte = start_sample * 2
        end_byte = start_byte + len(pcm16)
        target = self._tracks[track]
        if len(target) < end_byte:
            target.extend(b"\x00" * (end_byte - len(target)))
        target[start_byte:end_byte] = pcm16
        self._stream_end_samples[key] = end_sample

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

    def mark_termination(self, *, at: float | None = None) -> None:
        """Record when worker-owned termination began (the end-call action).

        Terminal-message attribution requires the paired agent interval to
        START at or after this boundary: the goodbye is scheduled only after
        the termination action runs, so an interval beginning earlier also
        contains model reply speech whose text never committed — replacing
        that with just the goodbye would certify a truncated turn as
        complete.  The first mark wins; the action itself is idempotent.
        """

        when = time.monotonic() if at is None else float(at)
        if self._termination_started_at is None or when < self._termination_started_at:
            self._termination_started_at = when

    def add_terminal_message(self, text: str, *, at: float | None = None) -> None:
        """Persist one completed worker-owned goodbye outside semantic turns."""

        normalized = " ".join(str(text or "").split())
        if normalized:
            self._terminal_messages.append(
                TranscriptItem(
                    role="agent",
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
            self._finalized = False
            raise NoRoomAudioCaptured("no room audio was captured")
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
        attributed_terminal_turns: dict[int, int] = {}
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
            if response_entry is None:
                continue
            response_position, response_interval = response_entry
            used_agent_intervals.add(response_position)
            # LiveKit Agents 1.6.6 commits the assistant ChatMessage only at
            # the END of the reply task, so a caller who hangs up mid-reply
            # leaves the spoken response with no text.  The audio of that
            # response exists and is still the semantic answer -- the
            # interval filter above only accepts agent speech starting at or
            # after the caller stopped, so a greeting (which precedes the
            # caller's turn) can never be paired here, and
            # ``semantic_response_audio_policy`` pins
            # ``filler_acknowledgements_enabled: False``.  Record the turn with
            # its clips and mark it DEGRADED; never present it as complete.
            agent_item = agent_items[index - 1] if index <= len(agent_items) else None
            # The worker's own goodbye is the second way a spoken reply loses
            # its committed text: when the model answers the caller's final
            # utterance by requesting termination, turn admission closes
            # before the reply task commits, and the goodbye reaches this
            # recorder only through ``add_terminal_message`` after completed
            # playout.  That text exists and IS the turn's response, so
            # attribute it instead of degrading a healthy call.  Attribution
            # is deliberately narrow: only the call's FINAL agent speech
            # interval (the terminal goodbye is by construction the last
            # thing the agent says), only when that interval STARTED at or
            # after the recorded termination boundary (so it is wholly
            # worker-owned goodbye speech, not a merged interval that also
            # contains uncommitted model reply audio), only when no OTHER
            # caller utterance began between this turn's utterance and the
            # goodbye (the goodbye answers the caller's last words, and an
            # intervening utterance means this pairing is not that turn),
            # and only a message stamped within the interval's own playout
            # window.  A reply whose text really was lost can therefore
            # never borrow the goodbye.
            terminal_item = None
            if (
                agent_item is None
                and self._termination_started_at is not None
                and response_interval.start >= self._termination_started_at
                and response_position == len(self._intervals["agent"]) - 1
                and not any(
                    caller_interval.start < other.start < response_interval.start
                    for other in self._intervals["caller"]
                )
            ):
                terminal_entry = next(
                    (
                        (position, message)
                        for position, message in enumerate(self._terminal_messages)
                        if position not in attributed_terminal_turns
                        and response_interval.start
                        <= message.at
                        <= response_interval.end + _TERMINAL_ATTRIBUTION_SLACK_SECONDS
                    ),
                    None,
                )
                if terminal_entry is not None:
                    attributed_terminal_turns[terminal_entry[0]] = index
                    terminal_item = terminal_entry[1]
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
            recorded_item = agent_item or terminal_item
            turns.append(
                {
                    "index": index,
                    "user": user_item.text,
                    "assistant": recorded_item.text if recorded_item else "",
                    "assistant_spoken": recorded_item.text if recorded_item else "",
                    "assistant_text_status": (
                        "recorded"
                        if agent_item
                        else "recorded_terminal_message"
                        if terminal_item
                        else "unavailable_reply_truncated"
                    ),
                    "degraded": recorded_item is None,
                    "semantic": True,
                    "interrupted": interrupted,
                    "start_ms": self._relative_ms(caller_interval.start),
                    "end_ms": self._relative_ms(response_interval.end),
                    "caller_audio": caller_name,
                    "agent_audio": agent_name,
                }
            )
        self._turns = turns
        (self.root / "transcript.json").write_text(
            json.dumps(
                {
                    **self._transcript_metadata,
                    "measurement_source": "livekit_room_sidecar_synchronized_tracks",
                    "semantic_response_audio_policy": semantic_response_audio_policy(),
                    "summary": {
                        "turn_count": len(turns),
                        "degraded_turn_count": sum(
                            1 for turn in turns if turn["degraded"]
                        ),
                        "terminal_message_count": len(self._terminal_messages),
                        # Always present, even when 0: declares this artifact
                        # attribution-aware, so a consumer can tell a turn the
                        # worker deliberately left degraded from one an older
                        # recorder merely failed to attribute.
                        "attributed_terminal_message_count": len(
                            attributed_terminal_turns
                        ),
                    },
                    "terminal_messages": [
                        {
                            "role": message.role,
                            "text": message.text,
                            "at_ms": self._relative_ms(message.at),
                            # Mark a message that already appears as its
                            # turn's assistant text so consumers do not
                            # duplicate it; unattributed entries keep their
                            # original shape.
                            **(
                                {
                                    "attributed_turn_index": attributed_terminal_turns[
                                        position
                                    ]
                                }
                                if position in attributed_terminal_turns
                                else {}
                            ),
                        }
                        for position, message in enumerate(self._terminal_messages)
                    ],
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
        # Derived from the recorded turns rather than the clip files on disk:
        # a degraded turn also writes turn-*-caller.wav/turn-*-agent.wav, so
        # counting clips would report a truncated reply as complete evidence.
        semantic_turns = len(self._turns)
        degraded_turns = sum(1 for turn in self._turns if turn["degraded"])
        caller_audio_observed = any(self._tracks["caller"])
        caller_speech_intervals = len(self._intervals["caller"])
        caller_transcript_items = sum(
            1 for item in self._texts if item.role == "caller"
        )
        return {
            "root": str(self.root),
            "sample_rate": self.sample_rate,
            "semantic_turns": semantic_turns,
            "degraded_semantic_turns": degraded_turns,
            "semantic_evidence_status": (
                "unavailable_no_linked_turn"
                if not semantic_turns
                else "degraded_missing_agent_text"
                if degraded_turns
                else "complete"
            ),
            "caller_audio_observed": caller_audio_observed,
            "caller_speech_intervals": caller_speech_intervals,
            "caller_transcript_items": caller_transcript_items,
            "short_call_without_caller_speech": not (
                caller_speech_intervals or caller_transcript_items
            ),
            "tracks_synchronized": True,
        }

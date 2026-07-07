from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from typing import Iterable

from .frames import AudioFrame


class StreamingCallRecorder:
    """Writes caller and agent tracks while the call is live."""

    def __init__(self, root: Path, *, sample_rate: int = 8000):
        self.root = root
        self.sample_rate = sample_rate
        self.root.mkdir(parents=True, exist_ok=True)
        self._caller = self._open("caller.wav")
        self._agent = self._open("agent.wav")
        self._mixed = self._open("mixed.wav")
        self._closed = False
        self._waveforms_written = False

    @property
    def caller_path(self) -> Path:
        return self.root / "caller.wav"

    @property
    def agent_path(self) -> Path:
        return self.root / "agent.wav"

    @property
    def mixed_path(self) -> Path:
        return self.root / "mixed.wav"

    def caller_turn_path(self, turn_index: int) -> Path:
        return self.root / f"turn-{turn_index:03d}-caller.wav"

    def agent_turn_path(self, turn_index: int) -> Path:
        return self.root / f"turn-{turn_index:03d}-agent.wav"

    def _open(self, name: str) -> wave.Wave_write:
        handle = wave.open(str(self.root / name), "wb")
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(self.sample_rate)
        return handle

    def write_caller(self, frame: AudioFrame) -> None:
        self._caller.writeframes(frame.pcm16)
        self._mixed.writeframes(frame.pcm16)

    def write_agent(self, frame: AudioFrame) -> None:
        self._agent.writeframes(frame.pcm16)
        self._mixed.writeframes(frame.pcm16)

    def write_caller_turn(self, turn_index: int, frames: Iterable[AudioFrame]) -> Path:
        return self._write_turn_clip(self.caller_turn_path(turn_index), frames)

    def write_agent_turn(self, turn_index: int, frames: Iterable[AudioFrame]) -> Path:
        return self._write_turn_clip(self.agent_turn_path(turn_index), frames)

    def _write_turn_clip(self, path: Path, frames: Iterable[AudioFrame]) -> Path:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            for frame in frames:
                handle.writeframes(frame.pcm16)
        return path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in (self._caller, self._agent, self._mixed):
            handle.close()
        self.write_waveform_json()

    def __enter__(self) -> "StreamingCallRecorder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def write_waveform_json(self, *, buckets: int = 120) -> dict[str, object]:
        if self._waveforms_written:
            return read_waveform_manifest(self.root)
        self._waveforms_written = True
        tracks = {
            "caller": waveform_summary(self.caller_path, buckets=buckets),
            "agent": waveform_summary(self.agent_path, buckets=buckets),
            "mixed": waveform_summary(self.mixed_path, buckets=buckets),
        }
        primary = tracks["mixed"]
        manifest = {
            "version": 1,
            "generated_by": "openphonex_realtime_recorder",
            "primary": "mixed",
            "tracks": tracks,
        }
        (self.root / "waveform.json").write_text(json.dumps(primary, indent=2, sort_keys=True) + "\n")
        (self.root / "waveforms.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def read_waveform_manifest(root: Path) -> dict[str, object]:
    path = root / "waveforms.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def waveform_summary(path: Path, *, buckets: int = 120) -> dict[str, object]:
    info = _wav_info(path)
    peaks = waveform_peaks(path, buckets=buckets)
    return {
        "track": path.stem.replace(".waveform", ""),
        "source": path.name,
        "sample_rate": info.get("sample_rate", 0),
        "channels": info.get("channels", 0),
        "sample_width_bytes": info.get("sample_width_bytes", 0),
        "frames": info.get("frames", 0),
        "duration_seconds": info.get("duration_seconds", 0.0),
        "bucket_count": len(peaks),
        "peaks": peaks,
    }


def waveform_peaks(path: Path, *, buckets: int = 120) -> list[float]:
    samples = _wav_pcm16_samples(path)
    if not samples:
        return []
    bucket_size = max(1, len(samples) // max(1, buckets))
    peaks: list[float] = []
    for index in range(0, len(samples), bucket_size):
        chunk = samples[index : index + bucket_size]
        if chunk:
            peaks.append(round(min(1.0, max(abs(sample) for sample in chunk) / 32768.0), 4))
        if len(peaks) >= buckets:
            break
    return peaks


def _wav_info(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frames = handle.getnframes()
            return {
                "sample_rate": frame_rate,
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "frames": frames,
                "duration_seconds": round(frames / frame_rate, 3) if frame_rate else 0.0,
            }
    except Exception:
        return {}


def _wav_pcm16_samples(path: Path) -> list[int]:
    if not path.is_file():
        return []
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                return []
            payload = handle.readframes(handle.getnframes())
    except Exception:
        return []
    usable = payload[:-1] if len(payload) % 2 else payload
    if not usable:
        return []
    return list(struct.unpack("<" + "h" * (len(usable) // 2), usable))

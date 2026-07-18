# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioFrame:
    """A small PCM frame moving through the realtime pipeline."""

    pcm16: bytes
    sample_rate: int = 8000
    channels: int = 1
    timestamp_ms: int | None = None
    direction: str = "caller"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return len(self.pcm16) // 2

    @property
    def duration_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return round(self.samples / self.sample_rate * 1000)

    @property
    def rms(self) -> float:
        if len(self.pcm16) < 2:
            return 0.0
        usable = self.pcm16[:-1] if len(self.pcm16) % 2 else self.pcm16
        values = struct.unpack("<" + "h" * (len(usable) // 2), usable)
        if not values:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in values) / len(values))

    @classmethod
    def silence(cls, duration_ms: int = 20, sample_rate: int = 8000, *, direction: str = "caller") -> "AudioFrame":
        samples = max(0, int(sample_rate * duration_ms / 1000))
        return cls(b"\0\0" * samples, sample_rate=sample_rate, direction=direction)

    @classmethod
    def tone(
        cls,
        duration_ms: int = 20,
        sample_rate: int = 8000,
        *,
        hz: float = 440.0,
        amplitude: int = 9000,
        direction: str = "caller",
    ) -> "AudioFrame":
        samples = max(0, int(sample_rate * duration_ms / 1000))
        payload = bytearray()
        for idx in range(samples):
            value = int(amplitude * math.sin(2 * math.pi * hz * idx / sample_rate))
            payload.extend(struct.pack("<h", value))
        return cls(bytes(payload), sample_rate=sample_rate, direction=direction)


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    language: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    text: str
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

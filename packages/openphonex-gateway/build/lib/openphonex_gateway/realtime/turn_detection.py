from __future__ import annotations

from dataclasses import dataclass, field

from .frames import AudioFrame


@dataclass(frozen=True)
class TurnDetectionConfig:
    speech_rms: float = 650.0
    silence_rms: float = 220.0
    silence_hold_ms: int = 350
    min_speech_ms: int = 250
    preroll_ms: int = 250


@dataclass(frozen=True)
class DetectedTurn:
    frames: list[AudioFrame]
    start_ms: int
    end_ms: int
    speech_ms: int
    speech_end_ms: int | None = None

    @property
    def pcm16(self) -> bytes:
        return b"".join(frame.pcm16 for frame in self.frames)


class RmsTurnDetector:
    """Streaming RMS endpointing for the first production media worker slice."""

    def __init__(self, config: TurnDetectionConfig | None = None):
        self.config = config or TurnDetectionConfig()
        self.reset()

    def reset(self) -> None:
        self._clock_ms = 0
        self._speech_started = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._start_ms = 0
        self._frames: list[AudioFrame] = []
        self._preroll: list[AudioFrame] = []
        self._preroll_ms = 0

    def feed(self, frame: AudioFrame) -> DetectedTurn | None:
        frame_ms = max(1, frame.duration_ms)
        frame_start_ms = self._clock_ms
        self._clock_ms += frame_ms

        if not self._speech_started:
            self._preroll.append(frame)
            self._preroll_ms += frame_ms
            while self._preroll and self._preroll_ms > self.config.preroll_ms:
                removed = self._preroll.pop(0)
                self._preroll_ms -= max(1, removed.duration_ms)

        if frame.rms >= self.config.speech_rms:
            if not self._speech_started:
                self._speech_started = True
                self._start_ms = max(0, frame_start_ms - self._preroll_ms)
                self._frames = list(self._preroll)
            self._frames.append(frame)
            self._speech_ms += frame_ms
            self._silence_ms = 0
            return None

        if not self._speech_started:
            return None

        self._frames.append(frame)
        if frame.rms <= self.config.silence_rms:
            self._silence_ms += frame_ms
        else:
            self._silence_ms = 0

        if self._speech_ms >= self.config.min_speech_ms and self._silence_ms >= self.config.silence_hold_ms:
            turn = DetectedTurn(
                frames=list(self._frames),
                start_ms=self._start_ms,
                end_ms=self._clock_ms,
                speech_ms=self._speech_ms,
                speech_end_ms=max(0, self._clock_ms - self._silence_ms),
            )
            self.reset()
            return turn
        return None

    def force_endpoint(self) -> DetectedTurn | None:
        """Return the active turn immediately when another endpointing signal fires."""

        if not self._speech_started or self._speech_ms < self.config.min_speech_ms:
            return None
        turn = DetectedTurn(
            frames=list(self._frames),
            start_ms=self._start_ms,
            end_ms=self._clock_ms,
            speech_ms=self._speech_ms,
            speech_end_ms=max(0, self._clock_ms - self._silence_ms),
        )
        self.reset()
        return turn

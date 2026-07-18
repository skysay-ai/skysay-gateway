# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

from typing import Iterable, Protocol

from .frames import AudioFrame, TranscriptEvent
from .turn_detection import DetectedTurn


class StreamingSTTSession(Protocol):
    """A single live STT stream for one caller turn."""

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        ...

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        ...

    def close(self) -> None:
        ...


class StreamingSTT(Protocol):
    """STT adapter contract.

    New adapters should implement start_stream(). The transcribe() method remains
    as the compatibility path for existing turn-buffered adapters and tests.
    """

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        ...


class SessionCapableStreamingSTT(StreamingSTT, Protocol):
    def start_stream(self) -> StreamingSTTSession:
        ...


class TurnBufferedSTTSession:
    """Compatibility wrapper for older turn-buffered STT adapters."""

    def __init__(self, stt: StreamingSTT):
        self._stt = stt
        self._frames: list[AudioFrame] = []
        self._closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if not self._closed:
            self._frames.append(frame)
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if self._closed:
            return ()
        if turn is None:
            turn = self._build_turn()
        if turn is None:
            return ()
        return self._stt.transcribe(turn)

    def close(self) -> None:
        self._closed = True
        self._frames.clear()

    def _build_turn(self) -> DetectedTurn | None:
        if not self._frames:
            return None
        duration_ms = sum(max(1, frame.duration_ms) for frame in self._frames)
        return DetectedTurn(frames=list(self._frames), start_ms=0, end_ms=duration_ms, speech_ms=duration_ms)


def start_stt_stream(stt: StreamingSTT) -> StreamingSTTSession:
    start_stream = getattr(stt, "start_stream", None)
    if callable(start_stream):
        return start_stream()
    return TurnBufferedSTTSession(stt)

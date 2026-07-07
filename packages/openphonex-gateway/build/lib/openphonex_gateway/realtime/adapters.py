from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterable

from .frames import AudioFrame, TextChunk, TranscriptEvent
from .pipeline import StreamingLLM
from .stt import StreamingSTT, StreamingSTTSession
from .tts import StreamingTTS
from .turn_detection import DetectedTurn


@dataclass(frozen=True)
class FakeRealtimeProviderConfig:
    transcript: str = "Tere, soovin aega broneerida."
    reply: str = "Tere! Mis auto mudeliga tulete?"
    language: str = "et"
    tts_hz: float = 440.0


class FakeStreamingSTT(StreamingSTT):
    """Deterministic STT adapter for safe worker tests and dry-runs."""

    def __init__(self, config: FakeRealtimeProviderConfig | None = None):
        self.config = config or FakeRealtimeProviderConfig()

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        words = self.config.transcript.split()
        if words:
            yield TranscriptEvent(words[0], is_final=False, language=self.config.language, metadata={"provider": "fake"})
        yield TranscriptEvent(self.config.transcript, is_final=True, language=self.config.language, metadata={"provider": "fake", "audio_ms": turn.end_ms - turn.start_ms})

    def start_stream(self) -> StreamingSTTSession:
        return _FakeStreamingSTTSession(self.config)


class _FakeStreamingSTTSession:
    def __init__(self, config: FakeRealtimeProviderConfig):
        self.config = config
        self.frames: list[AudioFrame] = []
        self._sent_interim = False
        self._saw_speech = False
        self._closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if self._closed:
            return ()
        self.frames.append(frame)
        if frame.rms >= 650.0:
            self._saw_speech = True
            words = self.config.transcript.split()
            if words and not self._sent_interim:
                self._sent_interim = True
                return (TranscriptEvent(words[0], is_final=False, language=self.config.language, metadata={"provider": "fake", "streaming": True}),)
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if self._closed or (turn is None and not self._saw_speech):
            return ()
        audio_ms = turn.end_ms - turn.start_ms if turn else sum(max(1, frame.duration_ms) for frame in self.frames)
        return (TranscriptEvent(self.config.transcript, is_final=True, language=self.config.language, metadata={"provider": "fake", "streaming": True, "audio_ms": audio_ms}),)

    def close(self) -> None:
        self._closed = True
        self.frames.clear()


class TemplateStreamingLLM(StreamingLLM):
    """Streams a configured answer in phrase-sized chunks."""

    def __init__(self, config: FakeRealtimeProviderConfig | None = None):
        self.config = config or FakeRealtimeProviderConfig()

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        parts = _split_for_speech(self.config.reply)
        for idx, part in enumerate(parts):
            yield TextChunk(part, is_final=idx == len(parts) - 1, metadata={"provider": "fake"})


class ToneStreamingTTS(StreamingTTS):
    """Small streaming TTS replacement that emits PCM frames immediately."""

    def __init__(self, config: FakeRealtimeProviderConfig | None = None, *, frame_ms: int = 20, frame_count: int = 8):
        self.config = config or FakeRealtimeProviderConfig()
        self.frame_ms = frame_ms
        self.frame_count = frame_count

    def stream(self, text: str) -> Iterable[AudioFrame]:
        for idx in range(self.frame_count):
            yield _tone_frame(self.frame_ms, hz=self.config.tts_hz + idx * 25)


def _split_for_speech(text: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in ".!?,":
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks or [text]


def _tone_frame(duration_ms: int, *, sample_rate: int = 8000, hz: float = 440.0, amplitude: int = 9000) -> AudioFrame:
    samples = max(1, int(sample_rate * duration_ms / 1000))
    payload = bytearray()
    for idx in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * hz * idx / sample_rate))
        payload.extend(struct.pack("<h", value))
    return AudioFrame(bytes(payload), sample_rate=sample_rate, direction="agent", metadata={"provider": "fake_tts"})

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import importlib.util
import json
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .frames import AudioFrame, TranscriptEvent
from .stt import StreamingSTT, StreamingSTTSession
from .turn_detection import DetectedTurn


class WebSocketLike(Protocol):
    def send_binary(self, payload: bytes) -> object:
        ...

    def send(self, payload: str) -> object:
        ...

    def recv(self) -> str | bytes:
        ...

    def settimeout(self, timeout: float) -> object:
        ...

    def close(self) -> object:
        ...


WebSocketFactory = Callable[[str, list[str], float], WebSocketLike]

# Deepgram rejects the live WebSocket handshake with HTTP 400 Bad Request when
# ``utterance_end_ms`` is below 1000 (its documented minimum). An operator env
# override such as ``DEEPGRAM_UTTERANCE_END_MS=800`` therefore silently kills STT
# for every call. Clamp to this floor when building the URL so an out-of-range
# value degrades to the slowest valid endpoint fallback instead of a dead stream.
DEEPGRAM_MIN_UTTERANCE_END_MS = 1000


def missing_deepgram_streaming_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("websocket") is None:
        missing.append("websocket-client")
    return missing


@dataclass(frozen=True)
class DeepgramStreamingSTTConfig:
    api_key: str
    model_id: str = "nova-3"
    language: str = "et"
    encoding: str = "linear16"
    sample_rate: int = 8000
    channels: int = 1
    interim_results: bool = True
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000
    vad_events: bool = True
    punctuate: bool = True
    smart_format: bool = True
    timeout_seconds: float = 3.0
    # SCH-210: 0.001s drained a single ~1ms read before returning, so Deepgram's
    # is_final/speech_final often arrived after the drain gave up (stalled turns,
    # dropped finals). 0.005s + 3 empty reads (see _drain_events) reliably catch
    # the terminal transcript without adding perceptible latency.
    drain_timeout_seconds: float = 0.005
    finalize_timeout_seconds: float = 2.0
    url_base: str = "wss://api.deepgram.com/v1/listen"


class DeepgramStreamingSTT(StreamingSTT):
    """Deepgram live WebSocket STT adapter.

    The adapter keeps the OpenPhonex pipeline provider-neutral: it only emits
    TranscriptEvent objects and does not know about DIDWW, Asterisk, billing, or
    call routing.
    """

    def __init__(self, config: DeepgramStreamingSTTConfig, *, websocket_factory: WebSocketFactory | None = None):
        if not config.api_key:
            raise ValueError("Deepgram API key is required")
        if config.sample_rate <= 0:
            raise ValueError("Deepgram sample_rate must be positive")
        self.config = config
        self.websocket_factory = websocket_factory or _default_websocket_factory

    def start_stream(self) -> StreamingSTTSession:
        socket = self.websocket_factory(
            self.connection_url(),
            [f"Authorization: Token {self.config.api_key}"],
            self.config.timeout_seconds,
        )
        socket.settimeout(self.config.drain_timeout_seconds)
        return DeepgramStreamingSTTSession(socket, self.config)

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        session = self.start_stream()
        try:
            for frame in turn.frames:
                yield from session.send_audio(frame)
            yield from session.end_input(turn)
        finally:
            session.close()

    def connection_url(self) -> str:
        query: dict[str, str] = {
            "model": self.config.model_id,
            "encoding": self.config.encoding,
            "sample_rate": str(self.config.sample_rate),
            "channels": str(self.config.channels),
            "interim_results": _bool_query(self.config.interim_results),
            "endpointing": str(self.config.endpointing_ms),
            "utterance_end_ms": str(max(DEEPGRAM_MIN_UTTERANCE_END_MS, int(self.config.utterance_end_ms))),
            "vad_events": _bool_query(self.config.vad_events),
            "punctuate": _bool_query(self.config.punctuate),
            "smart_format": _bool_query(self.config.smart_format),
        }
        if self.config.language:
            query["language"] = self.config.language
        return f"{self.config.url_base}?{urllib.parse.urlencode(query)}"


class DeepgramStreamingSTTSession:
    skip_finalization_without_transcript = True

    def __init__(self, socket: WebSocketLike, config: DeepgramStreamingSTTConfig):
        self.socket = socket
        self.config = config
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if self.closed:
            return ()
        if frame.pcm16:
            # Socket sits at the tiny drain timeout for non-blocking reads, but a
            # binary write at ~1ms times out on a real network. Raise the timeout
            # for the write, then restore the drain timeout for the read below.
            self.socket.settimeout(self.config.timeout_seconds)
            try:
                self.socket.send_binary(frame.pcm16)
            finally:
                self.socket.settimeout(self.config.drain_timeout_seconds)
        return self._drain_events()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if self.closed:
            return ()
        # The Finalize control frame is a socket write: it must go out at the full
        # write timeout, not the ~1ms drain timeout left over from the last read.
        self.socket.settimeout(self.config.timeout_seconds)
        self.socket.send(json.dumps({"type": "Finalize"}))
        self.socket.settimeout(self.config.finalize_timeout_seconds)
        try:
            return self._drain_events(max_events=32, max_empty_reads=3)
        finally:
            self.socket.settimeout(self.config.drain_timeout_seconds)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.socket.send(json.dumps({"type": "CloseStream"}))
        except Exception:
            pass
        self.socket.close()

    def _drain_events(self, *, max_events: int = 16, max_empty_reads: int = 3) -> Iterable[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        empty_reads = 0
        while len(events) < max_events:
            try:
                raw = self.socket.recv()
            except Exception as exc:
                if _looks_like_timeout(exc):
                    empty_reads += 1
                    if empty_reads >= max_empty_reads:
                        break
                    continue
                raise
            empty_reads = 0
            events.extend(parse_deepgram_message(raw))
        return events


def parse_deepgram_message(raw: str | bytes) -> list[TranscriptEvent]:
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    data = json.loads(raw)
    if not isinstance(data, dict):
        return []
    if data.get("type") != "Results":
        return []
    channel = data.get("channel") or {}
    if not isinstance(channel, dict):
        return []
    alternatives = channel.get("alternatives") or []
    if not alternatives or not isinstance(alternatives[0], dict):
        return []
    alternative = alternatives[0]
    transcript = str(alternative.get("transcript") or "").strip()
    if not transcript:
        return []
    language = _event_language(alternative)
    metadata = {
        "provider": "deepgram",
        "speech_final": bool(data.get("speech_final")),
        "from_finalize": bool(data.get("from_finalize")),
        "request_id": ((data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}).get("request_id", ""),
        "start": data.get("start"),
        "duration": data.get("duration"),
    }
    return [
        TranscriptEvent(
            transcript,
            is_final=bool(data.get("is_final")) or bool(data.get("speech_final")),
            language=language,
            confidence=_confidence(alternative),
            metadata=metadata,
        )
    ]


def _default_websocket_factory(url: str, headers: list[str], timeout_seconds: float) -> WebSocketLike:
    import websocket

    return websocket.create_connection(url, header=headers, timeout=timeout_seconds)


def _bool_query(value: bool) -> str:
    return "true" if value else "false"


def _confidence(alternative: dict[str, object]) -> float | None:
    value = alternative.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_language(alternative: dict[str, object]) -> str | None:
    language = alternative.get("language")
    if language:
        return str(language)
    languages = alternative.get("languages")
    if isinstance(languages, list) and languages:
        return str(languages[0])
    words = alternative.get("words")
    if isinstance(words, list):
        for word in words:
            if isinstance(word, dict) and word.get("language"):
                return str(word["language"])
    return None


def _looks_like_timeout(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if "timeout" in name:
        return True
    return "timed out" in str(exc).lower()

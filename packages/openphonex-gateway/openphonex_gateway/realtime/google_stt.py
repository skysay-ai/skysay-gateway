# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Google Cloud Speech STT adapter (registry name ``google``).

Two paths behind one spec, chosen at construction:

* STREAMING (SessionCapable) when ``google-cloud-speech`` is importable (or a
  ``recognize_fn`` transport is injected for tests): ``start_stream()`` returns a
  session that drives ``StreamingRecognize`` (bidi gRPC) and yields interim +
  final ``TranscriptEvent`` objects. This is the ONE GA-5 provider that truly
  needs an SDK, so ``google.cloud.speech`` is imported LAZILY inside the
  transport — importing this module (to read ``PROVIDER_SPEC`` and flip catalog
  selectability) stays stdlib-only.
* REST FALLBACK (stdlib ``urllib``) when the SDK is absent but a
  ``GOOGLE_API_KEY`` is present: only ``transcribe(turn)`` is implemented (no
  ``start_stream``), so ``TurnBufferedSTTSession`` wraps it — buffered turn PCM is
  base64-posted to ``v1/speech:recognize`` with the API key.

8 kHz is sent directly (``sample_rate_hertz=8000``; telephony / latest_long
models accept it) — no resample on the default path. Only a 16 kHz-only model
(``GOOGLE_STT_SAMPLE_RATE=16000``) triggers ``resample.resample_pcm16`` (behind
the SDK extra where numpy is present).

Config (env):
  GOOGLE_API_KEY / GOOGLE_APPLICATION_CREDENTIALS   credential (either)
  GOOGLE_STT_MODEL              recognition model (default ``telephony``)
  GOOGLE_STT_LANGUAGE          BCP-47 code (default reuses REALTIME_LANGUAGE / ``et-EE``)
  GOOGLE_STT_SAMPLE_RATE       recognizer input rate (default 8000)
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable

from .frames import AudioFrame, TranscriptEvent
from .registry import ProviderSpec
from .stt import StreamingSTTSession
from .turn_detection import DetectedTurn

_REST_URL = "https://speech.googleapis.com/v1/speech:recognize"
_DEFAULT_MODEL = "telephony"
_EDGE_RATE = 8000

# Transport seam: (pcm16, params) -> iterable of StreamingRecognize responses.
RecognizeFn = Callable[[bytes, "dict[str, Any]"], Iterable[Any]]


def _model() -> str:
    return (os.environ.get("GOOGLE_STT_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()


def _language() -> str:
    return (
        os.environ.get("GOOGLE_STT_LANGUAGE")
        or os.environ.get("REALTIME_LANGUAGE")
        or "et-EE"
    ).strip()


def _sample_rate() -> int:
    try:
        return max(8000, int(os.environ.get("GOOGLE_STT_SAMPLE_RATE", str(_EDGE_RATE))))
    except (TypeError, ValueError):
        return _EDGE_RATE


def _has_speech_sdk() -> bool:
    # find_spec on a dotted path imports the parent packages, so it raises
    # ModuleNotFoundError (not returns None) when the top-level ``google``
    # namespace is entirely absent — treat any failure as "SDK not present".
    try:
        return importlib.util.find_spec("google.cloud.speech") is not None
    except (ImportError, ValueError):
        return False


class GoogleStreamingSTT:
    def __init__(self, config: object, *, recognize_fn: "RecognizeFn | None" = None, opener=None):
        self.config = config
        self._recognize_fn = recognize_fn
        self._opener = opener
        self._params: dict[str, Any] = {
            "language": _language(),
            "model": _model(),
            "sample_rate": _sample_rate(),
        }
        self._timeout = float(os.environ.get("GOOGLE_STT_TIMEOUT_SECONDS", "30.0"))
        # Streaming needs the SDK (or an injected transport). Without either, hide
        # start_stream so start_stt_stream() wraps transcribe() in the
        # TurnBufferedSTTSession REST fallback.
        if recognize_fn is None and not _has_speech_sdk():
            self.start_stream = None  # type: ignore[assignment]

    def start_stream(self) -> StreamingSTTSession:
        recognize_fn = self._recognize_fn or _default_recognize_fn()
        return GoogleStreamingSTTSession(recognize_fn, dict(self._params))

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        if getattr(self, "start_stream", None) is not None:
            session = self.start_stream()
            try:
                for frame in turn.frames:
                    yield from session.send_audio(frame)
                yield from session.end_input(turn)
            finally:
                session.close()
            return
        yield from self._transcribe_rest(turn)

    def _transcribe_rest(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        pcm = b"".join(frame.pcm16 for frame in turn.frames)
        if not pcm:
            return
        key = os.environ.get("GOOGLE_API_KEY", "")
        config: dict[str, Any] = {
            "encoding": "LINEAR16",
            "sampleRateHertz": _EDGE_RATE,
            "languageCode": self._params["language"],
        }
        if self._params["model"]:
            config["model"] = self._params["model"]
        body = json.dumps(
            {"config": config, "audio": {"content": base64.b64encode(pcm).decode("ascii")}}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{_REST_URL}?key={urllib.parse.quote(key)}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self._opener is not None:
            response = self._opener(request, timeout=self._timeout)
        else:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        with response:
            raw = response.read().decode("utf-8", errors="replace")
        obj = json.loads(raw or "{}")
        texts: list[str] = []
        confidence: float | None = None
        for result in (obj.get("results") or []) if isinstance(obj, dict) else []:
            alternatives = result.get("alternatives") or [] if isinstance(result, dict) else []
            if not alternatives:
                continue
            alt = alternatives[0]
            transcript = str((alt.get("transcript") if isinstance(alt, dict) else "") or "").strip()
            if transcript:
                texts.append(transcript)
                if confidence is None and isinstance(alt, dict):
                    confidence = _confidence(alt.get("confidence"))
        text = " ".join(texts).strip()
        if not text:
            return
        yield TranscriptEvent(
            text,
            is_final=True,
            language=self._params["language"],
            confidence=confidence,
            metadata={"provider": "google", "path": "rest", "audio_ms": max(0, turn.end_ms - turn.start_ms)},
        )


class GoogleStreamingSTTSession:
    def __init__(self, recognize_fn: RecognizeFn, params: "dict[str, Any]"):
        self._recognize_fn = recognize_fn
        self._params = params
        self._frames: list[bytes] = []
        self._closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if not self._closed and frame.pcm16:
            self._frames.append(frame.pcm16)
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if self._closed:
            return ()
        pcm = b"".join(self._frames)
        self._frames = []
        if not pcm:
            return ()
        target = int(self._params.get("sample_rate", _EDGE_RATE) or _EDGE_RATE)
        if target != _EDGE_RATE:
            from . import resample  # numpy only reached on a 16k-only model path

            pcm = resample.resample_pcm16(pcm, _EDGE_RATE, target)
        events: list[TranscriptEvent] = []
        for response in self._recognize_fn(pcm, self._params):
            events.extend(parse_google_stream_response(response, language=self._params.get("language")))
        return events

    def close(self) -> None:
        self._closed = True
        self._frames = []


def parse_google_stream_response(response: Any, *, language: "str | None" = None) -> "list[TranscriptEvent]":
    """Parse one StreamingRecognize response into TranscriptEvents.

    Accepts both real google protobuf objects (attribute access) and plain
    dict/namespace fakes used by the mock-transport tests.
    """

    results = _get(response, "results") or []
    events: list[TranscriptEvent] = []
    for result in results:
        alternatives = _get(result, "alternatives") or []
        if not alternatives:
            continue
        alt = alternatives[0]
        transcript = str(_get(alt, "transcript") or "").strip()
        if not transcript:
            continue
        is_final = bool(_get(result, "is_final"))
        events.append(
            TranscriptEvent(
                transcript,
                is_final=is_final,
                language=str(_get(result, "language_code") or language or "") or None,
                confidence=_confidence(_get(alt, "confidence")),
                metadata={"provider": "google", "path": "streaming"},
            )
        )
    return events


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _confidence(value: Any) -> "float | None":
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_recognize_fn() -> RecognizeFn:
    def _recognize(pcm: bytes, params: "dict[str, Any]") -> Iterable[Any]:
        from google.cloud import speech  # lazy heavy import (SDK extra only)

        client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=int(params.get("sample_rate", _EDGE_RATE)),
            language_code=params.get("language", "et-EE"),
            model=(params.get("model") or None),
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config, interim_results=True
        )

        def _requests() -> Iterable[Any]:
            step = 3200  # ~200 ms of 8 kHz PCM16 per request
            for start in range(0, len(pcm), step):
                yield speech.StreamingRecognizeRequest(audio_content=pcm[start : start + step])

        return client.streaming_recognize(config=streaming_config, requests=_requests())

    return _recognize


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not _has_speech_sdk():
        missing.append("google-cloud-speech")
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        missing.append("GOOGLE_API_KEY")
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "google",
        "adapter": "GoogleStreamingSTT",
        "real": True,
        "model_id": _model(),
        "language": _language(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="google",
    component="stt",
    factory=GoogleStreamingSTT,
    describe=_describe,
    availability=_availability,
    real=True,
)

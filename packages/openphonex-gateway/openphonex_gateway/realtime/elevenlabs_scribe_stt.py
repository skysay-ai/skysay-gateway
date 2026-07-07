# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""ElevenLabs Scribe batch STT adapter (registry name ``elevenlabs_scribe``).

Turn-buffered speech-to-text: buffer the caller turn's 8 kHz PCM16, wrap it in
a WAV container with stdlib ``wave`` (no resample, no ffmpeg) and POST it as a
multipart file to ``POST https://api.elevenlabs.io/v1/speech-to-text`` (model
``scribe_v1``). Implements only ``transcribe(turn)`` so ``TurnBufferedSTTSession``
wraps it automatically. stdlib ``urllib`` only — no SDK.

Config (env):
  ELEVENLABS_API_KEY           reused for live use (graceful-absent otherwise)
  ELEVENLABS_SCRIBE_MODEL      model id (default ``scribe_v1``)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .audio_encode import pcm16_to_wav
from .frames import TranscriptEvent
from .provider_http import encode_multipart
from .registry import ProviderSpec
from .turn_detection import DetectedTurn

_SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_DEFAULT_MODEL = "scribe_v1"
_EDGE_RATE = 8000


def _model() -> str:
    return (os.environ.get("ELEVENLABS_SCRIBE_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()


class ElevenLabsScribeSTT:
    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self._model = _model()
        self._language = (getattr(config, "elevenlabs_language_code", "") or "").strip()
        self._timeout = float(
            os.environ.get("ELEVENLABS_SCRIBE_TIMEOUT_SECONDS", str(getattr(config, "gemini_timeout_seconds", 30.0)))
        )
        self._opener = opener

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        pcm = b"".join(frame.pcm16 for frame in turn.frames)
        if not pcm:
            return
        wav = pcm16_to_wav(pcm, _EDGE_RATE)
        fields: dict[str, str] = {"model_id": self._model}
        if self._language:
            fields["language_code"] = self._language
        body, content_type = encode_multipart(
            fields, file_field="file", file_name="audio.wav", file_bytes=wav
        )
        request = urllib.request.Request(
            _SCRIBE_URL,
            data=body,
            headers={"xi-api-key": self._api_key, "Content-Type": content_type},
            method="POST",
        )
        if self._opener is not None:
            response = self._opener(request, timeout=self._timeout)
        else:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        with response:
            raw = response.read().decode("utf-8", errors="replace")
        obj = json.loads(raw or "{}")
        text = str((obj.get("text") if isinstance(obj, dict) else "") or "").strip()
        if not text:
            return
        language = None
        if isinstance(obj, dict):
            language = obj.get("language_code") or obj.get("language") or (self._language or None)
        yield TranscriptEvent(
            text,
            is_final=True,
            language=language,
            metadata={"provider": "elevenlabs_scribe", "model": self._model, "audio_ms": max(0, turn.end_ms - turn.start_ms)},
        )


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("ELEVENLABS_API_KEY"):
        missing.append("ELEVENLABS_API_KEY")
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "elevenlabs_scribe",
        "adapter": "ElevenLabsScribeSTT",
        "real": True,
        "model_id": _model(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="elevenlabs_scribe",
    component="stt",
    factory=ElevenLabsScribeSTT,
    describe=_describe,
    availability=_availability,
    real=True,
)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Deepgram Aura TTS adapter (registry name ``deepgram_tts``, alias ``aura``).

Streaming-ish synthesis to the 8 kHz pipeline edge: request raw 8 kHz linear16
PCM directly from ``POST https://api.deepgram.com/v1/speak`` with
``encoding=linear16&sample_rate=8000&container=none`` and chunk the response
incrementally into ~20 ms ``AudioFrame`` (direction ``agent``). Because Deepgram
emits 8 kHz PCM natively there is NO ffmpeg and NO resample (contrast the
ElevenLabs mp3+ffmpeg path). stdlib ``urllib`` only — no SDK.

Config (env):
  DEEPGRAM_API_KEY             reused for live use (graceful-absent otherwise)
  DEEPGRAM_TTS_MODEL           Aura voice/model id (default ``aura-2-thalia-en``)
  DEEPGRAM_TTS_READ_BYTES      HTTP read chunk size (default 4096)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from .audio_encode import chunk_pcm16_to_frames
from .frames import AudioFrame
from .registry import ProviderSpec

_SPEAK_URL = "https://api.deepgram.com/v1/speak"
_DEFAULT_MODEL = "aura-2-thalia-en"
_EDGE_RATE = 8000


def _model() -> str:
    return (os.environ.get("DEEPGRAM_TTS_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()


def _read_bytes() -> int:
    try:
        return max(2, int(os.environ.get("DEEPGRAM_TTS_READ_BYTES", "4096")))
    except (TypeError, ValueError):
        return 4096


class DeepgramAuraTTS:
    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        self._model = _model()
        self._timeout = float(os.environ.get("DEEPGRAM_TTS_TIMEOUT_SECONDS", "30.0"))
        self._read_bytes = _read_bytes()
        self._opener = opener

    def _url(self) -> str:
        query = {
            "model": self._model,
            "encoding": "linear16",
            "sample_rate": str(_EDGE_RATE),
            "container": "none",
        }
        return f"{_SPEAK_URL}?{urllib.parse.urlencode(query)}"

    def stream(self, text: str) -> Iterable[AudioFrame]:
        text = (text or "").strip()
        if not text:
            return
        request = urllib.request.Request(
            self._url(),
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        if self._opener is not None:
            response = self._opener(request, timeout=self._timeout)
        else:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        with response:
            yield from chunk_pcm16_to_frames(
                _read_stream(response, self._read_bytes),
                rate=_EDGE_RATE,
                metadata={"provider": "deepgram_aura", "model": self._model},
            )


def _read_stream(response, read_bytes: int) -> Iterable[bytes]:
    while True:
        chunk = response.read(read_bytes)
        if not chunk:
            break
        yield chunk


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "deepgram_aura",
        "adapter": "DeepgramAuraTTS",
        "real": True,
        "model_id": _model(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="deepgram_tts",
    component="tts",
    factory=DeepgramAuraTTS,
    describe=_describe,
    availability=_availability,
    real=True,
    aliases=("aura",),
)

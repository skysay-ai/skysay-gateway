# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Cartesia Sonic TTS adapter (registry name ``cartesia``).

Streaming synthesis to the 8 kHz pipeline edge: request raw 8 kHz
``pcm_s16le`` directly from ``POST https://api.cartesia.ai/tts/bytes`` with
``output_format={container:"raw", encoding:"pcm_s16le", sample_rate:8000}`` and
chunk the response incrementally into ~20 ms ``AudioFrame`` (direction
``agent``). Cartesia emits 8 kHz PCM natively, so there is NO ffmpeg and NO
resample. stdlib ``urllib`` only — no SDK.

Config (env):
  CARTESIA_API_KEY             required for live use (graceful-absent otherwise)
  CARTESIA_MODEL               model id (default ``sonic-2``)
  CARTESIA_VOICE_ID            voice id (required for a live request)
  CARTESIA_VERSION             ``Cartesia-Version`` header (default ``2024-11-13``)
  CARTESIA_LANGUAGE            language hint (default ``en``)
  CARTESIA_TTS_READ_BYTES      HTTP read chunk size (default 4096)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .audio_encode import chunk_pcm16_to_frames
from .frames import AudioFrame
from .registry import ProviderSpec

_TTS_URL = "https://api.cartesia.ai/tts/bytes"
_DEFAULT_MODEL = "sonic-2"
_DEFAULT_VERSION = "2024-11-13"
_EDGE_RATE = 8000


def _model() -> str:
    return (os.environ.get("CARTESIA_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()


def _read_bytes() -> int:
    try:
        return max(2, int(os.environ.get("CARTESIA_TTS_READ_BYTES", "4096")))
    except (TypeError, ValueError):
        return 4096


class CartesiaTTS:
    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._api_key = os.environ.get("CARTESIA_API_KEY", "")
        self._model = _model()
        self._voice_id = os.environ.get("CARTESIA_VOICE_ID", "")
        self._version = os.environ.get("CARTESIA_VERSION", _DEFAULT_VERSION)
        self._language = os.environ.get("CARTESIA_LANGUAGE", "en")
        self._timeout = float(os.environ.get("CARTESIA_TTS_TIMEOUT_SECONDS", "30.0"))
        self._read_bytes = _read_bytes()
        self._opener = opener

    def _payload(self, text: str) -> bytes:
        return json.dumps(
            {
                "model_id": self._model,
                "transcript": text,
                "voice": {"mode": "id", "id": self._voice_id},
                "language": self._language,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": _EDGE_RATE,
                },
            }
        ).encode("utf-8")

    def stream(self, text: str) -> Iterable[AudioFrame]:
        text = (text or "").strip()
        if not text:
            return
        request = urllib.request.Request(
            _TTS_URL,
            data=self._payload(text),
            headers={
                "X-API-Key": self._api_key,
                "Cartesia-Version": self._version,
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
                metadata={"provider": "cartesia", "model": self._model},
            )


def _read_stream(response, read_bytes: int) -> Iterable[bytes]:
    while True:
        chunk = response.read(read_bytes)
        if not chunk:
            break
        yield chunk


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("CARTESIA_API_KEY"):
        missing.append("CARTESIA_API_KEY")
    if not os.environ.get("CARTESIA_VOICE_ID"):
        missing.append("CARTESIA_VOICE_ID")
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "cartesia",
        "adapter": "CartesiaTTS",
        "real": True,
        "model_id": _model(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="cartesia",
    component="tts",
    factory=CartesiaTTS,
    describe=_describe,
    availability=_availability,
    real=True,
)

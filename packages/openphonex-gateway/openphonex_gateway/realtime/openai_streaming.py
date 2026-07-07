# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""OpenAI hosted STT + LLM adapters (registry names ``openai``, one per slot).

Two specs share this module because the (component, name) registry key means STT
``openai`` and LLM ``openai`` are distinct slots:

* ``STT_SPEC`` (component ``stt``) — MVP batch transcription: buffer the caller
  turn's 8 kHz PCM16, wrap it in a WAV container with stdlib ``wave`` (no
  resample, no ffmpeg) and POST it as a multipart file to
  ``POST {base}/audio/transcriptions`` (model ``gpt-4o-transcribe``). Implements
  only ``transcribe(turn)`` so ``TurnBufferedSTTSession`` wraps it automatically.
* ``LLM_SPEC`` (component ``llm``) — streaming chat: the exact
  ``POST {base}/chat/completions`` ``stream=true`` wire format
  ``openai_compatible_llm`` speaks, pointed at ``https://api.openai.com/v1`` and
  reusing the shared SSE parser.

Both are stdlib ``urllib`` only — no SDK. Keys are read from env with a graceful
absent (construction never raises; the request fails closed at call time).

Config (env):
  OPENAI_API_KEY               required for live use
  OPENAI_BASE_URL              API base (default ``https://api.openai.com/v1``)
  OPENAI_LLM_MODEL             chat model (default ``gpt-4o-mini``)
  OPENAI_STT_MODEL             transcription model (default ``gpt-4o-transcribe``)
  OPENAI_SYSTEM_PROMPT         system prompt (falls back to the Gemini prompt)
  OPENAI_TEMPERATURE / OPENAI_MAX_TOKENS / OPENAI_TIMEOUT_SECONDS
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .audio_encode import pcm16_to_wav
from .frames import TextChunk, TranscriptEvent
from .openai_chat import stream_chat_completion_chunks
from .provider_http import encode_multipart
from .registry import ProviderSpec
from .turn_detection import DetectedTurn

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL = "gpt-4o-mini"
_DEFAULT_STT_MODEL = "gpt-4o-transcribe"
_EDGE_RATE = 8000


def _base_url() -> str:
    return (os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL).rstrip("/")


def _llm_model() -> str:
    return (os.environ.get("OPENAI_LLM_MODEL", _DEFAULT_LLM_MODEL) or _DEFAULT_LLM_MODEL).strip()


def _stt_model() -> str:
    return (os.environ.get("OPENAI_STT_MODEL", _DEFAULT_STT_MODEL) or _DEFAULT_STT_MODEL).strip()


# ---------------------------------------------------------------------------
# LLM: streaming chat.completions
# ---------------------------------------------------------------------------
class OpenAIStreamingLLM:
    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._base_url = _base_url()
        self._model = _llm_model()
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._system_prompt = os.environ.get(
            "OPENAI_SYSTEM_PROMPT", getattr(config, "gemini_system_prompt", "") or ""
        )
        self._temperature = float(
            os.environ.get("OPENAI_TEMPERATURE", str(getattr(config, "gemini_temperature", 0.3)))
        )
        self._max_tokens = int(
            os.environ.get("OPENAI_MAX_TOKENS", str(getattr(config, "gemini_max_output_tokens", 120)))
        )
        self._timeout = float(
            os.environ.get("OPENAI_TIMEOUT_SECONDS", str(getattr(config, "gemini_timeout_seconds", 30.0)))
        )
        self._opener = opener

    def _payload(self, messages: list[dict[str, str]]) -> bytes:
        chat_messages: list[dict[str, str]] = []
        if self._system_prompt:
            chat_messages.append({"role": "system", "content": self._system_prompt})
        chat_messages.extend(messages)
        return json.dumps(
            {
                "model": self._model,
                "messages": chat_messages,
                "stream": True,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
        ).encode("utf-8")

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=self._payload(messages),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        if self._opener is not None:
            response = self._opener(request, timeout=self._timeout)
        else:
            response = urllib.request.urlopen(request, timeout=self._timeout)
        with response:
            yield from stream_chat_completion_chunks(response, provider="openai")


# ---------------------------------------------------------------------------
# STT: batch (turn-buffered) transcription
# ---------------------------------------------------------------------------
class OpenAIBatchSTT:
    """Turn-buffered OpenAI transcription; wrapped by ``TurnBufferedSTTSession``."""

    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._base_url = _base_url()
        self._model = _stt_model()
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._language = (getattr(config, "fake_language", "") or "").strip()
        self._timeout = float(
            os.environ.get("OPENAI_STT_TIMEOUT_SECONDS", str(getattr(config, "gemini_timeout_seconds", 30.0)))
        )
        self._opener = opener

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        pcm = b"".join(frame.pcm16 for frame in turn.frames)
        if not pcm:
            return
        wav = pcm16_to_wav(pcm, _EDGE_RATE)
        fields: dict[str, str] = {"model": self._model}
        if self._language:
            fields["language"] = self._language
        body, content_type = encode_multipart(
            fields, file_field="file", file_name="audio.wav", file_bytes=wav
        )
        request = urllib.request.Request(
            f"{self._base_url}/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": content_type,
            },
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
        yield TranscriptEvent(
            text,
            is_final=True,
            language=self._language or None,
            metadata={"provider": "openai", "model": self._model, "audio_ms": max(0, turn.end_ms - turn.start_ms)},
        )


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    return missing


def _describe_llm(config: object) -> dict:
    return {"provider": "openai", "adapter": "OpenAIStreamingLLM", "real": True, "model_id": _llm_model()}


def _describe_stt(config: object) -> dict:
    return {"provider": "openai", "adapter": "OpenAIBatchSTT", "real": True, "model_id": _stt_model()}


LLM_SPEC = ProviderSpec(
    name="openai",
    component="llm",
    factory=OpenAIStreamingLLM,
    describe=_describe_llm,
    availability=_availability,
    real=True,
)

STT_SPEC = ProviderSpec(
    name="openai",
    component="stt",
    factory=OpenAIBatchSTT,
    describe=_describe_stt,
    availability=_availability,
    real=True,
)

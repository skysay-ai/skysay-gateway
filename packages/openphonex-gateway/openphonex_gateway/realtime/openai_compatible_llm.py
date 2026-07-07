# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""OpenAI-compatible streaming LLM adapter (Ollama / vLLM / LM Studio).

Fully local against any server exposing ``POST {base_url}/chat/completions``
with SSE ``stream=true`` semantics. No SDK is required — it uses stdlib
``urllib`` so importing this module never pulls a heavy client. Reachability is
never checked at import time.

Config (env):
  REALTIME_LLM_BASE_URL     e.g. ``http://localhost:11434/v1`` (default)
  REALTIME_LLM_MODEL        model id served by the backend
  REALTIME_LLM_API_KEY      optional bearer token (``ollama`` dummy allowed)
  REALTIME_LLM_SYSTEM_PROMPT  system prompt (falls back to GEMINI_SYSTEM_PROMPT)
  REALTIME_LLM_TEMPERATURE / REALTIME_LLM_MAX_TOKENS / REALTIME_LLM_TIMEOUT_SECONDS
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .frames import TextChunk
from .openai_chat import stream_chat_completion_chunks
from .pipeline import StreamingLLM
from .registry import ProviderSpec


def _base_url() -> str:
    return (os.environ.get("REALTIME_LLM_BASE_URL", "http://localhost:11434/v1") or "").rstrip("/")


def _model() -> str:
    return os.environ.get("REALTIME_LLM_MODEL", "llama3")


class OpenAICompatibleLLM(StreamingLLM):
    def __init__(self, config: object):
        self.config = config
        self._base_url = _base_url()
        self._model = _model()
        self._api_key = os.environ.get("REALTIME_LLM_API_KEY", "ollama")
        self._system_prompt = os.environ.get(
            "REALTIME_LLM_SYSTEM_PROMPT", getattr(config, "gemini_system_prompt", "") or ""
        )
        self._temperature = float(
            os.environ.get("REALTIME_LLM_TEMPERATURE", str(getattr(config, "gemini_temperature", 0.3)))
        )
        self._max_tokens = int(
            os.environ.get("REALTIME_LLM_MAX_TOKENS", str(getattr(config, "gemini_max_output_tokens", 120)))
        )
        self._timeout = float(
            os.environ.get("REALTIME_LLM_TIMEOUT_SECONDS", str(getattr(config, "gemini_timeout_seconds", 30.0)))
        )

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
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            yield from stream_chat_completion_chunks(response, provider="openai_compatible")


def _availability(config: object) -> list[str]:
    # Purely local, stdlib-only: nothing to import. Reachability is checked at
    # call time (fails closed for live), never at import time (skips in CI).
    return []


def _describe(config: object) -> dict:
    return {
        "provider": "openai_compatible",
        "adapter": "OpenAICompatibleLLM",
        "real": True,
        "model_id": _model(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="openai_compatible",
    component="llm",
    factory=OpenAICompatibleLLM,
    describe=_describe,
    availability=_availability,
    real=True,
    aliases=("ollama", "vllm", "local_llm"),
)

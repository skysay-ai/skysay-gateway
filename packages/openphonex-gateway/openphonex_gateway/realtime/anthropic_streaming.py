# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Anthropic Claude streaming LLM adapter (registry name ``claude``).

Streams short voice-agent replies over the Anthropic Messages API
(``POST https://api.anthropic.com/v1/messages`` with ``"stream": true``) using
stdlib ``urllib`` only — no SDK. The SSE framing is the same ``data:`` JSON
framing Gemini uses, so it reuses ``iter_sse_json`` from ``gemini_streaming``.
``content_block_delta.delta.text`` events become ``TextChunk`` deltas and a
terminal ``TextChunk("", is_final=True)`` is emitted on ``message_stop``. The
system prompt is sent as the top-level ``system`` field (never as a message).

Config (env):
  ANTHROPIC_API_KEY            required for live use (graceful-absent otherwise)
  ANTHROPIC_MODEL              model id (default ``claude-3-5-haiku-latest``)
  ANTHROPIC_SYSTEM_PROMPT      system prompt (falls back to the Gemini prompt)
  ANTHROPIC_TEMPERATURE / ANTHROPIC_MAX_TOKENS / ANTHROPIC_TIMEOUT_SECONDS
  ANTHROPIC_VERSION            API version header (default ``2023-06-01``)

Tool-calling seam (not yet implemented):
  This adapter does NOT set ``supports_tools``, so the pipeline never passes
  ``tools=`` and keeps the prompt-only path even when an agent has tools
  configured. To implement native tool calling here, mirror the Gemini
  adapter (``gemini_streaming.GeminiStreamingLLM``):

  * ``_payload``: add the top-level ``tools`` array (name / description /
    ``input_schema`` from each declaration's JSON-schema ``parameters``) and
    map ``{"role": "assistant", "tool_calls": [...]}`` conversation entries to
    ``tool_use`` content blocks and ``{"role": "tool", ...}`` entries to
    ``tool_result`` blocks on a user message;
  * ``stream_reply``: parse ``content_block_start`` events with
    ``content_block.type == "tool_use"`` plus their ``input_json_delta``
    deltas, and on ``content_block_stop`` yield
    ``TextChunk("", metadata={"tool_call": {"name": ..., "arguments": ...}})``;
  * set ``supports_tools = True`` and accept the optional ``tools`` kwarg.

  A placeholder skipped test pins this contract in
  tests/test_realtime_tool_calling.py.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterable

from .frames import TextChunk
from .gemini_streaming import iter_sse_json
from .registry import ProviderSpec

_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-3-5-haiku-latest"


def _model() -> str:
    return (os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL).strip()


class AnthropicStreamingLLM:
    """Anthropic Messages streaming adapter (StreamingLLM + FallbackLLM)."""

    def __init__(self, config: object, *, opener=None):
        self.config = config
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = _model()
        self._version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
        self._system_prompt = os.environ.get(
            "ANTHROPIC_SYSTEM_PROMPT", getattr(config, "gemini_system_prompt", "") or ""
        )
        self._temperature = float(
            os.environ.get("ANTHROPIC_TEMPERATURE", str(getattr(config, "gemini_temperature", 0.3)))
        )
        self._max_tokens = int(
            os.environ.get("ANTHROPIC_MAX_TOKENS", str(getattr(config, "gemini_max_output_tokens", 120)))
        )
        self._timeout = float(
            os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", str(getattr(config, "gemini_timeout_seconds", 30.0)))
        )
        self._opener = opener

    # -- payload / request ------------------------------------------------
    def _payload(self, messages: list[dict[str, str]], *, stream: bool, model_id: str = "") -> bytes:
        chat_messages: list[dict[str, str]] = []
        for message in messages:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            role = "assistant" if message.get("role") == "assistant" else "user"
            chat_messages.append({"role": role, "content": content})
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Respond briefly."}]
        payload: dict[str, object] = {
            "model": (model_id or self._model),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": chat_messages,
            "stream": stream,
        }
        system = self._system_prompt.strip()
        if system:
            payload["system"] = system  # top-level, NOT a message
        return json.dumps(payload).encode("utf-8")

    def _request(self, payload: bytes, *, stream: bool) -> urllib.request.Request:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._version,
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return urllib.request.Request(_MESSAGES_URL, data=payload, headers=headers, method="POST")

    def _urlopen(self, request: urllib.request.Request):
        if self._opener is not None:
            return self._opener(request, timeout=self._timeout)
        return urllib.request.urlopen(request, timeout=self._timeout)

    # -- streaming leg ----------------------------------------------------
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        meta = {"provider": "anthropic", "model": self._model}
        saw_text = False
        request = self._request(self._payload(messages, stream=True), stream=True)
        try:
            response = self._urlopen(request)
        except urllib.error.HTTPError as exc:  # pragma: no cover - live-only error surface
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Anthropic messages HTTP {exc.code}: {detail[-800:]}") from exc
        with response:
            for event in iter_sse_json(response):
                etype = event.get("type")
                if etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if isinstance(delta, dict):
                        text = str(delta.get("text") or "")
                        if text:
                            saw_text = True
                            yield TextChunk(text, is_final=False, metadata=dict(meta))
                elif etype == "message_stop":
                    yield TextChunk("", is_final=True, metadata={**meta, "empty": not saw_text})
                    return
        yield TextChunk("", is_final=True, metadata={**meta, "empty": not saw_text})

    # -- non-streaming fallback leg ---------------------------------------
    def generate_reply(self, messages: list[dict[str, str]], *, model_id: str = "") -> str:
        request = self._request(self._payload(messages, stream=False, model_id=model_id), stream=False)
        try:
            with self._urlopen(request) as response:
                raw = response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:  # pragma: no cover - live-only error surface
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Anthropic messages HTTP {exc.code}: {detail[-800:]}") from exc
        obj = json.loads(raw or "{}")
        if not isinstance(obj, dict):
            raise ValueError("Anthropic messages response was not a JSON object")
        parts = obj.get("content") or []
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                texts.append(str(part["text"]))
        return "".join(texts)


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "anthropic",
        "adapter": "AnthropicStreamingLLM",
        "real": True,
        "model_id": _model(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="claude",
    component="llm",
    factory=AnthropicStreamingLLM,
    describe=_describe,
    availability=_availability,
    real=True,
    aliases=("anthropic",),
)

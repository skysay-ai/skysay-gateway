# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from .frames import TextChunk


@dataclass(frozen=True)
class GeminiStreamingLLMConfig:
    api_key: str
    model_id: str = "gemini-3.5-flash-lite"
    temperature: float = 0.3
    max_output_tokens: int = 120
    # How little the model may think, in the shape its family accepts. The 2.5
    # models take a token budget and reject a level; 3.x replaced the budget
    # with a level and rejects the budget. Neither is a superset, so pinning a
    # 2.5 model means setting thinking_level=None and thinking_budget=0.
    thinking_level: str | None = "minimal"
    thinking_budget: int | None = None
    system_prompt: str = ""
    timeout_seconds: float = 30.0


class GeminiStreamingLLM:
    """Gemini streamGenerateContent adapter for short voice-agent replies."""

    # The pipeline only passes ``tools=`` to adapters that declare
    # native tool-calling support; other adapters keep the prompt-only path.
    supports_tools = True

    def __init__(self, config: GeminiStreamingLLMConfig):
        if not config.api_key:
            raise ValueError("Gemini API key is required")
        if not config.model_id:
            raise ValueError("Gemini model_id is required")
        self.config = config

    def stream_reply(self, messages: list[dict[str, str]], tools: list[dict[str, object]] | None = None) -> Iterable[TextChunk]:
        saw_text = False
        meta = {"provider": "gemini", "model": self.config.model_id}
        for part in self._stream_parts(messages, tools=tools):
            if isinstance(part, dict):
                # Model-emitted function call: surfaced as an empty-text chunk
                # with tool_call metadata so nothing reaches TTS.
                yield TextChunk("", is_final=False, metadata={**meta, "tool_call": part})
                continue
            if not part:
                continue
            saw_text = True
            yield TextChunk(part, is_final=False, metadata=dict(meta))
        yield TextChunk("", is_final=True, metadata={**meta, "empty": not saw_text})

    def generate_reply(self, messages: list[dict[str, str]], *, model_id: str = "") -> str:
        """Single non-streaming generateContent call (LLM fallback leg).

        Reuses the streaming client's payload/request plumbing; ``model_id``
        optionally overrides the configured model (e.g. a faster fallback).
        """

        payload = self._payload(messages)
        request = self._request(payload, stream=False, model_id=model_id)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if self._rejected_the_thinking_config(detail):
                payload = self._payload(messages, include_thinking=False)
                request = self._request(payload, stream=False, model_id=model_id)
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    raw = response.read().decode(errors="replace")
            else:
                raise RuntimeError(f"Gemini generateContent HTTP {exc.code}: {detail[-800:]}") from exc
        event = json.loads(raw or "{}")
        if not isinstance(event, dict):
            raise ValueError("Gemini generateContent response was not a JSON object")
        return "".join(extract_gemini_text(event))

    def _stream_text(self, messages: list[dict[str, str]], *, retry_without_thinking: bool = True) -> Iterable[str]:
        for part in self._stream_parts(messages, retry_without_thinking=retry_without_thinking):
            if isinstance(part, str):
                yield part

    def _stream_parts(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, object]] | None = None,
        retry_without_thinking: bool = True,
    ) -> Iterable[str | dict[str, object]]:
        """Yield text deltas (str) and function calls (dict) from one stream."""

        payload = self._payload(messages, tools=tools)
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                for event in iter_sse_json(response):
                    yield from extract_gemini_parts(event)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if retry_without_thinking and self._rejected_the_thinking_config(detail):
                payload = self._payload(messages, tools=tools, include_thinking=False)
                request = self._request(payload)
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    for event in iter_sse_json(response):
                        yield from extract_gemini_parts(event)
                return
            raise RuntimeError(f"Gemini streamGenerateContent HTTP {exc.code}: {detail[-800:]}") from exc

    def _thinking_config(self) -> dict[str, object] | None:
        if self.config.thinking_level is not None:
            return {"thinkingLevel": self.config.thinking_level}
        if self.config.thinking_budget is not None:
            return {"thinkingBudget": self.config.thinking_budget}
        return None

    def _rejected_the_thinking_config(self, detail: str) -> bool:
        """Whether a 4xx is the model refusing the thinking field we sent.

        A model that has moved to the other shape answers a bare
        ``INVALID_ARGUMENT`` naming no field at all, so matching on the word
        alone leaves the caller with a hard failure instead of a reply.

        The bare form is the whole signal, and it has to stay that narrow:
        when the provider objects to some other field it says which, as
        ``Invalid value at '<path>'``. Retrying those without thinking would
        send the same rejected request a second time and report the wrong
        cause.
        """
        lowered = detail.lower()
        if "thinking" in lowered:
            return True
        if self._thinking_config() is None or "invalid_argument" not in lowered:
            return False
        return "invalid value at" not in lowered

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        include_thinking: bool = True,
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_output_tokens,
        }
        if include_thinking:
            thinking = self._thinking_config()
            if thinking is not None:
                generation_config["thinkingConfig"] = thinking
        payload: dict[str, object] = {
            "contents": _messages_to_contents(messages),
            "generationConfig": generation_config,
        }
        if not payload["contents"]:
            payload["contents"] = [{"role": "user", "parts": [{"text": "Respond briefly."}]}]
        if self.config.system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": self.config.system_prompt.strip()}]}
        if tools:
            # Native Gemini function calling. Declarations use the
            # OpenAPI-subset schema Gemini shares with the JSON-schema specs.
            payload["tools"] = [{"functionDeclarations": [dict(tool) for tool in tools]}]
        return payload

    def _request(self, payload: dict[str, object], *, stream: bool = True, model_id: str = "") -> urllib.request.Request:
        resolved_model_id = (model_id or self.config.model_id).removeprefix("models/")
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(resolved_model_id, safe='')}:{action}"
        )
        return urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"x-goog-api-key": self.config.api_key, "Content-Type": "application/json"},
            method="POST",
        )


def iter_sse_json(lines: Iterable[bytes]) -> Iterable[dict[str, object]]:
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode(errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield _json_event("\n".join(data_lines))
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data = line[5:].lstrip()
            if data == "[DONE]":
                break
            data_lines.append(data)
    if data_lines:
        yield _json_event("\n".join(data_lines))


def extract_gemini_text(event: dict[str, object]) -> Iterable[str]:
    for part in extract_gemini_parts(event):
        if isinstance(part, str):
            yield part


def extract_gemini_parts(event: dict[str, object]) -> Iterable[str | dict[str, object]]:
    """Yield text deltas (str) and functionCall parts (dict) from one event.

    A function call is normalized to ``{"name": ..., "arguments": {...}}`` so
    downstream code never touches Gemini's ``args`` field name. Gemini's
    ``functionCall.id`` (parallel-call correlation) and the part-level
    ``thoughtSignature`` (thinking continuity) are carried through as ``id`` /
    ``thought_signature`` — they MUST be echoed on the follow-up turn or the
    continuation fails HTTP 400 (so the tool side effect runs but the model
    never gets to answer the caller).
    """

    for candidate in event.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if part.get("text"):
                yield str(part["text"])
            function_call = part.get("functionCall")
            if isinstance(function_call, dict) and function_call.get("name"):
                args = function_call.get("args")
                emitted: dict[str, object] = {
                    "name": str(function_call["name"]),
                    "arguments": dict(args) if isinstance(args, dict) else {},
                }
                call_id = function_call.get("id")
                if call_id:
                    emitted["id"] = str(call_id)
                thought_signature = part.get("thoughtSignature")
                if thought_signature:
                    emitted["thought_signature"] = str(thought_signature)
                yield emitted


def _json_event(data: str) -> dict[str, object]:
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini SSE event was not a JSON object")
    return parsed


def _message_to_content(message: dict[str, str]) -> dict[str, object]:
    role = "model" if message.get("role") == "assistant" else "user"
    return {"role": role, "parts": [{"text": str(message.get("content", "")).strip()}]}


def _messages_to_contents(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    """Map conversation messages (incl. tool exchanges) to contents.

    * ``{"role": "assistant", "tool_calls": [...]}`` -> a model turn of
      ``functionCall`` parts. Each call's ``id`` is written back into
      ``functionCall.id`` and its ``thought_signature`` onto the part's
      ``thoughtSignature`` (H6) so Gemini accepts the continuation;
    * ``{"role": "tool", ...}`` results -> ``functionResponse`` parts. Gemini
      requires ONE turn holding EVERY matching response for a model turn, so a
      run of consecutive tool results is grouped into a SINGLE ``user`` turn
      (H7); each response echoes its ``tool_call_id`` as ``functionResponse.id``.
      A tool result is only mapped when the matching ``functionCall`` survived
      the ``[-10:]`` window — an orphaned result (its call sliced off) degrades
      to plain text so the payload is never structurally invalid;
    * plain messages keep the exact pre-tool-calling mapping.
    """

    contents: list[dict[str, object]] = []
    called_names: set[str] = set()
    pending_tool_parts: list[dict[str, object]] = []

    def flush_tool_parts() -> None:
        if pending_tool_parts:
            contents.append({"role": "user", "parts": list(pending_tool_parts)})
            pending_tool_parts.clear()

    for message in messages:
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            flush_tool_parts()
            parts: list[dict[str, object]] = []
            for call in tool_calls:
                if not isinstance(call, dict) or not call.get("name"):
                    continue
                name = str(call["name"])
                called_names.add(name)
                arguments = call.get("arguments")
                function_call: dict[str, object] = {
                    "name": name,
                    "args": dict(arguments) if isinstance(arguments, dict) else {},
                }
                call_id = call.get("id")
                if call_id:
                    function_call["id"] = str(call_id)
                part: dict[str, object] = {"functionCall": function_call}
                thought_signature = call.get("thought_signature")
                if thought_signature:
                    part["thoughtSignature"] = str(thought_signature)
                parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            name = str(message.get("name") or "")
            raw = str(message.get("content", ""))
            try:
                response = json.loads(raw or "{}")
            except json.JSONDecodeError:
                response = {"result": raw}
            if not isinstance(response, dict):
                response = {"result": response}
            if name and name in called_names:
                function_response: dict[str, object] = {"name": name, "response": response}
                call_id = message.get("tool_call_id")
                if call_id:
                    function_response["id"] = str(call_id)
                pending_tool_parts.append({"functionResponse": function_response})
            elif raw.strip():
                flush_tool_parts()
                contents.append({"role": "user", "parts": [{"text": f"Tool {name or 'call'} returned: {raw.strip()}"}]})
            continue
        flush_tool_parts()
        if str(message.get("content", "")).strip():
            contents.append(_message_to_content(message))
    flush_tool_parts()
    return contents

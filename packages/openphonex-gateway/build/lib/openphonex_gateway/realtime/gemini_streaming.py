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
    model_id: str = "gemini-2.5-flash-lite"
    temperature: float = 0.3
    max_output_tokens: int = 120
    thinking_budget: int | None = 0
    system_prompt: str = ""
    timeout_seconds: float = 30.0


class GeminiStreamingLLM:
    """Gemini streamGenerateContent adapter for short voice-agent replies."""

    def __init__(self, config: GeminiStreamingLLMConfig):
        if not config.api_key:
            raise ValueError("Gemini API key is required")
        if not config.model_id:
            raise ValueError("Gemini model_id is required")
        self.config = config

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        saw_text = False
        for text in self._stream_text(messages):
            if not text:
                continue
            saw_text = True
            yield TextChunk(text, is_final=False, metadata={"provider": "gemini", "model": self.config.model_id})
        yield TextChunk("", is_final=True, metadata={"provider": "gemini", "model": self.config.model_id, "empty": not saw_text})

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
            if "thinking" in detail.lower():
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
        payload = self._payload(messages)
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                for event in iter_sse_json(response):
                    yield from extract_gemini_text(event)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if retry_without_thinking and "thinking" in detail.lower():
                payload = self._payload(messages, include_thinking=False)
                request = self._request(payload)
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    for event in iter_sse_json(response):
                        yield from extract_gemini_text(event)
                return
            raise RuntimeError(f"Gemini streamGenerateContent HTTP {exc.code}: {detail[-800:]}") from exc

    def _payload(self, messages: list[dict[str, str]], *, include_thinking: bool = True) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_output_tokens,
        }
        if include_thinking and self.config.thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": self.config.thinking_budget}
        payload: dict[str, object] = {
            "contents": [_message_to_content(message) for message in messages if str(message.get("content", "")).strip()],
            "generationConfig": generation_config,
        }
        if not payload["contents"]:
            payload["contents"] = [{"role": "user", "parts": [{"text": "Respond briefly."}]}]
        if self.config.system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": self.config.system_prompt.strip()}]}
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
    for candidate in event.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text"):
                yield str(part["text"])


def _json_event(data: str) -> dict[str, object]:
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini SSE event was not a JSON object")
    return parsed


def _message_to_content(message: dict[str, str]) -> dict[str, object]:
    role = "model" if message.get("role") == "assistant" else "user"
    return {"role": role, "parts": [{"text": str(message.get("content", "")).strip()}]}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Shared OpenAI chat.completions SSE parsing.

Both the local ``openai_compatible_llm`` adapter (Ollama / vLLM / LM Studio)
and the hosted ``openai_streaming`` OpenAI adapter speak the identical
``POST .../chat/completions`` ``stream=true`` wire format, so the delta-parsing
loop lives here once. Given the raw response line iterator it yields
``TextChunk`` deltas and always terminates with a single
``TextChunk("", is_final=True)`` so downstream ``speakable_chunks`` sees exactly
one final marker per reply.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator

from .frames import TextChunk


def stream_chat_completion_chunks(response_lines: Iterable[bytes], *, provider: str) -> Iterator[TextChunk]:
    """Parse chat.completions SSE ``data:`` lines into ``TextChunk`` deltas."""

    emitted_final = False
    for raw_line in response_lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            yield TextChunk("", is_final=True, metadata={"provider": provider})
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        finish = choice.get("finish_reason")
        if text:
            yield TextChunk(text, is_final=bool(finish), metadata={"provider": provider})
            emitted_final = emitted_final or bool(finish)
        elif finish:
            yield TextChunk("", is_final=True, metadata={"provider": provider})
            emitted_final = True
    if not emitted_final:
        yield TextChunk("", is_final=True, metadata={"provider": provider})

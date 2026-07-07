# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import re
from typing import Iterable, Protocol

from .frames import AudioFrame, TextChunk


class StreamingTTS(Protocol):
    def stream(self, text: str) -> Iterable[AudioFrame]:
        ...


def speakable_chunks(
    chunks: Iterable[TextChunk],
    *,
    min_chars: int = 4,
    first_phrase_soft_limit: int = 48,
    phrase_soft_limit: int = 96,
) -> Iterable[str]:
    """Turn LLM deltas into short phrases that TTS can start speaking early."""

    buffer = ""
    emitted_phrases = 0
    for chunk in chunks:
        buffer += chunk.text
        while True:
            match = re.search(r"([.!?]\s+|,\s+)", buffer)
            phrase_end: int | None = None
            if match and (match.end() >= min_chars or chunk.is_final):
                phrase_end = match.end()
            else:
                soft_limit = first_phrase_soft_limit if emitted_phrases == 0 else phrase_soft_limit
                phrase_end = _soft_phrase_boundary(buffer, soft_limit=soft_limit, min_chars=min_chars)
            if phrase_end is None:
                break
            phrase = buffer[:phrase_end].strip()
            buffer = buffer[phrase_end:]
            if phrase:
                emitted_phrases += 1
                yield phrase
        if chunk.is_final and buffer.strip():
            yield buffer.strip()
            buffer = ""


def _soft_phrase_boundary(buffer: str, *, soft_limit: int, min_chars: int) -> int | None:
    if soft_limit <= 0 or len(buffer) < soft_limit:
        return None
    search_start = max(min_chars, soft_limit)
    boundary = -1
    for index in range(min(len(buffer), soft_limit), min_chars - 1, -1):
        if buffer[index - 1].isspace():
            boundary = index
            break
    if boundary < min_chars:
        boundary = -1
        for index in range(search_start, len(buffer)):
            if buffer[index].isspace():
                boundary = index + 1
                break
    if boundary < min_chars:
        return None
    return boundary

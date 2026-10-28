# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Exact, bounded speech partitions; no word cuts or orthography changes."""
from __future__ import annotations

try:
    from .workflow_speech import MAX_REPLY_SEGMENTS, SpeechRefused
except ImportError:
    from workflow_speech import MAX_REPLY_SEGMENTS, SpeechRefused


def source_text_fits(text: str, declaration) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        bound = declaration.max_input_utf8
        return (declaration.role == "authored" and type(bound) is int and 1 <= bound <= 64000
                and len(text.encode("utf-8")) <= bound)
    except UnicodeError:
        return False


def split_source_line(text: str, declaration) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise SpeechRefused("speech segmentation requires bounded authored text")
    try:
        text.encode("utf-8")
    except UnicodeError:
        raise SpeechRefused("speech segmentation requires valid authored text") from None
    cuts, index = [], 0
    while index < len(text):
        if text[index].isspace():
            while index < len(text) and text[index].isspace():
                index += 1
            cuts.append(index)
        else:
            index += 1
    if not cuts or cuts[-1] != len(text):
        cuts.append(len(text))
    pieces, start = [], 0
    while start < len(text):
        selected = None
        for end in cuts:
            if end <= start or not text[start:end].strip():
                continue
            if not source_text_fits(text[start:end], declaration):
                break
            selected = end
        if selected is None:
            raise SpeechRefused("an authored word exceeds its source text bound")
        pieces.append(text[start:selected])
        if len(pieces) > MAX_REPLY_SEGMENTS:
            raise SpeechRefused("speech line exceeds its segment capacity")
        start = selected
    return tuple(pieces)

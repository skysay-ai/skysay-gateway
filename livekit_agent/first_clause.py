# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Release the first clause of a reply to the synthesizer before the sentence ends.

LiveKit's ``tts.StreamAdapter`` turns a streamed LLM reply into sentence-sized
synthesis requests. Its sentence stream releases a sentence only once the text
AFTER it has started arriving, so the first audio of every turn waits for the
whole first sentence plus that lookahead. Measured on the phone path, that wait
is the largest part of the response-time regression the chunked transport
brought with it: the provider's own time to first byte moved by tens of
milliseconds, the sentence wait by hundreds.

A clause is the smallest span a listener already hears as a pause -- the text
before a comma, semicolon, colon or spaced dash. This tokenizer releases the
first such span as soon as it is long enough to carry the next request's
time-to-first-byte, then hands every later character to the ordinary sentence
tokenizer unchanged. Only the first request of a reply is shortened; the rest of
the reply keeps sentence-level phrasing.

The splitting rules are pure functions with no SDK dependency so they can be
tested without a LiveKit installation. Only :func:`first_clause_tokenizer`
imports the SDK, at call time, inside the worker image.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: The clause boundary is only real once the character AFTER the punctuation is
#: known to be whitespace: "1,000", "3:30" and "a;b" must never split.
CLAUSE_MARKS = (",", ";", ":")
#: A dash separates clauses only when it is spaced on both sides.
DASH_MARKS = (" — ", " – ", " - ")
SENTENCE_MARKS = (".", "!", "?", "。", "！", "？")
#: Abbreviations whose period ends no sentence. Lower-cased, no trailing period.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc", "no", "inc", "ltd"}
)
#: Fewer characters than this synthesize to less audio than the next request's
#: time to first byte, which would open a gap after the clause instead of
#: closing one before it. About one second of speech at conversational pace.
DEFAULT_MIN_CLAUSE_CHARS = 16
#: Past this much buffered text with no clause boundary, stop looking and let
#: the sentence tokenizer take over; the reply is one long clause anyway.
DEFAULT_MAX_WAIT_CHARS = 120


_OPENERS = frozenset("\"'“‘«(")


def _period_ends_a_sentence(text: str, index: int) -> bool:
    """Whether the period at ``index`` ends a sentence rather than a token.

    Only a period whose preceding word is an ordinary word and whose following
    character (after the whitespace) starts a new sentence qualifies: an
    initial (``J.``), a dotted abbreviation (``a.m.``, ``U.S.``), a numeric
    ordinal (``23.`` in Estonian or German dates), a listed abbreviation
    (``Dr.``) and a period followed by a lowercase continuation are all left
    to the sentence tokenizer, which keeps such phrases intact. When the
    character after the whitespace has not arrived yet the answer is "not
    yet", and the caller keeps buffering.
    """

    words = text[:index].rstrip(".").split()
    if not words:
        return False
    word = words[-1].lstrip("(\"'“‘«")
    if not word or (len(word) == 1 and word.isalpha()) or "." in word:
        return False
    if word[-1].isdigit() or word.lower() in _ABBREVIATIONS:
        return False
    following = text[index + 2 : index + 3]
    if not following:
        return False
    return following.isupper() or following.isdigit() or following in _OPENERS


def first_clause_split(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_CLAUSE_CHARS,
    max_wait_chars: int = DEFAULT_MAX_WAIT_CHARS,
) -> tuple[str, str] | None:
    """Split ``text`` at its earliest usable clause boundary.

    Returns ``(clause, remainder)`` where ``clause`` keeps its original
    formatting and ends with the boundary punctuation, or ``None`` when no
    boundary has arrived yet. A boundary counts only when the whitespace after
    it is already present, so a comma at the end of the buffer waits for the
    next chunk to prove it is not the thousands separator of a number still
    being streamed. Once ``max_wait_chars`` have accumulated without a
    boundary, ``None`` is returned for good: the caller stops holding text.
    """

    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    if max_wait_chars < min_chars:
        raise ValueError("max_wait_chars must be at least min_chars")
    limit = min(len(text), max_wait_chars)
    index = 0
    while index < limit:
        char = text[index]
        boundary_end: int | None = None
        if char in CLAUSE_MARKS or char in SENTENCE_MARKS:
            after = text[index + 1 : index + 2]
            if after and after.isspace():
                if char == "." and not _period_ends_a_sentence(text, index):
                    boundary_end = None
                else:
                    boundary_end = index + 1
        else:
            for dash in DASH_MARKS:
                if index >= 1 and text.startswith(dash, index - 1):
                    # ``index`` sits on the dash itself. The clause keeps the
                    # dash and the remainder starts at the following space, so
                    # a flush right after the dash never leaves a bare dash to
                    # be synthesized on its own.
                    boundary_end = index + 1
                    break
        if boundary_end is not None:
            clause = text[:boundary_end]
            if len(clause.strip()) >= min_chars:
                return clause, text[boundary_end:]
        index += 1
    return None


def first_clause_tokenizer(
    *,
    min_chars: int = DEFAULT_MIN_CLAUSE_CHARS,
    max_wait_chars: int = DEFAULT_MAX_WAIT_CHARS,
    base: Any | None = None,
) -> Any:
    """Build the LiveKit sentence tokenizer that releases the first clause early.

    ``base`` is the tokenizer that owns everything after the first clause; it
    defaults to the exact tokenizer ``tts.StreamAdapter`` would have used on its
    own, so the rest of the reply is phrased identically to today.
    """

    from livekit.agents import tokenize

    base_tokenizer = (
        base
        if base is not None
        else tokenize.blingfire.SentenceTokenizer(retain_format=True)
    )

    class FirstClauseSentenceStream(tokenize.SentenceStream):
        def __init__(self) -> None:
            super().__init__()
            self._base = base_tokenizer.stream()
            self._held = ""
            self._released = False
            self._forward_task = asyncio.create_task(self._forward())

        async def _forward(self) -> None:
            try:
                async for token in self._base:
                    if not self._event_ch.closed:
                        self._event_ch.send_nowait(token)
            finally:
                self._do_close()

        def _release(self) -> None:
            self._released = True
            held, self._held = self._held, ""
            if held:
                self._base.push_text(held)

        def push_text(self, text: str) -> None:
            self._check_not_closed()
            if not text:
                return
            if self._released:
                self._base.push_text(text)
                return
            self._held += text
            split = first_clause_split(
                self._held, min_chars=min_chars, max_wait_chars=max_wait_chars
            )
            if split is not None:
                clause, remainder = split
                self._released = True
                self._held = ""
                self._event_ch.send_nowait(tokenize.TokenData(token=clause))
                if remainder:
                    self._base.push_text(remainder)
            elif len(self._held) >= max_wait_chars:
                self._release()

        def flush(self) -> None:
            self._check_not_closed()
            self._release()
            self._base.flush()

        def end_input(self) -> None:
            self._check_not_closed()
            self._release()
            self._base.end_input()

        async def aclose(self) -> None:
            await self._base.aclose()
            self._forward_task.cancel()
            try:
                await self._forward_task
            except (asyncio.CancelledError, Exception):
                pass
            self._do_close()

    class FirstClauseSentenceTokenizer(tokenize.SentenceTokenizer):
        def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
            return base_tokenizer.tokenize(text, language=language)

        def stream(self, *, language: str | None = None) -> tokenize.SentenceStream:
            return FirstClauseSentenceStream()

    return FirstClauseSentenceTokenizer()


__all__ = [
    "CLAUSE_MARKS",
    "DASH_MARKS",
    "DEFAULT_MAX_WAIT_CHARS",
    "DEFAULT_MIN_CLAUSE_CHARS",
    "SENTENCE_MARKS",
    "first_clause_split",
    "first_clause_tokenizer",
]

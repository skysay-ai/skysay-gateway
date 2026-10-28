# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded final text from one granted logical source, never partial hypotheses."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from .capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS
    from .workflow_input import WorkflowInputCapacityExceeded
except ImportError:
    from capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS
    from workflow_input import WorkflowInputCapacityExceeded


class CallerRecognitionRefused(RuntimeError):
    pass


class CallerRecognitionCapacityExceeded(CallerRecognitionRefused, WorkflowInputCapacityExceeded):
    pass


@dataclass(frozen=True, slots=True)
class FinalCallerTranscript:
    """An adapter's final observation; construction is not acoustic approval."""
    text: str = field(repr=False)
    locale: object = None
    source_languages: object = None


@dataclass(frozen=True, slots=True)
class CallerLanguageHint:
    locale: str | None
    source_languages: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class CallerRecognition:
    text: str = field(repr=False)
    language: CallerLanguageHint
    #: The listener's own stage marks (``CallerTurnTiming``), evidence only.
    timing: object = field(default=None, repr=False, compare=False)


def _language_tag(value: object) -> str | None:
    return value if (isinstance(value, str) and len(value) <= 64
                     and re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", value)) else None


class CallerTextCollector:
    """One complete turn; interim events never consume the final text budget."""
    def __init__(self):
        self._parts, self._locales, self._sources = [], set(), set()
        self._characters = self._finals = 0

    def append(self, value: FinalCallerTranscript):
        if not isinstance(value, FinalCallerTranscript) or type(value.text) is not str:
            raise CallerRecognitionRefused("caller recognition has no typed final hypothesis")
        try:
            value.text.encode("utf-8")
        except UnicodeError:
            raise CallerRecognitionRefused("caller recognition has malformed final text") from None
        self._finals += 1
        if self._finals > MAX_CALLER_TURN_CHARACTERS:
            raise CallerRecognitionCapacityExceeded("caller recognition exceeded its final event budget")
        if not value.text.strip():
            return
        self._characters += len(value.text) + bool(self._parts)
        if self._characters > MAX_CALLER_TURN_CHARACTERS:
            raise CallerRecognitionCapacityExceeded("caller recognition exceeded its logical text capacity")
        self._parts.append(value.text)
        self._locales.add(_language_tag(value.locale))
        sources = value.source_languages
        source = _language_tag(sources[0]) if isinstance(sources, (list, tuple)) and len(sources) == 1 else None
        self._sources.add((source,) if source else None)

    def result(self) -> CallerRecognition | None:
        if not self._parts:
            return None
        return CallerRecognition(" ".join(self._parts), CallerLanguageHint(
            next(iter(self._locales)) if len(self._locales) == 1 else None,
            next(iter(self._sources)) if len(self._sources) == 1 else None))

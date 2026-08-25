# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict native-usage parsing for Gemini text-to-speech responses.

Gemini TTS prices both the prompt text and generated audio in provider token
units.  The response carries those cumulative counts in ``usage_metadata``;
this module deliberately accepts only that provider shape so a duration or
character estimate can never be substituted into a paid settlement.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class GeminiTTSUsageError(RuntimeError):
    """Gemini did not return complete exact token evidence for one synthesis."""


_MISSING = object()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def _whole_count(value: Any, *, field: str) -> int:
    # ``google.genai`` exposes native counts as integers.  Refuse booleans,
    # strings and floats rather than coercing an arbitrary payload into a
    # billable unit; the caller can retry safely while the wallet hold remains.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GeminiTTSUsageError(f"Gemini TTS usage metadata omitted exact {field}")
    return value


def native_usage_tokens(usage_metadata: Any) -> tuple[int, int]:
    """Return Gemini's exact ``(prompt, generated_audio)`` token totals.

    Gemini's REST response names output tokens ``candidates_token_count``.
    Some generated client types expose the compatible
    ``response_token_count`` spelling instead, so accept that exact alternate
    only when the canonical field is absent.  Both values are cumulative for
    the request, not per-stream-chunk deltas.
    """
    input_tokens = _field(usage_metadata, "prompt_token_count")
    output_tokens = _field(usage_metadata, "candidates_token_count")
    if output_tokens is _MISSING or output_tokens is None:
        output_tokens = _field(usage_metadata, "response_token_count")
    return (
        _whole_count(input_tokens, field="prompt token count"),
        _whole_count(output_tokens, field="generated audio token count"),
    )


class GeminiTTSUsageTracker:
    """Retain the latest complete native report, while preserving a final error.

    Gemini can send incomplete interim ``usage_metadata`` chunks.  A later
    complete cumulative report supersedes one of those.  The opposite ordering
    must fail closed for a paid synthesis: an invalid later trailer must not
    silently settle from an older, potentially partial count.
    """

    def __init__(self) -> None:
        self._tokens: tuple[int, int] | None = None
        self._latest_error: GeminiTTSUsageError | None = None

    def record(self, usage_metadata: Any) -> None:
        try:
            self._tokens = native_usage_tokens(usage_metadata)
            self._latest_error = None
        except GeminiTTSUsageError as exc:
            self._latest_error = exc

    @property
    def latest_tokens(self) -> tuple[int, int] | None:
        """Return the most recent valid report for an unmetered stream."""
        return self._tokens

    def require_latest_complete(self) -> tuple[int, int]:
        """Return exact paid-call evidence or raise for a missing/bad trailer."""
        if self._latest_error is not None:
            raise self._latest_error
        if self._tokens is None:
            raise GeminiTTSUsageError("Gemini TTS usage metadata is unavailable")
        return self._tokens

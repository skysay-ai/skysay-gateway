# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""How much of what a speech-to-speech agent said reached its transcript.

``evidence_incomplete`` cannot answer this on a voice-to-voice call: it is the
control plane's PR #79 pairing check (one caller clip per agent clip), and a
duplex model that talks over the caller fails it on every call whatever the
transcript holds. This is the measure that can: agent speeches the worker
counted, against the agent speeches the transcript carries text for.

Two signals raise the alarm, each named in ``reasons``:

* ``coverage_below_threshold`` -- the share of agent speeches with text is
  under the threshold. The denominator is the larger of the recorder's agent
  intervals and the worker's own count of the SDK's speaking edges, so a
  speech the recorder never kept still counts as missing. A duplex tool call
  is a generation that never speaks, so it moves neither.
* ``unlinked_audible_speech`` -- an agent speech of at least a second has
  neither text nor a speech identity, anywhere in the call. That is the join
  itself failing, which a ratio can hide on a long call.

Counts and spans only; never text.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping

#: The worker event and GlitchTip phase for a call below the bar.
ALARM_EVENT = "evidence_agent_speech_uncaptured"
#: The worker event carrying the measure on every speech-to-speech call.
MEASURE_EVENT = "agent_speech_coverage"
#: Share of agent speeches that must carry text. A duplex burst can be sound
#: the model never transcribes, so the bar is not 1.0.
DEFAULT_MIN_COVERAGE = 0.8
#: Below this many speeches a single audio-only burst would decide the ratio.
MIN_SPEECHES_FOR_RATIO = 3
#: An untranscribed, unidentified speech this long is not a breath.
UNLINKED_AUDIBLE_MS = 1_000
COVERAGE_ENV = "SKYSAY_TRANSCRIPT_COVERAGE_ALARM_BELOW"


def coverage_threshold(environ: Mapping[str, str] | None = None) -> float:
    """The alarm threshold, from the environment when it names a valid share."""

    raw = (os.environ if environ is None else environ).get(COVERAGE_ENV, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_MIN_COVERAGE
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return DEFAULT_MIN_COVERAGE
    return value


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


class TranscriptCoverageAlarm(RuntimeError):
    """Reported to error tracking when a call's agent speech went uncaptured."""


def agent_speech_coverage_verdict(
    coverage: Mapping[str, Any] | None,
    *,
    agent_speeches_counted: int | None,
    threshold: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any] | None:
    """The call's measure and whether it alarms; ``None`` without a record.

    ``coverage`` is the recorder's ``agent_speech_coverage``. A missing record
    is not a pass: it means the call was not a speech-to-speech recording, and
    the caller decides what that means.
    """

    if not isinstance(coverage, Mapping):
        return None
    intervals = _count(coverage.get("agent_speech_intervals"))
    with_text = min(_count(coverage.get("with_text")), intervals)
    counted = _count(agent_speeches_counted)
    speeches = max(intervals, counted)
    ratio = None if speeches == 0 else round(with_text / speeches, 3)
    reasons: list[str] = []
    if speeches >= MIN_SPEECHES_FOR_RATIO and ratio is not None and ratio < threshold:
        reasons.append("coverage_below_threshold")
    if "longest_unlinked_audio_without_text_ms" in coverage:
        # The recorder's figure spans every untranscribed speech; the listed
        # spans stop at a cap and would hide one past it.
        longest_unlinked = _count(coverage.get("longest_unlinked_audio_without_text_ms"))
        unlinked_count = _count(coverage.get("unlinked_audio_without_text_count"))
    else:
        spans = [
            _count(span.get("end_ms")) - _count(span.get("start_ms"))
            for span in coverage.get("audio_without_text") or []
            if isinstance(span, Mapping) and span.get("identity_linked") is False
        ]
        longest_unlinked = max(spans, default=0)
        unlinked_count = len(spans)
    if longest_unlinked >= UNLINKED_AUDIBLE_MS:
        reasons.append("unlinked_audible_speech")
    return {
        "agent_speeches": speeches,
        "agent_speech_intervals": intervals,
        "agent_speeches_counted": counted,
        "agent_speeches_with_text": with_text,
        "audio_without_text": _count(coverage.get("audio_without_text_count")),
        "unlinked_speeches_without_text": unlinked_count,
        "longest_unlinked_speech_ms": longest_unlinked,
        "text_without_audio": _count(coverage.get("text_without_audio_count")),
        "coverage": ratio,
        "threshold": threshold,
        "alarm": bool(reasons),
        "reasons": reasons,
    }


def report_agent_speech_coverage(
    summary: Mapping[str, Any] | None,
    *,
    agent_speeches_counted: int | None,
    emit: Callable[..., None],
    capture: Callable[..., Any],
    threshold: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any] | None:
    """Emit the measure for one finalized call, and alarm when it is short.

    ``summary`` is the recorder's finalize summary; a call without an
    ``agent_speech_coverage`` record (cascade, or no room audio) emits
    nothing. The alarm is a worker event plus one error-tracking event whose
    tags are the bounded reason codes; neither carries transcript text.
    Never raises: evidence reporting must not decide how teardown ends.
    """

    try:
        coverage = summary.get("agent_speech_coverage") if isinstance(summary, Mapping) else None
        verdict = agent_speech_coverage_verdict(
            coverage,
            agent_speeches_counted=agent_speeches_counted,
            threshold=threshold,
        )
        if verdict is None:
            return None
        emit(MEASURE_EVENT, **verdict)
        if verdict["alarm"]:
            emit(ALARM_EVENT, **verdict)
            capture(
                TranscriptCoverageAlarm(
                    "speech-to-speech agent speech missing from the transcript"
                ),
                phase=ALARM_EVENT,
                reason=",".join(verdict["reasons"]),
            )
        return verdict
    except Exception:  # noqa: BLE001 - a report never breaks teardown
        return None


__all__ = [
    "ALARM_EVENT",
    "COVERAGE_ENV",
    "DEFAULT_MIN_COVERAGE",
    "MEASURE_EVENT",
    "MIN_SPEECHES_FOR_RATIO",
    "TranscriptCoverageAlarm",
    "UNLINKED_AUDIBLE_MS",
    "agent_speech_coverage_verdict",
    "coverage_threshold",
    "report_agent_speech_coverage",
]

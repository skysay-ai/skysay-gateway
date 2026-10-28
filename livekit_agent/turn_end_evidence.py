# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What happened around each caller turn end, kept per call.

Why this exists, concretely.  Asking "did the agent reply while the caller was
still talking?" or "which setting decided this turn?" about a real call meant
pulling its recordings and replaying them offline.  The worker
records two delays per caller turn in ``call-telemetry.json`` and nothing about
the settings the call ran with, the pauses the agent waited through, or whether
the caller spoke again right after the turn was committed.

This module keeps those observations.  It records what was MEASURED and what
was OBSERVED, and nothing else:

* No cause.  Timing alone cannot establish why a turn ended -- the SDK also
  waits on detector inference and on its own commit hook -- so nothing here
  says "the transcript decided" or "the hold decided".  A reader who wants a
  label derives one offline (``scripts/turn_end_evidence_report.py``), marked
  derived, and gets ``unknown`` whenever the numbers do not settle it.
* No transcript text.  Counts, offsets and booleans only.
* No detector probability.  The pinned SDK does not publish it as a session
  event; the summary says so (``detector_observation``) rather than leaving a
  reader to infer it from an absent field.

What the numbers can and cannot say.  A commit followed by caller speech
within :data:`SPEECH_AFTER_COMMIT_WINDOW_SECONDS` is a cut-in CANDIDATE, and a
commit whose last final preceded the last detected speech
(``transcription_delay_zero``) is a partial-transcript CANDIDATE.  Neither is
a finding: that takes the words or a listener.

Nothing here ends, alters or bills a call.  Loop-affine and inert by contract,
like ``caller_audio_flow``: every method is called from the single asyncio
loop running the media session, performs no I/O, and never raises into a media
callback -- the guard lives here, in the module, not at each call site.  The
one I/O path is :func:`write_turn_end_evidence`, called once at teardown.

Host-local operator evidence, like ``interruption-summary.json``: the file is
written into the call's evidence directory and is NOT in the upload list, so
it stays on the worker host.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "ARTIFACT_NAME",
    "MAX_RECORDED_TURNS",
    "SCHEMA",
    "SPEECH_AFTER_COMMIT_WINDOW_SECONDS",
    "STATUSES",
    "STATUS_FIELD",
    "SUMMARY_EVENT",
    "TurnEndEvidence",
    "write_turn_end_evidence",
]

SCHEMA = "skysay.turn_end_evidence.v1"
ARTIFACT_NAME = "turn-end-evidence.json"
SUMMARY_EVENT = "turn_end_evidence_summary"
# Its own key rather than the journal's shared ``status``, whose reviewed
# vocabulary belongs to other events.
STATUS_FIELD = "turn_end_evidence_status"
STATUSES = ("written", "write_failed", "summary_failed", "not_applicable")

# The same ceiling the customer telemetry artifact uses for turns
# (``MAX_TELEMETRY_TURNS``): a call past it keeps counting and stops storing.
MAX_RECORDED_TURNS = 128

# How long after a commit a start-of-speech edge still counts as "the caller
# spoke again right after".  Two seconds covers the reply's own latency on the
# cascade stack (the model's first token plus first audio), which is the window
# in which a caller who was not finished is still mid-thought.
SPEECH_AFTER_COMMIT_WINDOW_SECONDS = 2.0

# The SDK's per-turn timing fields, read off the committed message's metrics.
_DELAY_FIELDS = ("transcription_delay", "end_of_turn_delay")
_ANCHOR_FIELDS = ("started_speaking_at", "stopped_speaking_at")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _milliseconds(seconds: float | None) -> int | None:
    return None if seconds is None else int(round(seconds * 1_000))


class TurnEndEvidence:
    """Bounded per-call record of the observations around each caller commit.

    Fed from three places the worker already has: the agent's
    ``on_user_turn_completed`` hook, the session's ``user_state_changed``
    edges, and its ``user_input_transcribed`` events.

    A row is a caller turn the SDK handed to that hook, including one the
    worker then refuses to answer.  It is NOT every turn the SDK commits: the
    SDK returns before the hook when a reply was already triggered for the
    turn, when the speech in progress cannot be interrupted, and when
    scheduling is paused.  Such a turn has no row here; where it still became
    a conversation item it is counted in ``user_items_added_count``, which is
    why that total can exceed the rows.  One clock for all of
    them -- wall-clock seconds, because the SDK's own speech anchors on the
    committed message are wall-clock -- reported as offsets from the moment
    this object was built.
    """

    def __init__(
        self,
        *,
        settings: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        max_turns: int = MAX_RECORDED_TURNS,
    ) -> None:
        self._settings = dict(settings) if isinstance(settings, Mapping) else {}
        self._clock = clock
        self._origin = self._read_clock()
        self._max_turns = max(1, int(max_turns))
        self._turns: list[dict[str, Any]] = []
        self._turn_count = 0
        self._user_items_added = 0
        self._faults = 0
        # The row the latest commit produced, kept even when it was not stored,
        # so a refusal or a later start-of-speech lands on the turn it belongs
        # to and never on the stored row before it.
        self._current: dict[str, Any] | None = None
        self._committed_at: float | None = None
        self._speaking = False
        self._stopped_at: float | None = None
        self._reset_window()

    # ------------------------------------------------------------ observations
    def note_user_state(self, new_state: Any, *, at: float | None = None) -> None:
        """One ``user_state_changed`` transition."""

        try:
            speaking = str(new_state or "") == "speaking"
            if speaking == self._speaking:
                # Level-tracked: only a CHANGE is an edge.
                return
            self._speaking = speaking
            moment = self._now(at)
            if not speaking:
                self._stopped_at = moment
                return
            if self._stopped_at is not None:
                # The caller stopped and started again with no commit between:
                # a pause the agent waited through.
                pause = max(0.0, moment - self._stopped_at)
                self._held_pauses += 1
                self._longest_held_pause = max(self._longest_held_pause or 0.0, pause)
                self._stopped_at = None
            if self._current is not None and self._committed_at is not None:
                elapsed = moment - self._committed_at
                if 0.0 <= elapsed <= SPEECH_AFTER_COMMIT_WINDOW_SECONDS:
                    self._current["caller_speech_after_commit_ms"] = _milliseconds(elapsed)
                # Only the FIRST start-of-speech after a commit is "right
                # after" it; a later one belongs to the next turn.
                self._committed_at = None
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            self._faults += 1

    def note_transcript(self, *, is_final: Any, has_text: Any, at: float | None = None) -> None:
        """One ``user_input_transcribed`` event; the text itself is never passed in."""

        try:
            if not has_text:
                return
            moment = self._now(at)
            if is_final:
                self._finals += 1
                self._last_final_at = moment
                return
            self._interims += 1
            if self._first_interim_at is None:
                self._first_interim_at = moment
            self._last_interim_at = moment
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            self._faults += 1

    def note_turn_committed(self, message: Any, *, at: float | None = None) -> None:
        """A committed caller turn the SDK handed to ``on_user_turn_completed``."""

        # Detached and counted BEFORE anything fallible: whatever happens
        # below, this commit is a turn of its own, and a refusal or a later
        # start-of-speech must never land on the turn before it.
        self._current = None
        self._committed_at = None
        self._turn_count += 1
        try:
            moment = self._now(at)
            try:
                metrics = getattr(message, "metrics", None)
                source = dict(metrics) if isinstance(metrics, Mapping) else {}
            except Exception:  # noqa: BLE001 - an unreadable message is a turn with no measurements
                self._faults += 1
                source = {}
            delays = {field: _finite(source.get(field)) for field in _DELAY_FIELDS}
            anchors = {field: _finite(source.get(field)) for field in _ANCHOR_FIELDS}
            row: dict[str, Any] = {
                "index": self._turn_count,
                "committed_offset_ms": self._offset(moment),
                "transcription_delay_ms": _milliseconds(delays["transcription_delay"]),
                "end_of_turn_delay_ms": _milliseconds(delays["end_of_turn_delay"]),
                # Exactly zero means the last final arrived no later than the
                # last detected speech.  It is a fact about two timestamps and
                # by itself proves no cause.
                "transcription_delay_zero": delays["transcription_delay"] == 0.0,
                "speech_started_offset_ms": self._offset(anchors["started_speaking_at"]),
                "speech_stopped_offset_ms": self._offset(anchors["stopped_speaking_at"]),
                "held_pauses": self._held_pauses,
                "longest_held_pause_ms": _milliseconds(self._longest_held_pause),
                "interims": self._interims,
                "finals": self._finals,
                "first_interim_offset_ms": self._offset(self._first_interim_at),
                "last_interim_offset_ms": self._offset(self._last_interim_at),
                "last_final_offset_ms": self._offset(self._last_final_at),
                "caller_speaking_at_commit": self._speaking,
                "caller_speech_after_commit_ms": None,
                "reply_refused": False,
            }
            if len(self._turns) < self._max_turns:
                self._turns.append(row)
            self._current = row
            # A caller already speaking at the commit has no later start edge
            # to time; ``caller_speaking_at_commit`` is that case's record.
            self._committed_at = None if self._speaking else moment
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            self._faults += 1
        finally:
            # The next turn's window starts here even when this row could not
            # be built; its pauses and transcripts are not the next turn's.
            self._reset_window()

    def note_reply_refused(self) -> None:
        """The worker refused the reply to the turn it just saw committed."""

        try:
            if self._current is not None:
                self._current["reply_refused"] = True
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            self._faults += 1

    def note_user_item_added(self) -> None:
        """A caller item reached the conversation.

        Counted, not matched to a turn: the SDK may insert a different object
        than the one the commit hook saw, and a refused turn adds none, so the
        two totals are reported side by side and never reconciled by id.
        """

        self._user_items_added += 1

    # ----------------------------------------------------------------- outputs
    def snapshot(self) -> dict[str, Any]:
        """The whole bounded record, as written to the evidence directory."""

        return {
            "schema": SCHEMA,
            "settings": dict(self._settings),
            "turns": [dict(row) for row in self._turns],
            **self._counts(),
        }

    def summary(self) -> dict[str, Any]:
        """Per-call counts for the one worker log line.

        Every key is a ``_count`` measurement, a reviewed flag or a closed
        label, because the durable per-call journal (``worker_events``) keeps
        a field only when its registry recognises the key and drops the rest.
        """

        rows = self._turns
        return {
            **self._counts(),
            "reply_refused_count": sum(1 for row in rows if row["reply_refused"]),
            "transcription_delay_zero_count": sum(
                1 for row in rows if row["transcription_delay_zero"]
            ),
            "turns_without_delays_count": sum(
                1 for row in rows if row["end_of_turn_delay_ms"] is None
            ),
            "turns_with_held_pauses_count": sum(1 for row in rows if row["held_pauses"]),
            "held_pauses_count": sum(int(row["held_pauses"]) for row in rows),
            "caller_speaking_at_commit_count": sum(
                1 for row in rows if row["caller_speaking_at_commit"]
            ),
            "caller_speech_after_commit_count": sum(
                1 for row in rows if row["caller_speech_after_commit_ms"] is not None
            ),
        }

    # ---------------------------------------------------------------- internal
    def _counts(self) -> dict[str, Any]:
        dropped = self._turn_count - len(self._turns)
        return {
            "turn_count": self._turn_count,
            "turns_recorded_count": len(self._turns),
            "turns_dropped_count": dropped,
            "has_more": dropped > 0,
            "user_items_added_count": self._user_items_added,
            "collector_faults_count": self._faults,
            "detector_observation": "unavailable",
        }

    def _reset_window(self) -> None:
        """Start the next turn's window; a stop before the commit was its end."""

        self._stopped_at = None
        self._held_pauses = 0
        self._longest_held_pause: float | None = None
        self._interims = 0
        self._finals = 0
        self._first_interim_at: float | None = None
        self._last_interim_at: float | None = None
        self._last_final_at: float | None = None

    def _read_clock(self) -> float:
        try:
            return float(self._clock())
        except Exception:  # noqa: BLE001 - a broken clock must not break the call
            return 0.0

    def _now(self, at: float | None) -> float:
        moment = _finite(at)
        return moment if moment is not None else self._read_clock()

    def _offset(self, moment: float | None) -> int | None:
        return None if moment is None else _milliseconds(moment - self._origin)


def write_turn_end_evidence(
    evidence_dir: Path, *, call_id: str, evidence: TurnEndEvidence
) -> dict[str, Any]:
    """Write the record beside the transcript; return the summary to log.

    Never raises: the caller is the teardown of a call that has already ended,
    and a diagnostic that fails to land must say so in the log line it returns
    (``turn_end_evidence_status``) rather than take the rest of the teardown
    with it.  Created
    owner-only, before the first byte, the way the pack builder writes private
    files.
    """

    try:
        summary = evidence.summary()
    except Exception as exc:  # noqa: BLE001
        return {STATUS_FIELD: "summary_failed", "error_type": type(exc).__name__}
    try:
        payload = json.dumps(
            {"call_id": call_id, **evidence.snapshot()}, sort_keys=True, allow_nan=False
        )
        descriptor = os.open(
            Path(evidence_dir) / ARTIFACT_NAME, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except Exception as exc:  # noqa: BLE001
        return {**summary, STATUS_FIELD: "write_failed", "error_type": type(exc).__name__}
    return {**summary, STATUS_FIELD: "written"}

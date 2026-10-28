# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded, provider-neutral call telemetry for the LiveKit media worker.

The collector deliberately owns no network, storage, database, or logging
dependencies.  Live media callbacks may only hand it already-available values;
the worker serializes the completed bundle after media has stopped.
"""
from __future__ import annotations

import json
import math
import os
import string
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping


CALL_TELEMETRY_SCHEMA = "skysay.call_telemetry.v1"
CALL_TELEMETRY_VERSION = 1
DEFAULT_MAX_SPANS = 256
DEFAULT_MAX_PENDING_TURNS = 128
DEFAULT_MAX_TURNS = 128
DEFAULT_MAX_LANGUAGE_SWITCHES = 64
DEFAULT_MAX_OFFSET_MS = 24 * 60 * 60 * 1_000
DEFAULT_MAX_DURATION_MS = 60 * 60 * 1_000
# The commit-offset gap under which two IMMEDIATELY-adjacent trailing pending
# user turns are treated as the SAME final utterance the end_call goodbye
# answers (STT/endpointing can split one utterance into several committed
# turns). A larger gap marks a genuinely-distinct earlier turn the goodbye does
# not answer, which the terminal drain leaves pending so the bundle honestly
# reports partial.
#
# This deliberately errs TIGHT: a timing-based FALSE ``complete`` (certifying a
# genuinely-unanswered earlier turn as answered) corrupts exactly the
# performance evidence this exists to protect, so when the spacing is ambiguous
# the earlier turn stays pending and the call reports partial -- the correct
# output, never a merge on doubt. Grouping by an explicit turn-correlation id
# would be better, but the SDK commits each turn as a fresh ChatMessage with its
# own id, so committed fragments share no id to group by; wall-clock is the only
# signal available.
#
# CALIBRATION-PENDING. Offsets are stamped at ``conversation_item_added`` (turn
# COMMIT, ``on_end_of_turn``, after the endpoint hold), not at the
# ``user_input_transcribed`` finals -- so the ~142ms fragment spacing seen in a
# recorder transcript is a lower bound at the wrong layer. The drain emits the
# observed boundary gap (``terminal_turns_resolved``), so a live end_call call
# yields the real commit-layer spacing to set this against a distinct-turn gap.
# Until then it stays tight: a genuine fragment committed farther apart than
# this reports partial (honest), never a distinct turn certified complete.
_TERMINAL_FRAGMENT_GAP_MS = 500
_SAFE_LABEL_CHARACTERS = frozenset(string.ascii_letters + string.digits + "._:-/")


def _finite_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _milliseconds(value: float | None) -> int | None:
    return max(0, int(round(value * 1_000))) if value is not None else None


def _safe_label(value: Any, *, limit: int = 96) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    filtered = "".join(character for character in text if character in _SAFE_LABEL_CHARACTERS)
    # Match the normalizer's public-label grammar exactly: optional labels may
    # be empty, while non-empty labels must begin with an ASCII alphanumeric.
    return filtered.lstrip("._:-/")[:limit]


def _character_count(value: Any) -> int:
    # Text already arrives in memory from the LiveKit event.  Count it without
    # normalizing/copying an arbitrarily long utterance in a media callback.
    # The exact whitespace-normalized length is not meaningful telemetry; the
    # bounded character bucket is.
    try:
        return min(16_384, len(value) if isinstance(value, str) else len(str(value or "")))
    except Exception:
        return 0


class CallTelemetryCollector:
    """Collect one bounded v1 telemetry bundle without retaining call text.

    ``observe_*`` is intentionally best-effort.  It never performs I/O and
    suppresses malformed SDK metrics rather than letting observability break a
    call.  The mutable details stay private; ``bundle`` returns only the
    customer-safe, bounded artifact contract.
    """

    # Private SDK metadata, not part of the exported telemetry artifact.
    REPLY_USER_ITEM_KEY = "skysay_reply_user_item_id"

    def __init__(
        self,
        *,
        call_id: str,
        organization_id: str,
        project_id: str,
        max_spans: int = DEFAULT_MAX_SPANS,
        max_pending_turns: int = DEFAULT_MAX_PENDING_TURNS,
        max_turns: int = DEFAULT_MAX_TURNS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._origin = float(clock())
        self._call_id = _safe_label(call_id, limit=160)
        self._organization_id = _safe_label(organization_id, limit=160)
        self._project_id = _safe_label(project_id, limit=160)
        self._max_spans = max(1, min(int(max_spans or DEFAULT_MAX_SPANS), DEFAULT_MAX_SPANS))
        self._max_pending_turns = max(1, min(int(max_pending_turns or DEFAULT_MAX_PENDING_TURNS), DEFAULT_MAX_PENDING_TURNS))
        self._max_turns = max(1, min(int(max_turns or DEFAULT_MAX_TURNS), DEFAULT_MAX_TURNS))
        self._lock = RLock()
        self._pending_users: list[dict[str, Any]] = []
        self._spans: list[dict[str, Any]] = []
        self._stage_attempts: dict[tuple[int, str], int] = {}
        self._next_turn_index = 0
        self._dropped_span_count = 0
        self._partial_reasons: list[str] = []
        # This is deliberately separate from generic timing/turn telemetry.
        # A call can have partial latency evidence yet still have a complete
        # language-switch timeline, and vice versa.
        self._language_switches: list[dict[str, Any]] = []
        self._language_evidence_partial_reasons: list[str] = []

    def observe_language_switch(self, event: Any, **details: Any) -> None:
        """Capture only terminal, customer-safe switch outcomes.

        The LiveKit worker still writes its operational log events, including
        ``queued``.  This final artifact is different evidence: it records
        only a transition that applied, was rejected, or failed.  In
        particular, a scheduled request is never represented as an applied
        response-language change.
        """
        outcomes = {
            "language_switch_applied": "applied",
            "language_switch_rejected": "rejected",
            "language_switch_failed": "failed",
        }
        outcome = outcomes.get(str(event or ""))
        if outcome is None:
            return
        try:
            source = str(details.get("source") or "").strip()
            if source not in {"explicit_request", "provider_stable_evidence", "workflow_interrupt"}:
                self._mark_language_evidence_partial("collector_callback_error")
                return
            language = _safe_label(details.get("language"), limit=32).casefold()
            old_language = _safe_label(details.get("old_language"), limit=32).casefold()
            publication_version = details.get("publication_version")
            if (
                not language
                or not language.replace("-", "").isalnum()
                or (old_language and not old_language.replace("-", "").isalnum())
                or isinstance(publication_version, bool)
                or not isinstance(publication_version, int)
                or publication_version < 1
            ):
                self._mark_language_evidence_partial("collector_callback_error")
                return
            # Never persist an arbitrary model-authored reason. These values
            # are an intentionally closed customer-safe enum.
            raw_reason = str(details.get("reason") or "")
            reason_code = ""
            if outcome == "rejected":
                reason_code = (
                    "segment_limit_reached" if "segment limit" in raw_reason
                    else "unsupported_language" if "not in the frozen allowed profiles" in raw_reason
                    else "terminal" if "terminal" in raw_reason
                    else "invalid_request"
                )
            elif outcome == "failed":
                reason_code = "runtime_error"
            with self._lock:
                if len(self._language_switches) >= DEFAULT_MAX_LANGUAGE_SWITCHES:
                    # Already holding the collector lock. Keep this inline so
                    # the limit path remains safe even if a future collector
                    # swaps the current RLock for a non-reentrant primitive.
                    if "event_limit_reached" not in self._language_evidence_partial_reasons:
                        self._language_evidence_partial_reasons.append("event_limit_reached")
                    return
                sequence = len(self._language_switches) + 1
                self._language_switches.append(
                    {
                        "event_id": f"language-switch:{sequence}",
                        "sequence": sequence,
                        "old_language": old_language,
                        "language": language,
                        "source": source,
                        "outcome": outcome,
                        "reason_code": reason_code,
                        "publication_version": publication_version,
                        # Worker-monotonic call clock only. It is deliberately
                        # NOT a recording seek time: no origin mapping exists.
                        "call_offset_ms": self._offset_ms(),
                    }
                )
        except Exception:
            self._mark_language_evidence_partial("collector_callback_error")

    @property
    def current_turn_index(self) -> int:
        with self._lock:
            return self._next_turn_index

    def observe_user(self, *, text: Any, metrics: Mapping[str, Any] | None, item_id: str = "") -> None:
        """Remember only countable user metrics for one future semantic turn."""
        try:
            with self._lock:
                if self._next_turn_index >= self._max_turns:
                    self._mark_partial("turn_limit_reached")
                    return
                self._next_turn_index += 1
                if len(self._pending_users) >= self._max_pending_turns:
                    self._mark_partial("pending_turn_limit_reached")
                    return
                self._pending_users.append(
                    {
                        "turn_index": self._next_turn_index,
                        "item_id": item_id if isinstance(item_id, str) and len(item_id) <= 160 else "",
                        "offset_ms": self._offset_ms(),
                        "input_characters": _character_count(text),
                        "metrics": self._livekit_metrics(metrics, ("transcription_delay", "end_of_turn_delay")),
                    }
                )
        except Exception:  # telemetry must never break a media callback
            self._mark_partial("collector_callback_error")

    def observe_assistant(
        self,
        *,
        text: Any,
        metrics: Mapping[str, Any] | None,
        provider_labels: Mapping[str, Any] | None = None,
        model_labels: Mapping[str, Any] | None = None,
        playout_reason_code: Any = "",
        user_item_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Pair a cascade reply by its frozen input ID, not completion order.

        None retains the native/workflow legacy path. An explicitly empty or
        unknown ID MUST NOT fall back to FIFO and steal another caller's turn.
        """
        try:
            with self._lock:
                user = next((row for row in self._pending_users
                             if user_item_id is None or (user_item_id and row["item_id"] == user_item_id)), None)
                if user is None:
                    self._mark_partial("unpaired_assistant_turn")
                    return None
                self._pending_users.remove(user)
                assistant_metrics = self._livekit_metrics(
                    metrics,
                    ("llm_node_ttft", "tts_node_ttfb", "playback_latency", "e2e_latency"),
                )
                providers = provider_labels if isinstance(provider_labels, Mapping) else {}
                models = model_labels if isinstance(model_labels, Mapping) else {}
                output_characters = _character_count(text)
                turn_index = int(user["turn_index"])
                offset_ms = int(user["offset_ms"])
                common = {
                    "turn_index": turn_index,
                    "attempt": 1,
                    "start_offset_ms": offset_ms,
                    "outcome": "success",
                    "retry": False,
                    "fallback": False,
                    "input_characters": int(user["input_characters"]),
                    "output_characters": output_characters,
                    "measurement_source": "livekit_chat_message_metrics",
                    "confidence": "measured",
                    "reason_code": "",
                    "audit_correlation_id": "",
                    "tool_name": "",
                }
                # Every stage below is bounded by a clock this worker owns, and
                # that is the whole list -- there is deliberately no
                # caller-side or carrier stage here.
                #
                # A worker cannot measure what a caller waited. Inbound audio
                # has already crossed the carrier before this process sees it,
                # and outbound audio has not crossed it yet when this process
                # emits it, so both endpoints of any span assembled here sit on
                # the same side of the PSTN leg. That is also why the room-side
                # WAV recordings cannot supply it: they are captured at these
                # same two points, so an acoustic gap measured between them is
                # a worker-side number too, not a caller-perceived one.
                #
                # Closing that gap needs a capture at the far end of the call
                # (a reference handset recording, correlated by call id), not
                # another field on this collector. An earlier parallel
                # recorder accepted a ``caller_side_voice_to_voice_ms``
                # argument nothing could ever supply, and its unfillable null
                # was later read as evidence of a ~2.8s carrier delay that no
                # instrument had measured. Do not add the field back without
                # the far-end instrument that populates it.
                # Model is recorded per stage, and only on the three stages one
                # model actually owns. The livekit stages below stay blank on
                # purpose: ``response.internal_total`` spans a whole turn that
                # every component contributed to, so naming any single model on
                # it would invite exactly the false attribution this field
                # exists to prevent.
                # ``playout.first_worker`` is the only stage that carries a
                # reason: it spans the interval between synthesized audio
                # existing and its first frame reaching the track, and the
                # runtime withholds exactly that audio while it yields the
                # floor to a talking far end. Without the label a floor yield
                # and a platform stall are the same opaque number, and telling
                # them apart once cost a night of packet-level forensics.
                expected_metrics = (
                    ("stt.final", user["metrics"].get("transcription_delay"), providers.get("stt"), models.get("stt"), ""),
                    ("endpoint.turn_commit", user["metrics"].get("end_of_turn_delay"), "livekit", "", ""),
                    ("llm.first_token", assistant_metrics.get("llm_node_ttft"), providers.get("llm"), models.get("llm"), ""),
                    ("tts.first_audio", assistant_metrics.get("tts_node_ttfb"), providers.get("tts"), models.get("tts"), ""),
                    ("playout.first_worker", assistant_metrics.get("playback_latency"), "livekit", "", playout_reason_code),
                    ("response.internal_total", assistant_metrics.get("e2e_latency"), "livekit", "", ""),
                )
                for stage, seconds, provider, model, reason_code in expected_metrics:
                    if not self._append_metric_span(
                        common,
                        stage=stage,
                        seconds=seconds,
                        provider=provider,
                        model=model,
                        reason_code=reason_code,
                    ):
                        self._mark_partial("missing_stage_measurement")
                return {"turn_index": turn_index, "span_count": len(self._spans)}
        except Exception:  # telemetry must never break a media callback
            self._mark_partial("collector_callback_error")
            return None

    def _terminal_pending_cluster(self) -> tuple[list[dict[str, Any]], int]:
        """Caller holds the lock; return the existing tight trailing cluster."""
        pending = self._pending_users
        start = len(pending) - 1
        boundary_gap_ms = -1
        while start > 0:
            gap = int(pending[start]["offset_ms"]) - int(pending[start - 1]["offset_ms"])
            if gap > _TERMINAL_FRAGMENT_GAP_MS:
                boundary_gap_ms = gap
                break
            start -= 1
        return pending[max(0, start):], boundary_gap_ms

    def snapshot_terminal_turns(self, *, user_item_id: str | None = None) -> tuple[int, ...]:
        """Freeze a proposal's pending caller identities before playout yields.

        No turn is consumed here. A model preamble may still resolve one, and
        caller speech arriving during the farewell must never join this set.
        """
        with self._lock:
            if user_item_id is not None:
                # A model tool owns the caller item in its frozen input, not
                # whatever happened to become the latest pending item while
                # it was running. No timing-based fragment guess on this path.
                return tuple(int(user["turn_index"]) for user in self._pending_users
                             if user_item_id and user["item_id"] == user_item_id)
            users, _ = self._terminal_pending_cluster()
            return tuple(int(user["turn_index"]) for user in users)

    def observe_terminal_assistant(
        self,
        *,
        provider_labels: Mapping[str, Any] | None = None,
        goodbye_playout_start: float | None = None,
        cancelled_turns: tuple[int, ...] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve the final caller utterance with the worker-owned end_call goodbye.

        The farewell is spoken by the worker via ``session.say`` without a
        chat-context item, so it never arrives as an assistant
        conversation item and ``observe_assistant`` never runs for it.  Without
        this, the resolving caller turn finalizes ``unmatched_user_turn`` and the
        whole bundle reads ``partial`` for a healthy call -- the exact loss the
        workspace Performance view and the nightly gates cannot see.

        The goodbye answers the caller's FINAL utterance only.  STT/endpointing
        can commit one utterance as several user turns, so the trailing run whose
        consecutive commit offsets sit within ``_TERMINAL_FRAGMENT_GAP_MS`` is
        resolved together.  A pending turn separated by a larger gap is an
        earlier turn the goodbye does not answer; it is deliberately LEFT pending
        so the bundle honestly reports partial rather than certifying a
        genuinely-unanswered turn as complete (the transcript still surfaces it
        as degraded/unmatched).

        Only the caller-side stages this collector already measured (``stt.final``
        and ``endpoint.turn_commit``) are emitted; a worker goodbye carries no
        model ``llm``/``tts`` turn metrics and those are not emitted at all.  But
        a resolved turn MISSING one of its two caller-side measurements is marked
        ``missing_stage_measurement`` exactly as a normal turn is -- a missing
        metric is missing evidence, distinct from a goodbye honestly recorded.
        A call that ended before any caller turn (end_call on the greeting) has
        nothing pending and is left untouched.

        When ``goodbye_playout_start`` is given (the goodbye's observed first
        frame), one DERIVED ``response.internal_total`` span is emitted for the
        resolving turn -- from the final utterance's commit to that playout-start
        -- so the Performance view shows a real reply-latency for the terminal
        turn instead of a hollow ``complete``. No model llm/tts/e2e stages are
        fabricated; the span is skipped entirely when the goodbye produced no
        audio.

        Returns the resolved/kept counts and the boundary gap (``-1`` when none)
        so the worker can emit them; a live end_call call then yields the datum
        that calibrates ``_TERMINAL_FRAGMENT_GAP_MS``.

        A reclaimed proposal supplies its frozen ``cancelled_turns`` identities.
        Only those still pending are resolved, with cancelled outcome, no
        fabricated response latency, and no claim of completed farewell audio.
        A later caller question and any older unrelated pending turn are kept.
        """

        try:
            with self._lock:
                if not self._pending_users:
                    return None
                providers = (
                    provider_labels if isinstance(provider_labels, Mapping) else {}
                )
                pending = self._pending_users
                cancelled = cancelled_turns is not None
                if cancelled:
                    identities = set(cancelled_turns)
                    resolved_users = [user for user in pending if user["turn_index"] in identities]
                    boundary_gap_ms = -1
                else:
                    resolved_users, boundary_gap_ms = self._terminal_pending_cluster()
                    identities = {user["turn_index"] for user in resolved_users}
                pending[:] = [user for user in pending if user["turn_index"] not in identities]
                outcome = "cancelled" if cancelled else "success"
                source = "worker_reclaimed_goodbye" if cancelled else "worker_terminal_goodbye"
                reason = "caller_took_floor" if cancelled else ""
                resolved = 0
                for user in resolved_users:
                    common = {
                        "turn_index": int(user["turn_index"]),
                        "attempt": 1,
                        "start_offset_ms": int(user["offset_ms"]),
                        "outcome": outcome,
                        "retry": False,
                        "fallback": False,
                        "input_characters": int(user["input_characters"]),
                        "output_characters": 0,
                        "measurement_source": source,
                        "confidence": "measured",
                        "reason_code": reason,
                        "audit_correlation_id": "",
                        "tool_name": "",
                    }
                    for stage, seconds, provider in (
                        ("stt.final", user["metrics"].get("transcription_delay"), providers.get("stt")),
                        ("endpoint.turn_commit", user["metrics"].get("end_of_turn_delay"), "livekit"),
                    ):
                        if not self._append_metric_span(
                            common,
                            stage=stage,
                            seconds=seconds,
                            provider=provider,
                            model="",
                            reason_code=reason,
                        ):
                            # A caller-side stage the runtime never reported is
                            # missing evidence, exactly as for a normal turn.
                            self._mark_partial("missing_stage_measurement")
                    resolved += 1
                # One DERIVED reply-latency span for the resolving turn: from the
                # final utterance's commit to when the goodbye actually started
                # playing. It is what the Performance view shows as the terminal
                # turn's response time -- the latency the Observed section names.
                # No model llm/tts/e2e stages are fabricated; skipped entirely
                # when the goodbye produced no audio (no playout-start).
                if not cancelled and resolved_users and goodbye_playout_start is not None:
                    final_user = resolved_users[-1]
                    playout_offset_ms = int(
                        round((float(goodbye_playout_start) - self._origin) * 1_000)
                    )
                    reply_ms = playout_offset_ms - int(final_user["offset_ms"])
                    if 0 <= reply_ms <= DEFAULT_MAX_DURATION_MS:
                        self._append_span(
                            {
                                "turn_index": int(final_user["turn_index"]),
                                "stage": "response.internal_total",
                                "attempt": 1,
                                "start_offset_ms": int(final_user["offset_ms"]),
                                "duration_ms": reply_ms,
                                "outcome": "success",
                                "provider": "skysay",
                                "model": "",
                                "tool_name": "",
                                "retry": False,
                                "fallback": False,
                                "input_characters": int(final_user["input_characters"]),
                                "output_characters": 0,
                                "measurement_source": "worker_terminal_goodbye",
                                "confidence": "derived",
                                "reason_code": "",
                                "audit_correlation_id": "",
                            }
                        )
                return {
                    "resolved_pending_turns": resolved,
                    "kept_pending_turns": len(pending),
                    "boundary_gap_ms": boundary_gap_ms,
                }
        except Exception:  # telemetry must never break teardown
            self._mark_partial("collector_callback_error")
            return None

    def observe_tool(
        self,
        *,
        name: Any,
        started_at: float,
        outcome: str,
        audit_correlation_id: Any = "",
        input_characters: int = 0,
        reason_code: Any = "",
    ) -> None:
        """Record one post-call-safe tool or knowledge duration.

        This is called by the existing tool-control-plane path, never by the
        hot media SDK event handler.  The caller passes a monotonic start that
        it already captured around the tool request.
        """
        try:
            with self._lock:
                tool_name = _safe_label(name)
                stage = "knowledge.total" if tool_name == "search_knowledge_base" else "tool.invocation"
                now = float(self._clock())
                duration_ms = max(0, int(round((now - float(started_at)) * 1_000)))
                start_offset_ms = max(0, int(round((float(started_at) - self._origin) * 1_000)))
                if duration_ms > DEFAULT_MAX_DURATION_MS or start_offset_ms > DEFAULT_MAX_OFFSET_MS:
                    self._mark_partial("timing_limit_reached")
                self._append_span(
                    {
                        "turn_index": max(0, self._next_turn_index),
                        "stage": stage,
                        "attempt": 1,
                        "start_offset_ms": min(DEFAULT_MAX_OFFSET_MS, start_offset_ms),
                        "duration_ms": min(DEFAULT_MAX_DURATION_MS, duration_ms),
                        "outcome": outcome if outcome in {"success", "empty", "cancelled", "timeout", "error", "fallback"} else "error",
                        "provider": "skysay",
                        "model": "",
                        "tool_name": tool_name,
                        "retry": False,
                        "fallback": False,
                        "input_characters": max(0, min(16_384, int(input_characters or 0))),
                        "output_characters": 0,
                        "measurement_source": "control_plane_tool_request",
                        "confidence": "measured",
                        "reason_code": _safe_label(reason_code),
                        "audit_correlation_id": _safe_label(audit_correlation_id, limit=160),
                    }
                )
        except Exception:  # telemetry must never break tool execution either
            self._mark_partial("collector_callback_error")

    def observe_workflow_turn(
        self,
        *,
        started_at: float,
        timing: Mapping[str, Any] | None,
        decision_provider: Any = "",
        decision_model: Any = "",
    ) -> None:
        """Record where one workflow turn's round trip went.

        The control plane measured its own split and returned it with the
        turn response (``load_ms``, ``decide_ms``, ``compose_ms``,
        ``total_ms``); this worker measured the whole round trip. The
        difference is the transport, so the four stages together are the
        ``llm.first_token`` the workflow path reports. Called from the
        thread that made the request, never from a media callback.
        """
        try:
            with self._lock:
                now = float(self._clock())
                round_trip_ms = max(0, int(round((now - float(started_at)) * 1_000)))
                start_offset_ms = max(0, int(round((float(started_at) - self._origin) * 1_000)))
                split = timing if isinstance(timing, Mapping) else {}

                def integer(key: str) -> int:
                    value = split.get(key)
                    return max(0, int(round(float(value)))) if isinstance(value, (int, float)) else 0

                control_plane_ms = integer("total_ms")
                stages = (
                    ("workflow.state_load", integer("load_ms"), "skysay", ""),
                    ("workflow.decision", integer("decide_ms"), _safe_label(decision_provider) or "skysay", _safe_label(decision_model)),
                    ("workflow.reply_compose", integer("compose_ms"), "skysay", ""),
                    ("workflow.transport", max(0, round_trip_ms - control_plane_ms), "skysay", ""),
                )
                if round_trip_ms > DEFAULT_MAX_DURATION_MS or start_offset_ms > DEFAULT_MAX_OFFSET_MS:
                    self._mark_partial("timing_limit_reached")
                for stage, duration_ms, provider, model in stages:
                    self._append_span(
                        {
                            "turn_index": max(0, self._next_turn_index),
                            "stage": stage,
                            "attempt": 1,
                            "start_offset_ms": min(DEFAULT_MAX_OFFSET_MS, start_offset_ms),
                            "duration_ms": min(DEFAULT_MAX_DURATION_MS, duration_ms),
                            "outcome": "success",
                            "provider": provider,
                            "model": model,
                            "tool_name": "",
                            "retry": False,
                            "fallback": False,
                            "input_characters": 0,
                            "output_characters": 0,
                            "measurement_source": "control_plane_turn_timing",
                            "confidence": "measured",
                            "reason_code": "",
                            "audit_correlation_id": "",
                        }
                    )
        except Exception:  # telemetry must never break a turn
            self._mark_partial("collector_callback_error")

    def bundle(self) -> dict[str, Any]:
        """Return a JSON-safe immutable v1 artifact projection."""
        try:
            with self._lock:
                if self._dropped_span_count:
                    self._mark_partial("span_limit_reached")
                if self._pending_users:
                    self._mark_partial("unmatched_user_turn")
                if not self._spans:
                    self._mark_partial("no_semantic_turns")
                reason = self._evidence_reason()
                status = "partial" if reason else "complete"
                return {
                    "schema": CALL_TELEMETRY_SCHEMA,
                    "version": CALL_TELEMETRY_VERSION,
                    "call_id": self._call_id,
                    "organization_id": self._organization_id,
                    "project_id": self._project_id,
                    "evidence": {"status": status, "reason": reason},
                    "turn_count": self._next_turn_index,
                    "span_count": len(self._spans),
                    "dropped_span_count": self._dropped_span_count,
                    "spans": [dict(span) for span in self._spans],
                    "language_switch_evidence": {
                        "status": "partial" if self._language_evidence_reason() else "complete",
                        "reason": self._language_evidence_reason(),
                        "events": [dict(event) for event in self._language_switches],
                    },
                }
        except Exception:  # final artifact construction must not obscure cleanup
            return {
                "schema": CALL_TELEMETRY_SCHEMA,
                "version": CALL_TELEMETRY_VERSION,
                "call_id": self._call_id,
                "organization_id": self._organization_id,
                "project_id": self._project_id,
                "evidence": {"status": "partial", "reason": "collector_finalize_error"},
                "turn_count": 0,
                "span_count": 0,
                "dropped_span_count": 0,
                "spans": [],
                "language_switch_evidence": {
                    "status": "partial",
                    "reason": "collector_callback_error",
                    "events": [],
                },
            }

    def write(self, path: Path) -> dict[str, Any]:
        """Atomically write the canonical artifact after media has ended."""
        payload = self.bundle()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return payload

    def _append_metric_span(
        self,
        common: dict[str, Any],
        *,
        stage: str,
        seconds: Any,
        provider: Any,
        model: Any = "",
        reason_code: Any = "",
    ) -> bool:
        duration_ms = _milliseconds(seconds if isinstance(seconds, (int, float)) else None)
        if duration_ms is None:
            return False
        if duration_ms > DEFAULT_MAX_DURATION_MS:
            self._mark_partial("timing_limit_reached")
            duration_ms = DEFAULT_MAX_DURATION_MS
        self._append_span(
            {
                **common,
                "stage": stage,
                "duration_ms": duration_ms,
                "provider": _safe_label(provider),
                "model": _safe_label(model),
                "reason_code": _safe_label(reason_code),
            }
        )
        return True

    def _append_span(self, span: dict[str, Any]) -> None:
        with self._lock:
            if len(self._spans) >= self._max_spans:
                self._dropped_span_count = min(DEFAULT_MAX_SPANS, self._dropped_span_count + 1)
                return
            key = (int(span["turn_index"]), str(span["stage"]))
            attempt = self._stage_attempts.get(key, 0) + 1
            self._stage_attempts[key] = attempt
            span["attempt"] = attempt
            self._spans.append(span)

    def _offset_ms(self) -> int:
        offset_ms = max(0, int(round((float(self._clock()) - self._origin) * 1_000)))
        if offset_ms > DEFAULT_MAX_OFFSET_MS:
            self._mark_partial("timing_limit_reached")
        return min(DEFAULT_MAX_OFFSET_MS, offset_ms)

    def _mark_partial(self, reason: str) -> None:
        with self._lock:
            if reason and reason not in self._partial_reasons:
                self._partial_reasons.append(reason)

    def _evidence_reason(self) -> str:
        """Choose one stable primary reason when bounded evidence is partial."""
        for reason in (
            "span_limit_reached",
            "turn_limit_reached",
            "pending_turn_limit_reached",
            "unmatched_user_turn",
            "unpaired_assistant_turn",
            "missing_stage_measurement",
            "no_semantic_turns",
            "collector_callback_error",
            "timing_limit_reached",
        ):
            if reason in self._partial_reasons:
                return reason
        return self._partial_reasons[0] if self._partial_reasons else ""

    def _mark_language_evidence_partial(self, reason: str) -> None:
        with self._lock:
            if reason and reason not in self._language_evidence_partial_reasons:
                self._language_evidence_partial_reasons.append(reason)

    def _language_evidence_reason(self) -> str:
        for reason in ("event_limit_reached", "collector_callback_error"):
            if reason in self._language_evidence_partial_reasons:
                return reason
        return self._language_evidence_partial_reasons[0] if self._language_evidence_partial_reasons else ""

    @staticmethod
    def _livekit_metrics(metrics: Mapping[str, Any] | None, fields: tuple[str, ...]) -> dict[str, float]:
        source = metrics if isinstance(metrics, Mapping) else {}
        return {
            field: value
            for field in fields
            if (value := _finite_seconds(source.get(field))) is not None
        }


__all__ = [
    "CALL_TELEMETRY_SCHEMA",
    "CALL_TELEMETRY_VERSION",
    "CallTelemetryCollector",
]

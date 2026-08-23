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


CALL_TELEMETRY_SCHEMA = "openphonex.call_telemetry.v1"
CALL_TELEMETRY_VERSION = 1
DEFAULT_MAX_SPANS = 256
DEFAULT_MAX_PENDING_TURNS = 128
DEFAULT_MAX_TURNS = 128
DEFAULT_MAX_OFFSET_MS = 24 * 60 * 60 * 1_000
DEFAULT_MAX_DURATION_MS = 60 * 60 * 1_000
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

    @property
    def current_turn_index(self) -> int:
        with self._lock:
            return self._next_turn_index

    def observe_user(self, *, text: Any, metrics: Mapping[str, Any] | None) -> None:
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
    ) -> dict[str, Any] | None:
        """Pair the next final assistant item with a previously seen caller item."""
        try:
            with self._lock:
                if not self._pending_users:
                    self._mark_partial("unpaired_assistant_turn")
                    return None
                user = self._pending_users.pop(0)
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
                expected_metrics = (
                    ("stt.final", user["metrics"].get("transcription_delay"), providers.get("stt"), models.get("stt")),
                    ("endpoint.turn_commit", user["metrics"].get("end_of_turn_delay"), "livekit", ""),
                    ("llm.first_token", assistant_metrics.get("llm_node_ttft"), providers.get("llm"), models.get("llm")),
                    ("tts.first_audio", assistant_metrics.get("tts_node_ttfb"), providers.get("tts"), models.get("tts")),
                    ("playout.first_worker", assistant_metrics.get("playback_latency"), "livekit", ""),
                    ("response.internal_total", assistant_metrics.get("e2e_latency"), "livekit", ""),
                )
                for stage, seconds, provider, model in expected_metrics:
                    if not self._append_metric_span(
                        common, stage=stage, seconds=seconds, provider=provider, model=model
                    ):
                        self._mark_partial("missing_stage_measurement")
                return {"turn_index": turn_index, "span_count": len(self._spans)}
        except Exception:  # telemetry must never break a media callback
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
                        "provider": "openphonex",
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
        self, common: dict[str, Any], *, stage: str, seconds: Any, provider: Any, model: Any = ""
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

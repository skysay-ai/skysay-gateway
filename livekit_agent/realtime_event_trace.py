# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded, opt-in event-shape diagnostics; never a transcript or a turn writer.

Measure the actual SDK event stream before choosing persistence
semantics. No speech, provider payload, raw source identifier or text digest is
exported. Local item references correlate events only within this trace and are
NOT durable idempotency keys. Callback ingestion offsets share the recording's
monotonic zero; SDK creation timestamps remain separately named wall time.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from typing import Any, Callable

EVENTS = ("user_input_transcribed", "conversation_item_added", "agent_state_changed", "speech_created")
MAX_EVENTS = 512
MAX_ITEMS = 128


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


class RealtimeEventTrace:
    def __init__(self, *, relative_ms: Callable[[float], int], enabled: bool = False,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = enabled
        self._relative_ms = relative_ms
        self._clock = clock
        self._rows: list[dict[str, Any]] = []
        self._items: dict[str, tuple[int, bytes | None]] = {}
        self._counts: Counter[str] = Counter()
        self._dropped = 0
        self._errors = 0

    def observe(self, kind: str, event: Any) -> None:
        if not self.enabled or kind not in EVENTS:
            return
        self._counts[kind] += 1
        if len(self._rows) >= MAX_EVENTS:
            self._dropped += 1
            return
        try:
            item = getattr(event, "item", None)
            source_id = getattr(event, "item_id", None) if kind == "user_input_transcribed" else getattr(item, "id", None)
            text = getattr(event, "transcript", None) if kind == "user_input_transcribed" else getattr(item, "text_content", None)
            role = "user" if kind == "user_input_transcribed" else getattr(item, "role", None)
            row: dict[str, Any] = {
                "sequence": len(self._rows) + 1,
                "event": kind,
                "ingested_ms": _number(self._relative_ms(self._clock())),
                "sdk_created_at_unix_seconds": _number(getattr(event, "created_at", None)),
                "role": role if role in ("user", "assistant") else None,
                "is_final": getattr(event, "is_final", None) if isinstance(getattr(event, "is_final", None), bool) else None,
                "interrupted": getattr(item, "interrupted", None) if isinstance(getattr(item, "interrupted", None), bool) else None,
                "text_characters": len(text) if isinstance(text, str) else None,
                "item_reference": None,
                "text_changed_since_previous": None,
                "rendered_to_far_end": "unproven",
            }
            if isinstance(source_id, str) and 0 < len(source_id) <= 256:
                previous = self._items.get(source_id)
                if previous is not None or len(self._items) < MAX_ITEMS:
                    digest = hashlib.sha256(text.encode()).digest() if isinstance(text, str) and len(text) <= 65536 else None
                    reference = previous[0] if previous else len(self._items) + 1
                    row["item_reference"] = reference
                    if previous and previous[1] is not None and digest is not None:
                        row["text_changed_since_previous"] = previous[1] != digest
                    self._items[source_id] = (reference, digest)
            if kind == "agent_state_changed":
                for key in ("old_state", "new_state"):
                    value = getattr(event, key, None)
                    row[key] = value if value in ("initializing", "idle", "listening", "thinking", "speaking") else None
            self._rows.append(row)
        except Exception:
            # Instrumentation cannot interrupt a live callback.
            self._errors += 1

    def snapshot(self) -> dict[str, Any]:
        return {"version": 1, "enabled": self.enabled,
                "purpose": "event_matrix_only_not_transcript", "clock": "recording_monotonic_zero",
                "item_reference_scope": "this_trace_only_not_provider_event_identity",
                "events": [dict(row) for row in self._rows], "counts": dict(self._counts),
                "returned": len(self._rows), "dropped": self._dropped,
                "has_more": self._dropped > 0, "observation_errors": self._errors,
                "source_identifiers_retained": len(self._items), "source_identifier_limit": MAX_ITEMS}


def install_realtime_event_trace(session: Any, trace: RealtimeEventTrace) -> None:
    if not trace.enabled:
        return
    for kind in EVENTS:
        session.on(kind, lambda event, kind=kind: trace.observe(kind, event))

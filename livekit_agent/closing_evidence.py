# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded observations of the existing close owner; never a closing policy."""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Callable
import json
import re

SCHEMA = "skysay.call-closing.v1"
KINDS = frozenset({"request", "accepted", "joined", "refused", "failed", "already_finished",
                   "reclaimed", "owner_commit", "owner_changed", "provider_close_sent", "media_ended"})
SOURCES = frozenset({"close_owner", "native_provider", "room", "participant"})
OWNERS = frozenset({"model", "platform", "remote", "transport"})
REASONS = frozenset({"", "caller_speech", "guard", "navigation", "transfer", "interrupted",
                     "runtime_error", "already_finished", "platform_rule", "media_disconnect"})


class ClosingEvidence:
    """First 64 + last 128, with independently reserved final witnesses.

    No request map, text, arguments, exceptions or provider IDs. The callback
    only records observations and cannot delay the close with IO or awaits.
    """
    def __init__(self, *, call_id: str, relative_ms: Callable[[], int]) -> None:
        self.call_id = call_id
        self.clock = relative_ms
        self.head: list[dict[str, Any]] = []
        self.tail: deque[dict[str, Any]] = deque()
        self.head_bytes = self.tail_bytes = 0
        self.counts: Counter[str] = Counter()
        self.dropped: Counter[str] = Counter()
        self.seq = 0
        self.commit: dict[str, Any] | None = None
        self.terminals: dict[str, dict[str, Any]] = {}
        self.conflicts = 0
        self.capture_errors = 0
        self.external_endings = 0
        self.responded_requests = 0

    def record(self, kind: str, *, request_id: str = "", proposal_id: str = "",
               owner: str = "model", reason: str = "", source: str = "close_owner") -> None:
        try:
            if kind not in KINDS or owner not in OWNERS or reason not in REASONS:
                raise ValueError
            if source not in SOURCES:
                raise ValueError
            for value, prefix in ((request_id, "r"), (proposal_id, "p")):
                if not isinstance(value, str) or len(value) > 96 or (value and not re.fullmatch(prefix + r"[1-9][0-9]*", value, flags=re.ASCII)):
                    raise ValueError
            self.seq += 1
            try:
                at_ms = self.clock()
                if type(at_ms) is not int or not 0 <= at_ms <= 2**53 - 1:
                    at_ms = None
            except Exception:
                at_ms = None
            row = dict(event_id=f"close-{self.seq}", seq=self.seq, kind=kind, source=source,
                       request_id=request_id, proposal_id=proposal_id, owner=owner, reason=reason,
                       at_ms=at_ms, offset_precision="exact" if at_ms is not None else "unavailable",
                       offset_source="recording_monotonic" if at_ms is not None else "unavailable")
            self.counts[kind] += 1
            if request_id and kind in {"accepted", "joined", "refused", "failed", "already_finished"}:
                self.responded_requests += 1
            if kind == "owner_changed" and owner == "transport" and reason == "already_finished":
                self.external_endings += 1
            if kind == "owner_commit":
                if self.commit is None:
                    self.commit = row
                elif self.commit["proposal_id"] != proposal_id:
                    self.conflicts += 1
            if kind == "media_ended":
                if source not in self.terminals:
                    self.terminals[source] = row
                else:
                    self.conflicts += 1
            size = len(json.dumps(row, separators=(",", ":")).encode())
            if len(self.head) < 64 and self.head_bytes + size <= 16 * 1024:
                self.head.append(row)
                self.head_bytes += size
            else:
                while self.tail and (len(self.tail) >= 128 or self.tail_bytes + size > 32 * 1024):
                    removed = self.tail.popleft()
                    self.tail_bytes -= len(json.dumps(removed, separators=(",", ":")).encode())
                    self.dropped[removed["kind"]] += 1
                self.tail.append(row)
                self.tail_bytes += size
        except Exception:
            self.capture_errors += 1

    def observe_native(self, event: Any) -> None:
        try:
            if not isinstance(event, dict):
                self.capture_errors += 1
            elif event.get("type") == "session.close":
                self.record("provider_close_sent", source="native_provider", owner="transport")
        except Exception:
            self.capture_errors += 1

    def section(self) -> dict[str, Any]:
        # Called only by artifact finalization, after the owner has settled.
        return dict(schema=SCHEMA, call_id=self.call_id, revision=self.seq,
                    completeness="partial" if (self.dropped or self.conflicts or self.capture_errors or
                        self.counts["request"] > self.responded_requests or
                        self.counts["accepted"] > self.counts["reclaimed"] + self.counts["owner_commit"] + self.external_endings) else "complete",
                    capture_finished=True, counts=dict(self.counts), dropped=dict(self.dropped),
                    conflicts=self.conflicts, capture_errors=self.capture_errors,
                    events=[*self.head, *self.tail], final_commit=self.commit,
                    terminal_witnesses=list(self.terminals.values()))

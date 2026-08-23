# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Crash-safe reconciliation for LiveKit media sessions.

The worker persists only opaque call/room identifiers and lifecycle times.
After a restart, the reconciler closes sessions whose room is no longer
active.  Billing is finalized at the last gateway heartbeat, never at the
later reconciliation wall-clock time.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - exercised via the package import in tests
    from .control_plane import AUTHORITATIVE_TERMINAL_BILLING_STATUSES
except ImportError:  # copied into /app in the pinned sandbox image
    from control_plane import AUTHORITATIVE_TERMINAL_BILLING_STATUSES


# The control plane deliberately remaps a reconciliation sweep's canonical
# type to ``call.failed`` when the SIP layer already reported this call
# failed (see call_lifecycle.py: "a reconciliation sweep after a terminal SIP
# failure is evidence that cleanup completed, not proof the failed call
# became billable"). That is a second legitimate close, not a rejection.
_ACCEPTED_RECONCILED_CANONICAL_TYPES = frozenset({"call.ended", "call.failed"})


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("session timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def reconcile_stale_sessions(
    root: Path,
    *,
    now: str,
    stale_after_seconds: int,
    room_is_active: Callable[[str], bool],
    report: Callable[[dict[str, Any]], dict[str, Any]],
    delete_room: Callable[[str], None],
) -> dict[str, list[Any]]:
    reconciled: list[str] = []
    active: list[str] = []
    skipped: list[str] = []
    # A record this sweep cannot process must never wedge the others: one
    # unhandled session used to abort the whole cycle via an uncaught
    # exception, leaving every OTHER stale session unreconciled behind it and
    # retrying the same poisoned record forever. Isolate each session's work
    # so one failure is counted and reported by call id while the sweep
    # continues.
    failed: list[dict[str, str]] = []
    current = _parse(now)
    for path in sorted(root.glob("*.json")):
        call_id = path.stem
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            call_id = str(state.get("call_id") or "") or path.stem
            room_name = str(state.get("room_name") or "")
            heartbeat = str(state.get("last_heartbeat_at") or "")
            if not all((call_id, room_name, heartbeat)) or state.get("status") in {
                "complete",
                "reconciled",
            }:
                skipped.append(call_id)
                continue
            age = (current - _parse(heartbeat)).total_seconds()
            if age < stale_after_seconds or room_is_active(room_name):
                active.append(call_id)
                continue
            result = report({
                "event_id": f"{call_id}:reconciled:{heartbeat}",
                "call_id": call_id,
                "event_type": "session.reconciled",
                "source_component": "livekit_reconciler",
                "observed_at": heartbeat,
                "details": {"room_name": room_name, "reason": "worker_restart_orphan"},
            })
            canonical_type = result.get("canonical_type") if isinstance(result, dict) else None
            billing = result.get("billing") if isinstance(result, dict) else None
            if canonical_type not in _ACCEPTED_RECONCILED_CANONICAL_TYPES or not isinstance(
                billing, dict
            ):
                raise RuntimeError("control plane did not finalize reconciled session billing")
            if billing.get("status") not in AUTHORITATIVE_TERMINAL_BILLING_STATUSES:
                raise RuntimeError("reconciled session has no authoritative billing verdict")
            delete_room(room_name)
            state["status"] = "reconciled"
            state["reconciled_at"] = now
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(path)
            reconciled.append(call_id)
        except Exception as exc:
            failed.append(
                {
                    "call_id": call_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return {"reconciled": reconciled, "active": active, "skipped": skipped, "failed": failed}

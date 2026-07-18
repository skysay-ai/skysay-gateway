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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
) -> dict[str, list[str]]:
    reconciled: list[str] = []
    active: list[str] = []
    skipped: list[str] = []
    current = _parse(now)
    for path in sorted(root.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        call_id = str(state.get("call_id") or "")
        room_name = str(state.get("room_name") or "")
        heartbeat = str(state.get("last_heartbeat_at") or "")
        if not all((call_id, room_name, heartbeat)) or state.get("status") in {
            "complete",
            "reconciled",
        }:
            skipped.append(call_id or path.stem)
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
        billing = result.get("billing") if isinstance(result, dict) else None
        if result.get("canonical_type") != "call.ended" or not isinstance(billing, dict):
            raise RuntimeError("control plane did not finalize reconciled session billing")
        if billing.get("status") not in {"settled", "no_reservation"}:
            raise RuntimeError("reconciled session has no authoritative billing verdict")
        delete_room(room_name)
        state["status"] = "reconciled"
        state["reconciled_at"] = now
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
        reconciled.append(call_id)
    return {"reconciled": reconciled, "active": active, "skipped": skipped}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit import api

from control_plane import UrllibJsonTransport
from production import post_livekit_worker_heartbeat, worker_heartbeats_enabled
from reconciliation import reconcile_stale_sessions


ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reconcile_error_line(payload: dict[str, str]) -> str:
    """Render one ``SCH236_RECONCILE_ERROR`` log line.

    Kept pure (no I/O) so the exact shape a poisoned session or a cycle-level
    failure produces is unit-testable without a live LiveKit API or event
    loop. Every caller must supply ``error_type``/``error_message`` at
    minimum; a per-session failure also supplies ``call_id``.
    """
    return "SCH236_RECONCILE_ERROR " + json.dumps(payload, sort_keys=True)


async def reconcile_once() -> dict[str, list[Any]]:
    client = api.LiveKitAPI()
    transport = UrllibJsonTransport(
        base_url=os.environ["OPENPHONEX_CONTROL_PLANE_URL"],
        bearer_token=os.environ["OPENPHONEX_PLATFORM_MEDIA_TOKEN"],
    )
    try:
        # A heartbeat must never suppress reconciliation: it is release
        # evidence, whereas settlement cleanup remains the worker's job.
        worker_token = str(os.environ.get("OPENPHONEX_WORKER_API_TOKEN") or "").strip()
        if worker_heartbeats_enabled(os.environ) and worker_token:
            try:
                await asyncio.to_thread(
                    post_livekit_worker_heartbeat,
                    UrllibJsonTransport(
                        base_url=os.environ["OPENPHONEX_CONTROL_PLANE_URL"],
                        bearer_token=worker_token,
                    ),
                    os.environ,
                    process_id=os.getpid(),
                    worker_type="livekit_reconciler",
                )
            except Exception as exc:
                # Non-fatal (a heartbeat must never suppress reconciliation),
                # but the same "type name only" observability trap applies
                # here too -- carry the message and traceback as well.
                print(
                    "SCH236_RECONCILER_HEARTBEAT_ERROR "
                    + json.dumps(
                        {
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        rooms = await client.room.list_rooms(api.ListRoomsRequest())
        active_agent_rooms: set[str] = set()
        for room in rooms.rooms:
            participants = await client.room.list_participants(
                api.ListParticipantsRequest(room=room.name)
            )
            if any(
                participant.identity.startswith("agent-")
                or participant.identity.startswith("AJ_")
                for participant in participants.participants
            ):
                active_agent_rooms.add(room.name)
        delete_after_settlement: list[str] = []
        verdict = await asyncio.to_thread(
            reconcile_stale_sessions,
            ROOT,
            now=utcnow(),
            stale_after_seconds=int(os.environ.get("SCH236_RECONCILE_STALE_SECONDS") or "15"),
            room_is_active=lambda room: room in active_agent_rooms,
            report=lambda payload: transport.request(
                "POST",
                f"/v1/gateway-deployments/{os.environ['OPENPHONEX_GATEWAY_ID']}/call-lifecycle-events",
                payload,
            ),
            delete_room=delete_after_settlement.append,
        )
        for room_name in delete_after_settlement:
            await client.room.delete_room(api.DeleteRoomRequest(room=room_name))
        return verdict
    finally:
        await client.aclose()


async def main() -> None:
    while True:
        try:
            verdict = await reconcile_once()
            # A session the sweep could not process is isolated inside
            # reconcile_stale_sessions and never aborts the cycle; report
            # each one by call id with its real message and traceback
            # instead of the opaque cycle-wide failure this used to be.
            for failure in verdict.get("failed", []):
                print(
                    _reconcile_error_line(
                        {
                            "call_id": failure.get("call_id", ""),
                            "error_type": failure.get("error_type", ""),
                            "error_message": failure.get("error_message", ""),
                            "traceback": failure.get("traceback", ""),
                        }
                    ),
                    flush=True,
                )
            if verdict["reconciled"]:
                print(
                    "SCH236_RECONCILED " + json.dumps(verdict, sort_keys=True),
                    flush=True,
                )
        except Exception as exc:
            # A failure here is outside any single session (LiveKit API
            # unreachable, a missing required env var, ...) and still aborts
            # the cycle; it must still be diagnosable from its own line.
            print(
                _reconcile_error_line(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())

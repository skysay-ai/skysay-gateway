# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit import api

from control_plane import UrllibJsonTransport
from reconciliation import reconcile_stale_sessions


ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def reconcile_once() -> dict[str, list[str]]:
    client = api.LiveKitAPI()
    transport = UrllibJsonTransport(
        base_url=os.environ["OPENPHONEX_CONTROL_PLANE_URL"],
        bearer_token=os.environ["OPENPHONEX_GATEWAY_API_TOKEN"],
    )
    try:
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
            if verdict["reconciled"]:
                print(
                    "SCH236_RECONCILED " + json.dumps(verdict, sort_keys=True),
                    flush=True,
                )
        except Exception as exc:
            print(
                "SCH236_RECONCILE_ERROR "
                + json.dumps({"error_type": type(exc).__name__}, sort_keys=True),
                flush=True,
            )
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())

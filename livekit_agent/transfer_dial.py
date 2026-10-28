# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Place the one human leg for a transfer into the call's existing room.

The caller stays on the call. This adds the person they asked for, using the
workspace's own number as the caller ID. The free-form ``transfer_to_human``
tool and a workflow transfer step share these room operations.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

try:
    from .transfer import HUMAN_IDENTITY, classify_dial_failure
except ImportError:  # pragma: no cover - flat import in the worker image
    from transfer import HUMAN_IDENTITY, classify_dial_failure


async def place_transfer_leg(room: Any, plan: dict[str, Any]) -> str:
    """Dial ``plan['destination_e164']`` into ``room``. Return an outcome name.

    Cancelling this coroutine does NOT stop the SIP call: the SIP service runs
    the request detached from its caller. Removing the leg's participant
    (:func:`remove_transfer_leg`) is what ends a ringing INVITE; a cancelled
    dial only stops waiting, and its client is closed either way.
    """
    destination = str(plan.get("destination_e164") or "").strip()
    caller_id = str(plan.get("caller_id_e164") or "").strip()
    ring = int(plan.get("ring_seconds") or 25)
    room_name = str(getattr(room, "name", "") or "").strip()
    if not destination or not caller_id or not room_name:
        return "failed"
    from google.protobuf.duration_pb2 import Duration
    from livekit import api

    client = _client()

    # The SIP leg's own limits, as the outbound originator sets them: the ring
    # ends on the trunk (a CANCEL), not only when this request times out, and
    # a leg the room outlives is still bounded.
    limits = {"ringing_timeout": Duration(seconds=ring)}
    max_call = plan.get("max_call_seconds")
    if type(max_call) is int and max_call > 0:
        limits["max_call_duration"] = Duration(seconds=max_call)
    try:
        trunks = await client.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
        caller_digits = caller_id.removeprefix("+")
        matches = [
            trunk
            for trunk in getattr(trunks, "items", ())
            if caller_digits in {str(number).removeprefix("+") for number in getattr(trunk, "numbers", ())}
        ]
        if len(matches) != 1:
            return "failed"
        await client.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=matches[0].sip_trunk_id,
                sip_call_to=destination,
                sip_number=caller_id,
                participant_identity=HUMAN_IDENTITY,
                participant_name="Transfer",
                wait_until_answered=True,
                **limits,
            ),
            timeout=ring + 5,
        )
        return "answered"
    except Exception as exc:
        return classify_dial_failure(str(exc))
    finally:
        await _close(client)


def _client():
    from livekit import api

    return api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )


async def _close(client: Any) -> None:
    """Close ``client`` even when the request it served was cancelled: a
    second cancellation during the close leaves the close running."""
    await asyncio.shield(client.aclose())


async def subscribe_agent_audio(room: Any, caller_identity: str, listener: str, heard: bool) -> None:
    """Whether ``listener`` (``caller`` or ``human``) hears the agent's tracks.

    The server ``update_subscriptions`` call, not the agent's own
    ``set_subscribed``, which only controls what the agent hears.
    """
    from livekit import api

    identity = HUMAN_IDENTITY if listener == "human" else str(caller_identity or "")
    local = getattr(room, "local_participant", None)
    publications = getattr(local, "track_publications", {}) or {}
    track_sids = [str(getattr(item, "sid", "") or "") for item in publications.values()]
    track_sids = [sid for sid in track_sids if sid]
    if not identity or not track_sids:
        return
    client = _client()
    try:
        await client.room.update_subscriptions(
            api.UpdateSubscriptionsRequest(
                room=str(getattr(room, "name", "") or ""),
                identity=identity,
                track_sids=track_sids,
                subscribe=heard,
            )
        )
    finally:
        await _close(client)


async def remove_transfer_leg(room: Any) -> bool:
    """Take the transferred person's leg out of the room, if there is one.

    Returns whether a leg was removed: ``False`` when LiveKit has no such
    participant or no such room (never created, already gone, or not yet
    joined by a dial still pending). Any other failure is raised, since it may
    leave a phone ringing.
    """
    from livekit import api

    client = _client()
    try:
        await client.room.remove_participant(
            api.RoomParticipantIdentity(
                room=str(getattr(room, "name", "") or ""),
                identity=HUMAN_IDENTITY,
            )
        )
    except Exception as exc:
        if str(getattr(exc, "code", "") or "") == "not_found":
            return False
        raise
    finally:
        await _close(client)
    return True

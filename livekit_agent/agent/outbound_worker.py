# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from ..control_plane import Gate2ControlPlaneAdapter, UrllibJsonTransport
    from ..production import terminate_pre_media_call
except ImportError:  # flat /app import in the worker image
    from control_plane import Gate2ControlPlaneAdapter, UrllibJsonTransport
    from production import terminate_pre_media_call


AGENT_NAME = "sch236-livekit-demo"
OUTBOUND_TRUNK_NAME = "openphonex-selfhost-outbound"
DISPATCH_SCHEMA = "openphonex.outbound-dispatch.v1"


class OutboundWorkerError(RuntimeError):
    pass


class OutboundPostAdmissionError(OutboundWorkerError):
    """A terminal outcome after the outbound authorization may have dialed."""

    outbound_retry_safe = False


def _required(environ: Mapping[str, str], name: str) -> str:
    value = str(environ.get(name) or "").strip()
    if not value:
        raise OutboundWorkerError(f"missing required setting {name}")
    return value


def _optional(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name) or "").strip()


def _enabled(environ: Mapping[str, str], name: str) -> bool:
    return str(environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_id(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in "-_")[:96]


def _redact_phone(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    return "..." + digits[-4:] if len(digits) >= 4 else "[redacted]"


@dataclass(frozen=True)
class OutboundWorkerSettings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    control_plane_url: str
    gateway_id: str
    gateway_token: str
    worker_token: str
    organization_id: str
    worker_id: str
    trunk_name: str
    poll_seconds: float
    ringing_timeout_seconds: int
    max_call_duration_seconds: int
    execute: bool

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "OutboundWorkerSettings":
        poll_seconds = float(environ.get("OPENPHONEX_OUTBOUND_POLL_SECONDS") or "2")
        ringing_timeout = int(environ.get("OPENPHONEX_OUTBOUND_RINGING_TIMEOUT_SECONDS") or "45")
        max_duration = int(environ.get("OPENPHONEX_OUTBOUND_MAX_CALL_DURATION_SECONDS") or "1800")
        if poll_seconds < 0:
            raise OutboundWorkerError("OPENPHONEX_OUTBOUND_POLL_SECONDS must be non-negative")
        if ringing_timeout < 5 or ringing_timeout > 120:
            raise OutboundWorkerError("outbound ringing timeout must be between 5 and 120 seconds")
        if max_duration < 30 or max_duration > 14_400:
            raise OutboundWorkerError("outbound max call duration must be between 30 and 14400 seconds")
        return cls(
            livekit_url=_required(environ, "LIVEKIT_URL"),
            livekit_api_key=_required(environ, "LIVEKIT_API_KEY"),
            livekit_api_secret=_required(environ, "LIVEKIT_API_SECRET"),
            control_plane_url=_required(environ, "OPENPHONEX_CONTROL_PLANE_URL"),
            gateway_id=_required(environ, "OPENPHONEX_GATEWAY_ID"),
            gateway_token=_required(environ, "OPENPHONEX_GATEWAY_API_TOKEN"),
            # Only the QUEUE originator needs this: claiming jobs is
            # organization authority. A box that originates its own calls
            # locally never claims anything, so it sets no organization-scoped
            # key at all -- and `_require_worker_token` below is what makes
            # that a clear refusal on the paths that do need one, instead of a
            # missing-setting error on a box that legitimately has none.
            worker_token=_optional(environ, "OPENPHONEX_WORKER_API_TOKEN"),
            organization_id=_required(environ, "OPENPHONEX_ORGANIZATION_ID"),
            worker_id=str(environ.get("OPENPHONEX_OUTBOUND_WORKER_ID") or "livekit-outbound-1").strip(),
            trunk_name=str(environ.get("OPENPHONEX_OUTBOUND_TRUNK_NAME") or OUTBOUND_TRUNK_NAME).strip(),
            poll_seconds=poll_seconds,
            ringing_timeout_seconds=ringing_timeout,
            max_call_duration_seconds=max_duration,
            execute=_enabled(environ, "OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND"),
        )


def _gateway_transport(settings: OutboundWorkerSettings) -> UrllibJsonTransport:
    return UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=settings.gateway_token,
    )


def _require_worker_token(settings: OutboundWorkerSettings) -> str:
    if not settings.worker_token:
        raise OutboundWorkerError(
            "the outbound job queue needs OPENPHONEX_WORKER_API_TOKEN; a deployment that "
            "originates its own calls uses `originate` instead and needs no organization key"
        )
    return settings.worker_token


def _transports(settings: OutboundWorkerSettings) -> tuple[UrllibJsonTransport, UrllibJsonTransport]:
    worker = UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=_require_worker_token(settings),
    )
    return _gateway_transport(settings), worker


def _validate_job(settings: OutboundWorkerSettings, job: dict[str, Any]) -> dict[str, str]:
    if job.get("kind") != "call.outbound_originate" or job.get("status") != "running":
        raise OutboundWorkerError("claimed job is not a running outbound originate job")
    if str(job.get("organization_id") or "") != settings.organization_id:
        raise OutboundWorkerError("claimed job belongs to a different organization")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise OutboundWorkerError("outbound job payload must be an object")
    fields = {
        "call_id": str(payload.get("call_id") or "").strip(),
        "agent_id": str(payload.get("agent_id") or "").strip(),
        "from_number_id": str(payload.get("number_id") or "").strip(),
        "from_e164": str(payload.get("from_phone_number") or "").strip(),
        "to_e164": str(payload.get("to_phone_number") or "").strip(),
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise OutboundWorkerError(f"outbound job omitted required fields: {', '.join(missing)}")
    if not fields["from_e164"].startswith("+") or not fields["to_e164"].startswith("+"):
        raise OutboundWorkerError("outbound job numbers must use E.164 format")
    gate = payload.get("live_call_gate")
    if not isinstance(gate, dict) or gate.get("enabled") is not True:
        raise OutboundWorkerError("outbound job live-call gate is not enabled")
    return fields


def _duration_limits(settings: OutboundWorkerSettings) -> dict[str, Any]:
    from google.protobuf.duration_pb2 import Duration

    return {
        "ringing_timeout": Duration(seconds=settings.ringing_timeout_seconds),
        "max_call_duration": Duration(seconds=settings.max_call_duration_seconds),
    }


async def _select_outbound_trunk(
    client: Any,
    api_module: Any,
    settings: OutboundWorkerSettings,
    from_e164: str,
    *,
    trunk_name: str = "",
) -> Any:
    # Per-number trunk dispatch: the control plane resolves the trunk name
    # from the NUMBER's provider account (BYO SIP trunk carriers record
    # theirs) and threads it through the job payload; the worker's
    # env-configured name stays the default for managed numbers. Caller-ID
    # membership remains a hard filter either way, so a job can never dial
    # out of a trunk that does not carry its number.
    wanted_name = str(trunk_name or "").strip() or settings.trunk_name
    response = await client.sip.list_outbound_trunk(api_module.ListSIPOutboundTrunkRequest())
    from_digits = from_e164.removeprefix("+")
    matches = [
        trunk
        for trunk in response.items
        if str(getattr(trunk, "name", "")) == wanted_name
        and from_digits in {str(number).removeprefix("+") for number in getattr(trunk, "numbers", ())}
    ]
    if len(matches) != 1:
        raise OutboundWorkerError(
            "expected exactly one configured outbound trunk containing the requested caller ID"
        )
    return matches[0]


async def _originate_admitted_call(
    settings: OutboundWorkerSettings,
    adapter: Any,
    admit: Any,
    *,
    known_call_id: str = "",
    api_module: Any | None = None,
) -> dict[str, Any]:
    """Dial one call whose ORIGINATION the control plane has already approved.

    ``admit`` is the one step that differs between the two originators: the
    queue path claimed somebody else's job and mints its token from the call it
    was handed, while a locally-originating deployment asks for admission and
    the token together. Everything after it -- spending that token on media
    admission, selecting the trunk that carries the caller ID, the room, the
    explicit agent dispatch, the SIP participant, and the pre-media teardown
    that releases a reservation -- is deliberately ONE body, so the two
    originators cannot drift into two different dialers.

    ``admit`` reports the admitted call id through ``observe_call_id`` as soon
    as it knows one, so a failure between admission and the session is still
    reported against the call that is holding the reservation.
    """
    fields: dict[str, str] = {}
    trunk_name = ""
    resolved_call_id = str(known_call_id or "")
    call: Any | None = None
    client: Any | None = None
    room_name = ""
    participant_identity = ""
    room_cleanup_required = False
    participant_cleanup_required = False
    may_have_originated = False
    stage = "outbound_authorization"

    def observe_call_id(call_id: str) -> None:
        nonlocal resolved_call_id
        resolved_call_id = str(call_id or "") or resolved_call_id

    async def terminate_failure() -> None:
        async def remove_participant() -> None:
            if client is None or not participant_cleanup_required:
                return
            await client.room.remove_participant(
                api_module.RoomParticipantIdentity(
                    room=room_name,
                    identity=participant_identity,
                )
            )

        async def delete_room() -> None:
            if client is None or not room_cleanup_required:
                return
            await client.room.delete_room(api_module.DeleteRoomRequest(room=room_name))

        await terminate_pre_media_call(
            call=call,
            adapter=adapter,
            call_id=resolved_call_id,
            reason={
                "outbound_authorization": "outbound_authorization_rejected",
                "outbound_media_admission": "outbound_media_admission_rejected",
                "outbound_trunk": "outbound_trunk_rejected",
                "outbound_room": "outbound_room_rejected",
                "outbound_dispatch": "outbound_dispatch_rejected",
                "outbound_sip_participant": "outbound_sip_participant_rejected",
            }.get(stage, "outbound_pre_media_rejected"),
            remove_participant=remove_participant,
            delete_room=delete_room,
        )

    try:
        fields, trunk_name, call = await asyncio.to_thread(admit, observe_call_id)
        resolved_call_id = fields["call_id"]
        room_name = f"opx-out-{_safe_id(resolved_call_id)}"
        participant_identity = f"pstn-{_safe_id(resolved_call_id)}"
        dispatch = {
            "schema": DISPATCH_SCHEMA,
            **fields,
            "participant_identity": participant_identity,
        }
        # Once this starts, a failure may race real SIP origination. The
        # worker must settle the one claim terminally rather than redial.
        stage = "outbound_media_admission"
        may_have_originated = True
        await asyncio.to_thread(call.admit_media)
        if api_module is None:
            from livekit import api as api_module
        client = api_module.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        stage = "outbound_trunk"
        # Per-number trunk dispatch, read fresh from the job payload on every
        # attempt, so a retry dials the same BYO trunk the control plane
        # resolved for this number - never the env default by accident.
        trunk = await _select_outbound_trunk(
            client,
            api_module,
            settings,
            fields["from_e164"],
            trunk_name=trunk_name,
        )
        # Set before the RPC: a timeout can occur after LiveKit created it.
        room_cleanup_required = True
        stage = "outbound_room"
        await client.room.create_room(
            api_module.CreateRoomRequest(
                name=room_name,
                empty_timeout=60,
                departure_timeout=15,
                max_participants=3,
            )
        )
        stage = "outbound_dispatch"
        await client.agent_dispatch.create_dispatch(
            api_module.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(dispatch, sort_keys=True),
            )
        )
        # A timeout can occur after the SIP participant exists; remove this
        # deterministic identity before deleting the call-scoped room.
        participant_cleanup_required = True
        stage = "outbound_sip_participant"
        participant = await client.sip.create_sip_participant(
            api_module.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=trunk.sip_trunk_id,
                sip_call_to=fields["to_e164"],
                sip_number=fields["from_e164"].removeprefix("+"),
                participant_identity=participant_identity,
                participant_name="OpenPhonex outbound call",
                participant_attributes={
                    "openphonex.call_id": fields["call_id"],
                    "openphonex.direction": "outbound",
                },
                wait_until_answered=True,
                media=api_module.SIPMediaConfig(
                    only_listed_codecs=True,
                    codecs=[api_module.SIPCodec(name="PCMU", rate=8_000)],
                ),
                **_duration_limits(settings),
            ),
            timeout=settings.ringing_timeout_seconds + 10,
        )
        return {
            "executed": True,
            "provider": "livekit_sip",
            "call_id": fields["call_id"],
            "room_name": room_name,
            "participant_identity": participant_identity,
            "provider_call_id": str(getattr(participant, "sip_call_id", "") or ""),
            "call_status": "answered",
            "from": _redact_phone(fields["from_e164"]),
            "to": _redact_phone(fields["to_e164"]),
            "secrets_redacted": True,
        }
    except asyncio.CancelledError:
        await terminate_failure()
        raise
    except Exception as exc:
        await terminate_failure()
        if may_have_originated:
            raise OutboundPostAdmissionError(
                "outbound execution ended after media authorization"
            ) from exc
        raise
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                # A close failure must not hide the canonical SIP failure or
                # change the worker's retry decision.
                pass


async def originate_job(
    settings: OutboundWorkerSettings,
    job: dict[str, Any],
    *,
    api_module: Any | None = None,
) -> dict[str, Any]:
    """Dial a call the control plane admitted for somebody else to place."""

    fields = _validate_job(settings, job)
    gateway_transport, worker_transport = _transports(settings)
    payload = job["payload"]
    auth_metadata = payload.get("authorization_metadata")
    if not isinstance(auth_metadata, dict):
        auth_metadata = {"channel": "voice"}
    adapter = Gate2ControlPlaneAdapter(
        gateway_id=settings.gateway_id,
        transport=gateway_transport,
        call_transport=worker_transport,
        tool_transport=worker_transport,
    )

    def admit(_observe_call_id: Any) -> tuple[dict[str, str], str, Any]:
        session = adapter.prepare_outbound(
            call_id=fields["call_id"],
            agent_id=fields["agent_id"],
            from_number_id=fields["from_number_id"],
            to_e164=fields["to_e164"],
            metadata=auth_metadata,
        )
        # Per-number trunk dispatch, read fresh from the job payload on every
        # attempt, so a retry dials the same BYO trunk the control plane
        # resolved for this number - never the env default by accident.
        return fields, str(payload.get("outbound_trunk_name") or ""), session

    return await _originate_admitted_call(
        settings,
        adapter,
        admit,
        known_call_id=fields["call_id"],
        api_module=api_module,
    )


async def originate_local_call(
    settings: OutboundWorkerSettings,
    *,
    to_e164: str,
    from_number_id: str = "",
    from_phone_number: str = "",
    request: dict[str, Any] | None = None,
    api_module: Any | None = None,
) -> dict[str, Any]:
    """Place a call this deployment asked for, with only its own credential.

    The queue is not involved and neither is an organization-scoped key: the
    control plane admits the call against this deployment's token, applies the
    same policy decision and wallet reserve a remote request gets, and returns
    the single-use token that admits its media.
    """

    if not settings.execute:
        raise OutboundWorkerError(
            "outbound execution is disabled; set OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND=1 to place calls"
        )
    adapter = Gate2ControlPlaneAdapter(
        gateway_id=settings.gateway_id,
        transport=_gateway_transport(settings),
    )

    def admit(observe_call_id: Any) -> tuple[dict[str, str], str, Any]:
        dial_plan, token = adapter.admit_local_outbound(
            to_e164=to_e164,
            from_number_id=from_number_id,
            from_phone_number=from_phone_number,
            request=request,
        )
        # Reported before the config fetch below: from here on the call exists
        # and holds a reservation, so a later failure must be attributable.
        observe_call_id(dial_plan["call_id"])
        session = adapter.session_for_local_outbound(dial_plan, token)
        return (
            {
                "call_id": dial_plan["call_id"],
                "agent_id": dial_plan["agent_id"],
                "from_number_id": dial_plan["number_id"],
                "from_e164": dial_plan["from_phone_number"],
                "to_e164": dial_plan["to_phone_number"],
            },
            dial_plan.get("outbound_trunk_name", ""),
            session,
        )

    return await _originate_admitted_call(settings, adapter, admit, api_module=api_module)


def _job_error_code(exc: BaseException) -> str:
    """Exception CLASS NAMES only - never a message, which can carry provider detail.

    A post-admission wrapper is terminal for the job, so on its own it tells an
    operator nothing about WHY. Appending the wrapped class keeps a failure
    caused by a per-number trunk that does not exist or does not carry the
    caller ID (OutboundWorkerError) distinguishable from a transient SDK or
    transport error, without widening what the job record reveals.
    """

    name = type(exc).__name__
    cause = exc.__cause__
    if cause is None or type(cause) is type(exc):
        return name
    return f"{name}:{type(cause).__name__}"


class OutboundWorker:
    def __init__(self, settings: OutboundWorkerSettings) -> None:
        self.settings = settings
        self.gateway_transport, self.worker_transport = _transports(settings)

    def preview(self) -> dict[str, Any]:
        heartbeat = self.worker_transport.request(
            "POST",
            "/v1/worker-heartbeats",
            {
                "worker_id": self.settings.worker_id,
                "worker_type": "livekit_outbound",
                "status": "staging" if not self.settings.execute else "ready",
                "metadata": {
                    "execute_gate_enabled": self.settings.execute,
                    "trunk_name": self.settings.trunk_name,
                },
            },
        )
        return {
            "execute": self.settings.execute,
            "worker_id": self.settings.worker_id,
            "organization_id": self.settings.organization_id,
            "trunk_name": self.settings.trunk_name,
            "worker_token_authenticated": bool(heartbeat.get("id")),
            "secrets_redacted": True,
        }

    def claim(self) -> dict[str, Any] | None:
        response = self.worker_transport.request(
            "POST",
            "/v1/worker-jobs/claim",
            {
                "worker_id": self.settings.worker_id,
                "kind": "call.outbound_originate",
                "organization_id": self.settings.organization_id,
            },
        )
        job = response.get("job")
        return job if isinstance(job, dict) else None

    async def run_once(self) -> dict[str, Any]:
        if not self.settings.execute:
            return {"claimed": False, "dry_run": True, **self.preview()}
        job = await asyncio.to_thread(self.claim)
        if job is None:
            return {"claimed": False, "dry_run": False, "idle": True}
        try:
            result = await originate_job(self.settings, job)
            completed = await asyncio.to_thread(
                self.worker_transport.request,
                "POST",
                f"/v1/worker-jobs/{job['id']}/complete",
                {"result": result},
            )
            return {"claimed": True, "result": result, "job_status": completed.get("status", "succeeded")}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed = await asyncio.to_thread(
                self.worker_transport.request,
                "POST",
                f"/v1/worker-jobs/{job['id']}/fail",
                {
                    "error": _job_error_code(exc),
                    "retry": bool(getattr(exc, "outbound_retry_safe", True)),
                    "delay_seconds": 5 if getattr(exc, "outbound_retry_safe", True) else 0,
                    "max_attempts": 3 if getattr(exc, "outbound_retry_safe", True) else 1,
                },
            )
            return {
                "claimed": True,
                "error_type": type(exc).__name__,
                "job_status": failed.get("status", "failed"),
                "secrets_redacted": True,
            }

    async def run(self, *, watch: bool) -> int:
        while True:
            result = await self.run_once()
            print("OPENPHONEX_OUTBOUND " + json.dumps(result, sort_keys=True), flush=True)
            if not watch:
                return 0 if not result.get("error_type") else 1
            await asyncio.sleep(self.settings.poll_seconds)


def _default_caller_number(environ: Mapping[str, str]) -> str:
    """The first DID this deployment serves, used when none is named.

    A single-DID box answers and dials on the same number, and that number is
    already configured once. Taking the first entry keeps `originate --to ...`
    a complete command there, while a box with several numbers names the one
    it wants.
    """
    configured = str(environ.get("SCH236_TEST_DID_E164") or "").strip()
    return configured.split(",")[0].strip() if configured else ""


def _originate_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="outbound_worker.py originate",
        description="Place one outbound call from this deployment, without the job queue",
    )
    parser.add_argument("--to", required=True, help="destination in E.164, e.g. +37255501234")
    parser.add_argument(
        "--from",
        dest="from_phone_number",
        default="",
        help="caller id in E.164; defaults to the first number in SCH236_TEST_DID_E164",
    )
    parser.add_argument(
        "--from-number-id", default="", help="caller id by control-plane number id, instead of --from"
    )
    parser.add_argument("--agent-id", default="", help="agent to speak; defaults to the number's own agent")
    # Required, not defaulted: outbound authorization is a declaration about
    # WHY this call is being placed, and inventing one here would put a
    # compliance claim in the record that nobody actually made. A call without
    # them is refused by policy anyway -- better to say so before dialling.
    parser.add_argument("--purpose", required=True, help="declared call purpose, evaluated by policy")
    parser.add_argument("--reason", required=True, help="why this call was asked for")
    parser.add_argument("--recipient-type", default="", help="declared recipient type, evaluated by policy")
    parser.add_argument("--relationship", default="", help="declared relationship, evaluated by policy")
    parser.add_argument("--authorized-by", default="", help="who asked for this call")
    parser.add_argument("--idempotency-key", default="", help="retry this exact request safely")
    args = parser.parse_args(argv)

    if not args.to.startswith("+"):
        raise OutboundWorkerError("--to must be an E.164 number starting with +")
    from_phone_number = args.from_phone_number.strip() or (
        "" if args.from_number_id else _default_caller_number(os.environ)
    )
    if from_phone_number and not from_phone_number.startswith("+"):
        raise OutboundWorkerError("--from must be an E.164 number starting with +")
    if not from_phone_number and not args.from_number_id:
        raise OutboundWorkerError(
            "no caller id: pass --from in E.164, or set SCH236_TEST_DID_E164"
        )
    request: dict[str, Any] = {}
    for key, value in (
        ("agent_id", args.agent_id),
        ("purpose", args.purpose),
        ("recipient_type", args.recipient_type),
        ("relationship", args.relationship),
        ("idempotency_key", args.idempotency_key),
    ):
        if value:
            request[key] = value
    request["user_authorization"] = {
        "authorized_by": args.authorized_by or "self_hosted_gateway",
        "reason": args.reason,
    }
    settings = OutboundWorkerSettings.from_environ(os.environ)
    result = asyncio.run(
        originate_local_call(
            settings,
            to_e164=args.to,
            from_number_id=args.from_number_id,
            from_phone_number=from_phone_number,
            request=request,
        )
    )
    print("OPENPHONEX_OUTBOUND " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "originate":
            # A one-shot originator, not a second long-running service: it
            # opens no port, so nothing on this host can be talked into
            # dialling. `docker compose run` does NOT apply profiles, so the
            # execution gate this path enforces itself
            # (OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND) is the one that gates it;
            # `--profile outbound` still gates the queue watcher.
            return _originate_main(arguments[1:])
        parser = argparse.ArgumentParser(description="OpenPhonex LiveKit outbound originate worker")
        parser.add_argument("--once", action="store_true", help="poll once and exit")
        parser.add_argument("--watch", action="store_true", help="poll continuously")
        parser.add_argument("--status", action="store_true", help="show a redacted queue preview and exit")
        args = parser.parse_args(arguments)
        worker = OutboundWorker(OutboundWorkerSettings.from_environ(os.environ))
        if args.status:
            print(json.dumps(worker.preview(), sort_keys=True))
            return 0
        return asyncio.run(worker.run(watch=args.watch and not args.once))
    except SystemExit:
        raise
    except BaseException as exc:
        print(
            "OPENPHONEX_OUTBOUND_FAILED "
            + json.dumps({"error_type": type(exc).__name__}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

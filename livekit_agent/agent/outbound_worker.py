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
except ImportError:  # flat /app import in the worker image
    from control_plane import Gate2ControlPlaneAdapter, UrllibJsonTransport


AGENT_NAME = "sch236-livekit-demo"
OUTBOUND_TRUNK_NAME = "openphonex-selfhost-outbound"
DISPATCH_SCHEMA = "openphonex.outbound-dispatch.v1"


class OutboundWorkerError(RuntimeError):
    pass


def _required(environ: Mapping[str, str], name: str) -> str:
    value = str(environ.get(name) or "").strip()
    if not value:
        raise OutboundWorkerError(f"missing required setting {name}")
    return value


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
            worker_token=_required(environ, "OPENPHONEX_WORKER_API_TOKEN"),
            organization_id=_required(environ, "OPENPHONEX_ORGANIZATION_ID"),
            worker_id=str(environ.get("OPENPHONEX_OUTBOUND_WORKER_ID") or "livekit-outbound-1").strip(),
            trunk_name=str(environ.get("OPENPHONEX_OUTBOUND_TRUNK_NAME") or OUTBOUND_TRUNK_NAME).strip(),
            poll_seconds=poll_seconds,
            ringing_timeout_seconds=ringing_timeout,
            max_call_duration_seconds=max_duration,
            execute=_enabled(environ, "OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND"),
        )


def _transports(settings: OutboundWorkerSettings) -> tuple[UrllibJsonTransport, UrllibJsonTransport]:
    gateway = UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=settings.gateway_token,
    )
    worker = UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=settings.worker_token,
    )
    return gateway, worker


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


async def _select_outbound_trunk(client: Any, api_module: Any, settings: OutboundWorkerSettings, from_e164: str) -> Any:
    response = await client.sip.list_outbound_trunk(api_module.ListSIPOutboundTrunkRequest())
    from_digits = from_e164.removeprefix("+")
    matches = [
        trunk
        for trunk in response.items
        if str(getattr(trunk, "name", "")) == settings.trunk_name
        and from_digits in {str(number).removeprefix("+") for number in getattr(trunk, "numbers", ())}
    ]
    if len(matches) != 1:
        raise OutboundWorkerError(
            "expected exactly one configured outbound trunk containing the requested caller ID"
        )
    return matches[0]


async def originate_job(
    settings: OutboundWorkerSettings,
    job: dict[str, Any],
    *,
    api_module: Any | None = None,
) -> dict[str, Any]:
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
    call = await asyncio.to_thread(
        adapter.prepare_outbound,
        call_id=fields["call_id"],
        agent_id=fields["agent_id"],
        from_number_id=fields["from_number_id"],
        to_e164=fields["to_e164"],
        metadata=auth_metadata,
    )
    # Consumes the short-lived destination-bound token before any SIP dial.
    await asyncio.to_thread(call.admit_media)

    if api_module is None:
        from livekit import api as api_module

    client = api_module.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    room_name = f"opx-out-{_safe_id(fields['call_id'])}"
    participant_identity = f"pstn-{_safe_id(fields['call_id'])}"
    dispatch = {
        "schema": DISPATCH_SCHEMA,
        **fields,
        "participant_identity": participant_identity,
    }
    room_created = False
    answered = False
    try:
        trunk = await _select_outbound_trunk(client, api_module, settings, fields["from_e164"])
        await client.room.create_room(
            api_module.CreateRoomRequest(
                name=room_name,
                empty_timeout=60,
                departure_timeout=15,
                max_participants=3,
            )
        )
        room_created = True
        await client.agent_dispatch.create_dispatch(
            api_module.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(dispatch, sort_keys=True),
            )
        )
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
        answered = True
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
    except BaseException as exc:
        try:
            await asyncio.to_thread(
                call.report_lifecycle,
                event_id=f"{fields['call_id']}:sip-failed",
                event_type="sip.failed",
                source_component="livekit_sip",
                details={"reason": type(exc).__name__},
            )
        except Exception:
            pass
        raise
    finally:
        if room_created and not answered:
            try:
                await client.room.delete_room(api_module.DeleteRoomRequest(room=room_name))
            except Exception:
                pass
        await client.aclose()


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
        except BaseException as exc:
            failed = await asyncio.to_thread(
                self.worker_transport.request,
                "POST",
                f"/v1/worker-jobs/{job['id']}/fail",
                {
                    "error": type(exc).__name__,
                    "retry": True,
                    "delay_seconds": 5,
                    "max_attempts": 3,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenPhonex LiveKit outbound originate worker")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--watch", action="store_true", help="poll continuously")
    parser.add_argument("--status", action="store_true", help="show a redacted queue preview and exit")
    args = parser.parse_args(argv)
    try:
        worker = OutboundWorker(OutboundWorkerSettings.from_environ(os.environ))
        if args.status:
            print(json.dumps(worker.preview(), sort_keys=True))
            return 0
        return asyncio.run(worker.run(watch=args.watch and not args.once))
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

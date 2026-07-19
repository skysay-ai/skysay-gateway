from __future__ import annotations

import asyncio
import types
import unittest
from unittest.mock import patch

from livekit_agent.agent import outbound_worker as worker


class _Message:
    def __init__(self, **values):
        self.__dict__.update(values)


class _Session:
    def __init__(self, events):
        self.events = events

    def admit_media(self):
        self.events.append("authorization_consumed")

    def report_lifecycle(self, **_kwargs):
        self.events.append("sip_failed_reported")
        return {"canonical_type": "call.failed", "billing": {"status": "released"}}


class _Adapter:
    events = []

    def __init__(self, **_kwargs):
        pass

    def prepare_outbound(self, **kwargs):
        self.events.append(("prepared", kwargs["call_id"], kwargs["to_e164"]))
        return _Session(self.events)


class _Sip:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    async def list_outbound_trunk(self, _request):
        return _Message(items=[_Message(
            name=worker.OUTBOUND_TRUNK_NAME,
            sip_trunk_id="STO_1",
            numbers=["15551230000", "37255501234"],
        )])

    async def create_sip_participant(self, request, timeout=None):
        self.events.append(("dial", request.sip_number, request.sip_call_to, timeout))
        if self.fail:
            raise RuntimeError("carrier unavailable")
        return _Message(sip_call_id="sip_1")


class _Room:
    def __init__(self, events):
        self.events = events

    async def create_room(self, request):
        self.events.append(("room", request.name))

    async def delete_room(self, request):
        self.events.append(("delete_room", request.room))


class _Dispatch:
    def __init__(self, events):
        self.events = events

    async def create_dispatch(self, request):
        self.events.append(("dispatch", request.agent_name, request.room, request.metadata))


class _Client:
    def __init__(self, events, fail=False):
        self.sip = _Sip(events, fail=fail)
        self.room = _Room(events)
        self.agent_dispatch = _Dispatch(events)
        self.events = events

    async def aclose(self):
        self.events.append("closed")


class _Api(types.SimpleNamespace):
    def __init__(self, events, fail=False):
        super().__init__()
        self.client = _Client(events, fail=fail)
        self.LiveKitAPI = lambda *_args: self.client
        for name in (
            "ListSIPOutboundTrunkRequest",
            "CreateRoomRequest",
            "CreateAgentDispatchRequest",
            "CreateSIPParticipantRequest",
            "SIPMediaConfig",
            "SIPCodec",
            "DeleteRoomRequest",
        ):
            setattr(self, name, _Message)


def _settings(**overrides):
    values = dict(
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="key",
        livekit_api_secret="secret",
        control_plane_url="https://api.openphonex.com",
        gateway_id="gw_1",
        gateway_token="tgw_secret",
        worker_token="tai_secret",
        organization_id="org_1",
        worker_id="worker_1",
        trunk_name=worker.OUTBOUND_TRUNK_NAME,
        poll_seconds=0,
        ringing_timeout_seconds=45,
        max_call_duration_seconds=1800,
        execute=True,
    )
    values.update(overrides)
    return worker.OutboundWorkerSettings(**values)


def _job(**payload_overrides):
    payload = {
        "call_id": "call_1",
        "agent_id": "agent_1",
        "number_id": "number_1",
        "from_phone_number": "+15551230000",
        "to_phone_number": "+37255509999",
        "live_call_gate": {"enabled": True},
        "authorization_metadata": {"purpose": "support", "user_requested": True},
    }
    payload.update(payload_overrides)
    return {
        "id": "job_1",
        "organization_id": "org_1",
        "kind": "call.outbound_originate",
        "status": "running",
        "payload": payload,
    }


class LiveKitOutboundWorkerTests(unittest.TestCase):
    def test_authorization_is_consumed_before_room_dispatch_and_dial(self):
        events = []
        _Adapter.events = events
        api = _Api(events)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_transports", return_value=(object(), object())), \
             patch.object(worker, "_duration_limits", return_value={}):
            result = asyncio.run(worker.originate_job(_settings(), _job(), api_module=api))

        self.assertEqual(result["call_status"], "answered")
        self.assertLess(events.index("authorization_consumed"), next(i for i, item in enumerate(events) if isinstance(item, tuple) and item[0] == "room"))
        self.assertLess(next(i for i, item in enumerate(events) if isinstance(item, tuple) and item[0] == "dispatch"), next(i for i, item in enumerate(events) if isinstance(item, tuple) and item[0] == "dial"))
        dial = next(item for item in events if isinstance(item, tuple) and item[0] == "dial")
        self.assertEqual(dial[1:3], ("15551230000", "+37255509999"))
        dispatch = next(item for item in events if isinstance(item, tuple) and item[0] == "dispatch")
        self.assertIn('"call_id": "call_1"', dispatch[3])
        self.assertNotIn("tgw_secret", dispatch[3])
        self.assertNotIn("tai_secret", dispatch[3])

    def test_failed_dial_reports_terminal_lifecycle_and_deletes_room(self):
        events = []
        _Adapter.events = events
        api = _Api(events, fail=True)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_transports", return_value=(object(), object())), \
             patch.object(worker, "_duration_limits", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "carrier unavailable"):
                asyncio.run(worker.originate_job(_settings(), _job(), api_module=api))

        self.assertIn("sip_failed_reported", events)
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "delete_room" for item in events))

    def test_job_validation_rejects_disabled_live_gate(self):
        with self.assertRaisesRegex(worker.OutboundWorkerError, "live-call gate"):
            worker._validate_job(_settings(), _job(live_call_gate={"enabled": False}))

    def test_settings_keep_live_execution_off_by_default(self):
        environ = {
            "LIVEKIT_URL": "ws://127.0.0.1:7880",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
            "OPENPHONEX_CONTROL_PLANE_URL": "https://api.openphonex.com",
            "OPENPHONEX_GATEWAY_ID": "gw_1",
            "OPENPHONEX_GATEWAY_API_TOKEN": "tgw_secret",
            "OPENPHONEX_WORKER_API_TOKEN": "tai_secret",
            "OPENPHONEX_ORGANIZATION_ID": "org_1",
        }
        self.assertFalse(worker.OutboundWorkerSettings.from_environ(environ).execute)


if __name__ == "__main__":
    unittest.main()

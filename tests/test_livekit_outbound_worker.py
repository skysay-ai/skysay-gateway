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

    def report_failed_before_media(self, **_kwargs):
        self.events.append("sip_failed_reported")
        return {"canonical_type": "call.failed", "billing": {"status": "released"}}


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

    async def remove_participant(self, request):
        self.events.append(("remove_participant", request.room, request.identity))


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
            "RoomParticipantIdentity",
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
            with self.assertRaisesRegex(
                worker.OutboundPostAdmissionError,
                "after media authorization",
            ):
                asyncio.run(worker.originate_job(_settings(), _job(), api_module=api))

        self.assertIn("sip_failed_reported", events)
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "delete_room" for item in events))
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "remove_participant" for item in events))

    def test_post_admission_failure_is_terminal_and_never_requests_a_redial(self):
        class _Transport:
            def __init__(self):
                self.requests = []

            def request(self, method, path, payload):
                self.requests.append((method, path, payload))
                return {"status": "failed"}

        transport = _Transport()
        outbound = worker.OutboundWorker(_settings())
        outbound.worker_transport = transport
        outbound.claim = lambda: _job()  # type: ignore[method-assign]

        async def fail_after_admission(*_args, **_kwargs):
            raise worker.OutboundPostAdmissionError("already attempted")

        with patch.object(worker, "originate_job", fail_after_admission):
            result = asyncio.run(outbound.run_once())

        self.assertEqual(result["job_status"], "failed")
        self.assertEqual(len(transport.requests), 1)
        payload = transport.requests[0][2]
        self.assertFalse(payload["retry"])
        self.assertEqual(payload["max_attempts"], 1)

    def test_payload_trunk_name_reaches_the_dial_through_originate_job(self):
        # End-to-end over the merged call site: the payload trunk name must
        # survive the staged pre-media path and be the trunk the SIP
        # participant is actually created on - not the env default.
        events = []
        _Adapter.events = events

        class _TwoTrunkSip(_Sip):
            async def list_outbound_trunk(self, _request):
                return _Message(items=[
                    _Message(name=worker.OUTBOUND_TRUNK_NAME, sip_trunk_id="STO_DEFAULT", numbers=["15551230000"]),
                    _Message(name="openphonex-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["15551230000"]),
                ])

            async def create_sip_participant(self, request, timeout=None):
                self.events.append(("dial_trunk", request.sip_trunk_id))
                return await super().create_sip_participant(request, timeout=timeout)

        api = _Api(events)
        api.client.sip = _TwoTrunkSip(events)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_transports", return_value=(object(), object())), \
             patch.object(worker, "_duration_limits", return_value={}):
            result = asyncio.run(worker.originate_job(
                _settings(),
                _job(outbound_trunk_name="openphonex-didlogic-outbound"),
                api_module=api,
            ))

        self.assertEqual(result["call_status"], "answered")
        self.assertIn(("dial_trunk", "STO_DIDLOGIC"), events)

    def test_originate_job_falls_back_to_the_env_trunk_without_a_payload_name(self):
        # Managed numbers send no trunk name; the env default must still win.
        events = []
        _Adapter.events = events

        class _RecordingSip(_Sip):
            async def create_sip_participant(self, request, timeout=None):
                self.events.append(("dial_trunk", request.sip_trunk_id))
                return await super().create_sip_participant(request, timeout=timeout)

        api = _Api(events)
        api.client.sip = _RecordingSip(events)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_transports", return_value=(object(), object())), \
             patch.object(worker, "_duration_limits", return_value={}):
            asyncio.run(worker.originate_job(_settings(), _job(), api_module=api))

        self.assertIn(("dial_trunk", "STO_1"), events)

    def test_terminal_job_error_keeps_the_underlying_failure_class(self):
        # A misconfigured per-number trunk is terminal, so the worker-job error
        # must still say WHICH failure it was. Class names only - never the
        # exception message, which can carry provider detail.
        class _Transport:
            def __init__(self):
                self.requests = []

            def request(self, method, path, payload):
                self.requests.append((method, path, payload))
                return {"status": "failed"}

        transport = _Transport()
        outbound = worker.OutboundWorker(_settings())
        outbound.worker_transport = transport
        outbound.claim = lambda: _job()  # type: ignore[method-assign]

        async def fail_with_cause(*_args, **_kwargs):
            try:
                raise worker.OutboundWorkerError(
                    "expected exactly one configured outbound trunk containing the requested caller ID"
                )
            except worker.OutboundWorkerError as exc:
                raise worker.OutboundPostAdmissionError(
                    "outbound execution ended after media authorization"
                ) from exc

        with patch.object(worker, "originate_job", fail_with_cause):
            result = asyncio.run(outbound.run_once())

        payload = transport.requests[0][2]
        self.assertEqual(payload["error"], "OutboundPostAdmissionError:OutboundWorkerError")
        self.assertNotIn("caller ID", payload["error"])
        self.assertFalse(payload["retry"])
        self.assertEqual(result["error_type"], "OutboundPostAdmissionError")

    def test_unchained_job_error_reports_the_bare_class_name(self):
        class _Transport:
            def __init__(self):
                self.requests = []

            def request(self, method, path, payload):
                self.requests.append((method, path, payload))
                return {"status": "failed"}

        transport = _Transport()
        outbound = worker.OutboundWorker(_settings())
        outbound.worker_transport = transport
        outbound.claim = lambda: _job()  # type: ignore[method-assign]

        async def fail_plain(*_args, **_kwargs):
            raise worker.OutboundWorkerError("outbound job payload must be an object")

        with patch.object(worker, "originate_job", fail_plain):
            asyncio.run(outbound.run_once())

        self.assertEqual(transport.requests[0][2]["error"], "OutboundWorkerError")

    def test_job_validation_rejects_disabled_live_gate(self):
        with self.assertRaisesRegex(worker.OutboundWorkerError, "live-call gate"):
            worker._validate_job(_settings(), _job(live_call_gate={"enabled": False}))

    def test_job_trunk_name_overrides_env_default(self):
        # The control plane threads the NUMBER's provider-account trunk name
        # through the job payload; the worker must dial out of that trunk,
        # not the env default.
        events = []

        class _TwoTrunkSip(_Sip):
            async def list_outbound_trunk(self, _request):
                return _Message(items=[
                    _Message(name=worker.OUTBOUND_TRUNK_NAME, sip_trunk_id="STO_DEFAULT", numbers=["15551230000"]),
                    _Message(name="openphonex-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["15551230000"]),
                ])

        settings = _settings()
        api = _Api(events)
        api.client.sip = _TwoTrunkSip(events)
        trunk = asyncio.run(worker._select_outbound_trunk(
            api.client, api, settings, "+15551230000", trunk_name="openphonex-didlogic-outbound",
        ))
        self.assertEqual(trunk.sip_trunk_id, "STO_DIDLOGIC")
        # Default stays the env-configured trunk when the job omits the name.
        default_trunk = asyncio.run(worker._select_outbound_trunk(api.client, api, settings, "+15551230000"))
        self.assertEqual(default_trunk.sip_trunk_id, "STO_DEFAULT")

    def test_job_trunk_name_still_requires_caller_id_membership(self):
        # A trunk name cannot smuggle a caller ID the trunk does not carry.
        events = []

        class _WrongNumberSip(_Sip):
            async def list_outbound_trunk(self, _request):
                return _Message(items=[
                    _Message(name="openphonex-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["94711111001"]),
                ])

        api = _Api(events)
        api.client.sip = _WrongNumberSip(events)
        with self.assertRaisesRegex(worker.OutboundWorkerError, "exactly one configured outbound trunk"):
            asyncio.run(worker._select_outbound_trunk(
                api.client, api, _settings(), "+15551230000", trunk_name="openphonex-didlogic-outbound",
            ))

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

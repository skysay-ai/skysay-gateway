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

    def admit_local_outbound(self, **kwargs):
        self.events.append(("admitted", kwargs.get("to_e164"), kwargs.get("from_phone_number")))
        return (
            {
                "call_id": "call_local",
                "agent_id": "agent_1",
                "number_id": "number_1",
                "from_phone_number": "+15551230000",
                "to_phone_number": "+37255509999",
                "outbound_trunk_name": "",
            },
            "tca_local",
        )

    def session_for_local_outbound(self, dial_plan, token):
        self.events.append(("session", dial_plan["call_id"]))
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


class LocallyOriginatedCallTests(unittest.TestCase):
    """A box that asks for the call it is about to place, with one credential."""

    ENVIRON = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "key",
        "LIVEKIT_API_SECRET": "secret",
        "OPENPHONEX_CONTROL_PLANE_URL": "https://api.openphonex.com",
        "OPENPHONEX_GATEWAY_ID": "gw_1",
        "OPENPHONEX_GATEWAY_API_TOKEN": "tgw_secret",
        "OPENPHONEX_ORGANIZATION_ID": "org_1",
    }

    def test_a_deployment_needs_no_organization_key_to_start(self):
        settings = worker.OutboundWorkerSettings.from_environ(self.ENVIRON)
        self.assertEqual(settings.worker_token, "")

    def test_the_job_queue_still_says_which_credential_it_needs(self):
        settings = worker.OutboundWorkerSettings.from_environ(self.ENVIRON)
        with self.assertRaisesRegex(worker.OutboundWorkerError, "OPENPHONEX_WORKER_API_TOKEN"):
            worker._transports(settings)

    def test_admission_precedes_media_dispatch_and_the_dial(self):
        events = []
        _Adapter.events = events
        api = _Api(events)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_gateway_transport", return_value=object()), \
             patch.object(worker, "_duration_limits", return_value={}):
            result = asyncio.run(
                worker.originate_local_call(
                    _settings(worker_token=""),
                    to_e164="+37255509999",
                    from_phone_number="+15551230000",
                    api_module=api,
                )
            )

        self.assertEqual(result["call_status"], "answered")
        self.assertEqual(result["call_id"], "call_local")
        order = [item[0] if isinstance(item, tuple) else item for item in events]
        self.assertLess(order.index("admitted"), order.index("authorization_consumed"))
        self.assertLess(order.index("authorization_consumed"), order.index("room"))
        self.assertLess(order.index("dispatch"), order.index("dial"))
        dispatch = next(item for item in events if isinstance(item, tuple) and item[0] == "dispatch")
        self.assertIn('"call_id": "call_local"', dispatch[3])
        self.assertNotIn("tca_local", dispatch[3])
        self.assertNotIn("tgw_secret", dispatch[3])

    def test_the_execution_gate_is_enforced_by_this_path_itself(self):
        """`docker compose run` ignores profiles, so the env gate is the gate."""
        events = []
        _Adapter.events = events
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_gateway_transport", return_value=object()):
            with self.assertRaisesRegex(worker.OutboundWorkerError, "OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND"):
                asyncio.run(
                    worker.originate_local_call(
                        _settings(worker_token="", execute=False),
                        to_e164="+37255509999",
                        from_phone_number="+15551230000",
                    )
                )
        # Nothing was asked of the control plane, so nothing was reserved.
        self.assertEqual(events, [])

    def test_a_failed_dial_releases_the_reservation_for_the_admitted_call(self):
        events = []
        _Adapter.events = events
        api = _Api(events, fail=True)
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_gateway_transport", return_value=object()), \
             patch.object(worker, "_duration_limits", return_value={}):
            with self.assertRaises(worker.OutboundPostAdmissionError):
                asyncio.run(
                    worker.originate_local_call(
                        _settings(worker_token=""),
                        to_e164="+37255509999",
                        from_phone_number="+15551230000",
                        api_module=api,
                    )
                )
        self.assertIn("sip_failed_reported", events)
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "delete_room" for item in events))

    def test_a_session_failure_after_admission_is_still_reported_against_the_call(self):
        """The call already holds money once admission returns."""
        events = []

        class _BrokenSessionAdapter(_Adapter):
            def session_for_local_outbound(self, dial_plan, token):
                raise RuntimeError("realtime config rejected")

            def report_failed_before_media(self, **kwargs):
                self.events.append(("sip_failed_reported", kwargs["call_id"]))
                return {"canonical_type": "call.failed", "billing": {"status": "released"}}

        _BrokenSessionAdapter.events = events
        with patch.object(worker, "Gate2ControlPlaneAdapter", _BrokenSessionAdapter), \
             patch.object(worker, "_gateway_transport", return_value=object()):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    worker.originate_local_call(
                        _settings(worker_token=""),
                        to_e164="+37255509999",
                        from_phone_number="+15551230000",
                    )
                )
        self.assertIn(("sip_failed_reported", "call_local"), events)

    def test_the_originate_command_refuses_a_destination_that_is_not_e164(self):
        with patch.dict("os.environ", self.ENVIRON, clear=False):
            self.assertEqual(worker.main(["originate", "--to", "0037255509999", "--purpose", "support", "--reason", "confirm"]), 1)

    def test_the_originate_command_refuses_when_no_caller_id_is_configured(self):
        environ = dict(self.ENVIRON)
        environ["SCH236_TEST_DID_E164"] = ""
        with patch.dict("os.environ", environ, clear=False):
            self.assertEqual(worker.main(["originate", "--to", "+37255509999", "--purpose", "support", "--reason", "confirm"]), 1)

    def test_the_originate_command_defaults_the_caller_id_to_the_served_number(self):
        environ = dict(self.ENVIRON)
        environ["SCH236_TEST_DID_E164"] = "+15551230000,+37255501234"
        environ["OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND"] = "1"
        placed = {}

        async def _placed(settings, **kwargs):
            placed.update(kwargs)
            return {"executed": True, "call_id": "call_local", "secrets_redacted": True}

        with patch.dict("os.environ", environ, clear=False), \
             patch.object(worker, "originate_local_call", _placed):
            self.assertEqual(
                worker.main(
                    ["originate", "--to", "+37255509999", "--purpose", "support", "--reason", "confirm"]
                ),
                0,
            )
        self.assertEqual(placed["to_e164"], "+37255509999")
        self.assertEqual(placed["from_phone_number"], "+15551230000")
        self.assertEqual(placed["request"]["purpose"], "support")
        self.assertEqual(placed["request"]["user_authorization"]["reason"], "confirm")
        self.assertNotIn("organization_id", placed["request"])

    def test_the_originate_command_will_not_invent_a_purpose_or_a_reason(self):
        with patch.dict("os.environ", self.ENVIRON, clear=False):
            with self.assertRaises(SystemExit):
                worker.main(["originate", "--to", "+37255509999"])
            with self.assertRaises(SystemExit):
                worker.main(["originate", "--to", "+37255509999", "--purpose", "support"])


if __name__ == "__main__":
    unittest.main()

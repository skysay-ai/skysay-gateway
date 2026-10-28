from __future__ import annotations

import asyncio
import contextlib
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

    async def list_participants(self, request):
        # The dispatched agent joins at once; the dial waits for it.
        self.events.append(("agent_joined", request.room))
        return _Message(participants=[_Message(kind=4, identity="agent-AJ_test")])


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
            "ListParticipantsRequest",
        ):
            setattr(self, name, _Message)


def _settings(**overrides):
    values = dict(
        livekit_url="ws://127.0.0.1:7880",
        livekit_api_key="key",
        livekit_api_secret="secret",
        control_plane_url="https://api.skysay.ai",
        gateway_id="gw_1",
        gateway_token="skygw_faketoken",
        worker_token="sky_faketoken",
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


def _job_outcome_requests(transport):
    """The complete/fail reports, ignoring the loop's own heartbeat."""

    return [entry for entry in transport.requests if "/worker-jobs/" in entry[1]]


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
        self.assertNotIn("skygw_faketoken", dispatch[3])
        self.assertNotIn("sky_faketoken", dispatch[3])

    def _originate_with_join(self, list_participants):
        events = []
        _Adapter.events = events
        api = _Api(events)
        api.client.room.list_participants = list_participants
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_transports", return_value=(object(), object())), \
             patch.object(worker, "_duration_limits", return_value={}), \
             patch.object(worker, "AGENT_JOIN_TIMEOUT_SECONDS", 0.2):
            with self.assertRaises(worker.OutboundPostAdmissionError):
                asyncio.run(asyncio.wait_for(worker.originate_job(_settings(), _job(), api_module=api), timeout=5))
        return events

    def test_a_participant_list_that_never_answers_fails_the_call_without_a_dial(self):
        # livekit-api sends each request with no timeout of its own.
        async def never_answers(_request):
            await asyncio.Event().wait()

        events = self._originate_with_join(never_answers)
        self.assertFalse(any(isinstance(item, tuple) and item[0] == "dial" for item in events))
        self.assertIn("sip_failed_reported", events)
        self.assertTrue(any(isinstance(item, tuple) and item[0] == "delete_room" for item in events))

    def test_an_agent_reported_after_the_deadline_is_not_dialled(self):
        async def answers_late(_request):
            await asyncio.sleep(0.5)
            return _Message(participants=[_Message(kind=4, identity="agent-AJ_test")])

        events = self._originate_with_join(answers_late)
        self.assertFalse(any(isinstance(item, tuple) and item[0] == "dial" for item in events))
        self.assertIn("sip_failed_reported", events)

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
        outcomes = _job_outcome_requests(transport)
        self.assertEqual(len(outcomes), 1, "exactly one terminal settle, never a redial")
        payload = outcomes[0][2]
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
                    _Message(name="skysay-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["15551230000"]),
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
                _job(outbound_trunk_name="skysay-didlogic-outbound"),
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

        payload = _job_outcome_requests(transport)[0][2]
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

        self.assertEqual(_job_outcome_requests(transport)[0][2]["error"], "OutboundWorkerError")

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
                    _Message(name="skysay-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["15551230000"]),
                ])

        settings = _settings()
        api = _Api(events)
        api.client.sip = _TwoTrunkSip(events)
        trunk = asyncio.run(worker._select_outbound_trunk(
            api.client, api, settings, "+15551230000", trunk_name="skysay-didlogic-outbound",
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
                    _Message(name="skysay-didlogic-outbound", sip_trunk_id="STO_DIDLOGIC", numbers=["94711111001"]),
                ])

        api = _Api(events)
        api.client.sip = _WrongNumberSip(events)
        with self.assertRaisesRegex(worker.OutboundWorkerError, "exactly one configured outbound trunk"):
            asyncio.run(worker._select_outbound_trunk(
                api.client, api, _settings(), "+15551230000", trunk_name="skysay-didlogic-outbound",
            ))

    def test_settings_keep_live_execution_off_by_default(self):
        environ = {
            "LIVEKIT_URL": "ws://127.0.0.1:7880",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
            "SKYSAY_CONTROL_PLANE_URL": "https://api.skysay.ai",
            "SKYSAY_GATEWAY_ID": "gw_1",
            "SKYSAY_GATEWAY_API_TOKEN": "skygw_faketoken",
            "SKYSAY_WORKER_API_TOKEN": "sky_faketoken",
            "SKYSAY_ORGANIZATION_ID": "org_1",
        }
        self.assertFalse(worker.OutboundWorkerSettings.from_environ(environ).execute)


class LocallyOriginatedCallTests(unittest.TestCase):
    """A box that asks for the call it is about to place, with one credential."""

    ENVIRON = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "key",
        "LIVEKIT_API_SECRET": "secret",
        "SKYSAY_CONTROL_PLANE_URL": "https://api.skysay.ai",
        "SKYSAY_GATEWAY_ID": "gw_1",
        "SKYSAY_GATEWAY_API_TOKEN": "skygw_faketoken",
        "SKYSAY_ORGANIZATION_ID": "org_1",
    }

    def test_a_deployment_needs_no_organization_key_to_start(self):
        settings = worker.OutboundWorkerSettings.from_environ(self.ENVIRON)
        self.assertEqual(settings.worker_token, "")

    def test_the_job_queue_still_says_which_credential_it_needs(self):
        settings = worker.OutboundWorkerSettings.from_environ(self.ENVIRON)
        with self.assertRaisesRegex(worker.OutboundWorkerError, "SKYSAY_WORKER_API_TOKEN"):
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
        self.assertNotIn("skygw_faketoken", dispatch[3])

    def test_the_execution_gate_is_enforced_by_this_path_itself(self):
        """`docker compose run` ignores profiles, so the env gate is the gate."""
        events = []
        _Adapter.events = events
        with patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter), \
             patch.object(worker, "_gateway_transport", return_value=object()):
            with self.assertRaisesRegex(worker.OutboundWorkerError, "SKYSAY_EXECUTE_LIVEKIT_OUTBOUND"):
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
        environ["SKYSAY_EXECUTE_LIVEKIT_OUTBOUND"] = "1"
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


class ContractRefusalTests(unittest.TestCase):
    """The control plane refuses a worker on its contract before admission;
    the job fails once, names the refusal, and never enters the retry loop."""

    def _run(self, reason: str, mode: str = "organization"):
        from livekit_agent.control_plane import ControlPlaneRejected

        class _Transport:
            def __init__(self):
                self.requests = []

            def request(self, method, path, payload):
                self.requests.append((method, path, payload))
                return {"status": "failed"}

        transport = _Transport()
        outbound = worker.OutboundWorker(_settings(worker_mode=mode))
        outbound.worker_transport = transport
        outbound.gateway_transport = transport
        outbound.claim = lambda: _job()  # type: ignore[method-assign]

        async def refused(*_args, **_kwargs):
            # Raised the way UrllibJsonTransport._send raises it: from the
            # HTTP error the control plane answered with, which is what the
            # job's error code then carries beside the refusal reason.
            import io
            import urllib.error

            http_error = urllib.error.HTTPError("http://127.0.0.1/v1/x", 409, "Conflict", {}, io.BytesIO(b"{}"))
            raise ControlPlaneRejected("control plane returned HTTP 409: refused", status_code=409, reason=reason) from http_error

        with patch.object(worker, "originate_job", refused):
            result = asyncio.run(outbound.run_once())
        return result, _job_outcome_requests(transport)

    def test_a_contract_refusal_fails_the_job_once_and_names_the_check(self) -> None:
        for reason in ("worker_contract_required", "call_contract_missing"):
            with self.subTest(reason=reason):
                result, outcomes = self._run(reason)
                self.assertEqual(result["job_status"], "failed")
                self.assertEqual(len(outcomes), 1)
                payload = outcomes[0][2]
                self.assertFalse(payload["retry"])
                self.assertEqual(payload["max_attempts"], 1)
                self.assertEqual(payload["error"], f"ControlPlaneRejected:HTTPError:{reason}")

    def test_another_control_plane_rejection_keeps_its_retry(self) -> None:
        # The contract set is exact: a different refusal is treated as before.
        result, outcomes = self._run("route_missing")
        self.assertEqual(result["job_status"], "failed")
        payload = outcomes[0][2]
        self.assertTrue(payload["retry"])
        self.assertEqual(payload["error"], "ControlPlaneRejected:HTTPError")

    def test_a_wrapped_refusal_is_still_recognised(self) -> None:
        from livekit_agent.control_plane import ControlPlaneRejected

        try:
            raise worker.OutboundPostAdmissionError("ended after media authorization") from ControlPlaneRejected(
                "refused", status_code=409, reason="call_contract_missing"
            )
        except worker.OutboundPostAdmissionError as exc:
            self.assertEqual(worker.contract_refusal(exc), "call_contract_missing")
            self.assertEqual(worker._job_error_code(exc), "OutboundPostAdmissionError:ControlPlaneRejected:call_contract_missing")


if __name__ == "__main__":
    unittest.main()


class SupervisorStallTests(unittest.TestCase):
    """Organization mode has no overall originate deadline: every wait is bounded on its own."""

    def _supervise(self, scenario, *, stall=None, originate=None, cap=None):
        events = []
        _Adapter.events = events
        api = _Api(events)
        entered = asyncio.Event()
        if stall is not None:
            service, method = stall

            async def never_answers(*_args, **_kwargs):
                entered.set()
                await asyncio.Event().wait()

            stalled = getattr(api.client, service)
            real = getattr(stalled, method)
            calls = []

            async def first_call_never_answers(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return await never_answers()
                return await real(*args, **kwargs)

            setattr(stalled, method, first_call_never_answers)
        real_originate = worker.originate_job

        async def originate_with_fakes(settings, job, **kwargs):
            if originate is not None:
                return await originate(entered, job, **kwargs)
            return await real_originate(settings, job, api_module=api, **kwargs)

        claims, settled = [], []

        def claim():
            claims.append(1)
            return _job() | {"id": f"job_{len(claims)}"} if len(claims) <= 2 else None

        def settle(job, action, payload):
            settled.append((job["id"], action))
            return {"id": job["id"], "status": "succeeded"}

        patches = [
            patch.object(worker, "Gate2ControlPlaneAdapter", _Adapter),
            patch.object(worker, "_transports", return_value=(object(), object())),
            patch.object(worker, "_duration_limits", return_value={}),
            patch.object(worker, "originate_job", originate_with_fakes),
            patch.object(worker, "LIVEKIT_RPC_TIMEOUT_SECONDS", 0.2),
        ]
        if cap is not None:
            patches.append(patch.object(worker, "CLAIM_PACING_CAP_SECONDS", cap))
        dialer = worker.OutboundWorker(_settings(max_in_flight=3, poll_seconds=0.01))
        self.assertEqual(dialer.settings.worker_mode, "organization")
        dialer.claim, dialer.heartbeat, dialer.settle = claim, (lambda _status: True), settle
        stop = asyncio.Event()

        async def run():
            supervisor = asyncio.create_task(dialer._supervise(stop))
            try:
                return await scenario(supervisor, stop, entered, claims, settled, events)
            finally:
                stop.set()
                await asyncio.wait_for(supervisor, timeout=5)

        with contextlib.ExitStack() as stack:
            for active in patches:
                stack.enter_context(active)
            return asyncio.run(run())

    def test_a_stalled_livekit_call_before_the_join_ends_its_job_and_the_next_is_claimed(self):
        for stall in (("sip", "list_outbound_trunk"), ("room", "create_room"), ("agent_dispatch", "create_dispatch")):
            with self.subTest(stall=stall):

                async def scenario(supervisor, stop, entered, claims, settled, events):
                    await asyncio.wait_for(entered.wait(), timeout=5)
                    for _ in range(300):
                        if len([item for item in settled if item[1] != "renew"]) == 2:
                            break
                        await asyncio.sleep(0.01)
                    return claims, settled, events

                claims, settled, events = self._supervise(scenario, stall=stall)
                self.assertGreaterEqual(len(claims), 2, "one stalled call stopped the originator claiming")
                outcomes = dict(item for item in settled if item[1] != "renew")
                self.assertEqual(outcomes, {"job_1": "fail", "job_2": "complete"})
                self.assertEqual(len([item for item in events if isinstance(item, tuple) and item[0] == "dial"]), 1)

    def test_a_stop_during_a_stall_drains_within_the_bound(self):
        async def scenario(supervisor, stop, entered, claims, settled, events):
            await asyncio.wait_for(entered.wait(), timeout=5)
            stop.set()
            # The stalled call is never answered; its own deadline ends it.
            await asyncio.wait_for(supervisor, timeout=2)
            return settled, events

        settled, events = self._supervise(scenario, stall=("room", "create_room"))
        self.assertIn(("job_1", "fail"), settled)
        self.assertFalse(any(isinstance(item, tuple) and item[0] == "dial" for item in events))

    def test_a_job_that_neither_joins_nor_ends_holds_up_the_next_claim_only_until_the_cap(self):
        hold: list[asyncio.Event] = []

        async def originate(entered, job, **_kwargs):
            if job["id"] == "job_1":
                gate = asyncio.Event()
                hold.append(gate)
                entered.set()
                await gate.wait()
            return {"executed": True, "call_status": "answered"}

        async def scenario(supervisor, stop, entered, claims, settled, events):
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.sleep(0.05)
            before_cap = len(claims)
            await asyncio.sleep(0.4)
            after_cap = len(claims)
            hold[0].set()
            return before_cap, after_cap

        before_cap, after_cap = self._supervise(scenario, originate=originate, cap=0.2)
        self.assertEqual(before_cap, 1, "the next job was claimed before the first one's agent")
        self.assertGreaterEqual(after_cap, 2, "one job stopped the originator claiming past the cap")

    def test_the_pacing_cap_covers_every_bounded_step_before_a_join(self):
        self.assertGreaterEqual(
            worker.CLAIM_PACING_CAP_SECONDS,
            4 * 10 + 3 * worker.LIVEKIT_RPC_TIMEOUT_SECONDS + worker.AGENT_JOIN_TIMEOUT_SECONDS,
        )


class HostedOriginateDeadlineTests(unittest.TestCase):
    """The hosted deadline must outlast every bounded pre-dial step AND the dial's own timeout."""

    def test_the_dial_keeps_its_whole_window_after_every_pre_dial_step_runs_to_its_bound(self):
        import inspect

        from livekit_agent.control_plane import UrllibJsonTransport

        http = inspect.signature(UrllibJsonTransport).parameters["timeout_seconds"].default
        pre_dial = 2 * http + 3 * worker.LIVEKIT_RPC_TIMEOUT_SECONDS + worker.AGENT_JOIN_TIMEOUT_SECONDS
        self.assertEqual(worker.PRE_DIAL_BOUND_SECONDS, pre_dial)
        for ringing in (15, 45, 120):
            with self.subTest(ringing=ringing):
                settings = _settings(ringing_timeout_seconds=ringing)
                # The dial's own client timeout is ringing + 10 (create_sip_participant).
                left_for_the_dial = worker._originate_deadline_seconds(settings) - pre_dial
                self.assertGreater(left_for_the_dial, ringing + 10, "the deadline could cut off a ring timeout")

    def test_the_hosted_originate_runs_under_that_deadline(self):
        # Wiring at the call site: a hosted job's originate is cancelled at
        # _originate_deadline_seconds, not at any other figure.
        settled = []
        started = []

        async def never_finishes(*_args, **_kwargs):
            started.append(1)
            await asyncio.Event().wait()

        dialer = worker.OutboundWorker(_settings(worker_mode="platform"))
        job = _job() | {"claim_target": "platform", "lease_token": "lt_1", "locked_by": "worker_1",
                        "lease_expires_at": "2999-01-01T00:00:00Z"}
        dialer.settle = lambda job, action, payload: settled.append((action, payload)) or dict(job)

        async def renew_forever(_job):
            await asyncio.Event().wait()

        with patch.object(worker, "originate_job", never_finishes), \
             patch.object(worker, "_validate_job", lambda *_: {}), \
             patch.object(worker, "_originate_deadline_seconds", lambda _settings: 0.2), \
             patch.object(dialer, "_renew_until_settled", renew_forever):
            result = asyncio.run(asyncio.wait_for(dialer.process_job(job, True), timeout=5))
        self.assertEqual(started, [1])
        self.assertEqual(result["error_type"], "TimeoutError", result)
        failures = [payload for action, payload in settled if action == "fail"]
        self.assertEqual(len(failures), 1, settled)
        self.assertFalse(failures[0]["retry"], "a hosted originate cut off by its deadline must never be redialled")

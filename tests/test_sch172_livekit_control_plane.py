from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from livekit_agent.control_plane import (
    ControlPlaneRejected,
    Gate2CallSession,
    Gate2ControlPlaneAdapter,
    IncompleteArtifactEvidence,
    SessionStateError,
    UrllibJsonTransport,
    _SameOriginRedirectHandler,
    _validate_pr79_artifact_set,
    parse_outbound_dispatch,
)
from livekit_agent.evidence_contract import semantic_response_audio_policy
from livekit_agent.call_telemetry import CallTelemetryCollector


class FakeTransport:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self.calls.append((method, path, payload))
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {path}")
        return self.responses.pop(0)


def _config(*, tools: list[str] | None = None, language: str = "en") -> dict:
    return {
        "agent_id": "agent_1",
        "organization_id": "org_1",
        "project_id": "project_1",
        "language": language,
        "system_prompt": "Help the caller.",
        "first_message": "Hello",
        "tools": tools or [],
        "recording_enabled": True,
        "transcriber": {"registry_name": "deepgram", "model": "nova-3", "language": language},
        "model": {"registry_name": "gemini", "model": "gemini-2.5-flash-lite"},
        "voice": {
            "registry_name": "elevenlabs",
            "model": "eleven_flash_v2_5" if language == "en" else "eleven_v3",
            "voice_id": "voice_1",
            "language": language,
        },
    }


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(8_000)
        recording.writeframes(b"\x10\x00" * 800)


class Gate2ControlPlaneAdapterTests(unittest.TestCase):
    def _adapter(self, transport: FakeTransport) -> Gate2ControlPlaneAdapter:
        return Gate2ControlPlaneAdapter(
            gateway_id="gateway_1",
            transport=transport,
        )

    def test_outbound_dispatch_contains_identifiers_but_no_credentials(self) -> None:
        metadata = json.dumps({
            "schema": "openphonex.outbound-dispatch.v1",
            "call_id": "call_1",
            "agent_id": "agent_1",
            "from_number_id": "number_1",
            "from_e164": "+15551230000",
            "to_e164": "+37255509999",
            "participant_identity": "pstn-call_1",
        })
        dispatch = parse_outbound_dispatch(metadata)
        self.assertEqual(dispatch.call_id, "call_1")
        self.assertIsNone(parse_outbound_dispatch(""))
        with self.assertRaisesRegex(ValueError, "omitted required fields"):
            parse_outbound_dispatch(json.dumps({"schema": "openphonex.outbound-dispatch.v1"}))

    def test_policy_authorization_and_config_precede_media(self) -> None:
        transport = FakeTransport(
            [
                {
                    "authorized": True,
                    "call_id": "call_1",
                    "token": "tca_once",
                    "decision": {"decision": "allowed"},
                },
                _config(tools=["get_coverage"]),
                {"call_id": "call_1", "status": "consumed"},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1",
            agent_id="agent_1",
            from_number_id="number_1",
            to_e164="+15551230001",
            metadata={"purpose": "support"},
        )

        self.assertFalse(session.media_admitted)
        self.assertEqual(session.call_id, "call_1")
        self.assertEqual(session.authorization_call_id, "call_1")
        self.assertEqual(session.call_authorization_token, "tca_once")
        self.assertNotIn("tca_once", repr(session))
        self.assertEqual(
            [(method, path) for method, path, _ in transport.calls],
            [
                ("POST", "/v1/gateway-deployments/gateway_1/call-authorizations"),
                ("GET", "/v1/agents/agent_1/realtime-config"),
            ],
        )
        session.admit_media()
        self.assertTrue(session.media_admitted)
        self.assertEqual(session.audit_events[:4], ["authorization_allowed", "realtime_config_loaded", "authorization_consumed", "media_admitted"])

    def _admitted(self, **overrides) -> dict:
        response = {
            "authorized": True,
            "gateway_id": "gateway_1",
            "authorization": {
                "id": "tca_row",
                "token": "tca_once",
                "expires_at": "2099-01-01T00:00:00Z",
                "call_id": "call_1",
            },
            "dial_plan": {
                "call_id": "call_1",
                "agent_id": "agent_1",
                "number_id": "number_1",
                "from_phone_number": "+15551230000",
                "to_phone_number": "+37255509999",
                "outbound_trunk_name": "openphonex-didlogic-pa_1",
            },
            "call": {"id": "call_1", "status": "dialing"},
        }
        response.update(overrides)
        return response

    def test_local_admission_returns_a_dial_plan_and_a_single_use_token(self) -> None:
        transport = FakeTransport([self._admitted(), _config()])
        adapter = self._adapter(transport)
        dial_plan, token = adapter.admit_local_outbound(
            to_e164="+37255509999", from_phone_number="+15551230000"
        )

        self.assertEqual(token, "tca_once")
        self.assertEqual(dial_plan["call_id"], "call_1")
        self.assertEqual(dial_plan["outbound_trunk_name"], "openphonex-didlogic-pa_1")
        method, path, body = transport.calls[0]
        self.assertEqual((method, path), ("POST", "/v1/gateway-deployments/gateway_1/outbound-calls"))
        # The workspace is never named by the box: the control plane reads it
        # from this deployment's own token.
        self.assertNotIn("organization_id", body)
        self.assertNotIn("project_id", body)
        self.assertEqual(body["to"], "+37255509999")

        session = adapter.session_for_local_outbound(dial_plan, token)
        self.assertFalse(session.media_admitted)
        self.assertEqual(session.call_authorization_token, "tca_once")
        self.assertEqual(session.to_e164, "+37255509999")
        self.assertEqual(
            [(method, path) for method, path, _ in transport.calls],
            [
                ("POST", "/v1/gateway-deployments/gateway_1/outbound-calls"),
                ("GET", "/v1/agents/agent_1/realtime-config"),
            ],
        )

    def test_local_admission_refuses_a_dial_plan_for_another_destination(self) -> None:
        transport = FakeTransport([self._admitted()])
        with self.assertRaisesRegex(ControlPlaneRejected, "different destination"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255500000", from_phone_number="+15551230000"
            )

    def test_local_admission_refuses_a_dial_plan_for_another_caller_id(self) -> None:
        transport = FakeTransport([self._admitted()])
        with self.assertRaisesRegex(ControlPlaneRejected, "different caller id"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255509999", from_phone_number="+15559999999"
            )
        transport = FakeTransport([self._admitted()])
        with self.assertRaisesRegex(ControlPlaneRejected, "different caller id"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255509999", from_number_id="number_2"
            )

    def test_local_admission_refuses_a_token_bound_to_another_call(self) -> None:
        response = self._admitted()
        response["authorization"]["call_id"] = "call_other"
        transport = FakeTransport([response])
        with self.assertRaisesRegex(ControlPlaneRejected, "not bound to the admitted call"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255509999", from_phone_number="+15551230000"
            )

    def test_local_admission_refuses_an_unauthorized_or_tokenless_answer(self) -> None:
        transport = FakeTransport(
            [{"authorized": False, "authorization": None, "dial_plan": None, "call": {"status": "policy_blocked"}}]
        )
        with self.assertRaisesRegex(ControlPlaneRejected, "did not authorize origination"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255509999", from_phone_number="+15551230000"
            )

        response = self._admitted()
        response["authorization"] = {"id": "tca_row", "token": "", "call_id": "call_1"}
        transport = FakeTransport([response])
        with self.assertRaisesRegex(ControlPlaneRejected, "omitted its single-use token"):
            self._adapter(transport).admit_local_outbound(
                to_e164="+37255509999", from_phone_number="+15551230000"
            )

    def test_local_admission_needs_a_caller_id_before_it_asks_for_anything(self) -> None:
        transport = FakeTransport([])
        with self.assertRaisesRegex(ControlPlaneRejected, "needs a caller id"):
            self._adapter(transport).admit_local_outbound(to_e164="+37255509999")
        self.assertEqual(transport.calls, [])

    def test_platform_release_canary_session_is_non_billing_and_non_customer_state(self) -> None:
        transport = FakeTransport(
            [
                {
                    "purpose": "platform_release_canary",
                    "source_e164": "+15551230001",
                    "target_e164": "+15551230000",
                    "config": _config(tools=[]),
                }
            ]
        )
        adapter = Gate2ControlPlaneAdapter(
            gateway_id="gateway_1",
            transport=FakeTransport([]),
            platform_transport=transport,
        )
        session = adapter.prepare_platform_release_canary(
            {
                "purpose": "platform_release_canary",
                "source_e164": "+15551230001",
                "target_e164": "+15551230000",
            },
            "carrier-leg-1",
        )
        self.assertTrue(session.call_id.startswith("platform-release-canary-"))
        session.admit_media()
        self.assertEqual(session.tool_definitions(), [])
        self.assertEqual(session.lease_component_credentials(), {})
        self.assertEqual(session.report_lifecycle(event_type="sip.answered")["status"], "platform_release_canary")
        self.assertEqual(session.complete(source_path=Path("."))["artifact_import"]["status"], "platform_release_canary")
        # The carrier leg identity is sent so a retransmitted dispatch replays
        # the claim its own leg spent. It is the only field: the worker
        # describes which leg it is, never what the route should be.
        self.assertEqual(
            transport.calls,
            [
                (
                    "POST",
                    "/v1/platform-media/release-canary/route",
                    {"provider_call_id": "carrier-leg-1"},
                )
            ],
        )

    def test_the_release_route_runs_without_caller_side_recording(self) -> None:
        """The platform route has no recording, and must still be runnable.

        Caller-side recording is a customer obligation: it exists so a
        tenant's call produces the consent-bearing artifact set the platform
        keeps. The release route creates no call record and no artifact, so
        the control plane returns ``recording_enabled: False`` deliberately.
        Rejecting that config rejects the only config this route can return,
        and the leg dies about a second after the agent joins — after the
        claim has already been spent, which makes the failure look like a
        successful admission in every record outside the worker.
        """
        config = _config(tools=[])
        config["recording_enabled"] = False
        adapter = Gate2ControlPlaneAdapter(
            gateway_id="gateway_1",
            transport=FakeTransport([]),
            platform_transport=FakeTransport(
                [
                    {
                        "purpose": "platform_release_canary",
                        "source_e164": "+15551230001",
                        "target_e164": "+15551230000",
                        "config": config,
                    }
                ]
            ),
        )
        session = adapter.prepare_platform_release_canary(
            {
                "purpose": "platform_release_canary",
                "source_e164": "+15551230001",
                "target_e164": "+15551230000",
            },
            "carrier-leg-1",
        )
        session.admit_media()
        self.assertTrue(session.media_admitted)

    def test_a_customer_call_still_requires_caller_side_recording(self) -> None:
        """Relaxing the release route must not relax the customer route.

        The two share one validator, and only one of them owes a caller an
        artifact.
        """
        config = _config()
        config["recording_enabled"] = False
        transport = FakeTransport(
            [
                {
                    "id": "call_in_1",
                    "status": "queued",
                    "route": {
                        "routed": True,
                        "agent_id": "agent_1",
                        "organization_id": "org_1",
                        "project_id": "project_1",
                    },
                },
                config,
            ]
        )
        with self.assertRaises(ControlPlaneRejected):
            self._adapter(transport).prepare_inbound(
                to_e164="+15551230001",
                from_e164="+15557654321",
                provider_call_id="didww-call-1",
            )

    def test_a_refused_release_claim_never_downgrades_to_a_customer_call(self) -> None:
        """An unarmed leg must fail the whole preparation, not fall back.

        The reserved tuple reaching ``prepare_inbound`` is exactly the defect
        this route exists to prevent: it would create the customer call,
        billing entry, dashboard row, and usage record that a release proof
        must never produce.
        """

        class RefusingTransport(FakeTransport):
            def request(self, method: str, path: str, payload: dict | None = None) -> dict:
                self.calls.append((method, path, payload))
                raise ControlPlaneRejected("control plane returned HTTP 403: not armed")

        transport = RefusingTransport([])
        inbound = FakeTransport([])
        adapter = Gate2ControlPlaneAdapter(
            gateway_id="gateway_1",
            transport=FakeTransport([]),
            call_transport=inbound,
            platform_transport=transport,
        )
        with self.assertRaises(ControlPlaneRejected):
            adapter.prepare_platform_release_canary(
                {
                    "purpose": "platform_release_canary",
                    "source_e164": "+15551230001",
                    "target_e164": "+15551230000",
                },
                "carrier-leg-1",
            )
        self.assertEqual([path for _, path, _ in transport.calls], ["/v1/platform-media/release-canary/route"])
        self.assertEqual(inbound.calls, [])

    def test_denied_authorization_never_fetches_config_or_admits_media(self) -> None:
        transport = FakeTransport([{"authorized": False, "token": None, "decision": {"decision": "blocked"}}])
        with self.assertRaises(ControlPlaneRejected):
            self._adapter(transport).prepare_outbound(
                call_id="call_1",
                agent_id="agent_1",
                from_number_id="number_1",
                to_e164="+15551230002",
                metadata={"purpose": "cold_sales"},
            )
        self.assertEqual(len(transport.calls), 1)

    def test_denied_pre_media_call_releases_its_reservation(self) -> None:
        transport = FakeTransport(
            [
                {
                    "canonical_type": "call.failed",
                    "billing": {"status": "released"},
                }
            ]
        )

        result = self._adapter(transport).report_failed_before_media(
            call_id="call_denied",
            event_id="call_denied:policy-denied",
            reason="authorization_denied",
        )

        self.assertEqual(result["billing"]["status"], "released")
        self.assertEqual(
            transport.calls,
            [
                (
                    "POST",
                    "/v1/gateway-deployments/gateway_1/call-lifecycle-events",
                    {
                        "event_id": "call_denied:policy-denied",
                        "call_id": "call_denied",
                        "event_type": "sip.failed",
                        "source_component": "livekit_sip",
                        "details": {"reason": "authorization_denied"},
                    },
                )
            ],
        )

    def test_managed_pre_media_failure_uses_platform_transport(self) -> None:
        gateway_transport = FakeTransport([])
        platform_transport = FakeTransport(
            [{"canonical_type": "call.failed", "billing": {"status": "no_reservation"}}]
        )
        adapter = Gate2ControlPlaneAdapter(
            gateway_id="gateway_shared",
            transport=gateway_transport,
            platform_transport=platform_transport,
        )

        adapter.report_failed_before_media(
            call_id="call_customer",
            event_id="call_customer:failed",
            reason="worker_failed",
        )

        self.assertEqual(gateway_transport.calls, [])
        self.assertEqual(
            platform_transport.calls[0][1],
            "/v1/gateway-deployments/gateway_shared/call-lifecycle-events",
        )

    def test_authorized_outbound_resume_fetches_call_bound_config(self) -> None:
        config = {**_config(), "call_id": "call_1", "direction": "outbound"}
        transport = FakeTransport([config])

        session = self._adapter(transport).resume_authorized_outbound(
            call_id="call_1",
            agent_id="agent_1",
            from_number_id="number_1",
            to_e164="+15551230001",
        )

        self.assertTrue(session.media_admitted)
        self.assertEqual(session.call_id, "call_1")
        self.assertEqual(
            transport.calls,
            [("GET", "/v1/calls/call_1/realtime-config", None)],
        )
        # Already consumed by the originator: no second consume request.
        session.admit_media()
        self.assertEqual(len(transport.calls), 1)

    def test_inbound_route_policy_resolves_agent_before_media(self) -> None:
        transport = FakeTransport(
            [
                {
                    "id": "call_in_1",
                    "status": "queued",
                    "route": {
                        "routed": True,
                        "agent_id": "agent_1",
                        "organization_id": "org_1",
                        "project_id": "project_1",
                    },
                },
                _config(),
            ]
        )
        adapter = Gate2ControlPlaneAdapter(
            gateway_id="gateway_1",
            organization_id="org_1",
            transport=transport,
        )
        session = adapter.prepare_inbound(
            to_e164="+15551230001",
            from_e164="+15557654321",
            provider_call_id="didww-call-1",
        )
        self.assertEqual(session.call_id, "call_in_1")
        self.assertFalse(session.media_admitted)
        self.assertEqual(
            [(method, path) for method, path, _ in transport.calls],
            [
                ("POST", "/v1/calls/inbound/route"),
                ("GET", "/v1/calls/call_in_1/realtime-config"),
            ],
        )
        session.admit_media()
        self.assertTrue(session.media_admitted)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.calls[0][2]["organization_id"], "org_1")

    def test_tool_allowlist_blocks_model_input_before_mcp(self) -> None:
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(tools=["get_coverage"]),
                {"call_id": "call_1", "status": "consumed"},
                {"jsonrpc": "2.0", "id": "call_1:tools", "result": {"tools": [
                    {"name": "get_coverage", "inputSchema": {"type": "object"}},
                    {"name": "send_sms", "inputSchema": {"type": "object"}},
                ]}},
                {"result": {"coverage": []}},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        self.assertEqual([tool["name"] for tool in session.tool_definitions()], ["get_coverage"])
        result = session.invoke_tool("get_coverage", {"countries": ["EE"]})
        self.assertIn("result", result)
        with self.assertRaises(ControlPlaneRejected):
            session.invoke_tool("send_sms", {"to": "+15551230001", "body": "no"})
        self.assertEqual([call[1] for call in transport.calls].count("/mcp"), 1)
        self.assertIn("/v1/calls/call_1/agent-tool-calls", [call[1] for call in transport.calls])
        self.assertIn("/v1/gateway-call-authorizations/consume", [call[1] for call in transport.calls])

    def test_structured_builtin_tool_does_not_require_an_mcp_listing(self) -> None:
        transport = FakeTransport([])
        session = Gate2CallSession(
            transport=transport,
            authorization_transport=transport,
            tool_transport=transport,
            call_id="call_builtin",
            gateway_id="gateway_1",
            authorization_call_id="",
            call_authorization_token="",
            config=_config(tools=[{"type": "builtin", "name": "end_call"}]),
            media_admitted=True,
        )

        self.assertEqual(
            session.tool_definitions(),
            [
                {
                    "name": "end_call",
                    "description": "End the phone call politely once the caller's needs are met.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Final goodbye sentence to speak before hanging up.",
                            }
                        },
                    },
                }
            ],
        )
        self.assertEqual(transport.calls, [])

    def test_tool_execution_emits_only_safe_telemetry_correlation(self) -> None:
        transport = FakeTransport(
            [{"tool": "get_coverage", "ok": True, "telemetry_correlation_id": "audit_tool_1"}]
        )
        session = Gate2CallSession(
            transport=transport,
            authorization_transport=transport,
            tool_transport=transport,
            call_id="call_tool_telemetry",
            gateway_id="gateway_1",
            authorization_call_id="",
            call_authorization_token="",
            config=_config(tools=["get_coverage"]),
            media_admitted=True,
        )
        observations: list[dict] = []
        session.telemetry_observer = lambda **event: observations.append(event)

        result = session.invoke_tool(
            "get_coverage",
            {"country": "EE", "private_query": "never serialize this text"},
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["name"], "get_coverage")
        self.assertEqual(observations[0]["audit_correlation_id"], "audit_tool_1")
        self.assertGreater(observations[0]["input_characters"], 0)
        self.assertNotIn("private_query", json.dumps(observations[0]))
        self.assertNotIn("never serialize", json.dumps(observations[0]))

    def test_inbound_rejects_realtime_config_from_a_different_tenant(self) -> None:
        wrong = _config()
        wrong["organization_id"] = "org_other"
        transport = FakeTransport(
            [
                {
                    "id": "call_in_wrong",
                    "status": "queued",
                    "route": {
                        "routed": True,
                        "agent_id": "agent_1",
                        "organization_id": "org_1",
                        "project_id": "project_1",
                    },
                },
                wrong,
            ]
        )
        with self.assertRaisesRegex(ControlPlaneRejected, "organization"):
            self._adapter(transport).prepare_inbound(
                to_e164="+15551230001",
                from_e164="+15557654321",
                provider_call_id="didww-call-wrong-org",
            )

    def test_completion_requires_pr79_artifacts_and_duration_provenance(self) -> None:
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {"canonical_type": "call.answered", "duplicate": False},
                {
                    "canonical_type": "call.ended",
                    "duplicate": False,
                    "billing": {
                        "status": "settled",
                        "duration_seconds": 12.0,
                        "settle_pricing": {"duration_sources": {"call_events": 12.0}},
                    },
                },
                {"imported_count": 1, "skipped_count": 0},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        session.report_lifecycle(
            event_id="sip-answer-1",
            event_type="sip.answered",
            source_component="livekit_sip",
        )
        ended = session.report_lifecycle(
            event_id="sip-end-1",
            event_type="sip.ended",
            source_component="livekit_sip",
        )
        self.assertEqual(ended["billing"]["settle_pricing"]["duration_sources"], {"call_events": 12.0})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Restated for the mid-reply hangup fix: incomplete evidence is
            # now uploaded and classified rather than discarded, but an
            # evidence directory with NOTHING in it has no partial record to
            # preserve, so completion still refuses and leaves the session
            # un-completed for a retry.
            with self.assertRaises(IncompleteArtifactEvidence):
                session.complete(
                    source_path=root,
                    duration_sources={"sip_participant": 12.0, "recording": 11.5},
                )
            self.assertEqual(
                [path for _method, path, *_ in transport.calls if "artifacts" in path], []
            )
            self.assertNotIn("artifacts_imported", session.audit_events)

            for name in ("caller.wav", "agent.wav", "mixed.wav", "turn-001-caller.wav", "turn-001-agent.wav"):
                _write_wav(root / name)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "turns": [{"user": "Tere", "assistant": "Tere!"}],
                        "semantic_response_audio_policy": semantic_response_audio_policy(),
                    }
                )
            )
            collector = CallTelemetryCollector(
                call_id="call_1",
                organization_id="org_1",
                project_id="project_1",
            )
            collector.observe_user(text="private caller text", metrics={"transcription_delay": 0.2})
            collector.observe_assistant(text="private assistant text", metrics={"e2e_latency": 0.4})
            collector.write(root / "call-telemetry.json")
            result = session.complete(
                source_path=root,
                duration_sources={"sip_participant": 12.0, "room_track": 11.8, "recording": 11.5},
            )

        self.assertEqual(result["billing_duration_seconds"], 12.0)
        self.assertEqual(result["billing_duration_source"], "sip_participant")
        artifact_payload = transport.calls[-1][2]
        self.assertEqual(transport.calls[-1][:2], ("POST", "/v1/calls/call_1/artifacts"))
        self.assertNotIn("duration_seconds", artifact_payload)
        self.assertIn("duration-provenance.json", {part["filename"] for part in artifact_payload["parts"]})
        self.assertIn("call-telemetry.json", {part["filename"] for part in artifact_payload["parts"]})
        self.assertNotIn("private caller text", json.dumps(artifact_payload))
        self.assertNotIn("private assistant text", json.dumps(artifact_payload))
        self.assertEqual(session.audit_events[-1], "artifacts_imported")

    def test_component_usage_is_posted_before_terminal_sip_settlement(self) -> None:
        """Teardown ordering contract: telemetry/artifacts are later, but the
        native billing report must reach the control plane before sip.ended."""
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {"call_id": "call_1", "duplicate": False},
                {"canonical_type": "call.ended", "duplicate": False, "billing": {"status": "settled"}},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        session.report_component_usage(
            {
                "source": "livekit_agent_sdk",
                "stt_audio_seconds": 1,
                "llm_input_tokens": 1,
                "llm_cached_input_tokens": 0,
                "llm_reasoning_tokens": 0,
                "llm_output_tokens": 1,
                "tts_characters": 1,
                "tts_audio_seconds": 1,
                "tool_units": {},
            }
        )
        session.report_lifecycle(
            event_id="sip-end-1",
            event_type="sip.ended",
            source_component="livekit_sip",
        )
        paths = [path for _method, path, _payload in transport.calls]
        self.assertLess(paths.index("/v1/calls/call_1/billing-usage"), paths.index("/v1/gateway-deployments/gateway_1/call-lifecycle-events"))
        self.assertIn("component_usage_reported", session.audit_events)

    def test_terminal_payment_due_is_an_authoritative_billing_verdict(self) -> None:
        """A completed call with an explicit above-reserve balance must not
        be treated as a media/lifecycle failure by the worker gate."""
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {
                    "canonical_type": "call.ended",
                    "duplicate": False,
                    "billing": {
                        "status": "payment_due",
                        "settled_amount_minor": 108,
                        "uncollected_amount_minor": 400,
                    },
                },
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        result = session.report_lifecycle(
            event_id="sip-payment-due", event_type="sip.ended", source_component="livekit_sip"
        )
        self.assertEqual(result["billing"]["status"], "payment_due")
        self.assertTrue(session._lifecycle_finalized)

    def test_component_usage_failed_post_is_retried_then_forwarded_with_terminal_event(self) -> None:
        """The normal separate endpoint has a bounded retry; a failure is not
        silently lost in worker memory before terminal settlement."""

        class UsageUnavailableTransport(FakeTransport):
            def request(self, method: str, path: str, payload: dict | None = None) -> dict:
                if path == "/v1/calls/call_1/billing-usage":
                    self.calls.append((method, path, payload))
                    raise OSError("transient control-plane failure")
                return super().request(method, path, payload)

        usage = {
            "source": "livekit_agent_sdk",
            "stt_audio_seconds": 1,
            "llm_input_tokens": 1,
            "llm_cached_input_tokens": 0,
            "llm_reasoning_tokens": 0,
            "llm_output_tokens": 1,
            "tts_characters": 1,
            "tts_audio_seconds": 1,
            "tool_units": {},
        }
        transport = UsageUnavailableTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {"canonical_type": "call.ended", "duplicate": False, "billing": {"status": "settled"}},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        with self.assertRaises(OSError):
            session.report_component_usage(usage)
        session.report_lifecycle(
            event_id="sip-end-1", event_type="sip.ended", source_component="livekit_sip"
        )
        self.assertEqual(
            [path for _method, path, _payload in transport.calls].count("/v1/calls/call_1/billing-usage"),
            3,
        )
        terminal_payload = transport.calls[-1][2] or {}
        self.assertEqual(terminal_payload["component_usage"], usage)
        self.assertIn("component_usage_fallback_forwarded", session.audit_events)

    def test_component_usage_http_503_is_retried_and_kept_for_terminal_fallback(self) -> None:
        class Http503Transport(FakeTransport):
            def request(self, method: str, path: str, payload: dict | None = None) -> dict:
                if path == "/v1/calls/call_1/billing-usage":
                    self.calls.append((method, path, payload))
                    raise ControlPlaneRejected("control plane returned HTTP 503")
                return super().request(method, path, payload)

        usage = {
            "source": "livekit_agent_sdk",
            "stt_audio_seconds": 1,
            "llm_input_tokens": 1,
            "llm_cached_input_tokens": 0,
            "llm_reasoning_tokens": 0,
            "llm_output_tokens": 1,
            "tts_characters": 1,
            "tts_audio_seconds": 1,
            "tool_units": {},
        }
        transport = Http503Transport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {"canonical_type": "call.ended", "duplicate": False, "billing": {"status": "settled"}},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        session.admit_media()
        with self.assertRaisesRegex(ControlPlaneRejected, "503"):
            session.report_component_usage(usage)
        session.report_lifecycle(
            event_id="sip-end-503", event_type="sip.ended", source_component="livekit_sip"
        )
        self.assertEqual(
            [path for _method, path, _payload in transport.calls].count("/v1/calls/call_1/billing-usage"),
            3,
        )
        self.assertEqual((transport.calls[-1][2] or {})["component_usage"], usage)

    def test_short_call_completion_explicitly_allows_base_recording_without_turn_clips(self) -> None:
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
                {"call_id": "call_1", "status": "consumed"},
                {
                    "canonical_type": "call.ended",
                    "duplicate": False,
                    "billing": {"status": "settled"},
                },
                {"imported_count": 1, "skipped_count": 0},
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1",
            agent_id="agent_1",
            from_number_id="number_1",
            to_e164="+15551230001",
            metadata={},
        )
        session.admit_media()
        session.report_lifecycle(
            event_id="sip-end-1",
            event_type="sip.ended",
            source_component="livekit_sip",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("caller.wav", "agent.wav", "mixed.wav"):
                _write_wav(root / name)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "turns": [],
                        "semantic_response_audio_policy": semantic_response_audio_policy(),
                        "summary": {"turn_count": 0},
                    }
                )
            )

            result = session.complete(
                source_path=root,
                duration_sources={"sip_participant": 1.0, "recording": 1.0},
                require_semantic_turns=False,
            )

        self.assertEqual(result["billing_duration_seconds"], 1.0)
        uploaded_names = {part["filename"] for part in transport.calls[-1][2]["parts"]}
        self.assertEqual(
            uploaded_names,
            {
                "caller.wav",
                "agent.wav",
                "mixed.wav",
                "transcript.json",
                "duration-provenance.json",
                "evidence-status.json",
            },
        )
        self.assertEqual(result["evidence"]["status"], "complete")

    def test_strict_completion_rejects_base_recording_without_semantic_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("caller.wav", "agent.wav", "mixed.wav"):
                _write_wav(root / name)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "turns": [],
                        "semantic_response_audio_policy": semantic_response_audio_policy(),
                    }
                )
            )

            with self.assertRaises(IncompleteArtifactEvidence):
                _validate_pr79_artifact_set(root)

    def test_relaxed_completion_rejects_transcript_turn_without_audio_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("caller.wav", "agent.wav", "mixed.wav"):
                _write_wav(root / name)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "turns": [{"user": "Hello", "assistant": "Hi"}],
                        "semantic_response_audio_policy": semantic_response_audio_policy(),
                    }
                )
            )

            with self.assertRaises(IncompleteArtifactEvidence):
                _validate_pr79_artifact_set(root, require_semantic_turns=False)

    def test_relaxed_completion_rejects_audio_clips_without_transcript_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "caller.wav",
                "agent.wav",
                "mixed.wav",
                "turn-001-caller.wav",
                "turn-001-agent.wav",
            ):
                _write_wav(root / name)
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "turns": [],
                        "semantic_response_audio_policy": semantic_response_audio_policy(),
                    }
                )
            )

            with self.assertRaises(IncompleteArtifactEvidence):
                _validate_pr79_artifact_set(root, require_semantic_turns=False)

    def test_cleanup_is_idempotent_and_attempts_participant_then_room(self) -> None:
        transport = FakeTransport(
            [
                {"authorized": True, "call_id": "call_1", "token": "tca_once", "decision": {"decision": "allowed"}},
                _config(),
            ]
        )
        session = self._adapter(transport).prepare_outbound(
            call_id="call_1", agent_id="agent_1", from_number_id="number_1", to_e164="+15551230001", metadata={}
        )
        deleted: list[str] = []
        first = session.cleanup(
            remove_participant=lambda: deleted.append("participant"),
            delete_room=lambda: deleted.append("room"),
        )
        second = session.cleanup(
            remove_participant=lambda: deleted.append("participant-again"),
            delete_room=lambda: deleted.append("room-again"),
        )
        self.assertEqual(deleted, ["participant", "room"])
        self.assertTrue(first["clean"])
        self.assertEqual(second, {"clean": True, "already_clean": True})
        with self.assertRaises(SessionStateError):
            session.admit_media()


class UrllibJsonTransportHardeningTests(unittest.TestCase):
    """Bearer credentials only travel over https or verified loopback,
    and redirects can never replay the Authorization header off-origin."""

    @staticmethod
    def _transport(base_url: str) -> UrllibJsonTransport:
        return UrllibJsonTransport(base_url=base_url, bearer_token="token-1")

    def test_https_remote_base_url_is_accepted(self) -> None:
        # The production shape: https control plane on a remote host.
        transport = self._transport("https://api.example.com")
        self.assertEqual(transport.base_url, "https://api.example.com")

    def test_http_loopback_base_urls_are_accepted(self) -> None:
        # The other production shape: host-local control plane over loopback.
        for base_url in (
            "http://127.0.0.1:18765",
            "http://127.8.9.10:18765",
            "http://localhost:18765",
            "http://[::1]:18765",
        ):
            with self.subTest(base_url=base_url):
                self._transport(base_url)

    def test_http_non_loopback_base_urls_are_refused(self) -> None:
        for base_url in (
            "http://api.example.com",
            "http://10.0.0.5:18765",  # private LAN is still plaintext on a wire
            "http://192.168.1.20",
            "http://0.0.0.0:18765",
            "http://loopback.example.com",  # a NAME is not a verified loopback
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    self._transport(base_url)

    def test_non_http_schemes_are_refused(self) -> None:
        for base_url in ("ftp://api.example.com", "ws://127.0.0.1:7880", "api.example.com", ""):
            with self.subTest(base_url=base_url):
                with self.assertRaises(ValueError):
                    self._transport(base_url)

    # -- redirect policy ---------------------------------------------------- #

    @staticmethod
    def _redirect(source_url: str, location: str):
        import urllib.request

        handler = _SameOriginRedirectHandler()
        request = urllib.request.Request(source_url, method="GET")
        request.add_header("Authorization", "Bearer token-1")
        return handler.redirect_request(request, None, 302, "Found", {}, location)

    def test_same_origin_redirects_still_work(self) -> None:
        for location in (
            "https://api.example.com/v1/other",
            "https://api.example.com:443/v1/other",  # explicit default port == same origin
            "/v1/other",  # relative
        ):
            with self.subTest(location=location):
                redirected = self._redirect("https://api.example.com/v1/thing", location)
                self.assertIsNotNone(redirected)
                self.assertTrue(redirected.full_url.startswith("https://api.example.com"))

    def test_cross_host_redirect_is_refused(self) -> None:
        with self.assertRaises(ControlPlaneRejected):
            self._redirect("https://api.example.com/v1/thing", "https://evil.example.net/steal")

    def test_scheme_downgrade_redirect_is_refused(self) -> None:
        with self.assertRaises(ControlPlaneRejected):
            self._redirect("https://api.example.com/v1/thing", "http://api.example.com/v1/thing")

    def test_cross_port_redirect_is_refused(self) -> None:
        with self.assertRaises(ControlPlaneRejected):
            self._redirect("http://127.0.0.1:18765/v1/thing", "http://127.0.0.1:9999/v1/thing")

    # -- proxy policy ------------------------------------------------------- #
    # The default ProxyHandler only carries proxies when the environment sets
    # them, so these assertions build the transports WITH env proxies present.

    @staticmethod
    def _active_proxies(transport: UrllibJsonTransport) -> dict:
        import urllib.request

        merged: dict = {}
        for h in transport._opener.handlers:
            if isinstance(h, urllib.request.ProxyHandler):
                merged.update(h.proxies)
        return merged

    def test_loopback_transport_ignores_env_proxy(self) -> None:
        # A loopback control plane must never route its bearer through a proxy:
        # even with HTTP_PROXY set, the transport carries no active proxy map.
        import os
        from unittest import mock

        proxy = "http://proxy.example.com:8080"
        with mock.patch.dict(os.environ, {"HTTP_PROXY": proxy, "http_proxy": proxy}):
            transport = self._transport("http://127.0.0.1:18765")
            self.assertEqual(
                self._active_proxies(transport), {},
                "loopback transport must not honour env proxies for the bearer request",
            )

    def test_https_transport_keeps_default_proxy_behaviour(self) -> None:
        # For https we do not pin an empty proxy map: an operator's HTTPS_PROXY
        # must still apply (TLS keeps the token confidential end-to-end).
        import os
        from unittest import mock

        proxy = "http://proxy.example.com:8080"
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": proxy, "https_proxy": proxy}):
            transport = self._transport("https://api.example.com")
            self.assertIn(
                "https", self._active_proxies(transport),
                "https transport must keep the env HTTPS_PROXY active",
            )

    def test_loopback_request_ignores_env_proxy(self) -> None:
        # End-to-end: with HTTP_PROXY pointed at a dead address, a loopback
        # request still reaches the loopback server (proxy ignored). Without the
        # pin, urllib would try the proxy and fail.
        import os
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from unittest import mock

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *_a):  # noqa: D401
                return

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            proxy = "http://127.0.0.1:1"  # nothing listens here
            with mock.patch.dict(
                os.environ,
                {"HTTP_PROXY": proxy, "http_proxy": proxy, "ALL_PROXY": proxy},
            ):
                transport = self._transport(f"http://127.0.0.1:{port}")
                self.assertEqual(transport.request("GET", "/ok"), {"ok": True})
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

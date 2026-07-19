from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from livekit_agent.control_plane import (
    ControlPlaneRejected,
    Gate2ControlPlaneAdapter,
    IncompleteArtifactEvidence,
    SessionStateError,
    UrllibJsonTransport,
    _SameOriginRedirectHandler,
    parse_outbound_dispatch,
)
from livekit_agent.evidence_contract import semantic_response_audio_policy


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
                    "route": {"routed": True, "agent_id": "agent_1"},
                },
                _config(),
            ]
        )
        session = self._adapter(transport).prepare_inbound(
            organization_id="org_1",
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
                ("GET", "/v1/agents/agent_1/realtime-config"),
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
                {"jsonrpc": "2.0", "id": "call_1:1", "result": {"content": [{"json": {"coverage": []}}]}},
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
        self.assertEqual([call[1] for call in transport.calls].count("/mcp"), 2)
        self.assertIn("/v1/gateway-call-authorizations/consume", [call[1] for call in transport.calls])

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
            with self.assertRaises(IncompleteArtifactEvidence):
                session.complete(
                    source_path=root,
                    duration_sources={"sip_participant": 12.0, "recording": 11.5},
                )

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
        self.assertEqual(session.audit_events[-1], "artifacts_imported")

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

from __future__ import annotations

import json
import math
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path
from unittest.mock import patch

from openphonex_gateway.realtime.adapters import FakeRealtimeProviderConfig, FakeStreamingSTT, TemplateStreamingLLM
from openphonex_gateway.realtime.asterisk_external_media import (
    AriWebSocketEventClient,
    AsteriskExternalMediaConfig,
    AsteriskExternalMediaSession,
)
from openphonex_gateway.realtime.frames import AudioFrame, TextChunk, TranscriptEvent
from openphonex_gateway.realtime.worker import (
    DEFAULT_INBOUND_OPENING_UTTERANCE,
    RealtimeMediaWorker,
    RealtimeWorkerConfig,
    main as realtime_worker_main,
)


class FakeAri:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict | None, dict | None]] = []
        self.deletes: list[str] = []

    def post(self, path: str, params: dict | None = None, body: dict | None = None):
        self.posts.append((path, params, body))
        if path == "/bridges":
            return {"id": "bridge-1"}
        if path == "/channels/externalMedia":
            return {
                "id": "external-1",
                "channelvars": {
                    "UNICASTRTP_LOCAL_ADDRESS": "172.18.0.3",
                    "UNICASTRTP_LOCAL_PORT": "16000",
                },
            }
        return {}

    def delete(self, path: str):
        self.deletes.append(path)
        return {}


class MultiCallFakeAri:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict | None, dict | None]] = []
        self.deletes: list[str] = []
        self._bridge_index = 0
        self._external_index = 0

    def post(self, path: str, params: dict | None = None, body: dict | None = None):
        self.posts.append((path, params, body))
        if path == "/bridges":
            self._bridge_index += 1
            return {"id": f"bridge-{self._bridge_index}"}
        if path == "/channels/externalMedia":
            self._external_index += 1
            return {
                "id": f"external-{self._external_index}",
                "channelvars": {
                    "UNICASTRTP_LOCAL_ADDRESS": "172.18.0.3",
                    "UNICASTRTP_LOCAL_PORT": str(16000 + self._external_index),
                },
            }
        return {}

    def delete(self, path: str):
        self.deletes.append(path)
        return {}


class RealtimeAsteriskWorkerTests(unittest.TestCase):
    def test_asterisk_session_sets_up_bridge_and_external_media(self) -> None:
        ari = FakeAri()
        transport = _FakeTransport()
        config = AsteriskExternalMediaConfig(
            ari_user="ari",
            ari_password="secret",
            external_host="worker.internal",
            rtp_port=12000,
            asterisk_rtp_host="asterisk.internal",
        )
        session = AsteriskExternalMediaSession(ari=ari, config=config, transport=transport)  # type: ignore[arg-type]

        result = session.handle_event(
            {
                "type": "StasisStart",
                "channel": {"id": "primary-1", "name": "PJSIP/didww-0001", "state": "Ring"},
            }
        )

        self.assertEqual(result, "primary_started")
        self.assertEqual(session.state.bridge_id, "bridge-1")
        self.assertEqual(session.state.external_channel_id, "external-1")
        self.assertEqual(session.state.rtp_peer, ("asterisk.internal", 16000))
        self.assertEqual(transport.peer, ("asterisk.internal", 16000))
        self.assertIn(("/channels/primary-1/answer", None, None), ari.posts)
        self.assertIn(("/bridges/bridge-1/addChannel", {"channel": "primary-1"}, None), ari.posts)
        self.assertIn(("/bridges/bridge-1/addChannel", {"channel": "external-1"}, None), ari.posts)
        external_posts = [params for path, params, _body in ari.posts if path == "/channels/externalMedia"]
        self.assertEqual(external_posts[0]["external_host"], "worker.internal:12000")
        self.assertEqual(external_posts[0]["direction"], "both")

    def test_asterisk_session_cleans_external_media_on_primary_end(self) -> None:
        ari = FakeAri()
        session = AsteriskExternalMediaSession(
            ari=ari,
            config=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
        )
        session.handle_event({"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Up"}})
        result = session.handle_event({"type": "StasisEnd", "channel": {"id": "primary-1"}})
        self.assertEqual(result, "primary_ended")
        self.assertIn("/channels/external-1", ari.deletes)

    def test_asterisk_session_detects_outbound_direction_from_stasis_app_args(self) -> None:
        ari = FakeAri()
        session = AsteriskExternalMediaSession(
            ari=ari,
            config=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
        )
        action = session.handle_event(
            {
                "type": "StasisStart",
                "args": ["outbound"],
                "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Up"},
            }
        )
        self.assertEqual(action, "primary_started")
        self.assertEqual(session.state.call_direction, "outbound")
        self.assertEqual(session.state.call_direction_source, "stasis_app_args")
        self.assertTrue(session.state.primary_channel_is_up())

    def test_asterisk_session_channel_variable_direction_wins_over_app_args(self) -> None:
        ari = FakeAri()
        session = AsteriskExternalMediaSession(
            ari=ari,
            config=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
        )
        session.handle_event(
            {
                "type": "StasisStart",
                "args": ["outbound"],
                "channel": {
                    "id": "primary-1",
                    "name": "PJSIP/demo",
                    "state": "Up",
                    "variables": {"OPENPHONEX_CALL_DIRECTION": "inbound"},
                },
            }
        )
        self.assertEqual(session.state.call_direction, "inbound")
        self.assertEqual(session.state.call_direction_source, "channel_variable:OPENPHONEX_CALL_DIRECTION")

    def test_asterisk_session_ignores_ambiguous_app_args_and_tracks_channel_state(self) -> None:
        ari = FakeAri()
        session = AsteriskExternalMediaSession(
            ari=ari,
            config=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
        )
        session.handle_event(
            {
                "type": "StasisStart",
                "args": ["inbound,outbound"],
                "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Ringing"},
            }
        )
        self.assertEqual(session.state.call_direction, "inbound")
        self.assertEqual(session.state.call_direction_source, "default_inbound")
        self.assertFalse(session.state.primary_channel_is_up())
        action = session.handle_event(
            {"type": "ChannelStateChange", "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Up"}}
        )
        self.assertEqual(action, "primary_state_changed")
        self.assertTrue(session.state.primary_channel_is_up())

    def test_realtime_worker_status_uses_simple_env_names_and_redacts_secrets(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CALLS_DIR": "/tmp/openphonex-calls",
                "RECORDINGS_DIR": "/tmp/openphonex-recordings",
                "VOICE_RTP_PORT": "12042",
                "REALTIME_HEALTH_PORT": "12043",
                "ASTERISK_ARI_USER": "ari-user",
                "ASTERISK_ARI_PASSWORD": "super-secret",
                "REALTIME_PROVIDER_MODE": "fake",
                "REALTIME_MAX_PENDING_PHRASES": "2",
                "REALTIME_SEMANTIC_ENDPOINTING_ENABLED": "0",
                "REALTIME_SEMANTIC_ENDPOINTING_MIN_WORDS": "3",
                "REALTIME_SEMANTIC_ENDPOINTING_MIN_CHARS": "12",
                "REALTIME_SEMANTIC_ENDPOINTING_TERMINAL_PUNCTUATION": ".!",
            },
            clear=True,
        ):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        encoded = json.dumps(status)
        self.assertEqual(status["config"]["rtp"]["port"], 12042)
        self.assertEqual(status["config"]["rtp"]["port_end"], 12061)
        self.assertEqual(status["capacity"]["effective_call_capacity"], 20)
        self.assertEqual(status["config"]["health"]["port"], 12043)
        self.assertEqual(status["config"]["provider_mode"], "fake")
        self.assertEqual(status["config"]["max_pending_phrases"], 2)
        self.assertEqual(
            status["config"]["semantic_endpointing"],
            {
                "enabled": False,
                "min_words": 3,
                "min_chars": 12,
                "terminal_punctuation": ".!",
            },
        )
        self.assertNotIn("super-secret", encoded)
        self.assertEqual(status["pipeline"]["pipecat_dependency"], "none")
        self.assertEqual(status["pipeline"]["livekit_dependency"], "none")
        self.assertIn("phrase_queue_status", status["pipeline"])
        self.assertIn("semantic final-text endpointing disabled", status["pipeline"]["stt_status"])
        self.assertEqual(status["pipeline"]["provider_kind"], "fake")
        self.assertTrue(status["pipeline"]["ready_for_offline"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertFalse(status["pipeline"]["operator_approval"]["required"])
        self.assertEqual(status["pipeline"]["missing_live_env"], ["REALTIME_LIVE_ENABLED"])
        self.assertEqual(status["config"]["missing_live_env"], ["REALTIME_LIVE_ENABLED"])
        self.assertIn("serve_health", status["commands"])
        self.assertIn("--offline-streaming-overlap-check", status["commands"]["offline_streaming_overlap_check"])
        self.assertIn("--check-live-startup", status["commands"]["check_live_startup"])

    def test_realtime_worker_status_commands_use_openphonex_python_override(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(),
                provider_mode="fake",
            )
        )
        with patch.dict("os.environ", {"OPENPHONEX_PYTHON": "/opt/openphonex/bin/python"}, clear=True):
            commands = worker.status()["commands"]

        self.assertEqual(commands["status"], "/opt/openphonex/bin/python -m openphonex_gateway.realtime.worker --status")
        self.assertEqual(commands["serve_health"], "/opt/openphonex/bin/python -m openphonex_gateway.realtime.worker --serve-health")
        self.assertIn(
            "/opt/openphonex/bin/python -m openphonex_gateway.realtime.worker --env-file .env --check-live-startup",
            commands["check_live_startup"],
        )
        self.assertNotIn(" python -m openphonex_gateway.realtime.worker", json.dumps(commands))

    def test_realtime_worker_live_readiness_requires_live_gate_and_ari_credentials(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(),
                provider_mode="fake",
                live_enabled=False,
            )
        )
        status = worker.status()
        self.assertTrue(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(
            status["pipeline"]["missing_live_env"],
            ["REALTIME_LIVE_ENABLED", "ASTERISK_ARI_USER", "ASTERISK_ARI_PASSWORD"],
        )

        live_worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="fake",
                live_enabled=True,
            )
        )
        live_status = live_worker.status()
        self.assertTrue(live_status["ok"])
        self.assertTrue(live_status["pipeline"]["ready_for_live"])
        self.assertEqual(live_status["pipeline"]["missing_live_env"], [])
        self.assertFalse(live_status["pipeline"]["operator_approval"]["required"])

    def test_realtime_worker_live_placeholders_fail_closed(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(
                    ari_user="<asterisk-ari-user>",
                    ari_password="<set-secret>",
                    external_host="<realtime-worker-host-reachable-from-asterisk>",
                ),
                provider_mode="fake",
                live_enabled=True,
                operator_approval_required=True,
                operator_approval_record="/path/to/operator-approval.json",
                operator_approval_packet_fingerprint="<latest-approval-packet-fingerprint>",
                cutover_readiness_required=True,
                cutover_readiness_report="/path/to/validation-report.json",
                cutover_readiness_fingerprint="<latest-approval-packet-fingerprint>",
            )
        )

        status = worker.status()
        preflight = worker.live_startup_preflight()

        expected = [
            "ASTERISK_ARI_USER",
            "ASTERISK_ARI_PASSWORD",
            "ASTERISK_EXTERNAL_MEDIA_HOST",
            "REALTIME_OPERATOR_APPROVAL_RECORD",
            "REALTIME_APPROVAL_PACKET_FINGERPRINT",
            "REALTIME_CUTOVER_READINESS_REPORT",
            "REALTIME_CUTOVER_READINESS_FINGERPRINT",
        ]
        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(status["pipeline"]["missing_live_env"], [])
        self.assertEqual(status["pipeline"]["unresolved_live_env_placeholders"], expected)
        self.assertFalse(preflight["ok"])
        self.assertFalse(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["ari_connection_performed"])
        self.assertEqual(preflight["provider_network_actions_performed"], [])
        self.assertEqual(preflight["unresolved_live_env_placeholders"], expected)
        self.assertEqual(preflight["next_step"], "replace_live_env_placeholders")
        self.assertIn("Unresolved live env placeholders", preflight["live_error"]["message"])

    def test_realtime_worker_live_operator_approval_gate_fails_closed_when_required(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="fake",
                live_enabled=True,
                operator_approval_required=True,
            )
        )

        status = worker.status()

        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(
            status["pipeline"]["missing_live_env"],
            ["REALTIME_OPERATOR_APPROVAL_RECORD", "REALTIME_APPROVAL_PACKET_FINGERPRINT"],
        )
        approval = status["pipeline"]["operator_approval"]
        self.assertTrue(approval["required"])
        self.assertFalse(approval["ok"])
        self.assertEqual(approval["reason"], "operator_approval_record_not_configured")
        with self.assertRaisesRegex(RuntimeError, "operator_approval_record_not_configured"):
            worker.run_asterisk_event_loop([])

        preflight = worker.live_startup_preflight()
        self.assertFalse(preflight["ok"])
        self.assertFalse(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["live_actions_performed"])
        self.assertFalse(preflight["ari_connection_performed"])
        self.assertEqual(preflight["provider_network_actions_performed"], [])
        self.assertEqual(preflight["operator_approval"]["reason"], "operator_approval_record_not_configured")
        self.assertEqual(preflight["next_step"], "configure_live_worker_operator_approval_record")

    def test_realtime_worker_live_operator_approval_record_unlocks_live_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fingerprint = "b" * 64
            approval_record = root / "operator-approval.json"
            approval_record.write_text(
                json.dumps(
                    {
                        "name": "openphonex_live_measurement_operator_approval",
                        "approval_packet_fingerprint": fingerprint,
                        "approved_by": "operator@example.com",
                        "approved_at": "2026-06-22T10:00:00Z",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approved_directions": ["inbound", "outbound"],
                        "approval_note": "unit test live gate",
                    }
                )
            )
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    operator_approval_required=True,
                    operator_approval_record=str(approval_record),
                    operator_approval_packet_fingerprint=fingerprint,
                )
            )

            status = worker.status()
            result = worker.run_asterisk_event_loop([])
            preflight = worker.live_startup_preflight()

        approval = status["pipeline"]["operator_approval"]
        self.assertTrue(status["ok"], status)
        self.assertTrue(status["pipeline"]["ready_for_live"])
        self.assertEqual(status["pipeline"]["missing_live_env"], [])
        self.assertTrue(approval["ok"], approval)
        self.assertTrue(approval["fingerprint_match"])
        self.assertEqual(approval["fingerprint_short"], fingerprint[:12])
        self.assertEqual(approval["approved_directions"], ["inbound", "outbound"])
        self.assertEqual(approval["missing_directions"], [])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["events"], 0)
        self.assertTrue(preflight["ok"], preflight)
        self.assertTrue(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["ari_connection_performed"])
        self.assertEqual(preflight["next_step"], "start_live_worker")

    def test_realtime_worker_cutover_readiness_gate_fails_closed_when_required(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="fake",
                live_enabled=True,
                cutover_readiness_required=True,
            )
        )

        status = worker.status()

        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(
            status["pipeline"]["missing_live_env"],
            ["REALTIME_CUTOVER_READINESS_REPORT", "REALTIME_CUTOVER_READINESS_FINGERPRINT"],
        )
        cutover = status["pipeline"]["cutover_readiness"]
        self.assertTrue(cutover["required"])
        self.assertFalse(cutover["ok"])
        self.assertEqual(cutover["reason"], "cutover_readiness_report_not_configured")
        with self.assertRaisesRegex(RuntimeError, "cutover_readiness_report_not_configured"):
            worker.run_asterisk_event_loop([])

        preflight = worker.live_startup_preflight()
        self.assertFalse(preflight["ok"])
        self.assertFalse(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["live_actions_performed"])
        self.assertFalse(preflight["ari_connection_performed"])
        self.assertEqual(preflight["cutover_readiness"]["reason"], "cutover_readiness_report_not_configured")
        self.assertEqual(preflight["next_step"], "configure_realtime_cutover_readiness_report")

    def test_realtime_worker_cutover_readiness_report_unlocks_live_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fingerprint = "e" * 64
            report = root / "validation-report.json"
            report.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "cutover_readiness": {
                            "ready": True,
                            "next_step": "ready_to_cut_over_realtime_worker",
                            "legacy_replacement_target": "asterisk_voicebot_file_pipeline",
                            "missing": [],
                        },
                        "checks": [{"name": "realtime_cutover_readiness", "ok": True}],
                        "live_approval_packet": {"fingerprint": {"value": fingerprint}},
                    }
                )
            )
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    cutover_readiness_required=True,
                    cutover_readiness_report=str(report),
                    cutover_readiness_fingerprint=fingerprint,
                )
            )

            status = worker.status()
            result = worker.run_asterisk_event_loop([])
            preflight = worker.live_startup_preflight()

        cutover = status["pipeline"]["cutover_readiness"]
        self.assertTrue(status["ok"], status)
        self.assertTrue(status["pipeline"]["ready_for_live"])
        self.assertEqual(status["pipeline"]["missing_live_env"], [])
        self.assertTrue(cutover["ok"], cutover)
        self.assertTrue(cutover["ready"])
        self.assertTrue(cutover["fingerprint_match"])
        self.assertEqual(cutover["fingerprint_short"], fingerprint[:12])
        self.assertEqual(cutover["legacy_replacement_target"], "asterisk_voicebot_file_pipeline")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["events"], 0)
        self.assertTrue(preflight["ok"], preflight)
        self.assertTrue(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["ari_connection_performed"])
        self.assertTrue(preflight["cutover_readiness"]["ok"])
        self.assertEqual(preflight["next_step"], "start_live_worker")

    def test_realtime_worker_cutover_readiness_report_requires_matching_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / "validation-report.json"
            report.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "cutover_readiness": {"ready": True, "missing": []},
                        "checks": [{"name": "realtime_cutover_readiness", "ok": True}],
                        "live_approval_packet": {"fingerprint": {"value": "f" * 64}},
                    }
                )
            )
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    cutover_readiness_required=True,
                    cutover_readiness_report=str(report),
                    cutover_readiness_fingerprint="e" * 64,
                )
            )

            status = worker.status()
            cutover = status["pipeline"]["cutover_readiness"]
            self.assertFalse(status["ok"])
            self.assertFalse(cutover["ok"])
            self.assertFalse(cutover["fingerprint_match"])
            self.assertEqual(cutover["reason"], "cutover_readiness_fingerprint_mismatch")
            with self.assertRaisesRegex(RuntimeError, "cutover_readiness_fingerprint_mismatch"):
                worker.run_asterisk_event_loop([])

    def test_realtime_worker_check_live_startup_cli_does_not_connect_to_ari(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fingerprint = "d" * 64
            approval_record = root / "operator-approval.json"
            approval_record.write_text(
                json.dumps(
                    {
                        "name": "openphonex_live_measurement_operator_approval",
                        "approval_packet_fingerprint": fingerprint,
                        "approved_by": "operator@example.com",
                        "approved_at": "2026-06-22T10:00:00Z",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approved_directions": ["inbound", "outbound"],
                    }
                )
            )
            with (
                patch.dict(
                    "os.environ",
                    {
                        "CALLS_DIR": str(root),
                        "RECORDINGS_DIR": str(root),
                        "REALTIME_LIVE_ENABLED": "1",
                        "REALTIME_REQUIRE_OPERATOR_APPROVAL": "1",
                        "REALTIME_OPERATOR_APPROVAL_RECORD": str(approval_record),
                        "REALTIME_APPROVAL_PACKET_FINGERPRINT": fingerprint,
                        "ASTERISK_ARI_USER": "ari",
                        "ASTERISK_ARI_PASSWORD": "secret",
                    },
                    clear=True,
                ),
                patch("builtins.print") as printed,
            ):
                code = realtime_worker_main(["--check-live-startup"])

        self.assertEqual(code, 0)
        payload = json.loads(printed.call_args.args[0])
        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["ready_to_connect_ari"])
        self.assertFalse(payload["ari_connection_performed"])
        self.assertFalse(payload["live_actions_performed"])
        self.assertEqual(payload["provider_network_actions_performed"], [])
        self.assertEqual(payload["operator_approval"]["fingerprint_short"], fingerprint[:12])

    def test_realtime_worker_check_live_startup_cli_loads_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fingerprint = "f" * 64
            approval_record = root / "operator-approval.json"
            approval_record.write_text(
                json.dumps(
                    {
                        "name": "openphonex_live_measurement_operator_approval",
                        "approval_packet_fingerprint": fingerprint,
                        "approved_by": "operator@example.com",
                        "approved_at": "2026-06-22T10:00:00Z",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approved_directions": ["inbound", "outbound"],
                    }
                )
            )
            env_file = root / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        f"CALLS_DIR={root}",
                        f"RECORDINGS_DIR={root}",
                        "REALTIME_LIVE_ENABLED=1",
                        "REALTIME_REQUIRE_OPERATOR_APPROVAL=1",
                        f"REALTIME_OPERATOR_APPROVAL_RECORD={approval_record}",
                        f"REALTIME_APPROVAL_PACKET_FINGERPRINT={fingerprint}",
                        "ASTERISK_ARI_USER=ari",
                        "ASTERISK_ARI_PASSWORD=super-secret",
                    ]
                )
                + "\n"
            )
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("builtins.print") as printed,
            ):
                code = realtime_worker_main(["--env-file", str(env_file), "--check-live-startup"])

        self.assertEqual(code, 0)
        payload = json.loads(printed.call_args.args[0])
        encoded = json.dumps(payload)
        self.assertTrue(payload["ok"], payload)
        self.assertTrue(payload["ready_to_connect_ari"])
        self.assertFalse(payload["ari_connection_performed"])
        self.assertFalse(payload["live_actions_performed"])
        self.assertNotIn("super-secret", encoded)

    def test_realtime_worker_check_live_startup_cli_rejects_template_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env_file = root / "sch191-live-env.template"
            env_file.write_text(
                "\n".join(
                    [
                        f"CALLS_DIR={root}",
                        f"RECORDINGS_DIR={root}",
                        "REALTIME_LIVE_ENABLED=1",
                        "REALTIME_REQUIRE_OPERATOR_APPROVAL=1",
                        "REALTIME_OPERATOR_APPROVAL_RECORD=/path/to/operator-approval.json",
                        "REALTIME_APPROVAL_PACKET_FINGERPRINT=<latest-approval-packet-fingerprint>",
                        "REALTIME_REQUIRE_CUTOVER_READINESS=1",
                        "REALTIME_CUTOVER_READINESS_REPORT=/path/to/validation-report.json",
                        "REALTIME_CUTOVER_READINESS_FINGERPRINT=<latest-approval-packet-fingerprint>",
                        "ASTERISK_ARI_USER=<asterisk-ari-user>",
                        "ASTERISK_ARI_PASSWORD=<set-secret>",
                        "ASTERISK_EXTERNAL_MEDIA_HOST=<realtime-worker-host-reachable-from-asterisk>",
                    ]
                )
                + "\n"
            )
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("builtins.print") as printed,
            ):
                code = realtime_worker_main(["--env-file", str(env_file), "--check-live-startup"])

        payload = json.loads(printed.call_args.args[0])
        self.assertEqual(code, 1)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["ready_to_connect_ari"])
        self.assertFalse(payload["ari_connection_performed"])
        self.assertFalse(payload["live_actions_performed"])
        self.assertEqual(payload["provider_network_actions_performed"], [])
        self.assertEqual(payload["next_step"], "replace_live_env_placeholders")
        self.assertEqual(
            payload["unresolved_live_env_placeholders"],
            [
                "ASTERISK_ARI_USER",
                "ASTERISK_ARI_PASSWORD",
                "ASTERISK_EXTERNAL_MEDIA_HOST",
                "REALTIME_OPERATOR_APPROVAL_RECORD",
                "REALTIME_APPROVAL_PACKET_FINGERPRINT",
                "REALTIME_CUTOVER_READINESS_REPORT",
                "REALTIME_CUTOVER_READINESS_FINGERPRINT",
            ],
        )

    def test_realtime_worker_live_operator_approval_record_requires_requested_direction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fingerprint = "c" * 64
            approval_record = root / "operator-approval.json"
            approval_record.write_text(
                json.dumps(
                    {
                        "name": "openphonex_live_measurement_operator_approval",
                        "approval_packet_fingerprint": fingerprint,
                        "approved_by": "operator@example.com",
                        "approved_at": "2026-06-22T10:00:00Z",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "approved_directions": ["inbound"],
                    }
                )
            )
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    operator_approval_required=True,
                    operator_approval_record=str(approval_record),
                    operator_approval_packet_fingerprint=fingerprint,
                    operator_approval_required_directions=("inbound", "outbound"),
                )
            )

            status = worker.status()

        approval = status["pipeline"]["operator_approval"]
        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertFalse(approval["ok"])
        self.assertEqual(approval["reason"], "operator_approval_record_missing_required_directions")
        self.assertEqual(approval["missing_directions"], ["outbound"])

    def test_realtime_worker_health_server_exposes_redacted_status(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                health_bind="127.0.0.1",
                health_port=0,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari-user", ari_password="super-secret"),
                provider_mode="fake",
            )
        )
        server = worker.build_health_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            status = _read_json(f"http://{host}:{port}/status")
            health = _read_json(f"http://{host}:{port}/health?probe=1")
            self.assertTrue(status["ok"])
            self.assertEqual(status["service"], "openphonex-realtime-media-worker")
            self.assertEqual(health["service"], "openphonex-realtime-media-worker")
            self.assertEqual(status["config"]["health"]["port"], 0)
            self.assertNotIn("super-secret", json.dumps(status))
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(f"http://{host}:{port}/missing", timeout=3)
            self.assertEqual(raised.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_realtime_worker_gateway_health_report_payload_redacts_token_and_includes_capacity(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                rtp_port_end=12003,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari-user", ari_password="super-secret"),
                provider_mode="fake",
                control_plane_api_url="http://127.0.0.1:18765",
                gateway_deployment_id="gw_test",
                gateway_api_token="tai_gateway_secret_token",
                gateway_public_url="https://gateway.example/audio",
                gateway_local_status_url="http://127.0.0.1:18772/status",
            )
        )

        payload = worker.gateway_health_report_payload()
        encoded = json.dumps(payload)

        self.assertEqual(payload["public_url"], "https://gateway.example/audio")
        self.assertEqual(payload["local_status_url"], "http://127.0.0.1:18772/status")
        health = payload["realtime_worker_health"]
        self.assertTrue(health["ok"])
        body = health["body"]
        self.assertEqual(body["capacity"]["effective_call_capacity"], 4)
        self.assertIn("runtime_dependencies", body["config"])
        self.assertEqual(body["config"]["missing_runtime_dependencies"], [])
        self.assertTrue(body["config"]["runtime_dependencies"]["ok"])
        self.assertEqual(body["pipeline"]["missing_runtime_dependencies"], [])
        self.assertEqual(body["config"]["control_plane_reporter"]["gateway_deployment_id"], "gw_test")
        self.assertTrue(body["config"]["control_plane_reporter"]["api_token_configured"])
        self.assertTrue(body["control_plane_reporter"]["configured"])
        self.assertNotIn("super-secret", encoded)
        self.assertNotIn("tai_gateway_secret_token", encoded)

    def test_realtime_worker_posts_gateway_health_report_with_bearer_token(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url, payload, headers):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            return {"status": "connected", "id": "gw_test"}

        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari-user", ari_password="super-secret"),
                provider_mode="fake",
                control_plane_api_url="http://api.openphonex.test",
                gateway_deployment_id="gw_test",
                gateway_api_token="tai_gateway_secret_token",
            )
        )

        result = worker.post_gateway_health_report(post_json=fake_post)
        status = worker.status()

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["posted"])
        self.assertEqual(captured["url"], "http://api.openphonex.test/v1/gateway-deployments/gw_test/health-report")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer tai_gateway_secret_token")
        self.assertTrue(captured["payload"]["realtime_worker_health"]["ok"])
        self.assertEqual(status["control_plane_reporter"]["last_report"]["response_status"], "connected")
        self.assertNotIn("tai_gateway_secret_token", json.dumps(status))

    def test_realtime_worker_health_report_fails_closed_when_unconfigured(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari-user", ari_password="super-secret"),
                provider_mode="fake",
            )
        )

        result = worker.post_gateway_health_report()

        self.assertFalse(result["ok"])
        self.assertFalse(result["posted"])
        self.assertEqual(result["reason"], "missing OPENPHONEX_API_URL, OPENPHONEX_GATEWAY_ID, or OPENPHONEX_GATEWAY_API_TOKEN")
        self.assertFalse(worker.status()["control_plane_reporter"]["configured"])

    def test_realtime_worker_can_serve_asterisk_with_health_endpoint(self) -> None:
        events: list[str] = []

        class FakeServer:
            server_address = ("127.0.0.1", 18772)

            def __init__(self) -> None:
                self.stopped = False

            def serve_forever(self) -> None:
                events.append("health_started")
                while not self.stopped:
                    time.sleep(0.01)

            def shutdown(self) -> None:
                events.append("health_shutdown")
                self.stopped = True

            def server_close(self) -> None:
                events.append("health_closed")

        def fake_build_health_server(self):
            events.append("health_built")
            return FakeServer()

        def fake_serve_asterisk(self):
            for _ in range(50):
                if "health_started" in events:
                    break
                time.sleep(0.01)
            events.append("serve_asterisk")
            return {"ok": True, "events": 0}

        with (
            patch.object(RealtimeMediaWorker, "build_health_server", fake_build_health_server),
            patch.object(RealtimeMediaWorker, "serve_asterisk", fake_serve_asterisk),
            patch("builtins.print") as printed,
        ):
            code = realtime_worker_main(["--serve-health", "--serve-asterisk"])

        self.assertEqual(code, 0)
        self.assertIn("health_built", events)
        self.assertIn("health_started", events)
        self.assertIn("serve_asterisk", events)
        self.assertIn("health_shutdown", events)
        self.assertIn("health_closed", events)
        payload = json.loads(printed.call_args.args[0])
        self.assertTrue(payload["ok"])

    # --- GA-6 / SCH-217: ARI connect must be observable + resilient -----------
    #
    # Regression coverage for the container startup bug: previously
    # serve_asterisk_with_health bound health and then called serve_asterisk with
    # ZERO logging and ZERO retry, so a healthy connect, a hung connect, and a
    # crash-looping connect all looked identical from `docker logs` ("empty") and
    # any transient ARI-not-ready at startup tore the worker down invisibly.

    @staticmethod
    def _fake_health_server_cls():
        class FakeHealthServer:
            server_address = ("127.0.0.1", 18772)

            def __init__(self) -> None:
                self.stopped = False

            def serve_forever(self) -> None:
                while not self.stopped:
                    time.sleep(0.005)

            def shutdown(self) -> None:
                self.stopped = True

            def server_close(self) -> None:
                pass

        return FakeHealthServer

    @staticmethod
    def _fake_live_worker():
        return RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(
                    ari_url="http://asterisk:8088/ari",
                    ari_user="ari",
                    ari_password="secret",
                    app="voicebot",
                ),
                provider_mode="fake",
                live_enabled=True,
            )
        )

    def test_serve_asterisk_with_health_retries_and_logs_transient_ari_failure(self) -> None:
        logs: list[dict] = []
        worker = self._fake_live_worker()
        worker._log_sink = lambda line: logs.append(json.loads(line))
        FakeHealthServer = self._fake_health_server_cls()

        attempts = {"n": 0}

        def flaky_serve_asterisk(_self):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionRefusedError("ARI not ready yet")
            return {"ok": True, "events": 5}

        slept: list[float] = []
        with (
            patch.object(RealtimeMediaWorker, "build_health_server", lambda _self: FakeHealthServer()),
            patch.object(RealtimeMediaWorker, "serve_asterisk", flaky_serve_asterisk),
        ):
            result = worker.serve_asterisk_with_health(max_attempts=5, sleep=slept.append)

        self.assertTrue(result["ok"])
        self.assertEqual(result["events"], 5)
        self.assertEqual(attempts["n"], 3)

        events = [entry["event"] for entry in logs]
        self.assertIn("health_server_listening", events)
        self.assertEqual(events.count("ari_connect_attempt"), 3)
        self.assertEqual(events.count("ari_connect_error"), 2)
        self.assertIn("ari_event_loop_returned", events)
        self.assertIn("health_server_stopping", events)
        # Backoff was applied before each of the two retries, and it is positive.
        self.assertEqual(len(slept), 2)
        self.assertTrue(all(delay > 0 for delay in slept))
        # Errors were logged loudly with the failure class + retry intent.
        errors = [entry for entry in logs if entry["event"] == "ari_connect_error"]
        self.assertEqual(errors[0]["error"], "ConnectionRefusedError")
        self.assertTrue(errors[0]["will_retry"])

    def test_serve_asterisk_with_health_gives_up_and_logs_after_max_attempts(self) -> None:
        logs: list[dict] = []
        worker = self._fake_live_worker()
        worker._log_sink = lambda line: logs.append(json.loads(line))
        FakeHealthServer = self._fake_health_server_cls()

        calls = {"n": 0}

        def always_fail(_self):
            calls["n"] += 1
            raise ConnectionRefusedError("ARI down")

        with (
            patch.object(RealtimeMediaWorker, "build_health_server", lambda _self: FakeHealthServer()),
            patch.object(RealtimeMediaWorker, "serve_asterisk", always_fail),
        ):
            with self.assertRaises(ConnectionRefusedError):
                worker.serve_asterisk_with_health(max_attempts=3, sleep=lambda _s: None)

        self.assertEqual(calls["n"], 3)
        events = [entry["event"] for entry in logs]
        self.assertEqual(events.count("ari_connect_attempt"), 3)
        self.assertIn("ari_connect_giving_up", events)
        # Health server is still torn down cleanly in the finally, even on give-up.
        self.assertIn("health_server_stopping", events)

    def test_serve_asterisk_with_health_opens_ari_ws_and_logs_connect(self) -> None:
        """End-to-end: the worker actually OPENS the ARI WebSocket and logs the
        confirmed connect (this is the behavior the container was accused of
        never doing). A fake WS confirms the handshake, then ends the loop."""

        logs: list[dict] = []
        worker = self._fake_live_worker()
        worker._log_sink = lambda line: logs.append(json.loads(line))
        FakeHealthServer = self._fake_health_server_cls()

        class _StopServe(RuntimeError):
            pass

        class FakeWebSocketConnection:
            def __init__(self, url: str) -> None:
                self.url = url

            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> bool:
                return False

            def recv_text(self, *, timeout: float = 1.0):
                # Handshake already succeeded (on_open fired); end the loop.
                raise _StopServe("stop after connect")

        with (
            patch.object(RealtimeMediaWorker, "build_health_server", lambda _self: FakeHealthServer()),
            patch(
                "openphonex_gateway.realtime.asterisk_external_media._WebSocketConnection",
                FakeWebSocketConnection,
            ),
        ):
            with self.assertRaises(_StopServe):
                worker.serve_asterisk_with_health(max_attempts=1, sleep=lambda _s: None)

        events = [entry["event"] for entry in logs]
        self.assertIn("health_server_listening", events)
        self.assertIn("ari_ws_opening", events)
        # The decisive assertion: the ARI WS handshake completed and was logged.
        self.assertIn("ari_ws_connected", events)
        connected = next(entry for entry in logs if entry["event"] == "ari_ws_connected")
        self.assertEqual(connected["app"], "voicebot")
        self.assertEqual(connected["ari_url"], "http://asterisk:8088/ari")

    def test_ari_ws_client_fires_on_open_only_after_successful_handshake(self) -> None:
        calls: list[str] = []

        class FakeConn:
            def __init__(self, url: str) -> None:
                calls.append("connect")

            def __enter__(self):
                calls.append("enter")
                return self

            def __exit__(self, *_exc: object) -> bool:
                calls.append("exit")
                return False

            def recv_text(self, *, timeout: float = 1.0):
                return None

        stop = threading.Event()

        def on_open() -> None:
            calls.append("on_open")
            stop.set()  # end the loop right after the confirmed connect

        client = AriWebSocketEventClient(
            AsteriskExternalMediaConfig(ari_user="u", ari_password="p", app="voicebot")
        )
        with patch(
            "openphonex_gateway.realtime.asterisk_external_media._WebSocketConnection",
            FakeConn,
        ):
            produced = list(client.iter_events(stop, on_open=on_open))

        self.assertEqual(produced, [])
        self.assertEqual(calls, ["connect", "enter", "on_open", "exit"])

    def test_ari_ws_client_skips_on_open_when_handshake_fails(self) -> None:
        fired: list[str] = []

        class FailingConn:
            def __init__(self, url: str) -> None:
                pass

            def __enter__(self):
                raise RuntimeError("ARI WebSocket handshake failed: HTTP/1.1 401 Unauthorized")

            def __exit__(self, *_exc: object) -> bool:
                return False

        client = AriWebSocketEventClient(
            AsteriskExternalMediaConfig(ari_user="u", ari_password="p", app="voicebot")
        )
        with patch(
            "openphonex_gateway.realtime.asterisk_external_media._WebSocketConnection",
            FailingConn,
        ):
            with self.assertRaisesRegex(RuntimeError, "handshake failed"):
                list(client.iter_events(on_open=lambda: fired.append("on_open")))

        self.assertEqual(fired, [])

    def test_elevenlabs_provider_mode_reports_missing_env_and_fails_closed(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "elevenlabs_v3_tts"}, clear=True):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["provider_kind"], "elevenlabs_v3_tts")
        self.assertIn("ELEVENLABS_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE", status["config"]["missing_provider_env"])
        with self.assertRaisesRegex(RuntimeError, "Missing realtime provider env"):
            worker.run_offline_synthetic_call()

    def test_fake_provider_mode_does_not_require_provider_runtime_dependencies(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(),
                provider_mode="fake",
            )
        )
        with (
            patch("openphonex_gateway.realtime.worker.missing_deepgram_streaming_runtime_dependencies", return_value=["websocket-client"]),
            patch("openphonex_gateway.realtime.worker.missing_elevenlabs_streaming_runtime_dependencies", return_value=["ffmpeg"]),
        ):
            status = worker.status()

        self.assertTrue(status["ok"])
        self.assertEqual(status["config"]["runtime_dependencies"], {"ok": True, "missing": [], "required": []})
        self.assertEqual(status["config"]["missing_runtime_dependencies"], [])
        self.assertTrue(status["pipeline"]["ready_for_offline"])

    def test_deepgram_stt_provider_requires_websocket_runtime_dependency(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="fake",
                stt_provider="deepgram",
                deepgram_api_key="test-deepgram-key",
                live_enabled=True,
            )
        )
        with patch("openphonex_gateway.realtime.worker.missing_deepgram_streaming_runtime_dependencies", return_value=["websocket-client"]):
            status = worker.status()
            prewarm = worker.prewarm_provider_clients()
            preflight = worker.live_startup_preflight()

        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_offline"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(status["config"]["missing_provider_env"], [])
        self.assertEqual(status["config"]["missing_runtime_dependencies"], ["websocket-client"])
        runtime = status["config"]["runtime_dependencies"]
        self.assertFalse(runtime["ok"])
        self.assertEqual(runtime["missing"], ["websocket-client"])
        self.assertEqual(runtime["required"][0]["name"], "websocket-client")
        self.assertFalse(runtime["required"][0]["configured"])
        self.assertFalse(prewarm["ok"], prewarm)
        self.assertEqual(prewarm["reason"], "runtime_dependencies_missing")
        self.assertEqual(prewarm["missing_runtime_dependencies"], ["websocket-client"])
        self.assertFalse(preflight["ok"], preflight)
        self.assertFalse(preflight["provider_ready"])
        self.assertEqual(preflight["next_step"], "install_realtime_runtime_dependencies")
        self.assertEqual(preflight["missing_runtime_dependencies"], ["websocket-client"])
        with patch("openphonex_gateway.realtime.worker.missing_deepgram_streaming_runtime_dependencies", return_value=["websocket-client"]):
            with self.assertRaisesRegex(RuntimeError, "Missing realtime runtime dependencies: websocket-client"):
                worker.run_offline_synthetic_call()

    def test_elevenlabs_provider_mode_requires_ffmpeg_runtime_dependency(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="elevenlabs_v3_tts",
                elevenlabs_api_key="test-elevenlabs-key",
                elevenlabs_voice_id="voice-123",
                live_enabled=True,
            )
        )
        with patch("openphonex_gateway.realtime.worker.missing_elevenlabs_streaming_runtime_dependencies", return_value=["ffmpeg"]):
            status = worker.status()
            prewarm = worker.prewarm_provider_clients()
            preflight = worker.live_startup_preflight()

        self.assertFalse(status["ok"])
        self.assertFalse(status["pipeline"]["ready_for_offline"])
        self.assertFalse(status["pipeline"]["ready_for_live"])
        self.assertEqual(status["config"]["missing_provider_env"], [])
        self.assertEqual(status["config"]["missing_runtime_dependencies"], ["ffmpeg"])
        runtime = status["config"]["runtime_dependencies"]
        self.assertFalse(runtime["ok"])
        self.assertEqual(runtime["missing"], ["ffmpeg"])
        self.assertEqual(runtime["required"][0]["name"], "ffmpeg")
        self.assertFalse(runtime["required"][0]["configured"])
        self.assertFalse(prewarm["ok"], prewarm)
        self.assertEqual(prewarm["reason"], "runtime_dependencies_missing")
        self.assertEqual(prewarm["missing_runtime_dependencies"], ["ffmpeg"])
        self.assertFalse(preflight["ok"], preflight)
        self.assertFalse(preflight["provider_ready"])
        self.assertEqual(preflight["next_step"], "install_realtime_runtime_dependencies")
        self.assertEqual(preflight["missing_runtime_dependencies"], ["ffmpeg"])
        with patch("openphonex_gateway.realtime.worker.missing_elevenlabs_streaming_runtime_dependencies", return_value=["ffmpeg"]):
            with self.assertRaisesRegex(RuntimeError, "Missing realtime runtime dependencies: ffmpeg"):
                worker.run_offline_synthetic_call()

    def test_full_realtime_provider_mode_reports_all_runtime_dependencies(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="full_realtime",
                deepgram_api_key="test-deepgram-key",
                gemini_api_key="test-gemini-key",
                elevenlabs_api_key="test-elevenlabs-key",
                elevenlabs_voice_id="voice-123",
            )
        )
        with (
            patch("openphonex_gateway.realtime.worker.missing_deepgram_streaming_runtime_dependencies", return_value=["websocket-client"]),
            patch("openphonex_gateway.realtime.worker.missing_elevenlabs_streaming_runtime_dependencies", return_value=["ffmpeg"]),
        ):
            status = worker.status()

        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["missing_provider_env"], [])
        self.assertEqual(status["config"]["missing_runtime_dependencies"], ["websocket-client", "ffmpeg"])
        runtime = status["config"]["runtime_dependencies"]
        self.assertFalse(runtime["ok"])
        self.assertEqual(runtime["missing"], ["websocket-client", "ffmpeg"])
        self.assertEqual([item["name"] for item in runtime["required"]], ["websocket-client", "ffmpeg"])

    def test_gemini_provider_mode_reports_missing_env_and_fails_closed(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "gemini_llm_fake_tts"}, clear=True):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["provider_kind"], "gemini_llm_fake_tts")
        self.assertIn("GEMINI_API_KEY", status["config"]["missing_provider_env"])
        self.assertNotIn("ELEVENLABS_API_KEY", status["config"]["missing_provider_env"])
        with self.assertRaisesRegex(RuntimeError, "Missing realtime provider env"):
            worker.run_offline_synthetic_call()

    def test_deepgram_stt_provider_reports_missing_env_and_fails_closed(self) -> None:
        with patch.dict("os.environ", {"REALTIME_STT_PROVIDER": "deepgram"}, clear=True):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["stt_kind"], "deepgram")
        self.assertIn("DEEPGRAM_API_KEY", status["config"]["missing_provider_env"])
        with self.assertRaisesRegex(RuntimeError, "Missing realtime provider env"):
            worker.run_offline_synthetic_call()

    def test_gemini_and_elevenlabs_provider_mode_reports_both_missing_env_groups(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "gemini_llm_elevenlabs_v3_tts"}, clear=True):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["provider_kind"], "gemini_llm_elevenlabs_v3_tts")
        self.assertIn("GEMINI_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("ELEVENLABS_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE", status["config"]["missing_provider_env"])

    def test_full_realtime_provider_mode_implies_deepgram_and_reports_all_missing_env(self) -> None:
        with patch.dict("os.environ", {"REALTIME_PROVIDER_MODE": "full_realtime"}, clear=True):
            worker = RealtimeMediaWorker(RealtimeWorkerConfig.from_env())
        status = worker.status()
        self.assertFalse(status["ok"])
        self.assertEqual(status["config"]["provider_mode"], "full_realtime")
        self.assertEqual(status["config"]["provider_kind"], "deepgram_gemini_elevenlabs_v3")
        self.assertEqual(status["config"]["stt_kind"], "deepgram")
        self.assertIn("DEEPGRAM_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("GEMINI_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("ELEVENLABS_API_KEY", status["config"]["missing_provider_env"])
        self.assertIn("ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE", status["config"]["missing_provider_env"])
        with self.assertRaisesRegex(RuntimeError, "Missing realtime provider env"):
            worker.run_offline_synthetic_call()

    def test_gemini_provider_mode_streams_llm_with_fake_tts_in_offline_call(self) -> None:
        created_configs = []

        class FakeGeminiLLM:
            def __init__(self, config):
                created_configs.append(config)

            def stream_reply(self, messages):
                self.messages = messages
                yield TextChunk("Tere, ")
                yield TextChunk("broneerin aja.", is_final=True)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="gemini_llm_fake_tts",
                    gemini_api_key="test-gemini-key",
                    gemini_model_id="gemini-test-model",
                    gemini_system_prompt="Reply in Estonian.",
                    fake_transcript="Soovin autopesu aega.",
                )
            )
            with patch("openphonex_gateway.realtime.worker.GeminiStreamingLLM", FakeGeminiLLM):
                result = worker.run_offline_synthetic_call(root / "call")

        self.assertTrue(result["ok"])
        self.assertGreater(result["rtp_packets"], 0)
        self.assertEqual(created_configs[0].api_key, "test-gemini-key")
        self.assertEqual(created_configs[0].model_id, "gemini-test-model")
        self.assertEqual(created_configs[0].system_prompt, "Reply in Estonian.")
        self.assertEqual(result["provider_components"]["llm"]["provider"], "gemini")
        self.assertTrue(result["provider_components"]["llm"]["real"])
        self.assertEqual(result["provider_components"]["tts"]["provider"], "tone")
        self.assertIn("autopesu", result["transcript"])
        self.assertIn("broneerin", result["assistant"])

    def test_deepgram_stt_provider_wires_real_stt_adapter_into_offline_call(self) -> None:
        created_configs = []

        class FakeDeepgramSTT:
            def __init__(self, config):
                created_configs.append(config)

            def start_stream(self):
                return _ScriptedSttSession()

            def transcribe(self, turn):
                return ()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="fake",
                    stt_provider="deepgram",
                    deepgram_api_key="test-deepgram-key",
                    deepgram_model_id="nova-test",
                    deepgram_language="et",
                    deepgram_endpointing_ms=280,
                )
            )
            with patch("openphonex_gateway.realtime.worker.DeepgramStreamingSTT", FakeDeepgramSTT):
                result = worker.run_offline_synthetic_call(root / "call")

        self.assertTrue(result["ok"])
        self.assertEqual(created_configs[0].api_key, "test-deepgram-key")
        self.assertEqual(created_configs[0].model_id, "nova-test")
        self.assertEqual(created_configs[0].language, "et")
        self.assertEqual(created_configs[0].endpointing_ms, 280)
        self.assertEqual(result["provider_components"]["stt"]["provider"], "deepgram")
        self.assertTrue(result["provider_components"]["stt"]["real"])
        self.assertEqual(result["provider_components"]["llm"]["provider"], "template")
        self.assertIn("autopesu", result["transcript"])
        self.assertIn("auto mudeliga", result["assistant"])

    def test_elevenlabs_provider_mode_streams_tts_in_offline_call(self) -> None:
        created_configs = []

        class FakeElevenLabsTTS:
            def __init__(self, config):
                created_configs.append(config)

            def stream(self, text):
                yield AudioFrame.tone(20, direction="agent")
                yield AudioFrame.tone(20, hz=660, direction="agent")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="elevenlabs_v3_tts",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice_id="voice-123",
                    elevenlabs_model_id="eleven_v3",
                    elevenlabs_language_code="et",
                    fake_transcript="Soovin autopesu aega.",
                    fake_reply="Tere! Pakun homme kell kümme.",
                )
            )
            with patch("openphonex_gateway.realtime.worker.ElevenLabsV3StreamingTTS", FakeElevenLabsTTS):
                result = worker.run_offline_synthetic_call(root / "call")

        self.assertTrue(result["ok"])
        self.assertGreater(result["rtp_packets"], 0)
        self.assertEqual(created_configs[0].voice_id, "voice-123")
        self.assertEqual(created_configs[0].model_id, "eleven_v3")
        self.assertEqual(created_configs[0].language_code, "et")
        self.assertEqual(result["provider_components"]["tts"]["provider"], "elevenlabs")
        self.assertTrue(result["provider_components"]["tts"]["real"])
        self.assertEqual(result["provider_components"]["stt"]["provider"], "fake")
        self.assertIn("autopesu", result["transcript"])
        self.assertIn("homme", result["assistant"])

    def test_provider_prewarm_caches_configured_elevenlabs_voice_id_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="elevenlabs_v3_tts",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice_id="voice-123",
                    elevenlabs_model_id="eleven_v3",
                )
            )
            with patch("openphonex_gateway.realtime.worker.resolve_elevenlabs_voice_id", return_value="voice-123") as resolve:
                check = worker.prewarm_provider_clients()

        self.assertTrue(check["ok"], check)
        resolve.assert_called_once_with(api_key="test-elevenlabs-key", voice_id="voice-123", voice_name="")
        self.assertEqual(check["provider_network_actions_required"], [])
        self.assertEqual(check["provider_network_actions_performed"], [])
        self.assertTrue(check["cached"]["elevenlabs_voice_id"])
        self.assertEqual(check["cached"]["provider_adapters"], ["llm", "stt", "tts"])
        tts = next(item for item in check["prewarmed_components"] if item["component"] == "tts")
        self.assertEqual(tts["provider"], "elevenlabs")
        self.assertTrue(tts["voice_id_cached"])
        self.assertEqual(tts["network_action"], "none")
        self.assertTrue(tts["adapter_cached"])

    def test_provider_prewarm_requires_network_approval_for_elevenlabs_voice_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="elevenlabs_v3_tts",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice="Matilda",
                    elevenlabs_model_id="eleven_v3",
                )
            )
            with patch("openphonex_gateway.realtime.worker.resolve_elevenlabs_voice_id") as resolve:
                check = worker.prewarm_provider_clients()

        self.assertFalse(check["ok"], check)
        resolve.assert_not_called()
        self.assertEqual(check["reason"], "provider_prewarm_network_approval_required")
        self.assertEqual(check["provider_network_actions_required"], ["elevenlabs_voice_lookup"])
        self.assertFalse(check["cached"]["elevenlabs_voice_id"])
        self.assertEqual(check["cached"]["provider_adapters"], [])

    def test_live_startup_requires_no_network_provider_prewarm_before_ari(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="elevenlabs_v3_tts",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice="Matilda",
                    elevenlabs_model_id="eleven_v3",
                    live_enabled=True,
                )
            )
            with (
                patch("openphonex_gateway.realtime.worker.missing_elevenlabs_streaming_runtime_dependencies", return_value=[]),
                patch("openphonex_gateway.realtime.worker.resolve_elevenlabs_voice_id") as resolve,
            ):
                preflight = worker.live_startup_preflight()
                with self.assertRaisesRegex(RuntimeError, "provider_prewarm_network_approval_required"):
                    worker.run_asterisk_event_loop([])

        resolve.assert_not_called()
        self.assertFalse(preflight["ok"], preflight)
        self.assertFalse(preflight["ready_to_connect_ari"])
        self.assertFalse(preflight["provider_ready"])
        self.assertTrue(preflight["live_ready"])
        self.assertEqual(preflight["provider_error"]["reason"], "provider_prewarm_network_approval_required")
        self.assertEqual(preflight["provider_prewarm"]["reason"], "provider_prewarm_network_approval_required")
        self.assertEqual(preflight["provider_prewarm"]["provider_network_actions_required"], ["elevenlabs_voice_lookup"])
        self.assertEqual(preflight["provider_network_actions_performed"], [])
        self.assertEqual(preflight["next_step"], "configure_elevenlabs_voice_id_or_run_approved_provider_prewarm")

    def test_provider_prewarm_reuses_adapter_instances_for_next_call(self) -> None:
        created_deepgram_configs = []
        created_gemini_configs = []
        created_elevenlabs_configs = []

        class FakeDeepgramSTT:
            def __init__(self, config):
                created_deepgram_configs.append(config)

            def start_stream(self):
                return _ScriptedSttSession()

            def transcribe(self, turn):
                return ()

        class FakeGeminiLLM:
            def __init__(self, config):
                created_gemini_configs.append(config)

            def stream_reply(self, messages):
                yield TextChunk("Tere, ")
                yield TextChunk("broneerin aja.", is_final=True)

        class FakeElevenLabsTTS:
            def __init__(self, config):
                created_elevenlabs_configs.append(config)

            def stream(self, text):
                yield AudioFrame.tone(20, direction="agent")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="full_realtime",
                    deepgram_api_key="test-deepgram-key",
                    gemini_api_key="test-gemini-key",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice_id="voice-123",
                )
            )
            with (
                patch("openphonex_gateway.realtime.worker.DeepgramStreamingSTT", FakeDeepgramSTT),
                patch("openphonex_gateway.realtime.worker.GeminiStreamingLLM", FakeGeminiLLM),
                patch("openphonex_gateway.realtime.worker.ElevenLabsV3StreamingTTS", FakeElevenLabsTTS),
            ):
                check = worker.prewarm_provider_clients(allow_provider_network=True)
                result = worker.run_offline_synthetic_call(root / "call")

        self.assertTrue(check["ok"], check)
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(created_deepgram_configs), 1)
        self.assertEqual(len(created_gemini_configs), 1)
        self.assertEqual(len(created_elevenlabs_configs), 1)
        self.assertEqual(check["cached"]["provider_adapters"], ["llm", "stt", "tts"])
        self.assertTrue(all(item["adapter_cached"] for item in check["prewarmed_components"]))

    def test_full_realtime_provider_mode_wires_real_stt_llm_and_tts_adapters(self) -> None:
        created_deepgram_configs = []
        created_gemini_configs = []
        created_elevenlabs_configs = []

        class FakeDeepgramSTT:
            def __init__(self, config):
                created_deepgram_configs.append(config)

            def start_stream(self):
                return _ScriptedSttSession()

            def transcribe(self, turn):
                return ()

        class FakeGeminiLLM:
            def __init__(self, config):
                created_gemini_configs.append(config)

            def stream_reply(self, messages):
                yield TextChunk("Tere, ")
                yield TextChunk("broneerin aja.", is_final=True)

        class FakeElevenLabsTTS:
            def __init__(self, config):
                created_elevenlabs_configs.append(config)

            def stream(self, text):
                yield AudioFrame.tone(20, direction="agent")
                yield AudioFrame.tone(20, hz=660, direction="agent")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="full_realtime",
                    deepgram_api_key="test-deepgram-key",
                    deepgram_model_id="nova-test",
                    deepgram_language="et",
                    gemini_api_key="test-gemini-key",
                    gemini_model_id="gemini-test-model",
                    gemini_system_prompt="Reply in Estonian.",
                    elevenlabs_api_key="test-elevenlabs-key",
                    elevenlabs_voice_id="voice-123",
                    elevenlabs_model_id="eleven_v3",
                    elevenlabs_language_code="et",
                )
            )
            with (
                patch("openphonex_gateway.realtime.worker.DeepgramStreamingSTT", FakeDeepgramSTT),
                patch("openphonex_gateway.realtime.worker.GeminiStreamingLLM", FakeGeminiLLM),
                patch("openphonex_gateway.realtime.worker.ElevenLabsV3StreamingTTS", FakeElevenLabsTTS),
            ):
                result = worker.run_offline_synthetic_call(root / "call")

        self.assertTrue(result["ok"])
        self.assertEqual(created_deepgram_configs[0].api_key, "test-deepgram-key")
        self.assertEqual(created_deepgram_configs[0].model_id, "nova-test")
        self.assertEqual(created_gemini_configs[0].api_key, "test-gemini-key")
        self.assertEqual(created_elevenlabs_configs[0].voice_id, "voice-123")
        self.assertEqual(result["provider_components"]["stt"]["provider"], "deepgram")
        self.assertEqual(result["provider_components"]["llm"]["provider"], "gemini")
        self.assertEqual(result["provider_components"]["tts"]["provider"], "elevenlabs")
        self.assertTrue(result["provider_components"]["stt"]["real"])
        self.assertTrue(result["provider_components"]["llm"]["real"])
        self.assertTrue(result["provider_components"]["tts"]["real"])
        self.assertIn("autopesu", result["transcript"])
        self.assertIn("broneerin", result["assistant"])

    def test_offline_synthetic_call_writes_realtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="fake",
                )
            )
            result = worker.run_offline_synthetic_call(root / "call")
            self.assertTrue(result["ok"])
            self.assertGreater(result["rtp_packets"], 0)
            for name in ("caller.wav", "agent.wav", "mixed.wav", "turn-001-caller.wav", "turn-001-agent.wav"):
                path = root / "call" / name
                self.assertTrue(path.exists(), name)
                with wave.open(str(path), "rb") as wf:
                    self.assertEqual(wf.getframerate(), 8000)
                    self.assertGreater(wf.getnframes(), 0)
            latency = json.loads((root / "call" / "latency.json").read_text())
            self.assertIn("first_rtp_packet_sent", [event["name"] for event in latency["events"]])
            transcript = json.loads((root / "call" / "transcript.json").read_text())
            self.assertIn("soovin aega", transcript["user"])
            self.assertIn("auto mudeliga", transcript["assistant"])
            self.assertEqual(transcript["summary"]["turn_count"], 1)
            self.assertIn("Caller said", transcript["summary"]["text"])
            self.assertEqual(transcript["turns"][0]["caller_audio"], "turn-001-caller.wav")
            self.assertEqual(transcript["turns"][0]["agent_audio"], "turn-001-agent.wav")
            self.assertIn("turn-001-caller.wav", " ".join(result["artifacts"]["turn_clips"]))
            self.assertIn("turn-001-agent.wav", " ".join(result["artifacts"]["turn_clips"]))
            waveform = json.loads((root / "call" / "waveform.json").read_text())
            waveforms = json.loads((root / "call" / "waveforms.json").read_text())
            self.assertEqual(result["artifacts"]["waveform"], str(root / "call" / "waveform.json"))
            self.assertEqual(result["artifacts"]["waveforms"], str(root / "call" / "waveforms.json"))
            self.assertEqual(waveform["track"], "mixed")
            self.assertGreater(len(waveform["peaks"]), 0)
            self.assertEqual(waveforms["primary"], "mixed")
            self.assertIn("caller", waveforms["tracks"])
            self.assertIn("agent", waveforms["tracks"])
            self.assertIn("mixed", waveforms["tracks"])

    def test_offline_recorded_call_uses_local_wav_audio_without_live_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            recorded = root / "caller.wav"
            _write_speech_wav(recorded)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="fake",
                )
            )
            result = worker.run_offline_recorded_call(recorded, root / "call")

            self.assertTrue(result["ok"])
            self.assertEqual(result["input_audio"], str(recorded))
            self.assertGreater(result["rtp_packets"], 0)
            self.assertTrue((root / "call" / "mixed.wav").exists())
            transcript = json.loads((root / "call" / "transcript.json").read_text())
            self.assertEqual(transcript["source_audio"], str(recorded))
            self.assertEqual(transcript["turns"][0]["source_audio"], str(recorded))
            self.assertIn("soovin aega", transcript["user"])

    def test_offline_streaming_overlap_check_writes_overlap_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="fake",
                )
            )
            result = worker.run_offline_streaming_overlap_check(root / "call")

            self.assertTrue(result["ok"])
            self.assertTrue(result["llm_stream_completed_before_first_phrase_done"])
            self.assertTrue(result["second_phrase_ready_before_first_phrase_done"])
            self.assertGreaterEqual(result["phrase_count"], 2)
            self.assertGreaterEqual(result["tts_phrase_count"], 2)
            self.assertGreater(result["rtp_packets"], 0)
            transcript = json.loads((root / "call" / "transcript.json").read_text())
            self.assertTrue(transcript["streaming_overlap_check"])
            latency = json.loads((root / "call" / "latency.json").read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("llm_stream_completed", event_names)
            self.assertIn("tts_phrase_rtp_done", event_names)
            self.assertIn("offline_streaming_overlap_check_completed", event_names)

    def test_offline_streaming_overlap_check_cli_writes_overlap_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with (
                patch.dict(
                    "os.environ",
                    {
                        "CALLS_DIR": str(root),
                        "RECORDINGS_DIR": str(root),
                        "REALTIME_PROVIDER_MODE": "fake",
                    },
                    clear=True,
                ),
                patch("builtins.print") as printed,
            ):
                code = realtime_worker_main(["--offline-streaming-overlap-check", "--offline-output-dir", str(root / "call")])

            self.assertEqual(code, 0)
            payload = json.loads(printed.call_args.args[0])
            self.assertTrue(payload["ok"], payload)
            self.assertTrue(payload["llm_stream_completed_before_first_phrase_done"])
            self.assertTrue(payload["second_phrase_ready_before_first_phrase_done"])
            self.assertTrue((root / "call" / "transcript.json").exists())
            transcript = json.loads((root / "call" / "transcript.json").read_text())
            self.assertTrue(transcript["streaming_overlap_check"])

    def test_offline_barge_in_check_writes_interrupted_transcript_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="", ari_password=""),
                    provider_mode="fake",
                )
            )
            result = worker.run_offline_barge_in_check(root / "call")

            self.assertTrue(result["ok"])
            self.assertTrue(result["barge_in_detected"])
            self.assertTrue(result["output_cancelled"])
            self.assertTrue(result["pending_frames_replayed"])
            self.assertTrue(result["llm_stream_closed"])
            self.assertTrue(result["llm_phrase_backpressure"])
            self.assertTrue(result["llm_phrase_producer_cancelled"])
            self.assertFalse(result["llm_phrase_producer_still_running"])
            self.assertEqual(result["interrupted_turns"], 1)
            self.assertGreater(result["rtp_packets"], 0)
            transcript = json.loads((root / "call" / "transcript.json").read_text())
            self.assertTrue(transcript["barge_in_check"])
            self.assertEqual(len(transcript["turns"]), 2)
            self.assertTrue(transcript["turns"][0]["interrupted"])
            self.assertFalse(transcript["turns"][1]["interrupted"])
            latency = json.loads((root / "call" / "latency.json").read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("barge_in_speech_detected", event_names)
            self.assertIn("barge_in_pending_frames_replayed", event_names)
            self.assertIn("llm_chunk_stream_closed", event_names)
            self.assertIn("llm_phrase_queue_full", event_names)
            self.assertIn("llm_phrase_producer_cancelled", event_names)
            self.assertNotIn("llm_phrase_producer_still_running", event_names)
            self.assertTrue((root / "call" / "turn-001-agent.wav").exists())
            self.assertTrue((root / "call" / "turn-002-agent.wav").exists())

    def test_ari_websocket_event_url_uses_ari_base_app_and_api_key(self) -> None:
        config = AsteriskExternalMediaConfig(
            ari_url="https://asterisk.example.com:9443/ari",
            ari_user="ari",
            ari_password="secret",
            app="openphonex",
        )
        self.assertEqual(
            AriWebSocketEventClient(config).events_url(),
            "wss://asterisk.example.com:9443/ari/events?app=openphonex&api_key=ari%3Asecret",
        )

    def test_live_asterisk_worker_is_guarded_by_live_flag(self) -> None:
        worker = RealtimeMediaWorker(
            RealtimeWorkerConfig(
                calls_dir=Path(tempfile.gettempdir()),
                recordings_dir=Path(tempfile.gettempdir()),
                rtp_bind="127.0.0.1",
                rtp_port=12000,
                asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                provider_mode="fake",
                live_enabled=False,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "REALTIME_LIVE_ENABLED"):
            worker.run_asterisk_event_loop([])

    def test_live_asterisk_event_loop_writes_artifacts_with_fake_rtp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transport = _ScriptedTransport()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    idle_timeout_seconds=1.0,
                    control_plane_api_url="http://openphonex-api.test",
                    gateway_api_token="gateway-token",
                )
            )
            active_snapshots: list[dict[str, object]] = []
            posted_jobs: list[tuple[str, dict[str, object], dict[str, str]]] = []

            def fake_post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
                posted_jobs.append((url, payload, headers))
                return {"id": "job_artifact_123", "status": "queued"}

            def events():
                yield {
                    "type": "StasisStart",
                    "channel": {
                        "id": "primary-1",
                        "name": "PJSIP/demo",
                        "state": "Up",
                        "variables": {
                            "OPENPHONEX_CALL_DIRECTION": "outbound",
                            "OPENPHONEX_CALL_ID": "call_live_123",
                            "OPENPHONEX_ORGANIZATION_ID": "org_123",
                            "OPENPHONEX_PROJECT_ID": "proj_123",
                            "OPENPHONEX_NUMBER_ID": "num_123",
                            "OPENPHONEX_AGENT_ID": "agent_123",
                            "OPENPHONEX_AGENT_ENDPOINT_ID": "endpoint_123",
                            "OPENPHONEX_PHONE_SERVER_ID": "phone_server_123",
                            "OPENPHONEX_TRIAL_CALL_ID": "trial_123",
                            "OPENPHONEX_MEDIA_PIPELINE": "realtime",
                            "OPENPHONEX_TO": "+15555550123",
                            "OPENPHONEX_API_KEY": "secret-token",
                            "OPENPHONEX_WORKER_JOB_ID": "+15555550124",
                        },
                    },
                }
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline:
                    snapshot = worker.capacity_status()
                    if int(snapshot["active_calls"]) > 0:
                        active_snapshots.append(snapshot)
                    # The outbound opening utterance sends RTP before the caller
                    # turn; wait for the caller frames to drain and the turn
                    # reply to start before hanging up.
                    if len(transport.sent_packets) >= 12 and not transport.frames:
                        break
                    time.sleep(0.01)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}

            with patch("openphonex_gateway.realtime.worker._post_json", fake_post_json):
                result = worker.run_asterisk_event_loop(events(), ari=FakeAri(), transport=transport, close_transport=False)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["actions"], ["primary_started", "primary_ended"])
            self.assertEqual(len(result["calls"]), 1)
            call = result["calls"][0]
            self.assertEqual(call["turns"], 1)
            self.assertEqual(call["direction"], "outbound")
            self.assertEqual(call["direction_source"], "channel_variable:OPENPHONEX_CALL_DIRECTION")
            self.assertEqual(call["openphonex_call_id"], "call_live_123")
            self.assertEqual(call["asterisk_channel_id"], "primary-1")
            self.assertEqual(call["artifact_import_job"]["job_id"], "job_artifact_123")
            self.assertTrue(call["artifact_import_job"]["posted"])
            self.assertEqual(len(posted_jobs), 1)
            posted_url, posted_payload, posted_headers = posted_jobs[0]
            self.assertEqual(posted_url, "http://openphonex-api.test/v1/worker-jobs")
            self.assertEqual(posted_headers["Authorization"], "Bearer gateway-token")
            self.assertEqual(posted_payload["organization_id"], "org_123")
            self.assertEqual(posted_payload["kind"], "call.artifact_import")
            artifact_payload = posted_payload["payload"]
            self.assertEqual(artifact_payload["organization_id"], "org_123")
            self.assertEqual(artifact_payload["project_id"], "proj_123")
            self.assertEqual(artifact_payload["call_id"], "call_live_123")
            self.assertEqual(artifact_payload["source_path"], call["call_dir"])
            self.assertEqual(artifact_payload["direction"], "outbound")
            self.assertEqual(
                call["openphonex"],
                {
                    "agent_endpoint_id": "endpoint_123",
                    "agent_id": "agent_123",
                    "call_id": "call_live_123",
                    "media_pipeline": "realtime",
                    "number_id": "num_123",
                    "organization_id": "org_123",
                    "phone_server_id": "phone_server_123",
                    "project_id": "proj_123",
                    "trial_call_id": "trial_123",
                },
            )
            self.assertTrue(active_snapshots)
            active_detail = active_snapshots[0]["active_call_details"][0]
            self.assertEqual(active_detail["asterisk_channel_id"], "primary-1")
            self.assertEqual(active_detail["call_id"], "call_live_123")
            self.assertEqual(active_detail["organization_id"], "org_123")
            self.assertEqual(active_detail["number_id"], "num_123")
            self.assertEqual(active_detail["agent_id"], "agent_123")
            self.assertEqual(active_detail["media_pipeline"], "realtime")
            self.assertEqual(active_detail["rtp_port"], 12000)
            self.assertTrue(Path(call["artifacts"]["latency"]).exists())
            transcript_text = Path(call["artifacts"]["transcript"]).read_text()
            transcript = json.loads(transcript_text)
            self.assertNotIn("+15555550123", transcript_text)
            self.assertNotIn("+15555550124", transcript_text)
            self.assertNotIn("secret-token", transcript_text)
            self.assertEqual(transcript["call_id"], "call_live_123")
            self.assertEqual(transcript["asterisk_channel_id"], "primary-1")
            self.assertEqual(transcript["openphonex"], call["openphonex"])
            self.assertEqual(transcript["call_direction"], "outbound")
            self.assertEqual(transcript["call_direction_source"], "channel_variable:OPENPHONEX_CALL_DIRECTION")
            self.assertIn("opening_utterance", transcript)
            self.assertTrue(transcript["opening_utterance"]["text"])
            self.assertFalse(transcript["opening_utterance"]["skipped"])
            self.assertEqual(call["opening_utterance"], transcript["opening_utterance"])
            self.assertEqual(transcript["latency"], call["latency"])
            self.assertEqual(transcript["rtp_receive_stats"], call["rtp_receive_stats"])
            self.assertIn("soovin aega", transcript["turns"][0]["user"])
            self.assertIn("auto mudeliga", transcript["turns"][0]["assistant"])
            self.assertEqual(transcript["turns"][0]["assistant_spoken"], transcript["turns"][0]["assistant"])
            self.assertEqual(transcript["provider_components"]["stt"]["provider"], "fake")
            self.assertEqual(transcript["provider_components"]["llm"]["provider"], "template")
            self.assertEqual(transcript["provider_components"]["tts"]["provider"], "tone")
            self.assertGreater(len(transport.sent_packets), 0)
            self.assertEqual(call["rtp_receive_stats"]["packets_received"], 39)
            self.assertIn("caller_hears_first_audio_estimate_ms", call["latency"])
            latency_text = Path(call["artifacts"]["latency"]).read_text()
            self.assertNotIn("+15555550123", latency_text)
            self.assertNotIn("+15555550124", latency_text)
            self.assertNotIn("secret-token", latency_text)
            latency = json.loads(latency_text)
            direction_events = [event for event in latency["events"] if event["name"] == "call_direction"]
            self.assertEqual(direction_events[-1]["attrs"]["direction"], "outbound")
            self.assertEqual(direction_events[-1]["attrs"]["source"], "channel_variable:OPENPHONEX_CALL_DIRECTION")
            metadata_events = [event for event in latency["events"] if event["name"] == "openphonex_call_metadata"]
            self.assertEqual(metadata_events[-1]["attrs"]["call_id"], "call_live_123")
            self.assertEqual(metadata_events[-1]["attrs"]["organization_id"], "org_123")
            self.assertEqual(metadata_events[-1]["attrs"]["agent_id"], "agent_123")
            self.assertNotIn("worker_job_id", metadata_events[-1]["attrs"])
            rtp_events = [event for event in latency["events"] if event["name"] == "rtp_receive_stats"]
            self.assertEqual(rtp_events[-1]["attrs"]["packets_received"], 39)

    def test_live_asterisk_event_loop_handles_barge_in_with_fake_rtp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transport = _BargeInScriptedTransport()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    max_turns=2,
                    idle_timeout_seconds=1.0,
                )
            )

            def events():
                yield {"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Up"}}
                deadline = time.monotonic() + 2.0
                while len(transport.sent_packets) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                deadline = time.monotonic() + 2.0
                while not transport.barge_frame_served and time.monotonic() < deadline:
                    time.sleep(0.01)
                deadline = time.monotonic() + 2.0
                while transport.continuation_frames and time.monotonic() < deadline:
                    time.sleep(0.01)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}

            result = worker.run_asterisk_event_loop(events(), ari=FakeAri(), transport=transport, close_transport=False)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["actions"], ["primary_started", "primary_ended"])
            self.assertEqual(len(result["calls"]), 1)
            call = result["calls"][0]
            self.assertEqual(call["turns"], 2)
            transcript = json.loads(Path(call["artifacts"]["transcript"]).read_text())
            self.assertTrue(transcript["turns"][0]["interrupted"])
            self.assertEqual(transcript["turns"][0]["assistant_spoken"], "")
            self.assertFalse(transcript["turns"][1]["interrupted"])
            self.assertIn("soovin aega", transcript["turns"][1]["user"])
            latency = json.loads(Path(call["artifacts"]["latency"]).read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("barge_in_speech_detected", event_names)
            self.assertIn("barge_in_output_cancelled", event_names)
            self.assertIn("barge_in_pending_frames_replayed", event_names)
            self.assertIn("assistant_turn_interrupted", event_names)
            self.assertTrue(any(Path(call["call_dir"]).glob("turn-002-caller.wav")))

    def test_realtime_worker_config_reads_llm_fallback_and_outbound_initiative_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "REALTIME_LLM_FALLBACK_MODEL": "gemini-test-fallback",
                "REALTIME_LLM_RETRY_BUDGET_SECONDS": "1.5",
                "REALTIME_RECOVERY_UTTERANCE": "Vabandust, palun korrake.",
                "REALTIME_OPENING_UTTERANCE": "Tere! Helistan teile OpenPhonexist.",
                "REALTIME_OUTBOUND_ANSWER_TIMEOUT_SECONDS": "5",
            },
            clear=True,
        ):
            config = RealtimeWorkerConfig.from_env()
        self.assertEqual(config.llm_fallback_model_id, "gemini-test-fallback")
        self.assertEqual(config.llm_retry_budget_seconds, 1.5)
        self.assertEqual(config.recovery_utterance, "Vabandust, palun korrake.")
        self.assertEqual(config.opening_utterance, "Tere! Helistan teile OpenPhonexist.")
        self.assertEqual(config.outbound_answer_timeout_seconds, 5.0)
        redacted = config.redacted()
        self.assertEqual(redacted["llm_fallback"]["model_id"], "gemini-test-fallback")
        self.assertTrue(redacted["llm_fallback"]["recovery_utterance_configured"])
        self.assertTrue(redacted["outbound_initiative"]["opening_utterance_configured"])
        self.assertEqual(redacted["outbound_initiative"]["answer_timeout_seconds"], 5.0)
        with patch.dict("os.environ", {}, clear=True):
            defaults = RealtimeWorkerConfig.from_env()
        self.assertEqual(defaults.llm_fallback_model_id, "gemini-2.5-flash-lite")
        self.assertTrue(defaults.opening_utterance)
        self.assertEqual(defaults.recovery_utterance, "")

    def test_live_asterisk_event_loop_speaks_opening_after_outbound_channel_up(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transport = _NoCallerAudioTransport()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    max_turns=1,
                    idle_timeout_seconds=0.3,
                )
            )
            packets_before_up: list[int] = []

            def events():
                yield {
                    "type": "StasisStart",
                    "args": ["outbound"],
                    "channel": {"id": "primary-1", "name": "PJSIP/didww-out", "state": "Ringing"},
                }
                time.sleep(0.3)
                packets_before_up.append(len(transport.sent_packets))
                yield {"type": "ChannelStateChange", "channel": {"id": "primary-1", "name": "PJSIP/didww-out", "state": "Up"}}
                deadline = time.monotonic() + 3.0
                while len(transport.sent_packets) < 8 and time.monotonic() < deadline:
                    time.sleep(0.01)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}

            result = worker.run_asterisk_event_loop(events(), ari=FakeAri(), transport=transport, close_transport=False)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["actions"], ["primary_started", "primary_state_changed", "primary_ended"])
            call = result["calls"][0]
            self.assertEqual(call["direction"], "outbound")
            self.assertEqual(call["direction_source"], "stasis_app_args")
            # The bot waited for the callee channel before speaking.
            self.assertEqual(packets_before_up, [0])
            self.assertGreaterEqual(len(transport.sent_packets), 8)
            self.assertTrue(call["opening_utterance"]["spoken"])
            self.assertFalse(call["opening_utterance"]["interrupted"])
            self.assertFalse(call["opening_utterance"]["skipped"])
            self.assertEqual(call["turns"], 0)
            transcript = json.loads(Path(call["artifacts"]["transcript"]).read_text())
            self.assertTrue(transcript["opening_utterance"]["spoken"])
            self.assertEqual(transcript["call_direction"], "outbound")
            latency = json.loads(Path(call["artifacts"]["latency"]).read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("outbound_channel_up_wait_finished", event_names)
            self.assertIn("scripted_utterance_started", event_names)
            self.assertIn("scripted_utterance_finished", event_names)
            wait_event = next(event for event in latency["events"] if event["name"] == "outbound_channel_up_wait_finished")
            self.assertTrue(wait_event["attrs"]["channel_up"])

    def test_live_asterisk_event_loop_greets_first_on_inbound(self) -> None:
        # SCH-210 regression: an inbound caller must hear a greeting instead of
        # silence. The bot speaks the configured inbound opener immediately (no
        # callee channel-up wait) before it starts listening for turns.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transport = _NoCallerAudioTransport()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    max_turns=1,
                    idle_timeout_seconds=0.3,
                    inbound_opening_utterance="Tere! Kuidas saan teid aidata?",
                )
            )

            def events():
                yield {"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/didww-in", "state": "Up"}}
                deadline = time.monotonic() + 3.0
                while len(transport.sent_packets) < 8 and time.monotonic() < deadline:
                    time.sleep(0.01)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}

            result = worker.run_asterisk_event_loop(events(), ari=FakeAri(), transport=transport, close_transport=False)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            call = result["calls"][0]
            self.assertEqual(call["direction"], "inbound")
            self.assertEqual(call["direction_source"], "default_inbound")
            self.assertTrue(call["opening_utterance"]["spoken"])
            self.assertFalse(call["opening_utterance"]["interrupted"])
            self.assertFalse(call["opening_utterance"]["skipped"])
            self.assertEqual(call["turns"], 0)
            self.assertGreaterEqual(len(transport.sent_packets), 8)
            transcript = json.loads(Path(call["artifacts"]["transcript"]).read_text())
            self.assertEqual(transcript["call_direction"], "inbound")
            self.assertTrue(transcript["opening_utterance"]["spoken"])
            self.assertEqual(transcript["opening_utterance"]["text"], "Tere! Kuidas saan teid aidata?")
            latency = json.loads(Path(call["artifacts"]["latency"]).read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("scripted_utterance_started", event_names)
            self.assertIn("inbound_opening_utterance_finished", event_names)

    def test_inbound_default_config_stays_silent_and_env_sources_greeting_and_barge_in(self) -> None:
        # The dataclass default is intentionally empty so direct-construction (unit
        # test) callers keep the legacy "wait for the caller" inbound behavior...
        direct_config = RealtimeWorkerConfig(
            calls_dir=Path("/tmp/openphonex-test"),
            recordings_dir=Path("/tmp/openphonex-test"),
            rtp_bind="127.0.0.1",
            rtp_port=12000,
            asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
        )
        self.assertEqual(direct_config.inbound_opening_utterance, "")
        self.assertEqual(direct_config.barge_in_speech_rms, 650.0)

        # ...while from_env supplies the production inbound greeting default and the
        # live half-duplex barge-in threshold (SCH-210).
        with patch.dict("os.environ", {"REALTIME_BARGE_IN_RMS": "32000"}, clear=True):
            env_config = RealtimeWorkerConfig.from_env()
        self.assertEqual(env_config.inbound_opening_utterance, DEFAULT_INBOUND_OPENING_UTTERANCE)
        self.assertTrue(env_config.inbound_opening_utterance)
        self.assertEqual(env_config.barge_in_speech_rms, 32000.0)

        # The inbound-specific override wins; REALTIME_OPENING_UTTERANCE is the
        # documented fallback when the inbound-specific var is unset.
        with patch.dict("os.environ", {"REALTIME_INBOUND_OPENING_UTTERANCE": "Tere kliendile!"}, clear=True):
            self.assertEqual(RealtimeWorkerConfig.from_env().inbound_opening_utterance, "Tere kliendile!")
        with patch.dict("os.environ", {"REALTIME_OPENING_UTTERANCE": "Uhine tervitus."}, clear=True):
            self.assertEqual(RealtimeWorkerConfig.from_env().inbound_opening_utterance, "Uhine tervitus.")

    def test_live_asterisk_event_loop_writes_artifacts_when_provider_error_aborts_call(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transport = _ScriptedTransport()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    idle_timeout_seconds=0.5,
                )
            )
            fake_config = FakeRealtimeProviderConfig()

            class ExplodingTTS:
                def stream(self, text):
                    raise RuntimeError("tts exploded")
                    yield  # pragma: no cover

            def events():
                yield {"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/demo", "state": "Up"}}
                deadline = time.monotonic() + 3.0
                while transport.frames and time.monotonic() < deadline:
                    time.sleep(0.01)
                time.sleep(0.3)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}

            with patch.object(
                worker,
                "_provider_adapters",
                return_value=(FakeStreamingSTT(fake_config), TemplateStreamingLLM(fake_config), ExplodingTTS()),
            ):
                result = worker.run_asterisk_event_loop(events(), ari=FakeAri(), transport=transport, close_transport=False)  # type: ignore[arg-type]

            self.assertFalse(result["ok"], result)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["errors"][0]["error"], "RuntimeError")
            self.assertIn("tts exploded", result["errors"][0]["message"])
            call_dirs = sorted(root.glob("realtime-primary-1-*"))
            self.assertEqual(len(call_dirs), 1)
            latency_path = call_dirs[0] / "latency.json"
            transcript_path = call_dirs[0] / "transcript.json"
            self.assertTrue(latency_path.exists())
            self.assertTrue(transcript_path.exists())
            transcript = json.loads(transcript_path.read_text())
            self.assertTrue(transcript["aborted"])
            self.assertEqual(transcript["call_error"]["error"], "RuntimeError")
            self.assertEqual(transcript["turns"], [])
            latency = json.loads(latency_path.read_text())
            event_names = [event["name"] for event in latency["events"]]
            self.assertIn("call_media_error", event_names)
            self.assertIn("call_media_stopped", event_names)
            stopped = next(event for event in latency["events"] if event["name"] == "call_media_stopped")
            self.assertTrue(stopped["attrs"]["aborted"])

    def test_live_asterisk_event_loop_runs_overlapping_calls_on_separate_rtp_ports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            transports: dict[int, _ScriptedTransport] = {}
            ari = MultiCallFakeAri()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    rtp_port_end=12003,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    max_turns=1,
                    idle_timeout_seconds=1.0,
                )
            )

            def transport_factory(port: int):
                transports[port] = _ScriptedTransport()
                return transports[port]

            def events():
                yield {"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/demo-1", "state": "Up"}}
                yield {"type": "StasisStart", "channel": {"id": "primary-2", "name": "PJSIP/demo-2", "state": "Up"}}
                deadline = time.monotonic() + 2.0
                while len([item for item in transports.values() if item.sent_packets]) < 2 and time.monotonic() < deadline:
                    time.sleep(0.01)
                yield {"type": "StasisEnd", "channel": {"id": "primary-1"}}
                yield {"type": "StasisEnd", "channel": {"id": "primary-2"}}

            result = worker.run_asterisk_event_loop(events(), ari=ari, transport_factory=transport_factory)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["actions"], ["primary_started", "primary_started", "primary_ended", "primary_ended"])
            self.assertEqual(len(result["calls"]), 2)
            self.assertEqual(sorted(transports), [12000, 12001])
            self.assertTrue(all(item.closed for item in transports.values()))
            external_hosts = [
                params["external_host"]
                for path, params, _body in ari.posts
                if path == "/channels/externalMedia" and params
            ]
            self.assertEqual(external_hosts, ["telephony-ai-realtime-worker:12000", "telephony-ai-realtime-worker:12001"])
            call_ids = {call["call_id"] for call in result["calls"]}
            self.assertEqual(call_ids, {"primary-1", "primary-2"})
            self.assertEqual(result["capacity"]["active_calls"], 0)
            self.assertEqual(result["capacity"]["available_call_slots"], 4)

    def test_live_asterisk_event_loop_rejects_calls_over_configured_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ari = MultiCallFakeAri()
            worker = RealtimeMediaWorker(
                RealtimeWorkerConfig(
                    calls_dir=root,
                    recordings_dir=root,
                    rtp_bind="127.0.0.1",
                    rtp_port=12000,
                    rtp_port_end=12003,
                    asterisk=AsteriskExternalMediaConfig(ari_user="ari", ari_password="secret"),
                    provider_mode="fake",
                    live_enabled=True,
                    max_concurrent_calls=1,
                    idle_timeout_seconds=0.2,
                )
            )

            def transport_factory(_port: int):
                return _EmptyTransport()

            events = [
                {"type": "StasisStart", "channel": {"id": "primary-1", "name": "PJSIP/demo-1", "state": "Up"}},
                {"type": "StasisStart", "channel": {"id": "primary-2", "name": "PJSIP/demo-2", "state": "Up"}},
                {"type": "StasisEnd", "channel": {"id": "primary-1"}},
            ]

            result = worker.run_asterisk_event_loop(events, ari=ari, transport_factory=transport_factory)  # type: ignore[arg-type]

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["actions"], ["primary_started", "capacity_rejected", "primary_ended"])
            self.assertIn("/channels/primary-2", ari.deletes)
            self.assertEqual(len(result["calls"]), 1)
            self.assertEqual(result["capacity"]["active_calls"], 0)


class _FakeTransport:
    peer = None


class _ScriptedTransport:
    def __init__(self) -> None:
        self.peer = None
        self.stats = _ScriptedStats()
        self.sent_packets: list[bytes] = []
        self.closed = False
        self.frames = [AudioFrame.silence(20) for _ in range(3)]
        self.frames += [AudioFrame.tone(20) for _ in range(16)]
        self.frames += [AudioFrame.silence(20) for _ in range(20)]

    def recv_frame(self):
        if self.frames:
            return self.frames.pop(0)
        time.sleep(0.01)
        return None

    def send_packet(self, packet: bytes) -> None:
        self.sent_packets.append(packet)

    def close(self) -> None:
        self.closed = True


class _BargeInScriptedTransport:
    def __init__(self) -> None:
        self.peer = None
        self.stats = _ScriptedStats(packets_received=75)
        self.sent_packets: list[bytes] = []
        self.closed = False
        self.barge_frame_served = False
        self.initial_frames = [AudioFrame.silence(20) for _ in range(3)]
        self.initial_frames += [AudioFrame.tone(20) for _ in range(16)]
        self.initial_frames += [AudioFrame.silence(20) for _ in range(20)]
        self.barge_frames = [AudioFrame.tone(20, direction="caller")]
        self.continuation_frames = [AudioFrame.tone(20, direction="caller") for _ in range(16)]
        self.continuation_frames += [AudioFrame.silence(20, direction="caller") for _ in range(20)]

    def recv_frame(self):
        if self.initial_frames:
            return self.initial_frames.pop(0)
        if len(self.sent_packets) >= 2 and self.barge_frames:
            self.barge_frame_served = True
            return self.barge_frames.pop(0)
        if self.barge_frame_served and self.continuation_frames:
            return self.continuation_frames.pop(0)
        time.sleep(0.01)
        return None

    def send_packet(self, packet: bytes) -> None:
        self.sent_packets.append(packet)

    def close(self) -> None:
        self.closed = True


class _NoCallerAudioTransport:
    """Records outbound RTP while the (fake) callee stays silent."""

    def __init__(self) -> None:
        self.peer = None
        self.stats = _ScriptedStats(packets_received=0)
        self.sent_packets: list[bytes] = []
        self.closed = False

    def recv_frame(self):
        time.sleep(0.005)
        return None

    def send_packet(self, packet: bytes) -> None:
        self.sent_packets.append(packet)

    def close(self) -> None:
        self.closed = True


class _EmptyTransport:
    def __init__(self) -> None:
        self.peer = None
        self.stats = _ScriptedStats(packets_received=0)
        self.closed = False

    def recv_frame(self):
        time.sleep(0.01)
        return None

    def send_packet(self, packet: bytes) -> None:
        raise AssertionError("empty transport should not send RTP")

    def close(self) -> None:
        self.closed = True


class _ScriptedStats:
    def __init__(self, packets_received: int = 39) -> None:
        self.packets_received = packets_received

    def as_dict(self):
        return {
            "packets_received": self.packets_received,
            "packets_lost": 0,
            "out_of_order_packets": 0,
            "duplicate_packets": 0,
            "payload_bytes": self.packets_received * 160,
            "decode_errors": 0,
            "peer": {"host": "127.0.0.1", "port": 12000},
        }


class _ScriptedSttSession:
    def __init__(self) -> None:
        self.sent_interim = False

    def send_audio(self, frame: AudioFrame):
        if frame.rms > 650 and not self.sent_interim:
            self.sent_interim = True
            return [TranscriptEvent("Soovin", is_final=False, language="et")]
        return []

    def end_input(self, turn=None):
        return [TranscriptEvent("Soovin autopesu aega.", is_final=True, language="et")]

    def close(self) -> None:
        pass


def _read_json(url: str):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.loads(response.read().decode())


def _write_speech_wav(path: Path) -> None:
    rate = 8000
    samples: list[int] = []
    samples.extend([0] * int(rate * 0.2))
    for idx in range(int(rate * 0.8)):
        samples.append(int(9000 * math.sin(2 * math.pi * 440 * idx / rate)))
    samples.extend([0] * int(rate * 0.6))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


if __name__ == "__main__":
    unittest.main()

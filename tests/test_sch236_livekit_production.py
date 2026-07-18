from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import wave
from pathlib import Path

from livekit_agent.production import (
    ProductionConfigurationError,
    build_livekit_model_tools,
    executed_tool_names,
    is_greeting_fragment,
    profile_from_realtime_config,
)
from livekit_agent.livekit_evidence import SynchronizedEvidenceRecorder
from livekit_agent.control_plane import _validate_pr79_artifact_set
from livekit_agent.evidence_contract import (
    effective_voice_stack_identity,
    semantic_response_audio_policy as livekit_semantic_response_audio_policy,
    valid_effective_voice_stack_identity,
)
from openphonex_gateway.realtime.semantic_audio_policy import (
    semantic_response_audio_policy as referee_semantic_response_audio_policy,
)
from livekit_agent.reconciliation import reconcile_stale_sessions
from livekit_agent.latency_evidence import (
    TurnLatencyRecorder,
    latency_breakdown,
)


def _config(*, language: str = "et", tools: list[str] | None = None) -> dict:
    return {
        "agent_id": "agent_1",
        "organization_id": "org_1",
        "project_id": "project_1",
        "language": language,
        "system_prompt": "Kasuta list_calls tööriista, kui helistaja küsib kõnede kohta.",
        "first_message": "Tere! Kuidas saan aidata?",
        "tools": tools or ["list_calls"],
        "recording_enabled": True,
        "transcriber": {
            "registry_name": "deepgram",
            "model": "nova-3",
            "language": language,
        },
        "model": {"registry_name": "gemini", "model": "gemini-2.5-flash-lite"},
        "voice": {
            "registry_name": "elevenlabs",
            "model": "eleven_v3" if language == "et" else "eleven_flash_v2_5",
            "voice_id": "voice_et_1",
            "language": language,
        },
    }


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.config = {"organization_id": "org_1", "project_id": "project_1"}

    def tool_definitions(self) -> list[dict]:
        return [
            {
                "name": "list_calls",
                "description": "List calls for the current organization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "organization_id": {"type": "string"},
                        "project_id": {"type": "string"},
                    },
                    "required": ["organization_id"],
                },
            }
        ]

    def invoke_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return {"result": {"content": [{"json": {"calls": []}}]}}


class _Tool:
    def __init__(self, function, raw_schema: dict) -> None:
        self.function = function
        self.raw_schema = raw_schema


class _FunctionCall:
    def __init__(self, name: str) -> None:
        self.name = name


class _ToolsExecutedEvent:
    def __init__(self) -> None:
        self.function_calls = [_FunctionCall("list_calls")]
        self.function_call_outputs = [object()]


class SCH236LiveKitProductionTests(unittest.TestCase):
    def test_latency_breakdown_separates_stt_endpoint_and_carrier_residual(self) -> None:
        breakdown = latency_breakdown(
            {"transcription_delay": 0.18, "end_of_turn_delay": 0.43},
            {
                "llm_node_ttft": 0.45,
                "tts_node_ttfb": 0.10,
                "playback_latency": 0.01,
                "e2e_latency": 1.08,
            },
            caller_side_voice_to_voice_ms=1_310,
        )

        self.assertEqual(breakdown["speech_end_to_stt_final_ms"], 180.0)
        self.assertEqual(breakdown["stt_final_to_turn_commit_ms"], 250.0)
        self.assertEqual(breakdown["worker_e2e_ms"], 1_080.0)
        self.assertEqual(breakdown["carrier_and_remote_playout_residual_ms"], 230.0)

    def test_latency_recorder_pairs_only_semantic_user_assistant_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = TurnLatencyRecorder()
            recorder.observe_user(
                text="List my calls.",
                metrics={"transcription_delay": 0.2, "end_of_turn_delay": 0.4},
            )
            self.assertIsNone(
                recorder.observe_assistant(text="Hi", metrics={}, semantic=False)
            )
            turn = recorder.observe_assistant(
                text="I found nine calls.",
                metrics={"llm_node_ttft": 0.3, "tts_node_ttfb": 0.1, "e2e_latency": 0.8},
                semantic=True,
            )
            payload = recorder.finalize(Path(directory) / "latency-stages.json")

            self.assertEqual(turn["turn"], 1)
            self.assertEqual(payload["turn_count"], 1)
            self.assertNotIn("List my calls", json.dumps(payload))
            self.assertEqual(
                json.loads((Path(directory) / "latency-stages.json").read_text())["schema"],
                "openphonex.livekit_turn_latency.v1",
            )

    def test_pinned_livekit_tool_event_reads_calls_not_outputs(self) -> None:
        self.assertEqual(executed_tool_names(_ToolsExecutedEvent()), ["list_calls"])

    def test_interrupted_greeting_fragment_is_not_a_semantic_answer(self) -> None:
        greeting = "Tere! Olete helistanud OpenPhonexile. Kuidas saan aidata?"
        self.assertTrue(
            is_greeting_fragment("Tere! Olete helistanud OpenPhonexile. Kuidas saan", greeting)
        )
        self.assertFalse(is_greeting_fragment("Leidsin üksteist kõnet.", greeting))

    def test_restart_reconciles_stale_media_and_finalizes_billing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "call-1.json"
            session.write_text(json.dumps({
                "call_id": "call-1",
                "room_name": "room-1",
                "started_at": "2026-07-15T00:00:00Z",
                "last_heartbeat_at": "2026-07-15T00:00:07Z",
                "status": "media_started",
            }))
            reports: list[dict[str, object]] = []
            deleted: list[str] = []

            verdict = reconcile_stale_sessions(
                root,
                now="2026-07-15T00:01:00Z",
                stale_after_seconds=10,
                room_is_active=lambda _room: False,
                report=lambda payload: reports.append(payload) or {
                    "canonical_type": "call.ended",
                    "billing": {"status": "settled"},
                },
                delete_room=deleted.append,
            )

            self.assertEqual(verdict["reconciled"], ["call-1"])
            self.assertEqual(reports[0]["event_type"], "session.reconciled")
            self.assertEqual(reports[0]["observed_at"], "2026-07-15T00:00:07Z")
            self.assertEqual(deleted, ["room-1"])
            self.assertEqual(json.loads(session.read_text())["status"], "reconciled")

    def test_room_sidecar_builds_synchronized_semantic_pr79_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            # Agent greeting, caller turn, then semantic response. Arrival
            # timestamps mark the end of each real-time PCM chunk.
            recorder.append_pcm("agent", b"\x10\x27" * 4_000, received_at=100.5)
            recorder.mark_state("agent", "listening", "speaking", at=100.0)
            recorder.mark_state("agent", "speaking", "listening", at=100.5)
            recorder.append_pcm("caller", b"\x20\x4e" * 3_200, received_at=101.4)
            recorder.mark_state("caller", "listening", "speaking", at=101.0)
            recorder.mark_state("caller", "speaking", "listening", at=101.4)
            recorder.add_text("caller", "Palun kontrolli minu kõnesid.", at=101.4)
            recorder.append_pcm("agent", b"\x30\x75" * 4_800, received_at=102.2)
            recorder.mark_state("agent", "listening", "speaking", at=101.6)
            recorder.mark_state("agent", "speaking", "listening", at=102.2)
            recorder.add_text("agent", "Teil ei ole veel kõnesid.", at=101.6)

            summary = recorder.finalize()

            self.assertEqual(summary["semantic_turns"], 1)
            _validate_pr79_artifact_set(root)
            with wave.open(str(root / "caller.wav"), "rb") as caller, wave.open(
                str(root / "agent.wav"), "rb"
            ) as agent:
                self.assertEqual(caller.getnframes(), agent.getnframes())
                self.assertEqual(caller.getframerate(), 8_000)
            transcript = json.loads((root / "transcript.json").read_text())
            self.assertEqual(
                transcript["semantic_response_audio_policy"],
                referee_semantic_response_audio_policy(),
            )
            self.assertEqual(transcript["turns"][0]["user"], "Palun kontrolli minu kõnesid.")
            self.assertEqual(transcript["turns"][0]["agent_audio"], "turn-001-agent.wav")

    def test_livekit_and_pr79_referee_share_the_exact_audio_policy(self) -> None:
        self.assertEqual(
            livekit_semantic_response_audio_policy(),
            referee_semantic_response_audio_policy(),
        )

    def test_livekit_effective_stack_identity_is_secret_free_and_self_verifying(self) -> None:
        config = _config(language="en")
        identity = effective_voice_stack_identity(config)

        self.assertTrue(valid_effective_voice_stack_identity(identity))
        self.assertNotIn("secret", json.dumps(identity))
        self.assertEqual(
            identity["configuration"]["measurement"]["semantic_response_audio_policy"],
            referee_semantic_response_audio_policy(),
        )

    def test_realtime_config_is_the_only_semantic_stack_source(self) -> None:
        profile = profile_from_realtime_config(_config())

        self.assertEqual(profile.language, "et")
        self.assertEqual(profile.stt_language, "et")
        self.assertEqual(profile.llm_model, "gemini-2.5-flash-lite")
        self.assertEqual(profile.tts_model, "eleven_v3")
        self.assertEqual(profile.voice_id, "voice_et_1")
        self.assertIn("list_calls", profile.instructions)
        self.assertEqual(profile.first_message, "Tere! Kuidas saan aidata?")

        bad = _config()
        bad["voice"]["registry_name"] = "unknown"
        with self.assertRaisesRegex(ProductionConfigurationError, "ElevenLabs"):
            profile_from_realtime_config(bad)

    # Source-inspection guard over a private control-plane seed script that is
    # deliberately not part of the OSS export; skips in the public repo.
    @unittest.skipUnless(
        Path("scripts/seed_gate2_test_org.py").exists(),
        "private-repo source-inspection test",
    )
    def test_seed_prompt_does_not_let_latency_questions_trigger_call_tools(self) -> None:
        source = Path("scripts/seed_gate2_test_org.py").read_text()

        self.assertIn("Never invoke list_calls unless the caller explicitly", source)
        self.assertIn("asks to retrieve, list, or review those records", source)
        self.assertIn("Do not invoke it for greetings, latency tests", source)

    def test_allowlisted_mcp_schema_becomes_model_callable_and_audited(self) -> None:
        session = _Session()
        tools = build_livekit_model_tools(
            session,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].raw_schema["name"], "list_calls")
        self.assertEqual(tools[0].raw_schema["parameters"]["required"], [])
        self.assertNotIn(
            "organization_id", tools[0].raw_schema["parameters"]["properties"]
        )
        self.assertNotIn(
            "project_id", tools[0].raw_schema["parameters"]["properties"]
        )
        output = asyncio.run(tools[0].function({}))
        self.assertEqual(
            session.calls,
            [
                (
                    "list_calls",
                    {"organization_id": "org_1", "project_id": "project_1"},
                )
            ],
        )
        self.assertEqual(json.loads(output)["result"]["content"][0]["json"]["calls"], [])

    def test_model_cannot_override_authenticated_tool_tenant(self) -> None:
        session = _Session()
        tools = build_livekit_model_tools(
            session,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )

        asyncio.run(
            tools[0].function(
                {"organization_id": "org_attacker", "project_id": "project_attacker"}
            )
        )

        self.assertEqual(
            session.calls,
            [
                (
                    "list_calls",
                    {"organization_id": "org_1", "project_id": "project_1"},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()

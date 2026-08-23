from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

from livekit_agent.production import (
    PreMediaTermination,
    ProductionConfigurationError,
    SemanticTurnAdmission,
    WorkerLocalEndCall,
    build_livekit_model_tools,
    cleanup_livekit_room,
    executed_tool_names,
    is_greeting_fragment,
    post_livekit_worker_heartbeat,
    PeriodicWorkerHeartbeat,
    component_model_labels,
    profile_from_realtime_config,
    service_heartbeat_environ,
    terminate_pre_media_call,
    worker_heartbeat_metadata,
    worker_heartbeats_enabled,
)
from livekit_agent.livekit_evidence import SynchronizedEvidenceRecorder
from livekit_agent.control_plane import (
    Gate2CallSession,
    _validate_pr79_artifact_set,
)
from livekit_agent.component_billing import SuccessfulToolUsageCounter
from livekit_agent.evidence_contract import (
    effective_voice_stack_identity,
    semantic_response_audio_policy as livekit_semantic_response_audio_policy,
    valid_effective_voice_stack_identity,
)
from openphonex_gateway.realtime.semantic_audio_policy import (
    semantic_response_audio_policy as referee_semantic_response_audio_policy,
)
from livekit_agent.reconciliation import reconcile_stale_sessions


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


class _EndCallSpeech:
    def __init__(
        self,
        events: list[str],
        *,
        wait_for: asyncio.Event | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.wait_for = wait_for
        self.error = error

    async def wait_for_playout(self) -> None:
        self.events.append("playout")
        if self.wait_for is not None:
            await self.wait_for.wait()

    def exception(self) -> BaseException | None:
        return self.error


class _EndCallAgentSession:
    def __init__(self, speech: _EndCallSpeech, events: list[str]) -> None:
        self.speech = speech
        self.events = events
        self.say_calls: list[tuple[str, bool]] = []
        self.shutdown_calls: list[bool] = []

    def say(self, message: str, *, allow_interruptions: bool):
        self.events.append("say")
        self.say_calls.append((message, allow_interruptions))
        return self.speech

    def shutdown(self, *, drain: bool) -> None:
        self.events.append("shutdown")
        self.shutdown_calls.append(drain)


class _CleanupCall:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def cleanup(self, *, remove_participant, delete_room) -> None:
        self.events.append("control_plane_cleanup")
        remove_participant()
        delete_room()


async def _delete_room(events: list[str]) -> None:
    events.append("room_deleted")


class _MixedToolSession:
    def __init__(self) -> None:
        self.config = {"organization_id": "org_1", "project_id": "project_1"}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def tool_definitions(self) -> list[dict[str, object]]:
        return [
            {
                "name": "end_call",
                "inputSchema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            },
            {
                "name": "check_availability",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
            {
                "name": "search_knowledge_base",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        ]

    def invoke_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((name, arguments))
        return {"name": name, "arguments": arguments}


class _FunctionCall:
    def __init__(self, name: str) -> None:
        self.name = name


class _ToolsExecutedEvent:
    def __init__(self) -> None:
        self.function_calls = [_FunctionCall("list_calls")]
        self.function_call_outputs = [object()]


class _HeartbeatTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method: str, path: str, payload: dict) -> dict:
        self.calls.append((method, path, payload))
        return {"id": "whb_1", **payload}


class SCH236LiveKitProductionTests(unittest.TestCase):
    def test_pre_media_failure_removes_participant_and_room_before_sip_failure(self) -> None:
        async def scenario() -> tuple[dict[str, object], list[str]]:
            events: list[str] = []

            class _Adapter:
                def report_failed_before_media(self, **_kwargs):
                    events.append("sip_failed")
                    return {"canonical_type": "call.failed", "billing": {"status": "released"}}

            class _Call:
                def cleanup(self, **_kwargs):
                    events.append("call_cleaned")

            async def remove_participant() -> None:
                events.append("participant_removed")

            async def delete_room() -> None:
                events.append("room_deleted")

            result = await terminate_pre_media_call(
                call=_Call(),
                adapter=_Adapter(),
                call_id="call_1",
                reason="provider credentials bad",
                remove_participant=remove_participant,
                delete_room=delete_room,
            )
            return result, events

        result, events = asyncio.run(scenario())

        self.assertEqual(
            events,
            ["participant_removed", "room_deleted", "call_cleaned", "sip_failed"],
        )
        self.assertEqual(result["reason"], "pre_media_rejected")
        self.assertTrue(result["participant_terminated"])
        self.assertTrue(result["room_terminated"])
        self.assertTrue(result["lifecycle_recorded"])

    def test_cancelled_pre_media_waiter_cannot_abandon_cleanup(self) -> None:
        async def scenario() -> list[str]:
            events: list[str] = []
            participant_started = asyncio.Event()
            release_participant = asyncio.Event()

            async def remove_participant() -> None:
                participant_started.set()
                await release_participant.wait()
                events.append("participant_removed")

            async def delete_room() -> None:
                events.append("room_deleted")

            terminator = PreMediaTermination(
                remove_participant=remove_participant,
                delete_room=delete_room,
            )
            task = asyncio.create_task(terminator.terminate(reason="worker_cancelled"))
            await participant_started.wait()
            task.cancel()
            release_participant.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return events

        self.assertCountEqual(
            asyncio.run(scenario()), ["participant_removed", "room_deleted"]
        )

    def test_livekit_process_posts_generic_worker_heartbeat_without_secrets(self) -> None:
        transport = _HeartbeatTransport()
        response = post_livekit_worker_heartbeat(
            transport,
            {
                "OPENPHONEX_WORKER_ID": "livekit-prod",
                "OPENPHONEX_WORKER_API_TOKEN": "must-not-leak",
            },
            process_id=42,
        )

        self.assertEqual(response["worker_id"], "livekit-prod-42")
        self.assertEqual(transport.calls[0][0:2], ("POST", "/v1/worker-heartbeats"))
        payload = transport.calls[0][2]
        self.assertEqual(payload["worker_type"], "livekit_agent")
        self.assertEqual(payload["metadata"]["media_stack"], "livekit")
        self.assertNotIn("must-not-leak", json.dumps(payload))

    def test_worker_heartbeats_default_on_and_allow_an_explicit_self_host_opt_out(self) -> None:
        self.assertTrue(worker_heartbeats_enabled({}))
        for value in ("0", "false", "no", "off", "FALSE"):
            with self.subTest(value=value):
                self.assertFalse(worker_heartbeats_enabled({"OPENPHONEX_WORKER_HEARTBEAT_ENABLED": value}))

    def test_service_heartbeat_reports_exact_revision_without_a_job_process(self) -> None:
        posted = threading.Event()
        payloads: list[dict[str, object]] = []

        def send() -> None:
            payloads.append({"revision": "a" * 40})
            posted.set()

        heartbeat = PeriodicWorkerHeartbeat(
            send=send,
            interval_seconds=60,
            on_failure=lambda _exc: self.fail("heartbeat unexpectedly failed"),
        )
        self.assertTrue(heartbeat.start())
        self.assertFalse(heartbeat.start(), "service start must not create a duplicate loop")
        self.assertTrue(posted.wait(timeout=1), "the idle worker must heartbeat before any job")
        started = time.monotonic()
        heartbeat.close(timeout_seconds=0.1)
        self.assertLess(time.monotonic() - started, 0.5, "shutdown must be bounded")
        self.assertEqual(payloads, [{"revision": "a" * 40}])

    def test_idle_service_heartbeat_has_its_own_exact_revision_row(self) -> None:
        service_environ = service_heartbeat_environ(
            {
                "OPENPHONEX_WORKER_ID": "livekit-prod",
                "OPENPHONEX_VOICE_IMAGE_REVISION": "b" * 40,
            }
        )
        payload = post_livekit_worker_heartbeat(
            _HeartbeatTransport(), service_environ, process_id=7
        )
        self.assertEqual(payload["worker_id"], "livekit-prod-service-7")
        # The revision row comes from the platform release-canary contract,
        # which the public self-host export deliberately does not ship -- there
        # `production` falls back to its documented no-op and the row is
        # legitimately absent. This file IS exported, so it must assert the
        # shape its own build produces rather than a platform-only one. The
        # branch cannot hide a broken contract: the private
        # test_minimal_voice_release_canary.py asserts worker_heartbeat_metadata
        # directly, including that it drops a malformed revision.
        if worker_heartbeat_metadata(service_environ):
            self.assertEqual(payload["metadata"]["runtime"]["image_revision"], "b" * 40)
        else:
            self.assertNotIn("runtime", payload["metadata"])

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

    def test_poisoned_session_does_not_wedge_the_cycle_and_is_reported_by_call_id(self) -> None:
        # A session the control plane cannot finalize used to raise an
        # uncaught RuntimeError from inside the sweep loop, aborting the
        # whole cycle and leaving every OTHER stale session (sorted
        # alphabetically AFTER the poisoned one) unreconciled -- forever,
        # since each following cycle hit the exact same record first.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # "call-bad" sorts before "call-good", so a cycle-aborting bug
            # would prevent call-good from ever being reached in this cycle.
            (root / "call-bad.json").write_text(json.dumps({
                "call_id": "call-bad",
                "room_name": "room-bad",
                "last_heartbeat_at": "2026-07-15T00:00:07Z",
                "status": "media_started",
            }))
            (root / "call-good.json").write_text(json.dumps({
                "call_id": "call-good",
                "room_name": "room-good",
                "last_heartbeat_at": "2026-07-15T00:00:07Z",
                "status": "media_started",
            }))
            deleted: list[str] = []

            def report(payload: dict[str, object]) -> dict[str, object]:
                if payload["call_id"] == "call-bad":
                    # Shaped exactly like the production incident: a terminal
                    # call.ended response whose billing verdict the control
                    # plane never actually reached.
                    return {"canonical_type": "call.ended", "billing": None}
                return {"canonical_type": "call.ended", "billing": {"status": "settled"}}

            verdict = reconcile_stale_sessions(
                root,
                now="2026-07-15T00:01:00Z",
                stale_after_seconds=10,
                room_is_active=lambda _room: False,
                report=report,
                delete_room=deleted.append,
            )

            self.assertEqual(verdict["reconciled"], ["call-good"])
            self.assertEqual(deleted, ["room-good"])
            self.assertEqual(len(verdict["failed"]), 1)
            failure = verdict["failed"][0]
            self.assertEqual(failure["call_id"], "call-bad")
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertEqual(
                failure["error_message"],
                "control plane did not finalize reconciled session billing",
            )
            self.assertIn("RuntimeError", failure["traceback"])
            # The poisoned record is left untouched (not marked reconciled,
            # not deleted) so it is retried -- and reported -- next cycle.
            self.assertEqual(
                json.loads((root / "call-bad.json").read_text())["status"], "media_started"
            )
            self.assertEqual(
                json.loads((root / "call-good.json").read_text())["status"], "reconciled"
            )

    def test_authoritative_holds_and_a_prior_sip_failure_both_close_a_reconciled_session(self) -> None:
        # The actual production incident shape: a call whose worker crashed
        # (ElevenLabs TTSError) before it could report native hosted-AI-
        # component usage settles into "awaiting_usage" -- a deliberate,
        # documented, authoritative hold (see
        # hosted_ai_component_usage.py::_await_component_usage), never an
        # estimate and never a failure. The reconciler's own accepted set was
        # narrower than the control plane's actual contract and rejected it
        # forever. "released"/"payment_due" are the same class of already-
        # authoritative verdict on other billing branches.
        for status in ("awaiting_usage", "released", "payment_due"):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "call-1.json").write_text(json.dumps({
                        "call_id": "call-1",
                        "room_name": "room-1",
                        "last_heartbeat_at": "2026-08-11T19:04:59Z",
                        "status": "media_started",
                    }))

                    verdict = reconcile_stale_sessions(
                        root,
                        now="2026-08-11T19:05:16Z",
                        stale_after_seconds=15,
                        room_is_active=lambda _room: False,
                        report=lambda _payload: {
                            "canonical_type": "call.ended",
                            "billing": {"status": status},
                        },
                        delete_room=lambda _room: None,
                    )

                    self.assertEqual(verdict["reconciled"], ["call-1"])
                    self.assertEqual(verdict["failed"], [])

    def test_reconciliation_sweep_after_a_prior_sip_failure_still_closes_the_session(self) -> None:
        # call_lifecycle.py deliberately remaps a session.reconciled sweep to
        # canonical_type "call.failed" when the SIP layer already reported
        # this call failed, preserving the failed terminal state instead of
        # re-settling it as billable. Treating any non-"call.ended" result as
        # a rejection (the pre-fix behavior) fights that intentional path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "call-1.json").write_text(json.dumps({
                "call_id": "call-1",
                "room_name": "room-1",
                "last_heartbeat_at": "2026-07-15T00:00:07Z",
                "status": "media_started",
            }))
            deleted: list[str] = []

            verdict = reconcile_stale_sessions(
                root,
                now="2026-07-15T00:01:00Z",
                stale_after_seconds=10,
                room_is_active=lambda _room: False,
                report=lambda _payload: {
                    "canonical_type": "call.failed",
                    "billing": {"status": "released"},
                },
                delete_room=deleted.append,
            )

            self.assertEqual(verdict["reconciled"], ["call-1"])
            self.assertEqual(verdict["failed"], [])
            self.assertEqual(deleted, ["room-1"])

    def test_a_genuinely_unauthoritative_verdict_is_still_rejected(self) -> None:
        # Regression guard for the widened accepted set: a status outside the
        # authoritative set (e.g. still "reserved", meaning the control plane
        # never actually finalized anything) must still be treated as a
        # failure, not silently accepted.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "call-1.json").write_text(json.dumps({
                "call_id": "call-1",
                "room_name": "room-1",
                "last_heartbeat_at": "2026-07-15T00:00:07Z",
                "status": "media_started",
            }))

            verdict = reconcile_stale_sessions(
                root,
                now="2026-07-15T00:01:00Z",
                stale_after_seconds=10,
                room_is_active=lambda _room: False,
                report=lambda _payload: {
                    "canonical_type": "call.ended",
                    "billing": {"status": "reserved"},
                },
                delete_room=lambda _room: None,
            )

            self.assertEqual(verdict["reconciled"], [])
            self.assertEqual(len(verdict["failed"]), 1)
            self.assertEqual(
                verdict["failed"][0]["error_message"],
                "reconciled session has no authoritative billing verdict",
            )

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
            recorder.add_terminal_message("Aitäh helistamast. Head aega.", at=102.2)

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
            self.assertEqual(transcript["summary"]["terminal_message_count"], 1)
            self.assertEqual(
                transcript["terminal_messages"],
                [
                    {
                        "role": "agent",
                        "text": "Aitäh helistamast. Head aega.",
                        "at_ms": 2200,
                    }
                ],
            )

    def test_room_sidecar_preserves_short_call_without_a_semantic_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            recorder.append_pcm("agent", b"\x10\x27" * 4_000, received_at=100.5)
            recorder.mark_state("agent", "listening", "speaking", at=100.0)
            recorder.mark_state("agent", "speaking", "listening", at=100.5)

            summary = recorder.finalize(ended_at=100.6)

            self.assertEqual(summary["semantic_turns"], 0)
            self.assertEqual(summary["semantic_evidence_status"], "unavailable_no_linked_turn")
            self.assertTrue(summary["short_call_without_caller_speech"])
            transcript = json.loads((root / "transcript.json").read_text())
            self.assertEqual(transcript["summary"]["turn_count"], 0)
            self.assertEqual(transcript["turns"], [])
            for name in ("caller.wav", "agent.wav", "mixed.wav", "transcript.json"):
                self.assertTrue((root / name).is_file())

    def test_line_noise_without_vad_or_transcript_is_still_a_short_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedEvidenceRecorder(Path(directory), epoch=100.0)
            recorder.append_pcm("caller", b"\x20\x4e" * 3_200, received_at=100.4)

            summary = recorder.finalize(ended_at=100.5)

            self.assertEqual(summary["semantic_turns"], 0)
            self.assertTrue(summary["caller_audio_observed"])
            self.assertTrue(summary["short_call_without_caller_speech"])

    def test_no_room_audio_finalize_remains_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = SynchronizedEvidenceRecorder(Path(directory), epoch=100.0)

            with self.assertRaisesRegex(RuntimeError, "no room audio"):
                recorder.finalize(ended_at=100.1)
            with self.assertRaisesRegex(RuntimeError, "no room audio"):
                recorder.finalize(ended_at=100.2)

    def test_livekit_stream_frames_use_media_order_when_callbacks_are_batched(self) -> None:
        """Queued frames must not overwrite each other at one callback time."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            stream_id = object()
            first = b"\x11\x11" * 160
            second = b"\x22\x22" * 160

            # An event-loop stall lets AudioStream queue both 20 ms frames;
            # draining that queue gives them the same observation time. The
            # old callback-time placement wrote both into the same WAV range.
            recorder.append_stream_pcm(
                "agent", first, stream_id=stream_id, received_at=100.04
            )
            recorder.append_stream_pcm(
                "agent", second, stream_id=stream_id, received_at=100.04
            )
            recorder.finalize(ended_at=100.06)

            with wave.open(str(root / "agent.wav"), "rb") as recording:
                payload = recording.readframes(recording.getnframes())
            self.assertEqual(payload[160 * 2 : 320 * 2], first)
            self.assertEqual(payload[320 * 2 : 480 * 2], second)

    def test_republished_livekit_stream_preserves_real_timeline_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            first = b"\x11\x11" * 160
            republished = b"\x22\x22" * 160

            recorder.append_stream_pcm(
                "agent", first, stream_id="publication-1", received_at=100.02
            )
            recorder.append_stream_pcm(
                "agent", republished, stream_id="publication-2", received_at=100.10
            )
            recorder.finalize(ended_at=100.10)

            with wave.open(str(root / "agent.wav"), "rb") as recording:
                payload = recording.readframes(recording.getnframes())
            self.assertEqual(payload[: 160 * 2], first)
            self.assertEqual(payload[160 * 2 : 640 * 2], b"\x00" * (480 * 2))
            self.assertEqual(payload[640 * 2 : 800 * 2], republished)

    def test_livekit_stream_preserves_silence_when_frames_pause(self) -> None:
        """A participant may stop publishing frames between utterances."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            stream_id = object()
            first = b"\x11\x11" * 160
            second = b"\x22\x22" * 160

            recorder.append_stream_pcm(
                "agent", first, stream_id=stream_id, received_at=100.02
            )
            recorder.append_stream_pcm(
                "agent", second, stream_id=stream_id, received_at=100.10
            )
            recorder.finalize(ended_at=100.10)

            with wave.open(str(root / "agent.wav"), "rb") as recording:
                payload = recording.readframes(recording.getnframes())
            self.assertEqual(payload[: 160 * 2], first)
            self.assertEqual(payload[160 * 2 : 640 * 2], b"\x00" * (480 * 2))
            self.assertEqual(payload[640 * 2 : 800 * 2], second)

    def test_stream_start_near_epoch_does_not_overlap_the_second_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = SynchronizedEvidenceRecorder(root, epoch=100.0)
            first = b"\x11\x11" * 160
            second = b"\x22\x22" * 160
            stream_id = object()

            # The recorder can subscribe less than one 20 ms frame after its
            # epoch. Clamping the first frame to sample zero must advance the
            # cursor by the full frame, not by the shorter wall-clock offset.
            recorder.append_stream_pcm(
                "caller", first, stream_id=stream_id, received_at=100.005
            )
            recorder.append_stream_pcm(
                "caller", second, stream_id=stream_id, received_at=100.025
            )
            recorder.finalize(ended_at=100.04)

            with wave.open(str(root / "caller.wav"), "rb") as recording:
                payload = recording.readframes(recording.getnframes())
            self.assertEqual(payload[: 160 * 2], first)
            self.assertEqual(payload[160 * 2 : 320 * 2], second)

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

    def test_model_labels_come_from_the_resolved_profile_not_the_raw_config(self) -> None:
        # Telemetry must record what this worker BUILT. Reading the raw config
        # would record what it was asked for, which is a different claim.
        config = _config()
        labels = component_model_labels(profile_from_realtime_config(config))

        self.assertEqual(labels, {"stt": "nova-3", "llm": "gemini-2.5-flash-lite", "tts": "eleven_v3"})

    def test_sinhala_model_labels_follow_the_registry_that_actually_runs(self) -> None:
        # si-LK routes STT to Google Chirp 2 and TTS to Gemini-TTS. The labels
        # must track the registry rather than assuming the ElevenLabs default.
        config = _config(language="si")
        config["transcriber"] = {"registry_name": "google", "model": "chirp_2", "language": "si"}
        config["voice"] = {
            "registry_name": "gemini_tts",
            "model": "gemini-2.5-flash-preview-tts",
            "voice_id": "voice_si_1",
            "language": "si",
        }
        labels = component_model_labels(profile_from_realtime_config(config))

        self.assertEqual(labels["stt"], "chirp_2")
        self.assertEqual(labels["tts"], "gemini-2.5-flash-preview-tts")

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

    def test_sinhala_realtime_config_requires_google_stt_and_gemini_tts(self) -> None:
        config = {
            "agent_id": "agent_1",
            "organization_id": "org_1",
            "project_id": "project_1",
            "language": "si",
            "system_prompt": "Answer in Sinhala about the caller's loan payment.",
            "first_message": "Ayubowan!",
            "tools": [],
            "recording_enabled": True,
            "transcriber": {"registry_name": "google", "model": "chirp_2", "language": "si"},
            "model": {"registry_name": "gemini", "model": "gemini-2.5-flash-lite"},
            "voice": {
                "registry_name": "gemini_tts",
                "model": "gemini-2.5-flash-preview-tts",
                "voice_id": "Kore",
                "language": "si",
            },
        }

        profile = profile_from_realtime_config(config)

        self.assertEqual(profile.language, "si")
        self.assertEqual(profile.stt_language, "si-LK")
        self.assertEqual(profile.stt_registry, "google")
        self.assertEqual(profile.tts_registry, "gemini_tts")
        self.assertEqual(profile.voice_id, "Kore")

        # Deepgram cannot serve si-LK STT at all.
        deepgram_stt = dict(config)
        deepgram_stt["transcriber"] = {"registry_name": "deepgram", "model": "nova-3", "language": "si"}
        with self.assertRaisesRegex(ProductionConfigurationError, "Sinhala"):
            profile_from_realtime_config(deepgram_stt)

        # ElevenLabs cannot serve si-LK TTS at all.
        elevenlabs_tts = dict(config)
        elevenlabs_tts["voice"] = {
            "registry_name": "elevenlabs",
            "model": "eleven_v3",
            "voice_id": "voice_1",
            "language": "si",
        }
        with self.assertRaisesRegex(ProductionConfigurationError, "Sinhala"):
            profile_from_realtime_config(elevenlabs_tts)

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
        tool_usage = SuccessfulToolUsageCounter()
        tools = build_livekit_model_tools(
            session,
            tool_usage_counter=tool_usage,
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
        self.assertEqual(tool_usage.snapshot(), {"list_calls": 1})

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

    def test_pinned_livekit_end_call_returns_none_while_other_tools_keep_json_dispatch(self) -> None:
        session = _MixedToolSession()
        local_calls: list[dict[str, object]] = []
        tool_usage = SuccessfulToolUsageCounter()

        async def end_call(arguments: dict[str, object]) -> dict[str, object]:
            local_calls.append(arguments)
            return {"ok": True, "status": "teardown_requested"}

        tools = build_livekit_model_tools(
            session,
            end_call=end_call,
            tool_usage_counter=tool_usage,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )
        by_name = {tool.raw_schema["name"]: tool for tool in tools}

        end_output = asyncio.run(by_name["end_call"].function({"message": "Goodbye."}))
        webhook_output = asyncio.run(
            by_name["check_availability"].function({"query": "Tomorrow"})
        )
        knowledge_output = asyncio.run(
            by_name["search_knowledge_base"].function({"query": "Hours"})
        )

        self.assertEqual(local_calls, [{"message": "Goodbye."}])
        self.assertEqual(
            session.calls,
            [
                (
                    "check_availability",
                    {
                        "query": "Tomorrow",
                        "organization_id": "org_1",
                        "project_id": "project_1",
                    },
                ),
                (
                    "search_knowledge_base",
                    {
                        "query": "Hours",
                        "organization_id": "org_1",
                        "project_id": "project_1",
                    },
                ),
            ],
        )
        self.assertIsNone(end_output)
        self.assertIsInstance(webhook_output, str)
        self.assertIsInstance(knowledge_output, str)
        self.assertEqual(json.loads(webhook_output)["name"], "check_availability")
        self.assertEqual(json.loads(knowledge_output)["name"], "search_knowledge_base")
        self.assertEqual(
            tool_usage.snapshot(),
            {
                "check_availability": 1,
                "end_call": 1,
                "search_knowledge_base": 1,
            },
        )

    def test_end_call_fails_closed_when_the_worker_handler_is_missing(self) -> None:
        session = _MixedToolSession()
        tools = build_livekit_model_tools(
            session,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )
        end_call = next(tool for tool in tools if tool.raw_schema["name"] == "end_call")

        with self.assertRaisesRegex(
            ProductionConfigurationError, "worker-local termination handler"
        ):
            asyncio.run(end_call.function({}))

        self.assertEqual(session.calls, [])

    def test_end_call_ack_returns_before_scheduler_can_play_goodbye(self) -> None:
        async def scenario() -> tuple[dict[str, object], dict[str, object], list[str], _EndCallAgentSession, list[str]]:
            events: list[str] = []
            tool_returned = asyncio.Event()
            terminal_messages: list[str] = []
            session = _EndCallAgentSession(
                _EndCallSpeech(events, wait_for=tool_returned), events
            )
            action = WorkerLocalEndCall(
                session=session,
                language="et",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish"),
                record_final_message=terminal_messages.append,
                emit=lambda event, **_details: events.append(event),
            )
            acknowledgement = await asyncio.wait_for(
                action({"message": "  Head aega.  "}), timeout=0.05
            )
            self.assertEqual(session.say_calls, [])
            tool_returned.set()
            completion = await action.wait_for_completion()
            assert completion is not None
            return acknowledgement, completion, events, session, terminal_messages

        acknowledgement, completion, events, session, terminal_messages = asyncio.run(
            scenario()
        )

        self.assertEqual(
            acknowledgement,
            {"ok": True, "status": "termination_scheduled", "playout": "pending"},
        )
        self.assertEqual(
            completion,
            {"ok": True, "status": "teardown_requested", "playout": "completed"},
        )
        self.assertEqual(session.say_calls, [("Head aega.", False)])
        self.assertEqual(session.shutdown_calls, [False])
        self.assertEqual(terminal_messages, ["Head aega."])
        self.assertEqual(
            events,
            ["ending", "say", "playout", "shutdown", "finish", "end_call"],
        )

    def test_end_call_is_idempotent_for_concurrent_model_invocations(self) -> None:
        async def scenario() -> tuple[list[str], _EndCallAgentSession, list[dict[str, object]]]:
            events: list[str] = []
            unblock_playout = asyncio.Event()
            session = _EndCallAgentSession(
                _EndCallSpeech(events, wait_for=unblock_playout), events
            )
            action = WorkerLocalEndCall(
                session=session,
                language="en",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish"),
                emit=lambda event, **_details: events.append(event),
            )
            first, second = await asyncio.gather(
                action({"message": "Goodbye."}),
                action({"message": "A different goodbye."}),
            )
            await asyncio.sleep(0)
            unblock_playout.set()
            completion = await action.wait_for_completion()
            assert completion is not None
            return events, session, [first, second, completion]

        events, session, results = asyncio.run(scenario())

        self.assertEqual(session.say_calls, [("Goodbye.", False)])
        self.assertEqual(session.shutdown_calls, [False])
        self.assertEqual(events.count("ending"), 1)
        self.assertEqual(events.count("finish"), 1)
        self.assertEqual(events.count("end_call"), 1)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[2]["playout"], "completed")

    def test_cancelled_waiter_cannot_cancel_worker_owned_end_call_teardown(self) -> None:
        async def scenario() -> tuple[list[str], _EndCallAgentSession, dict[str, object]]:
            events: list[str] = []
            unblock_playout = asyncio.Event()
            session = _EndCallAgentSession(
                _EndCallSpeech(events, wait_for=unblock_playout), events
            )
            action = WorkerLocalEndCall(
                session=session,
                language="en",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish"),
                emit=lambda event, **_details: events.append(event),
            )
            await action({"message": "Goodbye."})
            cancelled = asyncio.create_task(action.wait_for_completion())
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            unblock_playout.set()
            result = await action.wait_for_completion()
            assert result is not None
            return events, session, result

        events, session, result = asyncio.run(scenario())

        self.assertEqual(session.say_calls, [("Goodbye.", False)])
        self.assertEqual(session.shutdown_calls, [False])
        self.assertEqual(result["playout"], "completed")
        self.assertEqual(events.count("finish"), 1)
        self.assertEqual(events.count("end_call"), 1)

    def test_caller_first_disconnect_does_not_start_a_second_end_call(self) -> None:
        async def scenario() -> tuple[dict[str, object], _EndCallAgentSession, list[str]]:
            events: list[str] = []
            session = _EndCallAgentSession(_EndCallSpeech(events), events)
            action = WorkerLocalEndCall(
                session=session,
                language="en",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish"),
                emit=lambda event, **_details: events.append(event),
                is_finished=lambda: True,
            )
            return await action({"message": "Goodbye."}), session, events

        result, session, events = asyncio.run(scenario())

        self.assertEqual(
            result,
            {"ok": True, "status": "already_finished", "playout": "not_started"},
        )
        self.assertEqual(session.say_calls, [])
        self.assertEqual(session.shutdown_calls, [])
        self.assertEqual(events, [])

    def test_end_call_records_the_et_fallback_goodbye_exactly_once(self) -> None:
        async def scenario() -> tuple[_EndCallAgentSession, list[str], dict[str, object]]:
            events: list[str] = []
            terminal_messages: list[str] = []
            session = _EndCallAgentSession(_EndCallSpeech(events), events)
            action = WorkerLocalEndCall(
                session=session,
                language="et",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish"),
                record_final_message=terminal_messages.append,
            )
            await action({})
            completion = await action.wait_for_completion()
            assert completion is not None
            return session, terminal_messages, completion

        session, terminal_messages, completion = asyncio.run(scenario())

        self.assertEqual(
            session.say_calls, [("Aitäh helistamast. Head aega.", False)]
        )
        self.assertEqual(terminal_messages, ["Aitäh helistamast. Head aega."])
        self.assertEqual(completion["playout"], "completed")

    def test_end_call_closes_production_semantic_admission_but_keeps_goodbye(self) -> None:
        async def scenario() -> tuple[list[str], list[str], list[str]]:
            admission = SemanticTurnAdmission()
            transcripts: list[str] = []
            conversation_items: list[str] = []
            terminal_messages: list[str] = []

            def observe_transcript(text: str) -> None:
                if admission.admits_transcript():
                    transcripts.append(text)

            def observe_conversation_item(text: str) -> None:
                if admission.admits_conversation_item():
                    conversation_items.append(text)

            session = _EndCallAgentSession(_EndCallSpeech([]), [])
            observe_transcript("before ending")
            observe_conversation_item("before ending")
            action = WorkerLocalEndCall(
                session=session,
                language="en",
                on_ending=admission.close,
                finish=lambda: None,
                record_final_message=terminal_messages.append,
            )
            await action({"message": "Goodbye."})
            observe_transcript("must not be recorded")
            observe_conversation_item("must not be recorded")
            await action.wait_for_completion()
            return transcripts, conversation_items, terminal_messages

        transcripts, conversation_items, terminal_messages = asyncio.run(scenario())

        self.assertEqual(transcripts, ["before ending"])
        self.assertEqual(conversation_items, ["before ending"])
        self.assertEqual(terminal_messages, ["Goodbye."])

    def test_end_call_finish_signal_reaches_room_finalizer(self) -> None:
        async def scenario() -> tuple[list[str], dict[str, object]]:
            events: list[str] = []
            finished = asyncio.Event()
            call = _CleanupCall(events)

            async def existing_finalizer() -> None:
                await finished.wait()
                await cleanup_livekit_room(
                    call,
                    lambda: _delete_room(events),
                )

            finalizer = asyncio.create_task(existing_finalizer())
            session = _EndCallAgentSession(_EndCallSpeech(events), events)
            action = WorkerLocalEndCall(
                session=session,
                language="en",
                on_ending=lambda: events.append("ending"),
                finish=lambda: events.append("finish") or finished.set(),
            )
            await action({"message": "Goodbye."})
            completion = await action.wait_for_completion()
            assert completion is not None
            await finalizer
            return events, completion

        events, completion = asyncio.run(scenario())

        self.assertEqual(
            events,
            ["ending", "say", "playout", "shutdown", "finish", "control_plane_cleanup", "room_deleted"],
        )
        self.assertEqual(completion["status"], "teardown_requested")

    def test_end_call_timeout_or_playout_failure_still_requests_teardown(self) -> None:
        async def scenario() -> tuple[tuple[dict[str, object], list[str], _EndCallAgentSession], tuple[dict[str, object], list[str], _EndCallAgentSession]]:
            timeout_events: list[str] = []
            timeout_session = _EndCallAgentSession(
                _EndCallSpeech(timeout_events, wait_for=asyncio.Event()), timeout_events
            )
            timeout_action = WorkerLocalEndCall(
                session=timeout_session,
                language="en",
                on_ending=lambda: timeout_events.append("ending"),
                finish=lambda: timeout_events.append("finish"),
                emit=lambda event, **_details: timeout_events.append(event),
                playout_timeout_seconds=0.001,
            )
            await timeout_action({})
            timed_out = await timeout_action.wait_for_completion()
            assert timed_out is not None

            failed_events: list[str] = []
            failed_session = _EndCallAgentSession(
                _EndCallSpeech(failed_events, error=RuntimeError("playout failed")), failed_events
            )
            failed_action = WorkerLocalEndCall(
                session=failed_session,
                language="en",
                on_ending=lambda: failed_events.append("ending"),
                finish=lambda: failed_events.append("finish"),
                emit=lambda event, **_details: failed_events.append(event),
            )
            await failed_action({})
            failed = await failed_action.wait_for_completion()
            assert failed is not None
            return (timed_out, timeout_events, timeout_session), (failed, failed_events, failed_session)

        timed_out, failed = asyncio.run(scenario())

        self.assertEqual(timed_out[0]["playout"], "timeout")
        self.assertEqual(failed[0]["playout"], "failed")
        for result, events, session in (timed_out, failed):
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "teardown_requested")
            self.assertEqual(session.shutdown_calls, [False])
            self.assertEqual(events.count("finish"), 1)
            self.assertEqual(events.count("end_call"), 1)

    def test_call_scoped_knowledge_tool_has_only_query_and_bypasses_mcp(self) -> None:
        transport = _HeartbeatTransport()
        call = Gate2CallSession(
            transport=transport,
            authorization_transport=transport,
            tool_transport=transport,
            call_id="call_knowledge_1",
            gateway_id="gateway_1",
            authorization_call_id="",
            call_authorization_token="",
            config={
                "organization_id": "org_1",
                "project_id": "project_1",
                "tools": [],
                "knowledge": {"tool": "search_knowledge_base"},
            },
            media_admitted=True,
        )

        tools = build_livekit_model_tools(
            call,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].raw_schema["name"], "search_knowledge_base")
        self.assertEqual(tools[0].raw_schema["parameters"]["required"], ["query"])
        self.assertEqual(
            set(tools[0].raw_schema["parameters"]["properties"]), {"query"}
        )
        output = asyncio.run(
            tools[0].function(
                {
                    "query": "Are towels included?",
                    "organization_id": "org_attacker",
                    "knowledge_base_id": "kb_attacker",
                }
            )
        )

        self.assertEqual(
            transport.calls,
            [
                (
                    "POST",
                    "/v1/calls/call_knowledge_1/knowledge/query",
                    {"query": "Are towels included?"},
                )
            ],
        )
        self.assertEqual(json.loads(output)["query"], "Are towels included?")
        self.assertIn("knowledge_tool_invoked_call_scoped", call.audit_events)

    def test_structured_realtime_tool_config_is_reduced_to_its_allowlisted_name(self) -> None:
        transport = _HeartbeatTransport()
        call = Gate2CallSession(
            transport=transport,
            authorization_transport=transport,
            tool_transport=transport,
            call_id="call_structured_tools",
            gateway_id="gateway_1",
            authorization_call_id="",
            call_authorization_token="",
            config={
                "organization_id": "org_1",
                "project_id": "project_1",
                "tools": [{"type": "builtin", "name": "end_call"}],
            },
        )

        self.assertEqual(call.allowed_tools, frozenset({"end_call"}))
        tools = build_livekit_model_tools(
            call,
            tool_builder=lambda function, raw_schema: _Tool(function, raw_schema),
        )
        self.assertEqual([tool.raw_schema["name"] for tool in tools], ["end_call"])
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()

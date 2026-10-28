from __future__ import annotations

import ast
import asyncio
import json
import tempfile
import threading
import time
import unittest
import wave
from enum import Enum
from types import SimpleNamespace
from pathlib import Path

from livekit_agent.production import (
    require_pre_media_participant,
    require_inbound_sip_answer,
    PreMediaTermination,
    ProductionConfigurationError,
    SemanticTurnAdmission,
    WorkerLocalEndCall,
    build_livekit_model_tools,
    cleanup_livekit_room,
    executed_tool_names,
    is_greeting_fragment,
    post_livekit_worker_heartbeat,
    pre_media_lifecycle_reason,
    PeriodicWorkerHeartbeat,
    browser_test_close_component,
    browser_test_error_metadata,
    component_model_labels,
    MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS,
    admitted_tts_text,
    await_bounded_teardown_task,
    await_bounded_production_session_close,
    close_production_session_after_room_end,
    profile_from_realtime_config,
    production_room_disconnect_handler,
    safe_component_error_diagnostics,
    session_options_with_managed_pstn_turn_taking,
    service_heartbeat_environ,
    shutdown_evidence_sidecar,
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
    is_noncaller_ambience_track,
    semantic_response_audio_policy as livekit_semantic_response_audio_policy,
    valid_effective_voice_stack_identity,
)
from skysay_gateway.evidence.semantic_audio_policy import (
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


class PreMediaParticipantPresenceTests(unittest.TestCase):

    def test_departed_and_replaced_leg_are_refused_but_same_sid_is_live(self):
        participant = SimpleNamespace(identity="caller", sid="PA_original")
        room = SimpleNamespace(remote_participants={})
        for current in (None, SimpleNamespace(identity="caller", sid="PA_reconnected")):
            room.remote_participants = {} if current is None else {"caller": current}
            with self.assertRaises(ProductionConfigurationError) as raised:
                require_pre_media_participant(room, participant)
            self.assertEqual(raised.exception.reason, "caller_left_before_media")
        room.remote_participants = {"caller": SimpleNamespace(identity="caller", sid="PA_original")}
        require_pre_media_participant(room, participant)


    def test_inbound_answer_requires_live_exact_leg_and_active_status(self):
        async def exercise():
            for outcome in ("active", "departed", "replaced", "timeout", "missing", "cancelled"):
                with self.subTest(outcome=outcome):
                    leg = SimpleNamespace(identity="caller", sid="PA_original", attributes={"sip.callStatus": "ringing"})
                    room = SimpleNamespace(remote_participants={"caller": leg})
                    async def wait():
                        if outcome == "active":
                            leg.attributes["sip.callStatus"] = "active"
                        elif outcome == "departed":
                            room.remote_participants.clear()
                            raise RuntimeError("participant disconnected")
                        elif outcome == "replaced":
                            room.remote_participants["caller"] = SimpleNamespace(sid="PA_new", attributes={"sip.callStatus": "active"})
                        elif outcome == "timeout":
                            raise TimeoutError()
                        elif outcome == "cancelled":
                            raise asyncio.CancelledError()
                        else:
                            leg.attributes.clear()
                    if outcome == "active":
                        await require_inbound_sip_answer(room, leg, wait_for_active=wait, timeout_seconds=1)
                    elif outcome == "cancelled":
                        with self.assertRaises(asyncio.CancelledError):
                            await require_inbound_sip_answer(room, leg, wait_for_active=wait, timeout_seconds=1)
                    else:
                        with self.assertRaises(ProductionConfigurationError) as raised:
                            await require_inbound_sip_answer(room, leg, wait_for_active=wait, timeout_seconds=1)
                        self.assertEqual(raised.exception.reason, "caller_left_before_media" if outcome in {"departed", "replaced"} else "inbound_answer_unconfirmed")
        asyncio.run(exercise())


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
        return {
            "result": {
                "content": [{"type": "text", "text": '{"calls": []}'}],
                "structuredContent": {"calls": []},
                "isError": False,
            }
        }


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

    def say(self, message: str, *, allow_interruptions: bool, add_to_chat_ctx: bool = True):
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


# The self-hosted gateway export ships this suite AND the worker, but relocates
# the worker to `livekit_agent/`. Both layouts are resolved so these guards run
# against the exported copy too: skipping there would switch them off in the one
# tree where drift between the two is most likely to appear first, which is
# worse than the FileNotFoundError this replaced.
_PRODUCTION_AGENT_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "apps" / "voice" / "agent" / "production_agent.py",
    Path(__file__).resolve().parents[1] / "livekit_agent" / "agent" / "production_agent.py",
)
_PRODUCTION_AGENT = next(
    (candidate for candidate in _PRODUCTION_AGENT_CANDIDATES if candidate.exists()), None
)
_worker_source_guard = unittest.skipUnless(
    _PRODUCTION_AGENT is not None, "no worker entrypoint in this tree"
)


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    """Every ``name(...)`` call anywhere under ``node``, by attribute or plain name."""
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            found.append(child)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            found.append(child)
    return found


def _mentions(node: ast.AST, name: str) -> bool:
    """Whether ``name`` appears anywhere under ``node`` as a call or a bare name."""
    if _calls_named(node, name):
        return True
    return any(
        (isinstance(child, ast.Name) and child.id == name)
        or (isinstance(child, ast.Attribute) and child.attr == name)
        for child in ast.walk(node)
    )


class SCH236LiveKitProductionTests(unittest.TestCase):
    def test_room_disconnect_session_close_is_bounded_even_when_close_ignores_cancellation(self) -> None:
        async def scenario() -> None:
            release = asyncio.Event()
            late_results: list[str] = []
            late_seen = asyncio.Event()

            def note_late_result(error_type: str) -> None:
                late_results.append(error_type)
                late_seen.set()

            class Session:
                async def aclose(self) -> None:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        await release.wait()

            result = await asyncio.wait_for(
                await_bounded_production_session_close(
                    Session(),
                    timeout_seconds=0.01,
                    cancellation_grace_seconds=0.01,
                    on_late_completion=note_late_result,
                ),
                timeout=1.0,
            )
            self.assertFalse(result.completed)
            self.assertTrue(result.cancellation_resistant)
            self.assertEqual(result.error_type, "TimeoutError")
            release.set()
            await asyncio.wait_for(late_seen.wait(), timeout=1.0)
            self.assertEqual(late_results, [""])

        asyncio.run(scenario())

    def test_detached_browser_report_preserves_late_failure_type(self) -> None:
        async def scenario() -> None:
            release = asyncio.Event()
            late_seen = asyncio.Event()
            late_results: list[str] = []

            async def resistant_report() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                    raise RuntimeError("late provider failure")

            report = asyncio.create_task(resistant_report())
            result = await await_bounded_teardown_task(
                report,
                timeout_seconds=0.01,
                cancellation_grace_seconds=0.01,
                on_late_completion=lambda error_type: (
                    late_results.append(error_type),
                    late_seen.set(),
                ),
            )
            self.assertTrue(result.cancellation_resistant)
            release.set()
            await asyncio.wait_for(late_seen.wait(), timeout=1.0)
            self.assertEqual(late_results, ["RuntimeError"])

        asyncio.run(scenario())

    def test_sidecar_and_capture_tasks_cannot_hold_terminal_settlement(self) -> None:
        async def scenario() -> None:
            release = asyncio.Event()
            capture_started = asyncio.Event()
            late: list[tuple[str, str]] = []
            late_seen = asyncio.Event()

            async def resist_cancellation(*, started: asyncio.Event | None = None) -> None:
                if started is not None:
                    started.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue

            class Sidecar:
                async def disconnect(self) -> None:
                    await resist_cancellation()

            capture = asyncio.create_task(
                resist_cancellation(started=capture_started)
            )
            await capture_started.wait()

            def note_late(phase: str, error_type: str) -> None:
                late.append((phase, error_type))
                if len(late) == 2:
                    late_seen.set()

            result = await asyncio.wait_for(
                shutdown_evidence_sidecar(
                    Sidecar(),
                    {capture},
                    timeout_seconds=0.01,
                    cancellation_grace_seconds=0.01,
                    on_late_completion=note_late,
                ),
                timeout=1.0,
            )
            self.assertTrue(result.disconnect.cancellation_resistant)
            self.assertTrue(result.captures.cancellation_resistant)
            release.set()
            await asyncio.wait_for(late_seen.wait(), timeout=1.0)
            self.assertEqual(
                sorted(late),
                [("captures", "CancelledError"), ("disconnect", "")],
            )

        asyncio.run(scenario())

    def test_sidecar_shutdown_completes_cleanly_when_room_and_captures_end(self) -> None:
        async def scenario() -> None:
            disconnected = False
            capture_cancelled = False

            class Sidecar:
                async def disconnect(self) -> None:
                    nonlocal disconnected
                    disconnected = True

            async def finish_capture() -> None:
                nonlocal capture_cancelled
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    capture_cancelled = True
                    raise

            capture = asyncio.create_task(finish_capture())
            result = await shutdown_evidence_sidecar(Sidecar(), {capture})
            self.assertTrue(disconnected)
            self.assertFalse(capture_cancelled)
            self.assertTrue(result.disconnect.completed)
            self.assertTrue(result.captures.completed)

        asyncio.run(scenario())

    def test_sdk_close_self_cancellation_is_an_action_failure_not_worker_cancellation(self) -> None:
        async def scenario() -> None:
            class Session:
                async def aclose(self) -> None:
                    raise asyncio.CancelledError

            task = asyncio.current_task()
            assert task is not None
            before = task.cancelling()
            result = await await_bounded_production_session_close(Session())
            self.assertFalse(result.completed)
            self.assertEqual(result.error_type, "CancelledError")
            self.assertEqual(task.cancelling(), before)

        asyncio.run(scenario())

    def test_room_disconnect_skips_sdk_close_task_entirely(self) -> None:
        async def scenario() -> None:
            class Session:
                def aclose(self) -> None:
                    raise AssertionError("SDK close must not start after transport loss")

            result = await close_production_session_after_room_end(
                Session(),
                room_transport_disconnected=lambda: True,
            )
            self.assertFalse(result.completed)
            self.assertFalse(result.required)
            self.assertEqual(result.skipped_reason, "room_transport_disconnected")

        asyncio.run(scenario())

    def test_browser_report_that_ignores_cancellation_cannot_hold_teardown(self) -> None:
        async def scenario() -> None:
            release = asyncio.Event()
            late_seen = asyncio.Event()
            late_results: list[str] = []

            async def resistant_report() -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()

            report = asyncio.create_task(resistant_report())
            result = await asyncio.wait_for(
                await_bounded_teardown_task(
                    report,
                    timeout_seconds=0.01,
                    cancellation_grace_seconds=0.01,
                    on_late_completion=lambda error_type: (
                        late_results.append(error_type),
                        late_seen.set(),
                    ),
                ),
                timeout=0.1,
            )
            self.assertFalse(result.completed)
            self.assertTrue(result.cancellation_resistant)
            self.assertEqual(result.error_type, "TimeoutError")
            release.set()
            await asyncio.wait_for(late_seen.wait(), timeout=0.1)
            self.assertEqual(late_results, [""])

        asyncio.run(scenario())

    def test_connected_room_still_closes_sdk_session(self) -> None:
        async def scenario() -> None:
            closed = False

            class Session:
                async def aclose(self) -> None:
                    nonlocal closed
                    closed = True

            result = await close_production_session_after_room_end(
                Session(),
                room_transport_disconnected=lambda: False,
            )
            self.assertTrue(result.completed)
            self.assertTrue(result.required)
            self.assertTrue(closed)
            self.assertEqual(result.skipped_reason, "")

        asyncio.run(scenario())

    def test_disconnect_state_is_read_when_close_coroutine_runs(self) -> None:
        async def scenario() -> None:
            disconnected = False

            class Session:
                def aclose(self) -> None:
                    raise AssertionError("late transport loss must prevent SDK close")

            close = close_production_session_after_room_end(
                Session(),
                room_transport_disconnected=lambda: disconnected,
            )
            disconnected = True
            result = await close
            self.assertFalse(result.completed)
            self.assertFalse(result.required)
            self.assertEqual(result.skipped_reason, "room_transport_disconnected")

        asyncio.run(scenario())

    def test_production_entrypoint_bounds_sdk_close_after_terminal_goodbye_join(self) -> None:
        source = self._production_agent_source()
        entrypoint = source[source.index("async def production_agent(") :]
        terminal_join = entrypoint.index("await settle_worker_terminal_end_call(")
        bounded_close = entrypoint.index("await close_production_session_after_room_end(")
        evidence_freeze = entrypoint.index("finalized = await finalize_off_loop(recorder)")
        self.assertLess(terminal_join, bounded_close)
        self.assertLess(bounded_close, evidence_freeze)
        self.assertNotIn("await session.aclose()", entrypoint)

    def test_room_disconnect_wakes_the_single_production_teardown_path_once(self) -> None:
        finishes: list[str] = []
        events: list[tuple[str, dict]] = []
        handler = production_room_disconnect_handler(
            finish=lambda: finishes.append("finished"),
            emit=lambda event, **details: events.append((event, details)),
        )

        handler("transport_closed")
        handler("duplicate")
        self.assertIn(handler, {handler})
        self.assertEqual(finishes, ["finished"])
        self.assertTrue(handler.triggered)
        self.assertEqual(
            events,
            [
                (
                    "room_disconnected",
                    {"shutdown_source": "livekit_room"},
                )
            ],
        )

    def test_production_entrypoint_skips_sdk_close_after_room_transport_disconnect(self) -> None:
        source = self._production_agent_source()
        entrypoint = source[source.index("async def production_agent(") :]
        room_signal = entrypoint.index(
            "room_disconnect = production_room_disconnect_handler("
        )
        teardown = entrypoint.index("await settle_worker_terminal_end_call(")
        close_decision = entrypoint.index(
            "await close_production_session_after_room_end(", teardown
        )
        disconnected_input = entrypoint.index(
            "room_transport_disconnected=lambda: room_disconnect.triggered", close_decision
        )
        disconnected_guard = entrypoint.index("if not session_close.required:", teardown)
        skipped = entrypoint.index('"session_close_skipped"', disconnected_guard)
        evidence_freeze = entrypoint.index("finalized = await finalize_off_loop(recorder)", skipped)
        self.assertLess(room_signal, teardown)
        self.assertLess(teardown, close_decision)
        self.assertLess(close_decision, disconnected_input)
        self.assertLess(disconnected_input, disconnected_guard)
        self.assertLess(disconnected_guard, skipped)
        self.assertLess(skipped, evidence_freeze)

    def test_production_entrypoint_registers_room_waker_before_waiting(self) -> None:
        source = self._production_agent_source()
        entrypoint = source[source.index("async def production_agent(") :]
        registration = entrypoint.index('ctx.room.on(\n        "disconnected"')
        blocking_wait = entrypoint.index("await finished.wait()")
        self.assertLess(registration, blocking_wait)
        self.assertIn("production_room_disconnect_handler(", entrypoint)
        self.assertIn("finish=finish_session", entrypoint)
        self.assertNotIn("ctx.add_shutdown_callback(", entrypoint)

    @staticmethod
    def _session_options_for_turn_taking() -> dict[str, object]:
        # This deliberately resembles the immutable stack built once per job:
        # the VAD object represents the prewarmed process-shared Silero VAD,
        # while nested turn maps are the SDK's per-session input.
        return {
            "vad": {"prewarmed": True, "min_silence_duration": 0.25},
            "turn_handling": {
                "turn_detection": "turn-detector",
                "endpointing": {"mode": "fixed", "min_delay": 0.2, "max_delay": 0.6},
                "interruption": {
                    "enabled": True,
                    "mode": "vad",
                    "min_duration": 0.15,
                    "min_words": 0,
                    "false_interruption_timeout": 1.0,
                    "resume_false_interruption": True,
                },
            },
        }

    def test_managed_english_pstn_defaults_require_half_second_of_speech(self) -> None:
        config = _config(language="en")
        config["direction"] = "outbound"
        source = self._session_options_for_turn_taking()

        options = session_options_with_managed_pstn_turn_taking(source, realtime_config=config)
        interruption = options["turn_handling"]["interruption"]

        self.assertEqual(
            interruption["min_duration"],
            MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS,
        )
        # Deterministic policy fixtures in place of copyrighted media: silence
        # and a short background/backchannel segment do not meet the threshold,
        # while a sustained foreground "stop" does and remains within the
        # 800 ms caller-interruption contract.
        self.assertLess(0.30, interruption["min_duration"])
        self.assertLess(0.25, interruption["min_duration"])
        self.assertLessEqual(interruption["min_duration"], 0.80)
        self.assertEqual(interruption["min_words"], 0)
        self.assertIs(interruption["resume_false_interruption"], True)

    def test_managed_english_pstn_turn_taking_maps_only_safe_session_controls(self) -> None:
        config = _config(language="en")
        config.update(
            {
                "direction": "inbound",
                "turn_taking": {
                    "interruption_sensitivity": "high",
                    "min_speech_ms": 640,
                    "silence_hold_ms": 450,
                    "barge_in_speech_rms": 400,
                },
            }
        )
        source = self._session_options_for_turn_taking()
        shared_vad = source["vad"]

        options = session_options_with_managed_pstn_turn_taking(source, realtime_config=config)

        # Explicit duration and silence controls deterministically override the
        # preset via real AgentSession options.  Audio-RMS gating has no safe
        # per-session LiveKit equivalent, so it is intentionally absent.
        self.assertEqual(options["turn_handling"]["interruption"]["min_duration"], 0.64)
        self.assertEqual(options["turn_handling"]["endpointing"]["min_delay"], 0.45)
        self.assertNotIn("barge_in_speech_rms", options["turn_handling"]["interruption"])
        self.assertIs(options["vad"], shared_vad)
        self.assertEqual(source["vad"], {"prewarmed": True, "min_silence_duration": 0.25})
        self.assertEqual(source["turn_handling"]["interruption"]["min_duration"], 0.15)
        self.assertEqual(source["turn_handling"]["endpointing"]["min_delay"], 0.2)

    def test_sensitivity_presets_are_per_session_and_do_not_contaminate_shared_vad(self) -> None:
        source = self._session_options_for_turn_taking()
        high = _config(language="en")
        high.update({"direction": "outbound", "turn_taking": {"interruption_sensitivity": "high"}})
        low = _config(language="en")
        low.update({"direction": "outbound", "turn_taking": {"interruption_sensitivity": "low"}})

        start = threading.Barrier(2)
        results: dict[str, dict[str, object]] = {}
        failures: list[BaseException] = []

        def build(name: str, config: dict) -> None:
            try:
                start.wait(timeout=5)
                results[name] = session_options_with_managed_pstn_turn_taking(
                    source, realtime_config=config
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)

        threads = [
            threading.Thread(target=build, args=("high", high)),
            threading.Thread(target=build, args=("low", low)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(failures)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        high_options = results["high"]
        low_options = results["low"]

        # A soft-spoken caller can explicitly choose high sensitivity; a low
        # sensitivity caller must sustain speech longer.  Neither setting
        # changes the shared VAD or leaks into the other session.
        self.assertEqual(high_options["turn_handling"]["interruption"]["min_duration"], 0.15)
        self.assertEqual(low_options["turn_handling"]["interruption"]["min_duration"], 0.75)
        self.assertEqual(low_options["turn_handling"]["endpointing"]["min_delay"], 0.9)
        self.assertEqual(low_options["turn_handling"]["endpointing"]["max_delay"], 1.8)
        self.assertIs(high_options["vad"], source["vad"])
        self.assertIs(low_options["vad"], source["vad"])
        self.assertEqual(source["turn_handling"]["interruption"]["min_duration"], 0.15)

    def test_invalid_turn_taking_falls_back_to_the_safe_managed_default(self) -> None:
        config = _config(language="en")
        config.update(
            {
                "direction": "outbound",
                "turn_taking": {
                    "interruption_sensitivity": "yolo",
                    "min_speech_ms": False,
                    "silence_hold_ms": float("inf"),
                    "barge_in_speech_rms": float("nan"),
                },
            }
        )
        options = session_options_with_managed_pstn_turn_taking(
            self._session_options_for_turn_taking(), realtime_config=config
        )

        self.assertEqual(
            options["turn_handling"]["interruption"]["min_duration"],
            MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS,
        )
        # Junk reads as balanced: the managed phone line's default detector hold.
        self.assertEqual(options["turn_handling"]["endpointing"]["min_delay"], 0.6)
        self.assertEqual(options["turn_handling"]["endpointing"]["max_delay"], 1.2)

    def test_browser_session_options_remain_unchanged(self) -> None:
        """Nothing without a customer call direction is touched at all.

        Browser tests, the standalone demo and the platform release canary
        keep the shared demo stack's own mapping, by identity.
        """

        source = self._session_options_for_turn_taking()
        config = _config(language="en")
        config.update(
            {
                "direction": "web_test",
                "turn_taking": {"interruption_sensitivity": "low", "min_speech_ms": 900},
            }
        )

        options = session_options_with_managed_pstn_turn_taking(source, realtime_config=config)

        self.assertIs(options["turn_handling"], source["turn_handling"])
        self.assertEqual(options["turn_handling"]["interruption"]["min_duration"], 0.15)

    def test_a_non_english_managed_call_takes_explicit_values_over_the_preset(self) -> None:
        """Explicit values win over the preset, and the preset applies too.

        Three decisions in sequence, and a reader restoring an earlier belief
        would undo a later one. First, non-English calls sat on the demo
        baseline. Then the 0.50 s default crossed into every language. Then the
        2026-09-08 decision made an explicit ``min_speech_ms``/``silence_hold_ms``
        a platform behaviour in every language, and the 2026-09-15 decision did the
        same for ``interruption_sensitivity`` once the Sinhala replays showed
        the floor is a property of the line and the VAD.

        This fixture is detector-shaped (``turn_detection`` is not ``"vad"``),
        so the preset's hold row is the model one: the row follows the built
        stack's mode, which ``demo.py`` derives from the language set, rather
        than a second reading of the language here.
        """

        for language, direction in (("et", "inbound"), ("si", "outbound"), ("ta", "outbound")):
            source = self._session_options_for_turn_taking()
            config = _config(language=language)
            config.update(
                {
                    "direction": direction,
                    "turn_taking": {"interruption_sensitivity": "low", "min_speech_ms": 900},
                }
            )

            options = session_options_with_managed_pstn_turn_taking(source, realtime_config=config)
            interruption = options["turn_handling"]["interruption"]

            # The explicit 900 ms wins over the "low" preset's own 0.75 s.
            self.assertEqual(interruption["min_duration"], 0.9, (language, direction))
            # No silence_hold_ms was set, so the preset's holds apply: the
            # model row's 0.9 s / 1.8 s on this detector-shaped stack.
            self.assertEqual(
                options["turn_handling"]["endpointing"],
                {"mode": "fixed", "min_delay": 0.9, "max_delay": 1.8},
                (language, direction),
            )
            self.assertEqual(source["turn_handling"]["endpointing"]["min_delay"], 0.2)
            # The caller's own mapping is never mutated in place.
            self.assertEqual(source["turn_handling"]["interruption"]["min_duration"], 0.15)

    def test_a_non_english_managed_call_with_no_tuning_gets_the_floor(self) -> None:
        """No per-agent request at all: the platform floor and the default phone-line hold."""

        for language, direction in (("et", "inbound"), ("si", "outbound"), ("ta", "outbound")):
            source = self._session_options_for_turn_taking()
            config = _config(language=language)
            config.update({"direction": direction})

            options = session_options_with_managed_pstn_turn_taking(source, realtime_config=config)

            self.assertEqual(
                options["turn_handling"]["interruption"]["min_duration"],
                MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS,
                (language, direction),
            )
            # Detector-shaped fixture: the balanced detector row applies.
            self.assertEqual(
                options["turn_handling"]["endpointing"],
                {**source["turn_handling"]["endpointing"], "min_delay": 0.6, "max_delay": 1.2},
                (language, direction),
            )

    @_worker_source_guard
    def test_worker_applies_the_call_bound_session_policy_before_constructing_agent_session(self) -> None:
        source = self._production_agent_source()
        session_setup = source[
            source.index("session_options = session_options_with_managed_pstn_turn_taking("):
            source.index("session = AgentSession(**session_options)")
        ]
        self.assertIn("realtime_config=call.config", session_setup)
        # Browser-specific connect-error behaviour remains an additive option
        # on the copied session mapping, after the policy seam has selected
        # managed PSTN versus browser-test behaviour.
        self.assertIn("if browser_test is not None:", session_setup)

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

    def test_browser_participant_mismatch_has_a_safe_distinct_lifecycle_reason(self) -> None:
        self.assertEqual(
            pre_media_lifecycle_reason(
                "adapter_configuration", participant_mismatch=True
            ),
            "participant_mismatch_rejected",
        )

    def test_browser_test_error_metadata_keeps_the_reservation_document(self) -> None:
        """The route writes this document first; the worker adds to it, never over it."""
        existing = json.dumps(
            {"kind": "workspace_browser_test", "agent_id": "agent_1"}, sort_keys=True
        )

        merged = json.loads(browser_test_error_metadata(existing, component="tts"))

        self.assertEqual(merged["kind"], "workspace_browser_test")
        self.assertEqual(merged["agent_id"], "agent_1")
        self.assertEqual(merged["session_error"], {"component": "tts"})

    def test_browser_test_error_metadata_publishes_no_provider_prose(self) -> None:
        """This document is readable by the customer's browser.

        LiveKit's own error text carries the full request URL -- voice id, model,
        output format, every query parameter -- and a different provider could
        put an account identifier there just as easily. Nothing outside the
        closed component set may cross into a room a customer is sitting in.
        """
        hostile = (
            "message='Invalid response status', status_code=403, url='wss://"
            "api.elevenlabs.io/v1/text-to-speech/Xb7hH8MSUJpSbSDYk0k2/multi-stream-input"
            "?model_id=eleven_v3&xi-api-key=sk-secret'"
        )

        published = browser_test_error_metadata("{}", component=hostile)

        self.assertNotIn("elevenlabs", published)
        self.assertNotIn("sk-secret", published)
        self.assertNotIn("Xb7hH8MSUJpSbSDYk0k2", published)
        self.assertEqual(json.loads(published)["session_error"], {"component": "session"})

    def test_browser_test_error_metadata_survives_a_document_it_cannot_parse(self) -> None:
        """Unreadable metadata must not cost the browser its only failure signal."""
        for existing in ("", "not json", "[]", "null"):
            with self.subTest(existing=existing):
                merged = json.loads(browser_test_error_metadata(existing, component="tts"))
                self.assertEqual(merged["session_error"], {"component": "tts"})

    def test_an_ordinary_ending_is_not_reported_as_a_failure(self) -> None:
        """Most closes are somebody hanging up. Only ``error`` is LiveKit giving up."""
        for reason in ("user_initiated", "participant_disconnected", "job_shutdown", "task_completed"):
            with self.subTest(reason=reason):
                self.assertIsNone(browser_test_close_component(reason=reason, error=None))

    def test_a_failed_turn_is_not_a_failed_session(self) -> None:
        """The signal is the CLOSE, and a component error is not one.

        ``AgentSession`` absorbs ``max_unrecoverable_errors`` consecutive
        unrecoverable failures per component -- three by default -- and resets
        that count on the next turn that works. So ``recoverable=False`` means a
        turn failed, not that the pipeline stopped. Reporting one would end
        sessions LiveKit was about to recover: the same silent-failure bug with
        the sign flipped, and the more irritating half, because it fires on calls
        that were fine.
        """
        failed_turn = SimpleNamespace(type="tts_error", recoverable=False)
        self.assertIsNone(browser_test_close_component(reason=None, error=failed_turn))

    def test_a_close_on_error_names_the_component_that_stopped(self) -> None:
        for error_type, component in (
            ("tts_error", "tts"),
            ("llm_error", "llm"),
            ("stt_error", "stt"),
            ("realtime_model_error", "realtime"),
        ):
            with self.subTest(error_type=error_type):
                error = SimpleNamespace(type=error_type, recoverable=False)
                self.assertEqual(
                    browser_test_close_component(reason="error", error=error), component
                )

    def test_the_reason_is_read_through_a_str_enum_as_livekit_sends_it(self) -> None:
        """``CloseReason`` is a ``str`` Enum, and ``str()`` on one is its NAME.

        Comparing the enum member itself against "error" without going through
        ``.value`` yields "CloseReason.ERROR" and matches nothing, so every real
        failure would report as an ordinary hangup -- silently, which is the one
        way this must not break.
        """

        class _CloseReason(str, Enum):
            ERROR = "error"
            USER_INITIATED = "user_initiated"

        error = SimpleNamespace(type="tts_error", recoverable=False)
        self.assertEqual(browser_test_close_component(reason=_CloseReason.ERROR, error=error), "tts")
        self.assertIsNone(
            browser_test_close_component(reason=_CloseReason.USER_INITIATED, error=error)
        )

    def test_an_unrecognised_failure_shape_is_reported_rather_than_swallowed(self) -> None:
        """Silence is the bug. A close on error with no readable error is still a failure.

        A future LiveKit release, or a plugin raising its own model, can close
        carrying something with no ``type``. Defaulting that to "say nothing"
        would rebuild the exact silence this path exists to end.
        """
        self.assertEqual(browser_test_close_component(reason="error", error=None), "session")
        self.assertEqual(
            browser_test_close_component(reason="error", error=RuntimeError("boom")), "session"
        )

    def test_component_error_diagnostics_are_type_only_and_bounded(self) -> None:
        secret = "https://speech.googleapis.com?key=sk-must-not-log"
        deepest = RuntimeError(secret)
        current: BaseException = deepest
        for index in range(10):
            wrapped = ValueError(f"provider detail {index}: {secret}")
            wrapped.__cause__ = current
            current = wrapped
        outer = SimpleNamespace(error=current)

        diagnostics = safe_component_error_diagnostics(outer)

        self.assertEqual(diagnostics["error_type"], "SimpleNamespace")
        self.assertEqual(diagnostics["provider_error_type"], "ValueError")
        self.assertEqual(len(diagnostics["provider_cause_types"]), 6)
        self.assertTrue(
            all(
                name in {"ValueError", "RuntimeError"}
                for name in diagnostics["provider_cause_types"]
            )
        )
        self.assertNotIn("speech.googleapis.com", json.dumps(diagnostics))
        self.assertNotIn("sk-must-not-log", json.dumps(diagnostics))

    def test_component_error_diagnostics_tolerate_livekit_shape_without_provider_error(
        self,
    ) -> None:
        self.assertEqual(
            safe_component_error_diagnostics(SimpleNamespace(type="stt_error")),
            {"error_type": "SimpleNamespace"},
        )

    @staticmethod
    def _production_agent_source() -> str:
        return _PRODUCTION_AGENT.read_text()

    @_worker_source_guard
    def test_preplayout_text_evidence_wraps_both_normal_and_workflow_agents(self) -> None:
        """Pin the LiveKit-only glue to the source when the SDK is absent.

        The recorder itself is unit-tested independently.  This guard catches
        the more dangerous regression where a future worker refactor leaves
        the durable pre-playout path intact but starts one of the two agent
        modes with the bare SDK ``Agent`` again.
        """

        source = self._production_agent_source()
        tree = ast.parse(source)
        classes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        evidence = classes["EvidenceAgent"]
        self.assertEqual(
            [base.id for base in evidence.bases if isinstance(base, ast.Name)],
            ["Agent"],
        )
        tts_node = next(
            node
            for node in evidence.body
            if isinstance(node, ast.FunctionDef) and node.name == "tts_node"
        )
        self.assertTrue(
            any(_calls_named(statement, "admitted_tts_text") for statement in tts_node.body),
            "TTS text must be bound before the SDK starts synthesis",
        )
        self.assertTrue(
            any(
                _calls_named(statement, "complete_preplayout_agent_speech")
                for statement in tts_node.body
            ),
            "handle completion must retain interruption metadata",
        )
        self.assertIn("admits_conversation_item", ast.unparse(tts_node))
        # A voice-to-voice engine has no TTS node: its own output transcript
        # is bound at the transcription node, under the speech's id, and only
        # when the agent was built for that engine.
        transcription_node = next(
            node
            for node in evidence.body
            if isinstance(node, ast.FunctionDef) and node.name == "transcription_node"
        )
        self.assertTrue(
            any(
                _calls_named(statement, "bound_output_transcript")
                for statement in transcription_node.body
            ),
            "the engine's output transcript must be bound under its speech id",
        )
        self.assertIn("if not self._speech_to_speech:", ast.unparse(transcription_node))
        self.assertIn('evidence_agent_kwargs["speech_to_speech"] = True', source)
        self.assertNotIn("WorkflowTurnAgent", classes)

        construction = next(node.value for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "session_agent")
        self.assertEqual(ast.unparse(construction.func), "EvidenceAgent")
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in construction.keywords}
        self.assertEqual(keywords["recorder"], "recorder")
        self.assertEqual(keywords["admission"], "semantic_admission")
        # And that same object is the one the session actually starts.
        self.assertIn("agent=session_agent,", source)

    def test_admitted_tts_text_binds_before_yield_and_stops_after_admission_closes(
        self,
    ) -> None:
        class Recorder:
            def __init__(self) -> None:
                self.bound: list[tuple[str, str]] = []

            def bind_preplayout_agent_text(self, speech_id: str, chunk: str) -> None:
                self.bound.append((speech_id, chunk))

        async def run() -> tuple[list[str], list[tuple[str, str]]]:
            admission = SemanticTurnAdmission()
            recorder = Recorder()

            async def source():
                yield "before close"
                admission.close()
                yield "after close"

            delivered: list[str] = []
            async for chunk in admitted_tts_text(
                source(),
                speech_id="speech_1",
                recorder=recorder,
                admission=admission,
            ):
                # Binding precedes the synthesizer's receipt of this chunk.
                if chunk == "before close":
                    self.assertEqual(recorder.bound, [("speech_1", chunk)])
                delivered.append(chunk)
            return delivered, recorder.bound

        delivered, bound = asyncio.run(run())

        self.assertEqual(delivered, ["before close", "after close"])
        self.assertEqual(bound, [("speech_1", "before close")])

    @_worker_source_guard
    def test_the_browser_failure_report_is_wired_to_the_session_and_to_the_pre_media_guard(
        self,
    ) -> None:
        """The predicate and the metadata are unit-tested; the glue that calls them is not.

        ``production_agent.py`` cannot be imported here -- no livekit package
        exists anywhere in this repo -- so its wiring is asserted against the
        source, the same way ``test_workflow_talk_worker`` asserts the turn glue.
        Weaker than executing it, and named as such. What it catches is the whole
        reporting path being deleted or rerouted while every unit test around it
        stays green, which is exactly the shape a reviewer found here.
        """
        source = self._production_agent_source()

        # The session seam: keyed on the close, reading the reason AND the error,
        # and handing the component to the one reporter.
        self.assertIn('@session.on("close")', source)
        close = source[source.index('@session.on("close")') :]
        close = close[: close.index("\n    ctx.room.on(")]
        self.assertIn("browser_test_close_component(", close)
        self.assertIn("reason=getattr(event, \"reason\", None)", close)
        self.assertIn("report_browser_test_failure(component)", close)
        # Phone calls have no dock listening and must not pay for this at all.
        self.assertIn("if browser_test is None:", close)
        # The same close ends a PHONE call whose engine gave up, through the
        # arbiter, inside the phone-call branch and before its return: the
        # browser test keeps its report-and-deadline contract untouched.
        phone_branch = close[close.index("if browser_test is None:") :]
        phone_branch = phone_branch[: phone_branch.index("component = browser_test_close_component(")]
        self.assertIn("session_closed(event)", phone_branch)
        self.assertLess(phone_branch.index("session_closed(event)"), phone_branch.index("return"))
        self.assertNotIn("session_closed(event)", close.replace(phone_branch, ""))
        wiring = source[: source.index('@session.on("close")')]
        wiring = wiring[wiring.rindex("production_session_close_handler(") :]
        self.assertIn("TERMINATION_MODEL_SESSION_LOST", wiring)
        self.assertIn("DETECTED_VIA_SESSION_CLOSED", wiring)
        self.assertIn("is_finishing=finished.is_set", wiring)

        # The pre-media seam: a stack that will not build never reaches the
        # session, so the guard reports before the teardown deletes the room.
        from livekit_agent import production as shared_production
        import inspect
        guard = inspect.getsource(shared_production.settle_pre_media_failure)
        self.assertIn('report_browser_test_failure("session")', guard)

        # Structure, not adjacency. Asserting that a `finally:` appears after the
        # report only proves ordering, and a dummy `try: pass finally: pass`
        # before the teardown satisfies that while restoring the exact strand
        # this closed. The teardown must be IN the finally of the try that runs
        # the report, which only the tree can say.
        tree = ast.parse(source)
        # Matched on the CALL, not a mention: a decoy `try: report_browser_test_failure`
        # (a bare name) beside a real `finally: terminate_pre_media_call()` would
        # satisfy a mention-based check while the real report sat in a dummy
        # try/finally elsewhere. And it must be the SAME Try, not any two.
        reporting_tries = [
            node
            for node in ast.walk(ast.parse(guard))
            if isinstance(node, ast.Try)
            and any(_calls_named(stmt, "report_browser_test_failure") for stmt in node.body)
        ]
        self.assertEqual(len(reporting_tries), 1, "one try may run the pre-media report")
        self.assertTrue(
            any(
                _calls_named(stmt, "terminate_pre_media_call")
                for stmt in reporting_tries[0].finalbody
            ),
            "terminate_pre_media_call must run in the finally of that same try",
        )

        # And the teardown waits for the write, because the room it lives in is
        # about to be deleted.
        waits = [
            call
            for call in _calls_named(tree, "await_bounded_teardown_task")
            if _mentions(call, "browser_test_report_task")
        ]
        self.assertEqual(len(waits), 1, "exactly one bounded teardown wait on the report task")
        wait = waits[0]
        # Placed between the close that produces the report and the delete that
        # destroys the room it lives in. That ordering is the entire reason this
        # wait exists, so it is pinned by line rather than left to the prose.
        closes = [
            call
            for call in _calls_named(tree, "close_production_session_after_room_end")
            if _mentions(call, "session")
        ]
        self.assertTrue(closes, "the session close must precede the wait")
        self.assertLess(min(c.lineno for c in closes), wait.lineno)
        # EVERY shape a room delete takes, not just the wrapper. A reviewer put
        # `await cleanup_livekit_room(call, ctx.delete_room)` and `await
        # ctx.delete_room()` before the wait and an earlier version of this
        # assertion passed both, while either one destroys the room the report is
        # still writing to -- the silent failure this whole branch exists to end.
        #
        # Scoped to the entrypoint, not the module: line order only stands in for
        # run order inside one function body, and an unrelated helper elsewhere
        # that happens to delete a room is not this contract's business.
        entrypoint = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_production_agent_job"
        )
        # Nested delete helpers are excluded by what they DO, not by their name:
        # they wrap the delete and are defined before the wait, and pinning the
        # name would break on a rename that changes no behaviour.
        wrappers = [
            node
            for node in ast.walk(entrypoint)
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node is not entrypoint
            and _calls_named(node, "cleanup_livekit_room")
        ]
        inside_wrapper = {id(node) for wrapper in wrappers for node in ast.walk(wrapper)}
        # Known limit, and the reason this is a guard rather than a proof: it
        # matches call SHAPES, so `delete = ctx.delete_room; await delete()`
        # slips past. Closing that needs the teardown extracted into something
        # importable and driven for real, which is a refactor of the worker
        # entrypoint rather than part of this fix.
        deletes = [
            call
            for name in ("cleanup_room", "cleanup_livekit_room", "delete_room")
            for call in _calls_named(entrypoint, name)
            if id(call) not in inside_wrapper
        ]
        self.assertTrue(deletes, "the room delete must follow the wait")
        early = [call.lineno for call in deletes if call.lineno < wait.lineno]
        self.assertFalse(
            early, f"no room delete may run before the report wait; found at {early}"
        )
        # The SAME constant the reporter budgets against, as the timeout ARGUMENT.
        # A literal reads identically and lets the two clocks drift apart, which
        # is the whole reason that constant exists.
        timeouts = [kw.value for kw in wait.keywords if kw.arg == "timeout_seconds"]
        self.assertEqual(len(timeouts), 1, "the wait must carry an explicit timeout")
        self.assertIsInstance(timeouts[0], ast.Name)
        self.assertEqual(timeouts[0].id, "BROWSER_TEST_REPORT_BUDGET_SECONDS")
        # Unshielded, so owner cancellation reaches the bounded helper and it
        # can cancel/observe the reporter before the caller defers cancellation.
        self.assertFalse(
            _calls_named(wait, "shield"), "the awaited task must not be shielded"
        )
        # Nor the wait itself. `await asyncio.shield(asyncio.wait_for(...))` is the
        # realistic "make cancellation safer" move, and it puts the job's
        # CancelledError on the shield rather than in the handler below -- so the
        # deferral never records and the wait can outlive shutdown.
        self.assertFalse(
            [
                call
                for call in _calls_named(tree, "shield")
                if any(inner is wait for inner in _calls_named(call, "wait_for"))
            ],
            "the wait must not be wrapped in a shield either",
        )

        # The handlers around that wait, pinned by shape. This one DEFERS
        # cancellation rather than choosing a side. Re-raising here would skip
        # the sidecar, capture tasks, billing and the lifecycle reports, leaving
        # only the outer handler's room delete; swallowing it outright would let
        # a cancelled job return as though it had succeeded. So it records the
        # cancellation, the teardown finishes, and the raise happens at the end.
        # A bare `except BaseException` would also swallow KeyboardInterrupt, so
        # the narrow handler is the contract.
        # The try wrapping the WAIT, not any enclosing one: the outer teardown
        # try also contains this task by name, and matching it would read the
        # wrong handlers entirely.
        def _awaits_directly(stmt: ast.stmt, call: ast.Call) -> bool:
            value = (
                stmt.value
                if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign))
                else None
            )
            return isinstance(value, ast.Await) and value.value is call

        teardown_tries = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(_awaits_directly(stmt, wait) for stmt in node.body)
        ]
        self.assertEqual(len(teardown_tries), 1, "the wait must be guarded by exactly one try")
        teardown_try = teardown_tries[0]
        caught = [
            handler.type for handler in teardown_try.handlers if handler.type is not None
        ]
        self.assertTrue(
            any(_mentions(exc, "CancelledError") for exc in caught),
            "the teardown wait must handle cancellation rather than propagate it",
        )
        cancel_handler = next(
            handler
            for handler in teardown_try.handlers
            if handler.type is not None and _mentions(handler.type, "CancelledError")
        )
        # Nothing anywhere under it may leave early. `if True: raise` and a bare
        # `return` both restore the skip this closed, and both passed a top-level
        # `isinstance` check.
        self.assertFalse(
            any(
                isinstance(node, (ast.Raise, ast.Return))
                for stmt in cancel_handler.body
                for node in ast.walk(stmt)
            ),
            "this handler must not leave the teardown early, at any depth",
        )
        # It defers instead: records the cancellation, and the teardown re-raises
        # it once the cleanup below has finished. Swallowing it outright would
        # let a cancelled job return as though it had succeeded.
        self.assertTrue(
            any(
                isinstance(stmt, ast.Assign)
                and any(
                    _mentions(t, "browser_test_report_cancellation") for t in stmt.targets
                )
                for stmt in cancel_handler.body
            ),
            "the cancellation must be recorded for the deferred re-raise",
        )
        deferred = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Name)
            and node.exc.id == "browser_test_report_cancellation"
        ]
        self.assertEqual(len(deferred), 1, "the deferred cancellation must be re-raised once")
        # AFTER the teardown, which is the entire point of deferring it. Moving
        # this above the lifecycle reports, the artifact finalization or the room
        # cleanup reinstates the abandonment it was written to prevent, and
        # "a raise exists somewhere" cannot tell the difference.
        finalizers = _calls_named(tree, "finalize_call_artifacts") + _calls_named(
            tree, "cleanup_room"
        )
        self.assertTrue(finalizers, "expected the teardown calls this must follow")
        self.assertGreater(
            deferred[0].lineno,
            max(call.lineno for call in finalizers),
            "the deferred cancellation must be re-raised after the teardown finishes",
        )
        self.assertFalse(
            any(_mentions(exc, "BaseException") for exc in caught),
            "BaseException here would swallow KeyboardInterrupt too",
        )

    @_worker_source_guard
    def test_a_browser_test_closes_on_the_first_unrecoverable_error(self) -> None:
        """Reporting on the close is only prompt if the close is prompt.

        LiveKit absorbs three unrecoverable errors per component before giving
        up. On a phone call that is right -- a caller is on the line and three
        more tries beat hanging up on them. On a browser test nobody is on the
        line and somebody is watching a page, so three silent turns IS the
        connected-but-mute session. The budget is set to zero for browser tests
        only; the phone path must keep LiveKit's default.
        """
        source = self._production_agent_source()

        self.assertIn("from livekit.agents.voice.agent_session import SessionConnectOptions", source)
        tree = ast.parse(source)

        # INSIDE the browser-test branch, and nowhere else. Ordering alone is not
        # enough: `if browser_test is not None: pass` followed by a hoisted
        # assignment reads almost identically and would put every phone call and
        # the release canary on a zero error budget. Containment is the claim, so
        # containment is what gets asserted.
        options = _calls_named(tree, "SessionConnectOptions")
        self.assertEqual(len(options), 1, "exactly one place may set the session budget")
        budget = options[0]
        self.assertEqual([kw.arg for kw in budget.keywords], ["max_unrecoverable_errors"])
        zero = budget.keywords[0].value
        self.assertIsInstance(zero, ast.Constant)
        # `is 0`, not `== 0`: False compares equal to zero and would read as a
        # deliberate budget while being a different value entirely.
        self.assertIs(type(zero.value), int)
        self.assertEqual(zero.value, 0)

        # POLARITY, exactly. `if browser_test is None:` reads almost the same and
        # puts every phone call and the release canary on a zero error budget; a
        # reviewer applied that mutation and an earlier version of this test
        # stayed green.
        def _is_browser_test_guard(test: ast.expr) -> bool:
            return (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "browser_test"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.IsNot)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            )

        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and _is_browser_test_guard(node.test)
            and any(_calls_named(stmt, "SessionConnectOptions") for stmt in node.body)
        ]
        self.assertEqual(
            len(guards), 1, "the zero budget must live under `browser_test is not None`"
        )
        # And not in that guard's `else`, which would invert it onto phone calls.
        self.assertFalse(
            any(_calls_named(stmt, "SessionConnectOptions") for stmt in guards[0].orelse)
        )
        # USED, not merely constructed. A discarded call in a correct branch
        # leaves browser tests on LiveKit's default of three.
        def _sets_conn_options(target: ast.expr) -> bool:
            # The exact key, not merely a subscript of the right dict.
            # `session_options["wrong_key"] = ...` leaves LiveKit on its default
            # of three and reads as done.
            return (
                isinstance(target, ast.Subscript)
                and _mentions(target.value, "session_options")
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "conn_options"
            )

        stored = [
            stmt
            for stmt in guards[0].body
            if isinstance(stmt, ast.Assign)
            and _calls_named(stmt.value, "SessionConnectOptions")
            and any(_sets_conn_options(target) for target in stmt.targets)
        ]
        self.assertEqual(
            len(stored), 1, "the budget must be stored as session_options['conn_options']"
        )

        # A copy, so the shared stack the phone path also receives is not mutated.
        self.assertIn("dict(stack.session_options)", source)

    @_worker_source_guard
    def test_the_report_retries_rather_than_counting_attempts_it_never_makes(self) -> None:
        """Both callers report once, so a per-call counter is a budget nothing draws on.

        The first version incremented an attempt counter and wrote once. After a
        transient timeout, attempts two and three were unreachable and the
        browser stayed silent -- the failure this path exists to report, hidden
        by the code reporting it.
        """
        source = self._production_agent_source()

        body = source[source.index("    async def report_browser_test_failure(") :]
        body = body[: body.index("\n    try:")]
        self.assertIn("for attempt in range(3):", body)
        # ONE deadline for the whole thing. Per-attempt timeouts multiply, and
        # three five-second attempts outlast the teardown's five-second wait --
        # so the retries would still be writing to a room already being deleted.
        self.assertIn("BROWSER_TEST_REPORT_BUDGET_SECONDS", body)
        self.assertIn("remaining = deadline - time.monotonic()", body)
        self.assertIn("timeout=remaining,", body)
        self.assertNotIn("except BaseException", body)

        # Cancellation is a caller withdrawing the budget, so it propagates. The
        # claim is that the handler RE-RAISES, and only the tree can say that:
        # searching the handler text for "raise" is satisfied by a comment, by an
        # identifier like `reraise`, or by a `return` sitting beside either.
        reporter = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "report_browser_test_failure"
        )
        cancel_handlers = [
            handler
            for node in ast.walk(reporter)
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if handler.type is not None and _mentions(handler.type, "CancelledError")
        ]
        self.assertEqual(len(cancel_handlers), 1, "one explicit cancellation handler")
        self.assertTrue(
            all(
                isinstance(stmt, ast.Raise) and stmt.exc is None
                for stmt in cancel_handlers[0].body
            ),
            "the handler must do nothing but re-raise, bare -- `raise RuntimeError(...)` "
            "is a Raise too and stops being cancellation",
        )
        # The loop must exit on the FIRST success, not keep rewriting.
        self.assertIn("browser_test_failure_reported = True", body)
        self.assertLess(
            body.index("browser_test_failure_reported = True"),
            body.index("finally:"),
            "success must be recorded inside the loop, before the guard is released",
        )
        self.assertIn("return", body[body.index("browser_test_failure_reported = True") :])

    @_worker_source_guard
    def test_a_duplicate_browser_worker_exits_without_failing_or_deleting_the_owned_room(
        self,
    ) -> None:
        tree = ast.parse(self._production_agent_source())
        handlers = [
            handler
            for handler in ast.walk(tree)
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "BrowserSessionAlreadyOwned"
        ]
        self.assertEqual(len(handlers), 1)
        handler = handlers[0]
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(handler)))
        self.assertFalse(_calls_named(handler, "terminate_pre_media_call"))
        self.assertFalse(_calls_named(handler, "report_browser_test_failure"))

    def test_pre_media_teardown_accepts_the_tracked_task_livekit_actually_returns(self) -> None:
        """``ctx.delete_room`` is passed straight in, and it does not return a coroutine.

        LiveKit registers the room delete with its own pending-task tracker and
        hands back that Task. The test above supplies plain ``async def``
        factories, which is not the shape production uses, so it kept passing
        while every real pre-media rejection died inside the teardown with
        ``TypeError: a coroutine was expected``. The call then kept its ringing
        SIP leg and its wallet reservation, and reported no failure at all.
        """

        async def scenario() -> tuple[dict[str, object], list[str]]:
            events: list[str] = []

            class _Adapter:
                def report_failed_before_media(self, **_kwargs):
                    events.append("sip_failed")
                    return {"canonical_type": "call.failed", "billing": {"status": "released"}}

            class _Call:
                def cleanup(self, **_kwargs):
                    events.append("call_cleaned")

            async def _delete() -> None:
                events.append("room_deleted")

            def delete_room():
                # Exactly what JobContext.delete_room does: schedule it, track
                # the task, return the task.
                return asyncio.ensure_future(_delete())

            async def remove_participant() -> None:
                events.append("participant_removed")

            result = await terminate_pre_media_call(
                call=_Call(),
                adapter=_Adapter(),
                call_id="call_1",
                reason="component configuration rejected",
                remove_participant=remove_participant,
                delete_room=delete_room,
            )
            return result, events

        result, events = asyncio.run(scenario())

        self.assertIn("room_deleted", events)
        self.assertIn("participant_removed", events)
        self.assertIn("sip_failed", events)
        self.assertTrue(result["room_terminated"])
        self.assertTrue(result["participant_terminated"])
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
                "SKYSAY_WORKER_ID": "livekit-prod",
                "SKYSAY_WORKER_API_TOKEN": "must-not-leak",
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
                self.assertFalse(worker_heartbeats_enabled({"SKYSAY_WORKER_HEARTBEAT_ENABLED": value}))

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
                "SKYSAY_WORKER_ID": "livekit-prod",
                "SKYSAY_VOICE_IMAGE_REVISION": "b" * 40,
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
        greeting = "Tere! Olete helistanud Skysayile. Kuidas saan aidata?"
        self.assertTrue(
            is_greeting_fragment("Tere! Olete helistanud Skysayile. Kuidas saan", greeting)
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

    def test_effective_stack_marks_ambience_as_configured_not_observed(self) -> None:
        config = _config(language="en")
        config["audio_environment"] = {
            "preset_id": "cafe",
            "asset_version": "v1",
            "volume": 35,
            "runtime_gain_linear": 0.07,
            "runtime_gain_db": -23.098,
            "channel": "outgoing_background_track",
        }

        identity = effective_voice_stack_identity(config)

        self.assertEqual(identity["configuration"]["audio_environment"], {
            "preset_id": "cafe",
            "asset_version": "v1",
            "volume": 35,
            "channel": "outgoing_background_track",
            "status": "configured",
        })

    def test_effective_stack_never_calls_an_unshaped_ambience_mapping_configured(self) -> None:
        config = _config(language="en")
        config["audio_environment"] = {"preset_id": "cafe", "volume": 35}

        identity = effective_voice_stack_identity(config)

        self.assertEqual(identity["configuration"]["audio_environment"]["status"], "invalid")

    def test_effective_stack_does_not_trust_unrecognized_ambience_status(self) -> None:
        config = _config(language="en")
        config["audio_environment"] = {
            "preset_id": "cafe",
            "asset_version": "v1",
            "volume": 35,
            "runtime_gain_linear": 0.07,
            "runtime_gain_db": -23.098,
            "channel": "outgoing_background_track",
            # The worker rejects this unsupported key, so it cannot turn
            # configuration evidence into a false runtime observation.
            "status": "observed",
        }

        identity = effective_voice_stack_identity(config)

        self.assertEqual(identity["configuration"]["audio_environment"]["status"], "invalid")

    def test_evidence_excludes_only_a_noncaller_ambience_track(self) -> None:
        self.assertFalse(
            is_noncaller_ambience_track(
                publication_name="background_audio",
                participant_identity="caller-1",
                caller_identity="caller-1",
                agent_identity="agent-1",
                ambience_track_name="background_audio",
            )
        )
        self.assertTrue(
            is_noncaller_ambience_track(
                publication_name="background_audio",
                participant_identity="agent-1",
                caller_identity="caller-1",
                agent_identity="agent-1",
                ambience_track_name="background_audio",
            )
        )
        self.assertFalse(
            is_noncaller_ambience_track(
                publication_name="semantic_response",
                participant_identity="agent-1",
                caller_identity="caller-1",
                agent_identity="agent-1",
                ambience_track_name="background_audio",
            )
        )
        self.assertFalse(
            is_noncaller_ambience_track(
                publication_name="background_audio",
                participant_identity="bridge-1",
                caller_identity="caller-1",
                agent_identity="agent-1",
                ambience_track_name="background_audio",
            )
        )
        self.assertFalse(
            is_noncaller_ambience_track(
                publication_name="background_audio",
                participant_identity="agent-1",
                caller_identity="caller-1",
                agent_identity="",
                ambience_track_name="background_audio",
            )
        )

    def test_model_labels_come_from_the_resolved_profile_not_the_raw_config(self) -> None:
        # Telemetry must record what this worker BUILT. Reading the raw config
        # would record what it was asked for, which is a different claim.
        config = _config()
        labels = component_model_labels(profile_from_realtime_config(config))

        self.assertEqual(labels, {"stt": "nova-3", "llm": "gemini-2.5-flash-lite", "tts": "eleven_v3"})

    def test_sinhala_model_labels_follow_the_registry_that_actually_runs(self) -> None:
        # si-LK routes STT to Google streaming STT and TTS to Gemini-TTS. The labels
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

    def test_sinhala_realtime_config_requires_a_declared_stt_and_tts(self) -> None:
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

        self.assertEqual(profile.stt_model, "chirp_2")

        # Google Speech V1 ``command_and_search`` is gone from the live Google
        # allowlist. A leftover stored identity is refused before media, not
        # remapped onto Chirp 2.
        leftover_v1 = dict(config)
        leftover_v1["transcriber"] = {
            "registry_name": "google",
            "model": "command_and_search",
            "language": "si",
        }
        with self.assertRaisesRegex(ProductionConfigurationError, "supported STT provider/model"):
            profile_from_realtime_config(leftover_v1)

        # A third Google model, with no evidence behind it, is still refused
        # before media.
        unqualified = dict(config)
        unqualified["transcriber"] = {
            "registry_name": "google",
            "model": "latest_long",
            "language": "si",
        }
        with self.assertRaisesRegex(ProductionConfigurationError, "supported STT provider/model"):
            profile_from_realtime_config(unqualified)

        # Deepgram cannot serve si-LK STT at all.
        deepgram_stt = dict(config)
        deepgram_stt["transcriber"] = {"registry_name": "deepgram", "model": "nova-3", "language": "si"}
        with self.assertRaisesRegex(ProductionConfigurationError, "supported STT provider/model"):
            profile_from_realtime_config(deepgram_stt)

        # ElevenLabs cannot serve si-LK TTS at all, and the refusal no longer
        # comes from a rule that names Sinhala. The named Sinhala allow-lists
        # were removed once an enumeration showed they flipped no verdict; the
        # declaration refuses this the same way it refuses every other
        # language a provider does not declare, which is the whole point of
        # removing them.
        elevenlabs_tts = dict(config)
        elevenlabs_tts["voice"] = {
            "registry_name": "elevenlabs",
            "model": "eleven_v3",
            "voice_id": "voice_1",
            "language": "si",
        }
        with self.assertRaisesRegex(
            ProductionConfigurationError, "supported TTS provider/model"
        ):
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
        self.assertEqual(json.loads(output)["result"]["structuredContent"]["calls"], [])
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
                record_final_message=lambda message, *, interrupted=False, speech_id="", scheduled_at=None: terminal_messages.append(message),
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
        self.assertEqual(session.say_calls, [("Head aega.", True)])
        self.assertEqual(session.shutdown_calls, [False])
        self.assertEqual(terminal_messages, ["Head aega."])
        self.assertEqual(
            events,
            ["end_call_proposed", "say", "playout", "ending", "shutdown", "finish", "end_call"],
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

        self.assertEqual(session.say_calls, [("Goodbye.", True)])
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

        self.assertEqual(session.say_calls, [("Goodbye.", True)])
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
                record_final_message=lambda message, *, interrupted=False, speech_id="", scheduled_at=None: terminal_messages.append(message),
            )
            await action({})
            completion = await action.wait_for_completion()
            assert completion is not None
            return session, terminal_messages, completion

        session, terminal_messages, completion = asyncio.run(scenario())

        self.assertEqual(
            session.say_calls, [("Aitäh helistamast. Head aega.", True)]
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
                record_final_message=lambda message, *, interrupted=False, speech_id="", scheduled_at=None: terminal_messages.append(message),
            )
            await action({"message": "Goodbye."})
            observe_transcript("during proposed close")
            observe_conversation_item("during proposed close")
            await action.wait_for_completion()
            observe_transcript("must not be recorded")
            observe_conversation_item("must not be recorded")
            return transcripts, conversation_items, terminal_messages

        transcripts, conversation_items, terminal_messages = asyncio.run(scenario())

        self.assertEqual(transcripts, ["before ending", "during proposed close"])
        self.assertEqual(conversation_items, ["before ending", "during proposed close"])
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
            ["say", "playout", "ending", "shutdown", "finish", "control_plane_cleanup", "room_deleted"],
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

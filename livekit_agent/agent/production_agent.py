# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    cli,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import RoomOptions
from livekit.agents.metrics import ModelUsageCollector

from control_plane import (
    Gate2ControlPlaneAdapter,
    UrllibJsonTransport,
    call_credential_token,
    parse_outbound_dispatch,
    parse_workspace_browser_test_dispatch,
)
from workflow_turns import (
    OpeningAdvanceFailed,
    WorkflowTurnCoordinator,
    await_playout_then_finish,
    build_turn_outcome,
    latest_caller_text,
    pinned_workflow_run_id,
    workflow_turn_error_message,
)
from component_billing import (
    MeteringContractError,
    SuccessfulToolUsageCounter,
    normalized_usage_from_model_usage,
)
from component_credentials import (
    has_byok_component,
    runtime_environ_with_component_credentials,
)
from demo import build_demo_stack
from prewarm import prewarm_plugin_modules, resolve_idle_process_count
from livekit_evidence import NoRoomAudioCaptured, SynchronizedEvidenceRecorder
from call_telemetry import CallTelemetryCollector
from evidence_contract import effective_voice_stack_identity
from production import (
    SemanticTurnAdmission,
    SemanticTurnEventRecorder,
    WorkerLocalEndCall,
    PeriodicWorkerHeartbeat,
    BROWSER_TEST_REPORT_BUDGET_SECONDS,
    browser_test_close_component,
    browser_test_error_metadata,
    build_livekit_model_tools,
    caller_disconnect_handler,
    cleanup_livekit_room,
    component_model_labels,
    executed_tool_names,
    finalize_call_artifacts,
    inbound_sip_identity,
    post_livekit_worker_heartbeat,
    profile_from_realtime_config,
    requires_gemini_tts_usage_metadata,
    requires_soniox_usage_correlation,
    safe_component_error_diagnostics,
    service_heartbeat_environ,
    terminate_pre_media_call,
    worker_heartbeats_enabled,
    write_session_state,
)
try:
    from release_canary_contract import (
        INBOUND_CUSTOMER,
        INBOUND_REJECT,
        INBOUND_RELEASE_CANARY,
        MAX_RELEASE_CANARY_SESSION_SECONDS,
        acknowledge_release_dispatch_probe,
        classify_inbound_leg,
        fixed_canary,
        is_release_dispatch_probe,
    )
except ModuleNotFoundError as exc:  # Public self-host export omits the private release dial.
    if exc.name != "release_canary_contract":
        raise

    INBOUND_CUSTOMER = "customer"
    INBOUND_RELEASE_CANARY = "release_canary"
    INBOUND_REJECT = "reject"
    MAX_RELEASE_CANARY_SESSION_SECONDS = 120

    def classify_inbound_leg(**_kwargs: Any) -> str:
        # A self-host deployment owns no release loopback, so every inbound
        # leg is customer traffic and no tuple is reserved.
        return INBOUND_CUSTOMER

    def fixed_canary() -> Any:
        raise RuntimeError("platform release canary is unavailable in the self-host export")

    def is_release_dispatch_probe(**_kwargs: Any) -> bool:
        # No release hook exists here either, so no job is ever a probe and
        # every dispatch is served as a real session.
        return False

    async def acknowledge_release_dispatch_probe(**_kwargs: Any) -> None:
        raise RuntimeError("platform release dispatch probe is unavailable in the self-host export")


AGENT_NAME = "sch236-livekit-demo"
SESSION_ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")
EVIDENCE_ROOT = Path(os.environ.get("SCH236_CALL_EVIDENCE_ROOT") or "/call-evidence")
_service_heartbeat: PeriodicWorkerHeartbeat | None = None
_service_heartbeat_lock = threading.Lock()


def emit(event: str, **details: Any) -> None:
    print(
        "SCH236_PRODUCTION "
        + json.dumps({"event": event, **details}, sort_keys=True, allow_nan=False),
        flush=True,
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_component(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in "-_")[:96]


def _transport(token_variable: str) -> UrllibJsonTransport:
    return UrllibJsonTransport(
        base_url=os.environ["OPENPHONEX_CONTROL_PLANE_URL"],
        bearer_token=os.environ[token_variable],
    )


def _call_transport() -> UrllibJsonTransport:
    """Call-bound transport: the org worker key if present, else the gateway token."""
    return UrllibJsonTransport(
        base_url=os.environ["OPENPHONEX_CONTROL_PLANE_URL"],
        bearer_token=call_credential_token(os.environ),
    )


def build_adapter() -> Gate2ControlPlaneAdapter:
    platform_token = os.environ.get("OPENPHONEX_PLATFORM_MEDIA_TOKEN", "").strip()
    call_transport = _call_transport()
    return Gate2ControlPlaneAdapter(
        gateway_id=os.environ["OPENPHONEX_GATEWAY_ID"],
        organization_id=os.environ.get("OPENPHONEX_ORGANIZATION_ID", ""),
        transport=_transport("OPENPHONEX_GATEWAY_API_TOKEN"),
        call_transport=call_transport,
        tool_transport=call_transport,
        platform_transport=(
            _transport("OPENPHONEX_PLATFORM_MEDIA_TOKEN") if platform_token else None
        ),
    )


def setup_process(process: JobProcess) -> None:
    # Best-effort pre-import of provider plugin modules so warm
    # processes do not pay the import cost on their first job.  A missing or
    # broken optional plugin must not crash the warm process, but its failure
    # is reported (not silently swallowed) via the process_warm event.
    warmed, failed = prewarm_plugin_modules(
        (
            "livekit.plugins.deepgram",
            "livekit.plugins.google",
            "livekit.plugins.elevenlabs",
            "livekit.plugins.soniox",
            "livekit.plugins.fishaudio",
            "livekit.plugins.silero",
        )
    )

    from livekit.plugins import silero

    process.userdata["vad"] = silero.VAD.load(min_silence_duration=0.25)
    _start_process_heartbeat(process)
    emit(
        "process_warm",
        components=warmed,
        failed=[{"module": name, "error": error} for name, error in failed],
    )


def _heartbeat_interval_seconds() -> float:
    try:
        return max(10.0, float(os.environ.get("OPENPHONEX_WORKER_HEARTBEAT_SECONDS", "30")))
    except ValueError:
        return 30.0


def _new_worker_heartbeat(*, heartbeat_environ: dict[str, str]) -> PeriodicWorkerHeartbeat | None:
    if not worker_heartbeats_enabled(os.environ):
        emit("worker_heartbeat_disabled", reason="disabled_by_environment")
        return None
    base_url = str(os.environ.get("OPENPHONEX_CONTROL_PLANE_URL") or "").strip()
    worker_token = str(os.environ.get("OPENPHONEX_WORKER_API_TOKEN") or "").strip()
    if not base_url or not worker_token:
        emit(
            "worker_heartbeat_disabled",
            missing=[
                name
                for name, value in (
                    ("OPENPHONEX_CONTROL_PLANE_URL", base_url),
                    ("OPENPHONEX_WORKER_API_TOKEN", worker_token),
                )
                if not value
            ],
        )
        return None
    transport = UrllibJsonTransport(
        base_url=base_url,
        bearer_token=worker_token,
    )
    return PeriodicWorkerHeartbeat(
        send=lambda: post_livekit_worker_heartbeat(
            transport,
            heartbeat_environ,
            process_id=os.getpid(),
        ),
        interval_seconds=_heartbeat_interval_seconds(),
        on_failure=lambda exc: emit("worker_heartbeat_failed", error=type(exc).__name__),
    )


def _start_process_heartbeat(process: JobProcess) -> None:
    if process.userdata.get("worker_heartbeat") is not None:
        return
    heartbeat = _new_worker_heartbeat(heartbeat_environ=dict(os.environ))
    if heartbeat is None:
        return
    process.userdata["worker_heartbeat"] = heartbeat
    heartbeat.start()


def _stop_service_heartbeat() -> None:
    global _service_heartbeat
    with _service_heartbeat_lock:
        heartbeat = _service_heartbeat
        _service_heartbeat = None
    if heartbeat is not None:
        heartbeat.close(timeout_seconds=1.0)


def _start_service_heartbeat() -> None:
    """Emit readiness from PID 1 even when LiveKit has not assigned a job yet."""
    global _service_heartbeat
    with _service_heartbeat_lock:
        if _service_heartbeat is not None:
            return
        # The service heartbeat is deliberately a different row than a child
        # JobProcess heartbeat, so neither can mask the other's exact revision.
        service_environ = service_heartbeat_environ(os.environ)
        heartbeat = _new_worker_heartbeat(heartbeat_environ=service_environ)
        if heartbeat is None:
            return
        _service_heartbeat = heartbeat
        heartbeat.start()


async def start_evidence_sidecar(
    *, room_name: str, caller_identity: str, recorder: SynchronizedEvidenceRecorder
) -> tuple[rtc.Room, set[asyncio.Task[None]]]:
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(f"evidence-{safe_component(caller_identity)}")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    tasks: set[asyncio.Task[None]] = set()

    async def capture(track: rtc.Track, role: str) -> None:
        # One cursor per concrete subscription keeps queued frames in media
        # order. A republished track gets a new wall-clock anchor.
        stream_id = object()
        stream = rtc.AudioStream(
            track,
            sample_rate=8_000,
            num_channels=1,
            frame_size_ms=20,
        )
        try:
            async for event in stream:
                recorder.append_stream_pcm(
                    role,  # type: ignore[arg-type]
                    bytes(event.frame.data),
                    stream_id=stream_id,
                    received_at=time.monotonic(),
                )
        finally:
            await stream.aclose()

    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        role = "caller" if participant.identity == caller_identity else "agent"
        task = asyncio.create_task(capture(track, role))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    room.on("track_subscribed", on_track_subscribed)
    await room.connect(os.environ["LIVEKIT_URL"], token)
    return room, tasks


async def remove_sip_participant_before_media(
    ctx: JobContext, participant: Any | None
) -> None:
    """Hang up a pre-media SIP participant before the room backstop.

    A room delete usually removes participants too, but an explicit removal
    bounds the ringing leg when configuration or authorization fails before
    the agent begins consuming audio.
    """

    if participant is None:
        return
    room_name = str(getattr(ctx.room, "name", "") or "").strip()
    participant_identity = str(getattr(participant, "identity", "") or "").strip()
    if not room_name or not participant_identity:
        return
    client = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        await client.room.remove_participant(
            api.RoomParticipantIdentity(room=room_name, identity=participant_identity)
        )
    finally:
        await client.aclose()


class WorkflowTurnAgent(Agent):
    """A Talk-mode Agent whose replies are dictated by a pinned workflow run.

    The control plane stays the sole execution authority: every reply comes
    from POST /v1/calls/{call_id}/workflow-turns, never from this session's
    own model. ``llm_node`` is LiveKit's documented override point for
    substituting a non-model reply source without touching STT/TTS/turn
    detection -- see https://docs.livekit.io/agents (Agent.llm_node). tools
    is deliberately empty: the deterministic runner already executes
    tools/transfers server-side inside advance_workflow_turn, so there is
    nothing for the LLM-style function-tool machinery to call here.

    Every DECISION this makes (what to speak, whether the call is over) is
    delegated to workflow_turns.py, which is unit-tested; this class is only
    the untestable wiring to LiveKit's session/speech primitives -- kept as
    small as the SDK's contract allows. Live-session behavior (in
    particular whether yielding no text here is safe, and whether
    current_speech is already the handle for the turn being generated at
    the moment llm_node runs) is unverified in this repo (no livekit
    dependency at all) and is proven post-merge against a real LiveKit
    session, per the same bar PR #343 shipped Talk's sibling browser-test
    feature against. A missing or unusable handle is not a failure: the
    termination path falls back to the fixed grace.
    """

    def __init__(self, *, instructions: str, call: Any, language: str, on_terminal: Any) -> None:
        super().__init__(instructions=instructions, tools=[])
        self._call = call
        self._language = language
        self._on_terminal = on_terminal
        # session.start enables caller audio before the opening advance has
        # run, so ordering the two is not optional -- see the coordinator's
        # own docstring for both failure modes. It lives in workflow_turns
        # because this class cannot be imported, let alone tested, here.
        self._turns = WorkflowTurnCoordinator()

    def _current_speech_handle(self) -> Any:
        """This turn's SpeechHandle, or None if the SDK will not give one.

        ``Agent.session`` raises when the agent is not running, and
        ``current_speech`` is legitimately None outside a speech, so a
        missing handle is an ordinary outcome here -- the termination path
        falls back to its fixed grace when it gets one.
        """
        try:
            return self.session.current_speech
        except Exception:
            return None

    async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any):
        caller_text = latest_caller_text(chat_ctx.items)
        try:
            response = await self._turns.run_caller_turn(
                lambda: asyncio.to_thread(self._call.advance_workflow_turn, caller_text)
            )
        except OpeningAdvanceFailed:
            # The opening turn already ended this call and is speaking the
            # goodbye right now. Saying anything here would talk over it and
            # schedule a second hangup, so this turn produces nothing at all.
            return
        except Exception as exc:
            emit("workflow_turn_advance_failed", call_id=self._call.call_id, error=type(exc).__name__)
            yield workflow_turn_error_message(self._language)
            self._on_terminal(self._current_speech_handle())
            return
        outcome = build_turn_outcome(response, self._language)
        if outcome.reply_text:
            yield outcome.reply_text
        if outcome.terminal:
            # The speech the pipeline is building from the text just yielded.
            # Captured, never awaited here: wait_for_playout refuses to be
            # called from within a function that owns the same handle, and
            # awaiting the speech this generator is feeding would deadlock.
            # The detached termination task does the waiting.
            self._on_terminal(self._current_speech_handle())

    async def _speak_final(self, session: Any, text: str) -> None:
        """Speak an opening line that ends the call, then tear the call down.

        The teardown is in a finally because session.say can raise (TTS or
        session failure) and the call still has to end: with the termination
        after the say, that exception left the session open with nothing
        left to speak. An empty text still terminates -- a terminal outcome
        with no authored farewell is silent by design, not undecided.
        """
        try:
            # SpeechHandle.__await__ calls wait_for_playout, so this returns
            # only once the line has finished playing.
            if text:
                await session.say(text, allow_interruptions=True)
        finally:
            self._on_terminal(None)

    async def speak_opening_turn(self, session: Any) -> None:
        """Speak the workflow's entry step instead of the agent's configured
        first_message -- the opening turn goes through the same call-bound
        endpoint as every later one, with caller_input=None.

        Waiting caller turns are released only in the finally, once this has
        finished deciding AND speaking: session.start enables audio before
        this runs, so a caller who speaks immediately is already parked
        inside llm_node, and releasing them any earlier let them talk over
        this turn's own goodbye.
        """
        conversation_over = False
        try:
            try:
                response = await self._turns.run_opening(
                    lambda: asyncio.to_thread(self._call.advance_workflow_turn, None)
                )
            except Exception as exc:
                emit("workflow_turn_advance_failed", call_id=self._call.call_id, error=type(exc).__name__)
                conversation_over = True
                await self._speak_final(session, workflow_turn_error_message(self._language))
                return
            outcome = build_turn_outcome(response, self._language)
            if outcome.terminal:
                # The opening WAS the whole conversation. The flag is set
                # BEFORE anything is spoken because the finally below hands
                # it to every parked caller turn: recorded after the say, a
                # say that raised released those turns as if the call were
                # still going, and one of them advanced the finished run and
                # spoke a second goodbye over this one.
                conversation_over = True
                await self._speak_final(session, outcome.reply_text)
            elif outcome.reply_text:
                await session.say(outcome.reply_text, allow_interruptions=True)
        finally:
            # Always, on every path -- a caller turn parked on this event
            # would otherwise hang the session instead of ending it.
            self._turns.opening_finished(conversation_over=conversation_over)


def schedule_workflow_turn_termination(finish: Any, speech_handle: Any = None) -> None:
    """End the call once the terminal farewell has actually been heard.

    Detached on purpose: the caller is usually the very generator feeding
    the speech we are waiting on. The decision of what to wait for lives in
    workflow_turns.await_playout_then_finish, which is unit-tested; this is
    only the task creation. See TERMINATION_GRACE_SECONDS for what the
    grace now bounds.
    """
    asyncio.create_task(await_playout_then_finish(speech_handle, finish))


server = AgentServer(
    num_idle_processes=resolve_idle_process_count(os.environ), setup_fnc=setup_process
)


@server.rtc_session(agent_name=AGENT_NAME)
async def production_agent(ctx: JobContext) -> None:
    # The release host asks LiveKit for one throwaway job before it dials, to
    # learn whether this worker can be given a job at all.  Serving it as a
    # session would mean waiting for a participant that is never coming and
    # then reporting a rejected call that never existed -- fabricated failure
    # rows in the one log a release is read from.  Answering it here, before
    # connecting to anything, keeps the probe's cost at one job slot and its
    # meaning exact: LiveKit assigned this worker a job and this worker ran it.
    probe_room_name = str(getattr(getattr(ctx.job, "room", None), "name", "") or "")
    if is_release_dispatch_probe(
        room_name=probe_room_name,
        metadata=str(getattr(ctx.job, "metadata", "") or ""),
    ):
        # `AgentDispatch.state.jobs` is transient: once this entrypoint
        # returns, LiveKit may remove the job before the host's next poll.
        # Persist an acknowledgement in the throwaway room before ending the
        # job, so the host observes *this worker* having served the exact
        # dispatch rather than racing a scheduler-internal snapshot.
        #
        try:
            await acknowledge_release_dispatch_probe(
                room_service=ctx.api.room,
                room_name=probe_room_name,
                update_room_metadata_request=api.UpdateRoomMetadataRequest,
                shutdown=ctx.shutdown,
            )
        except BaseException as exc:
            emit("release_dispatch_probe_ack_failed", error=type(exc).__name__)
            raise
        else:
            emit("release_dispatch_probe_served")
        return
    # Room creation and SIP signaling are not authorization.  The worker joins
    # unsubscribed and cannot consume caller audio until the control plane has
    # resolved the number route and semantic config.
    call: Any | None = None
    adapter: Gate2ControlPlaneAdapter | None = None
    participant: Any | None = None
    # Bound before the guard below, not inside it: the dispatch is parsed after
    # ``ctx.connect``, which can raise, and the handler reads this to decide
    # whether anybody is listening.
    browser_test: Any | None = None
    pre_media_call_id = ""
    pre_media_stage = "session_connect"
    inbound_route = INBOUND_CUSTOMER
    leased_components: dict[str, dict[str, str]] | None = None
    runtime_environ: dict[str, str] | None = None
    browser_test_failure_reported = False
    browser_test_report_in_flight = False
    # Held so the teardown can wait for it. The report has to land while the
    # room still exists -- it IS the room's metadata, and the browser reads it
    # from there -- so a write still in flight when cleanup deletes the room is
    # a report nobody receives.
    browser_test_report_task: asyncio.Task[None] | None = None

    async def report_browser_test_failure(component: str) -> None:
        """Tell the browser its session will not speak.

        Defined before the pre-media guard because BOTH sides need it: a stack
        that fails to build never reaches the session at all, and the worker can
        already be in the room by then -- so without this the dock would show a
        live session ending in an ordinary hangup.

        Success is what closes this, not the attempt: marking it done up front
        meant one failed write silenced every later report, which is precisely
        the connected-but-mute session this exists to prevent. The retries are a
        LOOP rather than a counter checked once per call, because both callers
        report exactly once -- a budget spent one attempt per invocation is a
        budget nothing can ever draw on.

        One deadline for the whole thing, not one per attempt. The pre-media
        caller awaits this DIRECTLY, holding up the teardown behind it, and the
        teardown of a live session waits on it for exactly this budget;
        per-attempt timeouts multiply past both and go on writing to a room
        already being deleted.

        Cancellation is not a failure to log -- it is a caller withdrawing the
        budget -- so it propagates instead of being caught and retried. Both
        callers are prepared for that: the pre-media guard runs its teardown in
        a `finally`, and the session teardown waits unshielded so its own
        timeout is what does the cancelling.
        """
        nonlocal browser_test_failure_reported, browser_test_report_in_flight
        if browser_test_failure_reported or browser_test_report_in_flight:
            return
        browser_test_report_in_flight = True
        deadline = time.monotonic() + BROWSER_TEST_REPORT_BUDGET_SECONDS
        try:
            for attempt in range(3):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    emit("workspace_browser_test_error_publish_timeout", call_id=browser_test.call_id)
                    return
                try:
                    await asyncio.wait_for(
                        ctx.api.room.update_room_metadata(
                            api.UpdateRoomMetadataRequest(
                                room=ctx.room.name,
                                metadata=browser_test_error_metadata(
                                    str(getattr(ctx.room, "metadata", "") or ""),
                                    component=component,
                                ),
                            )
                        ),
                        timeout=remaining,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # A room that cannot be told is still a room the deadline
                    # reaps, and this must never displace the failure it is
                    # reporting.
                    emit(
                        "workspace_browser_test_error_publish_failed",
                        error=type(exc).__name__,
                        attempt=attempt + 1,
                    )
                else:
                    browser_test_failure_reported = True
                    return
        finally:
            browser_test_report_in_flight = False

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
        pre_media_stage = "dispatch_validation"
        job_metadata = str(getattr(ctx.job, "metadata", "") or "")
        outbound = parse_outbound_dispatch(job_metadata)
        browser_test = None if outbound is not None else parse_workspace_browser_test_dispatch(job_metadata)
        if outbound is not None:
            pre_media_call_id = outbound.call_id
        elif browser_test is not None:
            pre_media_call_id = browser_test.call_id
        pre_media_stage = "participant_wait"
        participant = await ctx.wait_for_participant()
        attributes = dict(participant.attributes or {})
        caller = str(attributes.get("sip.phoneNumber") or attributes.get("sip.from") or "unknown")
        provider_call_id = str(attributes.get("sip.callIDFull") or attributes.get("sip.callID") or participant.identity)
        pre_media_stage = "adapter_configuration"
        adapter = build_adapter()
        if browser_test is not None:
            if participant.identity != browser_test.participant_identity:
                raise ValueError("browser-test participant identity does not match dispatch")
            caller = f"browser:{browser_test.call_id}"
            pre_media_stage = "outbound_resume"
            call = await asyncio.to_thread(
                adapter.prepare_workspace_browser_test,
                call_id=browser_test.call_id,
                agent_id=browser_test.agent_id,
                organization_id=browser_test.organization_id,
                project_id=browser_test.project_id,
            )
        elif outbound is not None:
            if participant.identity != outbound.participant_identity:
                raise ValueError("outbound SIP participant identity does not match dispatch")
            if str(attributes.get("openphonex.call_id") or "") != outbound.call_id:
                raise ValueError("outbound SIP participant is not bound to the dispatched call")
            caller = outbound.to_e164
            pre_media_stage = "outbound_resume"
            call = await asyncio.to_thread(
                adapter.resume_authorized_outbound,
                call_id=outbound.call_id,
                agent_id=outbound.agent_id,
                from_number_id=outbound.from_number_id,
                to_e164=outbound.to_e164,
            )
        else:
            caller, called_did, provider_call_id = inbound_sip_identity(
                attributes, participant.identity
            )
            pre_media_stage = "inbound_prepare"
            # The reserved release tuple has exactly two outcomes here, and
            # normal customer routing is not one of them. A missing host
            # switch fails the release canary before it can create a customer
            # call, billing entry, dashboard row, or usage record.
            #
            # The inbound room name is deliberately not consulted. It is
            # chosen by the production dispatch rule, not by the host-side
            # LiveKit API, so it proves nothing about who placed the call.
            inbound_route = classify_inbound_leg(
                release_canary_enabled=os.environ.get("OPENPHONEX_RELEASE_CANARY_ENABLED", "") == "1",
                from_e164=caller,
                to_e164=called_did,
            )
            if inbound_route == INBOUND_REJECT:
                raise ValueError("untrusted_release_canary_inbound")
            if inbound_route == INBOUND_RELEASE_CANARY:
                # No caller-controlled billing exemption exists here. The
                # control plane admits this route only while a single-use
                # claim armed by the release host is open, and rejects it
                # otherwise — which terminates the leg without a customer
                # record rather than downgrading it to customer traffic.
                call = await asyncio.to_thread(
                    adapter.prepare_platform_release_canary,
                    fixed_canary().as_dict(),
                    provider_call_id,
                )
            elif inbound_route == INBOUND_CUSTOMER:
                call = await asyncio.to_thread(
                    adapter.prepare_inbound,
                    to_e164=called_did,
                    from_e164=caller,
                    provider_call_id=provider_call_id,
                )
            else:
                # An unrecognized classification is a bug, and customer
                # routing is the one place it must not default to.
                raise ValueError("unclassified_inbound_leg")
        pre_media_call_id = call.call_id
        pre_media_stage = "component_configuration"
        profile = profile_from_realtime_config(call.config)
        # Presence alone flips this session from the free-form AgentSession
        # path to the workflow-driven one; the branch below is the entire
        # gate. Only ever set for a Talk-mode browser-test call (4b) --
        # every other call keeps profile.instructions/first_message and the
        # plain Agent exactly as before.
        workflow_run_id = pinned_workflow_run_id(call.config)
        # Selected customer keys are leased only after the call-bound config
        # arrives and before any provider object/media session exists. The
        # overlay is local to this invocation; it never mutates os.environ,
        # job metadata, room metadata, telemetry, or logs.
        leased_components = call.lease_component_credentials() if has_byok_component(call.config) else {}
        runtime_environ = runtime_environ_with_component_credentials(
            call.config,
            base_environ=os.environ,
            leased_components=leased_components,
        )
        # Built before admit_media(): a misconfigured provider (including a
        # rejected/missing BYOK lease) fails before caller audio is consumed.
        # Fish's exact UTF-8 byte meter has no Soniox usage-log correlation.
        # Do not label a Fish-only component-metered call as a Soniox stack;
        # only the selected Soniox STT/TTS adapters need this opaque reference.
        soniox_correlation_required = requires_soniox_usage_correlation(profile, call.config)
        gemini_tts_usage_required = requires_gemini_tts_usage_metadata(profile, call.config)
        stack = build_demo_stack(
            profile,
            environ=runtime_environ,
            load_vad=lambda **_kwargs: ctx.proc.userdata["vad"],
            soniox_usage_reference=(
                call.call_id
                if soniox_correlation_required
                else ""
            ),
            # New component-metered Soniox stacks need an opaque call
            # reference before media. Historical pinned stacks without this
            # billing contract remain runnable rather than being rewritten or
            # newly rejected merely because the registry evolved.
            require_soniox_usage_reference=soniox_correlation_required,
            require_gemini_tts_usage_metadata=gemini_tts_usage_required,
        )
        # This is the last operation before caller audio can be consumed.  It
        # must remain inside the pre-media guard: a rejected admission still
        # owns a ringing SIP leg and an existing wallet reservation.
        pre_media_stage = "media_admission"
        call.admit_media()
    except BaseException:
        # The report is attempted first -- before the teardown deletes the room
        # it lives in -- but it can never cost the teardown. A browser test
        # rejected here has nothing to show for itself otherwise: the worker may
        # already be in the room, so the dock reads a live session that simply
        # hung up.
        #
        # `finally`, because the reporter re-raises `CancelledError` now. Without
        # it a cancelled job skips `terminate_pre_media_call` and strands exactly
        # what that call exists to release: a ringing SIP leg, a LiveKit room,
        # and a wallet reservation. Telling a browser is never worth that.
        try:
            if browser_test is not None:
                await report_browser_test_failure("session")
        finally:
            result = await terminate_pre_media_call(
                call=call,
                adapter=adapter,
                call_id=pre_media_call_id,
                reason={
                    "session_connect": "session_connect_rejected",
                    "dispatch_validation": "dispatch_validation_rejected",
                    "participant_wait": "participant_wait_rejected",
                    "adapter_configuration": "adapter_configuration_rejected",
                    "outbound_resume": "outbound_resume_rejected",
                    "inbound_prepare": "inbound_prepare_rejected",
                    "component_configuration": "component_configuration_rejected",
                    "media_admission": "media_admission_rejected",
                }.get(pre_media_stage, "pre_media_rejected"),
                remove_participant=lambda: remove_sip_participant_before_media(ctx, participant),
                delete_room=ctx.delete_room,
            )
            emit(
                "pre_media_terminated",
                stage=pre_media_stage,
                call_id=pre_media_call_id,
                participant_terminated=result["participant_terminated"],
                room_terminated=result["room_terminated"],
                lifecycle_recorded=result["lifecycle_recorded"],
            )
        raise
    finally:
        # Provider instances hold their own configuration, but the transient
        # response/overlay mappings must not remain in this worker scope.
        if leased_components is not None:
            del leased_components
        if runtime_environ is not None:
            del runtime_environ
    for publication in participant.track_publications.values():
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            publication.set_subscribed(True)

    evidence_dir = EVIDENCE_ROOT / safe_component(call.call_id)
    recorder = SynchronizedEvidenceRecorder(
        evidence_dir,
        transcript_metadata={
            "openphonex": {
                "agent_id": str(call.config.get("agent_id") or ""),
                "organization_id": str(call.config.get("organization_id") or ""),
                "project_id": str(call.config.get("project_id") or ""),
                "call_id": call.call_id,
            },
            "effective_voice_stack": effective_voice_stack_identity(call.config),
        },
    )
    sidecar, capture_tasks = await start_evidence_sidecar(
        room_name=ctx.room.name,
        caller_identity=participant.identity,
        recorder=recorder,
    )
    state_path = SESSION_ROOT / f"{safe_component(call.call_id)}.json"
    started_at = utcnow()
    write_session_state(
        state_path,
        call_id=call.call_id,
        room_name=ctx.room.name,
        participant_identity=participant.identity,
        started_at=started_at,
        last_heartbeat_at=started_at,
        status="media_started",
    )
    await asyncio.to_thread(
        call.report_lifecycle,
        event_id=f"{call.call_id}:sip-answered",
        event_type="sip.answered",
        source_component="livekit_sip",
        observed_at=started_at,
        details={"provider_call_id": provider_call_id},
    )
    await asyncio.to_thread(
        call.report_lifecycle,
        event_id=f"{call.call_id}:media-started",
        event_type="room.media_started",
        source_component="livekit_room",
        observed_at=started_at,
        details={"room_name": ctx.room.name},
    )

    async def heartbeat() -> None:
        sequence = 0
        while True:
            await asyncio.sleep(5)
            sequence += 1
            observed_at = utcnow()
            await asyncio.to_thread(
                call.report_lifecycle,
                event_id=f"{call.call_id}:heartbeat:{sequence}",
                event_type="room.media_heartbeat",
                source_component="livekit_room",
                observed_at=observed_at,
                details={"room_name": ctx.room.name},
            )
            write_session_state(
                state_path,
                call_id=call.call_id,
                room_name=ctx.room.name,
                participant_identity=participant.identity,
                started_at=started_at,
                last_heartbeat_at=observed_at,
                status="media_started",
            )

    heartbeat_task = asyncio.create_task(heartbeat())

    session_options = dict(stack.session_options)
    if browser_test is not None:
        # A browser test closes on the FIRST unrecoverable component error.
        #
        # LiveKit's default absorbs three per component before giving up, which
        # is right for a phone call: a caller is on the line, and three more
        # tries beat hanging up on them. A browser test is the opposite. Nobody
        # is on the line, somebody is watching a page, and three silent turns is
        # exactly the connected-but-mute session this work exists to abolish.
        # Closing here is what makes the report below prompt rather than
        # eventual, and it keeps the decision inside LiveKit's own budget
        # instead of second-guessing it from a handler.
        session_options["conn_options"] = SessionConnectOptions(max_unrecoverable_errors=0)
    session = AgentSession(**session_options)
    telemetry_collector = CallTelemetryCollector(
        call_id=call.call_id,
        organization_id=str(call.config.get("organization_id") or ""),
        project_id=str(call.config.get("project_id") or ""),
    )
    # Independent from telemetry: this is the SDK's typed usage collector, not
    # timing/message metrics and not a duration estimate.  Its callback only
    # updates bounded in-memory counters; the single control-plane report is
    # emitted during teardown before terminal SIP lifecycle settlement.
    billing_usage_collector = ModelUsageCollector()
    # A separate bounded counter records only successful tool wrappers. It
    # contains no arguments/results and is passed once with the native SDK
    # usage report after media ends, never from a latency-critical callback.
    component_metering_enabled = (
        isinstance(call.config.get("component_billing"), dict)
        and call.config["component_billing"].get("usage_required") is True
    )
    successful_tool_usage = SuccessfulToolUsageCounter() if component_metering_enabled else None
    # ``invoke_tool`` already crosses the control plane off the media loop.
    # Its optional observer only appends bounded in-memory timing metadata, so
    # it cannot add I/O or failure modes to media callbacks.
    call.telemetry_observer = telemetry_collector.observe_tool
    provider_labels = {
        "stt": str((call.config.get("transcriber") or {}).get("registry_name") or ""),
        "llm": str((call.config.get("model") or {}).get("registry_name") or ""),
        "tts": str((call.config.get("voice") or {}).get("registry_name") or ""),
    }
    finished = asyncio.Event()
    semantic_admission = SemanticTurnAdmission()
    turn_events = SemanticTurnEventRecorder(
        recorder=recorder,
        telemetry_collector=telemetry_collector,
        admission=semantic_admission,
        first_message=profile.first_message,
        provider_labels=provider_labels,
        model_labels=component_model_labels(profile),
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
    )

    def stop_semantic_turns() -> None:
        semantic_admission.close()

    def finish_session() -> None:
        stop_semantic_turns()
        finished.set()

    def begin_worker_termination() -> None:
        # The end-call action starts here, before its goodbye is scheduled:
        # the recorder uses this boundary to prove a final agent interval is
        # wholly worker-owned goodbye speech before attributing the terminal
        # message to its turn.
        recorder.mark_termination()
        stop_semantic_turns()

    end_call = WorkerLocalEndCall(
        session=session,
        language=profile.language,
        on_ending=begin_worker_termination,
        finish=finish_session,
        record_final_message=recorder.add_terminal_message,
        is_finished=finished.is_set,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
    )
    tools = await asyncio.to_thread(
        build_livekit_model_tools,
        call,
        end_call=end_call,
        tool_usage_counter=successful_tool_usage,
    )

    # The semantic-turn event contract lives in ``production.py`` so it can be
    # replayed offline; these handlers must stay thin adapters over it.
    @session.on("user_state_changed")
    def on_user_state(event: Any) -> None:
        turn_events.on_user_state_changed(event)

    @session.on("agent_state_changed")
    def on_agent_state(event: Any) -> None:
        turn_events.on_agent_state_changed(event)

    @session.on("user_input_transcribed")
    def on_user_transcript(event: Any) -> None:
        turn_events.on_user_input_transcribed(event)

    @session.on("conversation_item_added")
    def on_conversation_item(event: Any) -> None:
        turn_events.on_conversation_item_added(event)

    @session.on("metrics_collected")
    def on_metrics_collected(event: Any) -> None:
        # LiveKit Agents 1.6.6 emits typed STT/LLM/TTS metrics here.  Never
        # serialize/log the event: the collector retains only provider/model
        # identities and numeric usage, and a malformed metrics callback must
        # not interrupt media.
        try:
            billing_usage_collector.collect(event.metrics)
        except Exception:
            emit("component_usage_metrics_rejected", call_id=call.call_id)

    @session.on("function_tools_executed")
    def on_tools(event: Any) -> None:
        emit(
            "model_tools_executed",
            call_id=call.call_id,
            tools=executed_tool_names(event),
        )

    @session.on("error")
    def on_error(event: Any) -> None:
        emit(
            "session_error",
            call_id=call.call_id,
            **safe_component_error_diagnostics(getattr(event, "error", None)),
        )

    @session.on("close")
    def on_close(event: Any) -> None:
        nonlocal browser_test_report_task
        # A browser test that connects, transcribes and never speaks looks
        # healthy from the page: the transcript fills with the customer's own
        # words and nothing says the pipeline is dead. The worker is the only
        # party that knows, so tell the room. Phone calls are untouched --
        # there is no dock listening, and a caller hears the silence.
        #
        # The CLOSE, not the error above: LiveKit absorbs several unrecoverable
        # component errors before it gives up, and resets that budget on a turn
        # that works. Reporting an error would end sessions it was about to
        # recover.
        if browser_test is None:
            return
        component = browser_test_close_component(
            reason=getattr(event, "reason", None), error=getattr(event, "error", None)
        )
        if component is None:
            return
        browser_test_report_task = asyncio.create_task(report_browser_test_failure(component))

    ctx.room.on(
        "participant_disconnected",
        caller_disconnect_handler(
            caller_identity=participant.identity, finish=finish_session
        ),
    )
    workflow_agent: WorkflowTurnAgent | None = None
    if workflow_run_id:
        workflow_agent = WorkflowTurnAgent(
            instructions=profile.instructions,
            call=call,
            language=profile.language,
            on_terminal=lambda handle=None: schedule_workflow_turn_termination(finish_session, handle),
        )
    try:
        await session.start(
            room=ctx.room,
            agent=workflow_agent if workflow_agent is not None else Agent(instructions=profile.instructions, tools=tools),
            room_options=RoomOptions(
                audio_input=True,
                audio_output=True,
                close_on_disconnect=False,
                delete_room_on_close=False,
            ),
        )
        if workflow_agent is not None:
            await workflow_agent.speak_opening_turn(session)
        else:
            await session.say(profile.first_message, allow_interruptions=True)
        emit(
            "media_admitted",
            call_id=call.call_id,
            language=profile.language,
            semantic_config_source="realtime-config",
        )
        if browser_test is not None:
            try:
                await asyncio.wait_for(finished.wait(), timeout=180)
            except (TimeoutError, asyncio.TimeoutError):
                emit("workspace_browser_test_deadline", call_id=call.call_id)
                finish_session()
        elif inbound_route == INBOUND_RELEASE_CANARY:
            # The platform-only route is outside customer billing and
            # lifecycle, so nothing else would ever end this session. A
            # release exchange takes seconds; anything still holding the line
            # after the ceiling is not the release, and must not keep
            # consuming provider capacity unmetered.
            try:
                await asyncio.wait_for(
                    finished.wait(), timeout=MAX_RELEASE_CANARY_SESSION_SECONDS
                )
            except (TimeoutError, asyncio.TimeoutError):
                emit("release_canary_session_deadline", call_id=call.call_id)
                finish_session()
        else:
            await finished.wait()
    finally:
        ended_at = utcnow()
        heartbeat_task.cancel()
        # Set if the browser-test report wait below is cancelled. Held so the
        # teardown can finish and the cancellation can still surface afterwards.
        browser_test_report_cancellation: BaseException | None = None
        async def cleanup_room() -> None:
            await cleanup_livekit_room(call, ctx.delete_room)

        try:
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            await session.aclose()
            # `aclose` is what emits the close event, so any browser-test report
            # exists by now. Waited for HERE because the report is the room's own
            # metadata and cleanup below deletes that room: a write still in
            # flight then is a report the browser never receives. Bounded on the
            # same budget the reporter uses, so neither side outlives the other.
            if browser_test_report_task is not None:
                # NOT shielded. A shield would make this timeout cancel the
                # shield and leave the reporter running, so a write that ignores
                # its own deadline could still be in flight when cleanup_room()
                # below deletes the room -- the one outcome this wait exists to
                # prevent. Unshielded, the timeout cancels the reporter, which is
                # what its CancelledError handler is written for.
                try:
                    await asyncio.wait_for(
                        browser_test_report_task,
                        timeout=BROWSER_TEST_REPORT_BUDGET_SECONDS,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    emit("workspace_browser_test_error_publish_timeout", call_id=call.call_id)
                except asyncio.CancelledError as exc:
                    # DEFERRED, not swallowed and not re-raised here.
                    #
                    # Re-raising immediately would skip everything after this
                    # wait -- the sidecar, capture tasks, component billing,
                    # room.ended and sip.ended -- and leave only the outer
                    # handler's room delete. Swallowing it outright would let a
                    # cancelled job run to the end and return as though it had
                    # succeeded, which is a lie to the worker supervising it.
                    #
                    # So the teardown runs to completion and the cancellation is
                    # re-raised at the end of it. Narrower than BaseException on
                    # purpose, so a KeyboardInterrupt still stops the worker now.
                    browser_test_report_cancellation = exc
                    emit("workspace_browser_test_error_publish_cancelled", call_id=call.call_id)
                except Exception as exc:
                    emit(
                        "workspace_browser_test_error_publish_failed",
                        error=type(exc).__name__,
                    )
            await sidecar.disconnect()
            if capture_tasks:
                await asyncio.gather(*capture_tasks, return_exceptions=True)
            if isinstance(call.config.get("component_billing"), dict) and call.config["component_billing"].get("usage_required") is True:
                try:
                    if stack.soniox_usage_tracker is not None:
                        try:
                            soniox_usage = await asyncio.to_thread(stack.soniox_usage_tracker.collect)
                        finally:
                            stack.soniox_usage_tracker.discard_credentials()
                    else:
                        soniox_usage = None
                    usage_payload = normalized_usage_from_model_usage(
                        call.config,
                        billing_usage_collector.flatten(),
                        tool_units=(successful_tool_usage.snapshot() if successful_tool_usage else {}),
                        tts_utf8_bytes=(
                            stack.tts_utf8_byte_counter.snapshot()
                            if stack.tts_utf8_byte_counter is not None
                            else None
                        ),
                        soniox_usage=soniox_usage,
                    )
                    await asyncio.to_thread(call.report_component_usage, usage_payload)
                    emit(
                        "component_usage_reported",
                        call_id=call.call_id,
                        stt_audio_seconds=usage_payload["stt_audio_seconds"],
                        stt_input_audio_tokens=usage_payload["stt_input_audio_tokens"],
                        stt_output_text_tokens=usage_payload["stt_output_text_tokens"],
                        llm_input_tokens=usage_payload["llm_input_tokens"],
                        llm_cached_input_tokens=usage_payload["llm_cached_input_tokens"],
                        llm_output_tokens=usage_payload["llm_output_tokens"],
                        tts_characters=usage_payload["tts_characters"],
                        tts_audio_seconds=usage_payload["tts_audio_seconds"],
                        tts_utf8_bytes=usage_payload["tts_utf8_bytes"],
                        tts_input_text_tokens=usage_payload["tts_input_text_tokens"],
                        tts_output_audio_tokens=usage_payload["tts_output_audio_tokens"],
                    )
                except MeteringContractError as exc:
                    # Do not replace missing native evidence with a duration
                    # estimate. sip.ended will retain the existing hold as
                    # awaiting_usage for an explicit retry/reconciliation path.
                    emit("component_usage_unavailable", call_id=call.call_id, error_type=type(exc).__name__)
                except Exception as exc:
                    # Network/report failures are also non-billable until the
                    # durable control-plane retry/reconciliation path obtains
                    # the report. The media teardown must continue.
                    emit("component_usage_report_failed", call_id=call.call_id, error_type=type(exc).__name__)
            summary: dict[str, object] | None = None
            try:
                summary = recorder.finalize()
            except NoRoomAudioCaptured:
                emit(
                    "evidence_unavailable",
                    call_id=call.call_id,
                    semantic_evidence_status="unavailable_no_room_audio",
                )
            try:
                telemetry_bundle = telemetry_collector.write(
                    evidence_dir / "call-telemetry.json"
                )
                emit(
                    "call_telemetry_written",
                    call_id=call.call_id,
                    evidence_status=telemetry_bundle["evidence"]["status"],
                    span_count=telemetry_bundle["span_count"],
                )
            except Exception as exc:
                # Telemetry is audit evidence, never a media/teardown gate.
                emit(
                    "call_telemetry_write_failed",
                    call_id=call.call_id,
                    error_type=type(exc).__name__,
                )
            await asyncio.to_thread(
                call.report_lifecycle,
                event_id=f"{call.call_id}:room-ended",
                event_type="room.ended",
                source_component="livekit_room",
                observed_at=ended_at,
            )
            await asyncio.to_thread(
                call.report_lifecycle,
                event_id=f"{call.call_id}:sip-ended",
                event_type="sip.ended",
                source_component="livekit_sip",
                observed_at=ended_at,
            )
        except BaseException:
            await asyncio.shield(cleanup_room())
            raise
        try:
            await finalize_call_artifacts(
                call=call,
                call_id=call.call_id,
                evidence_dir=evidence_dir,
                state_path=state_path,
                summary=summary,
                room_name=ctx.room.name,
                started_at=started_at,
                ended_at=ended_at,
                telemetry_turns=telemetry_collector.bundle()["turn_count"],
                emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            )
        finally:
            await asyncio.shield(cleanup_room())
        # The teardown has finished, including both room-delete paths, so the
        # cancellation deferred at the report wait can surface now. Without this
        # a cancelled job returns as though it had succeeded.
        if browser_test_report_cancellation is not None:
            raise browser_test_report_cancellation


if __name__ == "__main__":
    atexit.register(_stop_service_heartbeat)
    _start_service_heartbeat()
    cli.run_app(server)

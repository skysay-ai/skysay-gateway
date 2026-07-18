# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import json
import os
import time
import wave
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
from livekit.agents.voice.room_io import RoomOptions

from control_plane import Gate2ControlPlaneAdapter, UrllibJsonTransport
from demo import build_demo_stack
from prewarm import prewarm_plugin_modules, resolve_idle_process_count
from livekit_evidence import SynchronizedEvidenceRecorder
from latency_evidence import TurnLatencyRecorder
from evidence_contract import effective_voice_stack_identity
from production import (
    build_livekit_model_tools,
    executed_tool_names,
    is_greeting_fragment,
    profile_from_realtime_config,
)


AGENT_NAME = "sch236-livekit-demo"
SESSION_ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")
EVIDENCE_ROOT = Path(os.environ.get("SCH236_CALL_EVIDENCE_ROOT") or "/call-evidence")


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


def build_adapter() -> Gate2ControlPlaneAdapter:
    return Gate2ControlPlaneAdapter(
        gateway_id=os.environ["OPENPHONEX_GATEWAY_ID"],
        transport=_transport("OPENPHONEX_GATEWAY_API_TOKEN"),
        call_transport=_transport("OPENPHONEX_WORKER_API_TOKEN"),
        tool_transport=_transport("OPENPHONEX_WORKER_API_TOKEN"),
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
            "livekit.plugins.silero",
        )
    )

    from livekit.plugins import silero

    process.userdata["vad"] = silero.VAD.load(min_silence_duration=0.25)
    emit(
        "process_warm",
        components=warmed,
        failed=[{"module": name, "error": error} for name, error in failed],
    )


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
        stream = rtc.AudioStream(
            track,
            sample_rate=8_000,
            num_channels=1,
            frame_size_ms=20,
        )
        try:
            async for event in stream:
                recorder.append_pcm(
                    role, bytes(event.frame.data), received_at=time.monotonic()  # type: ignore[arg-type]
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


def write_session_state(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as recording:
        return recording.getnframes() / recording.getframerate()


server = AgentServer(
    num_idle_processes=resolve_idle_process_count(os.environ), setup_fnc=setup_process
)


@server.rtc_session(agent_name=AGENT_NAME)
async def production_agent(ctx: JobContext) -> None:
    # Room creation and SIP signaling are not authorization.  The worker joins
    # unsubscribed and cannot consume caller audio until the control plane has
    # resolved the number route and semantic config.
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
    participant = await ctx.wait_for_participant()
    attributes = dict(participant.attributes or {})
    caller = str(attributes.get("sip.phoneNumber") or attributes.get("sip.from") or "unknown")
    provider_call_id = str(attributes.get("sip.callID") or participant.identity)
    to_e164 = os.environ["SCH236_TEST_DID_E164"]
    adapter = build_adapter()
    call = await asyncio.to_thread(
        adapter.prepare_inbound,
        organization_id=os.environ["OPENPHONEX_ORGANIZATION_ID"],
        to_e164=to_e164,
        from_e164=caller,
        provider_call_id=provider_call_id,
    )
    profile = profile_from_realtime_config(call.config)
    call.admit_media()
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

    stack = build_demo_stack(
        profile,
        environ=os.environ,
        load_vad=lambda **_kwargs: ctx.proc.userdata["vad"],
    )
    tools = await asyncio.to_thread(build_livekit_model_tools, call)
    session = AgentSession(**stack.session_options)
    latency_recorder = TurnLatencyRecorder()
    finished = asyncio.Event()
    caller_turns = 0

    @session.on("user_state_changed")
    def on_user_state(event: Any) -> None:
        recorder.mark_state("caller", str(event.old_state), str(event.new_state))

    @session.on("agent_state_changed")
    def on_agent_state(event: Any) -> None:
        recorder.mark_state("agent", str(event.old_state), str(event.new_state))

    @session.on("user_input_transcribed")
    def on_user_transcript(event: Any) -> None:
        nonlocal caller_turns
        if event.is_final and str(event.transcript).strip():
            caller_turns += 1
            recorder.add_text("caller", str(event.transcript))

    @session.on("conversation_item_added")
    def on_conversation_item(event: Any) -> None:
        item = event.item
        role = str(getattr(item, "role", ""))
        text = str(getattr(item, "text_content", "") or getattr(item, "raw_text_content", ""))
        metrics = getattr(item, "metrics", None)
        if role == "user":
            latency_recorder.observe_user(text=text, metrics=metrics)
            return
        if role != "assistant" or caller_turns < 1:
            return
        if is_greeting_fragment(text, profile.first_message):
            return
        recorder.add_text("agent", text)
        turn_latency = latency_recorder.observe_assistant(
            text=text,
            metrics=metrics,
            semantic=True,
        )
        if turn_latency is not None:
            emit(
                "turn_latency",
                call_id=call.call_id,
                turn=turn_latency["turn"],
                breakdown=turn_latency["breakdown"],
            )

    @session.on("function_tools_executed")
    def on_tools(event: Any) -> None:
        emit(
            "model_tools_executed",
            call_id=call.call_id,
            tools=executed_tool_names(event),
        )

    @session.on("error")
    def on_error(event: Any) -> None:
        emit("session_error", call_id=call.call_id, error_type=type(event.error).__name__)

    def on_participant_disconnected(disconnected: rtc.RemoteParticipant) -> None:
        if disconnected.identity == participant.identity:
            finished.set()

    ctx.room.on("participant_disconnected", on_participant_disconnected)
    try:
        await session.start(
            room=ctx.room,
            agent=Agent(instructions=profile.instructions, tools=tools),
            room_options=RoomOptions(
                audio_input=True,
                audio_output=True,
                close_on_disconnect=False,
                delete_room_on_close=False,
            ),
        )
        await session.say(profile.first_message, allow_interruptions=True)
        emit(
            "media_admitted",
            call_id=call.call_id,
            language=profile.language,
            semantic_config_source="realtime-config",
        )
        await finished.wait()
    finally:
        ended_at = utcnow()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await session.aclose()
        await sidecar.disconnect()
        if capture_tasks:
            await asyncio.gather(*capture_tasks, return_exceptions=True)
        summary = recorder.finalize()
        latency_summary = latency_recorder.finalize(evidence_dir / "latency-stages.json")
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
        duration = wav_duration(evidence_dir / "mixed.wav")
        imported = await asyncio.to_thread(
            call.complete,
            source_path=evidence_dir,
            duration_sources={
                "sip_participant": duration,
                "room_track": duration,
                "recording": duration,
            },
        )
        call.cleanup(remove_participant=lambda: None, delete_room=lambda: None)
        write_session_state(
            state_path,
            call_id=call.call_id,
            room_name=ctx.room.name,
            started_at=started_at,
            last_heartbeat_at=ended_at,
            status="complete",
        )
        emit(
            "call_complete",
            call_id=call.call_id,
            semantic_turns=summary["semantic_turns"],
            latency_turns=latency_summary["turn_count"],
            artifact_status=str(imported.get("artifact_import", {}).get("status") or "imported"),
            billing_source="gateway_sip_lifecycle",
        )
        await ctx.delete_room()


if __name__ == "__main__":
    cli.run_app(server)

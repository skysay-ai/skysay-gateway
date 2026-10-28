# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import atexit
import contextlib
import functools
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from capabilities.opening import opening_policy
from capabilities.registry import load_registry as load_voice_registry

from livekit import api, rtc
from livekit.agents.types import NOT_GIVEN, APIConnectOptions
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    JobRequest,
    cli,
    function_tool,
)
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.room_io import RoomOptions, _ParticipantAudioOutput
from livekit.agents import llm as agents_llm
# Not re-exported from ``livekit.agents.utils``; the pinned runtime's own
# answering-machine detector imports it from this module too.
from livekit.agents.utils.participant import wait_for_agent, wait_for_participant_attribute
from livekit.agents.metrics import ModelUsageCollector

try:
    from gpt_live_runtime import (
        CallAgentRegistry,
        LiveEngineUnavailable,
        build_live_model,
        close_live_session_and_transport,
        gpt_live_session_instructions,
        greet_directive,
        await_duplex_session,
        is_gemini_live_config,
        is_eleven_native_config,
        supports_manual_live_replies,
        is_s2s_live_config,
        live_call_ending_policy,
        live_model_needs_http_session,
        live_duplex_session,
        notify_native_close_reclaimed,
        openai_http_session,
        NativeFarewellFallback,
        request_native_reply_recovery,
        voice_frontend,
    )
except ImportError:  # pragma: no cover
    from apps.voice.gpt_live_runtime import (  # type: ignore
        CallAgentRegistry,
        LiveEngineUnavailable,
        build_live_model,
        close_live_session_and_transport,
        gpt_live_session_instructions,
        greet_directive,
        await_duplex_session,
        is_gemini_live_config,
        is_eleven_native_config,
        supports_manual_live_replies,
        is_s2s_live_config,
        live_call_ending_policy,
        live_model_needs_http_session,
        live_duplex_session,
        notify_native_close_reclaimed,
        openai_http_session,
        NativeFarewellFallback,
        request_native_reply_recovery,
        voice_frontend,
    )

# Flat, like every other sibling below and unlike the gpt_live_runtime block
# above: a try/except ImportError around this would declare the module OPTIONAL,
# and the export's import-closure check reads that declaration literally. It is
# not optional -- this file calls into it at process setup.
from error_tracking import (
    capture_exception,
    clear_provider_close_in_progress,
    error_tracking_job_scope,
    install_error_tracking,
    mark_provider_close_in_progress,
    note_error_tracking_organization,
    provider_close_in_progress,
)

from control_plane import (
    BrowserSessionAlreadyOwned,
    ControlPlaneRejected,
    Gate2ControlPlaneAdapter,
    UrllibJsonTransport,
    adopt_deployment_content_storage,
    arms_late_caller_publications,
    call_credential_token,
    linked_participant_kinds,
    parse_outbound_dispatch,
    parse_workspace_browser_test_dispatch,
)
from workflow_turns import pinned_workflow_run_id
from workflow_root_plan import create_workflow_root_plan
from workflow_root import WorkflowRoot
from funded_runtime import FundedRuntime
from native_tts_usage import NativeTTSUsageError, settle_meters
from soniox_usage import SonioxUsageEvidenceError, close_soniox_stt_streams
from usage_report_outbox import UsageReportOutbox
from realtime_session_clock import report_session_time
from component_billing import (
    component_usage_report_required,
    MeteringContractError,
    SuccessfulToolUsageCounter,
    normalized_usage_from_model_usage,
    normalized_usage_from_profile_segments,
    snapshot_refused_metering,
    diagnostic_counter_snapshot,
    unmeasured_pinned_components,
)
from component_credentials import (
    runtime_environ_with_component_credentials,
    should_lease_components,
)
from demo import admit_voice_targets, build_demo_stack, frozen_far_end_listening
from prewarm import (
    PREWARM_PLUGIN_MODULES,
    SILERO_VAD_PINNED_PATH,
    SILERO_VAD_PINNED_SHA256,
    phone_capacity,
    prewarm_plugin_modules,
    resolve_drain_timeout_seconds,
    resolve_job_memory_thresholds,
    resolve_phone_agent_idle_processes,
    resolve_max_concurrent_jobs,
    resolve_worker_http_port,
)
from agents_readiness import (
    BEACON_INTERVAL_SECONDS,
    readiness_dir,
    remove_beacon,
    write_beacon,
)
from phone_admission import (
    EndedCalls,
    PhoneAdmission,
    active_job_rooms,
    drain_refusals,
    handle_dispatch,
    live_call_test_from,
    resolve_simulation_test_reserve,
    watch_call_end,
    with_capacity_refusal,
)
from livekit_evidence import (
    SynchronizedEvidenceRecorder,
    caller_transcript_timing,
    finalize_off_loop,
)
from evidence_sidecar import (
    evidence_sidecar_identity,
    safe_component,
    start_evidence_sidecar,
)
from audio_environment import (
    start_filler_background_player,
    AudioEnvironmentUnavailable,
    close_outgoing_audio_environment,
    start_outgoing_audio_environment,
)
from realtime_event_trace import RealtimeEventTrace, install_realtime_event_trace
from call_telemetry import CallTelemetryCollector
from caller_audio_flow import CallerAudioFlowLog, CallerVadEdgeLog
from turn_end_evidence import STATUS_FIELD as TURN_END_EVIDENCE_STATUS_FIELD
from turn_end_evidence import SUMMARY_EVENT as TURN_END_EVIDENCE_SUMMARY_EVENT
from turn_end_evidence import TurnEndEvidence, write_turn_end_evidence
from evidence_contract import (
    STT_CANDIDATE_LOCALES_REQUESTED,
    effective_voice_stack_identity,
    stt_candidate_locales_requested,
)
from graceful_stop import AlignmentFeed, GracefulStopStats
from graceful_stop_output import install_graceful_stop, observe_timed_transcript
from leading_silence import trim_leading_silence
from playout_hold import no_word_hold_budget_seconds
from playout_hold_output import guard_playout_hold
from browser_startup import BrowserStartup, publish_before_room_delete
from browser_startup_output import BrowserStartupOutput
from answer_classifier import WORD_SOURCE_LIVE_CAPTION, WORD_SOURCE_STT_FINAL
from answer_detection import (
    CLASSIFY_INTERVAL_SECONDS as ANSWER_CLASSIFY_INTERVAL_SECONDS,
    AnswerDetectionSession,
    frame_amplitude,
    resolved_mode as resolved_answer_detection_mode,
)
from voicemail_watch import (
    PROCESS_INTERVAL_SECONDS as VOICEMAIL_PROCESS_INTERVAL_SECONDS,
    resolved_setting as resolved_voicemail_mode,
)
from navigation_entry import (
    ENDER_ARBITER,
    ENDER_CLOSING_TEXT_FALLBACK,
    ENDER_END_CALL_TOOL,
    ENDER_MODEL_UNRESPONSIVE,
    ENDER_MUTUAL_SILENCE,
    SPEECH_ACKNOWLEDGEMENT,
    SPEECH_CHECK_IN,
    SPEECH_LANGUAGE_SWITCH,
    SPEECH_RECOVERY,
    STATE_SETTLING,
    NavigationSettings,
    Redactor,
    navigation_engine_refused,
    NavigationPolicy,
    observer_saw_a_person,
    follow_far_end,
    leave_episode,
    navigation_settings,
    scrub_registered_structure,
)
from sip_answer_gate import (
    DISCONNECTED as SIP_ANSWER_DISCONNECTED,
    TIMED_OUT as SIP_ANSWER_TIMED_OUT,
    NOT_APPLICABLE as SIP_ANSWER_NOT_APPLICABLE,
    NOT_REPORTED as SIP_ANSWER_NOT_REPORTED,
    SIP_CALL_STATUS_ACTIVE,
    SIP_CALL_STATUS_ATTRIBUTE,
    await_callee_answer,
    clock_anchor as answer_clock_anchor,
    has_answer_evidence,
    may_speak,
    ringing_timeout_seconds,
)
from opening_turn import (
    EXCLUDED_NO_VAD as OPENING_EXCLUDED_NO_VAD,
    FIRST_REPLY_GENERATED,
    FIRST_REPLY_LITERAL,
    FIRST_REPLY_LITERAL_FAST,
    FIRST_REPLY_PICKUP,
    is_greeting_only,
    OPEN as OPENING_TURN_OPEN,
    ONSET as OPENING_TURN_ONSET,
    OPENING as OPENING_TURN_OPENING,
    RESUMED as OPENING_TURN_RESUMED,
    POLICY_AGENT_FIRST as OPENING_POLICY_AGENT_FIRST,
    POLICY_CALLEE_FIRST as OPENING_POLICY_CALLEE_FIRST,
    SUMMARY_EVENT as OPENING_SUMMARY_EVENT,
    VAD_SPEECH_PROBABILITY,
    OpeningTurnTaking,
    OpeningObservationTimeout,
    VadFrameAligner,
    callee_first_enabled,
    callee_first_exclusion,
    frame_level_dbfs,
    pickup_reply_instruction,
)
from interruption_boundary import DEFAULT_BOUNDARY_TAIL_SECONDS, BoundaryStopPolicy
from acknowledgements import (
    AcknowledgementPolicy,
    AcknowledgementScheduler,
    acknowledgement_settings,
    background_speaker,
    synthesize_acknowledgement_clips,
)
from sip_call_statistics import capture_directory as sip_statistics_capture_directory
from tool_wait import ToolWaitFeedback, tool_wait_settings, typing_sound
import worker_events
from transcript_events import (
    TOPIC as TRANSCRIPT_EVENTS_TOPIC,
    TranscriptEventPublisher,
    install_transcript_events,
)
from closing_evidence import ClosingEvidence
from call_transcript_events import (
    CallTranscriptEventRecorder,
    frozen_allowed_languages as transcript_events_allowed_languages,
    install_call_transcript_events,
    section_engine as transcript_events_engine,
)
from production import (
    require_pre_media_participant,
    settle_pre_media_failure,
    report_call_answered,
    watch_outbound_answer,
    utcnow,
    require_inbound_sip_answer,
    admitted_tts_text,
    bound_output_transcript,
    await_bounded_teardown_task,
    _report_transfer_outcome,
    close_production_session_after_room_end,
    ProductionSessionCloseResult,
    CallerTranscriptGate,
    CallerTranscriptGateStream,
    InterruptionTelemetry,
    InterruptionWordGate,
    OpeningTurnProtection,
    SemanticTurnAdmission,
    SemanticTurnEventRecorder,
    WorkerLocalEndCall,
    apply_end_call_post_refusal,
    post_refusal_hooks,
    voicemail_stands,
    TERMINAL_EGRESS_OBSERVER_STALE_SECONDS,
    drain_terminal_egress,
    settle_end_call_guard_summary,
    settle_worker_terminal_end_call,
    shutdown_evidence_sidecar,
    PeriodicWorkerHeartbeat,
    BROWSER_TEST_REPORT_BUDGET_SECONDS,
    browser_test_close_component,
    livekit_error_component,
    browser_test_error_metadata,
    build_livekit_model_tools,
    caller_disconnect_handler,
    caller_transcripts_until_terminal,
    cancel_stray_model_generation,
    component_model_labels,
    engine_has_language_axis,
    engine_reports_caller_turns,
    configured_interruption_min_duration,
    turn_end_settings_evidence,
    caller_backchannel_tokens,
    is_caller_backchannel_only,
    configured_min_interruption_words,
    effective_min_interruption_words,
    installed_min_interruption_words,
    executed_tool_names,
    finalize_call_artifacts,
    finalize_through_cancellation,
    inbound_sip_identity,
    is_managed_pstn_call,
    post_livekit_worker_heartbeat,
    pre_media_lifecycle_reason,
    safe_pre_media_reason,
    ProductionConfigurationError,
    profile_from_realtime_config,
    production_room_disconnect_handler,
    production_session_close_handler,
    report_call_teardown_lifecycle,
    REPLY_IN_FLIGHT_AGENT_STATES,
    adopt_hangup_intent,
    CallEndSnapshot,
    CallerAudioBinding,
    CallerAudioBindWitness,
    CallerSpeechWitness,
    caller_audio_binding_from_room_io,
    wire_caller_audio_subscription,
    SessionEndArbiter,
    counting_caller_audio,
    DETECTED_VIA_AGENT_SCRIPT,
    DETECTED_VIA_ROOM_LOST,
    DETECTED_VIA_SESSION_CLOSED,
    DETECTED_VIA_DISCONNECT_EVENT,
    DETECTED_VIA_HANGUP_INTENT,
    DETECTED_VIA_WORKER_TIMER,
    pending_hangup_intent,
    resolved_max_call_seconds,
    resolved_no_caller_speech_seconds,
    build_call_floor_policy,
    server_side_remaining_parties,
    drop_carrier_leg_then_continue,
    teardown_room_and_acknowledge_hangup,
    TERMINATION_AGENT_COMPLETED,
    TERMINATION_NO_HUMAN_AFTER_ANSWER,
    TERMINATION_MEDIA_SESSION_LOST,
    TERMINATION_MODEL_SESSION_LOST,
    DETAIL_TRANSCRIBER_REFUSED_REQUEST,
    TERMINATION_AGENT_UNRESPONSIVE,
    TERMINATION_DETAIL_MODEL_UNRESPONSIVE,
    TERMINATION_CALLER_UNRESPONSIVE,
    TERMINATION_HANGUP_REQUESTED,
    TERMINATION_REMOTE_PARTY_DISCONNECTED,
    watch_for_worker_call_end,
    requires_soniox_usage_correlation,
    stt_registry_reports_confidence,
    build_call_tts_component,
    restore_speech_interruptions,
    safe_component_error_diagnostics,
    managed_pstn_min_speech_clamp,
    session_options_with_managed_pstn_turn_taking,
    service_heartbeat_environ,
    end_call_fallback_message,
    await_playout_or_session_end,
    run_until_session_end,
    speak_opening_line,
    terminate_pre_media_call,
    worker_heartbeats_enabled,
    write_session_state,
)
from spoken_language import generated_opening_instruction, language_name, pickup_answer_words
from runtime_language_switch import (
    PROVIDER_LANGUAGE_EVIDENCE,
    ControlledLanguageSwitcher,
    FollowCallerProfiles,
    LanguageSwitchError,
    SegmentNativeUsageUnavailable,
    SWITCH_LANGUAGE_TOOL_NAMES,
    switch_language_tool_schema,
)
from transcription_stall import TranscriptionStallRecovery
from unrecognised_speech import (
    UnrecognisedSpeechReply,
    repair_instruction,
    unrecognised_speech_settings,
)
from model_unresponsive import (
    DECISION_EVENT as MODEL_UNRESPONSIVE_DECISION_EVENT,
    ModelUnresponsiveRule,
    OUTPUT_REPLY,
    OUTPUT_TOOL,
    agent_audio_observed as agent_audio_observed_recently,
    model_unresponsive_seconds,
)
from duplex_liveness import DuplexLivenessWitness
from transcript_coverage import coverage_threshold, report_agent_speech_coverage
from mutual_silence import HOLD_ENTRY_WINDOW, HOLD_NAVIGATION, MutualSilenceRule, resolve_mode as mutual_silence_mode
from end_call_guard import (
    MODE_OFF as GUARD_MODE_OFF,
    SUMMARY_EVENT as GUARD_SUMMARY_EVENT,
    CallerEvidence,
    EndCallGuard,
    LocalCloseRefusalRecovery,
    TerminalIntent,
    Exchange as GuardExchange,
    Snapshot as GuardSnapshot,
    caller_busy as guard_caller_busy,
    cut_replies_of_last,
    effective_guard_mode,
    exchange_from_history,
    resolve_budget as resolve_guard_budget,
    resolve_mode as resolve_guard_mode,
)
from closing_classifier import build_platform_classifier
from navigation_entry import FarEndWatch, build_far_end_classifier
from navigation_entry import (
    ENGINE_CASCADE,
    ENGINE_LIVE,
    EVENT_INTERVALS,
    GptLiveInput,
    NavDecider,
    NavigationActions,
    NavigationTimeline,
    NavigationToolGate,
    SilencedInput,
    WorkflowFinalGate,
    WorkflowNavigator,
    WorkflowSettling,
    build_navigation_actions,
    build_perception,
    cascade_model_feed,
    cascade_value_covers,
    cascade_voice,
    fed_prompt,
    flush_controller_track,
    greeting_site,
    hand_native_opening_to_navigation,
    keypad_fields,
    line_is_a_menu,
    line_is_holding_music,
    listeners_of,
    navigation_base_language,
    navigation_intervals,
    navigation_menu_language_trusted,
    navigation_plan_sources,
    navigation_untrusted_sources,
    sip_identities_of,
    start_controller_track,
    voice_says_held_value,
    with_a_user_turn,
    without_preemptive_generation,
    yield_to_far_end,
)
from duplex_opening_hold import DuplexOpeningHold
from live_opening import (capture_pickup, NativeOpeningAuthorization, finish_native_pickup_reply,
                          finish_native_silence_opening)
from native_caller_floor import install_native_caller_floor
from check_in import SILENCE_FAREWELL_INSTRUCTION, speak_check_in


def _validate_follow_caller_billing_profile(language: str, config: dict[str, Any], billing: Any) -> None:
    """Bind a credential-free frozen worker profile to its priced identity.

    The worker must never receive credential ids, but it can still prove the
    selected provider/model tuple is one the immutable reservation admitted.
    ``credential_source`` is deliberately required on the reservation snapshot
    so the control plane, not a mutable worker profile, owns BYOK authority.
    """
    segments = billing.get("profile_segments") if isinstance(billing, dict) else None
    expected = segments.get(language) if isinstance(segments, dict) else None
    components = expected.get("components") if isinstance(expected, dict) else None
    if not isinstance(components, list):
        raise LanguageSwitchError("follow_caller profile lacks frozen billing segment")
    expected_identities: dict[str, tuple[str, str]] = {}
    for component in components:
        if not isinstance(component, dict):
            raise LanguageSwitchError("follow_caller billing component is invalid")
        kind = str(component.get("component") or "")
        provider = str(component.get("provider") or "").strip().lower()
        model = str(component.get("model") or "").strip()
        if kind not in {"stt", "llm", "tts"} or not provider or not model or not str(component.get("credential_source") or "").strip():
            raise LanguageSwitchError("follow_caller billing component is incomplete")
        expected_identities[kind] = (provider, model)
    if set(expected_identities) != {"stt", "llm", "tts"}:
        raise LanguageSwitchError("follow_caller billing component set is incomplete")
    actual = {
        "stt": (str((config.get("transcriber") or {}).get("provider") or (config.get("transcriber") or {}).get("registry_name") or "").lower(), str((config.get("transcriber") or {}).get("model") or "")),
        "llm": (str((config.get("model") or {}).get("provider") or (config.get("model") or {}).get("registry_name") or "").lower(), str((config.get("model") or {}).get("model") or "")),
        "tts": (str((config.get("voice") or {}).get("provider") or (config.get("voice") or {}).get("registry_name") or "").lower(), str((config.get("voice") or {}).get("model") or "")),
    }
    if actual != expected_identities:
        raise LanguageSwitchError("follow_caller profile does not match frozen billing identity")
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


#: The control plane's refusal code for the reserved release loopback while
#: no release claim is open (``ReleaseCanaryNotArmedError`` in the API): the
#: designed outcome of a leg outside a release window, not a worker fault.
#: The control plane's refusal code while an operator's maintenance window is
#: open (``PlatformMaintenanceError`` in the API): a designed refusal, not a
#: worker fault.
AGENT_NAME = "skysay-voice-c3"
SESSION_ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")
EVIDENCE_ROOT = Path(os.environ.get("SCH236_CALL_EVIDENCE_ROOT") or "/call-evidence")
_service_heartbeat: PeriodicWorkerHeartbeat | None = None
_service_heartbeat_lock = threading.Lock()


def emit(event: str, **details: Any) -> None:
    # A keypad value this process holds is masked in anything it logs.
    # The call id routes the evidence journal, so it is left as it is: it names
    # the call and never carries a value, and masking a coincidence in it would
    # silently lose the call's later events.
    details = {**scrub_registered_structure(details), **({"call_id": details["call_id"]} if "call_id" in details else {})}
    print(
        "SCH236_PRODUCTION "
        + json.dumps({"event": event, **details}, sort_keys=True, allow_nan=False),
        flush=True,
    )
    # And into the call's own evidence directory, which outlives this
    # container (worker_events.py). A no-op for an event without an open call.
    worker_events.record(event, details)





#: The control plane's reason code for component usage that arrived after the
#: call's billing was final (``UsageAfterSettlementError``).
USAGE_AFTER_SETTLEMENT_REASON = "component_usage_after_settlement"


async def persist_refused_metering(
    *,
    call: Any,
    reason: str,
    model_usage_records: list[Any],
    stack: Any,
    soniox_usage: Any,
) -> asyncio.CancelledError | None:
    """Keep the counts a report that could not be billed still held.

    Diagnostic only, and never allowed to fail the teardown. Returns the
    cancellation it absorbed, if any, for the caller to surface once the
    teardown has finished.
    """
    try:
        evidence = snapshot_refused_metering(
            reason,
            model_usage_records,
            tts_native_usage=diagnostic_counter_snapshot(stack.tts_unit_meter),
            soniox_usage=soniox_usage,
            native_diagnostics=(
                stack.tts_unit_meter.diagnostic()
                if callable(getattr(stack.tts_unit_meter, "diagnostic", None))
                else None
            ),
        )
        result = await asyncio.to_thread(call.report_refused_metering, evidence)
        if isinstance(result, dict) and result.get("session_duration_dropped"):
            # The API predates the field; the rest of the report was kept.
            emit("refused_metering_session_duration_dropped", call_id=call.call_id)
    except asyncio.CancelledError as persist_exc:
        emit("refused_metering_persist_failed", call_id=call.call_id, error_type="CancelledError")
        return persist_exc
    except Exception as persist_exc:
        emit(
            "refused_metering_persist_failed",
            call_id=call.call_id,
            error_type=type(persist_exc).__name__,
        )
    return None


def _transport(token_variable: str) -> UrllibJsonTransport:
    return UrllibJsonTransport(
        base_url=os.environ["SKYSAY_CONTROL_PLANE_URL"],
        bearer_token=os.environ[token_variable],
    )


def _call_transport() -> UrllibJsonTransport:
    """Call-bound transport: the org worker key if present, else the gateway token."""
    return UrllibJsonTransport(
        base_url=os.environ["SKYSAY_CONTROL_PLANE_URL"],
        bearer_token=call_credential_token(os.environ),
    )


def build_adapter() -> Gate2ControlPlaneAdapter:
    platform_token = os.environ.get("SKYSAY_PLATFORM_MEDIA_TOKEN", "").strip()
    call_transport = _call_transport()
    return Gate2ControlPlaneAdapter(
        gateway_id=os.environ["SKYSAY_GATEWAY_ID"],
        organization_id=os.environ.get("SKYSAY_ORGANIZATION_ID", ""),
        transport=_transport("SKYSAY_GATEWAY_API_TOKEN"),
        call_transport=call_transport,
        tool_transport=call_transport,
        platform_transport=(
            _transport("SKYSAY_PLATFORM_MEDIA_TOKEN") if platform_token else None
        ),
    )


def setup_process(process: JobProcess) -> None:
    # Crash reporting FIRST, and here rather than only in __main__. On Linux
    # the agent framework starts job processes through a forkserver, so this
    # process inherited nothing from the entrypoint's __main__ block -- and
    # this is the process a call actually runs in, where an unhandled
    # exception leaves a live caller listening to silence while the container
    # stays up and registered. Never fatal: a dead backend returns False.
    install_error_tracking(component="voice-worker", report=emit)

    # Best-effort pre-import of provider plugin modules so warm
    # processes do not pay the import cost on their first job.  A missing or
    # broken optional plugin must not crash the warm process, but its failure
    # is reported (not silently swallowed) via the process_warm event.
    warmed, failed = prewarm_plugin_modules(PREWARM_PLUGIN_MODULES)

    from livekit.plugins import silero

    process.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.25, onnx_file_path=SILERO_VAD_PINNED_PATH
    )
    _start_process_heartbeat(process)
    emit(
        "process_warm",
        components=warmed,
        failed=[{"module": name, "error": error} for name, error in failed],
    )


def _heartbeat_interval_seconds() -> float:
    try:
        return max(10.0, float(os.environ.get("SKYSAY_WORKER_HEARTBEAT_SECONDS", "30")))
    except ValueError:
        return 30.0


def _new_worker_heartbeat(*, heartbeat_environ: dict[str, str]) -> PeriodicWorkerHeartbeat | None:
    if not worker_heartbeats_enabled(os.environ):
        emit("worker_heartbeat_disabled", reason="disabled_by_environment")
        return None
    base_url = str(os.environ.get("SKYSAY_CONTROL_PLANE_URL") or "").strip()
    worker_token = str(os.environ.get("SKYSAY_WORKER_API_TOKEN") or "").strip()
    if not base_url or not worker_token:
        emit(
            "worker_heartbeat_disabled",
            missing=[
                name
                for name, value in (
                    ("SKYSAY_CONTROL_PLANE_URL", base_url),
                    ("SKYSAY_WORKER_API_TOKEN", worker_token),
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


def _quietly(emit: Callable[..., None], event: str, **details: Any) -> None:
    """Emit, or carry on.  For the two sites where evidence precedes a call."""

    try:
        emit(event, **details)
    except Exception:  # noqa: BLE001 - a note about a call cannot end one
        pass


def _trimmed_ms(trimmed: list[float]) -> int | None:
    """The opening's trimmed front in ms; ``None`` when nothing was measured."""

    return round(sum(trimmed) * 1000) if trimmed else None


async def _first_of(*events: asyncio.Event, timeout: float | None = None) -> None:
    """Return when any of ``events`` is set, or after ``timeout`` seconds."""

    waiters = [asyncio.create_task(event.wait()) for event in events]
    try:
        await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            if not waiter.done():
                waiter.cancel()


#: How long a decided opening waits for the pickup's turn to commit, so the
#: reply is generated with it in context. Past it the reply quotes the pickup's
#: transcript instead (``pickup_source``); a longer wait only moved the
#: slowest commits later still (3.5 s after the callee stopped, host). A turn
#: that commits afterwards cannot cut the opening: the transcript gate holds
#: caller transcripts from dispatch through the opening's playout.
PICKUP_COMMIT_WAIT_SECONDS = 0.7
#: The runtime waits this long for a final transcript when the pickup's turn is
#: committed on demand (reply-less, in the background) and its last final is
#: older than half a second.
PICKUP_COMMIT_TRANSCRIPT_TIMEOUT_SECONDS = 1.0
#: The most the opening waits for the runtime to take the sealing reply-less
#: commit (it waits for no transcript, so this is a bound on a stuck runtime).
PICKUP_SEAL_TIMEOUT_SECONDS = 0.3


def signal_opening_action(
    action: str,
    *,
    released: asyncio.Event,
    resumed: asyncio.Event,
    pickup_committed: asyncio.Event,
) -> None:
    """Turn one decider action into the events the opening block waits on."""

    if action == OPENING_TURN_OPEN:
        # A resumption before this OPEN is spent: left set, it would cut the
        # new OPEN's wait for the pickup's transcript short.
        resumed.clear()
        released.set()
    elif action == OPENING_TURN_RESUMED:
        # Whatever committed so far was not the whole turn.
        pickup_committed.clear()
        resumed.set()
    elif action == OPENING_TURN_ONSET:
        pickup_committed.clear()


def pickup_source(*, committed: bool, finals: list[str]) -> tuple[str, str]:
    """How the first reply knows what the callee said, and the quote if any.

    ``committed``: the pickup is the newest turn in context. Otherwise its
    turn is still being closed (the host saw one commit more than 2 s late),
    and its final transcripts so far stand in for it -- ``transcript`` -- so
    its content is still answered. ``none``: nothing was transcribed.
    """

    if committed:
        return "committed", ""
    heard = " ".join(str(part).strip() for part in finals if str(part).strip())
    return ("transcript", heard) if heard else ("none", "")


async def wait_for_opening_release(
    decider: OpeningTurnTaking,
    *,
    released: asyncio.Event,
    resumed: asyncio.Event,
    pickup_committed: asyncio.Event,
    finished: asyncio.Event,
    hold_pickup: Callable[[], None],
    commit_pickup: Callable[[], None] = lambda: None,
    pickup_wait_seconds: float = PICKUP_COMMIT_WAIT_SECONDS,
    hold_for_menu: Callable[[], Awaitable[bool]] | None = None,
    hold_for_music: Callable[[], Awaitable[bool]] | None = None,
) -> bool:
    """Wait until the callee-first opening may be dispatched; ``False`` if never.

    ``released`` is set by the decider's tasks on OPEN (and on any way out);
    ``resumed`` when the callee took the floor back before dispatch. When the
    callee spoke, the pickup's own transcript may still be in flight: the
    reply waits a bounded moment for it (``hold_pickup`` keeps that turn out
    of the ordinary answer path meanwhile). A callee who carries on in that
    moment -- or at any point before this returns -- is waited out again.

    ``commit_pickup`` closes the pickup's turn NOW, without a reply. Left to
    the runtime, a turn that commits after the opening was dispatched cancels
    the opening on its way in -- the runtime interrupts the current speech for
    every committed turn before any hook can refuse the reply -- and the
    callee hears nothing at all (seen on the host).
    """

    while not finished.is_set():
        await _first_of(released, finished)
        if decider.state == OPENING_TURN_OPENING and decider.callee_spoke and not pickup_committed.is_set():
            hold_pickup()
            commit_pickup()
            await _first_of(pickup_committed, resumed, finished, timeout=pickup_wait_seconds)
        if decider.owns_floor and not finished.is_set():
            released.clear()
            resumed.clear()
            continue
        dispatchable = decider.state == OPENING_TURN_OPENING and decider.reason is not None and not finished.is_set()
        # The callee's first words may be a phone menu's, and a line with no callee may be playing music (a queue): the
        # opening is spoken into neither.
        hold = hold_for_menu if decider.callee_spoke else hold_for_music
        if dispatchable and hold is not None:
            if await hold():
                decider.note_machine_verdict("navigation_menu", acting=True, at=time.monotonic())
                return False
            if decider.owns_floor and not finished.is_set():
                # The callee carried on while the opening was held: wait for the end of their turn again.
                released.clear()
                resumed.clear()
                continue
            dispatchable = decider.state == OPENING_TURN_OPENING and not finished.is_set()
        return dispatchable
    return False


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


class EvidenceAgent(Agent):
    """Bind assistant text before LiveKit starts provider synthesis.

    ``conversation_item_added`` remains useful reconciliation evidence, but
    LiveKit 1.7.1 can omit it when a caller disconnects during a reply.  The
    TTS node is the earliest authoritative point at which response text is
    handed to playback.  The room sidecar remains the authority for whether
    audio was actually observed; this wrapper never creates a transcript turn
    on its own.
    """

    def __init__(
        self,
        *,
        recorder: SynchronizedEvidenceRecorder,
        admission: SemanticTurnAdmission,
        transcript_gate: CallerTranscriptGateStream | None = None,
        interruption_telemetry: InterruptionTelemetry | None = None,
        provider_language_evidence: Any | None = None,
        alignment_feed: AlignmentFeed | None = None,
        speech_to_speech: bool = False,
        end_call_evidence: Any | None = None,
        opening_owns_turn: Callable[[], bool] | None = None,
        on_opening_pickup: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._speech_to_speech = speech_to_speech
        #: True while the outbound opening decider owns the first response: a
        #: caller turn committed then is the pickup, kept for the opening to
        #: answer and never answered on its own.
        self._opening_owns_turn = opening_owns_turn
        self._on_opening_pickup = on_opening_pickup
        self._evidence_recorder = recorder
        self._semantic_admission = admission
        self._transcript_gate = transcript_gate
        #: The end_call guard's caller evidence: the committed turn's text is
        #: bound to the finals that made it in ``on_user_turn_completed``.
        self._end_call_evidence = end_call_evidence
        self._interruption_telemetry = interruption_telemetry
        self._provider_language_evidence = provider_language_evidence
        self._alignment_feed = alignment_feed
        self._caller_audio_bind_witness: CallerAudioBindWitness | None = None
        self._caller_audio_flow: CallerAudioFlowLog | None = None
        self._turn_end_evidence: TurnEndEvidence | None = None
        #: Set after construction when the call navigates: the door to the model's context
        #: (``navigation_chat.ModelContextGate``). ``None`` leaves every turn as it was.
        self._model_context: Any | None = None
        self._evidence_completion_hooks: set[str] = set()
        self._interruption_completion_hooks: set[str] = set()
        #: The session's first synthesis is the opening; see ``_synthesize``.
        self._opening_synthesized = False
        self._opening_silence_trimmed: list[float] | None = None

    def record_opening_silence_trim(self, trimmed: list[float]) -> None:
        """Where the seconds trimmed from the opening's front are appended.

        Set after construction, like the caller-audio witness: the workflow
        agent is built on another branch.
        """

        self._opening_silence_trimmed = trimmed

    def bind_model_context(self, gate: Any | None, on_generation: Any | None = None, on_tool_call: Any | None = None) -> None:
        """Route what the far end says through the value-mode redactor before the model sees it.

        Set after construction, like the caller-audio witness. ``on_generation`` is told when each
        reply of the model begins (the navigation tools bind a request to the line as it stood then).
        """

        self._model_context = gate
        self._on_generation = on_generation
        self._on_tool_call = on_tool_call

    def bind_turn_end_evidence(self, evidence: TurnEndEvidence | None) -> None:
        """Record each caller turn the SDK hands to this agent, answered or refused.

        Set after construction, like the caller-audio witness.  The commit is
        observed in ``on_user_turn_completed`` rather than at
        ``conversation_item_added`` because a turn this agent refuses to
        answer never becomes a conversation item.  A turn the SDK commits
        without calling that hook is not recorded; see ``TurnEndEvidence``.
        """

        self._turn_end_evidence = evidence

    def bind_caller_audio_witness(
        self,
        witness: CallerAudioBindWitness,
        flow: CallerAudioFlowLog | None = None,
    ) -> None:
        """Count caller frames entering this agent's STT seam.

        Two recorders off one pass: ``witness`` reports the binding once, and
        ``flow`` keeps reporting for the rest of the call.  See the
        ``caller_audio_flow`` module for why a once-only row cannot say
        whether caller audio was still arriving later in the call.

        Set after construction: the workflow agent is built on another branch.
        """

        self._caller_audio_bind_witness = witness
        self._caller_audio_flow = flow

    def _evidence_speech_handle(self) -> Any | None:
        try:
            return self.session.current_speech
        except Exception:
            return None

    async def llm_node(self, chat_ctx: Any, tools: Any, model_settings: Any):
        # Read by name: the SDK proofs run this node on a bare ``Agent`` that has no such attribute.
        model_context = getattr(self, "_model_context", None)
        if model_context is not None and model_context.holding:
            # Waiting in a queue: the model never answers what the line says. The turn hook
            # drops the item; this stops a preemptive generation the SDK started on the
            # transcript before the hook ran (its reply would be silenced anyway, but it
            # would be billed and could reach a tool).
            return
        on_generation = getattr(self, "_on_generation", None)
        heard = (
            on_generation(fed_prompt(chat_ctx), self._evidence_speech_handle()) if on_generation is not None else None
        )
        on_tool_call = getattr(self, "_on_tool_call", None)
        if model_context is not None:
            # A user turn created while a keypad value is in play is not shown to the model,
            # whichever SDK path appended it (the turn hook is not the only one).
            chat_ctx = model_context.filtered_context(chat_ctx)
            # (and a history of system notes and tool calls with no user turn is one Gemini refuses: see the function)
            chat_ctx = with_a_user_turn(
                chat_ctx, lambda text: agents_llm.ChatMessage(role="user", content=[text])
            )
        # Capture at inference, including preemptive inference, not at playout
        # or tool completion: another caller may already have taken the floor
        # by then. The SDK retains this context's original user ID when it
        # schedules a preemptive reply and carries delta.extra onto its final
        # assistant item. Native engines and governed WorkflowRoot do not use this
        # node. The marker contains no text and produces no audible chunk.
        user_id = next((str(getattr(item, "id", "") or "")
                        for item in reversed(chat_ctx.items)
                        if getattr(item, "role", "") == "user"), "")
        yield agents_llm.ChatChunk(id="skysay-telemetry", delta=agents_llm.ChoiceDelta(
            extra={CallTelemetryCollector.REPLY_USER_ITEM_KEY: user_id}))
        async with contextlib.aclosing(Agent.default.llm_node(self, chat_ctx, tools, model_settings)) as stream:
            async for chunk in stream:
                if isinstance(chunk, agents_llm.ChatChunk) and chunk.delta is not None:
                    # Tools run before their assistant item is committed. Carry
                    # the same input identity on the tool's own SDK extra too.
                    delta = chunk.delta
                    calls = [tool.model_copy(update={"extra": {
                        **(tool.extra or {}), CallTelemetryCollector.REPLY_USER_ITEM_KEY: user_id}})
                        for tool in (delta.tool_calls or [])]
                    chunk = chunk.model_copy(update={"delta": delta.model_copy(update={
                        "extra": {**(delta.extra or {}), CallTelemetryCollector.REPLY_USER_ITEM_KEY: user_id},
                        "tool_calls": calls or delta.tool_calls})})
                    if on_tool_call is not None and heard is not None:
                        # The tool runs later, perhaps after a newer reply began: it answers the line THIS
                        # reply heard, which is tied to its call id here.
                        for tool in (delta.tool_calls or []):
                            on_tool_call(getattr(tool, "call_id", None), heard)
                yield chunk

    def stt_node(self, audio: Any, model_settings: Any) -> Any:
        """Route caller transcripts through the managed-PSTN interruption gate.

        ``Agent.stt_node`` is LiveKit's documented STT override point and the
        only producer of the events that reach the SDK's ungated
        transcript-interruption hooks, so it is where a duration-consistent
        barge-in contract has to be enforced -- see the seam trace in
        ``production.CallerTranscriptGate``.  Without a gate (browser tests,
        non-English, the release canary) this is the stock node, unchanged.

        Overriding this method also opts the session out of the SDK's STT
        pipeline reuse across agent handoff (``agent_activity.py:763-771``).
        The production worker starts exactly one agent per job and never hands
        off, so nothing is reused today either way.
        """

        # Frames that reached this SESSION, not frames the room received.
        witness = self._caller_audio_bind_witness
        if witness is not None:
            audio = counting_caller_audio(audio, witness, self._caller_audio_flow)
        node = Agent.default.stt_node(self, audio, model_settings)
        gate = self._transcript_gate
        if gate is not None:
            node = gate.filter(node)
        if self._provider_language_evidence is not None:
            node = self._provider_language_evidence.filter(node)
        # Outermost on purpose: once end_call has closed admission, nothing
        # released by the word gate above may reach the SDK either, or the
        # caller's answer to the goodbye starts a model turn into an ended call.
        return caller_transcripts_until_terminal(node, admission=self._semantic_admission)

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        """Refuse the reply to any caller turn committed after end_call.

        ``stt_node`` above stops caller transcripts at the source once the
        end-call action closes admission. One transcript can still be in
        flight past that point: yielded just before the close and sitting in
        the SDK's own STT event channel, one event-loop hop from being
        committed as a turn. The SDK's documented way to skip the reply to a
        committed turn is to raise ``StopResponse`` here, so a turn that slips
        through that hop is dropped instead of generating into an ended call.
        The import is deferred because the SDK is installed only where this
        hook can run.
        """

        turn_end_evidence = self._turn_end_evidence
        if turn_end_evidence is not None:
            # Before either refusal below, so a refused turn is still a
            # recorded commit. Inert by its own contract: it cannot raise.
            turn_end_evidence.note_turn_committed(new_message)
        metrics = getattr(new_message, "metrics", None)
        said_at = metrics.get("started_speaking_at") if isinstance(metrics, dict) else None
        said_until = metrics.get("stopped_speaking_at") if isinstance(metrics, dict) else None
        if not self._semantic_admission.admits_transcript() or (
            # While a keypad value is in play the model hears nothing of the far end: the SDK appends
            # a user message only AFTER this hook returns, and drops it on ``StopResponse``, so the
            # turn never enters the chat context and no reply is generated for it. Asked only for a
            # call the admission still serves (the order is the first refusal's).
            self._model_context is not None
            and not self._model_context.admit_message(new_message, started_at=said_at, stopped_at=said_until)
        ):
            from livekit.agents.llm import StopResponse

            if turn_end_evidence is not None:
                turn_end_evidence.note_reply_refused()
            raise StopResponse()
        if self._end_call_evidence is not None:
            # The frozen turn text, before the SDK inserts the item: bind it to
            # the forwarded finals (consume-once) so the guard knows which
            # caller evidence the model's reply consumed and which came after.
            try:
                self._end_call_evidence.bind_turn_text(
                    str(getattr(new_message, "text_content", None) or "")
                )
            except Exception:  # noqa: BLE001 - a diagnostic feed cannot decide media
                pass
        owns = self._opening_owns_turn
        if owns is not None and owns():
            # The callee's pickup on an outbound call. It belongs in the
            # conversation -- the opening is generated with it in context --
            # but it is not a turn to reply to: the opening decider owns the
            # first response. StopResponse alone would drop the item, so it
            # is committed here the way the SDK commits a turn it skips.
            try:
                self._chat_ctx.items.append(new_message)
                self.session._conversation_item_added(new_message)
            except Exception:  # noqa: BLE001 - losing context is better than a second reply
                pass
            if self._on_opening_pickup is not None:
                try:
                    self._on_opening_pickup()
                except Exception:  # noqa: BLE001
                    pass
            from livekit.agents.llm import StopResponse

            if turn_end_evidence is not None:
                turn_end_evidence.note_reply_refused()
            raise StopResponse()
        # The pinned SDK's base hook is an explicit no-op (``agent.py``,
        # ``on_user_turn_completed: pass``); a live turn needs nothing from it.

    def observe_interruption_completion(self, handle: Any) -> None:
        """Attach the reply interrupted/completed observation exactly once.

        Called from ``speech_created`` -- not only from ``tts_node`` -- because
        a reply cancelled while the agent is still ``thinking`` never reaches
        TTS at all, and those are precisely the interruptions this telemetry
        exists to count. Registering here and again from the TTS path is safe:
        the speech id is recorded, so a handle is only ever hooked once.

        Separate from the evidence hook below because the two answer different
        questions and have different admission rules: evidence describes the
        customer's transcript, this one describes whether the reply survived
        to the end of its playout.
        """

        telemetry = self._interruption_telemetry
        speech_id = str(getattr(handle, "id", "") or "").strip()
        if telemetry is None or not speech_id or speech_id in self._interruption_completion_hooks:
            return
        if not bool(getattr(handle, "allow_interruptions", True)):
            # An uninterruptible reply -- the end_call farewell -- can never
            # enter the interrupted numerator, so counting it as completed
            # playout would inflate the denominator of the ratio this
            # telemetry exists to compare.
            return
        self._interruption_completion_hooks.add(speech_id)

        def finished(completed_handle: Any) -> None:
            try:
                failure = completed_handle.exception()
            except Exception:
                # A handle that cannot report an outcome is not evidence of a
                # failure; treat it as the SDK's plain interrupted/completed.
                failure = None
            telemetry.on_agent_speech_finished(
                interrupted=bool(getattr(completed_handle, "interrupted", False)),
                failed=failure is not None,
                speech_id=str(getattr(completed_handle, "id", "") or ""),
            )

        try:
            handle.add_done_callback(finished)
        except Exception:
            # Observability must never decide whether media runs.
            self._interruption_completion_hooks.discard(speech_id)

    def _aligned(self, stream: Any) -> Any:
        """Record the word timings the TTS declares, and pass the audio on.

        The frames crossing this node carry the provider's own per-word
        timings when it declares ``capabilities.aligned_transcript``
        (ElevenLabs, Cartesia); reading them here lets a stop land on a word
        end instead of a fade.  Providers that declare none -- Gemini TTS, and
        therefore the whole Sinhala stack -- simply contribute no words, and
        their stops keep the fade.  Nothing about the audio or the
        customer-visible transcript changes either way.
        """

        feed = self._alignment_feed
        return stream if feed is None else observe_timed_transcript(stream, feed)

    def _synthesis_is_aligned(self) -> bool:
        """Does this session's TTS declare per-word timings.

        An unknown answer is treated as yes, which leaves the audio untouched.
        """

        try:
            tts = self.tts if self.tts is not NOT_GIVEN else self.session.tts
            return tts is None or bool(tts.capabilities.aligned_transcript)
        except Exception:
            return True

    def _synthesize(self, text: Any, model_settings: Any) -> Any:
        """The SDK's synthesis, with the opening started at its first sound.

        The session's first synthesis to play audio is the opening (written,
        generated, or the fallback line after a configured line that produced
        none), and a provider's leading silence there is dead air the
        callee waits through after their hello: 200-310 ms on Gemini
        Flash-Lite TTS. ``leading_silence.trim_leading_silence`` drops it,
        bounded. Not on a provider that declares word timings: those timings
        count from the provider's first sample, and the graceful stop checks
        them against the audio it plays (``graceful_stop.note_word``), so a
        trimmed front would distrust them. Those providers measured 40-60 ms.
        Every later synthesis passes as produced.
        """

        stream = super().tts_node(text, model_settings)
        if not self._opening_synthesized:
            self._opening_synthesized = True
            if not self._synthesis_is_aligned():
                stream = self._opening_stream(
                    trim_leading_silence(
                        stream, make_frame=rtc.AudioFrame, trimmed=self._opening_silence_trimmed
                    )
                )
        return self._aligned(stream)

    async def _opening_stream(self, stream: Any) -> Any:
        """The opening's frames; a synthesis that played none hands the trim on.

        When the configured line produced no audio, the worker speaks its
        fallback line next, and THAT is the opening the callee hears, so it
        takes the trim instead.
        """

        played = False
        try:
            async for frame in stream:
                played = True
                yield frame
        finally:
            if not played:
                self._opening_synthesized = False

    def tts_node(self, text: Any, model_settings: Any) -> Any:
        handle = self._evidence_speech_handle()
        speech_id = self._evidence_recorder.speech_identity(handle)
        if speech_id:
            self.observe_interruption_completion(handle)
        if not speech_id or not self._semantic_admission.admits_conversation_item():
            return self._synthesize(text, model_settings)
        if speech_id not in self._evidence_completion_hooks:
            self._evidence_completion_hooks.add(speech_id)

            def completed(completed_handle: Any) -> None:
                self._evidence_recorder.complete_preplayout_agent_speech(
                    speech_id,
                    interrupted=bool(getattr(completed_handle, "interrupted", False)),
                )

            try:
                handle.add_done_callback(completed)
            except Exception:
                # The candidate is still useful if the SDK does not expose a
                # completion hook.  We simply leave its completion certainty
                # conservative rather than making media execution depend on
                # observability.
                pass

        return self._synthesize(
            admitted_tts_text(
                text,
                speech_id=speech_id,
                recorder=self._evidence_recorder,
                admission=self._semantic_admission,
            ),
            model_settings,
        )

    def transcription_node(self, text: Any, model_settings: Any) -> Any:
        """Bind a voice-to-voice engine's own transcript of each speech it plays.

        There is no TTS node on a duplex engine, so ``tts_node`` never runs
        there, and a transcript fed only by it held no opening and no line the
        model spoke over the caller. The SDK hands each realtime speech's
        output transcript to this node while that speech is the session's
        current one, so the text is bound under the same SpeechHandle id the
        speaking edge stamps on the recorder's interval: a join by identity,
        not by timing. A cascade agent passes straight through, unchanged --
        its text was already bound at ``tts_node``.
        """

        if not self._speech_to_speech:
            return super().transcription_node(text, model_settings)
        handle = self._evidence_speech_handle()
        speech_id = str(getattr(handle, "id", "") or "").strip()
        if not speech_id:
            return super().transcription_node(text, model_settings)
        self.observe_interruption_completion(handle)
        if speech_id not in self._evidence_completion_hooks:
            self._evidence_completion_hooks.add(speech_id)

            def completed(completed_handle: Any) -> None:
                self._evidence_recorder.complete_preplayout_agent_speech(
                    speech_id,
                    interrupted=bool(getattr(completed_handle, "interrupted", False)),
                )

            try:
                handle.add_done_callback(completed)
            except Exception:
                # Without a completion the text still binds; its playout just
                # stays uncertified. Media never waits on this.
                pass
        return super().transcription_node(
            bound_output_transcript(
                text,
                speech_id=speech_id,
                recorder=self._evidence_recorder,
                admission=self._semantic_admission,
            ),
            model_settings,
        )




async def say_opening_utterance(
    session: Any,
    text: str,
    *,
    protection: OpeningTurnProtection | None = None,
    transcript_gate: CallerTranscriptGateStream | None = None,
    add_to_chat_ctx: bool = True,
    ended: Any = None,
) -> tuple[BaseException | None, bool]:
    """Speak the call's opening line so it cannot be killed before it is heard.

    Returns ``(provider failure or None, was interrupted)``, which is the
    speaker contract ``speak_opening_line`` calls this through. Awaiting a
    ``SpeechHandle`` never raises for a synthesis failure -- the error is on
    ``handle.exception()`` -- so a caller that ignored the return value could
    not tell a greeting that played from one that produced no audio at all.
    Two guarantees, one seam: this one keeps the opening from being cancelled
    before it is heard, and reporting the outcome is what lets the other
    notice there was nothing to hear.

    The defect (``call_MPXHrjeG3WhhMAYQ``): the callee's "hello?" was
    transcribed as "love" while Gemini TTS was still synthesizing the
    greeting, and the pinned runtime cancelled a reply that had never made a
    sound -- ``interruption_observed`` with ``playout_started: false,
    transcript_characters: 4``.  The window is wide because
    ``session.say`` never reports ``thinking``: the session sits in
    ``listening`` for the whole synthesis, so every state-driven protection in
    this worker is silent there.

    So the ``allow_interruptions=False`` window is opened HERE, at the call
    site the worker owns, and it stays open while the line plays: the second
    defect was the callee's pickup "hello?" cutting the opening a second
    after its first frame, on thirteen of fifteen answered calls of one
    batch, before the question it asked had been heard.  It closes when the
    caller has spoken for longer than a pickup (the protection's own timer,
    through the release handler the worker binds) or in the ``finally``
    below when the say ends.  ``OpeningTurnProtection`` holds the decision
    and is unit-tested; this function is the wiring.

    The transcript gate's OWN opening window is a separate, wider thing: it
    is armed earlier, at construction, so that a caller turn arriving before
    this function is even called is held too, not just one arriving after
    the first audio frame.  The ``set_opening_turn_in_flight(True)`` call
    below is therefore usually a no-op re-assert of a flag already ``True``;
    it still runs so this function is correct on its own if that ever
    changes.

    Two consequences of an uninterruptible handle in the pinned runtime, both
    deliberate and both bounded by the say:

    * The managed stack sets ``discard_audio_if_uninterruptible`` to False
      (``session_options_with_managed_pstn_turn_taking``), so a callee who
      talks over the opening IS transcribed; the transcript gate holds those
      events for as long as the opening is in flight and hands them to the
      runtime when it ends, which is how a pickup "hello" becomes the
      caller's turn after the question rather than an interruption of it.
      VAD, AMD and the interruption detector receive the real frame
      throughout.
    * ``_tts_task_impl`` only waits on ``_user_silence_event`` when the handle
      allows interruptions, so the greeting starts synthesizing over a talking
      callee instead of waiting for them to stop.  For an opening line that is
      the requirement, not a regression: the reported call opened in silence.

    Unprotected calls (browser tests, the simulations lab, the standalone
    demo, the platform release canary) pass ``protection=None`` and get
    exactly the ``allow_interruptions=True`` say they had before.
    """

    protected = protection.arm() if protection is not None else False
    if protected and transcript_gate is not None:
        transcript_gate.set_opening_turn_in_flight(True)

    def close_window() -> None:
        # ``disarm`` alone: it hands the handle to the bound release handler
        # (``release_opening``, bound once by the worker), which restores
        # ``allow_interruptions`` AND closes the transcript-gate window
        # itself. Doing either again here duplicated both on every ordinary
        # completion -- harmless only because each is idempotent -- and it
        # scattered forwarding the gate's released events (item 1) across two
        # call sites instead of the one the bound handler owns.
        if protection is not None:
            protection.disarm()

    try:
        with greeting_site():  # the controller learns which handle is the opening from where it is made
            handle = session.say(
                text, allow_interruptions=not protected, add_to_chat_ctx=add_to_chat_ctx
            )
    except BaseException:
        # say() raises synchronously when the session is not running. Nothing
        # was scheduled, so the window must close with it rather than outlive
        # the utterance it was opened for.
        close_window()
        raise
    if protection is not None:
        protection.attach(handle)
    try:
        # Raced against the session's end: a protected line cannot be
        # interrupted and the SDK never fails it once the far end has gone,
        # so a bare await here parked the entrypoint above its own teardown.
        played_out = await await_playout_or_session_end(handle, ended=ended)
    finally:
        # The ordinary close: the line played to its end, or failed, or the
        # session ended first. Also the path where no frame ever arrives, and
        # it is what stops a failed greeting from leaving the transcript gate
        # armed for the rest of the call. A close the caller's own speech
        # already performed is a no-op here.
        close_window()
    if handle is None:  # A stub session that returns nothing has nothing to fail.
        return None, False
    if not played_out:
        # The session ended first: the line was not heard to the end, and
        # ``interrupted`` is what tells the callers (the fallback accounting,
        # a terminal opening's farewell) not to count it as heard.
        return None, True
    try:
        return handle.exception(), bool(getattr(handle, "interrupted", False))
    except Exception:
        # ``exception()`` raises when the handle is somehow not done. No
        # evidence of failure is not the same as evidence of failure.
        return None, bool(getattr(handle, "interrupted", False))


async def generate_opening_reply(
    session: Any,
    *,
    language: str,
    protection: OpeningTurnProtection | None = None,
    transcript_gate: CallerTranscriptGateStream | None = None,
    instructions: str | None = None,
    ended: Any = None,
    recovery: Any = None,
    greeting: bool = True,
) -> BaseException | None:
    """Generate one owned opening through AgentSession's native model path.

    This is intentionally not an SDK-independent LLM call followed by
    ``session.say``: that would bypass the session's native usage collector,
    transcript/evidence events, and TTS accounting.  The returned handle is
    protected exactly as a literal opening is, for as long as it plays.  A
    failure is reported to the caller. A supported never-spoken opening can
    recover once after reconnection; its directive is preserved verbatim.

    ``instructions`` is the per-turn instruction the opening is generated
    from; the default asks for a generated opening in ``language``.  A
    GPT-Live call hands in its greeting directive (the configured opening,
    its mode and its language) so the session's one awaited reply -- with a
    handle, the runtime's reply timeout and this protection -- is the only
    owner of the opening.
    """

    protected = protection.arm() if protection is not None else False
    if protected and transcript_gate is not None:
        transcript_gate.set_opening_turn_in_flight(True)

    def close_window() -> None:
        # Same reasoning as ``say_opening_utterance``'s ``close_window``: the
        # bound release handler owns the restore and the gate close, exactly
        # once.
        if protection is not None:
            protection.disarm()

    directive = instructions if instructions is not None else generated_opening_instruction(language)

    def site() -> Any:
        # `greeting` is False for the opening that answers something the callee said: that reply may take a
        # speech window like any other, so it is not made at the greeting site.
        return greeting_site() if greeting else contextlib.nullcontext()

    if recovery is not None:
        def create() -> Any:
            with site():
                return session.generate_reply(
                    instructions=directive, tool_choice="none", allow_interruptions=not protected)

        try:
            return await recovery.run(
                instructions=directive,
                generate=create,
                wait=await_playout_or_session_end,
                attach=protection.attach if protection is not None else lambda handle: None,
            )
        finally:
            close_window()

    try:
        with site():
            handle = session.generate_reply(
                instructions=(
                    instructions
                    if instructions is not None
                    else generated_opening_instruction(language)
                ),
                # This is the pre-caller opening, not an opportunity to send an
                # SMS, transfer a call, or make any other business-side effect.
                # The shared system policy still applies to the generated words.
                tool_choice="none",
                allow_interruptions=not protected,
            )
    except Exception as exc:
        close_window()
        return exc
    if protection is not None:
        protection.attach(handle)
    try:
        # The same race as ``say_opening_utterance``: a generated opening is
        # protected too.
        await await_playout_or_session_end(handle, ended=ended)
    except Exception as exc:
        return exc
    finally:
        close_window()
    if handle is None:
        return None
    try:
        return handle.exception()
    except Exception:
        return None


def _playout_hold_stats(playout_hold: Any) -> dict[str, int]:
    """The hold guard's per-call counters, or nothing.

    Read defensively because it is merged into the same statement as every
    other per-call total: a guard that was never installed, or one whose
    counters cannot be read, must cost these four numbers, not the summary.
    """

    stats = getattr(getattr(playout_hold, "policy", None), "stats", None)
    return dict(stats) if isinstance(stats, Mapping) else {}


def _release_interruption_word_gate(
    gate: InterruptionWordGate,
    *,
    by_words: bool,
    words_heard: int = 0,
    session: Any = None,
    call_id: str = "",
) -> bool:
    """Give the floor back, and report whether it is actually clear.

    Two callers, both on the media loop: the transcript gate reporting that the
    caller has now said enough, and the agent-state handler reporting that the
    reply is over. Neither may fail a call: the worst outcome of a swallowed
    failure is that this reply stays uninterruptible until it ends, which is
    strictly better than a raised exception in a session event handler.

    The return value answers a question this gate cannot answer alone: is the
    current speech interruptible NOW. Giving back what this gate holds is not
    the same as the floor being clear, because it is not the only writer of
    ``allow_interruptions``. The opening-turn protection keeps the greeting
    uninterruptible while it plays, until the caller has spoken past the
    pickup bound, and this gate refuses a handle that is already held -- so a
    callee speaking over the opening can satisfy the criterion while this gate
    holds nothing, and a caller reading "released" as "safe to forward" would
    hand their turn to a runtime that refuses to answer it.
    """

    try:
        gate.release(by_words=by_words, words_heard=words_heard)
    except Exception:
        # Identified, because this event IS the floor-hold diagnosis. The whole
        # function exists to swallow the failure, so a bare event name would
        # leave an operator with the symptom and no call to look at.
        emit(
            "interruption_word_gate_release_failed",
            call_id=call_id,
            by_words=by_words,
            words_heard=int(words_heard),
        )
        return False
    speech = getattr(session, "current_speech", None) if session is not None else None
    if speech is None:
        return True
    try:
        return bool(getattr(speech, "allow_interruptions", True))
    except Exception:
        return False




#: How many calls this worker admits before it reports itself full.
#:
#: Read once, at import, so every load sample answers the same question. The
#: library calls ``load_fnc`` from a thread pool executor; re-reading the
#: environment there would let the gate's denominator change under a running
#: comparison for no gain, since nothing edits this process's environment.
MAX_CONCURRENT_JOBS = resolve_max_concurrent_jobs(os.environ)

#: The memory a job process may reach before it is logged, and before it is
#: killed (``0``, the default, never kills). Read once, as a pair, so the
#: warning is always below an enabled limit.
JOB_MEMORY_WARN_MB, JOB_MEMORY_LIMIT_MB = resolve_job_memory_thresholds(os.environ)


#: How many phone calls this worker holds at once: the host's declared
#: capacity, or -- where none is declared -- the ceiling the agents' old job
#: gate gave (four at the default budget of five). See phone_admission.
PHONE_CAPACITY = phone_capacity(os.environ)


def worker_job_load(worker: AgentServer) -> float:
    """Zero: the library's own job gate is not the admission control.

    Admission is ``phone_request``: an exact count of the calls this worker
    accepted, against ``PHONE_CAPACITY``. The library's gate cannot be exact
    -- it counts a launching job twice, once as active and once as reserved --
    and when it refuses, nothing of ours runs: livekit-server gives the
    dispatch up with no retry, and an inbound caller rings unanswered for as
    long as the carrier allows. Reporting no load keeps every dispatch coming
    to ``phone_request``, which refuses what it must and ends the ringing leg.

    One path still reaches the library's gate, and it is the library's own:
    with NO job active it weighs each reservation at ``load_threshold /
    num_idle_processes`` whatever this returns. On a host that declares its
    capacity the warm pool is at least that capacity
    (``resolve_phone_agent_idle_processes``), so only a burst larger than the
    capacity, arriving at an idle worker in one instant, is refused there -- a
    call that would have been refused anyway, but that rings until the trunk's
    ringing timeout instead of getting a busy at once. A host that declares
    none keeps its configured pool (2 by default) below the ceiling it admits
    (4): there a burst of three or four at an idle worker is refused by the
    library and rings out, as before this admission existed.

    Taking the worker as its only argument is the library's own selector:
    ``_invoke_load_fnc`` passes the server when the callable declares a
    parameter. It is not read.
    """
    return 0.0


#: How many Simulations Lab live-call tests this worker may hold, and the call
#: slots and processes a test must leave free for customers (design F4: two
#: and two). Read once, as a pair. See phone_admission.
SIM_TEST_SLOTS, SIM_CUSTOMER_HEADROOM = resolve_simulation_test_reserve(os.environ)

#: The calls that have ended while their job process finishes its teardown:
#: marked by the job process, read by the worker's count.
ENDED_CALLS = EndedCalls()

PHONE_ADMISSION = PhoneAdmission(
    capacity=PHONE_CAPACITY,
    active_jobs=lambda: active_job_rooms(server),
    ended_calls=ENDED_CALLS.ended,
    # A mark left from before the job was admitted is removed at admission.
    clear_mark=ENDED_CALLS.clear,
    # The host's job budget (SKYSAY_WORKER_MAX_JOBS; the capacity plus two on
    # every host): an ended call gives its slot back, but its process, still
    # mixing the call's evidence, counts here until it exits.
    process_cap=MAX_CONCURRENT_JOBS,
    test_slots=SIM_TEST_SLOTS,
    customer_headroom=SIM_CUSTOMER_HEADROOM,
)

#: The server API client ``phone_request`` ends an inbound leg with, built on
#: first use inside the worker's event loop (it owns an aiohttp session).
_refusal_api: list[api.LiveKitAPI] = []


async def phone_request(request: JobRequest) -> None:
    """Every dispatch's admission: see ``phone_admission.handle_dispatch``."""

    await handle_dispatch(
        request,
        admission=PHONE_ADMISSION,
        list_participants=_room_participants,
        delete_room=_delete_room,
        record_refusal=lambda attributes, identity: asyncio.to_thread(
            _record_inbound_refusal, dict(attributes), identity
        ),
        emit=emit,
        mark_outbound_refused=_mark_capacity_refused,
        report_capacity_reserved=lambda call_id, room_name: asyncio.to_thread(
            _report_capacity_reserved, call_id, room_name
        ),
    )


def _refusal_client() -> api.LiveKitAPI:
    if not _refusal_api:
        _refusal_api.append(api.LiveKitAPI())
    return _refusal_api[0]


async def _room_participants(room_name: str) -> list[Any]:
    listed = await _refusal_client().room.list_participants(
        api.ListParticipantsRequest(room=room_name)
    )
    return list(listed.participants)


async def _mark_capacity_refused(room_name: str, job_id: str) -> bool:
    """Tell an outbound room's originator its dispatch was refused for capacity.

    Read-modify-write of the room's metadata; see
    ``phone_admission.CAPACITY_REFUSED_KEY`` for why nothing else can write it
    in between. False when the room is gone or its metadata is not the
    originator's.
    """

    listed = await _refusal_client().room.list_rooms(api.ListRoomsRequest(names=[room_name]))
    room = next((r for r in listed.rooms if r.name == room_name), None)
    if room is None:
        return False
    merged = with_capacity_refusal(str(room.metadata or ""), job_id)
    if merged is None:
        return False
    await _refusal_client().room.update_room_metadata(
        api.UpdateRoomMetadataRequest(room=room_name, metadata=merged)
    )
    return True


async def _delete_room(room_name: str) -> None:
    await _refusal_client().room.delete_room(api.DeleteRoomRequest(room=room_name))


def _report_capacity_reserved(call_id: str, room_name: str) -> bool:
    """Tell the control plane a live-call test waits for capacity; whether it recorded it.

    The platform media worker's route (the inbound refusal's credential). The
    control plane defers the test's run; nothing is an attempt outcome.
    """

    if not os.environ.get("SKYSAY_PLATFORM_MEDIA_TOKEN", "").strip():
        return False
    response = _transport("SKYSAY_PLATFORM_MEDIA_TOKEN").request(
        "POST",
        "/v1/simulations-lab/capacity-refusals",
        {
            "schema": "skysay.simulations-lab-capacity-refusal.v1",
            "call_id": call_id,
            "room_name": room_name,
            "reason": "capacity_reserved",
        },
    )
    return isinstance(response, dict) and response.get("recorded") is True


def _record_inbound_refusal(attributes: dict[str, str], identity: str) -> bool:
    """Record the refused inbound call for its workspace; whether a call was recorded."""

    if not os.environ.get("SKYSAY_PLATFORM_MEDIA_TOKEN", "").strip():
        # A self-hosted gateway has no platform credential; the worker event
        # is its record.
        return False
    caller, called, provider_call_id = inbound_sip_identity(attributes, identity)
    classification = classify_inbound_leg(
        release_canary_enabled=os.environ.get("SKYSAY_RELEASE_CANARY_ENABLED", "") == "1",
        from_e164=caller,
        to_e164=called,
    )
    if classification != INBOUND_CUSTOMER:
        return False
    response = _transport("SKYSAY_PLATFORM_MEDIA_TOKEN").request(
        "POST",
        "/v1/platform-media/inbound-refusals",
        {
            "to_phone_number": called,
            "from_phone_number": caller,
            "provider_call_id": provider_call_id,
            "reason": "inbound_capacity_reached",
        },
    )
    # False for a number the platform does not route (nothing to record).
    return isinstance(response, dict) and response.get("recorded") is True


class PhoneAgentServer(AgentServer):
    """The agents' worker, which lets refused legs end before it shuts down.

    It also publishes this host's readiness beacon (``agents_readiness``): the
    outbound originator claims no job while no agents worker here is
    registered with LiveKit and admitting. ``registered`` is true
    exactly while the SDK runs its registered connection, ``_run_ws``: the
    pinned 1.8.1 calls it right after the register response and returns from
    it (or raises) when that connection ends. Its own ``_connecting`` flag is
    NOT used: it stays false through the reconnect back-off after a dropped
    connection, which would report a disconnected worker as registered for up
    to 10 s. The image build asserts the pinned SDK still has this shape.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._skysay_registered = False
        self._skysay_draining = False
        self._skysay_beacon_closed = False
        self._skysay_beacon_task: asyncio.Task[None] | None = None
        self._skysay_beacon_dir = readiness_dir(os.environ)
        self._skysay_beacon_key = str(os.environ.get("SKYSAY_WORKER_ID") or "livekit-agent").strip()
        if self._skysay_beacon_dir is not None:
            self.on("worker_started", self._start_readiness_beacon)

    async def _run_ws(self, ws: Any) -> None:  # type: ignore[override]
        self._skysay_registered = True
        self._write_readiness_beacon()
        try:
            await super()._run_ws(ws)
        finally:
            self._skysay_registered = False
            self._write_readiness_beacon()

    def _write_readiness_beacon(self) -> None:
        directory = self._skysay_beacon_dir
        if directory is None or self._skysay_beacon_closed:
            return
        try:
            write_beacon(
                directory,
                self._skysay_beacon_key,
                registered=self._skysay_registered,
                draining=self._skysay_draining or bool(self.draining),
                now=time.time(),
            )
        except Exception as exc:  # noqa: BLE001 - a stale beacon is the safe failure
            print(
                "SKYSAY_AGENTS_READINESS " + json.dumps({"event": "beacon_write_failed", "error_type": type(exc).__name__}),
                flush=True,
            )

    def _start_readiness_beacon(self) -> None:
        async def beat() -> None:
            while True:
                self._write_readiness_beacon()
                await asyncio.sleep(BEACON_INTERVAL_SECONDS)

        if self._skysay_beacon_task is None:
            self._skysay_beacon_task = asyncio.get_running_loop().create_task(beat())

    async def drain(self, *args: Any, **kwargs: Any) -> None:
        # Said at once, before the SDK's own flag flips inside its lock: a
        # draining worker takes no new dispatch, so the originator must stop
        # claiming for it now, not at the next beat.
        self._skysay_draining = True
        self._write_readiness_beacon()
        await super().drain(*args, **kwargs)

    async def aclose(self) -> None:
        if self._skysay_beacon_task is not None:
            self._skysay_beacon_task.cancel()
        # No more beacon writes from here: closing ends the registered
        # connection, and its ``_run_ws`` exit would otherwise write the file
        # back after it was removed.
        self._skysay_beacon_closed = True
        await drain_refusals()
        await super().aclose()
        if self._skysay_beacon_dir is not None:
            try:
                remove_beacon(self._skysay_beacon_dir, self._skysay_beacon_key)
            except OSError:
                pass


server = PhoneAgentServer(
    num_idle_processes=resolve_phone_agent_idle_processes(os.environ),
    setup_fnc=setup_process,
    # Admission is phone_request's exact count; see worker_job_load for why
    # the library's own gate is given no load. load_threshold is left at the
    # library's default on purpose -- 0.7 in production, disabled in dev mode.
    load_fnc=worker_job_load,
    # The library samples PSS. The limit defaults to disabled: a long call's
    # evidence finalization legitimately reaches well over a gigabyte, so any
    # limit this host could afford would kill real calls' post-call reporting
    # (sizing beside DEFAULT_JOB_MEMORY_LIMIT_MB). An operator-set limit is
    # resolved as a pair with the warning so the warning cannot end up at or
    # above it.
    job_memory_warn_mb=JOB_MEMORY_WARN_MB,
    job_memory_limit_mb=JOB_MEMORY_LIMIT_MB,
    # Without this the library waits an hour for its sessions to finish and
    # the container runtime kills it long before that, so the drain never
    # actually completes and a release cuts whatever call was live.
    drain_timeout=resolve_drain_timeout_seconds(os.environ),
    # The health server's bind port. Unset resolves to 8081, which is what the
    # library's own production default already gave this worker, so nothing
    # about a normal deployment changes. It is settable because a blue/green
    # roll runs a SECOND copy of this role beside the one already holding
    # 8081 on the host's shared network namespace, and the second one dies on
    # bind unless it can be told to take whatever port the OS has free.
    port=resolve_worker_http_port(os.environ),
)


@server.rtc_session(agent_name=AGENT_NAME, on_request=phone_request)
async def production_agent(ctx: JobContext) -> None:
    # The scope is the job, not the warm process. The organization tag is set
    # once the call's config is known and drops when this returns. An exception
    # that escapes is captured HERE, while the tag is still on the scope: the
    # framework logs it afterwards, and the SDK's dedupe keeps that second
    # report, which no longer has the tag, from becoming a second issue.
    # ``job_journals``: the call's worker-events.jsonl is drained to disk
    # before the job returns, on every exit, because a job process can end
    # without the interpreter's exit hooks and the writer is a daemon thread.
    #
    # The worker's capacity slot is the CALL's, not this process's: it is
    # given back when the room disconnects, not when the process exits after
    # the evidence and reports (phone_admission.EndedCalls). The return is
    # the backstop for a room whose disconnect never arrives.
    call_ended = watch_call_end(ctx.room, str(getattr(ctx.job, "id", "") or ""), ENDED_CALLS)
    with error_tracking_job_scope():
        try:
            async with worker_events.job_journals():
                await _production_agent_job(ctx)
        except asyncio.CancelledError:
            # A deploy or a host shutdown cancels the job task. That is the
            # process stopping, not a fault in the call, and filing it would
            # page on every release that lands over a live session.
            raise
        except BaseException as exc:
            capture_exception(exc)
            raise
        finally:
            call_ended()


async def _production_agent_job(ctx: JobContext) -> None:
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
    governed_root: WorkflowRoot | None = None
    # Bound before the guard below, not inside it: the dispatch is parsed after
    # ``ctx.connect``, which can raise, and the handler reads this to decide
    # whether anybody is listening.
    browser_test: Any | None = None
    pre_media_call_id = ""
    live_pickup = None
    native_opening_authorization = None
    gemini_pickup_generation = None
    native_caller_revision = 0
    opening_reply_recovery = None
    pre_media_stage = "session_connect"
    browser_participant_mismatch = False
    inbound_route = INBOUND_CUSTOMER
    leased_components: dict[str, dict[str, str]] | None = None
    runtime_environ: dict[str, str] | None = None
    # Pre-declared (like the two locals above) so the ``finally`` below can
    # read it even when the try block raised before parsing reached it.
    follow_caller: FollowCallerProfiles | None = None
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

    gpt_live_http = None
    gpt_live_llm = None
    transcript_publisher: TranscriptEventPublisher | None = None
    # The typed rows stored in the call's own recording bundle, for EVERY
    # post-admission hosted session -- phone and browser alike. Built inside
    # the guard below, beside the evidence recorder whose clock it borrows,
    # and read by the session handlers and the language tee further down.
    transcript_event_recorder: CallTranscriptEventRecorder | None = None
    # Created after media admission but before the control plane has been
    # told media started, so the guard below may have to close them. Declared
    # here so it can read them whether or not the try block got that far.
    sidecar: Any | None = None
    inbound_audio_output: Any | None = None
    capture_tasks: list[Any] = []
    state_path: Path | None = None
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
        if browser_test is not None:
            if browser_test.participant_agent_name:
                # LiveKit Agent jobs join as ``agent-<job id>``. Their stable,
                # authenticated selector is the ``lk.agent.name`` attribute,
                # and the pinned SDK helper waits for exactly that value. A
                # synthetic caller therefore cannot be matched by the static
                # participant identity used by ordinary browser clients.
                participant = await wait_for_agent(
                    ctx.room,
                    agent_name=browser_test.participant_agent_name,
                )
            else:
                # Human browser participants retain the exact identity the
                # control plane minted into their room token.
                participant = await ctx.wait_for_participant(
                    identity=browser_test.participant_identity,
                )
        else:
            participant = await ctx.wait_for_participant()
        attributes = dict(participant.attributes or {})
        caller = str(attributes.get("sip.phoneNumber") or attributes.get("sip.from") or "unknown")
        provider_call_id = str(attributes.get("sip.callIDFull") or attributes.get("sip.callID") or participant.identity)
        pre_media_stage = "adapter_configuration"
        adapter = build_adapter()
        if browser_test is not None:
            if browser_test.participant_agent_name:
                if (
                    str((participant.attributes or {}).get("lk.agent.name") or "")
                    != browser_test.participant_agent_name
                ):
                    browser_participant_mismatch = True
                    raise ValueError("browser-test agent participant does not match dispatch")
            elif participant.identity != browser_test.participant_identity:
                browser_participant_mismatch = True
                raise ValueError("browser-test participant identity does not match dispatch")
            caller = f"browser:{browser_test.call_id}"
            # The panel beside the editor shows tool calls and language
            # switches inline in the live transcript. They travel on a
            # second room text-stream topic addressed to THIS participant
            # (the identity the control plane minted into its token, or
            # the authenticated synthetic caller), so the channel inherits
            # the tenant binding of the room and adds no API surface. A
            # phone call has no browser to feed and builds none.
            browser_identity = str(participant.identity)
            transcript_publisher = TranscriptEventPublisher(
                send_text=lambda text: ctx.room.local_participant.send_text(
                    text,
                    topic=TRANSCRIPT_EVENTS_TOPIC,
                    destination_identities=[browser_identity],
                ),
                emit=lambda event, **details: _quietly(
                    emit, event, call_id=browser_test.call_id, **details
                ),
            )
            pre_media_stage = "outbound_resume"
            call = await asyncio.to_thread(
                adapter.prepare_workspace_browser_test,
                call_id=browser_test.call_id,
                agent_id=browser_test.agent_id,
                organization_id=browser_test.organization_id,
                project_id=browser_test.project_id,
                # The branch phone_request admitted this job on, asked of the
                # same room and metadata: the control plane binds it (8.1).
                admission="live_call_test" if live_call_test_from(
                    str(getattr(getattr(ctx.job, "room", None), "name", "") or ""), job_metadata
                ) else "",
            )
        elif outbound is not None:
            if participant.identity != outbound.participant_identity:
                raise ValueError("outbound SIP participant identity does not match dispatch")
            if str(attributes.get("skysay.call_id") or "") != outbound.call_id:
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
                release_canary_enabled=os.environ.get("SKYSAY_RELEASE_CANARY_ENABLED", "") == "1",
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
                    inbound_trunk_id=str(attributes.get("sip.trunkID") or ""),
                )
            else:
                # An unrecognized classification is a bug, and customer
                # routing is the one place it must not default to.
                raise ValueError("unclassified_inbound_leg")
        pre_media_call_id = call.call_id
        # The SIP caller may have cancelled during admission's network I/O.
        # Do not start a provider or manufacture an answered call from the
        # stale participant object; pre-media cleanup releases its reservation.
        require_pre_media_participant(ctx.room, participant)
        note_error_tracking_organization(str(call.config.get("organization_id") or ""))
        pre_media_stage = "component_configuration"
        if pinned_workflow_run_id(call.config):
            from workflow_root import managed_source_keys_for_workflow
            plan = create_workflow_root_plan(call.config)
            source_keys = managed_source_keys_for_workflow(plan, call.config, environment=os.environ,
                                                            leased_components=leased_components or {})
            governed_root = WorkflowRoot(plan=plan, ctx=ctx, participant=participant, call=call,
                adapter=adapter, browser_test=browser_test, outbound=outbound, provider_call_id=provider_call_id,
                inbound_route=inbound_route, inbound_release_canary=INBOUND_RELEASE_CANARY,
                report_browser_test_failure=report_browser_test_failure,
                remove_participant=lambda: remove_sip_participant_before_media(ctx, participant),
                start_sidecar=start_evidence_sidecar, evidence_dir=EVIDENCE_ROOT / safe_component(call.call_id),
                state_path=SESSION_ROOT / f"{safe_component(call.call_id)}.json",
                ringing_timeout_seconds=ringing_timeout_seconds(os.environ),
                callee_first_enabled=callee_first_enabled(os.environ), emit=emit, managed_source_keys=source_keys)
            governed_root.navigation_classifier_factory = lambda: build_far_end_classifier(
                environ=os.environ, emit=lambda event, **details: emit(event, call_id=call.call_id, **details))
            await governed_root.run()
            return
        profile = profile_from_realtime_config(call.config)
        gemini_live_call = is_gemini_live_config(call.config)
        s2s_live_call = is_s2s_live_config(call.config)
        # Follow-the-caller is an opt-in, immutable admission envelope.  Parse
        # every allowed profile before any credential lease/provider object so
        # a malformed or injected profile cannot become a mid-call fallback.
        # Voice-to-voice engines follow the caller in the live session; cascade
        # STT/TTS profile switching does not apply.
        follow_caller = None
        if not s2s_live_call:
            follow_caller = FollowCallerProfiles.from_realtime_config(
                call.config,
                profile_from_config=lambda candidate: profile_from_realtime_config(
                    candidate, require_component_billing=False
                ),
                validate_billing_profile=lambda language, candidate: _validate_follow_caller_billing_profile(
                    language, candidate, call.config.get("component_billing")
                ),
            )
        if follow_caller is not None and follow_caller.active_language != profile.language:
            raise LanguageSwitchError("follow_caller active language does not match the admitted profile")
        # Selected customer keys are leased only after the call-bound config
        # arrives and before any provider object/media session exists. The
        # overlay is local to this invocation; it never mutates os.environ,
        # job metadata, room metadata, telemetry, or logs.
        leased_components = (
            call.lease_component_credentials()
            if should_lease_components(
                call.config, platform_media_worker=call.platform_media_worker
            )
            else {}
        )
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
        voice_call_owner = object()
        gpt_live_llm = None
        gpt_live_http = None
        # The call-ending policy for a voice-to-voice call, composed by the
        # control plane. ``policy_in_prompt`` says the stored prompt
        # already carries it, so only the GPT-Live backend below needs it added.
        live_policy, live_policy_in_prompt = live_call_ending_policy(call.config)
        # Set below for a cascade call whose far end is heard in any language.
        far_end_listening: Any | None = None
        if s2s_live_call:
            # One construction for every live entrypoint: the
            # homepage demo builds its model through the same function.
            if live_model_needs_http_session(call.config):
                gpt_live_http = openai_http_session()
            try:
                gpt_live_llm = build_live_model(
                    call.config,
                    environ=runtime_environ,
                    http_session=gpt_live_http,
                    native_call_id=call.call_id,
                    native_max_duration_seconds=int(resolved_max_call_seconds(call.config)[0]),
                    native_registry=CallAgentRegistry(call),
                    # An agent that navigates may wait in a queue for minutes; its engine is built
                    # to outlive a refused resumption handle (``gemini_live_hold``).
                    navigation=outbound is not None and navigation_settings(call.config).enabled,
                    emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
                )
            except Exception as exc:
                if isinstance(exc, LiveEngineUnavailable):
                    raise ProductionConfigurationError(str(exc)) from exc
                raise
            stack = type(
                "S2SLiveStack",
                (),
                {
                    "session_options": {"vad": ctx.proc.userdata.get("vad")},
                    "soniox_usage_tracker": None,
                    "tts_unit_meter": None,
                    "tts": None,
                    "stt": None,
                    "llm": gpt_live_llm,
                    "stt_candidate_locales": (),
                },
            )()
        else:
            # The other party's language is not the agent's to choose: an
            # airline menu, a person who answers, an inbound caller may speak
            # any language, and a transcriber pinned to the agent's own hears
            # nothing of it. Admission froze the open construction for every
            # call that listens open (the agent may lock it), so the direction
            # is not asked here; it is used only when this image's registry
            # agrees, never on a follow-the-caller call, whose declaration
            # already chose one, and -- like the detector's whole-line plan --
            # never on a call that holds a keypad value: its redactor judges
            # numbers only in the languages of its numeral table.
            far_end_listening = (
                frozen_far_end_listening(call.config, profile)
                if follow_caller is None and not keypad_fields(call.config)
                else None
            )
            _quietly(
                emit, "far_end_listening", call_id=call.call_id,
                listening="open" if far_end_listening is not None else "profile",
            )
            stack = build_demo_stack(
            profile,
            config=call.config, call_owner=voice_call_owner,
            environ=runtime_environ,
            simultaneous_stt=(follow_caller.simultaneous_stt if follow_caller is not None else None),
            far_end_listening=far_end_listening,
            load_vad=lambda **_kwargs: ctx.proc.userdata["vad"],
            soniox_usage_reference=(
                (f"{call.call_id}:segment:1" if follow_caller is not None else call.call_id)
                if soniox_correlation_required
                else ""
            ),
            # New component-metered Soniox stacks need an opaque call
            # reference before media. Historical pinned stacks without this
            # billing contract remain runnable rather than being rewritten or
            # newly rejected merely because the registry evolved.
            require_soniox_usage_reference=soniox_correlation_required,
        )
        # Whether this call can speak a line of TEXT: a stack with a TTS. The
        # SDK asks the same question before ``say(text)`` and refuses it on a
        # voice-to-voice session, so it is read off the stack, not the engine
        # name.
        speaks_text = getattr(stack, "tts", None) is not None
        voice_admission = (
            None
            if s2s_live_call
            else (
            admit_voice_targets(
                stack, call_owner=voice_call_owner,
                allowed_profiles=follow_caller.profiles,
                simultaneous_stt=follow_caller.simultaneous_stt,
            ) if follow_caller is not None else None
            )
        )
        # This is the last operation before caller audio can be consumed.  It
        # must remain inside the pre-media guard: a rejected admission still
        # owns a ringing SIP leg and an existing wallet reservation.
        pre_media_stage = "media_admission"
        call.admit_media()
        # Admission is local: the control plane learns that media started only
        # from the ``room.media_started`` report at the end of this block. Until
        # that report lands, a failure anywhere below leaves an answered
        # customer in a silent room and a call the control plane cannot settle
        # (2026-09-17: every GPT-Live call crashed building its evidence rows,
        # one line past the old guard, and five answered customers heard
        # silence). So everything up to and including that report stays inside
        # this one guard, under one stage.
        pre_media_stage = "worker_entry"
        # The caller's audio, subscribed whenever it appears -- not only if it
        # happens to exist right now.
        #
        # This used to be a bare loop over the publications this participant
        # held at this instant, and with ``auto_subscribe=SUBSCRIBE_NONE``
        # above it was the only thing in the product that ever subscribed
        # anything. On a browser test the caller is already publishing its
        # microphone when ``wait_for_participant`` returns, so the loop worked
        # -- by timing. On an outbound SIP call it does not: the caller's track
        # was measured arriving 0.51s AFTER this point, a quarter-second after
        # ``session.start`` had returned, and nothing subscribed it, so the
        # agent session's input adopted nothing and every consumer of caller
        # audio read zero for the whole call.
        #
        # Registering the handler stays inside the pre-media guard and after
        # ``admit_media()`` for the same reason the sweep did: a rejected
        # admission still owns a ringing leg and must not start pulling its
        # media. Subscribing on publish cannot bring back the early media a
        # duplex engine would otherwise speak over ringback either --
        # ``session.input.set_audio_enabled(False)`` above holds the input
        # muted until the answer, upstream of every engine.
        def note_caller_audio_subscribed(trigger: str, track_sids: list[str]) -> None:
            emit(
                "caller_audio_subscribed",
                call_id=call.call_id,
                trigger=trigger,
                track_sids=track_sids,
            )

        # Opened before the subscription below: its worker-entry sweep emits
        # synchronously, and that event belongs to this call's journal.
        evidence_dir = EVIDENCE_ROOT / safe_component(call.call_id)
        worker_events.open_journal(call.call_id, evidence_dir)
        # A human browser test keeps exactly the behaviour it has today: the
        # sweep is the whole story there and no late-publication handler is
        # armed for it. A synthetic agent caller does get the handler:
        # ``wait_for_agent`` returns as soon as it joins, and it publishes its
        # microphone only after connecting, so the sweep can run first.
        wire_caller_audio_subscription(
            ctx.room,
            participant,
            audio_kind=rtc.TrackKind.KIND_AUDIO,
            note=note_caller_audio_subscribed,
            late_publications=arms_late_caller_publications(browser_test),
        )

        # Resolved before the evidence identity is fingerprinted: whether the
        # caller may hear a filler is part of the measured configuration, so two
        # otherwise identical calls with the feature on and off never share it.
        ack_settings = acknowledgement_settings(
            call.config, language=profile.language, environ=os.environ
        )
        # Same rule for what plays during a slow tool call: resolved here so the
        # identity below records the mode the caller could hear.
        tool_wait = tool_wait_settings(
            call.config, language=profile.language, environ=os.environ
        )
        recorder = SynchronizedEvidenceRecorder(
            evidence_dir,
            speech_to_speech=s2s_live_call,
            # The engine's timing row (data, one per engine); none on cascade.
            caller_transcript_timing=(
                caller_transcript_timing(voice_frontend(call.config)) if s2s_live_call else None
            ),
            transcript_metadata={
                "skysay": {
                    "agent_id": str(call.config.get("agent_id") or ""),
                    "organization_id": str(call.config.get("organization_id") or ""),
                    "project_id": str(call.config.get("project_id") or ""),
                    "call_id": call.call_id,
                },
                "effective_voice_stack": effective_voice_stack_identity(
                    call.config,
                    filler_acknowledgements_enabled=ack_settings.enabled,
                    tool_wait_feedback=tool_wait.mode,
                    stt_candidate_locales=stack.stt_candidate_locales,
                ),
            },
        )
        # One call-wide typed-event recorder per admitted session, anchored to
        # the evidence recorder's PUBLIC epoch so a tool row and a transcript
        # turn are offsets from the same zero. The pinned SDK offers no
        # durable turn id, so that shared clock is the only thing that can
        # order them (measured, not assumed). The browser test panel's live
        # publisher above stays a separate sink: its envelopes summarise
        # arguments and results for a page and are never persisted.
        transcript_event_recorder = CallTranscriptEventRecorder(
            relative_ms=recorder.relative_ms,
            engine=transcript_events_engine(call.config, workflow=False),
            # The tools THIS call resolved, with the server-issued ref each
            # spec carries. The SDK hands a callback the model-facing name, so
            # the map is built once here rather than looked up per event.
            tools=call.config.get("tools"),
            # The locales this call FROZE at admission, or None when it froze
            # no follow-caller contract -- which is what makes "could not
            # switch" distinguishable from "could and did not". Read from the
            # frozen envelope rather than from the parsed switcher, because
            # the envelope is the object the importer validates rows against
            # and the parser casefolds the keys it builds.
            allowed_languages=transcript_events_allowed_languages(call.config),
            emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
        )
        recorder.set_transcript_events_source(transcript_event_recorder.section)
        closing_evidence = ClosingEvidence(call_id=call.call_id, relative_ms=lambda: recorder.relative_ms(time.monotonic()))
        recorder.set_closing_evidence_source(closing_evidence.section)
        candidate_locales_row = stt_candidate_locales_requested(
            stack.stt_candidate_locales,
            language=profile.language,
            registry_name=profile.stt_registry,
            model=profile.stt_model,
        )
        if candidate_locales_row is not None:
            try:
                emit(STT_CANDIDATE_LOCALES_REQUESTED, call_id=call.call_id, **candidate_locales_row)
            except Exception:
                # A diagnostic row must not be the thing that ends a call. This
                # write happens before the session starts and only on a widened
                # call, so an unguarded failure here would take down exactly the
                # calls the row exists to explain, and only those.
                pass
        if ack_settings.enabled:
            recorder.mark_acknowledgements_enabled()
        if tool_wait.enabled:
            recorder.mark_tool_wait_feedback(tool_wait.mode)
        # The drain below trusts the sidecar's live agent edge only while the
        # sidecar is demonstrably still observing: connected, and delivering
        # frames on some track within the last quarter second.
        sidecar_lost = {"at": None}
        # Whether this call is waiting in a phone menu or a hold queue. ONE
        # object, read by every rule that would otherwise end such a call; off
        # (and inert) unless the agent opted in, and only ever on an outbound
        # call, which is the only call the answer detectors arm for.
        # Value mode's redactor: inert until a keypad value is armed, and only
        # on a call that may navigate at all. Everything the recorder keeps goes
        # through it; the executor that presses a value arms it.
        value_redactor = Redactor(
            call.call_id, default_language=profile.language,
            # (every language the call may hear: the agent's, and every profile a follow-the-caller switch may pick)
            call_languages=(
                profile.language, *(profile.languages or ()), *(follow_caller.profiles if follow_caller is not None else ()),
            ),
        )
        # The menu, hold and person rows of the Calls drawer. The policy announces
        # its transitions to it; the executor writes the rest.
        navigation_timeline = NavigationTimeline(
            transcript_event_recorder,
            redactor=value_redactor,
            music_active=lambda: bool(answer_detection.music_reading.active),
        )

        #: Other owners of a policy transition (the workflow agent's settling exchange).
        navigation_listeners: list[Callable[..., None]] = []

        def announce_navigation(event: str, **details: Any) -> None:
            _quietly(emit, event, call_id=call.call_id, **details)
            navigation_timeline.on_policy_event(event, **details)
            for listener in list(navigation_listeners):
                _quietly(listener, event, **details)

        navigation, answer_detection = build_call_floor_policy(
            call.config, outbound=outbound is not None, language=profile.language,
            announce=announce_navigation, policy_type=NavigationPolicy,
            no_caller_speech_seconds=resolved_no_caller_speech_seconds(call.config)[0],
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            quiet_emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
            # The call's own transcript source: a live engine's captions, else the session's transcriber.
            navigation_sources=(
                (("captions" if s2s_live_call else "stt"), profile.stt_registry, profile.stt_model),
            ) if profile.stt_model else (),
            # Every language the call may switch to as well: its own transcriber hears them all.
            navigation_languages=(profile.language, *(profile.languages or ()),
                                  *(follow_caller.profiles if follow_caller is not None else ())),
        )
        if navigation.enabled:
            recorder.set_redactor(value_redactor)
            # The call's own transcript source, as built: it feeds navigation's hints, the model's menu input and
            # recovery, so its construction is attested beside the detector's (never a credential).
            _quietly(emit, "navigation_source_built", call_id=call.call_id,
                     role="captions" if s2s_live_call else "session", registry_name=profile.stt_registry,
                     model=profile.stt_model, languages=[profile.language, *(profile.languages or ())],
                     settings={} if s2s_live_call else dict(getattr(stack, "stt_construction", None) or {}))
        # What the line says, one label per final utterance, applied to the
        # policy in order. Only for an agent that opted in; a classifier whose
        # self-test fails turns navigation OFF for the call (and, if it was
        # already in an episode, restarts the clocks the way an exit does).
        far_end_watch: FarEndWatch | None = None
        far_end_perception = None
        final_router = None
        # What the call does on a menu (keys, speech, values, the output switch, the model's
        # context). ``None`` unless the agent navigates AND the far end can be listened to: a call
        # with no detector cannot read a prompt, so it keeps the behaviour it had.
        navigation_actions: NavigationActions | None = None
        navigation_player: Any | None = None
        # A live engine's input: GPT-Live has a server-side mute; Gemini Live has none and does not
        # survive an input-idle hold, so its caller frames keep flowing as silence.
        live_input_backend: Any | None = None
        if s2s_live_call:
            live_input_backend = (
                SilencedInput(emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details))
                if gemini_live_call
                else GptLiveInput(
                    lambda: live_duplex_session(session_agent, gpt_live_llm),
                    emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
                )
            )
        if navigation.enabled:
            far_end_classifier = build_far_end_classifier(
                os.environ,
                emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
            )
            far_end_watch = FarEndWatch(
                policy=navigation,
                classifier=far_end_classifier,
                # Every label crosses the watch: one the call's own source (its transcriber or captions, the source of a
                # session hint) was measured to fail in only observes. The detector's sources add to this in perception.
                label_trusted=lambda language: not navigation_untrusted_sources(
                    load_voice_registry(),
                    ((("captions" if s2s_live_call else "stt"), profile.stt_registry, profile.stt_model),)
                    if profile.stt_model else (),
                    (language,),
                ),
                emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
                on_unavailable=lambda was_in_episode: (
                    answer_detection.note_navigation_exit() if was_in_episode else None
                ),
                external_unresolved=lambda: caller_transcript_withheld() or (
                    far_end_perception is not None and far_end_perception.unresolved()
                ),
            )
            far_end_watch.start()

            async def settle_navigation_labels() -> bool:
                """A model request is judged on the label of the utterance it answers.

                ``False`` when a label is still unapplied after the bounded wait, or a final is in
                flight or withheld upstream: the state may be stale, and the request is refused.
                """
                if far_end_watch is None:
                    return True
                if far_end_watch.pending and not await far_end_watch.settled_through_now():
                    return False
                return not far_end_watch.unresolved

            def session_hears_language(language: str) -> bool:
                """Whether the session's own transcriber is listening for ``language`` (a cascade model reads the far
                end through it). A call that follows the caller switches it to any declared language; any other call
                listens for the call's language and the candidate locales it was widened with."""

                if far_end_listening is not None:
                    # The open construction names no language: it hears whatever the far end speaks.
                    return True
                code = navigation_base_language(language)
                heard = {navigation_base_language(profile.stt_language or profile.language)}
                heard.update(navigation_base_language(item) for item in getattr(stack, "stt_candidate_locales", ()) or ())
                if follow_caller is not None:
                    heard.update(navigation_base_language(item) for item in (*(profile.languages or ()), *follow_caller.profiles))
                return code in heard

            # What the call DOES on a menu: keys, speech, values, the output switch and the
            # model's context, all behind one executor. Built before the session exists; every
            # port that needs the session, the room or the TTS reads it when it is first used.
            # A workflow run is driven by a NavDecider through the same executor (no model tools, no generated
            # replies: see the ``model_feed`` and ``engine_generate`` arguments below).
            built_actions = build_navigation_actions(
                policy=navigation,
                redactor=value_redactor,
                timeline=navigation_timeline,
                participant=participant,
                sip_identities=lambda: sip_identities_of(ctx.room),
                # Anyone else in the room can subscribe to the controller's track: only the worker's own recorder is known.
                listeners=lambda: listeners_of(ctx.room, known={evidence_sidecar_identity(participant.identity)}),
                publish=lambda code, digit: ctx.room.local_participant.publish_dtmf(code=code, digit=digit),
                session=lambda: session,
                set_output_enabled=lambda enabled: session.output.set_audio_enabled(enabled),
                fetch_value=lambda name: call.fetch_keypad_value(name),
                config=call.config,
                speaks_text=speaks_text,
                end_call=lambda reason, detail: call_end_arbiter.end(
                    reason, detected_via=DETECTED_VIA_WORKER_TIMER, reason_detail=detail
                ),
                is_finished=lambda: finished.is_set() or room_disconnect.triggered,
                settle_labels=settle_navigation_labels,
                music_active=lambda: bool(answer_detection.music_reading.active),
                # A workflow run advances on any generated reply: nothing here may ask this session to
                # reply. Its menus are answered by the navigator and its pickup by the settling exchange.
                model_feed=cascade_model_feed(lambda: session, lambda: finished.is_set() or room_disconnect.triggered, live=s2s_live_call),
                input_backend=live_input_backend,
                session_hears=session_hears_language,
                # A menu is answered only in one of the call's own languages, and only where none of its sources (the
                # detector's sockets and failovers, its own transcriber or captions) was measured to fail. No
                # perception, no menu is answered.
                menu_language_heard=lambda language: far_end_perception is not None and navigation_menu_language_trusted(
                    load_voice_registry(),
                    (*navigation_plan_sources(far_end_perception.detector.plan),
                     ("captions" if s2s_live_call else "stt", profile.stt_registry, profile.stt_model)),
                    language,
                    far_end_perception.detector.plan.languages,
                ),
                deaf_feed=None if s2s_live_call else cascade_model_feed(
                    lambda: session, lambda: finished.is_set() or room_disconnect.triggered, as_turn=True
                ),
                line_quiet_seconds=answer_detection.far_end_quiet_seconds,
                line_busy=lambda: bool(answer_detection.far_end_audible())
                or (far_end_perception is not None and bool(far_end_perception.unresolved())),
                engine_generate=lambda text: session.generate_reply(instructions=text),
                # Raced against the session's end (``finished`` is the Event the worker sets), so a
                # probe the far end hung up during cannot park the entrypoint above its teardown.
                await_playout=lambda handle: await_playout_or_session_end(handle, ended=finished),
                call_language=lambda: profile.language,
                emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
            )
            # The far end's own stream, transcribed on sockets chosen from the capability
            # artifact for this agent's languages, merged and labelled off the media path.
            # A plan that cannot be made leaves the call on the session STT's finals alone.
            built_perception = build_perception(
                policy=navigation,
                watch=far_end_watch,
                classifier=far_end_classifier,
                environ=os.environ,
                registry=load_voice_registry(),
                call_language=profile.language,
                languages=list(profile.languages or ()),
                session_stt=(profile.stt_registry, profile.stt_model) if profile.stt_model else None,
                # The far end's language is not the agent's: a call that holds no keypad value hears the whole line.
                # One that holds a value hears only the agent's languages, the ones its redactor can judge numbers in.
                whole_line=built_actions is not None and not built_actions.fields,
                own_sources=(("captions" if s2s_live_call else "stt", profile.stt_registry, profile.stt_model),)
                if profile.stt_model else (),
                engine=ENGINE_LIVE if s2s_live_call else ENGINE_CASCADE,
                music_active=lambda: bool(answer_detection.music_reading.active),
                language=lambda: profile.language,
                session_alive=lambda: not finished.is_set(),
                end_call=lambda reason, detail: call_end_arbiter.end(
                    reason, detected_via=DETECTED_VIA_WORKER_TIMER, reason_detail=detail
                ),
                **(
                    {
                        "speaker": built_actions.probe_speaker,
                        "handover": built_actions.handover,
                        "retry_handover": built_actions.retry_handover,
                        "live_ports": built_actions.live_ports(),
                        "invalidate_interaction": built_actions.invalidate,
                        "executor": built_actions.executor,
                        "on_label": built_actions.executor.note_label,
                    }
                    if built_actions is not None
                    else {}
                ),
                value_mode=lambda: value_redactor.value_mode,
                has_numeral=value_redactor.has_numeral,
                emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
            )
            if built_perception is not None:
                far_end_perception, final_router = built_perception
                if built_actions is not None:
                    built_actions.executor.bind_tracker(far_end_perception.tracker)
                    navigation_actions = built_actions
                    navigation_actions.start()
                await far_end_perception.start()
        def caller_transcript_withheld() -> bool:
            """A caller final the transcript gate is holding back: not labelled yet."""

            try:
                return bool(transcript_gate is not None and transcript_gate.gate.holds_final_transcript)
            except NameError:  # the gate is built later in this function
                return False

        # The model's tool seam while the call is in a menu or a queue: a tool is
        # held for the labels in flight and judged on the state they produced;
        # an opening cut off by a menu is a navigation stop, not a failure.
        navigation_gate = NavigationToolGate(
            policy=navigation,
            watch=far_end_watch,
            is_finished=lambda: finished.is_set() or room_disconnect.triggered,
            emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
        )
        # Built here rather than beside the other witnesses below, because the
        # sidecar's frame callback is wired on the next statement and the
        # observer needs it. See the caller_audio_flow module.
        caller_audio_flow = CallerAudioFlowLog(
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            clock=time.monotonic,
        )

        def caller_frame_observer(pcm: bytes) -> None:
            """One caller-track frame, to the classifier and to the flow log.

            The flow log is asked FIRST whether it wants the measurement, and
            the answer is no on a healthy call. That gate is the point: this
            runs on the loop draining the caller's media for the whole call,
            and measuring every frame unconditionally would add two
            allocations per frame for the life of every call -- exactly the
            cost AnswerDetection.note_frame already refuses to pay. Amplitude
            is only worth computing while the session's own input is starved,
            which is when it distinguishes a caller talking into a call that
            is not listening from a caller who has simply stopped speaking.
            """

            answer_detection.note_frame(pcm)
            if far_end_perception is not None:
                # One append to a bounded ring; it never blocks and never raises.
                far_end_perception.feed(pcm)
            # Guarded like note_frame itself (@_never_raises), and for the same
            # reason: this is the media drain, and a diagnostic that measures a
            # call must never be able to break one by raising into it. The
            # classifier is called FIRST and outside this guard so its own
            # behaviour is exactly what it was before this existed.
            try:
                if not caller_audio_flow.wants_energy:
                    return
                measured = frame_amplitude(pcm)
                if measured is not None:
                    caller_audio_flow.note_track_energy(measured)
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass

        # The outbound opening waits for the callee's hello (see the
        # ``opening_turn`` module). The decider is built once the join instant
        # exists, below; its frame tap is wired into the sidecar's caller loop
        # here, and reads nothing until an answer has been observed.
        opening_vad = ctx.proc.userdata.get("vad")
        # Inbound calls are not subject to the rule at all: the caller dialled
        # and is greeted. An outbound call records why it did not wait.
        opening_exclusion: str | None = None
        outbound_opening = opening_policy(call.config.get("opening"))
        if outbound is not None:
            opening_exclusion = callee_first_exclusion(
                managed_pstn=is_managed_pstn_call(call.config),
                speech_to_speech=s2s_live_call,
                live_pickup_supported=voice_frontend(call.config) in {"gpt_live", "gemini_live"},
                enabled=callee_first_enabled(os.environ),
                first_speaker=outbound_opening["first_speaker"],
            )
            if opening_exclusion is None and opening_vad is None:
                opening_exclusion = OPENING_EXCLUDED_NO_VAD
        callee_first = outbound is not None and opening_exclusion is None
        live_pickup = (capture_pickup(
            ctx.room, participant,
            answer_is_known=lambda: participant.attributes.get(SIP_CALL_STATUS_ATTRIBUTE) == SIP_CALL_STATUS_ACTIVE,
        ) if callee_first and s2s_live_call else None)
        opening: dict[str, Any] = {
            "decider": None,
            "stream": None,
            "aligner": None,
            # Seconds of provider silence trimmed from the opening's front.
            "silence_trimmed": [],
            "tasks": [],
            # After the opening is requested, the pickup's own final may still
            # be in flight. It is the pickup until it commits, or until the
            # caller starts speaking again after the request.
            "pickup_pending": False,
            "pickup_committed": asyncio.Event(),
            # Every final transcript of the callee's speech before the opening
            # was dispatched: what the first reply quotes when the pickup's own
            # turn has not committed in time. Bounded.
            "pickup_finals": [],
            # How the first reply knew what the callee said: "committed",
            # "transcript", or "none"; ``None`` when the callee did not speak.
            "pickup_in_context": None,
            # How the first reply was produced (``FIRST_REPLY_*``).
            "first_reply_path": None,
            # Monotonic edges a request-scoped opening recovery reads: the
            # agent's playout starts and the caller's speech starts.
            "agent_speaking_starts": 0,
            "caller_speaking_starts": 0,
        }
        #: Set when the decider releases the opening -- or stops owning it for
        #: any other reason; the opening block reads which from the decider.
        opening_released = asyncio.Event()
        #: Set when the callee took the floor back before the opening was
        #: dispatched; the opening block goes back to waiting.
        opening_resumed = asyncio.Event()

        def opening_feed(pcm: bytes, media_start: float, *, sample_rate: int = 8000) -> None:
            """One caller frame into the decider's own VAD. Never raises.

            ``media_start`` is where the recorder placed the frame on the
            monotonic clock: frames drained from a burst keep their spacing,
            which their receipt instants do not.
            """

            decider = opening["decider"]
            if decider is None or not decider.listening:
                return
            try:
                samples = len(pcm) // 2
                duration = samples / sample_rate
                # Before the stream check: a VAD that failed still leaves the
                # decider a live edge, a first-audio anchor, and so a deadline.
                decider.note_live_edge(until=float(media_start) + duration, start=float(media_start))
                stream = opening["stream"]
                if stream is None:
                    return
                opening["aligner"].push(
                    at=float(media_start),
                    duration=duration,
                    level_dbfs=frame_level_dbfs(pcm),
                )
                stream.push_frame(
                    rtc.AudioFrame(
                        data=pcm, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
                    )
                )
            except Exception:  # noqa: BLE001 - the tap cannot break the media loop
                if live_pickup is not None:
                    live_pickup.buffer.fail("live pickup speech feed failed")

        def media_frame_observer(
            role: str, pcm: bytes, placement: Any, received_at: float
        ) -> None:
            """Both tracks, placed on the recorder's media timeline, for the
            voicemail watch. A bounded append; never raises."""

            try:
                media_start, discontinuous = placement
                if role == "caller" and live_pickup is None:
                    opening_feed(pcm, float(media_start))
                answer_detection.note_media_frame(
                    role,
                    pcm,
                    media_start=float(media_start),
                    received_at=float(received_at),
                    gap=bool(discontinuous),
                )
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass

        sidecar, capture_tasks, evidence_identity = await start_evidence_sidecar(
            room_name=ctx.room.name,
            caller_identity=participant.identity,
            agent_identity=str(
                getattr(getattr(ctx.room, "local_participant", None), "identity", "") or ""
            ),
            recorder=recorder,
            # The caller track is already being drained frame by frame here for the
            # evidence recording; the amplitude the classifier needs comes off the
            # same frames rather than from a second subscription to the same audio.
            on_caller_frame=caller_frame_observer,
            on_media_frame=media_frame_observer,
        )

        def _on_sidecar_disconnected(*_args: Any) -> None:
            sidecar_lost["at"] = time.monotonic()

        sidecar.on("disconnected", _on_sidecar_disconnected)
        # Joining the sidecar yielded again; check before reporting media.
        require_pre_media_participant(ctx.room, participant)
        if outbound is None and browser_test is None:
            # SIP answers when IT subscribes to our output track, not when the
            # recorder subscribes to its input. Prepare the SAME SDK output
            # that the session will use, without generating or sending audio.
            # AgentSession preserves pre-attached output and does not publish
            # a duplicate. This pinned SDK seam is exercised by the SIP gate.
            output_options = RoomOptions().get_audio_output_options()
            inbound_audio_output = _ParticipantAudioOutput(
                ctx.room, sample_rate=output_options.sample_rate,
                num_channels=output_options.num_channels,
                track_publish_options=output_options.track_publish_options,
            )
            await asyncio.wait_for(inbound_audio_output.start(),
                                   timeout=ringing_timeout_seconds(os.environ))
            await require_inbound_sip_answer(
                ctx.room, participant,
                wait_for_active=lambda: wait_for_participant_attribute(
                    ctx.room, identity=participant.identity,
                    attribute=SIP_CALL_STATUS_ATTRIBUTE, value=SIP_CALL_STATUS_ACTIVE),
                timeout_seconds=ringing_timeout_seconds(os.environ),
            )
        call.usage_report_outbox = UsageReportOutbox(
            SESSION_ROOT / "usage-reports", call.call_id,
            alert=lambda reason: capture_exception(RuntimeError(reason), component="voice-worker", phase="usage_outbox"),
        )
        state_path = SESSION_ROOT / f"{safe_component(call.call_id)}.json"
        started_at = utcnow()
        # A session that is not a customer call says so in its own state file, at
        # creation rather than at teardown -- the crash reconciler reads this file
        # precisely in the case where teardown never ran. Without it a platform
        # release canary is indistinguishable from a crashed customer call, and
        # the sweep asks the control plane to close a call id that is not a call
        # row and can only ever answer 404. Only stamped when the session has a
        # purpose, so an ordinary call's state file keeps the shape it has today.
        session_purpose = str(getattr(call, "purpose", "") or "")
        purpose_stamp = {"purpose": session_purpose} if session_purpose else {}
        write_session_state(
            state_path,
            call_id=call.call_id,
            room_name=ctx.room.name,
            participant_identity=participant.identity,
            started_at=started_at,
            last_heartbeat_at=started_at,
            status="media_started",
            **purpose_stamp,
        )
        # Both timers measure from the ANSWER, not from the room. An outbound leg
        # joins the room ~30 ms after the INVITE while it is still dialing, so a
        # room-clock timer would spend a caller's ringing time against their call
        # length and could hang up before anyone said hello.
        call_end_clock: dict[str, float | None] = {
            "answered_at": None,
            # Set where session.input.set_audio_enabled(True) is, below --
            # SIP answer evidence and this worker actually being able to
            # hear the caller are two different instants, and the
            # no-caller-speech timer must be judged from the second one.
            "input_released_at": None,
        }
        # The common zero for the two measurements below: how long since this
        # worker joined the room, which is the only clock available before an
        # answer disposition exists at all.
        joined_at_monotonic = time.monotonic()
        if callee_first:
            opening["decider"] = OpeningTurnTaking(
                origin=joined_at_monotonic,
                listen_window_seconds=outbound_opening["silence_wait_seconds"],
                require_observed_audio=live_pickup is not None,
                emit=lambda event, **details: _quietly(emit, event, call_id=call.call_id, **details),
            )
            if navigation.enabled:
                # A call that turns out to be a menu, a queue or a recording owes nobody an opening: the floor the
                # decider holds for it goes to the navigation, whatever the line is saying meanwhile (a menu that
                # keeps reading never pauses long enough for the opening to be released, and until it is, nothing
                # else may generate: the model could not have answered the menu). A live engine finishes its native
                # pickup where the opening wait returns (see `native_opening_handed_to_navigation`).
                navigation.add_episode_listener(
                    lambda _episode: opening["decider"].note_machine_verdict(
                        "navigation_menu", acting=True, at=time.monotonic()
                    )
                )

        async def run_opening_vad(decider: OpeningTurnTaking, stream: Any, aligner: VadFrameAligner) -> None:
            """The decider's own VAD: every judged frame, in order, into the decider."""

            try:
                async for vad_event in stream:
                    if str(getattr(vad_event, "type", "")) not in ("inference_done", "VADEventType.INFERENCE_DONE"):
                        continue
                    consumed = sum(
                        frame.samples_per_channel / frame.sample_rate
                        for frame in (getattr(vad_event, "frames", None) or [])
                        if getattr(frame, "sample_rate", 0)
                    )
                    speech = float(getattr(vad_event, "probability", 0.0) or 0.0) >= VAD_SPEECH_PROBABILITY
                    for at, duration, is_speech, level in aligner.judged(consumed, speech=speech):
                        action = decider.note_frame(
                            at=at,
                            duration=duration,
                            speech=is_speech,
                            level_dbfs=level,
                            live_edge=aligner.live_edge,
                            observed_at=time.monotonic(),
                        )
                        if native_opening_authorization is not None:
                            if is_speech:
                                native_opening_authorization.note_caller_speech(until=at + duration)
                            if action in (OPENING_TURN_RESUMED, OPENING_TURN_ONSET):
                                native_opening_authorization.resumed(until=at + duration)
                        signal_opening_action(
                            action,
                            released=opening_released,
                            resumed=opening_resumed,
                            pickup_committed=opening["pickup_committed"],
                        )
                    if not decider.listening:
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _quietly(emit, "opening_turn_vad_failed", call_id=call.call_id, error_type=type(exc).__name__,
                         **({"observation_failure": str(exc)} if isinstance(exc, OpeningObservationTimeout) else {}))
                if live_pickup is not None:
                    live_pickup.buffer.fail("live pickup speech detector failed")
            finally:
                opening["stream"] = None
                with contextlib.suppress(Exception):
                    await stream.aclose()

        async def poll_opening(decider: OpeningTurnTaking) -> None:
            """Deadlines no frame can reach: a stalled feed, or the far end gone quiet."""

            try:
                # Until dispatch, not until OPEN: a callee who takes the floor
                # back puts the decider's deadlines in play again.
                while decider.listening:
                    if live_pickup is not None and live_pickup.buffer.error is not None:
                        decider.note_ended(at=time.monotonic())
                        break
                    signal_opening_action(
                        decider.poll(time.monotonic()),
                        released=opening_released,
                        resumed=opening_resumed,
                        pickup_committed=opening["pickup_committed"],
                    )
                    await asyncio.sleep(0.05)
            except OpeningObservationTimeout:
                if live_pickup is None:
                    raise
                live_pickup.buffer.fail("live pickup observation deadline exceeded")
                decider.note_ended(at=time.monotonic())
            finally:
                # Released for every way out -- the opening, an ended call, a
                # machine hand-off -- so the opening block never waits on a
                # decider that will not speak again.
                opening_released.set()

        def start_opening_listen(disposition: str, observed_at: float) -> None:
            """The answer is known: anchor the window and start listening."""

            decider = opening["decider"]
            if decider is None:
                return
            if live_pickup is not None and live_pickup.buffer.answered_at is not None:
                observed_at = min(observed_at, live_pickup.buffer.answered_at)
            decider.note_answer(answer_clock_anchor(disposition), at=observed_at)
            if live_pickup is not None and may_speak(disposition, still_present=True):
                live_pickup.buffer.arm()
            if not decider.owns_floor:
                opening_released.set()
                return
            try:
                stream = opening_vad.stream()
            except Exception as exc:  # noqa: BLE001 - no tap: the poll still opens at W + grace
                _quietly(emit, "opening_turn_vad_failed", call_id=call.call_id, error_type=type(exc).__name__)
                stream = None
                if live_pickup is not None:
                    live_pickup.buffer.fail("live pickup speech detector unavailable")
            if stream is not None:
                aligner = VadFrameAligner()
                opening["aligner"] = aligner
                opening["stream"] = stream
                if live_pickup is not None:
                    # Decide on the same post-answer PCM that the live model
                    # receives, including preserved startup speech. The
                    # recorder remains independent evidence of room playout.
                    resampler = rtc.AudioResampler(input_rate=24000, output_rate=16000, num_channels=1)
                    def observe_live_pickup(frame: Any, received_at: float) -> None:
                        if not decider.listening:
                            return
                        for sampled in resampler.push(frame):
                            duration = sampled.samples_per_channel / sampled.sample_rate
                            opening_feed(bytes(sampled.data), received_at - duration, sample_rate=16000)
                    live_pickup.buffer.observe(observe_live_pickup)
                opening["tasks"].append(asyncio.create_task(run_opening_vad(decider, stream, aligner)))
            opening["tasks"].append(asyncio.create_task(poll_opening(decider)))
        # Populated by watch_for_answer() (outbound only) and by the answer
        # gate's own may_speak branch below; read once at teardown into this
        # call's persisted evidence. A call this gate never watches (inbound,
        # browser test) keeps its NOT_APPLICABLE/None defaults, so a batch can
        # tell "no gate ran" from "the gate ran and never resolved" instead of
        # a missing key doing both jobs -- this is the diagnostic the ringback
        # opening defect had no writer for.
        answer_gate_evidence: dict[str, Any] = {
            "disposition": SIP_ANSWER_NOT_APPLICABLE,
            "wait_ms": None,
            "input_released_ms": None,
            "opening_generate_ms": None,
        }
        # The VAD on every call; on a GPT-Live call also the engine's own captions,
        # because the pinned VAD stayed under threshold on a narrowband PSTN leg
        # while the engine transcribed the caller (2026-09-15 batch), and the SDK
        # keeps a client VAD's user_state apart from the realtime input-speech
        # event. See CallerSpeechWitness.
        caller_speech_witness = CallerSpeechWitness(
            captions_are_evidence=s2s_live_call,
            # The third witness, on an engine whose unasked generations are
            # server turns (Gemini Live today; declared per engine, never per
            # language or carrier). The turn recorder below reports them.
            engine_turns_are_evidence=engine_reports_caller_turns(call.config),
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            seconds_since_answer=lambda: (
                None
                if call_end_clock["answered_at"] is None
                else time.monotonic() - call_end_clock["answered_at"]
            ),
        )
        # The same answer clock as the silence rule above, so the two rows
        # describe one call from one zero.
        caller_audio_bind_witness = CallerAudioBindWitness(
            expected_identity=participant.identity,
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            seconds_since_answer=lambda: (
                None
                if call_end_clock["answered_at"] is None
                else time.monotonic() - call_end_clock["answered_at"]
            ),
        )

        # The second CONTINUOUS record the witnesses above cannot keep (the
        # first is caller_audio_flow, built before the sidecar). Both are
        # diagnostic only: nothing here ends, alters or bills a call. The
        # absence of recognised speech is not evidence of the absence of a
        # person, so these make such a call countable and never act on one.
        # See the caller_audio_flow module.
        caller_vad_edges = CallerVadEdgeLog(
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        )

        def read_caller_audio_binding() -> CallerAudioBinding | None:
            """RoomIO's caller linkage, or ``None`` when it cannot be read.

            ``session.room_io`` raises before start and after close; ``None`` is
            "we could not tell", which the witness reports as ``bound`` false.
            """

            try:
                return caller_audio_binding_from_room_io(
                    session.room_io, microphone_source=rtc.TrackSource.SOURCE_MICROPHONE
                )
            except Exception:  # noqa: BLE001 - a measurement is never a media gate
                return None

        # Rule 4. The control plane cannot reach LiveKit at all, so an operator's
        # hangup arrives as an INTENT in the lifecycle-heartbeat response -- the one
        # channel that was demonstrably alive for all 94 minutes of the incident.
        hangup_intent: dict[str, Any] = {}
        funded_runtime = FundedRuntime(call.config.get("funded_runtime") or {})

        async def report_sip_answered(observed_at: str) -> None:
            await report_call_answered(call=call, call_end_clock=call_end_clock,
                open_entry_window=navigation.open_entry_window, funded_runtime=funded_runtime,
                provider_call_id=provider_call_id, observed_at=observed_at)

        # The inbound SIP answer was explicitly witnessed above. A browser
        # session has no SIP handshake. An OUTBOUND leg joins the
        # room ~30 ms after the INVITE while it is still dialing, so this instant
        # is 5-18 s early; that report moves to the answer gate below, where the
        # carrier's own answer signal arrives. `room.media_started` deliberately
        # stays put: it is what keeps the call's status transition and the
        # settlement window unchanged by this fix, which is a truthfulness fix
        # and not a billing change.
        if outbound is None:
            await report_sip_answered(started_at)
        await asyncio.to_thread(
            call.report_lifecycle,
            event_id=f"{call.call_id}:media-started",
            event_type="room.media_started",
            source_component="livekit_room",
            observed_at=started_at,
            details={"room_name": ctx.room.name},
        )
    except BrowserSessionAlreadyOwned:
        if governed_root is not None and governed_root.started:
            raise
        # LiveKit normally assigns one job per room, but a churn/re-dispatch
        # loser must not report sip.failed or delete the room owned by the
        # worker whose claim already committed. Returning disconnects only this
        # worker job; the owner remains responsible for terminal lifecycle.
        emit("workspace_browser_test_claim_already_owned", call_id=pre_media_call_id)
        return
    except BaseException as exc:
        if governed_root is not None and governed_root.started:
            raise
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
        #
        # Read from the exception rather than from the stage: a refusal that
        # knows which of its own rules fired says so.
        #
        # Read ONLY from this worker's own class, never by duck-typing a
        # ``reason`` attribute. This handler catches every exception type that
        # can reach it, and ``reason`` is a common attribute name on other
        # people's: a ``websockets`` ``ConnectionClosed`` carries the peer's
        # close reason there. Publishing that would let a provider choose the
        # text on a customer's call -- and even bounded by
        # ``safe_pre_media_reason`` it would replace a specific stage reason
        # with a vaguer one.
        handled = await settle_pre_media_failure(exc, ctx=ctx, participant=participant,
            call=call, adapter=adapter, pre_media_call_id=pre_media_call_id,
            pre_media_stage=pre_media_stage, browser_test=browser_test,
            browser_participant_mismatch=browser_participant_mismatch, inbound_route=inbound_route,
            live_pickup=live_pickup, report_browser_test_failure=report_browser_test_failure,
            state_path=state_path, sidecar=sidecar, capture_tasks=capture_tasks,
            inbound_audio_output=inbound_audio_output, gpt_live_http=gpt_live_http,
            native_model=gpt_live_llm,
            emit=emit, inbound_release_canary=INBOUND_RELEASE_CANARY,
            remove_participant=lambda: remove_sip_participant_before_media(ctx, participant))
        if handled:
            return
        raise
    finally:
        # Provider instances hold their own configuration, but the transient
        # response/overlay mappings must not remain in this worker scope.
        if leased_components is not None:
            del leased_components
        # A follow-caller call builds a language switcher below that closes
        # over this same name to rebuild a credentialed stack whenever the
        # caller later switches -- possibly minutes from now. Deleting the
        # local here unconditionally used to remove the binding out from
        # under that later closure: a nested function resolves a free
        # variable at CALL time, not at the time it was defined, so the very
        # first live switch on any follow-caller call hit a NameError
        # (Python raises it as "free variable ... referenced before
        # assignment") the instant it tried to read the deleted name -- every
        # switch, not only ones landing on a particular provider. Keep it
        # alive for the rest of this call when a switcher will need it; it is
        # released with everything else in this frame when the call ends.
        if follow_caller is None and runtime_environ is not None:
            del runtime_environ

    def callee_attributes() -> dict[str, str] | None:
        """The leg's live attributes, or ``None`` once it has left.

        ``None`` means gone. A participant that is present but carries no
        ``sip.callStatus`` returns an empty mapping instead: the gate must not
        read those two the same way, or a callee who hung up while the phone
        was ringing would still be greeted.
        """

        leg = ctx.room.remote_participants.get(participant.identity)
        return None if leg is None else dict(leg.attributes or {})

    # Watching starts HERE, at room join -- several seconds before the opening
    # turn, because provider warm-up sits between them. A callee who answers
    # and hangs up inside that window has genuinely answered and is owed the
    # lifecycle report; starting the watch at the greeting would miss them, and
    # would miss an answer followed by a session-start failure entirely.
    #
    # Deliberately AFTER the `room.media_started` report above: settlement
    # takes the FIRST of `call.answered`/`media.started` by insertion order, so
    # awaiting that write before this task can exist is what keeps the billed
    # window byte-identical to what it was before this gate existed.
    answer_watch: asyncio.Task[str] | None = None
    if outbound is not None:

        async def watch_for_answer() -> str:
            return await watch_outbound_answer(await_answer=await_callee_answer,
                participant_attributes=callee_attributes,
                wait_for_active=lambda: wait_for_participant_attribute(ctx.room, identity=participant.identity,
                    attribute=SIP_CALL_STATUS_ATTRIBUTE, value=SIP_CALL_STATUS_ACTIVE),
                timeout_seconds=ringing_timeout_seconds(os.environ), start_opening_listen=start_opening_listen,
                answer_gate_evidence=answer_gate_evidence, joined_at_monotonic=joined_at_monotonic,
                emit=emit, call_id=call.call_id, arm_answer_detection=answer_detection.arm,
                report_sip_answered=report_sip_answered)

        answer_watch = asyncio.create_task(watch_for_answer())

    async def heartbeat() -> None:
        sequence = 0
        while True:
            await asyncio.sleep(5)
            sequence += 1
            observed_at = utcnow()
            try:
                lifecycle_response = await asyncio.to_thread(
                    call.report_lifecycle,
                    event_id=f"{call.call_id}:heartbeat:{sequence}",
                    event_type="room.media_heartbeat",
                    source_component="livekit_room",
                    observed_at=observed_at,
                    details={"room_name": ctx.room.name},
                    funding_generation=funded_runtime.request_generation(),
                )
            except Exception as exc:
                funded_runtime.poll_failed()
                emit("call_heartbeat_failed", call_id=call.call_id, error=type(exc).__name__)
                continue
            funded_runtime.adopt(lifecycle_response)
            # This response body was discarded until now, so nothing depended
            # on its shape. The intent is re-served on every poll until the
            # call is terminal, so seeing it repeatedly must be harmless: the
            # arbiter ends the call once and the acknowledgement is a
            # compare-and-set the loser of which does nothing.
            # Keyed on the attempt number, not on holding something already:
            # the same pending request rides EVERY heartbeat until it settles,
            # and a re-arm after a failed teardown is a new number that must
            # supersede the old one rather than be ignored.
            if adopt_hangup_intent(
                hangup_intent,
                pending_hangup_intent(lifecycle_response),
                call_id=call.call_id,
                emit=lambda event, **details: emit(event, **details),
            ):
                emit(
                    "hangup_intent_received",
                    call_id=call.call_id,
                    attempt=hangup_intent["attempt"],
                )
                # At RECEIPT. Leaving this to the watchdog poll cost an
                # operator's reason on a real call.
                operator_asked_to_end()
            write_session_state(
                state_path,
                call_id=call.call_id,
                room_name=ctx.room.name,
                participant_identity=participant.identity,
                started_at=started_at,
                last_heartbeat_at=observed_at,
                status="media_started",
                **purpose_stamp,
            )

    # Whether this call can ever silence its agent: the SETTING, plus a watch
    # that can actually arm.  ``acts`` is false for every call at this moment,
    # because the answer gate has not resolved yet.
    silence_the_agent_when_nobody_is_there = (
        answer_detection.acting_mode and answer_watch is not None
    )
    # Whether the caller can hear the agent.  Toggled, never latched: the
    # floor rule is released by a pause and a flat-tone verdict by human
    # evidence, and this follows both.
    agent_is_audible = {"now": True}

    def gate_the_agents_voice(*, suppressed: bool) -> None:
        """Do not hold a conversation with a song.

        Read at the start of the next generation, so the sentence already
        playing finishes and the turn after it produces no audio.  Cascade
        skips synthesis outright; voice-to-voice still generates and is still
        billed, and what changes there is that the far end stops hearing an
        answer to a song.

        Only transitions reach the session, and the flag moves only after the
        call returned, so a failed mute is retried rather than recorded.  In
        shadow and in off this returns before naming ``session`` at all.
        """

        if navigation_actions is not None:
            # The output gate is the only writer of this switch while the agent
            # navigates: this request is one of its inputs, not a second writer.
            navigation_actions.gate.set_arbiter_acts(silence_the_agent_when_nobody_is_there)
            navigation_actions.gate.request_silence(suppressed)
            return
        if (
            not silence_the_agent_when_nobody_is_there
            or suppressed != agent_is_audible["now"]
        ):
            return
        session.output.set_audio_enabled(not suppressed)
        agent_is_audible["now"] = not suppressed

    def follow_navigation_evidence() -> None:
        """Move the call into or out of a phone menu / hold queue.

        From evidence the detectors already produce, read on every classify
        tick and acted on at an EDGE only (see ``NavigationPolicy``). A person
        the observer confirms ends the episode; the arbiter then measures from
        that moment, because what it concluded from the queue is not about the
        person who picked up. Inert unless the agent opted in.
        """

        outcome = follow_far_end(
            navigation,
            observer=answer_detection.voicemail.observer,
            arbiter=answer_detection.arbiter,
            music=answer_detection.music_reading,
        )
        if outcome == "exited":
            leave_episode(navigation, answer_detection)

    def end_for_recorded_message_only() -> None:
        """S8: only recordings were heard, none of them a menu or a queue, then quiet.

        Valid only in a conversation that never entered an episode (a hold's
        silence between announcements is not a recording ending), with nothing
        in flight at the classifier and no person evidence -- the policy's own
        or the observer's confirmed tier. The post-answer arbiter's own verdict
        is not asked (it calls any recording with a word a caller). The ending
        is the arbiter's own, through the same first-wins seam as every other.
        """
        if far_end_watch is None or far_end_watch.unresolved:
            return
        if not navigation.recorded_only_due(
            human_seen=observer_saw_a_person(answer_detection.voicemail.observer),
            unresolved=bool(far_end_watch.unresolved),
        ):
            return
        call_end_arbiter.end(
            TERMINATION_NO_HUMAN_AFTER_ANSWER,
            detected_via=DETECTED_VIA_WORKER_TIMER,
            reason_detail="recorded_message_only",
        )

    async def classify_answer() -> None:
        """Re-read the post-answer window on its own cadence.

        Its OWN task: the frame loop must not be made to wait on anything, and
        the watchdog polls far too slowly for the windows this reads.

        In ``end_call`` it ends the call, and only then -- every decision
        carries whether it may be acted on, and in shadow that flag is false.
        Exceptions are swallowed for the same reason the evidence recorder's
        are: measuring a call must not be able to end one by failing.
        """

        while True:
            await asyncio.sleep(ANSWER_CLASSIFY_INTERVAL_SECONDS)
            try:
                # The witness is read INSIDE the guard: a raise from it out
                # here would report every second for the rest of the call, and
                # kill this task while the frame loop keeps filling a buffer
                # only ``tick`` trims.
                decision = answer_detection.tick(
                    caller_has_spoken=caller_speech_witness.heard,
                    navigation_suspended=navigation.suspends(ENDER_ARBITER),
                )
                follow_navigation_evidence()
                end_for_recorded_message_only()
                try:
                    gate_the_agents_voice(
                        suppressed=answer_detection.arbiter.silence_the_agent
                    )
                except Exception:  # noqa: BLE001 - a mute cannot hold an ending
                    # Its own guard, because it runs before the ending below.
                    # Sharing the loop's guard meant a session closing under it
                    # took the whole tick with it, and the ending that tick was
                    # about waited for the next one.
                    _quietly(
                        emit,
                        "answer_detection_mute_failed",
                        call_id=call.call_id,
                    )
                if decision is not None and decision.acts and navigation.suspends(ENDER_ARBITER):
                    # Vetoed here, on the tick the call entered an episode: the
                    # ending never happened, so it is not what the call ends on.
                    answer_detection.arbiter.discard_decision()
                if decision is not None and decision.acts and not navigation.suspends(ENDER_ARBITER):
                    # Read again HERE, at the ending's own call site: the state
                    # may have moved since the tick, and a call waiting in a
                    # menu or a queue is never ended by the non-human rules.
                    # Through the same compare-and-set seam every other ending
                    # uses, so this cannot become a second terminal reason on a
                    # call something else already ended. First reason wins and
                    # the rest are no-ops.
                    call_end_arbiter.end(
                        decision.reason, detected_via=DETECTED_VIA_WORKER_TIMER
                    )
                if decision is not None and opening["decider"] is not None:
                    # An acting verdict takes the call from the opening for
                    # good; a shadow one is only recorded on its summary.
                    opening["decider"].note_machine_verdict(
                        str(decision.reason),
                        acting=bool(decision.acts) and not navigation.suspends(ENDER_ARBITER),
                        at=time.monotonic(),
                    )
                # After the answer-detection ending and before the rows: an
                # ending decided on this tick must not wait for a diagnostic.
                # It runs on the same compare-and-set seam, so whichever of
                # the two decided first is the one the customer is told.
                model_unresponsive.poll(time.monotonic())
                # Right after it, on the same seam: the two rules own
                # disjoint silences (the caller spoke last vs the agent did).
                mutual_silence.poll(time.monotonic())
                # Last in the tick: a row about the binding must not delay an
                # ending decided on the same tick.
                caller_audio_bind_witness.report(read_caller_audio_binding())
                # Same position, same reason. The flow log can only see a
                # starved window from a tick: the thing that would report the
                # starvation is the frame that never arrived, so a recorder
                # driven purely by frames can never observe its own absence.
                caller_audio_flow.poll()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never a media gate
                # This handler exists for the day the session's own guard
                # fails. If reporting that day also raised, the task would die
                # here and classification would stop for the rest of the call
                # with nothing said about it.
                try:
                    emit(
                        "answer_classification_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                    )
                except Exception:  # noqa: BLE001 - reporting cannot fail louder
                    pass

    # The caller spoke and the agent produced nothing for the whole bound.
    #
    # Wired here, before the session exists, because the poll loop above is
    # what drives it and the loop is created on the next line. Every gate is
    # late-bound and every one of them is read through a guard, so a tick that
    # lands before the name it reads has been assigned answers the gate's own
    # safe default rather than killing the loop.
    #: The tool calls the SDK has started and not yet ended, by call id.
    tool_execution: dict[str, set[str]] = {"open": set()}
    # Installed on the provider session once it exists (after session start);
    # until then, and on any engine without the hooks, it reports unsupported.
    duplex_liveness = DuplexLivenessWitness(relative_ms=recorder.relative_ms)

    def emit_model_unresponsive(event: str, **details: Any) -> None:
        if s2s_live_call and event == MODEL_UNRESPONSIVE_DECISION_EVENT:
            # What reached the provider and what came back, beside the
            # decision it explains. Counts and times only.
            try:
                details = {**details, **duplex_liveness.decision_details()}
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass
        emit(event, call_id=call.call_id, **details)

    def recover_native_reply() -> bool:
        # GPT-Live's witness proves input reached the engine and that no
        # delegated response is still running. Other engines keep their
        # existing watchdog behavior until that contract is supported.
        details = duplex_liveness.decision_details()
        if (
            not details.get('duplex_input_heard_since_output') or details.get('duplex_delegation_open') or details.get('duplex_unsent_appends') or (details.get('duplex_backend_tool_choice') == 'none') or (not s2s_live_call) or (not any((isinstance(tool, dict) and tool.get('name') == 'end_call' and (tool.get('type') == 'builtin') for tool in call.config.get('tools', []))))
        ):
            return False
        return request_native_reply_recovery(
            session_agent.realtime_llm_session,
            emit=lambda event, **fields: emit(event, call_id=call.call_id, **fields)
        )

    model_unresponsive = ModelUnresponsiveRule(
        recover_reply=recover_native_reply,
        seconds=model_unresponsive_seconds(call.config),
        # The same compare-and-set seam every other ending uses. Its own
        # reason, `agent_unresponsive`: this clock observes an agent that
        # stopped answering, never an engine failure, so it must not say
        # `model_session_lost` -- that is the SDK close handler's to report.
        end_call=lambda: call_end_arbiter.end(
            TERMINATION_AGENT_UNRESPONSIVE,
            detected_via=DETECTED_VIA_WORKER_TIMER,
            reason_detail=TERMINATION_DETAIL_MODEL_UNRESPONSIVE,
        ),
        emit=emit_model_unresponsive,
        # The room's own agent edge, NOT the runtime's agent state and NOT a
        # gate. A voice-to-voice runtime was measured holding that state at
        # "speaking" with its provider gone -- forty-seven polls, no audio,
        # nothing ended. The sidecar is an independent room subscriber whose
        # edge advances only on audible agent frames and skips the ambience
        # track the fillers play on; the rule reads it as OUTPUT, so a reply
        # the caller can hear clears the window instead of postponing it.
        agent_audible_at=lambda: recorder.agent_audible_until(),
        # And whether that edge means anything: a sidecar delivering no frames
        # on any track is not observing, and a frozen edge then says nothing
        # about what the room heard.
        agent_audio_observed=lambda: (
            sidecar_lost["at"] is None
            and agent_audio_observed_recently(
                recorder.observer_last_frame_at(), now=time.monotonic()
            )
        ),
        tool_in_flight=lambda: bool(tool_execution["open"]),
        # A workflow turn is a round trip to the control plane, which owns the
        # reply; the model is not the thing being waited on.
        workflow_turn_pending=lambda: (
            False
        ),
        # A call waiting in a phone menu or a hold queue owes nobody a reply.
        # An episode re-clocks the rule; the entry window alone only defers it.
        navigation_suspended=lambda: (
            navigation.in_episode and navigation.suspends(ENDER_MODEL_UNRESPONSIVE)
        ),
        deferred=lambda: navigation.suspends(ENDER_MODEL_UNRESPONSIVE),
        # LAMBDAS, not the bound methods. Every gate here is late-bound on
        # purpose -- this rule is built before the poll loop that drives it,
        # and that loop is created before the session, the admission and the
        # arbiter exist. Passing `semantic_admission.admits_transcript`
        # directly evaluates the NAME here, and the name is not bound yet:
        # measured on the local call harness, every call on the stack died
        # with `UnboundLocalError: semantic_admission` before it spoke a word.
        # Inside a lambda the same names resolve at poll time, when they do
        # exist. The rule's own guard around each gate is the second line of
        # defence and not a substitute for this one.
        admits=lambda: semantic_admission.admits_transcript(),
        is_ending=lambda: finished.is_set(),
    )
    # Both sides silent after the agent spoke: check in once, then end
    # politely. Built here for the same reason as the rule above --
    # the poll loop drives it -- and every name it reads is late-bound.
    def mutual_silence_hold() -> str:
        """Why the silence clock must not run right now, or ``""``."""

        if finished.is_set() or not semantic_admission.admits_transcript():
            # An ending is already in flight; it owns the call.
            return "call_ending"
        if navigation.suspends(ENDER_MUTUAL_SILENCE) or navigation.suspends(SPEECH_CHECK_IN):
            # The silence of a queue is not mutual silence. Re-starts the
            # episode rather than freezing it (see ``HOLD_NAVIGATION``); while
            # only the entry window defers it, an ordinary freeze.
            return HOLD_NAVIGATION if navigation.in_episode else HOLD_ENTRY_WINDOW
        if tool_execution["open"]:
            return "tool_in_flight"
        if language_switcher is not None and language_switcher.switch_in_flight:
            return "language_switch"
        if answer_detection.arbiter.silence_the_agent:
            # The agent is muted while the far end is classified: a check-in
            # now would not be heard.
            return "agent_muted"
        if sidecar_lost["at"] is not None or not agent_audio_observed_recently(
            recorder.observer_last_frame_at(), now=time.monotonic()
        ):
            # A frozen agent edge from a sidecar that stopped observing says
            # nothing about what the room heard.
            return "agent_audio_unobserved"
        return ""

    def speak_mutual_silence_check_in() -> Any:
        return speak_check_in(
            session,
            speaks_text=speaks_text,
            scripted=False,
            language=(
                language_switcher.active_language
                if language_switcher is not None
                else profile.language
            ),
        )

    def end_for_mutual_silence() -> bool:
        if not supports_manual_live_replies(call.config):
            return call_end_arbiter.end(
                TERMINATION_CALLER_UNRESPONSIVE, detected_via=DETECTED_VIA_WORKER_TIMER
            )
        # Through the worker-owned goodbye, with this rule's own reason. The
        # first ending wins: a model end_call already running keeps its own.
        return end_call.terminate(
            finish=lambda: call_end_arbiter.end(
                TERMINATION_CALLER_UNRESPONSIVE, detected_via=DETECTED_VIA_WORKER_TIMER
            ),
            farewell_instruction=SILENCE_FAREWELL_INSTRUCTION,
        )

    mutual_silence = MutualSilenceRule(
        check_in=lambda: speak_mutual_silence_check_in(),
        end=lambda: end_for_mutual_silence(),
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        agent_audible_at=lambda: recorder.agent_audible_until(),
        hold=lambda: mutual_silence_hold(),
        # A workflow's replies are scripted: a caller's "yes" to a check-in
        # would be delivered to the pending business step as its answer.
        # The scripted check-in owns that path; here such a call ends at
        # the same N+M.
        can_check_in=supports_manual_live_replies(call.config),
        mode=mutual_silence_mode(os.environ),
    )

    def note_tool_execution(event: Any) -> None:
        """Track which tool calls are open, for the silent-model rule.

        Its own reader of the pinned ``ToolExecutionUpdatedEvent`` rather than
        a second consumer of the tool-wait feedback's state: that component
        exists to decide whether the CALLER hears anything and drops tools it
        never covers, so borrowing its open set would hand this rule a window
        that closes while a tool it deliberately ignores is still running.

        A tool STARTING is also agent output -- the model reached for
        something, which is the thing this rule asks whether it can still do.
        """

        try:
            update = getattr(event, "update", None)
            kind = str(getattr(update, "type", "") or "")
            if kind == "tool_call_started":
                call_ref = getattr(update, "function_call", None)
                key = str(getattr(call_ref, "call_id", "") or "")
                if key:
                    tool_execution["open"].add(key)
                model_unresponsive.note_agent_output(OUTPUT_TOOL, at=time.monotonic())
            elif kind == "tool_call_ended":
                tool_execution["open"].discard(str(getattr(update, "call_id", "") or ""))
        except Exception:  # noqa: BLE001 - never break a media callback
            _quietly(emit, "tool_execution_tracking_failed", call_id=call.call_id)

    async def watch_voicemail() -> None:
        """Drain the voicemail watch's frame rings on their own clock.

        Its own task, at a tenth of a second, because a record beep is judged
        from the frames around it and the one-second classify tick is too
        coarse to place it.  It measures and records; it never ends, mutes or
        advances anything.  ``process_voicemail`` never raises, and the sleep
        is the only await, so cancellation lands cleanly.
        """

        while True:
            await asyncio.sleep(VOICEMAIL_PROCESS_INTERVAL_SECONDS)
            answer_detection.process_voicemail()

    classify_task = asyncio.create_task(classify_answer())
    voicemail_task = (
        asyncio.create_task(watch_voicemail())
        if answer_detection.voicemail.enabled and answer_watch is not None
        else None
    )
    heartbeat_task = asyncio.create_task(heartbeat())

    session_options = session_options_with_managed_pstn_turn_taking(
        # Keep an outer copy before applying the per-call policy.  The demo
        # stack is shared by the browser and managed-call paths.
        dict(stack.session_options),
        realtime_config=call.config,
    )
    if navigation.enabled:
        session_options = without_preemptive_generation(session_options)
    min_speech_clamp = managed_pstn_min_speech_clamp(call.config)
    if min_speech_clamp is not None:
        # Once per call, not per turn: a customer who asked for 120 ms and
        # hears 150 ms finds the reason in this call's own evidence instead of
        # concluding the setting is broken.
        requested_seconds, effective_seconds = min_speech_clamp
        emit(
            "min_speech_ms_clamped_to_pstn_floor",
            call_id=call.call_id,
            requested_ms=int(round(requested_seconds * 1_000)),
            effective_ms=int(round(effective_seconds * 1_000)),
        )
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
    if voice_frontend(call.config) == "gpt_live":
        # This entrypoint installs DuplexOpeningHold before enabling input.
        # It owns the opening's mute and quiet-tail recovery. The SDK's
        # independent browser/inbound AEC warm-up would substitute silence
        # BEFORE that hold receives frames, so a retained reply would already
        # be empty. Outbound SIP already disables that SDK warm-up by default.
        session_options["aec_warmup_duration"] = None
    session = AgentSession(**session_options)
    from action_inputs import ACTION_INSTRUCTIONS, bind_caller_inputs
    caller_inputs = bind_caller_inputs(session, call)
    confirmation_listener = None
    if inbound_audio_output is not None:
        session.output.audio = inbound_audio_output
    if transcript_publisher is not None:
        install_transcript_events(session, transcript_publisher)
    if transcript_event_recorder is not None:
        # Registered independently of the publisher above: the pinned SDK
        # keeps per-event handlers in a set, so nothing here may depend on
        # which of the two runs first.
        install_call_transcript_events(session, transcript_event_recorder)
    # One predicate decides both halves of the interruption contract: the
    # session interruption threshold above and the caller-transcript gate
    # below. Every managed phone call gets both, in any language -- the
    # transcript interrupt path the gate closes is reached identically by a
    # VAD-only stack. A browser test, the simulations lab, the standalone demo
    # and the platform release canary get neither and keep their existing
    # behaviour byte for byte. (Endpointing is the one part that stays on the
    # English stack; the gate never reads it.)
    managed_pstn_interruption = is_managed_pstn_call(call.config)
    # The per-agent word requirement, and what this exact stack can enforce of
    # it. The pair is resolved ONCE, here, and handed to every consumer --
    # the session options below, the transcript gate that counts the words,
    # the gate that holds the floor, and the telemetry that reports both
    # numbers. Two independent readings of the same setting are how a session
    # ends up tuned by one and gated by the other.
    configured_word_requirement = configured_min_interruption_words(call.config)
    # Read from the BUILT session options, not resolved a second time: the
    # session option the holder depends on is installed by the same function
    # that built them, and a shape it declined to edit must not be armed
    # against. See installed_min_interruption_words.
    word_requirement = installed_min_interruption_words(
        session_options, realtime_config=call.config
    )
    if (
        configured_word_requirement > 0
        and word_requirement == 0
        and managed_pstn_interruption
        and effective_min_interruption_words(call.config) > 0
    ):
        # Enforceable on this stack, but the session this call built could not
        # carry the option it depends on. Distinct from the transcriber case
        # below, and a different thing for an operator to go and look at.
        emit(
            "min_interruption_words_not_installed",
            call_id=call.call_id,
            configured_words=configured_word_requirement,
        )
    elif configured_word_requirement > 0 and word_requirement == 0 and managed_pstn_interruption:
        # Once per call, and it names the reason rather than the setting: an
        # agent whose interruptions were not behaving as configured is exactly
        # the support question this line has to answer on its own.
        emit(
            "min_interruption_words_ignored_no_interims",
            call_id=call.call_id,
            configured_words=configured_word_requirement,
            language=str(call.config.get("language") or ""),
        )
    interruption_word_gate = InterruptionWordGate(min_words=word_requirement)
    # The end_call guard's caller evidence is fed by the transcript gate (every
    # speech-carrying event as it enters, every final it forwards), so it is
    # built here, before the gate. Armed on managed-PSTN cascade calls whose
    # replies the model generates: a workflow run's replies are scripted and a
    # voice-to-voice engine speaks its goodbye before the tool. ``off`` builds
    # nothing, so an unarmed call runs byte for byte as before.
    end_call_guard_mode, end_call_guard_mode_moved = effective_guard_mode(
        resolve_guard_mode(os.environ), mutual_silence.mode
    )
    if end_call_guard_mode_moved:
        # Enforcing without a silence rule that can end a refused call is the
        # one configuration the rollout rule forbids: the guard only logs.
        emit(
            "end_call_guard_mode_downgraded",
            call_id=call.call_id,
            requested=resolve_guard_mode(os.environ),
            mode=end_call_guard_mode,
            reason=end_call_guard_mode_moved,
        )
    # The close floor's measured first-word budget: the playout hold's own
    # per-stack table, under the SAME predicate the hold resolves it with (the
    # installed word requirement is what keeps the transcriber hearing the
    # caller while the farewell is uninterruptible). A follow-caller switch
    # away from the measured language leaves it unmeasured for the rest of the
    # call's attempts; every other source keeps raw VAD on the close.
    close_floor_transcriber = call.config.get("transcriber")
    close_floor_transcriber = close_floor_transcriber if isinstance(close_floor_transcriber, Mapping) else {}
    close_floor_language = str(call.config.get("language") or "").strip().lower()
    close_floor_budget_seconds = no_word_hold_budget_seconds(
        stt_registry_name=close_floor_transcriber.get("registry_name"),
        stt_model=close_floor_transcriber.get("model"),
        language=close_floor_language,
        word_requirement=word_requirement,
        voice_to_voice=s2s_live_call,
    )

    def close_floor_budget() -> float | None:
        if close_floor_budget_seconds is None:
            return None
        active = language_switcher.active_language if language_switcher is not None else close_floor_language
        if str(active or "").strip().lower() != close_floor_language:
            return None
        return close_floor_budget_seconds

    close_floor_backchannels = caller_backchannel_tokens(call.config)
    # One terminal owner for this call: the agent's close and a transfer
    # arbitrate here, on every engine and in both directions.
    terminal_intent = TerminalIntent()
    end_call_evidence: CallerEvidence | None = (
        CallerEvidence(wordless_runs_count=lambda: not end_call.close_floor_measured(), counts_as_words=lambda text: end_call.caller_counts_as_words(text)) if managed_pstn_interruption and speaks_text and (not s2s_live_call) and (end_call_guard_mode != GUARD_MODE_OFF) else None
    )
    #: Set while the caller is not speaking; a deferred guard decision waits on it.
    caller_run_ended = asyncio.Event()
    transcript_gate = (
        CallerTranscriptGateStream(
            CallerTranscriptGate(
                min_interruption_words=word_requirement,
                observer=end_call_evidence,
                # English backchannels, applied only to an English call. An
                # Estonian Deepgram stack reaches the word gate too, and
                # handing it this list would refuse nothing while the public
                # contract said listening noises do not count.
                backchannel_tokens=caller_backchannel_tokens(call.config),
                # The floor THIS session runs, from the same built options the
                # runtime is handed. A high-sensitivity agent runs 0.15 s in
                # any language; a word gate holding out for the managed 0.50 s would
                # refuse interruptions that agent's own configuration allows.
                min_duration_seconds=configured_interruption_min_duration(session_options),
                # Whether this call's recogniser measures confidence at all.
                # Read by registry name, the same key the word gate resolves a
                # transcriber by; unknown registries report nothing rather
                # than republishing a plugin's placeholder as a measurement.
                stt_reports_confidence=stt_registry_reports_confidence(
                    call.config.get("transcriber", {}).get("registry_name")
                    if isinstance(call.config.get("transcriber"), Mapping)
                    else ""
                ),
            ),
            on_interruption_qualified=lambda words: _release_interruption_word_gate(
                interruption_word_gate,
                by_words=True,
                words_heard=words,
                session=session,
                call_id=call.call_id,
            ),
            on_caller_transcript=lambda text: end_call.note_caller_transcript(text),
        )
        if managed_pstn_interruption
        else None
    )
    if transcript_gate is not None:
        # Armed here, at construction, rather than at the say call site the
        # opening protection itself arms at: the loop is already running (this
        # function is a coroutine), so this is a zero-window arm that covers
        # every path that can speak the opening -- free-form, workflow,
        # GPT-Live -- and the gap between them, the answer-detection warm-up
        # and any control-plane round trip the opening's own advance takes.
        # Before this, a caller turn that reached ``llm_node`` before the
        # opening was issued was refused outright with a dedicated exception,
        # discarding "who is this?", "stop", "wrong person" along with every
        # ordinary pickup. Now no final transcript reaches the SDK at all
        # until the opening closes this same window (the say's own finally,
        # or the pickup-release timer), exactly as a mid-opening pickup is
        # already held -- so it takes the ordinary turn path afterwards
        # instead of being dropped. The say/generate-reply call sites still
        # set this flag themselves right before speaking; by then it is
        # already ``True`` and that call is a no-op transition. Every path
        # that can leave without ever reaching a say -- the callee already
        # gone, a non-terminal workflow reply with no text -- closes this
        # same window explicitly (see the backstop right after this whole
        # ``if may_speak`` block, and ``speak_opening_turn``'s own finally),
        # so nothing is left to the gate's wedge recovery
        # (``CallerTranscriptGate.release_expired``) to unstick on its own.
        #
        # Not on a callee-first outbound call: there the pickup is not held
        # behind an opening, it comes BEFORE the opening and is what the
        # opening answers. Its finals reach the SDK as they arrive, and the
        # agent commits them to context without replying (``EvidenceAgent``'s
        # opening ownership) until the decider releases the opening.
        if not callee_first:
            transcript_gate.set_opening_turn_in_flight(True)
    # Both halves of "the agent must sound natural", both admitted by
    # the SAME predicate as the transcript gate so a browser test, the
    # simulations lab, the standalone demo and the platform release canary
    # keep their existing opening line and their existing pause behaviour.
    opening_protection = OpeningTurnProtection(enabled=managed_pstn_interruption)
    boundary_stop = BoundaryStopPolicy(
        tail_seconds=DEFAULT_BOUNDARY_TAIL_SECONDS if managed_pstn_interruption else 0.0
    )
    # The third half of "the agent must sound natural", and the one the
    # founder actually hears: when the stop finally engages, the ~200 ms
    # already committed to the audio source is played out under a ramp
    # instead of being discarded mid-word.  Admitted by the SAME predicate as
    # the tail it executes, so the browser tests, the simulations lab, the
    # standalone demo and the release canary keep byte-identical stop
    # behaviour and only a managed phone call changes.
    graceful_stop_stats = GracefulStopStats()
    alignment_feed = AlignmentFeed() if managed_pstn_interruption else None
    interruption_telemetry = InterruptionTelemetry(
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        gated=managed_pstn_interruption,
        # Read from the options actually handed to AgentSession below, so the
        # log quotes the floor this call ran with rather than the one a reader
        # assumes. The mid-word-cut investigation needed exactly this: events
        # said caller_speech_ms 147, and nothing said the floor was 200.
        min_duration_seconds=configured_interruption_min_duration(session_options),
        min_interruption_words_configured=configured_word_requirement,
        min_interruption_words_effective=word_requirement,
    )
    telemetry_collector = CallTelemetryCollector(
        call_id=call.call_id,
        organization_id=str(call.config.get("organization_id") or ""),
        project_id=str(call.config.get("project_id") or ""),
    )

    def emit_language_switch(event: str, **details: Any) -> None:
        # Keep the pre-existing structured worker log, but feed only its
        # terminal applied/rejected/failed outcomes into the final bounded
        # artifact. The collector ignores queued/skipped/operational events,
        # so a scheduled request can never be shown as an applied switch.
        telemetry_collector.observe_language_switch(event, **details)
        if event == "language_switch_applied" and details.get("language"):
            # An untagged text is judged in the language the call now speaks.
            value_redactor.set_default_language(details["language"])
        emit(event, call_id=call.call_id, **details)
        if transcript_publisher is not None:
            # Only an APPLIED switch becomes a transcript moment; the
            # publisher ignores queued/skipped/failed by name.
            transcript_publisher.on_language_switch(event, **details)
        if transcript_event_recorder is not None:
            # The stored half of the same moment, on the recording clock.
            transcript_event_recorder.on_language_switch(event, **details)

    language_switcher: ControlledLanguageSwitcher | None = None
    if follow_caller is not None:
        # Build only a previously parsed profile from this call's immutable
        # envelope.  The closure receives no model/caller input and reuses the
        # call-local credential overlay; it never rereads mutable agent state.
        def build_switch_stack(target_profile: Any, simultaneous_stt: Any, boundary: int) -> Any:
            # The call is leaving the language its acknowledgement clips were
            # made in.  Stop them for the rest of the call rather than play a
            # cached phrase in the wrong language.
            scheduler = acknowledgements["scheduler"]
            if scheduler is not None:
                scheduler.stop()
            feedback = acknowledgements["tool_wait"]
            if feedback is not None:
                feedback.stop()
            target_config = follow_caller.profiles[str(target_profile.language)].config
            return build_call_tts_component(
                target_profile,
                call_id=call.call_id,
                call_config=call.config,
                boundary=boundary,
                environ=runtime_environ,
                admission_context=voice_admission,
                call_owner=voice_call_owner,
                target_config=target_config,
                simultaneous_stt=simultaneous_stt,
            )

        language_switcher = ControlledLanguageSwitcher(
            frozen=follow_caller,
            session=session,
            initial_profile=profile,
            initial_stack=stack,
            build_tts=build_switch_stack,
            collector_factory=ModelUsageCollector,
            emit=emit_language_switch,
            speech_suspended=lambda: navigation.suspends(SPEECH_LANGUAGE_SWITCH),
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
    # What happened around each caller turn end, on the cascade
    # stack only: a conversation engine ends the caller's turn itself, so none
    # of the settings recorded here ran and there is nothing of ours to keep.
    # Diagnostic only -- no ending, disposition or billing reads it.
    turn_end_evidence: TurnEndEvidence | None = None
    if not s2s_live_call:
        turn_end_evidence = TurnEndEvidence(
            settings={
                # Read from the options handed to AgentSession above.
                **turn_end_settings_evidence(session_options, realtime_config=call.config),
                "vad_checkpoint_sha256": SILERO_VAD_PINNED_SHA256,
                "language": str(profile.language or ""),
                "stt_provider": provider_labels["stt"],
                "stt_model": str(component_model_labels(profile).get("stt") or ""),
            }
        )
    finished = asyncio.Event()
    semantic_admission = SemanticTurnAdmission()
    # Installed after session.start() below, once RoomIO owns the audio chain.
    playout_hold: Any = None
    realtime_event_trace = RealtimeEventTrace(
        relative_ms=recorder.relative_ms,
        enabled=s2s_live_call and os.environ.get("SKYSAY_REALTIME_EVENT_TRACE") == "1",
    )
    install_realtime_event_trace(session, realtime_event_trace)
    turn_events = SemanticTurnEventRecorder(
        recorder=recorder,
        telemetry_collector=telemetry_collector,
        correlate_replies=not s2s_live_call,
        admission=semantic_admission,
        first_message=profile.first_message,
        provider_labels=provider_labels,
        model_labels=component_model_labels(profile),
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        playout_hold_reason=lambda: (
            playout_hold.policy.take_turn_report()[0] if playout_hold is not None else ""
        ),
        caller_transcript_observation=(
            transcript_gate.gate.consume_final_transcript_observation
            if transcript_gate is not None
            else None
        ),
        speech_to_speech=s2s_live_call,
        # The caller spoke, the engine answered, and no caption exists for it:
        # a typed moment in the stored transcript events, never a turn.
        caller_spoke_without_caption=(
            (
                lambda started_at: transcript_event_recorder.on_caller_spoke_no_transcript(
                    started_at=started_at
                )
            )
            if s2s_live_call and transcript_event_recorder is not None
            else None
        ),
        # An unasked, unwithheld, non-tool-reply generation: the engine heard
        # the caller. The witness ignores it on an engine without server-side
        # turn detection.
        engine_turn_heard=caller_speech_witness.on_engine_turn,
    )

    def stop_semantic_turns() -> None:
        semantic_admission.close()

    def finish_session() -> None:
        stop_semantic_turns()
        finished.set()

    call_end_arbiter = SessionEndArbiter(
        finish=finish_session,
        # The same file the heartbeat above writes and the crash reconciler
        # reads. The arbiter records WHY here at the instant it decides, so a
        # teardown that never runs cannot take the reason with it; the writer
        # merges, so the heartbeat's next tick leaves that reason alone.
        state_path=state_path,
        # One clock for this call. Every lifecycle timestamp on it comes from
        # here, and the reconciler compares this instant against its own.
        now=utcnow,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        # Read at the instant the reason is decided and sealed with it. An agent
        # that does not navigate reports nothing (not "false").
        prior_person_source=lambda: (
            navigation.person_reached_state(
                unresolved=bool(far_end_watch is not None and far_end_watch.unresolved)
            )
            if navigation.enabled
            else None
        ),
    )

    def agent_finished() -> None:
        """The agent ended this call itself -- its script or workflow finished.

        Not a platform rule and not a hang-up: nobody left, no limit was hit,
        the conversation reached its end. It is stamped because leaving it
        empty told a customer nothing at exactly the moment they most want an
        answer, and a scripted workflow ends most of its calls this way, so
        empty would have been the commonest reading of a field whose whole
        purpose is "why did this call end".

        Stands down when the session is already ending, and that guard is not
        symmetry for its own sake. If the room went away first the agent did
        NOT end this call -- the room did -- and claiming otherwise would put
        the platform's word on someone else's ending. It also keeps a rule that
        already fired, a maximum duration say, from being relabelled by a
        goodbye that was still in flight when the rule ended the call.
        """

        if finished.is_set():
            return
        call_end_arbiter.end(
            TERMINATION_AGENT_COMPLETED, detected_via=DETECTED_VIA_AGENT_SCRIPT
        )

    def operator_asked_to_end() -> None:
        """An operator asked for this call to end. Act now, not at the next tick.

        The intent was only ever READ here and acted on by the watchdog poll,
        fifteen seconds away. That made the reason a race the operator could
        lose: on one production call the agent's goodbye began finishing the
        session before the next tick, so the call ended, the teardown was
        acknowledged, and the row carried no reason at all -- while an
        identical request on another call was stamped because the tick happened
        to win. Same API path, two different answers.

        Acting at receipt makes it the heartbeat interval instead, and makes
        the promise a promise. The watchdog keeps its own hangup branch as a
        backstop; the arbiter is first-wins, so the two cannot disagree.

        Precedence when the two endings collide, which is the founder's product
        intent rather than an implementation convenience: an operator who asks
        before the teardown has started gets ``hangup_requested``, because that
        is what happened. Once the agent's own ending owns the teardown the
        call is already ending for its own reason, and relabelling it would
        tell the customer an operator cut off a conversation that had in fact
        finished.
        """

        if finished.is_set():
            return
        call_end_arbiter.end(
            TERMINATION_HANGUP_REQUESTED, detected_via=DETECTED_VIA_HANGUP_INTENT
        )

    def call_end_clock_snapshot() -> CallEndSnapshot:
        """In-memory only, so it cannot fail and take the timers with it."""

        answered_at = call_end_clock["answered_at"]
        released_at = call_end_clock["input_released_at"]
        if answered_at is not None:
            # The overall limit, frozen at the answer, is the navigation
            # state's deadline: a call that runs out of time waiting in a menu
            # or a queue ends ``ivr_timeout``, not as a plain maximum duration.
            # Set here, on the watchdog's own read, so it cannot be set before
            # the limit has been resolved.
            navigation.set_deadline(answered_at + max_call_seconds)
        return CallEndSnapshot(
            answered=answered_at is not None,
            seconds_since_answer=(
                0.0 if answered_at is None else time.monotonic() - answered_at
            ),
            caller_has_spoken=caller_speech_witness.heard,
            hangup_requested=bool(hangup_intent.get("pending")),
            seconds_since_input_released=(
                None if released_at is None else time.monotonic() - released_at
            ),
        )

    async def call_end_occupancy() -> int:
        parties = await server_side_remaining_parties(
            ctx.api.room,
            room_name=ctx.room.name,
            agent_identity=str(
                getattr(getattr(ctx.room, "local_participant", None), "identity", "") or ""
            ),
            # Captured where the recorder was LAUNCHED, not rebuilt here.
            # Rebuilding it meant two spellings that agreed only by habit, and
            # a divergence would have left the occupancy rule permanently
            # inert with nothing red.
            evidence_identity=evidence_identity,
            request_factory=api.ListParticipantsRequest,
        )
        return len(parties)

    max_call_seconds, max_call_clamped = resolved_max_call_seconds(call.config)
    funding_watch = asyncio.create_task(funded_runtime.watch(
        arbiter=call_end_arbiter, max_seconds=max_call_seconds, is_finishing=finished.is_set,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        # One decision for the call-limit ending, shared with the watchdog.
        duration_ending=navigation.duration_ending,
    ))
    no_speech_seconds, no_speech_clamped = resolved_no_caller_speech_seconds(call.config)
    if max_call_clamped or no_speech_clamped:
        # Never silent: a customer whose configured limit was moved finds out
        # from their own call's evidence rather than by timing the recording.
        emit(
            "worker_call_limit_clamped",
            call_id=call.call_id,
            max_call_seconds=max_call_seconds,
            max_call_clamped=max_call_clamped,
            no_caller_speech_seconds=no_speech_seconds,
            no_caller_speech_clamped=no_speech_clamped,
        )

    # A SEPARATE task from the heartbeat on purpose: the heartbeat kept posting
    # for all 94 minutes of the incident while the media path was dead, and its
    # post is a blocking network call handed to a thread. A check living inside
    # it inherits every way it can stall.
    call_end_watch = asyncio.create_task(
        watch_for_worker_call_end(
            arbiter=call_end_arbiter,
            clock=call_end_clock_snapshot,
            occupancy=call_end_occupancy,
            max_call_seconds=max_call_seconds,
            no_caller_speech_seconds=no_speech_seconds,
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            # Any ending stands the watchdog down, not just one the arbiter
            # made: several paths end the session without going through it.
            is_finishing=finished.is_set,
            # The silence rule stands down, and the budgets and the call limit
            # end the call, while it waits in a phone menu or a hold queue.
            navigation=navigation,
        )
    )

    # A synthetic caller is another Agent job. Its room can close before this
    # worker receives the remote-participant event used by ordinary SIP/browser
    # sessions. Wake the SAME single teardown path from the earlier room event
    # so usage, evidence and terminal lifecycle run before LiveKit's hard
    # entrypoint-cancellation deadline.
    room_disconnect = production_room_disconnect_handler(
        finish=finish_session,
        emit=lambda event, **details: emit(
            event, call_id=call.call_id, **details
        ),
        # The room going away IS an ending, and until now it was the one
        # ending with no name: the session finished with an empty reason and
        # the customer was told nothing about a call that stopped mid-sentence.
        # The arbiter is first-wins, so on an ordinary ending -- where this
        # worker deleted the room itself moments earlier -- the real reason is
        # already stamped and this attempt is refused. It wins only when
        # nothing else had named the ending, which is exactly the gap.
        stamp_lost=lambda: call_end_arbiter.end(
            TERMINATION_MEDIA_SESSION_LOST, detected_via=DETECTED_VIA_ROOM_LOST
        ),
    )
    ctx.room.on(
        "disconnected",
        room_disconnect,
    )
    def note_room_closed(*_args: Any) -> None:
        closing_evidence.record("media_ended", source="room", owner="transport", reason="media_disconnect")
    ctx.room.on("disconnected", note_room_closed)

    async def summarise_opening() -> None:
        """The outbound opening's one record, written the moment the call ends.

        Not from the teardown: an evidence step failing there must not take
        the record with it, and the decider's tasks stop here either way.
        """

        try:
            await _first_of(finished)
        finally:
            decider = opening["decider"]
            for opening_task in opening["tasks"]:
                if not opening_task.done():
                    opening_task.cancel()
            if decider is not None:
                decider.note_ended(at=time.monotonic())
                _quietly(
                    emit,
                    OPENING_SUMMARY_EVENT,
                    call_id=call.call_id,
                    opening_silence_trimmed_ms=_trimmed_ms(opening["silence_trimmed"]),
                    policy=OPENING_POLICY_CALLEE_FIRST,
                    callee_spoke=decider.callee_spoke,
                    pickup_in_context=opening["pickup_in_context"],
                    first_reply_path=opening["first_reply_path"],
                    **decider.telemetry(),
                )
            else:
                _quietly(
                    emit,
                    OPENING_SUMMARY_EVENT,
                    call_id=call.call_id,
                    policy=OPENING_POLICY_AGENT_FIRST,
                    excluded_by=opening_exclusion,
                    opening_silence_trimmed_ms=_trimmed_ms(opening["silence_trimmed"]),
                )

    opening_summary_task: asyncio.Task[None] | None = (
        asyncio.create_task(summarise_opening()) if outbound is not None else None
    )

    # Filled in once the session is live and the customer enabled
    # acknowledgements; read by the state handlers and the termination path.
    acknowledgements: dict[str, Any] = {
        "scheduler": None,
        "player": None,
        "task": None,
        "prepare": None,
        # The tool-wait feedback owner shares the player and the clips.
        "tool_wait": None,
    }

    def stop_acknowledgements() -> None:
        # Nothing may be scheduled in front of a goodbye, whoever chooses it.
        scheduler = acknowledgements["scheduler"]
        if scheduler is not None:
            scheduler.stop()
        feedback = acknowledgements["tool_wait"]
        if feedback is not None:
            feedback.stop()

    def begin_worker_termination() -> None:
        # Model closes reach this only after the reclaimable goodbye settles.
        # Platform limits close admission immediately. record_final_message
        # retains the actual say-time boundary for terminal audio attribution.
        recorder.mark_termination()
        if speaks_text:
            stop_semantic_turns()
        else:
            # A voice-to-voice model says its goodbye itself, in the turn that
            # calls end_call, and the SDK commits that goodbye as an assistant
            # item only after it has played. Close the caller's side now and
            # keep the goodbye's item admissible until the teardown closes the
            # rest (``finish_session``) -- a full close here dropped the one
            # utterance that ended the call from its own evidence.
            semantic_admission.close_caller_input()
        stop_acknowledgements()
        if language_switcher is not None:
            # A language transition queued in the same model turn as the
            # hangup would otherwise still think the conversation was live:
            # it reaches its component boundary by cancelling whatever is
            # speaking -- here, the goodbye -- and an applied switch now
            # schedules a reply of its own. Neither belongs on a call whose
            # last words have already been chosen. Teardown marks this too;
            # this is only the earliest honest moment.
            language_switcher.mark_terminal()

    # The goodbye's OWN playout-start, stamped at the agent-state "speaking"
    # edge whose ``current_speech`` is the goodbye handle -- identity by
    # speech_id, so a same-generation model preamble (whose first audio can land
    # AFTER say-time) never matches. Used to bind the goodbye to the interval
    # CONTAINING its real first frame, and to derive its reply-latency span;
    # a say-time proxy cannot do either when a preamble precedes the goodbye.
    goodbye_playout: dict[str, Any] = {"speech_id": "", "at": None}
    agent_speaking_started: dict[str, float | None] = {"at": None}
    goodbye_telemetry_turns: tuple[int, ...] = ()
    goodbye_origin_speech: Any = None

    def note_close_proposed(origin_speech: Any = None, user_item_id: str | None = None) -> None:
        nonlocal goodbye_telemetry_turns, goodbye_origin_speech
        goodbye_telemetry_turns = telemetry_collector.snapshot_terminal_turns(user_item_id=user_item_id)
        goodbye_origin_speech = origin_speech
        # A failed subsequent say must not reuse an earlier proposal's edge.
        goodbye_playout["at"] = None

    def note_close_reclaimed() -> None:
        if speaks_text:
            # Freeze THIS proposal, not mutable state read by a later callback.
            proposal_turns = goodbye_telemetry_turns
            def settle_reclaimed(_done: Any = None) -> None:
                outcome = telemetry_collector.observe_terminal_assistant(
                    provider_labels=provider_labels, cancelled_turns=proposal_turns)
                if outcome is not None:
                    emit("reclaimed_turns_resolved", call_id=call.call_id,
                         proposal_turns=proposal_turns, **outcome)
            if goodbye_origin_speech is not None:
                # The tool may precede its own spoken preamble's history item.
                # Keep that caller until the originating speech has settled.
                # Other replies already pair by ID, so they cannot take it.
                goodbye_origin_speech.add_done_callback(settle_reclaimed)
            else:
                settle_reclaimed()
        else:
            delivered = notify_native_close_reclaimed(
                session_agent, gpt_live_llm,
                navigation_actions.model_context.thinking if navigation_actions is not None else None,
            )
            emit("end_call_reclaim_context", call_id=call.call_id,
                 queued=delivered, mechanism="append_thinking" if delivered else "tool_result_and_silence_recovery")

    def note_goodbye_scheduled(speech_id: str) -> None:
        recorder.register_worker_goodbye(speech_id)
        if end_call_guard is not None:
            try:
                end_call_guard.utterances.note_goodbye_say(str(speech_id or ""))
            except Exception:  # noqa: BLE001
                pass
        goodbye_playout["speech_id"] = str(speech_id or "")
        goodbye_playout["at"] = None

    def commit_terminal_telemetry() -> None:
        outcome = telemetry_collector.observe_terminal_assistant(
            provider_labels=provider_labels,
            goodbye_playout_start=goodbye_playout["at"],
        )
        if outcome is not None:
            emit("terminal_turns_resolved", call_id=call.call_id, **outcome)

    def record_final_message(
        message: str,
        *,
        interrupted: bool,
        speech_id: str = "",
        scheduled_at: float | None = None,
        egress_confirmed: bool | None = None,
        egress_signal: str = "",
    ) -> None:
        if scheduled_at is not None:
            recorder.mark_termination(at=scheduled_at)
        # Bind the goodbye as a preplayout candidate ONLY when its OWN playout
        # start was observed (the "speaking" edge for this handle fired), so
        # finalize links it by containment of that stamp. Without that stamp
        # there is NO honest interval to bind to: the edge is missed exactly when
        # a preamble flows straight into the goodbye with no new speaking edge,
        # and a say-time-bound candidate then steals the preamble's clip (Grok
        # P1). In that case record the terminal message only -- audio-less is
        # honest; the wrong clip is not. The terminal message is always recorded:
        # the legacy/fallback channel, deduped by speech_id when the goodbye
        # actually binds.
        playout_start = goodbye_playout["at"]
        if speech_id and playout_start is not None:
            recorder.bind_worker_goodbye_text(speech_id, message, at=playout_start)
            recorder.complete_preplayout_agent_speech(speech_id, interrupted=interrupted)
        elif speech_id:
            recorder.discard_unanchored_worker_goodbye(speech_id)
        recorder.add_terminal_message(
            message,
            interrupted=interrupted,
            speech_id=speech_id,
            egress_confirmed=egress_confirmed,
            egress_signal=egress_signal,
        )

    async def drain_goodbye_egress() -> dict[str, object]:
        # The recorder's live agent edge is fed by the evidence sidecar, an
        # independent room subscriber: quiet there means the farewell left the
        # room, not merely the SDK's local source.  A caller who leaves during
        # the drain wakes ``finished`` (the worker's own room stays connected,
        # so ``room_disconnect`` alone would miss it); either ends the drain
        # unconfirmed, because what the far end heard is then unknowable.
        def observer_alive() -> bool:
            if sidecar_lost["at"] is not None:
                return False
            last_frame = recorder.observer_last_frame_at()
            return (
                last_frame is not None
                and time.monotonic() - last_frame <= TERMINAL_EGRESS_OBSERVER_STALE_SECONDS
            )

        return await drain_terminal_egress(
            agent_audible_until=recorder.agent_audible_until,
            room_disconnected=lambda: room_disconnect.triggered or finished.is_set(),
            observer_alive=observer_alive,
            # A voice-to-voice goodbye is the model's own speech, never the
            # armed ``say`` handle, so it is keyed on the agent's latest
            # speaking start instead. Only reached when the room is still
            # hearing the agent (``WorkerLocalEndCall._agent_audible_now``).
            goodbye_playout_start=(
                goodbye_playout["at"] if speaks_text else agent_speaking_started["at"]
            ),
        )

    # The end_call guard: declines the agent's own hang-up on the shape that
    # ends calls (a run of the agent's replies cut by a caller who keeps
    # talking, held caller words, a live caller run). Managed-PSTN cascade
    # calls with the SDK's own turn hooks only: a workflow run's replies are
    # scripted, a voice-to-voice engine speaks its goodbye before the tool,
    # and a browser or lab call carries no transcript gate. Every name the
    # snapshot reads is late-bound: ``session`` and ``end_call`` exist by the
    # time a tool can be invoked.
    end_call_guard: EndCallGuard | None = None
    end_call_classifier: Any | None = None

    def promote_silence_after_refusal(*, reclaimed: bool = False) -> None:
        if navigation.suspends(ENDER_END_CALL_TOOL):
            # A refusal inside a menu or a queue is not a caller who went quiet:
            # promoting here would turn mutual silence on for the rest of the
            # call, and it would then check in on the person who picks up.
            return
        # A refused hang-up puts the existing mutual-silence rule on the
        # clock, so a refused caller who then goes quiet is ended the way
        # every quiet call is. ``off`` never promotes nor begins an episode;
        # that configuration relies on the endings that exist without it.
        now = time.monotonic()
        mutual_silence.promote_to_enforce(at=now)
        mutual_silence.begin_post_refusal_episode(at=now, recover_unanswered=reclaimed)
        if end_call_guard is not None:
            # The ACTUAL state, not the intent: ``off`` never promotes.
            end_call_guard.silence_promoted = mutual_silence.mode == "enforce"

    # The close owner's OWN floor refusals, on every engine and guard mode:
    # their automatic tool follow-up is cancelled (no farewell loop) and the
    # silence rule is put on the clock. If mutual silence is ``off`` the call
    # still ends by the endings that exist without it: the caller's next
    # turn, the maximum call duration (``funded_runtime.watch``), the
    # agent-unresponsive rule (``ModelUnresponsiveRule``) and remote hangup.
    close_refusal_recovery = LocalCloseRefusalRecovery(
        on_refusal=promote_silence_after_refusal,
        on_reclaimed=lambda: promote_silence_after_refusal(reclaimed=True),
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
    )
    if end_call_evidence is not None:
        caller_run_ended.set()

        async def wait_for_caller_run_end(cap_seconds: float) -> bool:
            try:
                await asyncio.wait_for(caller_run_ended.wait(), timeout=cap_seconds)
            except TimeoutError:
                return False
            return True

        def machine_corroborated() -> bool:
            # CURRENT signals only. ``repeating_recording_stands`` is a
            # permanent latch: three loop cycles, then a live person answers,
            # and the latch would let the model hang up on that person while
            # they speak. The audio arbiter's verdict reverses on human
            # evidence and the voicemail observer refuses a cut-off human.
            arbiter = answer_detection.arbiter
            try:
                if bool(arbiter.non_human_active):
                    return True
            except Exception:  # noqa: BLE001 - an unreadable detector corroborates nothing
                return False
            observer = getattr(answer_detection.voicemail, "observer", None)
            if observer is None:
                return False
            try:
                # The policy outcome, not ``classification()``: the class
                # latches on the first proof, the outcome records a human
                # who arrived after it (``cut_off_human`` / provisional).
                return voicemail_stands(observer.hang_up())
            except Exception:  # noqa: BLE001
                return False

        def _history_messages() -> list[Any]:
            items = getattr(getattr(session, "history", None), "items", None) or []
            return [
                item
                for item in list(items)
                if str(getattr(item, "type", "") or "") in ("", "message")
                and str(getattr(item, "role", "") or "") in ("user", "assistant")
            ]

        def read_guard_snapshot() -> GuardSnapshot:
            items = getattr(getattr(session, "history", None), "items", None) or []
            cut, considered = cut_replies_of_last(items)
            return GuardSnapshot(
                finished=finished.is_set(),
                ending_scheduled=end_call.ending_scheduled,
                # The close floor's reading, not raw VAD: on a measured
                # transcript source an open run vetoes only with words behind
                # it, and a wordless one defers for the source's budget.
                user_speaking=end_call.guard_user_speaking(
                    str(getattr(session, "user_state", "") or "") == "speaking"
                ),
                caller_wordless_remaining=end_call.guard_wordless_remaining(),
                holds_final=(
                    # A held final the close floor would not read as words (a
                    # backchannel, on a measured source) holds nothing back.
                    transcript_gate.gate.holds_final_transcript_where(end_call.caller_counts_as_words)
                    if transcript_gate is not None
                    else False
                ),
                unconsumed=end_call_evidence.unconsumed(),
                machine_corroborated=machine_corroborated(),
                cut_replies=cut,
                replies_considered=considered,
                last_final_confidence=end_call_evidence.last_final_confidence,
                outbound=outbound is not None,
            )

        def read_guard_exchange() -> GuardExchange | None:
            # The LAST exchange only: the caller turn the model answered and
            # the assistant line before it, in history order. Not the latest
            # assistant item, which may be the reply that carried this very
            # tool call.
            return exchange_from_history(
                _history_messages(),
                language=(
                    language_switcher.active_language if language_switcher is not None else profile.language
                ),
            )

        def on_guard_refusal(outcome: Any) -> None:
            promote_silence_after_refusal()

        # The platform's own key from the BASE environment: the component
        # credential overlay is a dict copy the worker hands its own stack and
        # never mutates os.environ. Built per job; its self-test starts now,
        # concurrently with session bring-up, and is awaited lazily before the
        # classifier's first real request, never on the call-answer path.
        end_call_classifier = build_platform_classifier(
            os.environ,
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        )
        end_call_classifier.start_self_test()

        end_call_guard = EndCallGuard(
            mode=end_call_guard_mode,
            budget=resolve_guard_budget(os.environ),
            evidence=end_call_evidence,
            read_snapshot=read_guard_snapshot,
            wait_for_listening=wait_for_caller_run_end,
            # A wordless run's defer waits for the caller's WORDS, not its end.
            wait_for_floor=lambda seconds: end_call.wait_for_floor(seconds),
            future_factory=lambda: asyncio.get_running_loop().create_future(),
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            on_refusal=on_guard_refusal,
            classifier=end_call_classifier,
            read_exchange=read_guard_exchange,
            # Rule 9's threshold is pinned per recogniser language and read
            # per decision from the ACTIVE language, so a follow-caller switch
            # mid-call moves the rule to that language's pin (or off, where
            # its gap is unproven).
            active_language=lambda: (
                language_switcher.active_language if language_switcher is not None else profile.language
            ),
        )

    bind_tool_speech = getattr(stack.llm, "bind_tool_speech", None)
    tool_speech_claim = bind_tool_speech(session) if callable(bind_tool_speech) else None
    end_call = WorkerLocalEndCall(
        tool_speech_claim=tool_speech_claim,
        closing_receipt=closing_evidence.record,
        session=session,
        language=profile.language,
        on_ending=begin_worker_termination,
        guard=end_call_guard,
        close_floor_budget=close_floor_budget,
        caller_words=lambda text: not is_caller_backchannel_only(text, close_floor_backchannels),
        terminal_intent=terminal_intent,
        refusal_recovery=close_refusal_recovery,
        navigation_suspended=lambda: (
            navigation.suspends(ENDER_END_CALL_TOOL) or navigation_gate.unresolved()
        ),
        on_reclaimed=note_close_reclaimed,
        on_proposed=note_close_proposed,
        # The agent's own goodbye. Deliberately NOT finish_session: this is
        # one of exactly two endings the agent itself performs, and the one
        # that previously carried no reason at all.
        finish=agent_finished,
        record_final_message=record_final_message,
        # Hold ``shutdown`` until the room stops hearing the goodbye (bounded),
        # so the farewell's last frames are not torn down in flight and the
        # evidence certifies only what the room observed.
        drain=drain_goodbye_egress,
        # The goodbye never arrives as an assistant conversation item, so the
        # resolving caller turn would finalize ``unmatched_user_turn``. Pair the
        # final utterance here, with this call's own STT provider label, at the
        # same seam that records the farewell text. Emit the resolved/kept counts
        # and the boundary gap so a live end_call call calibrates the fragment
        # gap that separates the final utterance from an earlier unanswered turn.
        commit_terminal_telemetry=commit_terminal_telemetry,
        # Arm the playout-start watcher with the goodbye's handle id the moment
        # say() returns it, so the next agent "speaking" edge for THAT handle is
        # stamped as the goodbye's real first frame.
        on_say=note_goodbye_scheduled,
        is_finished=finished.is_set,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        # The SDK refuses ``say(text)`` on a session with no TTS, which is
        # every voice-to-voice engine: there the model speaks its own goodbye
        # and this action only lets it finish.
        speaks_text=speaks_text,
        agent_audible_until=recorder.agent_audible_until,
        agent_speaking=lambda: str(getattr(session, "agent_state", "") or "") == "speaking",
    )
    try:
        from ..transfer import RoomMedia, TransferController
        from ..transfer_dial import place_transfer_leg, remove_transfer_leg, subscribe_agent_audio
    except ImportError:  # pragma: no cover - flat import in the worker image
        from transfer import RoomMedia, TransferController
        from transfer_dial import place_transfer_leg, remove_transfer_leg, subscribe_agent_audio

    caller_identity = str(getattr(participant, "identity", "") or "")

    transfer_media = RoomMedia(
        session,
        ctx.room,
        place_transfer_leg,
        functools.partial(subscribe_agent_audio, ctx.room, caller_identity),
        functools.partial(remove_transfer_leg, ctx.room),
        # The fact LiveKit itself checks before it refuses say(text): a
        # voice-to-voice session has no TTS, so the model speaks the lines.
        speaks_text=speaks_text,
        is_finished=lambda: finished.is_set() or room_disconnect.triggered,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
    )
    # The call's one transfer: the tool places it, ``cancel_transfer`` and the
    # teardown stop it, and it reports its own outcome exactly once. A room
    # going away finishes the session, so ``finished`` covers both endings.
    transfer_controller = TransferController(
        transfer_media,
        report=lambda outcome: _report_transfer_outcome(
            call, outcome, lambda event, **details: emit(event, call_id=call.call_id, **details)
        ),
        ended=finished.wait,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
    )

    async def transfer_call(dial: dict, **options: Any) -> str:
        if navigation_actions is not None:
            # Keys and a transfer's initiation are serialised under one lock, and no key is
            # sent for the rest of the call once a transfer has started: the room is about
            # to hold a second SIP participant, and ``publish_dtmf`` reaches all of them.
            async with navigation_actions.line.transfer():
                pass
        outcome = await transfer_controller.run(dial, **options)
        if outcome == "answered":
            # The caller and the human are bridged and cannot hear the agent:
            # their pauses are not this worker's silence to end.
            mutual_silence.note_transferred(at=time.monotonic())
        return outcome

    tools = await asyncio.to_thread(
        build_livekit_model_tools,
        call,
        caller_inputs=caller_inputs,
        end_call=end_call,
        transfer=transfer_call,
        cancel_transfer=transfer_controller.cancel,
        terminal_intent=terminal_intent,
        tool_usage_counter=successful_tool_usage,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        navigation_gate=navigation_gate,
        worker_local_names=SWITCH_LANGUAGE_TOOL_NAMES if language_switcher is not None else frozenset(),
    )
    if language_switcher is not None:
        # This is a worker-local controller, deliberately outside the customer
        # MCP allowlist and successful-tool billing counter.  Its source is
        # fixed in the invoker, so an LLM cannot turn a transcript into a
        # claimed provider signal or choose a profile/configuration.
        # Built from THIS call's frozen allowed set, never a static shape: the
        # model can only pick a value the switcher will accept.
        tools.append(
            function_tool(
                raw_schema=switch_language_tool_schema(language_switcher.allowed_languages)
            )(language_switcher.model_tool)
        )

    if navigation_actions is not None:
        # The controller's own tools (pinned at reservation as zero-rate tools): requests the
        # executor judges against what was observed. Not business tools; none reads a variable.
        if successful_tool_usage is not None:
            navigation_actions.tools.bind_usage(successful_tool_usage)
        for navigation_schema, navigation_handler in navigation_actions.tool_specs_with_context():
            tools.append(function_tool(raw_schema=navigation_schema)(navigation_handler))

    provider_language_evidence: Any | None = None
    if language_switcher is not None and follow_caller is not None:
        # Only a frozen declaration whose mode names an adapter can opt into
        # this typed STT metadata hook, and the mode is what names it -- the
        # strict-hint and language-identification declarations both preserve a
        # per-final source language. Deepgram multi reports one too (the pinned
        # plugin sets each final's language), but its mode is not routed in
        # the table yet, so it maps to nothing and stays explicit-request only.
        evidence = PROVIDER_LANGUAGE_EVIDENCE.get(
            str(follow_caller.simultaneous_stt.get("mode") or "")
        )
        if evidence is not None:
            provider_language_evidence = evidence(language_switcher)


    # Built BEFORE the session handlers so ``speech_created`` has an agent to
    # hand a handle to. Nothing here starts media; ``session.start`` below is
    # still the only admission point.
    evidence_agent_kwargs: dict[str, Any] = {
        "instructions": (
            gpt_live_session_instructions(
                profile.instructions,
                language_name=language_name(profile.language),
                opening_mode=profile.opening_mode,
                call_ending_policy="" if live_policy_in_prompt else live_policy,
            )
            if s2s_live_call
            else profile.instructions
        ),
        "tools": tools,
    }
    if caller_inputs is not None:
        evidence_agent_kwargs["instructions"] += ACTION_INSTRUCTIONS
    if callee_first and s2s_live_call:
        evidence_agent_kwargs["instructions"] += (
            "\nThis is an outbound call. Let the person speak first. Listen to their complete "
            "pickup before speaking, including pauses between sentences. Do not interrupt or "
            "backchannel during their opening statement. Answer any question or request first; "
            "Treat the entire pickup, including a hello followed by more speech, as one statement. "
            "If it contains a request, question, or correction (for example a callback request or wrong number), "
            "your first spoken sentence must address that content directly; do not lead with the saved greeting. "
            "Never repeat a greeting already spoken. A person giving their name or business "
            "and offering help is a normal greeting, not evidence of a wrong number. "
            "Do not end the call or claim a wrong number from that greeting alone. "
            "Only when the complete pickup contains "
            "nothing beyond a hello, follow this opening instruction: "
            + greet_directive(profile.first_message, opening_mode=profile.opening_mode,
                              language_name=language_name(profile.language))
        )
    if s2s_live_call:
        evidence_agent_kwargs["llm"] = gpt_live_llm
        evidence_agent_kwargs["speech_to_speech"] = True
    def opening_owns_turn() -> bool:
        """A caller turn now is the callee's pickup, kept for the opening."""

        decider = opening["decider"]
        if decider is None:
            return False
        return decider.listening or bool(opening["pickup_pending"]) or bool(opening.get("pickup_sealing"))

    def commit_opening_pickup() -> None:
        """Close the callee's pickup turn now, into the context, with no reply.

        The runtime's own commit would come later and, landing after the
        opening was dispatched, would cancel it. A commit without a reply
        never interrupts. The item it adds is noticed by
        ``on_conversation_item`` like any other committed pickup.
        """

        try:
            session.commit_user_turn(
                skip_reply=True,
                transcript_timeout=PICKUP_COMMIT_TRANSCRIPT_TIMEOUT_SECONDS,
                stt_flush_duration=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - the ordinary commit still comes
            _quietly(emit, "opening_pickup_commit_failed", call_id=call.call_id, error_type=type(exc).__name__)

    async def seal_opening_pickup() -> None:
        """Make the pickup's pending end-of-turn reply-less, right before the
        opening goes out.

        A final that reaches the runtime after the first reply-less commit
        re-schedules the turn as an ORDINARY one (the runtime's final-transcript
        path), and that one interrupts the opening when it lands. Called with
        the transcript gate already holding, a second reply-less commit
        replaces whatever end-of-turn is pending, and nothing further can reach
        the runtime to schedule another before the opening's first frame.
        """

        try:
            await asyncio.wait_for(
                session.commit_user_turn(
                    skip_reply=True, transcript_timeout=0.0, stt_flush_duration=0.0
                ),
                timeout=PICKUP_SEAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - never hold the opening on it
            _quietly(emit, "opening_pickup_seal_failed", call_id=call.call_id, error_type=type(exc).__name__)

    def on_opening_pickup() -> None:
        opening["pickup_pending"] = False
        decider = opening["decider"]
        # A turn that commits while the callee is mid-utterance belongs to
        # what they said before: the reply waits for the one they are saying.
        if decider is None or not decider.callee_speaking_now:
            opening["pickup_committed"].set()

    session_agent: EvidenceAgent = EvidenceAgent(admission=semantic_admission, transcript_gate=transcript_gate, interruption_telemetry=interruption_telemetry, provider_language_evidence=provider_language_evidence, alignment_feed=alignment_feed, recorder=recorder, end_call_evidence=end_call_evidence, opening_owns_turn=opening_owns_turn if callee_first else None, on_opening_pickup=on_opening_pickup if callee_first else None, **evidence_agent_kwargs)
    session_agent.bind_caller_audio_witness(caller_audio_bind_witness, caller_audio_flow)
    session_agent.bind_turn_end_evidence(turn_end_evidence)
    if navigation_actions is not None:
        session_agent.bind_model_context(
            navigation_actions.model_context, navigation_actions.begin_generation, navigation_actions.bind_tool_call
        )
    session_agent.record_opening_silence_trim(opening["silence_trimmed"])
    if language_switcher is not None:
        # A workflow agent owns its turns: the switcher must repeat a turn it
        # cut through this agent's own seam, never by asking the session for
        # a model reply that would advance the run again.
        language_switcher.bind_agent(
            session_agent,
            turn_cut=None,
            speak_owed_turn=(
                None
            ),
        )

    # The semantic-turn event contract lives in ``production.py`` so it can be
    # replayed offline; these handlers must stay thin adapters over it.
    def release_opening(handle: Any) -> None:
        """Hand the opening line back to the ordinary interruption rules.

        Every close of the opening protection lands here: the caller talking
        past the pickup bound, and the say's own ``finally``. Three things,
        in this order. The flag goes back first so the runtime's next VAD tick
        can act on it; the transcript gate's opening window closes so the
        words the caller said over the opening reach the runtime once the
        reply-in-flight signal drops; and the utterance is registered for
        interruption telemetry HERE, not from ``speech_created`` -- that event
        fires inside ``session.say`` while the handle is still deliberately
        uninterruptible, and ``observe_interruption_completion`` refuses an
        uninterruptible handle so the end_call farewell cannot inflate the
        denominator. The greeting is not that: it is interruptible from this
        moment, so it is counted like any other reply. Registration is
        idempotent on the speech id.
        """

        restore_speech_interruptions(handle)
        if transcript_gate is not None:
            transcript_gate.set_opening_turn_in_flight(False)
        if handle is None:
            return
        try:
            session_agent.observe_interruption_completion(handle)
        except Exception:
            emit("interruption_observer_not_attached", call_id=call.call_id)

    navigation.bind_conversation_activity(
        lambda: str(session.agent_state) == "speaking",
        pending=lambda: bool(far_end_watch is not None and far_end_watch.unresolved)
    )
    opening_protection.bind_release(release_opening)

    @session.on("user_state_changed")
    def on_user_state(event: Any) -> None:
        if str(event.new_state) == "speaking":
            opening["caller_speaking_starts"] += 1
        turn_events.on_user_state_changed(event)
        if navigation_actions is not None and str(event.new_state) == "speaking":
            # The signal that places a far-end turn the SDK did not time (see ``ModelContextGate``).
            navigation_actions.model_context.note_user_speaking()
        if turn_end_evidence is not None:
            turn_end_evidence.note_user_state(event.new_state)
        native_farewell.note_caller_speaking(str(event.new_state) == "speaking")
        if not s2s_live_call and str(event.new_state) == "speaking":
            # The cascade's ``user_state`` is raised by its own VAD on the
            # caller's input. A live engine's is not caller-only (Gemini Live
            # raises it before a generation the agent starts), so there the
            # native caller floor below feeds the transfer instead.
            transfer_controller.note_caller_speech_started("user_state")
        if opening["pickup_pending"] and str(event.new_state) == "speaking":
            # The caller has started a new utterance after the opening was
            # requested: what they say now is a reply to the opening, not the
            # pickup's late final.
            opening["pickup_pending"] = False
        if end_call_evidence is not None:
            # The guard's caller evidence: the START edge is evidence, the stop
            # edge is its absence. The run-end event wakes a deferred decision.
            speaking = str(event.new_state) == "speaking"
            end_call_evidence.note_caller_speaking(speaking)
            if speaking:
                caller_run_ended.clear()
            else:
                caller_run_ended.set()
            if speaking and end_call_guard is not None and end_call.ending_scheduled:
                # The caller started speaking while the goodbye plays after an
                # allow: the phase-2 tripwire.
                end_call_guard.allowed_then_caller_spoke = True
        # The silence rule keys on the VAD's start-of-speech. On the cascade
        # stack that is its only witness: the caller-transcript gate WITHHOLDS
        # STT events while a reply is in flight, so a transcript-backed check
        # there would read the agent's own opener as caller silence and hang
        # up on someone who was talking. The GPT-Live captions branch lives in
        # on_user_transcript below.
        caller_speech_witness.on_user_state_changed(event.new_state)
        # Every edge, not only the first. The witness above latches by design,
        # so on a call where the caller speaks at 10 s and again at 22 s it
        # reports the first and is silent about the second -- which is exactly
        # the question a silent-call investigation could not answer.
        caller_vad_edges.note_user_state(event.new_state, at=time.monotonic())
        # The END of this run is what puts the agent on the clock. The same
        # locally-produced edge on every engine -- a client VAD is attached to
        # the voice-to-voice sessions too -- which is why this rule needs no
        # branch per engine and survives the provider it is watching for.
        model_unresponsive.note_caller_speaking(
            str(event.new_state) == "speaking", at=time.monotonic()
        )
        # Freezes the silence clock (capped); never counts as words.
        mutual_silence.note_caller_vad(
            str(event.new_state) == "speaking", at=time.monotonic()
        )
        # Same coercion the transcript gate below and the barge-in path use on
        # this event, and have used in production since before this existed.
        # If it were wrong, those two would be silently dead already.
        answer_detection.note_activity(str(event.new_state) == "speaking")
        scheduler = acknowledgements["scheduler"]
        if scheduler is not None:
            scheduler.on_user_state_changed(event.old_state, event.new_state)
        feedback = acknowledgements["tool_wait"]
        if feedback is not None:
            feedback.on_user_state_changed(event.old_state, event.new_state)
        interruption_telemetry.on_user_state_changed(event.old_state, event.new_state)
        # The caller's speech run is what decides whether the opening line is
        # being talked over by someone taking the floor or by the phone being
        # picked up. Fed from the same edge the transcript gate reads below.
        opening_protection.note_caller_speaking(str(event.new_state) == "speaking")
        if transcript_gate is not None:
            # The DURATION half of the barge-in criterion. ``user_state`` is
            # raised from the VAD's own start-of-speech, so it is the same
            # measurement the runtime gates on -- and it is the run's END that
            # matters most: a caller who says one word and stops is silent by
            # the time the transcriber reports it, and nothing else would
            # re-evaluate.
            #
            # This does not widen the ARMING contract, which is still the one
            # ``agent_state`` signal. Whether a reply is in flight and whether
            # an interruption qualifies are different decisions.
            transcript_gate.note_caller_speaking(str(event.new_state) == "speaking")

    def current_speech_id() -> str:
        try:
            return recorder.speech_identity(session.current_speech)
        except Exception:
            return ""

    @session.on("agent_state_changed")
    def on_agent_state(event: Any) -> None:
        if str(getattr(event, "new_state", "") or "") == "speaking":
            opening["agent_speaking_starts"] += 1
        if opening["decider"] is not None and str(getattr(event, "new_state", "") or "") == "speaking":
            # The opening's first audible frame (a no-op before the opening
            # was requested, and after the first time).
            was_opening = opening["decider"].state == OPENING_TURN_OPENING
            if s2s_live_call:
                if native_opening_authorization is not None and native_opening_authorization.note_speaking(at=time.monotonic()):
                    opening["first_reply_path"] = opening["first_reply_path"] or "native_pickup"
                    opening_released.set()
            else:
                opening["decider"].note_opening_first_audio(at=time.monotonic())
            if was_opening and transcript_gate is not None:
                # The window armed at dispatch has done its job: from the
                # first frame the reply-in-flight gating holds transcripts,
                # and a barge-in that pauses the reply releases them, as on
                # any reply. Held any longer, a callee talking over the
                # opening could not interrupt it. The reply gate is armed
                # FIRST: a literal opening never reports ``thinking``, and
                # closing the opening window with neither gate up would
                # release a held pickup final straight into the runtime.
                transcript_gate.set_reply_in_flight(True)
                transcript_gate.set_opening_turn_in_flight(False)
        if end_call_guard is not None and str(getattr(event, "new_state", "") or "") == "speaking":
            # The observable audio outcome per speech: playout started for
            # the handle that is current now (a synthesis failure leaves a
            # finished handle with no exception and no audio).
            try:
                current = getattr(session, "current_speech", None)
                end_call_guard.utterances.note_speaking(str(getattr(current, "id", "") or ""))
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass
        # The edge names its SpeechHandle on both engines. A worker-owned
        # cascade farewell must not borrow a preamble's interval after reclaim.
        turn_events.on_agent_state_changed(
            event, speech_id=current_speech_id()
        )
        # A reply reaching the line clears the silent-model clock. The state
        # edge rather than a synthesis callback: the background track the
        # acknowledgement and tool-wait fillers play on never moves it, so a
        # cached filler cannot keep a call with a dead model alive.
        if str(getattr(event, "new_state", "")) == "speaking":
            model_unresponsive.note_agent_output(OUTPUT_REPLY, at=time.monotonic())
            # The latest time the agent STARTED speaking, whoever's speech it
            # was. Observation only -- unlike the goodbye watcher below it
            # cancels nothing -- so a voice-to-voice goodbye, which the model
            # speaks itself, can key the egress drain on its real start.
            agent_speaking_started["at"] = time.monotonic()
            mutual_silence.note_agent_speaking(at=agent_speaking_started["at"])
        scheduler = acknowledgements["scheduler"]
        if scheduler is not None:
            # The background-track speaker never moves the agent state, so this
            # only arms on "thinking" and cancels on the reply's own "speaking".
            scheduler.on_agent_state_changed(event.old_state, event.new_state)
        feedback = acknowledgements["tool_wait"]
        if feedback is not None:
            feedback.on_agent_state_changed(event.old_state, event.new_state)
        # Stamp the goodbye's real playout-start: the first "speaking" edge whose
        # current_speech IS the armed goodbye handle. mark_state above already
        # opened this interval at the same edge, so monotonic-now here sits at or
        # just after interval.start -- the recorder binds by CONTAINMENT of this
        # stamp. Identity by speech_id means a preamble's speaking edge (a
        # different current_speech) can never claim it.
        if (
            str(getattr(event, "new_state", "")) == "speaking"
            and goodbye_playout["speech_id"]
            and goodbye_playout["at"] is None
        ):
            current = getattr(session, "current_speech", None)
            if current is not None and str(getattr(current, "id", "")) == goodbye_playout["speech_id"]:
                goodbye_playout["at"] = time.monotonic()
                # Only an authoritative close has stopped admission. A model
                # proposal still accepts caller input and its goodbye is
                # interruptible: session.interrupt() would cut that goodbye.
                if not semantic_admission.admits_transcript():
                    # A platform limit may claim a proposal AFTER say() made
                    # its handle interruptible. Never cut that handle ourselves.
                    if not bool(getattr(current, "allow_interruptions", True)):
                        cancel_stray_model_generation(session)
        if transcript_gate is not None:
            # BEFORE the measurement below, and before the release: a barge-in
            # whose transcript is still in the buffer has to be recorded as
            # having one. The release goes through the STT node and the SDK
            # asynchronously, so user_input_transcribed cannot have fired yet.
            gate_state = transcript_gate.gate.stats
            interruption_telemetry.note_withheld_transcripts(gate_state["pending_speech"])
            # Read at the same instant and for the same reason: the caller's
            # words are in the gate's buffer, not in the SDK, when playout
            # stops. This is the number the founder's A/B compares before
            # choosing a default for the setting.
            interruption_telemetry.note_caller_words(
                heard=gate_state["words_heard"],
                at_qualification=gate_state["words_at_qualification"],
            )
        interruption_telemetry.on_agent_state_changed(
            event.old_state,
            event.new_state,
            # The handle whose first frame just played. Read at the edge, so
            # the measurement taken at the exit edge belongs to this handle
            # even if the session has moved on by then; an unhooked farewell's
            # number can then never be stamped onto a later failed handle.
            speech_id=str(getattr(getattr(session, "current_speech", None), "id", "") or ""),
        )
        if transcript_gate is not None:
            # The release trigger for held caller transcripts. Leaving the
            # reply-in-flight states covers both endings that matter: the
            # reply finished on its own, and a VAD-confirmed barge-in paused
            # it -- after which the caller's words are exactly what the
            # session needs.
            reply_in_flight = str(event.new_state) in REPLY_IN_FLIGHT_AGENT_STATES
            transcript_gate.set_reply_in_flight(reply_in_flight)
            if not reply_in_flight:
                # The reply is over (or a qualified barge-in just paused it).
                # Whatever the word requirement was holding, it is not holding
                # this turn any more -- and the held transcripts released one
                # line above are about to reach a runtime that refuses to
                # answer a completed user turn while the current speech is
                # uninterruptible.
                _release_interruption_word_gate(
                    interruption_word_gate, by_words=False, call_id=call.call_id
                )
        # AFTER the line above, never before: at ``speaking`` the reply is
        # already in flight, so the combined gate predicate stays true and the
        # caller's held words are not handed to the SDK in the gap between the
        # two signals. ``speaking`` is raised from the first audio frame's
        # callback: from here the caller can talk over the opening, and the
        # protection starts measuring their speech against the pickup bound.
        # The handle is handed back by ``release_opening`` below -- when the
        # caller keeps talking past that bound, or when the say ends -- never
        # from this transition.
        opening_protection.note_agent_state(event.new_state)
        # LAST, and only at ``speaking``. Three reasons this is the arming
        # point rather than ``speech_created``:
        #
        #  * the runtime gates a reply's wait for user silence on the same
        #    flag (``agent_activity.py`` :2636 and :3088), and this stack runs
        #    preemptive generation -- so a handle held uninterruptible at
        #    CREATION would skip that wait and start talking over a caller who
        #    was still speaking;
        #  * by ``speaking`` the handle exists and is the session's current
        #    speech, with no race against the scheduler;
        #  * the greeting has just been handed back by the opening protection
        #    above, so this one call site covers the greeting and every
        #    ordinary reply identically instead of colliding with the other
        #    writer of the flag.
        #
        # The cost is that the ``thinking`` window -- turn commit to first
        # audio frame -- stays duration-only. Nothing is audible there for a
        # caller to talk over, and the transcript-cancel path that made that
        # window dangerous is already closed by the gate above.
        if transcript_gate is not None and str(event.new_state) == "speaking":
            # Re-scoped FIRST: the thinking window is 2-3 s of LLM/TTS latency
            # in this stack and a backchannel there ("mm", "okay") would
            # otherwise satisfy the requirement before the reply was audible,
            # leaving the next wordless burst free to cut it on the duration
            # floor alone. A caller speaking continuously across the boundary
            # loses nothing: a streaming transcriber's events are cumulative
            # within an utterance, so the words re-deliver themselves on the
            # next one.
            transcript_gate.note_reply_became_audible()
            if not transcript_gate.gate.interruption_qualified:
                interruption_word_gate.arm(getattr(session, "current_speech", None))

    @session.on("speech_created")
    def on_speech_created(event: Any) -> None:
        # The earliest handle-creation seam, for OBSERVATION only -- it never
        # arms the gate. A reply cancelled before its first TTS frame never
        # reaches tts_node, so hooking only there would under-count those
        # cancellations to zero. Arming from here instead would also catch the
        # speculative handles preemptive generation creates and never
        # schedules, which would withhold the later segments of a
        # multi-segment caller utterance.
        try:
            session_agent.observe_interruption_completion(event.speech_handle)
        except Exception:
            emit("interruption_observer_not_attached", call_id=call.call_id)
        if navigation_actions is not None:
            # A speech window (a spoken answer, the person probe) takes the first handle.
            navigation_actions.on_speech_created(event)
        # Held until the speech finishes, when a turn it took claims the
        # caller speech before it (the uncaptioned-speech moment).
        turn_events.on_speech_created(event)
        if end_call_guard is not None:
            # Attribution of every agent utterance to the mechanism that
            # spoke it (an ordinary reply, text alongside an end_call
            # attempt, the post-refusal regeneration, or the platform's
            # goodbye), settled when the speech is done or cut.
            handle = event.speech_handle
            try:
                end_call_guard.utterances.note_speech(
                    str(getattr(handle, "id", "") or ""),
                    source=str(getattr(event, "source", "") or ""),
                    user_initiated=bool(getattr(event, "user_initiated", False)),
                    created_at=time.monotonic(),
                )
                handle.add_done_callback(
                    lambda done: end_call_guard.utterances.settle(
                        str(getattr(done, "id", "") or ""),
                        list(getattr(done, "chat_items", None) or []),
                        interrupted=bool(getattr(done, "interrupted", False)),
                        done_at=time.monotonic(),
                    )
                )
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                emit("agent_utterance_attribution_failed", call_id=call.call_id)

    @session.on("agent_false_interruption")
    def on_false_interruption(event: Any) -> None:
        # A pause that resumed is evidence the trigger was never a caller
        # turn, and it is the number that separates a false barge-in from a
        # real one when the interruption policy is measured on a live call.
        resumed = bool(getattr(event, "resumed", False))
        # The pause that this resolves carries the measurement the criterion
        # asks for -- how long playout ran before it was cancelled, and
        # whether a transcript was present. The resumed segment would
        # otherwise overwrite it and the false pause would go unmeasured.
        interruption_telemetry.on_false_interruption(resumed=resumed)

    # The SDK's ``transcription_timeout`` (set in the shared stack builder)
    # expires here when the caller stopped and no final transcript arrived.
    # The SDK emits and does nothing else; this is what turns the event into
    # a bounded wait. It is NOT admitted by the managed-PSTN predicate above:
    # the Simulations Lab and the browser tests drive the same transcriber
    # and are where a stall is reproduced, so they get the same recovery --
    # and the same log lines -- as a phone call.
    stall_recovery = TranscriptionStallRecovery(
        session=session,
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        provider=provider_labels["stt"],
        # Logged, not obeyed: whether the word gate is holding a final for
        # this reply says whether the rebuild was needless, and nothing about
        # a later utterance in the same reply that stalled. See
        # CallerTranscriptGate.holds_final_transcript.
        holds_final=(
            (lambda: transcript_gate.gate.holds_final_transcript)
            if transcript_gate is not None
            else None
        ),
        # Once end_call has closed admission nothing released now may start
        # a turn, so nothing is worth flushing.
        admits=semantic_admission.admits_transcript,
    )

    # The same expiry, read for what it says about the CALLER rather than
    # about the transcriber. The recovery above answers "can this
    # provider be made to finalize"; this answers "has the caller now been
    # sitting in dead air", which on a recognizer pinned to one language is
    # what a caller speaking another language produces. It is consulted with
    # the recovery's own answer so it can stand down while a flush may still
    # deliver the words it would otherwise claim were never understood.
    unrecognised_speech = UnrecognisedSpeechReply(
        settings=unrecognised_speech_settings(call.config),
        instruction=repair_instruction(profile.language, call.config.get("languages")),
        speak_phrase=lambda text: session.say(
            text, allow_interruptions=True, add_to_chat_ctx=True
        ),
        # ``tool_choice="none"``: the agent has just been told it did not
        # understand the caller, which is the worst possible moment for the
        # model to reach for ``end_call`` or a webhook. The line is a repair,
        # never an action.
        speak_generated=lambda instructions: session.generate_reply(
            instructions=instructions, tool_choice="none", allow_interruptions=True
        ),
        emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
        # A live engine has no separate STT step to go silent, so this event
        # cannot mean there what it means on a cascade call. Asked of the
        # engine's DECLARATION rather than of `is_s2s_live_config`, whose
        # `frozenset({"gpt_live", "gemini_live"})` is the engine-id literal
        # this repository has spent a ticket deleting: an engine declared
        # tomorrow is covered here by its row and by no edit of this line.
        has_language_axis=engine_has_language_axis(call.config),
        # A menu nobody transcribed is not a caller to repair, and a worker-initiated reply must not take a
        # speech window the model has still to answer.
        suspended=lambda: navigation.suspends(SPEECH_RECOVERY),
        # A pinned workflow run owns every reply: governed WorkflowRoot delivers
        # that by overriding llm_node, so a generate_reply here would speak
        # and STEP the flow's next turn rather than this line.
        replies_are_scripted=False,
        # The gate the harness measured: the same en+de agent is silent
        # without follow-the-caller and healthy with it, so the variable is
        # the switcher, not how many languages the agent declares.
        has_language_switcher=language_switcher is not None,
        holds_final=(
            (lambda: transcript_gate.gate.holds_final_transcript)
            if transcript_gate is not None
            else None
        ),
        admits=semantic_admission.admits_transcript,
        agent_state=lambda: str(getattr(session, "agent_state", "") or ""),
    )
    # The one writer of `engine_has_no_transcriber`. On an engine with no
    # separate transcription step nothing below can ever run -- the SDK emits
    # no `user_transcription_timeout` without an STT node -- so without this
    # line such a call's evidence for the repair line is an absence, which is
    # indistinguishable from a worker that never wired the module at all. A
    # no-op on cascade.
    unrecognised_speech.on_call_started()

    @session.on("user_transcription_timeout")
    def on_user_transcription_timeout(event: Any) -> None:
        recovery = stall_recovery.on_timeout(event)
        if opening_owns_turn():
            # The pickup belongs to the opening decider: a repair line here
            # would be a second owner of the first response.
            emit("unrecognised_speech_held_for_opening", call_id=call.call_id)
            return
        unrecognised_speech.on_transcription_timeout(recovery=recovery)

    @session.on("user_input_transcribed")
    def on_user_transcript(event: Any) -> None:
        caption_ingested_at = time.monotonic()
        turn_events.on_user_input_transcribed(event, ingested_at=caption_ingested_at)
        if turn_end_evidence is not None:
            # Whether it carried words, never the words.
            turn_end_evidence.note_transcript(
                is_final=getattr(event, "is_final", False),
                has_text=bool(str(getattr(event, "transcript", "") or "").strip()),
            )
        # A no-op unless this is a GPT-Live call: there the engine's live
        # captions are the second witness for the silence rule (they reach
        # this handler ungated -- GPT-Live has no STT node and so no
        # transcript gate), and the SDK does not move user_state from the
        # realtime input-speech event while a client VAD is configured.
        caller_speech_witness.on_user_input_transcribed(getattr(event, "transcript", ""))
        if (
            getattr(event, "is_final", False)
            and opening_owns_turn()
            and str(getattr(event, "transcript", "") or "").strip()
            and len(opening["pickup_finals"]) < 20
        ):
            # The pickup's words, kept for a first reply whose turn is late.
            opening["pickup_finals"].append(str(event.transcript).strip())
        # Any non-blank transcript, interim or final and unfiltered, is the
        # caller's words for the silence rule -- a "yes" to its check-in
        # counts. Dated by its VAD run inside the rule, not by arrival.
        if str(getattr(event, "transcript", "") or "").strip():
            mutual_silence.note_caller_words(at=caption_ingested_at)
            # The same words tell the unresponsive rule's shadow bound that
            # something is still listening to the caller.
            model_unresponsive.note_caller_transcript(at=caption_ingested_at)
        if getattr(event, "is_final", False):
            # Finals only, exactly as the caller-transcript path below reads
            # them. An interim transcript is a growing snapshot of the phrase
            # so far, so counting every one of them would count the same words
            # again and again -- and the count is what decides whether a
            # person is on this call.
            answer_detection.note_transcript(
                getattr(event, "transcript", ""),
                at=caption_ingested_at,
                source=(
                    WORD_SOURCE_LIVE_CAPTION if s2s_live_call else WORD_SOURCE_STT_FINAL
                ),
            )
            if final_router is not None:
                # The detector is the source of finals while it listens; the session's
                # final is advisory otherwise, or supports the episode in an outage.
                final_router.route(getattr(event, "transcript", ""))
            elif far_end_watch is not None:
                # Labelled off the media path; its text never reaches an event.
                far_end_watch.note_final(getattr(event, "transcript", ""))
        interruption_telemetry.on_user_transcript(
            characters=len(str(getattr(event, "transcript", "") or ""))
        )
        # Every final with text the SDK hears, so the recovery's second log
        # line can say whether a forced flush produced one. Text only: the
        # plugin never forwards an empty final, and this is where that
        # silence becomes measurable.
        if getattr(event, "is_final", False) and str(getattr(event, "transcript", "") or "").strip():
            stall_recovery.note_final_transcript()

    native_farewell = NativeFarewellFallback(
        end_call, end_call_fallback_message(profile.language),
        lambda event: emit(event, call_id=call.call_id),
        suspended=lambda: navigation.suspends(ENDER_CLOSING_TEXT_FALLBACK))

    @session.on("conversation_item_added")
    def on_conversation_item(event: Any) -> None:
        turn_events.on_conversation_item_added(event)
        if (
            turn_end_evidence is not None
            and str(getattr(getattr(event, "item", None), "role", "") or "") == "user"
        ):
            turn_end_evidence.note_user_item_added()
        if (
            opening["decider"] is not None
            and str(getattr(getattr(event, "item", None), "role", "") or "") == "user"
            and opening_owns_turn()
        ):
            # The pickup reached the context: through the hook below, or
            # through the reply-less commit, which does not call the hook.
            on_opening_pickup()
        if end_call_evidence is not None and str(
            getattr(getattr(event, "item", None), "role", "") or ""
        ) == "user":
            # The id the SDK actually inserted (under preemptive reuse not the
            # object the hook saw) is the caller turn id the guard pins and
            # budgets on.
            end_call_evidence.bind_turn_item(str(getattr(event.item, "id", "") or ""))
        # The engine produced a response. It is not the trigger and it is not
        # the corroborator either: it is the DISCRIMINATOR that stands the
        # rule down. An assistant item with no audio behind it is a synthesis
        # or playout fault, which is a different defect from a lost model.
        if str(getattr(getattr(event, "item", None), "role", "") or "") == "assistant":
            model_unresponsive.note_model_response(at=time.monotonic())
            # What we said, so the voicemail watch can tell our own
            # words coming back through a speakerphone from the far end's.
            answer_detection.note_agent_text(
                getattr(event.item, "text_content", None) or getattr(event.item, "content", "")
            )
            if far_end_watch is not None:
                far_end_watch.note_agent_text(
                    getattr(event.item, "text_content", None) or getattr(event.item, "content", "")
                )
            if far_end_perception is not None:
                far_end_perception.note_agent_text(
                    getattr(event.item, "text_content", None) or getattr(event.item, "content", "")
                )
        if s2s_live_call:
            item = getattr(event, "item", None)
            role = str(getattr(item, "role", "") or "")
            text = str(getattr(item, "text_content", None) or getattr(item, "content", "") or "")
            native_farewell.note_item(role=role, text=text)

    @session.on("metrics_collected")
    def on_metrics_collected(event: Any) -> None:
        # LiveKit Agents 1.7.1 emits typed STT/LLM/TTS metrics here.  Never
        # serialize/log the event: the collector retains only provider/model
        # identities and numeric usage, and a malformed metrics callback must
        # not interrupt media.
        try:
            billing_usage_collector.collect(event.metrics)
        except Exception:
            emit("component_usage_metrics_rejected", call_id=call.call_id)

    # Post-refusal steps in flight, held so the loop cannot collect them
    # before they finish; each removes itself.
    post_refusal_tasks: set[asyncio.Task[Any]] = set()

    def call_is_ending() -> bool:
        # Re-read on every call: the step's own checks are live, not frozen.
        snapshot = read_guard_snapshot()
        return bool(snapshot.finished or snapshot.ending_scheduled)

    @session.on("function_tools_executed")
    def on_tools(event: Any) -> None:
        # SYNCHRONOUS on purpose: the SDK's emitter invokes handlers inline and
        # reads ``has_tool_reply`` right after, so a coroutine handler would
        # be dropped and the cancel below would never happen.
        emit(
            "model_tools_executed",
            call_id=call.call_id,
            tools=executed_tool_names(event),
        )
        if navigation_actions is not None:
            navigation_actions.on_tools_executed(event)
        # The close owner's own floor refusals, guard or no guard, FIRST: a
        # batch it refused must not get a regenerated reply from the guard's
        # step below while the caller is speaking.
        locally_refused = close_refusal_recovery.on_function_tools_executed(event)
        if end_call_guard is not None:
            call_ids = [
                str(getattr(function_call, "call_id", "") or "")
                for function_call in (getattr(event, "function_calls", None) or [])
            ]
            plan = end_call_guard.post_refusal_plan(call_ids)
            if plan is not None and locally_refused:
                # Its context edit stands; nothing is spoken over the caller.
                plan = plan.without_regeneration()
            if plan is not None:
                # A refusal: the SDK's own follow-up is not generated (batch-
                # wide by SDK design: a sibling tool's output stays in context,
                # its verbalisation goes with it). The rest runs after this
                # handler returns, once the SDK has inserted the tool items:
                # the refused pair leaves the context, the line takes its
                # place, and the reply is regenerated with tools off unless
                # held caller words drive the next turn themselves.
                try:
                    event.cancel_tool_reply()
                except Exception:  # noqa: BLE001 - the SDK's own follow-up then plays
                    pass
                task = asyncio.get_running_loop().create_task(
                        apply_end_call_post_refusal(
                            plan,
                            agent=session_agent,
                            session=session,
                            make_line=lambda text: agents_llm.ChatMessage(role="system", content=[text]),
                            make_context=lambda items: agents_llm.ChatContext(list(items)),
                            finished=call_is_ending,
                            navigation_suspended=lambda: navigation.suspends(SPEECH_RECOVERY),
                            **post_refusal_hooks(end_call_guard),
                            emit=lambda event_name, **details: emit(event_name, call_id=call.call_id, **details),
                            # The provenance-free read: a final that arrived
                            # after the refusal counts whatever the turn's
                            # provenance (unconsumed() alone claims nothing
                            # under unknown provenance).
                            caller_busy=lambda plan=plan: guard_caller_busy(
                                read_guard_snapshot(), end_call_evidence, plan
                            ),
                        ),
                        name="end_call_post_refusal",
                )
                post_refusal_tasks.add(task)
                task.add_done_callback(post_refusal_tasks.discard)
        # A tool ran: the recorder treats the engine's next unasked generation
        # as the answer to its output (Gemini answers a refused end_call's
        # output too). The event precedes the SDK handing the output over, so
        # the exclusion is armed before that reply's speech_created.
        turn_events.on_function_tools_executed(event)

    @session.on("tool_execution_updated")
    def on_tool_execution(event: Any) -> None:
        # Each tool's start and end, as the SDK dispatches them; the owner
        # decides whether the caller hears anything for the wait between.
        feedback = acknowledgements["tool_wait"]
        if feedback is not None:
            feedback.on_tool_execution_updated(event)
        note_tool_execution(event)

    @session.on("error")
    def on_error(event: Any) -> None:
        diagnostics = safe_component_error_diagnostics(getattr(event, "error", None))
        # A provider error raised while this worker is taking the socket down
        # ON PURPOSE is teardown noise, not a fault: there is no caller left
        # for it to affect. Reported as one, it made every unanswered outbound
        # call page as a GPT-Live failure.
        #
        # Tied to that deliberate close and NOT to "the call is ending". A
        # provider failure during ringing ends the call too, and its own error
        # callback can arrive after the close callback -- so a handler keyed
        # on the ending would relabel the page this worker exists to send.
        # Asked about THIS call, so another call's teardown can never answer
        # for it. One window, opened around the close and read here and by the
        # error tracker's filter alike.
        if provider_close_in_progress(call.call_id):
            emit(
                "provider_error_during_teardown",
                call_id=call.call_id,
                cause=call_end_arbiter.reason or "call_end_before_answer",
                **diagnostics,
            )
            return
        emit(
            "session_error",
            call_id=call.call_id,
            **diagnostics,
        )
        # Corroboration for the silent-model rule, never its trigger, and only
        # when the failing component IS the model. The cascade stack is the
        # only engine measured to produce one of these during a mid-call
        # provider outage, so a rule that required it would be dead on both
        # voice-to-voice engines -- but a synthesis fault beside an assistant
        # response is the case that rule stands DOWN for, so it must not be
        # able to corroborate the opposite verdict.
        model_unresponsive.note_provider_error(
            at=time.monotonic(),
            component=livekit_error_component(getattr(event, "error", None)),
        )

    # The engine giving up underneath a live call is an ending, and until now
    # it was the one ending nothing acted on: the session closed, the worker
    # stayed parked on ``finished.wait()``, and the caller's leg ran, billed,
    # with nobody on it. Stamped through the arbiter like every other ending
    # this worker decides, so the teardown, the usage report and the terminal
    # lifecycle run once and carry the reason.
    session_closed = production_session_close_handler(
        stamp_lost=lambda: call_end_arbiter.end(
            TERMINATION_MODEL_SESSION_LOST, detected_via=DETECTED_VIA_SESSION_CLOSED
        ),
        stamp_refused=lambda: call_end_arbiter.end(
            TERMINATION_MODEL_SESSION_LOST, detected_via=DETECTED_VIA_SESSION_CLOSED,
            reason_detail=DETAIL_TRANSCRIBER_REFUSED_REQUEST,
        ),
        is_finishing=finished.is_set,
    )

    @session.on("close")
    def on_close(event: Any) -> None:
        nonlocal browser_test_report_task
        # A browser test that connects, transcribes and never speaks looks
        # healthy from the page: the transcript fills with the customer's own
        # words and nothing says the pipeline is dead. The worker is the only
        # party that knows, so tell the room. A phone call has no dock
        # listening; its ending is the stamp below, and only a phone call
        # takes it: the browser test keeps its report-and-deadline contract.
        #
        # The CLOSE, not the error above: LiveKit absorbs several unrecoverable
        # component errors before it gives up, and resets that budget on a turn
        # that works. Reporting an error would end sessions it was about to
        # recover.
        if browser_test is None:
            session_closed(event)
            return
        component = browser_test_close_component(
            reason=getattr(event, "reason", None), error=getattr(event, "error", None)
        )
        if component is None:
            return
        browser_test_report_task = asyncio.create_task(report_browser_test_failure(component))

    def caller_left() -> None:
        """The caller's leg is gone -- unless this call was already ending.

        The SAME ending as the occupancy poll finds, so a customer's reason
        answers "why did it end" rather than "which of our internal paths
        noticed". The two are told apart by the provenance marker, which never
        leaves our own evidence.

        The stand-down matters more here than in the watchdog. Deleting the
        room during teardown KICKS the SIP leg, so this event arrives on almost
        every ending, including the ones the agent itself decided. Without the
        guard a workflow that finished normally would be reported to the
        customer as the other party hanging up, and ``finish_session`` would run
        a second time against a teardown already settling the wallet.
        """

        closing_evidence.record("media_ended", source="participant", owner="transport", reason="media_disconnect")
        if finished.is_set():
            return
        call_end_arbiter.end(
            TERMINATION_REMOTE_PARTY_DISCONNECTED,
            detected_via=DETECTED_VIA_DISCONNECT_EVENT,
            # Read as the party leaves: a hang-up while waiting in a queue is
            # retried like an unanswered call (see ``DETAIL_DURING_HOLD``).
            reason_detail=navigation.remote_hangup_detail(),
        )

    ctx.room.on(
        "participant_disconnected",
        caller_disconnect_handler(
            caller_identity=participant.identity,
            finish=caller_left,
        ),
    )
    if navigation_actions is not None:
        # A second SIP participant hears what the call says: in-flight keys and the controller's audio stop at once.
        ctx.room.on("participant_connected", navigation_actions.on_participant_change)
        ctx.room.on("participant_disconnected", navigation_actions.on_participant_change)
    audio_environment_player: Any | None = None
    # Voice-to-voice only: what the model hears is withheld from input release
    # until its opening has been spoken. See the duplex_opening_hold module.
    duplex_opening_hold: DuplexOpeningHold | None = None
    native_caller_floor = None
    duplex_opening_hold_task: asyncio.Task[None] | None = None
    browser_startup = BrowserStartup() if browser_test is not None else None

    def browser_opening_failed(error: BaseException | None) -> None:
        nonlocal browser_test_report_task, browser_test_failure_reported
        if browser_startup is None or not browser_startup.opening_failed(error):
            return
        browser_test_failure_reported = False
        call.browser_outcome = browser_startup.freeze(call.browser_session_id)
        write_session_state(state_path, browser_outcome=call.browser_outcome,
                            browser_outcome_pending=True)
        call_end_arbiter.end("browser_startup_failed", detected_via="opening_failure")
        browser_test_report_task = asyncio.create_task(report_browser_test_failure("startup"))

    def request_scoped_opening_recovery() -> Any:
        """The opening owner's one recovery for a request-scoped (cascade) reply.

        The same per-call intent, single retry and reply budget as a duplex
        reconnect recovery; only the safety evidence differs (see
        ``live_reply_recovery``). A caller turn or a terminal owner wins.
        """
        from live_reply_recovery import OpeningReplyRecovery
        return OpeningReplyRecovery(
            ended=finished, caller_revision=lambda: opening["caller_speaking_starts"],
            may_generate=lambda: not end_call.ending_scheduled and not tool_execution["open"] and not navigation.suspends(SPEECH_RECOVERY),
            audible=lambda: opening["agent_speaking_starts"],
            emit=lambda **details: _quietly(emit, "opening_reply_recovery", call_id=call.call_id, **details))

    async def end_for_opening_failure(site: str) -> None:
        """An opening that never reached the caller ends the call once, as itself.

        Left silent, the line would wait for a person who is waiting for the
        agent, and end later under a reason that blames the caller.

        EXCEPT when it was a phone menu or a hold queue that cut it off: the
        opening is spoken at the answer, the far end's own speech then stops it,
        and that stop is the navigation's, intentional -- the explicit outcome
        every failure site below gives a stop, on a live engine and on cascade.
        The call then goes on under the navigation bounds, and the greeting
        counts as spoken (the one re-greeting owed to a person is the recovery
        for a caller who never heard it).
        """
        if finished.is_set():
            return
        if await navigation_gate.opening_failure_excused(site):
            try:
                answer_detection.arbiter.note_greeting_spoken(at=time.monotonic())
            except Exception:  # noqa: BLE001 - bookkeeping never decides a call
                pass
            return
        if not finished.is_set():
            call_end_arbiter.end("agent_unresponsive", detected_via="opening_failure")

    async def end_call_for_unheard_opening(error: BaseException, site: str) -> None:
        _quietly(emit, "opening_generation_failed", call_id=call.call_id,
                 error_type=type(error).__name__)
        await end_for_opening_failure(site)
    try:
        # A duplex model generates from whatever it hears, not only from an
        # explicit reply request: on an outbound call the early media the
        # carrier plays before the callee answers (ringback, network
        # announcements) rides the SAME caller track this session is about
        # to subscribe to, seconds before ``may_speak`` below is reached. A
        # measured production call spoke its opening 9-15s before the
        # callee picked up because of exactly this.  ``set_audio_enabled``
        # is on ``AgentInput`` itself, upstream of both the cascade STT node
        # and a realtime session's own audio input, so muting it here holds
        # EVERY engine's input the same way with no branch on which one this
        # call is running.  Set before ``start`` so ``RoomIO`` -- which
        # assigns ``session.input.audio`` through this same setter -- finds
        # the input already muted for its very first frame; a mute issued
        # after ``start`` returns cannot close that race.
        session.input.set_audio_enabled(False)
        if browser_startup is not None:
            session.output.set_audio_enabled(False)
        if live_pickup is not None:
            if live_pickup.buffer.error is not None:
                raise ProductionConfigurationError("live pickup capture failed before model startup")
            session.input.audio = live_pickup
        if navigation_actions is not None and not s2s_live_call:
            # The declared voice is known from the call's stack, not from the track, so it is attached
            # before the session can take a request: a spoken value asked for in the first seconds
            # must not read "no voice" and end the call. Only the track is awaited (in the speaker).
            speak_voice, speak_covers = cascade_voice(
                lambda: session,
                declared=lambda language: load_voice_registry().declares_language(
                    "tts", profile.tts_registry, profile.tts_model, language
                ),
                resampler_factory=lambda rate: rtc.AudioResampler(rate, 48_000, num_channels=1),
                frame_factory=lambda data, rate, samples: rtc.AudioFrame(
                    data=data, sample_rate=rate, num_channels=1, samples_per_channel=samples
                ),
                connect_options=APIConnectOptions(max_retry=0),
            )
            # (a held value is said only by the voices and languages released for it: declared data, narrower than
            # the languages the voice speaks, which still cover the fixed words)
            speak_value_covers = cascade_value_covers(
                lambda language: voice_says_held_value(
                    load_voice_registry(), profile.tts_registry, profile.tts_model, language
                )
            )
            navigation_actions.controller_speaker.attach_voice(
                speak_voice, speak_covers, covers_value=speak_value_covers,
                silence=lambda seconds: rtc.AudioFrame(
                    data=bytes(2 * int(48_000 * seconds)), sample_rate=48_000, num_channels=1,
                    samples_per_channel=int(48_000 * seconds),
                ),
            )
        if caller_inputs is not None:
            from action_listener import ConfirmationListener
            confirmation_listener = ConfirmationListener(call=call, inputs=caller_inputs, environment=os.environ)
            await confirmation_listener.start(room=ctx.room, participant=participant)
        await session.start(
            room=ctx.room,
            agent=session_agent,
            room_options=RoomOptions(
                # By name: without it RoomIO links the FIRST participant, and
                # the evidence sidecar is already in the room and eligible.
                participant_identity=participant.identity,
                # RoomIO also filters that participant by kind, and its default
                # kinds exclude the AGENT a synthetic caller joins as.
                participant_kinds=linked_participant_kinds(
                    browser_test,
                    agent_kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                    sdk_default=NOT_GIVEN,
                ),
                audio_input=live_pickup is None,
                audio_output=True,
                close_on_disconnect=False,
                delete_room_on_close=False,
            ),
        )
        if navigation_actions is not None:
            navigation_actions.probe_speaker.playback.bind(
                far_end_perception.pickup.note_agent_speaking if far_end_perception is not None else None)
        if live_pickup is not None:
            from opening_output import OpeningOutput
            native_opening_output = OpeningOutput(session.output.audio)
            session.output.audio = native_opening_output
            native_opening_authorization = NativeOpeningAuthorization(session, opening["decider"],
                input_delivered_through=lambda: live_pickup.buffer.delivered_through,
                emit=lambda **details: _quietly(emit, "native_opening_authorization", call_id=call.call_id, **details))
            if is_gemini_live_config(call.config):
                from gemini_opening import GeminiPickupGeneration
                gemini_pickup_generation = GeminiPickupGeneration(session_agent.realtime_llm_session)
            live_pickup.buffer.release()
        if s2s_live_call:
            def native_floor_failed() -> None:
                try:
                    emit("native_caller_floor_failed", call_id=call.call_id)
                finally:
                    finish_session()
            def note_native_floor(speaking: bool) -> None:
                nonlocal native_caller_revision
                if speaking:
                    native_caller_revision += 1
                    # The worker's own VAD on the caller's frames, before the
                    # model receives them: a withdrawal's precondition.
                    transfer_controller.note_caller_speech_started("native_floor")
                    if native_opening_authorization is not None:
                        native_opening_authorization.note_caller_started()
                end_call.note_acoustic_caller_speaking(speaking)
                native_farewell.note_caller_speaking(speaking)
            native_caller_floor = install_native_caller_floor(
                session, ctx.proc.userdata["vad"],
                note=note_native_floor, failed=native_floor_failed)
            if navigation_actions is not None and isinstance(live_input_backend, SilencedInput):
                # After the floor, so the floor and the detector keep the real frames.
                live_input_backend.install(session)

        # The cadence has been running since long before this await, and the
        # answer can land inside it; the deadline needs this second clock.
        caller_audio_bind_witness.note_session_started()
        if browser_startup is not None:
            sink = session.output.audio
            if sink is not None:
                session.output.audio = BrowserStartupOutput(sink, browser_startup)
                browser_startup.installed = True
            session.output.set_audio_enabled(True)
        gpt_live_injected = False
        if s2s_live_call:
            # The provider session exists from session start; witnessed before
            # the context injection waits on it. A realtime (not duplex)
            # engine raises here, and stays ``unsupported``.
            try:
                live_session = session_agent.duplex_session
            except Exception:  # noqa: BLE001 - absent is an answer, not a fault
                live_session = None
            if live_session is not None:
                try:
                    live_session.on("openai_client_event_queued", closing_evidence.observe_native)
                except Exception:
                    closing_evidence.capture_errors += 1
            emit(
                "duplex_liveness_installed",
                call_id=call.call_id,
                installed=(
                    duplex_liveness.install(live_session) if live_session is not None else False
                ),
            )
            gpt_live_injected = await await_duplex_session(
                session_agent,
                model=gpt_live_llm,
            )
            from live_reply_recovery import OpeningReplyRecovery
            candidate_recovery = OpeningReplyRecovery(
                ended=finished, caller_revision=lambda: native_caller_revision,
                may_generate=lambda: not end_call.ending_scheduled and not tool_execution["open"] and not navigation.suspends(SPEECH_RECOVERY),
                emit=lambda **details: _quietly(emit, "opening_reply_recovery", call_id=call.call_id, **details))
            try:
                recovery_installed = candidate_recovery.install(session_agent.realtime_llm_session)
            except Exception as exc:
                candidate_recovery.close()
                recovery_installed = False
                _quietly(emit, "opening_reply_recovery", call_id=call.call_id,
                         stage="install_failed", error_type=type(exc).__name__)
            _quietly(emit, "opening_reply_recovery_installed", call_id=call.call_id,
                     installed=recovery_installed)
            if recovery_installed:
                opening_reply_recovery = candidate_recovery
            # Wrapped now, while input is still disabled, so no caller frame
            # can reach the model through an unwrapped session. An engine
            # without a duplex session (Gemini Live) is left as it was and
            # recorded as ``unsupported`` when the hold would have engaged.
            duplex_opening_hold = DuplexOpeningHold(
                opening_text=profile.first_message,
                literal=str(profile.opening_mode or "").strip().casefold() == "literal",
                agent_audible_until=recorder.agent_audible_until,
                observed_until=recorder.observer_last_frame_at,
                agent_audible_seconds=recorder.agent_audible_seconds,
                observed_since=recorder.observer_resumed_at,
                emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            )
            duplex_opening_hold.install(live_duplex_session(session_agent, gpt_live_llm))
            if voice_frontend(call.config) == "gpt_live" and not duplex_opening_hold.installed:
                raise ProductionConfigurationError(
                    "GPT-Live opening input protection could not be installed"
                )
            answer_detection.set_words_unobservable_until(
                duplex_opening_hold.words_unobservable_until
            )
            # The witness sits inside the hold and sees the silence it swaps
            # in; this is how its record tells that silence was chosen.
            opening_hold_for_witness = duplex_opening_hold
            duplex_liveness.note_withheld_by(lambda: opening_hold_for_witness.withholding)
            # Speech the model never heard cannot be what its next turn
            # answered, so it never becomes an uncaptioned-speech moment.
            turn_events.note_caller_audio_withheld_by(lambda: opening_hold_for_witness.withholding)
        # The optional environment is deliberately started only after the
        # caller input/session chain exists.  The pinned LiveKit primitive
        # publishes a separate local output track; it never gets subscribed
        # as STT input.  Any integrity/SDK/media failure is logged and leaves
        # the core call, its admission and its reservation unchanged.
        try:
            audio_environment_player = await start_outgoing_audio_environment(
                call.config.get("audio_environment"),
                room=ctx.room,
                agent_session=session,
            )
            if audio_environment_player is not None:
                selected_environment = call.config.get("audio_environment") or {}
                emit(
                    "audio_environment_started",
                    call_id=call.call_id,
                    preset_id=str(selected_environment.get("preset_id") or ""),
                    asset_version=str(selected_environment.get("asset_version") or ""),
                    channel="outgoing_background_track",
                )
        except AudioEnvironmentUnavailable as exc:
            selected_environment = call.config.get("audio_environment") or {}
            emit(
                "audio_environment_unavailable",
                call_id=call.call_id,
                preset_id=str(selected_environment.get("preset_id") or ""),
                asset_version=str(selected_environment.get("asset_version") or ""),
                error_type=type(exc).__name__,
            )
        # Acknowledgements ride a DEDICATED background track: same track name
        # as the ambience (so the evidence sidecar keeps ignoring it) but its
        # own player, so cutting a filler can flush that player's queue without
        # touching the ambience.  Clips are synthesized in the background with
        # the call's own adapter, so the greeting is never delayed by them and
        # the first acknowledgement simply waits for its clip.
        # Tool-wait feedback shares that player and, in ``acknowledge`` mode,
        # the clip pipeline; either feature alone is reason to start them.
        if navigation_actions is not None:
            # The controller speaker's OWN track: it speaks a supplied value or the fixed words
            # of a callback decline without the conversational model and without the output
            # switch the model's voice goes through. A bare source with a short queue (no mixer),
            # same track name as the ambience, so the evidence sidecar keeps skipping it. A cascade call speaks with the call's own TTS
            # where the registry declares it for the menu's language; a voice-to-voice call has
            # none here, and a spoken value is refused (never the model's voice).
            navigation_player = await start_controller_track(ctx.room)
            if navigation_player is None:
                emit("navigation_controller_track_unavailable", call_id=call.call_id)
                # Never coming: a spoken value is refused before it is fetched, not after it is armed.
                navigation_actions.controller_speaker.retire_player_wait()
            else:
                navigation_actions.controller_speaker.attach_player(navigation_player, flush=flush_controller_track)
        if ack_settings.enabled or tool_wait.enabled:
            error_budget = getattr(
                getattr(session, "conn_options", None), "max_unrecoverable_errors", None
            )
            # Clip synthesis shares the session's adapter and a refused
            # phrase counts as one unrecoverable error; a session that
            # closes on its first error (the browser test) must not be
            # ended by an optional filler.  Reported, not silent.  The
            # typing bed synthesizes nothing, so the budget never keeps it off.
            clips_allowed = error_budget is None or error_budget >= 1
            if not clips_allowed:
                if ack_settings.enabled:
                    emit(
                        "acknowledgements_unavailable",
                        call_id=call.call_id,
                        reason="session_error_budget",
                    )
                if tool_wait.mode == "acknowledge":
                    emit(
                        "tool_wait_feedback_unavailable",
                        call_id=call.call_id,
                        reason="session_error_budget",
                    )
            wants_scheduler = ack_settings.enabled and clips_allowed
            wants_tool_wait = tool_wait.mode == "sound" or (
                tool_wait.mode == "acknowledge" and clips_allowed
            )
            if wants_scheduler or wants_tool_wait:
                ack_player = await start_filler_background_player(
                    room=ctx.room, agent_session=session
                )
                acknowledgements["player"] = ack_player
                if ack_player is None:
                    if wants_scheduler:
                        emit(
                            "acknowledgements_unavailable",
                            call_id=call.call_id,
                            reason="background_track_unavailable",
                        )
                    if wants_tool_wait:
                        emit(
                            "tool_wait_feedback_unavailable",
                            call_id=call.call_id,
                            reason="background_track_unavailable",
                        )
                else:
                    ack_clips: dict[str, list[Any]] = {}
                    scheduler = None
                    if wants_scheduler:
                        scheduler = AcknowledgementScheduler(
                            policy=AcknowledgementPolicy(
                                language=profile.language,
                                phrases=ack_settings.phrases,
                                # Rotate only over phrases whose clip exists.
                                is_available=lambda phrase: bool(ack_clips.get(phrase)),
                            ),
                            speak=background_speaker(ack_player, ack_clips),
                            delay_seconds=ack_settings.delay_seconds,
                            emit=lambda event, **details: emit(
                                event, call_id=call.call_id, **details
                            ),
                            # Not into a phone menu or a hold queue.
                            suppressed=lambda: navigation.suspends(SPEECH_ACKNOWLEDGEMENT),
                        )
                        acknowledgements["scheduler"] = scheduler
                    if wants_tool_wait:

                        def emit_tool_wait(event: str, **details: Any) -> None:
                            # Into the bundle first: what the caller heard
                            # during a tool call is call evidence, not only
                            # a log line.
                            recorder.note_tool_wait_feedback(event, **details)
                            emit(event, call_id=call.call_id, **details)

                        acknowledgements["tool_wait"] = ToolWaitFeedback(
                            mode=tool_wait.mode,
                            phrases=tool_wait.phrases,
                            delay_seconds=tool_wait.delay_seconds,
                            speak=background_speaker(ack_player, ack_clips),
                            sound=lambda: typing_sound(ack_player),
                            acknowledgements=scheduler,
                            emit=emit_tool_wait,
                            suppressed=lambda: navigation.suspends(SPEECH_ACKNOWLEDGEMENT),
                        )
                    # One clip per distinct phrase across both features;
                    # ``sound`` mode contributes none and needs no synthesis,
                    # and nothing is synthesized when the budget forbids it.
                    clip_phrases = (
                        tuple(dict.fromkeys((*ack_settings.phrases, *tool_wait.phrases)))
                        if clips_allowed
                        else ()
                    )

                    async def prepare_clips() -> None:
                        try:
                            clips = await synthesize_acknowledgement_clips(
                                stack.tts,
                                clip_phrases,
                                resampler_factory=lambda rate: rtc.AudioResampler(
                                    rate, 48_000, num_channels=1
                                ),
                                frame_factory=lambda data,
                                rate,
                                samples: rtc.AudioFrame(
                                    data=data,
                                    sample_rate=rate,
                                    num_channels=1,
                                    samples_per_channel=samples,
                                ),
                                connect_options=APIConnectOptions(max_retry=0),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:  # noqa: BLE001 - optional audio only
                            emit(
                                "acknowledgement_clips_failed",
                                call_id=call.call_id,
                                error_type=type(exc).__name__,
                            )
                            return
                        ack_clips.update(clips)
                        emit(
                            "acknowledgement_clips_ready",
                            call_id=call.call_id,
                            clips=len(clips),
                            phrases=len(clip_phrases),
                        )

                    # Started only once the opening has been spoken (below):
                    # an outbound leg that never answers must not pay for
                    # clips, and the greeting must not compete with them.
                    if clip_phrases:
                        acknowledgements["prepare"] = prepare_clips
        # Why a requested feature stayed off is reported, never silent, for
        # each feature on its own.
        if not ack_settings.enabled and ack_settings.reason:
            emit(
                "acknowledgements_unavailable",
                call_id=call.call_id,
                reason=ack_settings.reason,
            )
        if tool_wait.reason:
            emit(
                "tool_wait_feedback_unavailable",
                call_id=call.call_id,
                reason=tool_wait.reason,
            )
        # The runtime yields the floor to a talking far end by pausing this
        # output, and bounds that pause nowhere: on an IVR or autoresponder a
        # finished utterance can wait indefinitely; one measured call held a
        # completed reply for 29.5 s of silence. Installed here, after
        # RoomIO published the track and before the first word is spoken, so
        # the opening turn is covered too.
        #
        # A call that counts caller words on a transcriber whose timing is
        # measured also caps a hold with NO words behind it:
        # background noise scores as speech on the VAD, held a finished reply,
        # and the caller's "Hello?" into the silence then cancelled it. Words
        # are what separate the two. The released reply goes to the word gate,
        # so one real word still interrupts it. Scoped by the SAME resolved
        # requirement the word gate runs on, and by the per-stack table in
        # playout_hold.py; every other call -- other providers and languages,
        # and every voice-to-voice engine -- keeps the 8 s budget and the
        # stand-down.
        hold_transcriber = call.config.get("transcriber")
        hold_transcriber = hold_transcriber if isinstance(hold_transcriber, Mapping) else {}
        no_word_hold_budget = no_word_hold_budget_seconds(
            stt_registry_name=hold_transcriber.get("registry_name"),
            stt_model=hold_transcriber.get("model"),
            language=call.config.get("language"),
            word_requirement=word_requirement,
            voice_to_voice=s2s_live_call,
        )
        counts_hold_words = transcript_gate is not None and no_word_hold_budget is not None
        playout_hold = guard_playout_hold(
            session,
            # Same wrapper carries the boundary stop: it is the same
            # pause() both policies have an opinion about.
            boundary=boundary_stop,
            on_release=lambda held_ms, reason: emit(
                "playout_hold_released",
                call_id=call.call_id,
                held_ms=held_ms,
                reason=reason,
            ),
            on_boundary_tail_granted=interruption_telemetry.note_stop_boundary_tail,
            word_evidence=(
                (lambda: transcript_gate.gate.caller_words_observed)
                if counts_hold_words
                else None
            ),
            arm_word_gate=interruption_word_gate.arm if counts_hold_words else None,
            no_word_budget_seconds=no_word_hold_budget if counts_hold_words else None,
            # The states the transcript gate holds in: a hand-over is only
            # safe when the state it restores is one of them.
            reply_in_flight_states=REPLY_IN_FLIGHT_AGENT_STATES,
            # Named when the hold BEGINS, not when the segment ends: a reply
            # cancelled while held clears its buffer after a preempting reply
            # may already be current.
            current_speech_id=lambda: str(
                getattr(getattr(session, "current_speech", None), "id", "") or ""
            ),
            on_hold_ended=lambda held_ms, words, reached_wire, speech_id: (
                interruption_telemetry.note_pre_playout_hold(
                    held_ms=held_ms,
                    words_during_hold=words,
                    speech_id=speech_id,
                ),
                # A reply the model produced and the floor hold kept off the
                # line: the model answered, so the silent-model clock clears.
                (
                    model_unresponsive.note_reply_withheld(
                        at=time.monotonic(),
                        held_since=time.monotonic() - held_ms / 1000.0,
                    )
                    if not reached_wire
                    else None
                ),
            ),
        )
        # The stop the tail above decides still has to be EXECUTED, and the
        # runtime executes it by discarding the ~200 ms already committed to
        # the audio source -- the tail of the word the caller is hearing.
        # Installed after the guard, on the same started session, because the
        # object that does the discarding is the audio source at the bottom of
        # that same chain. Declines silently on any runtime shape it does not
        # recognise, leaving today's behaviour in place.
        graceful_stop = (
            install_graceful_stop(
                session, stats=graceful_stop_stats, feed=alignment_feed
            )
            if managed_pstn_interruption
            else None
        )
        if graceful_stop is None and managed_pstn_interruption:
            emit("graceful_stop_unavailable", call_id=call.call_id)
        # The opening turn -- and ONLY the opening turn -- waits for the
        # callee to pick up, on the watch opened at room join above. Dispatch,
        # session start and the whole provider warm-up stay where they were,
        # so the pipeline is hot when the answer lands; what changes is that
        # the first word is no longer rendered into a leg with no RTP
        # destination. Bounded by the dialer's own ringing budget, and
        # fail-open on every disposition except a callee who has already gone.
        answer_disposition = SIP_ANSWER_NOT_APPLICABLE
        still_present = True
        # A fatal session close can set ``finished`` while this worker is
        # still parked here, and the dialer's own ringing budget runs up to
        # 120s -- long enough that a call already decided over stayed open,
        # billing, until that budget ran out. Race the two: whichever
        # resolves first is the one this worker acts on.
        ended_before_answer = False
        if answer_watch is not None:
            finished_wait_task = asyncio.create_task(finished.wait())
            try:
                done, _pending = await asyncio.wait(
                    {answer_watch, finished_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if answer_watch in done:
                    answer_disposition = answer_watch.result()
                    # Asked again HERE, not inferred from the disposition: a
                    # callee who answered during warm-up and hung up before
                    # this line really did answer -- and is really not on
                    # the line to be greeted.
                    still_present = callee_attributes() is not None
                else:
                    # The session ended before an answer was ever known.
                    # Stop watching for one and enter the existing teardown
                    # at once, rather than waiting out a budget the ending
                    # already made moot.
                    ended_before_answer = True
                    answer_watch.cancel()
                    try:
                        await answer_watch
                    except asyncio.CancelledError:
                        pass
            finally:
                if not finished_wait_task.done():
                    finished_wait_task.cancel()
                try:
                    await finished_wait_task
                except asyncio.CancelledError:
                    pass
        if ended_before_answer:
            pass
        elif may_speak(answer_disposition, still_present=still_present):
            # The same disposition that allows the opening also releases the
            # caller's audio into the engine -- one predicate for both, so
            # audio can never reach a duplex model, and the model can never
            # be asked to generate, on opposite sides of the answer gate.
            # This is the fail-open gate's own disposition: a callee who
            # answered gets ANSWERED/ALREADY_ACTIVE, and a call this worker
            # cannot get answer evidence for still opens exactly as it did
            # before this gate existed (NOT_REPORTED/TIMED_OUT). Nothing
            # about when that gate opens changes here.
            release_moment = time.monotonic()
            # The no-caller-speech timer's real zero: SIP answer evidence and
            # this worker actually re-enabling the engine's input are two
            # different instants, and a customer who spoke into that gap
            # must not be judged deaf by a clock that started before this
            # worker could possibly have heard them.
            call_end_clock["input_released_at"] = release_moment
            if duplex_opening_hold is not None and not (callee_first and s2s_live_call):
                # BEFORE the input opens: the first caller frame the model
                # could hear is already silence. Engaged here, not at
                # install, so the pre-answer window -- where input is disabled
                # anyway -- is not counted as held.
                if duplex_opening_hold.engage():
                    duplex_opening_hold_task = asyncio.create_task(
                        duplex_opening_hold.run()
                    )
            if live_pickup is not None and live_pickup.buffer.error is not None:
                raise ProductionConfigurationError("live pickup capture failed during model startup")
            session.input.set_audio_enabled(True)
            # The caller can be heard from here, so silence can be measured.
            mutual_silence.arm(at=release_moment)
            # From the same zero as the wait above, so a batch can read them
            # side by side: this call had no way to say whether ringback
            # audio at 5.5s was spoken because the gate fell open early
            # (small wait_ms) or because the engine spoke despite a still-
            # pending gate (opening_generate_ms less than wait_ms -- which this
            # fix makes structurally impossible, so seeing it again would be
            # a regression worth paging on).
            answer_gate_evidence["input_released_ms"] = int(
                round((release_moment - joined_at_monotonic) * 1000)
            )
            answer_gate_evidence["opening_generate_ms"] = int(
                round((time.monotonic() - joined_at_monotonic) * 1000)
            )
            # Voicemail evidence only: when the engine could first hear the
            # caller (words before it are not bound to any greeting) and when
            # the opening was initiated (the MR2 opener-hold calibration).
            answer_detection.note_input_released()
            # The configured opening line, noted now: its conversation item
            # only arrives once it has played, after its echo would have. A
            # workflow call speaks its authored first step instead and never
            # this line, so noting it there would let the voicemail watch
            # mistake words nobody said for our own echo.
            answer_detection.note_agent_text(profile.first_message)
            if far_end_watch is not None:
                far_end_watch.note_agent_text(profile.first_message)
            if far_end_perception is not None:
                far_end_perception.note_agent_text(profile.first_message)
            opening_decider = opening["decider"]
            if opening_decider is None:
                # The opening starts now. A callee-first opening starts when
                # it is dispatched, which is noted there.
                answer_detection.note_opener_started()
            def hold_opening_for_menu(*, only_a_person_ends_it: bool = False, a_second_label_ends_it: bool = False) -> Any:
                """The callee-first opening waits for the first words of the line to be labelled (a menu is not opened over)."""

                return line_is_a_menu(
                    navigation, far_end_watch, is_finished=finished.is_set,
                    only_a_person_ends_it=only_a_person_ends_it,
                    a_second_label_ends_it=a_second_label_ends_it,
                    only_answered_the_phone=lambda: bool(opening["pickup_finals"]) and is_greeting_only(
                        opening["pickup_finals"],
                        pickup_answer_words(profile.language),
                        filler=caller_backchannel_tokens(call.config),
                    ),
                )

            def hold_opening_for_music() -> Any:
                """A callee-first opening that finds the line playing music, not a callee, waits for the hold it turns into."""

                return line_is_holding_music(
                    navigation,
                    lambda: bool(answer_detection.music_reading.onset or answer_detection.music_reading.active),
                    is_finished=finished.is_set,
                )

            if opening_decider is not None and s2s_live_call:
                # The model hears continuously. SDK speech/tool authorization
                # waits for the platform opening decision, and a resumption
                # invalidates any reply to only part of the pickup. Native
                # caption commits do not shorten this resumption window.
                # Only observed silence requests an explicit first response.
                opening_decider.note_input_released(at=release_moment)
                live_opening_dispatchable = await wait_for_opening_release(
                    opening_decider, released=opening_released, resumed=opening_resumed,
                    pickup_committed=asyncio.Event(), finished=finished,
                    # Retain the 700ms resumption window: a shorter margin
                    # let a two-part greeting escape before VAD qualified
                    # the resumed speech in actual SIP qualification.
                    hold_pickup=lambda: None,
                    # (a live engine's native opening cannot be cut once it is under way: only a person's label ends the wait)
                    hold_for_menu=(
                        functools.partial(hold_opening_for_menu, only_a_person_ends_it=True)
                        if navigation.enabled and far_end_watch is not None else None
                    ),
                    hold_for_music=hold_opening_for_music if navigation.enabled else None,
                )
                if live_pickup.buffer.error is not None:
                    raise ProductionConfigurationError("live pickup capture failed before opening")
                if not live_opening_dispatchable and not finished.is_set() and navigation.enabled and navigation.in_episode:
                    # The line is a phone menu, a queue or a recording: the navigation owns the call and the opening
                    # is never spoken. The provider's held replies are discarded and its authorization resumed (the
                    # model must still be able to answer the menu), the pickup is finished, and the call goes on.
                    hand_native_opening_to_navigation(
                        authorization=native_opening_authorization, pickup_buffer=live_pickup.buffer,
                        pickup_generation=gemini_pickup_generation,
                        emit=lambda event: _quietly(emit, event, call_id=call.call_id),
                        opening_site=greeting_site,
                        refeed=(
                            (lambda: navigation_actions.executor.request_refeed(settle_navigation_labels))
                            if navigation_actions is not None else None
                        ),
                    )
                elif not live_opening_dispatchable and not finished.is_set():
                    native_opening_authorization.close()
                    await end_for_opening_failure("native_not_dispatchable")
                if live_opening_dispatchable:
                    opening_decider.note_opening_dispatched(at=time.monotonic())
                    answer_detection.note_opener_started()
                    live_pickup.buffer.pickup_completed = True
                    opening["first_reply_path"] = "native_pickup" if opening_decider.callee_spoke else FIRST_REPLY_GENERATED
                    opening_failure = None
                    def generate_pickup_replacement():
                        opening["first_reply_path"] = "native_replacement"
                        return session.generate_reply(
                            instructions=("Respond to the person's complete pickup now. Address their "
                                          "request or question first. If they only said hello, "
                                          + greet_directive(profile.first_message,
                                                            opening_mode=profile.opening_mode,
                                                            language_name=language_name(profile.language))),
                            tool_choice="none", allow_interruptions=True)
                    async def prepare_pickup_replacement(superseded):
                        # A caption-only Gemini generation may still be thinking.
                        # Give its first output ownership before asking again.
                        if gemini_pickup_generation is not None:
                            await gemini_pickup_generation.wait_for_response(
                                ended=finished, superseded=superseded)
                        from live_opening import settle_cancelled_duplex_output
                        await settle_cancelled_duplex_output(
                            session_agent.realtime_llm_session, ended=finished, superseded=superseded)
                    try:
                        if opening_decider.callee_spoke:
                            opening_failure = await finish_native_pickup_reply(
                                native_opening_authorization, generate=generate_pickup_replacement,
                                wait=await_playout_or_session_end, ended=finished,
                                audible_seconds=lambda: native_opening_output.audible_seconds,
                                caller_revision=lambda: native_caller_revision,
                                prepare_replacement=prepare_pickup_replacement,
                                # A reply that only called a navigation tool refused outside a menu or a hold put
                                # nothing on the line: still an empty opening once the engine's own reply to the
                                # refusal is silent too, unless the navigation owns the line.
                                answered_nothing=(navigation_actions.answered_nothing
                                                  if navigation_actions is not None else None),
                                line_unsettled=(navigation_actions.line_unsettled
                                                if navigation_actions is not None else (lambda: False)))
                        else:
                            # Silence/noise and pickup recovery use the same
                            # explicit response ownership until playout ends,
                            # and the same settling of discarded output first.
                            opening_failure = await finish_native_silence_opening(
                                native_opening_authorization,
                                generate=lambda: generate_opening_reply(
                                    session, language=profile.language, protection=None,
                                    instructions=greet_directive(
                                        profile.first_message, opening_mode=profile.opening_mode,
                                        language_name=language_name(profile.language)), ended=finished,
                                    recovery=opening_reply_recovery),
                                ended=finished,
                                prepare_replacement=prepare_pickup_replacement)
                    except Exception as exc:
                        opening_failure = exc
                    if gemini_pickup_generation is not None:
                        # A newer caller turn can supersede pickup recovery.
                        # Hand its pending generation back to ordinary handling.
                        gemini_pickup_generation.close(
                            publish_pending=opening_failure is None and not finished.is_set())
                    if opening_failure is not None:
                        _quietly(emit, "opening_generation_failed", call_id=call.call_id,
                                 error_type=type(opening_failure).__name__)
                        await end_for_opening_failure("native_opening_reply")
            elif opening_decider is not None:
                # Outbound, managed-PSTN cascade: the callee speaks first. The
                # caller can be heard from here (the input is open), and the
                # decider holds the opening until the person has said hello
                # and finished, or the line has been silent for the window.
                opening_decider.note_input_released(at=release_moment)
                opening_dispatchable = await wait_for_opening_release(
                    opening_decider,
                    released=opening_released,
                    resumed=opening_resumed,
                    pickup_committed=opening["pickup_committed"],
                    finished=finished,
                    hold_pickup=functools.partial(opening.__setitem__, "pickup_pending", True),
                    commit_pickup=commit_opening_pickup,
                    # (a recording's first label ("Thank you for calling ...") does not release the opening into the menu that
                    # follows it: only a person's label, or a menu, ends the wait early; the bound is the backstop)
                    hold_for_menu=(
                        functools.partial(hold_opening_for_menu, only_a_person_ends_it=True, a_second_label_ends_it=True)
                        if navigation.enabled and far_end_watch is not None else None
                    ),
                    hold_for_music=hold_opening_for_music if navigation.enabled else None,
                )
                answer_gate_evidence["opening_generate_ms"] = int(
                    round((time.monotonic() - joined_at_monotonic) * 1000)
                )
                pickup_heard = ""
                if opening_decider.callee_spoke:
                    opening["pickup_in_context"], pickup_heard = pickup_source(
                        committed=opening["pickup_committed"].is_set(),
                        finals=opening["pickup_finals"],
                    )
                    if opening["pickup_in_context"] == "none":
                        # The opening goes out without the pickup: when its
                        # turn does commit, it is answered like any turn,
                        # not kept silently.
                        opening["pickup_pending"] = False
                if opening_dispatchable:
                    if transcript_gate is not None:
                        # Until the first frame: a literal say never reports
                        # ``thinking``, so without this a pickup final landing
                        # during synthesis would commit a turn, and the runtime
                        # cancels the current speech for every committed turn
                        # -- the opening, before a word of it was heard.
                        transcript_gate.set_opening_turn_in_flight(True)
                    # Freeze the VAD decision before awaiting the SDK commit.
                    # A separate ownership flag prevents speculative workflow
                    # execution while the pending pickup hook is being sealed.
                    opening_decider.note_opening_dispatched(at=time.monotonic())
                    if opening_decider.callee_spoke:
                        opening["pickup_sealing"] = True
                        try:
                            await seal_opening_pickup()
                        finally:
                            opening["pickup_sealing"] = False
                    answer_detection.note_opener_started()
                    # Nothing the caller says can change the decision now: the
                    # decider's VAD stream is closed rather than left waiting
                    # for frames the feed no longer sends.
                    for opening_task in opening["tasks"]:
                        if not opening_task.done():
                            opening_task.cancel()
                    greeting_only = (
                        opening_decider.callee_spoke
                        and str(profile.opening_mode or "").strip().casefold() == "literal"
                        and bool(str(profile.first_message or "").strip())
                        and is_greeting_only(
                            opening["pickup_finals"],
                            pickup_answer_words(profile.language),
                            filler=caller_backchannel_tokens(call.config),
                        )
                    )
                    if greeting_only:
                        # The callee only answered the phone: the reply is the
                        # configured opening as written, which is what the
                        # generated reply would say (``pickup_reply_instruction``)
                        # -- without the model's generation before the first
                        # word. The pickup is in context either way.
                        opening["first_reply_path"] = FIRST_REPLY_LITERAL_FAST
                    elif opening_decider.callee_spoke and (opening["pickup_committed"].is_set() or pickup_heard):
                        opening["first_reply_path"] = FIRST_REPLY_PICKUP
                    elif profile.opening_mode == "generated":
                        opening["first_reply_path"] = FIRST_REPLY_GENERATED
                    else:
                        opening["first_reply_path"] = FIRST_REPLY_LITERAL
                    if opening["first_reply_path"] == FIRST_REPLY_PICKUP:
                        # The person said something and it is in context: the
                        # first reply answers it. A hello gets the opening; a
                        # pickup with content gets an answer first.
                        opening_failure = await generate_opening_reply(
                            session,
                            language=profile.language,
                            # No opening protection on outbound (lead,
                            # 2026-09-27): real speech interrupts the first
                            # reply like any other. Measured with it on, a
                            # 2.3 s question over the opening was not yielded
                            # to before the opening ended. The pickup's own
                            # late turn is kept from cancelling it by the
                            # reply-less commit above instead.
                            protection=None,
                            instructions=pickup_reply_instruction(
                                profile.first_message,
                                opening_mode=profile.opening_mode,
                                language_name=language_name(profile.language),
                                heard=pickup_heard,
                            ),
                            ended=finished,
                            recovery=request_scoped_opening_recovery(),
                            # It answers something the callee said, so it may take a speech window.
                            greeting=False,
                        )
                    elif opening["first_reply_path"] == FIRST_REPLY_GENERATED:
                        opening_failure = await generate_opening_reply(
                            session,
                            language=profile.language,
                            protection=None,
                            ended=finished,
                            recovery=request_scoped_opening_recovery(),
                        )
                    else:
                        opening_failure = None
                        await speak_opening_line(
                            session=session,
                            message=profile.first_message,
                            language=profile.language,
                            speak=functools.partial(
                                say_opening_utterance,
                                protection=None,
                                ended=finished,
                            ),
                            ended=finished,
                            # Whether the opening reached the wire, from the
                            # decider's own record: the opening protection is
                            # not engaged here, so its counter never moves.
                            audio_frames_seen=lambda: (
                                0
                                if opening_decider.telemetry()["opening_first_audio_ms"] is None
                                else 1
                            ),
                            emit=lambda event, **details: emit(
                                event, call_id=call.call_id, **details
                            ),
                            note_fallback=interruption_telemetry.note_greeting_fallback,
                            still_present=lambda: callee_attributes() is not None,
                        )
                    if opening_failure is not None:
                        emit(
                            "generated_opening_failed",
                            call_id=call.call_id,
                            language=profile.language,
                            error_type=type(opening_failure).__name__,
                        )
                        await end_call_for_unheard_opening(opening_failure, "pickup_opening_reply")
            elif s2s_live_call:
                # One owner of the opening: the session's awaited reply, with
                # its handle, the runtime's reply timeout and the opening
                # protection. The greeting directive is its instruction. It
                # used to be appended as commentary instead -- one queued
                # event nothing awaited -- and a call whose model never
                # answered it ran silent until the no-caller-speech timer.
                opening_failure = await generate_opening_reply(
                    session,
                    language=profile.language,
                    protection=opening_protection,
                    transcript_gate=transcript_gate,
                    recovery=opening_reply_recovery,
                    instructions=greet_directive(
                        profile.first_message,
                        opening_mode=profile.opening_mode,
                        language_name=language_name(profile.language),
                    ),
                    ended=finished,
                )
                browser_opening_failed(opening_failure)
                if opening_failure is not None:
                    emit(
                        "generated_opening_failed",
                        call_id=call.call_id,
                        language=profile.language,
                        error_type=type(opening_failure).__name__,
                    )
            elif profile.opening_mode == "generated":
                # The answer gate remains outside the model call: generation
                # cannot spend tokens or speak into a ringing/gone SIP leg.
                # An empty opening is asked once more when that is provably
                # safe; one that still never reached the caller ends the call
                # truthfully. It never substitutes an unrelated literal.
                opening_failure = await generate_opening_reply(
                    session,
                    language=profile.language,
                    protection=opening_protection,
                    transcript_gate=transcript_gate,
                    ended=finished,
                    recovery=request_scoped_opening_recovery(),
                )
                browser_opening_failed(opening_failure)
                if opening_failure is not None:
                    emit(
                        "generated_opening_failed",
                        call_id=call.call_id,
                        language=profile.language,
                        error_type=type(opening_failure).__name__,
                    )
                    await end_call_for_unheard_opening(opening_failure, "generated_opening_reply")
            else:
                # A greeting whose synthesis fails must not leave an
                # answered call in silence. Only the free-form path is wired
                # here; the workflow path speaks an authored step and owns its
                # own farewell, a published line of its contract.
                #
                # ``still_present`` is passed only when there is an answer
                # watch to read it from: on an inbound call there is no callee
                # participant to look up and no several-second warm-up gap to
                # protect against, so the fallback must not be gated on it.
                await speak_opening_line(
                    session=session,
                    message=profile.first_message,
                    language=profile.language,
                    # The protected seam does the speaking, so the opening
                    # keeps its uninterruptible window; this only adds the
                    # question that window cannot answer -- did anything
                    # actually play.
                    speak=functools.partial(
                        say_opening_utterance,
                        protection=opening_protection,
                        transcript_gate=transcript_gate,
                        ended=finished,
                    ),
                    ended=finished,
                    # The protection already counts openings that reached the
                    # wire ("protection that never produced audio is not a
                    # greeting that played"). Reusing it is what keeps a
                    # greeting that broke MID-sentence from getting a second,
                    # unrelated line spoken over it.
                    audio_frames_seen=lambda: int(
                        opening_protection.stats["opening_turns_reached_audio"]
                    ),
                    emit=lambda event, **details: emit(
                        event, call_id=call.call_id, **details
                    ),
                    note_fallback=interruption_telemetry.note_greeting_fallback,
                    still_present=(
                        (lambda: callee_attributes() is not None)
                        if answer_watch is not None
                        else None
                    ),
                )
            if acknowledgements["prepare"] is not None and not s2s_live_call:
                # The callee is on the line and has heard the opening: only
                # now are the metered acknowledgement clips worth making.
                acknowledgements["task"] = asyncio.create_task(acknowledgements["prepare"]())
        elif answer_watch is not None:
            # The greeting was suppressed because the callee is confirmed
            # gone -- DISCONNECTED, or answered-then-left during warm-up.
            # `caller_disconnect_handler` above only fires on a FUTURE
            # `participant_disconnected` event; the watch that produced this
            # disposition started at room join, seconds earlier, so a callee
            # who left before that handler was registered is a disconnect
            # nothing else will ever observe. Finish here, or the plain
            # `await finished.wait()` below waits for an event that will
            # never come and wedges the job, its provider connections, and
            # its heartbeat.
            finish_session()
        if transcript_gate is not None:
            # Every path out of the block above -- free-form, GPT-Live,
            # generated, the workflow branch, or the callee-already-gone
            # ``elif`` that never speaks at all -- closes the window armed at
            # construction. The say/generate-reply call sites and
            # ``speak_opening_turn``'s own finally already close it on their
            # paths; this is the backstop for the one that doesn't reach any
            # of them: ``may_speak`` false with no answer watch, or a
            # non-terminal workflow reply with no text. Idempotent either way.
            transcript_gate.set_opening_turn_in_flight(False)
        emit(
            "media_admitted",
            call_id=call.call_id,
            language=profile.language,
            semantic_config_source="realtime-config",
            # Whether the facts reached a duplex session. Recorded, never
            # routed on: the opening takes the same seam either way.
            **({"gpt_live_context_injected": gpt_live_injected} if s2s_live_call else {}),
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
        native_floor_cancellation = None
        if confirmation_listener is not None:
            try:
                await confirmation_listener.aclose()
            except asyncio.CancelledError as exc:
                native_floor_cancellation = exc
            except Exception as exc:
                emit("confirmation_listener_close_failed", call_id=call.call_id, error_type=type(exc).__name__)
        if opening_reply_recovery is not None:
            opening_reply_recovery.close()
        try:
            await await_bounded_teardown_task(
                asyncio.create_task(native_farewell.close()), timeout_seconds=1.0)
        except asyncio.CancelledError as exc:
            native_floor_cancellation = exc
        if native_caller_floor is not None:
            try:
                await await_bounded_teardown_task(
                    asyncio.create_task(native_caller_floor.close()), timeout_seconds=1.0)
            except asyncio.CancelledError as exc:
                native_floor_cancellation = exc
        if native_opening_authorization is not None:
            native_opening_authorization.close()
        if gemini_pickup_generation is not None:
            gemini_pickup_generation.close()
        if live_pickup is not None:
            _quietly(emit, "live_pickup_audio_summary", call_id=call.call_id,
                 startup_frames_retained=live_pickup.buffer.startup_retained,
                 frames_delivered=live_pickup.buffer.delivered,
                 frames_dropped_after_pickup=live_pickup.buffer.dropped_after_pickup,
                 pickup_failed=live_pickup.buffer.error is not None)
            try:
                await await_bounded_teardown_task(
                    asyncio.create_task(live_pickup.aclose()), timeout_seconds=1.0)
            except asyncio.CancelledError as exc:
                native_floor_cancellation = exc
        funding_watch.cancel()
        heartbeat_task.cancel()
        # Cancelled with the rest of this call's periodic work, and joined
        # below with it. A measurement task that outlived its call would go on
        # re-reading a window nobody is in.
        classify_task.cancel()
        if voicemail_task is not None:
            voicemail_task.cancel()
        if duplex_opening_hold is not None:
            # A hold still on at teardown ended with the call, and says so.
            duplex_opening_hold.end_of_call()
        if duplex_opening_hold_task is not None:
            duplex_opening_hold_task.cancel()
        call_end_watch.cancel()
        # Same shape as the heartbeat: a leg that never answered leaves this
        # pending, and teardown must not wait out the ringing budget. A report
        # already in flight at this instant is lost; the window is one
        # control-plane roundtrip wide, and every second of the warm-up that
        # preceded it is covered.
        if answer_watch is not None and not answer_watch.done():
            answer_watch.cancel()
        # Set if the browser-test report wait below is cancelled. Held so the
        # teardown can finish and the cancellation can still surface afterwards.
        browser_test_report_cancellation: BaseException | None = None
        # The optional background player's close has the same requirement: an
        # external job cancellation must not skip the session close and the
        # terminal settlement that releases the caller's reservation.  Its
        # helper joins the player's own close task before surfacing this.
        audio_environment_close_cancellation: BaseException | None = None
        acknowledgements_close_cancellation: BaseException | None = None
        # The farewell has the same settlement rule.  A cancellation while its
        # worker-owned task is still playing must not let immutable evidence
        # finalize ahead of the task's terminal record.
        terminal_end_call_cancellation: BaseException | None = None
        end_call_guard_settle_cancellation: asyncio.CancelledError | None = None
        # Persist of a refused metering snapshot is diagnostic only. A
        # cancellation of this task during that POST must not skip sip.ended
        # or the reservation stays in-progress with no terminal settlement.
        refused_metering_cancellation: BaseException | None = None
        # The evidence bundle is written off the loop and always waited for;
        # a cancellation meanwhile is held until the teardown has finished.
        evidence_finalize_cancellation: BaseException | None = None
        artifacts_finalize_cancellation: BaseException | None = None
        carrier_hung_up = False
        browser_outcome_attempted = False
        async def _teardown_room(*, session_cleanup: bool) -> None:
            nonlocal browser_outcome_attempted
            if browser_startup is not None and not browser_outcome_attempted:
                browser_outcome_attempted = True
                await publish_before_room_delete(
                    witness=browser_startup, call=call,
                    persist=functools.partial(write_session_state, state_path),
                    emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
                    timeout=BROWSER_TEST_REPORT_BUDGET_SECONDS,
                )
            # The delete goes to the LiveKit SERVER, not through
            # ``ctx.delete_room``: that wrapper swallows every failure and
            # returns as though the room were gone, so an operator's
            # acknowledgement could only ever say ``room_deleted``. The
            # acknowledgement travels with the teardown for the same reason --
            # the outcomes worth reporting are the ones that raise.
            await teardown_room_and_acknowledge_hangup(
                call=call,
                room_service=ctx.api.room,
                room_name=ctx.room.name,
                request_factory=api.DeleteRoomRequest,
                intent=hangup_intent,
                emit=lambda event, **details: emit(
                    event, call_id=call.call_id, **details
                ),
                session_cleanup=session_cleanup,
            )

        async def hangup_carrier() -> None:
            # DeleteRoom + hangup ack only. ``call.cleanup()`` marks the
            # session cleaned, after which usage, sip.ended and artifact
            # import raise SessionStateError. Settlement still needs those.
            nonlocal carrier_hung_up
            if carrier_hung_up:
                return
            await _teardown_room(session_cleanup=False)
            carrier_hung_up = True

        async def cleanup_room() -> None:
            # After settlement, or on an exception before it: mark the session
            # cleaned. If the carrier is already gone, skip a second DeleteRoom.
            nonlocal carrier_hung_up
            if carrier_hung_up:
                call.cleanup(remove_participant=lambda: None, delete_room=lambda: None)
                return
            await _teardown_room(session_cleanup=True)
            carrier_hung_up = True

        try:
            await asyncio.gather(
                funding_watch,
                heartbeat_task,
                classify_task,
                *([] if voicemail_task is None else [voicemail_task]),
                call_end_watch,
                *([] if answer_watch is None else [answer_watch]),
                return_exceptions=True,
            )
            # A transfer still short of its bridge stops now, while the session
            # and the room it speaks and dials through still exist: its line is
            # cut, its leg removed and ``caller_hangup`` reported, before the
            # carrier drop and the SDK close below (bounded; a cancellation
            # here is held like the others).
            try:
                await transfer_controller.cancel_and_join("caller_hangup")
            except asyncio.CancelledError as exc:
                native_floor_cancellation = native_floor_cancellation or exc
            # A caller disconnect can wake ``finished`` while the worker-owned
            # end_call goodbye is still waiting on its playout handle.  Join
            # that bounded task before closing the SDK session or freezing the
            # evidence bundle, otherwise its terminal message can arrive after
            # ``recorder.finalize`` and be silently absent from the artifact.
            # The helper is a no-op when no end_call task was scheduled.
            (
                _,
                terminal_end_call_cancellation,
            ) = await settle_worker_terminal_end_call(
                end_call,
                emit=lambda event, **details: emit(
                    event, call_id=call.call_id, **details
                ),
            )
            if end_call_guard is not None:
                # Settle post-refusal work still in flight (bounded) before the
                # summary is frozen: a regeneration failing after the caller
                # hung up must land in the counters the verifier reads, and a
                # handle still pending at the bound is reported as unsettled
                # rather than certified. An external cancellation during that
                # wait is held, like the goodbye join above, and re-raised only
                # after settlement: letting it through here would skip the
                # summary, the SDK close, usage and lifecycle reporting.
                (
                    guard_summary,
                    end_call_guard_settle_cancellation,
                ) = await settle_end_call_guard_summary(
                    end_call_guard,
                    post_refusal_tasks,
                    emit=lambda event, **details: emit(
                        event, call_id=call.call_id, **details
                    ),
                )
                # On EVERY armed call, attempts or not: a run in which the model
                # never reached for end_call is a pass the verifier can tell
                # from a run whose guard evidence went missing.
                emit(GUARD_SUMMARY_EVENT, call_id=call.call_id, **guard_summary)
            if far_end_perception is not None:
                try:
                    await far_end_perception.aclose()
                except Exception:  # noqa: BLE001 - teardown continues regardless
                    pass
            if navigation.enabled and far_end_perception is not None:
                # Information only: where the answered call's seconds went and what the internal
                # side calls cost. Never part of the usage report.
                try:
                    emit(
                        EVENT_INTERVALS,
                        call_id=call.call_id,
                        **navigation_intervals(
                            navigation.snapshot(),
                            answered_for=(
                                None
                                if call_end_clock["answered_at"] is None
                                else time.monotonic() - call_end_clock["answered_at"]
                            ),
                            detector_audio_seconds=far_end_perception.detector.position,
                            classifier_calls=(far_end_watch.classified if far_end_watch is not None else 0),
                            decider_calls=(0),
                        ),
                    )
                except Exception:  # noqa: BLE001 - an information-only record never moves teardown
                    pass
            if navigation_actions is not None:
                try:
                    await navigation_actions.aclose()
                except Exception:  # noqa: BLE001 - teardown continues regardless
                    pass
                if navigation_player is not None:
                    try:
                        await navigation_player.aclose()
                    except Exception:  # noqa: BLE001 - optional audio only
                        pass
            if far_end_watch is not None:
                try:
                    await far_end_watch.aclose()
                except Exception:  # noqa: BLE001 - teardown continues regardless
                    pass
            if end_call_classifier is not None:
                # Job teardown closes the classifier's shared client; each
                # request already closed its own stream.
                try:
                    await end_call_classifier.aclose()
                except Exception:  # noqa: BLE001 - teardown
                    pass
            if language_switcher is not None:
                # A queued internal tool may be waiting behind the final model
                # turn. Bound/cancel it before closing the SDK session so one
                # stuck provider activity cannot prevent terminal settlement.
                await language_switcher.drain_for_terminal()
            if transcript_publisher is not None:
                # A tool result landing after teardown has no panel left.
                transcript_publisher.close()
            if transcript_event_recorder is not None:
                # A reply the caller hung up on finishes only when the session
                # closes, after this point; its claim on the caller's
                # uncaptioned speech is settled first, or that moment is lost.
                turn_events.settle_engine_speeches()
                # Same rule for the stored rows: the recording this section
                # describes has ended, so a late moment belongs to nothing.
                # ``section()`` still answers -- close stops recording, it
                # does not discard what was recorded.
                transcript_event_recorder.close()
            # Acknowledgements first: cancel a pending filler and its clip
            # synthesis, then release the player this call started for them.
            # Failures here are diagnostic only, like the ambience close below.
            feedback = acknowledgements["tool_wait"]
            if feedback is not None:
                try:
                    await feedback.aclose()
                except asyncio.CancelledError as exc:
                    # Same rule as the scheduler close below: only an external
                    # cancellation of this task comes out, and it is held.
                    acknowledgements_close_cancellation = exc
                except Exception as exc:  # noqa: BLE001 - optional audio only
                    emit(
                        "tool_wait_close_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                    )
            scheduler = acknowledgements["scheduler"]
            if scheduler is not None:
                try:
                    await scheduler.aclose()
                except asyncio.CancelledError as exc:
                    # ``aclose`` re-raises only an external cancellation of
                    # this task; hold it until settlement like the joins below.
                    acknowledgements_close_cancellation = exc
                except Exception as exc:  # noqa: BLE001 - optional audio only
                    emit(
                        "acknowledgements_close_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                    )
            clips_task = acknowledgements["task"]
            if clips_task is not None and not clips_task.done():
                # Cancel AND join, bounded: the task still has to close its
                # synthesis stream and flush native metrics, and those must
                # land before the usage snapshot below reads the counters.
                clips_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(clips_task), timeout=2.0)
                except asyncio.CancelledError as exc:
                    # The shield re-raises the INNER task's own cancellation
                    # (the one requested just above) as CancelledError too.
                    # Only a pending request on this waiter is an external
                    # cancellation, and that one is held until settlement.
                    waiter = asyncio.current_task()
                    if waiter is not None and waiter.cancelling() > 0:
                        acknowledgements_close_cancellation = exc
                        emit("acknowledgements_close_cancelled", call_id=call.call_id)
                except (TimeoutError, Exception):  # noqa: BLE001 - optional audio only
                    pass
            if acknowledgements["player"] is not None:
                # Same settlement rule as the ambience close below: an
                # external cancellation here is held and re-raised only after
                # the session close, evidence, usage and lifecycle settle.
                try:
                    await close_outgoing_audio_environment(acknowledgements["player"])
                except asyncio.CancelledError as exc:
                    acknowledgements_close_cancellation = exc
                    emit("acknowledgements_close_cancelled", call_id=call.call_id)
                except Exception as exc:  # noqa: BLE001 - optional audio only
                    emit(
                        "acknowledgements_close_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                    )
            # Close the optional output independently before the SDK session.
            # A failed ambience close is diagnostic, never a reason to skip
            # session close, evidence finalization, usage reporting or room
            # cleanup (all of which release the caller's reservation).
            try:
                await close_outgoing_audio_environment(audio_environment_player)
            except asyncio.CancelledError as exc:
                audio_environment_close_cancellation = exc
                emit(
                    "audio_environment_close_cancelled",
                    call_id=call.call_id,
                )
            except Exception as exc:  # noqa: BLE001 - optional media only
                emit(
                    "audio_environment_close_failed",
                    call_id=call.call_id,
                    error_type=type(exc).__name__,
                )
            async def close_sdk_session() -> None:
                nonlocal native_floor_cancellation
                try:
                    session_close = await close_production_session_after_room_end(
                        session,
                        room_transport_disconnected=lambda: room_disconnect.triggered,
                        on_late_completion=lambda error_type: emit(
                            "session_close_late_result",
                            call_id=call.call_id,
                            completed=not bool(error_type),
                            error_type=error_type,
                        ),
                    )
                except asyncio.CancelledError as exc:
                    # The job's own cancellation while the SDK close was pending:
                    # held like every close below, so the recognizer close,
                    # usage and lifecycle settlement still run before it surfaces.
                    native_floor_cancellation = native_floor_cancellation or exc
                    session_close = ProductionSessionCloseResult(completed=False, error_type="CancelledError")
                if not session_close.required:
                    emit(
                        "session_close_skipped",
                        call_id=call.call_id,
                        reason=session_close.skipped_reason,
                    )
                elif not session_close.completed:
                    emit(
                        "session_close_incomplete",
                        call_id=call.call_id,
                        error_type=session_close.error_type,
                        cancellation_resistant=session_close.cancellation_resistant,
                    )
                if inbound_audio_output is not None:
                    # RoomIO did not create this pre-attached output, so the
                    # worker owns its close after session playout has stopped.
                    try:
                        await await_bounded_teardown_task(
                            asyncio.create_task(inbound_audio_output.aclose()), timeout_seconds=1.0)
                    except asyncio.CancelledError as cleanup_exc:
                        native_floor_cancellation = native_floor_cancellation or cleanup_exc
                # The worker, not ``aclose`` above, closes the Soniox
                # recognizer, whether that close completed, timed out or was
                # skipped: a session's usage row exists only once its socket
                # has closed, and a stream left open is reconnected onto the
                # job's closed HTTP session at exit (``soniox_stt``).
                # Both collection sites run after this.
                stt_close = await close_soniox_stt_streams(getattr(stack, "stt", None))
                if stt_close is not None:
                    if stt_close.cancellation is not None:
                        native_floor_cancellation = native_floor_cancellation or stt_close.cancellation
                    if stt_close.status != "closed":
                        emit(
                            "soniox_stream_close_incomplete",
                            call_id=call.call_id,
                            outcome=stt_close.status,
                            streams=stt_close.streams,
                            sessions=stt_close.sessions,
                        )
                    if stack.soniox_usage_tracker is not None:
                        stack.soniox_usage_tracker.record_stt_close(
                            status=stt_close.status, sessions=stt_close.sessions
                        )

            # A browser-test report is room metadata produced by ``aclose``.
            # That path must close the SDK session while the room still exists.
            # A PSTN call has no such report; the 2s aclose bound times out on
            # every sampled worker-ended call and is dead time on a silent
            # line, so it runs after hangup instead.
            if browser_test_report_task is not None:
                await close_sdk_session()
            if language_switcher is not None:
                await language_switcher.aclose_retired_components()
                if browser_test_report_task is not None:
                    try:
                        await language_switcher.collect_segment_native_usage()
                    except asyncio.CancelledError as exc:
                        # Held like the closes above: settlement still runs.
                        native_floor_cancellation = native_floor_cancellation or exc
            # `aclose` is what emits the close event, so any browser-test report
            # exists by now. Waited for HERE because the report is the room's own
            # metadata and cleanup below deletes that room: a write still in
            # flight then is a report the browser never receives. Bounded on the
            # same budget the reporter uses, so neither side outlives the other.
            if browser_test_report_task is not None:
                try:
                    browser_report_wait = await await_bounded_teardown_task(
                        browser_test_report_task,
                        timeout_seconds=BROWSER_TEST_REPORT_BUDGET_SECONDS,
                        on_late_completion=lambda error_type: emit(
                            "workspace_browser_test_error_publish_late_result",
                            call_id=call.call_id,
                            completed=not bool(error_type),
                            error_type=error_type,
                        ),
                    )
                    if not browser_report_wait.completed:
                        if browser_report_wait.error_type == "TimeoutError":
                            emit(
                                "workspace_browser_test_error_publish_timeout",
                                call_id=call.call_id,
                                cancellation_resistant=(
                                    browser_report_wait.cancellation_resistant
                                ),
                            )
                        else:
                            emit(
                                "workspace_browser_test_error_publish_failed",
                                call_id=call.call_id,
                                error=browser_report_wait.error_type,
                            )
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
            evidence_sidecar_shutdown = await shutdown_evidence_sidecar(
                sidecar,
                capture_tasks,
                on_late_completion=lambda phase, error_type: emit(
                    "evidence_sidecar_shutdown_late_result",
                    call_id=call.call_id,
                    phase=phase,
                    completed=not bool(error_type),
                    error_type=error_type,
                ),
            )
            for phase, outcome in (
                ("disconnect", evidence_sidecar_shutdown.disconnect),
                ("captures", evidence_sidecar_shutdown.captures),
            ):
                if not outcome.completed:
                    emit(
                        "evidence_sidecar_shutdown_incomplete",
                        call_id=call.call_id,
                        phase=phase,
                        error_type=outcome.error_type,
                        cancellation_resistant=outcome.cancellation_resistant,
                    )
            # Drop the carrier leg before usage reporting, evidence mix,
            # lifecycle posts and artifact upload. Those are ``to_thread``
            # network calls and local I/O; they do not need the room. Leaving
            # the line up for them was measured at ~8 s of silence after the
            # farewell. The SIP bridge writes call statistics only
            # after this delete, so ``attach_sip_call_statistics`` in
            # ``finalize_call_artifacts`` must run afterwards.
            await drop_carrier_leg_then_continue(
                lambda: asyncio.shield(hangup_carrier()),
                on_failure=lambda exc: emit(
                    "room_hangup_failed",
                    call_id=call.call_id,
                    error_type=type(exc).__name__,
                ),
            )
            if browser_test_report_task is None:
                await close_sdk_session()
                if language_switcher is not None:
                    # After the session closed: the reply the caller hung up on
                    # ends only then, and its TTS meter must see that ending.
                    try:
                        await language_switcher.collect_segment_native_usage()
                    except asyncio.CancelledError as exc:
                        # Held like the closes above: settlement still runs.
                        native_floor_cancellation = native_floor_cancellation or exc
            if is_eleven_native_config(call.config):
                # The API prices the realtime line on the call's own connected
                # duration; only successful tool counts cross the worker boundary,
                # and the fact that the provider never started a conversation.
                # That one is sent only when false, so a started call's report is
                # unchanged and an older api still settles it (both deploy orders).
                try:
                    await asyncio.to_thread(call.report_component_usage, {
                        "source": "livekit_agent_sdk",
                        "tool_units": successful_tool_usage.snapshot() if successful_tool_usage else {},
                        **({} if gpt_live_llm is not None and gpt_live_llm.conversation_started
                           else {"realtime_conversation_started": False}),
                    })
                except asyncio.CancelledError as exc:
                    native_floor_cancellation = native_floor_cancellation or exc
                except Exception as exc:
                    emit("component_usage_report_failed", call_id=call.call_id, error_type=type(exc).__name__,
                         status_code=getattr(exc, "status_code", None), reason=str(getattr(exc, "reason", "") or ""))
            elif isinstance(call.config.get("component_billing"), dict) and call.config["component_billing"].get("usage_required") is True:
                model_usage_records: list[Any] = []
                soniox_usage = None
                try:
                    if language_switcher is not None:
                        soniox_usage = None
                    elif stack.soniox_usage_tracker is not None:
                        try:
                            soniox_usage = await stack.soniox_usage_tracker.collect(
                                deadline=stack.soniox_usage_tracker.default_deadline()
                            )
                        except asyncio.CancelledError as exc:
                            # The job's own cancellation, never a deadline:
                            # held, re-raised after settlement and room cleanup.
                            native_floor_cancellation = native_floor_cancellation or exc
                            raise SonioxUsageEvidenceError(
                                "Soniox usage collection was cancelled", kind="cancelled"
                            ) from None
                        finally:
                            stack.soniox_usage_tracker.discard_credentials()
                    else:
                        soniox_usage = None
                    # Nothing more may reach a TTS provider once its evidence
                    # is taken; a still-running attempt may end on its own.
                    await settle_meters(stack.tts_unit_meter)
                    # A live session whose provider reports no session time
                    # reports what its clock accrued now, whether the SDK close
                    # ran, was skipped or is still running.
                    report_session_time(gpt_live_llm)
                    # One snapshot feeds both the priced payload and the audit
                    # line: re-flattening could observe a different set.
                    model_usage_records = billing_usage_collector.flatten()
                    tts_native_usage = None
                    if language_switcher is None and stack.tts_unit_meter is not None:
                        # Numbers only: requests per ending and what each was
                        # billed, so a barge-in's charge is visible in the log.
                        emit(
                            "tts_meter_settled",
                            call_id=call.call_id,
                            unit=stack.tts_unit_meter.billing.unit,
                            **stack.tts_unit_meter.outcome_summary(),
                        )
                        tts_native_usage = stack.tts_unit_meter.snapshot()
                    usage_payload = (
                        normalized_usage_from_profile_segments(
                            call.config,
                            language_switcher.usage_segments(),
                            tool_units=(successful_tool_usage.snapshot() if successful_tool_usage else {}),
                        )
                        if language_switcher is not None
                        else normalized_usage_from_model_usage(
                            call.config,
                            model_usage_records,
                            tool_units=(successful_tool_usage.snapshot() if successful_tool_usage else {}),
                            tts_native_usage=tts_native_usage,
                            soniox_usage=soniox_usage,
                        )
                    )
                    report_required = component_usage_report_required(
                        usage_payload,
                        confirmed_unanswered=(
                            outbound is not None
                            and call_end_clock["answered_at"] is None
                            and answer_gate_evidence["disposition"]
                            in {SIP_ANSWER_DISCONNECTED, SIP_ANSWER_TIMED_OUT}
                        ),
                    )
                    if not report_required:
                        emit("component_usage_skipped", call_id=call.call_id, reason="unanswered_zero_usage")
                    else:
                        await asyncio.to_thread(call.report_component_usage, usage_payload)
                        if language_switcher is not None:
                            emit(
                                "component_usage_reported",
                                call_id=call.call_id,
                                profile_segment_count=len(usage_payload["segments"]),
                            )
                        else:
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
                                # A zero above is either "this component consumed
                                # nothing" or "this component never reported". Both
                                # settle the same, but only one is a fault worth
                                # chasing, so name the second explicitly.
                                unmeasured_components=list(
                                    unmeasured_pinned_components(
                                        call.config,
                                        model_usage_records,
                                        tts_native_usage=tts_native_usage,
                                    )
                                ),
                            )
                except (
                    MeteringContractError,
                    NativeTTSUsageError,
                    SonioxUsageEvidenceError,
                    SegmentNativeUsageUnavailable,
                ) as exc:
                    # Do not replace missing native evidence with a duration
                    # estimate. sip.ended will retain the existing hold as
                    # awaiting_usage for an explicit retry/reconciliation path.
                    #
                    # ``reason`` carries the metering module's OWN message --
                    # authored text naming catalog identifiers and field names,
                    # never provider output or credentials. It is here because
                    # the class name alone made this branch undiagnosable: a
                    # hold stranded for three weeks logged nothing an operator
                    # could act on. Provider evidence also names WHY it is
                    # missing (``soniox:missing_rows``, ``soniox:http_status:429``):
                    # the refused-metering API takes no extra field, so that
                    # bounded discriminator travels in its ``reason``.
                    discriminator = getattr(exc, "discriminator", None)
                    emit(
                        "component_usage_unavailable",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                        reason=str(exc),
                        kind=getattr(exc, "kind", None),
                        status_code=getattr(exc, "status_code", None),
                    )
                    capture_exception(RuntimeError("native_metering_unavailable"), component="voice-worker", phase="usage_reporting")
                    persist_cancelled = await persist_refused_metering(
                        call=call,
                        reason=f"{discriminator}: {exc}" if discriminator else str(exc),
                        # A Soniox failure lands before the billable snapshot,
                        # which stays after collection so a late LLM metric is
                        # not lost; the refused record takes its own.
                        model_usage_records=model_usage_records or billing_usage_collector.flatten(),
                        stack=stack,
                        soniox_usage=soniox_usage,
                    )
                    if persist_cancelled is not None:
                        refused_metering_cancellation = persist_cancelled
                except Exception as exc:
                    # Network/report failures are also non-billable until the
                    # durable control-plane retry/reconciliation path obtains
                    # the report. The media teardown must continue.
                    #
                    # The HTTP status and the API's reason code, not only the
                    # class: an unanswered call's report arrives after the ring
                    # timeout released its reservation, and that expected 409
                    # must be told apart from a real failure.
                    rejection_reason = str(getattr(exc, "reason", "") or "")
                    emit(
                        "component_usage_report_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                        status_code=getattr(exc, "status_code", None),
                        reason=rejection_reason,
                    )
                    if rejection_reason == USAGE_AFTER_SETTLEMENT_REASON:
                        # Not billable -- the customer is not charged for an
                        # unanswered call -- but the session the platform paid
                        # for is kept where the other refused reports are.
                        persist_cancelled = await persist_refused_metering(
                            call=call,
                            reason="component usage reported after the call's billing was final",
                            model_usage_records=model_usage_records,
                            stack=stack,
                            soniox_usage=soniox_usage,
                        )
                        if persist_cancelled is not None:
                            refused_metering_cancellation = persist_cancelled
            if gpt_live_http is not None:
                # The live session borrows this transport, so it is told the
                # call is over BEFORE the socket under it goes, and the
                # transport is closed either way. Closed the other way round
                # the plugin reads its own websocket dying as a provider
                # fault, retries onto the closed aiohttp session, and reports
                # an unrecoverable AI error -- which is what made every
                # unanswered outbound call page as a GPT-Live failure. Nothing
                # here can extend the session's life: on the paths that reach
                # this line the room is already gone.
                # Stamped immediately before OUR deliberate close, never at
                # the head of the teardown (Codex round 4). "No input was
                # released" says the answer gate never completed; it does NOT
                # say the ending was teardown-caused. A real provider failure
                # during ringing ends the call the same way, and a flag set
                # for the whole teardown would drop that failure's own delayed
                # records -- hiding exactly what the voice-worker push exists
                # for. Set here, only records produced BY this close are
                # dropped; the original failure reported long before.
                if answer_gate_evidence.get("input_released_ms") is None:
                    mark_provider_close_in_progress(call_id=call.call_id)
                try:
                    live_session_close = await close_live_session_and_transport(
                        session_agent,
                        gpt_live_llm,
                        gpt_live_http,
                        on_transport_error=lambda error_type: emit(
                            "gpt_live_http_close_failed",
                            call_id=pre_media_call_id,
                            error_type=error_type,
                        ),
                    )
                finally:
                    # Bounded to the close itself. A record arriving after it
                    # is a fault again -- reported, not dropped, which is the
                    # safe direction for a straggler.
                    clear_provider_close_in_progress()
                emit(
                    "live_session_closed_before_transport",
                    call_id=call.call_id,
                    status=live_session_close,
                    # Who ended the provider session, in the evidence bundle.
                    # ``input_released_ms`` is the answer gate actually
                    # releasing the caller's audio into the engine, so its
                    # absence is the one fact that says the engine never had a
                    # live call to lose.
                    closed_by=(
                        "call_end_before_answer"
                        if answer_gate_evidence.get("input_released_ms") is None
                        else "call_end"
                    ),
                    # NOT the raw disposition: it defaults to
                    # ``not_applicable``, which means "this leg is not an
                    # outbound SIP call". A gate that was armed and never
                    # resolved -- exactly this incident -- would inherit that
                    # default and tell an operator the gate did not apply to a
                    # call it was pending on. ``wait_ms`` is written only when
                    # the watch resolves, which is the discriminator the
                    # evidence dict already keeps for telling "no gate ran"
                    # from "the gate ran and never resolved".
                    answer_gate_disposition=(
                        answer_gate_evidence.get("disposition")
                        if answer_gate_evidence.get("wait_ms") is not None
                        else ("pending" if answer_watch is not None else SIP_ANSWER_NOT_APPLICABLE)
                    ),
                )
            # The voicemail record is frozen HERE, before the
            # artifact is written, because transcript.json is serialized by
            # finalize and the operational commit below comes after it. It
            # carries which rule ended the call; it changes nothing about how.
            try:
                voicemail_section = answer_detection.voicemail_section(
                    call_end_arbiter.reason or ""
                )
                if voicemail_section is not None:
                    recorder.set_voicemail_detection(voicemail_section)
            except Exception as exc:  # noqa: BLE001 - evidence never gates teardown
                _quietly(
                    emit,
                    "voicemail_detection_failed",
                    call_id=call.call_id,
                    error_type=type(exc).__name__,
                )
            if s2s_live_call:
                # A voice-to-voice engine closes a caller turn only on its own
                # quiet input, which stops at hangup: the caller's last words
                # are still an interim. Recorded as finals, marked as cut.
                try:
                    flushed = turn_events.flush_open_caller_turns()
                    if flushed:
                        emit(
                            "caller_turns_finalized_at_close",
                            call_id=call.call_id,
                            count=flushed,
                        )
                    liveness = duplex_liveness.section()
                    recorder.set_duplex_liveness(liveness)
                    # How often the uncaptioned-speech rule could fire and
                    # did, and how much of it began over the agent's own
                    # speech (a barge-in or an echo; nothing here can tell).
                    emit(
                        "caller_speech_without_caption",
                        call_id=call.call_id,
                        segments_recorded=turn_events.caller_segments_recorded,
                        segments_over_agent=turn_events.caller_segments_over_agent,
                        segments_withheld=turn_events.caller_segments_withheld,
                        segments_withheld_then_heard=turn_events.caller_segments_withheld_then_heard,
                        marked=turn_events.caller_spoke_marked,
                        kept=turn_events.caller_spoke_kept,
                        marked_over_agent=turn_events.caller_spoke_marked_over_agent,
                        segments_dropped_at_cap=turn_events.caller_segments_dropped_at_cap,
                        speeches_untracked=turn_events.engine_speeches_untracked,
                    )
                    emit(
                        "duplex_liveness",
                        call_id=call.call_id,
                        **duplex_liveness.decision_details(),
                    )
                except Exception as exc:  # noqa: BLE001 - evidence never gates teardown
                    _quietly(
                        emit,
                        "s2s_evidence_close_failed",
                        call_id=call.call_id,
                        error_type=type(exc).__name__,
                    )
            # Off the loop: mixing a long call's tracks is seconds of CPU, and
            # on the loop it stalled every other call this worker carries.
            finalized = await finalize_off_loop(recorder)
            summary: dict[str, object] | None = finalized.summary
            if finalized.cancellation is not None:
                evidence_finalize_cancellation = finalized.cancellation
            if finalized.no_room_audio:
                emit(
                    "evidence_unavailable",
                    call_id=call.call_id,
                    semantic_evidence_status="unavailable_no_room_audio",
                )
            if s2s_live_call and summary is not None:
                # The completeness measure a duplex call can actually fail:
                # agent speeches the worker saw start playing, at the SDK's
                # speaking edge, against the ones the transcript carries text
                # for. ``evidence_incomplete`` is the pairing check and is
                # structurally true on these calls. Not the handle-completion
                # counters: a duplex tool call is a generation of its own that
                # completes without ever speaking.
                report_agent_speech_coverage(
                    summary,
                    agent_speeches_counted=turn_events.agent_speaking_edges,
                    emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
                    capture=capture_exception,
                    threshold=coverage_threshold(),
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
            if opening_summary_task is not None and not opening_summary_task.done():
                # A teardown the call's ending did not start: the task's own
                # ``finally`` writes the record on cancellation.
                opening_summary_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await opening_summary_task
            try:
                # Emitted once after media has stopped: the per-call totals a
                # controlled interruption A/B compares against its baseline.
                # Bounded counters only -- no transcript text, no audio, no
                # provider identity.
                if transcript_gate is not None:
                    gate_stats = transcript_gate.gate.stats
                    interruption_telemetry.on_transcripts_gated(
                        withheld=int(gate_stats["held"]),
                        released=int(gate_stats["released"]),
                    )
                    # Why a reply may have run to the end: the caller was
                    # agreeing with it, not asking for the floor.  Without this
                    # number a call where every barge-in was refused looks
                    # identical to one where nobody spoke.  Read together with
                    # ``min_interruption_words_effective`` in the same summary,
                    # which is what says whether these utterances were actually
                    # denied the floor or merely observed.
                    interruption_telemetry.note_call_totals(
                        caller_backchannel_utterances=int(
                            gate_stats["caller_backchannel_utterances"]
                        ),
                    )
                # What the two halves actually did on this call: was the
                # greeting protected, how many pickups it ignored, and how
                # much audio the boundary tails cost. Bounded scalars only.
                interruption_telemetry.note_call_totals(
                    **opening_protection.stats,
                    **boundary_stop.stats,
                    **graceful_stop_stats.stats,
                    **(alignment_feed.stats if alignment_feed is not None else {}),
                    **interruption_word_gate.stats,
                    **_playout_hold_stats(playout_hold),
                    **model_unresponsive.shadow_stats,
                )
                if duplex_opening_hold is not None:
                    # After the protection's totals, which it corrects: on a
                    # duplex engine the protection credits whatever burst
                    # came first, and only the hold knows what was spoken.
                    interruption_telemetry.note_call_totals(**duplex_opening_hold.stats)
                    interruption_telemetry.note_duplex_opening(
                        outcome=duplex_opening_hold.outcome,
                        delivered=duplex_opening_hold.opening_delivered(),
                    )
                # The per-call record: whether caller audio kept reaching
                # recognition, and every VAD run rather than the first. One row
                # each, bounded, with their own truncation flags. Diagnostic
                # only -- no ending, disposition or billing reads these.
                # Polled once more first, so a gap still open at teardown is
                # measured to here instead of being lost with the call.
                caller_audio_flow.poll()
                emit(
                    "caller_audio_flow_summary",
                    call_id=call.call_id,
                    **caller_audio_flow.snapshot(),
                )
                emit(
                    "caller_vad_edge_summary",
                    call_id=call.call_id,
                    # The clock, so a caller still speaking when the call ends
                    # has that run measured instead of dropped: it has no
                    # end-of-speech edge, and it is the shape worth keeping.
                    **caller_vad_edges.snapshot(at=time.monotonic()),
                )
                # Whether the callee was ever confirmed answered, how long
                # this worker waited to find out, and how long after that the
                # opening was actually issued -- none of this survived past
                # the worker's own stdout before. ``None`` entries (no gate
                # ran; the opening was never issued) are omitted, not written
                # as null.
                interruption_telemetry.note_answer_gate(
                    disposition=answer_gate_evidence["disposition"],
                    wait_ms=answer_gate_evidence["wait_ms"],
                    input_released_ms=answer_gate_evidence["input_released_ms"],
                    opening_generate_ms=answer_gate_evidence["opening_generate_ms"],
                )
                # Its own name, never ``summary``. These are interruption
                # counters; ``summary`` above is the recorder's evidence
                # contract and is the only thing ``finalize_call_artifacts``
                # can read. Binding both to one name silently handed the
                # counters to finalize, whose first field read then raised
                # and took the whole artifact import with it.
                interruption_summary = interruption_telemetry.summary()
                emit(
                    "interruption_summary",
                    call_id=call.call_id,
                    **interruption_summary,
                )
                # The same counters, on disk beside the transcript and the
                # telemetry. They were stdout-only, and the campaign-readiness
                # scorer -- which measures sudden cut-offs and graceful
                # barge-in from exactly these numbers -- reads an evidence
                # directory and cannot read a log. Bounded scalars only, the
                # same ones already emitted: no transcript text, no audio, no
                # provider identity.
                (evidence_dir / "interruption-summary.json").write_text(
                    json.dumps(
                        {"call_id": call.call_id, **interruption_summary},
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                emit(
                    "interruption_summary_failed",
                    call_id=call.call_id,
                    error_type=type(exc).__name__,
                )
            # One line per call whatever happened, so a call with no record is
            # a named status in the log and never an absent file to explain.
            # The write cannot raise (see ``write_turn_end_evidence``), and the
            # line is emitted quietly: what follows it here commits the call's
            # classification and reports its end, and a diagnostic that cannot
            # reach stdout must not be what skips them.
            _quietly(
                emit,
                TURN_END_EVIDENCE_SUMMARY_EVENT,
                call_id=call.call_id,
                **(
                    write_turn_end_evidence(
                        evidence_dir, call_id=call.call_id, evidence=turn_end_evidence
                    )
                    if turn_end_evidence is not None
                    else {TURN_END_EVIDENCE_STATUS_FIELD: "not_applicable"}
                ),
            )
            if realtime_event_trace.enabled:
                try:
                    (evidence_dir / "realtime-event-trace.json").write_text(
                        json.dumps(realtime_event_trace.snapshot(), sort_keys=True), encoding="utf-8"
                    )
                except Exception as exc:
                    emit("realtime_event_trace_write_failed", call_id=call.call_id, error_type=type(exc).__name__)
            # Committed BEFORE the report that carries it: the record is
            # frozen once, here, and the same two strings go to the call row,
            # the results page and the customer's webhook. A classification
            # computed afterwards would be a second answer to a question the
            # customer had already been given.
            answer_detection.commit(call_end_arbiter.reason or "")
            terminal_lifecycle_reported = await report_call_teardown_lifecycle(
                call=call,
                call_id=call.call_id,
                ended_at=ended_at,
                emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
                answer_classification=answer_detection.terminal_details(),
                # Empty unless THIS worker decided the call was over -- which
                # now includes an ordinary caller hangup, because the delivered
                # disconnect event routes through the same arbiter as the
                # occupancy poll. So from this release the commonest ending
                # carries remote_party_disconnected where it carried nothing.
                # The detection path deliberately does NOT go here: these
                # details reach the customer's call.ended webhook verbatim, and
                # it is already recorded in this worker's own evidence.
                termination_detail=call_end_arbiter.reason,
                # Qualifies the reason above, not the classification beside it.
                # Empty on every ending that needs no qualifier, which is all
                # of them but the silent-model rule.
                ended_reason_detail=call_end_arbiter.reason_detail,
                prior_person=call_end_arbiter.prior_person,
            )
        except BaseException:
            await asyncio.shield(cleanup_room())
            raise
        if browser_startup is not None and call.browser_outcome_finalized:
            write_session_state(state_path, browser_outcome_pending=False)
        try:
            # Held through a cancellation like every other teardown step: the
            # room is already gone here, so the SDK's 15 s budget can run out
            # during the upload. What the hold buys is the upload itself. By
            # the time a cancelled finalize writes its session state the
            # reconciler has usually swept the call already (the heartbeat
            # stopped at teardown), and first terminal wins; the sweep bills
            # from the worker's own decided reason and instant, so that is
            # harmless.
            _, artifacts_finalize_cancellation = await finalize_through_cancellation(
                finalize_call_artifacts(
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
                    session_status=(
                        "complete" if terminal_lifecycle_reported else "awaiting_reconciliation"
                    ),
                    participant_identity=participant.identity,
                    sip_statistics_dir=sip_statistics_capture_directory(SESSION_ROOT),
                ),
                emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            )
        finally:
            await asyncio.shield(cleanup_room())
            # The call's events stop here; anything the job still emits is
            # the worker's, not this call's.
            worker_events.close_journal(call.call_id)
            value_redactor.close()
        # The teardown has finished, including both room-delete paths, so the
        # cancellation deferred at the report wait can surface now. Without this
        # a cancelled job returns as though it had succeeded.
        if native_floor_cancellation is not None:
            raise native_floor_cancellation
        if terminal_end_call_cancellation is not None:
            raise terminal_end_call_cancellation
        if end_call_guard_settle_cancellation is not None:
            raise end_call_guard_settle_cancellation
        if browser_test_report_cancellation is not None:
            raise browser_test_report_cancellation
        if audio_environment_close_cancellation is not None:
            raise audio_environment_close_cancellation
        if acknowledgements_close_cancellation is not None:
            raise acknowledgements_close_cancellation
        if refused_metering_cancellation is not None:
            raise refused_metering_cancellation
        if evidence_finalize_cancellation is not None:
            raise evidence_finalize_cancellation
        if artifacts_finalize_cancellation is not None:
            raise artifacts_finalize_cancellation


if __name__ == "__main__":
    # The supervisor process gets its own client: registration, dispatch and
    # drain all fail here, in a process no job ever runs in.
    install_error_tracking(component="voice-worker", report=emit)
    # Import the provider plugins HERE, in the supervisor, before the library
    # starts. Not for this process's benefit -- it never runs a call -- but
    # because importing a ``livekit.plugins.*`` package registers it, and
    # livekit-agents builds its forkserver preload list from that registry.
    # Registered nothing, and the forkserver preloads nothing: the production
    # host's own boot log reads
    # ``preloading plugins {"packages": ["av", ...warmup, ...freeze]}``, and
    # every job process then rebuilds the whole import set for itself, ~0.8 s
    # and 0.7 CPU-s of it, at the moment a call is waiting for that process.
    # Doing it here puts the modules in the forkserver, where children inherit
    # them copy-on-write and the ``setup_process`` call below finds them warm.
    #
    # Best-effort for the same reason ``setup_process`` is: a plugin that
    # fails to import must not stop this worker from starting and taking
    # calls with the providers that did load. Its failure is reported, not
    # swallowed -- and reported from the supervisor, whose stdout is the
    # container log an operator actually reads.
    supervisor_warmed, supervisor_failed = prewarm_plugin_modules(PREWARM_PLUGIN_MODULES)
    emit(
        "supervisor_plugins_registered",
        components=supervisor_warmed,
        failed=[{"module": name, "error": error} for name, error in supervisor_failed],
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
        phone_capacity=PHONE_CAPACITY,
    )
    # Once, before any job process forks: every call this worker runs
    # inherits where its deployment keeps call content.
    adopt_deployment_content_storage(os.environ)
    atexit.register(_stop_service_heartbeat)
    _start_service_heartbeat()
    cli.run_app(server)

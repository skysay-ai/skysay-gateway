# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One call owner for governed input, authored or live speech and source facts.

The metadata plan is built before this owner exists. No freeform constructor,
ambient provider credential or automatic model reply belongs on this path.
"""
from __future__ import annotations

import asyncio
import functools
import time
from pathlib import Path
from collections.abc import Mapping
from uuid import uuid4

try:
    from . import production as lifecycle
    from . import worker_events
    from .control_plane import linked_participant_kinds
    from .funded_runtime import FundedRuntime
    from .answer_detection import AnswerDetectionSession, resolved_mode, CLASSIFY_INTERVAL_SECONDS
    from .answer_classifier import WORD_SOURCE_STT_FINAL
    from .voicemail_watch import resolved_setting as voicemail_mode
    from .navigation_entry import NavigationPolicy, NavigationSettings, ENDER_ARBITER, ENDER_NO_CALLER_SPEECH
    from .opening_turn import callee_first_exclusion, SUMMARY_EVENT as OPENING_SUMMARY_EVENT
    from .caller_audio_flow import CallerAudioFlowLog, CallerVadEdgeLog
    from .answer_detection import frame_amplitude
    from .sip_answer_gate import clock_anchor
    from .workflow_root_evidence import PickupEvidence, RootOpeningRecord, WorkflowLineTimeline
    from .livekit_evidence import SynchronizedEvidenceRecorder, finalize_off_loop
    from .evidence_contract import effective_voice_stack_identity
    from .call_transcript_events import CallTranscriptEventRecorder
    from .call_telemetry import CallTelemetryCollector
    from .workflow_detector import load_workflow_detector
    from .usage_report_outbox import UsageReportOutbox
    from .sip_answer_gate import await_callee_answer, may_speak
    from .workflow_authored_tts import AuthoredTtsPreparation
    from .workflow_call import WorkflowCall
    from .workflow_input import (CallerInput, WorkflowInputRefused, WorkflowMediaEnded, WorkflowInputBacklogExceeded,
        MAX_CALLER_INGRESS_BACKLOG_SECONDS)
    from .workflow_listener_source import WorkflowListenerSource
    from .workflow_live_output import LiveReplyOutput, open_live_workflow_output, sdk_reply_timeout_seconds
    from .workflow_lifecycle_ack import WorkflowLifecycleAcknowledgements
    from .workflow_pickup import WorkflowPickup
    from .workflow_playback import create_workflow_playback_session
    from .workflow_reply_output import WorkflowReplyOutput, WorkflowControllerAudio, create_workflow_audio_floor
    from .workflow_source_meter import WorkflowSourceMeter
    from .workflow_source_owner import WorkflowListenerRefusal
    from .workflow_root_language import RootLanguageEvidence, RootOutputLanguage
    from .workflow_input_recognition import CallerLanguageHint
    from .workflow_input_ring import ListenerReconnect
    from .capabilities.workflow_sources import MAX_CALL_ATTEMPTS
    from .workflow_speech_segments import split_source_line
    from .workflow_speech import LineNotVoiced
    from .workflow_tasks import join_owned_task
except ImportError:
    import production as lifecycle
    import worker_events
    from control_plane import linked_participant_kinds
    from funded_runtime import FundedRuntime
    from answer_detection import AnswerDetectionSession, resolved_mode, CLASSIFY_INTERVAL_SECONDS
    from answer_classifier import WORD_SOURCE_STT_FINAL
    from voicemail_watch import resolved_setting as voicemail_mode
    from navigation_entry import NavigationPolicy, NavigationSettings, ENDER_ARBITER, ENDER_NO_CALLER_SPEECH
    from opening_turn import callee_first_exclusion, SUMMARY_EVENT as OPENING_SUMMARY_EVENT
    from caller_audio_flow import CallerAudioFlowLog, CallerVadEdgeLog
    from answer_detection import frame_amplitude
    from sip_answer_gate import clock_anchor
    from workflow_root_evidence import PickupEvidence, RootOpeningRecord, WorkflowLineTimeline
    from livekit_evidence import SynchronizedEvidenceRecorder, finalize_off_loop
    from evidence_contract import effective_voice_stack_identity
    from call_transcript_events import CallTranscriptEventRecorder
    from call_telemetry import CallTelemetryCollector
    from workflow_detector import load_workflow_detector
    from usage_report_outbox import UsageReportOutbox
    from sip_answer_gate import await_callee_answer, may_speak
    from workflow_authored_tts import AuthoredTtsPreparation
    from workflow_call import WorkflowCall
    from workflow_input import (CallerInput, WorkflowInputRefused, WorkflowMediaEnded, WorkflowInputBacklogExceeded,
        MAX_CALLER_INGRESS_BACKLOG_SECONDS)
    from workflow_listener_source import WorkflowListenerSource
    from workflow_live_output import LiveReplyOutput, open_live_workflow_output, sdk_reply_timeout_seconds
    from workflow_lifecycle_ack import WorkflowLifecycleAcknowledgements
    from workflow_pickup import WorkflowPickup
    from workflow_playback import create_workflow_playback_session
    from workflow_reply_output import WorkflowReplyOutput, WorkflowControllerAudio, create_workflow_audio_floor
    from workflow_source_meter import WorkflowSourceMeter
    from workflow_source_owner import WorkflowListenerRefusal
    from workflow_root_language import RootLanguageEvidence, RootOutputLanguage
    from workflow_input_recognition import CallerLanguageHint
    from workflow_input_ring import ListenerReconnect
    from capabilities.workflow_sources import MAX_CALL_ATTEMPTS
    from workflow_speech_segments import split_source_line
    from workflow_speech import LineNotVoiced
    from workflow_tasks import join_owned_task


def managed_source_keys_for_workflow(plan, config, *, environment, leased_components=None):
    """Copy selected platform keys at bootstrap; never lease or read ambient state."""
    try:
        from .workflow_listener_runtime import WorkflowSourceCredentialRefused
    except ImportError:
        from workflow_listener_runtime import WorkflowSourceCredentialRefused
    if not isinstance(environment, Mapping) or not isinstance(config, dict):
        raise WorkflowSourceCredentialRefused()
    leases = {} if leased_components is None else leased_components
    if not isinstance(leases, Mapping):
        raise WorkflowSourceCredentialRefused()
    billing = config.get("component_billing")
    components = billing.get("components") if isinstance(billing, dict) else None
    if not isinstance(components, list) or not 2 <= len(components) <= 16:
        raise WorkflowSourceCredentialRefused()
    try:
        from .source_keys import managed_keys_for_sources
    except ImportError:
        from source_keys import managed_keys_for_sources
    profiles = getattr(plan, "output_profiles", {})
    entries = billing.get("workflow_output_profiles") if profiles else None
    if profiles and (type(entries) is not dict or set(entries) != set(profiles)):
        raise WorkflowSourceCredentialRefused()
    selected = [(plan.listener_adapter, None)]
    selected.extend((profile.adapter, language) for language, profile in profiles.items())
    if not profiles:
        selected.append((plan.output_adapter, None))
    entries_to_resolve = []
    for declaration, language in selected:
        matched = ([entries[language]] if language is not None else
                   [item for item in components if isinstance(item, dict) and item.get("source_role") == declaration.role])
        if len(matched) != 1:
            raise WorkflowSourceCredentialRefused()
        item = matched[0]
        if any(item.get(key) != getattr(declaration, key) for key in ("component", "provider", "model")):
            raise WorkflowSourceCredentialRefused()
        entries_to_resolve.append((declaration, item))
    return managed_keys_for_sources(entries_to_resolve, environment=environment, leased_components=leases)


async def _thread(function, *args, **kwargs):
    """Cancellation joins a blocking HTTP request before terminal settlement."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    await join_owned_task(task)
    return task.result()





#: How long an answered transfer's person may take to appear in the room.
TRANSFER_PARTY_JOIN_SECONDS = 10.0
#: How long a call whose person already left waits for the run to record
#: its transfer before it ends: past the outcome report's own 3 s bound and
#: one outcome turn.
TRANSFER_OUTCOME_COMMIT_SECONDS = 15.0
#: The runner's fail reason for a step the caller spoke over, while its line
#: was still unheard, more often than the runner's bound allows
#: (``conversation_workflows.runner.FAIL_UNHEARD_REPEATS_EXHAUSTED``; a test
#: pins the two copies together). The worker cannot import the control plane.
UNHEARD_REPEATS_EXHAUSTED = "UNHEARD_REPEATS_EXHAUSTED"


class _ClassifiedOutput:
    def __init__(self, root, owner, report=None):
        # ``report(started_at, status)``: a reply-level latency record for an
        # output that reports no line of its own. Evidence only.
        self.root, self.owner, self.report = root, owner, report

    async def pause_for_floor(self):
        return await self.owner.pause_for_floor()

    async def resume_from_floor(self):
        return await self.owner.resume_from_floor()

    async def pause_for_caller(self):
        pause = getattr(self.owner, "pause_for_caller", None)
        return callable(pause) and await pause() is True

    def resume_for_caller(self):
        resume = getattr(self.owner, "resume_for_caller", None)
        if callable(resume):
            resume()

    async def run(self):
        while self.root.answer_detection.arbiter.silence_the_agent and not self.root.ended.is_set():
            await asyncio.sleep(.05)
        if self.root.ended.is_set():
            return "interrupted"
        self.root.answer_detection.note_opener_started()
        if self.report is None:
            return await self.owner.run()
        started, status = time.monotonic(), None
        try:
            status = await self.owner.run()
            return status
        finally:
            self.report(started, status)

    async def present(self, boundary):
        await self.owner.present(boundary)


class WorkflowRoot:
    def __init__(self, *, plan, ctx, participant, call, adapter, browser_test, outbound,
                 provider_call_id, inbound_route, inbound_release_canary,
                 report_browser_test_failure, remove_participant, start_sidecar,
                 evidence_dir: Path, state_path: Path, ringing_timeout_seconds: float, callee_first_enabled: bool, emit,
                 managed_source_keys=None):
        self.plan, self.ctx, self.participant, self.call = plan, ctx, participant, call
        self.managed_source_keys = managed_source_keys
        self.authored_resources = {}
        # The live role's one GPT-Live model, built in _prepare and closed in
        # _join_media; every other role leaves it None.
        self.live_output = None
        self.adapter, self.browser_test, self.outbound = adapter, browser_test, outbound
        self.provider_call_id, self.inbound_route = provider_call_id, inbound_route
        self.inbound_release_canary = inbound_release_canary
        self.report_browser_test_failure, self.remove_participant = report_browser_test_failure, remove_participant
        self.start_sidecar, self.evidence_dir = start_sidecar, evidence_dir
        self.ringing_timeout_seconds, self.emit = ringing_timeout_seconds, emit
        self.callee_first_enabled = callee_first_enabled
        self.started = self.media_started = False
        self.acknowledgements = WorkflowLifecycleAcknowledgements(call.call_id)
        self.stage = "component_configuration"
        self.session = self.input = self.coordinator = self.pickup = None
        self.audio_floor = self.navigation_input = self.navigation_actions = None
        self.navigation_settling = self.navigation_track = self.value_redactor = None
        self.navigation_watch = self.navigation_audio = None
        self._navigation_pause = None
        self._settling_handoff = False
        self.recorder = self.sidecar = None
        self.transcript_events = None
        self.state_path = state_path
        self.capture_tasks, self.tasks, self.handlers = [], [], []
        self.sources_joined = False
        self.evidence_identity = ""
        self.ended = asyncio.Event()
        self.failure = None
        # Cleanup failures ``_join_media`` saw that are not the call's own
        # failure: an owned line refusal is only settled quietly without them.
        self.cleanup_failures = 0
        # Set once a workflow transfer step's person answered and the two
        # legs are bridged: from then the call is theirs (see ``_bridge``).
        self.transfer_bridged = False
        self._input_task = None
        self._input_failure = None
        # The transferred person's RTC presence: seen joining, then leaving.
        self._transfer_party_joined = asyncio.Event()
        self._transfer_party_left = False
        # The call's one transfer step dial, once placed (``_transfer``).
        self._transfer_controller = None
        self._transfer_party_ending = False
        self._transfer_party_joined_at = None
        self.transport_disconnected = False
        self.started_at = ""
        self.input_released_at = None
        self.caller_has_spoken = False
        self.call_end_clock = {"answered_at": None}
        self.funding = FundedRuntime(call.config.get("funded_runtime") or {})
        self.hangup_intent = {}
        self.transcripts = asyncio.Queue(maxsize=16)
        profiles = getattr(plan, "output_profiles", {})
        declarations = tuple(profile.adapter for profile in profiles.values()) if profiles else (plan.output_adapter,)
        if plan.output_adapter.role == "live" and (any(item.role != "live" for item in declarations) or any(
                profile.constructor != plan.output_constructor for profile in profiles.values())):
            # One session speaks the whole call; its voice is fixed at start.
            raise WorkflowInputRefused("live workflow output requires one session voice for every language")
        self.meter = WorkflowSourceMeter((plan.listener_adapter, *declarations), source_limit=plan.source_limit)
        # The call's one reconnect of a listener whose stream never acknowledged.
        self.listener_reconnect = ListenerReconnect(report=self._listener_reconnected,
                                                    available=lambda: self.meter.remaining_source_budget() > 0)
        self.output_language = self.language_evidence = None
        self._held_hints = {}
        self._last_held_hint_key = None
        # When each turn navigation holds reached the root, for its release's ledger record.
        self._held_arrivals = {}
        if getattr(plan, "follow_caller", None) is not None:
            self.output_language = RootOutputLanguage(languages=tuple(profiles), opening_language=plan.language)
            self.language_evidence = RootLanguageEvidence(plan.follow_caller,
                active_language=lambda: self.output_language.active_language,
                transition_pending=self.output_language.transition_pending, schedule=self.output_language.schedule)
        self.arbiter = None
        self.detector = None
        self.playout_starts = {}
        self.last_playout_start = None
        self.terminal_parts = []
        self.terminal_reply = False
        self.telemetry = CallTelemetryCollector(call_id=call.call_id,
            organization_id=str(call.config.get("organization_id") or ""),
            project_id=str(call.config.get("project_id") or ""))
        # Evidence parity with the freeform path (``workflow_root_evidence``):
        # the caller-audio flow log fed from the capture's ingress, every VAD
        # edge, the outbound opening's record and the per-line latency ledger.
        # None of them decides anything; every feed is guarded.
        def call_emit(event, **details):
            self.emit(event, call_id=call.call_id, **details)
        self.caller_audio_flow = CallerAudioFlowLog(emit=call_emit, clock=time.monotonic)
        self.caller_vad_edges = CallerVadEdgeLog(emit=call_emit)
        self.line_timeline = WorkflowLineTimeline(emit=call_emit)
        self.opening_record = None
        self._first_frame_at = None
        self._speaking_marks = []
        self._evidence_emitted = False
        self._turn_timing = None
        constructor = (getattr(plan, "listener", None) or {}).get("constructor") or {}
        self.navigation, self.answer_detection = lifecycle.build_call_floor_policy(
            call.config, outbound=outbound is not None, language=plan.language, announce=self._announce_navigation,
            emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            quiet_emit=lambda event, **details: emit(event, call_id=call.call_id, **details),
            # The frozen listener is this call's only transcript source; admission already refused one it may not be
            # trusted in, and this is the worker's own check on the config it was given.
            navigation_sources=(("stt", str(constructor.get("provider") or ""), str(constructor.get("model") or "")),)
            if constructor.get("model") else (),
            navigation_languages=self.navigation_languages(),
        )

    def navigation_languages(self):
        """The call's frozen navigation language set: the languages admission froze controller lines for (it
        requires every line in every language of the set, and a listener that covers and is trusted in each)."""
        manifest = self.call.config.get("workflow_controller_manifest")
        lines = manifest.get("lines") if isinstance(manifest, dict) else None
        return tuple(sorted(lines)) if isinstance(lines, dict) and lines else (self.plan.language,)

    def _announce_navigation(self, event, **details):
        self.emit(event, call_id=self.call.call_id, **details)
        if event == "navigation_state":
            if details.get("navigation_to") == "settling":
                self._settling_handoff = True
            elif details.get("navigation_from") == "settling" and details.get("navigation_to") != "conversation":
                self._settling_handoff = False
        if self.navigation_input is not None:
            self.navigation_input.on_policy_event(event, **details)
        if self.navigation_actions is not None:
            self.navigation_actions.timeline.on_policy_event(event, **details)
            self.navigation_actions.gate.sync()
        if self.navigation_settling is not None:
            self.navigation_settling.on_policy_event(event, **details)

    async def run(self):
        self.started = True
        # Ownership has transferred before the first possible suspension.
        error = None
        handled = cancelled = False
        try:
            await self._prepare()
            await self._serve()
        except BaseException as exc:
            error = exc
        finally:
            async def settle():
                await self._join_media()
                if self.acknowledgements.uncertain:
                    await self._settle_unresolved_start()
                    return False
                if self.media_started:
                    await self._settle_media()
                    return False
                if error is not None:
                    return await lifecycle.settle_pre_media_failure(error,
                        ctx=self.ctx, participant=self.participant, call=self.call, adapter=self.adapter,
                        pre_media_call_id=self.call.call_id, pre_media_stage=self.stage,
                        browser_test=self.browser_test, browser_participant_mismatch=False,
                        inbound_route=self.inbound_route, live_pickup=None,
                        report_browser_test_failure=self.report_browser_test_failure,
                        state_path=self.state_path, sidecar=self.sidecar, capture_tasks=self.capture_tasks,
                        inbound_audio_output=None, gpt_live_http=None, emit=self.emit,
                        inbound_release_canary=self.inbound_release_canary, remove_participant=self.remove_participant)
                return False
            cleanup = asyncio.create_task(settle())
            try:
                await join_owned_task(cleanup)
                handled = cleanup.result()
            except asyncio.CancelledError as exc:
                cancelled = True
                error = error or exc
                handled = cleanup.result() if not cleanup.cancelled() else False
            finally:
                worker_events.close_journal(self.call.call_id)
        if (isinstance(error, WorkflowMediaEnded) and self.sources_joined
                and self.arbiter is not None and self.arbiter.reason == lifecycle.TERMINATION_REMOTE_PARTY_DISCONNECTED
                and self.participant.identity not in self.ctx.room.remote_participants):
            # RTC can retire the microphone before emitting participant_left.
            # The joined cleanup gives that authoritative callback a chance
            # to arrive. Only its actual departure verdict classifies this
            # expected media end; source failures and live-track loss escape.
            return
        if (isinstance(error, LineNotVoiced) and error is self.failure and not cancelled and self.media_started
                and self.arbiter is not None and self.arbiter.reason == lifecycle.TERMINATION_WORKFLOW_LINE_UNDELIVERED
                and not self.cleanup_failures):
            # The call's own, owned ending: the line refusal was reported when
            # it happened (``workflow_line_undelivered``), stamped as the call's
            # reason and settled with every owner joined cleanly. Raising it
            # again would only make an expected ending an unhandled job error.
            return
        if error is not None and (isinstance(error, asyncio.CancelledError) or not handled):
            raise error

    def _listen(self, name, callback):
        self.ctx.room.on(name, callback)
        self.handlers.append((name, callback))

    def _spawn(self, coroutine, *, failed=None):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        def done(completed):
            if not completed.cancelled() and (error := completed.exception()) is not None:
                if not self.ended.is_set():
                    (failed or self._failed)(error)
        task.add_done_callback(done)
        return task

    def _workflow_failed(self, error):
        """A failure of the workflow's own machinery: its turns, its output session.

        Once the caller and the person they were transferred to are talking,
        the workflow has nothing more to do on the call, so such a failure is
        recorded and does not hang up on them. Every other failure (the
        heartbeat, the watchdog, funding) still ends the call (``_failed``).
        """
        if self.transfer_bridged and not self.ended.is_set():
            self.emit("workflow_failure_after_transfer", error=type(error).__name__, call_id=self.call.call_id)
            return
        if isinstance(error, LineNotVoiced):
            # The line was played but not said in full, so the workflow cannot
            # go on: the call ends with that reason, never the caller-hung-up
            # reading an unstamped ending gets. A caller who already left keeps
            # their ending (the arbiter's first reason stands).
            try:
                self._end_line_undelivered("not_spoken")
            except Exception:  # noqa: BLE001 - the ending below must still happen
                pass
        self._failed(error)

    def _end_line_undelivered(self, reason):
        """Stamp the call's ending; a failed stamp raises, so its caller still ends the call."""
        try:
            self.emit("workflow_line_undelivered", call_id=self.call.call_id, reason=reason)
        except Exception:  # noqa: BLE001 - evidence cannot stop the ending it records
            pass
        self.arbiter.end(lifecycle.TERMINATION_WORKFLOW_LINE_UNDELIVERED,
                         detected_via=lifecycle.DETECTED_VIA_WORKFLOW_LINE)

    def _input_failed(self, error):
        """The caller's input failed: latch the ending now, not after its cleanup.

        Capture calls this before joining its owned cleanup, and the ingress
        from inside the SDK's enqueue when its backlog passes the cap, so
        ``ended`` is set before the terminal path could stamp
        ``agent_completed``. Only the first input failure is reported: the
        input is finished after it, and its cleanup or the task's completion
        may repeat or replace it. A call that already ended keeps its ending.
        """
        if self.ended.is_set() or self._input_failure is not None:
            return
        self._input_failure = error
        if isinstance(error, WorkflowInputBacklogExceeded):
            try:
                self.emit("workflow_input_backlog_exceeded", call_id=self.call.call_id,
                          backlog_seconds=error.backlog_seconds, cap_seconds=MAX_CALLER_INGRESS_BACKLOG_SECONDS)
            except Exception:  # noqa: BLE001 - evidence cannot stop the ending it records
                pass
        refused = isinstance(error, WorkflowListenerRefusal)
        if refused:
            try:
                self.emit("workflow_listener_refused", call_id=self.call.call_id,
                          listener_provider=self.plan.listener_adapter.provider, reason=error.reason)
            except Exception:  # noqa: BLE001 - evidence cannot stop the ending it records
                pass
        if self.transfer_bridged:
            try:
                self._workflow_failed(error)
            except Exception:  # noqa: BLE001 - its record cannot replace the failure capture re-raises
                pass
        else:
            if refused:
                # The listener's refusal ends the call: its own reason, never
                # the caller-hung-up reading an unstamped teardown gets.
                try:
                    self.arbiter.end(lifecycle.TERMINATION_LISTENER_UNAVAILABLE,
                                     detected_via=lifecycle.DETECTED_VIA_LISTENER_REFUSED)
                except Exception:  # noqa: BLE001 - the ending below must still happen
                    pass
            self._failed(error)

    def _listener_reconnected(self, **fields):
        self.emit("workflow_listener_reconnect", call_id=self.call.call_id,
                  listener_provider=self.plan.listener_adapter.provider, **fields)

    def _failed(self, error):
        if self.failure is None:
            self.failure = error
        self.ended.set()

    def _attributes(self):
        current = self.ctx.room.remote_participants.get(self.participant.identity)
        return None if current is None else dict(current.attributes or {})

    async def _wait_active(self):
        from livekit.agents.utils.participant import wait_for_participant_attribute
        await wait_for_participant_attribute(self.ctx.room, identity=self.participant.identity,
            attribute="sip.callStatus", value="active")

    async def _answered(self, observed_at):
        try:
            return await self.acknowledgements.report(lifecycle.report_call_answered(call=self.call, call_end_clock=self.call_end_clock,
                open_entry_window=self.navigation.open_entry_window, funded_runtime=self.funding,
                provider_call_id=self.provider_call_id, observed_at=observed_at),
                event_id=f"{self.call.call_id}:sip-answered", canonical_type="call.answered")
        finally:
            if self.call_end_clock["answered_at"] is not None:
                self.line_timeline.note_call("answered", self.call_end_clock["answered_at"])

    async def _start_authored_resources(self):
        try:
            from .workflow_source_owner import AuthoredSourceResourceBinding
            from .capabilities.workflow_sources import qualified_source_adapter
        except ImportError:
            from workflow_source_owner import AuthoredSourceResourceBinding
            from capabilities.workflow_sources import qualified_source_adapter
        declarations = {id(item): item for item in (self.plan.listener_adapter, self.plan.output_adapter,
            *(profile.adapter for profile in getattr(self.plan, "output_profiles", {}).values()))}
        selected = [item for item in declarations.values() if item.resource_factory is not None]
        if any(item.role != "authored" for item in selected):
            raise WorkflowInputRefused("non-authored workflow source cannot own an authored resource")
        if not selected:
            return
        run_id = self.call.config["workflow_run_id"]
        lanes = [("business", item) for item in selected]
        if self.navigation.enabled and self.plan.output_adapter.resource_factory is not None:
            lanes.append(("controller", self.plan.output_adapter))
        for lane, declaration in lanes:
            key = lane, id(declaration)
            if (key in self.authored_resources or qualified_source_adapter(
                    declaration.role, declaration.provider, declaration.model) is not declaration):
                raise WorkflowInputRefused("workflow source resource has no selected qualification")
            binding = AuthoredSourceResourceBinding(call_id=self.call.call_id, run_id=run_id,
                                                    declaration=declaration)
            self.authored_resources[key] = binding
            await binding.start()

    async def _prepare(self):
        from livekit import rtc
        from livekit.agents import Agent, NOT_GIVEN, room_io
        lifecycle.require_pre_media_participant(self.ctx.room, self.participant)
        self.detector = load_workflow_detector(self.plan.listener)
        if self.detector is None:
            raise lifecycle.ProductionConfigurationError("workflow listener has no local detector",
                reason="workflow_listener_detector_unavailable")
        await self._start_authored_resources()
        self.call.admit_media()
        self.call.workflow_source_observer = self.meter.observe
        self.call.workflow_turn_observer = self._observe_turn
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        worker_events.open_journal(self.call.call_id, self.evidence_dir)
        self.call.usage_report_outbox = UsageReportOutbox(
            self.state_path.parent / "usage-reports", self.call.call_id,
            alert=lambda reason: self.emit("usage_outbox_failed", reason=reason, call_id=self.call.call_id))
        self.recorder = SynchronizedEvidenceRecorder(self.evidence_dir,
            transcript_metadata={"skysay": {"call_id": self.call.call_id,
                **{key: str(self.call.config.get(key) or "") for key in ("agent_id", "organization_id", "project_id")}},
                "effective_voice_stack": effective_voice_stack_identity({**self.call.config,
                    "transcriber": self.plan.listener["constructor"], "model": {}, "system_prompt": ""})})
        self.transcript_events = CallTranscriptEventRecorder(relative_ms=self.recorder.relative_ms,
            engine="workflow", tools=self.call.config.get("tools"), emit=self.emit)
        self.recorder.set_transcript_events_source(self.transcript_events.section)
        self.arbiter = lifecycle.SessionEndArbiter(finish=self.ended.set, emit=self.emit,
            state_path=self.state_path, now=lifecycle.utcnow,
            prior_person_source=lambda: self.navigation.person_reached_state(
                unresolved=self.navigation_input is not None and self.navigation_input.unresolved)
                if self.navigation.enabled else None)
        self._listen("participant_disconnected", lifecycle.caller_disconnect_handler(
            caller_identity=self.participant.identity,
            finish=lambda: self.arbiter.end(lifecycle.TERMINATION_REMOTE_PARTY_DISCONNECTED,
                detected_via=lifecycle.DETECTED_VIA_DISCONNECT_EVENT)))
        room_end = lifecycle.production_room_disconnect_handler(finish=self.ended.set, emit=self.emit,
            stamp_lost=lambda: self.arbiter.end(lifecycle.TERMINATION_MEDIA_SESSION_LOST,
                detected_via=lifecycle.DETECTED_VIA_ROOM_LOST))
        def disconnected(*_):
            self.transport_disconnected = True
            room_end()
        self._listen("disconnected", disconnected)
        self.session = create_workflow_playback_session()
        self.session.on("agent_state_changed", self._agent_state)
        self.session.on("close", lambda _event: self._workflow_failed(WorkflowInputRefused("workflow output session ended"))
            if not self.ended.is_set() else None)
        agent = Agent(instructions="", tools=[], stt=None, llm=None, tts=None, vad=None)
        if self.plan.output_adapter.role == "live":
            # The live role's model rides THIS session: the provider socket
            # opens at start, before the inbound answer, as on the freeform
            # path. Its guard is bound before start; caller audio never is.
            self.live_output = open_live_workflow_output(self.plan.output_adapter,
                self.plan.output_constructor, self.managed_source_keys)
            self.live_output.bind(self.session,
                emit=lambda event, **details: self.emit(event, call_id=self.call.call_id, **details))
            self.session.on("error", self._live_error)
            agent = self.live_output.build_agent(language=self.plan.language)
        # The live role's duplex model clocks the caller's turns on pushed
        # audio (the SDK counts quiet from the frames it is handed), so its
        # session takes the room's input stream; the model never hears it,
        # because ``bind`` opens the duplex session with its input muted,
        # ahead of the first pushed frame. The first real call showed the
        # alternative: with no frames at all the model never takes a turn and
        # no ask is answered.
        await self.session.start(agent, room=self.ctx.room,
            room_options=room_io.RoomOptions(audio_input=self.live_output is not None, text_input=False,
                video_input=False, text_output=False, participant_identity=self.participant.identity,
                participant_kinds=linked_participant_kinds(self.browser_test,
                    agent_kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT, sdk_default=NOT_GIVEN),
                close_on_disconnect=False))
        if self.live_output is not None:
            self.live_output.attach()
        # Publishing an output track is necessary for an inbound SIP answer.
        # No spoken PCM exists yet. Only the live role subscribes to the
        # caller's audio here, and only it has a provider socket: its duplex
        # input was muted when the session opened, and it is unasked.
        self.stage = "inbound_answer"
        if self.outbound is None and self.browser_test is None:
            await lifecycle.require_inbound_sip_answer(self.ctx.room, self.participant,
                wait_for_active=self._wait_active, timeout_seconds=self.ringing_timeout_seconds)
        lifecycle.require_pre_media_participant(self.ctx.room, self.participant)
        self.started_at = lifecycle.utcnow()
        lifecycle.write_session_state(self.state_path, call_id=self.call.call_id,
            room_name=self.ctx.room.name, started_at=self.started_at, last_heartbeat_at=self.started_at,
            status="media_started")
        if self.outbound is None:
            await self._answered(self.started_at)
        self.stage = "media_start"
        media_ack = asyncio.to_thread(self.call.report_lifecycle,
            event_id=f"{self.call.call_id}:media-started", event_type="room.media_started",
            source_component="livekit_room", observed_at=self.started_at,
            details={"room_name": self.ctx.room.name})
        try:
            await self.acknowledgements.report(media_ack,
                event_id=f"{self.call.call_id}:media-started", canonical_type="media.started")
        finally:
            # A cancelled waiter still joins the HTTP ACK. A committed start
            # transfers settlement even when cancellation arrives with it.
            self.media_started = self.acknowledgements.media_started
            if self.media_started:
                self.line_timeline.note_call("media_started")
                if self.live_output is not None:
                    self.line_timeline.reply_timeout_seconds = sdk_reply_timeout_seconds()

    async def _serve(self):
        from livekit import rtc
        from livekit.agents.utils.participant import wait_for_track_publication
        if __package__:
            from .workflow_room_input import WorkflowRoomInput
        else:
            from workflow_room_input import WorkflowRoomInput
        self._spawn(self._heartbeat())
        self._spawn(self._transcript_loop())
        self._spawn(self._classify())
        maximum = lifecycle.resolved_max_call_seconds(self.call.config)[0]
        self._spawn(self.funding.watch(arbiter=self.arbiter, max_seconds=maximum,
            is_finishing=self.ended.is_set, emit=self.emit, duration_ending=self.navigation.duration_ending))
        self._spawn(self._watch_end(maximum))
        exclusion = callee_first_exclusion(managed_pstn=lifecycle.is_managed_pstn_call(self.call.config),
            speech_to_speech=False, enabled=self.callee_first_enabled,
            first_speaker=self.plan.opening["first_speaker"])
        joined_at = time.monotonic()
        if self.outbound is not None:
            self.opening_record = RootOpeningRecord(origin=joined_at, excluded_by=exclusion,
                                                    clock_anchor=clock_anchor)
            disposition = await lifecycle.watch_outbound_answer(await_answer=await_callee_answer,
                participant_attributes=self._attributes, wait_for_active=self._wait_active,
                timeout_seconds=self.ringing_timeout_seconds, start_opening_listen=self.opening_record.note_answer,
                answer_gate_evidence={}, joined_at_monotonic=joined_at, emit=self.emit,
                call_id=self.call.call_id, arm_answer_detection=self.answer_detection.arm,
                report_sip_answered=self._answered)
            if not may_speak(disposition):
                self.arbiter.end(lifecycle.TERMINATION_REMOTE_PARTY_DISCONNECTED,
                    detected_via=lifecycle.DETECTED_VIA_DISCONNECT_EVENT)
                return
        if self.ended.is_set():
            return
        self.input_released_at = time.monotonic()
        self.answer_detection.note_input_released()
        self.line_timeline.note_call("input_released", self.input_released_at)
        if self.opening_record is not None:
            self.opening_record.note_input_released(self.input_released_at)
        # Sidecar audio is evidence only. It never supplies caller recognition.
        self.sidecar, self.capture_tasks, self.evidence_identity = await self.start_sidecar(room_name=self.ctx.room.name,
            caller_identity=self.participant.identity, agent_identity=self.ctx.room.local_participant.identity,
            recorder=self.recorder, on_caller_frame=self._sidecar_caller_frame,
            on_media_frame=self._media_frame)
        publication = await asyncio.wait_for(wait_for_track_publication(self.ctx.room,
            identity=self.participant.identity, kind=rtc.TrackKind.KIND_AUDIO), timeout=3)
        lifecycle.require_pre_media_participant(self.ctx.room, self.participant)
        publication.set_subscribed(True)
        selected = await asyncio.wait_for(wait_for_track_publication(self.ctx.room,
            identity=self.participant.identity, kind=rtc.TrackKind.KIND_AUDIO, wait_for_subscription=True), timeout=3)
        if selected.sid != publication.sid or selected.track is None:
            raise WorkflowInputRefused("workflow microphone changed while subscribing")
        async def source_factory(*, first_sample, sample_rate):
            if type(first_sample) is not int or first_sample < 0 or sample_rate != rate:
                raise WorkflowInputRefused("workflow source has no current input format")
            self._listener_generation = getattr(self, "_listener_generation", 0) + 1
            generation = self._listener_generation
            return await WorkflowListenerSource.create(journal=self.call.workflow_source_journal(), policy=self.plan.listener,
                managed=self.managed_source_keys, attest=lambda built: self.emit(
                    "navigation_source_built", call_id=self.call.call_id, role="listener", generation=generation, **built))
        # The closures run only after run() starts, by which point both the
        # input's fence and the coordinator have been installed atomically.
        async def caller_input(observation, hint):
            if self.navigation_input is not None:
                self._navigation_holds(observation, hint)
                return
            if self.language_evidence is not None:
                self.language_evidence.observe(hint)
            if isinstance(observation, CallerInput):
                self._note_caller_final(observation)
            await self.pickup.on_input(observation, hint)
            self._project_caller(observation, observation.text if isinstance(observation, CallerInput) else "")
        selection = self.plan.listener["selection"]
        rate = self.plan.listener["capture"]["sample_rate"]
        self.input = WorkflowRoomInput(room=self.ctx.room, participant=self.participant, track=selected.track,
            sample_rate=rate, input_epoch=1,
            prefix_padding_samples=rate * self.plan.listener["vad"]["prefix_padding_ms"] // 1000,
            detector=self.detector, source_factory=source_factory, silence_hold_ms=selection["silence_hold_ms"],
            interruption_min_duration_ms=selection["interruption_min_duration_ms"],
            on_speech=lambda: self.pickup.on_speech(), on_interruption=lambda: self.pickup.on_interruption(),
            on_input=caller_input, on_clock=self._clock, on_frame=self._caller_frame, on_failure=self._input_failed,
            listener_reconnect=self.listener_reconnect)
        self.coordinator = WorkflowCall(fence=self.input.fence,
            advance=self._advance, output=self._output,
            on_terminal=lambda complete: self._spawn(self._finish_workflow(complete)), on_failure=self._workflow_failed,
            transfer=self._transfer,
            action_status=lambda action_id: _thread(self.call.action_command, "action-status", {"action_id": action_id}),
            action_release=lambda action_id, request_id: _thread(self.call.action_command, "workflow-action-release",
                {"action_id": action_id, "request_id": request_id}),
            action_cancel=lambda action_id: _thread(self.call.action_command, "workflow-action-cancel", {"action_id": action_id}),
            capture_barrier=self.input.barrier)
        self.pickup = WorkflowPickup(self.coordinator, silence_wait_seconds=self.plan.opening["silence_wait_seconds"],
            observer=None if self.opening_record is None else PickupEvidence(self.opening_record,
                sample_time=self._sample_time))
        if self.navigation.enabled:
            await self._start_navigation()
        if self.outbound is None or exclusion is not None:
            self.pickup.agent_first()
        # The caller's listener, stopped at a bridge: an error from its
        # cleanup then is workflow machinery, not a reason to hang up.
        self._input_task = self._spawn(self.input.run(), failed=self._input_failed)
        self._spawn(self._pickup_stall())
        await self.ended.wait()
        if self.failure is not None:
            raise self.failure

    async def _advance(self, wire):
        if self.ended.is_set():
            raise WorkflowMediaEnded("ended workflow has no fresh business action")
        turn, self._turn_timing = self.line_timeline.turn_sent(), None
        reply = await _thread(self.call.advance_workflow_turn, wire)
        # The observer ran on the request thread before it returned.
        self.line_timeline.turn_returned(turn, timing=self._turn_timing)
        if self.output_language is not None and reply["status"] != "running":
            self.output_language.mark_terminal()
        return reply

    async def _transfer(self, plan):
        """Place a workflow transfer step's one dial; return how it ended.

        The free-form ``transfer_to_human`` primitive: a ``TransferController``
        over the same room operations (``place_transfer_leg`` through the trunk
        that holds the call's workspace number, the server subscription
        change, the leg's removal), saying nothing itself. The controller puts
        the outcome on the call's transfer attempt row by the same report the
        tool makes, before the coordinator submits it as the run's next turn,
        and reports ``caller_hangup`` for a dial the call's end stopped.
        """
        try:
            from .transfer import HUMAN_IDENTITY, AuthoredLinesRoomMedia, TransferController
            from .transfer_dial import place_transfer_leg, remove_transfer_leg, subscribe_agent_audio
        except ImportError:
            from transfer import HUMAN_IDENTITY, AuthoredLinesRoomMedia, TransferController
            from transfer_dial import place_transfer_leg, remove_transfer_leg, subscribe_agent_audio
        def emit(event, **details):
            self.emit(event, call_id=self.call.call_id, **details)
        if self.navigation_actions is not None:
            # As the free-form tool does: no key is sent once a second SIP
            # participant may join the room.
            async with self.navigation_actions.line.transfer():
                pass
        # Observed from before the dial: the SIP answer and the room's
        # participant events travel separate paths, so the person may join
        # the room after the dial reports the answer.
        def joined(participant):
            if getattr(participant, "identity", None) == HUMAN_IDENTITY:
                self._note_transfer_party_joined()
        def left(participant):
            if getattr(participant, "identity", None) != HUMAN_IDENTITY:
                return
            # Kept even before the answer arrives: RTC may deliver join and
            # leave ahead of the dial's own answer response.
            self._transfer_party_left = True
            if self.transfer_bridged:
                self._end_for_transfer_party(lifecycle.DETECTED_VIA_DISCONNECT_EVENT)
        self._listen("participant_connected", joined)
        self._listen("participant_disconnected", left)
        media = AuthoredLinesRoomMedia(self.ctx.room, place_transfer_leg,
            functools.partial(subscribe_agent_audio, self.ctx.room, self.participant.identity),
            functools.partial(remove_transfer_leg, self.ctx.room),
            is_finished=self.ended.is_set, on_connected=self._bridge, emit=emit)
        answered_at = self.call_end_clock["answered_at"]
        maximum = lifecycle.resolved_max_call_seconds(self.call.config)[0]
        if answered_at is not None and maximum:
            # The transferred leg never outlives this call's own limit, and a
            # call already at it rings nobody.
            remaining = int(maximum - (time.monotonic() - answered_at))
            if remaining < 1:
                emit("workflow_transfer_outcome", outcome="failed", reason="call_limit_reached")
                await lifecycle._report_transfer_outcome(self.call, "failed", emit)
                return "failed"
            plan = {**plan, "max_call_seconds": remaining}
        # The one permission to place this dial: the attempt moves to
        # ``claimed`` exactly once, so a second consumer of the same waiting
        # turn (or a replay of it) never rings anyone.
        claim = await _thread(self.call.claim_transfer_dial)
        if not isinstance(claim, dict) or claim.get("ok") is not True:
            emit("workflow_transfer_claim_refused", reason_code=str((claim or {}).get("reason_code") or ""))
            raise WorkflowInputRefused("workflow transfer dial was not claimable")
        emit("workflow_transfer_dialing", mode=str(plan.get("mode") or ""), ring_seconds=plan.get("ring_seconds"))

        async def report(outcome):
            emit("workflow_transfer_outcome", outcome=outcome)
            await lifecycle._report_transfer_outcome(self.call, outcome, emit)

        controller = self._transfer_controller = TransferController(
            media, report=report, ended=self.ended.wait, emit=emit)
        try:
            return await controller.run(plan)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Reported as ``failed`` with its leg removed: the run takes its fallback.
            emit("workflow_transfer_error", error=type(exc).__name__)
            return "failed"

    def _bridge(self):
        """The person answered and the two legs hear each other: the call is theirs.

        The caller's listener stops (no recognition, transcript or listener
        source for a conversation between two people), no-caller-speech and
        answer detection no longer end it, and a failure of the workflow's
        machinery no longer hangs up. The call ends when the caller or the
        transferred person leaves (handlers installed before the dial), on an
        operator hangup, on funding or at the maximum duration.
        """
        try:
            from .transfer import HUMAN_IDENTITY
        except ImportError:
            from transfer import HUMAN_IDENTITY
        self.transfer_bridged = True
        if self._input_task is not None and not self._input_task.done():
            self._input_task.cancel()
        present = HUMAN_IDENTITY in self.ctx.room.remote_participants
        if present:
            self._note_transfer_party_joined()
        self.emit("workflow_transfer_bridged", call_id=self.call.call_id)
        if self._transfer_party_left and not present:
            # Joined and already gone before the answer reached this worker.
            self._end_for_transfer_party(lifecycle.DETECTED_VIA_DISCONNECT_EVENT)
            return
        self._spawn(self._await_transfer_party())

    def _end_for_transfer_party(self, detected_via):
        """The transferred person is gone: end the call once the run has
        recorded its transfer (bounded), so the run never reads incomplete."""
        if self._transfer_party_ending or self.ended.is_set():
            return
        self._transfer_party_ending = True
        self._spawn(self._end_after_transfer_outcome(detected_via))

    async def _end_after_transfer_outcome(self, detected_via):
        deadline = time.monotonic() + TRANSFER_OUTCOME_COMMIT_SECONDS
        while (not self.ended.is_set() and not self.coordinator.reply_terminal
               and time.monotonic() < deadline):
            await asyncio.sleep(0.05)
        if not self.ended.is_set():
            self.arbiter.end(lifecycle.TERMINATION_REMOTE_PARTY_DISCONNECTED, detected_via=detected_via)

    def _note_transfer_party_joined(self):
        if not self._transfer_party_joined.is_set():
            self._transfer_party_joined_at = time.monotonic()
            self._transfer_party_joined.set()

    def _transfer_party_roster(self, parties, *, asked_at):
        """The server's roster backs up a missed departure of the transferred person.

        Only a roster ASKED FOR after the person was seen joining counts: a
        response to a request already in flight at the join can be the
        server's earlier snapshot, without them.
        """
        try:
            from .transfer import HUMAN_IDENTITY
        except ImportError:
            from transfer import HUMAN_IDENTITY
        joined_at = self._transfer_party_joined_at
        if (self.transfer_bridged and joined_at is not None and asked_at > joined_at and HUMAN_IDENTITY not in parties
                and not self._transfer_party_ending and not self.ended.is_set()):
            self.emit("workflow_transfer_party_absent", call_id=self.call.call_id)
            self._end_for_transfer_party(lifecycle.DETECTED_VIA_OCCUPANCY_PROBE)

    async def _await_transfer_party(self):
        """A person who answered and never reaches the room leaves the caller alone."""
        try:
            await asyncio.wait_for(self._transfer_party_joined.wait(), timeout=TRANSFER_PARTY_JOIN_SECONDS)
        except TimeoutError:
            if not self.ended.is_set():
                self.emit("workflow_transfer_party_never_joined", call_id=self.call.call_id)
                self._end_for_transfer_party(lifecycle.DETECTED_VIA_DISCONNECT_EVENT)

    async def _pause_navigation_output(self):
        if self.ended.is_set():
            raise WorkflowInputRefused("ended workflow has no controller floor")
        generation = await self.coordinator.pause_output()
        if self.ended.is_set():
            raise WorkflowInputRefused("workflow ended while its controller floor joined")
        return generation

    def _project_caller(self, observation, text):
        if isinstance(observation, CallerInput) and text:
            self.caller_has_spoken = True
            self.telemetry.observe_user(text=text, metrics=None)
            self.answer_detection.note_transcript(text, source=WORD_SOURCE_STT_FINAL, at=time.monotonic())
            self.recorder.add_text("caller", text,
                at=self.input_released_at + observation.first_sample / self.input.fence.sample_rate,
                measured_duration=(observation.end_sample - observation.first_sample) / self.input.fence.sample_rate,
                language=self.output_language.active_language if self.output_language is not None else self.plan.language)
            self._transcript(text, self.participant.identity)

    def _set_navigation_output(self, enabled):
        if not enabled:
            self._held_hints.clear()
            self._held_arrivals.clear()
            if self.language_evidence is not None:
                self.language_evidence.observe(None)
                self.output_language.clear_pending()
            self._navigation_pause = self._spawn(self._pause_navigation_output())
        elif self.navigation.state == "conversation" and not self._settling_handoff:
            self._spawn(self._resume_navigation_output())

    async def _resume_navigation_output(self):
        if self._navigation_pause is not None:
            if self.navigation_audio is not None:
                await self.navigation_audio.join_current()
            generation = await self._navigation_pause
            if not await self.coordinator.resume_output(generation):
                raise WorkflowInputRefused("navigation lost its acknowledged business floor")
            self._navigation_pause = None

    async def _start_navigation(self):
        if __package__:
            from .navigation_entry import Redactor, FarEndWatch, NavigationTimeline, build_navigation_actions, sip_identities_of, listeners_of, WorkflowSettling, start_controller_track, navigation_menu_language_trusted, navigation_untrusted_sources
            from .capabilities.registry import load_registry
        else:
            from navigation_entry import Redactor, FarEndWatch, NavigationTimeline, build_navigation_actions, sip_identities_of, listeners_of, WorkflowSettling, start_controller_track, navigation_menu_language_trusted, navigation_untrusted_sources
            from capabilities.registry import load_registry
        classifier = self.navigation_classifier_factory()
        constructor = (self.plan.listener or {}).get("constructor") or {}
        listener_source = (("stt", str(constructor.get("provider") or ""), str(constructor.get("model") or "")),)
        registry = load_registry()
        watch = self.navigation_watch = FarEndWatch(policy=self.navigation, classifier=classifier,
            external_unresolved=lambda: self.navigation_input is not None and self.navigation_input.unresolved,
            label_trusted=lambda language: not navigation_untrusted_sources(registry, listener_source, (language,)))
        self.navigation.bind_conversation_activity(
            lambda: str(self.session.agent_state) == "speaking", pending=lambda: watch.unresolved)
        heard = self.navigation_languages()
        self.value_redactor = Redactor(self.call.call_id, default_language=self.plan.language,
            call_languages=tuple(sorted(set(heard) | set(self.plan.listener.get("required_languages") or ()))))
        self.recorder.set_redactor(self.value_redactor)
        timeline = NavigationTimeline(self.transcript_events, redactor=self.value_redactor)
        def fetch_value(name):
            value = self.call.fetch_keypad_value(name)
            audio.bind_value_field(name)
            return value
        async def publish(code, digit):
            revision = actions.executor.line_revision
            await self._pause_navigation_output()
            if actions.executor.line_revision != revision or actions.line.refusal() is not None:
                raise WorkflowInputRefused("controller line changed while business output joined")
            await self.ctx.room.local_participant.publish_dtmf(code=code, digit=digit)
        actions = self.navigation_actions = build_navigation_actions(policy=self.navigation,
            redactor=self.value_redactor, timeline=timeline, participant=self.participant,
            sip_identities=lambda: sip_identities_of(self.ctx.room),
            listeners=lambda: listeners_of(self.ctx.room, known={self.evidence_identity}),
            publish=publish, session=lambda: self.session, set_output_enabled=self._set_navigation_output,
            fetch_value=fetch_value, config=self.call.config, speaks_text=True,
            end_call=lambda reason, detail: self.arbiter.end(reason,
                detected_via=lifecycle.DETECTED_VIA_WORKER_TIMER, reason_detail=detail),
            is_finished=self.ended.is_set, settle_labels=watch.settled_through_now,
            model_feed=None, engine_generate=None, call_language=lambda: self.plan.language,
            session_hears=lambda _language: True,
            # The frozen listener is the call's only source: a menu in a language it was measured to fail in is not answered.
            menu_language_heard=lambda language: navigation_menu_language_trusted(registry, listener_source, language, heard))
        self.navigation_track = await start_controller_track(self.ctx.room)
        audio = self.navigation_audio = WorkflowControllerAudio(call=self.call, manifest=self.call.config.get("workflow_controller_manifest", {}),
            declaration=self.plan.output_adapter, constructor=self.plan.output_constructor,
            managed=self.managed_source_keys, episode=lambda: self.navigation.snapshot().episode,
            media_epoch=lambda: self.coordinator._output_epoch, before_source=self._pause_navigation_output,
            resource=self.authored_resources.get(("controller", id(self.plan.output_adapter))))
        def offered_options(language):
            prompt = actions.executor.current_prompt()
            if prompt is None or prompt.language != language:
                return ()
            return tuple(" ".join(option.label.split()) for option in prompt.options if option.label)
        audio.attach_player(self.navigation_track, valid=lambda: not self.ended.is_set()
            and self.navigation.in_episode and actions.line.room_refusal() is None,
            context=lambda: actions.executor.line_revision,
            fixed_valid=lambda: self.navigation.state in {"probing", "settling"}, offered_options=offered_options)
        actions.controller_speaker.attach_voice(audio.synthesize,
            covers=lambda language: language in audio.manifest.get("lines", {}),
            silence=audio.silence, covers_value=lambda language: all(
                "digit-" + str(digit) in audio.manifest.get("lines", {}).get(language, {}) for digit in range(10)))
        actions.controller_speaker.attach_player(audio.composed_player, flush=lambda player: player.clear())
        on_person = self._navigation_releases
        self.navigation_input = actions.workflow_listener_input(watch=watch, classifier=classifier,
            fence=self.input.fence, on_person=on_person, retire=self.pickup.retire_held_input,
            speaker=audio, language=lambda: self.plan.language, names=lambda: {},
            source_failed=lambda: self.failure is not None, on_failure=self._failed,
            project=self._project_caller, join_actions=audio.join_current,
            # The listener is the call's only source: a label in a language it was measured to fail in only observes.
            label_trusted=lambda language: not navigation_untrusted_sources(registry, listener_source, (language,)))
        self.coordinator.bind_input_owner(self.navigation_input)
        async def submit(text):
            originals = self.navigation_input.take_retained(text)
            await on_person(originals)
            self._settling_handoff = False
            await self._resume_navigation_output()
        self.navigation_settling = WorkflowSettling(gate=self.navigation_input.gate, speaker=audio,
            language=lambda: self.plan.language, names=lambda: {},
            note_agent_spoke=self.navigation_input.pickup.note_agent_spoke,
            note_agent_text=watch.note_agent_text, submit=submit,
            floor=lambda: self.navigation_input.wait_for_far_end(is_finishing=self.ended.is_set))
        actions.engine_handover = self.navigation_settling.begin
        self._listen("participant_connected", actions.on_participant_change)
        self._listen("participant_disconnected", actions.on_participant_change)
        actions.start()

    def _output(self, reply, epoch):
        declaration = self.plan.output_adapter
        constructor, language = self.plan.output_constructor, self.plan.language
        if self.output_language is not None:
            language = self.output_language.choose(reply["request_id"], requested_language=reply.get("requested_language"),
                                                   terminal=reply["status"] != "running")
            profile = self.plan.output_profiles[language]
            declaration, constructor = profile.adapter, profile.constructor
        # Install the official sink only when governed speech acquires it.
        # A callee-first controller can hold an empty business floor before
        # this point; no approved business PCM exists to drain or replay.
        # Authored speech plays through it on the room's output: a caller
        # sound holds the line there while its words are recognized.
        sink = getattr(getattr(self.session, "output", None), "audio", None)
        if (self.navigation.enabled or declaration.role == "authored" and sink is not None) and self.audio_floor is None:
            self.audio_floor = create_workflow_audio_floor(self.session.output.audio)
            self.session.output.audio = self.audio_floor
            if self.navigation.enabled:
                self.navigation_actions.probe_speaker.playback.bind(self.navigation_input.pickup.note_agent_speaking)
        self.terminal_reply = reply["status"] != "running"
        self.terminal_parts = []
        if self.opening_record is not None:
            self.opening_record.note_dispatched(time.monotonic())
        turn = self.line_timeline.current_turn()
        if declaration.role == "live":
            # No segmentation: the call's one session speaks each published
            # line whole, one journaled ask each.
            return _ClassifiedOutput(self, LiveReplyOutput(live=self.live_output, call=self.call, reply=reply,
                language=language, project=self._project_live, media_epoch=epoch,
                declaration=declaration, constructor=constructor, managed=self.managed_source_keys,
                remaining_source_budget=self.meter.remaining_source_budget,
                floor_deadline=(self.call_end_clock["answered_at"] or time.monotonic())
                    + lifecycle.resolved_max_call_seconds(self.call.config)[0],
                observe_line=lambda stages, status: self.line_timeline.line(turn, stages, status=status,
                                                                            role="live")))
        if declaration.role != "authored":
            raise WorkflowInputRefused("workflow output role has no speech path")
        marks = {}  # the reply's first segment's output stages, for its line record
        def preparation(_gate, journal):
            return AuthoredTtsPreparation(declaration=declaration,
                constructor=constructor, journal=journal, managed=self.managed_source_keys,
                resource=self.authored_resources.get(("business", id(declaration))),
                observe=lambda stage, at: marks.setdefault(stage, at))
        output = WorkflowReplyOutput(call=self.call, reply=reply, session=self.session, media_epoch=epoch,
            language=language, sample_rate=24000,
            segmenter=lambda text: split_source_line(text, declaration), preparation=preparation,
            project=self._project, remaining_source_budget=self.meter.remaining_source_budget,
            audio_floor=self.audio_floor,
            floor_deadline=(self.call_end_clock["answered_at"] or time.monotonic())
                + lifecycle.resolved_max_call_seconds(self.call.config)[0])
        def report(started_at, status):
            # A segmented reply journals per segment; its record is the reply's.
            playout = next((mark for mark in self._speaking_marks if mark >= started_at), None)
            self.line_timeline.line(turn, {**marks, "playout_started": playout, "line_finished": time.monotonic()},
                                    status=status, role="authored")
        return _ClassifiedOutput(self, output, report)

    async def _project(self, segment):
        completed = segment.delivery
        self.last_playout_start = self.playout_starts.pop(completed.speech_id, None)
        if self.last_playout_start is not None:
            self.recorder.bind_preplayout_agent_text(completed.speech_id, completed.text, at=self.last_playout_start)
            self.recorder.complete_preplayout_agent_speech(completed.speech_id, interrupted=False)
        if self.terminal_reply:
            self.terminal_parts.append(completed)
        self.answer_detection.note_agent_text(completed.text, at=completed.scheduled_at)
        self._transcript(completed.text, self.ctx.room.local_participant.identity)

    async def _project_live(self, line):
        # The configured line is the agent's recorded turn; the spoken wording
        # is the model's own. The event says which, without the text.
        self.last_playout_start = self.playout_starts.pop(line.speech_id, None)
        for continuation in line.speech_ids:
            self.playout_starts.pop(continuation, None)
        if self.last_playout_start is not None:
            self.recorder.bind_preplayout_agent_text(line.speech_id, line.text, at=self.last_playout_start)
            self.recorder.complete_preplayout_agent_speech(line.speech_id, interrupted=False)
        if self.terminal_reply:
            self.terminal_parts.append(line)
        self.emit("workflow_live_reply_projected", call_id=self.call.call_id, source=line.source,
                  steps=len(line.step_ids), language=line.language)
        self.answer_detection.note_agent_text(line.text, at=line.scheduled_at)
        self._transcript(line.text, self.ctx.room.local_participant.identity)

    def _live_error(self, event):
        error = getattr(event, "error", None)
        if getattr(error, "recoverable", True) is False and not self.ended.is_set():
            self._workflow_failed(WorkflowInputRefused("workflow live session failed"))

    def _agent_state(self, event):
        if event.new_state != "speaking":
            return
        now = time.monotonic()
        if len(self._speaking_marks) < self.plan.source_limit:
            self._speaking_marks.append(now)
        if self.opening_record is not None:
            self.opening_record.note_first_audio(now)
        handle = self.session.current_speech
        if handle is not None and handle.id not in self.playout_starts and len(self.playout_starts) < self.plan.source_limit:
            self.playout_starts[handle.id] = time.monotonic()

    async def _finish_workflow(self, completed):
        verdict = {"signal": "unobserved", "confirmed": False}
        try:
            if completed and self.terminal_parts:
                def observer_alive():
                    last = self.recorder.observer_last_frame_at()
                    return (self.sidecar is not None and self.sidecar.isconnected() and last is not None
                        and time.monotonic() - last <= lifecycle.TERMINAL_EGRESS_OBSERVER_STALE_SECONDS)
                verdict = await lifecycle.drain_terminal_egress(
                    agent_audible_until=self.recorder.agent_audible_until,
                    room_disconnected=lambda: self.transport_disconnected or self.ended.is_set(),
                    goodbye_playout_start=self.last_playout_start, observer_alive=observer_alive)
        except asyncio.CancelledError:
            verdict = {"signal": "drain_cancelled", "confirmed": False}
            raise
        except Exception as exc:
            verdict = {"signal": "drain_timeout" if isinstance(exc, TimeoutError) else "drain_failed", "confirmed": False}
            self.emit("workflow_terminal_evidence_failed", error=type(exc).__name__, call_id=self.call.call_id)
        finally:
            try:
                if self.terminal_parts:
                    self.recorder.add_terminal_message("".join(part.text for part in self.terminal_parts),
                        interrupted=not completed, speech_id=self.terminal_parts[-1].speech_id,
                        egress_confirmed=verdict.get("confirmed") is True, egress_signal=str(verdict.get("signal") or "unobserved"))
            except Exception as exc:
                self.emit("workflow_terminal_evidence_failed", error=type(exc).__name__, call_id=self.call.call_id)
            finally:
                # A transferred run ends the workflow, not the call: the two
                # bridged people end it (``_bridge``).
                if not self.ended.is_set() and not self.transfer_bridged:
                    if (self.coordinator is not None
                            and self.coordinator.terminal_fail_reason == UNHEARD_REPEATS_EXHAUSTED):
                        # The run closed because the caller kept speaking over
                        # a line they never heard: not the agent finishing.
                        self._end_line_undelivered("repeat_limit")
                    else:
                        self.arbiter.end(lifecycle.TERMINATION_AGENT_COMPLETED,
                                         detected_via=lifecycle.DETECTED_VIA_AGENT_SCRIPT)

    def _clock(self, tick):
        try:
            self.caller_vad_edges.note_user_state("speaking" if tick.physical_speech else "listening",
                                                  at=time.monotonic())
        except Exception:  # noqa: BLE001 - a diagnostic cannot break the detector clock
            pass
        self.answer_detection.note_activity(tick.physical_speech)
        if self.navigation_input is not None:
            self.navigation_input.note_clock(tick)
        if tick.physical_speech:
            self.caller_has_spoken = True
        if (not self.navigation.suspends(ENDER_NO_CALLER_SPEECH)
                and not (self.navigation_input is not None and self.navigation_input.unresolved)):
            self.pickup.on_clock(tick)

    def _caller_frame(self):
        """One caller frame reached the capture: the flow log's recognition seam."""
        now = time.monotonic()
        if self._first_frame_at is None:
            self._first_frame_at = now
        self.caller_audio_flow.note_caller_frame(now)

    def _note_caller_final(self, observation, at=None):
        """A recognised caller turn reached business: its stages onto the line ledger."""
        timing = observation.timing
        self.line_timeline.note_caller_final(
            speech_started=self._sample_time(observation.turn_first_sample),
            speech_ended=self._sample_time(observation.end_sample), at=at, timing=timing,
            voice_ended=self._sample_time(getattr(timing, "voiced_end_sample", None)))

    def _navigation_holds(self, observation, hint):
        """Navigation owns the caller's input: it holds this turn, with its hint and arrival."""
        if isinstance(observation, CallerInput):
            # When it reached the root is its delivery, whenever business gets it.
            if len(self._held_arrivals) >= MAX_CALL_ATTEMPTS:
                self._held_arrivals.pop(next(iter(self._held_arrivals)))
            self._held_arrivals[(observation.input_epoch, observation.turn_first_sample, observation.first_sample,
                                 observation.end_sample)] = time.monotonic()
        self.navigation_input.submit(observation)
        if self.language_evidence is not None and isinstance(observation, CallerInput):
            key = (observation.input_epoch, observation.turn_first_sample, observation.first_sample, observation.end_sample)
            if key in self._held_hints:
                self._held_hints[key] = None
            elif self._last_held_hint_key is None or key > self._last_held_hint_key:
                self._last_held_hint_key = key
                if len(self._held_hints) >= MAX_CALL_ATTEMPTS:
                    self._held_hints.pop(next(iter(self._held_hints)))
                self._held_hints[key] = hint if isinstance(hint, CallerLanguageHint) else None

    async def _navigation_releases(self, observations):
        """Navigation hands its held turns to business, in order: each is noted as it arrived."""
        if self.language_evidence is not None:
            for observation in observations:
                key = (observation.input_epoch, observation.turn_first_sample, observation.first_sample, observation.end_sample)
                self.language_evidence.observe(self._held_hints.pop(key, None))
        for observation in observations:
            if isinstance(observation, CallerInput):
                key = (observation.input_epoch, observation.turn_first_sample, observation.first_sample,
                       observation.end_sample)
                # An arrival evicted past the bound is noted at its release, the later instant.
                self._note_caller_final(observation, at=self._held_arrivals.pop(key, None))
        await self.pickup.release_held_inputs(self.navigation_input, observations)

    def _sample_time(self, sample):
        """A capture sample index on the monotonic clock, from the first frame's arrival."""
        if self._first_frame_at is None or type(sample) is not int or sample < 0:
            return None
        return self._first_frame_at + sample / self.plan.listener["capture"]["sample_rate"]

    def _sidecar_caller_frame(self, pcm):
        """The sidecar's caller track: the classifier first, then energy while starved."""
        self.answer_detection.note_frame(pcm)
        try:
            if self.caller_audio_flow.wants_energy:
                measured = frame_amplitude(pcm)
                if measured is not None:
                    self.caller_audio_flow.note_track_energy(measured)
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass

    def _observe_turn(self, **details):
        self._turn_timing = details.get("timing")
        return self.telemetry.observe_workflow_turn(**details)

    def _emit_call_evidence(self):
        """The per-call summaries the freeform path emits, once, at teardown."""
        if self._evidence_emitted or not self.media_started:
            return
        self._evidence_emitted = True
        try:
            self.caller_audio_flow.poll()
            self.emit("caller_audio_flow_summary", call_id=self.call.call_id, **self.caller_audio_flow.snapshot())
            self.emit("caller_vad_edge_summary", call_id=self.call.call_id,
                      **self.caller_vad_edges.snapshot(at=time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - evidence cannot break teardown
            self.emit("workflow_evidence_failed", error=type(exc).__name__, call_id=self.call.call_id)
        if self.opening_record is not None:
            self.opening_record.emit(self.emit, OPENING_SUMMARY_EVENT, call_id=self.call.call_id)

    def _media_frame(self, role, pcm, placement, received_at):
        media_start, discontinuous = placement
        self.answer_detection.note_media_frame(role, pcm, media_start=float(media_start),
            received_at=float(received_at), gap=bool(discontinuous))

    async def _classify(self):
        while not self.ended.is_set():
            await asyncio.sleep(CLASSIFY_INTERVAL_SECONDS)
            try:
                self.caller_audio_flow.poll()
            except Exception:  # noqa: BLE001 - a diagnostic cannot break classification
                pass
            decision = self.answer_detection.tick(caller_has_spoken=self.caller_has_spoken,
                navigation_suspended=self.navigation.suspends(ENDER_ARBITER))
            self.answer_detection.process_voicemail()
            if self.answer_detection.arbiter.silence_the_agent and self.coordinator is not None:
                await self.coordinator.on_interruption()
            if decision is not None and decision.acts and not self.transfer_bridged:
                self.arbiter.end(decision.reason, detected_via=lifecycle.DETECTED_VIA_WORKER_TIMER)

    def _transcript(self, text, identity):
        if self.browser_test is None:
            return
        try:
            self.transcripts.put_nowait((text, identity))
        except asyncio.QueueFull:
            self.emit("workflow_transcript_dropped", call_id=self.call.call_id)

    async def _transcript_loop(self):
        while True:
            text, identity = await self.transcripts.get()
            try:
                writer = await self.ctx.room.local_participant.stream_text(topic="lk.transcription",
                    sender_identity=identity, destination_identities=[self.participant.identity],
                    attributes={"lk.segment_id": uuid4().hex, "lk.transcription_final": "true"})
                try:
                    await writer.write(text)
                finally:
                    await writer.aclose()
            except Exception as exc:
                self.emit("workflow_transcript_failed", error=type(exc).__name__, call_id=self.call.call_id)

    async def _pickup_stall(self):
        while not self.ended.is_set():
            if (not self.navigation.suspends(ENDER_NO_CALLER_SPEECH)
                    and not (self.navigation_input is not None and self.navigation_input.unresolved)):
                self.pickup.check_stall()
            await asyncio.sleep(.05)

    async def _heartbeat(self):
        sequence = 0
        while not self.ended.is_set():
            await asyncio.sleep(5)
            sequence += 1
            try:
                response = await _thread(self.call.report_lifecycle,
                    event_id=f"{self.call.call_id}:heartbeat:{sequence}", event_type="room.media_heartbeat",
                    source_component="livekit_room", observed_at=lifecycle.utcnow(),
                    details={"room_name": self.ctx.room.name}, funding_generation=self.funding.request_generation())
            except Exception as exc:
                self.funding.poll_failed()
                self.emit("call_heartbeat_failed", call_id=self.call.call_id, error=type(exc).__name__)
                continue
            self.funding.adopt(response)
            if (lifecycle.adopt_hangup_intent(self.hangup_intent, lifecycle.pending_hangup_intent(response),
                    call_id=self.call.call_id, emit=self.emit) and not self.ended.is_set()):
                # At receipt, as the freeform heartbeat does: the watchdog is
                # 15 s away and a farewell in flight would stamp its own ending.
                # An ending that already owns teardown is not relabelled.
                self.emit("hangup_intent_received", call_id=self.call.call_id, attempt=self.hangup_intent["attempt"])
                self.arbiter.end(lifecycle.TERMINATION_HANGUP_REQUESTED,
                    detected_via=lifecycle.DETECTED_VIA_HANGUP_INTENT)
            lifecycle.write_session_state(self.state_path, call_id=self.call.call_id,
                last_heartbeat_at=lifecycle.utcnow(), status="media_started")

    async def _watch_end(self, maximum):
        from livekit import api
        async def occupancy():
            asked_at = time.monotonic()
            parties = await lifecycle.server_side_remaining_parties(self.ctx.api.room,
                room_name=self.ctx.room.name, request_factory=api.ListParticipantsRequest,
                agent_identity=self.ctx.room.local_participant.identity,
                evidence_identity=self.evidence_identity)
            self._transfer_party_roster(parties, asked_at=asked_at)
            return len(parties)
        def clock():
            answered = self.call_end_clock["answered_at"]
            return lifecycle.CallEndSnapshot(answered=answered is not None,
                seconds_since_answer=0 if answered is None else time.monotonic() - answered,
                # A bridged caller is talking to a person: not this call's silence to end.
                caller_has_spoken=self.caller_has_spoken or self.transfer_bridged,
                hangup_requested=bool(self.hangup_intent),
                seconds_since_input_released=None if self.input_released_at is None
                    else time.monotonic() - self.input_released_at)
        await lifecycle.watch_for_worker_call_end(arbiter=self.arbiter, clock=clock, occupancy=occupancy,
            max_call_seconds=maximum, no_caller_speech_seconds=lifecycle.resolved_no_caller_speech_seconds(self.call.config)[0],
            emit=self.emit, is_finishing=self.ended.is_set, navigation=self.navigation)

    async def _join_media(self):
        # Before any join: a transfer still ringing must not be answered for
        # a call that is ending. The coordinator stops waiting first, so no
        # outcome turn is submitted for it; the controller then removes the
        # leg and reports ``caller_hangup``.
        if self.coordinator is not None:
            self.coordinator.cancel_dial()
        if self._transfer_controller is not None:
            await self._transfer_controller.cancel_and_join("caller_hangup")
        for resource in self.authored_resources.values():
            resource.retire()
        self.ended.set()
        self._emit_call_evidence()
        self._held_hints.clear()
        self._held_arrivals.clear()
        if self.output_language is not None:
            self.output_language.mark_terminal()
        # Stop input before business/playback. Its source finally blocks join
        # all provider children and durable facts before the whole-call meter.
        for task in self.tasks:
            if not task.done():
                task.cancel()
        errors = []
        for owner in (self.input, self.navigation_input,
                      self.navigation_watch if self.navigation_input is None else None, self.navigation_settling,
                      self.navigation_actions, self.navigation_audio, self.navigation_track, self.coordinator, self.pickup):
            if owner is not None:
                try:
                    await owner.aclose()
                except BaseException as exc:
                    errors.append(exc)
        if self.tasks:
            for result in await asyncio.gather(*self.tasks, return_exceptions=True):
                # Counted only: an owned line refusal is settled quietly
                # without any distinct cleanup failure (``run``).
                if (isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
                        and result is not self.failure):
                    self.cleanup_failures += 1
        # Every owner that can begin or seal a source has joined above. What
        # follows closes resources, the live session and the room session,
        # none of which owns a source fact, so their failure cannot hold a
        # report of sealed sources; the server still refuses one
        # with an unsealed source, and the report fences new grants.
        source_errors = len(errors)
        for resource in self.authored_resources.values():
            try:
                await resource.aclose()
            except BaseException as exc:
                errors.append(exc)
        if self.value_redactor is not None:
            self.value_redactor.close()
        if self.live_output is not None:
            # Session first, then its borrowed transport, before the room
            # session closes: every reply has been joined by the coordinator.
            # Each line's source fact was journaled when its reply joined.
            try:
                await self.live_output.aclose()
            except BaseException as exc:
                errors.append(exc)
        if self.session is not None:
            try:
                result = await lifecycle.close_production_session_after_room_end(self.session,
                    room_transport_disconnected=lambda: self.transport_disconnected)
                if result.required and not result.completed:
                    errors.append(WorkflowInputRefused("workflow output session did not close"))
            except BaseException as exc:
                errors.append(exc)
        for name, callback in self.handlers:
            try:
                self.ctx.room.off(name, callback)
            except Exception as exc:
                self.emit("workflow_handler_cleanup_failed", error=type(exc).__name__, call_id=self.call.call_id)
        self.handlers.clear()
        if self.transcript_events is not None:
            try:
                self.transcript_events.close()
            except Exception as exc:
                self.emit("workflow_transcript_cleanup_failed", error=type(exc).__name__, call_id=self.call.call_id)
        self.sources_joined = not source_errors
        self.cleanup_failures += len(errors)
        if errors:
            self.emit("workflow_cleanup_failed", error=type(errors[0]).__name__, call_id=self.call.call_id)
        if self.media_started:
            # Every reply has joined: a call that spoke no line still owes its record.
            self.line_timeline.close(role=self.plan.output_adapter.role)

    async def _settle_media(self):
        from livekit import api
        ended_at = self.arbiter.decided_at or lifecycle.utcnow()
        self.answer_detection.commit(self.arbiter.reason)
        try:
            await lifecycle.teardown_room_and_acknowledge_hangup(call=self.call,
                room_service=self.ctx.api.room, room_name=self.ctx.room.name,
                request_factory=api.DeleteRoomRequest, intent=self.hangup_intent,
                emit=self.emit, session_cleanup=False)
        except Exception as exc:
            self.emit("call_room_cleanup_failed", error=type(exc).__name__, call_id=self.call.call_id)
        if self.sidecar is not None:
            try:
                await lifecycle.shutdown_evidence_sidecar(self.sidecar, self.capture_tasks,
                    on_late_completion=lambda *_: None)
            except Exception as exc:
                self.emit("workflow_sidecar_cleanup_failed", error=type(exc).__name__, call_id=self.call.call_id)
        report = self.meter.report() if self.sources_joined else None
        if report is not None:
            try:
                await _thread(self.call.report_component_usage, report)
            except Exception as exc:
                # The shared call session retains this immutable report for
                # terminal fallback and outbox replay. Settlement still runs.
                self.emit("call_usage_report_failed", error=type(exc).__name__, call_id=self.call.call_id)
        terminal = await lifecycle.report_call_teardown_lifecycle(call=self.call, call_id=self.call.call_id,
            ended_at=ended_at, emit=self.emit, termination_detail=self.arbiter.reason,
            ended_reason_detail=self.arbiter.reason_detail, prior_person=self.arbiter.prior_person,
            answer_classification=self.answer_detection.terminal_details())
        summary = None
        try:
            self.recorder.set_voicemail_detection(self.answer_detection.voicemail_section(self.arbiter.reason))
            summary = (await finalize_off_loop(self.recorder)).summary
        except Exception as exc:
            self.emit("workflow_recording_finalize_failed", error=type(exc).__name__, call_id=self.call.call_id)
        try:
            self.telemetry.write(self.evidence_dir / "call-telemetry.json")
        except Exception as exc:
            self.emit("call_telemetry_write_failed", error_type=type(exc).__name__, call_id=self.call.call_id)
        try:
            if terminal:
                await lifecycle.finalize_call_artifacts(call=self.call, call_id=self.call.call_id,
                    evidence_dir=self.evidence_dir, state_path=self.state_path, summary=summary,
                    room_name=self.ctx.room.name, started_at=self.started_at, ended_at=ended_at,
                    telemetry_turns=self.telemetry.bundle()["turn_count"], emit=self.emit,
                    participant_identity=self.participant.identity)
            else:
                lifecycle.write_session_state(self.state_path, call_id=self.call.call_id,
                    last_heartbeat_at=ended_at, status="lifecycle_pending")
        finally:
            self.call.cleanup(remove_participant=lambda: None, delete_room=lambda: None)

    async def _settle_unresolved_start(self):
        # No wallet verdict can follow a start whose ACK is unknown. Hang up
        # without lifecycle settlement; the existing reconciler owns the
        # authoritative fixed-id event and this nonterminal session record.
        for cleanup in (self.remove_participant, self.ctx.delete_room):
            try:
                await cleanup()
            except Exception as exc:
                self.emit("workflow_unresolved_start_cleanup_failed", error=type(exc).__name__, call_id=self.call.call_id)
        if self.sidecar is not None:
            await lifecycle.shutdown_evidence_sidecar(self.sidecar, self.capture_tasks,
                on_late_completion=lambda *_: None)
        lifecycle.write_session_state(self.state_path, call_id=self.call.call_id,
            room_name=self.ctx.room.name, started_at=self.started_at,
            last_heartbeat_at=lifecycle.utcnow(), status="lifecycle_ack_pending")

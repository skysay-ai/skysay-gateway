# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""A call's own structured worker events, kept beside its evidence.

The worker's ``SCH236_PRODUCTION`` lines go to the container's stdout, and
Docker deletes that log with the container. A release recreates every voice
role, and on 2026-09-24 one took a campaign's worker logs minutes before they
were read. The per-call evidence directory survives recreation (it is a host
bind mount), so each call's own events are also appended to
``<evidence_dir>/worker-events.jsonl``.

What lands there is decided by a REVIEWED FIELD REGISTRY, never by what a
value looks like: a code-shaped string can still be a name ("AliceSmith"), and
a short integer can still be a local number. So:

* a field is kept only when its key is in ``FIELDS`` (with the validator for
  that key) or is a measurement named by one of ``NUMBER_SUFFIXES``;
* a reviewed string field keeps only a value from its own closed vocabulary
  (``UNLISTED`` otherwise), an exception class name, a locale, or a value from
  the shipped audio-environment pack -- never free text;
* a number is kept only under a measurement key, a boolean only under a
  reviewed flag, and nothing nested is kept but a reviewed list of codes;
* every other field is dropped with its value and counted in
  ``dropped_fields``, so an unreviewed key -- including every key of a spread
  ``**details`` nobody listed -- is how data is LOST, never how it leaks.

``call_id`` is the journal's own id, not the event's value. The file is
bounded (``MAX_FILE_BYTES`` including its one truncation marker,
``MAX_LINES``, ``MAX_LINE_BYTES``) and written by one background thread fed
through a bounded queue, so an emit never touches the disk on the media event
loop. The journal registry is separate from that lossy queue: an overload can
drop events, never a journal. A job drains its events before it returns
(``job_journals``), and the process drains at exit. Recording is best effort
and never raises into the caller: losing a forensic line must never cost a
call.
"""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import json
import math
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

try:
    from .capabilities.workflow_sources import DECLARED_SOURCE_ADAPTERS
    from .duplex_liveness import BACKEND_STATUSES, CLIENT_EVENT_TYPES
    from .soniox_usage import EVIDENCE_ERROR_KINDS, SONIOX_STT_CLOSE_STATUSES
    from .spoken_language import LANGUAGE_NAMES
    from .workflow_requests import TRANSFER_OUTCOMES
    from .workflow_root_evidence import LINE_STAGES, UNKNOWN_NOT_REPORTED
except ImportError:  # pragma: no cover - direct execution inside the image
    from capabilities.workflow_sources import DECLARED_SOURCE_ADAPTERS
    from duplex_liveness import BACKEND_STATUSES, CLIENT_EVENT_TYPES
    from soniox_usage import EVIDENCE_ERROR_KINDS, SONIOX_STT_CLOSE_STATUSES
    from spoken_language import LANGUAGE_NAMES
    from workflow_requests import TRANSFER_OUTCOMES
    from workflow_root_evidence import LINE_STAGES, UNKNOWN_NOT_REPORTED

FILE_NAME = "worker-events.jsonl"
MAX_FILE_BYTES = 256 * 1024
MAX_LINES = 2000
MAX_LINE_BYTES = 4096
TRUNCATED_EVENT = "worker_events_truncated"
REDACTED = "[redacted]"
#: A reviewed vocabulary field carried a value outside its vocabulary.
UNLISTED = "[unlisted]"
#: Room kept for the truncation marker, so the file never passes MAX_FILE_BYTES.
MARKER_RESERVE_BYTES = 256
MAX_ITEMS = 32
#: Journals kept open at once. A worker process runs a bounded number of
#: calls; a journal a job never closed is evicted oldest-first.
MAX_OPEN_JOURNALS = 256
QUEUE_SIZE = 8192
#: How long a finishing job waits for its events to reach the disk.
JOB_DRAIN_SECONDS = 2.0
EXIT_DRAIN_SECONDS = 1.0

_EVENT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REGION = re.compile(r"^[A-Z]{2}$")
#: The control plane's call ids (``call_`` plus 16 url-safe characters), with room.
_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
#: ``type(exc).__name__``. The suffix is what keeps a bare proper noun out.
_EXCEPTION_NAME = re.compile(
    r"^[A-Z][A-Za-z0-9]{0,63}(Error|Exception|Timeout|Cancelled|Exit|Interrupt|Failure|Refused|Unavailable)$"
)

#: A number is kept under a key naming what it measures.
NUMBER_SUFFIXES = ("_ms", "_seconds", "_count", "_tokens", "_characters", "_bytes", "_frames", "_total")
#: Integers at or above this are refused even as a measurement.
MAX_KEPT_NUMBER = 10**9


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return REDACTED
    if isinstance(value, float) and not math.isfinite(value):
        return REDACTED
    return value if abs(value) < MAX_KEPT_NUMBER else REDACTED


def _flag(value: Any) -> Any:
    return value if isinstance(value, bool) else REDACTED


def _counter(value: Any) -> Any:
    return value if type(value) is int and 0 <= value < MAX_KEPT_NUMBER else REDACTED


def _vocabulary(*values: str) -> Callable[[Any], Any]:
    allowed = frozenset(values)

    def validate(value: Any) -> Any:
        return value if isinstance(value, str) and value in allowed else UNLISTED

    validate.vocabulary = allowed  # type: ignore[attr-defined]
    return validate


def _language(value: Any) -> Any:
    """A language the worker can name (spoken_language.LANGUAGE_NAMES), with
    an optional two-letter region. A model's language-switch request is
    arbitrary text, so a pattern over tags is not enough."""
    if not isinstance(value, str):
        return REDACTED
    code, separator, region = value.partition("-")
    if code in LANGUAGE_NAMES and (not separator or _REGION.match(region)):
        return value
    return REDACTED


def _http_status(value: Any) -> Any:
    return value if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599 else REDACTED


def _pattern(pattern: re.Pattern[str]) -> Callable[[Any], Any]:
    def validate(value: Any) -> Any:
        return value if isinstance(value, str) and pattern.match(value) else REDACTED

    return validate


_GUARD_BASES = (
    "caller_words_pending", "caller_still_speaking", "agent_replies_cut", "machine", "budget_exhausted",
    "no_run", "caller_not_understood", "not_closing", "input_overflow", "closing", "wrong_person",
    "classifier_unavailable", "interrupted", "aborted", "finished", "ending_scheduled",
    "pending", "reclaimed", "committed",
)
_PROMPT_ID = re.compile(r"^(|p[0-9]{1,4})$")
_GUARD_ID = re.compile(r"^[A-Za-z0-9_.:-]{0,128}$")
_GUARD_WITHHELD = re.compile(r"^(|held_caller_words|call_ending|generation_cut|caller_busy|navigating|generate_reply_failed:[A-Za-z_][A-Za-z0-9_]{0,63})$")


def _counts_by(key: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """A small mapping of closed-vocabulary keys to counts."""

    def validate(value: Any) -> Any:
        if not isinstance(value, dict):
            return REDACTED
        out: dict[str, Any] = {}
        for name, count in list(value.items())[:MAX_ITEMS]:
            out[str(key(name))] = _number(count)
        return out

    return validate


def _by_language(shape: dict[str, Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """A small mapping of language keys to a fixed-shape record."""

    def validate(value: Any) -> Any:
        if not isinstance(value, dict):
            return REDACTED
        out: dict[str, Any] = {}
        for language, record in list(value.items())[:MAX_ITEMS]:
            name = _language(language) if language != "unknown" else "unknown"
            if not isinstance(record, dict):
                out[str(name)] = REDACTED
                continue
            out[str(name)] = {
                # A pin of None is a fact (rule 9 off for that language), not a value to redact.
                field: (None if record.get(field) is None else check(record.get(field)))
                for field, check in shape.items()
            }
        return out

    return validate


def _stage_reasons(value: Any) -> Any:
    """The workflow line ledger's owed-but-unreported stages: its stage names to a reason code."""
    if not isinstance(value, dict):
        return REDACTED
    stage, reason = _vocabulary(*LINE_STAGES), _vocabulary(UNKNOWN_NOT_REPORTED)
    return {str(stage(name)): reason(code) for name, code in list(value.items())[:MAX_ITEMS]}


def _codes(item: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def validate(value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return REDACTED
        return [item(entry) for entry in list(value)[:MAX_ITEMS]]

    validate.item = item  # type: ignore[attr-defined]
    return validate


#: ``SessionEndArbiter`` reasons (production.WORKER_TERMINATION_DETAILS) plus
#: the teardown's own fallback; a test pins the two sets together.
TERMINATION_CAUSES = _vocabulary(
    "hangup_requested",
    "remote_party_disconnected",
    "max_call_duration_reached",
    "no_caller_speech_after_answer",
    "no_human_after_answer",
    "agent_completed",
    "media_session_lost",
    "model_session_lost",
    "listener_unavailable",
    "workflow_line_undelivered",
    "caller_unresponsive",
    "agent_unresponsive",
    "browser_startup_failed",
    "call_end_before_answer",
    "funding_exhausted",
    "ivr_timeout",
)
#: sip_answer_gate's dispositions, plus the answer gate's own ``pending``.
ANSWER_DISPOSITIONS = _vocabulary(
    "answered", "already_active", "disconnected", "timed_out", "not_reported", "not_applicable", "pending",
    # The end_call guard's decision dispositions share the field name.
    "allowed", "refused", "would_refuse", "superseded", "aborted",
)
#: The audio-environment pack's presets (audio_environment._PRESET_IDS and
#: OFF_PRESET_ID); a test pins them together.
AUDIO_PRESETS = _vocabulary("off", "quiet-office", "call-centre", "cafe", "street", "television")

_SONIOX_EVIDENCE_KINDS = _vocabulary(*sorted(EVIDENCE_ERROR_KINDS))

FIELDS: dict[str, Callable[[Any], Any]] = {
    "queued": _flag,
    "mechanism": _vocabulary("append_thinking", "tool_result_and_silence_recovery"),
    "agent_speeches": _counter,
    "agent_speech_intervals": _counter,
    "agent_speeches_counted": _counter,
    "agent_speeches_with_text": _counter,
    "audio_without_text": _counter,
    "unlinked_speeches_without_text": _counter,
    "text_without_audio": _counter,
    "coverage": lambda value: None if value is None else _number(value),
    "threshold": _number,
    "alarm": _flag,
    "reasons": _codes(_vocabulary("coverage_below_threshold", "unlinked_audible_speech")),
    "action": _vocabulary("response_created", "caller_resumed", "released", "speaking_while_held", "replacement_requested", "replacement_after_refused_tool", "automatic_cut_for_replacement", "handover_retry", "handover_reply_completed", "handover_reply_failed", "handover_unmute_failed"),
    "first_reply_path": _vocabulary("native_pickup", "native_replacement", "workflow_pickup", "literal", "literal_fast", "pickup_reply", "generated"),
    "caller_speaking": _flag,
    "resumptions": _counter,
    "held_responses": _counter,
    "discarded": _counter,
    "pickup_failed": _flag,
    "frames_dropped_after_pickup": _counter,
    "observation_failure": _vocabulary("caller audio observation did not catch up"),
    "frames_delivered": _counter,
    "startup_frames_retained": _counter,
    "generation": _number,
    "ending_won": _flag,
    # Every qualifier a worker ending can carry (``production.
    # TERMINATION_REASON_DETAILS``, which a test pins this set to): the
    # funded-runtime and watchdog endings emit the very value they stamp.
    "reason_detail": _vocabulary(
        "", "insufficient_balance", "renewal_unavailable", "model_unresponsive",
        "hold_limit", "navigation_limit", "deadline", "menu_no_path", "data_not_authorised",
        "recorded_message_only", "navigation_detector_lost", "during_hold", "transcriber_refused_request",
    ),
    # The end_call guard (``end_call_guard``, ``end_call_guard_summary``,
    # ``end_call_guard_post_refusal``, ``agent_utterance``,
    # ``end_call_guard_mode_downgraded``): closed vocabularies, flags, ids
    # and counts only; no caller or agent text ever reaches these fields.
    "mode": _vocabulary("off", "shadow", "enforce"),
    # ``far_end_listening``: whether the session transcriber listens with the
    # open construction frozen at admission, or the profile's own locale.
    "listening": _vocabulary("open", "profile"),
    "requested": _vocabulary("off", "shadow", "enforce"),
    "basis": _vocabulary(*_GUARD_BASES),
    # The close floor (``end_call_floor_deferred``, ``end_call_floor_settled``,
    # ``end_call_local_refusal_recovered``): how the caller's floor read when
    # a close waited on it, or which reading refused it.
    "floor": _vocabulary("words", "wordless", "quiet", "vad", "navigation", ""),
    # The navigation state's transitions (the navigation policy module): the
    # states, and why the call moved. A test pins both sets to that module.
    "navigation_from": _vocabulary("conversation", "navigating", "holding", "probing", "settling"),
    "navigation_to": _vocabulary("conversation", "navigating", "holding", "probing", "settling"),
    "navigation_why": _vocabulary(
        "menu", "hold", "probe", "probe_done", "resume", "confirmed_human", "music",
        "settled", "settling_failed", "observer:ivr_phrase", "verdict:looping_recording",
        "navigation_unavailable",
        "label:menu_press", "label:menu_speech", "label:menu_either", "label:numeric_challenge",
        "label:queue_announcement", "label:queue_position", "label:callback_offer",
    ),
    "navigation_observation": _vocabulary("opened", "exhausted"),
    "navigation_window": _vocabulary("answer", "music_onset"),
    # Where confirmed human evidence came from, and why navigation was turned
    # off for the call (the classifier failed). Closed; no caller text.
    "navigation_person": _vocabulary("observer_confirmed", "classifier_person", "probe_confirmed"),
    "navigation_disabled": _vocabulary("navigation_unavailable"),
    # The far-end classifier's label for a final utterance (labels, languages and
    # counts only; the utterance text never reaches an event), why it was
    # unavailable, and a business tool held for the labels in flight.
    "far_end_label": _vocabulary(
        "person", "person_reply", "menu_press", "menu_speech", "menu_either", "numeric_challenge",
        "queue_announcement", "queue_position", "callback_offer", "voicemail", "recorded_info",
        "readback", "other", "unknown",
    ),
    "far_end_continues": _flag,
    "far_end_same_menu": _flag,
    # A menu piece held inside the open menu until its key arrives, and how it was settled (counts only).
    "head_outcome": _vocabulary("held", "folded", "replayed"),
    "head_revision": _counter,
    "head_pieces": _counter,
    # Which check refused a model's request as stale, and the prompt ids it was judged against (p<n>, or "").
    "stale_branch": _vocabulary("fed_mismatch", "unnamed_span", "named_mismatch", "not_after", "line_moved", "other"),
    "stale_open": _pattern(_PROMPT_ID),
    "stale_fed": _pattern(_PROMPT_ID),
    "far_end_options": _counter,
    "navigation_cause": _vocabulary("no_platform_key", "self_test_wrong_answer", "exception"),
    "navigation_tool": _vocabulary(
        "end_call", "set_call_disposition", "request_callback", "send_sms", "transfer_to_human"
    ),
    "navigation_settled": _flag,
    # Which opening-failure site a navigation stop was excused at.
    "navigation_site": _vocabulary(
        "native_not_dispatchable", "native_opening_reply", "pickup_opening_reply", "generated_opening_reply"
    ),
    "navigation_episode": _counter,
    "provenance": _vocabulary("matched", "unknown", ""),
    "verdict": _vocabulary("closing", "not_closing", "wrong_person", "machine", "unavailable", ""),
    "origin": _vocabulary("reply", "same_generation_as_end_call", "post_refusal_regeneration", "end_call_message", "unknown"),
    "skipped": _vocabulary("call_ending", "pair_not_in_context", ""),
    "regeneration_withheld": _pattern(_GUARD_WITHHELD),
    "regeneration_failed": _pattern(_EXCEPTION_NAME),
    "attempt_id": _pattern(_GUARD_ID),
    "caller_turn_id": _pattern(_GUARD_ID),
    "speech_id": _pattern(_GUARD_ID),
    "item_id": _pattern(_GUARD_ID),
    "attempt_ids": _codes(_pattern(_GUARD_ID)),
    # Recorder-local numeric turn indices captured by a close proposal, not
    # caller text, provider IDs, or contact identifiers. Bound the list.
    "proposal_turns": _codes(_counter),
    "language_active": _language,
    "inherited": _flag,
    "deferred": _flag,
    "user_speaking": _flag,
    "held_final": _flag,
    "silence_promoted": _flag,
    "allowed_then_caller_spoke": _flag,
    "classifier_armed": _flag,
    "applied": _flag,
    "regenerated": _flag,
    "carried_end_call": _flag,
    "farewell": _flag,
    "interrupted": _flag,
    "audio_heard": _flag,
    "user_initiated": _flag,
    "after_refusal": _flag,
    "anchor_revision": _number,
    "decision_revision": _number,
    "cut_replies_of_last_4": _number,
    "refused_turns": _number,
    "caller_turns": _number,
    "last_final_confidence": _number,
    "low_confidence_threshold": _number,
    "budget": _number,
    "attempts": _number,
    "premature_allows": _number,
    "aborted": _number,
    "classifier_calls": _number,
    "classifier_unavailable": _number,
    "classifier_skipped": _number,
    "post_refusal_planned": _number,
    "post_refusal_applied": _number,
    "post_refusal_failed": _number,
    "post_refusal_regeneration_failed": _number,
    "post_refusal_outstanding": _number,
    "post_refusal_unsettled": _number,
    "guard_refusals": _number,
    "dropped": _number,
    "line_index": _number,
    "chars": _number,
    "exchange_chars": _number,
    "created_at": _number,
    "done_at": _number,
    "refusals_by_basis": _counts_by(_vocabulary(*_GUARD_BASES)),
    "utterances_by_origin": _counts_by(_vocabulary("reply", "same_generation_as_end_call", "post_refusal_regeneration", "end_call_message", "unknown")),
    "farewells_by_origin": _counts_by(_vocabulary("reply", "same_generation_as_end_call", "post_refusal_regeneration", "end_call_message", "unknown")),
    "low_confidence_by_language": _by_language(
        {
            "threshold": _number,
            "provisional": _flag,
            "attempts_with_confidence": _number,
            "refused": _number,
            "min_confidence": _number,
        }
    ),
    "reason": _vocabulary(
        "unanswered_zero_usage",
        "mutual_silence_off",
        "disabled_by_environment",
        "session_error_budget",
        "background_track_unavailable",
        "no_phrases_for_language",
        "already_acknowledged",
        "acknowledgement_playing",
        "clip_not_ready",
        "far_end_floor_hold",
        "far_end_floor_hold_released",
        "far_end_floor_hold_released_no_words",
        "room_transport_disconnected",
        "gate_stood_down",
        "no_reply_in_flight",
        "callee_gone",
        "timeout",
        # runtime_language_switch
        "caller_explicit_request",
        "provider_stable_language_evidence",
        "terminal",
        "requested language is not in the frozen allowed profiles",
        # workflow_listener_refused and workflow_listener_reconnect:
        # workflow_source_owner.LISTENER_REFUSAL_REASONS; a test pins them together.
        "config",
        "frame",
        "end",
        "wall",
        "reclaimed",
        "no_ack",
        # workflow_line_undelivered: why the line was not delivered.
        "not_spoken",
        "repeat_limit",
    ),
    # workflow_listener_refused, workflow_listener_reconnect: the declared listener providers. Its own key,
    # because other events name providers from other vocabularies.
    "listener_provider": _vocabulary(
        *sorted({item.provider for item in DECLARED_SOURCE_ADAPTERS if item.role == "listener"})),
    # ...and the SDK's speech sources, for the guard's agent_utterance.
    "source": _vocabulary("explicit_request", "provider_stable_evidence", "say", "generate_reply", ""),
    "cause": TERMINATION_CAUSES,
    "disposition": ANSWER_DISPOSITIONS,
    "answer_gate_disposition": ANSWER_DISPOSITIONS,
    "closed_by": _vocabulary("call_end_before_answer", "call_end"),
    "status": _vocabulary("absent", "unsupported", "timeout", "failed", "closed"),
    # turn_end_evidence_summary: whether the per-call record landed, and that
    # the detector's own verdict was not observable.
    "turn_end_evidence_status": _vocabulary(
        "written", "write_failed", "summary_failed", "not_applicable"
    ),
    "detector_observation": _vocabulary("unavailable"),
    "phase": _vocabulary("disconnect", "captures"),
    "stage": _vocabulary(
        "session_connect",
        "dispatch_validation",
        "participant_wait",
        "adapter_configuration",
        "outbound_resume",
        "inbound_prepare",
        "component_configuration",
        "media_admission",
        "worker_entry",
        # caller_spoke_marker_failed: where the uncaptioned-speech rule broke.
        "speech_created",
        "speech_done",
        "segment_ended",
        "settle",
    ),
    "trigger": _vocabulary("track_published", "worker_entry_sweep"),
    "refusal_outcome": _vocabulary(
        "leg_ended",
        "leg_end_failed",
        "leg_already_ending",
        "live_call_left_alone",
        "no_inbound_caller",
        "outbound_marked",
        "outbound_mark_failed",
        "capacity_reserved_reported",
        "capacity_reserved_report_failed",
    ),
    "evidence_status": _vocabulary("complete", "partial"),
    "semantic_config_source": _vocabulary("realtime-config"),
    "semantic_evidence_status": _vocabulary("unavailable_no_room_audio"),
    "channel": _vocabulary("outgoing_background_track"),
    "preset_id": AUDIO_PRESETS,
    "asset_version": _vocabulary("v1"),
    "unmeasured_components": _codes(_vocabulary("stt", "llm", "tts")),
    "language": _language,
    "old_language": _language,
    "publication_version": _number,
    "status_code": _http_status,
    "error": _pattern(_EXCEPTION_NAME),
    "error_type": _pattern(_EXCEPTION_NAME),
    # transfer_error_after_call_end: where an expected post-hangup error landed.
    "transfer_step": _vocabulary("transfer", "cleanup"),
    # transfer_finished: how the transfer ended, and what stopped it before its bridge, if anything;
    # soniox_stream_close_incomplete: how the worker's recognizer close ended;
    # workflow_listener_reconnect: whether the call's one listener reconnect
    # delivered the turn ("recovered") or not ("failed").
    "outcome": _vocabulary(*sorted(TRANSFER_OUTCOMES | SONIOX_STT_CLOSE_STATUSES | {"recovered"})),
    # component_usage_unavailable: why native provider evidence is missing; none when it has no kind.
    "kind": lambda value: None if value is None else _SONIOX_EVIDENCE_KINDS(value),
    # soniox_stream_close_incomplete: the recognizer streams and provider sessions it covered.
    "streams": _counter,
    "sessions": _counter,
    "stopped_by": _vocabulary("caller_cancelled", "caller_hangup", "failed", ""),
    # transfer_cancel: what a withdrawal request was answered, and its evidence.
    "speech_feed": _vocabulary("user_state", "native_floor", ""),
    "decision": _vocabulary("cancelled", "no_caller_request", "not_in_progress", "already_connected"),
    "caller_spoke_since_placed": _flag,
    # transfer_leg_removal: how the removal of a stopped leg ended.
    "removal_result": _vocabulary("removed", "not_found_after_settle", "call_ended", "error"),
    "attempt": _number,
    "clips": _number,
    "phrases": _number,
    "words_heard": _number,
    "configured_words": _number,
    "max_concurrent_jobs": _number,
    # Phone admission: the host's capacity, and a refusal's account.
    "phone_capacity": _number,
    "capacity": _number,
    "held": _number,
    "refused_total": _number,
    # The refused dispatch's LiveKit job id (the same shape as a call id), and
    # what the refusal found in the room -- never the room's name, which for
    # an inbound call carries the caller's number.
    "job_id": _pattern(_CALL_ID),
    "room_kind": _vocabulary(
        "inbound_ringing", "occupied", "no_inbound_caller", "unknown", "outbound", "live_call_test"
    ),
    "recorded": _flag,
    "max_amplitude": _number,
    "interruptions": _number,
    "count": _number,
    # caller_speech_without_caption: counts only, of caller voice-activity
    # segments and of the uncaptioned-speech moments written from them.
    "segments_recorded": _number,
    "segments_over_agent": _number,
    "segments_withheld": _number,
    "segments_withheld_then_heard": _number,
    "segments_dropped_at_cap": _number,
    "speeches_untracked": _number,
    "marked": _number,
    "kept": _number,
    "marked_over_agent": _number,
    "by_words": _flag,
    "completed": _flag,
    "cancellation_resistant": _flag,
    "lifecycle_recorded": _flag,
    "participant_terminated": _flag,
    "room_terminated": _flag,
    "max_call_clamped": _flag,
    "no_caller_speech_clamped": _flag,
    "has_more": _flag,
    "installed": _flag,
    # model_unresponsive_decision: the rule's own flags, the caller's
    # continuity numbers and the duplex witness's counts, flags and
    # allowlisted types (``duplex_liveness``). The *_ms and *_seconds fields
    # are kept by their suffix.
    "armed": _flag,
    "model_response_seen": _flag,
    "provider_error_seen": _flag,
    "caller_runs_since_armed": _number,
    "caller_longest_run_seconds_since_armed": _number,
    "caller_speech_share_since_armed": _number,
    "caller_speech_seconds_call": _number,
    "caller_longest_run_seconds_call": _number,
    "duplex_appends_sent_last_5s": _number,
    "duplex_output_events_last_5s": _number,
    "duplex_unsent_appends": _number,
    "duplex_reconnects": _number,
    "duplex_frames_withheld_by_hold": _number,
    "duplex_delegations_created": _number,
    "duplex_delegation_open": _flag,
    "duplex_backend_failed": _number,
    "duplex_backend_end_call_items": _number,
    "duplex_backend_tool_choice": _vocabulary("unknown", "auto", "none", "required", "function"),
    "duplex_backend_responses_tools_disabled": _number,
    "duplex_backend_last_status": _vocabulary("", *BACKEND_STATUSES),
    "duplex_context_usage_ratio": _number,
    "duplex_last_client_event": _vocabulary("", "other", *CLIENT_EVENT_TYPES),
    "duplex_input_heard_since_output": _flag,
    # tts_meter_settled: the native TTS meter's unit (its counts end in _count).
    "unit": _vocabulary("character", "utf8_byte", "token"),
    # workflow_live_line_audio and workflow_live_line_not_voiced: a GPT-Live
    # line's audio witness (``None`` where a frame in its window was unreadable).
    "terminal": _flag,
    "step_index": _counter,
    "audible_seconds": lambda value: None if value is None else _number(value),
    "configured_weight": _counter,
    "letters_per_second": lambda value: None if value is None else _number(value),
    "short_audio": _flag,
    "audio_enforced": _flag,
    # workflow_first_line_timeline / workflow_line_timeline: a stage that did not
    # happen, or that nothing reported, is a null, never a redaction.
    **{f"{stage}_ms": (lambda value: None if value is None else _number(value)) for stage in LINE_STAGES},
    "unknown_stages": _stage_reasons,
}

#: Emitted keys that were reviewed and deliberately NOT kept: customer-authored
#: names, per-module import errors, or lists with no closed vocabulary. The
#: census test fails on an emitted key in neither this set nor ``FIELDS``.
EXCLUDED_FIELDS = frozenset({"tools", "components", "failed", "missing", "track_sids"})


def validator_for(key: str) -> Callable[[Any], Any] | None:
    validate = FIELDS.get(key)
    if validate is not None:
        return validate
    if _KEY.match(key) and key.endswith(NUMBER_SUFFIXES):
        return _number
    return None


def fields(details: dict[str, Any]) -> dict[str, Any]:
    """The reviewed fields of ``details``; everything else is only counted."""
    kept: dict[str, Any] = {}
    dropped = 0
    for key, value in details.items():
        if key in ("event", "call_id"):
            continue
        validate = validator_for(key) if isinstance(key, str) and key not in EXCLUDED_FIELDS else None
        if validate is None:
            dropped += 1
            continue
        try:
            kept[key] = validate(value)
        except Exception:  # noqa: BLE001 - an odd value must not cost the line
            kept[key] = REDACTED
    if dropped:
        kept["dropped_fields"] = dropped
    return kept


def _encode(event: str, call_id: str, details: dict[str, Any]) -> bytes:
    name = event if isinstance(event, str) and _EVENT.match(event) else REDACTED
    # Bounded like everything else, so no line can outgrow MAX_LINE_BYTES.
    call_id = call_id if _CALL_ID.match(call_id) else REDACTED
    record = {"event": name, "call_id": call_id, **fields(details)}
    encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_LINE_BYTES:
        encoded = (
            json.dumps({"event": name, "call_id": call_id, "oversize_bytes": len(encoded)}, sort_keys=True) + "\n"
        ).encode("utf-8")
    return encoded


class CallEventJournal:
    """Bounded ``worker-events.jsonl`` for one call. Written only by the writer thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._loaded = False
        self._bytes = 0
        self._lines = 0
        self._truncated = False

    def _load(self) -> None:
        # A reopened call continues the SAME budget: the file on disk is the
        # authority, not a fresh counter.
        self._loaded = True
        try:
            data = self.path.read_bytes()
        except OSError:
            return
        self._bytes = len(data)
        self._lines = data.count(b"\n")
        self._truncated = TRUNCATED_EVENT.encode() in data

    def write(self, encoded: bytes) -> None:
        if not self._loaded:
            self._load()
        if self._truncated:
            return
        if (
            self._bytes + len(encoded) + MARKER_RESERVE_BYTES > MAX_FILE_BYTES
            or self._lines + 2 > MAX_LINES
        ):
            marker = {
                "event": TRUNCATED_EVENT,
                "has_more": True,
                "lines_written": self._lines,
                "bytes_written": self._bytes,
            }
            self._append((json.dumps(marker, sort_keys=True) + "\n").encode("utf-8"))
            self._truncated = True
            return
        self._append(encoded)

    def _append(self, encoded: bytes) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(encoded)
            self._bytes += len(encoded)
            self._lines += 1
        except OSError:
            pass


class _Writer:
    """One daemon thread per process; the only code that touches the files.

    It carries writes and flush requests, never the journal lifecycle: an item
    names the journal object it belongs to, so dropping one under overload
    loses that line and nothing after it.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[CallEventJournal | None, Any]] = queue.Queue(maxsize=QUEUE_SIZE)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pid = 0
        self.dropped = 0

    def _ensure_thread(self) -> None:
        # A forked job process inherits the object but not the thread.
        if self._thread is not None and self._thread.is_alive() and self._pid == os.getpid():
            return
        self._pid = os.getpid()
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread = threading.Thread(target=self._run, name="worker-events", daemon=True)
        self._thread.start()

    def submit(self, journal: CallEventJournal | None, payload: Any) -> bool:
        with self._lock:
            self._ensure_thread()
        try:
            self._queue.put_nowait((journal, payload))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            journal, payload = self._queue.get()
            try:
                if journal is None:
                    payload.set()
                else:
                    journal.write(payload)
            except Exception:  # noqa: BLE001 - the writer must outlive any one bad line
                pass
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything submitted so far is on disk.

        Never called on the event loop, so it may wait for room in a full
        queue: a flush is exactly the moment the queue is likely to be full.
        """
        deadline = time.monotonic() + timeout
        done = threading.Event()
        with self._lock:
            self._ensure_thread()
        try:
            self._queue.put((None, done), timeout=timeout)
        except queue.Full:
            return False
        return done.wait(max(0.0, deadline - time.monotonic()))


_writer = _Writer()
_journals: OrderedDict[str, CallEventJournal] = OrderedDict()
_journals_lock = threading.Lock()
_job_calls: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("worker_events_job_calls", default=None)


def open_journal(call_id: str, evidence_dir: Path) -> None:
    """Start keeping ``call_id``'s events in ``evidence_dir``. No disk I/O."""
    path = Path(evidence_dir) / FILE_NAME
    with _journals_lock:
        journal = _journals.pop(call_id, None)
        if journal is None or journal.path != path:
            journal = CallEventJournal(path)
        _journals[call_id] = journal
        while len(_journals) > MAX_OPEN_JOURNALS:
            _journals.popitem(last=False)
    calls = _job_calls.get()
    if calls is not None:
        calls.append(call_id)


def close_journal(call_id: str) -> None:
    """Stop keeping ``call_id``'s events. Lines already recorded are still written."""
    with _journals_lock:
        _journals.pop(call_id, None)


def flush(timeout: float = 5.0) -> bool:
    return _writer.flush(timeout)


@contextlib.asynccontextmanager
async def job_journals(timeout: float = JOB_DRAIN_SECONDS) -> AsyncIterator[None]:
    """Close every journal this job opened and drain them, however it exits.

    A job process can end without running interpreter exit hooks, and the
    writer is a daemon thread, so its last lines are on disk only if the job
    waits for them. The wait runs off the event loop and is bounded.
    """
    calls: list[str] = []
    token = _job_calls.set(calls)
    unwinding = False
    try:
        yield
    except BaseException:
        unwinding = True
        raise
    finally:
        _job_calls.reset(token)
        for call_id in calls:
            close_journal(call_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        drain = asyncio.ensure_future(asyncio.to_thread(flush, timeout))
        # Retrieved whenever it finishes, even after the deadline: a drain
        # that failed must not surface later through the loop's handler.
        drain.add_done_callback(_consume_outcome)
        # A cancellation does not end the wait: the process may be about to
        # stop, which is exactly when the lines must already be on disk. It
        # is surfaced once the drain finishes or its deadline passes -- unless
        # the job is already unwinding with its own exception, which wins.
        interrupted: asyncio.CancelledError | None = None
        while not drain.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({drain}, timeout=remaining)
            except asyncio.CancelledError as cancelled:
                interrupted = cancelled
        if interrupted is not None and not unwinding:
            raise interrupted


def _consume_outcome(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


def record(event: str, details: dict[str, Any]) -> None:
    """Keep ``event`` if it belongs to a call whose journal is open. Never raises."""
    try:
        call_id = details.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return
        with _journals_lock:
            journal = _journals.get(call_id)
        if journal is None:
            return
        _writer.submit(journal, _encode(event, call_id, details))
    except Exception:  # noqa: BLE001 - forensics must never cost a call
        pass


atexit.register(flush, EXIT_DRAIN_SECONDS)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The typed tool-call, language-switch and caller-spoke rows of one recording.

This is the PRODUCER half of
``contracts/voice/call_transcript_events_section.json``. The consumer half is
the control plane's ``call_transcript_event_evidence`` service, which parses
what this module writes into ``transcript.json``'s ``transcript_events`` key
at artifact import. The two sides never import each other -- this worker
package imports nothing from the control plane -- so
``contracts/voice/call_transcript_events_section.json`` is the contract, and
the worker test suite asserts every closed enum and required-key set in this
file against that file.

Two rules shape every line here, and both are stricter than "bounded".

*Typed fields only.* No tool argument, no tool result, no exception text and
no configured tool NAME is written. A configured name is customer-chosen text,
so a row carries the closed ``tool_kind`` plus the opaque server-issued
``tool_ref``, and the display name is resolved at read time from the call's own
configuration. The SDK hands a callback the model-facing function name, not the
spec, so ``tool_index`` maps name -> (kind, ref) once at session start; a name
that is not in that map still produces a row, with ``tool_ref: ""`` and the
kind alone, because the row is real evidence and the drawer has a label for
its kind.

*A row is never worth a call.* Every hook runs on the media loop from
``session.on`` and swallows everything into a named worker log event. The
section is assembled from bounded integers and closed literals only, and
``section()`` answers with a ``malformed`` header rather than raising, so a
defect here costs the evidence and nothing else.

Unlike ``transcript_events.py`` -- the browser test panel's live-only sink,
which summarises arguments and results for a page nobody stores -- nothing
here is published anywhere and nothing here reads a value the privacy contract
forbids. The two are deliberately separate sinks over the same SDK events.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: The exact literal the importer demands. A different value is an unreadable
#: section, not a tolerated variant, so adding a field is a new literal here
#: and in the schema file rather than a silent extension.
SCHEMA = "skysay.call-transcript-events.v1"

EVENT_TOOL_CALL = "tool_call"
EVENT_LANGUAGE_SWITCH = "language_switch"
#: The caller spoke, the engine answered, and no caption exists for it: a
#: moment with a time and no text.
EVENT_CALLER_SPOKE_NO_TRANSCRIPT = "caller_spoke_no_transcript"

#: Phone-menu and hold moments of an outbound call whose agent navigates
#: (``contracts/voice/call_transcript_events_section.json``, written here and
#: validated by the importer's ``call_transcript_nav_events``).
EVENT_NAV_MENU = "nav_menu"
EVENT_DTMF_SENT = "dtmf_sent"
EVENT_DTMF_UNSUPPORTED = "dtmf_unsupported"
EVENT_NAV_HOLD = "nav_hold"
EVENT_NAV_PERSON = "nav_person"
NAV_EVENT_TYPES = (EVENT_NAV_MENU, EVENT_DTMF_SENT, EVENT_DTMF_UNSUPPORTED, EVENT_NAV_HOLD, EVENT_NAV_PERSON)

#: Per spec 2 and the schema: a language switch sorts before a tool call at an
#: equal ``at_ms``, and so does the start of the caller's speech, which a tool
#: call is only ever observed after.
SOURCE_ORDER_BY_TYPE = {
    EVENT_LANGUAGE_SWITCH: 0,
    EVENT_CALLER_SPOKE_NO_TRANSCRIPT: 0,
    EVENT_TOOL_CALL: 1,
    # `seq` orders the phone-menu moments among themselves (the contract's const 0).
    **{name: 0 for name in NAV_EVENT_TYPES},
}

MAX_EVENTS = 200
MAX_NAV_TEXT_CHARS = 300
MAX_NAV_OPTIONS = 12
MAX_NAV_LABEL_CHARS = 40
MAX_NAV_KEYS = 32
MAX_NAV_LOOPS = 1000
NAV_HOLD_SOUNDS = ("", "music", "recording", "silence")
_NAV_KEY = re.compile(r"[0-9*#]")
_NAV_KEYS = re.compile(r"[0-9*#]{1,32}")
#: A call variable's name, the shape the importer demands of a value row.
_NAV_FIELD = re.compile(r"[a-z][a-z0-9_]{0,63}")
#: In-flight bookkeeping per call id, drained by the batch that reports those
#: calls. Deliberately larger than the row cap: terminal entries can arrive in
#: a different order than the batch lists its calls, so a map bounded AT the
#: row cap could drop the status of a call whose row does ship -- and a lost
#: cancellation reads as a success. Bounded all the same, because a stream of
#: started events with no batch behind them must not grow without end.
MAX_IN_FLIGHT_CALLS = 2 * MAX_EVENTS
MAX_AT_MS = 24 * 60 * 60 * 1_000
MAX_DURATION_MS = 600_000
MAX_TOOL_REF_CHARS = 64
MAX_LANGUAGE_TAG_CHARS = 35

TOOL_KINDS = (
    "end_call",
    "transfer_to_human",
    "send_sms",
    "knowledge_lookup",
    "webhook",
    "connection",
    "mcp",
    "other",
)
ERROR_CLASSES = ("", "cancelled", "unknown")
LANGUAGE_SWITCH_REASONS = ("follow_caller", "explicit")
ENGINES = ("", "cascade", "gpt_live", "workflow")
STATUSES = ("complete", "partial", "none_recorded", "unavailable", "malformed")

ENGINE_WORKFLOW = "workflow"
ENGINE_GPT_LIVE = "gpt_live"

REASON_ENGINE_GPT_LIVE = "engine_gpt_live"
REASON_ENGINE_WORKFLOW = "engine_workflow"
REASON_NO_FROZEN_LANGUAGE_CONTRACT = "no_frozen_language_contract"

#: The call-scoped retrieval tool is declared in the worker's own session
#: interface rather than in the agent's ``tools`` array, so it carries no
#: server-issued ref and would otherwise fall to ``other``. Its closed kind
#: exists; name it here so the drawer labels it as retrieval.
KNOWLEDGE_TOOL_NAME = "search_knowledge_base"

#: The approved drawer renders an explicit switch as ONE language row, so a
#: SUCCESSFUL ``switch_language`` execution does not also produce a tool row.
#: One that failed or was cancelled produces no language row either, so it
#: keeps its tool row rather than vanishing. The row is therefore RECORDED and
#: held provisionally suppressed, never dropped at the batch: on the deferred
#: path the SDK delivers the terminal entry after the batch, and a row dropped
#: there could not be restored when the cancellation finally arrived.
SUPPRESSED_TOOLS = frozenset({"switch_language"})

#: How a switch's ``source`` (the language switcher's own emit detail) maps
#: onto the closed reason. An unrecognised source is follow-caller, which is
#: what every non-explicit path is.
REASON_BY_SOURCE = {
    "explicit_request": "explicit",
    "provider_stable_evidence": "follow_caller",
}

_TOOL_CALL_KEYS = (
    "type",
    "seq",
    "at_ms",
    "source_order",
    "tool_kind",
    "tool_ref",
    "ok",
    "error_class",
    "duration_ms",
)
_LANGUAGE_SWITCH_KEYS = ("type", "seq", "at_ms", "source_order", "from", "to", "reason")
_CALLER_SPOKE_NO_TRANSCRIPT_KEYS = ("type", "seq", "at_ms", "source_order")
_NAV_COMMON = ("type", "seq", "at_ms", "source_order")
_KEYS_BY_TYPE = {
    EVENT_NAV_MENU: (*_NAV_COMMON, "text", "options"),
    EVENT_DTMF_SENT: (*_NAV_COMMON, "mode", "keys", "label", "field"),
    EVENT_DTMF_UNSUPPORTED: _NAV_COMMON,
    EVENT_NAV_HOLD: (*_NAV_COMMON, "end_ms", "sound", "loops"),
    EVENT_NAV_PERSON: (*_NAV_COMMON, "confirmed"),
    EVENT_TOOL_CALL: _TOOL_CALL_KEYS,
    EVENT_LANGUAGE_SWITCH: _LANGUAGE_SWITCH_KEYS,
    EVENT_CALLER_SPOKE_NO_TRANSCRIPT: _CALLER_SPOKE_NO_TRANSCRIPT_KEYS,
}


def section_engine(config: Mapping[str, Any] | None, *, workflow: bool = False) -> str:
    """The engine literal for this call, or ``""`` when the contract has none.

    ``workflow`` wins: a workflow-bound call's steps belong to the flow run,
    and the importer turns that literal into a status-only header whatever the
    worker wrote. Otherwise the call's own declared ``voice_frontend`` is
    stamped when the shared contract's closed enum names it. An engine the
    contract does not name is stamped ``""`` rather than guessed into
    ``cascade``: "" asks the importer for no engine-specific filtering, and the
    language half of a new voice-to-voice engine is already answered by the
    absence of a frozen follow-caller contract on such a call (the control
    plane refuses the language axis for an engine whose declaration has no
    language axis), so a wrong guess here could only mislabel, never
    over-report.
    """
    if workflow:
        return ENGINE_WORKFLOW
    data = config if isinstance(config, Mapping) else {}
    nested = data.get("voice") if isinstance(data.get("voice"), Mapping) else {}
    declared = str(data.get("voice_frontend") or nested.get("voice_frontend") or "").strip().lower()
    if not declared:
        # Omitted means cascade everywhere in the control plane; the worker
        # must not turn that default into "unknown engine".
        return "cascade"
    return declared if declared in ENGINES and declared != ENGINE_WORKFLOW else ""


def frozen_allowed_languages(config: Mapping[str, Any] | None) -> frozenset[str] | None:
    """The locale keys this call FROZE, exactly as the importer will read them.

    ``None`` when the call froze no follow-caller contract, which is the
    difference between "this call could not switch language" and "it could and
    did not". Deliberately the same read as the control plane's
    ``frozen_allowed_language_profiles``, over the same object: the frozen
    envelope the worker is handed IS the one persisted on the call, so a row
    validated here is a row the importer accepts.

    And the keys are taken RAW. The worker's own language parser casefolds
    every tag it builds, while the importer compares against these keys
    verbatim -- so a stored ``EN`` would make a perfectly correct ``en`` row
    fail the importer's allowlist and mark the WHOLE section malformed, taking
    the call's unrelated tool evidence with it. The raw keys are what a row
    must carry, and ``_resolve_language`` maps a tag back onto one.
    """
    data = config if isinstance(config, Mapping) else {}
    follow_caller = data.get("follow_caller")
    if not isinstance(follow_caller, Mapping):
        return None
    allowed = follow_caller.get("allowed_profiles")
    if not isinstance(allowed, Mapping) or not allowed:
        return None
    locales = frozenset(str(key) for key in allowed if isinstance(key, str) and key)
    return locales or None


def tool_index(tools: Any) -> dict[str, tuple[str, str]]:
    """``{model-facing name: (tool_kind, tool_ref)}`` for one call's tools.

    Keyed by NAME because that is what an SDK tool callback carries; the
    importer's ``tool_ref_index`` is the same rule keyed by ref, and a test
    asserts the two are inverses over one shared fixture so the kind rule
    cannot drift between producer and consumer.

    A builtin tool's closed kind is its own name -- ``end_call``, ``send_sms``
    and ``transfer_to_human`` are themselves members of ``TOOL_KINDS`` --
    while a webhook, connection or mcp tool's kind is its declared type.
    Anything else is ``other``. A spec with no usable ``ref`` still earns its
    kind: the row is emitted with ``tool_ref: ""`` and the drawer falls back
    to the kind's own label.
    """
    index: dict[str, tuple[str, str]] = {}
    if not isinstance(tools, Iterable) or isinstance(tools, (str, bytes, Mapping)):
        return index
    for spec in tools:
        if not isinstance(spec, Mapping):
            # A plain-string placeholder carries no type and no ref. It is
            # skipped rather than guessed: its kind is unknowable here, and
            # an unmapped name still produces a row through the fallback.
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        spec_type = str(spec.get("type") or "")
        if spec_type == "builtin":
            kind = name if name in TOOL_KINDS else "other"
        elif spec_type in TOOL_KINDS:
            kind = spec_type
        else:
            kind = "other"
        ref = spec.get("ref")
        ref = ref if isinstance(ref, str) and 0 < len(ref) <= MAX_TOOL_REF_CHARS else ""
        if ref and any(ord(char) < 32 or ord(char) == 127 for char in ref):
            # The importer refuses a control character outright. Emitting one
            # would malform the whole section and lose the tool evidence.
            ref = ""
        index[name] = (kind, ref)
    return index


def tool_kind_for(name: str, index: Mapping[str, tuple[str, str]]) -> tuple[str, str]:
    """The kind and ref for one model-facing tool name.

    A name absent from the index is the normal path for a tool the agent no
    longer has, for the worker-local retrieval tool, and for a builtin whose
    spec never reached the worker. It is never a reason to skip the row.
    """
    mapped = index.get(name)
    if mapped is not None:
        return mapped
    if name == KNOWLEDGE_TOOL_NAME:
        return ("knowledge_lookup", "")
    return (name if name in TOOL_KINDS else "other", "")


def _has_digit(text: str) -> bool:
    """The importer's rule for a label: no digit form of any script, ``Nd`` or ``No`` (the worker strips them)."""

    return any(unicodedata.category(char) in ("Nd", "No") for char in text)


class CallTranscriptEventRecorder:
    """Accumulate one recording's typed rows on the recorder's own clock.

    ``relative_ms`` is the evidence recorder's public offset function, so a row
    and a transcript turn are measured from the same zero -- the only way to
    order them, because LiveKit 1.8.1 exposes no durable turn id and every SDK
    ``created_at`` is a wall clock that cannot be differenced against a
    monotonic epoch.

    ``allowed_languages`` is the locale set this call FROZE at admission, or
    ``None`` when it froze no follow-caller contract. None means no language
    row can exist and the capability reports why.
    """

    def __init__(
        self,
        *,
        relative_ms: Callable[[float], int],
        engine: str,
        tools: Any = None,
        allowed_languages: frozenset[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[..., None] | None = None,
    ) -> None:
        self._relative_ms = relative_ms
        self._engine = engine if engine in ENGINES else ""
        self._tools = tool_index(tools)
        self._allowed_languages = allowed_languages
        self._clock = clock
        self._emit = emit or (lambda event, **details: None)
        self._events: list[dict[str, Any]] = []
        self._dropped = 0
        self._closed = False
        self._started_at: dict[str, float] = {}
        #: Language rows refused while no suppressed tool row was waiting for
        #: them, which is a refusal that arrived before its batch.
        self._pending_language_refusals = 0
        # Per call id: the typed terminal status and the monotonic instant it
        # arrived at, which is the end of THAT call rather than of its batch.
        self._ended: dict[str, tuple[str, float]] = {}
        # Nothing else may hold a reference into ``_events``: the rows carry
        # private keys that the section's projection strips, and a second
        # writer would be a second place the cap has to hold.

    # -- observed state ---------------------------------------------------

    @property
    def recorded(self) -> int:
        return len(self._events)

    def close(self) -> None:
        """Stop recording; a late tool result after teardown lands nowhere."""
        self._closed = True
        self._started_at.clear()
        self._ended.clear()
        self._pending_language_refusals = 0

    # -- session hooks (sync, media-loop safe) ----------------------------

    def on_tool_execution_updated(self, event: Any) -> None:
        """Remember each call's start time and its typed terminal status.

        Both halves are per ``call_id`` and bounded like the event list. The
        ended entry is what distinguishes ``cancelled`` from a failure --
        ``FunctionCallOutput.is_error`` cannot -- and it is read from the
        typed ``ToolCallEnded.status`` only. ``ToolCallEnded.message`` is
        ``str(exception)`` on failure and ``str(return_value)`` on success and
        is never touched.
        """
        try:
            if self._closed or self._engine == ENGINE_WORKFLOW:
                return
            update = getattr(event, "update", None)
            update_type = str(getattr(update, "type", "") or "")
            if update_type == "tool_call_started":
                call = getattr(update, "function_call", None)
                call_id = str(getattr(call, "call_id", "") or "")
                if call_id and len(self._started_at) < MAX_IN_FLIGHT_CALLS:
                    self._started_at.setdefault(call_id, self._clock())
                elif call_id:
                    self._note("in_flight_cap_reached", half="started")
                return
            if update_type != "tool_call_ended":
                return
            call_id = str(getattr(update, "call_id", "") or "")
            status = str(getattr(update, "status", "") or "")
            if not call_id or not status:
                return
            observed_at = self._clock()
            if len(self._ended) < MAX_IN_FLIGHT_CALLS:
                # The deferred-reply path can end a call twice (``_final``);
                # the first terminal entry is the one that describes the call.
                self._ended.setdefault(call_id, (status, observed_at))
            else:
                # Losing a terminal status means a CANCELLED call -- whose SDK
                # output is not an error -- could be recorded as a success, so
                # the loss is named rather than silent.
                self._note("in_flight_cap_reached", half="ended")
            self._revise_recorded_row(call_id, status, observed_at)
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook="tool_execution_updated", error=type(exc).__name__)

    def _revise_recorded_row(self, call_id: str, status: str, ended_at: float) -> None:
        """Apply a terminal status that arrived AFTER its batch row was written.

        In the ordinary path the terminal entry precedes
        ``function_tools_executed`` (measured against the pinned SDK for done,
        error and cancelled alike), so this does nothing. On the deferred-reply
        path -- a tool that calls ``ctx.update()`` -- the SDK delivers the
        terminal entry after the batch, and a row written in the meantime would
        keep a CANCELLED tool's ``is_error=False`` output and read as a
        success. The row is still in hand until finalization, so it is
        corrected rather than left wrong.

        Matched on ``ToolCallEnded.call_id``, which stays the ORIGINAL call id
        on that path: only the separate ``id`` field gains the ``_final``
        suffix. Measured against the pinned SDK, and pinned again by the
        deferred scenario in the worker image's build assertion, because
        matching the wrong field would leave the correction silently dead.

        A late ``done`` never upgrades anything: the output already decided
        whether the call succeeded, and a terminal "done" is also what a tool
        that raised ``StopResponse`` reports.
        """
        if status not in ("cancelled", "error"):
            return
        for row in reversed(self._events):
            if row["type"] != EVENT_TOOL_CALL or row.get("_call_id") != call_id:
                continue
            if not row["ok"] and row["error_class"] == "cancelled":
                return
            row["ok"] = False
            row["error_class"] = "cancelled" if status == "cancelled" else "unknown"
            # And the duration with it. A deferred tool's batch arrives while
            # the call is still running, so the row was measured to that
            # instant: a tool cancelled after thirty seconds would otherwise
            # be stored as having run for milliseconds, an under-claim on the
            # very path this correction exists to make truthful.
            started_at = row.get("_started_at")
            if started_at is not None:
                row["duration_ms"] = self._duration_ms(started_at, ended_at)
            # A suppressed row was only suppressed because it looked
            # successful, and a successful internal switch is represented by
            # its language row. This one was not successful, so there is no
            # language row to represent it and it ships as itself.
            row["_suppressed"] = False
            return

    def on_function_tools_executed(self, event: Any) -> None:
        """One ``tool_call`` row per executed call, in the SDK's own order.

        The row is written here rather than on the terminal entry because this
        is the one event whose arrival is deterministic relative to the batch:
        a deferred tool's ``ToolCallEnded`` can land after it, and a row that
        waited for one would be lost. A status that has not arrived yet leaves
        a failure classified ``unknown``, which is what the pinned SDK offers
        anyway for everything that is not a cancellation.
        """
        try:
            if self._closed or self._engine == ENGINE_WORKFLOW:
                return
            calls = list(getattr(event, "function_calls", None) or [])
            outputs = {
                str(getattr(output, "call_id", "") or ""): output
                for output in (getattr(event, "function_call_outputs", None) or [])
            }
            for call in calls:
                name = str(getattr(call, "name", "") or "")
                call_id = str(getattr(call, "call_id", "") or "")
                started_at = self._started_at.pop(call_id, None)
                # A `None` sentinel, not a falsy default: a terminal entry
                # genuinely observed at clock 0.0 is a timestamp, and treating
                # it as absent would measure that call to its batch instead.
                ended = self._ended.pop(call_id, None)
                ended_status, ended_at = ended if ended is not None else ("", None)
                if not name:
                    continue
                output = outputs.get(call_id)
                # A call the SDK executed but returned nothing for did not
                # succeed. Only the presence of a non-error output is ok, and
                # neither the output nor the arguments are read to decide it.
                #
                # The typed terminal status is part of that decision, not only
                # of the classification: a CANCELLED tool call reaches this
                # batch with ``is_error=False`` and a ``None`` output, because
                # the executor resolves the dispatch future before re-raising
                # the cancellation (pinned tool_executor.py). Read by
                # ``is_error`` alone, an interrupted tool would be stored as a
                # success. This is the one thing ``FunctionCallOutput`` cannot
                # tell us and ``ToolCallEnded.status`` can.
                cancelled = ended_status == "cancelled"
                ok = (
                    output is not None
                    and not bool(getattr(output, "is_error", False))
                    and not cancelled
                )
                # Suppressed only when the language row that REPLACES it can
                # exist. Three ways it cannot, and all three are checked here
                # rather than argued: this call froze no follow-caller
                # contract; this ENGINE records no language rows at all
                # (``on_language_switch`` returns before any refusal path on
                # those, so nothing downstream could give the row back); or a
                # switch was already refused and is still waiting for the row
                # it should have been. Every one of them is unreachable in
                # production -- the internal switch tool exists only on a
                # cascade call that froze a contract -- but "absence is never
                # silent" must not rest on that argument, which is the same
                # standard the rest of this module is held to.
                suppressed = (
                    ok
                    and name in SUPPRESSED_TOOLS
                    and self._allowed_languages is not None
                    and self._engine not in (ENGINE_WORKFLOW, ENGINE_GPT_LIVE)
                    and not self._pending_language_refusals
                )
                if ok and name in SUPPRESSED_TOOLS and self._pending_language_refusals:
                    # Consumed, not cleared: one refusal answers for one
                    # execution, so a later switch is judged on its own.
                    self._pending_language_refusals -= 1
                kind, ref = tool_kind_for(name, self._tools)
                self._append(
                    {
                        "type": EVENT_TOOL_CALL,
                        # Internal only: the projection below ships exactly the
                        # sanctioned keys. ``_call_id`` lets a late terminal
                        # status find the row it describes, and
                        # ``_suppressed`` marks a row that exists only in case
                        # one arrives -- see ``SUPPRESSED_TOOLS`` -- and
                        # ``_started_at`` lets that late entry re-measure the
                        # call it ends.
                        "_call_id": call_id,
                        "_suppressed": suppressed,
                        "_started_at": started_at,
                        "tool_kind": kind,
                        "tool_ref": ref,
                        "ok": ok,
                        "error_class": "" if ok else ("cancelled" if cancelled else "unknown"),
                        # Measured to THIS call's own terminal entry when the
                        # SDK delivered one, not to the batch that reports
                        # several calls at once: a 100 ms tool that ran beside
                        # a five-second one must not be stored as five
                        # seconds.
                        "duration_ms": (
                            None if started_at is None else self._duration_ms(started_at, ended_at)
                        ),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook="function_tools_executed", error=type(exc).__name__)

    def on_language_switch(self, event: str, **details: Any) -> None:
        """Tee of the switcher's emit: only an APPLIED switch is a row."""
        try:
            if self._closed or event != "language_switch_applied":
                return
            if self._engine in (ENGINE_WORKFLOW, ENGINE_GPT_LIVE):
                # GPT-Live follows the caller by instruction, so it has no
                # switch to record; the importer enforces the same rule.
                return
            allowed = self._allowed_languages
            if allowed is None:
                self._refused_language_row(REASON_NO_FROZEN_LANGUAGE_CONTRACT)
                return
            to_language = self._resolve_language(details.get("language"), allowed)
            from_language = self._resolve_language(details.get("old_language"), allowed)
            if not to_language:
                # A tag outside the call's own frozen allowlist would make the
                # importer mark the WHOLE section malformed, taking this
                # call's tool rows with it. Refuse the row, keep the evidence,
                # and say so under a name an operator can search for.
                self._refused_language_row("outside_frozen_allowed_profiles")
                return
            self._append(
                {
                    "type": EVENT_LANGUAGE_SWITCH,
                    "from": from_language,
                    "to": to_language,
                    "reason": REASON_BY_SOURCE.get(str(details.get("source") or ""), "follow_caller"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook="language_switch", error=type(exc).__name__)

    def on_caller_spoke_no_transcript(self, *, started_at: float) -> bool:
        """One row for caller speech the engine answered but never captioned.

        ``started_at`` is the monotonic instant the caller's speech began,
        which is earlier than now: the decision is made at the reply. The row
        is placed there, on the recording clock, so it sits before the reply
        it prompted rather than after it. Only the time is kept; there is no
        text to keep. A workflow call records no moments of its own. Returns
        whether a row was kept, so the caller counts rows, not attempts.
        """
        try:
            if self._closed or self._engine == ENGINE_WORKFLOW:
                return False
            return self._append({"type": EVENT_CALLER_SPOKE_NO_TRANSCRIPT}, at=started_at)
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook="caller_spoke_no_transcript", error=type(exc).__name__)
            return False

    # -- phone-menu and hold moments ------------------------------------------

    def on_navigation_event(self, event_type: str, *, at: float | None = None, **fields: Any) -> bool:
        """One phone-menu moment, validated against the closed contract, or nothing.

        The importer marks the WHOLE section malformed for a single field outside
        its shape, which would cost the call its tool and language rows too, so a
        row that does not fit is refused here (named in a worker log event, never
        with its content) and the rest of the evidence ships. A value-mode
        ``dtmf_sent`` has no field in which a digit could travel: it carries the
        variable's NAME and empty ``keys`` and ``label``. A workflow call records
        no navigation moments of its own. Returns whether a row was kept.
        """
        try:
            if self._closed or self._engine == ENGINE_WORKFLOW or event_type not in NAV_EVENT_TYPES:
                return False
            row = self._nav_row(event_type, fields)
            if row is None:
                self._note("navigation_row_refused", row_type=event_type)
                return False
            row["type"] = event_type
            return self._append(row, at=at)
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook=event_type, error=type(exc).__name__)
            return False

    def open_hold(self, *, sound: str = "", loops: int = 0, at: float | None = None) -> bool:
        """The call began waiting in a queue; the row's end is filled in by :meth:`close_hold`."""
        return self.on_navigation_event(EVENT_NAV_HOLD, at=at, end_ms=None, sound=sound, loops=loops)

    def close_hold(self, *, at: float | None = None) -> bool:
        """The wait ended: the latest open hold row gets its end (never before its start)."""
        try:
            if self._closed:
                return False
            for row in reversed(self._events):
                if row["type"] == EVENT_NAV_HOLD and row.get("end_ms") is None:
                    end = self._at_ms() if at is None else self._at_ms_of(at)
                    row["end_ms"] = max(int(row["at_ms"]), end)
                    return True
            return False
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._note("hook_failed", hook="close_hold", error=type(exc).__name__)
            return False

    @staticmethod
    def _nav_text(value: Any, limit: int) -> str | None:
        if not isinstance(value, str) or len(value) > limit:
            return None
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            return None
        return value

    def _nav_row(self, event_type: str, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        if event_type == EVENT_DTMF_UNSUPPORTED:
            return {}
        if event_type == EVENT_NAV_PERSON:
            confirmed = fields.get("confirmed")
            return {"confirmed": confirmed} if isinstance(confirmed, bool) else None
        if event_type == EVENT_NAV_MENU:
            text = self._nav_text(fields.get("text", ""), MAX_NAV_TEXT_CHARS)
            options = fields.get("options", ())
            if text is None or not isinstance(options, (list, tuple)) or len(options) > MAX_NAV_OPTIONS:
                return None
            kept: list[dict[str, str]] = []
            for option in options:
                key = option.get("key") if isinstance(option, Mapping) else None
                label = self._nav_text(option.get("label", "") if isinstance(option, Mapping) else None, MAX_NAV_LABEL_CHARS)
                if not isinstance(key, str) or not _NAV_KEY.fullmatch(key) or label is None or _has_digit(label):
                    return None
                kept.append({"key": key, "label": label})
            return {"text": text, "options": kept}
        if event_type == EVENT_DTMF_SENT:
            mode, keys = fields.get("mode"), fields.get("keys", "")
            label, field = fields.get("label", ""), fields.get("field", "")
            if mode == "keys":
                clean = self._nav_text(label, MAX_NAV_LABEL_CHARS)
                if not isinstance(keys, str) or not _NAV_KEYS.fullmatch(keys) or field != "" or clean is None or _has_digit(clean):
                    return None
                return {"mode": "keys", "keys": keys, "label": clean, "field": ""}
            if mode == "value":
                if keys != "" or label != "" or not isinstance(field, str) or not _NAV_FIELD.fullmatch(field):
                    return None
                return {"mode": "value", "keys": "", "label": "", "field": field}
            return None
        if event_type == EVENT_NAV_HOLD:
            end, sound, loops = fields.get("end_ms"), fields.get("sound", ""), fields.get("loops", 0)
            if end is not None or sound not in NAV_HOLD_SOUNDS:
                return None
            if isinstance(loops, bool) or not isinstance(loops, int) or not 0 <= loops <= MAX_NAV_LOOPS:
                return None
            return {"end_ms": None, "sound": sound, "loops": loops}
        return None

    def _refused_language_row(self, reason: str) -> None:
        """A switch that produced no row gives its tool row back.

        A successful ``switch_language`` execution is held suppressed because
        the language row REPRESENTS it. If that row is then refused -- an
        unverifiable tag, a contract that turned out to be absent -- the
        moment would vanish from both halves at once, which is the one thing
        this feature exists to prevent. Structural rather than argued: the
        switcher only applies frozen languages, so this needs a defect
        upstream to happen at all.
        """
        self._note("language_row_refused", reason=reason)
        for row in reversed(self._events):
            if row["type"] == EVENT_TOOL_CALL and row.get("_suppressed"):
                row["_suppressed"] = False
                return
        # No row to give back yet: the refusal arrived BEFORE the batch that
        # reports the execution. Remembered so that batch does not suppress
        # the row it is about to write, bounded because a refusal stream with
        # no executions behind it must not grow a counter without end.
        self._pending_language_refusals = min(
            MAX_IN_FLIGHT_CALLS, self._pending_language_refusals + 1
        )

    # -- the artifact subsection ------------------------------------------

    def section(self) -> dict[str, Any]:
        """The ``transcript_events`` object for this recording's transcript.json.

        Answers with a ``malformed`` header instead of raising: this is called
        from artifact finalization, and a defect in a note about a call must
        not cost the call its recording and transcript.
        """
        try:
            return self._section()
        except Exception as exc:  # noqa: BLE001 - finalization must not fail here
            self._note("section_failed", error=type(exc).__name__)
            return self._header(
                tool_status="malformed",
                language_status="malformed",
                events=[],
                dropped=0,
                engine="",
            )

    def _shippable(self) -> tuple[list[dict[str, Any]], int]:
        """The rows this section ships, in contract order, and what was lost.

        The cap is enforced HERE, at the one seam that decides what leaves,
        not only where rows are appended: a row held provisionally suppressed
        can become shippable later (a late cancellation un-suppresses it), and
        a 201st row would make the importer mark the WHOLE section malformed
        and lose every tool and language moment in it. Overflow is truncated
        and counted, which downgrades the status to `partial` -- a bounded
        slice must never imply a complete record.

        Sorted by the contract's own ordering key before anything is cut:
        `at_ms` first, then the type tie-break that puts a language switch
        ahead of a tool call at an equal offset. The sort is stable, so two
        rows of one type keep their arrival order, and it cannot make `at_ms`
        run backwards -- which the importer would read as a broken clock.
        """
        shippable = sorted(
            (row for row in self._events if not row.get("_suppressed")),
            key=lambda row: (row["at_ms"], row["source_order"]),
        )
        overflow = max(0, len(shippable) - MAX_EVENTS)
        return [self._sanitized(row) for row in shippable[:MAX_EVENTS]], self._dropped + overflow

    def set_row_sanitizer(self, sanitize: Callable[[dict[str, Any]], dict[str, Any]] | None) -> None:
        """A last pass over a phone-menu row as it leaves: it is masked against every value armed by then.

        A menu heard before a keypad value was armed was stored as it was said; the value, once armed,
        must not survive in that text. The pass runs on a copy, so the stored row is never edited.
        """

        self._row_sanitizer = sanitize

    def _sanitized(self, row: dict[str, Any]) -> dict[str, Any]:
        sanitize = getattr(self, "_row_sanitizer", None)
        if sanitize is None or row.get("type") != EVENT_NAV_MENU:
            return row
        try:
            return sanitize(dict(row))
        except Exception:  # noqa: BLE001 - a mask that fails ships no text rather than the stored text
            return {**row, "text": "", "options": []}

    def _section(self) -> dict[str, Any]:
        events, dropped = self._shippable()
        if self._engine == ENGINE_WORKFLOW:
            return self._header(
                tool_status="unavailable",
                tool_reason=REASON_ENGINE_WORKFLOW,
                language_status="unavailable",
                language_reason=REASON_ENGINE_WORKFLOW,
                events=[],
                dropped=dropped,
                engine=self._engine,
            )
        tool_rows = [row for row in events if row["type"] == EVENT_TOOL_CALL]
        language_rows = [row for row in events if row["type"] == EVENT_LANGUAGE_SWITCH]
        tool_status = self._observed_status(bool(tool_rows), dropped=dropped)
        tool_reason = ""
        if self._engine == ENGINE_GPT_LIVE:
            language_status, language_reason = "unavailable", REASON_ENGINE_GPT_LIVE
        elif self._allowed_languages is None:
            language_status, language_reason = "unavailable", REASON_NO_FROZEN_LANGUAGE_CONTRACT
        else:
            language_status = self._observed_status(bool(language_rows), dropped=dropped)
            language_reason = ""
        if language_status == "unavailable":
            # A status is a claim and the rows are evidence: a capability the
            # worker disclaims must ship no rows for it, or the drawer prints
            # "not recorded" above concrete moments. Only the language rows go:
            # the caller's own moments are not part of that capability.
            events = [row for row in events if row["type"] != EVENT_LANGUAGE_SWITCH]
        return self._header(
            tool_status=tool_status,
            tool_reason=tool_reason,
            language_status=language_status,
            language_reason=language_reason,
            events=events,
            dropped=dropped,
            engine=self._engine,
        )

    def _observed_status(self, has_rows: bool, *, dropped: int) -> str:
        if dropped:
            # A bounded slice must never imply a complete record, and
            # "observed, and nothing happened" contradicts a discard count
            # just as badly.
            return "partial"
        return "complete" if has_rows else "none_recorded"

    def _header(
        self,
        *,
        tool_status: str,
        language_status: str,
        events: list[dict[str, Any]],
        dropped: int,
        engine: str,
        tool_reason: str = "",
        language_reason: str = "",
    ) -> dict[str, Any]:
        # Every key is present with a documented empty value, never omitted:
        # the importer demands exactly the sanctioned key set, so an omission
        # would store the whole section as malformed and read like corruption.
        return {
            "schema": SCHEMA,
            "engine": engine,
            "dropped": max(0, int(dropped)),
            "tool_status": tool_status,
            "tool_reason": tool_reason,
            "language_status": language_status,
            "language_reason": language_reason,
            # Already ordered and capped by ``_shippable``; `seq` is
            # contiguous 1..N over exactly the rows that leave.
            "events": [self._renumbered(row, seq) for seq, row in enumerate(events, start=1)],
        }

    def _renumbered(self, row: Mapping[str, Any], seq: int) -> dict[str, Any]:
        """``seq`` is contiguous 1..N over the rows the section SHIPS.

        Recomputed here rather than kept from recording time, because a
        disclaimed capability's rows are removed above and a hole in ``seq``
        is what the importer reads as silent loss.
        """
        keys = _KEYS_BY_TYPE[row["type"]]
        out = {**row, "seq": seq}
        return {key: out[key] for key in keys}

    # -- internals --------------------------------------------------------

    def _append(self, row: dict[str, Any], *, at: float | None = None) -> bool:
        shippable = sum(1 for held in self._events if not held.get("_suppressed"))
        if shippable >= MAX_EVENTS or len(self._events) >= 2 * MAX_EVENTS:
            # The cap is on the rows this section would SHIP, so a provisional
            # suppressed row cannot spend a real moment's slot. The second
            # bound keeps the held list finite anyway: a call that ran
            # hundreds of internal switches must not grow it without end.
            self._dropped += 1
            self._note("event_dropped", reason="session_cap")
            return False
        row["source_order"] = SOURCE_ORDER_BY_TYPE[row["type"]]
        row["at_ms"] = self._at_ms() if at is None else self._at_ms_of(at)
        row["seq"] = len(self._events) + 1
        self._events.append(row)
        return True

    def _at_ms(self) -> int:
        """This moment on the recording clock, bounded and never backwards.

        A value over the contract's ceiling, or one somehow below the previous
        row's, would make the importer mark the whole section malformed. The
        clamp costs a single offset on a call longer than a day; the
        alternative costs the whole recording's evidence.
        """
        try:
            observed = int(self._relative_ms(self._clock()))
        except Exception:  # noqa: BLE001 - a clock must not lose the row
            observed = self._events[-1]["at_ms"] if self._events else 0
        bounded = min(MAX_AT_MS, max(0, observed))
        if self._events:
            bounded = max(bounded, int(self._events[-1]["at_ms"]))
        return bounded

    def _at_ms_of(self, instant: float) -> int:
        """An instant already past, on the recording clock and bounded.

        Unlike ``_at_ms`` it is not clamped to the previous row: a caller's
        speech began before the reply that decided it, and a tool row may have
        been written in between. ``_shippable`` sorts every row by the
        contract's key before anything leaves, so the file never runs
        backwards; clamping here would move the moment to the wrong place.
        """
        try:
            observed = int(self._relative_ms(float(instant)))
        except Exception:  # noqa: BLE001 - a clock must not lose the row
            return self._at_ms()
        return min(MAX_AT_MS, max(0, observed))

    def _duration_ms(self, started_at: float, ended_at: float | None) -> int | None:
        try:
            ended = self._clock() if ended_at is None else ended_at
            measured = int(round((ended - started_at) * 1_000))
        except Exception:  # noqa: BLE001
            return None
        return min(MAX_DURATION_MS, max(0, measured))

    def _resolve_language(self, value: Any, allowed: frozenset[str]) -> str:
        """One tag as the call's frozen contract spells it, or ``""``.

        The switcher hands over a casefolded tag; the frozen keys are compared
        verbatim by the importer. Resolving onto the stored spelling is what
        keeps a correct row from being refused over case alone -- and a tag
        that matches no frozen locale resolves to nothing, which is the one
        answer that cannot poison the section.
        """
        text = str(value or "").strip()
        if not text or len(text) > MAX_LANGUAGE_TAG_CHARS:
            return ""
        if text in allowed:
            return text
        folded = text.casefold()
        for candidate in allowed:
            if candidate.casefold() == folded:
                return candidate
        return ""

    def _note(self, event: str, **details: Any) -> None:
        try:
            self._emit(f"call_transcript_event_{event}", **details)
        except Exception:  # noqa: BLE001 - the log sink cannot take a call down
            pass


def install_call_transcript_events(session: Any, recorder: CallTranscriptEventRecorder) -> None:
    """Register the two SDK tool hooks.

    The language tee is wired by the caller that owns the switcher's emit,
    because that emit exists before the switcher does. Registration order
    against any other handler of the same event is deliberately not relied
    on: the pinned SDK stores handlers per event in a set.
    """
    session.on("tool_execution_updated", recorder.on_tool_execution_updated)
    session.on("function_tools_executed", recorder.on_function_tools_executed)

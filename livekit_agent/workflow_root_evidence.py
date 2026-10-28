# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""A workflow call's evidence parity with the freeform path, and its line latency ledger.

The freeform worker emits two per-call summaries the SIP harness grades every
call on: ``opening_turn_summary`` (who spoke first and why, with the opening's
timings) and ``caller_audio_flow_summary`` (whether caller audio kept reaching
recognition). The workflow root emitted neither, so its first real call failed
both rows for want of a record rather than for anything the call did. The
evidence contract holds for every engine; this module is the root's half.

**Caller audio flow.** The root reuses :class:`caller_audio_flow.CallerAudioFlowLog`
and :class:`~caller_audio_flow.CallerVadEdgeLog` unchanged. "Reaching this
session's recognition input" means, for the root, a frame the finite
listener's capture took from its RTC ingress (``WorkflowRtcIngress``): the
VAD and the listener's PCM ring are fed from exactly that stream, as
``Agent.stt_node`` feeds the freeform transcriber. Frames are the ingress's
20 ms frames, so ``frames_total`` counts differ from a freeform call's for the
same audio; the semantics (frames that reached recognition, starvation gaps,
sidecar energy while starved) are the same. No flow field lacks a root meaning.

**Opening summary.** One per OUTBOUND call, like the freeform path. The
root's pickup (``WorkflowPickup``) is its first-speaker owner:

=======================  ==================================================
field                    root meaning
=======================  ==================================================
policy / excluded_by     ``callee_first`` unless ``callee_first_exclusion``
                         named a reason, then ``agent_first`` + the reason
first_reply_path         ``workflow_pickup`` on callee-first (the pickup owns
                         the first business turn); absent on agent-first, as
                         on the freeform path
callee_spoke             a caller onset reached the pickup before it opened
opening_reason           ``after_greeting`` (opened on the caller's turn),
                         ``silence_timeout`` (opened on the silence window)
clock_anchor, *_ms       the answer gate's anchor; ms from the room join
                         instant the answer watch measured from (``origin``)
callee_onset_ms /        the first sample / end sample of the caller turn
callee_greeting_end_ms   the pickup consumed, on the capture's sample clock
opening_requested_ms     the pickup started the opening turn
opening_dispatched_ms    the opening reply reached the speech path
opening_first_audio_ms   the first agent playout after that
resumes                  caller onsets between requested and dispatched
state                    the furthest point the opening reached
=======================  ==================================================

:data:`ROOT_NULL_OPENING_FIELDS` have no root meaning and are emitted as an
explicit ``None`` (never omitted): the freeform decider's level floor, its
feed-gap and unobserved-audio flags, the machine handoff (the root's answer
detection never hands its floor over), the pickup's silence trim (the root
trims no silence) and ``pickup_in_context`` (the root's pickup is a business
turn's input, never a model's context).

**Line latency ledger.** One ``workflow_first_line_timeline`` per call and one
``workflow_line_timeline`` per later spoken line, each with every stage of
:data:`LINE_STAGES` as ``<stage>_ms`` relative to the media start, or an
explicit ``None`` when the stage did not happen on that line (an opening on
silence has no caller speech; an authored line has no approval before its ask).
One clock: the worker's monotonic clock. The caller's speech start, voice end
(the VAD's END less the trailing silence it measured) and speech end (the VAD's
END) are capture samples placed on it from the capture's first frame; every
other stage is the monotonic instant the worker observed it. A recognised
caller turn's :data:`REPORTED_STAGES` (voice end, seal, input EOF, the
provider's final, its close) that nothing reported are ``None`` AND named in
``unknown_stages`` with a reason, never zero or a neighbour's value. The two roles share the output
stage names: ``ask_sent`` is the live session's ask or the authored source's
synthesis request, ``session_first_audio`` the first audio either returned
(the authored marks are its first segment's). Durations the
server measured (``decide_ms``, ``server_total_ms``) and the SDK's reply
timeout budget ride alongside and are not stages. Stages are first-wins and
checked non-decreasing; a stage earlier than one before it is NAMED in
``out_of_order`` rather than dropped, so a broken clock is visible in the data.
The provider's final is not in that chain (a provider may endpoint before the
turn is sealed); it is named only outside the caller's speech start and the
listener's close.

Pure and synchronous: no livekit import, no I/O, clocks injected. Every emit is
guarded here, because the callers are media and teardown paths that a lost
diagnostic must never break.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

OWNER = "workflow_root"
FIRST_LINE_EVENT = "workflow_first_line_timeline"
LINE_EVENT = "workflow_line_timeline"
LINE_TRUNCATED_EVENT = "workflow_line_timeline_truncated"
FIRST_REPLY_PATH = "workflow_pickup"
POLICY_CALLEE_FIRST = "callee_first"
POLICY_AGENT_FIRST = "agent_first"
REASON_AFTER_GREETING = "after_greeting"
REASON_SILENCE_TIMEOUT = "silence_timeout"

#: The freeform decider's telemetry keys; a root summary carries every one.
DECIDER_FIELDS = (
    "state", "clock_anchor", "answer_observed_ms", "anchor_ms", "input_released_ms", "callee_onset_ms",
    "callee_greeting_end_ms", "opening_requested_ms", "opening_dispatched_ms", "opening_first_audio_ms",
    "resumes", "opening_reason", "feed_gap_in_window", "unobserved_audio_at_open", "machine_verdict",
    "floor_dbfs",
)
#: Fields with no workflow-root meaning, always emitted as ``None``.
ROOT_NULL_OPENING_FIELDS = frozenset({
    "feed_gap_in_window", "unobserved_audio_at_open", "machine_verdict", "floor_dbfs",
    "opening_silence_trimmed_ms", "pickup_in_context",
})

#: Call-level marks, reported on every line record; not ordered against lines.
CALL_STAGES = ("media_started", "answered", "input_released")
#: One spoken line's causal chain, in the order it must happen.
LINE_STAGES = (
    "callee_speech_started", "callee_voice_ended", "callee_speech_ended", "callee_speech_sealed",
    "listener_input_ended", "provider_final", "listener_closed", "listener_final", "turn_sent", "turn_returned",
    "delivery_claimed", "source_granted", "delivery_approved", "delivery_started", "ask_sent",
    "session_first_audio", "playout_started", "line_finished",
)
#: The stages a reply output reports for its own line.
OUTPUT_STAGES = frozenset(LINE_STAGES[LINE_STAGES.index("delivery_claimed"):])
#: The stages the root's own turn records.
TURN_STAGES = frozenset(LINE_STAGES[:LINE_STAGES.index("delivery_claimed")])
#: A recognised caller turn's own marks (``CallerTurnTiming``), by its fields.
LISTENER_STAGES = {"callee_speech_sealed": "sealed_at", "listener_input_ended": "input_ended_at",
                   "provider_final": "provider_final_at", "listener_closed": "listener_closed_at"}
#: Every stage a recognised caller turn owes; one nothing reported is unknown.
REPORTED_STAGES = ("callee_voice_ended", *LISTENER_STAGES)
#: Not ordered against its neighbours; bounded by the speech start and the close.
CONCURRENT_STAGES = frozenset({"provider_final"})
#: Why a stage the turn had carries no time: nothing on that path reported it.
UNKNOWN_NOT_REPORTED = "not_reported"
#: Line records kept per call; past it a line is counted, not emitted.
MAX_LINE_RECORDS = 64


def _quiet(emit: Callable[..., None], event: str, **details: Any) -> None:
    try:
        emit(event, **details)
    except Exception:  # noqa: BLE001 - a diagnostic cannot break a call
        pass


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


class RootOpeningRecord:
    """The root's ``opening_turn_summary``: first-wins marks, one summary."""

    def __init__(self, *, origin: float, excluded_by: str | None, clock_anchor: Callable[[str], str | None]) -> None:
        self._origin = float(origin)
        self._excluded_by = excluded_by
        self._anchor_of = clock_anchor
        self._marks: dict[str, float] = {}
        self._clock_anchor: str | None = None
        self._reason: str | None = None
        self._onsets = 0
        self._resumes = 0
        self.emitted = False

    @property
    def policy(self) -> str:
        return POLICY_AGENT_FIRST if self._excluded_by is not None else POLICY_CALLEE_FIRST

    def _mark(self, name: str, at: Any) -> None:
        value = _finite(at)
        if value is not None and name not in self._marks:
            self._marks[name] = value

    def note_answer(self, disposition: Any, at: float) -> None:
        try:
            anchor = self._anchor_of(str(disposition or ""))
        except Exception:  # noqa: BLE001 - a diagnostic cannot break the answer watch
            anchor = None
        if self._clock_anchor is None and anchor is not None:
            self._clock_anchor = str(anchor)
        self._mark("answer_observed", at)
        if anchor is not None:
            self._mark("anchor", at)

    def note_input_released(self, at: float) -> None:
        self._mark("input_released", at)

    def note_onset(self) -> None:
        """A caller onset reached the pickup (before or after it opened)."""
        if "requested" not in self._marks:
            self._onsets += 1
        elif "dispatched" not in self._marks:
            self._resumes += 1

    def note_pickup_observation(self, *, first: float | None, end: float | None) -> None:
        """The pickup consumed a caller turn (worded or blank) before opening."""
        self._mark("callee_onset", first)
        if end is not None:
            # The LAST consumed turn's end is where the greeting ended.
            value = _finite(end)
            if value is not None:
                self._marks["callee_greeting_end"] = value

    def note_requested(self, reason: str, at: float) -> None:
        if "requested" in self._marks:
            return
        self._mark("requested", at)
        if reason in (REASON_AFTER_GREETING, REASON_SILENCE_TIMEOUT):
            self._reason = reason

    def note_dispatched(self, at: float) -> None:
        if "requested" in self._marks:
            self._mark("dispatched", at)

    def note_first_audio(self, at: float) -> None:
        if "dispatched" in self._marks:
            self._mark("first_audio", at)

    def _ms(self, name: str) -> int | None:
        value = self._marks.get(name)
        return None if value is None else int(round((value - self._origin) * 1000))

    def _state(self) -> str:
        if "dispatched" in self._marks:
            return "conversation"
        if "requested" in self._marks:
            return "opening"
        if "callee_onset" in self._marks or self._onsets:
            return "callee_speaking"
        return "listening" if "answer_observed" in self._marks else "waiting_for_answer"

    def summary(self) -> dict[str, Any]:
        """The one record, with every decider key (``None`` where the root has none)."""
        if self._excluded_by is not None:
            return {"owner": OWNER, "policy": POLICY_AGENT_FIRST, "excluded_by": self._excluded_by,
                    "opening_silence_trimmed_ms": None}
        record: dict[str, Any] = {
            "owner": OWNER,
            "policy": POLICY_CALLEE_FIRST,
            "callee_spoke": bool(self._onsets),
            "first_reply_path": FIRST_REPLY_PATH,
            "state": self._state(),
            "clock_anchor": self._clock_anchor,
            "answer_observed_ms": self._ms("answer_observed"),
            "anchor_ms": self._ms("anchor"),
            "input_released_ms": self._ms("input_released"),
            "callee_onset_ms": self._ms("callee_onset"),
            "callee_greeting_end_ms": self._ms("callee_greeting_end"),
            "opening_requested_ms": self._ms("requested"),
            "opening_dispatched_ms": self._ms("dispatched"),
            "opening_first_audio_ms": self._ms("first_audio"),
            "resumes": self._resumes,
            "opening_reason": self._reason,
        }
        for field in ROOT_NULL_OPENING_FIELDS:
            record[field] = None
        return record

    def emit(self, emit: Callable[..., None], event: str, **details: Any) -> None:
        """Emit the summary once; never raises."""
        if self.emitted:
            return
        self.emitted = True
        try:
            record = self.summary()
        except Exception:  # noqa: BLE001 - a diagnostic cannot break teardown
            return
        _quiet(emit, event, **details, **record)


class TurnMarks:
    """One business turn's root-side stages, shared by every line of its reply."""

    __slots__ = ("stages", "unknown", "decide_ms", "server_total_ms")

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.unknown: dict[str, str] = {}
        self.decide_ms: float | None = None
        self.server_total_ms: float | None = None


class WorkflowLineTimeline:
    """The per-line latency ledger: monotonic marks in, one bounded event per line out."""

    def __init__(self, *, emit: Callable[..., None], clock: Callable[[], float] = time.monotonic,
                 max_lines: int = MAX_LINE_RECORDS) -> None:
        self._emit = emit
        self._clock = clock
        self._max_lines = max(1, int(max_lines))
        self._call: dict[str, float] = {}
        self._final: TurnMarks | None = None
        self._current: TurnMarks | None = None
        self.reply_timeout_seconds: float | None = None
        self.lines = 0
        self.dropped = 0
        self._closed = False

    def now(self) -> float:
        return float(self._clock())

    # ------------------------------------------------------------ call marks
    def note_call(self, stage: str, at: float | None = None) -> None:
        if stage in CALL_STAGES and stage not in self._call:
            value = _finite(self.now() if at is None else at)
            if value is not None:
                self._call[stage] = value

    # ------------------------------------------------------------ turn marks
    def note_caller_final(self, *, speech_started: float | None, speech_ended: float | None,
                          at: float | None = None, timing: Any = None, voice_ended: float | None = None) -> None:
        """A recognised caller turn reached the root; it belongs to the next turn sent.

        ``timing`` is the turn's ``CallerTurnTiming``; ``voice_ended`` its voice
        end on this clock. Coalesced finals keep the first one's speech start and
        take every later stage from the last one, so a stage the last final
        lacks is unknown, never an earlier final's.
        """
        marks = self._final or TurnMarks()
        started = _finite(speech_started)
        if started is not None and "callee_speech_started" not in marks.stages:
            # Coalesced finals: the turn began at the first one's speech.
            marks.stages["callee_speech_started"] = started
        for stage, value in (("callee_speech_ended", speech_ended), ("listener_final", self.now() if at is None else at)):
            value = _finite(value)
            if value is not None:
                marks.stages[stage] = value
        try:
            reported = {"callee_voice_ended": voice_ended,
                        **{stage: getattr(timing, name, None) for stage, name in LISTENER_STAGES.items()}}
        except Exception:  # noqa: BLE001 - a diagnostic cannot cost the caller's turn
            reported = dict.fromkeys(REPORTED_STAGES)
        for stage, value in reported.items():
            value = _finite(value)
            if value is None:
                marks.stages.pop(stage, None)
                marks.unknown[stage] = UNKNOWN_NOT_REPORTED
            else:
                marks.stages[stage] = value
                marks.unknown.pop(stage, None)
        self._final = marks

    def turn_sent(self, at: float | None = None) -> TurnMarks:
        marks, self._final = self._final or TurnMarks(), None
        value = _finite(self.now() if at is None else at)
        if value is not None:
            marks.stages["turn_sent"] = value
        return marks

    def turn_returned(self, marks: TurnMarks, *, timing: Any = None, at: float | None = None) -> None:
        value = _finite(self.now() if at is None else at)
        if value is not None:
            marks.stages["turn_returned"] = value
        if isinstance(timing, Mapping):
            marks.decide_ms = _finite(timing.get("decide_ms"))
            marks.server_total_ms = _finite(timing.get("total_ms"))
        self._current = marks

    def current_turn(self) -> TurnMarks | None:
        """The turn the next reply output speaks for (the last one returned)."""
        return self._current

    # ----------------------------------------------------------------- lines
    def line(self, turn: TurnMarks | None, stages: Mapping[str, Any], *, status: Any, role: str) -> None:
        """One spoken line finished (or failed); emits its record. Never raises."""
        try:
            self._line(turn, stages, status=status, role=role)
        except Exception:  # noqa: BLE001 - a diagnostic cannot break playout
            pass

    def _line(self, turn, stages, *, status, role):
        if self._closed:
            return
        if self.lines >= self._max_lines:
            self.dropped += 1
            if self.dropped == 1:
                _quiet(self._emit, LINE_TRUNCATED_EVENT, kept=self._max_lines)
            return
        index = self.lines
        self.lines += 1
        merged: dict[str, float] = dict(turn.stages) if turn is not None else {}
        for stage, value in dict(stages).items():
            value = _finite(value)
            if stage in OUTPUT_STAGES and value is not None:
                merged[stage] = value
        _quiet(self._emit, FIRST_LINE_EVENT if index == 0 else LINE_EVENT,
               **self.record(merged, turn=turn, index=index, status=status, role=role))

    def record(self, stages: Mapping[str, float], *, turn: TurnMarks | None, index: int, status: Any,
               role: str) -> dict[str, Any]:
        origin = self._call.get("media_started")

        def ms(value: float | None) -> int | None:
            if value is None or origin is None:
                return None
            return int(round((value - origin) * 1000))

        record: dict[str, Any] = {"owner": OWNER, "line_index": index, "line_role": role,
                                  "line_status": status if isinstance(status, str) else None,
                                  "clock_origin": "media_started"}
        for stage in CALL_STAGES:
            record[f"{stage}_ms"] = ms(self._call.get(stage))
        for stage in LINE_STAGES:
            record[f"{stage}_ms"] = ms(stages.get(stage))
        record["decide_ms"] = None if turn is None or turn.decide_ms is None else round(turn.decide_ms, 2)
        record["server_total_ms"] = (None if turn is None or turn.server_total_ms is None
                                     else round(turn.server_total_ms, 2))
        budget = _finite(self.reply_timeout_seconds) if role == "live" else None
        record["reply_timeout_budget_ms"] = None if budget is None else int(round(budget * 1000))
        record["unknown_stages"] = {} if turn is None else {stage: turn.unknown[stage] for stage in LINE_STAGES
                                                            if stage in turn.unknown and stages.get(stage) is None}
        record["out_of_order"] = out_of_order(record)
        return record

    def close(self, *, role: str) -> None:
        """At teardown: a call that spoke no line still owes its first-line record."""
        if self._closed:
            return
        try:
            if self.lines == 0:
                self._line(self._current, {}, status=None, role=role)
        finally:
            self._closed = True


def out_of_order(record: Mapping[str, Any]) -> list[str]:
    """Every line stage earlier than a stage before it (or than the answer/media start).

    A concurrent stage is named only when it falls outside its own bounds.
    """
    def number(stage: str) -> float | None:
        value = record.get(f"{stage}_ms")
        return None if not isinstance(value, (int, float)) or isinstance(value, bool) else value

    floor = None
    for stage in ("media_started", "answered"):
        value = number(stage)
        if value is not None:
            floor = value if floor is None else max(floor, value)
    start = floor
    wrong: list[str] = []
    for stage in LINE_STAGES:
        value = number(stage)
        if value is None:
            continue
        if stage in CONCURRENT_STAGES:
            low = max((bound for bound in (start, number("callee_speech_started")) if bound is not None), default=None)
            high = number("listener_closed")
            if (low is not None and value < low) or (high is not None and value > high):
                wrong.append(stage)
            continue
        if floor is not None and value < floor:
            wrong.append(stage)
            continue
        floor = value
    return wrong


class PickupEvidence:
    """The pickup's observer: what it did, onto the opening record, in seconds.

    ``sample_time`` maps a capture sample index to the monotonic clock (``None``
    before the capture's first frame). Duck-typed on the observation, so this
    module stays importable without the voice runtime.
    """

    def __init__(self, record: RootOpeningRecord, *, sample_time: Callable[[int], float | None],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.record, self.sample_time, self.clock = record, sample_time, clock

    def onset(self) -> None:
        self.record.note_onset()

    def pickup_observation(self, observation: Any) -> None:
        first = getattr(observation, "turn_first_sample", None)
        if type(first) is not int:
            first = getattr(observation, "first_sample", None)
        end = getattr(observation, "end_sample", None)
        self.record.note_pickup_observation(
            first=self.sample_time(first) if type(first) is int else None,
            end=self.sample_time(end) if type(end) is int else None)

    def opening_requested(self, reason: Any, _pickup: Any = None) -> None:
        self.record.note_requested(str(reason or ""), self.clock())


__all__ = [
    "CALL_STAGES", "CONCURRENT_STAGES", "DECIDER_FIELDS", "FIRST_LINE_EVENT", "FIRST_REPLY_PATH", "LINE_EVENT",
    "LINE_STAGES", "LINE_TRUNCATED_EVENT", "LISTENER_STAGES", "MAX_LINE_RECORDS", "OUTPUT_STAGES", "OWNER",
    "POLICY_AGENT_FIRST", "POLICY_CALLEE_FIRST", "PickupEvidence", "REASON_AFTER_GREETING", "REASON_SILENCE_TIMEOUT",
    "REPORTED_STAGES",
    "ROOT_NULL_OPENING_FIELDS", "RootOpeningRecord", "TURN_STAGES", "TurnMarks", "UNKNOWN_NOT_REPORTED",
    "WorkflowLineTimeline", "out_of_order",
]

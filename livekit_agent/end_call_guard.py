# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The end_call guard: decline the agent's own hang-up on the shape that ends calls.

**What it is for.** A cascade phone agent talking to a caller who speaks in
short bursts (1-2 s of speech, ~2 s of quiet) has every reply cut by the next
burst. After eight to ten such cut replies the model calls its ``end_call``
tool in one of the pauses, whatever the ending policy says: the live contexts
replay at about 47/100 on that shape and 0/190 on every shallower one. The
trigger is the model's own accumulated history of replies to a caller who
keeps talking; the run of cut replies is what the worker can see of it.

**What it does.** Inside the tool invocation, BEFORE anything irreversible
(``WorkerLocalEndCall`` closes admission and schedules the goodbye only after
this guard allows), it refuses the hang-up when

* the caller's words are still on their way to the model (a held final);
* the caller is speaking right now, or resumed after the turn the model
  answered (unconsumed caller evidence);
* at least three of the agent's last four replies were cut off by the caller
  (the run), up to a measured budget of caller turns per call.

Everything else is ALLOWED exactly as today: the guard never judges the
caller's words. A refusal is a tool output telling the model to keep
helping; every refusal also puts the mutual-silence rule on the clock (the
worker does that, through ``MutualSilenceRule.promote_to_enforce`` and
``begin_post_refusal_episode``), so a refused caller who then goes quiet is
ended the way every quiet call is.

**Signal kinds, never time windows.** Every veto keys on a transition (the
caller's ``speaking`` edge), an arrival (a transcript entering the caller
transcript gate before it is held), or an identity (the committed turn's
item id, the tool call id). The one duration here, the defer cap, bounds a
wait for the run's end transition; it classifies nothing.

**Provenance is consume-once.** The turn the model answered is bound to the
forwarded finals that made its text, in order, by the SDK's own join; a
later duplicate final (burst callers repeat themselves) can never be
consumed by an earlier turn. A turn whose text no run of finals reproduces
gets provenance ``unknown`` and falls back to the synchronous signals.

Everything here is pure and synchronous except :meth:`EndCallGuard.evaluate`,
which only awaits the injected defer. Loop-affine: called from the media
loop, no locks.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

#: The worker environment variable an operator flips without an image change.
MODE_ENV = "SKYSAY_END_CALL_GUARD_MODE"
#: The per-call budget of caller turns the run rule may refuse; measured on
#: the live contexts (gate 2 of the build) and frozen before acceptance.
BUDGET_ENV = "SKYSAY_END_CALL_GUARD_BUDGET"
MODE_ENFORCE = "enforce"
MODE_SHADOW = "shadow"
MODE_OFF = "off"
MODES = frozenset({MODE_ENFORCE, MODE_SHADOW, MODE_OFF})

#: The run: at least CUT_RUN_MIN of the agent's last CUT_RUN_WINDOW replies
#: cut off by the caller. Count-based, not a strict streak: the second live
#: hang-up had eight cut replies, then one 32-character reply that happened
#: to complete, then the caller's next fragment, then the tool.
CUT_RUN_WINDOW = 4
#: Caller-evidence events kept per call (VAD edges, interims, finals); a
#: decision reads only those above its turn's anchor.
EVIDENCE_EVENT_CAP = 4096
CUT_RUN_MIN = 3
#: How long rule 4 waits for the caller's run to end before refusing anyway.
DEFER_CAP_SECONDS = 1.5
#: The smallest wait worth deferring for; below it the loop refuses instead.
DEFER_MIN_SECONDS = 0.1
#: A structural bound on defers per decision, next to the deadline, so a
#: clock that does not advance (or a wait that returns at once) cannot spin.
MAX_DEFERS = 8
#: Placeholder until gate 2 freezes it; the worker reads BUDGET_ENV first.
DEFAULT_BUDGET = 3

DECISION_EVENT = "end_call_guard"
SUMMARY_EVENT = "end_call_guard_summary"
#: The whole decision, from tool entry through any defer, a follower's wait
#: and the classifier, fits in this. Past it a missing verdict is
#: ``classifier_unavailable``; a structural refusal already reached stands.
DECISION_DEADLINE_SECONDS = 2.5
#: The classifier's own per-request bound; the effective timeout is
#: ``min(CLASSIFIER_TIMEOUT_SECONDS, remaining deadline)``.
CLASSIFIER_TIMEOUT_SECONDS = 1.5
#: A caller turn longer than this is not a short goodbye and is never sent
#: truncated (prefix truncation would keep "that's everything, goodbye" and
#: drop "actually, don't hang up"); the assistant turn is sent as its tail.
CALLER_TEXT_MAX_CHARS = 1500
ASSISTANT_TAIL_CHARS = 1500

# Bases (the ``basis`` field of every decision row).
BASIS_CALLER_WORDS_PENDING = "caller_words_pending"
BASIS_CALLER_STILL_SPEAKING = "caller_still_speaking"
BASIS_AGENT_REPLIES_CUT = "agent_replies_cut"
BASIS_MACHINE = "machine"
BASIS_BUDGET_EXHAUSTED = "budget_exhausted"
BASIS_NO_RUN = "no_run"
BASIS_CALLER_NOT_UNDERSTOOD = "caller_not_understood"
BASIS_NOT_CLOSING = "not_closing"
BASIS_INPUT_OVERFLOW = "input_overflow"
BASIS_CLOSING = "closing"
BASIS_WRONG_PERSON = "wrong_person"
BASIS_CLASSIFIER_UNAVAILABLE = "classifier_unavailable"

# The classifier's four classes, by what the CALLER's words do.
VERDICT_CLOSING = "closing"
VERDICT_WRONG_PERSON = "wrong_person"
VERDICT_MACHINE = "machine"
VERDICT_NOT_CLOSING = "not_closing"
VERDICTS = frozenset({VERDICT_CLOSING, VERDICT_WRONG_PERSON, VERDICT_MACHINE, VERDICT_NOT_CLOSING})
#: Not a verdict: the adapter timed out, failed, or answered off the set.
VERDICT_UNAVAILABLE = "unavailable"

DISPOSITION_ALLOWED = "allowed"
DISPOSITION_REFUSED = "refused"
DISPOSITION_WOULD_REFUSE = "would_refuse"
DISPOSITION_ABORTED = "aborted"

#: What the model reads on a refusal. Fixed, pinned by test, never the
#: caller's words. The first is also the text recorded when the follow-up is
#: cancelled: the released caller words start the next turn with it in
#: context.
REFUSAL_TEXT: dict[str, str] = {
    BASIS_CALLER_WORDS_PENDING: (
        "Call not ended: the caller has not said goodbye or confirmed they need "
        "nothing else. Keep helping with what they are telling you; end the call "
        "only after they say goodbye or confirm they need nothing else."
    ),
    # The same instruction as the words-pending string on purpose: measured
    # closed-loop on the live pre-hang-up context (N=60), this wording took
    # the model's same-turn spoken farewells from 30/60 to 1/60 and its
    # next-turn re-attempts from 240/240 to 107/240, where an explanation of
    # WHY ("your replies were cut off because...") changed nothing.
    BASIS_AGENT_REPLIES_CUT: (
        "Call not ended: the caller has not said goodbye or confirmed they need "
        "nothing else. Keep helping with what they are telling you; end the call "
        "only after they say goodbye or confirm they need nothing else."
    ),
    BASIS_CALLER_NOT_UNDERSTOOD: (
        "Call not ended: the caller's last words were not understood clearly. Ask "
        "them to repeat what they said, and end the call only after they clearly "
        "say goodbye or confirm they need nothing else."
    ),
}
REFUSAL_TEXT[BASIS_NOT_CLOSING] = REFUSAL_TEXT[BASIS_CALLER_WORDS_PENDING]
REFUSAL_TEXT[BASIS_INPUT_OVERFLOW] = REFUSAL_TEXT[BASIS_CALLER_WORDS_PENDING]
# A refusal the guard could not judge yet (the caller still speaking, or
# words held for the next turn) is treated alike by the worker: the SDK's
# follow-up is cancelled, the refused pair is dropped without a line and
# nothing is regenerated; the caller's own words drive the next turn. The
# text is recorded on the error output only, and never reaches the model.
REFUSAL_TEXT[BASIS_CALLER_STILL_SPEAKING] = REFUSAL_TEXT[BASIS_CALLER_WORDS_PENDING]

EVIDENCE_FINAL = "final"
EVIDENCE_INTERIM = "interim"
EVIDENCE_VAD_EDGE = "vad_edge"

PROVENANCE_MATCHED = "matched"
PROVENANCE_UNKNOWN = "unknown"

#: Rule 9's threshold per recogniser language: a final AT OR BELOW it is
#: "not understood". Pinned from data, never chosen: strictly between the
#: garbled finals that preceded a premature ending and the LOWEST correctly
#: transcribed caller final, on the harness bundles and a numbers-only
#: production export (30 days, 79 cascade calls), per language, biased low.
#: A language absent here has rule 9 OFF (its gap is unproven or unmeasured),
#: and the classifier is the net for garble there.
#:
#: ``en`` = 0.60: 60 production caller finals immediately preceding an ending,
#: minimum 0.92 (agent-ended 8, minimum 0.93); 0 of 60 refused. Harness floor
#: (nova-3 en, clean TTS at 8 kHz, 11 calls): 90 whole-fixture finals min
#: 0.977; the lowest CORRECT final overall a one-word split fragment at 0.67,
#: so the margin above the pin is 0.07 on the easiest audio. One-sided pin
#: (no English garble incident); exposure on ending finals 0/60.
#: ``et`` = 0.70, PROVISIONAL: six scored production finals from three
#: agent-ended calls. Two garbles, both refused: 0.464 (the incident, one
#: word) and 0.619 (one word, judged a garble from its text by the lead, audio
#: not heard, preceding another premature ending). Four correct, none
#: refused: 0.83, 0.90, 0.961, 0.998. Unmeasured on the harness (no Estonian
#: rows). Re-pin from the per-language counters in the guard summary once
#: more Estonian data exists.
LOW_CONFIDENCE_BY_LANGUAGE: dict[str, float] = {"en": 0.60, "et": 0.70}

#: Languages whose pin rests on too few finals to be more than provisional.
LOW_CONFIDENCE_PROVISIONAL: frozenset[str] = frozenset({"et"})


def low_confidence_language_key(language: str) -> str:
    """The table key for a call language: ``et`` → ``et``, ``en-US`` → ``en``, unknown → ``""``."""

    key = str(language or "").strip().lower()
    if not key:
        return ""
    if key in LOW_CONFIDENCE_BY_LANGUAGE:
        return key
    return key.split("-")[0]


def low_confidence_threshold_for(language: str) -> float | None:
    """The pinned threshold for a call language (``et``, ``en-US`` → ``en``), or None."""

    return LOW_CONFIDENCE_BY_LANGUAGE.get(low_confidence_language_key(language))


def low_confidence_provisional(language: str) -> bool:
    return low_confidence_language_key(language) in LOW_CONFIDENCE_PROVISIONAL


def resolve_mode(environ: Mapping[str, str]) -> str:
    """The guard's mode from the worker environment; SHADOW unless set.

    Like the silence rule's resolver: ``enforce`` is set explicitly, and junk
    reads as shadow, never as enforce.
    """

    value = str(environ.get(MODE_ENV) or "").strip().lower()
    return value if value in MODES else MODE_SHADOW


def resolve_budget(environ: Mapping[str, str]) -> int:
    """The per-call run budget; the frozen default unless the operator sets one."""

    raw = str(environ.get(BUDGET_ENV) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BUDGET
    return value if value >= 1 else DEFAULT_BUDGET


def _item_role(item: Any) -> str:
    return str(getattr(item, "role", "") or "")


def _item_type(item: Any) -> str:
    return str(getattr(item, "type", "") or "")


def cut_replies_of_last(items: Any, *, window: int = CUT_RUN_WINDOW) -> tuple[int, int]:
    """``(cut, considered)`` over the agent's last ``window`` replies.

    A reply counts as cut only when it is ``interrupted`` AND a user item
    follows it: the caller cut the agent, not the agent's own scheduler.
    Function-call items between messages are ignored. ``considered`` is how
    many assistant messages the window actually held, so a call with two
    replies cannot satisfy a three-of-four rule.
    """

    messages = [
        item
        for item in list(items or [])
        if _item_type(item) in ("", "message") and _item_role(item) in ("assistant", "user")
    ]
    assistant_positions = [i for i, item in enumerate(messages) if _item_role(item) == "assistant"]
    considered = assistant_positions[-window:]
    cut = 0
    for position in considered:
        item = messages[position]
        followed_by_caller = any(_item_role(later) == "user" for later in messages[position + 1 :])
        if bool(getattr(item, "interrupted", False)) and followed_by_caller:
            cut += 1
    return cut, len(considered)


def has_cut_run(items: Any) -> bool:
    cut, _considered = cut_replies_of_last(items)
    return cut >= CUT_RUN_MIN


@dataclass
class _Run:
    revision: int
    open: bool = True
    words: bool = False


class CallerEvidence:
    """The caller-evidence revision, its runs, and consume-once provenance.

    Fed by the worker from three places: the caller's VAD ``speaking`` edge
    (a start edge; the stop edge is the ABSENCE of evidence and never
    counts), every speech-carrying STT event as it enters the transcript
    gate (before it is held, so a final the gate withholds is still seen),
    and the finals the gate forwards (which the SDK assembles into the turn
    text). ``bind_turn_text`` runs in the agent's ``on_user_turn_completed``
    hook with the frozen text; ``bind_turn_item`` runs on the user item's
    ``conversation_item_added`` with the id the SDK actually inserted, which
    under preemptive reuse is not the object the hook saw.
    """

    def __init__(
        self,
        *,
        wordless_runs_count: Callable[[], bool] | None = None,
        counts_as_words: Callable[[str], bool] | None = None,
    ) -> None:
        #: Whether an OPEN run with no words is caller evidence. True on every
        #: transcript source whose first-word latency is unmeasured; False on a
        #: measured source, where a wordless run is the close floor's to defer
        #: on (``CloseFloorEvidence``) and never counts as the caller's words.
        self._wordless_runs_count = wordless_runs_count or (lambda: True)
        #: Whether a non-empty transcript is caller evidence at all: always on
        #: an unmeasured source; on a measured one only what the close floor
        #: reads as words (a filtered backchannel is not), so the guard and the
        #: close owner never disagree about what the caller said.
        self._counts_as_words = counts_as_words or (lambda _text: True)
        self.revision = 0
        self._runs: list[_Run] = []
        #: Finals the gate forwarded and no turn has consumed yet, in order:
        #: text, arrival revision, the recogniser's confidence for THAT final.
        self._forwarded: list[tuple[str, int, float | None]] = []
        #: The evidence kinds recorded, by revision, for ``unconsumed``; a
        #: start edge carries its run by reference. Bounded: a decision reads
        #: only evidence above its anchor, which is never older than the
        #: current turn, and ``EVIDENCE_EVENT_CAP`` is far above one turn.
        self._events: list[tuple[int, str, Any]] = []
        self.turn_id: str | None = None
        self.anchor: int | None = None
        self.provenance: str | None = None
        self._pending: tuple[int | None, str, float | None] | None = None
        #: The latest interim's text and revision, cleared by a forwarded
        #: final that arrived AFTER it; what a turn closed on a pending
        #: interim ends with.
        self._last_interim: tuple[str, int] | None = None
        #: Finals seen entering the gate and not yet forwarded, with their
        #: ARRIVAL revision and the recogniser's confidence: the gate may
        #: hold a final and release it later, and the turn's anchor must be
        #: where the final arrived, not where it was released, or evidence
        #: in between would sit below it.
        self._seen_finals: list[tuple[str, int, float | None]] = []
        #: The confidence of the LAST final of the current turn, or ``None``
        #: (a promoted interim, an unknown provenance, a recogniser that
        #: reports none). Never averaged over the turn.
        self.last_final_confidence: float | None = None
        self.caller_turns = 0

    # -- feeds ----------------------------------------------------------------

    def note_caller_speaking(self, speaking: bool) -> None:
        if speaking:
            self.revision += 1
            run = _Run(revision=self.revision)
            self._runs.append(run)
            del self._runs[:-8]
            self._events.append((self.revision, EVIDENCE_VAD_EDGE, run))
            del self._events[:-EVIDENCE_EVENT_CAP]
            return
        if self._runs:
            self._runs[-1].open = False

    def note_speech_event(
        self, *, is_final: bool, text: str = "", confidence: float | None = None
    ) -> None:
        if not str(text or "").strip():
            # Deepgram emits only non-empty transcripts, the gate drops an
            # empty record and the SDK ignores one: not caller evidence, so
            # no revision, no run words, nothing to veto with.
            return
        try:
            counts = bool(self._counts_as_words(str(text)))
        except Exception:  # noqa: BLE001 - an unreadable reading keeps today's veto
            counts = True
        if not counts:
            # A measured source's filtered backchannel: the close floor does
            # not read it as words, so neither does the guard.
            return
        self.revision += 1
        if self._runs:
            # Attributed to the current or most recent run, as the gate does.
            self._runs[-1].words = True
        self._events.append(
            (self.revision, EVIDENCE_FINAL if is_final else EVIDENCE_INTERIM, None)
        )
        del self._events[:-EVIDENCE_EVENT_CAP]
        if is_final:
            # The guard's OWN ledger of per-final confidence (the gate's
            # observation FIFO has another consumer, telemetry). ``None`` is
            # "the recogniser reported none", never a synthetic zero.
            self._seen_finals.append((str(text or "").strip(), self.revision, confidence))
            del self._seen_finals[:-64]
        if not is_final:
            # The latest interim, for a turn the SDK closes on a pending
            # interim: at end of turn it emits that interim as a final through
            # its own hook (never through the STT stream, so never through the
            # gate) and appends the text to the turn
            # (``audio_recognition.py:1062-1078``).
            self._last_interim = (" ".join(str(text or "").split()), self.revision)

    def note_forwarded_final(self, text: str) -> None:
        """A final the gate forwarded to the SDK, at the revision it ARRIVED
        (the gate may have held it): the earliest unforwarded final seen with
        this text. A final never seen entering (a path without the observer
        feed) is stamped with the current revision. The latest interim is
        cleared only when this final arrived after it; an interim that came
        while the final was held is still the caller's newest evidence."""

        text = str(text or "").strip()
        if not text:
            return
        arrived = self.revision
        confidence: float | None = None
        for index, (seen_text, seen_revision, seen_confidence) in enumerate(self._seen_finals):
            if seen_text == text:
                arrived = seen_revision
                confidence = seen_confidence
                del self._seen_finals[index]
                break
        self._forwarded.append((text, arrived, confidence))
        if self._last_interim is not None and self._last_interim[1] < arrived:
            self._last_interim = None

    # -- the turn ---------------------------------------------------------------

    def bind_turn_text(self, text: str) -> str:
        """Bind the frozen turn text to the finals that made it; the provenance.

        The SDK's join is ``" ".join`` of the forwarded finals' raw text, in
        order (``audio_recognition.py``: ``_audio_transcript += f" {t}"``,
        then ``lstrip``). The next k unconsumed finals must reproduce the
        text exactly; they are consumed and the anchor is the k-th final's
        arrival revision. A turn the SDK closed on a pending interim carries
        that interim's text as its tail; the join of the finals plus the
        gate's latest interim is accepted too, anchored on the interim.
        Leading finals no turn claimed (a turn the SDK never committed) are
        dropped only when a later window matches. No match at all:
        provenance ``unknown``, nothing consumed, and the stale finals are
        cleared so the next turn can match fresh.
        """

        wanted = " ".join(str(text or "").split())
        finals = self._forwarded
        interim = self._last_interim

        def _match(anchor: int, consume: int, last_final_confidence: float | None) -> str:
            del self._forwarded[:consume]
            self._last_interim = None
            self._pending = (anchor, PROVENANCE_MATCHED, last_final_confidence)
            return PROVENANCE_MATCHED

        # The forwarded finals are tried FIRST. A later interim that happens
        # to repeat a final's text must not take the match: it would advance
        # the anchor past the evidence that came between, leave the real
        # final unconsumed and replace its confidence with None (a garbled
        # 0.2 final would then never trip rule 9). The interim-only match is
        # for a turn the SDK closed on a pending interim with nothing
        # forwarded before it.
        def _windows():
            for start in range(len(finals)):
                joined = ""
                for k in range(start, len(finals)):
                    joined = finals[k][0] if not joined else f"{joined} {finals[k][0]}"
                    normalised = " ".join(joined.split())
                    if len(normalised) > len(wanted):
                        break
                    yield k, normalised

        # Pass 1: a window of finals reproduces the text exactly; the k-th
        # final's OWN confidence, never a text lookup. Every window is tried
        # before any interim is considered, so an interim repeating a final
        # cannot take a match a final owns.
        for k, normalised in _windows():
            if normalised == wanted:
                return _match(finals[k][1], k + 1, finals[k][2])
        # Pass 2: finals plus the promoted interim as the tail: no confidence.
        if interim is not None and interim[0]:
            for k, normalised in _windows():
                if f"{normalised} {interim[0]}" == wanted:
                    return _match(interim[1], k + 1, None)
            # Pass 3: the interim alone, for a turn closed on a pending
            # interim with nothing forwarded before it; finals no window
            # claimed are stale (a turn the SDK never committed) and go.
            if interim[0] == wanted:
                self._forwarded.clear()
                return _match(interim[1], 0, None)
        self._forwarded.clear()
        self._last_interim = None
        self._pending = (None, PROVENANCE_UNKNOWN, None)
        return PROVENANCE_UNKNOWN

    def bind_turn_item(self, item_id: str) -> None:
        """The SDK inserted the user item: this is the caller turn from now on."""

        self.caller_turns += 1
        self.turn_id = str(item_id or "") or f"turn-{self.caller_turns}"
        pending = self._pending
        self._pending = None
        if pending is None:
            # No hook ran for this item (nothing forwarded, or a path without
            # the agent hook): anchor at the item's own arrival, unknown.
            self.anchor, self.provenance = self.revision, PROVENANCE_UNKNOWN
            self.last_final_confidence = None
            return
        anchor, provenance, confidence = pending
        self.anchor = self.revision if anchor is None else anchor
        self.provenance = provenance
        self.last_final_confidence = confidence

    # -- reads ------------------------------------------------------------------

    def unconsumed(self) -> dict[str, int]:
        """Caller evidence above the current turn's anchor, by kind.

        A start edge whose run has ended with no words attributed is retired:
        a cough cannot veto a turn forever. A run with words, or still open,
        counts. Unknown provenance claims nothing here (fail-open about PAST
        attribution); evidence after a known instant is
        :meth:`evidence_since`, which needs no provenance.
        """

        if self.provenance == PROVENANCE_UNKNOWN or self.anchor is None:
            return {EVIDENCE_FINAL: 0, EVIDENCE_INTERIM: 0, EVIDENCE_VAD_EDGE: 0}
        return self.evidence_since(self.anchor)

    def evidence_since(self, revision: int) -> dict[str, int]:
        """Caller evidence recorded after ``revision``, by kind, retiring
        wordless closed runs. Provenance-free: the guard reads it against the
        revision it captured at tool entry, so a caller who resumes during an
        await is seen even when the turn's anchor is unknown."""

        counts = {EVIDENCE_FINAL: 0, EVIDENCE_INTERIM: 0, EVIDENCE_VAD_EDGE: 0}
        try:
            open_wordless_counts = bool(self._wordless_runs_count())
        except Exception:  # noqa: BLE001 - an unreadable source keeps today's veto
            open_wordless_counts = True
        for event_revision, kind, run in self._events:
            if event_revision <= revision:
                continue
            if kind == EVIDENCE_VAD_EDGE and run is not None and not run.words:
                if not run.open or not open_wordless_counts:
                    continue
            counts[kind] += 1
        return counts


@dataclass(frozen=True)
class Snapshot:
    """What the guard reads at one point of the loop; the worker builds it."""

    finished: bool
    ending_scheduled: bool
    user_speaking: bool  # ``user_state == "speaking"``; ``listening`` and ``away`` are both not speaking
    holds_final: bool
    unconsumed: Mapping[str, int]
    machine_corroborated: bool
    cut_replies: int
    replies_considered: int
    #: The recogniser's confidence for the last final of the turn the model
    #: answered, or ``None`` (rule 9 then does not fire).
    last_final_confidence: float | None = None
    #: The call's direction: ``wrong_person`` ends only an outbound call.
    outbound: bool = False
    #: Seconds left in the close floor's bounded wait for words behind a
    #: WORDLESS caller run (measured transcript sources only; 0 elsewhere).
    #: While it runs the decision defers; once it is spent a wordless run
    #: vetoes nothing and the verdict decides.
    caller_wordless_remaining: float = 0.0


@dataclass(frozen=True)
class Exchange:
    """What the classifier reads: the last exchange, nothing before it."""

    caller_text: str
    agent_text: str
    agent_interrupted: bool
    language: str


class ClosingClassifier(Protocol):
    """The adapter contract. ``classify`` returns one of ``VERDICTS`` or
    raises; anything else is ``unavailable``. ``available`` is False once the
    adapter disarmed itself (construction or self-test failed for this job)."""

    @property
    def available(self) -> bool: ...

    async def classify(self, exchange: Exchange, *, timeout: float) -> str: ...


@dataclass(frozen=True)
class Decision:
    disposition: str  # allowed | refused | defer
    basis: str
    words_held: bool = False
    text: str = ""
    consumes_budget: bool = False

    @property
    def refused(self) -> bool:
        return self.disposition == DISPOSITION_REFUSED

    @property
    def deferred(self) -> bool:
        return self.disposition == "defer"


ALLOW_FINISHED = Decision(DISPOSITION_ALLOWED, "finished")
ALLOW_SCHEDULED = Decision(DISPOSITION_ALLOWED, "ending_scheduled")
ABORTED = Decision(DISPOSITION_ABORTED, "interrupted")
#: Every structural rule passed and a classifier is armed: the loop must ask it.
CLASSIFY = Decision("classify", "classify")
#: A wordless caller run on a measured transcript source: wait for its words.
DEFER_WORDLESS = Decision("defer", BASIS_CALLER_STILL_SPEAKING)

#: Refusals the same-turn pin repeats without re-deciding: the verdicts that
#: cost a network round trip or rest on the turn's own transcript. A RUN
#: refusal is never pinned: the run is re-read from history on every attempt,
#: so two completed acknowledgements clear it and the next attempt may pass.
PINNED_BASES = frozenset({BASIS_NOT_CLOSING, BASIS_CALLER_NOT_UNDERSTOOD})
#: Refusals that spend the per-call budget: the ones a persistent model could
#: otherwise collect on every caller turn. Structural vetoes never spend it.
BUDGETED_BASES = frozenset({BASIS_AGENT_REPLIES_CUT, BASIS_NOT_CLOSING, BASIS_CALLER_NOT_UNDERSTOOD})
#: Refusals that JUDGED the caller's words (the keep-helping line and a
#: regenerated reply follow): the budgeted ones, and an overlong turn, which
#: is refused without a request and without budget but still owes a reply.
JUDGED_BASES = BUDGETED_BASES | frozenset({BASIS_INPUT_OVERFLOW})


def decide(
    snapshot: Snapshot,
    *,
    pinned: str | None,
    refused_turns: int,
    budget: int,
    deferred_once: bool,
    low_confidence_threshold: float | None = None,
    classifier_armed: bool = False,
    verdict: str | None = None,
    turn_refused: bool = False,
) -> Decision:
    """One pass of the ordered loop; pure.

    Order as executed: finished, ending scheduled, machine corroborated
    (first among the substantive rules: it overrides a same-turn pin and is
    never refused by the caller-evidence rules a recording keeps hot),
    caller words pending, caller speaking (defer until the deadline, then
    refuse), the same-turn pin, budget exhausted, the low-confidence final,
    then the classifier where armed (``verdict`` is its answer on the second
    pass), with the cut-reply run deciding what an unhelpful verdict means;
    without a classifier the run stands alone; otherwise allow.
    """

    if snapshot.finished:
        return ALLOW_FINISHED
    if snapshot.ending_scheduled:
        return ALLOW_SCHEDULED
    if snapshot.machine_corroborated:
        # BEFORE the caller-evidence rules: a recording talks over every
        # reply, so held finals and live speech stay hot for as long as it
        # plays, and a machine judged by those rules would be refused
        # forever, unbudgeted, and given a farewell check-in it never had.
        return Decision(DISPOSITION_ALLOWED, BASIS_MACHINE)
    unconsumed = snapshot.unconsumed
    if snapshot.holds_final or unconsumed.get(EVIDENCE_FINAL, 0) > 0:
        return Decision(
            DISPOSITION_REFUSED,
            BASIS_CALLER_WORDS_PENDING,
            words_held=True,
            text=REFUSAL_TEXT[BASIS_CALLER_WORDS_PENDING],
        )
    live_run = snapshot.user_speaking
    resumed = unconsumed.get(EVIDENCE_VAD_EDGE, 0) > 0 or unconsumed.get(EVIDENCE_INTERIM, 0) > 0
    if live_run or resumed:
        if live_run and not deferred_once:
            # ``deferred_once`` reads "the deadline is spent": the loop keeps
            # deferring while the caller speaks, so a second final that is
            # still arriving (an answer split in two) is judged in full.
            return Decision("defer", BASIS_CALLER_STILL_SPEAKING)
        return Decision(
            DISPOSITION_REFUSED,
            BASIS_CALLER_STILL_SPEAKING,
            text=REFUSAL_TEXT[BASIS_CALLER_STILL_SPEAKING],
        )
    if snapshot.caller_wordless_remaining > 0 and not deferred_once:
        # A wordless run on a measured transcript source: words, if any are
        # behind it, are due inside the measured budget. Wait for them (the
        # words then make it a live run above), and once the budget is spent
        # judge the turn as if the room were quiet.
        return DEFER_WORDLESS
    if pinned is not None:
        return Decision(DISPOSITION_REFUSED, pinned, text=REFUSAL_TEXT[pinned])
    if refused_turns >= budget and not turn_refused:
        # The ceiling counts DISTINCT caller turns; the turn just refused is
        # never the one that spends it, so a second attempt in the same
        # generation cannot slip past the refusal its sibling got.
        return Decision(DISPOSITION_ALLOWED, BASIS_BUDGET_EXHAUSTED)
    confidence = snapshot.last_final_confidence
    if (
        low_confidence_threshold is not None
        and confidence is not None
        and confidence <= low_confidence_threshold
    ):
        # Deterministic, before the network and before the run: a turn the
        # recogniser itself doubted never becomes "the answer", whatever the
        # classifier would make of the garble. ``None`` (a promoted interim,
        # an unknown provenance, a recogniser reporting none) never fires it.
        return Decision(
            DISPOSITION_REFUSED,
            BASIS_CALLER_NOT_UNDERSTOOD,
            text=REFUSAL_TEXT[BASIS_CALLER_NOT_UNDERSTOOD],
            consumes_budget=True,
        )
    run = snapshot.cut_replies >= CUT_RUN_MIN
    if classifier_armed and verdict is None:
        # The run is not a verdict on the CALLER: a caller who has closed
        # the conversation while the agent's replies kept getting cut is
        # allowed to end it (the harness refused a genuine goodbye twice on
        # the run alone). The classifier reads the caller's last words; the
        # run then decides only what an unhelpful verdict means.
        return CLASSIFY
    if verdict is not None:
        judged = verdict_decision(verdict, outbound=snapshot.outbound)
        if judged.basis in (BASIS_CLOSING, BASIS_WRONG_PERSON):
            # The caller closed the conversation (or the wrong person
            # answered an outbound call): allowed, run or no run.
            return judged
        if run:
            # Not closing, or no usable verdict (the classifier fails open
            # only where nothing structural stands), with the agent's
            # replies cut: the shape of the live hang-ups, refused as such.
            return Decision(
                DISPOSITION_REFUSED,
                BASIS_AGENT_REPLIES_CUT,
                text=REFUSAL_TEXT[BASIS_AGENT_REPLIES_CUT],
                consumes_budget=True,
            )
        return judged
    if run:
        # No classifier on this call: the run stands on its own, as measured
        # on the logged contexts.
        return Decision(
            DISPOSITION_REFUSED,
            BASIS_AGENT_REPLIES_CUT,
            text=REFUSAL_TEXT[BASIS_AGENT_REPLIES_CUT],
            consumes_budget=True,
        )
    return Decision(DISPOSITION_ALLOWED, BASIS_NO_RUN)


def verdict_decision(verdict: str, *, outbound: bool) -> Decision:
    """The classifier's verdict as a decision; pure. A corroborated machine
    was allowed by the machine rule before the classifier ran, so a
    ``machine`` verdict here is uncorroborated and is treated as not
    closing. ``wrong_person`` ends only an outbound call (the policy's own
    rule is about calling a specific person); inbound it is not closing."""

    if verdict == VERDICT_CLOSING:
        return Decision(DISPOSITION_ALLOWED, BASIS_CLOSING)
    if verdict == VERDICT_WRONG_PERSON and outbound:
        return Decision(DISPOSITION_ALLOWED, BASIS_WRONG_PERSON)
    if verdict in (VERDICT_WRONG_PERSON, VERDICT_MACHINE, VERDICT_NOT_CLOSING):
        return Decision(
            DISPOSITION_REFUSED,
            BASIS_NOT_CLOSING,
            text=REFUSAL_TEXT[BASIS_NOT_CLOSING],
            consumes_budget=True,
        )
    return Decision(DISPOSITION_ALLOWED, BASIS_CLASSIFIER_UNAVAILABLE)


def parse_verdict(raw: str) -> str:
    """One of the four tokens, or ``unavailable``; case-insensitive, the
    first token of the answer, nothing inferred from a longer reply."""

    token = str(raw or "").strip().strip(".,!\"'`").split()
    if not token:
        return VERDICT_UNAVAILABLE
    word = token[0].strip(".,!\"'`").lower()
    return word if word in VERDICTS else VERDICT_UNAVAILABLE


@dataclass
class Outcome:
    """What one tool invocation got, for the worker and the telemetry."""

    disposition: str
    basis: str
    text: str
    words_held: bool
    attempt_id: str
    caller_turn_id: str
    deferred: bool
    inherited: bool = False
    verdict: str | None = None
    classifier_ms: int | None = None
    #: The recogniser language the decision ran under, the rule-9 threshold
    #: it resolved for it, and the confidence it read: what a later re-pin
    #: of the table needs per language.
    language: str = ""
    low_confidence_threshold: float | None = None
    last_final_confidence: float | None = None
    #: The classifier's input length (the caller's whole stretch), so a
    #: split answer judged whole is visible in the evidence; None when no
    #: request was made.
    exchange_chars: int | None = None


@dataclass(frozen=True)
class PostRefusal:
    """What the worker does to the model's context after a refusal, measured
    closed-loop on the live pre-hang-up contexts (N=60 per cell): the SDK's
    own tool follow-up is cancelled, the refused call and its output are
    dropped from the agent context, ``line`` takes their place (the SDK
    renders it to the model as an instructions block inside the caller's next
    turn), and, unless caller words are already held for the next turn, the
    reply is regenerated with tools off. On the hardest context this took the
    model's re-attempts from 240/240 turns (refusal kept, tool offered) to
    4/240 and spoken farewells without the tool to 4/240, with 0 quotes of
    the line in 1,200 turns."""

    attempt_ids: tuple[str, ...]
    line: str
    regenerate: bool
    #: The refused attempts' own "was my generation cut?" reads, re-checked
    #: right before a regeneration: a reply for a generation the caller has
    #: already cut is not owed.
    interrupted: tuple[Callable[[], bool], ...] = ()
    #: The caller-evidence revision at the refusal: a final that arrived after
    #: it is the caller's next words whatever the turn's provenance.
    evidence_revision: int = 0

    def without_regeneration(self) -> "PostRefusal":
        """The same context step, regenerating nothing: for a batch the close
        owner also refused on the caller's floor, where the caller is
        speaking and their own turn produces the next reply."""

        return replace(self, regenerate=False)

    def cut(self) -> bool:
        for read in self.interrupted:
            try:
                if read():
                    return True
            except Exception:  # noqa: BLE001 - an unreadable handle counts as cut
                return True
        return False


def caller_busy(snapshot: Snapshot, evidence: CallerEvidence, plan: PostRefusal) -> bool:
    """Whether a regenerated reply would speak over the caller: speaking, a
    held final, an unconsumed final on the turn, or (provenance-free, as the
    decision loop reads it) a final that arrived after the refusal."""

    if snapshot.user_speaking or snapshot.holds_final:
        return True
    if snapshot.unconsumed.get(EVIDENCE_FINAL, 0) > 0:
        return True
    return evidence.evidence_since(plan.evidence_revision).get(EVIDENCE_FINAL, 0) > 0


def exchange_from_history(messages: Sequence[Any], *, language: str) -> Exchange | None:
    """The last exchange for the classifier, from the agent's history in
    order: the caller's last STRETCH of consecutive user items (an answer the
    recogniser split across finals is one stretch, judged whole) joined in
    order, and the assistant line before that stretch (not the latest
    assistant item, which may be the reply that carried the tool call).
    ``None`` when no user item exists."""

    def _role(item: Any) -> str:
        return str(getattr(item, "role", "") or "")

    def _text(item: Any) -> str:
        return str(getattr(item, "text_content", None) or getattr(item, "content", "") or "")

    last = next((i for i in range(len(messages) - 1, -1, -1) if _role(messages[i]) == "user"), None)
    if last is None:
        return None
    first = last
    while first - 1 >= 0 and _role(messages[first - 1]) == "user":
        first -= 1
    preceding = next((messages[i] for i in range(first - 1, -1, -1) if _role(messages[i]) == "assistant"), None)
    caller_text = " ".join(part for part in (_text(messages[i]).strip() for i in range(first, last + 1)) if part)
    return Exchange(
        caller_text=caller_text,
        agent_text=_text(preceding) if preceding is not None else "",
        agent_interrupted=bool(getattr(preceding, "interrupted", False)) if preceding is not None else False,
        language=str(language or ""),
    )


def effective_guard_mode(guard_mode: str, silence_mode: str) -> tuple[str, str | None]:
    """The mode the guard runs in next to the mutual-silence rule, and why it
    moved. ``enforce`` needs the silence rule to be able to end a refused
    call that goes quiet (it promotes ``shadow`` to ``enforce``, and ``off``
    stays off), so with silence ``off`` the guard only logs."""

    if guard_mode == MODE_ENFORCE and silence_mode == "off":
        return MODE_SHADOW, "mutual_silence_off"
    return guard_mode, None


def replace_refused_items(
    items: Sequence[Any], attempt_ids: Sequence[str], line_item: Any
) -> tuple[list[Any], int | None]:
    """The agent context without the refused call/output items, ``line_item``
    (when given) where the first of them stood. ``None`` index when no item
    matched: the caller then leaves the context alone rather than adding a
    line to a context that still holds the pair."""

    wanted = {str(a) for a in attempt_ids if a}
    kept: list[Any] = []
    index: int | None = None
    for item in items:
        if str(getattr(item, "call_id", "") or "") in wanted:
            if index is None:
                index = len(kept)
                if line_item is not None:
                    kept.append(line_item)
            continue
        kept.append(item)
    return kept, index


UTTERANCE_EVENT = "agent_utterance"
ORIGIN_END_CALL_MESSAGE = "end_call_message"
ORIGIN_REGENERATION = "post_refusal_regeneration"
ORIGIN_WITH_END_CALL = "same_generation_as_end_call"
ORIGIN_REPLY = "reply"
ORIGIN_UNKNOWN = "unknown"
_FAREWELL_RE = re.compile(
    r"\b(goodbye|good-bye|bye|bye-bye|take care|have a (good|great|nice|wonderful|lovely|pleasant) (day|evening|night|one|weekend)"
    r"|thank you for calling|thanks for calling|talk to you (soon|later)|see you|farewell)\b",
    re.IGNORECASE,
)


def looks_like_farewell(text: str) -> bool:
    return bool(_FAREWELL_RE.search(str(text or "")))


class UtteranceAttribution:
    """Which mechanism produced each agent utterance, so a spoken goodbye on
    a call that then carries on can be traced to its source: the model's
    text in the same generation as an ``end_call`` attempt (spoken before
    the guard can decide), the post-refusal regeneration, the platform
    speaking ``end_call``'s ``message`` after an allow, or an ordinary
    reply. Fed at the SDK's speech seams by the worker; engine-agnostic."""

    def __init__(self, emit: Callable[..., None] = lambda *_a, **_k: None) -> None:
        self._emit = emit
        self._speeches: dict[str, dict[str, Any]] = {}
        self._attempt_speeches: set[str] = set()
        self._regenerations: set[str] = set()
        #: Regenerated speeches issued and not yet settled by their handle's
        #: completion; the teardown summary waits for them (bounded) and
        #: reports what stayed unsettled.
        self._regenerations_pending: set[str] = set()
        self._goodbye_says: set[str] = set()
        #: Speeches whose playout actually started (the agent-state
        #: "speaking" edge while that handle was current): a synthesis
        #: failure in the pinned SDK can leave a handle finished with no
        #: exception and no audio, so success is read from audio, not from
        #: the handle.
        self._spoken: set[str] = set()
        self.utterances_by_origin: dict[str, int] = {}
        self.farewells_by_origin: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []
        self._settled: set[str] = set()

    def note_speech(self, speech_id: str, *, source: str, user_initiated: bool, created_at: float) -> None:
        if speech_id:
            self._speeches[speech_id] = {"source": str(source or ""), "user_initiated": bool(user_initiated), "created_at": created_at}

    def note_attempt_speech(self, speech_id: str) -> None:
        if speech_id:
            self._attempt_speeches.add(speech_id)

    def note_regeneration(self, speech_id: str) -> None:
        if speech_id:
            self._regenerations.add(speech_id)
            self._regenerations_pending.add(speech_id)

    def note_regeneration_settled(self, speech_id: str) -> None:
        """The regenerated speech completed, failed or was cancelled: its
        outcome is now in the counters, so the summary may freeze."""

        self._regenerations_pending.discard(speech_id)

    @property
    def regenerations_pending(self) -> int:
        return len(self._regenerations_pending)

    def note_goodbye_say(self, speech_id: str) -> None:
        if speech_id:
            self._goodbye_says.add(speech_id)

    def note_speaking(self, speech_id: str) -> None:
        """The agent-state "speaking" edge while ``speech_id`` is current."""

        if speech_id:
            self._spoken.add(speech_id)

    def spoke(self, speech_id: str) -> bool:
        return speech_id in self._spoken

    def origin_of(self, speech_id: str, *, carried_end_call: bool, known: bool | None = None) -> str:
        if speech_id in self._goodbye_says:
            return ORIGIN_END_CALL_MESSAGE
        if speech_id in self._regenerations:
            return ORIGIN_REGENERATION
        if carried_end_call or speech_id in self._attempt_speeches:
            return ORIGIN_WITH_END_CALL
        if known if known is not None else speech_id in self._speeches:
            return ORIGIN_REPLY
        return ORIGIN_UNKNOWN

    def settle(self, speech_id: str, items: Sequence[Any], *, interrupted: bool, done_at: float) -> list[dict[str, Any]]:
        """The speech finished (or was cut): attribute every assistant text it
        produced and emit one ``agent_utterance`` per text. Idempotent per
        speech."""

        if not speech_id or speech_id in self._settled:
            return []
        self._settled.add(speech_id)
        speech = self._speeches.pop(speech_id, None)
        carried_end_call = any(
            str(getattr(item, "type", "") or "") == "function_call" and str(getattr(item, "name", "") or "") == "end_call"
            for item in items
        )
        origin = self.origin_of(speech_id, carried_end_call=carried_end_call, known=speech is not None)
        out: list[dict[str, Any]] = []
        for item in items:
            if str(getattr(item, "type", "") or "") != "message" or str(getattr(item, "role", "") or "") != "assistant":
                continue
            text = str(getattr(item, "text_content", None) or getattr(item, "content", "") or "")
            farewell = looks_like_farewell(text)
            record = {
                "speech_id": speech_id,
                "item_id": str(getattr(item, "id", "") or ""),
                "origin": origin,
                "source": (speech or {}).get("source", ""),
                "user_initiated": (speech or {}).get("user_initiated"),
                "carried_end_call": carried_end_call,
                "farewell": farewell,
                "interrupted": bool(interrupted),
                "spoke": speech_id in self._spoken,
                # Length and reading only: the agent's words can repeat what
                # the caller said (an address, a code), so no text leaves
                # here; the transcript record holds the words.
                "chars": len(text),
                "created_at": (speech or {}).get("created_at"),
                "done_at": done_at,
            }
            self.utterances_by_origin[origin] = self.utterances_by_origin.get(origin, 0) + 1
            if farewell:
                self.farewells_by_origin[origin] = self.farewells_by_origin.get(origin, 0) + 1
            self.records.append(record)
            del self.records[:-200]
            out.append(record)
            try:
                self._emit(UTTERANCE_EVENT, **record)
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "utterances_by_origin": dict(self.utterances_by_origin),
            "farewells_by_origin": dict(self.farewells_by_origin),
        }


async def _sleep(seconds: float) -> None:
    import asyncio  # loop-affine by contract; imported here to keep the module pure otherwise

    await asyncio.sleep(seconds)


@dataclass
class _TurnState:
    leader: Any = None  # asyncio.Future | None
    pinned: str | None = None


class EndCallGuard:
    """Per-call state and the restartable decision loop.

    ``read_snapshot`` is the worker's synchronous read of everything the loop
    needs; ``wait_for_listening`` awaits the caller's run end, bounded by the
    cap, and returns whether it ended; ``now`` is monotonic time. The guard
    holds the leader future per caller turn: a second ``end_call`` in the
    same generation awaits the first's outcome and inherits it.
    """

    def __init__(
        self,
        *,
        mode: str,
        budget: int,
        evidence: CallerEvidence,
        read_snapshot: Callable[[], Snapshot],
        wait_for_listening: Callable[[float], Awaitable[bool]],
        future_factory: Callable[[], Any],
        emit: Callable[..., None] = lambda *_a, **_k: None,
        on_refusal: Callable[[Outcome], None] = lambda _o: None,
        classifier: ClosingClassifier | None = None,
        read_exchange: Callable[[], Exchange | None] | None = None,
        active_language: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
        wait_for_floor: Callable[[float], Awaitable[Any]] | None = None,
    ) -> None:
        import time as _time

        #: Awaits a change of the caller's floor (words arriving), bounded by
        #: its argument; a plain sleep when the worker supplies none.
        self._wait_for_floor = wait_for_floor or _sleep
        self._classifier = classifier
        self._read_exchange = read_exchange
        #: The call's ACTIVE recogniser language, read per decision (a
        #: follow-caller switch mid-call moves rule 9 to that language's pin);
        #: None leaves rule 9 off.
        self._active_language = active_language
        #: Per language key: attempts that read a confidence, the refusals
        #: rule 9 made, and the lowest confidence seen; the re-pin evidence.
        self.low_confidence_by_language: dict[str, dict[str, Any]] = {}
        self._clock = clock or _time.monotonic
        self.classifier_calls = 0
        self.classifier_unavailable = 0
        #: Decisions that fell open without a request (deadline spent, no exchange).
        self.classifier_skipped = 0
        self._mode = mode if mode in MODES else MODE_SHADOW
        self._budget = max(1, int(budget))
        self._evidence = evidence
        self._read = read_snapshot
        self._wait_for_listening = wait_for_listening
        self._future_factory = future_factory
        self._emit = emit
        self._on_refusal = on_refusal
        self._turns: dict[str, _TurnState] = {}
        self._refused_turns: set[str] = set()
        self.attempts = 0
        self.refusals_by_basis: dict[str, int] = {}
        self.premature_allows = 0
        self.aborted = 0
        self.silence_promoted = False
        #: Set by the worker when the caller starts speaking while the goodbye
        #: plays after an ALLOW: the phase-2 tripwire.
        self.allowed_then_caller_spoke = False
        #: Per refused attempt id, the post-refusal step the worker performs,
        #: consumed once by its synchronous ``function_tools_executed``
        #: handler; and how those steps went.
        self._post_refusal: dict[str, PostRefusal] = {}
        self._interrupted_by_attempt: dict[str, Callable[[], bool]] = {}
        #: Which mechanism spoke each agent utterance (see the class).
        self.utterances = UtteranceAttribution(emit=emit)
        #: Steps planned (one per batch of refused attempts, a leader and its
        #: follower merged), applied and failed: the harness compares the
        #: applied count to the PLANNED count, never to attempts.
        self.post_refusal_planned = 0
        self.post_refusal_applied = 0
        self.post_refusal_failed = 0
        self.post_refusal_regeneration_failed = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def armed(self) -> bool:
        return self._mode != MODE_OFF

    @property
    def enforcing(self) -> bool:
        return self._mode == MODE_ENFORCE

    def _turn(self, turn_id: str) -> _TurnState:
        state = self._turns.get(turn_id)
        if state is None:
            state = _TurnState()
            self._turns[turn_id] = state
        return state

    async def evaluate(
        self,
        *,
        attempt_id: str,
        interrupted: Callable[[], bool] | None = None,
    ) -> Outcome:
        """Decide one tool invocation.

        ``interrupted`` reads the originating speech handle: the SDK shields a
        running tool and WAITS for it on a caller barge-in rather than
        cancelling it, so after every await the guard asks whether the
        generation that carried this call was cut. A cut generation decides
        nothing (``aborted``): the SDK generates no follow-up for it, and the
        caller's new words drive the next turn.
        """

        turn_id = self._evidence.turn_id or "no-turn"
        turn = self._turn(turn_id)
        self.attempts += 1
        if interrupted is not None:
            self._interrupted_by_attempt[attempt_id] = interrupted
        import asyncio  # loop-affine by contract; imported here to keep the module pure otherwise

        leader = turn.leader
        if leader is not None and not leader.done():
            try:
                outcome: Outcome = await asyncio.shield(leader)
            except asyncio.CancelledError:
                if not leader.done():
                    # THIS follower was cancelled (its own generation was
                    # cut), not the leader: nothing is decided for it, and
                    # the leader goes on for the others.
                    self._interrupted_by_attempt.pop(attempt_id, None)
                    raise
                # The leader was cut: this attempt re-elects itself rather
                # than inheriting a decision that was never made.
                if leader is turn.leader:
                    turn.leader = None
            except Exception:
                # The leader failed: re-elect for the same reason.
                if leader is turn.leader:
                    turn.leader = None
            else:
                inherited = Outcome(**{**outcome.__dict__, "attempt_id": attempt_id, "inherited": True})
                self._record(inherited)
                return inherited
        leader = self._future_factory()
        turn.leader = leader
        outcome: Outcome | None = None
        try:
            outcome = await self._run_loop(
                attempt_id=attempt_id, turn_id=turn_id, turn=turn, interrupted=interrupted
            )
        except BaseException as exc:
            # A real cancellation: nothing was decided, nothing is promoted
            # or budgeted; recorded as aborted before it continues on its way.
            self._emit_aborted(attempt_id=attempt_id, caller_turn_id=turn_id, error=type(exc).__name__)
            raise
        finally:
            # Leader ownership is settled here whatever happened above: a
            # follower shielded on this future sees a decision or a
            # cancellation (an aborted leader decided nothing, so followers
            # re-elect), never a future that stays pending forever.
            if not leader.done():
                if outcome is not None and outcome.disposition != DISPOSITION_ABORTED:
                    leader.set_result(outcome)
                else:
                    leader.cancel()
        assert outcome is not None
        if outcome.disposition == DISPOSITION_ABORTED:
            self._emit_aborted(attempt_id=attempt_id, caller_turn_id=turn_id, error="speech_interrupted")
            return outcome
        self._record(outcome)
        return outcome

    def _interrupted(self, interrupted: Callable[[], bool] | None) -> bool:
        if interrupted is None:
            return False
        try:
            return bool(interrupted())
        except Exception:  # noqa: BLE001 - an unreadable handle is not proof the caller is gone
            return False

    async def _run_loop(
        self,
        *,
        attempt_id: str,
        turn_id: str,
        turn: _TurnState,
        interrupted: Callable[[], bool] | None,
    ) -> Outcome:
        deferred = False
        defers = 0
        entered_at = self._clock()
        # The classifier's verdict, once obtained, consumed exactly once by
        # the pass that follows its await: the loop restarts so a hangup, a
        # platform ending, a held final or a live run since the call was
        # placed wins over it, and nothing classifies twice.
        pending_verdict: str | None = None
        classifier_ms: int | None = None
        exchange_chars: int | None = None
        # The revision at tool entry: caller evidence after THIS instant is
        # judged whatever the turn's provenance, so an anchor the hook could
        # not bind never hides a caller who resumed during an await.
        entry_revision = self._evidence.revision

        def _aborted() -> Outcome:
            self._interrupted_by_attempt.pop(attempt_id, None)
            return Outcome(
                disposition=DISPOSITION_ABORTED,
                basis="interrupted",
                text="",
                words_held=False,
                attempt_id=attempt_id,
                caller_turn_id=turn_id,
                deferred=deferred,
            )

        while True:
            if self._interrupted(interrupted):
                return _aborted()
            snapshot = self._read()
            since_entry = self._evidence.evidence_since(entry_revision)
            merged = {
                kind: max(snapshot.unconsumed.get(kind, 0), since_entry.get(kind, 0))
                for kind in (EVIDENCE_FINAL, EVIDENCE_INTERIM, EVIDENCE_VAD_EDGE)
            }
            snapshot = Snapshot(**{**snapshot.__dict__, "unconsumed": merged})
            classifier_armed = (
                self._classifier is not None
                and self._classifier.available
                and self._read_exchange is not None
                and pending_verdict is None
            )
            language, threshold = self._low_confidence_pin()
            elapsed = self._clock() - entered_at
            decision = decide(
                snapshot,
                pinned=turn.pinned,
                refused_turns=len(self._refused_turns),
                budget=self._budget,
                # The deadline, not a count, bounds the defers: while the
                # caller is speaking the loop waits for the run to end, up to
                # what the decision deadline leaves.
                deferred_once=(elapsed >= DECISION_DEADLINE_SECONDS - DEFER_MIN_SECONDS) or defers >= MAX_DEFERS,
                low_confidence_threshold=threshold,
                classifier_armed=classifier_armed,
                verdict=pending_verdict,
                turn_refused=turn_id in self._refused_turns,
            )
            if decision.deferred:
                deferred = True
                defers += 1
                remaining = DECISION_DEADLINE_SECONDS - (self._clock() - entered_at)
                if decision is DEFER_WORDLESS:
                    # Not the run's END: a wordless run may never end (a noisy
                    # line), and its words are what this waits for. Never past
                    # the window: no minimum wait here (the window's own end,
                    # the deadline and MAX_DEFERS bound the loop).
                    await self._wait_for_floor(
                        max(0.0, min(DEFER_CAP_SECONDS, remaining, snapshot.caller_wordless_remaining))
                    )
                    continue
                await self._wait_for_listening(max(DEFER_MIN_SECONDS, min(DEFER_CAP_SECONDS, remaining)))
                continue
            if decision is CLASSIFY:
                exchange = self._read_exchange() if self._read_exchange is not None else None
                exchange_chars = len(exchange.caller_text) if exchange is not None else None
                if exchange is not None and len(exchange.caller_text) > CALLER_TEXT_MAX_CHARS:
                    decision = Decision(
                        DISPOSITION_REFUSED,
                        BASIS_INPUT_OVERFLOW,
                        text=REFUSAL_TEXT[BASIS_INPUT_OVERFLOW],
                    )
                else:
                    remaining = DECISION_DEADLINE_SECONDS - (self._clock() - entered_at)
                    if exchange is None or remaining <= 0:
                        # No request was made: the deadline was spent by the
                        # awaits before it, or there is no exchange to judge.
                        pending_verdict = VERDICT_UNAVAILABLE
                        self.classifier_skipped += 1
                    else:
                        pending_verdict, classifier_ms = await self._classify(
                            exchange, timeout=min(CLASSIFIER_TIMEOUT_SECONDS, remaining)
                        )
                    # Restart: the structural rules re-read the world the
                    # await may have changed; the verdict is consumed below.
                    continue
            if self._interrupted(interrupted):
                # Revalidate before committing anything: the generation that
                # carried this call may have been cut while the loop ran.
                return _aborted()
            if decision.refused:
                if decision.consumes_budget:
                    self._refused_turns.add(turn_id)
                    if decision.basis in PINNED_BASES:
                        turn.pinned = decision.basis
            elif decision.basis in (BASIS_NO_RUN, BASIS_BUDGET_EXHAUSTED) and snapshot.cut_replies >= CUT_RUN_MIN:
                self.premature_allows += 1
            return Outcome(
                disposition=decision.disposition,
                basis=decision.basis,
                text=decision.text,
                words_held=decision.words_held,
                attempt_id=attempt_id,
                caller_turn_id=turn_id,
                deferred=deferred,
                verdict=pending_verdict,
                classifier_ms=classifier_ms,
                exchange_chars=exchange_chars,
                language=language,
                low_confidence_threshold=threshold,
                last_final_confidence=snapshot.last_final_confidence,
            )

    def _low_confidence_pin(self) -> tuple[str, float | None]:
        """The active language's table key and rule-9 threshold, or ``("", None)``."""

        if self._active_language is None:
            return "", None
        try:
            language = str(self._active_language() or "")
        except Exception:  # noqa: BLE001 - a language read cannot break the ending path
            return "", None
        key = low_confidence_language_key(language)
        return key, LOW_CONFIDENCE_BY_LANGUAGE.get(key)

    def _count_low_confidence(self, outcome: Outcome) -> None:
        if outcome.last_final_confidence is None or outcome.inherited:
            return
        key = outcome.language or "unknown"
        row = self.low_confidence_by_language.setdefault(
            key,
            {
                "threshold": outcome.low_confidence_threshold,
                "provisional": key in LOW_CONFIDENCE_PROVISIONAL,
                "attempts_with_confidence": 0,
                "refused": 0,
                "min_confidence": None,
            },
        )
        row["attempts_with_confidence"] += 1
        if outcome.disposition == DISPOSITION_REFUSED and outcome.basis == BASIS_CALLER_NOT_UNDERSTOOD:
            row["refused"] += 1
        seen = row["min_confidence"]
        if seen is None or outcome.last_final_confidence < seen:
            row["min_confidence"] = outcome.last_final_confidence

    async def _classify(self, exchange: Exchange, *, timeout: float) -> tuple[str, int]:
        """One classifier request under the remaining deadline; never raises.
        Timeout, error and an answer off the set are all ``unavailable``."""

        import asyncio

        assert self._classifier is not None
        self.classifier_calls += 1
        started = self._clock()
        try:
            raw = await asyncio.wait_for(self._classifier.classify(exchange, timeout=timeout), timeout=timeout)
            verdict = parse_verdict(raw)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException:  # noqa: BLE001 - every failure is the same fail-open, counted
            verdict = VERDICT_UNAVAILABLE
        if verdict == VERDICT_UNAVAILABLE:
            self.classifier_unavailable += 1
        return verdict, int((self._clock() - started) * 1000)

    def _emit_aborted(self, *, attempt_id: str, caller_turn_id: str, error: str) -> None:
        self.aborted += 1
        try:
            self._emit(
                DECISION_EVENT,
                attempt_id=attempt_id,
                caller_turn_id=caller_turn_id,
                mode=self._mode,
                basis="aborted",
                disposition=DISPOSITION_ABORTED,
                error_type=error,
            )
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass

    def _record(self, outcome: Outcome) -> None:
        disposition = outcome.disposition
        self._count_low_confidence(outcome)
        # Every outcome settles its attempt's cut-read; a refusal below
        # carries it into the plan, anything else drops it.
        interrupted_read = self._interrupted_by_attempt.pop(outcome.attempt_id, None)
        if disposition == DISPOSITION_REFUSED:
            self.refusals_by_basis[outcome.basis] = self.refusals_by_basis.get(outcome.basis, 0) + 1
            if not self.enforcing:
                disposition = DISPOSITION_WOULD_REFUSE
            else:
                if outcome.attempt_id:
                    # A budgeted refusal judged the caller not closing: the
                    # keep-helping line takes the pair's place and the reply
                    # is regenerated (the measured cell). A refusal made
                    # because the guard could NOT judge yet (the caller still
                    # speaking, words held) only drops the pair: the caller's
                    # words drive the next turn and the model may reach for
                    # the tool again, which the guard then judges in full.
                    judged = outcome.basis in JUDGED_BASES
                    self._post_refusal[outcome.attempt_id] = PostRefusal(
                        attempt_ids=(outcome.attempt_id,),
                        line=REFUSAL_TEXT.get(outcome.basis, REFUSAL_TEXT[BASIS_CALLER_WORDS_PENDING]) if judged else "",
                        regenerate=judged,
                        interrupted=tuple(read for read in (interrupted_read,) if read),
                        evidence_revision=self._evidence.revision,
                    )
                if not outcome.inherited:
                    # Once per generation: the follower shares the leader's
                    # refusal, its promotion and its silence episode.
                    try:
                        self._on_refusal(outcome)
                    except Exception:  # noqa: BLE001 - the ending path must not fail on the hook
                        pass
        snapshot = self._read()
        try:
            self._emit(
                DECISION_EVENT,
                attempt_id=outcome.attempt_id,
                caller_turn_id=outcome.caller_turn_id,
                anchor_revision=self._evidence.anchor,
                decision_revision=self._evidence.revision,
                mode=self._mode,
                basis=outcome.basis,
                disposition=disposition,
                inherited=outcome.inherited,
                deferred=outcome.deferred,
                cut_replies_of_last_4=snapshot.cut_replies,
                refused_turns=len(self._refused_turns),
                caller_turns=self._evidence.caller_turns,
                user_speaking=snapshot.user_speaking,
                held_final=snapshot.holds_final,
                provenance=self._evidence.provenance or "",
                last_final_confidence=snapshot.last_final_confidence,
                language_active=outcome.language,
                low_confidence_threshold=outcome.low_confidence_threshold,
                verdict=outcome.verdict or "",
                classifier_ms=outcome.classifier_ms,
                exchange_chars=outcome.exchange_chars,
            )
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass

    def refuses(self, outcome: Outcome) -> bool:
        """Whether this outcome is an actual refusal the worker must act on."""

        return self.enforcing and outcome.disposition == DISPOSITION_REFUSED

    @staticmethod
    def was_aborted(outcome: Outcome) -> bool:
        """Whether the generation that carried this call was cut mid-decision.
        (A method, never an attribute: ``self.aborted`` is the COUNT.)"""

        return outcome.disposition == DISPOSITION_ABORTED

    def post_refusal_plan(self, call_ids: list[str]) -> PostRefusal | None:
        """The post-refusal step for this batch of executed tool calls, or
        None when the batch carries no refusal. Consumed once per attempt; a
        batch with several refused attempts (a leader and its follower) gets
        one merged step that drops every one of them."""

        plans = [self._post_refusal.pop(call_id) for call_id in call_ids if call_id in self._post_refusal]
        if not plans:
            return None
        self.post_refusal_planned += 1
        return PostRefusal(
            attempt_ids=tuple(a for plan in plans for a in plan.attempt_ids),
            line=next((plan.line for plan in plans if plan.line), ""),
            regenerate=any(plan.regenerate for plan in plans),
            interrupted=tuple(read for plan in plans for read in plan.interrupted),
            evidence_revision=min(plan.evidence_revision for plan in plans),
        )

    def note_post_refusal(self, applied: bool) -> None:
        if applied:
            self.post_refusal_applied += 1
        else:
            self.post_refusal_failed += 1

    def note_regeneration_failed(self, error: str) -> None:
        """The SDK failed the regenerated reply after the step was applied:
        the caller never heard it, so the step counts as failed after all."""

        self.post_refusal_applied = max(0, self.post_refusal_applied - 1)
        self.post_refusal_failed += 1
        self.post_refusal_regeneration_failed += 1

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "budget": self._budget,
            "attempts": self.attempts,
            "refused_turns": len(self._refused_turns),
            "refusals_by_basis": dict(self.refusals_by_basis),
            "premature_allows": self.premature_allows,
            "aborted": self.aborted,
            "silence_promoted": self.silence_promoted,
            "allowed_then_caller_spoke": self.allowed_then_caller_spoke,
            "post_refusal_planned": self.post_refusal_planned,
            "post_refusal_applied": self.post_refusal_applied,
            "post_refusal_failed": self.post_refusal_failed,
            "post_refusal_regeneration_failed": self.post_refusal_regeneration_failed,
            # Refusals whose step was never consumed by the worker's handler:
            # a refusal without the measured post-refusal treatment.
            "post_refusal_outstanding": len(self._post_refusal),
            # Regenerations issued whose handle had not completed when the
            # summary was taken: their success or failure is unknown here.
            "post_refusal_unsettled": self.utterances.regenerations_pending,
            **self.utterances.summary(),
            "classifier_armed": bool(self._classifier is not None and self._classifier.available),
            "classifier_calls": self.classifier_calls,
            "classifier_unavailable": self.classifier_unavailable,
            "classifier_skipped": self.classifier_skipped,
            "low_confidence_by_language": {
                key: dict(row) for key, row in self.low_confidence_by_language.items()
            },
        }


# ---------------------------------------------------------------------------
# The close floor: who holds the floor while the agent's own close is pending
# ---------------------------------------------------------------------------
#
# Silero scores background noise as speech (0.95-0.99, ``playout_hold.py``),
# so a VAD run is not by itself the caller taking the floor; WORDS are. That
# reading is safe only where the transcriber is MEASURED to deliver a speaking
# caller's first word inside a known budget, which is the playout hold's own
# per-stack table (``playout_hold.NO_WORD_HOLD_BUDGETS``). The worker resolves
# that budget per call; a source without one (every other provider, model and
# language, and every voice-to-voice engine) keeps raw VAD exactly as before.

#: The three readings of the floor.
FLOOR_WORDS = "words"
FLOOR_WORDLESS = "wordless"
FLOOR_QUIET = "quiet"
#: Not a reading of the caller's floor: the call is waiting in a phone menu or
#: a hold queue and the model's own hang-up is refused (see
#: ``WorkerLocalEndCall``). Kept in the same vocabulary so the worker's
#: recovery cancels the SDK's automatic follow-up for it too.
FLOOR_NAVIGATION = "navigation"

#: VAD runs remembered for the windows below, newest last. A window spans one
#: budget (seconds), far fewer runs than this.
CLOSE_FLOOR_RUN_MEMORY = 32
#: Local floor refusals remembered until their tool batch is executed.
LOCAL_REFUSAL_MEMORY = 64


class CloseFloorEvidence:
    """The caller's floor around the agent's close, classified by words.

    One per call, owned by ``WorkerLocalEndCall`` and fed from both transcript
    feeds (the STT stream before the transcript gate holds anything, and the
    session's own ``user_input_transcribed``), the VAD edges and the caller
    turn commits. ``words`` is a non-empty interim or final that survives the
    call's backchannel filter: no duration floor and no VAD run required, so a
    clearly transcribed "wait" with no VAD edge counts, and a gate that is
    suspended or disabled is never read as words.
    """

    def __init__(self, *, is_words: Callable[[str], bool], clock: Callable[[], float]) -> None:
        self._is_words = is_words
        self._clock = clock
        #: Words events seen on this call; monotone.
        self.words = 0
        #: ``[onset, end]`` per VAD run, ``end`` None while open.
        self._runs: list[list[float | None]] = []
        #: The word count at the later of the open run's onset and the last
        #: committed caller turn: words before it belong to a turn the model
        #: already answered, not to the run now open.
        self._words_mark = 0

    @property
    def vad_open(self) -> bool:
        return bool(self._runs) and self._runs[-1][1] is None

    def note_vad(self, speaking: bool) -> None:
        now = float(self._clock())
        if speaking:
            if self.vad_open:
                return
            self._runs.append([now, None])
            del self._runs[:-CLOSE_FLOOR_RUN_MEMORY]
            self._words_mark = self.words
            return
        if self.vad_open:
            self._runs[-1][1] = now

    def is_words(self, text: str) -> bool:
        """The floor's reading of one transcript, without counting it."""

        if not str(text or "").strip():
            return False
        try:
            return bool(self._is_words(str(text)))
        except Exception:  # noqa: BLE001 - an unreadable filter protects the caller
            return True

    def note_transcript(self, text: str) -> bool:
        """Count one transcript event; whether it was the caller's words."""

        words = self.is_words(text)
        if words:
            self.words += 1
        return words

    def note_turn_committed(self) -> None:
        self._words_mark = self.words

    def run_has_words(self) -> bool:
        """An open run carrying words the model has not yet been handed."""

        return self.vad_open and self.words > self._words_mark

    def active_between(self, start: float, end: float) -> bool:
        """Whether any VAD run overlapped ``[start, end]``."""

        for onset, ended in self._runs:
            assert onset is not None
            if onset <= end and (ended is None or ended >= start):
                return True
        return False

    def first_onset_since(self, at: float) -> float | None:
        for onset, _ended in self._runs:
            assert onset is not None
            if onset >= at:
                return onset
        return None

    def settlement_start(
        self, *, playout_end: float, budget: float, farewell_start: float | None = None
    ) -> float | None:
        """Where the terminal settlement's window opens, or None (quiet).

        The later of the farewell's playout end and the first wordless onset
        in its final budget-length: a run open in that stretch (one already
        open at playout end included) opens it at playout end, since speech
        inside it owes its words by then plus the budget; otherwise the first
        onset after playout end opens it there. The stretch never reaches back
        before the farewell itself: a run that had already ENDED when the
        farewell started (the caller's own completed goodbye) holds nothing;
        one still open into it does.
        """

        since = playout_end - budget
        if farewell_start is not None:
            since = max(since, farewell_start)
        if self.active_between(since, playout_end):
            return playout_end
        return self.first_onset_since(playout_end)


class CloseFloorEntry:
    """The entry/guard-wait deferral of one ``end_call`` attempt.

    Words refuse. A wordless run (open at entry, starting after it, or a
    duration-only SDK interruption of the attempt's own generation) opens ONE
    window of the source's budget from its first onset; later edges never
    extend it, so the delay an attempt can add is bounded once. Separate from
    the terminal settlement: nothing here spends that window.
    """

    def __init__(self, floor: CloseFloorEvidence, *, budget: float, now: float) -> None:
        self._floor = floor
        self.budget = float(budget)
        self.entered_at = float(now)
        self._words_at_entry = floor.words
        self._open_at_entry = floor.vad_open
        self._interrupted_at: float | None = None
        self._start: float | None = None

    def note_wordless_interruption(self, now: float) -> None:
        if self._interrupted_at is None:
            self._interrupted_at = float(now)

    def has_words(self) -> bool:
        return self._floor.words > self._words_at_entry or self._floor.run_has_words()

    def window_start(self) -> float | None:
        if self._start is None:
            starts = [self.entered_at] if self._open_at_entry else []
            onset = self._floor.first_onset_since(self.entered_at)
            if onset is not None:
                starts.append(onset)
            if self._interrupted_at is not None:
                starts.append(self._interrupted_at)
            if starts:
                # Frozen: a repeated noise edge can never move it later.
                self._start = min(starts)
        return self._start

    def remaining(self, now: float) -> float:
        start = self.window_start()
        if start is None:
            return 0.0
        return max(0.0, start + self.budget - float(now))

    def state(self, now: float) -> str:
        if self.has_words():
            return FLOOR_WORDS
        if self.remaining(now) > 0:
            return FLOOR_WORDLESS
        return FLOOR_QUIET


#: The call's terminal direction, owned once (``TerminalIntent``).
INTENT_NONE = "none"
INTENT_CLOSING = "closing"
INTENT_TRANSFERRING = "transferring"


class TerminalIntent:
    """Compare-and-set arbitration between closing the call and transferring it.

    ``none`` becomes ``closing`` when an end_call is accepted (before its
    terminal task is scheduled) or ``transferring`` when the transfer tool is
    entered (before any control-plane work). A close is refused while a
    transfer is in flight or bridged; a transfer is refused while the call is
    closing. A reclaimed close or a failed transfer returns to ``none``. A
    platform ending overrides both (:meth:`force_close`). Loop-affine.
    """

    def __init__(self) -> None:
        self.state = INTENT_NONE

    def claim_close(self) -> bool:
        if self.state == INTENT_TRANSFERRING:
            return False
        self.state = INTENT_CLOSING
        return True

    def force_close(self) -> None:
        self.state = INTENT_CLOSING

    def release_close(self) -> None:
        if self.state == INTENT_CLOSING:
            self.state = INTENT_NONE

    def claim_transfer(self) -> bool:
        if self.state != INTENT_NONE:
            return False
        self.state = INTENT_TRANSFERRING
        return True

    def release_transfer(self) -> None:
        if self.state == INTENT_TRANSFERRING:
            self.state = INTENT_NONE


class LocalCloseRefusalRecovery:
    """The worker's recovery for the close owner's OWN floor refusals.

    The semantic guard's post-refusal step exists only where that guard is
    armed. ``WorkerLocalEndCall`` refuses on every engine and in every guard
    mode, and before this its refusal was a tool error the SDK answered with an
    automatic follow-up: the model said goodbye again and called end_call
    again, eight times on one noisy line. On a refused batch this cancels that
    follow-up (batch-wide: sibling tool outputs stand, their verbalisation goes
    with it) and regenerates nothing -- the caller is speaking, so their own
    turn produces the next reply, and end_call stays available on it. One
    recovery per batch however many attempts it refused.

    If no caller turn follows, ``on_refusal`` puts the existing mutual-silence
    rule on the clock (a no-op when that rule is ``off``, where the call ends
    by the endings that exist without it: the caller's next turn, the maximum
    call duration, the agent-unresponsive rule and remote hangup).
    """

    def __init__(
        self,
        *,
        on_refusal: Callable[[], None] = lambda: None,
        on_reclaimed: Callable[[], None] | None = None,
        emit: Callable[..., None] = lambda *_a, **_k: None,
    ) -> None:
        self._refused: dict[str, str] = {}
        self._on_refusal = on_refusal
        self._on_reclaimed = on_reclaimed if on_reclaimed is not None else on_refusal
        self._emit = emit
        self.recovered = 0

    def note_refusal(self, attempt_id: str, *, floor: str) -> None:
        attempt_id = str(attempt_id or "")
        if not attempt_id:
            return
        self._refused[attempt_id] = str(floor or "")
        while len(self._refused) > LOCAL_REFUSAL_MEMORY:
            del self._refused[next(iter(self._refused))]

    def note_reclaimed(self) -> None:
        """An accepted close was withdrawn; reuse the refusal's silence bound.

        Called once by the close owner after settlement, independently of tool
        output timing. Do not cancel a batch here: it may be the caller's new
        reply, and siblings may carry business results they still need.
        """
        try:
            self._on_reclaimed()
        except Exception:  # a recovery hook must not turn a reclaim into a hang-up
            pass
        try:
            self._emit("end_call_reclaim_recovery_requested")
        except Exception:
            pass

    def on_function_tools_executed(self, event: Any) -> bool:
        """SYNCHRONOUS, from the SDK's ``function_tools_executed`` emitter."""

        call_ids = [
            str(getattr(call, "call_id", "") or "") for call in (getattr(event, "function_calls", None) or [])
        ]
        floors = [self._refused.pop(call_id) for call_id in call_ids if call_id in self._refused]
        if not floors:
            return False
        try:
            event.cancel_tool_reply()
        except Exception:  # noqa: BLE001 - the SDK's follow-up then plays, as before this
            pass
        if any(floor != FLOOR_NAVIGATION for floor in floors):
            # A refusal on the caller's floor puts the silence rule on the
            # clock. A refusal because the call is waiting in a menu or a queue
            # must not: it would promote that rule for the rest of the call,
            # and the queue is not a caller who went quiet.
            try:
                self._on_refusal()
            except Exception:  # noqa: BLE001 - the ending path must not fail on the hook
                pass
        self.recovered += 1
        try:
            self._emit(
                "end_call_local_refusal_recovered",
                attempts=len(floors),
                floor=FLOOR_WORDS if FLOOR_WORDS in floors else (floors[0] or ""),
            )
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass
        return True


__all__ = [
    "BASIS_AGENT_REPLIES_CUT",
    "BASIS_BUDGET_EXHAUSTED",
    "BASIS_CALLER_STILL_SPEAKING",
    "BASIS_CALLER_WORDS_PENDING",
    "BASIS_MACHINE",
    "BASIS_NO_RUN",
    "BUDGET_ENV",
    "JUDGED_BASES",
    "CUT_RUN_MIN",
    "CUT_RUN_WINDOW",
    "CALLER_TEXT_MAX_CHARS",
    "CLASSIFIER_TIMEOUT_SECONDS",
    "CallerEvidence",
    "CloseFloorEntry",
    "CloseFloorEvidence",
    "ClosingClassifier",
    "FLOOR_NAVIGATION",
    "FLOOR_QUIET",
    "FLOOR_WORDLESS",
    "FLOOR_WORDS",
    "INTENT_CLOSING",
    "INTENT_NONE",
    "INTENT_TRANSFERRING",
    "LocalCloseRefusalRecovery",
    "TerminalIntent",
    "DECISION_DEADLINE_SECONDS",
    "DECISION_EVENT",
    "DEFAULT_BUDGET",
    "DEFER_CAP_SECONDS",
    "Decision",
    "EndCallGuard",
    "Exchange",
    "LOW_CONFIDENCE_BY_LANGUAGE",
    "PostRefusal",
    "caller_busy",
    "UtteranceAttribution",
    "UTTERANCE_EVENT",
    "looks_like_farewell",
    "effective_guard_mode",
    "exchange_from_history",
    "replace_refused_items",
    "LOW_CONFIDENCE_PROVISIONAL",
    "low_confidence_language_key",
    "low_confidence_provisional",
    "MODE_ENFORCE",
    "MODE_ENV",
    "MODE_OFF",
    "MODE_SHADOW",
    "Outcome",
    "PINNED_BASES",
    "REFUSAL_TEXT",
    "SUMMARY_EVENT",
    "Snapshot",
    "VERDICTS",
    "VERDICT_UNAVAILABLE",
    "cut_replies_of_last",
    "decide",
    "parse_verdict",
    "verdict_decision",
    "has_cut_run",
    "low_confidence_threshold_for",
    "resolve_budget",
    "resolve_mode",
]

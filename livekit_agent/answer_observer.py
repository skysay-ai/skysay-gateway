# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What answered an outbound call, and what each voicemail policy would do.

The post-answer voicemail observer.  It consumes typed signals produced by
``voicemail_signals.py`` (through the worker's ``voicemail_watch.py``) and
never sees audio, transcripts or phone numbers.  It decides a CLASSIFICATION
and, for each voicemail policy, the action that policy WOULD take and when.

In the first release every agent runs it in shadow: nothing here ends a call,
mutes anything or reaches the call's outcome.  Its record is evidence for the
Calls page and for the accuracy audit that gates the acting modes.  The
arbiter (``post_answer_arbiter.py``) stays the only thing that ends a call;
when an acting mode ships, this decision reaches it as one more signal.

The rules, and why (founder decisions and two adversarial reviews, 2026-09-23)
----------------------------------------------------------------------------

* **Only mailbox-specific evidence is proof**: a validated record beep after
  verified speech, or a strong mailbox phrase.  A fast answer, a long greeting,
  music, IVR, screener and "unavailable" prompts are suspicion or states.
* **Unsure is human.**  There are two tiers of human evidence.  CONFIRMED
  (a second responsive reply with words, or speech after a validated beep) is
  a permanent veto.  PROVISIONAL (a "hello" and a pause, a reply-timed run) is
  a shape a recording can also produce, so it is not a veto -- but proof that
  follows it is a ``conflict``, and a conflict never acts.  The arbiter's own
  human verdict is not used: it counts a recording's words as a person's.
* **Deadlines are separate.**  Proof must be KNOWN inside the classification
  window.  Human evidence is accepted until the record is frozen, so a person
  who appears after a would-be action is counted: that is the number the
  accuracy audit exists for.  ``unknown`` continues the call; there is no
  "unresolved" ending.
* **Each policy has its own instant.**  hang_up would act when proof is known.
  leave_message would start at the END of a candidate beep -- the founder's
  300 ms rule -- which is before the beep can be validated (a second beep
  within four seconds withdraws it).  So that start is SPECULATIVE, and a
  withdrawn candidate is recorded with the exposure it would have caused.
* **Only an answer with carrier evidence is eligible.**  A fail-open answer
  disposition is recorded, and its counterfactual is ``ineligible``.
* **Decisive facts are never evicted.**  The display history is bounded and
  counted; the first proof, the first human evidence of each tier and the
  states seen are kept apart from it.

Times: every signal carries ``at`` (media or ingestion time, for ordering and
display) and ``known_at`` (when the detector could first act on it).  Policy
instants use ``known_at``.  Both are on the worker's monotonic clock; the
record reports offsets from the answer in milliseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import shape only
    from . import voicemail_signals as _signals
except ImportError:  # pragma: no cover - direct execution inside the image
    import voicemail_signals as _signals  # type: ignore[no-redef]


CLASSIFICATION_WINDOW_SECONDS = 30.0
#: How long leave_message may wait for a record beep after proof.
CUE_WINDOW_SECONDS = 30.0
#: The playout cap a leave_message clip is held to; a person heard inside it
#: would have been talked over.
MESSAGE_PLAYOUT_SECONDS = 30.0
MAX_HISTORY = 64
#: Slots reserved for decisive signals, so a flood of ordinary ones can never
#: crowd them out and the total never exceeds ``MAX_HISTORY``.
MAX_PINNED = 16
MAX_CUES = 8
MAX_HUMAN_FACTS = 16

CLASS_HUMAN = "human"
CLASS_VOICEMAIL = "voicemail"
CLASS_CONFLICT = "conflict"
CLASS_IVR_OR_QUEUE = "ivr_or_queue"
CLASS_SCREENER = "screener"
CLASS_UNAVAILABLE = "unavailable"
CLASS_UNKNOWN = "unknown"
CLASSIFICATIONS = (
    CLASS_HUMAN,
    CLASS_VOICEMAIL,
    CLASS_CONFLICT,
    CLASS_IVR_OR_QUEUE,
    CLASS_SCREENER,
    CLASS_UNAVAILABLE,
    CLASS_UNKNOWN,
)

_STATE_CLASS = {
    _signals.KIND_IVR_PHRASE: CLASS_IVR_OR_QUEUE,
    _signals.KIND_SCREENER_PHRASE: CLASS_SCREENER,
    _signals.KIND_UNAVAILABLE_PHRASE: CLASS_UNAVAILABLE,
}

POLICY_HANG_UP = "hang_up"
POLICY_LEAVE_MESSAGE = "leave_message"

OUTCOME_WOULD_ACT = "would_act"
OUTCOME_WOULD_PLAY = "would_play"
OUTCOME_WOULD_PLAY_WITHDRAWN = "would_play_withdrawn"
OUTCOME_NO_CUE = "no_cue"
OUTCOME_CANCELLED_BY_HUMAN = "cancelled_by_human"
OUTCOME_CONFLICT = "conflict"
OUTCOME_HUMAN_FIRST = "human_first"
OUTCOME_NO_PROOF = "no_proof"
OUTCOME_INELIGIBLE = "ineligible"
OUTCOMES = (
    OUTCOME_WOULD_ACT,
    OUTCOME_WOULD_PLAY,
    OUTCOME_WOULD_PLAY_WITHDRAWN,
    OUTCOME_NO_CUE,
    OUTCOME_CANCELLED_BY_HUMAN,
    OUTCOME_CONFLICT,
    OUTCOME_HUMAN_FIRST,
    OUTCOME_NO_PROOF,
    OUTCOME_INELIGIBLE,
)

CUE_VALIDATED = "validated"
CUE_WITHDRAWN = "withdrawn"
#: The tone itself held, but the speech before it could not be verified
#: (no words bound to it, no classifier reading).  Recorded so the speculative
#: start a later release would have made is visible; never startable here.
CUE_UNVERIFIED = "unverified"
CUE_OUTCOMES = (CUE_VALIDATED, CUE_WITHDRAWN, CUE_UNVERIFIED)


@dataclass(frozen=True, slots=True)
class AnswerSignal:
    """One typed observation.  ``detail`` is a closed label, never caller text."""

    kind: str
    at: float
    known_at: float
    detail: str = ""
    frequency_hz: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _signals.SIGNAL_STRENGTH:
            raise ValueError(f"unknown signal kind: {self.kind!r}")
        for name in ("at", "known_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"signal {name} must be a finite number")
        if len(self.detail) > 48:
            raise ValueError("signal detail must be a short label")

    @property
    def strength(self) -> str:
        return _signals.SIGNAL_STRENGTH[self.kind]


@dataclass(frozen=True, slots=True)
class CueCandidate:
    """A tone that a speculative leave_message would have started at.

    ``at`` is the tone's end (the would-be playout start).  ``known_at`` is
    when its fate was known; ``outcome`` is validated or withdrawn.
    """

    at: float
    known_at: float
    outcome: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in CUE_OUTCOMES:
            raise ValueError(f"unknown cue outcome: {self.outcome!r}")


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    policy: str
    outcome: str
    #: When the action would have happened (hang up, or playout start).
    action_at: float | None = None
    #: A confirmed person was heard after the action instant (within the
    #: playout for leave_message).  The number the accuracy audit gates on.
    cut_off_human: bool = False
    #: Provisional human evidence came after the action instant.
    provisional_human_after: bool = False
    #: How long a withdrawn speculative playout would have run, in seconds.
    exposure_seconds: float | None = None


@dataclass(slots=True)
class _Facts:
    first_proof: AnswerSignal | None = None
    provisional_humans: list[AnswerSignal] = field(default_factory=list)
    confirmed_humans: list[AnswerSignal] = field(default_factory=list)
    states: dict[str, AnswerSignal] = field(default_factory=dict)
    latest_state: AnswerSignal | None = None
    late_proofs: int = 0
    #: IVR/queue phrases seen, ever: a monotone count, because
    #: ``classification()`` cannot say so once a human-shaped fact exists.
    ivr_phrases: int = 0
    #: Confirmed-tier human signals seen, ever: uncapped, unlike the list above
    #: (kept bounded for the audit). A plateau would silence the only automatic
    #: exit from a hold on a call that had a long back-and-forth first.
    confirmed_humans_seen: int = 0


class AnswerObserver:
    """The per-call state machine.  Pure; bounded memory; never raises on data
    it produced itself (a malformed signal is refused at construction)."""

    def __init__(
        self,
        *,
        armed_at: float,
        eligible: bool,
        classification_window_seconds: float = CLASSIFICATION_WINDOW_SECONDS,
    ) -> None:
        self._armed_at = float(armed_at)
        self._eligible = bool(eligible)
        self._window = float(classification_window_seconds)
        self._facts = _Facts()
        self._history: list[AnswerSignal] = []
        self._pinned: list[AnswerSignal] = []
        self._dropped = 0
        self._source_count = 0
        self._cues: list[CueCandidate] = []
        self._cues_dropped = 0

    @property
    def armed_at(self) -> float:
        return self._armed_at

    @property
    def eligible(self) -> bool:
        return self._eligible

    @property
    def ivr_phrase_count(self) -> int:
        """IVR/queue phrases seen so far (a monotone count).

        The menu signal for navigation. ``classification()`` is NOT it: it
        reports ``human`` for the rest of the call as soon as ANY human-shaped
        fact exists, and a menu read after a short greeting-shaped phrase
        records one first.
        """

        return self._facts.ivr_phrases

    @property
    def confirmed_human_count(self) -> int:
        """Confirmed-tier human facts seen so far (a monotone latch count)."""

        return self._facts.confirmed_humans_seen

    @property
    def has_proof(self) -> bool:
        return self._facts.first_proof is not None

    @property
    def proof_kind(self) -> str:
        proof = self._facts.first_proof
        return "" if proof is None else proof.kind

    @property
    def proof_known_at(self) -> float | None:
        proof = self._facts.first_proof
        return None if proof is None else proof.known_at

    @property
    def provisional_humans(self) -> tuple[AnswerSignal, ...]:
        """Durable provisional human facts (never evicted with the history)."""

        return tuple(self._facts.provisional_humans)

    def observe(self, signal: AnswerSignal) -> None:
        self._source_count += 1
        decisive = self._record_fact(signal)
        if decisive and len(self._pinned) < MAX_PINNED:
            self._pinned.append(signal)
            return
        if len(self._history) >= MAX_HISTORY - MAX_PINNED:
            self._dropped += 1
            return
        self._history.append(signal)

    def note_cue(self, cue: CueCandidate) -> None:
        if len(self._cues) >= MAX_CUES:
            self._cues_dropped += 1
            return
        self._cues.append(cue)

    # -- facts ---------------------------------------------------------

    def _record_fact(self, signal: AnswerSignal) -> bool:
        facts = self._facts
        strength = signal.strength
        if strength == _signals.STRENGTH_PROOF:
            if signal.known_at - self._armed_at > self._window:
                facts.late_proofs += 1
                return False
            if facts.first_proof is None or signal.known_at < facts.first_proof.known_at:
                facts.first_proof = signal
                return True
            return False
        if strength == _signals.STRENGTH_HUMAN:
            facts.confirmed_humans_seen += 1
            if len(facts.confirmed_humans) < MAX_HUMAN_FACTS:
                facts.confirmed_humans.append(signal)
                return len(facts.confirmed_humans) <= 2
            return False
        if strength == _signals.STRENGTH_PROVISIONAL_HUMAN:
            if len(facts.provisional_humans) < MAX_HUMAN_FACTS:
                facts.provisional_humans.append(signal)
                return len(facts.provisional_humans) <= 2
            return False
        if strength == _signals.STRENGTH_STATE:
            if signal.kind == _signals.KIND_IVR_PHRASE:
                facts.ivr_phrases += 1
            first = signal.kind not in facts.states
            facts.states.setdefault(signal.kind, signal)
            if facts.latest_state is None or signal.known_at >= facts.latest_state.known_at:
                facts.latest_state = signal
            return first
        return False

    # -- decision ------------------------------------------------------

    def classification(self) -> str:
        facts = self._facts
        proof = facts.first_proof
        if proof is not None:
            if any(item.known_at <= proof.known_at for item in facts.confirmed_humans):
                return CLASS_HUMAN
            if any(item.known_at <= proof.known_at for item in facts.provisional_humans):
                return CLASS_CONFLICT
            return CLASS_VOICEMAIL
        if facts.confirmed_humans or facts.provisional_humans:
            return CLASS_HUMAN
        if facts.latest_state is not None:
            return _STATE_CLASS[facts.latest_state.kind]
        return CLASS_UNKNOWN

    def _not_voicemail_outcome(self, classification: str) -> str:
        if classification == CLASS_CONFLICT:
            return OUTCOME_CONFLICT
        if classification == CLASS_HUMAN:
            return OUTCOME_HUMAN_FIRST
        return OUTCOME_NO_PROOF

    def hang_up(self) -> PolicyOutcome:
        classification = self.classification()
        if classification != CLASS_VOICEMAIL:
            return PolicyOutcome(POLICY_HANG_UP, self._not_voicemail_outcome(classification))
        if not self._eligible:
            return PolicyOutcome(POLICY_HANG_UP, OUTCOME_INELIGIBLE)
        proof = self._facts.first_proof
        assert proof is not None
        action_at = proof.known_at
        return PolicyOutcome(
            POLICY_HANG_UP,
            OUTCOME_WOULD_ACT,
            action_at=action_at,
            cut_off_human=any(item.known_at > action_at for item in self._facts.confirmed_humans),
            provisional_human_after=any(
                item.known_at > action_at for item in self._facts.provisional_humans
            ),
        )

    def leave_message(self) -> PolicyOutcome:
        classification = self.classification()
        if classification != CLASS_VOICEMAIL:
            return PolicyOutcome(POLICY_LEAVE_MESSAGE, self._not_voicemail_outcome(classification))
        if not self._eligible:
            return PolicyOutcome(POLICY_LEAVE_MESSAGE, OUTCOME_INELIGIBLE)
        proof = self._facts.first_proof
        assert proof is not None
        cue = self._startable_cue(proof)
        if cue is None:
            return PolicyOutcome(POLICY_LEAVE_MESSAGE, OUTCOME_NO_CUE)
        start = cue.at
        humans_before = [
            item for item in self._facts.confirmed_humans + self._facts.provisional_humans
            if item.known_at <= start
        ]
        if humans_before:
            return PolicyOutcome(POLICY_LEAVE_MESSAGE, OUTCOME_CANCELLED_BY_HUMAN, action_at=start)
        if cue.outcome == CUE_WITHDRAWN:
            # The playout that WOULD have run until the withdrawal was known:
            # a person heard inside it was talked over (Codex code review #10).
            stop = max(start, cue.known_at)
            return PolicyOutcome(
                POLICY_LEAVE_MESSAGE,
                OUTCOME_WOULD_PLAY_WITHDRAWN,
                action_at=start,
                exposure_seconds=round(stop - start, 3),
                cut_off_human=any(start < item.known_at <= stop for item in self._facts.confirmed_humans),
                provisional_human_after=any(
                    start < item.known_at <= stop for item in self._facts.provisional_humans
                ),
            )
        end = start + MESSAGE_PLAYOUT_SECONDS
        return PolicyOutcome(
            POLICY_LEAVE_MESSAGE,
            OUTCOME_WOULD_PLAY,
            action_at=start,
            cut_off_human=any(start < item.known_at <= end for item in self._facts.confirmed_humans),
            provisional_human_after=any(
                start < item.known_at <= end for item in self._facts.provisional_humans
            ),
        )

    def _startable_cue(self, proof: AnswerSignal) -> CueCandidate | None:
        """The first candidate beep a leave_message could have started on.

        A beep proof carries its own tone: its candidate ends at the proof's
        ``at``.  A phrase proof can only start on a candidate that ends after
        the phrase was KNOWN, inside the cue window.
        """

        for cue in sorted(self._cues, key=lambda item: item.at):
            if cue.outcome == CUE_UNVERIFIED:
                continue
            if proof.kind == _signals.KIND_RECORD_BEEP:
                if abs(cue.at - proof.at) <= 0.05:
                    return cue
                continue
            if proof.known_at <= cue.at <= proof.known_at + CUE_WINDOW_SECONDS:
                return cue
        return None

    def _human_during(self, start: float, stop: float) -> bool:
        return any(
            start < item.known_at <= stop
            for item in self._facts.confirmed_humans + self._facts.provisional_humans
        )

    # -- record --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The bounded, typed record.  Offsets in ms from the answer."""

        def offset(value: float | None) -> int | None:
            if value is None:
                return None
            return int(round((value - self._armed_at) * 1000))

        def policy(outcome: PolicyOutcome) -> dict[str, Any]:
            return {
                "outcome": outcome.outcome,
                "action_after_ms": offset(outcome.action_at),
                "cut_off_human": outcome.cut_off_human,
                "provisional_human_after": outcome.provisional_human_after,
                "exposure_ms": (
                    None
                    if outcome.exposure_seconds is None
                    else int(round(outcome.exposure_seconds * 1000))
                ),
            }

        proof = self._facts.first_proof
        signals = sorted(self._pinned + self._history, key=lambda item: (item.at, item.known_at))
        pinned = {id(item) for item in self._pinned}
        return {
            "classification": self.classification(),
            "eligible": self._eligible,
            "proof_kind": proof.kind if proof is not None else "",
            "proof_after_ms": offset(proof.known_at) if proof is not None else None,
            "hang_up": policy(self.hang_up()),
            "leave_message": policy(self.leave_message()),
            "late_proofs": self._facts.late_proofs,
            "signal_count": self._source_count,
            "signals_dropped": self._dropped,
            "cues_dropped": self._cues_dropped,
            "signals": [
                {
                    "kind": item.kind,
                    "strength": item.strength,
                    "detail": item.detail,
                    "after_ms": offset(item.at),
                    "known_after_ms": offset(item.known_at),
                    "frequency_hz": int(item.frequency_hz),
                    "duration_ms": int(item.duration_ms),
                    "pinned": id(item) in pinned,
                }
                for item in signals
            ],
            "cues": [
                {
                    "after_ms": offset(cue.at),
                    "known_after_ms": offset(cue.known_at),
                    "outcome": cue.outcome,
                    "reason": cue.reason,
                    # Whether a person was heard while a speculative playout
                    # started at this cue would have been running -- whatever
                    # the call was finally classified as (Codex stamp #4).
                    "human_heard": self._human_during(cue.at, max(cue.at, cue.known_at) if cue.outcome != CUE_VALIDATED else cue.at + MESSAGE_PLAYOUT_SECONDS),
                }
                for cue in sorted(self._cues, key=lambda item: item.at)
            ],
        }


__all__ = [
    "AnswerObserver",
    "AnswerSignal",
    "CLASSIFICATIONS",
    "CLASSIFICATION_WINDOW_SECONDS",
    "CUE_OUTCOMES",
    "CUE_UNVERIFIED",
    "CUE_VALIDATED",
    "CUE_WITHDRAWN",
    "CUE_WINDOW_SECONDS",
    "CueCandidate",
    "MESSAGE_PLAYOUT_SECONDS",
    "OUTCOMES",
    "POLICY_HANG_UP",
    "POLICY_LEAVE_MESSAGE",
    "PolicyOutcome",
]

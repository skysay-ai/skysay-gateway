# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The one thing that decides how an answered call ends, once it is answered.

Four signals can each conclude that an outbound call should stop after the
carrier reports an answer: a person spoke, the carrier dropped the leg, nobody
said anything at all, or the far end is a caller tune that will play until
somebody pays for it.  Before this module the first was implicit, the second
and third were rungs on the watchdog ladder in ``production.py``, and the
fourth did not exist -- which is how two calls on 2026-09-15 ran 180 s against
a subscriber's ring-back tune and settled as conversations.

Adding a second watchdog for the fourth signal would have been the wrong
shape, and the adversarial review of 2026-09-15 (findings 2, 3, 7, 8) is a
list of the ways it fails: two policies with different minimums racing to
stamp different reasons on the same call, a silence rule ending a call at 20 s
that a music rule was holding for a person who picks up at 40 s, and two
terminal records where the contract promises one.  So there is one arbiter.
It is given every post-answer signal, it applies one precedence, and it
produces at most one decision.

Precedence (founder decision, 2026-09-15)
-----------------------------------------

``human evidence > carrier disconnect > silent > non-human cap``

* **Human evidence** outranks everything and is accepted at ANY time up to the
  terminal commit.  It cancels the cap, clears suppression and buys one
  re-greeting.  It never ends a call; it prevents this module from ending one.
* **Carrier disconnect** is somebody else's ending.  The watchdog ladder owns
  it, above this module, and once it is observed the arbiter stops producing
  decisions of its own -- a call cannot both be ended by the far end and
  ended by us.
* **Silent** is the existing ``no_caller_speech_after_answer`` rule, unchanged:
  the same 20 s default and the same 10 s floor.
* **Non-human cap** is last.

That order is why a non-human verdict SUPPRESSES the silent branch rather than
losing to it, which looks backwards against the list until you read what the
two claims are.  ``silent`` asserts that nothing was heard.  A non-human
verdict asserts that at least 8 s of continuous audio WAS heard.  They are
contradictory readings of the same evidence, not two rules racing, and the
verdict is the one with the measurement behind it.  The ordering above governs
signals that can genuinely coincide, and it guarantees the thing that matters:
this module never DELAYS today's silence protection.  A call with no verdict
is ended at 20 s exactly as it is today.

Arming
------

Only ``sip.callStatus`` evidence arms the arbiter: the gate's ``ANSWERED``
(the attribute reached ``active`` while we waited) and ``ALREADY_ACTIVE`` (it
was ``active`` when we looked).  The two fail-open dispositions --
``NOT_REPORTED``, where the runtime never published the attribute, and
``TIMED_OUT``, where it never reached ``active`` -- classify in SHADOW and can
never end a call, whatever the mode says.  There is no answer evidence to
measure a window from on those calls, and a detector that ends a call on a
clock it cannot anchor is measuring ring-back.

A second arming is a no-op.  The attribute can flap or be published twice, and
a clock that restarts on the second event would give a tune a fresh 45 s every
time the carrier republished an attribute.

Modes
-----

``off`` collects nothing.  ``shadow`` -- the default -- classifies, records
what it WOULD have done, and takes no action whatsoever: no suppression, no
re-greeting, no ending, and the watchdog ladder keeps its own silence rule
byte for byte.  ``end_call`` acts.  Shadow exists because the two available
defaults were both unacceptable: on by default lets an unproven classifier
hang up on a real customer on a speakerphone, and off by default leaves the
tune calls billing for minutes with nothing measured.

One terminal record
-------------------

``commit`` is called once, with the reason that actually ended the call
whoever decided it, and returns the record that is written to the call row,
``call.completed``, results and groups.  The classification in that record is
``human`` if human evidence ever arrived before the commit, whatever ended the
call -- that is what "human outranks a carrier disconnect" means for the
record, as opposed to for the decision.  Nothing reopens after a commit: late
evidence is counted so an operator can see it happened, and applied to
nothing.

Relationship to ``answer_observer.py``
-------------------------------------

That module is the parent answer-detection work's policy state machine for
voicemail, IVR menus and call screeners.  It is in the tree and nothing
imports it yet.  The two are not alternatives and must not both become the
post-answer authority: this arbiter owns the TERMINAL decision and the
existing idempotent end-call seam, and when the observer is wired its
classification should arrive here as one more signal -- the same way a
carrier disconnect does -- rather than ending calls on its own.  Two state
machines each ending the same call for its own reason is the exact failure
the single-arbiter shape exists to prevent.

This module imports no livekit and holds no I/O.  It takes signals and
returns decisions.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import shape only
    from . import answer_classifier as _classifier
    from .sip_answer_gate import (
        ALREADY_ACTIVE,
        ANSWERED,
        DISCONNECTED,
        NOT_APPLICABLE,
        NOT_REPORTED,
        TIMED_OUT,
    )
except ImportError:  # pragma: no cover - direct execution inside the image
    import answer_classifier as _classifier  # type: ignore[no-redef]
    from sip_answer_gate import (  # type: ignore[no-redef]
        ALREADY_ACTIVE,
        ANSWERED,
        DISCONNECTED,
        NOT_APPLICABLE,
        NOT_REPORTED,
        TIMED_OUT,
    )


AnswerClassification = _classifier.AnswerClassification

#: Real answer evidence: ``sip.callStatus`` was seen at ``active``.  Both
#: spellings of that observation arm, because they differ only in whether the
#: attribute was already set when this worker looked.
ARMING_DISPOSITIONS = frozenset({ANSWERED, ALREADY_ACTIVE})
#: The gate's two fail-open dispositions.  They speak, so the call happens;
#: they carry no answer evidence, so this module only ever watches.
SHADOW_ONLY_DISPOSITIONS = frozenset({NOT_REPORTED, TIMED_OUT})
#: Every disposition the answer gate can return.  A string outside this set is
#: not recorded at all: the second-arming rule latches the first disposition
#: forever, so accepting a damaged one would leave a real answer unable to arm
#: the call it belongs to.
KNOWN_DISPOSITIONS = ARMING_DISPOSITIONS | SHADOW_ONLY_DISPOSITIONS | frozenset(
    {DISCONNECTED, NOT_APPLICABLE}
)

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_END_CALL = "end_call"
#: The per-agent setting's closed choice set.  Ordered for a UI that shows the
#: default first.
MODES = (MODE_SHADOW, MODE_END_CALL, MODE_OFF)
DEFAULT_MODE = MODE_SHADOW

#: This worker's existing silence rule, repeated here rather than imported so
#: that a 6,000-line worker module does not become a dependency of a pure
#: decision module.  ``tests/test_post_answer_arbiter.py`` pins both strings
#: against ``apps.voice.production``, so the copy cannot drift in silence.
TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER = "no_caller_speech_after_answer"
#: The new ending this work exists to produce.  The worker's closed reason set
#: and the control plane's enum gain it in the changes that act on it; until
#: then the arbiter can name it and nothing may stamp it.
TERMINATION_NO_HUMAN_AFTER_ANSWER = "no_human_after_answer"
ARBITER_TERMINATIONS = frozenset(
    {TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER, TERMINATION_NO_HUMAN_AFTER_ANSWER}
)

#: How long after arming a verdict may still be accepted.  A verdict that
#: arrives later is discarded: the window is what bounds the cost, and a
#: detector allowed to decide at any time is a detector with no window.
DECISION_WINDOW_DEFAULT_SECONDS = 15.0
#: The lower bound is the watchdog's own floor, so the two policies cannot
#: reach different conclusions before either of their minimums.
DECISION_WINDOW_MIN_SECONDS = 10.0
DECISION_WINDOW_MAX_SECONDS = 30.0
#: The hard stop.  A human who picks up after it is lost, and that is the
#: stated price of not paying for minutes of music; the retry policy dials
#: them again.
CAP_DEFAULT_SECONDS = 45.0
CAP_MIN_SECONDS = 30.0
CAP_MAX_SECONDS = 90.0

#: At most two greetings are ever spoken on one call: the opening, and one
#: re-greeting for a person who appears after the music.
MAX_GREETINGS = 2

#: The existing silence rule's own floor, repeated for the same reason the
#: reason strings are, and pinned against the worker by test.  Below it an
#: agent's own greeting outlives the rule, so a per-agent value under it would
#: hang up on a caller who has not finished hearing hello.
NO_CALLER_SPEECH_FLOOR_SECONDS = 10.0

#: A recording on repeat.  The far end says the same ordered cycle of things
#: again and again: a hold loop, a voicemail menu, an announcement that
#: restarts.  It has the pauses and the words of speech, so the audio
#: classifier above reads it as a person, and on 2026-09-17 one such loop ran
#: a GPT-Live call for 28 minutes -- the model answered it 58 times -- until
#: the maximum call length.  Text is the only thing that tells it apart, and
#: the shape of the evidence is a CYCLE: measured over every completed call
#: since 2026-09-13 (55 calls with turns, cascade and GPT-Live, all agents),
#: no call repeats any 2-, 3- or 4-text cycle even once, while that call's
#: period-4 cycle recurred 15 times and its 3rd recurrence completed at 3.4
#: minutes.  A single text repeated across consecutive agent turns does NOT
#: separate: humans say "hello? hello? hello?" into a cut greeting (3 on the
#: batch), and the loop's identical text recurs only every 4th agent turn.
#:
#: Three recurrences of a cycle of two to four distinct texts.  The nearest
#: human value is one, a margin of two whole recurrences; four would keep the
#: same margin and fire a minute later.  Period one is deliberately excluded:
#: that is the human "hello? hello?" shape.  No knob: the numbers are the
#: measurement, cited above, not a preference.
REPEATING_CYCLE_MIN_PERIOD = 2
REPEATING_CYCLE_MAX_PERIOD = 4
REPEATING_CYCLE_RECURRENCES = 3
#: How many normalized far-end texts the detector keeps.  Enough for the
#: longest cycle at its recurrence count several times over; a loop that has
#: not shown within this many turns is not one this rule claims.
REPEATING_CYCLE_HISTORY = 64
#: Two readings that separate a recording from a person who happens to say
#: the same few things in turn, both measured on the incident and on every
#: completed call of the five days before it (numbers in the commit).  A
#: recording recites lines: every text of the incident's cycle has 2-5
#: tokens once normalized.  A person answering a pair of questions says
#: "yes", "no", "yes", "no" -- one token each; 35% of the batch's caller
#: captions are one token.  Text with no word spacing never reaches two
#: tokens, so this rule claims nothing in such a script rather than guess.
REPEATING_CYCLE_MIN_TOKENS = 2
#: A recording plays on its own clock, so its recurrences complete at a
#: steady interval; a person waits for questions of varying length.  The
#: largest relative deviation of a recurrence interval from the mean of the
#: counted intervals, at any point the incident's loop stood, was 0.034
#: (0.013 at the point the rule decides).  No call in the batch ever formed
#: even a second recurrence of a 2-4-text cycle, so there is no human cycle
#: to compare against directly; the nearest population -- every run of three
#: consecutive caller texts of two or more tokens in the batch, the texts
#: that clear the first reading -- has a median relative deviation of 0.26,
#: under 0.10 about one time in five.  So this is the second reading, and it
#: does real work alongside the first rather than standing in for it.
REPEATING_CYCLE_CADENCE_TOLERANCE = 0.10

#: How long a syllable-band reading may hold the cap open before it stops
#: counting as "a voice is here NOW".  Without a bound, one speech-shaped
#: classification holds a call open for as long as it lasts if the
#: classifier then stops reporting -- and the call runs to the maximum call
#: duration, billed as a conversation, which is the incident this exists to
#: end.  The classifier bounds its own inputs by age for the same reason.
VOICE_HOLD_MAX_AGE_SECONDS = 5.0


def normalize_far_end_text(text: Any) -> str:
    """One spelling for "the same thing was said again", in any language.

    Unicode-normalized, case-folded, format characters dropped, punctuation
    read as a word boundary, whitespace collapsed.  Language-agnostic by
    construction: nothing here knows a word list or a script.  A zero-width
    joiner or non-joiner is invisible and shapes rather than spells, so a
    caption with and without one is the same caption and a caption of
    nothing but them is empty.  A zero-width SPACE is also invisible but,
    unlike those two, marks a word boundary in a script that does not use
    ordinary spaces (Khmer, Lao, Thai line-breaking), so it reads as a
    boundary rather than being dropped -- dropping it merged two words into
    one token, silently taking a caption below the two-token floor below
    (Codex review, 2026-09-18).  "123-456" and "123 456" are the same
    number.  Empty text is what no detector counts.
    """

    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    pieces: list[str] = []
    for character in folded:
        if character == "​":
            pieces.append(" ")
            continue
        category = unicodedata.category(character)
        if category == "Cf":
            continue
        pieces.append(" " if category.startswith("P") else character)
    return " ".join("".join(pieces).split())


class RepeatingFarEndDetector:
    """How many times the far end's latest cycle of texts has recurred.

    Pure: takes texts with the time they arrived, keeps a bounded history,
    returns a count.  Text stays in memory here and is never emitted; the
    callers report only the period and the count.
    """

    def __init__(
        self,
        *,
        min_period: int = REPEATING_CYCLE_MIN_PERIOD,
        max_period: int = REPEATING_CYCLE_MAX_PERIOD,
        min_tokens: int = REPEATING_CYCLE_MIN_TOKENS,
        cadence_tolerance: float = REPEATING_CYCLE_CADENCE_TOLERANCE,
        history: int = REPEATING_CYCLE_HISTORY,
    ) -> None:
        self._min_period = int(min_period)
        self._max_period = int(max_period)
        self._min_tokens = int(min_tokens)
        self._tolerance = float(cadence_tolerance)
        self._history = int(history)
        self._turns: list[tuple[str, float]] = []

    def note(self, text: Any, *, at: float) -> tuple[int, int]:
        """Add one far-end text heard at ``at``; return ``(period, recurrences)``.

        ``recurrences`` counts consecutive, non-overlapping repeats of the
        ordered cycle that ENDS at this text, over every period in range,
        and only for a cycle a recording would produce: its texts all
        distinct (a repeated single text is the human shape), each at least
        ``min_tokens`` long (a line, not an answer), and its recurrences
        completing on one clock -- the count stops at the first recurrence
        whose interval leaves the tolerance around the mean of those already
        counted.  ``(0, 0)`` when nothing recurs (one recitation is not a
        recurrence), and for empty text.
        """

        normalized = normalize_far_end_text(text)
        if not normalized:
            return (0, 0)
        self._turns.append((normalized, float(at)))
        del self._turns[: -self._history]
        texts = [turn_text for turn_text, _ in self._turns]
        best = (0, 0)
        for period in range(self._min_period, self._max_period + 1):
            if len(texts) < period * 2:
                continue
            unit = texts[-period:]
            if len(set(unit)) != period:
                continue
            if any(len(turn_text.split()) < self._min_tokens for turn_text in unit):
                continue
            # When each recurrence completed, latest first.
            completions = [self._turns[-1][1]]
            end = len(texts) - period
            while end - period >= 0 and texts[end - period : end] == unit:
                completions.append(self._turns[end - 1][1])
                end -= period
            recurrences = self._on_one_clock(completions)
            if recurrences >= 2 and recurrences > best[1]:
                best = (period, recurrences)
        return best

    def _on_one_clock(self, completions: list[float]) -> int:
        """How many of the latest recurrences completed at a steady interval."""

        intervals: list[float] = []
        counted = 1
        for later, earlier in zip(completions, completions[1:]):
            intervals.append(later - earlier)
            mean = sum(intervals) / len(intervals)
            if mean <= 0.0 or max(abs(interval - mean) for interval in intervals) > self._tolerance * mean:
                break
            counted += 1
        return counted


def resolved_windows(
    decision_window_seconds: float | None = None,
    cap_seconds: float | None = None,
) -> tuple[float, float]:
    """Clamp a per-agent window and cap, and keep the cap at or above it.

    A cap below the decision window would end the call before the verdict it
    is waiting for could be accepted, so the pair is repaired rather than
    rejected: this runs inside a live call, where refusing a configuration
    means dropping it.
    """

    window = DECISION_WINDOW_DEFAULT_SECONDS
    if decision_window_seconds is not None:
        window = min(
            DECISION_WINDOW_MAX_SECONDS,
            max(DECISION_WINDOW_MIN_SECONDS, float(decision_window_seconds)),
        )
    cap = CAP_DEFAULT_SECONDS
    if cap_seconds is not None:
        cap = min(CAP_MAX_SECONDS, max(CAP_MIN_SECONDS, float(cap_seconds)))
    return window, max(cap, window)


def _would_have_ended(
    would_have_ended_at: float | None,
    *,
    human_at: float | None,
    carrier_disconnect_at: float | None,
    cap_held_by_voice_at: float | None = None,
    repeating_at: float | None = None,
    would_have_ended_by_repeating: bool = False,
) -> bool:
    """Would the detector have ended this call, had it been allowed to act?

    Only evidence that arrived BEFORE the cap can have stopped the cap.  The
    first version of this asked whether human evidence existed at all, which
    read a person who spoke at 52 s on a 45 s cap as a call the detector would
    not have cut -- when in fact it would have cut them off seven seconds
    before they ever spoke.  That is precisely the population the shadow
    rollout exists to count, and the metric was hiding it in the direction
    that makes turning the action on look safe.
    """

    if would_have_ended_at is None:
        # Nothing stamped: the cap never came, or the disposition was one
        # this rule only watches.  A loop noted on a fail-open answer is
        # such a case, and must not count as a call the rule would have cut.
        return False
    if would_have_ended_by_repeating:
        # ``would_have_ended_at`` IS ``repeating_at`` here (only the loop's
        # own stamp sets this flag): a recording on repeat is ended over its
        # own human evidence -- its words are what the audio classifier
        # heard -- so that evidence cannot excuse it here either.  Only the
        # far end leaving first can.  This ordering is valid ONLY for a
        # loop-origin stamp: applying it to a cap-origin stamp read a call
        # the cap would have ended well before a later carrier disconnect as
        # one the disconnect excused, because the comparison used the
        # loop's own (later) timestamp instead of the cap's (earlier) one
        # (Codex review, 2026-09-18).
        return carrier_disconnect_at is None or carrier_disconnect_at > repeating_at
    if cap_held_by_voice_at is not None:
        # The cap came and the arbiter did not fire, because a voice was in
        # the envelope.  This call is not one the detector would have ended.
        return False
    if human_at is not None and human_at <= would_have_ended_at:
        return False
    if (
        carrier_disconnect_at is not None
        and carrier_disconnect_at <= would_have_ended_at
    ):
        return False
    return True


@dataclass(frozen=True)
class ArbiterDecision:
    """One ending this arbiter concluded, and whether it may be acted on."""

    reason: str
    classification: str
    detail: str
    #: False in shadow, in ``off``, and on the fail-open dispositions.  A
    #: caller that acts on a decision with ``acts=False`` has broken the
    #: rollout contract, so the flag travels WITH the decision rather than
    #: being a property the caller has to remember to ask about.
    acts: bool
    decided_at: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TerminalRecord:
    """What is written about this call, once, at the terminal commit."""

    classification: str
    detail: str
    reason: str
    mode: str
    arming_disposition: str
    armed: bool
    acts: bool
    classifier_version: str
    score: float
    reversed_by: str
    decision_window_seconds: float
    cap_seconds: float
    armed_at: float | None
    verdict_at: float | None
    human_evidence_at: float | None
    committed_at: float
    #: When the cap WOULD have ended this call, set the moment a non-human
    #: verdict was accepted and never cleared afterwards.  A record carrying
    #: both this and a ``human_evidence_at`` is a call the detector would have
    #: cut off and a person then answered -- the one measurement that gates
    #: turning the action on, and the reason the pair is kept rather than
    #: collapsed into the boolean below.
    would_have_ended_at: float | None
    #: The detector would have ended this call and nothing reversed it.
    would_have_ended: bool
    greetings_spoken: int
    re_greeted: bool
    #: How many times the answer attribute was published again after the
    #: first. Never acted on -- the clocks are latched to the first -- but
    #: worth counting, because a carrier that flaps it is a thing an operator
    #: would want to know before reading anything else on this record.
    duplicate_armings: int
    features: dict[str, Any] = field(default_factory=dict)
    #: The voicemail observer's classification, recorded beside the
    #: verdict and deliberately not part of it: in shadow it changes nothing
    #: this record says to the customer.  Empty when the watch is off.
    voicemail_classification: str = ""

    def evidence(self) -> dict[str, Any]:
        """Bounded scalars for the call summary and telemetry.  Never audio.

        Times are reported as offsets from arming, not as clock readings: the
        arbiter runs on a monotonic clock whose zero means nothing outside
        this process, and an operator reading the record wants "the verdict
        landed 9 s after the answer".
        """

        def offset(value: float | None) -> float | None:
            if value is None or self.armed_at is None:
                return None
            return round(value - self.armed_at, 3)

        record: dict[str, Any] = {
            "answer_classification": self.classification,
            "reason_detail": self.detail,
            "termination_detail": self.reason,
            "answer_detection_mode": self.mode,
            "arming_disposition": self.arming_disposition,
            "armed": self.armed,
            "acts": self.acts,
            "classifier_version": self.classifier_version,
            "score": round(float(self.score), 4),
            "reversed_by": self.reversed_by,
            "decision_window_seconds": self.decision_window_seconds,
            "cap_seconds": self.cap_seconds,
            "verdict_after_seconds": offset(self.verdict_at),
            "human_evidence_after_seconds": offset(self.human_evidence_at),
            "committed_after_seconds": offset(self.committed_at),
            "would_have_ended_after_seconds": offset(self.would_have_ended_at),
            "would_have_ended": self.would_have_ended,
            "greetings_spoken": self.greetings_spoken,
            "re_greeted": self.re_greeted,
            "duplicate_armings": self.duplicate_armings,
        }
        record.update(self.features)
        if self.voicemail_classification:
            record["voicemail_classification"] = self.voicemail_classification
        return record


@dataclass(eq=False)
class PostAnswerArbiter:
    """The post-answer state machine for one call.

    Construct it at session start with the per-agent setting, arm it with the
    answer gate's disposition, feed it classifications and signals, and poll
    it from the same watchdog tick that already polls the rest of the ladder.
    """

    mode: str = DEFAULT_MODE
    decision_window_seconds: float = DECISION_WINDOW_DEFAULT_SECONDS
    cap_seconds: float = CAP_DEFAULT_SECONDS
    #: The existing silence rule's threshold for this call, or ``None`` when
    #: the customer turned it off.  Passed in rather than read here so the two
    #: places that own it cannot disagree.
    no_caller_speech_seconds: float | None = 20.0

    _arming_disposition: str = ""
    _armed_at: float | None = None
    #: When the call last returned from a phone menu or a hold queue to
    #: conversation. Every clock below that measures "how long since the
    #: answer" measures from the LATER of this and the arming, so ten minutes
    #: of legitimate hold do not count as ten minutes of silence at pickup.
    _resumed_at: float | None = None
    _duplicate_armings: int = 0
    _classification: AnswerClassification | None = None
    #: The most recent classification of any kind, as opposed to the standing
    #: verdict.  The two differ on exactly the call this exists for: a person
    #: speaking over music that never stops leaves the non-human verdict
    #: standing while the CURRENT reading of the envelope carries their voice.
    _latest: AnswerClassification | None = None
    _verdict_at: float | None = None
    _human_at: float | None = None
    _reversed_by: str = ""
    _carrier_disconnect_at: float | None = None
    _would_have_ended_at: float | None = None
    #: Whether ``_would_have_ended_at`` was stamped by the repeating-cycle
    #: rule rather than by the cap.  The two stamp on different clocks -- the
    #: cap projects a future firing moment, the loop records when its own
    #: recurrence actually completed -- and only the loop's own carrier
    #: ordering is valid to apply to a loop-origin stamp (Codex review,
    #: 2026-09-18: a cap-origin stamp compared with the loop's ordering read
    #: a call that would have ended on the cap, well before a carrier
    #: disconnect, as one the disconnect excused).
    _would_have_ended_by_repeating: bool = False
    _cap_held_by_voice_at: float | None = None
    _greetings_spoken: int = 0
    _re_greeted: bool = False
    _repeating: RepeatingFarEndDetector = field(default_factory=RepeatingFarEndDetector)
    _repeating_period: int = 0
    _repeating_recurrences: int = 0
    _repeating_at: float | None = None
    _decision: ArbiterDecision | None = None
    #: The decision this call was ended on.  ``_decision`` above is the dedupe
    #: latch and is CLEARED the moment a decision stops applying; this one is
    #: not, because the record has to be able to say what the customer was
    #: told even if the reading moved afterwards.
    _decided: ArbiterDecision | None = None
    _terminal: TerminalRecord | None = None
    _late_evidence_after_commit: int = 0
    #: Voicemail detection.  Record-only in this release: no branch of ``_applies``, no
    #: property the worker acts on and neither customer field reads it.
    _voicemail_classification: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            # An unknown mode is the customer's setting arriving damaged, and
            # the safe reading of damage is the default: collect, act on
            # nothing.
            self.mode = DEFAULT_MODE
        self.decision_window_seconds, self.cap_seconds = resolved_windows(
            self.decision_window_seconds, self.cap_seconds
        )
        if self.no_caller_speech_seconds is not None:
            self.no_caller_speech_seconds = max(
                NO_CALLER_SPEECH_FLOOR_SECONDS, float(self.no_caller_speech_seconds)
            )

    # -- state ---------------------------------------------------------

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    @property
    def armed_at(self) -> float | None:
        """When the clocks started, or ``None`` while this call is unarmed.

        Exposed because the worker glue windows its own sample buffers to the
        same instant.  Two answers to "when did this call answer" would be one
        answer too many: the classifier would then be measuring a window that
        started somewhere the arbiter's clocks did not.
        """

        return self._armed_at

    @property
    def collecting(self) -> bool:
        """Whether classification runs at all.  ``off`` collects nothing."""

        return self.mode != MODE_OFF

    @property
    def acts(self) -> bool:
        """Whether a decision from this arbiter may change the call.

        Both conditions, and they are independent: the customer chose to act,
        AND this call carries the answer evidence that bounds the window.
        """

        return self.mode == MODE_END_CALL and self._arming_disposition in ARMING_DISPOSITIONS

    @property
    def classification(self) -> str:
        if self._human_at is not None:
            return _classifier.VERDICT_HUMAN
        if self._classification is None:
            return _classifier.VERDICT_UNKNOWN
        return self._classification.verdict

    @property
    def detail(self) -> str:
        if self._human_at is not None or self._classification is None:
            return _classifier.DETAIL_NONE
        return self._classification.detail

    @property
    def no_conversation_stands(self) -> bool:
        """Nobody is conversing, read from the CURRENT classification.

        Not latched, so the release is the same event as the cause: the far
        end pauses, the span it was holding ends, and this goes false on the
        next tick.  Human evidence does NOT cancel it -- singing produces that
        evidence for the length of the call -- but the pause a person makes
        does.
        """

        return self._latest is not None and self._latest.no_conversation

    @property
    def current_reading(self) -> tuple[str, str]:
        """``(verdict, detail)`` of the CURRENT classification, or ``("", "")``.

        Not the standing verdict: that is latched inside the decision window
        and is reversed by any human evidence, so it cannot say that hold
        music began ten minutes after the answer.
        """

        latest = self._latest
        return ("", "") if latest is None else (str(latest.verdict), str(latest.detail))

    @property
    def non_human_active(self) -> bool:
        """A non-human verdict stands and nothing has reversed it."""

        return (
            self._human_at is None
            and self._classification is not None
            and self._classification.is_non_human
        )

    def voice_holds_cap(self, *, at: float) -> bool:
        """The envelope being measured right now carries syllable-band shape.

        This is what stops the cap from ending a call on a person who arrived
        after the verdict and is speaking over music that never stops.  They
        produce no pause, so no word of theirs can count and no utterance
        shape can form -- but their voice is in the envelope, and the
        modulation that proves a talker is the only thing left that can speak
        for them.

        It holds the call rather than reversing the verdict.  A rise in
        modulation is not proof of a person: a track with a syllabic beat
        would produce one, and treating that as human evidence would clear
        suppression and deliver the account balance to a radio.  So
        suppression stays, the cap waits, and the call ends on the first tick
        where the voice is no longer there.  False whenever no depth could be
        measured, so a dead envelope feed leaves the cap free to fire -- and
        neither does a reading that has stopped being renewed, because "a
        voice is in the envelope" is a claim about the present tense.
        """

        if self._latest is None or not self._latest.speech_like_modulation:
            return False
        return float(at) - self._latest.evaluated_at <= VOICE_HOLD_MAX_AGE_SECONDS

    @property
    def silence_the_agent(self) -> bool:
        """Stop the agent speaking, from its next turn onward.

        False in shadow.  It does NOT stop the caller's audio reaching the
        recogniser or the model: every way of doing that also starves the
        voice-activity feed this rule reads, and what it reads there is the
        pause that says a person has arrived.
        """

        return self.acts and (self.non_human_active or self.no_conversation_stands)

    @property
    def committed(self) -> bool:
        return self._terminal is not None

    @property
    def late_evidence_after_commit(self) -> int:
        """Human evidence that arrived too late to change anything.

        Counted rather than applied.  A call ending with this above zero is
        one where a person spoke as the call was being torn down, and that is
        worth being able to see -- but a record that has been published
        cannot be corrected in place for the consumers who already have it.
        """

        return self._late_evidence_after_commit

    @property
    def terminal(self) -> TerminalRecord | None:
        return self._terminal

    # -- signals -------------------------------------------------------

    def arm(self, disposition: str, *, at: float) -> bool:
        """Record the answer disposition.  True when this call may be ACTED on.

        Which means the mode as well as the evidence: in ``shadow`` and in
        ``off`` this is False for every disposition, because nothing that
        happens on the call can change it.

        The fail-open dispositions start the clocks too -- they have to, or
        shadow mode would collect nothing from them -- so this returning False
        does not mean nothing happened.  It means no ending can ever come of
        it.  ``armed`` answers the other question.

        The disposition is kept even when it does not arm, because the record
        has to be able to say WHY a call was only ever watched.
        """

        if str(disposition) not in KNOWN_DISPOSITIONS:
            return False
        if self._arming_disposition:
            self._duplicate_armings += 1
            return False
        self._arming_disposition = str(disposition)
        if disposition in ARMING_DISPOSITIONS or disposition in SHADOW_ONLY_DISPOSITIONS:
            self._armed_at = float(at)
        return self.acts

    def observe(self, classification: AnswerClassification, *, at: float) -> None:
        """Take one classification of the window so far.

        Human evidence is accepted at any time before the commit.  A non-human
        verdict is accepted only inside the decision window, and once accepted
        it stands until human evidence reverses it -- a later re-evaluation
        cannot quietly downgrade it to ``unknown`` and release the call,
        because the audio it measured did happen.
        """

        if not self.collecting:
            return
        if self._terminal is not None:
            if classification.is_human:
                self._late_evidence_after_commit += 1
            return
        if self._armed_at is not None and (
            float(at) < self._armed_at or classification.armed_at < self._armed_at
        ):
            # A classification of audio from before the answer, whichever
            # timestamp gives it away: the moment it was delivered, or the
            # window it was computed over.  The human path has refused these
            # from the start; without the same floor here, a verdict computed
            # over ring-back arms suppression at the answer, holds off the
            # existing silence rule, and ends a call at the cap on a customer
            # who picked up and listened -- on no post-answer evidence at all.
            return
        if not self.armed:
            # Nothing was classified, because there was no window to classify.
            # Recording a verdict here would put a reading of pre-answer media
            # on the call's record.
            return
        # Above every verdict branch, because this is the one observation that
        # has to survive a ``human`` classification.  The CURRENT reading is
        # what the decision uses, and it is the only thing kept.
        self._latest = classification
        if classification.is_human:
            # The tick that first read ``human``.  ``note_human_evidence``
            # returns once ``_human_at`` is set, so only this first one is
            # ever kept and a re-read of the same old utterance cannot move
            # it.
            self.note_human_evidence(
                classification.reversed_by or _classifier.REVERSED_BY_UTTERANCE_SHAPE,
                at=at,
            )
            if self._classification is None or not self._classification.is_non_human:
                self._classification = classification
            # A standing non-human verdict is KEPT when a person reverses it.
            # The reversal is carried by ``_human_at``, which the reported
            # classification already honours, and the tune's own measurements
            # are the evidence worth keeping on a reversed call.  Overwriting
            # them also silently withdrew the re-greeting, which is owed to
            # precisely this caller: the one who appeared out of the music.
            return
        if self._human_at is not None:
            return
        self._latest = classification
        if classification.is_non_human:
            armed_at = self._armed_at or 0.0
            # The decision window re-opens at an exit from a menu or a queue,
            # exactly as the clocks in ``_applies`` do.
            if self._resumed_at is not None:
                armed_at = max(armed_at, self._resumed_at)
            if at - armed_at > self.decision_window_seconds:
                return
            if self._classification is None or not self._classification.is_non_human:
                self._classification = classification
                self._verdict_at = float(at)
                # Recorded when the verdict lands, not when the cap elapses.
                # In shadow the silence rule ends most of these calls at 20 s,
                # so a would-have-ended time written only at the cap would be
                # missing from exactly the calls the rollout is measuring.
                if (
                    self._arming_disposition in ARMING_DISPOSITIONS
                    and not self._would_have_ended_by_repeating
                ):
                    # Only a call with real answer evidence has a cap that
                    # would have fired.  On the fail-open dispositions the
                    # clock is anchored to a moment that may be ring-back, and
                    # recording a would-have-ended against it would put
                    # pre-answer audio into the metric that decides whether
                    # this may start ending calls.  A loop that already
                    # stamped this is left alone: first ending wins, on
                    # whichever clock reached it first.
                    self._would_have_ended_at = armed_at + self.cap_seconds
            return
        if self._classification is None or not self._classification.is_non_human:
            self._classification = classification
            self._verdict_at = float(at)

    def note_human_evidence(self, source: str, *, at: float) -> None:
        """A person is on the line.  Cancels the cap and clears suppression."""

        if not self.collecting:
            return
        if self._terminal is not None:
            self._late_evidence_after_commit += 1
            return
        if not self.armed or (
            self._armed_at is not None and float(at) < self._armed_at
        ):
            # Evidence from before the answer is not evidence about the
            # answer.  Carrier early media can carry a recorded voice, and a
            # word recognised in it would mark the call ``human`` in the
            # record and poison exactly the population the shadow rollout is
            # measuring.  The classifier windows its own inputs the same way,
            # so a timestamp is judged against the arming moment and not
            # merely against the ORDER the signals happened to arrive in.
            return
        if self._human_at is not None:
            return
        if self._cap_held_by_voice_at is not None and not self.voice_holds_cap(at=at):
            # The voice that held the cap had already gone stale by the time
            # this person spoke, so the detector would have fired in between
            # and they would never have been heard.  Whether that was noticed
            # used to depend on a watchdog tick landing inside the gap, which
            # made the rollout metric a function of poll cadence rather than
            # of what happened on the call.
            self._cap_held_by_voice_at = None
        self._human_at = float(at)
        self._reversed_by = str(source)

    def note_far_end_text(self, text: Any, *, at: float) -> tuple[int, int]:
        """One recognised far-end utterance: an STT final or a live caption.

        Returns the ``(period, recurrences)`` the detector reads after it, so
        the caller can report the moment a loop is established without ever
        seeing the text.  Gated exactly as every other signal: nothing before
        the answer, nothing after the commit, nothing while not collecting.
        """

        if not self.collecting or self._terminal is not None:
            return (0, 0)
        if not self.armed or (self._armed_at is not None and float(at) < self._armed_at):
            return (0, 0)
        period, recurrences = self._repeating.note(text, at=float(at))
        if recurrences > self._repeating_recurrences:
            self._repeating_period, self._repeating_recurrences = period, recurrences
        if recurrences >= REPEATING_CYCLE_RECURRENCES and self._repeating_at is None:
            self._repeating_at = float(at)
            if (
                self._arming_disposition in ARMING_DISPOSITIONS
                and self._would_have_ended_at is None
            ):
                # The shadow metric: a loop the rule WOULD have ended, at the
                # moment it stood.  Real answer evidence only, for the same
                # reason the cap's own stamp above requires it.
                self._would_have_ended_at = float(at)
                self._would_have_ended_by_repeating = True
        return (period, recurrences)

    @property
    def repeating_recording_stands(self) -> bool:
        """The far end has recited the same cycle three times over."""

        return self._repeating_at is not None

    def note_voicemail(self, classification: str) -> None:
        """The voicemail observer's current classification, for the record.

        This is where the observer arrives "as one more signal", as the module
        docstring requires.  While every agent runs voicemail detection in
        shadow it is recorded and nothing else: the ending precedence above
        is unchanged, and the acting release adds its branch here.
        """

        if self._terminal is None and self.collecting:
            self._voicemail_classification = str(classification or "")

    def note_carrier_disconnect(self, *, at: float) -> None:
        """The far end went away.  That ending is not this module's to make."""

        if self._carrier_disconnect_at is None:
            self._carrier_disconnect_at = float(at)

    def note_greeting_spoken(self, *, at: float) -> None:
        """One greeting was spoken, whether or not the caller heard all of it.

        An opening cut off by music counts: it was spoken, it is one of the
        two, and the recovery for a caller who never heard it is the
        re-greeting below, not a replay at the moment of the interruption.
        """

        del at
        if self._terminal is not None:
            return
        self._greetings_spoken = min(MAX_GREETINGS, self._greetings_spoken + 1)

    def claim_regreet(self) -> bool:
        """Take the single re-greeting, if one is owed.  True at most once.

        Owed only when suppression actually engaged and a person then
        appeared.  A call that was never suppressed has a caller who heard
        their opening, and greeting them twice would be the bug.

        The grant IS the greeting for counting purposes: a caller that takes
        it must not also report it through ``note_greeting_spoken``.
        """

        if not self.acts or self._terminal is not None:
            return False
        if self._re_greeted or self._human_at is None:
            return False
        if self._classification is None or not self._classification.is_non_human:
            return False
        if self._greetings_spoken >= MAX_GREETINGS:
            return False
        self._re_greeted = True
        self._greetings_spoken += 1
        return True

    # -- decision ------------------------------------------------------

    def discard_decision(self) -> None:
        """Forget a decision that was returned but never acted on.

        The worker calls this when the acting call site vetoes the ending (the
        call entered a menu or a queue on the same tick): an ending that never
        happened must not be what the terminal record says the call ended on.
        A decision already committed is untouched.
        """

        if self._terminal is None:
            self._decision = None
            self._decided = None

    def note_navigation_exit(self, *, at: float) -> None:
        """The call left a phone menu or a hold queue: start these clocks again.

        Everything this object concluded from the queue's audio belongs to the
        episode that just ended: the standing non-human verdict, the repeating
        cycle, and any would-have-ended stamp. They are cleared, so a person
        who picks up is judged on what is heard from now, not on the music.
        Human evidence and the carrier's disconnect are facts about the call
        and stay.
        """

        self._resumed_at = float(at)
        # Evidence is episode-local. Words the queue's announcements were
        # recognised as are not a person on THIS conversation; whoever is on
        # the line now is judged from here (a real person's words re-latch it).
        self._human_at = None
        self._reversed_by = ""
        self._classification = None
        self._verdict_at = None
        self._latest = None
        self._would_have_ended_at = None
        self._would_have_ended_by_repeating = False
        self._cap_held_by_voice_at = None
        self._repeating = RepeatingFarEndDetector()
        self._repeating_period = 0
        self._repeating_recurrences = 0
        self._repeating_at = None
        self._decision = None
        if self._terminal is None:
            self._decided = None

    def poll(
        self, *, at: float, caller_has_spoken: bool, navigation_suspended: bool = False
    ) -> ArbiterDecision | None:
        """The ending that applies right now, or ``None`` to keep going.

        While the call is in a phone menu or a hold queue no ending applies:
        every rule below reads silence or recorded audio as "nobody is here",
        which is exactly what a queue is. Only the navigation bounds end it.
        """

        if not self.collecting:
            # ``off`` is inert, not quiet: it produces no decisions at all, so
            # the watchdog ladder keeps every rung it has today.
            return None
        if navigation_suspended:
            # Nothing applies now, and the returned decision is forgotten so
            # that a later one is issued afresh. The RECORD of what was decided
            # is left alone: it is cleared by the exit that ends an episode and
            # by ``discard_decision`` (a veto at the acting site), and an entry
            # window's deferral is not either of those -- the same ending can
            # still happen after it, and must seal the classification it was
            # decided with, as it does with the feature off.
            self._decision = None
            return None
        decision = self._applies(at=at, caller_has_spoken=caller_has_spoken)
        if decision is None:
            # Whatever stood a moment ago no longer does: a person spoke, the
            # carrier went, or a voice appeared in the envelope.  A decision
            # that has not been committed is still only a reading of the
            # present, and the present has changed.
            self._decision = None
            return None
        if self._decision is not None and self._decision.reason == decision.reason:
            # The SAME decision, not another one like it.  A watchdog polls on
            # a timer, and one that asked twice before committing used to be
            # handed two terminal decisions for one call -- which is two
            # end-call requests, against a contract that promises one.
            return self._decision
        self._decision = decision
        self._decided = decision
        return decision

    def _applies(self, *, at: float, caller_has_spoken: bool) -> ArbiterDecision | None:
        """The ending the current evidence supports, recomputed from scratch."""

        if self._terminal is not None or not self.armed or self._armed_at is None:
            return None
        if self._carrier_disconnect_at is not None:
            return None
        since = self._armed_at if self._resumed_at is None else max(self._armed_at, self._resumed_at)
        elapsed = float(at) - since
        if self._repeating_at is not None:
            # Above the human-evidence stop on purpose: a loop's own words ARE
            # human evidence to the audio classifier (they start and stop, a
            # recogniser hears them), which is exactly why nothing below could
            # ever end one.  The cycle is the stronger reading of the same
            # words, and it is not released by another recitation of them.
            return ArbiterDecision(
                reason=TERMINATION_NO_HUMAN_AFTER_ANSWER,
                classification=_classifier.VERDICT_NON_HUMAN_AUDIO,
                detail=_classifier.DETAIL_LOOPING_RECORDING,
                acts=self.acts,
                decided_at=float(at),
                elapsed_seconds=round(elapsed, 3),
            )
        if self._human_at is not None and not self.no_conversation_stands:
            # Human evidence stops every ending below, with one exception it
            # cannot stop: a far end that has not paused long enough for
            # anyone to reply.  Singing supplies that evidence from about five
            # seconds and keeps supplying it, which is why it cannot govern
            # this one rule.
            return None
        if self.no_conversation_stands:
            # Its own branch, above the flat-tone one and not sharing its
            # voice-hold: on the calls this exists for the envelope IS a
            # singer, so holding the cap on "a voice is present" would stop
            # the rule firing on precisely the audio it is for.
            if elapsed >= self.cap_seconds:
                return ArbiterDecision(
                    reason=TERMINATION_NO_HUMAN_AFTER_ANSWER,
                    classification=_classifier.VERDICT_NON_HUMAN_AUDIO,
                    detail=_classifier.DETAIL_HOLD_MUSIC,
                    acts=self.acts,
                    decided_at=float(at),
                    elapsed_seconds=round(elapsed, 3),
                )
            return None
        if self.non_human_active:
            if elapsed >= self.cap_seconds and self.voice_holds_cap(at=at):
                # Recorded so the shadow metric can tell a call the detector
                # would have ended from one it would have kept open.  The
                # boolean on the record has to answer with the same predicate
                # this line decides on, or it counts the opposite thing.
                self._cap_held_by_voice_at = float(at)
            if elapsed >= self.cap_seconds and not self.voice_holds_cap(at=at):
                # The hold is over and the cap applies after all.  Clearing it
                # here is what keeps the record honest: a latch that survived
                # this line would report a call the detector went on to end as
                # one it had kept open, which is the same metric error as
                # counting a person who spoke after the cap as a reversal.
                self._cap_held_by_voice_at = None
                return ArbiterDecision(
                    reason=TERMINATION_NO_HUMAN_AFTER_ANSWER,
                    classification=self.classification,
                    detail=self.detail,
                    acts=self.acts,
                    decided_at=float(at),
                    elapsed_seconds=round(elapsed, 3),
                )
            return None
        if (
            self.no_caller_speech_seconds is not None
            and not caller_has_spoken
            and elapsed >= self.no_caller_speech_seconds
        ):
            return ArbiterDecision(
                reason=TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER,
                classification=self.classification,
                detail=self.detail,
                acts=self.acts,
                decided_at=float(at),
                elapsed_seconds=round(elapsed, 3),
            )
        return None

    def commit(self, reason: str, *, at: float) -> TerminalRecord:
        """Freeze the record for this call.  The first call wins, forever."""

        if self._terminal is not None:
            return self._terminal
        # What the customer is told is the DECISION's, not this object's
        # standing verdict.  They disagree on exactly the call the floor rule
        # exists for: ``classification`` reports ``human`` there, because a
        # singer produced human evidence at five seconds, while the ending
        # sent to the customer says ``non_human_audio`` / ``hold_music``.  A
        # record carrying the other answer is a second answer to one question,
        # and the consumers of the two are the same people.
        decided = self._decided
        if decided is not None and decided.reason == str(reason):
            classification = decided.classification
            detail = decided.detail
        else:
            classification = self.classification
            detail = self.detail
        self._terminal = TerminalRecord(
            classification=classification,
            detail=detail,
            reason=str(reason),
            mode=self.mode,
            arming_disposition=self._arming_disposition,
            armed=self.armed,
            acts=self.acts,
            classifier_version=(
                self._classification.version
                if self._classification is not None
                else _classifier.CLASSIFIER_VERSION
            ),
            score=(
                self._classification.score if self._classification is not None else 0.0
            ),
            reversed_by=self._reversed_by,
            decision_window_seconds=self.decision_window_seconds,
            cap_seconds=self.cap_seconds,
            armed_at=self._armed_at,
            verdict_at=self._verdict_at,
            human_evidence_at=self._human_at,
            committed_at=float(at),
            would_have_ended_at=self._would_have_ended_at,
            would_have_ended=_would_have_ended(
                self._would_have_ended_at,
                human_at=self._human_at,
                carrier_disconnect_at=self._carrier_disconnect_at,
                cap_held_by_voice_at=self._cap_held_by_voice_at,
                repeating_at=self._repeating_at,
                would_have_ended_by_repeating=self._would_have_ended_by_repeating,
            ),
            greetings_spoken=self._greetings_spoken,
            re_greeted=self._re_greeted,
            duplicate_armings=self._duplicate_armings,
            features=dict(
                self._classification.features
                if self._classification is not None
                else {}
            ),
            voicemail_classification=self._voicemail_classification,
        )
        return self._terminal


__all__ = [
    "ARBITER_TERMINATIONS",
    "ARMING_DISPOSITIONS",
    "KNOWN_DISPOSITIONS",
    "ArbiterDecision",
    "CAP_DEFAULT_SECONDS",
    "CAP_MAX_SECONDS",
    "CAP_MIN_SECONDS",
    "DECISION_WINDOW_DEFAULT_SECONDS",
    "DECISION_WINDOW_MAX_SECONDS",
    "DECISION_WINDOW_MIN_SECONDS",
    "DEFAULT_MODE",
    "MAX_GREETINGS",
    "NO_CALLER_SPEECH_FLOOR_SECONDS",
    "VOICE_HOLD_MAX_AGE_SECONDS",
    "MODES",
    "MODE_END_CALL",
    "MODE_OFF",
    "MODE_SHADOW",
    "PostAnswerArbiter",
    "REPEATING_CYCLE_CADENCE_TOLERANCE",
    "REPEATING_CYCLE_HISTORY",
    "REPEATING_CYCLE_MAX_PERIOD",
    "REPEATING_CYCLE_MIN_PERIOD",
    "REPEATING_CYCLE_MIN_TOKENS",
    "REPEATING_CYCLE_RECURRENCES",
    "RepeatingFarEndDetector",
    "normalize_far_end_text",
    "SHADOW_ONLY_DISPOSITIONS",
    "TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER",
    "TERMINATION_NO_HUMAN_AFTER_ANSWER",
    "TerminalRecord",
    "resolved_windows",
]

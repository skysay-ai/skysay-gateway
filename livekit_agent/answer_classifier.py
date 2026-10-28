# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What is making noise on an answered outbound call: a person, or a tune.

The carrier's SIP answer proves the transport was answered, never that a human
picked up.  On the finconnect batch of 2026-09-15 two calls
(``call_Jxjxs0tm6lkbFk``, ``call_dNdRfss4x0LQ2c``) ran 180 s and 181 s with
~176 s of continuous audio, no human speech and no agent turn: the far end was
playing a caller tune (a ring-back replacement tone the SUBSCRIBER chooses),
``sip.callStatus`` was ``active``, and the GPT-Live captions transcribed 73
"customer" turns of English song lyrics.  The call was billed as a
conversation.  The answer gate is not wrong -- the transport really was
answered -- so the missing signal is a second answer state, decided from the
audio itself.

This module is the decision, and nothing else.  It imports no livekit, holds
no state between calls and touches no audio buffer: it takes already-measured
samples and returns a verdict.  That is what lets the same code run three
ways -- on synthetic traces in the test suite, on recorded room-side tracks in
an offline replay, and on live frames in the worker -- and give the same
answer to the same evidence.  The worker glue that feeds it lives in
``agent/production_agent.py``; the policy that acts on the verdict lives in
``post_answer_arbiter.py``.

The rules, in the order they are applied
----------------------------------------

**One signal outranks the human rules, and only one.**  A far end that has
held the floor without a pause long enough to answer in carries
``no_conversation`` whatever the verdict says.  Every other rule here asks
what the audio IS; this one asks whether anybody is talking WITH us.  The
threshold, and why it is the only rule that separates a caller tune from a
person, are at ``conversation_hold_seconds``.

**Human wins immediately, on either of two signals.**  One utterance-shaped
run of voice activity -- 0.3-3 s followed by at least 300 ms of silence -- or
one recognised word from STT finals or captions in ANY language, where the
audio it belongs to has been TERMINATED by a pause.  Either one ends the
classification at ``human`` and nothing later can move it.  The bar is
deliberately low, because the two error directions are not symmetrical:
delivering the opening line into a hold queue costs a few seconds of TTS,
while hanging up on a real customer costs the customer.  A caller on
speakerphone with a television, a market stall, a child, an elderly caller
answering slowly, a Sinhala or Tamil speaker whose prosody the recogniser
handles badly -- all of them pause, and all of them are ``human`` here.

**A word inside audio that has not stopped is not evidence of a person**
(decision of 2026-09-15, revising the rule that any word won).  The reason is
the incident itself: a recogniser pointed at a song transcribes lyrics, and on
the two calls that motivated this module the captions produced 73 "customer"
turns over 176 s of unbroken audio.  A rule that trusted any word would have
called both of them human.  So a word is attributed to the last run of
activity that began at or before it -- STT finals and captions arrive AFTER
the speech they describe, usually in the silence following it -- and it counts
only once that run has ended.  Audio that never stops is judged by its shape
and its modulation alone.

The price is stated rather than hidden: a person talking continuously is not
recognised as a person until their first pause, and a person who speaks over
music that never stops is not recognised at all.  Their protection is that
neither of them can be classified non-human on words: that verdict needs a
flat envelope, and a talker's envelope is not flat.

**Non-human requires ALL THREE of** (a) continuous activity of at least 8 s
with no gap reaching 300 ms, (b) a syllable-band (2-8 Hz) envelope modulation
depth below the named threshold, and (c) zero COUNTED words in the window --
counted in the sense above, so words heard inside the unbroken audio itself
are not a defence against this verdict.
Any one of the three missing is ``unknown``, and ``unknown`` behaves exactly
as this platform behaved before this module existed.

**Nothing is silent that has not been observed.**  ``silent`` needs activity
samples that say "inactive", not an absent feed.  A caller whose activity
samples never arrived is ``unknown``; a measurement that was never taken is
not evidence of quiet.  The same applies to the envelope: a missing or too
sparse envelope makes the modulation feature UNKNOWN, and an unknown feature
can never satisfy (b).  Every fail-open path in a detector that can hang up on
a customer has to be written the other way round.

What this is NOT
----------------

**It is not a voicemail, IVR or announcement detector, and it is built so that
it cannot become one by accident.**  The continuity rule is the whole reason:
a recorded announcement ("thank you for calling, please hold") and an IVR menu
have the pauses of speech, so they produce utterance shapes and words and are
classified ``human`` or ``unknown`` here.  That miss is deliberate and
accepted (adversarial review of 2026-09-15, finding 5); those classes belong
to the parent answer-detection work, which has voicemail policy, beep evidence
and per-agent actions this module must not pre-empt.

**It is not a second voice activity detector.**  The activity timeline comes
from the VAD the session already runs, handed in as transitions.  This module
decides what a run of activity MEANS; it never decides whether a frame is
loud.  ``first_continuous_threshold_run`` in ``demo.py`` is the reference for
the shape of that reasoning -- sustained activity, short bursts ignored --
but it answers a different question (when did audio start) and is not reused,
because every rule here needs a run's END as well as its start.

**A verdict is not a decision.**  Nothing in this module ends a call,
suppresses content or writes a record.  It is re-evaluated from the whole
window on every tick rather than accumulated, so a replay of the same samples
always produces the same verdict -- which is what makes offline evaluation on
recorded calls mean anything about live behaviour.

**The two non-human details are a provenance guess, not an acoustic
distinction.**  A caller tune and hold music are the same signal to these
features.  They are told apart by WHEN the sustained audio started relative to
the answer -- a subscriber's tune is already playing when the call is
answered, hold music generally starts later -- and they are treated
identically by every consumer of the verdict, including the retry policy.  The
split exists so an operator can read what happened, and it is worth nothing
more than that.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any


#: Stamped on every verdict and recorded per call.  A threshold moved without
#: moving this makes two incomparable populations look like one, which is the
#: failure that makes shadow-mode evidence worthless.
CLASSIFIER_VERSION = "answer-classifier/2026-09-15"

VERDICT_HUMAN = "human"
VERDICT_NON_HUMAN_AUDIO = "non_human_audio"
VERDICT_SILENT = "silent"
VERDICT_UNKNOWN = "unknown"

#: Closed.  The control plane serves this as an enum and the arbiter stamps it
#: on the call, so a string invented at a call site would reach a customer as
#: nothing at all.
VERDICTS = frozenset(
    {VERDICT_HUMAN, VERDICT_NON_HUMAN_AUDIO, VERDICT_SILENT, VERDICT_UNKNOWN}
)

DETAIL_CALLER_TUNE = "caller_tune"
DETAIL_HOLD_MUSIC = "hold_music"
#: A recording that repeats itself: the far end said the same ordered cycle of
#: things again and again, whatever the audio's shape.  Decided from recognised
#: text by the arbiter, never by this classifier, which reads audio only.
DETAIL_LOOPING_RECORDING = "looping_recording"
DETAIL_NO_AUDIO = "no_audio"
DETAIL_NONE = ""

#: Which detail may accompany which verdict.  ``unknown`` and ``human`` carry
#: none: a detail is a claim about what was heard, and neither verdict makes
#: one.
VERDICT_DETAILS: dict[str, frozenset[str]] = {
    VERDICT_HUMAN: frozenset({DETAIL_NONE}),
    VERDICT_NON_HUMAN_AUDIO: frozenset(
        {DETAIL_CALLER_TUNE, DETAIL_HOLD_MUSIC, DETAIL_LOOPING_RECORDING}
    ),
    VERDICT_SILENT: frozenset({DETAIL_NO_AUDIO}),
    VERDICT_UNKNOWN: frozenset({DETAIL_NONE}),
}

#: Which signal made a ``human`` verdict.  Recorded per call because the two
#: have different latencies and different failure modes, and shadow mode has to
#: be able to say which one saved a call.
REVERSED_BY_WORD = "word"
REVERSED_BY_UTTERANCE_SHAPE = "utterance_shape"
REVERSED_BY_NOTHING = ""

#: Where a recognised word came from.  The caller's contract is that only
#: CALLER-attributed words are handed in: the agent's own speech reaching the
#: caller track through a speakerphone is not caller evidence, and this module
#: cannot tell the difference.
WORD_SOURCE_STT_FINAL = "stt_final"

#: How much of a non-human stretch must lie outside a window in which no word
#: could be recognised. Two seconds is longer than a caption takes to follow
#: a short word on either engine, and short enough that a verdict on music
#: behind a held opening still lands inside the arbiter's decision window.
WORDS_OBSERVABLE_MIN_SECONDS = 2.0
WORD_SOURCE_LIVE_CAPTION = "live_caption"


@dataclass(frozen=True)
class ActivitySample:
    """One voice-activity TRANSITION on the caller track.

    A step function: ``active`` holds from ``at`` until the next sample.  A
    transition is therefore enough to describe a CHANGE, which is the shape of
    the runtime's ``user_state_changed`` events and of a VAD stream's
    start/end pairs.

    Transitions alone are not enough, though, and the caller owes more than
    them: the feed must also report periodically while nothing changes.  A
    detector that speaks only on transitions cannot be told from one that has
    died, and that difference decides whether a run of activity is still
    running or stopped being observed an hour ago.
    ``activity_max_age_seconds`` is the bound, and a feed quieter than it is
    treated as absent rather than as continuing.
    """

    at: float
    active: bool


@dataclass(frozen=True)
class EnvelopeSample:
    """Short-term amplitude of the caller track at one instant.

    Any consistent non-negative amplitude unit (RMS is the obvious one).  The
    modulation feature is a RATIO of this signal to its own mean, so the unit
    cancels and the worker does not have to normalise anything.
    """

    at: float
    amplitude: float


@dataclass(frozen=True)
class WordEvidence:
    """Recognised caller words delivered at one instant, and where from."""

    at: float
    words: int
    source: str = WORD_SOURCE_STT_FINAL


@dataclass(frozen=True)
class ClassifierThresholds:
    """Every number the verdict depends on, in one versioned place.

    Named rather than inline so the evidence recorded per call can be read
    against the thresholds that produced it, and so moving one is a reviewable
    change to a contract rather than an edit to an expression.
    """

    version: str = CLASSIFIER_VERSION
    #: A run shorter than this is a click, a breath or a codec artefact.
    utterance_min_seconds: float = 0.3
    #: A run longer than this is not one utterance.  It does not make the
    #: audio non-human -- it just stops being evidence of a person by SHAPE,
    #: and a real speaker who runs past it is caught by their words instead.
    utterance_max_seconds: float = 3.0
    #: The pause that turns a run into an utterance.  The same 300 ms is the
    #: continuity rule's largest tolerated gap, deliberately: one number
    #: separates "speech with breaths in it" from "a sustained source".
    utterance_trailing_silence_seconds: float = 0.3
    continuous_activity_seconds: float = 8.0
    continuity_max_gap_seconds: float = 0.3
    #: The syllable band.  Speech modulates its own envelope at roughly 2-8 Hz
    #: (one syllable per 125-500 ms) whatever the language; sustained music and
    #: tones do not, and that difference survives 8 kHz transcoding better than
    #: any spectral feature does.
    modulation_low_hz: float = 2.0
    modulation_high_hz: float = 8.0
    #: Below this depth the envelope is flat enough to be a sustained source.
    #:
    #: Measured, not assumed, and measured at the WORST point of the band
    #: rather than at a convenient one.  The pass attenuates most at 8 Hz, so
    #: that is where a real talker comes closest to looking flat:
    #:
    #: ===================  ==========================
    #: true modulation      worst measured across 2-8 Hz
    #: ===================  ==========================
    #: talker, index 0.20   0.138
    #: talker, index 0.25   0.173
    #: talker, index 0.30   0.207
    #: talker, index 0.40   0.276
    #: sustained, index 0.15  0.112 (its own maximum)
    #: sustained, index 0.10  0.075
    #: sustained, index 0.05  0.037
    #: ===================  ==========================
    #:
    #: 0.25 was chosen against a single 0.8-index fixture and would have
    #: classified a compressed talker with a 0.30 index at 6-8 Hz as
    #: non-human -- a real person, hung up on, from an envelope that is not
    #: remotely flat.  0.15 sits in the gap the table actually leaves: it
    #: keeps a talker down to roughly a 0.22 index, and still catches a
    #: sustained source wobbling up to roughly 0.20.
    #:
    #: Below a 0.22 index the two populations overlap, and this line does NOT
    #: resolve that overlap in the caller's favour: a wordless talker who
    #: never pauses, at a 0.20 index near the top of the band, measures 0.138
    #: and is classified non-human.  Nothing downstream rescues them either,
    #: since the arbiter's cap-hold reads the same measurement.  Say so
    #: plainly here, because the next person to move this number will be
    #: deciding where that boundary sits, and a comment promising a fail-safe
    #: that the code does not implement would let them move it in the belief
    #: that the cost is only a missed tune.
    modulation_depth_threshold: float = 0.15
    #: The envelope has to be sampled fast enough to SEE 8 Hz and to cover the
    #: run being judged.  Both are fail-closed: unmet means the feature is
    #: unknown, never that it passed.
    envelope_max_hop_seconds: float = 0.06
    envelope_min_coverage: float = 0.8
    envelope_min_samples: int = 32
    #: How much of a stretch the transform reads, counting back from its end.
    #: The cost of the transform is linear in the samples it is given and this
    #: runs on the event loop of a live call, so an unbounded stretch would
    #: make the classifier slower exactly on the calls that last longest -- at
    #: the 90 s ceiling, twice the work of a 45 s one, on the thread carrying
    #: the caller's audio.  Modulation depth is a stationary property of the
    #: source, so the most recent seconds answer the same question at a fixed
    #: price.  Ten of them is comfortably above the eight the continuity rule
    #: requires.
    modulation_span_seconds: float = 10.0
    #: How many such spans are probed, counting back from the end of the
    #: stretch.  One is not enough, and the reason is a call that ends with a
    #: person on it: a caller says hello, is put on hold, and fourteen seconds
    #: of music follow.  The gap between them is under the continuity
    #: tolerance so it is all one stretch, and a measurement taken only at the
    #: end reports a flat envelope and hangs up on somebody who answered.
    #: Three bounds the cost at three passes while reaching back far enough to
    #: find a talker at the start of a stretch as long as the hard cap.
    modulation_probe_windows: int = 3
    #: The window used at the very start of a stretch, where an answer lives.
    #: Short on purpose: voice activity does not always split a greeting from
    #: the music that follows it, and when it does not, the greeting is only
    #: whatever fraction of the window it occupies.  Measured against a
    #: compressed talker at a 0.25 modulation index speaking for 1.5 s with
    #: the music running straight on: a ten-second window reads 0.084 and
    #: hangs up on them, three seconds reads 0.162, and 1.5 s reads 0.233,
    #: while a pure tune reads 0.008 at every one of those lengths.  It must
    #: stay at or above two cycles of the low edge or nothing is measurable.
    head_probe_seconds: float = 1.5
    #: The transform is linear in the samples it is handed, and the caller's
    #: sample rate is its own business.  An envelope arriving at a media
    #: sample rate rather than a frame rate would be hundreds of thousands of
    #: points per probe, on the event loop of a live call, so it is thinned to
    #: this many before any arithmetic runs.
    envelope_max_samples: int = 1024
    #: How long the far end must hold the floor, unbroken, before this stops
    #: being a conversation.  The other rules here ask what the audio IS; this
    #: one asks whether anybody is conversing, and it is the only one that
    #: separates a caller tune from a person at all.
    #:
    #: Measured by replaying the whole finconnect batch of 2026-09-15 --
    #: seventeen completed calls, caller track -- through the same Silero
    #: voice-activity model this worker loads, at the same 0.25 s minimum
    #: silence.
    #:
    #: =========================  ==================
    #: call                       longest floor held
    #: =========================  ==================
    #: the four tune calls        at least 24.7 s each
    #: highest of the other 13    8.3 s
    #: =========================  ==================
    #:
    #: Twenty ends all four and none of the other thirteen.  Replayed at
    #: twenty they end at 51.0, 65.3, 74.5 and 74.5 s, each on the tick its
    #: own hold crosses the line and all of them well past ``cap_seconds``
    #: (45 s, in ``post_answer_arbiter``).  A song stops between phrases and
    #: between tracks -- each of those calls produces 22 to 33 separate runs
    #: of activity -- so its first UNBROKEN twenty seconds comes late.  What
    #: lets a tune survive this rule is pausing OFTEN, not pausing for long.
    #:
    #: What twenty costs, stated rather than argued away: a real person who
    #: holds the floor for twenty unbroken seconds at 80% activity, having
    #: never once left a one-second gap, is silenced and then ended.  No call
    #: in this batch came within 11 s of that, but seventeen calls in one
    #: language is not evidence about every caller, and shadow has not run.
    #: Two of the seventeen are seconds long with no transcript on either
    #: side and are unlabelled; both sit far below any line considered.
    conversation_hold_seconds: float = 20.0
    #: The pause that would make it a conversation.  Longer than the 300 ms
    #: that separates a syllable gap from a stopped source: this is the pause
    #: where a person would expect an answer.
    conversation_gap_seconds: float = 1.0
    #: How much of that span the far end must actually be producing audio
    #: through.  Without it, a call with sparse activity and gaps just under
    #: the tolerance accumulates a long "held floor" that is mostly silence --
    #: which it did, on my first measurement, firing on seven of seventeen
    #: calls including one where the far end was active for 1.7 s in total.
    conversation_active_fraction: float = 0.8
    #: How long after a run has ended a recogniser's word may still be
    #: attributed to it.  Without this a word delivered in the moment after a
    #: phrase, once the NEXT sound has already begun, is attributed to that
    #: next sound: the caller says hello, hold music starts 200 ms later, the
    #: transcript arrives at 500 ms, and the word that proves a person is
    #: credited to the music and discarded.
    word_attribution_lag_seconds: float = 1.0
    #: Sustained audio that was already running this soon after the answer is
    #: reported as a caller tune rather than hold music.  Provenance only; see
    #: the module docstring.
    tune_onset_seconds: float = 1.0
    #: How long the activity feed may go quiet before it stops being an
    #: observation.  The samples are a step function whose last value holds
    #: forward, so a feed that DIED mid-run would otherwise keep accumulating
    #: continuous activity out of one stale sample and reach the eight-second
    #: floor on its own.  The caller's side of this contract is that it
    #: samples the detector periodically rather than only on transitions;
    #: three seconds is generous against any such period and far below the
    #: floor it protects.
    activity_max_age_seconds: float = 3.0


DEFAULT_THRESHOLDS = ClassifierThresholds()


@dataclass(frozen=True)
class AnswerClassification:
    """One verdict about one window, with the evidence that produced it."""

    verdict: str
    detail: str
    score: float
    reversed_by: str
    version: str
    armed_at: float
    evaluated_at: float
    #: A modulation depth was measured and it is at or above the speech
    #: threshold.  Carried as its own flag rather than left for a consumer to
    #: recover from the feature summary, because it is the one reading that
    #: says "whatever else this is, something in it is shaped like a voice",
    #: and the arbiter needs it to decide whether a cap may still fire.  False
    #: when no depth could be measured: an absent measurement is not a voice.
    speech_like_modulation: bool = False
    #: The far end has held the floor long enough that no conversation is
    #: happening, whatever the verdict says the audio is.  Carried beside the
    #: verdict rather than inside it because it answers a different question,
    #: and because it is the only signal here that may stand while the verdict
    #: is ``human``.
    no_conversation: bool = False
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def is_human(self) -> bool:
        return self.verdict == VERDICT_HUMAN

    @property
    def is_non_human(self) -> bool:
        return self.verdict == VERDICT_NON_HUMAN_AUDIO

    def evidence(self) -> dict[str, Any]:
        """Bounded scalars for telemetry.  Never audio, never transcript text."""

        record = {
            "answer_classification": self.verdict,
            "reason_detail": self.detail,
            "score": round(float(self.score), 4),
            "reversed_by": self.reversed_by,
            "classifier_version": self.version,
            "window_seconds": round(max(0.0, self.evaluated_at - self.armed_at), 3),
        }
        record.update(self.features)
        return record


@dataclass(frozen=True)
class _Run:
    started_at: float
    ended_at: float
    #: False when the run was still going at ``now``: an open run has no
    #: trailing silence and therefore cannot be utterance-shaped yet.
    closed: bool

    @property
    def seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


def activity_runs(
    samples: Iterable[ActivitySample], *, after: float, now: float
) -> tuple[_Run, ...]:
    """Contiguous stretches of voice activity in ``[after, now]``.

    The samples are transitions, so a run opens at the first ``active`` sample
    at or after ``after`` and closes at the next inactive one.  A run still
    open at ``now`` is returned with ``closed=False`` rather than being
    dropped: the continuity rule needs it (music does not stop to be counted)
    and the utterance rule must refuse it (a person who has not paused yet has
    not finished an utterance).
    """

    ordered = sorted(
        (sample for sample in samples if sample.at <= now),
        key=lambda sample: sample.at,
    )
    runs: list[_Run] = []
    opened_at: float | None = None
    for sample in ordered:
        # A run already in progress when the window opened starts AT the
        # window: pre-answer ring-back must never be counted as post-answer
        # audio, but the activity that continues past the answer is real.
        at = max(float(sample.at), after)
        if sample.active:
            if opened_at is None:
                opened_at = at
        elif opened_at is not None:
            if at > opened_at:
                runs.append(_Run(started_at=opened_at, ended_at=at, closed=True))
            opened_at = None
    if opened_at is not None and now >= opened_at:
        # ``>=`` so that a sample saying "active" that lands exactly at ``now``
        # opens a run of zero length rather than vanishing.  Dropped, the only
        # observation of the window was "there is sound" and the verdict came
        # back ``silent``.
        runs.append(_Run(started_at=opened_at, ended_at=now, closed=False))
    return tuple(runs)


def _trailing_silence(
    runs: Sequence[_Run], index: int, *, observed_until: float
) -> float:
    """How long the line stayed quiet after this run, as far as anyone saw."""

    run = runs[index]
    following = runs[index + 1] if index + 1 < len(runs) else None
    return (
        following.started_at if following is not None else observed_until
    ) - run.ended_at


def _utterance_shaped(
    runs: Sequence[_Run], *, observed_until: float, thresholds: ClassifierThresholds
) -> tuple[_Run, ...]:
    """Runs of 0.3-3 s whose trailing silence has actually been OBSERVED.

    The trailing silence is what makes a run an utterance rather than the
    first 3 s of a tune, so it is required to have happened: either the next
    run starts at least ``utterance_trailing_silence_seconds`` later, or no
    run follows and ``observed_until`` is already that far past this one.  A
    closed run followed immediately by more activity is not an utterance.

    ``observed_until`` is how far the feed can be believed, which is not the
    same as how far the clock has run.  A detector that stopped reporting
    proves silence up to its freshness horizon and no further, so a dead feed
    cannot manufacture an utterance out of the hours after it died -- and,
    equally, an utterance whose silence WAS observed before the feed died
    stays an utterance.
    """

    shaped: list[_Run] = []
    for index, run in enumerate(runs):
        if not run.closed:
            continue
        if not (
            thresholds.utterance_min_seconds
            <= run.seconds
            <= thresholds.utterance_max_seconds
        ):
            continue
        silence = _trailing_silence(runs, index, observed_until=observed_until)
        if silence >= thresholds.utterance_trailing_silence_seconds:
            shaped.append(run)
    return tuple(shaped)


def _current_floor_span(
    runs: Sequence[_Run],
    *,
    observed_until: float,
    thresholds: ClassifierThresholds,
) -> tuple[float, float, float] | None:
    """The span the far end is holding RIGHT NOW, or ``None``.

    Runs merged across gaps shorter than a reply.  Only the last span is
    wanted, so none of the earlier ones is kept; the merge itself is
    unchanged, including its running maximum end, which is what keeps a run
    nested inside a longer one from reading as a boundary.
    """

    if not runs:
        return None
    ordered = sorted(runs, key=lambda run: run.started_at)
    span: tuple[float, float, float] | None = None
    index = 0
    while index < len(ordered):
        start = ordered[index].started_at
        end = min(ordered[index].ended_at, observed_until)
        active = max(0.0, end - start)
        following = index + 1
        while (
            following < len(ordered)
            and ordered[following].started_at - end < thresholds.conversation_gap_seconds
        ):
            run_end = min(ordered[following].ended_at, observed_until)
            active += max(0.0, run_end - ordered[following].started_at)
            end = max(end, run_end)
            following += 1
        span = (start, end, active)
        index = following
    return span


def held_floor_seconds(
    runs: Sequence[_Run],
    *,
    observed_until: float,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> float:
    """How long the far end has been holding the floor RIGHT NOW.

    Not the longest span it ever held: a maximum could never be released, so
    a person who takes over from the music could not clear it however long
    they talk.  Reading the current span makes the release automatic -- the
    far end pauses, the span ends, this falls to nothing.

    A span qualifies when no gap inside it reaches ``conversation_gap_seconds``
    AND the far end is producing audio for ``conversation_active_fraction`` of
    it.  Both halves are load-bearing: the first alone reads a sparse call
    with sub-second gaps as one long hold, and the second alone would be
    satisfied by any dense burst however short.
    """

    span = _current_floor_span(
        runs, observed_until=observed_until, thresholds=thresholds
    )
    if span is None:
        return 0.0
    start, end, active = span
    # ``<=``, to match the merge above: a gap of exactly
    # ``conversation_gap_seconds`` splits two runs into separate spans, so a
    # trailing pause of exactly that length has to release the hold as well.
    # Strict on one side and not the other made a 1.0 s pause a gap when it
    # fell between two runs and not a gap when it was the last thing heard.
    if end <= observed_until - thresholds.conversation_gap_seconds:
        return 0.0
    span = end - start
    if span <= 0 or active / span < thresholds.conversation_active_fraction:
        return 0.0
    return span


def _continuous_stretch(
    runs: Sequence[_Run], *, thresholds: ClassifierThresholds
) -> _Run | None:
    """The longest stretch whose internal gaps all stay under the tolerance.

    Hold music with half-second rests is NOT continuous by this rule and is
    therefore never classified non-human -- an accepted miss, recorded in the
    adversarial review of 2026-09-15 (finding 5), and one the utterance rule
    usually converts into ``human`` outright.  Widening the gap tolerance to
    catch it would put ordinary speech pauses inside a "continuous" stretch,
    which is the failure this threshold exists to prevent.
    """

    best: _Run | None = None
    stretch_start: float | None = None
    previous_end: float | None = None
    for run in runs:
        if stretch_start is None or previous_end is None:
            stretch_start = run.started_at
        elif run.started_at - previous_end >= thresholds.continuity_max_gap_seconds:
            stretch_start = run.started_at
        previous_end = run.ended_at
        candidate = _Run(
            started_at=stretch_start, ended_at=run.ended_at, closed=run.closed
        )
        if best is None or candidate.seconds > best.seconds:
            best = candidate
    return best


def _boxcar(values: Sequence[float], width: int) -> list[float]:
    """Centred moving average of ``width`` samples, by prefix sum."""

    half = width // 2
    totals = [0.0]
    for value in values:
        totals.append(totals[-1] + value)
    smoothed: list[float] = []
    for index in range(len(values)):
        low = max(0, index - half)
        high = min(len(values), index + half + 1)
        smoothed.append((totals[high] - totals[low]) / (high - low))
    return smoothed


def modulation_depth(
    samples: Sequence[EnvelopeSample],
    *,
    start: float,
    end: float,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> float | None:
    """Syllable-band modulation depth of the envelope, or ``None`` if unmeasured.

    ``None`` is returned whenever the samples cannot support the measurement --
    too few, too sparse to represent the top of the band, not covering enough
    of the stretch, or a non-positive mean.  It is a THIRD value, never a low
    depth: the non-human rule requires a measured depth BELOW the threshold, so
    a feed that is missing or damaged can never be read as "flat, therefore
    music".

    Only the last ``modulation_span_seconds`` of ``[start, end]`` are measured,
    and at most ``envelope_max_samples`` of those samples reach the arithmetic.
    The selection itself still looks at every sample it is handed, so the cost
    of this call grows with the size of that list and not with the window --
    which is bounded only because the caller trims its history, as
    :func:`classify` says it must.

    The band is isolated with two centred moving averages -- one whose length
    is a full cycle of the low edge, one a half cycle of the high edge -- and
    the depth is the root-mean-square of their difference against the
    envelope's own mean, scaled so that a pure sinusoid returns its own
    modulation index.  It is deliberately NOT a bank of discrete frequency
    probes.  A probe bank only answers at the frequencies it probes: over a
    ten-second window each probe is about a tenth of a hertz wide, so a
    talker whose syllable rate falls between two of them correlates with
    neither and measures as FLAT.  That is not a rounding error, it is a
    caller being hung up on for speaking at 4.25 syllables a second, and no
    practical probe spacing removes it.  A moving-average pass has no gaps
    between frequencies at all, and it costs one pass over the samples.

    Its price is a passband that is not flat: measured against pure sinusoids,
    this returns between roughly 0.7 (at 8 Hz) and 1.2 (around 3 Hz) of the
    true modulation index.  The gain matters most where the two populations
    approach each other, which is why the threshold beside
    ``modulation_depth_threshold`` is chosen from the worst point of the band
    at several modulation indices rather than from one comfortable fixture.
    """

    analysed_from = max(start, end - thresholds.modulation_span_seconds)
    # Windowed before sorting, not after: this runs per tick on the event loop
    # of a live call, and the history handed in is the caller's whole call.
    cap = max(1, thresholds.envelope_max_samples)
    selected: list[EnvelopeSample] = []
    stride = 1
    seen = 0
    for sample in samples:
        if not (analysed_from <= sample.at <= end):
            continue
        if not math.isfinite(sample.amplitude) or sample.amplitude < 0:
            # ``nan`` compares False against every bound, so it passes a
            # ``< 0`` test, poisons the mean, and lands in the telemetry
            # record as a value no JSON encoder will take.
            continue
        if seen % stride:
            seen += 1
            continue
        seen += 1
        selected.append(sample)
        if len(selected) > 2 * cap:
            selected = selected[::2]
            stride *= 2
    # Thinned WHILE selecting, so neither the sort nor the list ever reaches
    # full size.  An envelope arriving at a media sample rate is a hundred and
    # sixty thousand points per probe: sorting them to then throw all but a
    # thousand away put the cost back on the event loop that the thinning
    # exists to protect, and materialising them first did the same to memory.
    # One pass is the floor -- finding the window without one means assuming
    # the caller's samples are ordered, which is its contract but not this
    # function's to enforce.
    window = sorted(selected, key=lambda sample: sample.at)
    # Proportional to the span being measured, not a flat count.  A window of
    # 1.5 s at the coarsest permitted hop holds 26 samples, so a flat floor of
    # 32 made a perfectly legal envelope void every verdict -- the opening
    # probe could never be satisfied and the whole stretch went unjudged.
    required = max(
        2,
        min(
            thresholds.envelope_min_samples,
            int((end - analysed_from) / max(thresholds.envelope_max_hop_seconds, 1e-9)),
        ),
    )
    if len(window) < required:
        return None
    # Two cycles of the low edge, or the pass is reading a fraction of one
    # oscillation and calling it a depth.
    if (end - analysed_from) < 2.0 / max(thresholds.modulation_low_hz, 1e-9):
        return None
    span = window[-1].at - window[0].at
    if span <= 0:
        return None
    if span < max(0.0, (end - analysed_from)) * thresholds.envelope_min_coverage:
        return None
    hops = [window[index + 1].at - window[index].at for index in range(len(window) - 1)]
    if max(hops) > thresholds.envelope_max_hop_seconds:
        return None
    amplitudes = [sample.amplitude for sample in window]
    mean = sum(amplitudes) / len(amplitudes)
    if mean <= 0:
        return None

    hop = span / (len(window) - 1)
    slow_width = max(3, int(round(1.0 / max(thresholds.modulation_low_hz, 1e-9) / hop)))
    fast_width = max(
        1, int(round(1.0 / (2 * max(thresholds.modulation_high_hz, 1e-9)) / hop))
    )
    if len(amplitudes) < 2 * slow_width:
        # Not enough samples for the low edge's own cycle to sit inside the
        # window with anything left to measure.
        return None
    slow = _boxcar(amplitudes, slow_width)
    fast = _boxcar(amplitudes, fast_width)
    margin = slow_width // 2
    band = [fast[index] - slow[index] for index in range(margin, len(amplitudes) - margin)]
    if not band:
        return None
    power = sum(value * value for value in band) / len(band)
    return math.sqrt(2.0 * power) / mean


def stretch_modulation_depth(
    samples: Sequence[EnvelopeSample],
    runs: Sequence[_Run] = (),
    *,
    start: float,
    end: float,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> tuple[float | None, int]:
    """The most speech-like reading anywhere in a stretch, and how many probes.

    A non-human verdict is a claim about the WHOLE stretch, so one measurement
    at its end cannot support it.  Up to ``modulation_probe_windows`` spans
    are read, counting back from the end, and the largest depth any of them
    returns is the answer: a talker anywhere in the stretch keeps it above the
    threshold.

    The spans are spread ACROSS the stretch rather than packed at its end.
    Packed, three of them reach back thirty seconds, and the caller who said
    hello at the start of a seventy-second stretch of hold music is outside
    all of them -- which is the same hang-up this exists to prevent, moved
    later in the call.  Spread, the first span always covers the beginning,
    where a person who answered and then went quiet actually is.  The cost is
    identical and the coverage is sparse in between; a talker who speaks only
    in an unprobed middle is missed here, and is caught instead by the
    arbiter's rule that a voice in the CURRENT reading holds the cap open.

    The last span decides whether an answer exists at all.  Earlier spans that
    cannot be measured -- because the caller trimmed its envelope, or the feed
    had not started -- are skipped rather than voiding the result, since
    otherwise any gap in the envelope would make every verdict impossible.
    The number of spans actually read is returned with the depth, so the
    record can say how much of the stretch the verdict rests on.
    """

    span = thresholds.modulation_span_seconds
    if span <= 0:
        return None, 0
    tail = modulation_depth(samples, start=start, end=end, thresholds=thresholds)
    if tail is None:
        return None, 0
    peak = tail
    measured_windows = {(round(max(start, end - span), 3), round(end, 3))}
    minimum = 2.0 / max(thresholds.modulation_low_hz, 1e-9)

    first = next((run for run in runs if run.ended_at > start), None)
    if first is not None:
        # The opening of the stretch is measured over the FIRST RUN, not over
        # a fixed span starting at the same place.  A greeting swallowed by
        # hold music is a second and a half of voice; averaged into ten
        # seconds of the music that follows it, its band energy is divided by
        # the square root of the ratio and a perfectly ordinary talker reads
        # as flat.  The window is short enough to hold a greeting and little
        # else, because voice activity does not always split one from the
        # music: on a call where it never drops there is no run boundary to
        # measure up to, and only the window's own length keeps the greeting
        # from being averaged away.
        head_start = max(start, first.started_at)
        head_end = min(first.ended_at, head_start + thresholds.head_probe_seconds)
        if head_end - head_start < minimum:
            # The first run is too short to measure -- a click as the line
            # connects, say.  Measuring ACROSS it rather than skipping is what
            # keeps the opening from going unread: a run that short holds no
            # greeting, so there is nothing to dilute, and skipping would let
            # a verdict rest entirely on the tail while the seconds a person
            # might have spoken in were never measured at all.
            head_end = min(end, head_start + thresholds.head_probe_seconds)
        window = (round(head_start, 3), round(head_end, 3))
        if head_end - head_start >= minimum and window not in measured_windows:
            head = modulation_depth(
                samples, start=head_start, end=head_end, thresholds=thresholds
            )
            if head is None:
                # The beginning of a stretch is the one place a verdict may
                # not be reached by DEFAULT.  A worker that started measuring
                # amplitude late, or trimmed its envelope to the span the
                # transform reads, would otherwise have the flat tail speak
                # for audio nobody measured -- an absence used as evidence
                # for the verdict that ends calls.
                return None, 0
            measured_windows.add(window)
            peak = max(peak, head)

    wanted = max(1, thresholds.modulation_probe_windows)
    while len(measured_windows) < wanted and (end - span) > start:
        step = (end - span - start) / max(1, wanted - 1)
        placed = False
        for index in range(wanted):
            probe_end = min(end - span, start + span + index * step)
            probe_start = max(start, probe_end - span)
            window = (round(probe_start, 3), round(probe_end, 3))
            if probe_end - probe_start < minimum or window in measured_windows:
                continue
            measured_windows.add(window)
            placed = True
            measured = modulation_depth(
                samples, start=probe_start, end=probe_end, thresholds=thresholds
            )
            if measured is not None:
                peak = max(peak, measured)
            break
        if not placed:
            break
    return peak, len(measured_windows)


def counted_words(
    words: Iterable[WordEvidence],
    runs: Sequence[_Run],
    *,
    after: float,
    now: float,
    attributable: Collection[_Run] = (),
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
) -> tuple[int, int]:
    """Words whose audio has ended, and words still inside unbroken audio.

    A recognised word is attributed to the last run of activity that began at
    or before it, never to a run it happens to sit inside.  Recognisers report
    after the fact: an STT final or a caption for a phrase arrives in the
    silence that ended it, and sometimes not until the next phrase has begun.
    Requiring containment would therefore discard the ordinary case and count
    only the late ones.

    A word attributed to a run that is still going does not count.  That is
    the whole revision of 2026-09-15: a song transcribes as words, and the
    only thing separating a song from a talker at that moment is that the
    talker stops.
    """

    counted = 0
    inside_unbroken = 0
    ordered = sorted(runs, key=lambda run: run.started_at)
    for evidence in words:
        try:
            spoken = max(0, int(evidence.words))
        except (TypeError, ValueError):
            # Total by contract.  A damaged count is no count; raising here
            # would take down the decision path of a live call over a field
            # that only ever adds evidence.
            continue
        if spoken <= 0 or not (after <= evidence.at <= now):
            continue

        landed_in: _Run | None = None
        for run in ordered:
            if run.started_at <= evidence.at:
                landed_in = run
            else:
                break

        owner: _Run | None = None
        if (
            landed_in is not None
            and landed_in in attributable
            and evidence.at
            <= landed_in.ended_at + thresholds.word_attribution_lag_seconds
        ):
            owner = landed_in
        else:
            # The run this word landed in cannot hold it -- it is still going,
            # or it is a click, or it is a stretch of music split by a
            # detector flicker.  A recogniser reports late, so the phrase may
            # be the one that ended just before this sound started.  The most
            # RECENT eligible run within the lag wins; there is no break,
            # because the list is in ascending order.
            for candidate in ordered:
                if (
                    candidate in attributable
                    and candidate.ended_at <= evidence.at
                    and evidence.at - candidate.ended_at
                    <= thresholds.word_attribution_lag_seconds
                ):
                    owner = candidate

        if owner is not None:
            counted += spoken
        elif landed_in is not None and not landed_in.closed:
            # Heard inside audio that has not stopped.  Not evidence of a
            # person, and recorded separately so the shadow population can say
            # how much of what a recogniser produced was discarded this way.
            inside_unbroken += spoken
    return counted, inside_unbroken


def classify(
    *,
    activity: Sequence[ActivitySample] = (),
    envelope: Sequence[EnvelopeSample] = (),
    words: Sequence[WordEvidence] = (),
    armed_at: float,
    now: float,
    thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS,
    words_unobservable_until: float | None = None,
) -> AnswerClassification:
    """Classify everything observed in ``[armed_at, now]``.

    Pure and total: the same samples always give the same verdict, and there
    is no input for which this raises.  The cost is bounded by the window, not
    by the length of the call, but only because the CALLER keeps the samples
    trimmed to the window it asks about: this runs on the event loop of a live
    call and a history nobody trimmed would be re-read on every tick.

    Nothing before ``armed_at`` is counted, and a run already in progress at
    the answer is credited only from the answer onward -- so pre-answer
    ring-back can lengthen no measurement and satisfy no rule, while audio
    that genuinely continues across the answer is not thrown away either.

    ``words_unobservable_until`` is the end of a window in which no word
    COULD have been recognised -- a voice-to-voice engine whose input is held
    while its opening plays captions nothing. Zero words from that window is
    not evidence, so a non-human verdict additionally needs
    :data:`WORDS_OBSERVABLE_MIN_SECONDS` of its stretch AFTER the window: the
    recogniser must have had a chance to hear a word before "no words" counts.
    Deliberately not the whole 8 s continuity: the arbiter accepts a verdict
    only inside its decision window (15 s by default), and music that began
    at the answer behind a 7 s opening would otherwise reach 8 s of
    observable audio only after that window had closed. Every other feature
    is unchanged.
    """

    # Only samples INSIDE the window describe the window.  A transition from
    # before the answer is still used to open a run that continues across it
    # (``activity_runs`` clamps that), but it is not an observation of the
    # window, and treating it as one would let a single pre-answer sample
    # stand in for a feed that never reported after the call was answered.
    in_window = [
        sample for sample in activity if armed_at <= sample.at <= now
    ]
    latest_sample = max(in_window, key=lambda sample: sample.at, default=None)
    latest_activity_at = None if latest_sample is None else latest_sample.at
    runs = activity_runs(activity, after=armed_at, now=now)
    # Stale means the detector STOPPED while it had something to report.  A
    # newest sample that says "inactive" is observed silence however old it
    # is, and calling that a dead feed would retract an utterance-shaped
    # human verdict as the clock moved past it.  A run credited with no
    # in-window sample at all is stale by construction: it exists only
    # because a sample from before the answer was held forward.
    feed_stale = bool(runs) and (
        latest_sample is None
        or (
            latest_sample.active
            and now - latest_sample.at > thresholds.activity_max_age_seconds
        )
    )
    observed_until = (
        now
        if latest_activity_at is None
        else min(now, latest_activity_at + thresholds.activity_max_age_seconds)
    )
    shaped = _utterance_shaped(
        runs, observed_until=observed_until, thresholds=thresholds
    )
    # Which runs a recognised word may be attributed to at all.  A run holds
    # something a person said if it is the length of an utterance, or if it is
    # longer but genuinely stopped afterwards.  Everything else -- a click as
    # the line connects, a stretch of music split by a detector flicker --
    # carries no word, and crediting one to it turns noise into a caller.
    attributable = {
        run
        for index, run in enumerate(runs)
        if run.closed
        and run.seconds >= thresholds.utterance_min_seconds
        and (
            run.seconds <= thresholds.utterance_max_seconds
            or _trailing_silence(runs, index, observed_until=observed_until)
            >= thresholds.utterance_trailing_silence_seconds
        )
    }
    word_count, words_inside_unbroken = counted_words(
        words,
        runs,
        after=armed_at,
        now=now,
        attributable=attributable,
        thresholds=thresholds,
    )
    stretch = _continuous_stretch(runs, thresholds=thresholds)
    held_floor = held_floor_seconds(
        runs, observed_until=observed_until, thresholds=thresholds
    )
    no_conversation = held_floor >= thresholds.conversation_hold_seconds
    observed_activity = bool(in_window)
    # A run of utterance LENGTH that ended inside the stretch, whose following
    # pause was too short to make it an utterance.  It is not proof of a
    # person -- that is why it is not a ``human`` verdict -- but it is proof
    # that this audio is not one unbroken source, and a second and a half of
    # it is too short to measure reliably at the bottom of the modulation
    # band.  Refusing the non-human verdict outright is the only answer that
    # does not depend on a measurement the window is too short to take.
    interrupted = [
        run
        for index, run in enumerate(runs)
        if run.closed
        and run.seconds >= thresholds.utterance_min_seconds
        and (
            # It STOPPED, for at least as long as the continuity rule's own
            # tolerance.  A sustained source does not do that, so whatever
            # made this sound is not the thing the verdict would describe.
            # Any length qualifies here, and that is the point: a caller who
            # answers with a five-second sentence and then pauses is not
            # utterance-shaped (that rule tops out at three seconds) and
            # produces no counted word until the recogniser reports, so
            # without this the hold music that follows them satisfies every
            # non-human clause on its own.
            _trailing_silence(runs, index, observed_until=observed_until)
            >= thresholds.utterance_trailing_silence_seconds
            # Or it ended inside the stretch itself, where the pause after it
            # was too short to qualify -- a greeting swallowed by the music
            # that starts 200 ms later.  Bounded by utterance length here,
            # because voice activity flickers on music and a fifty-millisecond
            # hole in a long tune closes a run that means nothing.
            or (
                run.seconds <= thresholds.utterance_max_seconds
                and stretch is not None
                and stretch.started_at <= run.started_at
                and run.ended_at <= stretch.ended_at
            )
        )
    ]

    # Measured on the longest stretch whenever one exists, not only on a
    # stretch already long enough to end a call: the shadow rollout's whole
    # purpose is a depth distribution over real human calls, and a feature
    # recorded only on the calls that were about to be classified non-human
    # would be sampled on exactly the population that cannot answer the
    # question.
    depth: float | None = None
    probes = 0
    if stretch is not None:
        depth, probes = stretch_modulation_depth(
            envelope,
            [run for run in runs if run.ended_at > stretch.started_at],
            start=stretch.started_at,
            end=stretch.ended_at,
            thresholds=thresholds,
        )

    features: dict[str, Any] = {
        "activity_seconds": round(sum(run.seconds for run in runs), 3),
        "runs": len(runs),
        "utterance_runs": len(shaped),
        "utterance_length_runs_inside": len(interrupted),
        "longest_run_seconds": round(
            max((run.seconds for run in runs), default=0.0), 3
        ),
        "continuous_seconds": round(stretch.seconds if stretch else 0.0, 3),
        "continuous_started_at_seconds": round(
            (stretch.started_at - armed_at) if stretch else 0.0, 3
        ),
        "words": word_count,
        "words_inside_unbroken_audio": words_inside_unbroken,
        "envelope_samples": len(envelope),
        "modulation_depth": None if depth is None else round(depth, 4),
        "modulation_probes": probes,
        "activity_feed_stale": feed_stale,
        "held_floor_seconds": round(held_floor, 3),
        "no_conversation": no_conversation,
    }
    # The margin by which the envelope is flatter than speech, and nothing
    # else.  It is EVIDENCE, never a second decision path: every verdict below
    # is decided by the named thresholds, so a score cannot move one.
    score = (
        0.0
        if depth is None
        else max(0.0, min(1.0, 1.0 - depth / thresholds.modulation_depth_threshold))
    )

    speech_like = depth is not None and depth >= thresholds.modulation_depth_threshold
    features["speech_like_modulation"] = speech_like

    def verdict(
        value: str,
        *,
        detail: str = DETAIL_NONE,
        reversed_by: str = REVERSED_BY_NOTHING,
    ) -> AnswerClassification:
        return AnswerClassification(
            verdict=value,
            detail=detail,
            score=score,
            reversed_by=reversed_by,
            version=thresholds.version,
            armed_at=armed_at,
            evaluated_at=now,
            speech_like_modulation=speech_like,
            no_conversation=no_conversation,
            features=features,
        )

    if word_count > 0:
        return verdict(VERDICT_HUMAN, reversed_by=REVERSED_BY_WORD)
    if shaped:
        # Read before the staleness check, and deliberately.  An utterance
        # whose trailing silence was observed is as complete a human signal as
        # a word from the same audio, and the feed dying afterwards does not
        # unmake it.  Ranking them differently would have left the shadow
        # record's human population missing the callers whose detector failed.
        return verdict(VERDICT_HUMAN, reversed_by=REVERSED_BY_UTTERANCE_SHAPE)
    if feed_stale:
        # The detector stopped reporting.  Neither claim is available now: the
        # activity that was running may have stopped unseen, and the quiet
        # that was there may have ended unseen.  A word is still evidence,
        # which is why it is read above this.
        return verdict(VERDICT_UNKNOWN)
    if observed_activity and not runs:
        # Samples arrived and every one of them said "inactive".  An absent
        # feed takes the ``unknown`` path below instead: a measurement nobody
        # took is not evidence of quiet.
        return verdict(VERDICT_SILENT, detail=DETAIL_NO_AUDIO)
    words_observable = True
    if stretch is not None and words_unobservable_until is not None:
        words_observable = (
            stretch.ended_at - max(stretch.started_at, float(words_unobservable_until))
            >= WORDS_OBSERVABLE_MIN_SECONDS
        )
    if (
        stretch is not None
        and not interrupted
        and words_observable
        and stretch.seconds >= thresholds.continuous_activity_seconds
        and depth is not None
        and depth < thresholds.modulation_depth_threshold
    ):
        started_within = stretch.started_at - armed_at <= thresholds.tune_onset_seconds
        return verdict(
            VERDICT_NON_HUMAN_AUDIO,
            detail=DETAIL_CALLER_TUNE if started_within else DETAIL_HOLD_MUSIC,
        )
    return verdict(VERDICT_UNKNOWN)


def with_version(thresholds: ClassifierThresholds, version: str) -> ClassifierThresholds:
    """A tuned threshold set carries its own version, never the default one."""

    return replace(thresholds, version=version)


__all__ = [
    "ActivitySample",
    "AnswerClassification",
    "CLASSIFIER_VERSION",
    "ClassifierThresholds",
    "DEFAULT_THRESHOLDS",
    "DETAIL_CALLER_TUNE",
    "DETAIL_HOLD_MUSIC",
    "DETAIL_LOOPING_RECORDING",
    "DETAIL_NONE",
    "DETAIL_NO_AUDIO",
    "EnvelopeSample",
    "REVERSED_BY_NOTHING",
    "REVERSED_BY_UTTERANCE_SHAPE",
    "REVERSED_BY_WORD",
    "VERDICTS",
    "VERDICT_DETAILS",
    "VERDICT_HUMAN",
    "VERDICT_NON_HUMAN_AUDIO",
    "VERDICT_SILENT",
    "VERDICT_UNKNOWN",
    "WORD_SOURCE_LIVE_CAPTION",
    "WORD_SOURCE_STT_FINAL",
    "WORDS_OBSERVABLE_MIN_SECONDS",
    "WordEvidence",
    "activity_runs",
    "classify",
    "counted_words",
    "held_floor_seconds",
    "modulation_depth",
    "stretch_modulation_depth",
    "with_version",
]

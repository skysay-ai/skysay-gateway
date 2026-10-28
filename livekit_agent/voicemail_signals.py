# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What the far end's audio and words say about a mailbox, as typed signals.

The producers for ``answer_observer.py``.  That module decides; this
one measures.  Neither imports livekit, so both are driven by tests with
synthetic PCM and nothing else.

Only two things here may ever authorise an action on a call, and both are
specific to a mailbox that is about to record:

* a **record beep after a spoken greeting** -- a single pure tone that follows
  speech, where "speech" is measured, not assumed; and
* a **mailbox phrase** in the caller's recognised words ("leave a message",
  "after the tone"), in the call's language or English.

Everything else this module produces is suspicion or a different state.  The
reasons are the founder's listening labels of 2026-09-23 and the adversarial
review of the research (``voicemail_codex.txt``):

* ``call_tlQcn8df6Dmp1q`` was music with bird sounds and then TWO beeps, with
  no spoken announcement at all.  A beep after music is a network tone as
  often as it is a mailbox, so a tone proves nothing unless the audio before
  it was SPEECH.  The research had read that music as an announcement from its
  energy contour; the speech test below is what would have told them apart:
  that call had no pause of 100 ms or more in eight seconds, and speech has
  several a second.
* A fast answer, a long greeting and a known recording only raise suspicion.
  A person on a headset answers instantly and talks for five seconds.
* Fax, SIT, DTMF and ringback tones fit a naive "pure tone" envelope, so each
  has an explicit rejection, and a tone that repeats is a cadence, not a beep.
* Our own voice comes back on the caller track through a speakerphone.  A
  tone heard while we were audible, or a phrase we said ourselves, is echo.

Pure Python over ``array('h')``: one Goertzel at one frequency over 160
samples is too small for an array round-trip to pay, and cheap enough to run for the
bounded window this is used in.  It never runs on the media drain itself --
the worker queues frames and drains them from its own task.
"""
from __future__ import annotations

import array
import math
import re
from collections import deque
from dataclasses import dataclass

try:  # pragma: no cover - import shape only
    from .post_answer_arbiter import normalize_far_end_text
except ImportError:  # pragma: no cover - direct execution inside the image
    from post_answer_arbiter import normalize_far_end_text  # type: ignore[no-redef]


#: Stamped on every record.  A threshold moved without moving this makes two
#: incomparable shadow populations look like one.
DETECTOR_VERSION = "voicemail-detector/2026-09-23"

#: The evidence sidecar's format: 8 kHz, 16-bit mono, 20 ms frames.
SAMPLE_RATE = 8_000
FRAME_SECONDS = 0.02

#: A frame whose energy is concentrated this much at one frequency is tonal.
#: A clean sine measures at least 0.98 anywhere in the beep band at any phase
#: or level (the estimator below is sub-sample, so the probe lands on the
#: tone).  0.95 leaves every other component together at most 5 % of the
#: energy, which by itself rejects two-tone signals: DTMF at the 8 dB twist a
#: receiver tolerates measures ~0.81.  Speech and noise measure under 0.1.
TONE_PURITY_MIN = 0.95
#: Frames of one tone stay within this fraction of the tone's median frequency.
TONE_FREQUENCY_TOLERANCE = 0.03
TONE_LEVEL_MIN_DBFS = -30.0
#: Record beeps observed in the field sit well inside this band (the 070
#: network tone of 2026-09-23 was 2200 Hz; many US systems use ~1000 Hz).
#: Below it are ringback and busy tones (350-480 Hz across plans).
BEEP_BAND_HZ = (500.0, 2600.0)
BEEP_SECONDS = (0.30, 2.0)
#: Fax calling tone (CNG, 1100 Hz, 0.5 s) and answer tone (CED, 2100 Hz,
#: 2.6-4 s).  CED is also excluded by duration; both are excluded by band so a
#: clipped CED cannot pass as a beep.
FAX_BANDS_HZ = ((1080.0, 1120.0), (2080.0, 2120.0))
#: Special information tone segments (913.8/985.2, 1370.6/1428.5, 1776.7 Hz,
#: each 274-380 ms) precede "number not in service" style announcements.
#: The third segment is listed as a band: plans round it to 1800 Hz.
SIT_BANDS_HZ = ((893.8, 1005.2), (1350.6, 1448.5), (1740.0, 1820.0))
SIT_MAX_SECONDS = 0.40
#: DTMF row and column frequencies.  A tone near one of them is DTMF when a
#: partner-group frequency also carries energy -- a check that stays explicit
#: rather than resting on the purity threshold alone.
DTMF_LOW_HZ = (697.0, 770.0, 852.0, 941.0)
DTMF_HIGH_HZ = (1209.0, 1336.0, 1477.0, 1633.0)
DTMF_MATCH_TOLERANCE = 0.015
DTMF_PARTNER_SHARE = 0.005
#: A tone with a same-frequency predecessor this recently is a cadence.
CADENCE_LOOKBACK_SECONDS = 8.0
#: And one with a same-frequency SUCCESSOR this soon.  A candidate beep is
#: only validated once this much media has passed after it with no repeat,
#: because the first tone of a cadence has nothing before it: the network
#: tones on ``call_tlQcn8df6Dmp1q`` were two beeps three seconds apart.
CADENCE_LOOKAHEAD_SECONDS = 4.0
#: A tone starting this close to our own audible playout may be our echo.
ECHO_GUARD_SECONDS = 0.30

#: The spoken-greeting test, run over the audio before a tone.
GREETING_LOOKBACK_SECONDS = 12.0
GREETING_MIN_ACTIVE_SECONDS = 1.5
GREETING_MAX_GAP_TO_TONE_SECONDS = 3.0
GREETING_ACTIVE_ABOVE_FLOOR_DB = 10.0
DIP_MIN_SECONDS = 0.10
DIP_BELOW_MEDIAN_DB = 10.0
#: Speech pauses at least twice per four seconds of greeting.
DIPS_PER_FOUR_SECONDS = 2

_SILENCE_DBFS = -96.0

# -- signal vocabulary (closed; the control plane validates the same set) ----

KIND_RECORD_BEEP = "record_beep_after_greeting"
KIND_MAILBOX_PHRASE = "mailbox_phrase"
KIND_FAST_ANSWER = "fast_answer"
KIND_GREETING_SHAPE = "greeting_shape"
KIND_NON_HUMAN_AUDIO = "non_human_audio"
KIND_TONE_AFTER_NON_SPEECH = "tone_after_non_speech"
KIND_TONE_AFTER_UNVERIFIED_SPEECH = "tone_after_unverified_speech"
KIND_MAILBOX_PHRASE_AMBIGUOUS = "mailbox_phrase_ambiguous"
KIND_TONE_REJECTED = "tone_rejected"
KIND_PHRASE_ECHO = "phrase_echo"
KIND_UNAVAILABLE_PHRASE = "unavailable_phrase"
KIND_IVR_PHRASE = "ivr_phrase"
KIND_SCREENER_PHRASE = "screener_phrase"
KIND_HELLO_THEN_YIELD = "hello_then_yield"
KIND_RESPONSIVE_REPLY = "responsive_reply"
KIND_SPEECH_AFTER_BEEP = "speech_after_beep"
KIND_CONFIRMED_REPLY = "confirmed_reply"

STRENGTH_PROOF = "proof"
STRENGTH_SUSPICION = "suspicion"
STRENGTH_STATE = "state"
#: Timing shapes a recording can also produce ("hello", pause).  They block a
#: mailbox verdict (the result is ``conflict``), but they are not the veto.
STRENGTH_PROVISIONAL_HUMAN = "provisional_human"
STRENGTH_HUMAN = "human"
STRENGTH_REJECTED = "rejected"

SIGNAL_STRENGTH: dict[str, str] = {
    KIND_RECORD_BEEP: STRENGTH_PROOF,
    KIND_MAILBOX_PHRASE: STRENGTH_PROOF,
    KIND_FAST_ANSWER: STRENGTH_SUSPICION,
    KIND_GREETING_SHAPE: STRENGTH_SUSPICION,
    KIND_NON_HUMAN_AUDIO: STRENGTH_SUSPICION,
    KIND_TONE_AFTER_NON_SPEECH: STRENGTH_SUSPICION,
    KIND_TONE_AFTER_UNVERIFIED_SPEECH: STRENGTH_SUSPICION,
    KIND_MAILBOX_PHRASE_AMBIGUOUS: STRENGTH_SUSPICION,
    KIND_TONE_REJECTED: STRENGTH_REJECTED,
    KIND_PHRASE_ECHO: STRENGTH_REJECTED,
    KIND_UNAVAILABLE_PHRASE: STRENGTH_STATE,
    KIND_IVR_PHRASE: STRENGTH_STATE,
    KIND_SCREENER_PHRASE: STRENGTH_STATE,
    KIND_HELLO_THEN_YIELD: STRENGTH_PROVISIONAL_HUMAN,
    KIND_RESPONSIVE_REPLY: STRENGTH_PROVISIONAL_HUMAN,
    KIND_SPEECH_AFTER_BEEP: STRENGTH_HUMAN,
    KIND_CONFIRMED_REPLY: STRENGTH_HUMAN,
}
STRENGTHS = (
    STRENGTH_PROOF,
    STRENGTH_SUSPICION,
    STRENGTH_STATE,
    STRENGTH_PROVISIONAL_HUMAN,
    STRENGTH_HUMAN,
    STRENGTH_REJECTED,
)
SIGNAL_KINDS = tuple(SIGNAL_STRENGTH)

REJECT_BAND = "band"
REJECT_DURATION = "duration"
REJECT_FAX = "fax"
REJECT_SIT = "sit"
REJECT_CADENCE = "cadence"
REJECT_ECHO = "echo"
REJECT_DTMF = "dtmf"
REJECT_GAP = "gap"
REJECT_STATE = "automation_state"
TONE_REJECT_REASONS = (
    REJECT_BAND,
    REJECT_DURATION,
    REJECT_FAX,
    REJECT_SIT,
    REJECT_DTMF,
    REJECT_CADENCE,
    REJECT_ECHO,
    REJECT_GAP,
    REJECT_STATE,
)


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """One 20 ms caller frame, reduced to the three numbers the rules read."""

    at: float
    rms_dbfs: float
    frequency_hz: float
    purity: float
    #: The frame sits on a DTMF frequency and a partner-group frequency also
    #: carries energy.
    dtmf: bool = False
    #: The media before this frame was not contiguous with it: frames were
    #: lost, or the sidecar stopped delivering.  Nothing measured across a gap
    #: may be read as continuous -- not a tone, not a quiet.
    gap: bool = False
    #: Our own playout may be in this frame (judged on receipt time with the
    #: echo tail, or the frame arrived too late to be placed).  Excluded from
    #: every far-end measurement.
    echo: bool = False


@dataclass(frozen=True, slots=True)
class Tone:
    """A completed run of tonal frames at one frequency."""

    started_at: float
    ended_at: float
    frequency_hz: float
    level_dbfs: float
    dtmf: bool = False

    @property
    def seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


def frame_features(pcm: bytes, *, at: float, gap: bool = False) -> FrameFeatures | None:
    """RMS level, a zero-crossing frequency estimate, and tonal purity.

    The frequency is estimated from the interpolated positions of the first
    and last zero crossings, which is accurate to about one percent over a
    20 ms frame for tones in the beep band -- counting crossings alone would be
    off by up to five percent at 500 Hz.  Purity is one Goertzel evaluated AT
    that frequency, as a share of the frame's energy: it is what separates a
    sine from two tones (DTMF) or speech whose crossings happen to be regular.

    ``None`` for a frame that cannot be read; the caller records nothing for
    it, because a damaged frame is not a quiet one.
    """

    samples = array.array("h")
    try:
        samples.frombytes(pcm[: len(pcm) - (len(pcm) % samples.itemsize)])
    except (TypeError, ValueError):
        return None
    count = len(samples)
    if count < 16:
        return None
    energy = 0.0
    for value in samples:
        energy += float(value) * float(value)
    if energy <= 0.0:
        return FrameFeatures(at=float(at), rms_dbfs=_SILENCE_DBFS, frequency_hz=0.0, purity=0.0, gap=gap)
    rms = math.sqrt(energy / count)
    rms_dbfs = 20.0 * math.log10(rms / 32768.0)
    first = last = None
    crossings = 0
    previous = samples[0]
    for index in range(1, count):
        current = samples[index]
        if (previous < 0) != (current < 0):
            denominator = float(current - previous)
            position = index - 1 + (-previous / denominator if denominator else 0.0)
            if first is None:
                first = position
            last = position
            crossings += 1
        previous = current
    if crossings < 3 or first is None or last is None or last <= first:
        return FrameFeatures(at=float(at), rms_dbfs=rms_dbfs, frequency_hz=0.0, purity=0.0, gap=gap)
    frequency = (crossings - 1) / 2.0 / ((last - first) / SAMPLE_RATE)
    purity = _share(samples, frequency, energy)
    dtmf = False
    for group, partners in ((DTMF_LOW_HZ, DTMF_HIGH_HZ), (DTMF_HIGH_HZ, DTMF_LOW_HZ)):
        if any(abs(frequency - tone) <= DTMF_MATCH_TOLERANCE * tone for tone in group):
            dtmf = any(_share(samples, partner, energy) >= DTMF_PARTNER_SHARE for partner in partners)
            break
    return FrameFeatures(
        at=float(at), rms_dbfs=rms_dbfs, frequency_hz=frequency, purity=purity, dtmf=dtmf, gap=gap
    )


def _share(samples: array.array, frequency: float, energy: float) -> float:
    """The share of a frame's energy at one (non-integer-bin) frequency."""

    coefficient = 2.0 * math.cos(2.0 * math.pi * frequency / SAMPLE_RATE)
    s_prev = s_prev2 = 0.0
    for value in samples:
        s_now = float(value) + coefficient * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s_now
    power = s_prev * s_prev + s_prev2 * s_prev2 - coefficient * s_prev * s_prev2
    return min(1.0, max(0.0, 2.0 * power / (len(samples) * energy)))


def frame_rms_dbfs(pcm: bytes) -> float | None:
    """RMS level of one frame in dBFS, for the agent's own track."""

    samples = array.array("h")
    try:
        samples.frombytes(pcm[: len(pcm) - (len(pcm) % samples.itemsize)])
    except (TypeError, ValueError):
        return None
    if not samples:
        return None
    energy = 0.0
    for value in samples[::2]:
        energy += float(value) * float(value)
    count = len(samples[::2])
    if energy <= 0.0:
        return _SILENCE_DBFS
    return 20.0 * math.log10(math.sqrt(energy / count) / 32768.0)


Interval = tuple[float, float]


def overlaps(start: float, end: float, intervals: tuple[Interval, ...] | list[Interval]) -> bool:
    """Whether ``[start, end)`` touches any of ``intervals``."""

    return any(low < end and start < high for low, high in intervals)


def tone_rejection(
    tone: Tone,
    *,
    previous_tones: tuple[Tone, ...] | list[Tone] = (),
    echo: bool = False,
    spans_gap: bool = False,
) -> str:
    """Why this tone is not a record beep, or ``""`` if it may still be one.

    The order is the order of cost to be wrong about: a tone that is not even
    beep-shaped is rejected first, then a known signalling tone, then one that
    repeats, then one that may be our own echo or that spans lost media.
    ``echo`` and ``spans_gap`` are judged by the caller, which holds the
    receipt-time view of our own playout and of the feed.  Surviving here makes a tone a CANDIDATE; the look-ahead for a
    repeat and the spoken-greeting test decide the rest.
    """

    frequency = tone.frequency_hz
    if not BEEP_BAND_HZ[0] <= frequency <= BEEP_BAND_HZ[1]:
        return REJECT_BAND
    if not BEEP_SECONDS[0] <= tone.seconds <= BEEP_SECONDS[1]:
        return REJECT_DURATION
    if any(low <= frequency <= high for low, high in FAX_BANDS_HZ):
        return REJECT_FAX
    if tone.seconds < SIT_MAX_SECONDS and any(low <= frequency <= high for low, high in SIT_BANDS_HZ):
        return REJECT_SIT
    if tone.dtmf:
        return REJECT_DTMF
    for earlier in previous_tones:
        if (
            earlier.ended_at <= tone.started_at
            and tone.started_at - earlier.ended_at <= CADENCE_LOOKBACK_SECONDS
            and same_frequency(earlier.frequency_hz, frequency)
        ):
            return REJECT_CADENCE
    if echo:
        return REJECT_ECHO
    if spans_gap:
        return REJECT_GAP
    return ""


def same_frequency(left: float, right: float) -> bool:
    return abs(left - right) <= TONE_FREQUENCY_TOLERANCE * max(left, right)


GREETING_SPEECH = "speech"
GREETING_NOT_SPEECH = "not_speech"
GREETING_TOO_SHORT = "too_short"
GREETING_TOO_EARLY = "ended_too_early"
GREETING_UNOBSERVED = "unobserved"


def greeting_shape(
    frames: tuple[FrameFeatures, ...] | list[FrameFeatures],
    *,
    tone_started_at: float,
) -> tuple[str, tuple[float, float] | None]:
    """Whether the far end's audio before a tone is SHAPED like speech.

    Shape only: this is necessary for a record beep to count, never enough.
    Codex (2026-09-23) is right that music with two rests can pass a pause
    count, so the caller also requires recognised words over the span this
    returns.

    Measured on our own frame energy, not on the client VAD: a VAD that never
    reported speaking during an audible announcement is a known failure
    (an announcement recorded as silence).  Frames
    inside our own audible playout (echo tail included) are excluded before
    anything is measured, so an echoed opener cannot supply the greeting; a
    lost-media gap inside the span makes the span unobserved.

    Returns the verdict and the active span ``(start, end)`` it measured.
    """

    window = [
        frame
        for frame in frames
        if tone_started_at - GREETING_LOOKBACK_SECONDS <= frame.at < tone_started_at
        and not frame.echo
    ]
    if not window:
        return GREETING_TOO_SHORT, None
    levels = sorted(frame.rms_dbfs for frame in window)
    floor = levels[max(0, len(levels) // 10)]
    active = [frame for frame in window if frame.rms_dbfs >= floor + GREETING_ACTIVE_ABOVE_FLOOR_DB]
    if len(active) * FRAME_SECONDS < GREETING_MIN_ACTIVE_SECONDS:
        return GREETING_TOO_SHORT, None
    span_start, span_end = active[0].at, active[-1].at + FRAME_SECONDS
    if tone_started_at - span_end > GREETING_MAX_GAP_TO_TONE_SECONDS:
        return GREETING_TOO_EARLY, (span_start, span_end)
    if any(frame.gap for frame in frames if span_start <= frame.at < tone_started_at):
        return GREETING_UNOBSERVED, (span_start, span_end)
    span = [frame for frame in window if span_start <= frame.at < span_end]
    active_levels = sorted(frame.rms_dbfs for frame in active)
    median = active_levels[len(active_levels) // 2]
    dip_frames_needed = max(1, round(DIP_MIN_SECONDS / FRAME_SECONDS))
    dips = 0
    run = 0
    previous_at: float | None = None
    for frame in span:
        # An excluded (echo) stretch breaks a run rather than extending it:
        # a pause is only a pause if we observed the far end being quiet.
        if previous_at is not None and frame.at - previous_at > FRAME_SECONDS * 1.5:
            run = 0
        previous_at = frame.at
        if frame.rms_dbfs <= median - DIP_BELOW_MEDIAN_DB:
            run += 1
            if run == dip_frames_needed:
                dips += 1
        else:
            run = 0
    span_seconds = span_end - span_start
    needed = max(DIPS_PER_FOUR_SECONDS, int(DIPS_PER_FOUR_SECONDS * span_seconds / 4.0))
    return (GREETING_SPEECH if dips >= needed else GREETING_NOT_SPEECH), (span_start, span_end)


class ToneTracker:
    """Frames in, completed tones out, with the frame history they came from.

    Holds a bounded history (enough for the greeting look-back) and at most
    one open tone.  A tone is reported when it ENDS, because a record beep's
    usable moment is its end and its duration is part of what validates it.
    Times are MEDIA times from the recorder's sample cursor, so a burst of
    queued frames still measures the duration it had on the wire.  A gap
    frame closes an open tone and marks it, because a tone measured across
    lost media has an unknown duration.
    """

    def __init__(self, *, history_seconds: float = GREETING_LOOKBACK_SECONDS + 3.0) -> None:
        self._history: deque[FrameFeatures] = deque(
            maxlen=max(1, int(round(history_seconds / FRAME_SECONDS)))
        )
        self._open: list[FrameFeatures] = []
        self._open_spans_gap = False

    @property
    def history(self) -> tuple[FrameFeatures, ...]:
        return tuple(self._history)

    @property
    def open_tone(self) -> tuple[float, float] | None:
        """``(started_at, frequency)`` of a tone still sounding, or ``None``."""

        if not self._open:
            return None
        return self._open[0].at, self._median()

    def feed(self, frame: FrameFeatures) -> tuple[Tone, bool] | None:
        """Take one frame; return ``(tone, spans_gap)`` when a tone ends."""

        completed: tuple[Tone, bool] | None = None
        if frame.gap and self._open:
            self._open_spans_gap = True
            completed = self._close()
        if self._is_tonal(frame):
            if self._open and not same_frequency(self._median(), frame.frequency_hz):
                completed = self._close()
            self._open.append(frame)
        elif self._open:
            completed = self._close()
        self._history.append(frame)
        return completed

    def flush(self) -> tuple[Tone, bool] | None:
        return self._close() if self._open else None

    def _is_tonal(self, frame: FrameFeatures) -> bool:
        return (
            frame.purity >= TONE_PURITY_MIN
            and frame.rms_dbfs >= TONE_LEVEL_MIN_DBFS
            and frame.frequency_hz > 0.0
        )

    def _median(self) -> float:
        return sorted(item.frequency_hz for item in self._open)[len(self._open) // 2]

    def _close(self) -> tuple[Tone, bool]:
        frames = self._open
        spans_gap = self._open_spans_gap
        self._open = []
        self._open_spans_gap = False
        levels = sorted(item.rms_dbfs for item in frames)
        tone = Tone(
            started_at=frames[0].at,
            ended_at=frames[-1].at + FRAME_SECONDS,
            frequency_hz=sorted(item.frequency_hz for item in frames)[len(frames) // 2],
            level_dbfs=levels[len(levels) // 2],
            dtmf=sum(1 for item in frames if item.dtmf) * 2 > len(frames),
        )
        return tone, spans_gap


# -- phrases -----------------------------------------------------------------

#: Mailbox phrases: what a mailbox says just before it records.  Small and
#: curated on purpose -- a phrase list is only as good as its worst entry, and
#: "not available" alone is a receptionist as often as a mailbox (that is the
#: unavailable state below).  Which lists may PROVE a mailbox is the
#: registry's decision (``PHRASE_LANGUAGES`` below), not this table's.
#: Sinhala and Tamil have none yet: no phrase has been verified, and a guessed
#: one would be a false-action generator.  The record says per call whether
#: the call's language has a verified list (``phrase_cues_available``).
_MAILBOX_PHRASES: dict[str, tuple[str, ...]] = {
    "en": (
        "leave a message",
        "leave me a message",
        "leave your message",
        "leave your name and number",
        "after the tone",
        "after the beep",
        "at the tone",
        "record your message",
    ),
    "et": (
        "jätke sõnum",
        "jätke teade",
        "jätta sõnum",
        "pärast piiksu",
        "pärast helisignaali",
    ),
    "de": (
        "hinterlassen sie eine nachricht",
        "nachricht hinterlassen",
        "nach dem signalton",
        "nach dem piepton",
    ),
    "es": (
        "deje su mensaje",
        "deje un mensaje",
        "después de la señal",
        "después del tono",
        "tras la señal",
    ),
    "fr": (
        "laissez un message",
        "laissez moi un message",
        "laissez votre message",
        "après le bip",
        "après le signal sonore",
    ),
    "ru": (
        "оставьте сообщение",
        "оставьте ваше сообщение",
        "после звукового сигнала",
        "после сигнала",
    ),
}

_UNAVAILABLE_PHRASES: dict[str, tuple[str, ...]] = {
    "en": (
        "is not available",
        "switched off",
        "cannot be reached",
        "can not be reached",
        "not reachable",
        "is not in service",
        "no longer in service",
        "the number you have dialled",
        "the number you have dialed",
    ),
}

_IVR_PATTERNS: dict[str, tuple[str, ...]] = {
    "en": (
        r"\bpress (?:\d|one|two|three|four|five|six|seven|eight|nine|zero|star|pound|hash)\b",
        r"\bplease hold\b",
        r"\byour call is important\b",
        r"\ball (?:of )?our (?:agents|representatives|operators)\b",
    ),
    "et": (r"\bvajutage\b",),
    "de": (r"\bdrücken sie\b",),
    "es": (r"\bpulse\b", r"\bmarque\b"),
    "fr": (r"\bappuyez sur\b",),
    "ru": (r"\bнажмите\b",),
}

_SCREENER_PHRASES: dict[str, tuple[str, ...]] = {
    "en": (
        "screening service",
        "call screening",
        "say your name",
        "state your name",
        "why you re calling",
        "what you re calling about",
        "the person you re calling is using",
    ),
}

#: The instruction forms that make a mailbox phrase PROOF.  Each is a
#: regular expression over normalized text (casefolded, punctuation read as a
#: space).  A pair means both must appear in the one utterance: "after the
#: tone" alone is also what an IVR says before a menu.
_MAILBOX_STRONG: dict[str, tuple[str | tuple[str, str], ...]] = {
    "en": (
        r"\bplease leave (?:me |us )?(?:a |your )?(?:message|voicemail)\b",
        r"\bleave (?:a|your) message (?:after|at) the (?:tone|beep)\b",
        r"\brecord your message\b",
        (r"\b(?:after|at) the (?:tone|beep)\b", r"\bmessage\b"),
    ),
    "et": (
        r"\bjätke (?:oma |palun )?(?:sõnum|teade)\b",
        (r"\bpärast (?:piiksu|helisignaali|signaali)\b", r"\b(?:sõnum|teade)\w*"),
    ),
    "de": (
        r"\bhinterlassen sie (?:bitte )?(?:eine |ihre )?nachricht\b",
        (r"\bnach dem (?:signalton|piepton|ton)\b", r"\bnachricht\b"),
    ),
    "es": (
        r"\bdeje (?:su|un) mensaje\b",
        (r"\b(?:después de la señal|después del tono|tras la señal|tras el tono)\b", r"\bmensaje\b"),
    ),
    "fr": (
        r"\blaissez (?:moi |nous )?(?:un|votre) message\b",
        (r"\baprès le (?:bip|signal sonore|signal)\b", r"\bmessage\b"),
    ),
    "ru": (
        r"\bоставьте (?:ваше |своё |свое |пожалуйста )?сообщение\b",
        (r"\bпосле (?:звукового )?сигнала\b", r"\bсообщени\w*"),
    ),
}

#: Words that turn an instruction into its opposite when they come just
#: before it ("please DON'T leave a message").  Normalization splits "don't"
#: into "don t", so both halves are listed.
_NEGATIONS = frozenset(
    {
        "not", "no", "never", "don", "dont", "doesn", "didn", "won", "cannot", "t", "nobody",
        "ära", "ärge", "ei",
        "nicht", "kein", "keine", "keinen",
        "nunca", "ni",
        "ne", "pas", "jamais",
        "не", "нет", "ни",
    }
)
#: A question about a message is a person talking about one.  Openers, as
#: token sequences: a bare "por" or "вы" also begins "por favor, deje su
#: mensaje" and "вы позвонили...", which are mailbox greetings (Grok code
#: review).
_INTERROGATIVE_OPENERS = frozenset(
    {
        ("did",), ("do",), ("does",), ("can",), ("could",), ("would",), ("will",), ("should",),
        ("have",), ("has",), ("is",), ("are",), ("was",), ("why",), ("where",), ("when",),
        ("what",), ("who",), ("how",),
        ("kas",), ("miks",),
        ("haben",), ("hast",), ("kannst",), ("können", "sie"), ("warum",), ("wann",),
        ("por", "qué"), ("por", "que"), ("puede", "usted"), ("puedes",),
        ("est", "ce"), ("pourquoi",), ("avez", "vous"), ("pouvez", "vous"),
        ("почему",), ("вы", "можете"), ("ты", "можешь"),
    }
)
#: Reporting a message is not leaving one: "my phone says please leave a
#: message".  Any of these before the instruction makes it weak.
_REPORTING = frozenset(
    {
        "says", "said", "saying", "say", "told", "tells", "asked", "heard", "hear", "reads",
        "ütles", "ütleb", "öeldi",
        "sagt", "sagte", "gesagt",
        "dice", "dijo", "decía",
        "dit", "disait",
        "говорит", "сказал", "сказала", "сказали", "пишет",
    }
)
_QUOTES = ('"', "“", "”", "„", "«", "»", "'", "‘", "’")
STRONG_PHRASE_MIN_TOKENS = 4
#: Our own words are recognised on the caller track by a speakerphone: a run
#: of this many consecutive tokens shared with something the agent said is
#: echo, not the far end.
ECHO_SHINGLE_TOKENS = 3

PHRASE_MAILBOX = "mailbox"
PHRASE_UNAVAILABLE = "unavailable"
PHRASE_IVR = "ivr"
PHRASE_SCREENER = "screener"


def primary_language(language: str | None) -> str:
    """``en-US`` -> ``en``; anything unreadable -> ``""``."""

    return str(language or "").strip().lower().replace("_", "-").split("-")[0]


#: The phrase-language registry: the ONE place that says which languages'
#: mailbox phrases may prove a mailbox.  ``verified`` lists have been checked
#: against real mailbox greetings or by a native speaker; ``candidate`` lists
#: are written but unchecked, so their instruction forms are recorded as
#: evidence and never count as proof -- a guessed phrase is a false-action
#: generator (lead decision 2026-09-23).  A language with no entry has no
#: phrases at all (Sinhala and Tamil today).  The control plane mirrors this
#: registry (call_voicemail_evidence.VOICEMAIL_PHRASE_LANGUAGES) for the
#: public docs, and a test binds the two and the phrase tables together.
PHRASE_LIST_VERIFIED = "verified"
PHRASE_LIST_CANDIDATE = "candidate"
PHRASE_LANGUAGES: dict[str, str] = {
    "en": PHRASE_LIST_VERIFIED,
    "et": PHRASE_LIST_CANDIDATE,
    "de": PHRASE_LIST_CANDIDATE,
    "es": PHRASE_LIST_CANDIDATE,
    "fr": PHRASE_LIST_CANDIDATE,
    "ru": PHRASE_LIST_CANDIDATE,
}
VERIFIED_PHRASE_LANGUAGES = tuple(
    sorted(language for language, status in PHRASE_LANGUAGES.items() if status == PHRASE_LIST_VERIFIED)
)


def phrase_cues_available(language: str | None) -> bool:
    """Whether this call's own language has a VERIFIED mailbox phrase list.

    A candidate list does not count: its phrases can never prove a mailbox,
    so for the call's evidence the language has no usable list.
    """

    return PHRASE_LANGUAGES.get(primary_language(language)) == PHRASE_LIST_VERIFIED


def mailbox_instruction_is_verified(text: str, *, language: str | None) -> bool:
    """Whether a mailbox INSTRUCTION in ``text`` comes from a verified list.

    ``classify_phrase`` reads the language; this reads the registry.  A strong
    form matched only by a candidate list is still an instruction, and is
    recorded as one, but it is not proof.  English is always checked, so an
    English instruction on an Estonian call is verified.
    """

    normalized = normalize_far_end_text(text)
    verified = tuple(
        lang for lang in _languages(language) if PHRASE_LANGUAGES.get(lang) == PHRASE_LIST_VERIFIED
    )
    if not normalized or not verified:
        return False
    match_at = _strong_mailbox_match(normalized, verified)
    return match_at is not None and _strong_context(str(text), normalized, match_at)


def _languages(language: str | None) -> tuple[str, ...]:
    primary = primary_language(language)
    return (primary, "en") if primary and primary != "en" else ("en",)


def _contains(normalized: str, phrase: str) -> bool:
    return f" {normalize_far_end_text(phrase)} " in f" {normalized} "


def is_ivr(text: str, *, language: str | None) -> bool:
    normalized = normalize_far_end_text(text)
    return bool(normalized) and any(
        re.search(pattern, normalized)
        for lang in _languages(language)
        for pattern in _IVR_PATTERNS.get(lang, ())
    )


def classify_phrase(text: str, *, language: str | None) -> tuple[str, bool]:
    """``(category, strong)`` for one caller utterance; ``("", False)`` if none.

    An IVR marker vetoes a mailbox phrase in the same utterance: "to leave a
    message, press one" is a menu.  A screener outranks a mailbox phrase, and
    a mailbox phrase outranks "not available" ("I'm not available, please
    leave a message" is a mailbox).

    ``strong`` is only ever true for a mailbox phrase, and only for an
    INSTRUCTION form in an utterance long enough to be a greeting, that is not
    a question and not negated just before the instruction.  Everything else
    that mentions a message is a person as often as a machine ("did you leave
    me a message?", "please don't leave a message"), so it is suspicion.
    """

    normalized = normalize_far_end_text(text)
    if not normalized:
        return "", False
    languages = _languages(language)
    if is_ivr(text, language=language):
        return PHRASE_IVR, False
    if any(_contains(normalized, phrase) for lang in languages for phrase in _SCREENER_PHRASES.get(lang, ())):
        return PHRASE_SCREENER, False
    weak = any(_contains(normalized, phrase) for lang in languages for phrase in _MAILBOX_PHRASES.get(lang, ()))
    strong_at = _strong_mailbox_match(normalized, languages)
    if strong_at is not None:
        return PHRASE_MAILBOX, _strong_context(str(text), normalized, strong_at)
    if weak:
        return PHRASE_MAILBOX, False
    if any(_contains(normalized, phrase) for lang in languages for phrase in _UNAVAILABLE_PHRASES.get(lang, ())):
        return PHRASE_UNAVAILABLE, False
    return "", False


def _strong_mailbox_match(normalized: str, languages: tuple[str, ...]) -> tuple[int, ...] | None:
    """Character offsets where the first instruction form's parts start.

    Both parts of a pair are returned, because a negation may sit before
    either one: "don't leave a message after the tone" negates the message,
    not the tone.
    """

    for lang in languages:
        for entry in _MAILBOX_STRONG.get(lang, ()):
            if isinstance(entry, tuple):
                first = re.search(entry[0], normalized)
                second = re.search(entry[1], normalized)
                if first and second:
                    return (first.start(), second.start())
            else:
                match = re.search(entry, normalized)
                if match:
                    return (match.start(),)
    return None


#: Tokens before an instruction verb or part that a negation may sit in,
#: within the same clause.
NEGATION_WINDOW_TOKENS = 3
#: The verbs an instruction is built on, per language: "please DON'T leave"
#: negates the verb, wherever the rest of the instruction sits.
_INSTRUCTION_VERBS = frozenset(
    {"leave", "record", "jätke", "jätta", "hinterlassen", "deje", "dejar", "laissez", "laisser", "оставьте", "оставить"}
)
_CLAUSE_BREAK = re.compile(r"[.,;:!?¡¿…。，、]+")


def _clause_tokens(raw: str) -> list[tuple[str, int]]:
    """Normalized tokens of ``raw``, each with the index of its clause.

    Clauses come from the recogniser's own punctuation, which normalization
    otherwise discards: in "I'm not available, please leave a message" the
    "not" belongs to another clause than the instruction.  A transcript with
    no punctuation is one clause, which errs toward reading a nearby "not" as
    negating the instruction -- the safe direction, since that only makes the
    phrase suspicion.
    """

    tokens: list[tuple[str, int]] = []
    for index, clause in enumerate(_CLAUSE_BREAK.split(str(raw))):
        tokens.extend((token, index) for token in normalize_far_end_text(clause).split())
    return tokens


def _quoted(raw: str) -> bool:
    """Whether the utterance quotes something.  An apostrophe between two
    letters is a contraction ("you've", "can't"), not a quotation mark."""

    for index, mark in enumerate(raw):
        if mark not in _QUOTES:
            continue
        if mark in ("'", "’"):
            before = raw[index - 1] if index > 0 else ""
            after = raw[index + 1] if index + 1 < len(raw) else ""
            if before.isalpha() and after.isalpha():
                continue
        return True
    return False


def _strong_context(raw: str, normalized: str, match_at: tuple[int, ...]) -> bool:
    tokens = normalized.split()
    if len(tokens) < STRONG_PHRASE_MIN_TOKENS:
        return False
    if "?" in raw or "？" in raw or "¿" in raw:
        return False
    if any(tuple(tokens[: len(opener)]) == opener for opener in _INTERROGATIVE_OPENERS):
        return False
    if any(token in _REPORTING for token in normalized[: min(match_at)].split()):
        return False
    if _quoted(raw):
        return False
    clause_tokens = _clause_tokens(raw)
    if [token for token, _ in clause_tokens] != tokens:
        # Punctuation that normalization treats differently from the clause
        # split: without a trustworthy token map, fall back to one clause.
        clause_tokens = [(token, 0) for token in tokens]
    anchors = {len(normalized[:offset].split()) for offset in match_at}
    anchors |= {index for index, (token, _) in enumerate(clause_tokens) if token in _INSTRUCTION_VERBS}
    for anchor in anchors:
        if anchor >= len(clause_tokens):
            continue
        clause = clause_tokens[anchor][1]
        window = clause_tokens[max(0, anchor - NEGATION_WINDOW_TOKENS) : anchor]
        if any(token in _NEGATIONS and token_clause == clause for token, token_clause in window):
            return False
    return True


def _shingles(tokens: list[str]) -> set[tuple[str, ...]]:
    size = ECHO_SHINGLE_TOKENS
    return {tuple(tokens[index : index + size]) for index in range(0, max(0, len(tokens) - size + 1))}


def echo_free_word_count(text: str, agent_texts: tuple[str, ...] | list[str]) -> int:
    """How many of the caller's recognised words were NOT our own echo.

    A token is echo when it sits inside a run of three consecutive tokens that
    also occurs in something the agent said recently.  Shorter overlaps
    ("yes", "thank you") are ordinary conversation, not echo.
    """

    tokens = normalize_far_end_text(text).split()
    if not tokens:
        return 0
    spoken: set[tuple[str, ...]] = set()
    for item in agent_texts:
        spoken |= _shingles(normalize_far_end_text(item).split())
    covered = [False] * len(tokens)
    size = ECHO_SHINGLE_TOKENS
    for index in range(0, max(0, len(tokens) - size + 1)):
        if tuple(tokens[index : index + size]) in spoken:
            for offset in range(size):
                covered[index + offset] = True
    return sum(1 for flag in covered if not flag)


def phrase_is_echo(text: str, agent_texts: tuple[str, ...] | list[str], *, language: str | None) -> bool:
    """The phrase this utterance carries was also said by the agent.

    Judged on the phrase, not the whole utterance: an agent that said "please
    leave a message" and a caller track that returns "... please leave a
    message" is echo even if the recogniser added words around it.
    """

    category, _ = classify_phrase(text, language=language)
    if not category:
        return False
    for spoken in agent_texts:
        spoken_category, _ = classify_phrase(spoken, language=language)
        if spoken_category == category and echo_free_word_count(text, [spoken]) < len(
            normalize_far_end_text(text).split()
        ):
            return True
    return False


__all__ = [
    "DETECTOR_VERSION",
    "FrameFeatures",
    "SIGNAL_KINDS",
    "SIGNAL_STRENGTH",
    "STRENGTHS",
    "TONE_REJECT_REASONS",
    "Tone",
    "ToneTracker",
    "classify_phrase",
    "echo_free_word_count",
    "frame_features",
    "frame_rms_dbfs",
    "greeting_shape",
    "is_ivr",
    "PHRASE_LANGUAGES",
    "PHRASE_LIST_CANDIDATE",
    "PHRASE_LIST_VERIFIED",
    "VERIFIED_PHRASE_LANGUAGES",
    "mailbox_instruction_is_verified",
    "phrase_cues_available",
    "phrase_is_echo",
    "primary_language",
    "tone_rejection",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Stop agent audio by playing the last queued milliseconds out, not dropping them.

The founder's report, twice from live calls, is "the last ~20% of the agent's
final word gets cut off".  This module owns the arithmetic that fixes it.

Where the audio is actually lost
--------------------------------

Pinned runtime ``livekit-agents==1.7.1``.  ``rtc.AudioSource`` is built with
``queue_size_ms=200`` (``voice/room_io/_output.py``), and its
``capture_frame`` "will wait until there is enough space in the queue"
(``rtc/audio_source.py``).  So the forwarder runs at most ~200 ms ahead of the
wire, and that 200 ms is audio the caller is about to hear.

Two places throw that queue away, and BOTH of them do it by calling
``clear_queue()`` on the same ``rtc.AudioSource`` object:

* the pause branch of ``_ParticipantAudioOutput._forward_audio`` -- the
  barge-in path, reached when the bounded speech tail expires;
* the interrupted branch of ``_ParticipantAudioOutput._wait_for_playout`` --
  the ``clear_buffer()`` hard cancel, reached at the end of a reply and when
  the session is torn down under a hangup.

(``pause()`` itself does not clear; its ``clear_queue()`` call is commented
out in this release.  Reading only ``pause()`` suggests the discard is
elsewhere, and reading only ``_forward_audio`` misses the second site.)

Up to 200 ms of *already-committed* speech is dropped mid-frame at whichever
site fires.  At 150-250 ms per syllable that is the tail of the word being
spoken -- exactly the reported symptom, and exactly why a gain ramp applied
where the audio is *captured* cannot help: by the time the stop is requested
those frames left the capture seam seconds ago.

What this module decides
------------------------

Given how much audio is queued, where the playhead sits inside it, and
whichever word boundaries the TTS declared, it answers one question: **how
much of the doomed queue should be played out, and with what gain shape.**

* ``word_aligned`` -- the word currently sounding ends inside the budget, so
  play up to that end at full level and taper only the final few
  milliseconds, which removes the click a hard sample discontinuity makes
  without eating into the next word.
* ``drain_gain`` -- no usable boundary, so play a short prefix of the queue
  under a linear ramp from full level to silence.  The ramp starts at full
  level so there is no step where it begins, and reaches zero so there is
  none where it ends.
* ``hard_cut`` -- recorded by the executor when it could do neither (the
  retained audio no longer covers the playhead, the source is closing, or
  there is no event loop to push on).  That is the pre-existing behaviour, so
  a ``hard_cut`` is a *missed* improvement, never a regression.

Cost, stated exactly, because the deferral it rides on is already bounded.
The drain is taken FROM the queue that was going to be discarded, so it adds
no audio to the reply and cannot reach past what the source is holding -- the
200 ms queue is the ceiling on both rules before either of their own applies.
Worst case from the runtime asking for the floor to the caller hearing
silence:

* ``drain_gain``   -- boundary tail ceiling 0.30 s + ``MAX_DRAIN_SECONDS``
  0.12 s = **0.42 s**;
* ``word_aligned`` -- boundary tail ceiling 0.30 s +
  ``MAX_WORD_ALIGNED_SECONDS`` 0.20 s = **0.50 s**, which is the tail cap plus
  the whole queue and is the most this design can ever cost;
* on the ``clear_buffer`` path no tail is granted, so it is the second term
  alone -- 0.12 s and 0.20 s respectively.

With the shipped 0.24 s tail rather than its ceiling, the typical figures are
0.34 s and up to 0.44 s.

This module imports nothing from ``livekit`` so the main test suite (which
has no livekit dependency) can exercise every decision here.  The audio-source
glue that executes them lives in ``agent/graceful_stop_output.py``.
"""
from __future__ import annotations

import array
from typing import Sequence


# Which rule chose the stop point, recorded per stop so a recording can be
# read against the counters without guessing.
STOP_SOURCE_WORD_ALIGNED = "word_aligned"
STOP_SOURCE_DRAIN_GAIN = "drain_gain"
STOP_SOURCE_HARD_CUT = "hard_cut"

# How much of the doomed queue a ramp-only stop plays out.
#
# Long enough to be heard as a word ending rather than a cut: a linear ramp
# under 50 ms still reads as a click, and the syllable tails this exists to
# rescue run 150-250 ms.  Short enough that it disappears inside the floor
# yield the caller is waiting for -- at 100 ms the total from pause request to
# silence stays at 0.34 s with the shipped 0.24 s tail, well inside the one
# second the interruption budget is documented against.
DEFAULT_DRAIN_SECONDS = 0.10

# The ceiling a ramp-only drain may never exceed, whatever it is configured
# with, enforced here rather than trusted to the caller for the same reason
# the boundary tail enforces its own: an unbounded "graceful" stop is an agent
# that will not yield.
MAX_DRAIN_SECONDS = 0.12

# The ceiling for a WORD-ALIGNED stop, which is allowed to run further because
# it is buying something a ramp cannot: the rest of a real word.  It is the
# whole source queue (``queue_size_ms=200``) and no more, so the worst case
# from the runtime asking for the floor to the caller hearing silence is the
# boundary tail ceiling plus the queue -- 0.30 s + 0.20 s -- and never more,
# whatever the alignment claims.  Nothing beyond the queue exists to play: the
# audio past it was never handed to the source.
MAX_WORD_ALIGNED_SECONDS = 0.20

# A word-aligned stop plays the word to its end at full level, so all it needs
# is enough taper to avoid a click at the splice.  Two zero crossings at
# telephone pitch is about 20 ms; more would audibly soften a boundary that is
# meant to sound like the speaker simply stopped there.
WORD_ALIGNED_RAMP_SECONDS = 0.02

# How much recently pushed audio the ring keeps.  It must cover the whole
# source queue (200 ms) plus the largest drain, with margin for the forwarder
# being momentarily further ahead than the queue estimate says.
DEFAULT_RETAIN_SECONDS = 0.60

# Bounds on the skew between a declared word end and the audio produced so
# far, used to decide whether the two clocks describe the same stream at all.
#
# Alignment can only ever describe audio the provider has already emitted, so
# a word end AHEAD of the audio by more than a rounding error means the two
# are not the same clock and the index must not be trusted.  Lagging is
# normal and bounded: the plugins hold back the trailing partial word, so the
# last complete word can end up to about one word behind the audio.
MAX_ALIGNMENT_LEAD_SECONDS = 0.25
MAX_ALIGNMENT_LAG_SECONDS = 1.50

# After this many syntheses whose alignment did not match their own audio, the
# feed stops offering boundaries for the rest of the call.  One bad synthesis
# is a provider hiccup; a third is a time base this build cannot read.
MAX_DESYNCED_SYNTHESES = 3

_BYTES_PER_SAMPLE = 2


class StopPlan:
    """How much doomed audio to play out, and the gain shape to play it under."""

    __slots__ = ("source", "drain_seconds", "ramp_seconds")

    def __init__(self, *, source: str, drain_seconds: float, ramp_seconds: float) -> None:
        self.source = source
        self.drain_seconds = float(drain_seconds)
        self.ramp_seconds = float(ramp_seconds)

    @property
    def hold_seconds(self) -> float:
        """Seconds played at full level before the taper begins."""

        return max(0.0, self.drain_seconds - self.ramp_seconds)

    @property
    def drained_ms(self) -> int:
        return max(0, int(round(self.drain_seconds * 1_000)))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"StopPlan(source={self.source!r}, drain_seconds={self.drain_seconds:.4f}, "
            f"ramp_seconds={self.ramp_seconds:.4f})"
        )


class PlayoutRing:
    """The most recent audio handed to the audio source, addressed by content offset.

    The unit is the one the runtime itself counts in: seconds of audio pushed
    since this playout run began.  That is what makes a plan expressed as
    "play from the playhead for N seconds" executable -- the playhead is
    ``pushed - queued``, and the ring still holds it because the queue is only
    200 ms deep.

    Frames are kept as raw PCM bytes exactly as they arrived; only the slice
    actually played out is ever converted to samples, so the steady-state cost
    is a bounded ``bytearray`` and no per-frame arithmetic.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        num_channels: int = 1,
        retain_seconds: float = DEFAULT_RETAIN_SECONDS,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if num_channels <= 0:
            raise ValueError("channel count must be positive")
        if not retain_seconds > 0:
            raise ValueError("retention must be positive")
        self._sample_rate = int(sample_rate)
        self._num_channels = int(num_channels)
        self._bytes_per_second = self._sample_rate * self._num_channels * _BYTES_PER_SAMPLE
        self._retain_bytes = max(
            self._frame_bytes(), int(retain_seconds * self._bytes_per_second)
        )
        self._buffer = bytearray()
        # Content offset, in bytes, of the first byte still held.
        self._base_bytes = 0

    def _frame_bytes(self) -> int:
        return self._num_channels * _BYTES_PER_SAMPLE

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def end_seconds(self) -> float:
        """Content offset just past the newest byte held."""

        return (self._base_bytes + len(self._buffer)) / self._bytes_per_second

    @property
    def start_seconds(self) -> float:
        """Content offset of the oldest byte still held."""

        return self._base_bytes / self._bytes_per_second

    def push(self, data: bytes) -> None:
        """Record audio just handed to the source, evicting what aged out."""

        if not data:
            return
        self._buffer += data
        excess = len(self._buffer) - self._retain_bytes
        if excess > 0:
            del self._buffer[:excess]
            self._base_bytes += excess

    def reset(self) -> None:
        """A new playout run starts its content offsets at zero."""

        self._buffer.clear()
        self._base_bytes = 0

    def take(self, start_seconds: float, duration_seconds: float) -> bytes | None:
        """Audio at ``[start, start+duration)``, or ``None`` when not fully held.

        Fully held or nothing: a partial slice would splice a shortened
        fragment onto the wire, which is the artefact this exists to remove.
        """

        if duration_seconds <= 0:
            return b""
        start = self._aligned(start_seconds * self._bytes_per_second)
        stop = self._aligned((start_seconds + duration_seconds) * self._bytes_per_second)
        if start < self._base_bytes or stop > self._base_bytes + len(self._buffer):
            return None
        return bytes(self._buffer[start - self._base_bytes : stop - self._base_bytes])

    def _aligned(self, offset: float) -> int:
        """Round a byte offset onto a whole sample frame."""

        frame = self._frame_bytes()
        return max(0, int(round(offset / frame)) * frame)


def apply_stop_gain(
    data: bytes,
    *,
    num_channels: int,
    hold_samples: int,
    ramp_samples: int,
) -> bytes:
    """Full level for ``hold_samples``, then linear to silence over ``ramp_samples``.

    The ramp reaches exactly zero on its last sample frame, so the splice to
    silence has no step in it.  Anything past the ramp is silence: a caller
    that passes more audio than the ramp covers wants the remainder muted, not
    left at full level.
    """

    block = array.array("h")
    block.frombytes(data)
    channels = max(1, int(num_channels))
    span = max(1, int(ramp_samples))
    hold = max(0, int(hold_samples))
    for index in range(len(block)):
        position = index // channels
        if position < hold:
            continue
        step = position - hold
        if step >= span:
            block[index] = 0
            continue
        block[index] = int(block[index] * (1.0 - (step + 1) / span))
    return block.tobytes()


class WordBoundaryIndex:
    """Word ends declared by one synthesis, in that synthesis's own audio clock.

    ElevenLabs (``sync_alignment``) and Cartesia (``word_timestamps``) declare
    ``TTS.capabilities.aligned_transcript`` and attach per-word timings to the
    synthesized frames.  Those timings are measured from the start of the
    synthesis, and the audio the runtime pushes for a playout run is exactly
    that synthesis from its first frame, so the two clocks are the same one --
    a claim this index CHECKS rather than assumes.

    Every recorded word carries the cumulative audio duration produced at the
    moment it was declared.  A word that ends AHEAD of that audio cannot be
    describing it; a word that lags by more than a clause is not usable for a
    decision about the syllable in progress.  Either way the synthesis is
    marked untrusted and the caller falls back to the ramp, which is always
    correct and merely less precise.
    """

    def __init__(
        self,
        *,
        max_words: int = 512,
        lead_tolerance: float = MAX_ALIGNMENT_LEAD_SECONDS,
        lag_tolerance: float = MAX_ALIGNMENT_LAG_SECONDS,
    ) -> None:
        self._max_words = max(1, int(max_words))
        self._lead_tolerance = float(lead_tolerance)
        self._lag_tolerance = float(lag_tolerance)
        self._ends: list[float] = []
        self._trusted = True
        self._words_seen = 0
        self._queued = False

    @property
    def trusted(self) -> bool:
        return self._trusted

    @property
    def queued(self) -> bool:
        """Whether this synthesis has ever been offered to a playout run.

        Asked ONCE per index, never "is it in the queue right now".  The
        observer announces its synthesis on every frame, and a run can adopt
        the index after the first of them -- so a membership test would see
        the index gone and offer the same synthesis a second time, handing a
        later run the timings of an earlier one.  That is the very shift the
        queue exists to prevent.
        """

        return self._queued

    def mark_queued(self) -> None:
        self._queued = True

    @property
    def words_seen(self) -> int:
        return self._words_seen

    @property
    def usable(self) -> bool:
        """Whether a boundary from this index may be acted on."""

        return bool(self._ends) and self._trusted

    def note_word(self, *, end_seconds: float, audio_seconds: float) -> None:
        """Record one declared word end against the audio produced so far."""

        try:
            end = float(end_seconds)
            produced = float(audio_seconds)
        except (TypeError, ValueError):
            self._trusted = False
            return
        if end != end or produced != produced:  # NaN compares unequal to itself
            self._trusted = False
            return
        skew = end - produced
        if skew > self._lead_tolerance or -skew > self._lag_tolerance:
            self._trusted = False
            return
        self._words_seen += 1
        if self._ends and end <= self._ends[-1]:
            # A restated span. Overwriting the last entry keeps the list sorted
            # only while the new value still sits above the one before it; a
            # provider that revises further back than that would leave the list
            # unsorted, and the scan below -- which stops at the first end past
            # the playhead -- would then hand back a boundary LATER than the
            # corrected one and drain past the word it was meant to finish.
            # Alignment that contradicts itself is not alignment, so the
            # synthesis is distrusted and this run falls back to the ramp.
            if len(self._ends) > 1 and end <= self._ends[-2]:
                self._trusted = False
                return
            self._ends[-1] = end
            return
        self._ends.append(end)
        if len(self._ends) > self._max_words:
            del self._ends[0]

    def word_end_within(self, *, playhead: float, window: float) -> float | None:
        """End of the word now sounding, when it falls inside ``window``.

        The FIRST end after the playhead, not the last: the word being spoken
        is the one the playhead sits inside, and finishing it is the whole
        claim.  Running on to a later boundary would spend the budget saying
        more than the caller interrupted.
        """

        if not self.usable or window <= 0:
            return None
        for end in self._ends:
            if end <= playhead:
                continue
            return end if end - playhead <= window else None
        return None


class AlignmentFeed:
    """Hand each synthesis's word index to the playout run that plays it.

    The TTS runs seconds ahead of the wire, so an index cannot be reset when
    playback starts -- the words for the run starting now were declared long
    before it.  Nor can one index be shared: each synthesis restarts its own
    clock at zero.  So indices are queued as they are opened and adopted, in
    order, by the runs that play them.

    An index is handed over EMPTY and fills as its own audio streams: at the
    moment a run starts, the provider has usually declared no words for it
    yet.  Usability is therefore a question for the stop, not for the handover
    -- ``WordBoundaryIndex.word_end_within`` answers it -- and an index that
    never receives a word simply never offers a boundary.

    A run with no index left to adopt has no alignment and falls back to the
    ramp.  That is the honest outcome for every provider that declares no
    aligned transcript at all -- Gemini TTS, and therefore the whole Sinhala
    stack -- and for a synthesis whose alignment never arrived.

    Opening an index and QUEUEING it are separate steps, and the separation is
    load-bearing.  A synthesis cancelled before it yields a frame -- an
    ordinary interruption while the provider is still starting up -- produces
    no playout run at all.  Had it been queued when it opened, the next run
    would adopt it and every later run would stay shifted onto the synthesis
    before its own, reading word ends from unrelated audio.  So an index
    becomes adoptable only once its synthesis has actually produced audio,
    which is exactly when a run for it is going to exist.
    """

    def __init__(
        self,
        *,
        max_pending: int = 4,
        max_tracked: int = 8,
        max_desynced_syntheses: int = MAX_DESYNCED_SYNTHESES,
    ) -> None:
        self._max_pending = max(1, int(max_pending))
        self._max_tracked = max(1, int(max_tracked))
        self._max_desynced = max(1, int(max_desynced_syntheses))
        # Opened but not yet adopted by a playout run.
        self._pending: list[WordBoundaryIndex] = []
        # Still counted individually; older ones are folded into the scalars
        # below so one long call cannot grow this without bound.
        self._tracked: list[WordBoundaryIndex] = []
        self._folded_words = 0
        self._folded_desynced = 0
        self._syntheses = 0

    @property
    def abandoned(self) -> bool:
        """Whether alignment has been given up on for the rest of the call."""

        return self._desynced() >= self._max_desynced

    def open_index(self) -> WordBoundaryIndex | None:
        """Start an index for one synthesis; ``None`` once alignment is abandoned.

        The index is not adoptable yet -- see ``publish``.
        """

        if self.abandoned:
            self._pending.clear()
            return None
        index = WordBoundaryIndex()
        self._tracked.append(index)
        self._syntheses += 1
        while len(self._tracked) > self._max_tracked:
            self._fold(self._tracked.pop(0))
        return index

    def publish(self, index: WordBoundaryIndex) -> None:
        """This synthesis has produced audio, so a run will play it.

        Idempotent, because the caller announces it on every frame rather than
        having to remember which one was first.
        """

        if self.abandoned or index.queued:
            return
        index.mark_queued()
        self._pending.append(index)
        while len(self._pending) > self._max_pending:
            # A published synthesis whose audio still never reached a run
            # cannot be allowed to shift the queue for the rest of the call.
            self._pending.pop(0)

    def adopt(self) -> WordBoundaryIndex | None:
        """Take the index belonging to the playout run that is starting.

        Handed over whatever it currently holds: the words for this run arrive
        while it plays, and refusing an index that is merely still empty would
        mean no run ever got one.

        Exactly one candidate or none at all, and the queue is emptied either
        way.  A synthesis can produce audio that never reaches a playout run --
        interrupted between the TTS node and the source -- and with a plain
        FIFO that orphan would be handed to the NEXT run, leaving every run
        after it one synthesis behind for the rest of the call, stopping on
        word ends measured from unrelated audio.  So more than one pending
        candidate is read as what it is, an ambiguity, and answered by giving
        this run no alignment at all: it falls back to the ramp, which is
        always correct and merely less precise, and the following run starts
        from a clean queue.  Wrong timings are worse than no timings.
        """

        pending, self._pending = self._pending, []
        if self.abandoned or len(pending) != 1:
            return None
        return pending[0]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "stop_boundary_aligned_syntheses": self._syntheses,
            "stop_boundary_aligned_words": self._folded_words
            + sum(index.words_seen for index in self._tracked),
            "stop_boundary_alignment_desynced": self._desynced(),
            "stop_boundary_alignment_abandoned": int(self.abandoned),
        }

    def _desynced(self) -> int:
        return self._folded_desynced + sum(
            1 for index in self._tracked if not index.trusted
        )

    def _fold(self, index: WordBoundaryIndex) -> None:
        """Reduce one index to the scalars it contributes, and let it go."""

        self._folded_words += index.words_seen
        if not index.trusted:
            self._folded_desynced += 1


def plan_stop(
    *,
    queued_seconds: float,
    playhead_seconds: float,
    index: WordBoundaryIndex | None = None,
    drain_seconds: float = DEFAULT_DRAIN_SECONDS,
    max_drain_seconds: float = MAX_DRAIN_SECONDS,
    max_word_aligned_seconds: float = MAX_WORD_ALIGNED_SECONDS,
) -> StopPlan | None:
    """Choose the stop point inside the queue that is about to be discarded.

    The two rules get different budgets on purpose.  A ramp is cosmetic, so it
    is held to ``max_drain_seconds``; finishing a word is worth more, so it may
    use the whole queue, which is where the audio runs out anyway.  Neither can
    reach past what the source is holding, so the queue is the real ceiling on
    both.

    ``None`` when the queue holds nothing: a stop with an empty queue cuts no
    audio, so it is not a cut and must not be counted as one.
    """

    if queued_seconds <= 0:
        return None
    queued = float(queued_seconds)
    if index is not None:
        window = min(queued, max(0.0, float(max_word_aligned_seconds)))
        word_end = index.word_end_within(playhead=playhead_seconds, window=window)
        if word_end is not None:
            drain = max(0.0, word_end - playhead_seconds)
            if drain > 0:
                return StopPlan(
                    source=STOP_SOURCE_WORD_ALIGNED,
                    drain_seconds=drain,
                    ramp_seconds=min(WORD_ALIGNED_RAMP_SECONDS, drain),
                )
    drain = min(queued, max(0.0, float(max_drain_seconds)), max(0.0, float(drain_seconds)))
    if drain <= 0:
        return None
    return StopPlan(source=STOP_SOURCE_DRAIN_GAIN, drain_seconds=drain, ramp_seconds=drain)


class GracefulStopStats:
    """Bounded per-call counters describing what every stop actually did."""

    def __init__(self) -> None:
        self._by_source: dict[str, int] = {
            STOP_SOURCE_WORD_ALIGNED: 0,
            STOP_SOURCE_DRAIN_GAIN: 0,
            STOP_SOURCE_HARD_CUT: 0,
        }
        self._drained_ms_total = 0
        self._drained_ms_max = 0

    def note_stop(self, *, source: str, drained_ms: int = 0) -> None:
        if source not in self._by_source:
            return
        self._by_source[source] += 1
        drained = max(0, int(drained_ms))
        self._drained_ms_total = min(3_600_000, self._drained_ms_total + drained)
        self._drained_ms_max = max(self._drained_ms_max, drained)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "stop_boundary_stops_word_aligned": self._by_source[STOP_SOURCE_WORD_ALIGNED],
            "stop_boundary_stops_drain_gain": self._by_source[STOP_SOURCE_DRAIN_GAIN],
            "stop_boundary_stops_hard_cut": self._by_source[STOP_SOURCE_HARD_CUT],
            "stop_boundary_drained_ms_total": self._drained_ms_total,
            "stop_boundary_drained_ms_max": self._drained_ms_max,
        }


def samples_from(data: bytes) -> Sequence[int]:
    """PCM16 bytes as signed samples, for assertions and the measurement probe."""

    block = array.array("h")
    block.frombytes(data)
    return block


__all__ = [
    "DEFAULT_DRAIN_SECONDS",
    "DEFAULT_RETAIN_SECONDS",
    "MAX_ALIGNMENT_LAG_SECONDS",
    "MAX_ALIGNMENT_LEAD_SECONDS",
    "MAX_DESYNCED_SYNTHESES",
    "MAX_DRAIN_SECONDS",
    "MAX_WORD_ALIGNED_SECONDS",
    "STOP_SOURCE_DRAIN_GAIN",
    "STOP_SOURCE_HARD_CUT",
    "STOP_SOURCE_WORD_ALIGNED",
    "WORD_ALIGNED_RAMP_SECONDS",
    "AlignmentFeed",
    "GracefulStopStats",
    "PlayoutRing",
    "StopPlan",
    "WordBoundaryIndex",
    "apply_stop_gain",
    "plan_stop",
    "samples_from",
]

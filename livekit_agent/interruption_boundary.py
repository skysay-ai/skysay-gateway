# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Let a qualified interruption finish the current syllable, not cut through it.

The pinned LiveKit runtime (``livekit-agents==1.7.1``) yields the floor by
*pausing the audio output*: ``AgentActivity._interrupt_by_audio_activity``
calls ``audio_output.pause()``, and ``_ParticipantAudioOutput.pause`` clears
``_playback_enabled``, so ``_forward_audio`` stops mid-frame -- 10 ms into
whatever syllable was being spoken.  Nothing in the runtime rounds that up to
a boundary.

Measured on ``call_MPXHrjeG3WhhMAYQ`` (2026-08-28 03:58Z): three of three
agent utterances were cut with ``playout_ms`` between 301 and 412 ms, and
seven of those pauses resumed ~1 s later because the caller had only made a
backchannel noise.  The founder's description of the recording is the
requirement: "stops speaking instantly, does not finish the word, not
natural".

So the pause is *deferred*, never refused.  A qualified interruption still
wins; it simply takes effect after the current word rather than inside it.

What this is NOT
----------------

**This is a bounded tail, not word-boundary detection, and it does not claim
to align the stop with a word.**  Read that before trusting the name.

The pinned runtime exposes no live playback position.
``_ParticipantAudioOutput`` reports ``playback_position`` only in
``on_playback_finished``; 1.7.1 adds ``on_playback_progressed`` (#6470), but
``_report_run`` fires it retrospectively once per playout run -- on
interruption or drain -- so it is post-hoc evidence, not the continuous clock
a real-time boundary decision would need.  Frames handed to ``capture_frame``
sit in the audio source's queue ahead of the wire by an amount nothing reports,
so energy analysis at the capture layer describes audio the caller has not
heard yet.
Aligning a stop with a real word would mean rebuilding that playback clock from
public events -- which is exactly the failure mode that cost the
caller-transcript gate six review rounds of oscillation, and it is out of scope
here by decision rather than by oversight.

So be precise about what the tail buys and what it does not:

* it does buy the rest of the syllable in progress, which is the difference
  between a clipped consonant and a finished sound, and it shortens the silent
  gap of a false interruption by the same amount;
* it does NOT guarantee the stop lands between words.  An interruption arriving
  early in a long word still stops inside it, one syllable later; one arriving
  near a word's end can carry into the next word and stop inside that.

The ceiling is what keeps the second case harmless: at 300 ms the tail cannot
run past roughly one word, so the stop point moves by less than a word in
either direction and the agent still yields the floor promptly.  A stop that
genuinely lands on a boundary needs a timing source this stack does not have --
aligned TTS transcripts (the pinned chunked-HTTP transport provides none) or
the runtime's own transcript-synchronizer clock.

Why the tail is short
---------------------

``DEFAULT_BOUNDARY_TAIL_SECONDS`` has to be long enough to carry a syllable
to its end and short enough that the agent still audibly yields:

* a spoken syllable runs roughly 150-250 ms, and the cuts in the recording
  landed 301-412 ms into an utterance -- inside the first word;
* the runtime's ``false_interruption_timeout`` is 1.0 s in this stack, so a
  tail well under half of it keeps the pause/resume pair legible: a false
  interruption still resumes into audio that never fully stopped;
* the shortest interruption floor this stack configures is 0.15 s and the
  longest 0.75 s (``demo.py`` and the managed-English policy in
  ``production.py``), so a 0.24 s tail keeps total time from
  caller speech onset to agent silence under one second in every configured
  combination.

``MAX_BOUNDARY_TAIL_SECONDS`` is the ceiling a caller would notice as the
agent refusing to yield.  It is enforced here rather than trusted to the
caller, because the whole point of the deferral is that it stays bounded.

This module owns the DECISION only and imports nothing from ``livekit``, so
the main test suite (which has no livekit dependency) can exercise it.  The
``AudioOutput`` glue that acts on these decisions lives in
``agent/playout_hold_output.py`` beside the bounded playout hold it shares a
wrapper with.
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple


# How long audio may keep playing after a qualified interruption asked for the
# floor.  See the module docstring for why it is this short.
DEFAULT_BOUNDARY_TAIL_SECONDS = 0.24

# The ceiling the deferral may never exceed, whatever it is configured with.
MAX_BOUNDARY_TAIL_SECONDS = 0.30

# What the glue must do with one ``pause()`` call.
PAUSE_NOW = "pause_now"
PAUSE_DEFER = "defer"
PAUSE_ALREADY_DEFERRED = "already_deferred"

# Why a deferral ended, recorded so a repeat is attributable from the worker
# log alone.
TAIL_ENDED_BUDGET = "budget"
TAIL_ENDED_RESUMED = "resumed"
TAIL_ENDED_CLEARED = "cleared"
TAIL_ENDED_FINISHED = "finished"
TAIL_ENDED_DETACHED = "detached"


class PauseDecision(NamedTuple):
    """What the audio output should do about one ``pause()`` request."""

    action: str
    delay_seconds: float


class BoundaryStopPolicy:
    """Decide whether a pause may be deferred by a bounded speech tail.

    One instance serves a whole call.  Every method is synchronous and
    loop-affine by contract: the pinned runtime calls ``pause``/``resume``/
    ``clear_buffer`` and reports playback from the single asyncio loop running
    the media session, so there are no locks here.
    """

    def __init__(
        self,
        *,
        tail_seconds: float = DEFAULT_BOUNDARY_TAIL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not tail_seconds >= 0:
            raise ValueError("boundary tail must not be negative")
        self._tail_seconds = min(float(tail_seconds), MAX_BOUNDARY_TAIL_SECONDS)
        self._clock = clock
        # Whether audio is currently reaching the wire.  Only then is there a
        # word to finish; before the first frame a pause withholds silence,
        # which is the bounded playout hold's business and must not be delayed.
        self._playing = False
        self._deferring_since: float | None = None
        self._tails_granted = 0
        self._tails_ended: dict[str, int] = {
            TAIL_ENDED_BUDGET: 0,
            TAIL_ENDED_RESUMED: 0,
            TAIL_ENDED_CLEARED: 0,
            TAIL_ENDED_FINISHED: 0,
            TAIL_ENDED_DETACHED: 0,
        }
        self._max_tail_ms = 0
        self._total_tail_ms = 0

    # -- state a caller reads ---------------------------------------------

    @property
    def tail_seconds(self) -> float:
        return self._tail_seconds

    @property
    def deferring(self) -> bool:
        return self._deferring_since is not None

    @property
    def stats(self) -> dict[str, int]:
        """Bounded per-call counters for the interruption summary."""

        return {
            "stop_boundary_tails_granted": self._tails_granted,
            "stop_boundary_tail_ms_max": self._max_tail_ms,
            "stop_boundary_tail_ms_total": self._total_tail_ms,
            "stop_boundary_tails_ended_at_budget": self._tails_ended[TAIL_ENDED_BUDGET],
            "stop_boundary_tails_ended_by_resume": self._tails_ended[TAIL_ENDED_RESUMED],
            "stop_boundary_tails_ended_by_clear": self._tails_ended[TAIL_ENDED_CLEARED],
            "stop_boundary_tails_ended_at_playout_end": self._tails_ended[TAIL_ENDED_FINISHED],
            "stop_boundary_tails_ended_at_detach": self._tails_ended[TAIL_ENDED_DETACHED],
        }

    # -- events the audio output reports ----------------------------------

    def note_playback_started(self) -> None:
        """Audio reached the wire, so there is now a word that can be cut."""

        self._playing = True

    def note_playback_finished(self) -> int:
        """The turn's playout ended; a deferral in flight has nothing left to serve."""

        self._playing = False
        return self._end_tail(TAIL_ENDED_FINISHED)

    def request_pause(self) -> PauseDecision:
        """Answer one ``pause()`` from the runtime.

        ``PAUSE_DEFER`` grants the tail: audio keeps playing for that long and
        the pause takes effect afterwards.  It buys the syllable in progress;
        it does not locate a word boundary -- the module docstring says
        precisely what that would take and why it is not done here.

        ``PAUSE_NOW`` when nothing audible is being cut -- the pre-first-frame
        pause the bounded playout hold measures, and every pause once a deferral has
        already been served for this stretch of audio.

        ``PAUSE_ALREADY_DEFERRED`` when a deferral is running.  The runtime
        re-pauses on every far-end burst, and a tail that restarted on each
        one would be unbounded: the second request must neither extend the
        first nor cancel it.
        """

        if self._deferring_since is not None:
            return PauseDecision(PAUSE_ALREADY_DEFERRED, 0.0)
        if not self._playing or self._tail_seconds <= 0:
            return PauseDecision(PAUSE_NOW, 0.0)
        self._deferring_since = float(self._clock())
        self._tails_granted += 1
        return PauseDecision(PAUSE_DEFER, self._tail_seconds)

    def tail_expired(self) -> int:
        """The tail budget ran out; the pause engages now.

        Returns the milliseconds of audio the tail actually bought, or ``-1``
        when no deferral was running -- a stale timer that must not pause a
        turn nobody asked to pause.

        Audio stops here, so the next pause is not deferrable.  The runtime
        re-pauses on EVERY far-end burst -- ``_interrupt_by_audio_activity``
        is gated on the speech handle, not on whether the output is already
        paused -- and granting a second tail against silence would spend a
        budget that buys no audio and report a boundary stop that never
        happened.  ``_ParticipantAudioOutput.resume`` clears its first-frame
        event, so the next audible frame reports ``playback_started`` again
        and the following pause is deferrable once more.
        """

        if self._deferring_since is None:
            return -1
        self._playing = False
        return self._end_tail(TAIL_ENDED_BUDGET)

    def note_resume(self) -> int:
        """The runtime resumed; a deferral in flight is cancelled, not served."""

        return self._end_tail(TAIL_ENDED_RESUMED)

    def note_buffer_cleared(self) -> int:
        """A real barge-in discarded the audio; there is no word left to finish.

        ``clear_buffer`` is the runtime's hard cancel and carries the
        ``wait_for_playout``/``synchronized_transcript`` bookkeeping behind
        it, so it is never deferred -- only the pause is.

        Clearing marks the output not-playing, which makes the next pause
        immediate rather than deferred.  That is correct and self-healing: the
        runtime's clear sets its interrupted event, which ends the segment and
        reports ``playback_finished``; the following segment reports
        ``playback_started`` from its own first frame, and this policy is
        deferrable again from there.
        """

        self._playing = False
        return self._end_tail(TAIL_ENDED_CLEARED)

    def note_detached(self) -> int:
        """The sink left the chain; the tail ends with its own reason.

        Counted separately from a resume: a detached output is teardown, and
        folding it into the resume counter would make the per-call totals
        claim a false interruption that never happened.
        """

        self._playing = False
        return self._end_tail(TAIL_ENDED_DETACHED)

    # -- internals ---------------------------------------------------------

    def _end_tail(self, reason: str) -> int:
        if self._deferring_since is None:
            return -1
        elapsed_ms = max(0, int(round((float(self._clock()) - self._deferring_since) * 1_000)))
        self._deferring_since = None
        self._tails_ended[reason] = self._tails_ended.get(reason, 0) + 1
        self._max_tail_ms = max(self._max_tail_ms, elapsed_ms)
        self._total_tail_ms = min(3_600_000, self._total_tail_ms + elapsed_ms)
        return elapsed_ms


__all__ = [
    "DEFAULT_BOUNDARY_TAIL_SECONDS",
    "MAX_BOUNDARY_TAIL_SECONDS",
    "PAUSE_ALREADY_DEFERRED",
    "PAUSE_DEFER",
    "PAUSE_NOW",
    "TAIL_ENDED_BUDGET",
    "TAIL_ENDED_CLEARED",
    "TAIL_ENDED_DETACHED",
    "TAIL_ENDED_FINISHED",
    "TAIL_ENDED_RESUMED",
    "BoundaryStopPolicy",
    "PauseDecision",
]

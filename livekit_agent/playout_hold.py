# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bound how long ready agent audio may be withheld from the wire.

The pinned LiveKit runtime (``livekit-agents==1.8.1``)
yields the floor to a talking far end by *pausing the audio output* rather
than by cancelling the reply:

* ``AgentActivity._interrupt_by_audio_activity`` pauses and arms a
  ``false_interruption_timeout`` resume timer;
* ``AgentActivity.on_start_of_speech`` pauses -- with no duration gate at all
  -- whenever the far end starts speaking while the agent is not yet speaking;
* every later ``_interrupt_by_audio_activity`` **cancels** that pending resume
  timer and re-arms it.

While paused, ``_ParticipantAudioOutput._forward_audio`` blocks on
``await self._playback_enabled.wait()``, so synthesized audio sits in the
worker and never becomes RTP.  Nothing anywhere bounds that wait.  Against a
far end that keeps the floor -- an IVR announcement, a voicemail greeting, an
autoresponder -- the agent can hold a finished utterance indefinitely.

Measured on a real Sri Lanka outbound call: the reply's audio existed 29.5 s
before its first frame reached the track, and it was released 1.5 s after the
announcement finally stopped.  The agent's own recorded track held digital
silence for that whole window -- the audio was HELD, not repeatedly cancelled,
which is what distinguishes this from the turn-taking defect that chops agent
speech into audible fragments.

Scope is deliberately the *opening* of a turn: the budget runs from the first
synthesized frame until the first frame reaches the wire, which is exactly the
interval the ``playout.first_worker`` span reports.  A pause arriving after the
agent is audibly speaking is ordinary barge-in handling and is left alone.

This module owns the DECISION only, and imports nothing from ``livekit`` so
the main test suite (which has no livekit dependency) can exercise it.  The
LiveKit ``AudioOutput`` glue that acts on these decisions lives in
``agent/playout_hold_output.py``.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable


# How long one agent turn's ready audio may be withheld before the guard plays
# it anyway.
#
# This is a product decision, not a tuning constant: it is how long the agent
# will politely wait for a far end that will not stop talking.  For a
# payment-reminder campaign dialling businesses, an agent that never speaks is
# strictly worse than one that speaks over the tail of an announcement -- the
# message at least gets delivered, and gets recorded by voicemail.
#
# Eight seconds is deliberately generous against the measured benign range:
# across every production ``playout.first_worker`` span recorded between
# 2026-08-09 and 2026-08-27 the largest non-pathological value was 789 ms and
# the median was 4 ms, so a human-answered call never approaches the budget.
# A far end still holding the floor eight seconds after the agent was already
# given its turn is monologuing, and the break-in stays bounded: the release
# covers one turn, and the next turn starts polite again.
DEFAULT_MAX_HOLD_SECONDS = 8.0

# How long a hold with NO caller words behind it may last, per
# transcriber stack: ``(stt registry_name, model, language)``.
#
# The eight seconds above is sized for a far end that really is talking. On a
# human call the same pause is taken for any sound the VAD scores as speech,
# and background noise scores 0.95-0.99 -- the same as a voice. On
# call_GzZQWzYr8cjVitkR the noise held a finished reply until the caller said
# "Hello?" into the silence, and that real word cancelled it: twice, 14.6 s of
# dead air. What separates the two is words: the transcriber produced none
# during the noise.
#
# A budget shorter than eight seconds is only safe where the transcriber is
# MEASURED to deliver a caller's first word well inside it; otherwise a caller
# who really is talking would be spoken over. So a stack is listed here only
# with its measurement, and every stack not listed -- every other provider,
# model and language, and every voice-to-voice engine -- keeps the full budget.
#
# ``deepgram``/``nova-3``/``en``: measured 2026-08-29 (docs/VOICE_QUALIFICATION.md,
# "Measured 2026-08-29 for Deepgram nova-3 on English"). Results arrive on a
# fixed grid aligned to the stream, 16 gaps p50 1004 ms / p95 1013 ms / max
# 1013 ms, and a first word landed 339-942 ms after speech onset across a
# 200-1300 ms phase sweep; the incident call's "Hello?"s were transcribed
# within the same 0.3-1.0 s. The hold's clock starts no earlier than the
# caller's onset, so a word is in by 1.013 s at the latest; 1.5 s is that
# worst case plus half a second of margin. Estonian shares the adapter and
# model but was not itself timed, so it is not listed.
#
# Deliberately not a customer setting: it is a property of the transcriber's
# timing, like ``stt_interims_usable`` on the capability artifact.
NO_WORD_HOLD_BUDGETS: Mapping[tuple[str, str, str], float] = MappingProxyType(
    {("deepgram", "nova-3", "en"): 1.5}
)


def no_word_hold_budget_seconds(
    *,
    stt_registry_name: Any,
    stt_model: Any,
    language: Any,
    word_requirement: int,
    voice_to_voice: bool,
) -> float | None:
    """The no-word hold budget for this call, or ``None`` for the full budget.

    ``None`` unless the call enforces a word requirement (the word gate is what
    the released reply is handed to) on a cascade stack whose transcriber
    timing is in :data:`NO_WORD_HOLD_BUDGETS`. A voice-to-voice engine has no
    transcriber in the loop and is excluded outright, whatever its config
    carries.
    """
    if voice_to_voice or int(word_requirement or 0) <= 0:
        return None
    key = (
        str(stt_registry_name or "").strip().lower(),
        str(stt_model or "").strip(),
        str(language or "").strip().lower(),
    )
    return NO_WORD_HOLD_BUDGETS.get(key)

# A deadline timer may fire a hair before the budget it was armed for; a
# refusal inside this slack would leave a hold nobody re-arms.
_DEADLINE_SLACK_SECONDS = 0.001

# Values recorded on the turn's ``playout.first_worker`` span so a repeat is
# attributable from the telemetry table alone, without a packet capture.
HOLD_REASON_NONE = ""
HOLD_REASON_FAR_END_YIELDED = "far_end_floor_hold"
HOLD_REASON_BUDGET_EXPIRED = "far_end_floor_hold_released"
HOLD_REASON_RELEASED_NO_WORDS = "far_end_floor_hold_released_no_words"

# A hold shorter than this is ordinary scheduling jitter, not a floor yield,
# and must not label an otherwise healthy turn.
_REPORTABLE_HOLD_SECONDS = 0.25

# Which label survives when one reply plays as several segments: the worst
# thing that happened to the reply is what its span should say.
_REASON_SEVERITY = {
    HOLD_REASON_NONE: 0,
    HOLD_REASON_FAR_END_YIELDED: 1,
    HOLD_REASON_RELEASED_NO_WORDS: 2,
    HOLD_REASON_BUDGET_EXPIRED: 3,
}

# Per-call counters are reported in a log summary; keep them bounded.
_COUNTER_LIMIT = 1_000_000
_HOLD_MS_LIMIT = 3_600_000


class PlayoutHoldPolicy:
    """Decide when a pause has withheld a turn's opening audio for too long.

    One instance serves a whole call.  A *segment* runs from the first audio
    frame captured for an agent turn until that turn's playback finishes.  The
    budget is cumulative across every pause between that first frame and the
    moment audio reaches the wire, so a far end that pauses, releases, and
    re-pauses cannot reset it.

    The window closes for good at ``note_playback_started``.  Two reasons: the
    ``playout.first_worker`` span this labels ends there, so accounting past it
    would attribute later pauses to a metric that already stopped; and a pause
    after the agent is audibly speaking is real barge-in handling, which this
    guard has no business overriding.

    ``word_evidence``, when given, is a monotone count of the caller's words
    (``CallerTranscriptGate.caller_words_observed``). It is read once when a
    segment's first hold begins, and while no new word has arrived since, the
    budget is ``no_word_max_hold_seconds`` rather than ``max_hold_seconds``.
    One new word restores the full budget for the rest of the segment. Without
    it the policy behaves exactly as it did before words were counted, which is
    what every call that cannot count words in time gets.
    """

    def __init__(
        self,
        *,
        max_hold_seconds: float = DEFAULT_MAX_HOLD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        word_evidence: Callable[[], int] | None = None,
        no_word_max_hold_seconds: float | None = None,
    ) -> None:
        if not max_hold_seconds > 0:
            raise ValueError("playout hold budget must be positive")
        if no_word_max_hold_seconds is not None and not no_word_max_hold_seconds > 0:
            raise ValueError("no-word hold budget must be positive")
        self._max_hold_seconds = float(max_hold_seconds)
        # Both or neither: a word count with no measured budget, or a budget
        # with nothing to count, is the full budget.
        if word_evidence is None or no_word_max_hold_seconds is None:
            word_evidence = None
            no_word_max_hold_seconds = self._max_hold_seconds
        self._no_word_max_hold_seconds = min(
            float(no_word_max_hold_seconds), self._max_hold_seconds
        )
        self._word_evidence = word_evidence
        self._clock = clock
        # The word count when this segment's first hold began; ``None`` until
        # then. Held for the whole segment, so pause/resume bursts share it.
        self._words_at_hold_start: int | None = None
        self._words_seen_during_hold = False
        self._no_word_budget_forgone = False
        # The word count when the opening window closed, so words the caller
        # says over the audible reply are not reported as said during the hold.
        self._words_at_window_close: int | None = None
        self._released_no_words = False
        self._holds = 0
        self._hold_ms_max = 0
        self._holds_cancelled = 0
        self._holds_released_no_words = 0
        self._last_hold: tuple[int, int, bool] | None = None
        # Whether the real sink is currently paused.
        self._sink_paused = False
        self._audio_pending = False
        self._reached_wire = False
        self._paused_since: float | None = None
        self._held_seconds = 0.0
        self._released = False
        # Accumulated across every playback segment of one reply; see
        # ``take_turn_report``.
        self._pending_hold_seconds = 0.0
        self._pending_reason = HOLD_REASON_NONE

    # -- state a caller reads --------------------------------------------

    @property
    def max_hold_seconds(self) -> float:
        return self._max_hold_seconds

    @property
    def released(self) -> bool:
        """Whether this turn's opening audio was released by the budget."""
        return self._released

    @property
    def released_no_words(self) -> bool:
        """Whether that release was the no-word budget rather than the full one."""
        return self._released_no_words

    @property
    def words_during_hold(self) -> int:
        """Caller words that arrived between this segment's first hold and its wire.

        Frozen when the opening window closes -- the release, the first frame,
        or the discard -- so a word said over the audible reply is not counted.
        """
        if self._words_at_hold_start is None:
            return 0
        words = (
            self._words_at_window_close
            if self._words_at_window_close is not None
            else self._read_words()
        )
        if words is None:
            return 0
        return min(4_096, max(0, words - self._words_at_hold_start))

    def take_last_hold(self) -> tuple[int, int, bool] | None:
        """``(held_ms, words_during_hold, reached_wire)`` of the segment just ended, once.

        Set when a segment that held long enough to be counted ends -- played,
        interrupted, or discarded before its first frame -- so the caller can
        attribute the hold to the reply it belonged to. A hold cancelled before
        playout has no span to carry it, and is otherwise invisible.
        """
        last, self._last_hold = self._last_hold, None
        return last

    @property
    def stats(self) -> dict[str, int]:
        """Bounded per-call counters for the interruption summary.

        A hold counts once per playback segment and only when it lasted long
        enough to be named on a span, so scheduling jitter does not inflate
        it. ``cancelled`` is a held segment whose audio never reached the wire:
        exactly the replies that leave no ``playout.first_worker`` span.
        """
        return {
            "pre_playout_holds": self._holds,
            "pre_playout_hold_ms_max": self._hold_ms_max,
            "pre_playout_holds_cancelled": self._holds_cancelled,
            "pre_playout_holds_released_no_words": self._holds_released_no_words,
        }

    @property
    def held_seconds(self) -> float:
        """Hold accrued before the wire, including any open pause."""
        held = self._held_seconds
        if self._paused_since is not None:
            held += max(0.0, self._clock() - self._paused_since)
        return held

    def remaining_seconds(self) -> float | None:
        """Seconds until the budget expires, or ``None`` when nothing is held."""
        if self._released or self._reached_wire or self._paused_since is None:
            return None
        return max(0.0, self._budget_seconds() - self.held_seconds)

    def take_turn_report(self) -> tuple[str, int]:
        """Reason code and held milliseconds for the reply just finished, once.

        One assistant reply can play as SEVERAL playback segments -- the
        runtime starts a new one at every ``FlushSentinel`` in the LLM stream
        -- and each segment reports its own ``playback_finished``, while the
        conversation item that carries the turn's telemetry arrives only after
        the last of them.  So the report accumulates the worst thing that
        happened across the reply's segments and is cleared when the turn's
        telemetry reads it; otherwise a hold on the opening segment would be
        overwritten by the next segment finishing cleanly, and the span would
        lose exactly the attribution this exists to provide.
        """
        report = (self._pending_reason, max(0, int(round(self._pending_hold_seconds * 1_000))))
        self._pending_reason = HOLD_REASON_NONE
        self._pending_hold_seconds = 0.0
        return report

    # -- events the audio output reports ---------------------------------

    def note_audio_ready(self) -> None:
        """The turn's first synthesized frame has been handed to the output.

        This is the same instant the runtime records as
        ``started_forwarding_at``, so the budget measures exactly the interval
        the ``playout.first_worker`` span reports.
        """
        if self._audio_pending:
            return
        self._audio_pending = True
        if self._sink_paused and not self._reached_wire and self._paused_since is None:
            # Paused before any audio existed: the hold starts now, because
            # only now is there something being withheld.
            self._start_hold()

    def note_pause(self) -> None:
        """The runtime paused the output.

        Recorded, never refused.  Refusing a pause would leave the runtime
        believing the agent had fallen silent -- it records ``_paused_speech``
        and moves the session to ``listening`` immediately after calling
        ``pause()`` -- while audio kept playing, and the evidence recorder
        closes the agent's interval on exactly that state change.  The release
        instead stands the runtime down for the turn, so its state and the
        audio stay in agreement.
        """
        self._sink_paused = True
        if self._audio_pending and not self._reached_wire and self._paused_since is None:
            self._start_hold()

    def note_resume(self) -> None:
        self._sink_paused = False
        self._accrue()

    def note_playback_started(self) -> None:
        """Audio reached the wire; the opening window closes for good."""
        self._accrue()
        self._close_window()
        self._reached_wire = True

    def note_playback_finished(self) -> None:
        """The turn's playout ended (completed, interrupted, or discarded)."""
        self._accrue()
        self._remember_turn()
        self._count_hold()
        self._reset_segment()

    def note_buffer_cleared(self) -> None:
        """Pending audio was discarded, so nothing is being withheld.

        An interruption clears the buffer BEFORE reporting playback finished,
        so a reply that was held, reached the wire, and was then barged into
        would otherwise arrive at ``note_playback_finished`` with its hold
        already erased -- and lose the attribution on the very span that
        records it.  Audio that reached the wire keeps its report; audio that
        never played owes none, because no span will describe it.
        """
        self._accrue()
        self._close_window()
        if self._reached_wire:
            self._remember_turn()
        self._count_hold()
        self._reset_segment()

    def expire(self) -> bool:
        """Spend the budget on this turn's opening audio.

        Returns ``True`` when this call is what released the hold, so the
        caller resumes the real output and stands the runtime down exactly
        once. ``False`` also while budget remains: a no-word deadline that
        finds a word has arrived must not release, and the caller re-arms from
        :meth:`remaining_seconds` instead.
        """
        if (
            self._released
            or self._reached_wire
            or not self._audio_pending
            or not self._sink_paused
        ):
            return False
        remaining = self.remaining_seconds()
        if remaining is not None and remaining > _DEADLINE_SLACK_SECONDS:
            return False
        no_words = self._no_word_budget_applies()
        self._accrue()
        self._close_window()
        self._sink_paused = False
        self._released = True
        self._released_no_words = no_words
        if no_words:
            self._holds_released_no_words = min(
                _COUNTER_LIMIT, self._holds_released_no_words + 1
            )
        return True

    # -- internals --------------------------------------------------------

    def _read_words(self) -> int | None:
        if self._word_evidence is None:
            return None
        try:
            return max(0, int(self._word_evidence()))
        except Exception:
            return None

    def _start_hold(self) -> None:
        self._paused_since = self._clock()
        if self._words_at_hold_start is None and self._word_evidence is not None:
            words = self._read_words()
            if words is None:
                # A source that cannot answer counts as "a word arrived": the
                # full budget is what this call had before words were counted,
                # never a shorter hold on a broken reading.
                self._words_seen_during_hold = True
                words = 0
            self._words_at_hold_start = words

    def no_word_release_due(self) -> bool:
        """Whether a deadline firing now would release under the no-word budget.

        Asked BEFORE :meth:`expire`, so the caller can hand the reply to the
        word gate first -- or, when it cannot, give up the short budget with
        :meth:`forgo_no_word_budget` rather than release a reply nobody could
        then interrupt with a word.
        """
        if (
            self._released
            or self._reached_wire
            or not self._audio_pending
            or not self._sink_paused
            or not self._no_word_budget_applies()
        ):
            return False
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= _DEADLINE_SLACK_SECONDS

    def forgo_no_word_budget(self) -> None:
        """This segment keeps the full budget: the short one cannot be served safely."""
        self._no_word_budget_forgone = True

    def _close_window(self) -> None:
        if self._words_at_window_close is None and self._words_at_hold_start is not None:
            self._words_at_window_close = self._read_words()

    def _no_word_budget_applies(self) -> bool:
        if (
            self._word_evidence is None
            or self._words_at_hold_start is None
            or self._no_word_budget_forgone
        ):
            return False
        if not self._words_seen_during_hold:
            words = self._read_words()
            if words is None or words > self._words_at_hold_start:
                # Latched: a word proves the hold is a person talking, for the
                # rest of this segment.
                self._words_seen_during_hold = True
        return not self._words_seen_during_hold

    def _budget_seconds(self) -> float:
        if self._no_word_budget_applies():
            return self._no_word_max_hold_seconds
        return self._max_hold_seconds

    def _count_hold(self) -> None:
        if self._held_seconds < _REPORTABLE_HOLD_SECONDS:
            return
        self._holds = min(_COUNTER_LIMIT, self._holds + 1)
        held_ms = min(_HOLD_MS_LIMIT, int(round(self._held_seconds * 1_000)))
        self._hold_ms_max = max(self._hold_ms_max, held_ms)
        if not self._reached_wire:
            self._holds_cancelled = min(_COUNTER_LIMIT, self._holds_cancelled + 1)
        self._last_hold = (held_ms, self.words_during_hold, self._reached_wire)

    def _accrue(self) -> None:
        if self._paused_since is not None:
            self._held_seconds += max(0.0, self._clock() - self._paused_since)
            self._paused_since = None

    def _remember_turn(self) -> None:
        if self._released and self._released_no_words:
            reason = HOLD_REASON_RELEASED_NO_WORDS
        elif self._released:
            reason = HOLD_REASON_BUDGET_EXPIRED
        elif self._held_seconds >= _REPORTABLE_HOLD_SECONDS:
            reason = HOLD_REASON_FAR_END_YIELDED
        else:
            reason = HOLD_REASON_NONE
        if _REASON_SEVERITY[reason] >= _REASON_SEVERITY[self._pending_reason]:
            self._pending_reason = reason
        self._pending_hold_seconds = max(self._pending_hold_seconds, self._held_seconds)

    def _reset_segment(self) -> None:
        self._audio_pending = False
        self._reached_wire = False
        self._paused_since = None
        self._held_seconds = 0.0
        self._released = False
        self._released_no_words = False
        self._words_at_hold_start = None
        self._words_seen_during_hold = False
        self._no_word_budget_forgone = False
        self._words_at_window_close = None


__all__ = [
    "DEFAULT_MAX_HOLD_SECONDS",
    "HOLD_REASON_BUDGET_EXPIRED",
    "HOLD_REASON_FAR_END_YIELDED",
    "HOLD_REASON_NONE",
    "HOLD_REASON_RELEASED_NO_WORDS",
    "NO_WORD_HOLD_BUDGETS",
    "PlayoutHoldPolicy",
    "no_word_hold_budget_seconds",
]

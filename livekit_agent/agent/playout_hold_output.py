# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""LiveKit ``AudioOutput`` glue for the bounded playout hold.

Every DECISION belongs to ``playout_hold.PlayoutHoldPolicy``, which imports no
livekit and is unit-tested directly.  This module is only the wiring: it sits
at the top of the session's audio chain, reports what the runtime does to the
policy, and acts on the policy's answers.

Placement is deliberate.  ``AgentActivity`` pauses ``session.output.audio``, so
the guard must be *that* object to see a pause at all.  Wrapping after
``session.start()`` is the runtime's own pattern -- ``AgentSession`` wraps the
same attribute with ``RecorderIO.record_output`` a few lines after
``RoomIO.start()`` -- and it keeps the transcript synchronizer attached,
because ``AgentOutput.audio``'s setter detaches the old sink and then attaches
the new one, whose ``on_attached`` forwards straight back down the chain.
``can_pause`` also stays ``True`` through the wrapper, so the runtime's
``_pause_enabled()`` decision is unchanged and nothing outside the hold budget
moves.

Releasing has two halves, and the second one is what keeps the runtime honest.
Resuming the sink alone would be undone by the far end's next syllable, and
merely *refusing* the next pause would be worse than that: ``AgentActivity``
records ``_paused_speech`` and moves the session to ``listening`` right after
calling ``pause()``, and the evidence recorder closes the agent's speaking
interval on that state change -- so audio would keep playing while the call's
own evidence said the agent had stopped.  Instead the release marks the turn's
speech handle uninterruptible.  All three pause sites in the pinned runtime
(``agent_activity.py`` 2213, 2259 and 4658 at 1.8.1 -- the third,
``_reconcile_playout_pause``, gates on the same flag via its
``pause_is_allowed``) are gated on
``current_speech.allow_interruptions``, so the runtime simply stops asking:
no pause, no ``_paused_speech``, no state change, and the audio and the
evidence stay in agreement.  That flag is put back when the released segment
finishes -- a tool-using reply re-enters the reply task with the same handle,
and the stand-down must not follow it there.

The same wrapper carries the boundary stop, because it is the same
``pause()`` call that both policies have an opinion about and splitting them
across two sinks would make their ordering unreviewable.  They act on opposite
halves of a turn and cannot both be running: the hold measures a pause taken
BEFORE the first frame reaches the wire, the boundary stop defers a pause
taken AFTER it.  ``BoundaryStopPolicy`` never defers a pause while nothing is
audible, so ``PlayoutHoldPolicy`` sees the pre-playout pause exactly as it did
before this change; and once a deferral is running the hold's own budget has
already closed for the turn.

Only ``pause()`` is deferred.  ``clear_buffer()`` is the runtime's hard cancel
and carries ``wait_for_playout``/``synchronized_transcript`` bookkeeping behind
it, so it still takes effect immediately.  The bounded cost of the deferral is
that up to ``MAX_BOUNDARY_TAIL_SECONDS`` of audio plays after the runtime has
already moved the session to ``listening`` and the evidence recorder has closed
the agent's speaking interval -- the same class of drift this module documents
above for the hold release, with a ceiling three orders of magnitude smaller.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from livekit import rtc
from livekit.agents.voice import io

# Flat ``/app`` import, exactly like production_agent.py's own dependencies:
# this repository has no livekit dependency, so nothing importing livekit is
# importable from the test suite at all. That is precisely why every decision
# lives in playout_hold.py instead of here.
from interruption_boundary import (
    PAUSE_ALREADY_DEFERRED,
    PAUSE_DEFER,
    BoundaryStopPolicy,
)
from playout_hold import (
    HOLD_REASON_BUDGET_EXPIRED,
    HOLD_REASON_RELEASED_NO_WORDS,
    PlayoutHoldPolicy,
)


class BoundedPlayoutHoldOutput(io.AudioOutput):
    """Pass audio through, but never let a pause withhold it without bound."""

    def __init__(
        self,
        *,
        next_in_chain: io.AudioOutput,
        policy: PlayoutHoldPolicy | None = None,
        boundary: BoundaryStopPolicy | None = None,
        stand_down: Callable[[], Callable[[], None]] | None = None,
        hand_over: Callable[[], bool] | None = None,
        on_release: Callable[[int, str], None] | None = None,
        on_boundary_tail_granted: Callable[[int], None] | None = None,
        on_hold_ended: Callable[[int, int, bool, str], None] | None = None,
        current_speech_id: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(
            label="SkysayPlayoutHold",
            capabilities=io.AudioOutputCapabilities(pause=True),
            next_in_chain=next_in_chain,
        )
        self._policy = policy if policy is not None else PlayoutHoldPolicy()
        # An omitted boundary means NO deferral, not the managed default.
        # Only a managed phone call opts into a tail; every other caller of
        # this wrapper must keep the runtime's own instant pause, and a
        # default that silently enabled a 240 ms tail would give a browser
        # test behaviour nobody asked it to have.
        self._boundary = (
            boundary if boundary is not None else BoundaryStopPolicy(tail_seconds=0.0)
        )
        self._stand_down = stand_down
        self._hand_over = hand_over
        self._on_hold_ended = on_hold_ended
        self._current_speech_id = current_speech_id
        # The reply a hold belongs to, named when the hold begins.
        self._hold_owner: str | None = None
        self._on_release = on_release
        self._on_boundary_tail_granted = on_boundary_tail_granted
        self._deadline: asyncio.TimerHandle | None = None
        self._boundary_stop: asyncio.TimerHandle | None = None
        # Undoes the stand-down when the released segment finishes; see
        # ``stand_down_current_speech``.
        self._restore_interruptions: Callable[[], None] | None = None

    @property
    def policy(self) -> PlayoutHoldPolicy:
        return self._policy

    @property
    def boundary(self) -> BoundaryStopPolicy:
        return self._boundary

    @property
    def sample_rate(self) -> int | None:
        if self._sample_rate is not None:
            return self._sample_rate
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    # -- audio path -------------------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        # Told before the frame moves on: this is the runtime's own
        # ``started_forwarding_at`` instant, and a pause already in force
        # starts withholding audio from exactly here.
        self._policy.note_audio_ready()
        self._arm_deadline()
        await super().capture_frame(frame)
        assert self.next_in_chain is not None
        await self.next_in_chain.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        # A real interruption discards the pending audio, so there is nothing
        # left to withhold and the segment's budget starts over.  It also ends
        # any boundary tail: the word being finished has just been thrown away,
        # so there is nothing left to finish.
        self._cancel_boundary_stop()
        self._boundary.note_buffer_cleared()
        self._policy.note_buffer_cleared()
        self._report_hold()
        self._cancel_deadline()
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()

    # -- pause / resume ---------------------------------------------------

    def pause(self) -> None:
        """Yield the floor after the current syllable rather than inside it.

        A pause taken while audio is audible is deferred by the bounded tail;
        every other pause (and every re-pause during a deferral that is
        already running) behaves exactly as it did before the boundary stop.
        The tail is a time budget, NOT a detected word boundary --
        ``interruption_boundary`` documents what it does and does not buy.
        """

        decision = self._boundary.request_pause()
        if decision.action == PAUSE_ALREADY_DEFERRED:
            # The far end bursts repeatedly and the runtime re-pauses on each
            # one.  The tail must neither restart nor be cut short.
            return
        if decision.action == PAUSE_DEFER and self._schedule_boundary_stop(
            decision.delay_seconds
        ):
            if self._on_boundary_tail_granted is not None:
                self._on_boundary_tail_granted(int(round(decision.delay_seconds * 1_000)))
            return
        if decision.action == PAUSE_DEFER:
            # No running loop to defer on (never true in the media session).
            # Give the grant back so the counters describe what happened.
            self._boundary.note_resume()
        self._engage_pause()

    def _engage_pause(self) -> None:
        self._policy.note_pause()
        self._arm_deadline()
        if self.next_in_chain:
            self.next_in_chain.pause()

    def resume(self) -> None:
        # Ordered before the hold policy on purpose: a resume arriving inside
        # the tail means the sink was never actually paused, and the deferral
        # must be cancelled rather than served after the runtime has already
        # decided the floor is the agent's again.
        self._cancel_boundary_stop()
        self._boundary.note_resume()
        self._policy.note_resume()
        self._cancel_deadline()
        if self.next_in_chain:
            self.next_in_chain.resume()

    # -- playback reports (forwarded up the chain by AudioOutput) ---------

    def on_playback_started(self, *, created_at: float) -> None:
        self._policy.note_playback_started()
        # From here there is an audible word a pause would cut in half.
        self._boundary.note_playback_started()
        self._cancel_deadline()
        super().on_playback_started(created_at=created_at)

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        self._cancel_boundary_stop()
        self._boundary.note_playback_finished()
        self._policy.note_playback_finished()
        self._report_hold()
        self._cancel_deadline()
        self._restore_interruption_policy()
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )

    def on_detached(self) -> None:
        self._cancel_boundary_stop()
        self._boundary.note_detached()
        self._cancel_deadline()
        self._restore_interruption_policy()
        super().on_detached()

    def _report_hold(self) -> None:
        """Hand a finished segment's hold to telemetry; never fails the audio path."""
        owner, self._hold_owner = self._hold_owner, None
        last = self._policy.take_last_hold()
        if last is None or self._on_hold_ended is None:
            return
        try:
            self._on_hold_ended(*last, owner or "")
        except Exception:  # pragma: no cover - observability only
            pass

    def _name_hold_owner(self) -> None:
        if self._hold_owner is not None or self._current_speech_id is None:
            return
        if self._policy.remaining_seconds() is None:
            return
        try:
            self._hold_owner = str(self._current_speech_id() or "")
        except Exception:  # pragma: no cover - observability only
            self._hold_owner = ""

    def _restore_interruption_policy(self) -> None:
        restore, self._restore_interruptions = self._restore_interruptions, None
        if restore is not None:
            restore()

    # -- the bound --------------------------------------------------------

    def _arm_deadline(self) -> None:
        self._name_hold_owner()
        remaining = self._policy.remaining_seconds()
        if remaining is None:
            self._cancel_deadline()
            return
        if self._deadline is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no media loop, nothing to bound
            return
        self._deadline = loop.call_later(remaining, self._release_hold)

    def _cancel_deadline(self) -> None:
        if self._deadline is not None:
            self._deadline.cancel()
            self._deadline = None

    # -- the boundary stop -------------------------------------------------

    def _schedule_boundary_stop(self, delay_seconds: float) -> bool:
        """Arm the tail; ``False`` when there is no loop to arm it on."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no media loop, pause at once
            return False
        self._boundary_stop = loop.call_later(delay_seconds, self._stop_at_boundary)
        return True

    def _cancel_boundary_stop(self) -> None:
        if self._boundary_stop is not None:
            self._boundary_stop.cancel()
            self._boundary_stop = None

    def _stop_at_boundary(self) -> None:
        """The tail is spent: take the pause the runtime asked for."""

        self._boundary_stop = None
        if self._boundary.tail_expired() < 0:
            # The deferral was cancelled between this timer firing and the
            # loop reaching it.  Pausing now would stop a turn nobody asked
            # to stop.
            return
        self._engage_pause()

    def _release_hold(self) -> None:
        self._deadline = None
        held_ms = int(round(self._policy.held_seconds * 1_000))
        # A hold with no words behind it was noise, and the reply it held must
        # still yield to a caller who then really speaks: it goes to the word
        # gate, which refuses the runtime's wordless pauses but gives the floor
        # back on one recognised word. Handed over BEFORE the release, so the
        # short budget is only ever spent on a reply the gate has taken; when
        # the gate declines it -- a handle it already gave back, one another
        # writer holds -- the segment keeps the full budget instead of being
        # released unbarge-in-able at 1.5 s.
        handed_over = False
        if self._policy.no_word_release_due():
            if self._hand_over is not None:
                try:
                    handed_over = bool(self._hand_over())
                except Exception:  # pragma: no cover - keep the full budget
                    handed_over = False
            if not handed_over:
                self._policy.forgo_no_word_budget()
                self._arm_deadline()
                return
        if not self._policy.expire():
            # Either nothing is held any more, or the budget grew under this
            # timer: a no-word deadline that finds the caller has now said a
            # word owes the rest of the full budget, and only a new deadline
            # can serve it. ``_arm_deadline`` disarms when nothing is held.
            self._arm_deadline()
            return
        # Order matters: stand the runtime down BEFORE the audio starts, so a
        # far-end syllable arriving in between cannot re-pause a turn that is
        # already committed to speaking.
        if not handed_over and self._stand_down is not None:
            self._restore_interruptions = self._stand_down()
        if self.next_in_chain:
            self.next_in_chain.resume()
        if self._on_release is not None:
            self._on_release(
                held_ms,
                HOLD_REASON_RELEASED_NO_WORDS
                if self._policy.released_no_words
                else HOLD_REASON_BUDGET_EXPIRED,
            )


def stand_down_current_speech(session: Any) -> Callable[[], None]:
    """Tell the runtime to stop trying to yield the floor on this turn.

    Two things, and both are needed.

    **Forward:** ``SpeechHandle.allow_interruptions`` is a public setter and
    both of the pinned runtime's pause sites are gated on it, so clearing it
    stops the far end re-pausing a turn already committed to speaking.
    ``current_speech`` is the right handle by construction: scheduling a new
    speech cancels the old one, and that path clears the audio buffer, which
    resets this guard's segment and cancels its deadline, so a release cannot
    fire against a turn that is no longer current.  The flag lives on the
    handle, so it expires with the turn and the next one starts interruptible.

    **Backward:** the pause that has just been released is still recorded in
    ``AgentActivity._paused_speech``, holding the agent state captured BEFORE
    playout -- ``thinking``.  Left there, the runtime's zero-timeout
    false-interruption callback restores that state once the far end goes
    quiet, i.e. while the released audio is audible.  Two things then break:
    the evidence recorder closes the agent's interval on any
    ``speaking -> not speaking`` transition, so the rest of the utterance
    lands outside every recorded agent interval; and the end of the reply only
    transitions to ``listening`` when the state is still ``speaking``, so the
    session stays wedged in ``thinking`` for the remainder of the turn.
    Dropping the record makes that callback a no-op, which is the reconciliation
    the release needs; nothing else consumes it once this turn can no longer be
    paused.

    That second step reads one private attribute of the pinned
    ``livekit-agents==1.8.1`` runtime (``agent_activity.py`` :240), so it is
    deliberately best-effort: if a
    future runtime moves it, the audio is still released and only the state
    artefact returns.

    Returns a callable that puts ``allow_interruptions`` back the way it was.
    The stand-down must NOT outlive the segment it released: a reply that
    invokes a tool re-enters ``_pipeline_reply_task`` with the SAME handle for
    its tool response, so a flag left cleared would silently make that later
    reply unbarge-in-able and skip its user-silence wait. Restoring costs the
    ceiling nothing, because each playback segment gets its own opening budget.
    """
    speech = getattr(session, "current_speech", None)
    restore: Callable[[], None] = lambda: None
    if speech is not None and not getattr(speech, "interrupted", False):
        previous = getattr(speech, "allow_interruptions", True)
        try:
            speech.allow_interruptions = False
        except Exception:  # pragma: no cover - runtime refused the flag
            pass
        else:

            def restore() -> None:
                try:
                    speech.allow_interruptions = previous
                except Exception:  # pragma: no cover - handle already gone
                    pass

    _forget_released_pause(session)
    return restore


def _forget_released_pause(session: Any) -> None:
    """Drop the pause record a release has just overruled.

    The backward half of both release paths; see
    :func:`stand_down_current_speech` for what the stale record does if left.
    """
    activity = getattr(session, "_activity", None)
    if activity is not None:
        try:
            if getattr(activity, "_paused_speech", None) is not None:
                activity._paused_speech = None
        except Exception:  # pragma: no cover - runtime moved its bookkeeping
            pass


def hand_current_speech_to_word_gate(
    session: Any,
    arm: Callable[[Any], bool],
    reply_in_flight_states: frozenset[str],
) -> bool:
    """Release a wordless hold into the word gate; ``False`` if the gate declines.

    ``arm`` is ``InterruptionWordGate.arm``. It clears the handle's
    ``allow_interruptions`` exactly as the stand-down does -- so the far end's
    next wordless burst cannot re-pause the released audio at any of the three
    pause sites of ``livekit-agents==1.8.1`` (``agent_activity.py`` :2213 in
    ``_interrupt_by_audio_activity``, :2259 in ``on_start_of_speech``, :4658 in
    ``_reconcile_playout_pause``, all gated on that flag) -- but the gate OWNS
    the flag afterwards: the transcript gate gives the floor back when the
    caller has said a word for the duration floor, and the agent-state handler
    gives it back when the reply ends. The ``speaking`` edge's own ``arm`` of
    the same handle is then a no-op.

    The gate declines a handle it has already released, one already
    interrupted, and one another writer holds uninterruptible; the caller then
    keeps the full budget.

    The pause the runtime recorded is RESOLVED the way the runtime resolves a
    false interruption (``_on_false_interruption``, ``agent_activity.py``
    :4697-4700): the agent state it captured before the pause -- ``thinking``
    -- is restored first, then the record is dropped and the audio resumes.
    Leaving the session in ``listening`` instead would open a window between
    the release and the first frame in which the transcript gate is not
    holding, so a caller's final would reach a runtime that refuses to answer
    a turn while the current speech is uninterruptible (``agent_activity.py``
    :2606-2613) and the words would be dropped. Restored here, before the
    audio moves, the gate holds them as it does for any reply in flight.

    So the hand-over is refused unless the state being restored is one the
    transcript gate holds in (``reply_in_flight_states``). A ``session.say``
    line -- the repair phrase -- never reports ``thinking``: its pause
    captured ``listening``, restoring that would leave the gate open, and the
    reply keeps the full budget instead.
    """
    speech = getattr(session, "current_speech", None)
    if speech is None:
        return False
    activity = getattr(session, "_activity", None)
    paused = getattr(activity, "_paused_speech", None) if activity is not None else None
    if (
        paused is None
        or getattr(paused, "handle", None) is not speech
        or str(getattr(paused, "agent_state", "")) not in reply_in_flight_states
    ):
        return False
    try:
        armed = bool(arm(speech))
    except Exception:  # pragma: no cover - the gate never raises by contract
        armed = False
    if armed:
        _restore_paused_state(session, speech)
        _forget_released_pause(session)
    return armed


def _restore_paused_state(session: Any, speech: Any) -> None:
    """Put back the agent state the runtime's pause of ``speech`` displaced."""
    activity = getattr(session, "_activity", None)
    paused = getattr(activity, "_paused_speech", None) if activity is not None else None
    if paused is None or getattr(paused, "handle", None) is not speech:
        return
    try:
        activity._restore_paused_speech_state(paused)
    except Exception:  # pragma: no cover - runtime moved its bookkeeping
        pass


def guard_playout_hold(
    session: Any,
    *,
    policy: PlayoutHoldPolicy | None = None,
    boundary: BoundaryStopPolicy | None = None,
    on_release: Callable[[int, str], None] | None = None,
    on_boundary_tail_granted: Callable[[int], None] | None = None,
    word_evidence: Callable[[], int] | None = None,
    arm_word_gate: Callable[[Any], bool] | None = None,
    no_word_budget_seconds: float | None = None,
    reply_in_flight_states: frozenset[str] = frozenset(),
    on_hold_ended: Callable[[int, int, bool, str], None] | None = None,
    current_speech_id: Callable[[], str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> BoundedPlayoutHoldOutput | None:
    """Install the guard on a started session; ``None`` when it has no audio.

    Called after ``session.start()`` so ``RoomIO`` has already published its
    output and the transcript synchronizer is wired: the guard goes on top of
    whatever that produced rather than replacing any of it.

    ``word_evidence``, ``arm_word_gate`` and ``no_word_budget_seconds`` act
    together or not at all: a call whose transcriber timing is measured
    (``playout_hold.no_word_hold_budget_seconds``) gets the short no-word
    budget AND the word gate after it, and every other call keeps the full
    budget and the stand-down exactly as before. Given only some of
    them, none applies. A supplied ``policy`` is used as built.
    """
    existing = getattr(getattr(session, "output", None), "audio", None)
    if existing is None or isinstance(existing, BoundedPlayoutHoldOutput):
        return None
    if word_evidence is None or arm_word_gate is None or no_word_budget_seconds is None:
        word_evidence = arm_word_gate = no_word_budget_seconds = None
    if policy is None:
        policy = PlayoutHoldPolicy(
            word_evidence=word_evidence,
            no_word_max_hold_seconds=no_word_budget_seconds,
            clock=clock,
        )
    gate_arm = arm_word_gate
    guard = BoundedPlayoutHoldOutput(
        next_in_chain=existing,
        policy=policy,
        boundary=boundary,
        stand_down=lambda: stand_down_current_speech(session),
        hand_over=(
            (
                lambda: hand_current_speech_to_word_gate(
                    session, gate_arm, reply_in_flight_states
                )
            )
            if gate_arm is not None
            else None
        ),
        on_release=on_release,
        on_boundary_tail_granted=on_boundary_tail_granted,
        on_hold_ended=on_hold_ended,
        current_speech_id=current_speech_id,
    )
    session.output.audio = guard
    return guard


__all__ = [
    "BoundedPlayoutHoldOutput",
    "guard_playout_hold",
    "hand_current_speech_to_word_gate",
    "stand_down_current_speech",
]

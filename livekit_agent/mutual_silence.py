# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""End a call where both sides have gone quiet.

**What it is for.** Once a caller had said one word, nothing ended a line on
which nobody spoke any more. The no-caller-speech rule is disarmed by that first
word; the answer-detection arbiter stands down on any human evidence; the
silent-model rule (``model_unresponsive``) needs an unanswered caller turn to arm. A 2026-09-23
call spoke three words at 4.7 s and then sat silent for 15 minutes, billed as a
conversation, until an operator deleted the room.

**The numbers are measured.** Across 2,197 mid-call silences on 127 production
calls with caller words, none reached 45 s and one reached 30 s (34.1 s, broken
by the caller). So the agent checks in after :data:`CHECK_IN_AFTER_SECONDS` of
mutual silence and the call ends :data:`END_AFTER_CHECK_IN_SECONDS` after that
check-in has played. A caller who answers resumes the conversation. The three
production tails of 45 s or more (1022 s, 65 s, 65 s) would each have ended
about 45 s into their silence.

**Whose silence it is.** The clock runs only while the AGENT HAD THE LAST WORD:
the room heard the agent more recently than the caller's last words. While the
caller spoke last the agent owes a reply, which is ``model_unresponsive``'s question and not
this one; this rule never reads that rule's state, so neither can hold the
other open.

* *Caller words* are any non-blank transcript, interim or final, from the
  engine's own recognition -- unfiltered, so a "yes" answering the check-in
  counts. A transcript is dated by the END of the caller VAD run it belongs
  to, not by its arrival: an engine can deliver the final text of a turn after
  the agent has already answered it, and dating it by arrival would hand the
  floor back to a caller whose turn was over.
* *Agent output* is the room hearing the agent -- the evidence sidecar's live
  edge advancing -- or the runtime's ``speaking`` TRANSITION, as an instant.
  Never the state as a level: a voice-to-voice runtime was measured holding
  ``speaking`` for 47 polls with its provider gone.
* *Voice activity* never resets the clock and never counts as words: the pinned
  VAD scores line noise at 0.95-0.99. An open caller run FREEZES the clock --
  it may be a caller whose words have not arrived yet -- up to
  :data:`VAD_FREEZE_CAP_SECONDS` per silent episode, after which a wordless run
  is silence.

**What it never does.** It never speaks over, or hangs up on, a caller who is
mid-run inside the cap; never runs while a tool, a workflow turn, a language
switch or the ending of the call is in flight; never runs after a transfer to a
human was answered (the two people are bridged and cannot hear the agent); and
never checks in more than :data:`MAX_CHECK_INS_PER_CALL` times. A silent episode
after the cap ends at the same N+M, never sooner.

Everything here is pure and synchronous. Speaking the check-in and ending the
call are injected; the worker wiring is its event handlers and one call from
the poll loop that already runs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

#: The worker log event: one line per state change or action, never per tick.
DECISION_EVENT = "mutual_silence_decision"

#: Mutual silence before the agent asks whether the caller is still there.
CHECK_IN_AFTER_SECONDS = 30.0
#: Further silence after the check-in has played before the call ends.
END_AFTER_CHECK_IN_SECONDS = 15.0
#: How long an open caller VAD run with no words may freeze one silent
#: episode's clock. Beyond it a wordless run is treated as silence.
VAD_FREEZE_CAP_SECONDS = 20.0
#: Check-ins per call. Past it, an episode ends at the same N+M, unannounced.
MAX_CHECK_INS_PER_CALL = 3
#: How long the check-in may take to start and play before the rule stops
#: waiting for it. GPT-Live allows a requested reply ten seconds to start.
CHECK_IN_BUDGET_SECONDS = 15.0
#: Agent output within this long after the check-in finished is still the
#: check-in's own tail, not the agent resuming the conversation.
CHECK_IN_TAIL_SECONDS = 1.0
#: How soon after the agent's last output a transcript may still be the late
#: final of the caller turn that output answered. Engines and the transcript
#: gate hold a turn's final text until the reply to it has PLAYED, so a late
#: final lands right after the reply ends however long the reply was; words
#: arriving later than this belong to new speech -- possibly speech the VAD
#: never saw (it was measured peaking at 0.33 on one narrowband leg). Misjudging
#: fresh words as late here costs one unnecessary check-in, never an ending:
#: the answer to that check-in arrives well past this grace.
LATE_FINAL_GRACE_SECONDS = 5.0

#: The ``hold`` reason a worker returns while the call is waiting in a phone
#: menu or a hold queue. Unlike the other holds it does not merely freeze the
#: clock: the silence of a queue is not the silence of two people, so the
#: episode starts again when conversation resumes.
HOLD_NAVIGATION = "navigation"
#: The ``hold`` reason while only the ENTRY WINDOW defers the rule: the call may
#: still be an ordinary conversation, so this is an ordinary freeze -- silence
#: already counted is kept, not restarted.
HOLD_ENTRY_WINDOW = "entry_window"

MODE_ENFORCE = "enforce"
MODE_SHADOW = "shadow"
MODE_OFF = "off"
MODES = frozenset({MODE_ENFORCE, MODE_SHADOW, MODE_OFF})
#: The worker environment variable an operator flips without an image change.
MODE_ENV = "SKYSAY_MUTUAL_SILENCE_MODE"

# States.
IDLE = "idle"  # not counting: nothing armed yet, or the caller spoke last
COUNTING = "counting"
CHECKING_IN = "checking_in"
AWAITING = "awaiting"  # the check-in played; waiting for the caller
AWAITING_NO_CHECK_IN = "awaiting_no_check_in"  # no check-in available; same deadline
ENDED = "ended"
RETIRED = "retired"

# Outcomes (the ``outcome`` field of every decision row).
OUTCOME_ARMED = "armed"
OUTCOME_CALLER_SPOKE = "reset_caller_words"
OUTCOME_AGENT_SPOKE = "reset_agent_audio"
OUTCOME_CHECK_IN_SPOKEN = "check_in_spoken"
OUTCOME_CHECK_IN_NOT_DELIVERED = "check_in_not_delivered"
OUTCOME_CHECK_IN_ABANDONED = "check_in_abandoned"
OUTCOME_CHECK_IN_PLAYED = "check_in_played"
OUTCOME_NO_CHECK_IN = "check_in_unavailable"
OUTCOME_ENDED = "ended"
OUTCOME_ALREADY_ENDING = "call_already_ending"
OUTCOME_WOULD_CHECK_IN = "would_check_in"
OUTCOME_WOULD_END = "would_end"
OUTCOME_RETIRED_TRANSFERRED = "retired_transferred"
OUTCOME_PROMOTED = "promoted_after_refusal"
OUTCOME_EPISODE_BEGUN = "post_refusal_episode"


def resolve_mode(environ: Mapping[str, str]) -> str:
    """The rule's mode from the worker environment; SHADOW unless set.

    Production default is ``shadow`` (lead decision, 2026-09-23): the rule
    logs every check-in and ending it WOULD perform and changes nothing a
    caller hears, until shadow data proves the state machine on real calls.
    ``enforce`` is set explicitly; junk reads as shadow, never as enforce --
    an unreadable switch must not start hanging up on callers.
    """

    value = str(environ.get(MODE_ENV) or "").strip().lower()
    return value if value in MODES else MODE_SHADOW


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and number not in (float("inf"), float("-inf")) else None


class MutualSilenceRule:
    """Check in after N seconds of mutual silence; end M seconds after that."""

    def __init__(
        self,
        *,
        check_in: Callable[[], Any],
        end: Callable[[], bool],
        emit: Callable[..., None] = lambda *_a, **_k: None,
        agent_audible_at: Callable[[], float | None] = lambda: None,
        hold: Callable[[], str] = lambda: "",
        can_check_in: bool = True,
        mode: str = MODE_SHADOW,
        check_in_after_seconds: float = CHECK_IN_AFTER_SECONDS,
        end_after_check_in_seconds: float = END_AFTER_CHECK_IN_SECONDS,
    ) -> None:
        self._check_in = check_in
        self._end = end
        self._emit = emit
        self._agent_audible_at = agent_audible_at
        self._hold = hold
        #: False for a call whose replies are scripted by a workflow: a
        #: check-in there would be answered into the pending business step.
        self._can_check_in = bool(can_check_in)
        self._mode = mode if mode in MODES else MODE_SHADOW
        self._n = float(check_in_after_seconds)
        self._m = float(end_after_check_in_seconds)
        self._state = IDLE
        self._armed = False
        #: When the agent last produced output the room heard, and when the
        #: caller last spoke words (dated by their VAD run).
        self._agent_at: float | None = None
        self._caller_at: float | None = None
        #: The furthest the sidecar's agent edge has been seen to reach.
        self._audible_seen: float | None = None
        #: The open caller VAD run's start, and the end of the latest closed
        #: run -- what a transcript is dated by.
        self._vad_open_since: float | None = None
        self._last_run_end: float | None = None
        #: The silent episode's own clock: when it started and how much of it
        #: was frozen (held, or an open wordless run inside the cap).
        self._since: float | None = None
        self._frozen = 0.0
        self._vad_frozen = 0.0
        self._last_poll: float | None = None
        #: The check-in in flight, when it was asked for, and when it finished.
        self._handle: Any = None
        self._check_in_at: float | None = None
        self._check_in_done_at: float | None = None
        self._check_ins = 0
        self._eligible = True
        #: Whether the latest check-in, or the ending, was shadow's
        #: hypothetical one: recorded like a real one so shadow data reads
        #: right, but never heard. A post-refusal episode clears exactly
        #: these, so the first ENFORCED check-in is real and audible.
        self._hypothetical_check_in = False
        self._hypothetical_end = False
        self._last_hold = ""
        #: The latest caller evidence of any kind (a VAD edge or words), for
        #: the shadow-mode counterfactual below.
        self._caller_activity_at: float | None = None
        self._recover_unanswered_close = False

    # -- properties ------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def check_ins(self) -> int:
        return self._check_ins

    @property
    def mode(self) -> str:
        return self._mode

    # -- feeds -----------------------------------------------------------

    def arm(self, *, at: float) -> None:
        """The caller can now be heard (input released after answer)."""

        if self._armed or self._mode == MODE_OFF:
            return
        self._armed = True
        self._agent_at = float(at) if self._agent_at is None else self._agent_at
        self._begin_counting(float(at))
        self._log(OUTCOME_ARMED, at=at)

    def note_caller_vad(self, speaking: bool, *, at: float) -> None:
        at = float(at)
        self._caller_activity_at = at
        if speaking:
            if self._vad_open_since is None:
                self._vad_open_since = at
            return
        if self._vad_open_since is not None:
            self._last_run_end = at
        self._vad_open_since = None

    def note_caller_words(self, *, at: float) -> None:
        """A non-blank transcript, dated by the END of the VAD run it belongs to.

        The run's end, not its start and not the transcript's arrival. An
        engine can deliver a turn's final text after the agent has answered
        it: that run ended before the answer, so the answer still has the last
        word. A caller who talks over the agent's last words is the opposite
        case: their run ends after the agent's audio, so they have the floor.
        A run still open is the caller speaking now. A transcript arriving
        within ``LATE_FINAL_GRACE_SECONDS`` after the agent's output is dated
        by the last run's end -- a late final if that run ended before the
        output, and the caller holding the floor if it ended after. Every other
        transcript is dated by its arrival: it may be speech the VAD did not
        see at all.
        """

        at = float(at)
        self._caller_activity_at = at
        self._reconcile_check_in(at)
        # The agent's output as the room has heard it up to now, not as of the
        # last poll: the grace below is measured from it.
        self._pump_audible()
        spoken_at = at
        if (
            self._vad_open_since is None
            and self._last_run_end is not None
            and self._agent_at is not None
            and 0.0 <= at - self._agent_at <= LATE_FINAL_GRACE_SECONDS
        ):
            # The final of a turn the agent has since answered.
            spoken_at = self._last_run_end
        if self._caller_at is None or spoken_at > self._caller_at:
            self._caller_at = spoken_at
        if self._state in (RETIRED, ENDED):
            return
        # Words after the check-in was asked ARE the answer to it, whoever
        # holds the floor now -- a late final can land after the agent has
        # already replied to them. The next silence gets its own check-in.
        if self._check_in_at is not None and spoken_at >= self._check_in_at:
            self._eligible = True
        # Only a caller who spoke AFTER the agent's last output takes the
        # floor. The final text of a turn the agent already answered arrives
        # late on some engines; it does not hand the floor back.
        if self._agent_at is not None and spoken_at <= self._agent_at:
            return
        if self._state == CHECKING_IN:
            self._abandon_check_in(at)
        previous = self._state
        if self._recover_unanswered_close:
            # A cancelled native close can produce a caption after its reply,
            # without another model turn. Keep a bounded clarification clock
            # from these words; every fresh turn gets the full interval.
            self._begin_counting(at)
        else:
            self._state = IDLE
            self._since = None
        if previous != IDLE:
            self._log(OUTCOME_CALLER_SPOKE, at=at)

    def note_agent_speaking(self, *, at: float) -> None:
        """The runtime's ``speaking`` TRANSITION. An instant, never a level.

        Outside the check-in a new transition is a NEW reply, never the
        check-in's tail -- the check-in's own transition fired when it
        started -- so it ends the tail window: only the room's audio edges,
        which cannot say whose audio they are, stay suppressed inside it.
        """

        self._reconcile_check_in(float(at))
        if self._state != CHECKING_IN:
            self._check_in_done_at = None
        self._note_agent_output(float(at))

    def note_transferred(self, *, at: float) -> None:
        """A transfer to a human was answered: the two people are bridged."""

        if self._state == RETIRED:
            return
        self._abandon_check_in(float(at))
        self._state = RETIRED
        self._log(OUTCOME_RETIRED_TRANSFERRED, at=at)

    # -- after the end_call guard refused a hang-up -----------------------

    def promote_to_enforce(self, *, at: float) -> bool:
        """Enforce for the rest of this call; whether the mode moved.

        The end_call guard calls this on every refusal: the platform declined
        the agent's own hang-up, so the platform owes the caller the ending a
        quiet call gets, whatever the workspace's silence setting was. Only
        the MODE moves here. ``shadow`` becomes ``enforce``; ``enforce``
        stays; ``off`` stays off, and a deployment that switched the rule off
        has no silence bound for refused calls either (the rollout rule says
        not to run the guard in enforce there). The episode itself is
        :meth:`begin_post_refusal_episode`: a mode flip alone would inherit
        whatever the shadow run left behind, which review found twice to be a
        rule that never speaks again.
        """

        if self._mode != MODE_SHADOW:
            return False
        self._mode = MODE_ENFORCE
        self._log(OUTCOME_PROMOTED, at=float(at))
        return True

    def begin_post_refusal_episode(self, *, at: float, recover_unanswered: bool = False) -> str:
        """Put the rule on the clock after a refused hang-up; the state after.

        An explicit matrix over the rule's ACTUAL states, independent of the
        mode it ran in until now:

        * ``RETIRED`` (a transfer bridged the caller) and a REAL ending keep
          their state: nothing here may speak into a bridged call or relabel
          an ending already stamped.
        * A hypothetical shadow history -- a ``would_check_in`` the caller
          never heard, a ``would_end`` that never happened, or the
          no-check-in branch it led to -- is cleared: budget, eligibility and
          check-in timing reset, and a fresh ``COUNTING`` episode starts at
          ``at``, so the first enforced check-in is real and audible. Shadow's
          per-call cap must not be spent by check-ins nobody heard.
        * ``IDLE`` (the caller holds the floor) becomes ``COUNTING`` from
          ``at``: a refusal can produce no agent output at all (a cancelled
          follow-up, an interrupted tool), and an idle rule polls to nothing
          forever. The refusal is the agent's last word for this purpose;
          the caller's next words hand the floor back exactly as today.
        * ``COUNTING``, and a REAL check-in in flight or played under
          enforce, keep their clocks.

        ``recover_unanswered`` is for an accepted close withdrawn on caller
        activity. Later words restart a full quiet interval rather than parking
        the rule in IDLE: a native caption may arrive after the model's reply,
        or without a further model turn. This enables a clarification, never
        an immediate close. Ordinary refused-before-acceptance calls retain
        their existing caller-last-word behavior.

        A rule that was never armed (input not yet released) is armed here:
        a refusal means the call is live.
        """

        at = float(at)
        if self._mode == MODE_OFF:
            return self._state
        if self._state == RETIRED:
            return self._state
        if self._state == ENDED and not self._hypothetical_end:
            return self._state
        # Only accepted-then-reclaimed closes opt in. A missing native reply or
        # late caption must not leave this recovered call idle forever. The
        # existing holds, caller activity and check-in budgets still apply.
        self._recover_unanswered_close |= recover_unanswered
        if self._state == CHECKING_IN:
            return self._state
        if self._state == AWAITING and not self._hypothetical_check_in:
            return self._state
        if self._hypothetical_check_in or self._hypothetical_end:
            self._check_ins = 0
            self._eligible = True
            self._check_in_at = None
            self._check_in_done_at = None
            self._handle = None
            self._hypothetical_check_in = False
            self._hypothetical_end = False
        self._armed = True
        # The refusal is the agent's last word: a final dated before it is a
        # late final of a turn already answered, never the floor taken back.
        if self._agent_at is None or at > self._agent_at:
            self._agent_at = at
        if self._state != COUNTING:
            self._begin_counting(at)
        self._log(OUTCOME_EPISODE_BEGUN, at=at)
        return self._state

    # -- the decision ----------------------------------------------------

    def poll(self, now: float) -> str:
        """Advance the clock, act if due, and name what happened."""

        now = float(now)
        self._pump_audible()
        elapsed = 0.0 if self._last_poll is None else max(0.0, now - self._last_poll)
        self._last_poll = now
        if self._mode == MODE_OFF or not self._armed:
            return IDLE
        if self._state in (ENDED, RETIRED, IDLE):
            return self._state
        hold = self._read_hold()
        if hold:
            if hold == HOLD_NAVIGATION and self._state in (
                COUNTING,
                AWAITING,
                AWAITING_NO_CHECK_IN,
            ):
                # A pre-spent episode must not survive the hold: 29 seconds of
                # silence before the menu would otherwise leave one second of
                # it at pickup, and the agent would check in on a person who
                # had just said hello.
                self._begin_counting(now)
            else:
                self._frozen += elapsed
            if hold != self._last_hold:
                self._log("held_" + hold, at=now)
            self._last_hold = hold
            return "held_" + hold
        self._last_hold = ""
        if self._vad_open_since is not None and self._vad_frozen < VAD_FREEZE_CAP_SECONDS:
            freeze = min(elapsed, VAD_FREEZE_CAP_SECONDS - self._vad_frozen)
            self._vad_frozen += freeze
            self._frozen += freeze
        if self._state == COUNTING:
            if self._silent_for(now) < self._n:
                return COUNTING
            return self._due_check_in(now)
        if self._state == CHECKING_IN:
            return self._poll_check_in(now)
        if self._state in (AWAITING, AWAITING_NO_CHECK_IN):
            deadline = self._m if self._state == AWAITING else self._n + self._m
            if self._silent_for(now) < deadline:
                return self._state
            return self._due_end(now)
        return self._state

    # -- internals ---------------------------------------------------------

    def _pump_audible(self) -> None:
        try:
            edge = _finite(self._agent_audible_at())
        except Exception:  # noqa: BLE001 - a reader cannot decide by failing
            return
        if edge is None:
            return
        if self._audible_seen is not None and edge <= self._audible_seen:
            return
        self._audible_seen = edge
        self._note_agent_output(edge)

    def _note_agent_output(self, at: float) -> None:
        # The check-in's own audio is not the agent taking the last word: a
        # caller answering over it ("yes, I'm here") must still hold the floor
        # when their transcript arrives after the question has finished. So it
        # moves nothing -- not the clock, and not the agent's last word.
        if self._state == CHECKING_IN:
            return
        # Its room tail too -- after it finished, and after a caller answering
        # over it cut it short. A tail that restarted the clock would ask a
        # caller who had just answered whether they were still there.
        done = self._check_in_done_at
        if done is not None and at <= done + CHECK_IN_TAIL_SECONDS:
            return
        if self._agent_at is None or at > self._agent_at:
            self._agent_at = at
        if self._state in (RETIRED, ENDED) or not self._armed:
            return
        # The agent had the last word, so the silence, if any, starts here.
        # A caller who spoke after this instant still holds the floor.
        if self._caller_at is not None and self._caller_at > at:
            return
        previous = self._state
        self._begin_counting(at)
        if previous not in (COUNTING, IDLE):
            self._log(OUTCOME_AGENT_SPOKE, at=at)

    def _begin_counting(self, at: float) -> None:
        self._state = COUNTING
        self._since = at
        self._frozen = 0.0
        self._vad_frozen = 0.0

    def _silent_for(self, now: float) -> float:
        if self._since is None:
            return 0.0
        return max(0.0, now - self._since - self._frozen)

    def _read_hold(self) -> str:
        try:
            return str(self._hold() or "")
        except Exception:  # noqa: BLE001 - an unreadable gate holds
            return "unreadable"

    def _due_check_in(self, now: float) -> str:
        if not (self._can_check_in and self._eligible and self._check_ins < MAX_CHECK_INS_PER_CALL):
            # Same deadline as a check-in would have given, never sooner.
            self._state = AWAITING_NO_CHECK_IN
            self._log(OUTCOME_NO_CHECK_IN, at=now)
            return AWAITING_NO_CHECK_IN
        if self._mode == MODE_SHADOW:
            # Record the hypothetical check-in exactly as a real one, so a
            # caller's answer restores eligibility and shadow data shows what
            # enforce would do on the NEXT episode too. It counts toward the
            # per-call cap for the same reason.
            self._check_ins += 1
            self._check_in_at = now
            self._hypothetical_check_in = True
            self._log(OUTCOME_WOULD_CHECK_IN, at=now)
            self._enter_awaiting(now, check_in_done=now)
            self._eligible = False
            return OUTCOME_WOULD_CHECK_IN
        self._check_ins += 1
        self._eligible = False
        self._check_in_at = now
        self._hypothetical_check_in = False
        try:
            self._handle = self._check_in()
        except Exception:  # noqa: BLE001 - a failed check-in still waits M
            self._handle = None
        if self._handle is None:
            self._log(OUTCOME_CHECK_IN_NOT_DELIVERED, at=now)
            self._enter_awaiting(now, check_in_done=now)
            return OUTCOME_CHECK_IN_NOT_DELIVERED
        self._state = CHECKING_IN
        self._log(OUTCOME_CHECK_IN_SPOKEN, at=now)
        return OUTCOME_CHECK_IN_SPOKEN

    def _reconcile_check_in(self, at: float) -> None:
        """Notice a finished check-in NOW, not at the next poll.

        The handle can finish between polls. Until this runs the rule would
        read everything in that gap -- a reply to the caller's answer
        included -- as the check-in's own audio.
        """

        if self._state != CHECKING_IN or self._handle is None:
            return
        try:
            done = bool(self._handle.done())
        except Exception:  # noqa: BLE001 - an unreadable handle is not playing
            done = True
        if done:
            self._log(OUTCOME_CHECK_IN_PLAYED, at=at)
            self._enter_awaiting(at, check_in_done=at)

    def _poll_check_in(self, now: float) -> str:
        handle = self._handle
        done = False
        try:
            done = bool(handle.done())
        except Exception:  # noqa: BLE001 - an unreadable handle is not playing
            done = True
        if done:
            self._log(OUTCOME_CHECK_IN_PLAYED, at=now)
            self._enter_awaiting(now, check_in_done=now)
            return AWAITING
        started = self._check_in_at if self._check_in_at is not None else now
        if now - started >= CHECK_IN_BUDGET_SECONDS:
            self._abandon_check_in(now)
            self._log(OUTCOME_CHECK_IN_ABANDONED, at=now)
            self._enter_awaiting(now, check_in_done=now)
            return AWAITING
        return CHECKING_IN

    def _abandon_check_in(self, at: float) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        # Its remaining room audio is still the check-in's own tail.
        self._check_in_done_at = at
        try:
            if not handle.done():
                handle.interrupt()
        except Exception:  # noqa: BLE001 - best effort; the clock moves on
            pass

    def _enter_awaiting(self, now: float, *, check_in_done: float) -> None:
        self._handle = None
        self._check_in_done_at = check_in_done
        self._state = AWAITING
        self._since = now
        self._frozen = 0.0
        self._vad_frozen = 0.0

    def _due_end(self, now: float) -> str:
        if self._mode == MODE_SHADOW:
            self._log(OUTCOME_WOULD_END, at=now)
            self._state = ENDED
            self._hypothetical_end = True
            return OUTCOME_WOULD_END
        try:
            ended = bool(self._end())
        except Exception:  # noqa: BLE001 - not latched; the next tick retries
            ended = False
        if not ended:
            # Something else already stamped an ending, or the stamp was
            # released: not latched, so a released stamp is retried.
            self._log(OUTCOME_ALREADY_ENDING, at=now)
            return OUTCOME_ALREADY_ENDING
        self._state = ENDED
        self._log(OUTCOME_ENDED, at=now)
        return OUTCOME_ENDED

    def _log(self, outcome: str, *, at: float) -> None:
        details: dict[str, Any] = {
            "outcome": outcome,
            "state": self._state,
            "mode": self._mode,
            "check_ins": self._check_ins,
        }
        if self._since is not None:
            details["silent_for_seconds"] = round(self._silent_for(float(at)), 3)
            details["frozen_seconds"] = round(self._frozen, 3)
        if outcome in (OUTCOME_WOULD_CHECK_IN, OUTCOME_WOULD_END, OUTCOME_CHECK_IN_SPOKEN, OUTCOME_ENDED):
            # The counterfactual shadow data needs: did the caller do
            # anything within one second of a poll boundary before this
            # decision? Five review findings on this rule were timing corner
            # cases in exactly that window, so a would-end with this flag set
            # is the first one to listen to before enforcing.
            activity = self._caller_activity_at
            poll = self._last_poll
            details["caller_active_within_1s_of_poll"] = bool(
                activity is not None
                and poll is not None
                and abs(float(poll) - activity) <= 1.0
            )
            if activity is not None:
                details["seconds_since_caller_activity"] = round(
                    max(0.0, float(at) - activity), 3
                )
        try:
            self._emit(DECISION_EVENT, **details)
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass


__all__ = [
    "CHECK_IN_AFTER_SECONDS",
    "CHECK_IN_BUDGET_SECONDS",
    "DECISION_EVENT",
    "END_AFTER_CHECK_IN_SECONDS",
    "MAX_CHECK_INS_PER_CALL",
    "MODE_ENFORCE",
    "MODE_ENV",
    "MODE_OFF",
    "MODE_SHADOW",
    "MutualSilenceRule",
    "VAD_FREEZE_CAP_SECONDS",
    "resolve_mode",
]

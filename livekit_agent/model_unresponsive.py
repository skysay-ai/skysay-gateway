# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""End a call whose conversation engine has stopped answering the caller.

**Measured, not assumed.** The local SIP harness takes the engine's provider
host away from a call that is ALREADY a conversation, by blackholing it inside
the worker's own network namespace after the agent has been heard on the line.
Nothing is patched and the worker is not told. Across three engines, held for
thirty seconds while the caller kept talking:

* the stack whose model is a separate component reported three to five
  consecutive component errors with a connection cause, and the SDK session
  did **not** close;
* both voice-to-voice engines reported **nothing at all** -- no error, no
  ending, no reason code -- and then simply resumed when the provider came
  back.

In every case the leg stayed up and the call kept billing through half a
minute of dead air. So the ending decided here cannot be built on the engine
saying anything: on two of the three engines the engine says nothing, and on
the third the SDK's own error budget does not spend once a session has been
established.

**What survives the outage.** The worker's own pinned VAD. A client VAD is
attached to the voice-to-voice sessions as well as to the component stack, so
the speaking/not-speaking edges the session reports are produced locally and
keep arriving after the provider socket is gone. Input captions do not: on a
voice-to-voice engine they come off the very socket that died, which is why a
transcript-shaped witness would be blind in exactly the case this module
exists for.

So the rule is the pair that is available everywhere, with no branch per
engine: **a caller turn that ENDED, and no agent output after it within a
bound.** Nothing here names an engine, and nothing here reads a provider.

**What arms it, and what does not.** The arming edge is the END of a caller
turn -- the falling edge of the VAD run -- so the opening, before the caller
has ever spoken, can never arm it, and a caller who says nothing at all never
arms it either.

And only a run that BEGAN with the agent idle arms anything. The state that
decides is captured on the run's opening edge, not read at its close: a caller
who says "uh-huh" over a reply that is already playing has not asked the model
for anything, and the reply that answers them happened before their run
started, so no later output edge will ever arrive to clear a window opened by
it. Reading the gates at the falling edge instead would close nothing -- by
then the agent has finished and is idle again, which is exactly the shape that
ended a healthy call twenty seconds after a completed answer. Any output
DURING the run disqualifies it for the same reason. A silent CALLER is a different rule with a different name
(``no_caller_speech_after_answer``); this one is about a silent MODEL, and it
needs someone to have spoken TO the model before it can claim the model owed a
reply.

It arms on the FIRST unanswered caller turn end and does not re-arm on later
ones. A caller who keeps talking into dead air is the incident's own shape,
and a clock that restarted on every one of their turns would be pushed along
by the very silence it is meant to bound.

But it does not RUN while the caller is talking either, and the difference
between those two matters. The budget is what the AGENT owes, so it counts
only the time the floor was free: every caller speech run is subtracted, and
the rule declines to act while one is open. Without that, a caller who paused
at two seconds, resumed at three and then explained their problem for twenty
would be hung up on mid-sentence by a healthy agent that was correctly
listening to them. Subtracting rather than restarting is what keeps the
incident's shape covered: a caller speaking in short bursts into dead air
accumulates silence between the bursts and still reaches the bound.

**What clears it.** Agent OUTPUT, in three forms: a reply reaching the line, a
tool starting, and -- the one that matters most and was added last -- audio
the ROOM heard. Not an acknowledgement or tool-wait filler: those are played
by a ``BackgroundAudioPlayer``, whose track the evidence sidecar skips by
name, so a cached filler looping at a caller while the model is dead is the
failure this bounds and is invisible to every one of the three.

An output disqualifies a caller run that opens within
:data:`AGENT_AUDIBLE_WINDOW_SECONDS` of it, and that is not belt and braces.
The state edge fires when the reply is handed to the local source; the room
hears it about a seventh of a second later, so a backchannel landing in that
gap would read the audible edge as stale and arm on a perfectly healthy reply.

**What holds it.** A gate that says the model is legitimately not expected to
answer yet -- a tool is running, a workflow turn is in flight with the control
plane, or admission is already closed -- does not decide anything and does not
cancel either. It resets the clock, so the model gets a full bound from the
moment the reason for waiting goes away rather than being judged on a deadline
that expired while it was entitled to be quiet.

The agent SPEAKING is deliberately not on that list any more. It was, twice,
and both times it was wrong in the same direction.

**The agent speaking was a hold twice, and a hold is the wrong shape.** The
first build asked the runtime for its agent state. On one voice-to-voice call
the runtime left that state at "speaking" with the provider gone: the hold
re-clocked on all forty-seven polls, the budget never accrued a second, and
the rule that exists to end exactly that call ended nothing. A cap on the hold
was added and could not be reached -- the same flag also disqualified every
caller run that opened under it, so on a stuck runtime nothing ever armed.

A state is not a sound, so the second build read the ROOM: the evidence
sidecar's live agent edge, an independent subscriber whose timestamp advances
only on agent frames above the same quiet threshold the post-hoc clip
measurement uses. That fixed the signal and kept the shape, and the shape was
the other half of the defect. All three review engines drove the same
sequence: a gap longer than :data:`AGENT_AUDIBLE_WINDOW_SECONDS` inside one
streamed reply, a caller backchannel in the gap -- which arms, because by then
both run-open windows have expired -- and then the reply RESUMING. No new SDK
state edge is owed for a reply that never stopped, so nothing cleared the
armed window; the hold merely postponed it, and the call was ended
mid-playback with the room still hearing the agent.

So audible agent audio is not a hold. It is OUTPUT, and it clears. Every poll,
and every caller VAD edge, asks the sidecar where its agent edge has reached;
an advance beyond the furthest already seen is the caller hearing the agent,
and the window goes. A stuck runtime with a dead provider advances nothing and
is ended on the ordinary bound. There is no cap left to be unreachable.

The residual, named rather than implied: a provider that streamed audio at a
caller without ever answering them would clear this rule for as long as the
audio lasted. Nothing measured on any of the three engines does that -- an
outage is silence, not noise -- and the fillers that DO loop are published on
a track the sidecar skips. It is unmeasured rather than excluded.

The other half of reading a measurement is knowing when you are not making
one. A frozen edge from a sidecar that has stopped delivering frames on ANY
track is "unobserved", not "quiet": it holds, under its own name, uncapped,
and keeping the window's evidence rather than clearing it.

**The corroborator, and its honest asymmetry.** The decision also asks whether
a MODEL error was seen or no model response event arrived. Model: the SDK
names the failing component on every error it reports, and only the two that
are the model -- the LLM and the realtime engine -- corroborate here. A
synthesis error is the other half of the very distinction the stand-down
below draws, so letting a ``TTSError`` corroborate would have stamped
``model_unresponsive`` on exactly the call the module promises to leave
alone. A recognition error is not the model either, and an error this worker
cannot place is not evidence of anything. On a stack
whose model is its own component the first half does real work: that
component reports its errors. On a voice-to-voice engine neither half
discriminates: the engine emits nothing during an outage, so "no model
response arrived" is vacuously true the instant the bound expires. What it
DOES buy, on every engine, is the converse: a model that answered -- an
assistant item exists -- and produced no audio is a synthesis or playout
fault, which is a different defect with a different fix, and this rule stands
down and says so rather than stamping the model.

That corroborator is asked of the WINDOW, and the log says so in two fields
rather than one. ``model_response_seen`` is what the deciding window saw, and
on a healthy call it is false at every line on every engine: audible audio
clears the window at the first frame the room hears, and the assistant item
arrives only once the reply has finished playing, so an armed window is
already gone by the time an item could set it. It is true only on a call the
window is still holding open with an answer in hand -- and not even on the
stand-down row itself, which clears before the poll logs. Read as "the
engine never answered" it says something else entirely, and in production it
was: a 2026-09-23 batch reported this module's response witness as dead on
voice-to-voice engines from exactly that reading, on a call whose engine had
answered five times. ``model_responses_total`` is the per-CALL count,
written whatever the window is doing, and zero there is the only zero that
means an engine which has never answered. Nothing decides on it.

**The shadow bound, and why it is only a shadow.** The budget above stops
while the caller talks, and that is what let a dead GPT-Live session run 77 s
on 2026-09-24 (call_rlzb9WI0hLVDlyK): the caller kept talking into the silence,
44.6 s of speech were subtracted, and the bound was reached only after the
caller had given up. A second, wall-clock bound -- :data:`SHADOW_BOUND_SECONDS`
since arming, caller speech NOT subtracted -- is evaluated inside the same
decision, after the same holds, but only on a window with no sign of life at
all: no agent output (which clears the window anyway), no model response item,
and no transcript of the caller's words. The transcript condition is what
spares a healthy caller's long monologue: that caller is being transcribed. It
acts at a caller pause, or at :data:`SHADOW_NO_PAUSE_FACTOR` times the bound
if no pause comes. Today it only RECORDS that it would have ended the call
(``shadow_would_end``, once per window, and a per-call count in the
interruption summary); it ends nothing until production data shows it firing
only on dead sessions. Any transcript since arming counts, including the late
final of the very turn that armed it, so an engine that transcribed that turn
and then died is left to the main bound -- the cautious direction.

Everything here is pure and synchronous: the clock, the edges and the decision
live in this module, and the ending is one injected callable. The production
wiring is a handful of lines at the SDK event handlers and one call from the
poll loop that already runs.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

#: The worker log event. One line per decision this module reaches, whatever
#: it decided -- a silent no-op is the failure mode the whole ticket is about,
#: so every branch is named and countable.
DECISION_EVENT = "model_unresponsive_decision"

#: How long the agent has to produce something after a caller turn ended.
#:
#: Twenty seconds, and the number comes from the measurements rather than from
#: taste. A healthy model on every engine measured answers within a few
#: seconds -- the harness's own healthy runs put the next agent playout two
#: seconds after the provider came back. The floor is set by what else is
#: allowed to be waiting: the SDK's transcription timeout is 5.0 s, and the
#: repair line that answers it (``unrecognised_speech``) then generates and
#: speaks, which is agent output and clears this rule. Firing before that
#: sequence can complete would hang up on a call that was about to repair
#: itself. Twenty leaves it four times its measured room and still ends
#: a dead call inside a third of the thirty-second outage the harness holds.
#:
#: This default lives HERE and nowhere else. The control plane validates the
#: BOUNDS of a customer's override and stores only what they wrote, so a
#: stored value is a choice and an absent one is this constant -- there is no
#: second copy to drift.
DEFAULT_BOUND_SECONDS = 20.0

#: Bounds the control plane enforces on an override, repeated here because the
#: worker must also survive a value written straight into agent metadata by a
#: generic PATCH. Kept equal by ``tests/test_sch978_model_unresponsive.py``.
#:
#: Below the floor the rule would race the transcription timeout and the
#: repair line above it; above the ceiling it stops being a bound at all and a
#: customer is better served by the call-length limit they already have. Zero
#: is outside the range on purpose: it is not a duration, it is the off
#: switch, and it is read before the range is.
BOUND_SECONDS_BOUNDS = (10.0, 120.0)

#: How recently the ROOM must have heard the agent for this rule to treat it
#: as speaking, and how recently the sidecar must have delivered a frame on
#: some track for its answer to mean anything at all.
#:
#: Two seconds, and it is bounded on both sides by things that were measured
#: rather than chosen. Below it: the poll that reads this runs once a second,
#: the sidecar's edge advances per media frame (10-20 ms), and the room hears
#: a reply about 140 ms after the SDK hands it to the local source -- so a
#: window under a second would report a healthy reply as silence on scheduling
#: jitter alone. Above it: every second here is a second of a pause INSIDE a
#: reply that reads as speech, and the whole of a dead call's evidence is that
#: nothing is arriving. Inter-sentence pauses in synthesised speech are a few
#: hundred milliseconds; two seconds clears them several times over and is a
#: tenth of the bound it gates, so it can delay an ending but never hide one.
#:
#: The same number does both jobs on purpose. "Nothing from the agent for two
#: seconds" and "nothing from the sidecar for two seconds" are the same
#: observation made about two different sources, and holding them apart would
#: be two numbers to keep in step for no behaviour anyone can name.
AGENT_AUDIBLE_WINDOW_SECONDS = 2.0

#: Which SDK error components are the MODEL. The names are LiveKit's own, as
#: the worker's error mapping produces them: an ``llm_error`` on the component
#: stack and a ``realtime_model_error`` on a voice-to-voice engine. Anything
#: else -- synthesis, recognition, or a shape the worker cannot place -- is
#: not evidence that the model stopped answering.
MODEL_ERROR_COMPONENTS = frozenset({"llm", "realtime"})

#: The shadow bound: seconds since arming, caller speech NOT subtracted, on a
#: window with no output, no model response and no caller transcript. Replayed
#: against the 2026-09-24 dead GPT-Live call it would have acted at its first
#: caller pause after 45 s (56.4 s after arming) instead of the 65.5 s the
#: budget took. Recorded, never acted on, until production data says it only
#: ever fires on dead sessions.
SHADOW_BOUND_SECONDS = 45.0

#: With no caller pause, the shadow acts at this multiple of its bound anyway:
#: a caller who never stops talking into a dead line must not keep it up.
SHADOW_NO_PAUSE_FACTOR = 1.5

#: Outcomes, in the order they are decided. Named rather than boolean so the
#: harness run table and a production log say WHY a call was or was not ended.
ENDED = "ended"
SKIP_DISABLED = "disabled_by_agent"
SKIP_CALLER_SPEAKING = "caller_still_speaking"
SKIP_NOT_ARMED = "no_unanswered_caller_turn"
SKIP_WITHIN_BOUND = "within_bound"
SKIP_TOOL_IN_FLIGHT = "tool_in_flight"
SKIP_WORKFLOW_TURN_PENDING = "workflow_turn_pending"
SKIP_ADMISSION_CLOSED = "admission_closed"
SKIP_MODEL_ANSWERED = "model_answered_without_audio"
SKIP_ALREADY_ENDING = "call_already_ending"
#: A caller run that opened while the agent was already busy. It asked the
#: model for nothing, so nothing is owed for it.
SKIP_RUN_OVERLAPPED_OUTPUT = "caller_run_overlapped_agent_output"
#: The sidecar that measures whether the room heard the agent has stopped
#: delivering frames. A frozen edge then means "nobody is watching", not
#: "nothing was heard", and a rule that hangs up must not read the second out
#: of the first. It holds -- and, unlike the speaking hold, it is NOT capped:
#: a cap would eventually let an unobserved call be ended on no evidence,
#: which is the failure this outcome exists to refuse. The cost, named: a call
#: whose evidence sidecar has stopped observing is not covered by this rule,
#: and ``evidence_sidecar_disconnected`` is where that shows up.
SKIP_AGENT_AUDIO_UNOBSERVED = "agent_audio_unobserved"
#: The call is waiting in a phone menu or a hold queue: the agent is meant to
#: be silent and the caller side is a recording, so no answer is owed. Re-clocks
#: like the holds above, so the bound restarts when conversation resumes.
SKIP_NAVIGATION = "navigation_suspended"
#: The entry window defers this rule without resetting it: the call may turn
#: out to be an ordinary conversation, in which case the bound it was already
#: spending is still owed. Nothing ends and no recovery is asked while deferred.
SKIP_DEFERRED = "entry_window_deferred"
#: Not a skip: the window closed because the agent produced something.
OUTCOME_CLEARED = "cleared"
#: Not a decision either: the shadow bound would have ended this window's call.
#: Logged once per window, beside whatever the real decision was.
SHADOW_WOULD_END = "shadow_would_end"
#: Why the shadow would have acted: the caller paused, or never did.
SHADOW_AT_CALLER_PAUSE = "caller_pause"
SHADOW_AT_NO_PAUSE_CAP = "no_pause_cap"

#: Why the clock was cleared, for the log line on the edge that cleared it.
#: The set is closed for the same reason the outcomes are.
OUTPUT_REPLY = "agent_reply"
OUTPUT_TOOL = "tool_call"
#: Not an SDK edge but a measurement: the room heard the agent. It is output
#: in the only sense this rule means by the word -- the caller heard
#: something -- and it is the only one that keeps arriving through a reply
#: the SDK announced once and never mentions again.
OUTPUT_AUDIO = "agent_audio_on_the_line"
#: Not output, and not a hold either: a reply the agent HAD, which our own
#: turn-taking withheld (the far-end floor hold) until the caller spoke again
#: and it was cancelled. The model answered; see ``note_reply_withheld``.
OUTCOME_REPLY_WITHHELD = "reply_withheld"


def bounded_seconds(raw: Any) -> float | None:
    """A configured bound in seconds, ``0.0`` for off, or ``None`` when unset.

    Junk is ``None`` -- the platform default -- rather than a guess or an
    accidental disable. A value this cannot parse is not a number someone
    chose, and reading it as "off" would silently give an agent back the
    unbounded call this module exists to prevent.
    """

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        # The type as well as the range. A string is what a generic PATCH
        # writes when somebody typed the number into a text field, and
        # ``float("20")`` would quietly accept it here while the control
        # plane refuses it -- two deployment units disagreeing about whether
        # a customer configured anything.
        return None
    seconds = float(raw)
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    if seconds == 0.0:
        return 0.0
    low, high = BOUND_SECONDS_BOUNDS
    if not low <= seconds <= high:
        return None
    return seconds


def agent_audio_observed(
    observer_last_frame_at: Any,
    *,
    now: float,
    window: float = AGENT_AUDIBLE_WINDOW_SECONDS,
) -> bool:
    """Whether the sidecar is still reporting, so its agent edge means anything.

    ``observer_last_frame_at`` is when the sidecar last delivered a frame on
    EITHER track. The caller's leg publishes continuously while connected, so
    on the dead-air call this rule exists for the caller's own frames keep
    this true and the frozen agent edge is read, correctly, as silence. A
    sidecar that has delivered nothing at all is not observing -- and neither
    is one that has never delivered anything, which is why ``None`` is false
    rather than a benign default.
    """

    if not isinstance(observer_last_frame_at, (int, float)) or isinstance(
        observer_last_frame_at, bool
    ):
        return False
    last = float(observer_last_frame_at)
    if last != last:  # NaN
        return False
    return float(now) - last <= float(window)


def agent_audio_audible(
    agent_audible_until: Any,
    *,
    now: float,
    window: float = AGENT_AUDIBLE_WINDOW_SECONDS,
) -> bool:
    """Whether the room heard the agent within ``window`` seconds of ``now``.

    ``agent_audible_until`` is the evidence sidecar's live agent edge: the
    last moment an agent frame above the quiet threshold reached the room. A
    call on which the agent has never been audible answers false rather than
    holding: there is no reply in flight to wait for, and the opening this
    rule can never arm before has always produced one by the time a caller
    turn has ended.
    """

    if not isinstance(agent_audible_until, (int, float)) or isinstance(
        agent_audible_until, bool
    ):
        return False
    audible = float(agent_audible_until)
    if audible != audible:  # NaN
        return False
    return float(now) - audible <= float(window)


def model_unresponsive_seconds(realtime_config: Mapping[str, Any] | None) -> float:
    """Resolve ``voice_polish.model_unresponsive.seconds`` for one call.

    ON unless the agent says otherwise: an absent section, an absent key and a
    section this version cannot read all mean the platform default. That is
    the same polarity as ``unrecognised_speech`` and for the same reason --
    the behaviour not running IS the reported defect, so an unreadable setting
    must not be a way to reinstate it.

    An explicit ``0`` is the off switch, and the only thing that turns this
    off. It is a real choice a customer may need -- an agent that waits on a
    long human-in-the-loop tool, say -- so it is honoured exactly, and it is
    the only value outside the bounds that is.
    """

    polish = (
        realtime_config.get("voice_polish")
        if isinstance(realtime_config, Mapping)
        else None
    )
    section = (
        polish.get("model_unresponsive") if isinstance(polish, Mapping) else None
    )
    if not isinstance(section, Mapping):
        return DEFAULT_BOUND_SECONDS
    seconds = bounded_seconds(section.get("seconds"))
    return DEFAULT_BOUND_SECONDS if seconds is None else seconds


class ModelUnresponsiveRule:
    """The caller spoke, the agent produced nothing, the bound elapsed.

    Feed it the three things the session already reports -- caller VAD edges,
    agent output, and what the engine said -- and poll it from the loop that
    already runs. It decides; the injected ``end_call`` performs.
    """

    def __init__(
        self,
        *,
        seconds: float,
        end_call: Callable[[], bool],
        emit: Callable[..., None] = lambda *_args, **_details: None,
        agent_audible_at: Callable[[], float | None] = lambda: None,
        agent_audio_observed: Callable[[], bool] = lambda: True,
        tool_in_flight: Callable[[], bool] = lambda: False,
        workflow_turn_pending: Callable[[], bool] = lambda: False,
        navigation_suspended: Callable[[], bool] = lambda: False,
        deferred: Callable[[], bool] = lambda: False,
        admits: Callable[[], bool] = lambda: True,
        is_ending: Callable[[], bool] = lambda: False,
        recover_reply: Callable[[], bool] | None = None,
    ) -> None:
        self._seconds = float(seconds)
        self._end_call = end_call
        self._emit = emit
        self._agent_audible_at = agent_audible_at
        self._agent_audio_observed = agent_audio_observed
        self._tool_in_flight = tool_in_flight
        self._workflow_turn_pending = workflow_turn_pending
        self._navigation_suspended = navigation_suspended
        self._deferred = deferred
        self._admits = admits
        self._is_ending = is_ending
        self._recover_reply = recover_reply
        self._recovery_used = False
        self._caller_speaking = False
        self._speaking_since: float | None = None
        #: Captured when the caller's run OPENS, and read when it closes.
        #: ``""`` means the run may arm; anything else is the outcome the
        #: closing edge logs, so a disqualified run says WHICH gate stopped it
        #: rather than all of them saying the same thing.
        self._run_blocked_by = ""
        #: The last agent output edge, for the window it takes to be heard.
        self._last_output_at: float | None = None
        #: The furthest the room's agent edge has been seen to reach. Every
        #: advance beyond it is a moment the caller heard the agent.
        self._audible_seen: float | None = None
        self._caller_seconds = 0.0
        self._armed_at: float | None = None
        self._provider_error = False
        self._model_response = False
        #: Every response item the engine has produced on this call, armed or
        #: not, and when the last one arrived. Neither is read by the
        #: decision: they exist so a log line can say whether the ENGINE has
        #: ever answered, separately from whether THIS window saw an answer.
        self._model_responses = 0
        self._last_model_response_at: float | None = None
        self._decided = False
        #: The latest caller run that closed, as (started, ended).
        self._last_closed_run: tuple[float, float] | None = None
        #: Per WINDOW: the caller's words were transcribed since arming, so
        #: something is still listening and the shadow bound does not apply.
        self._window_transcribed = False
        #: Per WINDOW: the shadow already recorded this window.
        self._shadow_reported = False
        #: Per CALL, for the interruption summary.
        self._caller_transcripts = 0
        self._shadow_would_end = 0
        self._shadow_first_armed_ms: int | None = None
        # Caller continuity, for the log only: a caller side that is
        # one long run of audio (a hold tune, a caller tune) reads here as a
        # few long runs covering most of the window, where a person talking
        # into a silent line reads as short bursts with gaps. Nothing decides
        # on these.
        self._arming_run_seconds: float | None = None
        self._runs_since_armed = 0
        self._longest_run_since_armed = 0.0
        self._caller_speech_call = 0.0
        self._longest_run_call = 0.0
        # When the open run really began and how long the last closed run
        # really lasted. ``_speaking_since`` is moved forward by a rearm (only
        # the new window's share is subtracted), so it cannot say either.
        self._run_opened_at: float | None = None
        self._last_run_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self._seconds > 0.0

    @property
    def seconds(self) -> float:
        return self._seconds

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    # -- the three feeds ------------------------------------------------

    def note_caller_speaking(self, speaking: bool, *, at: float) -> None:
        """One ``user_state_changed`` transition, level-tracked.

        Only the CHANGE is an edge, so a runtime that re-announces the same
        state cannot manufacture turns, and only the falling edge -- the end
        of the caller's run -- arms anything. A runtime that grows a third
        state reads as not-speaking here and so can only ever close a run,
        never open one it did not open.
        """

        # BEFORE the level below is updated, so the run that is closing is
        # still open here: audio heard during it disqualifies it, exactly as
        # an SDK output edge during it does.
        self._pump_audible(at)
        speaking = bool(speaking)
        if speaking == self._caller_speaking:
            return
        self._caller_speaking = speaking
        if speaking:
            # The caller taking the floor again does NOT restart the clock --
            # the agent still owes a reply to the turn that armed it, and a
            # caller filling the dead air with more speech would otherwise
            # push the bound along indefinitely. It does STOP it: the run
            # that opens here is subtracted when it closes below.
            self._speaking_since = float(at)
            self._run_opened_at = float(at)
            # Captured HERE, on entry, and read when the run closes. A gate
            # read at the closing edge has already gone quiet by then.
            self._run_blocked_by = self._entry_block(float(at))
            return
        started = self._speaking_since
        self._speaking_since = None
        self._last_closed_run = (float(started) if started is not None else float(at), float(at))
        run = max(0.0, float(at) - started) if started is not None else 0.0
        opened = self._run_opened_at
        self._run_opened_at = None
        self._last_run_seconds = max(0.0, float(at) - opened) if opened is not None else run
        self._caller_speech_call += self._last_run_seconds
        self._longest_run_call = max(self._longest_run_call, self._last_run_seconds)
        if self._armed_at is None:
            if self._run_blocked_by:
                # The agent was already busy -- or could not be observed --
                # when this run opened, so the run did not ask the model for
                # anything and no output edge is owed for it.
                self._log(outcome=self._run_blocked_by, at=at)
                return
            # This run's END is what arms the window, so nothing the run spent
            # belongs to it: the window did not exist while they were talking.
            self._arming_run_seconds = self._last_run_seconds
            self._arm(at)
            return
        if started is not None:
            self._caller_seconds += run
            self._runs_since_armed += 1
            self._longest_run_since_armed = max(self._longest_run_since_armed, run)

    def _pump_audible(self, at: float) -> None:
        """Audio the ROOM heard since the last look is agent OUTPUT.

        This is the whole of the round-3 correction, and the reason the gate
        it replaces is gone rather than fixed. A hold says "wait a little
        longer"; it leaves the window armed, and a reply that resumes after a
        pause longer than the window arrives with no new SDK state edge
        behind it, so nothing ever clears what the pause let arm. All three
        review engines drove the same sequence: a two-second gap inside a
        streamed reply, a caller backchannel in the gap, the reply resuming --
        and the call ended mid-playback, with the room still hearing the
        agent.

        An advance is not a hint that the agent might be busy. It is the
        caller hearing the agent, which is the one thing this rule exists to
        find missing, so it CLEARS. Two facts make that safe rather than a
        way to keep a dead call alive: the sidecar's agent edge advances only
        on frames above a quiet threshold, and the acknowledgement and
        tool-wait fillers are published by a ``BackgroundAudioPlayer``, whose
        track name is one SDK-level constant the sidecar skips. A looping
        filler at a caller with a dead model moves nothing here.

        Timestamped at the EDGE, not at ``at``: what the run-open window
        measures is when the room last heard something, and reading it at
        poll time would make that window depend on the poll cadence.
        """

        try:
            edge = self._agent_audible_at()
        except Exception:  # noqa: BLE001 - a reader cannot decide by failing
            return
        if not isinstance(edge, (int, float)) or isinstance(edge, bool):
            return
        moment = float(edge)
        if moment != moment:  # NaN
            return
        if self._audible_seen is not None and moment <= self._audible_seen:
            return
        self._audible_seen = moment
        self.note_agent_output(OUTPUT_AUDIO, at=moment)

    def _entry_block(self, at: float) -> str:
        """Why a caller run opening at ``at`` may not arm, or ``""``.

        Read once, on the run's opening edge. The order is the order of the
        decision below, so one call reported two ways is impossible.
        """

        # A read that RAISES is unobserved too, and that is the deliberate
        # choice: this gate is the one whose safe answer is "I cannot say",
        # and the recorder behind it is a real object, not a ``getattr``.
        if not _quietly_true(self._agent_audio_observed):
            return SKIP_AGENT_AUDIO_UNOBSERVED
        if (
            self._last_output_at is not None
            and at - self._last_output_at <= AGENT_AUDIBLE_WINDOW_SECONDS
        ):
            # Output within the window: either a reply handed to the local
            # source that the room has not heard yet -- the state edge leads
            # the room by about a seventh of a second -- or audio the room
            # HAS just heard, pumped in above. One check covers both, because
            # both are the same statement: the agent produced something, and
            # a caller run opening on top of it asked the model for nothing.
            return SKIP_RUN_OVERLAPPED_OUTPUT
        if _quietly_true(self._tool_in_flight) or _quietly_true(
            self._workflow_turn_pending
        ):
            return SKIP_RUN_OVERLAPPED_OUTPUT
        return ""

    def note_reply_withheld(self, *, at: float, held_since: float) -> None:
        """A synthesized reply was held back for the far end and never played.

        The far-end floor hold keeps a ready reply off the line while the
        caller holds the floor, and cancels it when they go on talking. The
        model answered -- there was audio -- so the question that armed this
        window has been answered, and the window clears. Counting only replies
        that PLAYED hung up on a caller who talked in ~2 s bursts, every reply
        held for their floor and cancelled when they resumed (release gate,
        2026-09-25): no speaking edge, no audible audio, no assistant item.

        Deliberately NOT output: it sets no run-overlap window, so the
        caller's run that cancelled the reply still arms a window when it
        ends and puts the model on the clock for it. A dead model produces no
        audio, so nothing is ever held and the bound still ends the call; a
        cached filler plays on a track the hold never sees.
        """

        if self._armed_at is None:
            return
        armed_at = self._armed_at
        self._log(outcome=OUTCOME_REPLY_WITHHELD, at=at)
        self._clear()
        # The held reply answered the latest question asked before its audio
        # was READY (``held_since``, when the hold began). A caller run that
        # closed after that -- a newer question the armed window absorbed
        # rather than re-arming for -- is still owed: the cancellation can
        # land after its falling edge (a late final transcript cancels the
        # held reply). Put it back on the clock from its own end; a run still
        # open arms itself when it closes.
        run = self._last_closed_run
        if (
            run is not None
            and not self._caller_speaking
            and run[1] > armed_at
            and run[1] > float(held_since)
        ):
            self._arming_run_seconds = self._last_run_seconds
            self._rearm(run[1])
            self._log(outcome="armed", at=at)

    def note_agent_output(self, kind: str, *, at: float) -> None:
        """A reply reaching the line or a tool starting: the clock clears.

        The agent ending the call needs no edge of its own: ``end_call``
        closes semantic admission, which is one of the gates above, and it is
        dispatched as a tool, which is one of these.
        """

        # FORWARD only. The audible edge is a past moment, so the first sight
        # of an edge that predates a reply already handed to the local source
        # would otherwise move this backwards and reopen the run-open window
        # the SDK edge had just closed -- caught by the test for exactly that
        # race, which then armed on a healthy reply in flight.
        if self._last_output_at is None or float(at) > self._last_output_at:
            self._last_output_at = float(at)
        # Output DURING a caller run disqualifies that run: the reply it
        # would have owed has already been produced.
        if self._caller_speaking:
            self._run_blocked_by = SKIP_RUN_OVERLAPPED_OUTPUT
        if self._armed_at is None:
            return
        # Logged BEFORE the clock and the flags are wiped. The edge that saved
        # this call is the one worth counting, and a line that reported
        # `armed: false` with no duration described nothing (Grok round 1, P2).
        self._log(outcome=OUTCOME_CLEARED, at=at, output=str(kind))
        self._clear()

    def note_caller_transcript(self, *, at: float) -> None:
        """A non-blank transcript of the caller's words arrived.

        Interim or final, on any engine. Read only by the shadow bound: a
        caller whose words are being transcribed is talking to something that
        is still listening, whatever the budget above says.
        """

        self._caller_transcripts += 1
        # Set whatever the window is doing: opening a window (``_rearm``)
        # clears it, so words from before the window never count against it.
        self._window_transcribed = True

    def note_provider_error(self, *, at: float, component: str) -> None:
        """A component error. Corroboration only when the component is the MODEL.

        ``component`` is required, with no default, because the answer to
        "which component failed?" is the whole of this method's contract: a
        call site that cannot say must not be able to corroborate by omission.
        A synthesis fault beside an assistant response is the case the
        stand-down below is FOR, and counting its error here would end that
        call as ``model_unresponsive`` -- the exact outcome this module's own
        docstring promises it will not produce.
        """

        if self._armed_at is None:
            return
        if str(component) not in MODEL_ERROR_COMPONENTS:
            return
        self._provider_error = True

    def note_model_response(self, *, at: float) -> None:
        """The engine produced a response item. The model was not the silent part.

        The COUNT and the moment are recorded whatever the window is doing;
        only the window's own flag is gated on being armed. That separation is
        the whole of this method's second job. ``model_response_seen`` answers
        "did the window that is deciding see an answer?", and on a healthy
        call the answer is no on every engine -- audible audio clears the
        window at the first frame the room hears, while the assistant item
        arrives only when the reply's playout has finished -- so the flag is
        false at every log line of every healthy call by construction. Read
        as "the engine never answered", that shape cost a production defect
        report against this method on voice-to-voice engines (2026-09-23).
        The totals are what actually answer that question,
        and they are in the same log line as the flag so the two cannot be
        confused again.

        The moment is recorded as the caller gives it. There is one producer
        -- the session handler for an assistant conversation item -- so there
        is no second clock to disagree with and nothing to guard against.
        """

        self._model_responses += 1
        self._last_model_response_at = float(at)
        if self._armed_at is not None:
            self._model_response = True

    # -- the decision ---------------------------------------------------

    def poll(self, now: float) -> str:
        """Decide, act, and name the branch. Safe to call on any cadence."""

        # First, and before anything decides: audio the room heard since the
        # last look clears the window outright.
        self._pump_audible(now)
        outcome = self._decide(now)
        # A single nudge on a proven unanswered turn, after the SAME caller,
        # tool, observation and admission gates as the watchdog. This neither
        # authorizes a hang-up nor resets/extends its original deadline.
        if (outcome == SKIP_WITHIN_BOUND and not self._model_response
                and not self._recovery_used and self._recover_reply is not None
                and self._owed(now) >= min(10.0, self._seconds / 2)):
            try:
                self._recovery_used = bool(self._recover_reply())
            except Exception:
                # A failed ask must not turn into a retry loop or disable the
                # watchdog. The runtime reports its own failure details.
                self._recovery_used = True
        if outcome == ENDED:
            # The compare-and-set seam every other ending uses. First reason
            # wins: a call something else already ended is not re-stamped, and
            # this module does not learn about that from a flag of its own.
            if not self._end_call():
                # NOT latched. False is the arbiter's "something already
                # stamped an ending", and a stamp that never becomes a
                # teardown is RELEASED so the next tick can try again -- every
                # other worker ending simply calls it again. Latching here
                # would make this the one rule that gives up on a released
                # stamp and leaves the dead-air call up and billing, which is
                # the defect it exists to close (Grok round 1, P1). The
                # arbiter is compare-and-set, so asking again is free.
                self._log(outcome=SKIP_ALREADY_ENDING, at=now)
                return SKIP_ALREADY_ENDING
            self._decided = True
        self._log(outcome=outcome, at=now)
        return outcome

    def _decide(self, now: float) -> str:
        if not self.enabled:
            return SKIP_DISABLED
        if self._decided or self._is_ending():
            return SKIP_ALREADY_ENDING
        if self._armed_at is None:
            return SKIP_NOT_ARMED
        # A call waiting in a phone menu or a hold queue owes nobody an answer:
        # the agent is meant to be silent, and the "caller" the window armed on
        # was a recording. Read first, before the observation holds, and
        # re-clocked like them so the bound starts afresh when the call
        # returns to conversation.
        if _quietly_true(self._navigation_suspended):
            self._rearm(now)
            return SKIP_NAVIGATION
        # Each hold RESETS rather than merely declining, so the model gets a
        # whole bound from the moment it stops being entitled to the silence.
        if not _quietly_true(self._agent_audio_observed):
            # Uncapped, and the EVIDENCE is kept across it. A cap would let an
            # unobserved call be ended on the absence of a measurement rather
            # than on a measurement of absence; and clearing the flags would
            # lose an assistant response the engine had already produced, so
            # the next fully observed bound would end as `model_unresponsive`
            # the very call the stand-down below had refused (Grok round 3,
            # P2). A gap in the observation is not the model owing a new
            # answer, which is what the tool and workflow holds below mean.
            self._rearm(now, keep_evidence=True)
            return SKIP_AGENT_AUDIO_UNOBSERVED
        for held, outcome in (
            (self._tool_in_flight, SKIP_TOOL_IN_FLIGHT),
            (self._workflow_turn_pending, SKIP_WORKFLOW_TURN_PENDING),
        ):
            if _quietly_true(held):
                self._rearm(now)
                return outcome
        # After the holds, which re-clock it, and before the caller-speaking
        # decline, which the shadow's no-pause cap must be able to see past.
        self._shadow(now)
        if self._caller_speaking:
            # Declines WITHOUT re-clocking, unlike the holds above. The budget
            # is already frozen for the duration of this run, so there is
            # nothing to give back -- and hanging up on someone mid-word is
            # the one outcome a rule about dead air must never produce.
            return SKIP_CALLER_SPEAKING
        if not _quietly_true(self._admits, default=True):
            # Admission is closed: nothing released now may start a turn, so
            # there is no reply left to be owed and no call left to save.
            self._rearm(now)
            return SKIP_ADMISSION_CLOSED
        if _quietly_true(self._deferred):
            # Before the within-bound check, which is also where the recovery
            # nudge is asked: a deferred rule neither ends nor nudges, and keeps
            # its clock (see ``SKIP_DEFERRED``).
            return SKIP_DEFERRED
        if self._owed(now) < self._seconds:
            return SKIP_WITHIN_BOUND
        if self._model_response and not self._provider_error:
            # The engine answered and the caller heard nothing: a synthesis or
            # playout fault, not a lost model. Stand down under its own name
            # rather than stamping the wrong component.
            #
            # CLEARED, not re-clocked. Re-clocking kept an armed window with
            # the response evidence erased, so the very next bound ended the
            # same call as `model_unresponsive` with no new caller turn and no
            # provider error -- the ending this branch had just refused,
            # delayed by one bound rather than refused (Codex round 1, P1-by-
            # verification). The question this window asked has been answered;
            # the NEXT caller turn asks it again.
            self._clear()
            return SKIP_MODEL_ANSWERED
        return ENDED

    def _shadow(self, now: float) -> None:
        """Record, once per window, that the shadow bound would end the call.

        Never ends anything and never changes the decision it sits in.
        """

        if (
            self._shadow_reported
            or self._armed_at is None
            or self._window_transcribed
            or self._model_response
            or not _quietly_true(self._admits, default=True)
        ):
            return
        armed_for = float(now) - self._armed_at
        if armed_for < SHADOW_BOUND_SECONDS:
            return
        if not self._caller_speaking:
            trigger = SHADOW_AT_CALLER_PAUSE
        elif armed_for >= SHADOW_BOUND_SECONDS * SHADOW_NO_PAUSE_FACTOR:
            trigger = SHADOW_AT_NO_PAUSE_CAP
        else:
            return
        self._shadow_reported = True
        self._shadow_would_end += 1
        if self._shadow_first_armed_ms is None:
            self._shadow_first_armed_ms = int(round(armed_for * 1000))
        self._log(outcome=SHADOW_WOULD_END, at=now, shadow_trigger=trigger)

    @property
    def shadow_stats(self) -> dict[str, int]:
        """Bounded per-call scalars for the interruption summary."""

        stats = {
            "model_unresponsive_shadow_would_end": self._shadow_would_end,
            "caller_transcripts_seen": self._caller_transcripts,
        }
        if self._shadow_first_armed_ms is not None:
            stats["model_unresponsive_shadow_armed_ms"] = self._shadow_first_armed_ms
        return stats

    def _owed(self, now: float) -> float:
        """Seconds the agent has owed a reply, excluding caller speech.

        The budget is what the AGENT owes, so the floor being occupied does
        not spend it. An open run is subtracted here; closed ones were
        subtracted as they closed.
        """

        if self._armed_at is None:
            return 0.0
        open_run = (
            max(0.0, float(now) - self._speaking_since)
            if self._caller_speaking and self._speaking_since is not None
            else 0.0
        )
        return max(0.0, float(now) - self._armed_at - self._caller_seconds - open_run)

    def _continuity(self, at: float) -> dict[str, Any]:
        """How continuous the caller side has been since the window armed.

        A run still open counts with its length so far, so a caller tune that
        never pauses shows as one run covering the window rather than as none.
        """

        if self._armed_at is None:
            return {}
        armed_for = max(0.0, float(at) - self._armed_at)
        runs = self._runs_since_armed
        longest = self._longest_run_since_armed
        speech = self._caller_seconds
        if self._caller_speaking and self._speaking_since is not None:
            open_run = max(0.0, float(at) - self._speaking_since)
            runs += 1
            longest = max(longest, open_run)
            speech += open_run
        details: dict[str, Any] = {
            "caller_runs_since_armed": runs,
            "caller_longest_run_seconds_since_armed": round(longest, 3),
            "caller_speech_share_since_armed": round(speech / armed_for, 3) if armed_for > 0 else 0.0,
        }
        if self._arming_run_seconds is not None:
            details["arming_run_seconds"] = round(self._arming_run_seconds, 3)
        return details

    def _call_totals(self, at: float) -> dict[str, float]:
        """The caller's speech over the whole call, an open run included."""

        speech, longest = self._caller_speech_call, self._longest_run_call
        if self._caller_speaking and self._run_opened_at is not None:
            open_run = max(0.0, float(at) - self._run_opened_at)
            speech += open_run
            longest = max(longest, open_run)
        return {
            "caller_speech_seconds_call": round(speech, 3),
            "caller_longest_run_seconds_call": round(longest, 3),
        }

    def _arm(self, at: float) -> None:
        self._rearm(at)
        self._log(outcome="armed", at=at)

    def _rearm(self, at: float, *, keep_evidence: bool = False) -> None:
        """Open a fresh window at ``at``.

        The flags go with the clock by DEFAULT. A window that kept them would
        answer the NEXT window's question with the last one's evidence: a
        response the engine produced before it died would stand the rule down
        forever, and an error from a component that has since recovered would
        keep corroborating an ending nobody should reach.

        ``keep_evidence`` is the one exception, and it is a different kind of
        re-clock. A tool or a workflow turn means the model owes a NEW answer,
        so the old one is spent. An observation gap means nobody was watching;
        nothing about the model changed, and discarding what it had already
        said would turn the next fully observed bound into the ending this
        module's own stand-down had just refused.
        """

        self._armed_at = float(at)
        self._caller_seconds = 0.0
        self._runs_since_armed = 0
        self._longest_run_since_armed = 0.0
        if self._caller_speaking and self._speaking_since is not None:
            # A run that was ALREADY open when this window opened. Its earlier
            # seconds belong to the window that has just been replaced, and
            # leaving the start where it was would subtract them from a
            # budget that had not started yet -- a hold re-clocking during a
            # long caller run credited the whole run to the new window and
            # pushed the bound out by the length of the run (Codex round 2,
            # P2). Only speech from here is this window's to subtract.
            self._speaking_since = max(self._speaking_since, float(at))
        if not keep_evidence:
            self._provider_error = False
            self._model_response = False
            self._window_transcribed = False
            self._shadow_reported = False

    def _clear(self) -> None:
        self._armed_at = None
        self._caller_seconds = 0.0
        self._provider_error = False
        self._model_response = False

    def _log(
        self, *, outcome: str, at: float, output: str = "", shadow_trigger: str = ""
    ) -> None:
        """Emit, never raising into the caller.

        Fed from the session's own handlers, which several other rules run
        from; a raise here would take them with it.
        """

        details: dict[str, Any] = {
            "outcome": outcome,
            "bound_seconds": round(self._seconds, 3),
            "armed": self._armed_at is not None,
            "provider_error_seen": self._provider_error,
            # Per WINDOW: an answer this armed window saw. False on every
            # healthy call, on every engine -- see ``note_model_response``.
            "model_response_seen": self._model_response,
            # Per CALL: what the engine has produced since the call began.
            # Zero here, and only zero here, is an engine that has never
            # answered.
            "model_responses_total": self._model_responses,
        }
        if self._last_model_response_at is not None:
            details["last_model_response_seconds_ago"] = round(
                max(0.0, float(at) - self._last_model_response_at), 3
            )
        if self._armed_at is not None:
            details["armed_for_seconds"] = round(max(0.0, float(at) - self._armed_at), 3)
            # What the bound is actually compared against: the armed interval
            # minus the caller's own speech. A run table that printed only the
            # first would read as a rule that fired late.
            details["owed_seconds"] = round(self._owed(at), 3)
            details["caller_speech_seconds"] = round(self._caller_seconds, 3)
            details.update(self._continuity(at))
        details.update(self._call_totals(at))
        if output:
            details["output"] = output
        if shadow_trigger:
            details["shadow_trigger"] = shadow_trigger
            details["shadow_bound_seconds"] = SHADOW_BOUND_SECONDS
        try:
            self._emit(DECISION_EVENT, **details)
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass


def _quietly_true(predicate: Callable[[], bool], *, default: bool = False) -> bool:
    """Read an injected gate; a gate that raises answers its safe default.

    A predicate that blows up must not be able to hang up on a caller, and
    must not be able to keep a dead call alive forever either. Each caller
    picks which of those two its own default is.
    """

    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001 - a gate cannot decide by failing
        return default


__all__ = [
    "AGENT_AUDIBLE_WINDOW_SECONDS",
    "MODEL_ERROR_COMPONENTS",
    "SKIP_AGENT_AUDIO_UNOBSERVED",
    "SKIP_DEFERRED",
    "SKIP_NAVIGATION",
    "agent_audio_audible",
    "agent_audio_observed",
    "BOUND_SECONDS_BOUNDS",
    "DECISION_EVENT",
    "DEFAULT_BOUND_SECONDS",
    "ENDED",
    "ModelUnresponsiveRule",
    "OUTPUT_REPLY",
    "OUTPUT_AUDIO",
    "OUTPUT_TOOL",
    "OUTCOME_REPLY_WITHHELD",
    "SKIP_ADMISSION_CLOSED",
    "OUTCOME_CLEARED",
    "SKIP_ALREADY_ENDING",
    "SKIP_RUN_OVERLAPPED_OUTPUT",
    "SKIP_CALLER_SPEAKING",
    "SKIP_DISABLED",
    "SKIP_MODEL_ANSWERED",
    "SKIP_NOT_ARMED",
    "SKIP_TOOL_IN_FLIGHT",
    "SKIP_WITHIN_BOUND",
    "SKIP_WORKFLOW_TURN_PENDING",
    "SHADOW_AT_CALLER_PAUSE",
    "SHADOW_AT_NO_PAUSE_CAP",
    "SHADOW_BOUND_SECONDS",
    "SHADOW_NO_PAUSE_FACTOR",
    "SHADOW_WOULD_END",
    "bounded_seconds",
    "model_unresponsive_seconds",
]

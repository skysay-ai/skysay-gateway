# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""What a transfer does to the two audio legs, and who may stop it.

The control plane has already authorized the destination and recorded the one
attempt. This module decides who hears what, and in which order, and owns the
transfer from the hold line to its one reported outcome. A test passes a fake
media port. The live worker passes the room.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Protocol

HOLD_LINE = "One moment, I'll try them now."
HUMAN_IDENTITY = "skysay-transfer"
#: The transfer's phases, in order. Nobody is connected before
#: ``BRIDGE_COMMITTED``; from it on the call is the two people's.
IDLE, HOLD, RINGING, BRIDGE_COMMITTED, TERMINAL = "idle", "hold", "ringing", "bridge_committed", "terminal"
#: Why a transfer stopped before anyone was connected: the caller withdrew it
#: (``cancel_transfer``), or the call ended.
CALLER_CANCELLED = "caller_cancelled"
CALLER_HANGUP = "caller_hangup"
#: What a request to cancel the transfer answers. ``no_caller_request``: the
#: caller has not started speaking since the transfer was placed, so a
#: withdrawal cannot have come from them, and the ring goes on.
CANCELLED, NOT_IN_PROGRESS, ALREADY_CONNECTED = "cancelled", "not_in_progress", "already_connected"
NO_CALLER_REQUEST = "no_caller_request"
#: How often a stopped leg's removal is retried while its dial is pending: from
#: the first interval, doubling to the ceiling. The SIP service joins the leg's
#: participant to the room only just before its INVITE, and runs that request
#: detached from the worker, so a participant can appear at any moment until
#: the dial request itself ends (at most ~30 s: a few dozen tries).
LEG_REMOVAL_INTERVAL_SECONDS = 0.25
LEG_REMOVAL_MAX_INTERVAL_SECONDS = 1.0
#: How long a cancel or the call's teardown waits for the transfer to settle.
#: Only the waiting is bounded: the controller's own task goes on removing the
#: leg until its dial has ended, whoever stopped waiting.
SETTLE_SECONDS = 8.0
#: How long a voice-to-voice line may take, from the ask to its last frame.
#: The ask itself can wait 10 s for the model to start (LiveKit's duplex
#: adapter), and a transfer line is a few seconds long. The dial goes ahead
#: either way: a line the model never spoke must not keep the caller from the
#: person they asked for.
MODEL_LINE_TIMEOUT_SECONDS = 15.0


def model_line_instruction(listener: str, text: str) -> str:
    """Ask a voice-to-voice model to speak one fixed line, word for word."""
    person = "the person who just answered the phone" if listener == "human" else "the caller"
    return (
        f"Say exactly this sentence to {person}, word for word, and nothing else. "
        f"Do not call any tool. The sentence: {text}"
    )


class TransferMedia(Protocol):
    def call_ended(self) -> bool: ...

    async def speak(self, who: str, text: str) -> None: ...

    async def set_heard(self, listener: str, source: str, heard: bool) -> None: ...

    async def dial(self, plan: dict[str, Any]) -> str: ...

    async def remove_human(self) -> bool | None:
        """``False`` when there was no leg to remove; anything else removed it."""
        ...

    async def bridge(self) -> None: ...


def classify_dial_failure(message: str) -> str:
    """Map a carrier or SIP failure onto the outcomes the caller is told about."""
    text = str(message or "").lower()
    if "busy" in text or "486" in text:
        return "busy"
    if "rejected" in text or "603" in text or "403" in text:
        return "rejected"
    if "timeout" in text or "no answer" in text or "no_answer" in text or "408" in text or "480" in text:
        return "no_answer"
    if "caller" in text and "hang" in text:
        return "caller_hangup"
    return "failed"


class RoomMedia:
    """Drive a LiveKit room with the same calls the tests assert on a fake.

    ``subscribe(listener, heard)`` changes whether that person receives the
    agent's audio. It is the server ``update_subscriptions`` call, not the
    agent's own ``set_subscribed``, which only controls what the agent hears.
    """

    def __init__(
        self,
        session: Any,
        room: Any,
        dial: Any,
        subscribe: Any = None,
        remove: Any = None,
        *,
        speaks_text: bool,
        is_finished: Callable[[], bool],
        emit: Callable[..., None] | None = None,
        line_timeout_seconds: float = MODEL_LINE_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._room = room
        self._dial = dial
        self._subscribe = subscribe
        self._remove = remove
        # ``False`` on a voice-to-voice session: it has no TTS, and LiveKit
        # refuses ``say(text)`` there with a RuntimeError, so every line is
        # asked of the model instead. Required, because the old default was
        # the cascade's, and it broke every voice-to-voice transfer.
        self._speaks_text = bool(speaks_text)
        # The call's own "this call is over" signal. Closing the session cuts
        # the hold line without cancelling the tool, so the wait returns as if
        # the line had merely been interrupted; this is what says the caller
        # is gone before a person is dialled for nobody.
        self._is_finished = is_finished
        self._emit = emit
        self._line_timeout_seconds = float(line_timeout_seconds)

    def call_ended(self) -> bool:
        try:
            return bool(self._is_finished())
        except Exception:
            # Unknown is not "still here": do not dial a person for a call
            # whose state cannot be read.
            return True

    async def speak(self, who: str, text: str) -> None:
        if not text:
            return
        if self._speaks_text:
            handle = self._session.say(text, allow_interruptions=False)
            try:
                await handle
            except asyncio.CancelledError:
                # The transfer stopped: the line must not play on. LiveKit
                # shields the speech from its waiter.
                _stop(handle)
                raise
            return
        try:
            handle = self._session.generate_reply(
                instructions=model_line_instruction(who, text),
                tool_choice="none",
            )
        except Exception as exc:
            # LiveKit raises here only when the session is not running or is
            # closing: the call is ending, so the transfer stops here, as the
            # cascade's say() does, and nobody is dialled.
            self._report(who, "failed", type(exc).__name__)
            raise
        playout, error_type = await self._await_model_line(handle)
        self._report(who, playout, error_type)

    def _report(self, who: str, playout: str, error_type: str) -> None:
        if self._emit is not None:
            self._emit("transfer_line", listener=who, path="model", playout=playout, error_type=error_type)

    async def _await_model_line(self, handle: Any) -> tuple[str, str]:
        """Wait for the model's line, bounded. Never raises but on cancellation.

        Returns ``(playout, error_type)``. ``playout`` is ``completed``,
        ``interrupted``, ``timeout`` or ``failed``. ``error_type`` is an
        exception class name, never its message.
        """
        try:
            await asyncio.wait_for(handle.wait_for_playout(), timeout=self._line_timeout_seconds)
        except TimeoutError:
            # LiveKit shields the speech from its waiter. Left running, the
            # line could play over the ring or to the wrong person, and it
            # would hold back the tool result the model answers.
            _stop(handle)
            return "timeout", ""
        except asyncio.CancelledError:
            _stop(handle)
            raise
        except Exception as exc:
            _stop(handle)
            return "failed", type(exc).__name__
        # A failed generation completes its handle without raising. LiveKit
        # keeps the error on the handle.
        try:
            probe = getattr(handle, "exception", None)
            error = probe() if callable(probe) else None
        except Exception as exc:
            error = exc
        if error is not None:
            return "failed", type(error).__name__
        if bool(getattr(handle, "interrupted", False)):
            return "interrupted", ""
        return "completed", ""

    async def set_heard(self, listener: str, source: str, heard: bool) -> None:
        if source == "agent" and self._subscribe is not None:
            await self._subscribe(listener, bool(heard))

    async def dial(self, plan: dict[str, Any]) -> str:
        return await self._dial(self._room, plan)

    async def remove_human(self) -> bool | None:
        if self._remove is None:
            return None
        return await self._remove()

    async def bridge(self) -> None:
        # The two people hear each other through the room. The AI agent stays
        # unsubscribed so it cannot talk over them.
        await self.set_heard("caller", "agent", False)
        await self.set_heard("human", "agent", False)


class AuthoredLinesRoomMedia(RoomMedia):
    """A workflow transfer step's two legs: the same dial, bridge and removal.

    A workflow speaks only its own authored lines, as steps around the
    transfer step (a "connecting you" line before it, its fallback after), so
    the transfer itself says nothing to anyone: no platform hold or resume
    line, and no handoff summary, which a workflow never has.
    """

    def __init__(self, room: Any, dial: Any, subscribe: Any, remove: Any, *,
                 is_finished: Callable[[], bool], on_connected: Callable[[], None],
                 emit: Callable[..., None] | None = None) -> None:
        super().__init__(None, room, dial, subscribe, remove, speaks_text=True,
                         is_finished=is_finished, emit=emit)
        self._on_connected = on_connected

    async def speak(self, who: str, text: str) -> None:
        return None

    async def bridge(self) -> None:
        # The person answered: the call is theirs from this instant, before
        # any further await can fail. Muting the agent for both is then best
        # effort -- a workflow that transferred speaks no further line.
        self._on_connected()
        for listener in ("caller", "human"):
            try:
                await self.set_heard(listener, "agent", False)
            except Exception as exc:  # noqa: BLE001 - the two people are already connected
                if self._emit is not None:
                    self._emit("transfer_mute_failed", listener=listener, error_type=type(exc).__name__)


def _stop(handle: Any) -> None:
    """Cut a model line nobody should hear any more. Best effort."""
    try:
        handle.interrupt(force=True)
    except Exception:
        pass


class TransferController:
    """A call's one transfer, from the hold line to its one reported outcome.

    Loop-affine: every method runs on the call's own event loop. The transfer
    runs in this controller's task, not the tool's, so it is cleaned up and
    reported exactly once even when the coroutine that started it is
    cancelled. Its speech is awaited by that task, so stopping the task cuts
    the line being spoken.

    Until ``BRIDGE_COMMITTED`` a stop wins: the leg is removed, nobody is
    bridged and the outcome is the stop's reason. The commit is one
    synchronous step, so a cancel and an answer cannot both win.
    """

    def __init__(
        self,
        media: TransferMedia,
        *,
        report: Callable[[str], Awaitable[None]] | None = None,
        ended: Callable[[], Awaitable[Any]] | None = None,
        emit: Callable[..., None] | None = None,
        removal_interval_seconds: float = LEG_REMOVAL_INTERVAL_SECONDS,
        removal_max_interval_seconds: float = LEG_REMOVAL_MAX_INTERVAL_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
    ) -> None:
        self._media = media
        self._report = report
        # Returns once the call is over (the room gone or the session
        # finished): the caller leaving wins over everything before the bridge.
        self._ended = ended
        self._emit = emit
        self._removal_interval_seconds = float(removal_interval_seconds)
        self._removal_max_interval_seconds = max(float(removal_max_interval_seconds), self._removal_interval_seconds)
        self._settle_seconds = float(settle_seconds)
        self.phase = IDLE
        self._task: asyncio.Task | None = None
        self._started = False
        self._dial: asyncio.Task | None = None
        self._dial_outcome: str | None = None
        self._reason: str | None = None
        self._settling = False
        self._settled = asyncio.Event()
        self._removed = False
        self._caller_muted = False
        self._outcome: str | None = None
        # The caller-spoke precondition of a withdrawal, on one monotonic
        # clock: when the transfer was placed, and when the caller's speech
        # last STARTED (a local voice-activity edge on the caller's own input,
        # never a transcript and never the agent's own audio).
        self._placed_at: float | None = None
        self._caller_speech_started_at: float | None = None
        self._caller_speech_feed = ""

    def note_caller_speech_started(self, feed: str = "") -> None:
        """The caller's voice-activity detector saw their speech start.

        ``feed`` names the detector (``user_state`` or ``native_floor``), for
        the cancel decision's evidence only.
        """
        self._caller_speech_started_at = time.monotonic()
        self._caller_speech_feed = feed

    async def run(self, plan: dict[str, Any], *, on_dialing: Callable[[], Awaitable[Any]] | None = None) -> str:
        """Play the hold line, dial once, then either bridge or bring the AI agent back.

        ``plan`` is the worker dial block; ``mode`` is ``warm`` or ``cold``.
        ``on_dialing`` runs once the dial is placed. Returns the outcome:
        answered, no_answer, busy, rejected, failed, caller_hangup or
        caller_cancelled. A failure while the call is live is reported as
        ``failed`` and raised; once the call has ended it is ``caller_hangup``.
        """
        if self._task is not None:
            raise RuntimeError("this call's transfer has already been placed")
        self.phase = HOLD
        self._placed_at = time.monotonic()
        self._task = asyncio.create_task(self._drive(dict(plan), on_dialing))
        # Retrieved here even when nobody awaits it any more.
        self._task.add_done_callback(lambda task: task.cancelled() or task.exception())
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # The SDK cancels a running tool only when its session closes:
            # the call is ending. Nobody may be left ringing for it.
            if not self._task.done():
                self._stop(CALLER_HANGUP if self._call_ended() else "failed")
                await asyncio.wait({self._task}, timeout=self._settle_seconds)
            raise

    async def cancel(self, reason: str = CALLER_CANCELLED) -> str:
        """Stop the transfer before anyone is connected.

        Returns ``cancelled`` once the leg is gone, ``already_connected`` after
        the bridge, ``not_in_progress`` when there is nothing to stop, and
        ``no_caller_request`` when the caller has not started speaking since
        the transfer was placed: only they can withdraw it, and a model that
        cancels on its own (seen on the cascade's SIP rehearsal, calls
        call_PIrrNtTWDta8bh9b and call_9qb49LFez7Qeg) leaves it ringing.
        """
        decision = await self._decide_cancel(reason)
        if reason == CALLER_CANCELLED:
            # Every withdrawal request, honoured or not: structured evidence
            # only, never what anybody said.
            placed = self._placed_at
            self._note(
                "transfer_cancel",
                decision=decision,
                caller_spoke_since_placed=self._caller_spoke_since_placed(),
                since_placed_seconds=None if placed is None else round(time.monotonic() - placed, 3),
                speech_feed=self._caller_speech_feed,
            )
        return decision

    async def _decide_cancel(self, reason: str) -> str:
        if self.phase == BRIDGE_COMMITTED or self._outcome == "answered":
            return ALREADY_CONNECTED
        if self._stoppable(reason) and reason == CALLER_CANCELLED and not self._caller_spoke_since_placed():
            return NO_CALLER_REQUEST
        if not self._stop(reason) and self._reason != reason:
            return NOT_IN_PROGRESS
        try:
            await asyncio.wait_for(self._settled.wait(), timeout=self._settle_seconds)
        except TimeoutError:
            pass
        return CANCELLED

    async def cancel_and_join(self, reason: str = CALLER_HANGUP) -> None:
        """The call is ending: stop the transfer if it is still short of the
        bridge, and wait (bounded) for its cleanup and report. Never raises
        but on cancellation."""
        if self._task is None:
            return
        self._stop(reason)
        await asyncio.wait({self._task}, timeout=self._settle_seconds)

    def _caller_spoke_since_placed(self) -> bool:
        placed, spoke = self._placed_at, self._caller_speech_started_at
        return placed is not None and spoke is not None and spoke >= placed

    def _stoppable(self, reason: str) -> bool:
        """Whether a stop for ``reason`` would take effect now.

        The first reason wins. The caller can withdraw a transfer only while
        it may still connect; the call's end also cuts a resume line.
        """
        if self._task is None or self._reason is not None or self._settling:
            return False
        if self.phase in (IDLE, BRIDGE_COMMITTED, TERMINAL):
            return False
        return not (reason == CALLER_CANCELLED and self._dial_outcome not in (None, "answered"))

    def _stop(self, reason: str) -> bool:
        """Record why the transfer stops and stop its task. Returns whether it did."""
        if not self._stoppable(reason):
            return False
        self._reason = reason
        if self._started:
            self._task.cancel()
        # Else the stop is seen at the transfer's first step: a task cancelled
        # before it ever ran would never clean up or report.
        return True

    def _call_ended(self) -> bool:
        try:
            return bool(self._media.call_ended())
        except Exception:
            return True

    def _note(self, event: str, **details: Any) -> None:
        if self._emit is not None:
            try:
                self._emit(event, **details)
            except Exception:
                pass

    async def _watch_end(self) -> None:
        assert self._ended is not None
        await self._ended()
        self._stop(CALLER_HANGUP)

    async def _drive(self, plan: dict[str, Any], on_dialing: Callable[[], Awaitable[Any]] | None) -> str:
        self._started = True
        watch = asyncio.create_task(self._watch_end()) if self._ended is not None else None
        error: BaseException | None = None
        try:
            outcome = await self._steps(plan, on_dialing)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            outcome = self._reason or (CALLER_HANGUP if self._call_ended() else "failed")
        except Exception as exc:
            if self._call_ended():
                # The call is gone, and so is the room this step needed: an
                # expected ending, not an error to raise to the error tracker.
                outcome = CALLER_HANGUP
                self._note("transfer_error_after_call_end", transfer_step="transfer", error_type=type(exc).__name__)
            else:
                outcome, error = "failed", exc
        self._settling = True
        if watch is not None:
            watch.cancel()
        if outcome != "answered":
            try:
                await self._clean_up()
            except Exception as exc:
                if self._call_ended():
                    self._note("transfer_error_after_call_end", transfer_step="cleanup", error_type=type(exc).__name__)
                else:
                    # A leg that may still ring stays visible.
                    error = error or exc
        self.phase = TERMINAL
        self._outcome = outcome
        self._settled.set()
        # The one record of how the transfer ended, before its (bounded) report.
        self._note("transfer_finished", outcome=outcome, stopped_by=self._reason or "")
        if self._report is not None:
            try:
                await self._report(outcome)
            except Exception as exc:  # noqa: BLE001 - the report annotates the attempt row only
                self._note("transfer_outcome_not_recorded", error_type=type(exc).__name__)
        if error is not None:
            raise error
        return outcome

    async def _steps(self, plan: dict[str, Any], on_dialing: Callable[[], Awaitable[Any]] | None) -> str:
        media = self._media
        hold = str(plan.get("hold_line") or HOLD_LINE)
        resume = str(plan.get("resume_line") or "")
        summary = str(plan.get("handoff_summary") or "")
        mode = str(plan.get("mode") or "warm")
        if self._reason is not None:
            # Stopped before this task first ran.
            raise asyncio.CancelledError()
        await media.speak("caller", hold)
        if self._call_ended():
            # The caller hung up during the hold line. Nobody is dialled.
            return CALLER_HANGUP
        self.phase = RINGING
        self._dial = asyncio.create_task(media.dial(plan))
        if on_dialing is not None:
            try:
                await on_dialing()
            except Exception as exc:  # noqa: BLE001 - the dial goes on without the notice
                self._note("transfer_dialing_notice_failed", error_type=type(exc).__name__)
        # ``wait`` leaves the dial running when this task is stopped: a
        # cancelled leg is removed first, and the dial joined after.
        await asyncio.wait({self._dial})
        outcome = self._dial_outcome = self._dial.result()
        if self._call_ended():
            return CALLER_HANGUP
        if outcome != "answered":
            await self._remove_leg()
            if outcome != CALLER_HANGUP and resume and not self._call_ended():
                await media.speak("caller", resume)
            return outcome
        if mode != "cold" and summary:
            # The caller must not hear the introduction. Drop them from the
            # agent's voice for that sentence, then connect the two people.
            self._caller_muted = True
            await media.set_heard("caller", "agent", False)
            if self._call_ended():
                return CALLER_HANGUP
            await media.speak("human", summary)
            if self._call_ended():
                return CALLER_HANGUP
            await media.set_heard("human", "agent", False)
            if self._call_ended():
                return CALLER_HANGUP
        # The commit: no await since the last check, so no stop lands between.
        self.phase = BRIDGE_COMMITTED
        try:
            await media.bridge()
        except Exception as exc:  # noqa: BLE001 - the two people are already connected
            self._note("transfer_mute_failed", error_type=type(exc).__name__)
        return "answered"

    async def _clean_up(self) -> None:
        """Leave nobody behind: the leg removed, its dial joined, and a caller
        dropped from the agent's voice for a summary that never ended in a
        bridge able to hear the agent again. Every step is attempted; the
        first failure is raised once all of them have run."""
        errors: list[Exception] = []
        if self._dial is not None and not self._removed:
            try:
                await self._remove_leg()
            except Exception as exc:  # noqa: BLE001 - raised below, after the rest of the cleanup
                errors.append(exc)
        if self._dial is not None:
            if not self._dial.done():
                self._dial.cancel()
            await asyncio.gather(self._dial, return_exceptions=True)
        if self._caller_muted and not self._call_ended():
            try:
                await self._media.set_heard("caller", "agent", True)
            except Exception as exc:  # noqa: BLE001 - raised below
                errors.append(exc)
        if errors:
            raise errors[0]

    def _dial_pending(self) -> bool:
        return self._dial is not None and not self._dial.done()

    async def _remove_leg(self) -> None:
        """Take the leg out of the room now, and keep at it while its dial is
        pending. Removing the participant is what sends the SIP CANCEL:
        cancelling the dial request alone leaves the phone ringing, and the
        SIP service can still put the participant in the room until that
        request ends (its own timeout and the trunk's ringing limit bound it).
        While the dial is pending, "not found" means "not yet" and an error
        is retried too; the try made after the dial has ended is the last
        one. The call's end changes none of this: the room outlives it until
        the worker deletes it, and a late join rings until then. It changes
        only how a last error is reported (quietly, by the caller)."""
        started = time.monotonic()
        tries = 0
        result = "error"
        interval = self._removal_interval_seconds
        try:
            while True:
                pending = self._dial_pending()
                tries += 1
                try:
                    removed = await self._media.remove_human()
                except Exception:
                    if not pending:
                        if self._call_ended():
                            result = "call_ended"
                        raise
                    removed = False
                if removed is not False:
                    self._removed = True
                    result = "removed"
                    return
                if not pending:
                    # Nobody there, and no dial left that could still put them there.
                    result = "not_found_after_settle"
                    return
                # Not in the room yet: try again after a growing pause, or as
                # soon as the dial ends (a person who answered just then is in
                # the room).
                assert self._dial is not None
                await asyncio.wait({self._dial}, timeout=interval)
                interval = min(interval * 2, self._removal_max_interval_seconds)
        finally:
            self._note(
                "transfer_leg_removal",
                attempt_count=tries,
                removal_seconds=round(time.monotonic() - started, 3),
                removal_result=result,
            )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Audible feedback while a slow tool call runs.

A tool call is the one wait the model cannot talk through: between "let me
check" and the answer the agent is silent for as long as the customer's
webhook, the knowledge search or the transfer takes.  Measured on production
the median tool finishes in about 200 ms, but the slow tail runs to several
seconds, and dead air of that length reads as a dropped call.

This module covers that wait in one of two ways the customer chooses per agent
(``voice_polish.tool_wait`` on the voice stack, off unless set):

* ``sound`` -- a quiet looping keyboard-typing bed on the background track,
  started once the tool has been running for ``delay_ms`` and cut the instant
  the tool finishes.  The industry convention (ElevenLabs, Vapi) for the same
  wait; wordless, so it is language-independent and can never be mistaken for
  an answer.
* ``acknowledge`` -- one short reviewed phrase ("One moment while I check.")
  in the agent's own voice, synthesized once per call like the acknowledgement
  clips it shares a player with.

Both ride the dedicated filler ``BackgroundAudioPlayer`` that acknowledgements
use, never the session's speech queue: they start when asked and are cut with
the same queue flush, so nothing here changes the chat context, the transcript
or the agent's state machine.  The evidence bundle records the mode and every
start/cut so a reviewer knows what the caller heard.

Rules, in the order they matter:

* Never over anyone.  Feedback arms only while the caller and the agent are
  both silent and is cut the moment either starts speaking.  Words the agent
  says while the tool is still running ("let me look that up") only postpone
  it: the wait starts again when they end.  An acknowledgement that is still
  playing when the delay expires is waited out, never mixed under.
* One feedback per wait.  Parallel tool calls share one window (first start to
  last end); the caller hears one bed or one phrase, not one per tool.
* A wait the caller interrupts stays quiet.  Caller speech cancels the pending
  feedback and does not re-arm it for that window.
* The typing bed stops at the tool's end, on error and on cancel; it is also
  capped so a hung tool cannot type forever.  A started phrase is allowed to
  finish (a cut "One mo--" is worse than the phrase) unless the reply or the
  caller begins, which cut it like any filler.
* The agent's own control tools never get feedback: ``end_call`` and
  ``transfer_to_human`` are the end of the conversation, and a language switch
  is the worker's own bookkeeping.
* It defers to acknowledgements.  While a tool window is open the
  acknowledgement scheduler is suspended, and in ``acknowledge`` mode a turn
  that already heard an acknowledgement hears no second phrase.

The owner is driven from the session's callbacks and therefore never raises:
feedback is a nicety and a call is not.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

try:  # package layout (tests) vs flat image layout (worker)
    from .acknowledgements import (
        BackgroundSpeech,
        flush_background_player,
        neutral_phrase_problem,
    )
except ImportError:  # pragma: no cover - flat layout inside the worker image
    from acknowledgements import (  # type: ignore[no-redef]
        BackgroundSpeech,
        flush_background_player,
        neutral_phrase_problem,
    )


#: The customer's choice.  ``off`` is the default and is never stored.
TOOL_WAIT_MODES: tuple[str, ...] = ("off", "acknowledge", "sound")

#: Reviewed phrases per language for ``acknowledge`` mode.  Same rule as the
#: acknowledgement list: a language absent here turns the mode off for that
#: call rather than speaking English at a caller who chose another language.
TOOL_WAIT_ACKNOWLEDGEMENTS: Mapping[str, tuple[str, ...]] = {
    "en": ("One moment while I check.", "Let me look that up.", "Just a moment."),
}

#: How long a tool may run before feedback is worth starting.  Below this the
#: median tool has already returned and the caller has not registered a wait.
DEFAULT_TOOL_WAIT_DELAY_SECONDS = 1.0

#: Bounds mirrored from the control plane's ``voice_polish.tool_wait``
#: contract; re-checked here so a value the API never accepted cannot become
#: behaviour because it reached the realtime config another way.
DELAY_BOUNDS_SECONDS = (0.6, 3.0)
MAX_PHRASES = 6

#: The typing bed's ceiling.  A tool that runs longer than this is a tool that
#: has hung, and typing for half a minute tells the caller nothing true.
SOUND_MAX_SECONDS = 15.0
#: Quiet: it sits under the agent's voice level so it reads as room sound.
SOUND_VOLUME = 0.35
#: Faded in so the bed does not start with a click; NOT faded out.  The cut has
#: to be instant -- a fade would keep the mixer producing frames after the
#: queue flush -- and a keyboard transient does not click on a hard stop.
SOUND_FADE_IN_SECONDS = 0.25
SOUND_FADE_OUT_SECONDS = 0.0

#: How long feedback may hold for an acknowledgement that is still playing
#: when its own delay expires, and how often it looks.  A phrase is at most
#: five words; two seconds covers the slowest voice, and a phrase still
#: playing after that is waited for no longer -- the feedback is skipped, it
#: never starts underneath.
ACKNOWLEDGEMENT_PLAYOUT_WAIT_MAX_SECONDS = 2.0
ACKNOWLEDGEMENT_PLAYOUT_POLL_SECONDS = 0.1
#: How long after a phrase handle reports done its audio can still be in the
#: shared player's pipeline.  The pinned player marks a handle done when the
#: mixer has pulled its last frame; behind that point sit the block being
#: mixed and one finished block (100 ms each) and the 400 ms audio-source
#: queue, so the last word is heard up to ~600 ms later.  The bed's cut
#: flushes that queue, so the bed starts no sooner than this after the phrase
#: -- with margin, because a truncated acknowledgement is the one failure this
#: module must never produce.
PLAYER_PLAYOUT_TAIL_SECONDS = 1.0

#: Operator kill switch for a whole deployment, same shape as the
#: acknowledgement one.
TOOL_WAIT_DISABLED_VAR = "SKYSAY_TOOL_WAIT_DISABLED"

#: Tools whose wait is never covered.  Ending or transferring the call is not a
#: wait the caller should hear typing through, and the language switch is the
#: worker's own step, invisible to the caller by design.
NEVER_COVERED_TOOLS = frozenset({"end_call", "transfer_to_human", "switch_language"})

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})


def tool_wait_disabled_here(environ: Mapping[str, str]) -> bool:
    """Whether the deployment refuses tool-wait feedback for every call."""
    return (
        str(environ.get(TOOL_WAIT_DISABLED_VAR, "") or "").strip().lower()
        in _TRUE_VALUES
    )


def tool_wait_acknowledgements_for(language: Any) -> tuple[str, ...]:
    """Return the reviewed tool-wait phrases for a language, or none."""
    key = str(language or "").strip().lower()
    base = key.split("-")[0].split("_")[0]
    return TOOL_WAIT_ACKNOWLEDGEMENTS.get(base, ())


@dataclass(frozen=True)
class ToolWaitSettings:
    """What this call does during a slow tool, resolved from its realtime config."""

    mode: str
    phrases: tuple[str, ...]
    delay_seconds: float
    #: Why ``mode`` is ``off`` when the customer asked for feedback.
    reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def tool_wait_settings(
    realtime_config: Mapping[str, Any] | None,
    *,
    language: Any,
    environ: Mapping[str, str] | None = None,
) -> ToolWaitSettings:
    """Resolve the customer's ``voice_polish.tool_wait`` for one call.

    Off unless the section names a known mode.  ``acknowledge`` needs phrases:
    the customer's own when they pass the shared neutral-phrase rule, else the
    reviewed list for the call's language, else the mode is off with the reason
    ``no_phrases_for_language`` -- never English at a non-English caller.
    ``sound`` needs no language at all.
    """

    polish = (
        (realtime_config or {}).get("voice_polish")
        if isinstance(realtime_config, Mapping)
        else None
    )
    section = polish.get("tool_wait") if isinstance(polish, Mapping) else None
    off = ToolWaitSettings("off", (), DEFAULT_TOOL_WAIT_DELAY_SECONDS, "")
    if not isinstance(section, Mapping):
        return off
    mode = str(section.get("mode") or "").strip().lower()
    if mode not in TOOL_WAIT_MODES or mode == "off":
        return off
    if tool_wait_disabled_here(environ if environ is not None else {}):
        return ToolWaitSettings(
            "off", (), DEFAULT_TOOL_WAIT_DELAY_SECONDS, "disabled_by_operator"
        )
    delay = DEFAULT_TOOL_WAIT_DELAY_SECONDS
    raw_delay = section.get("delay_ms")
    if isinstance(raw_delay, (int, float)) and not isinstance(raw_delay, bool):
        low, high = DELAY_BOUNDS_SECONDS
        candidate = float(raw_delay) / 1000.0
        if low <= candidate <= high:
            delay = candidate
    if mode == "sound":
        return ToolWaitSettings("sound", (), delay, "")
    phrases: list[str] = []
    raw_phrases = section.get("phrases")
    if isinstance(raw_phrases, (list, tuple)):
        for raw in raw_phrases[:MAX_PHRASES]:
            if not isinstance(raw, str):
                continue
            text = " ".join(raw.split())
            if text and neutral_phrase_problem(text) is None and text not in phrases:
                phrases.append(text)
    if not phrases:
        phrases = list(tool_wait_acknowledgements_for(language))
    if not phrases:
        return ToolWaitSettings("off", (), delay, "no_phrases_for_language")
    return ToolWaitSettings("acknowledge", tuple(phrases), delay, "")


def typing_sound(player: Any, *, source: Any = None) -> BackgroundSpeech:
    """Start the looping typing bed on the dedicated filler player.

    Returns the same interruptible handle acknowledgements use, so cutting it
    stops the mixer input AND flushes the player's queued audio; without the
    flush up to 400 ms of typing would still play over the answer.  The flush
    is player-wide, which is why the owner never starts the bed while an
    acknowledgement is audible or still queued (see ``_wait_then_start``).
    """

    if source is None:
        from livekit.agents.voice.background_audio import AudioConfig, BuiltinAudioClip

        source = AudioConfig(
            BuiltinAudioClip.KEYBOARD_TYPING,
            volume=SOUND_VOLUME,
            fade_in=SOUND_FADE_IN_SECONDS,
            fade_out=SOUND_FADE_OUT_SECONDS,
        )
    return BackgroundSpeech(
        player.play(source, loop=True), flush=lambda: flush_background_player(player)
    )


class ToolWaitFeedback:
    """Own the feedback for one call's tool waits.

    Driven by the session's ``tool_execution_updated``, ``user_state_changed``
    and ``agent_state_changed`` callbacks.  ``speak`` and ``sound`` are the two
    ways of making noise, both returning a handle with ``interrupt()`` and
    ``done()``; ``acknowledgements`` is the acknowledgement scheduler to defer
    to (``suspend``/``resume``/``spoke_this_turn``), or None.  The clock is
    injected so every timing rule is testable without a loop that really waits.
    """

    def __init__(
        self,
        *,
        mode: str,
        phrases: Sequence[str] = (),
        delay_seconds: float = DEFAULT_TOOL_WAIT_DELAY_SECONDS,
        speak: Callable[[str], Any] | None = None,
        sound: Callable[[], Any] | None = None,
        acknowledgements: Any | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        emit: Callable[..., None] | None = None,
        sound_max_seconds: float = SOUND_MAX_SECONDS,
        suppressed: Callable[[], bool] | None = None,
    ) -> None:
        if mode not in TOOL_WAIT_MODES or mode == "off":
            raise ValueError("tool-wait feedback is built only for an active mode")
        if mode == "acknowledge" and (speak is None or not phrases):
            raise ValueError("acknowledge mode needs phrases and a speaker")
        if mode == "sound" and sound is None:
            raise ValueError("sound mode needs a sound")
        if delay_seconds < 0 or sound_max_seconds <= 0:
            raise ValueError(
                "tool-wait delay must be non-negative and the sound cap positive"
            )
        self._mode = mode
        self._phrases = tuple(phrases)
        self._delay_seconds = float(delay_seconds)
        self._speak = speak
        self._sound = sound
        self._acknowledgements = acknowledgements
        self._sleep = sleep or asyncio.sleep
        self._emit = emit
        self._sound_max_seconds = float(sound_max_seconds)
        # Read at arming and at the instant before anything is played (a phrase
        # or the typing bed): a call in a phone menu or a hold queue is not
        # waiting on a tool the caller should hear about. A reader that raises
        # suppresses.
        self._suppressed = suppressed
        # Tools currently running, by call id.  The window is open while this
        # is non-empty; parallel tools share it.
        self._open: dict[str, str] = {}
        self._caller_speaking = False
        self._agent_speaking = False
        # Set for the rest of a window once something made feedback wrong for
        # it: the caller spoke, the reply started, or feedback already played.
        self._quiet_this_window = False
        # One spoken filler per caller turn, counting our own phrases as well
        # as the acknowledgement scheduler's.
        self._spoke_this_turn = False
        # Whether THIS window spoke a phrase, handed to the acknowledgement
        # scheduler when the window closes so it does not add a second one.
        self._spoke_this_window = False
        self._pending: asyncio.Task[None] | None = None
        self._playing: Any | None = None
        self._playing_kind = ""
        self._cap: asyncio.Task[None] | None = None
        self._rotation = 0
        self._closed = False
        # Bumped by everything that means "do not start after all", so a wait
        # whose cancellation has not been delivered yet still knows it lost.
        self._epoch = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def window_open(self) -> bool:
        return bool(self._open)

    # -- session callbacks -------------------------------------------------

    def on_tool_execution_updated(self, event: Any) -> None:
        """Read the pinned ``ToolExecutionUpdatedEvent`` shape safely."""
        try:
            update = getattr(event, "update", None)
            kind = str(getattr(update, "type", "") or "")
            if kind == "tool_call_started":
                call = getattr(update, "function_call", None)
                self.on_tool_started(
                    str(getattr(call, "call_id", "") or ""),
                    str(getattr(call, "name", "") or ""),
                )
            elif kind == "tool_call_ended":
                self.on_tool_ended(
                    str(getattr(update, "call_id", "") or ""),
                    str(getattr(update, "status", "") or ""),
                )
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def on_tool_started(self, call_id: str, name: str) -> None:
        try:
            if self._closed or not call_id or name in NEVER_COVERED_TOOLS:
                return
            first = not self._open
            self._open[call_id] = name
            if not first:
                return
            # A window that opens while the caller is talking stays quiet: the
            # caller does not need to be told the agent is busy.
            self._quiet_this_window = self._caller_speaking
            if self._acknowledgements is not None:
                self._acknowledgements.suspend("tool_window")
            self._arm()
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def on_tool_ended(self, call_id: str, status: str) -> None:
        try:
            if self._open.pop(call_id, None) is None or self._open:
                return
            reason = {
                "done": "tool_done",
                "error": "tool_error",
                "cancelled": "tool_cancelled",
            }.get(str(status or ""), "tool_done")
            self._cancel_pending(reason)
            # The bed stops with the tool; a phrase already speaking may finish
            # (see the module rules) and stays cuttable by the reply.
            self._interrupt(reason, kinds=("sound",))
            self._window_closed()
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def on_user_state_changed(self, old_state: Any, new_state: Any) -> None:
        try:
            self._caller_speaking = str(new_state) == "speaking"
            if self._caller_speaking:
                # A new caller turn: a fresh chance to speak once, and any
                # feedback for the previous one must not land on the caller.
                # Whatever this window spoke belonged to the turn that just
                # ended, so it must not be charged to the new one when the
                # window closes.
                self._spoke_this_turn = False
                self._spoke_this_window = False
                self._quiet_this_window = True
                self._cancel_pending("caller_speaking")
                self._interrupt("caller_speaking")
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def on_agent_state_changed(self, old_state: Any, new_state: Any) -> None:
        try:
            state = str(new_state)
            self._agent_speaking = state == "speaking"
            if state == "speaking":
                if self._open:
                    # The SDK starts the tools before the first frame of the
                    # words that precede them plays, so speech inside the
                    # window is the agent's preamble (or a spoken tool
                    # update), not the reply.  Cut anything playing -- never
                    # over the agent -- but keep the window: the wait starts
                    # again when the words end, unless feedback already
                    # played, which ``_start`` recorded.
                    self._cancel_pending("agent_speaking")
                    self._interrupt("agent_speaking")
                    return
                # No tool running: this is the reply, and a phrase that was
                # allowed to finish after its tool is cut and flushed now.
                self._cancel_pending("reply_started")
                self._interrupt("reply_started")
                return
            if state == "listening":
                # The turn is over.  Anything still booked as running is stale
                # bookkeeping (a tool whose end was never reported), and it
                # must not keep the acknowledgement scheduler suspended.
                self._cancel_pending("turn_ended")
                self._interrupt("turn_ended")
                if self._open:
                    self._open.clear()
                    self._window_closed()
                return
            # "thinking" (or any other non-speaking state) with the window open
            # is the moment a deferred wait may start: the agent has finished
            # the words it said before calling the tool.
            self._arm()
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def stop(self) -> None:
        """Stop for the rest of the call, without closing.

        Used when the call is on its way out or leaves the language its clips
        were made in; nothing may play in front of a goodbye.
        """
        self._closed = True
        self._cancel_pending("stopped")
        self._interrupt("stopped")
        if self._open:
            self._open.clear()
            self._window_closed()

    # -- internals ---------------------------------------------------------

    def _window_closed(self) -> None:
        self._quiet_this_window = False
        spoke = self._spoke_this_window
        self._spoke_this_window = False
        if self._acknowledgements is not None:
            # Resumed, not re-armed by us: the scheduler decides whether the
            # reply is still worth covering, and is told when this window
            # already spoke so the turn is not acknowledged twice.
            self._acknowledgements.resume(acknowledged=spoke)

    def _occupied(self) -> bool:
        """Whether feedback is still audible.

        A phrase that has played out no longer occupies the player, but its
        handle is kept: up to 400 ms of it can still sit in the player's queue,
        and the reply's edge owes that queue a flush.  The next feedback simply
        replaces the handle; it starts at least a delay later, by which time
        the queue has drained on its own.
        """
        if self._playing is None:
            return False
        if self._playing_kind == "phrase" and self._done(self._playing):
            return False
        return True

    def _is_suppressed(self) -> bool:
        if self._suppressed is None:
            return False
        try:
            return bool(self._suppressed())
        except Exception:  # noqa: BLE001 - an unreadable state is not a licence to speak
            return True

    def _arm(self) -> None:
        if (
            self._closed
            or not self._open
            or self._is_suppressed()
            or self._quiet_this_window
            or self._pending is not None
            or self._occupied()
            or self._caller_speaking
            or self._agent_speaking
        ):
            return
        if self._mode == "acknowledge" and self._already_acknowledged():
            # One spoken filler per turn across both features: the caller has
            # already heard "one moment" for this turn.
            self._quiet_this_window = True
            self._notify("tool_wait_feedback_skipped", reason="already_acknowledged")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)
            return
        self._pending = loop.create_task(
            self._wait_then_start(self._epoch), name="ToolWaitFeedback._wait"
        )

    @staticmethod
    def _done(handle: Any) -> bool:
        try:
            return bool(handle.done())
        except Exception:  # noqa: BLE001 - a foreign handle must not break the owner
            return True

    def _already_acknowledged(self) -> bool:
        return self._spoke_this_turn or bool(
            getattr(self._acknowledgements, "spoke_this_turn", False)
        )

    async def _wait_then_start(self, epoch: int) -> None:
        try:
            await self._sleep(self._delay_seconds)
            # An acknowledgement that started just before the tool did is
            # still audible and never moves the agent state; do not mix the
            # bed under it.  Bounded: a phrase is a few words long.
            waited = 0.0
            while (
                epoch == self._epoch
                and getattr(self._acknowledgements, "speaking", False)
                and waited < ACKNOWLEDGEMENT_PLAYOUT_WAIT_MAX_SECONDS
            ):
                await self._sleep(ACKNOWLEDGEMENT_PLAYOUT_POLL_SECONDS)
                waited += ACKNOWLEDGEMENT_PLAYOUT_POLL_SECONDS
            acknowledgement_still_playing = bool(
                getattr(self._acknowledgements, "speaking", False)
            )
            if (
                epoch == self._epoch
                and not acknowledgement_still_playing
                and (
                    waited or getattr(self._acknowledgements, "spoke_this_turn", False)
                )
            ):
                # An acknowledgement played in this turn -- possibly finishing
                # just before this wait expired.  A handle reports done while
                # its last frames are still in the shared player's pipeline,
                # and the bed's cut flushes that pipeline's queue; give them
                # time to be heard.
                await self._sleep(PLAYER_PLAYOUT_TAIL_SECONDS)
        except asyncio.CancelledError:
            raise
        try:
            if (
                epoch != self._epoch
                or self._closed
                or not self._open
                or self._quiet_this_window
                or self._caller_speaking
                or self._agent_speaking
                or self._is_suppressed()
            ):
                return
            if acknowledgement_still_playing:
                # A custom phrase in a slow voice can outlast the bound.  That
                # is real speech, not a stuck handle: nothing starts under it.
                self._quiet_this_window = True
                self._notify(
                    "tool_wait_feedback_skipped", reason="acknowledgement_playing"
                )
                return
            if self._mode == "acknowledge" and self._already_acknowledged():
                self._quiet_this_window = True
                self._notify(
                    "tool_wait_feedback_skipped", reason="already_acknowledged"
                )
                return
            self._start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - feedback must never break a call
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)
        finally:
            # Only this task's own slot: a wait that lost the epoch after its
            # sleep must not clear a newer wait armed in the meantime.
            if self._pending is asyncio.current_task():
                self._pending = None

    def _start(self) -> None:
        tools = sorted(set(self._open.values()))
        # Whatever happens next, this window has had its one chance.
        self._quiet_this_window = True
        if self._mode == "sound":
            assert self._sound is not None
            self._playing = self._sound()
            self._playing_kind = "sound"
            self._notify("tool_wait_feedback_started", mode="sound", tools=tools)
            self._cap = asyncio.get_running_loop().create_task(
                self._cap_sound(self._playing), name="ToolWaitFeedback._cap"
            )
            return
        assert self._speak is not None
        # Round-robin, skipping phrases whose clip is not ready yet: a slow
        # provider must not silence the ones that did render.
        for offset in range(len(self._phrases)):
            phrase = self._phrases[(self._rotation + offset) % len(self._phrases)]
            try:
                handle = self._speak(phrase)
            except LookupError:
                continue
            self._rotation += offset + 1
            self._spoke_this_turn = True
            self._spoke_this_window = True
            self._playing = handle
            self._playing_kind = "phrase"
            self._notify(
                "tool_wait_feedback_started",
                mode="acknowledge",
                tools=tools,
                phrase=phrase,
            )
            return
        self._notify("tool_wait_feedback_skipped", reason="clip_not_ready")

    async def _cap_sound(self, handle: Any) -> None:
        try:
            await self._sleep(self._sound_max_seconds)
        except asyncio.CancelledError:
            raise
        if self._playing is handle:
            self._interrupt("capped")

    def _cancel_pending(self, reason: str) -> None:
        self._epoch += 1
        pending = self._pending
        self._pending = None
        if pending is not None and not pending.done():
            pending.cancel()
            self._notify("tool_wait_feedback_cancelled", reason=reason)

    def _interrupt(
        self, reason: str, *, kinds: Sequence[str] = ("sound", "phrase")
    ) -> None:
        playing = self._playing
        if playing is None or self._playing_kind not in kinds:
            return
        self._playing = None
        kind = self._playing_kind
        self._playing_kind = ""
        cap = self._cap
        self._cap = None
        if cap is not None and not cap.done():
            cap.cancel()
        try:
            # Unconditionally, like an acknowledgement: a finished handle still
            # has queued audio and ``interrupt`` is the only path that flushes it.
            playing.interrupt()
            self._notify("tool_wait_feedback_interrupted", reason=reason, kind=kind)
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("tool_wait_feedback_failed", error=type(exc).__name__)

    def _notify(self, event: str, **details: Any) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event, **details)
        except Exception:
            pass

    async def aclose(self) -> None:
        self._closed = True
        self._epoch += 1
        self._interrupt("closed")
        tasks = [task for task in (self._pending, self._cap) if task is not None]
        self._pending = None
        self._cap = None
        for task in tasks:
            if task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Our own cancellation settling.  A pending request on the
                # CLOSING task is the worker job being cancelled, and that one
                # keeps propagating.
                closer = asyncio.current_task()
                if closer is not None and closer.cancelling() > 0:
                    raise
            except Exception:  # noqa: BLE001 - feedback must never break a call
                pass


__all__ = [
    "ACKNOWLEDGEMENT_PLAYOUT_POLL_SECONDS",
    "ACKNOWLEDGEMENT_PLAYOUT_WAIT_MAX_SECONDS",
    "DEFAULT_TOOL_WAIT_DELAY_SECONDS",
    "DELAY_BOUNDS_SECONDS",
    "MAX_PHRASES",
    "NEVER_COVERED_TOOLS",
    "PLAYER_PLAYOUT_TAIL_SECONDS",
    "SOUND_FADE_IN_SECONDS",
    "SOUND_FADE_OUT_SECONDS",
    "SOUND_MAX_SECONDS",
    "SOUND_VOLUME",
    "TOOL_WAIT_ACKNOWLEDGEMENTS",
    "TOOL_WAIT_DISABLED_VAR",
    "TOOL_WAIT_MODES",
    "ToolWaitFeedback",
    "ToolWaitSettings",
    "tool_wait_acknowledgements_for",
    "tool_wait_disabled_here",
    "tool_wait_settings",
    "typing_sound",
]

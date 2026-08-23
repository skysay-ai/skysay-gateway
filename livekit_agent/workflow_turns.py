# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Pure, livekit-free turn-loop logic for a workflow-pinned Talk call.

This module intentionally imports nothing from livekit.agents so it stays
importable and unit-testable from the main test suite -- unlike
production_agent.py, which this repo cannot even import (no livekit
dependency at all). It owns every DECISION a media worker needs to make
around a workflow-pinned call: whether a session is workflow-pinned at all,
what text (if any) a turn response should speak, and whether the call is
now over. production_agent.py's job is only to wire these pure answers to
LiveKit's session/speech primitives -- see its ``llm_node`` override.

The control plane stays the sole execution authority throughout: every
function here operates on data the control plane already returned (trusted
call config, or a workflow-turns response body), never invents routing or
policy of its own.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

# A run's runner status the client (Chat, Simulations, or Talk) must treat
# as "the conversation is over" -- mirrors conversation_workflow_runtime's
# STATUS_COMPLETED/STATUS_TRANSFERRED/STATUS_FAILED_CLOSED trio exactly.
TERMINAL_STATUSES = frozenset({"completed", "transferred", "failed_closed"})

# A CAP on the wait for a terminal farewell to finish playing, no longer the
# whole mechanism.
#
# The earlier note here claimed llm_node had no handle to the speech the
# SDK's pipeline builds from its yielded text, so the only option was to
# guess a duration. That was wrong at the pinned livekit-agents (1.6.6, see
# the agent requirements file): Agent exposes a `session` property,
# AgentSession exposes `current_speech` -> SpeechHandle, and SpeechHandle
# has `wait_for_playout()`, documented as waiting for "the entire assistant
# turn to complete playback". So the real signal is reachable and is what we
# now wait on; this constant only bounds that wait, so a farewell longer
# than the old guess is no longer cut off mid-sentence.
#
# Two things the same reading settled:
#   - `session.say(...)` returns a SpeechHandle whose __await__ calls
#     wait_for_playout, so the opening turn's `await session.say(...)` has
#     always waited for full playout. That path never had this bug.
#   - wait_for_playout refuses to be called from within a function owning
#     the same handle, so llm_node must NOT await it inline -- it captures
#     the handle and the detached termination task does the waiting.
TERMINATION_GRACE_SECONDS = 8.0

WORKFLOW_TURN_ERROR_FALLBACK: dict[str, str] = {
    "en": "Sorry, I'm having trouble continuing this test. Goodbye.",
    "et": "Vabandust, selle testi jätkamisega tekkis probleem. Head aega.",
    "si": "සමාවෙන්න, මෙම පරීක්ෂණය දිගටම කරගෙන යාමට අපහසුතාවයක් තිබේ. ආයුබෝවන්.",
}


def pinned_workflow_run_id(config: dict[str, Any]) -> str:
    """The workflow run id pinned to this call's trusted config, or "".

    Presence alone is what flips a Talk session from the free-form
    AgentSession path to the workflow-driven one. ``config`` (the call's
    own resolve_call_realtime_config response) is the ONLY source ever
    consulted -- job/room dispatch metadata is never trusted for this.
    """
    return str((config or {}).get("workflow_run_id") or "").strip()


def step_display_text(contract: dict[str, Any], step_id: str) -> str:
    """The verbatim authored text for one spoken step.

    A direct Python port of frontend/src/lib/agent-workflow-run-view.js's
    stepDisplayText -- keep the two in lockstep on any step-shape change.
    Never model-phrased: what the workflow author wrote in the visual
    builder is exactly what gets spoken, for every decision provider
    (deterministic or Gemini) and every client (Chat, Simulations, Talk).
    """
    for step in (contract or {}).get("steps") or []:
        if not isinstance(step, dict) or step.get("id") != step_id:
            continue
        step_type = step.get("type")
        if step_type in ("ask", "say"):
            return str(step.get("prompt") or "")
        if step_type == "end":
            return str(step.get("farewell") or "")
        return ""
    return ""


def spoken_reply_text(contract: dict[str, Any], steps: Iterable[dict[str, Any]]) -> str:
    """Every spoken step's text from one turn's evidence, joined for TTS.

    Mirrors the frontend's agentBubblesFromSteps, collapsed into one
    utterance: Chat renders each spoken step as its own bubble, but a
    phone call speaks one continuous reply.
    """
    parts = [
        step_display_text(contract, str(step.get("step_id") or ""))
        for step in steps
        if isinstance(step, dict) and step.get("spoken")
    ]
    return " ".join(part for part in parts if part)


def is_terminal_status(status: str) -> bool:
    return status in TERMINAL_STATUSES


@dataclass(frozen=True)
class WorkflowTurnOutcome:
    reply_text: str
    terminal: bool


def build_turn_outcome(response: dict[str, Any], language: str = "en") -> WorkflowTurnOutcome:
    """Turn one control-plane advance-turn response into what to speak.

    ``response`` is the exact JSON body POST /v1/calls/{id}/workflow-turns
    returns: the run DTO (status + its pinned_contract) plus this turn's
    sanitized step evidence. ``language`` selects the fallback farewell for a
    terminal turn that has nothing authored to say.
    """
    contract = response.get("pinned_contract") if isinstance(response.get("pinned_contract"), dict) else {}
    steps = response.get("steps") if isinstance(response.get("steps"), list) else []
    status = str(response.get("status") or "")
    terminal = is_terminal_status(status)
    reply_text = spoken_reply_text(contract, steps)
    if terminal and not reply_text and status == "failed_closed":
        # A run that failed closed often has no spoken step to quote -- it
        # stopped BECAUSE something went wrong. Yielding nothing here hung
        # the call up in silence, which reads to the caller as the line
        # dropping. The localized error farewell is the honest thing to say:
        # something went wrong and the call is ending.
        reply_text = workflow_turn_error_message(language)
    return WorkflowTurnOutcome(reply_text=reply_text, terminal=terminal)


def latest_caller_text(chat_items: Iterable[Any]) -> str | None:
    """The most recent caller utterance from a LiveKit ChatContext's items.

    Duck-typed on purpose: accepts anything exposing ``.role`` and
    ``.text_content`` (LiveKit's ChatMessage shape) so this stays testable
    with a plain fake, without importing livekit.agents. Returns None if no
    user-role item is found or its text is empty -- callers treat that the
    same as an opening turn.
    """
    for item in reversed(list(chat_items)):
        if getattr(item, "role", None) != "user":
            continue
        text = getattr(item, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    return None


def workflow_turn_error_message(language: str) -> str:
    return WORKFLOW_TURN_ERROR_FALLBACK.get(language, WORKFLOW_TURN_ERROR_FALLBACK["en"])


async def await_playout_then_finish(
    speech_handle: Any,
    finish: Callable[[], Any],
    *,
    grace: float = TERMINATION_GRACE_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """End the call once the terminal farewell has actually been heard.

    Waits on the SDK's own playout-completion signal when one is available,
    capped at ``grace`` so a stuck or never-completing handle cannot hold a
    session open forever. Without a usable handle it falls back to the old
    fixed-delay approximation, which is a guess but a documented one.
    """
    waiter = getattr(speech_handle, "wait_for_playout", None)
    if waiter is None:
        await sleep(grace)
        finish()
        return
    try:
        await asyncio.wait_for(waiter(), timeout=grace)
    except (TimeoutError, asyncio.TimeoutError):
        # The cap, reached. The farewell has had the full grace either way.
        pass
    except Exception:
        # The signal exists but would not serve us. Falling straight through
        # to finish() would cut the farewell off; take the old approximation
        # instead so behavior degrades to what it was, not to worse.
        await sleep(grace)
    finish()


class OpeningAdvanceFailed(Exception):
    """The opening turn already ended this call, so a caller turn waiting on
    it must not speak. Distinct from an ordinary turn failure, which SHOULD
    apologize -- here the apology (or the farewell) has already been said."""


class WorkflowTurnCoordinator:
    """Order the advances of one workflow-pinned session.

    ``session.start`` enables caller audio before the opening advance has
    finished, so a caller who speaks immediately puts two advances on the
    same fresh run at once. Both problems that causes are real:

    - The control plane's optimistic guard lets exactly one win, and the
      loser raises -- which the worker turns into the error farewell and
      hangs up. A caller who simply answered quickly got the call dropped.
    - Even without a conflict, a caller turn that wins the race advances a
      run whose ``active_step`` is still None. The runner treats that as a
      fresh run and jumps to the entry step, silently discarding what the
      caller said. Serializing alone does not fix this; the caller's turn
      has to WAIT for the opening one, not merely follow it.

    So this does both: one lock makes advances mutually exclusive, and an
    event makes caller turns wait until the opening advance is actually
    done. Living here rather than in the LiveKit Agent subclass is what
    makes it testable -- this module imports no livekit.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._opening_done = asyncio.Event()
        # What the opening advance raised, if it did, and whether the
        # opening path has already put this call into teardown. A caller turn
        # waiting on the event must not add a second goodbye to one already
        # being spoken.
        self._opening_error: BaseException | None = None
        self._conversation_over = False

    async def run_opening(self, advance: Callable[[], Awaitable[Any]]) -> Any:
        """Run the opening advance. Does NOT release waiting caller turns.

        Releasing here -- in a finally, the moment the advance returned or
        raised -- let a waiting turn run while the opening path had not yet
        decided what to say or said it. Both outcomes then produced two
        overlapping goodbyes: on failure the waiter re-raised into the
        caller-turn error handler and spoke a second farewell over the
        opening's, and on a terminal opening the waiter advanced a finished
        run, got a conflict, and spoke one too. The opening path calls
        :meth:`opening_finished` when it is genuinely done.
        """
        try:
            async with self._lock:
                return await advance()
        except BaseException as exc:
            self._opening_error = exc
            raise

    def opening_finished(self, *, conversation_over: bool = False) -> None:
        """Release waiting caller turns, once the opening path is really done.

        MUST be called on every path out of the opening -- success, failure,
        terminal -- or a caller turn parks forever and the session hangs
        instead of ending. ``conversation_over`` says the call is already
        being torn down, either because the opening failed or because it was
        itself terminal; a waiting turn must then stay silent rather than add
        a second goodbye to the one already being spoken.
        """
        if conversation_over:
            self._conversation_over = True
        self._opening_done.set()

    async def run_caller_turn(self, advance: Callable[[], Awaitable[Any]]) -> Any:
        """Run a caller-driven advance once the opening one has completed.

        Raises :class:`OpeningAdvanceFailed` instead of advancing when the
        conversation is already ending. That is a distinct type on purpose:
        the caller-turn path has to tell "this turn failed, apologize" from
        "the call is already over, say nothing" -- treating the second as the
        first is exactly what produced the double farewell.
        """
        await self._opening_done.wait()
        if self._opening_error is not None or self._conversation_over:
            raise OpeningAdvanceFailed(
                "the opening turn already ended this call"
            ) from self._opening_error
        async with self._lock:
            return await advance()


__all__ = [
    "TERMINAL_STATUSES",
    "TERMINATION_GRACE_SECONDS",
    "WORKFLOW_TURN_ERROR_FALLBACK",
    "OpeningAdvanceFailed",
    "WorkflowTurnCoordinator",
    "await_playout_then_finish",
    "WorkflowTurnOutcome",
    "build_turn_outcome",
    "is_terminal_status",
    "latest_caller_text",
    "pinned_workflow_run_id",
    "spoken_reply_text",
    "step_display_text",
    "workflow_turn_error_message",
]

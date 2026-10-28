# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Worker-owned caller turns for business-action confirmation.

Only source-owned, sealed audio turns enter this transport. SDK captions and
model arguments never create or advance a caller turn. Activity travels
independently so new speech invalidates slow confirmation classification.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

try:
    from .tool_declarations import has_business_writes
except ImportError:
    from tool_declarations import has_business_writes

ACTION_INSTRUCTIONS = """
Business writes other than transfer_to_human and set_call_disposition use a
server-owned confirmation.
Invoke the tool to obtain its
complete preview, read it back, ask its one yes/no question, then wait. After
each caller reply invoke the SAME tool with the SAME details to observe its
action status; this does not repeat a completed write. Follow its instruction:
ask only one clarification when requested, never invent success, and report
unknown honestly without retrying. If details change, invoke with the new
details for a new preview and a new yes. Reads need no confirmation.
Never tell the caller a write is done unless its result has write_performed
true. NOT DONE means this action was not executed; OUTCOME UNKNOWN means it
may have been, so say so and do not repeat it.
transfer_to_human has no preview: the caller's request for a person, or their
yes to your offer, is its confirmation, so invoke it once without asking again.
set_call_disposition has no preview either: record the outcome without asking,
except connected_callback_requested, which is read back like any other write.
"""
#: The producer-id prefix by which this worker tells the server it reports
#: read-back playout (``action-question``): an answer then counts only after
#: the caller heard the complete read-back. A server without that command
#: keeps its earlier rule.
PLAYOUT_PRODUCER = "playout-v1:"


@dataclass(frozen=True, slots=True)
class CallerTurn:
    """Identity allocated by the capture owner before asynchronous recognition."""
    epoch: str
    identity: str
    sequence: int
    activity: int
    #: Microphone sample at which the caller's voice began (VAD's voiced
    #: onset, without its prefix padding); None only without a capture clock.
    onset: int | None = None


#: Agent playouts remembered for judging answers; older ones end long before
#: any answer still being recognised.
MAX_PLAYOUTS = 64


class CallerInputs:
    def __init__(self, call: Any):
        self.call = call
        self.producer_id = PLAYOUT_PRODUCER + uuid.uuid4().hex
        self.generation = 0
        self.activity = 0
        self.sequence = 0
        self.committed_activity = 0
        self.turns: dict[str, CallerTurn] = {}
        self.sealed: set[str] = set()
        self.completed: dict[str, str | None] = {}
        self._changed = asyncio.Event()
        self.active: CallerTurn | None = None
        self.capture_barrier = None
        self.failed = False
        self.closed = False
        self._retiring: asyncio.Task | None = None
        self._activity_tail: asyncio.Task | None = None
        self._input_tail: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._question: dict[str, Any] | None = None
        #: (live edge, queued) microphone samples now, from the capture owner.
        self.mic_clock = None
        self._playouts: list[dict[str, Any]] = []
        self._reports_supported = False
        self._start = self._spawn(self._register())

    def _spawn(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda _task: self._changed.set())
        return task

    async def _command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.call.action_command, command, payload)

    async def _register(self) -> None:
        try:
            result = await self._command("action-producer", {"producer_id": self.producer_id})
            generation = result.get("generation")
            if type(generation) is not int or generation < 1:
                raise ValueError("invalid producer acknowledgement")
            self.generation = generation
        except Exception:
            self.failed = True

    def unavailable(self) -> None:
        """An unresolved source cannot be skipped to a later yes.

        Retire server authority independently of any slow input request. The
        replacement producer is deliberately never exposed to this worker.
        """
        self.failed = True
        self._changed.set()
        if self.closed or self._retiring is not None:
            return
        async def retire():
            await self._start
            try:
                await self._command("action-producer", {"producer_id": PLAYOUT_PRODUCER + uuid.uuid4().hex})
            except Exception:
                pass  # Local refusal persists; server expiry is not permission.
        self._retiring = self._spawn(retire())

    def expect_question(self, action_id: str, *, call_id: str = "") -> None:
        """A write tool returned a read-back awaiting the caller's answer.

        Its answer counts only if the caller's voice began while the agent was
        not speaking, after the read-back was delivered: a playout that began
        after this tool's ``function_tools_executed`` ended uninterrupted. Any
        cut playout makes it undelivered until a reply to a later tool result
        completes. One clock: microphone samples.
        """
        if self._question is None or self._question["action_id"] != action_id:
            self._question = {"action_id": action_id, "arms": []}
        self._question["call_id"] = call_id

    def tools_executed(self, event: Any) -> None:
        question = self._question
        if question is None or question.get("call_id") is None:
            return
        calls = getattr(event, "function_calls", None) or ()
        if not question["call_id"] or any(getattr(call, "call_id", None) == question["call_id"] for call in calls):
            question["call_id"] = None
            now = self._mic()
            if now is not None:
                question["arms"] = (question["arms"] + [now[0]])[-MAX_PLAYOUTS:]

    def agent_speaking(self, state: Any) -> None:
        """An interval opens at the first audio and closes when speech stops.

        A handle that never produced audio has no interval. Completion is
        known only from its assistant message (``item_added``).
        """
        now = self._mic()
        if now is None:
            return
        received, queued = now
        open_ = self._playouts[-1] if self._playouts and self._playouts[-1]["end"] is None else None
        if state == "speaking":
            if open_ is None:
                self._playouts = (self._playouts + [{"start": received, "end": None, "complete": None}])[-MAX_PLAYOUTS:]
        elif open_ is not None:
            open_["end"] = received + queued

    def item_added(self, item: Any) -> None:
        if getattr(item, "role", None) != "assistant" or not self._playouts:
            return
        playout = self._playouts[-1]
        if playout["end"] is None:
            now = self._mic()
            if now is None:
                return
            playout["end"] = now[0] + now[1]
        # A realtime response can add several messages for one playout (a
        # filler, then the read-back): any interrupted one means it was cut.
        if getattr(item, "interrupted", False):
            playout["complete"] = False
        elif playout["complete"] is None:
            playout["complete"] = True

    def _mic(self) -> tuple[int, int] | None:
        try:
            now = self.mic_clock() if self.mic_clock is not None else None
        except Exception:
            return None
        return now if isinstance(now, tuple) and len(now) == 2 else None

    def _delivered(self, question: dict[str, Any], at: int | None) -> bool:
        """Whether the read-back was delivered by microphone sample ``at``
        (None: by now): a playout that began after the latest tool result
        ended complete, and nothing was cut since."""
        if not question["arms"]:
            return False
        first_arm = question["arms"][0]
        events = sorted([(arm, 0, index, None) for index, arm in enumerate(question["arms"]) if at is None or arm <= at]
                        + [(p["end"], 1, index, p) for index, p in enumerate(self._playouts)
                           if p["end"] is not None and (at is None or p["end"] <= at) and p["start"] >= first_arm])
        delivered, armed_since = False, None
        for when, kind, _index, playout in events:
            if kind == 0:
                armed_since = when
            elif playout["complete"] is True:
                if armed_since is not None and playout["start"] >= armed_since:
                    delivered = True
            else:  # cut, or its completion was never confirmed
                delivered, armed_since = False, None
        return delivered

    def _eligible(self, question: dict[str, Any], onset: int | None) -> tuple[bool, bool]:
        """(heard, may answer) for caller voice beginning at ``onset``.

        An answer needs the read-back delivered before its onset and the onset
        outside every agent playout; that judgement stands whatever the agent
        does later. ``heard`` is the read-back's state now, for the inputs
        after an ordering-only one.

        Both are measured at our capture point: a playout ends when our
        output drained, and the caller hears it up to a transport delay
        (about 100-300 ms) later. An answer overlapping that last fraction of
        a second of the read-back counts; the caller heard essentially all of
        it (lead ruling, Codex r3). No margin is added.
        """
        if onset is not None:
            inside = any(p["start"] <= onset and (p["end"] is None or onset < p["end"]) for p in self._playouts)
            if not inside and self._delivered(question, onset):
                return True, True
        return self._delivered(question, None), False

    def begin_turn(self, *, onset_sample: int | None = None) -> CallerTurn | None:
        """Called once per logical audio turn, not once per VAD fragment."""
        if self.closed or self.failed:
            return None
        if self.active is not None or len(self.turns) >= 4096:
            self.unavailable()
            return None
        self.activity += 1
        if type(onset_sample) is not int:
            now = self._mic()
            onset_sample = now[0] if now is not None else None
        turn = CallerTurn(self.producer_id, uuid.uuid4().hex, len(self.turns) + 1, self.activity, onset_sample)
        self.turns[turn.identity] = turn
        self.active = turn
        self._changed.set()
        previous = self._activity_tail
        async def deliver():
            await self._start
            if previous:
                await previous
            if self.failed or self.closed:
                return
            try:
                result = await self._command("action-activity", {"generation": self.generation, "sequence": turn.activity})
                if result.get("generation") != self.generation or result.get("sequence") != turn.activity:
                    raise ValueError("invalid activity acknowledgement")
            except Exception:
                self.unavailable()
        self._activity_tail = self._spawn(deliver())
        return turn

    def seal_turn(self, turn: CallerTurn) -> None:
        """The capture owner closes the finite audio span before recognition."""
        if self.closed or self.failed:
            return
        if self.active is not turn or self.turns.get(turn.identity) is not turn:
            self.unavailable()
            return
        self.sealed.add(turn.identity)
        self.active = None
        self._changed.set()

    def blank_turn(self, turn: CallerTurn) -> None:
        """Finite recognition completed with no words; this is never a yes."""
        self._complete(turn, None)

    def completed_turn(self, turn: CallerTurn, text: str) -> None:
        if type(text) is not str or not text.strip() or len(text) > 2000:
            self.unavailable()
            return
        self._complete(turn, text)

    def _complete(self, turn: CallerTurn, text: str | None) -> None:
        if self.closed or self.failed:
            return
        if (not isinstance(turn, CallerTurn) or self.turns.get(turn.identity) is not turn
                or turn.epoch != self.producer_id or turn.identity not in self.sealed):
            self.unavailable()
            return
        if turn.identity in self.completed:
            if self.completed[turn.identity] != text:
                self.unavailable()
            return
        self.completed[turn.identity] = text
        self._changed.set()
        # Recognition may finish out of order; only a contiguous prefix moves.
        for known in self.turns.values():
            if known.sequence <= self.sequence:
                continue
            if known.identity not in self.completed:
                break
            self.sequence = known.sequence
            self._queue_input(known, self.completed[known.identity])

    def _queue_input(self, turn: CallerTurn, text: str | None) -> None:
        previous = self._input_tail
        async def deliver():
            await self._start
            if previous:
                await previous
            if self.failed or self.closed:
                return
            try:
                await self._capture_and_activity()
                if self.failed or self.closed:
                    return
                await self._judge_answer(turn)
                if self.failed or self.closed:
                    return
                result = await self._command("action-inputs", {"generation": self.generation,
                    "sequence": turn.sequence, "activity_sequence": turn.activity, "id": turn.identity, "text": text})
                if result.get("generation") != self.generation or result.get("sequence") != turn.sequence:
                    raise ValueError("invalid input acknowledgement")
                self.committed_activity = turn.activity
                action = result.get("action")
                if isinstance(action, dict) and action.get("status") == "authorized":
                    # Classification may take seconds. Drain another finite
                    # watermark before releasing this exact caller authority.
                    await self._capture_and_activity()
                    if self.failed or self.closed:
                        return
                    await self._command("action-release", {"generation": self.generation,
                        "sequence": turn.sequence, "source_id": turn.identity, "action_id": action["id"]})
            except Exception:
                self.unavailable()
        self._input_tail = self._spawn(deliver())

    async def _judge_answer(self, turn: CallerTurn) -> None:
        """Before its input, tell the server whether this turn may answer.

        Sent on the existing playout-v1 route as the boundary for this input:
        ``turn.activity - 1`` lets it be judged, ``turn.activity`` makes it
        ordering only, and ``heard: false`` also keeps the read-back
        undelivered. A server without the route keeps its earlier rule; once
        one has answered, a lost judgement fails closed.
        """
        question = self._question
        if question is None:
            return
        heard, eligible = self._eligible(question, turn.onset)
        try:
            result = await self._command("action-question", {"generation": self.generation,
                "action_id": question["action_id"], "activity": turn.activity - 1 if eligible else turn.activity,
                "heard": heard})
        except Exception:
            if self._reports_supported:
                self.unavailable()
            return
        self._reports_supported = True
        if self._question is question and isinstance(result, dict) and result.get("status") not in (None, "awaiting_confirmation"):
            self._question = None  # answered or ended: later turns are not its answer

    async def _capture_and_activity(self):
        async with asyncio.timeout(12):
            if self.capture_barrier is not None:
                await self.capture_barrier()
            while (tail := self._activity_tail) is not None:
                await asyncio.shield(tail)
                if tail is self._activity_tail:
                    break

    async def reference(self, *, timeout: float = 12) -> dict[str, int] | None:
        # Join recognition as well as HTTP delivery. A fast conversation model
        # must not strand a normal reply while the finite listener drains EOF.
        try:
            async with asyncio.timeout(timeout):
                await self._capture_and_activity()
                while True:
                    self._changed.clear()
                    if self.failed or self.closed:
                        return None
                    if not self.turns:
                        return None
                    if (self.generation and self.sequence == len(self.turns)
                            and self.active is None and self.activity == self.committed_activity
                            and all(task is None or task.done() for task in
                                    (self._start, self._activity_tail, self._input_tail))):
                        return {"generation": self.generation, "sequence": self.sequence}
                    await self._changed.wait()
        except TimeoutError:
            return None
        except Exception:
            self.unavailable()
            return None

    async def drain(self, *, timeout: float = 3) -> None:
        """Finish the caller's pending answer before our own hang-up.

        A yes classified just before ``end_call`` is still waiting for its
        release; teardown retires the producer and would cancel it. Join
        recognition, classification and release, bounded. Every teardown
        entry (a remote hang-up included) still retires first: only our own
        end_call drains, so a caller who hung up never gets a write.
        """
        if self.closed or self.failed or not self.turns:
            return
        if (self.active is None and self.sequence == len(self.turns)
                and all(task is None or task.done() for task in (self._start, self._activity_tail, self._input_tail))):
            return
        await self.reference(timeout=timeout)

    async def aclose(self) -> None:
        self.unavailable()
        # A blocking input/retirement HTTP request is joined before teardown
        # releases its call. Canceling to_thread would leave that write alive.
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
        self.close()

    def close(self) -> None:
        self.closed = True
        self._changed.set()
        for task in tuple(self._tasks):
            task.cancel()


def bind_caller_inputs(session: Any, call: Any) -> CallerInputs | None:
    if not has_business_writes(call.config):
        return None
    inputs = CallerInputs(call)
    # Public SDK captions do not identify complete local audio turns. Only a
    # qualified capture adapter may call begin/seal/completed_turn. In its
    # absence reference() refuses; captions must never be a fallback.
    @session.on("close")
    def closed(*_args: Any) -> None:
        inputs.unavailable()

    @session.on("function_tools_executed")
    def tools_executed(event: Any) -> None:
        inputs.tools_executed(event)

    @session.on("agent_state_changed")
    def agent_state_changed(event: Any) -> None:
        inputs.agent_speaking(getattr(event, "new_state", None))

    @session.on("conversation_item_added")
    def item_added(event: Any) -> None:
        inputs.item_added(getattr(event, "item", None))
    return inputs

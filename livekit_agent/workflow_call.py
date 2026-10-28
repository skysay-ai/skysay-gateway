# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Universal spoken workflow coordinator over the existing control-plane runner.

One microphone owner supplies immutable observations. An output adapter owns
all rendering/playback children and joins them when its run is cancelled.
No engine, language, customer or carrier choice belongs in this coordinator.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Awaitable, Callable, Protocol

try:
    from .workflow_input import BlankInput, CallerInput, WorkflowInputFence, WorkflowInputRefused, WorkflowInputCapacityExceeded
    from .workflow_tasks import join_owned_task
    from .workflow_speech import LineNotVoiced
    from .capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS
except ImportError:
    from workflow_input import BlankInput, CallerInput, WorkflowInputFence, WorkflowInputRefused, WorkflowInputCapacityExceeded
    from workflow_tasks import join_owned_task
    from workflow_speech import LineNotVoiced
    from capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS


#: Complete caller turns queued behind the serial business actor. Turns merge
#: only when no piece's answer permission would change, so few ever queue.
MAX_QUEUED_CALLER_TURNS = 4
#: How long caller sounds may hold one output in all, from its first hold. A
#: longer sound cuts the line as before, far inside a delivery's claim lease.
MAX_CALLER_HOLD_SECONDS = 30


class ReplyOutput(Protocol):
    async def run(self) -> str:
        """Return completed_local/interrupted; cancellation joins every child."""
        ...

    async def present(self, boundary: dict[str, int]) -> None:
        """Join the backend's exact whole-reply presentation acknowledgement."""
        ...


class WorkflowCall:
    def __init__(self, *, fence: WorkflowInputFence,
                 advance: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
                 output: Callable[[dict[str, Any], int], ReplyOutput],
                 on_terminal: Callable[[bool], None], on_failure: Callable[[BaseException], None],
                 transfer: Callable[[dict[str, Any]], Awaitable[str]] | None = None,
                 action_status: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
                 action_release: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
                 action_cancel: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
                 capture_barrier: Callable[[], Awaitable[None]] | None = None) -> None:
        self.fence, self.advance, self.output = fence, advance, output
        self.on_terminal, self.on_failure = on_terminal, on_failure
        # Places a waiting transfer step's dial and returns how it ended
        # (``transfer.TransferController``'s outcome). The run waits at that step
        # until the outcome is submitted as the next turn.
        self.transfer = transfer
        self.action_status = action_status
        self.action_release, self.action_cancel, self.capture_barrier = action_release, action_cancel, capture_barrier
        self._action_cancellations: set[asyncio.Task] = set()
        self._action_waiter: asyncio.Task | None = None
        self._dial: asyncio.Task | None = None
        self._dial_cancelled = False
        # Caller turns that BEGAN before this input sample are retired, never
        # submitted: everything said while the run waited at a transfer step
        # (infinite until its outcome is submitted).
        self._retire_before: float | None = None
        self._turn_lock = asyncio.Lock()
        self._http: asyncio.Task | None = None
        self._business: asyncio.Task | None = None
        self._waiting: deque[CallerInput | BlankInput] = deque()
        self._input_ready = asyncio.Event()
        self._input_ready.set()
        self._play: asyncio.Task | None = None
        self._close: asyncio.Task | None = None
        self._reply: dict[str, Any] | None = None
        self._attempted = self._started = self._closed = self._terminal = False
        self._pending_inputs = self._generation = self._output_epoch = 0
        self._terminal_notified = False
        self._floor_lock = asyncio.Lock()
        self._floor_suspended = False
        self._floor_ready = False
        self._floor_released = asyncio.Event()
        self._floor_released.set()
        self._floor_generation = 0
        self._active_output = self._preserved_output = None
        self._floor_cancelled: set[asyncio.Task] = set()
        self._partial: dict[str, Any] | None = None
        # The one output task a caller sound holds, until its turn resolves.
        self._held_output: asyncio.Task | None = None
        self._hold_deadline: tuple[asyncio.Task, float] | None = None
        self._hold_timer: asyncio.TimerHandle | None = None
        # Set whenever no caller sound holds the current output.
        self._hold_released = asyncio.Event()
        self._hold_released.set()
        self._input_owner = None
        # The one failure delivered through ``on_failure``. Every owned task
        # that failed with it stores the same object, so closing finds it
        # again (see ``aclose``).
        self._reported_failure: BaseException | None = None

    @property
    def turn_in_flight(self) -> bool:
        return ((self._http is not None and not self._http.done())
                or (self._business is not None and not self._business.done())
                or (self._play is not None and not self._play.done())
                or (not self._closed and bool(self._pending_inputs or self._partial or self._waiting)))

    def _fail(self, error: BaseException) -> None:
        if not self._closed:
            self._closed = True
            self._floor_released.set()
            self.fence.close()
            self._waiting.clear()
            self._input_ready.set()
            self._cancel_output()
            self._reported_failure = error
            self.on_failure(error)

    def start(self, *, pickup: CallerInput | None = None):
        """Claim synchronously; the returned waiter never owns business HTTP.

        Both existing ``await start()`` callers and root-owned waiter tasks
        join the same serial actor. Capture callbacks only schedule that
        waiter, so a slow opening cannot stall the microphone actors.
        """
        if self._started or self._closed:
            raise WorkflowInputRefused("workflow call cannot start twice")
        wire = {"kind": "opening"}
        if pickup is not None:
            if (not isinstance(pickup, CallerInput) or pickup.answer_to_request_id is not None
                    or pickup.continues or pickup.turn_first_sample != pickup.first_sample):
                raise WorkflowInputRefused("workflow pickup must be a complete interrupt-only caller turn")
            self._validate_observation(pickup)
            complete = self._collect_input(pickup)
            wire = complete.wire(sample_rate=self.fence.sample_rate)
        self._started = True
        initial = asyncio.get_running_loop().create_future()
        async def drive():
            while not self._closed:
                await self._floor_released.wait()
                await self._join_navigation_inputs()
                async with self._turn_lock:
                    if self._closed:
                        return
                    if self._floor_suspended or self._navigation_unresolved():
                        continue
                    await self._submit(wire)
                    break
            initial.set_result(None)
            await self._drain_inputs()
        task = self._business = asyncio.create_task(drive())
        def finished(done):
            if not initial.done():
                if done.cancelled():
                    initial.cancel()
                elif (error := done.exception()) is not None:
                    initial.set_exception(error)
                else:
                    initial.set_result(None)
            self._observe_business(done)
        task.add_done_callback(finished)
        async def wait_started():
            # Preserve start()'s initial-HTTP completion boundary. Later
            # queued caller turns remain owned by the serial business actor.
            await join_owned_task(initial)
        return wait_started()

    async def on_speech(self) -> None:
        """Hold new exposure while the listener qualifies a logical caller turn."""
        self._note_speech()

    def replay_prestart_onsets(self, count: int) -> None:
        """Atomic handoff of the capture owner's remaining initial onsets."""
        if type(count) is not int or not 0 <= count <= 2 or not self._started:
            raise WorkflowInputRefused("workflow pickup has no bounded onset handoff")
        for _ in range(count):
            self._note_speech()

    def _note_speech(self) -> None:
        if self._closed:
            raise WorkflowInputRefused("workflow call is closed")
        self._pending_inputs += 1
        self._input_ready.clear()
        if self._pending_inputs > 4:
            error = WorkflowInputCapacityExceeded("workflow caller backlog requires a controlled call end")
            self._fail(error)
            raise error
        self._generation += 1
        reply = self._reply or {}
        binding = reply.get("action_wait") or (reply.get("runtime_state") or {}).get("business_action") or {}
        if (binding.get("action_id") and self.action_cancel is not None
                and (reply.get("action_wait") is not None or self._http is not None and not self._http.done())):
            # Never wait on network I/O in the microphone callback. Cancellation
            # competes with claim independently of a slow business request.
            task = asyncio.create_task(self.action_cancel(binding["action_id"]))
            self._action_cancellations.add(task)
            task.add_done_callback(self._action_cancellations.discard)
            task.add_done_callback(self._observe_business)

    async def on_interruption(self) -> None:
        """Stop audible output once qualified; joining cannot stall ingestion."""
        if self._closed or self._retired_onset():
            return
        self._cancel_output()

    async def on_caller_interruption(self) -> None:
        """A qualified caller sound holds an authored line until its words decide.

        Words cancel it as before (``on_input``); a turn that resolves blank
        resumes the same playout. An output that cannot hold, and a transfer's
        connecting line, are cancelled now, as before.
        """
        if self._closed or self._retired_onset():
            return
        task, owner = self._play, self._active_output
        if self._held_output is not None and self._held_output is task:
            return
        pause = getattr(owner, "pause_for_caller", None)
        if (task is not None and not task.done() and not task.cancelling() and callable(pause)
                and (self._reply or {}).get("transfer") is None):
            # Registered first, so words that arrive meanwhile cut this task.
            self._held_output = task
            self._hold_released.clear()
            held = await pause() is True
            if self._held_output is task and task is self._play and not task.done() and not task.cancelling():
                if held:
                    # However this output ends, its hold ends with it.
                    task.add_done_callback(lambda _task: self._end_hold(task, owner))
                    self._arm_hold_deadline(task)
                    return
            else:
                # Words or an ending decided the turn while the hold was taken.
                owner.resume_for_caller()
            self._end_hold(task, owner)
        self._cancel_output()

    def _end_hold(self, task: asyncio.Task, owner) -> None:
        owner.resume_for_caller()
        if self._held_output is task:
            self._held_output = None
        if self._held_output is None:
            self._hold_released.set()

    def _arm_hold_deadline(self, task: asyncio.Task) -> None:
        loop = asyncio.get_running_loop()
        if self._hold_deadline is None or self._hold_deadline[0] is not task:
            self._hold_deadline = (task, loop.time() + MAX_CALLER_HOLD_SECONDS)
        if self._hold_timer is not None:
            self._hold_timer.cancel()
        self._hold_timer = loop.call_at(self._hold_deadline[1], self._expire_hold, task)

    def _expire_hold(self, task: asyncio.Task) -> None:
        self._hold_timer = None
        if self._held_output is task:
            self._held_output = None
            self._hold_released.set()
            self._cancel_output()

    def _retired_onset(self) -> bool:
        first = self.fence.active_turn_first_sample
        # Speech that began while the run waited at a transfer step,
        # qualified late: it cuts neither the transfer's line nor the
        # reply that follows the dial.
        return self._retire_before is not None and first is not None and first < self._retire_before

    def _release_held_output(self) -> None:
        task, self._held_output = self._held_output, None
        self._hold_released.set()
        if task is not None and task is self._play and not task.done() and not task.cancelling():
            self._active_output.resume_for_caller()

    @property
    def reply_terminal(self) -> bool:
        """Whether the latest committed reply ended the workflow."""
        return self._terminal

    @property
    def terminal_fail_reason(self) -> str:
        """The latest reply's fail reason when it closed the run failed, else ``""``."""
        reply = self._reply or {}
        reason = reply.get("fail_reason") if reply.get("status") == "failed_closed" else None
        return reason if type(reason) is str else ""

    def retire_held_input(self, observation: CallerInput | BlankInput) -> None:
        """Retire one capture onset into the navigation owner, without a turn."""
        if self._closed or not self._started or self._pending_inputs < 1:
            raise WorkflowInputRefused("held workflow input has no owned onset")
        self._validate_observation(observation)
        self._pending_inputs -= 1
        # Navigation owns the observation, but its sound decided a held line:
        # words cut it as a qualified sound did before, and a sound without
        # words lets it go on.
        if isinstance(observation, CallerInput):
            if self._held_output is not None and self._held_output is self._play:
                self._cancel_output()
            self._held_output = None
            self._hold_released.set()
        elif not self._pending_inputs and not observation.continues:
            self._release_held_output()
        if not self._pending_inputs and self._partial is None:
            self._input_ready.set()

    async def on_input(self, observation: CallerInput | BlankInput) -> None:
        """Accept immutable input without waiting on business HTTP or playback.

        One bounded ordered queue holds complete, unsubmitted caller turns while
        the serial business actor owns an earlier request. Nothing mutates an
        already submitted request or processes an incomplete caller intent.
        """
        if self._closed:
            return
        if not self._started or self._pending_inputs < 1:
            error = WorkflowInputRefused("workflow input has no owned logical turn")
            self._fail(error)
            raise error
        self._validate_observation(observation)
        self._pending_inputs -= 1
        # Words said while the run waited at a transfer step answer nothing,
        # including a final recognized after the dial ended; they neither
        # cut the line that follows nor become a turn.
        retired = self._retire_before is not None and observation.turn_first_sample < self._retire_before
        if isinstance(observation, CallerInput) and not retired:
            # Even a short recognized "no" qualifies as an interruption. The
            # duration floor filters blank noise; it never discards words.
            self._held_output = None
            self._hold_released.set()
            self._cancel_output()
        try:
            observation = self._collect_input(observation)
        except WorkflowInputRefused as error:
            self._fail(error)
            raise
        if not self._pending_inputs and self._partial is None:
            # Every held caller sound resolved without words: the line goes on.
            self._release_held_output()
        if observation is not None and not retired:
            try:
                self._queue_input(observation)
            except WorkflowInputRefused as error:
                self._fail(error)
                raise
        if not self._pending_inputs and self._partial is None:
            self._input_ready.set()
        if self._waiting and (self._business is None or self._business.done()):
            self._business = asyncio.create_task(self._drain_inputs())
            self._business.add_done_callback(self._observe_business)
        elif (retired and not self._waiting and not self._pending_inputs and self._partial is None
              and (self._business is None or self._business.done())):
            # The retired words were what held the reply that follows the
            # transfer; nothing else will launch it now.
            self._business = asyncio.create_task(self._launch_after_retirement())
            self._business.add_done_callback(self._observe_business)

    def _validate_observation(self, observation):
        if (not isinstance(observation, (CallerInput, BlankInput))
                or type(observation.input_epoch) is not int or observation.input_epoch != self.fence.input_epoch
                or type(observation.first_sample) is not int or type(observation.end_sample) is not int
                or type(observation.turn_first_sample) is not int or type(observation.continues) is not bool
                or not 0 <= observation.turn_first_sample <= observation.first_sample
                or not 0 <= observation.first_sample < observation.end_sample <= self.fence.received_samples):
            error = WorkflowInputRefused("workflow input requires an immutable observation")
            self._fail(error)
            raise error

    def _queue_input(self, observation):
        merged = self._coalesce_waiting(self._waiting[-1] if self._waiting else None, observation)
        if merged is not None and self._waiting:
            self._waiting[-1] = merged
            return
        if len(self._waiting) >= MAX_QUEUED_CALLER_TURNS:
            raise WorkflowInputCapacityExceeded("workflow caller backlog requires a controlled call end")
        self._waiting.append(observation)

    def _coalesce_waiting(self, previous, observation):
        """Merge two complete turns, or None when a piece would change permission."""
        if previous is None:
            return observation
        if (observation.first_sample < max(0, previous.end_sample - self.fence.prefix_padding_samples)
                or observation.end_sample <= previous.end_sample):
            raise WorkflowInputRefused("queued caller inputs changed their original sample order")
        words = [item for item in (previous, observation) if isinstance(item, CallerInput)]
        first, end = min(previous.first_sample, observation.first_sample), observation.end_sample
        if not words:
            return BlankInput(observation.input_epoch, first, end, first, False)
        permission = words[0].answer_to_request_id
        # An earlier interrupt-only piece, or a blank before words, never strips
        # a later answer of its permission; the answer keeps its own origin. A
        # trailing blank can join only if its complete prefix is after the ACK.
        if (any(item.answer_to_request_id != permission for item in words)
                or permission is not None and (isinstance(previous, BlankInput)
                                               or not self.fence.retains_answer_permission(permission, first))):
            return None
        if sum(len(item.text) for item in words) + len(words) - 1 > MAX_CALLER_TURN_CHARACTERS:
            raise WorkflowInputCapacityExceeded("workflow caller text capacity requires a controlled call end")
        return CallerInput(" ".join(item.text for item in words), observation.input_epoch, first, end,
                           permission, first, False)

    def _observe_business(self, task: asyncio.Task) -> None:
        if task.cancelled():
            if not self._closed:
                self._fail(WorkflowInputRefused("workflow business actor ended without a joined call close"))
        elif (error := task.exception()) is not None:
            self._fail(error)

    def bind_input_owner(self, owner) -> None:
        """Bind the root's navigation actor before any business actor starts."""
        if (self._started or self._closed or self._input_owner is not None
                or not callable(getattr(owner, "join", None))
                or type(getattr(owner, "unresolved", None)) is not bool):
            raise WorkflowInputRefused("workflow has no fresh owned input barrier")
        self._input_owner = owner

    def validate_held_inputs(self, owner, observations):
        """Only the bound input actor can return its original observation batch."""
        if (self._closed or owner is not self._input_owner or self._input_owner is None
                or type(observations) is not tuple or not 1 <= len(observations) <= 768):
            raise WorkflowInputRefused("workflow handoff has no bounded owned original inputs")
        for observation in observations:
            self._validate_observation(observation)
        return observations

    def prepare_held_pickup(self, owner, observations):
        """Use the canonical collector for one complete pre-business handoff."""
        observations = self.validate_held_inputs(owner, observations)
        if self._started or self._partial is not None:
            raise WorkflowInputRefused("held pickup has no fresh pre-business owner")
        pickup = None
        for observation in observations:
            complete = self._collect_input(observation)
            if complete is not None:
                if (pickup := self._coalesce_waiting(pickup, complete)) is None:
                    raise WorkflowInputRefused("held pickup is not one interrupt-only caller turn")
        if self._partial is not None or not isinstance(pickup, CallerInput):
            raise WorkflowInputRefused("held pickup is not a complete original caller turn")
        return pickup

    def _navigation_unresolved(self):
        return self._input_owner is not None and self._input_owner.unresolved

    async def _join_navigation_inputs(self):
        if self._input_owner is not None:
            while not self._closed:
                await self._input_owner.join()
                if not self._navigation_unresolved():
                    return
                # A still-open recognizer or partial group belongs to capture;
                # never wait for that future input inside the capture actor.
                await asyncio.sleep(.01)

    async def _join_inputs(self) -> None:
        if self._input_owner is not None:
            await self._input_owner.join()
        if self._business is not None:
            await join_owned_task(self._business)

    async def _drain_inputs(self) -> None:
        while not self._closed and self._waiting:
            await self._floor_released.wait()
            await self._input_ready.wait()
            await self._join_navigation_inputs()
            async with self._turn_lock:
                if self._closed:
                    return
                if (self._floor_suspended or self._pending_inputs or self._partial is not None
                        or self._navigation_unresolved()):
                    continue
                await self._process_input(self._waiting.popleft())

    async def _process_input(self, observation: CallerInput | BlankInput) -> None:
        if self._terminal:
            # The graph has ended, but its closing reply may still be
            # held behind caller audio. Retire input without reopening
            # business processing, then deliver that reply exactly once.
            await self._launch_if_ready()
            return
        if isinstance(observation, BlankInput):
            if self._pending_inputs:
                return
            if not self._attempted:
                await self._launch_if_ready()
                return
            if self._reply is not None and self.fence.question_presented(self._reply["request_id"]):
                return
            if self._play is not None and not self._play.done() and not self._play.cancelling():
                return  # Short blank noise did not cancel the audible owner.
            # Retire the interrupted owner before a fresh re-presentation;
            # never replay its possibly exposed speech handle.
            await self._join_output()
            if self._pending_inputs or self._partial is not None or self._waiting:
                return
            if self._reply is not None and self.fence.question_presented(self._reply["request_id"]):
                return
            await self._submit(observation.wire(sample_rate=self.fence.sample_rate))
            return
        wire = observation.wire(sample_rate=self.fence.sample_rate)
        current = self._reply.get("request_id") if self._reply else None
        if wire["kind"] == "answer" and wire["answers_request_id"] != current:
            # Reclassify BEFORE allocating any logical request identity.
            # The original observation remains immutable. An in-flight or
            # retried submission is never changed, upgraded or downgraded.
            wire = {key: value for key, value in wire.items() if key != "answers_request_id"}
            wire["kind"] = "interrupt"
        await self._submit(wire)

    def _collect_input(self, observation: CallerInput | BlankInput) -> CallerInput | BlankInput | None:
        """A budget cut is not the end of a caller's intent or a business turn."""
        partial = self._partial
        if partial is None:
            if observation.turn_first_sample != observation.first_sample:
                raise WorkflowInputRefused("caller continuation has no owned beginning")
            partial = {"first": observation.first_sample, "end": observation.first_sample,
                       "parts": [], "characters": 0, "answer": None, "has_words": False}
        if (observation.turn_first_sample != partial["first"] or observation.first_sample != partial["end"]):
            raise WorkflowInputRefused("caller continuation changed its logical audio ownership")
        if isinstance(observation, CallerInput):
            text = observation.text
            if not isinstance(text, str) or not text.strip():
                raise WorkflowInputRefused("caller continuation has no final words")
            try:
                text.encode("utf-8")
            except UnicodeError:
                raise WorkflowInputRefused("caller continuation has malformed words") from None
            partial["characters"] += len(text) + bool(partial["parts"])
            if partial["characters"] > MAX_CALLER_TURN_CHARACTERS:
                raise WorkflowInputCapacityExceeded("workflow caller text capacity requires a controlled call end")
            if not partial["has_words"]:
                partial["answer"] = observation.answer_to_request_id
            elif partial["answer"] != observation.answer_to_request_id:
                partial["answer"] = None  # A later fragment can only lose permission.
            partial["has_words"] = True
            partial["parts"].append(text)
        partial["end"] = observation.end_sample
        if observation.continues:
            self._partial = partial
            return None
        self._partial = None
        first, end = partial["first"], partial["end"]
        if not partial["has_words"]:
            return BlankInput(observation.input_epoch, first, end, first, False)
        return CallerInput(" ".join(partial["parts"]), observation.input_epoch, first, end,
                           partial["answer"], first, False)

    async def _submit(self, wire: dict[str, Any]) -> None:
        if self._closed:
            return
        generation = self._generation
        task = self._http = asyncio.create_task(self.advance(wire))
        try:
            try:
                reply = await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await join_owned_task(task)
                except BaseException:
                    pass
                raise
            if self._closed:
                return
            previous_wait = (self._reply or {}).get("action_wait")
            self._reply, self._attempted = reply, False
            self._terminal = reply["status"] != "running"
            prior_waiter = self._action_waiter
            if prior_waiter is not None and prior_waiter is not asyncio.current_task():
                prior_waiter.cancel()
            if reply.get("action_wait") is not None:
                await self._join_output()
                if self.action_release is None or self.action_cancel is None or self.capture_barrier is None:
                    raise WorkflowInputRefused("workflow action has no capture release owner")
                if previous_wait != reply["action_wait"]:
                    action_id = reply["action_wait"]["action_id"]
                    # A queued blank is no caller words: it is retired once
                    # this actor reaches it, and cannot withhold the release.
                    words = any(isinstance(item, CallerInput) for item in self._waiting)
                    if generation == self._generation and not self._pending_inputs and not words:
                        await self.capture_barrier()
                        words = any(isinstance(item, CallerInput) for item in self._waiting)
                    if (not self._closed and not self.fence.closed and generation == self._generation
                            and not (self._pending_inputs or self._partial or words or self._floor_suspended
                                     or self._navigation_unresolved())):
                        await join_owned_task(asyncio.create_task(self.action_release(action_id, reply["request_id"])))
                    else:
                        await join_owned_task(asyncio.create_task(self.action_cancel(action_id)))
                self._action_waiter = asyncio.create_task(self._wait_action(dict(reply["action_wait"])))
                self._action_waiter.add_done_callback(self._observe_business)
                return
            if reply.get("transfer") is not None:
                await self._transfer(reply)
                return
            if not self._terminal:
                self.fence.begin_question(request_id=reply["request_id"],
                    turn_index=reply["runtime_state"]["turn_count"])
            # A caller START during the HTTP operation invalidates the old
            # permission to begin output, even when the transcript is delayed.
            if generation == self._generation and not self._pending_inputs and not self._waiting:
                await self._launch_if_ready()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            if task.done() and self._http is task:
                self._http = None

    async def _wait_action(self, binding: dict[str, Any]) -> None:
        """Poll pure state off the input actor; callers still interrupt normally."""
        if self.action_status is None:
            raise WorkflowInputRefused("workflow action has no result reader")
        while not self._closed:
            status = await self.action_status(binding["action_id"])
            if status.get("id") != binding["action_id"]:
                raise WorkflowInputRefused("workflow action status changed identity")
            if status.get("status") in {"succeeded", "failed", "unknown", "canceled", "superseded", "expired"}:
                await self._floor_released.wait()
                await self._input_ready.wait()
                async with self._turn_lock:
                    if self._closed or (self._reply or {}).get("action_wait") != binding:
                        return
                    if not (self._pending_inputs or self._partial or self._waiting or self._floor_suspended
                            or self._navigation_unresolved()):
                        await self._submit({"kind": "action_result", "action_id": binding["action_id"]})
                        return
            await asyncio.sleep(.5)

    async def _transfer(self, reply: dict[str, Any]) -> None:
        """Speak this reply's lines, place its dial, submit how the dial ended.

        Runs in the serial business actor under the turn lock, so no caller
        turn is submitted while the run waits at the transfer step. The
        reply's lines (an authored "connecting you" step) play first and are
        never presented as a question: the transfer was decided, and words the
        caller says over them or during the ring are not a turn. A barge-in
        cuts the line, never the transfer.
        """
        plan = reply["transfer"]
        if self._terminal or not isinstance(plan, dict) or self.transfer is None:
            raise WorkflowInputRefused("workflow transfer reply has no dial owner")
        # This actor is the reply's one output owner: a floor resumption or a
        # retired input must not launch it a second time.
        self._attempted = True
        await self._floor_released.wait()
        await self._join_output()
        if self._closed:
            return
        self._output_epoch += 1
        owner = self.output(reply, self._output_epoch)
        self._active_output = owner
        self._play = asyncio.create_task(self._speak(owner, reply))
        self._play.add_done_callback(lambda task: self._observe_output(task, terminal=False))
        await self._join_output()
        if self._closed:
            return
        # From the dial on, what the caller says answers nothing (a barge-in
        # during the connecting line above still cut that line).
        self._retire_before = float("inf")
        dial = self._dial = asyncio.create_task(self.transfer(dict(plan)))
        try:
            outcome = await dial
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if self._dial_cancelled and not (current is not None and current.cancelling()):
                return  # The call is ending: there is no outcome to report.
            raise
        finally:
            if self._dial is dial:
                self._dial = None
        # Caller words heard while the run waited are not an answer to anything.
        self._retire_before = self.fence.received_samples
        self._waiting.clear()
        await self._submit({"kind": "transfer_outcome", "outcome": outcome})
        # A ring-time onset seen during that request withheld the reply that
        # follows (``_submit``'s generation check); if it was retired, nothing
        # else will launch it.
        if (not self._closed and not self._waiting and not self._pending_inputs
                and self._partial is None):
            await self._launch_if_ready()

    def cancel_dial(self) -> None:
        """The call is ending: stop ringing anyone for it, now, before any join."""
        if self._dial is not None and not self._dial.done():
            self._dial_cancelled = True
            self._dial.cancel()

    async def _launch_after_retirement(self) -> None:
        async with self._turn_lock:
            if not self._closed:
                await self._launch_if_ready()

    async def pause_output(self) -> int:
        """Close new output, then join or park its existing owner before handoff.

        Input classification remains the root's responsibility. This operation
        never submits a caller turn or allocates another business request.
        """
        async with self._floor_lock:
            if self._closed:
                raise WorkflowInputRefused("closed workflow has no output floor")
            if self._floor_suspended:
                if not self._floor_ready:
                    raise WorkflowInputRefused("workflow floor has no joined pause acknowledgement")
                return self._floor_generation
            self._floor_suspended = True
            self._floor_released.clear()
            self._floor_ready = False
            self._floor_generation += 1
            owner, task = self._active_output, self._play
            if task is not None and not task.done():
                park = getattr(owner, "pause_for_floor", None)
                if callable(park):
                    try:
                        parked = await park()
                    except BaseException:
                        # Nothing was parked; the running output still owns the floor.
                        self._floor_suspended = False
                        self._floor_released.set()
                        raise
                    if parked is not True:
                        raise WorkflowInputRefused("workflow output did not acknowledge its parked floor")
                    self._preserved_output = owner
                else:
                    if getattr(owner, "floor_replay_unexposed", False) is not True:
                        raise WorkflowInputRefused("workflow output has no durable floor pause port")
                    # Non-persistent output ports join completely. A production
                    # exposed speech owner must supply the durable park port.
                    self._floor_cancelled.add(task)
                    self._cancel_output()
                    await self._join_output()
                    self._attempted = False
            self._floor_ready = True
            return self._floor_generation

    async def resume_output(self, generation: int) -> bool:
        """Resume this exact paused generation once, without another advance."""
        async with self._floor_lock:
            if (type(generation) is not int or generation != self._floor_generation
                    or not self._floor_suspended or not self._floor_ready or self._closed):
                return False
            preserved = self._preserved_output
            if preserved is not None:
                task = self._play
                if task is None or task.done() or task.cancelling():
                    # A newly admitted person turn can retire the parked line.
                    # Join its real final fact; never briefly resume old audio
                    # before the queued business turn acquires the actor.
                    await self._join_output()
                elif await preserved.resume_from_floor() is not True:
                    raise WorkflowInputRefused("workflow output did not acknowledge its resumed floor")
                self._preserved_output = None
            if self._closed:
                return False
            self._floor_suspended = self._floor_ready = False
            self._floor_released.set()
            await self._launch_if_ready()
            return True

    def _cancel_output(self) -> None:
        if self._play is not None and not self._play.done() and not self._play.cancelling():
            self._play.cancel()

    async def _join_output(self) -> None:
        task = self._play
        if task is None:
            return
        try:
            await join_owned_task(task)
        except asyncio.CancelledError:
            # A cancelled OUTPUT is an expected interruption; cancellation of
            # this join's caller still propagates after its owned child joins.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            if task.done() and self._play is task:
                self._play = None

    async def _launch_if_ready(self) -> None:
        if self._closed or self._floor_suspended or self._pending_inputs or self._partial is not None or self._waiting or self._reply is None or self._attempted:
            return
        if self._reply.get("action_wait") is not None:
            return
        await self._join_output()
        if self._closed or self._floor_suspended or self._pending_inputs or self._partial is not None or self._waiting:
            return
        self._attempted = True
        self._output_epoch += 1
        try:
            owner = self.output(self._reply, self._output_epoch)
        except BaseException as error:
            self._fail(error)
            raise
        self._active_output = owner
        self._play = asyncio.create_task(self._speak(owner, self._reply))
        # This callback also sees cancellation BEFORE the coroutine starts.
        terminal = self._terminal
        self._play.add_done_callback(lambda task: self._observe_output(task, terminal=terminal))

    def _observe_output(self, task: asyncio.Task, *, terminal: bool) -> None:
        if task in self._floor_cancelled:
            self._floor_cancelled.discard(task)
            if not task.cancelled() and task.exception() is not None:
                self._fail(task.exception())
            return
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self._fail(error)
        if terminal:
            self._note_terminal(not task.cancelled() and task.exception() is None and task.result())

    def _note_terminal(self, completed: bool) -> None:
        if not self._closed and not self._terminal_notified:
            self._terminal_notified = True
            self.on_terminal(completed)

    async def _speak(self, owner: ReplyOutput, reply: dict[str, Any]) -> bool:
        status = await owner.run()
        if status == "not_voiced" and reply["status"] != "running":
            # An ending the caller did not hear in full: that line is on
            # record as not completed, the lines before it that were heard are
            # recorded as heard, and the call ends as the run did.
            return True
        if status not in ("completed_local", "interrupted"):
            raise WorkflowInputRefused("workflow output has no acknowledged lifecycle result")
        completed = status == "completed_local"
        if completed and reply["status"] == "running" and reply.get("transfer") is None:
            # A caller sound after the last frame decides first: words cut
            # this line before its presentation, as before; a blank presents it.
            await self._join_caller_hold()
            frontier = await self.fence.frontier()
            presentation = self.fence.prepare_question_presentation(
                request_id=reply["request_id"], input_epoch=self.fence.input_epoch, frontier=frontier)
            await owner.present(presentation.wire())
            await self._join_caller_hold()
            self.fence.completed_local_question(presentation)
        return completed

    async def _join_caller_hold(self) -> None:
        while self._held_output is asyncio.current_task():
            await self._hold_released.wait()

    async def aclose(self) -> None:
        if self._close is None:
            self._closed = True
            self._floor_released.set()
            self.fence.close()
            self._partial = None
            self._waiting.clear()
            self._input_ready.set()
            self._cancel_output()
            async def close():
                errors = []
                # The call is over: nobody is dialled for it any longer.
                self.cancel_dial()
                if self._action_waiter is not None:
                    self._action_waiter.cancel()
                    try:
                        await self._action_waiter
                    except asyncio.CancelledError:
                        pass
                    except BaseException as error:
                        errors.append(error)
                try:
                    await self._join_output()
                except BaseException as error:
                    errors.append(error)
                try:
                    await self._join_inputs()
                except asyncio.CancelledError:
                    # A cancelled actor has already reported its failure.
                    # The close owner itself is shielded and still joins it.
                    pass
                except BaseException as error:
                    errors.append(error)
                if self._http is not None:
                    try:
                        await join_owned_task(self._http)
                    except BaseException as error:
                        errors.append(error)
                for task in tuple(self._action_cancellations):
                    try:
                        await join_owned_task(task)
                    except BaseException as error:
                        errors.append(error)
                # A line the output played but did not say in full is the
                # call's failure, reported once when it happened; every owner
                # had sealed its line before it, so closing finds that same
                # object again and it is not a cleanup failure. Any other
                # failure is still raised here as before.
                if isinstance(self._reported_failure, LineNotVoiced):
                    errors = [error for error in errors if error is not self._reported_failure]
                if errors:
                    raise errors[0]
            self._close = asyncio.create_task(close())
        await join_owned_task(self._close)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One processed detector clock owns the first governed workflow turn."""
from __future__ import annotations

import asyncio
import math
import time

try:
    from .workflow_input import BlankInput, CallerInput, WorkflowInputRefused, WorkflowInputCapacityExceeded
    from .workflow_tasks import join_owned_task
    from .opening_turn import FEED_GAP_SECONDS, LIVE_EDGE_TOLERANCE_SECONDS, OBSERVATION_RECOVERY_SECONDS, OpeningObservationTimeout
except ImportError:
    from workflow_input import BlankInput, CallerInput, WorkflowInputRefused, WorkflowInputCapacityExceeded
    from workflow_tasks import join_owned_task
    from opening_turn import FEED_GAP_SECONDS, LIVE_EDGE_TOLERANCE_SECONDS, OBSERVATION_RECOVERY_SECONDS, OpeningObservationTimeout


class WorkflowPickup:
    """``observer`` is evidence only (the opening summary): it is told what the
    pickup did and can never change it; a raising observer is ignored."""

    def __init__(self, call, *, silence_wait_seconds, monotonic=time.monotonic, observer=None):
        if (type(silence_wait_seconds) not in (int, float)
                or not math.isfinite(silence_wait_seconds) or not 1 <= silence_wait_seconds <= 5):
            raise WorkflowInputRefused("workflow pickup has no frozen silence window")
        self.call, self.monotonic = call, monotonic
        self._window = math.ceil(silence_wait_seconds * call.fence.sample_rate)
        self._origin = self._samples = self._onsets = 0
        self._last_progress = monotonic()
        self._recovery_deadline = None
        self._closed = False
        self.startup = None
        self.observer = observer

    def _observe(self, name, *args):
        method = getattr(self.observer, name, None) if self.observer is not None else None
        if method is not None:
            try:
                method(*args)
            except Exception:  # noqa: BLE001 - evidence never decides the pickup
                pass

    def _start(self, pickup=None, reason=None):
        # start() claims synchronously. The credit replay has no suspension,
        # so the business actor cannot observe half of the capture handoff.
        waiter = self.call.start(pickup=pickup)
        self._observe("opening_requested", reason, pickup)
        self.call.replay_prestart_onsets(self._onsets)
        self._onsets = 0
        self.startup = asyncio.create_task(waiter)
        self.startup.add_done_callback(lambda task: None if task.cancelled() else task.exception())

    def agent_first(self):
        if self._closed or self.call._started:
            raise WorkflowInputRefused("workflow opening cannot be started twice")
        self._start()

    async def on_speech(self):
        if self._closed:
            raise WorkflowInputRefused("workflow pickup is closed")
        self._observe("onset")
        if self.call._started:
            await self.call.on_speech()
        else:
            self._onsets += 1
            if self._onsets > 2:
                raise WorkflowInputCapacityExceeded("workflow pickup exceeded its owned source backlog")

    async def on_interruption(self):
        if self.call._started:
            await self.call.on_caller_interruption()

    def retire_held_input(self, observation):
        """Give one original capture observation to navigation, never business."""
        if self._closed:
            raise WorkflowInputRefused("workflow pickup is closed")
        if self.call._started:
            self.call.retire_held_input(observation)
            return
        if self._onsets < 1:
            raise WorkflowInputRefused("held workflow pickup has no owned onset")
        self.call._validate_observation(observation)
        self._onsets -= 1
        self._origin = max(self._origin, observation.end_sample)

    async def release_held_inputs(self, owner, observations):
        """Atomically return the actor's originals; never wait for business HTTP."""
        if self._closed:
            raise WorkflowInputRefused("workflow pickup is closed")
        observations = self.call.validate_held_inputs(owner, observations)
        if not self.call._started:
            pickup = self.call.prepare_held_pickup(owner, observations)
            for observation in observations:
                self._observe("pickup_observation", observation)
            self._start(pickup, "after_greeting")
            return
        # on_speech/on_input only enqueue into the serial actor. This batch
        # contains no suspension on provider or business operations, so the
        # canonical collector coalesces it before its business task can run.
        for observation in observations:
            await self.call.on_speech()
            await self.call.on_input(observation)

    async def on_input(self, observation, _language_hint=None):
        if self._closed:
            return
        if self.call._started:
            await self.call.on_input(observation)
            return
        if self._onsets < 1:
            raise WorkflowInputRefused("workflow pickup has no owned caller onset")
        self.call._validate_observation(observation)
        if observation.continues or observation.turn_first_sample != observation.first_sample:
            raise WorkflowInputRefused("workflow pickup cannot consume a partial caller turn")
        self._onsets -= 1
        self._observe("pickup_observation", observation)
        if isinstance(observation, BlankInput):
            self._origin = max(self._origin, observation.end_sample)
            return
        if not isinstance(observation, CallerInput) or observation.answer_to_request_id is not None:
            raise WorkflowInputRefused("workflow pickup must be interrupt-only")
        self._start(observation, "after_greeting")

    def on_clock(self, tick):
        if self._closed:
            return
        if (type(tick.samples) is not int or tick.samples < self._samples
                or type(tick.received_samples) is not int or tick.received_samples < tick.samples
                or type(tick.physical_speech) is not bool or type(tick.candidate_speech) is not bool
                or type(tick.pending_sources) is not int or not 0 <= tick.pending_sources <= 2):
            raise WorkflowInputRefused("workflow pickup has no ordered processed clock")
        if tick.samples > self._samples:
            self._samples, self._last_progress = tick.samples, self.monotonic()
        if self.call._started:
            return
        live = tick.received_samples - tick.samples <= LIVE_EDGE_TOLERANCE_SECONDS * self.call.fence.sample_rate
        if (live and not self._onsets and not tick.physical_speech and not tick.candidate_speech
                and not tick.pending_sources and tick.samples >= self._origin + self._window):
            self._start(reason="silence_timeout")

    def check_stall(self):
        if self._closed or self.call._started:
            return
        now = self.monotonic()
        if now - self._last_progress < FEED_GAP_SECONDS:
            return
        if self._recovery_deadline is None:
            self._recovery_deadline = self._last_progress + FEED_GAP_SECONDS + OBSERVATION_RECOVERY_SECONDS
        if now >= self._recovery_deadline:
            raise OpeningObservationTimeout("workflow caller observation did not catch up")

    async def aclose(self):
        self._closed = True
        if self.startup is not None:
            await join_owned_task(self.startup)

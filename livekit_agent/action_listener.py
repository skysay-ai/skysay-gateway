# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Finite caller capture alongside an unchanged free-form conversation engine."""
from __future__ import annotations

import asyncio

from livekit import rtc
from livekit.agents.utils.participant import wait_for_track_publication

try:
    from .capabilities.workflow_listener import validate_frozen_voice_input
    from .capabilities.workflow_sources import qualified_source_adapter
    from .workflow_listener_source import WorkflowListenerSource
    from .workflow_room_input import WorkflowRoomInput
    from .workflow_detector import load_workflow_detector
    from .workflow_speech import SpeechRefused
    from .workflow_tasks import join_owned_task
    from .source_keys import managed_keys_for_sources
except ImportError:
    from capabilities.workflow_listener import validate_frozen_voice_input
    from capabilities.workflow_sources import qualified_source_adapter
    from workflow_listener_source import WorkflowListenerSource
    from workflow_room_input import WorkflowRoomInput
    from workflow_detector import load_workflow_detector
    from workflow_speech import SpeechRefused
    from workflow_tasks import join_owned_task
    from source_keys import managed_keys_for_sources


async def _ignore(*_args):
    pass


class ConfirmationListener:
    def __init__(self, *, call, inputs, environment):
        policy = call.config.get("action_voice_input")
        if type(policy) is not dict:
            raise SpeechRefused("write-enabled call has no admitted confirmation listener")
        self.policy = validate_frozen_voice_input(policy, required_languages=policy["required_languages"])
        constructor = self.policy["constructor"]
        declaration = qualified_source_adapter("listener", constructor["provider"], constructor["model"])
        for language in constructor["languages"]:
            if qualified_source_adapter("listener", constructor["provider"], constructor["model"], locale=language) is not declaration:
                raise SpeechRefused("confirmation listener language is not qualified")
        self.managed = managed_keys_for_sources([(declaration, self.policy["selection"]["listener"])],
            environment=environment, leased_components={})
        self.detector = load_workflow_detector(self.policy)
        self.call, self.inputs = call, inputs
        self.input = self.task = self._closing = None
        self._barriers = {}
        inputs.capture_barrier = self.barrier
        inputs.mic_clock = self.mic_position

    async def barrier(self):
        """Prove one finite microphone watermark, never chase live silence."""
        if self.input is None or self._closing is not None or self.inputs.failed:
            raise SpeechRefused("confirmation capture is unavailable")
        fence = self.input.fence
        target = await fence.frontier()
        # The frontier yields; a capture that ended in between refused every barrier it saw.
        if self._closing is not None or self.inputs.failed or self.task is None or self.task.done():
            raise SpeechRefused("confirmation capture is unavailable")
        future = asyncio.get_running_loop().create_future()
        self._barriers[future] = target
        try:
            await future
        finally:
            self._barriers.pop(future, None)

    def mic_position(self):
        """(live edge, queued) microphone samples now: the clock caller onsets use.

        The live edge is every sample the ingress delivered, consumed or still
        queued; consumption is FIFO, so it indexes like the onsets. Capture's
        consumed count lags it by any backlog and would stamp playout early.
        """
        if self.input is None:
            return None
        return self.input.ingress.delivered_samples, self.input.fence.queued_audio_samples

    async def start(self, *, room, participant):
        if self.task is not None or self._closing is not None:
            raise SpeechRefused("confirmation listener cannot restart")
        publication = await asyncio.wait_for(wait_for_track_publication(room,
            identity=participant.identity, kind=rtc.TrackKind.KIND_AUDIO), timeout=3)
        publication.set_subscribed(True)
        selected = await asyncio.wait_for(wait_for_track_publication(room,
            identity=participant.identity, kind=rtc.TrackKind.KIND_AUDIO, wait_for_subscription=True), timeout=3)
        if selected.sid != publication.sid or selected.track is None:
            raise SpeechRefused("confirmation microphone changed while subscribing")
        rate = self.policy["capture"]["sample_rate"]
        async def source_factory(*, first_sample, sample_rate):
            if self.inputs.failed or self.inputs.closed or sample_rate != rate or type(first_sample) is not int or first_sample < 0:
                raise SpeechRefused("confirmation source has no active caller owner")
            return await WorkflowListenerSource.create(journal=self.call.confirmation_source_journal(),
                policy=self.policy, managed=self.managed)
        def clock(value):
            if not value.candidate_speech and not value.physical_speech:
                for future, target in tuple(self._barriers.items()):
                    if value.samples >= target and not future.done():
                        future.set_result(None)
        selection = self.policy["selection"]
        self.input = WorkflowRoomInput(room=room, participant=participant, track=selected.track, sample_rate=rate,
            input_epoch=1, prefix_padding_samples=rate * self.policy["vad"]["prefix_padding_ms"] // 1000,
            detector=self.detector, source_factory=source_factory, silence_hold_ms=selection["silence_hold_ms"],
            interruption_min_duration_ms=selection["interruption_min_duration_ms"],
            on_speech=_ignore, on_interruption=_ignore, on_input=_ignore,
            source_observer=self.inputs, on_clock=clock)
        async def capture():
            try:
                await self.input.run()
            finally:
                self.inputs.unavailable()
                for future in tuple(self._barriers):
                    if not future.done():
                        future.set_exception(SpeechRefused("confirmation capture ended"))
        self.task = asyncio.create_task(capture())
        self.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        started = asyncio.create_task(self.input.started.wait())
        try:
            await asyncio.wait((started, self.task), return_when=asyncio.FIRST_COMPLETED)
            if self.task.done():
                await self.task
        finally:
            started.cancel()
            await asyncio.gather(started, return_exceptions=True)

    async def aclose(self):
        if self._closing is None:
            self.inputs.unavailable()
            async def close():
                if self.input is not None:
                    await self.input.aclose()
                if self.task is not None:
                    await asyncio.gather(self.task, return_exceptions=True)
                await self.inputs.aclose()
                # Silero owns no async detector close API. Capture has joined
                # its VAD stream before the detector can be released.
                self.detector = None
            self._closing = asyncio.create_task(close())
        await join_owned_task(self._closing)

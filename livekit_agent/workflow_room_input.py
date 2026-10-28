# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One selected remote microphone, retired by official Room lifecycle events."""
from __future__ import annotations

import asyncio

from livekit import rtc

try:
    from .workflow_input import WorkflowInputRefused, WorkflowMediaEnded
    from .workflow_input_capture import WorkflowInputCapture, WorkflowRtcIngress
    from .workflow_tasks import join_owned_task
except ImportError:
    from workflow_input import WorkflowInputRefused, WorkflowMediaEnded
    from workflow_input_capture import WorkflowInputCapture, WorkflowRtcIngress
    from workflow_tasks import join_owned_task


class WorkflowRoomInput:
    """The call root owns this once and ends the call when ``run`` fails.

    A track's unpublication is not an AudioStream EOF contract. The official
    selected-track, participant and room events explicitly cancel and join the
    capture owner. The borrowed VAD outlives this input; fresh STT sources belong to capture; no switch of
    an output stack can close or replace the caller's microphone here.
    """

    def __init__(self, *, room: rtc.Room, participant, track, sample_rate: int,
                 input_epoch: int, prefix_padding_samples: int, detector, source_factory,
                 silence_hold_ms: int, interruption_min_duration_ms: int,
                 on_speech, on_interruption, on_input, on_clock=None, on_frame=None, source_observer=None,
                 on_failure=None, listener_reconnect=None) -> None:
        self.room, self.participant, self.track = room, participant, track
        self.source_observer = source_observer
        self._barriers = {}
        self._assert_selected()
        # ``on_failure`` hears a capture failure before its cleanup joins, and
        # an ingress overflow from inside the SDK's enqueue: either way the
        # call's owner latches its ending before anything else can.
        self.ingress = WorkflowRtcIngress(track, sample_rate=sample_rate, input_epoch=input_epoch,
                                          prefix_padding_samples=prefix_padding_samples, on_frame=on_frame,
                                          on_overflow=on_failure)
        self.fence = self.ingress.fence
        def capture_clock(value):
            if not value.candidate_speech and not value.physical_speech:
                for future, target in tuple(self._barriers.items()):
                    if value.samples >= target and not future.done():
                        future.set_result(None)
            if on_clock is not None:
                on_clock(value)
        self.capture = WorkflowInputCapture(fence=self.fence, detector=detector, source_factory=source_factory,
            inference_sample_rate=16000, silence_hold_ms=silence_hold_ms,
            interruption_min_duration_ms=interruption_min_duration_ms,
            on_speech=on_speech, on_interruption=on_interruption, on_input=on_input, on_clock=capture_clock, source_observer=source_observer,
            on_failure=on_failure, listener_reconnect=listener_reconnect)
        self.started = asyncio.Event()
        self._used = False
        self._task: asyncio.Task | None = None
        self._closing: asyncio.Task | None = None
        self._handlers: list[tuple[str, object]] = []
        self._ended: asyncio.Future | None = None

    async def barrier(self):
        """Drain one finite microphone watermark: everything delivered, plus slack."""
        if self.fence.closed or self._closing is not None or not self.started.is_set():
            raise WorkflowMediaEnded("workflow capture is unavailable")
        target = await self.fence.frontier()
        # The frontier yields; an end in between already refused every barrier it saw.
        if self.fence.closed or self._closing is not None:
            raise WorkflowMediaEnded("workflow capture is unavailable")
        future = asyncio.get_running_loop().create_future()
        # After a long loop stall the target includes audio still queued in
        # the ingress; a detector that cannot judge it within the timeout
        # refuses the release rather than releasing over audio it never saw.
        self._barriers[future] = target
        try:
            async with asyncio.timeout(2):
                await future
        finally:
            self._barriers.pop(future, None)

    def _assert_selected(self) -> None:
        if (self.room.remote_participants.get(self.participant.identity) is not self.participant
                or self.track.kind != rtc.TrackKind.KIND_AUDIO
                or not any(publication.track is self.track and publication.sid == self.track.sid
                           for publication in self.participant.track_publications.values())):
            raise WorkflowInputRefused("workflow microphone is not a current remote audio publication")

    def _end(self) -> None:
        # Close eligibility synchronously at the event, before the consumer
        # can process a delayed final transcript or authorize another answer.
        self.fence.close()
        for future in tuple(self._barriers):
            if not future.done():
                future.set_exception(WorkflowMediaEnded("workflow capture ended"))
        if self.source_observer is not None:
            self.source_observer.unavailable()
        if self._ended is not None and not self._ended.done():
            self._ended.set_result(None)

    def _listen(self, name, callback) -> None:
        self.room.on(name, callback)
        self._handlers.append((name, callback))

    async def run(self) -> None:
        if self._used or self._closing is not None:
            raise WorkflowInputRefused("workflow room input cannot restart")
        self._used = True
        try:
            self._assert_selected()
            self._ended = asyncio.get_running_loop().create_future()

            def unsubscribed(track, publication, participant):
                if (participant is self.participant and publication.sid == self.track.sid
                        and track.sid == self.track.sid):
                    self._end()

            def participant_left(participant):
                if participant is self.participant:
                    self._end()

            self._listen("track_unsubscribed", unsubscribed)
            self._listen("participant_disconnected", participant_left)
            self._listen("disconnected", lambda *_: self._end())
            self._task = asyncio.create_task(self.capture.run(self.ingress))
            self.started.set()
            await asyncio.wait((self._task, self._ended), return_when=asyncio.FIRST_COMPLETED)
            if self._task.done():
                await self._task
            raise WorkflowMediaEnded("workflow caller media ended")
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closing is None:
            self._end()
            for name, callback in self._handlers:
                self.room.off(name, callback)
            self._handlers.clear()

            async def close():
                if self._task is not None:
                    self._task.cancel()
                    await asyncio.gather(self._task, return_exceptions=True)
                await self.ingress.aclose()

            self._closing = asyncio.create_task(close())
        await join_owned_task(self._closing)

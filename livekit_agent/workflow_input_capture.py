# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Uncalled root-owned continuous input: bounded VAD to isolated recognition.

Audio and question clocks belong to one input epoch. No SDK manual-commit
text, generated caption, model decision or wall-clock callback grants an
answer. The root supplies funded components and owns playback interruption.
"""
from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from dataclasses import dataclass
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

from livekit import rtc
from livekit.agents import vad

try:
    from .workflow_tasks import join_owned_task
    from .capabilities.workflow_listener import VAD_ACTIVATION_THRESHOLD
    from .workflow_input import (BlankInput, CallerInput, CallerTurnTiming, WorkflowInputFence, WorkflowInputRefused,
        WorkflowInputCapacityExceeded, WorkflowInputBacklogExceeded, WorkflowMediaEnded,
        MAX_VAD_SPEECH_BUFFER_SECONDS, MAX_CALLER_INGRESS_BACKLOG_SECONDS)
    from .workflow_input_recognition import CallerLanguageHint, CallerRecognition
    from .workflow_input_ring import CallerPcmRing, ListenerReconnect, LogicalCallerSource
except ImportError:  # flat worker image
    from workflow_tasks import join_owned_task
    from capabilities.workflow_listener import VAD_ACTIVATION_THRESHOLD
    from workflow_input import (BlankInput, CallerInput, CallerTurnTiming, WorkflowInputFence, WorkflowInputRefused,
        WorkflowInputCapacityExceeded, WorkflowInputBacklogExceeded, WorkflowMediaEnded,
        MAX_VAD_SPEECH_BUFFER_SECONDS, MAX_CALLER_INGRESS_BACKLOG_SECONDS)
    from workflow_input_recognition import CallerLanguageHint, CallerRecognition
    from workflow_input_ring import CallerPcmRing, ListenerReconnect, LogicalCallerSource


class CallerSourceObserver(Protocol):
    """Synchronous ownership notifications, never reconstructed from captions."""
    def begin_turn(self, *, onset_sample: int) -> object: ...
    def seal_turn(self, token: object) -> None: ...
    def completed_turn(self, token: object, text: str) -> None: ...
    def blank_turn(self, token: object) -> None: ...
    def unavailable(self) -> None: ...


class WorkflowAudioIngress(Protocol):
    fence: WorkflowInputFence
    # Resolves with a WorkflowInputBacklogExceeded once the ingress backlog
    # passes its cap; capture ends with it.
    overflow: asyncio.Future
    # Samples handed to this process so far, consumed or still queued: the
    # live edge. It equals ``fence.received_samples`` only with no backlog.
    delivered_samples: int

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkflowCaptureClock:
    """Observed in the one detector actor, after its current event is applied."""
    samples: int
    physical_speech: bool
    candidate_speech: bool
    pending_sources: int
    received_samples: int


def _delivery_counter(admit: Callable[[int], None]):
    """A pass-through SDK frame processor that counts each frame as it is enqueued.

    The pinned SDK's ``AudioStream._run`` calls its processor synchronously,
    on the loop, immediately before putting the frame on the stream's queue,
    and passes the frame through if the processor raises. ``admit`` never
    raises and never alters the frame.
    """
    class DeliveryCounter(rtc.FrameProcessor):
        @property
        def enabled(self) -> bool:
            return True

        @enabled.setter
        def enabled(self, value: bool) -> None:
            pass

        def _process(self, frame):
            admit(frame.samples_per_channel)
            return frame

        def _close(self) -> None:
            pass
    return DeliveryCounter()


def _voiced_end(event, *, first: int, end: int, rate: int) -> int | None:
    """The END event's sample less the trailing silence the VAD itself measured, or None."""
    try:
        silence = getattr(event, "silence_duration", None)
        if type(silence) not in (int, float) or not math.isfinite(silence) or silence < 0:
            return None
        voiced = end - int(round(silence * rate))
    except (ArithmeticError, ValueError):  # a diagnostic cannot end the capture
        return None
    return voiced if first <= voiced <= end else None


def _timed(observation: CallerInput, result: CallerRecognition, marks: CallerTurnTiming) -> CallerInput:
    """The delivered turn with capture's marks and its listener's; evidence never fails a turn."""
    try:
        timing = result.timing
        timing = (dataclasses.replace(timing, sealed_at=marks.sealed_at, voiced_end_sample=marks.voiced_end_sample)
                  if dataclasses.is_dataclass(timing) and hasattr(timing, "voiced_end_sample") else marks)
        return dataclasses.replace(observation, timing=timing)
    except Exception:  # noqa: BLE001 - a diagnostic cannot cost the caller's words
        return observation


class WorkflowRtcIngress:
    """Own the official RTC stream; never discard caller audio it delivered.

    The pinned SDK drains its native event queue into the stream's ring
    without yielding. A two-frame ring therefore discarded all but two frames
    after any loop stall, and the remaining audio was spliced into one caller
    turn. The ring now holds ``MAX_CALLER_INGRESS_BACKLOG_SECONDS``
    of frames, and a delivery counter that runs before each enqueue latches a
    controlled capacity failure the moment the backlog would exceed that: no
    frame is ever evicted from a call that is still listening.

    The question boundary and release barriers are the delivered frontier
    (``fence.frontier``), not the consumed count. The instant that matters is
    when the frontier is requested, after the question's completed_local
    acknowledgement: audio whose native callback was queued on the loop by
    then is drained through the counter before the read, by two scheduler
    turns, and audio arriving later arrived after the question. The first
    turn runs every pending FFI callback (each puts its event on the stream's
    FFI queue and wakes ``_run``); the second runs ``_run``, which enqueues
    them all without yielding. Audio still
    inside the native SDK is the fence's ``queued_audio_samples`` slack, as
    before. The SDK's iterator stops at end of stream once its producer task
    is done, abandoning anything still queued, so an end that leaves
    delivered audio unconsumed is a media end, never a clean end of input.
    A changed queue topology needs a fresh proof
    (``workflow_playback_selftest``). This is local input observation, not
    human audibility.
    """

    def __init__(self, track: rtc.Track, *, sample_rate: int, input_epoch: int, prefix_padding_samples: int,
                 on_frame: Callable[[], None] | None = None,
                 on_overflow: Callable[[WorkflowInputBacklogExceeded], None] | None = None) -> None:
        if sample_rate not in (8000, 16000, 24000, 48000):
            raise WorkflowInputRefused("caller RTC input has no proven sample format")
        self.fence = WorkflowInputFence(sample_rate=sample_rate, input_epoch=input_epoch,
                                        queued_audio_samples=2 * (sample_rate // 50),
                                        prefix_padding_samples=prefix_padding_samples,
                                        ingress_frontier=self._frontier)
        capacity = MAX_CALLER_INGRESS_BACKLOG_SECONDS * 50
        self._cap_samples = capacity * (sample_rate // 50)
        self._delivered = self._consumed = 0
        self.overflow = asyncio.get_running_loop().create_future()
        self._on_overflow = on_overflow
        self._stream = rtc.AudioStream(track, sample_rate=sample_rate, num_channels=1,
                                       capacity=capacity, frame_size_ms=20,
                                       noise_cancellation=_delivery_counter(self._admit))
        self._closing: asyncio.Task | None = None
        # Evidence only (the caller-audio flow log): told of each frame that
        # reached recognition, synchronously and before the consumer sees it.
        self._on_frame = on_frame

    def _admit(self, samples) -> None:
        """Runs inside the SDK's enqueue: count, and latch the cap before any eviction."""
        self._delivered += samples
        backlog = self._delivered - self._consumed
        if backlog > self._cap_samples and not self.overflow.done():
            error = WorkflowInputBacklogExceeded(backlog_samples=backlog, sample_rate=self.fence.sample_rate)
            self.overflow.set_result(error)
            if self._on_overflow is not None:
                try:
                    self._on_overflow(error)
                except Exception:  # noqa: BLE001 - the latch stands; never raise into the SDK
                    pass

    async def _frontier(self) -> int:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return self._delivered

    @property
    def delivered_samples(self) -> int:
        return self._delivered

    async def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        if self._closing is not None:
            raise WorkflowInputRefused("caller RTC input is already closing")
        async for event in self._stream:
            frame = event.frame
            if (frame.samples_per_channel != self.fence.sample_rate // 50
                    or frame.sample_rate != self.fence.sample_rate or frame.num_channels != 1):
                raise WorkflowInputRefused("caller RTC input changed its fixed frame format")
            self._consumed += frame.samples_per_channel
            if self._on_frame is not None:
                try:
                    self._on_frame()
                except Exception:  # noqa: BLE001 - a diagnostic cannot break caller media
                    pass
            yield frame
        if self._delivered != self._consumed:
            raise WorkflowMediaEnded("caller RTC input ended with delivered audio unconsumed")

    async def aclose(self) -> None:
        self.fence.close()
        if self._closing is None:
            self._closing = asyncio.create_task(self._stream.aclose())
        await join_owned_task(self._closing)


class WorkflowInputCapture:
    """One VAD stream and fresh granted recognition source per logical turn.

    The root marks a question complete on ``fence`` only after every segment's
    completed_local acknowledgement. ``on_speech`` holds new exposure and
    ``on_interruption`` requests cancellation; the output owner joins its SDK
    handle before replacement. ``on_input`` is serialized and still requires
    the control-plane runner's independent question/presentation check.
    ``on_failure`` hears a failure synchronously, before owned cleanup is
    joined, so the call's owner can latch its ending first. ``listener_reconnect``
    is the call's one reconnect of a listener that never acknowledged; when its
    bound expires the refusal reaches ``on_failure`` from the bound's own timer.
    """

    def __init__(self, *, fence: WorkflowInputFence, detector: vad.VAD,
                 source_factory, inference_sample_rate: int,
                 silence_hold_ms: int, interruption_min_duration_ms: int,
                 on_speech: Callable[[], Awaitable[None]],
                 on_interruption: Callable[[], Awaitable[None]],
                 on_input: Callable[[CallerInput | BlankInput, CallerLanguageHint | None], Awaitable[None]],
                 on_clock: Callable[[WorkflowCaptureClock], None] | None = None,
                 source_observer: CallerSourceObserver | None = None,
                 on_failure: Callable[[BaseException], None] | None = None,
                 listener_reconnect: ListenerReconnect | None = None) -> None:
        if (fence.sample_rate not in (8000, 16000, 24000, 48000)
                or type(inference_sample_rate) is not int or inference_sample_rate != 16000
                or type(silence_hold_ms) is not int or not 300 <= silence_hold_ms <= 3000
                or type(interruption_min_duration_ms) is not int or not 0 <= interruption_min_duration_ms <= 1000):
            raise WorkflowInputRefused("caller VAD has no proven sample-clock binding")
        if not callable(source_factory):
            raise WorkflowInputRefused("caller input requires its granted source factory")
        self.fence, self.detector, self.source_factory = fence, detector, source_factory
        self.inference_sample_rate = inference_sample_rate
        self.on_speech, self.on_interruption, self.on_input = on_speech, on_interruption, on_input
        self.on_clock = on_clock
        self.source_observer = source_observer
        self.on_failure = on_failure
        self.listener_reconnect = listener_reconnect
        self.silence_samples = silence_hold_ms * fence.sample_rate // 1000
        self.interruption_samples = interruption_min_duration_ms * fence.sample_rate // 1000
        self._used = False

    async def run(self, audio: WorkflowAudioIngress) -> None:
        if self._used or self.fence.closed or getattr(audio, "fence", None) is not self.fence:
            raise WorkflowInputRefused("caller capture owner cannot be restarted")
        overflow = getattr(audio, "overflow", None)
        if not isinstance(overflow, asyncio.Future) or type(getattr(audio, "delivered_samples", None)) is not int:
            raise WorkflowInputRefused("caller capture requires its ingress backlog bound")
        self._used = True
        try:
            stream = self.detector.stream()
        except BaseException as error:
            self.fence.close()
            if self.on_failure is not None and isinstance(error, Exception):
                try:
                    self.on_failure(error)
                except Exception:  # noqa: BLE001 - a reporter cannot mask the capture failure
                    pass
            await join_owned_task(asyncio.create_task(audio.aclose()))
            raise
        pending: asyncio.Queue = asyncio.Queue(maxsize=1)
        recognition: set[LogicalCallerSource] = set()
        # Sources whose words were already delivered while their own facts
        # were still being written: their failure stays this capture's.
        accounting: set[asyncio.Task] = set()
        reported = False
        recognition_failed = asyncio.get_running_loop().create_future()
        processed = 0
        input_ended = False
        current = None
        source = None
        source_token = None
        last_index = 0
        physical_first = None
        physical_start = None
        last_end = 0
        # Evidence only: where the VAD heard the voice stop, before its own
        # trailing silence; None when its event did not say.
        voiced_end = None
        voiced_samples = 0
        interruption_signalled = False
        candidate_speech = False
        ring = CallerPcmRing(self.fence.sample_rate)
        # After a loop stall the ingress hands capture its whole backlog at
        # once, up to its cap: that burst is never a detector failure. A
        # detector that stops processing still ends the call, by samples.
        detector_backlog = (MAX_CALLER_INGRESS_BACKLOG_SECONDS + 2) * self.fence.sample_rate

        def expired(error):
            # A reconnect's bound passed: latch the ending now, from its timer.
            if self.on_failure is not None:
                try:
                    self.on_failure(error)
                except Exception:  # noqa: BLE001 - a reporter cannot mask the capture failure
                    pass
            if not recognition_failed.done():
                recognition_failed.set_result(error)

        def begin_source(first):
            if len(recognition) >= 2:
                raise WorkflowInputCapacityExceeded("workflow listener source backlog requires a controlled call end")
            owned = LogicalCallerSource(ring=ring, first_sample=first, source_factory=self.source_factory,
                                        frame_factory=rtc.AudioFrame, reconnect=self.listener_reconnect,
                                        on_expired=expired)
            def failed(done):
                if not done.cancelled():
                    error = done.exception()
                    if error is not None and not recognition_failed.done():
                        recognition_failed.set_result(error)
            owned.task.add_done_callback(failed)
            recognition.add(owned)
            return owned

        def seal(end):
            nonlocal current, source, source_token
            source.seal(end)
            sealed = self.fence.speech_ended(current, end_sample=end)
            if self.source_observer is not None:
                self.source_observer.seal_turn(source_token)
            try:
                pending.put_nowait((sealed, source, source_token, CallerTurnTiming(
                    sealed_at=time.monotonic(), voiced_end_sample=voiced_end)))
            except asyncio.QueueFull:
                raise WorkflowInputCapacityExceeded("workflow listener backlog requires a controlled call end") from None
            current, source, source_token = None, None, None

        async def feed() -> None:
            nonlocal input_ended
            async for frame in audio:
                samples = frame.samples_per_channel
                if (frame.num_channels != 1 or frame.sample_rate != self.fence.sample_rate
                        or type(samples) is not int or not 0 < samples <= self.fence.sample_rate // 5
                        or memoryview(frame.data).nbytes != samples * 2):
                    raise WorkflowInputRefused("caller media exceeded its input ownership budget")
                if self.fence.received_samples + samples - processed > detector_backlog:
                    raise WorkflowInputCapacityExceeded("workflow detector backlog requires a controlled call end")
                pcm = bytes(frame.data)
                self.fence.receive_frame(samples=samples, sample_rate=frame.sample_rate)
                ring.append(pcm)
                # The input producer cannot mutate audio after handing it off.
                stream.push_frame(rtc.AudioFrame(data=pcm, sample_rate=frame.sample_rate,
                                                num_channels=1, samples_per_channel=samples))
            input_ended = True
            stream.end_input()

        def sample_index(event) -> int:
            if type(event.samples_index) is not int or event.samples_index < last_index:
                raise WorkflowInputRefused("caller VAD clock reset inside a input epoch")
            result = event.samples_index * self.fence.sample_rate // self.inference_sample_rate
            if result > self.fence.received_samples:
                raise WorkflowInputRefused("caller VAD clock outran the received audio")
            return result

        def speech_samples(frames) -> int:
            if not frames or len(frames) > 1024:
                raise WorkflowInputRefused("caller VAD audio exceeded its frame budget")
            count = 0
            for frame in frames:
                if (frame.sample_rate != self.fence.sample_rate or frame.num_channels != 1
                        or type(frame.samples_per_channel) is not int or frame.samples_per_channel <= 0
                        or memoryview(frame.data).nbytes != frame.samples_per_channel * 2):
                    raise WorkflowInputRefused("caller VAD audio has no input-clock binding")
                count += frame.samples_per_channel
                if count > MAX_VAD_SPEECH_BUFFER_SECONDS * self.fence.sample_rate + self.fence.prefix_padding_samples:
                    raise WorkflowInputRefused("caller VAD audio exceeded its utterance budget")
            return count

        async def observe() -> None:
            nonlocal processed, current, source, source_token, last_index
            nonlocal physical_first, physical_start, last_end, voiced_samples, interruption_signalled, voiced_end
            nonlocal candidate_speech
            async for event in stream:
                end = sample_index(event)
                last_index = event.samples_index
                if event.type == vad.VADEventType.INFERENCE_DONE:
                    probability = event.probability
                    if (type(probability) not in (float, int) or not math.isfinite(probability)
                            or not 0 <= probability <= 1):
                        raise WorkflowInputRefused("caller VAD has no bounded probability observation")
                    candidate_speech = probability >= VAD_ACTIVATION_THRESHOLD
                    if physical_first is not None and probability >= VAD_ACTIVATION_THRESHOLD:
                        voiced_samples += end - processed
                    processed = end
                    if (physical_first is not None and not interruption_signalled
                            and voiced_samples >= self.interruption_samples):
                        interruption_signalled = True
                        async with asyncio.timeout(2):
                            await self.on_interruption()
                    if source is not None and physical_first is not None:
                        source.advance(end)
                    if current is not None and physical_first is None:
                        # Use the processed detector clock. A received-frame
                        # clock could seal before a queued START is observed.
                        # A candidate voiced frame also defers endpoint until
                        # VAD can publish START or reject that short candidate.
                        if end >= last_end + self.silence_samples and probability < VAD_ACTIVATION_THRESHOLD:
                            seal(last_end)
                elif event.type == vad.VADEventType.START_OF_SPEECH:
                    frames = event.frames
                    # Every sample recognized, including the prefix, counts for
                    # eligibility; a later voiced onset cannot hide early text.
                    first = end - speech_samples(frames)
                    duration = event.speech_duration
                    if (physical_first is not None or not last_end < end
                            or first < max(0, last_end - self.fence.prefix_padding_samples)
                            or type(duration) not in (float, int) or not math.isfinite(duration)
                            or not 0 < duration <= (end - first) / self.fence.sample_rate):
                        raise WorkflowInputRefused("caller VAD started an unordered physical fragment")
                    physical_first, physical_start = first, end
                    voiced_samples, interruption_signalled = int(duration * self.fence.sample_rate), False
                    if current is None:
                        voiced_end = None
                        current = self.fence.speech_started(first_sample=first, start_observed=end,
                                                           input_epoch=self.fence.input_epoch)
                        if self.source_observer is not None:
                            # The voiced onset, without VAD's prefix padding:
                            # judged against agent playout on the same clock.
                            source_token = self.source_observer.begin_turn(
                                onset_sample=max(first, end - int(duration * self.fence.sample_rate)))
                        source = begin_source(first)
                        async with asyncio.timeout(2):
                            await self.on_speech()
                    source.advance(end)
                    if voiced_samples >= self.interruption_samples:
                        interruption_signalled = True
                        async with asyncio.timeout(2):
                            await self.on_interruption()
                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    if current is None or physical_first is None or end < physical_start:
                        raise WorkflowInputRefused("caller VAD ended an unowned utterance")
                    retained = speech_samples(event.frames)
                    if (end - physical_first <= MAX_VAD_SPEECH_BUFFER_SECONDS * self.fence.sample_rate
                            and retained != end - physical_first):
                        raise WorkflowInputRefused("caller VAD endpoint has no input-clock binding")
                    source.advance(end)
                    voiced_end = _voiced_end(event, first=physical_first, end=end, rate=self.fence.sample_rate)
                    last_end = end
                    physical_first = physical_start = None
                if self.on_clock is not None:
                    # The live edge is what reached this process, not what
                    # capture has pulled: queued audio is newer than the clock.
                    self.on_clock(WorkflowCaptureClock(processed, physical_first is not None,
                        candidate_speech, len(recognition), max(self.fence.received_samples, audio.delivered_samples)))
            if not input_ended:
                raise WorkflowInputRefused("caller VAD closed before its owned media input")
            # Silero does not emit END when media ends mid-speech. EOF is a
            # normal transport end, not a fabricated caller answer or a VAD
            # integrity failure. Retain/join all owned work during cleanup.
            if current is not None and physical_first is None:
                seal(last_end)
            if source is not None:
                await source.aclose()
                recognition.discard(source)
            await pending.put(None)

        async def deliver() -> None:
            while True:
                item = await pending.get()
                if item is None:
                    # Input ended: a released turn's source facts are still
                    # this capture's, and their failure is still its failure.
                    if accounting:
                        await asyncio.wait(accounting)
                        for task in accounting:
                            if not task.cancelled() and task.exception() is not None:
                                raise task.exception()
                    return
                sealed, owned, token, marks = item
                # The words go to the turn once the provider closed; the
                # source keeps its recognition slot (the call's two-listener
                # bound) until its own facts are written and its task ends.
                result = await owned.outcome()
                if owned.task.done():
                    recognition.discard(owned)
                else:
                    accounting.add(owned.task)

                    def settled(done, owned=owned):
                        recognition.discard(owned)
                        if not done.cancelled() and done.exception() is None:
                            accounting.discard(done)
                    owned.task.add_done_callback(settled)
                if result is not None and not isinstance(result, CallerRecognition):
                    raise WorkflowInputRefused("caller recognition has no complete typed final ownership")
                observation = self.fence.finish_speech(sealed, text=result.text if result else None)
                if isinstance(observation, CallerInput):
                    observation = _timed(observation, result, marks)
                if self.source_observer is not None:
                    if isinstance(observation, BlankInput):
                        self.source_observer.blank_turn(token)
                    else:
                        self.source_observer.completed_turn(token, result.text)
                await self.on_input(observation, result.language if result else None)

        actors = [asyncio.create_task(feed()), asyncio.create_task(observe()), asyncio.create_task(deliver())]
        group = asyncio.gather(*actors)
        try:
            await asyncio.wait((group, recognition_failed, overflow), return_when=asyncio.FIRST_COMPLETED)
            if overflow.done():
                raise overflow.result()
            if recognition_failed.done():
                raise recognition_failed.result()
            await group
        except Exception as error:
            reported = True
            if self.on_failure is not None:
                try:
                    self.on_failure(error)
                except Exception:  # noqa: BLE001 - a reporter cannot mask the capture failure
                    pass
            raise
        finally:
            # Repeated root cancellation must not detach SDK-owned consumers.
            if self.source_observer is not None:
                self.source_observer.unavailable()
            self.fence.close()

            async def close_owned() -> None:
                for task in actors:
                    task.cancel()
                await asyncio.gather(*actors, return_exceptions=True)
                await asyncio.gather(group, return_exceptions=True)
                await asyncio.gather(*(owned.aclose() for owned in recognition), return_exceptions=True)
                # Torn down (cancelled) while a delivered turn's facts were
                # still being written: their failure is still reported, once,
                # unless the capture already reported its own.
                failures = [task.exception() for task in accounting
                            if task.done() and not task.cancelled() and task.exception() is not None]
                if failures and not reported and self.on_failure is not None:
                    try:
                        self.on_failure(failures[0])
                    except Exception:  # noqa: BLE001 - a reporter cannot mask the teardown
                        pass
                errors = await asyncio.gather(stream.aclose(), audio.aclose(), return_exceptions=True)
                for error in errors:
                    if isinstance(error, BaseException):
                        raise error

            await join_owned_task(asyncio.create_task(close_owned()))

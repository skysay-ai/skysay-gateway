# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""LiveKit glue that plays the doomed audio queue out instead of dropping it.

Every DECISION belongs to ``graceful_stop``, which imports no livekit and is
unit-tested directly.  This module is only the wiring, and it is deliberately
one object: the ``rtc.AudioSource`` the runtime holds.

Why this seam and not the audio output
--------------------------------------

``BoundedPlayoutHoldOutput`` sits at the TOP of the audio chain, where the
forwarder runs seconds ahead of the wire, so a gain ramp applied there lands
on frames the caller will never hear -- measured, four profiles, three of them
byte-identical before and after.  The audio the caller loses is the ~200 ms
inside ``rtc.AudioSource``, and on the pinned ``livekit-agents==1.7.1`` two
call sites throw it away:

* ``_ParticipantAudioOutput._forward_audio``, in its ``not
  _playback_enabled.is_set()`` branch -- the barge-in path;
* ``_ParticipantAudioOutput._wait_for_playout``, in its ``interrupted``
  branch -- the ``clear_buffer()`` hard cancel at the end of a reply and under
  a hangup.

Both reach the same object by calling ``self._audio_source.clear_queue()``.
Wrapping THAT object therefore covers barge-in, ``end_call`` and hangup with
one intervention, which is why the proxy lives here rather than at the output
where the pause is requested.

What the proxy does to the runtime's bookkeeping
------------------------------------------------

Nothing, in the direction that matters.  The ramp is pushed AFTER the runtime
has already counted the discard, and the frames it pushes never touch
``_source_pushed_duration``, so the runtime's own arithmetic is byte-for-byte
what it was.  The one drift is ``_source_discarded_duration``, which
over-counts by however much the ramp actually played -- at most the queue --
and that flows into ``playback_position`` on ``on_playback_finished``.

Where that number is read, on the pinned runtime, is worth stating rather than
assuming, because the obvious guess is wrong.  It is NOT what records the
agent's transcript: ``agent_activity`` takes that from
``synchronized_transcript``, computed by the transcript synchronizer's own
clock, and only consults ``playback_position`` as a ``> 0`` liveness test
alongside the first-frame future.  Its one arithmetic consumer is
``_rt_session.truncate(audio_end_ms=...)``, which belongs to the realtime
speech-to-speech path and is gated on ``llm.capabilities.message_truncation``
-- a managed phone call runs the STT/LLM/TTS pipeline and never reaches it.
So the drift is bounded, one-directional, and lands on a reported duration
rather than on anything a customer reads.

Placement, and the SDK coupling it costs
----------------------------------------

Installed after ``session.start()``, exactly like ``guard_playout_hold``: the
published track was created from the real ``rtc.AudioSource`` inside
``_ParticipantAudioOutput._publish_track``, so swapping the attribute
afterwards leaves the track and everything already subscribed to it untouched,
and only the runtime's own later uses go through the proxy.

Three private names of the pinned runtime are read here --
``_audio_source``, ``_source_pushed_duration`` and the ``next_in_chain`` walk
that finds the owner.  Each is pinned by the import/def-and-call smoke test
beside this module's own tests, so a runtime bump fails loudly in CI instead
of silently restoring the hard cut.  If a bump ever removes one, the guard
declines to install and the runtime keeps its own behaviour, which is the
behaviour shipping today.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, AsyncIterable, Callable

from livekit import rtc
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

# Flat ``/app`` import, exactly like production_agent.py's own dependencies:
# this repository has no livekit dependency, so nothing importing livekit is
# importable from the test suite at all. That is precisely why every decision
# lives in graceful_stop.py instead of here.
from graceful_stop import (
    DEFAULT_DRAIN_SECONDS,
    DEFAULT_RETAIN_SECONDS,
    MAX_DRAIN_SECONDS,
    STOP_SOURCE_HARD_CUT,
    AlignmentFeed,
    GracefulStopStats,
    PlayoutRing,
    StopPlan,
    apply_stop_gain,
    plan_stop,
)

# How much audio one pushed ramp frame carries.  Matches the 10 ms the
# runtime's own ``AudioByteStream`` re-chunks to, so the ramp queues at the
# same granularity as the speech it replaces.
_DRAIN_FRAME_SECONDS = 0.010


class DrainingAudioSource:
    """Play the queue out under a gain ramp where the runtime would drop it.

    Every attribute the proxy does not define is delegated, so the runtime
    keeps using the real source for everything except the three calls this
    class has an opinion about.
    """

    def __init__(
        self,
        source: Any,
        *,
        owner: Any,
        stats: GracefulStopStats,
        feed: AlignmentFeed | None = None,
        drain_seconds: float = DEFAULT_DRAIN_SECONDS,
        max_drain_seconds: float = MAX_DRAIN_SECONDS,
        retain_seconds: float = DEFAULT_RETAIN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._owner = owner
        self._stats = stats
        self._feed = feed
        self._drain_seconds = float(drain_seconds)
        self._max_drain_seconds = float(max_drain_seconds)
        self._clock = clock
        self._ring = PlayoutRing(
            sample_rate=int(source.sample_rate),
            num_channels=int(source.num_channels),
            retain_seconds=retain_seconds,
        )
        # Serialises the ramp against the forwarder's own pushes so the two
        # never interleave inside the source.
        self._gate = asyncio.Lock()
        self._drain_task: asyncio.Task[None] | None = None
        self._drain_generation = 0
        self._index: Any | None = None
        self._last_pushed: float | None = None
        self._closed = False
        # Content-only queue estimate, reset by every clear.  The SDK's own
        # ``queued_duration`` uses this formula but never resets it on
        # ``clear_queue``, so after a clear it reports a queue that no longer
        # exists; the playhead has to be computed from one that does.
        self._queue_seconds = 0.0
        self._queue_stamp: float | None = None
        # When the ramp already pushed will have finished SOUNDING. Not the
        # same question as whether the task that pushed it has returned:
        # the pushes complete in microseconds into an empty queue, while
        # the audio they queued plays for the whole drain.
        self._ramp_until = 0.0

    # -- delegation --------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for names this class does not define. The guard covers
        # the one case that would otherwise recurse forever: a proxy whose
        # __init__ raised before ``_source`` was assigned.
        if name == "_source":
            raise AttributeError(name)
        return getattr(self._source, name)

    @property
    def wrapped(self) -> Any:
        """The real ``rtc.AudioSource``, for tests and teardown."""

        return self._source

    # -- the audio path ----------------------------------------------------

    async def capture_frame(self, frame: Any) -> None:
        async with self._gate:
            self._observe_run(frame)
            self._ring.push(_frame_bytes(frame))
            self._note_queued(float(frame.duration))
            await self._source.capture_frame(frame)

    def clear_queue(self) -> None:
        """The runtime discards the queue; play its opening out first.

        The real clear still happens, immediately and unconditionally: the
        runtime asked for the floor and must get it.  What follows is a short
        replacement built from audio the caller was already going to hear, so
        the stop lands on a fade or a word end instead of mid-sample.
        """

        queued = self._content_queued()
        if queued <= 0 and self._ramp_in_flight():
            # A SECOND clear against a ramp that is still sounding, which
            # belongs to the stop already being served rather than to a new
            # one.
            #
            # A qualified barge-in reaches the source twice: the pause branch
            # of ``_forward_audio`` clears when the boundary tail expires, and
            # the confirmed interruption then cancels the reply, so
            # ``_wait_for_playout`` clears again. When the second lands inside
            # the ramp the first scheduled -- a ~100 ms window -- treating it
            # as a new stop wipes the unplayed remainder of that ramp and
            # restores the hard cut, on the single path this change exists to
            # fix. Measured: the probe's barge-in tail went back to 4599 with
            # a step of 480 until this guard existed.
            #
            # Skipping the real clear discards nothing belonging to the
            # runtime: no content is left to discard, and the only audio in
            # the source queue is the ramp we deliberately put there.
            return
        self._cancel_drain()
        self._source.clear_queue()
        self._reset_queue_estimate()
        if queued <= 0 or self._closed:
            # Nothing was going to be discarded, so nothing was cut.
            return
        playhead = max(0.0, self._pushed_seconds() - queued)
        plan = plan_stop(
            queued_seconds=queued,
            playhead_seconds=playhead,
            index=self._index,
            drain_seconds=self._drain_seconds,
            max_drain_seconds=self._max_drain_seconds,
        )
        if plan is None:
            return
        # The slice is taken here, synchronously, because the ring keeps
        # moving; the per-sample gain runs in the task, because this call is
        # on the media loop and a barge-in must not wait for arithmetic.
        raw = self._ring.take(playhead, plan.drain_seconds)
        if not raw:
            self._stats.note_stop(source=STOP_SOURCE_HARD_CUT)
            return
        if not self._start_drain(raw, plan):
            self._stats.note_stop(source=STOP_SOURCE_HARD_CUT)

    def _ramp_in_flight(self) -> bool:
        """Whether a ramp is still on its way to the caller, either way it can be.

        TWO windows, and a guard that covers only one of them is no guard at
        all. While ``_drain`` is awaiting a ``capture_frame`` the ramp is
        partly queued but ``_ramp_until`` is not set yet -- it is written in
        that method's ``finally``. Once the pushes return, the task is done but
        the audio it queued has only started playing. The first window is the
        task; the second is the clock.
        """

        task = self._drain_task
        if task is not None and not task.done():
            return True
        return self._clock() < self._ramp_until

    async def aclose(self) -> None:
        self._closed = True
        task = self._drain_task
        self._cancel_drain()
        if task is not None:
            # Waited for rather than merely cancelled: a ramp still inside
            # ``capture_frame`` when the source closes would otherwise be
            # reported as a task destroyed while pending, and its counter
            # would never be written.
            #
            # ``wait`` rather than ``await task``: the ramp's own cancellation
            # is not an outcome this method has an opinion about, and ``wait``
            # simply does not raise it. Catching it instead would be
            # indistinguishable from cancellation aimed at whoever is closing
            # US -- asyncio cancels the ramp too on its way through -- and
            # swallowing that would make teardown uninterruptible.
            await asyncio.wait({task})
        await self._source.aclose()

    # -- run tracking ------------------------------------------------------

    def _pushed_seconds(self) -> float:
        """The runtime's own content total for the open playout run."""

        try:
            return max(0.0, float(self._owner._source_pushed_duration))
        except Exception:  # pragma: no cover - runtime moved its bookkeeping
            return 0.0

    def _observe_run(self, frame: Any) -> None:
        """Notice the runtime starting a new playout run, and follow it.

        ``_wait_for_playout`` zeroes ``_source_pushed_duration`` when a run
        ends, so a total that does not advance past the previous frame's is
        the only signal a new run has begun -- and it is the runtime's own
        signal, not a guess about ordering somewhere else.
        """

        duration = float(getattr(frame, "duration", 0.0) or 0.0)
        if duration <= 0:
            # A frame that carries no audio moves neither the runtime's total
            # nor this one. Recording it would be worse than ignoring it: an
            # empty frame arriving right after a run ended would park the
            # remembered total at zero, and the next real frame would then
            # look like an ADVANCE rather than the restart it is, so the ring
            # would keep the previous run's audio under the new run's offsets.
            return
        pushed = self._pushed_seconds()
        # Two tests, and the first is the exact one. The runtime adds a frame's
        # duration to its total BEFORE awaiting the capture, and zeroes that
        # total when a run ends -- so on the first frame of a run the total IS
        # this frame's duration, and on every later frame it is strictly
        # greater, because durations are positive. Comparing against the
        # previous frame alone would miss the boundary whenever a run's WHOLE
        # total was shorter than the next run's first frame: a one-frame
        # utterance ending on a 10 ms remainder, followed by a normal 50 ms
        # frame, still reads as an advance. The ring and the alignment index
        # would then stay attached to the utterance before, and an early
        # interruption in the new run would replay it.
        first_of_run = abs(pushed - duration) < 1e-9
        if first_of_run or self._last_pushed is None or pushed <= self._last_pushed:
            self._ring.reset()
            self._index = self._feed.adopt() if self._feed is not None else None
        self._last_pushed = pushed

    # -- the content-only queue estimate ------------------------------------

    def _drained_since(self, now: float) -> float:
        """Seconds of CONTENT the source has actually played since the last note.

        Content pushed behind a ramp does not start draining until that ramp
        has finished, so the elapsed wall clock is not the answer while one is
        sounding. Getting this wrong is not a rounding error: the estimate
        would decay to zero while cancelled speech was still queued, and the
        duplicate-clear guard would then skip a clear the runtime genuinely
        needed -- letting audio the runtime cancelled reach the caller.
        """

        if self._queue_stamp is None:
            return 0.0
        started = max(self._queue_stamp, min(self._ramp_until, now))
        return max(0.0, now - started)

    def _note_queued(self, seconds: float) -> None:
        now = self._clock()
        self._queue_seconds = (
            max(0.0, self._queue_seconds - self._drained_since(now)) + max(0.0, seconds)
        )
        self._queue_stamp = now

    def _content_queued(self) -> float:
        """Seconds of pushed CONTENT the caller has not heard yet.

        Ramp frames are excluded from the total on purpose: they are a replay
        of content the playhead has already passed, so counting them would move
        the playhead backwards.  They are NOT excluded from the timing -- see
        ``_drained_since`` -- because content queued behind a ramp is not
        playing yet, and pretending it is would report an empty queue while the
        caller still had speech coming.
        """

        if self._queue_stamp is None:
            return 0.0
        mine = max(0.0, self._queue_seconds - self._drained_since(self._clock()))
        try:
            reported = float(self._source.queued_duration)
        except Exception:  # pragma: no cover - runtime moved its estimate
            return mine
        # The smaller of the two on purpose: never claim more unheard audio
        # than the source itself admits to holding. Over-claiming would place
        # the playhead earlier than it is and replay a sliver the caller has
        # already heard, which is the artefact this whole change removes.
        return min(mine, max(0.0, reported))

    def _reset_queue_estimate(self) -> None:
        self._queue_seconds = 0.0
        self._queue_stamp = None

    # -- rendering and pushing the ramp -------------------------------------

    def _render(self, raw: bytes, plan: StopPlan) -> list[Any]:
        rate = self._ring.sample_rate
        channels = self._ring.num_channels
        faded = apply_stop_gain(
            raw,
            num_channels=channels,
            hold_samples=int(plan.hold_seconds * rate),
            ramp_samples=max(1, int(plan.ramp_seconds * rate)),
        )
        chunk = max(1, int(_DRAIN_FRAME_SECONDS * rate)) * channels * 2
        frames: list[Any] = []
        for offset in range(0, len(faded), chunk):
            block = faded[offset : offset + chunk]
            samples = len(block) // (channels * 2)
            if samples <= 0:
                continue
            frames.append(rtc.AudioFrame(block, rate, channels, samples))
        return frames

    def _start_drain(self, raw: bytes, plan: StopPlan) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no media loop, nothing to push on
            return False
        self._drain_generation += 1
        generation = self._drain_generation
        self._drain_task = loop.create_task(self._drain(raw, plan, generation))
        return True

    def _cancel_drain(self) -> None:
        """Retire any ramp still in flight before the next stop is planned.

        Both halves matter.  Cancelling stops the task at its next await;
        bumping the generation stops it even if it is already inside one, so a
        second stop can never leave the previous utterance's tail queued
        behind the new silence.
        """

        self._drain_generation += 1
        self._ramp_until = 0.0
        task, self._drain_task = self._drain_task, None
        if task is not None and not task.done():
            task.cancel()

    async def _drain(self, raw: bytes, plan: StopPlan, generation: int) -> None:
        pushed = 0
        pushed_seconds = 0.0
        planned = 0
        try:
            frames = self._render(raw, plan)
            planned = len(frames)
            async with self._gate:
                if generation != self._drain_generation or self._closed:
                    return
                # Clear again, now that the forwarder is not mid-push.
                #
                # The runtime's clear can arrive from ``_wait_for_playout``
                # while ``_forward_audio`` is blocked inside the source's
                # ``capture_frame``; clearing frees the space that call was
                # waiting for, so its frame -- audio NEWER than the ramp --
                # lands in the queue. Appending the ramp behind it would play
                # the utterance out of order, which is worse than the cut this
                # replaces. Nothing legitimate is lost: the forwarder cannot
                # push again until it has this gate, and on the interrupted
                # path it is skipping captures anyway.
                self._source.clear_queue()
                for frame in frames:
                    if generation != self._drain_generation or self._closed:
                        break
                    await self._source.capture_frame(frame)
                    pushed += 1
                    pushed_seconds += frame.samples_per_channel / self._ring.sample_rate
        except asyncio.CancelledError:
            raise
        except Exception:
            # A source that refuses the ramp still owes the caller the stop the
            # runtime already performed; it is counted, never re-raised into
            # the media loop.
            pass
        finally:
            # A drain a LATER stop superseded reports nothing at all: it never
            # reached silence, so calling it a graceful stop would tell the
            # readiness scoring a fade happened when the caller heard a cut,
            # and the stop that superseded it reports its own outcome anyway.
            current = generation == self._drain_generation
            if pushed_seconds > 0 and current:
                self._ramp_until = self._clock() + pushed_seconds
            if not current:
                pass
            elif pushed > 0 and pushed == planned:
                # Summed from the frames actually pushed, not from a frame
                # count: a drain is only a whole number of 10 ms frames by
                # accident, and a word-aligned boundary almost never is, so
                # counting frames would overstate every such stop by up to one
                # frame -- in the counter this change is meant to be judged by.
                self._stats.note_stop(
                    source=plan.source,
                    drained_ms=int(round(pushed_seconds * 1_000)),
                )
            else:
                # Either nothing played, or the source stopped accepting the
                # ramp part way through. Both are hard cuts: only a ramp that
                # ran to its last frame reaches silence, and a partial one ends
                # on whatever level it had got down to -- which is the abrupt
                # stop this change exists to remove, and must not be reported
                # as a graceful one. A ramp retired by a LATER stop is neither,
                # and is excluded above: that stop reports its own outcome.
                self._stats.note_stop(source=STOP_SOURCE_HARD_CUT)


def _frame_bytes(frame: Any) -> bytes:
    """The frame's PCM payload, trimmed to the samples it declares."""

    try:
        wanted = int(frame.samples_per_channel) * int(frame.num_channels) * 2
    except Exception:  # pragma: no cover - frame without the documented shape
        return b""
    if wanted <= 0:
        return b""
    return bytes(frame.data)[:wanted]


def find_participant_audio_output(sink: Any) -> Any | None:
    """Walk the sink chain to the runtime output that owns the audio source."""

    seen = 0
    node = sink
    while node is not None and seen < 8:
        if getattr(node, "_audio_source", None) is not None:
            return node
        node = getattr(node, "next_in_chain", None)
        seen += 1
    return None


def install_graceful_stop(
    session: Any,
    *,
    stats: GracefulStopStats,
    feed: AlignmentFeed | None = None,
    drain_seconds: float = DEFAULT_DRAIN_SECONDS,
    max_drain_seconds: float = MAX_DRAIN_SECONDS,
) -> DrainingAudioSource | None:
    """Wrap the started session's audio source; ``None`` when it has none.

    Declines rather than raises on every shape it does not recognise.  A
    runtime that moved its audio source keeps its own stop behaviour, which
    is the behaviour production runs today -- a graceful stop is worth having
    and is never worth a failed call.
    """

    sink = getattr(getattr(session, "output", None), "audio", None)
    owner = find_participant_audio_output(sink)
    if owner is None:
        return None
    source = getattr(owner, "_audio_source", None)
    if source is None or isinstance(source, DrainingAudioSource):
        return None
    if not callable(getattr(source, "clear_queue", None)):
        return None
    try:
        proxy = DrainingAudioSource(
            source,
            owner=owner,
            stats=stats,
            feed=feed,
            drain_seconds=drain_seconds,
            max_drain_seconds=max_drain_seconds,
        )
    except Exception:
        return None
    owner._audio_source = proxy
    return proxy


async def observe_timed_transcript(
    stream: AsyncIterable[Any], feed: AlignmentFeed
) -> Any:
    """Pass TTS frames through, recording the word ends they declare.

    ``Agent.tts_node`` is LiveKit's documented TTS override point, and the
    frames crossing it carry the provider's own word timings in
    ``userdata[USERDATA_TIMED_TRANSCRIPT]`` -- attached by ``AudioEmitter``
    for every plugin that declares ``capabilities.aligned_transcript``,
    independently of ``use_tts_aligned_transcript``.  Reading them here
    therefore costs the session nothing: the customer-visible transcript keeps
    coming from the model's own text rather than from the provider's
    normalisation of it.

    The cumulative audio duration is carried alongside each word so the index
    can check that the two clocks describe the same stream instead of assuming
    it.  Frames are yielded untouched, in order, whatever the alignment says.

    One synthesis feeds one playout run, and the feed pairs them in order.  If
    a future runtime ever overlapped two syntheses, a run could adopt the
    wrong index -- and the damage is still bounded, because a boundary is only
    ever acted on when it lies inside the audio the source is holding.  The
    worst a mis-paired index can do is stop up to the queue later than a ramp
    would, at a point that is not a word end; it can never play audio the
    caller was not already about to hear.
    """

    if inspect.isawaitable(stream):
        # ``Agent.tts_node`` is typed as returning either an async iterable or
        # a coroutine that produces one; the shipped default returns the
        # former, but a runtime bump must not turn that into a TypeError
        # inside the media loop.
        stream = await stream
    if stream is None:
        return
    index = feed.open_index()
    produced = 0.0
    async for frame in stream:
        produced += float(getattr(frame, "duration", 0.0) or 0.0)
        if index is not None:
            # Announced from the first frame on, never at open: a synthesis
            # cancelled before it produces audio never gets a playout run, and
            # queueing it anyway would shift every later run onto it.
            feed.publish(index)
            _record_words(index, frame, produced)
        yield frame


def _record_words(index: Any, frame: Any, produced: float) -> None:
    try:
        userdata = frame.userdata
        declared = userdata.get(USERDATA_TIMED_TRANSCRIPT) or ()
    except Exception:  # pragma: no cover - frame without userdata
        return
    for word in declared:
        end = getattr(word, "end_time", None)
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            continue
        index.note_word(end_seconds=float(end), audio_seconds=produced)


__all__ = [
    "DrainingAudioSource",
    "find_participant_audio_output",
    "install_graceful_stop",
    "observe_timed_transcript",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Preserve the outbound pickup while a live provider session starts.

Only the authenticated caller track is consumed, at the engine's input rate.
The recorder remains an independent observer, not the model's audio source.
Before answer all frames are discarded. After answer startup audio is bounded
and delivered once, in order, through the ordinary AgentInput seam. No PCM is
logged. A failed or overflowing capture is a failed qualification, not silence.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

STARTUP_SECONDS = 8.0
STARTUP_BYTES = 768_000
STARTUP_FRAMES = 400


class NativeOpeningAuthorization:
    """Use the pinned SDK's speech/tool authorization while still hearing input.

    Limited to the initial non-workflow outbound pickup. Provider output that
    starts while the caller speaks stays paused before playout; a tentative
    reply held during a pause is discarded if the caller takes the floor back.
    """

    def __init__(self, session: Any, decider: Any, *, input_delivered_through,
                 emit=lambda **details: None) -> None:
        self.session = session
        self.decider = decider
        self.emit = emit
        self.input_delivered_through = input_delivered_through
        self.resumptions = 0
        self.activity = getattr(session, '_activity', None)
        if not all(callable(getattr(self.activity, name, None)) for name in
                   ('_pause_authorization', '_resume_authorization')):
            raise RuntimeError('native opening authorization unavailable')
        self.held = True
        self.handles: set[Any] = set()
        self.response_input_edges: dict[Any, float] = {}
        self.discarded = 0
        self.explicit_reply = False
        self.explicit_handle_created = False
        self.watching_reply = False
        #: Every response created while the pickup reply is watched, in order (the live set forgets a finished one).
        self.watched: list[Any] = []
        self.response_sequence = 0
        self.activity._pause_authorization()
        session.on('speech_created', self.on_speech)

    def on_speech(self, event: Any) -> None:
        if self.explicit_reply and getattr(event, "user_initiated", False):
            self.explicit_handle_created = True
        if self.explicit_reply and not getattr(event, 'user_initiated', False):
            self._discard(event.speech_handle)
            self.emit(action="automatic_cut_for_replacement", discarded=self.discarded)
            return
        if not self.held and not self.watching_reply:
            return
        handle = event.speech_handle
        self.response_sequence += 1
        delivered_at = self.input_delivered_through()
        boundary_lag_ms = max(0, round((time.monotonic() - delivered_at) * 1000)) if delivered_at is not None else 0
        self.emit(action="response_created", caller_speaking=self.decider.callee_speaking_now,
                  resumptions=self.resumptions, held_responses=len(self.handles),
                  boundary_lag_ms=boundary_lag_ms)
        if len(self.handles) >= 16:
            self._discard(handle)
            return
        self.handles.add(handle)
        if self.watching_reply and len(self.watched) < 16:
            self.watched.append(handle)
        self.response_input_edges[handle] = self.input_delivered_through() or 0.0
        handle.add_done_callback(self._forget_response)

    def _forget_response(self, handle: Any) -> None:
        self.handles.discard(handle)
        self.response_input_edges.pop(handle, None)

    def note_caller_speech(self, *, until: float) -> None:
        """Invalidate replies only for caller audio arriving after they began.

        Queued VAD work may describe audio already supplied to the SDK when the provider
        created its reply. It still holds authorization, but cannot by itself
        make that reply stale. Newly received qualified speech can.
        """
        if self.held and self.decider.callee_speaking_now:
            for handle, boundary in tuple(self.response_input_edges.items()):
                if until > boundary + 1e-6:
                    self._discard(handle)

    def _discard(self, handle: Any) -> None:
        if not handle.done() and not handle.interrupted:
            handle.interrupt(force=True)
            self.discarded += 1

    def resumed(self, *, until: float | None = None) -> None:
        if self.held:
            self.resumptions += 1
            self.emit(action="caller_resumed", held_responses=len(self.handles))
            if until is None:
                self.discard_pending()
            else:
                self.note_caller_speech(until=until)

    def discard_pending(self) -> None:
        for handle in tuple(self.handles):
            self._discard(handle)

    def request_missing_reply(self, generate) -> Any:
        """Claim one explicit reply when no usable provider response survived.

        Keep rejecting unsolicited generations until that reply settles, so
        a late provider turn cannot race the replacement onto the line.
        """
        if not self.held or not self.decider.callee_spoke or any(
                not handle.done() and not handle.interrupted for handle in self.handles):
            return None
        return self.request_explicit_reply(generate)

    def request_explicit_reply(self, generate) -> Any:
        self.claim_explicit_reply()
        try:
            handle = generate()
        except BaseException:
            self.finish_explicit_reply()
            raise
        self.emit(action="replacement_requested", held_responses=len(self.handles))
        handle.add_done_callback(lambda _: self.finish_explicit_reply())
        return handle

    def claim_explicit_reply(self) -> None:
        if self.explicit_reply:
            raise RuntimeError("native opening already has an explicit reply")
        self.explicit_reply = True
        self.explicit_handle_created = False

    def finish_explicit_reply(self) -> None:
        self.explicit_reply = False
        self.explicit_handle_created = False
        if not self.held and not self.watching_reply:
            self.session.off('speech_created', self.on_speech)

    def note_caller_started(self) -> None:
        # Ordinary barge-in owns the next response after opening release.
        if not self.held and self.explicit_handle_created:
            self.finish_explicit_reply()

    def note_speaking(self, *, at: float) -> bool:
        """An SDK state notification cannot release the platform's floor."""
        if self.held:
            self.emit(action="speaking_while_held", held_responses=len(self.handles))
            self.discard_pending()
            return False
        self.decider.adopt_native_response(at=at)
        return True

    def release(self) -> None:
        if self.held and self.decider.callee_speaking_now:
            raise RuntimeError("native opening cannot authorize while caller speaks")
        if self.held and self.decider.telemetry()["unobserved_audio_at_open"]:
            raise RuntimeError("native opening cannot authorize unobserved pickup audio")
        if self.held and not self.decider.callee_spoke:
            # Silence/noise owns one explicit opening. A provider response to
            # line noise (including a tool call) must never escape alongside it.
            self.discard_pending()
        self._release()

    def _release(self) -> None:
        if not self.held:
            return
        self.emit(action="released", held_responses=len(self.handles), discarded=self.discarded,
                  resumptions=self.resumptions)
        self.held = False
        if not self.explicit_reply and not self.watching_reply:
            self.session.off('speech_created', self.on_speech)
        self.activity._resume_authorization()

    def close(self) -> None:
        self.discard_pending()
        self.watching_reply = False
        self.finish_explicit_reply()
        self._release()


#: How long a pickup reply that only called a refused navigation tool waits for the engine's own reply to that refusal.
REFUSED_TOOL_FOLLOW_UP_SECONDS = 4.0
#: At most this many such follow-ups are waited for (a model can call the refused tool again).
REFUSED_TOOL_FOLLOW_UPS = 3
#: How long one follow-up is waited out once it exists; one still running then owns the line and nothing is replaced.
REFUSED_TOOL_FOLLOW_UP_PLAYOUT_SECONDS = 15.0
_FOLLOW_UP_POLL_SECONDS = 0.05


def _has_tool_items(handle: Any) -> bool:
    return any(getattr(item, 'type', '') in {'function_call', 'function_call_output'} for item in handle.chat_items)


def _quietly_true(predicate) -> bool:
    """A line predicate that cannot be read is not a reason to leave a person unanswered."""
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001
        return False


async def _waited_out(wait, handle, *, ended, seconds) -> bool:
    """Wait one follow-up out, bounded; ``False`` if it is still running (it then owns the line)."""
    waiter = asyncio.ensure_future(wait(handle, ended=ended))
    done, _ = await asyncio.wait({waiter}, timeout=max(0.0, float(seconds)))
    if not done:
        waiter.cancel()  # only the wait: the SDK's playout wait is shielded, so the follow-up plays on
        return False
    waiter.result()
    return True


async def finish_native_pickup_reply(gate, *, generate, wait, ended, audible_seconds,
                                     caller_revision, prepare_replacement=None, answered_nothing=None,
                                     line_unsettled=lambda: False,
                                     follow_up_seconds=REFUSED_TOOL_FOLLOW_UP_SECONDS,
                                     follow_up_playout_seconds=REFUSED_TOOL_FOLLOW_UP_PLAYOUT_SECONDS,
                                     sleep=asyncio.sleep, clock=time.monotonic):
    """Replace an empty opening once, unless a newer turn already owns output.

    Audio evidence must come from the output sink, before it reports completion,
    rather than the asynchronous room recorder. A duplex provider can continue
    its cancelled burst; let it finish before requesting a separate response.

    A reply whose only content was a tool call is normally answered by the
    engine's reply to that tool's result. ``answered_nothing(items)`` names the
    tool calls that did nothing (a navigation tool refused outside a menu or a
    hold): such a reply is still an empty opening. The engine's own reply to
    the refusal is waited for first (bounded), and the replacement is asked
    only when that reply, too, put nothing on the line, nothing else is
    running, the caller has not spoken, and ``line_unsettled()`` says the line
    is not a menu, a hold or unrecognised audio.
    """
    baseline = audible_seconds()
    revision = caller_revision()
    gate.watching_reply = True
    try:
        selected = tuple(h for h in gate.handles if not h.done() and not h.interrupted)
        response_sequence = gate.response_sequence
        # Every response from here on is the engine's: one may start while the selected reply is still waited out.
        seen = len(gate.watched)
        gate.release()
        for handle in selected:
            await wait(handle, ended=ended)

        def overtaken():
            return ended.is_set() or caller_revision() != revision or audible_seconds() > baseline

        def superseded():
            return (overtaken() or gate.response_sequence != response_sequence
                    or (refused_only and _quietly_true(line_unsettled)))

        refused_only = bool(
            answered_nothing is not None and selected
            and not any(h.interrupted for h in selected)
            and any(_has_tool_items(h) for h in selected)
            and all(answered_nothing(h.chat_items) for h in selected if _has_tool_items(h)))
        if refused_only:
            # The engine replies to the refusal on its own (or, with the reply suppressed, does not): watch that
            # reply out before deciding the opening was empty. Its arrival must not read as a newer turn, so every
            # response created since the snapshot is inspected before the sequence is rebased past it.
            first_seen, rounds, quiet = seen, 0, False
            while True:
                if len(gate.watched) == seen:
                    if quiet or rounds >= REFUSED_TOOL_FOLLOW_UPS:
                        break
                    deadline = clock() + max(0.0, float(follow_up_seconds))
                    while len(gate.watched) == seen and not overtaken() and clock() < deadline:
                        await sleep(_FOLLOW_UP_POLL_SECONDS)
                    if overtaken():
                        return None
                    if len(gate.watched) == seen:
                        break
                arrived = gate.watched[seen:]
                seen += len(arrived)
                rounds += 1
                if rounds > REFUSED_TOOL_FOLLOW_UPS + 1:
                    return None  # the model keeps answering with refused tools: it is not ours to speak over
                for handle in arrived:
                    if not await _waited_out(wait, handle, ended=ended, seconds=follow_up_playout_seconds):
                        return None
                if overtaken():
                    return None
                if any(h.interrupted or (_has_tool_items(h) and not answered_nothing(h.chat_items)) for h in arrived):
                    return None
                if not all(_has_tool_items(h) for h in arrived):
                    quiet = True  # the engine's reply to the refusal said nothing; anything after it is inspected too
            if gate.response_sequence - response_sequence != seen - first_seen:
                return None  # a response was not recorded (the watch list is full): it was never inspected
            if _quietly_true(line_unsettled):
                return None
            response_sequence = gate.response_sequence
            gate.emit(action="replacement_after_refused_tool", held_responses=len(gate.handles))
        if superseded():
            return None
        if not refused_only and any(h.interrupted or _has_tool_items(h) for h in selected):
            return None
        if any(not h.done() and not h.interrupted for h in gate.handles):
            return None
        if prepare_replacement is not None:
            await prepare_replacement(superseded)
            if superseded() or any(not h.done() and not h.interrupted for h in gate.handles):
                return None
        replacement = gate.request_explicit_reply(generate)
        await wait(replacement, ended=ended)
        return replacement.exception() if replacement.done() else None
    finally:
        gate.watching_reply = False
        gate.finish_explicit_reply()


async def finish_native_silence_opening(gate, *, generate, ended, prepare_replacement=None):
    """Own the one explicit opening a silent or noisy pickup is owed.

    Releasing authorization discards every provider response to silence or
    noise, but a forced SDK interruption does not stop a duplex provider's
    output: its burst can still be open. The adapter satisfies an explicit
    ask only on a NEW burst, so an ask sent inside the discarded one is lost
    and the call stays silent until the reply budget expires. The discarded
    output settles first (``prepare_replacement``, the same seam the pickup
    replacement uses), then the opening is asked.

    Only the session ending stands the opening down. The opening decider has
    already judged the pickup silence or noise and stopped listening; a raw
    voice-activity onset on that same noise is not a caller taking the floor,
    and treating it as one skipped the only ask this branch exists to place.
    """
    gate.claim_explicit_reply()
    try:
        gate.release()

        def superseded():
            return ended.is_set()
        if prepare_replacement is not None:
            started = time.monotonic()
            await prepare_replacement(superseded)
            # Bounded evidence for the opening's ownership hand-over: how long
            # discarded provider output held the explicit ask back.
            gate.emit(action="discarded_output_settled",
                      waited_ms=round((time.monotonic() - started) * 1000))
        if superseded():
            gate.emit(action="opening_superseded", discarded=gate.discarded)
            return None
        return await generate()
    finally:
        gate.finish_explicit_reply()


async def settle_cancelled_duplex_output(realtime_session, *, ended, superseded, timeout=4.0):
    """A forced SDK interruption cannot cancel a duplex provider's output.

    Its adapter satisfies an explicit request only on a NEW audio burst. Asking
    inside the interrupted burst can lose the whole replacement. Wait at most
    four seconds for that burst to end; new caller/output ownership wins at once.
    This pinned SDK seam is exercised with the real DuplexRealtimeAdapter.
    """
    from livekit.agents.llm.duplex_adapter import _DuplexRealtimeSession

    if not isinstance(realtime_session, _DuplexRealtimeSession):
        return
    deadline = time.monotonic() + timeout
    while realtime_session._burst is not None:
        if ended.is_set() or superseded():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("cancelled duplex opening output did not settle")
        try:
            await asyncio.wait_for(ended.wait(), timeout=.02)
        except asyncio.TimeoutError:
            pass


class PickupAudioBuffer:
    def __init__(self) -> None:
        self.frames: deque[Any] = deque()
        self.seconds = 0.0
        self.bytes = 0
        self.answered = False
        self.attached = False
        self.released = False
        self.pickup_completed = False
        self.was_attached = False
        self.closed = False
        self.error: Exception | None = None
        self.failed = asyncio.Event()
        self.ready = asyncio.Event()
        self.retained = 0
        self.delivered = 0
        self.delivered_through: float | None = None
        self.dropped_after_pickup = 0
        self.startup_retained = 0
        self.answered_at: float | None = None
        self.observer = None

    def arm(self) -> None:
        if not self.answered:
            self.answered_at = time.monotonic()
        self.answered = True

    def observe(self, callback) -> None:
        self.observer = callback
        for at, frame in self.frames:
            try:
                callback(frame, at)
            except Exception:
                self.fail("live pickup speech observation failed")
                return


    def release(self, *, preserve: bool = True) -> None:
        """Open model delivery once response authorization is installed."""
        if not preserve:
            self.clear()
        self.released = True
        self.ready.set()

    def attach(self) -> None:
        self.attached = self.was_attached = True
        self.ready.set()

    def detach(self) -> None:
        self.attached = False
        if self.was_attached:
            self.clear()

    def clear(self) -> None:
        self.frames.clear()
        self.seconds = self.bytes = 0

    def fail(self, reason: str) -> None:
        self.error = RuntimeError(reason)
        self.clear()
        self.failed.set()
        self.ready.set()

    def push(self, frame: Any) -> None:
        if self.closed or self.error or not self.answered or (self.was_attached and not self.attached):
            return
        try:
            duration = frame.samples_per_channel / frame.sample_rate
            size = memoryview(frame.data).nbytes
            if not (0 < duration <= .5) or size != frame.samples_per_channel * frame.num_channels * 2:
                raise ValueError('malformed caller frame')
            seconds, byte_limit, frame_limit = STARTUP_SECONDS, STARTUP_BYTES, STARTUP_FRAMES
            # After pickup, bounded buffering must not turn a slow consumer
            # into a permanently dead input stream. Prefer recent audio,
            # matching intentional mute/reconnect behavior, and count loss.
            while self.pickup_completed and self.frames and (
                    self.seconds + duration > seconds + 1e-6
                    or self.bytes + size > byte_limit or len(self.frames) >= frame_limit):
                _, oldest = self.frames.popleft()
                self.seconds -= oldest.samples_per_channel / oldest.sample_rate
                self.bytes -= memoryview(oldest.data).nbytes
                self.dropped_after_pickup += 1
            if (self.seconds + duration > seconds + 1e-6
                    or self.bytes + size > byte_limit or len(self.frames) >= frame_limit):
                raise ValueError('pickup buffer overflow')
            copied = type(frame)(data=bytes(frame.data), sample_rate=frame.sample_rate,
                                 num_channels=frame.num_channels, samples_per_channel=frame.samples_per_channel)
        except Exception:
            self.fail('live pickup audio could not be retained')
            return
        at = time.monotonic()
        self.frames.append((at, copied))
        if not self.was_attached:
            self.startup_retained += 1
        if self.observer is not None:
            try:
                self.observer(copied, at)
            except Exception:
                self.fail('live pickup speech observation failed')
                return
        self.seconds += duration
        self.bytes += size
        self.retained += 1
        self.ready.set()

    async def next(self) -> Any:
        while True:
            if self.closed:
                raise StopAsyncIteration
            if self.error:
                raise self.error
            if self.attached and self.released and self.frames:
                at, frame = self.frames.popleft()
                self.seconds -= frame.samples_per_channel / frame.sample_rate
                self.bytes -= memoryview(frame.data).nbytes
                self.delivered += 1
                self.delivered_through = at
                return frame
            self.ready.clear()
            await self.ready.wait()

    def close(self) -> None:
        self.closed = True
        self.observer = None
        self.clear()
        self.ready.set()


def capture_pickup(room: Any, participant: Any, *, answer_is_known) -> Any:
    """Buffer the pinned SDK's normal caller input, including its AGC and track lifecycle.

    This private SDK seam is covered by the image qualification probe. Keeping
    its existing media processing is preferable to a second raw RTP consumer
    that silently drops the normal gain control and reconnect behavior.
    """
    from livekit.agents.voice import io
    from livekit.agents.voice.room_io._input import _ParticipantAudioInputStream

    class PickupInput(io.AudioInput):
        def __init__(self) -> None:
            self.upstream = _ParticipantAudioInputStream(
                room, sample_rate=24000, num_channels=1, frame_size_ms=20,
                noise_cancellation=None, pre_connect_audio_handler=None)
            super().__init__(label='outbound-live-pickup', source=self.upstream)
            self.buffer = PickupAudioBuffer()
            if answer_is_known():
                self.buffer.arm()
            self.upstream.set_participant(participant)
            self.upstream.on_attached()
            room.on('reconnecting', self.on_reconnecting)
            self.task = asyncio.create_task(self.capture())

        def on_reconnecting(self, *_args: Any) -> None:
            if self.buffer.pickup_completed:
                self.buffer.clear()
            else:
                self.buffer.fail('caller transport reconnected during live pickup')

        async def capture(self) -> None:
            try:
                async for frame in self.upstream:
                    if answer_is_known():
                        self.buffer.arm()
                    self.buffer.push(frame)
                if self.buffer.pickup_completed:
                    self.buffer.close()
                elif not self.buffer.closed:
                    self.buffer.fail('caller audio capture ended')
            except asyncio.CancelledError:
                raise
            except Exception:
                self.buffer.fail('caller audio capture failed')

        def on_attached(self) -> None:
            self.buffer.attach()

        def on_detached(self) -> None:
            # Keep draining the upstream. Startup retains only post-answer
            # frames; a later intentional mute drops them instead of queuing.
            self.buffer.detach()

        async def __anext__(self) -> Any:
            return await self.buffer.next()

        async def aclose(self) -> None:
            room.off('reconnecting', self.on_reconnecting)
            self.buffer.close()
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            await self.upstream.aclose()

    return PickupInput()

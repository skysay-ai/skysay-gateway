# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One ordered spoken reply over engine-specific source preparation adapters."""
from __future__ import annotations

import copy
import asyncio
import math
import time
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

try:
    from .workflow_playback import WorkflowPlayback
    from .workflow_segment_delivery import CompletedSpeechSegment, WorkflowSegmentDelivery
    from .workflow_speech import MAX_REPLY_SEGMENTS, SpeechLine, SpeechRefused, WorkflowSpeechGate, speech_step_token
    from .workflow_turns import authorized_spoken_text
    from .capabilities.workflow_sources import MAX_CALL_ATTEMPTS
except ImportError:
    from workflow_playback import WorkflowPlayback
    from workflow_segment_delivery import CompletedSpeechSegment, WorkflowSegmentDelivery
    from workflow_speech import MAX_REPLY_SEGMENTS, SpeechLine, SpeechRefused, WorkflowSpeechGate, speech_step_token
    from workflow_turns import authorized_spoken_text
    from capabilities.workflow_sources import MAX_CALL_ATTEMPTS


@dataclass(frozen=True, slots=True)
class WorkflowSpokenSegment:
    """One actual speech attempt/ordinal, projected after its durable final."""
    request_id: str
    step_id: str
    language: str
    delivery: CompletedSpeechSegment


class _ActivePlayoutHandle:
    """A public SDK waiter with an active-audio budget and a whole-call end."""
    def __init__(self, handle, *, floor, budget, deadline, clock=time.monotonic):
        self.handle, self.floor, self.budget, self.deadline, self.clock = handle, floor, budget, deadline, clock
        self.id = handle.id

    @property
    def interrupted(self):
        return self.handle.interrupted

    def interrupt(self, **kwargs):
        return self.handle.interrupt(**kwargs)

    async def wait_for_playout(self):
        task = asyncio.create_task(self.handle.wait_for_playout())
        remaining, previous = self.budget, self.clock()
        paused = self.floor.paused
        try:
            while True:
                done, _pending = await asyncio.wait((task,), timeout=.05)
                now = self.clock()
                if not paused:
                    remaining -= max(0.0, now - previous)
                previous, paused = now, self.floor.paused
                if done:
                    return task.result()
                if now >= self.deadline or remaining <= 0:
                    raise SpeechRefused("workflow playout exhausted its active or call deadline")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class _FloorTimedPlayback(WorkflowPlayback):
    def __init__(self, session, *, floor, deadline, media_epoch):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
            raise SpeechRefused("workflow floor has no finite remaining call deadline")
        outer = self
        class SessionPort:
            def say(_self, *args, **kwargs):
                handle = session.say(*args, **kwargs)
                return _ActivePlayoutHandle(handle, floor=floor, budget=outer._budget, deadline=deadline)
            async def interrupt(_self, **kwargs):
                return await session.interrupt(**kwargs)
        super().__init__(SessionPort(), media_epoch=media_epoch)
        self._deadline, self._budget = deadline, 0.0

    def playout_timeout(self, _approved):
        return max(.001, self._deadline - time.monotonic())

    async def _play(self, approved):
        self._budget = super().playout_timeout(approved)
        return await super()._play(approved)


class _ReplySpeechGate(WorkflowSpeechGate):
    """Restart an unexposed suffix after this owner joined its previous attempt."""
    def __init__(self, line, *, media_epoch, sample_rate, completed_prefix):
        super().__init__(line, media_epoch=media_epoch, sample_rate=sample_rate)
        if type(completed_prefix) is not int or not 0 <= completed_prefix < len(line.segments):
            raise SpeechRefused("workflow replay has no bounded completed prefix")
        self._ordinal = completed_prefix


class WorkflowReplyOutput:
    def __init__(self, *, call, reply: dict[str, Any], session, media_epoch: int,
                 language: str, sample_rate: int, segmenter, preparation, project, remaining_source_budget,
                 audio_floor=None, floor_deadline=None) -> None:
        if (not isinstance(reply, dict) or not isinstance(reply.get("request_id"), str)
                or re.fullmatch(r"[A-Za-z0-9_-]{16,96}", reply["request_id"]) is None):
            raise SpeechRefused("workflow output requires its acknowledged request identity")
        self.call, self.reply = call, copy.deepcopy(reply)
        self.language, self.rate = language, sample_rate
        self.segmenter, self.preparation, self.project = segmenter, preparation, project
        self.remaining_source_budget = remaining_source_budget
        self._session = session
        self.audio_floor, self.floor_deadline = audio_floor, floor_deadline
        self.publisher = WorkflowPlayback(session, media_epoch=media_epoch)
        self.epoch = media_epoch
        self._started = self._completed = False
        self._last_journal = None
        self._delivery = self._journal = self._segment_waiter = None
        self._cancel_unexposed = False
        self._parked = None
        self._park_generation = 0
        self._floor_lock = asyncio.Lock()
        self._floor_released = asyncio.Event()
        self._floor_released.set()
        self._caller_held = False

    async def run(self) -> str:
        if self._started:
            raise SpeechRefused("workflow reply output cannot be run twice")
        self._started = True
        if self.audio_floor is not None:
            self.publisher = _FloorTimedPlayback(self._session, floor=self.audio_floor,
                deadline=self.floor_deadline, media_epoch=self.epoch)
        gate = None
        try:
            plans = self._preflight()
            for position, (index, step, line) in enumerate(plans):
                journal = self.call.workflow_delivery_journal(self.reply, step_index=index)
                ordinal = 0
                gate = WorkflowSpeechGate(line, media_epoch=self.epoch, sample_rate=self.rate)
                while ordinal < len(line.segments):
                    await self._wait_for_floor()
                    async def project(completed, step_id=step["step_id"]):
                        await self.project(WorkflowSpokenSegment(self.reply["request_id"], step_id, self.language, completed))
                    controller = WorkflowSegmentDelivery(gate=gate, publisher=self.publisher, journal=journal,
                        language=self.language, attempt_id=uuid4().hex,
                        preparation=self.preparation(gate, journal), project=project)
                    self._delivery, self._journal = controller, journal
                    self._park_generation = 0
                    self._segment_waiter = asyncio.create_task(controller.run())
                    try:
                        status = await self._segment_waiter
                    except asyncio.CancelledError:
                        if not self._cancel_unexposed or asyncio.current_task().cancelling():
                            raise
                        self._cancel_unexposed = False
                        # The aborted attempt kept its source: re-read the
                        # budget and refuse before the retry claims another.
                        needed = len(line.segments) - ordinal + sum(len(later.segments) for _, _, later in plans[position + 1:])
                        budget = self.remaining_source_budget()
                        if type(budget) is not int or not 0 <= budget <= MAX_CALL_ATTEMPTS or budget < needed:
                            raise SpeechRefused("workflow reply exceeds its remaining source capacity on retry")
                        # The old producer and its unexposed final are joined.
                        # Only this owner's acknowledged completed prefix is
                        # reused, with the unchanged exact line and ordinal.
                        gate = _ReplySpeechGate(line, media_epoch=self.epoch,
                                                sample_rate=self.rate, completed_prefix=ordinal)
                        self.publisher = _FloorTimedPlayback(self._session, floor=self.audio_floor,
                            deadline=self.floor_deadline, media_epoch=self.epoch)
                        continue
                    finally:
                        self._delivery = self._journal = self._segment_waiter = None
                    if status == "interrupted":
                        return status
                    if status != "completed_local":
                        raise SpeechRefused("workflow segment completion is uncertain")
                    ordinal += 1
                self._last_journal = journal
            self._completed = True
            return "completed_local"
        except BaseException:
            if gate is not None:
                gate.invalidate()
            await self.publisher.interrupt()
            raise
        finally:
            # A retired/completed owner has joined its publisher and final
            # receipt before the actor can replace it. Release the now-empty
            # physical floor; the actor's separate floor still blocks a new
            # business turn until its conversation handoff is acknowledged.
            if self.audio_floor is not None and not self._floor_released.is_set():
                self.audio_floor.resume()
            self.resume_for_caller()

    async def pause_for_caller(self):
        """Hold this line, exposed or not, while a caller sound is classified.

        Only the physical frames wait: the source attempt, its delivery and
        the business floor are untouched, so resuming plays the same audio.
        A line already played in full has nothing left to hold.
        """
        if self._completed:
            return True
        if self.audio_floor is None:
            return False
        if not self._caller_held:
            if self.audio_floor.hold(self) is not True:
                return False
            self._caller_held = True
        return True

    def resume_for_caller(self):
        if self._caller_held:
            self._caller_held = False
            self.audio_floor.release_hold(self)

    async def _wait_for_floor(self):
        if self.audio_floor is None:
            return await self._floor_released.wait()
        try:
            await asyncio.wait_for(self._floor_released.wait(), max(.001, self.floor_deadline - time.monotonic()))
        except TimeoutError:
            raise SpeechRefused("workflow floor exceeded the remaining call deadline") from None

    async def pause_for_floor(self):
        """Park the current exposed owner only after its physical drain ACK."""
        async with self._floor_lock:
            if self.audio_floor is None:
                raise SpeechRefused("workflow reply has no official audio floor")
            if self._parked is not None:
                return True
            self._floor_released.clear()
            if await self.audio_floor.pause_and_drain() is not True:
                # Nothing was parked: give the gate and the floor back first.
                self.audio_floor.resume()
                self._floor_released.set()
                raise SpeechRefused("workflow audio pause was not acknowledged")
            delivery, journal = self._delivery, self._journal
            if delivery is None or delivery._final_acknowledged:
                self._parked = (None, None)
                return True
            if not delivery._exposure_requested:
                self._cancel_unexposed = True
                waiter = self._segment_waiter
                if waiter is None:
                    raise SpeechRefused("unexposed workflow source has no owned join")
                waiter.cancel()
                try:
                    await waiter
                except asyncio.CancelledError:
                    pass
                self._parked = (None, None)
                return True
            if delivery._owner is None:
                raise SpeechRefused("workflow exposure has no durable source owner")
            self._park_generation = await journal.park_business(delivery._owner, generation=self._park_generation)
            self._parked = (journal, delivery._owner)
            return True

    async def resume_from_floor(self):
        async with self._floor_lock:
            if self.audio_floor is None or self._parked is None:
                return False
            journal, owner = self._parked
            if journal is not None:
                if await journal.resume_business(owner, generation=self._park_generation) is not True:
                    raise SpeechRefused("workflow business floor lost its one-time resume")
            self._parked = None
            self.audio_floor.resume()
            self._floor_released.set()
            return True

    def _preflight(self):
        plans, count = [], 0
        budget = self.remaining_source_budget()
        if type(budget) is not int or not 0 <= budget <= MAX_CALL_ATTEMPTS:
            raise SpeechRefused("workflow reply has no frozen remaining source budget")
        for index, step in enumerate(self.reply["steps"]):
            if step["spoken"] is not True:
                continue
            text = authorized_spoken_text(self.reply["pinned_contract"], step["step_id"], self.language)
            if text is None:
                raise SpeechRefused("workflow reply has no authorized text in the call language")
            try:
                segments = self.segmenter(text)
                line = SpeechLine(self.call.call_id, self.reply["run_id"],
                    self.reply["runtime_state"]["turn_count"], speech_step_token(step["step_id"]), text, segments)
            except (TypeError, ValueError):
                raise SpeechRefused("workflow reply has no bounded exact speech partition") from None
            count += len(segments)
            if count > MAX_REPLY_SEGMENTS or count > budget:
                raise SpeechRefused("workflow reply exceeds its remaining source or segment capacity")
            plans.append((index, step, line))
        return plans

    async def present(self, boundary: dict[str, int]) -> None:
        if not self._completed or self._last_journal is None:
            raise SpeechRefused("workflow reply has no complete spoken presentation")
        await self._last_journal.present(boundary)



@dataclass(frozen=True, slots=True)
class ControllerClip:
    """One public word with a closed source and an unexposed controller receipt."""
    attempt_id: str
    pcm_digest: str
    pcm: bytes = field(repr=False)
    sample_rate: int = 48000


class _ControllerResampledGate(WorkflowSpeechGate):
    """Approve at the controller track format, after finite public resampling."""
    def __init__(self, line, *, media_epoch):
        super().__init__(line, media_epoch=media_epoch, sample_rate=48000)
        self._source_rate = None
        self._resampler = None
        self._resampled_sealed = False

    def capture_frame(self, binding, frame):
        from livekit import rtc
        if (self._resampled_sealed or frame.num_channels != 1
                or type(frame.sample_rate) is not int or not 8000 <= frame.sample_rate <= 48000):
            raise SpeechRefused("controller source has no finite mono audio format")
        if self._source_rate is None:
            self._source_rate = frame.sample_rate
            if frame.sample_rate != 48000:
                self._resampler = rtc.AudioResampler(frame.sample_rate, 48000, num_channels=1)
        if frame.sample_rate != self._source_rate:
            raise SpeechRefused("controller source changed its audio format")
        frames = self._resampler.push(frame) if self._resampler is not None else (frame,)
        for converted in frames:
            super().capture_frame(binding, converted)

    def seal_authored(self, binding):
        if self._resampled_sealed:
            raise SpeechRefused("controller resampling cannot seal twice")
        self._resampled_sealed = True
        if self._resampler is not None:
            resampler, self._resampler = self._resampler, None
            for frame in resampler.flush():
                super().capture_frame(binding, frame)
        return super().seal_authored(binding)

    def invalidate(self):
        self._resampler = None
        super().invalidate()


#: The longest controller clip: 45 s of 48 kHz mono 16-bit PCM.
CONTROLLER_MAX_PCM_BYTES = 45 * 96000
#: Playout waits this long beyond the clip's own audio, so even a clip at the
#: byte cap has slack: the deadline is bounded by the cap plus this.
CONTROLLER_PLAYOUT_SLACK_SECONDS = 20.0


def _controller_playout_deadline(pcm_bytes):
    return pcm_bytes / 96000 + CONTROLLER_PLAYOUT_SLACK_SECONDS


class _ControllerTrackPublisher:
    """Finite approved controller PCM on the existing private track."""
    def __init__(self, player, *, media_epoch, valid, clock=time.monotonic):
        self.player, self.epoch, self.valid, self.clock = player, media_epoch, valid, clock
        self.handle = None
        self.started_at = self.ended_at = None
        self.closed = False

    def playout_timeout(self, approved):
        return _controller_playout_deadline(min(approved.candidate.samples * 2, CONTROLLER_MAX_PCM_BYTES))

    def stop(self):
        self.closed = True
        if self.handle is not None:
            self.handle.stop()

    async def interrupt(self):
        self.stop()
        if self.handle is not None:
            await asyncio.wait_for(self.handle.wait_for_playout(), 5)

    async def play(self, approved):
        candidate = approved.candidate
        if candidate.sample_rate != 48000:
            raise SpeechRefused("controller track requires its approved 48k format")
        return await self.play_pcm(candidate.pcm, attempt_id=candidate.binding.attempt_id,
                                   media_epoch=candidate.binding.media_epoch)

    async def play_pcm(self, pcm, *, attempt_id, media_epoch):
        from livekit import rtc
        try:
            from .workflow_playback import PlaybackResult
        except ImportError:
            from workflow_playback import PlaybackResult
        if (self.closed or self.handle is not None or type(pcm) is not bytes or not pcm
                or len(pcm) % 2 or len(pcm) > CONTROLLER_MAX_PCM_BYTES or media_epoch != self.epoch or not self.valid()):
            raise SpeechRefused("controller track has no current approved floor")
        submitted = 0
        async def frames():
            nonlocal submitted
            for offset in range(0, len(pcm), 1920):
                if self.closed or not self.valid():
                    raise SpeechRefused("controller track lost its current floor")
                data = pcm[offset:offset + 1920]
                if self.started_at is None:
                    self.started_at = self.clock()
                submitted += len(data) // 2
                yield rtc.AudioFrame(data=data, sample_rate=48000, num_channels=1,
                                     samples_per_channel=len(data) // 2)
        try:
            self.handle = self.player.play(frames())
            await asyncio.wait_for(self.handle.wait_for_playout(), _controller_playout_deadline(len(pcm)))
            self.ended_at = self.clock()
            if self.closed or self.handle.stopped or not self.valid():
                return PlaybackResult("interrupted", submitted, "controller-" + attempt_id,
                                      self.started_at or self.ended_at)
            if submitted != len(pcm) // 2 or self.started_at is None:
                raise SpeechRefused("controller track omitted approved audio")
            return PlaybackResult("completed_local", submitted, "controller-" + attempt_id,
                                  self.started_at)
        except BaseException:
            await self.interrupt()
            raise


@dataclass(frozen=True, slots=True)
class _ControllerWords:
    """Deferred exact offered words; no provider or PCM exists until playback."""
    text: str = field(repr=False)
    language: str
    options: tuple[str, ...] = field(repr=False)


class _ComposedControllerHandle:
    def __init__(self, audio, frames):
        self.stopped, self.publisher = False, None
        self._task = asyncio.create_task(audio._compose(frames, self))

    async def wait_for_playout(self):
        await asyncio.shield(self._task)

    def stop(self):
        self.stopped = True
        if self.publisher is not None:
            self.publisher.stop()
        self._task.cancel()


class _ComposedControllerPlayer:
    def __init__(self, audio):
        self.audio, self.handle = audio, None

    def play(self, frames):
        if self.handle is not None and not self.handle._task.done():
            raise SpeechRefused("controller composition already owns playback")
        self.handle = _ComposedControllerHandle(self.audio, frames)
        return self.handle

    def clear(self):
        if self.handle is not None:
            self.handle.stop()


class WorkflowControllerAudio:
    """Source adapter for the existing controller speaker, never a business turn.

    Only admitted public digit words and offered options enter a producer here.
    The private value is composed later from call-owned digit clip receipts;
    offered options are claimed before their deferred text reaches a producer.
    """
    def __init__(self, *, call, manifest, declaration, constructor, managed, episode, media_epoch, before_source,
                 resource=None):
        self.call, self.manifest = call, copy.deepcopy(manifest)
        self.declaration, self.constructor, self.managed = declaration, constructor, managed
        self.episode, self.media_epoch = episode, media_epoch
        self.before_source = before_source
        self.resource = resource
        self._ordinal = 0
        self._player = self._valid = self._active = None
        self._generation = 0
        self._speech_lock = asyncio.Lock()
        self._clip_frames, self._silences, self._word_lines = {}, {}, {}
        self._value_binding = self._context = self._offered_options = None
        self._closed = False
        self.composed_player = _ComposedControllerPlayer(self)

    def attach_player(self, player, *, valid, context=None, fixed_valid=None, offered_options=None):
        if self._player is not None or not callable(valid):
            raise SpeechRefused("controller track requires one explicit floor owner")
        self._player, self._valid, self._context = player, valid, context
        self._offered_options = offered_options
        self._fixed_valid = valid if fixed_valid is None else fixed_valid

    async def join_current(self):
        handle = self.composed_player.handle
        if handle is not None:
            await asyncio.gather(handle._task, return_exceptions=True)
        async with self._speech_lock:
            if self._active is not None:
                raise SpeechRefused("controller still owns an unjoined delivery")

    async def aclose(self):
        try:
            from .workflow_tasks import join_owned_task
        except ImportError:
            from workflow_tasks import join_owned_task
        self._closed = True
        self.stop()
        self.composed_player.clear()
        handle = self.composed_player.handle
        if handle is not None:
            try:
                await join_owned_task(handle._task)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
        await self.join_current()
        self._clip_frames.clear()
        self._silences.clear()
        self._word_lines.clear()
        self._value_binding = None

    def bind_value_field(self, field_name):
        if self._closed or field_name not in self.manifest.get("value_fields", ()) or not callable(self._context):
            raise SpeechRefused("controller value has no current supplied field")
        self._value_binding = (field_name, self.episode(), self._context())

    async def synthesize(self, text, language):
        from livekit import rtc
        if self._closed or language not in self.manifest.get("lines", {}):
            raise SpeechRefused("controller words have no admitted language")
        options = self._offered_options(language) if callable(self._offered_options) else ()
        if (type(options) is tuple and 1 <= len(options) <= 12 and text in options
                and all(type(option) is str and option.strip()
                    and len(option) <= self.manifest["option_max_characters"]
                    and len(option.encode("utf-8")) <= self.manifest["option_max_utf8"] for option in options)):
            if not callable(self._context) or len(self._word_lines) >= 128:
                raise SpeechRefused("controller offered words exceeded their owned bounded context")
            words = _ControllerWords(text, language, options)
            self._word_lines[id(words)] = (words, self.episode(), self._context(), self._generation)
            return (words,)
        matches = [key for key, value in self.manifest.get("lines", {}).get(language, {}).items()
                   if value.get("kind") == "clip" and value.get("text") == text]
        if len(matches) != 1:
            raise SpeechRefused("controller composition accepts public admitted word clips only")
        clip = await self.capture_clip(text, language, line_id=matches[0])
        frame = rtc.AudioFrame(data=clip.pcm, sample_rate=48000, num_channels=1, samples_per_channel=len(clip.pcm) // 2)
        self._clip_frames[id(frame)] = (frame, clip, matches[0], language)
        return (frame,)

    def silence(self, seconds):
        from livekit import rtc
        try:
            from .navigation_entry import DIGIT_GAP_SECONDS, GROUP_GAP_SECONDS
        except ImportError:
            from navigation_entry import DIGIT_GAP_SECONDS, GROUP_GAP_SECONDS
        if seconds not in (DIGIT_GAP_SECONDS, GROUP_GAP_SECONDS):
            raise SpeechRefused("controller gap is outside its declared composition")
        if seconds not in self._silences:
            pcm = bytes(round(seconds * 48000) * 2)
            self._silences[seconds] = rtc.AudioFrame(data=pcm, sample_rate=48000, num_channels=1,
                samples_per_channel=len(pcm) // 2)
        return self._silences[seconds]

    async def _compose(self, frames, handle):
        import hashlib
        chunks, parts, languages, words = [], [], set(), None
        async for frame in frames:
            if isinstance(frame, _ControllerWords):
                if words is not None or chunks:
                    raise SpeechRefused("controller offered words cannot be mixed with other audio")
                words = frame
                continue
            if words is not None or len(chunks) >= 63 or frame.sample_rate != 48000 or frame.num_channels != 1:
                raise SpeechRefused("controller composition exceeded its bounded audio format")
            if (owned := self._clip_frames.get(id(frame))) is not None and owned[0] is frame:
                clip = owned[1]
                if bytes(frame.data) != clip.pcm or hashlib.sha256(clip.pcm).hexdigest() != clip.pcm_digest:
                    raise SpeechRefused("controller word changed after its source approval")
                parts.append({"attempt_id": clip.attempt_id, "pcm_digest": clip.pcm_digest})
                languages.add(owned[3])
            elif not any(frame is gap for gap in self._silences.values()) or any(bytes(frame.data)):
                raise SpeechRefused("controller composition contains unowned audio")
            chunks.append(bytes(frame.data))
        if words is not None:
            await self._play_words(words, handle)
            return
        pcm = b"".join(chunks)
        binding = self._value_binding
        if (not 1 <= len(parts) <= 32 or len(languages) != 1 or not pcm or len(pcm) > CONTROLLER_MAX_PCM_BYTES
                or binding is None or self._ordinal >= 128 or self._player is None):
            raise SpeechRefused("controller composition has no bounded owned public parts")
        field_name, episode, revision = binding
        generation, epoch = self._generation, self.media_epoch()
        def valid():
            return (not handle.stopped and generation == self._generation and self._valid()
                    and episode == self.episode() and revision == self._context())
        async with self._speech_lock:
            paused = await self.before_source()
            if type(paused) is not int or paused < 1 or not valid():
                raise SpeechRefused("controller value has no acknowledged current floor")
            attempt, language = uuid4().hex, next(iter(languages))
            spec = {"attempt_id": attempt, "episode": episode, "line_id": "value", "ordinal": self._ordinal,
                    "language": language, "media_epoch": epoch, "manifest_digest": self.manifest["digest"],
                    "segments": [], "parts": parts, "field_name": field_name, "offered_options": []}
            self._ordinal += 1
            journal = self.call.workflow_controller_journal(spec)
            publisher = handle.publisher = _ControllerTrackPublisher(self._player, media_epoch=epoch, valid=valid)
            self._active = publisher
            owner, exposure = None, False
            try:
                owner = await self._operation(journal.claim_spec(spec), timeout=5)
                if owner is None:
                    raise SpeechRefused("controller value has no fresh composition owner")
                await self._operation(journal.approve_composed(owner, hashlib.sha256(pcm).hexdigest()), timeout=5)
                if not valid():
                    raise SpeechRefused("controller value became stale before exposure")
                exposure = True
                if await self._operation(journal.start(owner), timeout=5) is not True:
                    raise SpeechRefused("controller value has no fresh exposure acknowledgement")
                result = await publisher.play_pcm(pcm, attempt_id=attempt, media_epoch=epoch)
                status = await self._operation(journal.finish(owner, result.status), timeout=5)
                owner = None
                if status != "completed_local":
                    raise SpeechRefused("controller value did not complete its local delivery")
            finally:
                async def cleanup():
                    await publisher.interrupt()
                    if owner is not None:
                        await self._operation(journal.finish(owner, "uncertain" if exposure else "aborted_unexposed"), timeout=5)
                if owner is not None:
                    await self._operation(cleanup(), timeout=15)
                if self._active is publisher:
                    self._active = None

    async def _play_words(self, words, handle):
        try:
            from .navigation_entry import ProbeLine
        except ImportError:
            from navigation_entry import ProbeLine
        binding = self._word_lines.pop(id(words), None)
        if binding is None or binding[0] is not words:
            raise SpeechRefused("controller offered words are not call-owned")
        _words, episode, revision, generation = binding
        def valid():
            return (not handle.stopped and not self._closed and generation == self._generation
                and episode == self.episode() and revision == self._context()
                and words.options == self._offered_options(words.language))
        async with self._speech_lock:
            if not valid():
                raise SpeechRefused("controller offered words lost their current prompt")
            result = await self._speak_line(ProbeLine(words.text, "offered-option", words.language, True),
                generation, offered_options=words.options, line_valid=valid, handle=handle)
            if not result.completed:
                raise SpeechRefused("controller offered words did not complete their local delivery")

    def stop(self):
        self._generation += 1
        if self._active is not None:
            self._active.stop()

    async def speak(self, line):
        """Exact admitted probe/settling words; no model/session speech path."""
        generation = self._generation
        async with self._speech_lock:
            return await self._speak_line(line, generation)

    async def _speak_line(self, line, generation, *, offered_options=(), line_valid=lambda: True, handle=None):
        try:
            from .workflow_authored_tts import AuthoredTtsPreparation
            from .navigation_entry import ProbeSpeech
        except ImportError:
            from workflow_authored_tts import AuthoredTtsPreparation
            from navigation_entry import ProbeSpeech
        definition = self.manifest.get("lines", {}).get(line.language, {}).get(line.kind)
        admitted = (line.kind == "offered-option" and line.text in offered_options) if offered_options else (
            isinstance(definition, dict) and definition.get("kind") == "line" and definition.get("text") == line.text)
        if (self._closed or not line.fixed or not admitted or self._ordinal >= 128
                or self._player is None or self._active is not None):
            raise SpeechRefused("controller words have no admitted fixed-line player")
        paused = await self.before_source()
        if (type(paused) is not int or paused < 1 or generation != self._generation
                or not self._valid() or not line_valid() or (not offered_options and not self._fixed_valid())):
            raise SpeechRefused("controller words have no acknowledged current floor")
        attempt, epoch = uuid4().hex, self.media_epoch()
        spec = {"attempt_id": attempt, "episode": self.episode(), "line_id": line.kind,
                "ordinal": self._ordinal, "language": line.language, "media_epoch": epoch,
                "manifest_digest": self.manifest["digest"], "segments": [line.text],
                "parts": [], "field_name": "", "offered_options": list(offered_options)}
        self._ordinal += 1
        exact = SpeechLine(self.call.call_id, self.call.config["workflow_run_id"], 0,
                          "controller-" + line.kind, line.text, (line.text,))
        gate = _ControllerResampledGate(exact, media_epoch=epoch)
        journal = self.call.workflow_controller_journal(spec)
        publisher = _ControllerTrackPublisher(self._player, media_epoch=epoch,
            valid=lambda: generation == self._generation and self._valid() and line_valid()
                and (bool(offered_options) or self._fixed_valid()) and self.episode() == spec["episode"])
        if handle is not None:
            handle.publisher = publisher
        self._active = publisher
        async def project(_completed):
            pass  # Controller speech never presents a business question.
        delivery = None
        try:
            preparation = AuthoredTtsPreparation(declaration=self.declaration, constructor=self.constructor,
                                               journal=journal, managed=self.managed, resource=self.resource)
            delivery = WorkflowSegmentDelivery(gate=gate, publisher=publisher, journal=journal,
                language=line.language, attempt_id=attempt, preparation=preparation, project=project)
            status = await delivery.run()
            ended = publisher.ended_at or time.monotonic()
            return ProbeSpeech(publisher.started_at or ended, ended, status == "completed_local", publisher.started_at is not None)
        finally:
            gate.invalidate()
            if (self._active is publisher and (delivery is None
                    or delivery._final_acknowledged and not delivery._children)):
                self._active = None

    @staticmethod
    async def _operation(operation, *, timeout=30):
        try:
            from .workflow_tasks import join_owned_task
        except ImportError:
            from workflow_tasks import join_owned_task
        task = asyncio.create_task(operation)
        try:
            async with asyncio.timeout(timeout):
                return await asyncio.shield(task)
        finally:
            if not task.done():
                task.cancel()
            try:
                await join_owned_task(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise

    async def capture_clip(self, text, language, *, line_id):
        async with self._speech_lock:
            return await self._capture_clip(text, language, line_id=line_id)

    async def _capture_clip(self, text, language, *, line_id):
        try:
            from .workflow_authored_tts import AuthoredTtsPreparation
        except ImportError:
            from workflow_authored_tts import AuthoredTtsPreparation
        definition = self.manifest.get("lines", {}).get(language, {}).get(line_id)
        if (self._closed or not isinstance(definition, dict) or definition.get("kind") != "clip"
                or definition.get("text") != text or self._ordinal >= 128):
            raise SpeechRefused("controller clip is outside its frozen public word manifest")
        paused_generation = await self.before_source()
        if type(paused_generation) is not int or paused_generation < 1:
            raise SpeechRefused("controller source has no acknowledged business floor")
        attempt = uuid4().hex
        epoch = self.media_epoch()
        spec = {"attempt_id": attempt, "episode": self.episode(), "line_id": line_id,
                "ordinal": self._ordinal, "language": language, "media_epoch": epoch,
                "manifest_digest": self.manifest["digest"], "segments": [text],
                "parts": [], "field_name": "", "offered_options": []}
        self._ordinal += 1
        line = SpeechLine(self.call.call_id, self.call.config["workflow_run_id"], 0,
                          "controller-" + line_id, text, (text,))
        gate = _ControllerResampledGate(line, media_epoch=epoch)
        journal = self.call.workflow_controller_journal(spec)
        owner = None
        try:
            binding = gate.open_segment(attempt_id=attempt)
            owner = await self._operation(journal.claim(binding, gate, language), timeout=5)
            if owner is None:
                raise SpeechRefused("controller clip has no fresh source owner")
            preparation = AuthoredTtsPreparation(declaration=self.declaration, constructor=self.constructor,
                                               journal=journal, managed=self.managed, resource=self.resource)
            speech = await preparation.prepare(gate, binding, owner, operation=self._operation)
            final = await self._operation(journal.finish(owner, "aborted_unexposed"), timeout=5)
            if final != "aborted_unexposed":
                raise SpeechRefused("controller clip has no unexposed terminal receipt")
            owner = None
            candidate = speech.candidate
            return ControllerClip(attempt, candidate.pcm_digest, candidate.pcm)
        finally:
            gate.invalidate()
            if owner is not None:
                await self._operation(journal.finish(owner, "aborted_unexposed"), timeout=5)


def create_workflow_audio_floor(sink):
    """Drain each public SDK frame before pausing the room's existing output.

    The leaf's pause may discard queued audio. This wrapper gates new capture,
    joins the leaf's actual playout acknowledgement, then invokes its public
    pause. It owns no provider, publication, automatic reply or input track.
    """
    from livekit.agents.voice import io
    try:
        from .workflow_tasks import join_owned_task
    except ImportError:
        from workflow_tasks import join_owned_task

    class WorkflowAudioFloor(io.AudioOutput):
        def __init__(self):
            super().__init__(label="WorkflowAudioFloor", capabilities=io.AudioOutputCapabilities(pause=True),
                             sample_rate=sink.sample_rate)
            if not sink.can_pause:
                raise SpeechRefused("workflow output has no official pause capability")
            self._sink = sink
            self._gate = asyncio.Event()
            self._gate.set()
            # A caller sound's hold, separate from the navigation gate: the
            # SDK's resume() before each speech never releases it.
            self._held = asyncio.Event()
            self._held.set()
            self._holder = None
            self._capture_lock = asyncio.Lock()
            self._generation = 0
            self._accepting = True
            self._segment = self._clear_pending = False
            self._played = 0.0

        @property
        def paused(self):
            return not self._gate.is_set() or not self._held.is_set()

        def _complete(self, *, interrupted):
            if not self._segment:
                return
            super().flush()
            self.on_playback_finished(playback_position=self._played, interrupted=interrupted)
            self._segment = self._clear_pending = False
            self._played = 0.0

        def _ack(self, event, frame):
            position = event.playback_position
            if (type(position) not in (int, float) or not math.isfinite(position)
                    or not 0 <= position <= frame.duration + 1 / frame.sample_rate):
                raise SpeechRefused("SDK output returned an invalid playout acknowledgement")
            self._played += position

        async def capture_frame(self, frame):
            generation = self._generation
            while True:
                await self._gate.wait()
                await self._held.wait()
                async with self._capture_lock:
                    if generation != self._generation or not self._accepting:
                        raise asyncio.CancelledError
                    if not self._gate.is_set() or not self._held.is_set():
                        continue
                    await super().capture_frame(frame)
                    if not self._segment:
                        self._segment = True
                        self.on_playback_started(created_at=time.time())
                    submitted = False
                    try:
                        await self._sink.capture_frame(frame)
                        submitted = True
                        self._sink.flush()
                        event = await self._sink.wait_for_playout()
                        self._ack(event, frame)
                    except asyncio.CancelledError:
                        self.clear_buffer()
                        if submitted:
                            cleanup = asyncio.create_task(self._sink.wait_for_playout())
                            try:
                                await join_owned_task(cleanup)
                            finally:
                                if cleanup.done() and not cleanup.cancelled() and cleanup.exception() is None:
                                    self._ack(cleanup.result(), frame)
                                self._complete(interrupted=True)
                        else:
                            self._complete(interrupted=True)
                        raise
                    except BaseException:
                        self.clear_buffer()
                        self._complete(interrupted=True)
                        raise
                    if event.interrupted or generation != self._generation:
                        self._accepting = False
                        self._complete(interrupted=True)
                        raise asyncio.CancelledError
                    return

        def flush(self):
            if self._capture_lock.locked():
                raise SpeechRefused("workflow audio flushed before its captured frame joined")
            self._complete(interrupted=self._clear_pending)

        def pause(self):
            self._gate.clear()

        async def pause_and_drain(self):
            generation = self._generation
            self.pause()
            async with self._capture_lock:
                # The navigation gate itself, never the caller hold, proves
                # this pause stayed closed across the drain.
                if generation != self._generation or self._gate.is_set() or not self._accepting:
                    return False
                self._sink.pause()
                return True

        def hold(self, holder):
            """Hold every frame after the one in flight; nothing is discarded.

            Synchronous: the caller's VAD actor never waits on playout. The
            frame in flight finishes; the sink is not paused, so no queued
            audio is dropped.
            """
            if not self._accepting:
                return False
            self._holder = holder
            self._held.clear()
            return True

        def release_hold(self, holder):
            # Only the current holder releases: a finished line never frees a later one's hold.
            if self._holder is holder:
                self._holder = None
                self._held.set()

        def resume(self):
            self._sink.resume()
            self._accepting = True
            if not self._segment:
                self._clear_pending = False
            self._gate.set()

        def clear_buffer(self):
            self._generation += 1
            self._accepting = False
            self._clear_pending = True
            self._gate.set()
            self._holder = None
            self._held.set()
            self._sink.resume()
            self._sink.clear_buffer()
            if not self._capture_lock.locked():
                self._complete(interrupted=True)

        async def drain(self):
            async with self._capture_lock:
                if self._clear_pending:
                    self._complete(interrupted=True)

    return WorkflowAudioFloor()

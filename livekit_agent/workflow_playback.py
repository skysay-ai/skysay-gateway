# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Owned public LiveKit playback of already approved PCM, with no model output.

Use a dedicated session from create_workflow_playback_session, never an active
native duplex session. The one designed exception is the live output role
(``workflow_live_output``): that root starts this same session with the call's
GPT-Live model and plays no approved PCM through this module at all. The
controller journals possible exposure before play.
Local completion is not an acoustic, network-egress or human-hearing receipt.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

try:
    from .workflow_speech import AuthoredSpeech, SpeechRefused
except ImportError:  # The voice image installs these modules at /app.
    from workflow_speech import AuthoredSpeech, SpeechRefused


def create_workflow_playback_session():
    from livekit.agents import AgentSession
    # No automatic generation, false-interruption replay, away prompt or
    # model/TTS. Input configuration belongs to a future explicit adapter.
    return AgentSession(stt=None, llm=None, tts=None, vad=None, turn_handling={
        "turn_detection": "manual",
        "interruption": {"enabled": True, "resume_false_interruption": False,
                         "false_interruption_timeout": None},
        "preemptive_generation": {"enabled": False},
    }, user_away_timeout=None)


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    status: str
    samples_submitted: int
    speech_id: str
    scheduled_at: float


class WorkflowPlayback:
    def __init__(self, session: Any, *, media_epoch: int, startup_timeout: float = 20.0,
                 cleanup_timeout: float = 5.0) -> None:
        if (type(media_epoch) is not int or media_epoch < 0
                or type(startup_timeout) not in (int, float) or not math.isfinite(startup_timeout) or not 0 < startup_timeout <= 60
                or type(cleanup_timeout) not in (int, float) or not math.isfinite(cleanup_timeout)
                or not 0 < cleanup_timeout <= 15):
            raise ValueError("playback requires bounded deadlines and epoch")
        self._session, self._epoch = session, media_epoch
        self._startup_timeout, self._cleanup_timeout = startup_timeout, cleanup_timeout
        self._active: asyncio.Task[PlaybackResult] | None = None
        self._handle: Any = None
        self._say_attempted = False
        self._used: set[str] = set()
        self._closed = False

    def playout_timeout(self, approved: AuthoredSpeech) -> float:
        """Allow the supplied PCM's duration plus bounded SDK startup slack."""
        return approved.candidate.samples / approved.candidate.sample_rate + self._startup_timeout

    async def play(self, approved: AuthoredSpeech) -> PlaybackResult:
        if (self._closed or self._active is not None or not isinstance(approved, AuthoredSpeech)
                or approved.candidate.binding.media_epoch != self._epoch
                or approved.candidate.binding.attempt_id in self._used or len(self._used) >= 4096):
            raise SpeechRefused("playback has no fresh current authorization")
        self._used.add(approved.candidate.binding.attempt_id)
        self._active = asyncio.create_task(self._play(approved))
        try:
            # Shield retains ownership if the caller is cancelled; interrupt
            # the actual SDK speech and join it before allowing any handoff.
            return await asyncio.shield(self._active)
        except asyncio.CancelledError:
            self._closed = True
            await self._settle_interrupt()
            raise
        finally:
            if self._active is not None and self._active.done():
                self._active = None

    async def _play(self, approved: AuthoredSpeech) -> PlaybackResult:
        if self._closed:
            raise asyncio.CancelledError
        from livekit import rtc
        candidate = approved.candidate
        submitted = 0
        complete = False

        async def frames():
            nonlocal submitted, complete
            width = candidate.sample_rate//50  # At most 20 ms per PCM frame.
            for offset in range(0, len(candidate.pcm), width*2):
                if self._closed:
                    # End the frames; never raise into the SDK's speech task.
                    # ``complete`` stays false, so this is never clean speech:
                    # the closed publisher reports it as interrupted.
                    return
                data = candidate.pcm[offset:offset+width*2]
                submitted += len(data)//2
                yield rtc.AudioFrame(data=data, sample_rate=candidate.sample_rate,
                                     num_channels=1, samples_per_channel=len(data)//2)
            complete = True

        try:
            self._handle = None
            self._say_attempted = True
            scheduled_at = time.monotonic()
            self._handle = self._session.say("", audio=frames(), allow_interruptions=True, add_to_chat_ctx=False)
            speech_id = self._handle.id
            if not isinstance(speech_id, str) or not speech_id.strip() or len(speech_id) > 256:
                raise SpeechRefused("SDK speech has no bounded recording identity")
            try:
                await asyncio.wait_for(self._handle.wait_for_playout(), self.playout_timeout(approved))
            except TimeoutError:
                self._closed = True
                raise SpeechRefused("SDK playout deadline expired; delivery is uncertain") from None
            if self._closed or self._handle.interrupted:
                self._closed = True
                return PlaybackResult("interrupted", submitted, speech_id, scheduled_at)
            if not complete or submitted != candidate.samples:
                self._closed = True
                raise SpeechRefused("SDK completion omitted supplied audio")
            return PlaybackResult("completed_local", submitted, speech_id, scheduled_at)
        except BaseException:
            self._closed = True
            await self._stop_handle()
            raise

    async def _stop_handle(self) -> None:
        if self._handle is None:
            if self._say_attempted:
                # say() can fail after scheduling but before returning its
                # handle. This session owns only workflow PCM, so the public
                # session-wide barrier safely settles that unknown outcome.
                await asyncio.wait_for(self._session.interrupt(force=True), self._cleanup_timeout)
            return
        self._handle.interrupt(force=True)
        # Cancelling wait_for_playout's waiter alone never stops the SDK.
        await asyncio.wait_for(self._handle.wait_for_playout(), self._cleanup_timeout)

    async def _settle_interrupt(self) -> None:
        """Own cleanup despite repeated cancellation of the awaiting caller."""
        async def cleanup():
            await self._stop_handle()
            if self._active is not None:
                try:
                    await self._active
                except asyncio.CancelledError:
                    pass
        cleanup_task = asyncio.create_task(cleanup())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
        cleanup_task.result()

    async def interrupt(self) -> None:
        self._closed = True
        await self._settle_interrupt()
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError

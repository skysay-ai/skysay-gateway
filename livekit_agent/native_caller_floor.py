# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Local speech evidence for reclaiming a native model's pending close.

Provider user-state notifications may be transcript-driven and arrive after a
short closing deadline. Observe the same caller input before model delivery;
never make a platform/billing ending reclaimable. No transcript or PCM is kept.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable


class NativeCallerFloor:
    def __init__(self, vad: Any, note: Callable[[bool], None], failed: Callable[[], None]) -> None:
        self.stream = vad.stream()
        self.note = note
        self.failed = failed
        self.closed = False
        self.broken = False
        self.resampler = None
        self.input_rate = None
        self.task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for event in self.stream:
                kind = str(event.type).lower()
                if kind.endswith('start_of_speech'):
                    self.note(True)
                elif kind.endswith('end_of_speech'):
                    self.note(False)
            if not self.closed:
                self._fail()
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed:
                self._fail()

    def push(self, frame: Any) -> None:
        if self.closed or self.broken:
            return
        try:
            from livekit import rtc
            if self.input_rate != frame.sample_rate:
                if self.input_rate is not None:
                    raise ValueError('caller input sample rate changed')
                self.input_rate = frame.sample_rate
                if frame.sample_rate != 16000:
                    self.resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate, output_rate=16000, num_channels=1)
            for sampled in self.resampler.push(frame) if self.resampler else [frame]:
                self.stream.push_frame(sampled)
        except Exception:
            self._fail()

    def _fail(self) -> None:
        if not self.closed and not self.broken:
            self.broken = True
            self.failed()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        await self.stream.aclose()


def install_native_caller_floor(session: Any, vad: Any, *, note, failed) -> NativeCallerFloor:
    """Wrap the SDK's installed input; its attachment/mute lifecycle stays owned by it."""
    from livekit.agents.voice import io
    source = session.input.audio
    if source is None:
        raise RuntimeError('native caller floor requires the installed caller input')
    observer = NativeCallerFloor(vad, note, failed)

    class ObservedInput(io.AudioInput):
        def __init__(self) -> None:
            super().__init__(label='native-caller-floor', source=source)

        async def __anext__(self):
            try:
                frame = await source.__anext__()
            except StopAsyncIteration:
                raise
            except Exception:
                observer._fail()
                raise
            observer.push(frame)
            return frame

    session.input.audio = ObservedInput()
    return observer

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Exact local native-unit counters for TTS adapters without SDK usage fields.

Only an integer total is retained.  Text itself is never exported, logged, or
kept after the worker writes the provider request; this keeps the accounting
path independent from caller/agent content.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class NativeTTSUsageError(RuntimeError):
    """A worker attempted to add invalid TTS accounting evidence."""


class Utf8ByteCounter:
    """Count exactly the UTF-8 bytes delivered to a Fish Audio stream."""

    def __init__(self) -> None:
        self._total = 0

    def record(self, text: str) -> None:
        if not isinstance(text, str):
            raise NativeTTSUsageError("TTS input must be text")
        self._total += len(text.encode("utf-8"))

    def snapshot(self) -> int:
        return self._total


async def await_fish_stream_tasks(
    *,
    input_task: asyncio.Task[Any],
    send_task: asyncio.Task[Any],
    receive_task: asyncio.Task[Any],
    cancel: Callable[..., Awaitable[None]],
) -> None:
    """Drive Fish's three stream tasks without sending after provider finish.

    ``FIRST_EXCEPTION`` waits for every task when the receiver returns a
    successful ``finish`` frame. That would leave a queued sentence sender
    alive after the provider has ended the stream, so it could write and count
    text that the provider will never synthesize. Observe completions one at a
    time: ordinary input/sender completion still waits for the receiver, while
    receiver completion cancels remaining input/sends immediately.
    """
    pending: set[asyncio.Task[Any]] = {input_task, send_task, receive_task}
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # A normal terminal Fish frame wins over a queued sender/input even
            # when a sibling happened to be cancelled in the same event-loop turn.
            # Treating that sibling first would turn a successfully finished stream
            # into cancellation and skip the caller's ordinary terminal cleanup.
            if receive_task in done:
                if receive_task.cancelled():
                    await cancel(*pending)
                    raise asyncio.CancelledError
                receive_error = receive_task.exception()
                if receive_error is not None:
                    await cancel(*pending)
                    raise receive_error
                await cancel(*pending)
                return
            for task in done:
                if task.cancelled():
                    await cancel(*pending)
                    raise asyncio.CancelledError
                error = task.exception()
                if error is not None:
                    await cancel(*pending)
                    raise error
    except asyncio.CancelledError:
        # Barge-in/call teardown can cancel the outer stream driver while all
        # three provider tasks are still live. The tasks own sockets and the
        # Fish text counter, so leaving them behind can keep media and billing
        # work alive after the call is gone.
        await cancel(input_task, send_task, receive_task)
        raise


__all__ = ["NativeTTSUsageError", "Utf8ByteCounter", "await_fish_stream_tasks"]

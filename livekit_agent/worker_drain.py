# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Stop claiming on SIGTERM, but finish what is already claimed.

Every worker here that CLAIMS a `worker_jobs` row has the same obligation and
the same failure. A promote recreates its container; the default signal
disposition kills the process instantly -- no `CancelledError`, so a
settle-on-cancellation path never runs -- and if that lands between the claim
and the completion report, the row stays `status=running, locked_by=<worker>`
with nothing reclaiming it. For the outbound originator that permanently
burns one of an organization's bounded concurrent-call slots; for the
simulations room worker it can leave a dispatch half-created, so a retry
re-dispatches into a room that already has its participants.

This module exists because the first version of that fix lived inside one
worker and the other claiming role was simply forgotten -- a reviewer found it,
not a test. Making the drain something a claiming loop IMPORTS rather than
remembers is what stops the next role being added without it.

Pure module: no livekit import, no network, no side effects.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

#: The signals a container runtime sends to ask a process to stop. A second
#: one is taken as "stop arguing": the operator has decided the work in flight
#: is worth less than the delay, exactly as livekit-agents' own CLI treats it.
STOP_SIGNALS = ("SIGTERM", "SIGINT")


def install_stop_handlers(stop: asyncio.Event) -> Callable[[], None]:
    """Route stop signals into ``stop``; return a callable that undoes it.

    Best-effort by design: a platform without asyncio signal handlers (or a
    thread that is not the main one, which is how the tests drive this) keeps
    the previous behaviour rather than failing to start. A worker that cannot
    install a drain handler is worse off than one that can, but still better
    off than one that refuses to run.
    """
    import os
    import signal

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return lambda: None
    installed: list[Any] = []
    for name in STOP_SIGNALS:
        number = getattr(signal, name, None)
        if number is None:
            continue

        def _handle(number: Any = number) -> None:
            if stop.is_set():
                # Already draining and asked again: the second signal is a
                # hard stop, which is what an operator expects from pressing
                # it twice. The claimed job is left for the control plane.
                os._exit(1)
            stop.set()

        try:
            loop.add_signal_handler(number, _handle)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(number)

    def _release() -> None:
        for number in installed:
            try:
                loop.remove_signal_handler(number)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    return _release


async def sleep_unless_stopped(stop: asyncio.Event, seconds: float) -> bool:
    """Wait out the poll interval, or return early once a stop is requested.

    Returns whether the worker should keep polling. A plain ``sleep`` here
    would make every stop wait out the remainder of the interval before the
    process even noticed -- time taken directly out of the container's stop
    grace, which is the budget the work in flight needs.
    """
    if stop.is_set():
        return False
    try:
        await asyncio.wait_for(stop.wait(), timeout=max(0.0, float(seconds)))
    except (asyncio.TimeoutError, TimeoutError):
        return True
    return False

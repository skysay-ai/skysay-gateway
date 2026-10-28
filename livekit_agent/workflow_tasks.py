# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Shared task ownership for workflow HTTP, audio and recognition cleanup."""
from __future__ import annotations

import asyncio


async def join_owned_task(cleanup: asyncio.Task) -> None:
    """Retain ownership through repeated cancellation, then propagate it."""
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.wait((cleanup,))
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError

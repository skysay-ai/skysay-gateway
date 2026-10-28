# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Output evidence for a requested browser opening, frozen before teardown.

Count BEFORE forwarding, with no await between the fence and increment. A
frame in flight or buffered downstream is conservatively nonzero, even if it
never reaches a speaker. Zero is provable only when every frame crossed here.
"""
from __future__ import annotations
from typing import Any


class BrowserStartup:
    def __init__(self) -> None:
        self.frames = 0
        self.installed = False
        self.frozen = False
        self.failed = False

    def forward_frame(self) -> bool:
        if self.frozen:
            return False
        self.frames += 1
        return True

    def opening_failed(self, error: BaseException | None) -> bool:
        # CancelledError is a caller withdrawing work, never a provider fault.
        import asyncio
        if error is None or isinstance(error, asyncio.CancelledError) or self.frames:
            return False
        self.failed = True
        self.frozen = True
        return True

    def freeze(self, session_id: str) -> dict[str, str]:
        self.frozen = True
        return {"session_id": session_id,
                "outcome": "browser_startup_failed" if self.failed else "worker_ended",
                "output": "nonzero" if self.frames else "zero" if self.installed else "unknown"}


async def publish_before_room_delete(*, witness: BrowserStartup, call: Any,
                                     persist: Any, emit: Any, timeout: float = 5.0) -> None:
    """Freeze first, persist for crash replay, then bound the nonterminal POST.

    Network failure never prevents the caller deleting the room. A late HTTP
    completion can only publish the same immutable decision, never late audio.
    """
    import asyncio
    call.browser_outcome = witness.freeze(call.browser_session_id)
    try:
        persist(browser_outcome=call.browser_outcome, browser_outcome_pending=True)
        await asyncio.wait_for(asyncio.to_thread(
            call.report_lifecycle, event_id=f"{call.call_id}:browser-outcome",
            event_type="browser.outcome", source_component="livekit_room"), timeout=timeout)
    except Exception as exc:
        emit("browser_outcome_report_failed", error_type=type(exc).__name__)

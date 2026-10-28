# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""A funded deadline independent of heartbeat and occupancy I/O."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable


class FundedRuntime:
    def __init__(
        self, config: dict[str, Any], *, monotonic=time.monotonic, wall=time.time
    ):
        self.monotonic, self.wall = monotonic, wall
        self.enabled = config.get("enabled") is True
        self.generation = int(config.get("generation") or 0)
        self.seconds = int(
            config.get("authorized_seconds") or config.get("initial_seconds") or 0
        )
        self.answered = config.get("answered") is True
        self.outcome = "pending"
        self.renew_before = 60
        self.maximum = float("inf")
        self.anchor = self.monotonic()
        if self.enabled:
            try:
                self.anchor = self._anchor(config["started_at"])
                if self.seconds <= 0:
                    raise ValueError("invalid initial funding")
            except (KeyError, ValueError, TypeError):
                # Corrupt/missing authorization is a deadline now, not infinity.
                self.seconds = 0

    def _anchor(self, timestamp: str) -> float:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("authorization needs an absolute timestamp")
        return self.monotonic() - max(0.0, self.wall() - parsed.timestamp())

    def adopt(self, response: Any) -> None:
        if not self.enabled or not isinstance(response, dict):
            return
        grant = response.get("funded_runtime")
        if not isinstance(grant, dict):
            self.poll_failed()
            return
        generation, seconds = grant.get("generation"), grant.get("authorized_seconds")
        if (
            type(generation) is not int
            or type(seconds) is not int
            or generation < self.generation
            or seconds < self.seconds
        ):
            return
        try:
            anchor = self._anchor(grant["answered_at"])
        except (KeyError, TypeError, ValueError):
            return
        # Once answered, a delayed/reordered response cannot reset the clock.
        if not self.answered:
            self.anchor, self.answered = anchor, True
        self.generation, self.seconds = generation, seconds
        self.outcome = str(grant.get("outcome") or "pending")

    def poll_failed(self) -> None:
        # A transport failure cannot retract the last confirmed shortfall.
        if self.outcome != "still_short":
            self.outcome = "pending"

    def request_generation(self) -> int | None:
        if (
            self.enabled
            and self.seconds < self.maximum
            and self.monotonic() >= self.anchor + self.seconds - self.renew_before
        ):
            return self.generation
        return None

    async def watch(
        self,
        *,
        arbiter,
        max_seconds: float,
        is_finishing: Callable[[], bool],
        emit,
        duration_ending: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        """End the call at the funded allowance, or at the call limit if tighter.

        ``duration_ending`` is the navigation policy's one decision about the
        call-limit ending (``max_call_duration_reached`` in a conversation,
        ``ivr_timeout`` / ``deadline`` while the call waits in a menu or a
        queue); this timer is one of its producers, so it asks rather than
        stamps. The funding ending is never relabelled: a funded hold is cut by
        its allowance exactly as before.
        """

        if not self.enabled:
            return
        self.maximum = max_seconds
        while not is_finishing():
            limit = min(self.seconds, max_seconds)
            remaining = self.anchor + limit - self.monotonic()
            if remaining <= 0:
                reason = (
                    "max_call_duration_reached"
                    if max_seconds <= self.seconds
                    else "funding_exhausted"
                )
                detail = ""
                if reason == "max_call_duration_reached" and duration_ending is not None:
                    reason, detail = duration_ending()
                if reason == "funding_exhausted":
                    detail = (
                        "insufficient_balance"
                        if self.outcome == "still_short"
                        else "renewal_unavailable"
                    )
                won = arbiter.end(
                    reason, detected_via="worker_timer", reason_detail=detail
                )
                emit(
                    "funded_runtime_expired",
                    generation=self.generation,
                    authorized_seconds=self.seconds,
                    timer_lateness_ms=round(max(0, -remaining) * 1000),
                    ending_won=won,
                    reason_detail=detail,
                )
                return
            # Tight bounded timer; no API or occupancy calls on this task.
            await asyncio.sleep(min(remaining, 0.25))

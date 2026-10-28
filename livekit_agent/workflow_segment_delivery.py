# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive single-segment controller: journal before audio, final before text.

The journal is an authenticated async backend port, not a voice-side store.
Only an authored TTS source reaches it; the live role speaks without segments.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

try:
    from .workflow_speech import SpeechBinding, SpeechRefused, WorkflowSpeechGate
    from .workflow_playback import WorkflowPlayback
    from .workflow_speech_preparation import SpeechPreparation
except ImportError:
    from workflow_speech import SpeechBinding, SpeechRefused, WorkflowSpeechGate
    from workflow_playback import WorkflowPlayback
    from workflow_speech_preparation import SpeechPreparation


class DeliveryJournalPort(Protocol):
    async def claim(self, binding: SpeechBinding, gate: WorkflowSpeechGate, language: str) -> Any | None: ...
    async def start(self, owner: Any) -> bool: ...
    async def finish(self, owner: Any, status: str) -> str: ...


@dataclass(frozen=True, slots=True)
class CompletedSpeechSegment:
    """Durably acknowledged local completion, never proof of human hearing."""
    binding: SpeechBinding
    text: str = field(repr=False)
    pcm_digest: str
    sample_rate: int
    source_samples: int
    speech_id: str
    scheduled_at: float
    samples_submitted: int
    proof_kind: str


class WorkflowSegmentDelivery:
    """One owner; cancellation invalidates approval before cancelling children.

    A possibly exposed segment is never replayed. A cleanup timeout retains
    child ownership and refuses handoff; it cannot be called a clean stop.
    """

    def __init__(self, *, gate: WorkflowSpeechGate, publisher: WorkflowPlayback,
                 journal: DeliveryJournalPort, language: str, attempt_id: str,
                 preparation: SpeechPreparation,
                 project: Callable[[CompletedSpeechSegment], Awaitable[None]],
                 operation_timeout: float = 30, cleanup_timeout: float = 20) -> None:
        if (type(operation_timeout) not in (int, float) or not 0 < operation_timeout <= 60
                or type(cleanup_timeout) not in (int, float) or not 0 < cleanup_timeout <= 30):
            raise ValueError("delivery operations and cleanup must be bounded")
        self.gate, self.publisher, self.journal = gate, publisher, journal
        self.language, self.attempt_id = language, attempt_id
        self.preparation, self.project = preparation, project
        self._timeout, self._cleanup_timeout = operation_timeout, cleanup_timeout
        self._task: asyncio.Task[str] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._children: set[asyncio.Task] = set()
        self._started = False
        self._exposure_requested = False
        self._owner: Any = None
        self._final_requested: str | None = None
        self._final_acknowledged = False
        self._final_status: str | None = None

    async def _operation(self, operation: Awaitable, *, timeout: float | None = None,
                         invalidate_on_timeout: bool = True):
        child = asyncio.create_task(operation)
        self._children.add(child)
        try:
            done, _ = await asyncio.wait((child,), timeout=self._timeout if timeout is None else timeout)
            if not done:
                if invalidate_on_timeout:
                    self.gate.invalidate()
                child.cancel()
                raise TimeoutError("workflow delivery operation did not acknowledge")
            return child.result()
        finally:
            if child.done():
                self._children.discard(child)

    async def run(self) -> str:
        if self._started:
            raise SpeechRefused("segment owner cannot be run twice")
        self._started = True
        self._task = asyncio.create_task(self._drive())
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            self.gate.invalidate()
            if not self._task.done() and not self._task.cancelling():
                self._task.cancel()
            await self._join_owner()
            raise

    async def _final(self, status: str) -> str:
        self._final_requested = status
        # Retry the identical final fact after a lost reply. A start grant and
        # a source acquisition are never retried for new permission.
        for attempt in range(2):
            try:
                fact = await self._operation(self.journal.finish(self._owner, status), timeout=5,
                                             invalidate_on_timeout=False)
                permitted = {status, "uncertain"} if status in ("completed_local", "interrupted") else {status}
                if status == "uncertain":
                    permitted = {"completed_local", "interrupted", "uncertain", "aborted_unexposed"}
                if fact not in permitted:
                    raise SpeechRefused("journal did not acknowledge the requested terminal fact")
                self._final_acknowledged = True
                self._final_status = fact
                return fact
            except (TimeoutError, ConnectionError):
                if attempt:
                    raise

    async def _drive(self) -> str:
        try:
            binding = self.gate.open_segment(attempt_id=self.attempt_id)
            self._owner = await self._operation(self.journal.claim(binding, self.gate, self.language), timeout=5)
            if self._owner is None:
                raise SpeechRefused("journal did not grant fresh generation ownership")
            approved = await self.preparation.prepare(self.gate, binding, self._owner, operation=self._operation)
            self._exposure_requested = True
            grant = await self._operation(self.journal.start(self._owner), timeout=5)
            if grant is not True:
                raise SpeechRefused("journal did not grant a fresh exposure start")
            # The playback owner owns its duration-aware timeout and cleanup;
            # a shorter generic operation timer would cancel valid long PCM.
            result = await self._operation(self.publisher.play(approved),
                timeout=self.publisher.playout_timeout(approved) + self._cleanup_timeout)
            if result.status not in ("completed_local", "interrupted"):
                raise SpeechRefused("publisher returned an unknown lifecycle fact")
            final_status = await self._final(result.status)
            await self._settle_children()
            if final_status == "completed_local":
                self.preparation.completed(self.gate, approved)
                # This callback owns authored captions/continuation. Never use
                # a provider caption, retry a partial projection or call it
                # before the durable terminal acknowledgement.
                candidate = approved.candidate
                completed = CompletedSpeechSegment(candidate.binding, candidate.text, candidate.pcm_digest,
                    candidate.sample_rate, candidate.samples, result.speech_id, result.scheduled_at,
                    result.samples_submitted, "authored_tts")
                await self._operation(self.project(completed), timeout=5)
            else:
                self.gate.invalidate()
            return final_status
        except BaseException:
            self.gate.invalidate()
            # Run cleanup under a fresh task: the cancelled owner retains a
            # cancelling count, which public interruption correctly propagates.
            # Cleanup must still reach and join the source children.
            self._cleanup_task = asyncio.create_task(self._cleanup())
            await asyncio.shield(self._cleanup_task)
            raise

    async def _cleanup(self) -> None:
        errors: list[BaseException] = []
        for child in self._children:
            if not child.done() and not child.cancelling():
                child.cancel()
        try:
            # A failed SDK barrier must not prevent the durable uncertainty
            # write. It still makes this owner unsafe to replace.
            await self.publisher.interrupt()
        except BaseException as error:
            errors.append(error)
        if self._exposure_requested:
            try:
                # Fence before joining an unknown start command: if it has
                # not committed, the atomic abort rejects its later start.
                await self._reconcile_final()
            except BaseException as error:
                errors.append(error)
        joined = False
        try:
            await self._settle_children()
            joined = True
        except BaseException as error:
            errors.append(error)
        if not self._exposure_requested and joined:
            try:
                # No new source owner until the old producer/verifier joins.
                await self._reconcile_final()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    async def _reconcile_final(self) -> None:
        if self._owner is None or self._final_acknowledged:
            return
        desired = self._final_requested or ("uncertain" if self._exposure_requested else "aborted_unexposed")
        try:
            await self._final(desired)
        except (Exception, asyncio.CancelledError):
            if desired == "uncertain":
                raise
            # Reconciliation returns an existing immutable final, maps an
            # expired exposure to uncertain, or atomically retires unexposed.
            await self._final("uncertain")

    async def _settle_children(self) -> None:
        current = asyncio.current_task()
        pending = tuple(c for c in self._children if c is not current)
        for child in pending:
            if not child.done() and not child.cancelling():
                child.cancel()
        if pending:
            done, still_running = await asyncio.wait(pending, timeout=self._cleanup_timeout)
            for child in done:
                self._children.discard(child)
                try:
                    child.result()
                except (asyncio.CancelledError, Exception):
                    pass
            if still_running:
                raise SpeechRefused("delivery child remains owned; handoff refused")

    async def _join_owner(self) -> None:
        # A separate cleanup waiter prevents repeated cancellation from
        # cancelling the owner while it is joining the real source/SDK writer.
        async def join():
            assert self._task is not None
            done, _ = await asyncio.wait((self._task,), timeout=self._cleanup_timeout+40)
            if not done:
                raise SpeechRefused("segment owner has not settled; handoff refused")
            try:
                self._task.result()
            except asyncio.CancelledError:
                pass
        waiter = asyncio.create_task(join())
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                continue
        waiter.result()

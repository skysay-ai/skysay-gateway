# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One bounded original-PCM ring and independently owned source cursors."""
from __future__ import annotations

import asyncio

try:
    from .capabilities.workflow_listener import MAX_CALLER_SOURCE_SECONDS
    from .workflow_input import WorkflowInputRefused, WorkflowInputCapacityExceeded
    from .workflow_source_owner import WorkflowListenerRefusal
    from .workflow_tasks import join_owned_task
except ImportError:
    from capabilities.workflow_listener import MAX_CALLER_SOURCE_SECONDS
    from workflow_input import WorkflowInputRefused, WorkflowInputCapacityExceeded
    from workflow_source_owner import WorkflowListenerRefusal
    from workflow_tasks import join_owned_task


#: A reconnect's bound, from the later of its claim and the caller's endpoint:
#: the fresh listener exists by ``claim + RECONNECT_SECONDS`` and has delivered
#: the turn's final words by ``max(claim, endpoint) + RECONNECT_SECONDS``.
RECONNECT_SECONDS = 3


class CallerPcmRing:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.first = self.end = 0
        self._pcm = bytearray()
        self._maximum = sample_rate * 26
        # The first sample a reconnect has not yet resent: never evicted while
        # set, up to twice the window, so a resend cannot lose the turn.
        self.hold = None

    def append(self, data):
        self._pcm.extend(data)
        self.end += len(data) // 2
        floor = self.end - self._maximum
        if self.hold is not None:
            floor = max(min(floor, self.hold), self.end - 2 * self._maximum)
        excess = floor - self.first
        if excess > 0:
            del self._pcm[:2 * excess]
            self.first += excess

    def read(self, first, end):
        if not self.first <= first < end <= self.end:
            raise WorkflowInputCapacityExceeded("workflow source cursor fell outside its retained PCM window")
        return bytes(self._pcm[2 * (first - self.first):2 * (end - self.first)])


class ListenerReconnect:
    """The call's one reconnect of a listener whose provider stream was never acknowledged.

    Only a ``no_ack`` refusal (never connected, or the stream closed, errored or
    timed out without its terminal acknowledgement) is reconnected: the other
    reasons are setup errors or limits a second connection cannot fix.
    ``available`` refuses a claim the call's source budget cannot pay for.
    """

    def __init__(self, *, report, available=None):
        self._report, self._available = report, available
        self.claimed = False

    def claim(self, error):
        if (self.claimed or not isinstance(error, WorkflowListenerRefusal) or error.reason != "no_ack"
                or (self._available is not None and not self._available())):
            return False
        self.claimed = True
        return True

    def report(self, **fields):
        try:
            self._report(**fields)
        except Exception:  # noqa: BLE001 - evidence cannot change the turn it records
            pass


class _Retry:
    """One claimed reconnect: its deadline and its single reported outcome.

    The deadline is a predicate every transition checks first; the timer only
    wakes an await that would otherwise outlive it. Expiry settles ``failed``
    and hands the original refusal to ``on_expired``, which latches the call's
    ending synchronously, before any cleanup is joined.
    """

    def __init__(self, *, original, reconnect, on_expired, replayed_seconds):
        self.loop = asyncio.get_running_loop()
        self.original, self.reconnect, self.on_expired = original, reconnect, on_expired
        self.replayed_seconds = replayed_seconds
        self.claimed_at = self.loop.time()
        self.deadline = self._handle = None
        self.state = "establishing"
        self._settled = False

    def arm(self, when):
        self.unarm()
        self.deadline = when
        self._handle = self.loop.call_at(when, self._expire)

    def unarm(self):
        if self._handle is not None:
            self._handle.cancel()
        self.deadline = self._handle = None

    def _settle(self, outcome):
        if not self._settled:
            self._settled = True
            self.reconnect.report(reason=self.original.reason, outcome=outcome,
                                  replayed_seconds=self.replayed_seconds)

    def _expire(self):
        self._handle = None
        if self.state == "done":
            return
        self.state = "done"
        self._settle("failed")
        if self.on_expired is not None:
            try:
                self.on_expired(self.original)
            except Exception:  # noqa: BLE001 - the refusal below still ends the turn
                pass

    def _current(self):
        if self.state != "done" and self.deadline is not None and self.loop.time() >= self.deadline:
            self._expire()
        if self.state == "done":
            raise self.original

    def established(self, sealed_at):
        self._current()
        self.unarm()
        self.state = "following"
        if sealed_at is not None:
            self.sealed(sealed_at)

    def sealed(self, sealed_at):
        if self.state == "following":
            self.state = "finishing"
            self.arm(max(self.claimed_at, sealed_at) + RECONNECT_SECONDS)

    def recovered(self):
        self._current()
        self.unarm()
        self.state = "done"
        self._settle("recovered")

    def close(self):
        self.unarm()
        self.state = "done"
        self._settle("failed")


class LogicalCallerSource:
    """No PCM leaves until its factory has obtained fresh provider permission."""
    def __init__(self, *, ring, first_sample, source_factory, frame_factory, reconnect=None, on_expired=None):
        self.ring, self.first, self.cursor = ring, first_sample, first_sample
        self._desired = first_sample
        self._factory, self._frame = source_factory, frame_factory
        self._reconnect, self._on_expired = reconnect, on_expired
        self._changed = asyncio.Event()
        self._sealed = self._aborted = False
        self._sealed_at = self._retry = None
        self._holding = False
        # The turn's words, set once the provider closed (before its source
        # facts are written); the task still owns those writes to the end.
        self._released = asyncio.get_running_loop().create_future()
        self.task = asyncio.create_task(self._run())

    def _release(self, result):
        if not self._released.done():
            self._released.set_result(result)

    async def outcome(self):
        """The turn's words as soon as they are released, else the task's own result.

        Waiting never cancels the task: its source facts are still written by
        it, and its failure still reaches whoever watches ``task``.
        """
        await asyncio.wait((self._released, self.task), return_when=asyncio.FIRST_COMPLETED)
        if self._released.done():
            return self._released.result()
        return self.task.result()

    def advance(self, end):
        if self._sealed or self._aborted or end < self._desired:
            raise WorkflowInputRefused("workflow source changed its processed audio ownership")
        self._desired = end
        self._changed.set()

    def seal(self, end):
        self.advance(end)
        self._sealed = True
        self._sealed_at = asyncio.get_running_loop().time()
        if self._retry is not None:
            self._retry.sealed(self._sealed_at)
        self._changed.set()

    async def _feed(self, source):
        while True:
            self._changed.clear()
            maximum = self.first + MAX_CALLER_SOURCE_SECONDS * self.ring.sample_rate
            target = min(self._desired, maximum)
            exceeded = self._desired > maximum
            while self.cursor < target:
                end = min(target, self.cursor + self.ring.sample_rate // 50)
                data = self.ring.read(self.cursor, end)
                source.push_frame(self._frame(data=data, sample_rate=self.ring.sample_rate,
                    num_channels=1, samples_per_channel=end - self.cursor))
                self.cursor = end
                if self._holding:
                    self.ring.hold = end
                # A delayed grant may release a burst. Let the detector and
                # media producer advance without buffering another PCM copy.
                await asyncio.sleep(0)
            if exceeded:
                await source.finish()
                raise WorkflowInputCapacityExceeded("workflow caller exceeded the logical source duration capacity")
            if self._sealed and self.cursor == self._desired:
                # A reconnected turn keeps the strict order: its words reach
                # the workflow only after its replacement source is accounted
                # and the reconnect is recorded as recovered.
                return await source.finish(release=self._release if self._retry is None else None)
            await self._changed.wait()

    async def _run(self):
        source = None
        # Once a reconnect has failed or been cancelled, closing the fresh
        # source cannot replace that ending with its own cleanup failure.
        decided = False
        try:
            source = await self._factory(first_sample=self.first, sample_rate=self.ring.sample_rate)
            try:
                return await self._feed(source)
            except WorkflowListenerRefusal as error:
                # Only a turn whose every sample is still retained is resent.
                # A turn whose words were already released is never replayed.
                if (self._aborted or self._released.done() or self._reconnect is None
                        or self.ring.first > self.first or not self._reconnect.claim(error)):
                    raise
                original = error
            # From the claim to the first await nothing can skip the outcome.
            self._retry = retry = _Retry(original=original, reconnect=self._reconnect, on_expired=self._on_expired,
                                         replayed_seconds=(self.cursor - self.first) / self.ring.sample_rate)
            try:
                self.ring.hold, self._holding = self.first, True
                retry.arm(retry.claimed_at + RECONNECT_SECONDS)
                failed, source = source, None
                await join_owned_task(asyncio.create_task(failed.aclose()))
                source = await self._factory(first_sample=self.first, sample_rate=self.ring.sample_rate)
                retry.established(self._sealed_at)
                self.cursor = self.first
                result = await self._feed(source)
                retry.recovered()
                return result
            except asyncio.CancelledError:
                decided = True
                raise
            except Exception as failure:
                decided = True
                if asyncio.current_task().cancelling():
                    # The call is ending underneath: a cleanup that failed on
                    # the way out is not this listener's refusal.
                    raise asyncio.CancelledError from failure
                if isinstance(failure, (WorkflowListenerRefusal, WorkflowInputCapacityExceeded)):
                    # The original refusal or the fresh stream's own truthful
                    # reason; the turn's duration cap is no reconnect outcome.
                    raise
                # Anything else that stopped the reconnect is still this
                # listener being unavailable: the call ends as it would have.
                raise original from failure
            finally:
                if self._holding:
                    self.ring.hold, self._holding = None, False
                retry.close()
        finally:
            if source is not None:
                try:
                    await join_owned_task(asyncio.create_task(source.aclose()))
                except Exception:
                    if not decided:
                        raise

    async def aclose(self):
        if not self._aborted:
            self._aborted = True
            self.task.cancel()
        try:
            await join_owned_task(self.task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise

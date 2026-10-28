# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""A logical caller turn owns a fresh finite source and its final words."""
from __future__ import annotations

import asyncio
import dataclasses
import math
import time

try:
    from .capabilities.workflow_listener import validate_frozen_voice_input, STT_FINALIZATION_TIMEOUT_SECONDS
    from .capabilities.workflow_sources import qualified_source_adapter
    from .workflow_input import CallerTurnTiming
    from .workflow_input_recognition import CallerTextCollector
    from .workflow_source_owner import WorkflowSourceOwner
    from .workflow_speech import SpeechRefused
except ImportError:
    from capabilities.workflow_listener import validate_frozen_voice_input, STT_FINALIZATION_TIMEOUT_SECONDS
    from capabilities.workflow_sources import qualified_source_adapter
    from workflow_input import CallerTurnTiming
    from workflow_input_recognition import CallerTextCollector
    from workflow_source_owner import WorkflowSourceOwner
    from workflow_speech import SpeechRefused


class WorkflowListenerSource(WorkflowSourceOwner):
    @classmethod
    async def create(cls, *, journal, policy, managed=None, attest=None):
        policy = validate_frozen_voice_input(policy, required_languages=policy["required_languages"])
        constructor = policy["constructor"]
        declaration = qualified_source_adapter("listener", constructor["provider"], constructor["model"])
        source = cls(journal=journal, declaration=declaration, constructor=constructor, managed=managed)
        try:
            await source._open()
            if attest is not None:
                # The listener navigation hears through, as constructed for this turn (never a secret).
                try:
                    attest({"provider": constructor["provider"], "model": constructor["model"],
                            "languages": list(constructor.get("languages") or ()),
                            "settings": dict(getattr(source, "effective_settings", {}) or {})})
                except Exception:  # noqa: BLE001 - an attestation never moves a call
                    pass
            source._collector = CallerTextCollector()
            async def collect():
                async for final in source.adapter.transcripts():
                    source._collector.append(final)
            source.reader = asyncio.create_task(collect())
            return source
        except BaseException:
            await source.aclose()
            raise

    def push_frame(self, frame):
        if self._ended or self._closing is not None or self.adapter is None:
            raise SpeechRefused("workflow listener input has already ended")
        if self.reader.done():
            # An early channel end is not the finite adapter's terminal ACK.
            if not self.reader.cancelled() and self.reader.exception() is not None:
                raise self.reader.exception()
            raise SpeechRefused("workflow listener closed before its owned input")
        self.adapter.push_frame(frame)

    async def finish(self, release=None):
        """End input, then return the caller's words once the source is fully accounted.

        ``release`` hears the same words earlier: as soon as the provider
        acknowledged its terminal and closed cleanly, before the source's
        delta/close/terminal facts are written. Those writes still finish
        inside this call, exactly once; a turn never waits on them. A provider
        that refused, timed out or failed to close releases nothing.
        """
        if self._ended or self._closing is not None:
            raise SpeechRefused("workflow listener cannot send a second input EOF")
        self._ended = True
        input_ended_at, closed_at = time.monotonic(), None
        self.adapter.end_input()
        try:
            async with asyncio.timeout(STT_FINALIZATION_TIMEOUT_SECONDS):
                await self.reader
                if await self.adapter.wait_terminal() is not True:
                    raise SpeechRefused("workflow listener has no finite terminal acknowledgement")
            if release is not None and await self.close_provider():
                closed_at = time.monotonic()
                release(self._timed(input_ended_at, closed_at))
        finally:
            await self.aclose()
        # Without an early release the words wait for the source's own facts;
        # that later instant is not the provider's close, so it stays unknown.
        return self._timed(input_ended_at, closed_at)

    def _timed(self, input_ended_at, closed_at):
        """The turn's words with the listener's stage marks; evidence never fails a turn."""
        result = self._collector.result()
        if result is None:
            return None
        try:
            final_at = getattr(self.adapter, "provider_final_at", None)
            final_at = final_at() if callable(final_at) else None
            if type(final_at) not in (int, float) or not math.isfinite(final_at):
                final_at = None
            timing = CallerTurnTiming(input_ended_at=input_ended_at, listener_closed_at=closed_at,
                                      provider_final_at=None if final_at is None else float(final_at))
            return dataclasses.replace(result, timing=timing)
        except Exception:  # noqa: BLE001 - a diagnostic cannot cost the caller's words
            return result

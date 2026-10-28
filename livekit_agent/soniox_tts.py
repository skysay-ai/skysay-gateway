# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Pinned Soniox TTS adapter with opaque per-call usage correlation.

LiveKit's Soniox plugin is deliberately pinned at 1.7.1 because it owns the
streaming details we must preserve: sentence buffering, the short idle stream
timeout, stream cancellation, and rotation before the provider's two-minute
stream ceiling.  That plugin exposes ``client_reference_id`` for STT but not
TTS.  Soniox usage logs are the only exact native TTS-token evidence, so this
small subclass adds the one missing start-frame field without replacing the
plugin's streaming implementation.

This module intentionally reaches two private plugin internals.  The image
test pins the exact plugin release and asserts these seams, so an upstream
upgrade cannot quietly drop usage correlation or alter cancellation behavior.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from livekit.agents import APIConnectionError
from livekit.plugins import soniox
from livekit.plugins.soniox import tts as _soniox_tts


# Cleanup happens after the original connect/send failure has been surfaced, so
# it must never be allowed to create an unbounded retry side channel.  A stuck
# socket occupies one of these slots until its close coroutine honours the
# cancellation request; subsequent stream admission then fails closed instead
# of constructing more sockets.
_CLOSE_TIMEOUT_SECONDS = 5.0
_MAX_PENDING_CLOSE_TASKS = 8


class _ConnectionWithReference(_soniox_tts._Connection):  # type: ignore[name-defined]
    """The upstream persistent connection plus one provider config field."""

    def __init__(
        self,
        *args: Any,
        client_reference_id: str,
        close_after_failure: Callable[["_ConnectionWithReference"], Awaitable[None]],
        on_request_started: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._skysay_client_reference_id = client_reference_id
        self._skysay_close_after_failure = close_after_failure
        self._skysay_on_request_started = on_request_started

    async def _send_loop(self) -> None:
        """Copy 1.7.1's send loop, modifying only its start configuration.

        The message classes and all non-start branches remain the upstream
        ones.  In particular, a cancellation still sends the plugin's exact
        ``{stream_id, cancel: true}`` frame and the connection remains usable
        for subsequent sentence streams.
        """
        try:
            async for message in self._input_queue:
                if self._ws is None or self._ws.closed:
                    # A closed socket is just as terminal as a send exception:
                    # every queued stream must wake up with a retryable
                    # adapter failure instead of waiting forever for audio
                    # from a connection this loop is about to discard.
                    self._fail_all(APIConnectionError("Soniox TTS socket closed"))
                    break
                if isinstance(message, _soniox_tts._StartConfig):
                    config: dict[str, Any] = {
                        "api_key": message.opts.api_key,
                        "model": message.opts.model,
                        "language": message.opts.language,
                        "voice": message.opts.voice,
                        "audio_format": message.opts.audio_format,
                        "sample_rate": message.opts.sample_rate,
                        "speed": message.opts.speed,
                        "stream_id": message.stream_id,
                        "client_reference_id": self._skysay_client_reference_id,
                    }
                    if message.opts.bitrate is not None:
                        config["bitrate"] = message.opts.bitrate
                    await self._ws.send_str(json.dumps(config))
                elif isinstance(message, _soniox_tts._SendText):
                    payload: dict[str, Any] = {"stream_id": message.stream_id}
                    if message.text:
                        if self._skysay_on_request_started is not None:
                            self._skysay_on_request_started()
                        payload["text"] = message.text
                    if message.text_end:
                        payload["text_end"] = True
                    await self._ws.send_str(json.dumps(payload))
                elif isinstance(message, _soniox_tts._CancelStream):
                    await self._ws.send_str(
                        json.dumps({"stream_id": message.stream_id, "cancel": True})
                    )
        except Exception as exc:
            _soniox_tts.logger.warning("Soniox TTS send loop error", exc_info=exc)
            self._fail_all(APIConnectionError("Soniox TTS send loop error"))
        finally:
            if not self._closed:
                # A failed send loop is never reusable. Mark it non-current
                # before cleanup so a saturated cleanup queue cannot make a
                # later stream pick up this dead websocket.
                self.mark_non_current()
                # The parent TTS owns every fatal-close task, including this
                # pooled connection.  Do not create an unreferenced task here:
                # a long-lived worker must be able to bound and fail closed on
                # wedged provider socket cleanup.
                await self._skysay_close_after_failure(self)


class TTS(soniox.TTS):
    """Soniox 1.7.1 TTS with a safe opaque usage-log reference."""

    def __init__(
        self, *, client_reference_id: str,
        on_request_started: Callable[[], None] | None = None, **kwargs: Any,
    ) -> None:
        reference = str(client_reference_id or "").strip()
        if not reference:
            raise ValueError("Soniox TTS client_reference_id is required")
        self._skysay_client_reference_id = reference
        self._skysay_on_request_started = on_request_started
        # A connection that fails before becoming current, or whose send loop
        # has failed, has no successful stream owner. Retain every cleanup task
        # until completion so it cannot be garbage-collected while closing.
        # The set has a small fixed capacity; saturation blocks a new provider
        # socket before one is constructed.
        self._skysay_unowned_close_tasks: set[asyncio.Task[Any]] = set()
        super().__init__(**kwargs)

    async def _close_connection_after_failure(self, connection: Any) -> None:
        """Bound a fatal socket close without holding stream admission open."""

        if len(self._skysay_unowned_close_tasks) >= _MAX_PENDING_CLOSE_TASKS:
            # ``_current_connection`` normally checks capacity before
            # allocating. A send-loop or connect-failure race can fill the
            # slots after that check, however. Never drop this socket: attempt
            # its bounded close inline rather than starting a ninth task. This
            # path is fail-closed (and may delay the already-failed caller),
            # but it cannot grow the background task set or leave a current
            # dead websocket reusable.
            try:
                await asyncio.wait_for(connection.aclose(), timeout=_CLOSE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                _soniox_tts.logger.error("Soniox TTS saturated cleanup timed out")
            except Exception:
                _soniox_tts.logger.error("Soniox TTS saturated cleanup failed")
            return

        task = asyncio.create_task(
            connection.aclose(), name="soniox-tts-fatal-close"
        )
        self._skysay_unowned_close_tasks.add(task)

        def cancel_if_stuck() -> None:
            if not task.done():
                _soniox_tts.logger.warning("Soniox TTS socket cleanup timed out")
                task.cancel()

        timeout_handle = asyncio.get_running_loop().call_later(
            _CLOSE_TIMEOUT_SECONDS, cancel_if_stuck
        )

        def reap(done: asyncio.Task[Any]) -> None:
            timeout_handle.cancel()
            self._skysay_unowned_close_tasks.discard(done)
            if done.cancelled():
                return
            try:
                done.exception()
            except BaseException:
                # The original connection/cancellation failure is the result
                # visible to the caller; cleanup failure is deliberately not
                # a success path and cannot retain this task forever.
                pass

        task.add_done_callback(reap)

    async def _current_connection(self, *, timeout: float) -> tuple[Any, float, bool]:
        """Use upstream connection lifecycle with our start-frame wrapper."""
        # These are the exact name-mangled fields initialized by Soniox TTS
        # 1.7.1.  Keeping the rest of this method verbatim preserves pooling,
        # drain/close, cancellation and stream-rotation behavior.
        failed_connection: _ConnectionWithReference | None = None
        try:
            async with self._TTS__conn_lock:  # type: ignore[attr-defined]
                connection = self._TTS__current_connection  # type: ignore[attr-defined]
                if connection is not None and connection.is_current and not connection.closed:
                    return connection, 0.0, True
                if connection is not None and not connection.closed:
                    connection.mark_non_current()
                if len(self._skysay_unowned_close_tasks) >= _MAX_PENDING_CLOSE_TASKS:
                    raise APIConnectionError("Soniox TTS cleanup capacity is saturated")
                started = time.perf_counter()
                new_connection = _ConnectionWithReference(
                    self._opts,
                    self._ensure_session(),
                    client_reference_id=self._skysay_client_reference_id,
                    close_after_failure=self._close_connection_after_failure,
                    on_request_started=self._skysay_on_request_started,
                )
                try:
                    await asyncio.wait_for(new_connection.connect(), timeout=timeout)
                except BaseException:
                    # The connection has not become the pool's current
                    # instance, so no successful stream lifecycle owns it.
                    # Record it while the pool is protected, then release the
                    # lock before closing it. The saturated fallback performs
                    # a bounded inline close to avoid leaking the ninth socket;
                    # holding that I/O under the pool lock would serialize all
                    # later stream admissions behind a dead connection.
                    failed_connection = new_connection
                    raise
                self._TTS__current_connection = new_connection  # type: ignore[attr-defined]
                return new_connection, time.perf_counter() - started, False
        except BaseException:
            if failed_connection is not None:
                await self._close_connection_after_failure(failed_connection)
            raise


__all__ = ["TTS"]

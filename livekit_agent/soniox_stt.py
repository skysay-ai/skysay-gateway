# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Pinned Soniox STT whose streams the worker closes itself.

Soniox writes a session's usage row only after that WebSocket session closes,
and the SDK closes a recognizer stream only inside ``AgentSession.aclose()``,
which on a PSTN hang-up regularly does not finish within its bound. The stream
then kept its socket alive until process exit: its usage row could not exist
when the worker read it, so the call was never billed, and the job's shared
HTTP session closing under it made the plugin reconnect onto a closed session
(``RuntimeError: Session is closed``, GlitchTip #85).

So the worker owns the close. These are subclasses of the pinned plugin
classes, not a wrapper: SDK streams emit their metrics and errors on the STT
object that created them, and the language switcher and usage collector
listen there. The STT registers every stream it creates and refuses new ones
once teardown has begun; the stream takes its socket the moment ``ws_connect``
returns, before the configuration send, and counts each socket that completed
initialisation, which is how many provider sessions (and usage rows) exist.

This reaches private seams of livekit-plugins-soniox 1.8.8 (``_ensure_session``,
``_connect_ws``, ``_run``) and of the base stream (``_task``). The SDK proofs in
``tests/test_navigation_engines_sdk_proofs.py`` run against that pinned
release in the worker image, so an upgrade that moves them fails there.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins import soniox
from livekit.plugins.soniox import stt as _soniox_stt

try:  # package import in tests; flat /app import in the worker image
    from .soniox_usage import SonioxSTTCloseOutcome
except ImportError:  # pragma: no cover - exercised by container smoke
    from soniox_usage import SonioxSTTCloseOutcome


class _RegisteringSession:
    """The job's shared session, except that a new socket is registered on return.

    Nothing is awaited between ``ws_connect`` returning and the registration,
    so a cancellation during the configuration send cannot leave the socket
    outside the terminal close.
    """

    __slots__ = ("_session", "_register")

    def __init__(self, session: Any, register: Callable[[Any], None]) -> None:
        self._session = session
        self._register = register

    async def ws_connect(self, *args: Any, **kwargs: Any) -> Any:
        ws = await self._session.ws_connect(*args, **kwargs)
        self._register(ws)
        return ws


class SpeechStream(_soniox_stt.SpeechStream):
    """The plugin stream, with its sockets owned from acquisition."""

    def __init__(
        self,
        *,
        stt: "STT",
        conn_options: APIConnectOptions,
        refused: bool,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        # Set before the base class's task first runs (the next loop turn).
        self._skysay_owner = stt
        self._skysay_refused = refused
        self._skysay_sockets: list[Any] = []

    def _ensure_session(self) -> Any:
        # The plugin caches the REAL session on the STT; only the returned
        # handle registers, so nothing else ever sees the proxy.
        return _RegisteringSession(super()._ensure_session(), self._skysay_sockets.append)

    async def _connect_ws(self) -> Any:
        ws = await super()._connect_ws()
        # Configuration sent: Soniox now has a session that will write a row.
        self._skysay_owner._skysay_sessions += 1
        return ws

    def push_frame(self, frame: Any) -> None:
        # The SDK's ``stt_node`` keeps forwarding caller audio until it sees
        # the stream end; a frame landing after the worker's close would raise
        # "input ended" there, which the SDK logs as an error. It is dropped.
        if self._skysay_owner._skysay_closing:
            return
        super().push_frame(frame)

    def flush(self) -> None:
        if self._skysay_owner._skysay_closing:
            return
        super().flush()

    async def _run(self) -> None:
        if self._skysay_refused:
            # Ending without an error is what stops the SDK's STT pump; an
            # exception would either be retried or reported.
            return
        await super()._run()

    def _skysay_closed(self) -> bool:
        return self._task.done() and all(ws.closed for ws in self._skysay_sockets)


class STT(soniox.STT):
    """The plugin STT, with ``stream()`` registering what it creates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The provider identity the SDK records stays the plugin's own.
        self._label = f"{soniox.STT.__module__}.{soniox.STT.__name__}"
        self._skysay_streams: list[SpeechStream] = []
        self._skysay_closing = False
        self._skysay_sessions = 0

    @property
    def completed_sessions(self) -> int:
        return self._skysay_sessions

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        del language
        stream = SpeechStream(stt=self, conn_options=conn_options, refused=self._skysay_closing)
        if not self._skysay_closing:
            self._skysay_streams.append(stream)
        return stream

    async def _skysay_close_all(self, streams: list[SpeechStream]) -> bool:
        for stream in streams:
            # Cancels ``_run``: its receive task does not request a reconnect
            # on cancellation, and its ``finally`` closes the current socket.
            await stream.aclose()
        for stream in streams:
            for ws in stream._skysay_sockets:
                if not ws.closed:
                    # A socket ``_run`` never owned (cancelled mid-handshake).
                    await ws.close()
        # ``cancel_and_wait`` does not propagate the task's exception, so
        # finishing the loop above is not proof: ask every stream.
        return all(stream._skysay_closed() for stream in streams)

    async def skysay_close_streams(self, *, timeout: float) -> SonioxSTTCloseOutcome:
        """Close every stream this STT created, within ``timeout``.

        Never raises. A pending close is left running past the bound (its
        result consumed), and an external cancellation of the caller is
        returned in ``cancellation`` for it to re-raise after settlement.
        """
        self._skysay_closing = True
        streams = list(self._skysay_streams)
        closer = asyncio.ensure_future(self._skysay_close_all(streams))

        def outcome(status: str, cancellation: BaseException | None = None) -> SonioxSTTCloseOutcome:
            return SonioxSTTCloseOutcome(
                status=status,
                sessions=self._skysay_sessions,
                streams=len(streams),
                cancellation=cancellation,
            )

        try:
            done, _ = await asyncio.wait({closer}, timeout=timeout)
        except asyncio.CancelledError as exc:
            closer.add_done_callback(_consume)
            return outcome("cancelled", exc)
        if not done:
            closer.add_done_callback(_consume)
            return outcome("timeout")
        if closer.cancelled() or closer.exception() is not None:
            return outcome("error")
        return outcome("closed" if closer.result() else "error")


def _consume(task: "asyncio.Future[Any]") -> None:
    if not task.cancelled():
        task.exception()


__all__ = ["STT", "SpeechStream"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The pinned LiveKit ElevenLabs TTS, metered at its provider wire.

ElevenLabs bills the characters of the text it is handed, and it generates (and
bills) everything a context was handed even when the worker closes that context
early -- a barge-in. So every request is metered
by the one native unit meter (``native_tts_usage``): its billing rule is the
provider's data in the capability artifact, and this module only reports
facts at the wire.

The plugin multiplexes every synthesis of a TTS instance over ONE pooled
WebSocket and sends from the connection's own task. That task inherited
whichever request happened to open the connection, so a frame is attributed by
the ``context_id`` it carries -- bound to the request when the plugin names the
context inside the attempt -- and never by the sending task's context.

Nothing process-global is patched: ``stream()``, ``synthesize()`` and
``_ensure_session()`` hand the plugin metered classes and a metered session.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Mapping
from types import ModuleType
from typing import Any

try:  # package import in the API image/tests
    from .native_tts_usage import (
        MeteredHTTPSession,
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        current_tts_request,
        hand_off_context,
        run_metered_request,
    )
except ImportError:  # pragma: no cover - direct voice-worker entrypoint
    from native_tts_usage import (
        MeteredHTTPSession,
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        current_tts_request,
        hand_off_context,
        run_metered_request,
    )


_METERED_CLASS_MARKER = "_skysay_elevenlabs_metered_tts"
_PATCH_LOCK = threading.Lock()
#: The aiohttp message types on which the pinned plugin raises "connection
#: closed unexpectedly" for every context still registered.
_SOCKET_ENDINGS = frozenset({"CLOSE", "CLOSING", "CLOSED"})


class ElevenLabsRuntimeCompatibilityError(RuntimeError):
    """The pinned plugin no longer exposes the seams this meter relies on."""


def text_units(packet: Any, *, on_socket: bool) -> int:
    """Characters ElevenLabs bills for the text one frame or request body carries.

    Units are code points. What ElevenLabs does with surrounding whitespace
    differs by protocol, measured against its credit counter (2026-10-20):

    - the text-to-speech WebSocket (``text``) bills each generation with its
      surrounding whitespace trimmed: the context's initial ``" "``, each
      sentence's trailing space and leading spaces in a frame were not billed.
      The pinned plugin flushes every sentence frame on its own
      (``auto_mode``), so one frame is one generation and is trimmed alone;
    - the text-to-dialogue WebSocket (``inputs[].text``) bills the text whole:
      a trailing and a leading space were each billed;
    - an HTTP body (either endpoint) is billed whole, whitespace included (its
      ``character-cost`` header rose with leading and trailing spaces).
    """
    if not isinstance(packet, Mapping):
        return 0
    units = 0
    text = packet.get("text")
    if isinstance(text, str):
        units += len(text.strip() if on_socket else text)
    inputs = packet.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            value = item.get("text") if isinstance(item, Mapping) else None
            if isinstance(value, str):
                units += len(value)
    return units


def _message_context(data: Mapping[str, Any]) -> Any:
    # The plugin's own reading: snake_case today, camelCase on older responses.
    return data.get("contextId") or data.get("context_id")


class _MeteredSocket:
    """One pooled ElevenLabs socket that hands each frame's text to the meter."""

    def __init__(self, socket: Any, meter: NativeTTSUnitMeter) -> None:
        if getattr(socket, "compress", 0):
            # An uncompressed frame reaches the transport before the send's
            # first await, so a send the worker cancels was already handed.
            raise ElevenLabsRuntimeCompatibilityError("ElevenLabs socket must be uncompressed")
        self._socket = socket
        self._meter = meter
        #: Contexts this socket carried whose final message has not arrived.
        self._open: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._socket, name)

    def _carry(self, packet: Any) -> Any:
        context = packet.get("context_id") if isinstance(packet, Mapping) else None
        if isinstance(context, str) and context:
            self._open.add(context)
        return context

    async def send_json(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        context = self._carry(data)
        return await hand_off_context(
            self._meter, context, text_units(data, on_socket=True),
            lambda: self._socket.send_json(data, *args, **kwargs),
        )

    async def send_str(self, data: str, *args: Any, **kwargs: Any) -> Any:
        try:
            packet = json.loads(data)
        except (TypeError, ValueError):
            packet = None
        context = self._carry(packet)
        return await hand_off_context(
            self._meter, context, text_units(packet, on_socket=True),
            lambda: self._socket.send_str(data, *args, **kwargs),
        )

    async def send_bytes(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ElevenLabsRuntimeCompatibilityError("ElevenLabs text frames are JSON; a binary frame is unmetered")

    def _transport_failed(self) -> None:
        # Contexts whose final message never arrived: whatever the provider
        # did with them cannot be established, so they are not charged.
        for context in sorted(self._open):
            request = self._meter.request_for_context(context)
            if request is not None:
                request.finish("transport_failure")
        self._open.clear()

    async def receive(self, *args: Any, **kwargs: Any) -> Any:
        # Endings are recorded where they are observed, so a failure or a
        # refusal keeps its outcome even if the worker cancels a moment later.
        try:
            message = await self._socket.receive(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._transport_failed()
            raise
        kind = getattr(getattr(message, "type", None), "name", "")
        if kind in _SOCKET_ENDINGS:
            self._transport_failed()
            return message
        data = getattr(message, "data", None)
        # Parse only the two message kinds that end a context; audio frames
        # (the bulk of the traffic) carry neither marker.
        if kind == "TEXT" and isinstance(data, str) and (
            '"error"' in data or '"isFinal"' in data or '"is_final"' in data
        ):
            try:
                packet = json.loads(data)
            except ValueError:
                packet = None
            if isinstance(packet, Mapping):
                context = _message_context(packet)
                if packet.get("error") and isinstance(context, str):
                    request = self._meter.request_for_context(context)
                    if request is not None:
                        request.finish("provider_error")
                    self._open.discard(context)
                elif (packet.get("isFinal") or packet.get("is_final")) and isinstance(context, str):
                    # The provider ended this context: a barge-in's
                    # abandonment is now final (ElevenLabs billed its text).
                    request = self._meter.request_for_context(context)
                    if request is not None:
                        request.mark_provider_end()
                    self._open.discard(context)
        return message


class _MeteredSession(MeteredHTTPSession):
    """The plugin's HTTP session: request bodies and WebSockets metered."""

    async def ws_connect(self, *args: Any, **kwargs: Any) -> _MeteredSocket:
        return _MeteredSocket(await self._session.ws_connect(*args, **kwargs), self._meter)


def _elevenlabs_post_body(arguments: dict[str, Any]) -> tuple[bytes, str, int, dict[str, Any]]:
    """The chunked endpoint's JSON body, exactly as aiohttp's ``json=`` writes it."""

    body = arguments.pop("json", None)
    if not isinstance(body, dict) or "data" in arguments:
        raise ElevenLabsRuntimeCompatibilityError("ElevenLabs synthesis request has no JSON body")
    return json.dumps(body).encode("utf-8"), "application/json", text_units(body, on_socket=False), arguments


def _metered_tts_class(elevenlabs_tts_class: type, tts_module: ModuleType) -> type:
    """Build, once per plugin module, the ElevenLabs TTS that meters its wire."""

    with _PATCH_LOCK:
        installed = getattr(tts_module, _METERED_CLASS_MARKER, None)
        if installed is not None:
            return installed
        stream_base = getattr(tts_module, "SynthesizeStream", None)
        chunked_base = getattr(tts_module, "ChunkedStream", None)
        client_closed = getattr(tts_module, "APIStatusError", None)
        default_options = getattr(tts_module, "DEFAULT_API_CONNECT_OPTIONS", None)
        if (not isinstance(stream_base, type) or not isinstance(chunked_base, type)
                or not isinstance(client_closed, type) or default_options is None
                or not callable(getattr(stream_base, "_run", None))
                or not callable(getattr(chunked_base, "_run", None))
                or not callable(getattr(elevenlabs_tts_class, "_ensure_session", None))
                or not callable(getattr(elevenlabs_tts_class, "_current_connection", None))):
            raise ElevenLabsRuntimeCompatibilityError("ElevenLabs plugin does not expose compatible stream classes")

        async def metered_attempt(stream: Any, attempt: Any) -> None:
            try:
                await run_metered_request(stream._tts._skysay_meter, attempt)
            except TTSMeterFrozen:
                # Usage evidence was taken at teardown: close like a client
                # that hung up rather than inviting LiveKit to retry.
                raise client_closed(
                    "TTS usage evidence was already taken", status_code=499, request_id=None, body=None
                ) from None

        class MeteredSynthesizeStream(stream_base):
            # The plugin names each attempt's context first thing in ``_run``
            # (inside the metered attempt), and every frame of that attempt
            # carries it: binding it here is what attributes the shared
            # socket's frames to this request.
            @property
            def _context_id(self) -> str:
                return self.__dict__.get("_skysay_context_id", "")

            @_context_id.setter
            def _context_id(self, value: str) -> None:
                self.__dict__["_skysay_context_id"] = value
                request = current_tts_request()
                if request is not None and isinstance(value, str) and value:
                    request.bind_provider_context(value, awaits_provider_end=True)

            async def _run(self, output_emitter: Any) -> None:
                await metered_attempt(self, lambda: super(MeteredSynthesizeStream, self)._run(output_emitter))

        class MeteredChunkedStream(chunked_base):
            async def _run(self, output_emitter: Any) -> None:
                await metered_attempt(self, lambda: super(MeteredChunkedStream, self)._run(output_emitter))

        class MeteredElevenLabsTTS(elevenlabs_tts_class):
            def __init__(self, *, native_unit_meter: NativeTTSUnitMeter, **kwargs: Any) -> None:
                if not isinstance(native_unit_meter, NativeTTSUnitMeter):
                    raise ElevenLabsRuntimeCompatibilityError("ElevenLabs requires a native unit meter")
                if native_unit_meter.billing.unit != "character":
                    raise ElevenLabsRuntimeCompatibilityError("ElevenLabs bills submitted characters")
                self._skysay_meter = native_unit_meter
                super().__init__(**kwargs)

            def stream(self, *, conn_options: Any = None) -> Any:
                options = conn_options if conn_options is not None else default_options
                stream = MeteredSynthesizeStream(tts=self, conn_options=options)
                self._streams.add(stream)
                return stream

            def synthesize(self, text: str, *, conn_options: Any = None) -> Any:
                options = conn_options if conn_options is not None else default_options
                return MeteredChunkedStream(tts=self, input_text=text, conn_options=options)

            def _ensure_session(self) -> Any:
                return _MeteredSession(super()._ensure_session(), self._skysay_meter, _elevenlabs_post_body)

        setattr(tts_module, _METERED_CLASS_MARKER, MeteredElevenLabsTTS)
        return MeteredElevenLabsTTS


def build_tts(*, native_unit_meter: NativeTTSUnitMeter, **kwargs: Any) -> Any:
    """Construct the metered LiveKit ElevenLabs TTS (every model, both protocols)."""

    if not isinstance(native_unit_meter, NativeTTSUnitMeter):
        raise ElevenLabsRuntimeCompatibilityError("ElevenLabs requires a native unit meter")
    try:
        from livekit.plugins import elevenlabs
    except ImportError as exc:  # pragma: no cover - exercised in worker image
        raise ElevenLabsRuntimeCompatibilityError("ElevenLabs LiveKit plugin is unavailable") from exc
    tts_module = sys.modules.get(elevenlabs.TTS.__module__)
    if not isinstance(tts_module, ModuleType):
        raise ElevenLabsRuntimeCompatibilityError("ElevenLabs plugin module is unavailable")
    metered = _metered_tts_class(elevenlabs.TTS, tts_module)
    return metered(native_unit_meter=native_unit_meter, **kwargs)


__all__ = ["ElevenLabsRuntimeCompatibilityError", "build_tts", "text_units"]

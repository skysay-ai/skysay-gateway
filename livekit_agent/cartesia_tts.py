# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Compatibility construction for Cartesia's current LiveKit wire contract.

The pinned LiveKit Cartesia plugin still serializes an id as the retired
``{"mode": "id", "id": ...}`` shape and defaults its bytes endpoint to an
older Cartesia API revision. Current Cartesia transports require a direct
voice-id string; the bytes endpoint uses ``locale`` while WebSocket messages
retain ``language``. Keep the narrow transport compatibility boundary beside
the worker rather than teaching the control plane about an SDK implementation.
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
        hand_off,
        mark_provider_error,
        mark_transport_failure,
        run_metered_request,
    )
except ImportError:  # pragma: no cover - direct voice-worker entrypoint
    from native_tts_usage import (
        MeteredHTTPSession,
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        current_tts_request,
        hand_off,
        mark_provider_error,
        mark_transport_failure,
        run_metered_request,
    )


CARTESIA_API_VERSION = "2026-08-14"
CARTESIA_AUTH_HEADER = "Authorization"
_PINNED_LEGACY_AUTH_HEADER = "X-API-Key"
_PATCH_MARKER = "_skysay_cartesia_direct_voice_api_version"
_METERED_CLASS_MARKER = "_skysay_cartesia_metered_tts"
_PATCH_LOCK = threading.Lock()


class CartesiaRuntimeCompatibilityError(RuntimeError):
    """The pinned plugin cannot safely speak Cartesia's current protocol."""


def bearer_auth_value(api_key: Any) -> str:
    """Return the current Cartesia auth value without retaining/logging a key."""

    key = str(api_key or "").strip()
    if not key:
        raise CartesiaRuntimeCompatibilityError("Cartesia requires an API key")
    return f"Bearer {key}"


def install_current_cartesia_wire_format(tts_module: ModuleType) -> None:
    """Adapt one verified LiveKit plugin module to Cartesia's current API.

    The plugin owns websocket lifecycle, streaming, cancellation, and metrics.
    This only changes the one provider payload field whose documented shape
    changed.  It is process-global by necessity because upstream stream classes
    call their module-level helper; the lock makes first construction safe when
    a worker accepts more than one job at once.
    """

    with _PATCH_LOCK:
        installed = getattr(tts_module, _PATCH_MARKER, None)
        if installed == CARTESIA_API_VERSION:
            return
        if installed is not None:
            raise CartesiaRuntimeCompatibilityError("Cartesia plugin protocol adapter is incompatible")
        original = getattr(tts_module, "_to_cartesia_options", None)
        if (
            not callable(original)
            or not isinstance(getattr(tts_module, "API_VERSION", None), str)
            or getattr(tts_module, "API_AUTH_HEADER", None) != _PINNED_LEGACY_AUTH_HEADER
        ):
            raise CartesiaRuntimeCompatibilityError("Cartesia plugin does not expose a compatible wire adapter")

        def current_options(options: Any, *, streaming: bool) -> dict[str, Any]:
            payload = original(options, streaming=streaming)
            voice = getattr(options, "voice", None)
            if not isinstance(payload, dict) or not isinstance(voice, str) or not voice.strip():
                raise CartesiaRuntimeCompatibilityError("Cartesia requires an exact voice id")
            # Copy because the upstream helper may reuse an internal mapping.
            payload = dict(payload)
            payload["voice"] = voice
            # Current Cartesia bytes requests use ``locale`` while its
            # streaming WebSocket message retains ``language``.  The pinned
            # plugin shares this one helper between both transports, so the
            # conversion must be scoped to the HTTP path rather than blindly
            # applying the REST field to agent turns.
            if not streaming:
                language = payload.pop("language", None)
                if language is not None:
                    payload["locale"] = language
            return payload

        # ``ChunkedStream`` resolves API_VERSION from this module at request
        # time; a constructor argument alone would leave its HTTP fallback on
        # the old revision.  Synthesis streams use the explicit option below.
        setattr(tts_module, "_to_cartesia_options", current_options)
        setattr(tts_module, "API_VERSION", CARTESIA_API_VERSION)
        # The pinned plugin passes its stored key directly as the header
        # value.  ``build_tts`` supplies the Bearer prefix; changing both
        # pieces here fixes its HTTP and WebSocket paths together.
        setattr(tts_module, "API_AUTH_HEADER", CARTESIA_AUTH_HEADER)
        setattr(tts_module, _PATCH_MARKER, CARTESIA_API_VERSION)


def _transcript_units(packet: Any) -> int:
    """Characters of the ``transcript`` one Cartesia request body carries."""

    transcript = packet.get("transcript") if isinstance(packet, Mapping) else None
    return len(transcript) if isinstance(transcript, str) else 0


class _MeteredSocket:
    """One pooled Cartesia socket that hands each transcript to the meter.

    The plugin builds every packet and owns the socket's lifecycle, pooling and
    close; this reads the packet it is about to send and the messages it
    receives, and delegates everything else.
    """

    def __init__(self, socket: Any, meter: NativeTTSUnitMeter) -> None:
        if getattr(socket, "compress", 0):
            # An uncompressed frame reaches the transport before the send's
            # first await, so a send the worker cancels was already handed.
            raise CartesiaRuntimeCompatibilityError("Cartesia socket must be uncompressed")
        self._socket = socket
        self._meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._socket, name)

    async def send_str(self, data: str, *args: Any, **kwargs: Any) -> Any:
        try:
            packet = json.loads(data)
        except (TypeError, ValueError):
            packet = None
        request = current_tts_request()
        if request is not None and isinstance(packet, dict) and isinstance(packet.get("context_id"), str):
            request.provider_context = packet["context_id"]
        return await hand_off(
            self._meter, _transcript_units(packet), lambda: self._socket.send_str(data, *args, **kwargs)
        )

    async def receive(self, *args: Any, **kwargs: Any) -> Any:
        # Endings are recorded where they are observed, so a failure or a
        # refusal keeps its outcome even if the worker cancels a moment later.
        try:
            message = await self._socket.receive(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException:
            mark_transport_failure()
            raise
        kind = getattr(getattr(message, "type", None), "name", "")
        if kind in _SOCKET_ENDINGS:
            mark_transport_failure()
            return message
        data = getattr(message, "data", None)
        if isinstance(data, str) and "error" in data:
            request = current_tts_request()
            try:
                packet = json.loads(data)
            except ValueError:
                packet = None
            if request is not None and isinstance(packet, dict) and packet.get("type") == "error":
                context = packet.get("context_id")
                # The plugin's own filter: a message tagged with another
                # context belongs to that context's request and is skipped.
                if context is None or context == request.provider_context:
                    mark_provider_error()
        return message


#: The aiohttp message types on which the pinned plugin raises "connection
#: closed unexpectedly".
_SOCKET_ENDINGS = frozenset({"CLOSE", "CLOSING", "CLOSED"})


def _cartesia_post_body(arguments: dict[str, Any]) -> tuple[bytes, str, int, dict[str, Any]]:
    """The bytes endpoint's JSON body, exactly as aiohttp's ``json=`` writes it."""

    body = arguments.pop("json", None)
    if not isinstance(body, dict) or "data" in arguments:
        raise CartesiaRuntimeCompatibilityError("Cartesia bytes request has no JSON body")
    return json.dumps(body).encode("utf-8"), "application/json", _transcript_units(body), arguments


def _metered_tts_class(cartesia_tts_class: type, tts_module: ModuleType) -> type:
    """Build, once per plugin module, the Cartesia TTS that meters its wire.

    ``stream()`` and ``synthesize()`` construct the metered stream classes
    directly, so nothing process-global is patched for accounting. Every
    attempt is one metered request; its units are counted where the plugin
    hands them to the socket or HTTP session.
    """

    with _PATCH_LOCK:
        installed = getattr(tts_module, _METERED_CLASS_MARKER, None)
        if installed is not None:
            return installed
        stream_base = getattr(tts_module, "SynthesizeStream", None)
        chunked_base = getattr(tts_module, "ChunkedStream", None)
        client_closed = getattr(tts_module, "APIStatusError", None)
        if (not isinstance(stream_base, type) or not isinstance(chunked_base, type)
                or not isinstance(client_closed, type)
                or not callable(getattr(stream_base, "_run", None))
                or not callable(getattr(chunked_base, "_run", None))
                or not callable(getattr(cartesia_tts_class, "_connect_ws", None))
                or not callable(getattr(cartesia_tts_class, "_ensure_session", None))):
            raise CartesiaRuntimeCompatibilityError("Cartesia plugin does not expose compatible stream classes")

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
            async def _run(self, output_emitter: Any) -> None:
                await metered_attempt(self, lambda: super(MeteredSynthesizeStream, self)._run(output_emitter))

        class MeteredChunkedStream(chunked_base):
            async def _run(self, output_emitter: Any) -> None:
                await metered_attempt(self, lambda: super(MeteredChunkedStream, self)._run(output_emitter))

        class MeteredCartesiaTTS(cartesia_tts_class):
            def __init__(self, *, native_unit_meter: NativeTTSUnitMeter, **kwargs: Any) -> None:
                if not isinstance(native_unit_meter, NativeTTSUnitMeter):
                    raise CartesiaRuntimeCompatibilityError("Cartesia requires a native unit meter")
                if native_unit_meter.billing.unit != "character":
                    raise CartesiaRuntimeCompatibilityError("Cartesia bills submitted characters")
                self._skysay_meter = native_unit_meter
                super().__init__(**kwargs)

            def stream(self, *, conn_options: Any = None) -> Any:
                options = conn_options if conn_options is not None else tts_module.DEFAULT_API_CONNECT_OPTIONS
                stream = MeteredSynthesizeStream(tts=self, conn_options=options)
                self._streams.add(stream)
                return stream

            def synthesize(self, text: str, *, conn_options: Any = None) -> Any:
                options = conn_options if conn_options is not None else tts_module.DEFAULT_API_CONNECT_OPTIONS
                return MeteredChunkedStream(tts=self, input_text=text, conn_options=options)

            async def _connect_ws(self, timeout: float) -> Any:
                return _MeteredSocket(await super()._connect_ws(timeout), self._skysay_meter)

            def _ensure_session(self) -> Any:
                return MeteredHTTPSession(super()._ensure_session(), self._skysay_meter, _cartesia_post_body)

        setattr(tts_module, _METERED_CLASS_MARKER, MeteredCartesiaTTS)
        return MeteredCartesiaTTS


def build_tts(*, native_unit_meter: NativeTTSUnitMeter, **kwargs: Any) -> Any:
    """Construct the metered Cartesia LiveKit adapter pinned to the current API."""

    if not isinstance(native_unit_meter, NativeTTSUnitMeter):
        raise CartesiaRuntimeCompatibilityError("Cartesia requires a native unit meter")
    requested_version = str(kwargs.pop("api_version", CARTESIA_API_VERSION) or "")
    if requested_version != CARTESIA_API_VERSION:
        raise CartesiaRuntimeCompatibilityError("Cartesia API version must be current")
    kwargs["api_key"] = bearer_auth_value(kwargs.get("api_key"))
    try:
        from livekit.plugins import cartesia
    except ImportError as exc:  # pragma: no cover - exercised in worker image
        raise CartesiaRuntimeCompatibilityError("Cartesia LiveKit plugin is unavailable") from exc
    tts_module = sys.modules.get(cartesia.TTS.__module__)
    if not isinstance(tts_module, ModuleType):
        raise CartesiaRuntimeCompatibilityError("Cartesia plugin module is unavailable")
    install_current_cartesia_wire_format(tts_module)
    metered = _metered_tts_class(cartesia.TTS, tts_module)
    return metered(native_unit_meter=native_unit_meter, api_version=CARTESIA_API_VERSION, **kwargs)

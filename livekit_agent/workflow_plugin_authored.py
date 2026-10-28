# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One finite authored request over one pinned LiveKit TTS plugin, in process.

The ElevenLabs, Azure, Fish Audio and Soniox authored sources share this
adapter. Their pinned plugins speak plain aiohttp HTTP or websocket from this
event loop, cancel cooperatively (``utils.aio.cancel_and_wait``) and decode in
line (raw PCM, or the inline WAV decoder): none starts a thread or a child
process that could outlive a cancelled call. So, unlike the Gemini source,
no isolated worker is needed; the adapter owns the plugin objects, the HTTP
session they use and every task they start, and joins them on close.

Audio leaves only in the root gate's declared format (24 kHz mono 16-bit).
The plugin's own ``is_final`` is required exactly once and last, and this
adapter still holds its last frame until the plugin task has finished with
no error: a final frame is never released after an error, a cancellation or
a bare end of stream. Usage is reported by the provider module's own rule; a
local count is never reported as exact before a positive provider completion.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import re
import unicodedata
from types import SimpleNamespace

try:
    from .capabilities.workflow_sources import is_authored_constructor
    from .workflow_source_owner import SourceMetric, SourceObservation
    from .workflow_tasks import join_owned_task
    from .native_tts_usage import NativeTTSUnitMeter, NativeTTSUsageError, RequestBilling
    from .capabilities.registry import load_registry
except ImportError:
    from capabilities.workflow_sources import is_authored_constructor
    from workflow_source_owner import SourceMetric, SourceObservation
    from workflow_tasks import join_owned_task
    from native_tts_usage import NativeTTSUnitMeter, NativeTTSUsageError, RequestBilling
    from capabilities.registry import load_registry


SAMPLE_RATE = 24000
# The root gate seals at most 15 s of 24 kHz audio per segment. At ~14 spoken
# characters per second, 160 UTF-8 bytes keeps a segment under that bound with
# margin for number and abbreviation expansion; a 3-byte script gets ~53 chars.
MAX_INPUT_UTF8 = 160
MAX_SAMPLES = 15 * SAMPLE_RATE
MAX_FRAMES = 2048
# Wall bound of one provider request, below the gate's 30 s capture expiry.
REQUEST_DEADLINE = 20.0
CLOSE_TIMEOUT = 5.0
_REFERENCE = re.compile(r"[A-Za-z0-9_-]{16,96}")
_SECRET_FIELDS = ("api_key", "subscription_key", "auth_token")


class AuthoredSourceRefused(RuntimeError):
    """Fixed failure, without provider payloads or credentials."""


def refuse():
    raise AuthoredSourceRefused("finite authored source refused")


def require_inputs(constructor, provider, model, voice_pattern, secret, client_reference_id):
    """The exact frozen authored constructor, secret and reference, or a refusal.

    Returns ``(voice_id, locale)``; the shape is the one every authored
    factory and the root plan share.
    """
    if (not is_authored_constructor(constructor)
            or constructor["provider"] != provider
            or constructor["model"] != model
            or voice_pattern.fullmatch(constructor["voice_id"]) is None
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in constructor["voice_id"])
            or type(secret) is not str or not secret.strip() or len(secret) > 16384
            or any(char in secret for char in "\r\n\0")
            or type(client_reference_id) is not str or _REFERENCE.fullmatch(client_reference_id) is None):
        refuse()
    return constructor["voice_id"], constructor["locale"]


def _new_http_session():
    """The one network constructor; tests replace it with a scripted session."""
    import aiohttp
    from livekit.agents.utils import http_context
    # The pinned SDK's own trust-store resolution (certifi when the image
    # has no system store), without its process-wide shared session.
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=2, ssl=http_context._create_ssl_context()))


class OwnedHttpSession:
    """The plugin's ``http_session``: created on first use, closed by its source.

    ``touched`` is set BEFORE a request is handed to aiohttp. A failure after
    that cannot prove the provider received nothing, so it is never reported
    as a source that did not connect.
    """
    __slots__ = ("_inner", "_closed", "touched")

    def __init__(self):
        self._inner = None
        self._closed = False
        self.touched = False

    def __repr__(self):
        return "<OwnedHttpSession redacted>"

    def __reduce__(self):
        raise TypeError("authored sessions cannot be serialized")

    def _session(self):
        if self._closed:
            refuse()
        if self._inner is None:
            self._inner = _new_http_session()
        return self._inner

    def post(self, *args, **kwargs):
        session = self._session()
        self.touched = True
        return session.post(*args, **kwargs)

    def ws_connect(self, *args, **kwargs):
        session = self._session()
        self.touched = True
        return session.ws_connect(*args, **kwargs)

    async def aclose(self):
        self._closed = True
        inner, self._inner = self._inner, None
        if inner is not None:
            await inner.close()


class _PluginStream:
    """The public finite stream: plugin frames, final held until clean closure."""

    def __init__(self, source, plugin_stream, text):
        self._source, self._plugin, self._text = source, plugin_stream, text
        self._frames = []
        self._index = 0
        self._samples = 0
        self._changed = asyncio.Event()
        self._finished = self._successful = self._plugin_final = False
        self._closing = None
        self._pump = asyncio.create_task(self._run())

    def __repr__(self):
        return "<PluginAuthoredStream redacted>"

    def __reduce__(self):
        raise TypeError("authored streams cannot be serialized")

    @property
    def input_text(self):
        return self._text

    @property
    def done(self):
        return self._finished

    @property
    def exception(self):
        return None if self._successful else AuthoredSourceRefused("finite authored source refused")

    def _accept(self, event):
        frame = event.frame
        count, rate, channels = frame.samples_per_channel, frame.sample_rate, frame.num_channels
        data = memoryview(frame.data).cast("B").tobytes()
        if (self._plugin_final or type(event.is_final) is not bool
                or type(count) is not int or count <= 0 or rate != SAMPLE_RATE or channels != 1
                or len(data) != count * 2 or len(self._frames) >= MAX_FRAMES
                or self._samples + count > MAX_SAMPLES):
            refuse()
        self._frames.append(data)
        self._samples += count
        self._plugin_final = event.is_final

    async def _run(self):
        try:
            async with asyncio.timeout(REQUEST_DEADLINE):
                async for event in self._plugin:
                    self._accept(event)
                    self._changed.set()
            plugin = self._plugin
            # ``__anext__`` also ends silently on a cancelled plugin task, and
            # on the plugin's graceful 499 return: neither is a completion.
            self._successful = (self._plugin_final and bool(self._frames) and plugin.done is True
                                and not plugin._synthesize_task.cancelled()
                                and plugin._synthesize_task.exception() is None)
        except BaseException:
            self._successful = False
        finally:
            self._finished = True
            self._changed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            available = len(self._frames) - self._index
            if available > 1 or available == 1 and self._finished and self._successful:
                from livekit import rtc
                data = self._frames[self._index]
                self._index += 1
                return SimpleNamespace(frame=rtc.AudioFrame(data=data, sample_rate=SAMPLE_RATE,
                    num_channels=1, samples_per_channel=len(data) // 2),
                    is_final=self._finished and self._successful and self._index == len(self._frames))
            if self._finished:
                if not self._successful:
                    refuse()
                raise StopAsyncIteration
            self._changed.clear()
            await self._changed.wait()

    async def aclose(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._shutdown())
        await join_owned_task(self._closing)

    async def _shutdown(self):
        # Directly cancellable, so the owning source can bound it.
        if not self._pump.done():
            self._pump.cancel()
        await asyncio.gather(self._pump, return_exceptions=True)
        await self._plugin.aclose()


def native_unit_meter(provider, model):
    """The provider's call-local meter, charged by its capability's request rule.

    The same construction the Cascade path makes: what a request is charged
    once it ends (completed, abandoned by the worker, a transport failure, a
    provider error) is the capability artifact's ``billing.request_billing``.
    """
    return NativeTTSUnitMeter(RequestBilling.from_capability(load_registry().tts_metering_by_vendor(provider, model)))


def request_outcome_usage(meter):
    """``settle`` for a metered source: the units its request outcome is charged.

    Read once every task of the line is joined, exactly as the Cascade path
    settles a TTS request: each request is charged by its outcome under the
    capability rule (a completed or worker-abandoned request its handed units,
    a refused or failed one none). A request whose ending was never observed,
    or units handed outside a request, have no exact fact.
    """
    async def settle():
        await meter.settle(0)  # every request is joined: stop admission and freeze
        if meter.diagnostic()["uncertain_requests"]:
            return None
        if meter.billing.measure != "submitted_units":
            return None  # a provider-reported unit has no single handed count
        try:
            (billed,) = meter.snapshot().values()
        except NativeTTSUsageError:
            return None
        return ((meter.billing.unit, billed),)
    return settle


class PluginAuthoredSource:
    """One request, one stream; usage and closure owned by the provider rule.

    ``sent()`` says whether billable input may have reached the provider;
    ``exact(text)`` returns the native units of a positively completed request
    or ``None`` when no exact fact exists. A source on a native unit meter
    passes ``settle`` instead: its request outcome is the fact whatever the
    stream did, as the Cascade path settles it (``request_outcome_usage``).
    The plugin TTS's ``aclose`` must join every task that TTS started; the
    session closes after it.
    """

    def __init__(self, *, tts, session, reference, dimensions, sent, exact=None, settle=None):
        if (exact is None) == (settle is None):
            refuse()
        self._tts, self._session, self._reference = tts, session, reference
        self._dimensions, self._sent, self._exact, self._settle = dimensions, sent, exact, settle
        self._stream = None
        self._closing = None
        self._used = self._closed = self._acknowledged = False
        self._units = None
        self._was_sent = False

    def __repr__(self):
        return "<PluginAuthoredSource redacted>"

    def __reduce__(self):
        raise TypeError("authored sources cannot be serialized")

    def synthesize(self, text, *, conn_options):
        stream = None
        try:
            if (not self._used and not self._closed and type(text) is str and text.strip()
                    and len(text.encode("utf-8")) <= MAX_INPUT_UTF8
                    and type(conn_options.max_retry) is int and conn_options.max_retry == 0
                    and type(conn_options.timeout) in (int, float)
                    and math.isfinite(conn_options.timeout) and 0 < conn_options.timeout <= 10):
                self._used = True
                from livekit.agents import APIConnectOptions
                plugin = self._tts.synthesize(text, conn_options=APIConnectOptions(
                    max_retry=0, timeout=float(conn_options.timeout)))
                stream = _PluginStream(self, plugin, text)
        except Exception:
            stream = None
        if stream is None:
            del text, conn_options
            refuse()
        self._stream = stream
        return stream

    async def wait_terminal(self):
        if self._stream is None:
            return False
        try:
            await asyncio.shield(self._stream._pump)
        except asyncio.CancelledError:
            if not self._stream._pump.cancelled():
                raise
            return False
        return self._stream._successful

    def observation(self):
        zero = tuple((key, 0) for key in self._dimensions)
        # The PCM the provider returned for this request, whatever its ending:
        # what a measured-duration rule settles.
        measured = (self._stream._samples if self._stream is not None else 0) / SAMPLE_RATE
        # Without a joined closure there is no terminal fact, and the owner
        # records ``connect_unknown``, which must carry no provider session.
        if not self._acknowledged:
            return SourceObservation("connect_unknown", zero, (), "", False, measured_seconds=measured)
        if not self._was_sent:
            return SourceObservation("connect_unknown", zero, (), "", True, measured_seconds=measured)
        units = self._units
        if units is None:
            return SourceObservation("partial_unknown", zero, (), self._reference, self._acknowledged,
                                     measured_seconds=measured)
        return SourceObservation("exact", units, (SourceMetric(units, self._reference),),
                                 self._reference, self._acknowledged, measured_seconds=measured)

    async def aclose(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await join_owned_task(self._closing)

    async def _close(self):
        clean = False
        # Streaming plugins copy their options (and key) into each inner
        # stream; take them before the TTS forgets its stream set.
        holders = [self._tts, *list(getattr(self._tts, "_streams", ()))]
        if self._stream is not None:
            holders.append(self._stream._plugin)
        try:
            async with asyncio.timeout(CLOSE_TIMEOUT):
                if self._stream is not None:
                    await self._stream._shutdown()
                await self._tts.aclose()
                await self._session.aclose()
            clean = True
        except BaseException:
            clean = False
        finally:
            if not clean:
                with contextlib.suppress(BaseException):
                    async with asyncio.timeout(1):
                        await self._session.aclose()
            # Facts are fixed only now, from the joined stream and the plugin's
            # own counter; nothing later can promote them.
            self._was_sent = bool(self._sent())
            stream = self._stream
            if clean and stream is not None and self._was_sent:
                if self._settle is not None:
                    units = await self._settle()
                    if units is not None and self._valid_units(units, charged=True):
                        self._units = units
                elif stream._successful:
                    units = self._exact(stream.input_text)
                    if units is not None and self._valid_units(units):
                        self._units = units
            self._scrub(holders)
            self._closed = True
            self._acknowledged = clean

    def _valid_units(self, units, *, charged=False):
        # A request outcome may be charged nothing (a refusal): that is exact.
        return (type(units) is tuple and tuple(key for key, _ in units) == tuple(self._dimensions)
                and all(type(value) is int and (value >= 0 if charged else value > 0) for _, value in units))

    def _scrub(self, holders):
        for holder in holders:
            options = getattr(holder, "_opts", None)
            for name in _SECRET_FIELDS:
                if hasattr(options, name):
                    with contextlib.suppress(Exception):
                        setattr(options, name, None)
        self._tts = None

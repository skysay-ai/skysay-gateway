# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite authored source with an explicitly bound isolated worker."""
from __future__ import annotations

import asyncio
import contextlib
import math
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

try:
    from . import workflow_authored_wire as wire
    from .capabilities.workflow_sources import is_authored_constructor
    from .workflow_source_owner import SourceMetric, SourceObservation
    from .workflow_tasks import join_owned_task
except ImportError:
    import workflow_authored_wire as wire
    from capabilities.workflow_sources import is_authored_constructor
    from workflow_source_owner import SourceMetric, SourceObservation
    from workflow_tasks import join_owned_task


class AuthoredSourceRefused(RuntimeError):
    """Fixed failure, without provider payloads or credentials."""


def _fail():
    raise AuthoredSourceRefused("finite authored source refused")


def _worker_command():
    return sys.executable, "-I", str(Path(__file__).resolve().with_name("workflow_gemini_authored_worker.py"))


async def _receive(reader):
    kind, size = wire.header(await reader.readexactly(wire.HEADER.size))
    return kind, await reader.readexactly(size)


class GeminiAuthoredWorkerPool:
    """One process, one active lease, no queue and no automatic respawn."""
    def __init__(self):
        self._process = self._transport = self._reader = self._monitor = None
        self._starting = self._closing = None
        self._active = None
        self._references = set()
        self._started = self._ready = self._closed = self._poisoned = False

    def __repr__(self):
        return "<GeminiAuthoredWorkerPool redacted>"

    def __reduce__(self):
        raise TypeError("authored worker pools cannot be serialized")

    async def start(self):
        if self._started or self._closed:
            _fail()
        self._started = True
        self._starting = asyncio.create_task(self._start())
        try:
            await join_owned_task(self._starting)
        except asyncio.CancelledError:
            await self.aclose()
            raise
        return self

    async def _start(self):
        if self._closed:
            _fail()
        read_fd, write_fd = os.pipe()
        pipe = None
        try:
            self._process = await asyncio.create_subprocess_exec(*_worker_command(), str(write_fd),
                pass_fds=(write_fd,), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env={"LANG": "C.UTF-8", "TZ": "UTC"})
            os.close(write_fd)
            write_fd = None
            pipe = os.fdopen(read_fd, "rb", buffering=0)
            read_fd = None
            self._reader = asyncio.StreamReader(limit=wire.MAX_PACKET + wire.HEADER.size)
            self._transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(self._reader), pipe)
            pipe = None
            async with asyncio.timeout(8):
                kind, payload = await _receive(self._reader)
                if (kind != wire.Kind.PREFLIGHT
                        or wire.unpack_json(payload, {"protocol"}) != {"protocol": wire.PROTOCOL}):
                    wire.refuse()
            if self._closed:
                wire.refuse()
            self._ready = True
            self._monitor = asyncio.create_task(self._watch())
        except BaseException:
            await self._poison()
        finally:
            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)
            if pipe is not None:
                pipe.close()
        if not self._ready:
            _fail()

    async def _watch(self):
        await self._process.wait()
        if not self._closed:
            self._poisoned, self._ready = True, False

    def _claim(self, source):
        if (not self._ready or self._closed or self._poisoned or self._active is not None
                or self._process.returncode is not None
                or source._reference in self._references or len(self._references) >= wire.MAX_SOURCES):
            _fail()
        self._references.add(source._reference)
        self._active = source

    async def _send(self, kind, payload):
        self._process.stdin.write(wire.encode(kind, payload))
        async with asyncio.timeout(2):
            await self._process.stdin.drain()

    async def _poison(self):
        self._poisoned, self._ready = True, False
        if self._process is not None:
            if self._process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    self._process.kill()
            await self._process.wait()
        if self._transport is not None:
            self._transport.close()
        if self._monitor is not None and self._monitor is not asyncio.current_task():
            await self._monitor

    async def aclose(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await join_owned_task(self._closing)

    async def _close(self):
        if self._closed:
            return
        self._closed = True
        if self._starting is not None and not self._starting.done():
            with contextlib.suppress(Exception):
                await self._starting
        if self._active is not None:
            await self._active.aclose()
        if self._process is not None and self._process.returncode is None:
            if self._ready and not self._poisoned:
                try:
                    await self._send(wire.Kind.STOP, b"")
                    async with asyncio.timeout(2):
                        await self._process.wait()
                except Exception:
                    await self._poison()
            else:
                await self._poison()
        self._ready = False
        if self._transport is not None:
            self._transport.close()
        if self._monitor is not None:
            await self._monitor


class _AuthoredStream:
    def __init__(self, source, text, timeout):
        self._source, self._text, self._timeout = source, text, timeout
        self._frames = []
        self._index = 0
        self._changed = asyncio.Event()
        self._lease = wire.IncomingLease(secrets.token_hex(16))
        self._cancelled = self._finished = self._successful = False
        self._sent = self._control_sent = False
        self._config_sent = asyncio.Event()
        self._closing = None
        self._runner = asyncio.create_task(self._run())

    def __repr__(self):
        return "<GeminiAuthoredStream redacted>"

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

    async def _run(self):
        try:
            async with asyncio.timeout(self._timeout + 4):
                await self._source._pool._send(wire.Kind.CONFIG, wire.pack_json({
                    "lease": self._lease.identity, "provider": wire.PROVIDER, "model": wire.MODEL,
                    "voice_id": self._source._voice, "secret": self._source._secret, "text": self._text}))
                self._sent = True
                self._config_sent.set()
                self._source._secret = ""
                while not self._lease.closed:
                    kind, payload = await _receive(self._source._pool._reader)
                    data = self._lease.accept(kind, payload)
                    if data is not None:
                        self._frames.append(data)
                    if kind == wire.Kind.END and not self._cancelled:
                        self._control_sent = True
                        await self._source._pool._send(wire.Kind.RELEASE,
                            wire.pack_json({"lease": self._lease.identity}))
                    self._changed.set()
                self._successful = self._lease.end.functional and not self._cancelled
        except BaseException:
            await self._source._pool._poison()
        finally:
            self._source._secret = ""
            self._finished = True
            self._config_sent.set()
            if self._source._pool._active is self._source:
                self._source._pool._active = None
            self._changed.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            # Hold the final PCM frame until positive functional closure.
            available = len(self._frames) - self._index
            if available > 1 or available == 1 and self._finished and self._successful:
                from livekit import rtc
                data = self._frames[self._index]
                self._index += 1
                return SimpleNamespace(frame=rtc.AudioFrame(data=data, sample_rate=wire.SAMPLE_RATE,
                    num_channels=1, samples_per_channel=len(data) // 2),
                    is_final=self._finished and self._successful and self._index == len(self._frames))
            if self._finished:
                if not self._successful:
                    _fail()
                raise StopAsyncIteration
            self._changed.clear()
            await self._changed.wait()

    async def aclose(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await join_owned_task(self._closing)

    async def _close(self):
        if not self._runner.done():
            self._cancelled = True
            try:
                async with asyncio.timeout(3):
                    await self._config_sent.wait()
                    if self._sent and not self._control_sent:
                        self._control_sent = True
                        await self._source._pool._send(wire.Kind.CANCEL,
                            wire.pack_json({"lease": self._lease.identity}))
                    await asyncio.shield(self._runner)
            except BaseException:
                await self._source._pool._poison()
                try:
                    async with asyncio.timeout(1):
                        await asyncio.shield(self._runner)
                except BaseException:
                    self._runner.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._runner
        else:
            await self._runner


class GeminiAuthoredSource:
    def __init__(self, secret, voice, reference):
        self._secret, self._voice, self._reference = secret, voice, reference
        self._pool = self._stream = None
        self._closing = None
        self._bound = self._used = self._closed = False

    def __repr__(self):
        return "<GeminiAuthoredSource redacted>"

    def __reduce__(self):
        raise TypeError("authored sources cannot be serialized")

    def bind_worker(self, pool):
        if (self._bound or self._closed or type(pool) is not GeminiAuthoredWorkerPool
                or not pool._ready or pool._closed or pool._poisoned):
            _fail()
        self._bound, self._pool = True, pool

    def synthesize(self, text, *, conn_options):
        valid = False
        try:
            valid = (self._bound and not self._closed and not self._used
                     and type(text) is str and bool(text.strip())
                     and len(text.encode("utf-8")) <= wire.MAX_TEXT_UTF8
                     and type(conn_options.max_retry) is int and conn_options.max_retry == 0
                     and type(conn_options.timeout) in (int, float)
                     and math.isfinite(conn_options.timeout) and 0 < conn_options.timeout <= 10)
            if valid:
                self._pool._claim(self)
        except Exception:
            valid = False
        if not valid:
            del text, conn_options
            _fail()
        self._used = True
        self._stream = _AuthoredStream(self, text, conn_options.timeout)
        return self._stream

    async def wait_terminal(self):
        if self._stream is None:
            return False
        try:
            await asyncio.shield(self._stream._runner)
        except asyncio.CancelledError:
            if not self._stream._runner.cancelled():
                raise
            return False
        return self._stream._successful

    async def aclose(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._close())
        await join_owned_task(self._closing)

    async def _close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._stream is not None:
                await self._stream.aclose()
        finally:
            self._secret = ""

    def observation(self):
        zero = (("input_text_tokens", 0), ("output_audio_tokens", 0))
        lease = self._stream._lease if self._stream is not None else None
        if lease is None or not lease.seen:
            return SourceObservation("connect_unknown", zero, (), "", False)
        end = lease.end
        if end is None or end.tokens is None:
            return SourceObservation("partial_unknown", zero, (), self._reference, True)
        units = tuple(zip(("input_text_tokens", "output_audio_tokens"), end.tokens))
        metric = SourceMetric(units, self._reference)
        return SourceObservation("exact" if lease.closed else "partial_unknown",
                                 units, (metric,), self._reference, True)


def build_gemini_authored_source(*, secret, constructor, client_reference_id):
    result = value = identity = None
    try:
        # Gemini speech takes no language parameter: the shared constructor's
        # locale is validated and stays out of the worker wire.
        if not is_authored_constructor(constructor) or not wire.reference(client_reference_id):
            wire.refuse()
        identity = {key: constructor[key] for key in ("provider", "model", "voice_id")}
        value = wire.configuration(wire.pack_json(dict(identity, lease="0" * 32,
                                                       secret=secret, text="validation")))
        result = GeminiAuthoredSource(secret, value.voice_id, client_reference_id)
        value.clear()
    except Exception:
        pass
    finally:
        del secret, constructor, client_reference_id, value, identity
    if result is None:
        _fail()
    return result

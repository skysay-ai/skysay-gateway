# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Isolated, inactive finite recognition process; no usage or provider truth."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import faulthandler
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import sys
import sysconfig

CONFIG, AUDIO, EOF, PREFLIGHT, STARTED, FINAL, TERMINAL, ERROR, PUSHED = range(1, 10)
#: The recognizer's native input rate (validated in ``_build_stream``):
#: ``PUSHED`` counts 16-bit mono samples the SDK push stream accepted at it.
NATIVE_RATE = 16000
#: Native samples between two ``PUSHED`` acknowledgements (0.5 s).
PUSHED_EVERY = NATIVE_RATE // 2
HEADER = struct.Struct("!BI")
MAX_PACKET = 262144
MAX_SAMPLES = 7200000
MAX_CHARACTERS, MAX_UTF8, MAX_FINALS, MAX_EVENTS = 16000, 64000, 16000, 48032
_EOF_SECONDS = 10
_TAG = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_REGION = re.compile(r"[a-z][a-z0-9]{1,31}")
_PLUGIN_PIN = "8d4226720f01acdff445f9cd7402d45a753e4fb0b72b540ac664d4e42e302b61"
_BASE_PIN = "b8461e1e39616c84df12fd1eebad100cfa78d74f9b83a7cd6732d1d1ffd775d8"


class WireRefused(RuntimeError):
    """Fixed private failure; no provider details cross the boundary."""


def _refuse():
    raise WireRefused("finite recognition refused")


def _environment_allowed(names):
    return names <= {"LANG", "TZ"}


def _verify_pins(plugin, base):
    if hashlib.sha256(plugin).hexdigest() != _PLUGIN_PIN or hashlib.sha256(base).hexdigest() != _BASE_PIN:
        _refuse()


def _number(text):
    if len(text) > 32:
        _refuse()
    number = float(text) if any(c in text for c in ".eE") else int(text)
    if not math.isfinite(number):
        _refuse()
    return number


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            _refuse()
        result[key] = value
    return result


def parse_json(raw):
    if type(raw) is not bytes or len(raw) > MAX_PACKET:
        _refuse()
    depth, quoted, escaped = 0, False, False
    for value in raw:
        if quoted:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                quoted = False
        elif value == 34:
            quoted = True
        elif value in (91, 123):
            depth += 1
            if depth > 32:
                _refuse()
        elif value in (93, 125):
            depth -= 1
    failed = False
    result = None
    try:
        result = json.loads(raw, object_pairs_hook=_pairs, parse_int=_number, parse_float=_number,
                            parse_constant=lambda _: _refuse())
    except Exception:
        failed = True
    if failed or type(result) is not dict:
        _refuse()
    return result


def encode(kind, value):
    raw = value if type(value) is bytes else json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                       separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_PACKET:
        _refuse()
    return HEADER.pack(kind, len(raw)) + raw


async def receive(reader):
    raw = await reader.readexactly(HEADER.size)
    kind, size = HEADER.unpack(raw)
    if size > MAX_PACKET:
        _refuse()
    return kind, await reader.readexactly(size)


def _validate_config(value):
    if type(value) is not dict or set(value) != {"protocol", "secret", "region", "mode", "locales"}:
        _refuse()
    secret, region, mode, locales = (value[k] for k in ("secret", "region", "mode", "locales"))
    if (type(value["protocol"]) is not int or value["protocol"] != 1
            or type(secret) is not str or not 1 <= len(secret) <= 16384 or not secret.strip()
            or type(region) is not str or _REGION.fullmatch(region) is None
            or type(mode) is not str or mode not in {"single", "continuous_language_identification"}
            or type(locales) is not list or not 1 <= len(locales) <= 32
            or any(type(tag) is not str or len(tag) > 64 or _TAG.fullmatch(tag) is None for tag in locales)
            or len({tag.casefold() for tag in locales}) != len(locales)
            or (mode == "single") != (len(locales) == 1)):
        _refuse()
    return value


class Observer:
    """Only loop-queued callback order is visible, never native emission order."""
    def __init__(self):
        self.eof = self.invalid = self.frozen = False
        self.starts = self.stops = self.ends = self.users = 0
        self.started = asyncio.Event()

    def end_input(self):
        if self.eof:
            self.invalid = True
        self.eof = True

    def observe(self, kind, reason=""):
        if self.frozen:
            return
        if kind == "session_started":
            self.starts += 1
            self.invalid |= self.starts != 1 or self.stops != 0 or self.ends != 0 or self.users != 0
            self.started.set()
        elif kind == "session_stopped":
            self.stops += 1
            self.invalid |= not self.eof or self.starts != 1 or self.stops != 1
        elif kind == "canceled" and reason == "EndOfStream":
            self.ends += 1
            self.invalid |= not self.eof or self.starts != 1 or self.ends != 1
        elif kind == "canceled" and reason == "CancelledByUser":
            self.users += 1
            self.invalid |= self.stops != 1 or self.users != 1
        else:
            self.invalid = True

    def accepted(self, task):
        self.frozen = True
        return (self.eof and not self.invalid and self.starts == self.stops == 1
                and task.done() and not task.cancelled() and task.exception() is None)


def _forbidden_modules():
    return any(name == "sitecustomize" or name.startswith(("sentry", "ddtrace", "opentelemetry.instrumentation"))
               for name in sys.modules)


def _preflight():
    names = frozenset(os.environ)
    if not sys.flags.isolated or not _environment_allowed(names) or _forbidden_modules():
        _refuse()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.disable()
    logging.disable(sys.maxsize)
    sys.excepthook = lambda *_: None
    sys.unraisablehook = lambda *_: None
    from opentelemetry import trace
    trace.set_tracer_provider(trace.NoOpTracerProvider())
    from livekit.plugins.azure import stt as plugin
    if _forbidden_modules():
        _refuse()
    return {"clean": True, "environment_names": sorted(names)}


def _count_native_writes(push_stream, observer):
    """Count bytes the Azure SDK push stream accepts, after each successful write.

    The pinned plugin writes resampled 16 kHz PCM to this per-session object
    only once its session started; a write that raises counts nothing, and PCM
    still queued in the plugin is never counted.
    """
    write = push_stream.write

    def counted(buffer):
        write(buffer)
        observer.written += len(buffer)
    push_stream.write = counted
    return push_stream


def _build_stream(config):
    _validate_config(config)
    site = Path(sysconfig.get_paths()["purelib"])
    _verify_pins((site / "livekit/plugins/azure/stt.py").read_bytes(),
                 (site / "livekit/agents/stt/stt.py").read_bytes())
    path = Path(__file__).resolve().with_name("azure_speech.py")
    spec = importlib.util.spec_from_file_location("owned_azure_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from livekit.plugins.azure import stt as plugin
    from livekit.agents import APIConnectOptions, stt
    from livekit.agents.utils import is_given
    kwargs = {"language": config["locales"][0]} if config["mode"] == "single" else {"candidate_locales": config["locales"]}
    adapter = builder.build_stt(speech_key=config["secret"], speech_region=config["region"], **kwargs)
    opts = copy.deepcopy(adapter._config)
    if (any(is_given(value) for value in (opts.speech_host, opts.speech_endpoint, opts.speech_auth_token))
            or opts.speech_region != config["region"] or opts.language != config["locales"]
            or opts.sample_rate != 16000 or opts.num_channels != 1):
        _refuse()
    observer = Observer()
    # Native PCM bytes the Azure SDK push stream accepted: counted
    # at each successful ``PushAudioInputStream.write``, never at the queue.
    observer.written = 0
    native_locales = ({tag.casefold(): tag for tag in config["locales"]}
                      if config["mode"] == "continuous_language_identification" else {})

    class OwnedSpeechStream(plugin.SpeechStream):
        @property
        def _stream(self):
            return self._owned_push_stream

        @_stream.setter
        def _stream(self, push_stream):
            # The pinned plugin assigns a fresh SDK push stream per session.
            self._owned_push_stream = _count_native_writes(push_stream, observer)

        def _on_recognized(self, evt):
            # Read this final's native property before the plugin can default
            # it to the first configured locale. Never infer from the text.
            text = evt.result.text
            if type(text) is not str:
                with contextlib.suppress(RuntimeError):
                    self._loop.call_soon_threadsafe(observer.observe, "invalid_final")
                return
            if not text.strip():
                return
            language = ""
            if native_locales:
                with contextlib.suppress(Exception):
                    native = plugin.speechsdk.AutoDetectSourceLanguageResult(evt.result).language
                    if type(native) is str:
                        language = native_locales.get(native.casefold(), "")
            final = stt.SpeechData(language=language, confidence=1.0, text=text,
                start_time=evt.result.offset / 10**7 + self.start_time_offset,
                end_time=(evt.result.offset + evt.result.duration) / 10**7 + self.start_time_offset)
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._event_ch.send_nowait,
                    stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[final]))

        def _on_session_started(self, evt):
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(observer.observe, "session_started")
                self._loop.call_soon_threadsafe(self._session_started_event.set)

        def _on_session_stopped(self, evt):
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(observer.observe, "session_stopped")
            super()._on_session_stopped(evt)

        def _on_canceled(self, evt):
            reason = ""
            with contextlib.suppress(Exception):
                value = evt.cancellation_details.reason.name
                if type(value) is str:
                    reason = value
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(observer.observe, "canceled", reason)
            super()._on_canceled(evt)

        def update_options(self, **kwargs):
            observer.invalid = True
            _refuse()

    stream = OwnedSpeechStream(stt=adapter, opts=opts, conn_options=APIConnectOptions(max_retry=0, timeout=10))
    directory = Path(__file__).resolve().parent
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            actual = Path(filename).resolve()
            if actual.is_relative_to(directory) and actual not in {Path(__file__).resolve(), path}:
                _refuse()
    return adapter, stream, observer


class _Outbox:
    def __init__(self, fd):
        self.fd = fd
        os.set_blocking(fd, False)
        self.queue = asyncio.Queue(maxsize=32)

    def emit(self, kind, value):
        self.queue.put_nowait(encode(kind, value))

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            packet = await self.queue.get()
            try:
                if packet is None:
                    return
                offset = 0
                while offset < len(packet):
                    try:
                        count = os.write(self.fd, memoryview(packet)[offset:])
                        if count <= 0:
                            _refuse()
                        offset += count
                    except BlockingIOError:
                        ready = loop.create_future()
                        def writable():
                            if not ready.done():
                                ready.set_result(None)
                        loop.add_writer(self.fd, writable)
                        try:
                            await asyncio.wait_for(ready, 2)
                        finally:
                            loop.remove_writer(self.fd)
            finally:
                self.queue.task_done()


async def _serve(fd):
    outbox = _Outbox(fd)
    writer = asyncio.create_task(outbox.run())
    reader = asyncio.StreamReader(limit=MAX_PACKET + 1)
    transport = None
    try:
        outbox.emit(PREFLIGHT, _preflight())
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
        kind, raw = await asyncio.wait_for(receive(reader), 2)
        if kind != CONFIG:
            _refuse()
        config = _validate_config(parse_json(raw))
        raw = None
        adapter, stream, observer = _build_stream(config)
        native_locales = frozenset(config["locales"]) if config["mode"] == "continuous_language_identification" else frozenset()
        config.clear()
        config = None
        from livekit import rtc
        from livekit.agents import stt
        ended = asyncio.Event()
        totals = [0, 0, 0, 0]
        # The last native sample count acknowledged to the parent, and whether
        # STARTED has been sent: acknowledgements follow it.
        pushed = [0, False]

        def acknowledge(force=False):
            samples = observer.written // 2
            if pushed[1] and samples > pushed[0] and (force or samples - pushed[0] >= PUSHED_EVERY):
                pushed[0] = samples
                outbox.emit(PUSHED, {"samples": samples})

        async def feed():
            samples = 0
            while True:
                kind, data = await receive(reader)
                if kind == EOF:
                    if data or await reader.read(1):
                        _refuse()
                    observer.end_input()
                    stream.end_input()
                    acknowledge()
                    ended.set()
                    return
                if kind != AUDIO or not 2 <= len(data) <= 960 or len(data) % 2:
                    _refuse()
                samples += len(data) // 2
                if samples > MAX_SAMPLES:
                    _refuse()
                stream.push_frame(rtc.AudioFrame(data=data, sample_rate=24000, num_channels=1,
                                                 samples_per_channel=len(data) // 2))
                acknowledge()

        async def events():
            await asyncio.wait_for(observer.started.wait(), 10)
            outbox.emit(STARTED, {})
            pushed[1] = True
            acknowledge(force=True)
            async for event in stream:
                acknowledge()
                totals[3] += 1
                if totals[3] > MAX_EVENTS:
                    _refuse()
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    if len(event.alternatives) != 1 or type(event.alternatives[0].text) is not str:
                        _refuse()
                    text = event.alternatives[0].text
                    totals[0] += 1
                    totals[1] += len(text)
                    totals[2] += len(text.encode("utf-8"))
                    if totals[0] > MAX_FINALS or totals[1] > MAX_CHARACTERS or totals[2] > MAX_UTF8:
                        _refuse()
                    # SpeechData wraps strings in the SDK's LanguageCode.
                    # Normalize that string subtype at the SDK boundary; the
                    # wire and parent still accept only plain JSON strings.
                    native_locale = event.alternatives[0].language
                    if not isinstance(native_locale, str):
                        _refuse()
                    locale = str(native_locale) or None
                    if locale is not None and locale not in native_locales:
                        _refuse()
                    outbox.emit(FINAL, {"text": text, "locale": locale})
                elif event.type not in {stt.SpeechEventType.INTERIM_TRANSCRIPT, stt.SpeechEventType.START_OF_SPEECH,
                                       stt.SpeechEventType.END_OF_SPEECH, stt.SpeechEventType.RECOGNITION_USAGE}:
                    observer.invalid = True
            await stream._task
            await stream.aclose()
            if not observer.accepted(stream._task):
                _refuse()

        async def exchange():
            feeding = asyncio.create_task(feed())
            consuming = asyncio.create_task(events())
            eof_waiter = asyncio.create_task(ended.wait())
            try:
                done, _ = await asyncio.wait((feeding, consuming, eof_waiter), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if not ended.is_set():
                    _refuse()
                async with asyncio.timeout(_EOF_SECONDS):
                    await feeding
                    await consuming
            finally:
                for task in (feeding, consuming, eof_waiter):
                    if not task.done():
                        task.cancel()
                # Never await a stuck SDK cleanup here. The process is the owner.
                # Failure exits immediately; the parent reaps it.
            acknowledge(force=True)
            outbox.emit(TERMINAL, {"final_count": totals[0], "utf8_bytes": totals[2]})

        exchange_task = asyncio.create_task(exchange())
        done, _ = await asyncio.wait((exchange_task, writer), return_when=asyncio.FIRST_COMPLETED)
        if writer in done:
            writer.result()
            _refuse()
        exchange_task.result()
        await outbox.queue.put(None)
        await asyncio.wait_for(writer, 2)
        return 0
    except BaseException:
        with contextlib.suppress(Exception):
            outbox.emit(ERROR, {})
            await outbox.queue.put(None)
            await asyncio.wait_for(writer, 2)
        return 1
    finally:
        if transport is not None:
            transport.close()
        os.close(fd)


def main():
    fd = -1
    try:
        if len(sys.argv) != 2 or not sys.argv[1].isdigit():
            _refuse()
        fd = int(sys.argv[1])
        signal.signal(signal.SIGALRM, lambda *_: os._exit(70))
        signal.setitimer(signal.ITIMER_REAL, 340)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_serve(fd))
        if result == 0:
            os._exit(0)
    except BaseException:
        pass
    os._exit(1)


if __name__ == "__main__":
    main()

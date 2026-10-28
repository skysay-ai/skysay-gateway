# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Azure listener. Complete process exit gates every transcript."""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
import re
import sys
import time

try:
    from .azure_speech import azure_region
    from .workflow_input_recognition import FinalCallerTranscript
    from .workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from .workflow_tasks import join_owned_task
    from . import workflow_azure_listener_worker as wire
except ImportError:
    from azure_speech import azure_region
    from workflow_input_recognition import FinalCallerTranscript
    from workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from workflow_tasks import join_owned_task
    import workflow_azure_listener_worker as wire

MAX_QUEUE_FRAMES, MAX_QUEUE_BYTES, MAX_SAMPLES = 1800, 1728000, 7200000
_WALL_SECONDS, _BOOT_SECONDS, _EOF_SECONDS, _TERMINATE_SECONDS = 325, 8, 13, 3
_TAG = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")


class ListenerRefused(WorkflowListenerRefusal):
    """Public fixed error; child and provider payloads never escape."""


def _fail(reason="no_ack"):
    # Inside the exchange every refusal is caught and becomes an unacknowledged
    # stream; the sites a caller can see name their own reason.
    raise ListenerRefused(reason)


def _languages(values):
    return (type(values) is list and 1 <= len(values) <= 32
            and all(type(value) is str and len(value) <= 64 and _TAG.fullmatch(value) for value in values)
            and len(set(values)) == len(values))


def _configuration(value, secret, region):
    result = None
    try:
        common = {"provider", "registry_name", "model", "languages", "mode"}
        single = type(value) is dict and value.get("mode") == "single"
        if (type(value) is not dict or set(value) != common | ({"locale"} if single else
                {"declared_languages", "adapter_options", "max_candidates"})
                or any(type(value.get(k)) is not str for k in ("provider", "registry_name", "model", "mode"))
                or (value["provider"], value["registry_name"], value["model"]) != ("azure", "azure", "realtime")
                or not _languages(value["languages"]) or type(region) is not str or azure_region(region) != region):
            _fail()
        if single:
            if len(value["languages"]) != 1:
                _fail()
            locales = [value["locale"]]
        else:
            options = value["adapter_options"]
            if (value["mode"] != "continuous_language_identification"
                    or not _languages(value["declared_languages"])
                    or not set(value["languages"]) <= set(value["declared_languages"])
                    or type(value["max_candidates"]) is not int or not 2 <= value["max_candidates"] <= 32
                    or not 2 <= len(value["languages"]) <= value["max_candidates"]
                    or type(options) is not dict or set(options) != {"language_id_mode", "candidate_locales"}
                    or options["language_id_mode"] != "continuous" or type(options["language_id_mode"]) is not str
                    or type(options["candidate_locales"]) is not dict
                    or set(options["candidate_locales"]) != set(value["languages"])):
                _fail()
            locales = [options["candidate_locales"][key] for key in value["languages"]]
        result = wire._validate_config(dict(protocol=1, secret=secret, region=region,
                                            mode=value["mode"], locales=locales))
    except Exception:
        pass
    if result is None:
        value = secret = region = None
        _fail("config")
    return result


def _worker_command():
    return sys.executable, "-I", str(Path(__file__).resolve().with_name("workflow_azure_listener_worker.py"))


class AzureWorkflowListener:
    def effective_settings(self):
        """The settings this connection was built with, for attestation: no secret, no client reference."""
        return {key: value for key, value in dict(self._config or {}).items()
                if not any(word in key.lower() for word in ("secret", "key", "token", "credential", "reference"))}

    def __init__(self, *, secret, constructor, client_reference_id, region):
        config = _configuration(constructor, secret, region)
        if (type(client_reference_id) is not str
                or re.fullmatch(r"[A-Za-z0-9:_-]{16,96}", client_reference_id) is None):
            secret = constructor = config = client_reference_id = None
            _fail("config")
        self._secret, self._reference = secret, client_reference_id
        self._config = {key: value for key, value in config.items() if key != "secret"}
        self._created = time.monotonic()
        self._queue = asyncio.Queue()
        self._queued_frames = self._queued_bytes = self._samples = 0
        # Native samples the isolated recognizer acknowledged the Azure SDK push
        # stream accepting (``PUSHED``, at ``wire.NATIVE_RATE``): the measure.
        # The largest acknowledgement is kept across any later failure.
        self._acked_samples = 0
        self._ended = self._eof_sent = self._bad = self._ack = self._closed = self._claimed = False
        self._refusal = None
        self._eof_at = None
        self._runner = self._process = self._transport = self._opened = None
        self._owned_tasks = []
        self._parts = []
        # Evidence only: when the provider's last non-blank final arrived from
        # the recognizer process, before the terminal and exit release it.
        self._final_at = None
        self._characters = self._utf8 = self._finals = self._events = 0
        self._cleanup_error = False
        self._entered = False
        self._eof_timer = None

    def __repr__(self):
        return "<AzureWorkflowListener redacted>"

    def __getstate__(self):
        raise TypeError("workflow listener cannot be copied")

    def _invalidate(self, reason):
        # The first cause wins: a later refusal names what broke this source.
        if self._refusal is None:
            self._refusal = reason
        self._bad, self._ack = True, False
        if self._entered and self._runner is not None and not self._runner.done():
            self._runner.cancel()

    def push_frame(self, frame):
        valid = False
        pcm = view = None
        try:
            samples = frame.samples_per_channel
            valid = (not self._bad and not self._closed and not self._ended
                     and type(samples) is int and 1 <= samples <= 480
                     and type(frame.sample_rate) is int and frame.sample_rate == 24000
                     and type(frame.num_channels) is int and frame.num_channels == 1)
            if valid:
                view = memoryview(frame.data)
                valid = (view.nbytes == samples * 2 and self._queued_frames < MAX_QUEUE_FRAMES
                         and self._queued_bytes + view.nbytes <= MAX_QUEUE_BYTES
                         and self._samples + samples <= MAX_SAMPLES)
                if valid:
                    pcm = view.tobytes()
        except Exception:
            valid = False
        finally:
            if view is not None:
                view.release()
        if not valid:
            self._invalidate("frame")
            _fail(self._refusal)
        self._samples += samples
        self._queued_frames += 1
        self._queued_bytes += len(pcm)
        self._queue.put_nowait(pcm)

    def end_input(self):
        if self._bad or self._closed or self._ended:
            self._invalidate("end")
            _fail(self._refusal)
        self._ended = True
        self._eof_at = time.monotonic()
        self._queue.put_nowait(None)
        self._eof_timer = asyncio.get_running_loop().call_at(
            min(self._created + _WALL_SECONDS, self._eof_at + _EOF_SECONDS), self._invalidate, "no_ack")

    def _deadline(self):
        end = self._created + _WALL_SECONDS
        if self._eof_at is not None:
            end = min(end, self._eof_at + _EOF_SECONDS)
        return end

    async def _bounded(self, awaitable, seconds=None):
        deadline = self._deadline()
        if seconds is not None:
            deadline = min(deadline, time.monotonic() + seconds)
        async with asyncio.timeout_at(deadline):
            return await awaitable

    async def _spawn(self):
        read_fd, write_fd = os.pipe()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *_worker_command(), str(write_fd), pass_fds=(write_fd,), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env={"LANG": "C.UTF-8", "TZ": "UTC"})
            os.close(write_fd)
            write_fd = -1
            self._opened = os.fdopen(read_fd, "rb", buffering=0)
            read_fd = -1
            reader = asyncio.StreamReader(limit=wire.MAX_PACKET + 1)
            self._transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), self._opened)
            return reader
        finally:
            if write_fd >= 0:
                os.close(write_fd)
            if read_fd >= 0:
                os.close(read_fd)

    async def _stop(self):
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), _TERMINATE_SECONDS)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        else:
            await process.wait()

    async def _send(self):
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                self._eof_sent = True
                self._process.stdin.write(wire.encode(wire.EOF, b""))
                await self._bounded(self._process.stdin.drain(), 2)
                self._process.stdin.close()
                return
            try:
                self._process.stdin.write(wire.encode(wire.AUDIO, pcm))
                await self._bounded(self._process.stdin.drain(), 2)
            finally:
                self._queued_frames -= 1
                self._queued_bytes -= len(pcm)

    async def _receive(self, reader, connect_deadline):
        started = terminal = False
        while True:
            deadline = self._deadline() if started else min(self._deadline(), connect_deadline)
            async with asyncio.timeout_at(deadline):
                kind, raw = await wire.receive(reader)
            self._events += 1
            if self._events > wire.MAX_EVENTS:
                _fail()
            value = wire.parse_json(raw)
            if kind == wire.STARTED and not started and value == {}:
                started = True
            elif (kind == wire.PUSHED and started and set(value) == {"samples"} and type(value["samples"]) is int
                    and self._acked_samples < value["samples"]
                    and value["samples"] * 24000 <= (self._samples + 480) * wire.NATIVE_RATE):
                self._acked_samples = value["samples"]
            elif kind == wire.FINAL and started and set(value) == {"text", "locale"} and type(value["text"]) is str:
                text, locale = value["text"], value["locale"]
                if locale is not None and (type(locale) is not str
                        or self._config["mode"] != "continuous_language_identification"
                        or locale not in self._config["locales"]):
                    _fail()
                self._characters += len(text)
                self._utf8 += len(text.encode("utf-8"))
                self._finals += 1
                if (self._characters > wire.MAX_CHARACTERS or self._utf8 > wire.MAX_UTF8
                        or self._finals > wire.MAX_FINALS):
                    _fail()
                self._parts.append((text, locale))
                if text.strip():
                    self._final_at = time.monotonic()
            elif kind == wire.TERMINAL and started and self._eof_sent:
                if (set(value) != {"final_count", "utf8_bytes"}
                        or type(value["final_count"]) is not int or type(value["utf8_bytes"]) is not int
                        or value["final_count"] != self._finals or value["utf8_bytes"] != self._utf8):
                    _fail()
                terminal = True
                if await self._bounded(reader.read(1)):
                    _fail()
                break
            else:
                _fail()
        if not terminal or await self._bounded(self._process.wait()) != 0:
            _fail()

    async def _run(self):
        self._entered = True
        spawn = None
        tasks = []
        good = False
        try:
            if self._bad:
                _fail()
            spawn = asyncio.create_task(self._spawn(), name="workflow-azure-spawn")
            self._owned_tasks.append(spawn)
            boot_deadline = time.monotonic() + _BOOT_SECONDS
            async with asyncio.timeout_at(min(self._deadline(), boot_deadline)):
                reader = await asyncio.shield(spawn)
                kind, raw = await wire.receive(reader)
            value = wire.parse_json(raw)
            if (kind != wire.PREFLIGHT or set(value) != {"clean", "environment_names"}
                    or value["clean"] is not True or type(value["environment_names"]) is not list
                    or any(type(name) is not str for name in value["environment_names"])
                    or not set(value["environment_names"]) <= {"LANG", "TZ"}):
                _fail()
            self._process.stdin.write(wire.encode(wire.CONFIG, {**self._config, "secret": self._secret}))
            await self._bounded(self._process.stdin.drain(), 2)
            self._secret = None
            connect_deadline = time.monotonic() + 11
            tasks = [asyncio.create_task(self._send(), name="workflow-azure-input"),
                     asyncio.create_task(self._receive(reader, connect_deadline), name="workflow-azure-output")]
            self._owned_tasks.extend(tasks)
            await self._bounded(asyncio.gather(*tasks))
            good = not self._bad
        except (Exception, asyncio.CancelledError):
            # The recognizer failing is the first cause: a frame refused while
            # its child is still being reaped reports it, not "frame".
            if self._refusal is None:
                self._refusal = "no_ack"
            self._bad = True
        finally:
            self._secret = None
            # A spawn cannot be abandoned even when cancellation wins its await.
            if spawn is not None:
                try:
                    await join_owned_task(spawn)
                except asyncio.CancelledError:
                    self._bad = True
                except Exception:
                    self._bad = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                try:
                    await join_owned_task(task)
                except (Exception, asyncio.CancelledError):
                    self._bad = True
            cleanup = asyncio.create_task(self._stop(), name="workflow-azure-reap")
            self._owned_tasks.append(cleanup)
            try:
                await join_owned_task(cleanup)
            except asyncio.CancelledError:
                self._bad = True
            except Exception:
                self._bad = self._cleanup_error = True
            if self._transport is not None:
                self._transport.close()
            elif self._opened is not None:
                self._opened.close()
            self._transport = self._opened = None
            self._reference = self._config = None
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queued_frames = self._queued_bytes = 0
            if self._eof_timer is not None:
                self._eof_timer.cancel()
            self._closed = True
            self._ack = good and not self._bad and not self._cleanup_error
            if not self._ack:
                self._parts.clear()
                if self._refusal is None:
                    self._refusal = "no_ack"

    async def transcripts(self):
        if self._claimed:
            self._invalidate("reclaimed")
            await self.aclose()
            _fail(self._refusal)
        self._claimed = True
        if time.monotonic() > self._created + _WALL_SECONDS:
            self._invalidate("wall")
        if self._runner is None and not self._closed and not self._bad:
            self._runner = asyncio.create_task(self._run(), name="workflow-azure-runner")
            self._owned_tasks.append(self._runner)
        if self._runner is not None:
            try:
                await asyncio.shield(self._runner)
            except asyncio.CancelledError:
                await self.aclose()
                raise
        if not self._ack:
            await self.aclose()
            _fail(self._refusal or "no_ack")
        for text, locale in self._parts:
            if text.strip():
                yield FinalCallerTranscript(text, locale, None)

    async def wait_terminal(self):
        if self._runner is not None:
            await asyncio.shield(self._runner)
        return self._ack

    def provider_final_at(self):
        """When the provider's last non-blank final reached this worker (monotonic), or None."""
        return self._final_at

    def observation(self):
        return SourceObservation("connect_unknown", (("audio_second", 0),), (), "", self._ack,
                                 measured_seconds=self._acked_samples / wire.NATIVE_RATE)

    def _clear_unentered(self):
        self._closed = True
        if self._eof_timer is not None:
            self._eof_timer.cancel()
        self._secret = self._reference = self._config = None
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queued_frames = self._queued_bytes = 0
        # Retire an externally cancelled, never-entered runner only after join.
        # Its completed task remains in the ownership ledger.
        self._runner = None

    async def aclose(self):
        if self._runner is not None:
            if not self._runner.done():
                self._invalidate("no_ack")
            try:
                await join_owned_task(self._runner)
            finally:
                if not self._entered:
                    self._clear_unentered()
        else:
            self._clear_unentered()
        if self._cleanup_error:
            _fail(self._refusal or "no_ack")

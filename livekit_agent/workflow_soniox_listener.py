# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Soniox transport. Observations deliberately grant no usage truth."""
from __future__ import annotations

import asyncio
import json
import math
import re
import time

import aiohttp

try:
    from .workflow_input_recognition import FinalCallerTranscript
    from .workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from .workflow_tasks import join_owned_task
except ImportError:
    from workflow_input_recognition import FinalCallerTranscript
    from workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from workflow_tasks import join_owned_task


ENDPOINT = "wss://stt-rt.soniox.com/transcribe-websocket"
MAX_QUEUE_FRAMES = 1800
MAX_QUEUE_BYTES = 1728000
MAX_SAMPLES = 7200000
MAX_MESSAGE_BYTES = 262144
MAX_FINALS = MAX_CHARACTERS = 16000
MAX_UTF8_BYTES = 64000
CONNECT_SECONDS, WALL_SECONDS = 10, 325
FINISHED_SECONDS, PEER_CLOSE_SECONDS, CLOSE_SECONDS = 13, 15, 2
DIMENSIONS = ("input_audio_tokens", "output_text_tokens")
_now = time.monotonic
_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_TOP_KEYS = {"tokens", "final_audio_proc_ms", "total_audio_proc_ms", "finished"}
_TOKEN_KEYS = {"text", "start_ms", "end_ms", "confidence", "is_final", "speaker", "language"}


class ListenerRefused(WorkflowListenerRefusal):
    """Fixed public error; provider exceptions and payloads never escape."""


def _languages(value):
    return (type(value) is list and 1 <= len(value) <= 32
            and all(type(tag) is str and len(tag) <= 64 and _LANGUAGE.fullmatch(tag) for tag in value)
            and len(set(value)) == len(value))


def _config(value):
    if type(value) is not dict:
        return None
    common = {"provider", "registry_name", "model", "languages", "mode"}
    single = value.get("mode") == "single"
    required = common | ({"locale"} if single else {"declared_languages", "max_candidates", "adapter_options"})
    if (set(value) != required or any(type(value.get(key)) is not str for key in ("provider", "registry_name", "mode"))
            or value.get("provider") != "soniox" or value.get("registry_name") != "soniox"
            or type(value.get("model")) is not str
            or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value["model"])
            or not _languages(value.get("languages"))):
        return None
    if single:
        if type(value["locale"]) is not str or value["locale"] not in value["languages"]:
            return None
    elif (value.get("mode") != "language_identification" or not _languages(value["declared_languages"])
            or not set(value["languages"]) <= set(value["declared_languages"])
            or value["max_candidates"] is not None or type(value["adapter_options"]) is not dict
            or set(value["adapter_options"]) != {"enable_language_identification"}
            or value["adapter_options"]["enable_language_identification"] is not True):
        return None
    result = dict(model=value["model"], audio_format="pcm_s16le", sample_rate=24000, num_channels=1,
                  language_hints=list(value["languages"]))
    if not single:
        result["enable_language_identification"] = True
    return result


def _number(value):
    if len(value) > 32:
        raise ValueError("invalid numeric width")
    result = float(value) if any(char in value for char in ".eE") else int(value)
    if not math.isfinite(result):
        raise ValueError("invalid numeric value")
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("invalid numeric constant")


def _decode(raw):
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("invalid message size")
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 32:
                raise ValueError("invalid message depth")
        elif char in "]}":
            depth -= 1
    value = json.loads(raw, object_pairs_hook=_pairs, parse_int=_number, parse_float=_number, parse_constant=_constant)
    if type(value) is not dict or not set(value) <= _TOP_KEYS:
        raise ValueError("invalid message fields")
    for field in ("final_audio_proc_ms", "total_audio_proc_ms"):
        if field in value and not _bounded_number(value[field], 325000):
            raise ValueError("invalid processing metadata")
    if "finished" in value and value["finished"] is not True:
        raise ValueError("invalid completion")
    if "tokens" in value and (type(value["tokens"]) is not list or len(value["tokens"]) > MAX_FINALS):
        raise ValueError("invalid token collection")
    return value


def _bounded_number(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum


async def _stop_redirect(*_):
    # Raised inside the connect, which ``_run`` reports as never acknowledged.
    raise ListenerRefused("no_ack")


class SonioxWorkflowListener:
    def effective_settings(self):
        """The settings this connection was built with, for attestation: no secret, no client reference."""
        return {key: value for key, value in dict(self._config or {}).items()
                if not any(word in key.lower() for word in ("secret", "key", "token", "credential", "reference"))}

    def __init__(self, *, secret, constructor, client_reference_id):
        config = _config(constructor)
        valid = (config is not None and type(secret) is str and 1 <= len(secret) <= 4096
                 and all(33 <= ord(char) <= 126 for char in secret)
                 and type(client_reference_id) is str
                 and re.fullmatch(r"[A-Za-z0-9:_-]{16,256}", client_reference_id))
        if not valid:
            secret = client_reference_id = constructor = config = None
            raise ListenerRefused("config")
        self._secret, self._reference, self._config = secret, client_reference_id, config
        self._languages = frozenset(config["language_hints"])
        self._created = _now()
        self._queue = asyncio.Queue()
        self._queued_frames = self._queued_bytes = self._samples = 0
        # PCM samples written to the open provider connection: the measure.
        self._sent_samples = 0
        self._ended = self._eof_sent = self._bad = self._closed = self._ack = False
        self._refusal = None
        self._eof_at = None
        self._runner = self._receiver = self._socket = self._session = None
        self._owned_tasks = []
        self._receive_timer = self._run_timer = None
        self._cancelled_runner = False
        self._started = False
        self._transcript_claimed = False
        self._parts, self._hints = [], set()
        # Evidence only: when the provider's last non-blank final arrived.
        self._final_at = None
        self._events = self._characters = self._utf8_bytes = 0

    def __repr__(self):
        return "<SonioxWorkflowListener redacted>"

    def __getstate__(self):
        raise TypeError("workflow listener cannot be copied")

    def _cancel(self):
        if (self._runner is not None and self._started and not self._runner.done()
                and not self._cancelled_runner):
            self._cancelled_runner = True
            self._runner.cancel()

    def _invalidate(self, reason):
        # The first cause wins: a later refusal names what broke this source.
        if self._refusal is None:
            self._refusal = reason
        self._bad, self._ack = True, False
        self._cancel()

    def push_frame(self, frame):
        pcm = view = None
        valid = False
        try:
            samples = frame.samples_per_channel
            valid = (not self._bad and not self._closed and not self._ended
                     and type(samples) is int and 1 <= samples <= 480
                     and type(frame.sample_rate) is int and frame.sample_rate == 24000
                     and type(frame.num_channels) is int and frame.num_channels == 1)
            if valid:
                view = memoryview(frame.data)
                valid = (view.nbytes == samples * 2 and self._queued_frames + 1 <= MAX_QUEUE_FRAMES
                         and self._queued_bytes + view.nbytes <= MAX_QUEUE_BYTES
                         and self._samples + samples <= MAX_SAMPLES and _now() <= self._created + WALL_SECONDS)
                if valid:
                    pcm = bytes(view)
        except Exception:
            valid = False
        if not valid:
            self._invalidate("wall" if _now() > self._created + WALL_SECONDS else "frame")
            frame = pcm = view = None
            raise ListenerRefused(self._refusal)
        self._queued_frames += 1
        self._queued_bytes += len(pcm)
        self._samples += samples
        self._queue.put_nowait(pcm)

    def end_input(self):
        if self._ended or self._closed or self._bad or _now() > self._created + WALL_SECONDS:
            self._invalidate("wall" if _now() > self._created + WALL_SECONDS else "end")
            raise ListenerRefused(self._refusal)
        self._ended, self._eof_at = True, _now()
        # The EOF sentinel has a reserved control slot and consumes no PCM budget.
        self._queue.put_nowait(None)
        if self._receive_timer is not None:
            self._receive_timer.reschedule(self._deadline(False))
        if self._run_timer is not None:
            self._run_timer.reschedule(self._deadline(False))

    def _deadline(self, finished):
        duration = PEER_CLOSE_SECONDS if finished else FINISHED_SECONDS
        absolute = min(self._created + WALL_SECONDS,
                       self._eof_at + duration if self._eof_at is not None else float("inf"))
        return asyncio.get_running_loop().time() + max(0, absolute - _now())

    def _append(self, value):
        if (type(value) is not dict or not set(value) <= _TOKEN_KEYS
                or type(value.get("text")) is not str or type(value.get("is_final")) is not bool):
            raise ValueError("invalid token")
        text = value["text"]
        encoded = text.encode("utf-8")
        if len(text) > MAX_CHARACTERS or len(encoded) > MAX_UTF8_BYTES or "<end>" in text or "<fin>" in text:
            raise ValueError("invalid token text")
        for field in ("start_ms", "end_ms", "confidence"):
            if field in value and not _bounded_number(value[field], 1 if field == "confidence" else 325000):
                raise ValueError("invalid token metadata")
        for field in ("speaker", "language"):
            if field in value and (type(value[field]) is not str or not 1 <= len(value[field]) <= 64
                                   or not all(33 <= ord(char) <= 126 for char in value[field])):
                raise ValueError("invalid token label")
        self._events += 1
        if self._events > MAX_FINALS:
            raise ValueError("token event capacity")
        if not value["is_final"]:
            return
        self._characters += len(text)
        self._utf8_bytes += len(encoded)
        if self._characters > MAX_CHARACTERS or self._utf8_bytes > MAX_UTF8_BYTES:
            raise ValueError("final text capacity")
        self._parts.append(text)
        if text.strip():
            self._final_at = _now()
            language = value.get("language")
            self._hints.add(language if language in self._languages else None)

    async def _receive(self):
        finished = False
        try:
            async with asyncio.timeout_at(self._deadline(False)) as timer:
                self._receive_timer = timer
                while True:
                    message = await self._socket.receive()
                    now = _now()
                    if (now > self._created + WALL_SECONDS
                            or (self._eof_at is not None and now > self._eof_at +
                                (PEER_CLOSE_SECONDS if finished else FINISHED_SECONDS))):
                        return False
                    if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                        return finished and self._socket.close_code == 1000
                    if message.type != aiohttp.WSMsgType.TEXT or finished:
                        return False
                    value = _decode(message.data)
                    for item in value.get("tokens", ()):
                        self._append(item)
                    if value.get("finished") is True:
                        if not self._eof_sent:
                            return False
                        finished = True
                        timer.reschedule(self._deadline(True))
                        if self._run_timer is not None:
                            self._run_timer.reschedule(self._deadline(True))
        except (Exception, asyncio.CancelledError):
            return False
        finally:
            self._receive_timer = None

    def _receiver_done(self, task):
        if task.cancelled() or task.exception() is not None or task.result() is not True:
            self._invalidate("no_ack")

    async def _run(self):
        self._started = True
        success = False
        try:
            if self._bad:
                return False
            async with asyncio.timeout_at(self._deadline(False)) as timer:
                self._run_timer = timer
                success = await self._exchange()
        except (Exception, asyncio.CancelledError):
            success = False
        finally:
            # Receiver callbacks must never cancel the runner during its join.
            self._cancelled_runner = True
            self._run_timer = None
            if self._receiver is not None:
                if not self._receiver.done():
                    self._receiver.cancel()
                await asyncio.gather(self._receiver, return_exceptions=True)
            try:
                async with asyncio.timeout(CLOSE_SECONDS):
                    if self._socket is not None:
                        await self._socket.close()
            except (Exception, asyncio.CancelledError):
                success = False
            try:
                if self._session is not None:
                    await self._session.close()
            except (Exception, asyncio.CancelledError):
                success = False
            self._secret = self._reference = self._config = self._socket = self._session = None
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queued_frames = self._queued_bytes = 0
            self._closed = True
            self._ack = success and not self._bad
            if not self._ack and self._refusal is None:
                self._refusal = "no_ack"
        return self._ack

    async def _exchange(self):
        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(_stop_redirect)
        self._session = aiohttp.ClientSession(trust_env=False, trace_configs=[trace])
        async with asyncio.timeout_at(min(self._deadline(False),
                                         asyncio.get_running_loop().time() + CONNECT_SECONDS)):
            self._socket = await self._session.ws_connect(ENDPOINT,
                headers={"Authorization": "Bearer " + self._secret}, max_msg_size=MAX_MESSAGE_BYTES)
        config = {**self._config, "client_reference_id": self._reference}
        await self._socket.send_json(config)
        config = None
        self._receiver = asyncio.create_task(self._receive(), name="workflow-soniox-receiver")
        self._owned_tasks.append(self._receiver)
        self._receiver.add_done_callback(self._receiver_done)
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                if _now() > self._eof_at + FINISHED_SECONDS:
                    return False
                await self._socket.send_str("")
                self._eof_sent = True
                return await self._receiver
            try:
                await self._socket.send_bytes(pcm)
                self._sent_samples += len(pcm) // 2
            finally:
                self._queued_frames -= 1
                self._queued_bytes -= len(pcm)
                pcm = None


    async def transcripts(self):
        if self._transcript_claimed:
            self._invalidate("reclaimed")
            await self.aclose()
            raise ListenerRefused(self._refusal)
        self._transcript_claimed = True
        if _now() > self._created + WALL_SECONDS:
            self._invalidate("wall")
        if self._runner is None and not self._closed and not self._bad:
            self._runner = asyncio.create_task(self._run(), name="workflow-soniox-runner")
            self._owned_tasks.append(self._runner)
        if self._runner is not None:
            try:
                await asyncio.shield(self._runner)
            except asyncio.CancelledError:
                await self.aclose()
                raise
        if not self._ack:
            await self.aclose()
            raise ListenerRefused(self._refusal or "no_ack")
        text = "".join(self._parts).strip()
        if text:
            hint = next(iter(self._hints)) if len(self._hints) == 1 else None
            yield FinalCallerTranscript(text, None, (hint,) if hint is not None else None)

    async def wait_terminal(self):
        if self._runner is not None:
            await asyncio.shield(self._runner)
        return self._ack

    def provider_final_at(self):
        """When the provider's last non-blank final reached this worker (monotonic), or None."""
        return self._final_at

    def observation(self):
        # Zeros are mandatory unknown placeholders, never a known-zero invoice.
        return SourceObservation("connect_unknown", tuple((key, 0) for key in DIMENSIONS), (), "", self._ack,
                                 measured_seconds=self._sent_samples / 24000)

    async def aclose(self):
        if self._runner is not None:
            if not self._runner.done():
                self._bad = True
                self._cancel()
            await join_owned_task(self._runner)
        else:
            self._closed = True
            self._secret = self._reference = self._config = None
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queued_frames = self._queued_bytes = 0

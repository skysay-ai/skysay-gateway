# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Deepgram transport. Observations deliberately grant no usage truth.

The provider websocket is driven directly, like the Soniox listener. The pinned
plugin is not wrapped: its stream reconnects after an unexpected close, logs and
skips a message it cannot parse, and cannot refuse an unknown field, while this
source owns exactly one session and refuses anything outside its schema.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from urllib.parse import urlencode

import aiohttp

try:
    from .workflow_input_recognition import FinalCallerTranscript
    from .workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from .workflow_tasks import join_owned_task
except ImportError:
    from workflow_input_recognition import FinalCallerTranscript
    from workflow_source_owner import SourceObservation, WorkflowListenerRefusal
    from workflow_tasks import join_owned_task


ENDPOINT = "wss://api.deepgram.com/v1/listen"
MAX_QUEUE_FRAMES = 1800
MAX_QUEUE_BYTES = 1728000
MAX_SAMPLES = 7200000
MAX_MESSAGE_BYTES = 262144
MAX_EVENTS = MAX_FINALS = MAX_CHARACTERS = 16000
MAX_UTF8_BYTES = 64000
MAX_WORDS = 4096
CONNECT_SECONDS, WALL_SECONDS = 10, 325
FINISHED_SECONDS, PEER_CLOSE_SECONDS, CLOSE_SECONDS = 13, 15, 2
# The provider closes a stream that receives nothing for about ten seconds.
KEEPALIVE_SECONDS = 5
KEEPALIVE = '{"type": "KeepAlive"}'
CLOSE_STREAM = '{"type": "CloseStream"}'
DIMENSIONS = ("audio_second",)
_now = time.monotonic
_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_MESSAGE_KEYS = {
    "Results": ({"type", "channel", "is_final", "speech_final"},
                {"type", "channel", "is_final", "speech_final", "from_finalize", "channel_index",
                 "duration", "start", "metadata"}),
    "SpeechStarted": ({"type"}, {"type", "channel", "timestamp"}),
    "Metadata": ({"type"}, {"type", "transaction_key", "request_id", "sha256", "created", "duration",
                            "channels", "models", "model_info"}),
}
_ALTERNATIVE_KEYS = {"transcript", "confidence", "words", "languages"}
_WORD_KEYS = {"word", "punctuated_word", "start", "end", "confidence", "language", "speaker",
              "speaker_confidence"}
_RESULT_METADATA_KEYS = {"request_id", "model_info", "model_uuid"}
_MODEL_KEYS = {"name", "version", "arch"}


class ListenerRefused(WorkflowListenerRefusal):
    """Fixed public error; provider exceptions and payloads never escape."""


#: The provider's declared inventory a multilingual constructor carries (audit data, not what the call listens for):
#: bounded like a registry row, not like a selection.
_DECLARED_LIMIT = 48


def _languages(value, limit=32):
    return (type(value) is list and 1 <= len(value) <= limit
            and all(type(tag) is str and len(tag) <= 64 and _LANGUAGE.fullmatch(tag) for tag in value)
            and len(set(value)) == len(value))


def _config(value):
    """The released Cascade construction; only rate and interim delivery differ."""
    if type(value) is not dict:
        return None
    common = {"provider", "registry_name", "model", "languages", "mode"}
    single = value.get("mode") == "single"
    # A multi-language constructor carries the provider set it was admitted
    # against (``declared_languages``) or, for an open declaration, the scope
    # ``any`` in its place: exactly one, as the registry emits it.
    open_scope = not single and "language_scope" in value
    inventory = {"language_scope"} if open_scope else {"declared_languages"}
    required = common | ({"locale"} if single else inventory | {"adapter_options", "max_candidates"})
    if (set(value) != required or any(type(value.get(key)) is not str for key in common - {"languages"})
            or (value["provider"], value["registry_name"], value["model"]) != ("deepgram", "deepgram", "nova-3")
            or not _languages(value["languages"])):
        return None
    if single:
        if (len(value["languages"]) != 1 or type(value["locale"]) is not str
                or len(value["locale"]) > 64 or not _LANGUAGE.fullmatch(value["locale"])):
            return None
        language = value["locale"]
    else:
        options = value["adapter_options"]
        if (value["mode"] != "language_multi" or len(value["languages"]) < 2
                or (value["language_scope"] != "any" if open_scope
                    else not _languages(value["declared_languages"], _DECLARED_LIMIT)
                    or not set(value["languages"]) <= set(value["declared_languages"]))
                or value["max_candidates"] is not None or type(options) is not dict
                or set(options) != {"language"} or type(options["language"]) is not str
                or options["language"] != "multi"):
            return None
        language = "multi"
    return {"model": "nova-3", "punctuate": "true", "smart_format": "false", "no_delay": "true",
            "interim_results": "false", "encoding": "linear16", "vad_events": "true",
            "sample_rate": "24000", "channels": "1", "endpointing": "25", "filler_words": "true",
            "profanity_filter": "false", "numerals": "false", "mip_opt_out": "true", "language": language}


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


def _bounded_number(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum


def _label(value, minimum=1):
    return (type(value) is str and minimum <= len(value) <= 256
            and all(32 <= ord(char) <= 126 for char in value))


def _require(condition):
    if not condition:
        raise ValueError("invalid provider message")


def _optional(value, key, check):
    if key in value:
        _require(check(value[key]))


def _channel_index(value):
    return (type(value) is list and len(value) <= 2
            and all(type(item) is int and 0 <= item <= 1 for item in value))


def _seconds(value):
    return _bounded_number(value, WALL_SECONDS)


def _confidence(value):
    return _bounded_number(value, 1)


def _models(value):
    return (type(value) is dict and len(value) <= 8 and all(_label(key) for key in value)
            and all(type(item) is dict and set(item) <= _MODEL_KEYS
                    and all(_label(field, 0) for field in item.values()) for item in value.values()))


def _model(value):
    return type(value) is dict and set(value) <= _MODEL_KEYS and all(_label(item, 0) for item in value.values())


def _words(value):
    if type(value) is not list or len(value) > MAX_WORDS:
        return False
    for word in value:
        if (type(word) is not dict or not set(word) <= _WORD_KEYS
                or type(word.get("word")) is not str or len(word["word"]) > MAX_CHARACTERS):
            return False
        for key, check in (("punctuated_word", lambda item: type(item) is str and len(item) <= MAX_CHARACTERS),
                           ("start", _seconds), ("end", _seconds), ("confidence", _confidence),
                           ("speaker_confidence", _confidence), ("language", _label),
                           ("speaker", lambda item: type(item) is int and 0 <= item <= 64)):
            if key in word and not check(word[key]):
                return False
    return True


def _results(value):
    _require(type(value["is_final"]) is bool and type(value["speech_final"]) is bool)
    # A provider endpoint is a final by definition; the reverse is not.
    _require(value["is_final"] or not value["speech_final"])
    _optional(value, "from_finalize", lambda item: type(item) is bool)
    _optional(value, "channel_index", _channel_index)
    _optional(value, "duration", _seconds)
    _optional(value, "start", _seconds)
    _optional(value, "metadata", lambda item: type(item) is dict and set(item) <= _RESULT_METADATA_KEYS
              and all(_label(item[key]) for key in ("request_id", "model_uuid") if key in item)
              and ("model_info" not in item or _model(item["model_info"])))
    channel = value["channel"]
    _require(type(channel) is dict and set(channel) == {"alternatives"}
             and type(channel["alternatives"]) is list and len(channel["alternatives"]) == 1)
    alternative = channel["alternatives"][0]
    _require(type(alternative) is dict and "transcript" in alternative and set(alternative) <= _ALTERNATIVE_KEYS
             and type(alternative["transcript"]) is str)
    _optional(alternative, "confidence", _confidence)
    _optional(alternative, "words", _words)
    _optional(alternative, "languages", lambda item: type(item) is list and len(item) <= 32
              and all(_label(tag) for tag in item))
    return alternative["transcript"]


def _metadata(value):
    for key in ("transaction_key", "request_id", "sha256", "created"):
        _optional(value, key, _label)
    _optional(value, "duration", _seconds)
    _optional(value, "channels", lambda item: type(item) is int and 1 <= item <= 2)
    _optional(value, "models", lambda item: type(item) is list and len(item) <= 8 and all(_label(x) for x in item))
    _optional(value, "model_info", _models)


def _decode(raw):
    """One provider message as (kind, final text or None); anything else refuses."""
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
    _require(type(value) is dict and type(value.get("type")) is str and value["type"] in _MESSAGE_KEYS)
    kind = value["type"]
    required, allowed = _MESSAGE_KEYS[kind]
    _require(required <= set(value) <= allowed)
    if kind == "Results":
        text = _results(value)
        return kind, text if value["is_final"] else None
    if kind == "SpeechStarted":
        _optional(value, "channel", _channel_index)
        _optional(value, "timestamp", _seconds)
    else:
        _metadata(value)
    return kind, None


async def _stop_redirect(*_):
    # Raised inside the connect, which ``_run`` reports as never acknowledged.
    raise ListenerRefused("no_ack")


class DeepgramWorkflowListener:
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
        # The released path sends no request tag; the reference is held, never sent.
        self._secret, self._reference, self._config = secret, client_reference_id, config
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
        self._parts = []
        self._events = self._finals = self._characters = self._utf8_bytes = self._keepalives = 0
        # Evidence only: when the provider's last non-blank final arrived.
        self._final_at = None

    def __repr__(self):
        return "<DeepgramWorkflowListener redacted>"

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
        finally:
            if view is not None:
                view.release()
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

    def _append(self, text):
        encoded = text.encode("utf-8")
        self._finals += 1
        self._characters += len(text)
        self._utf8_bytes += len(encoded)
        if (self._finals > MAX_FINALS or self._characters > MAX_CHARACTERS
                or self._utf8_bytes > MAX_UTF8_BYTES):
            raise ValueError("final text capacity")
        self._parts.append(text)
        if text.strip():
            self._final_at = _now()

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
                        # The provider's own stream summary (Metadata after our
                        # CloseStream) is the terminal marker. The transport close
                        # that follows carries no information: the live service
                        # drops the connection without a close frame (1006) after
                        # it, as the first functional run showed, and a close code
                        # before the summary is still a refusal.
                        return finished
                    if message.type != aiohttp.WSMsgType.TEXT or finished:
                        return False
                    kind, text = _decode(message.data)
                    self._events += 1
                    if self._events > MAX_EVENTS:
                        return False
                    if text is not None:
                        self._append(text)
                    elif kind == "Metadata":
                        # The provider's stream summary follows CloseStream and
                        # every flushed final; before it, the stream is not done.
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
            if not self._ack:
                self._parts.clear()
                if self._refusal is None:
                    self._refusal = "no_ack"
        return self._ack

    async def _exchange(self):
        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(_stop_redirect)
        self._session = aiohttp.ClientSession(trust_env=False, trace_configs=[trace])
        async with asyncio.timeout_at(min(self._deadline(False),
                                         asyncio.get_running_loop().time() + CONNECT_SECONDS)):
            self._socket = await self._session.ws_connect(ENDPOINT + "?" + urlencode(self._config),
                headers={"Authorization": "Token " + self._secret}, max_msg_size=MAX_MESSAGE_BYTES)
        self._secret = None
        self._receiver = asyncio.create_task(self._receive(), name="workflow-deepgram-receiver")
        self._owned_tasks.append(self._receiver)
        self._receiver.add_done_callback(self._receiver_done)
        while True:
            try:
                async with asyncio.timeout(KEEPALIVE_SECONDS):
                    pcm = await self._queue.get()
            except TimeoutError:
                await self._socket.send_str(KEEPALIVE)
                self._keepalives += 1
                continue
            if pcm is None:
                if _now() > self._eof_at + FINISHED_SECONDS:
                    return False
                # Set before the write: the provider may answer before it returns.
                self._eof_sent = True
                await self._socket.send_str(CLOSE_STREAM)
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
            self._runner = asyncio.create_task(self._run(), name="workflow-deepgram-runner")
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
        # A single-language stream and the provider's multi keyword both carry
        # no language evidence this source may report; no hint is invented.
        for text in self._parts:
            if text.strip():
                yield FinalCallerTranscript(text.strip(), None, None)

    async def wait_terminal(self):
        if self._runner is not None:
            await asyncio.shield(self._runner)
        return self._ack

    def provider_final_at(self):
        """When the provider's last non-blank final reached this worker (monotonic), or None."""
        return self._final_at

    def observation(self):
        # Zeros are mandatory unknown placeholders, never a known-zero invoice.
        # The stream does not confirm billed seconds, so no exact claim exists.
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


def build_deepgram_listener(*, secret, constructor, client_reference_id):
    return DeepgramWorkflowListener(secret=secret, constructor=constructor, client_reference_id=client_reference_id)

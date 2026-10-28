# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Finite OpenAI gpt-live-transcribe transport over the realtime API.

The provider websocket is driven directly, like the Deepgram listener. The
pinned plugin is not wrapped: its stream runs a VAD that commits on end of
speech but never at end of input, reconnects through a socket pool, logs and
skips a message it cannot parse, and ignores unknown events, while this
source owns exactly one session, commits exactly once when its input ends and
refuses anything outside its schema.

The turn ends on a real provider marker: this source's own
``input_audio_buffer.commit`` -> ``input_audio_buffer.committed{item_id}`` ->
``conversation.item.input_audio_transcription.completed`` for that item ->
this source's own close (completion never ends the session).
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import time

import aiohttp

try:
    from .workflow_input_recognition import FinalCallerTranscript
    from .workflow_source_owner import SourceMetric, SourceObservation, WorkflowListenerRefusal
    from .workflow_tasks import join_owned_task
except ImportError:
    from workflow_input_recognition import FinalCallerTranscript
    from workflow_source_owner import SourceMetric, SourceObservation, WorkflowListenerRefusal
    from workflow_tasks import join_owned_task


# The plugin's own URL for api.openai.com: no ``model`` query, which would
# select a conversation session (plugins/openai/stt.py:571-588).
ENDPOINT = "wss://api.openai.com/v1/realtime?intent=transcription"
MODEL = "gpt-live-transcribe"
MAX_QUEUE_FRAMES = 1800
MAX_QUEUE_BYTES = 1728000
MAX_SAMPLES = 7200000
MAX_MESSAGE_BYTES = 262144
MAX_EVENTS = MAX_FINALS = MAX_CHARACTERS = 16000
MAX_UTF8_BYTES = 64000
# A wire count above this is not a plausible single 300-second turn.
MAX_USAGE_TOKENS = 10000000
CONNECT_SECONDS, WALL_SECONDS = 10, 325
FINISHED_SECONDS, CLOSE_SECONDS = 13, 2
# No idle message exists for a realtime transcription session; the plugin
# sends none either, and a 325-second source is far inside the session limit
# the plugin restarts before (plugins/openai/stt.py:77-79).
COMMIT = '{"type": "input_audio_buffer.commit"}'
DIMENSIONS = ("audio_second",)
_now = time.monotonic
# The plugin sends primary ISO-639-1 codes for this model (plugins/openai/stt.py:153-159).
_ISO639 = re.compile(r"[a-z]{2}")
_LOCALE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
COMMITTED = "input_audio_buffer.committed"
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
# Events a transcription session emits whose content this source never uses.
# Anything else (error, transcription failure, server-VAD, response, cleared
# buffer, an unknown type) refuses. Only ``type`` and the size/depth bounds
# are checked here: these are configuration echoes and item bookkeeping.
_HARMLESS = frozenset({"session.created", "session.updated", "conversation.item.created",
                       "conversation.item.added", "conversation.item.done", "rate_limits.updated"})
_KEYS = {
    COMMITTED: ({"type", "event_id", "item_id"}, {"type", "event_id", "item_id", "previous_item_id"}),
    DELTA: ({"type", "event_id", "item_id"}, {"type", "event_id", "item_id", "content_index", "delta", "logprobs", "obfuscation"}),
    COMPLETED: ({"type", "event_id", "item_id", "content_index", "transcript", "usage"},
                {"type", "event_id", "item_id", "content_index", "transcript", "usage", "languages", "logprobs"}),
}
_TOKEN_USAGE_KEYS = {"type", "input_tokens", "output_tokens", "total_tokens"}


class ListenerRefused(WorkflowListenerRefusal):
    """Fixed public error; provider exceptions and payloads never escape."""


def _languages(value, grammar):
    return (type(value) is list and 1 <= len(value) <= 32
            and all(type(tag) is str and len(tag) <= 64 and grammar.fullmatch(tag) for tag in value)
            and len(set(value)) == len(value))


def _config(value):
    """The plugin's session configuration for this model; only the documented differences."""
    if type(value) is not dict:
        return None
    common = {"provider", "registry_name", "model", "languages", "mode"}
    single = value.get("mode") == "single"
    required = common | ({"locale"} if single else {"declared_languages", "adapter_options", "max_candidates"})
    if (set(value) != required or any(type(value.get(key)) is not str for key in common - {"languages"})
            or (value["provider"], value["registry_name"], value["model"]) != ("openai", "openai_transcribe", MODEL)
            or not _languages(value["languages"], _ISO639)):
        return None
    if single:
        locale = value["locale"]
        if (len(value["languages"]) != 1 or type(locale) is not str or len(locale) > 64
                or not _LOCALE.fullmatch(locale) or locale.split("-")[0].lower() != value["languages"][0]):
            return None
    else:
        options = value["adapter_options"]
        if (value["mode"] != "language_multi" or len(value["languages"]) < 2
                or not _languages(value["declared_languages"], _LOCALE)
                or not set(value["languages"]) <= set(value["declared_languages"])
                or value["max_candidates"] is not None or type(options) is not dict
                or set(options) != {"language"} or type(options["language"]) is not str
                or options["language"] != "multi"):
            return None
    # The selected (frozen) languages, bare ISO-639 as the plugin sends them;
    # never the provider's wider declared set. Nothing customer-derived beyond
    # them: no prompt, an empty keyword list (the plugin always sends the
    # field). Explicitly disable provider turn detection: the finite source
    # owns the one commit and must never inherit a session's VAD default.
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "noise_reduction": None, "turn_detection": None,
        "transcription": {"model": MODEL, "languages": list(value["languages"]), "keywords": []}}}}}


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


def _label(value):
    return type(value) is str and 1 <= len(value) <= 256 and all(32 <= ord(char) <= 126 for char in value)


def _count(value):
    return type(value) is int and 0 <= value <= MAX_USAGE_TOKENS


def _require(condition):
    if not condition:
        raise ValueError("invalid provider message")


def _usage(value):
    """Only provider-reported duration is an exact fact for this duration bill."""
    _require(type(value) is dict and type(value.get("type")) is str)
    if value["type"] == "duration":
        _require(set(value) == {"type", "seconds"} and type(value["seconds"]) in (int, float)
                 and 0 <= value["seconds"] <= MAX_SAMPLES / 24000)
        return (("audio_second", value["seconds"]),)
    _require(value["type"] == "tokens" and _TOKEN_USAGE_KEYS <= set(value) <= _TOKEN_USAGE_KEYS | {"input_token_details"}
             and all(_count(value[key]) for key in ("input_tokens", "output_tokens", "total_tokens"))
             and value["total_tokens"] == value["input_tokens"] + value["output_tokens"])
    details = value.get("input_token_details")
    _require(details is None or (type(details) is dict and set(details) <= {"audio_tokens", "text_tokens"}
                                 and all(item is None or _count(item) for item in details.values())))
    # A different wire usage basis cannot be converted into this account's
    # duration invoice. Preserve a conservative unknown, never invent seconds
    # from tokens or from the local recording's shorter sample duration.
    return None


def _decode(raw):
    """One provider event as (kind, payload); anything else refuses."""
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
    _require(type(value) is dict and type(value.get("type")) is str)
    kind = value["type"]
    if kind in _HARMLESS:
        return kind, None
    _require(kind in _KEYS)
    required, allowed = _KEYS[kind]
    _require(required <= set(value) <= allowed and _label(value["event_id"]) and _label(value["item_id"]))
    # Log probabilities are never requested; a value would be unrequested data.
    _require(value.get("logprobs") is None)
    # Required on a completion; optional (and nullable) on a delta.
    if "content_index" in value and (kind == COMPLETED or value["content_index"] is not None):
        _require(type(value["content_index"]) is int and 0 <= value["content_index"] <= 64)
    if kind == COMMITTED:
        _require(value.get("previous_item_id") is None or _label(value["previous_item_id"]))
        return kind, value["item_id"]
    if kind == DELTA:
        # Realtime pads streamed deltas against traffic analysis. It is not
        # transcript content and never reaches the collector or logs.
        padding = value.get("obfuscation")
        _require(padding is None or (type(padding) is str and len(padding) <= 1024))
        _require(value.get("delta") is None or (type(value["delta"]) is str and len(value["delta"]) <= MAX_CHARACTERS))
        return kind, None
    _require(type(value["transcript"]) is str)
    languages = value.get("languages")
    _require(languages is None or (type(languages) is list and len(languages) <= 32
                                   and all(type(item) is dict and set(item) == {"code"} and _label(item["code"])
                                           for item in languages)))
    return kind, (value["item_id"], value["transcript"], _usage(value["usage"]))


async def _stop_redirect(*_):
    # Raised inside the connect, which ``_run`` reports as never acknowledged.
    raise ListenerRefused("no_ack")


class OpenAITranscribeWorkflowListener:
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
        # The reference is never sent; it names this source's native metric.
        self._secret, self._reference, self._config = secret, client_reference_id, config
        self._metric_reference = client_reference_id
        self._created = _now()
        self._queue = asyncio.Queue()
        self._queued_frames = self._queued_bytes = self._samples = 0
        self._ended = self._commit_sent = self._updated = self._bad = self._closed = self._ack = False
        self._refusal = None
        self._eof_at = self._item = self._units = None
        self._runner = self._receiver = self._socket = self._session = None
        self._owned_tasks = []
        self._receive_timer = self._run_timer = None
        self._cancelled_runner = False
        self._started = False
        self._transcript_claimed = False
        self._parts = []
        self._events = self._finals = self._interims = self._characters = self._utf8_bytes = self._appended = 0
        # Evidence only: when the provider's last non-blank final arrived.
        self._final_at = None

    def __repr__(self):
        return "<OpenAITranscribeWorkflowListener redacted>"

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
        # The commit sentinel has a reserved control slot and consumes no PCM budget.
        self._queue.put_nowait(None)
        if self._receive_timer is not None:
            self._receive_timer.reschedule(self._deadline())
        if self._run_timer is not None:
            self._run_timer.reschedule(self._deadline())

    def _deadline(self):
        absolute = min(self._created + WALL_SECONDS,
                       self._eof_at + FINISHED_SECONDS if self._eof_at is not None else float("inf"))
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
        try:
            async with asyncio.timeout_at(self._deadline()) as timer:
                self._receive_timer = timer
                while True:
                    message = await self._socket.receive()
                    now = _now()
                    if (now > self._created + WALL_SECONDS
                            or (self._eof_at is not None and now > self._eof_at + FINISHED_SECONDS)):
                        return False
                    # The provider never closes first on a clean turn: this
                    # source closes after its own item's transcription.
                    if message.type != aiohttp.WSMsgType.TEXT:
                        return False
                    kind, value = _decode(message.data)
                    self._events += 1
                    if self._events > MAX_EVENTS:
                        return False
                    if kind == "session.updated":
                        self._updated = True
                    elif kind == DELTA:
                        self._interims += 1
                    elif kind == COMMITTED:
                        # Only this source's one commit, after its configuration applied.
                        if not self._commit_sent or not self._updated or self._item is not None:
                            return False
                        self._item = value
                    elif kind == COMPLETED:
                        item, text, units = value
                        if self._item is None or item != self._item:
                            return False
                        self._units = units
                        self._append(text)
                        return True
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
            async with asyncio.timeout_at(self._deadline()) as timer:
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
                        # The session never ends on its own: completion is an item
                        # outcome, and this source owns the close. Its handshake must
                        # finish in time; the code the provider echoes is not a fact
                        # about the transcript.
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
        async with asyncio.timeout_at(min(self._deadline(),
                                         asyncio.get_running_loop().time() + CONNECT_SECONDS)):
            self._socket = await self._session.ws_connect(ENDPOINT,
                headers={"Authorization": "Bearer " + self._secret}, max_msg_size=MAX_MESSAGE_BYTES)
        self._secret = None
        await self._socket.send_str(json.dumps(self._config))
        self._receiver = asyncio.create_task(self._receive(), name="workflow-openai-transcribe-receiver")
        self._owned_tasks.append(self._receiver)
        self._receiver.add_done_callback(self._receiver_done)
        while True:
            pcm = await self._queue.get()
            if pcm is None:
                # An empty buffer cannot be committed (the provider answers with an
                # error); no audio is a refused turn, never an empty success.
                if _now() > self._eof_at + FINISHED_SECONDS or not self._appended:
                    return False
                # Set before the write: the provider may answer before it returns.
                self._commit_sent = True
                await self._socket.send_str(COMMIT)
                return await self._receiver
            try:
                # At most 960 PCM bytes: a 1320-byte event, far inside the 15 MiB event limit.
                await self._socket.send_str(json.dumps({"type": "input_audio_buffer.append",
                                                        "audio": base64.b64encode(pcm).decode("ascii")}))
                self._appended += 1
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
            self._runner = asyncio.create_task(self._run(), name="workflow-openai-transcribe-runner")
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
        # The provider's detected languages are not evidence this source may
        # report for either a single or a multi-language turn; no hint is invented.
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
        units = self._units
        if units is None:
            # Zeros are mandatory unknown placeholders, never a known-zero invoice.
            return SourceObservation("connect_unknown", tuple((key, 0) for key in DIMENSIONS), (), "", self._ack)
        # The provider's own billed duration for this source's one item. Without
        # a clean terminal they are a measured prefix, never an exact closure.
        return SourceObservation("exact" if self._ack else "partial_unknown", units,
                                 (SourceMetric(units, self._metric_reference),), self._metric_reference, self._ack)

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


def build_openai_transcribe_listener(*, secret, constructor, client_reference_id):
    return OpenAITranscribeWorkflowListener(secret=secret, constructor=constructor,
                                            client_reference_id=client_reference_id)

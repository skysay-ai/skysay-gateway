# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Private finite synthesis process. Idle workers retain no source credentials."""
from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import gc
import importlib.metadata
import importlib.util
import logging
import os
from pathlib import Path
import resource
import select
import sys
import threading
import time


def _sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wire = _sibling("_owned_authored_wire", "workflow_authored_wire.py")


def _read(size):
    result = bytearray()
    while len(result) < size:
        part = os.read(0, size - len(result))
        if not part:
            wire.refuse()
        result.extend(part)
    return bytes(result)


def _receive():
    kind, size = wire.header(_read(wire.HEADER.size))
    return kind, _read(size)


class Output:
    def __init__(self, fd):
        self.fd = fd
        os.set_blocking(fd, False)

    def send(self, kind, payload):
        encoded = wire.encode(kind, payload)
        until = time.monotonic() + 2
        while encoded:
            remaining = until - time.monotonic()
            if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                wire.refuse()
            try:
                size = os.write(self.fd, encoded)
            except BlockingIOError:
                continue
            if size <= 0:
                wire.refuse()
            encoded = encoded[size:]


class Emission:
    """Numeric provider facts and bounded PCM, never provider diagnostics."""
    def __init__(self, output, identity):
        self.output, self.identity = output, identity
        self.seen = self.ended = self.functional = False
        self.tokens = None
        self.frames = self.bytes = 0

    def response(self):
        if not self.seen:
            self.output.send(wire.Kind.SEEN, wire.pack_json({"lease": self.identity}))
            self.seen = True

    def audio(self, data):
        if type(data) is not bytes or not data or len(data) % 2 or not self.seen:
            wire.refuse()
        for offset in range(0, len(data), wire.MAX_PCM_FRAME):
            part = data[offset:offset + wire.MAX_PCM_FRAME]
            if self.frames >= wire.MAX_PCM_FRAMES or self.bytes + len(part) > wire.MAX_PCM_BYTES:
                wire.refuse()
            self.output.send(wire.Kind.PCM, self.identity.encode("ascii") + part)
            self.frames += 1
            self.bytes += len(part)

    def end(self):
        self.output.send(wire.Kind.END, wire.pack_json({
            "lease": self.identity, "response_seen": self.seen,
            "tokens": self.tokens if self.ended else None, "response_ended": self.ended,
            "functional": self.functional, "frames": self.frames, "bytes": self.bytes}))


class Control:
    def __init__(self, identity, loop, task):
        self.identity, self.loop, self.task = identity, loop, task
        self.valid = self.cancelled = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            kind, payload = _receive()
            if kind not in (wire.Kind.CANCEL, wire.Kind.RELEASE) or wire.control(payload) != self.identity:
                wire.refuse()
            self.valid = True
            if kind == wire.Kind.CANCEL:
                self.cancelled = True
                with contextlib.suppress(RuntimeError):
                    self.loop.call_soon_threadsafe(self.task.cancel)
        except BaseException:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.task.cancel)


async def _owned(operation, request, emission, controls):
    control = Control(request.lease_id, asyncio.get_running_loop(), asyncio.current_task())
    controls.append(control)
    control.thread.start()
    try:
        await operation(request, emission)
    except asyncio.CancelledError:
        emission.functional = False


def serve(output_fd, operation, preflight=None):
    """The fixture supplies an operation, but runs this same ownership loop."""
    _harden()
    output = Output(output_fd)
    if preflight is not None:
        preflight()
    output.send(wire.Kind.PREFLIGHT, wire.pack_json({"protocol": wire.PROTOCOL}))
    while True:
        kind, payload = _receive()
        if kind == wire.Kind.STOP and payload == b"":
            return
        if kind != wire.Kind.CONFIG:
            wire.refuse()
        request = wire.configuration(payload)
        del payload
        identity = request.lease_id
        emission, controls = Emission(output, identity), []
        baseline_threads = frozenset(threading.enumerate())
        # A new loop joins source tasks, async generators and executor work.
        # A source that cannot be joined never reaches the closure ACK.
        asyncio.run(_owned(operation, request, emission, controls))
        emission.end()
        control = controls.pop()
        control.thread.join(2)
        if control.thread.is_alive() or not control.valid:
            wire.refuse()
        if any(thread not in baseline_threads and thread.is_alive() for thread in threading.enumerate()):
            wire.refuse()
        request.clear()
        del request, control, controls, emission
        gc.collect()
        output.send(wire.Kind.CLOSED, wire.pack_json({"lease": identity}))


def _harden():
    if not sys.flags.isolated or frozenset(os.environ) - {"LANG", "TZ"}:
        wire.refuse()
    if any(name == "sitecustomize" or name.startswith(("sentry", "ddtrace", "opentelemetry.instrumentation"))
           for name in sys.modules):
        wire.refuse()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.disable()
    logging.disable(sys.maxsize)
    sys.excepthook = lambda *_: None
    sys.unraisablehook = lambda *_: None


def _preflight():
    if importlib.metadata.version("google-genai") != wire.SDK_VERSION:
        wire.refuse()
    from google.genai import types
    import inspect
    import httpx
    options = types.HttpRetryOptions(attempts=1)
    if (options.attempts != 1
            or not {"async_client_args", "client_args"} <= types.HttpOptions.model_fields.keys()):
        wire.refuse()
    inspect.signature(httpx.AsyncHTTPTransport).bind(retries=0, trust_env=False)


async def gemini_operation(request, emission):
    """Explicit single-attempt provider request; no ambient credential lookup."""
    from google import genai
    from google.genai import errors, types
    import httpx

    usage_module = _sibling("_owned_authored_usage", "gemini_tts_usage.py")
    delivery = _sibling("_owned_authored_delivery", "capabilities/delivery.py")
    tracker = usage_module.GeminiTTSUsageTracker()
    previous = None
    invalid_usage = False
    client = response = chunk = None
    stopped = 0
    invalid_audio = False
    first_format = None
    try:
        client = genai.Client(api_key=request.secret, vertexai=False, http_options=types.HttpOptions(
            timeout=10000, retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": httpx.AsyncHTTPTransport(retries=0, trust_env=False),
                               "trust_env": False}, client_args={"trust_env": False}))
        config = types.GenerateContentConfig(response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=request.voice_id))))
        response = await client.aio.models.generate_content_stream(model=wire.MODEL,
            contents=delivery.gemini_delivery_prompt(delivery.GEMINI_DEFAULT_INSTRUCTION, request.text),
            config=config)
        async for chunk in response:
            emission.response()
            metadata = chunk.usage_metadata
            if metadata is not None:
                tracker.record(metadata)
                try:
                    pair = tracker.require_latest_complete()
                    if (not wire.token_pair(pair) or previous is not None
                            and any(new < old for new, old in zip(pair, previous))):
                        invalid_usage = True
                    previous = pair
                except usage_module.GeminiTTSUsageError:
                    pass
            candidates = chunk.candidates
            if not candidates:
                continue
            if len(candidates) != 1:
                invalid_audio = True
                continue
            candidate = candidates[0]
            parts = candidate.content.parts if candidate.content is not None else None
            for part in parts or ():
                if part.text is not None or part.inline_data is None or stopped:
                    invalid_audio = True
                    continue
                blob = part.inline_data
                try:
                    current = wire.audio_format(blob.mime_type)
                    if first_format is not None and current != first_format:
                        wire.refuse()
                    first_format = current
                    emission.audio(blob.data)
                except wire.WireRefused:
                    invalid_audio = True
            if candidate.finish_reason is not None:
                if candidate.finish_reason != types.FinishReason.STOP:
                    invalid_audio = True
                stopped += 1
        emission.ended = True
        try:
            pair = tracker.require_latest_complete()
            if not invalid_usage and wire.token_pair(pair) and pair[0] > 0 and (not emission.bytes or pair[1] > 0):
                emission.tokens = pair
        except usage_module.GeminiTTSUsageError:
            pass
        emission.functional = not invalid_audio and stopped == 1 and emission.bytes > 0
    except errors.APIError:
        emission.response()
    except Exception:
        emission.functional = False
    finally:
        if response is not None:
            await response.aclose()
        if client is not None:
            await client.aio.aclose()
            client.close()
        del request, client, response, chunk


def main():
    if len(sys.argv) != 2:
        return 1
    try:
        serve(int(sys.argv[1]), gemini_operation, _preflight)
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

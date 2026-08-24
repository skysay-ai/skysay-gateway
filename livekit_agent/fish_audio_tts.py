# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Fish Audio adapter with exact UTF-8 input-byte accounting.

The pinned upstream LiveKit plugin owns WebSocket setup, cancellation and
cleanup. Fish bills the text bytes actually sent in its ``event: text`` frames,
not the Python characters offered by an LLM. The plugin's generic TTS metric
cannot represent that distinction, so this narrow subclass mirrors the pinned
stream send loop and records only a successfully-written provider frame.
"""
from __future__ import annotations

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins import fishaudio
from livekit.plugins.fishaudio import tts as fish_tts

try:  # package import in the API image/tests
    from .native_tts_usage import Utf8ByteCounter, await_fish_stream_tasks
except ImportError:  # pragma: no cover - direct voice-worker entrypoint
    from native_tts_usage import Utf8ByteCounter, await_fish_stream_tasks


class _ByteAccountingSynthesizeStream(fish_tts.SynthesizeStream):
    """Pinned Fish stream with accounting at its provider-frame boundary."""

    def __init__(self, *, tts: "TTS", conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._counter = tts.utf8_byte_counter

    async def _run_ws(self, ws: object, output_emitter: object) -> None:
        """Mirror fishaudio 1.6.6's send/receive/cancel behavior exactly.

        The only semantic difference is the counter update after a successful
        ``event: text`` write. Counting here includes the provider-required
        trailing space and correctly accounts for a framework retry that really
        sends the sentence again.
        """
        sent_stream = self._tts._sentence_tokenizer.stream()

        async def input_task() -> None:
            try:
                first_token = True
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        sent_stream.flush()
                        continue
                    if not data:
                        continue
                    if first_token:
                        self._mark_started()
                        first_token = False
                    sent_stream.push_text(data)
            finally:
                sent_stream.end_input()

        async def send_task() -> None:
            start_msg = {"event": "start", "request": fish_tts._build_tts_request(self._opts)}
            await ws.send_bytes(fish_tts.msgpack.packb(start_msg, use_bin_type=True))
            async for event in sent_stream:
                sentence = event.token
                if not sentence:
                    continue
                provider_text = sentence + " "
                await ws.send_bytes(
                    fish_tts.msgpack.packb(
                        {"event": "text", "text": provider_text}, use_bin_type=True
                    )
                )
                self._counter.record(provider_text)
                await ws.send_bytes(fish_tts.msgpack.packb({"event": "flush"}, use_bin_type=True))
            await ws.send_bytes(fish_tts.msgpack.packb({"event": "stop"}, use_bin_type=True))

        async def recv_task() -> None:
            while True:
                message = await ws.receive()
                if message.type in (
                    fish_tts.aiohttp.WSMsgType.CLOSED,
                    fish_tts.aiohttp.WSMsgType.CLOSE,
                    fish_tts.aiohttp.WSMsgType.CLOSING,
                ):
                    raise fish_tts.APIStatusError(
                        "Fish Audio websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=None,
                        body=f"message.data={message.data!r} message.extra={message.extra!r}",
                    )
                if message.type != fish_tts.aiohttp.WSMsgType.BINARY:
                    fish_tts.logger.debug("unexpected Fish Audio message type %s", message.type)
                    continue
                data = fish_tts.msgpack.unpackb(message.data, raw=False)
                event = data.get("event")
                if event == "audio":
                    audio = data.get("audio")
                    if audio:
                        output_emitter.push(audio)
                elif event == "finish":
                    if data.get("reason") == "error":
                        raise fish_tts.APIStatusError(
                            "Fish Audio TTS reported an error",
                            status_code=-1,
                            request_id=None,
                            body=str(data),
                        )
                    break
                else:
                    fish_tts.logger.debug("unknown Fish Audio event: %s", data)

        input_runner = fish_tts.asyncio.create_task(input_task())
        send_runner = fish_tts.asyncio.create_task(send_task())
        receive_runner = fish_tts.asyncio.create_task(recv_task())
        tasks = [input_runner, send_runner, receive_runner]
        try:
            await await_fish_stream_tasks(
                input_task=input_runner,
                send_task=send_runner,
                receive_task=receive_runner,
                cancel=fish_tts.utils.aio.gracefully_cancel,
            )
        finally:
            await fish_tts.utils.aio.gracefully_cancel(*tasks)
            await sent_stream.aclose()


class TTS(fishaudio.TTS):
    """Upstream Fish TTS with a call-local UTF-8 input counter."""

    def __init__(self, *, utf8_byte_counter: Utf8ByteCounter, **kwargs: object) -> None:
        # ``SynthesizeStream._run`` is inherited from the pinned LiveKit
        # implementation. It initializes its AudioEmitter from these options
        # before delegating the provider send/receive loop below, so make the
        # paid model and WAV framing an adapter invariant instead of relying
        # only on its production-builder call site.
        if kwargs.get("model") != "s2-pro":
            raise ValueError("Fish Audio TTS must use paid s2-pro")
        if kwargs.get("output_format", "wav") != "wav":
            raise ValueError("Fish Audio TTS must use WAV output")
        kwargs["output_format"] = "wav"
        super().__init__(**kwargs)
        self.utf8_byte_counter = utf8_byte_counter

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> object:
        """Keep direct synthesis on the metered streaming send path too.

        The normal AgentSession calls :meth:`stream`, but a future wrapper or
        diagnostic path may call ``synthesize``.  Delegating to the base
        stream bridge prevents that otherwise-valid Fish request from bypassing
        the exact byte counter used by settlement.
        """
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _ByteAccountingSynthesizeStream:
        stream = _ByteAccountingSynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream


__all__ = ["TTS"]

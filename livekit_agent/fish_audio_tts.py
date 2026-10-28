# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Fish Audio adapter with UTF-8 input-byte accounting at its provider wire.

The pinned upstream LiveKit plugin owns WebSocket setup, cancellation and
cleanup. Fish bills the text bytes sent in its ``event: text`` frames, not the
Python characters offered by an LLM. The plugin's generic TTS metric cannot
represent that distinction, so this narrow subclass mirrors the pinned stream
send loop and hands each text frame's bytes to the call's native unit meter;
how an attempt ends, and what that ending is charged, is the meter's data.
"""
from __future__ import annotations

from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins import fishaudio
from livekit.plugins.fishaudio import tts as fish_tts

try:  # package import in the API image/tests
    from .native_tts_usage import (
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        await_fish_stream_tasks,
        hand_off,
        mark_provider_error,
        mark_transport_failure,
        run_metered_request,
    )
except ImportError:  # pragma: no cover - direct voice-worker entrypoint
    from native_tts_usage import (
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        await_fish_stream_tasks,
        hand_off,
        mark_provider_error,
        mark_transport_failure,
        run_metered_request,
    )


class _ByteAccountingSynthesizeStream(fish_tts.SynthesizeStream):
    """Pinned Fish stream with accounting at its provider-frame boundary."""

    def __init__(self, *, tts: "TTS", conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._meter = tts.native_unit_meter

    async def _run(self, output_emitter: object) -> None:
        """Run each attempt as one metered provider request."""
        try:
            await run_metered_request(
                self._meter,
                lambda: super(_ByteAccountingSynthesizeStream, self)._run(output_emitter)
            )
        except TTSMeterFrozen:
            # Usage evidence was taken at teardown: close like a client that
            # hung up rather than inviting LiveKit to retry.
            raise fish_tts.APIStatusError(
                "TTS usage evidence was already taken", status_code=499, request_id=None, body=None
            ) from None


    async def _run_ws(self, ws: object, output_emitter: object) -> None:
        """Mirror the pinned plugin's send/receive/cancel behavior exactly.

        The semantic differences are accounting only: each ``event: text``
        frame's UTF-8 bytes, including the provider-required trailing space,
        are handed to the meter immediately before the frame is sent, so a
        framework retry that really sends the sentence again counts again; and
        a ``finish`` frame reporting an error marks a provider-refused request.
        """
        if getattr(ws, "compress", 0):
            # An uncompressed frame reaches the transport before the send's
            # first await, so a send the worker cancels was already handed.
            raise ValueError("Fish Audio socket must be uncompressed")
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
            start_frame = fish_tts.msgpack.packb(start_msg, use_bin_type=True)
            await hand_off(self._meter, 0, lambda: ws.send_bytes(start_frame))
            async for event in sent_stream:
                sentence = event.token
                if not sentence:
                    continue
                provider_text = sentence + " "
                frame = fish_tts.msgpack.packb({"event": "text", "text": provider_text}, use_bin_type=True)
                await hand_off(self._meter, len(provider_text.encode("utf-8")), lambda: ws.send_bytes(frame))
                flush_frame = fish_tts.msgpack.packb({"event": "flush"}, use_bin_type=True)
                await hand_off(self._meter, 0, lambda: ws.send_bytes(flush_frame))
            stop_frame = fish_tts.msgpack.packb({"event": "stop"}, use_bin_type=True)
            await hand_off(self._meter, 0, lambda: ws.send_bytes(stop_frame))

        async def recv_task() -> None:
            while True:
                try:
                    message = await ws.receive()
                except fish_tts.asyncio.CancelledError:
                    raise
                except BaseException:
                    mark_transport_failure()
                    raise
                if message.type in (
                    fish_tts.aiohttp.WSMsgType.CLOSED,
                    fish_tts.aiohttp.WSMsgType.CLOSE,
                    fish_tts.aiohttp.WSMsgType.CLOSING,
                ):
                    mark_transport_failure()
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
                        mark_provider_error()
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
    """Upstream Fish TTS with a call-local native UTF-8 byte meter."""

    def __init__(self, *, native_unit_meter: NativeTTSUnitMeter, **kwargs: object) -> None:
        # ``SynthesizeStream._run`` is inherited from the pinned LiveKit
        # implementation. It initializes its AudioEmitter from these options
        # before delegating the provider send/receive loop below, so make the
        # paid model and WAV framing an adapter invariant instead of relying
        # only on its production-builder call site.
        if kwargs.get("model") != "s2-pro":
            raise ValueError("Fish Audio TTS must use paid s2-pro")
        if not isinstance(native_unit_meter, NativeTTSUnitMeter) or native_unit_meter.billing.unit != "utf8_byte":
            raise ValueError("Fish Audio TTS requires a UTF-8 byte native unit meter")
        if kwargs.get("output_format", "wav") != "wav":
            raise ValueError("Fish Audio TTS must use WAV output")
        kwargs["output_format"] = "wav"
        super().__init__(**kwargs)
        self.native_unit_meter = native_unit_meter

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> object:
        """Keep direct synthesis on the metered streaming send path too.

        The normal AgentSession calls :meth:`stream`, but a future wrapper or
        diagnostic path may call ``synthesize``.  Delegating to the base
        stream bridge prevents that otherwise-valid Fish request from bypassing
        the native unit meter used by settlement.
        """
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _ByteAccountingSynthesizeStream:
        stream = _ByteAccountingSynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream


__all__ = ["TTS"]

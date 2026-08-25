# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Gemini TTS adapter that retains the provider's exact token evidence.

The pinned LiveKit Google Gemini TTS plugin streams audio but drops the
``usage_metadata`` attached to Gemini response chunks.  This narrow adapter
preserves the upstream request/error behaviour and reports those native token
counts through LiveKit's normal ``TTSMetrics`` path.  Paid calls can require
the evidence before any buffered audio is emitted.
"""
from __future__ import annotations

from typing import Any

from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.google.beta import gemini_tts as upstream

try:  # package import in tests; flat /app import in the worker image
    from .gemini_tts_usage import GeminiTTSUsageError, GeminiTTSUsageTracker
except ImportError:  # pragma: no cover - exercised by the container smoke
    from gemini_tts_usage import GeminiTTSUsageError, GeminiTTSUsageTracker


# Retain a hard limit so malformed provider output cannot grow one worker's
# memory without bound before the native usage trailer is received.
_MAX_BUFFERED_AUDIO_BYTES = 8 * 1024 * 1024


class TTS(upstream.TTS):
    """Upstream Gemini TTS with a default-on exact-usage admission guard."""

    _openphonex_exact_usage_adapter = True

    def __init__(self, *args: Any, require_usage_metadata: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Only the two explicit unmetered demo call sites receive literal
        # ``False``. A missing, stale, or malformed feature/billing value must
        # retain the fail-closed paid-media default instead of becoming a
        # truthiness-based opt-out.
        self._require_usage_metadata = require_usage_metadata is not False

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Copy the upstream chunked stream, adding only native usage capture."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._tts._opts.voice_name,
                        )
                    )
                ),
            )
            input_text = self._input_text
            if self._tts._opts.instructions is not None:
                input_text = f'{self._tts._opts.instructions}:\n"{input_text}"'

            response = await self._tts._client.aio.models.generate_content_stream(
                model=self._tts._opts.model,
                contents=input_text,
                config=config,
            )

            # Paid calls must not emit media before their exact native usage
            # trailer has arrived. Non-billed browser/demo sessions retain the
            # upstream incremental emission path instead of taking a full
            # response-latency regression just because this adapter is loaded.
            buffer_until_usage = self._tts._require_usage_metadata
            audio_chunks: list[bytes] = []
            audio_bytes = 0
            usage = GeminiTTSUsageTracker()
            if not buffer_until_usage:
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
            async for chunk in response:
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata is not None:
                    # A final trailer is authoritative.  Some streaming
                    # responses send incomplete interim metadata first, so
                    # retain a later complete cumulative report but fail a
                    # paid call if its latest trailer is incomplete.
                    usage.record(metadata)
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    for part in chunk.candidates[0].content.parts:
                        inline_data = part.inline_data
                        if not (
                            inline_data
                            and inline_data.data
                            and inline_data.mime_type
                            and inline_data.mime_type.startswith("audio/")
                        ):
                            continue
                        data = inline_data.data
                        if not isinstance(data, (bytes, bytearray)):
                            raise APIStatusError(
                                "gemini tts: audio payload is invalid",
                                status_code=502,
                                retryable=False,
                            )
                        if buffer_until_usage:
                            if len(data) > _MAX_BUFFERED_AUDIO_BYTES - audio_bytes:
                                raise APIStatusError(
                                    "gemini tts: generated audio exceeds the bounded worker contract",
                                    status_code=502,
                                    retryable=False,
                                )
                            audio_chunks.append(bytes(data))
                            audio_bytes += len(data)
                        else:
                            output_emitter.push(bytes(data))

            if buffer_until_usage:
                try:
                    token_usage = usage.require_latest_complete()
                except GeminiTTSUsageError as exc:
                    # Do not emit buffered audio on a paid call without exact
                    # provider evidence.  The normal LiveKit retry path sees
                    # this as non-retryable and the wallet reservation remains
                    # intact.
                    raise APIStatusError(
                        f"gemini tts: {exc}",
                        status_code=502,
                        retryable=False,
                    ) from exc
                # `require_latest_complete` returns an exact native tuple or
                # raises. Record it before the paid stream is initialized, so
                # media cannot be emitted without settlement evidence.
                self._set_token_usage(
                    input_tokens=token_usage[0], output_tokens=token_usage[1]
                )
            else:
                token_usage = usage.latest_tokens
                if token_usage is not None:
                    self._set_token_usage(
                        input_tokens=token_usage[0], output_tokens=token_usage[1]
                    )

            if buffer_until_usage:
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
                for data in audio_chunks:
                    output_emitter.push(data)

        except APIStatusError:
            raise
        except ClientError as exc:
            raise APIStatusError(
                "gemini tts: client error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=exc.code in {429, 499},
            ) from exc
        except ServerError as exc:
            raise APIStatusError(
                "gemini tts: server error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=True,
            ) from exc
        except APIError as exc:
            raise APIStatusError(
                "gemini tts: api error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise APIConnectionError(
                f"gemini tts: error generating speech {str(exc)}",
                retryable=True,
            ) from exc

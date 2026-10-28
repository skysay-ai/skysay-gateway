# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Gemini TTS adapter that meters the provider's exact token usage.

The pinned LiveKit Google Gemini TTS plugin streams audio but drops the
``usage_metadata`` attached to Gemini response chunks. This narrow adapter
preserves the upstream request/error behaviour and runs every attempt as one
request of the native unit meter (``native_tts_usage``): Gemini reports
CUMULATIVE prompt and generated-audio token counts on its streamed chunks (the
3.x models on every ~40 ms chunk, 2.5 once with the whole response), and each
report replaces the request's measured usage. What a request is charged once it
ends -- completed, abandoned by the worker (a barge-in), refused, or lost to a
transport failure -- is the provider's data in the capability artifact, never a
branch here; nothing is ever estimated from characters or duration.

Audio is always streamed to the caller as it arrives. A completed turn whose
final usage trailer is missing or incomplete is settled from the latest valid
report the provider sent (zero when it sent none) and logged; raising at that
point would `aclose()` the emitter and discard or duplicate audio the caller was
already going to hear. When nothing was queued, the turn still fails as a media
failure (see ``_run``).

Two different signals answer two different questions here, and using the
wrong one for either is a bug this file has already made once (see the two
Codex-review rounds cited in ``_run`` and ``_retryable``). "Was anything
queued for delivery" (a missing-trailer turn deciding whether to return
normally and let LiveKit's own ``end_input()``/``join()`` flush it) is
answered by this adapter's own ``push()`` call count. "Has a frame already
reached the downstream channel" (a mid-response error deciding whether a
retry would duplicate audio the caller may already be hearing) is answered by
``output_emitter.pushed_duration()``. The two are not interchangeable:
``AudioEmitter`` deliberately holds back a small tail of audio and flushes it
only on ``end_input()``, so a short reply can have real queued audio with
``pushed_duration() == 0``.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx
from google.genai import Client, types
from google.genai.errors import APIError, ClientError, ServerError
from livekit.agents import APIConnectionError, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.plugins.google.beta import gemini_tts as upstream

try:  # package import in tests; flat /app import in the worker image
    from .capabilities.delivery import gemini_raw_audio_generation_config, gemini_reads_transcript
    from .gemini_tts_usage import GeminiTTSUsageError, GeminiTTSUsageTracker
    from .native_tts_usage import (
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        current_tts_request,
        mark_provider_error,
        run_metered_request,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke
    from capabilities.delivery import gemini_raw_audio_generation_config, gemini_reads_transcript
    from gemini_tts_usage import GeminiTTSUsageError, GeminiTTSUsageTracker
    from native_tts_usage import (
        NativeTTSUnitMeter,
        TTSMeterFrozen,
        current_tts_request,
        mark_provider_error,
        run_metered_request,
    )


class GeminiTTSRuntimeError(RuntimeError):
    """The Gemini adapter was built without the meter its billing requires."""


class _AdmittedBody(httpx.AsyncByteStream):
    """A request body the meter admits at the moment it is written.

    httpx reaches the body only after it has a connection and has written the
    headers, so a request that waited for a connection while usage evidence
    was taken never delivers its body: Gemini sees an incomplete request and
    processes nothing.
    """

    def __init__(self, body: Any, meter: NativeTTSUnitMeter, attempt: Any) -> None:
        self._body = body
        self._meter = meter
        self._attempt = attempt

    async def __aiter__(self) -> Any:
        self._meter.refuse_if_settling(self._attempt)
        async for chunk in self._body:
            yield chunk

    async def aclose(self) -> None:
        close = getattr(self._body, "aclose", None)
        if callable(close):
            await close()


class _MeteredDispatch(httpx.AsyncHTTPTransport):
    """Admit every HTTP request the Gemini SDK makes against the meter.

    The attempt opens its metered request before entering the SDK, but the
    request leaves only when the SDK dispatches it -- possibly after usage
    evidence was taken at teardown. The meter admits it twice: when the SDK
    dispatches it, and again when its body is written after a connection was
    acquired. So no request body is admitted once the meter stops admitting,
    whatever the SDK or the connection pool did in between, and a dispatch
    outside a metered attempt is refused outright. As with every adapter's
    hand-off, a body admitted just before ``settle`` may still be written
    after it; that request is open at the freeze and is charged nothing.
    """

    def __init__(self, meter: NativeTTSUnitMeter) -> None:
        super().__init__(retries=0)
        self._meter = meter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        current = current_tts_request()
        if current is None:
            raise GeminiTTSRuntimeError("a Gemini request was dispatched outside a metered attempt")
        self._meter.refuse_if_settling(current)
        request.stream = _AdmittedBody(request.stream, self._meter, current)
        return await super().handle_async_request(request)


#: The token dimensions this adapter reports, in Gemini's own order:
#: ``prompt_token_count`` then generated-audio ``candidates_token_count``.
_REPORTED_DIMENSIONS = ("input_text_tokens", "output_audio_tokens")


logger = logging.getLogger(__name__)

# No worker-memory bound guards audio size here.  There used to be one
# (``_MAX_BUFFERED_AUDIO_BYTES``, 8 MiB) protecting the ``list[bytes]`` the
# paid path accumulated while withholding media.  Streaming removed that
# accumulator: every chunk goes straight to ``output_emitter.push`` and is not
# retained by this adapter, so there is nothing left here to bound.  The two
# unmetered demo call sites have pushed every chunk immediately, with no size
# cap, since before this change; the paid path now matches that same
# already-shipped, already-safe shape rather than inventing a new one.


# Gemini TTS can accept a request, stream a complete response, and
# produce NO AUDIO AT ALL -- either by refusing with a 400 whose message says
# the model "tried to generate text", or, far more often in production, by
# ending the stream with ``finish_reason=OTHER``, no audio parts, and a usage
# trailer carrying no generated-audio token count. The second shape is what
# silenced the greeting on every answered call of the first Sinhala campaign,
# and it reached the operator only as ``GeminiTTSUsageError: usage metadata
# omitted exact generated audio token count`` -- a billing-evidence message
# for what is really "the provider said nothing".
#
# Both shapes are handled here as one condition with one diagnostic, because
# to the callee they are the same event: the line did not play.
#
# What this does NOT do is fix it, and the reason is worth recording so the
# obvious repair is not attempted a third time. Measured 2026-08-28 against
# ``gemini-2.5-flash-preview-tts``: the campaign greeting (a Latin-script name
# inside Sinhala) returns no audio 0/6, deterministically, and stays 0/3 under
# every instruction wording tried -- none, the plugin default, read-verbatim,
# script-framing, a Sinhala-language instruction, an explicit TTS-engine
# framing -- and on a second voice and on ``gemini-2.5-pro-preview-tts``. The
# trigger is not the question form either: the same question with a
# pure-Sinhala name is 3/3, and the same Latin name as a plain statement is
# 0/3. The same text on ``gemini-3.1-flash-tts-preview`` is 15/15 and faster.
#
# So this is a MODEL property, not a prompt this adapter can improve, and a
# same-request retry has nothing to change. It is deliberately not retried:
# the caller's no-silence fallback is worth more than a second 3-6 second
# synthesis that measurement says will fail the same way. What is left here is
# the part that was actually missing -- naming the failure so an operator can
# count it. ``scripts/gemini_tts_greeting_probe.py`` reproduces the finding.
_TEXT_INSTEAD_OF_AUDIO_SIGNATURES = (
    "should only be used for tts",
    "tried to generate text",
)


def _is_text_instead_of_audio(exc: Exception) -> bool:
    """Whether this provider error is Gemini refusing to synthesize at all."""

    message = f"{getattr(exc, 'message', '')} {exc}".casefold()
    return any(signature in message for signature in _TEXT_INSTEAD_OF_AUDIO_SIGNATURES)


def _text_digest(text: str) -> str:
    """A short stable fingerprint of the requested text.

    A hash, never the text: this log line describes a customer's call and the
    adapter has no business copying what an agent says into worker logs. The
    digest is still enough to prove two failures were the same sentence, and
    to match a reported failure against a reproduction.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _retryable(default: bool, *, audio_delivered: bool, exc: Exception) -> bool:
    """A retry is only ever safe before any audio has reached the downstream pipeline.

    Streaming means a mid-response provider/connection error can happen
    AFTER some audio has already been pushed -- unlike the retired buffered
    path, where nothing reached the caller until the whole reply had
    synthesized successfully. ``audio_delivered`` must be
    ``output_emitter.pushed_duration() > 0``, not merely "was ``push()``
    called": ``push()`` only enqueues bytes for the emitter's own background
    task to decode and forward, and ``pushed_duration()`` is the one signal
    this adapter can read that reflects a frame actually reaching the
    downstream channel rather than sitting in that queue. When it is False,
    the emitter's ``aclose()`` (which LiveKit calls before any retry) is
    guaranteed to discard the queued bytes before they are ever forwarded --
    there is no ``await`` between where this is read and where ``aclose()``
    runs, so nothing forwards from this attempt in between. Once delivery IS
    confirmed, downgrade even an otherwise-retryable error to non-retryable:
    LiveKit's retry path would re-synthesize the same text from scratch on a
    fresh request, and the caller may already be hearing the prefix the first
    attempt already streamed.

    Do NOT reuse ``audio_bytes_emitted`` (a simple "was ``push()`` called"
    count) here even though ``_run``'s missing-trailer branch uses exactly
    that signal for its own, different decision -- see the module docstring
    for why they are not interchangeable.
    """
    if default and audio_delivered:
        logger.warning(
            "gemini_tts_retry_suppressed_after_partial_audio: %s",
            exc,
        )
        return False
    return default


class TTS(upstream.TTS):
    """Upstream Gemini TTS whose every attempt is one metered provider request."""

    _skysay_exact_usage_adapter = True

    def __init__(self, *args: Any, native_unit_meter: NativeTTSUnitMeter, **kwargs: Any) -> None:
        if (
            not isinstance(native_unit_meter, NativeTTSUnitMeter)
            or native_unit_meter.billing.unit != "token"
            or native_unit_meter.billing.dimensions != _REPORTED_DIMENSIONS
        ):
            raise GeminiTTSRuntimeError("Gemini TTS requires a token meter for its reported usage")
        api_key = kwargs.get("api_key")
        if not isinstance(api_key, str) or not api_key or kwargs.get("vertexai"):
            # Vertex is a different vendor relationship with its own rate card.
            raise GeminiTTSRuntimeError("Gemini TTS requires an explicit Gemini API key")
        super().__init__(*args, **kwargs)
        self._skysay_meter = native_unit_meter
        # The upstream client, rebuilt so every dispatch goes through the
        # meter's admission gate (an explicit transport selects the SDK's
        # httpx path, which has no hidden reconnect-and-resend of its own).
        self._client = Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                async_client_args={"transport": _MeteredDispatch(native_unit_meter), "trust_env": False}
            ),
        )

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
        # Survives across attempts on purpose: LiveKit's `_main_task` calls
        # `_run` again on this SAME instance (with a fresh AudioEmitter) after
        # a retryable failure, so this is what bounds the no-audio retry at
        # exactly one. It counts attempts and nothing else -- the retried
        # request is identical to the first.
        self._no_audio_attempts = 0

    def _no_audio_error(
        self,
        *,
        request_id: str,
        finish_reasons: list[str],
        provider_error: Exception | None = None,
        retryable: bool = False,
    ) -> APIStatusError:
        """Log the one diagnostic this failure never had, and build its error.

        Emitted on EVERY occurrence: an operator watching a campaign needs the
        rate, not a single sample.

        ``retryable`` is decided by the caller and defaults to False, which
        is what a positively identified refusal gets: the provider named the
        reason, an identical request will be refused identically, and the
        seconds spent proving that are seconds the caller spends in silence
        before the opening line's fallback can be spoken. A response that
        merely came back empty gets exactly one retry -- see the call site.

        Nothing free-form from the provider is recorded. `finish_reason` is a
        closed enum and the signature is one of this module's own constants;
        the provider's prose message is deliberately NOT logged and NOT put in
        the raised error's body, because LiveKit's own retry logger prints
        that body and a 400 that quotes the offending request would put a
        customer's greeting -- names included -- into worker logs. A changed
        provider message shows up as an unmatched signature, which is a
        diagnosis in itself.
        """

        # The provider answered without audio (or refused outright): free, as
        # every provider refusal is, whichever path raises this error.
        mark_provider_error()
        signature = ""
        if provider_error is not None:
            message = f"{getattr(provider_error, 'message', '')} {provider_error}".casefold()
            signature = next(
                (
                    candidate
                    for candidate in _TEXT_INSTEAD_OF_AUDIO_SIGNATURES
                    if candidate in message
                ),
                "",
            )
        logger.error(
            "gemini_tts_text_instead_of_audio: %s",
            signature or "no audio parts in a complete response",
            extra={
                "request_id": request_id,
                "model": self._tts._opts.model,
                "voice_name": self._tts._opts.voice_name,
                # Never the text itself -- see `_text_digest`.
                "text_digest": _text_digest(self._input_text),
                "characters_requested": len(self._input_text),
                "finish_reasons": finish_reasons,
                "provider_error_signature": signature,
                "provider_status_code": getattr(provider_error, "code", None),
                "will_retry": retryable,
            },
        )
        return APIStatusError(
            "gemini tts: provider returned no audio for this text",
            status_code=getattr(provider_error, "code", None) or 502,
            retryable=retryable,
        )

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # One attempt is one metered provider request: a return is completed,
        # the worker's cancel is abandoned, an observed refusal is a provider
        # error and any other exception a transport failure.
        try:
            await run_metered_request(self._tts._skysay_meter, lambda: self._attempt(output_emitter))
        except TTSMeterFrozen:
            # Usage evidence was taken at teardown: close like a client that
            # hung up rather than inviting LiveKit to retry.
            raise APIStatusError(
                "TTS usage evidence was already taken", status_code=499, request_id=None, body=None
            ) from None

    async def _attempt(self, output_emitter: tts.AudioEmitter) -> None:
        # Diagnostic only: the exact byte count handed to `output_emitter
        # .push`, logged for audit context. It is NOT what any correctness
        # decision below is gated on -- `push()` only enqueues bytes for the
        # emitter's own background task to decode/forward, so a nonzero count
        # here does not by itself prove audio reached the caller. Every
        # correctness check instead reads `output_emitter.pushed_duration()`
        # fresh, at the moment it matters (see `_retryable` above).
        audio_bytes_emitted = 0
        # LiveKit's retry loop (`_main_task`) calls `_run` again on the SAME
        # instance after a retryable failure; each attempt is its own metered
        # request, and the SDK metric's token fields (observability for the
        # unbilled homepage demo only) restart from zero with it.
        self._set_token_usage(input_tokens=0, output_tokens=0)
        request = current_tts_request()
        # Bound before the try so the no-audio diagnostic in the `ClientError`
        # handler can read them: a 400 refusal is raised by the request itself,
        # before either is assigned inside the block.
        request_id = ""
        finish_reasons: list[str] = []
        try:
            config_kwargs: dict[str, Any] = {
                "response_modalities": ["AUDIO"],
                "speech_config": types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._tts._opts.voice_name,
                        )
                    )
                ),
            }
            input_text = self._input_text
            if gemini_reads_transcript(self._tts._opts.model):
                # A transcript model (Flash-Lite) speaks whatever it
                # is sent, so it gets the sentence alone -- whatever
                # instruction the plugin constructor filled in (it defaults
                # one when none is passed) is deliberately NOT spliced in.
                # Raw PCM is asked for through ``responseFormat``, which the
                # pinned google-genai (>= 2.13 via livekit-plugins-google
                # 1.8.1) does not model on ``GenerateContentConfig``; its
                # per-request ``HttpOptions.extra_body`` is merged into the
                # JSON body by the SDK, and the worker image's build probe
                # asserts both fields exist.
                config_kwargs["http_options"] = types.HttpOptions(
                    extra_body={"generationConfig": gemini_raw_audio_generation_config(self._tts._opts.model)}
                )
            elif self._tts._opts.instructions is not None:
                input_text = f'{self._tts._opts.instructions}:\n"{input_text}"'
            config = types.GenerateContentConfig(**config_kwargs)

            response = await self._tts._client.aio.models.generate_content_stream(
                model=self._tts._opts.model,
                contents=input_text,
                config=config,
            )

            # Every call streams audio to the caller as it arrives. Each
            # valid cumulative usage report replaces the request's measured
            # usage as it arrives, so a turn the caller interrupts is charged
            # what Google reported up to that moment.
            usage = GeminiTTSUsageTracker()
            request_id = utils.shortuuid()
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type="audio/pcm",
            )
            async for chunk in response:
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata is not None:
                    # Cumulative counts: the latest VALID report is the
                    # request's measured usage. An incomplete report never
                    # replaces it (and is logged at the end if it was last).
                    usage.record(metadata)
                for candidate in chunk.candidates or []:
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason is not None:
                        finish_reasons.append(str(finish_reason))
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
                            # The provider's own malformed answer: refused, not lost.
                            mark_provider_error()
                            raise APIStatusError(
                                "gemini tts: audio payload is invalid",
                                status_code=502,
                                retryable=False,
                            )
                        output_emitter.push(bytes(data))
                        audio_bytes_emitted += len(data)
                # Usage reaches the meter only once this request has queued
                # audio, after this chunk's own audio: a request that never
                # produced audio is a refusal ("no audio"), and a cancel
                # before its first audio must not charge its prompt tokens.
                tokens = usage.latest_tokens
                if tokens is not None and request is not None and audio_bytes_emitted > 0:
                    request.report_usage(dict(zip(_REPORTED_DIMENSIONS, tokens)))

            # A complete response that produced no audio for a
            # non-empty request is a failed turn whatever the usage trailer
            # says, so it is decided HERE -- before the trailer is consulted
            # -- rather than being discovered as a billing-evidence error.
            # Until this check existed, the only report of the silent refusal
            # was `GeminiTTSUsageError: usage metadata omitted exact generated
            # audio token count`, which named the accounting symptom and not
            # the media failure, and which fired only on the metered path: an
            # unmetered demo turn with no audio simply returned and let
            # LiveKit raise a generic "no audio frames were pushed" that
            # echoes the customer's own text into the logs.
            #
            # `self._input_text.strip()` mirrors LiveKit's own guard in
            # `_main_task`: a request that asked for nothing is legitimately
            # silent and must still return normally.
            #
            # This deliberately checks `audio_bytes_emitted` (was anything
            # handed to `push()`), NOT `output_emitter.pushed_duration()` (has
            # a frame already reached the downstream channel). `AudioEmitter`
            # holds back a small tail of audio and flushes it only on
            # `end_input()`, so a short reply can legitimately have queued
            # audio with `pushed_duration() == 0` right here.
            # `pushed_duration()` remains the right test in `_retryable`,
            # where an actual retry -- not a graceful `end_input()` -- is on
            # the other side of the decision.
            if audio_bytes_emitted == 0 and self._input_text.strip():
                # EXACTLY ONE retry, on every no-audio path including the
                # metered missing-trailer one that used to fail closed here.
                #
                # Retrying is unconditionally safe in this branch and only in
                # this branch: nothing was pushed, so there is nothing for a
                # fresh attempt to duplicate -- the same property the module
                # docstring uses to justify refusing retries everywhere else.
                # And it is worth doing, because on the 2.5 models this
                # refusal is substantially random per attempt on short
                # payloads (measured 5-55%), so one more identical request
                # rescues most of those turns. A dead turn is the alternative,
                # and a mid-call reply -- unlike the opening line -- has no
                # fallback behind it. The error kills the whole tts_node, not
                # one sentence, so the caller loses the rest of the reply.
                #
                # Bounded at one because the cost is a whole extra synthesis
                # (3-6 s) and the deterministic class of this failure will
                # spend it to fail again; on an opening turn those seconds are
                # silence before the fallback can speak. The counter is the
                # only state involved -- the request is byte-identical to the
                # first, with the instruction the TTS was built with.
                self._no_audio_attempts += 1
                raise self._no_audio_error(
                    request_id=request_id,
                    finish_reasons=finish_reasons,
                    retryable=self._no_audio_attempts == 1,
                )

            try:
                usage.require_latest_complete()
            except GeminiTTSUsageError as exc:
                # Reaching this branch means audio WAS queued (or the request
                # asked for nothing): the zero-audio case is decided above and
                # raised as a media failure, so this is only ever "the audio
                # played but the final usage trailer was missing or
                # incomplete". Raising would make LiveKit re-synthesize speech
                # the caller already heard. The request stays settled at the
                # latest VALID cumulative report Gemini sent (zero when it sent
                # none) -- a provider measurement, never an estimate. Watch
                # this event's RATE for a systemic trailer change.
                logger.warning(
                    "gemini_tts_usage_missing: %s",
                    exc,
                    extra={
                        "request_id": request_id,
                        "audio_bytes_emitted": audio_bytes_emitted,
                        "characters_requested": len(self._input_text),
                    },
                )
            token_usage = usage.latest_tokens
            if token_usage is not None:
                # Observability only (the homepage demo's spend ceiling reads
                # SDK metrics); a paid call settles from the meter.
                self._set_token_usage(input_tokens=token_usage[0], output_tokens=token_usage[1])

        except (APIStatusError, TTSMeterFrozen, GeminiTTSRuntimeError):
            raise
        except ClientError as exc:
            # Gemini refused the request: 400/500 responses are not charged.
            mark_provider_error()
            # The louder half of the same failure: Gemini refuses the request
            # outright with "Model tried to generate text, but it should only
            # be used for TTS". Same event as the silent no-audio response
            # above -- the line did not play -- so it gets the same diagnostic
            # and the same fail-fast handling. Guarded on nothing having
            # been pushed, so a mid-stream client error after real audio still
            # takes the ordinary `_retryable` path below and cannot be
            # re-synthesized over speech the caller is already hearing.
            if audio_bytes_emitted == 0 and _is_text_instead_of_audio(exc):
                raise self._no_audio_error(
                    request_id=request_id,
                    finish_reasons=finish_reasons,
                    provider_error=exc,
                ) from exc
            raise APIStatusError(
                "gemini tts: client error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=_retryable(
                    exc.code in {429, 499},
                    audio_delivered=output_emitter.pushed_duration() > 0,
                    exc=exc,
                ),
            ) from exc
        except ServerError as exc:
            # Gemini refused the request: 400/500 responses are not charged.
            mark_provider_error()
            raise APIStatusError(
                "gemini tts: server error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=_retryable(
                    True, audio_delivered=output_emitter.pushed_duration() > 0, exc=exc
                ),
            ) from exc
        except APIError as exc:
            mark_provider_error()
            raise APIStatusError(
                "gemini tts: api error",
                status_code=exc.code,
                body=f"{exc.message} {exc.status}",
                retryable=_retryable(
                    True, audio_delivered=output_emitter.pushed_duration() > 0, exc=exc
                ),
            ) from exc
        except Exception as exc:
            raise APIConnectionError(
                f"gemini tts: error generating speech {str(exc)}",
                retryable=_retryable(
                    True, audio_delivered=output_emitter.pushed_duration() > 0, exc=exc
                ),
            ) from exc

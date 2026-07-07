# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import queue
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .barge_in import BargeInConfig, BargeInMonitor
from .frames import AudioFrame, TextChunk, TranscriptEvent
from .metrics import CallMetrics, MetricEvent
from .recorder import StreamingCallRecorder
from .semantic_endpointing import SemanticEndpointDetector
from .stt import StreamingSTT, start_stt_stream
from .transport import CancellationToken, PacedRtpSender
from .tts import StreamingTTS, speakable_chunks
from .turn_detection import DetectedTurn, RmsTurnDetector


class StreamingLLM(Protocol):
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        ...


class FallbackLLM(Protocol):
    def generate_reply(self, messages: list[dict[str, str]]) -> str:
        ...


DEFAULT_RECOVERY_UTTERANCES: dict[str, str] = {
    "et": "Vabandust, mul tekkis hetkeline tehniline tõrge. Kas te saaksite palun korrata?",
    "en": "Sorry, I hit a brief technical issue. Could you please repeat that?",
}


def default_recovery_utterance(language: str | None = None) -> str:
    key = (language or "et").strip().lower().replace("_", "-").split("-")[0]
    return DEFAULT_RECOVERY_UTTERANCES.get(key, DEFAULT_RECOVERY_UTTERANCES["et"])


@dataclass(frozen=True)
class LLMFallbackConfig:
    """Fallback chain that keeps a call alive when the streaming LLM fails.

    Legs, in order: retry the primary stream once with a short first-phrase
    budget, then a non-streaming fallback-model reply spoken as one phrase,
    then a canned language-aware recovery utterance.
    """

    retry_primary: bool = True
    retry_budget_seconds: float = 2.0
    fallback_llm: FallbackLLM | None = None
    fallback_model_id: str = ""
    recovery_utterance: str = ""
    language: str = "et"

    def resolved_recovery_utterance(self) -> str:
        return self.recovery_utterance.strip() or default_recovery_utterance(self.language)


@dataclass
class CapturedTurn:
    detected: DetectedTurn
    transcript_events: list[TranscriptEvent]

    @property
    def frames(self) -> list[AudioFrame]:
        return self.detected.frames

    @property
    def start_ms(self) -> int:
        return self.detected.start_ms

    @property
    def end_ms(self) -> int:
        return self.detected.end_ms

    @property
    def speech_ms(self) -> int:
        return self.detected.speech_ms

    @property
    def pcm16(self) -> bytes:
        return self.detected.pcm16


@dataclass
class PipelineResult:
    transcript: str
    assistant_text: str
    latency: dict[str, float | None]
    interrupted: bool = False
    assistant_spoken_text: str = ""
    llm_fallback_leg: str = ""


@dataclass
class _PhraseQueueItem:
    kind: str
    phrase_index: int = 0
    phrase: str = ""
    error: BaseException | None = None


class RealtimeMediaPipeline:
    """Pipecat-style OpenPhonex-owned streaming pipeline."""

    def __init__(
        self,
        *,
        stt: StreamingSTT,
        llm: StreamingLLM,
        tts: StreamingTTS,
        rtp_sender: PacedRtpSender,
        recorder: StreamingCallRecorder | None = None,
        detector: RmsTurnDetector | None = None,
        metrics: CallMetrics | None = None,
        barge_in_frame_reader: Callable[[], AudioFrame | None] | None = None,
        barge_in_config: BargeInConfig | None = None,
        max_pending_phrases: int = 4,
        semantic_endpointing: SemanticEndpointDetector | None = None,
        llm_fallback: LLMFallbackConfig | None = None,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.rtp_sender = rtp_sender
        self.recorder = recorder
        self.detector = detector or RmsTurnDetector()
        self.metrics = metrics or CallMetrics()
        self.conversation: list[dict[str, str]] = []
        self.output_cancel = CancellationToken()
        self.barge_in_frame_reader = barge_in_frame_reader
        self.barge_in_config = barge_in_config
        self.semantic_endpointing = semantic_endpointing or SemanticEndpointDetector()
        self.llm_fallback = llm_fallback or LLMFallbackConfig()
        self._pending_barge_in_frames: list[AudioFrame] = []
        self._assistant_interrupted = False
        self._turn_index = 0
        self._current_agent_turn_frames: list[AudioFrame] = []
        self._assistant_context_parts: list[str] = []
        self.max_pending_phrases = max(1, int(max_pending_phrases))
        self._caller_speech_active = False

    def consume_inbound_until_turn(self, frames: Iterable[AudioFrame]) -> CapturedTurn | None:
        turn_index = self._turn_index + 1
        self.metrics.begin_turn(turn_index)
        transcript_events: list[TranscriptEvent] = []
        pending_turns: list[DetectedTurn] = []
        last_candidate_endpoint = None
        stt_session = start_stt_stream(self.stt)
        self.metrics.mark("stt_stream_start")
        if not self.metrics.first("stt_request_start", turn=turn_index):
            self.metrics.mark("stt_request_start", mode="stream")
        try:
            for frame in self._frames_with_pending_barge_in(frames):
                self._observe_caller_frame(frame)
                if self.recorder:
                    self.recorder.write_caller(frame)
                frame_events = list(stt_session.send_audio(frame))
                for event in frame_events:
                    transcript_events.append(event)
                    self._observe_transcript_event(event)
                turn = self.detector.feed(frame)
                endpoint_source = "rms_silence"
                stt_endpoint_source = self._endpoint_source_from_stt_events(frame_events)
                if not turn and stt_endpoint_source:
                    turn = self.detector.force_endpoint()
                    endpoint_source = stt_endpoint_source
                if turn:
                    candidate_endpoint = self.metrics.mark(
                        "candidate_endpoint_detected",
                        end_ms=turn.end_ms,
                        speech_ms=turn.speech_ms,
                        source=endpoint_source,
                    )
                    self._mark_caller_speech_ended(turn, candidate_endpoint)
                    last_candidate_endpoint = candidate_endpoint
                    self._caller_speech_active = False
                    pending_turns.append(turn)
                    if (
                        endpoint_source == "rms_silence"
                        and not self._has_transcript_text(transcript_events)
                        and bool(getattr(stt_session, "skip_finalization_without_transcript", False))
                    ):
                        self.metrics.mark(
                            "stt_endpoint_without_transcript",
                            end_ms=turn.end_ms,
                            speech_ms=turn.speech_ms,
                            source=endpoint_source,
                            pending_turns=len(pending_turns),
                            finalization_skipped=True,
                        )
                        continue
                    endpoint_events = list(stt_session.end_input(turn))
                    transcript_events.extend(endpoint_events)
                    if self._should_accept_endpointed_transcript(endpoint_source, transcript_events):
                        self.metrics.mark(
                            "user_endpoint_detected",
                            at=candidate_endpoint.at,
                            end_ms=turn.end_ms,
                            speech_ms=turn.speech_ms,
                            source=endpoint_source,
                        )
                        for event in endpoint_events:
                            self._observe_transcript_event(event)
                        return CapturedTurn(self._combine_detected_turns(pending_turns), transcript_events)
                    if self._last_final_transcript_event(transcript_events):
                        self.metrics.mark(
                            "stt_endpoint_incomplete_transcript",
                            end_ms=turn.end_ms,
                            speech_ms=turn.speech_ms,
                            source=endpoint_source,
                            pending_turns=len(pending_turns),
                            text_len=len(self._final_transcript(transcript_events)),
                        )
                        continue
                    self.metrics.mark(
                        "stt_endpoint_without_transcript",
                        end_ms=turn.end_ms,
                        speech_ms=turn.speech_ms,
                        source=endpoint_source,
                        pending_turns=len(pending_turns),
                    )
            for event in stt_session.end_input(None):
                transcript_events.append(event)
                self._observe_transcript_event(event)
            self._caller_speech_active = False
            if pending_turns and self._final_transcript(transcript_events):
                if not self.metrics.first("user_endpoint_detected", turn=turn_index):
                    final_turn = self._combine_detected_turns(pending_turns)
                    self.metrics.mark(
                        "user_endpoint_detected",
                        at=last_candidate_endpoint.at if last_candidate_endpoint else None,
                        end_ms=final_turn.end_ms,
                        speech_ms=final_turn.speech_ms,
                        source="stream_end_after_pending_transcript",
                    )
                return CapturedTurn(self._combine_detected_turns(pending_turns), transcript_events)
            return None
        finally:
            stt_session.close()

    def process_turn(self, turn: DetectedTurn | CapturedTurn) -> PipelineResult:
        self.output_cancel = CancellationToken()
        self._assistant_interrupted = False
        self._current_agent_turn_frames = []
        self._assistant_context_parts = []
        self._turn_index += 1
        self.metrics.begin_turn(self._turn_index)
        detected_turn, transcript_events = self._split_turn(turn)
        if self.recorder:
            self.recorder.write_caller_turn(self._turn_index, detected_turn.frames)
            self.metrics.mark("caller_turn_clip_written", turn_index=self._turn_index, frames=len(detected_turn.frames))
        transcript = self._final_transcript(transcript_events)
        if not transcript:
            if not self.metrics.first("stt_request_start", turn=self._turn_index):
                self.metrics.mark("stt_request_start", mode="turn")
            fallback_events = list(self.stt.transcribe(detected_turn))
            for event in fallback_events:
                self._observe_transcript_event(event)
            transcript = self._final_transcript(fallback_events)
        if not transcript:
            return PipelineResult("", "", self.metrics.latency_breakdown(turn=self._turn_index))

        self.conversation.append({"role": "user", "content": transcript})
        self.metrics.mark("llm_request_start")
        producer_error, assistant_parts = self._speak_streamed_reply(leg="primary")
        llm_fallback_leg = ""
        if producer_error is not None and not self.output_cancel.cancelled:
            self.metrics.mark(
                "llm_primary_stream_failed",
                error=producer_error.__class__.__name__,
                message=str(producer_error)[:200],
            )
            llm_fallback_leg, fallback_parts = self._run_llm_fallback_chain()
            if fallback_parts:
                assistant_parts = fallback_parts
            self.metrics.mark("llm_fallback_used", leg=llm_fallback_leg)

        assistant_text = " ".join("".join(assistant_parts).split())
        assistant_spoken_text = " ".join(" ".join(self._assistant_context_parts).split())
        if self.recorder and self._current_agent_turn_frames:
            self.recorder.write_agent_turn(self._turn_index, self._current_agent_turn_frames)
            self.metrics.mark("agent_turn_clip_written", turn_index=self._turn_index, frames=len(self._current_agent_turn_frames))
        if assistant_spoken_text:
            self.conversation.append({"role": "assistant", "content": assistant_spoken_text})
        if assistant_text != assistant_spoken_text:
            self.metrics.mark(
                "assistant_context_truncated",
                generated_len=len(assistant_text),
                spoken_len=len(assistant_spoken_text),
                interrupted=self._assistant_interrupted,
            )
        return PipelineResult(
            transcript,
            assistant_text,
            self.metrics.latency_breakdown(turn=self._turn_index),
            interrupted=self._assistant_interrupted,
            assistant_spoken_text=assistant_spoken_text,
            llm_fallback_leg=llm_fallback_leg,
        )

    def speak_text(self, text: str, *, label: str = "scripted_utterance") -> dict[str, object]:
        """Speak a fixed, non-LLM utterance (e.g. the outbound opening greeting).

        The utterance is spoken through the normal TTS/RTP path with barge-in
        monitoring, and joins the conversation context as an assistant message
        when it completes so the LLM knows what the caller already heard.
        """

        phrase = " ".join(str(text or "").split())
        if not phrase:
            return {"text": "", "spoken": False, "interrupted": False}
        self.output_cancel = CancellationToken()
        self._assistant_interrupted = False
        self._current_agent_turn_frames = []
        self._assistant_context_parts = []
        self.metrics.mark("scripted_utterance_started", label=label, text_len=len(phrase))
        spoken = self._speak_phrase(phrase, phrase_index=1)
        if spoken:
            self.conversation.append({"role": "assistant", "content": phrase})
        self.metrics.mark(
            "scripted_utterance_finished",
            label=label,
            spoken=spoken,
            interrupted=self._assistant_interrupted,
        )
        return {"text": phrase, "spoken": spoken, "interrupted": self._assistant_interrupted}

    def _llm_chunks(self, assistant_parts: list[str], abort: threading.Event | None = None) -> Iterable[TextChunk]:
        chunk_stream = iter(self.llm.stream_reply(self.conversation[-10:]))
        stream_exhausted = False
        try:
            while True:
                if self.output_cancel.cancelled or (abort is not None and abort.is_set()):
                    break
                try:
                    chunk = next(chunk_stream)
                except StopIteration:
                    stream_exhausted = True
                    break
                if self.output_cancel.cancelled:
                    break
                if chunk.text and not self.metrics.first("first_llm_chunk", turn=self._turn_index):
                    self.metrics.mark("first_llm_chunk", text_len=len(chunk.text))
                assistant_parts.append(chunk.text)
                yield chunk
        finally:
            close = getattr(chunk_stream, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
                if not stream_exhausted or self.output_cancel.cancelled:
                    self.metrics.mark(
                        "llm_chunk_stream_closed",
                        cancelled=self.output_cancel.cancelled,
                        stream_exhausted=stream_exhausted,
                    )
            elif self.output_cancel.cancelled:
                self.metrics.mark("llm_chunk_stream_close_unavailable")

    def _speak_streamed_reply(
        self,
        *,
        leg: str,
        first_item_budget_seconds: float | None = None,
    ) -> tuple[BaseException | None, list[str]]:
        """Stream one LLM reply to TTS/RTP; return (producer error, generated parts)."""

        assistant_parts: list[str] = []
        abort = threading.Event()
        phrase_queue: queue.Queue[_PhraseQueueItem] = queue.Queue(maxsize=self.max_pending_phrases)
        phrase_thread = threading.Thread(
            target=self._produce_speakable_phrases,
            args=(self._llm_chunks(assistant_parts, abort), phrase_queue, abort),
            name=f"openphonex-llm-phrase-producer-{leg}",
            daemon=True,
        )
        phrase_thread.start()
        producer_error: BaseException | None = None
        deadline = None if first_item_budget_seconds is None else time.monotonic() + float(first_item_budget_seconds)
        while True:
            try:
                item = phrase_queue.get(timeout=0.05 if deadline is not None else 0.2)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    producer_error = TimeoutError(f"llm {leg} produced no phrase within {first_item_budget_seconds}s")
                    self.metrics.mark(
                        "llm_stream_first_phrase_budget_exceeded",
                        leg=leg,
                        budget_seconds=first_item_budget_seconds,
                    )
                    abort.set()
                    break
                if phrase_thread.is_alive():
                    continue
                self.metrics.mark("llm_phrase_queue_finished_without_done")
                break
            deadline = None
            if item.kind == "done":
                break
            if item.kind == "error":
                producer_error = item.error
                break
            if self.output_cancel.cancelled:
                break
            if item.kind != "phrase":
                continue
            if not self._speak_phrase(item.phrase, phrase_index=item.phrase_index):
                break
        if self.output_cancel.cancelled:
            self.metrics.mark("llm_phrase_consumer_cancelled")
        phrase_thread.join(timeout=0.5)
        if phrase_thread.is_alive():
            self.metrics.mark("llm_phrase_producer_still_running")
        return producer_error, assistant_parts

    def _run_llm_fallback_chain(self) -> tuple[str, list[str]]:
        """Keep the turn (and the call) alive after a primary streaming-LLM failure."""

        config = self.llm_fallback
        if config.retry_primary and not self.output_cancel.cancelled:
            self.metrics.mark(
                "llm_fallback_leg_started",
                leg="primary_retry",
                budget_seconds=config.retry_budget_seconds,
            )
            retry_error, retry_parts = self._speak_streamed_reply(
                leg="primary_retry",
                first_item_budget_seconds=config.retry_budget_seconds,
            )
            if retry_error is None:
                self.metrics.mark("llm_fallback_leg_completed", leg="primary_retry")
                return "primary_retry", retry_parts
            self.metrics.mark(
                "llm_fallback_leg_failed",
                leg="primary_retry",
                error=retry_error.__class__.__name__,
                message=str(retry_error)[:200],
            )
        if config.fallback_llm is not None and not self.output_cancel.cancelled:
            self.metrics.mark("llm_fallback_leg_started", leg="fallback_model", model_id=config.fallback_model_id)
            fallback_text = ""
            try:
                fallback_text = " ".join(str(config.fallback_llm.generate_reply(self.conversation[-10:])).split())
            except Exception as exc:
                self.metrics.mark(
                    "llm_fallback_leg_failed",
                    leg="fallback_model",
                    error=exc.__class__.__name__,
                    message=str(exc)[:200],
                )
            if fallback_text:
                self.metrics.mark("llm_fallback_text_ready", leg="fallback_model", text_len=len(fallback_text))
                spoken = self._speak_phrase(fallback_text, phrase_index=1)
                self.metrics.mark("llm_fallback_leg_completed", leg="fallback_model", spoken=spoken)
                return "fallback_model", [fallback_text]
        if self.output_cancel.cancelled:
            return "cancelled", []
        recovery = config.resolved_recovery_utterance()
        self.metrics.mark("llm_fallback_leg_started", leg="recovery_utterance", text_len=len(recovery))
        spoken = self._speak_phrase(recovery, phrase_index=1)
        self.metrics.mark("llm_fallback_leg_completed", leg="recovery_utterance", spoken=spoken)
        return "recovery_utterance", [recovery]

    def _produce_speakable_phrases(
        self,
        chunks: Iterable[TextChunk],
        phrase_queue: queue.Queue[_PhraseQueueItem],
        abort: threading.Event | None = None,
    ) -> None:
        phrase_count = 0
        phrase_stream = iter(speakable_chunks(chunks))
        try:
            for phrase in phrase_stream:
                if self.output_cancel.cancelled or (abort is not None and abort.is_set()):
                    break
                phrase_count += 1
                self.metrics.mark("speakable_phrase_ready", phrase_index=phrase_count, text_len=len(phrase))
                if not self.metrics.first("first_speakable_phrase", turn=self._turn_index):
                    self.metrics.mark("first_speakable_phrase", phrase_index=phrase_count, text_len=len(phrase))
                queued = self._put_phrase_queue_item(
                    phrase_queue,
                    _PhraseQueueItem(kind="phrase", phrase_index=phrase_count, phrase=phrase),
                    drop_on_cancel=True,
                    abort=abort,
                )
                if not queued:
                    break
            if self.output_cancel.cancelled or (abort is not None and abort.is_set()):
                self.metrics.mark("llm_phrase_producer_cancelled", phrases=phrase_count)
            else:
                self.metrics.mark("llm_stream_completed", phrases=phrase_count)
            self._put_phrase_queue_item(phrase_queue, _PhraseQueueItem(kind="done"), abort=abort)
        except BaseException as exc:
            self.metrics.mark("llm_stream_error", error=exc.__class__.__name__)
            self._put_phrase_queue_item(phrase_queue, _PhraseQueueItem(kind="error", error=exc), abort=abort)
        finally:
            close = getattr(phrase_stream, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    def _put_phrase_queue_item(
        self,
        phrase_queue: queue.Queue[_PhraseQueueItem],
        item: _PhraseQueueItem,
        *,
        drop_on_cancel: bool = False,
        abort: threading.Event | None = None,
    ) -> bool:
        backpressure_marked = False
        while True:
            if abort is not None and abort.is_set():
                return False
            if drop_on_cancel and self.output_cancel.cancelled:
                return False
            try:
                phrase_queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                if not backpressure_marked:
                    self.metrics.mark(
                        "llm_phrase_queue_full",
                        kind=item.kind,
                        phrase_index=item.phrase_index,
                        max_pending_phrases=self.max_pending_phrases,
                    )
                    backpressure_marked = True
                if self.output_cancel.cancelled:
                    return False

    def _speak_phrase(self, phrase: str, *, phrase_index: int) -> bool:
        self.metrics.mark("tts_request_start", text_len=len(phrase), phrase_index=phrase_index)
        self.metrics.mark("tts_phrase_request_start", text_len=len(phrase), phrase_index=phrase_index)
        monitor: BargeInMonitor | None = None
        if self.barge_in_frame_reader:
            monitor = BargeInMonitor(
                read_frame=self.barge_in_frame_reader,
                cancellation=self.output_cancel,
                metrics=self.metrics,
                config=self.barge_in_config,
            )
            monitor.start()

        def pcm_chunks() -> Iterable[bytes]:
            first = True
            frame_stream = iter(self.tts.stream(phrase))
            try:
                while True:
                    if self.output_cancel.cancelled:
                        break
                    try:
                        frame = next(frame_stream)
                    except StopIteration:
                        break
                    if first:
                        self.metrics.mark("tts_phrase_first_audio_byte", bytes=len(frame.pcm16), phrase_index=phrase_index)
                        if not self.metrics.first("first_tts_audio_byte", turn=self._turn_index):
                            self.metrics.mark("first_tts_audio_byte", bytes=len(frame.pcm16), phrase_index=phrase_index)
                        first = False
                    if self.recorder:
                        self.recorder.write_agent(frame)
                        self._current_agent_turn_frames.append(frame)
                    yield frame.pcm16
            finally:
                close = getattr(frame_stream, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()

        try:
            packets = self.rtp_sender.send_pcm16_stream(pcm_chunks(), cancellation=self.output_cancel, metrics=self.metrics)
            self.metrics.mark("tts_phrase_rtp_done", phrase_index=phrase_index, packets=packets, cancelled=self.output_cancel.cancelled)
            if packets and not self.output_cancel.cancelled:
                self._assistant_context_parts.append(phrase)
                self.metrics.mark("assistant_context_phrase_completed", phrase_index=phrase_index, text_len=len(phrase))
        finally:
            if monitor:
                monitor.stop()
                captured = monitor.captured_frames()
                if monitor.interrupted and captured:
                    self._pending_barge_in_frames.extend(captured)
                    self.metrics.mark("barge_in_frames_buffered", frames=len(captured))
                if monitor.interrupted:
                    self._assistant_interrupted = True
                    self.metrics.mark("assistant_turn_interrupted", phrase_index=phrase_index, phrase_len=len(phrase))
        return not self.output_cancel.cancelled

    def barge_in(self) -> None:
        self.output_cancel.cancel()
        self.metrics.mark("barge_in_output_cancelled")

    def _frames_with_pending_barge_in(self, frames: Iterable[AudioFrame]) -> Iterable[AudioFrame]:
        if self._pending_barge_in_frames:
            pending = list(self._pending_barge_in_frames)
            self._pending_barge_in_frames.clear()
            self.metrics.mark("barge_in_pending_frames_replayed", frames=len(pending))
            yield from pending
        yield from frames

    def _split_turn(self, turn: DetectedTurn | CapturedTurn) -> tuple[DetectedTurn, list[TranscriptEvent]]:
        if isinstance(turn, CapturedTurn):
            return turn.detected, list(turn.transcript_events)
        return turn, []

    def _observe_caller_frame(self, frame: AudioFrame) -> None:
        if frame.rms < self.detector.config.speech_rms:
            return
        if self._caller_speech_active:
            return
        self._caller_speech_active = True
        self.metrics.mark(
            "caller_speech_started",
            rms=round(frame.rms, 3),
            speech_rms=self.detector.config.speech_rms,
            duration_ms=frame.duration_ms,
        )

    def _mark_caller_speech_ended(self, turn: DetectedTurn, candidate: MetricEvent) -> None:
        """Anchor the true media-time end of caller speech to the wall clock.

        The endpoint detector stamps the candidate event only after the silence
        hold expires, so the caller actually stopped speaking earlier. Shift the
        candidate stamp back by the recorded media-time silence tail
        (``end_ms - speech_end_ms``) instead of trusting the stamp time.
        """

        speech_end_ms = turn.speech_end_ms if turn.speech_end_ms is not None else turn.end_ms
        tail_ms = max(0, turn.end_ms - speech_end_ms)
        self.metrics.mark(
            "caller_speech_ended",
            at=candidate.at - tail_ms / 1000.0,
            end_ms=speech_end_ms,
            detector_end_ms=turn.end_ms,
            silence_tail_ms=tail_ms,
        )

    def _observe_transcript_event(self, event: TranscriptEvent) -> None:
        if event.text and not self.metrics.first("first_transcript_delta", turn=self.metrics.current_turn):
            self.metrics.mark("first_transcript_delta", text_len=len(event.text), final=event.is_final)
        if event.is_final:
            transcript = event.text.strip()
            self.metrics.mark("final_transcript", text_len=len(transcript), language=event.language or "")

    def _endpoint_source_from_stt_events(self, events: Iterable[TranscriptEvent]) -> str | None:
        for event in events:
            if not event.text.strip() or not event.is_final:
                continue
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            if metadata.get("speech_final") or metadata.get("utterance_end") or metadata.get("endpoint"):
                return "stt_finality"
            if self.semantic_endpointing.should_endpoint(event):
                self.metrics.mark("semantic_endpoint_detected", text_len=len(event.text.strip()))
                return "semantic_final_text"
        return None

    def _should_accept_endpointed_transcript(self, endpoint_source: str, events: Iterable[TranscriptEvent]) -> bool:
        event = self._last_final_transcript_event(events)
        if not event:
            return False
        if self._is_explicit_semantic_endpoint(event):
            return True
        if self.semantic_endpointing.config.enabled and not self._provider_final_text_is_complete(event):
            return False
        if endpoint_source in {"stt_finality", "semantic_final_text"}:
            return True
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if metadata.get("speech_final") or metadata.get("utterance_end") or metadata.get("endpoint"):
            return True
        if not self.semantic_endpointing.config.enabled:
            return True
        return self.semantic_endpointing.should_endpoint(event)

    def _has_transcript_text(self, events: Iterable[TranscriptEvent]) -> bool:
        return any(bool(event.text.strip()) for event in events)

    def _is_explicit_semantic_endpoint(self, event: TranscriptEvent) -> bool:
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        return metadata.get("semantic_endpoint") is True

    def _provider_final_text_is_complete(self, event: TranscriptEvent) -> bool:
        if self.semantic_endpointing.should_endpoint(event):
            return True
        text = " ".join(event.text.strip().split())
        if not text:
            return False
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if metadata.get("semantic_endpoint") is False:
            return False
        if text.endswith(","):
            return False
        min_chars = self.semantic_endpointing.config.min_chars
        min_words = max(3, self.semantic_endpointing.config.min_words)
        return len(text) >= min_chars and len(text.split()) >= min_words

    def _last_final_transcript_event(self, events: Iterable[TranscriptEvent]) -> TranscriptEvent | None:
        last: TranscriptEvent | None = None
        for event in events:
            if event.is_final and event.text.strip():
                last = event
        return last

    def _final_transcript(self, events: Iterable[TranscriptEvent]) -> str:
        transcript = ""
        for event in events:
            if event.is_final and event.text.strip():
                transcript = event.text.strip()
        return transcript

    def _combine_detected_turns(self, turns: Iterable[DetectedTurn]) -> DetectedTurn:
        collected = list(turns)
        if not collected:
            return DetectedTurn(frames=[], start_ms=0, end_ms=0, speech_ms=0)
        frames: list[AudioFrame] = []
        for turn in collected:
            frames.extend(turn.frames)
        return DetectedTurn(
            frames=frames,
            start_ms=collected[0].start_ms,
            end_ms=collected[-1].end_ms,
            speech_ms=sum(turn.speech_ms for turn in collected),
            speech_end_ms=collected[-1].speech_end_ms,
        )

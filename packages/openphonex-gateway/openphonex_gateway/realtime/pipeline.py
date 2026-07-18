# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import json
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
from .tools import DEFAULT_MAX_TOOL_ITERATIONS, ToolCall
from .transport import CancellationToken, PacedRtpSender
from .tts import StreamingTTS, speakable_chunks
from .turn_detection import DetectedTurn, RmsTurnDetector


class StreamingLLM(Protocol):
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        ...


class ToolCallingStreamingLLM(Protocol):
    """Tools-aware StreamingLLM variant.

    Adapters that natively support function calling set ``supports_tools =
    True`` and accept the optional ``tools`` argument (a list of function
    declarations). Tool calls surface as empty-text chunks carrying a
    ``tool_call`` metadata dict (``{"name": ..., "arguments": {...}}``). The
    pipeline only ever passes ``tools=`` to adapters that declare support, so
    prompt-only adapters keep the original single-argument call unchanged.
    """

    supports_tools: bool

    def stream_reply(
        self, messages: list[dict[str, str]], tools: list[dict[str, object]] | None = None
    ) -> Iterable[TextChunk]:
        ...


class ToolExecutor(Protocol):
    def execute(self, call: ToolCall) -> dict[str, object]:
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


# How far (ms) the freshest interim transcript's audio coverage may lag behind
# the detected end of speech and still be promoted to a synthetic final when
# accept_interim_on_rms_endpoint is on. Deepgram interims trail real-time by
# well under a second and keep updating through the silence hold, so 2000ms
# only rejects genuinely stale interims (e.g. a stalled provider stream) while
# never false-rejecting a healthy one.
INTERIM_ENDPOINT_MAX_LAG_MS = 2000.0


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
    # The model invoked end_call this turn — the worker should hang
    # the primary channel up after this result.
    end_call: bool = False
    tool_calls_executed: int = 0


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
        accept_interim_on_rms_endpoint: bool = False,
        tools: list[dict[str, object]] | None = None,
        tool_executor: ToolExecutor | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
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
        # Opt-in latency cut (REALTIME_ACCEPT_INTERIM_ON_RMS_ENDPOINT): when
        # the local RMS endpointer fires, promote the freshest interim
        # transcript instead of waiting for the provider's Finalize round-trip.
        # See _interim_endpoint_transcript() for the completeness/recency
        # gates that keep a mid-sentence pause from shipping a fragment.
        self.accept_interim_on_rms_endpoint = bool(accept_interim_on_rms_endpoint)
        # Tool calling. Empty tools (the default) is a STRICT no-op:
        # _llm_chunks keeps the original single-argument stream_reply call and
        # the tool loop never runs. Tools only activate when the agent has
        # function declarations AND the adapter declares supports_tools.
        self.tools = [dict(tool) for tool in (tools or [])]
        self.tool_executor = tool_executor
        self.max_tool_iterations = max(1, int(max_tool_iterations))
        self.end_call_requested = False
        self._tools_suppressed = False
        self._tool_calls_executed_this_turn = 0
        # Codex round 3 (R1): monotonic per-turn generation id. Tool calls are
        # collected into a per-stream LOCAL list (never shared mutable state) and
        # only ever executed for the SUCCESSFUL PRIMARY stream of the CURRENT
        # turn. The generation guard means a timed-out/late producer thread from a
        # prior turn can never contribute a tool call that fires in a later turn.
        self._tool_call_generation = 0

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
                    if endpoint_source == "rms_silence" and self.accept_interim_on_rms_endpoint:
                        provisional = self._interim_endpoint_transcript(transcript_events, turn)
                        if provisional is not None:
                            transcript_events.append(provisional)
                            self._observe_transcript_event(provisional)
                            self.metrics.mark(
                                "stt_interim_promoted_to_final",
                                text_len=len(provisional.text),
                                source=endpoint_source,
                            )
                            self.metrics.mark(
                                "user_endpoint_detected",
                                at=candidate_endpoint.at,
                                end_ms=turn.end_ms,
                                speech_ms=turn.speech_ms,
                                source="rms_silence_interim",
                            )
                            return CapturedTurn(self._combine_detected_turns(pending_turns), transcript_events)
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
        self.end_call_requested = False
        self._tools_suppressed = False
        self._tool_calls_executed_this_turn = 0
        self._tool_call_generation += 1
        producer_error, assistant_parts, primary_tool_calls = self._speak_streamed_reply(
            leg="primary", collect_tool_calls=True
        )
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
        # Codex round 3 (R1): ONLY the successful primary stream's tool calls
        # execute. A failed primary means the fallback chain (retry / fallback
        # model / recovery utterance) produced a TEXT-ONLY reply — those legs
        # never contribute tool calls, so a stale SMS/transfer can never fire on a
        # later, intent-changed turn. `primary_tool_calls` is a per-stream local
        # list, so a late producer can't contaminate it either.
        if producer_error is None and primary_tool_calls and not self.output_cancel.cancelled:
            tool_leg, tool_parts = self._run_tool_loop(primary_tool_calls)
            llm_fallback_leg = tool_leg or llm_fallback_leg
            assistant_parts = [*assistant_parts, *tool_parts]
        self._tools_suppressed = False

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
            end_call=self.end_call_requested,
            tool_calls_executed=self._tool_calls_executed_this_turn,
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

    def _tools_active(self) -> bool:
        return bool(self.tools) and bool(getattr(self.llm, "supports_tools", False)) and not self._tools_suppressed

    def _context_window(self, limit: int = 10) -> list[dict[str, object]]:
        """The last ``limit`` conversation messages, never splitting a tool turn.

        A fixed ``conversation[-limit:]`` window can slice an assistant
        function-call marker away from its matching ``{"role": "tool"}`` results
        when a single turn produced >= ``limit`` parallel results. The model then
        receives orphaned functionResponse parts (Gemini rejects them / loses the
        continuity tokens). Keep the function-call marker together with its
        results by extending the window backwards over any leading tool results
        to include the assistant marker that owns them."""
        if len(self.conversation) <= limit:
            return self.conversation[:]
        start = len(self.conversation) - limit
        # `_append_tool_exchange` writes the assistant tool_calls marker
        # immediately followed by its contiguous tool results, so walking back
        # over leading tool results lands on (and includes) that marker.
        while start > 0 and str(self.conversation[start].get("role")) == "tool":
            start -= 1
        return self.conversation[start:]

    def _llm_chunks(
        self,
        assistant_parts: list[str],
        tool_calls: list[ToolCall],
        abort: threading.Event | None = None,
        *,
        collect_tool_calls: bool = False,
        generation: int = 0,
    ) -> Iterable[TextChunk]:
        # S5: only DECLARE tools on a COLLECTING leg (primary / tool-follow-up).
        # A retry/fallback leg (collect_tool_calls=False) never executes the tool
        # calls it collects, so declaring tools there only invites a tool-only
        # response that produces no speech; handing it a text-only prompt keeps
        # the retry focused on answering the caller.
        if collect_tool_calls and self._tools_active():
            chunk_stream = iter(self.llm.stream_reply(self._context_window(), tools=self.tools))
        else:
            chunk_stream = iter(self.llm.stream_reply(self._context_window()))
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
                tool_call = chunk.metadata.get("tool_call") if isinstance(chunk.metadata, dict) else None
                if isinstance(tool_call, dict) and tool_call.get("name"):
                    # Function call: never spoken, never part of the assistant
                    # text. It reaches the post-stream tool loop ONLY when this is
                    # a collecting leg (the primary / tool-follow-up stream) AND
                    # the turn has not advanced. A retry/fallback leg
                    # (collect_tool_calls=False) or a late producer from a prior
                    # turn (stale generation) has its call DISCARDED, not executed
                    # (R1).
                    if collect_tool_calls and generation == self._tool_call_generation:
                        arguments = tool_call.get("arguments")
                        tool_calls.append(
                            ToolCall(
                                name=str(tool_call["name"]),
                                arguments=dict(arguments) if isinstance(arguments, dict) else {},
                                id=str(tool_call.get("id") or ""),
                                thought_signature=str(tool_call.get("thought_signature") or ""),
                            )
                        )
                        self.metrics.mark("llm_tool_call_received", tool=str(tool_call["name"]))
                    else:
                        self.metrics.mark(
                            "llm_tool_call_discarded",
                            tool=str(tool_call["name"]),
                            collecting=collect_tool_calls,
                            stale=generation != self._tool_call_generation,
                        )
                    continue
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
        collect_tool_calls: bool = False,
    ) -> tuple[BaseException | None, list[str], list[ToolCall]]:
        """Stream one LLM reply to TTS/RTP.

        Returns ``(producer_error, generated_parts, tool_calls)``. ``tool_calls``
        is a per-call LOCAL list — populated only when ``collect_tool_calls`` is
        set (the primary and tool-follow-up legs) and only while the turn's
        generation is still current (R1). Retry/fallback legs pass
        ``collect_tool_calls=False`` and always return an empty list, so a retry
        yields a text-only reply and cannot smuggle a tool call into execution.
        """

        assistant_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        generation = self._tool_call_generation
        abort = threading.Event()
        phrase_queue: queue.Queue[_PhraseQueueItem] = queue.Queue(maxsize=self.max_pending_phrases)
        phrase_thread = threading.Thread(
            target=self._produce_speakable_phrases,
            args=(
                self._llm_chunks(
                    assistant_parts,
                    tool_calls,
                    abort,
                    collect_tool_calls=collect_tool_calls,
                    generation=generation,
                ),
                phrase_queue,
                abort,
            ),
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
        # Snapshot the collected calls: a producer that outlived the join can
        # only be a budgeted (retry) leg, which never collects — but the copy
        # keeps the returned list immune to any late append regardless (R1).
        return producer_error, assistant_parts, list(tool_calls)

    def _run_tool_loop(self, initial_calls: list[ToolCall]) -> tuple[str, list[str]]:
        """Execute the primary stream's tool calls and re-invoke the LLM, bounded.

        Runs after a clean primary stream that emitted function calls. Each
        iteration executes every collected call, appends the assistant
        tool-call marker + ``{"role": "tool"}`` results to the conversation,
        and streams a follow-up reply — whose own tool calls (collected into a
        fresh local list) drive the next iteration. Bounded by
        ``max_tool_iterations``: on cap the model is told the loop stopped and
        tools are suppressed for one final plain reply, so a call can never spin
        in tool calls forever. ``end_call`` stops the loop immediately (the
        goodbye is spoken as part of executing it).

        Returns ``(llm_fallback_leg, extra_assistant_parts)`` — a failed
        follow-up leg reuses the standard LLM fallback chain (text-only) so the
        caller still hears a reply.
        """

        extra_parts: list[str] = []
        iterations = 0
        pending: list[ToolCall] = list(initial_calls)
        while pending and not self.output_cancel.cancelled:
            calls = pending
            pending = []
            if iterations >= self.max_tool_iterations:
                # Cap reached: report the stop to the model, force a
                # tool-free reply, and never execute this batch.
                self.metrics.mark("tool_loop_cap_reached", pending=len(calls), iterations=iterations)
                self._tools_suppressed = True
                results = [
                    {"ok": False, "tool": call.name, "error": "tool iteration limit reached; answer the caller in plain text"}
                    for call in calls
                ]
            else:
                iterations += 1
                results = [self._execute_tool_call(call) for call in calls]
            self._append_tool_exchange(calls, results)
            if self.end_call_requested:
                return "", extra_parts
            followup_error, followup_parts, followup_tool_calls = self._speak_streamed_reply(
                leg=f"tool_followup_{iterations}", collect_tool_calls=True
            )
            extra_parts.extend(followup_parts)
            if followup_error is not None and not self.output_cancel.cancelled:
                self.metrics.mark(
                    "llm_tool_followup_failed",
                    error=followup_error.__class__.__name__,
                    message=str(followup_error)[:200],
                    iteration=iterations,
                )
                # R1(b): the follow-up's fallback chain (retry/fallback/recovery)
                # is text-only. Any tool call it might surface is neither
                # collected nor executed — the turn ends with a spoken reply.
                leg, fallback_parts = self._run_llm_fallback_chain()
                extra_parts.extend(fallback_parts)
                self.metrics.mark("llm_fallback_used", leg=leg)
                return leg, extra_parts
            if self._tools_suppressed:
                break
            pending = followup_tool_calls
        return "", extra_parts

    def _declared_tool_names(self) -> set[str]:
        return {str(tool.get("name")) for tool in self.tools if isinstance(tool, dict) and tool.get("name")}

    def _execute_tool_call(self, call: ToolCall) -> dict[str, object]:
        self.metrics.mark("tool_call_started", tool=call.name)
        if call.name == "end_call":
            if "end_call" not in self._declared_tool_names():
                # end_call terminates the live call, so it only fires when
                # the agent actually DECLARED it. A hallucinated/undeclared
                # end_call is rejected (surfaced to the model) instead of
                # silently hanging up the caller.
                self.metrics.mark("tool_call_rejected_not_declared", tool=call.name)
                result: dict[str, object] = {
                    "ok": False,
                    "tool": call.name,
                    "error": "end_call is not configured for this agent",
                    "reason_code": "tool_not_configured",
                }
            else:
                # Worker-local: speak the goodbye (if any), then ask the worker
                # to hang up. No outbound authorization needed — nothing is sent
                # or dialed.
                message = " ".join(str(call.arguments.get("message") or "").split())
                if message and not self.output_cancel.cancelled:
                    self._speak_phrase(message, phrase_index=1)
                self.end_call_requested = True
                result = {"ok": True, "tool": call.name, "result": {"status": "call_ending"}}
        elif self.tool_executor is None:
            result = {"ok": False, "tool": call.name, "error": "no tool executor is configured for this call"}
        else:
            try:
                result = self.tool_executor.execute(call)
            except Exception as exc:  # noqa: BLE001 - a tool failure must not abort the turn
                result = {"ok": False, "tool": call.name, "error": f"{exc.__class__.__name__}: {str(exc)[:200]}"}
        self._tool_calls_executed_this_turn += 1
        self.metrics.mark("tool_call_finished", tool=call.name, ok=bool(result.get("ok")))
        return result

    def _append_tool_exchange(self, calls: list[ToolCall], results: list[dict[str, object]]) -> None:
        tool_call_markers: list[dict[str, object]] = []
        for call in calls:
            marker: dict[str, object] = {"name": call.name, "arguments": dict(call.arguments)}
            # Carry the provider continuity tokens (H6) so the adapter can echo
            # them on the follow-up turn; omitted when the provider didn't emit
            # them, keeping tool-free / non-Gemini exchanges byte-identical.
            if call.id:
                marker["id"] = call.id
            if call.thought_signature:
                marker["thought_signature"] = call.thought_signature
            tool_call_markers.append(marker)
        self.conversation.append({"role": "assistant", "content": "", "tool_calls": tool_call_markers})
        for call, result in zip(calls, results):
            message: dict[str, object] = {
                "role": "tool",
                "name": call.name,
                "content": json.dumps(result, sort_keys=True, ensure_ascii=False, default=str),
            }
            if call.id:
                message["tool_call_id"] = call.id
            self.conversation.append(message)

    def _run_llm_fallback_chain(self) -> tuple[str, list[str]]:
        """Keep the turn (and the call) alive after a streaming-LLM failure.

        Every leg here — primary retry, fallback model, canned recovery — yields
        a TEXT-ONLY reply. Tool calls execute only for the successful primary
        stream (R1), so the retry passes ``collect_tool_calls=False`` and any
        function call it might emit is discarded, never executed.
        """

        config = self.llm_fallback
        if config.retry_primary and not self.output_cancel.cancelled:
            self.metrics.mark(
                "llm_fallback_leg_started",
                leg="primary_retry",
                budget_seconds=config.retry_budget_seconds,
            )
            retry_error, retry_parts, _retry_tool_calls = self._speak_streamed_reply(
                leg="primary_retry",
                first_item_budget_seconds=config.retry_budget_seconds,
            )
            if retry_error is None and "".join(retry_parts).strip():
                self.metrics.mark("llm_fallback_leg_completed", leg="primary_retry")
                return "primary_retry", retry_parts
            if retry_error is None:
                # S5: the retry finished cleanly but produced NO speakable text —
                # e.g. it emitted only a tool call, which a retry leg discards and
                # never executes. Treating that as a successful reply would return
                # an empty turn and leave the caller in silence. Fall through to
                # the fallback model / recovery utterance so the caller still
                # hears a spoken response.
                self.metrics.mark("llm_fallback_leg_yielded_no_text", leg="primary_retry")
            else:
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
                fallback_text = " ".join(str(config.fallback_llm.generate_reply(self._context_window())).split())
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
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if self._is_explicit_semantic_endpoint(event):
            return True
        if endpoint_source == "stt_finality":
            return True
        if (
            metadata.get("speech_final")
            or metadata.get("utterance_end")
            or metadata.get("endpoint")
        ):
            return True
        if metadata.get("from_finalize"):
            text = " ".join(event.text.strip().split())
            words = text.split()
            # Finalize is operator-forced, not natural end-of-speech. It may
            # legitimately return an atomic answer that cannot satisfy the
            # normal three-word completeness bar, but must not bless a comma-
            # terminated continuation fragment such as "Ee jah,".
            if 0 < len(words) <= 2 and not text.endswith(","):
                return True
        if not self._provider_final_text_is_complete(event):
            return False
        if endpoint_source == "semantic_final_text":
            return True
        if not self.semantic_endpointing.config.enabled:
            return True
        return self.semantic_endpointing.should_endpoint(event)

    def _has_transcript_text(self, events: Iterable[TranscriptEvent]) -> bool:
        return any(bool(event.text.strip()) for event in events)

    def _interim_endpoint_transcript(self, events: Iterable[TranscriptEvent], turn: DetectedTurn) -> TranscriptEvent | None:
        """Promote the freshest interim transcript to a synthetic final.

        Opt-in via accept_interim_on_rms_endpoint: skips the provider Finalize
        round-trip (the dominant post-endpoint wait on live calls) when the
        local RMS endpointer has already fired. Guarded so a mid-sentence pause
        cannot ship a stale fragment as the caller's turn:

        * only fires while the latest text-bearing event is still interim
          (a provider final always wins),
        * the interim must pass the same conservative completeness bar as
          provider finals (_provider_final_text_is_complete), independently
          of whether punctuation-only semantic endpointing is enabled,
        * when the provider reports coverage timings (Deepgram start+duration),
          the interim must cover audio up to near the detected end of speech.

        The promoted event keeps the interim's confidence and stamps
        provisional_from_interim=True so call artifacts can be audited.
        """

        latest: TranscriptEvent | None = None
        for event in events:
            if event.text.strip():
                latest = event
        if latest is None or latest.is_final:
            return None
        text = " ".join(latest.text.strip().split())
        if not text:
            return None
        metadata = dict(latest.metadata) if isinstance(latest.metadata, dict) else {}
        candidate = TranscriptEvent(
            text,
            is_final=True,
            language=latest.language,
            confidence=latest.confidence,
            metadata=dict(metadata),
        )
        if not self._provider_final_text_is_complete(candidate):
            self.metrics.mark("stt_interim_promotion_rejected", reason="incomplete_text", text_len=len(text))
            return None
        if not self._interim_covers_turn_audio(candidate, turn):
            self.metrics.mark("stt_interim_promotion_rejected", reason="stale_interim", text_len=len(text))
            return None
        metadata["semantic_endpoint"] = True
        metadata["provisional_from_interim"] = True
        return TranscriptEvent(
            text,
            is_final=True,
            language=latest.language,
            confidence=latest.confidence,
            metadata=metadata,
        )

    def _interim_covers_turn_audio(self, event: TranscriptEvent, turn: DetectedTurn) -> bool:
        """Recency gate: the interim must reach near the detected end of speech.

        Providers report per-result audio coverage (Deepgram: start + duration,
        in seconds from the start of the STT stream). If the freshest interim
        stopped covering audio long before the caller stopped speaking (e.g. a
        stalled provider connection), promoting it would truncate the turn.
        Both clocks count from stream start; the RMS detector's clock resets on
        each detected turn, so its end can only lag the provider clock — the
        check fails open in that direction, and also when a provider does not
        report timings at all.
        """

        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        start = metadata.get("start")
        duration = metadata.get("duration")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            return True
        covered_ms = (float(start) + float(duration)) * 1000.0
        speech_end_ms = turn.speech_end_ms if turn.speech_end_ms is not None else turn.end_ms
        return covered_ms >= speech_end_ms - INTERIM_ENDPOINT_MAX_LAG_MS

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

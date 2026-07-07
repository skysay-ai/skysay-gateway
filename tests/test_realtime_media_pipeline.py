from __future__ import annotations

import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from typing import Iterable

from openphonex_gateway.realtime import (
    AudioFrame,
    BargeInConfig,
    CallMetrics,
    LLMFallbackConfig,
    PacedRtpSender,
    RealtimeMediaPipeline,
    RmsTurnDetector,
    SemanticEndpointConfig,
    SemanticEndpointDetector,
    StreamingCallRecorder,
    TextChunk,
    TranscriptEvent,
    TurnDetectionConfig,
    default_recovery_utterance,
)
from openphonex_gateway.realtime.pipeline import StreamingLLM
from openphonex_gateway.realtime.stt import StreamingSTT, StreamingSTTSession
from openphonex_gateway.realtime.transport import CancellationToken
from openphonex_gateway.realtime.tts import StreamingTTS, speakable_chunks
from openphonex_gateway.realtime.turn_detection import DetectedTurn
from openphonex_gateway.rtp import decode_rtp_ulaw_payload


class FakeStreamingSTT(StreamingSTT):
    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        yield TranscriptEvent("Tere", is_final=False, language="et")
        yield TranscriptEvent("Tere, soovin aega broneerida.", is_final=True, language="et")


class FakeStreamingLLM(StreamingLLM):
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        yield TextChunk("Tere! ", is_final=False)
        yield TextChunk("Mis auto mudeliga tulete?", is_final=True)


class PunctuationlessStreamingLLM(StreamingLLM):
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        yield TextChunk("I can help book the car wash appointment and check the calendar for ", is_final=False)
        yield TextChunk("tomorrow morning", is_final=True)


class ManyPhraseStreamingLLM(StreamingLLM):
    def __init__(self, phrase_count: int = 6) -> None:
        self.phrase_count = phrase_count

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        for idx in range(self.phrase_count):
            yield TextChunk(f"Fraas {idx + 1}. ", is_final=idx == self.phrase_count - 1)


class ClosingManyPhraseStreamingLLM(StreamingLLM):
    def __init__(self, phrase_count: int = 20) -> None:
        self.phrase_count = phrase_count
        self.closed = False
        self.chunks_requested = 0

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        try:
            for idx in range(self.phrase_count):
                self.chunks_requested += 1
                yield TextChunk(f"Fraas {idx + 1}. ", is_final=idx == self.phrase_count - 1)
        finally:
            self.closed = True


class ErrorAfterFirstPhraseLLM(StreamingLLM):
    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        yield TextChunk("Tere! ", is_final=False)
        raise RuntimeError("stream failed")


class FailingThenWorkingLLM(StreamingLLM):
    """Fails the first stream (e.g. Gemini 503) and succeeds on the retry."""

    def __init__(self, reply: str = "Tere! Mis auto mudeliga tulete?") -> None:
        self.reply = reply
        self.calls = 0

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Gemini streamGenerateContent HTTP 503: unavailable")
        yield TextChunk(self.reply, is_final=True)


class AlwaysFailingLLM(StreamingLLM):
    def __init__(self) -> None:
        self.calls = 0

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        self.calls += 1
        raise RuntimeError("Gemini streamGenerateContent HTTP 503: unavailable")
        yield TextChunk("", is_final=True)  # pragma: no cover


class StallingRetryLLM(StreamingLLM):
    """Fails the first stream, then stalls long past the retry budget."""

    def __init__(self, stall_seconds: float = 0.6) -> None:
        self.stall_seconds = stall_seconds
        self.calls = 0

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Gemini streamGenerateContent HTTP 503: unavailable")
        time.sleep(self.stall_seconds)
        yield TextChunk("Liiga hilja.", is_final=True)


class RecordingFallbackLLM:
    def __init__(self, reply: str = "Vabandust, broneerin teile aja.") -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate_reply(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(message) for message in messages])
        return self.reply


class FailingFallbackLLM:
    def generate_reply(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("fallback model unavailable")


class ContextRecordingLLM(StreamingLLM):
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def stream_reply(self, messages: list[dict[str, str]]) -> Iterable[TextChunk]:
        self.calls.append([dict(message) for message in messages])
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        yield TextChunk(self.replies[index], is_final=True)


class FakeStreamingTTS(StreamingTTS):
    def stream(self, text: str) -> Iterable[AudioFrame]:
        yield AudioFrame.tone(40, direction="agent")
        yield AudioFrame.tone(40, hz=660, direction="agent")


class LongStreamingTTS(StreamingTTS):
    def __init__(self, frame_count: int = 80) -> None:
        self.frame_count = frame_count

    def stream(self, text: str) -> Iterable[AudioFrame]:
        for idx in range(self.frame_count):
            yield AudioFrame.tone(20, hz=440 + idx, direction="agent")


class ClosingStreamingTTS(StreamingTTS):
    def __init__(self, frame_count: int = 80) -> None:
        self.frame_count = frame_count
        self.closed = False
        self.frames_requested = 0

    def stream(self, text: str) -> Iterable[AudioFrame]:
        try:
            for idx in range(self.frame_count):
                self.frames_requested += 1
                yield AudioFrame.tone(20, hz=440 + idx, direction="agent")
        finally:
            self.closed = True


class FrameFedStreamingSTT(StreamingSTT):
    def __init__(self) -> None:
        self.session: FrameFedStreamingSTTSession | None = None
        self.transcribe_called = False

    def start_stream(self) -> StreamingSTTSession:
        self.session = FrameFedStreamingSTTSession()
        return self.session

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        self.transcribe_called = True
        return ()


class FrameFedStreamingSTTSession:
    def __init__(self) -> None:
        self.frames_seen = 0
        self.interim_emitted = False
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        self.frames_seen += 1
        if frame.rms > 650 and not self.interim_emitted:
            self.interim_emitted = True
            return (TranscriptEvent("Tere", is_final=False, language="et"),)
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if turn is None:
            return ()
        return (TranscriptEvent("Tere, soovin aega broneerida.", is_final=True, language="et"),)

    def close(self) -> None:
        self.closed = True


class EmptyFirstEndpointStreamingSTT(StreamingSTT):
    def __init__(self) -> None:
        self.session: EmptyFirstEndpointStreamingSTTSession | None = None

    def start_stream(self) -> StreamingSTTSession:
        self.session = EmptyFirstEndpointStreamingSTTSession()
        return self.session

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        return ()


class EmptyFirstEndpointStreamingSTTSession:
    def __init__(self) -> None:
        self.end_input_calls = 0
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if turn is None:
            return ()
        self.end_input_calls += 1
        if self.end_input_calls == 1:
            return ()
        return (
            TranscriptEvent(
                "Nii, selles mõttes, kas ma räägin, kellega ma räägin?",
                is_final=True,
                language="et",
                metadata={"provider": "test", "speech_final": True},
            ),
        )

    def close(self) -> None:
        self.closed = True


class IncompleteFirstEndpointStreamingSTT(StreamingSTT):
    def __init__(self) -> None:
        self.session: IncompleteFirstEndpointStreamingSTTSession | None = None

    def start_stream(self) -> StreamingSTTSession:
        self.session = IncompleteFirstEndpointStreamingSTTSession()
        return self.session

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        return ()


class IncompleteFirstEndpointStreamingSTTSession:
    def __init__(self) -> None:
        self.end_input_calls = 0
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        if turn is None:
            return ()
        self.end_input_calls += 1
        if self.end_input_calls == 1:
            return (TranscriptEvent("Nii,", is_final=True, language="et", metadata={"provider": "test"}),)
        return (
            TranscriptEvent(
                "Nii, selles mõttes, kas ma räägin, kellega ma räägin?",
                is_final=True,
                language="et",
                metadata={"provider": "test"},
            ),
        )

    def close(self) -> None:
        self.closed = True


class EarlyFinalStreamingSTT(StreamingSTT):
    def start_stream(self) -> StreamingSTTSession:
        return EarlyFinalStreamingSTTSession()

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        return ()


class EarlyFinalStreamingSTTSession:
    def __init__(self) -> None:
        self.speech_frames = 0
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if frame.rms > 650:
            self.speech_frames += 1
        if self.speech_frames == 13:
            self.speech_frames += 1
            return (
                TranscriptEvent(
                    "Tere, soovin aega broneerida.",
                    is_final=True,
                    language="et",
                    metadata={"provider": "test", "speech_final": True},
                ),
            )
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        return ()

    def close(self) -> None:
        self.closed = True


class EarlySemanticFinalStreamingSTT(StreamingSTT):
    def __init__(self, text: str = "Tere, soovin aega broneerida.") -> None:
        self.text = text

    def start_stream(self) -> StreamingSTTSession:
        return EarlySemanticFinalStreamingSTTSession(self.text)

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        return ()


class EarlySemanticFinalStreamingSTTSession:
    def __init__(self, text: str) -> None:
        self.text = text
        self.speech_frames = 0
        self.closed = False

    def send_audio(self, frame: AudioFrame) -> Iterable[TranscriptEvent]:
        if frame.rms > 650:
            self.speech_frames += 1
        if self.speech_frames == 13:
            self.speech_frames += 1
            return (
                TranscriptEvent(
                    self.text,
                    is_final=True,
                    language="et",
                    metadata={"provider": "test"},
                ),
            )
        return ()

    def end_input(self, turn: DetectedTurn | None = None) -> Iterable[TranscriptEvent]:
        return ()

    def close(self) -> None:
        self.closed = True


class RealtimeMediaPipelineTests(unittest.TestCase):
    def test_rms_turn_detector_endpoints_without_wav_files(self) -> None:
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        turn = None
        for frame in [AudioFrame.silence(20) for _ in range(5)]:
            turn = detector.feed(frame)
        for frame in [AudioFrame.tone(20) for _ in range(16)]:
            turn = detector.feed(frame)
        self.assertIsNone(turn)
        for frame in [AudioFrame.silence(20) for _ in range(15)]:
            turn = detector.feed(frame)
            if turn:
                break
        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertGreaterEqual(turn.speech_ms, 240)
        self.assertGreater(turn.end_ms, turn.start_ms)
        self.assertGreater(len(turn.pcm16), 0)

    def test_paced_rtp_sender_streams_chunks_and_can_cancel(self) -> None:
        packets: list[bytes] = []
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        cancellation = CancellationToken()

        def chunks() -> Iterable[bytes]:
            yield AudioFrame.tone(20).pcm16
            cancellation.cancel()
            yield AudioFrame.tone(20).pcm16

        metrics = CallMetrics(clock=_CounterClock())
        sent = sender.send_pcm16_stream(chunks(), cancellation=cancellation, metrics=metrics)
        self.assertEqual(sent, 1)
        self.assertEqual(len(packets), 1)
        self.assertEqual(len(decode_rtp_ulaw_payload(packets[0])), 320)
        self.assertIsNotNone(metrics.first("first_rtp_packet_sent"))
        caller_audio = metrics.first("caller_hears_first_audio_estimate")
        self.assertIsNotNone(caller_audio)
        assert caller_audio is not None
        self.assertEqual(caller_audio.attrs["packet_ms"], 20)

    def test_streaming_recorder_writes_live_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with StreamingCallRecorder(Path(td)) as recorder:
                recorder.write_caller(AudioFrame.tone(20, direction="caller"))
                recorder.write_agent(AudioFrame.tone(20, direction="agent"))
                recorder.write_caller_turn(1, [AudioFrame.tone(20, direction="caller")])
                recorder.write_agent_turn(1, [AudioFrame.tone(20, direction="agent")])
            for name in ("caller.wav", "agent.wav", "mixed.wav", "turn-001-caller.wav", "turn-001-agent.wav"):
                path = Path(td) / name
                self.assertTrue(path.exists(), name)
                with wave.open(str(path), "rb") as wf:
                    self.assertEqual(wf.getframerate(), 8000)
                    self.assertGreater(wf.getnframes(), 0)
            waveform = json.loads((Path(td) / "waveform.json").read_text())
            waveforms = json.loads((Path(td) / "waveforms.json").read_text())
            self.assertEqual(waveform["track"], "mixed")
            self.assertGreater(len(waveform["peaks"]), 0)
            self.assertIn("caller", waveforms["tracks"])
            self.assertIn("agent", waveforms["tracks"])
            self.assertIn("mixed", waveforms["tracks"])

    def test_pipeline_streams_tts_to_rtp_and_records_latency_json(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        with tempfile.TemporaryDirectory() as td:
            with StreamingCallRecorder(Path(td)) as recorder:
                pipeline = RealtimeMediaPipeline(
                    stt=FakeStreamingSTT(),
                    llm=FakeStreamingLLM(),
                    tts=FakeStreamingTTS(),
                    rtp_sender=sender,
                    recorder=recorder,
                    detector=detector,
                    metrics=metrics,
                )
                frames = [AudioFrame.silence(20) for _ in range(3)]
                frames += [AudioFrame.tone(20) for _ in range(16)]
                frames += [AudioFrame.silence(20) for _ in range(20)]
                turn = pipeline.consume_inbound_until_turn(frames)
                self.assertIsNotNone(turn)
                assert turn is not None
                result = pipeline.process_turn(turn)
                metrics.write_json(Path(td) / "latency.json")

            self.assertIn("soovin aega", result.transcript)
            self.assertIn("auto mudeliga", result.assistant_text)
            self.assertGreaterEqual(len(packets), 4)
            self.assertIsNotNone(result.latency["e2e_user_stop_to_first_rtp_ms"])
            self.assertIsNotNone(result.latency["caller_hears_first_audio_estimate_ms"])
            self.assertGreaterEqual(
                result.latency["caller_hears_first_audio_estimate_ms"] or 0,
                result.latency["e2e_user_stop_to_first_rtp_ms"] or 0,
            )
            data = json.loads((Path(td) / "latency.json").read_text())
            self.assertIn("first_tts_audio_byte", [event["name"] for event in data["events"]])
            self.assertIn("first_rtp_packet_sent", [event["name"] for event in data["events"]])
            self.assertIn("caller_hears_first_audio_estimate", [event["name"] for event in data["events"]])
            self.assertIn("caller_speech_started", [event["name"] for event in data["events"]])
            self.assertTrue((Path(td) / "turn-001-caller.wav").exists())
            self.assertTrue((Path(td) / "turn-001-agent.wav").exists())
            self.assertIn("caller_turn_clip_written", [event.name for event in metrics.events])
            self.assertIn("agent_turn_clip_written", [event.name for event in metrics.events])
            self.assertIsNotNone(result.latency["speech_start_to_endpoint_ms"])
            self.assertIsNotNone(result.latency["speech_start_to_first_transcript_delta_ms"])
            self.assertIsNotNone(result.latency["speech_start_to_first_rtp_ms"])
            self.assertIsNotNone(result.latency["transcript_delta_to_final_transcript_ms"])
            self.assertIsNotNone(result.latency["endpoint_to_final_transcript_ms"])
            self.assertIsNotNone(result.latency["final_transcript_to_llm_request_ms"])
            self.assertIsNotNone(result.latency["endpoint_to_llm_request_ms"])
            self.assertIsNotNone(result.latency["llm_first_speakable_phrase_ms"])
            self.assertIsNotNone(result.latency["llm_first_chunk_to_first_speakable_phrase_ms"])
            self.assertIsNotNone(result.latency["endpoint_to_first_speakable_phrase_ms"])
            self.assertIsNotNone(result.latency["first_phrase_to_tts_audio_ms"])
            event_names = [event.name for event in metrics.events]
            self.assertIn("first_transcript_delta", event_names)
            self.assertIn("final_transcript", event_names)
            self.assertIn("first_speakable_phrase", event_names)
            self.assertIn("tts_phrase_first_audio_byte", event_names)
            self.assertIn("tts_phrase_rtp_done", event_names)
            self.assertLess(event_names.index("first_speakable_phrase"), event_names.index("first_tts_audio_byte"))
            self.assertEqual(len([event for event in metrics.events if event.name == "speakable_phrase_ready"]), 2)
            self.assertEqual(len([event for event in metrics.events if event.name == "tts_phrase_rtp_done"]), 2)

    def test_pipeline_consumes_llm_stream_while_first_phrase_is_speaking(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.002))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=LongStreamingTTS(frame_count=24),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertIn("auto mudeliga", result.assistant_text)
        events = metrics.events
        event_names = [event.name for event in events]
        self.assertIn("llm_stream_completed", event_names)
        first_phrase_done_index = next(
            index
            for index, event in enumerate(events)
            if event.name == "tts_phrase_rtp_done" and event.attrs.get("phrase_index") == 1
        )
        self.assertLess(event_names.index("llm_stream_completed"), first_phrase_done_index)
        self.assertEqual(len([event for event in events if event.name == "speakable_phrase_ready"]), 2)

    def test_pipeline_starts_tts_before_final_chunk_for_long_punctuationless_reply(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=PunctuationlessStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertEqual(
            result.assistant_text,
            "I can help book the car wash appointment and check the calendar for tomorrow morning",
        )
        events = metrics.events
        phrase_events = [event for event in events if event.name == "speakable_phrase_ready"]
        tts_done_events = [event for event in events if event.name == "tts_phrase_rtp_done"]
        self.assertEqual(len(phrase_events), 2)
        self.assertEqual(len(tts_done_events), 2)
        self.assertLessEqual(phrase_events[0].attrs["text_len"], 64)

    def test_pipeline_bounds_pending_phrases_when_tts_is_slower_than_llm(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.002))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=ManyPhraseStreamingLLM(phrase_count=6),
            tts=LongStreamingTTS(frame_count=80),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            max_pending_phrases=1,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertIn("Fraas 6.", result.assistant_text)
        events = metrics.events
        event_names = [event.name for event in events]
        self.assertIn("llm_phrase_queue_full", event_names)
        self.assertIn("llm_stream_completed", event_names)
        self.assertNotIn("llm_phrase_producer_still_running", event_names)
        self.assertEqual(len([event for event in events if event.name == "tts_phrase_rtp_done"]), 6)
        queue_full = next(event for event in events if event.name == "llm_phrase_queue_full")
        self.assertEqual(queue_full.attrs["max_pending_phrases"], 1)

    def test_pipeline_closes_llm_stream_on_barge_in_cancel_under_backpressure(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        llm = ClosingManyPhraseStreamingLLM(phrase_count=20)
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.002))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        barge_frames = [AudioFrame.tone(20, direction="caller")]

        def read_barge_frame() -> AudioFrame | None:
            if len(packets) >= 2 and barge_frames:
                return barge_frames.pop(0)
            return None

        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=LongStreamingTTS(frame_count=80),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            barge_in_frame_reader=read_barge_frame,
            barge_in_config=BargeInConfig(poll_interval_seconds=0.001),
            max_pending_phrases=1,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertTrue(result.interrupted)
        self.assertTrue(llm.closed)
        self.assertLess(llm.chunks_requested, llm.phrase_count)
        event_names = [event.name for event in metrics.events]
        self.assertIn("llm_phrase_queue_full", event_names)
        self.assertIn("llm_phrase_producer_cancelled", event_names)
        self.assertIn("llm_chunk_stream_closed", event_names)
        self.assertNotIn("llm_phrase_producer_still_running", event_names)

    def test_pipeline_survives_llm_stream_error_and_completes_turn(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=ErrorAfterFirstPhraseLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertIn(result.llm_fallback_leg, {"primary_retry", "recovery_utterance"})
        self.assertTrue(result.assistant_text)
        event_names = [event.name for event in metrics.events]
        self.assertIn("llm_stream_error", event_names)
        self.assertIn("llm_primary_stream_failed", event_names)
        self.assertIn("llm_fallback_used", event_names)
        self.assertIn("tts_phrase_rtp_done", event_names)

    def test_pipeline_retries_primary_llm_once_after_stream_error(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        llm = FailingThenWorkingLLM()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertEqual(llm.calls, 2)
        self.assertEqual(result.llm_fallback_leg, "primary_retry")
        self.assertIn("auto mudeliga", result.assistant_text)
        self.assertIn("auto mudeliga", result.assistant_spoken_text)
        self.assertGreater(len(packets), 0)
        fallback_event = metrics.first("llm_fallback_used")
        self.assertIsNotNone(fallback_event)
        assert fallback_event is not None
        self.assertEqual(fallback_event.attrs["leg"], "primary_retry")

    def test_pipeline_falls_back_to_non_streaming_model_reply_as_one_phrase(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        llm = AlwaysFailingLLM()
        fallback_llm = RecordingFallbackLLM("Vabandust, broneerin teile aja homme kell kümme.")
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            llm_fallback=LLMFallbackConfig(
                retry_budget_seconds=0.5,
                fallback_llm=fallback_llm,
                fallback_model_id="gemini-2.5-flash-lite",
            ),
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(fallback_llm.calls), 1)
        self.assertEqual(result.llm_fallback_leg, "fallback_model")
        self.assertEqual(result.assistant_text, "Vabandust, broneerin teile aja homme kell kümme.")
        self.assertEqual(result.assistant_spoken_text, "Vabandust, broneerin teile aja homme kell kümme.")
        # The fallback text is fed to TTS as one phrase.
        tts_done = [event for event in metrics.events if event.name == "tts_phrase_rtp_done"]
        self.assertEqual(len(tts_done), 1)
        self.assertEqual([message["role"] for message in pipeline.conversation], ["user", "assistant"])
        fallback_event = metrics.first("llm_fallback_used")
        assert fallback_event is not None
        self.assertEqual(fallback_event.attrs["leg"], "fallback_model")

    def test_pipeline_speaks_language_aware_recovery_utterance_when_all_llm_legs_fail(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        llm = AlwaysFailingLLM()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            llm_fallback=LLMFallbackConfig(
                retry_budget_seconds=0.5,
                fallback_llm=FailingFallbackLLM(),
                fallback_model_id="gemini-2.5-flash-lite",
                language="et",
            ),
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertEqual(result.llm_fallback_leg, "recovery_utterance")
        self.assertEqual(result.assistant_text, default_recovery_utterance("et"))
        self.assertIn("Vabandust", result.assistant_text)
        self.assertGreater(len(packets), 0)

    def test_pipeline_recovery_utterance_env_override_and_retry_budget(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        llm = StallingRetryLLM(stall_seconds=0.6)
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            llm_fallback=LLMFallbackConfig(
                retry_budget_seconds=0.05,
                recovery_utterance="Palun vabandust, proovime uuesti.",
            ),
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertEqual(result.llm_fallback_leg, "recovery_utterance")
        self.assertEqual(result.assistant_text, "Palun vabandust, proovime uuesti.")
        event_names = [event.name for event in metrics.events]
        self.assertIn("llm_stream_first_phrase_budget_exceeded", event_names)

    def test_pipeline_reports_per_turn_latency_and_voice_to_voice(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )

        results = []
        for _ in range(2):
            frames = [AudioFrame.silence(20) for _ in range(3)]
            frames += [AudioFrame.tone(20) for _ in range(16)]
            frames += [AudioFrame.silence(20) for _ in range(20)]
            turn = pipeline.consume_inbound_until_turn(frames)
            self.assertIsNotNone(turn)
            assert turn is not None
            results.append(pipeline.process_turn(turn))

        # Each turn reports its own spans instead of repeating turn 1 numbers.
        self.assertIsNotNone(metrics.first("first_llm_chunk", turn=1))
        self.assertIsNotNone(metrics.first("first_llm_chunk", turn=2))
        self.assertIsNotNone(metrics.first("first_rtp_packet_sent", turn=2))
        self.assertIsNotNone(metrics.first("first_tts_audio_byte", turn=1))
        self.assertIsNotNone(metrics.first("first_tts_audio_byte", turn=2))
        for turn_index, result in enumerate(results, start=1):
            self.assertIsNotNone(result.latency["e2e_user_stop_to_first_rtp_ms"])
            self.assertIsNotNone(result.latency["voice_to_voice_ms"])
            # TTS spans resolve per turn, not just for the call's first utterance.
            self.assertIsNotNone(result.latency["tts_first_audio_ms"])
            self.assertIsNotNone(result.latency["first_phrase_to_tts_audio_ms"])
            self.assertIsNotNone(result.latency["transport_ms"])
            self.assertEqual(result.latency, metrics.latency_breakdown(turn=turn_index))
        # voice_to_voice starts at the true caller speech end (before the
        # silence hold), so it is at least the endpoint-stamped e2e span.
        self.assertGreaterEqual(
            results[0].latency["voice_to_voice_ms"] or 0.0,
            results[0].latency["e2e_user_stop_to_first_rtp_ms"] or 0.0,
        )
        self.assertEqual(metrics.turn_indices(), [1, 2])
        payload = metrics.as_dict()
        self.assertEqual([entry["turn"] for entry in payload["turns"]], [1, 2])
        self.assertEqual(payload["turns"][1]["latency"], metrics.latency_breakdown(turn=2))
        self.assertTrue(all("turn" in event for event in payload["events"]))
        speech_ended = metrics.first("caller_speech_ended", turn=1)
        self.assertIsNotNone(speech_ended)
        assert speech_ended is not None
        self.assertGreaterEqual(speech_ended.attrs["silence_tail_ms"], 300)

    def test_pipeline_reports_turn_one_tts_spans_after_outbound_opening(self) -> None:
        """Regression: the turn-0 opening utterance must not consume the single
        first_tts_audio_byte mark, leaving turn 1 without its own TTS spans."""
        packets: list[bytes] = []
        metrics = CallMetrics()
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )

        opening = pipeline.speak_text("Tere, siin OpenPhonex.", label="outbound_opening")
        self.assertTrue(opening["spoken"])
        self.assertIsNotNone(metrics.first("first_tts_audio_byte", turn=0))

        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None
        result = pipeline.process_turn(turn)

        self.assertIsNotNone(metrics.first("first_tts_audio_byte", turn=1))
        self.assertIsNotNone(result.latency["tts_first_audio_ms"])
        self.assertIsNotNone(result.latency["first_phrase_to_tts_audio_ms"])
        self.assertIsNotNone(result.latency["transport_ms"])
        self.assertEqual(result.latency, metrics.latency_breakdown(turn=1))

    def test_pipeline_keeps_completed_assistant_audio_in_next_turn_context(self) -> None:
        packets: list[bytes] = []
        llm = ContextRecordingLLM(["Esimene vastus.", "Teine vastus."])
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
        )

        first_frames = [AudioFrame.silence(20) for _ in range(3)]
        first_frames += [AudioFrame.tone(20) for _ in range(16)]
        first_frames += [AudioFrame.silence(20) for _ in range(20)]
        first_turn = pipeline.consume_inbound_until_turn(first_frames)
        self.assertIsNotNone(first_turn)
        assert first_turn is not None
        first_result = pipeline.process_turn(first_turn)
        self.assertEqual(first_result.assistant_text, "Esimene vastus.")
        self.assertEqual(first_result.assistant_spoken_text, "Esimene vastus.")

        second_frames = [AudioFrame.tone(20) for _ in range(16)]
        second_frames += [AudioFrame.silence(20) for _ in range(20)]
        second_turn = pipeline.consume_inbound_until_turn(second_frames)
        self.assertIsNotNone(second_turn)
        assert second_turn is not None
        pipeline.process_turn(second_turn)

        self.assertEqual([message["role"] for message in llm.calls[1]], ["user", "assistant", "user"])
        self.assertEqual(llm.calls[1][1]["content"], "Esimene vastus.")

    def test_pipeline_excludes_interrupted_assistant_audio_from_next_turn_context(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        llm = ContextRecordingLLM(["Palun tooge auto homme kell üks.", "Sain katkestuse."])
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.002))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        barge_frames = [AudioFrame.tone(20, direction="caller")]
        tts = ClosingStreamingTTS(frame_count=80)

        def read_barge_frame() -> AudioFrame | None:
            if len(packets) >= 2 and barge_frames:
                return barge_frames.pop(0)
            return None

        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=llm,
            tts=LongStreamingTTS(frame_count=80),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            barge_in_frame_reader=read_barge_frame,
            barge_in_config=BargeInConfig(poll_interval_seconds=0.001),
        )
        first_frames = [AudioFrame.silence(20) for _ in range(3)]
        first_frames += [AudioFrame.tone(20) for _ in range(16)]
        first_frames += [AudioFrame.silence(20) for _ in range(20)]
        first_turn = pipeline.consume_inbound_until_turn(first_frames)
        self.assertIsNotNone(first_turn)
        assert first_turn is not None
        first_result = pipeline.process_turn(first_turn)
        self.assertTrue(first_result.interrupted)
        self.assertEqual(first_result.assistant_text, "Palun tooge auto homme kell üks.")
        self.assertEqual(first_result.assistant_spoken_text, "")

        continuation_frames = [AudioFrame.tone(20) for _ in range(15)]
        continuation_frames += [AudioFrame.silence(20) for _ in range(20)]
        second_turn = pipeline.consume_inbound_until_turn(continuation_frames)
        self.assertIsNotNone(second_turn)
        assert second_turn is not None
        pipeline.process_turn(second_turn)

        self.assertEqual([message["role"] for message in llm.calls[1]], ["user", "user"])
        self.assertTrue(all(message["content"] != first_result.assistant_text for message in llm.calls[1]))
        self.assertIsNotNone(metrics.first("assistant_context_truncated"))

    def test_pipeline_feeds_stt_before_endpoint_detection(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        stt = FrameFedStreamingSTT()
        pipeline = RealtimeMediaPipeline(
            stt=stt,
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]

        turn = pipeline.consume_inbound_until_turn(frames)
        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertIsNotNone(stt.session)
        assert stt.session is not None
        self.assertGreater(stt.session.frames_seen, 0)
        self.assertTrue(stt.session.closed)
        event_names = [event.name for event in metrics.events]
        self.assertIn("caller_speech_started", event_names)
        self.assertLess(event_names.index("caller_speech_started"), event_names.index("first_transcript_delta"))
        self.assertLess(event_names.index("first_transcript_delta"), event_names.index("user_endpoint_detected"))

        result = pipeline.process_turn(turn)
        self.assertFalse(stt.transcribe_called)
        self.assertIn("soovin aega", result.transcript)
        self.assertIsNotNone(result.latency["e2e_user_stop_to_first_rtp_ms"])

    def test_pipeline_keeps_listening_after_endpoint_without_transcript(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        stt = EmptyFirstEndpointStreamingSTT()
        pipeline = RealtimeMediaPipeline(
            stt=stt,
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        frames += [AudioFrame.tone(20, hz=660) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertIsNotNone(stt.session)
        assert stt.session is not None
        self.assertEqual(stt.session.end_input_calls, 2)
        event_names = [event.name for event in metrics.events]
        self.assertEqual(event_names.count("candidate_endpoint_detected"), 2)
        self.assertEqual(event_names.count("user_endpoint_detected"), 1)
        self.assertEqual(event_names.count("stt_endpoint_without_transcript"), 1)
        self.assertGreater(turn.speech_ms, 500)
        result = pipeline.process_turn(turn)
        self.assertIn("kellega ma räägin", result.transcript)
        self.assertGreater(len(packets), 0)

    def test_pipeline_keeps_listening_after_incomplete_final_transcript(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        stt = IncompleteFirstEndpointStreamingSTT()
        pipeline = RealtimeMediaPipeline(
            stt=stt,
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]
        frames += [AudioFrame.tone(20, hz=660) for _ in range(16)]
        frames += [AudioFrame.silence(20) for _ in range(20)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertIsNotNone(stt.session)
        assert stt.session is not None
        self.assertEqual(stt.session.end_input_calls, 2)
        event_names = [event.name for event in metrics.events]
        self.assertEqual(event_names.count("candidate_endpoint_detected"), 2)
        self.assertEqual(event_names.count("user_endpoint_detected"), 1)
        self.assertEqual(event_names.count("stt_endpoint_incomplete_transcript"), 1)
        result = pipeline.process_turn(turn)
        self.assertEqual(result.transcript, "Nii, selles mõttes, kas ma räägin, kellega ma räägin?")
        self.assertGreater(len(packets), 0)

    def test_pipeline_can_endpoint_on_stt_finality_before_silence_hold(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=1000, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=EarlyFinalStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertIn("soovin aega", " ".join(event.text for event in turn.transcript_events))
        endpoint = metrics.first("user_endpoint_detected")
        self.assertIsNotNone(endpoint)
        assert endpoint is not None
        self.assertEqual(endpoint.attrs["source"], "stt_finality")
        self.assertLess(turn.end_ms, 1000)
        result = pipeline.process_turn(turn)
        self.assertIn("soovin aega", result.transcript)

    def test_pipeline_can_endpoint_on_semantic_final_text_without_speech_final(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=1000, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=EarlySemanticFinalStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNotNone(turn)
        assert turn is not None
        endpoint = metrics.first("user_endpoint_detected")
        self.assertIsNotNone(endpoint)
        assert endpoint is not None
        self.assertEqual(endpoint.attrs["source"], "semantic_final_text")
        self.assertIsNotNone(metrics.first("semantic_endpoint_detected"))
        self.assertLess(turn.end_ms, 1000)
        result = pipeline.process_turn(turn)
        self.assertIn("soovin aega", result.transcript)

    def test_pipeline_does_not_semantic_endpoint_on_incomplete_final_text(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=1000, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=EarlySemanticFinalStreamingSTT("Tere, soovin aega"),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNone(turn)
        self.assertIsNone(metrics.first("semantic_endpoint_detected"))

    def test_pipeline_can_disable_semantic_endpointing(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None)
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=1000, min_speech_ms=240))
        pipeline = RealtimeMediaPipeline(
            stt=EarlySemanticFinalStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=FakeStreamingTTS(),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            semantic_endpointing=SemanticEndpointDetector(SemanticEndpointConfig(enabled=False)),
        )
        frames = [AudioFrame.silence(20) for _ in range(3)]
        frames += [AudioFrame.tone(20) for _ in range(16)]

        turn = pipeline.consume_inbound_until_turn(frames)

        self.assertIsNone(turn)
        self.assertIsNone(metrics.first("semantic_endpoint_detected"))

    def test_pipeline_cancels_output_on_barge_in_and_replays_caller_frames(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.002))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        barge_frames = [AudioFrame.tone(20, direction="caller")]
        tts = ClosingStreamingTTS(frame_count=80)

        def read_barge_frame() -> AudioFrame | None:
            if len(packets) >= 2 and barge_frames:
                return barge_frames.pop(0)
            return None

        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=tts,
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            barge_in_frame_reader=read_barge_frame,
            barge_in_config=BargeInConfig(poll_interval_seconds=0.001),
        )
        initial_frames = [AudioFrame.silence(20) for _ in range(3)]
        initial_frames += [AudioFrame.tone(20) for _ in range(16)]
        initial_frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(initial_frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertTrue(result.interrupted)
        self.assertGreaterEqual(len(packets), 1)
        self.assertLess(len(packets), 80)
        self.assertTrue(tts.closed)
        self.assertLess(tts.frames_requested, 80)
        event_names = [event.name for event in metrics.events]
        self.assertIn("barge_in_speech_detected", event_names)
        self.assertIn("barge_in_output_cancelled", event_names)
        self.assertIn("assistant_turn_interrupted", event_names)
        self.assertIsNotNone(result.latency["barge_in_cancel_ms"])
        self.assertEqual(result.latency["barge_in_cancel_ms"], metrics.latency_breakdown(turn=1)["barge_in_cancel_ms"])

        continuation_frames = [AudioFrame.tone(20) for _ in range(15)]
        continuation_frames += [AudioFrame.silence(20) for _ in range(20)]
        next_turn = pipeline.consume_inbound_until_turn(continuation_frames)
        self.assertIsNotNone(next_turn)
        event_names = [event.name for event in metrics.events]
        self.assertIn("barge_in_pending_frames_replayed", event_names)

    def test_pipeline_does_not_replay_monitor_frames_without_barge_in(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics(clock=_CounterClock())
        sender = PacedRtpSender(packets.append, clock_sleep=lambda _seconds: time.sleep(0.001))
        detector = RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=300, min_speech_ms=240))
        monitor_frames = [AudioFrame.silence(20, direction="caller") for _ in range(3)]

        def read_monitor_frame() -> AudioFrame | None:
            if len(packets) >= 1 and monitor_frames:
                return monitor_frames.pop(0)
            return None

        pipeline = RealtimeMediaPipeline(
            stt=FrameFedStreamingSTT(),
            llm=FakeStreamingLLM(),
            tts=LongStreamingTTS(frame_count=6),
            rtp_sender=sender,
            detector=detector,
            metrics=metrics,
            barge_in_frame_reader=read_monitor_frame,
            barge_in_config=BargeInConfig(speech_rms=650, poll_interval_seconds=0.001),
        )
        initial_frames = [AudioFrame.silence(20) for _ in range(3)]
        initial_frames += [AudioFrame.tone(20) for _ in range(16)]
        initial_frames += [AudioFrame.silence(20) for _ in range(20)]
        turn = pipeline.consume_inbound_until_turn(initial_frames)
        self.assertIsNotNone(turn)
        assert turn is not None

        result = pipeline.process_turn(turn)

        self.assertFalse(result.interrupted)
        event_names = [event.name for event in metrics.events]
        self.assertIn("barge_in_monitor_stopped", event_names)
        self.assertNotIn("barge_in_frames_buffered", event_names)
        self.assertNotIn("barge_in_pending_frames_replayed", event_names)

    def test_speakable_chunks_emits_short_first_phrase(self) -> None:
        chunks = [
            TextChunk("Tere! ", is_final=False),
            TextChunk("Ma saan aidata ", is_final=False),
            TextChunk("broneeringuga.", is_final=True),
        ]
        self.assertEqual(list(speakable_chunks(chunks)), ["Tere!", "Ma saan aidata broneeringuga."])

    def test_speakable_chunks_emits_word_boundary_soft_first_phrase_without_punctuation(self) -> None:
        chunks = [
            TextChunk("one two three four five six seven ", is_final=False),
            TextChunk("eight nine", is_final=True),
        ]
        self.assertEqual(
            list(speakable_chunks(chunks, first_phrase_soft_limit=20, phrase_soft_limit=40)),
            ["one two three four", "five six seven eight nine"],
        )


class _CounterClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


if __name__ == "__main__":
    unittest.main()

"""M5 local-model provider adapter tests (SCH-203).

These exercise the optional local adapters (faster-whisper STT,
OpenAI-compatible LLM, Piper/Kokoro TTS) and the numpy resample helper. The
numpy-dependent paths are skipped when numpy is not installed so the core suite
and CI stay green without the ``local-*`` extras; the availability() checks and
the stdlib-only OpenAI-compatible SSE path always run.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openphonex_gateway.realtime.frames import AudioFrame
from openphonex_gateway.realtime.turn_detection import DetectedTurn
from openphonex_gateway.realtime.worker import _build_default_registry

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _pcm(samples: list[int]) -> bytes:
    return struct.pack("<" + "h" * len(samples), *samples)


class _Config:
    """Minimal stand-in for RealtimeWorkerConfig fields the adapters read."""

    fake_language = "et"
    elevenlabs_language_code = "et"
    gemini_system_prompt = ""
    gemini_temperature = 0.3
    gemini_max_output_tokens = 32
    gemini_timeout_seconds = 10.0


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed (local extra)")
class ResampleTests(unittest.TestCase):
    def test_upsample_8k_to_16k_doubles_sample_count(self) -> None:
        from openphonex_gateway.realtime import resample

        pcm = _pcm([0, 1000, 2000, 3000, 2000, 1000, 0, -1000])
        out = resample.upsample_pcm16(pcm, 8000, 16000)
        self.assertEqual(len(out) // 2, (len(pcm) // 2) * 2)

    def test_downsample_24k_to_8k_thirds_sample_count(self) -> None:
        from openphonex_gateway.realtime import resample

        samples = [int(1000 * ((i % 12) - 6)) for i in range(36)]
        pcm = _pcm(samples)
        out = resample.downsample_pcm16(pcm, 24000, 8000)
        self.assertEqual(len(out) // 2, len(samples) // 3)

    def test_downsample_22050_to_8k_length(self) -> None:
        from openphonex_gateway.realtime import resample

        import numpy as np

        n = 2205
        samples = (np.sin(np.linspace(0, 20 * np.pi, n)) * 8000).astype("<i2").tobytes()
        out = resample.downsample_pcm16(samples, 22050, 8000)
        self.assertEqual(len(out) // 2, round(n * 8000 / 22050))

    def test_same_rate_is_identity(self) -> None:
        from openphonex_gateway.realtime import resample

        pcm = _pcm([1, 2, 3, 4])
        self.assertEqual(resample.resample_pcm16(pcm, 8000, 8000), pcm)

    def test_empty_input(self) -> None:
        from openphonex_gateway.realtime import resample

        self.assertEqual(resample.resample_pcm16(b"", 8000, 16000), b"")

    def test_ramp_stays_monotonic(self) -> None:
        from openphonex_gateway.realtime import resample

        import numpy as np

        pcm = _pcm(list(range(0, 8000, 8)))  # a rising ramp
        out = resample.downsample_pcm16(pcm, 16000, 8000)
        values = np.frombuffer(out, dtype="<i2")
        self.assertTrue(np.all(np.diff(values.astype(np.int32)) >= 0))


class FasterWhisperAvailabilityTests(unittest.TestCase):
    def test_availability_reports_missing_dep_when_absent(self) -> None:
        from openphonex_gateway.realtime import faster_whisper_stt

        available = importlib.util.find_spec("faster_whisper") is not None
        missing = faster_whisper_stt.PROVIDER_SPEC.availability(_Config())
        if available:
            self.assertNotIn("faster-whisper", missing)
        else:
            self.assertIn("faster-whisper", missing)

    def test_describe_fragment_shape(self) -> None:
        from openphonex_gateway.realtime import faster_whisper_stt

        fragment = faster_whisper_stt.PROVIDER_SPEC.describe(_Config())
        self.assertEqual(fragment["provider"], "faster_whisper")
        self.assertEqual(fragment["adapter"], "FasterWhisperSTT")
        self.assertTrue(fragment["real"])
        self.assertIn("model_id", fragment)

    @unittest.skipUnless(_HAS_NUMPY, "numpy not installed (local extra)")
    def test_transcribe_resamples_and_calls_model(self) -> None:
        from openphonex_gateway.realtime import faster_whisper_stt

        captured: dict[str, object] = {}

        class _Segment:
            text = " hello world "

        class _Info:
            language = "en"

        class _Model:
            def transcribe(self, audio, language, beam_size):
                captured["samples"] = len(audio)
                captured["language"] = language
                return [_Segment()], _Info()

        adapter = faster_whisper_stt.FasterWhisperSTT(_Config())
        adapter._model = _Model()  # skip the heavy WhisperModel load
        frame = AudioFrame(_pcm([0, 5000, -5000, 0] * 200), sample_rate=8000)
        turn = DetectedTurn(frames=[frame], start_ms=0, end_ms=100, speech_ms=100)
        events = list(adapter.transcribe(turn))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].text, "hello world")
        self.assertTrue(events[0].is_final)
        # 8k -> 16k doubled the sample count fed to the model.
        self.assertEqual(captured["samples"], 800 * 2)


class OpenAICompatibleLLMTests(unittest.TestCase):
    def _start_sse_server(self, sse_body: bytes) -> ThreadingHTTPServer:
        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse_body)

            def log_message(self, *args) -> None:  # silence
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_stream_reply_parses_sse_deltas(self) -> None:
        from openphonex_gateway.realtime import openai_compatible_llm

        sse = (
            b'data: {"choices":[{"delta":{"content":"Tere"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":", broneerin"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" aja."},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        server = self._start_sse_server(sse)
        host, port = server.server_address

        env = {
            "REALTIME_LLM_BASE_URL": f"http://{host}:{port}/v1",
            "REALTIME_LLM_MODEL": "stub-model",
            "REALTIME_LLM_API_KEY": "stub",
        }
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            llm = openai_compatible_llm.OpenAICompatibleLLM(_Config())
            chunks = list(llm.stream_reply([{"role": "user", "content": "Tere"}]))
        texts = [chunk.text for chunk in chunks if chunk.text]
        self.assertEqual("".join(texts), "Tere, broneerin aja.")
        self.assertTrue(chunks[-1].is_final)

    def test_availability_is_empty_without_import_crash(self) -> None:
        from openphonex_gateway.realtime import openai_compatible_llm

        self.assertEqual(openai_compatible_llm.PROVIDER_SPEC.availability(_Config()), [])

    def test_describe_reports_model(self) -> None:
        from openphonex_gateway.realtime import openai_compatible_llm


        with unittest.mock.patch.dict(os.environ, {"REALTIME_LLM_MODEL": "llama3.1"}, clear=False):
            fragment = openai_compatible_llm.PROVIDER_SPEC.describe(_Config())
        self.assertEqual(fragment["provider"], "openai_compatible")
        self.assertEqual(fragment["model_id"], "llama3.1")


class LocalTTSTests(unittest.TestCase):
    def test_piper_availability_missing_binary(self) -> None:
        from openphonex_gateway.realtime import local_tts


        with unittest.mock.patch.dict(os.environ, {"REALTIME_TTS_PIPER_BIN": "definitely-not-a-real-binary"}, clear=False):
            missing = local_tts.PIPER_SPEC.availability(_Config())
        # piper python package + binary both absent in the test env.
        if importlib.util.find_spec("piper") is None:
            self.assertIn("piper", missing)

    def test_kokoro_availability_missing(self) -> None:
        from openphonex_gateway.realtime import local_tts

        missing = local_tts.KOKORO_SPEC.availability(_Config())
        if importlib.util.find_spec("kokoro") is None:
            self.assertIn("kokoro", missing)

    def test_describe_fragments(self) -> None:
        from openphonex_gateway.realtime import local_tts

        piper = local_tts.PIPER_SPEC.describe(_Config())
        self.assertEqual(piper["provider"], "piper")
        self.assertEqual(piper["language_code"], "et")
        kokoro = local_tts.KOKORO_SPEC.describe(_Config())
        self.assertEqual(kokoro["provider"], "kokoro")

    @unittest.skipUnless(_HAS_NUMPY, "numpy not installed (local extra)")
    def test_piper_stream_downsamples_to_8k_frames(self) -> None:
        from openphonex_gateway.realtime import local_tts

        import numpy as np

        native = (np.sin(np.linspace(0, 40 * np.pi, 22050)) * 8000).astype("<i2").tobytes()
        adapter = local_tts.PiperStreamingTTS(_Config())
        with unittest.mock.patch.object(adapter, "_synthesize", return_value=(native, 22050)):
            frames = list(adapter.stream("tere"))
        self.assertTrue(frames)
        for frame in frames:
            self.assertIsInstance(frame, AudioFrame)
            self.assertEqual(frame.sample_rate, 8000)
            self.assertEqual(frame.direction, "agent")
        total_samples = sum(frame.samples for frame in frames)
        self.assertAlmostEqual(total_samples, round(22050 * 8000 / 22050), delta=1)


class LocalProviderRegistryTests(unittest.TestCase):
    def test_registry_resolves_local_names_lazily(self) -> None:
        registry = _build_default_registry()
        self.assertIsNotNone(registry.get("stt", "faster_whisper"))
        self.assertIsNotNone(registry.get("llm", "openai_compatible"))
        self.assertIsNotNone(registry.get("tts", "piper"))
        self.assertIsNotNone(registry.get("tts", "kokoro"))

    def test_openai_compatible_aliases(self) -> None:
        registry = _build_default_registry()
        for alias in ("ollama", "vllm", "local_llm"):
            spec = registry.get("llm", alias)
            self.assertIsNotNone(spec, alias)
            self.assertEqual(spec.name, "openai_compatible")


if __name__ == "__main__":
    unittest.main()

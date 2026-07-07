"""Local TTS adapters: Piper and Kokoro (optional ``local-tts`` extra).

Both synthesise to the 8 kHz pipeline edge, downsampling from the model's
native rate and chunking into 20 ms ``AudioFrame`` (direction=``agent``). Heavy
imports (piper / kokoro / onnxruntime / numpy) are deferred to first use so
importing this module (to read the specs) stays cheap.

Piper: shells to the ``piper`` binary (like the ElevenLabs ffmpeg pattern) and
reads raw PCM; Kokoro: lazy-imports the ``kokoro`` pipeline. Both are fully
local, so a customer running faster_whisper + openai_compatible(Ollama) +
piper needs zero built-in provider API keys.

Config (env):
  REALTIME_TTS_MODEL / REALTIME_TTS_VOICE   model/voice id or path
  REALTIME_TTS_PIPER_BIN                     piper binary (default ``piper``)
  REALTIME_TTS_SAMPLE_RATE                   piper native rate (default 22050)
  REALTIME_TTS_LANG / REALTIME_TTS_SPEAKER   kokoro language / speaker
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from typing import Iterable

from . import resample
from .frames import AudioFrame
from .registry import ProviderSpec
from .tts import StreamingTTS

_FRAME_MS = 20
_EDGE_RATE = 8000


def _chunk_frames(pcm8k: bytes, *, frame_ms: int = _FRAME_MS) -> Iterable[AudioFrame]:
    bytes_per_frame = int(_EDGE_RATE * frame_ms / 1000) * 2
    if bytes_per_frame <= 0:
        return
    for start in range(0, len(pcm8k), bytes_per_frame):
        chunk = pcm8k[start : start + bytes_per_frame]
        if not chunk:
            continue
        yield AudioFrame(chunk, sample_rate=_EDGE_RATE, direction="agent", metadata={"provider": "local_tts"})


class PiperStreamingTTS(StreamingTTS):
    def __init__(self, config: object):
        self.config = config
        self._binary = os.environ.get("REALTIME_TTS_PIPER_BIN", "piper")
        self._model = os.environ.get("REALTIME_TTS_MODEL") or os.environ.get("REALTIME_TTS_VOICE", "")
        self._native_rate = int(os.environ.get("REALTIME_TTS_SAMPLE_RATE", "22050"))

    def _synthesize(self, text: str) -> tuple[bytes, int]:
        command = [self._binary, "--model", self._model, "--output-raw"]
        completed = subprocess.run(
            command,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return completed.stdout, self._native_rate

    def stream(self, text: str) -> Iterable[AudioFrame]:
        pcm, native_rate = self._synthesize(text)
        pcm8k = resample.downsample_pcm16(pcm, native_rate, _EDGE_RATE)
        yield from _chunk_frames(pcm8k)


class KokoroStreamingTTS(StreamingTTS):
    _NATIVE_RATE = 24000

    def __init__(self, config: object):
        self.config = config
        self._voice = os.environ.get("REALTIME_TTS_VOICE") or os.environ.get("REALTIME_TTS_SPEAKER", "af_heart")
        self._lang = os.environ.get("REALTIME_TTS_LANG", "a")

    def _synthesize(self, text: str) -> bytes:
        from kokoro import KPipeline  # lazy heavy import
        import numpy as np

        pipeline = KPipeline(lang_code=self._lang)
        chunks = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=self._voice):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            return b""
        audio = np.concatenate(chunks)
        return np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2").tobytes()

    def stream(self, text: str) -> Iterable[AudioFrame]:
        pcm = self._synthesize(text)
        pcm8k = resample.downsample_pcm16(pcm, self._NATIVE_RATE, _EDGE_RATE)
        yield from _chunk_frames(pcm8k)


def _piper_availability(config: object) -> list[str]:
    missing: list[str] = []
    binary = os.environ.get("REALTIME_TTS_PIPER_BIN", "piper")
    if importlib.util.find_spec("piper") is None and shutil.which(binary) is None:
        missing.append("piper")
    model = os.environ.get("REALTIME_TTS_MODEL") or os.environ.get("REALTIME_TTS_VOICE", "")
    if model and (os.sep in model) and not os.path.exists(os.path.expanduser(model)):
        missing.append(model)
    return missing


def _kokoro_availability(config: object) -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("kokoro") is None:
        missing.append("kokoro")
    if importlib.util.find_spec("onnxruntime") is None and importlib.util.find_spec("torch") is None:
        missing.append("onnxruntime")
    return missing


def _piper_describe(config: object) -> dict:
    return {
        "provider": "piper",
        "adapter": "PiperStreamingTTS",
        "real": True,
        "model_id": os.environ.get("REALTIME_TTS_MODEL") or os.environ.get("REALTIME_TTS_VOICE", ""),
        "language_code": getattr(config, "elevenlabs_language_code", ""),
    }


def _kokoro_describe(config: object) -> dict:
    return {
        "provider": "kokoro",
        "adapter": "KokoroStreamingTTS",
        "real": True,
        "model_id": os.environ.get("REALTIME_TTS_VOICE") or os.environ.get("REALTIME_TTS_SPEAKER", ""),
        "language_code": getattr(config, "elevenlabs_language_code", ""),
    }


PIPER_SPEC = ProviderSpec(
    name="piper",
    component="tts",
    factory=PiperStreamingTTS,
    describe=_piper_describe,
    availability=_piper_availability,
    real=True,
)

KOKORO_SPEC = ProviderSpec(
    name="kokoro",
    component="tts",
    factory=KokoroStreamingTTS,
    describe=_kokoro_describe,
    availability=_kokoro_availability,
    real=True,
)

PROVIDER_SPECS = [PIPER_SPEC, KOKORO_SPEC]

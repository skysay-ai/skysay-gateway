"""faster-whisper local STT adapter (optional ``local-stt`` extra).

Fully offline, turn-buffered STT. It implements only ``transcribe(turn)`` so
``start_stt_stream`` wraps it in ``TurnBufferedSTTSession`` automatically — no
streaming session needed. The heavy ``faster_whisper`` / ``numpy`` imports are
deferred to first use so importing this module (to read ``PROVIDER_SPEC``)
stays cheap and never pulls the model runtime.

Config (env):
  REALTIME_LOCAL_STT_MODEL         model name or local path (default ``base``)
  REALTIME_LOCAL_STT_DEVICE        device (default ``cpu``)
  REALTIME_LOCAL_STT_COMPUTE_TYPE  compute type (default ``int8``)
  REALTIME_LANGUAGE                language (reused; feeds config.fake_language)
"""

from __future__ import annotations

import importlib.util
import os
from typing import Iterable

from . import resample
from .frames import TranscriptEvent
from .registry import ProviderSpec
from .stt import StreamingSTT
from .turn_detection import DetectedTurn


def _model_name() -> str:
    return (os.environ.get("REALTIME_LOCAL_STT_MODEL", "base") or "base").strip()


def _looks_like_path(value: str) -> bool:
    return bool(value) and (os.sep in value or value.startswith(".") or value.startswith("~"))


class FasterWhisperSTT(StreamingSTT):
    def __init__(self, config: object):
        self.config = config
        self._model = None
        self._model_name = _model_name()
        self._device = os.environ.get("REALTIME_LOCAL_STT_DEVICE", "cpu")
        self._compute_type = os.environ.get("REALTIME_LOCAL_STT_COMPUTE_TYPE", "int8")
        self._language = (getattr(config, "fake_language", "") or "et").strip() or "et"

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy heavy import

            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(self, turn: DetectedTurn) -> Iterable[TranscriptEvent]:
        pcm = b"".join(frame.pcm16 for frame in turn.frames)
        if not pcm:
            return
        import numpy as np  # lazy: bundled with faster-whisper

        pcm_16k = resample.upsample_pcm16(pcm, 8000, 16000)
        audio = np.frombuffer(pcm_16k, dtype="<i2").astype(np.float32) / 32768.0
        model = self._ensure_model()
        segments, info = model.transcribe(audio, language=self._language, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        language = getattr(info, "language", None) or self._language
        yield TranscriptEvent(
            text,
            is_final=True,
            language=language,
            metadata={"provider": "faster_whisper", "audio_ms": max(0, turn.end_ms - turn.start_ms)},
        )


def _availability(config: object) -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("faster_whisper") is None:
        missing.append("faster-whisper")
    model = _model_name()
    if _looks_like_path(model) and not os.path.exists(os.path.expanduser(model)):
        missing.append(model)
    return missing


def _describe(config: object) -> dict:
    return {
        "provider": "faster_whisper",
        "adapter": "FasterWhisperSTT",
        "real": True,
        "language": getattr(config, "fake_language", ""),
        "model_id": _model_name(),
    }


PROVIDER_SPEC = ProviderSpec(
    name="faster_whisper",
    component="stt",
    factory=FasterWhisperSTT,
    describe=_describe,
    availability=_availability,
    real=True,
)

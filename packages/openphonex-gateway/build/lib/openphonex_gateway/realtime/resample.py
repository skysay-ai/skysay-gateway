"""PCM16 resampling helpers for the optional local-model adapters only.

The realtime pipeline moves ``pcm_s16le`` at 8000 Hz mono end to end. Local
models want other rates (faster-whisper wants 16 kHz input; Piper emits
16/22.05 kHz; Kokoro emits 24 kHz), so the optional local adapters convert
to/from the 8 kHz pipeline edge with these helpers.

Implemented with numpy linear interpolation. ``audioop`` is intentionally NOT
used — it was removed in Python 3.13+ and the repo runs on Python 3.14
(system) / 3.12 (.venv). numpy is a transitive dependency of the local extras
(faster-whisper / the tts stacks), so it is present exactly when a local
adapter is installed. The import is guarded so the core pipeline never needs
numpy.
"""

from __future__ import annotations


def _numpy():
    try:
        import numpy  # lazy: present only with a local extra installed
    except Exception as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "numpy is required for local-model PCM resampling; install an "
            "openphonex-gateway local extra (e.g. 'local-stt')"
        ) from exc
    return numpy


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample little-endian signed 16-bit mono PCM from ``src`` to ``dst`` Hz."""

    if not pcm:
        return b""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    if src_rate == dst_rate:
        return pcm
    np = _numpy()
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return b""
    dst_n = max(1, int(round(samples.size * dst_rate / src_rate)))
    if samples.size == 1:
        resampled = np.repeat(samples.astype(np.float64), dst_n)
    else:
        src_index = np.linspace(0.0, samples.size - 1, num=dst_n)
        resampled = np.interp(src_index, np.arange(samples.size), samples.astype(np.float64))
    clipped = np.clip(np.round(resampled), -32768, 32767).astype("<i2")
    return clipped.tobytes()


def upsample_pcm16(pcm: bytes, src_rate: int = 8000, dst_rate: int = 16000) -> bytes:
    """Upsample PCM (default the 8 kHz pipeline edge -> 16 kHz STT input)."""

    return resample_pcm16(pcm, src_rate, dst_rate)


def downsample_pcm16(pcm: bytes, src_rate: int, dst_rate: int = 8000) -> bytes:
    """Downsample PCM (default a TTS model rate -> the 8 kHz pipeline edge)."""

    return resample_pcm16(pcm, src_rate, dst_rate)

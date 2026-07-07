# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Stdlib audio-encode helpers shared by the GA-5 provider-edge adapters.

These are the two conversions the GA-5 batch/streaming adapters need at the
provider boundary, both stdlib-only (no ffmpeg, no numpy):

* ``pcm16_to_wav`` wraps buffered turn PCM in a RIFF/WAVE container using the
  stdlib :mod:`wave` module — the OpenAI / ElevenLabs-Scribe / Google-REST
  batch STT paths POST that WAV as a multipart file.
* ``chunk_pcm16_to_frames`` slices a raw PCM16 byte stream (either a single
  ``bytes`` blob or an iterable of blobs read incrementally off an HTTP
  response) into fixed ~20 ms ``AudioFrame`` objects, flushing any sub-frame
  tail at the end — the same 320-bytes/frame @ 8 kHz logic already in
  ``local_tts._chunk_frames`` and
  ``elevenlabs_v3_streaming.decode_audio_stream_to_pcm16_frames``, factored out
  so the Deepgram-Aura and Cartesia TTS adapters share one implementation.

The pipeline edge is 8 kHz mono PCM16 (``pcm_s16le``); both TTS providers can
emit that natively, so the default ``rate=8000`` keeps the whole outbound path
numpy/ffmpeg-free.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Iterable, Iterator

from .frames import AudioFrame

_EDGE_RATE = 8000
_FRAME_MS = 20


def pcm16_to_wav(pcm: bytes, rate: int = _EDGE_RATE, *, channels: int = 1) -> bytes:
    """Wrap little-endian signed 16-bit mono PCM in a WAV container (stdlib only)."""

    if rate <= 0:
        raise ValueError("wav sample rate must be positive")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _iter_blobs(pcm: "bytes | Iterable[bytes]") -> Iterable[bytes]:
    if isinstance(pcm, (bytes, bytearray)):
        yield bytes(pcm)
        return
    for blob in pcm:
        if blob:
            yield bytes(blob)


def chunk_pcm16_to_frames(
    pcm: "bytes | Iterable[bytes]",
    *,
    rate: int = _EDGE_RATE,
    frame_ms: int = _FRAME_MS,
    direction: str = "agent",
    metadata: "dict[str, Any] | None" = None,
) -> Iterator[AudioFrame]:
    """Yield fixed ~``frame_ms`` ``AudioFrame`` objects from a PCM16 stream.

    ``pcm`` may be a single ``bytes`` blob or an iterable of blobs (e.g. read
    incrementally off a streaming HTTP response). Full frames are emitted as
    soon as enough bytes accumulate; any remainder shorter than one frame is
    flushed as a final short frame at stream end.
    """

    bytes_per_frame = int(rate * frame_ms / 1000) * 2
    if bytes_per_frame <= 0:
        return
    meta = dict(metadata or {})
    buffer = bytearray()
    for blob in _iter_blobs(pcm):
        buffer.extend(blob)
        while len(buffer) >= bytes_per_frame:
            frame = bytes(buffer[:bytes_per_frame])
            del buffer[:bytes_per_frame]
            yield AudioFrame(frame, sample_rate=rate, direction=direction, metadata=dict(meta))
    if buffer:
        yield AudioFrame(bytes(buffer), sample_rate=rate, direction=direction, metadata=dict(meta))

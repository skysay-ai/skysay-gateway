# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.parse
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from typing import Iterable

from .frames import AudioFrame


@dataclass(frozen=True)
class ElevenLabsStreamingConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_v3"
    output_format: str = "mp3_44100_128"
    language_code: str = ""
    stability: float = 0.45
    similarity_boost: float = 0.8
    chunk_size: int = 4096


def resolve_elevenlabs_voice_id(
    *,
    api_key: str,
    voice_id: str = "",
    voice_name: str = "",
    timeout_seconds: float = 20.0,
) -> str:
    """Resolve a configured ElevenLabs voice id or human voice name.

    Production workers should cache the result at call startup so turn handling
    never performs a voice-list lookup on the latency path.
    """

    if voice_id.strip():
        return voice_id.strip()
    wanted = voice_name.strip()
    if not wanted:
        raise ValueError("ElevenLabs voice_id or voice name is required")
    if not api_key:
        raise ValueError("ElevenLabs API key is required")

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode())
    for voice in payload.get("voices", []):
        if str(voice.get("name", "")).casefold() == wanted.casefold():
            resolved = str(voice.get("voice_id", "")).strip()
            if resolved:
                return resolved
    raise ValueError(f"ElevenLabs voice not found: {wanted}")


class ElevenLabsStreamingTTS:
    """HTTP-streaming ElevenLabs adapter for the configured model.

    Both ElevenLabs v3 and Flash use the common HTTP streaming endpoint. This
    stateless adapter deliberately leaves persistent WebSocket lifecycle
    management to a separate implementation.
    """

    def __init__(self, config: ElevenLabsStreamingConfig):
        if not config.api_key:
            raise ValueError("ElevenLabs API key is required")
        if not config.voice_id:
            raise ValueError("ElevenLabs voice_id is required")
        language = config.language_code.strip().replace("_", "-").partition("-")[0].casefold()
        if config.model_id.strip() == "eleven_flash_v2_5" and language == "et":
            raise ValueError("ElevenLabs eleven_flash_v2_5 does not support Estonian")
        self.config = config

    def stream(self, text: str) -> Iterable[AudioFrame]:
        byte_chunks = self._stream_audio_bytes(text)
        yield from decode_audio_stream_to_pcm16_frames(byte_chunks)

    def _stream_audio_bytes(self, text: str) -> Iterable[bytes]:
        query = urllib.parse.urlencode({"output_format": self.config.output_format})
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{urllib.parse.quote(self.config.voice_id, safe='')}/stream?{query}"
        body: dict[str, object] = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
            },
        }
        if self.config.language_code:
            body["language_code"] = self.config.language_code
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={
                "xi-api-key": self.config.api_key,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            while True:
                chunk = response.read(self.config.chunk_size)
                if not chunk:
                    break
                yield chunk


# Backwards-compatible names for callers importing the original v3-specific
# adapter API. The default model remains ElevenLabs v3.
ElevenLabsV3Config = ElevenLabsStreamingConfig
ElevenLabsV3StreamingTTS = ElevenLabsStreamingTTS


def missing_elevenlabs_streaming_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    return missing


def decode_audio_stream_to_pcm16_frames(
    byte_chunks: Iterable[bytes],
    *,
    sample_rate: int = 8000,
    frame_ms: int = 20,
) -> Iterable[AudioFrame]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required to decode ElevenLabs streaming response")
    process = subprocess.Popen(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    stop = threading.Event()
    writer_error: list[BaseException] = []

    def writer() -> None:
        assert process.stdin is not None
        try:
            for chunk in byte_chunks:
                if stop.is_set():
                    break
                if chunk:
                    try:
                        process.stdin.write(chunk)
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break
        except BaseException as exc:
            writer_error.append(exc)
        finally:
            with suppress(Exception):
                process.stdin.close()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    assert process.stdout is not None
    frame_bytes = int(sample_rate * frame_ms / 1000) * 2
    finished_reading = False
    try:
        while True:
            payload = process.stdout.read(frame_bytes)
            if not payload:
                break
            if len(payload) == frame_bytes:
                yield AudioFrame(payload, sample_rate=sample_rate, direction="agent")
        finished_reading = True
    finally:
        stop.set()
        with suppress(Exception):
            if process.stdin:
                process.stdin.close()
        thread.join(timeout=2)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        else:
            process.wait(timeout=2)
        if finished_reading and writer_error:
            raise RuntimeError("audio stream writer failed while decoding ElevenLabs response") from writer_error[0]
        if finished_reading and process.returncode not in (0, None):
            raise RuntimeError("ffmpeg failed to decode ElevenLabs streaming response")

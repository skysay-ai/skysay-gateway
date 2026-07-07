from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:  # monolith: control-plane namespace present (telephony_ai.* compat shims)
    from telephony_ai import __version__
    from telephony_ai.runtime import deployment_environment, env_enabled, env_flag, service_version, structured_log
except ModuleNotFoundError:  # standalone openphonex-gateway (OSS, no control plane)
    from openphonex_gateway import __version__
    from openphonex_common.runtime import deployment_environment, env_enabled, env_flag, service_version, structured_log

# Report whichever import name loaded us (telephony_ai.media shim, or the
# standalone openphonex_gateway package) so `python -m ...` hints copy-paste.
_PKG = __package__ or "openphonex_gateway"

from .rtp import MULAW_DECODE, linear_to_mulaw

BYTES_PER_SECOND_WAV = 16000
MULTI_LANGUAGE_VALUES = {"", "auto", "multi", "multilingual"}


def load_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path or not path.exists():
        return values
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def merged_env(env_path: Path | None = None) -> dict[str, str]:
    values = load_env_file(env_path)
    values.update({key: value for key, value in os.environ.items() if isinstance(value, str)})
    return values


def _bool_env(value: str | None, default: bool) -> bool:
    return env_flag(value, default=default)


def read_wav_samples(path: Path) -> tuple[tuple[int, ...], int]:
    if not path.exists():
        return (), 8000
    data = path.read_bytes()
    if len(data) < 46:
        return (), 8000
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        rate = 8000
        frames = b""
        offset = 12
        while offset + 8 <= len(data):
            chunk_id = data[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            chunk_start = offset + 8
            if chunk_id == b"fmt " and chunk_start + 8 <= len(data):
                rate = struct.unpack_from("<I", data, chunk_start + 4)[0]
            elif chunk_id == b"data":
                chunk_end = min(len(data), chunk_start + chunk_size) if chunk_size else len(data)
                frames = data[chunk_start:chunk_end]
                break
            offset = chunk_start + chunk_size + (chunk_size % 2)
        if not frames and len(data) > 44:
            frames = data[44:]
    else:
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
        except (EOFError, wave.Error):
            return (), 8000
    if len(frames) < 2:
        return (), rate
    if len(frames) % 2:
        frames = frames[:-1]
    return struct.unpack("<" + "h" * (len(frames) // 2), frames), rate


def write_pcm16_wav(path: Path, samples: tuple[int, ...] | list[int], rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples) if samples else b"")


def wav_duration_from_size(path: Path) -> float:
    if not path.exists():
        return 0.0
    return max(0, path.stat().st_size - 44) / BYTES_PER_SECOND_WAV


def wav_rms_recent(path: Path, window_sec: float = 0.35) -> float:
    samples, rate = read_wav_samples(path)
    if not samples:
        return 0.0
    win = max(1, int(rate * window_sec))
    recent = samples[-win:]
    return math.sqrt(sum(sample * sample for sample in recent) / len(recent))


def max_window_rms(path: Path, window_sec: float = 0.25) -> float:
    samples, rate = read_wav_samples(path)
    if not samples:
        return 0.0
    win = max(1, int(rate * window_sec))
    best = 0.0
    for idx in range(0, max(1, len(samples) - win + 1), max(1, win // 2)):
        part = samples[idx : idx + win]
        if part:
            best = max(best, math.sqrt(sum(sample * sample for sample in part) / len(part)))
    return best


def listen_for_turn(
    remote_wav: Path,
    max_seconds: float,
    poll_sec: float = 0.2,
    speech_rms: float = 650.0,
    silence_rms: float = 220.0,
    silence_hold_sec: float = 1.0,
    min_speech_sec: float = 0.6,
    preroll_sec: float = 0.35,
) -> tuple[float, float]:
    base_start = wav_duration_from_size(remote_wav)
    start = base_start
    speech_started = False
    speech_time = 0.0
    silence_time = 0.0
    elapsed = 0.0
    while elapsed < max_seconds:
        time.sleep(poll_sec)
        elapsed += poll_sec
        rms = wav_rms_recent(remote_wav)
        current_end = wav_duration_from_size(remote_wav)
        if rms >= speech_rms:
            if not speech_started:
                start = max(base_start, current_end - preroll_sec)
            speech_started = True
            speech_time += poll_sec
            silence_time = 0.0
        elif speech_started:
            if rms <= silence_rms:
                silence_time += poll_sec
                if speech_time >= min_speech_sec and silence_time >= silence_hold_sec:
                    return start, current_end
            else:
                silence_time = 0.0
    return start, wav_duration_from_size(remote_wav)


def detect_turn_window(
    source_wav: Path,
    *,
    speech_rms: float = 650.0,
    silence_rms: float = 220.0,
    silence_hold_sec: float = 1.0,
    min_speech_sec: float = 0.6,
    preroll_sec: float = 0.35,
    frame_sec: float = 0.05,
) -> tuple[float, float]:
    samples, rate = read_wav_samples(source_wav)
    if not samples:
        return 0.0, 0.0
    frame = max(1, int(rate * frame_sec))
    speech_started = False
    speech_time = 0.0
    silence_time = 0.0
    start_sec = 0.0
    last_sec = 0.0
    for idx in range(0, len(samples), frame):
        chunk = samples[idx : idx + frame]
        if not chunk:
            continue
        rms = math.sqrt(sum(sample * sample for sample in chunk) / len(chunk))
        chunk_start = idx / rate
        chunk_end = min(len(samples), idx + len(chunk)) / rate
        last_sec = chunk_end
        if rms >= speech_rms:
            if not speech_started:
                start_sec = max(0.0, chunk_start - preroll_sec)
            speech_started = True
            speech_time += chunk_end - chunk_start
            silence_time = 0.0
        elif speech_started:
            if rms <= silence_rms:
                silence_time += chunk_end - chunk_start
                if speech_time >= min_speech_sec and silence_time >= silence_hold_sec:
                    return start_sec, chunk_end
            else:
                silence_time = 0.0
    return (start_sec, last_sec) if speech_started else (0.0, 0.0)


def extract_segment(src_wav: Path, start_sec: float, end_sec: float, out_wav: Path) -> bool:
    if end_sec <= start_sec + 0.2:
        return False
    samples, rate = read_wav_samples(src_wav)
    if not samples:
        return False
    start = max(0, int(start_sec * rate))
    end = min(len(samples), int(end_sec * rate))
    segment = list(samples[start:end])
    if len(segment) < int(rate * 0.2):
        return False
    write_pcm16_wav(out_wav, segment, rate)
    return out_wav.exists() and out_wav.stat().st_size > 1024 and max_window_rms(out_wav) >= 650


def waveform_peaks(path: Path, buckets: int = 80) -> list[float]:
    samples, _rate = read_wav_samples(path)
    if not samples:
        return []
    bucket_size = max(1, len(samples) // buckets)
    peaks: list[float] = []
    for idx in range(0, len(samples), bucket_size):
        chunk = samples[idx : idx + bucket_size]
        if chunk:
            peaks.append(round(min(1.0, max(abs(sample) for sample in chunk) / 32768.0), 4))
    return peaks[:buckets]


class RtpTap:
    def __init__(self, listen_host: str, listen_port: int, out_wav: Path):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.out_wav = out_wav
        self.stop = threading.Event()
        self.packet_count = 0
        self.payload_count = 0
        self.last_packet_ts = 0.0
        self.peer: tuple[str, int] | None = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((listen_host, listen_port))
        self.sock.settimeout(0.5)
        self.wav = wave.open(str(out_wav), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(8000)

    def run(self) -> None:
        try:
            while not self.stop.is_set():
                try:
                    packet, addr = self.sock.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(packet) < 12:
                    continue
                self.peer = addr
                self.packet_count += 1
                self.last_packet_ts = time.time()
                cc = packet[0] & 0x0F
                header_len = 12 + cc * 4
                payload = packet[header_len:]
                if not payload:
                    continue
                pcm = bytearray()
                for byte in payload:
                    pcm.extend(struct.pack("<h", MULAW_DECODE[byte]))
                self.payload_count += len(payload)
                self.wav.writeframes(bytes(pcm))
        finally:
            self.wav.close()
            self.sock.close()

    def close(self) -> None:
        self.stop.set()

    def send_pcm16_as_rtp(self, pcm16: bytes, dst: tuple[str, int], ssrc: int = 0x12345678) -> None:
        seq = 1
        ts = 0
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        samples = struct.unpack("<" + "h" * (len(pcm16) // 2), pcm16)
        for idx in range(0, len(samples), 160):
            chunk = samples[idx : idx + 160]
            if len(chunk) < 160:
                break
            payload = bytes(linear_to_mulaw(sample) for sample in chunk)
            header = struct.pack("!BBHII", 0x80, 0x00, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
            self.sock.sendto(header + payload, dst)
            seq += 1
            ts += 160
            time.sleep(0.02)


@dataclass(frozen=True)
class AsteriskVoiceGatewayConfig:
    ari_url: str
    ari_user: str
    ari_password: str
    app: str
    app_args: str
    calls_dir: Path
    recordings_dir: Path
    rtp_bind: str
    rtp_port: int
    rtp_start: int
    rtp_end: int
    asterisk_host_target: str
    asterisk_rtp_host: str
    speech_output: str
    opening: str
    goal: str
    language: str
    reply_language: str
    max_turns: int
    listen_seconds: float
    listen_poll_sec: float
    speech_rms: float
    silence_rms: float
    silence_hold_sec: float
    min_speech_sec: float
    preroll_sec: float
    native_record: bool
    live_enabled: bool
    legacy_script: Path | None
    legacy_python: str

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "AsteriskVoiceGatewayConfig":
        env = merged_env(env_path)
        calls_dir = Path(env.get("OPENPHONEX_CALLS_DIR", "/calls"))
        recordings_dir = Path(env.get("OPENPHONEX_RECORDINGS_DIR", "/recordings"))
        return cls(
            ari_url=env.get("ASTERISK_ARI_URL", "http://didww-asterisk-duplex:8088/ari"),
            ari_user=env.get("ASTERISK_ARI_USER", ""),
            ari_password=env.get("ASTERISK_ARI_PASSWORD", ""),
            app=env.get("ASTERISK_ARI_APP", "voicebot"),
            app_args=env.get("ASTERISK_ARI_APP_ARGS", "inbound,outbound"),
            calls_dir=calls_dir,
            recordings_dir=recordings_dir,
            rtp_bind=env.get("OPENPHONEX_VOICE_RTP_BIND", "0.0.0.0"),
            rtp_port=int(env.get("OPENPHONEX_VOICE_RTP_PORT", env.get("OPENPHONEX_RTP_START", "10000"))),
            rtp_start=int(env.get("OPENPHONEX_RTP_START", "10000")),
            rtp_end=int(env.get("OPENPHONEX_RTP_END", "10100")),
            asterisk_host_target=env.get("ASTERISK_EXTERNAL_MEDIA_HOST", "telephony-ai-voice-worker"),
            asterisk_rtp_host=env.get("ASTERISK_RTP_HOST", "didww-asterisk-duplex"),
            speech_output=env.get("OPENPHONEX_VOICE_SPEECH_OUTPUT", "playback"),
            opening=env.get("OPENPHONEX_VOICE_OPENING", "Hi, this is Telephony AI. Can you hear me clearly?"),
            goal=env.get("OPENPHONEX_VOICE_GOAL", ""),
            language=env.get("OPENPHONEX_VOICE_LANGUAGE", "multi"),
            reply_language=env.get("OPENPHONEX_VOICE_REPLY_LANGUAGE", "English"),
            max_turns=int(env.get("OPENPHONEX_VOICE_MAX_TURNS", "4")),
            listen_seconds=float(env.get("OPENPHONEX_VOICE_LISTEN_SECONDS", "10.0")),
            listen_poll_sec=float(env.get("OPENPHONEX_VOICE_LISTEN_POLL_SEC", "0.12")),
            speech_rms=float(env.get("OPENPHONEX_VOICE_SPEECH_RMS", "650.0")),
            silence_rms=float(env.get("OPENPHONEX_VOICE_SILENCE_RMS", "220.0")),
            silence_hold_sec=float(env.get("OPENPHONEX_VOICE_SILENCE_HOLD_SEC", "0.65")),
            min_speech_sec=float(env.get("OPENPHONEX_VOICE_MIN_SPEECH_SEC", "0.45")),
            preroll_sec=float(env.get("OPENPHONEX_VOICE_PREROLL_SEC", "0.25")),
            native_record=_bool_env(env.get("OPENPHONEX_VOICE_NATIVE_RECORD"), True),
            live_enabled=_bool_env(env.get("OPENPHONEX_VOICE_LIVE_ENABLED"), False),
            legacy_script=Path(env["OPENPHONEX_LEGACY_VOICEBOT_SCRIPT"]) if env.get("OPENPHONEX_LEGACY_VOICEBOT_SCRIPT") else None,
            legacy_python=env.get("OPENPHONEX_LEGACY_PYTHON", "python"),
        )

    @property
    def missing_live_env(self) -> list[str]:
        missing: list[str] = []
        if not self.ari_user:
            missing.append("ASTERISK_ARI_USER")
        if not self.ari_password:
            missing.append("ASTERISK_ARI_PASSWORD")
        return missing

    def public_status(self) -> dict[str, Any]:
        return {
            "ari_url": self.ari_url,
            "app": self.app,
            "app_args": self.app_args,
            "calls_dir": str(self.calls_dir),
            "recordings_dir": str(self.recordings_dir),
            "rtp": {
                "bind": self.rtp_bind,
                "port": self.rtp_port,
                "range": [self.rtp_start, self.rtp_end],
                "asterisk_host_target": self.asterisk_host_target,
                "asterisk_rtp_host": self.asterisk_rtp_host,
            },
            "speech_output": self.speech_output,
            "language": self.language,
            "reply_language": self.reply_language,
            "max_turns": self.max_turns,
            "native_record": self.native_record,
            "live_enabled": self.live_enabled,
            "legacy_script": str(self.legacy_script) if self.legacy_script else "",
            "legacy_script_exists": bool(self.legacy_script and self.legacy_script.exists()),
            "missing_live_env": self.missing_live_env,
            "secrets_redacted": True,
        }

    def external_media_params(self) -> dict[str, str]:
        return {
            "app": self.app,
            "external_host": f"{self.asterisk_host_target}:{self.rtp_port}",
            "format": "ulaw",
            "encapsulation": "rtp",
            "transport": "udp",
            "connection_type": "client",
            "direction": "both",
        }

    def legacy_voicebot_args(self) -> list[str]:
        return [
            "--ari-url",
            self.ari_url,
            "--rtp-bind",
            self.rtp_bind,
            "--rtp-port",
            str(self.rtp_port),
            "--asterisk-host-target",
            self.asterisk_host_target,
            "--asterisk-rtp-host",
            self.asterisk_rtp_host,
            "--opening",
            self.opening,
            "--speech-output",
            self.speech_output,
            "--max-turns",
            str(self.max_turns),
            "--listen-seconds",
            str(self.listen_seconds),
            "--listen-poll-sec",
            str(self.listen_poll_sec),
            "--speech-rms",
            str(self.speech_rms),
            "--silence-rms",
            str(self.silence_rms),
            "--silence-hold-sec",
            str(self.silence_hold_sec),
            "--min-speech-sec",
            str(self.min_speech_sec),
            "--preroll-sec",
            str(self.preroll_sec),
            "--language",
            self.language,
            "--reply-language",
            self.reply_language,
            "--goal",
            self.goal,
        ]

    def legacy_voicebot_command(self) -> list[str]:
        if not self.legacy_script:
            return []
        return [self.legacy_python, str(self.legacy_script), *self.legacy_voicebot_args()]


class AsteriskVoiceGatewayWorker:
    def __init__(self, config: AsteriskVoiceGatewayConfig):
        self.config = config
        self.started_at = time.time()

    def _ensure_dir(self, path: Path) -> dict[str, Any]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"path": str(path), "exists": path.exists(), "writable": os.access(path, os.W_OK), "error": ""}
        except OSError as exc:
            return {"path": str(path), "exists": path.exists(), "writable": False, "error": str(exc)}

    def status(self) -> dict[str, Any]:
        calls_dir = self._ensure_dir(self.config.calls_dir)
        recordings_dir = self._ensure_dir(self.config.recordings_dir)
        storage_ok = bool(calls_dir["exists"] and calls_dir["writable"] and recordings_dir["exists"] and recordings_dir["writable"])
        provider_ok = not self.config.live_enabled or not self.config.missing_live_env
        return {
            "ok": storage_ok and provider_ok,
            "service": "telephony-ai-asterisk-voice-gateway",
            "version": service_version(),
            "environment": deployment_environment(),
            "role": "asterisk-ari-external-media-worker",
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "readiness": {
                "api": {"ok": True, "status": "ready", "service": "telephony-ai-asterisk-voice-gateway", "version": __version__},
                "storage": {
                    "ok": storage_ok,
                    "status": "ready" if storage_ok else "error",
                    "checks": {"calls": calls_dir, "recordings": recordings_dir},
                },
                "provider": {
                    "ok": provider_ok,
                    "status": "ready" if provider_ok else "missing_live_env",
                    "missing_env": self.config.missing_live_env,
                    "secrets_redacted": True,
                },
                "worker": {"ok": True, "status": "ready", "uptime_seconds": round(time.time() - self.started_at, 3)},
            },
            "config": self.config.public_status(),
            "storage": {"calls_dir": calls_dir, "recordings_dir": recordings_dir},
            "external_media": self.config.external_media_params(),
            "realtime_pipeline": {
                "module": f"{_PKG}.realtime",
                "status": "worker_cli_ready",
                "pipecat_dependency": "none",
                "livekit_dependency": "none",
                "current_live_mode": "legacy_voicebot_default_realtime_worker_opt_in",
                "hot_path_target": "Asterisk ExternalMedia RTP -> realtime worker -> streaming STT/LLM/TTS -> RTP",
                "commands": {
                    "status": f"python -m {_PKG}.realtime.worker --status",
                    "offline_synthetic_call": f"python -m {_PKG}.realtime.worker --offline-synthetic-call --offline-output-dir /tmp/openphonex-realtime-test",
                    "offline_recorded_call": f"python -m {_PKG}.realtime.worker --offline-recorded-audio /path/to/speech.wav --offline-output-dir /tmp/openphonex-realtime-recorded-test",
                    "offline_barge_in_check": f"python -m {_PKG}.realtime.worker --offline-barge-in-check --offline-output-dir /tmp/openphonex-realtime-barge-in-test",
                    "serve_health": f"python -m {_PKG}.realtime.worker --serve-health",
                    "serve_asterisk": f"REALTIME_LIVE_ENABLED=1 python -m {_PKG}.realtime.worker --serve-asterisk",
                    "serve_asterisk_with_health": f"REALTIME_LIVE_ENABLED=1 python -m {_PKG}.realtime.worker --serve-health --serve-asterisk",
                    "docker_health": "docker compose --profile realtime up -d --build realtime-worker",
                },
                "docker_profile": "realtime",
                "docker_service": "realtime-worker",
                "provider_modes": {
                    "fake": "frame-fed fake streaming STT, fake LLM, fake tone TTS",
                    "gemini_llm_fake_tts": "frame-fed fake streaming STT, real Gemini streaming LLM, fake tone TTS",
                    "elevenlabs_v3_tts": "frame-fed fake streaming STT/LLM with real ElevenLabs v3 HTTP streaming TTS",
                    "gemini_llm_elevenlabs_v3_tts": "frame-fed fake streaming STT with real Gemini streaming LLM and ElevenLabs v3 HTTP streaming TTS",
                },
                "stt_providers": {
                    "fake": "frame-fed fake STT for safe dry-runs",
                    "deepgram": "Deepgram live WebSocket STT; live Estonian/Asterisk validation pending",
                },
                "stt_status": "StreamingSTTSession boundary ready; Deepgram provider adapter wired; STT speech_final endpointing wired, live validation pending",
                "barge_in_status": "Realtime worker cancels outbound RTP and replays caller frames on detected barge-in; live Asterisk validation pending",
                "live_gate": "REALTIME_LIVE_ENABLED=1 plus ASTERISK_ARI_USER and ASTERISK_ARI_PASSWORD",
                "artifacts": ["caller.wav", "agent.wav", "mixed.wav", "latency.json", "transcript turns"],
            },
            "prototype_compatibility": {
                "stasis_app": self.config.app,
                "external_media_format": "ulaw/rtp/both",
                "recorded_audio": "external-media-in.wav",
                "legacy_args": self.config.legacy_voicebot_args(),
                "legacy_command_available": bool(self.config.legacy_voicebot_command()),
            },
        }

    def process_recorded_audio(self, source_wav: Path, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        start, end = detect_turn_window(
            source_wav,
            speech_rms=self.config.speech_rms,
            silence_rms=self.config.silence_rms,
            silence_hold_sec=self.config.silence_hold_sec,
            min_speech_sec=self.config.min_speech_sec,
            preroll_sec=self.config.preroll_sec,
        )
        segment = output_dir / "user-1.wav"
        extracted = extract_segment(source_wav, start, end, segment)
        result = {
            "ok": extracted,
            "source_wav": str(source_wav),
            "segment_wav": str(segment) if extracted else "",
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(max(0.0, end - start), 3),
            "source_max_rms": round(max_window_rms(source_wav), 3),
            "segment_max_rms": round(max_window_rms(segment), 3) if extracted else 0.0,
            "waveform_peaks": waveform_peaks(segment if extracted else source_wav),
        }
        (output_dir / "offline-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


def make_health_handler(worker: AsteriskVoiceGatewayWorker) -> type[BaseHTTPRequestHandler]:
    class VoiceGatewayHealthHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if env_enabled("OPENPHONEX_HTTP_LOGS"):
                structured_log(
                    "http_request",
                    service="telephony-ai-asterisk-voice-gateway",
                    force=True,
                    method=getattr(self, "command", ""),
                    path=self.path.split("?", 1)[0],
                    client=self.client_address[0] if self.client_address else "",
                )

        def do_GET(self) -> None:
            if self.path not in {"/", "/health", "/v1/voice-gateway/status"}:
                self._send(404, {"ok": False, "error": "not_found", "path": self.path})
                return
            self._send(200, worker.status())

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return VoiceGatewayHealthHandler


def run_health_server(worker: AsteriskVoiceGatewayWorker, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_health_handler(worker))
    print(f"Telephony AI Asterisk voice gateway listening on http://{host}:{port}")
    server.serve_forever()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telephony AI Asterisk ExternalMedia gateway worker")
    parser.add_argument("--env-file", default="", help="Optional env file. Values are overridden by real environment variables.")
    parser.add_argument("--status", action="store_true", help="Print redacted worker status and exit.")
    parser.add_argument("--print-legacy-command", action="store_true", help="Print the config-derived legacy voicebot command and exit.")
    parser.add_argument("--exec-legacy-script", action="store_true", help="Run OPENPHONEX_LEGACY_VOICEBOT_SCRIPT with config-derived arguments.")
    parser.add_argument("--offline-audio", default="", help="Process a recorded 8kHz PCM WAV without connecting to Asterisk.")
    parser.add_argument("--offline-output-dir", default="", help="Where offline artifacts should be written.")
    parser.add_argument("--serve-health", action="store_true", help="Run the worker health/status HTTP server.")
    parser.add_argument("--host", default=os.environ.get("OPENPHONEX_VOICE_WORKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENPHONEX_VOICE_WORKER_PORT", "8780")))
    args = parser.parse_args(argv)

    config = AsteriskVoiceGatewayConfig.from_env(Path(args.env_file) if args.env_file else None)
    worker = AsteriskVoiceGatewayWorker(config)
    if args.print_legacy_command:
        command = config.legacy_voicebot_command()
        print(json.dumps({"ok": bool(command), "command": command, "secrets_redacted": True}, indent=2, sort_keys=True))
        return 0 if command else 2
    if args.exec_legacy_script:
        command = config.legacy_voicebot_command()
        if not command:
            print(json.dumps({"ok": False, "error": "missing_legacy_script", "env": "OPENPHONEX_LEGACY_VOICEBOT_SCRIPT"}, indent=2, sort_keys=True))
            return 2
        if not Path(command[1]).exists():
            print(json.dumps({"ok": False, "error": "legacy_script_not_found", "path": command[1]}, indent=2, sort_keys=True))
            return 2
        return subprocess.run(command, check=False).returncode
    if args.offline_audio:
        output_dir = Path(args.offline_output_dir) if args.offline_output_dir else config.calls_dir / f"offline-{int(time.time())}"
        print(json.dumps(worker.process_recorded_audio(Path(args.offline_audio), output_dir), indent=2, sort_keys=True))
        return 0
    if args.status or not args.serve_health:
        print(json.dumps(worker.status(), indent=2, sort_keys=True))
        return 0
    run_health_server(worker, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

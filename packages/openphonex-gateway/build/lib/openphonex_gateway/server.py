from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:  # monolith: control-plane namespace present (telephony_ai.* compat shims)
    from telephony_ai import __version__
    from telephony_ai.runtime import deployment_environment, env_enabled, path_readiness, service_version, structured_log
except ModuleNotFoundError:  # standalone openphonex-gateway (OSS, no control plane)
    from openphonex_gateway import __version__
    from openphonex_common.runtime import deployment_environment, env_enabled, path_readiness, service_version, structured_log

# The package resolves under the monolith shim name (telephony_ai.media) or as
# the standalone openphonex_gateway package; report whichever loaded us so the
# `python -m ...` command hints below are copy-pasteable in either deployment.
_PKG = __package__ or "openphonex_gateway"


def media_status() -> dict[str, Any]:
    media_dir = Path(os.environ.get("OPENPHONEX_MEDIA_DIR", "/media-state"))
    calls_dir = Path(os.environ.get("OPENPHONEX_CALLS_DIR", "/calls"))
    recordings_dir = Path(os.environ.get("OPENPHONEX_RECORDINGS_DIR", "/recordings"))
    storage_checks = {
        "media": path_readiness(media_dir, label="media_dir"),
        "calls": path_readiness(calls_dir, label="calls_dir"),
        "recordings": path_readiness(recordings_dir, label="recordings_dir"),
    }
    storage_ok = all(check["ok"] for check in storage_checks.values())
    return {
        "ok": storage_ok,
        "service": "telephony-ai-media",
        "version": service_version(),
        "environment": deployment_environment(),
        "role": "media-control-shell",
        "audio_protocol": "pcm_s16le_8000_mono",
        "readiness": {
            "api": {"ok": True, "status": "ready", "service": "telephony-ai-media", "version": __version__},
            "storage": {"ok": storage_ok, "status": "ready" if storage_ok else "error", "checks": storage_checks},
            "worker": {"ok": True, "status": "media_shell_only"},
            "provider": {"ok": True, "status": "provider_agnostic"},
        },
        "rtp": {
            "start": int(os.environ.get("OPENPHONEX_RTP_START", "10000")),
            "end": int(os.environ.get("OPENPHONEX_RTP_END", "10100")),
        },
        "storage": {
            "media_dir": str(media_dir),
            "media_dir_exists": media_dir.exists(),
            "calls_dir": str(calls_dir),
            "calls_dir_exists": calls_dir.exists(),
            "recordings_dir": str(recordings_dir),
            "recordings_dir_exists": recordings_dir.exists(),
        },
        "asterisk_voice_gateway": {
            "module": f"{_PKG}.asterisk_voice_gateway",
            "status_command": f"python -m {_PKG}.asterisk_voice_gateway --status",
            "health_command": f"python -m {_PKG}.asterisk_voice_gateway --serve-health",
            "docker_profile": "voice",
        },
        "realtime_pipeline": {
            "module": f"{_PKG}.realtime",
            "status": "worker_cli_ready",
            "transport": "Asterisk ExternalMedia RTP",
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
            "stages": [
                "transport.input",
                "turn_detection",
                "streaming_stt_session",
                "streaming_llm",
                "streaming_tts",
                "transport.output",
                "barge_in_monitor",
                "recording",
                "latency_metrics",
            ],
            "pipecat_dependency": "none",
            "livekit_dependency": "none",
            "note": "Pipecat and LiveKit are architecture/adapter references for v1; OpenPhonex owns DIDWW, Asterisk, Number rules, compliance, billing, and recordings.",
        },
    }


class MediaHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        if env_enabled("OPENPHONEX_HTTP_LOGS"):
            structured_log(
                "http_request",
                service="telephony-ai-media",
                force=True,
                method=getattr(self, "command", ""),
                path=self.path.split("?", 1)[0],
                client=self.client_address[0] if self.client_address else "",
            )

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/health", "/v1/media/status"}:
            self.send_json(200, media_status())
            return
        self.send_json(404, {"ok": False, "error": "not_found", "path": self.path})


def run(host: str = "127.0.0.1", port: int = 8770) -> ThreadingHTTPServer:
    for key in ("OPENPHONEX_MEDIA_DIR", "OPENPHONEX_CALLS_DIR", "OPENPHONEX_RECORDINGS_DIR"):
        Path(os.environ.get(key, "/tmp")).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), MediaHandler)
    print(f"Telephony AI media service listening on http://{host}:{port}")
    server.serve_forever()
    return server


if __name__ == "__main__":
    run(os.environ.get("OPENPHONEX_MEDIA_HOST", "127.0.0.1"), int(os.environ.get("OPENPHONEX_MEDIA_PORT", "8770")))

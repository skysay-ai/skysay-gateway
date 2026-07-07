# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable

try:  # monolith: control-plane namespace present (telephony_ai.* compat shims)
    from telephony_ai.config import load_env_file
    from telephony_ai.runtime import deployment_environment, env_flag, service_version
except ModuleNotFoundError:  # standalone openphonex-gateway (OSS, no control plane)
    from openphonex_common.config import load_env_file
    from openphonex_common.runtime import deployment_environment, env_flag, service_version

from .adapters import FakeRealtimeProviderConfig, FakeStreamingSTT, TemplateStreamingLLM, ToneStreamingTTS
from .asterisk_external_media import (
    AriClientProtocol,
    AriWebSocketEventClient,
    AsteriskExternalMediaConfig,
    AsteriskExternalMediaSession,
    AsteriskRestClient,
)
from .barge_in import BargeInConfig
from .deepgram_streaming_stt import (
    DeepgramStreamingSTT,
    DeepgramStreamingSTTConfig,
    missing_deepgram_streaming_runtime_dependencies,
)
from .elevenlabs_v3_streaming import (
    ElevenLabsV3Config,
    ElevenLabsV3StreamingTTS,
    missing_elevenlabs_streaming_runtime_dependencies,
    resolve_elevenlabs_voice_id,
)
from .frames import AudioFrame
from .gemini_streaming import GeminiStreamingLLM, GeminiStreamingLLMConfig
from .metrics import CallMetrics
from .pipeline import LLMFallbackConfig, RealtimeMediaPipeline
from .recorder import StreamingCallRecorder
from .registry import ProviderRegistry, ProviderSpec
from .semantic_endpointing import SemanticEndpointConfig, SemanticEndpointDetector
from .transport import PacedRtpSender, UdpRtpTransport
from .turn_detection import RmsTurnDetector, TurnDetectionConfig


# --- Provider registry wiring (M5, SCH-202 / SCH-203) ----------------------
#
# The five canonical REALTIME_PROVIDER_MODE kinds expressed as an explicit
# (stt, llm, tts) triple of registry names. active_llm_name / active_tts_name
# read straight from this table when their component-specific env var is unset;
# active_stt_name keeps the legacy REALTIME_STT_PROVIDER coupling.
PROVIDER_MODE_TRIPLES: dict[str, tuple[str, str, str]] = {
    "fake": ("fake", "template", "tone"),
    "elevenlabs_v3_tts": ("fake", "template", "elevenlabs"),
    "gemini_llm_fake_tts": ("fake", "gemini", "tone"),
    "gemini_llm_elevenlabs_v3_tts": ("fake", "gemini", "elevenlabs"),
    "deepgram_gemini_elevenlabs_v3": ("deepgram", "gemini", "elevenlabs"),
}
_UNSUPPORTED_TRIPLE = ("unsupported", "unsupported", "unsupported")

# Registry names of the built-in providers per component (used to short-circuit
# custom-availability aggregation so built-in combos never touch the registry).
_BUILTIN_PROVIDER_NAMES: dict[str, set[str]] = {
    "stt": {"fake", "deepgram"},
    "llm": {"template", "gemini"},
    "tts": {"tone", "elevenlabs"},
}

_PROVIDER_REGISTRY: ProviderRegistry | None = None


def _fake_provider_config(config: "RealtimeWorkerConfig") -> FakeRealtimeProviderConfig:
    return FakeRealtimeProviderConfig(
        transcript=config.fake_transcript,
        reply=config.fake_reply,
        language=config.fake_language,
    )


# Built-in factory closures. They resolve the adapter classes from THIS module's
# globals at call time, so ``mock.patch("telephony_ai.media.realtime.worker.
# DeepgramStreamingSTT", Fake)`` (and Gemini/ElevenLabs) still swaps the class
# the factory constructs. Do not capture the class at registration time.
def _make_fake_stt(config: "RealtimeWorkerConfig") -> object:
    return FakeStreamingSTT(_fake_provider_config(config))


def _make_deepgram_stt(config: "RealtimeWorkerConfig") -> object:
    return DeepgramStreamingSTT(
        DeepgramStreamingSTTConfig(
            api_key=config.deepgram_api_key,
            model_id=config.deepgram_model_id,
            language=config.deepgram_language,
            endpointing_ms=config.deepgram_endpointing_ms,
            utterance_end_ms=config.deepgram_utterance_end_ms,
            timeout_seconds=config.deepgram_timeout_seconds,
            finalize_timeout_seconds=config.deepgram_finalize_timeout_seconds,
        )
    )


def _make_template_llm(config: "RealtimeWorkerConfig") -> object:
    return TemplateStreamingLLM(_fake_provider_config(config))


def _make_gemini_llm(config: "RealtimeWorkerConfig") -> object:
    return GeminiStreamingLLM(
        GeminiStreamingLLMConfig(
            api_key=config.gemini_api_key,
            model_id=config.gemini_model_id,
            temperature=config.gemini_temperature,
            max_output_tokens=config.gemini_max_output_tokens,
            thinking_budget=config.gemini_thinking_budget,
            system_prompt=config.gemini_system_prompt,
            timeout_seconds=config.gemini_timeout_seconds,
        )
    )


def _make_tone_tts(config: "RealtimeWorkerConfig") -> object:
    return ToneStreamingTTS(_fake_provider_config(config))


def _make_elevenlabs_tts(config: "RealtimeWorkerConfig") -> object:
    # config.elevenlabs_voice_id is pre-resolved by _provider_adapters (via the
    # worker's cached _resolve_elevenlabs_voice_id) so this factory never repeats
    # the voice-name network lookup.
    return ElevenLabsV3StreamingTTS(
        ElevenLabsV3Config(
            api_key=config.elevenlabs_api_key,
            voice_id=config.elevenlabs_voice_id,
            model_id=config.elevenlabs_model_id,
            output_format=config.elevenlabs_output_format,
            language_code=config.elevenlabs_language_code,
            stability=config.elevenlabs_stability,
            similarity_boost=config.elevenlabs_similarity_boost,
            chunk_size=config.elevenlabs_chunk_size,
        )
    )


# Built-in describe fragments — byte-identical to the pre-M5 provider_components
# dicts (down to the presence/absence of every key) so downstream contract and
# test assertions are unchanged.
def _describe_fake_stt(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "fake", "adapter": "FakeStreamingSTT", "real": False, "language": config.fake_language, "model_id": ""}


def _describe_deepgram_stt(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "deepgram", "adapter": "DeepgramStreamingSTT", "real": True, "language": config.deepgram_language, "model_id": config.deepgram_model_id}


def _describe_template_llm(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "template", "adapter": "TemplateStreamingLLM", "real": False, "model_id": ""}


def _describe_gemini_llm(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "gemini", "adapter": "GeminiStreamingLLM", "real": True, "model_id": config.gemini_model_id}


def _describe_tone_tts(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "tone", "adapter": "ToneStreamingTTS", "real": False, "model_id": "", "language_code": ""}


def _describe_elevenlabs_tts(config: "RealtimeWorkerConfig") -> dict:
    return {"provider": "elevenlabs", "adapter": "ElevenLabsV3StreamingTTS", "real": True, "model_id": config.elevenlabs_model_id, "language_code": config.elevenlabs_language_code}


def _build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(ProviderSpec("fake", "stt", _make_fake_stt, _describe_fake_stt, real=False))
    registry.register(ProviderSpec("deepgram", "stt", _make_deepgram_stt, _describe_deepgram_stt, real=True))
    registry.register(ProviderSpec("template", "llm", _make_template_llm, _describe_template_llm, real=False))
    registry.register(ProviderSpec("gemini", "llm", _make_gemini_llm, _describe_gemini_llm, real=True))
    registry.register(ProviderSpec("tone", "tts", _make_tone_tts, _describe_tone_tts, real=False))
    registry.register(ProviderSpec("elevenlabs", "tts", _make_elevenlabs_tts, _describe_elevenlabs_tts, real=True))
    # Optional local-model adapters (SCH-203) are registered lazily so the heavy
    # deps (faster-whisper / numpy / onnxruntime) never load at import time and
    # importing this module stays as light as it was before M5.
    prefix = __package__ or "telephony_ai.media.realtime"
    registry.register_lazy("stt", "faster_whisper", f"{prefix}.faster_whisper_stt:PROVIDER_SPEC")
    registry.register_lazy("llm", "openai_compatible", f"{prefix}.openai_compatible_llm:PROVIDER_SPEC")
    registry.register_alias("llm", "ollama", "openai_compatible")
    registry.register_alias("llm", "vllm", "openai_compatible")
    registry.register_alias("llm", "local_llm", "openai_compatible")
    registry.register_lazy("tts", "piper", f"{prefix}.local_tts:PIPER_SPEC")
    registry.register_lazy("tts", "kokoro", f"{prefix}.local_tts:KOKORO_SPEC")
    # GA-5 hosted provider adapters (SCH-216 / SCH-206 / SCH-211). Registered
    # lazily under the catalog's registry_names so voice_catalog selectability
    # flips true: each module imports stdlib-only at top level (heavy SDKs — only
    # google-cloud-speech — are deferred inside methods), so can_resolve()
    # succeeds even before a key/SDK is present.
    registry.register_lazy("stt", "google", f"{prefix}.google_stt:PROVIDER_SPEC")
    registry.register_lazy("stt", "openai", f"{prefix}.openai_streaming:STT_SPEC")
    registry.register_lazy("stt", "elevenlabs_scribe", f"{prefix}.elevenlabs_scribe_stt:PROVIDER_SPEC")
    registry.register_lazy("llm", "claude", f"{prefix}.anthropic_streaming:PROVIDER_SPEC")
    registry.register_alias("llm", "anthropic", "claude")
    registry.register_lazy("llm", "openai", f"{prefix}.openai_streaming:LLM_SPEC")
    # Catalog display id "gpt" -> registry_name "openai" (also lets
    # REALTIME_LLM_PROVIDER=gpt resolve to the OpenAI chat adapter).
    registry.register_alias("llm", "gpt", "openai")
    registry.register_lazy("tts", "deepgram_tts", f"{prefix}.deepgram_aura_tts:PROVIDER_SPEC")
    registry.register_alias("tts", "aura", "deepgram_tts")
    registry.register_lazy("tts", "cartesia", f"{prefix}.cartesia_tts:PROVIDER_SPEC")
    return registry


def _provider_registry() -> ProviderRegistry:
    global _PROVIDER_REGISTRY
    if _PROVIDER_REGISTRY is None:
        _PROVIDER_REGISTRY = _build_default_registry()
    return _PROVIDER_REGISTRY


def _resolve_override_component_name(component: str, raw: str) -> str:
    """Map an explicit provider env value to a registry name (or 'unsupported').

    Applies the alias table, then confirms the registry can resolve the name
    (registered built-in, lazy local adapter, entry point, or dotted path).
    """

    raw = (raw or "").strip()
    if not raw:
        return "unsupported"
    registry = _provider_registry()
    canonical = registry.canonical_name(component, raw)
    if registry.can_resolve(component, canonical):
        return canonical
    return "unsupported"


def _python_executable() -> str:
    override = str(os.environ.get("OPENPHONEX_PYTHON", "")).strip()
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[4]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


# Report whichever import name actually loaded this module: the monolith shim
# name telephony_ai.media.realtime.worker, or openphonex_gateway.realtime.worker
# as the standalone OSS package — keeps `python -m ...` hints copy-pasteable in
# both. `mock.patch` sites and tests that assert on the string see the name that
# loaded them, so both the monolith and the exported (import-rewritten) suites
# assert the correct value.
_WORKER_MODULE = (
    __spec__.name if __spec__ is not None
    else (__name__ if __name__ != "__main__" else "openphonex_gateway.realtime.worker")
)


def _worker_command(args: str) -> str:
    return f"{shlex.quote(_python_executable())} -m {_WORKER_MODULE} {args}".strip()


def _default_log_sink(line: str) -> None:
    """Emit one structured worker log line to stderr (unbuffered).

    The serve/ARI path historically printed nothing, so a healthy connect, a
    hung connect, and a crash-looping connect all looked identical from
    ``docker logs`` ("empty"). Logs go to stderr so the ``--serve-asterisk``
    stdout JSON result stays clean and machine-parseable.
    """

    sys.stderr.write(line + "\n")
    sys.stderr.flush()


DEFAULT_OPENING_UTTERANCE = "Tere! Siin OpenPhonexi digiassistent. Kas teil on hetk aega?"
# SCH-210: an inbound caller dialed us, so the bot must greet FIRST instead of
# waiting silently for the caller to speak. Phrased for the inbound direction
# ("how can I help") rather than the outbound opener ("do you have a moment").
DEFAULT_INBOUND_OPENING_UTTERANCE = "Tere! Siin OpenPhonexi digiassistent. Kuidas saan teid aidata?"


@dataclass(frozen=True)
class RealtimeWorkerConfig:
    calls_dir: Path
    recordings_dir: Path
    rtp_bind: str
    rtp_port: int
    asterisk: AsteriskExternalMediaConfig
    rtp_port_end: int | None = None
    health_bind: str = "127.0.0.1"
    health_port: int = 18772
    # GA-6 / SCH-217: bounded-backoff retry around the ARI connect in
    # serve_asterisk_with_health. max_attempts=0 means retry forever (the
    # resilient container default so a transient "Asterisk not ready yet" at
    # startup self-heals instead of silently tearing the worker down). Backoff
    # grows linearly per attempt and is capped at backoff_max_seconds.
    ari_connect_max_attempts: int = 0
    ari_connect_backoff_seconds: float = 2.0
    ari_connect_backoff_max_seconds: float = 30.0
    provider_mode: str = "fake"
    stt_provider: str = "fake"
    llm_provider: str = ""
    tts_provider: str = ""
    live_enabled: bool = False
    max_concurrent_calls: int = 0
    max_turns: int = 4
    silence_hold_ms: int = 350
    min_speech_ms: int = 250
    # SCH-210: barge-in RMS threshold for the LIVE call pipeline. 650 is sensitive
    # (any caller frame cancels agent audio); raising it toward 32000 makes the
    # demo effectively half-duplex so telephony echo/comfort-noise no longer cuts
    # the greeting or replies. Sourced from REALTIME_BARGE_IN_RMS in from_env().
    barge_in_speech_rms: float = 650.0
    semantic_endpointing_enabled: bool = True
    semantic_endpointing_min_words: int = 2
    semantic_endpointing_min_chars: int = 6
    semantic_endpointing_terminal_punctuation: tuple[str, ...] = (".", "!", "?")
    accept_interim_on_rms_endpoint: bool = False
    max_pending_phrases: int = 4
    tts_rtp_prebuffer_ms: int = 0
    idle_timeout_seconds: float = 30.0
    fake_transcript: str = "Tere, soovin aega broneerida."
    fake_reply: str = "Tere! Mis auto mudeliga tulete?"
    fake_language: str = "et"
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_voice: str = ""
    elevenlabs_model_id: str = "eleven_v3"
    elevenlabs_output_format: str = "mp3_44100_128"
    elevenlabs_language_code: str = ""
    elevenlabs_stability: float = 0.45
    elevenlabs_similarity_boost: float = 0.8
    elevenlabs_chunk_size: int = 4096
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-2.5-flash-lite"
    gemini_temperature: float = 0.3
    gemini_max_output_tokens: int = 120
    gemini_thinking_budget: int | None = 0
    gemini_system_prompt: str = ""
    gemini_timeout_seconds: float = 30.0
    llm_fallback_model_id: str = "gemini-2.5-flash-lite"
    llm_retry_budget_seconds: float = 2.0
    recovery_utterance: str = ""
    opening_utterance: str = DEFAULT_OPENING_UTTERANCE
    # SCH-210: inbound greeting text. The dataclass default is EMPTY on purpose so
    # direct RealtimeWorkerConfig(...) construction (the worker unit tests) keeps
    # the legacy "stay silent on inbound" behavior; from_env() supplies the real
    # production default so live inbound calls greet first.
    inbound_opening_utterance: str = ""
    outbound_answer_timeout_seconds: float = 15.0
    deepgram_api_key: str = ""
    deepgram_model_id: str = "nova-3"
    deepgram_language: str = "et"
    deepgram_endpointing_ms: int = 300
    deepgram_utterance_end_ms: int = 1000
    deepgram_timeout_seconds: float = 3.0
    deepgram_finalize_timeout_seconds: float = 2.0
    operator_approval_required: bool = False
    operator_approval_record: str = ""
    operator_approval_packet_fingerprint: str = ""
    operator_approval_required_directions: tuple[str, ...] = ("inbound", "outbound")
    cutover_readiness_required: bool = False
    cutover_readiness_report: str = ""
    cutover_readiness_fingerprint: str = ""
    control_plane_api_url: str = ""
    gateway_deployment_id: str = ""
    gateway_api_token: str = ""
    gateway_public_url: str = ""
    gateway_local_status_url: str = ""
    gateway_report_interval_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "RealtimeWorkerConfig":
        rtp_port = int(os.environ.get("VOICE_RTP_PORT", os.environ.get("RTP_START", "10000")))
        default_pool_size = max(1, int(os.environ.get("REALTIME_RTP_PORT_POOL_SIZE", "20")))
        rtp_port_end = int(
            os.environ.get(
                "VOICE_RTP_PORT_END",
                os.environ.get("REALTIME_VOICE_RTP_PORT_END", str(rtp_port + default_pool_size - 1)),
            )
        )
        return cls(
            calls_dir=Path(os.environ.get("CALLS_DIR", "/calls")),
            recordings_dir=Path(os.environ.get("RECORDINGS_DIR", "/recordings")),
            rtp_bind=os.environ.get("VOICE_RTP_BIND", "0.0.0.0"),
            rtp_port=rtp_port,
            rtp_port_end=max(rtp_port, rtp_port_end),
            health_bind=os.environ.get("REALTIME_HEALTH_BIND", "127.0.0.1"),
            health_port=int(os.environ.get("REALTIME_HEALTH_PORT", "18772")),
            ari_connect_max_attempts=max(0, int(os.environ.get("REALTIME_ARI_CONNECT_MAX_ATTEMPTS", "0"))),
            ari_connect_backoff_seconds=max(0.0, float(os.environ.get("REALTIME_ARI_CONNECT_BACKOFF_SECONDS", "2"))),
            ari_connect_backoff_max_seconds=max(0.0, float(os.environ.get("REALTIME_ARI_CONNECT_BACKOFF_MAX_SECONDS", "30"))),
            asterisk=AsteriskExternalMediaConfig(
                ari_url=os.environ.get("ASTERISK_ARI_URL", "http://127.0.0.1:8088/ari"),
                ari_user=os.environ.get("ASTERISK_ARI_USER", ""),
                ari_password=os.environ.get("ASTERISK_ARI_PASSWORD", ""),
                app=os.environ.get("ASTERISK_ARI_APP", "voicebot"),
                external_host=os.environ.get("ASTERISK_EXTERNAL_MEDIA_HOST", "telephony-ai-realtime-worker"),
                rtp_port=rtp_port,
                asterisk_rtp_host=os.environ.get("ASTERISK_RTP_HOST", ""),
            ),
            provider_mode=os.environ.get("REALTIME_PROVIDER_MODE", "fake"),
            stt_provider=os.environ.get("REALTIME_STT_PROVIDER", "fake"),
            llm_provider=os.environ.get("REALTIME_LLM_PROVIDER", ""),
            tts_provider=os.environ.get("REALTIME_TTS_PROVIDER", ""),
            live_enabled=env_flag(os.environ.get("REALTIME_LIVE_ENABLED"), default=False),
            max_concurrent_calls=max(0, int(os.environ.get("REALTIME_MAX_CONCURRENT_CALLS", "0"))),
            max_turns=int(os.environ.get("REALTIME_MAX_TURNS", "4")),
            silence_hold_ms=int(os.environ.get("REALTIME_SILENCE_HOLD_MS", "350")),
            min_speech_ms=int(os.environ.get("REALTIME_MIN_SPEECH_MS", "250")),
            barge_in_speech_rms=float(os.environ.get("REALTIME_BARGE_IN_RMS", "650")),
            semantic_endpointing_enabled=env_flag(os.environ.get("REALTIME_SEMANTIC_ENDPOINTING_ENABLED"), default=True),
            semantic_endpointing_min_words=max(1, int(os.environ.get("REALTIME_SEMANTIC_ENDPOINTING_MIN_WORDS", "2"))),
            semantic_endpointing_min_chars=max(1, int(os.environ.get("REALTIME_SEMANTIC_ENDPOINTING_MIN_CHARS", "6"))),
            semantic_endpointing_terminal_punctuation=_semantic_punctuation_env(
                os.environ.get("REALTIME_SEMANTIC_ENDPOINTING_TERMINAL_PUNCTUATION")
            ),
            accept_interim_on_rms_endpoint=env_flag(
                os.environ.get("REALTIME_ACCEPT_INTERIM_ON_RMS_ENDPOINT"),
                default=False,
            ),
            max_pending_phrases=max(1, int(os.environ.get("REALTIME_MAX_PENDING_PHRASES", "4"))),
            tts_rtp_prebuffer_ms=max(0, int(os.environ.get("REALTIME_TTS_RTP_PREBUFFER_MS", "0"))),
            idle_timeout_seconds=float(os.environ.get("REALTIME_IDLE_TIMEOUT_SECONDS", "30")),
            fake_transcript=os.environ.get("REALTIME_FAKE_TRANSCRIPT", "Tere, soovin aega broneerida."),
            fake_reply=os.environ.get("REALTIME_FAKE_REPLY", "Tere! Mis auto mudeliga tulete?"),
            fake_language=os.environ.get("REALTIME_LANGUAGE", "et"),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.environ.get("ELEVENLABS_VOICE_ID", ""),
            elevenlabs_voice=os.environ.get("ELEVENLABS_VOICE", ""),
            elevenlabs_model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3"),
            elevenlabs_output_format=os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128"),
            elevenlabs_language_code=os.environ.get("ELEVENLABS_LANGUAGE_CODE", os.environ.get("REALTIME_LANGUAGE", "")),
            elevenlabs_stability=float(os.environ.get("ELEVENLABS_STABILITY", "0.45")),
            elevenlabs_similarity_boost=float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.8")),
            elevenlabs_chunk_size=int(os.environ.get("ELEVENLABS_CHUNK_SIZE", "4096")),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model_id=os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite"),
            gemini_temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.3")),
            gemini_max_output_tokens=int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "120")),
            gemini_thinking_budget=_optional_int_env("GEMINI_THINKING_BUDGET", default=0),
            gemini_system_prompt=os.environ.get("GEMINI_SYSTEM_PROMPT", ""),
            gemini_timeout_seconds=float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30")),
            llm_fallback_model_id=os.environ.get("REALTIME_LLM_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
            llm_retry_budget_seconds=max(0.1, float(os.environ.get("REALTIME_LLM_RETRY_BUDGET_SECONDS", "2"))),
            recovery_utterance=os.environ.get("REALTIME_RECOVERY_UTTERANCE", ""),
            opening_utterance=os.environ.get("REALTIME_OPENING_UTTERANCE", DEFAULT_OPENING_UTTERANCE),
            inbound_opening_utterance=os.environ.get(
                "REALTIME_INBOUND_OPENING_UTTERANCE",
                os.environ.get("REALTIME_OPENING_UTTERANCE", DEFAULT_INBOUND_OPENING_UTTERANCE),
            ),
            outbound_answer_timeout_seconds=max(0.0, float(os.environ.get("REALTIME_OUTBOUND_ANSWER_TIMEOUT_SECONDS", "15"))),
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
            deepgram_model_id=os.environ.get("DEEPGRAM_MODEL_ID", "nova-3"),
            deepgram_language=os.environ.get("DEEPGRAM_LANGUAGE", os.environ.get("REALTIME_LANGUAGE", "et")),
            deepgram_endpointing_ms=int(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "300")),
            deepgram_utterance_end_ms=int(os.environ.get("DEEPGRAM_UTTERANCE_END_MS", "1000")),
            deepgram_timeout_seconds=float(os.environ.get("DEEPGRAM_TIMEOUT_SECONDS", "3")),
            deepgram_finalize_timeout_seconds=float(os.environ.get("DEEPGRAM_FINALIZE_TIMEOUT_SECONDS", "2")),
            operator_approval_required=env_flag(os.environ.get("REALTIME_REQUIRE_OPERATOR_APPROVAL"), default=False),
            operator_approval_record=os.environ.get("REALTIME_OPERATOR_APPROVAL_RECORD", ""),
            operator_approval_packet_fingerprint=os.environ.get("REALTIME_APPROVAL_PACKET_FINGERPRINT", ""),
            operator_approval_required_directions=_approval_directions_env(
                os.environ.get("REALTIME_OPERATOR_APPROVAL_DIRECTIONS")
            ),
            cutover_readiness_required=env_flag(os.environ.get("REALTIME_REQUIRE_CUTOVER_READINESS"), default=False),
            cutover_readiness_report=os.environ.get("REALTIME_CUTOVER_READINESS_REPORT", ""),
            cutover_readiness_fingerprint=os.environ.get("REALTIME_CUTOVER_READINESS_FINGERPRINT", ""),
            control_plane_api_url=os.environ.get("OPENPHONEX_API_URL", "").rstrip("/"),
            gateway_deployment_id=os.environ.get("OPENPHONEX_GATEWAY_ID", ""),
            gateway_api_token=os.environ.get("OPENPHONEX_GATEWAY_API_TOKEN", ""),
            gateway_public_url=os.environ.get("OPENPHONEX_GATEWAY_PUBLIC_URL", ""),
            gateway_local_status_url=os.environ.get("OPENPHONEX_GATEWAY_LOCAL_STATUS_URL", ""),
            gateway_report_interval_seconds=max(1.0, float(os.environ.get("OPENPHONEX_GATEWAY_REPORT_INTERVAL_SECONDS", "15"))),
        )

    def provider_kind(self) -> str:
        return _normalize_provider_mode(self.provider_mode)

    def _mode_triple(self) -> tuple[str, str, str]:
        return PROVIDER_MODE_TRIPLES.get(self.provider_kind(), _UNSUPPORTED_TRIPLE)

    def stt_kind(self) -> str:
        stt_provider = _normalize_stt_provider(self.stt_provider)
        if self.provider_kind() == "deepgram_gemini_elevenlabs_v3" and stt_provider == "fake":
            return "deepgram"
        if stt_provider == "unsupported":
            # A non-legacy REALTIME_STT_PROVIDER value may still name a registry
            # provider (registered name / entry point / dotted path).
            return _resolve_override_component_name("stt", self.stt_provider)
        return stt_provider

    def llm_kind(self) -> str:
        raw = (self.llm_provider or "").strip()
        if not raw:
            return self._mode_triple()[1]
        return _resolve_override_component_name("llm", raw)

    def tts_kind(self) -> str:
        raw = (self.tts_provider or "").strip()
        if not raw:
            return self._mode_triple()[2]
        return _resolve_override_component_name("tts", raw)

    def active_stt_name(self) -> str:
        return self.stt_kind()

    def active_llm_name(self) -> str:
        return self.llm_kind()

    def active_tts_name(self) -> str:
        return self.tts_kind()

    def uses_deepgram_stt(self) -> bool:
        return self.active_stt_name() == "deepgram"

    def uses_elevenlabs_tts(self) -> bool:
        return self.active_tts_name() == "elevenlabs"

    def uses_gemini_llm(self) -> bool:
        return self.active_llm_name() == "gemini"

    def _custom_provider_availability(self) -> list[str]:
        """Missing deps/env reported by non-built-in (custom/local) providers.

        Built-in providers keep their explicit env checks in
        ``missing_provider_env`` and contribute nothing here, so this is a no-op
        (and never touches the registry) for the legacy env combinations.
        """

        components = [
            ("stt", self.active_stt_name()),
            ("llm", self.active_llm_name()),
            ("tts", self.active_tts_name()),
        ]
        custom = [
            (component, name)
            for component, name in components
            if name != "unsupported" and name not in _BUILTIN_PROVIDER_NAMES.get(component, set())
        ]
        if not custom:
            return []
        registry = _provider_registry()
        missing: list[str] = []
        for component, name in custom:
            spec = registry.get(component, name)
            if spec is not None:
                missing.extend(spec.missing(self))
        return missing

    def missing_provider_env(self) -> list[str]:
        missing: list[str] = []
        if self.uses_deepgram_stt() and not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if self.uses_gemini_llm() and not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if self.uses_elevenlabs_tts() and not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
        if self.uses_elevenlabs_tts() and not self.elevenlabs_voice_id and not self.elevenlabs_voice:
            missing.append("ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE")
        for item in self._custom_provider_availability():
            if item not in missing:
                missing.append(item)
        return missing

    def runtime_dependency_status(self) -> dict[str, object]:
        required: list[dict[str, object]] = []
        missing: list[str] = []
        if self.uses_deepgram_stt():
            deepgram_missing = set(missing_deepgram_streaming_runtime_dependencies())
            missing.extend(sorted(deepgram_missing))
            required.append(
                {
                    "name": "websocket-client",
                    "configured": "websocket-client" not in deepgram_missing,
                    "reason": "Deepgram live WebSocket streaming STT",
                    "used_by": ["stt", "recorded_audio_validation"],
                }
            )
        if self.uses_elevenlabs_tts():
            elevenlabs_missing = set(missing_elevenlabs_streaming_runtime_dependencies())
            missing.extend(sorted(elevenlabs_missing))
            required.append(
                {
                    "name": "ffmpeg",
                    "configured": "ffmpeg" not in elevenlabs_missing,
                    "reason": "ElevenLabs v3 HTTP streaming decode/transcode to 8 kHz PCM RTP",
                    "used_by": ["tts", "recorded_audio_validation"],
                }
            )
        unique_missing = list(dict.fromkeys(missing))
        return {"ok": not unique_missing, "missing": unique_missing, "required": required}

    def missing_runtime_dependencies(self) -> list[str]:
        status = self.runtime_dependency_status()
        missing = status.get("missing")
        if not isinstance(missing, list):
            return []
        return [str(item) for item in missing]

    def missing_live_env(self) -> list[str]:
        missing: list[str] = []
        if not self.live_enabled:
            missing.append("REALTIME_LIVE_ENABLED")
        missing.extend(self.asterisk.redacted()["missing_live_env"])
        if self.operator_approval_required:
            if not self.operator_approval_record:
                missing.append("REALTIME_OPERATOR_APPROVAL_RECORD")
            if not self.operator_approval_packet_fingerprint:
                missing.append("REALTIME_APPROVAL_PACKET_FINGERPRINT")
        if self.cutover_readiness_required:
            if not self.cutover_readiness_report:
                missing.append("REALTIME_CUTOVER_READINESS_REPORT")
            if not self.cutover_readiness_fingerprint:
                missing.append("REALTIME_CUTOVER_READINESS_FINGERPRINT")
        return missing

    def unresolved_live_env_placeholders(self) -> list[str]:
        candidates = [
            ("ASTERISK_ARI_URL", self.asterisk.ari_url),
            ("ASTERISK_ARI_USER", self.asterisk.ari_user),
            ("ASTERISK_ARI_PASSWORD", self.asterisk.ari_password),
            ("ASTERISK_EXTERNAL_MEDIA_HOST", self.asterisk.external_host),
        ]
        if self.operator_approval_required:
            candidates.extend(
                [
                    ("REALTIME_OPERATOR_APPROVAL_RECORD", self.operator_approval_record),
                    ("REALTIME_APPROVAL_PACKET_FINGERPRINT", self.operator_approval_packet_fingerprint),
                ]
            )
        if self.cutover_readiness_required:
            candidates.extend(
                [
                    ("REALTIME_CUTOVER_READINESS_REPORT", self.cutover_readiness_report),
                    ("REALTIME_CUTOVER_READINESS_FINGERPRINT", self.cutover_readiness_fingerprint),
                ]
            )
        return [name for name, value in candidates if _unresolved_env_placeholder(value)]

    def rtp_ports(self) -> list[int]:
        return list(range(self.rtp_port, self.resolved_rtp_port_end() + 1))

    def effective_call_capacity(self) -> int:
        port_capacity = len(self.rtp_ports())
        if self.max_concurrent_calls <= 0:
            return port_capacity
        return min(self.max_concurrent_calls, port_capacity)

    def resolved_rtp_port_end(self) -> int:
        return max(self.rtp_port, self.rtp_port_end or self.rtp_port)

    def semantic_endpoint_config(self) -> SemanticEndpointConfig:
        return SemanticEndpointConfig(
            enabled=self.semantic_endpointing_enabled,
            min_words=self.semantic_endpointing_min_words,
            min_chars=self.semantic_endpointing_min_chars,
            terminal_punctuation=self.semantic_endpointing_terminal_punctuation,
        )

    def redacted(self) -> dict[str, object]:
        return {
            "calls_dir": str(self.calls_dir),
            "recordings_dir": str(self.recordings_dir),
            "rtp": {
                "bind": self.rtp_bind,
                "port": self.rtp_port,
                "port_end": self.resolved_rtp_port_end(),
                "port_count": len(self.rtp_ports()),
                "tts_prebuffer_ms": self.tts_rtp_prebuffer_ms,
            },
            "health": {"bind": self.health_bind, "port": self.health_port},
            "provider_mode": self.provider_mode,
            "provider_kind": self.provider_kind(),
            "stt_provider": self.stt_provider,
            "stt_kind": self.stt_kind(),
            "missing_provider_env": self.missing_provider_env(),
            "missing_runtime_dependencies": self.missing_runtime_dependencies(),
            "runtime_dependencies": self.runtime_dependency_status(),
            "missing_live_env": self.missing_live_env(),
            "unresolved_live_env_placeholders": self.unresolved_live_env_placeholders(),
            "fake_provider": {
                "language": self.fake_language,
                "transcript_configured": bool(self.fake_transcript),
                "reply_configured": bool(self.fake_reply),
            },
            "elevenlabs": {
                "configured": bool(self.elevenlabs_api_key),
                "voice_id_configured": bool(self.elevenlabs_voice_id),
                "voice_name_configured": bool(self.elevenlabs_voice),
                "model_id": self.elevenlabs_model_id,
                "output_format": self.elevenlabs_output_format,
                "language_code": self.elevenlabs_language_code,
                "chunk_size": self.elevenlabs_chunk_size,
            },
            "gemini": {
                "configured": bool(self.gemini_api_key),
                "model_id": self.gemini_model_id,
                "temperature": self.gemini_temperature,
                "max_output_tokens": self.gemini_max_output_tokens,
                "thinking_budget_configured": self.gemini_thinking_budget is not None,
                "system_prompt_configured": bool(self.gemini_system_prompt.strip()),
                "timeout_seconds": self.gemini_timeout_seconds,
            },
            "llm_fallback": {
                "model_id": self.llm_fallback_model_id,
                "retry_budget_seconds": self.llm_retry_budget_seconds,
                "recovery_utterance_configured": bool(self.recovery_utterance.strip()),
            },
            "outbound_initiative": {
                "opening_utterance_configured": bool(self.opening_utterance.strip()),
                "answer_timeout_seconds": self.outbound_answer_timeout_seconds,
            },
            "deepgram": {
                "configured": bool(self.deepgram_api_key),
                "model_id": self.deepgram_model_id,
                "language": self.deepgram_language,
                "endpointing_ms": self.deepgram_endpointing_ms,
                "utterance_end_ms": self.deepgram_utterance_end_ms,
                "timeout_seconds": self.deepgram_timeout_seconds,
                "finalize_timeout_seconds": self.deepgram_finalize_timeout_seconds,
            },
            "live_enabled": self.live_enabled,
            "max_concurrent_calls": self.max_concurrent_calls,
            "effective_call_capacity": self.effective_call_capacity(),
            "max_turns": self.max_turns,
            "silence_hold_ms": self.silence_hold_ms,
            "min_speech_ms": self.min_speech_ms,
            "semantic_endpointing": {
                "enabled": self.semantic_endpointing_enabled,
                "min_words": self.semantic_endpointing_min_words,
                "min_chars": self.semantic_endpointing_min_chars,
                "terminal_punctuation": "".join(self.semantic_endpointing_terminal_punctuation),
                "accept_interim_on_rms_endpoint": self.accept_interim_on_rms_endpoint,
            },
            "max_pending_phrases": self.max_pending_phrases,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "asterisk": self.asterisk.redacted(),
            "operator_approval": {
                "required": self.operator_approval_required,
                "record_configured": bool(self.operator_approval_record),
                "fingerprint_configured": bool(self.operator_approval_packet_fingerprint),
                "fingerprint_short": self.operator_approval_packet_fingerprint[:12],
                "required_directions": list(self.operator_approval_required_directions),
            },
            "cutover_readiness": {
                "required": self.cutover_readiness_required,
                "report_configured": bool(self.cutover_readiness_report),
                "fingerprint_configured": bool(self.cutover_readiness_fingerprint),
                "fingerprint_short": self.cutover_readiness_fingerprint[:12],
            },
            "control_plane_reporter": {
                "configured": self.control_plane_reporter_configured(),
                "api_url": self.control_plane_api_url,
                "gateway_deployment_id": self.gateway_deployment_id,
                "api_token_configured": bool(self.gateway_api_token),
                "public_url_configured": bool(self.gateway_public_url),
                "local_status_url": self.gateway_local_status_url,
                "interval_seconds": self.gateway_report_interval_seconds,
                "secrets_redacted": True,
            },
            "artifact_import_queue": {
                "configured": self.artifact_import_queue_configured(),
                "api_url": self.control_plane_api_url,
                "api_token_configured": bool(self.gateway_api_token),
                "job_kind": "call.artifact_import",
                "secrets_redacted": True,
            },
        }

    def control_plane_reporter_configured(self) -> bool:
        return bool(self.control_plane_api_url and self.gateway_deployment_id and self.gateway_api_token)

    def artifact_import_queue_configured(self) -> bool:
        return bool(self.control_plane_api_url and self.gateway_api_token)


class RealtimeMediaWorker:
    def __init__(self, config: RealtimeWorkerConfig):
        self.config = config
        self.started_at = time.time()
        # Structured log sink for the long-running serve path. Injectable so
        # tests can capture emitted lines deterministically; defaults to stderr.
        self._log_sink: Callable[[str], None] = _default_log_sink
        self._resolved_elevenlabs_voice_id: str | None = None
        self._provider_lock = threading.RLock()
        # Per-agent adapter triples keyed by a stack fingerprint so two agents
        # in one worker process get distinct STT/LLM/TTS adapters. The default
        # env stack (self.config) is just one more fingerprint in this cache.
        self._provider_adapter_cache: dict[tuple, dict[str, object]] = {}
        # Resolved per-agent realtime configs fetched once from the control
        # plane and cached for the process lifetime, keyed by agent id.
        self._agent_config_lock = threading.RLock()
        self._agent_config_cache: dict[str, dict[str, object] | None] = {}
        self._capacity_lock = threading.Lock()
        self._active_call_ids: set[str] = set()
        self._active_call_details: dict[str, dict[str, object]] = {}
        self._active_rtp_ports: set[int] = set()
        self._next_rtp_port = self.config.rtp_port
        self._report_lock = threading.Lock()
        self._last_control_plane_report: dict[str, object] = {
            "configured": self.config.control_plane_reporter_configured(),
            "posted": False,
            "ok": None,
        }
        self._last_artifact_import_enqueue: dict[str, object] = {
            "configured": self.config.artifact_import_queue_configured(),
            "posted": False,
            "ok": None,
        }

    def _log(self, event: str, **fields: object) -> None:
        """Emit one structured JSON log line via the (injectable) log sink.

        Never raises: logging must not be able to crash the serve loop. Secrets
        (ARI credentials / api_key query) are never passed in by callers.
        """

        record: dict[str, object] = {
            "logger": "openphonex.realtime.worker",
            "event": event,
            "ts": round(time.time(), 3),
        }
        record.update(fields)
        with suppress(Exception):
            self._log_sink(json.dumps(record, sort_keys=True, default=str))

    def status(self) -> dict[str, object]:
        provider_missing = self.config.missing_provider_env()
        runtime_missing = self.config.missing_runtime_dependencies()
        unsupported_provider = (
            self.config.provider_kind() == "unsupported"
            or self.config.stt_kind() == "unsupported"
            or self.config.llm_kind() == "unsupported"
            or self.config.tts_kind() == "unsupported"
        )
        asterisk_missing = self.config.asterisk.redacted()["missing_live_env"]
        live_missing = self.config.missing_live_env()
        live_placeholders = self.config.unresolved_live_env_placeholders()
        operator_approval = self.operator_approval_status()
        cutover_readiness = self.cutover_readiness_status()
        live_runtime_blocked = self.config.live_enabled and (
            bool(asterisk_missing)
            or bool(live_placeholders)
            or (self.config.operator_approval_required and not operator_approval.get("ok"))
            or (self.config.cutover_readiness_required and not cutover_readiness.get("ok"))
        )
        capacity = self.capacity_status()
        return {
            "ok": not provider_missing and not runtime_missing and not unsupported_provider and not live_runtime_blocked,
            "service": "openphonex-realtime-media-worker",
            "version": service_version(),
            "environment": deployment_environment(),
            "role": "asterisk-external-media-realtime-worker",
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "config": self.config.redacted(),
            "pipeline": {
                "stages": [
                    "RTP input",
                    "turn detection",
                    "STT session",
                    "LLM",
                    "TTS",
                    "RTP output",
                    "RTP receive stats",
                    "barge-in monitor",
                    "recording",
                    "metrics",
                ],
                "stt_status": (
                    "StreamingSTTSession boundary ready; Deepgram provider adapter wired; "
                    f"STT speech_final wired; conservative semantic final-text endpointing "
                    f"{'enabled' if self.config.semantic_endpointing_enabled else 'disabled'}, live validation pending"
                ),
                "barge_in_status": "RTP output cancellation and next-turn caller-frame replay are wired in the realtime worker; live Asterisk validation pending",
                "phrase_queue_status": "LLM speakable phrase buffering is bounded; slow TTS/RTP playback records llm_phrase_queue_full when backpressure occurs",
                "pipecat_dependency": "none",
                "livekit_dependency": "none",
                "provider_mode": self.config.provider_mode,
                "provider_kind": self.config.provider_kind(),
                "stt_provider": self.config.stt_provider,
                "stt_kind": self.config.stt_kind(),
                "provider_components": self.provider_components(),
                "provider_adapter_cache": self._provider_adapter_cache_status(),
                "runtime_dependencies": self.config.runtime_dependency_status(),
                "missing_runtime_dependencies": runtime_missing,
                "ready_for_offline": not provider_missing and not runtime_missing and not unsupported_provider,
                "ready_for_live": (
                    not provider_missing
                    and not runtime_missing
                    and not unsupported_provider
                    and not live_missing
                    and not live_placeholders
                    and (not self.config.operator_approval_required or bool(operator_approval.get("ok")))
                    and (not self.config.cutover_readiness_required or bool(cutover_readiness.get("ok")))
                ),
                "missing_live_env": live_missing,
                "unresolved_live_env_placeholders": live_placeholders,
                "live_gate_enabled": self.config.live_enabled,
                "operator_approval": operator_approval,
                "cutover_readiness": cutover_readiness,
            },
            "capacity": capacity,
            "control_plane_reporter": self.control_plane_reporter_status(),
            "artifact_import_queue": self.artifact_import_queue_status(),
            "commands": {
                "status": _worker_command("--status"),
                "offline_synthetic_call": _worker_command("--offline-synthetic-call --offline-output-dir /tmp/openphonex-realtime-test"),
                "offline_recorded_call": _worker_command("--offline-recorded-audio /path/to/speech.wav --offline-output-dir /tmp/openphonex-realtime-recorded-test"),
                "offline_streaming_overlap_check": _worker_command("--offline-streaming-overlap-check --offline-output-dir /tmp/openphonex-realtime-overlap-test"),
                "offline_barge_in_check": _worker_command("--offline-barge-in-check --offline-output-dir /tmp/openphonex-realtime-barge-in-test"),
                "serve_health": _worker_command("--serve-health"),
                "check_live_startup": (
                    "REALTIME_LIVE_ENABLED=1 REALTIME_REQUIRE_OPERATOR_APPROVAL=1 "
                    "REALTIME_OPERATOR_APPROVAL_RECORD=/path/to/operator-approval.json "
                    "REALTIME_APPROVAL_PACKET_FINGERPRINT=<latest-packet-fingerprint> "
                    "REALTIME_REQUIRE_CUTOVER_READINESS=1 "
                    "REALTIME_CUTOVER_READINESS_REPORT=/path/to/validation-report.json "
                    "REALTIME_CUTOVER_READINESS_FINGERPRINT=<latest-packet-fingerprint> "
                    f"{_worker_command('--env-file .env --check-live-startup')}"
                ),
                "serve_asterisk": (
                    "REALTIME_LIVE_ENABLED=1 REALTIME_REQUIRE_OPERATOR_APPROVAL=1 "
                    "REALTIME_OPERATOR_APPROVAL_RECORD=/path/to/operator-approval.json "
                    "REALTIME_APPROVAL_PACKET_FINGERPRINT=<latest-packet-fingerprint> "
                    "REALTIME_REQUIRE_CUTOVER_READINESS=1 "
                    "REALTIME_CUTOVER_READINESS_REPORT=/path/to/validation-report.json "
                    "REALTIME_CUTOVER_READINESS_FINGERPRINT=<latest-packet-fingerprint> "
                    f"{_worker_command('--env-file .env --serve-asterisk')}"
                ),
                "serve_asterisk_with_health": (
                    "REALTIME_LIVE_ENABLED=1 REALTIME_REQUIRE_OPERATOR_APPROVAL=1 "
                    "REALTIME_OPERATOR_APPROVAL_RECORD=/path/to/operator-approval.json "
                    "REALTIME_APPROVAL_PACKET_FINGERPRINT=<latest-packet-fingerprint> "
                    "REALTIME_REQUIRE_CUTOVER_READINESS=1 "
                    "REALTIME_CUTOVER_READINESS_REPORT=/path/to/validation-report.json "
                    "REALTIME_CUTOVER_READINESS_FINGERPRINT=<latest-packet-fingerprint> "
                    f"{_worker_command('--env-file .env --serve-health --serve-asterisk')}"
                ),
                "report_health_once": _worker_command("--report-health-once"),
            },
        }

    def operator_approval_status(self) -> dict[str, object]:
        return _operator_approval_status(
            required=self.config.operator_approval_required,
            record_path=self.config.operator_approval_record,
            expected_fingerprint=self.config.operator_approval_packet_fingerprint,
            required_directions=set(self.config.operator_approval_required_directions),
        )

    def cutover_readiness_status(self) -> dict[str, object]:
        return _cutover_readiness_status(
            required=self.config.cutover_readiness_required,
            report_path=self.config.cutover_readiness_report,
            expected_fingerprint=self.config.cutover_readiness_fingerprint,
        )

    def live_startup_preflight(self) -> dict[str, object]:
        provider_error: dict[str, object] = {}
        live_error: dict[str, object] = {}
        provider_prewarm = self.prewarm_provider_clients(allow_provider_network=False)
        provider_ready = bool(provider_prewarm.get("ok"))
        if not provider_ready:
            provider_error = {
                "error": "ProviderPrewarmFailed",
                "message": str(provider_prewarm.get("message") or provider_prewarm.get("reason") or "provider_prewarm_failed"),
                "reason": str(provider_prewarm.get("reason") or "provider_prewarm_failed"),
            }
        try:
            self._require_live_asterisk()
            live_ready = True
        except Exception as exc:
            live_ready = False
            live_error = {"error": exc.__class__.__name__, "message": str(exc)}
        status = self.status()
        pipeline = status.get("pipeline") if isinstance(status.get("pipeline"), dict) else {}
        config = status.get("config") if isinstance(status.get("config"), dict) else {}
        ok = bool(provider_ready and live_ready)
        return {
            "name": "live_startup_preflight",
            "ok": ok,
            "ready_to_connect_ari": ok,
            "live_actions_performed": False,
            "ari_connection_performed": False,
            "provider_network_actions_performed": (
                provider_prewarm.get("provider_network_actions_performed")
                if isinstance(provider_prewarm.get("provider_network_actions_performed"), list)
                else []
            ),
            "provider_ready": provider_ready,
            "provider_prewarm": provider_prewarm,
            "live_ready": live_ready,
            "operator_approval": pipeline.get("operator_approval") if isinstance(pipeline.get("operator_approval"), dict) else {},
            "cutover_readiness": pipeline.get("cutover_readiness") if isinstance(pipeline.get("cutover_readiness"), dict) else {},
            "provider_error": provider_error,
            "live_error": live_error,
            "missing_provider_env": config.get("missing_provider_env") if isinstance(config.get("missing_provider_env"), list) else [],
            "missing_runtime_dependencies": config.get("missing_runtime_dependencies") if isinstance(config.get("missing_runtime_dependencies"), list) else [],
            "runtime_dependencies": config.get("runtime_dependencies") if isinstance(config.get("runtime_dependencies"), dict) else {},
            "missing_live_env": pipeline.get("missing_live_env") if isinstance(pipeline.get("missing_live_env"), list) else [],
            "unresolved_live_env_placeholders": pipeline.get("unresolved_live_env_placeholders")
            if isinstance(pipeline.get("unresolved_live_env_placeholders"), list)
            else [],
            "provider_mode": pipeline.get("provider_mode", ""),
            "provider_kind": pipeline.get("provider_kind", ""),
            "stt_kind": pipeline.get("stt_kind", ""),
            "effective_call_capacity": status.get("capacity", {}).get("effective_call_capacity") if isinstance(status.get("capacity"), dict) else None,
            "next_step": "start_live_worker" if ok else _live_startup_next_step(
                provider_ready=provider_ready,
                live_ready=live_ready,
                operator_approval=pipeline.get("operator_approval"),
                cutover_readiness=pipeline.get("cutover_readiness"),
                runtime_dependencies=pipeline.get("runtime_dependencies"),
                provider_prewarm=provider_prewarm,
                unresolved_live_env_placeholders=pipeline.get("unresolved_live_env_placeholders"),
            ),
        }

    def control_plane_reporter_status(self) -> dict[str, object]:
        with self._report_lock:
            last = dict(self._last_control_plane_report)
        return {
            "configured": self.config.control_plane_reporter_configured(),
            "api_url": self.config.control_plane_api_url,
            "gateway_deployment_id": self.config.gateway_deployment_id,
            "api_token_configured": bool(self.config.gateway_api_token),
            "interval_seconds": self.config.gateway_report_interval_seconds,
            "last_report": last,
            "secrets_redacted": True,
        }

    def artifact_import_queue_status(self) -> dict[str, object]:
        with self._report_lock:
            last = dict(self._last_artifact_import_enqueue)
        return {
            "configured": self.config.artifact_import_queue_configured(),
            "api_url": self.config.control_plane_api_url,
            "api_token_configured": bool(self.config.gateway_api_token),
            "job_kind": "call.artifact_import",
            "last_enqueue": last,
            "secrets_redacted": True,
        }

    def gateway_health_report_payload(self) -> dict[str, object]:
        status = self.status()
        payload: dict[str, object] = {
            "version": service_version(),
            "environment": deployment_environment(),
            "realtime_worker_health": {
                "ok": bool(status.get("ok")),
                "body": status,
            },
        }
        if self.config.gateway_public_url:
            payload["public_url"] = self.config.gateway_public_url
        if self.config.gateway_local_status_url:
            payload["local_status_url"] = self.config.gateway_local_status_url
        return payload

    def post_gateway_health_report(
        self,
        *,
        post_json: Callable[[str, dict[str, object], dict[str, str]], dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if not self.config.control_plane_reporter_configured():
            result = {
                "ok": False,
                "posted": False,
                "configured": False,
                "reason": "missing OPENPHONEX_API_URL, OPENPHONEX_GATEWAY_ID, or OPENPHONEX_GATEWAY_API_TOKEN",
            }
            self._remember_control_plane_report(result)
            return result
        url = (
            self.config.control_plane_api_url.rstrip("/")
            + f"/v1/gateway-deployments/{self.config.gateway_deployment_id}/health-report"
        )
        payload = self.gateway_health_report_payload()
        headers = {"Authorization": f"Bearer {self.config.gateway_api_token}", "Content-Type": "application/json"}
        try:
            response = post_json(url, payload, headers) if post_json else _post_json(url, payload, headers)
            result: dict[str, object] = {
                "ok": True,
                "posted": True,
                "configured": True,
                "url": url,
                "response_status": response.get("status", "ok") if isinstance(response, dict) else "ok",
            }
        except Exception as exc:
            result = {
                "ok": False,
                "posted": False,
                "configured": True,
                "url": url,
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
        self._remember_control_plane_report(result)
        return result

    def queue_call_artifact_import_job(
        self,
        call: dict[str, object],
        *,
        post_json: Callable[[str, dict[str, object], dict[str, str]], dict[str, object]] | None = None,
    ) -> dict[str, object]:
        if not self.config.artifact_import_queue_configured():
            result = {
                "ok": False,
                "posted": False,
                "configured": False,
                "reason": "missing OPENPHONEX_API_URL or OPENPHONEX_GATEWAY_API_TOKEN",
            }
            self._remember_artifact_import_enqueue(result)
            return result

        openphonex = call.get("openphonex") if isinstance(call.get("openphonex"), dict) else {}
        call_dir = str(call.get("call_dir") or "")
        organization_id = str(openphonex.get("organization_id") or "")
        project_id = str(openphonex.get("project_id") or "")
        call_id = str(openphonex.get("call_id") or call.get("openphonex_call_id") or "")
        missing = [
            name
            for name, value in {
                "call_dir": call_dir,
                "organization_id": organization_id,
                "project_id": project_id,
                "call_id": call_id,
            }.items()
            if not value
        ]
        if missing:
            result = {
                "ok": False,
                "posted": False,
                "configured": True,
                "reason": "missing_call_artifact_import_metadata",
                "missing": missing,
            }
            self._remember_artifact_import_enqueue(result)
            return result

        artifact_payload: dict[str, object] = {
            "organization_id": organization_id,
            "project_id": project_id,
            "call_id": call_id,
            "source_path": call_dir,
            "mode": "auto",
            "label": f"realtime call {call_id}",
            "direction": str(call.get("direction") or ""),
            "provider_call_id": str(call.get("asterisk_channel_id") or call.get("call_id") or call_id),
            "agent_id": str(openphonex.get("agent_id") or ""),
            "agent_endpoint_id": str(openphonex.get("agent_endpoint_id") or ""),
        }
        payload = {
            "organization_id": organization_id,
            "kind": "call.artifact_import",
            "payload": artifact_payload,
        }
        url = self.config.control_plane_api_url.rstrip("/") + "/v1/worker-jobs"
        headers = {"Authorization": f"Bearer {self.config.gateway_api_token}", "Content-Type": "application/json"}
        try:
            response = post_json(url, payload, headers) if post_json else _post_json(url, payload, headers)
            result = {
                "ok": True,
                "posted": True,
                "configured": True,
                "url": url,
                "job_id": response.get("id", "") if isinstance(response, dict) else "",
                "job_kind": "call.artifact_import",
                "call_id": call_id,
            }
        except Exception as exc:
            result = {
                "ok": False,
                "posted": False,
                "configured": True,
                "url": url,
                "error": exc.__class__.__name__,
                "message": str(exc),
                "call_id": call_id,
            }
        self._remember_artifact_import_enqueue(result)
        return result

    def start_control_plane_reporter_thread(self) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()

        def run() -> None:
            while not stop.is_set():
                self.post_gateway_health_report()
                stop.wait(self.config.gateway_report_interval_seconds)

        thread = threading.Thread(target=run, name="openphonex-realtime-control-plane-reporter", daemon=True)
        thread.start()
        return stop, thread

    def _remember_control_plane_report(self, result: dict[str, object]) -> None:
        safe = {
            key: value
            for key, value in result.items()
            if key not in {"response"}
        }
        safe["reported_at_unix"] = round(time.time(), 3)
        with self._report_lock:
            self._last_control_plane_report = safe

    def _remember_artifact_import_enqueue(self, result: dict[str, object]) -> None:
        safe = {
            key: value
            for key, value in result.items()
            if key not in {"response"}
        }
        safe["reported_at_unix"] = round(time.time(), 3)
        with self._report_lock:
            self._last_artifact_import_enqueue = safe

    def provider_components(self, *, force_fake: bool = False) -> dict[str, dict[str, object]]:
        if force_fake:
            return {
                "stt": {"provider": "fake", "adapter": "FakeStreamingSTT", "real": False, "language": self.config.fake_language},
                "llm": {"provider": "template", "adapter": "TemplateStreamingLLM", "real": False},
                "tts": {"provider": "tone", "adapter": "ToneStreamingTTS", "real": False},
            }
        registry = _provider_registry()
        return {
            "stt": registry.describe("stt", self.config.active_stt_name(), self.config),
            "llm": registry.describe("llm", self.config.active_llm_name(), self.config),
            "tts": registry.describe("tts", self.config.active_tts_name(), self.config),
        }

    def prewarm_provider_clients(self, *, allow_provider_network: bool = False) -> dict[str, object]:
        """Validate and cache provider setup before the caller-visible path.

        This intentionally does not send audio, prompts, TTS text, SIP traffic,
        or live calls. The only provider-network action it can take is an
        optional ElevenLabs voice-name lookup, and that is blocked unless the
        caller explicitly allows provider network access.
        """

        check: dict[str, object] = {
            "name": "provider_prewarm",
            "ok": True,
            "required": True,
            "live_actions_performed": False,
            "provider_network_actions_allowed": allow_provider_network,
            "provider_network_actions_performed": [],
            "provider_network_actions_required": [],
            "prewarmed_components": [],
            "runtime_dependencies": self.config.runtime_dependency_status(),
            "cached": {
                "elevenlabs_voice_id": bool(self._resolved_elevenlabs_voice_id),
                "provider_adapters": [],
            },
        }
        missing_runtime_dependencies = self.config.missing_runtime_dependencies()
        if missing_runtime_dependencies:
            check.update(
                {
                    "ok": False,
                    "reason": "runtime_dependencies_missing",
                    "missing_runtime_dependencies": missing_runtime_dependencies,
                }
            )
            return check
        try:
            self._require_provider_ready()
        except Exception as exc:
            check.update(
                {
                    "ok": False,
                    "reason": "provider_not_ready",
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            return check

        prewarmed_components: list[dict[str, object]] = []
        if self.config.uses_deepgram_stt():
            prewarmed_components.append(
                {
                    "component": "stt",
                    "provider": "deepgram",
                    "adapter": "DeepgramStreamingSTT",
                    "adapter_configured": True,
                    "network_action": "none",
                }
            )
        else:
            prewarmed_components.append(
                {
                    "component": "stt",
                    "provider": "fake",
                    "adapter": "FakeStreamingSTT",
                    "adapter_configured": True,
                    "network_action": "none",
                }
            )
        if self.config.uses_gemini_llm():
            prewarmed_components.append(
                {
                    "component": "llm",
                    "provider": "gemini",
                    "adapter": "GeminiStreamingLLM",
                    "adapter_configured": True,
                    "network_action": "none",
                }
            )
        else:
            prewarmed_components.append(
                {
                    "component": "llm",
                    "provider": "template",
                    "adapter": "TemplateStreamingLLM",
                    "adapter_configured": True,
                    "network_action": "none",
                }
            )

        if self.config.uses_elevenlabs_tts():
            needs_voice_lookup = not self.config.elevenlabs_voice_id and bool(self.config.elevenlabs_voice)
            if needs_voice_lookup and not allow_provider_network:
                check.update(
                    {
                        "ok": False,
                        "reason": "provider_prewarm_network_approval_required",
                        "provider_network_actions_required": ["elevenlabs_voice_lookup"],
                        "prewarmed_components": prewarmed_components,
                    }
                )
                return check
            try:
                voice_id = self._resolve_elevenlabs_voice_id()
            except Exception as exc:
                check.update(
                    {
                        "ok": False,
                        "reason": "provider_prewarm_failed",
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "provider_network_actions_required": ["elevenlabs_voice_lookup"] if needs_voice_lookup else [],
                        "provider_network_actions_performed": ["elevenlabs_voice_lookup"] if needs_voice_lookup and allow_provider_network else [],
                        "prewarmed_components": prewarmed_components,
                    }
                )
                return check
            prewarmed_components.append(
                {
                    "component": "tts",
                    "provider": "elevenlabs",
                    "adapter": "ElevenLabsV3StreamingTTS",
                    "adapter_configured": True,
                    "voice_id_cached": bool(voice_id),
                    "network_action": "elevenlabs_voice_lookup" if needs_voice_lookup else "none",
                }
            )
            if needs_voice_lookup and allow_provider_network:
                check["provider_network_actions_performed"] = ["elevenlabs_voice_lookup"]
        else:
            prewarmed_components.append(
                {
                    "component": "tts",
                    "provider": "tone",
                    "adapter": "ToneStreamingTTS",
                    "adapter_configured": True,
                    "network_action": "none",
                }
            )

        try:
            self._provider_adapters()
        except Exception as exc:
            check.update(
                {
                    "ok": False,
                    "reason": "provider_prewarm_failed",
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                    "prewarmed_components": prewarmed_components,
                }
            )
            return check

        adapter_cache = self._provider_adapter_cache_status()
        cached_components = set(adapter_cache.get("components", []))
        for item in prewarmed_components:
            component = str(item.get("component", ""))
            item["adapter_cached"] = component in cached_components
        check["cached"] = {
            "elevenlabs_voice_id": bool(self._resolved_elevenlabs_voice_id),
            "provider_adapters": adapter_cache.get("components", []),
        }
        check["prewarmed_components"] = prewarmed_components
        return check

    def capacity_status(self) -> dict[str, object]:
        with self._capacity_lock:
            active_calls = sorted(self._active_call_ids)
            active_call_details = [dict(self._active_call_details.get(call_id, {"asterisk_channel_id": call_id})) for call_id in active_calls]
            active_ports = sorted(self._active_rtp_ports)
            capacity = self.config.effective_call_capacity()
            available_ports = max(0, len(self.config.rtp_ports()) - len(active_ports))
            available_calls = max(0, capacity - len(active_calls))
            return {
                "active_calls": len(active_calls),
                "active_call_ids": active_calls,
                "active_call_details": active_call_details,
                "active_rtp_ports": active_ports,
                "rtp_port_range": {"start": self.config.rtp_port, "end": self.config.resolved_rtp_port_end()},
                "rtp_port_capacity": len(self.config.rtp_ports()),
                "configured_call_limit": self.config.max_concurrent_calls,
                "effective_call_capacity": capacity,
                "available_ports": available_ports,
                "available_call_slots": min(available_calls, available_ports),
            }

    def build_health_server(self) -> ThreadingHTTPServer:
        return ThreadingHTTPServer((self.config.health_bind, self.config.health_port), make_health_handler(self))

    def start_health_server_thread(self) -> tuple[ThreadingHTTPServer, threading.Thread]:
        server = self.build_health_server()
        thread = threading.Thread(target=server.serve_forever, name="openphonex-realtime-health", daemon=True)
        thread.start()
        return server, thread

    def serve_health(self) -> None:
        server = self.build_health_server()
        host, port = server.server_address[:2]
        print(f"OpenPhonex realtime media worker listening on http://{host}:{port}")
        server.serve_forever()

    def serve_asterisk_with_health(
        self,
        *,
        max_attempts: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, object]:
        """Start health (daemon thread) FIRST, then connect ARI with retry + logs.

        GA-6 / SCH-217. The container previously bound the health server and then
        called ``serve_asterisk`` with ZERO logging and ZERO retry, so:
          * a successful ARI connect, a hung connect, and a crash-looping connect
            were indistinguishable from ``docker logs`` (all "empty"), and
          * any transient ARI-not-ready at container start propagated out of
            ``main`` and tore the worker down invisibly instead of reconnecting.

        This method now: (1) binds health in a daemon thread and logs that it is
        listening, then (2) opens the ARI WebSocket via ``_serve_asterisk_with_retry``
        which logs every connect attempt, every error, the confirmed WS
        connect+subscribe, and retries with bounded backoff. Health stays up the
        whole time so ``/health`` and ``--status`` remain answerable even while ARI
        is still coming up.
        """

        server, thread = self.start_health_server_thread()
        host, port = server.server_address[:2]
        self._log("health_server_listening", url=f"http://{host}:{port}")
        try:
            return self._serve_asterisk_with_retry(max_attempts=max_attempts, sleep=sleep)
        finally:
            self._log("health_server_stopping", url=f"http://{host}:{port}")
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def _serve_asterisk_with_retry(
        self,
        *,
        max_attempts: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, object]:
        """Run ``serve_asterisk`` with loud logging and bounded-backoff retry.

        ``max_attempts=None`` uses the configured default (``ari_connect_max_attempts``;
        0 => retry forever). A live ARI WebSocket that later drops surfaces as an
        exception from ``serve_asterisk`` and is therefore reconnected here too.
        """

        if max_attempts is None:
            configured = self.config.ari_connect_max_attempts
            max_attempts = configured if configured and configured > 0 else None
        backoff = self.config.ari_connect_backoff_seconds
        backoff_max = self.config.ari_connect_backoff_max_seconds
        ari_url = self.config.asterisk.ari_url
        app = self.config.asterisk.app
        attempt = 0
        while True:
            attempt += 1
            self._log("ari_connect_attempt", attempt=attempt, ari_url=ari_url, app=app, max_attempts=max_attempts)
            try:
                result = self.serve_asterisk()
            except Exception as exc:  # noqa: BLE001 - resilience: log loudly, then retry/give up
                will_retry = max_attempts is None or attempt < max_attempts
                self._log(
                    "ari_connect_error",
                    attempt=attempt,
                    ari_url=ari_url,
                    app=app,
                    error=exc.__class__.__name__,
                    message=str(exc),
                    will_retry=will_retry,
                )
                if not will_retry:
                    self._log("ari_connect_giving_up", attempts=attempt, ari_url=ari_url, app=app)
                    raise
                delay = backoff * attempt
                if backoff_max > 0:
                    delay = min(delay, backoff_max)
                if delay > 0:
                    self._log("ari_connect_retry_wait", attempt=attempt, delay_seconds=round(delay, 3))
                    sleep(delay)
                continue
            self._log(
                "ari_event_loop_returned",
                attempt=attempt,
                ari_url=ari_url,
                app=app,
                ok=bool(result.get("ok")),
                events=result.get("events"),
            )
            return result

    def run_offline_synthetic_call(self, output_dir: Path | None = None) -> dict[str, object]:
        self._require_provider_ready()
        root = output_dir or self.config.calls_dir / f"realtime-offline-{int(time.time())}"
        root.mkdir(parents=True, exist_ok=True)
        packets: list[bytes] = []
        metrics = CallMetrics()
        recorder = StreamingCallRecorder(root)
        try:
            pipeline = self._build_pipeline(packets.append, recorder, metrics=metrics)
            turn = pipeline.consume_inbound_until_turn(_synthetic_caller_frames())
            if not turn:
                raise RuntimeError("synthetic caller frames did not produce a turn")
            result = pipeline.process_turn(turn)
            metrics.write_json(root / "latency.json")
            turns = [_turn_payload(1, result)]
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "user": result.transcript,
                        "assistant": result.assistant_text,
                        "assistant_spoken": result.assistant_spoken_text,
                        "latency": result.latency,
                        "provider_components": self.provider_components(),
                        "summary": _realtime_call_summary(turns),
                        "turns": turns,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return {
                "ok": True,
                "call_dir": str(root),
                "rtp_packets": len(packets),
                "transcript": result.transcript,
                "assistant": result.assistant_text,
                "assistant_spoken": result.assistant_spoken_text,
                "latency": result.latency,
                "provider_components": self.provider_components(),
                "artifacts": _realtime_artifacts_payload(recorder, root),
            }
        finally:
            recorder.close()

    def run_offline_streaming_overlap_check(self, output_dir: Path | None = None) -> dict[str, object]:
        root = output_dir or self.config.calls_dir / f"realtime-overlap-{int(time.time())}"
        root.mkdir(parents=True, exist_ok=True)
        packets: list[bytes] = []
        metrics = CallMetrics()
        recorder = StreamingCallRecorder(root)
        overlap_reply = " ".join(f"Fraas {idx}." for idx in range(1, 7))
        fake_config = FakeRealtimeProviderConfig(
            transcript=self.config.fake_transcript,
            reply=overlap_reply,
            language=self.config.fake_language,
        )
        try:
            pipeline = RealtimeMediaPipeline(
                stt=FakeStreamingSTT(fake_config),
                llm=TemplateStreamingLLM(fake_config),
                tts=ToneStreamingTTS(fake_config, frame_count=80),
                rtp_sender=PacedRtpSender(packets.append, clock_sleep=lambda seconds: time.sleep(min(seconds, 0.001))),
                recorder=recorder,
                detector=RmsTurnDetector(
                    TurnDetectionConfig(
                        silence_hold_ms=self.config.silence_hold_ms,
                        min_speech_ms=self.config.min_speech_ms,
                    )
                ),
                metrics=metrics,
                semantic_endpointing=SemanticEndpointDetector(self.config.semantic_endpoint_config()),
                max_pending_phrases=8,
            )
            turn = pipeline.consume_inbound_until_turn(_synthetic_caller_frames())
            if not turn:
                raise RuntimeError("synthetic caller frames did not produce a turn")
            result = pipeline.process_turn(turn)
            events = list(metrics.events)
            event_names = [event.name for event in events]
            first_phrase_done = next(
                (event for event in events if event.name == "tts_phrase_rtp_done" and event.attrs.get("phrase_index") == 1),
                None,
            )
            second_phrase_ready = next(
                (event for event in events if event.name == "speakable_phrase_ready" and event.attrs.get("phrase_index") == 2),
                None,
            )
            llm_completed = metrics.first("llm_stream_completed")
            overlap_observed = bool(first_phrase_done and llm_completed and llm_completed.at < first_phrase_done.at)
            second_phrase_ready_before_first_done = bool(first_phrase_done and second_phrase_ready and second_phrase_ready.at < first_phrase_done.at)
            metrics.mark(
                "offline_streaming_overlap_check_completed",
                overlap_observed=overlap_observed,
                second_phrase_ready_before_first_done=second_phrase_ready_before_first_done,
                rtp_packets=len(packets),
            )
            metrics.write_json(root / "latency.json")
            turns = [_turn_payload(1, result)]
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "summary": _realtime_call_summary(turns),
                        "turns": turns,
                        "streaming_overlap_check": True,
                        "provider_components": self.provider_components(force_fake=True),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return {
                "ok": True,
                "call_dir": str(root),
                "rtp_packets": len(packets),
                "phrase_count": len([event for event in event_names if event == "speakable_phrase_ready"]),
                "tts_phrase_count": len([event for event in event_names if event == "tts_phrase_rtp_done"]),
                "llm_stream_completed_before_first_phrase_done": overlap_observed,
                "second_phrase_ready_before_first_phrase_done": second_phrase_ready_before_first_done,
                "llm_stream_completed": "llm_stream_completed" in event_names,
                "first_phrase_done": bool(first_phrase_done),
                "turns": turns,
                "latency": result.latency,
                "provider_components": self.provider_components(force_fake=True),
                "artifacts": _realtime_artifacts_payload(recorder, root),
            }
        finally:
            recorder.close()

    def run_offline_barge_in_check(self, output_dir: Path | None = None) -> dict[str, object]:
        root = output_dir or self.config.calls_dir / f"realtime-barge-in-{int(time.time())}"
        root.mkdir(parents=True, exist_ok=True)
        packets: list[bytes] = []
        metrics = CallMetrics()
        recorder = StreamingCallRecorder(root)
        barge_in_backlog_reply = " ".join(f"Fraas {idx}." for idx in range(1, 9))
        fake_config = FakeRealtimeProviderConfig(
            transcript=self.config.fake_transcript,
            reply=barge_in_backlog_reply,
            language=self.config.fake_language,
        )
        barge_frames = [AudioFrame.tone(20, direction="caller")]

        def read_barge_frame() -> AudioFrame | None:
            if len(packets) >= 2 and barge_frames:
                return barge_frames.pop(0)
            return None

        try:
            pipeline = RealtimeMediaPipeline(
                stt=FakeStreamingSTT(fake_config),
                llm=TemplateStreamingLLM(fake_config),
                tts=ToneStreamingTTS(fake_config, frame_count=80),
                rtp_sender=PacedRtpSender(packets.append, clock_sleep=lambda seconds: time.sleep(min(seconds, 0.002))),
                recorder=recorder,
                detector=RmsTurnDetector(
                    TurnDetectionConfig(
                        silence_hold_ms=self.config.silence_hold_ms,
                        min_speech_ms=self.config.min_speech_ms,
                    )
                ),
                metrics=metrics,
                barge_in_frame_reader=read_barge_frame,
                barge_in_config=BargeInConfig(poll_interval_seconds=0.001, speech_rms=self.config.barge_in_speech_rms),
                semantic_endpointing=SemanticEndpointDetector(self.config.semantic_endpoint_config()),
                max_pending_phrases=1,
            )
            first_turn = pipeline.consume_inbound_until_turn(_synthetic_caller_frames())
            if not first_turn:
                raise RuntimeError("synthetic caller frames did not produce a first turn")
            first_result = pipeline.process_turn(first_turn)
            pipeline.llm = TemplateStreamingLLM(
                FakeRealtimeProviderConfig(
                    transcript=self.config.fake_transcript,
                    reply=self.config.fake_reply,
                    language=self.config.fake_language,
                )
            )
            continuation_frames = [AudioFrame.tone(20, direction="caller") for _ in range(15)]
            continuation_frames += [AudioFrame.silence(20) for _ in range(20)]
            second_turn = pipeline.consume_inbound_until_turn(continuation_frames)
            if not second_turn:
                raise RuntimeError("barge-in continuation frames did not produce a second turn")
            second_result = pipeline.process_turn(second_turn)

            turns = [
                _turn_payload(1, first_result),
                _turn_payload(2, second_result),
            ]
            event_names = [event.name for event in metrics.events]
            metrics.mark(
                "offline_barge_in_check_completed",
                interrupted=first_result.interrupted,
                replayed="barge_in_pending_frames_replayed" in event_names,
                rtp_packets=len(packets),
            )
            metrics.write_json(root / "latency.json")
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "summary": _realtime_call_summary(turns),
                        "turns": turns,
                        "barge_in_check": True,
                        "provider_components": self.provider_components(force_fake=True),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return {
                "ok": True,
                "call_dir": str(root),
                "rtp_packets": len(packets),
                "interrupted_turns": sum(1 for turn in turns if turn["interrupted"]),
                "barge_in_detected": "barge_in_speech_detected" in event_names,
                "output_cancelled": "barge_in_output_cancelled" in event_names,
                "pending_frames_replayed": "barge_in_pending_frames_replayed" in event_names,
                "llm_stream_closed": "llm_chunk_stream_closed" in event_names,
                "llm_phrase_backpressure": "llm_phrase_queue_full" in event_names,
                "llm_phrase_producer_cancelled": "llm_phrase_producer_cancelled" in event_names,
                "llm_phrase_producer_still_running": "llm_phrase_producer_still_running" in event_names,
                "turns": turns,
                "latency": second_result.latency,
                "provider_components": self.provider_components(force_fake=True),
                "artifacts": _realtime_artifacts_payload(recorder, root),
            }
        finally:
            recorder.close()

    def run_offline_recorded_call(self, input_audio: Path, output_dir: Path | None = None) -> dict[str, object]:
        self._require_provider_ready()
        if not input_audio.exists():
            raise FileNotFoundError(f"Recorded audio not found: {input_audio}")
        root = output_dir or self.config.calls_dir / f"realtime-recorded-{int(time.time())}"
        root.mkdir(parents=True, exist_ok=True)
        packets: list[bytes] = []
        metrics = CallMetrics()
        recorder = StreamingCallRecorder(root)
        try:
            pipeline = self._build_pipeline(packets.append, recorder, metrics=metrics)
            turn = pipeline.consume_inbound_until_turn(_audio_file_frames(input_audio))
            if not turn:
                raise RuntimeError("recorded audio did not produce a caller turn")
            result = pipeline.process_turn(turn)
            if not result.transcript.strip():
                raise RuntimeError("recorded audio did not produce a transcript")
            metrics.write_json(root / "latency.json")
            turns = [{**_turn_payload(1, result), "source_audio": str(input_audio)}]
            (root / "transcript.json").write_text(
                json.dumps(
                    {
                        "user": result.transcript,
                        "assistant": result.assistant_text,
                        "assistant_spoken": result.assistant_spoken_text,
                        "latency": result.latency,
                        "provider_components": self.provider_components(),
                        "summary": _realtime_call_summary(turns),
                        "turns": turns,
                        "source_audio": str(input_audio),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return {
                "ok": True,
                "call_dir": str(root),
                "input_audio": str(input_audio),
                "rtp_packets": len(packets),
                "transcript": result.transcript,
                "assistant": result.assistant_text,
                "assistant_spoken": result.assistant_spoken_text,
                "latency": result.latency,
                "provider_components": self.provider_components(),
                "artifacts": _realtime_artifacts_payload(recorder, root),
            }
        finally:
            recorder.close()

    def build_asterisk_session(self, transport: UdpRtpTransport | None = None) -> AsteriskExternalMediaSession:
        return AsteriskExternalMediaSession(ari=AsteriskRestClient(self.config.asterisk), config=self.config.asterisk, transport=transport)

    def serve_asterisk(self) -> dict[str, object]:
        """Connect to ARI events and run one media pipeline per ExternalMedia call.

        This is live infrastructure plumbing, but provider mode remains fake until
        real streaming STT/LLM/TTS adapters are explicitly wired.
        """

        self._require_live_asterisk()
        stop = threading.Event()
        event_client = AriWebSocketEventClient(self.config.asterisk)
        ari_url = self.config.asterisk.ari_url
        app = self.config.asterisk.app
        self._log("ari_ws_opening", ari_url=ari_url, app=app)

        def _on_open() -> None:
            # Fires only after the ARI WebSocket HTTP 101 handshake succeeds, i.e.
            # the worker is now subscribed to the Stasis app and receiving events.
            self._log("ari_ws_connected", ari_url=ari_url, app=app)

        return self.run_asterisk_event_loop(event_client.iter_events(stop, on_open=_on_open))

    def run_asterisk_event_loop(
        self,
        events: Iterable[dict[str, object]],
        *,
        ari: AriClientProtocol | None = None,
        transport: UdpRtpTransport | None = None,
        transport_factory: Callable[[int], UdpRtpTransport] | None = None,
        close_transport: bool = True,
    ) -> dict[str, object]:
        self._require_live_asterisk()
        self._require_provider_prewarmed_for_live()
        ari_client = ari or AsteriskRestClient(self.config.asterisk)
        transport_factory = transport_factory or (lambda port: UdpRtpTransport((self.config.rtp_bind, port)))
        sessions_by_primary: dict[str, AsteriskExternalMediaSession] = {}
        sessions_by_external: dict[str, AsteriskExternalMediaSession] = {}
        call_threads: list[threading.Thread] = []
        call_results: list[dict[str, object]] = []
        call_errors: list[dict[str, object]] = []
        lock = threading.Lock()

        def start_media_loop(session: AsteriskExternalMediaSession, rtp_transport: UdpRtpTransport, *, owned_transport: bool, allocated_port: int | None) -> None:
            call_id = session.state.primary_channel_id or f"unknown-{int(time.time())}"
            call_dir = self.config.calls_dir / f"realtime-{_safe_path_part(call_id)}-{int(time.time())}"
            self._register_active_call(call_id, metadata=session.state.openphonex, rtp_port=allocated_port or self.config.rtp_port)

            def run() -> None:
                try:
                    result = self._run_live_media_loop(session, rtp_transport, call_dir)
                    with lock:
                        call_results.append(result)
                except Exception as exc:
                    with lock:
                        call_errors.append({"call_id": call_id, "error": exc.__class__.__name__, "message": str(exc)})
                finally:
                    self._unregister_active_call(call_id)
                    if owned_transport or close_transport:
                        try:
                            rtp_transport.close()
                        except Exception:
                            pass
                    if allocated_port is not None:
                        self._release_rtp_port(allocated_port)

            thread = threading.Thread(target=run, name=f"openphonex-realtime-{call_id}", daemon=True)
            thread.start()
            call_threads.append(thread)

        def create_primary_session(event: dict[str, object]) -> str:
            channel_id = _event_channel_id(event)
            if not channel_id:
                return "ignored"
            if not self._has_call_capacity():
                _reject_over_capacity(ari_client, channel_id)
                return "capacity_rejected"

            allocated_port: int | None = None
            if transport is not None and call_threads:
                _reject_over_capacity(ari_client, channel_id)
                return "capacity_rejected"
            if transport is None:
                allocated_port = self._acquire_rtp_port()
                if allocated_port is None:
                    _reject_over_capacity(ari_client, channel_id)
                    return "capacity_rejected"
                rtp_transport = transport_factory(allocated_port)
                owned_transport = True
            else:
                allocated_port = self.config.rtp_port
                rtp_transport = transport
                owned_transport = False

            call_config = replace(self.config.asterisk, rtp_port=allocated_port)
            session = AsteriskExternalMediaSession(ari=ari_client, config=call_config, transport=rtp_transport)
            action = session.handle_event(event)
            primary_id = session.state.primary_channel_id
            if primary_id:
                sessions_by_primary[primary_id] = session
            if session.state.external_channel_id:
                sessions_by_external[session.state.external_channel_id] = session
            if action == "primary_started":
                start_media_loop(session, rtp_transport, owned_transport=owned_transport, allocated_port=allocated_port if transport is None else None)
            return action

        event_count = 0
        actions: list[str] = []
        try:
            for event in events:
                event_count += 1
                action = "ignored"
                event_type = str(event.get("type") or "")
                channel_id = _event_channel_id(event)
                channel_name = _event_channel_name(event)
                if event_type == "StasisStart" and channel_name.startswith("UnicastRTP"):
                    session = sessions_by_external.get(channel_id or "")
                    action = session.handle_event(event) if session else "external_unmatched"
                elif event_type == "StasisStart":
                    action = create_primary_session(event)
                elif event_type == "ChannelStateChange":
                    session = sessions_by_primary.get(channel_id or "")
                    action = session.handle_event(event) if session else "ignored"
                elif event_type == "StasisEnd":
                    session = sessions_by_primary.get(channel_id or "") or sessions_by_external.get(channel_id or "")
                    action = session.handle_event(event) if session else "ignored"
                actions.append(action)
                if action == "primary_started":
                    session = sessions_by_primary.get(channel_id or "")
                    if session and session.state.external_channel_id:
                        sessions_by_external[session.state.external_channel_id] = session
        finally:
            for session in sessions_by_primary.values():
                session.state.primary_ended = True
                session.cleanup()
            for thread in call_threads:
                thread.join(timeout=max(1.0, self.config.idle_timeout_seconds + 1.0))
        return {
            "ok": not call_errors,
            "events": event_count,
            "actions": actions,
            "calls": call_results,
            "errors": call_errors,
            "capacity": self.capacity_status(),
        }

    def _run_live_media_loop(
        self,
        session: AsteriskExternalMediaSession,
        transport: UdpRtpTransport,
        call_dir: Path,
    ) -> dict[str, object]:
        metrics = CallMetrics()
        call_direction = session.state.call_direction or "inbound"
        call_direction_source = session.state.call_direction_source or "default_inbound"
        openphonex_metadata = dict(session.state.openphonex)
        control_plane_call_id = openphonex_metadata.get("call_id", "")
        asterisk_channel_id = session.state.primary_channel_id or ""
        metrics.mark(
            "call_direction",
            direction=call_direction,
            source=call_direction_source,
            primary_channel_id=asterisk_channel_id,
        )
        if openphonex_metadata:
            metrics.mark("openphonex_call_metadata", primary_channel_id=asterisk_channel_id, **openphonex_metadata)
        metrics.mark("call_media_started", direction=call_direction)
        recorder = StreamingCallRecorder(call_dir)
        turns: list[dict[str, object]] = []
        opening_utterance: dict[str, object] | None = None
        call_error: dict[str, object] | None = None
        rtp_receive_stats: dict[str, object] = {}
        latency: dict[str, float | None] = {}
        # Per-agent voice stack (SCH-205/206): resolve the agent's chosen
        # providers/models/voice from the control plane once per call and build
        # a per-call config via dataclasses.replace. Missing agent_id or a failed
        # resolution degrades to the env-driven self.config — never a broken call.
        per_call_config = self._resolve_agent_call_config(openphonex_metadata.get("agent_id", ""))
        if per_call_config is not self.config:
            metrics.mark(
                "agent_voice_stack_resolved",
                agent_id=openphonex_metadata.get("agent_id", ""),
                stt=per_call_config.active_stt_name(),
                llm=per_call_config.active_llm_name(),
                tts=per_call_config.active_tts_name(),
                language=per_call_config.fake_language,
            )
        try:
            try:
                pipeline = self._build_pipeline(transport.send_packet, recorder, metrics=metrics, barge_in_frame_reader=transport.recv_frame, config=per_call_config)
                if call_direction == "outbound":
                    opening_utterance = self._speak_outbound_opening(session, pipeline, metrics, config=per_call_config)
                else:
                    opening_utterance = self._speak_inbound_opening(session, pipeline, metrics)
                turn_count = 0
                while not session.state.primary_ended and turn_count < self.config.max_turns:
                    turn = pipeline.consume_inbound_until_turn(self._live_frames_until_turn(session, transport))
                    if not turn:
                        break
                    result = pipeline.process_turn(turn)
                    turn_count += 1
                    turns.append(
                        {
                            "index": turn_count,
                            "user": result.transcript,
                            "assistant": result.assistant_text,
                            "assistant_spoken": result.assistant_spoken_text,
                            "interrupted": result.interrupted,
                            "llm_fallback_leg": result.llm_fallback_leg,
                            "latency": result.latency,
                            "caller_audio": f"turn-{turn_count:03d}-caller.wav",
                            "agent_audio": f"turn-{turn_count:03d}-agent.wav",
                        }
                    )
            except Exception as exc:
                call_error = {"error": exc.__class__.__name__, "message": str(exc)}
                metrics.mark("call_media_error", error=exc.__class__.__name__)
                raise
            finally:
                # Artifacts are written even when a provider error aborts the
                # call, so failed live calls stay debuggable from disk.
                metrics.mark("call_media_stopped", turns=len(turns), primary_ended=session.state.primary_ended, aborted=call_error is not None)
                rtp_receive_stats = _transport_stats_payload(transport)
                metrics.mark("rtp_receive_stats", **rtp_receive_stats)
                latency = metrics.latency_breakdown()
                transcript_payload: dict[str, object] = {
                    "summary": _realtime_call_summary(turns),
                    "call_id": control_plane_call_id or asterisk_channel_id,
                    "asterisk_channel_id": asterisk_channel_id,
                    "openphonex": openphonex_metadata,
                    "call_direction": call_direction,
                    "call_direction_source": call_direction_source,
                    "turns": turns,
                    "latency": latency,
                    "rtp_receive_stats": rtp_receive_stats,
                    "provider_components": self.provider_components(),
                }
                if opening_utterance is not None:
                    transcript_payload["opening_utterance"] = opening_utterance
                if call_error is not None:
                    transcript_payload["aborted"] = True
                    transcript_payload["call_error"] = call_error
                    with suppress(Exception):
                        metrics.write_json(call_dir / "latency.json")
                    with suppress(Exception):
                        (call_dir / "transcript.json").write_text(json.dumps(transcript_payload, indent=2, sort_keys=True) + "\n")
                else:
                    metrics.write_json(call_dir / "latency.json")
                    (call_dir / "transcript.json").write_text(json.dumps(transcript_payload, indent=2, sort_keys=True) + "\n")
        finally:
            recorder.close()
        call_result = {
            "ok": True,
            "call_id": session.state.primary_channel_id,
            "openphonex_call_id": control_plane_call_id,
            "asterisk_channel_id": asterisk_channel_id,
            "openphonex": openphonex_metadata,
            "direction": call_direction,
            "direction_source": call_direction_source,
            "call_dir": str(call_dir),
            "turns": len(turns),
            "latency": latency,
            "rtp_receive_stats": rtp_receive_stats,
            "provider_components": self.provider_components(),
            "artifacts": _realtime_artifacts_payload(recorder, call_dir),
        }
        if opening_utterance is not None:
            call_result["opening_utterance"] = opening_utterance
        call_result["artifact_import_job"] = self.queue_call_artifact_import_job(call_result)
        return call_result

    def _speak_outbound_opening(
        self,
        session: AsteriskExternalMediaSession,
        pipeline: RealtimeMediaPipeline,
        metrics: CallMetrics,
        config: "RealtimeWorkerConfig | None" = None,
    ) -> dict[str, object]:
        """Outbound initiative: wait for the callee channel, then greet first."""

        config = config or self.config
        text = " ".join(config.opening_utterance.split())
        channel_up = self._wait_for_outbound_channel_up(session)
        metrics.mark(
            "outbound_channel_up_wait_finished",
            channel_up=channel_up,
            primary_ended=session.state.primary_ended,
            channel_state=session.state.primary_channel_state,
            timeout_seconds=self.config.outbound_answer_timeout_seconds,
        )
        if not text or not channel_up or session.state.primary_ended:
            metrics.mark(
                "outbound_opening_utterance_skipped",
                configured=bool(text),
                channel_up=channel_up,
                primary_ended=session.state.primary_ended,
            )
            return {"text": text, "spoken": False, "interrupted": False, "skipped": True}
        spoken = pipeline.speak_text(text, label="outbound_opening")
        metrics.mark(
            "outbound_opening_utterance_finished",
            spoken=bool(spoken.get("spoken")),
            interrupted=bool(spoken.get("interrupted")),
        )
        return {**spoken, "skipped": False}

    def _speak_inbound_opening(
        self,
        session: AsteriskExternalMediaSession,
        pipeline: RealtimeMediaPipeline,
        metrics: CallMetrics,
    ) -> dict[str, object] | None:
        """Inbound initiative: the caller dialed us, so greet first.

        Unlike the outbound opener there is no callee channel to wait for -- the
        inbound channel is already answered by the time media starts. When no
        inbound greeting is configured (empty text) this is a no-op and the bot
        falls back to the legacy "wait for the caller" behavior.
        """

        text = " ".join(self.config.inbound_opening_utterance.split())
        if not text or session.state.primary_ended:
            metrics.mark(
                "inbound_opening_utterance_skipped",
                configured=bool(text),
                primary_ended=session.state.primary_ended,
            )
            return None if not text else {"text": text, "spoken": False, "interrupted": False, "skipped": True}
        spoken = pipeline.speak_text(text, label="inbound_opening")
        metrics.mark(
            "inbound_opening_utterance_finished",
            spoken=bool(spoken.get("spoken")),
            interrupted=bool(spoken.get("interrupted")),
        )
        return {**spoken, "skipped": False}

    def _wait_for_outbound_channel_up(self, session: AsteriskExternalMediaSession) -> bool:
        deadline = time.monotonic() + self.config.outbound_answer_timeout_seconds
        while not session.state.primary_ended:
            if session.state.primary_channel_is_up():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def _live_frames_until_turn(self, session: AsteriskExternalMediaSession, transport: UdpRtpTransport) -> Iterable[AudioFrame]:
        idle_deadline = time.monotonic() + self.config.idle_timeout_seconds
        while not session.state.primary_ended and time.monotonic() < idle_deadline:
            frame = transport.recv_frame()
            if frame is None:
                continue
            idle_deadline = time.monotonic() + self.config.idle_timeout_seconds
            yield frame

    def _require_live_asterisk(self) -> None:
        if not self.config.live_enabled:
            raise RuntimeError("Set REALTIME_LIVE_ENABLED=1 before starting the live Asterisk realtime worker")
        missing = self.config.asterisk.redacted()["missing_live_env"]
        if missing:
            raise RuntimeError(f"Missing live Asterisk env: {', '.join(missing)}")
        placeholders = self.config.unresolved_live_env_placeholders()
        if placeholders:
            raise RuntimeError(f"Unresolved live env placeholders: {', '.join(placeholders)}")
        approval = self.operator_approval_status()
        if self.config.operator_approval_required and not approval.get("ok"):
            raise RuntimeError(f"Live operator approval not valid: {approval.get('reason') or 'operator_approval_invalid'}")
        cutover = self.cutover_readiness_status()
        if self.config.cutover_readiness_required and not cutover.get("ok"):
            raise RuntimeError(f"Realtime cutover readiness not valid: {cutover.get('reason') or 'cutover_readiness_invalid'}")

    def _require_provider_ready(self) -> None:
        provider_kind = self.config.provider_kind()
        if provider_kind == "unsupported":
            raise RuntimeError(f"Unsupported REALTIME_PROVIDER_MODE={self.config.provider_mode!r}")
        stt_kind = self.config.stt_kind()
        if stt_kind == "unsupported":
            raise RuntimeError(f"Unsupported REALTIME_STT_PROVIDER={self.config.stt_provider!r}")
        if self.config.llm_kind() == "unsupported":
            raise RuntimeError(f"Unsupported REALTIME_LLM_PROVIDER={self.config.llm_provider!r}")
        if self.config.tts_kind() == "unsupported":
            raise RuntimeError(f"Unsupported REALTIME_TTS_PROVIDER={self.config.tts_provider!r}")
        missing = self.config.missing_provider_env()
        if missing:
            raise RuntimeError(f"Missing realtime provider env: {', '.join(missing)}")
        missing_runtime_dependencies = self.config.missing_runtime_dependencies()
        if missing_runtime_dependencies:
            raise RuntimeError(f"Missing realtime runtime dependencies: {', '.join(missing_runtime_dependencies)}")

    def _require_provider_prewarmed_for_live(self) -> dict[str, object]:
        check = self.prewarm_provider_clients(allow_provider_network=False)
        if check.get("ok"):
            return check
        reason = str(check.get("reason") or "provider_prewarm_failed")
        message = str(check.get("message") or reason)
        raise RuntimeError(f"Realtime provider prewarm failed before live ARI startup: {reason}: {message}")

    def _build_pipeline(
        self,
        send_packet,
        recorder: StreamingCallRecorder,
        *,
        metrics: CallMetrics | None = None,
        barge_in_frame_reader=None,
        config: "RealtimeWorkerConfig | None" = None,
    ) -> RealtimeMediaPipeline:
        config = config or self.config
        stt, llm, tts = self._provider_adapters(config)
        return RealtimeMediaPipeline(
            stt=stt,
            llm=llm,
            tts=tts,
            rtp_sender=PacedRtpSender(send_packet, prebuffer_ms=config.tts_rtp_prebuffer_ms),
            recorder=recorder,
            detector=RmsTurnDetector(TurnDetectionConfig(silence_hold_ms=config.silence_hold_ms, min_speech_ms=config.min_speech_ms)),
            metrics=metrics,
            barge_in_frame_reader=barge_in_frame_reader,
            barge_in_config=BargeInConfig(speech_rms=config.barge_in_speech_rms),
            semantic_endpointing=SemanticEndpointDetector(config.semantic_endpoint_config()),
            accept_interim_on_rms_endpoint=config.accept_interim_on_rms_endpoint,
            max_pending_phrases=config.max_pending_phrases,
            llm_fallback=self._llm_fallback_config(llm, config=config),
        )

    def _llm_fallback_config(self, llm: object, config: "RealtimeWorkerConfig | None" = None) -> LLMFallbackConfig:
        config = config or self.config
        fallback_llm = None
        fallback_model_id = ""
        if config.uses_gemini_llm():
            # Reuse the primary Gemini client plumbing with the fast fallback model.
            fallback_model_id = config.llm_fallback_model_id
            fallback_llm = _GeminiFallbackLLM(llm, fallback_model_id)
        return LLMFallbackConfig(
            retry_budget_seconds=config.llm_retry_budget_seconds,
            fallback_llm=fallback_llm,
            fallback_model_id=fallback_model_id,
            recovery_utterance=config.recovery_utterance,
            language=config.fake_language,
        )

    @staticmethod
    def _stack_fingerprint(config: "RealtimeWorkerConfig") -> tuple:
        """A hashable identity of the resolved stack for adapter-cache keying.

        Two per-call configs that select the same providers/models/voice/language
        share one adapter triple; different agents get distinct entries.
        """
        return (
            config.active_stt_name(),
            config.deepgram_model_id,
            config.active_llm_name(),
            config.gemini_model_id,
            config.active_tts_name(),
            config.elevenlabs_model_id,
            config.elevenlabs_voice_id,
            config.fake_language,
        )

    def _provider_adapters(self, config: "RealtimeWorkerConfig | None" = None) -> tuple[object, object, object]:
        config = config or self.config
        self._require_provider_ready()
        fingerprint = self._stack_fingerprint(config)
        with self._provider_lock:
            cached = self._provider_adapter_cache.get(fingerprint)
            if cached and {"stt", "llm", "tts"}.issubset(cached):
                return cached["stt"], cached["llm"], cached["tts"]
            registry = _provider_registry()
            stt = registry.build("stt", config.active_stt_name(), config)
            llm = registry.build("llm", config.active_llm_name(), config)
            tts_name = config.active_tts_name()
            tts_config = config
            if tts_name == "elevenlabs" and config is self.config:
                # Default env stack: resolve (and cache) the ElevenLabs voice id
                # once, then hand the built-in factory a config that already
                # carries it so no second voice-name network lookup is performed.
                tts_config = replace(config, elevenlabs_voice_id=self._resolve_elevenlabs_voice_id())
            # A per-call (agent) config already carries the resolved voice id from
            # the control plane, so it is used as-is with no extra network lookup.
            tts = registry.build("tts", tts_name, tts_config)
            self._provider_adapter_cache[fingerprint] = {"stt": stt, "llm": llm, "tts": tts}
            return stt, llm, tts

    def _provider_adapter_cache_status(self) -> dict[str, object]:
        with self._provider_lock:
            components: set[str] = set()
            for stack in self._provider_adapter_cache.values():
                components.update(key for key in ("stt", "llm", "tts") if key in stack)
        ordered = sorted(components)
        return {
            "components": ordered,
            "stt": "stt" in ordered,
            "llm": "llm" in ordered,
            "tts": "tts" in ordered,
        }

    def _resolve_elevenlabs_voice_id(self) -> str:
        with self._provider_lock:
            if self._resolved_elevenlabs_voice_id:
                return self._resolved_elevenlabs_voice_id
            self._resolved_elevenlabs_voice_id = resolve_elevenlabs_voice_id(
                api_key=self.config.elevenlabs_api_key,
                voice_id=self.config.elevenlabs_voice_id,
                voice_name=self.config.elevenlabs_voice,
            )
            return self._resolved_elevenlabs_voice_id

    # -- per-agent voice stack (SCH-205/206) ------------------------------
    def _agent_config_fetch_configured(self) -> bool:
        return bool(self.config.control_plane_api_url and self.config.gateway_api_token)

    def _fetch_agent_realtime_config(
        self,
        agent_id: str,
        *,
        get_json: "Callable[[str, dict[str, str]], dict[str, object]] | None" = None,
    ) -> dict[str, object] | None:
        """GET the resolved per-agent stack from the control plane, cached per
        agent id for the process lifetime. Returns None when unconfigured."""

        with self._agent_config_lock:
            if agent_id in self._agent_config_cache:
                return self._agent_config_cache[agent_id]
        if not self._agent_config_fetch_configured():
            return None
        url = self.config.control_plane_api_url.rstrip("/") + f"/v1/agents/{agent_id}/realtime-config"
        headers = {"Authorization": f"Bearer {self.config.gateway_api_token}", "Content-Type": "application/json"}
        resolved = get_json(url, headers) if get_json else _get_json(url, headers)
        if not isinstance(resolved, dict):
            resolved = None
        with self._agent_config_lock:
            self._agent_config_cache[agent_id] = resolved
        return resolved

    def _resolve_agent_call_config(
        self,
        agent_id: str,
        *,
        get_json: "Callable[[str, dict[str, str]], dict[str, object]] | None" = None,
    ) -> "RealtimeWorkerConfig":
        """Return a per-call config for ``agent_id`` (env stack when absent/failed)."""

        if not agent_id:
            return self.config
        try:
            resolved = self._fetch_agent_realtime_config(agent_id, get_json=get_json)
        except Exception:
            # Degrade to the env stack, never to a broken call.
            return self.config
        if not resolved:
            return self.config
        try:
            return self._apply_resolved_stack(self.config, resolved)
        except Exception:
            return self.config

    @staticmethod
    def _apply_resolved_stack(config: "RealtimeWorkerConfig", resolved: dict[str, object]) -> "RealtimeWorkerConfig":
        """Map a resolved agent stack onto RealtimeWorkerConfig via replace()."""

        transcriber = resolved.get("transcriber") if isinstance(resolved.get("transcriber"), dict) else {}
        model = resolved.get("model") if isinstance(resolved.get("model"), dict) else {}
        voice = resolved.get("voice") if isinstance(resolved.get("voice"), dict) else {}
        language = str(resolved.get("language") or config.fake_language)
        system_prompt = str(resolved.get("system_prompt") or config.gemini_system_prompt)
        first_message = str(resolved.get("first_message") or config.opening_utterance)
        return replace(
            config,
            stt_provider=str(transcriber.get("registry_name") or transcriber.get("provider") or config.stt_provider),
            deepgram_model_id=str(transcriber.get("model") or config.deepgram_model_id),
            deepgram_language=language,
            llm_provider=str(model.get("registry_name") or model.get("provider") or config.llm_provider),
            gemini_model_id=str(model.get("model") or config.gemini_model_id),
            gemini_system_prompt=system_prompt,
            tts_provider=str(voice.get("registry_name") or voice.get("provider") or config.tts_provider),
            elevenlabs_model_id=str(voice.get("model") or config.elevenlabs_model_id),
            elevenlabs_voice_id=str(voice.get("voice_id") or config.elevenlabs_voice_id),
            elevenlabs_language_code=language,
            fake_language=language,
            opening_utterance=first_message,
        )

    def _has_call_capacity(self) -> bool:
        status = self.capacity_status()
        return int(status["available_call_slots"]) > 0

    def _register_active_call(self, call_id: str, *, metadata: dict[str, str] | None = None, rtp_port: int | None = None) -> None:
        safe_metadata = dict(metadata or {})
        detail: dict[str, object] = {
            "asterisk_channel_id": call_id,
            "call_id": safe_metadata.get("call_id", ""),
            "openphonex": safe_metadata,
        }
        for key in (
            "organization_id",
            "project_id",
            "number_id",
            "agent_id",
            "agent_endpoint_id",
            "phone_server_id",
            "trial_call_id",
            "media_pipeline",
            "worker_job_id",
        ):
            if safe_metadata.get(key):
                detail[key] = safe_metadata[key]
        if rtp_port is not None:
            detail["rtp_port"] = rtp_port
        with self._capacity_lock:
            self._active_call_ids.add(call_id)
            self._active_call_details[call_id] = detail

    def _unregister_active_call(self, call_id: str) -> None:
        with self._capacity_lock:
            self._active_call_ids.discard(call_id)
            self._active_call_details.pop(call_id, None)

    def _acquire_rtp_port(self) -> int | None:
        with self._capacity_lock:
            ports = self.config.rtp_ports()
            if not ports:
                return None
            start_index = 0
            if self._next_rtp_port in ports:
                start_index = ports.index(self._next_rtp_port)
            for offset in range(len(ports)):
                port = ports[(start_index + offset) % len(ports)]
                if port not in self._active_rtp_ports:
                    self._active_rtp_ports.add(port)
                    next_index = (ports.index(port) + 1) % len(ports)
                    self._next_rtp_port = ports[next_index]
                    return port
            return None

    def _release_rtp_port(self, port: int) -> None:
        with self._capacity_lock:
            self._active_rtp_ports.discard(port)


class _GeminiFallbackLLM:
    """Non-streaming fallback replies on the shared Gemini client with a fast model."""

    def __init__(self, llm: object, model_id: str):
        self._llm = llm
        self.model_id = model_id

    def generate_reply(self, messages: list[dict[str, str]]) -> str:
        return str(self._llm.generate_reply(messages, model_id=self.model_id))


def _synthetic_caller_frames() -> Iterable[AudioFrame]:
    yield from (AudioFrame.silence(20) for _ in range(3))
    yield from (AudioFrame.tone(20, direction="caller") for _ in range(16))
    yield from (AudioFrame.silence(20) for _ in range(20))


def _audio_file_frames(input_audio: Path, *, sample_rate: int = 8000, frame_ms: int = 20) -> Iterable[AudioFrame]:
    try:
        yield from _pcm16_wav_frames(input_audio, sample_rate=sample_rate, frame_ms=frame_ms)
    except (EOFError, ValueError, wave.Error):
        yield from _ffmpeg_pcm16_frames(input_audio, sample_rate=sample_rate, frame_ms=frame_ms)


def _pcm16_wav_frames(input_audio: Path, *, sample_rate: int, frame_ms: int) -> Iterable[AudioFrame]:
    with wave.open(str(input_audio), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != sample_rate:
            raise ValueError("WAV must be 16-bit mono PCM at 8kHz for direct decoding")
        frames_per_chunk = max(1, int(sample_rate * frame_ms / 1000))
        frame_bytes = frames_per_chunk * 2
        timestamp_ms = 0
        while True:
            payload = wav.readframes(frames_per_chunk)
            if not payload:
                break
            if len(payload) < frame_bytes:
                payload = payload + (b"\0" * (frame_bytes - len(payload)))
            yield AudioFrame(payload, sample_rate=sample_rate, direction="caller", timestamp_ms=timestamp_ms)
            timestamp_ms += frame_ms


def _ffmpeg_pcm16_frames(input_audio: Path, *, sample_rate: int, frame_ms: int) -> Iterable[AudioFrame]:
    frame_bytes = max(1, int(sample_rate * frame_ms / 1000)) * 2
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_audio),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode non-8kHz PCM WAV recorded audio") from exc
    assert process.stdout is not None
    timestamp_ms = 0
    while True:
        payload = process.stdout.read(frame_bytes)
        if not payload:
            break
        if len(payload) < frame_bytes:
            payload = payload + (b"\0" * (frame_bytes - len(payload)))
        yield AudioFrame(payload, sample_rate=sample_rate, direction="caller", timestamp_ms=timestamp_ms)
        timestamp_ms += frame_ms
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    if process.wait(timeout=5) != 0:
        raise RuntimeError(f"ffmpeg failed to decode recorded audio: {stderr[-800:]}")


def _operator_approval_status(
    *,
    required: bool,
    record_path: str,
    expected_fingerprint: str,
    required_directions: set[str],
) -> dict[str, object]:
    base: dict[str, object] = {
        "required": required,
        "ok": not required,
        "record_configured": bool(record_path),
        "fingerprint_configured": bool(expected_fingerprint),
        "fingerprint_short": expected_fingerprint[:12],
        "required_directions": sorted(required_directions),
        "live_actions_performed": False,
    }
    if not required:
        return {**base, "reason": "not_required"}
    if not record_path:
        return {**base, "ok": False, "reason": "operator_approval_record_not_configured"}
    if not expected_fingerprint:
        return {**base, "ok": False, "reason": "approval_packet_fingerprint_not_configured"}
    path = Path(record_path).expanduser()
    base["record_path"] = str(path)
    if not path.is_file():
        return {**base, "ok": False, "reason": "operator_approval_record_file_not_found"}
    try:
        record = json.loads(path.read_text())
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "reason": "operator_approval_record_json_unreadable",
            "error": exc.__class__.__name__,
        }
    if not isinstance(record, dict):
        return {**base, "ok": False, "reason": "operator_approval_record_json_unreadable"}
    name = str(record.get("name") or "")
    if name and name != "openphonex_live_measurement_operator_approval":
        return {**base, "ok": False, "reason": "operator_approval_record_schema_invalid"}
    record_fingerprint = _operator_approval_record_fingerprint(record)
    approved_by = str(record.get("approved_by") or "").strip()
    approval_note = str(record.get("approval_note") or record.get("notes") or "").strip()
    approved_at = _parse_utc_datetime(record.get("approved_at"))
    expires_at = _parse_utc_datetime(record.get("expires_at"))
    approved_directions = _operator_approval_record_directions(record)
    missing_directions = sorted(required_directions - approved_directions)
    reason = ""
    if not record_fingerprint:
        reason = "operator_approval_record_missing_fingerprint"
    elif record_fingerprint != expected_fingerprint:
        reason = "operator_approval_record_fingerprint_mismatch"
    elif not approved_by:
        reason = "operator_approval_record_missing_approved_by"
    elif approved_at is None:
        reason = "operator_approval_record_missing_or_invalid_approved_at"
    elif expires_at is None:
        reason = "operator_approval_record_missing_or_invalid_expires_at"
    elif expires_at <= datetime.now(timezone.utc):
        reason = "operator_approval_record_expired"
    elif missing_directions:
        reason = "operator_approval_record_missing_required_directions"
    ok = reason == ""
    return {
        **base,
        "ok": ok,
        "reason": reason,
        "approved_by": approved_by,
        "approval_note_configured": bool(approval_note),
        "approved_at": approved_at.isoformat().replace("+00:00", "Z") if approved_at else "",
        "expires_at": expires_at.isoformat().replace("+00:00", "Z") if expires_at else "",
        "approved_directions": sorted(approved_directions),
        "missing_directions": missing_directions,
        "fingerprint": record_fingerprint,
        "fingerprint_short": record_fingerprint[:12],
        "fingerprint_match": bool(record_fingerprint) and record_fingerprint == expected_fingerprint,
    }


def _cutover_readiness_status(
    *,
    required: bool,
    report_path: str,
    expected_fingerprint: str,
) -> dict[str, object]:
    base: dict[str, object] = {
        "required": required,
        "ok": not required,
        "report_configured": bool(report_path),
        "fingerprint_configured": bool(expected_fingerprint),
        "fingerprint_short": expected_fingerprint[:12],
        "live_actions_performed": False,
        "ari_connection_performed": False,
        "provider_network_actions_performed": [],
    }
    if not required:
        return {**base, "reason": "not_required"}
    if not report_path:
        return {**base, "ok": False, "reason": "cutover_readiness_report_not_configured"}
    if not expected_fingerprint:
        return {**base, "ok": False, "reason": "cutover_readiness_fingerprint_not_configured"}
    path = Path(report_path).expanduser()
    base["report_path"] = str(path)
    if not path.is_file():
        return {**base, "ok": False, "reason": "cutover_readiness_report_file_not_found"}
    try:
        report = json.loads(path.read_text())
    except Exception as exc:
        return {**base, "ok": False, "reason": "cutover_readiness_report_json_unreadable", "error": exc.__class__.__name__}
    if not isinstance(report, dict):
        return {**base, "ok": False, "reason": "cutover_readiness_report_json_unreadable"}
    cutover = report.get("cutover_readiness") if isinstance(report.get("cutover_readiness"), dict) else {}
    cutover_check = _report_named_check(report, "realtime_cutover_readiness")
    report_fingerprint = _report_approval_fingerprint(report)
    ready = bool(cutover.get("ready"))
    report_ok = bool(report.get("ok"))
    check_ok = bool(cutover_check.get("ok"))
    fingerprint_match = bool(report_fingerprint) and report_fingerprint == expected_fingerprint
    reason = ""
    if not report_ok:
        reason = "cutover_readiness_report_not_ok"
    elif not ready:
        reason = "cutover_readiness_not_ready"
    elif not check_ok:
        reason = "cutover_readiness_check_not_ok"
    elif not report_fingerprint:
        reason = "cutover_readiness_report_missing_fingerprint"
    elif not fingerprint_match:
        reason = "cutover_readiness_fingerprint_mismatch"
    ok = reason == ""
    return {
        **base,
        "ok": ok,
        "reason": reason,
        "ready": ready,
        "report_ok": report_ok,
        "cutover_check_ok": check_ok,
        "next_step": str(cutover.get("next_step") or ""),
        "missing_count": len(cutover.get("missing")) if isinstance(cutover.get("missing"), list) else 0,
        "fingerprint": report_fingerprint,
        "fingerprint_short": report_fingerprint[:12],
        "fingerprint_match": fingerprint_match,
        "legacy_replacement_target": str(cutover.get("legacy_replacement_target") or ""),
    }


def _report_named_check(report: dict[str, object], name: str) -> dict[str, object]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return {}
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return {}


def _report_approval_fingerprint(report: dict[str, object]) -> str:
    packet = report.get("live_approval_packet")
    if isinstance(packet, dict):
        fingerprint = packet.get("fingerprint")
        if isinstance(fingerprint, dict) and isinstance(fingerprint.get("value"), str):
            return fingerprint["value"].strip()
    cutover = report.get("cutover_readiness")
    if isinstance(cutover, dict):
        short = cutover.get("approval_fingerprint_short")
        if isinstance(short, str) and len(short.strip()) == 64:
            return short.strip()
    return ""


def _live_startup_next_step(
    *,
    provider_ready: bool,
    live_ready: bool,
    operator_approval: object,
    cutover_readiness: object,
    runtime_dependencies: object = None,
    provider_prewarm: object = None,
    unresolved_live_env_placeholders: object = None,
) -> str:
    if isinstance(runtime_dependencies, dict) and runtime_dependencies.get("missing"):
        return "install_realtime_runtime_dependencies"
    if not provider_ready:
        if isinstance(provider_prewarm, dict):
            reason = str(provider_prewarm.get("reason") or "")
            if reason == "runtime_dependencies_missing":
                return "install_realtime_runtime_dependencies"
            if reason == "provider_not_ready":
                return "configure_realtime_provider_environment"
            if reason == "provider_prewarm_network_approval_required":
                return "configure_elevenlabs_voice_id_or_run_approved_provider_prewarm"
            if reason == "provider_prewarm_failed":
                return "fix_realtime_provider_prewarm"
        return "configure_realtime_provider_environment"
    if isinstance(unresolved_live_env_placeholders, list) and unresolved_live_env_placeholders:
        return "replace_live_env_placeholders"
    if isinstance(operator_approval, dict) and operator_approval.get("required") and not operator_approval.get("ok"):
        reason = str(operator_approval.get("reason") or "")
        if reason in {
            "operator_approval_record_not_configured",
            "approval_packet_fingerprint_not_configured",
            "operator_approval_record_file_not_found",
        }:
            return "configure_live_worker_operator_approval_record"
        if reason == "operator_approval_record_expired":
            return "refresh_live_worker_operator_approval_record"
        if reason in {"operator_approval_record_fingerprint_mismatch", "operator_approval_record_missing_required_directions"}:
            return "rerun_preflight_and_approve_matching_live_packet"
        return "fix_live_worker_operator_approval_record"
    if isinstance(cutover_readiness, dict) and cutover_readiness.get("required") and not cutover_readiness.get("ok"):
        reason = str(cutover_readiness.get("reason") or "")
        if reason in {
            "cutover_readiness_report_not_configured",
            "cutover_readiness_fingerprint_not_configured",
            "cutover_readiness_report_file_not_found",
        }:
            return "configure_realtime_cutover_readiness_report"
        if reason == "cutover_readiness_fingerprint_mismatch":
            return "rerun_cutover_readiness_for_matching_approval_packet"
        if reason in {"cutover_readiness_report_not_ok", "cutover_readiness_not_ready", "cutover_readiness_check_not_ok"}:
            return str(cutover_readiness.get("next_step") or "rerun_realtime_cutover_readiness")
        return "fix_realtime_cutover_readiness_report"
    if not live_ready:
        return "configure_asterisk_ari_and_enable_realtime_live_gate"
    return "start_live_worker"


def _unresolved_env_placeholder(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    if lowered.startswith("/path/to/") or lowered in {"changeme", "change-me", "replace-me", "todo", "tbd"}:
        return True
    placeholder_tokens = (
        "set-secret",
        "placeholder",
        "replace-with",
        "latest-approval-packet-fingerprint",
        "asterisk-ari-user",
        "realtime-worker-host-reachable-from-asterisk",
    )
    return any(token in lowered for token in placeholder_tokens)


def _operator_approval_record_fingerprint(record: dict[str, object]) -> str:
    for key in ("approval_packet_fingerprint", "fingerprint", "approved_fingerprint"):
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict) and isinstance(value.get("value"), str):
            return value["value"].strip()
    packet = record.get("live_approval_packet")
    if isinstance(packet, dict):
        fingerprint = packet.get("fingerprint")
        if isinstance(fingerprint, dict) and isinstance(fingerprint.get("value"), str):
            return fingerprint["value"].strip()
    return ""


def _operator_approval_record_directions(record: dict[str, object]) -> set[str]:
    directions: set[str] = set()
    raw = record.get("approved_directions")
    if isinstance(raw, list):
        directions.update(str(item).strip().lower() for item in raw)
    signoff = record.get("operator_signoff")
    if isinstance(signoff, dict):
        if signoff.get("inbound_call_approved"):
            directions.add("inbound")
        if signoff.get("outbound_call_approved"):
            directions.add("outbound")
    if record.get("inbound_call_approved"):
        directions.add("inbound")
    if record.get("outbound_call_approved"):
        directions.add("outbound")
    return {item for item in directions if item in {"inbound", "outbound"}}


def _parse_utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _approval_directions_env(raw: str | None) -> tuple[str, ...]:
    values = [item.strip().lower() for item in (raw or "inbound,outbound").split(",")]
    directions = tuple(item for item in values if item in {"inbound", "outbound"})
    return directions or ("inbound", "outbound")


def _get_json(url: str, headers: dict[str, str]) -> dict[str, object]:
    request = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read()
            parsed = json.loads(data.decode()) if data else {}
            if isinstance(parsed, dict):
                return parsed
            return {"status": "ok", "body": parsed}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"control plane GET failed: HTTP {exc.code} {detail[:500]}") from exc


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    body = json.dumps(payload, sort_keys=True).encode()
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read()
            parsed = json.loads(data.decode()) if data else {}
            if isinstance(parsed, dict):
                return parsed
            return {"status": "ok", "body": parsed}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"control plane health report failed: HTTP {exc.code} {detail[:500]}") from exc


def _safe_path_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)[:80] or "call"


def _turn_clip_artifacts(root: Path) -> list[str]:
    return [str(path) for path in sorted(root.glob("turn-*-*.wav"))]


def _realtime_artifacts_payload(recorder: StreamingCallRecorder, root: Path) -> dict[str, object]:
    return {
        "caller": str(recorder.caller_path),
        "agent": str(recorder.agent_path),
        "mixed": str(recorder.mixed_path),
        "waveform": str(root / "waveform.json"),
        "waveforms": str(root / "waveforms.json"),
        "latency": str(root / "latency.json"),
        "transcript": str(root / "transcript.json"),
        "turn_clips": _turn_clip_artifacts(root),
    }


def _turn_payload(index: int, result) -> dict[str, object]:
    return {
        "index": index,
        "user": result.transcript,
        "assistant": result.assistant_text,
        "assistant_spoken": result.assistant_spoken_text,
        "interrupted": result.interrupted,
        "llm_fallback_leg": result.llm_fallback_leg,
        "latency": result.latency,
        "caller_audio": f"turn-{index:03d}-caller.wav",
        "agent_audio": f"turn-{index:03d}-agent.wav",
    }


def _transport_stats_payload(transport: object) -> dict[str, object]:
    stats = getattr(transport, "stats", None)
    if not stats or not hasattr(stats, "as_dict"):
        return {}
    payload = stats.as_dict()
    return payload if isinstance(payload, dict) else {}


def _event_channel_id(event: dict[str, object]) -> str:
    channel = event.get("channel")
    if not isinstance(channel, dict):
        return ""
    return str(channel.get("id") or "")


def _event_channel_name(event: dict[str, object]) -> str:
    channel = event.get("channel")
    if not isinstance(channel, dict):
        return ""
    return str(channel.get("name") or "")


def _reject_over_capacity(ari: AriClientProtocol, channel_id: str) -> None:
    try:
        ari.delete(f"/channels/{channel_id}")
    except Exception:
        pass


def _realtime_call_summary(turns: list[dict[str, object]]) -> dict[str, object]:
    completed_turns = [turn for turn in turns if str(turn.get("user") or "").strip() or str(turn.get("assistant") or "").strip()]
    interrupted = any(bool(turn.get("interrupted")) for turn in completed_turns)
    first_user = str(completed_turns[0].get("user") or "").strip() if completed_turns else ""
    first_assistant = str(completed_turns[0].get("assistant") or "").strip() if completed_turns else ""
    if first_user and first_assistant:
        text = f"Caller said: {first_user} Assistant replied: {first_assistant}"
    elif first_user:
        text = f"Caller said: {first_user}"
    else:
        text = "No completed realtime conversation turns were captured."
    return {
        "text": text[:500],
        "turn_count": len(completed_turns),
        "interrupted": interrupted,
        "source": "realtime_worker_deterministic_summary",
    }


def _normalize_provider_mode(value: str | None) -> str:
    mode = (value or "fake").strip().lower()
    if mode in {"", "fake"}:
        return "fake"
    if mode in {"elevenlabs", "elevenlabs_v3", "elevenlabs_v3_tts"}:
        return "elevenlabs_v3_tts"
    if mode in {"gemini", "gemini_llm", "gemini_llm_fake_tts"}:
        return "gemini_llm_fake_tts"
    if mode in {"gemini_llm_elevenlabs", "gemini_llm_elevenlabs_v3", "gemini_llm_elevenlabs_v3_tts"}:
        return "gemini_llm_elevenlabs_v3_tts"
    if mode in {
        "full",
        "full_realtime",
        "full_realtime_estonian",
        "production_realtime",
        "production_estonian",
        "deepgram_gemini_elevenlabs",
        "deepgram_gemini_elevenlabs_v3",
        "deepgram_gemini_elevenlabs_v3_tts",
    }:
        return "deepgram_gemini_elevenlabs_v3"
    return "unsupported"


def _normalize_stt_provider(value: str | None) -> str:
    provider = (value or "fake").strip().lower()
    if provider in {"", "fake"}:
        return "fake"
    if provider in {"deepgram", "deepgram_live", "deepgram_streaming"}:
        return "deepgram"
    return "unsupported"


def _optional_int_env(name: str, *, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(text)


def _semantic_punctuation_env(raw: str | None) -> tuple[str, ...]:
    value = (raw or ".!?").strip()
    chars: list[str] = []
    for char in value:
        if char.isspace() or char == ",":
            continue
        if char not in chars:
            chars.append(char)
    return tuple(chars or [".", "!", "?"])


def make_health_handler(worker: RealtimeMediaWorker) -> type[BaseHTTPRequestHandler]:
    class RealtimeWorkerHealthHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            if env_flag(os.environ.get("REALTIME_HTTP_LOGS"), default=False):
                super().log_message(fmt, *args)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path not in {"/", "/health", "/status", "/v1/realtime-worker/status"}:
                self._send(404, {"ok": False, "error": "not_found", "path": path})
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

    return RealtimeWorkerHealthHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenPhonex realtime Asterisk ExternalMedia worker")
    parser.add_argument("--env-file", action="append", default=[], help="Read dotenv-style KEY=VALUE pairs before building worker config. Repeatable.")
    parser.add_argument("--status", action="store_true", help="Print redacted realtime worker status and exit.")
    parser.add_argument("--offline-synthetic-call", action="store_true", help="Run a safe fake-provider call through the realtime pipeline.")
    parser.add_argument("--offline-streaming-overlap-check", action="store_true", help="Run a deterministic local check proving LLM phrase production overlaps TTS/RTP playback.")
    parser.add_argument("--offline-barge-in-check", action="store_true", help="Run a deterministic local barge-in interruption/replay check.")
    parser.add_argument("--offline-recorded-audio", default="", help="Run a local recorded audio file through the realtime pipeline without Asterisk.")
    parser.add_argument("--offline-output-dir", default="", help="Output directory for --offline-synthetic-call artifacts.")
    parser.add_argument("--serve-health", action="store_true", help="Run the realtime worker health/status HTTP server.")
    parser.add_argument("--health-bind", default="", help="Host for --serve-health. Defaults to REALTIME_HEALTH_BIND.")
    parser.add_argument("--health-port", type=int, default=0, help="Port for --serve-health. Defaults to REALTIME_HEALTH_PORT.")
    parser.add_argument("--check-live-startup", action="store_true", help="Validate live/provider/operator gates without opening ARI or placing calls.")
    parser.add_argument("--serve-asterisk", action="store_true", help="Run the guarded ARI/ExternalMedia realtime worker.")
    parser.add_argument("--report-health", action="store_true", help="Continuously POST realtime worker health to the OpenPhonex gateway deployment while serving.")
    parser.add_argument("--report-health-once", action="store_true", help="POST one realtime worker health report to the OpenPhonex gateway deployment and exit.")
    args = parser.parse_args(argv)

    _apply_env_files(args.env_file)
    config = RealtimeWorkerConfig.from_env()
    if args.health_bind or args.health_port:
        config = replace(
            config,
            health_bind=args.health_bind or config.health_bind,
            health_port=args.health_port or config.health_port,
        )
    worker = RealtimeMediaWorker(config)
    if args.check_live_startup:
        result = worker.live_startup_preflight()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.report_health_once:
        result = worker.post_gateway_health_report()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.offline_streaming_overlap_check:
        output_dir = Path(args.offline_output_dir) if args.offline_output_dir else None
        print(json.dumps(worker.run_offline_streaming_overlap_check(output_dir), indent=2, sort_keys=True))
        return 0
    if args.offline_barge_in_check:
        output_dir = Path(args.offline_output_dir) if args.offline_output_dir else None
        print(json.dumps(worker.run_offline_barge_in_check(output_dir), indent=2, sort_keys=True))
        return 0
    if args.offline_recorded_audio:
        output_dir = Path(args.offline_output_dir) if args.offline_output_dir else None
        print(json.dumps(worker.run_offline_recorded_call(Path(args.offline_recorded_audio), output_dir), indent=2, sort_keys=True))
        return 0
    if args.offline_synthetic_call:
        output_dir = Path(args.offline_output_dir) if args.offline_output_dir else None
        print(json.dumps(worker.run_offline_synthetic_call(output_dir), indent=2, sort_keys=True))
        return 0
    if args.serve_asterisk and args.serve_health:
        print(json.dumps(_run_with_optional_reporter(worker, args.report_health, worker.serve_asterisk_with_health), indent=2, sort_keys=True))
        return 0
    if args.serve_asterisk:
        print(json.dumps(_run_with_optional_reporter(worker, args.report_health, worker.serve_asterisk), indent=2, sort_keys=True))
        return 0
    if args.serve_health:
        _run_with_optional_reporter(worker, args.report_health, worker.serve_health)
        return 0
    print(json.dumps(worker.status(), indent=2, sort_keys=True))
    return 0


def _apply_env_files(paths: Iterable[str]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        values = load_env_file(path)
        loaded_keys: list[str] = []
        for key, value in values.items():
            if key in os.environ:
                continue
            os.environ[key] = value
            loaded_keys.append(key)
        evidence.append(
            {
                "path": str(path),
                "present": path.exists(),
                "loaded_key_count": len(loaded_keys),
                "loaded_keys": sorted(loaded_keys),
                "existing_env_preserved_count": len(values) - len(loaded_keys),
                "secrets_redacted": True,
            }
        )
    return evidence


def _run_with_optional_reporter(worker: RealtimeMediaWorker, enabled: bool, fn: Callable[[], Any]) -> Any:
    if not enabled:
        return fn()
    stop, thread = worker.start_control_plane_reporter_thread()
    try:
        return fn()
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())

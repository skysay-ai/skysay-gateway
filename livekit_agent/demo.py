# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Standalone LiveKit demo profile and direct-provider stack.

It deliberately has no OpenPhonex control-plane dependency.  The small public
surface keeps provider mapping and interruption policy testable without
importing the LiveKit SDK or contacting a provider. ``load_demo_profile``
(the internal env-var-driven qualification demo) only ever builds an EN or ET
``DemoProfile``; ``build_demo_stack`` (shared with the real production path)
additionally builds the SI (si-LK / Sinhala) stack when given a profile from
``production.profile_from_realtime_config`` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


class DemoConfigurationError(RuntimeError):
    """The isolated demo cannot safely construct its direct-provider stack."""


# Languages the pinned turn-detector plugin (livekit-agents 1.6.6,
# ``inference.TurnDetector(version="v1-mini")``) actually supports, mirrored
# from ``livekit.agents.inference.eot.languages.LOCAL_LANGUAGES``. si-LK
# (Sinhala) is NOT in this set -- Estonian never was either -- so both fall
# back to VAD-only (Silero) endpointing below rather than silently running an
# uncalibrated (English-threshold) turn detector against audio it was never
# trained on. The tradeoff: VAD-only endpointing waits for a fixed silence
# window instead of a semantic end-of-turn signal, so turn-taking is slightly
# slower / less natural for si (and et) calls than it is for en.
TURN_DETECTOR_SUPPORTED_LANGUAGES = frozenset(
    {"ar", "de", "en", "es", "fr", "hi", "id", "it", "ja", "ko", "nl", "pt", "tr", "zh"}
)


def first_continuous_threshold_run(
    samples: Iterable[tuple[float, float]],
    *,
    after: float,
    duration: float,
    threshold: float,
) -> float | None:
    """Return the start of sustained activity, ignoring short ringback bursts."""

    started: float | None = None
    for observed_at, value in samples:
        if observed_at < after:
            continue
        if value >= threshold:
            started = started or observed_at
            if observed_at - started >= duration:
                return started
        else:
            started = None
    return None


@dataclass(frozen=True)
class DemoProfile:
    language: str
    stt_language: str
    tts_model: str
    voice_id: str
    llm_model: str
    instructions: str
    first_message: str
    # STT/TTS provider registry_name for this call. Defaults preserve the
    # pre-si-LK behavior for every existing caller that only ever set the
    # fields above (load_demo_profile, the estonian_probe fixtures, ...).
    stt_registry: str = "deepgram"
    tts_registry: str = "elevenlabs"


@dataclass(frozen=True)
class DemoStack:
    stt: Any
    llm: Any
    tts: Any
    vad: Any
    session_options: dict[str, Any]
    safe_identity: dict[str, str]


def semantic_marker_observed(language: str, transcript: str) -> bool:
    normalized = " ".join(transcript.casefold().split())
    # Nova-3 has produced all three spellings from the same caller-side
    # ElevenLabs sample.  This is an acoustic proof marker, not a spelling test.
    markers = ("ananass", "ananas", "annanas") if language == "et" else ("pineapple",)
    return any(marker in normalized for marker in markers)


def safe_selftest_failure(exc: BaseException) -> dict[str, str]:
    """Return useful diagnostics without leaking arbitrary provider exceptions."""

    result = {"result": "fail", "error_type": type(exc).__name__}
    if type(exc).__name__ == "DemoSelftestError":
        result["message"] = str(exc)[:300]
    return result


def load_demo_profile(environ: Mapping[str, str]) -> DemoProfile:
    profile = str(environ.get("SCH236_LANGUAGE_PROFILE") or "en").strip().lower()
    if profile not in {"en", "et"}:
        raise DemoConfigurationError("SCH236_LANGUAGE_PROFILE must be en or et")

    required = {
        "DEEPGRAM_API_KEY": environ.get("DEEPGRAM_API_KEY", ""),
        "GEMINI_API_KEY": environ.get("GEMINI_API_KEY", "")
        or environ.get("GOOGLE_API_KEY", ""),
        "ELEVENLABS_API_KEY": environ.get("ELEVENLABS_API_KEY", "")
        or environ.get("ELEVEN_API_KEY", ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise DemoConfigurationError("missing provider settings: " + ", ".join(missing))

    voice_variable = "SCH236_EN_VOICE_ID" if profile == "en" else "SCH236_ET_VOICE_ID"
    voice_id = str(
        environ.get(voice_variable) or environ.get("ELEVENLABS_VOICE_ID") or ""
    ).strip()
    if not voice_id:
        raise DemoConfigurationError(f"missing provider setting: {voice_variable}")

    llm_model = str(
        environ.get("SCH236_GEMINI_MODEL")
        or environ.get("GEMINI_LLM_MODEL")
        or "gemini-3.5-flash-lite"
    ).strip()
    if not llm_model.startswith("gemini-"):
        raise DemoConfigurationError("SCH236_GEMINI_MODEL must be a Gemini model")

    if profile == "et":
        return DemoProfile(
            language="et",
            stt_language="et",
            tts_model="eleven_v3",
            voice_id=voice_id,
            llm_model=llm_model,
            instructions=(
                "Oled OpenPhonexi soe ja loomulik eesti keelt kõnelev vastuvõtuassistent. "
                "Vasta lühidalt, ära korda juba öeldut ja küsi korraga ainult üks küsimus."
            ),
            first_message="Tere! Olete helistanud OpenPhonexile. Kuidas saan aidata?",
        )
    return DemoProfile(
        language="en",
        stt_language="en-US",
        tts_model="eleven_flash_v2_5",
        voice_id=voice_id,
        llm_model=llm_model,
        instructions=(
            "You are the warm, concise OpenPhonex receptionist. Speak naturally, keep replies "
            "short, never repeat facts the caller already gave, and ask only one question at a "
            "time."
        ),
        first_message="Hi, you've reached OpenPhonex. How can I help?",
    )


def _google_credentials_available(environ: Mapping[str, str]) -> bool:
    return bool(
        str(environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON") or "").strip()
        or str(environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    )


def _google_stt_credential_kwargs(environ: Mapping[str, str]) -> dict[str, Any]:
    """Resolve GCP credentials for LiveKit's Google STT plugin (Chirp 2).

    Two supported shapes, matching what ``livekit.plugins.google.STT`` itself
    accepts:

    * ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` -- the service-account JSON
      inlined as an env var (some container platforms prefer secrets-as-env
      over secrets-as-file); parsed and passed as ``credentials_info``.
    * ``GOOGLE_APPLICATION_CREDENTIALS`` -- the standard ADC file path;
      passed through explicitly as ``credentials_file`` below. (If NEITHER
      var is set, returning ``{}`` still works: the plugin then falls back to
      ``google.auth.default()``, which reads this same env var itself -- but
      that fallback path is not what's exercised when the var IS set, which
      is the expected case here.)

    NOTE this is a distinct credential from ``GEMINI_API_KEY`` -- Chirp 2 is a
    GCP Cloud Speech-to-Text API, not the Gemini API, and needs a real GCP
    service account (see the founder-facing setup steps in SELFHOST.md).
    """
    inline = str(environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON") or "").strip()
    if inline:
        try:
            return {"credentials_info": json.loads(inline)}
        except json.JSONDecodeError as exc:
            raise DemoConfigurationError(
                "GOOGLE_APPLICATION_CREDENTIALS_JSON must be valid service-account JSON"
            ) from exc
    credentials_file = str(environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if credentials_file:
        return {"credentials_file": credentials_file}
    return {}


def build_demo_stack(
    profile: DemoProfile,
    *,
    environ: Mapping[str, str],
    deepgram_stt: Callable[..., Any] | None = None,
    google_stt: Callable[..., Any] | None = None,
    gemini_llm: Callable[..., Any] | None = None,
    elevenlabs_tts: Callable[..., Any] | None = None,
    gemini_tts: Callable[..., Any] | None = None,
    stream_adapter: Callable[..., Any] | None = None,
    load_vad: Callable[..., Any] | None = None,
    load_turn_detector: Callable[..., Any] | None = None,
) -> DemoStack:
    stt_registry = profile.stt_registry
    tts_registry = profile.tts_registry
    turn_detector_supported = profile.language in TURN_DETECTOR_SUPPORTED_LANGUAGES

    missing_factory = (
        (stt_registry == "google" and google_stt is None)
        or (stt_registry != "google" and deepgram_stt is None)
        or gemini_llm is None
        or (tts_registry == "gemini_tts" and gemini_tts is None)
        or (tts_registry != "gemini_tts" and elevenlabs_tts is None)
        or load_vad is None
        or (turn_detector_supported and load_turn_detector is None)
    )
    if missing_factory:
        try:
            from livekit.agents import inference
            from livekit.plugins import deepgram, elevenlabs, google, silero
        except ImportError as exc:
            raise DemoConfigurationError(
                "install the pinned LiveKit Deepgram, Google, ElevenLabs, Silero, and turn-detector plugins"
            ) from exc
        deepgram_stt = deepgram_stt or deepgram.STT
        google_stt = google_stt or google.STT
        gemini_llm = gemini_llm or google.LLM
        elevenlabs_tts = elevenlabs_tts or elevenlabs.TTS
        gemini_tts = gemini_tts or google.beta.GeminiTTS
        load_vad = load_vad or silero.VAD.load
        load_turn_detector = load_turn_detector or inference.TurnDetector
    if tts_registry == "elevenlabs" and profile.language == "et" and stream_adapter is None:
        try:
            from livekit.agents import tts as livekit_tts
        except ImportError as exc:
            raise DemoConfigurationError(
                "install livekit-agents for the ElevenLabs v3 HTTP stream adapter"
            ) from exc
        stream_adapter = livekit_tts.StreamAdapter

    deepgram_key = str(environ.get("DEEPGRAM_API_KEY") or "")
    gemini_key = str(environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or "")
    eleven_key = str(environ.get("ELEVENLABS_API_KEY") or environ.get("ELEVEN_API_KEY") or "")
    if not gemini_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if stt_registry == "google":
        if not _google_credentials_available(environ):
            raise DemoConfigurationError("provider settings changed after profile validation")
    elif not deepgram_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "elevenlabs" and not eleven_key:
        raise DemoConfigurationError("provider settings changed after profile validation")

    assert gemini_llm is not None
    assert load_vad is not None

    if stt_registry == "google":
        assert google_stt is not None
        # Empirically verified against the live Google Speech-to-Text v2 API:
        # chirp_3 rejects si-LK with an HTTP 400 in both the eu and us
        # locations. chirp_2 in asia-southeast1 is the verified si-LK path --
        # it transcribed a Sinhala test phrase correctly at 0.89 confidence --
        # and asia-southeast1 is also geographically closer to Sri Lanka than
        # eu, so it is the default location too.
        stt = google_stt(
            model="chirp_2",
            languages=profile.stt_language,
            location=str(environ.get("GOOGLE_STT_LOCATION") or "asia-southeast1"),
            **_google_stt_credential_kwargs(environ),
        )
    else:
        assert deepgram_stt is not None
        stt = deepgram_stt(model="nova-3", language=profile.stt_language, api_key=deepgram_key)

    llm = gemini_llm(model=profile.llm_model, api_key=gemini_key)

    if tts_registry == "gemini_tts":
        assert gemini_tts is not None
        # Gemini-TTS has no explicit language parameter -- it infers the
        # spoken language from the input text itself, per Google's docs.
        tts = gemini_tts(model=profile.tts_model, voice_name=profile.voice_id, api_key=gemini_key)
    else:
        assert elevenlabs_tts is not None
        tts = elevenlabs_tts(
            model=profile.tts_model,
            voice_id=profile.voice_id,
            language=profile.language,
            api_key=eleven_key,
        )
        if profile.language == "et":
            assert stream_adapter is not None
            # ElevenLabs v3 supports Estonian over HTTP synthesis but rejects the
            # realtime WebSocket.  LiveKit's sentence adapter preserves streaming
            # AgentSession semantics while calling the plugin's HTTP synthesize().
            tts = stream_adapter(tts=tts)
    # LiveKit's Silero default waits 550 ms of silence. That was visible in the
    # real caller-side trace and consumed most of the latency budget before the
    # turn detector could act. Keep enough silence to reject short packet gaps
    # while letting the trained detector make the semantic decision.
    # LiveKit's streaming TurnDetector requires at least 250 ms of VAD
    # silence. Keep exactly that supported floor for turn-detector-supported
    # languages; languages without turn-detector support (Estonian, Sinhala)
    # stay on the looser VAD-only floor.
    vad = load_vad(min_silence_duration=0.25 if turn_detector_supported else 0.3)
    if turn_detector_supported:
        assert load_turn_detector is not None
        # v1-mini is LiveKit's local streaming audio EOU model. It keeps the
        # isolated sandbox self-hosted and avoids the deprecated transcript-
        # only multilingual model that measured a 1.25 s turn commit.
        turn_detection: Any = load_turn_detector(version="v1-mini")
        endpointing = "livekit_audio_turn_detector_v1_mini"
    else:
        # The pinned turn-detector plugin's local v1-mini model does not claim
        # this language (see TURN_DETECTOR_SUPPORTED_LANGUAGES above); keep
        # native AgentSession turn handling on Silero VAD + provider EOU
        # rather than running the detector uncalibrated.
        turn_detection = "vad"
        endpointing = "livekit_vad_plus_provider_eou"
    session_options = {
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "vad": vad,
        # PSTN/SIP media is already isolated into separate participant tracks.
        # LiveKit's default 3 s AEC warmup suppresses legitimate early barge-in
        # and makes the <=400 ms release gate impossible.
        "aec_warmup_duration": None,
        "turn_handling": {
            "turn_detection": turn_detection,
            "endpointing": {
                "mode": "fixed",
                "min_delay": 0.2 if turn_detector_supported else 0.35,
                # The trained model is the primary decision when supported;
                # 600 ms is a bounded fallback for uncertain short phrases.
                # Languages with no supported semantic model stay looser.
                "max_delay": 0.6 if turn_detector_supported else 1.5,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_duration": 0.15 if turn_detector_supported else 0.2,
                "min_words": 0,
                "false_interruption_timeout": 1.0,
                "resume_false_interruption": True,
            },
            "preemptive_generation": {
                "enabled": True,
                "preemptive_tts": turn_detector_supported,
                "max_speech_duration": 10.0,
                "max_retries": 2,
            },
        },
    }
    return DemoStack(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        session_options=session_options,
        safe_identity={
            "language": profile.language,
            "stt": (
                f"google/chirp_2/{profile.stt_language}"
                if stt_registry == "google"
                else f"deepgram/nova-3/{profile.stt_language}"
            ),
            "llm": f"gemini/{profile.llm_model}",
            "tts": (
                f"gemini_tts/{profile.tts_model}"
                if tts_registry == "gemini_tts"
                else f"elevenlabs/{profile.tts_model}"
            ),
            "tts_transport": (
                "chunked_http"
                if tts_registry == "gemini_tts"
                else "http_sentence_stream_adapter"
                if profile.language == "et"
                else "realtime_websocket"
            ),
            "voice_id": profile.voice_id,
            "turn_detection": endpointing,
            "barge_in": "enabled",
        },
    )

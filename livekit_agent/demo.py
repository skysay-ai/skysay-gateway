# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Standalone LiveKit demo profile and direct-provider stack.

It deliberately has no OpenPhonex control-plane dependency.  The small public
surface keeps the EN/ET switch, provider mapping, and interruption policy
testable without importing the LiveKit SDK or contacting a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping


class DemoConfigurationError(RuntimeError):
    """The isolated demo cannot safely construct its direct-provider stack."""


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
        or "gemini-2.5-flash-lite"
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


def build_demo_stack(
    profile: DemoProfile,
    *,
    environ: Mapping[str, str],
    deepgram_stt: Callable[..., Any] | None = None,
    gemini_llm: Callable[..., Any] | None = None,
    elevenlabs_tts: Callable[..., Any] | None = None,
    stream_adapter: Callable[..., Any] | None = None,
    load_vad: Callable[..., Any] | None = None,
    load_turn_detector: Callable[..., Any] | None = None,
) -> DemoStack:
    if any(
        factory is None
        for factory in (deepgram_stt, gemini_llm, elevenlabs_tts, load_vad)
    ) or (profile.language == "en" and load_turn_detector is None):
        try:
            from livekit.agents import inference
            from livekit.plugins import deepgram, elevenlabs, google, silero
        except ImportError as exc:
            raise DemoConfigurationError(
                "install the pinned LiveKit Deepgram, Google, ElevenLabs, Silero, and turn-detector plugins"
            ) from exc
        deepgram_stt = deepgram_stt or deepgram.STT
        gemini_llm = gemini_llm or google.LLM
        elevenlabs_tts = elevenlabs_tts or elevenlabs.TTS
        load_vad = load_vad or silero.VAD.load
        load_turn_detector = load_turn_detector or inference.TurnDetector
    if profile.language == "et" and stream_adapter is None:
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
    if not all((deepgram_key, gemini_key, eleven_key)):
        raise DemoConfigurationError("provider settings changed after profile validation")

    assert deepgram_stt is not None
    assert gemini_llm is not None
    assert elevenlabs_tts is not None
    assert load_vad is not None
    stt = deepgram_stt(model="nova-3", language=profile.stt_language, api_key=deepgram_key)
    llm = gemini_llm(model=profile.llm_model, api_key=gemini_key)
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
    # silence. Keep exactly that supported floor for English.
    vad = load_vad(min_silence_duration=0.25 if profile.language == "en" else 0.3)
    if profile.language == "en":
        assert load_turn_detector is not None
        # v1-mini is LiveKit's local streaming audio EOU model. It keeps the
        # isolated sandbox self-hosted and avoids the deprecated transcript-
        # only multilingual model that measured a 1.25 s turn commit.
        turn_detection: Any = load_turn_detector(version="v1-mini")
        endpointing = "livekit_audio_turn_detector_v1_mini"
    else:
        # LiveKit's multilingual semantic model does not currently claim ET;
        # keep native AgentSession turn handling on Silero VAD + Deepgram EOU.
        turn_detection = "vad"
        endpointing = "livekit_vad_plus_deepgram_eou"
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
                "min_delay": 0.2 if profile.language == "en" else 0.35,
                # The trained EN model is the primary decision; 600 ms is a
                # bounded fallback for uncertain short phrases. Estonian has
                # no supported semantic model in this build and stays looser.
                "max_delay": 0.6 if profile.language == "en" else 1.5,
            },
            "interruption": {
                "enabled": True,
                "mode": "vad",
                "min_duration": 0.15 if profile.language == "en" else 0.2,
                "min_words": 0,
                "false_interruption_timeout": 1.0,
                "resume_false_interruption": True,
            },
            "preemptive_generation": {
                "enabled": True,
                "preemptive_tts": profile.language == "en",
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
            "stt": f"deepgram/nova-3/{profile.stt_language}",
            "llm": f"gemini/{profile.llm_model}",
            "tts": f"elevenlabs/{profile.tts_model}",
            "tts_transport": (
                "http_sentence_stream_adapter"
                if profile.language == "et"
                else "realtime_websocket"
            ),
            "voice_id": profile.voice_id,
            "turn_detection": endpointing,
            "barge_in": "enabled",
        },
    )

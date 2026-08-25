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

try:  # package import in tests; flat /app import in the worker image
    from .native_tts_usage import Utf8ByteCounter
    from .soniox_usage import SonioxUsageTracker
except ImportError:  # pragma: no cover - exercised by container smoke
    from native_tts_usage import Utf8ByteCounter
    from soniox_usage import SonioxUsageTracker

try:  # package import in tests; flat /app import in the worker image
    from .capabilities.registry import load_registry
except ImportError:  # pragma: no cover - exercised by container smoke
    from capabilities.registry import load_registry


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

# These are the pre-registry adapter paths. Their exact historical profile is
# validated by ``production.profile_from_realtime_config`` before this worker
# builder runs, and an empty capability revision is intentionally durable for
# existing EN/ET/SI agents. New providers must always pass through the
# immutable artifact below.
_LEGACY_STT_REGISTRIES = frozenset({"deepgram", "google"})
_LEGACY_TTS_REGISTRIES = frozenset({"elevenlabs", "gemini_tts"})

# ElevenLabs answers its realtime WebSocket with a 403 for these models, which
# LiveKit surfaces as a non-retryable APIStatusError: the turn loop dies while
# STT and the LLM keep running, so the call connects, transcribes, and is
# silent. LiveKit's StreamAdapter preserves streaming AgentSession semantics
# over the plugin's HTTP synthesize(), which these models do serve.
#
# This is a property of the model. It was first met on Estonian -- the one
# language whose only voice is eleven_v3 -- and the fallback was written as a
# language test, which left the same model mute in every other language.
#
# The control plane keeps its own copy in
# ``telephony_ai.services.call_admission_profile`` because the worker ships in a
# separately deployed container and the two cannot import each other. The sets
# are held equal by test_sch236_livekit_demo.
NON_STREAMING_TTS_MODELS = frozenset({"eleven_v3"})


def tts_needs_http_stream_adapter(*, tts_registry: str, tts_model: str, language: str) -> bool:
    """Whether this stack must synthesize over HTTP instead of the realtime socket.

    Estonian stays in the condition even though its only approved voice is
    already a non-streaming model: the guard that keeps it there lives in
    another module, and losing the wrap here would be a silent line rather than
    a refusal.
    """
    if tts_registry != "elevenlabs":
        return False
    return tts_model.strip() in NON_STREAMING_TTS_MODELS or language == "et"


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
    # New registry-backed adapters carry their exact model and artifact
    # revision. Historical profiles keep these empty and resolve their existing
    # pinned model below, preserving agents created before this registry.
    stt_model: str = ""
    capability_revision: str = ""


@dataclass(frozen=True)
class DemoStack:
    stt: Any
    llm: Any
    tts: Any
    vad: Any
    session_options: dict[str, Any]
    safe_identity: dict[str, str]
    tts_utf8_byte_counter: Utf8ByteCounter | None = None
    # Kept only in the worker process until terminal settlement. It carries
    # provider keys in-memory to fetch the exact Soniox usage-log dimensions;
    # its repr/payload deliberately never carries those keys or references.
    soniox_usage_tracker: SonioxUsageTracker | None = None


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
    """Resolve GCP credentials for LiveKit's Google STT plugin.

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

    NOTE this is a distinct credential from ``GEMINI_API_KEY`` -- Cloud STT is a
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


def _soniox_component_key(environ: Mapping[str, str], component: str) -> str:
    """Resolve managed or explicitly leased Soniox credentials without crossover.

    Soniox STT and TTS normally share ``SONIOX_API_KEY`` for a managed call.
    A BYOK overlay marks only the selected component and supplies a scoped key;
    when that marker exists, accepting the generic key would charge the wrong
    account if the scoped lease were absent or corrupt.
    """
    scoped_variable, marker = {
        "stt": ("SONIOX_STT_API_KEY", "OPENPHONEX_BYOK_SONIOX_STT"),
        "tts": ("SONIOX_TTS_API_KEY", "OPENPHONEX_BYOK_SONIOX_TTS"),
    }.get(component, ("", ""))
    if not scoped_variable or not marker:
        raise DemoConfigurationError("Soniox component identity is invalid")
    marker_value = environ.get(marker)
    if marker_value is not None:
        if marker_value != "1":
            raise DemoConfigurationError("Soniox BYOK credential marker is invalid")
        key = str(environ.get(scoped_variable) or "")
        if not key:
            raise DemoConfigurationError("Soniox BYOK credential is missing")
        return key
    if str(environ.get(scoped_variable) or ""):
        raise DemoConfigurationError("Soniox BYOK credential marker is invalid")
    return str(environ.get("SONIOX_API_KEY") or "")


def _fish_audio_tts_key(environ: Mapping[str, str]) -> str:
    """Resolve a Fish TTS lease without falling back to its managed key."""

    marker = environ.get("OPENPHONEX_BYOK_FISHAUDIO_TTS")
    if marker is not None:
        if marker != "1":
            raise DemoConfigurationError("Fish Audio BYOK credential marker is invalid")
        key = str(environ.get("FISH_AUDIO_TTS_API_KEY") or "")
        if not key:
            raise DemoConfigurationError("Fish Audio BYOK credential is missing")
        return key
    if str(environ.get("FISH_AUDIO_TTS_API_KEY") or ""):
        raise DemoConfigurationError("Fish Audio BYOK credential marker is invalid")
    return str(environ.get("FISH_AUDIO_API_KEY") or "")


def _gemini_tts_key(environ: Mapping[str, str]) -> str:
    """Resolve a platform Gemini TTS key without crossing an LLM BYOK key.

    Gemini TTS is platform-managed for the first cohort. A scoped marker is
    still needed when a selected Gemini LLM BYOK lease replaces the generic
    key: TTS must keep using the platform account rather than silently using
    that LLM customer's key. A stale scoped variable is an error rather than a
    fallback to whichever generic Gemini key happens to be in the process.
    """
    platform_marker = environ.get("OPENPHONEX_PLATFORM_GEMINI_TTS")
    if platform_marker is not None:
        if platform_marker != "1":
            raise DemoConfigurationError("Gemini TTS credential marker is invalid")
        key = str(environ.get("GEMINI_TTS_API_KEY") or "")
        if not key:
            raise DemoConfigurationError("Gemini TTS platform credential is missing")
        return key
    if str(environ.get("GEMINI_TTS_API_KEY") or ""):
        raise DemoConfigurationError("Gemini TTS credential marker is invalid")
    return str(environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or "")


def build_demo_stack(
    profile: DemoProfile,
    *,
    environ: Mapping[str, str],
    deepgram_stt: Callable[..., Any] | None = None,
    google_stt: Callable[..., Any] | None = None,
    gemini_llm: Callable[..., Any] | None = None,
    elevenlabs_tts: Callable[..., Any] | None = None,
    gemini_tts: Callable[..., Any] | None = None,
    soniox_stt: Callable[..., Any] | None = None,
    soniox_stt_options: Callable[..., Any] | None = None,
    soniox_tts: Callable[..., Any] | None = None,
    fish_audio_tts: Callable[..., Any] | None = None,
    stream_adapter: Callable[..., Any] | None = None,
    load_vad: Callable[..., Any] | None = None,
    load_turn_detector: Callable[..., Any] | None = None,
    soniox_usage_reference: str = "",
    require_soniox_usage_reference: bool = True,
    # A caller that is not the production settlement path must opt out
    # explicitly. New hosted call paths therefore cannot accidentally turn on
    # Gemini TTS without native prompt/audio-token evidence.
    require_gemini_tts_usage_metadata: bool = True,
) -> DemoStack:
    stt_registry = profile.stt_registry
    tts_registry = profile.tts_registry
    turn_detector_supported = profile.language in TURN_DETECTOR_SUPPORTED_LANGUAGES

    # New adapter identities are never another worker-local list.  The same
    # immutable capability artifact that admitted the control-plane selection
    # authorizes Soniox/Fish here; the two small fallback sets are only the
    # historical Deepgram/Google and ElevenLabs/Gemini paths that pre-date it.
    registry = load_registry()
    if profile.language not in registry.runtime_languages:
        raise DemoConfigurationError("selected language is absent from the worker capability registry")
    stt_capability = registry.capability("stt", stt_registry, profile.stt_model)
    tts_capability = registry.capability("tts", tts_registry, profile.tts_model)
    registry_backed_stt = stt_registry not in _LEGACY_STT_REGISTRIES
    registry_backed_tts = tts_registry not in _LEGACY_TTS_REGISTRIES
    if registry_backed_stt or registry_backed_tts:
        if profile.capability_revision != registry.revision:
            raise DemoConfigurationError("selected capability revision does not match this worker image")
    elif profile.capability_revision and profile.capability_revision != registry.revision:
        raise DemoConfigurationError("selected capability revision does not match this worker image")
    if registry_backed_stt:
        if stt_capability is None:
            raise DemoConfigurationError("selected STT provider has no production adapter")
        if not stt_capability["runnable"] or profile.language not in stt_capability["languages"]:
            raise DemoConfigurationError("selected STT capability is unavailable")
    if registry_backed_tts:
        if tts_capability is None:
            raise DemoConfigurationError("selected TTS provider has no production adapter")
        if not tts_capability["runnable"] or profile.language not in tts_capability["languages"]:
            raise DemoConfigurationError("selected TTS capability is unavailable")
    supplied_soniox_tts = soniox_tts is not None
    supplied_gemini_tts = gemini_tts is not None
    missing_factory = (
        (stt_registry == "google" and google_stt is None)
        or (stt_registry == "deepgram" and deepgram_stt is None)
        or (stt_registry == "soniox" and (soniox_stt is None or soniox_stt_options is None))
        or gemini_llm is None
        or (tts_registry == "gemini_tts" and gemini_tts is None)
        or (tts_registry == "elevenlabs" and elevenlabs_tts is None)
        or (tts_registry == "soniox" and soniox_tts is None)
        or (tts_registry == "fishaudio" and fish_audio_tts is None)
        or load_vad is None
        or (turn_detector_supported and load_turn_detector is None)
    )
    if missing_factory:
        try:
            from livekit.agents import inference
            from livekit.plugins import deepgram, elevenlabs, google, silero
            needs_soniox_plugin = (
                (stt_registry == "soniox" and (soniox_stt is None or soniox_stt_options is None))
                or (tts_registry == "soniox" and soniox_tts is None)
            )
            if needs_soniox_plugin:
                from livekit.plugins import soniox
            if tts_registry == "fishaudio" and fish_audio_tts is None:
                try:
                    from .fish_audio_tts import TTS as MeteredFishAudioTTS
                except ImportError:  # flat worker import
                    from fish_audio_tts import TTS as MeteredFishAudioTTS
            if tts_registry == "gemini_tts" and gemini_tts is None:
                try:
                    from .gemini_tts import TTS as MeteredGeminiTTS
                except ImportError:  # pragma: no cover - flat worker import
                    from gemini_tts import TTS as MeteredGeminiTTS
        except ImportError as exc:
            raise DemoConfigurationError(
                "install the pinned LiveKit STT, TTS, VAD, and turn-detector plugins"
            ) from exc
        deepgram_stt = deepgram_stt or deepgram.STT
        google_stt = google_stt or google.STT
        gemini_llm = gemini_llm or google.LLM
        elevenlabs_tts = elevenlabs_tts or elevenlabs.TTS
        if tts_registry == "gemini_tts":
            gemini_tts = gemini_tts or MeteredGeminiTTS
        if stt_registry == "soniox":
            soniox_stt = soniox_stt or soniox.STT
            soniox_stt_options = soniox_stt_options or soniox.STTOptions
        if tts_registry == "soniox":
            soniox_tts = soniox_tts or soniox.TTS
        if tts_registry == "fishaudio":
            fish_audio_tts = fish_audio_tts or MeteredFishAudioTTS
        load_vad = load_vad or silero.VAD.load
        load_turn_detector = load_turn_detector or inference.TurnDetector
    needs_http_tts = tts_needs_http_stream_adapter(
        tts_registry=tts_registry, tts_model=profile.tts_model, language=profile.language
    )
    if needs_http_tts and stream_adapter is None:
        try:
            from livekit.agents import tts as livekit_tts
        except ImportError as exc:
            raise DemoConfigurationError(
                "install livekit-agents for the ElevenLabs v3 HTTP stream adapter"
            ) from exc
        stream_adapter = livekit_tts.StreamAdapter

    deepgram_key = str(environ.get("DEEPGRAM_API_KEY") or "")
    gemini_key = str(environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or "")
    gemini_tts_key = _gemini_tts_key(environ) if tts_registry == "gemini_tts" else ""
    eleven_key = str(environ.get("ELEVENLABS_API_KEY") or environ.get("ELEVEN_API_KEY") or "")
    # The per-call environment can carry a scoped lease for a different
    # component. Resolve and validate that lease only if this profile actually
    # selects Soniox; unrelated historical stacks must not be rejected by a
    # stale/inapplicable Soniox marker.
    soniox_stt_key = _soniox_component_key(environ, "stt") if stt_registry == "soniox" else ""
    soniox_tts_key = _soniox_component_key(environ, "tts") if tts_registry == "soniox" else ""
    fish_key = _fish_audio_tts_key(environ) if tts_registry == "fishaudio" else ""
    if not gemini_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if stt_registry == "google":
        if not _google_credentials_available(environ):
            raise DemoConfigurationError("provider settings changed after profile validation")
    elif stt_registry == "deepgram" and not deepgram_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    elif stt_registry == "soniox" and not soniox_stt_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "elevenlabs" and not eleven_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "soniox" and not soniox_tts_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "fishaudio" and not fish_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "gemini_tts" and not gemini_tts_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if require_gemini_tts_usage_metadata and tts_registry == "gemini_tts" and supplied_gemini_tts:
        raise DemoConfigurationError("paid Gemini TTS requires the native usage adapter")

    # The provider logs exact token quantities only if both realtime adapters
    # receive opaque client references. Browser demos do not have a paid call
    # id and intentionally leave this absent; production passes the call id
    # before media admission. The tracker transforms it one-way and retains
    # neither raw identifiers nor key material in its output.
    soniox_components: dict[str, tuple[str, str]] = {}
    if stt_registry == "soniox":
        soniox_components["stt"] = (soniox_stt_key, profile.stt_model)
    if tts_registry == "soniox":
        soniox_components["tts"] = (soniox_tts_key, profile.tts_model)
    if soniox_components and require_soniox_usage_reference and not str(soniox_usage_reference or "").strip():
        # New Soniox stacks always fail closed before media without opaque
        # token-correlation evidence. The two non-chargeable demo entrypoints
        # opt out explicitly; production retains the default so a future call
        # path cannot accidentally construct an unmetered Soniox adapter.
        raise DemoConfigurationError("Soniox usage correlation is required before media admission")
    soniox_usage_tracker = (
        SonioxUsageTracker(
            call_reference=soniox_usage_reference,
            components=soniox_components,
        )
        if soniox_components and str(soniox_usage_reference or "").strip()
        else None
    )
    if soniox_usage_tracker is not None and tts_registry == "soniox" and not supplied_soniox_tts:
        # LiveKit's pinned 1.6.6 Soniox STT accepts client_reference_id, but
        # its TTS constructor does not. This narrow subclass keeps the
        # upstream sentence buffering, idle timeout, cancellation and stream
        # rotation while adding just that provider-owned accounting field.
        try:
            from .soniox_tts import TTS as MeteredSonioxTTS
        except ImportError:  # pragma: no cover - flat worker import
            from soniox_tts import TTS as MeteredSonioxTTS
        soniox_tts = MeteredSonioxTTS

    assert gemini_llm is not None
    assert load_vad is not None

    if stt_registry == "google":
        assert google_stt is not None
        google_model = profile.stt_model or "chirp_2"
        google_kwargs: dict[str, Any] = {
            "model": google_model,
            "languages": profile.stt_language,
            **_google_stt_credential_kwargs(environ),
        }
        # LiveKit routes command_and_search through Speech-to-Text v1, whose
        # streaming endpoint is global. Chirp 2 uses v2 and needs the regional
        # recognizer location. Sending a v1 model to the v2 regional endpoint
        # is itself a connection failure, so the distinction is explicit.
        if google_model == "chirp_2":
            google_kwargs["location"] = str(
                environ.get("GOOGLE_STT_LOCATION") or "asia-southeast1"
            )
        stt = google_stt(**google_kwargs)
    elif stt_registry == "deepgram":
        assert deepgram_stt is not None
        stt = deepgram_stt(model="nova-3", language=profile.stt_language, api_key=deepgram_key)
    elif stt_registry == "soniox":
        if profile.stt_model != "stt-rt-v5":
            raise DemoConfigurationError("Soniox STT must use stt-rt-v5")
        assert soniox_stt is not None
        assert soniox_stt_options is not None
        # Language hints are strict deliberately: Tamil is not silently
        # recognized as a nearby language, and the provider owns endpointing.
        soniox_options_kwargs: dict[str, Any] = {
            "model": "stt-rt-v5",
            "language_hints": [profile.stt_language],
            "language_hints_strict": True,
            "enable_language_identification": True,
            "max_endpoint_delay_ms": 1200,
        }
        if soniox_usage_tracker is not None:
            soniox_options_kwargs["client_reference_id"] = soniox_usage_tracker.reference_for("stt")
        stt = soniox_stt(
            api_key=soniox_stt_key,
            params=soniox_stt_options(**soniox_options_kwargs),
        )
    else:  # Defensive guard if a future registry row is added without a builder.
        raise DemoConfigurationError("selected STT provider has no production adapter")

    llm = gemini_llm(model=profile.llm_model, api_key=gemini_key)

    tts_utf8_byte_counter: Utf8ByteCounter | None = None
    if tts_registry == "gemini_tts":
        assert gemini_tts is not None
        # Gemini-TTS has no explicit language parameter -- it infers the
        # spoken language from the input text itself, per Google's docs.
        gemini_tts_kwargs: dict[str, Any] = {
            "model": profile.tts_model,
            "voice_name": profile.voice_id,
            "api_key": gemini_tts_key,
        }
        if not supplied_gemini_tts:
            gemini_tts_kwargs["require_usage_metadata"] = require_gemini_tts_usage_metadata
        tts = gemini_tts(**gemini_tts_kwargs)
    elif tts_registry == "elevenlabs":
        assert elevenlabs_tts is not None
        tts = elevenlabs_tts(
            model=profile.tts_model,
            voice_id=profile.voice_id,
            language=profile.language,
            api_key=eleven_key,
        )
        if needs_http_tts:
            assert stream_adapter is not None
            # See NON_STREAMING_TTS_MODELS: these models serve HTTP synthesis and
            # 403 the realtime WebSocket.  LiveKit's sentence adapter preserves
            # streaming AgentSession semantics while calling the plugin's HTTP
            # synthesize().
            tts = stream_adapter(tts=tts)
    elif tts_registry == "soniox":
        if profile.tts_model != "tts-rt-v2":
            raise DemoConfigurationError("Soniox TTS must use tts-rt-v2")
        assert soniox_tts is not None
        # Pin the v2 model; the upstream plugin default is an obsolete preview.
        tts = soniox_tts(
            model="tts-rt-v2",
            language=profile.language,
            voice=profile.voice_id,
            audio_format="pcm_s16le",
            sample_rate=24000,
            api_key=soniox_tts_key,
            **(
                {"client_reference_id": soniox_usage_tracker.reference_for("tts")}
                if soniox_usage_tracker is not None
                else {}
            ),
        )
    elif tts_registry == "fishaudio":
        if profile.tts_model != "s2-pro":
            raise DemoConfigurationError("Fish Audio TTS must use paid s2-pro")
        assert fish_audio_tts is not None
        tts_utf8_byte_counter = Utf8ByteCounter()
        tts = fish_audio_tts(
            model="s2-pro",
            voice_id=profile.voice_id,
            output_format="wav",
            latency_mode="balanced",
            chunk_length=100,
            api_key=fish_key,
            utf8_byte_counter=tts_utf8_byte_counter,
        )
    else:  # Defensive guard if a future registry row is added without a builder.
        raise DemoConfigurationError("selected TTS provider has no production adapter")
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
                f"google/{profile.stt_model or 'chirp_2'}/{profile.stt_language}"
                if stt_registry == "google"
                else f"soniox/stt-rt-v5/{profile.stt_language}"
                if stt_registry == "soniox"
                else f"deepgram/nova-3/{profile.stt_language}"
            ),
            "llm": f"gemini/{profile.llm_model}",
            "tts": f"{tts_registry}/{profile.tts_model}",
            "tts_transport": (
                "chunked_http"
                if tts_registry == "gemini_tts"
                else "http_sentence_stream_adapter"
                if needs_http_tts
                else "realtime_websocket"
            ),
            "voice_id": profile.voice_id,
            "turn_detection": endpointing,
            "barge_in": "enabled",
        },
        tts_utf8_byte_counter=tts_utf8_byte_counter,
        soniox_usage_tracker=soniox_usage_tracker,
    )

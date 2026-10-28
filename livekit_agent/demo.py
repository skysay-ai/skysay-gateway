# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Standalone LiveKit demo profile and direct-provider stack.

It deliberately has no Skysay control-plane dependency.  The small public
surface keeps provider mapping and interruption policy testable without
importing the LiveKit SDK or contacting a provider. ``load_demo_profile``
(the internal env-var-driven qualification demo) only ever builds an EN or ET
``DemoProfile``; ``build_demo_stack`` (shared with the real production path)
additionally builds the SI (si-LK / Sinhala) stack when given a profile from
``production.profile_from_realtime_config`` instead.
"""

from __future__ import annotations

try:
    from .voice_configuration import refuse_freeform_workflow
except ImportError:
    from voice_configuration import refuse_freeform_workflow

import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

try:  # package import in tests; flat /app import in the pinned worker image
    from .prewarm import pinned_silero_vad_loader
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from prewarm import pinned_silero_vad_loader

try:  # package import in tests; flat /app import in the worker image
    from .native_tts_usage import NativeTTSUnitMeter, RequestBilling, TTSRequestActivity
    from .soniox_usage import SonioxUsageTracker
except ImportError:  # pragma: no cover - exercised by container smoke
    from native_tts_usage import NativeTTSUnitMeter, RequestBilling, TTSRequestActivity
    from soniox_usage import SonioxUsageTracker

try:  # package import in tests; flat /app import in the worker image
    from .capabilities.registry import load_registry
except ImportError:  # pragma: no cover - exercised by container smoke
    from capabilities.registry import load_registry

try:  # package import in tests; flat /app import in the worker image
    from .transcription_stall import TRANSCRIPTION_TIMEOUT_SECONDS
except ImportError:  # pragma: no cover - exercised by container smoke
    from transcription_stall import TRANSCRIPTION_TIMEOUT_SECONDS

try:  # package import in tests; flat /app import in the worker image
    from .capabilities.delivery import DeliveryTableError, resolve_delivery
except ImportError:  # pragma: no cover - exercised by container smoke
    from capabilities.delivery import DeliveryTableError, resolve_delivery


class DemoConfigurationError(RuntimeError):
    """The isolated demo cannot safely construct its direct-provider stack."""


# Languages the pinned turn-detector plugin (livekit-agents 1.7.1,
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
# builder runs. New providers must pass the candidate catalogue's concrete
# provider/model/language checks below.
_LEGACY_STT_REGISTRIES = frozenset({"deepgram", "google"})
_LEGACY_TTS_REGISTRIES = frozenset({"elevenlabs", "gemini_tts"})

# ``eleven_v3`` used to need an HTTP fallback here.  ElevenLabs answered the
# realtime multi-stream-input WebSocket with a 403 for it, LiveKit surfaced
# that as a non-retryable APIStatusError, and the turn loop died while STT and
# the LLM kept running -- the call connected, transcribed, and stayed silent.
# The workaround wrapped the plugin in LiveKit's ``StreamAdapter``, which
# preserves streaming AgentSession semantics over the plugin's HTTP
# ``synthesize()``.
#
# livekit-plugins-elevenlabs 1.7.1 routed ``eleven_v3`` and
# ``eleven_v3_conversational`` through ElevenLabs' text-to-dialogue API on both
# the chunked and the streaming path (``is_dialogue_model`` selects
# ``_DialogueConnection``), and the wrap was briefly removed. It is BACK, for
# every ``eleven_v3*`` model in every language, because that dialogue
# websocket clips the last word of each turn: see
# ``eleven_stream_transport_clips_tail`` below, which is what a call uses today.
# Measured against this pinned plugin on one Estonian sentence and our own
# voice: three consecutive turns on one reused connection, first audio 1.064 s
# cold and 0.897/0.939 s warm, no 403.  The removed adapter had to buffer a
# whole sentence out of the LLM before it could call ``synthesize()`` at all,
# which the native transport no longer needs.
#
# The control plane flagged ``eleven_v3`` outside Estonian
# (``TTS_CANNOT_STREAM``) until this worker was promoted and proven. The flag
# was dropped on 2026-09-24: this worker synthesizes every ``eleven_v3*``
# model over chunked HTTP (``eleven_stream_transport_clips_tail`` below) in
# every language, and English and Estonian calls on it completed with every
# reply audible.

#: ElevenLabs models the pinned plugin routes through the text-to-dialogue
#: multi-stream websocket.  Measured 2026-09-02 against that transport: it
#: truncates the final word's low-energy decay on nearly every utterance --
#: terminal-30 ms RMS ~2800-3300 across repeated same-text runs, where every
#: other transport in the catalog (this model over chunked HTTP or REST, the
#: regular flash websocket, and every non-ElevenLabs provider) decays below
#: ~500.  The caller hears the last word clipped on every agent turn.
#:
#: Keyed on the v3 prefix because all of those models use the affected
#: transport. SDK 1.8.6 also routes v4 to dialogue, but the measured v4 Turbo
#: tail is not clipped, so v4 must not inherit this quarantine. At the
#: 1.7.1 pin only ``eleven_v3`` and ``eleven_v3_conversational`` are declared,
#: but an exact-match table would leave the NEXT ``eleven_v3*`` id unwrapped
#: while the plugin still routed it onto the clipping websocket.  Admission
#: does not catch that: ``owns_component_provider("tts", "elevenlabs")`` is
#: False and en/et are legacy-runtime languages, so
#: ``profile_from_realtime_config`` admits any model string for them on this
#: registry with no catalog row -- this predicate is the only gate on that
#: path.  (si is the third legacy-runtime language but never reaches it: the
#: Sinhala policy gate rejects ElevenLabs outright, ahead of the registry
#: check.)
#:
#: Accepted cost: a future non-dialogue ``eleven_v3*`` model would be wrapped
#: unnecessarily and lose word-level timings.  That is one line to undo here;
#: shipping the clip is what the founder heard.
_ELEVEN_DIALOGUE_MODEL_PREFIX = "eleven_v3"
#: ElevenLabs rejects this id on the Flash text-to-speech websocket (handshake
#: 400 from plugin 1.8.1). The text-to-dialogue socket accepts it. Measured
#: 2026-09-29 on that socket, voice XrExE9yKIg1WjnnlVkGX, one English sentence:
#: terminal-30 ms decay ratio 0.066, so the chunked-HTTP quarantine that wraps
#: ``eleven_v3*`` does not apply. SDK 1.8.6 routes v4 natively but still
#: drops similarity_boost; the worker restores that setting only.
ELEVEN_V4_TURBO_MODEL = "eleven_v4_turbo"


def eleven_dialogue_settings_for_wire(model: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Voice settings the dialogue socket is asked to honor for this model.

    v3's socket honors stability only, which is all the pinned plugin sends.
    v4 turbo accepts similarity_boost on the same socket; measured by sending
    both and receiving audio with no error.
    """

    keys = ("stability", "similarity_boost") if model == ELEVEN_V4_TURBO_MODEL else ("stability",)
    return {key: settings[key] for key in keys if key in settings and isinstance(settings[key], (int, float))}


def install_eleven_v4_turbo_dialogue_route() -> None:
    """Retain v4 similarity on the dialogue route supplied by SDK 1.8.6."""

    try:
        import livekit.plugins.elevenlabs.tts as el_tts
    except ImportError as exc:
        raise DemoConfigurationError("install the pinned LiveKit TTS plugins") from exc
    if getattr(el_tts, "_skysay_v4_turbo_routed", False):
        return
    original_init = el_tts._build_dialogue_context_init_packet

    def init_packet(opts: Any, *, context_id: str) -> dict[str, Any]:
        packet = original_init(opts, context_id=context_id)
        if getattr(opts, "model", None) != ELEVEN_V4_TURBO_MODEL:
            return packet
        voice_settings = getattr(opts, "voice_settings", None)
        if voice_settings is None:
            return packet
        raw = voice_settings if isinstance(voice_settings, Mapping) else {
            "stability": getattr(voice_settings, "stability", None),
            "similarity_boost": getattr(voice_settings, "similarity_boost", None),
        }
        merged = dict(packet.get("voice_settings") or {})
        merged.update(eleven_dialogue_settings_for_wire(ELEVEN_V4_TURBO_MODEL, raw))
        if merged:
            packet["voice_settings"] = merged
        return packet

    el_tts._build_dialogue_context_init_packet = init_packet
    el_tts._skysay_v4_turbo_routed = True


def eleven_stream_transport_clips_tail(tts_model: str) -> bool:
    """Whether this ElevenLabs model's native streaming transport is quarantined.

    Quarantined models are wrapped in LiveKit's ``StreamAdapter``, which keeps
    streaming ``AgentSession`` semantics over the model's chunked HTTP
    synthesis.  That wrap costs nothing on the latency it was once removed to
    save: without ``auto_mode`` the dialogue websocket buffers the whole reply
    and only synthesizes on the final flush, while the adapter starts at the
    first sentence boundary -- measured first-audio p50 855 ms (adapter)
    vs 1004 ms (dialogue websocket) on identical paced text, alongside the
    tail fix (terminal-30 ms RMS median 60 vs 2836).
    """

    return tts_model.startswith(_ELEVEN_DIALOGUE_MODEL_PREFIX)


@functools.cache
def _quarantined_eleven_stream_adapter() -> type:
    """LiveKit's ``StreamAdapter`` for a quarantined model, without the plugin's prewarm.

    The SDK calls ``tts.prewarm()`` when an activity starts and again after
    every TTS replacement (``Agent.update_options``, which the language switch
    uses). ``StreamAdapter.prewarm()`` forwards to the plugin, and the plugin's
    prewarm for a dialogue model opens the text-to-dialogue websocket and
    reopens it whenever the server idles it out, for the rest of the call.
    Nothing synthesizes on that socket here. Each upgrade takes the pooled
    HTTPS keep-alive connection, so the next turn's first chunked request paid
    a fresh TCP+TLS handshake (first-audio p50 1244 -> 859 ms on the local
    stack). ``StreamAdapter.aclose()`` does not close the plugin either, so
    each language switch started one more reconnect loop nothing stopped.

    The label stays the SDK's own: LiveKit derives it from the concrete class
    and reports it in TTS errors and metrics. Built on first use, because this
    module must import without the SDK.
    """

    from livekit.agents.tts import StreamAdapter

    label = f"{StreamAdapter.__module__}.{StreamAdapter.__name__}"

    class QuarantinedElevenStreamAdapter(StreamAdapter):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._label = label

        def prewarm(self) -> None:
            pass

    return QuarantinedElevenStreamAdapter


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
    # fields above (load_demo_profile, ...).
    stt_registry: str = "deepgram"
    tts_registry: str = "elevenlabs"
    # Registry-backed adapters carry their exact model. Historical profiles
    # may leave it empty and resolve their existing pinned model below.
    stt_model: str = ""
    # Legacy profiles have an exact first_message. A selected language profile
    # without one uses the native session/model opening path instead.
    opening_mode: str = "literal"
    # How the voice performs its speech: one of the packaged delivery table's
    # profiles, or empty for the provider's own defaults (every agent that
    # never chose one). Resolved per provider/model at build time.
    delivery: str = ""
    # Every language this agent is configured for. ``language`` above is the
    # one this call speaks; this is the set a transcriber that can listen for
    # several at once is given. Empty for every caller that predates the
    # field, which is what keeps a one-language call building exactly as it
    # always did.
    languages: tuple[str, ...] = ()


_ADMISSION_AUTHORITY = object()


@dataclass(frozen=True, repr=False)
class _InitialStackAdmission:
    profile: DemoProfile
    worker_revision: str
    owner: object
    authority: object


def _admission_identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, repr=False)
class CallVoiceAdmission:
    """Worker-local authority for exact targets admitted by one running call.

    Opaque owner identity excludes another job, including another job with the
    same public call ID. Content hashes bind immutable profile/config snapshots;
    this object is never accepted from realtime-config JSON or serialized.
    """

    _owner: object
    _worker_revision: str
    _initial_identity: str
    _targets: tuple[tuple[str, str], ...]
    _simultaneous_stt: str
    _authority: object

    def __getstate__(self) -> object:
        raise TypeError("voice admission is worker-local and not serializable")

    def validate(
        self, profile: DemoProfile, *, registry: Any, call_owner: object | None,
        target_config: Mapping[str, Any] | None, simultaneous_stt: Mapping[str, Any] | None,
    ) -> None:
        if (
            self._authority is not _ADMISSION_AUTHORITY
            or self._owner is not call_owner
            or self._worker_revision != registry.revision
            or target_config is None
            or simultaneous_stt is None
            or self._simultaneous_stt != _admission_identity(simultaneous_stt)
            or dict(self._targets).get(profile.language) != _admission_identity(
                {"profile": asdict(profile), "config": target_config}
            )
        ):
            raise DemoConfigurationError("TTS target is outside this call's admitted profiles")


def validate_profile_capabilities(
    profile: DemoProfile, *, registry: Any,
    admission_context: CallVoiceAdmission | None = None,
    call_owner: object | None = None,
    target_config: Mapping[str, Any] | None = None,
    simultaneous_stt: Mapping[str, Any] | None = None,
) -> None:
    """Validate concrete choices at construction and bind any in-call switch."""
    if profile.language not in registry.runtime_languages:
        raise DemoConfigurationError("selected language is absent from the worker capability registry")
    registry_backed_stt = profile.stt_registry not in _LEGACY_STT_REGISTRIES
    registry_backed_tts = profile.tts_registry not in _LEGACY_TTS_REGISTRIES
    if admission_context is not None:
        if not isinstance(admission_context, CallVoiceAdmission):
            raise DemoConfigurationError("voice admission must be worker-local")
        admission_context.validate(
            profile, registry=registry, call_owner=call_owner,
            target_config=target_config, simultaneous_stt=simultaneous_stt,
        )
    for component, adapter, model, required in (
        ("stt", profile.stt_registry, profile.stt_model, registry_backed_stt),
        ("tts", profile.tts_registry, profile.tts_model, registry_backed_tts),
    ):
        if not required:
            continue
        capability = registry.capability(
            component, registry.vendor_provider(component, adapter, model), model
        )
        if capability is None:
            raise DemoConfigurationError(f"selected {component.upper()} provider has no production adapter")
        if not capability["runnable"] or profile.language not in capability["languages"]:
            raise DemoConfigurationError(f"selected {component.upper()} capability is unavailable")


def admit_voice_targets(
    initial_stack: Any, *, call_owner: object,
    allowed_profiles: Mapping[str, Any], simultaneous_stt: Mapping[str, Any],
) -> CallVoiceAdmission:
    """Mint after a successful initial stack and before admitting caller media."""
    initial = getattr(initial_stack, "_admission", None)
    registry = load_registry()
    if (
        not isinstance(initial, _InitialStackAdmission)
        or initial.authority is not _ADMISSION_AUTHORITY
        or initial.owner is not call_owner
        or initial.worker_revision != registry.revision
        or not allowed_profiles
    ):
        raise DemoConfigurationError("voice admission requires this call's successful initial stack")
    # Targets were parsed and checked against frozen billing before the initial
    # construction. Check the time-sensitive gate again before minting; later
    # switches retain exactly these contents, never newly fetched agent state.
    identities: list[tuple[str, str]] = []
    for language, target in sorted(allowed_profiles.items()):
        if language != target.profile.language:
            raise DemoConfigurationError("admitted target language is inconsistent")
        validate_profile_capabilities(target.profile, registry=registry)
        identities.append((language, _admission_identity({
            "profile": asdict(target.profile), "config": target.config,
        })))
    if initial.profile.language not in allowed_profiles:
        raise DemoConfigurationError("opening profile is outside the admitted targets")
    opening_target = allowed_profiles[initial.profile.language].profile
    # Call-specific instructions/opening overrides are intentionally applied
    # only to the active top-level config by calls_routing.
    # The same resolver keeps current top-level voice_polish when a queued
    # call owns an older language publication. Its initial delivery was
    # independently validated by construction; replacements still bind each
    # exact frozen target's delivery, never this newer mutable selection.
    # Preserve that established session policy,
    # while refusing a different provider/model/voice under the same language.
    if replace(
        initial.profile, instructions=opening_target.instructions,
        first_message=opening_target.first_message, opening_mode=opening_target.opening_mode,
        delivery=opening_target.delivery,
    ) != opening_target:
        raise DemoConfigurationError("opening component identity differs from the successful initial stack")
    return CallVoiceAdmission(
        call_owner, registry.revision, _admission_identity(asdict(initial.profile)), tuple(identities),
        _admission_identity(simultaneous_stt), _ADMISSION_AUTHORITY,
    )


@dataclass(frozen=True)
class DemoStack:
    stt: Any
    llm: Any
    tts: Any
    vad: Any
    session_options: dict[str, Any]
    safe_identity: dict[str, str]
    #: The constructor arguments the transcriber was built with, no credential (``construct_stt_component``'s
    #: ``observe``): what navigation attests for the call's own transcript source.
    stt_construction: dict[str, Any] = field(default_factory=dict)
    #: The one native TTS meter (``native_tts_usage``) whose unit and request
    #: billing are the TTS row's data; ``None`` only for Soniox's own tracker.
    tts_unit_meter: NativeTTSUnitMeter | None = None
    # Kept only in the worker process until terminal settlement. It carries
    # provider keys in-memory to fetch the exact Soniox usage-log dimensions;
    # its repr/payload deliberately never carries those keys or references.
    soniox_usage_tracker: SonioxUsageTracker | None = None
    tts_request_activity: TTSRequestActivity | None = None
    #: The locales this call's transcriber was CONSTRUCTED with, when there
    #: was more than one. Not a socket any provider confirmed: no adapter here
    #: reports one back, and a call can still be cancelled or refused after
    #: this. Empty for every other call, which is what lets a reader treat a
    #: non-empty value as a fact rather than a restatement of configuration --
    #: the configured set is what an agent asked for, and this is what
    #: survived the declaration, the ceiling and the region.
    stt_candidate_locales: tuple[str, ...] = ()
    _admission: _InitialStackAdmission | None = field(default=None, repr=False)


def semantic_marker_observed(language: str, transcript: str) -> bool:
    # Words are runs of letters and digits. A transcriber may split or
    # hyphenate the spoken marker ("pine apple", "Pine-apple!"), so two
    # adjacent words that together are exactly the marker count too; a looser
    # join across the whole transcript would let unrelated neighbouring words
    # ("alpine apple") spell it.
    words = "".join(
        character if character.isalnum() else " " for character in transcript.casefold()
    ).split()
    # Nova-3 has produced all three spellings from the same caller-side
    # ElevenLabs sample.  This is an acoustic proof marker, not a spelling test.
    markers = ("ananass", "ananas", "annanas") if language == "et" else ("pineapple",)
    if any(marker in word for marker in markers for word in words):
        return True
    joined = {first + second for first, second in zip(words, words[1:])}
    return any(marker in joined for marker in markers)


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
                "Oled Skysayi soe ja loomulik eesti keelt kõnelev vastuvõtuassistent. "
                "Vasta lühidalt, ära korda juba öeldut ja küsi korraga ainult üks küsimus."
            ),
            first_message="Tere! Olete helistanud Skysayile. Kuidas saan aidata?",
        )
    return DemoProfile(
        language="en",
        stt_language="en-US",
        tts_model="eleven_flash_v2_5",
        voice_id=voice_id,
        llm_model=llm_model,
        instructions=(
            "You are the warm, concise Skysay receptionist. Speak naturally, keep replies "
            "short, never repeat facts the caller already gave, and ask only one question at a "
            "time."
        ),
        first_message="Hi, you've reached Skysay. How can I help?",
    )


#: Chirp 2 recognizer regions that serve si-LK, measured rather than read.
#:
#: Chirp 2's language coverage is PER REGION, and Sinhala's coverage is
#: narrower than the European languages this deployment-wide variable was
#: introduced for. Probed against a real credential: ``asia-southeast1`` and
#: ``us-central1`` transcribe si-LK; ``eu`` and ``global`` answer that the
#: model does not exist there. So an operator who set ``GOOGLE_STT_LOCATION``
#: to ``eu`` for German -- a correct setting, and one that predates Sinhala
#: ever reaching this model -- would now have every Sinhala call connect,
#: reserve the caller's money, and transcribe nothing.
#:
#: Refusing before media is the honest outcome, and it is the same fail-closed
#: rule the STT locale lookup applies one layer up: a stack this worker cannot
#: actually run is turned away rather than sold. This guard covers the REAL
#: PHONE PATH, not only a demo -- ``build_demo_stack`` below is the builder
#: ``agent/production_agent.py`` calls (see this module's docstring); two
#: separate reviewers read the file name and assumed otherwise, so it is worth
#: saying here rather than only at the top of the file.
#:
#: The control plane cannot make this check. ``GOOGLE_STT_LOCATION`` is a
#: voice-host environment variable that no API process can observe, so a
#: customer can still SAVE a Sinhala Chirp 2 agent a mis-regioned deployment
#: cannot run. The worker refusing it before media is the last seam that can
#: see the truth, which is why the check lives here and not in
#: ``call_admission_profile``. The guard is deliberately
#: scoped to ``si``, the one language a measurement exists for. The other
#: Chirp 2 languages have always been sent to whatever region this variable
#: named, unchecked; that is pre-existing and unmeasured, and inventing an
#: allowlist for them from the same published table that got Sinhala wrong
#: would repeat the mistake this change exists to correct.
#:
#: ``europe-west4`` joined the set on 2026-09-05, measured the same
#: way: 12 of 12 si-LK streams from inside the production agents image, both
#: real caller-reply clips, no error, and transcripts byte-identical to the
#: ones ``asia-southeast1`` returned for the same audio. The earlier finding
#: that ``eu`` does not serve si-LK stands -- the ``eu`` MULTI-region and the
#: ``europe-west4`` region are different recognizer locations, and only the
#: latter was ever measured to work.
_CHIRP2_SINHALA_LOCATIONS = frozenset({"asia-southeast1", "us-central1", "europe-west4"})

#: The Chirp 2 recognizer region a deployment uses when its operator has not
#: named one. It was ``asia-southeast1`` because that is where si-LK was first
#: measured to stream, and every Chirp 2 language then rode along
#: to Singapore from a voice host in Amsterdam: Google terminates the TLS
#: connection at an edge next to the host (about 1 ms away), but the
#: recognizer runs in the named region, so every streaming exchange crossed
#: the backbone to Singapore and back. Measured from the production voice
#: host on 2026-09-05 (16 kHz clips at wall-clock pace, raw
#: Speech V2 client): the FINAL transcript after the caller stops arrived
#: 522 ms sooner for en-US and 531 ms sooner for si-LK in ``europe-west4``
#: (median 1208 ms against 1730 ms, and 1559 ms against 2090 ms), with
#: byte-identical Sinhala text. ``GOOGLE_STT_LOCATION`` remains the
#: operator's deployment-wide override; the guard above still refuses a
#: Sinhala stack on any region that was never measured to serve it.
_CHIRP2_DEFAULT_LOCATION = "europe-west4"

#: Google Speech-to-Text models this worker constructs. Chirp 2 is Speech V2
#: and needs a regional recognizer. V1 ``command_and_search`` is gone;
#: handing any other Google model to the plugin is a construction error, not a
#: silent fall-back to the global V1 endpoint.
_GOOGLE_STT_MODELS = frozenset({"chirp_2"})


#: The end-of-speech sensitivity every Chirp 2 stream asks Speech V2 for.
#:
#: Google waits for silence after speech before committing a FINAL; the
#: default ("STANDARD") waits longest. Measured from the production voice
#: host on 2026-09-05 (``europe-west4``, same audio at wall-clock pace, raw
#: Speech V2 client): ``SHORT`` returned the final after the caller stopped
#: 380 ms sooner on a 3.7 s English reply (median 965 ms against 1345 ms),
#: 358 ms sooner on a 7.4 s English sentence (850 ms against 1208 ms) and
#: 557 ms sooner on si-LK (1002 ms against 1559 ms), keeping every word.
#: ``SUPERSHORT`` is faster still on paper and is deliberately not used: it
#: cut every short English reply into two finals, the first landing about
#: 900 ms before the caller stopped, and on Sinhala it returned an account
#: number "226" as the word for "two hundred" in six of six streams -- a
#: correctness loss, not a latency trade. The pinned LiveKit plugin discards
#: this option for every model but ``chirp_3``; ``google_stt.STT`` restores
#: it, which is why the worker's default Google adapter is that subclass.
#: The setting shortens the wait for the transcript; whether the caller's
#: turn has ended is still the VAD's and turn detector's decision.
_CHIRP2_ENDPOINTING_SENSITIVITY = "ENDPOINTING_SENSITIVITY_SHORT"


def google_stt_endpointing(model: str) -> str:
    """The endpointing sensitivity a Google STT model is asked for.

    Chirp 2 is Speech V2 and carries the field. Any other Google model is not
    a live transcriber on this worker.
    """
    if model not in _GOOGLE_STT_MODELS:
        raise DemoConfigurationError("Google STT must use chirp_2")
    return _CHIRP2_ENDPOINTING_SENSITIVITY


def google_stt_location(model: str, language: str, environ: Mapping[str, str]) -> str:
    """The V2 recognizer region a Google STT stream is sent to.

    The operator's ``GOOGLE_STT_LOCATION`` wins for the whole deployment, and
    the code default is the region measured for Chirp 2. Sinhala is refused
    before media on any region not measured to serve si-LK, whichever way the
    region was chosen: a stack this worker cannot actually run is turned away
    rather than sold (see ``_CHIRP2_SINHALA_LOCATIONS``).
    """
    if model not in _GOOGLE_STT_MODELS:
        raise DemoConfigurationError("Google STT must use chirp_2")
    location = str(environ.get("GOOGLE_STT_LOCATION") or _CHIRP2_DEFAULT_LOCATION)
    if language == "si" and location not in _CHIRP2_SINHALA_LOCATIONS:
        raise DemoConfigurationError("Sinhala on Chirp 2 needs a recognizer region that serves si-LK")
    return location


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
        "stt": ("SONIOX_STT_API_KEY", "SKYSAY_BYOK_SONIOX_STT"),
        "tts": ("SONIOX_TTS_API_KEY", "SKYSAY_BYOK_SONIOX_TTS"),
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

    marker = environ.get("SKYSAY_BYOK_FISHAUDIO_TTS")
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


def _cartesia_tts_key(environ: Mapping[str, str]) -> str:
    """Resolve a Cartesia TTS lease without crossing into the managed key."""

    marker = environ.get("SKYSAY_BYOK_CARTESIA_TTS")
    if marker is not None:
        if marker != "1":
            raise DemoConfigurationError("Cartesia BYOK credential marker is invalid")
        key = str(environ.get("CARTESIA_TTS_API_KEY") or "")
        if not key:
            raise DemoConfigurationError("Cartesia BYOK credential is missing")
        return key
    if str(environ.get("CARTESIA_TTS_API_KEY") or ""):
        raise DemoConfigurationError("Cartesia BYOK credential marker is invalid")
    return str(environ.get("CARTESIA_API_KEY") or "")


def _azure_speech_credentials(environ: Mapping[str, str]) -> tuple[str, str]:
    """Resolve the platform Azure Speech subscription key and its region.

    One managed subscription serves both components, and there is no BYOK
    overlay for Azure: no per-call lease variable exists in
    ``component_credentials.py`` and no ``validate_provider_secret`` branch
    exists to prove a customer key, so a scoped-lease resolver here would be a
    credential path with nothing behind it.

    The region is not a secret but it IS load bearing: it selects the endpoint
    host. Both values are returned for the caller to CHECK before construction,
    because the plugin's own fallback is to read ``AZURE_SPEECH_KEY`` and
    ``AZURE_SPEECH_REGION`` from the ambient process -- and this worker is
    handed a per-call environment precisely so the ambient process is not what
    answers. Refusing on an empty value keeps that fallback unreachable.

    Both are stripped. An Azure subscription key is hex, so no meaning is lost,
    and leaving the key unstripped made a whitespace-only value truthy here and
    refused three calls later inside the adapter builder -- the same refusal,
    but reached after the caller had already been told the stack was fine.

    The region is shape-checked as well as stripped, for the same reason and
    one step further on: it is the endpoint HOST, and both pinned plugins
    defer building that host until stream time, so a malformed-but-truthy
    value passes every readiness check here and fails on the first spoken
    turn. A value refused before media is a refused call; a value refused at
    the first turn is a caller listening to silence.
    """
    key = str(environ.get("AZURE_SPEECH_KEY") or "").strip()
    # A key with an interior newline survives strip() and dies later, in a
    # different place for each component: TTS when aiohttp builds the
    # subscription-key header, STT only after media has started. The control
    # plane's preview and inventory already refuse this exact shape, so the
    # worker refusing it too means one malformed secret has one symptom.
    if any(character in key for character in "\r\n"):
        raise DemoConfigurationError("Azure speech key is malformed")
    region = str(environ.get("AZURE_SPEECH_REGION") or "").strip()
    if region:
        try:
            from .azure_speech import AzureTTSRuntimeCompatibilityError, azure_region
        except ImportError:  # pragma: no cover - flat worker import
            from azure_speech import AzureTTSRuntimeCompatibilityError, azure_region
        try:
            region = azure_region(region)
        except AzureTTSRuntimeCompatibilityError as exc:
            raise DemoConfigurationError("Azure speech region is not a region name") from exc
    return key, region


def _gemini_tts_key(environ: Mapping[str, str]) -> str:
    """Resolve a platform Gemini TTS key without crossing an LLM BYOK key.

    Gemini TTS is platform-managed for the first cohort. A scoped marker is
    still needed when a selected Gemini LLM BYOK lease replaces the generic
    key: TTS must keep using the platform account rather than silently using
    that LLM customer's key. A stale scoped variable is an error rather than a
    fallback to whichever generic Gemini key happens to be in the process.
    """
    platform_marker = environ.get("SKYSAY_PLATFORM_GEMINI_TTS")
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


@dataclass(frozen=True)
class TTSComponent:
    """One synthesis segment; never owns a recognizer or language model."""

    tts: Any
    tts_unit_meter: NativeTTSUnitMeter | None = None
    soniox_usage_tracker: SonioxUsageTracker | None = None
    tts_request_activity: TTSRequestActivity | None = None
    language: str = ""


def prepare_tts_component(
    profile: DemoProfile,
    *,
    environ: Mapping[str, str],
    elevenlabs_tts: Callable[..., Any] | None = None,
    elevenlabs_stream_adapter: Callable[..., Any] | None = None,
    elevenlabs_sentence_tokenizer: Callable[..., Any] | None = None,
    elevenlabs_voice_settings: Callable[..., Any] | None = None,
    gemini_tts: Callable[..., Any] | None = None,
    soniox_tts: Callable[..., Any] | None = None,
    fish_audio_tts: Callable[..., Any] | None = None,
    cartesia_tts: Callable[..., Any] | None = None,
    azure_tts: Callable[..., Any] | None = None,
    soniox_usage_reference: str = "",
    require_soniox_usage_reference: bool = True,
    _usage_tracker: SonioxUsageTracker | None = None,
    admission_context: CallVoiceAdmission | None = None,
    call_owner: object | None = None,
    target_config: Mapping[str, Any] | None = None,
    simultaneous_stt: Mapping[str, Any] | None = None,
) -> Callable[[], TTSComponent]:
    """Construct only synthesis, with the same adapter and billing rules as a call.

    The initial full stack may share its Soniox tracker with STT. A replacement
    creates a tracker containing only TTS, so no recognizer reference exists for
    a stream that will never run. Factories are lazy and component-specific.
    """
    registry = load_registry()
    tts_registry = profile.tts_registry
    validate_profile_capabilities(
        profile, registry=registry, admission_context=admission_context,
        call_owner=call_owner, target_config=target_config, simultaneous_stt=simultaneous_stt,
    )
    # The delivery profile is refused, not guessed, when this image's table
    # does not know it: the control plane validates against the same packaged
    # table, so a mismatch means an API ahead of this worker, and the repo's
    # rule for that skew is to refuse the config rather than run it wrong.
    try:
        delivery = resolve_delivery(tts_registry, profile.tts_model, profile.delivery or None)
    except DeliveryTableError as exc:
        # The packaged table is part of this image; an unreadable one is an
        # image fault, and must not be reported as the agent's configuration.
        raise DemoConfigurationError("delivery table unavailable in this worker image") from exc
    except ValueError as exc:
        raise DemoConfigurationError("unknown delivery profile") from exc
    supplied_soniox_tts = soniox_tts is not None
    eleven_key = str(environ.get("ELEVENLABS_API_KEY") or environ.get("ELEVEN_API_KEY") or "")
    gemini_tts_key = _gemini_tts_key(environ) if tts_registry == "gemini_tts" else ""
    soniox_tts_key = _soniox_component_key(environ, "tts") if tts_registry == "soniox" else ""
    fish_key = _fish_audio_tts_key(environ) if tts_registry == "fishaudio" else ""
    cartesia_key = _cartesia_tts_key(environ) if tts_registry == "cartesia" else ""
    azure_key, azure_region = _azure_speech_credentials(environ) if tts_registry == "azure_tts" else ("", "")
    required_key = {
        "elevenlabs": eleven_key, "gemini_tts": gemini_tts_key,
        "soniox": soniox_tts_key, "fishaudio": fish_key, "cartesia": cartesia_key,
        "azure_tts": azure_key and azure_region,
    }.get(tts_registry)
    if required_key is None:
        raise DemoConfigurationError("selected TTS provider has no production adapter")
    if not required_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if tts_registry == "soniox" and require_soniox_usage_reference and not soniox_usage_reference.strip():
        raise DemoConfigurationError("Soniox usage correlation is required before media admission")
    soniox_usage_tracker = _usage_tracker
    if tts_registry == "soniox" and soniox_usage_tracker is None and soniox_usage_reference.strip():
        soniox_usage_tracker = SonioxUsageTracker(
            call_reference=soniox_usage_reference,
            components={"tts": (soniox_tts_key, profile.tts_model)},
        )
    try:
        if tts_registry == "elevenlabs":
            if elevenlabs_tts is None:
                try:
                    from .elevenlabs_tts import build_tts as build_elevenlabs_tts
                except ImportError:  # pragma: no cover - flat worker import
                    from elevenlabs_tts import build_tts as build_elevenlabs_tts
                elevenlabs_tts = build_elevenlabs_tts
            if eleven_stream_transport_clips_tail(profile.tts_model) and elevenlabs_stream_adapter is None:
                elevenlabs_stream_adapter = _quarantined_eleven_stream_adapter()
            if delivery.control == "voice_settings" and elevenlabs_voice_settings is None:
                from livekit.plugins.elevenlabs import VoiceSettings
                elevenlabs_voice_settings = VoiceSettings
            if eleven_stream_transport_clips_tail(profile.tts_model) and elevenlabs_sentence_tokenizer is None:
                try:
                    from .first_clause import first_clause_tokenizer
                except ImportError:  # pragma: no cover - flat worker import
                    from first_clause import first_clause_tokenizer
                elevenlabs_sentence_tokenizer = first_clause_tokenizer
        elif tts_registry == "gemini_tts" and gemini_tts is None:
            try:
                from .gemini_tts import TTS
            except ImportError:  # pragma: no cover - flat worker import
                from gemini_tts import TTS
            gemini_tts = TTS
        elif tts_registry == "soniox" and not supplied_soniox_tts:
            if soniox_usage_tracker is not None:
                try:
                    from .soniox_tts import TTS
                except ImportError:  # pragma: no cover - flat worker import
                    from soniox_tts import TTS
            else:
                from livekit.plugins.soniox import TTS
            soniox_tts = TTS
        elif tts_registry == "fishaudio" and fish_audio_tts is None:
            try:
                from .fish_audio_tts import TTS
            except ImportError:  # pragma: no cover - flat worker import
                from fish_audio_tts import TTS
            fish_audio_tts = TTS
        elif tts_registry == "cartesia" and cartesia_tts is None:
            try:
                from .cartesia_tts import build_tts
            except ImportError:  # pragma: no cover - flat worker import
                from cartesia_tts import build_tts
            cartesia_tts = build_tts
        elif tts_registry == "azure_tts" and azure_tts is None:
            try:
                from .azure_speech import build_tts
            except ImportError:  # pragma: no cover - flat worker import
                from azure_speech import build_tts
            azure_tts = build_tts
    except ImportError as exc:
        raise DemoConfigurationError("install the pinned LiveKit TTS plugins") from exc
    tts_request_activity = TTSRequestActivity() if tts_registry == "soniox" and not supplied_soniox_tts else None

    def native_unit_meter() -> NativeTTSUnitMeter | None:
        # The TTS row's data says whether the worker meters this adapter, in
        # which unit, and what each request ending is charged -- never a
        # branch here. Only a row without request billing (Soniox, settled
        # from its own usage tracker) has no meter.
        row = registry.tts_metering(tts_registry, profile.tts_model)
        billing = row.get("billing") if isinstance(row, Mapping) else None
        if not isinstance(billing, Mapping) or "request_billing" not in billing:
            return None
        try:
            return NativeTTSUnitMeter(RequestBilling.from_capability(row))
        except RuntimeError as exc:
            raise DemoConfigurationError("selected TTS capability declares invalid request billing") from exc

    def construct() -> TTSComponent:
        azure_tts_language = ""
        tts_unit_meter = native_unit_meter()
        if tts_registry == "gemini_tts":
            assert gemini_tts is not None
            # Gemini-TTS has no explicit language parameter -- it infers the
            # spoken language from the input text itself, per Google's docs.
            gemini_tts_kwargs: dict[str, Any] = {
                "model": profile.tts_model,
                "voice_name": profile.voice_id,
                "api_key": gemini_tts_key,
                "native_unit_meter": tts_unit_meter,
            }
            if delivery.control == "instruction":
                # The adapter splices this in front of the quoted reply; the
                # voice preview builds the identical prompt from the same table.
                gemini_tts_kwargs["instructions"] = delivery.settings
            tts = gemini_tts(**gemini_tts_kwargs)
        elif tts_registry == "elevenlabs":
            assert elevenlabs_tts is not None
            if profile.tts_model == ELEVEN_V4_TURBO_MODEL:
                install_eleven_v4_turbo_dialogue_route()
            sentence_tokenizer = None
            if eleven_stream_transport_clips_tail(profile.tts_model):
                assert elevenlabs_sentence_tokenizer is not None
                try:
                    sentence_tokenizer = elevenlabs_sentence_tokenizer()
                except ImportError as exc:
                    raise DemoConfigurationError("install the pinned LiveKit TTS plugins") from exc
            eleven_kwargs: dict[str, Any] = {
                "model": profile.tts_model,
                "voice_id": profile.voice_id,
                "language": profile.language,
                "api_key": eleven_key,
                "native_unit_meter": tts_unit_meter,
            }
            if delivery.control == "voice_settings":
                assert elevenlabs_voice_settings is not None
                # The dataclass requires both stability and similarity_boost; a
                # table entry that omits similarity_boost (eleven_v3, whose
                # text-to-dialogue transport honours stability only) keeps the
                # plugin's documented default so the request carries exactly what
                # the table says and nothing the transport would have dropped
                # anyway.
                settings = dict(delivery.settings)
                settings.setdefault("similarity_boost", 0.75)
                eleven_kwargs["voice_settings"] = elevenlabs_voice_settings(**settings)
            tts = elevenlabs_tts(**eleven_kwargs)
            if eleven_stream_transport_clips_tail(profile.tts_model):
                # ``_ELEVEN_DIALOGUE_MODEL_PREFIX`` documents the measurement.
                # The adapter is asserted rather than skipped: a missing wrap here
                # would silently put every turn of the call back on the clipping
                # websocket, which is worse than refusing to build.
                assert elevenlabs_stream_adapter is not None
                assert elevenlabs_sentence_tokenizer is not None
                # The adapter's own tokenizer releases a sentence only once the
                # next one has begun, so a one-sentence reply waits for the whole
                # model stream. ``first_clause`` releases the opening clause as
                # soon as it can carry the next request's time to first byte.
                tts = elevenlabs_stream_adapter(
                    tts=tts, sentence_tokenizer=sentence_tokenizer
                )
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
                **({"speed": delivery.settings} if delivery.control == "speed" else {}),
                **(
                    {"client_reference_id": soniox_usage_tracker.reference_for("tts"),
                     **({"on_request_started": tts_request_activity.mark_started}
                        if tts_request_activity is not None else {})}
                    if soniox_usage_tracker is not None
                    else {}
                ),
            )
        elif tts_registry == "fishaudio":
            if profile.tts_model != "s2-pro":
                raise DemoConfigurationError("Fish Audio TTS must use paid s2-pro")
            assert fish_audio_tts is not None
            tts = fish_audio_tts(
                model="s2-pro",
                voice_id=profile.voice_id,
                output_format="wav",
                latency_mode="balanced",
                chunk_length=100,
                api_key=fish_key,
                native_unit_meter=tts_unit_meter,
            )
        elif tts_registry == "cartesia":
            if profile.tts_model != "sonic-3.6":
                raise DemoConfigurationError("Cartesia TTS must use sonic-3.6")
            assert cartesia_tts is not None
            try:
                tts = cartesia_tts(
                    model="sonic-3.6",
                    voice=profile.voice_id,
                    language=profile.language,
                    sample_rate=24000,
                    word_timestamps=False,
                    api_key=cartesia_key,
                    api_version="2026-08-14",
                    native_unit_meter=tts_unit_meter,
                    **({"speed": delivery.settings} if delivery.control == "speed" else {}),
                )
            except RuntimeError as exc:
                # A provider implementation mismatch is a worker configuration
                # fault, not an opportunity to change voice or API version.
                raise DemoConfigurationError("Cartesia TTS runtime is incompatible") from exc
        elif tts_registry == "azure_tts":
            if profile.tts_model != "neural":
                raise DemoConfigurationError("Azure TTS must use the neural model")
            assert azure_tts is not None
            azure_tts_language = registry.locale_for(
                "tts", tts_registry, profile.tts_model, profile.language
            ) or ""
            if not azure_tts_language:
                # Fail closed rather than let the plugin default ``xml:lang`` to
                # en-US: a Sinhala reply inside an English document is the silent
                # wrong-audio failure this whole table exists to prevent.
                raise DemoConfigurationError("Azure TTS has no locale for the selected language")
            # The artifact settles the LANGUAGE; the voice settles which region of
            # it, when the voice names one. ``en-GB-RyanNeural`` is a monolingual
            # voice offered for ``en``, and the artifact says ``en`` is ``en-US``:
            # opening its document in en-US contradicts the voice about a language
            # they already agree on.
            try:
                from .azure_speech import azure_speaking_locale
            except ImportError:  # pragma: no cover - flat worker import
                from azure_speech import azure_speaking_locale
            azure_tts_language = azure_speaking_locale(profile.voice_id, azure_tts_language)
            # Microsoft bills the escaped SSML body with Han characters doubled --
            # not LiveKit's ``characters_count``, which is the submitted length. So
            # Azure carries its own meter through the same settlement seam Cartesia
            # uses, and ``component_billing`` prefers it over the SDK event.
            try:
                tts = azure_tts(
                    voice=profile.voice_id,
                    language=azure_tts_language,
                    sample_rate=24000,
                    speech_key=azure_key,
                    speech_region=azure_region,
                    native_unit_meter=tts_unit_meter,
                )
            except RuntimeError as exc:
                # An unrecognised voice locale or a plugin whose SSML builder moved
                # is a worker configuration fault. Refusing before media is the
                # honest outcome: the alternative is a 400 on every turn, which the
                # caller experiences as a connected line that never speaks.
                raise DemoConfigurationError("Azure TTS runtime is incompatible") from exc
        else:  # Defensive guard if a future registry row is added without a builder.
            raise DemoConfigurationError("selected TTS provider has no production adapter")
        return TTSComponent(
            tts_request_activity=tts_request_activity,
            tts=tts,
            tts_unit_meter=tts_unit_meter,
            soniox_usage_tracker=soniox_usage_tracker,
            language=azure_tts_language,
        )

    return construct


def build_tts_component(profile: DemoProfile, **kwargs: Any) -> TTSComponent:
    """Build one synthesis segment, with no STT/LLM imports or construction."""
    return prepare_tts_component(profile, **kwargs)()


def construct_stt_component(
    profile, *, environ, configured_stt_locales=(), simultaneous_stt_language=None,
    simultaneous_soniox_options=None, simultaneous_azure_locales=None,
    google_stt=None, deepgram_stt=None, soniox_stt=None, soniox_stt_options=None, azure_stt=None,
    deepgram_key="", soniox_stt_key="", azure_key="", azure_region="", soniox_usage_tracker=None,
    observe=None,
):
    """Construct only a transcriber from already admitted, resolved choices.

    Both the complete cascade stack and the independent workflow listener use
    this provider adapter seam. This function constructs no LLM, TTS or VAD.
    ``observe`` is handed the constructor arguments actually passed, with every
    credential and usage reference left out, just before the constructor runs:
    what a construction attestation records (a region is a name, not a secret).
    """
    def built(**settings):
        if observe is not None:
            try:
                observe({"registry_name": stt_registry, "model": profile.stt_model, **settings})
            except Exception:  # noqa: BLE001 - an attestation never moves a call
                pass

    stt_registry = profile.stt_registry
    azure_stt_language = ""
    if stt_registry == "google":
        assert google_stt is not None
        google_model = profile.stt_model or "chirp_2"
        google_kwargs: dict[str, Any] = {
            "model": google_model,
            "languages": profile.stt_language,
            **_google_stt_credential_kwargs(environ),
        }
        # Chirp 2 needs the regional recognizer. ``google_stt_location`` owns
        # the operator override and the Sinhala region guard; a leftover V1
        # model raises rather than hitting the global V1 endpoint.
        google_kwargs["location"] = google_stt_location(
            google_model, profile.language, environ
        )
        if configured_stt_locales:
            # One recognizer region serves the whole stream, so every
            # candidate has to be servable AT THE REGION THIS CALL OPENS --
            # not merely servable somewhere. Both review engines read the
            # first version of this loop as asking the weaker question, so it
            # now compares the answer against the location already chosen
            # rather than only watching for a refusal. Asked through the same
            # function that owns the region rule, so there is no second copy
            # of the table; a candidate this region cannot serve drops the
            # call back to the single locale it runs on today.
            try:
                for candidate in profile.languages:
                    if (
                        google_stt_location(google_model, candidate, environ)
                        != google_kwargs["location"]
                    ):
                        raise DemoConfigurationError(
                            "a candidate language needs a different recognizer region"
                        )
            except DemoConfigurationError:
                # Cleared, not just unused: the call's identity is read as
                # evidence of the socket that opened, and a set this branch
                # declined to send must not be recorded as one it did.
                configured_stt_locales = ()
            else:
                google_kwargs["languages"] = list(configured_stt_locales)
        google_kwargs["endpointing_sensitivity"] = google_stt_endpointing(google_model)
        built(**{key: google_kwargs[key] for key in ("model", "languages", "location", "endpointing_sensitivity")})
        stt = google_stt(**google_kwargs)
    elif stt_registry == "deepgram":
        assert deepgram_stt is not None
        built(model="nova-3", language=simultaneous_stt_language or profile.stt_language, mip_opt_out=True)
        stt = deepgram_stt(
            model="nova-3",
            language=simultaneous_stt_language or profile.stt_language,
            api_key=deepgram_key,
            mip_opt_out=True,
        )
    elif stt_registry == "soniox":
        if profile.stt_model != "stt-rt-v5":
            raise DemoConfigurationError("Soniox STT must use stt-rt-v5")
        assert soniox_stt is not None
        assert soniox_stt_options is not None
        # New automatic publications use Soniox's broad language
        # identification mode without a restrictive hint list. Released
        # strict-hint publications keep their exact frozen construction.
        soniox_options_kwargs: dict[str, Any] = {
            "model": "stt-rt-v5",
            **(
                simultaneous_soniox_options
                if simultaneous_soniox_options is not None
                else {
                    "language_hints": [profile.stt_language],
                    "language_hints_strict": True,
                    "enable_language_identification": True,
                }
            ),
            "max_endpoint_delay_ms": 1200,
        }
        built(**soniox_options_kwargs)
        if soniox_usage_tracker is not None:
            soniox_options_kwargs["client_reference_id"] = soniox_usage_tracker.reference_for("stt")
        stt = soniox_stt(
            api_key=soniox_stt_key,
            params=soniox_stt_options(**soniox_options_kwargs),
        )
    elif stt_registry == "azure":
        if profile.stt_model != "realtime":
            raise DemoConfigurationError("Azure STT must use the realtime model")
        assert azure_stt is not None
        # ``profile.stt_language`` is the artifact's tag, resolved by
        # ``production.profile_from_realtime_config`` from the ONE locale table
        # and refused there if the artifact cannot name one. This builder used
        # to keep its own ``{"si": "si-LK"}`` map, which was a second authority
        # for the same question and could not grow past the language it was
        # written for.
        azure_stt_language = profile.stt_language
        if simultaneous_azure_locales is None and configured_stt_locales:
            # The declared candidate list, in the agent's own order. Only the
            # transcription socket widens: the reply language is still the one
            # this call was published in, and nothing here switches a profile.
            simultaneous_azure_locales = list(configured_stt_locales)
        # The key and region are passed rather than left to the plugin's own
        # ``AZURE_SPEECH_KEY``/``AZURE_SPEECH_REGION`` environment fallback:
        # this worker resolves credentials from the per-call environment it was
        # handed, and a plugin reading the ambient process instead would be the
        # one place a leased or absent credential could be silently replaced.
        built(speech_region=azure_region, **({"candidate_locales": list(simultaneous_azure_locales)}
              if simultaneous_azure_locales is not None else {"language": azure_stt_language}))
        try:
            stt = (
                azure_stt(
                    speech_key=azure_key,
                    speech_region=azure_region,
                    candidate_locales=simultaneous_azure_locales,
                )
                if simultaneous_azure_locales is not None
                else azure_stt(
                    speech_key=azure_key,
                    speech_region=azure_region,
                    language=azure_stt_language,
                )
            )
        except RuntimeError as exc:
            # A plugin whose STT class moved is a worker configuration fault,
            # not something to fall back from: the fallback would be the raw
            # adapter, whose usage metrics settlement cannot match.
            raise DemoConfigurationError("Azure STT runtime is incompatible") from exc
    else:  # Defensive guard if a future registry row is added without a builder.
        raise DemoConfigurationError("selected STT provider has no production adapter")

    return stt, tuple(configured_stt_locales), azure_stt_language


#: Where a call config carries the far-end listening construction the control
#: plane froze at admission (``transcriber.far_end_listening``).
FAR_END_LISTENING_KEY = "far_end_listening"


def frozen_far_end_listening(
    config: Mapping[str, Any], profile: "DemoProfile", *, registry: Any | None = None
) -> Mapping[str, Any] | None:
    """The frozen open construction for this call's transcriber, or ``None``.

    The control plane freezes it when the call is admitted; this image accepts
    it only when it is exactly what its own registry calls the open
    construction for the transcriber the profile runs. Anything else -- no
    value, a value for another transcriber, a registry that no longer declares
    one -- is ``None``, and the call listens the way the profile says, which
    is what every call did before. It never refuses a call: hearing less is
    the old behaviour, not a fault.
    """
    transcriber = config.get("transcriber") if isinstance(config, Mapping) else None
    value = transcriber.get(FAR_END_LISTENING_KEY) if isinstance(transcriber, Mapping) else None
    if not isinstance(value, Mapping):
        return None
    current = (registry or load_registry()).open_listening_config(profile.stt_registry, profile.stt_model)
    if current is None:
        return None
    expected = json.loads(json.dumps(_plain_mapping(current)))
    try:
        frozen = json.loads(json.dumps(_plain_mapping(value), allow_nan=False))
    except (TypeError, ValueError):
        return None
    return MappingProxyType(frozen) if frozen == expected else None


def _plain_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_mapping(item) for item in value]
    return value


def build_demo_stack(
    profile: DemoProfile,
    *,
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    call_owner: object | None = None,
    # A trusted, immutable registry resolution supplied only by the phase-3
    # call envelope.  It is intentionally not derived from a caller language,
    # a profile list, or a provider default: that would turn a controlled
    # multi-language admission into an unqualified worker-side fallback.
    simultaneous_stt: Mapping[str, Any] | None = None,
    # The open construction the control plane froze for this call's far end
    # (``far_end_listening``), already matched to this image's registry by
    # :func:`frozen_far_end_listening`. Never together with a follow-the-caller
    # declaration: that one chooses the transcriber's construction itself.
    far_end_listening: Mapping[str, Any] | None = None,
    deepgram_stt: Callable[..., Any] | None = None,
    google_stt: Callable[..., Any] | None = None,
    gemini_llm: Callable[..., Any] | None = None,
    elevenlabs_tts: Callable[..., Any] | None = None,
    elevenlabs_stream_adapter: Callable[..., Any] | None = None,
    # Builds the sentence tokenizer the stream adapter runs on. Injected by
    # tests; the worker image resolves the first-clause tokenizer below.
    elevenlabs_sentence_tokenizer: Callable[..., Any] | None = None,
    # Builds the plugin's ``VoiceSettings`` for a delivery profile. Injected
    # by tests; the worker image resolves the pinned plugin class below.
    elevenlabs_voice_settings: Callable[..., Any] | None = None,
    gemini_tts: Callable[..., Any] | None = None,
    soniox_stt: Callable[..., Any] | None = None,
    soniox_stt_options: Callable[..., Any] | None = None,
    soniox_tts: Callable[..., Any] | None = None,
    fish_audio_tts: Callable[..., Any] | None = None,
    cartesia_tts: Callable[..., Any] | None = None,
    azure_stt: Callable[..., Any] | None = None,
    azure_tts: Callable[..., Any] | None = None,
    load_vad: Callable[..., Any] | None = None,
    load_turn_detector: Callable[..., Any] | None = None,
    soniox_usage_reference: str = "",
    require_soniox_usage_reference: bool = True,
) -> DemoStack:
    refuse_freeform_workflow(config)
    stt_registry = profile.stt_registry
    tts_registry = profile.tts_registry
    turn_detector_supported = profile.language in TURN_DETECTOR_SUPPORTED_LANGUAGES
    # The delivery profile is refused, not guessed, when this image's table
    # does not know it: the control plane validates against the same packaged
    # table, so a mismatch means an API ahead of this worker, and the repo's
    # rule for that skew is to refuse the config rather than run it wrong.
    try:
        delivery = resolve_delivery(tts_registry, profile.tts_model, profile.delivery or None)
    except DeliveryTableError as exc:
        # The packaged table is part of this image; an unreadable one is an
        # image fault, and must not be reported as the agent's configuration.
        raise DemoConfigurationError("delivery table unavailable in this worker image") from exc
    except ValueError as exc:
        raise DemoConfigurationError("unknown delivery profile") from exc
    simultaneous_stt_language: str | None = None
    simultaneous_soniox_options: dict[str, Any] | None = None
    simultaneous_azure_locales: list[str] | None = None
    if simultaneous_stt is not None:
        if not isinstance(simultaneous_stt, Mapping):
            raise DemoConfigurationError("simultaneous STT declaration is invalid")
        declared_registry = str(simultaneous_stt.get("registry_name") or "").strip().lower()
        declared_provider = str(simultaneous_stt.get("provider") or "").strip().lower()
        declared_model = str(simultaneous_stt.get("model") or "").strip()
        declared_mode = str(simultaneous_stt.get("mode") or "").strip()
        declared_languages = simultaneous_stt.get("languages")
        adapter_options = simultaneous_stt.get("adapter_options")
        if (
            declared_registry != stt_registry
            or declared_provider != stt_registry
            or declared_model != profile.stt_model
            or not declared_provider
            or not isinstance(declared_languages, list)
            or profile.language not in {str(item).strip().casefold() for item in declared_languages}
        ):
            raise DemoConfigurationError("simultaneous STT declaration does not match the frozen profile")
        # The registry resolved the exact adapter keyword before media.  The
        # worker consumes the frozen value rather than maintaining a second
        # provider/language map.  This adapter only exposes its documented
        # language keyword, so any future mode needs an explicit builder seam.
        if (
            declared_mode == "language_multi"
            and stt_registry == "deepgram"
            and isinstance(adapter_options, Mapping)
            and set(adapter_options) == {"language"}
            and str(adapter_options.get("language") or "").strip()
        ):
            simultaneous_stt_language = str(adapter_options["language"]).strip()
        elif (
            declared_mode in {"language_identification", "strict_language_hints"}
            and stt_registry == "soniox"
            and isinstance(adapter_options, Mapping)
            and adapter_options.get("enable_language_identification") is True
            and (
                (
                    declared_mode == "language_identification"
                    and set(adapter_options) == {"enable_language_identification"}
                )
                or (
                    declared_mode == "strict_language_hints"
                    and set(adapter_options) == {
                        "language_hints",
                        "language_hints_strict",
                        "enable_language_identification",
                    }
                    and isinstance(adapter_options.get("language_hints"), list)
                    and profile.language in {
                        str(item).strip().casefold()
                        for item in adapter_options["language_hints"]
                    }
                    and adapter_options.get("language_hints_strict") is True
                )
            )
        ):
            simultaneous_soniox_options = dict(adapter_options)
        elif (
            declared_mode == "continuous_language_identification"
            and stt_registry == "azure"
            and isinstance(adapter_options, Mapping)
            and set(adapter_options) == {"language_id_mode", "candidate_locales"}
            and adapter_options.get("language_id_mode") == "continuous"
            and isinstance(adapter_options.get("candidate_locales"), Mapping)
            and len(adapter_options["candidate_locales"]) >= 2
            and profile.language in adapter_options["candidate_locales"]
            and all(
                isinstance(item, str) and item.strip()
                for item in adapter_options["candidate_locales"].values()
            )
        ):
            # Ordered by language code so the candidate list a call opens with
            # is a function of the frozen set and nothing else. The provider
            # is told which languages may arrive, never which is expected --
            # a candidate order that depended on the opening profile would
            # make two calls on one published agent listen differently.
            simultaneous_azure_locales = [
                str(adapter_options["candidate_locales"][language])
                for language in sorted(adapter_options["candidate_locales"])
            ]
        else:
            raise DemoConfigurationError("simultaneous STT adapter mode is unavailable in this worker")
    # Set by the Azure STT branch below and read again by ``safe_identity``.
    azure_stt_language = ""
    # The tag the SELECTED language resolves to for synthesis. Same table, same
    # accessor: a multilingual Azure voice speaks whatever this says, so
    # deriving it from the voice id would silently override the customer.
    azure_tts_language = ""

    registry = load_registry()
    validate_profile_capabilities(profile, registry=registry)
    if far_end_listening is not None:
        if simultaneous_stt is not None:
            raise DemoConfigurationError("far-end listening cannot be combined with a follow-the-caller declaration")
        if frozen_far_end_listening({"transcriber": {"far_end_listening": far_end_listening}}, profile,
                                    registry=registry) is None:
            raise DemoConfigurationError("far-end listening does not match this transcriber's open construction")
        # The open construction's one provider keyword, sent in place of the
        # profile's locale -- the same seam a frozen ``language_multi``
        # follow declaration uses, so the adapter needs no second path.
        simultaneous_stt_language = str(far_end_listening["adapter_options"]["language"])
    # What this call may HEAR, decided once for every transcriber.
    #
    # A recognizer locked to one locale cannot report that it heard a
    # different one. Azure in particular answers in the locale it was given
    # whatever was spoken: measured on a real caller recording, English
    # speech came back as 164 characters of plausible-looking Sinhala script
    # with no error, no empty result and no low-confidence signal, while the
    # incumbent Google recognizer surfaced the same speech as readable Latin
    # text and Azure's own identification mode named the language en-US. That
    # is a property of listening on one socket, not of any language, so the
    # rule is not written per provider: the artifact declares what each
    # adapter's API takes and the answer follows from the declaration.
    #
    # ``None`` from the accessor is the ordinary case and means this call
    # opens on the single locale below -- one configured language, an adapter
    # with no candidate list, a set larger than the adapter compares at once,
    # or a language it does not declare. None of those refuse the call. An
    # agent selected for twelve languages runs today on the one locale its
    # call speaks and is entitled to go on doing that; widening is offered
    # where the adapter supports it and is not a new way to fail.
    #
    # A per-call language override is the one shape that opts out: it rewrites
    # the language this call speaks without touching what the agent was
    # configured for, and a candidate set that cannot hear the language the
    # caller is being addressed in is worse than the single locale it would
    # replace. A frozen follow-the-caller construction opts out too -- that
    # publication already decided its own socket, above.
    #
    # The language this call speaks leads the list, and the agent's configured
    # order follows it. The set is identical either way -- the guard below
    # makes the call's language a member of it -- but an adapter that treats
    # the first tag as primary would otherwise be pointed at whichever
    # language the customer happened to list first, which for a call being
    # conducted in another one is a milder version of the failure this whole
    # rule exists to remove. Neither adapter shipped here ranks them, and
    # nobody should have to keep knowing that.
    configured_stt_locales: tuple[str, ...] = ()
    if simultaneous_stt is None and profile.language in profile.languages:
        configured_stt_locales = (
            registry.stt_candidate_locales(
                stt_registry, profile.stt_model, (profile.language, *profile.languages)
            )
            or ()
        )
    missing_factory = (
        (stt_registry == "google" and google_stt is None)
        or (stt_registry == "deepgram" and deepgram_stt is None)
        or (stt_registry == "soniox" and (soniox_stt is None or soniox_stt_options is None))
        or (stt_registry == "azure" and azure_stt is None)
        or gemini_llm is None
        or load_vad is None
        or (turn_detector_supported and load_turn_detector is None)
    )
    if missing_factory:
        try:
            from livekit.agents import inference
            from livekit.plugins import deepgram, google, silero
            if stt_registry == "soniox":
                from livekit.plugins import soniox
                try:
                    from .soniox_stt import STT as OwnedSonioxSTT
                except ImportError:  # pragma: no cover - flat worker import
                    from soniox_stt import STT as OwnedSonioxSTT
                # The worker closes these streams itself before reading usage
                # (``soniox_stt``); the plain plugin STT would leave that to the SDK.
                soniox_stt = soniox_stt or OwnedSonioxSTT
                soniox_stt_options = soniox_stt_options or soniox.STTOptions
            if stt_registry == "google" and google_stt is None:
                try:
                    from .google_stt import STT as EndpointedGoogleSTT
                except ImportError:  # pragma: no cover - flat worker import
                    from google_stt import STT as EndpointedGoogleSTT
            if stt_registry == "azure" and azure_stt is None:
                try:
                    from .azure_speech import build_stt as MeteredAzureSTT
                except ImportError:  # pragma: no cover - flat worker import
                    from azure_speech import build_stt as MeteredAzureSTT
                azure_stt = MeteredAzureSTT
        except ImportError as exc:
            raise DemoConfigurationError(
                "install the pinned LiveKit STT, LLM, VAD, and turn-detector plugins"
            ) from exc
        deepgram_stt = deepgram_stt or deepgram.STT
        if stt_registry == "google":
            google_stt = google_stt or EndpointedGoogleSTT
        gemini_llm = gemini_llm or google.LLM
        load_vad = load_vad or pinned_silero_vad_loader(silero.VAD.load)
        load_turn_detector = load_turn_detector or inference.TurnDetector
    deepgram_key = str(environ.get("DEEPGRAM_API_KEY") or "")
    gemini_key = str(environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or "")
    # The per-call environment can carry a scoped lease for a different
    # component. Resolve and validate that lease only if this profile actually
    # selects Soniox; unrelated historical stacks must not be rejected by a
    # stale/inapplicable Soniox marker.
    soniox_stt_key = _soniox_component_key(environ, "stt") if stt_registry == "soniox" else ""
    soniox_tts_key = _soniox_component_key(environ, "tts") if tts_registry == "soniox" else ""
    azure_key, azure_region = (
        _azure_speech_credentials(environ)
        if stt_registry == "azure" or tts_registry == "azure_tts"
        else ("", "")
    )
    if not gemini_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    if stt_registry == "google":
        if not _google_credentials_available(environ):
            raise DemoConfigurationError("provider settings changed after profile validation")
    elif stt_registry == "deepgram" and not deepgram_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    elif stt_registry == "soniox" and not soniox_stt_key:
        raise DemoConfigurationError("provider settings changed after profile validation")
    elif stt_registry == "azure" and not (azure_key and azure_region):
        raise DemoConfigurationError("provider settings changed after profile validation")
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
    construct_tts = prepare_tts_component(
        profile, environ=environ,
        elevenlabs_tts=elevenlabs_tts,
        elevenlabs_stream_adapter=elevenlabs_stream_adapter,
        elevenlabs_sentence_tokenizer=elevenlabs_sentence_tokenizer,
        elevenlabs_voice_settings=elevenlabs_voice_settings,
        gemini_tts=gemini_tts,
        soniox_tts=soniox_tts,
        fish_audio_tts=fish_audio_tts,
        cartesia_tts=cartesia_tts,
        azure_tts=azure_tts,
        soniox_usage_reference=soniox_usage_reference,
        require_soniox_usage_reference=require_soniox_usage_reference,
        _usage_tracker=soniox_usage_tracker,
    )

    assert gemini_llm is not None
    assert load_vad is not None

    stt_construction: dict[str, Any] = {}
    stt, configured_stt_locales, azure_stt_language = construct_stt_component(
        profile, environ=environ, configured_stt_locales=configured_stt_locales,
        simultaneous_stt_language=simultaneous_stt_language,
        simultaneous_soniox_options=simultaneous_soniox_options,
        simultaneous_azure_locales=simultaneous_azure_locales,
        google_stt=google_stt, deepgram_stt=deepgram_stt, soniox_stt=soniox_stt,
        soniox_stt_options=soniox_stt_options, azure_stt=azure_stt,
        deepgram_key=deepgram_key, soniox_stt_key=soniox_stt_key,
        azure_key=azure_key, azure_region=azure_region, soniox_usage_tracker=soniox_usage_tracker,
        observe=stt_construction.update,
    )

    llm = gemini_llm(model=profile.llm_model, api_key=gemini_key)

    synthesis = construct_tts()
    tts = synthesis.tts
    azure_tts_language = synthesis.language
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
        # Bound the wait for the caller's final transcript. The SDK
        # arms this at VAD end-of-speech, cancels it on VAD start-of-speech
        # and on any final with text, and on expiry emits ONE event --
        # ``user_transcription_timeout`` -- committing nothing by itself.
        # The production worker answers that event by forcing Azure to
        # finalize (transcription_stall.py); for every other provider, and
        # for every other consumer of this stack, it is a log line where
        # there used to be dead air. The value is measured, not tuned; see
        # the constant.
        "transcription_timeout": TRANSCRIPTION_TIMEOUT_SECONDS,
        "turn_handling": {
            "turn_detection": turn_detection,
            "endpointing": {
                "mode": "fixed",
                # Without a trained turn detector this is the ONLY endpointing
                # lever the pinned runtime has. ``audio_recognition.py`` sets
                # ``endpointing_delay = self._endpointing.min_delay`` (:1494)
                # and raises it to ``max_delay`` (:1556) only inside the branch
                # where a detector returned a probability below its unlikely
                # threshold -- so for a VAD-only language ``max_delay`` never
                # applies and the hold is entirely ``min_delay``.
                #
                # 0.35 s was a browser-demo value on a phone line. It commits
                # the turn while a caller is still mid-thought, which is what
                # produced turns that logged as a single word and answers that
                # arrived on top of the agent. 0.8 s is the platform default
                # for every language with no detector; an agent that wants
                # something else sets ``silence_hold_ms``, honoured in any
                # language by ``session_options_with_managed_pstn_turn_taking``.
                "min_delay": 0.2 if turn_detector_supported else 0.8,
                # The trained model is the primary decision when supported;
                # 600 ms is a bounded fallback for uncertain short phrases.
                # Left at 1.5 s for the detector-less case, where it is inert:
                # it is what the pair would use if a detector were ever added,
                # not a hold this stack applies today.
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
        stt_candidate_locales=tuple(configured_stt_locales),
        stt_construction=dict(stt_construction),
        _admission=(
            _InitialStackAdmission(profile, registry.revision, call_owner, _ADMISSION_AUTHORITY)
            if call_owner is not None else None
        ),
        tts_request_activity=synthesis.tts_request_activity,
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        session_options=session_options,
        safe_identity={
            "language": profile.language,
            "stt": (
                # A socket opened on a candidate list has no single tag to
                # name, so the identity names the whole set. An identity that
                # said "si-LK" for a stream listening to two locales would be
                # a diagnostic that lies about which call this was, and this
                # line is what the founder call is read from.
                f"{stt_registry}/{profile.stt_model}/{'+'.join(configured_stt_locales)}"
                if configured_stt_locales
                else f"google/{profile.stt_model or 'chirp_2'}/{profile.stt_language}"
                if stt_registry == "google"
                else f"soniox/stt-rt-v5/{profile.stt_language}"
                if stt_registry == "soniox"
                # The same set, named the same way, when it came from a frozen
                # follow-the-caller publication rather than from the agent's
                # configured languages above.
                else f"azure/{profile.stt_model}/{'+'.join(simultaneous_azure_locales)}"
                if stt_registry == "azure" and simultaneous_azure_locales is not None
                else f"azure/{profile.stt_model}/{azure_stt_language}"
                if stt_registry == "azure"
                else f"deepgram/nova-3/{simultaneous_stt_language or profile.stt_language}"
            ),
            "llm": f"gemini/{profile.llm_model}",
            "tts": f"{tts_registry}/{profile.tts_model}",
            "tts_transport": (
                "chunked_http"
                if tts_registry in {"gemini_tts", "azure_tts"}
                # Quarantined ElevenLabs models synthesize over chunked HTTP
                # behind the StreamAdapter; recording "realtime_websocket"
                # here would send the next tail investigation to the wrong
                # transport.
                or (
                    tts_registry == "elevenlabs"
                    and eleven_stream_transport_clips_tail(profile.tts_model)
                )
                else "realtime_websocket"
            ),
            "voice_id": profile.voice_id,
            # The delivery profile this stack runs with, and whether the
            # provider/model had a control for it, so a listening test can be
            # matched to the settings the call actually sent.
            "tts_delivery": (
                "default"
                if delivery.profile is None
                else f"{delivery.profile}/{delivery.control}"
            ),
            # The tag synthesis actually opened with, for the same reason the
            # STT identity names its own: the profile carries the bare product
            # language, and what Azure was told is the thing worth recording.
            **({"tts_language": azure_tts_language} if azure_tts_language else {}),
            "turn_detection": endpointing,
            "barge_in": "enabled",
        },
        tts_unit_meter=synthesis.tts_unit_meter,
        soniox_usage_tracker=soniox_usage_tracker,
    )

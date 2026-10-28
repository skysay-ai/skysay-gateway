# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Delivery profiles: one table, one resolver, read by both deployment units.

A customer picks how an agent's speech is PERFORMED -- ``natural``, ``lively``,
``calm`` or ``precise`` -- and this module turns that choice into the bounded,
allowlisted provider settings each TTS provider/model actually honours.  The
voice worker applies the result when it builds the provider adapter; the API's
voice preview applies the same result to its provider request bodies.  Both
import this module from the packaged capability directory, so a preview and
the call it auditions for cannot disagree about what a profile means.

Nothing here is a universal slider.  ``delivery_profiles.json`` lists, per
provider and model, which control the provider exposes and the exact values a
profile maps to.  A provider without a truthful control is listed as
``unsupported`` and the resolution says so; a profile never becomes provider
control syntax by any other path.  An absent profile resolves to ``none`` --
no settings sent -- which is exactly the behaviour every existing agent had
before profiles existed.

This file is pure standard library on purpose: it must import inside the
API wheel, the worker image and a livekit-less test environment alike.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DELIVERY_PROFILES: tuple[str, ...] = ("natural", "lively", "calm", "precise")

#: What the pinned Gemini TTS plugin (livekit-plugins-google 1.7.1,
#: ``beta.gemini_tts.DEFAULT_INSTRUCTIONS``) sends in front of every reply
#: when no instruction is configured -- i.e. what every existing Gemini agent's
#: calls already say.  The voice preview builds the identical prompt for an
#: unset profile so the audition matches the call; the worker image's build
#: probe asserts this string equals the plugin's constant.
GEMINI_DEFAULT_INSTRUCTION = (
    "Say the text with a proper tone, don't omit or add any words"
)

#: Controls a table entry may name, and what its per-profile value must be.
_CONTROLS = ("voice_settings", "instruction", "speed", "unsupported")
_VOICE_SETTINGS_BOUNDS: dict[str, tuple[float, float]] = {
    "stability": (0.0, 1.0),
    "similarity_boost": (0.0, 1.0),
    "style": (0.0, 1.0),
    # The plugin documents 0.8 - 1.2; the table stays inside it deliberately.
    "speed": (0.8, 1.2),
}
_SPEED_BOUNDS: dict[str, tuple[float, float]] = {
    "soniox": (0.7, 1.3),
    "cartesia": (0.6, 2.0),
}
_INSTRUCTION_MAX_CHARS = 300
#: An instruction is spliced in front of a quoted transcript; these would let
#: it escape the quoting or read as a second line of speech.
_INSTRUCTION_FORBIDDEN = ('"', "\n", "\r")


class DeliveryTableError(ValueError):
    """The packaged delivery table is malformed."""


@dataclass(frozen=True)
class DeliveryResolution:
    """What a provider/model does for a profile.

    ``control`` is one of ``none`` (no profile requested), ``unsupported`` (a
    profile was requested but this provider/model has no truthful control for
    it), ``voice_settings`` (ElevenLabs: ``settings`` is the exact
    ``voice_settings`` mapping), ``instruction`` (Gemini TTS: ``settings`` is
    the bounded synthesis instruction) or ``speed`` (Soniox, Cartesia:
    ``settings`` is the speaking-rate multiplier).
    """

    profile: str | None
    provider: str
    model: str
    control: str
    settings: Any = None

    @property
    def applied(self) -> bool:
        return self.control in ("voice_settings", "instruction", "speed")


def _root() -> Path:
    return Path(__file__).resolve().parent


def normalize_delivery_profile(value: Any) -> str | None:
    """Return the canonical profile name, ``None`` for absent, or raise ValueError."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("delivery must be a string")
    profile = value.strip().lower()
    if not profile:
        return None
    if profile not in DELIVERY_PROFILES:
        raise ValueError("delivery must be one of " + ", ".join(DELIVERY_PROFILES))
    return profile


def gemini_delivery_prompt(instruction: str, text: str) -> str:
    """The exact prompt the pinned Gemini TTS adapter builds from an instruction.

    Mirrors ``livekit.plugins.google.beta.gemini_tts.ChunkedStream``:
    ``f'{instructions}:\\n"{input_text}"'``.  The preview builds its request
    body with this function so the bytes match the worker's.
    """

    return f'{instruction}:\n"{text}"'


#: Gemini TTS models that read their input as a TRANSCRIPT.
#:
#: ``gemini-3.8-flash-lite-tts`` speaks whatever it is sent: the 2.5/3.1
#: instruction prefix above was spoken aloud on 5 of 5 tries, doubling each
#: clip.  So these models get the sentence alone, never an instruction, and no
#: delivery profile maps onto them (their ``delivery_profiles.json`` entry is
#: ``unsupported``).  They also answer a one-shot ``generateContent`` with a
#: WAV file unless the request names the raw format, so both the worker and
#: the voice preview ask for it explicitly with
#: ``generationConfig.responseFormat.audio`` (``responseMimeType: audio/l16``
#: is refused with HTTP 400).  One table, read by the worker adapter
#: (``gemini_tts.py``) and the preview
#: (``voice_preview_rest.render_gemini``), so a Listen and a call send the
#: same request.
GEMINI_TRANSCRIPT_MODELS: frozenset[str] = frozenset({"gemini-3.8-flash-lite-tts"})

#: The raw-PCM request both surfaces add for a transcript model.  24 kHz is
#: the rate the adapter's emitter and the preview's WAV wrapper already use.
GEMINI_RAW_AUDIO_FORMAT: dict[str, Any] = {"mimeType": "AUDIO_L16", "sampleRate": 24000}


def gemini_reads_transcript(model: str) -> bool:
    """Whether this Gemini TTS model is sent the sentence alone (no instruction)."""

    return str(model or "").strip() in GEMINI_TRANSCRIPT_MODELS


def gemini_raw_audio_generation_config(model: str) -> dict[str, Any]:
    """The ``generationConfig`` keys that ask this model for raw PCM, or ``{}``.

    Empty for 2.5/3.1: their request is unchanged, and they already stream
    headerless PCM.  A fresh dict each call, so a caller merging it into a
    request body cannot mutate the shared constant.
    """

    if not gemini_reads_transcript(model):
        return {}
    return {"responseFormat": {"audio": dict(GEMINI_RAW_AUDIO_FORMAT)}}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeliveryTableError(message)


def _number(value: Any, *, where: str, bounds: tuple[float, float]) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{where} must be a finite number",
    )
    number = float(value)
    low, high = bounds
    _require(low <= number <= high, f"{where} must be between {low:g} and {high:g}")
    return number


def _settings_contract(entry: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """The per-model rule a ``voice_settings`` entry must satisfy.

    Every ElevenLabs model is not the same instrument: the streaming models
    take stability, similarity, style and speed, while eleven_v3's dialogue
    transport honours stability alone and only at its three documented
    values.  The table states that per model and the loader enforces it, so a
    profile cannot ship a setting the transport would drop or reject.
    """

    contract = entry.get("settings_contract")
    _require(
        isinstance(contract, dict),
        f"{where}.settings_contract is required for voice_settings",
    )
    required = contract.get("required")
    allowed = contract.get("allowed")
    _require(
        isinstance(required, list)
        and bool(required)
        and all(k in _VOICE_SETTINGS_BOUNDS for k in required),
        f"{where}.settings_contract.required must name allowlisted voice settings",
    )
    _require(
        isinstance(allowed, list)
        and set(required) <= set(allowed) <= set(_VOICE_SETTINGS_BOUNDS),
        f"{where}.settings_contract.allowed must cover required and stay allowlisted",
    )
    discrete = contract.get("discrete", {})
    _require(
        isinstance(discrete, dict),
        f"{where}.settings_contract.discrete must be an object",
    )
    for key, values in discrete.items():
        _require(
            key in allowed,
            f"{where}.settings_contract.discrete.{key} is not an allowed setting",
        )
        _require(
            isinstance(values, list)
            and bool(values)
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
            ),
            f"{where}.settings_contract.discrete.{key} must list numbers",
        )
    return {
        "required": tuple(required),
        "allowed": frozenset(allowed),
        "discrete": {
            key: tuple(float(v) for v in values) for key, values in discrete.items()
        },
    }


def _validate_entry(provider: str, model: str, entry: Any) -> dict[str, Any]:
    where = f"providers.{provider}.{model}"
    _require(isinstance(entry, dict), f"{where} must be an object")
    _require(model != "*", f"{where}: wildcard models are not allowed; list each model")
    control = entry.get("control")
    _require(
        control in _CONTROLS, f"{where}.control must be one of {', '.join(_CONTROLS)}"
    )
    if control == "unsupported":
        _require(
            set(entry) == {"control"}, f"{where} unsupported entries carry no profiles"
        )
        return {"control": control, "profiles": {}}
    profiles = entry.get("profiles")
    _require(
        isinstance(profiles, dict) and set(profiles) == set(DELIVERY_PROFILES),
        f"{where}.profiles must map every delivery profile exactly once",
    )
    contract = (
        _settings_contract(entry, where=where) if control == "voice_settings" else None
    )
    normalized: dict[str, Any] = {}
    for profile, value in profiles.items():
        pwhere = f"{where}.profiles.{profile}"
        if control == "voice_settings":
            assert contract is not None
            _require(
                isinstance(value, dict) and bool(value),
                f"{pwhere} must be a non-empty object",
            )
            _require(
                set(value) <= contract["allowed"],
                f"{pwhere} names a setting this model does not honour",
            )
            _require(
                all(key in value for key in contract["required"]),
                f"{pwhere} must set {', '.join(contract['required'])}",
            )
            settings: dict[str, float] = {}
            for key, raw in value.items():
                settings[key] = _number(
                    raw, where=f"{pwhere}.{key}", bounds=_VOICE_SETTINGS_BOUNDS[key]
                )
                if key in contract["discrete"]:
                    _require(
                        settings[key] in contract["discrete"][key],
                        f"{pwhere}.{key} must be one of the model's documented values",
                    )
            normalized[profile] = settings
        elif control == "instruction":
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"{pwhere} must be a non-empty string",
            )
            _require(
                len(value) <= _INSTRUCTION_MAX_CHARS,
                f"{pwhere} exceeds {_INSTRUCTION_MAX_CHARS} characters",
            )
            _require(
                not any(char in value for char in _INSTRUCTION_FORBIDDEN),
                f"{pwhere} must not contain quotes or line breaks",
            )
            normalized[profile] = value.strip()
        else:  # speed
            bounds = _SPEED_BOUNDS.get(provider)
            _require(
                bounds is not None,
                f"{where} has no speed bounds for provider {provider}",
            )
            normalized[profile] = _number(value, where=pwhere, bounds=bounds)
    return {"control": control, "profiles": normalized}


class DeliveryTable:
    """The validated delivery table and its lookups."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        _require(isinstance(raw, Mapping), "delivery table must be an object")
        _require(
            raw.get("schema_version") == 1, "delivery table schema_version must be 1"
        )
        version = raw.get("table_version")
        _require(
            isinstance(version, str) and version.strip(),
            "delivery table needs a table_version",
        )
        self.version: str = version.strip()
        profiles = raw.get("profiles")
        _require(
            isinstance(profiles, dict) and tuple(profiles) == DELIVERY_PROFILES,
            "delivery table profiles must list natural, lively, calm, precise in that order",
        )
        for name, meta in profiles.items():
            _require(
                isinstance(meta, dict)
                and isinstance(meta.get("label"), str)
                and isinstance(meta.get("summary"), str),
                f"profiles.{name} needs a label and a summary",
            )
        self.profiles: dict[str, dict[str, str]] = {
            name: {"label": meta["label"], "summary": meta["summary"]}
            for name, meta in profiles.items()
        }
        providers = raw.get("providers")
        _require(
            isinstance(providers, dict) and providers, "delivery table needs providers"
        )
        self._providers: dict[str, dict[str, dict[str, Any]]] = {}
        for provider, models in providers.items():
            _require(
                isinstance(models, dict) and models,
                f"providers.{provider} must map models",
            )
            self._providers[provider] = {
                str(model): _validate_entry(provider, str(model), entry)
                for model, entry in models.items()
            }

    def entry(self, provider: str, model: str) -> dict[str, Any] | None:
        """The exact provider/model entry; no wildcard, so an unlisted model is unsupported."""

        models = self._providers.get(str(provider or "").strip().lower())
        if not models:
            return None
        return models.get(str(model or "").strip())

    def resolve(
        self, provider: str, model: str, profile: str | None
    ) -> DeliveryResolution:
        """Resolve one profile for one provider/model; never raises on lookup.

        ``provider`` is the capability registry name (``gemini_tts``,
        ``azure_tts``, ``elevenlabs``, ...).  An unknown provider or model
        with a requested profile is ``unsupported``: the request is honoured
        by sending nothing rather than by guessing a control.
        """

        canonical = normalize_delivery_profile(profile)
        provider_key = str(provider or "").strip().lower()
        model_key = str(model or "").strip()
        if canonical is None:
            return DeliveryResolution(None, provider_key, model_key, "none")
        entry = self.entry(provider_key, model_key)
        if entry is None or entry["control"] == "unsupported":
            return DeliveryResolution(canonical, provider_key, model_key, "unsupported")
        return DeliveryResolution(
            canonical,
            provider_key,
            model_key,
            entry["control"],
            entry["profiles"][canonical],
        )


@lru_cache(maxsize=1)
def load_delivery_table() -> DeliveryTable:
    path = _root() / "delivery_profiles.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DeliveryTableError(
            f"delivery table unreadable: {type(exc).__name__}"
        ) from exc
    return DeliveryTable(raw)


def resolve_delivery(
    provider: str, model: str, profile: str | None
) -> DeliveryResolution:
    return load_delivery_table().resolve(provider, model, profile)


__all__ = [
    "DELIVERY_PROFILES",
    "GEMINI_DEFAULT_INSTRUCTION",
    "DeliveryResolution",
    "DeliveryTable",
    "DeliveryTableError",
    "GEMINI_RAW_AUDIO_FORMAT",
    "GEMINI_TRANSCRIPT_MODELS",
    "gemini_delivery_prompt",
    "gemini_raw_audio_generation_config",
    "gemini_reads_transcript",
    "load_delivery_table",
    "normalize_delivery_profile",
    "resolve_delivery",
]

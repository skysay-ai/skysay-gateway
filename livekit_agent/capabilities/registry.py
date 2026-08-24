# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict immutable reader for the checked-in voice capability artifact.

This module is intentionally dependency-free and is copied verbatim into the
API image as well as the separately built voice image.  It validates the
source document before exposing any selection, therefore a malformed or
partially edited capability file cannot become a permissive runtime fallback.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class VoiceCapabilityError(RuntimeError):
    """The capability artifact is missing, malformed, or internally inconsistent."""


_COMPONENTS = frozenset({"stt", "tts"})
_BILLING_UNITS = frozenset({"audio_second", "character", "utf8_byte", "token"})
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_VERSION = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def _root() -> Path:
    return Path(__file__).resolve().parent


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoiceCapabilityError(message)


def _required_object(value: Any, *, where: str, keys: set[str]) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    unknown = set(value) - keys
    _require(not unknown, f"{where} has unsupported keys: {sorted(unknown)!r}")
    return value


def _strings(value: Any, *, where: str, pattern: re.Pattern[str] | None = None, nonempty: bool = True) -> tuple[str, ...]:
    _require(isinstance(value, list), f"{where} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        _require(isinstance(item, str) and item != "", f"{where}[{index}] must be a nonempty string")
        if pattern is not None:
            _require(pattern.fullmatch(item) is not None, f"{where}[{index}] is invalid")
        result.append(item)
    _require(bool(result) or not nonempty, f"{where} must not be empty")
    _require(len(result) == len(set(result)), f"{where} contains duplicates")
    return tuple(result)


def _validate_metadata(value: Any, *, where: str) -> None:
    metadata = _required_object(
        value,
        where=where,
        keys={"description", "tags", "gender", "accent_locale", "provenance"},
    )
    _require("provenance" in metadata, f"{where}.provenance is required")
    if "description" in metadata:
        _require(isinstance(metadata["description"], str), f"{where}.description must be text")
    if "tags" in metadata:
        _strings(metadata["tags"], where=f"{where}.tags", nonempty=False)
    if "gender" in metadata:
        _require(metadata["gender"] in {"female", "male", "nonbinary"}, f"{where}.gender is invalid")
    if "accent_locale" in metadata:
        _require(isinstance(metadata["accent_locale"], str) and _LANGUAGE.fullmatch(metadata["accent_locale"]) is not None, f"{where}.accent_locale is invalid")
    provenance = _required_object(
        metadata["provenance"],
        where=f"{where}.provenance",
        keys={"source", "observed_at"},
    )
    _require(isinstance(provenance.get("source"), str) and bool(provenance["source"]), f"{where}.provenance.source is required")
    _require(isinstance(provenance.get("observed_at"), str) and _VERSION.fullmatch(provenance["observed_at"]) is not None, f"{where}.provenance.observed_at is invalid")


_TOKEN_DIMENSIONS = {
    "stt": frozenset({"input_audio_tokens", "output_text_tokens"}),
    "tts": frozenset({"input_text_tokens", "output_audio_tokens"}),
}

# These adapters existed before the immutable registry.  The artifact may
# declare that they remain compatible, but it must not be able to classify a
# newly advertised language as "legacy" and thereby evade the exact runnable
# STT/TTS contract below.
_FROZEN_LEGACY_RUNTIME_LANGUAGES = frozenset({"en", "et", "si"})
_MAX_RUNTIME_LANGUAGES = 16


def _validate_billing(
    value: Any, *, where: str, component: str, selectable: bool
) -> None:
    """Validate one exact native billing contract.

    A single token price is not truthful for Soniox: STT bills input audio and
    output text independently, while TTS bills input text and output audio
    independently.  Keep those dimensions in the immutable artifact rather
    than collapsing provider token usage back into a duration estimate.
    """
    billing = _required_object(
        value,
        where=where,
        keys={
            "unit",
            "price_micro_usd_per_unit",
            "reserve_units_per_minute",
            "rate_dimensions",
            "provider_price_basis",
        },
    )
    _require(billing.get("unit") in _BILLING_UNITS, f"{where}.unit is invalid")
    is_token = billing["unit"] == "token"
    if not is_token:
        _require(
            {"unit", "price_micro_usd_per_unit", "reserve_units_per_minute"}.issubset(billing),
            f"{where} is incomplete",
        )
        _require("rate_dimensions" not in billing, f"{where}.rate_dimensions is only valid for token billing")
        for key in ("price_micro_usd_per_unit", "reserve_units_per_minute"):
            _require(
                isinstance(billing[key], int)
                and not isinstance(billing[key], bool)
                and billing[key] >= 0,
                f"{where}.{key} is invalid",
            )
        if selectable:
            _require(billing["price_micro_usd_per_unit"] > 0, f"{where} price is missing")
            _require(billing["reserve_units_per_minute"] > 0, f"{where} reserve is missing")
    else:
        _require(
            set(billing).issuperset({"unit", "rate_dimensions"})
            and "price_micro_usd_per_unit" not in billing
            and "reserve_units_per_minute" not in billing,
            f"{where} token dimensions are incomplete",
        )
        dimensions = billing["rate_dimensions"]
        _require(isinstance(dimensions, list), f"{where}.rate_dimensions must be an array")
        expected = _TOKEN_DIMENSIONS[component]
        observed: set[str] = set()
        for index, dimension in enumerate(dimensions):
            item_where = f"{where}.rate_dimensions[{index}]"
            item = _required_object(
                dimension,
                where=item_where,
                keys={"name", "price_micro_usd_per_unit", "reserve_units_per_minute"},
            )
            _require(set(item) == {"name", "price_micro_usd_per_unit", "reserve_units_per_minute"}, f"{item_where} is incomplete")
            name = item.get("name")
            _require(isinstance(name, str) and name in expected, f"{item_where}.name is invalid")
            _require(name not in observed, f"{where}.rate_dimensions contains duplicate names")
            observed.add(name)
            for key in ("price_micro_usd_per_unit", "reserve_units_per_minute"):
                _require(
                    isinstance(item[key], int)
                    and not isinstance(item[key], bool)
                    and item[key] >= 0,
                    f"{item_where}.{key} is invalid",
                )
                if selectable:
                    _require(item[key] > 0, f"{item_where}.{key} is missing")
        _require(observed == expected, f"{where}.rate_dimensions do not match the native {component} token contract")
    if "provider_price_basis" in billing:
        _require(
            isinstance(billing["provider_price_basis"], str) and bool(billing["provider_price_basis"]),
            f"{where}.provider_price_basis is invalid",
        )


def _validate_capability(value: Any, *, index: int) -> None:
    where = f"capabilities[{index}]"
    capability = _required_object(
        value,
        where=where,
        keys={
            "component", "provider", "registry_name", "model", "display", "languages",
            "streaming", "runnable", "selectable", "catalog", "typical_latency_ms",
            "credential", "billing", "voices",
        },
    )
    required = {
        "component", "provider", "registry_name", "model", "display", "languages",
        "streaming", "runnable", "selectable", "catalog", "typical_latency_ms", "credential", "billing",
    }
    _require(not (required - set(capability)), f"{where} is missing required keys")
    _require(capability["component"] in _COMPONENTS, f"{where}.component is invalid")
    for key in ("provider", "registry_name"):
        _require(isinstance(capability[key], str) and _ID.fullmatch(capability[key]) is not None, f"{where}.{key} is invalid")
    for key in ("model", "display"):
        _require(isinstance(capability[key], str) and bool(capability[key]), f"{where}.{key} is required")
    _strings(capability["languages"], where=f"{where}.languages", pattern=_LANGUAGE)
    for key in ("streaming", "runnable", "selectable", "catalog"):
        _require(isinstance(capability[key], bool), f"{where}.{key} must be boolean")
    _require(not capability["selectable"] or capability["runnable"], f"{where} cannot be selectable without a runtime")
    _require(
        isinstance(capability["typical_latency_ms"], int)
        and not isinstance(capability["typical_latency_ms"], bool)
        and capability["typical_latency_ms"] >= 0,
        f"{where}.typical_latency_ms is invalid",
    )
    credential = _required_object(
        capability["credential"],
        where=f"{where}.credential",
        keys={"managed_env", "byok_supported", "validation"},
    )
    _require(set(credential) == {"managed_env", "byok_supported", "validation"}, f"{where}.credential is incomplete")
    _require(isinstance(credential["managed_env"], str) and _ID.fullmatch(credential["managed_env"].lower()) is not None and credential["managed_env"] == credential["managed_env"].upper(), f"{where}.credential.managed_env is invalid")
    _require(isinstance(credential["byok_supported"], bool), f"{where}.credential.byok_supported must be boolean")
    _require(credential["validation"] in {"soniox_models", "fish_wallet"}, f"{where}.credential.validation is invalid")
    # Native provider evidence must be locked to its immutable billing unit.
    # Otherwise a syntactically valid artifact could make worker admission
    # accept Soniox duration/character estimates or Fish character/token usage
    # even though settlement has only exact token/UTF-8-byte evidence.
    provider_unit = {
        "soniox_models": ("soniox", "token"),
        "fish_wallet": ("fishaudio", "utf8_byte"),
    }[str(credential["validation"])]
    _require(
        capability["provider"] == provider_unit[0],
        f"{where}.credential.validation does not match its provider",
    )
    _require(
        isinstance(capability["billing"], dict) and capability["billing"].get("unit") == provider_unit[1],
        f"{where}.billing must use {provider_unit[1]} units for its provider",
    )
    # A row may remain runnable after it is no longer selectable so an
    # already-pinned call stack can keep serving. It must still carry a
    # complete positive native billing contract: the worker's runtime check is
    # intentionally allowed to admit it and settlement must never guess.
    _validate_billing(
        capability["billing"],
        where=f"{where}.billing",
        component=str(capability["component"]),
        selectable=bool(capability["selectable"] or capability["runnable"]),
    )
    voices = capability.get("voices", [])
    _require(isinstance(voices, list), f"{where}.voices must be an array")
    voice_ids: set[str] = set()
    for voice_index, voice in enumerate(voices):
        voice_where = f"{where}.voices[{voice_index}]"
        voice_data = _required_object(voice, where=voice_where, keys={"id", "name", "languages", "recommended", "metadata"})
        _require({"id", "name", "languages", "metadata"}.issubset(voice_data), f"{voice_where} is incomplete")
        _require(isinstance(voice_data["id"], str) and bool(voice_data["id"]), f"{voice_where}.id is invalid")
        _require(voice_data["id"] not in voice_ids, f"{where}.voices contains duplicate ids")
        voice_ids.add(voice_data["id"])
        _require(isinstance(voice_data["name"], str) and bool(voice_data["name"]), f"{voice_where}.name is invalid")
        languages = _strings(voice_data["languages"], where=f"{voice_where}.languages", pattern=_LANGUAGE)
        _require(set(languages).issubset(set(capability["languages"])), f"{voice_where}.languages are not supported by its capability")
        if "recommended" in voice_data:
            _require(isinstance(voice_data["recommended"], bool), f"{voice_where}.recommended must be boolean")
        _validate_metadata(voice_data["metadata"], where=f"{voice_where}.metadata")


def _validate_tts_voice_inventory(capabilities: list[Any]) -> None:
    """Reject ambiguous or unusable TTS selection data at artifact load time.

    ``default_voice`` and ``default_tts_capability`` retain defensive checks for
    callers holding an already-loaded registry. The artifact itself must not,
    however, boot successfully with a selectable/runnable TTS model that has
    no selectable voice for one of its declared languages, or with ambiguous
    recommendations that only fail later while a draft is being created.
    """
    default_provider_by_language: dict[str, int] = {}
    for index, capability in enumerate(capabilities):
        if capability["component"] != "tts":
            continue
        where = f"capabilities[{index}]"
        voices = capability.get("voices", [])
        requires_inventory = bool(capability["selectable"] or capability["runnable"])
        if requires_inventory:
            _require(
                bool(voices),
                f"{where}.voices must not be empty for a runnable or selectable TTS capability",
            )
        for language in capability["languages"]:
            language_voices = [voice for voice in voices if language in voice["languages"]]
            if requires_inventory:
                _require(
                    bool(language_voices),
                    f"{where}.voices has no approved voice for language {language!r}",
                )
            recommended = [voice for voice in language_voices if voice.get("recommended") is True]
            _require(
                len(recommended) <= 1,
                f"{where}.voices has multiple recommended voices for language {language!r}",
            )
            if capability["catalog"] and capability["selectable"] and recommended:
                previous = default_provider_by_language.get(language)
                _require(
                    previous is None,
                    f"multiple catalog TTS capabilities declare a default voice for language {language!r}",
                )
                default_provider_by_language[language] = index


@dataclass(frozen=True)
class VoiceCapabilityRegistry:
    """One validated, immutable capability revision."""

    data: Mapping[str, Any]
    revision: str

    @property
    def registry_version(self) -> str:
        return str(self.data["registry_version"])

    @property
    def runnable_registry_names(self) -> frozenset[str]:
        """Names of worker adapters that legacy stored stacks may resolve.

        This is intentionally a provider-adapter inventory, not a claim that
        every model under an adapter is available or billable.  Registry-backed
        Soniox/Fish selections must use :meth:`is_capability_runnable`, which
        checks the exact component/provider/model record and its language.
        The extra historical adapter names preserve pinned English, Estonian,
        and Sinhala configurations that predate this registry.
        """
        return frozenset(str(item) for item in self.data["runnable_registry_names"])

    @property
    def capabilities(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["capabilities"])

    @property
    def runtime_languages(self) -> tuple[str, ...]:
        """Languages the packaged worker can resolve without another list.

        This covers both the historical, pinned production paths and newly
        catalogued capabilities.  The API uses it to reject a language before
        it can be persisted; the voice image uses the same artifact before it
        constructs a provider client.
        """
        return tuple(self.data["runtime_languages"])

    @property
    def legacy_runtime_languages(self) -> frozenset[str]:
        """Languages whose pre-registry stacks remain runnable by compatibility.

        This is intentionally data in the immutable artifact, rather than a
        second worker/control-plane language list.  A newly registered
        language must resolve every component through an exact capability, so
        an installed historical adapter cannot accidentally admit it without
        the registry's provider, credential, voice, and billing contract.
        """
        return frozenset(str(item) for item in self.data["legacy_runtime_languages"])

    def is_registry_name_runnable(self, registry_name: str) -> bool:
        """Whether a historical adapter name remains constructible.

        Callers must not use this provider-wide fallback for a model represented
        by ``capabilities``; a provider can have an installed adapter while one
        of its models is unavailable, unselectable, or unmetered.
        """
        name = str(registry_name or "")
        return name in self.runnable_registry_names and not any(
            str(capability["registry_name"]) == name for capability in self.capabilities
        )

    def owns_component_provider(self, component: str, provider: str) -> bool:
        """Whether exact model admission is owned for this provider component.

        Historical adapter names are intentionally provider-wide only until a
        component/provider pair is represented here.  Once it is represented,
        every model for that pair must be an explicit artifact decision: a
        legacy adapter must not silently admit a future typo or withdrawn
        model.  Keeping this predicate beside ``capability`` lets the API and
        separately packaged worker enforce the same transition rule.
        """

        normalized_component = str(component or "").strip().lower()
        normalized_provider = str(provider or "").strip().lower()
        return any(
            str(item["component"]).lower() == normalized_component
            and str(item["provider"]).lower() == normalized_provider
            for item in self.capabilities
        )

    def is_capability_runnable(
        self,
        component: str,
        provider: str,
        model: str,
        *,
        language: str | None = None,
    ) -> bool:
        """Return exact-model runnable state from the immutable artifact."""
        capability = self.capability(component, provider, model)
        return bool(
            capability is not None
            and capability["runnable"]
            and (language is None or language in capability["languages"])
        )

    def capability(self, component: str, provider: str, model: str) -> Mapping[str, Any] | None:
        key = (str(component or ""), str(provider or "").lower(), str(model or ""))
        for capability in self.capabilities:
            if (capability["component"], capability["provider"], capability["model"]) == key:
                return capability
        return None

    def catalog_capabilities(self, *, component: str | None = None, language: str | None = None) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            capability
            for capability in self.capabilities
            if capability["catalog"]
            and (component is None or capability["component"] == component)
            and (language is None or language in capability["languages"])
        )

    def voices(self, provider: str, model: str, *, language: str | None = None) -> tuple[Mapping[str, Any], ...]:
        capability = self.capability("tts", provider, model)
        if capability is None:
            return ()
        return tuple(
            voice
            for voice in capability.get("voices", ())
            if language is None or language in voice["languages"]
        )

    def _validate_voice(
        self,
        provider: str,
        model: str,
        voice_id: str,
        *,
        language: str,
        require_selectable: bool,
    ) -> Mapping[str, Any]:
        capability = self.capability("tts", provider, model)
        if capability is None or not capability["runnable"]:
            raise VoiceCapabilityError("voice capability is not runnable")
        if require_selectable and not capability["selectable"]:
            raise VoiceCapabilityError("voice capability is not selectable")
        if language not in capability["languages"]:
            raise VoiceCapabilityError("voice capability does not support the requested language")
        for voice in self.voices(provider, model, language=language):
            if voice["id"] == voice_id:
                return voice
        raise VoiceCapabilityError("voice id is absent from the approved inventory")

    def validate_voice(self, provider: str, model: str, voice_id: str, *, language: str) -> Mapping[str, Any]:
        """Validate a fresh picker selection against the approved inventory."""

        return self._validate_voice(
            provider, model, voice_id, language=language, require_selectable=True
        )

    def validate_runnable_voice(
        self, provider: str, model: str, voice_id: str, *, language: str
    ) -> Mapping[str, Any]:
        """Validate an already-pinned voice without reopening the picker.

        A provider can withdraw a model from new selection while continuing to
        serve the exact billed model on existing credentials.  Runtime must
        still require its runnable capability, language, and inventory ID;
        only the fresh-selection flag is intentionally relaxed.
        """

        return self._validate_voice(
            provider, model, voice_id, language=language, require_selectable=False
        )

    def default_voice(self, provider: str, model: str, *, language: str) -> Mapping[str, Any] | None:
        """Return the one explicit recommended inventory item, if any.

        A provider with no recommended voice deliberately returns ``None`` so
        callers never invent a gender, name, locale, or default identifier.
        """
        voices = self.voices(provider, model, language=language)
        recommended = tuple(voice for voice in voices if voice.get("recommended") is True)
        if len(recommended) > 1:
            raise VoiceCapabilityError("voice inventory has multiple recommended voices for one language")
        return recommended[0] if recommended else None

    def default_tts_capability(self, *, language: str) -> Mapping[str, Any] | None:
        """Return the one catalogued TTS capability with a declared default voice.

        A recommendation belongs to the provider's inventory, not to a
        hand-maintained UI ordering rule.  Selecting the enclosing capability
        from the same immutable artifact keeps a fresh draft and a server-side
        default aligned.  More than one declared provider default for a
        language is ambiguous, so reject the artifact rather than selecting by
        incidental JSON order.
        """
        candidates = tuple(
            capability
            for capability in self.capabilities
            if capability["component"] == "tts"
            and capability["catalog"]
            and capability["selectable"]
            and language in capability["languages"]
            and self.default_voice(
                str(capability["provider"]), str(capability["model"]), language=language
            ) is not None
        )
        if len(candidates) > 1:
            raise VoiceCapabilityError("multiple TTS capabilities declare a default voice for one language")
        return candidates[0] if candidates else None

    def validate_registry_component_billing(
        self,
        config: Mapping[str, Any],
        *,
        stt_capability: Mapping[str, Any] | None,
        tts_capability: Mapping[str, Any] | None,
    ) -> None:
        """Require a complete pinned billing identity for new provider stacks.

        Registry-backed adapters are introduced with native usage contracts, so
        they may not take the historical unmetered worker path when a corrupt
        or forged per-call config reaches the voice host. Existing EN/ET/SI
        profiles remain compatible because neither selected component resolves
        through this new registry.
        """
        if stt_capability is None and tts_capability is None:
            return
        billing = config.get("component_billing")
        if not isinstance(billing, Mapping) or billing.get("usage_required") is not True:
            raise VoiceCapabilityError("registry-backed components require component-metered billing")
        components = billing.get("components")
        if not isinstance(components, list):
            raise VoiceCapabilityError("component billing configuration is incomplete")

        def selection_identity(
            name: str, *, capability: Mapping[str, Any] | None = None
        ) -> tuple[str, str]:
            selected = config.get(name)
            if not isinstance(selected, Mapping):
                return "", ""
            selected_provider = str(
                selected.get("provider") or selected.get("registry_name") or ""
            ).strip().lower()
            selected_model = str(selected.get("model") or "").strip()
            if capability is None:
                return selected_provider, selected_model

            # This method is deliberately a last line of defence for a
            # serialized realtime config.  Its production caller resolved the
            # capability from this same selection already, but do not let a
            # future caller accidentally pair a registry rate card with an
            # arbitrary model merely because its billing components agree with
            # one another.  ``provider`` may be omitted in favour of the
            # durable registry name, so accept either exact normalized value.
            capability_provider = str(capability.get("provider") or "").strip().lower()
            capability_registry = str(capability.get("registry_name") or "").strip().lower()
            capability_model = str(capability.get("model") or "").strip()
            if (
                selected_provider not in {capability_provider, capability_registry}
                or selected_model != capability_model
            ):
                raise VoiceCapabilityError("selected component does not match the capability billing identity")
            # The selector may identify an adapter by its durable registry
            # name, but component billing always pins the vendor/provider
            # identity that owns the native meter and rate card.
            return capability_provider, capability_model

        expected = {
            "stt": selection_identity("transcriber", capability=stt_capability),
            "llm": selection_identity("model"),
            "tts": selection_identity("voice", capability=tts_capability),
        }
        observed: dict[str, tuple[str, str]] = {}
        for component in components:
            if not isinstance(component, Mapping):
                raise VoiceCapabilityError("component billing configuration is incomplete")
            name = str(component.get("component") or "")
            if name not in expected or name in observed:
                raise VoiceCapabilityError("component billing configuration does not match the selected stack")
            observed[name] = (
                str(component.get("provider") or "").strip().lower(),
                str(component.get("model") or "").strip(),
            )
        if observed != expected or any(not all(identity) for identity in expected.values()):
            raise VoiceCapabilityError("component billing configuration does not match the selected stack")

    def managed_rate_contracts(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        result: dict[tuple[str, str, str], dict[str, Any]] = {}
        for capability in self.capabilities:
            # A model may be removed from the picker while retained for a
            # previously pinned stack. ``_validate_capability`` requires the
            # same complete native billing contract for either runnable or
            # selectable rows, so settlement must export both rather than
            # turn an already-admitted model into an unmetered hole.
            if not (capability["selectable"] or capability["runnable"]):
                continue
            billing = capability["billing"]
            contract: dict[str, Any] = {"unit": billing["unit"]}
            if billing["unit"] == "token":
                for dimension in billing["rate_dimensions"]:
                    name = str(dimension["name"])
                    singular = name[:-1] if name.endswith("s") else name
                    contract[f"price_micro_usd_per_{singular}"] = int(
                        dimension["price_micro_usd_per_unit"]
                    )
                    contract[f"reserve_{name}_per_minute"] = int(
                        dimension["reserve_units_per_minute"]
                    )
            else:
                contract.update(
                    {
                        "price_micro_usd_per_unit": int(billing["price_micro_usd_per_unit"]),
                        "reserve_units_per_minute": int(billing["reserve_units_per_minute"]),
                    }
                )
            result[(capability["component"], capability["provider"], capability["model"])] = contract
        return result

    def byok_contracts(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (capability["component"], capability["provider"])
            for capability in self.capabilities
            if capability["credential"]["byok_supported"]
        )


def _load(path: Path) -> VoiceCapabilityRegistry:
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceCapabilityError("voice capability source cannot be loaded") from exc
    _require(isinstance(data, dict), "voice capability source must be an object")
    expected_top_level = {
        "schema_version",
        "registry_version",
        "runtime_languages",
        "legacy_runtime_languages",
        "runnable_registry_names",
        "capabilities",
    }
    _require(set(data) == expected_top_level, "voice capability source has unsupported or missing top-level keys")
    _require(
        isinstance(data["schema_version"], int)
        and not isinstance(data["schema_version"], bool)
        and data["schema_version"] == 1,
        "voice capability schema version is unsupported",
    )
    _require(isinstance(data["registry_version"], str) and _VERSION.fullmatch(data["registry_version"]) is not None, "voice capability registry version is invalid")
    runtime_languages = _strings(data["runtime_languages"], where="runtime_languages", pattern=_LANGUAGE)
    _require(
        len(runtime_languages) <= _MAX_RUNTIME_LANGUAGES,
        "runtime_languages exceeds the bounded catalog response contract",
    )
    legacy_runtime_languages = _strings(
        data["legacy_runtime_languages"],
        where="legacy_runtime_languages",
        pattern=_LANGUAGE,
        nonempty=False,
    )
    _require(
        set(legacy_runtime_languages).issubset(set(runtime_languages)),
        "legacy_runtime_languages references a language absent from the worker runtime contract",
    )
    _require(
        frozenset(legacy_runtime_languages) == _FROZEN_LEGACY_RUNTIME_LANGUAGES,
        "legacy_runtime_languages must match the frozen historical compatibility set",
    )
    runnable_registry_names = set(
        _strings(data["runnable_registry_names"], where="runnable_registry_names", pattern=_ID)
    )
    _require(isinstance(data["capabilities"], list) and bool(data["capabilities"]), "capabilities must be a nonempty array")
    keys: set[tuple[str, str, str]] = set()
    capability_registry_names: set[str] = set()
    byok_by_component_provider: dict[tuple[str, str], bool] = {}
    for index, capability in enumerate(data["capabilities"]):
        _validate_capability(capability, index=index)
        key = (capability["component"], capability["provider"], capability["model"])
        _require(key not in keys, "capabilities contains duplicate component/provider/model identities")
        keys.add(key)
        capability_registry_names.add(str(capability["registry_name"]))
        component_provider = (capability["component"], capability["provider"])
        byok_supported = bool(capability["credential"]["byok_supported"])
        existing_byok = byok_by_component_provider.setdefault(component_provider, byok_supported)
        _require(
            existing_byok == byok_supported,
            "capabilities disagree about BYOK support for one component/provider",
        )
        _require(
            set(capability["languages"]).issubset(set(runtime_languages)),
            "capability references a language absent from the worker runtime contract",
        )
    # A language outside the explicit legacy compatibility set has no
    # provider-wide fallback.  Requiring both exact runnable components here
    # makes a partially edited artifact fail at image startup instead of
    # leaving a newly advertised language to be accidentally served by an
    # installed historical adapter.
    for language in set(runtime_languages) - set(legacy_runtime_languages):
        runnable_components = {
            str(capability["component"])
            for capability in data["capabilities"]
            if capability["runnable"] and language in capability["languages"]
        }
        _require(
            {"stt", "tts"}.issubset(runnable_components),
            "non-legacy runtime language lacks an exact runnable STT/TTS capability",
        )
    _require(
        not (capability_registry_names & runnable_registry_names),
        "registry-backed capability overlaps the historical adapter allowlist",
    )
    _validate_tts_voice_inventory(data["capabilities"])
    # Schema is intentionally versioned alongside the data.  Refusing a missing
    # schema catches an image that accidentally copied only one half of the
    # source artifact even though this stdlib reader enforces the same shape.
    schema = path.with_name("schema.json")
    try:
        schema_data = json.loads(schema.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceCapabilityError("voice capability schema cannot be loaded") from exc
    _require(isinstance(schema_data, dict), "voice capability schema must be an object")
    _require(schema_data.get("$schema") == "https://json-schema.org/draft/2020-12/schema", "voice capability schema draft is unsupported")
    schema_properties = schema_data.get("properties")
    _require(isinstance(schema_properties, dict), "voice capability schema properties are invalid")
    schema_version_rule = schema_properties.get("schema_version")
    _require(isinstance(schema_version_rule, dict), "voice capability schema version contract is invalid")
    schema_version_const = schema_version_rule.get("const")
    _require(
        isinstance(schema_version_const, int)
        and not isinstance(schema_version_const, bool)
        and schema_version_const == 1,
        "voice capability schema version contract is invalid",
    )
    _require(
        schema_version_rule.get("type") == "integer",
        "voice capability schema version contract is invalid",
    )
    schema_required = schema_data.get("required")
    _require(isinstance(schema_required, list) and set(schema_required) == expected_top_level, "voice capability schema top-level contract drifted")
    return VoiceCapabilityRegistry(data=_freeze(data), revision=hashlib.sha256(raw).hexdigest())


@lru_cache(maxsize=1)
def load_registry() -> VoiceCapabilityRegistry:
    """Load the one artifact once per process and return an immutable revision."""
    return _load(_root() / "capabilities.json")


def public_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy without exposing mutable registry state."""
    return _plain(value)


__all__ = ["VoiceCapabilityError", "VoiceCapabilityRegistry", "load_registry", "public_dict"]

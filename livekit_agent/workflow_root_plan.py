# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Metadata-only admission to the governed source root, before any media."""
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import math
from dataclasses import dataclass, field
from types import MappingProxyType

try:
    from .capabilities.workflow_listener import validate_frozen_voice_input, WorkflowListenerRefused
    from .capabilities.workflow_sources import (qualified_source_adapter, WorkflowSourceAdapterUnavailable,
                                                authored_constructor)
    from .capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS
    from .capabilities.opening import opening_policy
    from .conversation_engines_generated import ENGINE_DECLARATIONS, DEFAULT_CONVERSATION_ENGINE
    from .gpt_live_runtime import is_s2s_live_config, is_gemini_live_config, voice_frontend
    from .voice_configuration import ProductionConfigurationError
    from .runtime_language_switch import FollowCallerProfiles, LanguageSwitchError
except ImportError:
    from capabilities.workflow_listener import validate_frozen_voice_input, WorkflowListenerRefused
    from capabilities.workflow_sources import (qualified_source_adapter, WorkflowSourceAdapterUnavailable,
                                               authored_constructor)
    from capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS
    from capabilities.opening import opening_policy
    from conversation_engines_generated import ENGINE_DECLARATIONS, DEFAULT_CONVERSATION_ENGINE
    from gpt_live_runtime import is_s2s_live_config, is_gemini_live_config, voice_frontend
    from voice_configuration import ProductionConfigurationError
    from runtime_language_switch import FollowCallerProfiles, LanguageSwitchError


@dataclass(frozen=True, slots=True)
class WorkflowOutputProfile:
    adapter: object = field(repr=False)
    constructor: dict = field(repr=False)


@dataclass(frozen=True, slots=True)
class WorkflowRootPlan:
    language: str
    run_id: str
    source_limit: int
    listener: dict = field(repr=False)
    listener_adapter: object = field(repr=False)
    output_adapter: object = field(repr=False)
    output_constructor: dict = field(repr=False)
    opening: dict = field(repr=False)
    output_profiles: object = field(default_factory=dict, repr=False)
    follow_caller: object = field(default=None, repr=False)


def refuse_native_workflow(config):
    # Keep the native refusal ahead of all other metadata checks. Gemini Live
    # has no workflow source. GPT-Live is refused unless its engine row
    # declares the live role and the capability catalogue declares that live
    # source for the call language (any voice; its row declares every
    # language). The rest of the plan then checks the sources and price exactly.
    if not is_s2s_live_config(config):
        return
    if not is_gemini_live_config(config):
        voice, language = config.get("voice"), config.get("language")
        engine = ENGINE_DECLARATIONS.get(voice_frontend(config))
        if (engine is not None and engine.get("workflow_output_role") == "live"
                and isinstance(voice, dict) and type(language) is str and language):
            # The live role: declared for the call language, whatever the
            # voice. Undeclared, the refusal below is unchanged.
            try:
                qualified_source_adapter("live", voice.get("provider"), voice.get("model"), locale=language)
                return
            except WorkflowSourceAdapterUnavailable:
                pass
    engine = "Gemini Live" if is_gemini_live_config(config) else "GPT-Live"
    raise ProductionConfigurationError(f"{engine} is unavailable for workflow-bound calls")


def _qualified(role, provider, model, *, languages, voice_id=None):
    """One registered adapter whose catalogue row declares every language (the voice is not asked)."""
    adapters = [qualified_source_adapter(role, provider, model, voice_id=voice_id, locale=language)
                for language in languages]
    if not adapters or any(adapter != adapters[0] for adapter in adapters):
        raise WorkflowSourceAdapterUnavailable("workflow_source_adapter_unqualified")
    return adapters[0]


def _invalid():
    raise ProductionConfigurationError("governed workflow source metadata is invalid",
                                       reason="workflow_source_config_invalid")


def _output_constructor(role, voice, locale):
    """An authored source is built from exactly ``{provider, model, voice_id, locale}``.

    ``locale`` is the profile key that selected this voice: the call language
    for the active profile, each Follow-the-caller profile's own code. The
    live session takes its voice selection's provider, model and voice only.
    """
    if role == "live":
        # The live session's whole voice selection; the factory takes no more.
        return {key: voice.get(key) for key in ("provider", "model", "voice_id")}
    constructor = authored_constructor(voice, locale)
    if constructor is None:
        _invalid()
    return constructor


def _match_component(item, adapter, *, profile=False):
    keys = {"component", "provider", "model", "credential_source", "source_role", "unit",
            "max_native_units", "source_count_limit"}
    if profile:
        keys.add("voice_id")
    if (type(item) is not dict or set(item) != keys
            or any(item[key] != expected for key, expected in (
                ("component", adapter.component), ("provider", adapter.provider), ("model", adapter.model),
                ("source_role", adapter.role), ("unit", adapter.unit)))
            or type(item["source_count_limit"]) is not int
            or item["source_count_limit"] != adapter.source_count_limit
            or type(item["credential_source"]) is not str
            or item["credential_source"] not in {"platform_managed", "customer_byok"}):
        _invalid()
    fields = SOURCE_USAGE_FIELDS.get((adapter.component, adapter.unit))
    caps = item["max_native_units"]
    if (fields is None or type(caps) is not list or len(caps) != len(fields)
            or tuple(key for key, _ in adapter.max_native_units) != tuple(key for key, _ in fields)):
        _invalid()
    for pair, expected in zip(caps, adapter.max_native_units, strict=True):
        if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or pair[0] != expected[0]
                or type(pair[1]) not in (int, float) or not math.isfinite(pair[1]) or pair[1] <= 0
                or pair[1] != expected[1] or (adapter.unit != "audio_second" and type(pair[1]) is not int)):
            _invalid()


def _validate_navigation(config):
    navigation = config.get("navigation")
    if not isinstance(navigation, dict) or navigation.get("enabled") is not True:
        return
    try:
        engine = ENGINE_DECLARATIONS.get(voice_frontend(config) or DEFAULT_CONVERSATION_ENGINE)
        manifest = config.get("workflow_controller_manifest")
        fields = {"voice", "lines", "value_fields", "option_max_characters", "option_max_utf8", "max_value_digits", "digest"}
        if (engine is None or engine["workflow_output_role"] != "authored" or type(manifest) is not dict
                or set(manifest) != fields or type(manifest["digest"]) is not str
                or len(manifest["digest"]) != 64 or type(manifest["lines"]) is not dict
                or type(manifest["value_fields"]) is not list or len(manifest["value_fields"]) > 32
                or manifest["option_max_characters"] != 48 or manifest["option_max_utf8"] != 256
                or manifest["max_value_digits"] != 32):
            raise ValueError
        voice = config.get("voice")
        if not isinstance(voice, dict) or manifest["voice"] != {
                key: voice.get(key) for key in ("provider", "model", "voice_id")}:
            raise ValueError
        language = config.get("language")
        if not isinstance(language, str) or language.split("-", 1)[0] not in manifest["lines"]:
            raise ValueError
        document = {key: value for key, value in manifest.items() if key != "digest"}
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > 262144 or not secrets.compare_digest(manifest["digest"], hashlib.sha256(encoded).hexdigest()):
            raise ValueError
    except (TypeError, ValueError, KeyError, UnicodeError, RecursionError):
        raise ProductionConfigurationError("workflow navigation has no frozen admitted authority",
                                           reason="workflow_navigation_unqualified") from None


def create_workflow_root_plan(config: dict) -> WorkflowRootPlan:
    if type(config) is not dict:
        _invalid()
    refuse_native_workflow(config)
    _validate_navigation(config)
    language, run_id = config.get("language"), config.get("workflow_run_id")
    if (type(language) is not str or not language or language != language.strip().lower()
            or type(run_id) is not str or not run_id or len(run_id) > 128 or run_id != run_id.strip()):
        _invalid()
    frozen = config.get("frozen_language_profile")
    follow = frozen.get("follow_caller") if isinstance(frozen, dict) else config.get("follow_caller")
    profiled = isinstance(follow, dict) and follow.get("enabled") is True
    allowed = follow.get("allowed_profiles") if profiled else None
    if profiled and (type(allowed) is not dict or not 2 <= len(allowed) <= 32
            or any(type(code) is not str or code != code.strip().lower() for code in allowed)
            or follow.get("active_language") != language):
        _invalid()
    try:
        listener = validate_frozen_voice_input(config.get("workflow_voice_input"),
            required_languages=sorted(allowed) if profiled else [language])
    except WorkflowListenerRefused:
        raise ProductionConfigurationError("workflow has no valid frozen listener",
                                           reason="workflow_listener_snapshot_invalid") from None
    billing, voice = config.get("component_billing"), config.get("voice")
    if (type(billing) is not dict or billing.get("usage_required") is not True or billing.get("status") != "reserved"
            or type(billing.get("components")) is not list or len(billing["components"]) != 2
            or type(voice) is not dict):
        _invalid()
    engine = ENGINE_DECLARATIONS.get(voice_frontend(config) or DEFAULT_CONVERSATION_ENGINE)
    role = engine.get("workflow_output_role") if engine is not None else None
    if role not in {"authored", "live"}:
        _invalid()
    constructor = listener["constructor"]
    try:
        listener_languages = constructor.get("languages")
        if type(listener_languages) is not list or not listener_languages:
            raise WorkflowSourceAdapterUnavailable("workflow_source_adapter_unqualified")
        input_adapter = _qualified("listener", constructor["provider"], constructor["model"],
                                   languages=listener_languages)
        output_adapter = _qualified(role, voice.get("provider"), voice.get("model"),
                                    voice_id=None if role == "live" else voice.get("voice_id"), languages=[language])
    except WorkflowSourceAdapterUnavailable:
        raise ProductionConfigurationError("workflow sources have no deployed qualification",
                                           reason="workflow_source_adapter_unqualified") from None
    if input_adapter.source_count_limit != output_adapter.source_count_limit:
        _invalid()
    components = billing["components"]
    if (any(type(item) is not dict or type(item.get("component")) is not str for item in components)
            or {item["component"] for item in components} != {"stt", "tts"}):
        _invalid()
    source = next(item for item in components if item["component"] == "stt")
    output = next(item for item in components if item["component"] == "tts")
    _match_component(source, input_adapter)
    _match_component(output, output_adapter)
    if source["credential_source"] != listener["selection"]["listener"]["credential_source"]:
        _invalid()
    bindings = config.get("component_credential_bindings")
    if type(bindings) is not dict:
        _invalid()
    expected_credential = "customer_byok" if bindings.get("tts") else "platform_managed"
    if output["credential_source"] != expected_credential:
        _invalid()
    if type(voice.get("voice_id")) is not str or not voice["voice_id"]:
        _invalid()
    output_constructor = _output_constructor(role, voice, language)
    output_profiles, profile_voices, frozen_follow = {}, {}, None
    if profiled:
        entries = billing.get("workflow_output_profiles")
        if (type(entries) is not dict or set(entries) != set(allowed)
                or engine.get("has_language_axis") is not True):
            _invalid()
        for code, value in allowed.items():
            selected_config = value.get("config") if type(value) is dict else None
            selected_voice = selected_config.get("voice") if type(selected_config) is dict else None
            selected_engine = (ENGINE_DECLARATIONS.get(voice_frontend(selected_config) or DEFAULT_CONVERSATION_ENGINE)
                               if type(selected_config) is dict else None)
            if (type(selected_voice) is not dict or selected_engine is None
                    or selected_engine.get("workflow_output_role") != role
                    or selected_engine.get("has_language_axis") is not True):
                _invalid()
            try:
                selected_adapter = _qualified(role, selected_voice.get("provider"), selected_voice.get("model"),
                    voice_id=None if role == "live" else selected_voice.get("voice_id"), languages=[code])
            except WorkflowSourceAdapterUnavailable:
                raise ProductionConfigurationError("workflow sources have no deployed qualification",
                    reason="workflow_source_adapter_unqualified") from None
            _match_component(entries[code], selected_adapter, profile=True)
            if (selected_adapter.source_count_limit != input_adapter.source_count_limit
                    or entries[code]["credential_source"] != expected_credential
                    or entries[code]["voice_id"] != selected_voice.get("voice_id")
                    or type(selected_voice.get("voice_id")) is not str or not selected_voice["voice_id"]):
                _invalid()
            output_profiles[code] = WorkflowOutputProfile(selected_adapter,
                                                          _output_constructor(role, selected_voice, code))
            profile_voices[code] = selected_voice
        if (language not in output_profiles or voice != profile_voices[language]
                or output_constructor != output_profiles[language].constructor
                or output != {key: value for key, value in entries[language].items() if key != "voice_id"}):
            _invalid()
        navigation = config.get("navigation")
        if isinstance(navigation, dict) and navigation.get("enabled") is True:
            controller_voice = config["workflow_controller_manifest"]["voice"]
            matches = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in entries.values()
                       if controller_voice == {key: item[key] for key in ("provider", "model", "voice_id")}}
            if len(matches) != 1:
                _invalid()
        try:
            frozen_follow = FollowCallerProfiles.from_realtime_config({"follow_caller": follow},
                profile_from_config=lambda selected: None)
        except LanguageSwitchError:
            _invalid()
    return WorkflowRootPlan(language, run_id, input_adapter.source_count_limit, copy.deepcopy(listener),
        input_adapter, output_adapter, output_constructor, opening_policy(config.get("opening")),
        MappingProxyType(output_profiles), frozen_follow)

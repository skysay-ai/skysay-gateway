# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One customer selection and frozen listener contract for every voice engine.

Pure, secret-free and shared by the API wheel and separately built worker.
Only the capability registry resolves provider constructor options. Absence
means no spoken listener selected; it never selects a provider implicitly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

from .registry import load_registry, public_dict


SILERO_VAD_PINNED_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
STT_CONNECTION_TIMEOUT_SECONDS = 10
STT_FINALIZATION_TIMEOUT_SECONDS = 15
VAD_ACTIVATION_THRESHOLD = .5
MAX_CALLER_TURN_CHARACTERS = 16000
MAX_CALLER_SOURCE_SECONDS = 300
# The 32-language editable envelope with the shipped continuous-identification
# constructor serializes to 14824 ASCII characters. Keep one storage bound.
MAX_FROZEN_VOICE_INPUT_CHARACTERS = 16384


class WorkflowListenerRefused(ValueError):
    pass


def serialize_frozen_voice_input(value: dict | None) -> str | None:
    """Bound the exact persisted representation before any publication/run write.

    Executable validation belongs to ``validate_frozen_voice_input``. This
    storage seam neither repairs a snapshot nor resolves a provider.
    """
    if value is None:
        return None
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise WorkflowListenerRefused("workflow listener snapshot cannot be serialized") from None
    if not 2 <= len(encoded) <= MAX_FROZEN_VOICE_INPUT_CHARACTERS:
        raise WorkflowListenerRefused("workflow listener snapshot exceeds its storage limit")
    return encoded


def normalize_voice_input(value: object, *, max_languages: int = 32) -> dict | None:
    """Validate only the editable shape; model availability is publish-time."""
    if value is None:
        return None
    if type(value) is not dict or set(value) != {
        "listener", "silence_hold_ms", "interruption_min_duration_ms"
    }:
        raise WorkflowListenerRefused("voice_input requires a listener and its endpoint/interruption policy")
    listener = value["listener"]
    if type(listener) is not dict:
        raise WorkflowListenerRefused("listener requires an explicit credential source")
    source = listener.get("credential_source")
    keys = {"registry_name", "model", "languages", "credential_source"}
    if source == "customer_byok":
        keys.add("credential_id")
    if not isinstance(source, str) or source not in {"platform_managed", "customer_byok"} or set(listener) != keys:
        raise WorkflowListenerRefused("listener requires an explicit credential source and its exact fields")
    if source == "customer_byok" and (not isinstance(listener["credential_id"], str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", listener["credential_id"]) is None):
        raise WorkflowListenerRefused("listener BYOK requires a bounded opaque credential_id")
    name, model, languages = (listener[key] for key in ("registry_name", "model", "languages"))
    if (not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None
            or not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", model) is None):
        raise WorkflowListenerRefused("listener needs an exact catalog adapter and model")
    if (type(languages) is not list or not 1 <= len(languages) <= max_languages
            or any(not isinstance(item, str) or len(item) > 64
                   or re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{1,8})*", item) is None for item in languages)
            or len(set(languages)) != len(languages)):
        raise WorkflowListenerRefused("listener languages must be an explicit unique language set")
    for key, minimum, maximum in (("silence_hold_ms", 300, 3000), ("interruption_min_duration_ms", 0, 1000)):
        if type(value[key]) is not int or not minimum <= value[key] <= maximum:
            raise WorkflowListenerRefused(f"{key} must be an integer from {minimum} to {maximum}")
    credential = {"credential_source": source}
    if source == "customer_byok":
        credential["credential_id"] = listener["credential_id"]
    return {"listener": {"registry_name": name, "model": model, "languages": sorted(languages), **credential},
            "silence_hold_ms": value["silence_hold_ms"],
            "interruption_min_duration_ms": value["interruption_min_duration_ms"]}


def freeze_voice_input(value: object, *, required_languages: list[str], registry=None) -> dict:
    """Bind the selection, capability revision, timing and native adapter data."""
    selected = normalize_voice_input(value)
    if selected is None:
        raise WorkflowListenerRefused("select a workflow listener before starting a spoken workflow")
    registry = registry or load_registry()
    listener = selected["listener"]
    name, model, languages = (listener[key] for key in ("registry_name", "model", "languages"))
    if (type(required_languages) is not list or not required_languages
            or any(not isinstance(item, str) or item not in languages for item in required_languages)):
        raise WorkflowListenerRefused("listener must cover every workflow language")
    if not all(registry.declares_language("stt", name, model, language) for language in languages):
        raise WorkflowListenerRefused("selected listener/model does not declare every selected language")
    if len(languages) > 1:
        resolved = registry.simultaneous_stt_config(name, model, languages)
        if resolved is None:
            raise WorkflowListenerRefused("selected listener has no qualified simultaneous construction for these languages")
        constructor = public_dict(resolved)
    else:
        entry = next((row for row in (*registry.capabilities, *registry.declared_stack_languages)
                      if row["component"] == "stt" and row["registry_name"] == name and row["model"] == model), None)
        if entry is None:
            raise WorkflowListenerRefused("selected listener has no declared provider")
        locale = registry.stt_locale(name, model, languages[0])
        if registry.requires_stt_locale(name) and not locale:
            raise WorkflowListenerRefused("selected listener has no declared locale")
        constructor = {"provider": str(entry["provider"]), "registry_name": name, "model": model,
                       "languages": list(languages), "mode": "single", "locale": locale or languages[0]}
    frozen = {"selection": selected, "constructor": constructor,
              "required_languages": sorted(set(required_languages)),
              "capability_revision": registry.revision,
              "recognition": {"max_retry": 0, "connection_timeout_ms": STT_CONNECTION_TIMEOUT_SECONDS * 1000,
                              "finalization_timeout_ms": STT_FINALIZATION_TIMEOUT_SECONDS * 1000},
              "capture": {"sample_rate": 24000, "inference_sample_rate": 16000,
                          "queue_frames": 2, "frame_ms": 20,
                          "max_turn_ms": MAX_CALLER_SOURCE_SECONDS * 1000, "max_lag_ms": 2000,
                          "pcm_window_ms": 26000,
                          "recognition_concurrency": 2, "max_pending_turns": 1,
                          "max_text_characters": MAX_CALLER_TURN_CHARACTERS},
              "vad": {"checkpoint_sha256": SILERO_VAD_PINNED_SHA256, "activation_threshold": VAD_ACTIVATION_THRESHOLD,
                      "prefix_padding_ms": 500, "min_speech_ms": 64, "min_silence_ms": 550}}
    frozen["digest"] = _digest(frozen)
    return frozen


def validate_frozen_voice_input(value: object, *, required_languages: list[str], registry=None) -> dict:
    """Refuse mutated or stale executable snapshots before constructing I/O."""
    if type(value) is not dict:
        raise WorkflowListenerRefused("spoken workflow has no frozen listener")
    # Reconstruct the original coverage contract, not a later caller's or
    # mutable agent's language set. A consumer may require a subset of that
    # contract; it cannot extend an already pinned execution snapshot.
    current = freeze_voice_input(value.get("selection"),
                                 required_languages=value.get("required_languages"), registry=registry)
    # A catalogue addition does not invalidate a concrete frozen selection.
    # The revision is provenance, not a requirement to republish every agent.
    # Rebuild all executable fields against today's declaration, retaining only
    # the bounded historical provenance before comparing the full digest.
    revision = value.get("capability_revision")
    if type(revision) is not str or len(revision) != 64 or any(char not in "0123456789abcdef" for char in revision):
        raise WorkflowListenerRefused("workflow listener snapshot is invalid")
    current["capability_revision"] = revision
    # The provider's declared inventory is audit data, as for follow-the-caller (``frozen_simultaneous_stt_config``):
    # a declaration that grows (a provider adds languages to a multilingual mode) is a catalogue addition and does
    # not strand a frozen construction. One that drops a language the snapshot recorded is a change.
    stored, built = value.get("constructor"), current.get("constructor")
    if (
        type(stored) is dict and type(built) is dict and type(built.get("declared_languages")) is list
        and type(stored.get("declared_languages")) is list
        and all(type(item) is str for item in stored["declared_languages"])
        # (sorted and unique, as the registry writes it: a list that is not is not one this repository froze)
        and stored["declared_languages"] == sorted(set(stored["declared_languages"]))
        and set(stored["declared_languages"]) <= set(built["declared_languages"])
    ):
        built["declared_languages"] = list(stored["declared_languages"])
    elif (
        type(stored) is dict and type(built) is dict
        and ("language_scope" in stored or "language_scope" in built)
        and _same_construction(stored, built)
    ):
        # An open declaration (``language_scope: any``) on either side: the
        # socket is the same, and the provider set it was admitted against is
        # only inventory. So a declaration that drops its list for an open
        # scope keeps the publication's own copy instead of turning every
        # stored listener "stale; republish" (and a later list keeps an open one).
        current["constructor"] = copy.deepcopy(stored)
    current["digest"] = _digest({key: item for key, item in current.items() if key != "digest"})
    if not _same_snapshot(value, current):
        raise WorkflowListenerRefused("workflow listener snapshot is stale or changed; republish the workflow")
    if (type(required_languages) is not list or not required_languages
            or any(not isinstance(item, str) or item not in current["required_languages"]
                   for item in required_languages)):
        raise WorkflowListenerRefused("workflow language requirements exceed the pinned listener coverage")
    # Return an independent JSON value; no mutable publication object is lent.
    return current


#: What a multi-language constructor is built from. Everything else it may
#: carry is the declared inventory beside it, never sent to the provider.
_CONSTRUCTION_KEYS = frozenset({"provider", "registry_name", "model", "mode", "languages",
                                "adapter_options", "max_candidates"})
_INVENTORY_KEYS = (frozenset({"declared_languages"}), frozenset({"language_scope"}))
#: The registry's language tag grammar, which every declared inventory was written in.
_DECLARED_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Z]{2})?")


def _valid_inventory(constructor: dict) -> bool:
    """The inventory a constructor carries is one the registry could have emitted for its own languages.

    A declared list is unique registry tags containing every selected language;
    an open scope is ``any`` and only on the mode that names no language.
    """
    if "language_scope" in constructor:
        return constructor["language_scope"] == "any" and constructor.get("mode") == "language_multi"
    declared, selected = constructor.get("declared_languages"), constructor.get("languages")
    return (type(declared) is list and bool(declared)
            and all(type(tag) is str and _DECLARED_LANGUAGE.fullmatch(tag) for tag in declared)
            # (sorted and unique, as the registry writes it: a list that is not is not one this repository froze)
            and declared == sorted(set(declared))
            and type(selected) is list and set(selected) <= set(declared))


def _same_construction(stored: object, current: object) -> bool:
    """Whether two multi-language constructors build the same socket and differ at most in inventory.

    Both must be exactly one known shape -- the construction keys plus
    ``declared_languages`` or plus ``language_scope`` -- and every construction
    key must match with its JSON type. A single-language constructor, a
    malformed one, or one whose options, languages or ceiling moved is not the
    same construction, and the whole-snapshot comparison decides as before.
    """
    if type(stored) is not dict or type(current) is not dict or current.get("mode") == "single":
        return False
    for constructor in (stored, current):
        if set(constructor) - _CONSTRUCTION_KEYS not in _INVENTORY_KEYS or not _CONSTRUCTION_KEYS <= set(constructor):
            return False
    if not _valid_inventory(stored):
        return False
    return all(_same_snapshot(stored[key], current[key]) for key in _CONSTRUCTION_KEYS)


def _same_snapshot(value: object, expected: object) -> bool:
    # Compare against the bounded known schema, including JSON types. Python
    # equality alone accepts True as 1 in a supposedly frozen policy.
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_same_snapshot(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same_snapshot(item, other) for item, other in zip(value, expected))
    return value == expected


def _digest(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()

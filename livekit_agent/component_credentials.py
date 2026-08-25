# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict in-memory BYOK overlay for the supported managed worker stack."""
from __future__ import annotations

from typing import Any, Mapping


class ComponentCredentialContractError(RuntimeError):
    """A selected BYOK component cannot safely use platform credentials."""


_ENV_BY_COMPONENT = {
    ("stt", "deepgram"): ("DEEPGRAM_API_KEY", (), None),
    ("llm", "gemini"): ("GEMINI_API_KEY", ("GOOGLE_API_KEY",), None),
    ("tts", "elevenlabs"): ("ELEVENLABS_API_KEY", ("ELEVEN_API_KEY",), None),
    # Soniox provides both STT and TTS under one platform environment variable.
    # A call may, however, bind each component to a different customer key (or
    # bind only one while the other stays managed).  Keep the platform key
    # untouched and inject a component-scoped key plus an explicit marker that
    # the worker resolves without a generic-key fallback.
    ("stt", "soniox"): (
        "SONIOX_STT_API_KEY",
        (),
        "OPENPHONEX_BYOK_SONIOX_STT",
    ),
    ("tts", "soniox"): (
        "SONIOX_TTS_API_KEY",
        (),
        "OPENPHONEX_BYOK_SONIOX_TTS",
    ),
    # Fish has one TTS component, but still needs a distinct per-call lease
    # variable and marker. Reusing FISH_AUDIO_API_KEY would make a malformed
    # BYOK overlay indistinguishable from the ambient managed deployment key;
    # strip both ambient names because no other selected component uses them.
    ("tts", "fishaudio"): (
        "FISH_AUDIO_TTS_API_KEY",
        ("FISH_AUDIO_API_KEY", "FISH_API_KEY"),
        "OPENPHONEX_BYOK_FISHAUDIO_TTS",
    ),
}

# Names that exist only inside one call's overlay. They are also unset by the
# worker container entrypoint (agent/carrier-entrypoint.sh) so an operator who
# copies one into the deployment's environment file cannot make every call on
# that provider look like a corrupt lease. A test asserts the entrypoint's
# unset list equals this tuple, so adding a provider here means adding it
# there in the same change.
_SCOPED_COMPONENT_VARIABLES = (
    "SONIOX_STT_API_KEY",
    "SONIOX_TTS_API_KEY",
    "OPENPHONEX_BYOK_SONIOX_STT",
    "OPENPHONEX_BYOK_SONIOX_TTS",
    "FISH_AUDIO_TTS_API_KEY",
    "OPENPHONEX_BYOK_FISHAUDIO_TTS",
    "GEMINI_TTS_API_KEY",
    "OPENPHONEX_PLATFORM_GEMINI_TTS",
)


def has_byok_component(config: Mapping[str, Any]) -> bool:
    billing = config.get("component_billing")
    return isinstance(billing, dict) and any(
        isinstance(item, dict) and str(item.get("credential_source") or "") == "customer_byok"
        for item in billing.get("components") or []
    )


def runtime_environ_with_component_credentials(
    config: Mapping[str, Any],
    *,
    base_environ: Mapping[str, str],
    leased_components: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Return a local provider-environment overlay for one call.

    The input config carries only provider/model/source/opaque id. The lease
    carries plaintext only for explicitly BYOK-selected components. Every
    selected BYOK component must have an exact provider lease; aliases are
    removed so a missing customer key cannot silently resolve to a platform
    environment variable.
    """
    billing = config.get("component_billing")
    if not isinstance(billing, dict):
        return dict(base_environ)
    selections: dict[str, dict[str, Any]] = {}
    for item in billing.get("components") or []:
        if not isinstance(item, dict):
            continue
        component = str(item.get("component") or "")
        if component in selections:
            raise ComponentCredentialContractError("duplicate component credential selection")
        selections[component] = item
    selected_byok = {
        component: item
        for component, item in selections.items()
        if str(item.get("credential_source") or "") == "customer_byok"
    }
    if set(leased_components) != set(selected_byok):
        raise ComponentCredentialContractError("leased components do not match the selected BYOK components")
    overlay = dict(base_environ)
    # These names exist only for one call's in-memory overlay.  Ignoring any
    # inherited process value prevents an accidental deployment variable from
    # changing a managed component or pretending a BYOK selection was leased.
    for variable in _SCOPED_COMPONENT_VARIABLES:
        overlay.pop(variable, None)
    # Capture the platform Gemini TTS key before an independently selected
    # Gemini LLM BYOK lease replaces GEMINI_API_KEY. This keeps component
    # ownership exact in mixed calls without changing the platform's generic
    # key for the LLM or unrelated workers.
    selected_tts = selections.get("tts")
    if (
        isinstance(selected_tts, dict)
        and str(selected_tts.get("provider") or "").strip().lower() == "gemini"
    ):
        # Gemini TTS has no supported customer-BYOK adapter. A malformed
        # snapshot must fail before an independently leased Gemini LLM key
        # can become an accidental fallback for platform-managed synthesis.
        if selected_tts.get("credential_source") != "platform_managed":
            raise ComponentCredentialContractError(
                "Gemini TTS must use platform-managed credentials"
            )
        platform_gemini_tts_key = str(
            base_environ.get("GEMINI_API_KEY") or base_environ.get("GOOGLE_API_KEY") or ""
        )
        if not platform_gemini_tts_key:
            raise ComponentCredentialContractError("platform Gemini TTS key is unavailable")
        overlay["GEMINI_TTS_API_KEY"] = platform_gemini_tts_key
        overlay["OPENPHONEX_PLATFORM_GEMINI_TTS"] = "1"
    for component, item in selected_byok.items():
        provider = str(item.get("provider") or "").strip().lower()
        mapping = _ENV_BY_COMPONENT.get((component, provider))
        if mapping is None:
            raise ComponentCredentialContractError("selected BYOK provider has no worker credential adapter")
        lease = leased_components.get(component)
        if not isinstance(lease, Mapping) or str(lease.get("provider") or "").strip().lower() != provider:
            raise ComponentCredentialContractError("leased BYOK provider does not match the selected component")
        secret = lease.get("secret")
        if not isinstance(secret, str) or not secret:
            raise ComponentCredentialContractError("leased BYOK key is missing")
        variable, aliases, marker = mapping
        overlay[variable] = secret
        for alias in aliases:
            overlay.pop(alias, None)
        if marker is not None:
            overlay[marker] = "1"
    # The generic Soniox key remains necessary for a mixed call where one
    # Soniox component is still platform managed. If every selected Soniox
    # component is leased BYOK, remove it too: an SDK default lookup must not
    # be able to cross over to the platform account behind a scoped marker.
    selected_soniox_components = {
        component
        for component, item in selections.items()
        if str(item.get("provider") or "").strip().lower() == "soniox"
    }
    leased_soniox_components = {
        component
        for component, item in selected_byok.items()
        if str(item.get("provider") or "").strip().lower() == "soniox"
    }
    if selected_soniox_components and selected_soniox_components == leased_soniox_components:
        overlay.pop("SONIOX_API_KEY", None)
    return overlay

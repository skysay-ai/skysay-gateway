# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict in-memory BYOK overlay for the supported managed worker stack."""
from __future__ import annotations

from typing import Any, Mapping


class ComponentCredentialContractError(RuntimeError):
    """A selected BYOK component cannot safely use platform credentials."""


_ENV_BY_COMPONENT = {
    ("stt", "deepgram"): ("DEEPGRAM_API_KEY", ()),
    ("llm", "gemini"): ("GEMINI_API_KEY", ("GOOGLE_API_KEY",)),
    ("tts", "elevenlabs"): ("ELEVENLABS_API_KEY", ("ELEVEN_API_KEY",)),
}


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
        variable, aliases = mapping
        overlay[variable] = secret
        for alias in aliases:
            overlay.pop(alias, None)
    return overlay

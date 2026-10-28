# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Strict in-memory BYOK overlay for the supported managed worker stack."""
from __future__ import annotations

from typing import Any, Mapping
from types import MappingProxyType


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
        "SKYSAY_BYOK_SONIOX_STT",
    ),
    ("tts", "soniox"): (
        "SONIOX_TTS_API_KEY",
        (),
        "SKYSAY_BYOK_SONIOX_TTS",
    ),
    # Fish has one TTS component, but still needs a distinct per-call lease
    # variable and marker. Reusing FISH_AUDIO_API_KEY would make a malformed
    # BYOK overlay indistinguishable from the ambient managed deployment key;
    # strip both ambient names because no other selected component uses them.
    ("tts", "fishaudio"): (
        "FISH_AUDIO_TTS_API_KEY",
        ("FISH_AUDIO_API_KEY", "FISH_API_KEY"),
        "SKYSAY_BYOK_FISHAUDIO_TTS",
    ),
    # Cartesia's SDK also reads CARTESIA_API_KEY from the ambient process.
    # Keep a customer's call-scoped lease on a distinct name and remove that
    # ambient fallback, exactly as for Fish: an incomplete lease must stop the
    # call before media rather than charge the managed Cartesia account.
    ("tts", "cartesia"): (
        "CARTESIA_TTS_API_KEY",
        ("CARTESIA_API_KEY",),
        "SKYSAY_BYOK_CARTESIA_TTS",
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
    "SKYSAY_BYOK_SONIOX_STT",
    "SKYSAY_BYOK_SONIOX_TTS",
    "FISH_AUDIO_TTS_API_KEY",
    "SKYSAY_BYOK_FISHAUDIO_TTS",
    "CARTESIA_TTS_API_KEY",
    "SKYSAY_BYOK_CARTESIA_TTS",
    "GEMINI_TTS_API_KEY",
    "SKYSAY_PLATFORM_GEMINI_TTS",
)


# Source roles use explicit platform names only where they are disjoint from
# the call-scoped BYOK overlay grammar. This declaration grants no source.
_MANAGED_SOURCE_ENV_BY_ROLE = MappingProxyType({
    # The live role's call-lifetime GPT-Live session: the account the
    # freeform GPT-Live path reads, copied once at the workflow bootstrap.
    ("live", "openai"): "OPENAI_API_KEY",
    # The same managed OpenAI account, for the finite transcription listener;
    # its declaration is platform-managed only and refuses a customer key.
    ("listener", "openai"): "OPENAI_API_KEY",
    ("listener", "azure"): "AZURE_SPEECH_KEY",
    ("listener", "soniox"): "SONIOX_API_KEY",
    # Authored sources whose component overlay name is absent (Azure TTS) or
    # call-scoped only (Soniox TTS). The others use ``_ENV_BY_COMPONENT``.
    ("authored", "azure"): "AZURE_SPEECH_KEY",
    ("authored", "soniox"): "SONIOX_API_KEY",
})

# Platform configuration that a regional managed source needs beside its key,
# per ``(component, provider)``. Not a secret; never part of a constructor.
# The Azure listener and the Azure authored source share one platform Speech
# resource, so both read the region the released Cascade path reads.
_MANAGED_SOURCE_REGION_ENV = MappingProxyType({
    ("stt", "azure"): "AZURE_SPEECH_REGION",
    ("tts", "azure"): "AZURE_SPEECH_REGION",
})


def managed_source_environment_name(role: str, provider: str, *, leased_components: Mapping) -> str | None:
    """Resolve inert metadata; a leased component never consults a platform key."""
    if not isinstance(leased_components, Mapping):
        raise ComponentCredentialContractError("source credentials require explicit component lease state")
    component = {"listener": "stt", "authored": "tts", "live": "tts"}.get(role)
    if component is None or component in leased_components:
        return None
    return _MANAGED_SOURCE_ENV_BY_ROLE.get((role, provider))


def has_byok_component(config: Mapping[str, Any]) -> bool:
    billing = config.get("component_billing")
    return isinstance(billing, dict) and any(
        isinstance(item, dict) and str(item.get("credential_source") or "") == "customer_byok"
        for item in billing.get("components") or []
    )


def should_lease_components(config: Mapping[str, Any], *, platform_media_worker: bool) -> bool:
    """Whether this worker needs to ask the control plane for keys at all.

    A managed cloud worker asks only for BYOK components; it holds the platform
    keys for everything else. A self-hosted worker holds none of them, so any
    metered component at all is worth asking about -- what comes back is what
    Skysay issued for this customer, and a component it cannot issue is
    simply absent.
    """
    if has_byok_component(config):
        return True
    if platform_media_worker:
        return False
    billing = config.get("component_billing")
    return isinstance(billing, dict) and any(
        isinstance(item, dict) and str(item.get("credential_source") or "") == "platform_managed"
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
    # A BYOK component MUST be leased -- its absence is never a fallback. A
    # platform-managed component MAY be leased: on a self-hosted deployment the
    # control plane hands over a key it issued for this customer, and a
    # provider it cannot issue for stays on the deployment's own key. So the
    # rule is containment, not equality, and it is checked in both directions:
    # every BYOK selection is leased, and nothing unselected is.
    if not set(selected_byok) <= set(leased_components):
        raise ComponentCredentialContractError("leased components do not match the selected BYOK components")
    if not set(leased_components) <= set(selections):
        raise ComponentCredentialContractError("leased components include an unselected component")
    injected = {
        component: item
        for component, item in selections.items()
        if component in leased_components
    }
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
        # Google has no per-customer credential issuer yet -- real isolation
        # there is a GCP project per customer -- so Gemini TTS is never leased
        # and still reads the deployment's own key, exactly as before.
        platform_gemini_tts_key = str(
            base_environ.get("GEMINI_API_KEY") or base_environ.get("GOOGLE_API_KEY") or ""
        )
        if not platform_gemini_tts_key:
            raise ComponentCredentialContractError("platform Gemini TTS key is unavailable")
        overlay["GEMINI_TTS_API_KEY"] = platform_gemini_tts_key
        overlay["SKYSAY_PLATFORM_GEMINI_TTS"] = "1"
    for component, item in injected.items():
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
    # Soniox component still resolves from the deployment's own key. If every
    # selected Soniox component was leased -- whether the customer's own key or
    # one Skysay issued -- remove it too: an SDK default lookup must not be
    # able to cross over to the account key behind a scoped marker.
    selected_soniox_components = {
        component
        for component, item in selections.items()
        if str(item.get("provider") or "").strip().lower() == "soniox"
    }
    leased_soniox_components = {
        component
        for component, item in injected.items()
        if str(item.get("provider") or "").strip().lower() == "soniox"
    }
    if selected_soniox_components and selected_soniox_components == leased_soniox_components:
        overlay.pop("SONIOX_API_KEY", None)
    return overlay

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Production-facing LiveKit semantic bindings.

This module is intentionally SDK-light.  The control plane owns the resolved
semantic configuration and MCP allowlist; the LiveKit adapter only validates
and translates those values into provider/plugin and model-tool objects.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any, Callable, Protocol

try:  # package import in tests; flat /app import in the pinned worker image
    from .demo import DemoProfile
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from demo import DemoProfile


class ProductionConfigurationError(RuntimeError):
    """A resolved control-plane configuration cannot build a LiveKit agent."""


def inbound_sip_identity(attributes: dict[str, str], participant_identity: str) -> tuple[str, str, str]:
    """Return caller, called DID, and carrier call id from LiveKit SIP metadata."""

    caller = str(attributes.get("sip.phoneNumber") or "").strip()
    called = str(attributes.get("sip.trunkPhoneNumber") or "").strip()
    if caller.isdigit():
        caller = "+" + caller
    if called.isdigit():
        called = "+" + called
    if not called.startswith("+") or not called[1:].isdigit():
        raise ValueError("inbound SIP participant omitted a valid sip.trunkPhoneNumber")
    provider_call_id = str(
        attributes.get("sip.callIDFull")
        or attributes.get("sip.callID")
        or participant_identity
    ).strip()
    if not provider_call_id:
        raise ValueError("inbound SIP participant omitted its call identity")
    return caller or "unknown", called, provider_call_id


class ToolSession(Protocol):
    config: dict[str, Any]

    def tool_definitions(self) -> list[dict[str, Any]]: ...

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def livekit_worker_heartbeat_payload(
    environ: dict[str, str] | os._Environ[str],
    *,
    process_id: int,
) -> dict[str, Any]:
    worker_base = str(environ.get("OPENPHONEX_WORKER_ID") or "livekit-agent").strip()
    return {
        "worker_id": f"{worker_base}-{process_id}",
        "worker_type": "livekit_agent",
        "status": "ready",
        "metadata": {
            "media_stack": "livekit",
            "role": "semantic_agent_process",
        },
    }


def post_livekit_worker_heartbeat(
    transport: Any,
    environ: dict[str, str] | os._Environ[str],
    *,
    process_id: int,
) -> dict[str, Any]:
    return transport.request(
        "POST",
        "/v1/worker-heartbeats",
        livekit_worker_heartbeat_payload(environ, process_id=process_id),
    )


# STT registry_name -> the one model each is pinned to. Deepgram (nova-3) is
# the default English/Estonian path; Google (chirp_2) is the si-LK (Sinhala)
# path -- Deepgram does not support Sinhala at all. Chirp 2 in the
# asia-southeast1 recognizer location is the empirically verified si-LK
# path (Chirp 3 rejects si-LK with an HTTP 400 in both eu and us).
_STT_REGISTRY_MODELS: dict[str, str] = {
    "deepgram": "nova-3",
    "google": "chirp_2",
}
# TTS registry_names this worker knows how to build. ElevenLabs is the default
# English/Estonian path; ``gemini_tts`` (Gemini-TTS) is the si-LK (Sinhala)
# path -- ElevenLabs does not support Sinhala at all.
_TTS_REGISTRIES = frozenset({"elevenlabs", "gemini_tts"})


def profile_from_realtime_config(config: dict[str, Any]) -> DemoProfile:
    language = str(config.get("language") or "").strip().lower()
    if language not in {"en", "et", "si"}:
        raise ProductionConfigurationError("realtime-config language must be en, et, or si")
    transcriber = config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}

    stt_registry = str(transcriber.get("registry_name") or "").lower()
    expected_stt_model = _STT_REGISTRY_MODELS.get(stt_registry)
    if expected_stt_model is None:
        raise ProductionConfigurationError(
            "realtime-config must resolve a supported STT provider (Deepgram or Google)"
        )
    if str(transcriber.get("model") or "") != expected_stt_model:
        raise ProductionConfigurationError(
            f"realtime-config must resolve {stt_registry} {expected_stt_model}"
        )
    if str(model.get("registry_name") or "").lower() not in {"gemini", "google"}:
        raise ProductionConfigurationError("realtime-config must resolve Gemini LLM")
    tts_registry = str(voice.get("registry_name") or "").lower()
    if tts_registry not in _TTS_REGISTRIES:
        raise ProductionConfigurationError(
            "realtime-config must resolve a supported TTS provider (ElevenLabs or Gemini TTS)"
        )

    voice_id = str(voice.get("voice_id") or "").strip()
    tts_model = str(voice.get("model") or "").strip()
    llm_model = str(model.get("model") or "").strip()
    instructions = str(config.get("system_prompt") or "").strip()
    first_message = str(config.get("first_message") or "").strip()
    if not all((voice_id, tts_model, llm_model, instructions, first_message)):
        raise ProductionConfigurationError("realtime-config omitted runnable prompt/provider values")
    if language == "et" and tts_registry != "elevenlabs":
        raise ProductionConfigurationError("Estonian requires ElevenLabs TTS")
    if language == "et" and "flash" in tts_model.lower():
        raise ProductionConfigurationError("Estonian requires an ElevenLabs multilingual model")
    if language == "si" and (stt_registry != "google" or tts_registry != "gemini_tts"):
        raise ProductionConfigurationError(
            "Sinhala requires Google Chirp STT and Gemini TTS (Deepgram and ElevenLabs do not "
            "support si-LK)"
        )

    stt_language = str(transcriber.get("language") or language).strip()
    if language == "en" and stt_language == "en":
        stt_language = "en-US"
    if language == "si" and stt_language == "si":
        stt_language = "si-LK"
    return DemoProfile(
        language=language,
        stt_language=stt_language,
        tts_model=tts_model,
        voice_id=voice_id,
        llm_model=llm_model,
        instructions=instructions,
        first_message=first_message,
        stt_registry=stt_registry,
        tts_registry=tts_registry,
    )


def _default_tool_builder(function: Callable[..., Any], raw_schema: dict[str, Any]) -> Any:
    from livekit.agents import function_tool

    return function_tool(raw_schema=raw_schema)(function)


def executed_tool_names(event: Any) -> list[str]:
    """Read the pinned LiveKit ``FunctionToolsExecutedEvent`` shape safely."""

    return [
        name
        for call in getattr(event, "function_calls", [])
        if (name := str(getattr(call, "name", "")).strip())
    ]


def is_greeting_fragment(text: str, first_message: str) -> bool:
    """Exclude a barge-in-truncated first message from semantic turn pairing."""

    candidate = " ".join(text.casefold().split())
    greeting = " ".join(first_message.casefold().split())
    return bool(candidate and greeting) and (
        greeting.startswith(candidate) or candidate.startswith(greeting)
    )


def build_livekit_model_tools(
    session: ToolSession,
    *,
    tool_builder: Callable[[Callable[..., Any], dict[str, Any]], Any] = _default_tool_builder,
) -> list[Any]:
    """Translate the filtered MCP allowlist into LiveKit model-callable tools.

    The model receives only schemas already filtered by
    :meth:`Gate2CallSession.tool_definitions`.  Each invocation crosses the
    same session interface, which rechecks the allowlist and creates the MCP
    audit event before returning output to the model.
    """

    tools: list[Any] = []
    for definition in session.tool_definitions():
        name = str(definition.get("name") or "").strip()
        parameters = definition.get("inputSchema")
        if not name or not isinstance(parameters, dict):
            raise ProductionConfigurationError("MCP tool definition is missing name/inputSchema")
        model_parameters = copy.deepcopy(parameters)
        properties = model_parameters.get("properties")
        required = model_parameters.get("required")
        tenant_arguments = {
            name: str(session.config.get(name) or "").strip()
            for name in ("organization_id", "project_id")
        }
        if not all(tenant_arguments.values()):
            raise ProductionConfigurationError(
                "realtime-config omitted the authenticated organization/project binding"
            )
        for tenant_field in tenant_arguments:
            if isinstance(properties, dict):
                properties.pop(tenant_field, None)
            if isinstance(required, list):
                model_parameters["required"] = [
                    field
                    for field in model_parameters.get("required", [])
                    if field != tenant_field
                ]
        raw_schema = {
            "name": name,
            "description": str(definition.get("description") or f"OpenPhonex tool {name}"),
            "parameters": model_parameters,
        }

        async def invoke(
            raw_arguments: dict[str, object],
            *,
            _name: str = name,
            _tenant_arguments: dict[str, str] = tenant_arguments,
        ) -> str:
            arguments = dict(raw_arguments)
            arguments.update(_tenant_arguments)
            result = await asyncio.to_thread(
                session.invoke_tool, _name, arguments
            )
            return json.dumps(result, sort_keys=True, default=str)

        tools.append(tool_builder(invoke, raw_schema))
    return tools

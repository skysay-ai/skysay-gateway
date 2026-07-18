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
from typing import Any, Callable, Protocol

try:  # package import in tests; flat /app import in the pinned worker image
    from .demo import DemoProfile
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from demo import DemoProfile


class ProductionConfigurationError(RuntimeError):
    """A resolved control-plane configuration cannot build a LiveKit agent."""


class ToolSession(Protocol):
    config: dict[str, Any]

    def tool_definitions(self) -> list[dict[str, Any]]: ...

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def profile_from_realtime_config(config: dict[str, Any]) -> DemoProfile:
    language = str(config.get("language") or "").strip().lower()
    if language not in {"en", "et"}:
        raise ProductionConfigurationError("realtime-config language must be en or et")
    transcriber = config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}
    if str(transcriber.get("registry_name") or "").lower() != "deepgram":
        raise ProductionConfigurationError("realtime-config must resolve Deepgram STT")
    if str(transcriber.get("model") or "") != "nova-3":
        raise ProductionConfigurationError("realtime-config must resolve Deepgram nova-3")
    if str(model.get("registry_name") or "").lower() not in {"gemini", "google"}:
        raise ProductionConfigurationError("realtime-config must resolve Gemini LLM")
    if str(voice.get("registry_name") or "").lower() != "elevenlabs":
        raise ProductionConfigurationError("realtime-config must resolve ElevenLabs TTS")

    voice_id = str(voice.get("voice_id") or "").strip()
    tts_model = str(voice.get("model") or "").strip()
    llm_model = str(model.get("model") or "").strip()
    instructions = str(config.get("system_prompt") or "").strip()
    first_message = str(config.get("first_message") or "").strip()
    if not all((voice_id, tts_model, llm_model, instructions, first_message)):
        raise ProductionConfigurationError("realtime-config omitted runnable prompt/provider values")
    if language == "et" and "flash" in tts_model.lower():
        raise ProductionConfigurationError("Estonian requires an ElevenLabs multilingual model")
    stt_language = str(transcriber.get("language") or language).strip()
    if language == "en" and stt_language == "en":
        stt_language = "en-US"
    return DemoProfile(
        language=language,
        stt_language=stt_language,
        tts_model=tts_model,
        voice_id=voice_id,
        llm_model=llm_model,
        instructions=instructions,
        first_message=first_message,
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

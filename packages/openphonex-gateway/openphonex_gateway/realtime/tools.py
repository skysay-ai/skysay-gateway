# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Voice-agent tool calling for the realtime pipeline.

Worker-side half of the tool loop. The control plane resolves the agent's
``tools`` allowlist into the per-call config; this module turns those specs
into LLM function declarations, and executes model-emitted tool calls:

* ``end_call`` runs INSIDE the pipeline (speak the goodbye + request hangup) —
  it never leaves the worker.
* Everything else (``send_sms``, ``transfer_to_human``, custom webhook tools)
  is POSTed to the control plane's
  ``POST /v1/calls/{call_id}/agent-tool-calls`` where tenant context is
  injected server-side from the call row and the ``authorize_outbound``
  gate runs. The worker NEVER sends org/project/from-number values — the
  control plane would not trust them anyway.

The JSON result (``{"ok": ..., "result"|"error": ...}``) is appended to the
conversation as a ``{"role": "tool"}`` message and the LLM is re-invoked, up
to ``DEFAULT_MAX_TOOL_ITERATIONS`` per caller turn.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# Mirrors telephony_ai.services.agent_tools.BUILTIN_TOOL_DECLARATIONS (the
# gateway package must stay importable standalone, so the declarations are
# duplicated here rather than imported across the OSS boundary).
BUILTIN_TOOL_DECLARATIONS: dict[str, dict[str, Any]] = {
    "end_call": {
        "name": "end_call",
        "description": "End the phone call politely once the caller's needs are met.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final goodbye sentence to speak before hanging up.",
                }
            },
        },
    },
    "send_sms": {
        "name": "send_sms",
        "description": "Send an SMS text message to the caller's phone number.",
        "parameters": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "The SMS message text."}
            },
            "required": ["body"],
        },
    },
    "transfer_to_human": {
        "name": "transfer_to_human",
        "description": "Transfer this call to a human colleague at the configured destination.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short reason for the transfer.",
                }
            },
        },
    },
}

# Iteration cap for the per-turn tool loop: prevents a model that keeps
# emitting tool calls from looping forever mid-call. On cap the pipeline
# reports the cap to the model and forces a plain (tool-free) reply.
DEFAULT_MAX_TOOL_ITERATIONS = 5


@dataclass(frozen=True)
class ToolCall:
    """One function call emitted by the LLM.

    ``id`` and ``thought_signature`` are provider continuity tokens (Gemini's
    ``functionCall.id`` + part-level ``thoughtSignature``) that MUST be echoed
    back on the follow-up turn; they are inert for providers that don't emit
    them.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    thought_signature: str = ""


def tool_declarations(tool_specs: Any) -> list[dict[str, Any]]:
    """Resolve an agent's ``tools`` config into LLM function declarations.

    Dict specs become declarations: built-in names use the canonical
    declaration above (the model only ever sees the narrow schema — never the
    owner-configured transfer destination), webhook specs use their declared
    JSON-schema ``parameters``. Plain strings are legacy placeholder
    rows and stay inert — they produce no declaration and no callable tool.
    """

    declarations: list[dict[str, Any]] = []
    if not isinstance(tool_specs, list):
        return declarations
    for spec in tool_specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        builtin = BUILTIN_TOOL_DECLARATIONS.get(name)
        if spec.get("type", "builtin" if builtin else "webhook") == "builtin":
            if builtin:
                declarations.append(dict(builtin))
            continue
        parameters = spec.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        declarations.append(
            {
                "name": name,
                "description": str(spec.get("description") or ""),
                "parameters": parameters,
            }
        )
    return declarations


class ControlPlaneToolExecutor:
    """Executes non-local tools via the control plane's tool-call endpoint.

    Never raises into the pipeline: every failure becomes an
    ``{"ok": False, "error": ...}`` result the model can react to, so a flaky
    tool cannot abort a live call.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_token: str,
        call_id: str,
        agent_id: str = "",
        timeout_seconds: float = 15.0,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token
        self.call_id = call_id
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen

    def execute(self, call: ToolCall) -> dict[str, Any]:
        if not self.api_url or not self.api_token or not self.call_id:
            return {"ok": False, "tool": call.name, "error": "control plane tool endpoint is not configured"}
        body = {
            "tool": call.name,
            "arguments": call.arguments,
            "tool_call_id": call.id,
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        request = urllib.request.Request(
            f"{self.api_url}/v1/calls/{urllib.parse.quote(self.call_id, safe='')}/agent-tool-calls",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = self._urlopen or urllib.request.urlopen
        try:
            with opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="replace")[-200:]
            except Exception:  # noqa: BLE001 - error-body read best-effort
                pass
            return {"ok": False, "tool": call.name, "error": f"tool endpoint HTTP {exc.code}: {detail}"}
        except Exception as exc:  # noqa: BLE001 - a tool failure must not abort the call
            return {"ok": False, "tool": call.name, "error": f"tool endpoint unreachable: {exc.__class__.__name__}"}
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "tool": call.name, "error": "tool endpoint returned invalid JSON"}
        if not isinstance(parsed, dict):
            return {"ok": False, "tool": call.name, "error": "tool endpoint returned a non-object result"}
        return parsed

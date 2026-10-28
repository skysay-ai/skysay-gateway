# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Trusted workflow routing and exact locale-bound published speech text.

The universal coordinator owns turns and the backend owns business decisions.
No worker-authored farewell or runtime language switch belongs to this path.
"""
from __future__ import annotations

from typing import Any


def pinned_workflow_run_id(config: dict[str, Any]) -> str:
    """The workflow run id pinned to this call's trusted config, or "".

    Presence alone is what flips a Talk session from the free-form
    AgentSession path to the workflow-driven one. ``config`` (the call's
    own resolve_call_realtime_config response) is the ONLY source ever
    consulted -- job/room dispatch metadata is never trusted for this.
    """
    return str((config or {}).get("workflow_run_id") or "").strip()


def authorized_spoken_text(contract: dict[str, Any], step_id: str, language: str) -> str | None:
    """Exact source-language text or an explicit variant; no display fallback."""
    languages = contract.get("languages")
    if not isinstance(languages, list) or not languages or not isinstance(languages[0], str):
        return None
    steps = [step for step in contract.get("steps", ()) if isinstance(step, dict) and step.get("id") == step_id]
    if len(steps) != 1 or not isinstance(language, str) or not language.strip():
        return None
    step = steps[0]
    field = {"action": "prompt", "ask": "prompt", "say": "prompt", "end": "farewell"}.get(step.get("type"))
    if field is None:
        return None
    source = step.get("language") or languages[0]
    if not isinstance(source, str):
        return None
    language = language.strip().lower()
    if source.strip().lower() == language:
        text = step.get(field)
    else:
        renderings = contract.get("renderings")
        block = renderings.get(language) if isinstance(renderings, dict) else None
        variants = block.get("steps") if isinstance(block, dict) else None
        variant = variants.get(step_id) if isinstance(variants, dict) else None
        text = variant.get("text") if isinstance(variant, dict) else None
    return text if isinstance(text, str) and text.strip() else None

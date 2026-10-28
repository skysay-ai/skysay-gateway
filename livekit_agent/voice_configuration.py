# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Shared typed configuration refusal before a media constructor is entered."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ProductionConfigurationError(RuntimeError):
    """A resolved control-plane configuration cannot build a LiveKit agent.

    ``reason`` is an OPTIONAL closed label naming which rule refused, published
    to the customer on ``call.failed`` through the pre-media termination. The
    message is for the worker log and stays there; the label is the only part
    that leaves this process, which is why it is a separate field rather than
    something derived from the text. It never carries a value from the config
    -- naming the missing field is the whole point, and naming its content
    would put a system prompt on a dashboard.

    Left empty by every raise that has no label of its own; the handler then
    falls back to the stage's own reason exactly as before.
    """

    def __init__(self, *args: Any, reason: str = "") -> None:
        super().__init__(*args)
        self.reason = reason



def refuse_freeform_workflow(config: Mapping[str, Any]) -> None:
    billing = config.get("component_billing")
    components = billing.get("components") if isinstance(billing, Mapping) else None
    if (str(config.get("workflow_run_id") or "").strip()
            or isinstance(components, (list, tuple)) and any(
                isinstance(item, Mapping) and "source_role" in item for item in components)):
        raise ProductionConfigurationError("workflow media requires its governed source root",
                                           reason="workflow_source_root_required")

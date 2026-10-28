# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One outbound opening contract for the control plane and every voice engine."""
from __future__ import annotations

import math
from typing import Any

DEFAULT_FIRST_SPEAKER = "callee"
DEFAULT_SILENCE_WAIT_SECONDS = 2.0


def opening_policy(value: Any = None) -> dict[str, Any]:
    """Resolve defaults; refuse malformed settings instead of guessing timing.

    An empty object resets both settings. The wait measures observed silence
    after answer, never the maximum time allowed for the person's greeting.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"first_speaker", "silence_wait_seconds"}:
        raise ValueError("opening must contain only first_speaker and silence_wait_seconds")
    speaker = value.get("first_speaker", DEFAULT_FIRST_SPEAKER)
    if speaker not in ("callee", "agent"):
        raise ValueError("opening.first_speaker must be callee or agent")
    wait = value.get("silence_wait_seconds", DEFAULT_SILENCE_WAIT_SECONDS)
    if type(wait) not in (int, float) or not 1 <= wait <= 5 or not math.isfinite(wait):
        raise ValueError("opening.silence_wait_seconds must be a number from 1 to 5")
    return {"first_speaker": speaker, "silence_wait_seconds": float(wait)}

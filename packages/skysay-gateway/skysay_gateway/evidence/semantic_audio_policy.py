# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

from typing import Any


SEMANTIC_RESPONSE_AUDIO_POLICY_SCHEMA = "skysay.semantic_response_audio_policy.v1"

_SEMANTIC_RESPONSE_AUDIO_POLICY: dict[str, object] = {
    "schema": SEMANTIC_RESPONSE_AUDIO_POLICY_SCHEMA,
    "filler_acknowledgements_enabled": False,
    "first_post_caller_agent_audio": "assistant_spoken_semantic_response",
    "semantic_start_alignment": "caller_side_vad_worker_turn_clip_correlation",
}


def semantic_response_audio_policy(
    *, filler_acknowledgements_enabled: bool = False
) -> dict[str, object]:
    """Return the fingerprinted invariant used by caller-side latency evidence.

    Acknowledgements ("One moment") play on a separately published background
    track that the evidence sidecar never records, so the recorded agent track
    still begins with the assistant's spoken semantic response either way. The
    flag records that the caller may have heard a filler the transcript does
    not show; it does not change what the acoustic measurement sees.
    """

    policy = dict(_SEMANTIC_RESPONSE_AUDIO_POLICY)
    policy["filler_acknowledgements_enabled"] = bool(filler_acknowledgements_enabled)
    return policy


def valid_semantic_response_audio_policy(value: Any) -> bool:
    """Accept the exact contract understood by the analyzer, either flag value."""

    if not isinstance(value, dict):
        return False
    flag = value.get("filler_acknowledgements_enabled")
    if not isinstance(flag, bool):
        return False
    return value == semantic_response_audio_policy(filler_acknowledgements_enabled=flag)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

from typing import Any


SEMANTIC_RESPONSE_AUDIO_POLICY_SCHEMA = "openphonex.semantic_response_audio_policy.v1"

_SEMANTIC_RESPONSE_AUDIO_POLICY: dict[str, object] = {
    "schema": SEMANTIC_RESPONSE_AUDIO_POLICY_SCHEMA,
    "filler_acknowledgements_enabled": False,
    "first_post_caller_agent_audio": "assistant_spoken_semantic_response",
    "semantic_start_alignment": "caller_side_vad_worker_turn_clip_correlation",
}


def semantic_response_audio_policy() -> dict[str, object]:
    """Return the fingerprinted invariant used by caller-side latency evidence.

    The current pipeline sends no acknowledgement or filler audio before the
    assistant's spoken semantic response. If that behavior changes, this
    contract must change too so existing acoustic cohorts cannot be reused.
    """

    return dict(_SEMANTIC_RESPONSE_AUDIO_POLICY)


def valid_semantic_response_audio_policy(value: Any) -> bool:
    """Accept only the exact no-filler contract understood by the analyzer."""

    return isinstance(value, dict) and value == _SEMANTIC_RESPONSE_AUDIO_POLICY

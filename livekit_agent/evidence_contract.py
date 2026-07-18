# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""PR #79 evidence invariants shared by the isolated LiveKit services."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


_SEMANTIC_RESPONSE_AUDIO_POLICY: dict[str, object] = {
    "schema": "openphonex.semantic_response_audio_policy.v1",
    "filler_acknowledgements_enabled": False,
    "first_post_caller_agent_audio": "assistant_spoken_semantic_response",
    "semantic_start_alignment": "caller_side_vad_worker_turn_clip_correlation",
}


def semantic_response_audio_policy() -> dict[str, object]:
    return dict(_SEMANTIC_RESPONSE_AUDIO_POLICY)


def valid_semantic_response_audio_policy(value: Any) -> bool:
    return isinstance(value, dict) and value == _SEMANTIC_RESPONSE_AUDIO_POLICY


def effective_voice_stack_identity(config: Mapping[str, Any]) -> dict[str, object]:
    """Fingerprint the secret-free semantic stack resolved for one call."""

    transcriber = config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}
    prompt = str(config.get("system_prompt") or "")
    configuration: dict[str, object] = {
        "providers": {
            "stt": str(transcriber.get("registry_name") or ""),
            "llm": str(model.get("registry_name") or ""),
            "tts": str(voice.get("registry_name") or ""),
        },
        "stt": {
            "model_id": str(transcriber.get("model") or ""),
            "language": str(transcriber.get("language") or config.get("language") or ""),
        },
        "llm": {
            "model_id": str(model.get("model") or ""),
            "system_prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
        },
        "tts": {
            "model_id": str(voice.get("model") or ""),
            "voice_id": str(voice.get("voice_id") or ""),
            "language": str(voice.get("language") or config.get("language") or ""),
        },
        "turns": {"language": str(config.get("language") or "")},
        "measurement": {
            "semantic_response_audio_policy": semantic_response_audio_policy(),
        },
    }
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "schema": "openphonex.effective_voice_stack.v1",
        "fingerprint": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "configuration": configuration,
    }


def valid_effective_voice_stack_identity(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema") != "openphonex.effective_voice_stack.v1":
        return False
    configuration = value.get("configuration")
    if not isinstance(configuration, dict):
        return False
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    return value.get("fingerprint") == expected

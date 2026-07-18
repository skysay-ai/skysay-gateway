# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Per-turn latency evidence for the LiveKit production path.

LiveKit exposes the authoritative internal timing anchors on the user and
assistant ``ChatMessage.metrics`` mappings.  This module pairs those mappings
without importing the LiveKit SDK, keeps provider details out of the artifact,
and leaves caller-side carrier/playout residuals to the independent PR #79
acoustic measurement.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


_USER_FIELDS = ("transcription_delay", "end_of_turn_delay")
_ASSISTANT_FIELDS = (
    "llm_node_ttft",
    "tts_node_ttfb",
    "playback_latency",
    "e2e_latency",
)


def _seconds(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def latency_breakdown(
    user_metrics: Mapping[str, Any],
    assistant_metrics: Mapping[str, Any],
    *,
    caller_side_voice_to_voice_ms: float | None = None,
) -> dict[str, float | None]:
    """Return one non-overlapping stage view plus the measured LiveKit total.

    LiveKit's ``end_of_turn_delay`` spans speech stop through turn commit, while
    ``transcription_delay`` spans speech stop through final transcript.  Their
    difference is therefore the post-final semantic endpoint wait.  With
    preemptive generation enabled the provider stages can overlap the commit;
    ``worker_e2e_ms`` remains the authoritative internal total.
    """

    transcription = _seconds(user_metrics, "transcription_delay")
    end_of_turn = _seconds(user_metrics, "end_of_turn_delay")
    llm = _seconds(assistant_metrics, "llm_node_ttft")
    tts = _seconds(assistant_metrics, "tts_node_ttfb")
    playback = _seconds(assistant_metrics, "playback_latency")
    e2e = _seconds(assistant_metrics, "e2e_latency")
    endpoint_after_final = (
        max(0.0, end_of_turn - transcription)
        if end_of_turn is not None and transcription is not None
        else None
    )
    caller_side = (
        float(caller_side_voice_to_voice_ms)
        if caller_side_voice_to_voice_ms is not None
        and math.isfinite(float(caller_side_voice_to_voice_ms))
        and float(caller_side_voice_to_voice_ms) >= 0
        else None
    )
    carrier_residual = (
        max(0.0, caller_side - e2e * 1_000)
        if caller_side is not None and e2e is not None
        else None
    )
    return {
        "speech_end_to_stt_final_ms": _milliseconds(transcription),
        "stt_final_to_turn_commit_ms": _milliseconds(endpoint_after_final),
        "speech_end_to_turn_commit_ms": _milliseconds(end_of_turn),
        "llm_ttft_ms": _milliseconds(llm),
        "tts_ttfb_ms": _milliseconds(tts),
        "worker_playback_ms": _milliseconds(playback),
        "worker_e2e_ms": _milliseconds(e2e),
        "caller_side_voice_to_voice_ms": _rounded(caller_side),
        "carrier_and_remote_playout_residual_ms": _rounded(carrier_residual),
    }


def _milliseconds(value: float | None) -> float | None:
    return _rounded(value * 1_000) if value is not None else None


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


class TurnLatencyRecorder:
    """Pair user/assistant per-turn metrics behind a small recorder interface."""

    def __init__(self) -> None:
        self._pending_users: list[dict[str, Any]] = []
        self._turns: list[dict[str, Any]] = []

    def observe_user(self, *, text: str, metrics: Mapping[str, Any] | None) -> None:
        self._pending_users.append(
            {
                "text_characters": len(" ".join(str(text or "").split())),
                "metrics": self._safe_metrics(metrics or {}, _USER_FIELDS),
            }
        )

    def observe_assistant(
        self,
        *,
        text: str,
        metrics: Mapping[str, Any] | None,
        semantic: bool,
    ) -> dict[str, Any] | None:
        if not semantic or not self._pending_users:
            return None
        user = self._pending_users.pop(0)
        assistant = self._safe_metrics(metrics or {}, _ASSISTANT_FIELDS)
        turn = {
            "turn": len(self._turns) + 1,
            "user_text_characters": user["text_characters"],
            "assistant_text_characters": len(" ".join(str(text or "").split())),
            "user_metrics": user["metrics"],
            "assistant_metrics": assistant,
            "breakdown": latency_breakdown(user["metrics"], assistant),
        }
        self._turns.append(turn)
        return dict(turn)

    def finalize(self, path: Path) -> dict[str, Any]:
        payload = {
            "schema": "openphonex.livekit_turn_latency.v1",
            "measurement_source": "livekit_chat_message_metrics",
            "caller_side_residual_requires_pr79_correlation": True,
            "turn_count": len(self._turns),
            "turns": list(self._turns),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return payload

    @staticmethod
    def _safe_metrics(metrics: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, float]:
        return {
            field: value
            for field in fields
            if (value := _seconds(metrics, field)) is not None
        }

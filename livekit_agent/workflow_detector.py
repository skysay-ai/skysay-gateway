# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The qualified finite-input detector shared by workflow and free-form calls."""
from __future__ import annotations

try:
    from .prewarm import SILERO_VAD_PINNED_PATH
    from .workflow_input import MAX_VAD_SPEECH_BUFFER_SECONDS
except ImportError:
    from prewarm import SILERO_VAD_PINNED_PATH
    from workflow_input import MAX_VAD_SPEECH_BUFFER_SECONDS


def load_workflow_detector(policy):
    from livekit.plugins import silero
    vad = policy["vad"]
    return silero.VAD.load(onnx_file_path=SILERO_VAD_PINNED_PATH,
        max_buffered_speech=MAX_VAD_SPEECH_BUFFER_SECONDS,
        min_speech_duration=vad["min_speech_ms"] / 1000,
        min_silence_duration=vad["min_silence_ms"] / 1000,
        prefix_padding_duration=vad["prefix_padding_ms"] / 1000,
        activation_threshold=vad["activation_threshold"], sample_rate=16000)

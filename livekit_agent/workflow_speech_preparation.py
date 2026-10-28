# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Engine-specific source proof behind one segment delivery lifecycle."""
from __future__ import annotations

from typing import Any, Protocol

try:
    from .workflow_speech import AuthoredSpeech, SpeechBinding, WorkflowSpeechGate
except ImportError:
    from workflow_speech import AuthoredSpeech, SpeechBinding, WorkflowSpeechGate


class SpeechPreparation(Protocol):
    async def prepare(self, gate: WorkflowSpeechGate, binding: SpeechBinding, owner: Any, *, operation):
        """Own a complete source and its explicitly typed backend approval."""
        ...

    def completed(self, gate: WorkflowSpeechGate, speech: AuthoredSpeech) -> None: ...

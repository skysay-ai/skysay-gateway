from __future__ import annotations

from dataclasses import dataclass

from .frames import TranscriptEvent


@dataclass(frozen=True)
class SemanticEndpointConfig:
    enabled: bool = True
    min_words: int = 2
    min_chars: int = 6
    terminal_punctuation: tuple[str, ...] = (".", "!", "?")


class SemanticEndpointDetector:
    """Conservative final-text endpointing.

    This is deliberately deterministic. Provider `speech_final` remains a
    strong endpointing signal, but the pipeline still uses this detector to
    reject tiny filler fragments before closing a turn.
    """

    def __init__(self, config: SemanticEndpointConfig | None = None):
        self.config = config or SemanticEndpointConfig()

    def should_endpoint(self, event: TranscriptEvent) -> bool:
        if not self.config.enabled or not event.is_final:
            return False
        text = " ".join(event.text.strip().split())
        if not text:
            return False
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        if metadata.get("semantic_endpoint") is True:
            return True
        if metadata.get("semantic_endpoint") is False:
            return False
        if len(text) < self.config.min_chars:
            return False
        if len(text.split()) < self.config.min_words:
            return False
        return text.endswith(self.config.terminal_punctuation)

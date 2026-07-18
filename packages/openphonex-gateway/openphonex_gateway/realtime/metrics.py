# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricEvent:
    name: str
    at: float
    attrs: dict[str, Any] = field(default_factory=dict)
    turn: int = 0


class CallMetrics:
    """Latency event log for a single realtime call session.

    Events are stamped with the active conversation turn index (0 = before the
    first caller turn, e.g. the outbound opening utterance) so latency spans can
    be reported per turn instead of repeating the first turn's numbers for every
    turn in ``transcript.json``.
    """

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._turn = 0
        self.events: list[MetricEvent] = []

    def begin_turn(self, turn_index: int) -> None:
        with self._lock:
            self._turn = max(0, int(turn_index))

    @property
    def current_turn(self) -> int:
        with self._lock:
            return self._turn

    def mark(self, name: str, *, at: float | None = None, **attrs: Any) -> MetricEvent:
        with self._lock:
            event = MetricEvent(
                name=name,
                at=self._clock() if at is None else float(at),
                attrs=attrs,
                turn=self._turn,
            )
            self.events.append(event)
        return event

    def first(self, name: str, *, turn: int | None = None) -> MetricEvent | None:
        with self._lock:
            for event in self.events:
                if event.name == name and (turn is None or event.turn == turn):
                    return event
        return None

    def last(self, name: str, *, turn: int | None = None) -> MetricEvent | None:
        with self._lock:
            for event in reversed(self.events):
                if event.name == name and (turn is None or event.turn == turn):
                    return event
        return None

    def duration_ms(self, start: str, end: str, *, turn: int | None = None) -> float | None:
        start_event = self.first(start, turn=turn)
        end_event = self.first(end, turn=turn)
        if not start_event or not end_event:
            return None
        return round(max(0.0, end_event.at - start_event.at) * 1000, 3)

    def turn_indices(self) -> list[int]:
        with self._lock:
            return sorted({event.turn for event in self.events if event.turn > 0})

    def latency_breakdown(self, *, turn: int | None = None) -> dict[str, float | None]:
        return {
            "speech_start_to_endpoint_ms": self.duration_ms("caller_speech_started", "user_endpoint_detected", turn=turn),
            "speech_start_to_first_transcript_delta_ms": self.duration_ms("caller_speech_started", "first_transcript_delta", turn=turn),
            "speech_start_to_first_rtp_ms": self.duration_ms("caller_speech_started", "first_rtp_packet_sent", turn=turn),
            "e2e_user_stop_to_first_rtp_ms": self.duration_ms("user_endpoint_detected", "first_rtp_packet_sent", turn=turn),
            "caller_hears_first_audio_estimate_ms": self.duration_ms("user_endpoint_detected", "caller_hears_first_audio_estimate", turn=turn),
            "voice_to_voice_ms": self._voice_to_voice_ms(turn=turn),
            "barge_in_cancel_ms": self.duration_ms("barge_in_speech_detected", "barge_in_output_cancelled", turn=turn),
            "asr_ms": self.duration_ms("stt_request_start", "final_transcript", turn=turn),
            "transcript_delta_to_final_transcript_ms": self.duration_ms("first_transcript_delta", "final_transcript", turn=turn),
            "endpoint_to_final_transcript_ms": self.duration_ms("user_endpoint_detected", "final_transcript", turn=turn),
            "final_transcript_to_llm_request_ms": self.duration_ms("final_transcript", "llm_request_start", turn=turn),
            "endpoint_to_llm_request_ms": self.duration_ms("user_endpoint_detected", "llm_request_start", turn=turn),
            "llm_first_chunk_ms": self.duration_ms("llm_request_start", "first_llm_chunk", turn=turn),
            "llm_first_speakable_phrase_ms": self.duration_ms("llm_request_start", "first_speakable_phrase", turn=turn),
            "llm_first_chunk_to_first_speakable_phrase_ms": self.duration_ms("first_llm_chunk", "first_speakable_phrase", turn=turn),
            "endpoint_to_first_speakable_phrase_ms": self.duration_ms("user_endpoint_detected", "first_speakable_phrase", turn=turn),
            "tts_first_audio_ms": self.duration_ms("tts_request_start", "first_tts_audio_byte", turn=turn),
            "first_phrase_to_tts_audio_ms": self.duration_ms("first_speakable_phrase", "first_tts_audio_byte", turn=turn),
            "transport_ms": self.duration_ms("first_tts_audio_byte", "first_rtp_packet_sent", turn=turn),
        }

    def _voice_to_voice_ms(self, *, turn: int | None = None) -> float | None:
        """True caller-speech-end to first outbound audio frame of the reply.

        Uses the ``caller_speech_ended`` event, whose timestamp is derived from
        the detector's recorded media-time ``end_ms`` attrs rather than the
        (later) ``user_endpoint_detected`` stamp time.
        """

        first_rtp = self.first("first_rtp_packet_sent", turn=turn)
        speech_end = self.last("caller_speech_ended", turn=turn)
        if not speech_end or not first_rtp:
            return None
        return round(max(0.0, first_rtp.at - speech_end.at) * 1000, 3)

    def latency_breakdown_by_turn(self) -> list[dict[str, Any]]:
        return [{"turn": turn, "latency": self.latency_breakdown(turn=turn)} for turn in self.turn_indices()]

    def primary_turn_latency(self) -> dict[str, float | None]:
        """Compatibility latency payload anchored to the first caller turn.

        Turn zero may contain the opening greeting. Selecting events without a
        turn can otherwise combine that greeting's first RTP packet with a
        later caller endpoint and clamp the negative span to a bogus ``0.0``.
        """

        turns = self.turn_indices()
        return self.latency_breakdown(turn=turns[0] if turns else 1)

    def turn_latency_summary(self) -> dict[str, Any]:
        """Distribution of worker-event diagnostics across caller turns."""

        turns = self.turn_indices()
        per_turn = [self.latency_breakdown(turn=turn) for turn in turns]
        metric_names = list(self.latency_breakdown(turn=turns[0] if turns else 1))
        summary: dict[str, dict[str, float | int]] = {}
        for name in metric_names:
            values = sorted(float(item[name]) for item in per_turn if item.get(name) is not None)
            if not values:
                continue
            summary[name] = {
                "count": len(values),
                "min": round(values[0], 3),
                "p50": _percentile(values, 0.50),
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
                "max": round(values[-1], 3),
            }
        return {
            "measurement_source": "worker_events_diagnostic",
            "turn_count": len(turns),
            "metrics": summary,
        }

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            events = list(self.events)
        return {
            "measurement_source": "worker_events_diagnostic",
            "events": [{"name": event.name, "at": event.at, "attrs": event.attrs, "turn": event.turn} for event in events],
            "latency": self.primary_turn_latency(),
            "latency_summary": self.turn_latency_summary(),
            "turns": self.latency_breakdown_by_turn(),
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return round(values[0], 3)
    position = max(0.0, min(1.0, quantile)) * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return round(values[lower] + (values[upper] - values[lower]) * fraction, 3)

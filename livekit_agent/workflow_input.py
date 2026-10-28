# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive input provenance for a root-owned spoken workflow coordinator.

The caller audio clock and the controller's local presentation boundary decide
eligibility. ASR words, provider captions and delayed SDK callbacks cannot
authorize answering a question. This is not proof that a person heard it.
"""
from __future__ import annotations

# Detector event storage is independent of the continuous recognition source.
MAX_VAD_SPEECH_BUFFER_SECONDS = 20

#: Caller audio the RTC ingress has delivered but capture has not consumed
#: yet. Past this the call ends as capacity: it never discards caller audio
#: to keep up, and it bounds what the SDK queue retains.
MAX_CALLER_INGRESS_BACKLOG_SECONDS = 10

from dataclasses import dataclass, field
import math
import time

try:
    from .capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS
except ImportError:
    from capabilities.workflow_listener import MAX_CALLER_TURN_CHARACTERS


class WorkflowInputRefused(RuntimeError):
    pass


class WorkflowMediaEnded(WorkflowInputRefused):
    """The selected RTC media owner ended; this conveys no final caller words."""


class WorkflowInputCapacityExceeded(WorkflowInputRefused):
    """Controlled end under sustained load; caller provenance is not invalid."""


class WorkflowInputBacklogExceeded(WorkflowInputCapacityExceeded):
    """The caller ingress backlog passed its cap; carries the measured backlog."""

    def __init__(self, *, backlog_samples: int, sample_rate: int) -> None:
        self.backlog_seconds = round(backlog_samples / sample_rate, 3)
        super().__init__("workflow caller ingress backlog requires a controlled call end")


@dataclass(frozen=True, slots=True)
class CallerTurnTiming:
    """Where one recognised caller turn spent its time, on the worker's monotonic clock.

    ``voiced_end_sample``: the capture sample where the VAD heard the voice
    stop (its END less the trailing silence it measured), on the sample clock;
    ``sealed_at``: capture sealed the turn (the silence hold ran out);
    ``input_ended_at``: the listener was sent its input EOF;
    ``provider_final_at``: the provider's last non-blank final arrived here;
    ``listener_closed_at``: the provider closed cleanly and released the words.
    ``None`` is a stage nothing reported, never zero. Evidence only: no
    decision reads it, and it never reaches the wire.
    """
    voiced_end_sample: int | None = None
    sealed_at: float | None = None
    input_ended_at: float | None = None
    provider_final_at: float | None = None
    listener_closed_at: float | None = None


@dataclass(frozen=True, slots=True)
class CallerInput:
    text: str = field(repr=False)
    input_epoch: int
    first_sample: int
    end_sample: int
    answer_to_request_id: str | None
    turn_first_sample: int
    continues: bool
    timing: CallerTurnTiming | None = field(default=None, repr=False, compare=False)

    def wire(self, *, sample_rate: int) -> dict:
        if self.continues or self.turn_first_sample != self.first_sample:
            raise WorkflowInputRefused("partial caller input cannot advance the business workflow")
        value = {"kind": "interrupt" if self.interrupt_only else "answer", "text": self.text,
                 "origin": {"input_epoch": self.input_epoch, "first_sample": self.first_sample,
                            "end_sample": self.end_sample, "sample_rate": sample_rate}}
        if not self.interrupt_only:
            value["answers_request_id"] = self.answer_to_request_id
        return value

    @property
    def interrupt_only(self) -> bool:
        return self.answer_to_request_id is None


@dataclass(frozen=True, slots=True)
class BlankInput:
    input_epoch: int
    first_sample: int
    end_sample: int
    turn_first_sample: int
    continues: bool

    def wire(self, *, sample_rate: int) -> dict:
        if self.continues or self.turn_first_sample != self.first_sample:
            raise WorkflowInputRefused("partial caller input cannot resume the business workflow")
        return {"kind": "resume", "origin": {"input_epoch": self.input_epoch,
            "first_sample": self.first_sample, "end_sample": self.end_sample, "sample_rate": sample_rate}}


@dataclass(frozen=True, slots=True)
class QuestionPresentation:
    request_id: str
    input_epoch: int
    boundary_sample: int
    sample_rate: int

    def wire(self) -> dict[str, int]:
        return {"input_epoch": self.input_epoch, "boundary_sample": self.boundary_sample,
                "sample_rate": self.sample_rate}


@dataclass(frozen=True, slots=True)
class _Utterance:
    first_sample: int
    start_observed: int
    answer_to_request_id: str | None
    turn_first_sample: int


@dataclass(frozen=True, slots=True)
class _SealedUtterance:
    utterance: _Utterance
    end_sample: int
    continues: bool


class WorkflowInputFence:
    """One input stream and utterance owner; no PCM, transcript queue or I/O.

    Only the root caller-input adapter supplies absolute sample indices. Its
    VAD stream must share this input's clock, and a stream flush/reset requires
    closing this owner. Question completion is called only after every segment
    has its durable completed_local acknowledgement, outside SDK generation.
    """

    def __init__(self, *, sample_rate: int, input_epoch: int, queued_audio_samples: int,
                 prefix_padding_samples: int, monotonic=time.monotonic, ingress_frontier=None) -> None:
        if (type(sample_rate) is not int or not 8000 <= sample_rate <= 48000
                or type(input_epoch) is not int or not 0 <= input_epoch <= 2147483647
                or type(queued_audio_samples) is not int or not 0 <= queued_audio_samples <= sample_rate * 2
                or type(prefix_padding_samples) is not int or not 0 <= prefix_padding_samples <= sample_rate):
            raise ValueError("input requires a supported sample clock and input epoch")
        self.sample_rate, self.input_epoch = sample_rate, input_epoch
        self.queued_audio_samples = queued_audio_samples
        self.prefix_padding_samples = prefix_padding_samples
        self._ingress_frontier = ingress_frontier
        self._received = 0
        self._monotonic = monotonic
        self._received_at = None
        self._retired_end = 0
        self._request: str | None = None
        self._last_turn = -1
        self._seen: set[str] = set()
        self._boundary: int | None = None
        self._presentation: QuestionPresentation | None = None
        self._active: _Utterance | None = None
        self._pending: list[_SealedUtterance] = []
        self._closed = False

    @property
    def received_samples(self) -> int:
        return self._received

    @property
    def received_frame(self) -> tuple[int, float | None]:
        """Latest physical received-sample frontier and its local clock, no PCM."""
        return self._received, self._received_at

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True
        self._active = None
        self._pending.clear()
        self._boundary = None
        self._presentation = None

    def _require(self, condition: bool) -> None:
        if self._closed or not condition:
            self.close()
            raise WorkflowInputRefused("caller input has no current presentation provenance")

    def receive_frame(self, *, samples: int, sample_rate: int) -> None:
        self._require(type(sample_rate) is int and sample_rate == self.sample_rate
                      and type(samples) is int and 0 < samples <= self.sample_rate // 5
                      and self._received + samples < 2**53)
        observed_at = self._monotonic()
        self._require(type(observed_at) in (int, float) and math.isfinite(observed_at)
                      and (self._received_at is None or observed_at >= self._received_at))
        self._received += samples
        self._received_at = observed_at

    async def frontier(self) -> int:
        """The first sample index no audio that reached this worker can precede.

        "Reached" is as of the moment this is called; audio arriving while
        the ingress drains is after that moment, so it may index past it.

        Consumption is FIFO and never discards, so the k-th delivered sample
        is consumed as index k: every sample the ingress delivered before this
        instant indexes below ``delivered + queued_audio_samples``, the slack
        left for audio still inside the native SDK. That is
        ``received + (delivered - received) + slack``: the consumed count plus
        slack alone is too early whenever the ingress holds a backlog. A fence
        with no RTC ingress has nothing delivered beyond what it received.
        """
        delivered = self._received if self._ingress_frontier is None else await self._ingress_frontier()
        self._require(type(delivered) is int and delivered >= self._received
                      and delivered + self.queued_audio_samples < 2**53)
        return delivered + self.queued_audio_samples

    def begin_question(self, *, request_id: str, turn_index: int) -> None:
        self._require(isinstance(request_id, str) and 16 <= len(request_id) <= 96
                      and request_id.isascii() and all(c.isalnum() or c in "_-" for c in request_id)
                      and request_id not in self._seen and len(self._seen) < 201
                      and type(turn_index) is int and self._last_turn < turn_index <= 200)
        self._seen.add(request_id)
        self._request, self._last_turn, self._boundary = request_id, turn_index, None
        self._presentation = None

    def prepare_question_presentation(self, *, request_id: str, input_epoch: int,
                                      frontier: int | None = None) -> QuestionPresentation:
        """Freeze one guarded frontier after all segment completion ACKs.

        An RTC-owned fence requires ``frontier`` from :meth:`frontier`, taken
        after those ACKs; without an ingress the consumed frontier is exact.
        """
        if frontier is None:
            self._require(self._ingress_frontier is None)
            frontier = self._received + self.queued_audio_samples
        self._require(self._request is not None and request_id == self._request
                      and type(input_epoch) is int and input_epoch == self.input_epoch
                      and type(frontier) is int and frontier >= self._received + self.queued_audio_samples
                      and frontier < 2**53)
        if self._presentation is None:
            self._presentation = QuestionPresentation(request_id, input_epoch, frontier, self.sample_rate)
        return self._presentation

    def completed_local_question(self, presentation: QuestionPresentation) -> None:
        """Activate only the same frontier whose durable whole-reply ACK joined."""
        self._require(self._presentation is not None and presentation is self._presentation)
        self._boundary = presentation.boundary_sample

    def question_presented(self, request_id: str) -> bool:
        return not self._closed and request_id == self._request and self._boundary is not None

    def retains_answer_permission(self, request_id: str, first_sample: int) -> bool:
        """Confirm an existing grant after coalescing; never grant new input."""
        return (not self._closed and isinstance(request_id, str) and request_id == self._request
                and self._boundary is not None and type(first_sample) is int
                and first_sample >= self._boundary)

    def speech_started(self, *, first_sample: int, start_observed: int, input_epoch: int) -> _Utterance:
        """Order START observations separately from the complete recognized PCM.

        The frozen VAD prefix can overlap a prior END. Keep that earlier audio
        in the origin: trimming it could authorize an answer begun too soon.
        """
        self._require(self._active is None and len(self._pending) < 4
                      and type(first_sample) is int
                      and type(start_observed) is int
                      and self._retired_end < start_observed <= self._received
                      and max(0, self._retired_end - self.prefix_padding_samples) <= first_sample < start_observed
                      and type(input_epoch) is int and input_epoch == self.input_epoch)
        answer_to = (self._request if self._boundary is not None
                     and first_sample >= self._boundary else None)
        self._active = _Utterance(first_sample, start_observed, answer_to, first_sample)
        return self._active

    @property
    def active_turn_first_sample(self) -> int | None:
        """The first sample of the logical caller turn being heard now, or None."""
        return self._active.turn_first_sample if self._active is not None else None

    def speech_ended(self, utterance: _Utterance, *, end_sample: int, continues: bool = False) -> _SealedUtterance:
        self._require(utterance is self._active and self._active is not None
                      and type(end_sample) is int and type(continues) is bool
                      and utterance.first_sample < end_sample <= self._received
                      and utterance.start_observed <= end_sample and end_sample > self._retired_end)
        self._active = None
        self._retired_end = end_sample
        sealed = _SealedUtterance(utterance, end_sample, continues)
        self._pending.append(sealed)
        return sealed

    def continue_speech(self, sealed: _SealedUtterance) -> _Utterance:
        """Continue a duration-bounded turn without upgrading its original onset."""
        if len(self._pending) >= 4:
            self.close()
            raise WorkflowInputCapacityExceeded("workflow continuation backlog requires a controlled call end")
        self._require(self._active is None and bool(self._pending) and self._pending[-1] is sealed
                      and sealed.end_sample == self._retired_end and sealed.continues)
        self._active = _Utterance(sealed.end_sample, sealed.end_sample,
                                  sealed.utterance.answer_to_request_id, sealed.utterance.turn_first_sample)
        return self._active

    def finish_speech(self, sealed: _SealedUtterance, *, text: str | None) -> CallerInput | BlankInput:
        return self.finish_speech_parts(sealed, parts=((sealed.utterance.first_sample, sealed.end_sample, text),))[0]

    def finish_speech_parts(self, sealed: _SealedUtterance, *,
                            parts: tuple[tuple[int, int, str | None], ...]) -> tuple[CallerInput | BlankInput, ...]:
        # Recognition may finish out of order; the root adapter must drain it
        # in input order. Completion never owns the live microphone boundary.
        self._require(bool(self._pending) and sealed is self._pending[0])
        utterance = sealed.utterance
        self._require(type(parts) is tuple and 1 <= len(parts) <= 128)
        position = utterance.first_sample
        for part in parts:
            self._require(type(part) is tuple and len(part) == 3)
            first, end, text = part
            self._require(type(first) is int and type(end) is int and first == position
                          and first < end <= sealed.end_sample
                          and (text is None or isinstance(text, str) and len(text) <= MAX_CALLER_TURN_CHARACTERS))
            if text is not None:
                try:
                    text.encode("utf-8")
                except UnicodeError:
                    self._require(False)
            position = end
        self._require(position == sealed.end_sample)
        self._pending.pop(0)
        answer_to = (utterance.answer_to_request_id
                     if utterance.answer_to_request_id == self._request else None)
        return tuple(CallerInput(text, self.input_epoch, first, end, answer_to, utterance.turn_first_sample,
                                 index + 1 < len(parts) or sealed.continues) if text and text.strip()
                     else BlankInput(self.input_epoch, first, end, utterance.turn_first_sample,
                                     index + 1 < len(parts) or sealed.continues)
                     for index, (first, end, text) in enumerate(parts))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Immutable speech quarantine, with no audio or text output capability.

Only the owning authored TTS adapter's sealed PCM for its exact input becomes
playable speech. The controller owns durable delivery and public playback;
this gate cannot expose anything.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


#: The speech segments one workflow reply may claim, authored or live (a live
#: line is one segment): the bound every output checks before its first claim.
MAX_REPLY_SEGMENTS = 128


class SpeechRefused(RuntimeError):
    """No approval or complete delivery can be inferred from this attempt."""


class LineNotVoiced(SpeechRefused):
    """A workflow line before the call's ending was played but not said in full.

    The workflow cannot go on as if the caller had heard it: the call ends,
    and its ending names this (``workflow_line_undelivered``), never the
    caller leaving.
    """


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("speech identity must be a bounded nonblank string")


def speech_step_token(authored_id: str) -> str:
    """Bound a compiler-valid authored key to the private source's opaque ID domain."""
    if not isinstance(authored_id, str) or not authored_id.strip():
        raise ValueError("speech step requires its exact authored identifier")
    return hashlib.sha256(b"workflow-step\0" + authored_id.encode("utf-8")).hexdigest()


def _counter(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("speech counter must be a nonnegative integer")


@dataclass(frozen=True)
class SpeechLine:
    call_id: str
    run_id: str
    turn_index: int
    step_id: str
    text: str = field(repr=False)
    segments: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for value in (self.call_id, self.run_id, self.step_id):
            _identifier(value)
        _counter(self.turn_index)
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 16000:
            raise ValueError("speech line must contain bounded text")
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("speech line must be valid UTF-8 text") from exc
        if (type(self.segments) is not tuple or not 1 <= len(self.segments) <= MAX_REPLY_SEGMENTS
                or any(not isinstance(s, str) or not s.strip() for s in self.segments)
                or "".join(self.segments) != self.text):
            raise ValueError("segments must be an exact lossless partition of the line")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpeechBinding:
    call_id: str
    run_id: str
    turn_index: int
    step_id: str
    line_digest: str
    ordinal: int
    media_epoch: int
    attempt_id: str


@dataclass(frozen=True)
class SpeechCandidate:
    binding: SpeechBinding
    text: str = field(repr=False)
    pcm: bytes = field(repr=False)
    sample_rate: int
    samples: int
    caption: str = field(repr=False)

    @property
    def pcm_digest(self) -> str:
        return hashlib.sha256(self.pcm).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class AuthoredSpeech:
    """Configured TTS's closed PCM for its exact authored input.

    The owning TTS adapter establishes this fact. It cannot authorize a
    native conversation engine.
    """
    candidate: SpeechCandidate = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise SpeechRefused("authored speech must be minted by its owning gate")


def _mint_speech(kind, candidate):
    speech = object.__new__(kind)
    object.__setattr__(speech, "candidate", candidate)
    return speech


class WorkflowSpeechGate:
    """One lossless line, immutable candidate, epoch and bounded private PCM."""

    def __init__(self, line: SpeechLine, *, media_epoch: int,
                 sample_rate: int = 24000, max_seconds: float = 15.0,
                 max_frames: int = 2048, capture_timeout: float = 30.0) -> None:
        _counter(media_epoch)
        if not isinstance(line, SpeechLine):
            raise ValueError("speech needs an immutable line")
        if type(sample_rate) is not int or not 8000 <= sample_rate <= 48000:
            raise ValueError("unsupported PCM sample rate")
        if (type(max_frames) is not int or not 1 <= max_frames <= 4096
                or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= 30
                or type(capture_timeout) not in (int, float) or not math.isfinite(capture_timeout)
                or not max_seconds < capture_timeout <= 60):
            raise ValueError("speech buffers and capture must be bounded")
        self._line, self._media_epoch, self._rate = line, media_epoch, sample_rate
        self._max_bytes = int(max_seconds * sample_rate) * 2
        self._max_frames, self._timeout = max_frames, capture_timeout
        self._ordinal = 0
        self._binding: SpeechBinding | None = None
        self._frames: list[bytes] = []
        self._bytes = 0
        self._caption = ""
        self._candidate: SpeechCandidate | None = None
        self._authored: AuthoredSpeech | None = None
        self._proof_kind: str | None = None
        self._closed = False
        self._expires = math.inf
        self._now = time.monotonic

    @property
    def line(self) -> SpeechLine:
        return self._line

    @property
    def media_epoch(self) -> int:
        return self._media_epoch

    def open_segment(self, *, attempt_id: str | None = None) -> SpeechBinding:
        if self._closed or self._binding is not None or self._ordinal >= len(self.line.segments):
            raise SpeechRefused("speech line has no available segment")
        attempt_id = uuid4().hex if attempt_id is None else attempt_id
        _identifier(attempt_id)
        self._binding = SpeechBinding(self.line.call_id, self.line.run_id, self.line.turn_index,
                                      self.line.step_id, self.line.digest, self._ordinal,
                                      self.media_epoch, attempt_id)
        self._expires = self._now()+self._timeout
        return self._binding

    def _current(self, binding: SpeechBinding, *, check_expiry: bool = True) -> None:
        if self._closed or self._binding is None or binding != self._binding:
            raise SpeechRefused("stale or absent speech authorization")
        if check_expiry and self._now() >= self._expires:
            self._fail("speech candidate expired")

    def _discard(self) -> None:
        self._frames.clear()
        self._bytes = 0
        self._caption = ""
        self._candidate = None
        self._authored = None

    def invalidate(self) -> None:
        """Synchronous barrier: a late producer loses authority first."""
        if not self._closed:
            self._media_epoch += 1
        self._closed = True
        self._discard()

    async def interrupt(self) -> None:
        # This gate owns no asynchronous writer or customer sink. The caller
        # still joins its producer and publisher before replacing their owner.
        self.invalidate()

    def _fail(self, reason: str) -> None:
        self.invalidate()
        raise SpeechRefused(reason)

    def source_text(self, binding: SpeechBinding) -> str:
        self._current(binding)
        if self._candidate is not None:
            raise SpeechRefused("speech source is already sealed")
        return self.line.segments[self._ordinal]

    def capture_frame(self, binding: SpeechBinding, frame: Any) -> None:
        self._current(binding)
        if self._candidate is not None:
            self._fail("audio arrived beyond the complete source boundary")
        try:
            count, rate, channels = frame.samples_per_channel, frame.sample_rate, frame.num_channels
            view = memoryview(frame.data)
            size = view.nbytes
        except Exception:
            self._fail("malformed speech frame")
        if (type(count) is not int or count <= 0 or type(rate) is not int or rate != self._rate
                or type(channels) is not int or channels != 1 or size != count*2):
            self._fail("speech frame is not the declared mono PCM format")
        if len(self._frames) >= self._max_frames or self._bytes+size > self._max_bytes:
            self._fail("speech quarantine overflow")
        try:
            data = view.tobytes()
        except Exception:
            self._fail("speech PCM could not be copied")
        self._frames.append(data)
        self._bytes += len(data)

    def capture_text(self, binding: SpeechBinding, text: str) -> None:
        self._current(binding)
        if self._candidate is not None or not isinstance(text, str) or len(self._caption)+len(text) > 16000:
            self._fail("invalid or late speech caption")
        self._caption += text

    def seal_candidate(self, binding: SpeechBinding) -> SpeechCandidate:
        """Called only after the owned source establishes its complete boundary."""
        self._current(binding)
        if self._candidate is not None or not self._frames:
            self._fail("no complete unsealed speech candidate")
        self._candidate = SpeechCandidate(binding, self.line.segments[self._ordinal],
                                           b"".join(self._frames), self._rate, self._bytes//2, self._caption)
        self._frames.clear()
        return self._candidate

    def seal_authored(self, binding: SpeechBinding) -> AuthoredSpeech:
        if self._proof_kind not in (None, "authored_tts") or self._authored is not None:
            self._fail("speech line cannot mix source proof kinds")
        candidate = self.seal_candidate(binding)
        self._proof_kind = "authored_tts"
        self._authored = _mint_speech(AuthoredSpeech, candidate)
        return self._authored

    def completed_authored(self, speech: AuthoredSpeech) -> None:
        if not isinstance(speech, AuthoredSpeech) or speech is not self._authored:
            self._fail("authored completion has no owned source")
        self._current(speech.candidate.binding, check_expiry=False)
        self._advance_completed()

    def _advance_completed(self) -> None:
        self._ordinal += 1
        self._binding = None
        self._discard()

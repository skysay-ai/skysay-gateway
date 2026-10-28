# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One call's voicemail detection: feeds in, typed signals out, one record.

The stateful half of the voicemail detector.  ``voicemail_signals.py`` holds the
pure measurements and ``answer_observer.py`` the decision; this module owns
the per-call buffers and clocks that connect them, and it is what the worker
drives.  It imports no livekit.

Three disciplines, each from the adversarial reviews of 2026-09-23:

**Nothing heavy on the media drain.**  ``note_media_frame`` only appends to a
bounded ring.  ``process`` does the arithmetic from the worker's own task,
with a per-call frame budget, so a slow iteration defers work rather than
stalling media.  A full ring drops its oldest frame and the drop is a GAP.

**Our own voice is not the far end.**  Echo is judged on RECEIPT time for both
roles, because the same event loop stamps both, while the recorder's media
positions are per-stream cursors that can drift apart after a burst.  Agent
frames are processed before caller frames, and a caller frame is only judged
once agent frames up to its receipt are in (or the agent feed has gone quiet
long enough to call it stale).  With no agent feed at all, attribution is
impossible, and the watch refuses proof and human evidence rather than guess.

**Words belong to the audio they came from.**  A transcript final carries only
the time it was ingested.  It is bound to the far-end runs that ENDED since the
previous final, and only after the engine was allowed to hear the caller; words
bound to nothing verify nothing.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - import shape only
    from . import answer_observer as _observer
    from . import voicemail_signals as _signals
    from .sip_answer_gate import ALREADY_ACTIVE, ANSWERED
except ImportError:  # pragma: no cover - direct execution inside the image
    import answer_observer as _observer  # type: ignore[no-redef]
    import voicemail_signals as _signals  # type: ignore[no-redef]
    from sip_answer_gate import ALREADY_ACTIVE, ANSWERED  # type: ignore[no-redef]


SECTION_VERSION = 1

MODE_SHADOW = "shadow"
MODE_HANG_UP = "hang_up"
MODE_LEAVE_MESSAGE = "leave_message"
MODE_OFF = "off"
#: The per-agent setting's closed choice set, as the control plane publishes it.
MODES = (MODE_SHADOW, MODE_HANG_UP, MODE_LEAVE_MESSAGE, MODE_OFF)
DEFAULT_MODE = MODE_SHADOW
SETTING_KEY = "voicemail"
#: What every mode does in this release.  The acting modes are stored and
#: validated, and run as shadow until the worker that performs them ships;
#: the record says both, so nobody reads "hang_up" as "hung up".
EFFECTIVE_MODE = MODE_SHADOW

ARMING_DISPOSITIONS = frozenset({ANSWERED, ALREADY_ACTIVE})

#: How often the worker drains the rings (its own task, not the 1 s tick).
PROCESS_INTERVAL_SECONDS = 0.1
RING_FRAMES = 200
#: Frames processed per role per ``process`` call.  At a 100 ms cadence and 50
#: frames a second this keeps up with 1.2x real time, and bounds one
#: iteration's cost whatever backlog built up.
CALLER_BUDGET = 60
AGENT_BUDGET = 100
#: A frame whose receipt is this much later than its media end arrived in a
#: burst; its placement relative to our playout is not trustworthy.
UNALIGNED_SECONDS = 0.10
#: Media discontinuity that counts as a gap: anything beyond half a frame.
GAP_SECONDS = 0.010
ECHO_TAIL_SECONDS = _signals.ECHO_GUARD_SECONDS
UNALIGNED_ECHO_SECONDS = 1.0
#: No agent frame for this long while caller frames flow: the agent feed is
#: stale, and our own audio could be anywhere.
AGENT_STALE_SECONDS = 2.0
#: How long the caller side may wait for agent frames to catch up before it is
#: judged anyway (with the agent feed then counted stale).
AGENT_WAIT_SECONDS = 2.0
#: The agent is audible above this level: the recorder's own quiet floor
#: (``_PLAYOUT_TAIL_QUIET_RMS`` = 250 on a 16-bit scale).
AGENT_AUDIBLE_DBFS = 20.0 * math.log10(250.0 / 32768.0)
#: Full DSP (tones) runs until the classification window and the beep
#: look-ahead are over; after that only levels, which is what human evidence
#: needs for the rest of the call.
FULL_FEATURES_SECONDS = _observer.CLASSIFICATION_WINDOW_SECONDS + _signals.CADENCE_LOOKAHEAD_SECONDS + 1.0
FAST_ANSWER_MS = 1500
#: A non-human reading this close before a beep, or any time up to its
#: validation, vetoes it; one from earlier (a caller tune that gave way to a
#: mailbox greeting) does not.
NON_HUMAN_VETO_LOOKBACK_SECONDS = 4.0
PHRASE_FORWARD_VETO_SECONDS = 5.0
GREETING_WORDS_MIN = 3
#: Far-end activity: above the running floor by this much, for this long.
ACTIVE_ABOVE_FLOOR_DB = 10.0
RUN_MIN_FRAMES = 2
RUN_END_QUIET_SECONDS = 0.2
FLOOR_WINDOW_FRAMES = 150
HELLO_SECONDS = (0.2, 2.0)
HELLO_YIELD_SECONDS = 2.0
REPLY_WINDOW_SECONDS = (0.2, 2.5)
SPEECH_AFTER_BEEP_SECONDS = 0.5
GREETING_SHAPE_START_SECONDS = 3.0
GREETING_SHAPE_MIN_SECONDS = 4.0
AGENT_TEXT_SECONDS = 30.0
MAX_RUNS = 64
MAX_TONES = 16
MAX_PENDING_TEXTS = 32
MAX_AGENT_INTERVALS = 256

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
AGENT_FEED_OK = "ok"
AGENT_FEED_MISSING = "missing"
AGENT_FEED_STALE = "stale"


def resolved_setting(realtime_config: Any, *, emit: Callable[..., None] = lambda *_a, **_k: None) -> str:
    """This call's voicemail mode, or shadow when it cannot be read (logged).

    The control plane resolves the platform default before transmitting, so a
    missing key is a damaged or older config, and the reading of that which
    changes nothing is shadow.
    """

    section = realtime_config.get("answer_detection") if isinstance(realtime_config, dict) else None
    value = section.get(SETTING_KEY) if isinstance(section, dict) else None
    if isinstance(value, str) and value in MODES:
        return value
    emit("voicemail_mode_unreadable", present=value is not None, mode=DEFAULT_MODE)
    return DEFAULT_MODE


@dataclass(slots=True)
class _Frame:
    pcm: bytes
    media_start: float
    received_at: float
    gap: bool


@dataclass(slots=True)
class _Run:
    """A stretch of far-end activity on the caller track."""

    started_at: float
    received_start: float
    ended_at: float = 0.0
    received_end: float = 0.0
    closed: bool = False
    echo: bool = False
    gap: bool = False
    #: The finals bound to this run, by sequence number; their word counts
    #: live once in ``VoicemailWatch._final_words`` so a final bound to three
    #: runs is still counted once (Codex code review #5).
    finals: set = field(default_factory=set)
    automation_phrase: bool = False
    reply_to_agent: bool = False


@dataclass(slots=True)
class _Candidate:
    tone: _signals.Tone
    span: tuple[float, float] | None
    speech_shape: bool
    #: A clean tone after a mailbox PHRASE already proved the mailbox: it is
    #: the recording cue, and needs no greeting of its own behind it.
    phrase_cue: bool = False


@dataclass(slots=True)
class _PendingPhrase:
    at: float
    due: float
    vetoed: bool = False


@dataclass
class VoicemailWatch:
    """Per-call voicemail detection.  Every public method is best effort; the
    caller wraps them, and nothing here touches the call."""

    configured_mode: str = DEFAULT_MODE
    language: str = ""
    emit: Callable[..., None] = field(default=lambda *_a, **_k: None)

    _caller: deque = field(default_factory=lambda: deque(maxlen=RING_FRAMES))
    _agent: deque = field(default_factory=lambda: deque(maxlen=RING_FRAMES))
    _caller_overflow: int = 0
    _agent_overflow: int = 0
    _armed_at: float | None = None
    _disposition: str = ""
    _wait_ms: int | None = None
    _observer: _observer.AnswerObserver | None = None
    _input_released_at: float | None = None
    _opener_started_at: float | None = None
    _tracker: _signals.ToneTracker = field(default_factory=_signals.ToneTracker)
    _tones: list = field(default_factory=list)
    _candidates: list = field(default_factory=list)
    _agent_intervals: list = field(default_factory=list)
    _agent_last_received: float | None = None
    _agent_frames: int = 0
    _agent_stale_frames: int = 0
    _caller_frames: int = 0
    _caller_frames_after_arm: int = 0
    _caller_gaps: int = 0
    _caller_gap_seconds: float = 0.0
    _last_caller_media_end: float | None = None
    _caller_received_horizon: float | None = None
    #: Receipt times of caller gaps, for judging whether a window was heard.
    _caller_gap_at: list = field(default_factory=list)
    _floor: deque = field(default_factory=lambda: deque(maxlen=FLOOR_WINDOW_FRAMES))
    _runs: list = field(default_factory=list)
    _open_run: _Run | None = None
    _quiet_frames: int = 0
    _active_frames: int = 0
    _texts: deque = field(default_factory=lambda: deque(maxlen=MAX_PENDING_TEXTS))
    _texts_dropped: int = 0
    _agent_texts: deque = field(default_factory=lambda: deque(maxlen=16))
    _last_final_at: float | None = None
    _final_words: dict = field(default_factory=dict)
    _automation_final_at: list = field(default_factory=list)
    _non_human_readings: list = field(default_factory=list)
    _pending_phrases: list = field(default_factory=list)
    _states_standing: set = field(default_factory=set)
    _classifier_available: bool = False
    _speech_like_at: list = field(default_factory=list)
    _emitted_once: set = field(default_factory=set)
    _hello_run: _Run | None = None
    _validated_beeps: list = field(default_factory=list)
    _frozen: dict | None = None
    #: Why observation stopped before the record was frozen, or "".
    _stopped_by: str = ""

    def __post_init__(self) -> None:
        if self.configured_mode not in MODES:
            self.configured_mode = DEFAULT_MODE

    # -- state -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.configured_mode != MODE_OFF

    @property
    def active(self) -> bool:
        """Still collecting: enabled, not stopped early and not frozen."""

        return self.enabled and not self._stopped_by and self._frozen is None

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    @property
    def observer(self) -> _observer.AnswerObserver | None:
        return self._observer

    # -- feeds (cheap; called from media and event handlers) ---------------

    def note_media_frame(
        self, role: str, pcm: bytes, *, media_start: float, received_at: float, gap: bool = False
    ) -> None:
        if not self.enabled or self._frozen is not None or self._stopped_by:
            return
        ring = self._caller if role == "caller" else self._agent if role == "agent" else None
        if ring is None:
            return
        if len(ring) == ring.maxlen:
            if role == "caller":
                self._caller_overflow += 1
            else:
                self._agent_overflow += 1
            # The oldest frame is about to be lost: whatever comes after it
            # is no longer contiguous with what was processed.
            gap = True
        ring.append(_Frame(bytes(pcm), float(media_start), float(received_at), bool(gap)))

    def note_caller_final(self, text: Any, *, at: float) -> None:
        if not self.enabled or self._frozen is not None:
            return
        value = str(text or "").strip()
        if not value:
            return
        if len(self._texts) == self._texts.maxlen:
            self._texts_dropped += 1
        self._texts.append((float(at), value))

    def note_agent_text(self, text: Any, *, at: float) -> None:
        if not self.enabled or self._frozen is not None:
            return
        value = str(text or "").strip()
        if value:
            self._agent_texts.append((float(at), value))

    def note_input_released(self, *, at: float) -> None:
        if self._input_released_at is None:
            self._input_released_at = float(at)

    def note_opener_started(self, *, at: float) -> None:
        if self._opener_started_at is None:
            self._opener_started_at = float(at)

    def note_classification(self, *, verdict: str, speech_like: bool, at: float) -> None:
        """One reading of the audio classifier, which runs for this watch
        even when the non-human setting is off (it is not reported then)."""

        if not self.enabled or self._frozen is not None:
            return
        self._classifier_available = True
        if verdict == "non_human_audio":
            if len(self._non_human_readings) < 256:
                self._non_human_readings.append(float(at))
            self._signal(_signals.KIND_NON_HUMAN_AUDIO, at=float(at), known_at=float(at), once=True)
        if speech_like and len(self._speech_like_at) < 256:
            self._speech_like_at.append(float(at))

    def arm(self, disposition: str, *, wait_ms: int | None, at: float) -> None:
        if not self.enabled or self._armed_at is not None:
            return
        self._armed_at = float(at)
        self._disposition = str(disposition)
        self._wait_ms = int(wait_ms) if isinstance(wait_ms, int) and not isinstance(wait_ms, bool) else None
        self._observer = _observer.AnswerObserver(
            armed_at=self._armed_at, eligible=self._disposition in ARMING_DISPOSITIONS
        )
        if self._disposition == ANSWERED and self._wait_ms is not None and self._wait_ms <= FAST_ANSWER_MS:
            self._signal(_signals.KIND_FAST_ANSWER, at=self._armed_at, known_at=self._armed_at)

    # -- processing (the worker's own task) --------------------------------

    def process(self, *, now: float, drain: bool = False) -> None:
        """Measure what has arrived.  ``drain`` (teardown only) judges every
        queued caller frame without waiting for agent frames to catch up;
        it does NOT move the decision clock, so no window expires early."""

        if not self.enabled or self._frozen is not None or self._stopped_by:
            return
        if self._armed_at is None:
            # Before the answer the rings only hold the last few seconds (so a
            # greeting that starts AT the answer is complete); nothing is
            # measured on ringback.
            return
        armed_at = self._armed_at
        while self._agent and self._agent[0].received_at < armed_at - AGENT_WAIT_SECONDS:
            self._agent.popleft()
        # Caller media from before the answer is ringback or early media: it
        # must not be measured as a greeting, so it is dropped here.
        while self._caller and self._caller[0].media_start < armed_at:
            self._caller.popleft()
        # On the teardown drain every agent frame is judged first, whatever
        # the budget: a caller frame must never be judged against agent audio
        # still waiting in its ring (Grok code review).
        for _ in range(len(self._agent) if drain else min(AGENT_BUDGET, len(self._agent))):
            self._process_agent(self._agent.popleft())
        agent_horizon = self._agent_last_received
        processed = 0
        while self._caller and processed < CALLER_BUDGET:
            frame = self._caller[0]
            ready = (
                drain
                or (agent_horizon is not None and frame.received_at <= agent_horizon)
                or now - frame.received_at >= AGENT_WAIT_SECONDS
            )
            if not ready:
                break
            self._caller.popleft()
            self._process_caller(frame, now=now)
            processed += 1
        self._process_texts(now=now)
        self._resolve_candidates(now=now)
        self._resolve_phrases(now=now)
        self._check_hello(now=now)
        self._maybe_stop(now=now)

    def _full_features_until(self) -> float:
        """Tones are measured through the classification window and, once a
        proof stands, through its cue window too: leave_message waits up to
        that long for a record beep (Codex code review #9)."""

        armed_at = self._armed_at or 0.0
        until = armed_at + FULL_FEATURES_SECONDS
        observer = self._observer
        if observer is not None and observer.proof_known_at is not None:
            until = max(
                until,
                observer.proof_known_at + _observer.CUE_WINDOW_SECONDS + _signals.CADENCE_LOOKAHEAD_SECONDS,
            )
        return until

    def _maybe_stop(self, *, now: float) -> None:
        """Stop collecting once nothing more can change the record.

        After the classification window and the beep look-ahead, a call with
        no proof and nothing pending can no longer become voicemail, and the
        only thing more observation could add is human evidence -- which
        matters for the accuracy audit only after a would-be action.  So such
        a call stops here (most calls), and a call WITH proof keeps watching,
        on levels only, until the record is frozen.
        """

        armed_at = self._armed_at
        horizon = self._last_caller_media_end
        observer = self._observer
        if armed_at is None or observer is None or horizon is None:
            return
        if horizon - armed_at < FULL_FEATURES_SECONDS:
            return
        if self._candidates or self._pending_phrases or self._texts:
            return
        if observer.has_proof:
            return
        self._stopped_by = "window_closed_without_proof"
        self._caller.clear()
        self._agent.clear()

    def _process_agent(self, frame: _Frame) -> None:
        self._agent_frames += 1
        self._agent_last_received = frame.received_at
        level = _signals.frame_rms_dbfs(frame.pcm)
        if level is None or level < AGENT_AUDIBLE_DBFS:
            return
        start, end = frame.received_at - _signals.FRAME_SECONDS, frame.received_at
        if self._agent_intervals and start - self._agent_intervals[-1][1] <= 0.1:
            self._agent_intervals[-1] = (self._agent_intervals[-1][0], end)
        else:
            self._agent_intervals.append((start, end))
            if len(self._agent_intervals) > MAX_AGENT_INTERVALS:
                del self._agent_intervals[0]

    def _agent_audible_near(self, received_at: float, margin: float) -> bool:
        low, high = received_at - margin, received_at + margin
        return any(start <= high and low <= end + ECHO_TAIL_SECONDS for start, end in self._agent_intervals)

    def _is_echo(self, frame: _Frame) -> bool:
        """Whether our own playout may be in this caller frame, or we cannot
        tell (agent feed missing or stale): both refuse the frame as evidence."""

        if self._agent_last_received is None:
            return True
        if frame.received_at - self._agent_last_received > AGENT_STALE_SECONDS:
            self._agent_stale_frames += 1
            return True
        unaligned = frame.received_at - (frame.media_start + _signals.FRAME_SECONDS) > UNALIGNED_SECONDS
        if unaligned and self._agent_audible_near(frame.received_at, UNALIGNED_ECHO_SECONDS):
            return True
        return self._agent_audible_near(frame.received_at, 0.0)

    def _process_caller(self, frame: _Frame, *, now: float) -> None:
        self._caller_frames += 1
        gap = frame.gap
        if self._last_caller_media_end is not None:
            jump = frame.media_start - self._last_caller_media_end
            if jump > GAP_SECONDS:
                gap = True
                if self.armed:
                    self._caller_gaps += 1
                    self._caller_gap_seconds += jump
        elif frame.gap and self.armed:
            self._caller_gaps += 1
        self._last_caller_media_end = frame.media_start + _signals.FRAME_SECONDS
        self._caller_received_horizon = frame.received_at
        if gap and self.armed and len(self._caller_gap_at) < 256:
            self._caller_gap_at.append(frame.received_at)
        armed_at = self._armed_at
        full = armed_at is None or frame.media_start <= self._full_features_until()
        if full:
            features = _signals.frame_features(frame.pcm, at=frame.media_start, gap=gap)
        else:
            level = _signals.frame_rms_dbfs(frame.pcm)
            features = (
                None
                if level is None
                else _signals.FrameFeatures(
                    at=frame.media_start, rms_dbfs=level, frequency_hz=0.0, purity=0.0, gap=gap
                )
            )
        if features is None:
            return
        echo = self._is_echo(frame)
        if echo:
            features = _signals.FrameFeatures(
                at=features.at,
                rms_dbfs=features.rms_dbfs,
                frequency_hz=features.frequency_hz,
                purity=features.purity,
                dtmf=features.dtmf,
                gap=features.gap,
                echo=True,
            )
        if armed_at is not None and frame.media_start >= armed_at:
            self._caller_frames_after_arm += 1
        self._track_activity(features, frame, now=now)
        completed = self._tracker.feed(features)
        if completed is not None and armed_at is not None:
            self._on_tone(*completed, now=now)

    # -- far-end activity --------------------------------------------------

    def _track_activity(self, features: _signals.FrameFeatures, frame: _Frame, *, now: float) -> None:
        if not features.echo:
            self._floor.append(features.rms_dbfs)
        floor = sorted(self._floor)[len(self._floor) // 10] if self._floor else -96.0
        active = features.rms_dbfs >= floor + ACTIVE_ABOVE_FLOOR_DB
        run = self._open_run
        if active:
            self._active_frames += 1
            self._quiet_frames = 0
            if run is None and self._active_frames >= RUN_MIN_FRAMES:
                start = features.at - (RUN_MIN_FRAMES - 1) * _signals.FRAME_SECONDS
                run = _Run(started_at=start, received_start=frame.received_at)
                run.reply_to_agent = self._follows_agent_turn(frame.received_at)
                self._open_run = run
                # The far end spoke again before the yield was complete: that
                # "hello" was the start of something longer.
                self._hello_run = None
            if run is not None:
                run.ended_at = features.at + _signals.FRAME_SECONDS
                run.received_end = frame.received_at
                run.echo = run.echo or features.echo
                run.gap = run.gap or features.gap
        else:
            self._active_frames = 0
            self._quiet_frames += 1
            if run is not None:
                run.gap = run.gap or features.gap
                if self._quiet_frames * _signals.FRAME_SECONDS >= RUN_END_QUIET_SECONDS:
                    self._close_run(run, now=now)
        if features.gap and self._hello_run is not None:
            # A quiet that was not observed is not a yield.
            self._hello_run = None
            self._emitted_once.add("hello_checked")

    def _follows_agent_turn(self, received_at: float) -> bool:
        low, high = REPLY_WINDOW_SECONDS
        for _start, end in reversed(self._agent_intervals):
            if end <= received_at - low and received_at - end <= high:
                return not self._agent_audible_near(received_at, 0.0)
            if end < received_at - high:
                break
        return False

    def _close_run(self, run: _Run, *, now: float) -> None:
        run.closed = True
        self._open_run = None
        if self._armed_at is None or run.ended_at < self._armed_at:
            return
        if len(self._runs) >= MAX_RUNS:
            del self._runs[0]
        self._runs.append(run)
        seconds = run.ended_at - run.started_at
        clean = not run.echo and not run.gap
        if (
            "greeting_shape" not in self._emitted_once
            and clean
            and run.started_at - self._armed_at <= GREETING_SHAPE_START_SECONDS
            and seconds >= GREETING_SHAPE_MIN_SECONDS
        ):
            self._signal(_signals.KIND_GREETING_SHAPE, at=run.started_at, known_at=now, once=True)
        first_after_arm = "hello_checked" not in self._emitted_once
        if first_after_arm and run.started_at >= self._armed_at:
            self._emitted_once.add("hello_checked")
            if clean and HELLO_SECONDS[0] <= seconds <= HELLO_SECONDS[1]:
                self._hello_run = run
        if clean and run.reply_to_agent:
            self._signal(_signals.KIND_RESPONSIVE_REPLY, at=run.started_at, known_at=now)
        for beep in self._validated_beeps:
            if clean and run.started_at >= beep.ended_at + SPEECH_AFTER_BEEP_SECONDS and seconds >= 0.5:
                self._signal(_signals.KIND_SPEECH_AFTER_BEEP, at=run.started_at, known_at=now)
                break

    def _check_hello(self, *, now: float) -> None:
        run = self._hello_run
        if run is None or self._open_run is not None:
            return
        horizon = self._last_caller_media_end
        if horizon is None or horizon - run.ended_at < HELLO_YIELD_SECONDS:
            return
        self._hello_run = None
        if any(tone.started_at >= run.ended_at for tone in self._tones):
            return
        self._signal(_signals.KIND_HELLO_THEN_YIELD, at=run.started_at, known_at=now)

    # -- tones ---------------------------------------------------------------

    def _on_tone(self, tone: _signals.Tone, spans_gap: bool, *, now: float) -> None:
        if tone.started_at < (self._armed_at or 0.0):
            return
        history = self._tracker.history
        echo = any(frame.echo for frame in history if tone.started_at <= frame.at < tone.ended_at)
        reason = _signals.tone_rejection(
            tone, previous_tones=self._tones, echo=echo, spans_gap=spans_gap
        )
        if len(self._tones) >= MAX_TONES:
            del self._tones[0]
        self._tones.append(tone)
        if not reason and self._states_standing:
            reason = _signals.REJECT_STATE
        if reason:
            self._tone_signal(_signals.KIND_TONE_REJECTED, tone, detail=reason, known_at=now)
            return
        observer = self._observer
        proof = observer.proof_known_at if observer is not None else None
        if proof is not None and proof <= now and observer is not None and observer.proof_kind == _signals.KIND_MAILBOX_PHRASE:
            # The phrase already proved the mailbox: a clean tone after it is
            # the recording cue whatever came before it (Codex stamp #5).
            self._candidates.append(_Candidate(tone=tone, span=None, speech_shape=False, phrase_cue=True))
            return
        verdict, span = _signals.greeting_shape(history, tone_started_at=tone.started_at)
        if verdict != _signals.GREETING_SPEECH:
            self._tone_signal(_signals.KIND_TONE_AFTER_NON_SPEECH, tone, detail=verdict, known_at=now)
            return
        self._candidates.append(_Candidate(tone=tone, span=span, speech_shape=True))

    def _tone_signal(self, kind: str, tone: _signals.Tone, *, detail: str = "", known_at: float) -> None:
        self._signal(
            kind,
            at=tone.ended_at,
            known_at=known_at,
            detail=detail,
            frequency_hz=int(round(tone.frequency_hz)),
            duration_ms=int(round(tone.seconds * 1000)),
        )

    def _resolve_candidates(self, *, now: float) -> None:
        horizon = self._last_caller_media_end
        if horizon is None:
            return
        remaining: list[_Candidate] = []
        for candidate in self._candidates:
            tone = candidate.tone
            window_end = tone.ended_at + _signals.CADENCE_LOOKAHEAD_SECONDS
            if horizon < window_end:
                remaining.append(candidate)
                continue
            withdrawn = ""
            for later in self._tones:
                if tone.ended_at < later.started_at <= window_end and _signals.same_frequency(
                    later.frequency_hz, tone.frequency_hz
                ):
                    withdrawn = _signals.REJECT_CADENCE
                    break
            open_tone = self._tracker.open_tone
            if (
                not withdrawn
                and open_tone is not None
                and tone.ended_at < open_tone[0] <= window_end
                and _signals.same_frequency(open_tone[1], tone.frequency_hz)
            ):
                # A repeat still sounding when the look-ahead closes is a
                # repeat all the same (Codex code review #6).
                withdrawn = _signals.REJECT_CADENCE
            speech_after = [
                run
                for run in self._runs + ([self._open_run] if self._open_run else [])
                if not run.echo and run.started_at < window_end and run.ended_at > tone.ended_at + 0.1
            ]
            if not withdrawn and speech_after:
                withdrawn = "speech_after"
                # Somebody spoke into what would have been a speculative
                # playout: that is the person the audit exists to count, so
                # it is recorded as such and not only as a withdrawal (Grok
                # code review).
                # Only a DISTINCT run, starting half a second after the beep,
                # clean and long enough -- the same rule as after a validated
                # beep (Codex stamp #3); a run that merely continues through
                # the tone is not a person.
                distinct = [
                    run
                    for run in speech_after
                    if run.started_at >= tone.ended_at + SPEECH_AFTER_BEEP_SECONDS
                    and not run.gap
                    and run.ended_at - run.started_at >= SPEECH_AFTER_BEEP_SECONDS
                ]
                if distinct:
                    first = min(distinct, key=lambda run: run.started_at)
                    self._signal(_signals.KIND_SPEECH_AFTER_BEEP, at=first.started_at, known_at=now)
            if not withdrawn and self._gap_between(tone.ended_at, window_end):
                withdrawn = _signals.REJECT_GAP
            if withdrawn:
                self._observer_cue(tone.ended_at, now, _observer.CUE_WITHDRAWN, withdrawn)
                self._tone_signal(_signals.KIND_TONE_REJECTED, tone, detail=withdrawn, known_at=now)
                continue
            if candidate.phrase_cue:
                # The cue for a mailbox the phrase already proved: recorded
                # for leave_message, and not a second proof.
                self._observer_cue(tone.ended_at, now, _observer.CUE_VALIDATED, "")
                self._validated_beeps.append(tone)
                continue
            unverified = self._greeting_unverified(candidate, now=now)
            if unverified:
                self._observer_cue(tone.ended_at, now, _observer.CUE_UNVERIFIED, unverified)
                self._tone_signal(
                    _signals.KIND_TONE_AFTER_UNVERIFIED_SPEECH, tone, detail=unverified, known_at=now
                )
                continue
            self._observer_cue(tone.ended_at, now, _observer.CUE_VALIDATED, "")
            self._validated_beeps.append(tone)
            self._tone_signal(_signals.KIND_RECORD_BEEP, tone, known_at=now)
        self._candidates = remaining

    def _gap_between(self, start: float, end: float) -> bool:
        return any(
            frame.gap and start <= frame.at < end for frame in self._tracker.history
        )

    def _greeting_unverified(self, candidate: _Candidate, *, now: float) -> str:
        """Why the speech before this beep is not verified, or ``""``."""

        if self._states_standing:
            return _signals.REJECT_STATE
        span = candidate.span
        if span is None:
            return "no_span"
        start, end = span
        finals: set = set()
        for run in self._runs:
            if run.started_at < candidate.tone.ended_at and run.ended_at > start and not run.echo:
                finals |= run.finals
        words = sum(self._final_words.get(final, 0) for final in finals)
        if words < GREETING_WORDS_MIN:
            return "no_words"
        if not self._classifier_available:
            return "no_classifier"
        # Any non-human reading from the start of the greeting up to the moment
        # of validation vetoes it: a verdict that lands during the look-ahead
        # is still known before the beep could count (Codex code review #7).
        # A tune BEFORE the greeting does not -- a caller tune followed by a
        # mailbox is the no-answer divert.
        veto_from = candidate.tone.started_at - NON_HUMAN_VETO_LOOKBACK_SECONDS
        if any(veto_from <= at <= now for at in self._non_human_readings):
            return "non_human_audio"
        if not any(start <= at <= candidate.tone.ended_at + 1.0 for at in self._speech_like_at):
            return "not_speech_like"
        return ""

    def _observer_cue(self, at: float, known_at: float, outcome: str, reason: str) -> None:
        if self._observer is not None:
            self._observer.note_cue(_observer.CueCandidate(at=at, known_at=known_at, outcome=outcome, reason=reason))

    # -- transcripts ------------------------------------------------------------

    def _process_texts(self, *, now: float, flush: bool = False) -> None:
        """Bind finals once the audio they came from has been processed.

        A final is held until caller frames RECEIVED after its ingestion (plus
        the quiet that closes a run) have been judged: otherwise the run its
        words belong to is still open, and the words bind to nothing.
        """

        while self._texts:
            at, text = self._texts[0]
            horizon = self._caller_received_horizon
            if not flush and (horizon is None or horizon < at + RUN_END_QUIET_SECONDS + 0.05):
                break
            self._texts.popleft()
            self._on_final(text, at=at, now=now)

    def _recent_agent_texts(self, at: float) -> list[str]:
        return [text for when, text in self._agent_texts if at - when <= AGENT_TEXT_SECONDS]

    def _on_final(self, text: str, *, at: float, now: float) -> None:
        previous = self._last_final_at
        self._last_final_at = at
        armed_at = self._armed_at
        if armed_at is None or at < armed_at:
            return
        released = self._input_released_at
        if released is None:
            # Nothing the engine could have heard yet; the binding below also
            # requires a run to START after the release and END before the
            # final, which is what keeps early-media words out.
            return
        agent_texts = self._recent_agent_texts(at)
        words = _signals.echo_free_word_count(text, agent_texts)
        category, strong = _signals.classify_phrase(text, language=self.language)
        candidates = self._runs + ([self._open_run] if self._open_run is not None else [])
        # Judged as of the words' arrival, never as of this processing moment:
        # a run that had started by ``at`` either ended before it or was still
        # going, and both are the far end's audio behind these words. Whether
        # the run has CLOSED since is only how far processing got (a held
        # final is bound after the caller horizon passes it, and a starved
        # watch processes a whole greeting at once), so it must not decide
        # the binding. Likewise a run is new since the previous final exactly
        # when it has audio after that final.
        bound = [
            run
            for run in candidates
            if run.started_at >= released
            and (previous is None or run.received_end > previous)
            and run.received_start <= at
            and not run.echo
        ]
        final_id = len(self._final_words) + 1
        self._final_words[final_id] = words
        for run in bound:
            run.finals.add(final_id)
            if category in {_signals.PHRASE_IVR, _signals.PHRASE_SCREENER, _signals.PHRASE_MAILBOX}:
                run.automation_phrase = True
        if category and _signals.phrase_is_echo(text, agent_texts, language=self.language):
            self._signal(_signals.KIND_PHRASE_ECHO, at=at, known_at=now, detail=category)
            return
        if category in {_signals.PHRASE_IVR, _signals.PHRASE_SCREENER}:
            self._automation_final_at.append(at)
            del self._automation_final_at[:-16]
            for pending in self._pending_phrases:
                if at - pending.at <= PHRASE_FORWARD_VETO_SECONDS:
                    pending.vetoed = True
        if category == _signals.PHRASE_MAILBOX:
            if not strong:
                self._signal(_signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=at, known_at=now)
            elif not _signals.mailbox_instruction_is_verified(text, language=self.language):
                # An instruction only a CANDIDATE list recognises: recorded, so
                # the shadow data can verify the list, and never proof.
                self._signal(_signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=at, known_at=now, detail="candidate_language")
            elif any(0 <= at - earlier <= PHRASE_FORWARD_VETO_SECONDS for earlier in self._automation_final_at):
                # A menu just BEFORE the instruction vetoes it as surely as one
                # just after (Codex code review #8).
                self._signal(_signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=at, known_at=now, detail="automation_veto")
            elif not bound or not self._agent_feed_fresh(at):
                # Words with no far-end audio behind them, or heard while our
                # own feed could not be followed, may be our own echo or may
                # be anybody's: they cannot be proof (Codex code review #3).
                self._signal(_signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=at, known_at=now, detail="unattributed")
            else:
                self._pending_phrases.append(_PendingPhrase(at=at, due=at + PHRASE_FORWARD_VETO_SECONDS))
            return
        state_kind = {
            _signals.PHRASE_IVR: _signals.KIND_IVR_PHRASE,
            _signals.PHRASE_SCREENER: _signals.KIND_SCREENER_PHRASE,
            _signals.PHRASE_UNAVAILABLE: _signals.KIND_UNAVAILABLE_PHRASE,
        }.get(category)
        if state_kind:
            self._states_standing.add(state_kind)
            self._signal(state_kind, at=at, known_at=now)
            return
        if words >= 1 and not category:
            for run in bound:
                if run.reply_to_agent and self._observer is not None and self._has_provisional_before(run.started_at):
                    self._signal(_signals.KIND_CONFIRMED_REPLY, at=run.started_at, known_at=now)
                    break

    def _agent_feed_fresh(self, at: float) -> bool:
        last = self._agent_last_received
        return last is not None and at - last <= AGENT_STALE_SECONDS

    def _has_provisional_before(self, at: float) -> bool:
        observer = self._observer
        if observer is None:
            return False
        return any(signal.at < at for signal in observer.provisional_humans)

    def _resolve_phrases(self, *, now: float) -> None:
        remaining: list[_PendingPhrase] = []
        horizon = self._caller_received_horizon
        for pending in self._pending_phrases:
            # The window is closed by OBSERVED caller media, not by the wall
            # clock: a menu the feed never delivered cannot veto anything, so
            # a window nobody heard is not a window that passed (Codex stamp #2).
            if now < pending.due or horizon is None or horizon < pending.due:
                remaining.append(pending)
                continue
            if any(pending.at <= at <= pending.due for at in self._caller_gap_at):
                self._signal(_signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=pending.at, known_at=now, detail="gap")
                continue
            if pending.vetoed:
                self._signal(
                    _signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=pending.at, known_at=now, detail="automation_veto"
                )
                continue
            # A mailbox phrase is an explicit transition out of an automation
            # state: a menu that hands over to a mailbox says so.
            self._states_standing.clear()
            self._signal(_signals.KIND_MAILBOX_PHRASE, at=pending.at, known_at=now)
        self._pending_phrases = remaining

    # -- signals and the record -----------------------------------------------

    def _signal(
        self,
        kind: str,
        *,
        at: float,
        known_at: float,
        detail: str = "",
        frequency_hz: int = 0,
        duration_ms: int = 0,
        once: bool = False,
    ) -> None:
        observer = self._observer
        if observer is None:
            return
        if once:
            if kind in self._emitted_once:
                return
            self._emitted_once.add(kind)
        signal = _observer.AnswerSignal(
            kind=kind, at=at, known_at=known_at, detail=detail,
            frequency_hz=frequency_hz, duration_ms=duration_ms,
        )
        before = observer.classification()
        observer.observe(signal)
        after = observer.classification()
        try:
            self.emit(
                "answer_detection_voicemail_signal",
                kind=kind,
                strength=signal.strength,
                detail=detail,
                after_ms=int(round((at - observer.armed_at) * 1000)),
            )
            if after != before:
                self.emit("answer_detection_voicemail_decision", classification=after)
        except Exception:  # noqa: BLE001 - reporting cannot fail the watch
            pass

    def freeze(self, *, ended_by: str, now: float) -> dict[str, Any] | None:
        """The ``voicemail_detection`` section for transcript.json, once.

        Drains what is queued within the frame budget first; what remains is
        counted as ``drain_truncated`` rather than processed past the budget.
        ``None`` for ``off`` and for a call that never answered.
        """

        if self._frozen is not None:
            return self._frozen
        if not self.enabled or self._observer is None or self._armed_at is None:
            return None
        # Final drain: agent before caller, as always, then close what is open.
        self.process(now=now, drain=True)
        truncated = len(self._caller) + len(self._agent)
        if self._open_run is not None:
            self._close_run(self._open_run, now=now)
        self._process_texts(now=now, flush=True)
        self._resolve_phrases(now=now)
        # A phrase whose forward window was still open when the call ended
        # was never proof: the menu that would have vetoed it had no time to
        # speak.  Recorded as such rather than promoted.
        for pending in self._pending_phrases:
            self._signal(
                _signals.KIND_MAILBOX_PHRASE_AMBIGUOUS, at=pending.at, known_at=now, detail="window_open_at_end"
            )
        self._pending_phrases = []
        completed = self._tracker.flush()
        if completed is not None:
            self._on_tone(*completed, now=now)
        observer = self._observer
        snapshot = observer.snapshot()
        reasons: list[str] = []
        if self._caller_frames_after_arm == 0:
            status, reasons = STATUS_UNAVAILABLE, ["no_caller_audio"]
        else:
            if snapshot["signals_dropped"] or self._texts_dropped:
                reasons.append("signal_limit_reached")
            if self._caller_gaps or self._caller_overflow:
                reasons.append("caller_feed_gap")
            if self._agent_frames == 0:
                reasons.append("agent_feed_missing")
            elif self._agent_stale_frames:
                reasons.append("agent_feed_stale")
            if truncated:
                reasons.append("drain_truncated")
            status = STATUS_PARTIAL if reasons else STATUS_COMPLETE

        def offset(value: float | None) -> int | None:
            return None if value is None else int(round((value - self._armed_at) * 1000))

        hang_up = snapshot["hang_up"]
        leave = snapshot["leave_message"]
        fast = self._disposition == ANSWERED and self._wait_ms is not None and self._wait_ms <= FAST_ANSWER_MS
        opener_after = offset(self._opener_started_at)
        section = {
            "version": SECTION_VERSION,
            "detector_version": _signals.DETECTOR_VERSION,
            "status": status,
            "reason": reasons[0] if reasons else "",
            "reasons": reasons,
            "configured_mode": self.configured_mode,
            "effective_mode": EFFECTIVE_MODE,
            "language": _signals.primary_language(self.language),
            "phrase_cues_available": _signals.phrase_cues_available(self.language),
            "answer_disposition": self._disposition,
            "answer_wait_ms": self._wait_ms,
            "eligible": snapshot["eligible"],
            "classification": snapshot["classification"],
            "proof_kind": snapshot["proof_kind"],
            "proof_after_ms": snapshot["proof_after_ms"],
            "hang_up": hang_up,
            "leave_message": leave,
            "ended_by": str(ended_by or ""),
            "observed_until_ms": offset(self._last_caller_media_end),
            "observation_stopped_by": self._stopped_by,
            "opener_started_after_ms": opener_after,
            "would_have_held_opener": bool(fast and opener_after is not None and opener_after < 2000),
            "classifier_available": self._classifier_available,
            "feed": {
                "caller_frames": self._caller_frames_after_arm,
                "caller_gaps": self._caller_gaps,
                "caller_gap_ms": int(round(self._caller_gap_seconds * 1000)),
                "caller_overflow": self._caller_overflow,
                "agent_frames": self._agent_frames,
                "agent_overflow": self._agent_overflow,
                "agent_feed": (
                    AGENT_FEED_MISSING
                    if self._agent_frames == 0
                    else AGENT_FEED_STALE if self._agent_stale_frames else AGENT_FEED_OK
                ),
                "agent_stale_frames": self._agent_stale_frames,
                "drain_truncated": truncated,
            },
            "counts": {
                "signals": snapshot["signal_count"],
                "retained": len(snapshot["signals"]),
                "dropped": snapshot["signals_dropped"],
                "late_proofs": snapshot["late_proofs"],
                "texts_dropped": self._texts_dropped,
                "cues_dropped": snapshot["cues_dropped"],
            },
            "signals": snapshot["signals"],
            "cues": snapshot["cues"],
        }
        self._frozen = section
        try:
            self.emit(
                "answer_detection_voicemail_summary",
                classification=section["classification"],
                proof_kind=section["proof_kind"],
                hang_up=hang_up["outcome"],
                leave_message=leave["outcome"],
                cut_off_human=bool(hang_up["cut_off_human"] or leave["cut_off_human"]),
                status=status,
            )
        except Exception:  # noqa: BLE001 - reporting cannot fail the freeze
            pass
        return section


__all__ = [
    "PROCESS_INTERVAL_SECONDS",
    "DEFAULT_MODE",
    "EFFECTIVE_MODE",
    "MODES",
    "SECTION_VERSION",
    "VoicemailWatch",
    "resolved_setting",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Continuous caller-audio flow and VAD edges, for the whole call.

Why this exists, concretely.  On one production call a caller spoke 6.5 s of
speech-level audio and the worker produced no transcript, no turn and no
``user_transcription_timeout``.  Every gate that could have silenced them was
excluded from the artifacts, and the question that would have settled it --
was caller audio still reaching this session's recognition input at second 22?
-- had no writer.  Two shipped signals look like they answer it and cannot:

* :class:`~apps.voice.production.CallerAudioBindWitness` counts frames at the
  right seam but reports **at most once**, at the first caller frame.  It says
  audio once started; it cannot say audio was still flowing later.
* :class:`~apps.voice.production.CallerSpeechWitness` latches on the FIRST
  start-of-speech edge.  It says the caller was heard once; it cannot say
  whether a later stretch of speech produced VAD at all.

Both are first-sighting witnesses, and a first sighting reads as a silent line
for every moment after it.  This module adds the two continuous records, and
deliberately does NOT change either witness: their once-per-call semantics are
load-bearing elsewhere: the silence rule reads "was the caller EVER heard",
and a witness that re-fired would change what that rule decides.

Nothing here ends, alters or bills a call.  It is diagnostic only: the
absence of recognised speech is not evidence of the absence of a person, so
this module is the evidence that makes such a call countable, never a rule
that acts on one.

Loop-affine by contract, like the gate it sits beside: every method is called
from the single asyncio loop running the media session.  No locks, no threads,
no I/O.  Clocks are injected so the behaviour is testable without sleeping.

Inert by contract too, and that is load-bearing rather than tidy.  One of these
recorders is fed from ``Agent.stt_node`` -- the caller's recognition path -- so
an exception escaping it would break the very media this exists to measure, and
would do it on exactly the silent-call shape it was built to explain.  Every
emission is therefore guarded here, in the module, rather than at each call
site: the STT seam, the classification tick and the teardown summary all feed
these, and a guard that lives at the call sites is one refactor away from
having a hole.  Nothing else in these classes performs I/O or arithmetic that
can raise on a value the media path can produce.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

__all__ = [
    "CALLER_AUDIO_FLOW_WINDOW_SECONDS",
    "CALLER_TRACK_ENERGY_FLOOR",
    "ENERGY_WITHOUT_INPUT_ALARM_SECONDS",
    "ENERGY_SAMPLE_MAX_STEP_SECONDS",
    "MAX_RECORDED_FLOW_GAPS",
    "MAX_RECORDED_VAD_EDGES",
    "CallerAudioFlowLog",
    "CallerVadEdgeLog",
]

# One flow verdict per window.  1.0 s at 10 ms frames is 100 frames, so a
# window is far wider than any plausible scheduling jitter and still narrow
# enough to localise a gap to the second.
CALLER_AUDIO_FLOW_WINDOW_SECONDS = 1.0

# Every growing collection is bounded and says so.  A gap is one row per
# CONTIGUOUS run of starved windows, so a long outage costs one row, not one
# per second; the caps are a malfunction ceiling, not an expected limit.
MAX_RECORDED_FLOW_GAPS = 32
MAX_RECORDED_VAD_EDGES = 128

#: Mean-absolute amplitude (16-bit mono) above which the caller TRACK is
#: treated as carrying something.  Calibrated on the call described above,
#: where the caller's speech measured ~2400-3000 RMS and the quiet stretches
#: 0.1-71.  It is deliberately far below speech and far above that floor.
#:
#: Known blind spot, recorded here because a detector must name them: this is
#: a fixed floor, and one call in the calibration corpus had a NOISE floor of
#: 3703 (a noisy line).  On such a call this threshold reads noise as
#: energy.  That is why the snapshot reports ``max_track_amplitude_in_gap``
#: and the measured seconds alongside the boolean -- a wrong floor is then
#: visible in the data instead of silently deciding the verdict.
CALLER_TRACK_ENERGY_FLOOR = 200.0

#: How much TOTAL track energy during a session-input gap makes the call a
#: countable candidate.  It is a sum over the gap, not one unbroken run: a
#: caller who says "hello?  ...  hello?" into a call that is not listening is
#: the case this counts, and requiring one contiguous run would miss them.
#: Silence still breaks the run for ACCUMULATION, so the quiet between those
#: two words is not credited as energy -- only the words are.  Diagnostic only:
#: nothing ends, disposes or bills on this.  Two seconds is longer than any
#: single word and shorter than the 6.5 s stretch the real call lost.
ENERGY_WITHOUT_INPUT_ALARM_SECONDS = 2.0

#: Energy is sampled per media frame, so consecutive samples are ~10-20 ms
#: apart.  A larger step means the feed itself paused, and crediting that whole
#: span as "energy" would manufacture the very signal this measures.
ENERGY_SAMPLE_MAX_STEP_SECONDS = 0.1


class CallerAudioFlowLog:
    """Whether caller audio kept reaching this session, for the whole call.

    Fed from the same seam as the bind witness -- ``Agent.stt_node``, where
    caller frames enter the worker -- so "flowing" means reaching THIS
    session's recognition input, not reaching the room.  That distinction is
    the whole point: the evidence recorder writes ``caller.wav`` from a
    separate track subscription, so an intact wav proves audio reached the
    worker and says nothing about whether recognition saw it.

    A gap is opened when a window elapses with zero frames after flow had
    started, and closed when a frame arrives again.  Flow that never starts is
    not a gap: that is the binding fault the bind witness already reports, and
    reporting it twice under two names would make one failure look like two.
    """

    def __init__(
        self,
        *,
        window_seconds: float = CALLER_AUDIO_FLOW_WINDOW_SECONDS,
        max_gaps: int = MAX_RECORDED_FLOW_GAPS,
        energy_floor: float = CALLER_TRACK_ENERGY_FLOOR,
        alarm_seconds: float = ENERGY_WITHOUT_INPUT_ALARM_SECONDS,
        emit: Callable[..., None] = lambda *_args, **_details: None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window = max(0.05, float(window_seconds))
        self._alarm_seconds = max(0.0, float(alarm_seconds))
        self._max_gaps = max(1, int(max_gaps))
        self._emit = emit
        self._clock = clock
        self._frames_total = 0
        self._frames_this_window = 0
        self._window_opened_at: float | None = None
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._gap_started_at: float | None = None
        self._gaps: list[dict[str, float]] = []
        self._gaps_dropped = 0
        # The alarm half: caller TRACK energy measured while the session's own
        # input was starved.  The two come from different subscriptions on
        # purpose -- see :meth:`note_track_energy`.
        self._energy_floor = float(energy_floor)
        self._energy_seconds_in_gaps = 0.0
        self._energy_seconds_this_gap = 0.0
        self._max_amplitude_in_gap = 0.0
        self._last_energy_at: float | None = None

    # ------------------------------------------------------------------ input
    def note_caller_frame(self, at: float | None = None) -> None:
        """One caller audio frame reached this session's recognition input.

        The elapsed windows are settled BEFORE this frame is credited, and that
        ordering is the correctness of the whole class.  Crediting first lets a
        resumed frame land in a window that is only still "current" because the
        tick was late, which retroactively marks that window healthy: a real
        starvation then reports no gap at all, and the arriving frame starts a
        spurious one.  Measured, before this was fixed: a 2.9 s outage followed
        by a frame 0.1 s before a late tick produced zero gaps -- the module
        would have hidden exactly the call it was built to explain.
        """

        now = self._now(at)
        # Settle what already elapsed, then credit this frame to the window it
        # actually arrived in.
        self._advance_windows(now)
        self._frames_total += 1
        self._frames_this_window += 1
        if self._first_frame_at is None:
            self._first_frame_at = now
            self._window_opened_at = now
        if self._gap_started_at is not None:
            self._close_gap(now)
        self._last_frame_at = now

    def poll(self, at: float | None = None) -> None:
        """Advance the window clock.

        Called on the worker's existing cadence.  A starved window is only
        observable from a tick, because the thing that would have reported it
        is the frame that never arrived -- so a log driven purely by frames
        can never see its own absence.  ``note_caller_frame`` advances the same
        clock, so a late tick delays a gap's discovery but can no longer erase
        it.
        """

        self._advance_windows(self._now(at))

    def _advance_windows(self, now: float) -> None:
        """Close out every window that has fully elapsed by ``now``."""

        if self._window_opened_at is None:
            # Flow has not started; the bind witness owns that case.
            return
        while now - self._window_opened_at >= self._window:
            starved = self._frames_this_window == 0
            self._window_opened_at += self._window
            self._frames_this_window = 0
            if starved and self._gap_started_at is None:
                self._open_gap(self._last_frame_at if self._last_frame_at is not None else now)

    @property
    def wants_energy(self) -> bool:
        """Should the caller's track be measured right now?

        Only while the session's input is starved.  The caller for this is the
        evidence sidecar's media drain, where measuring every frame of every
        call would put two allocations per frame on the media loop for the life
        of the call -- the same cost ``AnswerDetection.note_frame`` gates
        against.  A healthy call pays one boolean read per frame and nothing
        else, because there is nothing to attribute unless input has stopped.
        """

        return self._gap_started_at is not None

    def note_track_energy(self, amplitude: float, at: float | None = None) -> None:
        """One amplitude measurement of the caller TRACK.

        Fed from the evidence sidecar's own subscription -- a DIFFERENT
        subscription from the session's recognition input this class otherwise
        counts.  That separation is the entire value of the pair: on
        that call the sidecar's copy of the audio was perfect
        (``caller.wav`` has the caller's speech) while the question of what
        recognition received had no answer at all.  Energy here plus starvation
        there is the one combination that means the caller was talking into a
        call that was not listening.

        Ignored unless a gap is open: outside one there is nothing to attribute
        it to, and the agent's own voice can reach the caller track on a
        speakerphone, so a measurement taken while everything is healthy would
        be echo as often as speech.

        A gap only opens after a full starved window, so the first window of
        any outage is never attributed and the reported seconds UNDERSTATE the
        real figure by up to ``window_seconds``.  That is deliberate: for a
        candidate count the safe error is to miss a marginal call, never to
        manufacture one.
        """

        if self._gap_started_at is None:
            return
        try:
            value = float(amplitude)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            # NaN would pass every comparison below; infinity would pass them
            # all and then land in the payload, where json.dumps writes a bare
            # Infinity that is not valid JSON for whatever reads these rows.
            return
        now = self._now(at)
        if value > self._max_amplitude_in_gap:
            self._max_amplitude_in_gap = value
        if value < self._energy_floor:
            # Below the floor breaks the run: silence between two loud frames
            # is not energy, and crediting it would smear one word across a gap.
            self._last_energy_at = None
            return
        previous = self._last_energy_at
        self._last_energy_at = now
        if previous is None:
            return
        step = now - previous
        if 0.0 < step <= ENERGY_SAMPLE_MAX_STEP_SECONDS:
            self._energy_seconds_this_gap += step

    # ----------------------------------------------------------------- output
    @property
    def frames_total(self) -> int:
        return self._frames_total

    @property
    def flowing(self) -> bool:
        """Is caller audio reaching recognition right now?"""

        return self._first_frame_at is not None and self._gap_started_at is None

    def snapshot(self, at: float | None = None) -> dict[str, Any]:
        """The bounded, call-scoped record.  Never implies a complete list."""

        now = self._now(at)
        gaps = [dict(gap) for gap in self._gaps]
        open_gap: dict[str, float] | None = None
        if self._gap_started_at is not None:
            open_gap = {
                "started_at": round(self._gap_started_at, 3),
                "seconds": round(max(0.0, now - self._gap_started_at), 3),
                "energy_seconds": round(self._energy_seconds_this_gap, 3),
                "max_amplitude": round(self._max_amplitude_in_gap, 1),
            }
        # The alarm-shaped outcome: the caller's TRACK carried sustained energy
        # while the session's own input received nothing. Countable from the
        # per-call summary, so the next occurrence does not need a log read.
        # Still diagnostic: nothing ends, disposes or bills on this.
        energy_without_input = round(
            self._energy_seconds_in_gaps + self._energy_seconds_this_gap, 3
        )
        # The counters INCLUDE a gap that is still open.  A call whose caller
        # audio stops and never resumes is one of the two shapes this exists to
        # separate, and it has no closed gap at all: counting only closed ones
        # reported ``gap_count: 0`` and ``longest_gap_seconds: 0.0`` in the same
        # payload as a large ``open_gap``, so anything counting candidates
        # across calls read that call as gap-free -- the exact case it was most
        # important to count.
        closed_seconds = [float(gap["seconds"]) for gap in gaps]
        if open_gap is not None:
            closed_seconds.append(float(open_gap["seconds"]))
        return {
            "frames_total": self._frames_total,
            "flow_started": self._first_frame_at is not None,
            "gaps": gaps,
            "gap_count": len(gaps) + self._gaps_dropped + (1 if open_gap is not None else 0),
            "gaps_dropped": self._gaps_dropped,
            "has_more": self._gaps_dropped > 0,
            "open_gap": open_gap,
            "longest_gap_seconds": round(max(closed_seconds), 3) if closed_seconds else 0.0,
            "energy_without_session_input_seconds": energy_without_input,
            "energy_without_session_input": energy_without_input >= self._alarm_seconds,
            "energy_floor": self._energy_floor,
            "alarm_seconds": self._alarm_seconds,
        }

    # ---------------------------------------------------------------- internal
    def _open_gap(self, started_at: float) -> None:
        self._gap_started_at = started_at
        self._energy_seconds_this_gap = 0.0
        self._max_amplitude_in_gap = 0.0
        self._last_energy_at = None
        if self._gaps_dropped == 0:
            self._safe_emit(
                "caller_audio_flow_gap_started",
                frames_total=self._frames_total,
                started_at=round(started_at, 3),
            )

    def _close_gap(self, now: float) -> None:
        started_at = self._gap_started_at
        self._gap_started_at = None
        if started_at is None:
            return
        seconds = round(max(0.0, now - started_at), 3)
        energy_seconds = round(self._energy_seconds_this_gap, 3)
        max_amplitude = round(self._max_amplitude_in_gap, 1)
        self._energy_seconds_in_gaps += self._energy_seconds_this_gap
        self._energy_seconds_this_gap = 0.0
        self._max_amplitude_in_gap = 0.0
        self._last_energy_at = None
        if len(self._gaps) < self._max_gaps:
            self._gaps.append(
                {
                    "started_at": round(started_at, 3),
                    "seconds": seconds,
                    "energy_seconds": energy_seconds,
                    "max_amplitude": max_amplitude,
                }
            )
            self._safe_emit(
                "caller_audio_flow_gap_ended",
                seconds=seconds,
                started_at=round(started_at, 3),
                frames_total=self._frames_total,
                energy_seconds=energy_seconds,
                max_amplitude=max_amplitude,
            )
            return
        # Past the cap the gap is counted and NOT emitted, for the same reason
        # the edge log stops: the cap must bound the sink as well as the list,
        # or a flapping flow fault turns a truncated record into unbounded log
        # volume. ``gap_count``/``has_more`` still carry the total.
        self._gaps_dropped += 1
        if self._gaps_dropped == 1:
            self._safe_emit("caller_audio_flow_gaps_truncated", kept=self._max_gaps)

    def _safe_emit(self, event: str, **details: Any) -> None:
        """Emit, never raising into the caller.

        This class is fed from ``Agent.stt_node``.  A sink that raises there
        would propagate into the caller's recognition path and break the media
        it is measuring, so a lost diagnostic row is always the cheaper loss.
        """

        try:
            self._emit(event, **details)
        except Exception:  # noqa: BLE001 - a diagnostic cannot break media
            pass

    def _now(self, at: float | None) -> float:
        if at is not None:
            return float(at)
        if self._clock is not None:
            return float(self._clock())
        raise ValueError("CallerAudioFlowLog needs an explicit time or a clock")


class CallerVadEdgeLog:
    """EVERY caller VAD edge, not only the first.

    The shipped ``CallerSpeechWitness`` latches on the first start-of-speech
    edge by construction, so on a call where the caller speaks at 10 s and
    again at 22 s it emits one line for the first and nothing for the second.
    Asking it whether the second stretch produced VAD is asking a question it
    was built not to answer.  This records each edge instead, with the duration
    of the run that just closed.

    Edges arrive as the session's ``user_state_changed`` transitions.  Only the
    speaking/not-speaking distinction is read, so a runtime that grows a third
    state does not silently start recording it as speech.
    """

    def __init__(
        self,
        *,
        max_edges: int = MAX_RECORDED_VAD_EDGES,
        emit: Callable[..., None] = lambda *_args, **_details: None,
    ) -> None:
        self._max_edges = max(1, int(max_edges))
        self._emit = emit
        self._edges: list[dict[str, Any]] = []
        self._edges_dropped = 0
        self._speaking = False
        self._speaking_since: float | None = None
        self._runs = 0
        self._speech_seconds = 0.0

    def note_user_state(self, new_state: Any, *, at: float) -> None:
        """One ``user_state_changed`` transition, at ``at`` seconds."""

        speaking = str(new_state or "") == "speaking"
        if speaking == self._speaking:
            # Level-tracked: only a CHANGE is an edge, so a runtime that
            # re-announces the same state does not manufacture runs.
            return
        self._speaking = speaking
        moment = float(at)
        if speaking:
            self._speaking_since = moment
            self._runs += 1
            self._record("start_of_speech", moment, seconds=None)
        else:
            started = self._speaking_since
            self._speaking_since = None
            seconds = round(max(0.0, moment - started), 3) if started is not None else None
            if seconds is not None:
                self._speech_seconds += seconds
            self._record("end_of_speech", moment, seconds=seconds)

    @property
    def runs(self) -> int:
        return self._runs

    def snapshot(self, at: float | None = None) -> dict[str, Any]:
        """The bounded, call-scoped record.

        ``at`` measures a run that is still OPEN.  A caller talking when the
        call ends has no end-of-speech edge, so without it that run is missing
        from ``speech_seconds`` entirely -- and a caller still speaking at
        teardown is precisely the shape worth keeping.  It is reported
        separately rather than folded in, so ``speech_seconds`` stays the sum
        of runs that actually closed.
        """

        open_run_seconds: float | None = None
        if self._speaking and self._speaking_since is not None and at is not None:
            open_run_seconds = round(max(0.0, float(at) - self._speaking_since), 3)
        return {
            "edges": [dict(edge) for edge in self._edges],
            "edge_count": len(self._edges) + self._edges_dropped,
            "edges_dropped": self._edges_dropped,
            "has_more": self._edges_dropped > 0,
            "runs": self._runs,
            "speech_seconds": round(self._speech_seconds, 3),
            "speaking": self._speaking,
            "open_run_seconds": open_run_seconds,
        }

    def _record(self, kind: str, at: float, *, seconds: float | None) -> None:
        row: dict[str, Any] = {"kind": kind, "at": round(at, 3)}
        if seconds is not None:
            row["seconds"] = seconds
        if len(self._edges) < self._max_edges:
            self._edges.append(row)
            self._emit_edge(kind, row)
            return
        # Past the cap the row is counted and NOT emitted. The cap has to bound
        # the sink as well as the list, or a flapping VAD turns a truncated
        # record into unbounded log volume -- the count still reaches the
        # summary through ``edge_count``/``has_more``.
        self._edges_dropped += 1
        if self._edges_dropped == 1:
            self._emit_edge("truncated", {"kind": "truncated", "at": round(at, 3)})

    def _emit_edge(self, kind: str, row: dict[str, Any]) -> None:
        """Emit, never raising into the caller.

        Fed from the session's ``user_state_changed`` handler, which several
        other rules run from; a raise here would take them with it.
        """

        try:
            self._emit(
                "caller_vad_edge", kind=kind, **{k: v for k, v in row.items() if k != "kind"}
            )
        except Exception:  # noqa: BLE001 - a diagnostic cannot break a call
            pass

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""A bounded wait for the caller's final transcript, and the flush that ends it.

The defect this closes: 8 of 86 Azure turns in one measured day waited 6-60
seconds for a final transcript after the caller had stopped speaking. The
caller sat in dead air. Three facts in the pinned ``livekit-agents==1.7.1``
stack combine into that wait, and each is read from the source rather than
inferred:

* At VAD end-of-speech ``AudioRecognition._run_eou_detection`` returns
  immediately when no final transcript has arrived yet
  (``audio_recognition.py``: ``if self._stt and not self._audio_transcript``).
  The turn is abandoned with no timer; only a later final resurrects it.
* ``AgentSession(transcription_timeout=...)`` defaults to ``None``. When set,
  the SDK arms a timer at VAD end-of-speech (``_arm_transcription_timeout``),
  cancels it on VAD start-of-speech and on any final transcript with text
  (``_mark_turn_transcribed``), and on expiry emits ONE event,
  ``user_transcription_timeout``. It commits nothing. Observability, not
  recovery.
* The Azure plugin has no flush. Its ``_FlushSentinel`` only reports billing
  usage; the only thing that makes it emit the finals it is holding is
  closing its push stream, which its run loop does on reconnect --
  ``update_options()`` sets ``_reconnect_event``, the loop closes the
  stream (the plugin's own comment: "to flush finals for buffered audio"),
  waits for the finals, and rebuilds the recognizer. Deepgram sends
  ``{"type": "Finalize"}`` itself, which is why the tail is Azure-specific.

So the fix is two lines of wiring around primitives the runtime already
owns, and deliberately no second timer: the SDK's own arm/cancel is the
safety property. A caller who resumes mid-thought raises VAD start-of-speech,
which cancels the timer -- structurally, not by tuning. What the recovery
does when the timer fires is the residual risk, and it is stated on
:meth:`TranscriptionStallRecovery.on_timeout`.

What the flush cannot do, stated once here: the Azure plugin also drops an
EMPTY final before it becomes an event (``_on_recognized``: ``if not text:
return``), so if the flushed segment resolves as nothing the SDK is never
told, the turn stays open, and the SDK's timer does not re-arm until the
caller's next VAD cycle. That is the same dead air as before, now with a log
line -- and the second log line this module emits (:data:`OUTCOME_EVENT`)
says whether a final with text followed the flush at all, so the next real
occurrence settles from production whether the catastrophic stalls are
dropped empties. Recovering THAT case means speaking unprompted (the SDK's
own suggestion for the event is "prompt the user to repeat themselves"), a
synthetic agent turn with its own false-positive analysis, and it is a
separate change.

Everything here is provider-agnostic and imports no SDK: the event is the
SDK's, the flush is a method only the Azure adapter defines
(``azure_speech.build_stt``), and a provider without one is logged and left
alone -- Google chirp_2 also serves Sinhala and has no flush either, so a
stall there now leaves a signal instead of forensics.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

#: Seconds after the caller's VAD end-of-speech before a turn that has
#: produced no final transcript is force-finalized.
#:
#: The SDK measures this from the same anchor our ``stt.final`` telemetry
#: uses: the timer is armed with ``timeout - (silence_duration +
#: inference_duration)`` remaining, and ``stt.final`` is
#: ``last_final_transcript_time - last_speaking_time`` where
#: ``last_speaking_time = now - silence_duration - inference_duration``. The
#: two numbers below are therefore directly comparable with this one.
#:
#: Measured on 2026-08-31 across 86 Azure turns in 60 real calls:
#: the longest ordinary Azure final arrived 3.769 s after speech
#: end (p90 3.35 s); the shortest stall was 6.486 s. The distribution is
#: bimodal with an EMPTY region between those two -- sorted tail:
#: ... 2574, 2735, 3347, 3769 | 6486, 7357, 7499, 8935, 10790, 12626,
#: 46798, 59729 ms -- so 5.0 is the midpoint of an empty band, a chosen
#: constant rather than a tuned one, and every value in the band behaves
#: identically on the corpus. A test pins a margin of at least a second on
#: each side.
#:
#: Hold at 5.0; do not lower it. The two failure directions are not
#: symmetric: a missed marginal stall degrades gracefully (one unrecovered
#: turn, logged), while a false fire rebuilds the recognizer under a caller
#: whose final was still in flight. The 6.486 s edge comes from n=8 and may
#: fall with more data; 3.769 s is the side that must not be crossed.
TRANSCRIPTION_TIMEOUT_SECONDS = 5.0
NORMAL_AZURE_FINAL_MAX_SECONDS = 3.769
SHORTEST_OBSERVED_STALL_SECONDS = 6.486

#: How long after a forced flush its outcome is judged. Classification only
#: -- it changes no behaviour -- so it is sized above any measured flush
#: latency (close-to-recognized was well under a second against the pinned
#: plugin; the teardown flush in the 195 s exhibit landed within about a
#: second) and no larger, so the two log lines sit together.
FLUSH_OUTCOME_WINDOW_SECONDS = 2.0

#: The worker log events. One line per expiry, whatever the outcome, so a
#: recurrence is a count in the log rather than a WAV to correlate; and one
#: line per forced flush saying what followed it.
TIMEOUT_EVENT = "user_transcription_timeout"
OUTCOME_EVENT = "user_transcription_timeout_outcome"

#: What the handler did, named in the log line. Every branch is named,
#: including the ones that do nothing: a silent no-op is the failure mode
#: this event exists to end.
RECOVERY_FORCED = "azure_force_finalization"
RECOVERY_NONE = "provider_has_no_flush"
RECOVERY_TERMINAL = "admission_closed"
RECOVERY_UNRESOLVED = "stt_unresolved"
RECOVERY_FAILED = "flush_failed"

#: What followed a forced flush, judged at the end of the window.
OUTCOME_FINAL = "final_with_text"
OUTCOME_HELD = "final_held_by_transcript_gate"
OUTCOME_HELD_BEFORE = "final_held_before_flush"
OUTCOME_NONE = "no_final_within_window"
OUTCOME_UNOBSERVABLE = "unobservable"


def active_stt(session: Any) -> Any | None:
    """The transcriber hearing the caller right now, or ``None``.

    Resolved at fire time in the SDK's own order -- the agent's component
    when one is given, else the session's (``agent_activity.py``: ``stt``
    property) -- rather than captured at construction. Today nothing in
    this worker installs a per-agent transcriber: the follow-caller switch
    swaps TTS only and the shared recognizer stays on the session, so the
    agent lookup is defensive, there so a future ``agent.update_options(stt=
    ...)`` flushes the transcriber that is actually listening instead of the
    one the session started with. ``Agent.stt`` is the falsy ``NOT_GIVEN``
    sentinel when never set, which has no ``stream`` and is skipped for that
    reason. Duck typed on purpose so this module imports no SDK.
    """

    try:
        agent = session.current_agent
    except Exception:
        # ``current_agent`` raises before the session runs; the event that
        # reaches here cannot fire before it does, but a resolver that
        # raised inside an SDK event handler would be a worse outcome than
        # a logged ``stt_unresolved``.
        agent = None
    for candidate in (getattr(agent, "stt", None), getattr(session, "stt", None)):
        if candidate is not None and callable(getattr(candidate, "stream", None)):
            return candidate
    return None


def _call_later(delay: float, callback: Callable[[], None]) -> None:
    asyncio.get_running_loop().call_later(delay, callback)


class TranscriptionStallRecovery:
    """Answer the SDK's ``user_transcription_timeout`` by forcing Azure to finalize.

    Held by the production worker for the life of one call and subscribed to
    two session events: the timeout it answers, and ``user_input_transcribed``
    so it can say afterwards whether the flush produced a final with text.
    It owns no recovery timer -- see the module docstring for why -- and
    performs no I/O of its own: the flush is a method on the Azure adapter
    that flips an event the plugin's run loop is already waiting on. The one
    timer it does hold is the outcome window, which decides nothing.
    """

    def __init__(
        self,
        *,
        session: Any,
        emit: Callable[..., Any],
        provider: str,
        holds_final: Callable[[], bool] | None = None,
        admits: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.time,
        schedule: Callable[[float, Callable[[], None]], Any] = _call_later,
        outcome_window: float = FLUSH_OUTCOME_WINDOW_SECONDS,
    ) -> None:
        self._session = session
        self._emit = emit
        self._provider = str(provider or "")
        self._holds_final = holds_final
        self._admits = admits
        self._clock = clock
        self._schedule = schedule
        self._outcome_window = float(outcome_window)
        self._occurrences = 0
        self._forced = 0
        # Forced flushes whose window has not closed yet, oldest first.
        self._pending: list[dict[str, Any]] = []

    @property
    def occurrences(self) -> int:
        """How many times the SDK's timer expired on this call."""

        return self._occurrences

    @property
    def forced(self) -> int:
        """How many of those were answered with a forced Azure finalization."""

        return self._forced

    def on_timeout(self, event: Any) -> str:
        """Handle one expiry; returns the recovery label that was logged.

        The branches, in the order they are decided:

        * ``admission_closed`` -- ``end_call`` has closed transcript admission.
          Nothing released now may start a turn, so nothing is flushed.
        * ``stt_unresolved`` -- no transcriber could be found on the agent or
          the session. Logged, because a resolver that cannot find the
          component is itself a defect worth a line.
        * ``provider_has_no_flush`` -- the transcriber is not the Azure
          adapter. Deepgram finalizes on its own; Google chirp_2 has no
          flush primitive in the pinned plugin. The event is the value here.
        * ``azure_force_finalization`` -- the Azure adapter's push stream is
          closed and its recognizer rebuilt; the finals it was holding
          arrive on the ordinary path and the SDK commits the turn through
          its normal ``trigger="stt"`` route. A second line follows at the
          end of the outcome window (see :meth:`note_final_transcript`).

        One thing is deliberately NOT a branch. During a reply the
        managed-PSTN word gate withholds finals from the SDK, so a turn can
        look untranscribed to the timer while Azure has in fact finalized
        it. The gate's answer (``held_final``) goes in the log line so the
        rebuild can be recognised as needless after the fact, but it never
        skips the flush: a final held for an EARLIER utterance during the
        reply says nothing about a LATER one in the same reply that stalled,
        and skipping on its account would leave that stall unrecovered
        until the caller spoke again. A needless rebuild costs one
        sub-second recognizer restart with frames queued; a skipped one
        costs the dead air this exists to end.

        The residual risk, stated plainly: the timer can only fire while VAD
        reports silence, so a rebuild mid-utterance needs a VAD miss. When
        that happens the audio is NOT dropped -- the plugin's input channel
        buffers frames until the new recognizer starts (measured against the
        pinned plugin) -- but the recognizer's context is reset at the
        boundary, so a word spanning it may be recognised worse. An empty
        flushed final is dropped by the plugin, the turn stays open, and the
        next VAD cycle re-arms the timer: a line carrying only noise rebuilds
        the recognizer once per cycle. That is left unbounded deliberately,
        because a cap would recreate the dead air this exists to end; the
        ``occurrence`` count in the log is what says it is happening.

        Never raises: this runs inside the SDK's event dispatch on the media
        loop, and an exception there would cost the call more than a stall.
        """

        self._occurrences += 1
        speech_duration = _finite(getattr(event, "speech_duration", 0.0))
        started_at = _finite(getattr(event, "vad_speech_started_at", 0.0))
        turn_age = max(0.0, self._clock() - started_at) if started_at > 0 else 0.0
        details: dict[str, Any] = {
            "provider": self._provider,
            "occurrence": self._occurrences,
            "speech_duration_ms": int(round(speech_duration * 1000)),
            "turn_age_ms": int(round(turn_age * 1000)),
        }
        recovery = self._recover(details)
        details["recovery"] = recovery
        self._log(TIMEOUT_EVENT, details)
        if recovery == RECOVERY_FORCED:
            self._watch_outcome(
                self._occurrences, held_at_flush=bool(details.get("held_final", False))
            )
        return recovery

    def note_final_transcript(self) -> None:
        """A final transcript with text reached the SDK.

        Wired to ``user_input_transcribed`` finals with text. Attributed to
        the oldest forced flush still inside its window, because finals
        arrive in order; a final with no flush waiting is ordinary traffic
        and is ignored here.
        """

        now = self._clock()
        for pending in self._pending:
            if pending["final_at"] is None:
                pending["final_at"] = now
                return

    def _recover(self, details: dict[str, Any]) -> str:
        # A raising ``admits`` reads as admitting and a raising ``holds_final``
        # as not holding: both let the flush proceed, because the recovery is
        # the safe default and a skipped one is the stall.
        if self._admits is not None and not _check(self._admits, default=True):
            return RECOVERY_TERMINAL
        if self._holds_final is not None:
            details["held_final"] = _check(self._holds_final, default=False)
        transcriber = active_stt(self._session)
        if transcriber is None:
            return RECOVERY_UNRESOLVED
        flush = getattr(transcriber, "force_finalization", None)
        if not callable(flush):
            return RECOVERY_NONE
        try:
            flush()
        except Exception as exc:
            details["error"] = type(exc).__name__
            return RECOVERY_FAILED
        self._forced += 1
        return RECOVERY_FORCED

    def _watch_outcome(self, occurrence: int, *, held_at_flush: bool) -> None:
        pending: dict[str, Any] = {
            "occurrence": occurrence,
            "flushed_at": self._clock(),
            "final_at": None,
            # Snapshotted at flush time, because the question the outcome
            # answers is whether the flush PRODUCED a final: a final the gate
            # was already holding, released or kept inside the window, must
            # not be credited to it.
            "held_at_flush": held_at_flush,
        }
        try:
            self._schedule(self._outcome_window, lambda: self._resolve_outcome(pending))
        except Exception:
            # No loop to wait on (only a synchronous caller can get here).
            # Say so rather than leave the flush without its second line.
            self._emit_outcome(pending, OUTCOME_UNOBSERVABLE)
            return
        self._pending.append(pending)

    def _resolve_outcome(self, pending: dict[str, Any]) -> None:
        if pending in self._pending:
            self._pending.remove(pending)
        if pending["held_at_flush"]:
            # A final was already being withheld when the flush was issued.
            # Whatever the gate releases or keeps holding inside the window
            # may be THAT final, so nothing can be credited to the flush;
            # named so the measurement excludes the case instead of
            # miscounting an empty flush as a recovered one.
            outcome = OUTCOME_HELD_BEFORE
        elif pending["final_at"] is not None:
            outcome = OUTCOME_FINAL
        elif self._holds_final is not None and _check(self._holds_final, default=False):
            # Nothing was held when the flush was issued, so a final the word
            # gate holds now is one the flush produced; it reaches the SDK
            # when the reply ends.
            outcome = OUTCOME_HELD
        else:
            outcome = OUTCOME_NONE
        self._emit_outcome(pending, outcome)

    def _emit_outcome(self, pending: dict[str, Any], outcome: str) -> None:
        details: dict[str, Any] = {
            "provider": self._provider,
            "occurrence": pending["occurrence"],
            "outcome": outcome,
            "window_ms": int(round(self._outcome_window * 1000)),
        }
        if outcome == OUTCOME_FINAL and pending["final_at"] is not None:
            details["final_after_ms"] = int(
                round(max(0.0, pending["final_at"] - pending["flushed_at"]) * 1000)
            )
        self._log(OUTCOME_EVENT, details)

    def _log(self, event: str, details: dict[str, Any]) -> None:
        try:
            self._emit(event, **details)
        except Exception:
            pass


def _check(check: Callable[[], bool], *, default: bool) -> bool:
    try:
        return bool(check())
    except Exception:
        return default


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number

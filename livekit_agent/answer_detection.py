# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""The worker's side of post-answer audio classification: buffers and cadence.

The decision lives in ``answer_classifier.py`` and the policy in
``post_answer_arbiter.py``.  Neither of them knows what a media frame is, and
that is what lets them be driven from a test suite with no runtime at all.
This module is the third piece: it holds the samples a live call produces,
hands them to the classifier on a fixed cadence, feeds the verdict to the
arbiter, and hands back the two strings the terminal lifecycle report stamps.

It imports no livekit either.  Frames arrive as amplitudes, transcripts as
text, and voice activity as a boolean, so the whole seam can be driven with
fake tracks on both stacks.

What it owes the classifier
---------------------------

The classifier's activity feed is a step function whose last value holds
forward, and it treats a feed quieter than ``activity_max_age_seconds`` as
absent rather than as continuing.  A detector that only speaks on transitions
therefore cannot be told from one that has died, so this module writes a
sample on every tick as well as on every transition.  The tick IS the
heartbeat; there is no second timer to fall out of step with.

Buffers are trimmed to the window the classifier is asked about, because
``classify`` re-reads the whole window on every tick and a history nobody
trimmed would be re-read on every tick for the length of the call.

What it refuses to do
---------------------

Nothing here ends a call, mutes anything, or withholds anything.  In shadow --
the default, and the only mode the control plane transmits until the action
ships -- the arbiter's decisions are recorded and discarded, and the worker's
existing silence rule is left exactly as it was.
"""
from __future__ import annotations

import array
import dataclasses
import functools
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import shape only
    from . import answer_classifier as _classifier
    from . import post_answer_arbiter as _arbiter
    from . import voicemail_watch as _voicemail
except ImportError:  # pragma: no cover - direct execution inside the image
    import answer_classifier as _classifier  # type: ignore[no-redef]
    import post_answer_arbiter as _arbiter  # type: ignore[no-redef]
    import voicemail_watch as _voicemail  # type: ignore[no-redef]


#: How often the window is re-classified.  Fast enough that the eight-second
#: continuity floor and the fifteen-second decision window are both observed
#: with room to spare, slow enough that the cost of a re-read is nothing on
#: the event loop of a live call.
CLASSIFY_INTERVAL_SECONDS = 1.0

#: How much of the call is kept to classify.  Long enough to cover the widest
#: cap a customer can set plus the span the transform reads behind it, and no
#: longer: a call runs for up to half an hour, and keeping every amplitude
#: measurement of one would hand the classifier ninety thousand samples to
#: re-read every second.
#:
#: What is lost past it is the ability to reach a NEW verdict about the early
#: part of a long call, and that costs nothing, because there is nothing left
#: to decide by then.  A non-human verdict is only accepted inside the
#: decision window, the arbiter keeps the first one it accepted rather than
#: letting a later reading overwrite it, and the cap has already come and
#: gone.  Human evidence is what still matters that late -- a person who
#: appears after the cap is the measurement the rollout gate counts -- and it
#: rides on words and activity, both of which are cheap and both of which are
#: still fed.
RETENTION_MARGIN_SECONDS = 15.0

#: The setting, as it arrives on the wire.  The control plane resolves the
#: platform default before transmitting, so a missing key means a config this
#: worker cannot read rather than a customer who chose nothing -- and the
#: safe reading of that is the mode that changes nothing.
SETTING_SECTION = "answer_detection"
SETTING_KEY = "non_human_audio"

#: The far end counts as producing audio when a frame's mean absolute level
#: reaches this (16-bit samples; a quiet PSTN line measures tens), or the voice
#: activity detector says speaking, and for a second after.
FAR_END_AUDIBLE_AMPLITUDE = 200.0
FAR_END_AUDIBLE_HOLD_SECONDS = 1.0
#: Music-class audio is "onset" after this long, and "active" after the
#: classifier's own continuity floor (``continuous_activity_seconds``).
MUSIC_ONSET_SECONDS = 2.0
#: The music signal reads a ROLLING window this long, not the call so far: music
#: that stops (a person, a pause) and starts again must read as new, and a long
#: silent stretch earlier in the call must not flatter or ruin the reading.
MUSIC_WINDOW_SECONDS = 16.0
#: Up to this share of a stretch may be flagged as voice by the activity detector and still be read as no voice at all
#: (it flickers on a tune; a person talking is flagged for most of what they say).
VOICE_FLICKER_SHARE = 0.1


@dataclass(frozen=True, slots=True)
class MusicReading:
    """What the last window says about music, with NO latch.

    ``onset``: music-class audio for :data:`MUSIC_ONSET_SECONDS`; ``active``:
    for the classifier's continuity floor (8 s); ``audible``: the far end is
    producing audio right now. Recomputed on every classify tick from the
    retained window, in every call state, so music that stops stops being
    reported and music that starts again is a new reading.
    """

    onset: bool = False
    active: bool = False
    audible: bool = False


def frame_amplitude(pcm: bytes, *, stride: int = 4) -> float | None:
    """Mean absolute amplitude of one 16-bit mono frame.

    Mean absolute rather than root-mean-square, and every fourth sample rather
    than all of them.  Both are deliberate and both are free: this runs on the
    loop that drains the caller's media, where the file it is called from
    already warns that evidence about the media must not deform the media, and
    the modulation feature downstream is a RATIO of this signal to its own
    mean, so any constant factor between the two measures cancels exactly.  A
    hundred and sixty samples per frame at fifty frames a second becomes forty.

    Returns ``None`` for a frame it cannot read, and the caller records
    nothing for it, because a damaged frame is not a quiet one.  Returning a
    zero would have been the reasonable-looking mistake: one unreadable frame
    among good ones is then a silent sample in the middle of audio, and enough
    of them pull the envelope's own mean down -- which is the denominator the
    modulation depth is measured against.

    ``array("h")`` reads native byte order and the frames are little-endian.
    Every host this runs on is little-endian; on one that is not, the measure
    would be noise rather than an envelope, and the classifier would see a
    modulation depth it could not have produced.
    """

    try:
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) - (len(pcm) % samples.itemsize)])
    except (TypeError, ValueError):
        return None
    if not samples:
        return None
    step = max(1, int(stride))
    taken = samples[::step]
    return sum(abs(sample) for sample in taken) / len(taken)


def resolved_mode(
    realtime_config: Mapping[str, Any] | None,
    *,
    emit: Callable[..., None] = lambda *_args, **_details: None,
) -> str:
    """The per-agent mode this call runs in, or ``shadow`` if it cannot be read.

    Deliberately without a compiled-in copy of the platform default: the
    control plane substitutes that before the config is sent, so exactly one
    place decides what an unconfigured agent does.  What this fills in is the
    other case -- a config that arrived damaged, truncated or from a control
    plane older than the setting -- and for that the answer is the mode that
    takes no action at all.  It is logged once so a call running in shadow
    because nobody could read its setting is distinguishable from one running
    in shadow because that is what its owner chose.
    """

    section: Any = None
    if isinstance(realtime_config, Mapping):
        section = realtime_config.get(SETTING_SECTION)
    value = section.get(SETTING_KEY) if isinstance(section, Mapping) else None
    if isinstance(value, str) and value in _arbiter.MODES:
        return value
    emit(
        "answer_detection_mode_unreadable",
        present=value is not None,
        mode=_arbiter.DEFAULT_MODE,
    )
    return _arbiter.DEFAULT_MODE


def _never_raises(default_factory: Callable[[], Any] = lambda: None) -> Any:
    """Make a measurement method incapable of ending a call.

    Every one of these is called from somewhere that matters more than they
    do: a media callback, a session event handler that updates the barge-in
    scheduler after it, the answer watcher that reports ``sip.answered`` after
    it, and teardown immediately before the terminal lifecycle report.  At
    every one of those sites this work was INSERTED AHEAD of work that already
    existed, so an exception here does not merely lose a measurement -- it
    strips the side effect that was supposed to follow.  The worst of them
    drops the terminal report, and with it the billing settlement and the
    customer's ``call.ended``.

    Wrapping the call sites instead would put the guarantee in four places and
    leave the fifth caller to remember it.  It belongs here, once, on the
    class whose docstring claims it.  ``BaseException`` deliberately still
    propagates: a cancelled task must stay cancelled.

    ``default_factory`` is what the method returns when it gives up, so one
    whose caller reads its RESULT can say what a safe result looks like.
    ``terminal_details`` is the method that needs it: it is evaluated as an
    argument to the terminal lifecycle report, which makes it the single most
    dangerous call in this class, and an empty mapping is exactly what a call
    that measured nothing already sends.
    """

    def decorate(method: Any) -> Any:
        @functools.wraps(method)
        def guarded(self: "AnswerDetectionSession", *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - never a call gate
                self._note_failure(method.__name__, exc)
                return default_factory()

        return guarded

    return decorate


class AnswerDetectionSession:
    """One call's samples, cadence and verdict.  Safe to drive from anywhere.

    Every public method is best-effort, enforced by ``_never_raises`` rather
    than promised in prose: they are called from media callbacks, event
    handlers and teardown, where an exception would take out something that
    matters far more than this evidence does.
    """

    def __init__(
        self,
        *,
        mode: str = _arbiter.DEFAULT_MODE,
        decision_window_seconds: float = _arbiter.DECISION_WINDOW_DEFAULT_SECONDS,
        cap_seconds: float = _arbiter.CAP_DEFAULT_SECONDS,
        no_caller_speech_seconds: float | None = 20.0,
        thresholds: _classifier.ClassifierThresholds = _classifier.DEFAULT_THRESHOLDS,
        emit: Callable[..., None] = lambda *_args, **_details: None,
        clock: Callable[[], float] = time.monotonic,
        voicemail_mode: str = _voicemail.MODE_OFF,
        language: str = "",
        measure_for_navigation: bool = False,
    ) -> None:
        # An agent that opted in to navigation needs the window measured whatever
        # its non-human-audio or voicemail setting says: the music signal and
        # the far end's audibility are read from it. Nothing is DECIDED from the
        # measurement here -- the arbiter still acts only in its own mode.
        self._measure_for_navigation = bool(measure_for_navigation)
        #: The last exit from a menu or a hold (``None`` until one): the floor of
        #: every window classified afterwards.
        self._exit_floor: float | None = None
        self._music_carry_until: float | None = None
        #: What the music read just before the tick an episode ended on, and on it.
        self._music_before = MusicReading()
        self._music_carried = MusicReading()
        self._last_audible_at: float | None = None
        #: When the voice activity detector last said the far end stopped speaking (``None`` before it did).
        self._last_speech_end_at: float | None = None
        #: When the far end last BECAME audible after at least a second of
        #: silence: the music window never reaches back past a silence, so
        #: music that restarts is read from its restart.
        self._audible_since: float | None = None
        self._music_reading = MusicReading()
        self._arbiter = _arbiter.PostAnswerArbiter(
            mode=mode,
            decision_window_seconds=decision_window_seconds,
            cap_seconds=cap_seconds,
            no_caller_speech_seconds=no_caller_speech_seconds,
        )
        self._thresholds = thresholds
        self._emit = emit
        self._clock = clock
        self._retention_seconds = (
            cap_seconds
            + thresholds.modulation_span_seconds
            + RETENTION_MARGIN_SECONDS
        )
        self._activity: list[_classifier.ActivitySample] = []
        self._envelope: list[_classifier.EnvelopeSample] = []
        self._words: list[_classifier.WordEvidence] = []
        # When the recogniser could not have heard the caller, if ever: set by
        # a worker whose engine's input is deliberately withheld.
        self._words_unobservable_until: Callable[[float], float | None] | None = None
        self._speaking = False
        self._last_classified_at: float | None = None
        self._classification: _classifier.AnswerClassification | None = None
        self._failures: dict[str, int] = {}
        # Voicemail detection: a second detector over the same call
        # with its OWN setting. Collection is the union of the two: the
        # classifier's buffers are filled and it is re-read when EITHER is on,
        # but only the non-human setting's arbiter ever sees a reading, and
        # only it reports a classification.  So ``non_human_audio=off`` still
        # reports nothing, and the voicemail watch -- shadow in this release --
        # reaches no outcome at all.
        self._voicemail = _voicemail.VoicemailWatch(
            configured_mode=voicemail_mode, language=language, emit=emit
        )
        self._measure_armed_at: float | None = None

    def _note_failure(self, where: str, exc: BaseException) -> None:
        """Say it once per kind, then count.  A classifier that fails every
        tick would otherwise emit eighteen hundred events on a long call."""

        seen = self._failures.get(where, 0)
        self._failures[where] = seen + 1
        if seen:
            return
        try:
            self._emit(
                "answer_detection_failed", where=where, error_type=type(exc).__name__
            )
        except Exception:  # noqa: BLE001 - reporting a failure cannot fail louder
            pass

    @property
    def failures(self) -> dict[str, int]:
        """How many times each measurement gave up, for the call's evidence."""

        return dict(self._failures)

    # -- state ---------------------------------------------------------

    @property
    def collecting(self) -> bool:
        return self._arbiter.collecting and not self._arbiter.committed

    @property
    def _measuring(self) -> bool:
        """Whether the classifier's buffers are filled at all: either detector."""

        return (
            self.collecting
            or (self._voicemail.active and not self._arbiter.committed)
            or (self._measure_for_navigation and not self._arbiter.committed)
        )

    @property
    def _measure_armed(self) -> bool:
        return self._arbiter.armed or self._measure_armed_at is not None

    @property
    def voicemail(self) -> _voicemail.VoicemailWatch:
        """The voicemail watch, for the worker task that drives it."""

        return self._voicemail

    @property
    def armed(self) -> bool:
        return self._arbiter.armed

    @property
    def acting_mode(self) -> bool:
        """Whether this call's SETTING acts, known before anything is armed.

        Distinct from the arbiter's ``acts``, which also requires real answer
        evidence and so is false for the first seconds of every call.  The
        mute's enable is decided when the session is built, before the gate
        resolves.
        """

        return self._arbiter.mode == _arbiter.MODE_END_CALL

    @property
    def arbiter(self) -> _arbiter.PostAnswerArbiter:
        """The policy state machine, for the caller that acts on it."""

        return self._arbiter

    # -- signals -------------------------------------------------------

    @_never_raises()
    def arm(self, disposition: str, *, wait_ms: int | None = None) -> None:
        """Take the answer gate's disposition.  Everything measures from here."""

        at = self._clock()
        self._arm_voicemail(disposition, wait_ms=wait_ms, at=at)
        if (
            self._measure_for_navigation
            and self._measure_armed_at is None
            and str(disposition) in _arbiter.ARMING_DISPOSITIONS | _arbiter.SHADOW_ONLY_DISPOSITIONS
        ):
            self._measure_armed_at = at
        if not self._arbiter.collecting:
            return
        acting = self._arbiter.arm(disposition, at=at)
        if self._arbiter.armed:
            self._emit(
                "answer_detection_armed",
                disposition=disposition,
                mode=self._arbiter.mode,
                # What this call can actually do, which is not the same as the
                # mode: a fail-open disposition watches in every mode.
                acts=acting,
            )

    def _arm_voicemail(self, disposition: str, *, wait_ms: int | None, at: float) -> None:
        # Its own guard: a failure here must not stop the arbiter arming.
        try:
            if not self._voicemail.enabled:
                return
            if str(disposition) not in _arbiter.ARMING_DISPOSITIONS | _arbiter.SHADOW_ONLY_DISPOSITIONS:
                return
            self._voicemail.arm(str(disposition), wait_ms=wait_ms, at=at)
            if self._measure_armed_at is None:
                self._measure_armed_at = at
        except Exception as exc:  # noqa: BLE001 - measurement never gates a call
            self._note_failure("voicemail_arm", exc)

    # -- voicemail feeds -------------------------------------------------

    @_never_raises()
    def note_media_frame(
        self, role: str, pcm: bytes, *, media_start: float, received_at: float, gap: bool = False
    ) -> None:
        """One frame of either track on the recorder's media timeline.

        An append to a bounded ring and nothing else: this is called from the
        media drain, and the arithmetic runs from ``process_voicemail``.
        """

        self._voicemail.note_media_frame(
            role, pcm, media_start=media_start, received_at=received_at, gap=gap
        )

    @_never_raises()
    def note_agent_text(self, text: Any, *, at: float | None = None) -> None:
        """Something the agent said, for recognising our own echo."""

        self._voicemail.note_agent_text(text, at=self._clock() if at is None else float(at))

    @_never_raises()
    def set_words_unobservable_until(
        self, source: Callable[[float], float | None] | None
    ) -> None:
        """Where to ask when recognised words were impossible, not merely absent.

        A voice-to-voice engine captions the caller from the same input the
        worker withholds while the opening plays; zero words from that window
        must not count toward a non-human verdict.
        """

        self._words_unobservable_until = source

    def _unobservable_until(self, now: float) -> float | None:
        source = self._words_unobservable_until
        if source is None:
            return None
        try:
            value = source(now)
        except Exception:
            # An unreadable window is the whole window: words were not
            # observable up to now, which only ever delays a non-human verdict.
            return now
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else now

    @_never_raises()
    def note_input_released(self) -> None:
        self._voicemail.note_input_released(at=self._clock())

    @_never_raises()
    def note_opener_started(self) -> None:
        self._voicemail.note_opener_started(at=self._clock())

    @_never_raises()
    def process_voicemail(self) -> None:
        self._voicemail.process(now=self._clock())
        observer = self._voicemail.observer
        if observer is not None:
            self._arbiter.note_voicemail(observer.classification())

    @_never_raises()
    def voicemail_section(self, ended_by: str) -> dict[str, Any] | None:
        """The frozen ``voicemail_detection`` section, or ``None`` (off, or
        a call that never answered)."""

        return self._voicemail.freeze(ended_by=str(ended_by or ""), now=self._clock())

    @_never_raises()
    def note_activity(self, speaking: bool) -> None:
        """A voice-activity transition on the caller track."""

        speaking = bool(speaking)
        if speaking == self._speaking and self._activity:
            return
        if self._speaking and not speaking:
            self._last_speech_end_at = self._clock()  # (speech too quiet to count as audible still ends somewhere)
        self._speaking = speaking
        self._append_activity(self._clock())

    @_never_raises()
    def note_amplitude(self, amplitude: float) -> None:
        """One short-term amplitude measurement of the caller track."""

        if not self._measuring or not self._measure_armed:
            return
        try:
            value = float(amplitude)
        except (TypeError, ValueError):
            return
        now = self._clock()
        self._envelope.append(_classifier.EnvelopeSample(at=now, amplitude=max(0.0, value)))
        if value >= FAR_END_AUDIBLE_AMPLITUDE:
            if self._last_audible_at is None or now - self._last_audible_at > FAR_END_AUDIBLE_HOLD_SECONDS:
                self._audible_since = now
            self._last_audible_at = now

    @property
    def music_reading(self) -> MusicReading:
        """The last classify tick's music reading (see :class:`MusicReading`)."""

        return self._music_reading

    def far_end_audible(self) -> bool:
        """Whether the far end is producing audio right now."""

        if self._speaking:
            return True
        last = self._last_audible_at
        return last is not None and self._clock() - last <= FAR_END_AUDIBLE_HOLD_SECONDS

    def far_end_quiet_seconds(self) -> float:
        """Seconds since the far end last produced audio or was last heard to stop speaking; ``0.0`` while it is audible or
        speaking, a very large number if it never has.

        The same signals as :meth:`far_end_audible` (the later of the last audible frame and the voice detector's last end of
        speech) without its one-second hold, for a reader that waits for the line to settle after something was said (a menu
        read in pieces).
        """

        if self._speaking:
            return 0.0
        marks = [mark for mark in (self._last_audible_at, self._last_speech_end_at) if mark is not None]
        if not marks:
            return float("inf")
        return max(0.0, self._clock() - max(marks))

    @_never_raises()
    def note_frame(self, pcm: bytes) -> None:
        """One caller frame, measured only if anything will read it.

        The gate comes BEFORE the arithmetic, which is the whole reason this
        exists beside ``note_amplitude``: a call spends its ringing seconds
        with frames arriving and nothing armed, and an agent whose owner chose
        ``off`` spends all of them that way.  Measuring first and discarding
        after put two allocations per frame on the loop that drains the
        caller's media, for the life of every such call.
        """

        if not self._measuring or not self._measure_armed:
            return
        measured = frame_amplitude(pcm)
        if measured is None:
            return
        self.note_amplitude(measured)

    @_never_raises()
    def note_transcript(self, text: Any, *, source: str = "", at: float | None = None) -> None:
        """Recognised caller words: an STT final, or a live caption."""

        now_for_voicemail = self._clock() if at is None else float(at)
        if math.isfinite(now_for_voicemail):
            try:
                self._voicemail.note_caller_final(text, at=now_for_voicemail)
            except Exception as exc:  # noqa: BLE001 - its own guard, see _arm_voicemail
                self._note_failure("voicemail_transcript", exc)
        if not self._measuring or not self._measure_armed:
            return
        words = len(str(text or "").split())
        if words <= 0:
            return
        now = self._clock() if at is None else float(at)
        if not math.isfinite(now):
            return
        self._words.append(
            _classifier.WordEvidence(
                at=now,
                words=words,
                source=source or _classifier.WORD_SOURCE_STT_FINAL,
            )
        )
        if not self.collecting or not self._arbiter.armed:
            return
        # The same utterance, read for repetition rather than for its word
        # count.  The text stays inside the arbiter's detector; what leaves is
        # the moment a loop is established, and how long its cycle is.
        stood_before = self._arbiter.repeating_recording_stands
        period, recurrences = self._arbiter.note_far_end_text(text, at=now)
        if self._arbiter.repeating_recording_stands and not stood_before:
            self._emit(
                "answer_detection_repeating_recording",
                period=period,
                recurrences=recurrences,
                source=source or _classifier.WORD_SOURCE_STT_FINAL,
            )

    # -- cadence -------------------------------------------------------

    @_never_raises()
    def tick(
        self, *, caller_has_spoken: bool = False, navigation_suspended: bool = False
    ) -> _arbiter.ArbiterDecision | None:
        """Re-classify the window and let the arbiter read it.

        Returns the arbiter's decision so a caller that is allowed to act can
        act on it.  In shadow the decision carries ``acts=False`` and this
        return value is evidence, not an instruction.
        """

        if not self._measuring or not self._measure_armed:
            return None
        now = self._clock()
        if (
            self._last_classified_at is not None
            and now - self._last_classified_at < CLASSIFY_INTERVAL_SECONDS
        ):
            return None
        self._last_classified_at = now
        # The heartbeat the classifier's staleness rule needs, written on the
        # same cadence as the classification so the two cannot drift.
        self._append_activity(now)
        # ``is None``, not ``or``: an arming time of exactly zero is a time,
        # and reading it as "no time" collapses the window onto the current
        # instant, so every classification reports silence on a call full of
        # audio.
        armed_at = self._arbiter.armed_at
        if armed_at is None:
            armed_at = self._measure_armed_at
        if armed_at is None:
            armed_at = now
        # Classified over what is actually RETAINED, not over the window the
        # call has been open for.  Handing the classifier an ``armed_at`` from
        # ten minutes ago while holding only the last minute of samples would
        # have it measure a window most of which nobody kept, and report the
        # silence of a trimmed buffer as the silence of a caller.
        window_start = max(armed_at, now - self._retention_seconds)
        if self._exit_floor is not None:
            window_start = max(window_start, self._exit_floor)
        self._trim(window_start)
        classification = _classifier.classify(
            activity=self._activity,
            envelope=self._envelope,
            words=self._words,
            armed_at=window_start,
            now=now,
            thresholds=self._thresholds,
            words_unobservable_until=self._unobservable_until(now),
        )
        self._read_music(window_start, now)
        previous = self._classification
        self._classification = classification
        try:
            self._voicemail.note_classification(
                verdict=classification.verdict,
                speech_like=classification.speech_like_modulation,
                at=now,
            )
        except Exception as exc:  # noqa: BLE001 - its own guard, see _arm_voicemail
            self._note_failure("voicemail_classification", exc)
        if not self.collecting or not self._arbiter.armed:
            # Measured for the voicemail watch only: the non-human setting is
            # off, so nothing is observed, emitted or decided about it.
            return None
        self._arbiter.observe(classification, at=now)
        if previous is None or previous.verdict != classification.verdict:
            self._emit(
                "answer_classification_changed",
                **classification.evidence(),
            )
        return self._arbiter.poll(
            at=now,
            caller_has_spoken=bool(caller_has_spoken),
            navigation_suspended=bool(navigation_suspended),
        )

    @_never_raises()
    def note_navigation_exit(self) -> None:
        """The call returned from a phone menu or a hold queue to conversation."""

        now = self._clock()
        # What the line carried BEFORE the exit belongs to the episode: the
        # retained window is floored at the exit so that the next classify tick
        # cannot rebuild, from queue audio, the verdict the exit just cleared.
        self._exit_floor = now
        # Music that keeps playing under the exit is the same music: until the
        # floored window has history of its own, the reading is carried, so
        # warming up is not mistaken for a falling edge (and then a rising one).
        self._music_carry_until = now + _classifier.DEFAULT_THRESHOLDS.continuous_activity_seconds
        # The reading on the exit tick itself is taken from the window that
        # still holds a confirming utterance (utterance-shaped, so "not music"):
        # carry what the tick BEFORE it read as well.
        self._music_carried = MusicReading(
            onset=self._music_reading.onset or self._music_before.onset,
            active=self._music_reading.active or self._music_before.active,
        )
        # ...and publish it now, so that what the policy takes as already seen
        # on this tick is what the next tick carries on from.
        self._music_reading = MusicReading(
            onset=self._music_carried.onset,
            active=self._music_carried.active,
            audible=self._music_reading.audible,
        )
        self._arbiter.note_navigation_exit(at=now)

    def _read_music(self, window_start: float, now: float) -> None:
        """The music signal over the same window, recognised words ignored.

        Words are left out on purpose: the classifier reads a recognised word as
        a person, and an announcement over music must not unmake the music. The
        onset reading is the same classifier with a 2 s continuity floor.
        """

        if not self._measure_for_navigation:
            return
        try:
            start = max(window_start, now - MUSIC_WINDOW_SECONDS)
            if self._audible_since is not None:
                start = max(start, self._audible_since)
            # The activity feed is a step function whose last value holds
            # forward: keep the sample that says what was true at ``start``.
            before = [sample for sample in self._activity if sample.at <= start]
            activity = before[-1:] + [sample for sample in self._activity if sample.at > start]
            envelope = [sample for sample in self._envelope if sample.at >= start]

            def music(floor: float | None) -> bool:
                thresholds = (
                    self._thresholds
                    if floor is None
                    else dataclasses.replace(self._thresholds, continuous_activity_seconds=floor)
                )
                reading = _classifier.classify(
                    activity=activity,
                    envelope=envelope,
                    words=[],
                    armed_at=start,
                    now=now,
                    thresholds=thresholds,
                )
                return reading.verdict == _classifier.VERDICT_NON_HUMAN_AUDIO

            onset, active = music(MUSIC_ONSET_SECONDS), music(None)
            if not active:
                # The activity detector decides speech from a voice model; a tune it never takes for a voice leaves
                # its feed silent, and the classifier above then reads the line as silent. A line that is AUDIBLE the
                # whole time while that feed stays silent is music or a tone all the same.
                held = self._sustained_audio_without_voice(now)
                onset = onset or held >= MUSIC_ONSET_SECONDS
                active = held >= self._thresholds.continuous_activity_seconds
            audible = self.far_end_audible()
            if self._music_carry_until is not None:
                # A dip of the line below audibility inside the carry is not
                # the music stopping: the carry runs to its end either way.
                if now >= self._music_carry_until:
                    self._music_carry_until = None
                else:
                    onset = onset or self._music_carried.onset
                    active = active or self._music_carried.active
            self._music_before = self._music_reading
            self._music_reading = MusicReading(onset=onset, active=active, audible=audible)
        except Exception as exc:  # noqa: BLE001 - a reading is never a media gate
            self._note_failure("music_reading", exc)

    def _sustained_audio_without_voice(self, now: float) -> float:
        """Seconds the line has been audible without a break while the voice detector stayed silent.

        0 when it is not audible now, or when the detector flagged more than a flicker of voice inside the stretch
        (the classifier owns that case: speech, an announcement over music, a tune the detector does follow). The
        stretch is the one :meth:`note_amplitude` keeps: it starts when audio returns after a gap of more than
        :data:`FAR_END_AUDIBLE_HOLD_SECONDS`. No flatness is asked of the envelope: the voice detector's silence over
        a stretch of audible audio is the evidence (measured: the harness's tune modulates at 0.16-0.19, above the
        talker line of 0.15, and no voice model flagged it).
        """

        since = self._audible_since
        last = self._last_audible_at
        if since is None or last is None or now - last > FAR_END_AUDIBLE_HOLD_SECONDS:
            return 0.0
        seconds = now - since
        if seconds < MUSIC_ONSET_SECONDS:
            return 0.0
        # The detector's silence is judged over the RECENT stretch (the length the music reading itself needs), not the
        # whole of it: after a minute of music a person's four seconds are a small share of the stretch and the whole of
        # what matters now, and retained activity does not reach back over a long stretch anyway.
        recent = min(seconds, self._thresholds.continuous_activity_seconds)
        runs = _classifier.activity_runs(self._activity, after=now - recent, now=now)
        if sum(run.seconds for run in runs) > VOICE_FLICKER_SHARE * recent:
            return 0.0
        return seconds

    # -- terminal ------------------------------------------------------

    @_never_raises()
    def commit(self, reason: str) -> _arbiter.TerminalRecord | None:
        """Freeze the record for this call, once, at teardown."""

        if not self._arbiter.collecting or not self._arbiter.armed:
            # A leg that never answered is a third population, and neither of
            # the two the control plane distinguishes.  Stamping ``unknown``
            # on it would say the detector ran and reached no verdict, when
            # the truth is that there was nothing to run on: no answer, no
            # window, no audio.  Every ringing-timeout and busy call would
            # otherwise enter the denominator the rollout gate divides by.
            return None
        already = self._arbiter.committed
        record = self._arbiter.commit(str(reason), at=self._clock())
        if record is None:  # pragma: no cover - defensive; commit returns a record
            return None
        if not already:
            self._emit("answer_detection_terminal", **record.evidence())
        return record

    @_never_raises(dict)
    def terminal_details(self) -> dict[str, str]:
        """The two keys the terminal lifecycle report stamps, or nothing.

        Empty when this call collected nothing, which is what the control
        plane stores as "not reported".  A committed record always carries a
        classification, ``unknown`` included: that is a verdict the detector
        reached, not a missing value, and the two are told apart here rather
        than by a reader guessing.
        """

        record = self._arbiter.terminal
        if record is None:
            return {}
        return {
            "answer_classification": record.classification,
            "reason_detail": record.detail,
        }

    # -- internals -----------------------------------------------------

    #: Transitions kept while the call is still ringing.  At most one of them
    #: survives the first armed tick anyway -- the feed is a step function and
    #: ``_trim`` keeps exactly one sample from before the window -- so this is
    #: a ceiling on a list nothing reads, not a policy.
    _PRE_ARM_ACTIVITY_LIMIT = 64

    def _append_activity(self, at: float) -> None:
        if not self._measuring:
            return
        self._activity.append(
            _classifier.ActivitySample(at=at, active=self._speaking)
        )
        if not self._measure_armed and len(self._activity) > self._PRE_ARM_ACTIVITY_LIMIT:
            self._activity = self._activity[-2:]

    def _trim(self, window_start: float) -> None:
        """Keep only what the next classification will read.

        One sample from before the window is KEPT deliberately, for each feed
        that is a step function: the activity feed's value holds forward, so
        dropping the transition that opened a run already in progress would
        turn continuing audio into audio that started later than it did.
        """

        self._envelope = [
            sample for sample in self._envelope if sample.at >= window_start
        ]
        self._words = [word for word in self._words if word.at >= window_start]
        kept: list[_classifier.ActivitySample] = []
        for sample in self._activity:
            if sample.at >= window_start:
                kept.append(sample)
            else:
                kept = [sample]
        self._activity = kept


__all__ = [
    "AnswerDetectionSession",
    "frame_amplitude",
    "CLASSIFY_INTERVAL_SECONDS",
    "SETTING_KEY",
    "SETTING_SECTION",
    "resolved_mode",
]

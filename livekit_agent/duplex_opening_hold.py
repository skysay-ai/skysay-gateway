# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Keep a voice-to-voice model from hearing the call until its opening is spoken.

**Why.** On a voice-to-voice engine the opening is not a playout. The worker
asks the model to say the greeting (``generate_reply`` becomes a commentary
event: "say this once, in your own words"), and the model receives that ask
while it is already hearing the far end. The engine answers whatever it hears,
so audio that arrives before or during the ask -- hold music, an IVR prompt, a
callee's "hello?" -- can take the first turn instead. Measured on the Sinhala
batch of 2026-09-24: an already-active line played music at 290 ms, caller
audio and the opening ask reached the model in the same instant, and the first
words the callee heard were an English reply to the song. The configured
opening was never spoken.

The runtime cannot protect this turn. Duplex sessions do their own barge-in, so
the SDK never pauses or interrupts them, and ``allow_interruptions=False`` on
the opening is rewritten to the default. Nor can the adapter say what was
spoken: it credits "the next speech to open" to the pending ask, so a reply to
the music is booked as the opening.

**What this does.** From the moment caller audio is released until the opening
has been spoken, every frame bound for the model is replaced with silence of
the same shape. Only the model is deafened: the frames are substituted at the
duplex session's own ``push_audio``, downstream of the voice activity
detector, the answer classifier, the recorder and every other local consumer,
which keep receiving the real audio. Substitution rather than the provider's
mute command, because that command is per-connection server state, is queued
behind audio already waiting to be sent, and is acknowledged only in a debug
log -- a reconnect or a lost send would leave the model hearing again with
nothing to notice it. Silence pushed locally needs no acknowledgement and
survives a reconnect by construction.

**When it lets go.** Never on the end of a speech burst: bursts close on
0.8 s of quiet, and a sentence pause inside the greeting is longer than that.
A literal opening is released once the model's own transcript of its speech
has covered the opening text AND the room has stopped hearing the agent for
:data:`QUIET_SECONDS`. A generated opening has no text to compare, so it is
released after the agent has been audible for at least
:data:`GENERATED_MIN_AUDIBLE_SECONDS` and then quiet for the longer
:data:`GENERATED_QUIET_SECONDS`. Quiet is measured to the last frame the evidence
sidecar delivered, so time it did not observe is never counted as silence, and
a provider reconnect restarts both the text coverage and the audio clocks,
because fragments streamed ahead of audio on the old connection are not
carried over. Anything else ends on a bounded backstop recorded as a
failure (``incomplete``, ``no_audio``), never as a greeting that played.

**What it cannot promise.** A duplex model may still paraphrase or translate a
literal line; the outcome names that (``divergent``) instead of crediting it.
A line that ends ``covered`` was covered by the model's transcript, heard by
the room across enough of its span, followed by quiet -- and the provider's
own output audio reached the text's ``end_ms`` on its timeline, to within
:data:`OUTPUT_SLACK_MS` (measured: that timeline is the count of output
samples). An engine that sends no output events cannot be judged this way;
its outcome stands and is counted as ``output_unjudged``. A line whose audio
stopped short is released all the same, since the rest is not coming, but it
is labelled ``incomplete``. What the room PLAYED is still judged by energy
against the span, which cannot tell pauses between words from lost audio
exactly.
The model still does not hear a caller speaking over the opening. After its
last audible frame, a bounded in-memory tail retains input until the normal
release. It is replayed once only when provider delivery and continuous room
observation support the opening's successful release. Resumed agent audio,
unknown delivery, reconnects, overflow and teardown discard it. This recovers
quick replies lost in the quiet tail, not speech overlapping the greeting.
No semantic classification or confirmation is inferred from retained audio.

Imports no livekit: the frame type is whatever the session pushes, and a
silent copy is built through that type's own constructor.
"""

from __future__ import annotations

import array
import asyncio
import math
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any

#: How long the room must stop hearing the agent before a covered literal
#: opening counts as finished. Above the adapter's 0.8 s burst gap, so the
#: pause between two sentences of the greeting is not its end.
QUIET_SECONDS = 1.2

#: The same for a generated opening, which has no text to prove it is over and
#: so waits longer on audio alone.
GENERATED_QUIET_SECONDS = 1.5

#: Share of the opening's character bigrams the model's transcript must
#: contain for the line to count as spoken as written. Recall, not similarity:
#: a model that says the line and then keeps talking has still said it.
MATCH_RECALL = 0.8

#: How much of the opening's length must have been transcribed before the
#: match is judged at all. Below it the line is still being spoken.
COVERAGE_FRACTION = 0.9

#: Backstop after the first audible frame: a base plus a per-character
#: allowance for the configured line, capped. A 68-character line gets 12.8 s
#: against its measured 6-7 s playout.
BACKSTOP_BASE_SECONDS = 6.0
BACKSTOP_SECONDS_PER_CHARACTER = 0.1
BACKSTOP_MAX_SECONDS = 30.0

#: How much agent audio a generated opening needs before quiet can end it. A
#: single loud frame followed by the model composing is not a greeting.
GENERATED_MIN_AUDIBLE_SECONDS = 1.0

#: Caller frames at or above this RMS (16-bit) count as speech the hold kept
#: from the model -- the evidence that a hold actually withheld something,
#: rather than a count that includes the silence of an empty line.
SPEECH_FRAME_RMS = 300.0

#: The longest the caller is ever kept from the model, whatever happens to the
#: connection: the longest per-line backstop plus the no-audio one.
ABSOLUTE_BACKSTOP_SECONDS = 42.0

#: Backstop when the model has produced nothing audible since the hold began.
#: Above the adapter's 10 s wait for a requested reply to start.
NO_AUDIO_BACKSTOP_SECONDS = 12.0

#: The sidecar must have delivered a frame at least this recently for quiet to
#: count; a longer gap restarts the observed quiet from the moment frames
#: resume, because time nobody watched is not silence anybody heard.
OBSERVATION_GAP_SECONDS = 0.3

#: The provider error codes the shared GPT-Live core treats as fatal (its
#: ``_FATAL_ERROR_CODES``). Mirrored so this module need not import livekit;
#: the image probe asserts the two are equal against the pinned SDK.
FATAL_PROVIDER_ERROR_CODES = frozenset(
    {"insufficient_quota", "invalid_api_key", "account_deactivated", "billing_hard_limit_reached",
     "forbidden"}
)

#: The room must have heard at least this many seconds of AUDIBLE agent audio,
#: as a share of the span the model's own transcript says the spoken text
#: occupies. Counted from the recorder's per-frame PCM, so a transport stall
#: inside the line adds nothing; a line cut short by a dropped connection
#: stops short of it however late the drop is reported. Below 1 because the
#: span includes the quiet between words, which the audible count does not:
#: measured on real GPT-Live openings (local SIP harness, 2026-09-24) the
#: audible share of the span was 0.58-0.65 (1.70-1.74 s of 2.6-3.0 s), so
#: 0.45 leaves a whole opening clear and still refuses a line that lost more
#: than about a quarter of its audio.
SPAN_COVERAGE_FRACTION = 0.45

#: Slack on the heard span: the audible edge is sampled per frame against a
#: loudness threshold, so a soft last syllable can end it a little early.
SPAN_SLACK_SECONDS = 0.3

#: The GPT-Live plugin's output audio rate (its ``SAMPLE_RATE``), mirrored so
#: this module need not import livekit; the image probe asserts the two agree.
OUTPUT_SAMPLE_RATE = 24000

#: How far the provider's output audio may fall short of the transcript's
#: ``end_ms`` and the line still count as ``covered``: one output chunk.
#:
#: Measured (5 real GPT-Live openings, 2026-09-25): the provider
#: streams output as fixed 100 ms chunks, silence included, in real time and
#: never ahead of it, and cumulative output samples ARE the timeline the
#: transcript's ``start_ms``/``end_ms`` are on (the first fragment starts at 0
#: with 0 samples received). Each fragment arrives 200-800 ms before its
#: audio, so text alone can cover a line whose last ~0.8 s never arrived.
OUTPUT_SLACK_MS = 100.0

#: How often the running hold re-reads the room.
POLL_SECONDS = 0.2

# Retain only the short, observed quiet tail AFTER the last opening audio.
# No audio overlapping the greeting is replayed. All limits are independent
# of frame format and scheduling; overflowing abandons that whole tail.
TAIL_SECONDS = 2.0
TAIL_BYTES = 384000
TAIL_FRAMES = 256

OUTCOME_COVERED = "covered"
OUTCOME_DIVERGENT = "divergent"
OUTCOME_GENERATED = "generated_spoken"
OUTCOME_INCOMPLETE = "incomplete"
OUTCOME_NO_AUDIO = "no_audio"
OUTCOME_CALL_ENDING = "call_ending"
OUTCOME_UNSUPPORTED = "unsupported"

#: Every outcome a hold can end with; the summary stores nothing else.
OUTCOMES = frozenset(
    {
        OUTCOME_COVERED,
        OUTCOME_DIVERGENT,
        OUTCOME_GENERATED,
        OUTCOME_INCOMPLETE,
        OUTCOME_NO_AUDIO,
        OUTCOME_CALL_ENDING,
        OUTCOME_UNSUPPORTED,
    }
)

#: Outcomes of a hold that actually withheld audio, and so can say what was
#: spoken. ``unsupported`` is the absence of a hold, not a judgement.
JUDGED_OUTCOMES = OUTCOMES - {OUTCOME_UNSUPPORTED}

#: Outcomes that mean the opening's audio reached the caller: the evidence the
#: summary's ``opening_turns_reached_audio`` credit rests on. ``covered`` is
#: coverage observed (see the module docstring), not delivery proven to the
#: last word. Everything else is a hold that ended without it.
OPENING_DELIVERED = frozenset({OUTCOME_COVERED, OUTCOME_GENERATED})

EVENT = "duplex_opening_hold"

_IGNORED = re.compile(r"[\W_]+", re.UNICODE)


def _normalized(text: str) -> str:
    """Letters and digits only, casefolded and NFC, so punctuation, spacing
    and a model's own rendering of a comma cannot decide the match."""

    return _IGNORED.sub("", unicodedata.normalize("NFC", str(text or "")).casefold())


def _bigrams(text: str) -> list[str]:
    return [text[index : index + 2] for index in range(len(text) - 1)] or (
        [text] if text else []
    )


def opening_recall(opening: str, spoken: str) -> float:
    """Share of the opening's character bigrams found in what was spoken.

    Counted as a multiset, so a bigram the opening uses three times must be
    spoken three times. ``0.0`` for an empty opening: nothing to recall is not
    a match.
    """

    wanted = _bigrams(_normalized(opening))
    if not wanted:
        return 0.0
    available: dict[str, int] = {}
    for gram in _bigrams(_normalized(spoken)):
        available[gram] = available.get(gram, 0) + 1
    found = 0
    for gram in wanted:
        if available.get(gram, 0) > 0:
            available[gram] -= 1
            found += 1
    return found / len(wanted)


def _frame_rms(frame: Any) -> float:
    """RMS of a 16-bit frame; 0.0 for anything that cannot be read."""

    try:
        samples = array.array("h")
        samples.frombytes(bytes(frame.data))
    except Exception:
        return 0.0
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _is_connection_failure(error: Any) -> bool:
    """Whether an error the plugin reported is its connection failing.

    By class name along the MRO, so this module need not import livekit:
    ``APIConnectionError`` (and its ``APITimeoutError``) is what the plugin
    raises when the socket drops or cannot be opened.
    """

    try:
        return any(cls.__name__ == "APIConnectionError" for cls in type(error).__mro__)
    except Exception:
        return False


def _finite_or_none(source: Callable[[], float | None]) -> float | None:
    try:
        value = source()
    except Exception:
        return None
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def backstop_seconds(opening: str) -> float:
    return min(
        BACKSTOP_MAX_SECONDS,
        BACKSTOP_BASE_SECONDS + BACKSTOP_SECONDS_PER_CHARACTER * len(str(opening or "")),
    )


def silent_copy(frame: Any) -> Any:
    """A frame of the same shape carrying digital silence.

    Built through the frame's own type, so the pinned ``rtc.AudioFrame`` is
    reconstructed without this module importing livekit, and a resampler
    downstream sees exactly the rate, channel count and length it was given.
    """

    samples = int(frame.samples_per_channel)
    channels = int(frame.num_channels)
    return type(frame)(
        data=bytes(samples * channels * 2),
        sample_rate=int(frame.sample_rate),
        num_channels=channels,
        samples_per_channel=samples,
    )


class DuplexOpeningHold:
    """One call's hold on what the model hears, from input release to opening end.

    Loop-affine like every other per-call rule in the worker: ``engage``,
    ``poll``, ``release`` and the transcript listener all run on the loop
    driving the media session, and none of them awaits.
    """

    def __init__(
        self,
        *,
        opening_text: str,
        literal: bool,
        agent_audible_until: Callable[[], float | None],
        observed_until: Callable[[], float | None] = lambda: None,
        observed_since: Callable[[], float | None] = lambda: None,
        agent_audible_seconds: Callable[[], float] | None = None,
        emit: Callable[..., None] = lambda *_args, **_details: None,
        clock: Callable[[], float] = time.monotonic,
        silence: Callable[[Any], Any] = silent_copy,
    ) -> None:
        self._opening = str(opening_text or "")
        # A literal mode with no line to compare falls back to the audio-only
        # rule: there is nothing to verify, and pretending to would release on
        # the first transcript fragment.
        self._literal = bool(literal) and bool(_normalized(self._opening))
        self._agent_audible_until = agent_audible_until
        self._observed_until = observed_until
        self._observed_since_source = observed_since
        self._emit = emit
        self._clock = clock
        self._silence = silence
        self._installed = False
        self._held = False
        self._engaged_at: float | None = None
        self._released_at: float | None = None
        self._first_audio_at: float | None = None
        self._last_audio_at: float | None = None
        # Audible time actually accumulated, not the span between two edges:
        # two blips a second apart are 40 ms of speech, not a second of it.
        self._audible_seconds = 0.0
        self._agent_audible_seconds = agent_audible_seconds
        self._audible_seconds_base: float | None = None
        # Start of the current stretch of continuous sidecar observation.
        self._observed_since: float | None = None
        # Room evidence older than this belongs to an earlier connection (or to
        # before the hold) and says nothing about the opening now in progress.
        self._evidence_from: float | None = None
        self._spoken: list[str] = []
        self._outcome = ""
        self._frames_withheld = 0
        self._speech_frames_withheld = 0
        self._reconnects = 0
        self._recall = 0.0
        # The span the model's transcript gives the spoken text, on its own
        # audio timeline, and whether any fragment came without one.
        self._span_start_ms: int | None = None
        self._span_end_ms: int | None = None
        self._span_missing = False
        self._span_unjudged = False
        # The last judgement's two sides, for the record and calibration.
        self._span_seconds = 0.0
        self._heard_seconds = 0.0
        # Output audio the provider has sent on this connection, in ms of its
        # own timeline; ``None`` until the first output event.
        self._output_ms: float | None = None
        self._output_short_ms = 0
        self._output_unjudged = False
        # The text's end_ms a released line still waits for the provider's
        # audio to reach; ``None`` once delivery is settled either way.
        self._delivery_target_ms: float | None = None
        self._push: Callable[[Any], None] | None = None
        self._tail: list[tuple[float, Any]] = []
        self._tail_edge: float | None = None
        self._tail_bytes = 0
        self._tail_seconds = 0.0
        self._tail_invalid = False
        self._tail_replayed = 0

    # -- wiring ------------------------------------------------------------

    def install(self, duplex_session: Any) -> bool:
        """Route the session's input through this hold. ``False`` if it cannot be.

        Needs the two things a duplex session has and a plain realtime session
        does not: a ``push_audio`` the adapter calls for every caller frame,
        and the ``transcript_delta`` event carrying the model's words. Wrapped
        on the INSTANCE, so the adapter's own ``self._duplex.push_audio(frame)``
        call reaches it and nothing else in the process is touched.
        """

        push = getattr(duplex_session, "push_audio", None)
        subscribe = getattr(duplex_session, "on", None)
        if self._installed or not callable(push) or not callable(subscribe):
            return False

        def held_push_audio(frame: Any) -> None:
            if self._held:
                self._retain_quiet_tail(frame)
                if _frame_rms(frame) >= SPEECH_FRAME_RMS:
                    self._speech_frames_withheld += 1
                try:
                    silent = self._silence(frame)
                except Exception:
                    # A frame this hold cannot copy is DROPPED, never passed
                    # through: forwarding it would let the far end reach the
                    # model in exactly the window this exists to close.
                    self._frames_withheld += 1
                    return
                self._frames_withheld += 1
                push(silent)
                return
            push(frame)

        try:
            subscribe("transcript_delta", self._on_transcript_delta)
            subscribe("session_reconnected", self._on_reconnected)
            # The plugin reports a dropped connection as a recoverable error
            # at once; the replacement's ``session_reconnected`` comes only
            # after the new socket is up, which can be longer than the quiet
            # tail. The drop is when the evidence stops being trustworthy.
            subscribe("error", self._on_connection_error)
            # The provider can also end the session itself (``session.closed``);
            # the socket close that follows then returns WITHOUT an error and
            # the plugin reconnects later, so this is the only early signal.
            subscribe("openai_server_event_received", self._on_server_event)
            duplex_session.push_audio = held_push_audio
        except Exception:
            return False
        self._installed = True
        self._push = push
        return True

    def _clear_tail(self) -> None:
        self._tail.clear()
        self._tail_bytes = 0
        self._tail_seconds = 0.0

    def _retain_quiet_tail(self, frame: Any) -> None:
        """Keep frames after the audible edge; resumed agent audio discards them.

        This is input delivery, not a speech/identity classifier. It admits
        the same post-opening audio the model would hear without a quiet tail.
        Caller speech over the actual greeting remains deliberately withheld.
        """
        edge = _finite_or_none(self._agent_audible_until)
        now = float(self._clock())
        if edge is None or self._evidence_from is None or edge <= self._evidence_from:
            return
        if self._tail_edge is None or edge > self._tail_edge:
            self._clear_tail()
            self._tail_edge = edge
            self._tail_invalid = False
        if self._tail_invalid:
            return
        try:
            duration = frame.samples_per_channel / frame.sample_rate
            size = memoryview(frame.data).nbytes
            start = now - duration
            if start < edge:
                return
            if (
                not math.isfinite(duration) or duration <= 0
                or size != frame.samples_per_channel * frame.num_channels * 2
                or now - edge > TAIL_SECONDS
                or self._tail_seconds + duration > TAIL_SECONDS
                # Arrival time is not capture time. A queued burst cannot
                # contain more audio than elapsed quiet/retention time, apart
                # from one frame of scheduling slack. Abandon the whole tail.
                or self._tail_seconds + duration > (
                    now - max(edge, self._tail[0][0] if self._tail else edge)
                    + duration + 1e-6
                )
                or self._tail_bytes + size > TAIL_BYTES
                or len(self._tail) >= TAIL_FRAMES
            ):
                raise ValueError("unbounded or malformed tail")
            # Own the samples: SDK frames may wrap storage reused by input.
            copy = type(frame)(data=bytes(frame.data), sample_rate=frame.sample_rate,
                               num_channels=frame.num_channels,
                               samples_per_channel=frame.samples_per_channel)
        except Exception:
            self._clear_tail()
            self._tail_invalid = True
            return
        self._tail.append((start, copy))
        self._tail_bytes += size
        self._tail_seconds += duration

    def _recover_quiet_tail(self, at: float, *, judged: bool) -> None:
        tail, self._tail = self._tail, []
        self._tail_bytes = 0
        self._tail_seconds = 0.0
        # Positive provider delivery and continuous room observation are both
        # required. Missing evidence releases normal input, but never replays.
        edge = _finite_or_none(self._agent_audible_until)
        observed_since = _finite_or_none(self._observed_since_source)
        if (
            not judged or self._outcome not in OPENING_DELIVERED
            or self._tail_invalid or not tail or self._push is None
            or edge is None or edge != self._tail_edge
            or at - edge > TAIL_SECONDS
            or at - self._observed_at(at) > OBSERVATION_GAP_SECONDS
            or (observed_since is not None and observed_since > edge)
            or self._span_missing or self._span_end_ms is None
            or self._output_ms is None or self._output_ms < self._span_end_ms
        ):
            return
        # Do not enqueue a second copy of a silent input interval. Retain a
        # short pre-roll before actual caller energy, rather than replaying
        # the entire quiet wait on every call. This is an energy threshold,
        # not speech or echo classification.
        first_energy = next((i for i, (_, frame) in enumerate(tail)
                             if _frame_rms(frame) >= SPEECH_FRAME_RMS), None)
        if first_energy is None:
            return
        first = first_energy
        pre_roll = 0.0
        while first > 0:
            previous = tail[first - 1][1]
            duration = previous.samples_per_channel / previous.sample_rate
            if pre_roll + duration > 0.2 + 1e-6:
                break
            pre_roll += duration
            first -= 1
        for _, frame in tail[first:]:
            try:
                self._push(frame)
                self._tail_replayed += 1
            except Exception:
                # A failed push is never retried: it may already have queued
                # part of the frame. Normal input is open and owns recovery.
                break

    def _on_server_event(self, event: Any) -> None:
        """Only the event TYPE is read: the raw events carry transcript text."""

        try:
            kind = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        except Exception:
            return
        if kind == "session.output_audio.delta":
            # Its LENGTH only, from the base64 text: the samples are not read.
            try:
                encoded = event.get("delta") if isinstance(event, dict) else None
                size = _base64_bytes(encoded) if isinstance(encoded, str) else 0
            except Exception:
                size = 0
            self._output_ms = (self._output_ms or 0.0) + size / 2 / OUTPUT_SAMPLE_RATE * 1000.0
            if (
                self._delivery_target_ms is not None
                and self._output_ms + OUTPUT_SLACK_MS >= self._delivery_target_ms
            ):
                self._settle_delivery()
        elif kind == "session.closed":
            self._restart_evidence()
        elif kind == "error":
            # A FATAL provider error ends the session, and the plugin reports
            # it only after awaiting the socket's close; this raw event is the
            # early signal. Only its code is read. A recoverable one leaves the
            # socket up and changes nothing.
            try:
                body = event.get("error") if isinstance(event, dict) else None
                code = (body.get("code") or body.get("type")) if isinstance(body, dict) else None
            except Exception:
                code = None
            if code in FATAL_PROVIDER_ERROR_CODES:
                self._restart_evidence()

    def _on_connection_error(self, event: Any = None) -> None:
        """The provider connection failed: what it streamed may never play.

        Treated exactly as a reconnect, because it is the start of one: text
        the model streamed ahead of its audio is no proof that the audio
        reached the caller, and the quiet that follows is the outage.

        Only a CONNECTION failure. The plugin reports a recoverable server
        error through the same event without dropping the socket, and
        restarting the evidence for one would throw away a greeting that
        just finished and keep the caller unheard for another backstop.
        """

        if getattr(event, "recoverable", True) is False or _is_connection_failure(
            getattr(event, "error", None)
        ):
            self._restart_evidence()

    def _on_reconnected(self, _event: Any = None) -> None:
        """A new provider connection restarts what the model is saying.

        Fragments of an utterance the old connection was still playing are
        not carried over by the adapter, so text already covered is no proof
        that the line reached the caller. Coverage starts again, and a line
        that is not re-covered ends on the backstop as ``incomplete``.
        """

        if self._restart_evidence():
            self._reconnects += 1

    def _restart_evidence(self) -> bool:
        # The output timeline is the CONNECTION's, whatever the hold is doing:
        # a line still waiting for its audio is judged on what the old
        # connection delivered, then a new count starts.
        self._clear_tail()
        self._tail_invalid = True
        self._tail_edge = None
        self._settle_delivery()
        self._output_ms = None
        if not self._held or self._outcome:
            return False
        self._spoken.clear()
        self._span_start_ms = None
        self._span_end_ms = None
        self._span_missing = False
        # The audio clock restarts too: the old connection's quiet tail is not
        # the new connection finishing anything.
        self._evidence_from = float(self._clock())
        self._first_audio_at = None
        self._last_audio_at = None
        self._audible_seconds = 0.0
        self._audible_seconds_base = self._read_audible_seconds()
        return True

    def _read_audible_seconds(self) -> float | None:
        if self._agent_audible_seconds is None:
            return None
        try:
            value = float(self._agent_audible_seconds())
        except Exception:
            return None
        return value if math.isfinite(value) else None

    def _on_transcript_delta(self, event: Any) -> None:
        if not self._held:
            return
        text = getattr(event, "text", None)
        if isinstance(text, str) and text:
            self._spoken.append(text)
            start = getattr(event, "start_ms", None)
            end = getattr(event, "end_ms", None)
            if isinstance(start, int) and isinstance(end, int) and end >= start:
                self._span_start_ms = start if self._span_start_ms is None else min(self._span_start_ms, start)
                self._span_end_ms = end if self._span_end_ms is None else max(self._span_end_ms, end)
            else:
                self._span_missing = True

    def _heard_the_span(self) -> bool:
        """Whether the room heard agent audio across the text's own span.

        The text streams ahead of its audio, so text alone says what the
        model MEANT to say. Its span on the model's audio timeline says how
        long that takes to play; the room's audible edges say how much of it
        played. Without a span on every fragment this cannot be judged, and
        the length rule stands alone (counted as ``span_unjudged``).
        """

        if self._span_missing or self._span_start_ms is None or self._span_end_ms is None:
            self._span_unjudged = True
            return True
        spoken_seconds = (self._span_end_ms - self._span_start_ms) / 1000.0
        if self._first_audio_at is None:
            return False
        heard_seconds = self._audible_time()
        self._span_seconds = spoken_seconds
        self._heard_seconds = heard_seconds
        return heard_seconds + SPAN_SLACK_SECONDS >= SPAN_COVERAGE_FRACTION * spoken_seconds

    # -- lifecycle ---------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def held(self) -> bool:
        return self._held

    @property
    def withholding(self) -> bool:
        """Whether the next caller frame bound for the model becomes silence.

        What a reader of the model's input (the liveness witness) needs to
        tell a deliberate substitution from a stalled or silent line.
        """

        return self._held

    @property
    def outcome(self) -> str:
        return self._outcome

    def engage(self) -> bool:
        """Start withholding. Call BEFORE the worker releases caller input.

        Refused when the session could not be wrapped, and recorded as such:
        a call whose model hears the far end during its opening is exactly the
        reported defect, and it must be countable rather than silent.
        """

        if self._engaged_at is not None:
            return self._held
        now = float(self._clock())
        self._engaged_at = now
        self._evidence_from = now
        self._audible_seconds_base = self._read_audible_seconds()
        if not self._installed:
            self._outcome = OUTCOME_UNSUPPORTED
            self._released_at = now
            self._log(now)
            return False
        self._held = True
        return True

    def poll(self, now: float | None = None) -> str:
        """Decide whether the opening is over; returns the outcome once it is."""

        if not self._held:
            return self._outcome
        at = float(self._clock() if now is None else now)
        self._read_room(at)
        engaged_at = self._engaged_at if self._engaged_at is not None else at
        if at - engaged_at >= ABSOLUTE_BACKSTOP_SECONDS:
            # Whatever the connection did in between -- reconnects restart the
            # clocks below -- the caller is never held longer than this.
            self.release(OUTCOME_INCOMPLETE, at=at)
            return self._outcome
        if self._first_audio_at is None:
            since = self._evidence_from if self._evidence_from is not None else engaged_at
            if at - since >= NO_AUDIO_BACKSTOP_SECONDS:
                self.release(OUTCOME_NO_AUDIO if self._reconnects == 0 else OUTCOME_INCOMPLETE, at=at)
            return self._outcome
        last_audio = self._last_audio_at or self._first_audio_at
        # Quiet is measured to the last moment the room was OBSERVED, not to
        # the wall clock: a sidecar that stopped delivering frames leaves the
        # agent edge frozen, and time it did not watch is not silence it saw.
        quiet_for = self._observed_quiet(at, last_audio)
        if self._literal:
            spoken = "".join(self._spoken)
            self._recall = opening_recall(self._opening, spoken)
            # Length first: recall alone crosses its threshold with the last
            # words of the line still to come.
            covered = len(_normalized(spoken)) >= COVERAGE_FRACTION * len(
                _normalized(self._opening)
            )
            if covered and quiet_for >= QUIET_SECONDS and self._heard_the_span():
                judged = OUTCOME_COVERED if self._recall >= MATCH_RECALL else OUTCOME_DIVERGENT
                self.release(judged, at=at, await_delivery=True)
                return self._outcome
        elif (
            quiet_for >= GENERATED_QUIET_SECONDS
            and self._heard_the_span()
            and self._audible_time() >= GENERATED_MIN_AUDIBLE_SECONDS
        ):
            self.release(OUTCOME_GENERATED, at=at, await_delivery=True)
            return self._outcome
        if at - self._first_audio_at >= backstop_seconds(self._opening):
            self.release(OUTCOME_INCOMPLETE, at=at)
        return self._outcome

    def _settle_delivery(self) -> None:
        """Judge a released line on the provider audio that has arrived.

        Settled when the audio reaches the text's end, when the connection is
        lost, or when the call ends -- never at the release itself, because
        the release comes after QUIET in the room, and the text runs ahead of
        its audio: the last words can still be on their way. Only the LABEL
        changes: a line whose audio stopped short is ``incomplete``. The
        release, and so what the caller hears, is never touched.
        """

        if self._delivery_target_ms is None:
            return
        short = self._delivery_target_ms - (self._output_ms or 0.0)
        self._delivery_target_ms = None
        self._output_short_ms = max(0, int(round(short)))
        if short > OUTPUT_SLACK_MS:
            self._outcome = OUTCOME_INCOMPLETE
        # The one line this hold writes, now that its outcome is final.
        self._log(self._released_at if self._released_at is not None else float(self._clock()))

    def _observed_quiet(self, at: float, last_audio: float) -> float:
        """Quiet since ``last_audio`` that the sidecar actually watched.

        Counted only across the current stretch of continuous observation: a
        delivery gap restarts it at the moment frames resume, so resuming
        does not credit the whole gap as observed quiet.
        """

        observed = self._observed_at(at)
        if at - observed > OBSERVATION_GAP_SECONDS:
            self._observed_since = None
            return 0.0
        if self._observed_since is None:
            self._observed_since = observed
        # The recorder marks a delivery gap when the next frame ARRIVES, so a
        # gap that ended between two polls is still seen; the poll-time check
        # above only covers a gap still open at the poll.
        resumed = _finite_or_none(self._observed_since_source)
        start = max(last_audio, self._observed_since, resumed if resumed is not None else last_audio)
        return min(at, observed) - start

    def _observed_at(self, at: float) -> float:
        try:
            observed = self._observed_until()
        except Exception:
            observed = None
        if observed is None:
            return float("-inf")
        try:
            value = float(observed)
        except (TypeError, ValueError):
            return float("-inf")
        return value if math.isfinite(value) else float("-inf")

    def _audible_time(self) -> float:
        """Seconds of agent sound since the hold (or the last reconnect).

        Measured from the recorder's per-frame counter when the worker wires
        it: a gap between two blips is not speech. The edge estimate below is
        the fallback for a caller that cannot supply one.
        """

        current = self._read_audible_seconds()
        if current is not None and self._audible_seconds_base is not None:
            return max(0.0, current - self._audible_seconds_base)
        return self._audible_seconds

    def _read_room(self, at: float) -> None:
        try:
            audible = self._agent_audible_until()
        except Exception:
            audible = None
        if audible is None or not math.isfinite(float(audible)):
            return
        audible = float(audible)
        # Audio the room heard before the hold began, or before a reconnect,
        # belongs to nothing here.
        if self._evidence_from is not None and audible <= self._evidence_from:
            return
        if self._first_audio_at is None:
            self._first_audio_at = audible
            self._last_audio_at = audible
            self._audible_seconds += POLL_SECONDS
            return
        if self._last_audio_at is None or audible > self._last_audio_at:
            # An advancing edge is audio since the last look; credited at most
            # one poll's worth, so a gap between two blips is not speech.
            advance = audible - (self._last_audio_at or audible)
            self._audible_seconds += min(advance, POLL_SECONDS * 1.5)
            self._last_audio_at = audible

    def release(self, outcome: str, *, at: float | None = None, await_delivery: bool = False) -> bool:
        """Let the model hear again. Idempotent; the first outcome stands.

        ``await_delivery`` (a line judged spoken) leaves the label open until
        the provider's audio has reached the text's end: see
        :meth:`_settle_delivery`. Without output events or a span it cannot be
        judged, and the outcome stands, counted as ``output_unjudged``.
        """

        if not self._held:
            return False
        when = float(self._clock() if at is None else at)
        self._held = False
        self._outcome = str(outcome)
        self._released_at = when
        self._recover_quiet_tail(when, judged=await_delivery)
        if await_delivery:
            if self._output_ms is None or self._span_end_ms is None or self._span_missing:
                self._output_unjudged = True
            else:
                self._delivery_target_ms = float(self._span_end_ms)
                if self._output_ms + OUTPUT_SLACK_MS >= self._delivery_target_ms:
                    self._settle_delivery()
                return True
        self._log(when)
        return True

    def end_of_call(self) -> None:
        """The call is ending with the hold still on: record it as that.

        Judged once more first, so an opening that finished inside the quiet
        tail -- the far end hung up a moment after it -- is recorded as what
        it was rather than as a hold the call ended.
        """

        self._clear_tail()
        self._tail_invalid = True
        self.poll()
        self.release(OUTCOME_CALL_ENDING)
        self._settle_delivery()

    async def run(self, *, interval: float = POLL_SECONDS) -> None:
        """Poll until released. The worker owns the task and cancels it at teardown."""

        while self._held:
            self.poll()
            if not self._held:
                return
            await asyncio.sleep(interval)

    # -- evidence ----------------------------------------------------------

    def opening_delivered(self) -> bool:
        return self._outcome in OPENING_DELIVERED

    def words_unobservable_until(self, now: float) -> float | None:
        """The end of the window in which the model could not caption the caller.

        ``None`` when this hold never withheld anything. While it is held the
        window is still open, so the answer is ``now``.
        """

        if self._engaged_at is None or self._outcome == OUTCOME_UNSUPPORTED:
            return None
        if self._held:
            return float(now)
        return self._released_at

    @property
    def stats(self) -> dict[str, Any]:
        """Bounded scalars for the interruption summary. No text of any kind."""

        held_ms = 0
        if self._engaged_at is not None:
            end = self._released_at if self._released_at is not None else self._clock()
            held_ms = max(0, int(round((float(end) - self._engaged_at) * 1000)))
        return {
            "duplex_opening_hold_ms": held_ms,
            "duplex_opening_frames_withheld": self._frames_withheld,
            "duplex_opening_speech_frames_withheld": self._speech_frames_withheld,
            "duplex_opening_reconnects": self._reconnects,
            "duplex_opening_span_unjudged": int(self._span_unjudged),
            "duplex_opening_output_short_ms": self._output_short_ms,
            "duplex_opening_output_unjudged": int(self._output_unjudged),
            "duplex_opening_tail_frames_replayed": self._tail_replayed,
        }

    def _log(self, at: float) -> None:
        details = dict(self.stats)
        details["outcome"] = self._outcome
        details["recall"] = round(self._recall, 3)
        details["literal"] = self._literal
        details["span_seconds"] = round(self._span_seconds, 3)
        details["heard_audible_seconds"] = round(self._heard_seconds, 3)
        details["opening_characters"] = len(self._opening)
        details["spoken_characters"] = len("".join(self._spoken))
        if self._first_audio_at is not None and self._engaged_at is not None:
            details["first_audio_ms"] = int(round((self._first_audio_at - self._engaged_at) * 1000))
        try:
            self._emit(EVENT, **details)
        except Exception:
            pass


def _base64_bytes(encoded: str) -> int:
    """Decoded length of a base64 string, without decoding it."""

    text = encoded.strip()
    if not text:
        return 0
    return len(text) * 3 // 4 - text[-2:].count("=")


__all__ = [
    "OUTPUT_SAMPLE_RATE",
    "OUTPUT_SLACK_MS",
    "SPAN_COVERAGE_FRACTION",
    "SPAN_SLACK_SECONDS",
    "FATAL_PROVIDER_ERROR_CODES",
    "OBSERVATION_GAP_SECONDS",
    "ABSOLUTE_BACKSTOP_SECONDS",
    "BACKSTOP_BASE_SECONDS",
    "BACKSTOP_MAX_SECONDS",
    "BACKSTOP_SECONDS_PER_CHARACTER",
    "COVERAGE_FRACTION",
    "DuplexOpeningHold",
    "EVENT",
    "GENERATED_MIN_AUDIBLE_SECONDS",
    "JUDGED_OUTCOMES",
    "GENERATED_QUIET_SECONDS",
    "MATCH_RECALL",
    "NO_AUDIO_BACKSTOP_SECONDS",
    "OPENING_DELIVERED",
    "OUTCOMES",
    "OUTCOME_CALL_ENDING",
    "OUTCOME_DIVERGENT",
    "OUTCOME_GENERATED",
    "OUTCOME_INCOMPLETE",
    "OUTCOME_NO_AUDIO",
    "OUTCOME_UNSUPPORTED",
    "OUTCOME_COVERED",
    "QUIET_SECONDS",
    "SPEECH_FRAME_RMS",
    "backstop_seconds",
    "opening_recall",
    "silent_copy",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Short acknowledgements that cover the gap before a reply starts.

Spoken on the separately published background track (LiveKit's
``BackgroundAudioPlayer``), never through ``session.say``: the pinned SDK
schedules the reply's own speech handle before it reports "thinking", so a
phrase queued on the session would play AFTER the answer it was meant to
cover. The background track is mixed in independently, starts the moment it is
asked to, and is stopped the moment the real reply begins. The evidence sidecar
deliberately does not record that track, so an acknowledgement never becomes a
transcript line or an agent interval; the bundle's audio policy states whether
the feature was enabled for the call.

The feature is a customer choice per agent (``voice_polish.acknowledgements``
on the voice stack), off unless enabled, and an operator can still switch it
off for a whole deployment. Phrases come from the customer, bounded, or from
the reviewed list below for the call's language; a language with neither stays
silent rather than acknowledging in English.

A caller who stops speaking currently hears nothing until the model and the
voice have both produced something, which measured around three seconds at the
worker. A human receptionist fills that with "okay" or "let me check". This
module does the same, under rules that decide when NOT to.

Three properties are not negotiable and are what most of this file is about:

* It never speaks over the caller, and it never speaks once the real reply has
  started. The reply's own start is the cancel signal, not a guess about how
  long a reply takes, so a fast turn simply never arms long enough to fire.
* It is never mistaken for the agent's answer. The acknowledgement is spoken
  outside the chat context, so the model does not read it back as its own
  previous utterance, it never becomes a transcript line, and it never pairs
  with a caller turn as a measured reply time.
* It is language-correct or absent. A language with no reviewed phrase list
  gets no acknowledgements at all rather than an English one.

The policy here is pure and synchronous so every rule is testable without a
LiveKit session, an event loop, or a provider. ``AcknowledgementScheduler``
holds the only timing state and takes its clock and its speak action by
injection.
"""

from __future__ import annotations

import array
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence


#: Reviewed short acknowledgements per language.
#:
#: A language absent from this mapping turns the feature off for that call.
#: That is deliberate: an acknowledgement in the wrong language is worse than
#: silence, and "we support this language" is a claim that needs a speaker of
#: it, not a translation guess made at the point of use.
#:
#: Estonian is absent on purpose even though the product speaks it. Its
#: end-of-turn detection falls back to voice-activity silence because the
#: pinned turn detector does not claim the language, so "the caller has
#: finished" is a less certain judgement there -- exactly the condition under
#: which an acknowledgement turns into an interruption. Its only available
#: voice is also the slowest to start speaking, so the acknowledgement itself
#: would arrive late enough to collide with the reply it is covering.
ACKNOWLEDGEMENTS: Mapping[str, tuple[str, ...]] = {
    "en": ("Okay.", "Let me check.", "One moment.", "Sure, one second."),
}

#: How long the reply may take before an acknowledgement is worth speaking.
#:
#: Below this the caller has not yet registered silence, and preemptive
#: generation means a fast turn often starts speaking within a few hundred
#: milliseconds -- an acknowledgement in front of one of those is worse than
#: the gap it claims to fill.
DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS = 1.2

#: Turns that must pass after one acknowledgement before another may be
#: spoken. Every mature implementation of this rate-limits, because the
#: failure mode of a correct acknowledgement repeated every turn is that it
#: stops sounding like listening and starts sounding like a tic.
DEFAULT_ACKNOWLEDGEMENT_COOLDOWN_TURNS = 2

#: Hard ceiling per call, independent of the cooldown.
DEFAULT_MAX_ACKNOWLEDGEMENTS_PER_CALL = 6


#: Operator kill switch for a whole deployment. The customer enables the
#: feature per agent; this refuses it everywhere regardless, for the case where
#: a live defect has to be stopped faster than agents can be reconfigured.
ACKNOWLEDGEMENTS_DISABLED_VAR = "SKYSAY_ACKNOWLEDGEMENTS_DISABLED"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})

#: Bounds mirrored from the control plane's ``voice_polish.acknowledgements``
#: contract. The worker re-checks them: a value the API never accepted must not
#: become behaviour because it reached the realtime config another way.
MAX_PHRASE_CHARS = 40
MAX_PHRASES = 6
DELAY_BOUNDS_SECONDS = (0.6, 3.0)


#: Words that turn a filler into a claim. English only, matched as whole words
#: after lowercasing; other languages rely on the length bounds and on the
#: customer's own review.  Kept equal to the control plane's list by test.
CLAIM_WORDS = frozenset(
    {
        "paid",
        "payment",
        "received",
        "refund",
        "refunded",
        "charge",
        "charged",
        "consent",
        "agree",
        "agreed",
        "approved",
        "confirmed",
        "confirm",
        "cancelled",
        "canceled",
        "balance",
        "due",
        "owe",
        "owed",
        "price",
        "cost",
        "fee",
        "invoice",
        "contract",
        "legal",
        "guarantee",
        "guaranteed",
    }
)
MAX_PHRASE_WORDS = 5


def neutral_phrase_problem(text: str) -> str | None:
    """Why ``text`` may not be spoken as an acknowledgement, or ``None``.

    Bounded neutral phrases only: short, few words, no digits in any script,
    no currency marks, no English claim word.  A filler that can state an
    amount, a date or an agreement is an answer the agent never meant to give.
    """

    import unicodedata

    if not text or len(text) > MAX_PHRASE_CHARS:
        return "length"
    words = text.split()
    if len(words) > MAX_PHRASE_WORDS:
        return "too_many_words"
    for char in text:
        if char.isdigit():
            return "digit"
        if unicodedata.category(char) == "Sc":
            return "currency"
    for word in words:
        if word.strip(".,;:!?\"'()").lower() in CLAIM_WORDS:
            return "claim_word"
    return None


def acknowledgements_disabled_here(environ: Mapping[str, str]) -> bool:
    """Whether the deployment refuses acknowledgements for every call."""
    return (
        str(environ.get(ACKNOWLEDGEMENTS_DISABLED_VAR, "") or "").strip().lower()
        in _TRUE_VALUES
    )


def acknowledgements_for(language: Any) -> tuple[str, ...]:
    """Return the reviewed phrases for a language, or none if it has none."""
    key = str(language or "").strip().lower()
    # An agent configured as "en-US" is still English; the phrase list is keyed
    # by the base language, never by the regional variant.
    base = key.split("-")[0].split("_")[0]
    return ACKNOWLEDGEMENTS.get(base, ())


@dataclass(frozen=True)
class AcknowledgementSettings:
    """What this call may acknowledge with, resolved from its realtime config."""

    enabled: bool
    phrases: tuple[str, ...]
    delay_seconds: float
    #: Why ``enabled`` is False when the customer asked for acknowledgements.
    reason: str = ""


def acknowledgement_settings(
    realtime_config: Mapping[str, Any] | None,
    *,
    language: Any,
    environ: Mapping[str, str] | None = None,
) -> AcknowledgementSettings:
    """Resolve the customer's ``voice_polish.acknowledgements`` for one call.

    Off unless the section says ``enabled``. Customer phrases win when they
    pass the same bounds the API enforces; otherwise the reviewed list for the
    call's language applies, and a language with neither is reported as
    ``no_phrases_for_language`` and stays silent -- never English.
    """

    polish = (
        (realtime_config or {}).get("voice_polish")
        if isinstance(realtime_config, Mapping)
        else None
    )
    section = polish.get("acknowledgements") if isinstance(polish, Mapping) else None
    if not isinstance(section, Mapping) or section.get("enabled") is not True:
        return AcknowledgementSettings(
            False, (), DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS, ""
        )
    if acknowledgements_disabled_here(environ if environ is not None else {}):
        return AcknowledgementSettings(
            False, (), DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS, "disabled_by_operator"
        )
    phrases: list[str] = []
    raw_phrases = section.get("phrases")
    if isinstance(raw_phrases, (list, tuple)):
        for raw in raw_phrases[:MAX_PHRASES]:
            if not isinstance(raw, str):
                continue
            text = " ".join(raw.split())
            if text and neutral_phrase_problem(text) is None and text not in phrases:
                phrases.append(text)
    if not phrases:
        phrases = list(acknowledgements_for(language))
    delay = DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS
    raw_delay = section.get("delay_ms")
    if isinstance(raw_delay, (int, float)) and not isinstance(raw_delay, bool):
        low, high = DELAY_BOUNDS_SECONDS
        candidate = float(raw_delay) / 1000.0
        if low <= candidate <= high:
            delay = candidate
    if not phrases:
        return AcknowledgementSettings(False, (), delay, "no_phrases_for_language")
    return AcknowledgementSettings(True, tuple(phrases), delay, "")


class BackgroundSpeech:
    """The playing acknowledgement, with the two verbs the scheduler needs.

    ``interrupt`` stops the mixer input AND flushes what the player had already
    queued towards the room.  The pinned player buffers 400 ms in its audio
    source and ``PlayHandle.stop`` deliberately leaves that buffer alone (so a
    stopped ambience loop tails out instead of clicking); for an acknowledgement
    that buffer is exactly the audio that would play over the reply or the
    caller, so it is discarded.  The player is dedicated to acknowledgements,
    never shared with ambience, precisely so this flush cannot cut anything else.

    What remains after the flush is bounded by the player's own mixer: one
    100 ms block already mixed plus the block being captured, so a cut filler
    can still be heard for up to ~200 ms.  Frames already handed to the room
    cannot be recalled; this is the floor of what any interruption can reach.
    """

    def __init__(self, handle: Any, flush: Callable[[], None] | None = None) -> None:
        self._handle = handle
        self._flush = flush

    def done(self) -> bool:
        try:
            return bool(self._handle.done())
        except Exception:  # noqa: BLE001 - a foreign handle must not break the scheduler
            return True

    def interrupt(self) -> None:
        self._handle.stop()
        if self._flush is not None:
            self._flush()


def flush_background_player(player: Any) -> None:
    """Discard audio the dedicated player has queued but not yet sent.

    Reaches the pinned ``BackgroundAudioPlayer``'s own ``rtc.AudioSource``.
    The coupling to that private attribute is asserted by an offline probe in
    the worker image's Dockerfile (it constructs the player and requires
    ``_audio_source.clear_queue``), so an SDK bump that moves it fails the
    image build instead of leaving every cut filler unflushed.
    """

    source = getattr(player, "_audio_source", None)
    clear = getattr(source, "clear_queue", None)
    if callable(clear):
        clear()


def background_speaker(
    player: Any, clips: Mapping[str, Sequence[Any]]
) -> Callable[[str], Any]:
    """Speak an acknowledgement on the dedicated background track from a clip.

    ``clips`` maps each phrase to its already-synthesized 48 kHz mono frames
    (see :func:`synthesize_acknowledgement_clips`). A phrase with no clip yet
    -- synthesis still running, or failed -- is skipped rather than spoken in
    another voice; the scheduler's ``acknowledgement_failed`` telemetry names
    it. Nothing here touches the session's speech queue or chat context.
    """

    def speak(phrase: str) -> Any:
        frames = clips.get(phrase)
        if not frames:
            raise LookupError("acknowledgement clip is not ready")

        async def frame_source() -> Any:
            for frame in frames:
                yield frame

        return BackgroundSpeech(
            player.play(frame_source()), flush=lambda: flush_background_player(player)
        )

    return speak


#: Bound on preparing every clip for one call: synthesis runs in the
#: background after the greeting and a slow provider only delays the first
#: acknowledgement, never the call.
CLIP_SYNTHESIS_TIMEOUT_SECONDS = 20.0
BACKGROUND_TRACK_SAMPLE_RATE = 48_000
#: What a synthesized word may keep of the silence around it, each side. A spoken digit is a word, and a TTS provider pads it
#: (measured: 1.2 s of trailing silence on every digit of one agent's voice, so eight digits took 15 s and a far end that
#: ends an utterance after 1.5 s of quiet heard two of them). The pacing between words is the speaker's own.
TRIM_KEEP_SECONDS = 0.04
#: A sample counts as sound above this share of the clip's peak (about -46 dB: a soft consonant at the edge of a loud
#: word stays) and above an absolute floor, for a clip of near-silence.
TRIM_PEAK_SHARE = 0.005
TRIM_FLOOR = 64


def trim_silence(pcm: bytes, rate: int, *, keep_seconds: float = TRIM_KEEP_SECONDS) -> bytes:
    """16-bit mono PCM without the silence at either end (``keep_seconds`` of it stays). A clip with no sound is returned as it is."""

    if len(pcm) < 4 or rate <= 0:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    peak = max((abs(value) for value in samples), default=0)
    threshold = max(TRIM_FLOOR, int(peak * TRIM_PEAK_SHARE))
    if peak < threshold:
        return pcm
    first = next((index for index, value in enumerate(samples) if abs(value) >= threshold), None)
    last = next((index for index in range(len(samples) - 1, -1, -1) if abs(samples[index]) >= threshold), None)
    if first is None or last is None:
        return pcm
    keep = int(keep_seconds * rate)
    start, end = max(0, first - keep), min(len(samples), last + 1 + keep)
    return samples[start:end].tobytes()


async def synthesize_acknowledgement_clips(
    tts: Any,
    phrases: Sequence[str],
    *,
    resampler_factory: Callable[[int], Any],
    frame_factory: Callable[[bytes, int, int], Any],
    timeout_seconds: float = CLIP_SYNTHESIS_TIMEOUT_SECONDS,
    connect_options: Any = None,
    trim: bool = False,
) -> dict[str, list[Any]]:
    """Synthesize each phrase once with the call's OWN voice and resample it.

    ``trim`` cuts the silence a provider pads a word with (see :func:`trim_silence`): for words the call says one after
    another (a spoken value), where the padding would become the pauses.

    Uses ``tts.synthesize`` -- the same adapter, voice, model and metering the
    call's replies use, so the characters are billed to this call and the
    phrase sounds like the agent. Frames are resampled to the background
    track's 48 kHz mono. A phrase whose synthesis fails is simply absent; the
    speaker then skips it.
    """

    clips: dict[str, list[Any]] = {}

    async def one(phrase: str) -> None:
        pcm = bytearray()
        rate: int | None = None
        # One attempt, no adapter retries: a refused phrase is skipped, not
        # retried three times.  This runs on the CALL'S OWN adapter so the
        # characters are metered like every reply -- which also means a final
        # failure is reported to the session as one unrecoverable error.  The
        # batch therefore stops at the first failure: a phone session's budget
        # absorbs three consecutive errors and resets on the next reply, so one
        # refused phrase can never end a call.
        async with tts.synthesize(phrase, conn_options=connect_options) as stream:
            async for event in stream:
                frame = event.frame
                if rate is None:
                    rate = int(frame.sample_rate)
                pcm.extend(bytes(frame.data))
        if rate is None or not pcm:
            return
        if trim:
            pcm = bytearray(trim_silence(bytes(pcm), rate))
        if rate == BACKGROUND_TRACK_SAMPLE_RATE:
            clips[phrase] = [frame_factory(bytes(pcm), rate, len(pcm) // 2)]
            return
        resampler = resampler_factory(rate)
        out: list[Any] = []
        out.extend(resampler.push(frame_factory(bytes(pcm), rate, len(pcm) // 2)))
        out.extend(resampler.flush())
        if out:
            clips[phrase] = out

    async def all_phrases() -> None:
        for phrase in phrases:
            try:
                await one(phrase)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one missing clip is not a broken call
                return

    try:
        await asyncio.wait_for(all_phrases(), timeout=timeout_seconds)
    except TimeoutError:
        pass
    return clips


class AcknowledgementPolicy:
    """Decide whether an acknowledgement may be spoken, and which one.

    Pure and synchronous. It owns every rule that is not about time passing:
    language support, once per turn, the cooldown between turns, the per-call
    ceiling, and never acknowledging before the caller has said anything.
    """

    def __init__(
        self,
        *,
        language: Any,
        phrases: Sequence[str] | None = None,
        cooldown_turns: int = DEFAULT_ACKNOWLEDGEMENT_COOLDOWN_TURNS,
        max_per_call: int = DEFAULT_MAX_ACKNOWLEDGEMENTS_PER_CALL,
        enabled: bool = True,
        is_available: Callable[[str], bool] | None = None,
    ) -> None:
        self._enabled_here = bool(enabled)
        # Whether a phrase can be spoken right now (its clip exists).  The
        # rotation skips phrases that cannot, so one phrase a provider refused
        # never blocks the ones it did render.
        self._is_available = is_available
        self._phrases = (
            tuple(phrases) if phrases is not None else acknowledgements_for(language)
        )
        self._cooldown_turns = max(0, int(cooldown_turns))
        self._max_per_call = max(0, int(max_per_call))
        self._turn = 0
        self._spoken = 0
        self._last_spoken_turn: int | None = None
        self._spoken_this_turn = False

    @property
    def enabled(self) -> bool:
        """False when this call can never acknowledge, so callers can skip the wiring."""
        return self._enabled_here and bool(self._phrases) and self._max_per_call > 0

    @property
    def spoken(self) -> int:
        return self._spoken

    @property
    def spoke_this_turn(self) -> bool:
        """Whether the current caller turn has already heard a filler."""
        return self._spoken_this_turn

    def mark_spoken_this_turn(self) -> None:
        """Record that another feature spoke a filler in this turn.

        The tool-wait phrase is a filler too; one per turn is the rule across
        both, so this turn is closed and the cooldown starts from it.  It does
        not count towards this policy's own per-call ceiling.
        """
        self._spoken_this_turn = True
        self._last_spoken_turn = self._turn

    def begin_caller_turn(self) -> None:
        """The caller started speaking: a new turn, and a fresh chance to acknowledge."""
        self._turn += 1
        self._spoken_this_turn = False

    def may_acknowledge(self) -> bool:
        """Whether the CURRENT turn is still eligible, before any waiting begins.

        Checked twice by design -- once to decide whether waiting is worth
        starting, and again after the wait, because a caller can start speaking
        or a reply can begin during it.
        """
        if not self.enabled:
            return False
        # Nothing to acknowledge before the caller has spoken at all: the
        # opening greeting is not a reply to anything.
        if self._turn < 1:
            return False
        if self._spoken_this_turn:
            return False
        if self._spoken >= self._max_per_call:
            return False
        if self._last_spoken_turn is not None:
            if self._turn - self._last_spoken_turn <= self._cooldown_turns:
                return False
        return True

    def _next_available(self) -> str | None:
        for offset in range(len(self._phrases)):
            phrase = self._phrases[(self._spoken + offset) % len(self._phrases)]
            if self._is_available is None or self._is_available(phrase):
                return phrase
        return None

    def peek(self) -> str | None:
        """The phrase ``take`` would claim now, without claiming it."""
        if not self.may_acknowledge():
            return None
        return self._next_available()

    def take(self) -> str | None:
        """Claim one acknowledgement for this turn, or None if it is not allowed.

        Rotates through the phrase list so a caller who triggers several in one
        call does not hear the same word each time.
        """
        if not self.may_acknowledge():
            return None
        phrase = self._next_available()
        if phrase is None:
            return None
        self._spoken += 1
        self._spoken_this_turn = True
        self._last_spoken_turn = self._turn
        return phrase


class AcknowledgementScheduler:
    """Speak one acknowledgement when a reply is slow, and never otherwise.

    Drive it from the session's state events. It arms when the agent starts
    working on a reply and cancels the moment anything makes speaking wrong:
    the reply begins, the caller begins, or the turn is abandoned. The wait
    itself is the only asynchronous part, and its clock is injected.
    """

    def __init__(
        self,
        *,
        policy: AcknowledgementPolicy,
        speak: Callable[[str], Any],
        delay_seconds: float = DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        emit: Callable[..., None] | None = None,
        speech_on_session: bool = False,
        suppressed: Callable[[], bool] | None = None,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("acknowledgement delay must be non-negative")
        self._policy = policy
        # Read at BOTH decision points (arming, and the instant before the
        # phrase is spoken): "one moment" said into a phone menu or a hold queue
        # is the agent talking to a recording. A reader that raises suppresses.
        self._suppressed = suppressed
        self._speak = speak
        # True only for a speaker that goes through the session's own speech
        # queue, whose "speaking" edge must then be claimed as the filler's.
        # The background-track speaker never moves the agent state, so claiming
        # the next edge would misattribute the REAL reply to the filler.
        self._speech_on_session = bool(speech_on_session)
        # The SDK reports "thinking" before it has waited for the caller to
        # fall silent, so a filler armed on that edge alone could play over a
        # caller who is still talking.  Track the caller's state and require
        # silence both when arming and when firing.
        self._caller_speaking = False
        # The SDK emits one "thinking" edge per reply; if the caller was still
        # talking at that edge, the caller's return to silence is the moment
        # to arm instead.
        self._agent_thinking = False
        self._delay_seconds = float(delay_seconds)
        self._sleep = sleep or asyncio.sleep
        self._emit = emit
        self._pending: asyncio.Task[None] | None = None
        # The handle for an acknowledgement that is already queued or playing.
        # Cancelling the wait cannot recall speech that has been handed to the
        # session, so the handle is what makes 'stop' still mean stop once the
        # phrase is out.
        self._speaking: Any | None = None
        # An acknowledgement puts the agent into "speaking" itself. Without
        # this the scheduler would read its own filler as the real reply --
        # cancelling and interrupting the phrase at its first audio frame, so
        # every filler it ever fired would be cut off instantly.
        self._expect_own_speech = False
        # True between the filler's own "into speaking" and "out of speaking"
        # transitions. BOTH ends are claimed: handing the recorder the end of
        # an interval whose beginning it never saw is its own kind of
        # corruption, so the filler is invisible to it rather than half-seen.
        self._in_own_speech = False
        self._closed = False
        # True while a tool window is open: the tool-wait feedback owns that
        # silence, so nothing arms here until it hands the turn back.
        self._suspended = False
        # Bumped by everything that means "do not speak after all". A wait that
        # has already returned cannot be un-run by cancelling it, and the
        # policy alone cannot tell: the caller starting to speak opens a NEW
        # eligible turn, so a superseded wait would find itself allowed and
        # speak straight over them. The epoch is what makes "superseded"
        # decidable without depending on when a cancellation is delivered.
        self._epoch = 0

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    @property
    def spoke_this_turn(self) -> bool:
        return self._policy.spoke_this_turn

    @property
    def speaking(self) -> bool:
        """Whether an acknowledgement is audible right now.

        Background speech never moves the agent state, so this is the only way
        another filler can avoid mixing itself under a phrase in progress.
        """
        speaking = self._speaking
        if speaking is None:
            return False
        try:
            return not bool(speaking.done())
        except Exception:  # noqa: BLE001 - a foreign handle must not break the caller
            return False

    def suspend(self, reason: str) -> None:
        """Yield the wait to another filler (a tool window opening).

        Cancels a PENDING acknowledgement and blocks arming until ``resume``.
        A phrase already speaking is left to finish: cutting "One mo--" for a
        tool that started a moment later is worse than the phrase, and the
        reply's own start still cuts it like always.
        """
        self._suspended = True
        self._epoch += 1
        pending = self._pending
        self._pending = None
        if pending is not None and not pending.done():
            pending.cancel()
            self._notify("acknowledgement_cancelled", reason=reason)

    def resume(self, *, acknowledged: bool = False) -> None:
        """Take the wait back once the tool window has closed.

        ``acknowledged`` says the window itself spoke a filler, which closes
        this turn here too.  Otherwise, if the reply is still being prepared,
        the wait starts over from now -- a fast tool must not have silenced the
        acknowledgement for the slow reply that follows it.
        """
        self._suspended = False
        if acknowledged:
            self._policy.mark_spoken_this_turn()
            return
        if self._agent_thinking:
            self._arm()

    # Both event entry points are driven straight from the media session's
    # callbacks, so neither may raise: an acknowledgement is a nicety and a
    # call is not. Failures are reported and swallowed, the same rule the
    # worker's telemetry callbacks already follow.

    def on_user_state_changed(self, old_state: Any, new_state: Any) -> None:
        try:
            self._caller_speaking = str(new_state) == "speaking"
            if self._caller_speaking:
                # Both halves matter: a new caller turn begins, and any
                # acknowledgement waiting to fill the previous one must not
                # land on top of the caller who just started talking.
                self._cancel("caller_speaking")
                self._policy.begin_caller_turn()
            elif self._agent_thinking:
                # The reply was already being prepared when the caller was
                # still talking; now that they are silent the wait can start.
                self._arm()
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("acknowledgement_failed", error=type(exc).__name__)

    def on_agent_state_changed(self, old_state: Any, new_state: Any) -> bool:
        """Handle one agent state change.

        Returns True when the transition belongs to an acknowledgement rather
        than to the conversation. The caller uses that to keep filler audio
        out of the semantic turn evidence, which pairs the first agent speech
        interval after the caller stops with the assistant's text -- and an
        acknowledgement creates no assistant text at all, so letting its
        interval in would attach the real reply's words to the filler's audio.
        """
        try:
            state = str(new_state)
            # Checked before anything else: the filler leaving "speaking" is
            # its own transition whatever it leaves for, and arming a fresh
            # wait off the back of it would schedule a second acknowledgement
            # inside one turn.
            if self._in_own_speech and str(old_state) == "speaking":
                self._in_own_speech = False
                self._speaking = None
                return True
            self._agent_thinking = state == "thinking"
            if state == "thinking":
                self._arm()
                return False
            if state == "speaking" and self._expect_own_speech:
                # This is the acknowledgement we just handed to the session,
                # not the reply it is covering.
                self._expect_own_speech = False
                self._in_own_speech = True
                return True
            # "speaking" is now genuinely the real reply starting, which is the
            # signal this whole feature waits on. Anything else means the turn
            # went away.
            self._cancel("reply_started" if state == "speaking" else "turn_ended")
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("acknowledgement_failed", error=type(exc).__name__)
        return False

    def stop(self) -> None:
        """Stop acknowledging for the rest of the call, without closing.

        Used when the call is on its way out: nothing may be scheduled in
        front of a goodbye.
        """
        self._closed = True
        self._cancel("stopped")

    def _is_suppressed(self) -> bool:
        if self._suppressed is None:
            return False
        try:
            return bool(self._suppressed())
        except Exception:  # noqa: BLE001 - an unreadable state is not a licence to speak
            return True

    def _arm(self) -> None:
        if (
            self._closed
            or self._suspended
            or self._pending is not None
            or self._caller_speaking
            or self._is_suppressed()
        ):
            return
        if not self._policy.may_acknowledge():
            return
        # The loop is looked up before the coroutine is built, not after:
        # there always is one here (these are loop callbacks), but building it
        # first and failing to schedule it leaves an un-awaited coroutine
        # behind every time -- a leak reported as a warning, in a process that
        # runs for the length of a call.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            self._notify("acknowledgement_failed", error=type(exc).__name__)
            return
        self._pending = loop.create_task(
            self._wait_then_speak(self._epoch), name="AcknowledgementScheduler._wait"
        )

    def _cancel(self, reason: str) -> None:
        self._epoch += 1
        pending = self._pending
        self._pending = None
        if pending is not None and not pending.done():
            pending.cancel()
            self._notify("acknowledgement_cancelled", reason=reason)
        self._interrupt_speech(reason)

    def _interrupt_speech(self, reason: str) -> None:
        """Cut an acknowledgement that is already speaking.

        Without this the reply waits behind "let me check" instead of
        replacing it -- the acknowledgement would have made the answer arrive
        LATER, which is the opposite of the point. The handle is interruptible
        by construction (see ``BackgroundSpeech.interrupt``).
        """
        speaking = self._speaking
        self._speaking = None
        self._expect_own_speech = False
        # _in_own_speech is deliberately NOT cleared here. Interrupting the
        # phrase does not stop the session reporting that it left "speaking",
        # and that report still belongs to the filler -- dropping the claim
        # now would hand the recorder the end of an interval it never saw
        # begin, which is the thing this whole classification exists to avoid.
        if speaking is None:
            return
        try:
            # Unconditionally: a background handle reports ``done()`` as soon
            # as its generator is drained, while up to half a second of its
            # audio is still queued towards the room.  ``interrupt`` is
            # idempotent on a finished handle and is the only path that
            # flushes that queue.
            speaking.interrupt()
            self._notify("acknowledgement_interrupted", reason=reason)
        except Exception as exc:  # noqa: BLE001 - never break a media callback
            self._notify("acknowledgement_failed", error=type(exc).__name__)

    async def _wait_then_speak(self, epoch: int) -> None:
        try:
            await self._sleep(self._delay_seconds)
        except asyncio.CancelledError:
            raise
        # Checked before the policy, not after: by now the caller may have
        # started speaking, which both cancels this wait AND opens a fresh
        # eligible turn, so asking the policy first would get a yes for the
        # wrong turn. Anything that superseded this wait has already moved the
        # epoch on, whether or not the cancellation reached the sleep.
        if epoch != self._epoch or self._closed or self._caller_speaking or self._is_suppressed():
            self._pending = None
            return
        phrase = self._policy.peek()
        if phrase is None:
            self._pending = None
            return
        try:
            # Claim the turn only once the speaker has accepted the phrase: a
            # clip that is not ready yet must not consume the per-call ceiling
            # or start the cooldown, or a slow provider silences the whole call.
            result = self._speak(phrase)
            self._policy.take()
            if asyncio.iscoroutine(result):
                await result
            elif result is not None and hasattr(result, "interrupt"):
                # A speech handle, not a coroutine: keep it so the reply can
                # cut it short rather than queue behind it.
                self._speaking = result
                if self._speech_on_session:
                    # The session is about to report "speaking" because of
                    # this phrase. Claim that transition in advance.
                    self._expect_own_speech = True
            self._notify("acknowledgement_spoken")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a filler must never break a call
            self._notify("acknowledgement_failed", error=type(exc).__name__)
        finally:
            self._pending = None

    def _notify(self, event: str, **details: Any) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event, **details)
        except Exception:
            pass

    async def aclose(self) -> None:
        self._closed = True
        self._epoch += 1
        self._interrupt_speech("closed")
        pending = self._pending
        self._pending = None
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                # The wait was cancelled just above, so its CancelledError is
                # the expected way for it to settle.  A pending request on the
                # CLOSING task is the worker job being cancelled, and that one
                # keeps propagating; swallowing it here would end the job as
                # if it had completed.
                closer = asyncio.current_task()
                if closer is not None and closer.cancelling() > 0:
                    raise
            except Exception:  # noqa: BLE001 - a filler must never break a call
                pass


__all__ = [
    "ACKNOWLEDGEMENTS",
    "ACKNOWLEDGEMENTS_DISABLED_VAR",
    "AcknowledgementSettings",
    "BackgroundSpeech",
    "CLAIM_WORDS",
    "MAX_PHRASE_WORDS",
    "acknowledgement_settings",
    "neutral_phrase_problem",
    "acknowledgements_disabled_here",
    "background_speaker",
    "flush_background_player",
    "synthesize_acknowledgement_clips",
    "DEFAULT_ACKNOWLEDGEMENT_COOLDOWN_TURNS",
    "DEFAULT_ACKNOWLEDGEMENT_DELAY_SECONDS",
    "DEFAULT_MAX_ACKNOWLEDGEMENTS_PER_CALL",
    "AcknowledgementPolicy",
    "AcknowledgementScheduler",
    "acknowledgements_for",
]

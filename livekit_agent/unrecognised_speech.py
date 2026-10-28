# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One short line when the caller spoke and the transcriber heard nothing.

The defect, seen on a production call and then reproduced on demand by the
local call harness on 2026-09-22: a caller speaks German to an
agent whose transcriber is pinned to English. Deepgram produces no final, the
worker's bounded wait expires with ``recovery=provider_has_no_flush``, and the
caller sits in dead air until they give up. The agent never says anything,
because nothing reaches the model: a turn that was never transcribed is a turn
the LLM is never asked about. The agent's own system prompt already carries
"if a transcript is unintelligible, ask the caller to repeat"
(``spoken_language.language_directive``) -- that instruction cannot fire on a
turn the pipeline never delivers.

**What the variable actually is.** Not "the agent declares one language". The
harness measured the same dead air on an agent declaring ``["en", "de"]`` with
follow-the-caller OFF, and a healthy call on the same agent with it ON. What
matters is whether a language SWITCHER is running: without one the recognizer
is pinned to a single language for the whole call and a caller outside it can
never be heard. So the gate here is the switcher, not the language count.

**What it does NOT claim.** It never asserts which language the caller spoke.
Nothing in this worker can establish that: ``_StableLanguageLatch.observe``
discards evidence for a language outside the call's frozen profiles, and both
evidence adapters restrict the recognizer to the configured set up front, so
no admitted configuration can produce stable evidence of an OUT-OF-SET
language. The line is therefore a statement about the AGENT -- "I did not
catch that; I can help in <the agent's languages>" -- which is true whether
the cause was an unsupported language, a withheld final or an audio dropout.
That is the correction a Codex adversarial review forced on the original
proposal, and it is why there is one branch here and not two.

**When it speaks.** Only on the SDK's own ``user_transcription_timeout``,
which by construction fires after the caller's VAD end-of-speech and is
cancelled by VAD start-of-speech and by any final with text
(``transcription_stall`` module docstring). So this can never talk over a live
caller: the arm/cancel is the SDK's, not a timer of ours, and the earlier
design that fired mid-utterance was rejected for exactly that reason.

**How it speaks, and why there is no phrase table.** The default is a one-shot
``generate_reply`` instruction, not a canned string. A per-language table of
sentences would have shipped English and left the other sixteen languages with
today's silence -- the defect, kept. The instruction is written in English
(the same choice, for the same reason, as ``language_directive``: a
multilingual model follows an English meta-instruction reliably, and a
hand-translated one is a silent steering bug rather than a visible typo) and
names the languages from the agent's OWN configured list, so the rendered
line comes out in the call's language with no per-language constant anywhere.
A customer who wants exact words configures ``phrase`` and gets it spoken
verbatim instead.

Everything here is pure and synchronous: the counter, the clock-free gates and
the decision live in this module, and the speaking is two injected callables.
The production wiring is a handful of lines at the SDK event handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

try:  # package import in tests; flat /app import in the pinned worker image
    from .spoken_language import language_name, normalize_language
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from spoken_language import language_name, normalize_language

# Imported rather than restated: the two recovery labels this module defers to
# belong to the component that produces them, and a copied literal here would
# be a second place to change when one of them is renamed.
try:  # package import in tests; flat /app import in the pinned worker image
    from .transcription_stall import RECOVERY_FORCED, RECOVERY_TERMINAL
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from transcription_stall import RECOVERY_FORCED, RECOVERY_TERMINAL

#: The worker log event. One line per expiry this module was consulted about,
#: whatever it decided -- a silent no-op is the failure mode the whole ticket
#: is about, so every branch is named and countable. Plus exactly one line at
#: the start of a call on an engine with no separate transcription step, where
#: no expiry can ever arrive to be consulted about
#: (``UnrecognisedSpeechReply.on_call_started``).
REPLY_EVENT = "unrecognised_speech_reply"

#: How many times one call may hear this line. Founder decision (2026-09-22):
#: twice. Once is a repair; a third is nagging a caller who has already been
#: told, and the caller-facing cost of getting this wrong is higher than the
#: cost of one unrepaired turn.
#:
#: This default lives HERE and nowhere else. The control plane validates the
#: BOUNDS of a customer's override and stores only what they wrote, so a
#: stored section is a choice and an absent one is this constant -- there is
#: no second copy to drift.
DEFAULT_MAX_REPLIES_PER_CALL = 2

#: Bounds the control plane enforces on an override, repeated here because the
#: worker must also survive a value written straight into agent metadata by a
#: generic PATCH. Kept equal by ``tests/test_sch1007_unrecognised_speech.py``.
MAX_REPLIES_BOUNDS = (1, 5)
PHRASE_MAX_CHARS = 300

#: Outcomes, in the order they are decided. Named rather than boolean so the
#: harness run table and a production log say WHY a call stayed silent.
SPOKE_GENERATED = "spoke_generated"
SPOKE_PHRASE = "spoke_phrase"
SKIP_DISABLED = "disabled_by_agent"
SKIP_NO_LANGUAGE_AXIS = "engine_has_no_transcriber"
SKIP_SCRIPTED_TURNS = "replies_come_from_a_workflow"
SKIP_LANGUAGE_SWITCHER = "language_switcher_running"
SKIP_RECOVERY_IN_FLIGHT = "recovery_in_flight"
SKIP_ADMISSION_CLOSED = "admission_closed"
SKIP_FINAL_HELD = "final_held_by_transcript_gate"
SKIP_AGENT_BUSY = "agent_not_idle"
SKIP_CAP_REACHED = "max_per_call_reached"
SKIP_UNNAMED_LANGUAGE = "call_language_has_no_name"
SKIP_SPEAK_FAILED = "speak_failed"
SKIP_STATE_UNAVAILABLE = "call_state_unavailable"
#: The call is navigating a phone menu: the repair line is a worker-initiated reply, and the far end speaks to nobody.
SKIP_NAVIGATING = "navigating_a_phone_menu"

#: Leads every rendered instruction, so a line the model produced under it is
#: identifiable in a prompt dump and cannot be mistaken for the agent's own
#: authored copy. Named in the same shape as ``spoken_language``'s markers;
#: unlike those it guards no idempotence, because this instruction is passed
#: once per firing and never appended to a durable prompt.
INSTRUCTION_MARKER = "Skysay unrecognised-speech repair:"

#: Agent states in which speaking would queue behind, or collide with, an
#: utterance already in flight. The timeout cannot fire while the caller is
#: speaking, but it CAN fire while the agent is mid-reply (the managed-PSTN
#: word gate withholds finals during a reply), and a repair line queued behind
#: an answer the caller is still hearing is worse than none.
BUSY_AGENT_STATES = frozenset({"speaking", "thinking", "initializing"})


@dataclass(frozen=True)
class UnrecognisedSpeechSettings:
    """The resolved per-agent choice for one call."""

    enabled: bool
    phrase: str
    max_per_call: int


def _bounded_int(value: Any, *, default: int) -> int:
    low, high = MAX_REPLIES_BOUNDS
    # A bool is an int in Python and 1.5 is not a cap of one: both are junk
    # that reached agent metadata another way, and the control plane's lenient
    # read drops each of them. Coercing here would make the worker and the
    # projection disagree about the same stored value.
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if low <= number <= high else default


def unrecognised_speech_settings(
    realtime_config: Mapping[str, Any] | None,
) -> UnrecognisedSpeechSettings:
    """Resolve ``voice_polish.unrecognised_speech`` for one call.

    ON unless the agent says otherwise: an absent section, an absent key and a
    section this version cannot read all mean the platform default. That is the
    opposite of ``acknowledgements``, and deliberately so -- an acknowledgement
    that never fires costs a caller nothing, while this one not firing IS the
    reported defect.

    A ``phrase`` is used verbatim; otherwise the call gets the generated line.
    An out-of-bounds or unreadable ``max_per_call`` falls back to the default
    rather than refusing the feature, because a bad number written straight
    into metadata must not reinstate the silence.
    """

    polish = (
        realtime_config.get("voice_polish")
        if isinstance(realtime_config, Mapping)
        else None
    )
    section = (
        polish.get("unrecognised_speech") if isinstance(polish, Mapping) else None
    )
    if not isinstance(section, Mapping):
        return UnrecognisedSpeechSettings(True, "", DEFAULT_MAX_REPLIES_PER_CALL)
    if section.get("enabled") is False:
        return UnrecognisedSpeechSettings(False, "", DEFAULT_MAX_REPLIES_PER_CALL)
    raw_phrase = section.get("phrase")
    phrase = " ".join(raw_phrase.split()) if isinstance(raw_phrase, str) else ""
    if len(phrase) > PHRASE_MAX_CHARS:
        # DROPPED rather than truncated, so all three readers of this field
        # agree: a strict write refuses it, the editor's hidden-field cleaner
        # drops it, and a value that reached metadata another way gets the
        # generated line -- not a sentence cut off mid-word, and not a
        # customer's wording that changes the next time they save something
        # unrelated.
        phrase = ""
    return UnrecognisedSpeechSettings(
        True,
        phrase,
        _bounded_int(section.get("max_per_call"), default=DEFAULT_MAX_REPLIES_PER_CALL),
    )


def configured_language_names(
    language: Any, configured: Sequence[Any] | None
) -> tuple[str, ...]:
    """The English names of the languages this agent can be helped in.

    The call's own language first, then the rest of the agent's configured
    list in its stored order, deduplicated, dropping anything this image
    cannot name. Read from the agent's configuration exactly as the cascade
    admission path reads it (``production.configured_languages``) -- this
    module owns no language list of its own, which is the whole point.
    """

    names: list[str] = []
    codes: list[Any] = [language]
    if isinstance(configured, (list, tuple)):
        codes.extend(configured)
    for code in codes:
        name = language_name(code) if isinstance(code, str) else ""
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _joined(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def repair_instruction(language: Any, configured: Sequence[Any] | None) -> str:
    """The one-shot instruction for the generated repair line, or ``""``.

    ``""`` when the call's language has no name in this image: the instruction
    could then neither say which language to speak nor which languages are on
    offer, and a repair line in the wrong language is worse than the silence
    it is trying to end -- the same rule ``greeting_fallback_message`` holds
    for the same reason.

    The clauses are each a failure this line could otherwise become. It must
    not guess at what the caller said (it has no transcript at all, so any
    guess is invention); it must not answer a question (there is no question);
    it must not call a tool (``end_call`` on a caller who has said nothing
    intelligible would turn a repair into a hang-up); and it must be ONE short
    line, because a non-streaming TTS is wrapped in ``tts.StreamAdapter`` and
    synthesises sentence by sentence, so every extra sentence is another
    provider request that can fail on its own.
    """

    names = configured_language_names(language, configured)
    spoken = language_name(language)
    if not spoken or not names:
        return ""
    code = normalize_language(language)
    return (
        f"{INSTRUCTION_MARKER} The caller just spoke and the transcriber "
        f"produced nothing, so you did not receive their words. Say ONE short "
        f"line in {spoken} ({code}) and nothing else: tell them you could not "
        f"understand what they said, and that you can help in "
        f"{_joined(names)}. Do not guess what they said, do not answer a "
        f"question, do not invent facts, do not invoke any tool, and do not "
        f"end the call."
    )


class _Unavailable(Exception):
    """A call-state probe could not answer."""


def _check(check: Callable[[], bool] | None, *, absent: bool) -> bool:
    """A call-state probe's answer; ``absent`` when it was not wired.

    A probe that RAISES is not answered with a default. Both of these gates
    exist to stop the line being spoken at a moment that would make it false,
    and both defaults that would let it through are the harmful ones: an
    unreadable admission state can be a call already tearing down, and an
    unreadable transcript gate can be a final it is holding for a caller who
    was in fact understood. The cost of the other direction is one unspoken
    repair on a call whose bookkeeping is already broken.
    """

    if check is None:
        return absent
    try:
        return bool(check())
    except Exception:
        raise _Unavailable from None


class UnrecognisedSpeechReply:
    """Decide, at most ``max_per_call`` times, to speak one repair line.

    Held by the worker for the life of one call and consulted from the SDK's
    ``user_transcription_timeout`` handler, immediately after
    ``TranscriptionStallRecovery.on_timeout`` has decided what it will do
    about the same event. The recovery's answer is an INPUT here rather than
    something re-derived: a forced Azure flush may still produce the caller's
    words a moment later, and speaking "I did not catch that" over the top of
    the turn it recovered would be a lie this module manufactured itself.
    """

    def __init__(
        self,
        *,
        settings: UnrecognisedSpeechSettings,
        instruction: str,
        replies_are_scripted: bool = False,
        speak_phrase: Callable[[str], Any],
        speak_generated: Callable[[str], Any],
        emit: Callable[..., Any],
        has_language_axis: bool,
        has_language_switcher: bool,
        holds_final: Callable[[], bool] | None = None,
        admits: Callable[[], bool] | None = None,
        agent_state: Callable[[], str] | None = None,
        suspended: Callable[[], bool] | None = None,
    ) -> None:
        self._settings = settings
        #: True while the call navigates a phone menu: a menu nobody transcribed is not a caller to repair, and a
        #: reply made now could take the speech window of a menu the model has still to answer.
        self._suspended = suspended
        self._instruction = str(instruction or "")
        self._replies_are_scripted = bool(replies_are_scripted)
        self._replying = False
        self._speak_phrase = speak_phrase
        self._speak_generated = speak_generated
        self._emit = emit
        self._has_language_axis = bool(has_language_axis)
        self._has_language_switcher = bool(has_language_switcher)
        self._holds_final = holds_final
        self._admits = admits
        self._agent_state = agent_state
        self._spoken = 0
        self._standing_answer_recorded = False

    def on_call_started(self) -> str:
        """Record this engine's standing answer once, or return ``""``.

        The outcome ``engine_has_no_transcriber`` had no writer. ``_decide``
        is reached only from the SDK's ``user_transcription_timeout``, which
        is an STT-node event: an engine with no separate transcription step
        never emits it, so the branch that exists for exactly that engine
        could never run there. The evidence for a voice-to-voice call was an
        ABSENCE, and an absence cannot tell "the module stood down" from "the
        module was never wired", "the worker crashed before the handler" or
        "the log was truncated". This is the writer.

        Called once, at the point the call's session is built. Emitted through
        ``_emit_outcome`` with the same field set every other outcome carries,
        because the whole value here is that one channel answers "what did the
        repair line do on this call" for every engine.

        ``""`` on a cascade engine: there the standing answer is not standing
        at all -- the module's answer depends on the recovery, the cap and the
        call state at the moment of an expiry, and a line recorded up front
        would be a claim about a decision nobody has made yet.

        Of the two outcomes this method can record, the disabled one wins,
        exactly as in ``_decide``: a customer who switched the line off is the
        reason it will not speak, and two orders for the same two gates would
        make the log depend on which door the decision came through. On a
        cascade engine it records NEITHER, disabled or not -- the paragraph
        above is why -- so "disabled first" is a rule about this method's two
        answers and not a claim that every disabled call produces a line.

        ONE line from HERE. A second ``engine_has_no_transcriber`` on the same
        call can only come from ``_decide``, which means the SDK emitted a
        ``user_transcription_timeout`` on an engine with no STT node. That is
        a finding about the SDK rather than about this module, it is left
        visible rather than suppressed, and the harness row for these engines
        asserts exactly one event so that it cannot pass unnoticed.
        """

        if self._has_language_axis or self._standing_answer_recorded:
            return ""
        self._standing_answer_recorded = True
        outcome = SKIP_NO_LANGUAGE_AXIS if self._settings.enabled else SKIP_DISABLED
        self._emit_outcome(outcome, "")
        return outcome

    @property
    def spoken(self) -> int:
        """How many repair lines this call has been given."""

        return self._spoken

    def on_transcription_timeout(self, *, recovery: str) -> str:
        """Handle one expiry; returns the outcome label that was logged.

        The branches, in the order they are decided, each with the reason it
        comes before the ones after it:

        * ``disabled_by_agent`` -- the customer switched the line off. Checked
          first so a disabled agent's log carries one honest reason rather
          than whichever gate happened to close first.
        * ``engine_has_no_transcriber`` -- GPT-Live and Gemini-Live have no
          separate STT step to go silent, so this event cannot mean there what
          it means on a cascade call. Gated explicitly rather than relying on
          the event not firing.
        * ``replies_come_from_a_workflow`` -- a pinned workflow run is the
          sole execution authority for what this agent says, and
          ``WorkflowTurnAgent`` delivers that by overriding ``llm_node``. A
          ``generate_reply`` there does not produce this module's line at all:
          it would drive the workflow runner and speak, and step, the flow's
          NEXT turn. The engine is cascade and the recognizer is pinned, so
          every other gate here would have let it through; this one exists
          because the seam the line is spoken through is not the model on such
          a call. A scripted flow that wants to answer an unintelligible turn
          says so in the flow.
        * ``language_switcher_running`` -- follow-the-caller is configured, so
          the recognizer is not pinned: a caller outside the call's current
          language can still be heard, and the controlled switch is the
          mechanism that should answer for them. This is the gate the harness
          measured: the same agent is silent without a switcher and healthy
          with one.
        * ``recovery_in_flight`` -- the stall recovery forced a flush, and the
          finals it was holding may arrive within its outcome window. Speaking
          now risks talking over a turn that was in fact understood. The cost
          of deferring is stated in the module's ticket: on a provider WITH a
          flush that comes back empty, the caller is answered on their second
          attempt rather than their first.
        * ``admission_closed`` -- ``end_call`` has closed transcript
          admission. The call is ending; a repair line is noise.
        * ``final_held_by_transcript_gate`` -- the managed-PSTN word gate is
          withholding a final for this reply. The caller WAS understood and
          their words reach the SDK when the reply ends, so "I could not
          understand you" would be false.
        * ``agent_not_idle`` -- the agent is mid-reply. The repair would queue
          behind an utterance the caller is still hearing.
        * ``max_per_call_reached`` -- the cap.
        * ``call_language_has_no_name`` -- no instruction could be rendered
          (and no phrase was configured), so nothing truthful can be said in
          this call's language.

        Never raises: this runs inside the SDK's event dispatch on the media
        loop, where an exception costs the call more than a missed line.
        """

        outcome = self._decide(recovery)
        self._emit_outcome(outcome, recovery)
        return outcome

    def _decide(self, recovery: str) -> str:
        if not self._settings.enabled:
            return SKIP_DISABLED
        try:
            if self._suspended is not None and self._suspended():
                return SKIP_NAVIGATING
        except Exception:  # noqa: BLE001 - an unreadable state is "navigating": the safe direction is silence
            return SKIP_NAVIGATING
        if not self._has_language_axis:
            return SKIP_NO_LANGUAGE_AXIS
        if self._replies_are_scripted:
            return SKIP_SCRIPTED_TURNS
        if self._has_language_switcher:
            return SKIP_LANGUAGE_SWITCHER
        if str(recovery or "") == RECOVERY_FORCED:
            return SKIP_RECOVERY_IN_FLIGHT
        if str(recovery or "") == RECOVERY_TERMINAL:
            return SKIP_ADMISSION_CLOSED
        # A probe that cannot answer is its own named outcome, not a guess in
        # either direction: see _check.
        try:
            if not _check(self._admits, absent=True):
                return SKIP_ADMISSION_CLOSED
            if _check(self._holds_final, absent=False):
                return SKIP_FINAL_HELD
        except _Unavailable:
            return SKIP_STATE_UNAVAILABLE
        if self._busy():
            return SKIP_AGENT_BUSY
        if self._spoken >= self._settings.max_per_call:
            return SKIP_CAP_REACHED
        phrase = self._settings.phrase
        if not phrase and not self._instruction:
            return SKIP_UNNAMED_LANGUAGE
        # The slot is RESERVED before the side effect, and a failed attempt
        # burns it. The earlier order -- speak, then count on a clean return
        # -- read as generous and was the one fail-OPEN rule left in a module
        # whose every other gate fails closed: a speak callable that had
        # already queued audio and then raised was logged as a failure and
        # left the budget untouched, so a later expiry could play the line
        # past a cap the customer configured and the docs promise. One missed
        # repair on a call whose speak is already broken is the cheaper of the
        # two mistakes, and it is the same trade the probes above make.
        #
        # ``_replying`` closes the same window against re-entry: this runs
        # synchronously inside the SDK's event dispatch, and a speak that
        # re-entered this handler would otherwise read the pre-increment
        # count. Neither SDK call does that today; the guard costs a boolean
        # and does not depend on that staying true.
        if self._replying:
            return SKIP_AGENT_BUSY
        self._replying = True
        self._spoken += 1
        try:
            if phrase:
                self._speak_phrase(phrase)
            else:
                self._speak_generated(self._instruction)
        except Exception:
            return SKIP_SPEAK_FAILED
        finally:
            self._replying = False
        return SPOKE_PHRASE if phrase else SPOKE_GENERATED

    def _busy(self) -> bool:
        # An UNWIRED probe means idle, the same optional-probe convention the
        # two gates above use. A WIRED one that cannot answer means busy: an
        # unreadable state is not evidence the agent is silent, and speaking
        # over a live reply is the worse of the two mistakes. A probe that
        # raises and one that hands back ``None`` or ``""`` are the same case
        # -- the earlier code caught only the first, and the second fell
        # through ``str(x or "")`` to a value no busy state matches, which
        # read as idle and spoke.
        if self._agent_state is None:
            return False
        try:
            state = self._agent_state()
        except Exception:
            return True
        if not isinstance(state, str) or not state:
            return True
        return state in BUSY_AGENT_STATES

    def _emit_outcome(self, outcome: str, recovery: str) -> None:
        try:
            self._emit(
                REPLY_EVENT,
                outcome=outcome,
                recovery=str(recovery or ""),
                spoken=self._spoken,
                max_per_call=self._settings.max_per_call,
                source=("phrase" if self._settings.phrase else "generated"),
            )
        except Exception:
            pass


__all__ = [
    "BUSY_AGENT_STATES",
    "DEFAULT_MAX_REPLIES_PER_CALL",
    "INSTRUCTION_MARKER",
    "MAX_REPLIES_BOUNDS",
    "PHRASE_MAX_CHARS",
    "REPLY_EVENT",
    "SKIP_ADMISSION_CLOSED",
    "SKIP_AGENT_BUSY",
    "SKIP_CAP_REACHED",
    "SKIP_DISABLED",
    "SKIP_FINAL_HELD",
    "SKIP_LANGUAGE_SWITCHER",
    "SKIP_NO_LANGUAGE_AXIS",
    "SKIP_RECOVERY_IN_FLIGHT",
    "SKIP_SCRIPTED_TURNS",
    "SKIP_SPEAK_FAILED",
    "SKIP_NAVIGATING",
    "SKIP_STATE_UNAVAILABLE",
    "SKIP_UNNAMED_LANGUAGE",
    "SPOKE_GENERATED",
    "SPOKE_PHRASE",
    "UnrecognisedSpeechReply",
    "UnrecognisedSpeechSettings",
    "configured_language_names",
    "repair_instruction",
    "unrecognised_speech_settings",
]

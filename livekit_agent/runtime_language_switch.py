# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Frozen, worker-local controlled language switching for hosted calls.

This module deliberately has no provider registry and no control-plane I/O.
The control plane freezes and validates the selectable profiles before media;
the worker verifies that envelope again, builds only one of those profiles, and
uses the public LiveKit agent APIs supplied by its caller.  Keeping the state
machine here makes its race and evidence contract unit-testable without a
LiveKit installation or a phone call.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import traceback
from dataclasses import dataclass
from typing import Any, AsyncIterable, AsyncIterator, Callable, Iterable, Mapping


class LanguageSwitchError(ValueError):
    """The frozen follow-caller contract is malformed or unsafe to apply."""


class SegmentNativeUsageUnavailable(LanguageSwitchError):
    """A segment owner's exact native usage could not be collected.

    Its own type, so the worker routes it to the metering-unavailable branch
    (captured, refused metering kept) rather than the generic report failure.
    ``kind`` and ``status_code`` are the provider evidence's bounded
    discriminator when it has one (Soniox), never a message.
    """

    def __init__(self, *, error_type: str, kind: str | None, status_code: int | None) -> None:
        self.error_type = error_type
        self.kind = kind
        self.status_code = status_code
        super().__init__("segment native usage is unavailable for settlement")

    @property
    def discriminator(self) -> str:
        if self.kind is None:
            return f"segment:{self.error_type}"
        suffix = f":{self.status_code}" if self.status_code is not None else ""
        return f"soniox:{self.kind}{suffix}"


class UnsupportedLanguageError(LanguageSwitchError):
    """A well-formed language code this call published no profile for.

    Separate from its parent because refusing it is an ordinary conversational
    outcome, not a contract fault: the code is canonical, so it is safe to
    record as durable rejected evidence, and the caller is owed a spoken "I
    can only speak these languages" rather than a generic malformed-input
    rejection with nothing to say.
    """

    def __init__(self, language: str, *, allowed: Iterable[str]) -> None:
        self.language = language
        self.allowed = tuple(allowed)
        super().__init__(
            f"{language!r} is not one of the languages this call can speak: "
            + ", ".join(self.allowed)
        )


_MAX_PROFILES = 32
# The worker's terminal payload is validated by the control-plane settlement
# contract. Keep this bound identical: emitting more segments would strand a
# valid call's native usage in awaiting_usage rather than silently dropping it.
_MAX_SEGMENTS = 64
_MAX_REASON_LENGTH = 240
_SOURCES = frozenset({"explicit_request", "provider_stable_evidence", "workflow_interrupt"})
#: The sources that carry a caller-facing request: the caller asked, so the
#: caller is answered whether or not the switch had to cut anything.
_REQUEST_SOURCES = frozenset({"explicit_request", "workflow_interrupt"})
# The one phrase every refusal of a well-formed but unpublished target emits.
# The durable evidence collector classifies a rejection by matching this text,
# so the two rejection sites -- the model tool and the trusted evidence queue
# -- must not spell it differently. It is deliberately NOT the message handed
# to the model: a model repeating internal vocabulary at a caller is how an
# agent once told someone their own configured language was "restricted by the
# validation profiles".
_UNSUPPORTED_LANGUAGE_REASON = "requested language is not in the frozen allowed profiles"
# The simultaneous-listening modes the control plane is allowed to freeze into
# a call envelope.  Deliberately a copy of the capability registry's set rather
# than an import of it: this module runs in the worker with no registry
# dependency by design, and the envelope it is re-validating was written by an
# image whose artifact revision it also checks.  A mode listed here still needs
# a builder in ``demo`` before a call can be constructed -- this only bounds
# what a frozen envelope may claim.
_SIMULTANEOUS_MODES = frozenset(
    {
        "language_multi",
        "language_identification",
        "strict_language_hints",
        "continuous_language_identification",
    }
)
#: The one open scope a frozen envelope may name instead of a declared
#: language list: its construction (``language_multi``) sends the provider no
#: language, so there is no provider set to be a subset of. A copy
#: of the registry's ``LANGUAGE_SCOPE_ANY`` and ``_OPEN_SCOPE_MODES`` for the
#: reason ``_SIMULTANEOUS_MODES`` is a copy.
_LANGUAGE_SCOPE_ANY = "any"
_OPEN_SCOPE_MODES = frozenset({"language_multi"})
# Terminal shutdown must proceed to the durable settlement/reconciliation
# path even when a provider activity or adapter close stops responding.
_TERMINAL_SWITCH_DRAIN_SECONDS = 2.0
_RETIRED_COMPONENT_CLOSE_SECONDS = 2.0
_TERMINAL_CANCEL_GRACE_SECONDS = 0.2
# Every wait a switch performs is a wait the CALLER spends in silence, so none
# of them may be unbounded. The session has no contractual upper bound on
# reaching a turn boundary -- a component that stops responding simply never
# reports idle -- and an unbounded wait there strands the transition holding
# the switch lock with the interrupted turn already cancelled: the worst
# possible state, because the reply that would have spoken is gone and nothing
# will replace it. The pre-swap bound is the generous one (exceeding it aborts
# a switch that might have worked); the correction bound is shorter, because
# past it the right move is to speak anyway rather than keep waiting.
_SWITCH_IDLE_WAIT_SECONDS = 10.0
_CORRECTION_IDLE_WAIT_SECONDS = 5.0
# An unexpected ``_apply`` failure used to be logged as only its exception
# class name -- diagnosable only by reproducing the failure from scratch.
# Frame count and character length are both bounded independently: one
# frame's own source line is not itself bounded, and a deep call stack is not
# bounded by frame count alone. Negative ``traceback.extract_tb(limit=)``
# keeps the LAST frames (innermost, where the failure actually happened), not
# the outermost caller -- the opposite of what a positive limit keeps.
_MAX_TRACEBACK_FRAMES = 6
_MAX_TRACEBACK_LENGTH = 4000


def _language(value: Any, *, field: str) -> str:
    language = str(value or "").strip().casefold()
    if not language or len(language) > 32 or not language.replace("-", "").isalnum():
        raise LanguageSwitchError(f"{field} must be a canonical language")
    return language


def _publication_version(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise LanguageSwitchError(f"{field} must be a positive integer")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise LanguageSwitchError(f"{field} must be a positive integer") from exc
    if version < 1:
        raise LanguageSwitchError(f"{field} must be a positive integer")
    return version


def _bounded_traceback(exc: BaseException) -> str:
    """A bounded, MESSAGE-FREE diagnostic for an ``_apply`` failure nobody has
    reproduced yet: file, line, function, and the literal source line as
    written in this repository -- never the exception's own ``str()``.

    ``traceback.format_exception`` was reviewed and rejected for this: its
    final line is ``ExceptionType: message``, and ``safe_component_error_diagnostics``
    (production.py) already documents why that message is not safe to log --
    a provider exception's message can carry a request URL, an account
    identifier, or credential material, and the frame/length bounds below
    would not redact that if it were included.  A source LINE is not the
    same risk: it is this repository's own text (``build_thing(api_key)``,
    never the runtime value ``api_key`` holds), exactly like a frame's file
    and line number are code locations, not data.  The exception CLASS name
    is reported separately as ``error_type``; this field exists only to
    answer "where," never "what did it say."
    """
    frames = traceback.extract_tb(exc.__traceback__, limit=-_MAX_TRACEBACK_FRAMES)
    formatted = "".join(traceback.format_list(frames))
    return formatted[-_MAX_TRACEBACK_LENGTH:]


def _contains_credential_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "credential" in str(key).casefold():
                return True
            if _contains_credential_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_credential_key(item) for item in value)
    return False


@dataclass(frozen=True)
class FrozenLanguageProfile:
    language: str
    publication_version: int
    config: dict[str, Any]
    profile: Any


@dataclass(frozen=True)
class FollowCallerProfiles:
    publication_version: int
    opening_language: str
    active_language: str
    profiles: dict[str, FrozenLanguageProfile]
    simultaneous_stt: dict[str, Any]

    @classmethod
    def from_realtime_config(
        cls,
        config: Mapping[str, Any],
        *,
        profile_from_config: Callable[[dict[str, Any]], Any],
        validate_billing_profile: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> "FollowCallerProfiles | None":
        raw = config.get("follow_caller")
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
            raise LanguageSwitchError("follow_caller must be an enabled frozen envelope")
        if _contains_credential_key(raw):
            raise LanguageSwitchError("follow_caller must not carry credential material or ids")
        publication_version = _publication_version(
            raw.get("publication_version"), field="follow_caller publication_version"
        )
        opening_language = _language(raw.get("opening_language"), field="follow_caller opening_language")
        active_language = _language(raw.get("active_language"), field="follow_caller active_language")
        raw_profiles = raw.get("allowed_profiles")
        if not isinstance(raw_profiles, Mapping) or not raw_profiles or len(raw_profiles) > _MAX_PROFILES:
            raise LanguageSwitchError("follow_caller allowed_profiles must be a bounded nonempty mapping")
        profiles: dict[str, FrozenLanguageProfile] = {}
        for raw_language, raw_profile in raw_profiles.items():
            language = _language(raw_language, field="follow_caller profile language")
            if language in profiles or not isinstance(raw_profile, Mapping):
                raise LanguageSwitchError("follow_caller profile is invalid")
            raw_config = raw_profile.get("config")
            if not isinstance(raw_config, Mapping) or _contains_credential_key(raw_config):
                raise LanguageSwitchError("follow_caller profile config is invalid")
            copied = dict(raw_config)
            if "follow_caller" in copied:
                raise LanguageSwitchError("follow_caller profile config cannot nest an envelope")
            if _language(copied.get("language"), field="follow_caller profile config language") != language:
                raise LanguageSwitchError("follow_caller profile key and config language disagree")
            try:
                profile = profile_from_config(copied)
            except Exception as exc:
                raise LanguageSwitchError("follow_caller profile config is not worker-runnable") from exc
            if validate_billing_profile is not None:
                try:
                    validate_billing_profile(language, copied)
                except LanguageSwitchError:
                    raise
                except Exception as exc:
                    raise LanguageSwitchError("follow_caller profile does not match frozen billing identity") from exc
            profiles[language] = FrozenLanguageProfile(
                language=language,
                publication_version=publication_version,
                config=copied,
                profile=profile,
            )
        if opening_language not in profiles or active_language not in profiles:
            raise LanguageSwitchError("follow_caller active/opening language is not an allowed profile")
        simultaneous_stt = raw.get("simultaneous_stt")
        if not isinstance(simultaneous_stt, Mapping) or _contains_credential_key(simultaneous_stt):
            raise LanguageSwitchError("follow_caller simultaneous_stt is invalid")
        languages = simultaneous_stt.get("languages")
        mode = str(simultaneous_stt.get("mode") or "")
        if (
            mode not in _SIMULTANEOUS_MODES
            or not isinstance(languages, list)
            or [_language(item, field="follow_caller simultaneous_stt language") for item in languages]
            != sorted(profiles)
        ):
            raise LanguageSwitchError("follow_caller simultaneous_stt does not exactly match allowed profiles")
        # The selected set above is always this call's frozen profiles. What
        # it must sit inside is either the provider set the envelope was
        # admitted against, or -- for a construction that names no language --
        # an explicit open scope. Exactly one of the two, never neither: an
        # envelope that simply lost its list is not an open one.
        if "language_scope" in simultaneous_stt:
            if (
                "declared_languages" in simultaneous_stt
                or simultaneous_stt["language_scope"] != _LANGUAGE_SCOPE_ANY
                or mode not in _OPEN_SCOPE_MODES
            ):
                raise LanguageSwitchError("follow_caller simultaneous_stt language scope is invalid")
        else:
            declared_languages = simultaneous_stt.get("declared_languages")
            if not isinstance(declared_languages, list) or not set(
                _language(item, field="follow_caller declared simultaneous STT language") for item in languages
            ).issubset(
                {
                    _language(item, field="follow_caller declared simultaneous STT language")
                    for item in declared_languages
                }
            ):
                raise LanguageSwitchError("follow_caller simultaneous_stt does not exactly match allowed profiles")
        for field in ("provider", "registry_name", "model"):
            if not str(simultaneous_stt.get(field) or "").strip():
                raise LanguageSwitchError("follow_caller simultaneous_stt is incomplete")
        if not isinstance(simultaneous_stt.get("adapter_options"), Mapping):
            raise LanguageSwitchError("follow_caller simultaneous_stt adapter options are invalid")
        return cls(
            publication_version=publication_version,
            opening_language=opening_language,
            active_language=active_language,
            profiles=profiles,
            simultaneous_stt=dict(simultaneous_stt),
        )


#: The worker-local tool a Follow-mode call adds; a configured tool of this name
#: is not built beside it (``production.build_livekit_model_tools``).
SWITCH_LANGUAGE_TOOL_NAMES: frozenset[str] = frozenset({"switch_language"})


def switch_language_tool_schema(languages: Iterable[str]) -> dict[str, Any]:
    """The only model-controlled inputs are target language and bounded reason.

    ``languages`` is REQUIRED and is this call's own frozen allowed set. The
    schema used to say only that ``language`` was a string of up to 32
    characters, so nothing told the model which literal values the strict
    membership test downstream would accept. A model asked to switch to
    Russian sent its own word for it, the switch was refused, and the agent
    told the caller that a language they had configured and published was
    "restricted by the validation profiles". Naming the accepted codes in the
    description -- the text the model is already reading -- fixes that.

    They are named there and NOT as a JSON-Schema ``enum``, which is what this
    tool carried until a live call showed what a closed vocabulary costs. An
    enum does not merely advertise the accepted values: it removes every other
    value from what the tool can express. A caller asking for a language this
    call never published then has no representable request, so the refusal
    branch below cannot run, no rejected evidence is recorded, and the model
    is left holding a caller request it has no way to resolve -- observed live
    as an agent that answered "switch to Russian" on an {en, et} publication
    with nothing at all, no tool call and no words, until the caller hung up.
    Refusing is a first-class outcome of this tool, so the tool has to be
    callable for a target that will be refused; ``model_tool`` answers such a
    call with the available languages and an instruction to say so, which is a
    sentence the caller can hear rather than a value the schema swallowed.
    """
    if isinstance(languages, str):
        # A bare string iterates as characters, which would offer the model a
        # vocabulary of letters. Refuse rather than build a nonsense list.
        raise LanguageSwitchError("switch_language languages must be a collection of codes")
    allowed = sorted({str(language) for language in languages})
    if not allowed:
        raise LanguageSwitchError("switch_language requires at least one allowed language")
    return {
        "name": "switch_language",
        "description": (
            "Request a switch to one of the caller-approved languages after a clear "
            "request. This call can speak: "
            + ", ".join(allowed)
            + ". Pass one of those codes as `language`. If the caller asks for any "
            "other language, call this tool with that language's code anyway: the "
            "result will tell you it is unavailable and which languages are, so you "
            "can say so instead of leaving the caller without an answer."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "language": {"type": "string", "minLength": 1, "maxLength": 32},
                "reason": {"type": "string", "minLength": 1, "maxLength": _MAX_REASON_LENGTH},
            },
            "required": ["language", "reason"],
        },
    }


@dataclass
class _UsageSegment:
    language: str
    publication_version: int
    started_boundary: int
    collector: Any
    stack: Any
    ended_boundary: int | None = None
    native_usage: dict[str, Any] | None = None
    native_usage_error: str | None = None
    native_usage_kind: str | None = None
    native_usage_status: int | None = None
    attached: bool = False
    committed: bool = False


class ControlledLanguageSwitcher:
    """Serialize only frozen-profile changes at a completed turn boundary.

    A model function executes inside its own reply, so its public method queues
    the actual operation and returns ``scheduled``; waiting for idle inside
    that function would deadlock the turn.  Provider evidence can use the same
    queue directly.  A success event is emitted only after all SDK updates.
    """

    def __init__(
        self,
        *,
        frozen: FollowCallerProfiles,
        session: Any,
        initial_profile: Any,
        initial_stack: Any,
        build_tts: Callable[[Any, Mapping[str, Any], int], Any],
        collector_factory: Callable[[], Any],
        emit: Callable[..., None],
        speech_suspended: Callable[[], bool] | None = None,
    ) -> None:
        # True while the call is waiting in a phone menu or a hold queue. The
        # switch itself still applies; what it must not do there is SPEAK, so
        # both the confirmation and the correction are read against this at
        # the moment they would be generated.
        self._speech_suspended = speech_suspended or (lambda: False)
        self._frozen = frozen
        self._session = session
        self._profile = initial_profile
        self._stack = initial_stack
        self._shared_soniox_stt = getattr(initial_profile, "stt_registry", "") == "soniox"
        # The initial stack retains call-wide STT/LLM ownership. Every later
        # handle contains only its own TTS and native usage evidence.
        self._build_tts = build_tts
        self._collector_factory = collector_factory
        self._emit = emit
        self._agent: Any | None = None
        self._turn_cut: Callable[[], None] | None = None
        self._speak_owed_turn: Callable[[], None] | None = None
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task[Any]] = set()
        self._pending_targets: dict[asyncio.Task[Any], str] = {}
        self._terminal = False
        # Settlement must use one immutable view of the segment boundary.  A
        # terminal drain may deliberately return while a cancellation-resistant
        # provider task is still unwinding, so never derive a billed end from
        # the live transition counter after terminal entry.
        self._settlement_boundary: int | None = None
        self._abandoned: set[asyncio.Future[Any]] = set()
        self._boundary = 1
        self._segments: list[_UsageSegment] = [
            _UsageSegment(
                language=initial_profile.language,
                publication_version=frozen.publication_version,
                started_boundary=self._boundary,
                collector=collector_factory(),
                stack=initial_stack,
                attached=True,
                committed=True,
            )
        ]
        self._attempts: list[_UsageSegment] = []
        self._usage_boundary = self._boundary
        self._native_usage_error: str | None = None
        self._attach_component_metrics(
            initial_stack, self._segments[-1].collector, initial_segment=True
        )

    @property
    def active_language(self) -> str:
        return str(self._profile.language)

    def _speech_is_suspended(self) -> bool:
        try:
            return bool(self._speech_suspended())
        except Exception:  # noqa: BLE001 - an unreadable gate stays quiet
            return True

    @property
    def switch_in_flight(self) -> bool:
        """Whether a queued switch has not finished applying yet.

        A switch interrupts whatever is playing and then speaks a recovery
        reply of its own, so the conversation is not idle while one is
        pending -- the mutual-silence rule holds for it.
        """

        return any(not task.done() for task in self._pending)

    @property
    def allowed_languages(self) -> tuple[str, ...]:
        """The frozen codes this switcher accepts, for the model tool schema.

        Read from the switcher that enforces them so the vocabulary offered to
        the model and the vocabulary the switch validates against cannot drift.
        """
        return tuple(sorted(self._frozen.profiles))

    def bind_agent(
        self,
        agent: Any,
        *,
        turn_cut: Callable[[], None] | None = None,
        speak_owed_turn: Callable[[], None] | None = None,
    ) -> None:
        """Bind the live agent, and optionally how it gives back a cut turn.

        The free-form agent owes a turn through ``session.generate_reply``:
        the model speaks again from the conversation. An agent whose turns
        are an authored engine -- the workflow agent -- hands in its own
        pair instead: ``turn_cut`` is told before the interrupt, so an
        outcome landing while the stack is being exchanged is held rather
        than spoken through the retiring one, and ``speak_owed_turn`` gives
        the turn back once the new stack is live. A generated reply on that
        agent would be a new caller turn against a run that already
        recorded the last one.
        """
        self._agent = agent
        self._turn_cut = turn_cut
        self._speak_owed_turn = speak_owed_turn

    def collect_metrics(self, metrics: Any) -> None:
        """Record native metrics in the currently active immutable segment."""
        self._segments[-1].collector.collect(metrics)

    def usage_segments(self) -> list[dict[str, Any]]:
        """Return bounded, ordered native usage evidence without transcript data."""
        if self._native_usage_error is not None:
            failed = next(
                segment for segment in [self._segments[0], *self._attempts]
                if segment.native_usage_error is not None
            )
            raise SegmentNativeUsageUnavailable(
                error_type=failed.native_usage_error,
                kind=failed.native_usage_kind,
                status_code=failed.native_usage_status,
            )
        result: list[dict[str, Any]] = []
        segments = [self._segments[0], *self._attempts]
        for segment in segments:
            if not segment.committed and not segment.attached:
                continue
            result.append(
                {
                    "language": segment.language,
                    "publication_version": segment.publication_version,
                    "started_boundary": segment.started_boundary,
                    "ended_boundary": segment.ended_boundary
                    or self._settlement_boundary
                    or self._boundary,
                    "model_usage": list(segment.collector.flatten()),
                    **(segment.native_usage or {}),
                }
            )
        return result

    async def collect_segment_native_usage(self, *, deadline: float | None = None) -> None:
        """Collect every owner, deferring metering failure to the billing seam.

        Production closes providers before entering its billing-report try.
        Missing exact evidence must leave settlement awaiting reconciliation,
        while still allowing terminal SIP lifecycle reporting and credential
        disposal for the other segments. Never retain a provider exception's
        traceback or message, which may carry key material: only its class
        and, for provider evidence, its bounded ``kind``/``status_code``.

        ``deadline`` is one absolute monotonic instant shared by every owner,
        so serial segments cannot multiply the teardown's Soniox budget. When
        omitted, the first tracker's own budget from now is used for all.
        """
        segments = [self._segments[0], *self._attempts]
        shared = {"deadline": deadline}
        try:
            for segment in segments:
                if segment.native_usage is not None or segment.native_usage_error is not None:
                    continue
                try:
                    if not segment.committed and not segment.attached:
                        # No synthesis could start before SDK attachment. This
                        # owner is excluded from settlement, so only release its
                        # credentials; irrelevant evidence cannot poison a call.
                        segment.native_usage = {}
                    else:
                        segment.native_usage = await self._collect_native_usage(segment, shared)
                except Exception as exc:
                    self._fail_segment(segment, exc)
                finally:
                    tracker = getattr(segment.stack, "soniox_usage_tracker", None)
                    if tracker is not None:
                        tracker.discard_credentials()
        except asyncio.CancelledError as exc:
            # A cancelled collection must not strand later owners' keys, nor
            # leave them looking collected: the job's own cancellation is never
            # reported as a deadline. The caller holds and re-raises it.
            for segment in segments:
                if segment.native_usage is None and segment.native_usage_error is None:
                    self._fail_segment(segment, exc, kind="cancelled")
                tracker = getattr(segment.stack, "soniox_usage_tracker", None)
                if tracker is not None:
                    tracker.discard_credentials()
            raise

    def _fail_segment(self, segment: _UsageSegment, exc: BaseException, *, kind: str | None = None) -> None:
        segment.native_usage_error = type(exc).__name__
        segment.native_usage_kind = kind or (
            str(exc.kind) if isinstance(getattr(exc, "kind", None), str) else None
        )
        status = getattr(exc, "status_code", None)
        segment.native_usage_status = status if type(status) is int else None
        self._native_usage_error = segment.native_usage_error

    async def _collect_native_usage(self, segment: _UsageSegment, shared: dict[str, Any]) -> dict[str, Any]:
        stack = segment.stack
        evidence: dict[str, Any] = {}
        # Nothing more may reach a provider once evidence is taken, and an
        # attempt still running gets a bounded chance to end on its own.
        meter = getattr(stack, "tts_unit_meter", None)
        if meter is not None:
            await meter.settle()
            evidence["tts_native_usage"] = meter.snapshot()
        tracker = getattr(stack, "soniox_usage_tracker", None)
        if tracker is not None:
            mark_unused = getattr(tracker, "mark_component_unused", None)
            if callable(mark_unused):
                # STT is NOT inferred unused from a missing SDK metric: a
                # connected recognizer cancelled before it emitted one would
                # have settled at zero. Its proven absence (no completed
                # session) is recorded on the tracker by the terminal close.
                # TTS request activity is captured at the provider-send
                # seam. An interrupted request can have no SDK metric and
                # still be billable; only proven absence is a free segment.
                activity = getattr(stack, "tts_request_activity", None)
                if not segment.attached or (
                    activity is not None and not activity.started
                ):
                    mark_unused("tts")
            if shared["deadline"] is None:
                shared["deadline"] = tracker.default_deadline()
            evidence["soniox_usage"] = await tracker.collect(deadline=shared["deadline"])
        if segment is not self._segments[0] and self._shared_soniox_stt:
            # The strict settlement contract expects every pinned Soniox
            # dimension, while the one recognizer bills exclusively to the
            # opening segment. This is ownership-derived zero evidence;
            # no phantom STT reference or provider request is registered.
            evidence.setdefault("soniox_usage", {})["stt"] = {
                "input_audio_tokens": 0, "output_text_tokens": 0,
            }
        return evidence

    def _requested_language(self, value: Any) -> str:
        """Map one model-supplied value onto exactly one frozen profile key.

        The control plane freezes bare primary tags -- its own canonicalizer
        strips a region from ``ru-RU`` before anything is published -- so the
        BCP-47 spelling a model most naturally produces for a language names
        an admitted profile rather than an unknown one. Accept it here, where
        the untrusted string arrives, and nowhere else: ``queue`` stays a
        strict membership test so the trusted provider-evidence source keeps
        matching only the exact codes the declaration froze.

        A value that resolves to nothing is rejected WITH the allowed set, so
        the model's next turn can correct itself instead of telling the caller
        their own configured language is unavailable.  That rejection carries
        its own type: a canonical code this call did not publish is safe to
        record and answerable in words, unlike a malformed one.
        """
        # Both separators before ``_language``, which rejects an underscore
        # outright: normalizing after it would leave ``ru_RU`` refused by a
        # branch written to accept it.
        candidate = value.replace("_", "-") if isinstance(value, str) else value
        requested = _language(candidate, field="switch language")
        if requested in self._frozen.profiles:
            return requested
        primary = requested.split("-", 1)[0]
        if primary in self._frozen.profiles:
            return primary
        raise UnsupportedLanguageError(requested, allowed=sorted(self._frozen.profiles))

    async def model_tool(self, raw_arguments: dict[str, object]) -> str:
        """Queue an explicit caller request; never accepts a caller-supplied source.

        Returns without ever awaiting the transition it queues.  That is the
        hang bound on this seam: the model's turn is blocked until this call
        resolves, so anything awaited here is time the caller spends in
        silence with the reply half-generated.  ``_apply`` runs as its own
        task after this returns, and every outcome it can reach ends in a
        spoken turn of its own.
        """
        language = ""
        try:
            language = self._requested_language(raw_arguments.get("language"))
            self._reason(raw_arguments.get("reason"))
            result = self.queue(
                language=language, reason="caller_explicit_request", source="explicit_request"
            )
        except UnsupportedLanguageError as exc:
            # A well-formed target this publication does not carry. Durable
            # rejected evidence: the code was canonicalized before it failed
            # membership, so it can be stored as a language, and the emitted
            # reason is the closed phrase the evidence collector classifies
            # -- never the model-facing sentence, which names the available
            # languages so the reply the model is about to give can too.
            self._emit(
                "language_switch_rejected",
                language=exc.language,
                old_language=self.active_language,
                publication_version=self._frozen.publication_version,
                source="explicit_request",
                reason=_UNSUPPORTED_LANGUAGE_REASON,
            )
            return json.dumps(
                {
                    "ok": False,
                    "status": "rejected",
                    "reason": str(exc),
                    # From the refusal itself, so the set the model is told to
                    # offer the caller is the same set the membership test
                    # just compared against.
                    "available_languages": list(exc.allowed),
                    "say": (
                        "Tell the caller you cannot speak that language on this "
                        "call, name the languages you can speak, and carry on in "
                        f"{self.active_language}."
                    ),
                }
            )
        except LanguageSwitchError as exc:
            # A malformed target cannot safely be stored as a language; emit a
            # rejected operational event without it, and the final collector
            # marks the language timeline partial rather than claiming a
            # measured no-switch call.
            self._emit(
                "language_switch_rejected",
                **({"language": language} if language else {}),
                old_language=self.active_language,
                publication_version=self._frozen.publication_version,
                source="explicit_request",
                reason=str(exc),
            )
            return json.dumps(
                {
                    "ok": False,
                    "status": "rejected",
                    "reason": str(exc),
                    "available_languages": list(self.allowed_languages),
                }
            )
        if result == "terminal":
            # queue() already recorded the terminal rejection. Keep the tool
            # response aligned with that durable terminal outcome.
            return json.dumps({"ok": False, "status": "rejected", "reason": "terminal"})
        if result == "scheduled":
            # ``_apply`` runs after this returns -- awaiting it here would
            # deadlock (see the class docstring), so "scheduled" is a
            # PROMISE, not a completed change. Telling the model that here,
            # in the same reply where it just heard "ok", is what stops it
            # from speaking the new language over the old voice as though
            # the switch had already landed: a failure surfaces later as an
            # explicit correction (see ``_notify_switch_failed``), but this
            # is the model's one chance to hedge in the reply it is about to
            # give right now.
            return json.dumps(
                {
                    "ok": True,
                    "status": result,
                    "note": "not applied yet; you will be told if it fails",
                }
            )
        return json.dumps({"ok": True, "status": result})

    def queue(self, *, language: Any, reason: Any, source: str) -> str:
        language = _language(language, field="switch language")
        reason = self._reason(reason)
        if self._terminal:
            self._emit(
                "language_switch_rejected",
                old_language=self.active_language,
                language=language,
                publication_version=self._frozen.publication_version,
                source=source,
                reason="terminal",
            )
            return "terminal"
        if source not in _SOURCES:
            raise LanguageSwitchError("switch source is not trusted")
        if language not in self._frozen.profiles:
            raise LanguageSwitchError(_UNSUPPORTED_LANGUAGE_REASON)
        # A second request can be a valid return switch while an earlier
        # request is queued (en -> de, then de -> en).  Only skip an already
        # active language when there is no serialized transition ahead of it.
        if language == self.active_language and not self._pending:
            self._emit("language_switch_skipped", language=language, source=source, reason=reason)
            return "already_active"
        if max(len(self._segments), 1 + len(self._attempts)) + len(self._pending) >= _MAX_SEGMENTS:
            raise LanguageSwitchError("switch segment limit reached for this call")
        task = asyncio.create_task(self._apply(language=language, reason=reason, source=source))
        self._pending.add(task)
        self._pending_targets[task] = language
        task.add_done_callback(self._pending_done)
        self._emit("language_switch_queued", language=language, source=source, reason=reason)
        return "scheduled"

    def provider_stable_evidence(self, *, language: Any) -> str:
        """Queue a registry-qualified provider signal without retaining words.

        The adapter hook supplies only a canonical language after its own
        confidence/stability policy.  Raw transcript/tokens never enter switch
        evidence, logs, or the model tool boundary.
        """
        try:
            return self.queue(
                language=language,
                reason="provider_stable_language_evidence",
                source="provider_stable_evidence",
            )
        except LanguageSwitchError as exc:
            if str(exc) != "switch segment limit reached for this call":
                raise
            # STT events must continue to flow after the bounded billing
            # contract is full. The current profile remains active; no 65th
            # segment can be emitted for a server that accepts at most 64.
            self._emit(
                "language_switch_rejected",
                old_language=self.active_language,
                # The only exception admitted here is raised after queue()
                # accepted this canonical target and found the capacity cap.
                # Do not re-parse inside the exception handler.
                language=language,
                source="provider_stable_evidence",
                reason=str(exc),
                publication_version=self._frozen.publication_version,
            )
            return "rejected"

    def transition_pending(self, language: str) -> bool:
        """Whether this exact frozen target still has an in-flight transition."""
        return any(
            target == language and not task.done()
            for task, target in self._pending_targets.items()
        )

    async def drain(self) -> None:
        """Await pending operations before teardown so no switch outlives media."""
        if self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)

    def mark_terminal(self) -> None:
        """Declare that this call is ending, at the moment that is decided.

        Terminal used to begin at teardown, which is several seconds and one
        whole farewell after the agent actually chose to hang up.  In that
        window a queued transition still believed the conversation was live:
        it could cancel the goodbye to reach its boundary, append a billing
        segment to a call already settling, and -- now that an applied switch
        speaks -- schedule a cheerful confirmation after the agent had said
        goodbye.  The end-call action calls this before its farewell is even
        scheduled, so every terminal guard in ``_apply`` sees the truth first.

        Synchronous on purpose: the caller is a hangup path with nothing to
        await, and freezing the settlement boundary is a single assignment.
        """
        self._terminal = True
        # Freeze before any later inspection.  This is intentionally separate
        # from native-counter collection: providers may still emit late
        # metrics to their already-associated segment, but no later
        # transition can alter the ordered billing boundary.
        if self._settlement_boundary is None:
            self._settlement_boundary = self._boundary

    async def drain_for_terminal(self, *, timeout_seconds: float = _TERMINAL_SWITCH_DRAIN_SECONDS) -> None:
        """Bound pending switch work before closing media and settling usage."""
        # Idempotent: an end-call hangup already marked this, and a call that
        # ends any other way is marked here for the first time.
        self.mark_terminal()
        pending = tuple(self._pending)
        if not pending:
            return
        _done, unfinished = await asyncio.wait(pending, timeout=timeout_seconds)
        if unfinished:
            for task in unfinished:
                task.cancel()
            _cancelled, resistant = await asyncio.wait(
                unfinished, timeout=min(timeout_seconds, _TERMINAL_CANCEL_GRACE_SECONDS)
            )
            for task in resistant:
                self._track_abandoned(task)
            self._emit(
                "language_switch_terminal_drain_timeout",
                pending_count=len(unfinished),
                cancellation_resistant_count=len(resistant),
                timeout_seconds=timeout_seconds,
            )

    async def aclose_retired_components(
        self, *, timeout_seconds: float = _RETIRED_COMPONENT_CLOSE_SECONDS
    ) -> None:
        """Release replaced provider objects after their delayed metrics flush.

        The active session owns the current objects.  Only retired adapters
        are closed here -- never the shared simultaneous recognizer, which
        outlives every segment -- and each best-effort close is bounded to
        its own object and cannot turn terminal settlement into a stuck call.
        Tracker credentials remain owned by terminal native-usage collection.
        """
        retired = [segment for segment in self._segments[:-1] if segment.ended_boundary is not None]
        close_tasks: dict[asyncio.Future[Any], str] = {}
        for segment in retired:
            stack = getattr(segment, "stack", None)
            if stack is None:
                continue
            for name in ("tts",):
                component = getattr(stack, name, None)
                close = getattr(component, "aclose", None)
                if callable(close):
                    try:
                        outcome = close()
                        if inspect.isawaitable(outcome):
                            close_tasks[asyncio.ensure_future(outcome)] = name
                    except Exception:
                        pass
        if not close_tasks:
            return
        # One deadline covers every retired adapter. Reserve a small bounded
        # tail for cancellation acknowledgement, rather than multiplying a
        # full timeout by every retired language segment.
        cancellation_grace = min(timeout_seconds / 2, _TERMINAL_CANCEL_GRACE_SECONDS)
        done, unfinished = await asyncio.wait(
            close_tasks, timeout=max(0.0, timeout_seconds - cancellation_grace)
        )
        for task in done:
            self._consume_retired_close_result(task, component=close_tasks[task])
        if not unfinished:
            return
        for task in unfinished:
            task.cancel()
        cancelled, resistant = await asyncio.wait(unfinished, timeout=cancellation_grace)
        for task in cancelled:
            self._consume_retired_close_result(task, component=close_tasks[task])
        for task in unfinished:
            if task in resistant:
                self._track_abandoned(task)
            self._emit(
                "language_switch_retired_component_close_timeout",
                component=close_tasks[task],
                cancellation_resistant=task in resistant,
                timeout_seconds=timeout_seconds,
            )

    def _reason(self, value: Any) -> str:
        reason = " ".join(str(value or "").split())
        if not reason or len(reason) > _MAX_REASON_LENGTH:
            raise LanguageSwitchError("switch reason must be a bounded nonempty string")
        return reason

    async def _apply(self, *, language: str, reason: str, source: str) -> None:
        async with self._lock:
            if self._terminal:
                return
            if language == self.active_language:
                return
            if self._agent is None:
                self._emit(
                    "language_switch_failed",
                    old_language=self.active_language,
                    language=language,
                    publication_version=self._frozen.publication_version,
                    source=source,
                    reason=reason,
                    error_type="RuntimeError",
                )
                return
            target = self._frozen.profiles[language]
            old_profile = self._profile
            old_stack = self._stack
            # Read once, here, while it still describes what is live. The
            # applied event reports it after the swap has replaced it, and
            # re-reading it there would either report the new language as the
            # old one or add another way for the commit to fail.
            old_language = self.active_language
            update_started = False
            # Whether this transition CANCELLED a turn the caller was owed.
            # An explicit request always does: the model's own reply is still
            # being generated when its tool call lands here, and cancelling it
            # is the only safe way to reach a component boundary. Read after
            # the swap to decide who has to speak next -- see
            # ``_notify_switch_applied``.
            preempted_turn = False
            new_stack: Any | None = None
            try:
                collector = self._collector_factory()
                self._usage_boundary += 1
                new_stack = self._build_tts(
                    target.profile, self._frozen.simultaneous_stt, self._usage_boundary
                )
                new_segment = _UsageSegment(
                    language=language,
                    publication_version=self._frozen.publication_version,
                    started_boundary=self._usage_boundary,
                    ended_boundary=self._usage_boundary,
                    collector=collector,
                    stack=new_stack,
                )
                # Keep a constructed handle reachable before any await, even
                # if attachment, cancellation or rollback later aborts it.
                self._attempts.append(new_segment)
                self._attach_component_metrics(new_stack, collector)
                if self._terminal:
                    await self._close_stack_components(new_stack)
                    return
                current = getattr(self._session, "current_speech", None)
                if current is not None:
                    interrupt = getattr(self._session, "interrupt", None)
                    if not callable(interrupt):
                        raise RuntimeError("session cannot cancel obsolete speech")
                    # Marked BEFORE the await, not after it. An interrupt that
                    # cancels the reply and then fails -- the activity closing
                    # underneath it, say -- would otherwise leave a turn taken
                    # from the caller with nothing recorded as owing it back.
                    # The two errors are not symmetric: owing a turn that was
                    # never actually taken costs one extra line of speech,
                    # while the other way round costs the caller the call.
                    preempted_turn = True
                    if self._turn_cut is not None:
                        self._turn_cut()
                    await interrupt(force=True)
                # The public SDK has no atomic idle-and-hold API. This must be
                # the last await before the pipeline-only instruction/TTS
                # update: the deployed Gemini pipeline's instruction update is
                # synchronous-in-effect, followed by synchronous update_options.
                await self._wait_for_idle(timeout_seconds=_SWITCH_IDLE_WAIT_SECONDS)
                if self._terminal:
                    await self._close_stack_components(new_stack)
                    return
                # This preserves the Agent instance, tools, conversation and
                # disclosure/consent prompt context; no handoff replaces it.
                # The publication has one shared LLM/tool identity. Changing
                # it here would turn a language request into a provider/BYOK
                # authority escalation. And the recognizer is one shared
                # simultaneous identity for the call, so
                # only the TTS is replaceable: the switch changes what the
                # agent SAYS, never what it can hear.
                await self._agent.update_instructions(target.profile.instructions)
                if self._terminal:
                    rollback_failed = False
                    try:
                        await self._agent.update_instructions(old_profile.instructions)
                    except Exception:
                        rollback_failed = True
                    await self._close_stack_components(new_stack)
                    self._emit(
                        "language_switch_terminal_aborted",
                        old_language=old_profile.language,
                        language=language,
                        publication_version=self._frozen.publication_version,
                        instruction_rollback_failed=rollback_failed,
                    )
                    return
                update_started = True
                new_segment.attached = True
                self._agent.update_options(tts=new_stack.tts)
                # ``update_options`` is synchronous in the pinned SDK, so the
                # normal event-loop path cannot interleave terminal teardown
                # between it and the bookkeeping below.  Keep this guard for
                # a re-entrant adapter/session shutdown, which otherwise could
                # append a switched segment after terminal settlement froze.
                if self._terminal:
                    rollback_failed = False
                    try:
                        self._agent.update_options(tts=old_stack.tts)
                        await self._agent.update_instructions(old_profile.instructions)
                    except Exception:
                        rollback_failed = True
                    await self._close_stack_components(new_stack)
                    self._emit(
                        "language_switch_terminal_aborted",
                        old_language=old_profile.language,
                        language=language,
                        publication_version=self._frozen.publication_version,
                        instruction_rollback_failed=rollback_failed,
                    )
                    return
            except (Exception, asyncio.CancelledError) as exc:
                rollback_failed = False
                if update_started:
                    try:
                        self._agent.update_options(tts=old_stack.tts)
                        await self._agent.update_instructions(old_profile.instructions)
                    except Exception:
                        rollback_failed = True
                        try:
                            self._session.shutdown(drain=False)
                        except Exception:
                            pass
                if new_stack is not None:
                    await self._close_stack_components(new_stack)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                try:
                    self._emit(
                        "language_switch_failed",
                        old_language=self.active_language,
                        language=language,
                        publication_version=self._frozen.publication_version,
                        source=source,
                        reason=reason,
                        boundary=self._boundary,
                        error_type=type(exc).__name__,
                        error_traceback=_bounded_traceback(exc),
                        terminal_safe_stop=rollback_failed,
                    )
                finally:
                    # Two invariants, and the ``finally`` is what holds the
                    # second. Ordering: the durable evidence lands BEFORE the
                    # best-effort correction, which swallows every exception
                    # it can raise (including from a broken session/agent), so
                    # a failure inside it can never cost the only record that
                    # the switch failed at all. Arrival: the correction runs
                    # even when the emit itself throws. By this point the
                    # caller's turn has usually already been cancelled, so an
                    # exception escaping here would leave them with no words
                    # coming and no way to ask for them -- the very silence
                    # this whole path exists to prevent. A telemetry fault is
                    # then reported by the task-failure hook, which is the
                    # right price: evidence lost, caller still spoken to.
                    await self._notify_switch_failed(
                        requested_language=language, source=source, preempted_turn=preempted_turn
                    )
                return
            # The commit: assignment only, so the switcher's idea of the
            # active language cannot end up disagreeing with the components
            # the caller is actually hearing. Everything that could throw was
            # done above, before the swap.
            self._segments[-1].ended_boundary = self._boundary
            self._boundary = new_segment.started_boundary
            new_segment.ended_boundary = None
            new_segment.committed = True
            self._segments.append(new_segment)
            self._profile = target.profile
            self._stack = new_stack
            # The guard starts HERE, below the commit and above everything
            # fallible that follows it. From this line on every statement runs
            # with the caller's turn cancelled and the new profile already
            # live, so an exception escaping instead of being caught would
            # leave a switched call with nothing speaking through it.
            # The emit reaches a real telemetry sink and can throw. The two
            # invariants: the durable evidence is written first, and the
            # caller gets a turn back whether or not any of this succeeds.
            try:
                self._emit(
                    "language_switch_applied",
                    old_language=old_language,
                    language=language,
                    publication_version=self._frozen.publication_version,
                    source=source,
                    reason=reason,
                    boundary=self._boundary,
                )
            finally:
                self._notify_switch_applied(
                    language=language, source=source, preempted_turn=preempted_turn
                )

    def _notify_switch_applied(
        self, *, language: str, source: str, preempted_turn: bool
    ) -> None:
        """Give back the turn the swap had to cancel, in the new language.

        Reaching a safe component boundary means cancelling whatever the agent
        was saying, and for an explicit request that is always the model's own
        reply to its own ``switch_language`` call -- the tool result is
        returned while the reply is still being generated, so the reply and
        the transition race, and the transition wins by design.  The SDK does
        not regenerate a cancelled reply: an interrupted speech ends its
        generation without ever producing the post-tool turn.  So a switch
        that applied perfectly left the caller with an administratively
        switched, conversationally dead call -- new profile live, billing
        segment open, and not one further word spoken until they hung up.

        So: a caller who ASKED is always answered, and beyond that, whoever
        cancels a turn owes a turn.  What is left owes nothing -- an
        evidence-driven switch that lands while the caller is still speaking
        is followed by their own end of turn, which produces a reply through
        the new stack on its own, and volunteering an unprompted line there
        would talk over them.

        Deliberately synchronous and awaits nothing: ``generate_reply`` only
        schedules, the boundary this needs was already reached by the idle
        wait above, and every await added here is silence the caller hears.
        Best-effort and silent on failure, so a broken correction cannot turn
        an applied switch into an unhandled exception or mask its evidence.
        """
        if self._terminal or self._agent is None:
            return
        if source not in _REQUEST_SOURCES and not preempted_turn:
            return
        if self._speech_is_suspended():
            return
        try:
            if self._speak_owed_turn is not None:
                self._speak_owed_turn()
                return
            self._session.generate_reply(
                instructions=(
                    f"The switch to {language} has completed: you are speaking "
                    f"{language} from now on. Reply in {language}. Briefly "
                    "confirm the change and continue from what the caller last "
                    "said. Do not say the switch failed or is still pending."
                ),
                tool_choice="none",
            )
        except Exception:
            pass

    async def _wait_for_idle(self, *, timeout_seconds: float) -> None:
        """Wait for a completed turn boundary, but never without a deadline.

        ``wait_for_idle`` has no upper bound of its own: a component that
        stops responding simply never reports idle, and this wait happens
        with the switch lock held and the caller's turn already cancelled.
        Timing out raises into ``_apply``'s handler, which restores the live
        profile, records the failure and speaks a correction -- an outcome the
        caller can hear, unlike waiting forever.
        """
        await asyncio.wait_for(self._session.wait_for_idle(), timeout_seconds)

    async def _notify_switch_failed(
        self, *, requested_language: str, source: str, preempted_turn: bool
    ) -> None:
        """Correct an explicit switch the model already told the caller worked.

        The "scheduled" tool result was returned to the model before this
        failure existed -- an ``_apply`` failure is known only after that
        reply is already in flight, so the tool result itself cannot carry
        it (see ``model_tool``). This is the other half: once the reply that
        made the claim has actually finished playing, generate one more
        turn, in whatever language/voice is still genuinely active, so the
        caller is told the switch did not happen instead of being left to
        infer it from an agent that quietly kept answering in one language
        through another one's voice.

        Only an ``explicit_request`` carries a caller-facing claim to
        retract. ``provider_stable_evidence`` never told anyone anything, so
        there is no unprompted "I could not switch" to volunteer for it --
        but if the attempt cancelled a reply on its way to failing, the
        caller is still owed the words that reply was going to be. That turn
        is resumed without mentioning a switch nobody asked for.

        Best-effort and silent on failure: a broken correction must not turn
        one handled switch failure into an unhandled one, and must never mask
        the ``language_switch_failed`` evidence already emitted above.
        """
        explicit = source in _REQUEST_SOURCES
        if self._terminal or self._agent is None or not (explicit or preempted_turn):
            return
        try:
            # Built inside the guard, like the applied path's: it reads
            # ``active_language`` off a caller-supplied profile object, and a
            # sentence that cannot be composed must not cost the turn it was
            # going to be spoken in.
            instructions = (
                f"The requested switch to {requested_language} could not be "
                f"completed. Briefly say so and continue in {self.active_language}."
                if explicit
                else (
                    f"Continue the conversation in {self.active_language}: answer what "
                    "the caller last said. Do not mention any language change."
                )
            )
            try:
                await self._wait_for_idle(timeout_seconds=_CORRECTION_IDLE_WAIT_SECONDS)
            except Exception as wait_failure:
                # A session that will not report idle is exactly the session
                # whose caller is already sitting in silence. Schedule the
                # correction anyway -- the SDK queues it behind whatever is
                # still holding the turn -- rather than adding this wait to
                # the silence and then saying nothing.
                #
                # Every failure, not only the deadline: a wait that RAISES is
                # a stronger broken-session signal than one that times out, so
                # a branch that speaks on the weaker signal and stays silent
                # on the stronger one has it exactly backwards.
                # ``CancelledError`` is a BaseException and still propagates,
                # which is what keeps terminal drain able to stop this.
                try:
                    self._emit(
                        "language_switch_correction_wait_failed",
                        language=requested_language,
                        error_type=type(wait_failure).__name__,
                        timeout_seconds=_CORRECTION_IDLE_WAIT_SECONDS,
                    )
                except Exception:
                    # Everything below this point in the enclosing ``try`` is
                    # skipped by any exception raised here, and what is below
                    # is the correction itself. A note DESCRIBING the silence
                    # must never be the reason for it, so this one diagnostic
                    # is written on a strictly best-effort basis -- unlike the
                    # outcome events in ``_apply``, whose loss is reported.
                    pass
            if self._terminal or self._speech_is_suspended():
                return
            if self._speak_owed_turn is not None:
                # The same repeat the applied path uses: the still-active
                # stack says the cut turn again. There is no switch claim to
                # retract on this agent, which never made one.
                self._speak_owed_turn()
                return
            self._session.generate_reply(instructions=instructions, tool_choice="none")
        except Exception:
            pass

    def _track_abandoned(self, task: asyncio.Future[Any]) -> None:
        """Observe cancellation-resistant provider work without awaiting it."""
        self._abandoned.add(task)

        def done(completed: asyncio.Future[Any]) -> None:
            self._abandoned.discard(completed)
            try:
                completed.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(done)

    def _pending_done(self, task: asyncio.Task[Any]) -> None:
        self._pending.discard(task)
        self._pending_targets.pop(task, None)
        try:
            task.exception()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._emit("language_switch_task_failed", error_type=type(exc).__name__)

    def _consume_retired_close_result(self, task: asyncio.Future[Any], *, component: str) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._emit(
                "language_switch_retired_component_close_failed",
                component=component,
                error_type=type(exc).__name__,
            )

    @staticmethod
    def _attach_component_metrics(
        stack: Any, collector: Any, *, initial_segment: bool = False
    ) -> None:
        """Keep old TTS listeners through their late native metric flush.

        Agent.update_options detaches its own listeners before replacing a
        component.  These segment-owned listeners intentionally remain on the
        old objects, so a cancellation/close metric cannot migrate to a new
        profile or disappear.  The immutable shared LLM and the shared
        simultaneous recognizer are listened to only on the initial segment:
        each is one identity for the whole call, so attaching either after a
        switch would multiply the same native metrics across segments. New
        handles contain only TTS; their listeners attach before the SDK swap.
        """
        component_names = ("stt", "tts", "llm") if initial_segment else ("tts",)
        for component_name in component_names:
            component = getattr(stack, component_name, None)
            subscribe = getattr(component, "on", None)
            if callable(subscribe):
                subscribe("metrics_collected", collector.collect)

    async def _close_stack_components(self, stack: Any) -> None:
        close = getattr(getattr(stack, "tts", None), "aclose", None)
        if not callable(close):
            return
        try:
            outcome = close()
            if not inspect.isawaitable(outcome):
                return
            task = asyncio.ensure_future(outcome)
            done, unfinished = await asyncio.wait(
                {task}, timeout=_RETIRED_COMPONENT_CLOSE_SECONDS
            )
            for completed in done:
                try:
                    completed.result()
                except (asyncio.CancelledError, Exception):
                    pass
            for stalled in unfinished:
                stalled.cancel()
                self._track_abandoned(stalled)
        except Exception:
            pass


class _StableLanguageLatch:
    """The two-uncontested-finals rule, once, for every provider that has one.

    Held by composition rather than inherited, because the two halves of a
    provider-evidence adapter are genuinely different questions and only one
    of them is provider-specific.  WHICH LANGUAGE a final claims is read
    differently per provider -- Soniox names canonical codes in a list, Azure
    names one locale -- while WHETHER A CLAIM IS ENOUGH TO SWITCH is the same
    decision in both, and it is the subtle half: a stale schedule has to be
    cleared without making the caller speak the old language again, and a
    transition already in flight must not restart the count.  Two copies of
    that would be two places to fix the next thing found wrong with it.

    The public adapters keep their own construction checks and their own
    reading of a final; both hand the answer here.
    """

    def __init__(self, switcher: ControlledLanguageSwitcher) -> None:
        self._switcher = switcher
        self._candidate = ""
        self._count = 0
        self._scheduled = ""

    def observe(self, language: str) -> str | None:
        """Return a scheduled target only after two qualified final signals."""
        if language not in self._switcher._frozen.profiles:
            return self.reset()
        if language == self._switcher.active_language:
            self._scheduled = ""
            return self.reset()
        if language == self._scheduled:
            if self._switcher.transition_pending(language):
                return None
            # The prior transition failed or completed without reaching this
            # target. Do not require the caller to speak the old language to
            # clear a stale evidence latch; begin a fresh two-final sequence.
            self._scheduled = ""
        if language != self._candidate:
            self._candidate, self._count = language, 1
            return None
        self._count += 1
        if self._count < 2:
            return None
        self._candidate, self._count = "", 0
        if self._switcher.provider_stable_evidence(language=language) == "scheduled":
            self._scheduled = language
            return language
        return None

    def reset(self) -> None:
        self._candidate, self._count = "", 0
        return None


class SonioxStableLanguageEvidence:
    """Require two uncontested final Soniox language observations to switch.

    The Soniox 1.7.1 adapter exposes ``SpeechData.source_languages`` on final
    STT alternatives.  This adapter consumes that typed metadata only: caller
    text, words, confidence guesses and any other provider's metadata never
    enter this decision. A switch means only that two consecutive final
    provider source-language detections agreed; it is not a semantic or
    minimum-duration proof about the caller's words. A multi-language final
    or unknown value resets the candidate.
    """

    _STRICT_OPTIONS = {
        "language_hints",
        "language_hints_strict",
        "enable_language_identification",
    }

    def __init__(self, switcher: ControlledLanguageSwitcher) -> None:
        frozen = switcher._frozen
        declaration = frozen.simultaneous_stt
        options = declaration.get("adapter_options")
        mode = declaration.get("mode")
        options_valid = (
            isinstance(options, Mapping)
            and options.get("enable_language_identification") is True
            and (
                (mode == "language_identification" and set(options) == {"enable_language_identification"})
                or (
                    mode == "strict_language_hints"
                    and set(options) == self._STRICT_OPTIONS
                    and options.get("language_hints_strict") is True
                )
            )
        )
        if (
            mode not in {"language_identification", "strict_language_hints"}
            or str(declaration.get("provider") or "").casefold() != "soniox"
            or str(declaration.get("registry_name") or "").casefold() != "soniox"
            or not options_valid
        ):
            raise LanguageSwitchError("Soniox stable evidence is not admitted by the frozen declaration")
        self._latch = _StableLanguageLatch(switcher)

    def observe_source_languages(self, source_languages: Any) -> str | None:
        """Return a scheduled target only after two qualified final signals."""
        if not isinstance(source_languages, (list, tuple)):
            return self._latch.reset()
        observed = {str(value or "").strip().casefold() for value in source_languages}
        observed.discard("")
        # Exactly one allowed canonical language is trustworthy.  Do not map
        # locales or infer language from text: the control plane froze hints
        # using exactly the canonical codes it admitted.
        if len(observed) != 1:
            return self._latch.reset()
        return self._latch.observe(next(iter(observed)))

    async def filter(self, events: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Observe final SpeechData metadata while preserving STT event flow."""
        async for event in events:
            event_type = str(getattr(getattr(event, "type", None), "value", getattr(event, "type", "")))
            if event_type == "final_transcript":
                for alternative in getattr(event, "alternatives", ()):
                    self.observe_source_languages(getattr(alternative, "source_languages", None))
            yield event


class ContinuousLanguageIdentificationEvidence:
    """The same two-final rule over a provider that answers in LOCALES.

    Keyed on the frozen declaration's MODE, never on a provider name.  The
    mode describes a mechanism -- give the recogniser a candidate list, get
    back which of them it heard -- and a second provider offering that
    mechanism should reach this adapter without editing it. The provider
    identity is already pinned by the control plane and re-checked by the
    stack builder; repeating it here would only add a place to forget.

    The reverse map is built from the frozen candidate list and nothing else.
    That is the whole reason the declaration carries the locales rather than
    letting the worker resolve them: ``et-EE`` means the profile published as
    ``et`` on THIS call because the envelope says so, not because a tag can be
    truncated at the hyphen. A locale the frozen set does not name -- a
    regional variant, a script-tagged form, or the empty string a provider
    returns when it could not decide -- is not evidence and resets the count.
    """

    _REQUIRED_OPTIONS = {"language_id_mode", "candidate_locales"}

    def __init__(self, switcher: ControlledLanguageSwitcher) -> None:
        declaration = switcher._frozen.simultaneous_stt
        options = declaration.get("adapter_options")
        if (
            declaration.get("mode") != "continuous_language_identification"
            or not isinstance(options, Mapping)
            or set(options) != self._REQUIRED_OPTIONS
            or options.get("language_id_mode") != "continuous"
            or not isinstance(options.get("candidate_locales"), Mapping)
            or len(options["candidate_locales"]) < 2
        ):
            raise LanguageSwitchError(
                "language identification evidence is not admitted by the frozen declaration"
            )
        languages: dict[str, str] = {}
        for raw_language, raw_locale in options["candidate_locales"].items():
            language = _language(raw_language, field="follow_caller candidate language")
            locale = str(raw_locale or "").strip().casefold()
            if not locale or locale in languages or language not in switcher._frozen.profiles:
                # A candidate naming two languages with one tag cannot be
                # resolved, and one naming a language this call did not freeze
                # a profile for could only ever schedule a switch that the
                # switcher must refuse. Both are contract faults in the
                # envelope, so fail before media rather than at the first
                # final.
                raise LanguageSwitchError("follow_caller candidate locales are not a usable identification map")
            languages[locale] = language
        self._languages = languages
        self._latch = _StableLanguageLatch(switcher)

    def observe_detected_locale(self, locale: Any) -> str | None:
        """Return a scheduled target only after two qualified final signals."""
        language = self._languages.get(str(locale or "").strip().casefold())
        if language is None:
            return self._latch.reset()
        return self._latch.observe(language)

    async def filter(self, events: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Observe final SpeechData metadata while preserving STT event flow."""
        async for event in events:
            event_type = str(getattr(getattr(event, "type", None), "value", getattr(event, "type", "")))
            if event_type == "final_transcript":
                for alternative in getattr(event, "alternatives", ()):
                    self.observe_detected_locale(getattr(alternative, "language", None))
            yield event


class LanguageMultiEvidence:
    """The same two-final rule over a recogniser that names no candidates.

    A ``language_multi`` construction sends the provider one keyword and no
    language list, and the pinned Deepgram plugin (1.8.8,
    ``live_transcription_to_speech_data``) sets each final's
    ``SpeechData.language`` to the first entry of the result's own
    ``languages``: the language most of that final's words were in. Measured
    on a recorded outbound call and the phone-menu corpus: every final
    carried it and it equalled the word-level majority in 563 of 563 finals.
    Without a provider ``languages`` field the plugin leaves the configured
    ``multi`` there instead, which names no profile and so resets.

    The reported code is matched EXACTLY against this call's frozen profiles
    -- no region truncation, no alias table -- because there is no candidate
    map to say which profile a regional or script-tagged tag would mean. An
    unknown, empty or unmatched value resets the count like any other
    non-answer.

    Not yet routed by :data:`PROVIDER_LANGUAGE_EVIDENCE`: turning automatic
    switching on for these calls is a separate change.
    """

    def __init__(self, switcher: ControlledLanguageSwitcher) -> None:
        declaration = switcher._frozen.simultaneous_stt
        options = declaration.get("adapter_options")
        if (
            declaration.get("mode") != "language_multi"
            or not isinstance(options, Mapping)
            or dict(options) != {"language": "multi"}
        ):
            raise LanguageSwitchError("language_multi evidence is not admitted by the frozen declaration")
        self._latch = _StableLanguageLatch(switcher)

    def observe_final_language(self, language: Any) -> str | None:
        """Return a scheduled target only after two qualified final signals."""
        code = str(language or "").strip().casefold()
        if not code:
            return self._latch.reset()
        return self._latch.observe(code)

    async def filter(self, events: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Observe final SpeechData metadata while preserving STT event flow."""
        async for event in events:
            event_type = str(getattr(getattr(event, "type", None), "value", getattr(event, "type", "")))
            if event_type == "final_transcript":
                for alternative in getattr(event, "alternatives", ()):
                    self.observe_final_language(getattr(alternative, "language", None))
            yield event


#: The frozen mode decides which provider-evidence adapter a call gets, and a
#: mode with no entry gets none: the explicit-request path still works, and
#: automatic switching stays off rather than being approximated. Deepgram's
#: ``language_multi`` is not routed yet: the pinned plugin does report a
#: per-final language (:class:`LanguageMultiEvidence` reads it), and enabling
#: automatic switching on it is its own change.
PROVIDER_LANGUAGE_EVIDENCE: dict[str, Any] = {
    "language_identification": SonioxStableLanguageEvidence,
    "strict_language_hints": SonioxStableLanguageEvidence,
    "continuous_language_identification": ContinuousLanguageIdentificationEvidence,
}


__all__ = [
    "ContinuousLanguageIdentificationEvidence",
    "ControlledLanguageSwitcher",
    "FollowCallerProfiles",
    "FrozenLanguageProfile",
    "LanguageMultiEvidence",
    "LanguageSwitchError",
    "PROVIDER_LANGUAGE_EVIDENCE",
    "SonioxStableLanguageEvidence",
    "UnsupportedLanguageError",
    "switch_language_tool_schema",
]

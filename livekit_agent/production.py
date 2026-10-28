# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Production-facing LiveKit semantic bindings.

This module is intentionally SDK-light.  The control plane owns the resolved
semantic configuration and MCP allowlist; the LiveKit adapter only validates
and translates those values into provider/plugin and model-tool objects.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import math
import os
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
)

try:  # package import in tests; flat /app import in the pinned worker image
    from .session_state_file import (
        SessionFileUnreadable,
        locked_session_file,
        publish_session_file,
        read_session_file,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from session_state_file import (  # type: ignore[no-redef]
        SessionFileUnreadable,
        locked_session_file,
        publish_session_file,
        read_session_file,
    )

try:  # package import in tests; flat /app import in the pinned worker image
    from .demo import DemoProfile, TTSComponent, build_tts_component
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from demo import DemoProfile, TTSComponent, build_tts_component

try:
    from .conversation_engines_generated import ELEVEN_REALTIME_MODEL, ENGINE_DECLARATIONS, engine_deferred_tool_conflicts
except ImportError:  # flat worker image
    from conversation_engines_generated import ELEVEN_REALTIME_MODEL, ENGINE_DECLARATIONS, engine_deferred_tool_conflicts

try:
    # The pinned SDK is installed only in the worker image. The end_call model
    # tool annotates one parameter with ``RunContext`` so the SDK injects the
    # call's tool-call id and speech handle (its ``get_type_hints`` resolves
    # the annotation string in THIS module's globals); the control plane and
    # its tests import this module without the SDK, where the name is only a
    # placeholder that is never injected.
    from livekit.agents.voice import RunContext
except ImportError:  # pragma: no cover - the SDK-less control-plane import
    RunContext = Any  # type: ignore[assignment,misc]

try:
    from livekit.agents.llm import ToolError
except ImportError:  # pragma: no cover - the SDK-less control-plane import

    class ToolError(Exception):  # type: ignore[no-redef]
        """Stand-in for the SDK's ToolError where the SDK is absent."""


try:  # package import in tests; flat /app import in the pinned worker image
    from .sip_answer_gate import (
        ALREADY_ACTIVE,
        ANSWERED,
        DISCONNECTED,
        NOT_APPLICABLE,
        NOT_REPORTED,
        TIMED_OUT,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from sip_answer_gate import (  # type: ignore[no-redef]
        ALREADY_ACTIVE,
        ANSWERED,
        DISCONNECTED,
        NOT_APPLICABLE,
        NOT_REPORTED,
        TIMED_OUT,
    )

try:  # package import in tests; flat /app import in the pinned worker image
    from .capabilities.delivery import normalize_delivery_profile
    from .capabilities.registry import (
        VoiceCapabilityError,
        load_registry,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from capabilities.delivery import normalize_delivery_profile
    from capabilities.registry import (
        VoiceCapabilityError,
        load_registry,
    )

try:  # package import in tests; flat /app import in the pinned worker image
    from .spoken_language import greeting_fallback_message, with_language_directive
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from spoken_language import greeting_fallback_message, with_language_directive

try:  # package import in tests; flat /app import in the pinned worker image
    from .duplex_opening_hold import (
        JUDGED_OUTCOMES as DUPLEX_OPENING_JUDGED,
        OUTCOMES as DUPLEX_OPENING_OUTCOMES,
    )
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from duplex_opening_hold import (  # type: ignore[no-redef]
        JUDGED_OUTCOMES as DUPLEX_OPENING_JUDGED,
        OUTCOMES as DUPLEX_OPENING_OUTCOMES,
    )

try:  # package import in tests; flat /app import in the pinned worker image
    from .sip_call_statistics import attach_sip_call_statistics
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from sip_call_statistics import attach_sip_call_statistics

try:  # package import in tests; flat /app import in the pinned worker image
    from .error_tracking import capture_exception
except ImportError:
    from error_tracking import capture_exception  # type: ignore[no-redef]

try:  # package import in tests; flat /app import in the pinned worker image
    from .control_plane import (
        ARTIFACT_UPLOAD_PENDING,
        ArtifactUploadFailed,
        artifact_retry_params,
        next_artifact_attempt_at,
    )
except ImportError:
    from control_plane import (  # type: ignore[no-redef]
        ARTIFACT_UPLOAD_PENDING,
        ArtifactUploadFailed,
        artifact_retry_params,
        next_artifact_attempt_at,
    )

try:  # package import in tests; flat /app import in the pinned worker image
    from .navigation_entry import (
        ENDER_NO_CALLER_SPEECH,
        LIMIT_DEADLINE,
        LIMIT_HOLD,
        LIMIT_NAVIGATION,
        DETAIL_DATA_NOT_AUTHORISED,
        DETAIL_DURING_HOLD,
        DETAIL_MENU_NO_PATH,
        DETAIL_NAVIGATION_DETECTOR_LOST,
        DETAIL_RECORDED_MESSAGE_ONLY,
        NavigationPolicy,
    )
except ImportError:
    from navigation_entry import (  # type: ignore[no-redef]
        ENDER_NO_CALLER_SPEECH,
        LIMIT_DEADLINE,
        LIMIT_HOLD,
        LIMIT_NAVIGATION,
        DETAIL_DATA_NOT_AUTHORISED,
        DETAIL_DURING_HOLD,
        DETAIL_MENU_NO_PATH,
        DETAIL_NAVIGATION_DETECTOR_LOST,
        DETAIL_RECORDED_MESSAGE_ONLY,
        NavigationPolicy,
    )

try:
    from .release_canary_contract import worker_heartbeat_metadata
except ImportError:
    try:
        from release_canary_contract import worker_heartbeat_metadata
    except (
        ModuleNotFoundError
    ) as exc:  # The public self-host export has no platform canary.
        if exc.name != "release_canary_contract":
            raise

        def worker_heartbeat_metadata(
            _environ: Mapping[str, str],
        ) -> dict[str, dict[str, str]]:
            return {}


try:
    from .voice_configuration import ProductionConfigurationError, refuse_freeform_workflow
except ImportError:
    from voice_configuration import ProductionConfigurationError, refuse_freeform_workflow


class PeriodicWorkerHeartbeat:
    """A bounded, idempotent daemon loop for service- or job-lifetime health.

    It sends immediately, then waits interruptibly.  The caller owns which
    worker identity the callback reports, allowing a service-level heartbeat
    to coexist with per-job-process heartbeats without a shared row.
    """

    def __init__(
        self,
        *,
        send: Callable[[], object],
        interval_seconds: float,
        on_failure: Callable[[BaseException], None],
    ) -> None:
        self._send = send
        self._interval_seconds = max(10.0, interval_seconds)
        self._on_failure = on_failure
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> bool:
        """Start exactly once; return false when the same loop is already live."""
        with self._lock:
            if self._started:
                return False
            self._started = True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="skysay-worker-heartbeat",
                daemon=True,
            )
            self._thread.start()
            return True

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        """Cancel promptly without allowing a heartbeat thread to block shutdown."""
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout_seconds))
        with self._lock:
            # A blocked HTTP request is allowed to outlive a bounded shutdown
            # join because the thread is a daemon. Keep it marked live so a
            # concurrent restart cannot publish a duplicate worker row.
            if thread is not None and thread.is_alive():
                return
            self._started = False
            self._thread = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._send()
                except (
                    BaseException
                ) as exc:  # A health signal must never kill the voice worker.
                    try:
                        self._on_failure(exc)
                    except BaseException:
                        pass
                if self._stop.wait(self._interval_seconds):
                    return
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._started = False
                    self._thread = None


def worker_heartbeats_enabled(environ: Mapping[str, str]) -> bool:
    """Whether this runtime should use the global worker-heartbeat surface.

    Production keeps the default enabled for release/readiness evidence. The
    self-host compose disables it because its least-privilege inbound worker
    credential deliberately excludes the global ``worker_jobs:*`` authority.
    """
    value = str(environ.get("SKYSAY_WORKER_HEARTBEAT_ENABLED", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def service_heartbeat_environ(environ: Mapping[str, str]) -> dict[str, str]:
    """Give the idle service heartbeat its own stable row from job children."""
    service_environ = dict(environ)
    worker_base = str(
        service_environ.get("SKYSAY_WORKER_ID") or "livekit-agent"
    ).strip()
    service_environ["SKYSAY_WORKER_ID"] = f"{worker_base}-service"
    return service_environ


def inbound_sip_identity(
    attributes: dict[str, str], participant_identity: str
) -> tuple[str, str, str]:
    """Return caller, called DID, and carrier call id from LiveKit SIP metadata."""

    caller = str(attributes.get("sip.phoneNumber") or "").strip()
    called = str(attributes.get("sip.trunkPhoneNumber") or "").strip()
    if caller.isdigit():
        caller = "+" + caller
    if called.isdigit():
        called = "+" + called
    if not called.startswith("+") or not called[1:].isdigit():
        raise ValueError("inbound SIP participant omitted a valid sip.trunkPhoneNumber")
    provider_call_id = str(
        attributes.get("sip.callIDFull")
        or attributes.get("sip.callID")
        or participant_identity
    ).strip()
    if not provider_call_id:
        raise ValueError("inbound SIP participant omitted its call identity")
    return caller or "unknown", called, provider_call_id


class ToolSession(Protocol):
    config: dict[str, Any]

    def tool_definitions(self) -> list[dict[str, Any]]: ...

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    def report_transfer_outcome(self, outcome: str) -> dict[str, Any]: ...


def livekit_worker_heartbeat_payload(
    environ: dict[str, str] | os._Environ[str],
    *,
    process_id: int,
    worker_type: str = "livekit_agent",
) -> dict[str, Any]:
    worker_base = str(environ.get("SKYSAY_WORKER_ID") or "livekit-agent").strip()
    metadata: dict[str, Any] = {
        "media_stack": "livekit",
        "role": "semantic_agent_process",
    }
    if __package__:
        from .calendar_tools import CALENDAR_TOOL_DECLARATIONS
    else:  # the image imports the same required source as a flat module
        from calendar_tools import CALENDAR_TOOL_DECLARATIONS
    metadata["agent_tool_names"] = sorted(CALENDAR_TOOL_DECLARATIONS)
    metadata.update(worker_heartbeat_metadata(environ))
    return {
        "worker_id": f"{worker_base}-{process_id}",
        "worker_type": worker_type,
        "status": "ready",
        "metadata": metadata,
    }


def post_livekit_worker_heartbeat(
    transport: Any,
    environ: dict[str, str] | os._Environ[str],
    *,
    process_id: int,
    worker_type: str = "livekit_agent",
) -> dict[str, Any]:
    return transport.request(
        "POST",
        "/v1/worker-heartbeats",
        livekit_worker_heartbeat_payload(
            environ, process_id=process_id, worker_type=worker_type
        ),
    )


# STT registry_name -> the historical models the worker can construct.
# Deepgram nova-3 remains the default English/Estonian path. Google Chirp 2 is
# the remaining Google streaming model (V1 command_and_search is gone).
# The capability artifact's ``declared_stack_languages`` is what actually
# gates language, not this set.
_LEGACY_STT_REGISTRY_MODELS: dict[str, frozenset[str]] = {
    "deepgram": frozenset({"nova-3"}),
    "google": frozenset({"chirp_2"}),
}
_LEGACY_STT_DEFAULT_MODELS = {"deepgram": "nova-3", "google": "chirp_2"}
# Sinhala once had a named STT model set and a named TTS adapter set here, and
# both are gone. Tamil never had them, and the reason given for that -- the
# artifact's own declaration is the complete admission answer, and a
# handwritten provider list beside it is the duplication this contract exists
# to remove -- was always true of Sinhala too. It is not a relaxation: over
# every catalogue STT row crossed with every catalogue TTS row, for every
# runtime language, disabling those sets flipped no configuration between
# accepted and refused, here or in the control plane's copy of these rules.
# What they still did was refuse a second time, in a language-shaped way, what
# the declaration checks below already refuse for every other language.
# TTS registry_names this worker knows how to build. ElevenLabs is the default
# English/Estonian path; ``gemini_tts`` (Gemini-TTS) is the si-LK (Sinhala) and
# ta-LK (Tamil) path -- ElevenLabs supports neither.
_LEGACY_TTS_REGISTRIES = frozenset({"elevenlabs", "gemini_tts"})


# LiveKit's session-level VAD interruption default is 0.50 s.  Keep that
# value at this production-only seam rather than changing ``build_demo_stack``:
# the latter is also used by the browser demo and browser-test paths, whose
# short-turn behaviour is a separate, deliberately compatible surface.
#
# This is the DEFAULT a managed phone call runs when the agent sets no
# explicit ``min_speech_ms`` and no preset, in every language.
# It is not the floor an explicit value is raised to; that is
# ``MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS`` below.
#
# What the number rests on, corrected 2026-09-15 from offline Silero v6.1
# replays of the production room-side tracks.  It was set on 2026-08-28 (#505)
# citing seven false interruptions on the first Sinhala campaign call
# (``call_MPXHrjeG3WhhMAYQ``) on caller bursts logged at 147-221 ms.  Those
# bursts straddle the 0.20 s VAD floor then in force -- on replay they are
# 128/224/192/256 ms of hangover-inclusive VAD ``speech_duration`` over actual
# agent playout, none of them transcribed -- and every one of the seven
# RESUMED (``false_interruptions_resumed: 7`` in that call's telemetry): they
# were the pause-and-resume described below, not cuts.  What actually killed
# the greeting were transcribed fragments in the greeting-synthesis window,
# before playout had started, through the transcript path
# (``on_interim_transcript``/``on_final_transcript`` ->
# ``_interrupt_by_audio_activity``) that #472 had closed with
# ``CallerTranscriptGate`` the day before and that ``OpeningTurnProtection``
# now covers for the greeting window.  A higher VAD floor is not what fixed
# that call.
#
# Across the 31-call Sinhala batch of 2026-09-04 (10 calls with agent audio,
# 67 s of agent speech), caller bursts over agent speech reached 0.30 s four
# times (one transcribed) and 0.50 s twice (none transcribed).  The floor
# therefore barely moves the wordless-burst rate.  What bounds a wordless burst
# today is the stack's ``resume_false_interruption`` with a
# ``false_interruption_timeout`` of 1.0 s (``demo.py``), which turns it into a
# pause-and-resume of at most a second rather than a cut.  The half-second
# stays as the default because nothing measured argues for moving a value no
# customer chose; it is no longer described as the threshold below which a
# phone call cuts itself off, because the replays do not show that.
MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS = 0.50

# The floor an EXPLICIT per-agent ``min_speech_ms`` is raised to on a managed
# phone call, in every language.  0.15 s is the value the ``high`` preset
# has run on English calls since the presets existed, so it asserts no
# threshold the platform has not already shipped.  The same batch that leaves
# the default alone is what moved this: genuine transcribed Sinhala caller
# turns have a median VAD duration of 0.29 s (5 of 9 under 0.30 s), so raising
# an explicit 300 ms to 0.50 s structurally excluded one-word answers in that
# language, and on an English demo (2026-09-14) three attention calls of the
# agent's name at 288/320/384 ms and a pickup "Hello?" at 384 ms were refused
# by the same clamp.  Values under this floor are still raised, and
# ``managed_pstn_min_speech_clamp`` still reports it per call.
#
# Not the effective English figure on its own.  A call with a word
# requirement also runs ``CallerTranscriptGate``, whose ``_spoken_seconds``
# discounts the VAD's 250 ms end-of-speech hangover
# (``VAD_END_OF_SPEECH_SILENCE_SECONDS``) on top of the SDK's own
# ``speech_duration`` test.  That discount is unchanged by this floor.
MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS = 0.15


def _explicit_min_speech_seconds(min_speech_ms: int | None) -> float | None:
    """An explicit ``min_speech_ms`` in seconds, never below the explicit floor.

    The floor is :data:`MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS`
    (0.15 s), not the 0.50 s default.  Until 2026-09-15 every explicit value
    was raised to the default, on the account that 147 ms of caller noise had
    cut a Sinhala agent through the VAD path and that a lower value would let
    a customer re-open that defect on their own line.  The replays behind the
    two constants above say otherwise: the seven logged bursts were resumed
    pauses rather than cuts, the greeting died through the transcript path,
    which is gated separately now, and raising every explicit value to 0.50 s
    was refusing
    genuine one-word Sinhala answers (median 0.29 s) and English attention
    calls of 288-384 ms.  A customer who types a number under 150 ms still
    gets 150 ms, and the clamp is still reported once per call.

    Browser and web calls never reach this function: they are not managed
    phone calls, so their sub-floor thresholds are untouched.
    """

    if min_speech_ms is None:
        return None
    return max(min_speech_ms / 1_000, MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS)


def managed_pstn_min_speech_clamp(
    realtime_config: Mapping[str, Any],
) -> tuple[float, float] | None:
    """``(requested, effective)`` when a call's ``min_speech_ms`` was raised.

    ``None`` when nothing was clamped -- no explicit request, a request already
    at or above the floor, or a call this policy does not govern. Read by the
    worker so the clamp is logged once per call rather than silently applied;
    a customer who set 120 ms and hears 150 ms deserves to find out why from
    the call's own evidence.
    """

    if not is_managed_pstn_call(realtime_config):
        return None
    raw_tuning = realtime_config.get("turn_taking")
    tuning = raw_tuning if isinstance(raw_tuning, Mapping) else {}
    min_speech_ms = _bounded_turn_taking_int(
        tuning.get("min_speech_ms"),
        low=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][0],
        high=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][1],
    )
    if min_speech_ms is None:
        return None
    requested = min_speech_ms / 1_000
    effective = _explicit_min_speech_seconds(min_speech_ms)
    if effective is None or effective <= requested:
        return None
    return (requested, effective)


def _endpointing_with_ordered_pair(endpointing: dict[str, Any]) -> dict[str, Any]:
    """Keep ``max_delay`` at or above ``min_delay``.

    ``min_delay`` is how long the session waits for silence before it will
    commit a turn at all; ``max_delay`` is how long it will wait for the
    trained detector before committing anyway.  A ``silence_hold_ms`` above the
    stack's ``max_delay`` inverts the pair, so the session would wait LONGER
    for a confident endpoint than for an unlikely one.  Raise the ceiling to
    the floor rather than clamping the floor: the customer asked to hold
    silence that long, and silently ignoring the setting is the failure mode
    the per-agent contract exists to prevent.  Latency grows only for an agent
    that opted in.

    Applied to every managed call.  An explicit ``silence_hold_ms`` or a
    preset can move ``min_delay`` in any language, so a VAD-only call can
    invert the pair too.  The
    write is inert where no detector runs -- ``max_delay`` is consulted only
    inside the detector branch -- but leaving a stored pair inverted would
    misdescribe the session to anyone reading it.
    """

    minimum_delay = endpointing.get("min_delay")
    maximum_delay = endpointing.get("max_delay")
    if (
        isinstance(minimum_delay, (int, float))
        and isinstance(maximum_delay, (int, float))
        and not isinstance(minimum_delay, bool)
        and not isinstance(maximum_delay, bool)
    ):
        endpointing["max_delay"] = max(float(maximum_delay), float(minimum_delay))
    return endpointing

# The per-agent preset (``interruption_sensitivity``) at the only self-hosted,
# per-session interruption control LiveKit 1.7.1 exposes:
# ``interruption.min_duration``.  ONE table for every managed phone call in
# every language (founder decision 2026-09-15: "make this universal, keep it
# super simple").  It was English-only on the ground that the values were
# English-measured; the 2026-09-15 Sinhala replays showed the floor
# is a property of the line and the VAD rather than of the language.  A
# high-sensitivity agent opts back into the short threshold the shared demo
# stack historically used; low requires a little more sustained caller
# speech.  An explicit ``min_speech_ms`` always wins over this preset.
_SENSITIVITY_MIN_DURATION_SECONDS = {
    "low": 0.75,
    "balanced": MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS,
    "high": 0.15,
}

# The end-of-turn HOLD (the fixed endpointing ``min_delay``) is TWO rows,
# keyed on the built stack's endpointing mode, because the number means a
# different thing in each.  With a trained turn detector it is the wait when
# the model judges the turn finished; with VAD-only endpointing it is the
# entire end-of-turn decision, and the stack's own default is 0.8 s
# (``demo.py``).  Handing a VAD-only language the model-language pad would
# commit a Sinhala turn 0.26 s into a mid-sentence pause, which is why the
# row exists.  An explicit ``silence_hold_ms`` always wins over this preset.
#
# A managed phone call holds longer than the shared stack's browser-test
# values (0.2 s confident / 0.6 s unsure).  On a narrowband phone line the
# detector judges an ordinary mid-message pause -- after a name, before a
# number -- a finished turn, and 0.2 s plus the 0.25 s speech-end silence
# committed the caller's half sentence: the agent answered a fragment and
# talked over the rest.  So the detector row carries ``balanced`` too, and a
# call with no preset (which reads as balanced) takes it.  The VAD-only row
# needs no balanced entry: the stack's own 0.8 s already is one.
_SENSITIVITY_MODEL_HOLD_SECONDS = {"balanced": 0.6, "low": 0.9, "high": 0.26}
_SENSITIVITY_VAD_ONLY_HOLD_SECONDS = {"low": 1.10, "high": 0.60}
# The detector's other bound, ``max_delay``: how long the session waits when
# the model judges the turn probably NOT finished before committing anyway.
# Every preset sets it, ``low`` included -- it used to stay at the stack's
# 0.6 s, so the patient choice still cut in on any pause the model was unsure
# about.  Inert where no detector runs (``_endpointing_with_ordered_pair``).
_SENSITIVITY_MODEL_MAX_HOLD_SECONDS = {"balanced": 1.2, "low": 1.8, "high": 0.6}


def _sensitivity_hold_table(turn_handling: Mapping[str, Any]) -> Mapping[str, float] | None:
    """The preset HOLD row this built session takes; ``None`` if the stack said nothing.

    Read from the built ``turn_handling`` the runtime is handed rather than
    from the call's language.  ``demo.py`` sets ``turn_detection`` to
    ``"vad"`` exactly when the language is outside its
    ``TURN_DETECTOR_SUPPORTED_LANGUAGES`` and to the loaded detector
    otherwise, so this is that decision one step later, with no second copy
    of the set to drift.  It is also how the floor is already read
    (:func:`configured_interruption_min_duration`).  A stack that supplied
    no ``turn_detection`` at all gets ``None``, and a preset then leaves its
    hold untouched rather than guessing which row it is.
    """

    mode = turn_handling.get("turn_detection")
    if mode is None:
        return None
    if isinstance(mode, str) and mode.strip().lower() == "vad":
        return _SENSITIVITY_VAD_ONLY_HOLD_SECONDS
    return _SENSITIVITY_MODEL_HOLD_SECONDS
_TURN_TAKING_INT_BOUNDS = {
    "silence_hold_ms": (100, 2_000),
    "min_speech_ms": (80, 2_000),
}

# The customer-facing ceiling on ``turn_taking.min_interruption_words``.  Five
# words is already most of a sentence and more than a caller says before they
# expect the agent to stop; anything above it would be a way to switch barge-in
# off while the response still called it enabled.  Mirrored by the control
# plane's own bound in ``services/agent_tuning.py`` -- the worker re-validates
# because historic or hand-written agent metadata reaches it directly.
MIN_INTERRUPTION_WORDS_BOUNDS = (0, 5)

# What a managed ENGLISH phone call requires when its agent has configured
# nothing.  The knob shipped at 0 -- duration alone -- because no measurement
# existed yet to choose between them; 1 is now the default on the one class of
# call where the evidence supports it.
#
# One word, and only English, for three reasons that have to hold together:
#
# * the duration floor alone cannot tell a caller apart from a television, a
#   cough or a car.  Three of three cuts on ``call_MPXHrjeG3WhhMAYQ`` were
#   wordless bursts, and seven pauses on that call resumed ~1 s later because
#   nothing the caller said had actually been said;
# * one word is the smallest requirement that changes that outcome, and it is
#   the requirement the founder's own acceptance sentence names ("at least one
#   recognized word").  Two would refuse "stop";
# * it is only ENFORCEABLE where the transcriber answers inside the decision
#   window -- Deepgram ``nova-3`` today.  ``effective_min_interruption_words``
#   is where that veto lives, and this default passes through it like any
#   customer-set value, so a language whose STT row says otherwise runs
#   duration-only exactly as it does now.
#
# An agent that wants the old behaviour writes ``min_interruption_words: 0``,
# which the control plane now stores rather than normalizing away -- see
# ``services/agent_tuning.py``.  Without that, "0 disables it" would have
# become false the moment this default moved off zero.
MANAGED_ENGLISH_DEFAULT_MIN_INTERRUPTION_WORDS = 1

# The silence the VAD waits through before it reports end-of-speech, which the
# managed worker fixes at 0.25 s by prewarming ONE ``silero.VAD.load`` per
# process (``agent/production_agent.py`` ``setup_process``) and handing that
# same object to every session -- ``load_vad`` there ignores the requested
# value precisely because the object is shared across concurrent calls and must
# not be retuned per tenant.
#
# It is a CORRECTION, not a threshold. ``user_state`` flips to ``listening``
# from the callback that fires after this silence has already elapsed, so the
# interval between the two state changes is longer than the caller actually
# spoke -- by about this much. Timing a speech run from those two events and
# comparing the raw result against the interruption floor would let roughly
# 300 ms of speech look like 500 ms, and the word gate would then force an
# interruption the runtime's own accumulated-``speech_duration`` gate had
# refused. Subtracting it leaves an error that is one-sided the safe way: the
# ``speaking`` callback also lands one ``min_speech_duration`` after the caller
# really started, so what remains UNDER-counts.
VAD_END_OF_SPEECH_SILENCE_SECONDS = 0.25


def _bounded_turn_taking_int(value: Any, *, low: int, high: int) -> int | None:
    """Return one existing control-plane integer setting, or safely ignore it.

    The worker receives a call-bound config that the control plane normally
    normalizes, but it must remain safe if historic/direct metadata contains
    junk.  This mirrors the control-plane's lenient read contract without
    importing it into the isolated voice image.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (
        not math.isfinite(value) or not value.is_integer()
    ):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if low <= parsed <= high else None


def _turn_taking_sensitivity(value: Any) -> str:
    if not isinstance(value, str):
        return "balanced"
    normalized = value.strip().lower()
    return normalized if normalized in _SENSITIVITY_MIN_DURATION_SECONDS else "balanced"


def is_managed_pstn_call(realtime_config: Mapping[str, Any]) -> bool:
    """Whether this call-bound config is a managed customer phone call.

    This is what admits the caller-transcript gate, in EVERY language.  The
    hole the gate closes is not an English one: the pinned SDK reaches the
    interrupt path from ``on_interim_transcript``/``on_final_transcript`` for
    any non-empty transcript whenever turn detection is not
    ``manual``/``realtime_llm``, and ``turn_detection="vad"`` -- what every
    language without turn-detector support runs -- is not in that exempt set.
    A transcribed television cancels a Sinhala reply exactly as it cancels an
    English one, so withholding caller transcripts while a reply is in flight
    is correct in every language.

    Deliberately separate from ``is_managed_english_pstn_call`` below, which
    admits only the two policies whose CONTENT is English (the default word
    requirement and the backchannel list).  Protecting a call is not the
    same act as retuning it: the gate reads no turn-taking value at all -- its
    whole contract is one agent-state arming signal -- so it cannot go out of
    step with a threshold it never consults.

    ``direction`` is set at one place in the control plane -- when a customer
    call's runtime config is built -- so browser tests and the simulations lab
    (``web_test``), the standalone demo, and the platform release canary (whose
    config comes straight from ``resolve_agent_realtime_config`` with no call
    row) all return False here and keep their existing behaviour untouched.
    """

    return str(realtime_config.get("direction") or "").strip().lower() in {
        "inbound",
        "outbound",
    }


def is_managed_english_pstn_call(realtime_config: Mapping[str, Any]) -> bool:
    """Whether this call-bound config is a managed English phone call.

    One predicate, shared by the two policies whose CONTENT is English rather
    than merely English-measured: the default word requirement
    (:func:`default_min_interruption_words`) and the backchannel list
    (:func:`caller_backchannel_tokens`).  Two copies of an ``en`` + direction
    check would drift, and a call defaulted by one and filtered by the other
    is the exact shape of the defect this predicate exists to remove.

    It no longer gates the turn-taking presets.  Since 2026-09-15 those apply in
    every language (:func:`session_options_with_managed_pstn_turn_taking`);
    what stays here is what a language cannot borrow -- a word count needs a
    transcriber that answers in time, and the backchannel tokens ARE English
    words.

    Everything ``is_managed_pstn_call`` excludes is excluded here too.
    """

    return (
        is_managed_pstn_call(realtime_config)
        and str(realtime_config.get("language") or "").strip().lower() == "en"
    )


def requested_min_interruption_words(realtime_config: Mapping[str, Any]) -> int | None:
    """The per-agent word requirement as the customer wrote it; ``None`` if unset.

    ``None`` and ``0`` are DIFFERENT answers, and keeping them apart is the
    whole reason this is a separate function from
    :func:`configured_min_interruption_words`.  Since the platform default is
    no longer zero for every call, "the customer said nothing" and "the
    customer said off" select opposite behaviour, and a helper that collapsed
    them -- as this one's predecessor did with ``or 0`` -- would silently
    overrule an agent that had explicitly disabled the word requirement.

    The control plane is what makes the distinction reach here at all: it
    stores an explicit ``0`` rather than normalizing it away as a default (see
    ``services/agent_tuning.py``).

    Junk is ``None`` rather than ``0``, deliberately.  Historic or
    hand-written metadata carrying ``"two"`` expressed no preference, so it
    must fall back to the platform default rather than assert the one value a
    customer would have had to type on purpose.

    Read for EVERY managed phone call, in every language, and deliberately
    before any language branch: what a language decides is whether the
    requirement can be ENFORCED (below), never whether the customer is allowed
    to express it.  A setting silently dropped at the language boundary is a
    setting the API would keep returning and the agent would never run.
    """

    raw_tuning = realtime_config.get("turn_taking")
    tuning = raw_tuning if isinstance(raw_tuning, Mapping) else {}
    low, high = MIN_INTERRUPTION_WORDS_BOUNDS
    return _bounded_turn_taking_int(
        tuning.get("min_interruption_words"), low=low, high=high
    )


def default_min_interruption_words(realtime_config: Mapping[str, Any]) -> int:
    """The requirement a call that configured nothing runs.

    ONE anchor, expressed against the same predicate as the backchannel list,
    the other policy whose content is English.  A second English test
    written here would be free to drift from
    :func:`is_managed_english_pstn_call`, and a call tuned by one predicate and
    defaulted by another is the exact shape of defect the shared predicate
    exists to remove.

    Zero everywhere else, and that is not a placeholder: a browser test, the
    simulations lab, the standalone demo and the release canary keep their
    existing barge-in byte for byte, and a VAD-only language keeps
    duration-only barge-in because its transcriber cannot answer in time to
    count words at all.  See
    :data:`MANAGED_ENGLISH_DEFAULT_MIN_INTERRUPTION_WORDS`.
    """

    return (
        MANAGED_ENGLISH_DEFAULT_MIN_INTERRUPTION_WORDS
        if is_managed_english_pstn_call(realtime_config)
        else 0
    )


def configured_min_interruption_words(realtime_config: Mapping[str, Any]) -> int:
    """The word requirement this call asks for, before the stack may veto it.

    The customer's explicit value when they set one -- including ``0``, which
    turns the requirement off -- and the platform default when they did not.

    Deliberately still named "configured": it is the number the call is
    configured to want, as opposed to
    :func:`effective_min_interruption_words`, which is what the transcriber
    can actually deliver.  Both are reported side by side in the per-call
    interruption summary, so a reader can tell "the agent asked for a word and
    this stack cannot count one" apart from "the agent asked for nothing".
    """

    requested = requested_min_interruption_words(realtime_config)
    if requested is None:
        return default_min_interruption_words(realtime_config)
    return requested


def effective_min_interruption_words(
    realtime_config: Mapping[str, Any], *, registry: Any = None
) -> int:
    """The word requirement this call can actually enforce; 0 means duration-only.

    Three conditions, and every one of them fails to zero.

    * A managed phone call. Browser tests, the simulations lab, the standalone
      demo and the release canary keep their existing barge-in byte for byte,
      exactly as they do for the floor and the transcript gate.
    * A requirement above zero -- the customer's, or the platform default that
      :func:`default_min_interruption_words` supplies for a managed English
      call that configured nothing. The default reaches this veto by the same
      route a customer-set value does, on purpose: a default the stack cannot
      enforce must fail to duration-only exactly as an explicit setting does,
      not acquire a private path around the measurement below.
    * A transcriber whose results arrive in time to count, as measured and
      recorded on the capability artifact's STT row
      (``registry.stt_interims_usable``). ``scripts/stt_interim_latency_probe.py``
      is what moves a row to ``true``.

    That third condition is the honest answer to "why not every language". It
    is not that other languages deserve less: it is that a word requirement is
    only a barge-in gate where a word EXISTS by the time the floor opens.
    Measured on the pinned Deepgram adapter's own live-config, ``nova-3``
    returns transcript events on a fixed ~1000 ms grid aligned to the audio
    stream rather than to speech onset (16 gaps: p50 1004 ms, p95 1013 ms), so
    a caller's first word is countable within about a second of them starting.
    The Google model this platform dispatches -- ``chirp_2`` for ``si`` and
    the wider European set --
    returns its content later than that, so a requirement there would refuse
    ordinary interruptions rather than only the wordless ones: the caller
    would talk over the agent for the whole reply. Sinhala is the one Google
    row whose ``false`` rests on a measurement rather than on the absence of
    one. Driven through this worker's own plugin construction over eleven
    clips, ``chirp_2`` emitted an interim on three of them and never earlier
    than roughly six seconds into a seven-second utterance. The remaining
    Google rows are unmeasured, which is its own reason to stay ``false``.

    The refusal is visible rather than silent: the worker emits
    ``min_interruption_words_ignored_no_interims`` once for a call that
    configured a requirement this stack cannot enforce, and the per-call
    interruption summary carries the configured and effective numbers side by
    side.
    """

    if not is_managed_pstn_call(realtime_config):
        return 0
    configured = configured_min_interruption_words(realtime_config)
    if configured <= 0:
        return 0
    transcriber = realtime_config.get("transcriber")
    transcriber = transcriber if isinstance(transcriber, Mapping) else {}
    resolved = registry if registry is not None else load_registry()
    usable = resolved.stt_interims_usable(
        str(transcriber.get("registry_name") or ""),
        str(transcriber.get("model") or ""),
        str(realtime_config.get("language") or ""),
    )
    return configured if usable else 0


def installed_min_interruption_words(
    session_options: Mapping[str, Any],
    *,
    realtime_config: Mapping[str, Any],
    registry: Any = None,
) -> int:
    """The word requirement this BUILT session can actually enforce.

    Read from the same mapping handed to ``AgentSession(**options)``, for the
    same reason :func:`configured_interruption_min_duration` is: the worker must
    not hold a floor on a session that cannot release it.

    :func:`session_options_with_managed_pstn_turn_taking` refuses to touch a
    session whose ``turn_handling`` shape it does not recognise -- a corrupt or
    future internal stack -- and returns the original options untouched, which
    is the right thing for a mapping it cannot safely edit. But
    ``discard_audio_if_uninterruptible`` is not optional decoration on this
    feature: without it the runtime substitutes SILENCE on the STT path while a
    reply is uninterruptible, so the word that would release the floor can never
    be transcribed and the caller cannot interrupt that reply at all. An
    enablement resolved INDEPENDENTLY of the options would arm the holder on
    exactly the shapes the option could not be installed on.

    So the option's presence IS the enablement. One consumer reads one mapping,
    and the pairing cannot come apart.
    """

    turn_handling = session_options.get("turn_handling")
    if not isinstance(turn_handling, Mapping):
        return 0
    interruption = turn_handling.get("interruption")
    if not isinstance(interruption, Mapping):
        return 0
    if interruption.get("discard_audio_if_uninterruptible") is not False:
        return 0
    return effective_min_interruption_words(realtime_config, registry=registry)


def session_options_with_managed_pstn_turn_taking(
    session_options: Mapping[str, Any],
    *,
    realtime_config: Mapping[str, Any],
    registry: Any = None,
) -> dict[str, Any]:
    """Apply the managed-PSTN turn-taking policy to one built session.

    ONE predicate, ``is_managed_pstn_call``, and one path for every managed
    phone call in every language (founder decision 2026-09-15).  For each
    knob the precedence is explicit value, then preset, then platform default:

    * ``interruption.min_duration`` -- explicit ``min_speech_ms`` (honoured
      down to :data:`MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS`), else
      the ``interruption_sensitivity`` preset, else the half-second
      :data:`MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS`.  The stack's
      own baseline is 0.15 or 0.20 s depending on turn-detector support, a
      browser-demo value this seam replaces rather than inherits.
    * ``endpointing.min_delay`` -- explicit ``silence_hold_ms``, else the
      preset's row for the stack's endpointing mode
      (:func:`_sensitivity_hold_table`; with a turn detector every preset,
      balanced included, has a phone-line hold), else the stack's own hold,
      0.8 s without a detector (``demo.py``).
    * ``endpointing.max_delay`` -- with a turn detector, the preset's
      unsure-turn wait (:data:`_SENSITIVITY_MODEL_MAX_HOLD_SECONDS`), never
      below ``min_delay``.

    Presets were English-only until 2026-09-15 on the ground that their values
    were English-measured.  The Sinhala replays of that day showed
    the interruption floor is a property of the line and the VAD rather than
    of the language, and the hold table's VAD-only row is what keeps a
    preset from handing a language without a turn detector the
    model-language pad.

    An EXPLICIT ``silence_hold_ms``/``min_speech_ms`` is not a preset and is
    honoured in every language: it is a number the customer chose for their own
    line, and accepting it while silently ignoring it is the failure the
    per-agent contract exists to prevent. The platform's own default hold for a
    language with no turn detector is set once in the stack (``demo.py``), not
    here.

    The prewarmed Silero VAD object is intentionally passed through untouched:
    it is shared by concurrent worker processes, so changing its options here
    could leak one tenant's tuning into another call.  The pinned LiveKit
    runtime does support independent ``AgentSession`` interruption and
    endpointing options, which are copied and set per session below.

    ``barge_in_speech_rms`` has no equivalent per-session LiveKit control in
    this self-hosted stack.  It remains accepted and round-tripped by the
    existing control-plane contract, but is deliberately not simulated by
    mutating shared VAD state or by adding a second audio gate.
    """

    copied_options = dict(session_options)
    if not is_managed_pstn_call(realtime_config):
        # Browser tests are direction=web_test, the standalone demo provides
        # no call direction, and release-canary configs have no customer call
        # direction.  All retain their original nested option mapping.
        return copied_options

    turn_handling = session_options.get("turn_handling")
    if not isinstance(turn_handling, Mapping):
        # The stack owns this internal shape.  Failing closed to the existing
        # options is safer than building a partial incompatible session from a
        # corrupt internal stack.
        return copied_options
    interruption = turn_handling.get("interruption")
    endpointing = turn_handling.get("endpointing")
    if not isinstance(interruption, Mapping) or not isinstance(endpointing, Mapping):
        return copied_options

    copied_interruption = dict(interruption)
    if effective_min_interruption_words(realtime_config, registry=registry) > 0:
        # REQUIRED by the word gate, and this is the one place the pairing can
        # be guaranteed.  The gate holds the reply's speech handle
        # uninterruptible until N words have been heard, and the pinned runtime
        # answers an uninterruptible speech by substituting SILENCE on the STT
        # path (``agent_activity.py`` :1279-1285, the option defaults to True in
        # ``voice/turn.py`` :192).  Left at the default, the word that would
        # release the floor can never be transcribed, so the requirement would
        # not delay a barge-in -- it would abolish it, for the whole reply, on
        # every call that switched the setting on.
        #
        # Deliberately scoped to a call that has an ENFORCEABLE requirement
        # rather than set for managed calls generally.  With ``N = 0`` the key
        # is absent and the built session is byte-identical to the one this
        # stack shipped before the setting existed; only an agent that opted in
        # changes what its uninterruptible turns do with caller audio.  The
        # cost of the change is bounded and already the norm here: an
        # interruptible reply -- which is every reply on a call that did not opt
        # in -- has always had the caller's real audio on the STT path, echo
        # included, so this widens an existing exposure by the length of the
        # turns the word gate holds rather than creating a new one.
        copied_interruption["discard_audio_if_uninterruptible"] = False
    raw_tuning = realtime_config.get("turn_taking")
    tuning = raw_tuning if isinstance(raw_tuning, Mapping) else {}
    min_speech_ms = _bounded_turn_taking_int(
        tuning.get("min_speech_ms"),
        low=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][0],
        high=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][1],
    )
    silence_hold_ms = _bounded_turn_taking_int(
        tuning.get("silence_hold_ms"),
        low=_TURN_TAKING_INT_BOUNDS["silence_hold_ms"][0],
        high=_TURN_TAKING_INT_BOUNDS["silence_hold_ms"][1],
    )
    copied_endpointing = dict(endpointing)
    sensitivity = _turn_taking_sensitivity(tuning.get("interruption_sensitivity"))

    # ONE path for every managed phone call, in every language: the explicit
    # value wins, then the preset, then the platform default.  The clamp on
    # the explicit value floors at the same 0.15 s the ``high`` preset runs;
    # the preset keeps its own literal rather than aliasing the constant,
    # because moving either number is a separate decision from moving the
    # other.  A missing or junk preset reads as ``balanced``, which is the
    # platform default on both knobs.
    copied_interruption["min_duration"] = (
        _explicit_min_speech_seconds(min_speech_ms)
        if min_speech_ms is not None
        else _SENSITIVITY_MIN_DURATION_SECONDS[sensitivity]
    )
    hold_table = _sensitivity_hold_table(turn_handling)
    if silence_hold_ms is not None:
        copied_endpointing["min_delay"] = silence_hold_ms / 1_000
    elif hold_table is not None and sensitivity in hold_table:
        copied_endpointing["min_delay"] = hold_table[sensitivity]
    if hold_table is _SENSITIVITY_MODEL_HOLD_SECONDS:
        # The preset's unsure-turn wait applies whatever the explicit hold;
        # the ordered pair below raises it to an explicit hold above it.
        copied_endpointing["max_delay"] = _SENSITIVITY_MODEL_MAX_HOLD_SECONDS[sensitivity]
    copied_turn_handling = dict(turn_handling)
    copied_turn_handling["interruption"] = copied_interruption
    copied_turn_handling["endpointing"] = _endpointing_with_ordered_pair(
        copied_endpointing
    )
    copied_options["turn_handling"] = copied_turn_handling
    return copied_options


def configured_interruption_min_duration(
    session_options: Mapping[str, Any],
) -> float | None:
    """The interruption floor the built session will actually run with.

    Read from the SAME mapping that is handed to ``AgentSession(**options)``,
    after the per-agent policy above has (or has not) applied, so a log line
    quoting it cannot disagree with the session that produced it.  ``None``
    when the stack did not supply one and the SDK default applies.
    """

    turn_handling = session_options.get("turn_handling")
    if not isinstance(turn_handling, Mapping):
        return None
    interruption = turn_handling.get("interruption")
    if not isinstance(interruption, Mapping):
        return None
    value = interruption.get("min_duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or float(value) < 0:
        return None
    return float(value)


def turn_end_settings_evidence(
    session_options: Mapping[str, Any],
    *,
    realtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """The turn-end settings one built session runs with, for its evidence file.

    Three sources, each read where it is authoritative, so the record cannot
    describe a session other than the one that ran:

    * what the session was built with -- the SAME mapping handed to
      ``AgentSession(**options)``, after
      :func:`session_options_with_managed_pstn_turn_taking` (and so after
      :func:`_endpointing_with_ordered_pair`) has applied;
    * what the caller's agent ASKED for -- the call's own ``turn_taking``
      block, read through the same lenient bounds the policy above uses, so a
      requested value and the delay it produced sit side by side;
    * the voice detector's end-of-speech silence, which is neither: the
      prewarmed detector is shared by every session in the process and is not
      in ``session_options`` at all.

    Numbers and short labels only.  A value the stack did not supply is
    ``None`` rather than a default this function would have to guess.
    """

    def seconds(mapping: Any, key: str) -> float | None:
        value = mapping.get(key) if isinstance(mapping, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(float(value)) and float(value) >= 0 else None

    turn_handling = session_options.get("turn_handling")
    handling = turn_handling if isinstance(turn_handling, Mapping) else {}
    endpointing = handling.get("endpointing")
    detection = handling.get("turn_detection")
    if detection is None:
        detector = "unset"
    elif isinstance(detection, str):
        detector = detection.strip().lower()[:64] or "unset"
    else:
        # A loaded detector object.  Its own model label when it has one,
        # because "a detector ran" does not say which.
        model = getattr(detection, "model", None)
        detector = model.strip()[:64] if isinstance(model, str) and model.strip() else "detector"
    mode = endpointing.get("mode") if isinstance(endpointing, Mapping) else None
    raw_tuning = realtime_config.get("turn_taking")
    tuning = raw_tuning if isinstance(raw_tuning, Mapping) else {}
    return {
        "managed_pstn": is_managed_pstn_call(realtime_config),
        "endpointing_mode": mode[:32] if isinstance(mode, str) else None,
        "endpointing_min_delay_seconds": seconds(endpointing, "min_delay"),
        "endpointing_max_delay_seconds": seconds(endpointing, "max_delay"),
        "turn_detection": detector,
        "interruption_min_duration_seconds": configured_interruption_min_duration(
            session_options
        ),
        "transcription_timeout_seconds": seconds(session_options, "transcription_timeout"),
        "vad_end_of_speech_silence_seconds": VAD_END_OF_SPEECH_SILENCE_SECONDS,
        # The preset the policy applies: ``balanced`` when the agent set none.
        "interruption_sensitivity": _turn_taking_sensitivity(
            tuning.get("interruption_sensitivity")
        ),
        "requested_silence_hold_ms": _bounded_turn_taking_int(
            tuning.get("silence_hold_ms"),
            low=_TURN_TAKING_INT_BOUNDS["silence_hold_ms"][0],
            high=_TURN_TAKING_INT_BOUNDS["silence_hold_ms"][1],
        ),
        "requested_min_speech_ms": _bounded_turn_taking_int(
            tuning.get("min_speech_ms"),
            low=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][0],
            high=_TURN_TAKING_INT_BOUNDS["min_speech_ms"][1],
        ),
    }


# The pinned SDK reaches ``AgentActivity._interrupt_by_audio_activity`` from
# three places (livekit-agents 1.7.1):
#
#   * ``on_vad_inference_done`` (voice/agent_activity.py:2250) -- gated by
#     ``interruption["min_duration"]``, the threshold this policy sets;
#   * ``on_interim_transcript`` (:2300-2323) and ``on_final_transcript``
#     (:2334-2355) -- called for ANY non-empty transcript under our
#     ``turn-detector`` mode, with no duration gate at all.  Their only gate is
#     ``min_words``, which the stack sets to 0, and which is deliberately not
#     the fix here: background speech contains real, countable words.
#
#     1.6.8 (#6599) narrowed the interim hook only: it now returns early when
#     a VAD is configured, and we always configure one, so interim transcripts
#     no longer reach the interrupt.  ``on_final_transcript`` is unchanged and
#     still calls it unconditionally, which is the hole this gate closes.
#
# The SDK's own transcript-holding machinery is inert for us.  1.7.1 moved it
# out of ``_should_hold_stt_event`` (gone) and into ``AudioRecognition``'s
# ``_transcript_gate_active`` (audio_recognition.py:594, held events buffered
# at :1155), but its precondition did not change: ``_on_start_of_overlap_speech``
# returns immediately unless ``_adaptive_interruption_active``, which needs
# ``interruption_detection`` to resolve to LiveKit Cloud's adaptive detector.
# ``_resolve_interruption_detection`` (:4718) still returns None for our
# ``mode="vad"`` and, absent an explicit opt-in, for self-hosted production
# generally (the ``utils.is_hosted()``/``is_dev_mode()`` gate at :4757-4763).
# Adopting adaptive means adopting LiveKit Cloud, which the issue names as a
# non-goal.
#
# So the gate is installed one layer earlier, at ``Agent.stt_node`` -- the
# SDK's documented STT override point and the SOLE producer of the events
# that reach those hooks:
#
#   AudioRecognition(stt=self._agent.stt_node)      agent_activity.py:1150
#     -> _STTPipeline._stt_pump() -> event_ch       audio_recognition.py:187-206
#     -> `async for ev in event_ch: _on_stt_event`  audio_recognition.py:1808
#     -> hooks.on_interim/final_transcript          audio_recognition.py:1059,1185,1241,1283
#
# Holding an event there means it cannot reach the ungated interrupt sites,
# which makes ``min_duration`` the single threshold that governs whether
# agent playout is cancelled -- the contract the public docs already state.
# The agent states in which an interruptible reply exists.  ``thinking``
# spans the turn commit to the first TTS frame; a final transcript there
# cancels the reply through ``_cancel_speech_pause`` with no duration gate,
# so it needs the same protection as playout itself.  ``listening`` is
# deliberately excluded: reaching it is exactly how a VAD-confirmed barge-in
# announces that the caller's words are now what the session wants.
REPLY_IN_FLIGHT_AGENT_STATES = frozenset({"thinking", "speaking"})
_STT_EVENT_TYPE_RECOGNITION_USAGE = "recognition_usage"
_STT_EVENT_TYPE_FINAL_TRANSCRIPT = "final_transcript"
_STT_EVENT_TYPES_CARRYING_SPEECH = frozenset(
    {"interim_transcript", "preflight_transcript", "final_transcript"}
)
# The MEMORY bound. Interims dominate the event volume by far, and the buffer
# coalesces them, so this is what actually keeps a hold bounded.
CALLER_TRANSCRIPT_HOLD_MAX_EVENTS = 128
# The WEDGE bound, and nothing else. It must sit well above any legitimate
# reply, because expiring it stands gating down and a reply is allowed to be
# long: a first message may be 1,000 characters and a model reply is uncapped,
# so a 30 s ceiling would let quiet background recreate the original false
# barge-in on an ordinary long answer. Five minutes is longer than a reply and
# shorter than caring; memory is already bounded by the event cap above.
CALLER_TRANSCRIPT_HOLD_MAX_SECONDS = 300.0


def _stt_event_transcript(event: Any) -> str:
    """The whitespace-normalized text of one STT event, or ``""``.

    Never raises for a malformed or foreign object: this runs on the media
    path for every caller event, and an adapter that hands over something
    unexpected must lose a word count, not a call.
    """

    try:
        alternatives = getattr(event, "alternatives", None) or ()
        raw_text = getattr(alternatives[0], "text", "") if alternatives else ""
        # Inside the guard, not after it. ``raw_text`` is a provider object,
        # so its truthiness and its string conversion are the provider's code
        # -- and this runs on the media loop for every caller event, where an
        # escaping exception fails the call rather than losing a word.
        return " ".join(str(raw_text or "").split())
    except Exception:
        return ""


#: STT registries whose pinned adapter reports a REAL per-utterance confidence.
#:
#: Measured against the pinned plugins, not assumed. ``deepgram`` 1.7.1 carries
#: the recogniser's own value (``alt["confidence"]``). ``azure`` 1.7.1 does not:
#: it hardcodes ``confidence=1.0`` on every final and ``0.0`` on every interim
#: and still carries the upstream TODO to read it from NBest under the
#: ``detailed`` output format, so its number describes the plugin rather than
#: the utterance. ``google`` and ``soniox`` are absent because nothing here has
#: measured them; absence means "do not claim a confidence", which is the
#: correct failure direction for a field a customer reads as recognition
#: quality.
#:
#: Deliberately NOT a ``capabilities.json`` declaration. That artifact owns
#: admission, and releasing it strands every agent holding a language-profile
#: publication until each is republished -- a cost this data-quality fact does
#: not justify. It is read by registry name, the same key
#: ``effective_min_interruption_words`` resolves a transcriber by.
STT_REGISTRIES_REPORTING_CONFIDENCE = frozenset({"deepgram"})


def stt_registry_reports_confidence(registry_name: Any) -> bool:
    """Whether this STT registry's pinned adapter measures confidence."""

    return str(registry_name or "").strip().lower() in STT_REGISTRIES_REPORTING_CONFIDENCE


def _stt_event_confidence(event: Any) -> float | None:
    """One STT event's reported confidence in ``[0, 1]``, or ``None``.

    ``None`` means "the provider said nothing usable", never "zero": a caller
    turn with no measurement must reach the database as an absent value rather
    than as a number a reader would take for a poor recognition. Same
    never-raises contract as :func:`_stt_event_transcript` -- this runs on the
    media path for every caller event.
    """

    try:
        alternatives = getattr(event, "alternatives", None) or ()
        if not alternatives:
            return None
        raw = getattr(alternatives[0], "confidence", None)
        if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value
    except Exception:
        return None


@dataclass(frozen=True)
class CallerFinalObservation:
    """What the recogniser reported about one released caller final.

    ``observed_at`` is the pre-gate monotonic time the final arrived, the same
    anchor the transcript/VAD join has always used. The other two are the
    provider's own measurements and are ``None`` whenever it did not make one;
    no field here is ever filled with a stand-in value.
    """

    observed_at: float
    measured_duration_seconds: float | None = None
    confidence: float | None = None


def _stt_event_measured_duration(event: Any) -> float | None:
    """How long the recogniser says this utterance lasted, in seconds.

    Only the DURATION is taken. The provider's ``SpeechData`` timestamps are
    stream-relative -- ``start_time_offset`` begins at zero and accumulates
    reconnect gaps -- so neither endpoint can be placed on the recorder's
    monotonic clock, but their difference is a real measurement of the speech
    the recogniser segmented.

    ``None`` for anything that is not a positive finite span, so a caller turn
    the provider did not measure stays visibly unmeasured instead of borrowing
    a fabricated length. A span longer than
    ``CALLER_TRANSCRIPT_HOLD_MAX_SECONDS`` is rejected as malfunctioning rather
    than describing a caller utterance; that is the module constant, not this
    gate instance's configurable ``max_hold_seconds``, because the check is a
    sanity bound on a provider object and not a per-call policy.
    """

    try:
        alternatives = getattr(event, "alternatives", None) or ()
        if not alternatives:
            return None
        alternative = alternatives[0]
        start = getattr(alternative, "start_time", None)
        end = getattr(alternative, "end_time", None)
        for raw in (start, end):
            if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
        span = float(end) - float(start)
        if not math.isfinite(span) or span <= 0.0:
            return None
        # A recogniser that reports a span longer than the gate's own wedge
        # bound is malfunctioning, not describing a caller utterance.
        if span > CALLER_TRANSCRIPT_HOLD_MAX_SECONDS:
            return None
        return span
    except Exception:
        return None


# The English sounds a caller makes to say "I am still listening", which are
# NOT a request for the floor.  Vapi (``acknowledgementPhrases``) and Retell
# (``enable_backchannel``) both ship a list of this shape; it is the technique
# the market buys naturalness with, and it is the half of that technique this
# stack was missing.
#
# ONE declared place, deliberately.  A list that also existed in the control
# plane, the capability artifact or a second worker module would be four lists
# that agree until they do not, and the failure would be silent -- a caller
# whose "mm-hm" cut the agent on one release and not the next.  It is a module
# constant rather than a row on ``capabilities.json`` because the tokens are a
# property of the LANGUAGE, not of a provider stack, and putting them on the
# capability artifact would make every list an input to the apps lane as well
# as the voice one for no gain.
#
# ENGLISH, and applied only to an English call -- see
# :func:`caller_backchannel_tokens`.  Estonian is the case that makes the
# scoping load-bearing rather than tidy: Deepgram ``nova-3`` serves it, so
# ``stt_interims_usable`` is true and an Estonian agent that sets
# ``min_interruption_words`` really does run the word gate.  Applying THIS list
# there would do nothing for "jah" or "okei" while claiming, in the public
# contract, that listening noises do not count -- and inventing a list for a
# language nobody here speaks, from no call audio, is how the wrong words get
# refused the floor.  So the list is scoped to the language it was written for,
# and another language earns its own list by measurement.
#
# Fillers ("uh", "um", "hmm") are here alongside the acknowledgements even
# though they are not agreement, because the pinned Deepgram adapter runs with
# ``filler_words`` on: they arrive as ordinary one-word transcripts and would
# otherwise satisfy a one-word requirement on their own.  A hesitation noise is
# the least deliberate thing a caller emits; refusing it the floor is the whole
# point.
#
# "no" is deliberately ABSENT.  It is a substantive answer -- the shortest
# disagreement a caller can make -- and a caller who says it while the agent is
# talking means to be heard.
#
# So is a bare "huh", for the same reason and less obviously: "huh?" is a
# request to STOP AND REPEAT, not an acknowledgement, so a caller who says it
# wants the floor.  Only the compound "uh-huh" is agreement, and it is matched
# in one piece below rather than by admitting its second half on its own.
CALLER_BACKCHANNEL_TOKENS = frozenset(
    {
        # acknowledgement
        "yeah",
        "yea",
        "yep",
        "yup",
        "yes",
        "ya",
        "okay",
        "ok",
        "kay",
        "alright",
        "right",
        "sure",
        "gotcha",
        # nasal assent
        "mhm",
        "mhmm",
        "mmhm",
        "mmhmm",
        "mm-hm",
        "mm-hmm",
        "m-hm",
        "uh-huh",
        "uhhuh",
        "mm",
        "mmm",
        # hesitation / thinking noises (Deepgram emits these: filler_words is on)
        "uh",
        "uhh",
        "um",
        "umm",
        "uhm",
        "hm",
        "hmm",
        "hmmm",
        "ah",
        "ahh",
        "aha",
        "oh",
        "ohh",
    }
)

# How many allowlisted tokens still read as a backchannel.  Two covers what a
# caller actually says -- "uh huh", "yeah yeah", "oh okay", "okay sure" -- and
# stops there on purpose: three or more acknowledgements in a row is sustained
# speech over the agent, which is a caller taking the floor however agreeable
# the words are.  It also bounds the damage of a mis-transcription that happens
# to land inside the list.
CALLER_BACKCHANNEL_MAX_WORDS = 2

# Deepgram returns display-formatted text -- "Yeah," and "Mm-hmm." -- so a
# match on the raw token would never fire.  Only the EDGES are stripped: the
# hyphen inside "mm-hmm" is part of the token, and splitting on it would turn
# one backchannel into two.
_CALLER_BACKCHANNEL_STRIP = ".,!?;:\"'()[]{}\u2026\u2019\u201c\u201d-"


def caller_backchannel_tokens(realtime_config: Mapping[str, Any]) -> frozenset[str]:
    """The backchannel list this call may apply; empty when there is none.

    The same predicate as the default word requirement, for the same reason:
    the tokens ARE English, and a second language
    test written here would be free to drift from
    :func:`is_managed_english_pstn_call`.

    Empty rather than English for everything else, and Estonian is why that
    matters.  It is not a language the word gate cannot reach -- Deepgram
    ``nova-3`` serves it, so an Estonian agent that sets
    ``min_interruption_words`` runs the gate for real.  It is a language whose
    backchannels this repository has not measured, and handing it an English
    list would refuse nothing while the public contract claimed otherwise.
    """

    return (
        CALLER_BACKCHANNEL_TOKENS
        if is_managed_english_pstn_call(realtime_config)
        else frozenset()
    )


def is_caller_backchannel_only(text: str, tokens: frozenset[str]) -> bool:
    """Whether a transcript is nothing but short listening noises.

    ``True`` when the transcript has at least one word, at most
    :data:`CALLER_BACKCHANNEL_MAX_WORDS` of them, and either EVERY one is in
    ``tokens`` or the tokens joined together are (which is how a spaced
    "uh huh" matches the compound "uh-huh").  "yes but" is two words of which
    one is not, so it is not a backchannel; neither is "no", which is not in
    the list at all.

    ``tokens`` is a REQUIRED argument rather than a default of
    :data:`CALLER_BACKCHANNEL_TOKENS`, because the list is language-scoped and
    a default would let a call apply English backchannels by forgetting to say
    which language it is.  An empty set matches nothing, which is what a
    language with no measured list must do.

    The length bound and the all-tokens rule are conjunctive on purpose.  A
    rule that suppressed a transcript because it merely CONTAINED a backchannel
    would refuse the commonest real interruption there is -- "yeah but wait" --
    and one with no length bound would refuse a caller who agrees at length.
    """

    if not tokens:
        return False
    words = [token.strip(_CALLER_BACKCHANNEL_STRIP).lower() for token in text.split()]
    if not words or len(words) > CALLER_BACKCHANNEL_MAX_WORDS:
        return False
    if all(word in tokens for word in words):
        return True
    # One backchannel a transcriber happened to space out: "uh huh" for
    # "uh-huh", "mm hm" for "mmhm".  Matching the JOINED form is what lets the
    # list hold the compound without also admitting its halves -- a bare "huh"
    # is a request to repeat, and allowlisting it to make "uh huh" work would
    # refuse the floor to a caller who did not follow.
    return "".join(words) in tokens


def _transcript_word_count(text: str) -> int:
    """How many words a transcript carries, by whitespace.

    Deliberately NOT the SDK's ``split_words``.  That helper takes
    ``split_character=True`` at both of the runtime's ``min_words`` sites,
    which counts CHARACTERS for scripts written without spaces -- so "one
    word" there means one Han character, and a requirement of 2 would be met
    by a two-character syllable.  This module owns a customer-facing setting
    documented in words, imports no livekit by design, and serves only
    languages this platform dispatches with space-separated orthographies
    (the capability artifact's ``stt_interims_usable`` is true for Deepgram
    ``nova-3`` alone).  A language whose words are not space-separated needs
    its own counter measured alongside its cadence, and until then its STT row
    stays ``false`` and this function is never asked.
    """

    return len(text.split())


def _stt_event_type_name(event: Any) -> str:
    """Return one pinned SDK ``SpeechEventType`` as its plain string value.

    ``SpeechEventType`` is a ``str`` Enum, so ``str(member)`` renders
    ``"SpeechEventType.RECOGNITION_USAGE"`` rather than the value.  Read
    ``.value`` and never format the member.
    """

    event_type = getattr(event, "type", None)
    return str(getattr(event_type, "value", event_type) or "")


class CallerTranscriptGate:
    """Withhold caller transcripts from the SDK while a reply is in flight.

    Barge-in stays enabled and unchanged; what changes is which signal is
    allowed to trigger it.  While the agent owns an interruptible reply,
    caller STT events are buffered instead of forwarded, so the only path
    that can cancel that reply is the VAD path -- the one the configured
    ``interruption.min_duration`` actually gates.  The moment the reply ends
    (including the instant a VAD-confirmed barge-in pauses it) the buffer is
    released in arrival order, so no caller turn is dropped: a caller who
    spoke under the threshold is answered when the reply ends instead of
    never.

    "In flight" is deliberately wider than "audio is playing".  Between the
    turn commit and the first TTS frame the session reports ``thinking``,
    and in the pinned SDK a final transcript in that window still reaches
    ``_interrupt_by_audio_activity`` and then ``_cancel_speech_pause``, which
    cancels the pending reply outright with no duration gate.  Gating only
    ``speaking`` would leave the whole LLM/TTS latency window open to exactly
    the background speech this exists to ignore -- and a reply killed before
    its first frame is worse for the caller than a truncated one.

    This mirrors the SDK's own ``_should_hold_stt_event`` policy: usage
    events bypass the buffer (they are billing, not speech), everything else
    is held while the agent speaks.

    Loop-affine by contract.  Every method must be called from the single
    asyncio loop running the media session; there are no locks and no
    threads, because the SDK invokes both the STT node and the session event
    handlers on that one loop.  It performs no I/O and never raises for a
    malformed event -- an unrecognized object is forwarded untouched rather
    than swallowed.
    """

    def __init__(
        self,
        *,
        max_held_events: int = CALLER_TRANSCRIPT_HOLD_MAX_EVENTS,
        max_hold_seconds: float = CALLER_TRANSCRIPT_HOLD_MAX_SECONDS,
        min_interruption_words: int = 0,
        backchannel_tokens: frozenset[str] = frozenset(),
        min_duration_seconds: float | None = None,
        end_of_speech_silence_seconds: float = VAD_END_OF_SPEECH_SILENCE_SECONDS,
        stt_reports_confidence: bool = False,
        clock: Callable[[], float] = time.monotonic,
        observer: Any | None = None,
    ) -> None:
        self._clock = clock
        #: The end_call guard's caller-evidence feed (``note_speech_event``,
        #: ``note_forwarded_final``): every speech-carrying event as it
        #: ENTERS this gate, before it is held, and every final it forwards.
        #: Fed from the session's transcript event it would be blind to a
        #: final this gate is withholding.
        self._observer = observer
        # Whether THIS call's recogniser measures confidence. Defaults to
        # False so a gate built without the fact records no confidence at all
        # rather than republishing a plugin's placeholder as a measurement.
        self._stt_reports_confidence = bool(stt_reports_confidence)
        self._max_held_events = max(1, int(max_held_events))
        self._max_hold_seconds = float(max_hold_seconds)
        # The word half of the barge-in criterion, enforced from the events
        # this gate is ALREADY holding.  See :meth:`interruption_qualified`.
        # ``0`` is off and is the shipped default; the caller resolves it from
        # ``effective_min_interruption_words`` so a language that cannot count
        # words never reaches here with a non-zero value.
        self._min_interruption_words = max(0, int(min_interruption_words))
        # The backchannel list THIS call may apply, resolved by the caller from
        # ``caller_backchannel_tokens``. Empty is the default and means no
        # filtering, so a language whose backchannels are unmeasured behaves
        # exactly as it did before this existed rather than inheriting a list
        # written for English.
        self._backchannel_tokens = frozenset(backchannel_tokens)
        # The floor the SESSION runs, read from the same built options the
        # runtime is handed, so this gate cannot enforce a stricter one than
        # the call it is protecting. A ``high``-sensitivity agent runs 0.15 s
        # and an explicit ``min_speech_ms`` runs something else again;
        # comparing against the managed constant instead would silently refuse
        # interruptions that agent's own configuration allows -- and would make
        # the published documentation false. ``None`` means the stack supplied
        # no floor, which falls back to the managed one rather than to zero.
        floor = (
            float(min_duration_seconds)
            if isinstance(min_duration_seconds, (int, float))
            and not isinstance(min_duration_seconds, bool)
            and math.isfinite(float(min_duration_seconds))
            and float(min_duration_seconds) > 0
            else MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS
        )
        self._min_duration_seconds = floor
        self._end_of_speech_silence = max(0.0, float(end_of_speech_silence_seconds))
        self._committed_words = 0
        self._pending_interim_words = 0
        self._words_heard = 0
        self._words_at_qualification = 0
        # Per CALL and monotone, fed by EVERY speech event whether held or
        # forwarded; see :attr:`caller_words_observed` for why the per-reply
        # evidence above cannot answer the playout hold's question.
        self._observed_committed_words = 0
        self._observed_pending_words = 0
        # Per CALL, not per reply: unlike the word evidence above, this is
        # telemetry rather than a decision input, and the question it answers
        # ("how often did this caller back-channel over the agent?") is a
        # property of the whole call.  Reset by ``_reset_word_evidence`` it
        # would only ever report the last reply.  It is counted on every call,
        # including one with no word requirement, so the two arms of a
        # comparison are measured the same way.
        self._backchannel_utterances = 0
        # Whether the reply in flight has actually reached the wire. Only the
        # backchannel COUNTER reads it: a "mm-hm" during the 2-3 s thinking
        # window, or during a Gemini greeting's synthesis, is not the caller
        # talking over the agent -- nothing was audible to talk over -- and
        # counting it would inflate the very number a comparison is read from.
        # The word evidence has its own, stricter rescope; see
        # :meth:`note_reply_became_audible`.
        self._reply_audible = False
        # The caller's VAD speech run, which is the DURATION half of the
        # criterion. Fed by ``user_state``, which the pinned runtime raises
        # from the VAD's own start-of-speech; see
        # :meth:`note_caller_speaking` for why that is a different decision
        # from the arming contract and why its error is safe.
        self._caller_speaking_since: float | None = None
        self._run_seen_since_audible = False
        self._duration_qualified = False
        self._reply_in_flight = False
        # The opening utterance's own window, which no agent state
        # reports.  ``session.say`` never moves the session to
        # ``thinking`` -- the pinned runtime only reaches ``speaking``, from
        # the first audio frame's callback (``agent_activity.py`` _tts_task_impl
        # ``_on_first_frame``) -- so during a 3-6 s Gemini TTS synthesis the
        # agent state is still ``listening`` and the arming signal below is
        # silent.  That is the window in which a mis-heard "hello?" cancelled
        # a greeting that had not yet made a sound.  It is a SEPARATE flag,
        # not a second writer of ``_reply_in_flight``: an ``agent_state``
        # transition landing inside the window (``initializing -> listening``
        # races the opening say) would otherwise release the hold the window
        # exists to keep.
        self._opening_turn_in_flight = False
        # Set only by the expiry path below, which is wedge recovery rather
        # than a turn-taking decision.
        self._suspended_until_next_reply = False
        self._held: list[tuple[Any, float]] = []
        # LiveKit creates a fresh session event after a final leaves this
        # gate, so raw-event identity cannot carry its observation time.
        self._released_final_observations: list[tuple[str, CallerFinalObservation]] = []
        self._holding_since: float | None = None
        self._held_total = 0
        self._released_total = 0
        self._discarded_total = 0
        self._pending_speech = 0
        self._release_reasons: list[str] = []

    @property
    def stats(self) -> dict[str, int | str | bool]:
        """Bounded counters for the call's interruption telemetry."""

        return {
            "held": self._held_total,
            "released": self._released_total,
            "discarded": self._discarded_total,
            "suspended": self._suspended_until_next_reply,
            "opening_turn": self._opening_turn_in_flight,
            "pending": len(self._held),
            # Only the held events that actually carry words. Telemetry needs
            # this to answer "was a transcript present?" at the moment playout
            # was cancelled: the transcript exists, it just has not been let
            # through yet, and counting speech markers would overstate it.
            "pending_speech": self._pending_speech,
            "last_release_reason": self._release_reasons[-1]
            if self._release_reasons
            else "",
            # The word evidence for the reply currently in flight. Read by the
            # interruption telemetry at the same instant as ``pending_speech``,
            # so a log line can say how many words the caller had actually got
            # out when playout stopped -- the measurement the founder needs
            # before choosing a default for this setting.
            "words_heard": self._words_heard,
            "words_at_qualification": self._words_at_qualification,
            "min_interruption_words": self._min_interruption_words,
            # How many committed caller utterances were nothing but listening
            # noises -- finals, not events, so one "yeah" counts once however
            # many interims carried it.
            #
            # OBSERVED, not "refused the floor", and the distinction is what
            # makes the number usable. Whether a backchannel actually cost the
            # caller an interruption depends on ``min_interruption_words``
            # beside it: with a requirement, these are the utterances that did
            # not qualify; with none, the criterion never consulted the word
            # evidence and nothing was denied. Counting only the first case
            # would report zero on exactly the calls a controlled comparison
            # needs as its baseline, so the count is unconditional and the
            # reader has both fields.
            #
            # It is a COUNT, not a rate. Two calls of different lengths are not
            # comparable on it directly; the summary's ``held``/``released``
            # transcript totals are the denominator a comparison needs.
            "caller_backchannel_utterances": self._backchannel_utterances,
        }

    @property
    def gating(self) -> bool:
        """Whether anything the caller could talk over is currently in flight."""

        return self._reply_in_flight or self._opening_turn_in_flight

    @property
    def holds_final_transcript(self) -> bool:
        """Whether a final transcript with words is being withheld right now.

        Reported by the transcription-stall recovery when the SDK's
        ``transcription_timeout`` expires during a reply: a final this gate
        is holding never reached the SDK, so from the SDK's side the turn
        looks untranscribed, but the transcriber DID finalize and the gate
        releases it the moment the reply ends. The recovery logs this and
        flushes anyway -- a held final for an earlier utterance says nothing
        about a later one that stalled -- so the value is what lets a log
        reader tell a needless rebuild from a recovered stall. FINALS only:
        an interim says the transcriber is working, not that it has
        finished, and a stall is exactly the case where interims may flow
        and no final follows.
        """

        return any(
            _stt_event_type_name(event) == _STT_EVENT_TYPE_FINAL_TRANSCRIPT
            and bool(_stt_event_transcript(event))
            for event, _observed_at in self._held
        )

    def holds_final_transcript_where(self, counts: Callable[[str], bool]) -> bool:
        """:attr:`holds_final_transcript`, counting only a final whose text
        ``counts`` accepts (the end_call guard's words reading)."""

        for event, _observed_at in self._held:
            if _stt_event_type_name(event) != _STT_EVENT_TYPE_FINAL_TRANSCRIPT:
                continue
            text = _stt_event_transcript(event)
            if not text:
                continue
            try:
                if counts(text):
                    return True
            except Exception:  # noqa: BLE001 - an unreadable reading keeps the veto
                return True
        return False

    @property
    def min_interruption_words(self) -> int:
        return self._min_interruption_words

    @property
    def words_heard(self) -> int:
        """Words the caller has produced since the current reply began.

        COUNTABLE words: a transcript that is nothing but listening noises
        contributes zero (see :func:`is_caller_backchannel_only`), so this
        answers "how much did the caller say that was a bid for the floor",
        not "how many tokens did the transcriber emit". The number of refused
        backchannel utterances is reported separately in :attr:`stats` rather
        than folded in here, so the two questions stay separable in the log.

        Monotone within a reply BY CONSTRUCTION, and that is load-bearing
        twice over.  A streaming transcriber revises: an interim reading
        "excuse me" can be replaced by one reading "excuse", and the buffer's
        own memory bound evicts superseded interims, so a count recomputed
        from what is currently held could go DOWN. A requirement that
        un-qualified itself would take the floor back from a caller who had
        already earned it, mid-sentence, which is worse than never granting
        it. So this is a latch over ``committed finals + latest interim``,
        reset only when a new reply arms the gate.
        """

        return self._words_heard

    @property
    def caller_words_observed(self) -> int:
        """Countable caller words seen on this call so far; never decreases.

        The bounded playout hold asks "has the caller said anything since this
        reply was withheld?", and :attr:`words_heard` cannot answer it. That
        latch belongs to the barge-in criterion: it restarts at every new VAD
        run, refuses words from a run that was not open once the reply became
        audible, and counts only events this gate is HOLDING -- while the
        runtime's own pause moves the session to ``listening``, which stops the
        gating, so on exactly the held-reply path it would count nothing.

        So this is a separate per-call count, fed from :meth:`offer` for every
        speech-carrying event before the gating decision. A caller of it takes a
        snapshot and compares; only the difference means anything. Listening
        noises are worth zero here as in :attr:`words_heard` -- "mm-hm" into
        the silence is not a bid for the floor. An interim raises the reading of
        the utterance in progress and a final commits it (at least as many
        words as its interims showed), so the count is monotone by
        construction and an interim "hello" moves it at once.
        """

        return self._observed_committed_words + self._observed_pending_words

    def _word_reading(self, event: Any) -> tuple[bool, int]:
        """``(backchannel_only, countable_words)`` for one speech event.

        The one place the listening-noise rule is applied to a transcript, so
        the barge-in evidence and the per-call count cannot disagree about
        what a word is.
        """

        text = _stt_event_transcript(event)
        backchannel = is_caller_backchannel_only(text, self._backchannel_tokens)
        return backchannel, 0 if backchannel else _transcript_word_count(text)

    def _observe_words(self, event: Any) -> None:
        _backchannel, words = self._word_reading(event)
        if _stt_event_type_name(event) == _STT_EVENT_TYPE_FINAL_TRANSCRIPT:
            self._observed_committed_words += max(words, self._observed_pending_words)
            self._observed_pending_words = 0
        else:
            self._observed_pending_words = max(self._observed_pending_words, words)

    def note_caller_speaking(self, speaking: bool) -> None:
        """Track the caller's VAD speech run: the DURATION half of the criterion.

        Fed from ``user_state_changed``, which the pinned runtime raises from
        the VAD's own ``on_start_of_speech`` -- the same instant the runtime
        starts accumulating the ``speech_duration`` it compares against
        ``interruption.min_duration``.

        This does NOT weaken the one-signal arming contract, and the
        distinction is worth being exact about because an earlier revision of
        this gate was reverted for blurring it. ARMING -- whether a reply is in
        flight and therefore whether transcripts are withheld -- is still the
        single ``agent_state`` signal, and nothing here touches it. What
        ``user_state`` decides is QUALIFICATION: whether an interruption the
        caller has already earned may proceed. Different decision, different
        owner, and this one has no other public source.

        The measurement is corrected before it is compared, and the correction
        is the whole reason this is safe to gate on. Two latencies sit between
        the caller and these two callbacks, and they do NOT cancel: ``speaking``
        arrives about one Silero ``min_speech_duration`` after the caller really
        started, which under-counts, but ``listening`` arrives only after the
        VAD has waited out :data:`VAD_END_OF_SPEECH_SILENCE_SECONDS` of silence,
        which over-counts by far more. Uncorrected, roughly 300 ms of speech
        measures as 500 ms and the gate would force an interruption the
        runtime's own accumulated-``speech_duration`` gate had refused --
        turning a guard against false barge-in into a cause of one. Subtracting
        the silence leaves the error one-sided the safe way: what remains
        under-counts, so a run measured over the floor really did clear it,
        while one measured just under may have and is refused.

        The qualification LATCHES at run end. A short "stop" is over before the
        transcriber reports it -- Deepgram's results arrive on a ~1000 ms
        cadence -- so a run that qualified must still count when its words turn
        up a moment later, or the commonest one-word interruption there is
        would never take effect.
        """

        now = float(self._clock())
        if speaking:
            if self._caller_speaking_since is None:
                # A NEW run, so both halves of the criterion start over. They
                # have to be earned by the SAME stretch of caller speech or
                # they are not the criterion: a 100 ms "uh" that the
                # transcriber recognises, followed by a separate half-second
                # of wordless noise, would otherwise contribute one half each
                # and cut the agent between them -- which is the burst this
                # setting exists to refuse, admitted by a word that was never
                # part of it.
                self._caller_speaking_since = now
                self._reset_word_evidence()
                self._duration_qualified = False
            # A run active now is a run at-or-after the moment this reply
            # became audible, which is what admits its words.
            self._run_seen_since_audible = True
            return
        started, self._caller_speaking_since = self._caller_speaking_since, None
        if (
            started is not None
            and self._run_seen_since_audible
            and self._spoken_seconds(now, started) >= self._min_duration_seconds
        ):
            self._duration_qualified = True

    def _spoken_seconds(self, now: float, started: float) -> float:
        """Speech the caller has certainly produced, between two state callbacks.

        ONE expression for both the run's end and a run still open, because the
        correction they need is the same and giving them different ones is how
        the two came to disagree. ``user_state`` does not leave ``speaking``
        until the VAD has waited out :data:`VAD_END_OF_SPEECH_SILENCE_SECONDS`
        of silence, so at ANY moment the interval between the two callbacks may
        already contain that much silence at its end -- a caller who stopped
        0.20 s ago still reads as speaking. Subtracting it unconditionally is
        what makes this a lower bound rather than a guess: at the run's end it
        is the real duration, and mid-run it under-counts by at most the
        hangover.

        The cost is that a caller talking over the agent continuously qualifies
        a quarter-second after the floor rather than exactly on it. That is the
        right side to be wrong on, and small next to the transcriber's own
        cadence: the alternative -- trusting the raw interval while the state
        still says ``speaking`` -- lets roughly 300 ms of speech qualify at
        500 ms and forces exactly the short false interruption the conjunction
        exists to reject.
        """

        return (now - started) - self._end_of_speech_silence

    @property
    def duration_requirement_met(self) -> bool:
        """Whether a speech run over this reply has cleared the session's floor."""

        if self._duration_qualified:
            return True
        if not self._run_seen_since_audible or self._caller_speaking_since is None:
            return False
        return (
            self._spoken_seconds(float(self._clock()), self._caller_speaking_since)
            >= self._min_duration_seconds
        )

    def seconds_until_duration_requirement(self) -> float | None:
        """How long until the OPEN run clears the floor, or ``None`` if nothing waits.

        The adapter arms a real timer from this, and without one the criterion
        can sit complete and unnoticed. Words routinely arrive BEFORE the floor
        -- the measured Deepgram phase sweep put a first word at 339 ms against
        a 500 ms floor -- and at that moment the criterion is false. Nothing
        then calls this gate again until the next transcript, a cadence period
        later, or the run ends: the caller has said their word and held the
        floor, both conditions become true at 500 ms, and the agent keeps
        talking for another second because nobody looked.

        ``None`` when there is no open run, when the floor is already cleared,
        or when no word requirement is configured -- in each case a timer would
        fire against a criterion that cannot change by the passage of time
        alone.
        """

        if self._min_interruption_words <= 0 or self._suspended_until_next_reply:
            return None
        if self._words_heard < self._min_interruption_words:
            return None
        if self._duration_qualified or not self._run_seen_since_audible:
            return None
        if self._caller_speaking_since is None:
            return None
        remaining = self._min_duration_seconds - self._spoken_seconds(
            float(self._clock()), self._caller_speaking_since
        )
        return max(0.0, remaining) if remaining > 0 else None

    @property
    def interruption_qualified(self) -> bool:
        """Whether an interruption may proceed: BOTH halves of the criterion.

        The founder's sentence is "roughly half a second of continuous speech
        AND at least one recognized word", so this is a conjunction and never a
        substitution. With no word requirement configured it is ``True`` and
        the runtime's own duration gate decides alone, exactly as it did
        before the setting existed.

        Requiring the duration half HERE as well as in the runtime is not
        belt-and-braces. Once this gate holds the floor, the runtime's decision
        has already been refused and nothing re-runs it; the release below is
        what makes the interruption happen, so the release has to carry the
        whole criterion. Releasing on words alone would let a 0.2 s "uh" --
        which Deepgram transcribes, ``filler_words`` being on -- cut the agent,
        which is the exact defect the transcript gate exists to prevent.

        It is deliberately ``True`` while the gate is stood down. Expiry means
        the reply-in-flight signal is no longer trusted -- the gate is already
        handing the caller's committed turns to the SDK -- and a word
        requirement still holding the floor at that moment would do more than
        delay a barge-in: the pinned runtime refuses to REPLY to a completed
        user turn while the current speech is uninterruptible
        (``agent_activity.py`` :2301-2307 logs "skipping reply to user input"
        and returns), so the caller would be answered by nothing at all. The
        gate's wedge recovery exists to prevent a deaf call and must not
        acquire a second way to cause one.
        """

        if self._min_interruption_words <= 0:
            return True
        if self._suspended_until_next_reply:
            return True
        return (
            self._words_heard >= self._min_interruption_words
            and self.duration_requirement_met
        )

    def note_interruption_qualified(self) -> None:
        """Record the word count at the moment the FULL criterion rose.

        Recorded here rather than when the word half alone is satisfied, and
        the difference is not cosmetic: a short utterance can reach the word
        count and never clear the duration floor, and this field is read as
        "the count when the requirement released the floor". Latching it on the
        word half would report a release that never happened, on exactly the
        short bursts the conjunction exists to refuse -- and it is the number
        the founder's A/B uses to choose a default, so it would argue for the
        wrong one.

        Recorded at the rise rather than at the end of the reply for the
        original reason: by then the count has kept climbing, and the question
        is how many words it took, not how many the caller went on to say.

        The stand-down is deliberately not a qualification. Expiry makes
        :attr:`interruption_qualified` true to keep a wedged call from going
        deaf; nothing was earned, so nothing is recorded.
        """

        if self._min_interruption_words <= 0 or self._suspended_until_next_reply:
            return
        if (
            self._words_at_qualification
            or self._words_heard < self._min_interruption_words
        ):
            return
        self._words_at_qualification = self._words_heard

    def release_on_qualification(self) -> list[Any]:
        """Hand the caller's words over BECAUSE the interruption now qualifies.

        This is the re-trigger, and without it the feature misses its own
        headline case. A caller who says one word for over half a second and
        stops has met the criterion, but the runtime already refused its
        duration-driven attempt against the held floor, and no further VAD
        inference is coming because the caller is silent. Restoring the flag
        alone changes nothing: the reply plays to the end and the caller is
        answered only afterwards.

        Forwarding the buffer is what makes it happen, through the runtime's
        own path rather than a new one: ``on_final_transcript`` ->
        ``_interrupt_by_audio_activity``, which now finds an interruptible
        handle, pauses playout, and commits the turn that answers the caller.
        The alternatives were both worse. Forwarding on words alone reopens the
        short-filler defect this gate exists to close, and ``session.interrupt``
        is a hard cancel that skips the pause and false-interruption machinery
        AND still leaves the caller's words undelivered.

        The caller must restore the speech handle BEFORE these events are
        queued; the stream adapter owns that ordering.
        """

        if not self.gating or not self.interruption_qualified:
            return []
        return self._release("word_and_duration_qualified")

    def set_reply_in_flight(self, in_flight: bool) -> list[Any]:
        """Track the agent's reply; return whatever the change releases.

        Called from the session's ``agent_state_changed`` handler with
        ``agent_state in REPLY_IN_FLIGHT_AGENT_STATES``.  A transition out of
        that set is the ordinary release: the reply is over (or a VAD
        barge-in just paused it), so the caller's words are no longer
        competing with it and the SDK may see them.

        This ONE signal is the whole arming contract for a REPLY,
        deliberately.  Earlier revisions of this change also armed from
        ``speech_created`` and released on a timer over ``user_state``, to
        close the windows before the first audio frame.  Both reconstruct the
        runtime's private speech lifecycle from public events that do not
        carry it, and each attempt traded the reported defect for a new one.
        One unambiguous signal that provably fixes the reported defect beats
        four that argue with each other.

        :meth:`set_opening_turn_in_flight` is not a fifth such signal.  It is
        not derived from the runtime at all: the worker OWNS the opening
        ``session.say`` call site, so it can say exactly when that utterance
        begins and hand the ending back to this same agent-state signal.
        """

        was_gating = self.gating
        self._reply_in_flight = bool(in_flight)
        return self._apply_gating(was_gating)

    def set_opening_turn_in_flight(self, in_flight: bool) -> list[Any]:
        """Track the opening utterance's window.

        Opened by the worker as soon as it has built this gate -- before the
        opening's own advance, before ``session.start``, before anything can
        speak -- so a caller turn that arrives that early is held exactly
        like one arriving after the first audio frame, rather than reaching
        a turn decision before any opening line was ever issued. The say
        call site re-asserts the same flag right before speaking, which by
        then is a no-op. Closed by the opening protection's release handler
        -- when the caller has spoken past the pickup bound, or when the say
        ends -- or, on a path that never reaches a say at all (the callee
        already gone, an authored step with nothing to speak), by an
        explicit close on that path. It outlives the first audio frame on
        purpose: the words a callee says over the opening are held until the
        opening is over, so that a pickup "hello" is answered after the
        question rather than cancelling it.
        """

        was_gating = self.gating
        self._opening_turn_in_flight = bool(in_flight)
        return self._apply_gating(was_gating)

    def note_reply_became_audible(self) -> None:
        """A reply's first audio frame reached the wire; re-scope the word evidence.

        The word requirement is about the reply the caller is TALKING OVER, and
        until this instant there was nothing to talk over.  Between the turn
        commit and the first frame the session reports ``thinking`` for the
        whole LLM/TTS latency -- 2-3 s in this stack -- and backchannels there
        are routine: "mm", "okay", "yeah". Counting them toward the reply that
        follows would satisfy the requirement before the reply was audible, and
        the agent would then be cut by the next wordless burst on the duration
        floor alone. That is not a corner: on campaign traffic a backchannel
        during the thinking window would disarm the word gate for most replies.

        The reset costs a caller who is speaking CONTINUOUSLY across the
        boundary nothing they must re-say, because a streaming transcriber's
        events are cumulative within an utterance: measured on Deepgram
        ``nova-3``, successive events for one segment carried 4, 6, 8 and 12
        words -- each the whole segment so far. So the words re-deliver
        themselves on the next event, within one ~1000 ms cadence period. The
        only speech this discards is an utterance a final already closed --
        the caller who said "okay" and stopped -- which is exactly what it is
        for.

        Called from the ``agent_state -> speaking`` transition, the same single
        runtime signal the rest of this gate is armed by. It is deliberately
        not a second lifecycle reconstructed from public events.
        """

        self._reply_audible = True
        self._rescope_word_evidence()

    def _rescope_word_evidence(self) -> None:
        """The evidence half of :meth:`note_reply_became_audible`.

        Split out because the arming edge in :meth:`_apply_gating` needs the
        same re-scoping WITHOUT declaring the reply audible: at that moment the
        session has only reached ``thinking`` and no frame has been sent.
        """

        # ONE rule, applied to both halves: evidence from a run that is still
        # OPEN is kept, evidence from runs that have closed is discarded.
        #
        # An open run spans the boundary, so it counts -- the caller is talking
        # over audio they can now hear, and they have not stopped to let the
        # agent start. Discarding its words would also break the latch's own
        # invariant by lowering the count mid-run, and it would rest the
        # recovery on an assumption nothing guarantees: that the transcriber
        # emits ANOTHER event after the boundary. Deepgram's events are
        # cumulative and would re-deliver the words, but that is an observation
        # about one provider, not a contract, and a caller who falls silent
        # just after the boundary may produce nothing further at all.
        #
        # A caller who spoke and STOPPED before the first frame has no run open
        # here, so their evidence goes -- including the transcript that arrives
        # for it a moment later, which ``_count_words`` refuses because no run
        # has been active since the boundary. Deepgram's ~1000 ms cadence is
        # comfortably longer than that gap, so the late delivery is the normal
        # case rather than a rare one, and resetting arrival-time counters
        # alone would miss all of it.
        self._run_seen_since_audible = self._caller_speaking_since is not None
        if not self._run_seen_since_audible:
            self._reset_word_evidence()
            self._duration_qualified = False

    def _apply_gating(self, was_gating: bool) -> list[Any]:
        """Release on the edge of the COMBINED predicate, deliberately.

        Releasing when either flag alone drops would hand the caller's words to
        the SDK while the other protection is still up -- which is the
        cancellation both flags exist to prevent, one moment later.  So a
        transcript held during the greeting is released when the greeting's
        playout ends, not when the greeting merely becomes audible.
        """

        now_gating = self.gating
        if was_gating and not now_gating:
            return self._release("agent_reply_ended")
        if now_gating and not was_gating:
            # A genuinely new reply. Any suspension raised for the previous
            # one has served its purpose.
            self._suspended_until_next_reply = False
            # ...and so has its word evidence. The requirement is per reply:
            # what the caller said over the LAST answer cannot pre-authorize
            # interrupting this one, and a latch carried across the boundary
            # would make the second reply interruptible from its first frame
            # while reporting a count nobody uttered against it.
            #
            # Re-scoped AGAIN when this reply becomes audible; see
            # :meth:`note_reply_became_audible` for why the thinking window
            # cannot pre-qualify the playout that follows it.
            #
            # Evidence only: the reply is NOT audible here. The session has
            # just reached ``thinking`` (or the worker has just opened the
            # greeting's synthesis window) and no frame has been sent.
            self._reply_audible = False
            self._rescope_word_evidence()
        return []

    def offer(self, event: Any) -> list[Any]:
        """Return the events to forward downstream for one incoming event."""

        observed_at = float(self._clock())
        if _stt_event_type_name(event) == _STT_EVENT_TYPE_RECOGNITION_USAGE:
            # Usage is metering, not speech.  The SDK exempts it from its own
            # hold for the same reason: delaying it would delay billing
            # evidence without affecting a single interruption decision.
            return [event]
        if _stt_event_type_name(event) in _STT_EVENT_TYPES_CARRYING_SPEECH:
            self._observe_words(event)
            if self._observer is not None:
                try:
                    self._observer.note_speech_event(
                        is_final=_stt_event_type_name(event) == _STT_EVENT_TYPE_FINAL_TRANSCRIPT,
                        text=_stt_event_transcript(event),
                        # Only where the pinned adapter measures it; a plugin
                        # placeholder must never read as a measurement.
                        confidence=(
                            _stt_event_confidence(event) if self._stt_reports_confidence else None
                        ),
                    )
                except Exception:  # noqa: BLE001 - a diagnostic feed cannot decide media
                    pass
        if not self.gating or self._suspended_until_next_reply:
            reason = "gate_stood_down" if self.gating else "no_reply_in_flight"
            released = self._release(reason)
            self._remember_released_final(event, observed_at)
            return released + [event]

        self._held.append((event, observed_at))
        self._held_total += 1
        if _stt_event_type_name(event) in _STT_EVENT_TYPES_CARRYING_SPEECH:
            self._pending_speech += 1
            self._count_words(event)
        if self._holding_since is None:
            self._holding_since = float(self._clock())
        self._enforce_event_limit()
        # Nothing is forwarded while a reply is in flight and the gate still
        # trusts that signal.  Expiry is the one exception, and it means the
        # signal is no longer trusted: it hands back the caller's committed
        # turns, which must reach the stream rather than be dropped here.
        return self.release_expired()

    def _reset_word_evidence(self) -> None:
        """Forget the words; the run they would have qualified is over.

        Deliberately NOT strict per-run attribution, because the signals do not
        support it and pretending otherwise would break the commonest
        interruption there is. A transcriber reports a word AFTER it was
        spoken -- Deepgram on a ~1000 ms cadence -- so a caller who says one
        word and stops is already silent when it arrives, and words that only
        counted while a run was open would never count at all.

        So the evidence belongs to the current run, or to the run that just
        ended if none is open. What remains is bounded and named rather than
        hidden: a word spoken in one run whose transcript lands after the NEXT
        run has begun is attributed to that next run. That interruption is never
        one the stack would otherwise have refused: the next run had to clear
        the duration floor on its own, which is exactly what the runtime's own
        gate asks and what a call with no word requirement already acts on. The
        residual is a failure to IMPROVE on the default in one sequence, never
        a regression below it. Closing the remainder needs speech-time
        correlation the stack cannot do: the provider's ``SpeechData``
        timestamps are stream-relative -- ``start_time_offset`` starts at zero
        and accumulates reconnect gaps -- so neither endpoint can be placed on
        this clock.

        Their DIFFERENCE is usable, and is what
        :func:`_stt_event_measured_duration` reads for turn boundaries. (An
        earlier version of this note also said the alternative-level
        ``end_time`` was the first word's end rather than the last. That was
        true of livekit-plugins-deepgram 1.6.6; the pinned 1.7.1 reports
        ``alt["words"][-1]["end"]``, and Azure reports the recognised
        segment's own offset plus duration.)
        """

        self._committed_words = 0
        self._pending_interim_words = 0
        self._words_heard = 0
        self._words_at_qualification = 0

    def _count_words(self, event: Any) -> None:
        """Fold one held speech event into the reply's word evidence.

        Interims and finals are counted DIFFERENTLY, because a streaming
        transcriber means two different things by them and summing both would
        triple-count an ordinary utterance: the interims for one utterance are
        successive readings of the SAME words ("excuse", "excuse me", "excuse
        me can"), and the final restates all of them once more.

        So a final COMMITS its words -- the utterance is closed and the next
        interim starts a new one -- while an interim only replaces the pending
        reading of the utterance still in progress.  ``committed + pending`` is
        then what the caller has actually said, and the latch in
        :attr:`words_heard` keeps it from falling when the transcriber revises
        downward or the buffer's memory bound evicts a superseded interim.
        """

        if not self._run_seen_since_audible:
            # Words the caller produced before this reply was audible, arriving
            # late. Counting them would qualify an interruption of a reply
            # nobody talked over. See :meth:`note_reply_became_audible`.
            return
        # A listening noise, not a request for the floor.  Worth ZERO words
        # rather than dropped from the buffer: the caller still said it, the
        # transcript gate still releases it when the reply ends, and the caller
        # is still answered -- they simply do not cut the agent off to do it.
        # That is the market norm (Vapi ``acknowledgementPhrases``, Retell
        # ``enable_backchannel``) and the whole behavioural difference here.
        #
        # It is a word-EVIDENCE rule, so it lives in ``_word_reading`` and
        # nowhere else.
        # ``interruption_qualified`` reads only ``words_heard``, so this one
        # seam is sufficient; a second check beside the qualification would be
        # a second copy of the policy to keep in step.
        backchannel, words = self._word_reading(event)
        if _stt_event_type_name(event) == _STT_EVENT_TYPE_FINAL_TRANSCRIPT:
            if backchannel and self._reply_audible:
                # TWO scopes the zero-word rule above does NOT have, because
                # they answer different questions. The rule decides whether an
                # interruption may proceed; this number says what the caller
                # did to an agent that was speaking.
                #
                # FINALS ONLY: one "yeah" arrives as several successive
                # interims plus a final, so a per-event counter would report it
                # as four and would record an interim "yeah" that the final
                # revises to "yeah but wait" as a backchannel that never
                # happened. A final is the caller's committed utterance.
                #
                # AUDIBLE ONLY: a backchannel in the thinking window, or during
                # a greeting's synthesis, was not said over anything.
                #
                # The stated limit of that choice: an utterance whose interims
                # were backchannels but which never reaches a final -- a stream
                # cut at teardown -- is not counted, even though the zero-word
                # rule did apply to it. Finals-only under-counts that case and
                # over-counts nothing, which is the right direction for a
                # number a comparison is read from.
                self._backchannel_utterances = min(
                    4_096, self._backchannel_utterances + 1
                )
            self._committed_words = min(4_096, self._committed_words + words)
            self._pending_interim_words = 0
        else:
            self._pending_interim_words = min(4_096, words)
        self._words_heard = min(
            4_096,
            max(self._words_heard, self._committed_words + self._pending_interim_words),
        )

    def _enforce_event_limit(self) -> None:
        """Bound memory by coalescing, never by forwarding and never by losing a turn.

        A limit that flushed the buffer downstream would hand the SDK a replay
        of interims and finals mid-reply and recreate the exact false
        interruption this gate exists to prevent -- a safety valve that
        reproduces the defect is not a safety valve.

        So it discards instead, and it discards the OLDEST INTERIM first: an
        interim is superseded by every later interim and by the final that
        closes it, so dropping one loses stale partial text.  A final is the
        caller's committed turn and is kept while any interim remains, because
        dropping it would silently ignore someone who spoke once and then
        waited -- reachable on a long reply, and a worse failure than the one
        being fixed.  Only when the cap is full of finals alone does the
        oldest final go, and that is counted.
        """

        while len(self._held) > self._max_held_events:
            index = self._oldest_discardable_index()
            discarded = self._held.pop(index)
            self._discarded_total += 1
            if _stt_event_type_name(discarded[0]) in _STT_EVENT_TYPES_CARRYING_SPEECH:
                self._pending_speech = max(0, self._pending_speech - 1)

    def _oldest_discardable_index(self) -> int:
        """Prefer the oldest superseded event; fall back to the oldest of all."""

        for index, (event, _) in enumerate(self._held):
            if _stt_event_type_name(event) != _STT_EVENT_TYPE_FINAL_TRANSCRIPT:
                return index
        return 0

    def hold_deadline_remaining(self) -> float | None:
        """Seconds until the current hold episode expires, or None if idle.

        The adapter arms a real timer from this.  Checking the deadline only
        when the next STT event happens to arrive would make the valve
        useless in the one situation it exists for: a stuck reply state with
        a silent caller produces no further events at all.
        """

        if not self._held or self._holding_since is None:
            return None
        return max(
            0.0, self._holding_since + self._max_hold_seconds - float(self._clock())
        )

    def release_expired(self) -> list[Any]:
        """Stand the gate down once the hold deadline passes.

        Returns the caller's committed FINAL transcripts, because a caller who
        spoke once early in a long reply must still be answered; superseded
        interim text is dropped and counted rather than replayed.

        This is wedge recovery, not a turn-taking decision, which is why the
        bound sits far above any legitimate reply: standing gating down inside
        a long-but-valid answer would recreate the defect on ordinary traffic.
        """

        remaining = self.hold_deadline_remaining()
        if remaining is None or remaining > 0:
            return []
        kept = [
            held
            for held in self._held
            if _stt_event_type_name(held[0]) == _STT_EVENT_TYPE_FINAL_TRANSCRIPT
        ]
        self._discarded_total += len(self._held) - len(kept)
        self._held = []
        self._holding_since = None
        self._pending_speech = 0
        self._released_total += len(kept)
        # Crossing this bound means the reply-state signal is not trustworthy:
        # no reply, generation and playout together, runs this long. So the
        # gate stops trusting it rather than silently holding forever -- the
        # caller's committed turns are handed over, superseded interim text is
        # dropped, and gating stands down until the next reply re-arms it. A
        # deaf call is worse than a late interruption.
        self._suspended_until_next_reply = True
        self._release_reasons.append("hold_time_limit_stood_down")
        del self._release_reasons[:-8]
        return self._forward(kept)

    def drain(self) -> list[Any]:
        """Release everything still held; the STT stream is ending."""

        return self._release("stream_ended")

    def _release(self, reason: str) -> list[Any]:
        if not self._held:
            return []
        released = self._held
        self._held = []
        self._holding_since = None
        self._pending_speech = 0
        self._released_total += len(released)
        self._release_reasons.append(reason)
        del self._release_reasons[:-8]
        return self._forward(released)

    def _forward(self, held: list[tuple[Any, float]]) -> list[Any]:
        for event, observed_at in held:
            self._remember_released_final(event, observed_at)
        return [event for event, _ in held]

    def _remember_released_final(self, event: Any, observed_at: float) -> None:
        if _stt_event_type_name(event) != _STT_EVENT_TYPE_FINAL_TRANSCRIPT:
            return
        # A malformed provider event is still forwarded by contract; it merely
        # cannot contribute a timestamp correlation record. The reader is
        # shared with the word count so there is one place where a provider
        # object is allowed to misbehave, rather than two that must agree.
        text = _stt_event_transcript(event)
        if not text:
            return
        if self._observer is not None:
            try:
                self._observer.note_forwarded_final(text)
            except Exception:  # noqa: BLE001 - a diagnostic feed cannot decide media
                pass
        observation = CallerFinalObservation(
            observed_at=observed_at,
            measured_duration_seconds=_stt_event_measured_duration(event),
            # Read only where the pinned adapter measures it. Azure's constant
            # 1.0 is a plugin placeholder, and forwarding it would put a
            # perfect score on every Sinhala utterance.
            confidence=(
                _stt_event_confidence(event) if self._stt_reports_confidence else None
            ),
        )
        self._released_final_observations.append((text, observation))
        del self._released_final_observations[: -self._max_held_events]

    def consume_final_transcript_observation(
        self, transcript: str
    ) -> CallerFinalObservation | None:
        """Return what the recogniser reported about one released final.

        Exact text plus FIFO order is the stable seam across LiveKit's new
        event object. Stale preceding entries are discarded rather than lent
        to a different caller utterance.

        This CONSUMES the match. It is deliberately the ONLY consuming reader
        the gate exposes: a second spelling of the same read is a second owner
        of the FIFO, and whichever ran first would silently take the entry the
        other needed. A ``consume_final_transcript_observed_at`` spelling used
        to sit beside it and was removed for exactly that reason.

        One production caller, in ``production_agent``, which hands this bound
        method to ``SemanticTurnEventRecorder``; every other reference in the
        tree is a test driving the gate directly.
        """

        normalized = " ".join(str(transcript or "").split())
        if not normalized:
            return None
        for index, (text, observation) in enumerate(self._released_final_observations):
            if text == normalized:
                del self._released_final_observations[: index + 1]
                return observation
        return None



class CallerTranscriptGateStream:
    """Async adapter turning :class:`CallerTranscriptGate` into an STT filter.

    The pure gate decides; this object only moves events between the SDK's
    STT generator and the agent's ``stt_node`` output, plus the session event
    that releases a held buffer.  A release can arrive while the generator is
    parked on the source stream, so the two are decoupled by a queue rather
    than by polling.

    Errors from the wrapped stream are re-raised in the generator on purpose:
    the SDK's ``_STTPipeline._stt_pump`` recreates the node after an
    ``APIError`` and swallowing it here would silently end transcription for
    the rest of the call.  A failed stream must NOT release the buffer,
    though: the pump reconnects and would hand the SDK exactly the transcripts
    the gate is withholding, so a transient provider blip would cancel playout
    the VAD threshold had already refused.  The buffer lives on the gate, not
    on the generator, so it survives the reconnect and is released by the
    ordinary trigger -- the agent finishing its reply.
    """

    _DONE = object()

    def __init__(
        self,
        gate: CallerTranscriptGate,
        *,
        on_interruption_qualified: Callable[[int], bool] | None = None,
        on_caller_transcript: Callable[[str], None] | None = None,
    ) -> None:
        self._gate = gate
        self._on_caller_transcript = on_caller_transcript
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._deadline: Any = None
        # The criterion's RELEASE edge. The gate decides; this reports, and
        # the listener answers whether the floor is actually clear -- see
        # :meth:`_settle_qualification` for why a forward is not safe without
        # that answer.
        # Level-tracked rather than event-driven because every mutation of the
        # gate can move it, including the ones that lower it (a new reply
        # resets the count) -- and only the rise is news.
        self._on_interruption_qualified = on_interruption_qualified
        self._interruption_qualified = gate.interruption_qualified
        self._floor_deadline: Any = None

    @property
    def gate(self) -> CallerTranscriptGate:
        return self._gate

    def set_reply_in_flight(self, in_flight: bool) -> None:
        # The previous reply's timer has nothing left to settle. Re-arming
        # below covers this, because the new reply's reset leaves the criterion
        # waiting on words rather than on the clock -- but a timer that only
        # stops as a side effect of another decision is one refactor away from
        # firing against a reply it was never armed for.
        self._cancel_floor_deadline()
        released = self._gate.set_reply_in_flight(in_flight)
        self._settle_qualification()
        for event in released:
            self._queue.put_nowait(event)
        self._arm_hold_deadline()

    def set_opening_turn_in_flight(self, in_flight: bool) -> None:
        released = self._gate.set_opening_turn_in_flight(in_flight)
        self._settle_qualification()
        for event in released:
            self._queue.put_nowait(event)
        self._arm_hold_deadline()

    def note_reply_became_audible(self) -> None:
        """Re-scope the interruption evidence to the reply the caller can hear.

        Routed through the adapter rather than called on the gate directly so
        the tracked level follows the reset DOWN. A listener that kept a stale
        ``True`` would never report the next rise, and the reply it belongs to
        would hold the floor until it ended.
        """

        self._gate.note_reply_became_audible()
        self._settle_qualification()

    def note_caller_speaking(self, speaking: bool) -> None:
        """Feed the duration half, and re-evaluate on the run's own edges.

        Both edges matter, for different reasons. The start of a run lowers
        nothing but may end a stale qualification's window; the END of a run is
        the last moment anything happens at all when the caller says one word
        and stops, so a criterion that was only evaluated when a transcript
        arrived would miss the run that qualified between two of them.
        """

        self._gate.note_caller_speaking(speaking)
        self._settle_qualification()

    def _settle_qualification(self) -> None:
        """Report the rising edge of the full criterion, and act on it.

        Called after EVERY gate mutation, and before the events that mutation
        released are queued.  Ordering matters in one direction only and it is
        this one: the expiry path stands the gate down and releases the
        caller's committed finals in the same call, and those finals reach a
        runtime that refuses to answer a completed turn while the current
        speech is uninterruptible.  Restoring first makes that impossible;
        restoring afterwards would make it a race against the media loop.

        The same ordering carries the re-trigger: the listener restores the
        speech handle, and only then are the qualifying transcripts queued
        toward the runtime that will act on them.
        """

        floor_is_clear = True
        qualified = self._gate.interruption_qualified
        rose = qualified and not self._interruption_qualified
        self._interruption_qualified = qualified
        if not rose:
            self._arm_floor_deadline()
            return
        self._cancel_floor_deadline()
        self._gate.note_interruption_qualified()
        if self._on_interruption_qualified is not None:
            try:
                floor_is_clear = self._on_interruption_qualified(self._gate.words_heard)
            except Exception:
                # The floor was NOT given back, so the forward below must not
                # happen either. Handing the caller's committed turn to a
                # runtime whose current speech is still uninterruptible is not
                # a degraded outcome, it is the dropped turn this whole
                # ordering exists to prevent: the runtime logs "skipping reply
                # to user input" and returns. Keeping the buffer instead leaves
                # the reply uninterruptible for its remaining playout and hands
                # the turn over at the ordinary release, which is the outcome
                # this stack had before the word requirement existed rather
                # than a lost caller.
                #
                # The tracked level goes back DOWN before returning. Leaving it
                # raised would record a qualification that never took effect,
                # and since only a RISING edge acts, no later event could ever
                # retry: one transient failure would hold the floor for the
                # whole reply instead of until the next gate event.
                self._interruption_qualified = False
                return
            if not floor_is_clear:
                # The listener gave back everything IT was holding and the
                # reply is still uninterruptible, so somebody else holds the
                # floor. The opening-turn protection is that somebody: it
                # keeps the greeting uninterruptible while it plays, until the
                # caller has spoken past the pickup bound, so a callee who
                # speaks over the opening can satisfy the
                # criterion while no word gate is holding anything at all.
                #
                # Forwarding there is not a degraded interruption, it is a lost
                # turn: the transcripts reach a runtime that refuses to answer
                # a completed turn against an uninterruptible speech, logs
                # "skipping reply to user input", and returns. Holding them
                # instead costs nothing -- the window is bounded by the say,
                # and the ordinary release hands them over when it closes.
                self._interruption_qualified = False
                return
        for event in self._gate.release_on_qualification():
            self._queue.put_nowait(event)

    def _arm_floor_deadline(self) -> None:
        """Keep one timer alive for a criterion waiting only on the clock.

        The words are in and the caller is still speaking; the sole remaining
        condition is time. Nothing else would look again until the next
        transcript or the end of the run, so the interruption the caller has
        earned would land up to a cadence period late.
        """

        remaining = self._gate.seconds_until_duration_requirement()
        if remaining is None:
            self._cancel_floor_deadline()
            return
        if self._floor_deadline is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No media loop: the next gate event settles it instead, which is
            # the pre-timer behaviour rather than a lost interruption.
            return
        self._floor_deadline = loop.call_later(remaining, self._on_floor_deadline)

    def _cancel_floor_deadline(self) -> None:
        if self._floor_deadline is not None:
            self._floor_deadline.cancel()
            self._floor_deadline = None

    def _on_floor_deadline(self) -> None:
        self._floor_deadline = None
        if not self._gate.gating:
            # The reply this timer was armed for is over. Cancellation on the
            # reply transition covers the ordinary case, but a timer that has
            # already fired is queued on the loop and cannot be recalled -- and
            # settling here would report a qualification against a reply that
            # no longer exists, and record its word count as the one that
            # released a floor nobody was holding.
            return
        self._settle_qualification()

    def _arm_hold_deadline(self) -> None:
        """Keep exactly one timer alive for the current hold episode."""

        remaining = self._gate.hold_deadline_remaining()
        if remaining is None:
            if self._deadline is not None:
                self._deadline.cancel()
                self._deadline = None
            return
        if self._deadline is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet; the next offer/release re-arms. The gate still
            # enforces the deadline on its own next event either way.
            return
        self._deadline = loop.call_later(remaining, self._on_hold_deadline)

    def _on_hold_deadline(self) -> None:
        self._deadline = None
        released = self._gate.release_expired()
        self._settle_qualification()
        for event in released:
            self._queue.put_nowait(event)
        self._arm_hold_deadline()

    async def filter(self, source: AsyncIterable[Any]) -> AsyncIterator[Any]:
        failure: list[BaseException] = []

        async def pump() -> None:
            ended_cleanly = False
            try:
                async for event in source:
                    # Closing is revocable before the SDK receives held words.
                    # Observe synchronously before offer/release can queue them;
                    # a post-playout pending count would miss already queued text.
                    if self._on_caller_transcript is not None:
                        self._on_caller_transcript(_stt_event_transcript(event))
                    admitted_events = self._gate.offer(event)
                    self._settle_qualification()
                    for admitted in admitted_events:
                        self._queue.put_nowait(admitted)
                    self._arm_hold_deadline()
                # Only a source that ran out of audio has nothing left to
                # protect: the caller is gone, so the last turn is released
                # rather than lost.  A failure or a cancellation keeps the
                # buffer on the gate for the reconnected stream.
                ended_cleanly = True
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # re-raised by the consumer below
                failure.append(exc)
            finally:
                if ended_cleanly:
                    for admitted in self._gate.drain():
                        self._queue.put_nowait(admitted)
                self._queue.put_nowait(self._DONE)

        task = asyncio.create_task(pump())
        # A reconnect creates a fresh generator over the SAME gate, so the
        # buffer it inherits may already be part-way through its hold episode
        # while the previous generator's timer was cancelled on the way out.
        # Re-arm here rather than waiting for the next STT event -- a silent
        # reconnect is exactly when none arrives.
        self._arm_hold_deadline()
        # The same argument, for the same reason, about the other timer. A
        # provider blip after the caller's words arrived but before the floor
        # elapsed would otherwise leave a complete criterion unnoticed until
        # some later event happened to look, and a caller who is still speaking
        # produces no such event.
        self._arm_floor_deadline()
        try:
            while True:
                item = await self._queue.get()
                if item is self._DONE:
                    if failure:
                        raise failure[0]
                    return
                yield item
        finally:
            self._cancel_floor_deadline()
            if self._deadline is not None:
                self._deadline.cancel()
                self._deadline = None
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


# The longest stretch of continuous caller speech that still reads as the
# phone being picked up rather than as a bid for the floor.  Measured on the
# finconnect Sinhala batch of 2026-09-17 by replaying the recorded caller
# tracks through the pinned Silero VAD: the seventeen bursts that landed on
# the opening line were 0.13-0.80 s long ("hello?", a syllable, a breath) and
# every one of them cut the greeting under the 250 ms threshold that agent
# runs, before the identity question had been asked.  The utterances that
# were genuinely the caller taking the floor were 1.3 s and longer.  One
# second sits in that gap with a margin on both sides; it is a property of
# how people answer a phone, not of a language, so it is not a per-language
# or per-agent setting.
OPENING_PICKUP_MAX_SECONDS = 1.0


class OpeningTurnProtection:
    """Keep the opening utterance alive until the caller has really taken the floor.

    ``production_agent`` opens a managed call with
    ``session.say(first_message)``.  Two windows in that utterance's life were
    each measured to kill it before the caller had heard what it asked:

    * **Before the first audio frame.**  In ``livekit-agents==1.8.1`` that path
      never reports ``thinking``: ``_tts_task_impl`` (agent_activity.py:2950)
      moves the session to ``speaking`` only from ``_on_first_frame`` (:3006),
      so for the whole synthesis window the caller-transcript gate's arming
      signal is silent and any non-empty transcript reaches
      ``on_final_transcript`` -> ``_interrupt_by_audio_activity`` ->
      ``_cancel_speech_pause``, which calls ``handle.interrupt()`` on a
      greeting that has not made a sound.  Measured on
      ``call_MPXHrjeG3WhhMAYQ``: the callee's "hello?" transcribed as "love"
      killed the greeting and the call opened in silence.
    * **While it plays.**  A callee answers the phone with "hello?" 0.4-1.4 s
      after the line connects, while the agent is still saying who it is.
      That burst is 0.13-0.8 s long and carries no answer to anything, but
      under the runtime's duration threshold it is an interruption like any
      other: it cut the opening on thirteen of the fifteen human-answered
      calls of the 2026-09-17 batch at 0.9-1.6 s of a 4.5 s line, the
      identity question was never heard, and every reply routed to the
      workflow's "unclear" branch until the run gave up.

    So the handle is created with ``allow_interruptions=False`` and stays that
    way from the say until ONE of two things happens: the caller speaks for
    longer than :data:`OPENING_PICKUP_MAX_SECONDS` continuously -- they are
    taking the floor, and the opening is handed back to the ordinary
    interruption rules -- or the utterance ends, on its own or by failing.
    Handing it back does not itself cut the line: on the managed stack's
    configured ``false_interruption_timeout``/``resume_false_interruption``,
    the runtime's next VAD tick PAUSES the opening and arms a resume window
    (``_interrupt_by_audio_activity``, agent_activity.py:2159, through
    ``_pause_enabled``, :4614) rather than interrupting it outright; only a
    committing signal -- a final transcript, or a turn that commits --
    interrupts the paused handle (``_cancel_speech_pause``, :4745), and the
    pause resumes if none arrives within the timeout, exactly as any other
    reply's false interruption would.  Both of the runtime's pause sites and
    ``_cancel_speech_pause`` are gated on ``allow_interruptions``, and so is
    the transcript-triggered path, so a pickup-shaped burst can neither pause
    the opening nor cancel it: its transcript is withheld by the
    caller-transcript gate for as long as the opening is in flight and
    handed to the runtime when the opening ends, exactly as a transcript
    during any reply is.

    The caller's speech run is the ``user_state`` signal the runtime raises
    from the VAD's own start-of-speech, corrected the same way
    :meth:`CallerTranscriptGate.note_caller_speaking` corrects it: the state
    leaves ``speaking`` only after :data:`VAD_END_OF_SPEECH_SILENCE_SECONDS`
    of silence, so a run is measured as ``elapsed - hangover`` and the timer
    that ends the hold is armed for the bound plus that hangover.  A run that
    began before the first audio frame is timed from the frame: until then the
    caller was not talking over anything.  The correction is per RUN, not
    across runs: two separate 0.6 s bursts with a 0.3 s gap between them are
    two ``user_state`` runs, each forgiven on its own end, never merged into
    one 1.5 s run that would read as taking the floor.

    The release needs a timer, because nothing else would look: a caller
    speaking continuously produces no further ``user_state`` edge until they
    stop.  It is armed on the running loop through ``schedule``, which tests
    replace with a manual scheduler; with no loop the hold simply lasts until
    the say ends, which the say bounds.

    Handing the floor back is one callback, ``on_release``, bound once by the
    worker: it restores the flag, closes the transcript gate's opening window
    and registers the utterance for interruption telemetry.  Every close --
    the caller's run, the say's ``finally`` -- goes through it, so a later
    edit cannot hand the handle back on one path and forget the gate on the
    other.  ``disarm`` still returns the handle for callers that restore it
    themselves; restoring twice is harmless, the flag only ever goes back to
    ``True``.

    Loop-affine by contract, like the transcript gate beside it: ``arm``,
    ``attach`` and ``disarm`` run in the coroutine that owns the say, the two
    ``note_*`` methods and the timer callback on the session's own event
    handlers, all on the single asyncio loop driving the media session, and
    none of them awaits.

    Deliberately not enabled outside managed phone calls.  Browser tests, the
    simulations lab, the standalone demo and the platform release canary all
    keep ``allow_interruptions=True`` on their opening line, byte for byte.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        clock: Callable[[], float] = time.monotonic,
        schedule: Callable[[float, Callable[[], None]], Callable[[], None] | None]
        | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._clock = clock
        self._schedule = schedule if schedule is not None else _schedule_on_running_loop
        # Not constructor knobs: both are properties of how people answer a
        # phone, not of a call, an agent or a language. ``clock``/``schedule``
        # stay injectable because they are test seams onto the loop, not
        # tunables of the decision itself.
        self._pickup_max = OPENING_PICKUP_MAX_SECONDS
        self._hangover = VAD_END_OF_SPEECH_SILENCE_SECONDS
        self._on_release: Callable[[Any], None] | None = None
        self._armed = False
        self._armed_at: float | None = None
        self._audible = False
        self._handle: Any = None
        # The caller's VAD state is tracked whether or not a window is open:
        # a run that is already going when the first frame lands must be timed
        # from that frame, and only the state can say it is going.
        self._caller_speaking = False
        self._run_started_at: float | None = None
        self._cancel_timer: Callable[[], None] | None = None
        self._turns_protected = 0
        self._turns_reached_audio = 0
        self._protected_ms_max = 0
        self._pickups_ignored = 0
        self._released_to_caller = 0
        self._release_handler_failures = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        return self._armed

    @property
    def audible(self) -> bool:
        """Whether the protected utterance has reached the wire."""

        return self._armed and self._audible

    @property
    def stats(self) -> dict[str, Any]:
        """Bounded per-call counters for the interruption summary."""

        return {
            # The customer-visible claim: an opening utterance ran under
            # protection AND reached the wire.  Protection that never
            # produced audio is not a greeting that played.
            "greeting_protected": bool(self._turns_reached_audio),
            "opening_turns_protected": self._turns_protected,
            "opening_turns_reached_audio": self._turns_reached_audio,
            "opening_protection_ms_max": self._protected_ms_max,
            # Caller speech runs that ended under the pickup bound while the
            # opening was audible: each one would have cut it before.
            "opening_pickups_ignored": self._pickups_ignored,
            # Openings handed back because the caller kept talking past the
            # bound: the runtime's next VAD tick pauses these (and arms a
            # resume window) or interrupts them outright, per the managed
            # stack's false-interruption configuration -- handing back is not
            # itself the cut.
            "opening_released_to_caller": self._released_to_caller,
            # The release handler is bound once by the worker and is best-effort
            # bookkeeping; a failure inside it must not be silent, or the
            # transcript gate's opening window can be left uncleared with
            # nothing to show for it. No caller text reaches this counter.
            "opening_release_handler_failures": self._release_handler_failures,
        }

    def bind_release(self, callback: Callable[[Any], None]) -> None:
        """Set the one handler every close hands the handle to."""

        self._on_release = callback

    def arm(self) -> bool:
        """Open the window for one opening utterance; ``False`` if it is not ours.

        Re-arming while already armed is refused rather than nested: the
        worker speaks the opening utterances one after another, and a second
        arm would silently discard the first handle's restore.
        """

        if not self._enabled or self._armed:
            return False
        self._armed = True
        self._armed_at = float(self._clock())
        self._audible = False
        self._handle = None
        self._run_started_at = None
        self._turns_protected += 1
        return True

    def attach(self, handle: Any) -> None:
        """Remember the handle created uninterruptible, so it can be restored."""

        if self._armed:
            self._handle = handle

    def note_agent_state(self, new_state: Any) -> None:
        """The first ``speaking`` transition: the opening is now audible.

        Nothing is handed back here any more.  From this frame the caller can
        talk over the opening, so this is where a run already in progress
        starts being measured against the pickup bound.
        """

        if not self._armed or self._audible or str(new_state) != "speaking":
            return
        self._audible = True
        self._turns_reached_audio += 1
        if self._caller_speaking:
            self._start_run(float(self._clock()))

    def note_caller_speaking(self, speaking: bool) -> None:
        """Feed the caller's VAD run; the only thing that can end the hold early."""

        speaking = bool(speaking)
        if speaking == self._caller_speaking:
            return
        self._caller_speaking = speaking
        if not self._armed or not self._audible:
            return
        if speaking:
            self._start_run(float(self._clock()))
            return
        self._stop_timer()
        if self._run_started_at is not None:
            # The run ended under the bound (the timer would have closed the
            # window otherwise): the phone being picked up, not an answer.
            self._pickups_ignored = min(4_096, self._pickups_ignored + 1)
        self._run_started_at = None

    def _start_run(self, now: float) -> None:
        self._stop_timer()
        self._run_started_at = now
        self._cancel_timer = self._schedule(
            self._pickup_max + self._hangover, self._release_to_caller
        )

    def _release_to_caller(self) -> None:
        """The timer: the caller has held the floor past the bound."""

        self._cancel_timer = None
        if not self._armed or not self._audible or self._run_started_at is None:
            return
        self._released_to_caller = min(4_096, self._released_to_caller + 1)
        self._run_started_at = None
        self._hand_back(self._close())

    def disarm(self) -> Any:
        """Close the window unconditionally; return the handle to restore."""

        if not self._armed:
            return None
        handle = self._close()
        self._hand_back(handle)
        return handle

    def _hand_back(self, handle: Any) -> None:
        if self._on_release is None:
            return
        try:
            self._on_release(handle)
        except Exception:
            # The handler is best-effort bookkeeping on a media callback; the
            # caller that asked for the close still gets the handle back.
            # Counted rather than silent: a failure here can leave the
            # transcript gate's opening window uncleared with nothing else to
            # show for it.
            self._release_handler_failures = min(4_096, self._release_handler_failures + 1)

    def _stop_timer(self) -> None:
        cancel, self._cancel_timer = self._cancel_timer, None
        if cancel is not None:
            try:
                cancel()
            except Exception:
                pass

    def _close(self) -> Any:
        self._stop_timer()
        handle, self._handle = self._handle, None
        started, self._armed_at = self._armed_at, None
        self._armed = False
        self._audible = False
        self._run_started_at = None
        if started is not None:
            elapsed_ms = max(0, int(round((float(self._clock()) - started) * 1_000)))
            self._protected_ms_max = max(
                self._protected_ms_max, min(600_000, elapsed_ms)
            )
        return handle


def _schedule_on_running_loop(
    delay: float, callback: Callable[[], None]
) -> Callable[[], None] | None:
    """Arm ``callback`` on the running loop; ``None`` when there is no loop.

    Without a loop the opening protection has no timer, so a caller who keeps
    talking cannot take the opening back early; the say still ends the hold,
    which is the bounded direction to fail in.
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.call_later(delay, callback).cancel


def restore_speech_interruptions(handle: Any) -> bool:
    """Put ``allow_interruptions`` back to ``True`` on one speech handle.

    Best-effort by contract: this runs from a media callback and from a
    ``finally``, and a handle that has already ended or a runtime that moved
    the setter must not turn a played greeting into a failed call.  Setting
    the flag back to ``True`` is always legal in the pinned runtime -- only
    clearing it on an already-interrupted handle raises.
    """

    if handle is None:
        return False
    try:
        handle.allow_interruptions = True
    except Exception:
        return False
    return True


class InterruptionWordGate:
    """Hold the floor for the agent until the caller has actually said something.

    The DURATION half of the barge-in criterion is the runtime's own
    (``interruption.min_duration``, set to
    :data:`MANAGED_PSTN_INTERRUPTION_MIN_DURATION_SECONDS` by default for
    every managed phone call, to the agent's explicit ``min_speech_ms`` down
    to :data:`MANAGED_PSTN_EXPLICIT_MIN_SPEECH_FLOOR_SECONDS`, or, without an
    explicit value, to the selected ``interruption_sensitivity`` preset in
    any language).  This is the word half, and it exists because the runtime's
    ``min_words`` cannot be it: that guard reads
    ``AudioRecognition._current_transcript``, a buffer fed only by STT events
    and cleared at every user-turn commit, and :class:`CallerTranscriptGate`
    withholds every STT event for the whole reply -- so ``min_words=1`` would
    find an empty string on every check and refuse EVERY interruption.  The
    stack sets ``min_words=0`` and keeps it there.

    So the count comes from the gate, which is holding those very events, and
    the enforcement happens here, on the reply's own speech handle.

    Why the handle flag, and not the pause
    --------------------------------------

    ``SpeechHandle.allow_interruptions`` is the only refusal the pinned
    runtime takes cleanly.  Both of its pause sites
    (``agent_activity.py`` :1935 in ``_interrupt_by_audio_activity``, :1999 in
    ``on_start_of_speech``) are gated on it, so a held reply is simply never
    asked to yield: no ``pause()``, no ``_paused_speech``, no agent-state
    change.  Refusing the pause at the audio sink instead would leave the
    runtime believing it had yielded -- it moves the session to ``listening``
    in the statement after ``pause()`` returns, and the evidence recorder
    closes the agent's speaking interval on that transition -- so the call's
    own evidence would say the agent had stopped while audio kept playing.
    ``agent/playout_hold_output.py`` documents that trade for the bounded
    playout hold and reaches the same conclusion.

    It costs one session option, which the caller must supply
    (``discard_audio_if_uninterruptible: False``, set for exactly these calls
    by :func:`session_options_with_managed_pstn_turn_taking`).  Left at the
    runtime's ``True`` default, an uninterruptible speech makes the runtime
    substitute SILENCE on the STT path (:1279-1285), the word that would
    release the floor is never transcribed, and the hold becomes permanent for
    the reply.

    The two are one decision, and the predicate they share is
    :func:`installed_min_interruption_words` -- which reads the option back out
    of the BUILT session rather than resolving the requirement a second time.
    Sharing :func:`effective_min_interruption_words` instead would look
    equivalent and is not: that one answers whether the stack COULD enforce the
    requirement, and a session shape the option could not be installed on
    answers yes to it.

    Deliberately one writer among several, and it never guesses
    ----------------------------------------------------------

    Two other things in this worker clear the same flag: the opening-turn
    protection keeps the greeting uninterruptible while it plays, until the
    caller has spoken past the pickup bound, and the bounded playout hold
    stands the runtime down when it
    releases withheld audio.  This gate therefore REFUSES a handle that is
    already uninterruptible -- somebody else owns it, and restoring it would
    hand the floor back on that owner's behalf -- and it only ever restores to
    ``True``, never to a remembered previous value that a third writer may
    have changed underneath.

    The two cannot interleave the other way either: while this gate holds a
    handle, no pause can happen, so the playout hold never starts a hold and
    never stands anything down. The hold may instead hand a handle TO this
    gate: a reply released because no caller word was behind its hold is
    armed here rather than stood down, so the caller can still take the floor
    with a word.

    Loop-affine by contract, like the transcript gate it reads from: armed and
    released from the session's own event handlers on the single media loop.
    """

    def __init__(self, *, min_words: int) -> None:
        self._min_words = max(0, int(min_words))
        self._handle: Any = None
        # The handle this gate last gave the floor back to. One is enough: the
        # only way a released handle returns is the runtime resuming THAT
        # speech, and a resume always concerns the most recent one.
        self._released_handle: Any = None
        self._replies_held = 0
        self._replies_released_by_words = 0
        self._replies_released_at_reply_end = 0
        self._words_at_release_max = 0
        self._rearms_refused = 0

    @property
    def enabled(self) -> bool:
        return self._min_words > 0

    @property
    def min_words(self) -> int:
        return self._min_words

    @property
    def holding(self) -> bool:
        return self._handle is not None

    @property
    def stats(self) -> dict[str, int]:
        """Bounded per-call counters for the interruption summary."""

        return {
            "word_gate_replies_held": self._replies_held,
            "word_gate_replies_released_by_words": self._replies_released_by_words,
            "word_gate_replies_released_at_reply_end": self._replies_released_at_reply_end,
            "word_gate_words_at_release_max": self._words_at_release_max,
            "word_gate_rearms_refused": self._rearms_refused,
        }

    def arm(self, handle: Any) -> bool:
        """Hold this reply's floor; ``False`` when it must not be held.

        Refused for a handle that is missing, already interrupted, already
        uninterruptible (another writer's), one this gate is already holding,
        or one it has ALREADY RELEASED.  A re-arm on a NEW handle releases the
        old one first: the previous reply is over by construction if a new one
        is speaking, and leaving its flag cleared would make a finished turn
        permanently unbarge-in-able if the runtime ever reused it.

        The released-handle refusal is not tidiness; without it the false
        interruption path drops a caller's words. A qualified barge-in pauses
        playout, the session reports ``listening``, the transcript gate hands
        its held finals to the SDK and this gate gives the floor back. If the
        caller then stays quiet, the runtime's own
        ``false_interruption_timeout`` (1.0 s here) fires
        ``_on_false_interruption``, which restores the agent state to
        ``speaking`` on the SAME handle -- an arming edge indistinguishable
        from a new reply's. Re-arming there makes the handle uninterruptible
        again, and the caller's turn is still on its way to committing: the
        endpointing delay is 0.2-0.6 s by default and a customer may set
        ``silence_hold_ms`` as high as 2000 ms, while Deepgram's own results
        arrive on a ~1000 ms cadence. A commit landing after the resume reaches
        ``agent_activity.py`` :2301-2307, which logs "skipping reply to user
        input" and RETURNS -- the caller said "stop it", the agent paused,
        resumed, kept talking, and their words were dropped.

        Identity is the right discriminator because it separates the three
        cases exactly: a new reply is always a new handle, a resume is always
        the same one, and a tool-call re-entry keeps the handle but never
        released it, so it is unaffected. The resumed segment then runs on the
        duration floor alone, which is the contract every managed call ran
        under before the word requirement existed.
        """

        if not self.enabled or handle is None:
            return False
        if handle is self._handle:
            return False
        if handle is self._released_handle:
            self._rearms_refused += 1
            return False
        if self._handle is not None:
            self.release(by_words=False)
        try:
            if bool(getattr(handle, "interrupted", False)):
                return False
            if not bool(getattr(handle, "allow_interruptions", True)):
                return False
            handle.allow_interruptions = False
        except Exception:
            # A runtime that refuses the setter leaves the reply interruptible
            # on the duration floor alone. That is the contract every managed
            # call ran under before the word requirement existed, which is the
            # correct direction to fail.
            return False
        self._handle = handle
        self._replies_held += 1
        return True

    def release(self, *, by_words: bool, words_heard: int = 0) -> bool:
        """Give the floor back; ``True`` when this gate was holding it.

        Idempotent, and safe from any of its three callers: the caller's word
        arriving, the reply ending, and teardown.
        """

        if self._handle is None:
            return False
        handle = self._handle
        if not restore_speech_interruptions(handle):
            # The runtime refused the flag, so the floor is still held. Keeping
            # the handle means the reply-end path retries; latching it as
            # released would do the opposite, refusing to re-arm a handle that
            # is uninterruptible and has no other writer left to fix it.
            return False
        self._handle = None
        self._released_handle = handle
        if by_words:
            self._replies_released_by_words += 1
            self._words_at_release_max = max(
                self._words_at_release_max, max(0, min(4_096, int(words_heard)))
            )
        else:
            self._replies_released_at_reply_end += 1
        return True


INTERRUPTION_EVENT_LIMIT = 64
# The only strings note_answer_gate will store; anything else becomes
# "unknown" rather than being admitted whole. Read from sip_answer_gate.py's
# own constants, not copied, so the two cannot drift apart.
_ANSWER_GATE_DISPOSITIONS = frozenset(
    {ANSWERED, ALREADY_ACTIVE, DISCONNECTED, TIMED_OUT, NOT_REPORTED, NOT_APPLICABLE}
)


class InterruptionTelemetry:
    """Bounded per-call evidence of what cancelled agent playout, and why.

    Deliberately NOT part of the
    ``skysay.call_telemetry.v1`` artifact: that schema's stage list is a
    closed allowlist in the control-plane normalizer
    (``services/call_telemetry.py``), and an artifact carrying an unlisted
    stage is rejected whole, so adding stages here would turn a voice-only
    worker fix into an API release and would break telemetry import for every
    call until both shipped.  These are structured worker events instead --
    the same instrument that produced the reported baseline (mixer restarts
    and discarded TTS per call), so the founder's A/B compares like with like.

    It retains no transcript text, only a character count, and emits at most
    ``INTERRUPTION_EVENT_LIMIT`` events per call; after that only the
    counters in :meth:`summary` keep moving.  Every method is best-effort and
    must never raise into a media callback.
    """

    def __init__(
        self,
        *,
        emit: Callable[..., None],
        gated: bool,
        min_duration_seconds: float | None = None,
        min_interruption_words_configured: int = 0,
        min_interruption_words_effective: int = 0,
        clock: Callable[[], float] = time.monotonic,
        max_events: int = INTERRUPTION_EVENT_LIMIT,
    ) -> None:
        self._emit = emit
        self._gated = bool(gated)
        # BOTH numbers, for the same reason the floor above is recorded: the
        # configured value is what the customer asked for and the effective
        # one is what this call could enforce, and they differ precisely on
        # the calls whose logs a reader would otherwise misread -- a Sinhala
        # agent with the setting on runs duration-only, and a summary carrying
        # only the customer's number would say the opposite.
        self._min_interruption_words_configured = max(
            0, int(min_interruption_words_configured)
        )
        self._min_interruption_words_effective = max(
            0, int(min_interruption_words_effective)
        )
        # The floor this session actually ran with.  Recorded because the
        # investigation of a mid-word cut could not answer "was that burst
        # above the threshold?" from the log alone: the events carried
        # ``caller_speech_ms`` but not the number it had to beat, and the
        # threshold is not the same for every call.  ``demo.py`` sets 0.15 or
        # 0.20 by turn-detector support, and only managed ENGLISH calls are
        # raised to 0.50 by the per-agent policy above, so a reader who
        # assumes one number reads the other call's evidence wrong.  Taking it
        # from the built session options means the log cannot disagree with
        # the session that produced it.
        self._min_duration_ms = _bounded_milliseconds(min_duration_seconds)
        self._clock = clock
        self._max_events = max(1, int(max_events))
        self._emitted = 0
        self._suppressed = 0
        self._agent_speaking_since: float | None = None
        self._caller_speaking_since: float | None = None
        self._transcript_characters = 0
        self._withheld_speech_events = 0
        # Word evidence for the reply being measured, read from the transcript
        # gate at the same instant as ``_withheld_speech_events``.
        self._caller_words_heard = 0
        self._words_at_qualification = 0
        # One playout measurement, consumed by the speech it belongs to. A
        # reply cancelled before its first audio frame never produces a
        # speaking->listening transition, so it must not inherit the previous
        # reply's duration and transcript state -- it reports no playout.
        self._playout: dict[str, int] | None = None
        # Which handle the open speaking window and the measured playout
        # belong to, when the caller can say. A measurement is consumed only
        # by its owner: an uninterruptible farewell is never hooked, so its
        # number used to sit here until the NEXT hooked handle finished --
        # a failed generation that never played -- and was stamped onto that.
        self._speaking_speech_id = ""
        self._playout_speech_id = ""
        # Granted by the boundary stop when a qualified pause was deferred to
        # the end of the current word; consumed by the reply it belongs to.
        self._stop_boundary_tail_ms = 0
        # How long the reply's audio was withheld before its first frame, and
        # how many caller words arrived meanwhile. Reported by the
        # playout-hold guard when a held segment ends, which the runtime does
        # before the handle's done callback, and consumed by that reply.
        self._hold_ms = 0
        self._hold_words = 0
        self._hold_speech_id = ""
        # Per-call totals contributed after media stops (gate, boundary stop,
        # greeting protection).  Bounded scalars only.
        self._totals: dict[str, Any] = {}
        self._counters = {
            "agent_speeches_interrupted": 0,
            "agent_speeches_completed": 0,
            "agent_speeches_failed": 0,
            "replies_cancelled_before_playout": 0,
            "false_interruptions_resumed": 0,
            "false_interruptions_abandoned": 0,
            "caller_transcript_events_withheld": 0,
            "caller_transcript_events_released": 0,
            # Not an interruption, but it belongs to the same
            # per-call summary for the reason that summary exists: it is the
            # one number that says whether the callee heard the opening they
            # were supposed to hear. A call where this is 1 played a short
            # fixed line instead of its configured opening -- whichever of
            # the three opening turns that was (the agent's first_message, a
            # workflow entry step, or a workflow whose opening was already
            # the whole conversation, which falls back to a goodbye). All
            # three are the opening turn; nothing later in a call increments
            # this.
            "greeting_fallback_used": 0,
        }

    @property
    def source(self) -> str:
        """Which interruption contract this call ran under."""

        return "vad_gated" if self._gated else "ungated"

    def on_user_state_changed(self, old_state: Any, new_state: Any) -> None:
        """Record the VAD-driven caller speech onset.

        ``user_state`` flips to ``speaking`` from ``on_start_of_speech``,
        which the pinned SDK drives from VAD -- so this timestamp is the VAD
        start the criterion asks for, without reaching into SDK internals.
        """

        try:
            if str(new_state) == "speaking":
                self._caller_speaking_since = float(self._clock())
            elif str(old_state) == "speaking":
                self._caller_speaking_since = None
        except Exception:
            pass

    def on_agent_state_changed(
        self, old_state: Any, new_state: Any, *, speech_id: str = ""
    ) -> None:
        """One agent-state edge; ``speech_id`` names the handle speaking at it.

        Passed at the ``speaking`` edge, where the session's ``current_speech``
        is the handle whose first frame just played; the measurement taken at
        the matching exit edge is then owned by that handle, whatever the
        session's current speech is by the time the exit edge fires.
        """

        try:
            if str(new_state) == "speaking":
                self._agent_speaking_since = float(self._clock())
                self._speaking_speech_id = str(speech_id or "")
                self._transcript_characters = 0
                return
            if str(old_state) != "speaking":
                return
            now = float(self._clock())
            self._playout_speech_id, self._speaking_speech_id = (
                self._speaking_speech_id,
                "",
            )
            self._playout = {
                "playout_ms": _bounded_milliseconds(
                    now - self._agent_speaking_since
                    if self._agent_speaking_since is not None
                    else None
                ),
                "caller_speech_ms": _bounded_milliseconds(
                    now - self._caller_speaking_since
                    if self._caller_speaking_since is not None
                    else None
                ),
                "transcript_characters": self._transcript_characters,
                "withheld_transcripts": self._withheld_speech_events,
                "caller_words_heard": self._caller_words_heard,
                "words_at_qualification": self._words_at_qualification,
                "stop_boundary_tail_ms": self._stop_boundary_tail_ms,
            }
            self._agent_speaking_since = None
            self._stop_boundary_tail_ms = 0
            # Consumed by the reply it belongs to. A later handle that never
            # reached playout must report what happened during ITS window,
            # not replay this one's.
            self._transcript_characters = 0
            self._withheld_speech_events = 0
            self._caller_words_heard = 0
            self._words_at_qualification = 0
        except Exception:
            pass

    def on_user_transcript(self, *, characters: int) -> None:
        try:
            self._transcript_characters = min(
                16_384, self._transcript_characters + max(0, int(characters))
            )
        except Exception:
            pass

    def note_withheld_transcripts(self, count: Any) -> None:
        """How many speech-bearing caller events the gate is holding right now.

        Read from the gate immediately BEFORE the playout measurement is
        taken.  Without it a gated barge-in reports ``transcript_present:
        false`` even when the caller's words were already buffered: the
        release happens after this handler runs, so ``user_input_transcribed``
        has not fired yet.  That field is exactly what separates a VAD-only
        trigger from one with speech content, so getting it wrong would make
        the A/B unreadable.
        """

        try:
            self._withheld_speech_events = max(0, int(count))
        except (TypeError, ValueError):
            self._withheld_speech_events = 0

    def note_caller_words(self, *, heard: Any, at_qualification: Any) -> None:
        """How many words the caller had produced against the reply being measured.

        Read from the transcript gate at the same instant as
        :meth:`note_withheld_transcripts` and for the same reason: the words
        are in the gate's buffer, not in the SDK, at the moment playout stops.
        ``at_qualification`` is the count when the word requirement released
        the floor, and is ``0`` on a call with no requirement -- which is the
        honest reading, since nothing qualified.
        """

        try:
            self._caller_words_heard = max(0, min(4_096, int(heard)))
        except (TypeError, ValueError):
            self._caller_words_heard = 0
        try:
            self._words_at_qualification = max(0, min(4_096, int(at_qualification)))
        except (TypeError, ValueError):
            self._words_at_qualification = 0

    def on_agent_speech_finished(
        self, *, interrupted: bool, failed: bool = False, speech_id: str = ""
    ) -> None:
        """One agent reply reached its end -- cancelled, failed, or fully played.

        A handle that ends on an LLM/TTS error is reported ``interrupted ==
        False`` by the SDK, so counting it as completed playout would inflate
        the denominator of the very ratio this telemetry exists to compare.

        ``speech_id`` names the finishing handle. A measured playout is
        consumed only by the handle that produced it; when both sides are
        known and disagree, the finisher never played -- it is reported with
        no playout, and the measurement stays for its owner (or the next
        speaking edge). Without ids the old first-come behaviour stands.
        """

        try:
            playout = self._playout
            owner = self._playout_speech_id
            finisher = str(speech_id or "")
            if playout is not None and finisher and owner and finisher != owner:
                playout = None
            else:
                self._playout = None
                self._playout_speech_id = ""
            hold_ms, hold_words = self._take_pre_playout_hold(finisher)
            if interrupted and playout is not None:
                key = "agent_speeches_interrupted"
            elif interrupted:
                # No playout ever started: a speculative or pre-playout handle
                # the runtime discarded. Counting it as a cancelled playout
                # would overstate the number this A/B exists to compare.
                key = "replies_cancelled_before_playout"
            elif failed:
                key = "agent_speeches_failed"
            else:
                key = "agent_speeches_completed"
            self._counters[key] += 1
            if playout is None:
                # A reply cancelled before its first audio frame: VAD fired
                # while the session was still ``thinking``, so there is no
                # speaking->listening transition to measure from. Report no
                # playout -- honestly zero -- but keep the caller-side
                # evidence, which is the half of criterion 6 that still
                # exists and the half that says whether the trigger carried
                # words. Reading zeros here would make the A/B wrong for
                # precisely the interruptions the gate now prevents.
                playout = {
                    "playout_ms": 0,
                    "caller_speech_ms": _bounded_milliseconds(
                        float(self._clock()) - self._caller_speaking_since
                        if self._caller_speaking_since is not None
                        else None
                    ),
                    "transcript_characters": self._transcript_characters,
                    "withheld_transcripts": self._withheld_speech_events,
                    "caller_words_heard": self._caller_words_heard,
                    "words_at_qualification": self._words_at_qualification,
                    "stop_boundary_tail_ms": self._stop_boundary_tail_ms,
                }
                started = False
                self._stop_boundary_tail_ms = 0
                # Consumed: a later handle must not count the same evidence.
                self._transcript_characters = 0
                self._withheld_speech_events = 0
                self._caller_words_heard = 0
                self._words_at_qualification = 0
            else:
                started = True
            measured = playout
            self._record(
                "interruption_observed"
                if interrupted
                else "agent_playout_failed"
                if failed
                else "agent_playout_completed",
                playout_started=started,
                transcript_present=bool(
                    measured["transcript_characters"]
                    or measured["withheld_transcripts"]
                ),
                interruption_min_duration_ms=self._min_duration_ms,
                paused_on_qualified_speech=self._qualified(
                    measured["caller_speech_ms"]
                ),
                held_ms=hold_ms,
                words_during_hold=hold_words,
                **measured,
            )
        except Exception:
            pass

    def note_pre_playout_hold(
        self, *, held_ms: Any, words_during_hold: Any, speech_id: str = ""
    ) -> None:
        """A playback segment ended after its opening audio was withheld.

        The one record a reply cancelled while held leaves: it never reached
        the wire, so no ``playout.first_worker`` span describes it, and without
        this the hold that preceded the cancellation is invisible. Kept per
        reply as the worst segment, and consumed by the reply that finishes
        next -- by ``speech_id`` when both sides know it.
        """

        try:
            speech_id = str(speech_id or "")
            if not speech_id and self._hold_speech_id:
                # A segment that could not name its reply merges into the one
                # that could, rather than erasing it before its owner reads it.
                speech_id = self._hold_speech_id
            if speech_id != self._hold_speech_id:
                self._hold_ms = 0
                self._hold_words = 0
                self._hold_speech_id = speech_id
            self._hold_ms = max(self._hold_ms, max(0, min(3_600_000, int(held_ms))))
            self._hold_words = max(
                self._hold_words, max(0, min(4_096, int(words_during_hold)))
            )
        except Exception:
            pass

    def _take_pre_playout_hold(self, finisher: str) -> tuple[int, int]:
        owner = self._hold_speech_id
        if finisher and owner and finisher != owner:
            return 0, 0
        taken = (self._hold_ms, self._hold_words)
        self._hold_ms = 0
        self._hold_words = 0
        self._hold_speech_id = ""
        return taken

    def on_false_interruption(self, *, resumed: bool) -> None:
        """The SDK decided a pause was not a real turn and (maybe) resumed."""

        try:
            key = (
                "false_interruptions_resumed"
                if resumed
                else "false_interruptions_abandoned"
            )
            self._counters[key] += 1
            # The pause being resolved is the measurable event, and the
            # resumed segment would overwrite its measurement. Consume it
            # here so the false pause is recorded with its duration and
            # transcript evidence rather than silently folded into an
            # ordinary completed playout.
            paused, self._playout = self._playout, None
            self._record(
                "interruption_resolved",
                resumed=bool(resumed),
                playout_ms=int((paused or {}).get("playout_ms", 0)),
                caller_speech_ms=int((paused or {}).get("caller_speech_ms", 0)),
                transcript_characters=int(
                    (paused or {}).get("transcript_characters", 0)
                ),
                withheld_transcripts=int((paused or {}).get("withheld_transcripts", 0)),
                caller_words_heard=int((paused or {}).get("caller_words_heard", 0)),
                words_at_qualification=int(
                    (paused or {}).get("words_at_qualification", 0)
                ),
                stop_boundary_tail_ms=int(
                    (paused or {}).get("stop_boundary_tail_ms", 0)
                ),
                transcript_present=bool(
                    (paused or {}).get("transcript_characters")
                    or (paused or {}).get("withheld_transcripts")
                ),
                interruption_min_duration_ms=self._min_duration_ms,
                paused_on_qualified_speech=self._qualified(
                    int((paused or {}).get("caller_speech_ms", 0))
                ),
            )
        except Exception:
            pass

    def note_greeting_fallback(self) -> None:
        """The configured opening line failed and the fallback was spoken instead.

        Counter only, deliberately: the event carrying the language and the
        provider failure is emitted at the call site, and doubling it into the
        bounded per-call event budget would push out an interruption
        measurement to say the same thing twice.
        """

        try:
            self._counters["greeting_fallback_used"] += 1
        except Exception:
            pass

    def on_transcripts_gated(self, *, withheld: int, released: int) -> None:
        try:
            self._counters["caller_transcript_events_withheld"] = max(0, int(withheld))
            self._counters["caller_transcript_events_released"] = max(0, int(released))
        except Exception:
            pass

    def note_stop_boundary_tail(self, milliseconds: Any) -> None:
        """A qualified pause was deferred by this many ms to finish the word.

        Recorded when the deferral is GRANTED rather than when it expires:
        the runtime moves the session to ``listening`` in the statement after
        ``pause()`` returns, so the measurement below is already being taken
        while the tail is still playing.  The tail actually spent is counted
        by the boundary policy and reported in the per-call totals.
        """

        try:
            self._stop_boundary_tail_ms = max(0, min(3_600_000, int(milliseconds)))
        except (TypeError, ValueError):
            self._stop_boundary_tail_ms = 0

    def note_call_totals(self, **totals: Any) -> None:
        """Merge bounded per-call scalars into the summary.

        Contributed after media stops by the caller-transcript gate, the
        boundary stop and the greeting protection.  Non-scalars are dropped
        rather than serialized: this event is read by a log pipeline, not by
        a schema.  Never a string -- see :meth:`note_answer_gate` for the one
        piece of this summary that is a string by nature, admitted only
        through its own closed vocabulary rather than by widening this
        general merge.
        """

        try:
            for key, value in totals.items():
                if isinstance(value, bool) or isinstance(value, int):
                    self._totals[str(key)] = value
        except Exception:
            pass

    def note_answer_gate(
        self,
        *,
        disposition: str,
        wait_ms: int | None = None,
        input_released_ms: int | None = None,
        opening_generate_ms: int | None = None,
    ) -> None:
        """Persist the answer gate's own disposition and timing.

        The one string this summary carries, and it is not free text: only a
        value from ``sip_answer_gate``'s own closed disposition set is
        stored, everything else becomes ``"unknown"`` rather than silently
        admitting whatever a caller passes -- this method exists precisely
        because :meth:`note_call_totals` must never carry a string at all.

        Three clocks, all join-relative milliseconds, and deliberately not
        the same instant: ``wait_ms`` is when the SIP leg's own answer
        evidence arrived; ``input_released_ms`` is when this worker actually
        re-enabled the engine's input -- these two differ by however long
        the gate's own resolution and the may_speak check take;
        ``opening_generate_ms`` is when the opening was asked for. A ``None``
        (no gate ran; input never released; no opening was issued) is
        omitted rather than written as ``null``, the same convention
        :meth:`note_call_totals` already uses for anything that fails its
        own type check.
        """

        def _bounded_int(value: object) -> int | None:
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        try:
            self._totals["answer_gate_disposition"] = (
                str(disposition)
                if disposition in _ANSWER_GATE_DISPOSITIONS
                else "unknown"
            )
            for key, value in (
                ("answer_gate_wait_ms", wait_ms),
                ("input_released_ms", input_released_ms),
                ("opening_generate_ms", opening_generate_ms),
            ):
                bounded = _bounded_int(value)
                if bounded is not None:
                    self._totals[key] = bounded
        except Exception:
            pass

    def note_duplex_opening(self, *, outcome: str, delivered: bool) -> None:
        """How a voice-to-voice opening ended, and whether it reached the caller.

        A second closed-vocabulary string, for the same reason as
        :meth:`note_answer_gate`: an unknown value is stored as ``"unknown"``.

        ``delivered`` REPLACES the greeting protection's audio credit on these
        calls. That credit counts the first burst the model produced after the
        opening was asked for, which on a duplex engine can be a reply to
        whatever the far end was playing -- a call whose opening was never
        spoken read ``opening_turns_reached_audio: 1``. Written last, over the
        protection's totals, so the summary carries the hold's answer.
        """

        try:
            self._totals["duplex_opening_hold_outcome"] = (
                str(outcome) if outcome in DUPLEX_OPENING_OUTCOMES else "unknown"
            )
            if outcome not in DUPLEX_OPENING_JUDGED:
                # The hold never withheld anything (an engine it cannot wrap,
                # such as Gemini Live), so it has no view on what was spoken
                # and the protection's own credit stands.
                return
            self._totals["opening_turns_reached_audio"] = 1 if delivered else 0
            self._totals["greeting_protected"] = bool(
                delivered and self._totals.get("opening_turns_protected")
            )
        except Exception:
            pass

    def _qualified(self, caller_speech_ms: Any) -> bool:
        """Whether the caller had held the floor for the configured floor.

        Measured from ``user_state -> speaking``, which the pinned runtime
        raises from the VAD's own start-of-speech -- roughly one Silero
        ``min_speech_duration`` (50 ms by default) AFTER the caller actually
        started.  The error is therefore ONE-SIDED and that is what makes the
        field safe to read: this clock always UNDER-counts the caller, so a
        ``True`` here means the burst really did clear the floor, while a
        ``False`` near the threshold may be a burst that did.  Never the
        other way round.

        It is evidence, not the decision.  The decision belongs to the
        runtime's ``on_vad_inference_done``, which compares the VAD's own
        accumulated ``speech_duration`` against the same floor; this exists so
        a reader of the log can tell a floor-qualified pause from a suspected
        one without a packet capture.
        """

        if self._min_duration_ms <= 0:
            return False
        try:
            return int(caller_speech_ms) >= self._min_duration_ms
        except (TypeError, ValueError):
            return False

    def summary(self) -> dict[str, Any]:
        """Per-call totals.

        ``interruption_min_duration_ms`` is ``0`` when the stack supplied no
        floor and the runtime's own default applies -- read it as "not
        configured here", never as "a floor of zero".  No configuration in this
        repository sets a zero floor, so the two cannot be confused in practice.
        """

        return {
            "interruption_source": self.source,
            "interruption_events_suppressed": self._suppressed,
            "interruption_min_duration_ms": self._min_duration_ms,
            "min_interruption_words_configured": self._min_interruption_words_configured,
            "min_interruption_words_effective": self._min_interruption_words_effective,
            **dict(self._counters),
            **dict(self._totals),
        }

    def _record(self, event: str, **details: Any) -> None:
        if self._emitted >= self._max_events:
            self._suppressed += 1
            return
        self._emitted += 1
        self._emit(event, interruption_source=self.source, **details)


def _bounded_milliseconds(seconds: float | None) -> int:
    if (
        seconds is None
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
    ):
        return 0
    return max(0, min(3_600_000, int(round(float(seconds) * 1_000))))


def requires_soniox_usage_correlation(
    profile: DemoProfile, config: Mapping[str, Any]
) -> bool:
    """Whether this exact selected stack needs Soniox usage-log references.

    ``usage_required`` covers every component-metered call, including Fish's
    independent UTF-8-byte TTS meter. Only a selected Soniox STT or TTS
    adapter owns provider usage logs, however, so a Fish-only English stack
    must never be classified as a Soniox correlation failure before media.
    """
    billing = config.get("component_billing")
    return bool(
        isinstance(billing, Mapping)
        and billing.get("usage_required") is True
        and (profile.stt_registry == "soniox" or profile.tts_registry == "soniox")
    )


def build_call_tts_component(
    profile: DemoProfile,
    *,
    call_id: str,
    call_config: Mapping[str, Any],
    boundary: int,
    **kwargs: Any,
) -> TTSComponent:
    """Build a replacement using the call's accounting and the exact target.

    Frozen language configs deliberately omit call-level component billing.
    Admission still receives that exact target via ``target_config``; native
    accounting instead follows the admitted call, including paid browser tests.
    """
    correlation_required = requires_soniox_usage_correlation(profile, call_config)
    return build_tts_component(
        profile,
        soniox_usage_reference=(f"{call_id}:segment:{boundary}" if correlation_required else ""),
        require_soniox_usage_reference=correlation_required,
        **kwargs,
    )


# The isolated worker reads the generated copy of the canonical declaration.
# These indexed views do not maintain another engine list or infer a property
# from whichever components happen to be present on a particular call.
ENGINE_REQUIRED_RUNTIME_FIELDS = {key: row["required_runtime_fields"] for key, row in ENGINE_DECLARATIONS.items()}
ENGINE_HAS_LANGUAGE_AXIS = {key: row["has_language_axis"] for key, row in ENGINE_DECLARATIONS.items()}
ENGINE_REPORTS_CALLER_TURNS = {key: row["reports_caller_turns"] for key, row in ENGINE_DECLARATIONS.items()}
ENGINE_SPEAKS_TEXT = {key: row["speaks_text"] for key, row in ENGINE_DECLARATIONS.items()}


def engine_reports_caller_turns(config: dict[str, Any] | None) -> bool:
    """Does this call's engine detect the caller's turns on its own side.

    Read from the declaration above rather than from an engine-id comparison.
    An engine this image does not declare gets no extra witness: the silence
    rule then rests on the VAD and the captions exactly as before, which is
    the conservative direction for a rule that ends calls.
    """

    frontend = str((config or {}).get("voice_frontend") or "cascade")
    return bool(ENGINE_REPORTS_CALLER_TURNS.get(frontend, False))


def engine_has_language_axis(config: dict[str, Any] | None) -> bool:
    """Does the engine this call runs on have a separate transcription step.

    Read from the declaration above rather than from an engine-id comparison,
    which is the shape this repository has spent a ticket deleting: the worker
    had ``voice_frontend in {"gpt_live", "gemini_live"}`` spelled at each door,
    and a new engine was then covered by whichever doors somebody remembered.

    An engine this image does not declare is treated as cascade -- the same
    fallback ``stack_required_runtime_fields`` makes, and the same direction:
    an undeclared engine keeps the caller-facing repair line rather than
    having it silently withdrawn.

    THE COST OF THAT DIRECTION, stated accurately. If the undeclared engine is
    in fact voice-to-voice, it emits no transcriber timeout, so the line never
    speaks AND the standing answer is never recorded -- the silent absence
    this change exists to remove, for exactly one engine: one nobody has
    declared in this image. The alternative is worse and is the reason the
    fallback points this way: treating every unknown engine as
    transcriber-less would withdraw the line from a cascade engine added to
    the catalogue and not yet mirrored here, which is a caller hearing nothing
    on a call that works. The mirror's parity test is what keeps the set of
    undeclared engines empty.
    """

    data = config if isinstance(config, dict) else {}
    voice = data.get("voice") if isinstance(data.get("voice"), dict) else {}
    frontend = str(
        data.get("voice_frontend") or voice.get("voice_frontend") or ""
    ).strip().lower()
    return ENGINE_HAS_LANGUAGE_AXIS.get(frontend or "cascade", ENGINE_HAS_LANGUAGE_AXIS["cascade"])


#: The published refusal label for each required field. Written out rather than
#: built as ``f"{field}_missing"`` because these strings are a customer-facing
#: contract -- ``frontend/messages/*.json`` keys copy on them in six locales --
#: and a field rename must not silently republish a different label.
RUNTIME_FIELD_MISSING_REASONS: dict[str, str] = {
    "voice_id": "voice_id_missing",
    "tts_model": "tts_model_missing",
    "llm_model": "llm_model_missing",
    "system_prompt": "system_prompt_missing",
}


def _configured_languages(config: dict[str, Any]) -> tuple[str, ...]:
    """The languages the agent declares, in order, each once; ``()`` for a config that omits or misspells them.

    Advisory beyond the call's own language (it decides nothing else), so a config that omits it, spells it oddly, or
    sends a type this worker did not expect builds the same one-language call it built before the field existed.
    """

    raw_languages = config.get("languages")
    seen: list[str] = []
    if isinstance(raw_languages, (list, tuple)):
        for item in raw_languages:
            if not isinstance(item, str):
                continue
            candidate = item.strip().lower()
            if candidate and candidate not in seen:
                seen.append(candidate)
    return tuple(seen)


def profile_from_realtime_config(
    config: dict[str, Any], *, require_component_billing: bool = True
) -> DemoProfile:
    refuse_freeform_workflow(config)
    language = str(config.get("language") or "").strip().lower()
    voice_early = config.get("voice") if isinstance(config.get("voice"), dict) else {}
    frontend = str(
        config.get("voice_frontend") or voice_early.get("voice_frontend") or ""
    ).strip().lower()
    conflicts = (engine_deferred_tool_conflicts(frontend or "cascade", config.get("tools", []))
                 if (frontend or "cascade") in ENGINE_DECLARATIONS else ())
    if conflicts:
        raise ProductionConfigurationError(
            f"{ENGINE_DECLARATIONS[frontend]['display']} cannot use these tools yet: {', '.join(conflicts)}",
            reason="engine_deferred_tool_conflict",
        )
    if frontend in {"gpt_live", "gemini_live", "eleven_realtime"}:
        if not language:
            raise ProductionConfigurationError("realtime-config language is required")
        model_sel = config.get("model") if isinstance(config.get("model"), dict) else {}
        first_message = str(config.get("first_message") or "").strip()
        stored_mode = str(config.get("opening_mode") or "").strip().casefold()
        if stored_mode not in {"literal", "generated"}:
            stored_mode = "literal" if first_message else "generated"
        if frontend == "eleven_realtime":
            if stored_mode != "literal" or not first_message:
                raise ProductionConfigurationError("Eleven Realtime requires a literal first message")
            return DemoProfile(
                language=language, stt_language=language,
                tts_model=str(voice_early.get("model") or ""),
                voice_id=str(voice_early.get("voice_id") or ""),
                llm_model=str(model_sel.get("model") or ""),
                instructions=str(config.get("system_prompt") or "").strip(),
                first_message=first_message, stt_registry="elevenlabs", tts_registry="elevenlabs",
                stt_model=ELEVEN_REALTIME_MODEL, opening_mode=stored_mode,
                languages=_configured_languages(config),
            )
        if frontend == "gemini_live":
            return DemoProfile(
                language=language,
                stt_language=language,
                tts_model=str(voice_early.get("model") or "gemini-3.8-live"),
                voice_id=str(voice_early.get("voice_id") or "Kore"),
                llm_model=str(model_sel.get("model") or "gemini-3.8-live"),
                instructions=str(config.get("system_prompt") or "").strip(),
                first_message=first_message,
                stt_registry="google",
                tts_registry="google",
                stt_model="gemini-live-captions",
                opening_mode=stored_mode,
                # The detector listens for the menu's language, and a live call has no session STT to say it.
                languages=_configured_languages(config),
            )
        return DemoProfile(
            language=language,
            stt_language=language,
            tts_model=str(voice_early.get("model") or "gpt-live-1"),
            voice_id=str(voice_early.get("voice_id") or "marin"),
            llm_model=str(model_sel.get("model") or "gpt-5.6-luna"),
            instructions=str(config.get("system_prompt") or "").strip(),
            first_message=first_message,
            stt_registry="openai",
            tts_registry="openai",
            stt_model="gpt-live-captions",
            opening_mode=stored_mode,
            languages=_configured_languages(config),
        )
    registry = load_registry()
    if language not in registry.runtime_languages:
        raise ProductionConfigurationError(
            "realtime-config language is absent from the worker capability registry"
        )
    transcriber = (
        config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    )
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}

    stt_registry = str(transcriber.get("registry_name") or "").lower()
    stt_model = str(transcriber.get("model") or "").strip()
    stt_provider = str(transcriber.get("provider") or stt_registry).lower()
    stt_capability = registry.capability("stt", stt_provider, stt_model)
    if stt_capability is not None:
        if (
            not stt_capability["runnable"]
            or language not in stt_capability["languages"]
        ):
            raise ProductionConfigurationError(
                "realtime-config selects an unavailable STT capability"
            )
        if stt_registry != stt_capability["registry_name"]:
            raise ProductionConfigurationError(
                "realtime-config STT registry does not match the capability artifact"
            )
    else:
        expected_stt_models = _LEGACY_STT_REGISTRY_MODELS.get(stt_registry)
        # Preserve the adapter/model construction guard; language support must
        # be declared for this exact stack, as in control-plane admission.
        if (
            not registry.declares_language("stt", stt_registry, stt_model, language)
            or registry.owns_component_provider("stt", stt_provider)
            or expected_stt_models is None
            or stt_model not in expected_stt_models
        ):
            raise ProductionConfigurationError(
                "realtime-config must resolve a supported STT provider/model"
            )
    if str(model.get("registry_name") or "").lower() not in {"gemini", "google"} or (str(model.get("model") or "").strip() and not registry.declares_language(
        "llm", "gemini", str(model.get("model") or "").strip(), language
    )):
        raise ProductionConfigurationError("realtime-config must resolve Gemini LLM")
    tts_registry = str(voice.get("registry_name") or "").lower()
    voice_id = str(voice.get("voice_id") or "").strip()
    tts_model = str(voice.get("model") or "").strip()
    tts_provider = str(voice.get("provider") or tts_registry).lower()
    llm_model = str(model.get("model") or "").strip()
    instructions = str(config.get("system_prompt") or "").strip()
    first_message = str(config.get("first_message") or "").strip()
    opening_mode = str(config.get("opening_mode") or "literal").strip().casefold()
    if opening_mode not in {"literal", "generated"}:
        raise ProductionConfigurationError(
            "realtime-config has an unknown opening mode"
        )
    if opening_mode == "literal" and not first_message:
        raise ProductionConfigurationError(
            "literal realtime-config opening requires a first message"
        )
    if opening_mode == "generated" and first_message:
        raise ProductionConfigurationError(
            "generated realtime-config opening must not carry a first message"
        )
    # One raise per missing field, each carrying the label that names it. The
    # single "omitted runnable prompt/provider values" refusal this replaces
    # published ``component_configuration_rejected`` -- the stage's reason,
    # shared with every other refusal in this builder -- so a customer whose
    # call failed for a missing voice and a customer whose call failed for a
    # language-switch mismatch read the same sentence -- both happened in
    # production on 2026-09-18, four hours apart.
    #
    # Reported in the order this table declares, and one at a time: the control
    # plane's predicate lists every missing field at once because a member
    # fixing a setup screen should see all of them, while a call already in a
    # room ends on the first thing it cannot do.
    #
    # The label names the field, never its value: ``system_prompt_missing``,
    # never any part of the prompt.
    engine_required = ENGINE_REQUIRED_RUNTIME_FIELDS.get(
        frontend or "cascade", ENGINE_REQUIRED_RUNTIME_FIELDS["cascade"]
    )
    resolved_runtime_fields = {
        "voice_id": voice_id,
        "tts_model": tts_model,
        "llm_model": llm_model,
        "system_prompt": instructions,
    }
    for field in engine_required:
        if not resolved_runtime_fields.get(field, ""):
            raise ProductionConfigurationError(
                f"realtime-config omitted the runnable {field} value",
                reason=RUNTIME_FIELD_MISSING_REASONS[field],
            )
    tts_capability = registry.capability("tts", tts_provider, tts_model)
    # ElevenLabs is the only unversioned adapter that historically served
    # Estonian. New Estonian stacks must resolve through an exact runnable
    # capability row, or a pre-registry model the artifact declares for
    # Estonian itself (Gemini 3.8 Flash-Lite); otherwise the broad
    # legacy-language allowance below would accidentally admit undeclared
    # adapters such as Gemini 2.5/3.1 TTS. The control plane's copy of this
    # rule is ``call_admission_profile.runnable_profile_blockers``.
    if (
        language == "et"
        and tts_capability is None
        and tts_registry != "elevenlabs"
        and not registry.declares_language("tts", tts_registry, tts_model, "et")
    ):
        raise ProductionConfigurationError(
            "Estonian requires a declared runnable TTS capability or ElevenLabs multilingual model"
        )
    # ElevenLabs' English-only Flash model remains the historical Estonian
    # model exception. Capability-backed providers are validated by their
    # exact provider/model/language row below.
    if language == "et" and tts_registry == "elevenlabs" and "flash" in tts_model.lower():
        raise ProductionConfigurationError(
            "Estonian requires an ElevenLabs multilingual model"
        )
    if tts_capability is not None:
        if (
            not tts_capability["runnable"]
            or language not in tts_capability["languages"]
        ):
            raise ProductionConfigurationError(
                "realtime-config selects an unavailable TTS capability"
            )
        if tts_registry != tts_capability["registry_name"]:
            raise ProductionConfigurationError(
                "realtime-config TTS registry does not match the capability artifact"
            )
        # Cache-backed providers have dynamic voice ids.  Their exact scoped
        # admission is enforced by the control plane before this worker config
        # exists; the artifact still validates provider/model/language here.
        # Static capability rows retain their historical worker-local id guard.
        if tts_capability.get("voice_inventory") != "provider_cache":
            try:
                registry.validate_runnable_voice(
                    str(tts_capability["provider"]),
                    tts_model,
                    voice_id,
                    language=language,
                )
            except VoiceCapabilityError as exc:
                raise ProductionConfigurationError(
                    "realtime-config voice id is absent from the approved inventory"
                ) from exc
    elif (
        not registry.declares_language("tts", tts_registry, tts_model, language)
        or registry.owns_component_provider("tts", tts_provider)
        or tts_registry not in _LEGACY_TTS_REGISTRIES
    ):
        raise ProductionConfigurationError(
            "realtime-config must resolve a supported TTS provider/model"
        )
    if require_component_billing:
        try:
            registry.validate_registry_component_billing(
                config,
                stt_capability=stt_capability,
                tts_capability=tts_capability,
            )
        except VoiceCapabilityError as exc:
            raise ProductionConfigurationError(
                "realtime-config registry-backed components require a complete billing contract"
            ) from exc

    # The STT tag, from the artifact's declaration rather than from a map kept
    # here. This used to be three hand-written remaps -- en->en-US, si->si-LK,
    # and (Google only) ta->ta-LK -- and the homepage worker kept a fourth,
    # shorter copy of the same idea. Neither grew when ten more European
    # languages were declared on Google ``chirp_2``, so a German or Russian
    # agent reached Google as bare "de"/"ru": codes Google publishes for no
    # model, on either API version. The session still connects and still bills.
    #
    # Two properties are preserved exactly, because both were reviewed
    # decisions rather than accidents:
    #
    # * An explicit ``transcriber.language`` still wins. It is only replaced
    #   when it equals the bare declared language, i.e. when it carries no
    #   information this table does not already have. Nothing on the ordinary
    #   product surface can produce a different one -- the agent-editor save
    #   path (agent_voice_stack.py) validates the full string against the
    #   catalog row's ``languages`` list, which holds bare codes, and a
    #   per-call language override (calls_routing/calls.py) stamps the bare
    #   requested language on unconditionally -- so this is a contract of this
    #   parser, not a customer-reachable locale picker.
    # * The remap stays scoped to the adapter that asked for it. ``ta-LK`` is
    #   a DEFAULT matched to the launch customer (Sri Lanka); Google documents
    #   ta-IN, ta-LK, ta-MY and ta-SG on Google V1, which this release removed
    #   from the live declaration. Tamil also runs on Soniox, which identifies
    #   Tamil as the bare "ta" and enforces it with a strict language hint, and
    #   Soniox is registry-backed and carries no locale table -- so a Soniox
    #   Tamil agent keeps the bare code, exactly as before. Deepgram is the
    #   same shape as Soniox here: it identifies Tamil as the bare
    #   "ta" too, and its declared_stack_languages row states that explicitly
    #   (``"ta": "ta"``) rather than leaving the locale table silent on it --
    #   this is a founder decision, not a Google API constraint: Deepgram has
    #   no notion of ta-LK/ta-IN at all, so there is nothing to flip if a
    #   future customer needs India Tamil.
    #
    # And the lookup FAILS CLOSED for an adapter that refuses bare codes.
    # `stt_locale` can only police declarations that exist under the name it is
    # asked for, while the legacy allowlist above admits ``google``/``chirp_2``
    # for en/et/si. Adversarial review built both holes -- rename a declared row's model,
    # or delete the row -- and each loaded cleanly and then sent Google a bare
    # "si". Refusing the config is the honest outcome: a caller hears an error
    # instead of paying for a call that transcribes nothing.
    # Every language this agent is configured for, read leniently and never
    # raised on: it widens what a capable transcriber may listen for and
    # decides nothing else, so a config that omits it, spells it oddly, or
    # sends a type this worker did not expect must build the same one-language
    # call it built before the field existed. A refusal here would also be a
    # new divergence from the control plane's copy of these rules.
    configured_languages = _configured_languages(config)
    stt_language = str(transcriber.get("language") or language).strip()
    if stt_language == language:
        locale = registry.stt_locale(stt_registry, stt_model, language)
        if locale:
            stt_language = locale
        elif registry.requires_stt_locale(stt_registry):
            raise ProductionConfigurationError(
                "realtime-config STT provider declares no locale for this language"
            )
    # ``voice_polish.delivery`` rides the realtime config only when an agent
    # chose a profile.  The name is checked against this image's packaged
    # table here, at the profile seam, so an API ahead of this worker refuses
    # before media instead of sending a provider a control it never reviewed.
    voice_polish = config.get("voice_polish")
    raw_delivery = voice_polish.get("delivery") if isinstance(voice_polish, dict) else None
    try:
        delivery = normalize_delivery_profile(raw_delivery) or ""
    except ValueError as exc:
        raise ProductionConfigurationError(
            "realtime-config voice_polish.delivery names an unknown profile"
        ) from exc
    return DemoProfile(
        language=language,
        stt_language=stt_language,
        tts_model=tts_model,
        voice_id=voice_id,
        llm_model=llm_model,
        # The model is told which language to speak, here, because
        # nothing upstream does it. The agent prompt's own "Speak only in the
        # agent's configured language" names a value the model cannot resolve,
        # so every generated turn on the first Sinhala campaign came back in
        # English while the literal ``first_message`` was Sinhala.
        #
        # Derived from ``language`` -- the value validated at the top of this
        # function -- so a per-call override is followed automatically:
        # ``calls_routing/calls.py`` rewrites ``config["language"]`` itself
        # (not only ``transcriber.language``) when a request carries
        # ``runtime_config.language``, so this reads whichever language the
        # call is actually being conducted in.
        #
        # Applied at the profile seam rather than in the control plane on
        # purpose: this is the worker's own last word on a call it is about to
        # run, it fixes every already-stored agent with no re-save, and it
        # ships on a voice release rather than needing the control plane and
        # the worker to move together.
        instructions=with_language_directive(instructions, language),
        first_message=first_message,
        stt_registry=stt_registry,
        tts_registry=tts_registry,
        stt_model=stt_model,
        # Preserve the durable identity and origin's absolute grace deadline.
        # Accepted newer-API configs need not equal this worker's artifact;
        # initial construction must make the same compatibility decision.
        # Legacy unstamped profiles must not acquire a synthetic revision.
        opening_mode=opening_mode,
        delivery=delivery,
        languages=configured_languages,
    )


def component_model_labels(profile: DemoProfile) -> dict[str, str]:
    """Name the model each component actually ran, for telemetry attribution.

    Read off the resolved profile rather than the raw config, so a span records
    what this worker built and not what it was asked for.

    Telemetry needs this at write time because a span carries no other durable
    link to a model. The alternative -- joining a recorded span back to the
    agent's stored configuration -- resolves against whatever that agent is
    configured with *now*, so reconfiguring an agent silently re-labels every
    turn it has ever recorded, with nothing in the data marking the change.
    """

    return {
        "stt": profile.stt_model
        or _LEGACY_STT_DEFAULT_MODELS.get(profile.stt_registry, ""),
        "llm": profile.llm_model,
        "tts": profile.tts_model,
    }


def _default_tool_builder(
    function: Callable[..., Any], raw_schema: dict[str, Any]
) -> Any:
    from livekit.agents import function_tool

    return function_tool(raw_schema=raw_schema)(function)


def _successful_tool_result(result: Any) -> bool:
    """Whether one completed wrapper invocation counts as billable usage.

    MCP/knowledge endpoints use a successful HTTP response as their success
    contract and may omit ``ok``; structured tool endpoints carry ``ok``.
    Rejected/failed structured calls are never counted, and this helper avoids
    retaining any result body beyond this immediate boolean decision.
    """
    if not isinstance(result, dict) or result.get("ok") is False:
        return False
    payload = result.get("result")
    action = payload.get("action") if isinstance(payload, dict) else None
    if isinstance(action, dict):
        return action.get("status") == "succeeded"
    return str(result.get("outcome") or "").strip().lower() not in {
        "empty",
        "timeout",
        "cancelled",
        "error",
        "fallback",
    }


def executed_tool_names(event: Any) -> list[str]:
    """Read the pinned LiveKit ``FunctionToolsExecutedEvent`` shape safely."""

    return [
        name
        for call in getattr(event, "function_calls", [])
        if (name := str(getattr(call, "name", "")).strip())
    ]


def is_greeting_fragment(text: str, first_message: str) -> bool:
    """Exclude a barge-in-truncated first message from semantic turn pairing."""

    candidate = " ".join(text.casefold().split())
    greeting = " ".join(first_message.casefold().split())
    return bool(candidate and greeting) and (
        greeting.startswith(candidate) or candidate.startswith(greeting)
    )


async def cleanup_livekit_room(
    call: Any,
    delete_room: Callable[[], Awaitable[None]],
    *,
    session_cleanup: bool = True,
) -> None:
    """Delete this call's room. Session cleanup is separate from hangup.

    ``call.cleanup()`` marks the control-plane session ``_cleaned``, after
    which usage, ``sip.ended`` and artifact import all raise. Hang up the
    carrier first (``session_cleanup=False``); settle; then run this with
    the default so the session is marked cleaned.
    """

    if session_cleanup:
        call.cleanup(remove_participant=lambda: None, delete_room=lambda: None)
    await delete_room()


_PRE_MEDIA_REASON_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def require_pre_media_participant(room: Any, participant: Any) -> None:
    """Recheck the exact leg after admission I/O, not a stale participant.

    A reconnect with the same identity does not revive the departed leg's
    authorization. Its SID must still match before providers or media start.
    """
    current = room.remote_participants.get(participant.identity)
    if current is None or current.sid != participant.sid:
        raise ProductionConfigurationError(
            "Caller left during media preparation", reason="caller_left_before_media")


async def require_inbound_sip_answer(
    room: Any, participant: Any, *, wait_for_active: Callable[[], Awaitable[None]],
    timeout_seconds: float,
) -> None:
    """Do not turn room membership into a fictitious inbound SIP answer.

    The actual session output track is already published when this runs, so
    SIP can answer without starting the model. Missing status is not an answer.
    """
    require_pre_media_participant(room, participant)
    failure: Exception | None = None
    try:
        await asyncio.wait_for(wait_for_active(), timeout=timeout_seconds)
    except (TimeoutError, RuntimeError) as exc:
        failure = exc
    require_pre_media_participant(room, participant)
    if room.remote_participants[participant.identity].attributes.get("sip.callStatus") != "active":
        raise ProductionConfigurationError(
            "Inbound SIP answer was not confirmed", reason="inbound_answer_unconfirmed") from failure


def safe_pre_media_reason(reason: str) -> str:
    """Return a bounded lifecycle reason that cannot contain provider data."""

    candidate = str(reason or "").strip().lower()
    if (
        candidate
        and len(candidate) <= 80
        and all(character in _PRE_MEDIA_REASON_CHARS for character in candidate)
    ):
        return candidate
    return "pre_media_rejected"


def pre_media_lifecycle_reason(
    stage: str, *, participant_mismatch: bool = False
) -> str:
    """Map one worker-owned admission stage to a closed safe reason code."""
    if participant_mismatch:
        return "participant_mismatch_rejected"
    return {
        "session_connect": "session_connect_rejected",
        "dispatch_validation": "dispatch_validation_rejected",
        "participant_wait": "participant_wait_rejected",
        "adapter_configuration": "adapter_configuration_rejected",
        "outbound_resume": "outbound_resume_rejected",
        "inbound_prepare": "inbound_prepare_rejected",
        "component_configuration": "component_configuration_rejected",
        "media_admission": "media_admission_rejected",
        # Admitted, but the control plane was never told media started: the
        # worker itself failed between the two (an answered customer heard
        # silence, 2026-09-17). The one published pre-media reason of this
        # worker's own; the rest above are configuration refusals.
        "worker_entry": "worker_entry_failed",
    }.get(str(stage or ""), "pre_media_rejected")


BROWSER_TEST_ERROR_KEY = "session_error"

# The whole vocabulary that may reach a customer's browser. LiveKit names the
# failing component on its error models (``tts_error``, ``llm_error``, ...);
# anything this worker cannot place is reported as the session having stopped
# rather than passed through. A closed set is the point: exception classes and
# provider prose are written for a log, can carry request URLs and account
# detail, and are not this room's to publish.
BROWSER_TEST_ERROR_COMPONENTS = frozenset({"stt", "llm", "tts", "realtime", "session", "startup"})

# ``livekit.agents.voice.events.CloseReason.ERROR``. The other reasons are
# ordinary endings: the worker closed the session itself at teardown, or the
# job was shut down.
SESSION_CLOSE_REASON_ERROR = "error"


def session_closed_on_error(reason: Any) -> bool:
    """Whether a session ``close`` event is the SDK giving up.

    Keyed on the CLOSE, not on an error event: LiveKit absorbs several
    unrecoverable component errors before it gives up, and resets that count
    on a turn that works, while a realtime model error it cannot recover
    closes the session at once. Either way the close is the one signal that
    the engine is gone, and the only one worth ending a call on.

    ``CloseReason`` is a ``str`` Enum, and ``str()`` on one is its NAME, so the
    value is read through ``.value`` and the enum member itself would never
    match otherwise.
    """

    return str(getattr(reason, "value", reason) or "") == SESSION_CLOSE_REASON_ERROR

# How long the worker may spend telling a browser test that it failed, end to
# end and across every retry. One number because two parties enforce it: the
# reporter's own deadline and the teardown's wait on it. Held apart they drift,
# and the losing side goes on writing to a room the winning side is deleting.
BROWSER_TEST_REPORT_BUDGET_SECONDS = 5.0

_LIVEKIT_ERROR_COMPONENTS = {
    "stt_error": "stt",
    "llm_error": "llm",
    "tts_error": "tts",
    "realtime_model_error": "realtime",
}


def safe_component_error_diagnostics(error: Any) -> dict[str, Any]:
    """Return bounded type-only diagnostics for a LiveKit component error.

    ``ErrorEvent.error`` is a Pydantic wrapper (``STTError``/``TTSError``/...)
    whose ``.error`` field contains the provider exception. LiveKit's normal
    log message deliberately reduces that exception to ``Connection error``;
    that is safe for callers but not enough to distinguish a Google gRPC
    status from a local frame/loop bug. Provider exception *messages* can
    contain URLs, voice IDs, account identifiers, and credential material, so
    this helper records only class names and a bounded cause chain. It never
    stringifies an exception.
    """

    def _type_name(value: Any) -> str:
        name = type(value).__name__
        return name[:80] if name else "UnknownError"

    diagnostics: dict[str, Any] = {"error_type": _type_name(error)}
    provider_error = getattr(error, "error", None)
    if not isinstance(provider_error, BaseException):
        return diagnostics

    diagnostics["provider_error_type"] = _type_name(provider_error)
    cause_types: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = provider_error
    while current is not None and len(cause_types) < 6 and id(current) not in seen:
        seen.add(id(current))
        cause_types.append(_type_name(current))
        cause = current.__cause__ or current.__context__
        current = cause if isinstance(cause, BaseException) else None
    diagnostics["provider_cause_types"] = cause_types
    return diagnostics


def browser_test_close_component(*, reason: Any, error: Any) -> str | None:
    """Which component to report to a browser test, or ``None`` to stay quiet.

    Keyed on the session CLOSING, not on an error event, because an error is not
    the same as a dead session. LiveKit tolerates
    ``conn_options.max_unrecoverable_errors`` consecutive unrecoverable failures
    per component (three, by default) and resets that count as soon as a turn
    succeeds -- so a single ``recoverable=False`` error is a turn that failed,
    not a pipeline that stopped. Reporting one would end sessions LiveKit was
    about to recover: the same silent-failure bug with the sign flipped, and the
    more irritating half, because it fires on calls that were fine.

    ``CloseReason.ERROR`` is LiveKit giving up. Every other reason is an ordinary
    ending -- the customer hung up, the job shut down -- and says nothing.

    The component name still comes from the error the close carries, and an
    unrecognised shape reports as the session rather than as nothing: this whole
    path exists because a dead pipeline presented as a healthy one.
    """

    if not session_closed_on_error(reason):
        return None
    return livekit_error_component(error)


def livekit_error_component(error: Any) -> str:
    """Which component a LiveKit error names: stt, llm, tts, realtime, session.

    One reader for the mapping, because two of them now ask: what a browser
    test is told its pipeline lost, and whether the silent-model rule may
    treat an error as corroboration that the MODEL stopped. Those two must
    never disagree about what a ``tts_error`` is.

    Read through ``.value`` like the close reason: these are plain strings on
    today's models, but a str Enum here would stringify to its NAME and
    quietly demote every component to "session".
    """

    error_type = getattr(error, "type", "")
    return _LIVEKIT_ERROR_COMPONENTS.get(
        str(getattr(error_type, "value", error_type) or ""), "session"
    )


def browser_test_error_metadata(existing: str, *, component: str) -> str:
    """Room metadata that tells a browser test its pipeline stopped.

    A session whose TTS cannot start keeps transcribing the caller, so from the
    browser the room looks healthy and simply never answers. The worker is the
    only party that knows otherwise, and room metadata is the one channel it can
    already write and the browser already receives.

    The reservation route writes ``kind`` and ``agent_id`` into this same
    metadata at creation, so the existing document is merged rather than
    replaced.
    """

    candidate = str(component or "").strip()
    if candidate not in BROWSER_TEST_ERROR_COMPONENTS:
        candidate = "session"
    try:
        document = json.loads(str(existing or "") or "{}")
    except ValueError:
        document = {}
    if not isinstance(document, dict):
        document = {}
    document[BROWSER_TEST_ERROR_KEY] = {"component": candidate}
    return json.dumps(document, sort_keys=True, allow_nan=False)


class PreMediaTermination:
    """Cancellation-safe, idempotent cleanup for a ringing SIP participant.

    A room can exist before an agent has a runnable configuration.  This
    terminator owns the concrete participant and room cleanup independently of
    the optional control-plane call object, using one shared deadline so a
    stuck SDK operation cannot leave a PSTN caller ringing indefinitely.
    """

    def __init__(
        self,
        *,
        remove_participant: Callable[[], Awaitable[None]],
        delete_room: Callable[[], Awaitable[None]],
        report_failure: Callable[[str], Awaitable[None]] | None = None,
        mark_call_cleaned: Callable[[], None] | None = None,
        timeout_seconds: float = 2.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("pre-media termination timeout must be positive")
        self._remove_participant = remove_participant
        self._delete_room = delete_room
        self._report_failure = report_failure
        self._mark_call_cleaned = mark_call_cleaned
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._task: asyncio.Task[dict[str, Any]] | None = None

    async def terminate(self, *, reason: str) -> dict[str, Any]:
        if self._task is None:
            self._task = asyncio.create_task(
                self._terminate_once(safe_pre_media_reason(reason))
            )
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # The child task owns the external teardown.  Do not leave it
            # halfway through merely because LiveKit is stopping this worker.
            await asyncio.shield(self._task)
            raise

    async def _terminate_once(self, reason: str) -> dict[str, Any]:
        deadline = self._clock() + self._timeout_seconds
        # ensure_future, not create_task: these are ``Awaitable[None]``
        # factories, and LiveKit's own ``JobContext.delete_room`` already hands
        # back a tracked Task rather than a bare coroutine. create_task rejects
        # that with a TypeError raised inside the teardown path itself, so the
        # failure that pre-media termination exists to clean up would take the
        # cleanup down with it.
        actions = {
            "participant": asyncio.ensure_future(self._remove_participant()),
            "room": asyncio.ensure_future(self._delete_room()),
        }
        remaining = max(0.0, deadline - self._clock())
        done, pending = await asyncio.wait(set(actions.values()), timeout=remaining)
        errors: list[str] = []
        results: dict[str, bool] = {}
        for name, task in actions.items():
            if task in pending:
                task.cancel()
                task.add_done_callback(self._consume_late_result)
                errors.append(f"{name}:TimeoutError")
                results[name] = False
                continue
            try:
                task.result()
                results[name] = True
            except asyncio.CancelledError:
                errors.append(f"{name}:CancelledError")
                results[name] = False
            except Exception as exc:
                if self._not_found_is_success(exc):
                    results[name] = True
                else:
                    errors.append(f"{name}:{type(exc).__name__}")
                    results[name] = False
        # Mark the in-memory session clean only after both real cleanup
        # backstops completed.  Otherwise a later finalizer can retry safely.
        if results["participant"] and results["room"] and self._mark_call_cleaned:
            try:
                self._mark_call_cleaned()
            except Exception as exc:
                errors.append(f"control_plane_cleanup:{type(exc).__name__}")
        lifecycle_recorded = False
        if self._report_failure is not None:
            try:
                await asyncio.wait_for(
                    self._report_failure(reason), timeout=self._timeout_seconds
                )
                lifecycle_recorded = True
            except Exception as exc:
                errors.append(f"lifecycle:{type(exc).__name__}")
        return {
            "reason": reason,
            "participant_terminated": results["participant"],
            "room_terminated": results["room"],
            "lifecycle_recorded": lifecycle_recorded,
            "errors": errors,
        }

    @staticmethod
    def _consume_late_result(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _not_found_is_success(exc: Exception) -> bool:
        return (
            str(getattr(exc, "code", "") or "").lower().replace("-", "_")
            in {"not_found", "notfound", "404"}
            or getattr(exc, "status", None) == 404
            or type(exc).__name__
            in {"NotFoundError", "RoomNotFoundError", "ParticipantNotFoundError"}
        )


async def terminate_pre_media_call(
    *,
    call: Any | None,
    adapter: Any | None,
    call_id: str,
    reason: str,
    remove_participant: Callable[[], Awaitable[None]],
    delete_room: Callable[[], Awaitable[None]],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bridge concrete LiveKit teardown to canonical ``sip.failed`` safely.

    ``details`` carries structured evidence beyond the bounded ``reason``
    string, at whichever pre-media stage the refusal happened: the SIP response
    code and reason phrase when the far end answered with one, and otherwise
    the client's own error code, from a closed vocabulary, for a refusal that
    never became a SIP response. That lets the control plane tell a ring
    timeout from a carrier rejection without depending on prose.

    Only what describes the CALL travels here, because this event is readable
    by the caller. The media plane's own message about a refusal -- which names
    our infrastructure rather than their call -- is operator-only and rides on
    the worker job record instead.
    """

    resolved_call_id = str(call_id or "").strip()

    async def report_failure(reason_code: str) -> None:
        if (
            adapter is None
            or not resolved_call_id
            or bool(getattr(call, "suppress_lifecycle_reporting", False))
        ):
            return
        await asyncio.to_thread(
            adapter.report_failed_before_media,
            call_id=resolved_call_id,
            event_id=f"{resolved_call_id}:pre-media-{reason_code}",
            reason=reason_code,
            details=details,
        )

    def mark_call_cleaned() -> None:
        if call is None:
            return
        cleanup = getattr(call, "cleanup", None)
        if callable(cleanup):
            cleanup(remove_participant=lambda: None, delete_room=lambda: None)

    return await PreMediaTermination(
        remove_participant=remove_participant,
        delete_room=delete_room,
        report_failure=report_failure
        if adapter is not None and resolved_call_id
        else None,
        mark_call_cleaned=mark_call_cleaned if call is not None else None,
    ).terminate(reason=reason)


class SemanticTurnAdmission:
    """Close transcript and conversation admission once terminal teardown starts.

    Two stages, because on a voice-to-voice call the ending and the goodbye
    overlap. Such a model says its goodbye itself and calls ``end_call`` in the
    same turn -- LiveKit cannot speak a text farewell for it -- and the SDK
    commits that goodbye as an assistant item only after its audio has played,
    which is AFTER the tool closed admission. A single switch threw the one
    utterance that ended the call out of its own evidence.

    ``close_caller_input`` is the first stage: nothing the caller says, and no
    TTS text, is admitted any more. The model's own assistant items still are,
    until ``close`` -- which the teardown calls once the goodbye has played.
    """

    def __init__(self) -> None:
        self._caller_closed = False
        self._closed = False

    def close_caller_input(self) -> None:
        self._caller_closed = True

    def close(self) -> None:
        self._caller_closed = True
        self._closed = True

    def admits_transcript(self) -> bool:
        return not self._caller_closed

    def admits_conversation_item(self) -> bool:
        return not self._caller_closed

    def admits_assistant_item(self) -> bool:
        return not self._closed


async def admitted_tts_text(
    text: AsyncIterable[Any],
    *,
    speech_id: str,
    recorder: Any,
    admission: SemanticTurnAdmission,
) -> AsyncIterator[Any]:
    """Bind text immediately before the TTS node receives each chunk.

    The worker wrapper owns the LiveKit ``SpeechHandle`` integration; this
    small SDK-free iterator owns the replayable ordering contract.  A chunk is
    recorded only while semantic admission is open, and only before it reaches
    the synthesizer.  It still yields every chunk after admission closes so
    LiveKit can finish its own teardown without this observability seam
    changing media behavior.
    """

    async for chunk in text:
        if admission.admits_conversation_item() and isinstance(chunk, str):
            recorder.bind_preplayout_agent_text(speech_id, chunk)
        yield chunk


async def bound_output_transcript(
    text: AsyncIterable[Any],
    *,
    speech_id: str,
    recorder: Any,
    admission: SemanticTurnAdmission,
) -> AsyncIterator[Any]:
    """Bind a speech-to-speech engine's transcript of its own speech.

    The counterpart of ``admitted_tts_text`` for an engine with no TTS node:
    the text crossing the agent's transcription node IS the model's output
    transcript for the speech named by ``speech_id``. Bound while the
    assistant side is admissible -- a voice-to-voice model speaks its own
    goodbye after end_call has closed the caller's side. Every chunk is still
    yielded; this seam never changes what reaches the transcript output.
    """

    async for chunk in text:
        if admission.admits_assistant_item() and isinstance(chunk, str):
            recorder.bind_preplayout_agent_text(speech_id, chunk)
        yield chunk


async def caller_transcripts_until_terminal(
    events: AsyncIterable[Any],
    *,
    admission: SemanticTurnAdmission,
) -> AsyncIterator[Any]:
    """Hand caller transcripts to the SDK only while the call is not terminal.

    The end-call action closes semantic admission before its goodbye is
    scheduled, and from that moment the model must not get another turn.
    Nothing in the pinned SDK enforces that: with VAD turn detection it starts
    an LLM generation on every caller final transcript that arrives while no
    speech is playing (preemptive generation), so a caller answering the
    farewell -- whose transcript lands right after the goodbye finishes and
    before the session close pauses scheduling -- opened a model turn into a
    conversation that had already ended. The model answered it with no
    candidates, the SDK retried through the bounded session close, and the
    failed handle was stamped onto the goodbye's playout evidence.

    Sits outermost in the agent's STT node, after the interruption word gate,
    so a transcript that gate releases late is dropped too. The source keeps
    being consumed: the SDK's STT pump treats an ended node as a closed audio
    input, and the stream must stay alive until the session itself closes.
    Interim and final events are dropped alike, so neither preemptive
    generation nor an end-of-turn can start from them.
    """

    async for event in events:
        if admission.admits_transcript():
            yield event


def cancel_stray_model_generation(session: Any) -> bool:
    """Cancel a model generation the SDK may have started after end_call.

    Called at the goodbye's own ``speaking`` edge, when the farewell is the
    session's current speech and cannot be interrupted. The one caller turn
    the STT filter cannot stop -- a final transcript already queued inside
    the SDK when admission closed -- can start a preemptive generation in
    the scheduler gap before the farewell becomes current. The pinned SDK's
    public ``interrupt()`` cancels that pending generation as its FIRST step
    and only then refuses to interrupt the uninterruptible farewell, so the
    ``RuntimeError`` it raises here is the expected outcome, not a failure.
    Nothing else is running at this edge: the model turn that requested the
    ending finished before the farewell could be scheduled behind it.

    Returns whether the SDK was asked. Never raises; media must not depend on
    this bookkeeping.
    """

    interrupt = getattr(session, "interrupt", None)
    if not callable(interrupt):
        return False
    try:
        interrupt()
    except RuntimeError:
        return True
    except Exception:
        return False
    return True


#: Caller speech segments waiting for the engine's next turn. Every turn
#: claims the segments before it, so this only fills when the engine stays
#: silent through a long stretch of caller activity; past it, later segments
#: are not recorded, and the count of them is reported at teardown.
MAX_UNCLAIMED_CALLER_SEGMENTS = 64
#: Engine speeches awaiting their done callback. Each is removed when it
#: finishes, so this bounds only a runaway of speeches that never finish.
MAX_PENDING_ENGINE_SPEECHES = 64
#: The chat items that prove a speech took its turn even when nothing played:
#: a reply the engine wrote down, or a tool it called.
_TURN_ITEM_TYPES = frozenset({"message", "function_call"})


class SemanticTurnEventRecorder:
    """Translate pinned LiveKit session events into PR #79 turn evidence.

    The worker owns the wiring, this object owns the contract.  It exists as
    a module-level seam so the event sequence that actually feeds the evidence
    recorder can be driven offline.  The mid-reply hangup bug was a hole in
    exactly that contract, and every test at the time fabricated recorder
    state instead of replaying the events that produce it.
    """

    def __init__(
        self,
        *,
        recorder: Any,
        telemetry_collector: Any,
        admission: SemanticTurnAdmission,
        first_message: str,
        provider_labels: Mapping[str, str],
        emit: Callable[..., None],
        model_labels: Mapping[str, str] | None = None,
        playout_hold_reason: Callable[[], str] | None = None,
        caller_transcript_observation: (
            Callable[[str], CallerFinalObservation | None] | None
        ) = None,
        speech_to_speech: bool = False,
        caller_spoke_without_caption: Callable[[float], bool] | None = None,
        engine_turn_heard: Callable[[], None] | None = None,
        correlate_replies: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recorder = recorder
        # A speech-to-speech engine ends a caller turn only on its own quiet
        # input, which stops arriving at hangup: the caller's last utterance
        # is then an interim that never becomes final. Kept per engine item,
        # admitted interims only, until a final for the same item arrives.
        self._speech_to_speech = bool(speech_to_speech)
        # The engine began a generation of its own: not asked by the worker,
        # not while the caller's audio was withheld from it, and not its
        # automatic reply to a tool result. On an engine whose generations are
        # server turns that is the server having heard the caller (the
        # silence rule's third witness); the receiver decides whether this
        # engine's are. Reported at creation, not at the end of the speech:
        # the rule polls, and a reply that plays for a while must not finish
        # after the deadline it would have averted.
        self._engine_turn_heard = engine_turn_heard if self._speech_to_speech else None
        # Once a tool has run (``function_tools_executed``), the engine's next
        # unasked generation is its answer to the tool's output, not the
        # caller: with the tool behaviour this worker configures, Gemini
        # answers every tool output it is sent, one whose reply the SDK did
        # not ask for included (the pinned adapter warns exactly that and
        # sends it regardless). Armed from the event, which the SDK emits
        # before it hands the output over, because the tool-calling speech's
        # own done callback can arrive AFTER the reply's creation.
        self._auto_tool_reply_expected = False
        self._open_caller_turns: dict[str, tuple[str, float]] = {}
        # When each caller item's first interim arrived, per engine item. The
        # engine sends the final late -- GPT-Live 0.8 s of input after the
        # last fragment or at the next utterance's first one, Gemini Live
        # when its reply completes -- so only the first interim is stamped
        # while the caller is still speaking. The recorder joins the item to
        # its speaking interval on it.
        self._caller_speech_started: dict[str, float] = {}
        # Agent speeches that reached the line, counted at the SDK's own
        # speaking edge -- a tool-only generation never moves it.
        self.agent_speaking_edges = 0
        self._telemetry = telemetry_collector
        self._correlate_replies = correlate_replies
        self._admission = admission
        self._first_message = first_message
        self._provider_labels = dict(provider_labels)
        self._model_labels = dict(model_labels or {})
        self._emit = emit
        # Reads the just-finished turn's floor-yield label from the playout
        # hold guard, so a turn whose audio waited on a talking far end says
        # so on its own ``playout.first_worker`` span instead of looking like
        # an unexplained platform stall.
        self._playout_hold_reason = playout_hold_reason
        self._caller_transcript_observation = caller_transcript_observation
        self.caller_turns = 0
        # The caller spoke, the engine took its next turn on its own, and no
        # caption exists for that speech. A voice-to-voice engine only: the
        # cascade's own transcriber answers the question there. The engine's
        # starting a reply is NOT the witness by itself -- a duplex engine
        # opens one generation per audio burst, so a pause inside its own
        # reply, a filler before a delegated answer and the tail of the
        # opening all look like replies nobody prompted. The caller's voice
        # activity is the witness; the engine's turn is what claims it.
        self._caller_spoke_without_caption = (
            caller_spoke_without_caption if self._speech_to_speech else None
        )
        self._clock = clock
        self._caller_audio_withheld: Callable[[], bool] | None = None
        # [started_at, covered, over_agent, recovered] per voice-activity
        # segment not yet claimed. Covered: a caption arrived for it, or the
        # worker asked the engine for a turn after it, so the engine's next
        # turn may be answering that ask rather than the caller. Recovered: it
        # began while the model was deafened and ended after it could hear.
        self._caller_segments: list[list[Any]] = []
        # speech id -> when it was created, who started it, whether it spoke.
        self._engine_speeches: dict[str, dict[str, Any]] = {}
        self._agent_speaking = False
        # A segment that began while the model was deafened, as (start, over
        # the agent): recorded after all if the model hears by its end.
        self._withheld_segment: tuple[float, bool] | None = None
        # A turn the model took after the hold let go finished while that
        # segment was still open. The segment is claimed when it ENDS, not
        # then: a caption can still arrive before the end edge, and a moment
        # written early could not be taken back.
        self._recovered_claim_pending = False
        # When a caption or a worker ask last covered the unclaimed speech: a
        # segment recorded late, at its end, is covered by anything since
        # its start, exactly as if it had been in the list all along.
        self._last_covered_at: float | None = None
        self.caller_segments_recorded = 0
        self.caller_segments_withheld = 0
        self.caller_segments_withheld_then_heard = 0
        # Moments the rule decided, and the ones the typed recorder kept (it
        # refuses past its cap and after teardown).
        self.caller_spoke_marked = 0
        self.caller_spoke_kept = 0
        # A segment that began while the agent was audible is a barge-in or
        # the agent's own voice returning on the caller's leg, and nothing
        # here can tell which. Counted apart, so the batch can.
        self.caller_segments_over_agent = 0
        self.caller_spoke_marked_over_agent = 0
        # Past a cap nothing is recorded, and that is counted, never silent.
        self.caller_segments_dropped_at_cap = 0
        self.engine_speeches_untracked = 0

    def note_caller_audio_withheld_by(self, withheld: Callable[[], bool]) -> None:
        """Bind the predicate saying the model cannot hear the caller now.

        Speech the model never received cannot be what its next turn
        answered: the duplex opening hold swaps the caller's audio for silence
        until the opening has been spoken, and the tail bursts of that opening
        are replies of the engine's own.
        """
        self._caller_audio_withheld = withheld

    def _withheld(self) -> bool:
        if self._caller_audio_withheld is None:
            return False
        try:
            return bool(self._caller_audio_withheld())
        except Exception:  # noqa: BLE001 - unknown is not evidence the model heard it
            return True

    def _note_caller_segment(self) -> None:
        if self._caller_spoke_without_caption is None or not self._admission.admits_transcript():
            return
        if self._withheld():
            self.caller_segments_withheld += 1
            self._withheld_segment = (self._clock(), self._agent_speaking)
            return
        self._record_caller_segment(self._clock(), over_agent=self._agent_speaking)

    def _note_caller_segment_ended(self) -> None:
        """A segment that began while the model was deafened, ending now.

        The hold lets go about a second after the opening goes quiet, and a
        callee who talked over the end of the greeting is often still talking
        then: the model hears the rest and answers it, with no new voice
        activity edge to record. So a withheld segment is re-checked when it
        ends, and recorded if the model can hear by then. It keeps its true
        start -- where the caller began speaking, which is what the moment
        marks -- and it is counted apart.
        """
        withheld, self._withheld_segment = self._withheld_segment, None
        claim_now, self._recovered_claim_pending = self._recovered_claim_pending, False
        if withheld is None or self._caller_spoke_without_caption is None or self._withheld():
            return
        started_at, over_agent = withheld
        # Once caller admission has closed (end_call closes caller input
        # during the goodbye; hang-up closes it before teardown settles), the
        # segment is still recorded when a turn that could be answering it
        # already finished, or is still in flight and will finish at settle
        # at the latest. The caller spoke while admission was open.
        if not (claim_now or self._admission.admits_transcript() or self._answer_in_flight(started_at)):
            return
        # Placed where the caller began. A speech created while the model was
        # still deafened -- a tail burst of the opening above all -- cannot be
        # answering it; one created after the hold let go can, even before
        # this end edge (the engine decides the caller stopped before the
        # voice-activity detector's own hangover does).
        if self._record_caller_segment(
            started_at,
            over_agent=over_agent,
            recovered=True,
            covered=self._last_covered_at is not None and self._last_covered_at >= started_at,
        ):
            self.caller_segments_withheld_then_heard += 1
            if claim_now:
                # The turn that answered it already finished: it claims the
                # segment now, with every caption that arrived until this
                # end edge already counted.
                segment = self._caller_segments.pop()
                if not segment[1]:
                    try:
                        self._write_moment(segment)
                    except Exception as exc:  # noqa: BLE001 - evidence never breaks media
                        self._emit("caller_spoke_marker_failed", stage="segment_ended", error=type(exc).__name__)

    def _answer_in_flight(self, started_at: float) -> bool:
        """An unfinished speech that may claim a recovered segment begun at ``started_at``."""
        return any(
            self._may_answer_recovered(speech) and speech["created_at"] >= started_at
            for speech in self._engine_speeches.values()
        )

    @staticmethod
    def _may_answer_recovered(speech: dict[str, Any]) -> bool:
        # Created while the caller was admitted and the model could hear
        # them, and not the automatic reply to a tool's output.
        return speech["admitted"] and not speech["withheld"] and not speech["auto_tool_reply"]

    def _record_caller_segment(
        self,
        started_at: float,
        *,
        over_agent: bool,
        recovered: bool = False,
        covered: bool = False,
    ) -> bool:
        if len(self._caller_segments) >= MAX_UNCLAIMED_CALLER_SEGMENTS:
            self.caller_segments_dropped_at_cap += 1
            return False
        self._caller_segments.append([started_at, covered, over_agent, recovered])
        self.caller_segments_recorded += 1
        if over_agent:
            self.caller_segments_over_agent += 1
        return True

    def _cover_unclaimed_segments(self) -> None:
        # Every unclaimed segment, not only the latest. A caption can arrive
        # late (Gemini Live sends it when its reply completes). A worker ask
        # may be answered first by a generation of the engine's own: a lone
        # tool call never resolves the ask it serves, so it is engine-
        # initiated. Covering too widely can only cost a moment, never
        # invent one.
        self._last_covered_at = self._clock()
        for segment in self._caller_segments:
            segment[1] = True

    def on_speech_created(self, event: Any) -> None:
        """Remember an engine speech until it finishes, then let it claim.

        An unasked generation is also reported as the engine's own turn, at
        creation, unless the caller's audio is withheld from the engine right
        now (the tail bursts of a held opening) or it is the automatic reply
        to the tool the previous turn called.
        """
        if self._caller_spoke_without_caption is None and self._engine_turn_heard is None:
            return
        try:
            handle = getattr(event, "speech_handle", None)
            speech_id = str(getattr(handle, "id", "") or "")
            user_initiated = bool(getattr(event, "user_initiated", False))
            auto_tool_reply = False
            if not user_initiated:
                auto_tool_reply, self._auto_tool_reply_expected = self._auto_tool_reply_expected, False
                if self._engine_turn_heard is not None and not auto_tool_reply and not self._withheld():
                    self._engine_turn_heard()
            if self._caller_spoke_without_caption is None:
                return
            if user_initiated:
                # The worker's own ask (the opening, a check-in, a
                # regeneration): whatever the engine does next may be
                # answering it, so the caller speech before it is covered --
                # even when this handle cannot be tracked below.
                self._cover_unclaimed_segments()
            if handle is None or not speech_id or speech_id in self._engine_speeches:
                return
            if len(self._engine_speeches) >= MAX_PENDING_ENGINE_SPEECHES:
                self.engine_speeches_untracked += 1
                return
            self._engine_speeches[speech_id] = {
                "created_at": self._clock(),
                # Created while the model could not hear the caller: it cannot
                # be answering speech it only began hearing later.
                "withheld": self._withheld(),
                # The automatic reply to a tool's output answers the tool,
                # not speech the model began hearing only after the hold.
                "auto_tool_reply": auto_tool_reply,
                # Created after caller admission closed (a generation after
                # end_call, say): it is not answering the caller.
                "admitted": self._admission.admits_transcript(),
                "spoke": False,
                "handle": handle,
            }
            handle.add_done_callback(
                lambda done, speech_id=speech_id: self._on_engine_speech_done(speech_id, done)
            )
        except Exception as exc:  # noqa: BLE001 - evidence never breaks media
            self._emit("caller_spoke_marker_failed", stage="speech_created", error=type(exc).__name__)

    def on_function_tools_executed(self, _event: Any) -> None:
        """A tool ran: the engine's next unasked generation answers its output.

        Not only when the SDK expects a reply. A ``FunctionCallOutput`` whose
        reply is not required (a refused ``end_call`` after the guard's
        ``cancel_tool_reply()``) is still sent, and Gemini answers it unless
        the tools were declared non-blocking, which this worker does not do;
        the pinned adapter warns of exactly that and sends it regardless.
        So every tool execution arms the exclusion, and the generation that
        answers it is never the caller's. The event is emitted before the SDK
        hands the output over, so the flag is up before the reply's
        ``speech_created`` however the tool-calling speech's done callback is
        ordered against it. The cost of arming too widely is one unasked
        generation not reported when the engine happens to stay silent; the
        VAD and the captions still witness that caller, and the one after is
        reported.
        """
        if self._engine_turn_heard is None:
            return
        self._auto_tool_reply_expected = True

    def settle_engine_speeches(self) -> int:
        """Let every speech still in flight claim now, oldest first.

        Called once at teardown, before the typed event recorder closes: a
        reply the caller hung up on finishes only when the session closes,
        which is after that recorder stops taking rows, and the moment its
        caller spoke would otherwise be lost with it. A speech that took no
        turn by now claims nothing, exactly as if it had finished. Returns
        how many were settled.
        """

        try:
            pending = sorted(
                self._engine_speeches.items(), key=lambda entry: entry[1]["created_at"]
            )
        except Exception as exc:  # noqa: BLE001 - teardown must not fail here
            self._emit("caller_spoke_marker_failed", stage="settle", error=type(exc).__name__)
            return 0
        for speech_id, speech in pending:
            self._on_engine_speech_done(speech_id, speech.get("handle"))
        if self._recovered_claim_pending and self._withheld_segment is not None:
            # The caller was still speaking when the call ended; the turn that
            # answered them has finished, so the segment ends here.
            self._note_caller_segment_ended()
        return len(pending)

    def _on_engine_speech_done(self, speech_id: str, handle: Any) -> None:
        """A finished speech that took a turn claims the speech before it.

        It took a turn when it was heard, wrote a reply down, or called a
        tool. It claims every caller segment that began BEFORE it was created
        -- a segment that began later cannot be what it answered, and waits
        for the next turn -- and writes one moment where the first uncovered
        one began. A turn the worker asked for covered them all when it was
        created, so it claims them and writes none.
        """
        try:
            speech = self._engine_speeches.pop(speech_id, None)
            if speech is None:
                return
            took_turn = speech["spoke"] or any(
                str(getattr(item, "type", "") or "") in _TURN_ITEM_TYPES
                for item in (getattr(handle, "chat_items", None) or [])
            )
            if not took_turn:
                return
            if (
                self._withheld_segment is not None
                and self._may_answer_recovered(speech)
                and not self._withheld()
                and self._withheld_segment[0] <= speech["created_at"]
            ):
                # A segment that began in the hold and is still running: this
                # turn, created after the model could hear, may be answering
                # it before the detector's end edge. It is claimed at that end
                # edge (or at teardown), once no caption can still cover it.
                self._recovered_claim_pending = True
            created_at = speech["created_at"]
            claimed = [
                segment
                for segment in self._caller_segments
                if segment[0] <= created_at and not (segment[3] and not self._may_answer_recovered(speech))
            ]
            if not claimed:
                return
            self._caller_segments = [
                segment
                for segment in self._caller_segments
                if not any(segment is taken for taken in claimed)
            ]
            # Only uncovered speech is a moment: a covered segment already
            # has its caption, or was followed by a worker ask the turn may
            # be answering. One unanswered covered segment must not hide an
            # uncovered one after it.
            uncovered = [segment for segment in claimed if not segment[1]]
            if not uncovered:
                return
            self._write_moment(uncovered[0])
        except Exception as exc:  # noqa: BLE001 - evidence never breaks media
            self._emit("caller_spoke_marker_failed", stage="speech_done", error=type(exc).__name__)

    def _write_moment(self, segment: list[Any]) -> None:
        """One moment, where ``segment`` began; counted as decided and as kept."""
        self.caller_spoke_marked += 1
        if segment[2]:
            self.caller_spoke_marked_over_agent += 1
        if self._caller_spoke_without_caption(float(segment[0])):
            self.caller_spoke_kept += 1

    def _latest_playout_hold_reason(self) -> str:
        """The floor-yield label for the turn just finished, or "".

        Never let an observability accessor break a turn: a missing or broken
        guard must cost the label, not the telemetry bundle.
        """
        if self._playout_hold_reason is None:
            return ""
        try:
            return str(self._playout_hold_reason() or "")
        except Exception:
            return ""

    def on_user_state_changed(self, event: Any) -> None:
        self._recorder.mark_state("caller", str(event.old_state), str(event.new_state))
        if str(event.new_state) == "speaking" and str(event.old_state) != "speaking":
            self._note_caller_segment()
        elif str(event.old_state) == "speaking" and str(event.new_state) != "speaking":
            self._note_caller_segment_ended()

    def on_agent_state_changed(self, event: Any, *, speech_id: str | None = None) -> None:
        if str(event.new_state) == "speaking" and str(event.old_state) != "speaking":
            self.agent_speaking_edges += 1
            speech = self._engine_speeches.get(str(speech_id or ""))
            if speech is not None:
                speech["spoke"] = True
        self._agent_speaking = str(event.new_state) == "speaking"
        if speech_id is None:
            self._recorder.mark_state("agent", str(event.old_state), str(event.new_state))
            return
        # The SpeechHandle whose first frame this edge is, so the recorder can
        # join that speech's text to this interval by identity.
        self._recorder.mark_state(
            "agent", str(event.old_state), str(event.new_state), speech_id=speech_id
        )

    def flush_open_caller_turns(self) -> int:
        """Record every still-open caller turn as final, marked as cut at close.

        Called once, at teardown, just before the recorder is finalized. Not
        gated on admission: this is evidence of words the caller said while
        the turn was admissible (only admitted interims are kept), and
        admission is already closed by the time a hangup reaches teardown.
        Returns how many turns were flushed.
        """

        pending = sorted(self._open_caller_turns.items(), key=lambda turn: turn[1][1])
        self._open_caller_turns.clear()
        for item_id, (text, observed_at) in pending:
            self._recorder.add_text(
                "caller",
                text,
                at=observed_at,
                caption_ingested_at=observed_at,
                finalized_at_close=True,
                speech_started_at=self._caller_speech_started.pop(item_id, None),
            )
        self._caller_speech_started.clear()
        return len(pending)

    def _note_caller_speech_started(self, event: Any, ingested_at: float | None) -> None:
        """Stamp an item's first admitted non-blank interim, once.

        An item with no id cannot be told from another, so it gets no stamp,
        and the recorder leaves it untimed. The stamp is consumed only by a
        final that is recorded, or by the flush at close.
        """

        item_id = str(getattr(event, "item_id", "") or "")
        if (
            item_id
            and not event.is_final
            and str(event.transcript or "").strip()
            and self._admission.admits_transcript()
        ):
            self._caller_speech_started.setdefault(
                item_id, time.monotonic() if ingested_at is None else float(ingested_at)
            )

    def _track_open_caller_turn(self, event: Any, ingested_at: float | None) -> None:
        item_id = str(getattr(event, "item_id", "") or "")
        if event.is_final:
            # Replaced only by a final that is recorded. A final refused
            # because end_call closed the caller's side after the interim was
            # admitted must not take the admitted words with it.
            if str(event.transcript or "").strip() and self._admission.admits_transcript():
                self._open_caller_turns.pop(item_id, None)
            return
        text = str(event.transcript or "")
        if not text.strip() or not self._admission.admits_transcript():
            return
        self._open_caller_turns[item_id] = (
            text,
            time.monotonic() if ingested_at is None else float(ingested_at),
        )

    def on_user_input_transcribed(self, event: Any, *, ingested_at: float | None = None) -> None:
        if self._speech_to_speech:
            self._track_open_caller_turn(event, ingested_at)
            self._note_caller_speech_started(event, ingested_at)
            if str(event.transcript or "").strip():
                self._cover_unclaimed_segments()
        if event.is_final and str(event.transcript).strip():
            text = str(event.transcript)
            observation = (
                self._caller_transcript_observation(text)
                if self._caller_transcript_observation is not None
                else None
            )
            # Consume the released observation even after admission closes,
            # otherwise a late event could shift the next FIFO match.
            if not self._admission.admits_transcript():
                return
            self.caller_turns += 1
            item_id = str(getattr(event, "item_id", "") or "")
            speech_started_at = (
                self._caller_speech_started.pop(item_id, None) if item_id else None
            )
            self._recorder.add_text(
                "caller",
                text,
                caption_ingested_at=ingested_at,
                at=None if observation is None else observation.observed_at,
                measured_duration=(
                    None if observation is None else observation.measured_duration_seconds
                ),
                confidence=None if observation is None else observation.confidence,
                speech_started_at=speech_started_at,
            )

    def on_conversation_item_added(self, event: Any) -> None:
        # Read defensively: before the role decided admission, a closed
        # admission returned before touching the event at all.
        item = getattr(event, "item", None)
        role = str(getattr(item, "role", ""))
        admitted = (
            self._admission.admits_assistant_item()
            if role == "assistant"
            else self._admission.admits_conversation_item()
        )
        if not admitted:
            return
        text = str(
            getattr(item, "text_content", "") or getattr(item, "raw_text_content", "")
        )
        metrics = getattr(item, "metrics", None)
        if role == "user":
            self._telemetry.observe_user(text=text, metrics=metrics,
                                         item_id=str(getattr(item, "id", "") or ""))
            return
        if role != "assistant":
            return
        if self.caller_turns < 1 or is_greeting_fragment(text, self._first_message):
            # This reply gets no telemetry span, so its hold report has no span
            # to land on. Drain it here: leaving it pending would hand the
            # opening greeting's hold to the first semantic reply and label a
            # clean turn as held, which is worse than no label at all.
            self._latest_playout_hold_reason()
            return
        self._recorder.add_text("agent", text)
        turn_telemetry = self._telemetry.observe_assistant(
            text=text,
            metrics=metrics,
            provider_labels=self._provider_labels,
            model_labels=self._model_labels,
            playout_reason_code=self._latest_playout_hold_reason(),
            user_item_id=((getattr(item, "extra", None) or {}).get(
                self._telemetry.REPLY_USER_ITEM_KEY, "")
                if self._correlate_replies else None),
        )
        if turn_telemetry is not None:
            self._emit(
                "turn_telemetry",
                turn=turn_telemetry["turn_index"],
                span_count=turn_telemetry["span_count"],
            )


def caller_disconnect_handler(
    *, caller_identity: str, finish: Callable[[], None]
) -> Callable[[Any], None]:
    """Finish the session only when the CALLER leaves, not any participant."""

    def handler(disconnected: Any) -> None:
        if str(getattr(disconnected, "identity", "")) == caller_identity:
            finish()

    return handler


@dataclass(eq=False)
class ProductionRoomDisconnectHandler:
    """Callable room callback that also records transport-loss provenance.

    ``stamp_lost`` names the ending before the session is finished. It used to
    be absent, and the call it woke reported no reason at all: the media
    session had gone away, which is not any of the endings the rest of this
    worker knows how to name, so the customer read an empty answer on exactly
    the calls that stopped mid-sentence.

    It is called BEFORE ``finish`` because the arbiter is first-wins: on an
    ORDINARY ending the room event also fires, moments after the worker
    deleted the room itself, and by then the real reason is already stamped
    and this one is refused. Stamping after ``finish`` would race that.

    It returns whether it stamped, and that return is load-bearing rather than
    informational: ``SessionEndArbiter.end`` FINISHES the session as part of
    stamping. A handler that stamped and then finished again would call the
    teardown twice for one ending. The second call is harmless today only
    because the session's finish happens to be idempotent, which is not a
    property this handler should be relying on. When the stamp is refused --
    something else already ended the call -- ``finish`` still runs, because
    waking the teardown on a lost room is what this handler existed for before
    it named anything.
    """

    finish: Callable[[], None]
    emit: Callable[..., None]
    stamp_lost: Callable[[], bool] | None = None
    _triggered: bool = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        if self._triggered:
            return
        self._triggered = True
        self.emit("room_disconnected", shutdown_source="livekit_room")
        stamped = bool(self.stamp_lost()) if self.stamp_lost is not None else False
        if not stamped:
            self.finish()


def production_room_disconnect_handler(
    *,
    finish: Callable[[], None],
    emit: Callable[..., None],
    stamp_lost: Callable[[], bool] | None = None,
) -> ProductionRoomDisconnectHandler:
    """Wake the production session when its LiveKit room disconnects.

    Agent-to-agent Simulation Lab rooms can close before a remote-participant
    disconnect callback is delivered.  The pinned LiveKit worker waits for the
    production entrypoint and cancels it *before* job-shutdown callbacks run,
    so a shutdown callback cannot prevent that cancellation.  The room event
    arrives earlier and only opens the existing teardown path; it never
    performs cleanup in a second task.
    """
    return ProductionRoomDisconnectHandler(finish=finish, emit=emit, stamp_lost=stamp_lost)


def transcriber_refused_request(error: Any) -> bool:
    """Whether a LiveKit component error is the transcriber refusing the request it was opened with.

    Only ``stt_error``, and only HTTP 400: what a speech-to-text provider answers
    a request it will not run (Deepgram: an unknown language code at the
    WebSocket handshake, the common case). A voice is not judged here -- a TTS
    400 can be about the synthesised output (Gemini TTS reports a reply with no
    audio as 400) -- and an auth, quota or server status is not a refused
    request. Reads the status off the provider exception; never its message.
    """

    if livekit_error_component(error) != "stt":
        return False
    status = getattr(getattr(error, "error", None), "status_code", None)
    return isinstance(status, int) and not isinstance(status, bool) and status == 400


def production_session_close_handler(
    *,
    stamp_lost: Callable[[], bool],
    is_finishing: Callable[[], bool],
    stamp_refused: Callable[[], bool] | None = None,
) -> Callable[[Any], None]:
    """End the call when the SDK session closes on an unrecoverable error.

    The session closing is the engine's own teardown, not the call's. Before
    this the worker did nothing with it on a phone call, so a fatal model or
    component error left the caller's leg up with nobody on it: the worker was
    parked on its bare ``await finished.wait()`` with a dead session, and the
    leg ran, billed, until the maximum call length or the reconciler.

    ``stamp_lost`` is the arbiter's ``end`` for ``model_session_lost``; it
    finishes the session as part of stamping, so this handler finishes nothing
    itself. It stands down when the call is already ending -- the ordinary
    teardown closes this same session and its close carries another reason,
    but a close that arrives on an ending some path decided WITHOUT the
    arbiter must not put a platform reason on it either.
    """

    def handler(event: Any) -> None:
        if not session_closed_on_error(getattr(event, "reason", None)):
            return
        if is_finishing():
            return
        if stamp_refused is not None and transcriber_refused_request(getattr(event, "error", None)):
            # The close carries the error the session gave up on: a
            # transcriber that refused the request is named for what it is.
            stamp_refused()
            return
        stamp_lost()

    return handler


# The strings this worker may stamp on ``call_sessions.termination_detail``
# through the ordinary terminal lifecycle report.  The control plane serves a
# CLOSED enum derived from this column, so a string invented here would reach a
# customer as nothing at all -- the read model drops what it does not
# recognise, and its writer narrows the value too, so an unknown one is never
# even stored.  Kept as one frozen set so a new ending has to be added on
# purpose, in both halves, rather than by typing a new literal at a call site.
#
# They divide three ways, and ``WORKER_RULE_TERMINATIONS`` below holds the
# first group: endings a platform RULE decided, as opposed to the agent
# finishing its own conversation and the media session simply going away.
# That distinction is what a customer is actually asking about, and the count
# is deliberately not written down -- it went stale the first time the
# vocabulary grew.
TERMINATION_HANGUP_REQUESTED = "hangup_requested"
TERMINATION_REMOTE_PARTY_DISCONNECTED = "remote_party_disconnected"
TERMINATION_MAX_CALL_DURATION_REACHED = "max_call_duration_reached"
TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER = "no_caller_speech_after_answer"
# The far end answered and produced audio, and none of it was a conversation:
# either a sustained source with no syllable structure, or a floor held for
# twenty seconds with no pause long enough for anyone to reply. Distinct from
# the silence above, which is the opposite evidence -- nothing was heard at
# all -- and a customer reading the two needs to be able to tell a line that
# stayed quiet from one that never stopped.
TERMINATION_NO_HUMAN_AFTER_ANSWER = "no_human_after_answer"
# The agent ended the call itself: its script reached a goodbye, or its
# workflow completed. Not a platform rule and not a hang-up -- nobody left and
# no limit was hit, the conversation simply finished.
#
# It is here because leaving it empty told a customer nothing at exactly the
# moment they most want an answer. A scripted workflow ends most of its calls
# this way, so an empty reason would have been the COMMONEST reading of a field
# whose whole purpose is "why did this call end".
TERMINATION_AGENT_COMPLETED = "agent_completed"
# The media session went away underneath a live call: the room was deleted,
# this worker's connection to it failed, or the transport dropped. Nobody hung
# up, no limit fired, and the agent did not finish -- which is why none of the
# reasons above fitted and such a call used to report nothing at all.
#
# It is deliberately NOT in ``WORKER_RULE_TERMINATIONS``: no platform rule ran.
# The worker did not decide this ending, it observed it, and saying otherwise
# would put the platform's word on someone else's ending. In particular it is
# never replaced by a pending operator hang-up that happened to be in flight;
# the operator is answered on their own channel, which reports that the room
# was already gone.
TERMINATION_MEDIA_SESSION_LOST = "media_session_lost"
# The conversation engine behind the call failed and could not be recovered:
# the SDK closed its session on an unrecoverable error -- a realtime model
# whose provider connection could not be re-established, or a component whose
# error budget ran out -- while the caller's leg was still up. The media
# session is intact, which is what separates this from the reason above: the
# room is there and the caller is on it, and nobody is left to talk to them.
#
# Before this the worker ended nothing on that close. It sat on its bare
# ``await finished.wait()`` with a dead session, and the leg ran, billed, to
# the maximum call length or a reconciler. Like ``media_session_lost`` it is
# NOT in ``WORKER_RULE_TERMINATIONS``: no platform rule decided it, the worker
# observed its own engine give up and reported what it saw.
TERMINATION_MODEL_SESSION_LOST = "model_session_lost"
# Qualifies ``model_session_lost``: the engine gave up because the TRANSCRIBER
# refused the request it was opened with (an HTTP 400). That is all the evidence
# says, so it is all the name says. The usual cause is a language or language
# code the provider does not take (measured: Deepgram answers an unknown code
# with 400 at the WebSocket handshake), and since languages are not kept as
# lists of what each provider covers, this is where one a provider does not run
# shows itself -- but the pinned plugin drops the response body, so the cause
# is not read, and the detail does not claim it.
# A detail rather than a reason of its own: an api released before it drops an
# unknown detail and keeps ``model_session_lost``, but would drop an unknown
# REASON to nothing (both deploy orders stay truthful).
DETAIL_TRANSCRIBER_REFUSED_REQUEST = "transcriber_refused_request"
# The workflow's speech listener (its speech-to-text provider) refused while
# the caller was on the line: its stream was never acknowledged, closed or
# failed, and the workflow cannot hear the caller without it. The engine that
# speaks may be fine, which is what separates this from
# ``model_session_lost``; nobody hung up, which is what the call used to be
# recorded as. Like both lost-session reasons it is an observation, not a
# rule, so it is NOT in ``WORKER_RULE_TERMINATIONS``.
TERMINATION_LISTENER_UNAVAILABLE = "listener_unavailable"
# A spoken workflow could not deliver a line before its ending: the engine
# played it without saying it in full (GPT-Live's own words fell short of the
# configured line), or the caller spoke over the step every time it was asked
# within the runner's bound. The workflow cannot go on as if the caller had
# heard it, so the call ends -- not because anyone hung up. A platform
# decision, but not the polled ladder's, so it is NOT in
# ``WORKER_RULE_TERMINATIONS``.
TERMINATION_WORKFLOW_LINE_UNDELIVERED = "workflow_line_undelivered"
# The caller finished a turn and the agent produced no reply, no tool call and
# no ending for the whole of the platform's bound, so the worker ended the call
# on its own clock. Nothing FAILED that anyone observed: on a voice-to-voice
# engine the provider reports nothing when it stops answering (measured), and
# on 2026-09-24 a GPT-Live call went silent for 77 s with no error, no close
# and no retry. This used to be stamped ``model_session_lost``, which tells a
# customer "the engine failed and could not be recovered" -- a claim about a
# failure this path never sees. It is its own reason now: what was observed is
# an agent that stopped answering, and that is what the customer is told.
#
# The two reasons are two different observations, not two detections of one
# event: the SDK closing its session is the engine giving up, and this rule is
# the agent going quiet with its session still open. Whichever reaches the
# arbiter first names the call, exactly as for every other pair of endings.
#
# A watchdog ending is never ``model_session_lost``, even when a provider error
# was seen during the window: a recoverable error is not an unrecoverable
# session loss, and only the close handler observes the latter.
TERMINATION_AGENT_UNRESPONSIVE = "agent_unresponsive"
# Qualifies the reason above: the agent was silent after somebody spoke to it.
TERMINATION_DETAIL_MODEL_UNRESPONSIVE = "model_unresponsive"
# Both sides went quiet after the agent had spoken, the agent asked whether the
# caller was still there, and nobody answered. A platform rule
# ended the call, with a goodbye: nobody hung up, no limit was hit, and the
# engine did not fail -- the caller stopped responding. It has its own path to
# ``SessionEndArbiter.end``, like ``no_human_after_answer``, so it is not in
# ``WORKER_RULE_TERMINATIONS`` (that set is the polled ladder's).
TERMINATION_CALLER_UNRESPONSIVE = "caller_unresponsive"
# The call was waiting in a phone menu or a hold queue (the navigation state)
# and a bound ran out: the hold budget, the menu budget, or the overall call
# limit reached while it was in that state. Qualified by one of the details
# below, which is what the customer's retry policy branches on. Only a call
# whose agent opted in to navigation can ever be in that state.
TERMINATION_IVR_TIMEOUT = "ivr_timeout"
# What a termination detail may be QUALIFIED with, per reason. Closed, and
# closed per reason rather than as one flat set, so a detail can never be
# stamped beside an ending it says nothing about. The control plane holds the
# same pairing and refuses anything outside it at its own writer; this one
# stops an unpairable value ever leaving the worker.
TERMINATION_REASON_DETAILS = {
    "funding_exhausted": frozenset({"insufficient_balance", "renewal_unavailable"}),
    TERMINATION_AGENT_UNRESPONSIVE: frozenset({TERMINATION_DETAIL_MODEL_UNRESPONSIVE}),
    TERMINATION_IVR_TIMEOUT: frozenset(
        {LIMIT_HOLD, LIMIT_NAVIGATION, LIMIT_DEADLINE, DETAIL_MENU_NO_PATH, DETAIL_DATA_NOT_AUTHORISED}
    ),
    # A recording and nothing else answered (no menu, no person).
    TERMINATION_NO_HUMAN_AFTER_ANSWER: frozenset({DETAIL_RECORDED_MESSAGE_ONLY}),
    # The far-end detector a navigating call depends on delivered nothing.
    TERMINATION_MEDIA_SESSION_LOST: frozenset({DETAIL_NAVIGATION_DETECTOR_LOST}),
    # The other party hung up while the call was waiting in a queue.
    TERMINATION_REMOTE_PARTY_DISCONNECTED: frozenset({DETAIL_DURING_HOLD}),
    # The transcriber refused the request it was opened with (an HTTP 400).
    TERMINATION_MODEL_SESSION_LOST: frozenset({DETAIL_TRANSCRIBER_REFUSED_REQUEST}),
}
# HOW the worker noticed, never WHAT the customer is told. Every string above
# is customer-facing, ``agent_completed`` included; NONE of the markers here
# ever is. They ride beside the reasons so an operator can tell a call that
# ended on a delivered disconnect event from one whose event never arrived and
# was caught by the poll instead. That difference
# is the incident, and it is worth being able to count -- but it is a fact
# about our own delivery, not about the customer's call, so it never becomes a
# second reason string.
DETECTED_VIA_DISCONNECT_EVENT = "disconnect_event"
DETECTED_VIA_OCCUPANCY_PROBE = "occupancy_probe"
DETECTED_VIA_WORKER_TIMER = "worker_timer"
DETECTED_VIA_HANGUP_INTENT = "hangup_intent"
# Not a detection at all: this worker did not notice the ending, it performed
# it. Kept in the same closed set so the provenance of every ending is
# answerable with one vocabulary.
DETECTED_VIA_AGENT_SCRIPT = "agent_script"
# Also not a detection of someone else's action: the room this worker was in
# went away. Kept distinct from the disconnect event so an operator can count
# a lost media session separately from a caller who hung up.
DETECTED_VIA_ROOM_LOST = "room_lost"
# The SDK session closed on an unrecoverable error underneath a live call.
# Distinct from the lost room above: the transport was fine, the engine was
# not, and an operator counting the two needs them apart.
DETECTED_VIA_SESSION_CLOSED = "session_closed"
# A workflow listener refused underneath a live call (``listener_unavailable``).
DETECTED_VIA_LISTENER_REFUSED = "listener_refused"
# A spoken workflow's line was not delivered (``workflow_line_undelivered``).
DETECTED_VIA_WORKFLOW_LINE = "workflow_line"
# Closed, like the reason set beside it. These never leave this worker's own
# evidence, but a marker invented at a call site would still be recorded as
# fact and counted in operator queries, so the set is enforced rather than
# documented.
WORKER_DETECTION_PATHS = frozenset(
    {
        "opening_failure",
        DETECTED_VIA_DISCONNECT_EVENT,
        DETECTED_VIA_OCCUPANCY_PROBE,
        DETECTED_VIA_WORKER_TIMER,
        DETECTED_VIA_HANGUP_INTENT,
        DETECTED_VIA_AGENT_SCRIPT,
        DETECTED_VIA_ROOM_LOST,
        DETECTED_VIA_SESSION_CLOSED,
        DETECTED_VIA_LISTENER_REFUSED,
        DETECTED_VIA_WORKFLOW_LINE,
    }
)


WORKER_TERMINATION_DETAILS = frozenset(
    {
        "funding_exhausted",
        TERMINATION_HANGUP_REQUESTED,
        TERMINATION_REMOTE_PARTY_DISCONNECTED,
        TERMINATION_MAX_CALL_DURATION_REACHED,
        TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER,
        TERMINATION_NO_HUMAN_AFTER_ANSWER,
        TERMINATION_AGENT_COMPLETED,
        TERMINATION_MEDIA_SESSION_LOST,
        TERMINATION_MODEL_SESSION_LOST,
        TERMINATION_LISTENER_UNAVAILABLE,
        TERMINATION_WORKFLOW_LINE_UNDELIVERED,
        "browser_startup_failed",
        TERMINATION_AGENT_UNRESPONSIVE,
        TERMINATION_CALLER_UNRESPONSIVE,
        TERMINATION_IVR_TIMEOUT,
    }
)

# The ones a platform RULE decides, as opposed to the agent finishing its own
# conversation or the media session going away. Kept apart because that
# distinction is the whole point of ``agent_completed``: a customer reading
# "why did this call end" needs to know whether something ended their call or
# the call simply finished. ``media_session_lost`` is absent for the same
# reason from the other side -- no rule ran, the session was observed to go.
# This set is what the polled rule ladder may return, so a value that is not
# produced by that ladder must not be added here.
WORKER_RULE_TERMINATIONS = frozenset(
    {
        TERMINATION_HANGUP_REQUESTED,
        TERMINATION_REMOTE_PARTY_DISCONNECTED,
        TERMINATION_MAX_CALL_DURATION_REACHED,
        TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER,
        # ``no_human_after_answer`` is deliberately absent: this set is what
        # the POLLED LADDER may return, and the arbiter decides that ending by
        # its own path to the same ``SessionEndArbiter.end`` seam.
    }
)


# LiveKit's own server-side ceiling, set by the outbound worker as
# ``SKYSAY_OUTBOUND_MAX_CALL_DURATION_SECONDS``. It kills the SIP leg with
# no reason attached, so the worker-owned limit has to fire BEFORE it or the
# ending is recorded as nothing at all.
LIVEKIT_HARD_KILL_SECONDS = 1_800.0
# One minute of headroom: enough for the worker to end the session, report
# usage, settle and stamp its reason before the reason-less kill lands.
WORKER_MAX_CALL_HEADROOM_SECONDS = 60.0
WORKER_MAX_CALL_SECONDS_DEFAULT = LIVEKIT_HARD_KILL_SECONDS - WORKER_MAX_CALL_HEADROOM_SECONDS
WORKER_NO_CALLER_SPEECH_SECONDS_DEFAULT = 20.0
# The witness is voice activity (the VAD, plus the engine's own captions on a
# GPT-Live call), which cannot tell a caller who is LISTENING from one who is
# not there. Below this an agent's own greeting outlives the rule.
WORKER_NO_CALLER_SPEECH_FLOOR_SECONDS = 10.0
# The room must be continuously empty for this long, not merely empty once: a
# single observation can catch a participant re-publishing or the instant
# between one leg leaving and its replacement joining.
WORKER_OCCUPANCY_GRACE_SECONDS = 15.0


def remaining_call_parties(
    participant_identities: Iterable[str],
    *,
    agent_identity: str,
    evidence_identity: str,
) -> tuple[str, ...]:
    """Identities in the room that are neither this worker nor its recorder.

    Deliberately NOT prefix matching.  ``reconcile_sessions`` recognises an
    agent by ``identity.startswith("agent-")``, which is a guess about a string
    the SDK owns and is blind to the evidence sidecar; both identities are in
    hand at the call site, so this compares them exactly.  A carrier-minted SIP
    identity that happened to start with ``agent-`` would otherwise make an
    occupied room look empty and hang up on a live caller.
    """

    known = {agent_identity, evidence_identity}
    return tuple(
        identity
        for identity in participant_identities
        if str(identity) and str(identity) not in known
    )


@dataclass(frozen=True)
class CallEndSnapshot:
    """What the watchdog can see about a call at one instant.

    ``seconds_room_empty`` is ``None`` for BOTH "somebody is still here" and
    "the server could not be asked", deliberately.  The two differ in what the
    worker should log, never in what it should do: neither is evidence a call
    is over, and collapsing them here makes it impossible to end a call on a
    probe failure by mistake.
    """

    answered: bool
    seconds_since_answer: float
    caller_has_spoken: bool
    hangup_requested: bool = False
    seconds_room_empty: float | None = None
    # ``None`` means "this caller does not know about the release clock",
    # not "the input was never released" -- worker_call_end_reason falls
    # back to seconds_since_answer in that case, so every existing
    # constructor of this snapshot keeps its exact prior behaviour. The real
    # entrypoint always passes a value: SIP answer evidence and the moment
    # this worker actually re-enables the engine's input are two different
    # instants, and a caller who answered but fell silent in that gap must
    # not be judged by a clock that started before anything they said could
    # possibly have been heard.
    seconds_since_input_released: float | None = None
    # The call is waiting in a phone menu or a hold queue (the navigation
    # state). Only the budgets and the call limit end it then; the silence
    # rule below stands down. ``False`` for every call whose agent has not
    # opted in, which is why every existing constructor keeps its behaviour.
    navigation_suspended: bool = False
    # Seconds since the call last returned from a menu or a hold to ordinary
    # conversation, or ``None`` if it never was in one. The silence rule is
    # measured from the LATER of this and the input release: a caller who was
    # on hold for ten minutes has not been silent for ten minutes at the moment
    # a person picks up.
    seconds_since_navigation_exit: float | None = None


CALLER_SPEECH_WITNESS_VAD = "vad"
CALLER_SPEECH_WITNESS_GPT_LIVE_CAPTIONS = "gpt_live_captions"
CALLER_SPEECH_WITNESS_ENGINE_TURN = "engine_turn"


class CallerSpeechWitness:
    """Whether the other party has been heard at all, for the silence rule.

    The VAD start-of-speech edge (``user_state`` -> ``speaking``) is the witness
    on every call, and on the cascade stack it is the only one: the
    caller-transcript gate withholds STT events while a reply is in flight, so
    a transcript-backed check there would read the agent's own opener as
    caller silence and hang up on someone who was talking.

    On a GPT-Live call the engine's own live captions are a second witness.
    Two facts from the first customer batch (2026-09-15) make that necessary
    rather than a convenience. The Silero v6.1 checkpoint pinned at the time
    (replaced on 2026-09-15 by the plugin-bundled one, which scored the same
    leg at 0.95) peaked at a probability of 0.33 on one caller's narrowband
    PSTN leg, so the VAD never crossed its 0.5 threshold in a call where the
    engine transcribed the caller three times; and the SDK deliberately does
    not move ``user_state`` from the realtime session's own input-speech event
    while a client VAD is configured (``agent_activity.py``
    ``_on_input_speech_started``), so those
    captions could not reach the rule by any existing path. GPT-Live has no
    STT node and therefore no transcript gate, so nothing withholds its
    captions and the cascade objection above does not apply.

    A caption is evidence only when it carries text. The one event this emits
    is the log line the batch was missing: which witness fired, and when.

    The third witness is the engine's own turn detection, on an engine that
    has one (``ENGINE_REPORTS_CALLER_TURNS``): a generation the engine began on
    its own, after the caller's audio was released to it and not as its
    automatic reply to a tool result, is the server having heard the caller
    -- the same caller the VAD may have scored under threshold and the
    captions may have missed. The turn recorder decides what counts as such a
    generation; this witness only records that one was reported.
    """

    def __init__(
        self,
        *,
        captions_are_evidence: bool,
        engine_turns_are_evidence: bool = False,
        emit: Callable[..., None] = lambda *_args, **_details: None,
        seconds_since_answer: Callable[[], float | None] = lambda: None,
    ) -> None:
        self._captions_are_evidence = bool(captions_are_evidence)
        self._engine_turns_are_evidence = bool(engine_turns_are_evidence)
        self._emit = emit
        self._seconds_since_answer = seconds_since_answer
        self._source = ""

    @property
    def heard(self) -> bool:
        return bool(self._source)

    @property
    def source(self) -> str:
        """``vad``, ``gpt_live_captions``, ``engine_turn`` or ``""`` while nothing has been heard."""

        return self._source

    def on_engine_turn(self) -> None:
        """The engine began a turn of its own after hearing the caller."""
        if not self._engine_turns_are_evidence:
            return
        self._witness(CALLER_SPEECH_WITNESS_ENGINE_TURN)

    def on_user_state_changed(self, new_state: Any) -> None:
        if str(new_state) == "speaking":
            self._witness(CALLER_SPEECH_WITNESS_VAD)

    def on_user_input_transcribed(self, transcript: Any) -> None:
        if not self._captions_are_evidence:
            return
        if not str(transcript or "").strip():
            return
        self._witness(CALLER_SPEECH_WITNESS_GPT_LIVE_CAPTIONS)

    def _witness(self, source: str) -> None:
        if self._source:
            return
        self._source = source
        try:
            since_answer = self._seconds_since_answer()
        except Exception:
            since_answer = None
        try:
            self._emit(
                "caller_speech_witnessed",
                source=source,
                seconds_since_answer=(
                    None
                    if since_answer is None
                    else round(max(0.0, float(since_answer)), 3)
                ),
            )
        except Exception:
            # Evidence only; the rule's own state is already recorded above.
            pass


#: Deadline for the bind row when no caller frame has reached the session.
CALLER_AUDIO_BIND_DEADLINE_SECONDS = 3.0


@dataclass(frozen=True)
class CallerAudioBinding:
    """Who RoomIO linked, and whether that link carries a microphone track."""

    linked_identity: str
    publication_bound: bool


def caller_audio_binding_from_room_io(
    room_io: Any, *, microphone_source: Any
) -> CallerAudioBinding:
    """Read RoomIO's caller linkage off its public surface. SDK-free, so tested.

    ``publication_source`` is the ADOPTED publication's source: the SDK returns
    ``SOURCE_UNKNOWN`` when none has been adopted, and only adopts one whose
    participant identity equals the identity the stream was set to.
    """

    linked = room_io.linked_participant
    return CallerAudioBinding(
        linked_identity=str(getattr(linked, "identity", "") or ""),
        publication_bound=room_io.audio_input.publication_source == microphone_source,
    )


def caller_audio_subscriber(
    *, caller_identity: str, audio_kind: Any
) -> Callable[[Any, Any], bool]:
    """Subscribe the caller's audio publications, whenever they appear.

    The worker connects with ``auto_subscribe=SUBSCRIBE_NONE``, so nothing in
    the room is subscribed unless this product asks for it, and the agent
    session's input adopts a track only in reaction to ``track_subscribed``.
    Asking ONCE, at worker entry, is the measured defect: on an outbound SIP
    call the caller's audio publication does not exist yet at that moment (it
    arrived 0.51s later, a quarter-second after ``session.start`` had already
    returned), the one-shot sweep found nothing, and the publication then sat
    ``subscribed: false`` for the life of the call. Every downstream consumer
    -- flow counter, VAD, STT, the answer classifier -- read zero from one
    unsubscribed track.

    So the question "is this a publication this call must hear" is asked of
    each publication as it arrives, not of the room once. The returned callable
    carries LiveKit's ``track_published`` signature ``(publication,
    participant)`` and is also applied to the publications that already exist,
    so the window before it is armed is closed by the same predicate.

    It returns True only when it ISSUED a subscription, so a caller can record
    exactly what it asked for. Deliberately narrow: the linked caller's
    identity and audio only. The evidence sidecar and the agent's own output
    track are in the same room and must stay unsubscribed, and a publication
    that is already subscribed is left alone so re-sweeping is free.
    """

    identity = str(caller_identity or "")

    def subscribe(publication: Any, participant: Any) -> bool:
        if not identity:
            return False
        if str(getattr(participant, "identity", "") or "") != identity:
            return False
        if getattr(publication, "kind", None) != audio_kind:
            return False
        if bool(getattr(publication, "subscribed", False)):
            return False
        publication.set_subscribed(True)
        return True

    return subscribe


def wire_caller_audio_subscription(
    room: Any,
    participant: Any,
    *,
    audio_kind: Any,
    note: Callable[[str, list[str]], None],
    late_publications: bool,
) -> None:
    """Arm the late-publication handler, then sweep what is already there.

    Both halves live here rather than in the worker so the handler itself can
    be DRIVEN by a test. Asserting that a ``track_published`` handler was
    registered proves nothing about what it does: replacing its body with
    ``pass`` would leave such a test green while every late caller track went
    unsubscribed again (Codex, round 1).

    The order is load-bearing. Registering first and sweeping second leaves no
    window; sweeping first would miss a publication landing between the two.

    Subscribing BEFORE the agent session starts is safe, and this must not be
    "fixed" by moving the wiring after ``session.start``. A reviewer read it
    as a race -- subscribe early and RoomIO misses its only
    ``track_subscribed`` event -- but ``_ParticipantInputStream`` does not
    rely on that event alone: ``set_participant`` scans the linked
    participant's publications and adopts any whose ``track`` is already set.
    A workspace browser test has proved it on every call since long before
    this function existed: its caller's microphone is subscribed at worker
    entry, seconds before the session starts, and the agent hears it. Moving
    the wiring later would also push it out of the pre-media guard.

    ``late_publications`` is false only for a human workspace browser test,
    whose caller is already publishing when the worker reaches this point:
    there the sweep is the whole story and the behaviour is exactly what it was
    before. A synthetic agent caller publishes after it joins, so it is armed
    like a phone call.
    ``note`` receives the trigger and the track sids actually subscribed, so
    an empty round reports nothing.
    """

    subscribe = caller_audio_subscriber(
        caller_identity=str(getattr(participant, "identity", "") or ""),
        audio_kind=audio_kind,
    )

    def sid(publication: Any) -> str:
        return str(getattr(publication, "sid", "") or "")

    def announce(trigger: str, track_sids: list[str]) -> None:
        # An empty round is not an event: the sweep finding nothing is the
        # ordinary outbound case, not something to report.
        if track_sids:
            note(trigger, track_sids)

    def on_track_published(publication: Any, who: Any) -> None:
        if subscribe(publication, who):
            announce("track_published", [sid(publication)])

    if late_publications:
        room.on("track_published", on_track_published)
    announce(
        "worker_entry_sweep",
        [
            sid(publication)
            for publication in list(
                getattr(participant, "track_publications", {}).values()
            )
            if subscribe(publication, participant)
        ],
    )


class CallerAudioBindWitness:
    """Was the caller's audio bound to this session -- reported at most once.

    ``bound`` is the binding, not audio flow: RoomIO linked the identity this
    worker asked for AND that link carries a microphone publication. An
    unreadable binding reports ``bound`` false, never silence. ``frames_seen``
    is counted where caller audio enters the worker's STT seam, so the two are
    independent and separate a binding fault from a no-RTP one. Emitted once
    the session has started, at the first caller frame or at the deadline. A
    call that never answered, or that ended before either, gets no row: the
    deaf call this exists for runs to the silence rule's twenty seconds.
    """

    def __init__(
        self,
        *,
        expected_identity: str,
        emit: Callable[..., None] = lambda *_args, **_details: None,
        seconds_since_answer: Callable[[], float | None] = lambda: None,
    ) -> None:
        self._expected_identity = str(expected_identity or "")
        self._emit = emit
        self._seconds_since_answer = seconds_since_answer
        self._frames_seen = 0
        self._reported = False
        self._session_started = False
        self._session_started_after: float | None = None

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def note_caller_frame(self) -> None:
        """One caller audio frame reached this session's STT seam."""

        self._frames_seen += 1

    def note_session_started(self) -> None:
        """``session.start`` has returned, so there is a binding to ask about.

        The cadence is armed long before that await and the answer can land
        during it, so the deadline is the later of answer and session start.
        """

        if self._session_started:
            return
        self._session_started = True
        self._session_started_after = self._seconds_since_answer()

    def report(self, binding: CallerAudioBinding | None) -> bool:
        """Emit the row if it is due; say whether this call emitted it."""

        if self._reported:
            return False
        since_answer = self._seconds_since_answer()
        if since_answer is None or not self._due(since_answer):
            return False
        linked_identity = ""
        publication_bound = False
        if binding is not None:
            linked_identity = str(binding.linked_identity or "")
            publication_bound = bool(binding.publication_bound)
        bound = bool(
            self._expected_identity
            and linked_identity == self._expected_identity
            and publication_bound
        )
        self._emit(
            "caller_audio_input_bound",
            bound=bound,
            linked_identity=linked_identity or None,
            expected_identity=self._expected_identity,
            frames_seen=self._frames_seen,
            seconds_since_answer=round(max(0.0, float(since_answer)), 3),
        )
        # Latched after the row is out, so a sink that raises does not lose it.
        self._reported = True
        return True

    def _due(self, since_answer: float) -> bool:
        """Is the row due yet?  Never before the session started; then a
        caller frame is proof and needs no clock.
        """

        if not self._session_started:
            return False
        if self._frames_seen > 0:
            return True
        if since_answer < CALLER_AUDIO_BIND_DEADLINE_SECONDS:
            return False
        started_after = self._session_started_after
        return (
            since_answer - (started_after or 0.0) >= CALLER_AUDIO_BIND_DEADLINE_SECONDS
        )


async def counting_caller_audio(
    audio: AsyncIterable[Any],
    witness: CallerAudioBindWitness,
    flow: Any | None = None,
) -> AsyncIterator[Any]:
    """Pass caller audio through untouched, counting what goes by.

    Two recorders, one pass over the frames, because they answer different
    questions from the same seam: the witness answers "did caller audio ever
    reach this session" ONCE, near the start of the call; ``flow`` answers
    "is it still reaching it" for the whole call.  A silent-call investigation
    needed the second and only the first existed.

    ``flow`` is structurally typed (anything with ``note_caller_frame()``) on
    purpose: this module is the leaf the other voice modules import, and
    naming ``CallerAudioFlowLog`` here would invert that direction.
    """

    async for frame in audio:
        witness.note_caller_frame()
        if flow is not None:
            flow.note_caller_frame()
        yield frame


def worker_call_end_reason(
    snapshot: CallEndSnapshot,
    *,
    max_call_seconds: float | None,
    no_caller_speech_seconds: float | None,
    occupancy_grace_seconds: float = WORKER_OCCUPANCY_GRACE_SECONDS,
) -> str | None:
    """Which of the four endings applies right now, or ``None`` to keep going.

    One place decides, so the four inputs cannot each end the same call for a
    different reason, and the precedence is testable without a room.  An
    operator's explicit request outranks everything: they are looking at the
    call and this worker is not.

    Nothing below the answer gate is meaningful before the SIP leg joins.  An
    unanswered room is EMPTY by construction -- the callee has not arrived --
    so an ungated occupancy rule would hang up on every outbound call while it
    was still ringing, and both timers would measure from a moment the caller
    was not yet on the line.

    Occupancy needs the room to have been continuously empty for the grace
    period, not merely empty once.  A single observation can catch a legitimate
    gap: a participant re-publishing, a brief server-side inconsistency, the
    instant between one leg leaving and its replacement joining.  The duration
    timer is what covers a room the server cannot be asked about at all.
    """

    if snapshot.hangup_requested:
        return TERMINATION_HANGUP_REQUESTED
    if not snapshot.answered:
        return None
    if (
        snapshot.seconds_room_empty is not None
        and snapshot.seconds_room_empty >= occupancy_grace_seconds
    ):
        return TERMINATION_REMOTE_PARTY_DISCONNECTED
    if max_call_seconds is not None and snapshot.seconds_since_answer >= max_call_seconds:
        return TERMINATION_MAX_CALL_DURATION_REACHED
    # The SIP leg can answer before this worker's engine can hear it -- the
    # SIP answer gate resolving and this worker re-enabling the engine's
    # input are two different instants. A caller who spoke into that gap
    # must be judged from when we could actually have heard them, not from
    # when the carrier says the leg went up. Callers of this function that
    # never learned about the release clock keep exactly their prior
    # behaviour via the fallback.
    since_released = (
        snapshot.seconds_since_input_released
        if snapshot.seconds_since_input_released is not None
        else snapshot.seconds_since_answer
    )
    if snapshot.seconds_since_navigation_exit is not None:
        since_released = min(since_released, snapshot.seconds_since_navigation_exit)
    if (
        no_caller_speech_seconds is not None
        and not snapshot.navigation_suspended
        and not snapshot.caller_has_spoken
        and since_released >= no_caller_speech_seconds
    ):
        return TERMINATION_NO_CALLER_SPEECH_AFTER_ANSWER
    return None


def navigation_adjusted_ending(
    reason: str | None,
    navigation: NavigationPolicy | None,
    *,
    now: float,
) -> tuple[str | None, str]:
    """The polled ladder's ending after the navigation state has been read.

    ``(reason, reason_detail)``. A call that is not in a menu or a hold, or
    whose agent did not opt in, passes through unchanged. While it is waiting:

    * the overall call limit is the policy's ONE duration decision
      (``duration_ending``): ``ivr_timeout`` / ``deadline`` instead of
      ``max_call_duration_reached``, so the customer is told the call ran out
      of time while waiting. The funded-runtime timer asks the same function;
    * with no other ending due, a spent navigation or hold budget ends it;
    * a remote hang-up in any non-conversation state keeps its reason and
      gains the ``during_hold`` detail the retry policy reads;
    * every other ending (an operator's hang-up, a lost media session)
      passes through, because nothing about a hold outranks them.
    """

    if navigation is None or not navigation.enabled:
        return reason, ""
    if reason == TERMINATION_MAX_CALL_DURATION_REACHED:
        return navigation.duration_ending()
    if reason == TERMINATION_REMOTE_PARTY_DISCONNECTED:
        return reason, navigation.remote_hangup_detail()
    if reason is None:
        limit = navigation.limit(now)
        if limit:
            return TERMINATION_IVR_TIMEOUT, limit
    return reason, ""


@dataclass(eq=False)
class SessionEndArbiter:
    """The single point where this worker decides a call is over.

    Four inputs can each conclude the call should end, and the incident this
    exists for proved they are not redundant: the caller's disconnect event
    never arrived at all across 94 minutes, so a predicate watching for it
    could not have saved that call.  First reason wins and the rest are
    no-ops, because ``finish`` must run once -- a second call would race a
    teardown that is already reporting usage and settling the wallet.

    The reason is kept so the terminal lifecycle report can stamp it.  Today
    ``report_call_teardown_lifecycle`` sends no details at all, so this is the
    first writer of a reason on an ordinary ``sip.ended``.

    The decision is also PERSISTED here, at the instant it is made rather than
    at teardown, when ``state_path`` is given.  The incident of 2026-09-08
    decided ``remote_party_disconnected`` and the customer's row settled with an
    empty reason: the decision was made in a disconnect callback and the only
    thing that reported it lived in a teardown that never ran.  Writing it here
    means it outlives the teardown, and the crash reconciler -- which reads that
    file precisely when teardown did not happen -- can report what this worker
    actually decided instead of a generic substitute.

    Persisting in ``end`` rather than in the entrypoint's ``finish`` callback is
    what makes it complete.  ``end`` is the single point every one of the four
    inputs passes through, and it is the ONLY place that knows about the
    release: a stamped reason whose ``finish`` raises belongs to an ending that
    did not happen, and must be deleted rather than left on disk.
    """

    finish: Callable[[], None]
    emit: Callable[..., None]
    #: Where this worker's decision is written, or ``None`` to keep it in memory
    #: only.  A path rather than a pair of callbacks on purpose: the real
    #: writer's merge, write-once and blank-guard rules are the contract being
    #: relied on, so a test that passes a real path exercises them, while a stub
    #: would be free to be kinder than the file ever is.
    state_path: Path | None = None
    #: The clock, injected by the entrypoint, which owns the one timestamp
    #: format every lifecycle event on this call is stamped with; a second clock
    #: here would be a second format to keep in step.
    now: Callable[[], str] | None = None
    _reason: str = ""
    _reason_detail: str = ""
    _detected_via: str = ""
    _decided_at: str = ""
    #: Read ONCE, inside :meth:`end`, at the instant the reason is decided, and
    #: sealed with it. ``None`` from the source (the agent does not navigate)
    #: leaves the key out entirely: "not reported" is a third state the control
    #: plane keeps distinct from ``False``.
    prior_person_source: Callable[[], bool | None] | None = None
    _prior_person: bool | None = None

    @property
    def reason(self) -> str:
        """The stamped ending, or ``""`` when this worker did not end the call."""

        return self._reason

    @property
    def reason_detail(self) -> str:
        """What qualifies the stamped ending, or ``""``. Customer-facing."""

        return self._reason_detail

    @property
    def prior_person(self) -> bool | None:
        """Whether a person was reached before the ending, as sealed with it, or ``None``."""

        return self._prior_person

    @property
    def detected_via(self) -> str:
        """How it was noticed. Internal provenance, never a customer string."""

        return self._detected_via

    @property
    def decided_at(self) -> str:
        """When this worker decided the call was over, or ``""``.

        The instant of the DECISION, which is not the instant teardown got round
        to reporting it.  Those were the same instant until a call parked above
        its own teardown for fifty-seven minutes, and the reconciler prefers
        this over anything it can observe for itself: its own first sighting is
        a fallback for a worker that never persisted one, and the heartbeat is
        worse than useless here because on that failure mode it stays fresh for
        half an hour after the call is over.
        """

        return self._decided_at

    def _record_decision(self) -> None:
        """Put the decision on disk. Best effort, and deliberately so.

        A failure here must not propagate: the caller treats an exception out of
        ``finish`` as "the call did NOT end", and a full disk must not be able to
        un-decide an ending and leave a live call with no rule watching it.  The
        reason still reaches the control plane by the terminal lifecycle report
        on any teardown that runs; only the crash backstop degrades, back to the
        generic reason it used before this was written at all.

        ``decided_at`` is omitted rather than blanked when there is no clock, so
        the writer's blank-guard is never asked to arbitrate a value this had no
        way to produce.
        """

        if self.state_path is None:
            return
        stamp: dict[str, Any] = {SESSION_REASON_FIELD: self._reason}
        if self._decided_at:
            stamp[SESSION_DECIDED_AT_FIELD] = self._decided_at
        if self._reason_detail:
            # The detail is what a retry policy branches on (``ivr_timeout``
            # with ``hold_limit`` is redialled, with ``navigation_limit`` it is
            # not): a crash that loses it must not turn a retry into a dead end.
            stamp[SESSION_REASON_DETAIL_FIELD] = self._reason_detail
        if self._prior_person is not None:
            # Same argument: a crash that loses it turns "a person was reached,
            # then the queue ran out" back into a retry.
            stamp[SESSION_PRIOR_PERSON_FIELD] = self._prior_person
        try:
            write_session_state(self.state_path, **stamp)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self._report_state_error("session_decision_not_persisted", exc)

    def _release_decision(self) -> None:
        """Delete a decision this worker has just disowned.

        The reason on disk is WRITE-ONCE, so without this a reason stamped for
        an ending that then failed would sit there permanently, immune to
        correction by the real ending that follows.  Write-once protects a
        reason and this removes one that was disowned; neither is safe alone.

        One call clearing both keys, so a release cannot half-happen and leave
        an instant belonging to no reason.

        VERIFIED and RETRIED, unlike the write. Failing open is not symmetric
        between the two: a write that fails open loses an annotation, and a
        delete that fails open freezes a LIE -- the reason is write-once, so a
        clear that silently did not happen means the real ending which follows
        can never correct it, and the customer is told their call ended for
        something that did not occur. A single swallowed error was therefore
        permanent, which is the one outcome this function exists to prevent.

        Still never raises. This runs on the way out of a failed ``finish`` and
        an exception escaping would replace the failure that actually matters
        with a bookkeeping one. Exhausting the retries is reported instead, and
        it is deliberately loud: at that point a wrong reason IS on disk and
        only an operator can tell.
        """

        if self.state_path is None:
            return
        last: BaseException | None = None
        for _ in range(RELEASE_ATTEMPTS):
            # Clear AND confirm under ONE lock hold. Confirming in a second hold
            # left a window a heartbeat could land in, re-publishing the reason
            # it had read before the clear.
            try:
                if clear_session_state_keys_confirmed(
                    self.state_path,
                    SESSION_REASON_FIELD,
                    SESSION_DECIDED_AT_FIELD,
                    SESSION_REASON_DETAIL_FIELD,
                    SESSION_PRIOR_PERSON_FIELD,
                ):
                    return
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last = exc
                continue
        self._report_state_error(
            "session_decision_release_unconfirmed",
            last or RuntimeError("released reason still present on disk"),
        )

    def _read_prior_person(self) -> bool | None:
        """The source's answer, only if it is a real boolean. Never raises: an
        unreadable source is "not reported", and nothing here may stop an ending."""

        if self.prior_person_source is None:
            return None
        try:
            value = self.prior_person_source()
        except Exception:  # noqa: BLE001 - a reading is never a reason to fail an ending
            return None
        return value if isinstance(value, bool) else None

    def _report_state_error(self, event: str, exc: BaseException) -> None:
        """Say a state write failed, without letting the saying of it fail."""

        try:
            self.emit(event, error_type=type(exc).__name__)
        except Exception:  # noqa: BLE001 - diagnostics never mask the real path
            pass

    def end(self, reason: str, *, detected_via: str, reason_detail: str = "") -> bool:
        """End the call for ``reason``; False when something already ended it.

        ``reason_detail`` qualifies the reason for a customer -- two paths can
        reach one ending and mean different things by it. It is validated
        against the pairing for THIS reason, not against a flat set, so a
        detail can never be stamped beside an ending it says nothing about.
        """

        if detected_via not in WORKER_DETECTION_PATHS:
            raise ValueError(f"unknown worker detection path: {detected_via!r}")
        if reason not in WORKER_TERMINATION_DETAILS:
            # A string the control plane's closed enum cannot serve would reach
            # the customer as an empty reason, which is worse than loud failure
            # here: the call would end and nobody could say why.
            raise ValueError(f"unknown worker termination detail: {reason!r}")
        if reason_detail and reason_detail not in TERMINATION_REASON_DETAILS.get(
            reason, frozenset()
        ):
            raise ValueError(
                f"detail {reason_detail!r} does not qualify ending {reason!r}"
            )
        if self._reason:
            return False
        self._reason = reason
        self._reason_detail = reason_detail
        self._detected_via = detected_via
        self._decided_at = self.now() if self.now is not None else ""
        self._prior_person = self._read_prior_person()
        # BEFORE the ending becomes observable to anything else. The whole point
        # of this file is that it survives a process that never reaches its own
        # teardown, so the write has to happen while this worker is still
        # certainly running -- not after handing control to a callback that may
        # park forever, which is exactly what the incident did.
        self._record_decision()
        # EVERYTHING after the persist is inside the release, not just
        # ``finish``. The reason on disk belongs to an ending that has not
        # happened yet, so any failure between stamping it and completing the
        # ending must give it back -- and it was ``emit`` that proved the point:
        # a diagnostic call, wrapped in nothing, sitting between the write and
        # the try. If it raised, the ending never completed and the reason was
        # frozen by write-once for a teardown that never ran.
        #
        # This is the argument this file's own writer already makes about the
        # persist: "true today and one edit away from being false, and it would
        # fail silently". The same objection applied here and was not acted on,
        # so the guard now covers the region rather than one call in it.
        try:
            self.emit(
                "worker_ended_call",
                termination_detail=reason,
                detected_via=detected_via,
            )
            self.finish()
        except BaseException:
            # The call did NOT end. Keeping the reason would be worse than the
            # failure it records: every other guard stands down on seeing one,
            # so the watchdog would go quiet for the rest of a call that is
            # still running -- the unbounded state again, now with a reason
            # already stamped for a teardown that never happened. Released so
            # the next tick can try again.
            self._reason = ""
            self._reason_detail = ""
            self._detected_via = ""
            self._decided_at = ""
            self._prior_person = None
            # And released ON DISK too, in the same breath. Releasing only the
            # in-memory copy would let the next tick stamp a new reason while
            # the disowned one stayed on the file forever, because the reason
            # there is write-once -- so the customer would be told the call
            # ended for something that never happened, and the record would
            # disagree with the ending that actually did.
            self._release_decision()
            raise
        return True


# Long enough that a healthy call pays almost nothing for it, short enough that
# a room nobody is in is not held open for minutes.  The incident ran 94
# minutes; anything in seconds is a different order of failure.
WORKER_CALL_END_POLL_SECONDS = 15.0


def _bounded_worker_seconds(raw: Any, *, low: float, high: float) -> float | None:
    """A configured limit in seconds, or ``None`` when unset or unusable.

    Junk disables the rule rather than guessing at it. A limit decides when to
    hang up on a paying customer, and a value this cannot parse is not a
    number someone chose.
    """

    if raw is None or isinstance(raw, bool):
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return seconds if low <= seconds <= high else None


def _turn_taking_limit(realtime_config: Mapping[str, Any], key: str) -> Any:
    raw = realtime_config.get("turn_taking")
    return raw.get(key) if isinstance(raw, Mapping) else None


def resolved_max_call_seconds(
    realtime_config: Mapping[str, Any],
    *,
    default_seconds: float = WORKER_MAX_CALL_SECONDS_DEFAULT,
    hard_kill_seconds: float = LIVEKIT_HARD_KILL_SECONDS,
) -> tuple[float, bool]:
    """``(effective, clamped)`` maximum call length for this call.

    A per-agent override may only make a call SHORTER. The ceiling is the
    LiveKit hard kill minus its headroom, because a worker limit above that
    never fires: LiveKit would drop the leg first, with no reason attached, and
    the customer would be told nothing about why their call ended. An override
    past the ceiling is clamped rather than refused -- the customer asked for a
    longer call and gets the longest one that can still be explained -- and the
    caller is told it happened so the clamp is never silent.

    The clamp is a defence, not the contract. Whether a customer can SET this
    key, and what the API does with an out-of-range value, is the control
    plane's business and is documented there. Values also reach agent metadata
    by paths that never pass through that boundary, so this stays either way.
    """

    ceiling = hard_kill_seconds - WORKER_MAX_CALL_HEADROOM_SECONDS
    requested = _bounded_worker_seconds(
        _turn_taking_limit(realtime_config, "max_call_seconds"), low=30.0, high=14_400.0
    )
    if requested is None:
        return (min(default_seconds, ceiling), False)
    if requested > ceiling:
        return (ceiling, True)
    return (requested, False)


def build_call_floor_policy(config, *, outbound, language, announce, emit, quiet_emit,
                            policy_type=None, no_caller_speech_seconds=None, navigation_sources=(),
                            navigation_languages=()):
    """Build the shared navigation and answer policy from explicit call metadata.

    ``navigation_sources`` are the call's own transcript sources (``(kind, name, model)``, see
    ``navigation_sockets.untrusted_source_languages``) and ``navigation_languages`` the set the call may hear. A source
    measured to fail in one of those languages would feed navigation outside the detector, so such a call runs
    without navigation, decided here before any watch, detector or action exists.
    """
    try:
        from .navigation_entry import NavigationSettings, navigation_settings, navigation_engine_refused
        from .answer_detection import AnswerDetectionSession, resolved_mode
        from .voicemail_watch import resolved_setting
    except ImportError:
        from navigation_entry import NavigationSettings, navigation_settings, navigation_engine_refused
        from answer_detection import AnswerDetectionSession, resolved_mode
        from voicemail_watch import resolved_setting
    if outbound and navigation_engine_refused(config):
        voice = config.get("voice") if isinstance(config.get("voice"), Mapping) else {}
        quiet_emit("navigation_engine_not_supported", engine=str(
            config.get("voice_frontend") or voice.get("voice_frontend") or "").strip().lower())
    settings = navigation_settings(config) if outbound else NavigationSettings()
    if settings.enabled:
        try:
            from .navigation_entry import navigation_untrusted_sources
        except ImportError:
            from navigation_entry import navigation_untrusted_sources
        heard = tuple(navigation_languages) or (str(language or ""),)
        untrusted = navigation_untrusted_sources(load_registry(), tuple(navigation_sources), heard) if navigation_sources else ()
        if untrusted:
            quiet_emit("navigation_transcriber_unverified", languages=list(untrusted))
            settings = NavigationSettings()
    navigation = (NavigationPolicy if policy_type is None else policy_type).from_settings(settings, emit=announce)
    silence = resolved_no_caller_speech_seconds(config)[0] if no_caller_speech_seconds is None else no_caller_speech_seconds
    answer = AnswerDetectionSession(mode=resolved_mode(config, emit=quiet_emit),
        no_caller_speech_seconds=silence, emit=emit,
        voicemail_mode=resolved_setting(config, emit=quiet_emit) if outbound else "off",
        language=str(language or ""), measure_for_navigation=navigation.enabled)
    navigation.bind_far_end_audio(answer.far_end_audible)
    return navigation, answer


def resolved_no_caller_speech_seconds(
    realtime_config: Mapping[str, Any],
    *,
    default_seconds: float = WORKER_NO_CALLER_SPEECH_SECONDS_DEFAULT,
) -> tuple[float, bool]:
    """``(effective, clamped)`` silence-after-answer limit for this call.

    Raised to the floor rather than honoured below it. The witness is voice
    activity -- the VAD on every call, and on a GPT-Live call also the
    engine's own captions (:class:`CallerSpeechWitness`) -- so a caller who is
    merely listening is indistinguishable from one who is not there, and a
    shorter value would hang up on someone partway through the agent's own
    opening sentence.

    As above, the floor is a defence rather than the contract: what the API
    accepts is the control plane's business.
    """

    requested = _bounded_worker_seconds(
        _turn_taking_limit(realtime_config, "no_caller_speech_seconds"),
        low=1.0,
        high=600.0,
    )
    if requested is None:
        return (default_seconds, False)
    if requested < WORKER_NO_CALLER_SPEECH_FLOOR_SECONDS:
        return (WORKER_NO_CALLER_SPEECH_FLOOR_SECONDS, True)
    return (requested, False)


async def server_side_remaining_parties(
    room_service: Any,
    *,
    room_name: str,
    agent_identity: str,
    evidence_identity: str,
    request_factory: Callable[..., Any],
) -> tuple[str, ...]:
    """Ask the LiveKit SERVER who is in the room, not this worker's own cache.

    This is the whole reason the occupancy check can work. In the incident the
    ``participant_disconnected`` event never reached this worker at all, so the
    SDK's ``room.remote_participants`` still listed a participant whose phone
    had been gone for over an hour. A poll of that local mapping would have
    read the same stale roster the missed event left behind and concluded the
    call was healthy, exactly as the event-driven predicate did.

    The server holds the truth the missed event failed to deliver, and asking
    it is the only occupancy answer that is independent of the media path this
    call lost.
    """

    listed = await room_service.list_participants(request_factory(room=room_name))
    return remaining_call_parties(
        (str(getattr(item, "identity", "") or "") for item in getattr(listed, "participants", ()) or ()),
        agent_identity=agent_identity,
        evidence_identity=evidence_identity,
    )


# The three outcomes the control plane accepts for a hangup acknowledgement.
# `teardown_failed` RE-ARMS the request rather than closing it -- the call is
# still up -- so a further request creates a genuinely new attempt with its own
# number, and three consecutive failures are refused so someone escalates
# instead of looping.
HANGUP_OUTCOME_ROOM_DELETED = "room_deleted"
HANGUP_OUTCOME_ROOM_ABSENT = "room_absent"
HANGUP_OUTCOME_TEARDOWN_FAILED = "teardown_failed"
HANGUP_OUTCOMES = frozenset(
    {
        HANGUP_OUTCOME_ROOM_DELETED,
        HANGUP_OUTCOME_ROOM_ABSENT,
        HANGUP_OUTCOME_TEARDOWN_FAILED,
    }
)


def pending_hangup_intent(response: Any) -> dict[str, Any] | None:
    """The operator's hangup request from a lifecycle response, or ``None``.

    Reads only what it must echo. The attempt number is REQUIRED and must be an
    integer: acknowledging with the wrong one is dropped and returns null, and
    an intent this worker cannot acknowledge is worse than one it never saw --
    it would tear a room down and leave the request open forever.

    Identity is never taken from ``hangup_requested_at``. That timestamp has
    second precision, so a request re-armed within the same second carries an
    identical value and a stale acknowledgement would land on an attempt an
    operator had just created. The integer generation is the marker that holds.
    """

    if not isinstance(response, Mapping):
        return None
    request = response.get("hangup_request")
    if not isinstance(request, Mapping):
        return None
    if str(request.get("status") or "").strip().lower() != "pending":
        return None
    attempt = request.get("hangup_attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        return None
    call_id = str(request.get("call_id") or "").strip()
    if not call_id:
        return None
    return {"call_id": call_id, "attempt": attempt, "pending": True}


def room_delete_says_absent(error: BaseException | None) -> bool:
    """Whether LiveKit's refusal means the room was already gone.

    Shared by the classifier and the emit so the two cannot disagree about
    which failures are benign: a room that is not there is the outcome the
    operator wanted, and logging it as a teardown failure would put a red
    event in the evidence of every ordinary call whose room the reconciler had
    already reaped.
    """

    return str(getattr(error, "code", "") or "").strip().lower() == "not_found"


def adopt_hangup_intent(
    held: MutableMapping[str, Any],
    incoming: Mapping[str, Any] | None,
    *,
    call_id: str,
    emit: Callable[..., None],
) -> bool:
    """Take an operator's hangup intent, unless it is one already being acted on.

    The control plane attaches the SAME pending request to every
    lifecycle-heartbeat response until it is settled, on the novel and the
    duplicate path alike -- and on the hosted platform-originator path the
    acknowledgement is refused outright by the platform-gateway route
    allowlist, so that request arrives
    every few seconds for the rest of the call's lifecycle reporting. Acting
    on each delivery would tear the room down repeatedly, re-post an
    acknowledgement that can only lose, and fill the evidence with one
    ``hangup_intent_received`` per heartbeat. The contract is once: one
    teardown, one acknowledgement, one log line.

    Identity is the integer attempt, never "do we already hold something".
    A request re-armed after a failed teardown is a genuinely new attempt with
    its own number, and it has to be adopted -- acknowledging the superseded
    one would lose the compare-and-set and leave the operator's live request
    pending with nobody acting on it.
    """

    if incoming is None:
        return False
    if str(incoming.get("call_id") or "") != call_id:
        # The request names a call, and this worker holds exactly one. Reading
        # the id and not comparing it left the whole rule trusting the envelope
        # it arrived in: a lookup that ever attached another call's pending
        # request to this heartbeat would tear down a healthy room, and the
        # acknowledgement would then carry this call's id with a foreign
        # attempt -- a guaranteed lost race, so the real request would stay
        # pending too and nobody would end the call anyone asked about.
        emit("hangup_intent_call_mismatch", call_id=call_id)
        return False
    if incoming.get("attempt") == held.get("attempt"):
        return False
    # A new attempt supersedes the old one outright, including its
    # ``acknowledged`` marker: this attempt has not been acknowledged, and the
    # number that was is gone.
    held.clear()
    held.update(incoming)
    return True


def hangup_teardown_outcome(error: BaseException | None) -> tuple[str, str]:
    """``(outcome, reason)`` for a room teardown this worker just attempted.

    A room LiveKit says is not there is a success for the operator's purpose --
    the thing they wanted gone is gone -- so it is reported as ``room_absent``
    and settles the request rather than re-arming it.

    The reason carries the exception CLASS, never its message. Provider
    response text can carry call detail, and this string is shown to an
    operator beside their next request.
    """

    if error is None:
        return (HANGUP_OUTCOME_ROOM_DELETED, "")
    if room_delete_says_absent(error):
        return (HANGUP_OUTCOME_ROOM_ABSENT, "")
    return (HANGUP_OUTCOME_TEARDOWN_FAILED, type(error).__name__)


async def acknowledge_hangup_teardown(
    *,
    call: Any,
    intent: Mapping[str, Any],
    outcome: str,
    reason: str,
    emit: Callable[..., None],
) -> bool | None:
    """Acknowledge one teardown. ``None`` means it was never delivered.

    Three outcomes, and the difference between the last two is what decides
    whether this attempt may be tried again:

    * ``True``  -- the row was ours and is now settled.
    * ``False`` -- a LOST RACE. Definitive: another consumer won it or the
      attempt was superseded, and this number can never match again.
    * ``None``  -- the acknowledgement never reached the control plane. NOT
      definitive: nothing was recorded, so the operator's request is still
      pending and a later attempt should still report it.

    Collapsing the last two into one falsy value is what stranded a request on
    a single transient network error: the caller marked it done, refused the
    same attempt on every later heartbeat, and nobody ever told the operator
    what happened to their room.

    A lost race is not a failure and must never be reported as one: reporting
    ``teardown_failed`` for a race would spend one of the three attempts an
    operator gets before the request is refused outright.

    Never raises into teardown. An acknowledgement that cannot be delivered
    leaves the request pending, which the control plane's own staleness alert
    already covers; taking the call's settlement down with it would not.
    """

    if outcome not in HANGUP_OUTCOMES:
        raise ValueError(f"unknown hangup acknowledgement outcome: {outcome!r}")
    try:
        settled = await asyncio.to_thread(
            call.acknowledge_hangup,
            attempt=int(intent["attempt"]),
            outcome=outcome,
            reason=reason,
        )
    except Exception as exc:  # never BaseException: cancellation propagates
        emit(
            "hangup_acknowledgement_failed",
            attempt=int(intent["attempt"]),
            outcome=outcome,
            error_type=type(exc).__name__,
        )
        # Not delivered, so not answered. Distinct from the lost race below.
        return None
    if settled is None:
        # Another consumer won it, or this attempt was superseded. Do nothing
        # and never retry this number: it can never match again.
        emit("hangup_acknowledgement_lost_race", attempt=int(intent["attempt"]))
        return False
    emit(
        "hangup_acknowledged",
        attempt=int(intent["attempt"]),
        outcome=outcome,
        **({"reason": reason} if reason else {}),
    )
    return True


@dataclass(eq=False)
class RoomTeardownRecord:
    """What actually happened to the room this worker was asked to delete.

    ``attempted`` is tracked apart from ``error`` because the two answer
    different questions, and a teardown can fail before the delete is ever
    reached: the control-plane cleanup that runs first can raise.  Calling that
    ``room_deleted`` merely because no delete error was recorded would tell an
    operator their call is down while nothing had yet tried to take it down.
    """

    attempted: bool = False
    error: BaseException | None = None


async def delete_room_observed(
    record: RoomTeardownRecord,
    *,
    room_service: Any,
    room_name: str,
    request_factory: Callable[..., Any],
    emit: Callable[..., None],
) -> None:
    """Delete the room through the LiveKit SERVER, recording what happened.

    ``JobContext.delete_room`` cannot be used for this.  It swallows every
    failure it meets -- a disconnected server, a Twirp error, anything
    unexpected -- logs a warning and returns as though the room were gone
    (livekit-agents 1.7.1, ``job.py`` lines 664-687).  For an ordinary call
    that is survivable: nothing is waiting on the answer and LiveKit reaps the
    room eventually.  For an operator's hangup it is not.  The acknowledgement
    has to say WHICH of the three outcomes happened, and a swallowed failure is
    reported as ``room_deleted`` -- telling the operator the call is down while
    it is still up and still billing, settling their request, and denying the
    control plane the ``teardown_failed`` its re-arm and its three-strikes
    escalation exist for.

    Asking the server directly is the same move the occupancy probe makes, for
    the same reason: the server holds the answer, and this worker's own
    convenience wrapper does not.

    Never raises a delete failure.  A room that will not go away must not also
    change the outcome of the job or skip the settlement that follows it, so
    the failure is recorded to be classified and reported instead -- which is
    what the SDK wrapper was doing too, minus the record.
    """

    record.attempted = True
    try:
        await room_service.delete_room(request_factory(room=room_name))
    except Exception as exc:  # never BaseException: cancellation propagates
        record.error = exc
        # A room that is already gone is the outcome the caller wanted, so it
        # is recorded but not reported as a failure. The SDK wrapper this
        # replaces was silent on NOT_FOUND for the same reason, and every
        # ordinary call whose room the reconciler reaped first would otherwise
        # carry a red teardown event.
        emit(
            "room_already_absent"
            if room_delete_says_absent(exc)
            else "room_teardown_failed",
            error_type=type(exc).__name__,
        )


def hangup_teardown_record_outcome(record: RoomTeardownRecord) -> tuple[str, str]:
    """``(outcome, reason)`` for a recorded teardown, attempted or not.

    A teardown that never reached the delete is a failure rather than a
    success: the room is still there.  It carries its own label instead of an
    exception class when there is no exception to name, because the step simply
    did not run.
    """

    if not record.attempted and record.error is None:
        return (HANGUP_OUTCOME_TEARDOWN_FAILED, "teardown_not_attempted")
    if not record.attempted:
        return (HANGUP_OUTCOME_TEARDOWN_FAILED, type(record.error).__name__)
    return hangup_teardown_outcome(record.error)


async def acknowledge_pending_hangup(
    *,
    call: Any,
    intent: MutableMapping[str, Any],
    record: RoomTeardownRecord,
    emit: Callable[..., None],
) -> bool:
    """Acknowledge an operator's hangup once the teardown has been attempted.

    A no-op when no operator asked.  Once only: this runs from the teardown's
    ``finally`` and the entrypoint reaches that teardown by more than one path,
    and a second acknowledgement would carry an attempt number that can never
    match again, scoring a lost race against a request that was already
    settled.
    """

    if not intent or intent.get("acknowledged"):
        return False
    intent["acknowledged"] = True
    outcome, reason = hangup_teardown_record_outcome(record)
    try:
        settled = await acknowledge_hangup_teardown(
            call=call, intent=intent, outcome=outcome, reason=reason, emit=emit
        )
    except BaseException:
        # Nothing was settled.  A delivery failure is caught and reported
        # inside, so reaching here means cancellation cut the acknowledgement
        # off mid-flight -- and holding the marker would make a later teardown
        # skip the report entirely, leaving the operator's request pending with
        # no record of what happened to their room.  Set before the attempt so
        # a DEFINITIVE answer, settled or lost race, is never sent twice;
        # released only when there was no answer at all.
        intent.pop("acknowledged", None)
        raise
    if settled is None:
        # Never delivered. The marker means "this attempt has had its
        # definitive answer", and a transport error is not one: the control
        # plane recorded nothing, the operator's request is still pending, and
        # holding the marker would make every later chance to report it a
        # no-op. Released so a retry can still say what happened -- and a retry
        # that is somehow redundant degrades to a harmless lost race.
        intent.pop("acknowledged", None)
        return False
    return settled


async def teardown_room_and_acknowledge_hangup(
    *,
    call: Any,
    room_service: Any,
    room_name: str,
    request_factory: Callable[..., Any],
    intent: MutableMapping[str, Any],
    emit: Callable[..., None],
    session_cleanup: bool = True,
) -> None:
    """Delete this call's room, then say what happened to it.

    One function because the two halves are one obligation: an operator asked
    for a call to end, and the answer they are given has to describe the
    teardown that actually ran.

    The acknowledgement runs from ``finally``, which is the whole point.  The
    outcomes worth reporting are exactly the ones that go wrong, and every way
    a teardown goes wrong leaves by an exception -- so an acknowledgement
    written after the teardown can only ever report success.  That was the
    defect: three outcomes were defined, tested and documented, and one of them
    was reachable.
    """

    record = RoomTeardownRecord()

    async def delete() -> None:
        await delete_room_observed(
            record,
            room_service=room_service,
            room_name=room_name,
            request_factory=request_factory,
            emit=emit,
        )

    try:
        await cleanup_livekit_room(call, delete, session_cleanup=session_cleanup)
    except BaseException as exc:
        if record.error is None:
            # Whatever stopped the teardown, the room is still up.  This
            # catches two shapes: the control-plane cleanup failing before the
            # delete could run, and the delete itself being cancelled
            # mid-flight -- ``delete_room_observed`` deliberately lets a
            # BaseException through, so without this the record would read
            # "attempted, no error" and an unfinished delete would be
            # acknowledged as a deletion.
            record.error = exc
        raise
    finally:
        await acknowledge_pending_hangup(
            call=call, intent=intent, record=record, emit=emit
        )


async def drop_carrier_leg_then_continue(
    hangup: Callable[[], Awaitable[None]],
    *,
    on_failure: Callable[[BaseException], None],
) -> None:
    """Drop the SIP leg without letting a teardown error skip settlement.

    Usage, mix, lifecycle and artifacts do not need the room. A failed
    ``DeleteRoom`` must still be retried from except/finally; it must not
    abort the reports that settle the caller's reservation.
    ``CancelledError`` is not an ``Exception`` and still propagates.
    """

    try:
        await hangup()
    except Exception as exc:
        on_failure(exc)


async def watch_for_worker_call_end(
    *,
    arbiter: SessionEndArbiter,
    clock: Callable[[], CallEndSnapshot],
    occupancy: Callable[[], Awaitable[int]],
    max_call_seconds: float | None,
    no_caller_speech_seconds: float | None,
    emit: Callable[..., None],
    occupancy_grace_seconds: float = WORKER_OCCUPANCY_GRACE_SECONDS,
    interval_seconds: float = WORKER_CALL_END_POLL_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    is_finishing: Callable[[], bool] = lambda: False,
    navigation: NavigationPolicy | None = None,
) -> str:
    """Poll the four endings until one applies; return the reason it ended.

    This is a SEPARATE task from the lifecycle heartbeat on purpose. The
    heartbeat's post is a blocking network call handed to a thread, and the
    call this exists for spent 94 minutes with a live heartbeat and a dead
    media path. A check living inside that loop inherits every way it can
    stall, which is exactly the failure being guarded against.

    It also cannot depend on the entrypoint task: that task had to be SIGKILLed
    in the incident. Polling is deliberately dumb for that reason -- no event
    subscription, no callback, nothing a missed delivery can silence.

    The clock and the occupancy probe are separated because they fail
    differently. The clock is in-memory and cannot fail. The probe is a network
    call to the LiveKit server and fails for ordinary reasons, so a server this
    worker cannot reach must not also disable the duration limit -- otherwise
    one outage returns the call to exactly the unbounded state the incident
    was. A failed probe leaves occupancy UNKNOWN, which never ends a call, and
    the timers carry on underneath it.
    """

    empty_since: float | None = None
    probe_failures = 0
    tick_failures = 0
    while True:
        try:
            await sleep(interval_seconds)
            if arbiter.reason or is_finishing():
                # Something else already ended the call -- the caller's disconnect
                # event, a room transport loss, the workflow. Keep polling and this
                # task outlives the teardown it cannot contribute to.
                #
                # ``arbiter.reason`` alone is not enough. The arbiter is
                # first-wins only among ITS OWN callers, and several paths still
                # end the session directly: the room-disconnect handler, workflow
                # completion, the synthetic-caller wake. After one of those the
                # arbiter is still empty, so a tick landing in the window before
                # this task is cancelled would end the call a second time and
                # stamp a platform reason on an ending the platform did not
                # decide -- telling a customer the other party hung up on a call
                # their own agent finished.
                return ""
            seconds_room_empty: float | None = None
            try:
                remaining = await occupancy()
            except Exception as exc:  # never BaseException: cancellation propagates
                probe_failures += 1
                empty_since = None
                emit(
                    "worker_call_end_probe_failed",
                    error_type=type(exc).__name__,
                    consecutive_failures=probe_failures,
                )
            else:
                if probe_failures:
                    emit("worker_call_end_probe_recovered", after_failures=probe_failures)
                    probe_failures = 0
                if remaining > 0:
                    empty_since = None
                else:
                    if empty_since is None:
                        empty_since = monotonic()
                    seconds_room_empty = monotonic() - empty_since
            current = clock()
            reason = worker_call_end_reason(
                CallEndSnapshot(
                    answered=current.answered,
                    seconds_since_answer=current.seconds_since_answer,
                    caller_has_spoken=current.caller_has_spoken,
                    hangup_requested=current.hangup_requested,
                    seconds_room_empty=seconds_room_empty,
                    # Read HERE, at the rule's own call site, so the silence
                    # rule below and the budgets after it see one state.
                    navigation_suspended=bool(
                        navigation is not None and navigation.suspends(ENDER_NO_CALLER_SPEECH)
                    ),
                    seconds_since_navigation_exit=(
                        navigation.seconds_since_resume(monotonic())
                        if navigation is not None and navigation.enabled
                        else None
                    ),
                ),
                max_call_seconds=max_call_seconds,
                no_caller_speech_seconds=no_caller_speech_seconds,
                occupancy_grace_seconds=occupancy_grace_seconds,
            )
            reason_detail = ""
            if navigation is not None:
                reason, reason_detail = navigation_adjusted_ending(
                    reason, navigation, now=monotonic()
                )
            if reason is None:
                tick_failures = 0
                continue
            if arbiter.reason or is_finishing():
                # Re-checked deliberately. The check at the top of the tick is
                # separated from this decision by an await on the network, and
                # another path can end the session inside that window -- which
                # would put a platform reason on an ending the platform did not
                # decide, exactly what the first check exists to prevent.
                return ""
            if arbiter.end(
                reason,
                detected_via=(
                    DETECTED_VIA_HANGUP_INTENT
                    if reason == TERMINATION_HANGUP_REQUESTED
                    else DETECTED_VIA_OCCUPANCY_PROBE
                    if reason == TERMINATION_REMOTE_PARTY_DISCONNECTED
                    else DETECTED_VIA_WORKER_TIMER
                ),
                **({"reason_detail": reason_detail} if reason_detail else {}),
            ):
                return reason
            # Something else finished the call between the probe and here.
            return ""
        except asyncio.CancelledError:
            # The job is tearing down. Ending promptly is correct: the entrypoint
            # cancels this task and then joins it, and swallowing the cancellation
            # would hang that join forever.
            raise
        except Exception as exc:
            # NOTHING ordinary may kill this task. Every guard against an unbounded
            # call lives inside this one loop -- the occupancy probe and both timers
            # -- so a single unhandled exception here returns the call to exactly
            # the state the incident was in, and it would do it silently: the task's
            # exception is not retrieved until teardown, where the gather discards
            # it. Surviving a bad tick and saying so is strictly better than dying
            # quietly with 29 minutes left on the clock.
            tick_failures += 1
            emit(
                "worker_call_end_tick_failed",
                error_type=type(exc).__name__,
                consecutive_failures=tick_failures,
            )
            continue


@dataclass(frozen=True)
class ProductionSessionCloseResult:
    """Outcome of the SDK close that must not outlive worker settlement."""

    completed: bool
    required: bool = True
    cancellation_resistant: bool = False
    error_type: str = ""
    skipped_reason: str = ""


@dataclass(frozen=True)
class BoundedTeardownTaskResult:
    """Bounded outcome for one provider-facing teardown task."""

    completed: bool
    cancellation_resistant: bool = False
    error_type: str = ""


def _consume_late_production_session_close(
    task: asyncio.Future[Any],
    *,
    on_late_completion: Callable[[str], None] | None,
) -> None:
    """Observe a close task that ignored cancellation without blocking exit."""
    error_type = ""
    try:
        task.result()
    except asyncio.CancelledError:
        error_type = "CancelledError"
    except Exception as exc:
        error_type = type(exc).__name__
    if on_late_completion is not None:
        try:
            on_late_completion(error_type)
        except Exception:
            pass


async def await_bounded_teardown_task(
    task: asyncio.Future[Any],
    *,
    timeout_seconds: float = BROWSER_TEST_REPORT_BUDGET_SECONDS,
    cancellation_grace_seconds: float = 0.2,
    on_late_completion: Callable[[str], None] | None = None,
) -> BoundedTeardownTaskResult:
    """Wait for provider teardown work without ceding the worker exit deadline.

    LiveKit can close the room transport before its session ``close`` callback
    schedules the metadata report.  Cancelling that report with
    ``asyncio.wait_for`` is not a bound: ``wait_for`` then waits for a
    cancellation-resistant provider coroutine forever.  Observe the existing
    task directly, give cancellation a short grace period, and detach only
    after installing a result consumer so teardown can continue to usage,
    evidence and terminal lifecycle settlement.
    """
    try:
        done, pending = await asyncio.wait((task,), timeout=max(0.0, timeout_seconds))
        if done:
            try:
                task.result()
            except asyncio.CancelledError:
                return BoundedTeardownTaskResult(
                    completed=False,
                    error_type="CancelledError",
                )
            except Exception as exc:
                return BoundedTeardownTaskResult(
                    completed=False,
                    error_type=type(exc).__name__,
                )
            return BoundedTeardownTaskResult(completed=True)

        task.cancel()
        done, pending = await asyncio.wait(
            pending, timeout=max(0.0, cancellation_grace_seconds)
        )
        if done:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
            return BoundedTeardownTaskResult(
                completed=False,
                error_type="TimeoutError",
            )

        task.add_done_callback(
            lambda completed: _consume_late_production_session_close(
                completed,
                on_late_completion=on_late_completion,
            )
        )
        return BoundedTeardownTaskResult(
            completed=False,
            cancellation_resistant=True,
            error_type="TimeoutError",
        )
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        task.add_done_callback(
            lambda completed: _consume_late_production_session_close(
                completed,
                on_late_completion=on_late_completion,
            )
        )
        raise


@dataclass(frozen=True)
class EvidenceSidecarShutdownResult:
    """Outcome of disconnecting the evidence room and draining capture tasks."""

    disconnect: BoundedTeardownTaskResult
    captures: BoundedTeardownTaskResult


async def shutdown_evidence_sidecar(
    sidecar: Any,
    capture_tasks: Iterable[asyncio.Future[Any]],
    *,
    timeout_seconds: float = 1.0,
    cancellation_grace_seconds: float = 0.2,
    on_late_completion: Callable[[str, str], None] | None = None,
) -> EvidenceSidecarShutdownResult:
    """Bound sidecar disconnect and capture drain as one evidence teardown.

    A LiveKit room can disappear before the sidecar's own peer connection has
    acknowledged ``disconnect()``. Neither that SDK coroutine nor an audio
    stream closing behind it may hold terminal usage/evidence settlement until
    the worker hard deadline. Each phase gets one shared-size bound; detached
    late completions remain observed.
    """
    disconnect_task = asyncio.ensure_future(sidecar.disconnect())
    disconnect = await await_bounded_teardown_task(
        disconnect_task,
        timeout_seconds=timeout_seconds,
        cancellation_grace_seconds=cancellation_grace_seconds,
        on_late_completion=(
            None
            if on_late_completion is None
            else lambda error_type: on_late_completion("disconnect", error_type)
        ),
    )

    pending_captures = tuple(capture_tasks)
    if not pending_captures:
        return EvidenceSidecarShutdownResult(
            disconnect=disconnect,
            captures=BoundedTeardownTaskResult(completed=True),
        )
    capture_group = asyncio.gather(*pending_captures, return_exceptions=True)
    captures = await await_bounded_teardown_task(
        capture_group,
        timeout_seconds=timeout_seconds,
        cancellation_grace_seconds=cancellation_grace_seconds,
        on_late_completion=(
            None
            if on_late_completion is None
            else lambda error_type: on_late_completion("captures", error_type)
        ),
    )
    return EvidenceSidecarShutdownResult(
        disconnect=disconnect,
        captures=captures,
    )


async def await_bounded_production_session_close(
    session: Any,
    *,
    timeout_seconds: float = 2.0,
    cancellation_grace_seconds: float = 0.2,
    on_late_completion: Callable[[str], None] | None = None,
) -> ProductionSessionCloseResult:
    """Close the LiveKit session without ceding the worker's exit deadline.

    A room-disconnect event can arrive after the SDK transport has already
    closed. In that state ``AgentSession.aclose()`` may wait for remote-session
    writes that can no longer complete. The production worker still owns
    evidence, usage reporting and terminal lifecycle settlement, so this one
    provider-facing close is bounded and observed rather than allowed to hold
    the entire entrypoint until LiveKit hard-cancels it.

    The caller joins the worker-owned end-call task before entering this helper;
    bounding SDK close therefore cannot cut a terminal goodbye or freeze its
    evidence early.
    """
    try:
        task = asyncio.ensure_future(session.aclose())
    except Exception as exc:
        return ProductionSessionCloseResult(
            completed=False,
            error_type=type(exc).__name__,
        )

    try:
        done, pending = await asyncio.wait((task,), timeout=max(0.0, timeout_seconds))
        if done:
            try:
                task.result()
            except asyncio.CancelledError:
                # This child task is created and owned here; no TaskGroup or
                # caller can cancel it independently. A CancelledError raised
                # by the close coroutine itself is therefore that action's
                # outcome, not cancellation of the owning production worker.
                # Cancellation of THIS waiter is handled by the outer except
                # below and remains propagated.
                return ProductionSessionCloseResult(
                    completed=False,
                    error_type="CancelledError",
                )
            except Exception as exc:
                return ProductionSessionCloseResult(
                    completed=False,
                    error_type=type(exc).__name__,
                )
            return ProductionSessionCloseResult(completed=True)

        task.cancel()
        done, pending = await asyncio.wait(
            pending, timeout=max(0.0, cancellation_grace_seconds)
        )
        if done:
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                pass
            return ProductionSessionCloseResult(
                completed=False,
                error_type="TimeoutError",
            )

        task.add_done_callback(
            lambda completed: _consume_late_production_session_close(
                completed,
                on_late_completion=on_late_completion,
            )
        )
        return ProductionSessionCloseResult(
            completed=False,
            cancellation_resistant=True,
            error_type="TimeoutError",
        )
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        task.add_done_callback(
            lambda completed: _consume_late_production_session_close(
                completed,
                on_late_completion=on_late_completion,
            )
        )
        raise


async def close_production_session_after_room_end(
    session: Any,
    *,
    room_transport_disconnected: Callable[[], bool],
    on_late_completion: Callable[[str], None] | None = None,
) -> ProductionSessionCloseResult:
    """Close a live session unless the room transport already owns teardown.

    Once LiveKit has emitted the room ``disconnected`` event, the SDK transport
    is closed and ``AgentSession.aclose()`` can become cancellation-resistant
    while draining remote-session writes.  Starting that close task at this
    point keeps the whole job alive until the worker's hard deadline.  The
    framework already tears the SDK session down with the disconnected room;
    the production entrypoint must instead continue its own usage, evidence and
    lifecycle settlement without creating another SDK close task.
    """
    # LiveKit room callbacks and this entrypoint execute on the same asyncio
    # loop. Read the callback-owned state HERE, immediately before the bounded
    # helper synchronously creates its close task; there is no await/yield in
    # between that can turn a fresh value into a stale pre-call snapshot.
    if room_transport_disconnected():
        return ProductionSessionCloseResult(
            completed=False,
            required=False,
            skipped_reason="room_transport_disconnected",
        )
    return await await_bounded_production_session_close(
        session,
        on_late_completion=on_late_completion,
    )


#: Statuses that mean this session is finished. Kept identical to the
#: reconciler's own ``TERMINAL_SESSION_STATUSES``; a test asserts the two agree,
#: because importing that module here would make every importer of this file
#: depend on the reconciler.
TERMINAL_SESSION_FILE_STATUSES = frozenset({"complete", "reconciled", "unreconcilable"})

#: Where a session-state file carries WHY the call ended. The worker writes it
#: at the instant the arbiter decides, before the ending is observable, so it
#: outlives a teardown that never runs.
SESSION_REASON_FIELD = "termination_detail"

#: When the arbiter decided, in the same ISO-Z form as every other lifecycle
#: timestamp on this call. Blank when this worker did not decide the ending.
#: Cleared together with the reason, never on its own: an instant belonging to
#: no reason is a worse record than neither.
SESSION_DECIDED_AT_FIELD = "decided_at"

#: What qualifies the reason (``hold_limit``, ``navigation_limit``,
#: ``deadline``, ``during_hold``): sealed WITH it and cleared with it, so a
#: disowned ``ivr_timeout/hold_limit`` can never sit beside a later
#: ``ivr_timeout/navigation_limit``. Blank for the many endings with no detail.
SESSION_REASON_DETAIL_FIELD = "termination_reason_detail"

#: Whether a person was reached EARLIER in the call, as the navigation policy
#: stated it at the instant of the decision (``true``/``false``; absent when
#: the agent does not navigate). Sealed WITH the reason: a retry policy and the
#: talk-time rule both read it beside the ending, so a half-decision that paired
#: a stale answer with a new reason would redial a person already spoken to.
SESSION_PRIOR_PERSON_FIELD = "termination_prior_person"

#: How many times a DISOWNED reason is cleared before giving up and saying so.
#: More than one because the clear holds a lock that fails open, so a heartbeat
#: landing in an unlocked window can put the record back; and bounded because
#: this runs inside a failed teardown, where looping is its own hazard.
RELEASE_ATTEMPTS = 3

#: The reason and the instant it was decided are ONE fact and are sealed
#: together. Guarding only the reason let a later write pair a STALE reason
#: with a FRESH instant -- the writer suppressed the new reason by write-once
#: and wrote the new timestamp beside the old one, and the reconciler reads the
#: two as a pair and bills at that instant. Half a decision is worse than
#: either half alone, and this file's own release comment already claimed they
#: could not come apart.
_DECIDED_TOGETHER = frozenset(
    {
        SESSION_REASON_FIELD,
        SESSION_DECIDED_AT_FIELD,
        SESSION_REASON_DETAIL_FIELD,
        SESSION_PRIOR_PERSON_FIELD,
    }
)

def clear_session_state_keys_confirmed(path: Path, *keys: str) -> bool:
    """Clear ``keys`` and confirm they are gone, WITHOUT releasing the lock.

    Clearing and then checking as two locked sections leaves a window between
    them, and a writer landing in it re-publishes what it read before the clear.
    The heartbeat is such a writer: it merges its own fields into whatever it
    last read, so a read taken before the clear and written after the check puts
    the removed reason back -- and the reason is write-once, so it is then
    permanent. Confirming inside the same hold removes the window rather than
    detecting what happened in it.

    Returns whether the keys are absent. A record that cannot be READ is not a
    confirmation: it raises, so the caller retries rather than believing it.
    """

    with locked_session_file(path):
        existing = read_session_file(path)
        if not existing:
            return True
        remaining = {k: v for k, v in existing.items() if k not in keys}
        if remaining != existing:
            publish_session_file(path, remaining)
        return not any(str(remaining.get(k) or "") for k in keys)


def clear_session_state_keys(path: Path, *keys: str) -> None:
    """Remove exactly these keys from the session file. The ONLY way to clear.

    ``write_session_state`` cannot express a removal, and that is deliberate
    rather than an omission: an omitted field keeps its value and an explicitly
    blank one keeps it too, so no caller can un-carry a reason by accident. That
    protection would be worthless if the same call could also delete.

    The one thing that genuinely must be removable is a reason for an ending
    that DID NOT HAPPEN. The worker's arbiter releases a stamped reason when the
    ending's finish raises, on the principle that the call is still running, and
    without a delete that released reason would sit on disk permanently, immune
    to correction by the real ending that follows, because the reason is
    write-once. So the pair is: write-once protects a reason, and this removes
    one that was disowned. Neither is safe alone.

    Being a separate function rather than a parameter is on purpose. The writer
    takes ``**state``, so a ``clear=`` keyword would be ambiguous with a field
    of that name; and a deletion should be hard to reach by accident, which a
    differently-named call is and a keyword on the everyday writer is not.

    A missing or unreadable file is a no-op: there is nothing to clear, and
    creating one here would invent a record for a call this process is
    abandoning.
    """

    with locked_session_file(path):
        # Deliberately NOT guarded. An unreadable record here must raise so the
        # caller retries: returning quietly would report a clear that never
        # happened, and the reason it failed to remove is write-once, so the
        # real ending that follows could never correct it.
        existing = read_session_file(path)
        if not existing:
            return
        remaining = {k: v for k, v in existing.items() if k not in keys}
        if remaining == existing:
            return
        publish_session_file(path, remaining)


def write_session_state(path: Path, **state: Any) -> None:
    """Merge these fields into the session-state file, preserving the rest.

    This file has TWO writers. The worker writes a fixed set of fields here;
    the reconciler annotates the same file with its own bookkeeping -- when a
    record first started failing, how many attempts it has cost, when the
    caller was first seen gone. A plain replace silently deleted every one of
    those on the worker's next heartbeat, which arrives every few seconds.

    That is not a tidiness problem. The reconciler's caller-gone clock could
    never accumulate on a call whose worker is still alive, so the backstop for
    a wedged-but-heartbeating worker could never fire on the exact incident it
    exists to catch, and a call already closed as ``reconciled`` was reopened to
    ``media_started`` by the next heartbeat and could be closed a second time.
    A terminal status therefore also wins over a later non-terminal one.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # The read and the replace are ONE step. The other container's writer runs
    # against this same file every few seconds on a wedged call, and a decision
    # committed between a bare read and its replace is read as absent and then
    # written away -- which is a terminal status reverted and the call settled
    # twice, or the reason the worker decided replaced by nothing.
    with locked_session_file(path):
        try:
            merged: dict[str, Any] = read_session_file(path)
        except SessionFileUnreadable:
            # The writer is the one caller that may legitimately recover: a
            # record it cannot parse is replaced by this write, which is what
            # every first write already does. Stated as a caught exception
            # rather than inherited from a lenient reader, so the recovery is
            # a decision this function makes rather than one made for it.
            merged = {}
        previous_status = str(merged.get("status") or "")
        # Rule 3, first terminal wins. Once this record is FINISHED and says why,
        # nothing may reopen it and nothing may relabel it -- not even another
        # terminal write. The case that rule exists for: this sweep's own generic
        # marker must never land on top of a reason the worker actually decided,
        # and whichever process happens to write last must not be what the customer
        # reads. A record that is terminal but has no reason yet is NOT sealed, so
        # a late writer can still supply the reason it was missing.
        # The record is FINISHED, whatever it does or does not say about why.
        # Gating this on a reason being present made "first terminal wins" false
        # for a terminal record that had none: a later write carrying a
        # different terminal status AND a reason would land both, relabelling an
        # ending that was already settled. No writer produces that pair today,
        # which is exactly why it had to be closed by the rule rather than left
        # to the accident that no caller currently reaches it.
        previously_settled = previous_status in TERMINAL_SESSION_FILE_STATUSES
        # The REASON stays fillable while it is absent, which is the whole point
        # of "write-once, not write-never": a late writer may still supply the
        # reason a settled record was missing. Only the status is sealed by the
        # settling itself -- the reason is sealed by ``reason_already_recorded``
        # below, independently and for a different rule.
        # WRITE-ONCE on the reason, from the moment it is written and independent
        # of status. Not an emergent effect of the reconciler's markers being
        # out-of-vocabulary and therefore harmless: even an in-vocabulary reason
        # from the sweep must not replace one the worker decided, and relying on
        # the vocabulary would put the protection in a different file from the
        # value it protects.
        #
        # Safe because the worker cannot produce two different non-blank reasons
        # for one session. Its arbiter is first-wins and is the only source of this
        # field. It does RELEASE a stamped reason when the ending's finish raises,
        # on the principle that the call did not end, and a later attempt can then
        # stamp a different one.
        #
        # What makes that release safe is that it is an EXPLICIT DELETE of this key,
        # so a disowned ending leaves nothing behind for write-once to protect.
        # Deliberately NOT the weaker argument that the persist happens late enough
        # in finish that nothing after it can raise: that is true today and is one
        # edit away from being false, and it would fail silently. A delete is
        # correct by construction whatever anyone later adds after the persist.
        #
        # The delete lives in ``SessionEndArbiter.end``, on the failure path that
        # disowns the ending, because that is the only place that knows an ending
        # was disowned. If that call is ever removed, THIS rule is what turns a
        # disowned ending into a permanent wrong reason, so the two are a pair and
        # neither is safe alone.
        reason_already_recorded = str(merged.get(SESSION_REASON_FIELD) or "") != ""
        for key, value in state.items():
            if key in _DECIDED_TOGETHER and reason_already_recorded:
                continue
            if previously_settled and key == "status":
                continue
            # An explicitly BLANK value never overwrites something already there.
            # A caller that stamps a field on every write -- a reason sourced from
            # an arbiter that has not decided yet, say -- would otherwise blank a
            # real value that an earlier or concurrent write had recorded, which is
            # the same family as a missing input being read as the reassuring
            # answer. Omitting a key keeps it and blanking a key keeps it, so no
            # caller can un-carry a reason by accident. Clearing a field is
            # deliberately NOT expressible here; a writer that owns a key clears it
            # by rewriting the whole record.
            if value in (None, "") and merged.get(key) not in (None, ""):
                continue
            merged[key] = value
        if previous_status in TERMINAL_SESSION_FILE_STATUSES and str(
            merged.get("status") or ""
        ) not in TERMINAL_SESSION_FILE_STATUSES:
            merged["status"] = previous_status
        publish_session_file(path, merged)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as recording:
        return recording.getnframes() / recording.getframerate()


#: The fields ``finalize_call_artifacts`` reads out of the recorder's own
#: finalize() result.  Named here so "is this the evidence summary?" is one
#: cheap membership test rather than four unguarded subscripts, and so a
#: reader can see at a glance which recorder fields the artifact contract
#: depends on.  Every one of them is produced by
#: ``livekit_evidence.SynchronizedEvidenceRecorder.finalize``.
EVIDENCE_SUMMARY_FIELDS = frozenset(
    {
        "semantic_turns",
        "semantic_evidence_status",
        "caller_audio_observed",
        "short_call_without_caller_speech",
    }
)


async def report_call_teardown_lifecycle(
    *,
    call: Any,
    call_id: str,
    ended_at: str,
    emit: Callable[..., None],
    termination_detail: str = "",
    ended_reason_detail: str = "",
    answer_classification: Mapping[str, str] | None = None,
    prior_person: bool | None = None,
) -> bool:
    """Report both teardown lifecycle events; say whether the terminal one landed.

    A REJECTED lifecycle report is a billing/contract outcome, not a reason to
    abandon the rest of teardown.  One 400 on the terminal report -- the
    worker's own usage payload violating the settlement contract -- used to
    crash the job task here, so nothing downstream ran at all and the call
    was recorded nowhere.  Classify it and let teardown finish.

    This does NOT recover the artifacts of a call whose terminal report was
    refused: the control-plane session refuses an upload while its lifecycle
    is unfinalized, deliberately, because storing evidence for a call whose
    billing never settled is a billing decision rather than a teardown one.
    What it buys is that the loss is classified, the session state is written,
    and the call stays reconcilable instead of vanishing.

    ``Exception``, never ``BaseException``: cancellation must still propagate.
    Only the exception CLASS is emitted -- the rejection detail is
    control-plane response text.

    The return value is the terminal report's fate, and the caller owes the
    session state a status that reflects it: a call whose ``sip.ended`` was
    refused has not settled, and writing ``complete`` would hide it from the
    reconciler forever.

    ``termination_detail`` is stamped on the TERMINAL event only, and only when
    this worker is the one that ended the call.

    That now INCLUDES an ordinary caller hangup.  The delivered
    ``participant_disconnected`` event routes through the same arbiter as the
    occupancy poll, so both stamp ``remote_party_disconnected`` -- deliberately,
    so a customer's reason answers "why did the call end" rather than "which of
    our internal paths noticed".  The consequence is worth stating plainly: from
    this release the most common ending in the system carries a reason where it
    previously carried none, and this payload reaches the customer's
    ``call.ended`` webhook.  Calls that completed before it stay empty.
    It rides ``details.reason``, the field the control plane already reads a
    reason from, rather than a second parallel channel.

    This ``details`` dict is stored by the control plane and forwarded to the
    customer's ``call.ended`` webhook VERBATIM -- the gateway lifecycle handler
    says so in as many words -- so a key added here is a published API field,
    not an internal note.  The detection path in particular must never travel
    this way: it is a fact about our own delivery plumbing, its meaning would
    change whenever that plumbing changed, and a customer would have no
    documentation for it. It belongs in this worker's own evidence, which is
    where ``worker_ended_call`` already records it.

    ``answer_classification`` is the second thing allowed here, added on
    purpose and with that cost understood: it is a documented customer field
    with its own enum, not an internal note, and it rides the same channel
    because it describes the same thing the reason does -- what this call
    turned out to be.

    It differs from the reason in one way that matters.  The reason is stamped
    only when this worker decided the ending, so a call the agent finished
    carries none.  The classification is stamped on EVERY terminal report the
    detector collected on, whatever ended the call, because the population it
    is measured over is the point: a verdict recorded only on calls something
    already acted on describes the calls we acted on, not the calls we get.
    """

    terminal_reported = True
    for suffix, event_type, source_component in (
        ("room-ended", "room.ended", "livekit_room"),
        ("sip-ended", "sip.ended", "livekit_sip"),
    ):
        details: dict[str, Any] | None = None
        if event_type == "sip.ended":
            # Published to the customer's call.ended webhook verbatim. See the
            # docstring: every key here is an API field.
            published: dict[str, Any] = {}
            if termination_detail:
                published["reason"] = termination_detail
            for key in ("answer_classification", "reason_detail"):
                value = str((answer_classification or {}).get(key, "") or "")
                if value:
                    published[key] = value
            # ONE column, two things it can qualify, so the two writers have to
            # be ordered rather than left to arrive in whatever sequence a call
            # produced them. The ending wins.
            #
            # A row that carries both is a call whose post-answer detector had
            # already formed a verdict about what answered AND which this
            # worker's own rule then ended for a reason of its own. The reason
            # column on that row says why it ended, so a detail describing what
            # was heard at the start would be read against an ending it does
            # not belong to. The classification itself is untouched and still
            # says what was heard; only the qualifier follows the reason.
            if ended_reason_detail:
                published["reason_detail"] = ended_reason_detail
            if prior_person is not None:
                # Whether a person was reached earlier, sealed with the ending
                # it qualifies. A published API field: it is what keeps a
                # transfer into a queue that ran out of time, after a real
                # conversation, from being redialled or dropped from talk time.
                published["person_reached"] = prior_person
            details = published or None
        try:
            await asyncio.to_thread(
                call.report_lifecycle,
                event_id=f"{call_id}:{suffix}",
                event_type=event_type,
                source_component=source_component,
                observed_at=ended_at,
                details=details,
            )
        except Exception as exc:
            if event_type == "sip.ended":
                terminal_reported = False
            emit(
                "lifecycle_report_rejected",
                event_type=event_type,
                error_type=type(exc).__name__,
            )
    return terminal_reported


#: How long a finalize already under way is still waited for once the job has
#: been cancelled. The SDK cancels the entrypoint 15 s after the room goes
#: away and then waits for it to return, so this is the only bound on that
#: wait. It matches the artifact upload's own ceiling
#: (``control_plane.artifact_upload_timeout``): a finalize still running past
#: it is not slow, it is lost.
FINALIZE_ARTIFACTS_CANCEL_GRACE_SECONDS = 600.0


async def finalize_through_cancellation(
    finalize: Awaitable[str],
    *,
    grace_seconds: float = FINALIZE_ARTIFACTS_CANCEL_GRACE_SECONDS,
    emit: Callable[..., None] | None = None,
) -> tuple[str | None, asyncio.CancelledError | None]:
    """Run the call's artifact finalize to its end even if the job is cancelled.

    The upload inside ``finalize_call_artifacts`` runs after the room has been
    deleted, which is when the SDK starts its 15 s budget for the entrypoint
    and then cancels it. A cancellation arriving during the upload used to
    abandon the rest of the finalize: the evidence import and its record.
    The session-state write at its end usually lands after the reconciler has
    already swept the call in that case, and first terminal wins, which is
    harmless: the sweep bills from the worker's decided reason and instant.

    The finalize runs as its own task. A cancellation is absorbed and returned
    for the caller to re-raise once the teardown is done, like every other
    teardown step's, and the finalize is given ``grace_seconds`` more. Past
    that it is cancelled and the call is left to the reconciler. Returns the
    finalize's ``artifact_status`` (``None`` if it was abandoned) and the
    absorbed cancellation. Any other failure is raised as before.
    """

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(finalize)
    cancellation: asyncio.CancelledError | None = None
    deadline: float | None = None
    abandoned = False
    while not task.done():
        timeout = None if deadline is None else max(0.0, deadline - loop.time())
        try:
            await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError as cancelled:
            if cancellation is None:
                cancellation = cancelled
                deadline = loop.time() + grace_seconds
            continue
        if not task.done() and deadline is not None and loop.time() >= deadline:
            abandoned = True
            task.cancel()
            deadline = None
    if abandoned or task.cancelled():
        if emit is not None:
            try:
                emit("artifact_finalize_abandoned", grace_seconds=grace_seconds)
            except Exception:
                pass
        return None, cancellation
    return task.result(), cancellation


async def finalize_call_artifacts(
    *,
    call: Any,
    call_id: str,
    evidence_dir: Path,
    state_path: Path,
    summary: Mapping[str, Any] | None,
    room_name: str,
    started_at: str,
    ended_at: str,
    telemetry_turns: int,
    emit: Callable[..., None],
    session_status: str = "complete",
    participant_identity: str = "",
    sip_statistics_dir: Path | None = None,
    retry_later: bool = True,
    billing_source: str = "gateway_sip_lifecycle",
) -> str:
    """Upload the call's evidence and durably record how complete it was.

    An artifact failure here happens AFTER the caller has hung up and
    after gateway ``sip.ended`` has already settled billing.  Raising out of
    this point bought nothing and cost two things -- an unhandled exception in
    the job task, and a session state that never reached ``complete``, so the
    reconciler later relabelled a healthy call ``worker_restart_orphan``.
    Classify the outcome in ``artifact_status`` and let teardown finish.

    ``summary`` must be the recorder's evidence contract.  Reading a field
    straight out of it made the FIRST statement in this function the one
    statement that could still escape, and a caller that passed a different
    bounded-counter mapping lost every artifact on every call while this
    function's own docstring promised the opposite.  A mapping that is not
    that contract is now classified like any other failure.

    ``retry_later`` is whether anything will retry a failed upload from the
    evidence directory left behind -- the phone host's reconciler does. A
    worker with no retrier (the homepage demo, whose evidence is purged when
    the session's teardown ends) passes ``False``: a failed upload is then
    final, ``finalization_failed`` and on the call's timeline, never
    ``upload_pending`` with nobody coming. ``billing_source`` names the fact
    that ended the call's billing, for the ``call_complete`` line only.

    ``session_status`` stays ``complete`` for a call whose terminal lifecycle
    event was accepted.  A caller whose terminal report was REJECTED passes a
    reconciler-eligible status instead: the evidence is still imported now,
    but settlement is not silently dropped by a state file claiming the
    worker finished cleanly.
    """

    imported: dict[str, Any] = {}
    artifact_status = "unavailable_no_room_audio"
    # Set only when the upload failed in a way a later attempt can outlive:
    # the fields the reconciler's artifact sweep reads to retry it.
    pending_upload: dict[str, Any] = {}
    if summary is not None:
        missing_evidence_fields = sorted(EVIDENCE_SUMMARY_FIELDS - set(summary))
        if missing_evidence_fields:
            emit(
                "artifact_evidence_summary_invalid",
                missing=missing_evidence_fields,
            )
            summary = None
            artifact_status = "evidence_summary_invalid"
    if summary is not None:
        short_call = bool(summary["short_call_without_caller_speech"])
        try:
            # The SIP bridge writes its carrier-leg counters after
            # DeleteRoom closes the SIP leg. This runs after
            # hangup in the production entrypoint; the bounded wait is for
            # the capture unit to land the file, not for the room to stay
            # up. Attached before the upload so it rides the same bundle;
            # never a reason to lose the rest of the evidence.
            try:
                sip_statistics = await asyncio.to_thread(
                    attach_sip_call_statistics,
                    evidence_dir,
                    capture_dir=sip_statistics_dir,
                    room_name=room_name,
                    participant_identity=participant_identity,
                )
                emit("sip_call_statistics_attached", **sip_statistics)
            except Exception as exc:
                emit("sip_call_statistics_attach_failed", error_type=type(exc).__name__)
            # Reading the mixed recording is inside the guard on purpose: an
            # unreadable WAV here must classify the call, not escape teardown.
            duration = wav_duration(evidence_dir / "mixed.wav")
            duration_sources = {
                "sip_participant": duration,
                "room_track": duration,
                "recording": duration,
            }
            imported = await asyncio.to_thread(
                call.complete,
                source_path=evidence_dir,
                duration_sources=duration_sources,
                require_semantic_turns=not short_call,
                semantic_evidence_status=str(summary["semantic_evidence_status"]),
            )
            evidence = imported.get("evidence") if isinstance(imported, dict) else None
            if isinstance(evidence, dict) and evidence.get("status") != "complete":
                artifact_status = "evidence_incomplete"
                emit(
                    "artifact_evidence_incomplete",
                    semantic_evidence_status=summary["semantic_evidence_status"],
                    missing=evidence.get("missing") or [],
                )
            else:
                artifact_status = str(
                    imported.get("artifact_import", {}).get("status") or "imported"
                )
        except Exception as exc:
            retryable = retry_later and isinstance(exc, ArtifactUploadFailed) and exc.retryable
            next_attempt_at = (
                next_artifact_attempt_at(1, datetime.now(timezone.utc)) if retryable else None
            )
            if next_attempt_at is not None:
                # NOT a finished record: the reconciler's artifact sweep retries
                # this bundle, with bounded backoff, from the evidence directory
                # the worker leaves behind. The lifecycle ``status`` below stays
                # what the call's lifecycle says -- billing is settled -- so the
                # lifecycle sweep does not report this call a second time.
                artifact_status = ARTIFACT_UPLOAD_PENDING
                pending_upload = {
                    "artifact_upload_attempts": 1,
                    "artifact_upload_next_at": next_attempt_at,
                    "artifact_upload_reason": str(getattr(exc, "reason", "") or ""),
                    "artifact_upload_params": artifact_retry_params(
                        duration_sources=duration_sources,
                        require_semantic_turns=not short_call,
                        semantic_evidence_status=str(summary["semantic_evidence_status"]),
                    ),
                }
            else:
                artifact_status = "finalization_failed"
            # The status, the API's reason code and the step are what tell a
            # size cap from an outage from a bad bundle; ``error_type`` alone
            # was all a lost 18-minute call ever logged. Never the body.
            failure = {
                "status_code": getattr(exc, "status_code", None),
                "reason": str(getattr(exc, "reason", "") or ""),
                "stage": str(getattr(exc, "stage", "") or ""),
                "total_bytes": getattr(exc, "total_bytes", None),
                "retry_next_at": next_attempt_at,
            }
            emit(
                "artifact_finalization_failed",
                semantic_evidence_status=summary["semantic_evidence_status"],
                caller_audio_observed=summary["caller_audio_observed"],
                short_call_without_caller_speech=short_call,
                error_type=type(exc).__name__,
                **failure,
            )
            # Upload failures only: every other class here (no evidence at
            # all, an unreadable WAV) was, and stays, a log line.
            if isinstance(exc, ArtifactUploadFailed):
                capture_exception(
                    exc,
                    phase="artifact_upload",
                    stage=failure["stage"] or "unknown",
                    reason=failure["reason"] or type(exc).__name__,
                    status_code=str(failure["status_code"] or ""),
                    retry="scheduled" if pending_upload else "none",
                )
            # A final verdict goes on the call's own timeline now; a scheduled
            # retry's does only if the sweep finally gives up, so a customer
            # never reads "failed" for a recording that arrives an hour later.
            report = getattr(call, "report_artifact_upload_failure", None)
            if isinstance(exc, ArtifactUploadFailed) and not pending_upload and callable(report):
                await asyncio.to_thread(report, exc)
    write_session_state(
        state_path,
        call_id=call_id,
        room_name=room_name,
        started_at=started_at,
        last_heartbeat_at=ended_at,
        status=session_status,
        artifact_status=artifact_status,
        **pending_upload,
    )
    emit(
        "call_complete",
        semantic_turns=summary["semantic_turns"] if summary else 0,
        semantic_evidence_status=(
            summary["semantic_evidence_status"]
            if summary
            else "unavailable_no_room_audio"
        ),
        caller_audio_observed=(summary["caller_audio_observed"] if summary else False),
        short_call_without_caller_speech=(
            summary["short_call_without_caller_speech"] if summary else True
        ),
        telemetry_turns=telemetry_turns,
        artifact_status=artifact_status,
        billing_source=billing_source,
    )
    return artifact_status


GREETING_SPOKEN = "spoken"
GREETING_FALLBACK_SPOKEN = "fallback_spoken"
# The caller barged in over the fallback. Distinct from ``SPOKEN`` because
# nothing available here can tell whether any of the line played first, and a
# value named "spoken" would invite a future caller to treat it as delivered.
GREETING_FALLBACK_INTERRUPTED = "fallback_interrupted"
GREETING_UNSPOKEN = "unspoken"


async def _awaited(handle: Any) -> None:
    """One await, wrappable in a task: a ``SpeechHandle`` is awaitable, not a future."""
    await handle


async def run_until_session_end(work: Awaitable[Any], *, ended: Any = None) -> tuple[bool, Any]:
    """Await ``work``, or stop waiting when the session ends, whichever is first.

    Returns ``(True, result)`` when the work finished and ``(False, None)``
    when the session ended first. With no ``ended`` it is a plain await.

    Anything the opening awaits before the entrypoint reaches its own wait on
    the ending must be raced this way. The worker decides a call is over by
    setting ``ended`` (``finished``); an await that cannot see it keeps the
    entrypoint parked above its own teardown until the reconciler sweeps the
    call and the SDK cancels the job, and such a call saves no evidence.

    It is a race, not a timeout: any timeout long enough not to cut a slow
    greeting is long enough to be a hang, and the question is whether the call
    is still going, not whether it has been a while. Only the wait is
    cancelled: work that is shielded underneath (a ``SpeechHandle``'s
    playout, a worker thread) runs to its own end.
    """

    wait_for_end = getattr(ended, "wait", None)
    if wait_for_end is None:
        return True, await work
    task = asyncio.ensure_future(work)
    end = asyncio.ensure_future(wait_for_end())
    try:
        await asyncio.wait({task, end}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # Both, on every path: the loser is ours and nothing else collects it,
        # and a cancelled caller must not leave either behind.
        for pending in (task, end):
            if not pending.done():
                pending.cancel()
    if task.done() and not task.cancelled():
        # What the plain await would have raised, so the callers' own except
        # clauses see what they always saw.
        return True, task.result()
    return False, None


async def await_playout_or_session_end(handle: Any, *, ended: Any = None) -> bool:
    """Wait for one utterance's playout, or for the session to end.

    Returns whether the playout finished. ``False`` means the session ended
    first and nobody heard the line out, so a caller must not count it as
    heard. The pinned SDK's playout wait is shielded, so giving up the wait
    does not cut the line for a caller who is still there.

    A bare ``await handle`` is the defect this replaces: the SDK never fails a
    handle whose far end has gone, and an opening spoken under protection
    cannot be interrupted, so a caller who hung up during the greeting left
    the entrypoint parked above its own teardown.
    """

    if handle is None:
        return True
    finished, _ = await run_until_session_end(_awaited(handle), ended=ended)
    return finished


async def _say_and_report_outcome(
    session: Any, message: str, *, add_to_chat_ctx: bool, ended: Any = None
) -> tuple[BaseException | None, bool]:
    """Speak one line; report ``(provider failure or None, was interrupted)``.

    ``SpeechHandle.__await__`` waits for full playout and, per the pinned
    SDK's own docstring, **never raises** -- a generation that failed puts the
    error on ``handle.exception()`` instead. That is exactly why a failed
    opening greeting was invisible at the call site: ``await session.say(...)``
    returned normally and the caller heard nothing. Reading ``exception()`` is
    the same shape ``WorkerLocalEndCall._execute`` already uses for the
    terminal message.

    ``interrupted`` is reported separately because it is neither success nor
    failure: the caller barged in, and the SDK completes such a handle with
    ``interrupted is True`` and ``exception() is None``. Nothing here can tell
    how much of the line played before that, so a caller who cut it off is
    never counted as having heard it.

    ``exception()`` raises ``InvalidStateError`` if the handle is somehow not
    done, and a stub session may return no handle at all; treat either as "no
    evidence of failure" rather than inventing one.
    """

    handle = session.say(
        message, allow_interruptions=True, add_to_chat_ctx=add_to_chat_ctx
    )
    if handle is None:  # A session/stub that returns nothing has nothing to fail.
        return None, False
    if not await await_playout_or_session_end(handle, ended=ended):
        # The session ended before the line finished: not heard to the end,
        # which is what ``interrupted`` tells every caller of this seam.
        return None, True
    try:
        return handle.exception(), bool(getattr(handle, "interrupted", False))
    except Exception:
        return None, bool(getattr(handle, "interrupted", False))


async def speak_opening_line(
    *,
    session: Any,
    message: str,
    language: str,
    fallback: str | None = None,
    emit: Callable[..., None] | None = None,
    note_fallback: Callable[[], None] | None = None,
    still_present: Callable[[], bool] | None = None,
    speak: Callable[..., Awaitable[tuple[BaseException | None, bool]]] | None = None,
    audio_frames_seen: Callable[[], int] | None = None,
    ended: Any = None,
) -> str:
    """Speak the configured greeting, and never leave an answered call silent.

    On the first Sinhala campaign the callee picked up and heard
    nothing for 24 seconds, because Gemini TTS returned no audio for the
    greeting and the failure surfaced only as a worker log line. One short
    measured line in the call's own language is spoken instead, so the callee
    knows somebody is there and has something to answer -- which is also what
    gets the conversation to the model's first real turn.

    This does NOT decide whether speaking is allowed. The answer gate
    (``may_speak``) is checked by the caller before this runs and is not
    re-litigated here; ``still_present`` is only the same best-effort
    "has the callee left during the several seconds we just spent failing"
    re-check the greeting itself does, and is skipped entirely when the caller
    has no such signal.

    ``audio_frames_seen`` reports a monotonic count of opening utterances
    that have reached the wire. It is read before and after the configured
    line, because a synthesis can fail PART WAY THROUGH: the handle carries
    an exception, but the callee already heard "Hello, this is FinConn-".
    Speaking a fallback on top of that would add an unrelated second line
    rather than rescue anything, so the fallback is reserved for the case it
    was built for -- nothing was heard at all. Callers with no such signal
    pass nothing and keep the plain exception-only behaviour.

    ``fallback`` overrides the line spoken when the configured one fails. A
    workflow whose opening IS the whole conversation needs a goodbye there,
    not "can you hear me" followed immediately by a hangup; the default is
    the language's greeting fallback.

    ``speak`` performs one utterance and reports ``(failure, interrupted)``.
    The worker passes ``say_opening_utterance``, which additionally holds an
    opening line uninterruptible while it plays; this module
    cannot import that (it owns no LiveKit dependency), and the default is
    the plain say. The two concerns compose rather than compete: one keeps
    the opening from being cancelled before it is heard, the other notices
    when there was nothing to hear.

    ``ended`` is the session's end signal. The default speaker stops waiting
    on it (``await_playout_or_session_end``); a ``speak`` the caller passes is
    handed it by that caller. Either way no fallback is spoken into a session
    that has already ended.

    Never raises for a provider failure -- the return value is the outcome.
    Cancellation still propagates: a cancelled session must tear down, not
    speak a fallback.

    ``greeting_fallback_used`` is best-effort by construction, and the limit
    is worth stating: ``still_present`` is read BEFORE the fallback is
    spoken, and the SDK's handle reports interruption but not "the callee
    hung up during playout", so a callee who leaves mid-line is still
    counted. Closing that would need a disconnect signal this seam does not
    have. It bounds the error to one line per call, and in the direction of
    over-counting delivery, never under-counting a silent call.
    """

    def _emit(event: str, **details: Any) -> None:
        if emit is None:
            return
        try:
            emit(event, **details)
        except Exception:
            pass

    async def utter(
        session: Any, message: str, *, add_to_chat_ctx: bool
    ) -> tuple[BaseException | None, bool]:
        if speak is not None:
            return await speak(session, message, add_to_chat_ctx=add_to_chat_ctx)
        return await _say_and_report_outcome(
            session, message, add_to_chat_ctx=add_to_chat_ctx, ended=ended
        )

    def _frames() -> int | None:
        if audio_frames_seen is None:
            return None
        try:
            return int(audio_frames_seen())
        except Exception:
            return None

    frames_before = _frames()
    try:
        # The configured line goes into the chat context exactly as it always
        # has, so the model knows it greeted.
        failure, _interrupted = await utter(session, message, add_to_chat_ctx=True)
    except Exception as exc:  # ``say`` itself refused, before any playout.
        failure = exc
    if failure is None:
        # A caller who barged in over the greeting still heard a greeting;
        # only a provider failure sends us to the fallback.
        return GREETING_SPOKEN

    _emit("greeting_failed", language=language, error_type=type(failure).__name__)
    frames_after = _frames()
    if (
        frames_before is not None
        and frames_after is not None
        and frames_after > frames_before
    ):
        # The line reached the wire and then broke. The callee heard the
        # start of their greeting; a fallback here would be a second,
        # unrelated sentence rather than a rescue.
        _emit("greeting_partially_played", language=language)
        return GREETING_SPOKEN
    fallback = fallback if fallback is not None else greeting_fallback_message(language)
    if not fallback or fallback == message:
        # No measured line for this language, or the fallback IS the line
        # that just failed and would fail again. Silence beats speaking
        # English into a call that is not in English, and beats an unmeasured
        # string that may not synthesize either.
        _emit("greeting_fallback_unavailable", language=language)
        return GREETING_UNSPOKEN
    if still_present is not None:
        try:
            present = bool(still_present())
        except Exception:
            # Fail open, like every other read of this signal: an unreadable
            # roster must not convert a recoverable greeting into silence.
            present = True
        if not present:
            _emit("greeting_fallback_skipped", language=language, reason="callee_gone")
            return GREETING_UNSPOKEN
    if callable(getattr(ended, "is_set", None)) and ended.is_set():
        _emit("greeting_fallback_skipped", language=language, reason="session_ended")
        return GREETING_UNSPOKEN

    try:
        # Deliberately NOT added to the chat context. Two reasons, both
        # load-bearing. The model must not believe it delivered the
        # configured opening when the callee heard a bare "can you hear me",
        # so its own first turn should still introduce the call. And
        # ``SemanticTurnEventRecorder`` classifies any assistant
        # conversation item that is not a fragment of ``first_message`` as a
        # semantic reply once the caller has spoken -- and the caller can
        # easily have spoken during the several seconds the failed synthesis
        # took -- which would file this fixed line as a model turn, with a
        # telemetry span and a reply-latency measurement that mean nothing.
        # ``acknowledgements.py`` keeps its fixed phrases out of the context
        # for the same reason.
        fallback_failure, fallback_interrupted = await utter(
            session, fallback, add_to_chat_ctx=False
        )
    except Exception as exc:
        fallback_failure, fallback_interrupted = exc, False
    if fallback_failure is not None:
        _emit(
            "greeting_fallback_failed",
            language=language,
            error_type=type(fallback_failure).__name__,
        )
        return GREETING_UNSPOKEN
    if fallback_interrupted:
        # The caller spoke over it. That is a healthy call -- they are
        # talking -- but nothing here knows whether any of the line played
        # first, so it is reported as its own event rather than counted as
        # delivered.
        _emit("greeting_fallback_interrupted", language=language)
        return GREETING_FALLBACK_INTERRUPTED
    # Counted only once the line actually played to the end, so the per-call
    # summary's ``greeting_fallback_used`` means "the callee heard the
    # fallback", not "we tried". A fallback that failed is a silent call, and
    # the event above is what says so.
    if note_fallback is not None:
        try:
            note_fallback()
        except Exception:
            pass
    _emit("greeting_fallback_used", language=language)
    return GREETING_FALLBACK_SPOKEN


# One TTS sentence each, not two -- see the same note on
# ``GREETING_FALLBACK_MESSAGES``. ``AgentSession`` wraps a non-streaming TTS in
# ``tts.StreamAdapter``, which synthesises sentence by sentence, so a
# two-sentence goodbye is two provider requests and either one failing fails
# the goodbye.
#
# The Sinhala entry used to read "ඔබ ඇමතූවාට ස්තුතියි. ආයුබෝවන්." and split
# into "ඔබ ඇමතූවාට ස්තුතියි." plus a ten-character " ආයුබෝවන්.". Measured
# against the production Gemini stack 2026-08-28: the first fragment 5/5, the
# second 1/5, this comma-joined single-fragment rewrite 5/5. So the Sinhala
# goodbye was failing on most calls, for the same silent no-audio reason as
# the greeting above. Estonian is already a single fragment; English
# splits the same way ("Thank you for calling." + " Goodbye.") but is left
# exactly as it is -- both of those languages run on ElevenLabs, which has
# never shown this failure, and English measured 2/2 on Gemini too. Only the
# language that actually loses audio is changed.
_END_CALL_FALLBACK_MESSAGES = {
    "en": "Thank you for calling. Goodbye.",
    "et": "Aitäh helistamast. Head aega.",
    "si": "ඔබ ඇමතූවාට ස්තුතියි, ආයුබෝවන්.",
}


def end_call_fallback_message(language: str) -> str:
    """The goodbye spoken when an authored farewell cannot be synthesized.

    Same table the terminal ``end_call`` tool falls back to, exposed because
    a workflow whose opening turn is already terminal needs it too: that
    turn's authored text can fail to synthesize exactly like a greeting, and
    hanging up in silence is the worse of the two endings.

    Deliberately WITHOUT that tool's English default. ``end_call`` reaches
    this table only after a whole conversation has happened, where an English
    goodbye on a Tamil call is odd but comprehensible; this function feeds
    ``speak_opening_line``, which holds the opposite rule for the same reason
    its own fallback table does -- speaking English into a call that is not
    in English is worse than the silence it is trying to prevent. An
    unlisted language gets ``""``, which that function reads as "no measured
    line" and says nothing. ``WorkerLocalEndCall`` keeps its own English
    default, unchanged.
    """

    return _END_CALL_FALLBACK_MESSAGES.get(str(language or "").strip().lower(), "")


#: How recently the room must have heard a voice-to-voice agent for its
#: goodbye to count as still playing when the tool's generation completes.
#: Wider than the drain's own quiet window (0.15 s) by the ~140 ms the room
#: lags the local source plus scheduling jitter; any narrower and a tail still
#: in flight would be cut, any wider and a goodbye that ended seconds earlier
#: would hold the leg open for a drain that can only report ``unobserved``.
LIVE_GOODBYE_FRESH_SECONDS = 0.5

#: The ``end_call`` playout outcomes after which the goodbye reached its end.
#: ``completed`` is the text path; ``drained`` and ``already_quiet`` are the
#: voice-to-voice path, where the model spoke it and the room went quiet.
END_CALL_HEARD_PLAYOUTS = frozenset({"completed", "drained", "already_quiet"})
#: How often a held terminal commit re-reads whether a platform ending (whose
#: finished flag has no event to wait on) has overtaken it.
CLOSE_FLOOR_POLL_SECONDS = 0.05
#: ``(owner, CloseFloorEntry | None)`` of the ``end_call`` attempt running in
#: THIS task. Task-local on purpose: two overlapping attempts each read their
#: own window, so a later one can never move the guard's deadline for the
#: attempt that leads its decision.
_CLOSE_FLOOR_ENTRY: contextvars.ContextVar[tuple[Any, Any] | None] = contextvars.ContextVar(
    "skysay_close_floor_entry", default=None
)


#: How long the room must stay quiet after the SDK's playout edge before the
#: goodbye counts as delivered.  Shorter than the 120 ms intra-utterance gap
#: finalization tolerates would misread a syllable pause as the end.
TERMINAL_EGRESS_QUIET_SECONDS = 0.15
#: Held after the room goes quiet: the room is the SFU side of the SIP bridge,
#: and the carrier leg beyond it is not observable from here.
TERMINAL_EGRESS_POST_QUIET_PAD_SECONDS = 0.15
#: One absolute deadline for polling AND padding.  The measured room lag past
#: the SDK edge is ~140 ms; the bound leaves room for a tail arriving up to
#: ~500 ms late and still confirming, keeps a lost observer from holding the
#: caller's line, and a drain that reaches it is reported as unconfirmed, never
#: as heard.
TERMINAL_EGRESS_MAX_WAIT_SECONDS = 0.8
_TERMINAL_EGRESS_POLL_SECONDS = 0.02


#: A sidecar that has delivered no frame on any track for this long, while the
#: room and caller are still connected, has stopped observing.  The caller leg
#: publishes 20 ms frames continuously, so a quarter second of nothing is loss.
TERMINAL_EGRESS_OBSERVER_STALE_SECONDS = 0.25


async def drain_terminal_egress(
    *,
    agent_audible_until: Callable[[], float | None],
    room_disconnected: Callable[[], bool],
    goodbye_playout_start: float | None = None,
    observer_alive: Callable[[], bool] = lambda: True,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    quiet_seconds: float = TERMINAL_EGRESS_QUIET_SECONDS,
    post_quiet_pad_seconds: float = TERMINAL_EGRESS_POST_QUIET_PAD_SECONDS,
    max_wait_seconds: float = TERMINAL_EGRESS_MAX_WAIT_SECONDS,
) -> dict[str, object]:
    """Wait, bounded, for the room to stop hearing the goodbye.

    ``SpeechHandle.wait_for_playout`` resolves when the last frame has been
    handed to the local WebRTC source; the room hears it ~140 ms later and the
    carrier later still.  ``shutdown`` right at that edge starts tearing the
    session down while the farewell is in flight.  This polls the evidence
    recorder's live agent edge -- the same sidecar that later measures each
    clip -- and returns one of:

    * ``room_quiet``: agent audio after the goodbye's own speech-ID-matched
      playout start was observed within one quiet window before the SDK
      completion or later, and then stopped for ``quiet_seconds``, and the pad was held on top with the room
      still quiet and still connected.  The only confirming signal.
    * ``room_disconnected``: the transport closed first, during the wait or
      during the pad; whether the far end heard the tail is unknowable.
    * ``tail_elapsed``: the deadline passed while agent audio was still
      arriving, or before a quiet window plus its pad could complete.
    * ``unobserved``: the deadline passed without both the goodbye's own
      playout-start identity and audio after it. A missing speaking edge,
      absent sidecar, earlier-turn edge or stale partial farewell cannot confirm.
    * ``observer_lost``: ``observer_alive`` turned false during the wait or
      the pad -- the sidecar disconnected or stopped delivering frames -- so
      a frozen agent edge proves nothing about the room.  Unconfirmed.

    Every sleep is clamped to the remaining budget, so the whole drain never
    outlives ``max_wait_seconds`` from its start.
    """

    if quiet_seconds <= 0 or max_wait_seconds <= 0 or post_quiet_pad_seconds < 0:
        raise ValueError("terminal egress drain bounds must be positive")
    started = now()
    deadline = started + max_wait_seconds
    # The room may already have received the whole farewell when the SDK's
    # playout wait resolves. Comparing with an edge sampled HERE loses that
    # proof. Use only the matching farewell handle's observed speaking edge;
    # say-time can precede a same-generation preamble and is not a substitute.
    start_observed = (
        goodbye_playout_start is not None
        and math.isfinite(goodbye_playout_start)
        and goodbye_playout_start <= started
    )
    while True:
        at = now()
        elapsed = at - started
        if room_disconnected():
            return {
                "signal": "room_disconnected",
                "waited_ms": round(elapsed * 1000),
                "confirmed": False,
            }
        if not observer_alive():
            return {
                "signal": "observer_lost",
                "waited_ms": round(elapsed * 1000),
                "confirmed": False,
            }
        audible_until = agent_audible_until()
        # Pre-drain audio must still be fresh at SDK completion: caller
        # frames can keep the observer alive even if the agent track froze
        # midway through a farewell. Reuse the quiet window as the maximum
        # lead, so accepting an already-delivered tail still holds at least
        # the full post-quiet pad after SDK completion. Later audio may recover
        # a stale initial edge; a frozen partial farewell remains unconfirmed.
        goodbye_heard = (
            start_observed
            and audible_until is not None
            and math.isfinite(audible_until)
            and audible_until > goodbye_playout_start
            and audible_until >= started - quiet_seconds
        )
        if goodbye_heard and at - audible_until >= quiet_seconds:
            pad_end = audible_until + quiet_seconds + post_quiet_pad_seconds
            if pad_end > deadline:
                return {
                    "signal": "tail_elapsed",
                    "waited_ms": round(elapsed * 1000),
                    "confirmed": False,
                }
            if at >= pad_end:
                return {
                    "signal": "room_quiet",
                    "waited_ms": round(elapsed * 1000),
                    "quiet_after_edge_ms": round(
                        max(0.0, audible_until - started) * 1000
                    ),
                    "confirmed": True,
                }
            # Padding: keep polling so audio that resumes or a room that
            # drops during the pad is seen on the next pass.
            await sleep(max(0.0, min(_TERMINAL_EGRESS_POLL_SECONDS, pad_end - at)))
            continue
        if at >= deadline:
            return {
                "signal": "tail_elapsed" if goodbye_heard else "unobserved",
                "waited_ms": round(elapsed * 1000),
                "confirmed": False,
            }
        await sleep(max(0.0, min(_TERMINAL_EGRESS_POLL_SECONDS, deadline - at)))


try:
    from .end_call_guard import (
        FLOOR_NAVIGATION,
        FLOOR_QUIET,
        FLOOR_WORDLESS,
        FLOOR_WORDS,
        BASIS_CALLER_STILL_SPEAKING,
        INTENT_TRANSFERRING,
        REFUSAL_TEXT,
        CloseFloorEntry,
        CloseFloorEvidence,
        LocalCloseRefusalRecovery,
        TerminalIntent,
        replace_refused_items,
    )
except ImportError:  # the flat /app import in the pinned worker image
    from end_call_guard import (
        FLOOR_NAVIGATION,
        FLOOR_QUIET,
        FLOOR_WORDLESS,
        FLOOR_WORDS,
        BASIS_CALLER_STILL_SPEAKING,
        INTENT_TRANSFERRING,
        REFUSAL_TEXT,
        CloseFloorEntry,
        CloseFloorEvidence,
        LocalCloseRefusalRecovery,
        TerminalIntent,
        replace_refused_items,
    )


POST_REFUSAL_EVENT = "end_call_guard_post_refusal"


def voicemail_stands(outcome: Any) -> bool:
    """Whether the voicemail observer's verdict is PRESENT authority for a
    hang-up: a proof it would act on, with no confirmed or provisional human
    signal recorded after that proof. The observer's ``classification()``
    latches on the first proof and ignores a person who answers later, so
    it is never read directly for this."""

    if outcome is None:
        return False
    if str(getattr(outcome, "outcome", "") or "") != "would_act":
        return False
    if bool(getattr(outcome, "cut_off_human", False)):
        return False
    if bool(getattr(outcome, "provisional_human_after", False)):
        return False
    return True


async def apply_end_call_post_refusal(
    plan: Any,
    *,
    agent: Any,
    session: Any,
    make_line: Callable[[str], Any],
    make_context: Callable[[list[Any]], Any],
    finished: Callable[[], bool],
    on_done: Callable[[bool], None],
    emit: Callable[..., None],
    caller_busy: Callable[[], bool] = lambda: False,
    on_regenerated: Callable[[str], None] = lambda _speech_id: None,
    on_regeneration_failed: Callable[[str], None] = lambda _error: None,
    spoke: Callable[[str], bool] = lambda _speech_id: True,
    on_regeneration_settled: Callable[[str], None] = lambda _speech_id: None,
    navigation_suspended: Callable[[], bool] = lambda: False,
) -> bool:
    """The guard's post-refusal step on the SDK: the refused call and its
    output leave the agent context, the plan's line stands where they were
    (rendered to the model as an instructions block inside the caller's next
    turn), and the reply is regenerated with tools off when the plan says so.

    Runs from a task scheduled by the synchronous ``function_tools_executed``
    handler, so the SDK has inserted the tool items by the time it reads the
    context. Never raises: a step that cannot be applied leaves the refusal
    standing as it was measured without this step (the model re-attempts on
    a later turn, and the budget bounds that), counted as failed.
    """

    attempt_ids = list(getattr(plan, "attempt_ids", ()) or ())
    try:
        if finished():
            emit(POST_REFUSAL_EVENT, attempt_ids=attempt_ids, applied=False, skipped="call_ending")
            on_done(False)
            return False
        items = list(agent.chat_ctx.items)
        line = str(getattr(plan, "line", "") or "")
        rewritten, index = replace_refused_items(items, attempt_ids, make_line(line) if line else None)
        if index is None:
            emit(POST_REFUSAL_EVENT, attempt_ids=attempt_ids, applied=False, skipped="pair_not_in_context")
            on_done(False)
            return False
        await agent.update_chat_ctx(make_context(rewritten))
        # Re-read right before speaking: the refused generation may have
        # been cut meanwhile, the caller may be speaking or have words held
        # for the next turn (which then answers them), or the call may be
        # ending. The context edit above stands either way.
        withheld = None
        if not bool(plan.regenerate):
            withheld = "held_caller_words"
        elif finished():
            withheld = "call_ending"
        elif navigation_suspended():
            # A recovery reply is speech like any other: not into a menu or a
            # queue. The context edit above stands.
            withheld = "navigating"
        elif plan.cut():
            withheld = "generation_cut"
        elif caller_busy():
            withheld = "caller_busy"
        regenerated = withheld is None
        if regenerated:
            # The context edit above has applied whatever happens here: a
            # generate_reply that raises (a session already closing, a race
            # with a platform ending) is a withheld regeneration, never a
            # failed step. A regeneration that fails LATER, inside the SDK
            # (an LLM error on the handle), is a failed step: the measured
            # mechanism did not reach the caller, and the handle's own
            # completion says so.
            try:
                handle = session.generate_reply(tools=[])
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:  # noqa: BLE001
                # The SDK's own follow-up is already cancelled and no reply
                # replaces it: the measured treatment was not delivered, so
                # this is a FAILED step (unlike the withheld cases, where
                # not speaking is the right thing).
                kind = type(exc).__name__
                emit(
                    POST_REFUSAL_EVENT,
                    attempt_ids=attempt_ids,
                    applied=True,
                    dropped=len(items) - len(rewritten) + (1 if line else 0),
                    line_index=index if line else -1,
                    regenerated=False,
                    regeneration_failed=f"generate_reply:{kind}",
                )
                on_done(True)
                on_regeneration_failed(f"generate_reply:{kind}")
                return True
            else:
                # The regenerated speech's id, so every utterance it produces
                # is attributed to the post-refusal step.
                on_regenerated(str(getattr(handle, "id", "") or ""))
                _settle_regeneration_from(
                    handle,
                    attempt_ids=attempt_ids,
                    on_failed=on_regeneration_failed,
                    emit=emit,
                    spoke=spoke,
                    on_settled=on_regeneration_settled,
                )
        receipt = dict(
            attempt_ids=attempt_ids,
            applied=True,
            dropped=len(items) - len(rewritten) + (1 if line else 0),
            line_index=index if line else -1,
            regenerated=regenerated,
            regeneration_withheld=withheld or "",
        )
        on_done(True)
        # The receipt is diagnostic: a failing emit cannot turn an applied
        # step into a failed one.
        try:
            emit(POST_REFUSAL_EVENT, **receipt)
        except Exception:  # noqa: BLE001
            pass
        return True
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:  # noqa: BLE001 - the refusal stands; the model re-attempts later, bounded
        emit(POST_REFUSAL_EVENT, attempt_ids=attempt_ids, applied=False, error_type=type(exc).__name__)
        try:
            on_done(False)
        except Exception:  # noqa: BLE001
            pass
        return False


def post_refusal_hooks(guard: Any) -> dict[str, Any]:
    """The guard's three post-refusal receipts, bound once here so the
    worker cannot wire one and forget another (a missing failure hook let
    an SDK-failed regeneration count as an applied step)."""

    return {
        "on_done": guard.note_post_refusal,
        "on_regenerated": guard.utterances.note_regeneration,
        "on_regeneration_failed": guard.note_regeneration_failed,
        "on_regeneration_settled": guard.utterances.note_regeneration_settled,
        "spoke": guard.utterances.spoke,
    }


def _settle_regeneration_from(
    handle: Any,
    *,
    attempt_ids: list[str],
    on_failed: Callable[[str], None],
    emit: Callable[..., None],
    spoke: Callable[[str], bool] = lambda _speech_id: True,
    on_settled: Callable[[str], None] = lambda _speech_id: None,
) -> None:
    """Report a regeneration that the SDK later failed: the handle completed
    with an exception, or it completed uninterrupted without ever starting
    to play (a synthesis failure in the pinned SDK leaves no exception and
    no audio). Either way the step's reply never reached the caller. Every
    completion, failed or not, settles the speech so the teardown summary
    knows nothing is pending."""

    add_done = getattr(handle, "add_done_callback", None)
    if not callable(add_done):
        on_settled(str(getattr(handle, "id", "") or ""))
        return

    def _done(done: Any) -> None:
        speech_id = str(getattr(done, "id", "") or "")
        try:
            _report(done, speech_id)
        finally:
            try:
                on_settled(speech_id)
            except Exception:  # noqa: BLE001
                pass

    def _report(done: Any, speech_id: str) -> None:
        try:
            error = done.exception()
        except asyncio.CancelledError:
            # Cancelled with the session (the call ended): not a failure.
            return
        except Exception:  # noqa: BLE001 - a handle without an exception() reads as fine
            error = None
        if error is None:
            if bool(getattr(done, "interrupted", False)):
                return
            if spoke(speech_id):
                return
            kind = "no_audio"
        else:
            kind = type(error).__name__
        try:
            on_failed(kind)
        except Exception:  # noqa: BLE001
            pass
        try:
            emit(POST_REFUSAL_EVENT, attempt_ids=attempt_ids, applied=True, regenerated=False, regeneration_failed=kind)
        except Exception:  # noqa: BLE001
            pass

    try:
        add_done(_done)
    except Exception:  # noqa: BLE001
        return


class _CloseReclaimed(Exception):
    """Caller speech revoked a model's pending close, not a platform ending."""


class WorkerLocalEndCall:
    """Perform the terminal ``end_call`` action inside the media worker.

    The control-plane endpoint intentionally remains a harmless acknowledgement
    for a misrouted request, but an admitted LiveKit model call must execute
    here: propose a final message, wait for its reclaimable playout
    with a bound, then request normal session/room cleanup.  The function-tool
    invocation itself must return before ``session.say`` starts: LiveKit waits
    for a function tool from the current speech handle, and awaiting queued
    playout inside that same tool would deadlock the scheduler.  One shared
    worker task makes repeated or concurrent model calls idempotent and outlives
    cancellation of an individual tool invocation.
    """

    def __init__(
        self,
        *,
        session: Any,
        language: str,
        on_ending: Callable[[], None],
        finish: Callable[[], None],
        record_final_message: Callable[..., None] | None = None,
        commit_terminal_telemetry: Callable[[], None] | None = None,
        on_say: Callable[[str], None] | None = None,
        emit: Callable[..., None] | None = None,
        is_finished: Callable[[], bool] | None = None,
        playout_timeout_seconds: float = 15.0,
        drain: Callable[[], Awaitable[Mapping[str, object]]] | None = None,
        drain_timeout_seconds: float = 2.0,
        speaks_text: bool = True,
        agent_audible_until: Callable[[], float | None] | None = None,
        agent_speaking: Callable[[], bool] | None = None,
        guard: Any | None = None,
        on_reclaimed: Callable[[], object] | None = None,
        on_proposed: Callable[[Any, str | None], None] | None = None,
        close_floor_budget: Callable[[], float | None] | None = None,
        caller_words: Callable[[str], bool] | None = None,
        terminal_intent: TerminalIntent | None = None,
        refusal_recovery: LocalCloseRefusalRecovery | None = None,
        navigation_suspended: Callable[[], bool] | None = None,
        closing_receipt: Callable[..., None] | None = None,
        tool_speech_claim: Callable[[str], Any] | None = None,
    ) -> None:
        if playout_timeout_seconds <= 0:
            raise ValueError("end_call playout timeout must be positive")
        if drain_timeout_seconds <= 0:
            raise ValueError("end_call drain timeout must be positive")
        self._tool_speech_factory = tool_speech_claim
        self._tool_speech = None
        self._tool_speech_ready = asyncio.Event()
        self._tool_speech_error = ""
        self._closing_receipt = closing_receipt
        self._receipt_requests = 0
        self._receipt_proposals = 0
        self._receipt_proposal = ""
        self._receipt_committed = False
        self._receipt_owner = "model"
        self._session = session
        # Whether the call is waiting in a phone menu or a hold queue. Read at
        # the tool's entry and again under the lock, never cached: the model's
        # own hang-up is refused then, and only the navigation bounds end it.
        self._navigation_suspended = navigation_suspended or (lambda: False)
        self._drain = drain
        self._drain_timeout_seconds = drain_timeout_seconds
        self._language = language.strip().lower()
        self._on_ending = on_ending
        self._finish = finish
        self._record_final_message = record_final_message
        self._commit_terminal_telemetry = commit_terminal_telemetry
        self._on_say = on_say
        self._on_reclaimed = on_reclaimed
        self._on_proposed = on_proposed
        self._emit = emit
        self._is_finished = is_finished or (lambda: False)
        self._playout_timeout_seconds = playout_timeout_seconds
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[dict[str, object]] | None = None
        # Whether this session can speak a line of TEXT. A voice-to-voice
        # session has no TTS, and LiveKit refuses ``say(text)`` on it: every
        # such goodbye failed with ``RuntimeError`` and the leg was torn down
        # at once. There the model says its own goodbye before (or
        # with) the tool, and this action only lets it finish.
        self._speaks_text = bool(speaks_text)
        self._agent_audible_until = agent_audible_until or (lambda: None)
        self._agent_speaking = agent_speaking or (lambda: False)
        #: The generation that carried the tool call, captured when the model
        #: invoked it. On a voice-to-voice engine a call made while the model
        #: is still talking joins that audio's generation, so its playout IS
        #: the goodbye; a call made after the audio ended is its own empty
        #: generation, over as soon as it is delivered.
        self._live_handle: Any = None
        #: How long the room had been quiet when the tool arrived, in ms, or
        #: ``None`` when the agent had never been heard. A measurement, not a
        #: claim about what was said.
        self._quiet_ms_at_tool: int | None = None
        #: Whether the room was hearing the agent when the tool arrived: the
        #: call then joined a goodbye still in flight, and that goodbye is
        #: drained after its generation completes whatever the room edge says
        #: at that instant -- the SDK marks a turn done at LOCAL playout end,
        #: ~140 ms before the room hears its last frame.
        self._speech_at_tool = False
        #: A platform-initiated ending's own ``finish`` (``terminate``), which
        #: stamps its own reason. Fixed by whichever ending schedules the task
        #: first: a later model ``end_call`` only joins it.
        self._finish_override: Callable[[], None] | None = None
        #: On a voice-to-voice session a platform-initiated ending has no
        #: model turn to wait on, so the model is asked for the goodbye.
        self._farewell_instruction = ""
        #: The end_call guard (``end_call_guard.EndCallGuard``), armed by the
        #: worker on managed-PSTN cascade calls only. It is asked BEFORE
        #: anything irreversible: a refusal is a tool output telling the
        #: model to keep helping, and nothing below runs.
        self._guard = guard
        # Native audio has no trusted caller transcript at this seam. Protect
        # the caller's floor acoustically without claiming a semantic verdict.
        self._caller_speaking = False
        self._caller_floor_revision = 0
        self._accepted_close_attempts: dict[str, dict[str, str]] = {}
        self._close_result_context: dict[str, str] = {}
        self._sdk_caller_speaking = False
        self._acoustic_caller_speaking = False
        self._native_close_reclaimed = asyncio.Event()
        self._close_committed = False
        self._reclaim_notified = False
        #: The measured first-word budget of this call's transcript source,
        #: read per attempt (``None``: unmeasured, and the floor is raw VAD
        #: exactly as before). Only a text session reads words at this seam.
        self._close_floor_budget = close_floor_budget or (lambda: None)
        #: The words-based floor (``end_call_guard.CloseFloorEvidence``); the
        #: filter is the call's own backchannel rule, so a listening noise is
        #: not words here either.
        self._floor = CloseFloorEvidence(
            is_words=caller_words or (lambda text: bool(text.strip())), clock=time.monotonic
        )
        #: Broadcast on caller words: every waiter holds the event current
        #: when it began, and a signal replaces it.
        self._floor_changed = asyncio.Event()
        #: The budget of the close in flight when it was accepted on a
        #: measured source; ``None`` for an unmeasured one or no close.
        self._close_budget: float | None = None
        #: The measured close's farewell handle while it plays.
        self._farewell_speech: Any = None
        self._intent = terminal_intent if terminal_intent is not None else TerminalIntent()
        self._refusal_recovery = refusal_recovery
        # The owner observes its own caller floor on every native entrypoint,
        # including the public demo. SDK-light callers may feed the same seam
        # directly when their session does not expose an event emitter.
        subscribe = getattr(session, "on", None)
        if callable(subscribe):
            subscribe("user_state_changed")(self._on_caller_state)
            subscribe("function_tools_executed")(self._on_native_tool_output)
            if self._speaks_text:
                subscribe("user_input_transcribed")(self._on_caller_transcript)
                subscribe("conversation_item_added")(self._on_conversation_item)

    def _on_native_tool_output(self, event: Any) -> None:
        """Describe the close and request speech only for an unmet frozen claim.

        Other outputs retain the legacy revocable status and no-reply flag.
        """
        if self._authoritative_close():
            return
        for call, output in zip(event.function_calls, event.function_call_outputs):
            if getattr(call, "name", "") != "end_call" or getattr(output, "is_error", False):
                continue
            attempt_id = str(getattr(call, "call_id", "") or "")
            accepting = next((key for key, value in self._accepted_close_attempts.items()
                              if value is self._close_result_context), None)
            context = self._accepted_close_attempts.pop(attempt_id, None)
            if context is None:
                continue  # Refusals and unrelated/unknown attempts retain their output.
            phase = "committed" if self._close_committed else context["phase"]
            output.output = {
                "pending": "The close request is pending and revocable; the call has not committed to ending.",
                "reclaimed": "The close request was canceled after caller-side activity. The call is still open; this request will not close it.",
                "committed": "The call has committed to ending. Do not speak or invoke any more tools.",
            }[phase]
            if phase != "committed":
                output.output += (
                    " Listen for the caller and answer any continuation. If they need nothing else, "
                    "invoke end_call again. A canceled request does not count as ending the call, "
                    "even if an earlier instruction said to use end_call once. Do not announce this internal status. "
                    "Do not say anything just to acknowledge this tool result."
                )
            output.reply_required = False
            if (self._tool_speech_factory is not None and not self._speaks_text
                    and self._receipt_owner == "model" and phase == "pending"
                    and context is self._close_result_context and attempt_id == accepting
                    and not self._tool_speech_ready.is_set()):
                try:
                    self._tool_speech = self._tool_speech_factory(attempt_id)
                    output.reply_required = self._tool_speech.reply_required
                    if output.reply_required is True:
                        output.output = (
                            "The close request is pending and revocable; the call has not committed to ending. "
                            "This close still requires its first spoken closing response. Say one brief, natural "
                            "closing response in the conversation's language now. Then wait for this pending "
                            "close to finish; do not request another close while it is pending. If the caller "
                            "continues, stop the closing response and answer them. A canceled request leaves "
                            "the call open; request a new close only when the caller again needs nothing else. "
                            "A canceled request does not count as ending the call, even if an earlier "
                            "instruction said to use end_call once. Do not announce this internal status."
                        )
                except Exception as exc:
                    self._tool_speech_error = type(exc).__name__
                self._tool_speech_ready.set()
            if self._emit is not None:
                self._emit("end_call_result_context", basis=phase)

    def _on_caller_state(self, event: Any) -> None:
        self.note_caller_speaking(str(getattr(event, "new_state", "")) == "speaking")

    def _on_caller_transcript(self, event: Any) -> None:
        self.note_caller_transcript(str(getattr(event, "transcript", "") or ""))

    def _on_conversation_item(self, event: Any) -> None:
        # A committed caller turn: the words before it are the model's to
        # answer, not a claim on the floor by the run that is open now.
        if str(getattr(getattr(event, "item", None), "role", "") or "") == "user":
            self._floor.note_turn_committed()

    def note_caller_transcript(self, text: str) -> None:
        """Admitted STT words revoke a stale close even when VAD missed speech."""
        if not text.strip():
            return
        words = self._floor.note_transcript(text)
        if words:
            self._signal_floor()
        self._caller_floor_revision += 1
        if self._task is not None and not self._close_committed and not self._authoritative_close():
            if self._close_budget is not None:
                if not words:
                    # A measured source: a listening noise is not a reclaim. Its
                    # VAD run, if any, is held by the terminal settlement instead.
                    return
                # Synchronously, before these words can reach the runtime as a
                # committed turn: the pinned SDK drops ("skipping reply to user
                # input") a turn completed while the current speech refuses
                # interruptions, and the measured farewell refuses them.
                restore_speech_interruptions(self._farewell_speech)
            self._close_result_context["phase"] = "reclaimed"
            self._native_close_reclaimed.set()
            self._notify_reclaimed()

    def _signal_floor(self) -> None:
        event, self._floor_changed = self._floor_changed, asyncio.Event()
        event.set()

    async def wait_for_floor(self, timeout: float) -> None:
        """Return when caller words arrive, when an ending overtakes the wait
        (``terminate``, the arbiter's finished flag, another close taking the
        task), or after ``timeout`` seconds. One call, however long: the
        guard counts each call as one defer."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        event = self._floor_changed
        task = self._task
        while not event.is_set():
            if self._authoritative_close() or self._task is not task:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                # The finished flag has no event to wait on: re-read it.
                await asyncio.wait_for(event.wait(), timeout=min(remaining, CLOSE_FLOOR_POLL_SECONDS))
            except TimeoutError:
                pass

    def caller_counts_as_words(self, text: str) -> bool:
        """Whether a transcript is caller evidence for the guard's ledger: any
        non-empty text on an unmeasured source (as before), the close floor's
        own words reading (backchannels filtered) on a measured one."""

        if not str(text or "").strip():
            return False
        if not self.close_floor_measured():
            return True
        return self._floor.is_words(str(text))

    def close_floor_measured(self) -> bool:
        """Whether this call's transcript source has a measured first-word
        budget right now, so a wordless run is not caller evidence."""

        return self._speaks_text and self._measured_budget() is not None

    def _measured_budget(self) -> float | None:
        try:
            budget = self._close_floor_budget()
        except Exception:  # noqa: BLE001 - an unreadable capability is an unmeasured one
            return None
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            return None
        return float(budget) if math.isfinite(budget) and budget > 0 else None

    def _current_entry(self) -> CloseFloorEntry | None:
        """The measured entry of the attempt evaluating in this task, if any."""

        current = _CLOSE_FLOOR_ENTRY.get()
        if current is None or current[0] is not self:
            return None
        return current[1]

    def guard_user_speaking(self, raw: bool) -> bool:
        """The guard's open-run veto, read from the same floor as the close.

        ``raw`` (the session's VAD state) on an unmeasured source, and outside
        an attempt; on a measured source, a run counts only with words behind it.
        """

        if self._current_entry() is None:
            return bool(raw)
        return self._floor.run_has_words()

    def guard_wordless_remaining(self) -> float:
        """Seconds left in this task's attempt's wait for words, or 0."""

        entry = self._current_entry()
        if entry is None or entry.has_words():
            return 0.0
        return entry.remaining(time.monotonic())

    def _notify_reclaimed(self) -> None:
        """Notify the proposal once; its originating speech owns settlement."""
        if self._reclaim_notified:
            return
        self._reclaim_notified = True
        if self._on_reclaimed is not None:
            try:
                self._on_reclaimed()
            except Exception:
                pass

    def _authoritative_close(self) -> bool:
        # Some platform/remote endings use the arbiter directly rather than
        # terminate(). Its finished flag has the same authority as an override.
        return self._finish_override is not None or self._is_finished()

    def note_caller_speaking(self, speaking: bool) -> None:
        """Withdraw a pending model close when the caller takes the floor.

        Loop-affine, like ``terminate``. Platform endings are authoritative;
        taking the floor cannot extend a funded deadline or operator close.
        """

        self._sdk_caller_speaking = bool(speaking)
        self._update_caller_floor()

    def note_acoustic_caller_speaking(self, speaking: bool) -> None:
        self._acoustic_caller_speaking = bool(speaking)
        self._update_caller_floor()

    def _update_caller_floor(self) -> None:
        was_speaking = self._caller_speaking
        self._caller_speaking = self._sdk_caller_speaking or self._acoustic_caller_speaking
        if self._caller_speaking != was_speaking:
            self._floor.note_vad(self._caller_speaking)
        if self._caller_speaking and not was_speaking:
            self._caller_floor_revision += 1
        if self._caller_speaking and self._task is not None and not self._close_committed and not self._authoritative_close():
            if self._close_budget is not None:
                # A measured source: a VAD run alone never revokes the close.
                # Words do (``note_caller_transcript``); a wordless run holds
                # the terminal commit for the budget instead.
                return
            self._close_result_context["phase"] = "reclaimed"
            self._native_close_reclaimed.set()
            if self._speaks_text:
                # Capture the reclaim before waiter cleanup yields. Telemetry
                # settles after the originating speech, not at this edge.
                self._notify_reclaimed()

    def _native_refusal(self) -> dict[str, object]:
        return {
            "ok": False,
            "status": "refused",
            "playout": "not_started",
            "basis": "caller_took_floor",
            "text": "The caller is speaking or interrupted the proposed closing. Keep the call open and listen. Do not repeat the goodbye or end the call before responding to the caller.",
            "words_held": False,
        }

    def _floor_refusal(self, attempt_id: str, *, floor: str) -> dict[str, object]:
        """A refusal on the caller's floor, handed to the worker's recovery so
        the SDK's automatic follow-up cannot answer it with another goodbye."""

        if self._refusal_recovery is not None:
            try:
                self._refusal_recovery.note_refusal(attempt_id, floor=floor)
            except Exception:  # noqa: BLE001 - recovery is best effort; the refusal stands
                pass
        return self._native_refusal()

    def caller_floor_refusal(self, attempt_id: str) -> dict[str, object]:
        """The caller took the floor before this attempt reached the owner
        (the end_call tool's confirmation drain): refused like the owner's own."""

        self._receipt_requests += 1
        request_id, owner = f"r{self._receipt_requests}", "model" if attempt_id else "platform"
        self._receipt("request", request_id=request_id, proposal_id="", owner=owner)
        result = self._floor_refusal(attempt_id, floor=FLOOR_WORDS)
        self._receipt("refused", request_id=request_id, proposal_id="", owner=owner, reason="caller_speech")
        return result

    def _navigation_refusal(self, attempt_id: str) -> dict[str, object]:
        """The model's hang-up refused because the call is in a menu or a queue.

        Handed to the same recovery as a floor refusal, so the SDK's automatic
        follow-up is cancelled: the model must not answer a refusal by speaking
        into a recorded menu, or by saying goodbye again.
        """

        if self._refusal_recovery is not None:
            try:
                self._refusal_recovery.note_refusal(attempt_id, floor=FLOOR_NAVIGATION)
            except Exception:  # noqa: BLE001 - recovery is best effort; the refusal stands
                pass
        return {
            "ok": False,
            "status": "refused",
            "playout": "not_started",
            "basis": "navigation",
            "text": "The call is waiting in a phone menu or on hold, so it cannot be ended now. Do not say goodbye and do not speak; keep waiting.",
            "words_held": False,
        }

    @staticmethod
    def _transfer_refusal() -> dict[str, object]:
        return {
            "ok": False,
            "status": "refused",
            "playout": "not_started",
            "basis": "transfer_in_progress",
            "text": "The call is being transferred to a person, so it cannot be ended now. Do not say goodbye.",
            "words_held": False,
        }

    async def _await_entry_floor(
        self, entry: CloseFloorEntry, interrupted: Callable[[], bool] | None = None
    ) -> str:
        """Wait, bounded once by the source's budget, for words behind a
        wordless run; the floor's reading when the wait ends. ``interrupted``
        reads the originating generation's cut, which opens the window too."""

        started = time.monotonic()
        while True:
            if interrupted is not None:
                try:
                    interrupted()
                except Exception:  # noqa: BLE001 - an unreadable handle opens nothing
                    pass
            now = time.monotonic()
            state = entry.state(now)
            if state != FLOOR_WORDLESS or self._task is not None or self._is_finished():
                break
            await self.wait_for_floor(entry.remaining(now))
        if self._emit is not None and time.monotonic() - started > 0.001:
            try:
                self._emit(
                    "end_call_floor_deferred",
                    floor=state,
                    waited_ms=round((time.monotonic() - started) * 1000),
                    budget_ms=round(entry.budget * 1000),
                )
            except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                pass
        return state

    @property
    def ending_scheduled(self) -> bool:
        """Whether an ending (any owner) already holds the terminal task."""

        return self._task is not None

    def _receipt(self, kind: str, **fields: Any) -> None:
        if self._closing_receipt is not None:
            try:
                self._closing_receipt(kind, proposal_id=fields.pop("proposal_id", self._receipt_proposal),
                                      owner=fields.pop("owner", "platform" if self._finish_override is not None else self._receipt_owner), **fields)
            except Exception:
                pass

    def _receipt_commit(self) -> None:
        if not self._receipt_committed:
            self._receipt_committed = True
            try:
                if self._is_finished() and self._finish_override is None:
                    # Cleanup after an independent ending did not commit the
                    # model's proposal or cause that disconnection.
                    self._receipt("owner_changed", owner="transport", reason="already_finished")
                else:
                    self._receipt("owner_commit")
            except Exception:
                # A missing diagnostic must not interrupt shutdown.
                pass

    async def __call__(self, raw_arguments: dict[str, object], **kwargs: Any) -> dict[str, object]:
        self._receipt_requests += 1
        request_id = f"r{self._receipt_requests}"
        request_owner = "model" if kwargs.get("attempt_id") else "platform"
        self._receipt("request", request_id=request_id, proposal_id="", owner=request_owner)
        try:
            result = await self._request(raw_arguments, receipt_request_id=request_id, receipt_request_owner=request_owner, **kwargs)
        except asyncio.CancelledError:
            self._receipt("refused", request_id=request_id, proposal_id="", owner=request_owner, reason="interrupted")
            raise
        except Exception:
            self._receipt("failed", request_id=request_id, proposal_id="", owner=request_owner, reason="runtime_error")
            raise
        if result.get("status") != "termination_scheduled":
            status = result.get("status")
            basis = str(result.get("basis") or "")
            reason = ("already_finished" if status == "already_finished" else
                      "interrupted" if status == "aborted" else
                      {"navigation": "navigation", "transfer_in_progress": "transfer",
                       "caller_took_floor": "caller_speech", "caller_words_pending": "caller_speech",
                       "caller_still_speaking": "caller_speech"}.get(basis, "guard"))
            self._receipt("already_finished" if status == "already_finished" else "refused",
                          request_id=request_id, proposal_id="", owner=request_owner, reason=reason)
        return result

    async def _request(
        self,
        raw_arguments: dict[str, object],
        *,
        receipt_request_id: str,
        receipt_request_owner: str,
        attempt_id: str = "",
        speech_id: str = "",
        speech_interrupted: Callable[[], bool] | None = None,
        origin_speech: Any = None,
        user_item_id: str | None = None,
    ) -> dict[str, object]:
        floor_revision = self._caller_floor_revision
        entry: CloseFloorEntry | None = None
        if self._task is None and not self._is_finished():
            if self._intent.state == INTENT_TRANSFERRING:
                return self._transfer_refusal()
            if self._navigation_suspended():
                return self._navigation_refusal(attempt_id)
            budget = self._measured_budget() if self._speaks_text else None
            if budget is None:
                # Unmeasured: raw VAD, exactly as before.
                if self._caller_speaking:
                    return self._floor_refusal(attempt_id, floor="vad")
            else:
                # Measured: words refuse; a wordless run defers, bounded.
                entry = CloseFloorEntry(self._floor, budget=budget, now=time.monotonic())
                if entry.has_words():
                    return self._floor_refusal(attempt_id, floor=FLOOR_WORDS)
        interrupted_read = speech_interrupted
        if entry is not None and speech_interrupted is not None:
            measured_entry = entry

            def interrupted_read() -> bool:
                # The ORIGINATING generation's cut, on a measured source: it
                # aborts the attempt only with words behind it. A duration-only
                # interruption with none is a wordless onset for the deferral.
                if not speech_interrupted():
                    return False
                measured_entry.note_wordless_interruption(time.monotonic())
                return measured_entry.has_words()

        entry_token = _CLOSE_FLOOR_ENTRY.set((self, entry))
        try:
            if self._guard is not None and self._speaks_text and self._task is None and not self._is_finished():
                # Outside the lock: the guard may await the caller's run end, and
                # ``terminate()`` (synchronous, lock-free) may claim the task
                # meanwhile; the checks inside the lock below re-read both.
                utterances = getattr(self._guard, "utterances", None)
                if utterances is not None and speech_id:
                    # The generation that carried this attempt: any text it spoke
                    # was spoken before the guard could decide.
                    utterances.note_attempt_speech(speech_id)
                try:
                    outcome = await self._guard.evaluate(
                        attempt_id=attempt_id, interrupted=interrupted_read
                    )
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:  # noqa: BLE001 - the check is unavailable: the request is honoured, and said so
                    if self._emit is not None:
                        try:
                            self._emit(
                                "end_call_guard_error",
                                attempt_id=attempt_id,
                                error_type=type(exc).__name__,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    outcome = None
                if self._task is None and self._navigation_suspended():
                    # The guard can wait for seconds, and the call may have
                    # entered a menu or a queue meanwhile. What a queue does to
                    # the caller's floor (a hot VAD, words in its transcript)
                    # is not the caller taking it: the refusal is THIS one,
                    # whatever the guard concluded.
                    return self._navigation_refusal(attempt_id)
                if outcome is not None and self._guard.was_aborted(outcome):
                    # The generation that carried the tool was cut by the caller
                    # while the guard deliberated: nothing is decided or
                    # scheduled, and the SDK generates no follow-up for it.
                    return {
                        "ok": False,
                        "status": "aborted",
                        "playout": "not_started",
                        "basis": outcome.basis,
                    }
                if outcome is not None and self._guard.refuses(outcome):
                    return {
                        "ok": False,
                        "status": "refused",
                        "playout": "not_started",
                        "basis": outcome.basis,
                        "text": outcome.text,
                        "words_held": outcome.words_held,
                    }
            if entry is not None and self._task is None and not self._is_finished():
                # The guard-wait check, measured: whatever the guard spent of
                # the window, the rest is waited out here for words.
                await self._await_entry_floor(entry, interrupted_read)
                if self._task is None and self._navigation_suspended():
                    return self._navigation_refusal(attempt_id)
        finally:
            _CLOSE_FLOOR_ENTRY.reset(entry_token)
        async with self._lock:
            task = self._task
            if task is None:
                if self._is_finished():
                    return {
                        "ok": True,
                        "status": "already_finished",
                        "playout": "not_started",
                    }
                if self._navigation_suspended():
                    # Re-read under the lock, FIRST: the waits above can run for
                    # seconds, the call may have entered a menu or a queue
                    # meanwhile, and a queue trips both floor readings below.
                    return self._navigation_refusal(attempt_id)
                if self._speaks_text and interrupted_read is not None and interrupted_read():
                    # A shielded tool can arrive after the caller cut its
                    # generation, even where no semantic guard is armed.
                    return {"ok": False, "status": "aborted", "playout": "not_started",
                            "basis": "originating_speech_interrupted"}
                if entry is None:
                    if self._caller_speaking or self._caller_floor_revision != floor_revision:
                        return self._floor_refusal(attempt_id, floor="vad")
                elif entry.has_words():
                    return self._floor_refusal(attempt_id, floor=FLOOR_WORDS)
                if not self._intent.claim_close():
                    return self._transfer_refusal()
                self._close_budget = entry.budget if entry is not None else None
                # Caller input stays open until playout and reclaim settle.
                # The background task cannot run until this function yields,
                # which is precisely what lets the current tool speech handle
                # complete and the final ``say`` be scheduled safely.
                self._receipt_owner = receipt_request_owner
                self._receipt_proposals += 1
                self._receipt_proposal = f"p{self._receipt_proposals}"
                self._receipt_committed = False
                self._receipt("accepted", request_id=receipt_request_id)
                self._close_committed = False
                self._reclaim_notified = False
                self._native_close_reclaimed.clear()
                self._close_result_context = {"phase": "pending"}
                if not self._speaks_text:
                    self._capture_live_goodbye()
                elif self._on_proposed is not None:
                    # Capture this proposal's telemetry identities before any
                    # playout/new caller callback can run. Duplicate requests
                    # join the existing proposal without replacing its set.
                    try:
                        self._on_proposed(origin_speech, user_item_id)
                    except Exception:
                        pass
                task = asyncio.create_task(self._execute(dict(raw_arguments)))
                self._task = task
                if self._emit is not None:
                    try:
                        self._emit("end_call_proposed", basis="cascade_goodbye" if self._speaks_text else "native_goodbye",
                                   source="model_tool" if attempt_id else "worker_fallback")
                    except Exception:
                        pass
            else:
                self._receipt("joined", request_id=receipt_request_id)
            if attempt_id:
                self._accepted_close_attempts[attempt_id] = self._close_result_context
                while len(self._accepted_close_attempts) > 128:
                    del self._accepted_close_attempts[next(iter(self._accepted_close_attempts))]
        return {
            "ok": True,
            "status": "termination_scheduled",
            "playout": "pending",
        }

    def terminate(
        self,
        *,
        finish: Callable[[], None],
        message: str = "",
        farewell_instruction: str = "",
    ) -> bool:
        """End the call on the PLATFORM's decision, with a goodbye; never waits.

        The worker's own rules call this, with a ``finish`` that
        stamps their own reason. True when this scheduled the ending; False
        when an authoritative ending was already under way or the call has
        finished. A pending native model close can be claimed by a platform
        limit; it keeps the same task, with the platform's reason. A model
        ``end_call`` arriving later only joins that task.

        Synchronous and never awaiting the task: a rule polled from the
        worker's tick must not park on a goodbye, and nothing between the
        check and the ``create_task`` yields, so it cannot race ``__call__``.

        ``message`` is the fixed goodbye a text session speaks (empty: the
        per-language fallback). ``farewell_instruction`` is what a
        voice-to-voice session's model is asked to say instead, bounded by the
        same playout limit; the call ends whatever it does.
        """

        if self._is_finished():
            return False
        if self._task is not None:
            # A model close is only a proposal until caller reclaim is
            # settled. A platform limit may claim that SAME task meanwhile.
            if self._finish_override is not None or self._close_committed or self._task.done():
                return False
            self._intent.force_close()
            self._finish_override = finish
            self._receipt("owner_changed", reason="platform_rule")
            # A held terminal commit re-reads its authority on its own poll.
            self._signal_floor()
            try:
                self._on_ending()
            except Exception:
                pass
            return True
        try:
            self._on_ending()
        except Exception:
            pass
        # The platform's ending overrides a transfer or a close in flight, and
        # nothing about the caller's floor delays it.
        self._intent.force_close()
        self._close_budget = None
        self._finish_override = finish
        self._receipt_proposals += 1
        self._receipt_proposal = f"p{self._receipt_proposals}"
        self._receipt_committed = False
        self._receipt("accepted", reason="platform_rule")
        if not self._speaks_text:
            self._farewell_instruction = str(farewell_instruction or "")
        self._task = asyncio.create_task(
            self._execute({"message": message} if message else {})
        )
        # An attempt waiting on the caller's floor stops waiting at once.
        self._signal_floor()
        return True

    async def _await_cascade_close(self, awaitable: Awaitable[Any], *, timeout: float | None = None) -> Any:
        """Cancel only the waiter, never the terminal owner, when speech resumes."""
        pending = asyncio.ensure_future(awaitable)
        reclaimed = asyncio.create_task(self._native_close_reclaimed.wait())
        started = time.monotonic()
        try:
            await asyncio.wait((pending, reclaimed), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if self._native_close_reclaimed.is_set() and not self._authoritative_close():
                raise _CloseReclaimed
            remaining = None if timeout is None else max(0, timeout - (time.monotonic() - started))
            return await asyncio.wait_for(pending, timeout=remaining)
        finally:
            for task in (pending, reclaimed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pending, reclaimed, return_exceptions=True)

    async def _settle_close_floor(
        self, playout_ended_at: float, budget: float, farewell_started_at: float | None = None
    ) -> None:
        """Hold the terminal commit while a wordless run may still owe words.

        Measured sources only. The window opens at the later of the farewell's
        playout end and the first wordless onset in its final budget-length
        (``CloseFloorEvidence.settlement_start``), runs one budget, and is
        never extended by a later edge. Words inside it reclaim; its end
        commits; a quiet room commits at once. A platform ending (``terminate``
        or the arbiter's finished flag) overtakes it within one poll.
        """

        started = time.monotonic()
        start: float | None = None
        outcome = FLOOR_QUIET
        try:
            while True:
                if self._authoritative_close():
                    return
                if self._native_close_reclaimed.is_set():
                    outcome = FLOOR_WORDS
                    raise _CloseReclaimed
                if start is None:
                    start = self._floor.settlement_start(
                        playout_end=playout_ended_at, budget=budget, farewell_start=farewell_started_at
                    )
                    if start is None:
                        return
                    outcome = FLOOR_WORDLESS
                remaining = start + budget - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    await asyncio.wait_for(
                        self._native_close_reclaimed.wait(),
                        timeout=min(remaining, CLOSE_FLOOR_POLL_SECONDS),
                    )
                except TimeoutError:
                    pass
        finally:
            if self._emit is not None and outcome != FLOOR_QUIET:
                try:
                    self._emit(
                        "end_call_floor_settled",
                        floor=outcome,
                        waited_ms=round((time.monotonic() - started) * 1000),
                        budget_ms=round(budget * 1000),
                    )
                except Exception:  # noqa: BLE001 - a diagnostic cannot break media
                    pass

    async def _settle_reclaimable_live_goodbye(self) -> tuple[str, dict[str, object] | None, str, str]:
        """Race playout against caller activity without cancelling the owner.

        Only the bounded playout waiter is cancelled on reclaim. Cancelling
        the terminal owner itself still runs its mandatory cleanup.
        """

        deadline = time.monotonic() + self._playout_timeout_seconds
        settlement = asyncio.create_task(self._settle_live_goodbye())
        reclaimed = asyncio.create_task(self._native_close_reclaimed.wait())
        try:
            done, _ = await asyncio.wait((settlement, reclaimed), return_when=asyncio.FIRST_COMPLETED)
            if reclaimed in done and not self._authoritative_close():
                return "reclaimed", None, "", ""
            result = await settlement
            remaining = deadline - time.monotonic()
            if not self._authoritative_close() and remaining > 0:
                # Even a tool-only generation after an already played farewell
                # needs a short opportunity for the caller to take the floor.
                try:
                    await asyncio.wait_for(self._native_close_reclaimed.wait(), timeout=min(0.75, remaining))
                except TimeoutError:
                    pass
                else:
                    if not self._authoritative_close():
                        return "reclaimed", None, "", ""
            # Freeze the lease decision BEFORE joining cancelled child waiters.
            # That join yields; speech arriving there is after the deadline.
            self._close_committed = True
            self._receipt_commit()
            return result
        finally:
            for task in (settlement, reclaimed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(settlement, reclaimed, return_exceptions=True)

    async def wait_for_completion(self) -> dict[str, object] | None:
        """Await the worker-owned terminal flow without exposing it to the model."""

        task = self._task
        return await asyncio.shield(task) if task is not None else None

    def completion_was_cancelled(self) -> bool:
        """Whether the owned task finished by cancellation after its ``finally``.

        This deliberately answers only about the worker-owned task. A waiter
        can be cancelled independently while this task is still running because
        ``wait_for_completion`` shields it. The production teardown uses that
        distinction to settle an already-committed cut goodbye instead of
        mistaking it for an outer worker cancellation.
        """

        task = self._task
        return task is not None and task.done() and task.cancelled()

    def _capture_live_goodbye(self) -> None:
        """Remember the tool's own generation and how quiet the room was.

        Read synchronously inside the tool invocation, where the scheduler's
        current speech is still the generation that emitted the call.
        """

        try:
            self._live_handle = getattr(self._session, "current_speech", None)
        except Exception:
            self._live_handle = None
        try:
            edge = self._agent_audible_until()
        except Exception:
            edge = None
        if isinstance(edge, (int, float)) and not isinstance(edge, bool) and math.isfinite(edge):
            self._quiet_ms_at_tool = max(0, round((time.monotonic() - float(edge)) * 1000))
        self._speech_at_tool = self._agent_audible_now()

    def _agent_audible_now(self) -> bool:
        """Whether the room is hearing the agent at this instant.

        The runtime's own state, or the sidecar's edge within the drain's own
        freshness window -- the drain can only confirm a tail it saw arrive.
        A reader that fails answers "audible", the side that keeps the goodbye.
        """

        try:
            if bool(self._agent_speaking()):
                return True
        except Exception:
            return True
        try:
            edge = self._agent_audible_until()
        except Exception:
            return True
        if not isinstance(edge, (int, float)) or isinstance(edge, bool):
            return False
        return math.isfinite(edge) and time.monotonic() - float(edge) <= LIVE_GOODBYE_FRESH_SECONDS

    async def _settle_live_goodbye(self) -> tuple[str, dict[str, object] | None, str, str]:
        """Let a voice-to-voice goodbye finish, bounded; never speak one.

        Returns ``(playout, egress, error_type, error_stage)``:

        * ``drained``: the room was still hearing the agent once the tool's
          generation completed, and the bounded drain ran from there;
        * ``already_quiet``: the goodbye had already played, the common case
          on GPT-Live, where the audio and a later tool call are separate
          generations -- the call ends at once, as it always did;
        * ``timeout``: the tool's generation did not complete inside the
          playout bound; the call ends anyway;
        * ``interrupted``: the generation was cut before it finished;
        * ``failed``: waiting on it raised, or the SDK recorded a failed
          generation on the handle; the call ends anyway.

        Nothing here generates speech. The provider's own reply to the tool
        result, if it starts, is cut by the shutdown that follows.
        """

        if self._tool_speech_factory is not None and self._receipt_owner == "model":
            try:
                async with asyncio.timeout(self._playout_timeout_seconds):
                    await self._tool_speech_ready.wait()
                    claim = self._tool_speech
                    if claim is None:
                        return "failed", None, self._tool_speech_error or "RuntimeError", "tool_speech_binding"
                    await claim.wait_for_playout()
                    if claim.interrupted:
                        return "interrupted", None, "", ""
                    egress = await self._drain_egress() if self._drain is not None else None
                    return "drained", egress, "", ""
            except TimeoutError:
                return "timeout", None, "", ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return "failed", None, type(exc).__name__, "tool_speech_wait"

        if self._farewell_instruction:
            # A platform-initiated ending: no model turn carried it, so the
            # model is asked for the goodbye, with no tools. Its handle is the
            # one waited on, and a goodbye is expected, so it is drained.
            try:
                self._live_handle = self._session.generate_reply(
                    instructions=self._farewell_instruction,
                    tool_choice="none",
                )
            except Exception as exc:
                return "failed", None, type(exc).__name__, "generate_reply"
            self._speech_at_tool = True
        # ONE budget for the wait and the drain together: the call ends inside
        # the playout bound, not the bound plus a drain started at its edge.
        deadline = time.monotonic() + self._playout_timeout_seconds
        handle = self._live_handle
        if handle is not None:
            try:
                await asyncio.wait_for(
                    handle.wait_for_playout(), timeout=self._playout_timeout_seconds
                )
            except TimeoutError:
                return "timeout", None, "", ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return "failed", None, type(exc).__name__, "wait_for_playout"
            # Awaiting a handle never raises for a failed GENERATION: the SDK
            # keeps that error on the handle, and a cut handle completes
            # normally too. Neither is a goodbye that was heard to its end.
            try:
                probe = getattr(handle, "exception", None)
                speech_error = probe() if callable(probe) else None
            except Exception as exc:
                speech_error = exc
            if speech_error is not None:
                return "failed", None, type(speech_error).__name__, "speech_exception"
            try:
                cut = bool(getattr(handle, "interrupted", False))
            except Exception:
                cut = True
            if cut:
                return "interrupted", None, "", ""
        # Exit at once ONLY when the room had already gone quiet before the
        # tool arrived and is still quiet now: a tool-only generation after a
        # goodbye that finished -- the production shape on GPT-Live, where the
        # call came ~2.5 s after the goodbye's last frame. A goodbye that was
        # still playing at the tool is drained even if the room edge has not
        # caught up with the local playout end yet.
        if not self._speech_at_tool and not self._agent_audible_now():
            return "already_quiet", None, "", ""
        if self._drain is None:
            return "drained", None, "", ""
        # Unconfirmed BEFORE the await, as on the text path: a cancellation
        # mid-drain reaches ``finally`` with this in place.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "drained", {"signal": "tail_elapsed", "confirmed": False}, "", ""
        egress: dict[str, object] = {"signal": "drain_cancelled", "confirmed": False}
        egress = await self._drain_egress(
            timeout=min(self._drain_timeout_seconds, remaining)
        )
        return "drained", egress, "", ""

    def _guard_terminal_details(self) -> dict[str, object]:
        guard = self._guard
        if guard is None:
            return {}
        try:
            summary = dict(guard.summary())
        except Exception:  # noqa: BLE001 - a diagnostic cannot decide the ending
            return {}
        refused_turns = int(summary.get("refused_turns") or 0)
        promoted = bool(summary.get("silence_promoted"))
        return {
            "guard_refusals": refused_turns,
            "silence_promoted": promoted,
            "after_refusal": promoted and self._finish_override is not None,
        }

    def _final_message(self, raw_arguments: dict[str, object]) -> str:
        requested = raw_arguments.get("message")
        if isinstance(requested, str):
            message = " ".join(requested.split())
            if message:
                return message
        return _END_CALL_FALLBACK_MESSAGES.get(
            self._language, _END_CALL_FALLBACK_MESSAGES["en"]
        )

    async def _drain_egress(self, timeout: float | None = None) -> dict[str, object]:
        """Run the bounded room-side drain; never raise, never certify on failure."""

        assert self._drain is not None
        try:
            result = dict(
                await asyncio.wait_for(
                    self._drain(),
                    timeout=self._drain_timeout_seconds if timeout is None else timeout,
                )
            )
        except TimeoutError:
            return {"signal": "drain_timeout", "confirmed": False}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "signal": "drain_failed",
                "confirmed": False,
                "error_type": type(exc).__name__,
            }
        result["confirmed"] = bool(result.get("confirmed"))
        result.setdefault("signal", "drain_failed")
        return result

    async def _execute(self, raw_arguments: dict[str, object]) -> dict[str, object]:
        playout = "failed"
        # Which step failed, and with what. The three failure branches below
        # used to share one bare ``except`` shape that dropped the exception,
        # so every S2S goodbye reported ``playout: failed`` with no
        # way to tell the SDK refusing text-to-speech on a session that has no
        # TTS from a provider fault. The class name only: a free
        # message can carry the farewell text the model wrote.
        error_type = ""
        error_stage = ""
        # ``None`` until a drain ran: the recorder then keeps the legacy rule.
        egress: dict[str, object] | None = None
        message = self._final_message(raw_arguments)
        # Captured BEFORE say() so it precedes the goodbye's first audio frame:
        # the recorder links the goodbye to the first interval starting at or
        # after this, forward-only, so a teardown blip before it cannot capture
        # the farewell text. speech_id ties the record to that bound candidate.
        scheduled_at = time.monotonic()
        speech_id = ""
        speech = None
        reclaimed = False
        try:
            if not self._speaks_text:
                playout, egress, error_type, error_stage = await self._settle_reclaimable_live_goodbye()
                reclaimed = playout == "reclaimed" and not self._authoritative_close()
                if playout == "reclaimed" and not reclaimed:
                    playout = "interrupted"
            else:
                if self._native_close_reclaimed.is_set() and not self._authoritative_close():
                    raise _CloseReclaimed
                # The budget fixed when this close was accepted on a measured
                # source; ``None`` keeps today's VAD-reclaimable farewell.
                budget = self._close_budget
                # Worker-owned text is recorded by the terminal seam (or the
                # TTS candidate when reclaimed), not also as a normal model
                # reply. Otherwise open admission double-counts its telemetry.
                # On a measured source the SDK may not cut the farewell at all:
                # a duration-only VAD interruption cannot tell noise from a
                # caller, so caller WORDS reclaim it explicitly instead
                # (``speech.interrupt(force=True)`` below).
                try:
                    speech = self._session.say(
                        message,
                        allow_interruptions=False if budget is not None else not self._authoritative_close(),
                        add_to_chat_ctx=False,
                    )
                except Exception as exc:
                    if budget is None:
                        raise
                    # A measured close settles a failed say like any failed
                    # playout: the caller's floor still decides the commit.
                    playout, error_type, error_stage = "failed", type(exc).__name__, "say"
                    speech = None
                if speech is not None:
                    if budget is not None:
                        self._farewell_speech = speech
                    speech_id = str(getattr(speech, "id", "") or "")
                    # Arm the playout-start watcher with THIS handle's id before awaiting
                    # playout, so the goodbye's first "speaking" edge is stamped.
                    if self._on_say is not None and speech_id:
                        try:
                            self._on_say(speech_id)
                        except Exception:
                            pass
                    try:
                        await self._await_cascade_close(
                            speech.wait_for_playout(), timeout=self._playout_timeout_seconds
                        )
                    except _CloseReclaimed:
                        raise
                    except TimeoutError:
                        playout = "timeout"
                    except Exception as exc:
                        playout = "failed"
                        error_type, error_stage = type(exc).__name__, "wait_for_playout"
                    else:
                        try:
                            speech_error = speech.exception()
                            if speech_error is not None:
                                playout = "failed"
                                error_type = type(speech_error).__name__
                                error_stage = "speech_exception"
                            elif bool(getattr(speech, "interrupted", False)):
                                if not self._authoritative_close() and (
                                    budget is None or self._native_close_reclaimed.is_set()
                                ):
                                    # A new transcript or language-switch boundary
                                    # may cut the SDK handle without a fresh VAD edge.
                                    # Measured, only caller words make a cut a reclaim.
                                    raise _CloseReclaimed
                                # wait_for_playout can return WITHOUT an exception when a
                                # caller hangs up mid-goodbye: the handle completes
                                # interrupted. That is a cut, not a completed playout --
                                # certifying it heard would be a lie (mirrors the
                                # opening-speech path's interrupted read).
                                playout = "interrupted"
                            else:
                                playout = "completed"
                        except _CloseReclaimed:
                            raise
                        except Exception as exc:
                            playout = "failed"
                            error_type, error_stage = type(exc).__name__, "speech_exception"
                playout_ended_at = time.monotonic()
                if playout == "completed" and self._drain is not None:
                    # The SDK edge is where the last frame was handed to the local
                    # transport.  Hold shutdown until the room has heard it go
                    # quiet, or the bound says the answer is unknowable.  The
                    # verdict is unconfirmed BEFORE the await: a cancellation that
                    # lands mid-drain reaches ``finally`` with this in place, so a
                    # goodbye whose tail was never observed is not certified
                    # through the legacy rule.
                    egress = {"signal": "drain_cancelled", "confirmed": False}
                    egress = await self._await_cascade_close(self._drain_egress())
                if budget is not None:
                    # Words still in flight after the farewell (a "wait" begun
                    # over its tail, transcribed after the last frame) reclaim
                    # here; a wordless run only holds the commit, bounded.
                    await self._settle_close_floor(playout_ended_at, budget, scheduled_at)
                # Includes a speech edge during child-waiter cleanup, or a
                # failed/instant playout. No await separates this from commit.
                if self._native_close_reclaimed.is_set() and not self._authoritative_close():
                    raise _CloseReclaimed
        except _CloseReclaimed:
            reclaimed = not self._authoritative_close()
            playout = "reclaimed" if reclaimed else "interrupted"
            if reclaimed and speech is not None:
                # A caller turn completed against a farewell that still refuses
                # interruptions would be dropped by the runtime.
                restore_speech_interruptions(speech)
                try:
                    # This exact worker-owned farewell has been revoked, not
                    # an ordinary reply asking the PSTN word gate to yield.
                    # That gate may have protected it at its speaking edge.
                    speech.interrupt(force=True)
                except Exception:
                    pass
        except Exception as exc:
            playout = "failed"
            if not error_type:
                # Nothing inner classified it, so on the text path it came from
                # ``say`` itself (or from arming the watcher, which is guarded)
                # -- before this path existed an S2S session with no TTS raised
                # RuntimeError right here.
                error_type = type(exc).__name__
                error_stage = "say" if self._speaks_text else "settle"
        finally:
            if self._tool_speech is not None:
                self._tool_speech.dispose()
            self._tool_speech = None
            self._tool_speech_error = ""
            self._tool_speech_ready.clear()
            self._farewell_speech = None
            if reclaimed:
                self._close_result_context["phase"] = "reclaimed"
                if self._refusal_recovery is not None:
                    self._refusal_recovery.note_reclaimed()
                self._receipt("reclaimed", reason="caller_speech")
                # Nothing irreversible ran for this proposal: caller admission
                # remains open, and a later tool can request a fresh close.
                self._task = None
                self._close_budget = None
                self._intent.release_close()
                self._notify_reclaimed()
                if self._emit is not None:
                    try:
                        self._emit("end_call_reclaimed", basis="caller_took_floor")
                    except Exception:
                        pass
            else:
                self._close_committed = True
                self._close_result_context["phase"] = "committed"
                self._receipt_commit()
                if self._finish_override is None:
                    try:
                        self._on_ending()
                    except Exception:
                        pass
                # Commit the resolving turn's evidence on EVERY outcome, and BEFORE
                # ``finish()`` releases the owning worker to finalize the bundle.
                # The goodbye is the response to the caller's final utterance but
                # never arrives as an assistant conversation item, so without this
                # commit the turn loses its transcript text and its telemetry span
                # -- ``evidence.status`` reads ``partial`` for a healthy call. A
                # farewell that timed out or failed is recorded as CUT (never
                # certified as fully heard), never silently dropped.
                cut = playout != "completed"
                egress_confirmed = None if egress is None else bool(egress.get("confirmed"))
                egress_signal = "" if egress is None else str(egress.get("signal") or "")
                # A voice-to-voice goodbye is the model's own speech and is recorded
                # as its own assistant item; the tool's ``message`` argument was
                # never spoken there, and certifying it as the final turn would
                # put words in the call's evidence that nobody heard.
                if self._record_final_message is not None and self._speaks_text:
                    try:
                        # The egress fields travel only when a drain actually ran,
                        # so a recorder seam without one keeps its exact call shape.
                        self._record_final_message(
                            message,
                            interrupted=cut,
                            speech_id=speech_id,
                            scheduled_at=scheduled_at,
                            **(
                                {
                                    "egress_confirmed": egress_confirmed,
                                    "egress_signal": egress_signal,
                                }
                                if egress is not None
                                else {}
                            ),
                        )
                    except Exception:
                        pass
                if self._commit_terminal_telemetry is not None:
                    try:
                        self._commit_terminal_telemetry()
                    except Exception:
                        pass
                # ``shutdown`` starts LiveKit's orderly close; the owning worker
                # awaits ``aclose`` in its existing cleanup/finalization path.
                # Do this even if speech construction or playout failed so SIP BYE
                # and the caller's PSTN leg cannot remain orphaned.
                try:
                    self._session.shutdown(drain=False)
                except Exception:
                    pass
                try:
                    (self._finish_override or self._finish)()
                except Exception:
                    pass
                if self._emit is not None:
                    try:
                        self._emit(
                            "end_call",
                            language=self._language,
                            playout=playout,
                            speaks_text=self._speaks_text,
                            initiated_by=(
                                "platform" if self._finish_override is not None else "agent"
                            ),
                            # The end_call guard's account of this call, on the
                            # terminal event the verifier and the shadow report
                            # already read: how many caller turns it refused, and
                            # whether a refusal put the silence rule on the clock.
                            # A platform ending after a refusal is the promoted
                            # silence rule's own, and says so as provenance.
                            **self._guard_terminal_details(),
                            **(
                                {"error_type": error_type, "error_stage": error_stage}
                                if error_type
                                else {}
                            ),
                            **(
                                {"quiet_ms_at_tool": self._quiet_ms_at_tool}
                                if not self._speaks_text
                                else {}
                            ),
                            **(
                                {
                                    "egress_signal": egress_signal,
                                    "egress_confirmed": egress_confirmed,
                                    "egress_waited_ms": egress.get("waited_ms"),
                                    "egress_quiet_after_edge_ms": egress.get(
                                        "quiet_after_edge_ms"
                                    ),
                                }
                                if egress is not None
                                else {}
                            ),
                        )
                    except Exception:
                        pass
        if reclaimed:
            return {"ok": False, "status": "reclaimed", "playout": playout}
        return {
            "ok": playout in END_CALL_HEARD_PLAYOUTS,
            "status": "teardown_requested",
            "playout": playout,
            **({"egress": egress_signal} if egress is not None else {}),
        }


async def await_worker_terminal_end_call(
    end_call: WorkerLocalEndCall,
    *,
    emit: Callable[..., None] | None = None,
) -> dict[str, object] | None:
    """Join an active worker-owned goodbye before immutable teardown.

    A caller can leave a LiveKit room while ``WorkerLocalEndCall`` is still
    awaiting its own speech handle.  The participant-disconnect callback then
    wakes the outer worker finalizer independently.  Closing the session or
    serializing the evidence bundle before the worker-owned task reaches its
    ``finally`` loses the terminal message/outcome race: the artifact can seal
    before the goodbye is recorded.

    ``wait_for_completion`` is already shielded from a cancelled model-tool
    waiter and the end-call task has its own fixed 15-second playout bound.
    Joining it here therefore preserves the existing media bound, while making
    the goodbye's record/telemetry commit happen before ``session.aclose()``
    and ``recorder.finalize()``.  A forced worker cancellation is deliberately
    still propagated so the outer teardown's existing cancellation policy owns
    it; ordinary task failures are classified but cannot skip settlement.
    """

    try:
        outcome = await end_call.wait_for_completion()
    except asyncio.CancelledError:
        # ``shield`` propagates the same exception for two very different
        # events: a cancellation of THIS waiter, or completion of the INNER
        # worker task by cancellation. When the inner task began, its
        # ``finally`` has already recorded a cut goodbye and terminal
        # telemetry by the time ``Task.cancelled()`` becomes true. If it was
        # cancelled before it began, no goodbye was ever scheduled to speak;
        # either way, settlement must continue without inventing evidence. An
        # outer cancellation has a pending cancellation request on this task
        # and is left for the teardown's deferred-cancellation policy below.
        waiter = asyncio.current_task()
        if end_call.completion_was_cancelled() and (
            waiter is None or waiter.cancelling() == 0
        ):
            if emit is not None:
                try:
                    emit("end_call_teardown_wait_inner_cancelled")
                except Exception:
                    pass
            return None
        raise
    except Exception as exc:
        if emit is not None:
            try:
                emit("end_call_teardown_wait_failed", error_type=type(exc).__name__)
            except Exception:
                pass
        return None
    if outcome is not None and emit is not None:
        try:
            emit(
                "end_call_teardown_wait_completed",
                playout=str(outcome.get("playout") or ""),
            )
        except Exception:
            pass
    return outcome


async def settle_worker_terminal_end_call(
    end_call: WorkerLocalEndCall,
    *,
    emit: Callable[..., None] | None = None,
) -> tuple[dict[str, object] | None, asyncio.CancelledError | None]:
    """Finish the bounded goodbye despite repeated outer cancellation.

    The underlying end-call task owns a fixed playout bound and commits its
    terminal evidence in ``finally``. If the outer worker is cancelled while
    it joins that task, retain one cancellation to re-raise after the existing
    teardown settles the SDK session, evidence, billing and room cleanup.
    Clear the pending request before retrying so a second cancellation is also
    observed and deferred rather than cutting through that settlement.

    An inner task cancellation is handled by
    :func:`await_worker_terminal_end_call` as a settled cut-goodbye outcome;
    it is not an external worker cancellation and must not be retried as one.
    """

    deferred_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return (
                await await_worker_terminal_end_call(end_call, emit=emit),
                deferred_cancellation,
            )
        except asyncio.CancelledError as exc:
            waiter = asyncio.current_task()
            # A CancelledError from an inner task has no pending request on
            # this waiter and is handled above. Do not accidentally suppress
            # a different cancellation source if the runtime cannot identify
            # the waiting task.
            if waiter is None or waiter.cancelling() <= 0:
                raise
            if deferred_cancellation is None:
                deferred_cancellation = exc
            # ``Task.cancelling`` counts requests rather than consuming them
            # when one is caught. Consume every currently-pending request so
            # this bounded join can make progress; a later request is caught
            # on its own retry, and the first one is re-raised by the caller
            # only after full terminal settlement.
            while waiter.cancelling() > 0:
                waiter.uncancel()
            if emit is not None:
                try:
                    emit("end_call_teardown_wait_cancelled")
                except Exception:
                    pass


#: How long the teardown waits for the guard's in-flight post-refusal steps
#: and the speeches they regenerated before the summary is frozen.
END_CALL_GUARD_SETTLE_SECONDS = 2.0


async def settle_end_call_guard_summary(
    guard: Any,
    tasks: Iterable[asyncio.Task[Any]],
    *,
    emit: Callable[..., None] | None = None,
    timeout: float = END_CALL_GUARD_SETTLE_SECONDS,
    poll: float = 0.05,
) -> tuple[dict[str, Any], asyncio.CancelledError | None]:
    """Settle the guard's in-flight work (bounded) and return its summary.

    A regeneration failing after the caller hung up must land in the counters
    the verifier reads, so the post-refusal tasks and then the regenerated
    speeches they issued are waited for, together bounded by ``timeout``; a
    speech still pending at the bound is reported as unsettled, never
    certified. Like :func:`settle_worker_terminal_end_call`, an external
    cancellation of the worker during these waits is HELD and returned for the
    caller to re-raise after the summary, session close, evidence, usage and
    lifecycle settle: letting it through here would skip all of them.
    """

    deferred_cancellation: asyncio.CancelledError | None = None
    deadline = time.monotonic() + timeout

    async def _wait() -> None:
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
        while guard.utterances.regenerations_pending and time.monotonic() < deadline:
            await asyncio.sleep(poll)

    while True:
        try:
            await _wait()
            break
        except asyncio.CancelledError as exc:
            waiter = asyncio.current_task()
            if waiter is None or waiter.cancelling() <= 0:
                raise
            if deferred_cancellation is None:
                deferred_cancellation = exc
            while waiter.cancelling() > 0:
                waiter.uncancel()
            if emit is not None:
                try:
                    emit("end_call_guard_settle_cancelled")
                except Exception:  # noqa: BLE001 - diagnostic only
                    pass
            # The bound is on the whole settle: a cancellation arriving at
            # every yield must not keep the retry going past it.
            if time.monotonic() >= deadline:
                break
        except Exception:  # noqa: BLE001 - teardown continues regardless
            break
    return guard.summary(), deferred_cancellation


#: How long the model waits for the outcome report before it hears the
#: result anyway. The report only annotates the attempt row.
TRANSFER_OUTCOME_REPORT_SECONDS = 3.0


async def _report_transfer_outcome(
    session: ToolSession, outcome: str, emit: Callable[..., None] | None
) -> None:
    """Record the transfer's outcome on its attempt row. Raises only on cancellation.

    A report that fails, stalls or is refused (an older control plane without
    the route, the network, no attempt row) leaves the row at ``connecting``,
    which is what it said before this report existed. It is logged, and it
    must not delay or turn a finished transfer into a tool error the model
    then apologises for.
    """
    reason = ""
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(session.report_transfer_outcome, outcome),
            timeout=TRANSFER_OUTCOME_REPORT_SECONDS,
        )
        if isinstance(reply, dict) and reply.get("ok") is False:
            reason = str(reply.get("reason_code") or "refused")
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        reason = "timeout"
    except Exception as exc:
        reason = type(exc).__name__
    if reason and emit is not None:
        try:
            emit("transfer_outcome_not_recorded", outcome=outcome, error_type=reason)
        except Exception:
            pass


#: The progress note ``transfer_to_human`` gives the model once the dial is
#: placed. Until a tool answers, a later tool call waits behind it (GPT-Live's
#: response chain, and the cascade too, measured on the SIP rehearsal), so
#: without it ``cancel_transfer`` could not run while the phone rings. It names
#: no tool: a note that named cancel_transfer led a cascade model to cancel
#: and re-place a transfer the caller had merely asked for twice. When to
#: cancel is the injected instruction's and the tool's own description's.
TRANSFER_DIALING_UPDATE = (
    "The transfer is under way and the colleague's phone is ringing; nobody is connected yet. It "
    "continues on its own and its result will follow, so it needs nothing more from you. If the caller "
    "asks for it again, tell them it is already ringing."
)
#: What the model is told when the transfer has ended (its deferred result).
#: ``caller_cancelled`` has none: ``cancel_transfer`` has already answered it.
TRANSFER_RESULT_MESSAGES = {
    "answered": "The person answered and the caller is now connected to them. Say nothing more.",
    "no_answer": "Nobody answered. The caller is still with you: keep helping them and do not try the transfer again.",
    "busy": "The line was busy. The caller is still with you: keep helping them and do not try the transfer again.",
    "rejected": "The call was declined. The caller is still with you: keep helping them and do not try the transfer again.",
    "failed": "The transfer could not be placed. The caller is still with you: keep helping them and do not try the transfer again.",
    "caller_hangup": "The caller hung up before anyone was connected.",
}
#: What ``cancel_transfer`` tells the model, by its answer.
CANCEL_TRANSFER_MESSAGES = {
    "cancelled": "The transfer is stopped and nobody was connected. Tell the caller, then keep helping them.",
    "not_in_progress": "No transfer is in progress, so there was nothing to stop.",
    "already_connected": "The caller is already connected to the person, so the transfer cannot be stopped.",
    "no_caller_request": (
        "The caller has not asked to stop the transfer since it began, so it was not stopped and is still "
        "ringing. Keep the caller informed that you are still trying to reach someone."
    ),
}


def build_livekit_model_tools(
    session: ToolSession,
    *,
    caller_inputs: Any | None = None,
    end_call: Callable[[dict[str, object]], Any] | None = None,
    transfer: Callable[..., Any] | None = None,
    cancel_transfer: Callable[[], Awaitable[str]] | None = None,
    tool_usage_counter: Any | None = None,
    emit: Callable[..., None] | None = None,
    tool_builder: Callable[
        [Callable[..., Any], dict[str, Any]], Any
    ] = _default_tool_builder,
    terminal_intent: TerminalIntent | None = None,
    navigation_gate: Any | None = None,
    worker_local_names: frozenset[str] = frozenset(),
) -> list[Any]:
    """Translate the filtered MCP allowlist into LiveKit model-callable tools.

    ``worker_local_names`` are the tools the caller adds itself after this
    (``switch_language``): a configured tool with the same name is not built,
    since the model's tool list cannot hold two functions of one name and the
    worker's own capability takes precedence over a customer declaration.

    The model receives only schemas already filtered by
    :meth:`Gate2CallSession.tool_definitions`.  Each invocation crosses the
    same session interface, which rechecks the allowlist and creates the MCP
    audit event before returning output to the model. ``end_call`` and
    ``cancel_transfer`` are the exceptions: their effect is owned by the
    running media worker, never by the control-plane tool endpoint.

    ``transfer`` runs an authorized dial (``TransferController.run``) and
    reports its outcome itself; ``cancel_transfer`` stops it. A call without
    the latter is not offered ``cancel_transfer``.

    A tool that cannot be declared or built is left out on its own and named
    in a ``configured_tool_omitted`` event (name and reason code only); the
    call starts with every other tool.
    """

    def omit(tool: str, reason: str) -> None:
        if emit is not None:
            emit("configured_tool_omitted", tool=tool, reason=reason)

    tools: list[Any] = []
    definitions = session.tool_definitions()
    for omission in getattr(session, "omitted_tools", None) or ():
        omit(str(omission.get("tool") or ""), str(omission.get("reason") or ""))
    tenant_arguments = {
        name: str(session.config.get(name) or "").strip()
        for name in ("organization_id", "project_id")
    }
    if definitions and not all(tenant_arguments.values()):
        # A call fault, not a tool's: every tool would run unbound.
        raise ProductionConfigurationError(
            "realtime-config omitted the authenticated organization/project binding"
        )
    for definition in definitions:
        name = str(definition.get("name") or "").strip()
        if name in worker_local_names:
            if emit is not None:
                emit("configured_tool_shadowed_by_worker_tool", tool=name)
            continue
        # Only a plain function name is ever logged, never free text.
        label = name if name.isascii() and name.replace("-", "_").isidentifier() and len(name) <= 64 else ""
        parameters = definition.get("inputSchema")
        if not name or not isinstance(parameters, dict):
            omit(label, "missing_schema")
            continue
        try:
            model_parameters = copy.deepcopy(parameters)
            properties = model_parameters.get("properties")
            required = model_parameters.get("required")
            for tenant_field in tenant_arguments:
                if isinstance(properties, dict):
                    properties.pop(tenant_field, None)
                if isinstance(required, list):
                    model_parameters["required"] = [
                        field
                        for field in model_parameters.get("required", [])
                        if field != tenant_field
                    ]
            raw_schema = {
                "name": name,
                "description": str(
                    definition.get("description") or f"Skysay tool {name}"
                ),
                "parameters": model_parameters,
            }
        except Exception:  # noqa: BLE001 - one unpreparable schema costs that tool, not the call
            omit(label, "invalid_schema")
            continue

        if name == "end_call":

            async def invoke(
                raw_arguments: dict[str, object],
                ctx: RunContext = None,  # type: ignore[assignment]
                *,
                _end_call: Callable[..., Any] | None = end_call,
                _name: str = name,
                _tool_usage_counter: Any | None = tool_usage_counter,
            ) -> str | None:
                if _end_call is None:
                    raise ProductionConfigurationError(
                        "end_call requires the worker-local termination handler"
                    )
                function_call = getattr(ctx, "function_call", None)
                attempt_id = str(getattr(function_call, "call_id", "") or "")
                if caller_inputs is not None:
                    # Our own hang-up first lets a just-given yes run (the
                    # quick-bye proof); a caller's hang-up never does. Before
                    # the navigation hold, whose refusal latch is short.
                    heard = caller_inputs.activity
                    await caller_inputs.drain()
                    if caller_inputs.activity != heard:
                        # The caller started speaking while it ran, before any
                        # close owner could reclaim it: refused on the caller's
                        # floor through the owner's recovery, like its own.
                        if isinstance(_end_call, WorkerLocalEndCall):
                            raise ToolError(str(_end_call.caller_floor_refusal(attempt_id).get("text") or ""))
                        raise ToolError(REFUSAL_TEXT[BASIS_CALLER_STILL_SPEAKING])
                if navigation_gate is not None:
                    # A menu is never answered with a declaration: the label for
                    # the utterance the model is reacting to is applied BEFORE
                    # the hang-up is judged. The refusal itself is the end_call
                    # owner's (it ends a refused hang-up without a farewell loop).
                    await navigation_gate.hold(_name)
                if isinstance(_end_call, WorkerLocalEndCall):
                    # The SDK injects ``RunContext`` into a raw tool only when
                    # the parameter is annotated with it (plain, not Optional:
                    # ``is_context_type`` does not unwrap a union). Its call
                    # id and speech id are the guard's attempt identity.
                    speech_handle = getattr(ctx, "speech_handle", None)
                    result = await _end_call(
                        dict(raw_arguments),
                        attempt_id=attempt_id,
                        speech_id=str(getattr(speech_handle, "id", "") or ""),
                        origin_speech=speech_handle,
                        user_item_id=(getattr(function_call, "extra", None) or {}).get(
                            "skysay_reply_user_item_id", ""),
                        # The SDK shields a running tool on a barge-in and waits
                        # for it; the guard reads the handle to notice the cut.
                        speech_interrupted=(
                            (lambda: bool(getattr(speech_handle, "interrupted", False)))
                            if speech_handle is not None
                            else None
                        ),
                    )
                else:
                    result = await _end_call(dict(raw_arguments))
                if isinstance(result, dict) and result.get("status") == "aborted":
                    return None
                if isinstance(result, dict) and result.get("status") == "refused":
                    # The guard declined the hang-up. Raised as the SDK's
                    # ToolError, so the executor records the refusal as an
                    # ERROR output carrying this text: the transcript's
                    # tool_call row reads ok=false (a hang-up that did not
                    # happen is not a successful end_call), and the model's
                    # view is unchanged in practice, since the worker's
                    # function_tools_executed handler cancels the follow-up
                    # and drops the pair before any regeneration. The text
                    # is the refusal string, never caller words.
                    raise ToolError(str(result.get("text") or ""))
                if _tool_usage_counter is not None and _successful_tool_result(result):
                    _tool_usage_counter.record_success(_name)
                # The pinned LiveKit tool executor treats any non-None output
                # as reply_required. The worker has already scheduled terminal
                # playout, so returning None suppresses a model follow-up turn.
                return None
        elif name == "cancel_transfer":
            if cancel_transfer is None:
                omit(label, "worker_local_unavailable")
                continue

            async def invoke(
                raw_arguments: dict[str, object],
                *,
                _name: str = name,
                _cancel: Callable[[], Awaitable[str]] = cancel_transfer,
            ) -> str:
                # Worker-local, like end_call: no control-plane call, no
                # outbound authorization, and nothing to meter.
                status = await _cancel()
                return json.dumps(
                    {
                        "ok": True,
                        "tool": _name,
                        "result": {"status": status, "message": CANCEL_TRANSFER_MESSAGES.get(status, "")},
                    },
                    sort_keys=True,
                )
        else:

            async def invoke(
                raw_arguments: dict[str, object],
                ctx: RunContext = None,  # type: ignore[assignment]
                *,
                _name: str = name,
                _tenant_arguments: dict[str, str] = tenant_arguments,
                _tool_usage_counter: Any | None = tool_usage_counter,
            ) -> str | None:
                arguments = dict(raw_arguments)
                arguments.update(_tenant_arguments)
                if navigation_gate is not None:
                    # Held for the labels in flight, then judged on the state
                    # they produced: the business tools are refused in a phone
                    # menu or a queue, with the contract's answer.
                    refusal = await navigation_gate.refusal(_name)
                    if refusal is not None:
                        return json.dumps(refusal, sort_keys=True)
                # One terminal owner: a transfer takes the call at ENTRY, before
                # any control-plane work, and is refused while it is closing.
                owns_transfer = (
                    _name == "transfer_to_human" and transfer is not None and terminal_intent is not None
                )
                if owns_transfer and not terminal_intent.claim_transfer():
                    if terminal_intent.state == INTENT_TRANSFERRING:
                        # A second request while one rings (a caller who said
                        # it twice): the first stands, and nothing is dialled.
                        return json.dumps(
                            {
                                "ok": False,
                                "status": "refused",
                                "reason_code": "transfer_in_progress",
                                "message": "A transfer is already in progress. Do not start another; wait for its result.",
                            },
                            sort_keys=True,
                        )
                    return json.dumps(
                        {
                            "ok": False,
                            "status": "refused",
                            "reason_code": "call_ending",
                            "message": "The call is ending, so it cannot be transferred.",
                        },
                        sort_keys=True,
                    )
                bridged = withdrawn = False
                try:
                    if caller_inputs is not None:
                        reference = await caller_inputs.reference()
                        result = await asyncio.to_thread(session.invoke_tool, _name, arguments, caller_input=reference)
                        pending = result.get("result") if isinstance(result, dict) else None
                        pending = pending.get("action") if isinstance(pending, dict) else None
                        if isinstance(pending, dict) and pending.get("status") == "awaiting_confirmation":
                            # Its answer counts only once the caller has heard it.
                            caller_inputs.expect_question(
                                str(pending["id"]),
                                call_id=str(getattr(getattr(ctx, "function_call", None), "call_id", "") or ""),
                            )
                    else:
                        result = await asyncio.to_thread(session.invoke_tool, _name, arguments)
                    dial = None
                    if isinstance(result, dict) and isinstance(result.get("result"), dict):
                        # The dial plan is for this worker. The model sees the outcome only.
                        dial = result["result"].pop("_worker_dial", None)
                    if _name == "transfer_to_human" and transfer is not None and isinstance(dial, dict) and result.get("ok"):
                        claim = await asyncio.to_thread(session.claim_transfer_dial)
                        if not isinstance(claim, dict) or claim.get("ok") is not True:
                            return json.dumps({"ok": False, "reason_code": "transfer_dial_not_claimable"})
                        update = getattr(ctx, "update", None)

                        async def dialing() -> None:
                            # The SDK injects ``RunContext`` into a raw tool only
                            # when the parameter is annotated with it (plain, not
                            # Optional). Its first update answers this call for
                            # the model, so the model can still act during the
                            # ring; the outcome follows as a deferred reply.
                            if callable(update):
                                await update(TRANSFER_DIALING_UPDATE)

                        try:
                            # The transfer reports its own outcome, exactly once,
                            # and raises only for a failure while the call is live.
                            outcome = await transfer(dial, on_dialing=dialing)
                        except asyncio.CancelledError:
                            # The session is closing. The call stays the
                            # transfer's: no farewell may end a bridge that may
                            # exist.
                            bridged = True
                            raise
                        # A bridged call stays the transfer's until it ends.
                        bridged = outcome == "answered"
                        withdrawn = outcome == "caller_cancelled"
                        result["result"] = {"status": outcome, "message": TRANSFER_RESULT_MESSAGES.get(outcome, "")}
                finally:
                    if owns_transfer and not bridged:
                        terminal_intent.release_transfer()
                if _tool_usage_counter is not None and _successful_tool_result(result):
                    action = (result.get("result") or {}).get("action")
                    if isinstance(action, dict):
                        _tool_usage_counter.record_success(_name, action_id=action["id"])
                    else:
                        _tool_usage_counter.record_success(_name)
                if withdrawn:
                    # cancel_transfer has already told the model; a None
                    # output asks for no second reply.
                    return None
                return json.dumps(result, sort_keys=True, default=str)

        try:
            built = tool_builder(invoke, raw_schema)
        except Exception:  # noqa: BLE001 - one unbuildable schema costs that tool, not the call
            omit(label, "builder_rejected")
            continue
        if tool_usage_counter is not None:
            # Register the exact function schemas given to this call's model,
            # not a broad workspace allowlist. The counter retains only names
            # and completed counts, never model arguments or tool output.
            tool_usage_counter.register(name)
        tools.append(built)
    return tools


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def settle_pre_media_failure(exc, *, ctx, participant, call, adapter, pre_media_call_id,
        pre_media_stage, browser_test, browser_participant_mismatch, inbound_route,
        live_pickup, report_browser_test_failure, state_path, sidecar, capture_tasks,
        inbound_audio_output, gpt_live_http, emit, inbound_release_canary, remove_participant,
        native_model=None):
    """One pre-media settlement classifier shared by freeform and governed roots.

    Return whether the original job handler treats the refusal as handled;
    otherwise its bare raise preserves the original exception object.
    """
    try:
        from .control_plane import ROUTES_THAT_REFUSE_BEFORE_ADMISSION, designed_refusal
    except ImportError:
        from control_plane import ROUTES_THAT_REFUSE_BEFORE_ADMISSION, designed_refusal
    pickup_cleanup_cancelled = None
    try:
        from .gpt_live_runtime import is_eleven_native_config
    except ImportError:
        from gpt_live_runtime import is_eleven_native_config
    if call is not None and native_model is not None and is_eleven_native_config(call.config):
        # The model deletes any per-call provider agent it already created.
        try:
            await await_bounded_teardown_task(
                asyncio.create_task(native_model.aclose()), timeout_seconds=30.0)
        except asyncio.CancelledError as cleanup_exc:
            pickup_cleanup_cancelled = cleanup_exc
        except Exception as cleanup_exc:
            emit("native_provider_cleanup_pending", call_id=call.call_id,
                 error_type=type(cleanup_exc).__name__)
    if live_pickup is not None:
        try:
            await await_bounded_teardown_task(
                asyncio.create_task(live_pickup.aclose()), timeout_seconds=1.0)
        except asyncio.CancelledError as cleanup_exc:
            # Preserve cancellation after the authoritative lifecycle
            # settlement below; capture cleanup cannot strand a call.
            pickup_cleanup_cancelled = cleanup_exc
    _pre_media_refusal_reason = (
        str(getattr(exc, "reason", "") or "")
        if isinstance(exc, ProductionConfigurationError)
        else ""
    )
    try:
        if browser_test is not None:
            await report_browser_test_failure("session")
    finally:
        result = await terminate_pre_media_call(
            call=call,
            adapter=adapter,
            call_id=pre_media_call_id,
            # A refusal that names its OWN rule publishes that name; every
            # other failure publishes the stage's reason, as before.
            # ``component_configuration`` alone covers around two dozen
            # distinct refusals in the profile builder and names none of
            # them, so a customer could not tell a missing voice from a
            # follow-the-caller language mismatch.
            #
            # A participant mismatch still wins: it is a statement about
            # WHO joined the room, which no configuration rule can answer,
            # and it must not be overwritten by one. Passed through
            # ``safe_pre_media_reason`` so a label is bounded and
            # lowercase-ASCII on the way out even though every label in the
            # table already is -- the guarantee belongs at the boundary,
            # not in the goodwill of each raise site.
            reason=(
                safe_pre_media_reason(_pre_media_refusal_reason)
                if _pre_media_refusal_reason and not browser_participant_mismatch
                else pre_media_lifecycle_reason(
                    pre_media_stage,
                    participant_mismatch=browser_participant_mismatch,
                )
            ),
            # Readable by the customer on ``call.failed``: the versioned
            # evidence marker and the bounded stage name, exactly what the
            # dialer publishes for its own pre-media refusals. Never the
            # exception -- its class goes to the worker log below, its
            # text nowhere.
            details={
                "failure_evidence_version": 1,
                "failure_stage": pre_media_stage,
            },
            remove_participant=remove_participant,
            delete_room=ctx.delete_room,
        )
        if state_path is not None and result["lifecycle_recorded"]:
            # The file already says ``media_started``, and the crash
            # reconciler would later close that record with a second
            # terminal report on a call the control plane has just
            # settled. Stamped FIRST, the moment the report is known to
            # have landed -- before the log row and before any cleanup
            # that can fail, be cancelled or lose the process -- and only
            # when it landed: otherwise that sweep is the one thing left
            # that can still settle this call, and the record stays open
            # for it.
            write_session_state(
                state_path,
                call_id=pre_media_call_id,
                last_heartbeat_at=utcnow(),
                status="complete",
            )
        emit(
            "pre_media_terminated",
            stage=pre_media_stage,
            call_id=pre_media_call_id,
            error=type(exc).__name__,
            participant_terminated=result["participant_terminated"],
            room_terminated=result["room_terminated"],
            lifecycle_recorded=result["lifecycle_recorded"],
        )
    if sidecar is not None:
        # Started after admission, so this guard is what stops it: the room
        # delete above kicks it, but its capture tasks are this job's.
        await shutdown_evidence_sidecar(
            sidecar,
            capture_tasks,
            on_late_completion=lambda phase, error_type: None,
        )
    if inbound_audio_output is not None:
        try:
            await await_bounded_teardown_task(
                asyncio.create_task(inbound_audio_output.aclose()), timeout_seconds=1.0)
        except asyncio.CancelledError as cleanup_exc:
            pickup_cleanup_cancelled = pickup_cleanup_cancelled or cleanup_exc
    if gpt_live_http is not None:
        # No live-session close to order this against: the duplex session
        # is created with the AgentSession, further down than this
        # pre-media handler can reach, so the transport closed here never
        # had a session on it. If that construction ever moves above this
        # handler, this site needs the same ordered close the media
        # teardown does.
        try:
            await gpt_live_http.close()
        except Exception:
            pass
        gpt_live_http = None
    if pickup_cleanup_cancelled is not None:
        raise pickup_cleanup_cancelled
    if (
        isinstance(exc, ProductionConfigurationError)
        and _pre_media_refusal_reason
        and result["lifecycle_recorded"]
    ):
        # A customer configuration this worker refused, published under its
        # own name, with the room, the participant and the reservation all
        # released. That is a finished job, not a worker fault -- and
        # re-raising made LiveKit log "unhandled exception while running the
        # job task", which is what put four of these on the voice-worker
        # error tracker and on the ntfy push that exists so a human wakes up
        # when a CALL goes silent.
        #
        # Narrow on purpose, all three conditions load-bearing: only this
        # exception class, only when it named its rule, and only when the
        # control plane actually acknowledged the termination. A refusal
        # whose lifecycle report did NOT land still raises, because then the
        # crash reconciler is the one thing left that can settle the call
        # and a green job would hide it. Everything else -- every
        # unlabelled failure, every cancellation, every bug in this
        # worker -- re-raises exactly as before. ``BrowserSessionAlreadyOwned``
        # above already returns from this handler for the same reason.
        return True
    if (
        designed_refusal(exc, release_canary_leg=inbound_route == inbound_release_canary)
        and result["participant_terminated"]
        and (
            result["lifecycle_recorded"]
            if str(pre_media_call_id or "").strip()
            else exc.request_path in ROUTES_THAT_REFUSE_BEFORE_ADMISSION
        )
    ):
        # The control plane refused this call ON PURPOSE -- an empty wallet, a
        # published agent that cannot run, a maintenance window -- and the
        # call's ending is on record: either this worker's ``sip.failed``
        # landed for the call it holds, or, holding none, the refusal came
        # from a route that refuses before it admits anything and writes the
        # refused call itself (``control_plane.ROUTES_THAT_REFUSE_BEFORE_ADMISSION``).
        # The caller's leg is gone too (a ringing inbound leg whose room is
        # deleted is answered 486 Busy Here). That is the product's designed
        # ending for a refused call, not a worker fault: re-raising made LiveKit
        # log "unhandled exception while running the job task" and paged a
        # human for a customer's empty wallet.
        #
        # Every condition is load-bearing. A refusal whose leg removal failed
        # still raises, so it is seen (room deletion is not the signal: the
        # pinned SDK's ``delete_room`` swallows its own errors). A refusal
        # whose ``sip.failed`` did not land still raises, because the crash
        # reconciler is then the one thing left that can settle the call. A
        # refusal from a request made AFTER the route admitted a call -- the
        # inbound path's realtime-config read -- still raises: nothing
        # recorded that call's ending and this job holds no id to. And which
        # refusals count is the table's decision, never this handler's.
        emit(
            "control_plane_refused",
            status=exc.status_code,
            code=exc.error_code,
            reason=exc.reason,
            stage=pre_media_stage,
            route=exc.request_path,
            call_id=str(pre_media_call_id or ""),
        )
        return True
    return False




async def report_call_answered(*, call, call_end_clock, open_entry_window, funded_runtime,
        provider_call_id, observed_at):
    """Report the shared answer clock and adopt only its authoritative funding."""
    if call_end_clock["answered_at"] is None:
        call_end_clock["answered_at"] = time.monotonic()
        # A queue that starts as music, or a menu read out after a
        # pause, is only recognised some seconds in: defer the
        # silence-reading enders for at most 60 s while the far end
        # speaks or plays (an opted-in agent only).
        open_entry_window("answer")
    funding_response = await asyncio.to_thread(
        call.report_lifecycle,
        event_id=f"{call.call_id}:sip-answered",
        event_type="sip.answered",
        source_component="livekit_sip",
        observed_at=observed_at,
        details={"provider_call_id": provider_call_id},
    )
    funded_runtime.adopt(funding_response)
    return funding_response



async def watch_outbound_answer(*, await_answer, participant_attributes, wait_for_active,
        timeout_seconds, start_opening_listen, answer_gate_evidence, joined_at_monotonic,
        emit, call_id, arm_answer_detection, report_sip_answered):
    """The existing outbound disposition/report ordering shared by both roots."""
    try:
        from .sip_answer_gate import has_answer_evidence
    except ImportError:
        from sip_answer_gate import has_answer_evidence
    try:
        disposition = await await_answer(participant_attributes=participant_attributes,
            wait_for_active=wait_for_active, timeout_seconds=timeout_seconds)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # No information is not the same as no answer: fall back to
        # exactly what this worker did before the gate existed.
        emit(
            "outbound_answer_gate_failed",
            call_id=call_id,
            error=type(exc).__name__,
        )
        disposition = NOT_REPORTED
    # The answer instant, taken before anything below awaits: the
    # callee-first listen window starts here, not after the
    # ``sip.answered`` report's round trip.
    start_opening_listen(disposition, time.monotonic())
    answer_gate_evidence["disposition"] = disposition
    answer_gate_evidence["wait_ms"] = int(
        round((time.monotonic() - joined_at_monotonic) * 1000)
    )
    emit(
        "outbound_answer_gate",
        call_id=call_id,
        disposition=disposition,
        wait_ms=answer_gate_evidence["wait_ms"],
    )
    # Armed HERE, where the disposition is known, not where the opening
    # turn later reads it: the two are separated by provider warm-up,
    # and a window measured from the second one would start seconds
    # after the audio it is supposed to describe.
    arm_answer_detection(disposition, wait_ms=answer_gate_evidence["wait_ms"])
    if has_answer_evidence(disposition):
        await report_sip_answered(utcnow())
    return disposition

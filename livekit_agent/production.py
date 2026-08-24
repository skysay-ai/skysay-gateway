# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Production-facing LiveKit semantic bindings.

This module is intentionally SDK-light.  The control plane owns the resolved
semantic configuration and MCP allowlist; the LiveKit adapter only validates
and translates those values into provider/plugin and model-tool objects.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

try:  # package import in tests; flat /app import in the pinned worker image
    from .demo import DemoProfile
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from demo import DemoProfile

try:  # package import in tests; flat /app import in the pinned worker image
    from .capabilities.registry import VoiceCapabilityError, load_registry
except ImportError:  # pragma: no cover - exercised by the container smoke test
    from capabilities.registry import VoiceCapabilityError, load_registry

try:
    from .release_canary_contract import worker_heartbeat_metadata
except ImportError:
    try:
        from release_canary_contract import worker_heartbeat_metadata
    except ModuleNotFoundError as exc:  # The public self-host export has no platform canary.
        if exc.name != "release_canary_contract":
            raise

        def worker_heartbeat_metadata(_environ: Mapping[str, str]) -> dict[str, dict[str, str]]:
            return {}


class ProductionConfigurationError(RuntimeError):
    """A resolved control-plane configuration cannot build a LiveKit agent."""


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
                name="openphonex-worker-heartbeat",
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
                except BaseException as exc:  # A health signal must never kill the voice worker.
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
    value = str(environ.get("OPENPHONEX_WORKER_HEARTBEAT_ENABLED", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def service_heartbeat_environ(environ: Mapping[str, str]) -> dict[str, str]:
    """Give the idle service heartbeat its own stable row from job children."""
    service_environ = dict(environ)
    worker_base = str(service_environ.get("OPENPHONEX_WORKER_ID") or "livekit-agent").strip()
    service_environ["OPENPHONEX_WORKER_ID"] = f"{worker_base}-service"
    return service_environ


def inbound_sip_identity(attributes: dict[str, str], participant_identity: str) -> tuple[str, str, str]:
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


def livekit_worker_heartbeat_payload(
    environ: dict[str, str] | os._Environ[str],
    *,
    process_id: int,
    worker_type: str = "livekit_agent",
) -> dict[str, Any]:
    worker_base = str(environ.get("OPENPHONEX_WORKER_ID") or "livekit-agent").strip()
    metadata: dict[str, Any] = {
        "media_stack": "livekit",
        "role": "semantic_agent_process",
    }
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


# STT registry_name -> the one model each is pinned to. Deepgram (nova-3) is
# the default English/Estonian path; Google (chirp_2) is the si-LK (Sinhala)
# path -- Deepgram does not support Sinhala at all. Chirp 2 in the
# asia-southeast1 recognizer location is the empirically verified si-LK
# path (Chirp 3 rejects si-LK with an HTTP 400 in both eu and us).
_LEGACY_STT_REGISTRY_MODELS: dict[str, str] = {
    "deepgram": "nova-3",
    "google": "chirp_2",
}
# TTS registry_names this worker knows how to build. ElevenLabs is the default
# English/Estonian path; ``gemini_tts`` (Gemini-TTS) is the si-LK (Sinhala)
# path -- ElevenLabs does not support Sinhala at all.
_LEGACY_TTS_REGISTRIES = frozenset({"elevenlabs", "gemini_tts"})


def requires_soniox_usage_correlation(profile: DemoProfile, config: Mapping[str, Any]) -> bool:
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


def profile_from_realtime_config(config: dict[str, Any]) -> DemoProfile:
    registry = load_registry()
    language = str(config.get("language") or "").strip().lower()
    if language not in registry.runtime_languages:
        raise ProductionConfigurationError("realtime-config language is absent from the worker capability registry")
    supplied_revision = str(config.get("capability_revision") or "").strip()
    if supplied_revision and supplied_revision != registry.revision:
        raise ProductionConfigurationError("realtime-config capability revision does not match this worker image")
    transcriber = config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}

    stt_registry = str(transcriber.get("registry_name") or "").lower()
    stt_model = str(transcriber.get("model") or "").strip()
    stt_provider = str(transcriber.get("provider") or stt_registry).lower()
    stt_capability = registry.capability("stt", stt_provider, stt_model)
    if stt_capability is not None:
        if not stt_capability["runnable"] or language not in stt_capability["languages"]:
            raise ProductionConfigurationError("realtime-config selects an unavailable STT capability")
        if stt_registry != stt_capability["registry_name"]:
            raise ProductionConfigurationError("realtime-config STT registry does not match the capability artifact")
    else:
        expected_stt_model = _LEGACY_STT_REGISTRY_MODELS.get(stt_registry)
        if (
            language not in registry.legacy_runtime_languages
            or registry.owns_component_provider("stt", stt_provider)
            or expected_stt_model is None
            or stt_model != expected_stt_model
        ):
            raise ProductionConfigurationError("realtime-config must resolve a supported STT provider/model")
    if str(model.get("registry_name") or "").lower() not in {"gemini", "google"}:
        raise ProductionConfigurationError("realtime-config must resolve Gemini LLM")
    tts_registry = str(voice.get("registry_name") or "").lower()
    voice_id = str(voice.get("voice_id") or "").strip()
    tts_model = str(voice.get("model") or "").strip()
    tts_provider = str(voice.get("provider") or tts_registry).lower()
    llm_model = str(model.get("model") or "").strip()
    instructions = str(config.get("system_prompt") or "").strip()
    first_message = str(config.get("first_message") or "").strip()
    if not all((voice_id, tts_model, llm_model, instructions, first_message)):
        raise ProductionConfigurationError("realtime-config omitted runnable prompt/provider values")
    # Keep the language policy error ahead of the generic registry error. A
    # malformed Estonian payload should still tell an operator its one valid
    # provider, rather than hiding that invariant behind "unsupported".
    if language == "et" and tts_registry != "elevenlabs":
        raise ProductionConfigurationError("Estonian requires ElevenLabs TTS")
    if language == "et" and "flash" in tts_model.lower():
        raise ProductionConfigurationError("Estonian requires an ElevenLabs multilingual model")
    if language == "si" and (stt_registry != "google" or tts_registry != "gemini_tts"):
        raise ProductionConfigurationError(
            "Sinhala requires Google Chirp STT and Gemini TTS (Deepgram and ElevenLabs do not "
            "support si-LK)"
        )
    tts_capability = registry.capability("tts", tts_provider, tts_model)
    if tts_capability is not None:
        if not tts_capability["runnable"] or language not in tts_capability["languages"]:
            raise ProductionConfigurationError("realtime-config selects an unavailable TTS capability")
        if tts_registry != tts_capability["registry_name"]:
            raise ProductionConfigurationError("realtime-config TTS registry does not match the capability artifact")
        try:
            registry.validate_runnable_voice(
                str(tts_capability["provider"]), tts_model, voice_id, language=language
            )
        except VoiceCapabilityError as exc:
            raise ProductionConfigurationError("realtime-config voice id is absent from the approved inventory") from exc
    elif (
        language not in registry.legacy_runtime_languages
        or registry.owns_component_provider("tts", tts_provider)
        or tts_registry not in _LEGACY_TTS_REGISTRIES
    ):
        raise ProductionConfigurationError("realtime-config must resolve a supported TTS provider/model")
    if stt_capability is not None or tts_capability is not None:
        if not supplied_revision:
            raise ProductionConfigurationError(
                "realtime-config capability revision is required for registry-backed components"
            )
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

    stt_language = str(transcriber.get("language") or language).strip()
    if language == "en" and stt_language == "en":
        stt_language = "en-US"
    if language == "si" and stt_language == "si":
        stt_language = "si-LK"
    return DemoProfile(
        language=language,
        stt_language=stt_language,
        tts_model=tts_model,
        voice_id=voice_id,
        llm_model=llm_model,
        instructions=instructions,
        first_message=first_message,
        stt_registry=stt_registry,
        tts_registry=tts_registry,
        stt_model=stt_model,
        # Preserve the durable configuration value.  Registry-backed stacks
        # have already proved this equals the packaged artifact above; legacy
        # EN/ET/SI stacks deliberately have no such value and must not acquire
        # one merely because this worker also ships the new capability source.
        capability_revision=supplied_revision,
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
        "stt": profile.stt_model or _LEGACY_STT_REGISTRY_MODELS.get(profile.stt_registry, ""),
        "llm": profile.llm_model,
        "tts": profile.tts_model,
    }


def _default_tool_builder(function: Callable[..., Any], raw_schema: dict[str, Any]) -> Any:
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
) -> None:
    """Run the worker's idempotent control-plane cleanup then delete its room."""

    call.cleanup(remove_participant=lambda: None, delete_room=lambda: None)
    await delete_room()


_PRE_MEDIA_REASON_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


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
            self._task = asyncio.create_task(self._terminate_once(safe_pre_media_reason(reason)))
        try:
            return await asyncio.shield(self._task)
        except asyncio.CancelledError:
            # The child task owns the external teardown.  Do not leave it
            # halfway through merely because LiveKit is stopping this worker.
            await asyncio.shield(self._task)
            raise

    async def _terminate_once(self, reason: str) -> dict[str, Any]:
        deadline = self._clock() + self._timeout_seconds
        actions = {
            "participant": asyncio.create_task(self._remove_participant()),
            "room": asyncio.create_task(self._delete_room()),
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
                await asyncio.wait_for(self._report_failure(reason), timeout=self._timeout_seconds)
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
) -> dict[str, Any]:
    """Bridge concrete LiveKit teardown to canonical ``sip.failed`` safely."""

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
        report_failure=report_failure if adapter is not None and resolved_call_id else None,
        mark_call_cleaned=mark_call_cleaned if call is not None else None,
    ).terminate(reason=reason)


class SemanticTurnAdmission:
    """Close transcript and conversation admission once terminal teardown starts."""

    def __init__(self) -> None:
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def admits_transcript(self) -> bool:
        return not self._closed

    def admits_conversation_item(self) -> bool:
        return not self._closed


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
    ) -> None:
        self._recorder = recorder
        self._telemetry = telemetry_collector
        self._admission = admission
        self._first_message = first_message
        self._provider_labels = dict(provider_labels)
        self._model_labels = dict(model_labels or {})
        self._emit = emit
        self.caller_turns = 0

    def on_user_state_changed(self, event: Any) -> None:
        self._recorder.mark_state("caller", str(event.old_state), str(event.new_state))

    def on_agent_state_changed(self, event: Any) -> None:
        self._recorder.mark_state("agent", str(event.old_state), str(event.new_state))

    def on_user_input_transcribed(self, event: Any) -> None:
        if not self._admission.admits_transcript():
            return
        if event.is_final and str(event.transcript).strip():
            self.caller_turns += 1
            self._recorder.add_text("caller", str(event.transcript))

    def on_conversation_item_added(self, event: Any) -> None:
        if not self._admission.admits_conversation_item():
            return
        item = event.item
        role = str(getattr(item, "role", ""))
        text = str(getattr(item, "text_content", "") or getattr(item, "raw_text_content", ""))
        metrics = getattr(item, "metrics", None)
        if role == "user":
            self._telemetry.observe_user(text=text, metrics=metrics)
            return
        if role != "assistant" or self.caller_turns < 1:
            return
        if is_greeting_fragment(text, self._first_message):
            return
        self._recorder.add_text("agent", text)
        turn_telemetry = self._telemetry.observe_assistant(
            text=text,
            metrics=metrics,
            provider_labels=self._provider_labels,
            model_labels=self._model_labels,
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


def write_session_state(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as recording:
        return recording.getnframes() / recording.getframerate()


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
) -> str:
    """Upload the call's evidence and durably record how complete it was.

    An artifact failure here happens AFTER the caller has hung up and
    after gateway ``sip.ended`` has already settled billing.  Raising out of
    this point bought nothing and cost two things -- an unhandled exception in
    the job task, and a session state that never reached ``complete``, so the
    reconciler later relabelled a healthy call ``worker_restart_orphan``.
    Classify the outcome in ``artifact_status`` and let teardown finish.
    """

    imported: dict[str, Any] = {}
    artifact_status = "unavailable_no_room_audio"
    if summary is not None:
        short_call = bool(summary["short_call_without_caller_speech"])
        try:
            # Reading the mixed recording is inside the guard on purpose: an
            # unreadable WAV here must classify the call, not escape teardown.
            duration = wav_duration(evidence_dir / "mixed.wav")
            imported = await asyncio.to_thread(
                call.complete,
                source_path=evidence_dir,
                duration_sources={
                    "sip_participant": duration,
                    "room_track": duration,
                    "recording": duration,
                },
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
            artifact_status = "finalization_failed"
            emit(
                "artifact_finalization_failed",
                semantic_evidence_status=summary["semantic_evidence_status"],
                caller_audio_observed=summary["caller_audio_observed"],
                short_call_without_caller_speech=short_call,
                error_type=type(exc).__name__,
            )
    write_session_state(
        state_path,
        call_id=call_id,
        room_name=room_name,
        started_at=started_at,
        last_heartbeat_at=ended_at,
        status="complete",
        artifact_status=artifact_status,
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
        billing_source="gateway_sip_lifecycle",
    )
    return artifact_status


_END_CALL_FALLBACK_MESSAGES = {
    "en": "Thank you for calling. Goodbye.",
    "et": "Aitäh helistamast. Head aega.",
    "si": "ඔබ ඇමතූවාට ස්තුතියි. ආයුබෝවන්.",
}


class WorkerLocalEndCall:
    """Perform the terminal ``end_call`` action inside the media worker.

    The control-plane endpoint intentionally remains a harmless acknowledgement
    for a misrouted request, but an admitted LiveKit model call must execute
    here: schedule one final non-interruptible message, wait for its playout
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
        record_final_message: Callable[[str], None] | None = None,
        emit: Callable[..., None] | None = None,
        is_finished: Callable[[], bool] | None = None,
        playout_timeout_seconds: float = 15.0,
    ) -> None:
        if playout_timeout_seconds <= 0:
            raise ValueError("end_call playout timeout must be positive")
        self._session = session
        self._language = language.strip().lower()
        self._on_ending = on_ending
        self._finish = finish
        self._record_final_message = record_final_message
        self._emit = emit
        self._is_finished = is_finished or (lambda: False)
        self._playout_timeout_seconds = playout_timeout_seconds
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[dict[str, object]] | None = None

    async def __call__(self, raw_arguments: dict[str, object]) -> dict[str, object]:
        async with self._lock:
            task = self._task
            if task is None:
                if self._is_finished():
                    return {
                        "ok": True,
                        "status": "already_finished",
                        "playout": "not_started",
                    }
                # Stop turn admission synchronously before returning the ACK.
                # The background task cannot run until this function yields,
                # which is precisely what lets the current tool speech handle
                # complete and the final ``say`` be scheduled safely.
                try:
                    self._on_ending()
                except Exception:
                    pass
                task = asyncio.create_task(self._execute(dict(raw_arguments)))
                self._task = task
        return {
            "ok": True,
            "status": "termination_scheduled",
            "playout": "pending",
        }

    async def wait_for_completion(self) -> dict[str, object] | None:
        """Await the worker-owned terminal flow without exposing it to the model."""

        task = self._task
        return await asyncio.shield(task) if task is not None else None

    def _final_message(self, raw_arguments: dict[str, object]) -> str:
        requested = raw_arguments.get("message")
        if isinstance(requested, str):
            message = " ".join(requested.split())
            if message:
                return message
        return _END_CALL_FALLBACK_MESSAGES.get(
            self._language, _END_CALL_FALLBACK_MESSAGES["en"]
        )

    async def _execute(self, raw_arguments: dict[str, object]) -> dict[str, object]:
        playout = "failed"
        message = self._final_message(raw_arguments)
        try:
            speech = self._session.say(
                message, allow_interruptions=False
            )
            try:
                await asyncio.wait_for(
                    speech.wait_for_playout(), timeout=self._playout_timeout_seconds
                )
            except TimeoutError:
                playout = "timeout"
            except Exception:
                playout = "failed"
            else:
                try:
                    playout = "failed" if speech.exception() is not None else "completed"
                except Exception:
                    playout = "failed"
            if playout == "completed" and self._record_final_message is not None:
                try:
                    self._record_final_message(message)
                except Exception:
                    pass
        except Exception:
            playout = "failed"
        finally:
            # ``shutdown`` starts LiveKit's orderly close; the owning worker
            # awaits ``aclose`` in its existing cleanup/finalization path.
            # Do this even if speech construction or playout failed so SIP BYE
            # and the caller's PSTN leg cannot remain orphaned.
            try:
                self._session.shutdown(drain=False)
            except Exception:
                pass
            try:
                self._finish()
            except Exception:
                pass
            if self._emit is not None:
                try:
                    self._emit("end_call", language=self._language, playout=playout)
                except Exception:
                    pass
        return {
            "ok": playout == "completed",
            "status": "teardown_requested",
            "playout": playout,
        }


def build_livekit_model_tools(
    session: ToolSession,
    *,
    end_call: Callable[[dict[str, object]], Any] | None = None,
    tool_usage_counter: Any | None = None,
    tool_builder: Callable[[Callable[..., Any], dict[str, Any]], Any] = _default_tool_builder,
) -> list[Any]:
    """Translate the filtered MCP allowlist into LiveKit model-callable tools.

    The model receives only schemas already filtered by
    :meth:`Gate2CallSession.tool_definitions`.  Each invocation crosses the
    same session interface, which rechecks the allowlist and creates the MCP
    audit event before returning output to the model. ``end_call`` is the only
    exception: its effect is owned by the running media worker, never by the
    control-plane tool endpoint.
    """

    tools: list[Any] = []
    for definition in session.tool_definitions():
        name = str(definition.get("name") or "").strip()
        parameters = definition.get("inputSchema")
        if not name or not isinstance(parameters, dict):
            raise ProductionConfigurationError("MCP tool definition is missing name/inputSchema")
        model_parameters = copy.deepcopy(parameters)
        properties = model_parameters.get("properties")
        required = model_parameters.get("required")
        tenant_arguments = {
            name: str(session.config.get(name) or "").strip()
            for name in ("organization_id", "project_id")
        }
        if not all(tenant_arguments.values()):
            raise ProductionConfigurationError(
                "realtime-config omitted the authenticated organization/project binding"
            )
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
            "description": str(definition.get("description") or f"OpenPhonex tool {name}"),
            "parameters": model_parameters,
        }
        if tool_usage_counter is not None:
            # Register the exact function schemas given to this call's model,
            # not a broad workspace allowlist. The counter retains only names
            # and completed counts, never model arguments or tool output.
            tool_usage_counter.register(name)

        if name == "end_call":
            async def invoke(
                raw_arguments: dict[str, object],
                *,
                _end_call: Callable[[dict[str, object]], Any] | None = end_call,
                _name: str = name,
                _tool_usage_counter: Any | None = tool_usage_counter,
            ) -> None:
                if _end_call is None:
                    raise ProductionConfigurationError(
                        "end_call requires the worker-local termination handler"
                    )
                result = await _end_call(dict(raw_arguments))
                if _tool_usage_counter is not None and _successful_tool_result(result):
                    _tool_usage_counter.record_success(_name)
                # The pinned LiveKit tool executor treats any non-None output
                # as reply_required. The worker has already scheduled terminal
                # playout, so returning None suppresses a model follow-up turn.
                return None
        else:
            async def invoke(
                raw_arguments: dict[str, object],
                *,
                _name: str = name,
                _tenant_arguments: dict[str, str] = tenant_arguments,
                _tool_usage_counter: Any | None = tool_usage_counter,
            ) -> str:
                arguments = dict(raw_arguments)
                arguments.update(_tenant_arguments)
                result = await asyncio.to_thread(
                    session.invoke_tool, _name, arguments
                )
                if _tool_usage_counter is not None and _successful_tool_result(result):
                    _tool_usage_counter.record_success(_name)
                return json.dumps(result, sort_keys=True, default=str)

        tools.append(tool_builder(invoke, raw_schema))
    return tools

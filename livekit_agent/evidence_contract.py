# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""PR #79 evidence invariants shared by the isolated LiveKit services."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

# Artifacts written before the rename carry the old schema id; both are read.
# Drop the legacy spelling once retention has aged those artifacts out.
EFFECTIVE_VOICE_STACK_SCHEMA = "skysay.effective_voice_stack.v1"
EFFECTIVE_VOICE_STACK_SCHEMAS = (EFFECTIVE_VOICE_STACK_SCHEMA, "openphonex.effective_voice_stack.v1")


_SEMANTIC_RESPONSE_AUDIO_POLICY: dict[str, object] = {
    "schema": "skysay.semantic_response_audio_policy.v1",
    "filler_acknowledgements_enabled": False,
    "first_post_caller_agent_audio": "assistant_spoken_semantic_response",
    "semantic_start_alignment": "caller_side_vad_worker_turn_clip_correlation",
}

# Keep evidence's configured/off labels tied to the worker's small resolved
# output contract.  In particular, ``status`` is *not* a worker field: the
# worker rejects it rather than treating it as evidence of a started track.
_OUTGOING_ENVIRONMENT_VERSION = "v1"
_OUTGOING_ENVIRONMENT_PRESET_IDS = frozenset(
    {"quiet-office", "call-centre", "cafe", "street", "television"}
)
_OUTGOING_ENVIRONMENT_ALLOWED_KEYS = frozenset(
    {
        "preset_id",
        "asset_version",
        "volume",
        "runtime_gain_linear",
        "runtime_gain_db",
        "channel",
    }
)
_OUTGOING_ENVIRONMENT_MAX_GAIN = 0.20


def semantic_response_audio_policy(
    *, filler_acknowledgements_enabled: bool = False
) -> dict[str, object]:
    """The bundle's statement of what the recorded agent track contains.

    Acknowledgements ("One moment") play on the separately published
    background track the sidecar deliberately does not record, so the semantic
    agent track keeps the same shape either way; the flag records that they
    were ENABLED for the call, so a reader knows the caller may have heard a
    filler the transcript does not show.
    """

    policy = dict(_SEMANTIC_RESPONSE_AUDIO_POLICY)
    policy["filler_acknowledgements_enabled"] = bool(filler_acknowledgements_enabled)
    return policy


def valid_semantic_response_audio_policy(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    flag = value.get("filler_acknowledgements_enabled")
    if not isinstance(flag, bool):
        return False
    return value == semantic_response_audio_policy(filler_acknowledgements_enabled=flag)


def _outgoing_environment_status(environment: Mapping[str, Any]) -> str:
    """Classify only a worker-acceptable resolved output selection.

    This deliberately duplicates no media observation: ``configured`` says
    that the control plane gave the worker a shape it accepts.  A later worker
    event remains the sole proof that the optional background track started.
    """

    if not environment:
        return "not_configured"
    if set(environment) - _OUTGOING_ENVIRONMENT_ALLOWED_KEYS:
        return "invalid"

    preset_id = str(environment.get("preset_id") or "").strip().lower()
    asset_version = str(environment.get("asset_version") or "").strip()
    volume = environment.get("volume")
    channel = str(environment.get("channel") or "")
    if asset_version != _OUTGOING_ENVIRONMENT_VERSION or type(volume) is not int:
        return "invalid"
    if preset_id == "off":
        return "off" if volume == 0 and channel == "none" else "invalid"
    if (
        preset_id not in _OUTGOING_ENVIRONMENT_PRESET_IDS
        or not 1 <= volume <= 100
        or channel != "outgoing_background_track"
    ):
        return "invalid"
    gain = environment.get("runtime_gain_linear")
    if (
        type(gain) not in (int, float)
        or isinstance(gain, bool)
        or not math.isfinite(float(gain))
    ):
        return "invalid"
    expected_gain = round(_OUTGOING_ENVIRONMENT_MAX_GAIN * volume / 100, 6)
    return "configured" if round(float(gain), 6) == expected_gain else "invalid"


def is_noncaller_ambience_track(
    *,
    publication_name: Any,
    participant_identity: Any,
    caller_identity: Any,
    agent_identity: Any,
    ambience_track_name: str,
) -> bool:
    """Whether a non-caller track is the outgoing ambience bed.

    The outgoing ambience belongs to this worker's own agent participant,
    rather than merely any non-caller in a room.  This prevents a bridge,
    observer, or other participant from suppressing its semantic audio by
    publishing a same-named track.  The sidecar's existing exact
    caller/agent role assignment remains at its call site.
    """

    return (
        bool(str(agent_identity or ""))
        and str(participant_identity or "") == str(agent_identity or "")
        and str(participant_identity or "") != str(caller_identity or "")
        and str(publication_name or "") == ambience_track_name
    )


STT_CANDIDATE_LOCALES_REQUESTED = "stt_candidate_locales_requested"


def stt_candidate_locales_requested(
    candidate_locales: Sequence[str],
    *,
    language: str,
    registry_name: str,
    model: str,
) -> dict[str, object] | None:
    """The row a call emits when it asks its transcriber for several locales.

    REQUESTED, not opened. This is the candidate list the adapter was
    constructed with, which is as much as this seam can honestly know: no
    provider confirms a recognition socket back to us, and a call can be
    cancelled, fail to authenticate or have its configuration normalised by
    the adapter after this point. A review round named the earlier
    ``_opened`` spelling as evidence that claimed more than the code could
    see, and it was right -- an evidence row that overstates is worse than no
    row, because the next investigation trusts it.

    What it does establish is the thing no other field can: the widening
    survived the declaration, the ceiling and the region, all of which are
    applied while the stack is built and none of which are visible in the
    call's configuration.

    ``None`` for every other call. A row on every call would restate
    configuration the evidence already carries.

    Returned rather than emitted so the decision can be driven without a
    LiveKit job: the caller adds its own ``call_id`` and hands it to the sink.
    """

    opened = [str(locale) for locale in candidate_locales if str(locale)]
    if len(opened) < 2:
        return None
    return {
        # Construction order, not sorted. It is what the adapter was handed,
        # and on an adapter that ranks candidates it is the ranking it got.
        "candidate_locales": opened,
        "language": str(language or ""),
        "stt": str(registry_name or ""),
        "model": str(model or ""),
    }


def effective_voice_stack_identity(
    config: Mapping[str, Any],
    *,
    filler_acknowledgements_enabled: bool = False,
    tool_wait_feedback: str = "off",
    stt_candidate_locales: Sequence[str] = (),
) -> dict[str, object]:
    """Fingerprint the secret-free semantic stack resolved for one call.

    ``filler_acknowledgements_enabled`` is part of the measured configuration:
    two otherwise identical calls, one with acknowledgements on, must not share
    a fingerprint, and the bundle's policy statements must agree with each
    other.

    ``tool_wait_feedback`` (the resolved ``voice_polish.tool_wait`` mode) is
    recorded the same way, but ONLY when it is on.  Cohorts are derived from
    this fingerprint, so an ``off`` call keeps the byte-identical configuration
    it had before the mode existed instead of splitting every cohort at the
    release that introduced it.

    ``stt_candidate_locales`` follows that rule exactly, and is the one field
    here that does NOT come from the config.  Every other line describes what
    the call asked for before the stack was built; this one is the candidate
    list the transcriber was actually constructed with, which is the only
    thing that can distinguish a call listening for two languages from one
    listening for the first of them.  It is what was REQUESTED of the adapter
    rather than a socket any provider confirmed.  Absent when a single locale
    was requested, so a call that did not widen keeps the fingerprint it had
    before this field existed.
    """

    transcriber = config.get("transcriber") if isinstance(config.get("transcriber"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}
    environment = (
        config.get("audio_environment")
        if isinstance(config.get("audio_environment"), dict)
        else {}
    )
    environment_preset = str(environment.get("preset_id") or "")
    environment_version = str(environment.get("asset_version") or "")
    environment_volume = environment.get("volume")
    environment_channel = str(environment.get("channel") or "")
    environment_status = _outgoing_environment_status(environment)
    prompt = str(config.get("system_prompt") or "")
    configuration: dict[str, object] = {
        "providers": {
            "stt": str(transcriber.get("registry_name") or ""),
            "llm": str(model.get("registry_name") or ""),
            "tts": str(voice.get("registry_name") or ""),
        },
        "stt": {
            "model_id": str(transcriber.get("model") or ""),
            "language": str(transcriber.get("language") or config.get("language") or ""),
        },
        "llm": {
            "model_id": str(model.get("model") or ""),
            "system_prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
        },
        "tts": {
            "model_id": str(voice.get("model") or ""),
            "voice_id": str(voice.get("voice_id") or ""),
            "language": str(voice.get("language") or config.get("language") or ""),
        },
        # The ambience is an independently published output track, not part of
        # the semantic TTS waveform.  Preserve only its pin and channel so an
        # artifact can explain what the worker was configured to attempt
        # without claiming a receiver-side post-mix, successful publication,
        # or leaking any unrelated worker config.  The later worker event is
        # the only observation that it actually started the optional track.
        "audio_environment": {
            "preset_id": environment_preset or "off",
            "asset_version": environment_version or "v1",
            "volume": environment_volume if type(environment_volume) is int else 0,
            "channel": environment_channel or "none",
            "status": environment_status,
        },
        "turns": {"language": str(config.get("language") or "")},
        "measurement": {
            "semantic_response_audio_policy": semantic_response_audio_policy(
                filler_acknowledgements_enabled=filler_acknowledgements_enabled
            ),
        },
    }
    candidates = [str(locale) for locale in stt_candidate_locales if str(locale)]
    if len(candidates) > 1:
        stt = configuration["stt"]
        assert isinstance(stt, dict)
        # The order is the adapter's, not sorted: two calls that asked for the
        # same locales in a different order are not the same configuration,
        # and this field exists to say which one a recording came from.
        stt["candidate_locales"] = candidates
    mode = str(tool_wait_feedback or "off")
    if mode != "off":
        measurement = configuration["measurement"]
        assert isinstance(measurement, dict)
        measurement["tool_wait_feedback"] = mode
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "schema": EFFECTIVE_VOICE_STACK_SCHEMA,
        "fingerprint": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "configuration": configuration,
    }


def valid_effective_voice_stack_identity(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema") not in EFFECTIVE_VOICE_STACK_SCHEMAS:
        return False
    configuration = value.get("configuration")
    if not isinstance(configuration, dict):
        return False
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    return value.get("fingerprint") == expected

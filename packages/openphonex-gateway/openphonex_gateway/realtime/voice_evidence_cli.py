# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .semantic_audio_policy import valid_semantic_response_audio_policy
from .voice_evidence import (
    HISTORY_SCHEMA,
    HISTORY_SCHEMA_VERSION,
    MEASUREMENT_SOURCE,
    analyze_split_tracks,
    append_history,
    evaluate_history,
    has_measurable_semantic_text,
)


COHORT_ENV_ALLOWLIST = (
    "VOICE_RTP_BIND",
    "VOICE_RTP_PORT",
    "VOICE_RTP_PORT_END",
    "RTP_START",
    "REALTIME_VOICE_RTP_PORT_END",
    "REALTIME_RTP_PORT_POOL_SIZE",
    "REALTIME_MAX_CONCURRENT_CALLS",
    "REALTIME_RTP_RECEIVE_QUEUE_FRAMES",
    "REALTIME_PROVIDER_MODE",
    "REALTIME_STT_PROVIDER",
    "DEEPGRAM_MODEL_ID",
    "DEEPGRAM_LANGUAGE",
    "DEEPGRAM_ENDPOINTING_MS",
    "DEEPGRAM_UTTERANCE_END_MS",
    "DEEPGRAM_TIMEOUT_SECONDS",
    "DEEPGRAM_FINALIZE_TIMEOUT_SECONDS",
    "DEEPGRAM_TTS_MODEL",
    "DEEPGRAM_TTS_READ_BYTES",
    "DEEPGRAM_TTS_TIMEOUT_SECONDS",
    "REALTIME_LLM_PROVIDER",
    "REALTIME_LLM_MODEL",
    "REALTIME_LLM_SYSTEM_PROMPT",
    "REALTIME_LLM_BASE_URL",
    "REALTIME_LLM_TEMPERATURE",
    "REALTIME_LLM_TIMEOUT_SECONDS",
    "GEMINI_LLM_MODEL",
    "GEMINI_SYSTEM_PROMPT",
    "GEMINI_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_THINKING_BUDGET",
    "GEMINI_TIMEOUT_SECONDS",
    "REALTIME_LLM_FALLBACK_PROVIDER",
    "REALTIME_LLM_FALLBACK_MODEL",
    "REALTIME_LLM_RETRY_BUDGET_SECONDS",
    "REALTIME_LLM_MAX_TOKENS",
    "OPENAI_MAX_TOKENS",
    "ANTHROPIC_MAX_TOKENS",
    "OPENAI_BASE_URL",
    "OPENAI_LLM_MODEL",
    "OPENAI_SYSTEM_PROMPT",
    "OPENAI_STT_MODEL",
    "OPENAI_STT_TIMEOUT_SECONDS",
    "OPENAI_TEMPERATURE",
    "OPENAI_TIMEOUT_SECONDS",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SYSTEM_PROMPT",
    "ANTHROPIC_TEMPERATURE",
    "ANTHROPIC_TIMEOUT_SECONDS",
    "ANTHROPIC_VERSION",
    "REALTIME_TTS_PROVIDER",
    "REALTIME_TTS_MODEL",
    "REALTIME_TTS_VOICE",
    "REALTIME_TTS_SAMPLE_RATE",
    "REALTIME_TTS_LANG",
    "REALTIME_TTS_SPEAKER",
    "ELEVENLABS_VOICE_ID",
    "ELEVENLABS_VOICE",
    "ELEVENLABS_MODEL_ID",
    "ELEVENLABS_OUTPUT_FORMAT",
    "ELEVENLABS_LANGUAGE_CODE",
    "ELEVENLABS_STABILITY",
    "ELEVENLABS_SIMILARITY_BOOST",
    "ELEVENLABS_CHUNK_SIZE",
    "ELEVENLABS_SCRIBE_MODEL",
    "ELEVENLABS_SCRIBE_TIMEOUT_SECONDS",
    "CARTESIA_LANGUAGE",
    "CARTESIA_MODEL",
    "CARTESIA_TTS_READ_BYTES",
    "CARTESIA_TTS_TIMEOUT_SECONDS",
    "CARTESIA_VERSION",
    "CARTESIA_VOICE_ID",
    "GOOGLE_STT_LANGUAGE",
    "GOOGLE_STT_MODEL",
    "GOOGLE_STT_SAMPLE_RATE",
    "GOOGLE_STT_TIMEOUT_SECONDS",
    "REALTIME_LANGUAGE",
    "REALTIME_MAX_TURNS",
    "REALTIME_SILENCE_HOLD_MS",
    "REALTIME_MIN_SPEECH_MS",
    "REALTIME_BARGE_IN_RMS",
    "REALTIME_SEMANTIC_ENDPOINTING_ENABLED",
    "REALTIME_SEMANTIC_ENDPOINTING_MIN_WORDS",
    "REALTIME_SEMANTIC_ENDPOINTING_MIN_CHARS",
    "REALTIME_SEMANTIC_ENDPOINTING_TERMINAL_PUNCTUATION",
    "REALTIME_ACCEPT_INTERIM_ON_RMS_ENDPOINT",
    "REALTIME_MAX_PENDING_PHRASES",
    "REALTIME_TTS_RTP_PREBUFFER_MS",
    "REALTIME_IDLE_TIMEOUT_SECONDS",
    "REALTIME_RECOVERY_UTTERANCE",
    "REALTIME_OPENING_UTTERANCE",
    "REALTIME_INBOUND_OPENING_UTTERANCE",
    "REALTIME_OUTBOUND_ANSWER_TIMEOUT_SECONDS",
    "REALTIME_FAKE_REPLY",
    "REALTIME_FAKE_TRANSCRIPT",
    "REALTIME_LOCAL_STT_COMPUTE_TYPE",
    "REALTIME_LOCAL_STT_DEVICE",
    "REALTIME_LOCAL_STT_MODEL",
    "REALTIME_TTS_PIPER_BIN",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure caller-side OpenPhonex voice evidence from synchronized WAV tracks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze one split-track self-test recording and append its evidence.",
    )
    analyze.add_argument("--caller-spoken", type=Path, required=True)
    analyze.add_argument("--agent-heard", type=Path, required=True)
    analyze.add_argument("--transcript", type=Path, required=True)
    analyze.add_argument("--cohort", type=_nonempty_text, required=True)
    analyze.add_argument("--campaign", type=_sha256_text, required=True)
    analyze.add_argument("--ledger-id", type=_sha256_text, required=True)
    analyze.add_argument("--run-id", type=_nonempty_text, required=True)
    analyze.add_argument("--topology", type=_nonempty_text, required=True)
    analyze.add_argument("--route", type=_nonempty_text, required=True)
    analyze.add_argument("--context", type=_nonempty_text, required=True)
    analyze.add_argument("--runtime-gate", choices=("pass", "fail"), required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--history", type=Path, required=True)
    analyze.add_argument("--max-semantic-ms", type=_positive_float, default=1_500.0)
    analyze.add_argument("--min-semantic-turns", type=_positive_int, default=1)
    analyze.add_argument("--max-barge-ms", type=_positive_float, default=400.0)
    analyze.add_argument("--require-barge-in", action="store_true")

    history_gate = subparsers.add_parser(
        "history-gate",
        help="Evaluate cohort-isolated release distribution gates from JSONL history.",
    )
    history_gate.add_argument("--history", type=Path, required=True)
    history_gate.add_argument("--cohort", type=_nonempty_text, required=True)
    history_gate.add_argument("--base-cohort", type=_nonempty_text)
    history_gate.add_argument("--campaign", type=_sha256_text, required=True)
    history_gate.add_argument("--ledger-id", type=_sha256_text, required=True)
    history_gate.add_argument("--topology", type=_nonempty_text, required=True)
    history_gate.add_argument("--route", type=_nonempty_text, required=True)
    history_gate.add_argument("--integrity-events", type=Path, required=True)
    history_gate.add_argument("--output", type=Path)
    history_gate.add_argument("--min-semantic-samples", type=_positive_int, default=30)
    history_gate.add_argument("--semantic-p90-ms", type=_positive_float, default=1_200.0)
    history_gate.add_argument("--semantic-p95-ms", type=_positive_float, default=1_500.0)
    history_gate.add_argument("--min-barge-samples", type=_positive_int, default=10)
    history_gate.add_argument("--barge-p90-ms", type=_positive_float, default=250.0)
    history_gate.add_argument("--barge-max-ms", type=_positive_float, default=400.0)
    history_gate.add_argument("--max-age-hours", type=_positive_float, default=14 * 24)

    cohort = subparsers.add_parser(
        "cohort",
        help="Derive an opaque cohort from image identity and non-secret runtime tuning.",
    )
    cohort.add_argument("--image-identity", type=_nonempty_text, required=True)
    cohort.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "cohort":
        return _run_cohort(args)
    return _run_history_gate(args)


def _run_analyze(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="openphonex-voice-evidence-") as temporary:
        snapshot_root = Path(temporary)
        transcript, transcript_payload, transcript_error = _read_json_object_once(args.transcript)
        semantic_turn_paths, clip_error = _semantic_turn_clip_paths(transcript, args.transcript.parent)
        caller_snapshot, caller_payload = _snapshot_file(
            args.caller_spoken,
            snapshot_root / "caller-spoken.wav",
        )
        agent_snapshot, agent_payload = _snapshot_file(
            args.agent_heard,
            snapshot_root / "agent-heard.wav",
        )
        semantic_turn_clips: dict[int, Path] = {}
        fingerprint_inputs: list[tuple[str, bytes | None]] = [
            ("caller_spoken", caller_payload),
            ("agent_heard", agent_payload),
            ("transcript", transcript_payload),
        ]
        for turn, source_path in sorted(semantic_turn_paths.items()):
            snapshot_path, payload = _snapshot_file(
                source_path,
                snapshot_root / f"turn-{turn:03d}-agent.wav",
            )
            fingerprint_inputs.append((f"semantic_turn_{turn:03d}", payload))
            if payload is None:
                clip_error = "semantic_turn_clip_missing"
            else:
                semantic_turn_clips[turn] = snapshot_path
        source_evidence_fingerprint = _source_evidence_fingerprint(fingerprint_inputs)
        evidence = analyze_split_tracks(
            caller_snapshot,
            agent_snapshot,
            transcript=transcript,
            semantic_turn_clips=semantic_turn_clips,
        )
    if transcript_error:
        evidence = replace(
            evidence,
            valid=False,
            semantic_turns=(),
            barge_ins=(),
            invalid_reasons=(transcript_error,),
        )
    semantic_policy = transcript.get("semantic_response_audio_policy") if isinstance(transcript, dict) else None
    if not transcript_error and not valid_semantic_response_audio_policy(semantic_policy):
        evidence = replace(
            evidence,
            valid=False,
            semantic_turns=(),
            barge_ins=(),
            invalid_reasons=("semantic_response_audio_policy_invalid",),
        )
    elif not transcript_error and clip_error:
        evidence = replace(
            evidence,
            valid=False,
            semantic_turns=(),
            barge_ins=(),
            invalid_reasons=(clip_error,),
        )
    stack_fingerprint, stack_error = _effective_stack_fingerprint(transcript)
    openphonex = transcript.get("openphonex") if isinstance(transcript, dict) else None
    agent_id = str(openphonex.get("agent_id") or "").strip() if isinstance(openphonex, dict) else ""
    stack_identity_present = isinstance(transcript, dict) and "effective_voice_stack" in transcript
    if (agent_id or stack_identity_present) and stack_error:
        evidence = replace(
            evidence,
            valid=False,
            semantic_turns=(),
            barge_ins=(),
            invalid_reasons=(stack_error,),
        )
    effective_cohort = (
        _derive_effective_cohort(args.cohort, stack_fingerprint)
        if agent_id and stack_fingerprint
        else args.cohort
    )
    semantic_values = [turn.semantic_voice_to_voice_ms for turn in evidence.semantic_turns]
    barge_values = [barge.audible_stop_ms for barge in evidence.barge_ins]
    semantic_complete = len(semantic_values) >= args.min_semantic_turns
    semantic_within_limit = bool(semantic_values) and max(semantic_values) <= args.max_semantic_ms
    semantic_ok = semantic_complete and semantic_within_limit
    barge_complete = not args.require_barge_in or bool(barge_values)
    barge_within_limit = not barge_values or max(barge_values) <= args.max_barge_ms
    barge_ok = barge_complete and barge_within_limit
    runtime_ok = args.runtime_gate == "pass"
    measurement_complete = evidence.valid and semantic_complete and barge_complete and runtime_ok
    smoke_ok = evidence.valid and semantic_ok and barge_ok and runtime_ok
    payload: dict[str, Any] = {
        "schema": HISTORY_SCHEMA,
        "schema_version": HISTORY_SCHEMA_VERSION,
        "source": MEASUREMENT_SOURCE,
        "cohort": effective_cohort,
        "base_cohort": args.cohort,
        "effective_voice_stack_fingerprint": stack_fingerprint,
        "semantic_response_audio_policy": semantic_policy,
        "campaign_id": args.campaign,
        "ledger_id": args.ledger_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_evidence_fingerprint": source_evidence_fingerprint,
        "run_id": args.run_id,
        "topology": args.topology,
        "route": args.route,
        "context": args.context,
        **asdict(evidence),
        "runtime_gate": {"ok": runtime_ok, "status": args.runtime_gate},
        "measurement_complete": measurement_complete,
        "smoke_gate": {
            "ok": smoke_ok,
            "semantic": {
                "ok": semantic_ok,
                "complete": semantic_complete,
                "within_limit": semantic_within_limit,
                "sample_count": len(semantic_values),
                "min_samples": args.min_semantic_turns,
                "max_ms": max(semantic_values) if semantic_values else None,
                "limit_ms": args.max_semantic_ms,
            },
            "barge_in": {
                "ok": barge_ok,
                "complete": barge_complete,
                "within_limit": barge_within_limit,
                "required": args.require_barge_in,
                "sample_count": len(barge_values),
                "max_ms": max(barge_values) if barge_values else None,
                "limit_ms": args.max_barge_ms,
            },
        },
    }
    _write_json(args.output, payload)
    append_history(args.history, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if smoke_ok else 1


def _run_history_gate(args: argparse.Namespace) -> int:
    records, malformed_records = _read_jsonl(args.history)
    integrity_events, malformed_integrity_events = _read_jsonl(args.integrity_events)
    gate = evaluate_history(
        records,
        cohort=args.cohort,
        base_cohort=args.base_cohort,
        campaign_id=args.campaign,
        ledger_id=args.ledger_id,
        topology=args.topology,
        route=args.route,
        min_semantic_samples=args.min_semantic_samples,
        semantic_p90_limit_ms=args.semantic_p90_ms,
        semantic_p95_limit_ms=args.semantic_p95_ms,
        min_barge_samples=args.min_barge_samples,
        barge_p90_limit_ms=args.barge_p90_ms,
        barge_max_limit_ms=args.barge_max_ms,
        malformed_records=malformed_records,
        integrity_events=integrity_events,
        malformed_integrity_events=malformed_integrity_events,
        max_age_seconds=args.max_age_hours * 60 * 60,
    )
    if args.output:
        _write_json(args.output, gate)
    print(json.dumps(gate, sort_keys=True))
    return 0 if gate["ok"] else 1


def _run_cohort(args: argparse.Namespace) -> int:
    digest = hashlib.sha256()
    _hash_field(digest, "schema", "openphonex-voice-evidence-cohort-v1")
    _hash_field(digest, "image_identity", args.image_identity)
    for name in _cohort_env_names():
        _hash_field(digest, name, os.environ.get(name, ""))
    identity = f"sha256:{digest.hexdigest()}"
    if args.output:
        _write_text(args.output, identity + "\n")
    print(identity)
    return 0


def _cohort_env_names() -> list[str]:
    return sorted(COHORT_ENV_ALLOWLIST)


def _effective_stack_fingerprint(transcript: dict[str, Any]) -> tuple[str, str | None]:
    identity = transcript.get("effective_voice_stack") if isinstance(transcript, dict) else None
    if identity is None:
        return "", "effective_voice_stack_identity_missing"
    if not isinstance(identity, dict):
        return "", "effective_voice_stack_identity_invalid"
    if identity.get("schema") != "openphonex.effective_voice_stack.v1":
        return "", "effective_voice_stack_identity_invalid"
    configuration = identity.get("configuration")
    fingerprint = identity.get("fingerprint")
    if not isinstance(configuration, dict) or not isinstance(fingerprint, str):
        return "", "effective_voice_stack_identity_invalid"
    measurement = configuration.get("measurement")
    identity_policy = (
        measurement.get("semantic_response_audio_policy")
        if isinstance(measurement, dict)
        else None
    )
    if not valid_semantic_response_audio_policy(identity_policy):
        return "", "effective_voice_stack_identity_invalid"
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    if fingerprint != expected:
        return "", "effective_voice_stack_identity_invalid"
    return fingerprint, None


def _semantic_turn_clip_paths(
    transcript: dict[str, Any],
    transcript_dir: Path,
) -> tuple[dict[int, Path], str | None]:
    turns = transcript.get("turns") if isinstance(transcript, dict) else None
    if not isinstance(turns, list):
        return {}, "semantic_turn_clip_mapping_invalid"
    paths: dict[int, Path] = {}
    for index, turn in enumerate(turns, start=1):
        if not has_measurable_semantic_text(turn, clip_backed=True):
            continue
        if not isinstance(turn, dict):
            return {}, "semantic_turn_clip_mapping_invalid"
        expected_name = f"turn-{index:03d}-agent.wav"
        raw_name = turn.get("agent_audio")
        if not isinstance(raw_name, str) or raw_name != expected_name:
            return {}, "semantic_turn_clip_mapping_invalid"
        if Path(raw_name).name != raw_name or Path(raw_name).is_absolute():
            return {}, "semantic_turn_clip_mapping_invalid"
        paths[index] = transcript_dir / raw_name
    return paths, None

def _derive_effective_cohort(base_cohort: str, stack_fingerprint: str) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, "schema", "openphonex-voice-evidence-effective-cohort-v1")
    _hash_field(digest, "base_cohort", base_cohort)
    _hash_field(digest, "effective_voice_stack", stack_fingerprint)
    return f"sha256:{digest.hexdigest()}"


def _source_evidence_fingerprint(inputs: list[tuple[str, bytes | None]]) -> str:
    """Bind one history sample to the exact source-call artifact bundle."""

    digest = hashlib.sha256()
    _hash_field(digest, "schema", "openphonex-source-evidence-v1")
    for label, payload in inputs:
        if payload is None:
            _hash_field(digest, label, "unreadable")
        else:
            _hash_field(digest, label, f"sha256:{hashlib.sha256(payload).hexdigest()}")
    return f"sha256:{digest.hexdigest()}"


def _hash_field(digest: Any, name: str, value: str) -> None:
    for part in (name.encode("utf-8"), str(value).encode("utf-8")):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)


def _read_json_object_once(path: Path) -> tuple[dict[str, Any], bytes | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError:
        return {}, None, "transcript_unreadable"
    try:
        payload = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}, raw, "transcript_unreadable"
    if not isinstance(payload, dict):
        return {}, raw, "transcript_invalid"
    return payload, raw, None


def _snapshot_file(source: Path, destination: Path) -> tuple[Path, bytes | None]:
    try:
        payload = source.read_bytes()
    except OSError:
        return destination, None
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination, payload


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 1
    records: list[dict[str, Any]] = []
    malformed_records = 0
    for line in lines:
        try:
            payload = _strict_json_loads(line)
        except ValueError:
            malformed_records += 1
            continue
        if isinstance(payload, dict):
            records.append(payload)
        else:
            malformed_records += 1
    return records, malformed_records


def _strict_json_loads(raw: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonempty_text(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("must not be empty")
    return parsed


def _sha256_text(value: str) -> str:
    parsed = _nonempty_text(value)
    if len(parsed) != 71 or not parsed.startswith("sha256:") or any(
        character not in "0123456789abcdef" for character in parsed[7:]
    ):
        raise argparse.ArgumentTypeError("expected sha256:<64 lowercase hex characters>")
    return parsed


if __name__ == "__main__":
    sys.exit(main())

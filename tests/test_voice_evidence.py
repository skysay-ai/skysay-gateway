from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import stat
import struct
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from skysay_gateway.evidence import voice_evidence as voice_evidence_module
from skysay_gateway.evidence.voice_evidence import (
    VadConfig,
    analyze_split_tracks,
    append_history,
    evaluate_history as _evaluate_history,
)
from skysay_gateway.evidence.voice_evidence_cli import (
    _source_evidence_fingerprint,
    main as voice_evidence_main,
)


SAMPLE_RATE = 8_000
HISTORY_SCHEMA = "skysay.voice_evidence.v1"
MEASUREMENT_SOURCE = "caller_side_mixmonitor_split_track_vad_semantic_clip_correlation"
UNVERIFIED_MEASUREMENT_SOURCE = "caller_side_mixmonitor_split_track_vad_unverified"
SEMANTIC_ALIGNMENT_SOURCE = "caller_side_vad_worker_turn_clip_correlation"
UNVERIFIED_ALIGNMENT_SOURCE = "unverified_first_agent_vad"
RECORDED_AT = datetime.now(timezone.utc).isoformat()
CAMPAIGN_ID = "sha256:" + "c" * 64
LEDGER_ID = "sha256:" + "e" * 64
SEMANTIC_RESPONSE_AUDIO_POLICY = {
    "schema": "skysay.semantic_response_audio_policy.v1",
    "filler_acknowledgements_enabled": False,
    "first_post_caller_agent_audio": "assistant_spoken_semantic_response",
    "semantic_start_alignment": SEMANTIC_ALIGNMENT_SOURCE,
}


class EvidencePackageBoundaryTests(unittest.TestCase):
    def test_referee_is_explicit_and_retired_runtime_is_not_importable(self) -> None:
        self.assertIsNotNone(
            importlib.import_module("skysay_gateway.evidence.voice_evidence")
        )
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("skysay_gateway.realtime")


def _write_track(
    path: Path,
    *,
    duration_ms: int,
    tones: list[tuple[int, int] | tuple[int, int, float]],
    noise_amplitude: int = 0,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    samples = [
        round(noise_amplitude * math.sin(2 * math.pi * 80 * index / sample_rate))
        for index in range(round(sample_rate * duration_ms / 1_000))
    ]
    for tone in tones:
        start_ms, end_ms = tone[:2]
        frequency_hz = float(tone[2]) if len(tone) == 3 else 440.0
        start = round(sample_rate * start_ms / 1_000)
        end = round(sample_rate * end_ms / 1_000)
        for index in range(start, min(end, len(samples))):
            samples[index] = round(
                8_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
            )
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def _write_track_with_clip_copies(
    path: Path,
    clip_path: Path,
    *,
    duration_ms: int,
    start_times_ms: list[int],
) -> None:
    with wave.open(str(clip_path), "rb") as clip_handle:
        sample_rate = clip_handle.getframerate()
        clip_frames = clip_handle.getnframes()
        clip_payload = clip_handle.readframes(clip_frames)
    clip_samples = list(struct.unpack(f"<{clip_frames}h", clip_payload))
    samples = [0] * round(sample_rate * duration_ms / 1_000)
    for start_ms in start_times_ms:
        start_sample = round(sample_rate * start_ms / 1_000)
        samples[start_sample : start_sample + clip_frames] = clip_samples
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def _history_record(
    run_id: str,
    *,
    cohort: str = "flash-v1",
    semantic_ms: float = 1_000.0,
    barge_ms: float | None = None,
    source: str = MEASUREMENT_SOURCE,
    valid: bool = True,
    measurement_complete: bool = True,
    topology: str = "pstn-loopback",
    route: str = "didww-outbound-us",
    base_cohort: str | None = None,
    recorded_at: str = RECORDED_AT,
    campaign_id: str = CAMPAIGN_ID,
    ledger_id: str = LEDGER_ID,
    source_evidence_fingerprint: str | None = None,
) -> dict[str, object]:
    caller_end_sample = SAMPLE_RATE
    agent_start_sample = caller_end_sample + round(SAMPLE_RATE * semantic_ms / 1_000)
    caller_barge_sample = SAMPLE_RATE * 3
    agent_stop_sample = (
        caller_barge_sample + round(SAMPLE_RATE * barge_ms / 1_000)
        if barge_ms is not None
        else None
    )
    return {
        "schema": HISTORY_SCHEMA,
        "schema_version": 2,
        "source": source,
        "run_id": run_id,
        "source_evidence_fingerprint": (
            source_evidence_fingerprint
            or f"sha256:{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}"
        ),
        "topology": topology,
        "route": route,
        "context": "selftest",
        "cohort": cohort,
        "base_cohort": base_cohort or cohort,
        "semantic_response_audio_policy": dict(SEMANTIC_RESPONSE_AUDIO_POLICY),
        "semantic_alignment_source": SEMANTIC_ALIGNMENT_SOURCE,
        "recorded_at": recorded_at,
        "campaign_id": campaign_id,
        "ledger_id": ledger_id,
        "valid": valid,
        "measurement_complete": measurement_complete,
        "sample_rate_hz": SAMPLE_RATE,
        "runtime_gate": {
            "ok": measurement_complete,
            "status": "pass" if measurement_complete else "fail",
        },
        "semantic_turns": [
            {
                "semantic_voice_to_voice_ms": semantic_ms,
                "turn": 1,
                "caller_speech_end_sample": caller_end_sample,
                "agent_speech_start_sample": agent_start_sample,
                "reference_sha256": "sha256:" + "a" * 64,
                "reference_vad_start_sample": 0,
                "alignment_sample_count": SAMPLE_RATE // 2,
                "alignment_score": 0.95,
                "runner_up_margin": 0.2,
            }
        ],
        "barge_ins": [
            {
                "audible_stop_ms": barge_ms,
                "turn": 1,
                "caller_speech_start_sample": caller_barge_sample,
                "agent_speech_end_sample": agent_stop_sample,
            }
        ]
        if barge_ms is not None
        else [],
    }


def _measurement_transcript(
    *, turns: list[dict[str, object]], **extra: object
) -> dict[str, object]:
    normalized_turns = [
        {
            **turn,
            "agent_audio": turn.get("agent_audio") or f"turn-{index:03d}-agent.wav",
        }
        for index, turn in enumerate(turns, start=1)
    ]
    return {
        "semantic_response_audio_policy": dict(SEMANTIC_RESPONSE_AUDIO_POLICY),
        "turns": normalized_turns,
        **extra,
    }


def _integrity_event(
    run_id: str,
    *,
    event: str = "failure",
    sequence: int = 1,
    cohort: str = "flash-v1",
    topology: str = "pstn-loopback",
    route: str = "didww-outbound-us",
    context: str = "caller-sim",
    reason: str = "history_append_failed",
    campaign_id: str = CAMPAIGN_ID,
    ledger_id: str = LEDGER_ID,
    evidence_record_hash: str = "none",
) -> dict[str, object]:
    if event == "attempt_finalized" and evidence_record_hash == "none":
        evidence_record_hash = "sha256:" + "f" * 64
    unsigned: dict[str, object] = {
        "schema": "openphonex.selftest_integrity_event.v2",
        "event": event,
        "sequence": sequence,
        "previous_hash": "sha256:" + "1" * 64,
        "run_id": run_id,
        "cohort": cohort,
        "topology": topology,
        "route": route,
        "context": context,
        "reason": reason,
        "evidence_record_hash": evidence_record_hash,
        "campaign_id": campaign_id,
        "ledger_id": ledger_id,
        "recorded_at": RECORDED_AT,
    }
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        **unsigned,
        "entry_hash": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    }


def _write_default_semantic_clip(root: Path, *, frequency_hz: float = 440.0) -> Path:
    path = root / "turn-001-agent.wav"
    _write_track(
        path,
        duration_ms=500,
        tones=[(0, 160, frequency_hz), (260, 500, frequency_hz)],
    )
    return path


def _semantic_pattern(
    offset_ms: int, *, frequency_hz: float = 440.0
) -> list[tuple[int, int, float]]:
    return [
        (offset_ms, offset_ms + 160, frequency_hz),
        (offset_ms + 260, offset_ms + 500, frequency_hz),
    ]


def _effective_stack_identity(configuration: dict[str, object]) -> dict[str, object]:
    configuration = {
        **configuration,
        "measurement": {
            "semantic_response_audio_policy": dict(SEMANTIC_RESPONSE_AUDIO_POLICY),
        },
    }
    canonical = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        "schema": "skysay.effective_voice_stack.v1",
        "fingerprint": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
        "configuration": configuration,
    }


def evaluate_history(
    records: list[dict[str, object]], **kwargs: object
) -> dict[str, object]:
    if "integrity_events" not in kwargs:
        kwargs["integrity_events"] = _integrity_pairs_for_history(records)
    return _evaluate_history(
        records,
        campaign_id=CAMPAIGN_ID,
        ledger_id=LEDGER_ID,
        **kwargs,
    )


def _canonical_record_hash(record: dict[str, object]) -> str:
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _integrity_pairs_for_history(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    sequence = 1
    for record in records:
        if not isinstance(record, dict):
            continue
        required = (
            "run_id",
            "campaign_id",
            "ledger_id",
            "base_cohort",
            "topology",
            "route",
            "context",
        )
        if any(
            not isinstance(record.get(name), str) or not record.get(name)
            for name in required
        ):
            continue
        key = tuple(str(record[name]) for name in required)
        if key in seen:
            continue
        seen.add(key)
        common = {
            "cohort": str(record["base_cohort"]),
            "topology": str(record["topology"]),
            "route": str(record["route"]),
            "context": str(record["context"]),
            "campaign_id": str(record["campaign_id"]),
            "ledger_id": str(record["ledger_id"]),
        }
        events.append(
            _integrity_event(
                str(record["run_id"]),
                event="attempt_started",
                sequence=sequence,
                reason="call_origination_started",
                **common,
            )
        )
        sequence += 1
        events.append(
            _integrity_event(
                str(record["run_id"]),
                event="attempt_finalized",
                sequence=sequence,
                reason="evidence_persisted",
                evidence_record_hash=_canonical_record_hash(record),
                **common,
            )
        )
        sequence += 1
    return events


def _write_integrity_events(
    path: Path,
    records: list[dict[str, object]],
    *,
    extra: list[dict[str, object]] | None = None,
) -> None:
    events = [*_integrity_pairs_for_history(records), *(extra or [])]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


class VoiceEvidenceTests(unittest.TestCase):
    def test_history_file_is_created_root_only_regardless_of_process_umask(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td, patch("os.umask") as mocked_umask:
            # Patch only proves append_history does not rely on changing the
            # process-wide umask; os.open's explicit 0600 mode remains real.
            mocked_umask.return_value = 0o022
            history = Path(td) / "history.jsonl"
            append_history(history, _history_record("secure-run"))

            self.assertEqual(stat.S_IMODE(history.stat().st_mode), 0o600)

    def test_measures_semantic_response_from_synchronized_audio_and_ignores_greeting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=5_000, tones=[(2_000, 2_400)])
            _write_track(
                heard,
                duration_ms=5_000,
                tones=[(500, 1_000), *_semantic_pattern(2_700)],
            )
            semantic_clip = _write_default_semantic_clip(root)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Hello", "assistant_spoken": "Hi there"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual(len(evidence.semantic_turns), 1)
            self.assertAlmostEqual(
                evidence.semantic_turns[0].semantic_voice_to_voice_ms, 300.0, delta=20.0
            )
            self.assertEqual(evidence.measurement_source, MEASUREMENT_SOURCE)

    def test_barge_run_aligns_the_longest_audible_prefix_before_interruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            clip = root / "turn-001-agent.wav"
            response_pattern = [
                (0, 160, 440.0),
                (260, 500, 440.0),
                (520, 800, 660.0),
            ]
            _write_track(caller, duration_ms=3_000, tones=[(200, 600), (1_850, 2_450)])
            _write_track(
                heard,
                duration_ms=3_000,
                tones=[
                    (1_300 + start, 1_300 + end, frequency)
                    for start, end, frequency in response_pattern
                ],
            )
            _write_track(clip, duration_ms=800, tones=response_pattern)

            transcript_payload = _measurement_transcript(
                turns=[
                    {
                        "user": "First",
                        "assistant": "Semantic answer",
                        # Runtime only commits fully played phrases to
                        # assistant_spoken. A real mid-phrase barge-in can
                        # therefore leave it empty even though the exact
                        # recorded turn clip proves semantic audio played.
                        "assistant_spoken": "",
                        "interrupted": True,
                    },
                    # Barge timing is evidence about the interrupted prior
                    # turn; it must not depend on a later response completing.
                    {"user": "Interrupt"},
                ]
            )
            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=transcript_payload,
                semantic_turn_clips={1: clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual(len(evidence.semantic_turns), 1)
            self.assertEqual(len(evidence.barge_ins), 1)
            self.assertGreaterEqual(
                evidence.semantic_turns[0].alignment_sample_count, SAMPLE_RATE // 2
            )
            self.assertLess(
                evidence.semantic_turns[0].alignment_sample_count,
                SAMPLE_RATE * 800 // 1_000,
            )

            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            transcript.write_text(json.dumps(transcript_payload))
            cli_args = [
                "analyze",
                "--caller-spoken",
                str(caller),
                "--agent-heard",
                str(heard),
                "--transcript",
                str(transcript),
                "--cohort",
                "flash-v1",
                "--campaign",
                CAMPAIGN_ID,
                "--ledger-id",
                LEDGER_ID,
                "--run-id",
                "barge-run",
                "--topology",
                "internal-loopback",
                "--route",
                "local",
                "--context",
                "caller-sim-barge",
                "--runtime-gate",
                "pass",
                "--output",
                str(output),
                "--history",
                str(history),
                "--min-semantic-turns",
                "1",
                "--require-barge-in",
            ]
            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(cli_args)

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["measurement_complete"])
            self.assertTrue(payload["smoke_gate"]["ok"])
            self.assertEqual(len(payload["semantic_turns"]), 1)
            self.assertEqual(len(payload["barge_ins"]), 1)

            clip.unlink()
            missing_output = root / "missing-voice-evidence.json"
            missing_args = [
                "missing-run" if item == "barge-run" else item for item in cli_args
            ]
            missing_args[missing_args.index(str(output))] = str(missing_output)
            missing_args[missing_args.index(str(history))] = str(
                root / "missing-history.jsonl"
            )
            with redirect_stdout(io.StringIO()):
                missing_exit_code = voice_evidence_main(missing_args)

            self.assertEqual(missing_exit_code, 1)
            missing_payload = json.loads(missing_output.read_text())
            self.assertFalse(missing_payload["valid"])
            self.assertEqual(
                missing_payload["invalid_reasons"], ["semantic_turn_clip_missing"]
            )

    def test_interrupted_generated_semantic_text_requires_clip_and_excludes_fallback(
        self,
    ) -> None:
        turn = {
            "user": "Question",
            "assistant": "Generated semantic answer",
            "assistant_spoken": "",
            "interrupted": True,
        }

        self.assertFalse(voice_evidence_module.has_measurable_semantic_text(turn))
        self.assertTrue(
            voice_evidence_module.has_measurable_semantic_text(turn, clip_backed=True)
        )
        self.assertFalse(
            voice_evidence_module.has_measurable_semantic_text(
                {**turn, "llm_fallback_leg": "recovery_utterance"},
                clip_backed=True,
            )
        )

    def test_semantic_alignment_skips_early_nonsemantic_energy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=4_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                heard,
                duration_ms=4_000,
                tones=[
                    (1_600, 1_800, 310.0),
                    *_semantic_pattern(2_500, frequency_hz=730.0),
                ],
            )
            _write_default_semantic_clip(root, frequency_hz=730.0)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual(
                evidence.semantic_alignment_source, SEMANTIC_ALIGNMENT_SOURCE
            )
            self.assertAlmostEqual(
                evidence.semantic_turns[0].semantic_voice_to_voice_ms,
                1_100.0,
                delta=20.0,
            )

    def test_semantic_alignment_recovers_a_ten_ms_vad_boundary_shift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                heard,
                duration_ms=3_000,
                tones=_semantic_pattern(1_710, frequency_hz=730.0),
            )
            _write_default_semantic_clip(root, frequency_hz=730.0)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            turn = evidence.semantic_turns[0]
            self.assertEqual(
                turn.agent_speech_start_sample, round(SAMPLE_RATE * 1_710 / 1_000)
            )
            self.assertAlmostEqual(turn.semantic_voice_to_voice_ms, 310.0, delta=0.001)

    def test_semantic_alignment_extracts_each_feature_series_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                heard,
                duration_ms=3_000,
                tones=_semantic_pattern(1_710, frequency_hz=730.0),
            )
            _write_default_semantic_clip(root, frequency_hz=730.0)

            with patch.object(
                voice_evidence_module,
                "_acoustic_features",
                wraps=voice_evidence_module._acoustic_features,
            ) as feature_extractor:
                evidence = analyze_split_tracks(
                    caller,
                    heard,
                    transcript=_measurement_transcript(
                        turns=[
                            {"user": "Question", "assistant_spoken": "Semantic answer"}
                        ]
                    ),
                    semantic_turn_clips={1: semantic_clip},
                )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual(feature_extractor.call_count, 2)

    def test_semantic_alignment_fails_closed_at_the_track_work_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)

            with patch.object(
                voice_evidence_module, "SEMANTIC_ALIGNMENT_MAX_TRACK_MS", 1_000
            ):
                evidence = analyze_split_tracks(
                    caller,
                    heard,
                    transcript=_measurement_transcript(
                        turns=[
                            {"user": "Question", "assistant_spoken": "Semantic answer"}
                        ]
                    ),
                    semantic_turn_clips={1: semantic_clip},
                )

            self.assertFalse(evidence.valid)
            self.assertIn(
                "semantic_turn_alignment_work_limit_exceeded", evidence.invalid_reasons
            )

    def test_semantic_alignment_fails_closed_at_the_candidate_work_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)

            with patch.object(
                voice_evidence_module,
                "SEMANTIC_ALIGNMENT_MAX_PRIMARY_CANDIDATES",
                1,
            ):
                evidence = analyze_split_tracks(
                    caller,
                    heard,
                    transcript=_measurement_transcript(
                        turns=[
                            {"user": "Question", "assistant_spoken": "Semantic answer"}
                        ]
                    ),
                    semantic_turn_clips={1: semantic_clip},
                )

            self.assertFalse(evidence.valid)
            self.assertIn(
                "semantic_turn_alignment_work_limit_exceeded", evidence.invalid_reasons
            )

    def test_semantic_alignment_fails_closed_at_the_correlation_work_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)

            with patch.object(
                voice_evidence_module,
                "SEMANTIC_ALIGNMENT_MAX_CORRELATION_FRAME_PAIRS",
                1,
            ):
                evidence = analyze_split_tracks(
                    caller,
                    heard,
                    transcript=_measurement_transcript(
                        turns=[
                            {"user": "Question", "assistant_spoken": "Semantic answer"}
                        ]
                    ),
                    semantic_turn_clips={1: semantic_clip},
                )

            self.assertFalse(evidence.valid)
            self.assertIn(
                "semantic_turn_alignment_work_limit_exceeded", evidence.invalid_reasons
            )

    def test_semantic_alignment_recovers_a_twenty_ms_vad_boundary_shift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                heard,
                duration_ms=3_000,
                tones=[
                    (1_700, 1_720, 310.0),
                    *_semantic_pattern(1_720, frequency_hz=730.0),
                ],
            )
            _write_default_semantic_clip(root, frequency_hz=730.0)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            turn = evidence.semantic_turns[0]
            self.assertEqual(
                turn.agent_speech_start_sample, round(SAMPLE_RATE * 1_720 / 1_000)
            )
            self.assertAlmostEqual(turn.semantic_voice_to_voice_ms, 320.0, delta=0.001)

    def test_semantic_alignment_uses_a_shorter_prefix_when_the_long_clip_diverges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            reference_frequencies = [
                280.0,
                430.0,
                610.0,
                350.0,
                760.0,
                520.0,
                890.0,
                300.0,
                680.0,
                470.0,
                820.0,
                390.0,
                720.0,
                560.0,
                940.0,
            ]
            matching_prefix_steps = 7
            _write_track(caller, duration_ms=4_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                semantic_clip,
                duration_ms=1_500,
                tones=[
                    (index * 100, (index + 1) * 100, frequency)
                    for index, frequency in enumerate(reference_frequencies)
                ],
            )
            heard_frequencies = [
                *reference_frequencies[:matching_prefix_steps],
                *([200.0] * (len(reference_frequencies) - matching_prefix_steps)),
            ]
            _write_track(
                heard,
                duration_ms=4_000,
                tones=[
                    (1_700 + index * 100, 1_700 + (index + 1) * 100, frequency)
                    for index, frequency in enumerate(heard_frequencies)
                ],
            )

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            turn = evidence.semantic_turns[0]
            self.assertGreaterEqual(turn.alignment_sample_count, SAMPLE_RATE // 2)
            self.assertLess(turn.alignment_sample_count, SAMPLE_RATE * 1_500 // 1_000)

    def test_semantic_alignment_rejects_an_ambiguous_replayed_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=4_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                heard,
                duration_ms=4_000,
                tones=[
                    *_semantic_pattern(1_700, frequency_hz=400.0),
                    *_semantic_pattern(2_700, frequency_hz=400.0),
                ],
            )
            _write_default_semantic_clip(root, frequency_hz=400.0)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertIn("semantic_turn_clip_ambiguous", evidence.invalid_reasons)

    def test_semantic_alignment_rejects_replay_inside_one_vad_span(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                semantic_clip,
                duration_ms=500,
                tones=[
                    (index * 50, (index + 1) * 50, 300.0 if index % 2 == 0 else 700.0)
                    for index in range(10)
                ],
            )
            _write_track(
                heard,
                duration_ms=3_000,
                tones=[
                    (
                        1_700 + index * 50,
                        1_700 + (index + 1) * 50,
                        300.0 if index % 2 == 0 else 700.0,
                    )
                    for index in range(12)
                ],
            )

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertIn("semantic_turn_clip_ambiguous", evidence.invalid_reasons)

    def test_semantic_alignment_rejects_later_replay_inside_one_vad_span(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            frequencies = [
                280.0,
                430.0,
                610.0,
                350.0,
                760.0,
                520.0,
                890.0,
                300.0,
                680.0,
                470.0,
            ]
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(
                semantic_clip,
                duration_ms=500,
                tones=[
                    (index * 50, (index + 1) * 50, frequency)
                    for index, frequency in enumerate(frequencies)
                ],
            )
            _write_track_with_clip_copies(
                heard,
                semantic_clip,
                duration_ms=3_000,
                start_times_ms=[1_710, 2_210],
            )

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertIn("semantic_turn_clip_ambiguous", evidence.invalid_reasons)

    def test_semantic_alignment_fails_closed_when_no_caller_side_span_matches_turn_clip(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            semantic_clip = root / "turn-001-agent.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400, 510.0)])
            _write_track(heard, duration_ms=3_000, tones=[(1_700, 2_200, 310.0)])
            _write_default_semantic_clip(root, frequency_hz=730.0)

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript=_measurement_transcript(
                    turns=[{"user": "Question", "assistant_spoken": "Semantic answer"}]
                ),
                semantic_turn_clips={1: semantic_clip},
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertIn(
                "semantic_turn_clip_low_correlation", evidence.invalid_reasons
            )

    def test_measures_when_barged_in_agent_becomes_inaudible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=5_000, tones=[(1_000, 1_400), (3_000, 3_400)]
            )
            _write_track(heard, duration_ms=5_000, tones=[(200, 500), (1_700, 3_180)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {
                            "user": "First question",
                            "assistant_spoken": "A long answer",
                            "interrupted": True,
                        },
                        {"user": "Please stop", "assistant_spoken": "Stopped"},
                    ]
                },
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual([turn.turn for turn in evidence.semantic_turns], [1])
            self.assertEqual(len(evidence.barge_ins), 1)
            self.assertEqual(evidence.barge_ins[0].turn, 2)
            self.assertAlmostEqual(
                evidence.barge_ins[0].audible_stop_ms, 180.0, delta=20.0
            )

    def test_barge_in_latency_includes_agent_speech_that_resumes_before_caller_stops(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=5_000, tones=[(3_000, 3_700)])
            _write_track(
                heard,
                duration_ms=5_000,
                tones=[(1_700, 3_060), (3_220, 3_600)],
            )

            evidence = analyze_split_tracks(caller, heard)

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.barge_ins, ())
            self.assertIn(
                "agent_resumed_before_caller_stopped", evidence.invalid_reasons
            )

    def test_natural_agent_end_during_caller_overlap_is_not_causal_barge_in(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=5_000, tones=[(1_000, 1_400), (3_000, 3_400)]
            )
            _write_track(heard, duration_ms=5_000, tones=[(1_700, 3_180)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {
                            "user": "First question",
                            "assistant_spoken": "Naturally ending answer",
                            "interrupted": False,
                        },
                        {
                            "user": "Speaking near the end",
                            "assistant_spoken": "Next reply",
                        },
                    ]
                },
            )

            self.assertEqual(evidence.barge_ins, ())

    def test_caller_starting_in_a_real_agent_pause_is_not_reported_as_barge_in(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=5_000, tones=[(1_000, 1_400), (3_000, 3_400)]
            )
            # The second caller starts in a 300 ms real pause between phrases.
            # Caller VAD needs a 600 ms merge window, but agent VAD must not
            # bridge this pause and pretend the agent was audibly interrupted.
            _write_track(
                heard, duration_ms=5_000, tones=[(1_700, 2_850), (3_150, 3_700)]
            )

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {
                            "user": "First question",
                            "assistant_spoken": "Two phrase answer",
                            "interrupted": True,
                        },
                        {"user": "I spoke in a pause", "assistant_spoken": "Reply"},
                    ]
                },
            )

            self.assertEqual(evidence.barge_ins, ())

    def test_missing_split_track_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 600)])

            evidence = analyze_split_tracks(caller, root / "missing-agent-heard.wav")

            self.assertFalse(evidence.valid)
            self.assertIn("agent_track_unreadable", evidence.invalid_reasons)

    def test_adapts_vad_threshold_to_continuous_low_level_line_noise(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=3_000, tones=[(1_000, 1_400)], noise_amplitude=500
            )
            _write_track(
                heard, duration_ms=3_000, tones=[(1_700, 2_200)], noise_amplitude=500
            )

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={"turns": [{"user": "Hello", "assistant_spoken": "Hi"}]},
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertAlmostEqual(
                evidence.semantic_turns[0].semantic_voice_to_voice_ms, 300.0, delta=20.0
            )
            self.assertEqual(evidence.measurement_source, UNVERIFIED_MEASUREMENT_SOURCE)

    def test_short_intra_sentence_pause_does_not_split_the_caller_turn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=4_000, tones=[(1_000, 1_300), (1_720, 2_040)]
            )
            _write_track(heard, duration_ms=4_000, tones=[(2_340, 2_940)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {"user": "A sentence with a pause", "assistant_spoken": "Reply"}
                    ]
                },
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual(len(evidence.semantic_turns), 1)
            self.assertEqual(
                evidence.semantic_turns[0].caller_speech_end_sample, 16_320
            )
            self.assertEqual(
                evidence.semantic_turns[0].agent_speech_start_sample, 18_720
            )
            self.assertAlmostEqual(
                evidence.semantic_turns[0].semantic_voice_to_voice_ms, 300.0, delta=20.0
            )

    def test_ambiguous_caller_audio_to_transcript_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=4_000, tones=[(1_000, 1_300), (2_100, 2_400)]
            )
            _write_track(heard, duration_ms=4_000, tones=[(2_700, 3_300)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {"user": "Only one mapped turn", "assistant_spoken": "Reply"}
                    ]
                },
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertEqual(evidence.barge_ins, ())
            self.assertIn(
                "caller_transcript_turn_count_mismatch", evidence.invalid_reasons
            )

    def test_recovery_utterance_is_not_counted_as_semantic_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=[(1_700, 2_200)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {
                            "user": "Hello",
                            "assistant_spoken": "Please repeat",
                            "llm_fallback_leg": "recovery_utterance",
                        }
                    ]
                },
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())

    def test_barge_in_without_confirmed_agent_silence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=4_000, tones=[(2_500, 2_900)])
            _write_track(heard, duration_ms=4_000, tones=[(1_000, 4_000)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [{"user": "Stop", "assistant_spoken": "A long answer"}]
                },
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.barge_ins, ())

    def test_barge_in_requires_agent_to_be_audibly_active_before_interruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=3_000, tones=[(2_040, 2_400)])
            _write_track(heard, duration_ms=3_000, tones=[(2_000, 2_600)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={"turns": [{"user": "Stop", "assistant_spoken": "Answer"}]},
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.barge_ins, ())

    def test_does_not_pair_a_response_that_begins_after_the_next_caller_turn(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=4_000, tones=[(1_000, 1_400), (2_200, 2_600)]
            )
            _write_track(heard, duration_ms=4_000, tones=[(2_900, 3_400)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {"user": "First", "assistant_spoken": "First reply"},
                        {"user": "Second", "assistant_spoken": "Second reply"},
                    ]
                },
            )

            self.assertTrue(evidence.valid, evidence.invalid_reasons)
            self.assertEqual([turn.turn for turn in evidence.semantic_turns], [2])
            self.assertAlmostEqual(
                evidence.semantic_turns[0].semantic_voice_to_voice_ms, 300.0, delta=20.0
            )

    def test_zero_latency_response_is_rejected_instead_of_reported_as_valid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=[(1_400, 2_000)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                transcript={
                    "turns": [
                        {"user": "Hello", "assistant_spoken": "Impossible overlap"}
                    ]
                },
            )

            self.assertFalse(evidence.valid)
            self.assertEqual(evidence.semantic_turns, ())
            self.assertIn("voice_evidence_not_found", evidence.invalid_reasons)

    def test_invalid_vad_configuration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 500)])
            _write_track(heard, duration_ms=1_000, tones=[(600, 900)])

            evidence = analyze_split_tracks(caller, heard, config=VadConfig(frame_ms=0))

            self.assertFalse(evidence.valid)
            self.assertIn("invalid_vad_config", evidence.invalid_reasons)

    def test_merge_gap_shorter_than_silence_confirmation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 500)])
            _write_track(heard, duration_ms=1_000, tones=[(600, 900)])

            evidence = analyze_split_tracks(
                caller,
                heard,
                config=VadConfig(silence_confirm_ms=80, merge_gap_ms=40),
            )

            self.assertFalse(evidence.valid)
            self.assertIn("invalid_vad_config", evidence.invalid_reasons)

    def test_non_finite_vad_threshold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 500)])
            _write_track(heard, duration_ms=1_000, tones=[(600, 900)])

            evidence = analyze_split_tracks(
                caller, heard, config=VadConfig(speech_rms=math.inf)
            )

            self.assertFalse(evidence.valid)
            self.assertIn("invalid_vad_config", evidence.invalid_reasons)

    def test_empty_split_tracks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=0, tones=[])
            _write_track(heard, duration_ms=0, tones=[])

            evidence = analyze_split_tracks(caller, heard)

            self.assertFalse(evidence.valid)
            self.assertIn("empty_split_tracks", evidence.invalid_reasons)

    def test_truncated_wav_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 500)])
            _write_track(heard, duration_ms=1_000, tones=[(600, 900)])
            caller.write_bytes(caller.read_bytes()[:-100])

            evidence = analyze_split_tracks(caller, heard)

            self.assertFalse(evidence.valid)
            self.assertIn("caller_track_unreadable", evidence.invalid_reasons)

    def test_split_tracks_with_different_sample_rates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(
                caller, duration_ms=1_000, tones=[(200, 500)], sample_rate=8_000
            )
            _write_track(
                heard, duration_ms=1_000, tones=[(600, 900)], sample_rate=16_000
            )

            evidence = analyze_split_tracks(caller, heard)

            self.assertFalse(evidence.valid)
            self.assertIn("split_track_sample_rate_mismatch", evidence.invalid_reasons)

    def test_split_tracks_with_different_durations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=1_000, tones=[(200, 500)])
            _write_track(heard, duration_ms=1_200, tones=[(600, 900)])

            evidence = analyze_split_tracks(caller, heard)

            self.assertFalse(evidence.valid)
            self.assertIn("split_track_duration_mismatch", evidence.invalid_reasons)

    def test_history_gate_uses_only_positive_samples_from_the_requested_cohort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            history = Path(td) / "history.jsonl"
            for index in range(30):
                append_history(
                    history,
                    _history_record(
                        f"run-{index}",
                        semantic_ms=1_000.0 if index < 27 else 1_300.0,
                        barge_ms=200.0 if index < 10 else None,
                    ),
                )
            append_history(
                history,
                _history_record("other-run", cohort="other-stack", semantic_ms=9_999.0),
            )
            records = [json.loads(line) for line in history.read_text().splitlines()]

            gate = evaluate_history(
                records,
                cohort="flash-v1",
                topology="pstn-loopback",
                route="didww-outbound-us",
            )

            self.assertTrue(gate["ok"], gate)
            self.assertEqual(gate["semantic"]["sample_count"], 30)
            self.assertEqual(gate["semantic"]["p90_ms"], 1_030.0)
            self.assertEqual(gate["barge_in"]["sample_count"], 10)
            self.assertEqual(gate["barge_in"]["p90_ms"], 200.0)

    def test_history_gate_requires_hash_bound_attempt_for_every_history_record(
        self,
    ) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        events = _integrity_pairs_for_history(records)

        missing_journal = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[],
        )
        deleted_history = evaluate_history(
            records[1:],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=events,
        )
        injected_history = evaluate_history(
            [*records, _history_record("injected-run")],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=events,
        )
        edited_history = evaluate_history(
            [
                _history_record("run-0", semantic_ms=1_100.0, barge_ms=200.0),
                *records[1:],
            ],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=events,
        )

        self.assertFalse(missing_journal["ok"], missing_journal)
        self.assertEqual(missing_journal["history_without_durable_attempt"], 30)
        self.assertFalse(deleted_history["ok"], deleted_history)
        self.assertEqual(deleted_history["durable_attempts_without_history"], 1)
        self.assertFalse(injected_history["ok"], injected_history)
        self.assertEqual(injected_history["history_without_durable_attempt"], 1)
        self.assertFalse(edited_history["ok"], edited_history)
        self.assertEqual(edited_history["evidence_record_hash_mismatches"], 1)

    def test_history_gate_rejects_repeated_metric_items_in_one_record(self) -> None:
        record = _history_record("inflated-run", barge_ms=200.0)
        record["semantic_turns"] = record["semantic_turns"] * 30
        record["barge_ins"] = record["barge_ins"] * 10

        gate = evaluate_history(
            [record],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["rejected_records"], 1)
        self.assertEqual(gate["semantic"]["sample_count"], 0)
        self.assertEqual(gate["barge_in"]["sample_count"], 0)

    def test_history_gate_rejects_run_and_source_reuse_across_campaigns(self) -> None:
        current = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        prior = _history_record(
            "run-0",
            barge_ms=200.0,
            campaign_id="sha256:" + "d" * 64,
        )

        gate = evaluate_history(
            [prior, *current],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=_integrity_pairs_for_history(current),
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["duplicate_run_ids"], 1)
        self.assertEqual(gate["conflicting_duplicate_run_ids"], 1)
        self.assertEqual(gate["duplicate_source_evidence"], 1)

    def test_old_incomplete_missing_artifacts_do_not_poison_a_new_campaign(
        self,
    ) -> None:
        current = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        unreadable_fingerprint = "sha256:" + "0" * 64
        prior_incomplete = []
        for index, campaign_character in enumerate(("d", "f")):
            record = _history_record(
                f"prior-incomplete-{index}",
                valid=False,
                measurement_complete=False,
                campaign_id="sha256:" + campaign_character * 64,
                source_evidence_fingerprint=unreadable_fingerprint,
            )
            record["semantic_turns"] = []
            record["barge_ins"] = []
            prior_incomplete.append(record)

        gate = evaluate_history(
            [*prior_incomplete, *current],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=_integrity_pairs_for_history(
                [*prior_incomplete, *current]
            ),
        )

        self.assertTrue(gate["ok"], gate)
        self.assertEqual(gate["duplicate_source_evidence"], 0)
        self.assertEqual(gate["invalid_or_incomplete_runs"], 0)

    def test_old_incomplete_unverified_alignment_does_not_poison_a_new_campaign(
        self,
    ) -> None:
        current = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        prior_incomplete = _history_record(
            "prior-transcript-mismatch",
            valid=False,
            measurement_complete=False,
            campaign_id="sha256:" + "d" * 64,
            topology="asterisk-duplex",
            route="local",
        )
        prior_incomplete["semantic_alignment_source"] = UNVERIFIED_ALIGNMENT_SOURCE
        prior_incomplete["semantic_turns"] = []
        prior_incomplete["barge_ins"] = []

        gate = evaluate_history(
            [prior_incomplete, *current],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertTrue(gate["ok"], gate)
        self.assertEqual(gate["rejected_records"], 0)
        self.assertEqual(gate["invalid_or_incomplete_runs"], 0)

        current_incomplete = _history_record(
            "current-transcript-mismatch",
            valid=False,
            measurement_complete=False,
        )
        current_incomplete["semantic_alignment_source"] = UNVERIFIED_ALIGNMENT_SOURCE
        current_incomplete["semantic_turns"] = []
        current_incomplete["barge_ins"] = []

        current_gate = evaluate_history(
            [*current, current_incomplete],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(current_gate["ok"], current_gate)
        self.assertEqual(current_gate["rejected_records"], 0)
        self.assertEqual(current_gate["invalid_or_incomplete_runs"], 1)

    def test_ledger_wide_hash_binding_rejects_deleted_edited_or_injected_old_rows(
        self,
    ) -> None:
        current = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        prior_campaign = "sha256:" + "d" * 64
        replayed_source = "sha256:" + "9" * 64
        prior = _history_record(
            "prior-run",
            campaign_id=prior_campaign,
            source_evidence_fingerprint=replayed_source,
        )
        current_with_replay = [
            _history_record(
                "run-0",
                barge_ms=200.0,
                source_evidence_fingerprint=replayed_source,
            ),
            *current[1:],
        ]
        retained_events = _integrity_pairs_for_history([prior, *current_with_replay])
        deleted = evaluate_history(
            current_with_replay,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=retained_events,
        )

        original_prior = _history_record("prior-edit", campaign_id=prior_campaign)
        edited_prior = _history_record(
            "prior-edit",
            campaign_id=prior_campaign,
            semantic_ms=1_100.0,
        )
        edited = evaluate_history(
            [edited_prior, *current],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=_integrity_pairs_for_history([original_prior, *current]),
        )

        injected_prior = _history_record("prior-injected", campaign_id=prior_campaign)
        injected = evaluate_history(
            [injected_prior, *current],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=_integrity_pairs_for_history(current),
        )

        self.assertFalse(deleted["ok"], deleted)
        self.assertEqual(deleted["duplicate_source_evidence"], 0)
        self.assertEqual(deleted["ledger_durable_attempts_without_history"], 1)
        self.assertFalse(edited["ok"], edited)
        self.assertEqual(edited["ledger_evidence_record_hash_mismatches"], 1)
        self.assertFalse(injected["ok"], injected)
        self.assertEqual(injected["ledger_history_without_durable_attempt"], 1)

    def test_history_gate_fails_on_tampered_records_and_conflicting_duplicate_run_ids(
        self,
    ) -> None:
        records = [
            _history_record(
                f"run-{index}",
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]
        records.append(_history_record("run-0", semantic_ms=9_999.0, barge_ms=9_999.0))
        records.append(
            _history_record(
                "foreign-run",
                semantic_ms=1.0,
                barge_ms=1.0,
                source="forged_worker_timestamp",
            )
        )
        tampered_schema = _history_record(
            "wrong-schema-run", semantic_ms=1.0, barge_ms=1.0
        )
        tampered_schema["schema"] = "foreign.voice_evidence.v1"
        records.append(tampered_schema)
        missing_context = _history_record(
            "missing-context-run", semantic_ms=1.0, barge_ms=1.0
        )
        missing_context["context"] = ""
        records.append(missing_context)
        missing_cohort = _history_record(
            "missing-cohort-run", semantic_ms=1.0, barge_ms=1.0
        )
        missing_cohort.pop("cohort")
        records.append(missing_cohort)

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["matching_records"], 29)
        self.assertEqual(gate["duplicate_run_ids"], 1)
        self.assertEqual(gate["conflicting_duplicate_run_ids"], 1)
        self.assertEqual(gate["rejected_records"], 4)
        self.assertEqual(gate["semantic"]["sample_count"], 29)
        self.assertEqual(gate["barge_in"]["sample_count"], 9)

    def test_history_gate_rejects_unroutable_objects_in_both_ledgers(self) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]

        history_gate = evaluate_history(
            [*records, {}],
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )
        integrity_gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[{}],
        )

        self.assertFalse(history_gate["ok"], history_gate)
        self.assertEqual(history_gate["rejected_records"], 1)
        self.assertFalse(integrity_gate["ok"], integrity_gate)
        self.assertEqual(integrity_gate["rejected_integrity_events"], 1)

    def test_history_gate_rejects_an_identical_duplicate_without_inflating_samples(
        self,
    ) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        records.append(dict(records[0]))

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["duplicate_run_ids"], 1)
        self.assertEqual(gate["conflicting_duplicate_run_ids"], 0)
        self.assertEqual(gate["semantic"]["sample_count"], 30)

    def test_history_gate_rejects_replayed_source_call_with_fresh_run_ids(self) -> None:
        replayed_source = "sha256:" + "9" * 64
        records = [
            _history_record(
                f"run-{index}",
                source_evidence_fingerprint=(
                    replayed_source if index in {0, 1} else None
                ),
                barge_ms=200.0 if index < 12 else None,
            )
            for index in range(32)
        ]

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["duplicate_source_evidence"], 1)
        self.assertEqual(gate["semantic"]["sample_count"], 30)
        self.assertEqual(gate["barge_in"]["sample_count"], 10)

    def test_history_gate_can_require_pstn_topology(self) -> None:
        internal_records = [
            _history_record(
                f"internal-{index}",
                topology="internal-loopback",
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]

        gate = evaluate_history(
            internal_records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["topology"], "pstn-loopback")
        self.assertEqual(gate["matching_records"], 0)

    def test_history_gate_library_requires_explicit_topology_and_route(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_history(
                [], cohort="flash-v1", topology="", route="didww-outbound-us"
            )
        with self.assertRaises(ValueError):
            evaluate_history([], cohort="flash-v1", topology="pstn-loopback", route="")

    def test_history_gate_does_not_pool_different_pstn_routes(self) -> None:
        ee_records = [
            _history_record(
                f"ee-{index}",
                route="didww-outbound-ee",
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]

        gate = evaluate_history(
            ee_records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["route"], "didww-outbound-us")
        self.assertEqual(gate["matching_records"], 0)

    def test_history_gate_fails_on_matching_or_unknown_integrity_failures(self) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        matching = _integrity_event("failed-run", reason="evidence_not_persisted")
        unknown = _integrity_event(
            "unknown-run",
            cohort="unknown",
            reason="evidence_not_persisted",
        )

        matching_gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[matching],
        )
        unknown_gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[unknown],
        )

        self.assertFalse(matching_gate["ok"], matching_gate)
        self.assertEqual(matching_gate["integrity_failures"], 1)
        self.assertFalse(unknown_gate["ok"], unknown_gate)
        self.assertEqual(unknown_gate["integrity_failures"], 1)

    def test_history_gate_rejects_integrity_records_from_a_different_ledger(
        self,
    ) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        mismatched = _integrity_event(
            "foreign-ledger-run",
            ledger_id="sha256:" + "f" * 64,
        )

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[mismatched],
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["rejected_integrity_events"], 1)

    def test_history_gate_requires_every_durable_attempt_to_be_finalized(self) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        complete_events = _integrity_pairs_for_history(records)
        pending_events = [
            event
            for event in complete_events
            if not (
                event["run_id"] == "run-0" and event["event"] == "attempt_finalized"
            )
        ]

        pending_gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=pending_events,
        )
        complete_gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=complete_events,
        )

        self.assertFalse(pending_gate["ok"], pending_gate)
        self.assertEqual(pending_gate["unfinalized_attempts"], 1)
        self.assertTrue(complete_gate["ok"], complete_gate)
        self.assertEqual(complete_gate["unfinalized_attempts"], 0)

    def test_history_gate_fails_when_any_well_formed_run_is_incomplete(self) -> None:
        records = [
            _history_record(
                f"run-{index}",
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]
        incomplete = _history_record(
            "unreadable-run",
            valid=False,
            measurement_complete=False,
        )
        incomplete["semantic_turns"] = []
        incomplete["barge_ins"] = []
        records.append(incomplete)

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"])
        self.assertEqual(gate["matching_records"], 31)
        self.assertEqual(gate["invalid_or_incomplete_runs"], 1)
        self.assertEqual(gate["semantic"]["sample_count"], 30)
        self.assertEqual(gate["barge_in"]["sample_count"], 10)

    def test_history_gate_fails_on_invalid_metrics_or_runtime_invariants(self) -> None:
        records = [
            _history_record(f"run-{index}", barge_ms=200.0 if index < 10 else None)
            for index in range(30)
        ]
        zero_metric = _history_record("zero-metric")
        zero_metric["semantic_turns"] = [{"semantic_voice_to_voice_ms": 0.0}]
        records.append(zero_metric)
        missing_metric = _history_record("missing-metric")
        missing_metric["semantic_turns"] = [{}]
        records.append(missing_metric)
        inconsistent_runtime = _history_record("runtime-failed")
        inconsistent_runtime["runtime_gate"] = {"ok": False, "status": "fail"}
        records.append(inconsistent_runtime)
        missing_semantic_policy = _history_record("missing-semantic-policy")
        missing_semantic_policy.pop("semantic_response_audio_policy")
        records.append(missing_semantic_policy)
        # A bundle from a call with acknowledgements enabled is VALID evidence:
        # the filler plays on a background track the sidecar never records, so
        # the measured agent track still opens with the semantic response and
        # the policy statement merely records the flag truthfully.
        filler_enabled = _history_record("filler-enabled")
        filler_enabled["semantic_response_audio_policy"] = {
            **SEMANTIC_RESPONSE_AUDIO_POLICY,
            "filler_acknowledgements_enabled": True,
        }
        records.append(filler_enabled)
        malformed_flag = _history_record("malformed-flag")
        malformed_flag["semantic_response_audio_policy"] = {
            **SEMANTIC_RESPONSE_AUDIO_POLICY,
            "filler_acknowledgements_enabled": "true",
        }
        records.append(malformed_flag)

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["rejected_records"], 5)
        # The filler-enabled record contributes its samples like any other.
        self.assertEqual(gate["semantic"]["sample_count"], 31)

    def test_history_gate_selects_derived_agent_cohort_and_consumes_base_integrity_failures(
        self,
    ) -> None:
        derived_cohort = "sha256:" + "d" * 64
        base_cohort = "sha256:" + "b" * 64
        records = [
            _history_record(
                f"agent-run-{index}",
                cohort=derived_cohort,
                base_cohort=base_cohort,
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]

        clean_gate = evaluate_history(
            records,
            cohort=derived_cohort,
            base_cohort=base_cohort,
            topology="pstn-loopback",
            route="didww-outbound-us",
        )
        base_failure = _integrity_event("failed-agent-run", cohort=base_cohort)
        failed_gate = evaluate_history(
            records,
            cohort=derived_cohort,
            base_cohort=base_cohort,
            topology="pstn-loopback",
            route="didww-outbound-us",
            integrity_events=[base_failure],
        )
        base_incomplete = _history_record(
            "base-unattributed-incomplete",
            cohort=base_cohort,
            base_cohort=base_cohort,
            valid=False,
            measurement_complete=False,
        )
        base_incomplete["semantic_turns"] = []
        base_incomplete["barge_ins"] = []
        base_history_gate = evaluate_history(
            [*records, base_incomplete],
            cohort=derived_cohort,
            base_cohort=base_cohort,
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertTrue(clean_gate["ok"], clean_gate)
        self.assertEqual(clean_gate["cohort"], derived_cohort)
        self.assertEqual(clean_gate["base_cohort"], base_cohort)
        self.assertFalse(failed_gate["ok"], failed_gate)
        self.assertEqual(failed_gate["integrity_failures"], 1)
        self.assertFalse(base_history_gate["ok"], base_history_gate)
        self.assertEqual(base_history_gate["matching_records"], 30)
        self.assertEqual(base_history_gate["base_incomplete_runs"], 1)
        self.assertEqual(base_history_gate["invalid_or_incomplete_runs"], 1)

    def test_history_gate_enforces_semantic_p95_limit(self) -> None:
        records = [
            _history_record(
                f"run-{index}",
                semantic_ms=1_000.0 if index < 94 else 2_000.0,
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(100)
        ]

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"])
        self.assertEqual(gate["semantic"]["p90_ms"], 1_000.0)
        self.assertEqual(gate["semantic"]["p95_ms"], 2_000.0)
        self.assertFalse(gate["semantic"]["p95_within_limit"])

    def test_history_gate_excludes_stale_measurements_from_the_release_distribution(
        self,
    ) -> None:
        now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        records = [
            _history_record(
                f"run-{index}",
                recorded_at=(now - timedelta(hours=1)).isoformat(),
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]
        records[0]["recorded_at"] = (now - timedelta(hours=169)).isoformat()

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            now=now,
            max_age_seconds=168 * 60 * 60,
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["stale_records"], 1)
        self.assertEqual(gate["semantic"]["sample_count"], 29)
        self.assertEqual(gate["freshness"]["max_age_hours"], 168.0)

    def test_history_gate_never_pools_successes_from_a_prior_container_campaign(
        self,
    ) -> None:
        prior_campaign = "sha256:" + "d" * 64
        records = [
            _history_record(
                f"prior-run-{index}",
                campaign_id=prior_campaign,
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["matching_records"], 0)
        self.assertEqual(gate["semantic"]["sample_count"], 0)
        self.assertEqual(gate["barge_in"]["sample_count"], 0)

    def test_history_gate_never_ages_out_an_incomplete_current_campaign_run(
        self,
    ) -> None:
        now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        records = [
            _history_record(
                f"run-{index}",
                recorded_at=(now - timedelta(hours=1)).isoformat(),
                barge_ms=200.0 if index < 10 else None,
            )
            for index in range(30)
        ]
        incomplete = _history_record(
            "old-incomplete",
            recorded_at=(now - timedelta(days=30)).isoformat(),
            valid=False,
            measurement_complete=False,
        )
        incomplete["semantic_turns"] = []
        incomplete["barge_ins"] = []
        records.append(incomplete)

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
            now=now,
        )

        self.assertFalse(gate["ok"], gate)
        self.assertEqual(gate["invalid_or_incomplete_runs"], 1)
        self.assertEqual(gate["stale_records"], 0)

    def test_history_gate_enforces_barge_hard_maximum(self) -> None:
        records = [
            _history_record(
                f"run-{index}",
                barge_ms=(500.0 if index == 9 else 200.0) if index < 10 else None,
            )
            for index in range(30)
        ]

        gate = evaluate_history(
            records,
            cohort="flash-v1",
            topology="pstn-loopback",
            route="didww-outbound-us",
        )

        self.assertFalse(gate["ok"])
        self.assertEqual(gate["barge_in"]["p90_ms"], 230.0)
        self.assertEqual(gate["barge_in"]["max_ms"], 500.0)
        self.assertFalse(gate["barge_in"]["max_within_limit"])

    def test_analyze_cli_writes_evidence_and_appends_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            transcript.write_text(
                json.dumps(
                    _measurement_transcript(
                        turns=[{"user": "Hello", "assistant_spoken": "Hi"}]
                    )
                )
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "run-123",
                        "--topology",
                        "asterisk-duplex",
                        "--route",
                        "local",
                        "--context",
                        "selftest",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                        "--max-semantic-ms",
                        "1500",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["smoke_gate"]["ok"])
            self.assertEqual(payload["cohort"], "flash-v1")
            self.assertEqual(payload["schema"], HISTORY_SCHEMA)
            self.assertEqual(payload["source"], MEASUREMENT_SOURCE)
            self.assertEqual(payload["run_id"], "run-123")
            self.assertEqual(payload["topology"], "asterisk-duplex")
            self.assertEqual(payload["context"], "selftest")
            self.assertTrue(payload["measurement_complete"])
            self.assertEqual(
                payload["semantic_response_audio_policy"],
                SEMANTIC_RESPONSE_AUDIO_POLICY,
            )
            self.assertEqual(
                payload["semantic_alignment_source"], SEMANTIC_ALIGNMENT_SOURCE
            )
            self.assertRegex(
                payload["source_evidence_fingerprint"], r"^sha256:[0-9a-f]{64}$"
            )
            self.assertIsNotNone(datetime.fromisoformat(payload["recorded_at"]))
            self.assertEqual(len(history.read_text().splitlines()), 1)

    def test_analyze_cli_hashes_and_measures_the_same_immutable_source_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            clip = _write_default_semantic_clip(root)
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            transcript.write_text(
                json.dumps(
                    _measurement_transcript(
                        turns=[{"user": "Hello", "assistant_spoken": "Hi"}]
                    )
                )
            )
            original_read_bytes = Path.read_bytes
            original_heard = original_read_bytes(heard)
            transcript_bytes = original_read_bytes(transcript)
            clip_bytes = original_read_bytes(clip)
            mutated = False

            def read_once_then_mutate(path: Path) -> bytes:
                nonlocal mutated
                payload = original_read_bytes(path)
                if path == heard and not mutated:
                    mutated = True
                    _write_track(
                        heard, duration_ms=3_000, tones=_semantic_pattern(2_200)
                    )
                return payload

            with (
                patch.object(Path, "read_bytes", read_once_then_mutate),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "immutable-run",
                        "--topology",
                        "asterisk-duplex",
                        "--route",
                        "local",
                        "--context",
                        "selftest",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            payload = json.loads(output.read_text())
            expected_fingerprint = _source_evidence_fingerprint(
                [
                    ("caller_spoken", original_read_bytes(caller)),
                    ("agent_heard", original_heard),
                    ("transcript", transcript_bytes),
                    ("semantic_turn_001", clip_bytes),
                ]
            )
            self.assertTrue(mutated)
            self.assertEqual(exit_code, 0)
            self.assertAlmostEqual(
                payload["semantic_turns"][0]["semantic_voice_to_voice_ms"],
                300.0,
                delta=20.0,
            )
            self.assertEqual(
                payload["source_evidence_fingerprint"], expected_fingerprint
            )

    def test_analyze_cli_rejects_missing_or_malformed_semantic_audio_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            _write_track(caller, duration_ms=4_000, tones=[(1_000, 1_400)])
            # The 200 ms span models a filler before the real answer at 2.5 s.
            _write_track(
                heard,
                duration_ms=4_000,
                tones=[(1_600, 1_800, 310.0), *_semantic_pattern(2_500)],
            )
            _write_default_semantic_clip(root)
            # A filler-enabled policy is a valid statement (the filler plays on
            # an unrecorded track); only a missing policy or a malformed flag is
            # rejected.
            invalid_payloads = {
                "missing": {
                    "turns": [{"user": "Hello", "assistant_spoken": "The answer"}]
                },
                "malformed-flag": _measurement_transcript(
                    turns=[{"user": "Hello", "assistant_spoken": "The answer"}],
                    semantic_response_audio_policy={
                        **SEMANTIC_RESPONSE_AUDIO_POLICY,
                        "filler_acknowledgements_enabled": "true",
                    },
                ),
            }

            for name, transcript_payload in invalid_payloads.items():
                with self.subTest(policy=name):
                    transcript = root / f"{name}-transcript.json"
                    output = root / f"{name}-evidence.json"
                    history = root / f"{name}-history.jsonl"
                    transcript.write_text(json.dumps(transcript_payload))
                    with redirect_stdout(io.StringIO()):
                        exit_code = voice_evidence_main(
                            [
                                "analyze",
                                "--caller-spoken",
                                str(caller),
                                "--agent-heard",
                                str(heard),
                                "--transcript",
                                str(transcript),
                                "--cohort",
                                "flash-v1",
                                "--campaign",
                                CAMPAIGN_ID,
                                "--ledger-id",
                                LEDGER_ID,
                                "--run-id",
                                f"policy-{name}",
                                "--topology",
                                "pstn-loopback",
                                "--route",
                                "didww-outbound-us",
                                "--context",
                                "caller-sim",
                                "--runtime-gate",
                                "pass",
                                "--output",
                                str(output),
                                "--history",
                                str(history),
                            ]
                        )

                    payload = json.loads(output.read_text())
                    self.assertEqual(exit_code, 1)
                    self.assertFalse(payload["valid"])
                    self.assertEqual(payload["semantic_turns"], [])
                    self.assertIn(
                        "semantic_response_audio_policy_invalid",
                        payload["invalid_reasons"],
                    )

    def test_analyze_cli_requires_run_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            transcript.write_text(
                json.dumps(
                    _measurement_transcript(
                        turns=[{"user": "Hello", "assistant_spoken": "Hi"}]
                    )
                )
            )

            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "flash-v1",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())
            self.assertFalse(history.exists())

    def test_analyze_cli_persists_explicit_failure_when_transcript_is_unreadable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=[(1_700, 2_200)])

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(root / "missing-transcript.json"),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "run-unreadable-transcript",
                        "--topology",
                        "asterisk-duplex",
                        "--route",
                        "local",
                        "--context",
                        "selftest",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.read_text())
            self.assertFalse(payload["valid"])
            self.assertFalse(payload["measurement_complete"])
            self.assertIn("transcript_unreadable", payload["invalid_reasons"])
            persisted = json.loads(history.read_text().strip())
            self.assertEqual(persisted["invalid_reasons"], payload["invalid_reasons"])

    def test_analyze_cli_enforces_minimum_semantic_turn_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            transcript.write_text(
                json.dumps(
                    _measurement_transcript(
                        turns=[{"user": "Hello", "assistant_spoken": "Hi"}]
                    )
                )
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "run-min-turns",
                        "--topology",
                        "asterisk-duplex",
                        "--route",
                        "local",
                        "--context",
                        "selftest",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                        "--min-semantic-turns",
                        "2",
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.read_text())
            self.assertFalse(payload["measurement_complete"])
            semantic_gate = payload["smoke_gate"]["semantic"]
            self.assertEqual(semantic_gate["sample_count"], 1)
            self.assertEqual(semantic_gate["min_samples"], 2)

    def test_analyze_cli_marks_acoustics_incomplete_when_runtime_gate_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            transcript.write_text(
                json.dumps(
                    _measurement_transcript(
                        turns=[{"user": "Hello", "assistant_spoken": "Hi"}]
                    )
                )
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "runtime-failed-run",
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--context",
                        "caller-sim",
                        "--runtime-gate",
                        "fail",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["valid"])
            self.assertFalse(payload["runtime_gate"]["ok"])
            self.assertFalse(payload["measurement_complete"])

    def test_analyze_cli_separates_effective_per_agent_stacks_from_base_cohort(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            stack = _effective_stack_identity(
                {"providers": {"stt": "deepgram", "llm": "gemini", "tts": "elevenlabs"}}
            )
            transcript.write_text(
                json.dumps(
                    {
                        "skysay": {"agent_id": "agent-one"},
                        "effective_voice_stack": stack,
                        "semantic_response_audio_policy": dict(
                            SEMANTIC_RESPONSE_AUDIO_POLICY
                        ),
                        "turns": [
                            {
                                "user": "Hello",
                                "assistant_spoken": "Hi",
                                "agent_audio": "turn-001-agent.wav",
                            }
                        ],
                    }
                )
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "base-stack",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "agent-stack-run",
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--context",
                        "caller-sim",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["base_cohort"], "base-stack")
            self.assertNotEqual(payload["cohort"], "base-stack")
            self.assertEqual(
                payload["effective_voice_stack_fingerprint"], stack["fingerprint"]
            )

    def test_analyze_cli_rejects_per_agent_transcript_without_effective_stack_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            caller = root / "caller-spoken.wav"
            heard = root / "agent-heard.wav"
            transcript = root / "transcript.json"
            output = root / "voice-evidence.json"
            history = root / "history.jsonl"
            _write_track(caller, duration_ms=3_000, tones=[(1_000, 1_400)])
            _write_track(heard, duration_ms=3_000, tones=_semantic_pattern(1_700))
            _write_default_semantic_clip(root)
            transcript.write_text(
                json.dumps(
                    {
                        "skysay": {"agent_id": "agent-one"},
                        "semantic_response_audio_policy": dict(
                            SEMANTIC_RESPONSE_AUDIO_POLICY
                        ),
                        "turns": [
                            {
                                "user": "Hello",
                                "assistant_spoken": "Hi",
                                "agent_audio": "turn-001-agent.wav",
                            }
                        ],
                    }
                )
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "analyze",
                        "--caller-spoken",
                        str(caller),
                        "--agent-heard",
                        str(heard),
                        "--transcript",
                        str(transcript),
                        "--cohort",
                        "base-stack",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--run-id",
                        "agent-stack-missing",
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--context",
                        "caller-sim",
                        "--runtime-gate",
                        "pass",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ]
                )

            self.assertEqual(exit_code, 1)
            payload = json.loads(output.read_text())
            self.assertFalse(payload["valid"])
            self.assertIn(
                "effective_voice_stack_identity_missing", payload["invalid_reasons"]
            )

    def test_cohort_cli_hashes_only_allowlisted_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "cohort.txt"
            base_env = {
                "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
                "REALTIME_TTS_RTP_PREBUFFER_MS": "240",
                "OPENAI_API_KEY": "first-secret",
                "CARTESIA_API_KEY": "first-cartesia-secret",
                "REALTIME_LLM_API_KEY": "first-compatible-secret",
                "SKYSAY_GATEWAY_API_TOKEN": "first-token",
                "ASTERISK_ARI_PASSWORD": "first-password",
            }
            with (
                patch.dict("os.environ", base_env, clear=True),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                first_exit = voice_evidence_main(
                    [
                        "cohort",
                        "--image-identity",
                        "gateway-image-123",
                        "--output",
                        str(output),
                    ]
                )
            changed_secrets = {
                **base_env,
                "OPENAI_API_KEY": "second-secret",
                "CARTESIA_API_KEY": "second-cartesia-secret",
                "REALTIME_LLM_API_KEY": "second-compatible-secret",
                "SKYSAY_GATEWAY_API_TOKEN": "second-token",
                "ASTERISK_ARI_PASSWORD": "second-password",
            }
            with (
                patch.dict("os.environ", changed_secrets, clear=True),
                redirect_stdout(io.StringIO()) as second_stdout,
            ):
                second_exit = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )
            changed_latency_config = {
                **changed_secrets,
                "DEEPGRAM_ENDPOINTING_MS": "250",
            }
            with (
                patch.dict("os.environ", changed_latency_config, clear=True),
                redirect_stdout(io.StringIO()) as third_stdout,
            ):
                third_exit = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )
            changed_effective_models = {
                **changed_latency_config,
                "REALTIME_LLM_MODEL": "different-llm",
                "REALTIME_LLM_SYSTEM_PROMPT": "different prompt",
                "REALTIME_TTS_MODEL": "different-tts",
                "REALTIME_TTS_SAMPLE_RATE": "24000",
                "REALTIME_RECOVERY_UTTERANCE": "Please repeat that.",
            }
            with (
                patch.dict("os.environ", changed_effective_models, clear=True),
                redirect_stdout(io.StringIO()) as fourth_stdout,
            ):
                fourth_exit = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )
            changed_alternative_provider = {
                **changed_effective_models,
                "CARTESIA_MODEL": "sonic-3",
                "CARTESIA_VOICE_ID": "public-voice-id",
                "ANTHROPIC_MODEL": "claude-new",
                "OPENAI_BASE_URL": "https://llm.internal/v1",
            }
            with (
                patch.dict("os.environ", changed_alternative_provider, clear=True),
                redirect_stdout(io.StringIO()) as fifth_stdout,
            ):
                fifth_exit = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )

            first = stdout.getvalue().strip()
            second = second_stdout.getvalue().strip()
            third = third_stdout.getvalue().strip()
            fourth = fourth_stdout.getvalue().strip()
            fifth = fifth_stdout.getvalue().strip()
            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)
            self.assertEqual(third_exit, 0)
            self.assertEqual(fourth_exit, 0)
            self.assertEqual(fifth_exit, 0)
            self.assertEqual(first, second)
            self.assertNotEqual(second, third)
            self.assertNotEqual(third, fourth)
            self.assertNotEqual(fourth, fifth)
            self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(output.read_text().strip(), first)
            combined_output = first + output.read_text()
            self.assertNotIn("eleven_flash_v2_5", combined_output)
            self.assertNotIn("secret", combined_output)
            self.assertNotIn("token", combined_output)
            self.assertNotIn("password", combined_output)

    def test_cohort_cli_separates_elevenlabs_model_and_language_but_not_credentials(
        self,
    ) -> None:
        def cohort(environment: dict[str, str]) -> str:
            with (
                patch.dict("os.environ", environment, clear=True),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )
            self.assertEqual(exit_code, 0)
            return stdout.getvalue().strip()

        english_flash = cohort(
            {
                "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
                "ELEVENLABS_LANGUAGE_CODE": "en",
                "ELEVENLABS_API_KEY": "first-credential-value",
            }
        )
        rotated_credential = cohort(
            {
                "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
                "ELEVENLABS_LANGUAGE_CODE": "en",
                "ELEVENLABS_API_KEY": "rotated-credential-value",
            }
        )
        english_v3 = cohort(
            {
                "ELEVENLABS_MODEL_ID": "eleven_v3",
                "ELEVENLABS_LANGUAGE_CODE": "en",
                "ELEVENLABS_API_KEY": "rotated-credential-value",
            }
        )
        french_flash = cohort(
            {
                "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
                "ELEVENLABS_LANGUAGE_CODE": "fr",
                "ELEVENLABS_API_KEY": "rotated-credential-value",
            }
        )

        self.assertEqual(english_flash, rotated_credential)
        self.assertNotEqual(english_flash, english_v3)
        self.assertNotEqual(english_flash, french_flash)
        self.assertRegex(english_flash, r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("credential", english_flash)

    def test_cohort_cli_hashes_supported_provider_tuning_without_hashing_secrets(
        self,
    ) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            base_exit = voice_evidence_main(
                ["cohort", "--image-identity", "gateway-image-123"]
            )
        base = stdout.getvalue().strip()

        for variable in (
            "REALTIME_LLM_MAX_TOKENS",
            "OPENAI_MAX_TOKENS",
            "ANTHROPIC_MAX_TOKENS",
            "OPENAI_SYSTEM_PROMPT",
            "ANTHROPIC_SYSTEM_PROMPT",
            "VOICE_RTP_BIND",
            "VOICE_RTP_PORT",
            "VOICE_RTP_PORT_END",
            "RTP_START",
            "REALTIME_VOICE_RTP_PORT_END",
            "REALTIME_RTP_PORT_POOL_SIZE",
            "REALTIME_MAX_CONCURRENT_CALLS",
            "REALTIME_RTP_RECEIVE_QUEUE_FRAMES",
        ):
            with (
                self.subTest(variable=variable),
                patch.dict(
                    "os.environ",
                    {variable: "321", "OPENAI_API_KEY": "must-not-affect-cohort"},
                    clear=True,
                ),
                redirect_stdout(io.StringIO()) as tuned_stdout,
            ):
                tuned_exit = voice_evidence_main(
                    ["cohort", "--image-identity", "gateway-image-123"]
                )

            self.assertEqual(base_exit, 0)
            self.assertEqual(tuned_exit, 0)
            self.assertNotEqual(tuned_stdout.getvalue().strip(), base)

    def test_cohort_cli_ignores_unknown_prefixed_environment_state(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            first_exit = voice_evidence_main(
                ["cohort", "--image-identity", "gateway-image-123"]
            )
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_SESSION_COOKIE": "operational-cookie",
                    "REALTIME_FUTURE_CREDENTIAL": "unknown-state",
                    "REALTIME_APPROVAL_PACKET_FINGERPRINT": "unrelated-approval",
                },
                clear=True,
            ),
            redirect_stdout(io.StringIO()) as second_stdout,
        ):
            second_exit = voice_evidence_main(
                ["cohort", "--image-identity", "gateway-image-123"]
            )

        self.assertEqual(first_exit, 0)
        self.assertEqual(second_exit, 0)
        self.assertEqual(stdout.getvalue().strip(), second_stdout.getvalue().strip())

    def test_history_gate_cli_writes_release_decision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            history = root / "history.jsonl"
            output = root / "release-gate.json"
            integrity = root / "integrity-events.jsonl"
            for index in range(30):
                append_history(
                    history,
                    _history_record(
                        f"run-{index}",
                        barge_ms=200.0 if index < 10 else None,
                    ),
                )
            records = [json.loads(line) for line in history.read_text().splitlines()]
            _write_integrity_events(integrity, records)

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "history-gate",
                        "--history",
                        str(history),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--integrity-events",
                        str(integrity),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            gate = json.loads(output.read_text())
            self.assertTrue(gate["ok"])
            self.assertEqual(gate["semantic"]["p90_limit_ms"], 1_200.0)
            self.assertEqual(gate["semantic"]["p95_limit_ms"], 1_500.0)
            self.assertEqual(gate["barge_in"]["p90_limit_ms"], 250.0)
            self.assertEqual(gate["barge_in"]["max_limit_ms"], 400.0)

    def test_history_gate_cli_fails_on_malformed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            history = root / "history.jsonl"
            integrity = root / "integrity-events.jsonl"
            output = root / "release-gate.json"
            for index in range(30):
                append_history(
                    history,
                    _history_record(
                        f"run-{index}", barge_ms=200.0 if index < 10 else None
                    ),
                )
            records = [json.loads(line) for line in history.read_text().splitlines()]
            _write_integrity_events(integrity, records)
            with history.open("a", encoding="utf-8") as handle:
                handle.write("{malformed-json\n")
                handle.write('{"run_id":"one","run_id":"two"}\n')
                handle.write('{"metric":NaN}\n')

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "history-gate",
                        "--history",
                        str(history),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--integrity-events",
                        str(integrity),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 1)
            gate = json.loads(output.read_text())
            self.assertEqual(gate["malformed_records"], 3)

    def test_history_gate_cli_consumes_host_integrity_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            history = root / "history.jsonl"
            integrity = root / "integrity-events.jsonl"
            for index in range(30):
                append_history(
                    history,
                    _history_record(
                        f"run-{index}", barge_ms=200.0 if index < 10 else None
                    ),
                )
            records = [json.loads(line) for line in history.read_text().splitlines()]
            _write_integrity_events(
                integrity,
                records,
                extra=[_integrity_event("lost-run", sequence=2 * len(records) + 1)],
            )

            with redirect_stdout(io.StringIO()):
                exit_code = voice_evidence_main(
                    [
                        "history-gate",
                        "--history",
                        str(history),
                        "--cohort",
                        "flash-v1",
                        "--campaign",
                        CAMPAIGN_ID,
                        "--ledger-id",
                        LEDGER_ID,
                        "--topology",
                        "pstn-loopback",
                        "--route",
                        "didww-outbound-us",
                        "--integrity-events",
                        str(integrity),
                    ]
                )

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()

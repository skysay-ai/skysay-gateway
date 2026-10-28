# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import struct
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .semantic_audio_policy import valid_semantic_response_audio_policy


MEASUREMENT_SOURCE = "caller_side_mixmonitor_split_track_vad_semantic_clip_correlation"
UNVERIFIED_MEASUREMENT_SOURCE = "caller_side_mixmonitor_split_track_vad_unverified"
SEMANTIC_ALIGNMENT_SOURCE = "caller_side_vad_worker_turn_clip_correlation"
UNVERIFIED_ALIGNMENT_SOURCE = "unverified_first_agent_vad"
HISTORY_SCHEMA = "skysay.voice_evidence.v1"
HISTORY_SCHEMA_VERSION = 2
SEMANTIC_FEATURE_FRAME_MS = 10
SEMANTIC_TEMPLATE_MIN_MS = 500
SEMANTIC_TEMPLATE_MAX_MS = 1_500
SEMANTIC_ALIGNMENT_MIN_SCORE = 0.85
SEMANTIC_ALIGNMENT_MIN_MARGIN = 0.08
SEMANTIC_ALIGNMENT_SEARCH_MS = 100
SEMANTIC_ALIGNMENT_RUNNER_EXCLUSION_MS = 100
SEMANTIC_ALIGNMENT_MAX_TRACK_MS = 90_000
SEMANTIC_ALIGNMENT_MAX_PRIMARY_CANDIDATES = 2_048
SEMANTIC_ALIGNMENT_MAX_CORRELATION_FRAME_PAIRS = 10_000_000
SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR = "semantic_turn_alignment_work_limit_exceeded"


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 20
    speech_rms: float = 300.0
    noise_multiplier: float = 4.0
    min_speech_ms: int = 60
    silence_confirm_ms: int = 80
    merge_gap_ms: int = 600
    agent_merge_gap_ms: int = 120
    min_agent_before_barge_ms: int = 200


@dataclass(frozen=True)
class SpeechSpan:
    start_sample: int
    end_sample: int
    end_confirmed: bool


@dataclass(frozen=True)
class SemanticTurnEvidence:
    turn: int
    caller_speech_end_sample: int
    agent_speech_start_sample: int
    semantic_voice_to_voice_ms: float
    reference_sha256: str = ""
    reference_vad_start_sample: int = 0
    alignment_sample_count: int = 0
    alignment_score: float = 0.0
    runner_up_margin: float = 0.0


@dataclass(frozen=True)
class BargeInEvidence:
    turn: int
    caller_speech_start_sample: int
    agent_speech_end_sample: int
    audible_stop_ms: float


@dataclass(frozen=True)
class VoiceEvidence:
    valid: bool
    semantic_turns: tuple[SemanticTurnEvidence, ...]
    barge_ins: tuple[BargeInEvidence, ...]
    invalid_reasons: tuple[str, ...]
    sample_rate_hz: int
    measurement_source: str = MEASUREMENT_SOURCE
    semantic_alignment_source: str = UNVERIFIED_ALIGNMENT_SOURCE


@dataclass(frozen=True)
class _SemanticAlignment:
    start_sample: int
    reference_sha256: str
    reference_vad_start_sample: int
    sample_count: int
    score: float
    runner_up_margin: float


@dataclass(frozen=True)
class _SemanticCandidate:
    score: float
    start_sample: int


@dataclass(frozen=True)
class _SemanticFeatureContext:
    frame_samples: int
    agent_features: list[tuple[float, float]]


@dataclass
class _AlignmentWorkBudget:
    remaining_frame_pairs: int

    def consume(self, frame_pairs: int) -> bool:
        if frame_pairs < 0 or frame_pairs > self.remaining_frame_pairs:
            return False
        self.remaining_frame_pairs -= frame_pairs
        return True


def analyze_split_tracks(
    caller_spoken_path: Path,
    agent_heard_path: Path,
    *,
    transcript: dict[str, Any] | None = None,
    semantic_turn_clips: dict[int, Path] | None = None,
    config: VadConfig | None = None,
) -> VoiceEvidence:
    """Measure semantic response and audible barge-stop from synchronized WAVs.

    Caller speech spans map to ``transcript.turns`` by caller ordinal. Short
    intra-utterance pauses are merged, while the span still ends at the final
    active audio frame. A supplied transcript must map one-to-one or the run is
    rejected as ambiguous. The opening greeting is agent-only audio and is
    therefore absent from both sequences; a caller span classified as barge-in
    still consumes its ordinal. Transcript content only validates semantic
    identity. Every timestamp comes from a WAV sample position, never a worker
    event or file timestamp.
    """

    config = config or VadConfig()
    if not _valid_config(config):
        return VoiceEvidence(False, (), (), ("invalid_vad_config",), 0)
    try:
        caller_rate, caller_samples = _read_pcm16_mono(caller_spoken_path)
    except (EOFError, OSError, ValueError, ZeroDivisionError, struct.error, wave.Error):
        return VoiceEvidence(False, (), (), ("caller_track_unreadable",), 0)
    try:
        agent_rate, agent_samples = _read_pcm16_mono(agent_heard_path)
    except (EOFError, OSError, ValueError, ZeroDivisionError, struct.error, wave.Error):
        return VoiceEvidence(False, (), (), ("agent_track_unreadable",), caller_rate)
    if caller_rate != agent_rate:
        return VoiceEvidence(False, (), (), ("split_track_sample_rate_mismatch",), caller_rate)
    if len(caller_samples) != len(agent_samples):
        return VoiceEvidence(False, (), (), ("split_track_duration_mismatch",), caller_rate)
    if not caller_samples:
        return VoiceEvidence(False, (), (), ("empty_split_tracks",), caller_rate)
    if (caller_rate * config.frame_ms) % 1_000:
        return VoiceEvidence(False, (), (), ("frame_size_not_integral_for_sample_rate",), caller_rate)

    caller_spans = _speech_spans(caller_samples, caller_rate, config)
    agent_spans = _speech_spans(
        agent_samples,
        agent_rate,
        config,
        merge_gap_ms=config.agent_merge_gap_ms,
    )
    semantic_agent_spans = _speech_spans(agent_samples, agent_rate, config, merge_gap_ms=0)
    alignment_source = (
        SEMANTIC_ALIGNMENT_SOURCE
        if semantic_turn_clips is not None
        else UNVERIFIED_ALIGNMENT_SOURCE
    )
    transcript_turns: list[object] = []
    if transcript is not None:
        raw_turns = transcript.get("turns") if isinstance(transcript, dict) else None
        if not isinstance(raw_turns, list):
            return VoiceEvidence(False, (), (), ("transcript_turns_invalid",), caller_rate)
        if len(raw_turns) != len(caller_spans):
            return VoiceEvidence(False, (), (), ("caller_transcript_turn_count_mismatch",), caller_rate)
        transcript_turns = raw_turns
    feature_context: _SemanticFeatureContext | None = None
    alignment_work_budget: _AlignmentWorkBudget | None = None
    if semantic_turn_clips is not None:
        feature_frame_samples = round(caller_rate * SEMANTIC_FEATURE_FRAME_MS / 1_000)
        maximum_track_samples = round(caller_rate * SEMANTIC_ALIGNMENT_MAX_TRACK_MS / 1_000)
        if feature_frame_samples <= 0:
            return VoiceEvidence(
                False,
                (),
                (),
                ("semantic_turn_clip_invalid_feature_rate",),
                caller_rate,
                semantic_alignment_source=alignment_source,
            )
        if len(agent_samples) > maximum_track_samples:
            return VoiceEvidence(
                False,
                (),
                (),
                (SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR,),
                caller_rate,
                semantic_alignment_source=alignment_source,
            )
        feature_context = _SemanticFeatureContext(
            frame_samples=feature_frame_samples,
            agent_features=_acoustic_features(
                agent_samples,
                0,
                len(agent_samples),
                feature_frame_samples,
            ),
        )
        alignment_work_budget = _AlignmentWorkBudget(
            SEMANTIC_ALIGNMENT_MAX_CORRELATION_FRAME_PAIRS
        )
    semantic_turns: list[SemanticTurnEvidence] = []
    barge_ins: list[BargeInEvidence] = []
    for index, caller_span in enumerate(caller_spans):
        transcript_turn = transcript_turns[index] if index < len(transcript_turns) else {}
        prior_turn = transcript_turns[index - 1] if transcript is not None and index > 0 else None
        barge_corroborated = transcript is None or (
            isinstance(prior_turn, dict) and prior_turn.get("interrupted") is True
        )
        overlapping_agent_entry = next(
            (
                (span_index, span)
                for span_index, span in enumerate(agent_spans)
                if span.start_sample < caller_span.start_sample < span.end_sample
                and span.end_confirmed
                and (caller_span.start_sample - span.start_sample) / caller_rate * 1_000.0
                >= config.min_agent_before_barge_ms
                and barge_corroborated
            ),
            None,
        )
        if overlapping_agent_entry is not None:
            overlapping_index, overlapping_agent = overlapping_agent_entry
            if overlapping_agent.end_sample > caller_span.end_sample:
                return VoiceEvidence(
                    False,
                    (),
                    (),
                    ("agent_did_not_stop_before_caller_finished",),
                    caller_rate,
                    semantic_alignment_source=alignment_source,
                )
            if any(
                span.start_sample < caller_span.end_sample
                for span in agent_spans[overlapping_index + 1 :]
            ):
                return VoiceEvidence(
                    False,
                    (),
                    (),
                    ("agent_resumed_before_caller_stopped",),
                    caller_rate,
                    semantic_alignment_source=alignment_source,
                )
            barge_ins.append(
                BargeInEvidence(
                    turn=index + 1,
                    caller_speech_start_sample=caller_span.start_sample,
                    agent_speech_end_sample=overlapping_agent.end_sample,
                    audible_stop_ms=round(
                        (overlapping_agent.end_sample - caller_span.start_sample) / caller_rate * 1_000.0,
                        3,
                    ),
                )
            )
            continue
        if transcript is not None and not has_measurable_semantic_text(
            transcript_turn,
            clip_backed=semantic_turn_clips is not None,
        ):
            continue
        next_caller_start = caller_spans[index + 1].start_sample if index + 1 < len(caller_spans) else None
        alignment: _SemanticAlignment | None = None
        if semantic_turn_clips is not None:
            assert feature_context is not None and alignment_work_budget is not None
            clip_path = semantic_turn_clips.get(index + 1)
            if clip_path is None:
                return VoiceEvidence(
                    False,
                    (),
                    (),
                    ("semantic_turn_clip_missing",),
                    caller_rate,
                    semantic_alignment_source=alignment_source,
                )
            alignment, alignment_error = _align_semantic_turn(
                agent_samples,
                caller_rate,
                semantic_agent_spans,
                clip_path,
                caller_span.end_sample,
                next_caller_start,
                config,
                feature_context,
                alignment_work_budget,
            )
            if alignment_error or alignment is None:
                return VoiceEvidence(
                    False,
                    (),
                    (),
                    (alignment_error or "semantic_turn_clip_not_aligned",),
                    caller_rate,
                    semantic_alignment_source=alignment_source,
                )
            response_start_sample = alignment.start_sample
        else:
            response = next(
                (
                    span
                    for span in agent_spans
                    if span.start_sample > caller_span.end_sample
                    and (next_caller_start is None or span.start_sample < next_caller_start)
                ),
                None,
            )
            response_start_sample = response.start_sample if response is not None else None
        if response_start_sample is None:
            continue
        latency_ms = (response_start_sample - caller_span.end_sample) / caller_rate * 1_000.0
        semantic_turns.append(
            SemanticTurnEvidence(
                turn=index + 1,
                caller_speech_end_sample=caller_span.end_sample,
                agent_speech_start_sample=response_start_sample,
                semantic_voice_to_voice_ms=round(latency_ms, 3),
                reference_sha256=alignment.reference_sha256 if alignment else "",
                reference_vad_start_sample=alignment.reference_vad_start_sample if alignment else 0,
                alignment_sample_count=alignment.sample_count if alignment else 0,
                alignment_score=alignment.score if alignment else 0.0,
                runner_up_margin=alignment.runner_up_margin if alignment else 0.0,
            )
        )

    reasons = () if semantic_turns or barge_ins else ("voice_evidence_not_found",)
    return VoiceEvidence(
        bool(semantic_turns or barge_ins),
        tuple(semantic_turns),
        tuple(barge_ins),
        reasons,
        caller_rate,
        measurement_source=(
            MEASUREMENT_SOURCE
            if semantic_turn_clips is not None
            else UNVERIFIED_MEASUREMENT_SOURCE
        ),
        semantic_alignment_source=alignment_source,
    )


def _align_semantic_turn(
    agent_samples: list[int],
    sample_rate: int,
    agent_spans: list[SpeechSpan],
    clip_path: Path,
    caller_end_sample: int,
    next_caller_start_sample: int | None,
    config: VadConfig,
    feature_context: _SemanticFeatureContext,
    work_budget: _AlignmentWorkBudget,
) -> tuple[_SemanticAlignment | None, str | None]:
    try:
        reference_bytes = clip_path.read_bytes()
        reference_rate, reference_samples = _read_pcm16_mono(clip_path)
    except (EOFError, OSError, ValueError, ZeroDivisionError, struct.error, wave.Error):
        return None, "semantic_turn_clip_unreadable"
    if reference_rate != sample_rate:
        return None, "semantic_turn_clip_sample_rate_mismatch"
    maximum_track_samples = round(sample_rate * SEMANTIC_ALIGNMENT_MAX_TRACK_MS / 1_000)
    if len(reference_samples) > maximum_track_samples:
        return None, SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR
    reference_spans = _speech_spans(
        reference_samples,
        reference_rate,
        config,
        merge_gap_ms=0,
        adaptive_noise=False,
    )
    if not reference_spans:
        return None, "semantic_turn_clip_without_voice"
    reference_start = reference_spans[0].start_sample
    minimum_samples = round(sample_rate * SEMANTIC_TEMPLATE_MIN_MS / 1_000)
    maximum_samples = round(sample_rate * SEMANTIC_TEMPLATE_MAX_MS / 1_000)
    maximum_template_samples = min(maximum_samples, len(reference_samples) - reference_start)
    if maximum_template_samples < minimum_samples:
        return None, "semantic_turn_clip_too_short"
    feature_frame_samples = feature_context.frame_samples
    if feature_frame_samples <= 0 or reference_start % feature_frame_samples:
        return None, "semantic_turn_clip_invalid_feature_rate"
    reference_feature_series = _acoustic_features(
        reference_samples,
        0,
        len(reference_samples),
        feature_frame_samples,
    )
    window_end = next_caller_start_sample if next_caller_start_sample is not None else len(agent_samples)
    eligible_spans = [
        span
        for span in agent_spans
        if caller_end_sample < span.start_sample < window_end
    ]
    if not eligible_spans:
        return None, "semantic_turn_clip_not_aligned"
    maximum_template_samples -= maximum_template_samples % feature_frame_samples
    minimum_template_frames = math.ceil(minimum_samples / feature_frame_samples)
    reference_start_frame = reference_start // feature_frame_samples
    template_lengths = range(
        maximum_template_samples,
        minimum_template_frames * feature_frame_samples - 1,
        -feature_frame_samples,
    )
    exclusion_samples = round(sample_rate * SEMANTIC_ALIGNMENT_RUNNER_EXCLUSION_MS / 1_000)
    search_samples = round(sample_rate * SEMANTIC_ALIGNMENT_SEARCH_MS / 1_000)
    search_samples -= search_samples % feature_frame_samples
    primary_candidate_starts = sorted(
        {
            candidate_start
            for span in eligible_spans
            for offset in range(-search_samples, search_samples + 1, feature_frame_samples)
            for candidate_start in (span.start_sample + offset,)
            if candidate_start >= 0
            and candidate_start % feature_frame_samples == 0
            and caller_end_sample < candidate_start < window_end
            and candidate_start + minimum_template_frames * feature_frame_samples <= window_end
        }
    )
    if not primary_candidate_starts:
        return None, "semantic_turn_clip_not_aligned"
    if len(primary_candidate_starts) > SEMANTIC_ALIGNMENT_MAX_PRIMARY_CANDIDATES:
        return None, SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR
    saw_high_correlation = False
    for template_samples in template_lengths:
        template_frames = template_samples // feature_frame_samples
        reference_features = reference_feature_series[
            reference_start_frame : reference_start_frame + template_frames
        ]
        if len(reference_features) != template_frames:
            continue
        candidates: list[_SemanticCandidate] = []
        for candidate_start in primary_candidate_starts:
            if candidate_start + template_samples > window_end:
                continue
            score, work_limit_reached = _score_feature_window(
                reference_features,
                feature_context.agent_features,
                candidate_start // feature_frame_samples,
                work_budget,
            )
            if work_limit_reached:
                return None, SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR
            if score is not None:
                candidates.append(_SemanticCandidate(score, candidate_start))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.score, reverse=True)
        best = candidates[0]
        if best.score < SEMANTIC_ALIGNMENT_MIN_SCORE:
            continue
        saw_high_correlation = True
        runner_score = 0.0
        first_runner_start = ((caller_end_sample // feature_frame_samples) + 1) * feature_frame_samples
        last_runner_start = window_end - template_samples
        for candidate_start in range(
            first_runner_start,
            last_runner_start + 1,
            feature_frame_samples,
        ):
            if abs(candidate_start - best.start_sample) < exclusion_samples:
                continue
            score, work_limit_reached = _score_feature_window(
                reference_features,
                feature_context.agent_features,
                candidate_start // feature_frame_samples,
                work_budget,
            )
            if work_limit_reached:
                return None, SEMANTIC_ALIGNMENT_WORK_LIMIT_ERROR
            if score is not None:
                runner_score = max(runner_score, score)
        margin = best.score - runner_score
        if margin < SEMANTIC_ALIGNMENT_MIN_MARGIN:
            continue
        return (
            _SemanticAlignment(
                start_sample=best.start_sample,
                reference_sha256=f"sha256:{hashlib.sha256(reference_bytes).hexdigest()}",
                reference_vad_start_sample=reference_start,
                sample_count=template_samples,
                score=round(best.score, 6),
                runner_up_margin=round(margin, 6),
            ),
            None,
        )
    return None, (
        "semantic_turn_clip_ambiguous"
        if saw_high_correlation
        else "semantic_turn_clip_low_correlation"
    )


def _acoustic_features(
    samples: list[int],
    start_sample: int,
    sample_count: int,
    frame_samples: int,
) -> list[tuple[float, float]]:
    features: list[tuple[float, float]] = []
    end_sample = min(len(samples), start_sample + sample_count)
    for offset in range(start_sample, end_sample - frame_samples + 1, frame_samples):
        frame = samples[offset : offset + frame_samples]
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        zero_crossings = sum(
            1
            for left, right in zip(frame, frame[1:])
            if (left < 0 <= right) or (left >= 0 > right)
        )
        features.append((math.log1p(rms), zero_crossings / max(1, len(frame) - 1)))
    return features


def _score_feature_window(
    reference_features: list[tuple[float, float]],
    agent_features: list[tuple[float, float]],
    candidate_start_frame: int,
    work_budget: _AlignmentWorkBudget,
) -> tuple[float | None, bool]:
    candidate_end_frame = candidate_start_frame + len(reference_features)
    if candidate_start_frame < 0 or candidate_end_frame > len(agent_features):
        return None, False
    if not work_budget.consume(len(reference_features)):
        return None, True
    return (
        _combined_feature_correlation(
            reference_features,
            agent_features[candidate_start_frame:candidate_end_frame],
        ),
        False,
    )


def _combined_feature_correlation(
    reference: list[tuple[float, float]],
    candidate: list[tuple[float, float]],
) -> float | None:
    if len(reference) != len(candidate) or not reference:
        return None
    scores: list[float] = []
    for feature_index in range(2):
        score = _normalized_correlation(
            [item[feature_index] for item in reference],
            [item[feature_index] for item in candidate],
        )
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else None


def _normalized_correlation(reference: list[float], candidate: list[float]) -> float | None:
    reference_mean = sum(reference) / len(reference)
    candidate_mean = sum(candidate) / len(candidate)
    centered_reference = [value - reference_mean for value in reference]
    centered_candidate = [value - candidate_mean for value in candidate]
    denominator = math.sqrt(
        sum(value * value for value in centered_reference)
        * sum(value * value for value in centered_candidate)
    )
    if denominator <= 1e-12:
        return None
    score = sum(
        left * right
        for left, right in zip(centered_reference, centered_candidate)
    ) / denominator
    return max(-1.0, min(1.0, score))


def _valid_config(config: VadConfig) -> bool:
    try:
        return bool(
            _config_int(config.frame_ms)
            and _config_int(config.min_speech_ms)
            and _config_int(config.silence_confirm_ms)
            and _config_int(config.merge_gap_ms)
            and _config_int(config.agent_merge_gap_ms)
            and _config_int(config.min_agent_before_barge_ms)
            and _config_number(config.speech_rms)
            and _config_number(config.noise_multiplier)
            and 0 < config.frame_ms <= 1_000
            and config.speech_rms > 0
            and config.noise_multiplier >= 1.0
            and config.min_speech_ms > 0
            and config.silence_confirm_ms > 0
            and config.merge_gap_ms >= config.silence_confirm_ms
            and config.agent_merge_gap_ms >= config.silence_confirm_ms
            and config.min_agent_before_barge_ms >= 0
        )
    except (OverflowError, TypeError):
        return False


def _config_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _config_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def append_history(path: Path, record: dict[str, Any]) -> None:
    """Append one complete JSONL record under an exclusive process lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _record_sha256(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def evaluate_history(
    records: list[dict[str, Any]],
    *,
    cohort: str,
    base_cohort: str | None = None,
    campaign_id: str,
    ledger_id: str,
    topology: str,
    route: str,
    min_semantic_samples: int = 30,
    semantic_p90_limit_ms: float = 1_200.0,
    semantic_p95_limit_ms: float = 1_500.0,
    min_barge_samples: int = 10,
    barge_p90_limit_ms: float = 250.0,
    barge_max_limit_ms: float = 400.0,
    malformed_records: int = 0,
    integrity_events: list[dict[str, Any]] | None = None,
    malformed_integrity_events: int = 0,
    now: datetime | None = None,
    max_age_seconds: float = 14 * 24 * 60 * 60,
) -> dict[str, Any]:
    """Evaluate authenticated-shape evidence without pooling stack cohorts."""

    selected_base_cohort = cohort if base_cohort is None else base_cohort
    if not _nonempty_record_text(cohort):
        raise ValueError("cohort must be nonempty")
    if not _nonempty_record_text(selected_base_cohort):
        raise ValueError("base_cohort must be nonempty")
    if not isinstance(campaign_id, str) or not _sha256_identity(campaign_id):
        raise ValueError("campaign_id must be a sha256 identity")
    if not isinstance(ledger_id, str) or not _sha256_identity(ledger_id):
        raise ValueError("ledger_id must be a sha256 identity")
    if not _nonempty_record_text(topology):
        raise ValueError("topology must be nonempty")
    if not _nonempty_record_text(route):
        raise ValueError("route must be nonempty")
    if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)):
        raise ValueError("max_age_seconds must be numeric")
    if not math.isfinite(float(max_age_seconds)) or float(max_age_seconds) <= 0:
        raise ValueError("max_age_seconds must be positive")
    evaluated_at = now or datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    evaluated_at = evaluated_at.astimezone(timezone.utc)
    freshness_cutoff = evaluated_at - timedelta(seconds=float(max_age_seconds))

    semantic: list[float] = []
    barge_in: list[float] = []
    matching_records = 0
    duplicate_run_ids = 0
    conflicting_duplicate_run_ids = 0
    duplicate_source_evidence = 0
    rejected_records = 0
    invalid_or_incomplete_runs = 0
    base_incomplete_runs = 0
    stale_records = 0
    history_attempt_hashes: dict[tuple[str, ...], str] = {}
    ledger_history_hashes: dict[tuple[str, ...], str] = {}
    foreign_ledger_records = 0
    seen_runs: dict[str, dict[str, Any]] = {}
    source_record_hashes: dict[str, set[str]] = {}
    unique_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not _valid_history_record(record):
            rejected_records += 1
            continue
        record_hash = _record_sha256(record)
        if record.get("measurement_complete") is True:
            source_fingerprint = str(record["source_evidence_fingerprint"])
            source_record_hashes.setdefault(source_fingerprint, set()).add(record_hash)
        run_id = str(record["run_id"])
        previous = seen_runs.get(run_id)
        if previous is not None:
            duplicate_run_ids += 1
            if record != previous:
                conflicting_duplicate_run_ids += 1
            continue
        seen_runs[run_id] = record
        unique_records.append(record)
        if record.get("ledger_id") != ledger_id:
            foreign_ledger_records += 1
            continue
        ledger_attempt_key = (
            run_id,
            str(record["campaign_id"]),
            str(record["base_cohort"]),
            str(record["topology"]),
            str(record["route"]),
            str(record["context"]),
        )
        ledger_history_hashes[ledger_attempt_key] = record_hash
    rejected_source_fingerprints = {
        fingerprint
        for fingerprint, record_hashes in source_record_hashes.items()
        if len(record_hashes) > 1
    }
    duplicate_source_evidence = len(rejected_source_fingerprints)
    for record in unique_records:
        record_cohort = record.get("cohort")
        selected_record = record_cohort == cohort
        supporting_base_record = (
            cohort != selected_base_cohort and record_cohort == selected_base_cohort
        )
        if not selected_record and not supporting_base_record:
            continue
        if record.get("campaign_id") != campaign_id or record.get("ledger_id") != ledger_id:
            continue
        if record.get("topology") != topology or record.get("route") != route:
            continue
        if record.get("base_cohort") != selected_base_cohort:
            rejected_records += 1
            continue
        recorded_at = _parse_utc_timestamp(record.get("recorded_at"))
        if recorded_at is None or recorded_at > evaluated_at + timedelta(minutes=5):
            rejected_records += 1
            continue
        run_id = str(record["run_id"])
        attempt_key = (
            run_id,
            str(record["campaign_id"]),
            str(record["base_cohort"]),
            str(record["topology"]),
            str(record["route"]),
            str(record["context"]),
        )
        history_attempt_hashes[attempt_key] = _record_sha256(record)
        source_fingerprint = str(record["source_evidence_fingerprint"])
        if (
            record.get("measurement_complete") is True
            and source_fingerprint in rejected_source_fingerprints
        ):
            continue
        if selected_record:
            matching_records += 1
        if record.get("valid") is not True or record.get("measurement_complete") is not True:
            invalid_or_incomplete_runs += 1
            if supporting_base_record:
                base_incomplete_runs += 1
            continue
        if supporting_base_record:
            continue
        if recorded_at < freshness_cutoff:
            stale_records += 1
            continue
        semantic.extend(_positive_metrics(record.get("semantic_turns"), "semantic_voice_to_voice_ms"))
        barge_in.extend(_positive_metrics(record.get("barge_ins"), "audible_stop_ms"))
    matching_integrity_failures = 0
    rejected_integrity_events = 0
    duplicate_integrity_events = 0
    attempt_starts: dict[tuple[str, ...], int] = {}
    attempt_finalizations: dict[tuple[str, ...], tuple[int, str]] = {}
    ledger_attempt_starts: dict[tuple[str, ...], int] = {}
    ledger_attempt_finalizations: dict[tuple[str, ...], tuple[int, str]] = {}
    for event in integrity_events or []:
        if not isinstance(event, dict) or not _valid_integrity_event(event):
            rejected_integrity_events += 1
            continue
        if event.get("ledger_id") != ledger_id:
            rejected_integrity_events += 1
            continue
        event_campaign = event.get("campaign_id")
        event_cohort = event.get("cohort")
        event_topology = event.get("topology")
        event_route = event.get("route")
        matches_current_gate = bool(
            event_campaign in {campaign_id, "unknown"}
            and event_cohort in {cohort, selected_base_cohort, "unknown"}
            and event_topology in {topology, "unknown"}
            and event_route in {route, "unknown"}
        )
        if event.get("event") == "failure":
            if matches_current_gate:
                matching_integrity_failures += 1
            continue
        attempt_key = (
            str(event["run_id"]),
            str(event["campaign_id"]),
            str(event["cohort"]),
            str(event["topology"]),
            str(event["route"]),
            str(event["context"]),
        )
        if event.get("event") == "attempt_started":
            if attempt_key in ledger_attempt_starts:
                duplicate_integrity_events += 1
                continue
            sequence = int(event["sequence"])
            ledger_attempt_starts[attempt_key] = sequence
            if matches_current_gate:
                attempt_starts[attempt_key] = sequence
        else:
            if attempt_key in ledger_attempt_finalizations:
                duplicate_integrity_events += 1
                continue
            finalization = (
                int(event["sequence"]),
                str(event["evidence_record_hash"]),
            )
            ledger_attempt_finalizations[attempt_key] = finalization
            if matches_current_gate:
                attempt_finalizations[attempt_key] = finalization
    ledger_orphan_attempt_finalizations = sum(
        1
        for key, (sequence, _record_hash) in ledger_attempt_finalizations.items()
        if key not in ledger_attempt_starts or sequence <= ledger_attempt_starts[key]
    )
    ledger_completed_attempt_keys = {
        key
        for key, sequence in ledger_attempt_starts.items()
        if key in ledger_attempt_finalizations
        and ledger_attempt_finalizations[key][0] > sequence
    }
    ledger_history_keys = set(ledger_history_hashes)
    ledger_history_without_durable_attempt = len(
        ledger_history_keys - ledger_completed_attempt_keys
    )
    ledger_durable_attempts_without_history = len(
        ledger_completed_attempt_keys - ledger_history_keys
    )
    ledger_evidence_record_hash_mismatches = sum(
        1
        for key in ledger_history_keys & ledger_completed_attempt_keys
        if ledger_attempt_finalizations[key][1] != ledger_history_hashes[key]
    )
    orphan_attempt_finalizations = sum(
        1
        for key, (sequence, _record_hash) in attempt_finalizations.items()
        if key not in attempt_starts or sequence <= attempt_starts[key]
    )
    unfinalized_attempts = sum(
        1
        for key, sequence in attempt_starts.items()
        if key not in attempt_finalizations or attempt_finalizations[key][0] <= sequence
    )
    completed_attempt_keys = {
        key
        for key, sequence in attempt_starts.items()
        if key in attempt_finalizations and attempt_finalizations[key][0] > sequence
    }
    history_attempt_keys = set(history_attempt_hashes)
    history_without_durable_attempt = len(history_attempt_keys - completed_attempt_keys)
    durable_attempts_without_history = len(completed_attempt_keys - history_attempt_keys)
    evidence_record_hash_mismatches = sum(
        1
        for key in history_attempt_keys & completed_attempt_keys
        if attempt_finalizations[key][1] != history_attempt_hashes[key]
    )
    semantic_gate = _metric_gate(
        semantic,
        min_samples=min_semantic_samples,
        p90_limit_ms=semantic_p90_limit_ms,
        p95_limit_ms=semantic_p95_limit_ms,
    )
    barge_gate = _metric_gate(
        barge_in,
        min_samples=min_barge_samples,
        p90_limit_ms=barge_p90_limit_ms,
        max_limit_ms=barge_max_limit_ms,
    )
    return {
        "ok": (
            invalid_or_incomplete_runs == 0
            and rejected_records == 0
            and foreign_ledger_records == 0
            and duplicate_run_ids == 0
            and conflicting_duplicate_run_ids == 0
            and duplicate_source_evidence == 0
            and malformed_records == 0
            and matching_integrity_failures == 0
            and rejected_integrity_events == 0
            and duplicate_integrity_events == 0
            and malformed_integrity_events == 0
            and unfinalized_attempts == 0
            and orphan_attempt_finalizations == 0
            and ledger_orphan_attempt_finalizations == 0
            and history_without_durable_attempt == 0
            and durable_attempts_without_history == 0
            and evidence_record_hash_mismatches == 0
            and ledger_history_without_durable_attempt == 0
            and ledger_durable_attempts_without_history == 0
            and ledger_evidence_record_hash_mismatches == 0
            and semantic_gate["ok"]
            and barge_gate["ok"]
        ),
        "schema": HISTORY_SCHEMA,
        "source": MEASUREMENT_SOURCE,
        "cohort": cohort,
        "base_cohort": selected_base_cohort,
        "campaign_id": campaign_id,
        "ledger_id": ledger_id,
        "topology": topology,
        "route": route,
        "matching_records": matching_records,
        "duplicate_run_ids": duplicate_run_ids,
        "conflicting_duplicate_run_ids": conflicting_duplicate_run_ids,
        "duplicate_source_evidence": duplicate_source_evidence,
        "rejected_records": rejected_records,
        "foreign_ledger_records": foreign_ledger_records,
        "malformed_records": malformed_records,
        "integrity_failures": matching_integrity_failures,
        "rejected_integrity_events": rejected_integrity_events,
        "duplicate_integrity_events": duplicate_integrity_events,
        "malformed_integrity_events": malformed_integrity_events,
        "unfinalized_attempts": unfinalized_attempts,
        "orphan_attempt_finalizations": orphan_attempt_finalizations,
        "ledger_orphan_attempt_finalizations": ledger_orphan_attempt_finalizations,
        "history_without_durable_attempt": history_without_durable_attempt,
        "durable_attempts_without_history": durable_attempts_without_history,
        "evidence_record_hash_mismatches": evidence_record_hash_mismatches,
        "ledger_history_without_durable_attempt": ledger_history_without_durable_attempt,
        "ledger_durable_attempts_without_history": ledger_durable_attempts_without_history,
        "ledger_evidence_record_hash_mismatches": ledger_evidence_record_hash_mismatches,
        "invalid_or_incomplete_runs": invalid_or_incomplete_runs,
        "base_incomplete_runs": base_incomplete_runs,
        "stale_records": stale_records,
        "all_runs_complete": (
            invalid_or_incomplete_runs == 0
            and duplicate_run_ids == 0
            and matching_integrity_failures == 0
            and unfinalized_attempts == 0
            and orphan_attempt_finalizations == 0
            and ledger_orphan_attempt_finalizations == 0
            and history_without_durable_attempt == 0
            and durable_attempts_without_history == 0
            and evidence_record_hash_mismatches == 0
            and ledger_history_without_durable_attempt == 0
            and ledger_durable_attempts_without_history == 0
            and ledger_evidence_record_hash_mismatches == 0
        ),
        "freshness": {
            "evaluated_at": evaluated_at.isoformat(),
            "cutoff": freshness_cutoff.isoformat(),
            "max_age_hours": round(float(max_age_seconds) / 3_600.0, 3),
        },
        "semantic": semantic_gate,
        "barge_in": barge_gate,
    }


def _valid_history_record(record: dict[str, Any]) -> bool:
    schema_version = record.get("schema_version")
    runtime_gate = record.get("runtime_gate")
    valid = record.get("valid")
    measurement_complete = record.get("measurement_complete")
    runtime_ok = runtime_gate.get("ok") if isinstance(runtime_gate, dict) else None
    runtime_status = runtime_gate.get("status") if isinstance(runtime_gate, dict) else None
    semantic_turns = record.get("semantic_turns")
    barge_ins = record.get("barge_ins")
    sample_rate_hz = record.get("sample_rate_hz")
    return bool(
        record.get("schema") == HISTORY_SCHEMA
        and isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == HISTORY_SCHEMA_VERSION
        and record.get("source") == MEASUREMENT_SOURCE
        and isinstance(valid, bool)
        and isinstance(measurement_complete, bool)
        and isinstance(runtime_gate, dict)
        and isinstance(runtime_ok, bool)
        and runtime_status in {"pass", "fail"}
        and runtime_ok == (runtime_status == "pass")
        and (not measurement_complete or (valid and runtime_ok))
        and isinstance(sample_rate_hz, int)
        and not isinstance(sample_rate_hz, bool)
        and sample_rate_hz >= 0
        and (not measurement_complete or sample_rate_hz > 0)
        and _valid_semantic_metric_items(
            semantic_turns,
            sample_rate_hz,
            require_nonempty=measurement_complete,
        )
        and _valid_semantic_alignment_items(
            semantic_turns,
            sample_rate_hz,
            require_nonempty=measurement_complete,
        )
        and _valid_barge_metric_items(barge_ins, sample_rate_hz)
        and valid_semantic_response_audio_policy(record.get("semantic_response_audio_policy"))
        and (
            record.get("semantic_alignment_source") == SEMANTIC_ALIGNMENT_SOURCE
            or (
                not measurement_complete
                and record.get("semantic_alignment_source") == UNVERIFIED_ALIGNMENT_SOURCE
            )
        )
        and _nonempty_record_text(record.get("run_id"))
        and isinstance(record.get("source_evidence_fingerprint"), str)
        and _sha256_identity(record["source_evidence_fingerprint"])
        and _nonempty_record_text(record.get("cohort"))
        and _nonempty_record_text(record.get("base_cohort"))
        and isinstance(record.get("campaign_id"), str)
        and _sha256_identity(record["campaign_id"])
        and isinstance(record.get("ledger_id"), str)
        and _sha256_identity(record["ledger_id"])
        and _parse_utc_timestamp(record.get("recorded_at")) is not None
        and _nonempty_record_text(record.get("topology"))
        and _nonempty_record_text(record.get("route"))
        and _nonempty_record_text(record.get("context"))
    )


def _valid_integrity_event(record: dict[str, Any]) -> bool:
    expected_keys = {
        "schema",
        "event",
        "sequence",
        "previous_hash",
        "entry_hash",
        "run_id",
        "campaign_id",
        "ledger_id",
        "cohort",
        "topology",
        "route",
        "context",
        "reason",
        "evidence_record_hash",
        "recorded_at",
    }
    sequence = record.get("sequence")
    event = record.get("event")
    unsigned = {key: value for key, value in record.items() if key != "entry_hash"}
    try:
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return False
    expected_hash = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return bool(
        set(record) == expected_keys
        and record.get("schema") in ("skysay.selftest_integrity_event.v2", "openphonex.selftest_integrity_event.v2")  # artifacts written before the rename
        and event in {"failure", "attempt_started", "attempt_finalized"}
        and isinstance(sequence, int)
        and not isinstance(sequence, bool)
        and sequence > 0
        and isinstance(record.get("previous_hash"), str)
        and _sha256_identity(record["previous_hash"])
        and isinstance(record.get("entry_hash"), str)
        and _sha256_identity(record["entry_hash"])
        and record["entry_hash"] == expected_hash
        and _nonempty_record_text(record.get("run_id"))
        and (
            record.get("campaign_id") == "unknown"
            or (
                isinstance(record.get("campaign_id"), str)
                and _sha256_identity(record["campaign_id"])
            )
        )
        and isinstance(record.get("ledger_id"), str)
        and _sha256_identity(record["ledger_id"])
        and _nonempty_record_text(record.get("cohort"))
        and _nonempty_record_text(record.get("topology"))
        and _nonempty_record_text(record.get("route"))
        and _nonempty_record_text(record.get("context"))
        and _nonempty_record_text(record.get("reason"))
        and (
            (
                event == "attempt_finalized"
                and isinstance(record.get("evidence_record_hash"), str)
                and _sha256_identity(record["evidence_record_hash"])
            )
            or (event != "attempt_finalized" and record.get("evidence_record_hash") == "none")
        )
        and _nonempty_record_text(record.get("recorded_at"))
        and _parse_utc_timestamp(record.get("recorded_at")) is not None
        and (
            event == "failure"
            or (
                record.get("campaign_id") != "unknown"
                and record.get("cohort") != "unknown"
                and record.get("topology") != "unknown"
                and record.get("route") != "unknown"
            )
        )
    )


def _valid_semantic_alignment_items(
    items: object,
    sample_rate_hz: int,
    *,
    require_nonempty: bool = False,
) -> bool:
    if not isinstance(items, list) or (require_nonempty and not items):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        reference = item.get("reference_sha256")
        reference_start = item.get("reference_vad_start_sample")
        sample_count = item.get("alignment_sample_count")
        score = item.get("alignment_score")
        margin = item.get("runner_up_margin")
        if not isinstance(reference, str) or not _sha256_identity(reference):
            return False
        if isinstance(reference_start, bool) or not isinstance(reference_start, int) or reference_start < 0:
            return False
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            return False
        minimum_samples = round(sample_rate_hz * SEMANTIC_TEMPLATE_MIN_MS / 1_000)
        maximum_samples = round(sample_rate_hz * SEMANTIC_TEMPLATE_MAX_MS / 1_000)
        if sample_count < minimum_samples or sample_count > maximum_samples:
            return False
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            return False
        if not math.isfinite(float(score)) or float(score) < SEMANTIC_ALIGNMENT_MIN_SCORE or float(score) > 1:
            return False
        if not math.isfinite(float(margin)) or float(margin) < SEMANTIC_ALIGNMENT_MIN_MARGIN:
            return False
    return True


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _sha256_identity(value: str) -> bool:
    return len(value) == 71 and value.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in value[7:]
    )


def _valid_semantic_metric_items(
    items: object,
    sample_rate_hz: int,
    *,
    require_nonempty: bool = False,
) -> bool:
    if not isinstance(items, list) or (require_nonempty and not items):
        return False
    previous_turn = 0
    previous_caller_end = -1
    previous_agent_start = -1
    for item in items:
        if not isinstance(item, dict):
            return False
        turn = item.get("turn")
        caller_end = item.get("caller_speech_end_sample")
        agent_start = item.get("agent_speech_start_sample")
        value = item.get("semantic_voice_to_voice_ms")
        if not _positive_record_int(turn):
            return False
        if turn <= previous_turn:
            return False
        if not _nonnegative_record_int(caller_end) or not _positive_record_int(agent_start):
            return False
        if caller_end <= previous_caller_end or agent_start <= previous_agent_start:
            return False
        if agent_start <= caller_end or not _consistent_sample_metric(
            value,
            agent_start - caller_end,
            sample_rate_hz,
        ):
            return False
        previous_turn = turn
        previous_caller_end = caller_end
        previous_agent_start = agent_start
    return True


def _valid_barge_metric_items(items: object, sample_rate_hz: int) -> bool:
    if not isinstance(items, list):
        return False
    previous_turn = 0
    previous_caller_start = -1
    previous_agent_end = -1
    for item in items:
        if not isinstance(item, dict):
            return False
        turn = item.get("turn")
        caller_start = item.get("caller_speech_start_sample")
        agent_end = item.get("agent_speech_end_sample")
        value = item.get("audible_stop_ms")
        if not _positive_record_int(turn):
            return False
        if turn <= previous_turn:
            return False
        if not _nonnegative_record_int(caller_start) or not _positive_record_int(agent_end):
            return False
        if caller_start <= previous_caller_start or agent_end <= previous_agent_end:
            return False
        if agent_end <= caller_start or not _consistent_sample_metric(
            value,
            agent_end - caller_start,
            sample_rate_hz,
        ):
            return False
        previous_turn = turn
        previous_caller_start = caller_start
        previous_agent_end = agent_end
    return True


def _consistent_sample_metric(value: object, sample_delta: int, sample_rate_hz: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if number <= 0 or not math.isfinite(number) or sample_rate_hz <= 0:
        return False
    expected = sample_delta / sample_rate_hz * 1_000.0
    return math.isclose(number, expected, abs_tol=1_000.0 / sample_rate_hz)


def _positive_record_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_record_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonempty_record_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_metrics(items: object, key: str) -> list[float]:
    if not isinstance(items, list):
        return []
    values: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number > 0 and math.isfinite(number):
            values.append(number)
    return values


def _metric_gate(
    values: list[float],
    *,
    min_samples: int,
    p90_limit_ms: float,
    p95_limit_ms: float | None = None,
    max_limit_ms: float | None = None,
) -> dict[str, Any]:
    ordered = sorted(values)
    p90 = round(_percentile(ordered, 0.90), 3) if ordered else None
    p95 = round(_percentile(ordered, 0.95), 3) if ordered else None
    maximum = round(ordered[-1], 3) if ordered else None
    enough = len(ordered) >= min_samples
    p90_within_limit = p90 is not None and p90 <= p90_limit_ms
    p95_within_limit = p95_limit_ms is None or (p95 is not None and p95 <= p95_limit_ms)
    max_within_limit = max_limit_ms is None or (maximum is not None and maximum <= max_limit_ms)
    return {
        "ok": enough and p90_within_limit and p95_within_limit and max_within_limit,
        "sample_count": len(ordered),
        "min_samples": min_samples,
        "p90_ms": p90,
        "p90_limit_ms": p90_limit_ms,
        "p95_ms": p95,
        "p95_limit_ms": p95_limit_ms,
        "max_ms": maximum,
        "max_limit_ms": max_limit_ms,
        "enough_samples": enough,
        "within_limit": p90_within_limit and p95_within_limit and max_within_limit,
        "p90_within_limit": p90_within_limit,
        "p95_within_limit": p95_within_limit,
        "max_within_limit": max_within_limit,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    lower = math.floor(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def has_measurable_semantic_text(turn: object, *, clip_backed: bool = False) -> bool:
    if not isinstance(turn, dict):
        return False
    spoken = str(turn.get("assistant_spoken") or "").strip()
    # A mid-phrase barge-in can cancel RTP before the pipeline commits any
    # complete phrase to assistant_spoken. The generated semantic reply still
    # identifies the exact recorded turn clip; alignment below proves that a
    # prefix of that clip was actually audible before the interruption.
    interrupted_semantic = clip_backed and turn.get("interrupted") is True and bool(
        str(turn.get("assistant") or "").strip()
    )
    return bool(
        str(turn.get("user") or "").strip()
        and (spoken or interrupted_semantic)
        and not str(turn.get("llm_fallback_leg") or "").strip()
    )


def _read_pcm16_mono(path: Path) -> tuple[int, list[int]]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("voice evidence requires mono 16-bit PCM WAV tracks")
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError("voice evidence WAV track has an invalid sample rate")
        declared_frames = handle.getnframes()
        payload = handle.readframes(declared_frames)
    if len(payload) != declared_frames * 2:
        raise ValueError("voice evidence WAV track is truncated")
    samples = list(struct.unpack(f"<{len(payload) // 2}h", payload)) if payload else []
    return rate, samples


def _speech_spans(
    samples: list[int],
    sample_rate: int,
    config: VadConfig,
    *,
    merge_gap_ms: int | None = None,
    adaptive_noise: bool = True,
) -> list[SpeechSpan]:
    frame_samples = max(1, round(sample_rate * config.frame_ms / 1_000))
    min_active_frames = max(1, math.ceil(config.min_speech_ms / config.frame_ms))
    silence_frames = max(1, math.ceil(config.silence_confirm_ms / config.frame_ms))
    frame_rms: list[float] = []
    for start in range(0, len(samples), frame_samples):
        frame = samples[start : start + frame_samples]
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame)) if frame else 0.0
        frame_rms.append(rms)
    ordered_rms = sorted(frame_rms)
    noise_floor = _percentile(ordered_rms, 0.20) if ordered_rms else 0.0
    threshold = (
        max(config.speech_rms, noise_floor * config.noise_multiplier)
        if adaptive_noise
        else config.speech_rms
    )
    active = [rms >= threshold for rms in frame_rms]

    spans: list[SpeechSpan] = []
    candidate_start: int | None = None
    active_count = 0
    last_active = 0
    silence_count = 0
    for index, is_active in enumerate(active):
        if is_active:
            if candidate_start is None:
                candidate_start = index
                active_count = 0
            active_count += 1
            last_active = index
            silence_count = 0
            continue
        if candidate_start is None:
            continue
        silence_count += 1
        if silence_count < silence_frames:
            continue
        if active_count >= min_active_frames:
            spans.append(
                SpeechSpan(
                    start_sample=candidate_start * frame_samples,
                    end_sample=min(len(samples), (last_active + 1) * frame_samples),
                    end_confirmed=True,
                )
            )
        candidate_start = None
        active_count = 0
        silence_count = 0
    if candidate_start is not None and active_count >= min_active_frames:
        spans.append(
            SpeechSpan(
                start_sample=candidate_start * frame_samples,
                end_sample=min(len(samples), (last_active + 1) * frame_samples),
                end_confirmed=False,
            )
        )
    return _merge_speech_spans(
        spans,
        sample_rate,
        config.merge_gap_ms if merge_gap_ms is None else merge_gap_ms,
    )


def _merge_speech_spans(
    spans: list[SpeechSpan],
    sample_rate: int,
    merge_gap_ms: int,
) -> list[SpeechSpan]:
    """Join short intra-utterance pauses without extending audible speech."""

    if not spans:
        return []
    maximum_gap_samples = round(sample_rate * merge_gap_ms / 1_000)
    merged = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if span.start_sample - previous.end_sample <= maximum_gap_samples:
            merged[-1] = SpeechSpan(
                start_sample=previous.start_sample,
                end_sample=span.end_sample,
                end_confirmed=span.end_confirmed,
            )
        else:
            merged.append(span)
    return merged

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Room-side synchronized evidence recorder for the LiveKit media path.

The listening mix is numpy array arithmetic. It used to be pure Python and
stdlib-only, and a long call's mix then took seconds of CPU holding the GIL:
even in a worker thread (``finalize_off_loop``) it starved the event loop that
serves the worker's other live calls. numpy's array operations release the
GIL, and the mix is byte-for-byte the one the pure-Python code wrote: float64
in the same order of operations, round half to even (``numpy.rint`` is
Python's ``round``), exact integer sums (golden-tested). numpy was already in
every voice image through livekit-agents; the worker image's requirements
pin the version those images carried.
"""
from __future__ import annotations

import array
import asyncio
import contextlib
import functools
import json
import hashlib
import logging
import math
import os
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping

import numpy

try:
    from .evidence_contract import semantic_response_audio_policy
except ImportError:  # copied into /app in the pinned sandbox image
    from evidence_contract import semantic_response_audio_policy

try:
    from .navigation_entry import WITHHELD_TEXT, Consumer, FarEnd, Own
except ImportError:  # copied into /app in the pinned sandbox image
    from navigation_entry import WITHHELD_TEXT, Consumer, FarEnd, Own  # type: ignore[no-redef]


Track = Literal["caller", "agent"]

# How far past an agent interval's end a terminal message may be stamped and
# still be SELECTED for that interval by the entries matcher, which resolves an
# unattributed goodbye against all unused intervals and needs timing to
# disambiguate.  ``add_terminal_message`` fires right after completed playout
# while the interval end comes from the agent state-change event, so the two
# observations of the same playout end can land in either order; a real call
# showed 3ms.  (``_terminal_item_for_response`` does NOT use this: it validates
# an interval already selected positionally, so it floors the stamp at the
# interval start with no ceiling -- a timed-out wait can delay the record many
# seconds past the audio's end.)
_TERMINAL_ATTRIBUTION_SLACK_SECONDS = 2.0

# ``add_text`` and ``mark_state`` are callbacks from independent LiveKit
# event streams.  A final STT item normally lands at the end of its caller
# interval, but a small callback delay is expected.  Keep this window narrow:
# accepting a distant interval is worse than leaving one transcript item
# without a clip because it silently plays another utterance.
_CALLER_TRANSCRIPT_INTERVAL_SLACK_SECONDS = 1.0
# Per-engine caller transcript timing is DATA, one row per engine, beside the
# voice capability artifact (``capabilities/caller_transcript_timing.json``).
_CALLER_TRANSCRIPT_TIMING_PATH = (
    Path(__file__).resolve().parent / "capabilities" / "caller_transcript_timing.json"
)
_CALLER_TRANSCRIPT_TIMING_FIELDS = (
    "first_interim_min_latency_seconds",
    "first_interim_max_latency_seconds",
    "first_interim_lookback_seconds",
    "utterance_gap_seconds",
)
_AGENT_TRANSCRIPT_INTERVAL_SLACK_SECONDS = 1.0

# How a speech-to-speech recorder labels agent text. It is the engine's own
# transcript of what it said, handed over while its audio played -- not text
# bound before synthesis -- so it never borrows ``bound_before_playout``.
MODEL_OUTPUT_TRANSCRIPT = "model_output_transcript"
# A caller turn the engine never finalized because the call ended while it was
# still open; its text is the last interim, so it may stop mid-sentence.
FINALIZED_AT_CLOSE = "finalized_at_close"
# Agent-only clips a speech-to-speech bundle adds. Deliberately not
# ``turn-*-agent.wav``: those pair one-to-one with caller clips.
SPEECH_CLIP_PREFIX = "speech"
# Audible agent speeches with no text are listed with their spans; a pathological
# call must not grow the transcript without bound.
_MAX_LISTED_UNCAPTURED_SPEECHES = 50
# The caller's raw speaking intervals kept in the bundle. A 30-minute call
# with a pause every second stays under it.
_MAX_LISTED_CALLER_SPEECH_INTERVALS = 2_000

# An agent interval closes on the SDK's state-change edge, which fires when the
# last frame has been handed to the transport -- the source queue, the WebRTC
# encoder and the SFU still hold it.  On a production goodbye the room track
# carried 140ms more speech after that edge, so every agent clip cut at the edge
# lost the tail of its last syllable and any measurement of the clip's ending
# read as truncation whether or not the caller heard one.  The lag is buffering
# and varies call to call, so the clip's end is MEASURED against the captured
# room track instead of padded by a constant.
#
# The window is bounded so a missing quiet edge can never swallow the next
# utterance, and the numbers below come from that recording (8kHz, 10ms RMS
# frames): the goodbye plateaus at ~7,300 and decays 4077 -> 1495 -> 814 -> 227
# -> 45 -> 6 into a floor whose frames sit near zero.  A threshold of 500 puts
# the audible edge at the 227 frame -- about -43 dBFS, inaudible -- and any
# value between 100 and 1,000 moves that edge by at most 20ms, so the result
# does not balance on the constant.
#
# What ends the search is a GAP, not the first quiet frame.  A tail is speech
# and speech contains silence: a stop closure runs to about 100ms and the same
# recording has a 10ms frame at 119 RMS in the middle of a word, so a rule that
# stopped at the first quiet frames would cut the syllable it exists to
# recover.  120ms is longer than any gap inside one utterance and far shorter
# than the pause before another, and the tail is bounded to 500ms regardless --
# 3.5x the measured lag.
_PLAYOUT_TAIL_SEARCH_SECONDS = 0.5
_PLAYOUT_TAIL_FRAME_SECONDS = 0.010
_PLAYOUT_TAIL_QUIET_RMS = 500.0
# A pause between two delivered frames (either track) longer than this is a
# gap in what the room observer saw. Well above LiveKit's 10-20 ms frames and
# its batched delivery, well below the quiet any reader here waits for.
_OBSERVER_GAP_SECONDS = 0.3
_PLAYOUT_TAIL_GAP_SECONDS = 0.120

# The measurement point, stated once and precisely, because a counter whose
# origin has to be inferred is how a number gets attributed to the wrong leg.
#
# These come from the EVIDENCE SIDECAR's WebRTC transport: a second room
# participant subscribing to the same published caller track. They are not the
# agent session's own transport, and they are not the carrier-side RTP of the
# SIP leg -- LiveKit's SIP service owns those, on the far side of the SFU.
#
# So this measures what reached the recorder, which is the closest available
# proxy for what reached STT but is not identical to it: WebRTC loss and jitter
# are per subscriber connection, so the two can diverge. Read this as "the
# caller's published audio arrived, and this much of it", never as a per-leg
# quality verdict about STT or the PSTN.  Naming the source in every snapshot
# is what keeps a later reader from collapsing the three into one.
#
# ``rtp_receive_stats`` lost its producer when the retired pre-LiveKit media
# worker was deleted (#441): the key survived in the artifact parser and every
# imported call has carried ``{}`` since, which is what made a one-way call
# invisible in the artifact and got that empty field misread as the cause of a
# settlement bug it has nothing to do with. This restores the LiveKit channel.
RTP_RECEIVE_STATS_SOURCE = "livekit_room_sidecar_inbound_rtp"

# The mixed WAV is a listening derivative. Raw caller/agent tracks remain the
# immutable evidence. Gains are inferred independently from speech-bearing
# frames, bounded, then reduced together when their overlap needs headroom.
_LISTENING_MIX_FRAME_SECONDS = 0.020
_LISTENING_MIX_MIN_SPEECH_SECONDS = 0.100
_LISTENING_MIX_MIN_SPEECH_RMS = 300.0
_LISTENING_MIX_TARGET_SPEECH_RMS = 5_000.0
_LISTENING_MIX_MIN_GAIN_DB = -6.0
_LISTENING_MIX_MAX_GAIN_DB = 12.0
_LISTENING_MIX_PEAK = 0.85 * 32767
# The listening mix is computed and written this many samples at a time
# (128 KiB of output), so finalizing a long call holds one chunk of Python
# ints rather than the whole call's.
_LISTENING_MIX_CHUNK_SAMPLES = 65_536
# Padding appends at most this much at a time: ``b"\x00" * shortfall`` would
# allocate the whole shortfall as a temporary, which is a full track when one
# side of the call never delivered audio.
_SILENCE_BLOCK = bytes(64 * 1024)
# What ``wave`` writes ahead of the samples of a mono PCM16 file.
_WAV_HEADER_BYTES = 44


def _pad_with_silence(track: bytearray, length: int) -> None:
    """Zero-fill ``track`` up to ``length`` bytes, one bounded block at a time."""

    silence = memoryview(_SILENCE_BLOCK)
    while len(track) < length:
        track.extend(silence[: length - len(track)])


def _pcm16_samples(payload: bytes | bytearray | memoryview) -> memoryview:
    """A zero-copy int16 view of little-endian PCM16.

    The view reads in native byte order, which is little-endian on every host
    this runs on; a big-endian host is refused rather than mixed wrongly.
    """

    if sys.byteorder != "little":
        raise RuntimeError("the listening mix reads PCM16 on little-endian hosts only")
    return memoryview(payload).cast("B").cast("h")


def _frame_rms(payload: bytes | memoryview, start_sample: int, samples: int) -> float:
    """RMS of one mono PCM16 frame (``audioop`` was removed in Python 3.13).

    Called per 20 ms frame on the live path, where a frame is too small for an
    array round-trip to pay for itself.
    """

    frame = struct.unpack_from(f"<{samples}h", payload, start_sample * 2)
    return math.sqrt(sum(sample * sample for sample in frame) / samples)


def _inbound_rtp_member(entry: Any) -> Any:
    """The entry's ``inbound_rtp`` payload, or ``None`` if it holds another kind.

    ``RtcStats`` is a protobuf ``oneof``, so ``entry.inbound_rtp`` returns a
    fully-populated DEFAULT message for a codec/transport/outbound entry rather
    than ``None``. Reading it without asking which member is set turns every
    unrelated stats row into an inbound audio stream carrying zeros, which
    would publish non-empty all-zero evidence for a call that received nothing
    -- the exact confusion between "no evidence" and "evidence of zero" this
    module exists to prevent.
    """
    which = getattr(entry, "WhichOneof", None)
    if callable(which):
        try:
            if which("stats") != "inbound_rtp":
                return None
        except (TypeError, ValueError):
            return None
        return getattr(entry, "inbound_rtp", None)
    # A plain object (a test stand-in) has no oneof; absence is expressed by
    # the attribute being missing or None.
    return getattr(entry, "inbound_rtp", None)


def merged_receive_stats(snapshots: Any) -> dict[str, Any]:
    """Combine per-publication snapshots into one artifact figure.

    Caller audio can be republished after a reconnect, and the RTP counters
    restart at zero for each concrete track. Keeping only the newest snapshot
    would silently drop every packet, loss and concealment count from the
    earlier publication, so each publication's latest cumulative snapshot is
    retained separately and summed here. Jitter is a rate and does not sum:
    the worst publication is the one that degraded the call.
    """
    totals = {
        "streams": 0,
        "packets_received": 0,
        "packets_lost": 0,
        "payload_bytes": 0,
        "samples_received": 0,
        "concealed_samples": 0,
    }
    jitter_ms = 0.0
    seen = False
    for snapshot in (snapshots or {}).values():
        if not snapshot:
            continue
        seen = True
        for key in totals:
            totals[key] += int(snapshot.get(key) or 0)
        jitter_ms = max(jitter_ms, float(snapshot.get("jitter_ms") or 0.0))
    if not seen:
        return {}
    return {"source": RTP_RECEIVE_STATS_SOURCE, **totals, "jitter_ms": jitter_ms}


def inbound_rtp_receive_stats(stats: Any) -> dict[str, Any]:
    """Flatten LiveKit ``RtcStats`` into the artifact's receive-stats shape.

    Returns ``{}`` when no inbound audio stream is present, so an empty dict
    keeps meaning "no evidence" rather than "evidence of zero". Reads by
    attribute path only: nothing here needs the SDK's protobuf types, which
    the control-plane test environment does not install.
    """
    packets_received = 0
    packets_lost = 0
    payload_bytes = 0
    samples_received = 0
    concealed_samples = 0
    jitter_seconds = 0.0
    streams = 0
    for entry in stats or ():
        inbound = _inbound_rtp_member(entry)
        if inbound is None:
            continue
        stream = getattr(inbound, "stream", None)
        # ``kind`` is the RTP media kind string ("audio"/"video"). A worker
        # publishes no video, but filtering keeps this honest if one appears.
        if str(getattr(stream, "kind", "audio") or "audio") != "audio":
            continue
        received = getattr(inbound, "received", None)
        detail = getattr(inbound, "inbound", None)
        if received is None and detail is None:
            continue
        streams += 1
        packets_received += int(getattr(received, "packets_received", 0) or 0)
        packets_lost += int(getattr(received, "packets_lost", 0) or 0)
        payload_bytes += int(getattr(detail, "bytes_received", 0) or 0)
        samples_received += int(getattr(detail, "total_samples_received", 0) or 0)
        concealed_samples += int(getattr(detail, "concealed_samples", 0) or 0)
        # Jitter is per-stream and does not sum. The worst stream is the one
        # that would have degraded the call, so keep the maximum.
        jitter_seconds = max(jitter_seconds, float(getattr(received, "jitter", 0.0) or 0.0))
    if not streams:
        return {}
    return {
        "source": RTP_RECEIVE_STATS_SOURCE,
        "streams": streams,
        "packets_received": packets_received,
        "packets_lost": packets_lost,
        "payload_bytes": payload_bytes,
        "samples_received": samples_received,
        "concealed_samples": concealed_samples,
        "jitter_ms": round(jitter_seconds * 1000.0, 3),
    }


@dataclass(frozen=True)
class SpeechInterval:
    """One speaking interval, with its SDK edge and its room-side audio edge.

    ``end`` is the state-change edge: the SDK reports it when the last frame
    has been handed to the transport, which is BEFORE that frame reaches the
    room.  ``audio_end`` is where the captured room track actually goes quiet
    and is set only for agent intervals, only during finalization, and only
    when the recorder observed audio past the edge.  Every attribution rule
    (transcript joins, goodbye containment, barge-in overlap) keeps reading
    ``end``, so the semantics of an interval are unchanged; the clip and the
    boundary reported alongside it read ``playout_end``.
    """

    start: float
    end: float
    audio_end: float | None = None
    #: The SpeechHandle whose first frame opened this interval, when the worker
    #: could name it. Read only by a speech-to-speech recorder.
    speech_id: str = ""

    @property
    def playout_end(self) -> float:
        """Where the room stopped hearing this speaker."""

        if self.audio_end is None or self.audio_end <= self.end:
            return self.end
        return self.audio_end

    @property
    def tail_seconds(self) -> float:
        """How far the observed audio ran past the SDK edge."""

        return max(0.0, self.playout_end - self.end)


def _positive_seconds_or_none(value: object) -> float | None:
    """A finite positive number of seconds, or ``None`` for anything else."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _finite_or_none(value: object) -> float | None:
    """A finite number, or ``None`` for anything else."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit_interval_or_none(value: object) -> float | None:
    """A finite number in ``[0, 1]``, or ``None`` for anything else."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


@dataclass(frozen=True)
class CallerTranscriptTiming:
    """How an engine's caller transcripts are timed against caller speech.

    Only an interval that opened between ``first_interim_min_latency`` and
    ``first_interim_max_latency`` before an item's first interim can be the
    speech that interim describes; intervals closer than ``utterance_gap``
    are one utterance for widening a clip. Read from the per-engine data
    file, never branched on by engine in code.
    """

    first_interim_min_latency: float
    first_interim_max_latency: float
    #: How far back, past ``first_interim_max_latency``, an item's own speech
    #: can still have begun when its first interim came unusually late. An
    #: unclaimed interval that opened in that band leaves the item ambiguous:
    #: it may be the item's own speech, making the one in the window the next
    #: utterance's.
    first_interim_lookback: float
    utterance_gap: float


@functools.lru_cache(maxsize=1)
def _caller_transcript_timing_rows() -> Mapping[str, CallerTranscriptTiming]:
    document = json.loads(_CALLER_TRANSCRIPT_TIMING_PATH.read_text(encoding="utf-8"))
    if document.get("schema") != "skysay.caller_transcript_timing.v1":
        raise ValueError("caller transcript timing: unknown schema")
    rows: dict[str, CallerTranscriptTiming] = {}
    for engine, row in dict(document.get("engines") or {}).items():
        values = [_finite_or_none(row.get(field)) for field in _CALLER_TRANSCRIPT_TIMING_FIELDS]
        if any(value is None or value <= 0 for value in values):
            raise ValueError(f"caller transcript timing: {engine} has a missing or bad value")
        minimum, maximum, lookback, gap = values  # type: ignore[misc]
        if not minimum < maximum <= lookback:
            raise ValueError(f"caller transcript timing: {engine} window is empty or inverted")
        rows[str(engine)] = CallerTranscriptTiming(minimum, maximum, lookback, gap)
    return rows


def caller_transcript_timing(engine: str) -> CallerTranscriptTiming | None:
    """The engine's timing row, or ``None`` when the data has none for it."""

    return _caller_transcript_timing_rows().get(str(engine or "").strip().lower())


@dataclass(frozen=True)
class TranscriptItem:
    role: Track
    text: str
    at: float
    #: What the recogniser measured about this utterance, when it measured
    #: anything. ``None`` is "not measured" and must never be rendered as a
    #: number: an invented duration is what made 55 of one batch's 86 caller
    #: turns report an identical length, and an invented confidence is what
    #: made every turn ever imported report 0.85.
    measured_duration: float | None = None
    confidence: float | None = None
    caption_ingested_at: float | None = None
    #: A caller turn flushed from its last interim at close; see
    #: ``FINALIZED_AT_CLOSE``.
    finalized_at_close: bool = False
    #: When a speech-to-speech engine's FIRST interim for this caller item
    #: arrived, which is while the caller is still saying it. ``at`` is when
    #: the final arrived, which these engines send late (see
    #: ``_caller_join_anchor``). Read only to join the item to its caller
    #: interval; every other rule keeps reading ``at``.
    speech_started_at: float | None = None


@dataclass(frozen=True)
class TerminalMessage:
    """A worker-owned farewell, with whether its playout was heard in full.

    The end_call goodbye is spoken via ``session.say`` and recorded by the
    worker after it waits for playout.  ``interrupted`` carries the outcome the
    worker observed: a farewell whose playout completed can stand as the turn's
    verbatim spoken response, while one that timed out or failed is recorded as
    the turn's text but never claimed as fully heard.
    """

    role: Track
    text: str
    at: float
    interrupted: bool = False
    # The worker-declared SpeechHandle id of the goodbye, when the farewell
    # produced audio.  Empty for a legacy bundle or a synthesis that never
    # returned a handle.  It ties the terminal message to the preplayout
    # candidate the worker bound at say-time, so attribution is by declared
    # identity rather than inferred from timing.
    speech_id: str = ""
    # Whether the room went quiet after the SDK reported the goodbye played
    # out, i.e. the last frames were observed leaving the room rather than
    # assumed.  ``None`` when the worker had no room-side observer to ask (the
    # legacy rule then stands); ``False`` when the drain ended without seeing
    # quiet -- the room dropped, the bound elapsed, no agent audio was ever
    # captured -- and the farewell must not be certified as heard in full.
    egress_confirmed: bool | None = None
    egress_signal: str = ""


@dataclass
class PreplayoutAgentSpeech:
    """Text admitted to TTS before a LiveKit reply can be committed.

    LiveKit 1.7.1 appends its assistant conversation item only after the
    reply task has completed.  A disconnect can cancel that final commit even
    though the sidecar already observed the response audio.  This is bounded
    in-memory evidence for that earlier, authoritative hand-off to TTS; it is
    only written into a customer transcript after an observed agent interval
    is linked during finalization.
    """

    speech_id: str
    at: float
    text_parts: list[str]
    completed_at: float | None = None
    interrupted: bool = False
    worker_owned: bool = False

    @property
    def text(self) -> str:
        return " ".join("".join(self.text_parts).split())


def _caller_text_status(item: TranscriptItem) -> str:
    return FINALIZED_AT_CLOSE if item.finalized_at_close else "recorded"


def _ignored_once_sealed(method: Callable[..., Any]) -> Callable[..., Any]:
    """Make an evidence mutator a no-op once the recorder is sealed.

    ``finalize`` used to run on the event loop, so nothing could touch the
    recorder while it read. Off the loop, a late state edge, caption or
    feedback event would change what the writer thread is iterating. What
    such a call did after a finalize was already invisible to the bundle, so
    ignoring it once sealed changes nothing that was ever written.
    """

    @functools.wraps(method)
    def guarded(self: "SynchronizedEvidenceRecorder", *args: Any, **kwargs: Any) -> Any:
        if self._sealed:
            return None
        return method(self, *args, **kwargs)

    guarded.ignored_once_sealed = True  # type: ignore[attr-defined]
    return guarded


class NoRoomAudioCaptured(RuntimeError):
    """The room ended before either synchronized track received audio."""


class _DeliveryBacklogWitness:
    """Bounded evidence of a PCM backlog, not a reconstructed media clock.

    A 500ms delivery window containing over 250ms more audio than elapsed
    time proves frames were queued. Regular delivery and genuine no-frame
    gaps reset the window. Once witnessed, callback-based boundaries for
    either speaker may be late, so the entire call stays ambiguous.
    """

    def __init__(self) -> None:
        self._windows: dict[Track, tuple[object, float, float, float]] = {}
        self.detected = False

    def observe(self, track: Track, stream_id: object, duration: float, at: float) -> None:
        if self.detected:
            return
        previous = self._windows.get(track)
        if previous is None or previous[0] != stream_id or at - previous[1] > 0.5:
            self._windows[track] = (stream_id, at, duration, at)
            return
        identity, start, audio_seconds, last = previous
        audio_seconds += duration
        self._windows[track] = (identity, start, audio_seconds, at)
        if at < last or audio_seconds - duration - (at - start) > 0.25:
            self.detected = True
            self._windows.clear()


class SynchronizedEvidenceRecorder:
    """Collects two real-time PCM tracks on one monotonic timeline.

    A separate LiveKit room participant feeds this recorder, so agent audio is
    captured after playout/interruptions rather than when TTS merely generated
    it.  Both tracks are padded to the same duration and per-turn clips are
    sliced from observed caller/agent speaking intervals.
    """

    def __init__(
        self,
        root: Path,
        *,
        sample_rate: int = 8_000,
        epoch: float | None = None,
        transcript_metadata: Mapping[str, Any] | None = None,
        speech_to_speech: bool = False,
        caller_transcript_timing: CallerTranscriptTiming | None = None,
    ) -> None:
        self.root = root
        # A speech-to-speech engine speaks over the caller and has no TTS
        # node, so its text is joined to its audio by speech identity and
        # every agent speech is written, paired with a caller turn or not.
        # Cascade also preserves additional text-backed speeches, while its
        # caller timing, paired-turn metrics and unplayed-text policy differ.
        self._speech_to_speech = bool(speech_to_speech)
        # The engine's caller transcript timing row. With one, caller items
        # join on their first interim; a voice-to-voice call without one gets
        # no caller clips, because its finals arrive too late to join on.
        self._caller_transcript_timing = caller_transcript_timing
        self._active_speech_id: dict[Track, str] = {"caller": "", "agent": ""}
        # The engine-reported liveness record for this call; see
        # ``set_duplex_liveness``.
        self._duplex_liveness: dict[str, object] | None = None
        self._agent_speech_coverage: dict[str, object] | None = None
        self.sample_rate = sample_rate
        self.epoch = time.monotonic() if epoch is None else float(epoch)
        self._delivery_backlog = _DeliveryBacklogWitness()
        self._tracks: dict[Track, bytearray] = {
            "caller": bytearray(),
            "agent": bytearray(),
        }
        # A LiveKit AudioStream can deliver several already-buffered frames in
        # one event-loop turn. Callback time anchors a subscription, but it is
        # not a media timestamp for every frame drained from that subscription.
        self._stream_end_samples: dict[tuple[Track, object], int] = {}
        self._agent_audible_until: float | None = None
        # Seconds of agent audio above the quiet threshold, summed frame by
        # frame: a duration of SOUND, where the edge above is only a moment.
        self._agent_audible_seconds = 0.0
        self._observer_last_frame_at: float | None = None
        # When frames last resumed after a delivery gap, or ``None``.
        self._observer_resumed_at: float | None = None
        self._filler_acknowledgements_enabled = False
        # ``voice_polish.tool_wait`` as resolved for this call, and what it did:
        # every start and cut of the typing bed or phrase, on the same clock as
        # the intervals, so a reviewer can place what the caller heard.
        self._tool_wait_feedback_mode = "off"
        self._tool_wait_feedback_events: list[dict[str, object]] = []
        # The voicemail detector's frozen record, set once before
        # finalization. Absent from transcript.json when the detector did not
        # run, which the importer reads as "not recorded", never as "no
        # voicemail".
        self._voicemail_detection: dict[str, object] | None = None
        self._active: dict[Track, float | None] = {"caller": None, "agent": None}
        self._intervals: dict[Track, list[SpeechInterval]] = {
            "caller": [],
            "agent": [],
        }
        self._texts: list[TranscriptItem] = []
        # Keyed by the LiveKit SpeechHandle id.  A handle may call ``tts_node``
        # more than once while streaming a reply, so chunks deliberately
        # accumulate into one text candidate rather than becoming duplicate
        # transcript turns.
        self._preplayout_agent_speech: dict[str, PreplayoutAgentSpeech] = {}
        self._terminal_messages: list[TerminalMessage] = []
        self._termination_started_at: float | None = None
        self._turns: list[dict[str, object]] = []
        # Caller speech can be observed and transcribed even when the TTS
        # provider never starts a reply.  Keep that caller-side evidence
        # separately from paired semantic turns: inventing an agent turn
        # would overstate what happened, while dropping the caller would make
        # a provider failure indistinguishable from no transcript at all.
        self._unpaired_caller_turns: list[
            tuple[int, TranscriptItem, SpeechInterval, str]
        ] = []
        self._unmatched_caller_turns: list[tuple[int, TranscriptItem]] = []
        # The typed tool-call / language-switch section, supplied by
        # ``call_transcript_events.CallTranscriptEventRecorder.section`` and
        # attached AFTER construction because that recorder is anchored to
        # this one's epoch. A callable rather than a value: the rows accrue
        # for the whole call and are read once, here, at finalization.
        self._closing_evidence_source: Callable[[], Mapping[str, Any]] | None = None
        self._closing_evidence_frozen: dict[str, Any] | None = None
        self._transcript_events_source: Callable[[], Mapping[str, Any]] | None = None
        self._transcript_metadata = dict(transcript_metadata or {})
        if {
            "turns",
            "summary",
            "semantic_response_audio_policy",
            "rtp_receive_stats",
            "tool_wait_feedback",
            "transcript_events",
            "closing_evidence",
            "voicemail_detection",
            "agent_speech_coverage",
            "duplex_liveness",
        } & set(self._transcript_metadata):
            raise ValueError("transcript metadata cannot replace PR #79 evidence fields")
        # Latest inbound-RTP snapshot PER caller publication, sampled while
        # that track is still subscribed. Empty until one is recorded: a call
        # that never carried media must not publish counters that read as a
        # measured zero. Keyed because counters restart on a republish.
        self._media_receive_stats: dict[str, dict[str, Any]] = {}
        self._finalized = False
        # Value mode's redactor (``navigation_values.Redactor``), when the call
        # can key a value on a phone menu; ``None`` otherwise, and then nothing
        # below changes.
        self._redactor: Any = None
        self._originals: dict[int, tuple[str, str]] = {}
        # Languages fixed when a preplayout speech or a goodbye was bound.
        self._speech_languages: dict[str, str] = {}
        self._terminal_languages: dict[int, str] = {}
        # Set on the event loop by ``finalize_off_loop`` before the writer
        # thread starts: from then on nothing may change what it reads.
        self._sealed = False
        # Why finalization failed once it had started writing, as text: the
        # exception itself would keep its frames, and the track views in
        # them, alive for as long as the recorder.
        self._finalize_failure: str | None = None

    def record_media_receive_stats(
        self, stats: Mapping[str, Any] | None, *, source: str = "default"
    ) -> None:
        """Retain the newest non-empty inbound-RTP snapshot for one publication.

        Counters are cumulative WITHIN a publication, so a later sample from
        the same ``source`` supersedes its predecessor; a different ``source``
        is a separate publication whose counters started again at zero and is
        kept alongside. An empty sample is ignored rather than stored: the
        caller track can be unsubscribed by the time a final read is attempted,
        and losing a real mid-call measurement to that race is exactly the
        blindness this restores. Writing after finalization is refused, like
        every other piece of evidence.
        """
        if self._finalized or self._sealed:
            raise RuntimeError("cannot record media stats after evidence finalization")
        if stats:
            self._media_receive_stats[str(source)] = dict(stats)

    def append_pcm(
        self,
        track: Track,
        pcm16: bytes,
        *,
        received_at: float | None = None,
    ) -> None:
        if self._finalized or self._sealed:
            raise RuntimeError("cannot append after evidence finalization")
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("PCM must be non-empty signed 16-bit mono")
        received = time.monotonic() if received_at is None else float(received_at)
        samples = len(pcm16) // 2
        end_sample = max(0, round((received - self.epoch) * self.sample_rate))
        start_sample = max(0, end_sample - samples)
        start_byte = start_sample * 2
        end_byte = start_byte + len(pcm16)
        target = self._tracks[track]
        if len(target) < end_byte:
            target.extend(b"\x00" * (end_byte - len(target)))
        target[start_byte:end_byte] = pcm16
        self._observe_live_agent_audio(track, pcm16, samples, received)

    def _observe_live_agent_audio(
        self, track: Track, pcm16: bytes, samples: int, received: float
    ) -> None:
        """Keep the live edge of the agent's room audio, frame by frame.

        Finalization measures where each agent clip went quiet after the fact.
        The terminal drain needs the same answer WHILE the call is still open:
        has the room stopped hearing the agent since the SDK said the goodbye
        played out?  This records only two timestamps and reuses the finalize
        quiet threshold, so the live and the post-hoc answers cannot disagree
        about what counts as audible.
        """

        previous = self._observer_last_frame_at
        if previous is not None and received - previous > _OBSERVER_GAP_SECONDS:
            # Marked when the frame ARRIVES: a reader that polls would miss a
            # gap that closed between two of its looks.
            self._observer_resumed_at = received
        self._observer_last_frame_at = received
        if track != "agent":
            return
        if _frame_rms(pcm16, 0, samples) >= _PLAYOUT_TAIL_QUIET_RMS:
            self._agent_audible_until = received
            self._agent_audible_seconds += samples / float(self.sample_rate)

    def observer_resumed_at(self) -> float | None:
        """When delivery last resumed after a gap longer than
        :data:`_OBSERVER_GAP_SECONDS`: nothing before it was watched
        continuously up to now."""

        return self._observer_resumed_at

    def agent_audible_seconds(self) -> float:
        """Seconds of audible agent audio the room has received so far."""

        return self._agent_audible_seconds

    def agent_audible_until(self) -> float | None:
        """When the room last heard audible agent audio, or ``None`` if never."""

        return self._agent_audible_until

    def observer_last_frame_at(self) -> float | None:
        """When the sidecar last delivered ANY frame, on either track.

        The caller's leg publishes continuously while connected, so a sidecar
        that has stopped delivering frames has stopped observing -- and a
        frozen agent edge then means "unobserved", not "quiet".
        """

        return self._observer_last_frame_at

    def append_stream_pcm(
        self,
        track: Track,
        pcm16: bytes,
        *,
        stream_id: object,
        received_at: float | None = None,
    ) -> tuple[float, bool]:
        """Append ordered PCM without turning scheduler jitter into bad audio.

        Observation time places each frame on the shared timeline while the
        stream cursor prevents already-buffered frames from overwriting one
        another. If a participant stops emitting frames, observation time can
        still advance the cursor and preserve that real silence. A republished
        track receives a new ``stream_id`` and is anchored again.

        Returns where the frame was placed: its start on the monotonic clock
        this recorder's epoch is on, and whether it does NOT follow the
        previous frame of the same stream contiguously (a first frame, a
        re-anchor, or observation time jumping past the cursor).  The voicemail
        detector measures tone durations and quiet on that placement rather
        than on arrival time, because frames queued behind a slow iteration
        arrive together.
        """

        if self._finalized or self._sealed:
            raise RuntimeError("cannot append after evidence finalization")
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("PCM must be non-empty signed 16-bit mono")
        try:
            key = (track, stream_id)
            previous_end = self._stream_end_samples.get(key)
        except TypeError as exc:
            raise ValueError("stream_id must be hashable") from exc
        samples = len(pcm16) // 2
        received = time.monotonic() if received_at is None else float(received_at)
        self._delivery_backlog.observe(track, stream_id, samples / self.sample_rate, received)
        observed_end = max(0, round((received - self.epoch) * self.sample_rate))
        observed_start = max(0, observed_end - samples)
        start_sample = (
            observed_start
            if previous_end is None
            else max(previous_end, observed_start)
        )
        end_sample = start_sample + samples
        start_byte = start_sample * 2
        end_byte = start_byte + len(pcm16)
        target = self._tracks[track]
        if len(target) < end_byte:
            target.extend(b"\x00" * (end_byte - len(target)))
        target[start_byte:end_byte] = pcm16
        self._stream_end_samples[key] = end_sample
        self._observe_live_agent_audio(track, pcm16, samples, received)
        # A jump of a few samples is scheduler jitter the cursor absorbed, not
        # lost media; only a jump past half a frame (10 ms) is a gap.
        discontinuous = previous_end is None or start_sample - previous_end > self.sample_rate // 100
        return self.epoch + start_sample / self.sample_rate, discontinuous

    @_ignored_once_sealed
    def mark_state(
        self,
        track: Track,
        old_state: str,
        new_state: str,
        *,
        at: float | None = None,
        speech_id: str | None = None,
    ) -> None:
        if track not in self._tracks:
            raise ValueError("track must be caller or agent")
        when = time.monotonic() if at is None else float(at)
        if new_state == "speaking" and self._active[track] is None:
            self._active[track] = when
            self._active_speech_id[track] = str(speech_id or "").strip()
        if old_state == "speaking" and new_state != "speaking":
            started = self._active[track]
            if started is not None and when > started:
                self._intervals[track].append(
                    SpeechInterval(started, when, speech_id=self._active_speech_id[track])
                )
            self._active[track] = None
            self._active_speech_id[track] = ""

    @_ignored_once_sealed
    def add_text(
        self,
        role: Track,
        text: str,
        *,
        at: float | None = None,
        measured_duration: float | None = None,
        confidence: float | None = None,
        caption_ingested_at: float | None = None,
        finalized_at_close: bool = False,
        speech_started_at: float | None = None,
        language: str = "",
    ) -> None:
        normalized = " ".join(str(text or "").split())
        original = ""
        if normalized and self._redactor is not None:
            # Kept (in memory only) for the terminal pass, which must see the
            # whole text: the delivery below may already have masked part of
            # a value that began in this text and ends in the next. A text
            # delivered in value mode is already free of numerals.
            try:
                in_value_mode = bool(self._redactor.value_mode)
            except Exception:  # noqa: BLE001 - read as in value mode: no original is kept
                in_value_mode = True
            # The far end's text in value mode has every numeral replaced, so
            # its original is not needed (nor kept); the agent's own text is
            # masked against the value only, so its original is.
            if not in_value_mode or role != "caller":
                original = normalized
            judged_in = self._effective_language(language)
            normalized = self._deliver_to_artifact(role, normalized, language)
        if normalized:
            if original:
                self._originals[len(self._texts)] = (original, judged_in)
            self._texts.append(
                TranscriptItem(
                    role=role,
                    text=normalized,
                    at=time.monotonic() if at is None else float(at),
                    measured_duration=_positive_seconds_or_none(measured_duration),
                    confidence=_unit_interval_or_none(confidence),
                    caption_ingested_at=caption_ingested_at,
                    finalized_at_close=bool(finalized_at_close),
                    speech_started_at=_finite_or_none(speech_started_at),
                )
            )

    @_ignored_once_sealed
    def set_redactor(self, redactor: Any) -> None:
        """Route every text this recording keeps through value mode's redactor.

        Each text is delivered to the transcript artifact as it is added, and
        at finalization the whole call's texts are masked once more in order,
        so digits a person read out turn by turn after the conversation
        resumed are caught across adjacent turns. Read-only to the evidence:
        a redactor holds no audio."""
        self._redactor = redactor

    def _note_speech_language(self, identifier: str) -> None:
        """Fix the language a bound speech is judged in when it is first bound."""
        if self._redactor is not None:
            self._speech_languages.setdefault(identifier, self._effective_language())

    def _effective_language(self, tag: str = "") -> str:
        """The language a text is judged in, fixed when it is added: a later
        switch of the call's language must not re-judge it."""
        try:
            return str(self._redactor.effective_language(tag) or "")
        except Exception:  # noqa: BLE001 - an unreadable language is the empty one
            return str(tag or "")

    def _deliver_to_artifact(self, role: Track, text: str, language: str = "") -> str:
        from_far_end = role == "caller"
        try:
            delivered = self._redactor.deliver(
                Consumer.TRANSCRIPT_ARTIFACT, FarEnd(text, language, final=True) if from_far_end else Own(text, language)
            )
        except Exception:  # noqa: BLE001 - a mask that fails keeps the fact of the text, never its content
            return WITHHELD_TEXT
        return " ".join(str(delivered or "").split())

    def _remask_whole_call(self) -> None:
        """The terminal pass: every kept text, in the order it happened."""
        if self._redactor is None:
            return
        slots: list[tuple[float, str, int, str, str]] = []
        for index, item in enumerate(self._texts):
            original, language = self._originals.get(index, (item.text, ""))
            slots.append((item.at, "item", index, original, language))
        for identifier, candidate in self._preplayout_agent_speech.items():
            if candidate.text:
                slots.append((candidate.at, "speech", 0, identifier, self._speech_languages.get(identifier, "")))
        for index, message in enumerate(self._terminal_messages):
            slots.append((message.at, "terminal", index, message.text, self._terminal_languages.get(index, "")))
        # Three copies of an agent's words can exist (the committed item, the
        # speech bound before playout, the worker's goodbye) and they cannot be
        # paired with certainty, so a text is masked wherever ANY view of it finds
        # a held value: its own kind in order, and the agent's words as one
        # chronological sequence. And one speaker's words are not inside the
        # other's run: a person reading a value out with the agent saying "go on"
        # between the parts.
        def is_caller(slot: tuple[float, str, int, str, str]) -> bool:
            return slot[1] == "item" and self._texts[slot[2]].role == "caller"

        def text_of(slot: tuple[float, str, int, str, str]) -> str:
            return self._preplayout_agent_speech[slot[3]].text if slot[1] == "speech" else slot[3]

        views = (
            [slot for slot in slots if is_caller(slot)],
            [slot for slot in slots if slot[1] == "item" and not is_caller(slot)],
            [slot for slot in slots if slot[1] in ("speech", "terminal")],
            [slot for slot in slots if not is_caller(slot)],
        )
        ranges: dict[tuple[str, int, str], list[tuple[int, int]]] = {}
        try:
            for view in views:
                group = sorted(view, key=lambda slot: slot[0])
                found = self._redactor.remask_ranges([text_of(slot) for slot in group])
                if len(found) != len(group):
                    raise ValueError("the terminal pass changed the number of texts")
                for slot, spans in zip(group, found):
                    ranges.setdefault((slot[1], slot[2], slot[3]), []).extend(spans)
            # A speech or a goodbye that says what a committed item says is that
            # item's copy: what is found in either is found in both (the nearest
            # in time stands for the pair; a wrong pairing can only mask more).
            def name(slot: tuple[float, str, int, str, str]) -> tuple[str, int, str]:
                return (slot[1], slot[2], slot[3])

            agent_items = [slot for slot in slots if slot[1] == "item" and not is_caller(slot)]
            for slot in slots:
                if slot[1] in ("speech", "terminal"):
                    twins = [item for item in agent_items if text_of(item) == text_of(slot)]
                    if twins:
                        twin = min(twins, key=lambda item: abs(item[0] - slot[0]))
                        united = ranges.get(name(slot), []) + ranges.get(name(twin), [])
                        ranges[name(slot)] = united
                        ranges[name(twin)] = united
            masked = {name(slot): self._redactor.replace(text_of(slot), ranges.get(name(slot), [])) for slot in slots}
        except Exception:  # noqa: BLE001 - never write what could not be masked, never lose the recording for it
            masked = {(slot[1], slot[2], slot[3]): WITHHELD_TEXT for slot in slots}
        for _at, kind, index, key, _language in slots:
            text = masked[(kind, index, key)]
            if kind == "item":
                self._texts[index] = replace(self._texts[index], text=text)
            elif kind == "speech":
                self._preplayout_agent_speech[key].text_parts = [text]
            else:
                self._terminal_messages[index] = replace(self._terminal_messages[index], text=text)

    def speech_identity(self, handle: Any) -> str:
        """A cascade tool follow-up is a new generation on the same SDK handle.

        Keep the original ID for first-step speech (including worker say
        handles), and scope later cascade generations by the SDK step counter.
        Pausing/resuming does not increment that counter. Duplex IDs retain
        their existing contract.
        """
        identifier = str(getattr(handle, "id", "") or "").strip()
        step = getattr(handle, "num_steps", 1)
        if identifier and not self._speech_to_speech and type(step) is int and step > 1:
            return f"{identifier}:step:{step}"
        return identifier

    def bind_preplayout_agent_text(
        self, speech_id: object, text: str, *, at: float | None = None
    ) -> None:
        """Record assistant text immediately before it is yielded to TTS.

        This method is intentionally not an assertion that text was heard.
        ``finalize`` only uses a candidate when room-side audio was observed
        for it.  Keeping that distinction prevents a provider failure or an
        interrupted synthesis from manufacturing a transcript turn.
        """

        # A media reply task can finish unwinding after the worker has already
        # written the immutable evidence bundle.  That late chunk cannot be
        # represented faithfully, but it must never turn an observability
        # boundary into a TTS/task failure during teardown.
        if self._finalized or self._sealed:
            return
        identifier = str(speech_id or "").strip()
        if not identifier or not isinstance(text, str) or not text:
            return
        when = time.monotonic() if at is None else float(at)
        candidate = self._preplayout_agent_speech.get(identifier)
        if candidate is None:
            candidate = PreplayoutAgentSpeech(
                speech_id=identifier,
                at=when,
                text_parts=[],
            )
            self._preplayout_agent_speech[identifier] = candidate
            self._note_speech_language(identifier)
        candidate.text_parts.append(text)

    @_ignored_once_sealed
    def register_worker_goodbye(self, speech_id: str) -> None:
        """Reserve the say handle before TTS; only its own audio may name it.

        Ownership lives on the existing per-speech candidate, including an
        empty candidate. Reclaim and late TTS chunks cannot turn a worker's
        queued, unheard farewell into a generic time-window candidate.
        """
        identifier = str(speech_id or "").strip()
        if identifier:
            candidate = self._preplayout_agent_speech.setdefault(
                identifier, PreplayoutAgentSpeech(identifier, time.monotonic(), []))
            candidate.worker_owned = True
            self._note_speech_language(identifier)

    @_ignored_once_sealed
    def bind_worker_goodbye_text(self, speech_id: str, text: str, *, at: float) -> None:
        """Re-anchor known worker-owned text at its observed playout start.

        A reclaimable goodbye passes through tts_node while input is open, so
        it can already have a pre-TTS candidate. Appending the full sentence
        again duplicates it and leaves the earlier timestamp outside the
        declared goodbye's strict containment join. Only the worker owning
        this speech may replace that candidate, after observing its own edge.
        """
        identifier = str(speech_id or "").strip()
        if not identifier or not isinstance(text, str) or not text:
            return
        candidate = self._preplayout_agent_speech.get(identifier)
        if candidate is None:
            self.bind_preplayout_agent_text(identifier, text, at=at)
        else:
            candidate.at = float(at)
            candidate.text_parts = [text]
            self._note_speech_language(identifier)

    @_ignored_once_sealed
    def discard_unanchored_worker_goodbye(self, speech_id: str) -> None:
        """A terminal owner lacking its own playout edge must use the fallback.

        Pre-TTS synthesis can begin inside a different speech's interval. Its
        timestamp is not evidence that the interval belongs to this goodbye.
        Clear its text but retain ownership against late TTS chunks. The
        terminal text is retained by add_terminal_message and all other
        utterances remain untouched.
        """
        self.register_worker_goodbye(speech_id)
        candidate = self._preplayout_agent_speech.get(str(speech_id or "").strip())
        if candidate is not None:
            candidate.text_parts.clear()

    @_ignored_once_sealed
    def complete_preplayout_agent_speech(
        self,
        speech_id: object,
        *,
        interrupted: bool,
        at: float | None = None,
    ) -> None:
        """Retain the handle's interruption state without exposing text logs."""

        candidate = self._preplayout_agent_speech.get(str(speech_id or "").strip())
        if candidate is None:
            return
        candidate.interrupted = candidate.interrupted or bool(interrupted)
        candidate.completed_at = time.monotonic() if at is None else float(at)

    @_ignored_once_sealed
    def mark_acknowledgements_enabled(self) -> None:
        """Record that the call may have played fillers on the background track.

        Those fillers are never on the recorded agent track, so nothing else in
        the bundle changes; the policy statement stops claiming they were off.
        """

        self._filler_acknowledgements_enabled = True

    @_ignored_once_sealed
    def set_voicemail_detection(self, section: Mapping[str, Any] | None) -> None:
        """Attach the voicemail detector's frozen record, once.

        Set before ``finalize`` writes transcript.json; a second call is
        ignored, because the record is a single freeze of one call.
        """

        if self._voicemail_detection is not None or not isinstance(section, Mapping):
            return
        self._voicemail_detection = dict(section)

    @_ignored_once_sealed
    def set_duplex_liveness(self, section: Mapping[str, Any] | None) -> None:
        """Attach the engine liveness witnesses' per-call record, once.

        Bounded counters and timestamps only (``duplex_liveness``): when this
        worker last handed caller audio to the provider's socket and when the
        provider last sent anything back. Set before ``finalize``.
        """

        if self._duplex_liveness is not None or not isinstance(section, Mapping):
            return
        self._duplex_liveness = dict(section)

    @_ignored_once_sealed
    def mark_tool_wait_feedback(self, mode: str) -> None:
        """Record which tool-wait feedback the call was configured to play.

        Like acknowledgements it rides the unrecorded background track, so the
        agent track keeps its shape; the bundle states the mode so a reviewer
        knows a typing bed or a phrase may have covered a tool call.
        """

        self._tool_wait_feedback_mode = str(mode or "off")

    @_ignored_once_sealed
    def note_tool_wait_feedback(
        self, event: str, *, at: float | None = None, **details: object
    ) -> None:
        """Append one tool-wait feedback event (started, cut, skipped, failed)."""

        when = time.monotonic() if at is None else float(at)
        record: dict[str, object] = {
            "event": str(event),
            "at_ms": int(round((when - self.epoch) * 1000.0)),
        }
        for key, value in details.items():
            if isinstance(value, (list, tuple)):
                record[str(key)] = [str(item) for item in value]
            elif value is not None:
                record[str(key)] = str(value)
        self._tool_wait_feedback_events.append(record)

    @_ignored_once_sealed
    def mark_termination(self, *, at: float | None = None) -> None:
        """Record when worker-owned termination began (the end-call action).

        Terminal-message attribution requires the paired agent interval to
        START at or after this boundary: the goodbye is scheduled only after
        the termination action runs, so an interval beginning earlier also
        contains model reply speech whose text never committed — replacing
        that with just the goodbye would certify a truncated turn as
        complete. The earliest timestamp wins, including a model close's
        deferred commit backdated to its own goodbye's scheduled start.
        """

        when = time.monotonic() if at is None else float(at)
        if self._termination_started_at is None or when < self._termination_started_at:
            self._termination_started_at = when

    @_ignored_once_sealed
    def add_terminal_message(
        self,
        text: str,
        *,
        interrupted: bool = False,
        speech_id: str = "",
        at: float | None = None,
        egress_confirmed: bool | None = None,
        egress_signal: str = "",
    ) -> None:
        """Persist one worker-owned goodbye, recorded on every playout outcome.

        Recorded whether the farewell playout completed or was cut: a goodbye
        that timed out or failed still has known text and must be preserved as
        the resolving turn's response, never dropped.  ``interrupted`` marks a
        cut goodbye so the turn it stands for is never certified as fully heard.
        ``speech_id`` ties this record to the preplayout candidate the worker
        binds at say-time; when present, that bound candidate carries the text
        onto its own interval and this record is deduped against it.
        """

        normalized = " ".join(str(text or "").split())
        if normalized:
            if self._redactor is not None:
                self._terminal_languages[len(self._terminal_messages)] = self._effective_language()
            self._terminal_messages.append(
                TerminalMessage(
                    role="agent",
                    text=normalized,
                    at=time.monotonic() if at is None else float(at),
                    interrupted=bool(interrupted),
                    speech_id=str(speech_id or ""),
                    egress_confirmed=(
                        None if egress_confirmed is None else bool(egress_confirmed)
                    ),
                    egress_signal=str(egress_signal or ""),
                )
            )

    def _last_audible_at(
        self, payload: bytes | memoryview, *, after: float, bound: float
    ) -> float:
        """Where ``payload`` was last audible, searching ``[after, bound]``.

        The LAST audible frame, not the first quiet one: a tail is speech, and
        speech contains gaps.  A stop closure inside the final syllable is tens
        of milliseconds of near-silence with more of the same word behind it, so
        stopping at the first quiet frames would cut exactly the tail this
        exists to recover.  Silence only ends the utterance once it has lasted
        longer than any gap inside one, which is where the search stops.

        Returns ``after`` when nothing past the edge was audible -- including
        the case where the recorder captured no audio there at all, a caller
        who hung up mid-reply.  Measurement is quantised to the frame, and the
        remainder below one frame at the far end of the window is not claimed,
        so the answer errs toward the SDK edge and never past the room.
        """

        frame_samples = max(1, round(self.sample_rate * _PLAYOUT_TAIL_FRAME_SECONDS))
        gap_samples = max(
            frame_samples,
            round(self.sample_rate * _PLAYOUT_TAIL_GAP_SECONDS),
        )
        first_sample = max(0, round((after - self.epoch) * self.sample_rate))
        bound_sample = min(
            len(payload) // 2,
            round((bound - self.epoch) * self.sample_rate),
        )
        last_audible_sample = first_sample
        quiet_since: int | None = None
        cursor = first_sample
        while cursor + frame_samples <= bound_sample:
            if _frame_rms(payload, cursor, frame_samples) < _PLAYOUT_TAIL_QUIET_RMS:
                if quiet_since is None:
                    quiet_since = cursor
                elif cursor + frame_samples - quiet_since >= gap_samples:
                    break
            else:
                last_audible_sample = cursor + frame_samples
                quiet_since = None
            cursor += frame_samples
        return self.epoch + last_audible_sample / self.sample_rate

    def _extend_agent_intervals_to_captured_audio(self) -> None:
        """Move each agent interval's audio edge to the room audio it delivered.

        This runs during finalization and nowhere else.  When the state-change
        edge fires there is nothing to measure: the frames whose absence this
        corrects are still in flight and have not been captured yet.  Only the
        audio edge is set, and only forward -- the interval's ``end`` stays the
        SDK edge every attribution rule is written against -- and positions are
        preserved in place, because ``finalize`` keys every match, every
        attribution map and the declared-goodbye position by an interval's
        position in this list.
        """

        with memoryview(self._tracks["agent"]) as payload:
            if not payload:
                return
            intervals = self._intervals["agent"]
            starts = [interval.start for interval in intervals]
            for position, interval in enumerate(intervals):
                # Never cross into the next interval: a clip may grow into the
                # silence that follows it, never into another utterance.  When two
                # intervals are closer together than the transport lag, the first
                # one's last frames land inside the second one's window -- but they
                # are not lost, because that window also opens at an SDK edge and
                # is early by the same lag.  Claiming them here would put the same
                # audio in two clips; moving the START edge is the other half of
                # this problem and a separate decision (the goodbye's text binds by
                # containment of a stamp in ``[start, end]``).
                next_start = min(
                    (start for start in starts if start >= interval.end),
                    default=float("inf"),
                )
                bound = min(interval.end + _PLAYOUT_TAIL_SEARCH_SECONDS, next_start)
                audible_until = self._last_audible_at(
                    payload, after=interval.end, bound=bound
                )
                if audible_until > interval.end:
                    intervals[position] = replace(interval, audio_end=audible_until)

    @_ignored_once_sealed
    def set_transcript_events_source(
        self, source: Callable[[], Mapping[str, Any]] | None
    ) -> None:
        """Attach the typed transcript-event section for this recording.

        Read exactly once, at finalization. A source that raises is recorded
        as an absent section rather than taking the recording down with it:
        a note about a call must never cost the call its artifact.
        """

        self._transcript_events_source = source

    @_ignored_once_sealed
    def set_closing_evidence_source(self, source: Callable[[], Mapping[str, Any]]) -> None:
        self._closing_evidence_source = source

    def _closing_evidence_section(self) -> dict[str, Any]:
        if self._closing_evidence_frozen is not None:
            return self._closing_evidence_frozen
        try:
            return {"closing_evidence": dict(self._closing_evidence_source())} if self._closing_evidence_source else {}
        except Exception:
            return {}

    def seal(self) -> None:
        """Refuse every further piece of evidence (see ``finalize_off_loop``)."""

        self._closing_evidence_frozen = self._closing_evidence_section()
        self._sealed = True

    def finalize(self, *, ended_at: float | None = None) -> dict[str, object]:
        """Write the evidence bundle once; later calls repeat its outcome.

        A call with no room audio at all raises ``NoRoomAudioCaptured`` and
        leaves capture open, as it always has: nothing was written. Any other
        failure is final. Capture stays closed and every later call raises,
        so a half-written bundle is never reported as a finished one.
        """

        if self._finalize_failure is not None:
            raise RuntimeError(
                f"evidence finalization already failed: {self._finalize_failure}"
            )
        if self._finalized:
            return self._summary()
        self._finalized = True
        try:
            self._remask_whole_call()
            ended = time.monotonic() if ended_at is None else float(ended_at)
            for track, started in self._active.items():
                if started is not None and ended > started:
                    self._intervals[track].append(
                        SpeechInterval(started, ended, speech_id=self._active_speech_id[track])
                    )
                    self._active[track] = None
            self.root.mkdir(parents=True, exist_ok=True)
            maximum = max(len(value) for value in self._tracks.values())
            if maximum == 0:
                self._finalized = False
                raise NoRoomAudioCaptured("no room audio was captured")
            # Measured before the tracks are padded to a common length, so "the
            # end of the captured agent audio" means the last frame the room
            # delivered and not the zero-fill that squares the two tracks off.
            self._extend_agent_intervals_to_captured_audio()
            for value in self._tracks.values():
                _pad_with_silence(value, maximum)
            # Views, not copies: a copy of each track is 27 MiB per 30 minutes
            # of call. Clips are cut from these views too.
            with memoryview(self._tracks["caller"]) as caller, memoryview(
                self._tracks["agent"]
            ) as agent:
                return self._write_evidence(caller, agent)
        except NoRoomAudioCaptured:
            raise
        except BaseException as error:
            self._finalize_failure = f"{type(error).__name__}: {error}"
            raise

    def _write_evidence(self, caller: memoryview, agent: memoryview) -> dict[str, object]:
        self._write_wav(self.root / "caller.wav", caller)
        self._write_wav(self.root / "agent.wav", agent)
        self._write_mixed_wav(self.root / "mixed.wav", caller, agent)
        # Finalized tracks and mix share sample zero. This manifest is verified
        # against the uploaded PCM before channel amplitudes become UI evidence.
        try:
            digests = {}
            for role in ("caller", "agent", "mixed"):
                with (self.root / f"{role}.wav").open("rb") as stream:
                    digests[role] = hashlib.file_digest(stream, "sha256").hexdigest()
            (self.root / "waveform-sources.json").write_text(json.dumps({
                "producer": "skysay_room_recorder", "clock": "recording_sample_zero",
                "sample_rate": self.sample_rate, "sample_count": len(caller) // 2,
                "sha256": digests, **self._waveform_mix_transform,
            }, sort_keys=True), encoding="utf-8")
        except (OSError, ValueError, TypeError, AttributeError):
            logging.getLogger(__name__).warning("Optional waveform source manifest unavailable", exc_info=True)

        user_items = [item for item in self._texts if item.role == "caller"]
        agent_items = [item for item in self._texts if item.role == "agent"]
        caller_interval_matches = self._caller_interval_matches(user_items)
        agent_item_matches = self._agent_item_matches(agent_items)
        if self._speech_to_speech:
            # An identified speech's words are its own transcript, joined by
            # identity. A committed item is placed by timing, and a late one
            # lands on whatever burst is playing when it arrives: it may only
            # speak for a speech the worker could not name.
            agent_item_matches = {
                position: match
                for position, match in agent_item_matches.items()
                if not self._intervals["agent"][position].speech_id
            }
        preplayout_by_interval = (
            self._preplayout_by_speech_identity()
            if self._speech_to_speech
            else self._preplayout_by_agent_interval()
        )
        bound_text_status = (
            MODEL_OUTPUT_TRANSCRIPT if self._speech_to_speech else "bound_before_playout"
        )
        # The worker-declared goodbye: the interval carrying the preplayout
        # candidate whose speech_id a terminal message also declares. When
        # present, this interval is the ONLY post-termination interval any turn
        # may select as its response, so a teardown blip on either side of the
        # goodbye can never stand in for it (see _first_agent_response).
        terminal_speech_ids = {
            message.speech_id for message in self._terminal_messages if message.speech_id
        }
        linked_speech_ids = {
            candidate.speech_id
            for candidate in preplayout_by_interval.values()
            if candidate.speech_id
        }
        declared_goodbye_position: int | None = None
        for position, candidate in preplayout_by_interval.items():
            if candidate.speech_id in terminal_speech_ids:
                declared_goodbye_position = position
                break
        # A terminal message whose goodbye was bound AND linked to an agent
        # interval is already represented there (as the bound turn, or as an
        # opening when end_call ended on the greeting with no caller). Its
        # terminal-message twin is redundant and is dropped from the serialized
        # array so the importer's unattributed-terminal path cannot mint a
        # second farewell turn. A goodbye with no linked audio (P3) or a legacy
        # bundle (no speech_id) keeps its terminal message.
        bound_terminal_positions = {
            position
            for position, message in enumerate(self._terminal_messages)
            if message.speech_id and message.speech_id in linked_speech_ids
        }
        turns: list[dict[str, object]] = []
        turn_contexts: list[
            tuple[
                dict[str, object],
                TranscriptItem,
                SpeechInterval,
                int,
                SpeechInterval,
            ]
        ] = []
        unpaired_caller_turns: list[
            tuple[int, TranscriptItem, SpeechInterval, str]
        ] = []
        # The caller's final utterance answered by a worker goodbye whose TTS
        # produced no room audio at all (synchronous synthesis failure): the
        # farewell text is known but there is no agent interval to attribute it
        # through, so it is carried here and rendered as a semantic turn whose
        # response was never heard.
        no_audio_terminal_turns: list[
            tuple[int, TranscriptItem, SpeechInterval, str, TerminalMessage]
        ] = []
        unmatched_caller_turns: list[tuple[int, TranscriptItem]] = []
        agent_only_turns: list[dict[str, object]] = []
        used_agent_intervals: set[int] = set()
        attributed_terminal_turns: dict[int, int] = {}
        for item_position, user_item in enumerate(user_items):
            index = item_position + 1
            prior_unanswered_caller = bool(
                unpaired_caller_turns or unmatched_caller_turns
            )
            caller_match = caller_interval_matches.get(item_position)
            caller_interval = caller_match[1] if caller_match is not None else None
            response_anchor = (
                caller_interval.end if caller_interval is not None else user_item.at
            )
            later_caller_boundaries = []
            for later_position, later_item in enumerate(
                user_items[item_position + 1 :], start=item_position + 1
            ):
                later_match = caller_interval_matches.get(later_position)
                boundary = (
                    later_match[1].start
                    if later_match is not None
                    else later_item.at
                )
                if boundary > response_anchor:
                    later_caller_boundaries.append(boundary)
            next_caller_boundary = min(later_caller_boundaries, default=None)
            response_entry = self._first_agent_response(
                after=response_anchor,
                before=next_caller_boundary,
                used_agent_intervals=used_agent_intervals,
                declared_goodbye_position=declared_goodbye_position,
                linked_positions=set(preplayout_by_interval),
            )
            if caller_interval is None:
                # Keep the transcript item visible even when an agent
                # response can be paired from its timestamp.  The caller
                # side remains untrusted and therefore has no clip.
                unmatched_caller_turns.append((index, user_item))
                if response_entry is None:
                    # The transcript is authoritative about text that STT
                    # delivered even when VAD never exposed a matching
                    # interval. Keep the item visible, but do not invent a
                    # clip (or borrow the next utterance's clip).
                    continue
                response_position, response_interval = response_entry
                if self._caller_speech_intervenes(
                    anchor=user_item.at,
                    response_interval=response_interval,
                ):
                    # The first transcript item cannot safely claim an agent
                    # interval that begins after a later caller utterance.
                    # Let the later matched item own it instead.
                    continue
                bound_speech = preplayout_by_interval.get(response_position)
                agent_match = agent_item_matches.get(response_position)
                agent_item = agent_match[1] if agent_match is not None else None
                terminal_item = None
                if (
                    agent_item is None
                    and bound_speech is None
                    and not prior_unanswered_caller
                ):
                    terminal_entry = self._terminal_item_for_response(
                        anchor=user_item.at,
                        response_interval=response_interval,
                        attributed_terminal_turns=attributed_terminal_turns,
                    )
                    if terminal_entry is not None:
                        attributed_terminal_turns[terminal_entry[0]] = index
                        terminal_item = terminal_entry[1]
                recorded_item = bound_speech or agent_item or terminal_item
                if recorded_item is not None:
                    used_agent_intervals.add(response_position)
                    agent_name = f"turn-{index:03d}-agent.wav"
                    self._write_wav(
                        self.root / agent_name,
                        self._slice(agent, response_interval),
                    )
                    interrupted, actually_spoken = self._agent_playout_evidence(
                        response_interval,
                        candidate=bound_speech,
                        terminal=terminal_item if recorded_item is terminal_item else None,
                    )
                    agent_only_turns.append(
                        {
                            "index": index,
                            "position": response_position,
                            "text": recorded_item.text,
                            "interval": response_interval,
                            "audio": agent_name,
                            "text_status": (
                                bound_text_status
                                if bound_speech is not None
                                else "recorded_terminal_message"
                                if terminal_item is not None
                                else "recorded"
                            ),
                            "interrupted": interrupted,
                            "actually_spoken": actually_spoken,
                            "order": response_position * 10 + 2,
                        }
                    )
                continue

            if response_entry is None:
                # Preserve the caller's observed clip and transcript as a
                # caller-only entry.  This is an honest partial call (for
                # example a TTS/provider failure), not an implied reply.
                caller_name = f"turn-{index:03d}-caller.wav"
                self._write_wav(
                    self.root / caller_name,
                    self._slice(caller, caller_interval),
                )
                # A worker goodbye whose synthesis failed before any audio still
                # answered THIS caller turn.  Attribute its text as a never-heard
                # response instead of dropping it into terminal_messages alone --
                # but only for the FINAL caller turn (no later caller), stamped
                # after termination, with nothing earlier already unanswered, so
                # a genuinely-lost mid-call reply can never borrow it.  Fails
                # toward the plain unpaired-caller entry, which stays lossless.
                terminal_entry = None
                if (
                    self._termination_started_at is not None
                    and not prior_unanswered_caller
                    and item_position == len(user_items) - 1
                    and caller_interval.end
                    <= self._termination_started_at + _TERMINAL_ATTRIBUTION_SLACK_SECONDS
                ):
                    terminal_entry = self._no_audio_terminal_for_caller(
                        caller_interval=caller_interval,
                        attributed_terminal_turns=attributed_terminal_turns,
                    )
                if terminal_entry is not None:
                    position, message = terminal_entry
                    attributed_terminal_turns[position] = index
                    no_audio_terminal_turns.append(
                        (index, user_item, caller_interval, caller_name, message)
                    )
                    continue
                unpaired_caller_turns.append(
                    (index, user_item, caller_interval, caller_name)
                )
                continue
            response_position, response_interval = response_entry
            used_agent_intervals.add(response_position)
            # LiveKit Agents 1.7.1 commits the assistant ChatMessage only at
            # the END of the reply task, so a caller who hangs up mid-reply
            # leaves the spoken response with no text.  The audio of that
            # response exists and is still the semantic answer -- the
            # interval filter above only accepts agent speech starting at or
            # after the caller stopped, so a greeting (which precedes the
            # caller's turn) can never be paired here, and
            # ``semantic_response_audio_policy`` pins
            # ``filler_acknowledgements_enabled: False``.  Record the turn with
            # its clips and mark it DEGRADED; never present it as complete.
            bound_speech = preplayout_by_interval.get(response_position)
            agent_match = agent_item_matches.get(response_position)
            agent_item = agent_match[1] if agent_match is not None else None
            # The worker's own goodbye is the second way a spoken reply loses
            # its committed text: when the model answers the caller's final
            # utterance by requesting termination, turn admission closes
            # before the reply task commits, and the goodbye reaches this
            # recorder only through ``add_terminal_message`` after completed
            # playout.  That text exists and IS the turn's response, so
            # attribute it instead of degrading a healthy call.  Attribution
            # is deliberately narrow: only the call's FINAL agent speech
            # interval (the terminal goodbye is by construction the last
            # thing the agent says), only when that interval STARTED at or
            # after the recorded termination boundary (so it is wholly
            # worker-owned goodbye speech, not a merged interval that also
            # contains uncommitted model reply audio), only when no OTHER
            # caller utterance began between this turn's utterance and the
            # goodbye (the goodbye answers the caller's last words, and an
            # intervening utterance means this pairing is not that turn),
            # and only a message stamped within the interval's own playout
            # window.  A reply whose text really was lost can therefore
            # never borrow the goodbye.
            terminal_item = None
            if (
                agent_item is None
                and bound_speech is None
                and not prior_unanswered_caller
            ):
                terminal_entry = self._terminal_item_for_response(
                    anchor=caller_interval.start,
                    response_interval=response_interval,
                    attributed_terminal_turns=attributed_terminal_turns,
                )
                if terminal_entry is not None:
                    attributed_terminal_turns[terminal_entry[0]] = index
                    terminal_item = terminal_entry[1]
            caller_name = f"turn-{index:03d}-caller.wav"
            agent_name = f"turn-{index:03d}-agent.wav"
            self._write_wav(
                self.root / caller_name,
                self._slice(caller, caller_interval),
            )
            self._write_wav(
                self.root / agent_name,
                self._slice(agent, response_interval),
            )
            recorded_item = bound_speech or agent_item or terminal_item
            text_status = (
                bound_text_status
                if bound_speech is not None
                else "recorded"
                if agent_item
                else "recorded_terminal_message"
                if terminal_item
                else "unavailable_reply_truncated"
            )
            interrupted, actually_spoken = self._agent_playout_evidence(
                response_interval,
                candidate=bound_speech,
                terminal=terminal_item if recorded_item is terminal_item else None,
            )
            assistant_text = recorded_item.text if recorded_item else ""
            turn = {
                "index": index,
                "user": user_item.text,
                # ``assistant`` is the exact response handed to TTS.  Do not
                # promote it into ``assistant_spoken`` if the sidecar/handle
                # cannot confirm that every word reached playout: callers
                # must be able to distinguish an interrupted response from a
                # fully heard one without losing the authoritative reply.
                "assistant": assistant_text,
                "assistant_spoken": (
                    assistant_text
                    if actually_spoken["verbatim_text_confirmed"]
                    else ""
                ),
                "assistant_text_status": text_status,
                "degraded": recorded_item is None,
                "semantic": True,
                "interrupted": interrupted,
                "actually_spoken": actually_spoken,
                "start_ms": self._relative_ms(caller_interval.start),
                # The turn ends where its response stopped being audible in the
                # room, which is what ``agent_audio`` now contains.
                "end_ms": self._relative_ms(response_interval.playout_end),
                "caller_audio": caller_name,
                "agent_audio": agent_name,
                # How far ``agent_audio`` reaches past the SDK's playout-done
                # edge.  Named after the clip it describes, and reported so a
                # later reader can measure the transport lag on a real call
                # instead of inferring it from a boundary that hides it.
                "agent_audio_tail_ms": self._duration_ms(response_interval.tail_seconds),
            }
            turns.append(turn)
            turn_contexts.append(
                (turn, user_item, caller_interval, response_position, response_interval)
            )
        # A final caller turn answered by a goodbye that produced no audio is a
        # semantic turn whose response was authored but never heard: it belongs
        # in the paired ``turns`` contract (Codex flagged an unpaired final
        # caller), carrying the farewell text with an explicitly not-heard
        # playout rather than the audio-observed default.
        for index, user_item, caller_interval, _caller_name, message in no_audio_terminal_turns:
            turns.append(
                {
                    "index": index,
                    "user": user_item.text,
                    "assistant": message.text,
                    "assistant_spoken": "",
                    "assistant_text_status": "recorded_terminal_message",
                    "degraded": False,
                    "semantic": True,
                    "interrupted": True,
                    "actually_spoken": {
                        "audio_observed": False,
                        "playout_completed": False,
                        "verbatim_text_confirmed": False,
                    },
                    "start_ms": self._relative_ms(caller_interval.start),
                    "end_ms": self._relative_ms(caller_interval.end),
                    "caller_audio": _caller_name,
                    "agent_audio": None,
                }
            )
        turns.sort(key=lambda turn: int(turn["index"]))
        self._turns = turns
        self._unpaired_caller_turns = unpaired_caller_turns
        self._unmatched_caller_turns = unmatched_caller_turns
        entries = self._canonical_entries(
            caller=caller,
            agent=agent,
            turn_contexts=turn_contexts,
            unpaired_caller_turns=unpaired_caller_turns,
            unmatched_caller_turns=unmatched_caller_turns,
            agent_only_turns=agent_only_turns,
            no_audio_terminal_turns=no_audio_terminal_turns,
            used_agent_intervals=used_agent_intervals,
            preplayout_by_interval=preplayout_by_interval,
            attributed_terminal_turns=attributed_terminal_turns,
            agent_item_matches=agent_item_matches,
            bound_text_status=bound_text_status,
        )
        # Clips are derived from independent callback clocks. Once either
        # channel proves delivery backlog, none can safely locate words.
        # Keep all text and full tracks, but remove every derived clip link.
        for rows in (entries, turns):
            for row in rows:
                row["timing_quality"] = "ambiguous" if self._delivery_backlog.detected else "heuristic"
                row["timing_reason"] = "delivery_backlog" if self._delivery_backlog.detected else "callback_timing"
                if self._delivery_backlog.detected:
                    for key in ("audio", "caller_audio", "agent_audio"):
                        if key in row:
                            row[key] = None
        (self.root / "transcript.json").write_text(
            json.dumps(
                {
                    **self._transcript_metadata,
                    **self._transcript_events_section(),
                    **self._closing_evidence_section(),
                    "caller_speech_intervals": self._caller_speech_intervals_section(),
                    "measurement_source": "livekit_room_sidecar_synchronized_tracks",
                    "timing_quality": "ambiguous" if self._delivery_backlog.detected else "heuristic",
                    "timing_reason": "delivery_backlog" if self._delivery_backlog.detected else "callback_timing",
                    "rtp_receive_stats": merged_receive_stats(self._media_receive_stats),
                    "semantic_response_audio_policy": semantic_response_audio_policy(
                        filler_acknowledgements_enabled=self._filler_acknowledgements_enabled
                    ),
                    "tool_wait_feedback": {
                        "mode": self._tool_wait_feedback_mode,
                        "events": list(self._tool_wait_feedback_events),
                    },
                    **(
                        {"voicemail_detection": self._voicemail_detection}
                        if self._voicemail_detection is not None
                        else {}
                    ),
                    **(
                        {"agent_speech_coverage": self._agent_speech_coverage}
                        if self._agent_speech_coverage is not None
                        else {}
                    ),
                    **(
                        {"duplex_liveness": self._duplex_liveness}
                        if self._duplex_liveness is not None
                        else {}
                    ),
                    "summary": {
                        "turn_count": len(turns),
                        "degraded_turn_count": sum(
                            1 for turn in turns if turn["degraded"]
                        ),
                        "unpaired_caller_turn_count": len(unpaired_caller_turns),
                        "unmatched_caller_turn_count": len(unmatched_caller_turns),
                        "terminal_message_count": len(self._terminal_messages)
                        - len(bound_terminal_positions),
                        "preplayout_agent_text_candidate_count": len(
                            self._preplayout_agent_speech
                        ),
                        "preplayout_agent_text_linked_count": sum(
                            1
                            for entry in entries
                            if entry.get("speaker") == "agent"
                            and entry.get("text_status") == bound_text_status
                        ),
                        "preplayout_agent_text_unlinked_count": max(
                            0,
                            len(self._preplayout_agent_speech)
                            - sum(
                                1
                                for entry in entries
                                if entry.get("speaker") == "agent"
                                and entry.get("text_status") == bound_text_status
                            ),
                        ),
                        # Always present, even when 0: declares this artifact
                        # attribution-aware, so a consumer can tell a turn the
                        # worker deliberately left degraded from one an older
                        # recorder merely failed to attribute.
                        "attributed_terminal_message_count": len(
                            attributed_terminal_turns
                        ),
                    },
                    # The worker-owned farewell has TWO legitimate artifact
                    # shapes, chosen by whether it was the resolving turn's reply:
                    #   * ``bound_before_playout`` as the turn's assistant text
                    #     when the goodbye WAS that turn's reply (no preamble) --
                    #     its bound-and-linked terminal twin is dropped here as
                    #     redundant (see ``bound_terminal_positions``);
                    #   * a standalone ``recorded_terminal_message`` entry (with
                    #     its own clip) when a same-generation model preamble took
                    #     the turn, or when end_call ended on the greeting.
                    # Both are honest; a consumer must handle either. A goodbye
                    # that produced no audio at all stays here as a text-only
                    # terminal message (its cut carried by ``interrupted``).
                    "terminal_messages": [
                        {
                            "role": message.role,
                            "text": message.text,
                            "at_ms": self._relative_ms(message.at),
                            # Present only when the farewell playout was cut, so
                            # a consumer of an unattributed message (its own
                            # goodbye interval was never observed) does not read
                            # a truncated goodbye as fully heard. A completed
                            # goodbye keeps its original shape.
                            **({"interrupted": True} if message.interrupted else {}),
                            # Mark a message that already appears as its
                            # turn's assistant text so consumers do not
                            # duplicate it; unattributed entries keep their
                            # original shape.
                            **(
                                {
                                    "attributed_turn_index": attributed_terminal_turns[
                                        position
                                    ]
                                }
                                if position in attributed_terminal_turns
                                else {}
                            ),
                        }
                        for position, message in enumerate(self._terminal_messages)
                        # A bound-and-linked goodbye is represented in ``entries``;
                        # dropping its redundant terminal twin here is what stops
                        # the importer minting a second farewell turn.
                        if position not in bound_terminal_positions
                    ],
                    "entries": entries,
                    "turns": turns,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for path in self.root.iterdir():
            if path.is_file():
                path.chmod(0o600)
        return self._summary()

    def _caller_interval_matches(
        self, user_items: list[TranscriptItem]
    ) -> dict[int, tuple[int, SpeechInterval] | None]:
        """Join caller transcript items to intervals by their timestamps.

        The transcript and VAD callbacks are independent.  Their list
        positions therefore cannot identify the same utterance: a lost VAD
        transition must not make the first transcript item consume the second
        interval.  A candidate is trusted only inside a narrow timestamp
        window and only when it is the sole unused candidate.  Ambiguous or
        distant matches intentionally remain audio-less.
        """

        matches: dict[int, tuple[int, SpeechInterval] | None] = {}
        used_positions: set[int] = set()
        intervals = self._intervals["caller"]
        anchors = [self._caller_join_anchor(item) for item in user_items]
        ordered_positions = sorted(
            range(len(user_items)), key=lambda position: (anchors[position], position)
        )
        for item_position in ordered_positions:
            anchor = anchors[item_position]
            timing = self._caller_transcript_timing
            if (
                self._speech_to_speech or timing is not None
            ) and not self._joins_on_first_interim(user_items[item_position]):
                # A realtime final with no first interim says nothing about
                # when its speech happened; its arrival is the late instant
                # this join exists to stop trusting.
                matches[item_position] = None
                continue
            if timing is not None:
                # Exactly one interval can have been heard as this first
                # interim: the one that opened inside the engine's latency
                # window. Two such, one another item already holds, or one
                # that ended more than the join window before the anchor is
                # not evidence for this item, and it stays untimed.
                heard = [
                    (position, interval)
                    for position, interval in enumerate(intervals)
                    if anchor - timing.first_interim_max_latency
                    <= interval.start
                    <= anchor - timing.first_interim_min_latency
                ]
                # Speech that began just before the window may be this item's
                # own after an unusually late first interim -- even speech
                # another item was joined to, since the engine can split one
                # utterance into several items.
                earlier_unclaimed = any(
                    anchor - timing.first_interim_lookback
                    <= interval.start
                    < anchor - timing.first_interim_max_latency
                    for interval in intervals
                )
                if (
                    len(heard) != 1
                    or heard[0][0] in used_positions
                    or earlier_unclaimed
                    or anchor > heard[0][1].end + _CALLER_TRANSCRIPT_INTERVAL_SLACK_SECONDS
                ):
                    matches[item_position] = None
                    continue
                used_positions.add(heard[0][0])
                matches[item_position] = heard[0]
                continue
            candidates = [
                (position, interval)
                for position, interval in enumerate(intervals)
                if position not in used_positions
                and interval.start - _CALLER_TRANSCRIPT_INTERVAL_SLACK_SECONDS
                <= anchor
                <= interval.end + _CALLER_TRANSCRIPT_INTERVAL_SLACK_SECONDS
            ]
            if not candidates:
                continue
            def distance(candidate: tuple[int, SpeechInterval]) -> float:
                interval = candidate[1]
                if interval.start <= anchor <= interval.end:
                    return 0.0
                return min(
                    abs(anchor - interval.start), abs(anchor - interval.end)
                )

            nearest_distance = min(distance(candidate) for candidate in candidates)
            nearest = [
                candidate
                for candidate in candidates
                if distance(candidate) == nearest_distance
            ]
            if len(nearest) != 1:
                # A transcript timestamp exactly between two nearby speech
                # intervals cannot identify either clip reliably.
                matches[item_position] = None
                continue
            position, interval = nearest[0]
            if interval.start > anchor:
                # Final caller transcripts normally arrive at or after the
                # utterance begins. A sole future interval is therefore the
                # next turn, not evidence for this item; fail closed instead
                # of recreating the positional shift this matcher replaces.
                matches[item_position] = None
                continue
            used_positions.add(position)
            matches[item_position] = (position, interval)
        if self._caller_transcript_timing is not None:
            return self._extend_to_whole_utterances(
                user_items, anchors, matches, self._caller_transcript_timing
            )
        return matches

    def _extend_to_whole_utterances(
        self,
        user_items: list[TranscriptItem],
        anchors: list[float],
        matches: dict[int, tuple[int, SpeechInterval] | None],
        timing: CallerTranscriptTiming,
    ) -> dict[int, tuple[int, SpeechInterval] | None]:
        """Widen each realtime item's clip over the rest of its utterance.

        The caller VAD splits a sentence at a quarter-second pause, and a
        realtime engine's item spans every pause shorter than its own 0.8 s
        segmentation gap, so an item joined on its first interim holds only
        the clause that interim landed in. The engine also splits one
        utterance into several items when its fragments arrive far apart,
        so a neighbouring interval may be another item's.

        Intervals chained by gaps under the engine's gap form a run. The
        intervals after the run's last joined interval that began early
        enough for their words to reach the engine before that item's final,
        and had ended by the time it arrived, are added to it.
        Between two joined items an interval is left to neither, a run near an
        item that could not be joined is not widened at all (its speech may be
        there), a clip never takes an interval an unjoined item's first
        interim could name, and a clip never reaches back before the interval
        its item was joined to: the engine sometimes drops an utterance, and a
        clause before the anchor may be that one.
        """

        if any(not self._joins_on_first_interim(item) for item in user_items):
            # An item with no first interim cannot be placed, so no tail can
            # be shown to exclude its speech: nothing is widened.
            return matches
        intervals = sorted(
            enumerate(self._intervals["caller"]), key=lambda pair: pair[1].start
        )
        runs: list[list[int]] = []
        for rank, (_position, interval) in enumerate(intervals):
            if (
                runs
                and interval.start - intervals[runs[-1][-1]][1].end
                < timing.utterance_gap
            ):
                runs[-1].append(rank)
            else:
                runs.append([rank])
        owner = {
            match[0]: item_position
            for item_position, match in matches.items()
            if match is not None
        }
        unjoined_anchors = [
            anchors[item_position]
            for item_position in range(len(user_items))
            if matches.get(item_position) is None
        ]
        # Speech an unjoined item's first interim could name is that item's
        # as far as anyone can tell; no other clip may take it.
        reserved = {
            position
            for anchor in unjoined_anchors
            for position, interval in enumerate(self._intervals["caller"])
            if anchor - timing.first_interim_max_latency
            <= interval.start
            <= anchor - timing.first_interim_min_latency
        }
        widened = dict(matches)
        for run in runs:
            first, last = intervals[run[0]][1], intervals[run[-1]][1]
            if any(
                first.start <= anchor <= last.end + _CALLER_TRANSCRIPT_INTERVAL_SLACK_SECONDS
                for anchor in unjoined_anchors
            ):
                continue
            joined = [rank for rank in run if intervals[rank][0] in owner]
            if not joined:
                continue
            # After the run's last joined interval, while each began before
            # that item's final arrived: its closing clauses. Never past a
            # refused one.
            item_position = owner[intervals[joined[-1]][0]]
            end: float | None = None
            for rank in run:
                if rank <= joined[-1]:
                    continue
                position, interval = intervals[rank]
                if position in reserved:
                    break
                # Its words had to reach the engine before the final did, and
                # had ended by then: speech still going on after the final is
                # not in it.
                if (
                    interval.start
                    > user_items[item_position].at
                    - timing.first_interim_min_latency
                    or interval.end > user_items[item_position].at
                ):
                    break
                end = interval.end
            if end is not None:
                position, seed = widened[item_position]  # type: ignore[misc]
                widened[item_position] = (position, replace(seed, end=max(seed.end, end)))
        return widened

    def _caller_join_anchor(self, item: TranscriptItem) -> float:
        """The instant a caller item is joined to its speaking interval on.

        A cascade transcriber finalizes an utterance within a few hundred
        milliseconds of its end, so the final's arrival is the anchor there.
        A speech-to-speech engine finalizes late -- GPT-Live after 0.8 s of
        input past the item's last fragment, or when the next utterance's
        first fragment arrives; Gemini Live when its reply completes -- so its
        final lands past the join window or inside the caller's next
        utterance. Its first interim arrives while the item is still being
        spoken, and is the anchor on those engines when the worker stamped one.
        """

        if self._joins_on_first_interim(item):
            assert item.speech_started_at is not None
            return item.speech_started_at
        return item.at

    def _untimed_caller_placements(
        self, unmatched: list[tuple[int, "TranscriptItem"]]
    ) -> dict[int, float]:
        """Where a voice-to-voice engine's untimed caller lines sit in order.

        An engine with no first-interim timing row (Gemini Live) is never
        joined to a speaking interval, so each caller line was placed at its
        final's ARRIVAL -- and that engine sends the final only once its reply
        is under way, which put the caller's line after the agent's answer to
        it. The line is placed instead at the onset of the caller's speech it
        belongs to. That engine's first interim usually arrives once the
        caller has stopped, so the speech is preferably the latest caller
        interval that had ENDED by the first interim -- an interval still open
        then is more likely the caller's next utterance -- and otherwise the
        latest one that had begun by it; failing both, the first interim
        itself. Lines are placed in first-interim order and only ever later
        than the line before, so no line can take speech that precedes an
        earlier line's and reverse the two. Placement only: the entry keeps its
        zero-length, ``unmeasured`` boundary and no clip, so nothing claims a
        measurement the join refused to make.
        """

        if not self._speech_to_speech or self._caller_transcript_timing is not None:
            return {}
        intervals = sorted(self._intervals["caller"], key=lambda interval: interval.start)
        placements: dict[int, float] = {}
        last = float("-inf")

        def anchor_of(item: "TranscriptItem") -> float:
            return item.speech_started_at if item.speech_started_at is not None else item.at

        for index, item in sorted(unmatched, key=lambda pair: (anchor_of(pair[1]), pair[0])):
            anchor = anchor_of(item)
            # Only speech that began after the previous line's placement: an
            # interval at or before it belongs to that line or to nothing.
            began = [interval for interval in intervals if last < interval.start <= anchor]
            ended = [interval for interval in began if interval.end <= anchor]
            chosen = ended[-1] if ended else (began[-1] if began else None)
            if chosen is not None:
                placements[index] = chosen.start
            elif item.speech_started_at is not None:
                placements[index] = item.speech_started_at
            last = placements.get(index, item.at)
        return placements

    def _joins_on_first_interim(self, item: TranscriptItem) -> bool:
        return self._caller_transcript_timing is not None and item.speech_started_at is not None

    def _first_agent_response(
        self,
        *,
        after: float,
        before: float | None,
        used_agent_intervals: set[int],
        declared_goodbye_position: int | None = None,
        linked_positions: set[int] | None = None,
    ) -> tuple[int, SpeechInterval] | None:
        """Find the next agent interval belonging to one caller item.

        A matched caller interval anchors this at its observed end.  If that
        interval is absent, the transcript timestamp is the only trustworthy
        anchor available.  A later caller transcript supplies an upper bound
        for the missing interval's response; the strict start check still
        permits a reply that overlaps that later caller's speech. The existing
        terminal/opening filters then decide whether an observed agent interval
        can be used without inventing a second conversation pairing rule.

        When the worker declared a goodbye interval, a post-termination interval
        that is NEITHER the declared goodbye NOR carries a bound candidate is a
        teardown blip and cannot be selected as any turn's response -- the real
        goodbye (or a same-generation model preamble, which DOES carry a bound
        candidate) is paired instead. A model reply always starts before
        termination, so earlier turns are untouched; a legacy bundle (no declared
        goodbye) keeps the plain earliest-interval pairing.
        """

        linked = linked_positions if linked_positions is not None else set()
        candidates = [
            (position, interval)
            for position, interval in enumerate(self._intervals["agent"])
            if position not in used_agent_intervals
            and interval.start >= after
            and (before is None or interval.start < before)
            and not (
                declared_goodbye_position is not None
                and self._termination_started_at is not None
                and interval.start >= self._termination_started_at
                and position != declared_goodbye_position
                and position not in linked
            )
        ]
        return min(candidates, key=lambda item: (item[1].start, item[0]), default=None)

    def _agent_item_matches(
        self, agent_items: list[TranscriptItem]
    ) -> dict[int, tuple[int, TranscriptItem] | None]:
        """Join committed agent text to intervals without positional drift.

        The assistant item is committed by a separate conversation callback and
        commonly lands at the end of its interval.  Resolve all items against
        all intervals together so a timestamp in the gap between two replies
        cannot be greedily assigned to the first one and shift later text.
        """

        matches: dict[int, tuple[int, TranscriptItem] | None] = {}
        used_interval_positions: set[int] = set()
        ordered_positions = sorted(
            range(len(agent_items)), key=lambda position: (agent_items[position].at, position)
        )
        for item_position in ordered_positions:
            item = agent_items[item_position]
            candidates = [
                (position, interval)
                for position, interval in enumerate(self._intervals["agent"])
                if position not in used_interval_positions
                and interval.start - _AGENT_TRANSCRIPT_INTERVAL_SLACK_SECONDS
                <= item.at
                <= interval.end + _AGENT_TRANSCRIPT_INTERVAL_SLACK_SECONDS
            ]
            if not candidates:
                continue

            def distance(candidate: tuple[int, SpeechInterval]) -> float:
                interval = candidate[1]
                if interval.start <= item.at <= interval.end:
                    return 0.0
                return min(abs(item.at - interval.start), abs(item.at - interval.end))

            nearest_distance = min(distance(candidate) for candidate in candidates)
            nearest = [
                candidate
                for candidate in candidates
                if distance(candidate) == nearest_distance
            ]
            if len(nearest) != 1:
                continue
            interval_position, _ = nearest[0]
            if nearest[0][1].start > item.at:
                # Committed assistant text cannot safely claim later playout.
                # Pre-playout text has its own explicit evidence channel.
                continue
            used_interval_positions.add(interval_position)
            matches[interval_position] = (item_position, item)
        return matches

    def _terminal_item_for_response(
        self,
        *,
        anchor: float,
        response_interval: SpeechInterval,
        attributed_terminal_turns: Mapping[int, int],
    ) -> tuple[int, TerminalMessage] | None:
        """Find a worker-owned farewell that can safely stand for a reply.

        This VALIDATES a pre-selected interval; it does not select one. The
        response interval was already chosen positionally by
        ``_first_agent_response``, and must begin at or after the recorded
        termination boundary -- so it is wholly worker-owned goodbye speech,
        never a model reply whose text was truly lost (a model reply always
        starts before termination, because the reply precedes the tool call) --
        with no caller utterance between the turn and the goodbye. Those checks,
        plus one goodbye per call and the once-only attribution below, uniquely
        identify the goodbye interval, so the farewell's stamp only has to fall
        at or after the interval START, as a sanity floor.

        There is deliberately NO upper bound on the stamp. The worker records
        the terminal message after ``wait_for_playout`` returns, and on the
        timeout path that wait is bounded, not prompt: a stuck speech handle can
        return the message stamp many seconds past the audio's actual end, so an
        ``interval.end + slack`` ceiling would orphan exactly the timed-out
        goodbye this recovery exists for. This intentionally DIVERGES from the
        sibling entries matcher, which keeps the window because it SELECTS an
        interval for an unattributed message and needs timing to disambiguate;
        this validator does not. Not gated on the goodbye being the last agent
        interval either -- a spurious teardown blip can play after it (observed
        in production).
        """

        if (
            self._termination_started_at is None
            or response_interval.start < self._termination_started_at
            or self._caller_speech_intervenes(
                anchor=anchor,
                response_interval=response_interval,
            )
        ):
            return None
        return next(
            (
                (position, message)
                for position, message in enumerate(self._terminal_messages)
                if position not in attributed_terminal_turns
                and message.at >= response_interval.start
            ),
            None,
        )

    def _no_audio_terminal_for_caller(
        self,
        *,
        caller_interval: SpeechInterval,
        attributed_terminal_turns: Mapping[int, int],
    ) -> tuple[int, TerminalMessage] | None:
        """Find the worker goodbye for a final caller turn with NO agent audio.

        Used only when the caller's final utterance has no agent interval after
        it -- a synchronous TTS failure at end_call.  There is no playout window
        to match against, so the goodbye is identified by its stamp alone: an
        unattributed terminal message recorded at or after termination and after
        the caller finished.  Exactly one goodbye exists per call, and the only
        caller turn in this state is the final one, so the stamp is sufficient;
        the finalize loop applies the stricter last-turn/prior-unanswered guards
        before calling this.
        """

        if self._termination_started_at is None:
            return None
        return next(
            (
                (position, message)
                for position, message in enumerate(self._terminal_messages)
                if position not in attributed_terminal_turns
                and message.at >= self._termination_started_at
                and message.at >= caller_interval.end
            ),
            None,
        )

    def _caller_speech_intervenes(
        self, *, anchor: float, response_interval: SpeechInterval
    ) -> bool:
        return any(
            anchor < caller_interval.start < response_interval.start
            for caller_interval in self._intervals["caller"]
        )

    def _preplayout_by_agent_interval(self) -> dict[int, PreplayoutAgentSpeech]:
        """Associate pre-TTS text only with nearby observed agent audio.

        The sidecar and the agent execute in independent processes, so this
        deliberately uses a narrow timing join rather than trusting an intent
        to speak.  When several candidates precede one interval, the most
        recent one wins: this prevents an earlier provider-failed synthesis
        from being attached to a later, unrelated response.

        Current identified speeches join by exact speech identity, including
        reclaimed proposals without a terminal message. They NEVER enter the
        generic timing join. Legacy worker-DECLARED goodbyes (speech_id on a
        terminal message) is linked first, by CONTAINMENT of its observed
        playout-start: it is bound at the stamp taken when the goodbye handle's
        own "speaking" edge fired, and linked to the interval
        ``[start, end]`` that contains that stamp.  ``mark_state`` opened that
        same interval at that same edge, so the stamp sits at or just after its
        start -- and a same-generation model preamble (whose audio can start
        AFTER say-time, which a forward-from-say-time rule would wrongly grab) is
        a DIFFERENT interval that does not contain the goodbye's stamp.  Pinned
        before the generic pass so that pass cannot hand the goodbye candidate to
        another interval; the goodbye interval is then claimed, so the generic
        pass's termination guard is relaxed for OTHER post-termination intervals
        (the preamble keeps its own text as its own turn). Legacy bundles (no
        declared goodbye) keep the strict guard -- a cancelled reply must not
        borrow the goodbye clip.
        """

        available = [candidate for candidate in self._preplayout_agent_speech.values() if candidate.text]
        used: set[str] = set()
        linked: dict[int, PreplayoutAgentSpeech] = {}
        declared_ids = {message.speech_id for message in self._terminal_messages if message.speech_id}
        # Current workers carry the actual handle on every speaking interval.
        # A reclaim has no terminal message; ownership must not depend on one.
        # Never use timestamp containment (or its 1ms tolerance) for these IDs.
        for position, interval in enumerate(self._intervals["agent"]):
            candidate = self._preplayout_agent_speech.get(interval.speech_id)
            if candidate is not None and candidate.text and candidate.speech_id not in used:
                linked[position] = candidate
                used.add(candidate.speech_id)
        for candidate in available:
            if candidate.worker_owned or candidate.speech_id in used or candidate.speech_id not in declared_ids:
                continue
            for position, interval in enumerate(self._intervals["agent"]):
                if position in linked:
                    continue
                # 1ms of armor against float/handler-ordering skew between the
                # stamp and mark_state's interval start (both taken at the same
                # "speaking" edge). Far too small to reach an adjacent interval
                # (inter-interval gaps are tens of ms), so it never mislinks.
                if interval.start - 0.001 <= candidate.at <= interval.end:
                    linked[position] = candidate
                    used.add(candidate.speech_id)
                    break
        for position, interval in enumerate(self._intervals["agent"]):
            if position in linked:
                continue
            if interval.speech_id:
                # A known handle cannot borrow a nearby handle's text. Older
                # bundles without speech identity retain the timing join.
                continue
            candidates = [
                candidate
                for candidate in available
                if candidate.speech_id not in used
                and not candidate.worker_owned
                # A DECLARED goodbye may only link by containment (the pass
                # above), never by this ±15s window: otherwise a candidate that
                # containment could not place (e.g. bound at say-time because its
                # speaking edge was missed) would attach to a LATER preamble
                # interval and win max(at). Skipping declared ids here means such
                # a candidate stays unlinked and falls to an honest terminal
                # message instead of stealing the preamble's clip.
                and candidate.speech_id not in declared_ids
                and candidate.at - 1.0 <= interval.start <= candidate.at + 15.0
                # A model response bound before termination that never produced
                # audio must not be retroactively attached to the worker-owned
                # goodbye clip. When a declared goodbye exists its interval is
                # already claimed above, so this guard is only needed for legacy
                # bundles; relaxing it there lets a same-generation preamble
                # (bound pre-termination, audio post-termination) keep its own
                # interval instead of being stranded.
                and not (
                    not declared_ids
                    and self._termination_started_at is not None
                    and candidate.at < self._termination_started_at <= interval.start
                )
            ]
            if not candidates:
                continue
            candidate = max(candidates, key=lambda item: item.at)
            linked[position] = candidate
            used.add(candidate.speech_id)
        return linked

    def _preplayout_by_speech_identity(self) -> dict[int, PreplayoutAgentSpeech]:
        """Join speech-to-speech text to its interval by SpeechHandle id alone.

        The engine's transcript of a speech and the speaking edge that opened
        its interval both name the same handle, so no timing window is
        consulted: a duplex model answers over the caller and splits one reply
        into several speeches, which is exactly where a timing join guesses.
        A handle names at most one interval, its first.
        """

        linked: dict[int, PreplayoutAgentSpeech] = {}
        used: set[str] = set()
        for position, interval in enumerate(self._intervals["agent"]):
            speech_id = interval.speech_id
            candidate = self._preplayout_agent_speech.get(speech_id) if speech_id else None
            if candidate is None or not candidate.text or speech_id in used:
                continue
            linked[position] = candidate
            used.add(speech_id)
        return linked

    def _cascade_continuation_positions(self, preplayout_by_interval) -> set[int]:
        if self._speech_to_speech:
            return set()
        first = {candidate.speech_id: position for position, candidate in preplayout_by_interval.items()
                 if candidate.text}
        return {position for position, interval in enumerate(self._intervals["agent"])
                if interval.speech_id in first and position > first[interval.speech_id]}

    def _stage_every_agent_speech(
        self,
        *,
        agent: memoryview,
        add_entry: Callable[..., None],
        used_for_entries: set[int],
        texted_positions: set[int],
        preplayout_by_interval: Mapping[int, PreplayoutAgentSpeech],
        agent_item_matches: Mapping[int, tuple[int, TranscriptItem] | None],
        bound_text_status: str,
    ) -> None:
        """Write the agent speech no turn, opening or goodbye already stands for.

        The pairing passes above emit an agent interval
        only as the answer to a caller turn, and a duplex model talks over the
        caller: every interval they leave is still something the caller heard,
        so it becomes an agent-only entry with its own clip. Its text is the
        engine's transcript of that speech (joined by identity), else the
        committed assistant item the timing join placed on it. Audio with no
        text remains an audio-only entry on speech-to-speech calls. Cascade
        also preserves additional text-backed replies; its legacy uncaptioned
        intervals and unplayed synthesis candidates remain absent. The wider
        speech-to-speech coverage and text-without-audio contract is unchanged.
        """

        speech_count = 0
        continuations = self._cascade_continuation_positions(preplayout_by_interval)
        for position, interval in enumerate(self._intervals["agent"]):
            if position in used_for_entries or position in continuations:
                continue
            used_for_entries.add(position)
            candidate = preplayout_by_interval.get(position)
            match = agent_item_matches.get(position)
            text = (
                candidate.text
                if candidate is not None
                else match[1].text
                if match is not None
                else ""
            )
            if not self._speech_to_speech and not text:
                continue
            speech_count += 1
            audio_name = f"{SPEECH_CLIP_PREFIX}-{speech_count:03d}-agent.wav"
            self._write_wav(self.root / audio_name, self._slice(agent, interval))
            interrupted, actually_spoken = self._agent_playout_evidence(
                interval,
                candidate=candidate,
            )
            if text:
                texted_positions.add(position)
            else:
                actually_spoken["verbatim_text_confirmed"] = False
            add_entry(
                speaker="agent",
                text=text,
                interval=interval,
                audio_name=audio_name,
                text_status=(bound_text_status if candidate is not None else "recorded") if text else "audio_only",
                interrupted=interrupted,
                order=position * 10 + 5,
                actually_spoken=actually_spoken,
            )

        if not self._speech_to_speech:
            return

        linked_ids = {
            interval.speech_id for interval in self._intervals["agent"] if interval.speech_id
        }
        declared_goodbye_ids = {
            message.speech_id for message in self._terminal_messages if message.speech_id
        }
        text_without_audio = 0
        for speech_id, candidate in self._preplayout_agent_speech.items():
            # A declared goodbye with no interval stays a terminal message,
            # which the importer already renders; writing it here too would
            # show the farewell twice.
            if (
                not candidate.text
                or speech_id in linked_ids
                or speech_id in declared_goodbye_ids
            ):
                continue
            text_without_audio += 1
            add_entry(
                speaker="agent",
                text=candidate.text,
                interval=None,
                at=candidate.at,
                audio_name=None,
                text_status=bound_text_status,
                interrupted=candidate.interrupted,
                order=7,
                actually_spoken={
                    "audio_observed": False,
                    "playout_completed": False,
                    "verbatim_text_confirmed": False,
                },
            )

        intervals = self._intervals["agent"]
        without_text = [
            position for position in range(len(intervals)) if position not in texted_positions
        ]
        # Over EVERY untranscribed speech, not the listed ones: the list below
        # is capped for size, and a speech the worker could not name must be
        # visible to the alarm wherever it falls in the call.
        unlinked_ms = [
            self._relative_ms(intervals[position].playout_end)
            - self._relative_ms(intervals[position].start)
            for position in without_text
            if not intervals[position].speech_id
        ]
        self._agent_speech_coverage = {
            "agent_speech_intervals": len(intervals),
            "with_text": len(intervals) - len(without_text),
            "audio_without_text_count": len(without_text),
            # Spans, never text: which stretches of agent audio the transcript
            # does not cover, and whether the worker could name their speech.
            "audio_without_text": [
                {
                    "start_ms": self._relative_ms(intervals[position].start),
                    "end_ms": self._relative_ms(intervals[position].playout_end),
                    "identity_linked": bool(intervals[position].speech_id),
                }
                for position in without_text[:_MAX_LISTED_UNCAPTURED_SPEECHES]
            ],
            "audio_without_text_truncated": len(without_text) > _MAX_LISTED_UNCAPTURED_SPEECHES,
            "unlinked_audio_without_text_count": len(unlinked_ms),
            "longest_unlinked_audio_without_text_ms": max(unlinked_ms, default=0),
            "intervals_without_identity": sum(
                1 for interval in intervals if not interval.speech_id
            ),
            "text_without_audio_count": text_without_audio,
        }

    def _agent_playout_evidence(
        self,
        interval: SpeechInterval,
        *,
        candidate: PreplayoutAgentSpeech | None,
        terminal: TerminalMessage | None = None,
    ) -> tuple[bool, dict[str, object]]:
        """State exactly what the sidecar and SDK completion can prove.

        A later caller utterance interrupts the *agent interval it overlaps*,
        not the agent response that follows it.  The room sidecar makes that
        overlap observable even if the SDK completion callback is delayed.
        For pre-playout text, a missing completion callback stays conservative:
        room audio proves that speech started, but not that the complete text
        was played through.  When the recorded text is a worker-owned goodbye,
        ``terminal`` carries the worker's observed playout outcome: a farewell
        that was cut (timed out or failed) is recorded but never certified as
        heard in full, even though its room audio started.
        """

        interrupted = (
            bool(candidate and candidate.interrupted)
            or bool(terminal is not None and terminal.interrupted)
            # A duplex engine listens while it speaks, and the runtime never
            # interrupts it: a caller talking over one of its speeches is the
            # conversation, not a cut. Only the handle can say it was cut.
            or (
                not self._speech_to_speech
                and any(
                    interval.start < caller_interval.start < interval.end
                    for caller_interval in self._intervals["caller"]
                )
            )
        )
        completed = (
            not terminal.interrupted
            if terminal is not None
            else candidate.completed_at is not None
            if candidate is not None
            else True
        )
        playout_completed = completed and not interrupted
        # The SDK edge proves the last frame was handed to the transport, not
        # that it left the room.  When the worker drained the goodbye against
        # the room-side observer, its verdict decides whether the text may be
        # certified as heard in full; a drain that ended without quiet leaves
        # the playout completed but unconfirmed.
        # A goodbye bound at say-time is certified through its candidate, and
        # its terminal record is deduped away -- but that record is where the
        # drain's verdict lives, so look it up by the declared speech id.
        verdict = terminal if terminal is not None else self._terminal_for_candidate(candidate)
        egress_confirmed = verdict.egress_confirmed if verdict is not None else None
        verbatim = playout_completed and egress_confirmed is not False
        certified: dict[str, object] = {
            "audio_observed": True,
            "playout_completed": playout_completed,
            "verbatim_text_confirmed": verbatim,
        }
        if verdict is not None and egress_confirmed is not None:
            certified["egress_confirmed"] = egress_confirmed
            if verdict.egress_signal:
                certified["egress_signal"] = verdict.egress_signal
        return interrupted, certified

    def _terminal_for_candidate(
        self, candidate: PreplayoutAgentSpeech | None
    ) -> TerminalMessage | None:
        """The worker's terminal record for a bound goodbye candidate, if any."""

        if candidate is None:
            return None
        speech_id = next(
            (key for key, value in self._preplayout_agent_speech.items() if value is candidate),
            "",
        )
        if not speech_id:
            return None
        for message in reversed(self._terminal_messages):
            if message.speech_id == speech_id:
                return message
        return None

    def _canonical_entries(
        self,
        *,
        caller: memoryview,
        agent: memoryview,
        turn_contexts: list[
            tuple[
                dict[str, object],
                TranscriptItem,
                SpeechInterval,
                int,
                SpeechInterval,
            ]
        ],
        unpaired_caller_turns: list[
            tuple[int, TranscriptItem, SpeechInterval, str]
        ],
        unmatched_caller_turns: list[tuple[int, TranscriptItem]],
        agent_only_turns: list[dict[str, object]],
        no_audio_terminal_turns: list[
            tuple[int, TranscriptItem, SpeechInterval, str, TerminalMessage]
        ],
        used_agent_intervals: set[int],
        preplayout_by_interval: Mapping[int, PreplayoutAgentSpeech],
        attributed_terminal_turns: Mapping[int, int],
        agent_item_matches: Mapping[int, tuple[int, TranscriptItem] | None] | None = None,
        bound_text_status: str = "bound_before_playout",
    ) -> list[dict[str, object]]:
        """Build the chronological v2 transcript while retaining legacy turns.

        ``turns`` remains the PR #79 paired contract.  ``entries`` is the
        importer's chronology-preserving form: it can represent a greeting or
        standalone completed terminal message without pretending either has a
        caller pair.  Each entry names its exact clip when one was observed.
        """

        staged: list[dict[str, object]] = []

        def add_entry(
            *,
            speaker: Track,
            text: str,
            interval: SpeechInterval | None,
            at: float | None = None,
            audio_name: str | None,
            text_status: str,
            interrupted: bool,
            order: int,
            actually_spoken: Mapping[str, bool] | None = None,
            expected_agent_reply_missing: bool = False,
            measured_duration: float | None = None,
            confidence: float | None = None,
            caption_ingested_at: float | None = None,
            speech_started_at: float | None = None,
        ) -> None:
            normalized = " ".join(str(text or "").split())
            if not normalized:
                if not (self._speech_to_speech and speaker == "agent" and interval is not None and audio_name):
                    return
                text_status = "audio_only"
                actually_spoken = {**(actually_spoken or {}), "audio_observed": True,
                                   "verbatim_text_confirmed": False}
            start = interval.start if interval is not None else at
            # The entry's boundary is the clip's boundary: ``playout_end``
            # matches what ``audio`` holds, edge or extended.
            end = interval.playout_end if interval is not None else at
            if start is None or end is None:
                return
            if interval is not None:
                boundary_source = "vad_interval"
            elif measured_duration is not None and end - measured_duration >= self.epoch:
                # VAD never exposed an interval for this utterance, but the
                # recogniser measured how long it lasted. Its timestamps are
                # stream-relative, so only the LENGTH transfers: it is hung
                # backwards off the observation, which is the one point both
                # clocks share. The length is measured; the placement carries
                # the provider's endpointing and network delay and is
                # therefore an estimate -- which is what the source label
                # says, so no reader mistakes it for the VAD boundary.
                #
                # Only when the span FITS. ``_relative_ms`` clamps a pre-epoch
                # value to 0, so hanging a 600 ms span off an observation 50 ms
                # after the epoch would report a 50 ms turn: a truncated number
                # no instrument produced, which is the defect this change
                # exists to remove rather than relocate. A span that cannot be
                # represented falls through to ``unmeasured`` below.
                start = end - measured_duration
                boundary_source = "stt_measured_duration_at_observation"
            else:
                # Nothing measured this utterance. It keeps a zero-length
                # boundary and says so, rather than being handed a plausible
                # duration no instrument produced.
                boundary_source = "unmeasured"
            staged.append(
                {
                    "_order": order,
                    "speaker": speaker,
                    "text": normalized,
                    "start_ms": self._relative_ms(start),
                    "end_ms": self._relative_ms(end),
                    "boundary_source": boundary_source,
                    **({"caption_ingested_ms": self._relative_ms(caption_ingested_at), "caption_ingestion_clock": "recording_monotonic_zero_v1"}
                       if caption_ingested_at is not None and math.isfinite(caption_ingested_at) and caption_ingested_at >= self.epoch else {}),
                    # The join anchor a realtime engine's caller item was
                    # matched on, kept so the join can be re-run from the
                    # bundle together with ``caller_speech_intervals``.
                    **({"speech_started_ms": self._relative_ms(speech_started_at)}
                       if speech_started_at is not None and speech_started_at >= self.epoch else {}),
                    **({"confidence": confidence} if confidence is not None else {}),
                    "audio": audio_name,
                    # Beside ``audio``, and only for an observed agent clip:
                    # how much of it lies past the SDK's playout-done edge.
                    **(
                        {"audio_tail_ms": self._duration_ms(interval.tail_seconds)}
                        if interval is not None and speaker == "agent"
                        else {}
                    ),
                    "text_status": text_status,
                    "interrupted": interrupted,
                    "actually_spoken": dict(actually_spoken)
                    if actually_spoken is not None
                    else {
                        "audio_observed": True,
                        "playout_completed": not interrupted,
                        "verbatim_text_confirmed": not interrupted,
                    }
                    if speaker == "agent"
                    else {},
                    **(
                        {"expected_agent_reply_missing": True}
                        if expected_agent_reply_missing
                        else {}
                    ),
                }
            )

        first_caller_start = min(
            (interval.start for interval in self._intervals["caller"]),
            default=float("inf"),
        )
        # A first-message greeting has no caller pair by definition.  It is
        # evidence only when the TTS-bound text has a matching room-side agent
        # interval; a failed synthesis remains absent instead of fabricated.
        opening_count = 0
        opening_positions: set[int] = set()
        # Agent intervals an entry carrying text now stands for; read only by
        # the speech-to-speech coverage below.
        texted_positions: set[int] = set()
        for position, interval in enumerate(self._intervals["agent"]):
            candidate = preplayout_by_interval.get(position)
            if (
                candidate is None
                or position in used_agent_intervals
                or interval.start >= first_caller_start
            ):
                continue
            opening_positions.add(position)
            texted_positions.add(position)
            opening_count += 1
            audio_name = "opening-agent.wav" if opening_count == 1 else f"opening-{opening_count:03d}-agent.wav"
            self._write_wav(self.root / audio_name, self._slice(agent, interval))
            interrupted, actually_spoken = self._agent_playout_evidence(
                interval,
                candidate=candidate,
            )
            add_entry(
                speaker="agent",
                text=candidate.text,
                interval=interval,
                audio_name=audio_name,
                text_status=bound_text_status,
                interrupted=interrupted,
                order=position * 10,
                actually_spoken=actually_spoken,
            )

        continuations = self._cascade_continuation_positions(preplayout_by_interval)
        for turn, user_item, caller_interval, response_position, response_interval in turn_contexts:
            caller_name = str(turn["caller_audio"])
            agent_name = str(turn["agent_audio"])
            if str(turn.get("assistant") or "").strip():
                texted_positions.add(response_position)
            add_entry(
                speaker="caller",
                text=user_item.text,
                interval=caller_interval,
                audio_name=caller_name,
                text_status=_caller_text_status(user_item),
                interrupted=False,
                order=response_position * 10 + 1,
                measured_duration=user_item.measured_duration,
                confidence=user_item.confidence,
                caption_ingested_at=user_item.caption_ingested_at,
                speech_started_at=user_item.speech_started_at,
            )
            # Keep the legacy paired-turn text and completeness contract,
            # but do not repeat the same generation's full text in canonical
            # entries when a caller backchannel paused and resumed its audio.
            if response_position in continuations:
                continue
            add_entry(
                speaker="agent",
                text=str(turn.get("assistant") or ""),
                interval=response_interval,
                audio_name=agent_name,
                text_status=str(turn.get("assistant_text_status") or "recorded"),
                interrupted=bool(turn.get("interrupted")),
                order=response_position * 10 + 2,
                actually_spoken=(
                    turn.get("actually_spoken")
                    if isinstance(turn.get("actually_spoken"), dict)
                    else None
                ),
            )

        # A transcript remains useful when the caller spoke but TTS never
        # produced a room-side agent interval.  Preserve exactly that known
        # caller-side evidence and let the importer mark the call partial;
        # never manufacture an assistant utterance or an agent clip.
        for index, user_item, caller_interval, caller_name in unpaired_caller_turns:
            add_entry(
                speaker="caller",
                text=user_item.text,
                interval=caller_interval,
                audio_name=caller_name,
                text_status=_caller_text_status(user_item),
                interrupted=False,
                order=index * 10 + 1,
                expected_agent_reply_missing=True,
                measured_duration=user_item.measured_duration,
                confidence=user_item.confidence,
                caption_ingested_at=user_item.caption_ingested_at,
                speech_started_at=user_item.speech_started_at,
            )

        # The final caller turn plus the worker goodbye that never synthesized:
        # the caller has a clip, the agent entry names the authored farewell with
        # NO clip and an explicitly not-heard playout, so the chronology shows
        # the intended closing without claiming the caller heard it.
        for (
            index,
            user_item,
            caller_interval,
            caller_name,
            message,
        ) in no_audio_terminal_turns:
            add_entry(
                speaker="caller",
                text=user_item.text,
                interval=caller_interval,
                audio_name=caller_name,
                text_status=_caller_text_status(user_item),
                interrupted=False,
                order=index * 10 + 1,
                measured_duration=user_item.measured_duration,
                confidence=user_item.confidence,
                caption_ingested_at=user_item.caption_ingested_at,
                speech_started_at=user_item.speech_started_at,
            )
            add_entry(
                speaker="agent",
                text=message.text,
                interval=None,
                at=message.at,
                audio_name=None,
                text_status="recorded_terminal_message",
                interrupted=True,
                order=index * 10 + 2,
                actually_spoken={
                    "audio_observed": False,
                    "playout_completed": False,
                    "verbatim_text_confirmed": False,
                },
            )

        placements = self._untimed_caller_placements(unmatched_caller_turns)
        for index, user_item in unmatched_caller_turns:
            add_entry(
                speaker="caller",
                text=user_item.text,
                interval=None,
                at=placements.get(index, user_item.at),
                audio_name=None,
                text_status=(
                    FINALIZED_AT_CLOSE if user_item.finalized_at_close else "unmatched_no_audio"
                ),
                interrupted=False,
                order=index * 10,
                measured_duration=user_item.measured_duration,
                confidence=user_item.confidence,
                caption_ingested_at=user_item.caption_ingested_at,
                speech_started_at=user_item.speech_started_at,
            )

        for agent_only in agent_only_turns:
            interval = agent_only.get("interval")
            if not isinstance(interval, SpeechInterval):
                continue
            texted_positions.add(int(agent_only.get("position", -1)))
            add_entry(
                speaker="agent",
                text=str(agent_only.get("text") or ""),
                interval=interval,
                audio_name=(
                    str(agent_only.get("audio"))
                    if agent_only.get("audio")
                    else None
                ),
                text_status=str(agent_only.get("text_status") or "recorded"),
                interrupted=bool(agent_only.get("interrupted")),
                order=int(agent_only.get("order") or 0),
                actually_spoken=(
                    agent_only.get("actually_spoken")
                    if isinstance(agent_only.get("actually_spoken"), dict)
                    else None
                ),
            )

        # A completed worker-owned farewell can be outside a paired semantic
        # turn.  It remains exactly once in ``terminal_messages`` and receives
        # a dedicated clip only when the sidecar observed an unused interval.
        terminal_count = 0
        # A rapid worker-owned farewell can land within the greeting's
        # attribution slack window. Opening clips have already been emitted,
        # so they are consumed here too; otherwise the terminal matcher could
        # relabel the greeting audio as a goodbye and hide the real farewell.
        used_for_entries = set(used_agent_intervals) | opening_positions
        # A bound goodbye's interval is known by identity (its linked candidate),
        # not by the stamp window: on the timeout path the terminal record lands
        # many seconds past the interval end, outside the window, yet the
        # interval is unambiguously the goodbye's.
        linked_position_by_speech_id = {
            candidate.speech_id: position
            for position, candidate in preplayout_by_interval.items()
            if candidate.speech_id
        }
        for message_position, message in enumerate(self._terminal_messages):
            if message_position in attributed_terminal_turns:
                continue
            linked_position = linked_position_by_speech_id.get(message.speech_id)
            if linked_position is not None:
                # Its bound interval is either already represented (a turn or
                # opening consumed it -- nothing to add) or unconsumed here (a
                # same-generation preamble took the turn -- this is the goodbye's
                # only path into the artifact, emitted by identity).
                if linked_position in used_for_entries:
                    continue
                match: tuple[int, SpeechInterval] | None = (
                    linked_position,
                    self._intervals["agent"][linked_position],
                )
            else:
                matches = [
                    (position, interval)
                    for position, interval in enumerate(self._intervals["agent"])
                    if position not in used_for_entries
                    and interval.start
                    <= message.at
                    <= interval.end + _TERMINAL_ATTRIBUTION_SLACK_SECONDS
                    # A no-handle goodbye (session.say raised before returning a
                    # handle, so no speech_id to link by identity) must not grab
                    # an EARLIER interval whose end happens to sit within the
                    # slack -- the greeting most of all. The worker-owned goodbye
                    # always plays after termination, so only a post-termination
                    # interval can be its audio; anything earlier is left
                    # audio-less (better than the greeting clip mislabeled as the
                    # farewell with audio_observed: true).
                    and (
                        self._termination_started_at is None
                        or interval.start >= self._termination_started_at
                    )
                ]
                # A terminal stamp normally lands in (or just after) its own
                # goodbye interval. If an earlier unpaired interval is also
                # inside the two-second slack, nearest observed playout wins;
                # iteration order must never relabel that earlier audio as the
                # terminal message.
                match = min(
                    matches,
                    key=lambda item: (
                        0 if message.at <= item[1].end else 1,
                        abs(message.at - item[1].end),
                    ),
                    default=None,
                )
            if match is None:
                continue
            position, interval = match
            used_for_entries.add(position)
            texted_positions.add(position)
            terminal_count += 1
            audio_name = "terminal-agent.wav" if terminal_count == 1 else f"terminal-{terminal_count:03d}-agent.wav"
            self._write_wav(self.root / audio_name, self._slice(agent, interval))
            interrupted, actually_spoken = self._agent_playout_evidence(
                interval,
                candidate=None,
                terminal=message,
            )
            add_entry(
                speaker="agent",
                text=message.text,
                interval=interval,
                audio_name=audio_name,
                text_status="recorded_terminal_message",
                interrupted=interrupted,
                order=position * 10 + 9,
                actually_spoken=actually_spoken,
            )

        self._stage_every_agent_speech(
            agent=agent,
            add_entry=add_entry,
            used_for_entries=used_for_entries,
            texted_positions=texted_positions,
            preplayout_by_interval=preplayout_by_interval,
            agent_item_matches=agent_item_matches or {},
            bound_text_status=bound_text_status,
        )

        staged.sort(key=lambda item: (int(item["start_ms"]), int(item["_order"])))
        entries: list[dict[str, object]] = []
        for sequence, entry in enumerate(staged, start=1):
            entry.pop("_order", None)
            entry["sequence"] = sequence
            entries.append(entry)
        return entries

    def _slice(self, payload: memoryview, interval: SpeechInterval) -> memoryview:
        # ``playout_end``, not ``end``: a clip must carry everything the room
        # heard, including whatever was still in flight when the SDK reported
        # the speaker had finished. A slice of the view copies nothing, which
        # matters when one clip spans the whole call.
        start = max(0, round((interval.start - self.epoch) * self.sample_rate)) * 2
        end = max(
            start + 2,
            round((interval.playout_end - self.epoch) * self.sample_rate) * 2,
        )
        return payload[start:end]

    def _transcript_events_section(self) -> dict[str, Any]:
        """``{"transcript_events": section}``, or ``{}`` when none was attached.

        An absent key is what an importer reads as an old worker, which is
        exactly what a recording with no attached source is.
        """

        source = self._transcript_events_source
        if source is None:
            return {}
        try:
            section = source()
        except Exception:  # noqa: BLE001 - the artifact outranks the section
            return {}
        return {"transcript_events": dict(section)} if isinstance(section, Mapping) else {}

    def _caller_speech_intervals_section(self) -> dict[str, object]:
        """Every caller speaking interval as observed, before any join.

        Timestamps only, on the entries' clock. A caller item left untimed can
        then be re-joined from the bundle alone (with each entry's
        ``speech_started_ms`` or ``caption_ingested_ms``), instead of being
        lost with the worker's memory. Bounded: a call past the cap keeps its
        first intervals and says the list is incomplete.
        """

        intervals = self._intervals["caller"]
        listed = intervals[:_MAX_LISTED_CALLER_SPEECH_INTERVALS]
        return {
            "clock": "recording_monotonic_zero_v1",
            "source": "caller_state_edges",
            # Which rule joined caller items to these intervals.
            "join": (
                "first_interim"
                if self._caller_transcript_timing is not None
                else "none_no_timing_row"
                if self._speech_to_speech
                else "final_arrival"
            ),
            "intervals": [
                [self._relative_ms(interval.start), self._relative_ms(interval.end)]
                for interval in listed
            ],
            "count": len(intervals),
            "returned": len(listed),
            "has_more": len(intervals) > len(listed),
        }

    def relative_ms(self, value: float) -> int:
        """One monotonic instant as an offset from this recording's epoch.

        The public half of the recorder's clock, so a second sink can place
        its own moments on the SAME axis as ``turns[].start_ms`` instead of
        anchoring a clock of its own. ``epoch`` has always been public, but an
        anchor is not a conversion: the two readings here are seconds apart in
        practice, so each extra anchor is an extra offset to explain
        (measured, not assumed: see the transcript-events contract).
        """

        return max(0, round((value - self.epoch) * 1_000))

    def _relative_ms(self, value: float) -> int:
        return self.relative_ms(value)

    @staticmethod
    def _duration_ms(seconds: float) -> int:
        return max(0, round(seconds * 1_000))

    def _write_wav(self, path: Path, payload: bytes | memoryview) -> None:
        if self._delivery_backlog.detected and path.name not in {"caller.wav", "agent.wav"}:
            return
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(payload)

    @staticmethod
    def _listening_gain(
        samples: memoryview | numpy.ndarray,
        *,
        sample_rate: int,
        speech_intervals: tuple[tuple[int, int], ...],
    ) -> float:
        """Bounded gain from known speech ranges; unknown/no speech stays raw."""

        merged_intervals: list[tuple[int, int]] = []
        for start, end in sorted(speech_intervals):
            if end <= start:
                continue
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0], max(end, merged_intervals[-1][1]))
            else:
                merged_intervals.append((start, end))
        frame_samples = max(1, round(sample_rate * _LISTENING_MIX_FRAME_SECONDS))
        # Whole frames only, as before: a range shorter than one frame adds none.
        # Squares are summed exactly in int64 a bounded block of frames at a
        # time, so no track-sized array is ever allocated. Each frame's mean is
        # the same correctly rounded int/int division Python did.
        frames_per_block = max(1, _LISTENING_MIX_CHUNK_SAMPLES // frame_samples)
        track = numpy.frombuffer(samples, dtype=numpy.int16) if not isinstance(samples, numpy.ndarray) else samples
        frame_rms = []
        try:
            for speech_start, speech_end in merged_intervals:
                frames = len(range(speech_start, speech_end - frame_samples + 1, frame_samples))
                for first in range(0, frames, frames_per_block):
                    count = min(frames_per_block, frames - first)
                    start = speech_start + first * frame_samples
                    block = track[start : start + count * frame_samples].astype(numpy.int64)
                    sums = (block * block).reshape(count, frame_samples).sum(axis=1)
                    frame_rms.extend(math.sqrt(int(total) / frame_samples) for total in sums.tolist())
                    del block, sums
        finally:
            del track
        if not frame_rms:
            return 1.0
        ordered = sorted(frame_rms)
        noise_floor = ordered[max(0, round((len(ordered) - 1) * 0.2))]
        upper = ordered[round((len(ordered) - 1) * 0.9)]
        threshold = _LISTENING_MIX_MIN_SPEECH_RMS
        if noise_floor * 1.5 < upper:
            threshold = max(threshold, noise_floor * 3.0)
        speech = [value for value in frame_rms if value >= threshold]
        minimum_frames = math.ceil(
            _LISTENING_MIX_MIN_SPEECH_SECONDS / _LISTENING_MIX_FRAME_SECONDS
        )
        if len(speech) < minimum_frames:
            return 1.0
        speech.sort()
        representative = speech[round((len(speech) - 1) * 0.7)]
        desired_db = 20.0 * math.log10(_LISTENING_MIX_TARGET_SPEECH_RMS / representative)
        bounded_db = max(_LISTENING_MIX_MIN_GAIN_DB, min(_LISTENING_MIX_MAX_GAIN_DB, desired_db))
        return 10.0 ** (bounded_db / 20.0)

    def _mix(self, caller: bytes, agent: bytes) -> bytes:
        """The whole listening mix in memory. Tests only: see ``_mix_chunks``."""

        return b"".join(self._mix_chunks(caller, agent))

    def _write_mixed_wav(
        self, path: Path, caller: memoryview, agent: memoryview
    ) -> None:
        """Stream the listening mix to ``path``, publishing only a complete file.

        The mix is written to a staging file and replaces ``path`` only once
        every sample is on disk. ``mixed.wav``'s duration is read back as the
        call's duration, so a truncated file must never appear under that name.
        The staging file sits in the same directory, so the rename cannot
        cross filesystems, and its dot-prefixed ``.partial`` name is never one
        of the bundle's files: artifact import skips dotfiles and names its
        audio files explicitly.
        """

        expected = len(caller)
        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        os.close(descriptor)
        staging = Path(staging_name)
        try:
            written = 0
            with wave.open(str(staging), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self.sample_rate)
                with contextlib.closing(self._mix_chunks(caller, agent)) as chunks:
                    for chunk in chunks:
                        output.writeframes(chunk)
                        written += len(chunk) * chunk.itemsize
            size = staging.stat().st_size
            if written != expected or size != _WAV_HEADER_BYTES + expected:
                raise RuntimeError(
                    f"listening mix wrote {written} of {expected} bytes"
                    f" ({size} on disk)"
                )
            os.replace(staging, path)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    def _mix_chunks(
        self,
        caller: bytes | bytearray | memoryview,
        agent: bytes | bytearray | memoryview,
    ) -> Iterator[array.array]:
        """The listening mix, ``_LISTENING_MIX_CHUNK_SAMPLES`` samples at a time.

        Both tracks are read through int16 views, never unpacked, so the only
        per-sample Python objects alive at once are one chunk's. Every output
        sample depends only on its two inputs and on the call-wide gains and
        headroom, so chunking changes no byte.
        """

        with _pcm16_samples(caller) as left, _pcm16_samples(agent) as right:
            if len(left) != len(right):
                raise ValueError("caller and agent tracks must be the same length")
            yield from self._mix_samples(left, right)

    def _mix_samples(self, left: memoryview, right: memoryview) -> Iterator[array.array]:
        def sample_ranges(track: Track, sample_count: int) -> tuple[tuple[int, int], ...]:
            return tuple(
                (
                    max(0, round((interval.start - self.epoch) * self.sample_rate)),
                    min(sample_count, round((interval.playout_end - self.epoch) * self.sample_rate)),
                )
                for interval in self._intervals[track]
                if interval.playout_end > self.epoch
                and interval.start < self.epoch + sample_count / self.sample_rate
            )

        caller_ranges = sample_ranges("caller", len(left))
        agent_ranges = sample_ranges("agent", len(right))
        caller_gain = self._listening_gain(
            left,
            sample_rate=self.sample_rate,
            speech_intervals=caller_ranges,
        )
        agent_gain = self._listening_gain(
            right,
            sample_rate=self.sample_rate,
            speech_intervals=agent_ranges,
        )
        combined_speech_ranges: list[tuple[int, int]] = []
        for start, end in sorted(caller_ranges + agent_ranges):
            if combined_speech_ranges and start <= combined_speech_ranges[-1][1]:
                combined_speech_ranges[-1] = (
                    combined_speech_ranges[-1][0],
                    max(end, combined_speech_ranges[-1][1]),
                )
            else:
                combined_speech_ranges.append((start, end))
        # Views over the two tracks' memory, released below before the
        # caller's memoryviews are: an array still exporting a view would make
        # that release fail, and pin the tracks against a later resize.
        caller = numpy.frombuffer(left, dtype=numpy.int16)
        agent = numpy.frombuffer(right, dtype=numpy.int16)
        try:
            # Estimate shared headroom only from known speech. A click outside
            # a speech interval is bounded locally below, rather than turning
            # down the whole conversation. Chunked, so no float64 copy of a
            # whole range is held.
            peak = 0.0
            for start, end in combined_speech_ranges:
                for first in range(start, end, _LISTENING_MIX_CHUNK_SAMPLES):
                    last = min(end, first + _LISTENING_MIX_CHUNK_SAMPLES)
                    mixed = caller[first:last] * caller_gain + agent[first:last] * agent_gain
                    peak = max(peak, float(numpy.abs(mixed).max()))
            headroom = min(1.0, _LISTENING_MIX_PEAK / peak) if peak else 1.0
            ceiling = round(_LISTENING_MIX_PEAK)
            self._waveform_mix_transform = {"caller_gain": caller_gain, "agent_gain": agent_gain,
                                            "headroom": headroom, "ceiling": ceiling}
            for start in range(0, len(left), _LISTENING_MIX_CHUNK_SAMPLES):
                end = start + _LISTENING_MIX_CHUNK_SAMPLES
                # (caller * gain + agent * gain) * headroom in float64, in the
                # order Python evaluated it; rint rounds half to even like round.
                mixed = (caller[start:end] * caller_gain + agent[start:end] * agent_gain) * headroom
                chunk = numpy.clip(numpy.rint(mixed), -ceiling, ceiling).astype(numpy.int16)
                del mixed
                yield array.array("h", chunk.tobytes())
                del chunk
        finally:
            del caller, agent

    def _summary(self) -> dict[str, object]:
        # Derived from the recorded turns rather than the clip files on disk:
        # a degraded turn also writes turn-*-caller.wav/turn-*-agent.wav, so
        # counting clips would report a truncated reply as complete evidence.
        semantic_turns = len(self._turns)
        degraded_turns = sum(1 for turn in self._turns if turn["degraded"])
        unpaired_caller_turns = len(self._unpaired_caller_turns)
        caller_audio_observed = any(self._tracks["caller"])
        caller_speech_intervals = len(self._intervals["caller"])
        caller_transcript_items = sum(
            1 for item in self._texts if item.role == "caller"
        )
        unmatched_caller_turns = len(self._unmatched_caller_turns)
        return {
            "root": str(self.root),
            "sample_rate": self.sample_rate,
            "semantic_turns": semantic_turns,
            # This counter remains scoped to paired semantic turns. Caller
            # evidence with no agent interval is deliberately represented by
            # its own count below, so a greedy terminal pairing cannot make
            # one degraded semantic turn appear as two.
            "degraded_semantic_turns": degraded_turns,
            "unpaired_caller_turns": unpaired_caller_turns,
            "unmatched_caller_turns": unmatched_caller_turns,
            "semantic_evidence_status": (
                "unavailable_no_linked_turn"
                if not semantic_turns
                and not unpaired_caller_turns
                and not unmatched_caller_turns
                else "degraded_missing_agent_text"
                if degraded_turns
                else "degraded_missing_caller_audio"
                if unmatched_caller_turns
                else "degraded_missing_agent_audio"
                if unpaired_caller_turns
                else "complete"
            ),
            "caller_audio_observed": caller_audio_observed,
            "caller_speech_intervals": caller_speech_intervals,
            "caller_transcript_items": caller_transcript_items,
            "short_call_without_caller_speech": not (
                caller_speech_intervals or caller_transcript_items
            ),
            "tracks_synchronized": True,
            **(
                {"agent_speech_coverage": dict(self._agent_speech_coverage)}
                if self._agent_speech_coverage is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class OffLoopFinalize:
    """What ``finalize_off_loop`` found: the summary, or no room audio at all.

    ``cancellation`` is a cancellation of the awaiting task that arrived while
    the bundle was being written. It is held, not raised, so the caller's
    teardown can finish and surface it the way it surfaces its others.
    """

    summary: dict[str, object] | None
    no_room_audio: bool
    cancellation: asyncio.CancelledError | None


async def finalize_off_loop(
    recorder: SynchronizedEvidenceRecorder, *, ended_at: float | None = None
) -> OffLoopFinalize:
    """``recorder.finalize()`` in a worker thread, so the mix never stalls the loop.

    Mixing and writing a long call's tracks takes seconds of CPU; on the event
    loop that froze every other call this worker was carrying. The recorder
    is sealed here, on the loop, before the thread starts, and the end of the
    call is read here too, so the bundle is exactly what a finalize on the
    loop at this instant would have written. The thread is always waited for:
    a cancellation cannot leave the bundle half-written behind a teardown
    that goes on to read it. Any other finalize failure is raised as before.
    """

    ended = time.monotonic() if ended_at is None else float(ended_at)
    if not any(recorder._tracks.values()):
        # Nothing to mix: answered on the loop exactly as before, so capture is
        # never closed, not even briefly, for a call that has no audio yet.
        try:
            return OffLoopFinalize(
                summary=recorder.finalize(ended_at=ended), no_room_audio=False, cancellation=None
            )
        except NoRoomAudioCaptured:
            return OffLoopFinalize(summary=None, no_room_audio=True, cancellation=None)
    recorder.seal()
    work = asyncio.ensure_future(asyncio.to_thread(recorder.finalize, ended_at=ended))
    cancellation: asyncio.CancelledError | None = None
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError as cancelled:
            cancellation = cancelled
    return OffLoopFinalize(summary=work.result(), no_room_audio=False, cancellation=cancellation)

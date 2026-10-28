# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Attach the SIP bridge's per-call statistics to a call's evidence bundle.

The host runs a small capture unit that follows the LiveKit SIP container's
log and writes one allowlisted JSON file per SIP participant (its own RTP
counters for the carrier leg, the room leg and the mixer, the negotiated
codec, and whether its jitter buffer was on) into a directory this worker
mounts.  The file is named by a digest of the room name and the SIP
participant identity, so no room or participant name -- an inbound room
name embeds the dialed number -- is written to disk.

At finalization the worker looks that file up for its own room and
participant, waits briefly for it, and writes ``sip-call-statistics.json``
beside the transcript with an explicit status.

The bridge writes the statistics line AFTER ``DeleteRoom`` closes the SIP
leg, not before. Production measured the line landing ~65 ms after hangup
and ~3.7 s after a pre-hangup poll had already given up. Attach
therefore runs from ``finalize_call_artifacts``, which the production
entrypoint reaches only after hangup. A missing capture is recorded as
``unavailable`` with the reason -- the unit not installed, the file never
appearing -- rather than silently omitted, so the control plane can tell
"not captured" from "an older worker".  Nothing here gates teardown or
billing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

CAPTURE_SCHEMA = "skysay.sip_call_statistics_capture.v1"
ARTIFACT_SCHEMA = "skysay.sip_call_statistics.v1"
ARTIFACT_NAME = "sip-call-statistics.json"
CAPTURE_DIRECTORY_NAME = "sip-call-statistics"
CAPTURE_DIRECTORY_ENV = "SKYSAY_SIP_CALL_STATISTICS_DIR"
DEFAULT_WAIT_SECONDS = 3.0
DEFAULT_POLL_SECONDS = 0.25
MAX_CAPTURE_BYTES = 65_536

STATUS_COMPLETE = "complete"
STATUS_UNAVAILABLE = "unavailable"
REASON_NOT_CONFIGURED = "capture_not_configured"
REASON_NOT_FOUND = "capture_not_found"
REASON_INCOMPLETE = "capture_incomplete"
REASON_UNREADABLE = "capture_unreadable"


def capture_key(room_name: str, participant_identity: str) -> str:
    """The capture file name for one SIP participant of one room.

    Shared with the host capture unit
    (``infra/voice-host/skysay-sip-call-statistics-capture``); both must
    derive the same digest from the same two strings.
    """
    digest = hashlib.sha256(f"{room_name}\n{participant_identity}".encode("utf-8"))
    return digest.hexdigest()[:32]


def capture_directory(session_root: Path | None, environ: Mapping[str, str] | None = None) -> Path | None:
    """Where the host unit writes, or None when this worker has no session root."""
    env = os.environ if environ is None else environ
    configured = str(env.get(CAPTURE_DIRECTORY_ENV) or "").strip()
    if configured:
        return Path(configured)
    if session_root is None:
        return None
    return session_root / CAPTURE_DIRECTORY_NAME


#: What the host unit writes, rebuilt here rather than copied: a key or a
#: value this worker does not know is not uploaded, whatever the file
#: carried.  The counter names mirror the host unit's allowlists; a string
#: field is validated against its closed shape, never merely truncated.
_PORT_COUNTERS = (
    "streams", "packets", "packets_ignored", "packets_input", "packets_failed",
    "mux_packets", "mux_resets", "mux_gaps", "mux_gaps_sum", "mux_late", "mux_late_sum",
    "mux_rapid_packets", "mux_delayed_packets", "mux_delayed_sum",
    "audio_packets", "audio_in_frames", "audio_out_frames", "dtmf_packets",
    "jitter_buffer_packets_lost", "jitter_buffer_packets_dropped",
)
_PORT_LATENCIES = ("latency_in_e2e", "latency_out")
_ROOM_COUNTERS = (
    "input_packets", "resets", "gaps", "gaps_sum", "late", "late_sum",
    "delayed_packets", "delayed_sum", "rapid_packets", "data_packets", "published_frames",
    "jitter_buffer_packets_lost", "jitter_buffer_packets_dropped",
    "track_subscribes", "resumes", "reconnects",
)
_ROOM_LATENCIES = ("latency_out_recv",)
_MIXER_COUNTERS = (
    "tracks", "tracks_total", "restarts", "timing_resets", "mixes", "mixes_timed",
    "mixes_jump", "mixes_zero", "mixes_negative", "input_frames", "input_frames_dropped",
    "mixed_frames", "output_frames", "write_errors", "blocked_mixes",
)
_LATENCY_FIELDS = ("count", "avg_ns", "max_ns")
_SECTIONS = (("port", _PORT_COUNTERS, _PORT_LATENCIES), ("room", _ROOM_COUNTERS, _ROOM_LATENCIES), ("mixer", _MIXER_COUNTERS, ()))
_DIRECTIONS = ("inbound", "outbound")
AUDIO_CODECS = frozenset({"PCMU/8000", "PCMA/8000", "G722/8000", "opus/48000", "opus/48000/2"})
_LIVEKIT_CALL_ID = re.compile(r"^SCL_[A-Za-z0-9]{1,64}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$")
MAX_COUNTER = 2**53


def _counter(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, min(MAX_COUNTER, value))


def _section(source: Any, counters: tuple[str, ...], latencies: tuple[str, ...]) -> dict[str, Any]:
    """Named integer counters and named ``{count, avg_ns, max_ns}`` latencies only."""
    if not isinstance(source, Mapping):
        return {}
    kept: dict[str, Any] = {}
    for name in counters:
        counter = _counter(source.get(name))
        if counter is not None:
            kept[name] = counter
    for name in latencies:
        raw = source.get(name)
        if not isinstance(raw, Mapping):
            continue
        latency = {field: _counter(raw.get(field)) for field in _LATENCY_FIELDS}
        latency = {field: item for field, item in latency.items() if item is not None}
        if latency:
            kept[name] = latency
    return kept


def _closed(value: Any, pattern: re.Pattern[str] | None = None, vocabulary: tuple[str, ...] | frozenset[str] | None = None) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if vocabulary is not None:
        return text if text in vocabulary else ""
    return text if pattern is not None and len(text) <= 64 and pattern.fullmatch(text) else ""


def allowlisted_capture(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The capture rebuilt from the fields this worker knows, nothing else."""
    rebuilt: dict[str, Any] = {
        "schema": CAPTURE_SCHEMA,
        "room_key": _closed(payload.get("room_key"), re.compile(r"^[0-9a-f]{32}$")),
        "direction": _closed(payload.get("direction"), vocabulary=_DIRECTIONS),
        "livekit_sip_call_id": _closed(payload.get("livekit_sip_call_id"), _LIVEKIT_CALL_ID),
        "audio_codec": _closed(payload.get("audio_codec"), vocabulary=AUDIO_CODECS),
        "codecs_logged_at": _closed(payload.get("codecs_logged_at"), _TIMESTAMP),
        "statistics_logged_at": _closed(payload.get("statistics_logged_at"), _TIMESTAMP),
    }
    jitter = payload.get("jitter_buffer_enabled")
    rebuilt["jitter_buffer_enabled"] = jitter if isinstance(jitter, bool) else None
    duration = _counter(payload.get("duration_minutes"))
    if duration is not None:
        rebuilt["duration_minutes"] = duration
    stats = payload.get("stats")
    if isinstance(stats, Mapping):
        rebuilt["stats"] = {name: _section(stats.get(name), counters, latencies) for name, counters, latencies in _SECTIONS}
    return rebuilt


def read_capture(path: Path, *, room_key: str) -> tuple[dict[str, Any] | None, str]:
    """``(capture, reason)``: the allowlisted capture when complete, else why not.

    A file that exists but only carries the codecs line is ``capture_incomplete``
    (the statistics line has not landed yet); the caller keeps waiting.
    """
    try:
        if not path.is_file() or path.is_symlink():
            return None, REASON_NOT_FOUND
        if path.stat().st_size > MAX_CAPTURE_BYTES:
            return None, REASON_UNREADABLE
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, REASON_UNREADABLE
    if not isinstance(payload, dict) or payload.get("schema") != CAPTURE_SCHEMA or payload.get("room_key") != room_key:
        return None, REASON_UNREADABLE
    if not isinstance(payload.get("stats"), dict):
        return None, REASON_INCOMPLETE
    return allowlisted_capture(payload), ""


def attach_sip_call_statistics(
    evidence_dir: Path,
    *,
    capture_dir: Path | None,
    room_name: str,
    participant_identity: str,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, str]:
    """Write ``sip-call-statistics.json`` into ``evidence_dir``; return its status.

    Bounded: at most ``wait_seconds`` of polling, then an ``unavailable``
    record with the last reason.  Never raises for a missing or malformed
    capture; an unwritable evidence directory is the caller's failure to
    classify, like every other artifact.
    """
    room_key = capture_key(room_name, participant_identity)
    capture: dict[str, Any] | None = None
    reason = REASON_NOT_CONFIGURED
    if capture_dir is not None and capture_dir.is_dir():
        path = capture_dir / f"{room_key}.json"
        deadline = monotonic() + max(0.0, wait_seconds)
        while True:
            capture, reason = read_capture(path, room_key=room_key)
            if capture is not None or reason == REASON_UNREADABLE or monotonic() >= deadline:
                break
            sleep(max(0.01, poll_seconds))
    record: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "status": STATUS_COMPLETE if capture is not None else STATUS_UNAVAILABLE,
        "reason": "" if capture is not None else reason,
        "room_key": room_key,
        "captured": capture or {},
    }
    target = evidence_dir / ARTIFACT_NAME
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return {"status": record["status"], "reason": record["reason"]}


__all__ = [
    "ARTIFACT_NAME",
    "ARTIFACT_SCHEMA",
    "CAPTURE_SCHEMA",
    "attach_sip_call_statistics",
    "capture_directory",
    "capture_key",
    "read_capture",
]

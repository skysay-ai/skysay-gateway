# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Whether this voice host's phone agents are registered and admitting.

After a voice-host boot or a restart, the outbound originator used to claim a
``call.outbound_originate`` job before any agent worker had registered with
LiveKit. The dispatch then had nobody to take it, the originator's join wait
ran out, and the call failed terminally although the host could have served it
seconds later (a boot's first call failed 3 s before the agents registered).

So the phone agents publish a small **beacon** on this host, and the
originator claims nothing while no beacon says "registered and not draining".

* **Written by** each agents worker's main process (blue and green of a
  rolling release each write their own file, named by their worker id), every
  :data:`BEACON_INTERVAL_SECONDS`, from the moment the worker starts until it
  closes. ``registered`` is true only while the worker holds a live
  registration with LiveKit; ``draining`` is the SDK's own drain flag.
* **Read by** the originator before every claim. A beacon older than
  :data:`STALE_AFTER_SECONDS` is a dead or wedged worker and counts as not
  ready, whatever it last said. Any one fresh, registered, non-draining beacon
  makes the host ready.

Only claiming is gated. A job the originator already claimed runs as before,
and a dispatch that still finds no agent fails exactly as it did (founder,
2026-09-27: no requeue, no new failure code).

Pure: no livekit import, an injected clock, and atomic file writes, so both
sides test without a LiveKit server.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: The directory both roles mount: read-write for the agents, read-only for
#: the originator. Unset means the barrier is off. The managed voice host's
#: release hook sets it on both roles; a self-hosted stack that runs its own
#: agents beside a queue originator gets the barrier only by setting it too
#: (a shared directory the agents can write), and otherwise claims as before.
READINESS_DIR_ENV = "SKYSAY_AGENTS_READINESS_DIR"
#: How often a worker rewrites its beacon.
BEACON_INTERVAL_SECONDS = 2.0
#: A beacon not rewritten for this long belongs to a worker that died or
#: wedged without removing it (SIGKILL, OOM, host loss): it is not ready.
STALE_AFTER_SECONDS = 10.0
#: How long the originator may wait for agents before its heartbeat says the
#: host is unavailable and an alert is raised. A healthy boot registers in
#: about 15 s after its container starts, and
#: a release roll hands over within its canary window; three minutes of no
#: registered agent is a fault, and queued calls must not wait on it silently.
UNAVAILABLE_AFTER_SECONDS = 180.0

_BEACON_SUFFIX = ".json"
_SAFE_KEY = re.compile(r"[^A-Za-z0-9_.-]")

#: Reasons, in the order the reader reports them: the best state it found.
READY = "ready"
DRAINING = "draining"
UNREGISTERED = "unregistered"
STALE = "stale"
NO_BEACON = "no_beacon"
_RANK = {READY: 0, DRAINING: 1, UNREGISTERED: 2, STALE: 3, NO_BEACON: 4}


def readiness_dir(environ: Mapping[str, str]) -> Path | None:
    value = str(environ.get(READINESS_DIR_ENV) or "").strip()
    return Path(value) if value else None


def beacon_path(directory: Path, worker_key: str) -> Path:
    key = _SAFE_KEY.sub("_", str(worker_key or "").strip()) or "agents"
    return directory / f"{key}{_BEACON_SUFFIX}"


def write_beacon(directory: Path, worker_key: str, *, registered: bool, draining: bool, now: float) -> None:
    """Replace this worker's beacon atomically (write, then rename)."""

    directory.mkdir(parents=True, exist_ok=True)
    target = beacon_path(directory, worker_key)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {"registered": bool(registered), "draining": bool(draining), "written_at": float(now), "pid": os.getpid()},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    # Readable by the originator whatever the writer's umask (the carrier
    # entrypoint runs the worker under 077); never writable by it.
    os.chmod(temporary, 0o644)
    os.replace(temporary, target)


def remove_beacon(directory: Path, worker_key: str) -> None:
    try:
        beacon_path(directory, worker_key).unlink()
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class Readiness:
    ready: bool
    #: The best state any beacon was in (see the module constants).
    reason: str
    #: Beacons read, fresh or not.
    beacons: int


def read_readiness(directory: Path, *, now: float, stale_after: float = STALE_AFTER_SECONDS) -> Readiness:
    """Ready iff at least one fresh beacon is registered and not draining.

    An unreadable, malformed or future-dated beacon is treated as stale: it
    can only ever make the host look LESS ready, never more.
    """

    try:
        paths = sorted(path for path in directory.iterdir() if path.name.endswith(_BEACON_SUFFIX) and not path.name.startswith("."))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return Readiness(False, NO_BEACON, 0)
    best = NO_BEACON
    for path in paths:
        state = STALE
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            written_at = float(document["written_at"])
            fresh = 0.0 <= now - written_at <= stale_after
            if fresh and isinstance(document.get("registered"), bool) and isinstance(document.get("draining"), bool):
                if document["draining"]:
                    state = DRAINING
                elif document["registered"]:
                    state = READY
                else:
                    state = UNREGISTERED
        except (OSError, ValueError, KeyError, TypeError):
            state = STALE
        if _RANK[state] < _RANK[best]:
            best = state
    return Readiness(best == READY, best, len(paths))

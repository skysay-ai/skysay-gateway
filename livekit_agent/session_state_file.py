# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Serialized read-modify-write for the session-state file.

That file has TWO writers in TWO CONTAINERS sharing the ``/sessions`` volume:
the call's own worker, which creates it, heartbeats into it every five seconds
and stamps the ending its arbiter decides; and the crash reconciler, which
sweeps calls whose worker died or stopped making progress and annotates the same
file with its own bookkeeping.

Both do read-modify-write. Until the reconciler grew an arm that acts on a call
whose worker is still ALIVE, the two never overlapped in practice -- the sweep
only ever touched records nobody was writing any more. That arm changed it:
roughly a sweep every five seconds against a heartbeat every five seconds, for
as long as a wedged call lasts, on the one file whose whole purpose is carrying
the reason a call ended through a crash.

Two things had to be true for that to be safe and neither was:

1. The read and the replace must be one step. Otherwise a decision committed
   between them is read as absent and written away -- and the losses are the
   exact failures this file exists to prevent: a terminal status reverted to
   ``media_started`` so the call is settled and billed a second time, or the
   reason the worker decided replaced by nothing at all.
2. The scratch file each writer builds must be its own. All three writers used
   ``<call_id>.tmp``. Overlapping, one writer's payload lands under the other's
   name and the loser's ``replace`` raises ``FileNotFoundError`` -- which in the
   heartbeat is not caught, so the exception ends the heartbeat task on a live
   call and the staleness arm sweeps it fifteen seconds later.

The lock is taken on a SIDECAR path, never on the state file's own descriptor:
publishing is a ``replace``, which swaps the inode, so a lock held on the file
being replaced protects nothing for the next process to open that name. It is
advisory and only these writers take it, which is all that is needed because
they are the only writers. ``flock`` is released when the descriptor closes,
including when a process dies, so there is no stale lock to clean up and a
crashed worker cannot wedge the sweep.

Acquisition FAILS OPEN, and is BOUNDED. If the lock file cannot be opened, or
the other writer still holds it after ``LOCK_WAIT_SECONDS``, the write goes
ahead unserialized -- which is exactly the behaviour that shipped before this
existed. Both writers run on the media event loop, so an unbounded wait here
would not merely delay a file write, it would stop the audio, and a rule that
can park the call loop is the incident this mechanism exists to prevent. A lock
that cannot be taken must never be the reason a live call stops heartbeating.

This lives in its own module, imported by both writers, because the reconciler
runs as a flat ``/app/reconcile_sessions.py`` in a container that does not have
the worker's import closure, and because a rule two files must agree on is
behaviour rather than a constant -- a second copy is a second thing to keep in
step.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


#: How long a writer will wait for the other container before going ahead
#: without the lock.
#:
#: Both writers run ON THE MEDIA EVENT LOOP -- the worker's heartbeat is a bare
#: call inside its async task, and the arbiter decides from a disconnect
#: callback -- so a blocking acquisition here does not merely delay a file
#: write, it stops the audio. An unbounded wait would put a rule that can park
#: the call loop indefinitely into the one process whose parking is the
#: incident this whole mechanism exists to prevent.
#:
#: The section being waited on is a small read, a dict update and a rename on a
#: local volume, so the honest expected wait is well under a millisecond.
#: Two seconds is far past any legitimate hold and still comfortably inside the
#: five-second heartbeat period, so a writer that hits this bound has found
#: something wrong rather than something busy -- and its answer is to write
#: anyway, which is what the whole file did before the lock existed.
LOCK_WAIT_SECONDS = 2.0


def _acquire_within(descriptor: int, seconds: float) -> bool:
    """Take the lock, or give up after ``seconds`` and let the caller proceed.

    Polled rather than blocking because the blocking form cannot be bounded and
    this runs on the event loop. Returns whether it was taken, for tests; the
    caller proceeds either way, because serialization is a correctness
    improvement over the previous behaviour rather than a precondition for it.
    """

    import fcntl

    deadline = time.monotonic() + seconds
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        except OSError:
            # A filesystem that cannot flock at all (some network mounts)
            # degrades to the unserialized behaviour rather than failing calls.
            return False


@contextmanager
def locked_session_file(path: Path) -> Iterator[None]:
    """Hold the exclusive lock for ``path`` for the body of the ``with``.

    The body must be read, modify, write, replace -- and nothing else. No
    network call, no control-plane report, no callback belongs inside it: this
    blocks the other container's writer for as long as it is held, and the
    sweep's own work between reading a record and writing it back is precisely
    where its network round trips live.

    Not re-entrant, and it does not need to be. The arbiter releases a disowned
    decision BEFORE ``finish`` is handed control, and the heartbeat's operator
    check completes its ``end`` call before making its own write, so the two
    sections in this codebase follow each other rather than nest.
    """

    lock_path = path.with_suffix(".lock")
    handle = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")
    except OSError:
        # Fail open: unserialized is what shipped before, and is survivable.
        # A lock we cannot take must not stop a heartbeat.
        if handle is not None:
            handle.close()
        yield
        return
    try:
        _acquire_within(handle.fileno(), LOCK_WAIT_SECONDS)
        yield
    finally:
        handle.close()


class SessionFileUnreadable(Exception):
    """The record could not be read. NOT the same as there being none.

    Collapsing those two into one empty answer is what made the release
    verification vacuous: a reader that says ``{}`` for "I could not look"
    tells a caller checking whether a key is gone that it is gone. Every
    fail-open bug tonight has this shape -- an absent input read as the
    reassuring answer -- and this one sat inside the very check written to stop
    it, which is why the two cases are now different types rather than
    different comments.
    """


def read_session_file(path: Path) -> dict[str, Any]:
    """The record on disk, or ``{}`` when there genuinely is none.

    ``{}`` means ABSENT and nothing else. A file that exists but cannot be read
    or parsed raises :class:`SessionFileUnreadable`, so a caller deciding
    something on the strength of a key's absence has to handle not knowing.

    Callers that legitimately recover from a corrupt record -- the writer, which
    overwrites it -- catch that exception explicitly. Callers that must not
    guess -- the delete, and the verification after it -- let it propagate and
    retry.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # ValueError as well as OSError, because ``read_text`` DECODES: a
        # non-UTF-8 record raises UnicodeDecodeError, which is a ValueError and
        # not an OSError, so an OSError-only guard let it escape uncaught. That
        # defeated the writers' recovery branch and, since the heartbeat's write
        # is unguarded, ended the heartbeat task on a LIVE call -- the same
        # consequence as two writers sharing one scratch filename.
        raise SessionFileUnreadable(f"session file could not be read: {exc}") from exc
    try:
        existing = json.loads(raw)
    except ValueError as exc:
        raise SessionFileUnreadable(f"session file is not valid JSON: {exc}") from exc
    if not isinstance(existing, dict):
        raise SessionFileUnreadable("session file is not an object")
    return dict(existing)


def publish_session_file(path: Path, record: dict[str, Any]) -> None:
    """Replace ``path`` with ``record``, through a scratch file of our own.

    The scratch name is unique per write. A shared one let two overlapping
    writers land each other's payloads and made the loser's ``replace`` raise.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as scratch:
            scratch.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

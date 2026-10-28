# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Admit a phone call only while the host has room for it, and refuse it cleanly.

**Why.** The phone host is CPU-bound: three concurrent calls run well, and at
four the voice activity detector falls tens of seconds behind (measured on
opx-voice, 2026-09-24). The agents' own job gate cannot hold that number. It
counts a job that is still launching twice (once as active, once as reserved),
so no job budget gives both an exact ceiling of three and room for a third
call arriving while the second launches. And when that gate refuses, nothing
of ours runs: livekit-server gives the dispatch up with no retry, and an
inbound caller hears ringback until the carrier or livekit/sip's three-minute
ringing timeout gives up. They never reach an agent and never appear anywhere
a customer can see.

**What this does.** The worker's ``request_fnc`` asks this module for every
dispatch. It counts the jobs this worker ACCEPTED and that have not ended --
at accept time, in our own code, so a launching job is counted exactly once --
against the host's declared capacity. Under it, the dispatch is accepted.
Over it, the dispatch is rejected, and the worker ends an inbound caller's
ringing leg at once (livekit/sip answers a leg whose room is gone with
``486 Busy Here``). An outbound dispatch needs nothing more: the originator
dials only after its agent has joined, so a rejected dispatch is never dialled.

**What the count is.** The union of two sets of job ids: the ones this module
admitted whose ``accept()`` has not returned yet, and the worker's own
``active_jobs``, less the jobs whose CALL has ended. livekit-agents 1.8.1 puts
a job into ``active_jobs`` inside ``launch_job``, before ``accept()`` returns,
and takes it out only when the job PROCESS exits -- after the call's evidence
and reports are written, 2-15 s after the room closed (p50 4.5 s, p90 9.5 s on
opx-voice, 282 calls to 2026-09-25). A slot held through that tail refuses a
caller the host has room for, so the job process marks its job ended
(``EndedCalls``) the moment its room disconnects, and the count leaves it out.
Its process still counts against a second, higher ceiling -- the host's job
budget, ``process_cap`` -- because live calls on the same CPU pay for its
evidence mix.
An id in both sets is one job. Deciding and marking happen with no ``await``
between them, so two dispatches arriving together cannot both see the same
free slot.

**Live-call voice tests take only what customers can spare.** A
Simulations Lab live-call test (room ``opx-suite-<call id>``, its agent
envelope naming that call) is admitted only while three things hold: fewer
than ``test_slots`` lab rooms are held on this worker, the calls held plus
this one leave ``customer_headroom`` slots free, and the job processes plus
this one leave as many processes free (design F4: two test slots per host,
two slots and two processes kept for customers). So a test never takes the
last slots a customer call needs, while a customer call may still use every
slot a test leaves. A refused test is not an attempt outcome: the control
plane is told ``capacity_reserved`` and the run waits. The job reports the
branch it was admitted on when it claims its session, and the control plane
refuses a mismatch either way (an envelope that looks like a test but has no
open, or a real test admitted as a customer), so the branch is never trusted
to the envelope alone. Every other dispatch
-- inbound, outbound, browser Talk, the post-verdict lab room -- is admitted
exactly as before.

**What it does not cover.** A blue/green release registers two workers for a
few seconds, each counting its own jobs, so the host can briefly admit up to
twice its capacity -- and, the other way round, a dispatch livekit-server
offers to the FULL one of the two is refused (``reject()`` ends the job; it is
not offered to the other) although the other had room. Releases do not run
during campaigns, and the overlap ends when the old worker drains. A second
phone host needs more than this module: a load that livekit-server can route
by, and a refusal that lets the dispatch try another worker first.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from pathlib import Path
from typing import Any

#: The worker log event for a dispatch this module refused.
REFUSED_EVENT = "phone_capacity_refused"

#: Every Simulations Lab room's name begins with this (the post-verdict room,
#: named for its attempt, and a live-call test, named for its call).
SUITE_ROOM_PREFIX = "opx-suite-"
#: The agent envelope a lab room's dispatch carries (the browser-test one,
#: ``control_plane.parse_workspace_browser_test_dispatch``), and the caller
#: identity a live-call test's envelope names (``suite-caller-<call id>``).
BROWSER_TEST_DISPATCH_SCHEMA = "skysay.workspace-browser-test.v1"
LIVE_CALL_CALLER_PREFIX = "suite-caller-"
#: What the control plane is told about a refused live-call test: the run
#: waits for capacity, the attempt has no outcome (design 8.1).
CAPACITY_RESERVED = "capacity_reserved"
#: The capacity report is sent this many times at most, the waits between
#: them growing by this much (an error or timeout only; ``false`` is final).
REPORT_ATTEMPTS = 3
REPORT_RETRY_SECONDS = 1.0
#: Per host (design F4), configuration: lab rooms a worker may hold at once,
#: and the call slots (and job processes) a test must leave free for customers.
SIM_TEST_SLOTS_VAR = "SKYSAY_SIM_TEST_SLOTS_PER_HOST"
SIM_CUSTOMER_HEADROOM_VAR = "SKYSAY_SIM_CUSTOMER_HEADROOM"
DEFAULT_SIM_TEST_SLOTS = 2
DEFAULT_SIM_CUSTOMER_HEADROOM = 2
SIM_RESERVE_CEILING = 16

#: Every ``refusal_outcome`` a refusal reports (worker_events keeps the list).
REFUSAL_OUTCOMES = (
    "leg_ended",
    "leg_end_failed",
    "leg_already_ending",
    "live_call_left_alone",
    "no_inbound_caller",
    "outbound_marked",
    "outbound_mark_failed",
    "capacity_reserved_reported",
    "capacity_reserved_report_failed",
)

#: The schema of the originator's dispatch metadata (outbound_worker
#: DISPATCH_SCHEMA; a test pins the two equal). A refused dispatch carrying it
#: is an outbound call that has not been dialled.
OUTBOUND_DISPATCH_SCHEMA = "skysay.outbound-dispatch.v1"

#: The room-metadata key a refusal of an OUTBOUND dispatch sets, for the
#: originator: its join wait reads it and gives the job back to the queue
#: instead of failing a call nobody dialled. The value is the
#: refused job's id.
#:
#: The write is read-modify-write, and safe as one: an outbound room's
#: metadata is written by the originator alone, once, when it creates the
#: room -- before it creates the dispatch, so before any agent can refuse it.
#: Nothing else writes an ``opx-out-`` room's metadata (the production agent's
#: browser-test error report writes browser-test rooms only; the release
#: canary writes its probe rooms only).
CAPACITY_REFUSED_KEY = "phone_capacity_refused"

#: What a refusal found in the room, reported in place of the room's name:
#: an inbound room is named for the caller's number, which is not logged.
ROOM_KINDS = (
    "inbound_ringing",  # a ringing inbound caller
    "occupied",  # an answered inbound leg, or an agent
    "no_inbound_caller",  # none appeared in time: outbound, or never joined
    "unknown",  # the lookup failed
    "outbound",  # the originator's own dispatch, not yet dialled
    "live_call_test",  # a Simulations Lab live-call test, kept off the customers' headroom
)

#: Ending the refused leg, or recording it, failed; the leg ends on the
#: trunk's ringing timeout instead.
REFUSAL_FAILED_EVENT = "phone_capacity_refusal_failed"

#: The participant attribute livekit/sip sets on an INBOUND caller only: the
#: dispatch rule that routed the call. An outbound leg carries none.
INBOUND_RULE_ATTRIBUTE = "sip.ruleID"

#: The SIP participant kind in LiveKit's ParticipantInfo.Kind enum.
SIP_PARTICIPANT_KIND = 3

#: LiveKit's ParticipantInfo.Kind for an agent.
AGENT_PARTICIPANT_KIND = 4

#: The call status livekit/sip gives an inbound leg it has not answered yet
#: (``sip.callStatus``). Only such a leg is ended: deleting a room whose call
#: is ``active`` would hang up a live call, not refuse a new one.
RINGING_STATUS = "ringing"
CALL_STATUS_ATTRIBUTE = "sip.callStatus"

#: livekit-server launches a room's agent job while it is still building the
#: room, before the caller's participant has joined it, so a refused job can
#: look at an empty room. It keeps looking this long for the ringing caller.
DISCOVERY_SECONDS = 3.0
DISCOVERY_POLL_SECONDS = 0.1

#: How long ending a refused leg may take. livekit-api sends each request with
#: no timeout of its own.
DELETE_SECONDS = 5.0

#: Where a job process marks its call ended, for the worker process to read.
#: Inside the container, one per host: job ids are unique, so workers that
#: share it cannot confuse each other's jobs.
ENDED_CALLS_DIR = Path(tempfile.gettempdir()) / "skysay-phone-ended-calls"

#: A mark this old no longer ends its job's call: a process still running a
#: minute after its call ended (15 s at most on opx-voice) is wedged, and
#: holds its slot again until it exits.
ENDED_TAIL_SECONDS = 60.0

#: A mark for a job this worker is not running is removed only once it is this
#: old: it may be another worker's on the same /tmp (a development machine),
#: whose process is still in its teardown tail (15 s at most on opx-voice).
ENDED_MARK_SWEEP_SECONDS = 600.0


class EndedCalls:
    """The jobs whose call has ended while their process finishes its teardown.

    The call runs in a job process and the count in the worker process, and
    livekit-agents 1.8.1 carries nothing of ours between them, so the job
    process leaves an empty file named for its job id. A job whose mark is
    missing -- it could not be written, or the room never reported its
    disconnect -- counts until its process exits, as before: the fault is
    always a slot held too long, never one given away. So does a job with a
    mark from before this worker admitted it (left by a crash, or by a
    previous worker handed the same id): admission removes it (``clear``),
    so no clock step can make it look new, and only a mark written since,
    less than ``ENDED_TAIL_SECONDS`` ago, counts.
    """

    def __init__(self, directory: str | os.PathLike[str] = ENDED_CALLS_DIR) -> None:
        self._dir = Path(directory)

    def mark(self, job_id: str) -> None:
        """In the job process: this job's call is over. Never raises."""

        name = str(job_id)
        if not _safe_name(name):
            return
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (self._dir / name).touch()
        except (OSError, ValueError):
            pass

    def clear(self, job_id: str) -> bool:
        """In the worker process, at admission: remove any mark ``job_id`` already has.

        True when no mark remains. A mark there before the job was admitted
        -- a crash's leftover, a reused id -- must never end its call, and
        removing it, rather than comparing times, does not depend on the
        clock. False (it could not be removed) and the caller never lets a
        mark end this job.
        """

        name = str(job_id)
        if not _safe_name(name):
            return True
        try:
            (self._dir / name).unlink()
        except FileNotFoundError:
            return True
        except (OSError, ValueError):
            return False
        return True

    def ended(self, since: Mapping[str, float]) -> set[str]:
        """In the worker process: the jobs in ``since`` whose mark is no older than their entry.

        ``since`` maps each running job to when this worker admitted it
        (wall clock); a mark from before that is someone else's, or stale.

        A mark for any other job belongs to a process that has exited, or to
        another worker sharing this directory; it is removed once it is
        ``ENDED_MARK_SWEEP_SECONDS`` old.
        """

        try:
            names = os.listdir(self._dir)
        except OSError:
            return set()
        ended: set[str] = set()
        now = time.time()
        for name in names:
            path = self._dir / name
            try:
                written = path.stat().st_mtime
                if name in since:
                    if written >= since[name] and now - written < ENDED_TAIL_SECONDS:
                        ended.add(name)
                elif now - written >= ENDED_MARK_SWEEP_SECONDS:
                    path.unlink()
            except OSError:
                pass
        return ended


def _safe_name(job_id: str) -> bool:
    return (
        bool(job_id)
        and not job_id.startswith(".")
        and "/" not in job_id
        and os.sep not in job_id
        and "\x00" not in job_id
    )


def is_outbound_dispatch(request: Any) -> bool:
    """Whether ``request`` is the originator's dispatch for an outbound call."""

    metadata = getattr(getattr(request, "job", None), "metadata", "") or ""
    try:
        parsed = json.loads(metadata) if isinstance(metadata, str) and metadata else {}
    except ValueError:
        return False
    return isinstance(parsed, dict) and parsed.get("schema") == OUTBOUND_DISPATCH_SCHEMA


def resolve_simulation_test_reserve(environ: Mapping[str, str]) -> tuple[int, int]:
    """``(test_slots, customer_headroom)`` from the host's configuration.

    Absent, blank or unusable values keep the design's defaults (two and two),
    never a refusal to start: a worker that will not start over a malformed
    number takes customer calls down with it. Test slots ``0..16``; the
    customers' headroom ``1..16`` (zero would remove the reserve).
    """

    def one(var: str, default: int, minimum: int) -> int:
        raw = (environ.get(var) or "").strip()
        try:
            value = int(raw) if raw else default
        except ValueError:
            return default
        return value if minimum <= value <= SIM_RESERVE_CEILING else default

    return (
        # Zero test slots is a host that runs no live-call tests at all.
        one(SIM_TEST_SLOTS_VAR, DEFAULT_SIM_TEST_SLOTS, 0),
        # Zero headroom would hand customers' last slots to tests: refused.
        one(SIM_CUSTOMER_HEADROOM_VAR, DEFAULT_SIM_CUSTOMER_HEADROOM, 1),
    )


def live_call_test_call_id(request: Any) -> str:
    """The call id of a Simulations Lab live-call test's dispatch, or ``""``.

    Structural, never a flag the sender sets: the room is ``opx-suite-`` plus
    the call the agent envelope names, and the envelope's caller identity is
    that call's (``agent_dispatch_metadata_v2``). The post-verdict lab room is
    named for its ATTEMPT, never for its call, so it is never one; neither is
    any customer room.
    """

    return live_call_test_from(
        str(getattr(getattr(request, "room", None), "name", "") or ""),
        getattr(getattr(request, "job", None), "metadata", "") or "",
    )


def live_call_test_from(room: str, metadata: Any) -> str:
    """``live_call_test_call_id`` over the room name and dispatch metadata themselves.

    The job process asks the same question of its own job, so the branch it
    reports when it claims its session is the branch its admission took.
    """
    if not room.startswith(SUITE_ROOM_PREFIX):
        return ""
    try:
        parsed = json.loads(metadata) if isinstance(metadata, str) and metadata else {}
    except ValueError:
        return ""
    if not isinstance(parsed, dict) or parsed.get("schema") != BROWSER_TEST_DISPATCH_SCHEMA:
        return ""
    call_id = str(parsed.get("call_id") or "")
    if (
        not call_id
        or room != SUITE_ROOM_PREFIX + call_id
        or parsed.get("participant_identity") != LIVE_CALL_CALLER_PREFIX + call_id
    ):
        return ""
    return call_id


def with_capacity_refusal(metadata: str, job_id: str) -> str | None:
    """``metadata`` with the capacity-refusal marker merged in, every other key kept.

    None when the existing metadata is not a JSON object: it is not the
    originator's, and is left alone (the originator then fails the call on its
    join wait, as it did before the marker existed).
    """

    try:
        current = json.loads(metadata) if metadata else {}
    except ValueError:
        return None
    if not isinstance(current, dict):
        return None
    return json.dumps({**current, CAPACITY_REFUSED_KEY: str(job_id)}, sort_keys=True)


def watch_call_end(room: Any, job_id: str, ended_calls: EndedCalls) -> Callable[[], None]:
    """In the job process: mark the call ended when ``room`` disconnects.

    Returns the mark, for the entrypoint's own return to call as well (a room
    whose disconnect never arrives). Neither raises.
    """

    def ended(*_: Any) -> None:
        ended_calls.mark(job_id)

    room.on("disconnected", ended)
    return ended


class PhoneAdmission:
    """An exact count of the phone jobs this worker holds, against a capacity."""

    def __init__(
        self,
        *,
        capacity: int,
        active_jobs: Callable[[], Iterable[tuple[str, str]]],
        ended_calls: Callable[[Mapping[str, float]], Iterable[str]] | None = None,
        wall_clock: Callable[[], float] = time.time,
        process_cap: int | None = None,
        clear_mark: Callable[[str], bool] | None = None,
        test_slots: int = DEFAULT_SIM_TEST_SLOTS,
        customer_headroom: int = DEFAULT_SIM_CUSTOMER_HEADROOM,
    ) -> None:
        if capacity < 1:
            raise ValueError("phone capacity must be at least 1")
        if test_slots < 0 or customer_headroom < 1:
            raise ValueError("the simulation test reserve keeps at least one slot for customers")
        #: Lab rooms this worker may hold at once, and the slots and processes
        #: a live-call test must leave free for customers (design F4).
        self._test_slots = int(test_slots)
        self._customer_headroom = int(customer_headroom)
        self._refused_tests = 0
        self._capacity = int(capacity)
        #: The most job PROCESSES this worker runs at once, ended calls still
        #: finishing their evidence included: live calls on the same CPU pay
        #: for every teardown's mix. None: no ceiling beyond the capacity.
        self._process_cap = None if process_cap is None else int(process_cap)
        self._active_jobs = active_jobs
        self._ended_calls = ended_calls
        self._wall_clock = wall_clock
        self._clear_mark = clear_mark
        #: Jobs whose leftover mark could not be removed at admission: no
        #: mark ever ends their call; they hold their slot until they exit.
        self._untrusted: set[str] = set()
        #: When this worker admitted each job it may still hold (wall clock):
        #: only a mark written after it can end that job's call.
        self._admitted_at: dict[str, float] = {}
        #: Accepts in flight: job id -> its room.
        self._accepting: dict[str, str] = {}
        #: Rooms a refusal has deleted or is deleting: a job admitted into one
        #: would be hung up by that delete.
        self._deleting: set[str] = set()
        self._refused = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def refused(self) -> int:
        return self._refused

    @property
    def refused_tests(self) -> int:
        """Live-call tests refused to keep the customers' headroom (of ``refused``)."""

        return self._refused_tests

    def _held(self) -> dict[str, str] | None:
        """The held jobs (id -> room): accepted, call not ended. None: unreadable."""

        pool = self._active()
        if pool is None:
            return None
        active = set(pool)
        rooms = {**pool, **self._accepting}
        return {job: rooms[job] for job in set(self._accepting) | (active - self._ended(active))}

    def held(self) -> int:
        """Calls this worker holds: jobs accepted whose call has not ended."""

        held = self._held()
        return self._capacity if held is None else len(held)

    def tests_held(self) -> int:
        """Lab rooms among the calls held (any ``opx-suite-`` room, a post-verdict one included)."""

        held = self._held()
        if held is None:
            return self._test_slots
        return sum(1 for room in held.values() if room.startswith(SUITE_ROOM_PREFIX))

    def holds_room(self, room_name: str) -> bool:
        """Whether a job this worker accepted, and whose process has not exited, is in ``room_name``.

        Every job, an ended call's included: this answers "may this room be
        deleted", not "is there a slot".
        """

        pool = self._active()
        if pool is None:
            return True
        return room_name in set(self._accepting.values()) | set(pool.values())

    def processes(self) -> int:
        """Job processes this worker runs: accepted and not yet exited, ended calls included."""

        pool = self._active()
        if pool is None:
            return self._process_cap if self._process_cap is not None else self._capacity
        return len(set(self._accepting) | set(pool))

    def try_admit(self, job_id: str, room_name: str = "", *, live_call_test: bool = False) -> bool:
        """Claim a slot for ``job_id``, or refuse it. No ``await`` inside.

        Refused when the calls held reach the capacity, when the job processes
        (ended calls still in their teardown included) reach ``process_cap``,
        or when a refusal is deleting the room. A live-call test is refused
        as well unless it leaves the customers' headroom free (``_test_fits``).
        """

        if (
            self.held() >= self._capacity
            or (self._process_cap is not None and self.processes() >= self._process_cap)
            or (room_name and room_name in self._deleting)
        ):
            self._refused += 1
            return False
        if live_call_test and not self._test_fits():
            self._refused += 1
            self._refused_tests += 1
            return False
        self._accepting[str(job_id)] = str(room_name)
        self._admitted_at[str(job_id)] = self._wall_clock()
        if self._clear_mark is not None:
            try:
                cleared = bool(self._clear_mark(str(job_id)))
            except Exception:  # noqa: BLE001 - an uncleared mark only holds the slot
                cleared = False
            if not cleared:
                self._untrusted.add(str(job_id))
        return True

    def _test_fits(self) -> bool:
        """Design 8.1: a test slot is free, and the slots and processes left free stay free.

        ``held() + 1 <= capacity - headroom`` and ``processes() + 1 <=
        process_cap - headroom``; a worker with no process cap measures its
        processes against the capacity.
        """

        process_cap = self._process_cap if self._process_cap is not None else self._capacity
        return (
            self.tests_held() < self._test_slots
            and self.held() + 1 <= self._capacity - self._customer_headroom
            and self.processes() + 1 <= process_cap - self._customer_headroom
        )

    def start_deleting(self, room_name: str) -> bool:
        """Claim the delete of ``room_name``; False if a refusal already has.

        From then on nothing is admitted into it, for the life of the worker:
        a room name is one call's, and a delete that failed or timed out may
        still land.
        """

        name = str(room_name)
        if name in self._deleting:
            return False
        self._deleting.add(name)
        return True

    def accepted(self, job_id: str) -> None:
        """``accept()`` returned (or failed): the job is ``active_jobs``' now."""

        self._accepting.pop(str(job_id), None)

    def _active(self) -> dict[str, str] | None:
        """The worker's running jobs (id -> room), or None when they cannot be read.

        None fails closed: the host counts as full and no room may be
        deleted. The SDK's own pool read does not raise today; if it ever
        does, calls are refused rather than admitted past the capacity.
        """

        try:
            return {str(job_id): str(room) for job_id, room in self._active_jobs()}
        except Exception:  # noqa: BLE001 - see the docstring
            return None

    def _ended(self, active: set[str]) -> set[str]:
        """The active jobs whose call has ended, by a mark written since this worker admitted them.

        A job this worker has no admission time for -- it came from a
        previous worker, or nowhere -- is never ended by a mark: a mark left
        over from before (a crash, a restarted worker, a reused id) holds a
        slot, it never frees one.
        """

        for job_id in [j for j in self._admitted_at if j not in active and j not in self._accepting]:
            del self._admitted_at[job_id]
            self._untrusted.discard(job_id)
        since = {
            job_id: self._admitted_at[job_id]
            for job_id in active
            if job_id in self._admitted_at and job_id not in self._untrusted
        }
        if self._ended_calls is None or not since:
            return set()
        try:
            return set(self._ended_calls(since)) & set(since)
        except Exception:  # noqa: BLE001 - an unreadable mark counts the job, as before
            return set()


async def handle_dispatch(
    request: Any,
    *,
    admission: PhoneAdmission,
    list_participants: Callable[[str], Awaitable[Iterable[Any]]],
    delete_room: Callable[[str], Awaitable[Any]],
    record_refusal: Callable[[Mapping[str, str], str], Awaitable[bool]],
    emit: Callable[..., None],
    spawn: Callable[[Coroutine[Any, Any, None]], Any] | None = None,
    sleep: Callable[[float], Awaitable[Any]] | None = None,
    clock: Callable[[], float] | None = None,
    discovery_seconds: float = DISCOVERY_SECONDS,
    delete_seconds: float = DELETE_SECONDS,
    mark_outbound_refused: Callable[[str, str], Awaitable[bool]] | None = None,
    report_capacity_reserved: Callable[[str, str], Awaitable[bool]] | None = None,
) -> bool:
    """Accept ``request`` while there is room; otherwise refuse it cleanly.

    ``request`` is livekit-agents' ``JobRequest`` (``id``, ``room``,
    ``accept()``, ``reject()``). Refused, the request returns at once --
    livekit-agents holds a reservation until it does -- and the refusal's
    cleanup runs on its own (``spawn``): a RINGING inbound caller is ended by
    deleting their room (livekit/sip answers a leg whose room is gone with
    486 Busy Here), and the refusal is recorded for the called number's
    workspace. An OUTBOUND dispatch refused (the originator's, not yet
    dialled) has its room marked instead (``mark_outbound_refused``), so the
    originator gives the job back to the queue. A Simulations Lab live-call
    test refused (for any reason: its headroom, or a full host) is reported
    to the control plane as ``capacity_reserved`` (``report_capacity_reserved``)
    and its room is left to the room worker that owns it: it holds no phone
    leg. Returns whether the dispatch was admitted.
    """

    job_id = str(request.id)
    room_name = str(getattr(request.room, "name", "") or "")
    test_call_id = live_call_test_call_id(request)
    if admission.try_admit(job_id, room_name, live_call_test=bool(test_call_id)):
        try:
            await request.accept()
        finally:
            admission.accepted(job_id)
        return True
    held = admission.held()
    await request.reject()
    if test_call_id:
        (spawn or _spawn)(
            _refuse_live_call_test(
                job_id, test_call_id, room_name, held, admission=admission,
                report=report_capacity_reserved, emit=emit, timeout=delete_seconds,
                sleep=sleep or asyncio.sleep,
            )
        )
        return False
    if mark_outbound_refused is not None and is_outbound_dispatch(request):
        # Nothing rings for an outbound dispatch: the originator dials only
        # after its agent joins. It is told, through the room, to queue the
        # job again.
        (spawn or _spawn)(
            _refuse_outbound(
                job_id, room_name, held, admission=admission, mark=mark_outbound_refused, emit=emit,
                timeout=delete_seconds,
            )
        )
        return False
    cleanup = _refuse(
        job_id,
        room_name,
        held,
        admission=admission,
        list_participants=list_participants,
        delete_room=delete_room,
        record_refusal=record_refusal,
        emit=emit,
        sleep=sleep or asyncio.sleep,
        clock=clock or time.monotonic,
        discovery_seconds=discovery_seconds,
        delete_seconds=delete_seconds,
    )
    (spawn or _spawn)(cleanup)
    return False


#: Refusal cleanups in flight, held so the event loop cannot collect them.
_CLEANUPS: set[asyncio.Task[None]] = set()


async def drain_refusals(timeout: float = DISCOVERY_SECONDS + DELETE_SECONDS) -> None:
    """Let the refusals in flight end their legs before the worker shuts down.

    Cut at ``timeout``; a leg left ringing ends on the trunk's ringing timeout.
    """

    pending = [task for task in _CLEANUPS if not task.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


def _spawn(cleanup: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.get_running_loop().create_task(cleanup, name="phone_capacity_refusal")
    _CLEANUPS.add(task)
    task.add_done_callback(_CLEANUPS.discard)
    return task


async def _refuse_outbound(
    job_id: str,
    room_name: str,
    held: int,
    *,
    admission: PhoneAdmission,
    mark: Callable[[str, str], Awaitable[bool]],
    emit: Callable[..., None],
    timeout: float,
) -> None:
    """Mark a refused outbound dispatch's room for its originator. Never raises."""

    outcome = "outbound_mark_failed"
    try:
        if await asyncio.wait_for(mark(room_name, job_id), timeout=timeout):
            outcome = "outbound_marked"
    except Exception as exc:  # noqa: BLE001 - a refusal must never take the worker down
        emit(REFUSAL_FAILED_EVENT, job_id=job_id, room_kind="outbound", error_type=type(exc).__name__)
    emit(
        REFUSED_EVENT,
        job_id=job_id,
        room_kind="outbound",
        capacity=admission.capacity,
        held=held,
        refused_total=admission.refused,
        refusal_outcome=outcome,
        recorded=False,
    )


async def _refuse_live_call_test(
    job_id: str,
    call_id: str,
    room_name: str,
    held: int,
    *,
    admission: PhoneAdmission,
    report: Callable[[str, str], Awaitable[bool]] | None,
    emit: Callable[..., None],
    timeout: float,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> None:
    """Tell the control plane a live-call test waits for capacity. Never raises.

    Nothing in the room is touched: it holds no phone leg, and its room worker
    deletes it. A report that does not arrive (an error or a timeout) is sent
    again, up to ``REPORT_ATTEMPTS`` in all: the refusal is the only fact the
    control plane gets about this room. A ``false`` answer is final (it is not
    a live-call open). Unreported, the attempt is left to its own deadlines.
    """

    outcome = "capacity_reserved_report_failed"
    for attempt in range(REPORT_ATTEMPTS):
        try:
            recorded = report is not None and await asyncio.wait_for(report(call_id, room_name), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - a refusal must never take the worker down
            emit(REFUSAL_FAILED_EVENT, job_id=job_id, room_kind="live_call_test", error_type=type(exc).__name__)
            if attempt + 1 < REPORT_ATTEMPTS:
                await sleep(REPORT_RETRY_SECONDS * (attempt + 1))
            continue
        if recorded:
            outcome = "capacity_reserved_reported"
        break
    emit(
        REFUSED_EVENT,
        job_id=job_id,
        room_kind="live_call_test",
        capacity=admission.capacity,
        held=held,
        refused_total=admission.refused,
        refusal_outcome=outcome,
        recorded=outcome == "capacity_reserved_reported",
    )


async def _refuse(
    job_id: str,
    room_name: str,
    held: int,
    *,
    admission: PhoneAdmission,
    list_participants: Callable[[str], Awaitable[Iterable[Any]]],
    delete_room: Callable[[str], Awaitable[Any]],
    record_refusal: Callable[[Mapping[str, str], str], Awaitable[bool]],
    emit: Callable[..., None],
    sleep: Callable[[float], Awaitable[Any]],
    clock: Callable[[], float],
    discovery_seconds: float,
    delete_seconds: float,
) -> None:
    """End a refused dispatch's ringing inbound leg, and record it. Never raises."""

    outcome = "no_inbound_caller"
    room_kind = "unknown"
    recorded = False
    try:
        found = await _find_ringing_caller(
            room_name, list_participants, sleep, clock, discovery_seconds
        )
        room_kind = (
            "no_inbound_caller" if found is None else "occupied" if found[0] is None else "inbound_ringing"
        )
        if found is not None:
            caller, identity = found
            # Checked again with no await before the delete is issued: a job
            # this worker accepted into the same room -- the call's own
            # dispatch, taken while this duplicate was being refused -- is a
            # call about to be answered, and deleting its room would hang it up.
            if caller is None or admission.holds_room(room_name):
                outcome = "live_call_left_alone"
            elif not admission.start_deleting(room_name):
                # Another refusal of this room is ending the leg, and records it.
                outcome = "leg_already_ending"
            else:
                # Claimed before the first await: a dispatch into this room
                # admitted while the delete is in flight would be hung up by it.
                outcome = "leg_end_failed"
                await asyncio.wait_for(delete_room(room_name), timeout=delete_seconds)
                outcome = "leg_ended"
                recorded = bool(await record_refusal(dict(caller), identity))
    except Exception as exc:  # noqa: BLE001 - a refusal must never take the worker down
        # Never the room's name: an inbound room is named for the caller's
        # number. The job id joins to livekit-server's own log.
        emit(
            REFUSAL_FAILED_EVENT,
            job_id=job_id,
            room_kind=room_kind,
            error_type=type(exc).__name__,
        )
    emit(
        REFUSED_EVENT,
        job_id=job_id,
        room_kind=room_kind,
        capacity=admission.capacity,
        held=held,
        refused_total=admission.refused,
        refusal_outcome=outcome,
        recorded=recorded,
    )


async def _find_ringing_caller(
    room_name: str,
    list_participants: Callable[[str], Awaitable[Iterable[Any]]],
    sleep: Callable[[float], Awaitable[Any]],
    clock: Callable[[], float],
    discovery_seconds: float,
) -> tuple[Mapping[str, str] | None, str] | None:
    """Wait, at most ``discovery_seconds`` in all, for the room's inbound caller.

    ``None`` when no inbound caller appeared in time (an outbound room, or a
    caller that never joined). ``(None, "")`` when the room holds a call
    this refusal must not end: an answered inbound leg, or an agent. Else the
    ringing caller's attributes and identity.
    """

    deadline = clock() + discovery_seconds
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        try:
            participants = list(
                await asyncio.wait_for(list_participants(room_name), timeout=remaining)
            )
        except asyncio.TimeoutError:
            return None
        if any(getattr(p, "kind", None) == AGENT_PARTICIPANT_KIND for p in participants):
            return None, ""
        for participant in participants:
            attributes = getattr(participant, "attributes", None) or {}
            if getattr(participant, "kind", None) != SIP_PARTICIPANT_KIND or not attributes.get(
                INBOUND_RULE_ATTRIBUTE
            ):
                continue
            if attributes.get(CALL_STATUS_ATTRIBUTE) != RINGING_STATUS:
                return None, ""
            return attributes, str(getattr(participant, "identity", "") or "")
        await sleep(min(DISCOVERY_POLL_SECONDS, max(0.0, deadline - clock())))


def prove_sdk_admission(server_class: Any, job_request_class: Any, idle_processes: int) -> None:
    """Assert the PINNED SDK hands every dispatch to ``request_fnc`` under zero load.

    Run at image build (Dockerfile), offline, the phone twin of
    ``browser_demo.prove_dispatch_admission``. With the worker reporting no
    load, the SDK's own ``_is_available`` must accept every dispatch while a
    job runs -- so admission is this module's -- and from cold exactly
    ``idle_processes`` simultaneous dispatches. And ``JobRequest`` must still
    be the shape ``handle_dispatch`` drives.
    """

    for name in ("id", "job", "room", "accept", "reject"):
        if not hasattr(job_request_class, name):
            raise AssertionError(f"the pinned SDK JobRequest no longer has {name!r}")
    idle = max(1, int(idle_processes))

    class _Probe(server_class):  # type: ignore[misc, valid-type]
        probe_active = 0

        @property
        def active_jobs(self) -> list[Any]:  # type: ignore[override]
            return [object()] * self.probe_active

    server = _Probe(num_idle_processes=idle, load_fnc=lambda: 0.0)
    server._devmode = False
    server._worker_load = 0.0
    for active in range(0, idle + 3):
        for reserved in range(0, idle + 3):
            server.probe_active = active
            server._reserved_slots = reserved
            expected = active > 0 or reserved < idle
            available = server._is_available()
            if available != expected:
                raise AssertionError(
                    f"SDK admission drifted: active={active} reserved={reserved} "
                    f"available={available} expected={expected}"
                )


def active_job_rooms(worker: Any) -> list[tuple[str, str]]:
    """The jobs livekit-agents counts as running on ``worker``: (job id, room name)."""

    return [(str(info.job.id), str(info.job.room.name)) for info in worker.active_jobs]


__all__ = [
    "DELETE_SECONDS",
    "ENDED_CALLS_DIR",
    "ENDED_TAIL_SECONDS",
    "EndedCalls",
    "CAPACITY_REFUSED_KEY",
    "CAPACITY_RESERVED",
    "SUITE_ROOM_PREFIX",
    "INBOUND_RULE_ATTRIBUTE",
    "OUTBOUND_DISPATCH_SCHEMA",
    "PhoneAdmission",
    "REFUSAL_FAILED_EVENT",
    "REFUSAL_OUTCOMES",
    "ROOM_KINDS",
    "REFUSED_EVENT",
    "SIP_PARTICIPANT_KIND",
    "active_job_rooms",
    "drain_refusals",
    "handle_dispatch",
    "is_outbound_dispatch",
    "live_call_test_call_id",
    "live_call_test_from",
    "resolve_simulation_test_reserve",
    "prove_sdk_admission",
    "watch_call_end",
    "with_capacity_refusal",
]

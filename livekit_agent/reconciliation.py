# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Crash-safe reconciliation for LiveKit media sessions.

The worker persists only opaque call/room identifiers and lifecycle times.
After a restart, the reconciler closes sessions whose room is no longer
active.  Billing is finalized at the last gateway heartbeat, never at the
later reconciliation wall-clock time.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - exercised via the package import in tests
    from .control_plane import (
        ARTIFACT_UPLOAD_ABANDONED,
        ARTIFACT_UPLOAD_PENDING,
        AUTHORITATIVE_TERMINAL_BILLING_STATUSES,
        CANARY_SESSION_PURPOSE,
        next_artifact_attempt_at,
    )
except ImportError:  # copied into /app in the pinned sandbox image
    from control_plane import (  # type: ignore[no-redef]
        ARTIFACT_UPLOAD_ABANDONED,
        ARTIFACT_UPLOAD_PENDING,
        AUTHORITATIVE_TERMINAL_BILLING_STATUSES,
        CANARY_SESSION_PURPOSE,
        next_artifact_attempt_at,
    )

try:  # pragma: no cover - exercised via the package import in tests
    from .session_state_file import (
        SessionFileUnreadable,
        locked_session_file,
        publish_session_file,
        read_session_file,
    )
except ImportError:  # copied into /app in the pinned sandbox image
    from session_state_file import (  # type: ignore[no-redef]
        SessionFileUnreadable,
        locked_session_file,
        publish_session_file,
        read_session_file,
    )


# The control plane deliberately remaps a reconciliation sweep's canonical
# type to ``call.failed`` when the SIP layer already reported this call
# failed (see call_lifecycle.py: "a reconciliation sweep after a terminal SIP
# failure is evidence that cleanup completed, not proof the failed call
# became billable"). That is a second legitimate close, not a rejection.
_ACCEPTED_RECONCILED_CANONICAL_TYPES = frozenset({"call.ended", "call.failed"})

#: A status this sweep will never look at again. ``complete`` and
#: ``reconciled`` are successes; ``unreconcilable`` is the sweep's own verdict
#: that this record can never be closed, written once so the next cycle does
#: not spend a control-plane request rediscovering it.
TERMINAL_SESSION_STATUSES = frozenset({"complete", "reconciled", "unreconcilable"})

#: HTTP statuses that mean "the control plane has no such call". The lifecycle
#: route answers an unknown call id with 404, and missing and foreign are
#: deliberately the same answer there, so a session file whose id is not a call
#: row -- a platform release canary, or a record left by a call the control
#: plane no longer has -- is 404 forever. Treating that as transient cost a
#: stuck record two to three control-plane requests every five seconds for a
#: week.
#:
#: It is NOT acted on immediately. 404 answers "no such call NOW", not "no such
#: call ever": the same status covers a call whose row this deployment cannot
#: currently see, and abandoning one of those would leave a real call unsettled
#: and unbilled. So a 404 ends the attempts only once the record has been
#: failing for longer than the grace window below -- long enough to outlast a
#: control-plane deploy or a misrouted gateway identity, short enough that a
#: genuine orphan stops within one release cycle instead of a week.
TERMINAL_REPORT_STATUS_CODES = frozenset({404, 410})
#: How long a record must have been failing before a "no such call" answer is
#: believed to be permanent.
TERMINAL_REPORT_GRACE_SECONDS = 900

#: How many consecutive failures a session may cost at the full sweep rate
#: before it is retried slowly instead. This is a RATE limit, never a licence
#: to stop: the reconciled event is what finalizes a call's billing, so a
#: record abandoned here is a real call that is never settled. A control plane
#: mid-deploy answers 502 for a minute or two, which at a five-second cycle is
#: far more than a handful of failures -- giving up on that count would forfeit
#: the billing of every call that crashed just before the deploy.
MAX_FAST_RECONCILE_ATTEMPTS = 5
#: How long a session that has exhausted its fast attempts waits before the
#: next try. One poisoned record then costs one request an hour rather than
#: seven hundred, and a genuine outage still settles once it ends.
SLOW_RECONCILE_RETRY_SECONDS = 3600

#: How long the far end must have been observably gone before this sweep will
#: close a call whose worker is still alive and heartbeating.
#:
#: This bounds the OTHER failure mode: a worker that decided the call
#: was over and then never tore down. Its heartbeat task is independent of the
#: parked coroutine, so it kept reporting for thirty minutes after the caller
#: left; the room stayed "active" because the ended agent's own participant
#: lingered in it. Neither of the two signals this sweep had could see it.
#:
#: The bound sits above a healthy worker's own teardown deadline, so a merely
#: SLOW worker is never overtaken and robbed of the chance to report its real
#: reason. Derivation from sch812-voice, read off the worker's own bounds
#: rather than measured: terminal egress drain 0.8s, outer drain guard 2s, then
#: a room delete and a terminal lifecycle POST on 10s HTTP timeouts each, so a
#: healthy worst case is on the order of 30 seconds. Two minutes clears that
#: comfortably while bounding the leak to a small fraction of the thirty
#: minutes this incident cost. Firm this up when the worker-side fix lands.
WORKER_TEARDOWN_GRACE_SECONDS = 120

#: Where the worker records WHY it ended the call, written at the instant its
#: arbiter decides rather than at teardown, so it survives a teardown that never
#: runs. Kept identical to production.py's ``SESSION_REASON_FIELD``; a test
#: asserts the two agree, because importing that module here would make the
#: reconciler depend on the whole worker.
WORKER_DECIDED_REASON_FIELD = "termination_detail"
#: When the worker decided, in the same ISO-Z form as every other lifecycle
#: timestamp. Blank when this worker did not decide the ending.
WORKER_DECIDED_AT_FIELD = "decided_at"
#: What qualifies the reason the worker decided (``hold_limit``, ``during_hold``,
#: ...). Kept identical to production.py's ``SESSION_REASON_DETAIL_FIELD``; the
#: same test pins it. Without it a crash turns a retryable hold into a dead end.
WORKER_DECIDED_REASON_DETAIL_FIELD = "termination_reason_detail"
#: Whether a person was reached before the ending the worker decided. Kept
#: identical to production.py's ``SESSION_PRIOR_PERSON_FIELD`` (the same test
#: pins it). Without it a crash turns a wait that followed a real conversation
#: back into a redial.
WORKER_DECIDED_PRIOR_PERSON_FIELD = "termination_prior_person"

#: The reason this sweep attaches when it closes a call under the rule above.
#: Deliberately specific rather than generic: the record it is closing already
#: has a real reason the worker decided and failed to deliver, and the control
#: plane's precedence rule is that the last REAL reason wins. A vague or empty
#: reason here would turn a merely blank record into a confidently wrong one.
WORKER_ENDED_WITHOUT_REPORT_REASON = "worker_ended_without_terminal_event"


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("session timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


#: Every field this sweep writes, and the complete set it is allowed to write.
#: It annotates a file whose other writer is the live worker, so it persists its
#: own bookkeeping and leaves every other key at whatever the worker last put
#: there. An allowlist rather than a list of keys to skip: a field added to the
#: sweep and forgotten here simply never persists, which its own test catches
#: immediately, whereas a field forgotten on a skip-list would silently start
#: overwriting the worker.
_RECONCILER_OWNED_FIELDS = frozenset(
    {
        # The artifact-upload retry's own state machine; see
        # ``retry_pending_artifact_uploads``. The worker writes these once,
        # when its own upload fails, and never again.
        "artifact_status",
        "artifact_upload_abandoned_at",
        "artifact_upload_attempts",
        "artifact_upload_last_attempt_at",
        "artifact_upload_next_at",
        "artifact_upload_reason",
        "far_end_absent_since",
        "first_failure_at",
        "last_reconcile_attempt_at",
        "reconcile_attempts",
        "reconciled_at",
        "reconciled_reason",
        "terminal_failure_since",
        "unreconcilable_at",
        "unreconcilable_reason",
    }
)


def _write_state(
    path: Path, state: dict[str, Any], *, clear: tuple[str, ...] = ()
) -> None:
    """Merge this record into the session file, then remove exactly ``clear``.

    MERGES rather than replaces, for the same reason the worker's writer does,
    and the case here is sharper. This function reads the record at the top of a
    sweep and writes it back later. A worker whose call ends inside that window
    writes the reason it decided and the instant it decided at; a wholesale
    replace then erases both, at the one moment they exist and by the very
    component that was about to prefer them. Found by the voice worker agent,
    which confirmed it in the pushed code rather than inferring it.

    Clearing is therefore EXPLICIT. Two keys are this sweep's own bookkeeping
    and must be removable -- the caller-gone clock when the caller reappears,
    and the terminal-failure streak when a non-terminal answer breaks it -- and
    a merge cannot express a removal by omission. Naming them keeps both
    behaviours without the writer silently reverting fields it never meant to
    touch.
    """

    # The read and the replace are ONE step, for the same reason the worker's
    # writer holds this lock: on a wedged call this sweep and that worker write
    # the same file every few seconds, and a decision committed between a bare
    # read and its replace is read as absent and then written away.
    with locked_session_file(path):
        _write_state_locked(path, state, clear=clear)


def _write_state_locked(
    path: Path, state: dict[str, Any], *, clear: tuple[str, ...] = ()
) -> None:
    """The critical section of :func:`_write_state`. Never called unlocked."""

    try:
        merged: dict[str, Any] = read_session_file(path)
    except SessionFileUnreadable:
        # Seed from the SNAPSHOT, never from nothing. Publishing `{}` here wrote
        # a record carrying only this sweep's own fields -- no `room_name`, no
        # `last_heartbeat_at`, no status -- and the next cycle skips exactly
        # those (`if not all((call_id, room_name, heartbeat))`), every cycle,
        # forever. A transient read error therefore turned a live call into one
        # that is never settled and never billed, which is the precise opposite
        # of what this file exists to do.
        #
        # This does NOT reintroduce the stale-snapshot defect that made the
        # allowlist necessary. That defect was republishing a snapshot OVER good
        # on-disk data; here there is no readable on-disk data to preserve, so
        # the snapshot is the only reconstruction available and writing it is
        # strictly better than writing a husk.
        merged = dict(state) if isinstance(state, dict) else {}
    # Only this sweep's OWN fields, never the whole snapshot it read at the top
    # of the loop. Between that read and this write the sweep asks the LiveKit
    # server who is in the room -- a network round trip on a call whose worker
    # may still be alive -- and writing the snapshot back put every key the
    # worker changed in that window back to its old value. All three of the
    # things the worker can do there are ones this file exists to respect: it
    # can finish (a completed session reopened to ``media_started``, swept
    # again, and settled a second time), it can heartbeat (a fresh instant
    # replaced by a stale one, on a live call), and it can DISOWN an ending it
    # had stamped (the released reason written back, and then frozen forever,
    # because the reason is write-once).
    for key in _RECONCILER_OWNED_FIELDS:
        if key in state:
            merged[key] = state[key]
    # The status is this sweep's to write only when it is settling the record,
    # and it settles it exactly once. A non-terminal value here is the snapshot
    # talking, not a decision, and a record already terminal on disk was
    # settled by whoever got there first -- the same rule the worker's own
    # writer applies, kept identical on purpose.
    settled = str(state.get("status") or "")
    if settled in TERMINAL_SESSION_STATUSES and str(
        merged.get("status") or ""
    ) not in TERMINAL_SESSION_STATUSES:
        merged["status"] = settled
    for key in clear:
        merged.pop(key, None)
    publish_session_file(path, merged)


def _waiting_for_a_slow_retry(state: dict[str, Any], *, now: datetime) -> bool:
    """Whether this record's next attempt is not due yet.

    Fails toward TRYING: an unparsable or missing timestamp means the record is
    attempted now rather than deferred forever on a value nobody can read.
    """

    try:
        attempts = int(state.get("reconcile_attempts") or 0)
    except (TypeError, ValueError):
        return False
    if attempts < MAX_FAST_RECONCILE_ATTEMPTS:
        return False
    last = str(state.get("last_reconcile_attempt_at") or "")
    if not last:
        return False
    try:
        since = (now - _parse(last)).total_seconds()
    except ValueError:
        return False
    return since < SLOW_RECONCILE_RETRY_SECONDS


def _far_end_gone_since(
    state: dict[str, Any],
    *,
    room_name: str,
    far_end_present: Callable[[str, str], bool] | None,
    now_text: str,
    now: datetime,
) -> tuple[str | None, bool, tuple[str, ...]]:
    """When the far end was first seen gone, once that is old enough to act on.

    Returns ``(observed_at, state_changed, keys_to_clear)``. A removal has to be
    named rather than expressed by omission, because the writer merges. ``observed_at`` is the timestamp
    to reconcile and bill at, or ``None`` when this record is not -- or is not
    yet -- a worker-ended-but-never-reported case.

    The far end's own presence is the discriminator, and it is what makes this
    rule safe. A max-age bound alone cannot be: a real forty-minute call is
    answered, heartbeating and has no terminal event, exactly like the wedge,
    so any age low enough to catch the wedge would also end live calls. A live
    call still has its CALLER in the room. Asking specifically about the
    caller's identity rather than "does the room have participants" is also
    what stops the ended agent's own lingering participant from holding a dead
    call open, which is half of why the incident was invisible.
    """

    if far_end_present is None:
        return None, False, ()
    # Only a call that actually reached media. This arm exists for a call the
    # worker ANSWERED and then failed to tear down; anything earlier in the
    # lifecycle belongs to the staleness path, which still runs.
    if str(state.get("status") or "") != "media_started":
        return None, False, ()
    identity = str(state.get("participant_identity") or "")
    if not identity:
        # A record written before the worker stamped the far end's identity.
        # There is nobody to ask about, so leave it to the staleness path
        # rather than guess from the room.
        return None, False, ()

    previous = str(state.get("far_end_absent_since") or "")
    if far_end_present(room_name, identity):
        # Present NOW, so any earlier absence was transient. A participant
        # listing that momentarily missed the caller must never accumulate
        # toward the bound across cycles.
        if previous:
            state.pop("far_end_absent_since", None)
            return None, True, ("far_end_absent_since",)
        return None, False, ()

    if not previous:
        # First observation. The clock starts now and this cycle acts on
        # nothing, so a single bad listing can never close a call by itself.
        state["far_end_absent_since"] = now_text
        return None, True, ()
    try:
        gone_for = (now - _parse(previous)).total_seconds()
    except ValueError:
        # Unreadable stamp: restart the clock rather than act on a value
        # nobody can parse. Fails toward leaving the call alone.
        state["far_end_absent_since"] = now_text
        return None, True, ()
    if gone_for < WORKER_TEARDOWN_GRACE_SECONDS:
        return None, False, ()
    return previous, False, ()


def _terminal_report_failure(
    exc: BaseException, state: dict[str, Any] | None, *, now: datetime
) -> bool:
    """Whether this failure means the control plane will never accept the close.

    Read off the exception's ``status_code``, never off its message text: the
    message is a human-facing string that may be reworded, while the status is
    the contract. An exception without one (a malformed body, a client-side
    check, a transport error) is treated as transient, which is the safe
    direction.

    A qualifying status is believed only once the record has been failing for
    longer than the grace window. "No such call" is an answer about NOW, and
    the same 404 covers a call this deployment cannot currently see; ending the
    attempts on the first one would leave such a call unsettled and unbilled
    forever. A genuine orphan keeps answering 404 and stops within the window.
    """

    if getattr(exc, "status_code", None) not in TERMINAL_REPORT_STATUS_CODES:
        return False
    if not isinstance(state, dict):
        # Nowhere to record when the failure began, so nowhere to measure the
        # window from. Retry rather than abandon.
        return False
    # The streak of "no such call" answers specifically, NOT the record's whole
    # failure history. Measuring from the first failure of any kind let an
    # unrelated outage consume this window: a record answering 502 for fifteen
    # minutes during a control-plane deploy, then answering 404 once the deploy
    # finished, would be abandoned on that first 404 with none of its own grace
    # spent. A 404 arriving after a 502 is a NEW signal and starts its own
    # clock, which is what the caller records before asking.
    first = str(state.get("terminal_failure_since") or "")
    if not first:
        return False
    try:
        return (now - _parse(first)).total_seconds() >= TERMINAL_REPORT_GRACE_SECONDS
    except ValueError:
        return False


def reconcile_stale_sessions(
    root: Path,
    *,
    now: str,
    stale_after_seconds: int,
    room_is_active: Callable[[str], bool],
    report: Callable[[dict[str, Any]], dict[str, Any]],
    delete_room: Callable[[str], None],
    far_end_present: Callable[[str, str], bool] | None = None,
) -> dict[str, list[Any]]:
    reconciled: list[str] = []
    active: list[str] = []
    skipped: list[str] = []
    # A record this sweep cannot process must never wedge the others: one
    # unhandled session used to abort the whole cycle via an uncaught
    # exception, leaving every OTHER stale session unreconciled behind it and
    # retrying the same poisoned record forever. Isolate each session's work
    # so one failure is counted and reported by call id while the sweep
    # continues.
    failed: list[dict[str, str]] = []
    # Sessions this sweep has just given up on for good. Reported separately
    # from ``failed`` so a caller can log the verdict once, on the cycle that
    # reached it, instead of once per cycle forever.
    unreconcilable: list[str] = []
    # Records whose next attempt is not due yet. Counted rather than silently
    # folded into ``skipped`` so a caller can tell "nothing to do" from
    # "something is still failing, more slowly".
    deferred: list[str] = []
    # The subset of ``reconciled`` closed by the worker-ended arm rather than
    # by staleness. Counted separately because it means something different
    # operationally: a worker is alive and wedged, not crashed, so a non-zero
    # count here is a signal to go and look at the host.
    worker_ended: list[str] = []
    current = _parse(now)
    for path in sorted(root.glob("*.json")):
        call_id = path.stem
        # Bound to the parsed record so the failure handler below can write a
        # verdict back. It stays None when the file itself could not be read,
        # which is the one failure that has nowhere to record anything.
        state: dict[str, Any] | None = None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            # The artifact-upload fields are the artifact sweep's alone. This
            # snapshot is written back on every failed report, and a snapshot
            # read before the artifact sweep's last write would put its old
            # schedule back -- collapsing the backoff, or reopening an upload
            # already stored or given up on. Dropped here so this sweep never
            # carries them at all.
            state = {key: value for key, value in state.items() if not key.startswith("artifact_")}
            call_id = str(state.get("call_id") or "") or path.stem
            room_name = str(state.get("room_name") or "")
            heartbeat = str(state.get("last_heartbeat_at") or "")
            # A platform release canary is not a customer call and has no call
            # row anywhere, so there is nothing for this sweep to close and the
            # lifecycle route can only answer 404. Skipped on purpose rather
            # than on status: a canary that died mid-call is left at
            # ``media_started`` exactly like a real crashed session, and that
            # is the state three week-old orphans were stuck in.
            if str(state.get("purpose") or "") == CANARY_SESSION_PURPOSE:
                skipped.append(call_id)
                continue
            if not all((call_id, room_name, heartbeat)) or (state.get("status") in TERMINAL_SESSION_STATUSES and not state.get("browser_outcome_pending")):
                skipped.append(call_id)
                continue
            # The reason this sweep is closing the record, and the time it
            # bills at. Both change when the worker-ended arm fires, so they
            # are named here rather than inlined into the report below.
            reason = "worker_restart_orphan"
            observed_at = heartbeat
            # WHICH ARM fired, tracked separately from which reason is
            # reported. Inferring the arm from the reason was wrong the moment
            # the worker's own decided reason began overriding it: the counter
            # that tells an operator a worker is alive and wedged would read
            # zero in exactly the case where wedged workers are being swept,
            # which after the worker-side fix is the COMMON case. Caught by the
            # named pin test rather than by review.
            ended_by_backstop = False

            age = (current - _parse(heartbeat)).total_seconds()
            if age < stale_after_seconds or room_is_active(room_name):
                # Neither of this sweep's two original signals can see a worker
                # that decided the call was over and then never tore down: its
                # heartbeat task is independent of the parked coroutine so it
                # stays punctual, and the ended agent's own participant keeps
                # the room "active". Ask instead whether the CALLER is still
                # there.
                gone_since, state_changed, drop = _far_end_gone_since(
                    state,
                    room_name=room_name,
                    far_end_present=far_end_present,
                    now_text=now,
                    now=current,
                )
                if state_changed:
                    _write_state(path, state, clear=drop)
                if gone_since is None:
                    active.append(call_id)
                    continue
                reason = WORKER_ENDED_WITHOUT_REPORT_REASON
                ended_by_backstop = True
                # NOT the heartbeat. On this failure mode the heartbeat is
                # fresh and wrong -- it kept ticking for thirty minutes after
                # the caller left -- and billing is finalized at whatever this
                # sweep reports, so using it would charge the customer for
                # every minute the worker stayed wedged. The first cycle that
                # saw the caller gone is the closest honest boundary this
                # sweep has.
                observed_at = gone_since
            # Only now, once this sweep would actually attempt the record. A
            # live call that collected failures during an earlier outage is
            # ACTIVE, and reporting it as deferred would make a healthy call
            # vanish from the count a monitor reads as "calls seen".
            #
            # A record past its fast attempts is retried slowly rather than
            # abandoned. It is never given up on for a transient reason: only
            # the control plane saying the call does not exist ends the
            # attempts, because closing a session is what finalizes billing.
            if _waiting_for_a_slow_retry(state, now=current):
                deferred.append(call_id)
                continue
            # The worker's OWN decision beats anything this sweep inferred, for
            # both the reason and the billable instant. It decided at the
            # moment the call actually ended and merely failed to deliver that;
            # everything below is this sweep's best reconstruction of a fact the
            # worker already knew. Last REAL reason wins, and the real one is
            # whichever was decided rather than whichever was written last.
            #
            # This is also why the sweep never needs to arbitrate at the file
            # level: it prefers the worker's reason when reporting and writes
            # its own marker under a different key entirely, so its marker
            # cannot land on top of a decided reason.
            # Read FRESH, under the lock, immediately before composing the
            # report -- not from the snapshot taken at the top of this loop.
            # Everything between that snapshot and here is network: the room
            # probe and the participant listing. A worker whose call ends inside
            # that window has by now written the reason it decided and the
            # instant it decided at, and reporting the stale snapshot would send
            # the generic marker and the fallback instant instead. The file
            # would keep the truth while the CUSTOMER got the substitute -- and
            # since this report makes the record terminal, no later cycle
            # revisits it, so the wrong reason and the wrong billed instant are
            # final. Preferring the decided values is worth nothing if they are
            # read before the moment they exist.
            # Deliberately UNGUARDED. If this record cannot be read, the
            # exception reaches this loop's per-record handler and the record is
            # retried next cycle. Recovering to `{}` here would instead report
            # the generic marker and the fallback instant as though the worker
            # had decided nothing, when the truth is that this sweep could not
            # tell -- and the report settles the record, so that answer would be
            # final. Not knowing costs one interval; guessing costs the reason
            # and the billed instant.
            with locked_session_file(path):
                fresh = read_session_file(path)
            decided_reason = str(fresh.get(WORKER_DECIDED_REASON_FIELD) or "")
            decided_at = str(fresh.get(WORKER_DECIDED_AT_FIELD) or "")
            # The detail belongs to a decided reason only: beside the generic
            # marker it would qualify an ending nobody decided.
            decided_detail = (
                str(fresh.get(WORKER_DECIDED_REASON_DETAIL_FIELD) or "") if decided_reason else ""
            )
            # Only a real boolean, and only beside a decided reason.
            decided_prior_person = fresh.get(WORKER_DECIDED_PRIOR_PERSON_FIELD) if decided_reason else None
            if not isinstance(decided_prior_person, bool):
                decided_prior_person = None
            if decided_reason:
                reason = decided_reason
            if decided_at:
                observed_at = decided_at

            result = report({
                "event_id": f"{call_id}:reconciled:{observed_at}",
                "call_id": call_id,
                "event_type": "session.reconciled",
                "source_component": "livekit_reconciler",
                "observed_at": observed_at,
                "details": {
                    "room_name": room_name,
                    "reason": reason,
                    **({"reason_detail": decided_detail} if decided_detail else {}),
                    **({"person_reached": decided_prior_person} if decided_prior_person is not None else {}),
                },
                **({"browser_outcome": fresh["browser_outcome"]} if fresh.get("browser_outcome_pending") and fresh.get("browser_outcome") else {}),
            })
            canonical_type = result.get("canonical_type") if isinstance(result, dict) else None
            billing = result.get("billing") if isinstance(result, dict) else None
            if canonical_type not in _ACCEPTED_RECONCILED_CANONICAL_TYPES or not isinstance(
                billing, dict
            ):
                raise RuntimeError("control plane did not finalize reconciled session billing")
            if billing.get("status") not in AUTHORITATIVE_TERMINAL_BILLING_STATUSES:
                raise RuntimeError("reconciled session has no authoritative billing verdict")
            if fresh.get("browser_outcome_pending") and billing.get("status") == "awaiting_usage":
                raise RuntimeError("browser outcome still awaits its financial winner")
            delete_room(room_name)
            state["status"] = "reconciled"
            state["reconciled_at"] = now
            # Which arm closed it, so a record found later says why it was
            # swept rather than leaving the reader to infer it from the shape.
            state["reconciled_reason"] = reason
            _write_state(path, state, clear=("browser_outcome_pending",))
            reconciled.append(call_id)
            if ended_by_backstop:
                worker_ended.append(call_id)
        except Exception as exc:
            failure = {
                "call_id": call_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            attempts = 0
            # Keys this handler removes, named so the merging writer can drop
            # them; a merge cannot express a removal by omission.
            cleared: tuple[str, ...] = ()
            if isinstance(state, dict):
                try:
                    attempts = int(state.get("reconcile_attempts") or 0)
                except (TypeError, ValueError):
                    attempts = 0
            attempts += 1
            # ONLY a terminal answer ends the attempts. A slow retry is what
            # bounds everything else; see the constants above for why a count
            # must not.
            if isinstance(state, dict):
                if not str(state.get("first_failure_at") or ""):
                    state["first_failure_at"] = now
                # Start or end the "no such call" streak the grace window
                # measures. A non-terminal answer ENDS it, so an outage cannot
                # spend the window a later 404 is entitled to.
                if getattr(exc, "status_code", None) in TERMINAL_REPORT_STATUS_CODES:
                    if not str(state.get("terminal_failure_since") or ""):
                        state["terminal_failure_since"] = now
                else:
                    state.pop("terminal_failure_since", None)
                    cleared = ("terminal_failure_since",)
            give_up = _terminal_report_failure(exc, state, now=current)
            if isinstance(state, dict):
                state["reconcile_attempts"] = attempts
                state["last_reconcile_attempt_at"] = now
                if give_up:
                    state["status"] = "unreconcilable"
                    state["unreconcilable_at"] = now
                    # The verdict has to say WHY, or a record found later is
                    # just as unexplained as the loop it replaced.
                    state["unreconcilable_reason"] = (
                        f"{type(exc).__name__}: {str(exc)[:300]}"
                    )
                try:
                    _write_state(path, state, clear=cleared)
                except Exception:
                    # Recording the verdict is best effort. If the state file
                    # cannot be written the record simply retries next cycle,
                    # which is exactly the behaviour that existed before.
                    give_up = False
            if give_up:
                failure["verdict"] = "unreconcilable"
                unreconcilable.append(call_id)
            failure["attempts"] = str(attempts)
            failed.append(failure)
    return {
        "reconciled": reconciled,
        "active": active,
        "skipped": skipped,
        "failed": failed,
        "unreconcilable": unreconcilable,
        "deferred": deferred,
        "worker_ended": worker_ended,
    }


def retry_pending_artifact_uploads(
    root: Path,
    *,
    now: str,
    evidence_root: Path,
    upload: Callable[[str, Path, dict[str, Any]], dict[str, Any]],
    report_failure: Callable[[str, BaseException], Any],
) -> dict[str, list[Any]]:
    """Retry the evidence uploads workers could not finish, with bounded backoff.

    A worker whose upload failed in a way a later attempt can outlive leaves
    ``artifact_status: upload_pending`` in the session file, with the attempt
    count, when the next one is due and what it needs to rebuild the SAME
    bundle. This sweep makes each due attempt from the evidence directory the
    worker left behind (``<evidence_root>/<session file stem>``).

    Independent of the lifecycle sweep above, on purpose. The call's lifecycle
    is already settled, so this never reports it or touches ``status``; and one
    retry can move a long call's whole bundle, which must not hold up the
    five-second cycle that settles crashed calls.

    Bounded: ``ARTIFACT_RETRY_DELAYS_SECONDS`` fixes how many attempts there
    are and how far apart. A refusal no retry can change (a cap, a credential,
    a call the API does not have) ends the attempts at once. Either way the
    record then says ``upload_abandoned`` and ``report_failure`` makes it loud.
    """

    current = _parse(now)
    imported: list[str] = []
    deferred: list[str] = []
    failed: list[dict[str, str]] = []
    abandoned: list[dict[str, str]] = []
    for path in sorted(root.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # The lifecycle sweep owns unreadable records and reports them.
            continue
        if not isinstance(state, dict) or state.get("artifact_status") != ARTIFACT_UPLOAD_PENDING:
            continue
        call_id = str(state.get("call_id") or "") or path.stem
        due = str(state.get("artifact_upload_next_at") or "")
        try:
            if due and _parse(due) > current:
                deferred.append(call_id)
                continue
        except ValueError:
            pass  # An unreadable due time means "now", never "never".
        try:
            attempts = int(state.get("artifact_upload_attempts") or 1)
        except (TypeError, ValueError):
            attempts = 1
        params = state.get("artifact_upload_params")
        attempts += 1
        try:
            upload(call_id, evidence_root / path.stem, params if isinstance(params, dict) else {})
        except Exception as exc:  # noqa: BLE001 - every failure is classified below
            reason = str(getattr(exc, "reason", "") or type(exc).__name__)
            next_at = (
                next_artifact_attempt_at(attempts, current)
                if bool(getattr(exc, "retryable", False))
                else None
            )
            update: dict[str, Any] = {}
            update["artifact_upload_attempts"] = attempts
            update["artifact_upload_last_attempt_at"] = now
            update["artifact_upload_reason"] = reason
            detail = {"call_id": call_id, "reason": reason, "attempts": str(attempts)}
            clear: tuple[str, ...] = ()
            if next_at is None:
                update["artifact_status"] = ARTIFACT_UPLOAD_ABANDONED
                update["artifact_upload_abandoned_at"] = now
                clear = ("artifact_upload_next_at",)
            else:
                update["artifact_upload_next_at"] = next_at
            # Decided on the record as it is NOW, not as it was when this
            # attempt began: an operator's recovery may have stored the bundle
            # while this attempt was on the network. A failure verdict must
            # not overwrite that, nor be reported.
            try:
                with locked_session_file(path):
                    still_pending = read_session_file(path).get("artifact_status") == ARTIFACT_UPLOAD_PENDING
                    if still_pending:
                        _write_state_locked(path, update, clear=clear)
            except SessionFileUnreadable:
                # Nothing trustworthy to decide on; the next sweep reads it again.
                continue
            if not still_pending:
                continue
            if next_at is None:
                abandoned.append(detail)
                report_failure(call_id, exc)
            else:
                failed.append({**detail, "next_at": next_at})
            continue
        update = {}
        update["artifact_status"] = "imported"
        update["artifact_upload_attempts"] = attempts
        update["artifact_upload_last_attempt_at"] = now
        _write_state(path, update, clear=("artifact_upload_next_at", "artifact_upload_reason"))
        imported.append(call_id)
    return {"imported": imported, "deferred": deferred, "failed": failed, "abandoned": abandoned}

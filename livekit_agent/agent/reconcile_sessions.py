# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from livekit import api

from control_plane import (
    UrllibJsonTransport,
    report_artifact_upload_failure,
    retry_call_artifact_upload,
)
from error_tracking import capture_exception, install_error_tracking
from production import post_livekit_worker_heartbeat, worker_heartbeats_enabled
from reconciliation import reconcile_stale_sessions, retry_pending_artifact_uploads
from usage_report_outbox import DeliveredReportPruner, retry_usage_reports


ROOT = Path(os.environ.get("SCH236_SESSION_ROOT") or "/sessions")
#: Where the workers leave each call's evidence directory. The release hook
#: mounts it read-only into this container; without it every retry fails as
#: ``evidence_unavailable`` and is abandoned loudly, never silently.
EVIDENCE_ROOT = Path(os.environ.get("SCH236_CALL_EVIDENCE_ROOT") or "/call-evidence")
#: How often the artifact retry sweep looks for a due upload. Its own backoff
#: starts at one minute, so looking more often than this buys nothing.
ARTIFACT_RETRY_INTERVAL_SECONDS = 30


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reconcile_error_line(payload: dict[str, str]) -> str:
    """Render one ``SCH236_RECONCILE_ERROR`` log line.

    Kept pure (no I/O) so the exact shape a poisoned session or a cycle-level
    failure produces is unit-testable without a live LiveKit API or event
    loop. Every caller must supply ``error_type``/``error_message`` at
    minimum; a per-session failure also supplies ``call_id``.
    """
    return "SCH236_RECONCILE_ERROR " + json.dumps(payload, sort_keys=True)


def _browser_reconcile_warning_line(payload: dict[str, Any]) -> str:
    """Expose deferred poison rows/backlog instead of discarding the response."""

    return "BROWSER_SESSION_RECONCILE_WARNING " + json.dumps(payload, sort_keys=True)


def _room_already_gone(exc: BaseException) -> bool:
    """Only the server's precise missing-room response makes cleanup a no-op."""
    return isinstance(exc, api.TwirpError) and (
        exc.code == "not_found"
        or (exc.code == "unknown" and exc.message == "requested room does not exist")
    )


def _platform_media_token() -> str:
    return str(os.environ.get("SKYSAY_PLATFORM_MEDIA_TOKEN") or "").strip()


def _control_plane_transport() -> UrllibJsonTransport:
    """Skysay's media worker token on the hosted box; on a self-hosted gateway,
    which never holds that credential, the deployment's own gateway token."""
    return UrllibJsonTransport(
        base_url=os.environ["SKYSAY_CONTROL_PLANE_URL"],
        bearer_token=_platform_media_token() or os.environ["SKYSAY_GATEWAY_API_TOKEN"],
    )


async def _sweep_browser_sessions(transport: UrllibJsonTransport) -> None:
    try:
        browser_verdict = await asyncio.to_thread(
            transport.request,
            "POST",
            "/v1/platform-media/browser-sessions/reconcile",
            {"limit": 100},
        )
        if browser_verdict.get("failed_count") or browser_verdict.get("has_more"):
            print(
                _browser_reconcile_warning_line(
                    {
                        "examined_count": int(
                            browser_verdict.get("examined_count") or 0
                        ),
                        "failed_count": int(browser_verdict.get("failed_count") or 0),
                        "has_more": bool(browser_verdict.get("has_more")),
                    }
                ),
                flush=True,
            )
    except Exception as exc:
        # This sweep repairs browser disconnects that never reached the
        # settle endpoint. It is independent from LiveKit room cleanup, so
        # a temporary API/version-skew failure must not suppress the older
        # reconciliation path.
        print(
            "BROWSER_SESSION_RECONCILE_ERROR "
            + json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            flush=True,
        )


async def reconcile_once() -> dict[str, list[Any]]:
    client = api.LiveKitAPI()
    transport = _control_plane_transport()
    try:
        # A heartbeat must never suppress reconciliation: it is release
        # evidence, whereas settlement cleanup remains the worker's job.
        worker_token = str(os.environ.get("SKYSAY_WORKER_API_TOKEN") or "").strip()
        if worker_heartbeats_enabled(os.environ) and worker_token:
            try:
                await asyncio.to_thread(
                    post_livekit_worker_heartbeat,
                    UrllibJsonTransport(
                        base_url=os.environ["SKYSAY_CONTROL_PLANE_URL"],
                        bearer_token=worker_token,
                    ),
                    os.environ,
                    process_id=os.getpid(),
                    worker_type="livekit_reconciler",
                )
            except Exception as exc:
                # Non-fatal (a heartbeat must never suppress reconciliation),
                # but the same "type name only" observability trap applies
                # here too -- carry the message and traceback as well.
                print(
                    "SCH236_RECONCILER_HEARTBEAT_ERROR "
                    + json.dumps(
                        {
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if _platform_media_token():
            # Browser sessions are the hosted box's alone.
            await _sweep_browser_sessions(transport)
        rooms = await client.room.list_rooms(api.ListRoomsRequest())
        active_agent_rooms: set[str] = set()
        # Every identity in each room, recorded from the SAME listing that
        # decides ``active_agent_rooms`` so this costs no extra API calls.
        # ``active_agent_rooms`` deliberately answers "is an AGENT here", which
        # is why it says yes about a call whose agent has already ended it and
        # merely lingers. Asking about the CALLER is a different question, and
        # it is the one that tells a wedged call from a live one.
        room_participants: dict[str, set[str]] = {}
        for room in rooms.rooms:
            try:
                participants = await client.room.list_participants(
                    api.ListParticipantsRequest(room=room.name)
                )
            except api.TwirpError as exc:
                if _room_already_gone(exc):
                    continue
                raise
            room_participants[room.name] = {
                participant.identity for participant in participants.participants
            }
            if any(
                participant.identity.startswith("agent-")
                or participant.identity.startswith("AJ_")
                for participant in participants.participants
            ):
                active_agent_rooms.add(room.name)
        delete_after_settlement: list[str] = []
        verdict = await asyncio.to_thread(
            reconcile_stale_sessions,
            ROOT,
            now=utcnow(),
            stale_after_seconds=int(os.environ.get("SCH236_RECONCILE_STALE_SECONDS") or "15"),
            room_is_active=lambda room: room in active_agent_rooms,
            report=lambda payload: transport.request(
                "POST",
                f"/v1/gateway-deployments/{os.environ['SKYSAY_GATEWAY_ID']}/call-lifecycle-events",
                payload,
            ),
            delete_room=delete_after_settlement.append,
            far_end_present=lambda room, identity: identity
            in room_participants.get(room, frozenset()),
        )
        cleanup_errors: list[Exception] = []
        for room_name in delete_after_settlement:
            try:
                await client.room.delete_room(api.DeleteRoomRequest(room=room_name))
            except Exception as exc:
                if not _room_already_gone(exc):
                    cleanup_errors.append(exc)
        if cleanup_errors:
            raise ExceptionGroup("LiveKit room cleanup failed", cleanup_errors)
        return verdict
    finally:
        await client.aclose()


def _artifact_failure_reporter(transport: UrllibJsonTransport):
    def report(call_id: str, exc: BaseException) -> None:
        # Loud on every channel that exists: the call's own timeline (when the
        # API can take it), the error tracker, and this container's log.
        if hasattr(exc, "stage") and hasattr(exc, "total_bytes"):
            report_artifact_upload_failure(transport, call_id, exc)  # type: ignore[arg-type]
        capture_exception(
            exc,
            component="voice-reconciler",
            phase="artifact_upload_retry",
            reason=str(getattr(exc, "reason", "") or type(exc).__name__),
            retry="abandoned",
        )

    return report


def retry_artifacts_once(transport: UrllibJsonTransport) -> dict[str, list[Any]]:
    return retry_pending_artifact_uploads(
        ROOT,
        now=utcnow(),
        evidence_root=EVIDENCE_ROOT,
        upload=lambda call_id, evidence_dir, params: retry_call_artifact_upload(
            transport, call_id, evidence_dir, params
        ),
        report_failure=_artifact_failure_reporter(transport),
    )


def report_missing_evidence_mount() -> bool:
    """Say loudly, once at start, when the evidence directory is not mounted.

    Without it every retry fails as ``evidence_unavailable`` and is abandoned
    after the last backoff step, so a host that missed the release hook's
    mount would lose exactly the calls this sweep exists to save. Returns
    whether the mount is present.
    """
    if EVIDENCE_ROOT.is_dir():
        return True
    print(
        "SCH1088_ARTIFACT_RETRY_EVIDENCE_UNMOUNTED "
        + json.dumps({"evidence_root": str(EVIDENCE_ROOT)}, sort_keys=True),
        flush=True,
    )
    capture_exception(
        RuntimeError("call-evidence directory is not mounted into the reconciler"),
        component="voice-reconciler",
        phase="artifact_upload_retry",
        reason="evidence_unmounted",
    )
    return False


async def artifact_retry_loop() -> None:
    """Retry failed evidence uploads, apart from the lifecycle sweep.

    One retry may move a long call's whole bundle; it must never delay the
    five-second cycle that settles crashed calls, so it runs as its own task.
    """
    transport = _control_plane_transport()
    report_missing_evidence_mount()
    while True:
        try:
            verdict = await asyncio.to_thread(retry_artifacts_once, transport)
            if verdict["imported"] or verdict["failed"] or verdict["abandoned"]:
                print(
                    "SCH1088_ARTIFACT_RETRY "
                    + json.dumps(
                        {key: verdict[key] for key in ("imported", "failed", "abandoned")},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as exc:
            capture_exception(exc, component="voice-reconciler", phase="artifact_upload_retry")
            print(
                "SCH1088_ARTIFACT_RETRY_ERROR "
                + json.dumps({"error_type": type(exc).__name__}, sort_keys=True),
                flush=True,
            )
        await asyncio.sleep(ARTIFACT_RETRY_INTERVAL_SECONDS)


async def usage_retry_loop() -> None:
    # Started only with the platform media token (see main): the same transport rule.
    transport = _control_plane_transport()
    cursor = ""
    pruner = DeliveredReportPruner(ROOT / "usage-reports")
    try:
        while True:
            try:
                result = await asyncio.to_thread(
                    retry_usage_reports, ROOT / "usage-reports", request=transport.request,
                    after=cursor, archive_pruner=pruner,
                    alert=lambda call_id, reason: capture_exception(
                        RuntimeError(reason), component="voice-reconciler", phase="usage_retry", call_id=call_id),
                )
                cursor = result["cursor"]
            except Exception:
                capture_exception(RuntimeError("usage_retry_failed"), component="voice-reconciler", phase="usage_retry")
            await asyncio.sleep(30)
    finally:
        await asyncio.to_thread(pruner.close)


async def main() -> None:
    # Held for the life of main(): the event loop keeps only a weak reference.
    artifact_retry = asyncio.create_task(artifact_retry_loop())  # noqa: F841
    # Only the managed media worker stages usage reports (a self-hosted call's
    # usage rides its terminal lifecycle event), so only its box replays them.
    usage_retry = asyncio.create_task(usage_retry_loop()) if _platform_media_token() else None  # noqa: F841
    while True:
        try:
            verdict = await reconcile_once()
            # A session the sweep could not process is isolated inside
            # reconcile_stale_sessions and never aborts the cycle; report
            # each one by call id with its real message and traceback
            # instead of the opaque cycle-wide failure this used to be.
            for failure in verdict.get("failed", []):
                print(
                    _reconcile_error_line(
                        {
                            "call_id": failure.get("call_id", ""),
                            "error_type": failure.get("error_type", ""),
                            "error_message": failure.get("error_message", ""),
                            "traceback": failure.get("traceback", ""),
                            # How many cycles this record has now cost, and
                            # whether the sweep has stopped retrying it. A
                            # line without the verdict is one of a bounded
                            # few; a line WITH it is the last one for that
                            # call id, which is the difference between a
                            # transient blip and a stuck record.
                            "attempts": failure.get("attempts", ""),
                            "verdict": failure.get("verdict", ""),
                        }
                    ),
                    flush=True,
                )
            if verdict["reconciled"]:
                print(
                    "SCH236_RECONCILED " + json.dumps(verdict, sort_keys=True),
                    flush=True,
                )
        except Exception as exc:
            # A failure here is outside any single session (LiveKit API
            # unreachable, a missing required env var, ...) and still aborts
            # the cycle; it must still be diagnosable from its own line.
            #
            # The line goes to stdout, where only somebody already reading
            # this container's logs will see it; the capture is what turns a
            # reconciler that has been failing every five seconds for a day
            # into something anyone is told about. Both, not either: the line
            # carries the full traceback, the report is scrubbed.
            capture_exception(exc, component="voice-reconciler")
            print(
                _reconcile_error_line(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                flush=True,
            )
        await asyncio.sleep(5)


if __name__ == "__main__":
    install_error_tracking(component="voice-reconciler")
    asyncio.run(main())

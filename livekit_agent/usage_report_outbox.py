# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Durable numeric reports on the voice host's shared local session volume.

The API owns idempotent settlement. This queue only replays an immutable report;
refused diagnostics have their own directory and can never become billable usage.
Local disk loss is outside its durability boundary.
"""
from __future__ import annotations

import fcntl
import hashlib
import heapq
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

_KINDS = {"usage": "billing-usage", "refused": "refused-metering"}
_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_MAX_BYTES = 131072
_RETRY_SECONDS = (60, 300, 900, 3600, 14400)


def _json(value: Any) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > _MAX_BYTES:
        raise ValueError("usage report exceeds bounded outbox")
    return data


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _publish(path: Path, value: dict[str, Any]) -> None:
    data = _json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _locked(path: Path):
    # One local host kernel, shared bind mount. Only tiny disk operations under
    # this lock; transport runs outside it. Replays racing on the network are
    # safe through the API's existing fingerprint + ledger transaction.
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / ".outbox.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _archived(path: Path) -> Path:
    return path.parent.parent / "archive" / path.parent.name / path.name


def _archive(path: Path) -> None:
    target = _archived(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    for parent in (path.parent, target.parent):
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("usage report exceeds bounded outbox")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("usage report record must be an object")
    return value


class UsageReportOutbox:
    def __init__(self, root: Path, call_id: str, *, alert: Callable[[str], None]) -> None:
        if not _ID.fullmatch(call_id):
            raise ValueError("invalid usage report call id")
        self.root, self.call_id, self.alert = root, call_id, alert

    def _path(self, kind: str) -> Path:
        if kind not in _KINDS:
            raise ValueError("invalid usage report kind")
        return self.root / kind / f"{self.call_id}.json"

    def stage(self, kind: str, payload: dict[str, Any]) -> str:
        """Persist before transport; failure must not prevent the live send."""
        try:
            digest = hashlib.sha256(_json(payload)).hexdigest()
            path = self._path(kind)
            with _locked(path):
                existing = path if path.exists() else _archived(path)
                if existing.exists():
                    previous = _read(existing)
                    if previous.get("digest") != digest:
                        raise ValueError("usage report changed after staging")
                    return digest
                _publish(path, {"call_id": self.call_id, "kind": kind, "digest": digest,
                                "payload": payload, "status": "pending", "attempts": 0,
                                "next_at": (_now() + timedelta(seconds=60)).isoformat()})
            return digest
        except Exception:
            self.alert("usage_outbox_stage_failed")
            return ""

    def acknowledge(self, kind: str, digest: str) -> None:
        if not digest:
            return
        try:
            path = self._path(kind)
            with _locked(path):
                existing = path if path.exists() else _archived(path)
                record = _read(existing)
                if record.get("digest") != digest:
                    raise ValueError("usage acknowledgement changed report")
                record["status"] = "delivered"
                record["delivered_at"] = _now().isoformat()
                _publish(path, record)
                _archive(path)
        except Exception:
            self.alert("usage_outbox_acknowledge_failed")


class DeliveredReportPruner:
    """Best-effort bounded archive cleanup; cursor survives between worker ticks.

    Visit at most limit directory entries, including abandoned/invalid files.
    A process restart starts a new harmless pass. Pending delivery does not
    depend on archive health. No sorting or full-history stat pass is required.
    """
    def __init__(self, root: Path) -> None:
        self.root = root
        self._scan = None
        self._lock = threading.Lock()

    def _paths(self):
        for kind in _KINDS:
            try:
                with os.scandir(self.root / "archive" / kind) as entries:
                    for entry in entries:
                        yield Path(entry.path)
            except FileNotFoundError:
                continue

    def close(self) -> None:
        with self._lock:
            if self._scan is not None:
                self._scan.close()
                self._scan = None

    def prune(self, *, now: datetime, alert: Callable[[str, str], None], limit: int = 50) -> int:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("archive cleanup limit must be between 1 and 100")
        cutoff = now - timedelta(days=30)
        removed = 0
        with self._lock:
            if self._scan is None:
                self._scan = self._paths()
            for _ in range(limit):
                try:
                    path = next(self._scan)
                except StopIteration:
                    self._scan = None
                    break
                except Exception:
                    self._scan.close()
                    self._scan = None
                    alert("", "usage_archive_scan_failed")
                    break
                try:
                    if path.suffix != ".json" or path.stat().st_mtime >= cutoff.timestamp():
                        continue
                    # Same lock as stage/ack; re-read after acquiring it.
                    with _locked(self.root / path.parent.name / path.name):
                        if not path.exists():
                            continue
                        record = _read(path)
                        if (record.get("status") == "delivered"
                                and datetime.fromisoformat(record["delivered_at"]) < cutoff):
                            path.unlink()
                            removed += 1
                except Exception:
                    alert(path.stem, "usage_archive_cleanup_failed")
        return removed


def retry_usage_reports(root: Path, *, request: Callable[..., Any],
                        alert: Callable[[str, str], None], now: datetime | None = None,
                        limit: int = 50, after: str = "",
                        archive_pruner: DeliveredReportPruner | None = None) -> dict[str, Any]:
    """Bound each sweep and cursor past poison rows; never invent usage."""
    current = now or _now()
    pruner = archive_pruner or DeliveredReportPruner(root)
    try:
        pruner.prune(now=current, alert=alert)
    except Exception:
        alert("", "usage_archive_cleanup_failed")
    finally:
        if archive_pruner is None:
            pruner.close()
    examined, delivered = 0, 0
    cursor = ""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("usage retry limit must be between 1 and 100")
    # Only pending files participate. Bounded memory even after an outage;
    # cursor moves past poison entries and each tick makes at most limit posts.
    paths = heapq.nsmallest(limit + 1, (
        path for kind in _KINDS for path in (root / kind).glob("*.json")
        if f"{path.parent.name}/{path.name}" > after
    ))
    for path in paths:
        key = f"{path.parent.name}/{path.name}"
        if key <= after:
            continue
        if examined >= limit:
            return {"examined": examined, "delivered": delivered, "cursor": cursor}
        examined += 1
        cursor = key
        try:
            record = _read(path)
            kind, call_id = record.get("kind"), record.get("call_id")
            if (kind != path.parent.name or kind not in _KINDS or not isinstance(call_id, str)
                    or not _ID.fullmatch(call_id) or path.stem != call_id
                    or not isinstance(record.get("payload"), dict)
                    or hashlib.sha256(_json(record["payload"])).hexdigest() != record.get("digest")):
                raise ValueError("usage outbox identity or fingerprint invalid")
            if record.get("status") != "pending":
                continue
            if datetime.fromisoformat(record["next_at"]) > current:
                continue
            try:
                result = request("POST", f"/v1/calls/{call_id}/{_KINDS[kind]}", record["payload"])
                if not isinstance(result, dict) or result.get("call_id") != call_id:
                    raise ValueError("usage response not bound to call")
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                reason = getattr(exc, "reason", "")
                # Only known authored API codes, never raw exception content.
                category = ("usage_after_settlement" if reason == "component_usage_after_settlement"
                            else "report_rejected")
                # A generic409 can mean the API failed to claim the report
                # and explicitly asks for identical replay. Only the authored
                # terminal-settlement code proves replay cannot help.
                retryable = reason != "component_usage_after_settlement" and (
                    isinstance(exc, OSError) or status in {408, 409, 425, 429}
                    or (isinstance(status, int) and status >= 500))
                with _locked(path):
                    if not path.exists():
                        continue  # A concurrent successful sender archived it.
                    latest = _read(path)
                    if latest.get("status") != "pending" or latest.get("digest") != record["digest"]:
                        continue
                    attempts = int(latest.get("attempts", 0)) + 1
                    latest["attempts"] = attempts
                    if retryable and attempts < len(_RETRY_SECONDS):
                        latest["next_at"] = (current + timedelta(seconds=_RETRY_SECONDS[attempts])).isoformat()
                    else:
                        latest["status"] = "abandoned"
                        latest["reason"] = category if not retryable else "retry_budget_exhausted"
                    _publish(path, latest)
                    if latest["status"] == "abandoned":
                        _archive(path)
                if latest["status"] == "abandoned":
                    alert(call_id, latest["reason"])
                continue
            outbox = UsageReportOutbox(root, call_id, alert=lambda reason: alert(call_id, reason))
            outbox.acknowledge(kind, record["digest"])
            delivered += 1
        except Exception:
            alert(path.stem, "usage_outbox_invalid_record")
    return {"examined": examined, "delivered": delivered, "cursor": ""}

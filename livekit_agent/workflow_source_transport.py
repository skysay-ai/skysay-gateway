# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One call-bound native-source wire, independent of spoken output turns.

Beginning a source is never retried. Only its inserting acknowledgement carries
connection permission; observing a fact cannot recover that permission.
"""
from __future__ import annotations

import asyncio
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

try:
    from .workflow_speech import SpeechRefused
    from .capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS
except ImportError:
    from workflow_speech import SpeechRefused
    from capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS

_ID = re.compile(r"[A-Za-z0-9_-]{16,96}")
_OWNER = re.compile(r"[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_DIMENSIONS = {identity: tuple(dimension for dimension, _ in fields)
               for identity, fields in SOURCE_USAGE_FIELDS.items()}
_STATUSES = {"open", "exact", "partial_unknown", "connect_unknown", "never_connected",
             "unreconciled", "orphaned_unknown"}


def _matches(pattern, value):
    return type(value) is str and pattern.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class SourceOwner:
    source_attempt_id: str
    owner_token: str = field(repr=False)
    binding_digest: str

    def __post_init__(self):
        if not (_matches(_ID, self.source_attempt_id) and _matches(_OWNER, self.owner_token)
                and _matches(_DIGEST, self.binding_digest)):
            raise SpeechRefused("source acknowledgement has no complete owned binding")


class FreshSourcePermission:
    """An invocation-local, single-consumption connection capability."""

    __slots__ = ("owner", "_nonce", "_consumed", "_lock")

    def __init__(self, owner, nonce):
        if not isinstance(owner, SourceOwner) or not _matches(_DIGEST, nonce):
            raise SpeechRefused("source acknowledgement has no fresh permission")
        self.owner, self._nonce = owner, nonce
        self._consumed, self._lock = False, threading.Lock()

    def __repr__(self):
        return "<FreshSourcePermission redacted>"

    def consume(self):
        """Call at the provider-touch boundary; no ACK retry resets this bit."""
        with self._lock:
            if self._consumed:
                raise SpeechRefused("source connection permission was already consumed")
            self._consumed = True
        return self.owner


@dataclass(frozen=True, slots=True)
class SourceCredential:
    source_attempt_id: str
    binding_digest: str
    component: str
    provider: str
    model: str
    credential_source: str
    funded_until: str
    secret: str | None = field(repr=False)

    def require_live(self):
        try:
            deadline = datetime.fromisoformat(self.funded_until.replace("Z", "+00:00"))
            if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise SpeechRefused("source credential acknowledgement has no live funding deadline") from None


def _units(component, unit, value):
    dimensions = _DIMENSIONS.get((component, unit))
    if dimensions is None or type(value) is not dict or set(value) != set(dimensions):
        raise SpeechRefused("source acknowledgement has invalid native dimensions")
    maximum = (300 if component == "stt" else 120) if unit == "audio_second" else 100_000_000
    if any(type(amount) not in ((int, float) if unit == "audio_second" else (int,))
           or not 0 <= amount <= maximum or not math.isfinite(amount) for amount in value.values()):
        raise SpeechRefused("source acknowledgement has invalid native quantities")
    return value


def _pairs(component, unit, value):
    if (type(value) is not list or not 1 <= len(value) <= 2
            or any(type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str for pair in value)
            or len({pair[0] for pair in value}) != len(value)):
        raise SpeechRefused("source acknowledgement has invalid native quantities")
    return _units(component, unit, dict(value))


def source_fact(response, source_id):
    fact = response.get("source") if type(response) is dict else None
    fields = {"source_attempt_id", "binding_digest", "status", "component", "unit", "known_units",
              "total_units", "delta_count", "metric_count", "closed", "absorbed"}
    if (type(fact) is not dict or set(fact) != fields or fact["source_attempt_id"] != source_id
            or not _matches(_DIGEST, fact["binding_digest"])
            or type(fact["component"]) is not str or type(fact["unit"]) is not str
            or type(fact["status"]) is not str or fact["status"] not in _STATUSES
            or type(fact["delta_count"]) is not int or not 0 <= fact["delta_count"] <= 128
            or type(fact["closed"]) is not bool or type(fact["absorbed"]) is not bool):
        raise SpeechRefused("source response contains an invalid obligation")
    known = _pairs(fact["component"], fact["unit"], fact["known_units"])
    count, total = fact["metric_count"], fact["total_units"]
    if fact["status"] in {"open", "orphaned_unknown"}:
        if total is not None or count is not None:
            raise SpeechRefused("open source acknowledgement contains a terminal measurement")
    else:
        total = _pairs(fact["component"], fact["unit"], total)
        epsilon = 1e-6 if fact["unit"] == "audio_second" else 0
        if (type(count) is not int or not fact["delta_count"] <= count <= 128
                or any(total[key] + epsilon < known[key] for key in known)):
            raise SpeechRefused("source terminal contradicts its native prefix")
        if fact["status"] == "exact" and (not count or count != fact["delta_count"]
                or any(abs(total[key] - known[key]) > epsilon for key in known)):
            raise SpeechRefused("source acknowledgement did not reconcile its native facts")
        if fact["status"] == "never_connected" and (
                count or fact["delta_count"] or any(known.values()) or any(total.values()) or fact["closed"]):
            raise SpeechRefused("never-connected source contains provider measurements")
    return fact


class WorkflowSourceTransport:
    def __init__(self, *, call_id, run_id, send, observe=None, allocation="workflow"):
        if (type(call_id) is not str or not call_id or not callable(send)
                or allocation not in ("workflow", "confirmation_listener")
                or (type(run_id) is not str or not run_id if allocation == "workflow" else run_id is not None)):
            raise SpeechRefused("source transport requires its admitted call allocation")
        self.allocation = allocation
        if observe is not None and not callable(observe):
            raise SpeechRefused("source transport observer is invalid")
        self.call_id, self.run_id, self._send, self._observe = call_id, run_id, send, observe

    async def _request(self, payload):
        operation = asyncio.create_task(asyncio.to_thread(self._send, payload))
        cancelled = False
        while True:
            try:
                response = await asyncio.shield(operation)
                break
            except asyncio.CancelledError:
                cancelled = True
                if operation.done():
                    try:
                        response = operation.result()
                        break
                    except BaseException:
                        raise asyncio.CancelledError from None
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
        if type(response) is not dict or response.get("call_id") != self.call_id:
            if cancelled:
                raise asyncio.CancelledError
            raise SpeechRefused("source response belongs to a different call")
        return response, cancelled

    async def begin(self, *, source_attempt_id, logical_turn_id, role, owner_token, delivery=None):
        if (not _matches(_ID, source_attempt_id) or not _matches(_ID, logical_turn_id)
                or not _matches(_OWNER, owner_token) or type(role) is not str
                or role not in ("listener", "authored", "live")
                or (role == "listener") != (delivery is None)):
            raise SpeechRefused("source requires its exact invocation and output parent")
        if self.allocation == "confirmation_listener" and (role != "listener" or delivery is not None):
            raise SpeechRefused("confirmation allocation cannot create output")
        spec = dict(source_attempt_id=source_attempt_id, logical_turn_id=logical_turn_id,
                    role=role, owner_token=owner_token, delivery=delivery)
        response, cancelled = await self._request({"command": "source-begin", "spec": spec})
        permission = None
        try:
            fact = source_fact(response, source_attempt_id)
            if self._observe is not None:
                self._observe(fact)
            raw = response.get("permission")
            if raw is not None:
                if type(raw) is not dict or set(raw) != {"lease", "nonce"} or type(raw["lease"]) is not dict:
                    raise SpeechRefused("source acknowledgement has malformed permission")
                expected = SourceOwner(source_attempt_id, owner_token, fact["binding_digest"])
                if raw["lease"] != asdict(expected) or fact["status"] != "open" or fact["delta_count"] or any(dict(fact["known_units"]).values()) or fact["closed"] or fact["absorbed"]:
                    raise SpeechRefused("source acknowledgement permission is not fresh")
                permission = FreshSourcePermission(expected, raw["nonce"])
            if cancelled and permission is not None:
                # We joined THIS invocation's original ACK. No provider was
                # entered; retire only its known source, without re-beginning.
                cleanup = asyncio.create_task(self.terminal(permission.owner, status="never_connected",
                    total_units=dict.fromkeys(dict(fact["known_units"]), 0), metric_count=0,
                    observed_connection_seconds=0, source_session_id="", measurement=None))
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
        finally:
            if cancelled:
                raise asyncio.CancelledError
        return fact, permission

    async def credentials(self, permission):
        if not isinstance(permission, FreshSourcePermission):
            raise SpeechRefused("source credential lease requires its original fresh permission")
        response, cancelled = await self._request({"command": "source-credentials", "permission": {
            "lease": asdict(permission.owner), "nonce": permission._nonce}})
        if cancelled:
            raise asyncio.CancelledError
        value = response.get("credential")
        fields = set(SourceCredential.__dataclass_fields__)
        if (type(value) is not dict or set(value) != fields
                or value["source_attempt_id"] != permission.owner.source_attempt_id
                or value["binding_digest"] != permission.owner.binding_digest
                or value["component"] not in ("stt", "tts")
                or any(type(value[key]) is not str or not value[key] or len(value[key]) > 256
                       for key in ("provider", "model", "funded_until"))
                or value["credential_source"] not in ("platform_managed", "customer_byok")
                or (value["secret"] is not None if value["credential_source"] == "platform_managed"
                    else type(value["secret"]) is not str or not value["secret"] or len(value["secret"]) > 16384)):
            raise SpeechRefused("source credential acknowledgement changed its binding")
        credential = SourceCredential(**value)
        credential.require_live()
        return credential

    async def _fact_command(self, owner, command, **fields):
        if not isinstance(owner, SourceOwner):
            raise SpeechRefused("source fact requires its acknowledged owned binding")
        payload = {"command": command, "source_lease": asdict(owner), **fields}
        for attempt in range(2):
            try:
                response, cancelled = await self._request(payload)
                break
            except (TimeoutError, ConnectionError):
                if attempt:
                    raise
        if cancelled:
            raise asyncio.CancelledError
        fact = source_fact(response, owner.source_attempt_id)
        if fact["binding_digest"] != owner.binding_digest:
            raise SpeechRefused("source fact acknowledgement changed its binding")
        if self._observe is not None:
            self._observe(fact)
        return fact

    async def delta(self, owner, *, index, units, source_session_id, metric_reference=""):
        fact = await self._fact_command(owner, "source-delta", index=index, units=units,
            source_session_id=source_session_id, metric_reference=metric_reference)
        increment = _units(fact["component"], fact["unit"], units)
        known = dict(fact["known_units"])
        epsilon = 1e-6 if fact["unit"] == "audio_second" else 0
        if type(index) is not int or fact["delta_count"] <= index or any(known[key] + epsilon < increment[key] for key in known):
            raise SpeechRefused("source metric was not durably acknowledged")
        return fact

    async def terminal(self, owner, *, status, total_units, metric_count, observed_connection_seconds, source_session_id,
                       measurement=None):
        if measurement is not None and (type(measurement) is not dict or set(measurement) != {"evidence", "seconds"}):
            raise SpeechRefused("source terminal measurement is malformed")
        fact = await self._fact_command(owner, "source-terminal", status=status, total_units=total_units,
            metric_count=metric_count, observed_connection_seconds=observed_connection_seconds,
            source_session_id=source_session_id, measurement=measurement)
        total = _units(fact["component"], fact["unit"], total_units)
        acknowledged = dict(fact["total_units"] or ())
        epsilon = 1e-6 if fact["unit"] == "audio_second" else 0
        if (fact["status"] != status or type(metric_count) is not int or fact["metric_count"] != metric_count
                or set(acknowledged) != set(total) or any(abs(acknowledged[key] - total[key]) > epsilon for key in total)):
            raise SpeechRefused("source terminal was not durably reconciled")
        return fact

    async def close(self, owner, *, connection_class, observed_connection_seconds):
        fact = await self._fact_command(owner, "source-close", connection_class=connection_class,
            observed_connection_seconds=observed_connection_seconds)
        if not fact["closed"]:
            raise SpeechRefused("source closure was not durably acknowledged")
        return fact

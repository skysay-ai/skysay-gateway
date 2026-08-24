# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded exact Soniox usage-log evidence for component settlement.

Soniox bills real-time STT and TTS in distinct token dimensions.  Its usage
logs are the provider-owned source for those quantities; LiveKit's generic
audio-duration metrics remain useful transport evidence but cannot substitute
for the provider token invoice.  This tracker attaches opaque, deterministic
per-call references to the adapter configurations and reads only the bounded
time window needed to settle that one call.

It retains the active API keys only in worker memory and never serializes
references, provider response bodies, request ids, text, or secrets to the
control plane, files, or logs.  A missing, late, malformed, over-paginated, or
model-mismatched log is an error: the normal worker path then leaves the
existing wallet hold awaiting explicit reconciliation rather than estimating
tokens from audio duration.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Mapping


SONIOX_USAGE_LOGS_URL = "https://api.soniox.com/v1/usage-logs"
USAGE_LOG_PAGE_SIZE = 1_000
USAGE_LOG_MAX_PAGES = 4
USAGE_LOG_MAX_ATTEMPTS = 3
USAGE_LOG_RETRY_SECONDS = 1.0
USAGE_LOG_TIMEOUT_SECONDS = 5.0
USAGE_LOG_WINDOW_PADDING_SECONDS = 300
USAGE_LOG_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
USAGE_LOG_READ_CHUNK_BYTES = 64 * 1024
# Provider usage rows can carry timestamps or opaque numeric fields larger
# than a token count, but no legitimate JSON integer here needs an arbitrary
# precision representation.  Decode a small fixed lexical width first; the
# per-token contract below remains the stricter 100M-unit bound.
USAGE_LOG_MAX_JSON_INTEGER_DIGITS = 18
# JSON's stdlib decoder converts decimal/exponent forms through ``float``
# rather than ``parse_int``.  Cap the full lexical token before that conversion
# too: the response may carry irrelevant numeric fields, but none need an
# arbitrarily long decimal representation in a bounded usage page.
USAGE_LOG_MAX_JSON_FLOAT_CHARS = 32
# A response body limit alone does not bound a recursive JSON parser: an
# attacker can fit millions of ``[``/``]`` characters inside four MiB. Walk
# the wire bytes first so the later stdlib decoder never sees an adversarial
# nesting depth. This is intentionally a structural preflight, not a second
# JSON parser; syntax and types still belong to ``json.loads`` below.
USAGE_LOG_MAX_JSON_NESTING = 64
# This applies to one persisted native usage dimension for one call, whether
# Soniox reports it in one request or across a bounded sequence of streams.
# Keeping the aggregate under the same contract as an individual row prevents
# a cursor page fan-in from turning many otherwise valid rows into an
# unbounded settlement quantity.
USAGE_LOG_MAX_TOKENS_PER_DIMENSION = 100_000_000

# Soniox's documented usage-log query is scoped by project/key and supports
# time/cursor pagination, not a server-side ``client_reference_id`` filter.
# Keep this scan finite even when unrelated traffic shares a managed key; an
# overflow retains the existing wallet hold for reconciliation instead of
# turning a paid call into an unbounded worker request or an estimated charge.

_DIMENSIONS = {
    "stt": ("input_audio_tokens", "output_text_tokens"),
    "tts": ("input_text_tokens", "output_audio_tokens"),
}


class SonioxUsageEvidenceError(RuntimeError):
    """Exact provider-native usage could not be obtained safely."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        # Only an eventual-consistency gap or a transient transport failure may
        # justify another paid-provider read.  A malformed or internally
        # inconsistent record is authoritative evidence that this settlement
        # cannot be trusted, so fail immediately and leave its reservation for
        # reconciliation instead of allowing a later page to hide the fault.
        self.retryable = retryable


class _RejectSonioxUsageRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never replay a worker-held Soniox key at a redirect target."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        target: str,
    ) -> urllib.request.Request:
        del message, target
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "Soniox usage-log redirect rejected",
            headers,
            response,
        )


def _open_soniox_usage_logs_without_redirect(
    request: urllib.request.Request, *, timeout: float
) -> Any:
    """Open the exact Soniox endpoint without trusting a 3xx Location."""

    return urllib.request.build_opener(_RejectSonioxUsageRedirectHandler).open(
        request, timeout=timeout
    )


def _read_bounded_response(response: Any) -> bytes:
    """Read one provider page without allowing a hostile body to exhaust RAM."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(
            min(USAGE_LOG_READ_CHUNK_BYTES, USAGE_LOG_MAX_RESPONSE_BYTES - total + 1)
        )
        if not chunk:
            return b"".join(chunks)
        if not isinstance(chunk, bytes):
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable")
        total += len(chunk)
        if total > USAGE_LOG_MAX_RESPONSE_BYTES:
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable")
        chunks.append(chunk)


def _bounded_json_integer(value: str) -> int:
    """Reject oversized JSON numbers before ``int`` allocates their digits."""

    if len(value.lstrip("-")) > USAGE_LOG_MAX_JSON_INTEGER_DIGITS:
        raise ValueError("Soniox usage log integer exceeds the parser contract")
    return int(value)


def _bounded_json_float(value: str) -> float:
    """Reject oversized or non-finite decimal tokens before float conversion."""

    if len(value) > USAGE_LOG_MAX_JSON_FLOAT_CHARS:
        raise ValueError("Soniox usage log decimal exceeds the parser contract")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Soniox usage log decimal is non-finite")
    return parsed


def _reject_json_constant(value: str) -> None:
    """The usage-log contract has no legitimate NaN/Infinity value."""

    raise ValueError(f"Soniox usage log JSON constant is invalid: {value}")


def _assert_bounded_json_nesting(raw: bytes) -> None:
    """Reject structurally deep JSON before the recursive stdlib decoder."""

    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte in (ord("["), ord("{")):
            depth += 1
            if depth > USAGE_LOG_MAX_JSON_NESTING:
                raise ValueError("Soniox usage log JSON nesting exceeds the parser contract")
        elif byte in (ord("]"), ord("}")):
            depth -= 1
            if depth < 0:
                raise ValueError("Soniox usage log JSON nesting is invalid")
    # Let ``json.loads`` retain ownership of malformed lexical syntax (for
    # example an unterminated string), but reject a malformed closing balance
    # here because it is just as incapable of producing trusted usage rows.
    if depth != 0:
        raise ValueError("Soniox usage log JSON nesting is invalid")


class _WorkerSecret:
    """One in-memory key whose ordinary Python representation is redacted.

    A tracker must retain a credential briefly to fetch its own provider usage,
    but the tracker is sometimes included in debugging structures.  Keeping a
    redacted handle in its visible object graph avoids leaking a key through a
    shallow ``vars()``/``repr()`` log while preserving same-key grouping for
    STT and TTS usage-log requests.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<worker-secret>"

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _WorkerSecret) and self._value == other._value

    def __getstate__(self) -> object:
        raise TypeError("worker secrets are not serializable")


class SonioxUsageTracker:
    """Collect exact token totals for the Soniox components of one call.

    ``components`` maps ``stt`` and/or ``tts`` to its actual in-memory API key
    and exact pinned model.  Component keys can differ (BYOK), so requests are
    grouped by key but filtered independently by their opaque references.
    """

    def __init__(
        self,
        *,
        call_reference: str,
        components: Mapping[str, tuple[str, str]],
        opener: Callable[..., Any] = _open_soniox_usage_logs_without_redirect,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        reference = str(call_reference or "").strip()
        if not reference:
            raise SonioxUsageEvidenceError("Soniox usage correlation requires a call reference")
        if not components:
            raise SonioxUsageEvidenceError("Soniox usage correlation requires a selected component")
        normalized: dict[str, tuple[_WorkerSecret, str]] = {}
        for component, values in components.items():
            name = str(component or "").strip().lower()
            if name not in _DIMENSIONS or name in normalized:
                raise SonioxUsageEvidenceError("Soniox usage correlation component is invalid")
            if not isinstance(values, tuple) or len(values) != 2:
                raise SonioxUsageEvidenceError("Soniox usage correlation contract is invalid")
            key, model = (str(value or "").strip() for value in values)
            if not key or not model:
                raise SonioxUsageEvidenceError("Soniox usage correlation contract is incomplete")
            normalized[name] = (_WorkerSecret(key), model)
        self._components = normalized
        self._references = {
            component: self._opaque_reference(reference, component)
            for component in sorted(normalized)
        }
        self._opener = opener
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._started_at = self._utc(self._now())

    def __repr__(self) -> str:
        # Default object repr is already safe, but keep debugging output useful
        # without ever showing a provider key or opaque call correlation value.
        return f"SonioxUsageTracker(components={sorted(self._components)!r})"

    def __getstate__(self) -> object:
        """Prevent accidental worker-task serialization of provider keys.

        Exact usage collection intentionally needs the active credential for a
        short in-memory window. It must not become a queue/cache/log payload
        merely because a future caller tries to pickle a stack object.
        """
        raise TypeError("Soniox usage trackers are worker-local and not serializable")

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _opaque_reference(call_reference: str, component: str) -> str:
        # The provider receives a stable opaque correlation value, never the
        # OpenPhonex call id or tenant identifiers.  Its fixed prefix also
        # lets support recognize this product integration without parsing an
        # internal identifier.
        digest = hashlib.sha256(f"openphonex-soniox:{component}:{call_reference}".encode("utf-8")).hexdigest()
        return f"opx-{component}-{digest[:40]}"

    def reference_for(self, component: str) -> str:
        name = str(component or "").strip().lower()
        value = self._references.get(name)
        if value is None:
            raise SonioxUsageEvidenceError("Soniox usage correlation component is unavailable")
        return value

    def discard_credentials(self) -> None:
        """Drop worker-only provider keys once terminal evidence is collected."""
        self._components = {}
        self._references = {}

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _whole_token(value: Any, *, field: str) -> int:
        del field
        # Usage logs are provider JSON.  Accept only a JSON integer rather
        # than coercing arbitrary strings: ``int("9" * millions)`` would
        # turn one otherwise bounded response page into unbounded CPU/memory.
        if isinstance(value, bool) or not isinstance(value, int):
            raise SonioxUsageEvidenceError("Soniox usage log contains an invalid token quantity")
        if value < 0:
            raise SonioxUsageEvidenceError("Soniox usage log contains an invalid token quantity")
        if value > USAGE_LOG_MAX_TOKENS_PER_DIMENSION:
            raise SonioxUsageEvidenceError("Soniox usage log token quantity exceeds the worker contract")
        return value

    def _read_key_logs(self, api_key: str, *, start_time: str, end_time: str) -> list[Mapping[str, Any]]:
        cursor: str | None = None
        result: list[Mapping[str, Any]] = []
        for page in range(USAGE_LOG_MAX_PAGES):
            query = {
                "start_time": start_time,
                "end_time": end_time,
                "limit": str(USAGE_LOG_PAGE_SIZE),
                "sort": "end_time_desc",
            }
            if cursor:
                query["cursor"] = cursor
            request = urllib.request.Request(
                f"{SONIOX_USAGE_LOGS_URL}?{urllib.parse.urlencode(query)}",
                headers={"Authorization": f"Bearer {api_key}"},
                method="GET",
            )
            try:
                with self._opener(request, timeout=USAGE_LOG_TIMEOUT_SECONDS) as response:
                    raw = _read_bounded_response(response)
            except (
                OSError,
                http.client.HTTPException,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ) as exc:
                # An unavailable transport can be transient; this is not
                # provider-native usage evidence and remains bounded by the
                # fixed retry budget in ``collect``.
                raise SonioxUsageEvidenceError("Soniox usage logs are unavailable", retryable=True) from exc
            try:
                _assert_bounded_json_nesting(raw)
                payload = json.loads(
                    raw.decode("utf-8"),
                    parse_int=_bounded_json_integer,
                    parse_float=_bounded_json_float,
                    parse_constant=_reject_json_constant,
                )
            except (
                RecursionError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                # A bounded but malformed provider page is a hard evidence
                # failure, not eventual-consistency lag.  Do not make further
                # reads that could hide it behind a later response.
                raise SonioxUsageEvidenceError("Soniox usage logs are unavailable") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("usage_logs"), list):
                raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid response")
            for row in payload["usage_logs"]:
                if not isinstance(row, dict):
                    raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid response")
                result.append(row)
            next_cursor = payload.get("next_page_cursor")
            if next_cursor is None:
                return result
            if not isinstance(next_cursor, str) or not next_cursor:
                raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid page cursor")
            if page == USAGE_LOG_MAX_PAGES - 1:
                raise SonioxUsageEvidenceError("Soniox usage logs exceeded the bounded page contract")
            cursor = next_cursor
        raise SonioxUsageEvidenceError("Soniox usage logs exceeded the bounded page contract")

    def _collect_once(self) -> dict[str, dict[str, int]]:
        # ``discard_credentials`` is terminal. Returning an empty mapping on a
        # second collection would turn absent paid-provider evidence into a
        # false zero-usage settlement.
        if not self._components:
            raise SonioxUsageEvidenceError("Soniox usage credentials were already discarded")
        started = self._started_at - timedelta(seconds=USAGE_LOG_WINDOW_PADDING_SECONDS)
        ended = self._utc(self._now())
        if ended <= started:
            raise SonioxUsageEvidenceError("Soniox usage-log time window is invalid")
        by_key: dict[_WorkerSecret, set[str]] = {}
        for component, (key, _model) in self._components.items():
            by_key.setdefault(key, set()).add(component)
        totals = {component: {dimension: 0 for dimension in _DIMENSIONS[component]} for component in self._components}
        seen: set[str] = set()
        # Cursor pages may overlap briefly while the provider appends a record.
        # ``uuid`` is the provider's per-request identity, so count a repeated
        # identical row once and reject an inconsistent duplicate rather than
        # ever double-settling native tokens.  This set is local to one
        # complete poll; a retry intentionally re-reads the same bounded
        # window from scratch.
        seen_logs: dict[str, tuple[str, str, tuple[int, ...]]] = {}
        reference_to_component = {reference: component for component, reference in self._references.items()}
        for key, key_components in by_key.items():
            for row in self._read_key_logs(
                key.reveal(),
                start_time=self._timestamp(started),
                end_time=self._timestamp(ended),
            ):
                component = reference_to_component.get(str(row.get("client_reference_id") or ""))
                if component is None:
                    continue
                # A reference identifies a component, but the provider usage
                # endpoint is scoped by the key that owns it. Never let a
                # row for a second customer/provider key be attributed from a
                # response obtained with this key, even if a malicious or
                # inconsistent response repeats an otherwise valid reference.
                if component not in key_components:
                    raise SonioxUsageEvidenceError(
                        "Soniox usage log reference does not match the selected credential"
                    )
                expected_model = self._components[component][1]
                if str(row.get("model") or "") != expected_model:
                    raise SonioxUsageEvidenceError("Soniox usage log model does not match the pinned component")
                log_uuid = str(row.get("uuid") or "").strip()
                if not log_uuid:
                    raise SonioxUsageEvidenceError("Soniox usage log omitted its provider request identity")
                dimensions = _DIMENSIONS[component]
                units = tuple(
                    self._whole_token(row.get(dimension), field=dimension)
                    for dimension in dimensions
                )
                fingerprint = (component, expected_model, units)
                previous = seen_logs.get(log_uuid)
                if previous is not None:
                    if previous != fingerprint:
                        raise SonioxUsageEvidenceError("Soniox usage log repeated an inconsistent provider request identity")
                    continue
                seen_logs[log_uuid] = fingerprint
                for dimension, units_for_dimension in zip(dimensions, units, strict=True):
                    total = totals[component][dimension] + units_for_dimension
                    if total > USAGE_LOG_MAX_TOKENS_PER_DIMENSION:
                        raise SonioxUsageEvidenceError(
                            "Soniox usage log token total exceeds the worker contract"
                        )
                    totals[component][dimension] = total
                seen.add(component)
        if seen != set(self._components):
            # The only consistency condition that Soniox documents as
            # eventually visible is a recently completed stream not yet being
            # present in its usage logs.  Preserve the fixed retry budget for
            # that case, while rejecting every malformed/inconsistent record
            # above without a second provider read.
            raise SonioxUsageEvidenceError(
                "Soniox usage logs have not recorded every selected component",
                retryable=True,
            )
        return totals

    def collect(self) -> dict[str, dict[str, int]]:
        """Return only exact native token totals, retrying a bounded log lag.

        The provider can publish a successful WebSocket record just after the
        stream closes.  Retry a small fixed number of times; no result is
        better than charging an estimated quantity.
        """
        last_error: SonioxUsageEvidenceError | None = None
        for attempt in range(USAGE_LOG_MAX_ATTEMPTS):
            try:
                return self._collect_once()
            except SonioxUsageEvidenceError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
                if attempt + 1 < USAGE_LOG_MAX_ATTEMPTS:
                    self._sleep(USAGE_LOG_RETRY_SECONDS)
        if last_error is None:
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable", retryable=True)
        raise last_error


__all__ = [
    "_RejectSonioxUsageRedirectHandler",
    "_open_soniox_usage_logs_without_redirect",
    "SONIOX_USAGE_LOGS_URL",
    "SonioxUsageEvidenceError",
    "SonioxUsageTracker",
    "USAGE_LOG_MAX_ATTEMPTS",
    "USAGE_LOG_MAX_PAGES",
]

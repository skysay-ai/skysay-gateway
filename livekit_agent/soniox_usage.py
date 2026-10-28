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

A Soniox usage row is written only after its WebSocket session closes, so the
worker closes its own recognizer streams first (``soniox_stt``) and records the
outcome here. Collection runs on the event loop under ONE absolute monotonic
deadline that every page, retry and segment owner shares: an urllib read in a
worker thread could not be cancelled, and its per-operation timeout bounded
each receive, not the request.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Mapping


SONIOX_USAGE_LOGS_URL = "https://api.soniox.com/v1/usage-logs"
USAGE_LOG_PAGE_SIZE = 1_000
USAGE_LOG_MAX_PAGES = 4
# Measured against the real API (stt-rt-v5 streams, one at a time): a closed
# session's row became visible 1.95-4.09 s after the close, one row per
# WebSocket session (a zero-audio session included). The usage-log endpoint
# itself answers 429 when polled faster than about once a second. So the first
# read waits out most of that lag and later reads are a second apart; the
# absolute deadline, not the attempt cap, is what bounds teardown.
USAGE_LOG_MAX_ATTEMPTS = 5
USAGE_LOG_RETRY_SECONDS = 1.0
USAGE_LOG_FIRST_READ_DELAY_SECONDS = 1.5
#: One collection's share of the SDK's 15 s entrypoint-shutdown window, shared
#: by every page, retry and segment owner (the stream close has its own bound).
USAGE_LOG_COLLECTION_BUDGET_SECONDS = 6.0
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


#: Why exact evidence is unavailable: bounded, non-secret, safe to log and to
#: carry in the refused-metering ``reason``. ``cancelled`` is the job's own
#: cancellation, held by the teardown; it is never reported as ``deadline``.
EVIDENCE_ERROR_KINDS = frozenset({
    "missing_rows", "http_status", "transport", "malformed", "pagination",
    "model_mismatch", "credential_scope", "contract", "stream_not_closed",
    "deadline", "cancelled",
})


#: What a worker-owned recognizer's terminal close can report
#: (``SonioxSTTCloseOutcome.status``).
SONIOX_STT_CLOSE_STATUSES = frozenset({"closed", "timeout", "error", "cancelled"})


#: Terminal close bound for a worker-owned recognizer (``soniox_stt``): the
#: stream task is cancelled, then any socket still open is closed.
SONIOX_STT_CLOSE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class SonioxSTTCloseOutcome:
    """What a terminal recognizer close established.

    ``status`` is ``closed``, ``timeout``, ``error`` or ``cancelled``; only
    ``closed`` lets a usage row be expected. ``sessions`` counts the sockets
    that completed initialisation. ``cancellation`` is the caller's own
    cancellation, held for it to re-raise after settlement.
    """

    status: str
    sessions: int
    streams: int
    cancellation: BaseException | None = field(default=None, compare=False)


async def close_soniox_stt_streams(
    stt: Any, *, timeout: float = SONIOX_STT_CLOSE_TIMEOUT_SECONDS
) -> SonioxSTTCloseOutcome | None:
    """Close a worker-owned Soniox recognizer; ``None`` for any other STT.

    Duck-typed so the agents need no plugin import to call it: only
    ``soniox_stt.STT`` defines ``skysay_close_streams``.
    """
    close = getattr(stt, "skysay_close_streams", None)
    if not callable(close):
        return None
    return await close(timeout=timeout)


class SonioxUsageEvidenceError(RuntimeError):
    """Exact provider-native usage could not be obtained safely."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        kind: str = "contract",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if kind not in EVIDENCE_ERROR_KINDS:
            raise ValueError(f"unknown Soniox evidence error kind: {kind}")
        # Only an eventual-consistency gap or a transient transport failure may
        # justify another paid-provider read.  A malformed or internally
        # inconsistent record is authoritative evidence that this settlement
        # cannot be trusted, so fail immediately and leave its reservation for
        # reconciliation instead of allowing a later page to hide the fault.
        self.retryable = retryable
        self.kind = kind
        self.status_code = status_code if type(status_code) is int else None

    @property
    def discriminator(self) -> str:
        """``soniox:<kind>[:<status>]``, the refused-metering ``reason`` form."""
        suffix = f":{self.status_code}" if self.status_code is not None else ""
        return f"soniox:{self.kind}{suffix}"


async def _read_bounded_body(content: Any) -> bytes:
    """Read one provider page without allowing a hostile body to exhaust RAM."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await content.read(
            min(USAGE_LOG_READ_CHUNK_BYTES, USAGE_LOG_MAX_RESPONSE_BYTES - total + 1)
        )
        if not chunk:
            return b"".join(chunks)
        if not isinstance(chunk, bytes):
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable", kind="malformed")
        total += len(chunk)
        if total > USAGE_LOG_MAX_RESPONSE_BYTES:
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable", kind="malformed")
        chunks.append(chunk)


class _UsageLogClient:
    """One short-lived aiohttp session for one collection.

    Owned here, never the job's shared session: that one is closed under any
    stream still running at process exit. The total timeout of each request is
    whatever the collection deadline leaves, so connect, headers and body are
    all inside it, and leaving the context closes every connection.
    """

    def __init__(self) -> None:
        self._session: Any = None

    async def __aenter__(self) -> "_UsageLogClient":
        # Imported here: the worker image installs aiohttp, but importing this
        # module (the cascade builder does, for every call) must not need it.
        import aiohttp

        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._session is not None:
            await self._session.close()

    async def get(self, url: str, *, api_key: str, timeout: float) -> bytes:
        import aiohttp

        assert self._session is not None
        # ``allow_redirects=False``: a 3xx is returned, never followed, so a
        # worker-held key is never replayed at a ``Location``.
        async with self._session.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                raise SonioxUsageEvidenceError(
                    "Soniox usage logs are unavailable",
                    retryable=True,
                    kind="http_status",
                    status_code=response.status,
                )
            return await _read_bounded_body(response.content)


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
        client_factory: Callable[[], Any] = _UsageLogClient,
        url: str = SONIOX_USAGE_LOGS_URL,
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_attempts: int = USAGE_LOG_MAX_ATTEMPTS,
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
        # Components whose adapter was constructed and then never attached to
        # media -- see ``mark_component_unused``.  Still part of the output
        # shape (settlement requires every pinned Soniox component to report
        # its dimensions), but exempt from the every-component-recorded check:
        # no stream ever ran under such a reference, so the usage-log row the
        # check waits for cannot arrive.
        self._unused: set[str] = set()
        self._references = {
            component: self._opaque_reference(reference, component)
            for component in sorted(normalized)
        }
        # The recognizer's terminal close, as ``record_stt_close`` was told it.
        # ``None`` means no worker-owned stream reported (a TTS-only tracker).
        self._stt_close_status: str | None = None
        self._stt_sessions: int | None = None
        self._not_before: float | None = None
        self._client_factory = client_factory
        self._url = url
        self._now = now or (lambda: datetime.now(UTC))
        self._clock = clock
        self._sleep = sleep
        self._max_attempts = max_attempts
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
        # Skysay call id or tenant identifiers.  Its fixed prefix also
        # lets support recognize this product integration without parsing an
        # internal identifier.
        digest = hashlib.sha256(f"skysay-soniox:{component}:{call_reference}".encode("utf-8")).hexdigest()
        return f"opx-{component}-{digest[:40]}"

    def reference_for(self, component: str) -> str:
        name = str(component or "").strip().lower()
        value = self._references.get(name)
        if value is None:
            raise SonioxUsageEvidenceError("Soniox usage correlation component is unavailable")
        return value

    def mark_component_unused(self, component: str) -> None:
        """Record that a selected component's adapter will never carry media.

        A replacement TTS registers only its own reference. A handle refused
        before SDK attachment, or a native adapter whose send-activity marker
        proves no text was ever submitted, can have measured zero use. The
        absence of an SDK metric alone does not establish that for a cancelled
        TTS request. References remain in the output shape with zero dimensions
        so settlement can account for every selected component.

        A component this tracker never selected is a no-op, not an error: a
        stack whose only Soniox component is TTS has no STT reference to
        exempt, and the caller cannot know which components a given stack
        metered without re-deriving the builder's provider table.
        """
        name = str(component or "").strip().lower()
        if name in self._components:
            self._unused.add(name)

    def default_deadline(self) -> float:
        """One collection's absolute deadline from now, on this tracker's clock."""
        return self._clock() + USAGE_LOG_COLLECTION_BUDGET_SECONDS

    def record_stt_close(self, *, status: str, sessions: int) -> None:
        """Take the recognizer's terminal close as the STT completeness rule.

        ``sessions`` is how many WebSocket sessions completed initialisation,
        counted where the socket was opened (``soniox_stt``), reconnects
        included. Soniox writes one usage row per session, and only after it
        closes, so:

        * a close that did not finish means the evidence is unavailable, and
          ``collect`` refuses without a read (``stream_not_closed``, or
          ``cancelled`` when the job's own cancellation interrupted it);
        * zero sessions is PROVEN absence: STT is settled at zero without a
          read. A missing SDK metric is not; it used to be read as one;
        * otherwise STT is complete only with that many distinct rows.

        A tracker without a Soniox STT component ignores this.
        """
        if "stt" not in self._components:
            return
        self._stt_close_status = status
        self._stt_sessions = max(0, int(sessions))
        if status == "closed" and self._stt_sessions == 0:
            self._unused.add("stt")
        elif status == "closed":
            # Absolute, so serial segment owners never wait it out twice.
            self._not_before = self._clock() + USAGE_LOG_FIRST_READ_DELAY_SECONDS

    def discard_credentials(self) -> None:
        """Drop worker-only provider keys once terminal evidence is collected."""
        self._components = {}
        self._references = {}
        self._unused = set()

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
            raise SonioxUsageEvidenceError("Soniox usage log contains an invalid token quantity", kind="malformed")
        if value < 0:
            raise SonioxUsageEvidenceError("Soniox usage log contains an invalid token quantity", kind="malformed")
        if value > USAGE_LOG_MAX_TOKENS_PER_DIMENSION:
            raise SonioxUsageEvidenceError("Soniox usage log token quantity exceeds the worker contract")
        return value

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise SonioxUsageEvidenceError("Soniox usage collection reached its deadline", kind="deadline")
        return remaining

    async def _get_page(self, client: Any, url: str, *, api_key: str, deadline: float) -> bytes:
        import aiohttp

        remaining = self._remaining(deadline)
        try:
            # ``asyncio.timeout`` is the authority: it cancels the request
            # wherever it is (a trickled header or body included), and the
            # client's own total timeout is the same remainder.
            async with asyncio.timeout(remaining):
                return await client.get(url, api_key=api_key, timeout=remaining)
        except SonioxUsageEvidenceError:
            raise
        except TimeoutError as exc:
            # Before ``aiohttp.ClientError``: aiohttp's timeouts are both.
            raise SonioxUsageEvidenceError("Soniox usage collection reached its deadline", kind="deadline") from exc
        except (aiohttp.ClientError, OSError, ValueError) as exc:
            # An unavailable transport can be transient; this is not
            # provider-native usage evidence and remains bounded by the
            # deadline in ``collect``.
            raise SonioxUsageEvidenceError(
                "Soniox usage logs are unavailable", retryable=True, kind="transport"
            ) from exc

    @staticmethod
    def _parse_page(raw: bytes) -> tuple[list[Mapping[str, Any]], Any]:
        """Pure, synchronous: bounded bytes in, rows and the next cursor out."""
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
            raise SonioxUsageEvidenceError("Soniox usage logs are unavailable", kind="malformed") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("usage_logs"), list):
            raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid response", kind="malformed")
        rows: list[Mapping[str, Any]] = []
        for row in payload["usage_logs"]:
            if not isinstance(row, dict):
                raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid response", kind="malformed")
            rows.append(row)
        return rows, payload.get("next_page_cursor")

    async def _read_key_logs(
        self, client: Any, api_key: str, *, start_time: str, end_time: str, deadline: float
    ) -> list[Mapping[str, Any]]:
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
            raw = await self._get_page(
                client, f"{self._url}?{urllib.parse.urlencode(query)}", api_key=api_key, deadline=deadline
            )
            rows, next_cursor = self._parse_page(raw)
            result.extend(rows)
            if next_cursor is None:
                return result
            if not isinstance(next_cursor, str) or not next_cursor:
                raise SonioxUsageEvidenceError("Soniox usage logs returned an invalid page cursor", kind="pagination")
            if page == USAGE_LOG_MAX_PAGES - 1:
                raise SonioxUsageEvidenceError("Soniox usage logs exceeded the bounded page contract", kind="pagination")
            cursor = next_cursor
        raise SonioxUsageEvidenceError("Soniox usage logs exceeded the bounded page contract", kind="pagination")

    async def _collect_once(self, client: Any, deadline: float) -> dict[str, dict[str, int]]:
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
        # A key selected only by never-attached components has nothing to
        # answer: skip the provider read entirely rather than polling for a
        # row that cannot exist.  A key shared with a live component keeps its
        # full component set, so a row that somehow does appear under an
        # unused reference is still attributed (harmlessly, to its zeroed
        # totals) instead of tripping the credential-scope check below.
        by_key = {key: components for key, components in by_key.items() if components - self._unused}
        totals = {component: {dimension: 0 for dimension in _DIMENSIONS[component]} for component in self._components}
        seen: set[str] = set()
        # Distinct provider rows per component: one WebSocket session writes
        # one row, so a reconnect's late second row must not be lost behind
        # the first (``record_stt_close``).
        rows_by_component: dict[str, int] = {}
        # Cursor pages may overlap briefly while the provider appends a record.
        # ``uuid`` is the provider's per-request identity, so count a repeated
        # identical row once and reject an inconsistent duplicate rather than
        # ever double-settling native tokens.  This set is local to one
        # complete poll; a retry intentionally re-reads the same bounded
        # window from scratch.
        seen_logs: dict[str, tuple[str, str, tuple[int, ...]]] = {}
        reference_to_component = {reference: component for component, reference in self._references.items()}
        for key, key_components in by_key.items():
            for row in await self._read_key_logs(
                client,
                key.reveal(),
                start_time=self._timestamp(started),
                end_time=self._timestamp(ended),
                deadline=deadline,
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
                        "Soniox usage log reference does not match the selected credential",
                        kind="credential_scope",
                    )
                expected_model = self._components[component][1]
                if str(row.get("model") or "") != expected_model:
                    raise SonioxUsageEvidenceError(
                        "Soniox usage log model does not match the pinned component", kind="model_mismatch"
                    )
                log_uuid = str(row.get("uuid") or "").strip()
                if not log_uuid:
                    raise SonioxUsageEvidenceError(
                        "Soniox usage log omitted its provider request identity", kind="malformed"
                    )
                dimensions = _DIMENSIONS[component]
                units = tuple(
                    self._whole_token(row.get(dimension), field=dimension)
                    for dimension in dimensions
                )
                fingerprint = (component, expected_model, units)
                previous = seen_logs.get(log_uuid)
                if previous is not None:
                    if previous != fingerprint:
                        raise SonioxUsageEvidenceError(
                            "Soniox usage log repeated an inconsistent provider request identity", kind="malformed"
                        )
                    continue
                seen_logs[log_uuid] = fingerprint
                rows_by_component[component] = rows_by_component.get(component, 0) + 1
                for dimension, units_for_dimension in zip(dimensions, units, strict=True):
                    total = totals[component][dimension] + units_for_dimension
                    if total > USAGE_LOG_MAX_TOKENS_PER_DIMENSION:
                        raise SonioxUsageEvidenceError(
                            "Soniox usage log token total exceeds the worker contract"
                        )
                    totals[component][dimension] = total
                seen.add(component)
        if (
            "stt" in seen
            and self._stt_sessions is not None
            and rows_by_component.get("stt", 0) < self._stt_sessions
        ):
            # A reconnect opened another session whose row is not visible yet.
            seen.discard("stt")
        if seen | self._unused != set(self._components):
            # The only consistency condition that Soniox documents as
            # eventually visible is a recently completed stream not yet being
            # present in its usage logs.  Preserve the fixed retry budget for
            # that case, while rejecting every malformed/inconsistent record
            # above without a second provider read.  A component marked
            # unused is exempt: its measured absence is the expected outcome,
            # not a lagging log.
            raise SonioxUsageEvidenceError(
                "Soniox usage logs have not recorded every selected component",
                retryable=True,
                kind="missing_rows",
            )
        return totals

    async def collect(self, *, deadline: float) -> dict[str, dict[str, int]]:
        """Return only exact native token totals, retrying a bounded log lag.

        ``deadline`` is absolute, on this tracker's monotonic clock, and shared
        with every other owner the caller collects. A request still running at
        it is cancelled (``deadline``). When a retryable failure leaves no time
        for another read, that failure is what is raised, so the log says
        ``missing_rows`` or ``http_status:429`` rather than only "late". No
        result is better than charging an estimated quantity. An external
        cancellation propagates unchanged; the caller holds it.
        """
        if self._stt_close_status == "cancelled":
            # The job's own cancellation interrupted the close; held by the
            # teardown, and named as that, not as a stream that would not stop.
            raise SonioxUsageEvidenceError(
                "Soniox STT stream close was cancelled before usage collection", kind="cancelled"
            )
        if self._stt_close_status is not None and self._stt_close_status != "closed":
            raise SonioxUsageEvidenceError(
                "Soniox STT stream did not close before usage collection", kind="stream_not_closed"
            )
        last_error: SonioxUsageEvidenceError | None = None
        async with self._client_factory() as client:
            for attempt in range(self._max_attempts):
                wait = (
                    USAGE_LOG_RETRY_SECONDS
                    if attempt
                    else (0.0 if self._not_before is None else self._not_before - self._clock())
                )
                if wait > 0:
                    remaining = deadline - self._clock()
                    if remaining <= wait:
                        break
                    await self._sleep(wait)
                try:
                    return await self._collect_once(client, deadline)
                except SonioxUsageEvidenceError as exc:
                    if not exc.retryable:
                        raise
                    last_error = exc
        if last_error is None:
            if not self._max_attempts:
                raise SonioxUsageEvidenceError("Soniox usage logs are unavailable")
            raise SonioxUsageEvidenceError("Soniox usage collection reached its deadline", kind="deadline")
        raise last_error


__all__ = [
    "EVIDENCE_ERROR_KINDS",
    "SONIOX_STT_CLOSE_STATUSES",
    "SONIOX_STT_CLOSE_TIMEOUT_SECONDS",
    "SONIOX_USAGE_LOGS_URL",
    "SonioxSTTCloseOutcome",
    "SonioxUsageEvidenceError",
    "SonioxUsageTracker",
    "USAGE_LOG_COLLECTION_BUDGET_SECONDS",
    "USAGE_LOG_MAX_ATTEMPTS",
    "USAGE_LOG_MAX_PAGES",
    "close_soniox_stt_streams",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Native TTS units, counted where the worker hands them to the provider.

One meter serves every TTS provider the worker meters itself. Its unit is
DATA in the capability artifact (``billing``): a unit the worker measures at
its wire (characters, UTF-8 bytes) is counted where an adapter hands it to the
provider's connection; a unit only the provider can measure (tokens) is the
provider's own cumulative usage report for the request. What a request is
charged once it ends -- completed, abandoned by the worker (a barge-in or
teardown), ended by a transport failure, or refused with a provider error --
is DATA too: ``billing.request_billing``. An adapter only reports facts at its
provider wire: the units it handed, the usage the provider reported, and how
the request ended. So the meter has no provider branch, and an interrupted
request settles at a measured quantity instead of making the call's usage
unknowable.

Only numbers are retained per request, never the text that was sent.
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any


class NativeTTSUsageError(RuntimeError):
    """A worker attempted to add invalid TTS accounting evidence."""


class TTSMeterFrozen(NativeTTSUsageError):
    """Usage evidence was taken; nothing more may reach the provider."""


class TTSRequestActivity:
    """Conservative evidence that synthesis text may have reached a provider."""

    def __init__(self) -> None:
        self.started = False

    def mark_started(self) -> None:
        # Mark before awaiting the send: a cancelled/failed acknowledgement
        # cannot prove that the remote provider did not receive the request.
        self.started = True


#: How one provider request (one attempt) can end, as an adapter observes it.
#: ``abandoned`` is the worker's own cancel or close on a healthy connection;
#: ``transport_failure`` is any other ending without a provider-reported error.
REQUEST_OUTCOMES = ("completed", "abandoned", "transport_failure", "provider_error")
#: What a request with one outcome is charged: the units the worker handed,
#: the provider's latest cumulative usage report, or nothing.
REQUEST_CHARGES = ("submitted_units", "reported_usage", "none")
#: Units the worker counts at its wire, and the settlement field each settles.
HANDED_UNIT_FIELDS = {"character": "tts_characters", "utf8_byte": "tts_utf8_bytes"}
#: Units only the provider measures, and the dimensions each may report.
REPORTED_UNIT_DIMENSIONS = {"token": ("input_text_tokens", "output_audio_tokens")}
#: The largest total the control plane accepts per unit (per token dimension).
UNIT_MAXIMUM = {"character": 100_000_000, "utf8_byte": 400_000_000, "token": 100_000_000}
#: The control plane's bound on every refused-metering diagnostic count.
DIAGNOSTIC_MAXIMUM = 100_000_000


class RequestBilling:
    """A provider's metering unit and charge per request outcome, from its data."""

    __slots__ = ("unit", "dimensions", "measure", "_charges")

    def __init__(self, *, unit: str, charges: Mapping[str, Any], dimensions: Any = ()) -> None:
        if unit in HANDED_UNIT_FIELDS:
            if tuple(dimensions or ()):
                raise NativeTTSUsageError("a worker-measured TTS unit has no reported dimensions")
            measure, chosen = "submitted_units", ()
        elif unit in REPORTED_UNIT_DIMENSIONS:
            chosen = tuple(dimensions) if isinstance(dimensions, (list, tuple)) else ()
            if (not chosen or len(set(chosen)) != len(chosen)
                    or any(d not in REPORTED_UNIT_DIMENSIONS[unit] for d in chosen)):
                raise NativeTTSUsageError("a provider-reported TTS unit must name its dimensions")
            measure = "reported_usage"
        else:
            raise NativeTTSUsageError("TTS request billing has no metered unit")
        if not isinstance(charges, Mapping) or any(
            charges.get(outcome) not in (measure, "none") for outcome in REQUEST_OUTCOMES
        ):
            raise NativeTTSUsageError("TTS request billing must name a charge for every outcome")
        if charges["completed"] != measure:
            raise NativeTTSUsageError("a completed TTS request must be charged what it measured")
        self.unit = unit
        self.dimensions = chosen
        self.measure = measure
        self._charges = {outcome: charges[outcome] for outcome in REQUEST_OUTCOMES}

    @classmethod
    def from_capability(cls, capability: Mapping[str, Any] | None) -> "RequestBilling":
        """Read a capability row's (or declared TTS row's) ``billing`` data."""
        billing = capability.get("billing") if isinstance(capability, Mapping) else None
        rules = billing.get("request_billing") if isinstance(billing, Mapping) else None
        if not isinstance(rules, Mapping):
            raise NativeTTSUsageError("TTS capability declares no request billing")
        return cls(unit=str(billing.get("unit") or ""), charges=rules, dimensions=billing.get("dimensions") or ())

    def charge(self, outcome: str) -> str:
        return self._charges[outcome]

    def billable(self, outcome: str, request: "TTSProviderRequest") -> dict[str, int]:
        """What one ended request is charged, per dimension of this unit."""
        charge = self._charges[outcome]
        if charge == "submitted_units":
            return {self.unit: request.units}
        if charge == "reported_usage":
            return {dimension: request.reported.get(dimension, 0) for dimension in self.dimensions}
        return {}

    def settlement_fields(self, totals: Mapping[str, int]) -> dict[str, int]:
        """The usage-report fields this unit settles (a table, not a provider branch)."""
        if self.measure == "submitted_units":
            return {HANDED_UNIT_FIELDS[self.unit]: int(totals.get(self.unit, 0))}
        return {f"tts_{dimension}": int(totals.get(dimension, 0)) for dimension in self.dimensions}


class TTSProviderRequest:
    """One provider request: units handed, usage reported, and how it ended."""

    __slots__ = (
        "_meter", "units", "reported", "outcome", "ended", "provider_context",
        "awaits_provider_end", "provider_ended",
    )

    def __init__(self, meter: "NativeTTSUnitMeter") -> None:
        self._meter = meter
        self.units = 0
        #: The provider's latest cumulative usage report for this request.
        self.reported: dict[str, int] = {}
        self.outcome: str | None = None
        self.ended = asyncio.Event()
        #: The provider's own id for this request on a shared socket, when the
        #: adapter needs it to attribute a provider message to the request.
        self.provider_context: str | None = None
        #: A request on a shared connection outlives the worker's attempt: the
        #: provider still finishes (or refuses) the context after a barge-in.
        #: Until the provider's own end of it is observed, the worker's
        #: abandonment is provisional and a refusal or a lost connection
        #: observed at the wire still decides the ending.
        self.awaits_provider_end = False
        self.provider_ended = False

    def handed(self, units: int) -> None:
        """Count units immediately before the adapter awaits their send."""
        self._meter._hand(self, units)

    def report_usage(self, usage: Mapping[str, Any]) -> None:
        """Replace the usage the provider reported so far (cumulative counts)."""
        self._meter._report(self, usage)

    def bind_provider_context(self, context: str, *, awaits_provider_end: bool = False) -> None:
        """Name the provider's id for this request on a shared connection."""
        self._meter._bind(self, context, awaits_provider_end=awaits_provider_end)

    def mark_provider_end(self) -> None:
        """The provider's final message for this request's context arrived."""
        self._meter._provider_end(self)

    def finish(self, outcome: str) -> None:
        """Record how the request ended, as observed; the first ending wins."""
        if outcome not in REQUEST_OUTCOMES:
            raise NativeTTSUsageError("unknown TTS request outcome")
        self._meter._finish(self, outcome)


class NativeTTSUnitMeter:
    """Call-wide billable native TTS usage, by request outcome.

    Endings are recorded where an adapter observes them at its wire, so a
    connection failure or a provider refusal keeps its own outcome even when
    the worker cancels the request a moment later. Before evidence is taken,
    ``settle`` stops admission -- nothing more can reach a provider -- and
    waits, bounded, for every open request to end on its own, so a receiver
    still reads whatever the provider already sent; ``snapshot`` then freezes.
    A request whose ending was never established is charged nothing. So
    interruption, cancellation, retries and provider refusals never make the
    quantity unknowable; the only refusals left are a total beyond the
    settlement bound and units handed with no open request (an adapter seam
    that bypassed the meter), and both keep the reservation held.
    """

    def __init__(self, billing: RequestBilling) -> None:
        if not isinstance(billing, RequestBilling):
            raise NativeTTSUsageError("TTS unit meter requires request billing rules")
        self.billing = billing
        self._maximum = UNIT_MAXIMUM[billing.unit]
        self._requests: list[TTSProviderRequest] = []
        self._contexts: dict[str, TTSProviderRequest] = {}
        self._unattributed_sends = 0
        self._unattributed_units = 0
        self._admitting = True
        self._frozen = False
        self._lock = threading.Lock()

    def _refuse_if_stopped(self) -> None:
        if not self._admitting:
            raise TTSMeterFrozen("TTS usage evidence is being taken")

    def refuse_if_settling(self, request: "TTSProviderRequest | None") -> None:
        """Refuse any further traffic once evidence is being taken.

        Refusing the rest of a healthy request is the worker cutting it off,
        so that request ends as abandoned.
        """
        refused = False
        with self._lock:
            if self._admitting:
                return
            if request is not None and request.outcome is None and not self._frozen:
                request.outcome = "abandoned"
                refused = True
        if refused and request is not None:
            request.ended.set()
        raise TTSMeterFrozen("TTS usage evidence is being taken")

    def open_request(self) -> TTSProviderRequest:
        request = TTSProviderRequest(self)
        with self._lock:
            self._refuse_if_stopped()
            self._requests.append(request)
        return request

    def request_for_context(self, context: Any) -> TTSProviderRequest | None:
        """The request a provider id was bound to, or ``None``; never a guess."""
        if not isinstance(context, str) or not context:
            return None
        with self._lock:
            return self._contexts.get(context)

    def handed_unattributed(self, units: int) -> None:
        """Units an adapter is about to send with no metered request open."""
        units = _whole_units(units)
        with self._lock:
            self._refuse_if_stopped()
            self._unattributed_sends += 1
            self._unattributed_units += units

    def _bind(self, request: TTSProviderRequest, context: Any, *, awaits_provider_end: bool) -> None:
        if not isinstance(context, str) or not context:
            raise NativeTTSUsageError("a provider request id must be a non-empty string")
        with self._lock:
            bound = self._contexts.get(context)
            if bound is not None and bound is not request:
                raise NativeTTSUsageError("a provider request id was bound to two requests")
            self._contexts[context] = request
            request.provider_context = context
            request.awaits_provider_end = request.awaits_provider_end or awaits_provider_end

    def _provider_end(self, request: TTSProviderRequest) -> None:
        with self._lock:
            if self._frozen:
                return
            request.provider_ended = True
            decided = request.outcome is not None
        if decided:
            request.ended.set()

    def _hand(self, request: TTSProviderRequest, units: int, *, after_ending: bool = False) -> None:
        units = _whole_units(units)
        with self._lock:
            if not self._admitting:
                # Refusing the rest of a healthy request is the worker cutting
                # it off: what it already handed is charged as abandoned.
                refused = request.outcome is None and not self._frozen
                if refused:
                    request.outcome = "abandoned"
            elif request.outcome is not None and not after_ending:
                raise NativeTTSUsageError("TTS units handed after the request ended")
            else:
                # ``after_ending``: a shared connection's own send loop can
                # still write frames the request queued before the worker
                # cancelled it. They reach the provider, so they count, under
                # the outcome already recorded.
                request.units += units
                return
        if refused:
            request.ended.set()
        raise TTSMeterFrozen("TTS usage evidence is being taken")

    def _report(self, request: TTSProviderRequest, usage: Mapping[str, Any]) -> None:
        if self.billing.measure != "reported_usage":
            raise NativeTTSUsageError("this TTS unit is measured by the worker, not reported")
        if not isinstance(usage, Mapping) or set(usage) != set(self.billing.dimensions):
            raise NativeTTSUsageError("a TTS usage report must carry exactly its declared dimensions")
        counts = {dimension: _whole_units(usage[dimension]) for dimension in self.billing.dimensions}
        with self._lock:
            if self._frozen:
                return
            request.reported = counts

    def _finish(self, request: TTSProviderRequest, outcome: str) -> None:
        with self._lock:
            if self._frozen:
                return
            provisional = request.awaits_provider_end and not request.provider_ended
            if request.outcome is None:
                request.outcome = outcome
                # The worker's own abandonment of a context the provider is
                # still finishing waits for the provider's end (bounded by
                # ``settle``); every other ending is final when observed.
                decided = not (outcome == "abandoned" and provisional)
            elif (request.outcome == "abandoned" and provisional
                  and outcome in ("provider_error", "transport_failure")):
                # Observed at the wire after the worker let go: the provider
                # refused it, or the connection carrying it was lost.
                request.outcome = outcome
                decided = True
            else:
                return
        if decided:
            request.ended.set()

    async def settle(self, timeout: float = 1.0) -> None:
        """Stop admission, let open requests end on their own, then freeze.

        Cancels nothing: an attempt's receiver keeps running and observes what
        the provider already sent. Bounded: a request that has not ended
        within ``timeout`` is charged nothing -- including a worker
        abandonment the provider has not ended yet, which it could still
        refuse. Idempotent; never raises for a request's own failure.
        """
        with self._lock:
            self._admitting = False
            # Open requests, and abandonments the provider has not yet ended.
            pending = [request for request in self._requests if not request.ended.is_set()]
        if pending and timeout > 0:
            waiters = {asyncio.ensure_future(request.ended.wait()) for request in pending}
            try:
                await asyncio.wait(waiters, timeout=timeout)
            finally:
                # Only these event waits are cancelled; provider tasks are not.
                for waiter in waiters:
                    waiter.cancel()
                await asyncio.gather(*waiters, return_exceptions=True)
        with self._lock:
            self._frozen = True

    @staticmethod
    def _established(request: TTSProviderRequest) -> str | None:
        """The request's ending if it is established, else ``None``.

        A worker abandonment the provider has not yet ended is not
        established: the provider may still refuse that context.
        """
        if request.outcome == "abandoned" and request.awaits_provider_end and not request.provider_ended:
            return None
        return request.outcome

    def _billable_totals(self) -> dict[str, int]:
        # No established ending -> charged nothing (see ``settle``).
        totals: dict[str, int] = {}
        for request in self._requests:
            outcome = self._established(request)
            if outcome is None:
                continue
            for dimension, count in self.billing.billable(outcome, request).items():
                totals[dimension] = totals.get(dimension, 0) + count
        return totals

    def _freeze(self) -> None:
        self._admitting = False
        self._frozen = True

    def snapshot(self) -> dict[str, int]:
        """Freeze and return the usage-report fields this meter settles."""
        with self._lock:
            self._freeze()
            totals = self._billable_totals()
            unattributed = self._unattributed_sends
        if any(count > self._maximum for count in totals.values()):
            raise NativeTTSUsageError("TTS native usage exceeded the bounded counter")
        if unattributed:
            raise NativeTTSUsageError("TTS native units reached the provider outside a metered request")
        return self.billing.settlement_fields(totals)

    def outcome_summary(self) -> dict[str, int]:
        """Request counts and billed quantities per ending; numbers only, never raises.

        Every key ends in ``_count`` so the worker's evidence journal keeps it.
        """
        with self._lock:
            summary: dict[str, int] = {"requests_count": len(self._requests)}
            for outcome in (*REQUEST_OUTCOMES, None):
                chosen = [request for request in self._requests if self._established(request) == outcome]
                name = outcome or "unended"
                summary[f"{name}_requests_count"] = len(chosen)
                billed = 0
                for request in chosen:
                    if outcome is not None:
                        billed += sum(self.billing.billable(outcome, request).values())
                summary[f"{name}_billed_count"] = billed
            summary["unattributed_sends_count"] = self._unattributed_sends
        return summary

    def diagnostic(self) -> dict[str, int | bool]:
        """Non-billable evidence for a refused report; never a settlement."""
        with self._lock:
            self._freeze()
            totals = self._billable_totals()
            return {
                "completed_total": min(sum(totals.values()), DIAGNOSTIC_MAXIMUM),
                "uncertain_requests": min(
                    self._unattributed_sends
                    + sum(1 for request in self._requests if self._established(request) is None),
                    DIAGNOSTIC_MAXIMUM,
                ),
                "uncertain_accepted_characters": min(
                    self._unattributed_units
                    + sum(request.units for request in self._requests if self._established(request) is None),
                    DIAGNOSTIC_MAXIMUM,
                ),
                "overflowed": any(count > self._maximum for count in totals.values()),
            }


async def settle_meters(*meters: Any, timeout: float = 1.0) -> None:
    """Settle every native unit meter a stack owns before taking evidence."""
    for meter in meters:
        if isinstance(meter, NativeTTSUnitMeter):
            await meter.settle(timeout)


def _whole_units(units: Any) -> int:
    if isinstance(units, bool) or not isinstance(units, int) or units < 0:
        raise NativeTTSUsageError("TTS units must be a non-negative integer")
    return units


_CURRENT_REQUEST: contextvars.ContextVar[TTSProviderRequest | None] = contextvars.ContextVar(
    "skysay_tts_provider_request", default=None
)


def current_tts_request() -> TTSProviderRequest | None:
    """The request whose attempt the calling task belongs to, if any.

    Plugins create their sender/receiver tasks inside the attempt, and a task
    copies the context it was created in, so the binding reaches them.
    """
    return _CURRENT_REQUEST.get()


async def run_metered_request(meter: NativeTTSUnitMeter, attempt: Callable[[], Awaitable[Any]]) -> Any:
    """Run one provider attempt as one metered request and classify its end.

    A return is ``completed``. A cancellation is ``abandoned``: only the
    worker cancels an attempt -- a barge-in, teardown, or a deadline the
    worker set around consuming it -- and LiveKit never retries a cancelled
    attempt. Any exception is ``transport_failure``: the plugins raise their
    own connection and provider timeouts as exceptions. An ending already
    observed at the wire (a provider refusal, a closed socket) wins.
    """
    request = meter.open_request()
    token = _CURRENT_REQUEST.set(request)
    try:
        result = await attempt()
    except asyncio.CancelledError:
        request.finish("abandoned")
        raise
    except BaseException:
        request.finish("transport_failure")
        raise
    else:
        request.finish("completed")
        return result
    finally:
        _CURRENT_REQUEST.reset(token)


async def hand_off(meter: NativeTTSUnitMeter, units: int, send: Callable[[], Awaitable[Any]],
                   *, request: TTSProviderRequest | None = None) -> Any:
    """Count ``units`` on the request, then await their send.

    Counted before the await: a send the worker cancels has already handed its
    frame to the connection. A send that fails any other way is observed here,
    at the wire, as the request's transport failure.
    """
    request = request if request is not None else current_tts_request()
    if units:
        if request is None:
            meter.handed_unattributed(units)
        else:
            request.handed(units)
    else:
        # A control frame carries no units, but nothing at all may reach a
        # provider once evidence is being taken.
        meter.refuse_if_settling(request)
    try:
        return await send()
    except asyncio.CancelledError:
        raise
    except BaseException:
        if request is not None:
            request.finish("transport_failure")
        raise


async def hand_off_context(meter: NativeTTSUnitMeter, provider_context: Any, units: int,
                           send: Callable[[], Awaitable[Any]]) -> Any:
    """``hand_off`` for a frame a SHARED connection's own task sends.

    The request is the one bound to the frame's provider id, never the sending
    task's context variable: a pooled socket's send loop inherited whichever
    request happened to open the connection. A frame whose id is bound to no
    request is unattributed (the snapshot refuses it). Units for a request that
    already ended still count under its recorded outcome: they were queued
    before the worker cancelled and reach the provider all the same.
    """
    request = meter.request_for_context(provider_context)
    if units:
        if request is None:
            meter.handed_unattributed(units)
        else:
            meter._hand(request, units, after_ending=True)
    else:
        meter.refuse_if_settling(request)
    try:
        return await send()
    except asyncio.CancelledError:
        raise
    except BaseException:
        if request is not None:
            request.finish("transport_failure")
        raise


def mark_transport_failure() -> None:
    """The adapter observed its connection fail for the current request."""
    request = current_tts_request()
    if request is not None:
        request.finish("transport_failure")


def mark_provider_error() -> None:
    """The provider reported an error for the current request."""
    request = current_tts_request()
    if request is not None:
        request.finish("provider_error")


def _counting_payload_class() -> type:
    from aiohttp import payload as aiohttp_payload

    class CountingPayload(aiohttp_payload.BytesPayload):
        """A request body whose units are handed when aiohttp writes it.

        aiohttp writes the body only after it has a connection, so a request
        cancelled while connecting hands nothing.
        """

        def __init__(self, value: bytes, *, meter: NativeTTSUnitMeter, request: TTSProviderRequest | None,
                     units: int, **kwargs: Any) -> None:
            super().__init__(value, **kwargs)
            self._skysay_meter = meter
            self._skysay_request = request
            self._skysay_units = units
            self._skysay_handed = False

        def _hand(self) -> None:
            if self._skysay_handed or not self._skysay_units:
                # Counted once, but every write is still admission-checked.
                self._skysay_meter.refuse_if_settling(self._skysay_request)
                return
            self._skysay_handed = True
            if self._skysay_request is None:
                self._skysay_meter.handed_unattributed(self._skysay_units)
            else:
                self._skysay_request.handed(self._skysay_units)

        def _failed(self) -> None:
            # aiohttp writes the body in a task of its own; a write failure
            # observed there is this request's transport failure, before any
            # later cancellation reaches the attempt.
            if self._skysay_request is not None:
                self._skysay_request.finish("transport_failure")

        async def write_with_length(self, writer: Any, content_length: Any) -> None:
            self._hand()
            try:
                await super().write_with_length(writer, content_length)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._failed()
                raise

        async def write(self, writer: Any) -> None:
            self._hand()
            try:
                await super().write(writer)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._failed()
                raise

    return CountingPayload


class _MeteredPost:
    """``session.post(...)``: body units handed when written; status observed.

    A 4xx/5xx status is the provider's own refusal of the request.
    """

    def __init__(self, open_request: Callable[[], Any]) -> None:
        self._open = open_request
        self._context: Any = None

    async def __aenter__(self) -> Any:
        self._context = self._open()
        response = await self._context.__aenter__()
        status = getattr(response, "status", None)
        if not isinstance(status, int) or status >= 300:
            # Redirects are not followed (a replayed body would bypass the
            # meter), so anything but a 2xx is the provider not processing it.
            mark_provider_error()
        return response

    async def __aexit__(self, *exc_info: Any) -> Any:
        return await self._context.__aexit__(*exc_info)


class MeteredHTTPSession:
    """An HTTP session whose ``post`` bodies hand their units to the meter.

    ``body_of`` turns the plugin's request arguments into ``(body bytes,
    content type, units, remaining arguments)``; everything else delegates.
    """

    def __init__(self, session: Any, meter: NativeTTSUnitMeter,
                 body_of: Callable[[dict[str, Any]], tuple[bytes, str | None, int, dict[str, Any]]]) -> None:
        self._session = session
        self._meter = meter
        self._body_of = body_of

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def post(self, *args: Any, **kwargs: Any) -> _MeteredPost:
        body, content_type, units, remaining = self._body_of(dict(kwargs))
        payload_kwargs = {"content_type": content_type} if content_type else {}
        payload = _counting_payload_class()(
            body, meter=self._meter, request=current_tts_request(), units=units, **payload_kwargs
        )
        remaining["allow_redirects"] = False
        return _MeteredPost(lambda: self._session.post(*args, data=payload, **remaining))


async def await_fish_stream_tasks(
    *,
    input_task: asyncio.Task[Any],
    send_task: asyncio.Task[Any],
    receive_task: asyncio.Task[Any],
    cancel: Callable[..., Awaitable[None]],
) -> None:
    """Drive Fish's three stream tasks without sending after provider finish.

    ``FIRST_EXCEPTION`` waits for every task when the receiver returns a
    successful ``finish`` frame. That would leave a queued sentence sender
    alive after the provider has ended the stream, so it could write and count
    text that the provider will never synthesize. Observe completions one at a
    time: ordinary input/sender completion still waits for the receiver, while
    receiver completion cancels remaining input/sends immediately.
    """
    pending: set[asyncio.Task[Any]] = {input_task, send_task, receive_task}
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # A normal terminal Fish frame wins over a queued sender/input even
            # when a sibling happened to be cancelled in the same event-loop turn.
            # Treating that sibling first would turn a successfully finished stream
            # into cancellation and skip the caller's ordinary terminal cleanup.
            if receive_task in done:
                if receive_task.cancelled():
                    await cancel(*pending)
                    raise asyncio.CancelledError
                receive_error = receive_task.exception()
                if receive_error is not None:
                    await cancel(*pending)
                    raise receive_error
                await cancel(*pending)
                return
            for task in done:
                if task.cancelled():
                    await cancel(*pending)
                    raise asyncio.CancelledError
                error = task.exception()
                if error is not None:
                    await cancel(*pending)
                    raise error
    except asyncio.CancelledError:
        # Barge-in/call teardown can cancel the outer stream driver while all
        # three provider tasks are still live. The tasks own sockets and the
        # Fish text counter, so leaving them behind can keep media and billing
        # work alive after the call is gone.
        await cancel(input_task, send_task, receive_task)
        raise


__all__ = [
    "DIAGNOSTIC_MAXIMUM",
    "HANDED_UNIT_FIELDS",
    "MeteredHTTPSession",
    "NativeTTSUnitMeter",
    "NativeTTSUsageError",
    "REQUEST_CHARGES",
    "REQUEST_OUTCOMES",
    "REPORTED_UNIT_DIMENSIONS",
    "RequestBilling",
    "TTSMeterFrozen",
    "TTSProviderRequest",
    "TTSRequestActivity",
    "UNIT_MAXIMUM",
    "await_fish_stream_tasks",
    "current_tts_request",
    "hand_off",
    "hand_off_context",
    "mark_provider_error",
    "mark_transport_failure",
    "run_metered_request",
    "settle_meters",
]

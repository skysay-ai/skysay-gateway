# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Fresh source permission and one owned provider's durable native facts."""
from __future__ import annotations

import asyncio
import math
import secrets
from dataclasses import asdict, dataclass, field

try:
    from .workflow_listener_runtime import build_workflow_source, enter_session_source, require_source_credential
    from .workflow_source_transport import SourceOwner, WorkflowSourceTransport
    from .capabilities.registry import load_registry
    from .capabilities.workflow_source_fields import MEASURED_EVIDENCE, measured_source_units
    from .capabilities.workflow_sources import SourceAdapterDeclaration
    from .workflow_speech import SpeechRefused
    from .workflow_tasks import join_owned_task
except ImportError:
    from workflow_listener_runtime import build_workflow_source, enter_session_source, require_source_credential
    from workflow_source_transport import SourceOwner, WorkflowSourceTransport
    from capabilities.registry import load_registry
    from capabilities.workflow_source_fields import MEASURED_EVIDENCE, measured_source_units
    from capabilities.workflow_sources import SourceAdapterDeclaration
    from workflow_speech import SpeechRefused
    from workflow_tasks import join_owned_task


#: Why a finite listener refused, by the site that refused it: its
#: constructor (``config``), a frame or the input EOF outside its contract
#: (``frame``, ``end``), its absolute lifetime (``wall``), a second transcript
#: claim (``reclaimed``), or a provider stream that ended without its terminal
#: acknowledgement (``no_ack``: never connected, closed, errored or timed out).
LISTENER_REFUSAL_REASONS = frozenset({"config", "frame", "end", "wall", "reclaimed", "no_ack"})


class WorkflowListenerRefusal(RuntimeError):
    """A listener's fixed public error and one bounded ``reason``.

    Every provider's ``ListenerRefused`` is one of these, so the call can name
    a refused listener without knowing which provider it was. The message
    never varies; provider exceptions and payloads never escape.
    """

    def __init__(self, reason):
        if reason not in LISTENER_REFUSAL_REASONS:
            raise ValueError("unknown listener refusal reason")
        super().__init__("workflow listener refused")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SourceMetric:
    units: tuple[tuple[str, int | float], ...]
    session_id: str = field(repr=False)
    reference: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """Code-owned adapter facts. Plain channel EOF cannot construct authority."""
    status: str
    total_units: tuple[tuple[str, int | float], ...]
    metrics: tuple[SourceMetric, ...]
    session_id: str = field(repr=False)
    terminal_ack: bool
    #: Seconds of audio the adapter measured itself: listener PCM written to
    #: the provider connection, output PCM received from the provider or
    #: played out. ``None`` for an adapter that does not measure. Whether the
    #: measurement settles is the registry's ``workflow_source_measurement``.
    measured_seconds: float | None = None


class AuthoredSourceResourceBinding:
    """One call/run and declaration own a reusable authored resource lane."""
    __slots__ = ("call_id", "run_id", "declaration", "resource", "_retired", "_starting", "_closing")

    def __init__(self, *, call_id, run_id, declaration):
        if (type(call_id) is not str or not call_id or type(run_id) is not str or not run_id
                or not isinstance(declaration, SourceAdapterDeclaration)
                or (declaration.role, declaration.component) != ("authored", "tts")
                or not callable(declaration.resource_factory)):
            raise SpeechRefused("workflow authored resource has no owned declaration")
        self.call_id, self.run_id, self.declaration = call_id, run_id, declaration
        self._retired = False
        self._starting = self._closing = None
        self.resource = declaration.resource_factory()

    def __repr__(self):
        return "<AuthoredSourceResourceBinding redacted>"

    def __reduce__(self):
        raise TypeError("workflow source resources cannot be serialized")

    async def start(self):
        if self._retired or self._starting is not None:
            raise SpeechRefused("workflow authored resource cannot start twice")
        self._starting = asyncio.create_task(self.resource.start())
        await join_owned_task(self._starting)

    def admit(self, journal, declaration):
        if (self._retired or type(journal) is not WorkflowSourceTransport
                or (journal.call_id, journal.run_id) != (self.call_id, self.run_id)
                or declaration is not self.declaration):
            raise SpeechRefused("workflow authored resource belongs to another owner")

    def attach(self, adapter):
        if self._retired:
            raise SpeechRefused("workflow authored resource is retired")
        adapter.bind_worker(self.resource)

    def retire(self):
        self._retired = True

    async def aclose(self):
        if self._closing is None:
            self.retire()
            async def close():
                if self._starting is not None:
                    try:
                        await self._starting
                    except BaseException:
                        pass
                await self.resource.aclose()
            self._closing = asyncio.create_task(close())
        await join_owned_task(self._closing)


class WorkflowSourceOwner:
    def __init__(self, *, journal, declaration, constructor, delivery=None, managed=None, resource=None):
        self.journal, self.declaration, self.constructor = journal, declaration, constructor
        self.managed = managed
        self.resource = resource
        self.source_id, self.owner_token = secrets.token_hex(24), secrets.token_hex(16)
        self.owner = self.adapter = self.reader = None
        self.delivery = delivery
        self._entered = self._ended = False
        self._closing = None
        self._provider_closing = None
        self._started_at = None
        self._live_grant = None

    async def _open(self):
        required = isinstance(self.declaration, SourceAdapterDeclaration) and self.declaration.resource_factory is not None
        if required != (self.resource is not None):
            raise SpeechRefused("workflow authored source has no matching resource")
        if required:
            if type(self.resource) is not AuthoredSourceResourceBinding:
                raise SpeechRefused("workflow authored source has no typed resource")
            self.resource.admit(self.journal, self.declaration)
        fact, permission = await self.journal.begin(source_attempt_id=self.source_id,
            logical_turn_id=self.delivery.attempt_id if self.delivery else secrets.token_hex(24),
            role=self.declaration.role, owner_token=self.owner_token,
            delivery=asdict(self.delivery) if self.delivery else None)
        if permission is None:
            raise SpeechRefused("workflow source has no fresh source permission")
        self.owner = SourceOwner(self.source_id, self.owner_token, fact["binding_digest"])
        credential = await self.journal.credentials(permission)
        require_source_credential(permission, credential, self.constructor, self.declaration, self.managed)
        self._entered = True
        self._started_at = asyncio.get_running_loop().time()
        self.adapter = build_workflow_source(permission, credential, self.constructor, self.declaration, self.managed)
        # What was really built (after the region and wire options were resolved), for attestation.
        settings = getattr(self.adapter, "effective_settings", None)
        self.effective_settings = settings() if callable(settings) else {}
        if self.resource is not None:
            failed = False
            try:
                self.resource.attach(self.adapter)
            except Exception:
                failed = True
            if failed:
                raise SpeechRefused("workflow authored source resource refused")

    async def open_live(self):
        """The live role: grant and lease while the line's delivery is capturing.

        The call's one session is the provider, so nothing is built here and
        nothing is entered: ``enter`` consumes the permission when the session
        is asked to speak. A line refused before that stays never connected.
        """
        if (not isinstance(self.declaration, SourceAdapterDeclaration) or self.declaration.role != "live"
                or self.resource is not None or self.delivery is None or self.owner is not None):
            raise SpeechRefused("workflow live source has no declared live output")
        fact, permission = await self.journal.begin(source_attempt_id=self.source_id,
            logical_turn_id=self.delivery.attempt_id, role="live", owner_token=self.owner_token,
            delivery=asdict(self.delivery))
        if permission is None:
            raise SpeechRefused("workflow source has no fresh source permission")
        self.owner = SourceOwner(self.source_id, self.owner_token, fact["binding_digest"])
        credential = await self.journal.credentials(permission)
        require_source_credential(permission, credential, self.constructor, self.declaration, self.managed)
        self._live_grant = permission, credential

    def enter(self, view):
        """At the provider touch: consume the one-time permission; ``view`` is this line's meter."""
        grant, self._live_grant = self._live_grant, None
        if grant is None or self._entered or self._closing is not None or not callable(getattr(view, "observation", None)):
            raise SpeechRefused("workflow live source has no fresh permission to enter")
        enter_session_source(*grant, self.constructor, self.declaration, self.managed)
        self._entered = True
        self._started_at = asyncio.get_running_loop().time()
        self.adapter = view

    def _observation(self):
        if self.adapter is None:
            return None
        value = self.adapter.observation()
        dimensions = {key for key, _ in self.declaration.max_native_units}
        if (not isinstance(value, SourceObservation) or type(value.metrics) is not tuple
                or len(value.metrics) > self.declaration.max_metric_count
                or type(value.terminal_ack) is not bool
                or value.status not in {"exact", "partial_unknown", "connect_unknown", "unreconciled"}):
            raise SpeechRefused("workflow source has no bounded native measurement")
        def units(pairs):
            if (type(pairs) is not tuple or len(pairs) != len(dimensions)
                    or any(type(pair) is not tuple or len(pair) != 2 for pair in pairs)
                    or {pair[0] for pair in pairs} != dimensions
                    or any(type(quantity) not in (int, float) or quantity < 0 or not math.isfinite(quantity)
                           for _, quantity in pairs)):
                raise SpeechRefused("workflow source changed its native dimensions")
            return dict(pairs)
        units(value.total_units)
        for metric in value.metrics:
            if not isinstance(metric, SourceMetric):
                raise SpeechRefused("workflow source has no typed native metric")
            units(metric.units)
        return value

    def _measured_rule(self):
        declaration = self.declaration
        return load_registry().measured_source_rule(declaration.role, declaration.provider, declaration.model)

    async def _persist_measured(self, rule, observation, closed, elapsed):
        """Seal the adapter's own measured duration at the rule's tariff.

        One stream metric carries the converted units under this source's own
        adapter-stream identity, then the exact terminal names its evidence. An
        adapter that was never built streamed nothing and measures zero.
        """
        seconds = 0.0 if observation is None else observation.measured_seconds
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise SpeechRefused("workflow source has no measured duration")
        declaration = self.declaration
        try:
            units = measured_source_units(rule, declaration.component, declaration.unit, seconds)
        except ValueError:
            raise SpeechRefused("workflow source has no measured duration") from None
        await self.journal.delta(self.owner, index=0, units=units, source_session_id=self.source_id)
        if closed:
            await self.journal.close(self.owner, connection_class="entered", observed_connection_seconds=elapsed)
        await self.journal.terminal(self.owner, status="exact", total_units=units, metric_count=1,
            observed_connection_seconds=elapsed, source_session_id=self.source_id,
            measurement={"evidence": MEASURED_EVIDENCE, "seconds": float(seconds)})

    async def _persist(self, closed):
        if self.owner is None:
            return
        dimensions = {key for key, _ in self.declaration.max_native_units}
        totals = dict.fromkeys(dimensions, 0)
        observation = self._observation()
        count, session_id = 0, ""
        status = "connect_unknown" if self._entered else "never_connected"
        elapsed = 0 if self._started_at is None else asyncio.get_running_loop().time() - self._started_at
        rule = self._measured_rule() if self._entered else None
        if rule is not None and (observation is None or not observation.metrics):
            # The worker's own measurement settles a source the registry
            # rules so, unless the provider stated usage for it: provider
            # facts are never mixed with a measurement.
            await self._persist_measured(rule, observation, closed, elapsed)
            return
        if observation is not None:
            for index, metric in enumerate(observation.metrics):
                await self.journal.delta(self.owner, index=index, units=dict(metric.units),
                    source_session_id=metric.session_id, metric_reference=metric.reference)
                count += 1
                for key, value in metric.units:
                    totals[key] += value
            session_id = observation.session_id
            # Numeric closure truth is independent of whether the caller
            # endpoint or authored text was usable for a business turn.
            if observation.terminal_ack:
                status, totals = observation.status, dict(observation.total_units)
            elif observation.metrics:
                status = "partial_unknown"
        # Every child is joined before a local close acknowledgement. That
        # close grants no monetary exactness or words and carries no key.
        if closed:
            await self.journal.close(self.owner, connection_class="entered", observed_connection_seconds=elapsed)
        await self.journal.terminal(self.owner, status=status, total_units=totals, metric_count=count,
            observed_connection_seconds=elapsed, source_session_id=session_id, measurement=None)

    def _provider_closure(self):
        """The provider's close, once: ``(closed, close_failed)``.

        Shared by :meth:`aclose` and by a listener releasing its words early,
        so the provider is closed exactly once whichever asks first.
        """
        if self._provider_closing is None:
            async def close_provider():
                if self.reader is not None:
                    self.reader.cancel()
                    await asyncio.gather(self.reader, return_exceptions=True)
                if self.adapter is None:
                    return False, False
                try:
                    await self.adapter.aclose()
                    return True, False
                except BaseException:
                    return False, True
            self._provider_closing = asyncio.create_task(close_provider())
        return self._provider_closing

    async def close_provider(self):
        """Close the provider now (joined despite cancellation); True when it closed cleanly."""
        closing = self._provider_closure()
        await join_owned_task(closing)
        closed, close_failed = closing.result()
        return closed and not close_failed

    async def aclose(self):
        if self._closing is None:
            async def close():
                closing = self._provider_closure()
                await join_owned_task(closing)
                closed, close_failed = closing.result()
                await self._persist(closed)
                if close_failed:
                    raise SpeechRefused("workflow source provider closure failed")
            self._closing = asyncio.create_task(close())
        await join_owned_task(self._closing)

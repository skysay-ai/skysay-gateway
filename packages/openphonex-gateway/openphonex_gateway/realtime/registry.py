# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Provider registry: the single source of truth for STT/LLM/TTS selection.

This module holds ONLY the generic machinery — the :class:`ProviderSpec`
data contract, the :class:`ProviderRegistry` (register / get / resolve), a
dotted-path loader, an entry-point loader and an alias table. It deliberately
imports nothing heavy and nothing from the wider ``telephony_ai`` control
plane, so importing it keeps the ``openphonex_gateway`` import-light guarantee.

The BUILT-IN provider specs (deepgram/fake, gemini/template, elevenlabs/tone)
are NOT registered here. They are registered from ``worker.py`` via a default
registry whose factory closures resolve the built-in adapter classes through
the worker module globals at call time, so that
``mock.patch("telephony_ai.media.realtime.worker.DeepgramStreamingSTT", ...)``
still swaps the constructed class. Registering built-ins here with direct
imports would break those patches — do not do it.

A ``ProviderSpec`` is the stable, third-party-facing plugin contract. A pip
installed package can advertise one three ways:

* an entry point in group ``openphonex.stt`` / ``openphonex.llm`` /
  ``openphonex.tts`` whose loaded value is a ``ProviderSpec`` (or a callable
  adapted into one);
* a dotted-path env var value, e.g. ``REALTIME_STT_PROVIDER=myco.stt:Spec``;
* a module attribute ``PROVIDER_SPEC`` (self-describing adapters).
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Component identifiers. These match the keys of the provider_components dict
# and the three independent env vars (STT/LLM/TTS).
STT = "stt"
LLM = "llm"
TTS = "tts"
COMPONENTS = (STT, LLM, TTS)

# Entry-point group per component. A third-party wheel advertises a provider by
# exposing e.g. ``[project.entry-points."openphonex.tts"] myvoice = "pkg:SPEC"``.
ENTRY_POINT_GROUPS = {
    STT: "openphonex.stt",
    LLM: "openphonex.llm",
    TTS: "openphonex.tts",
}

# Sentinel component name returned when a requested provider cannot be
# resolved. Callers map this back to the exact legacy RuntimeError strings.
UNSUPPORTED = "unsupported"


ConfigT = Any  # RealtimeWorkerConfig; kept loose to avoid importing worker here.
FactoryFn = Callable[[ConfigT], Any]
DescribeFn = Callable[[ConfigT], "dict[str, Any]"]
AvailabilityFn = Callable[[ConfigT], "list[str]"]


@dataclass(frozen=True)
class ProviderSpec:
    """A single registered provider for one component slot.

    ``factory`` builds the adapter instance from a ``RealtimeWorkerConfig``.
    ``describe`` returns the per-component ``provider_components`` fragment.
    ``availability`` returns a list of missing dependency / env names (empty
    when the provider is ready). ``real`` marks whether the provider is a real
    external/local provider (True) or a built-in deterministic fallback
    (fake/template/tone -> False).
    """

    name: str
    component: str
    factory: FactoryFn
    describe: Optional[DescribeFn] = None
    availability: Optional[AvailabilityFn] = None
    real: bool = True
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def build(self, config: ConfigT) -> Any:
        return self.factory(config)

    def describe_component(self, config: ConfigT) -> "dict[str, Any]":
        if self.describe is not None:
            return self.describe(config)
        return {
            "provider": self.name,
            "adapter": _callable_name(self.factory),
            "real": self.real,
            "model_id": "",
        }

    def missing(self, config: ConfigT) -> "list[str]":
        if self.availability is None:
            return []
        try:
            result = self.availability(config)
        except Exception:  # pragma: no cover - defensive; availability must not crash callers
            return []
        return [str(item) for item in (result or [])]


def _callable_name(obj: Any) -> str:
    return getattr(obj, "__name__", obj.__class__.__name__)


def _load_target(target: str) -> Any:
    """Import a ``pkg.module:attr`` (colon) or ``pkg.module.attr`` (dotted) target."""

    target = target.strip()
    if ":" in target:
        module_name, _, attr = target.partition(":")
    elif "." in target:
        module_name, _, attr = target.rpartition(".")
    else:
        raise ValueError(f"not an import target: {target!r}")
    module_name = module_name.strip()
    attr = attr.strip()
    if not module_name or not attr:
        raise ValueError(f"not an import target: {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _adapt(component: str, name: str, obj: Any) -> Optional[ProviderSpec]:
    """Coerce a loaded object into a ProviderSpec for the given component."""

    if obj is None:
        return None
    if isinstance(obj, ProviderSpec):
        return obj
    # A self-describing spec supplied via a classmethod ``.provider_spec()``.
    provider_spec = getattr(obj, "provider_spec", None)
    if callable(provider_spec):
        candidate = provider_spec()
        if isinstance(candidate, ProviderSpec):
            return candidate
    # Duck-typed spec: anything exposing .component + .factory.
    if hasattr(obj, "component") and hasattr(obj, "factory"):
        return obj  # type: ignore[return-value]
    # A raw class/callable adapter: adapt with a default describe/availability.
    if callable(obj):
        adapter_name = _callable_name(obj)

        def _factory(config: ConfigT, _target: Any = obj) -> Any:
            try:
                return _target(config)
            except TypeError:
                return _target()

        return ProviderSpec(
            name=name,
            component=component,
            factory=_factory,
            describe=lambda config, _n=name, _a=adapter_name: {
                "provider": _n,
                "adapter": _a,
                "real": True,
                "model_id": "",
            },
            availability=lambda config: [],
            real=True,
        )
    return None


class ProviderRegistry:
    """Owns three independent provider slots (stt, llm, tts).

    Resolution order for ``get(component, name)``:
      1. alias canonicalisation,
      2. an eagerly registered ``ProviderSpec``,
      3. a lazily registered import target (built-in local adapters),
      4. an entry point advertised by an installed third-party package,
      5. a dotted-path/colon import target from the raw name,
      6. otherwise ``None`` (the caller maps this to ``unsupported``).

    Steps 3-5 are wrapped so a broken plugin degrades to ``None`` for that name
    rather than crashing import.
    """

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], ProviderSpec] = {}
        self._lazy: dict[tuple[str, str], str] = {}
        self._aliases: dict[tuple[str, str], str] = {}

    # -- registration -----------------------------------------------------
    def register(self, spec: ProviderSpec) -> None:
        self._specs[(spec.component, spec.name)] = spec
        for alias in spec.aliases:
            self.register_alias(spec.component, alias, spec.name)

    def register_lazy(self, component: str, name: str, import_target: str) -> None:
        """Register a provider whose spec is imported on first resolution.

        Used for the built-in optional-extra local adapters so importing
        ``worker`` never imports faster-whisper / numpy / onnxruntime.
        """

        self._lazy[(component, name)] = import_target

    def register_alias(self, component: str, alias: str, canonical: str) -> None:
        self._aliases[(component, alias.strip().lower())] = canonical

    # -- resolution -------------------------------------------------------
    def canonical_name(self, component: str, name: str) -> str:
        key = (name or "").strip()
        return self._aliases.get((component, key.lower()), key)

    def get(self, component: str, name: str) -> Optional[ProviderSpec]:
        canonical = self.canonical_name(component, name)
        if not canonical:
            return None
        key = (component, canonical)
        spec = self._specs.get(key)
        if spec is not None:
            return spec
        # Lazily materialise a registered import target and cache it.
        target = self._lazy.get(key)
        if target is not None:
            loaded = self._safe_adapt(component, canonical, target)
            if loaded is not None:
                self._specs[key] = loaded
            return loaded
        # Entry points advertised by installed packages.
        loaded = self._from_entry_points(component, canonical)
        if loaded is not None:
            self._specs[key] = loaded
            return loaded
        # A raw dotted-path / colon import target from the env value itself.
        if ":" in canonical or "." in canonical:
            loaded = self._safe_adapt(component, canonical, canonical)
            if loaded is not None:
                self._specs[key] = loaded
            return loaded
        return None

    # ``resolve`` is an alias of ``get`` for readability at call sites.
    resolve = get

    def can_resolve(self, component: str, name: str) -> bool:
        return self.get(component, name) is not None

    def build(self, component: str, name: str, config: ConfigT) -> Any:
        spec = self.get(component, name)
        if spec is None:
            raise RuntimeError(f"No provider registered for {component}={name!r}")
        return spec.build(config)

    def describe(self, component: str, name: str, config: ConfigT) -> "dict[str, Any]":
        spec = self.get(component, name)
        if spec is None:
            return {"provider": name, "adapter": "", "real": False, "model_id": ""}
        return spec.describe_component(config)

    # -- internal ---------------------------------------------------------
    def _safe_adapt(self, component: str, name: str, target: str) -> Optional[ProviderSpec]:
        try:
            return _adapt(component, name, _load_target(target))
        except Exception:
            return None

    def _from_entry_points(self, component: str, name: str) -> Optional[ProviderSpec]:
        group = ENTRY_POINT_GROUPS.get(component)
        if not group:
            return None
        try:
            # Call through the module so ``mock.patch`` on
            # ``importlib.metadata.entry_points`` is honoured.
            entry_points = importlib.metadata.entry_points(group=group)
        except Exception:
            return None
        for entry_point in entry_points:
            if getattr(entry_point, "name", None) != name:
                continue
            try:
                loaded = entry_point.load()
            except Exception:
                return None
            return _adapt(component, name, loaded)
        return None

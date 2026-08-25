# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded SDK-usage -> component billing report adapter.

The LiveKit worker calls this only after media has stopped.  It intentionally
does not know about the database, wallet, telemetry artifact, credentials, or
network transport.  The caller passes ``AgentSession.usage.model_usage`` from
LiveKit Agents 1.6.x and this module returns the small numeric contract the
control plane accepts before terminal SIP settlement.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


class MeteringContractError(RuntimeError):
    """The selected adapter did not emit trustworthy native usage evidence."""


# Tool meter evidence is deliberately small and independent from telemetry.
# Tool arguments/results may contain customer data and must never be retained
# just to calculate a count.  The limits keep a malicious or runaway model
# from creating an unbounded teardown payload; an overflow fails settlement
# safely rather than silently undercounting a paid tool.
TOOL_USAGE_MAX_NAMES = 32
TOOL_USAGE_MAX_PER_NAME = 10_000
TOOL_USAGE_MAX_TOTAL = 100_000
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise MeteringContractError("SDK usage contains a non-numeric value")
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError) as exc:
        raise MeteringContractError("SDK usage contains a non-numeric value") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise MeteringContractError("SDK usage contains an invalid value")
    return parsed


def _whole_number(value: Any, *, field: str, maximum: int = 100_000_000) -> int:
    """A bounded provider-native count that cannot be truncated on settlement."""
    if value is None or value == "":
        raise MeteringContractError(f"{field} requires exact native-unit evidence")
    number = _number(value)
    if not number.is_integer():
        raise MeteringContractError(f"{field} must be a whole native-unit count")
    if number > maximum:
        raise MeteringContractError(f"{field} exceeds the bounded native-unit contract")
    return int(number)


def _expected_components(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    billing = config.get("component_billing")
    if not isinstance(billing, dict) or billing.get("usage_required") is not True:
        raise MeteringContractError("call is not configured for component-metered billing")
    components = billing.get("components")
    if not isinstance(components, list):
        raise MeteringContractError("component billing config omitted its pinned components")
    expected: dict[str, tuple[str, str]] = {}
    for item in components:
        if not isinstance(item, dict):
            continue
        component = _text(item.get("component"))
        if component not in {"stt", "llm", "tts"}:
            continue
        provider = _text(item.get("provider")).lower()
        model = _text(item.get("model"))
        if not provider or not model:
            raise MeteringContractError("pinned component identity is incomplete")
        expected[component] = (provider, model)
    if set(expected) != {"stt", "llm", "tts"}:
        raise MeteringContractError("pinned billing configuration is incomplete")
    return expected


def _field(usage: Any, name: str) -> Any:
    """One usage field, whether the record is an SDK object or its ``model_dump``.

    Production passes the SDK's own objects. Accepting the serialised shape too
    keeps a record readable after a JSON round-trip, which is how a test states
    a usage record without standing up a live session.
    """
    if isinstance(usage, dict):
        return usage.get(name, 0)
    return getattr(usage, name, 0)


def _tool_name(value: Any) -> str:
    name = _text(value)
    if not _TOOL_NAME_RE.fullmatch(name):
        raise MeteringContractError("tool usage contains an invalid tool name")
    return name


class SuccessfulToolUsageCounter:
    """Bounded worker-local counter for successful model tool invocations.

    ``build_livekit_model_tools`` registers exactly the schemas actually handed
    to the model, then calls :meth:`record_success` only after a wrapper gets a
    successful local/control-plane result.  It never keeps arguments, result
    bodies, provider output, or telemetry content.
    """

    def __init__(self) -> None:
        self._allowed_names: set[str] = set()
        self._counts: dict[str, int] = {}
        self._total = 0
        self._overflowed = False

    def register(self, raw_name: Any) -> None:
        name = _tool_name(raw_name)
        if name in self._allowed_names:
            return
        if len(self._allowed_names) >= TOOL_USAGE_MAX_NAMES:
            raise MeteringContractError("component billing tool contract exceeds the bounded name limit")
        self._allowed_names.add(name)

    def record_success(self, raw_name: Any) -> None:
        name = _tool_name(raw_name)
        if name not in self._allowed_names:
            raise MeteringContractError("successful tool is absent from the pinned tool contract")
        current = self._counts.get(name, 0)
        if current >= TOOL_USAGE_MAX_PER_NAME or self._total >= TOOL_USAGE_MAX_TOTAL:
            # Keep the media/tool result itself independent from billing.  The
            # later snapshot raises a metering contract error, retaining the
            # authorized hold for explicit reconciliation instead of guessing
            # or charging a truncated count.
            self._overflowed = True
            return
        self._counts[name] = current + 1
        self._total += 1

    def snapshot(self) -> dict[str, int]:
        if self._overflowed:
            raise MeteringContractError("component billing tool usage exceeded its bounded counter")
        return dict(sorted(self._counts.items()))


def demo_usage_from_model_usage(
    model_usage: Iterable[Any], expected_models: dict[str, str]
) -> dict[str, Any] | None:
    """The same native units, measured for a session nobody is charged for.

    The homepage demo needs the identical quantities as a paid call — the
    characters ElevenLabs synthesised, the tokens the model consumed — but not
    the same failure behaviour. A paid call has a pinned component contract and
    must refuse to bill when the evidence does not match it. A demo has no
    contract and no invoice: what it has is a spend ceiling that was, until
    this existed, being tracked in the wrong variable.

    So this reports what the SDK reported, aggregated by component kind, and
    returns ``None`` unless all three components supplied evidence. It never
    raises: it runs on a teardown path where an exception would cost the
    session's settle, and a missing cost line is worth far less than a
    reservation that expires against its whole reserved window.

    Absence is reported as absence — including *partial* absence, which is the
    dangerous kind. The control plane treats an unreported session as
    un-measured and falls back to a labelled estimate, which is the honest
    outcome; zeros would read as a free session or, worse, as a complete one
    that happens to omit a vendor.

    ``expected_models`` is the ``{component: model}`` triple the dispatch
    pinned, and a record naming anything else invalidates the measurement. The
    control plane prices these units against the rates it pinned for THAT
    triple, so a session that fell back to another model would have one
    provider's usage charged at another's rate — the same category of error as
    charging a per-character voice by the minute, one level down.
    """
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    kind_for_type = {"stt_usage": "stt", "llm_usage": "llm", "tts_usage": "tts"}
    seen: set[str] = set()
    for usage in model_usage or ():
        kind = kind_for_type.get(_text(_field(usage, "type")))
        if not kind:
            continue
        if _text(_field(usage, "model")) != _text(expected_models.get(kind)):
            return None
        try:
            if kind == "stt":
                record = {"audio_seconds": _number(_field(usage, "audio_duration"))}
            elif kind == "llm":
                input_tokens = _number(_field(usage, "input_tokens"))
                cached_tokens = _number(_field(usage, "input_cached_tokens"))
                # Same convention as the paid path: LiveKit's input_tokens
                # already includes cached input, so the two are reported
                # separately and never summed into a double count.
                if cached_tokens > input_tokens:
                    # Cache cannot exceed the total it is part of, so the record
                    # is internally inconsistent. Clamping to zero would price
                    # the excess at the cached rate, which is the cheaper one --
                    # provider telemetry drift would quietly understate spend.
                    raise MeteringContractError("LLM cached tokens exceed total input tokens")
                record = {
                    "input_tokens": input_tokens - cached_tokens,
                    "cached_input_tokens": cached_tokens,
                    "output_tokens": _number(_field(usage, "output_tokens")),
                }
            else:
                record = {
                    "characters": _number(_field(usage, "characters_count")),
                    "audio_seconds": _number(_field(usage, "audio_duration")),
                }
        except MeteringContractError:
            # A component reports one record per turn, so dropping a malformed
            # one and keeping its siblings would produce a total that looks
            # complete while knowably missing a turn. Once any record is
            # unreadable the measurement is incomplete, and an incomplete
            # measurement must present as absence -- the session then falls back
            # to the labelled estimate instead of to a figure we already know
            # is short. Still no exception: this is a teardown path.
            return None
        for field, value in record.items():
            totals[kind][field] += value
        seen.add(kind)
    if seen != {"stt", "llm", "tts"}:
        # All three or nothing, exactly as the paid path requires. A report
        # covering two components would emit the third as zero, and a zero is
        # indistinguishable from "this component cost nothing" once it reaches
        # the control plane — so a partial teardown would produce a
        # complete-looking settled cost that quietly omits a whole vendor.
        # Reporting nothing falls back to a labelled estimate instead.
        return None
    units = {
        "stt_audio_seconds": totals["stt"]["audio_seconds"],
        "llm_input_tokens": int(totals["llm"]["input_tokens"]),
        "llm_cached_input_tokens": int(totals["llm"]["cached_input_tokens"]),
        "llm_output_tokens": int(totals["llm"]["output_tokens"]),
        "tts_characters": int(totals["tts"]["characters"]),
        "tts_audio_seconds": totals["tts"]["audio_seconds"],
    }
    # This object is JSON-serialised into the settle request, and `json.dumps`
    # happily writes a bare NaN that no JSON parser will read back. A rejected
    # settle costs the session its whole reserved window, so a value that could
    # not survive the trip drops the cost line instead of the settle.
    if not all(math.isfinite(value) for value in units.values()):
        return None
    return units


def normalized_usage_from_model_usage(
    config: dict[str, Any],
    model_usage: Iterable[Any],
    *,
    tool_units: dict[str, int] | None = None,
    tts_utf8_bytes: int | None = None,
    soniox_usage: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return exact native usage for the pinned SDK adapter identities.

    LiveKit's ``LLMModelUsage.input_tokens`` includes cached input.  The
    control-plane contract therefore sends *non-cached* input separately from
    cached input, avoiding a double charge.  Agents 1.6.6 does not expose a
    separate Gemini thinking/reasoning count; output already includes billed
    thinking, so ``llm_reasoning_tokens`` remains zero rather than invented.
    """
    expected = _expected_components(config)
    expected_gemini_tts = expected["tts"][0] == "gemini"
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    gemini_tts_input_tokens = 0
    gemini_tts_output_tokens = 0
    seen: set[str] = set()
    kind_for_type = {"stt_usage": "stt", "llm_usage": "llm", "tts_usage": "tts"}
    for usage in model_usage:
        kind = kind_for_type.get(_text(getattr(usage, "type", "")))
        if not kind:
            continue
        provider = _text(getattr(usage, "provider", "")).lower()
        model = _text(getattr(usage, "model", ""))
        if (provider, model) != expected[kind]:
            raise MeteringContractError(
                f"{kind} SDK usage identity does not match the pinned component contract"
            )
        seen.add(kind)
        if kind == "llm":
            input_tokens = _number(getattr(usage, "input_tokens", 0))
            cached_tokens = _number(getattr(usage, "input_cached_tokens", 0))
            if cached_tokens > input_tokens:
                raise MeteringContractError("LLM cached tokens exceed total input tokens")
            totals[kind]["input_tokens"] += input_tokens - cached_tokens
            totals[kind]["cached_input_tokens"] += cached_tokens
            totals[kind]["output_tokens"] += _number(getattr(usage, "output_tokens", 0))
        elif kind in {"stt", "tts"}:
            totals[kind]["audio_seconds"] += _number(getattr(usage, "audio_duration", 0))
            if kind == "tts":
                totals[kind]["characters"] += _number(getattr(usage, "characters_count", 0))
                if expected_gemini_tts:
                    # The OpenPhonex Gemini adapter passes the provider's exact
                    # cumulative prompt/audio token counts through TTSMetrics.
                    # A missing field is not an SDK zero: paid settlement must
                    # retain its reservation rather than guess from duration.
                    gemini_tts_input_tokens += _whole_number(
                        getattr(usage, "input_tokens", None),
                        field="Gemini TTS input token usage",
                    )
                    gemini_tts_output_tokens += _whole_number(
                        getattr(usage, "output_tokens", None),
                        field="Gemini TTS output token usage",
                    )
        else:  # Defensive future-proofing if kind_for_type gains a new value.
            raise MeteringContractError("SDK usage has an unsupported component kind")
    # Absence is a metering failure, not a signal to estimate from duration.
    # A zero-valued native metrics object still counts as evidence.
    if seen != {"stt", "llm", "tts"}:
        missing = ", ".join(sorted({"stt", "llm", "tts"} - seen))
        raise MeteringContractError(f"SDK usage omitted pinned component metrics: {missing}")
    safe_tools: dict[str, int] = {}
    for name, count in (tool_units or {}).items():
        key = _tool_name(name)
        parsed = _number(count)
        if not parsed.is_integer():
            raise MeteringContractError("tool usage must be a whole-unit count")
        units = int(parsed)
        if units > TOOL_USAGE_MAX_PER_NAME:
            raise MeteringContractError("tool usage exceeds the bounded per-tool limit")
        safe_tools[key] = units
    if len(safe_tools) > TOOL_USAGE_MAX_NAMES or sum(safe_tools.values()) > TOOL_USAGE_MAX_TOTAL:
        raise MeteringContractError("tool usage exceeds the bounded billing contract")
    expected_tts_provider = expected["tts"][0]
    if expected_tts_provider == "fishaudio":
        if tts_utf8_bytes is None or tts_utf8_bytes == "":
            raise MeteringContractError("Fish Audio TTS usage omitted exact UTF-8 byte evidence")
        native_tts_utf8_bytes = _whole_number(
            tts_utf8_bytes,
            field="Fish Audio TTS UTF-8 byte usage",
            maximum=400_000_000,
        )
    elif tts_utf8_bytes is not None and _number(tts_utf8_bytes) != 0:
        # A counter belonging to another provider proves the worker's stack
        # identity drifted. Keep its unpriced integer out of settlement rather
        # than carrying it as ambiguous evidence.
        raise MeteringContractError("unexpected UTF-8 byte evidence for the selected TTS provider")
    else:
        native_tts_utf8_bytes = 0
    token_dimensions = {
        "stt": ("input_audio_tokens", "output_text_tokens"),
        "tts": ("input_text_tokens", "output_audio_tokens"),
    }
    expected_soniox = {
        component for component, (provider, _model) in expected.items() if provider == "soniox"
    }
    if expected_soniox - {"stt", "tts"}:
        raise MeteringContractError("Soniox is only supported for exact STT/TTS token evidence")
    if soniox_usage is None:
        supplied_soniox: Mapping[str, Mapping[str, Any]] = {}
    elif not isinstance(soniox_usage, Mapping):
        raise MeteringContractError("Soniox usage evidence must be a component mapping")
    else:
        supplied_soniox = soniox_usage
    supplied_names = {str(name) for name in supplied_soniox}
    if supplied_names != expected_soniox:
        raise MeteringContractError("Soniox usage evidence does not match the pinned component contract")
    native_token_usage: dict[str, int] = {
        "stt_input_audio_tokens": 0,
        "stt_output_text_tokens": 0,
        "tts_input_text_tokens": 0,
        "tts_output_audio_tokens": 0,
    }
    for component in sorted(expected_soniox):
        raw_component = supplied_soniox.get(component)
        if not isinstance(raw_component, Mapping):
            raise MeteringContractError("Soniox usage evidence component is invalid")
        dimensions = token_dimensions[component]
        if set(raw_component) != set(dimensions):
            raise MeteringContractError("Soniox usage evidence dimensions are incomplete")
        for dimension in dimensions:
            native_token_usage[f"{component}_{dimension}"] = _whole_number(
                raw_component[dimension], field=f"Soniox {component} {dimension}"
            )
    if expected_gemini_tts:
        native_token_usage["tts_input_text_tokens"] = gemini_tts_input_tokens
        native_token_usage["tts_output_audio_tokens"] = gemini_tts_output_tokens
    return {
        "source": "livekit_agent_sdk",
        "stt_audio_seconds": totals["stt"]["audio_seconds"],
        "stt_input_audio_tokens": native_token_usage["stt_input_audio_tokens"],
        "stt_output_text_tokens": native_token_usage["stt_output_text_tokens"],
        "llm_input_tokens": int(totals["llm"]["input_tokens"]),
        "llm_cached_input_tokens": int(totals["llm"]["cached_input_tokens"]),
        # LiveKit Agents 1.6.6 exposes no separate reasoning count; Gemini's
        # provider-billed thinking is part of output_tokens above.
        "llm_reasoning_tokens": 0,
        "llm_output_tokens": int(totals["llm"]["output_tokens"]),
        "tts_characters": int(totals["tts"]["characters"]),
        "tts_audio_seconds": totals["tts"]["audio_seconds"],
        "tts_utf8_bytes": int(native_tts_utf8_bytes),
        "tts_input_text_tokens": native_token_usage["tts_input_text_tokens"],
        "tts_output_audio_tokens": native_token_usage["tts_output_audio_tokens"],
        "tool_units": safe_tools,
    }

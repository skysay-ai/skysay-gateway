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
# This is the same frozen Follow-caller settlement bound enforced by the
# switcher and control plane. Keep the direct worker helper bounded too: its
# input may be a generator supplied by a future caller that bypasses today's
# runtime queue guard.
PROFILE_USAGE_SEGMENT_MAX = 64
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


_REFUSED_RECORD_MAX = 64
_REFUSED_REASON_MAX = 512
_REFUSED_IDENTIFIER_MAX = 128
_REFUSED_RECORD_NUMERIC_FIELDS = (
    "characters_count",
    "audio_duration",
    # A GPT-Live session's own length: what its provider bills by, and the one
    # number a released unanswered call's refused report must not lose.
    "session_duration",
    "input_tokens",
    "output_tokens",
    "input_cached_tokens",
)
#: Native meter settlement field -> the refused-metering counter it fills.
_REFUSED_NATIVE_COUNTERS = {"tts_characters": "tts_submitted_characters", "tts_utf8_bytes": "tts_utf8_bytes"}
_REFUSED_SONIOX_DIMENSIONS = {
    "stt": ("input_audio_tokens", "output_text_tokens"),
    "tts": ("input_text_tokens", "output_audio_tokens"),
}


def _diagnostic_text(value: Any, *, maximum: int) -> str:
    return str(value or "").strip()[:maximum]


def _diagnostic_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def diagnostic_counter_snapshot(counter: Any) -> Any:
    """A poisoned native counter must not discard other components' evidence."""
    try:
        return counter.snapshot() if counter is not None else None
    except Exception:
        return None


def snapshot_refused_metering(
    reason: str,
    model_usage: Iterable[Any],
    *,
    tts_native_usage: Any = None,
    soniox_usage: Mapping[str, Any] | None = None,
    native_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten the counts a refused report still held, for durable diagnosis.

    This is not billable evidence. It must never raise: it runs on the
    teardown path after the metering contract has already failed.
    """
    records: list[dict[str, Any]] = []
    for usage in model_usage:
        if len(records) >= _REFUSED_RECORD_MAX:
            break
        kind = _diagnostic_text(getattr(usage, "type", ""), maximum=_REFUSED_IDENTIFIER_MAX)
        provider = _diagnostic_text(getattr(usage, "provider", ""), maximum=_REFUSED_IDENTIFIER_MAX)
        if not kind or not provider:
            continue
        record: dict[str, Any] = {
            "type": kind,
            "provider": provider,
            "model": _diagnostic_text(getattr(usage, "model", ""), maximum=_REFUSED_IDENTIFIER_MAX),
        }
        for field in _REFUSED_RECORD_NUMERIC_FIELDS:
            number = _diagnostic_number(getattr(usage, field, None))
            if number is not None:
                record[field] = number
        records.append(record)
    counters: dict[str, float] = {}
    if isinstance(tts_native_usage, Mapping):
        # The control plane's refused-metering counters name the two
        # worker-measured units; a provider-reported unit (tokens) is carried
        # by ``native_diagnostics`` alone.
        for field, counter in _REFUSED_NATIVE_COUNTERS.items():
            number = _diagnostic_number(tts_native_usage.get(field))
            if number is not None:
                counters[counter] = number
    soniox: dict[str, dict[str, float]] = {}
    if isinstance(soniox_usage, Mapping):
        for component in ("stt", "tts"):
            dims = soniox_usage.get(component)
            if not isinstance(dims, Mapping):
                continue
            cleaned: dict[str, float] = {}
            for key in _REFUSED_SONIOX_DIMENSIONS[component]:
                number = _diagnostic_number(dims.get(key))
                if number is not None:
                    cleaned[key] = number
            if cleaned:
                soniox[component] = cleaned
    diagnostic = {}
    if isinstance(native_diagnostics, Mapping):
        for key in ("completed_total", "uncertain_requests", "uncertain_accepted_characters"):
            number = _diagnostic_number(native_diagnostics.get(key))
            if number is not None:
                diagnostic[key] = min(number, 100_000_000)
        diagnostic["overflowed"] = native_diagnostics.get("overflowed") is True
    return {
        **({"native_diagnostics": diagnostic} if diagnostic else {}),
        "reason": _diagnostic_text(reason, maximum=_REFUSED_REASON_MAX),
        "records": records,
        "counters": counters,
        "soniox": soniox,
    }


# LiveKit stamps a HUMAN-READABLE vendor label on every usage record
# (``STT.provider`` / ``TTS.provider`` / ``LLM.provider``), while the pinned
# reservation names the catalog's provider *identifier*.  Those are two
# different namespaces that happened to coincide once lowercased for five of
# the six runnable providers, so comparing them directly looked correct until a
# Google STT model was selected: the plugin reports ``Google Cloud Platform``,
# the catalog identifier is ``google``, and every such call therefore failed
# its metering contract and stranded the customer's reservation in
# ``awaiting_usage`` with no terminal path.
#
# Measured on 2026-08-27 against livekit-agents==1.6.6 and re-verified against
# 1.7.1 on 2026-08-29, with the pinned plugin
# set in ``agent/requirements.txt``.  This is an explicit allowlist and an
# unrecognised label still fails closed: silently coercing an unknown vendor
# label into a catalog identifier is how one provider's units get charged at
# another provider's pinned rate.
#
# ``Vertex AI`` is DELIBERATELY absent. The Google LLM and Gemini TTS plugins
# report it instead of ``Gemini`` whenever their client is Vertex-backed, and
# it is a different vendor relationship with its own rate card -- mapping it
# onto ``gemini`` would price one vendor's units at another's. Nothing in this
# worker configures Vertex today, so a call reporting it means the deployment
# changed underneath the catalog, and failing closed is the correct answer
# until a ``vertex`` provider exists with rates of its own.
#: The settlement fields a native TTS meter can carry, one set per unit
#: (``native_tts_usage.RequestBilling.settlement_fields``). Which set a call
#: carries is the TTS row's data; the settlement only checks it is one whole set.
NATIVE_TTS_FIELD_SETS = (
    frozenset({"tts_characters"}),
    frozenset({"tts_utf8_bytes"}),
    frozenset({"tts_input_text_tokens", "tts_output_audio_tokens"}),
)
_NATIVE_TTS_FIELD_MAXIMUM = {"tts_utf8_bytes": 400_000_000}


def _native_tts_usage(value: Any) -> dict[str, int]:
    """Validate one native TTS meter snapshot: exactly one unit's whole numbers."""
    if not isinstance(value, Mapping) or frozenset(value) not in NATIVE_TTS_FIELD_SETS:
        raise MeteringContractError("native TTS usage evidence is not one metered unit")
    return {
        field: _whole_number(
            value[field],
            field=f"native TTS {field}",
            maximum=_NATIVE_TTS_FIELD_MAXIMUM.get(field, 100_000_000),
        )
        for field in sorted(value)
    }

_SDK_PROVIDER_IDS = {
    # The adapters this worker builds report the catalog vendor directly
    # (see ``azure_speech.py``). The plugin's own two labels are mapped
    # as well so a raw plugin instance -- a future code path, or a test double
    # copied from upstream -- resolves to the same vendor instead of failing
    # closed on a spelling.
    "azure": "azure",
    "azure stt": "azure",
    "azure tts": "azure",
    "cartesia": "cartesia",
    "deepgram": "deepgram",
    "elevenlabs": "elevenlabs",
    "fishaudio": "fishaudio",
    "gemini": "gemini",
    "google": "google",
    "google cloud platform": "google",
    "soniox": "soniox",
    "openai": "openai",
    "api.openai.com": "openai",
}


def catalog_provider_id(sdk_provider: Any) -> str:
    """The catalog provider identifier for one SDK usage record's vendor label.

    Returns ``""`` for an unrecognised label so the caller can fail closed with
    the identity error it already raises, rather than inventing an identity.
    """
    return _SDK_PROVIDER_IDS.get(_text(sdk_provider).lower(), "")


# Voice-to-voice engines report both pinned components as ``llm_usage``
# records from one plugin. GPT-Live splits duration (``gpt-live-1``) from a
# delegated Responses backend. Gemini Live prices duration on TTS and keeps
# the LLM line at zero token rates; the same mapper sends the live model's
# duration record to TTS rather than leaving tts_audio_seconds at 0.
GPT_LIVE_DURATION_MODEL = "gpt-live-1"
GPT_LIVE_BACKEND_MODELS = frozenset({"gpt-5.6-luna", "gpt-5.6-terra"})
GEMINI_LIVE_DURATION_MODEL = "gemini-3.8-live"


def _s2s_live_record_kind(
    kind: str | None, model: str, expected: Mapping[str, tuple[str, str]]
) -> str | None:
    """Map one live-engine ``llm_usage`` record to the pinned component it evidences.

    Only an ``llm_usage`` record is remapped, as in the normalizer: a record of
    another type naming these models is not evidence for either component.
    Gemini Live has one realtime session priced on TTS: the session model may
    be the base or extended-thinking name, and both evidence the duration line.
    GPT-Live still splits duration (``gpt-live-1``) from a delegated backend.
    """
    if kind != "llm":
        return kind
    tts = expected.get("tts")
    llm = expected.get("llm")
    if tts and tts[1] == GEMINI_LIVE_DURATION_MODEL:
        if model == tts[1] or (llm and model == llm[1]):
            return "tts"
        return kind
    if tts and model == tts[1]:
        return "tts"
    if llm and model == llm[1]:
        return "llm"
    return kind


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
    gpt_live = False
    voice = config.get("voice") if isinstance(config.get("voice"), dict) else {}
    frontend = str(config.get("voice_frontend") or voice.get("voice_frontend") or "").strip().lower()
    if frontend == "gpt_live":
        gpt_live = True
    if frontend == "gemini_live":
        if set(expected) != {"llm", "tts"}:
            raise MeteringContractError("pinned Gemini Live billing configuration is incomplete")
        return expected
    if gpt_live or expected.get("tts") == ("openai", GPT_LIVE_DURATION_MODEL):
        if set(expected) != {"llm", "tts"}:
            raise MeteringContractError("pinned GPT-Live billing configuration is incomplete")
        return expected
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


def _native_count(usage: Any, name: str) -> int | None:
    """A provider-native whole count, or ``None`` when the record has none.

    Missing and ``None`` are absence. Anything present must be a whole,
    non-negative number; a boolean, fraction, NaN or negative value makes the
    measurement unreadable rather than being coerced.
    """
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeteringContractError("SDK usage contains a non-numeric token count")
    # Compared without converting an int to float, which raises for a huge
    # one; a count past 2**53 is not a whole number a float can carry.
    if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
        raise MeteringContractError("SDK usage contains an invalid token count")
    if value < 0 or value > 2**53:
        raise MeteringContractError("SDK usage contains an invalid token count")
    return int(value)


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
        self._action_ids: set[str] = set()

    def register(self, raw_name: Any) -> None:
        name = _tool_name(raw_name)
        if name in self._allowed_names:
            return
        if len(self._allowed_names) >= TOOL_USAGE_MAX_NAMES:
            raise MeteringContractError("component billing tool contract exceeds the bounded name limit")
        self._allowed_names.add(name)

    def record_success(self, raw_name: Any, *, action_id: str | None = None) -> None:
        name = _tool_name(raw_name)
        if name not in self._allowed_names:
            raise MeteringContractError("successful tool is absent from the pinned tool contract")
        if action_id is not None:
            if not isinstance(action_id, str) or not action_id or len(action_id) > 128:
                raise MeteringContractError("action usage has an invalid identity")
            if action_id in self._action_ids:
                return
            if len(self._action_ids) >= TOOL_USAGE_MAX_TOTAL:
                self._overflowed = True
                return
            self._action_ids.add(action_id)
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
    # A token-billed voice's native counts, reported only when EVERY TTS record
    # carried both as whole numbers. One record without them makes the pair
    # absent (never zero); which component needs them is the control plane's
    # frozen billing unit, not this worker's guess about a provider.
    tts_tokens_complete = True
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
                input_text = _native_count(usage, "input_tokens")
                output_audio = _native_count(usage, "output_tokens")
                if input_text is None or output_audio is None:
                    tts_tokens_complete = False
                else:
                    record["input_text_tokens"] = float(input_text)
                    record["output_audio_tokens"] = float(output_audio)
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
    if tts_tokens_complete:
        units["tts_input_text_tokens"] = int(totals["tts"]["input_text_tokens"])
        units["tts_output_audio_tokens"] = int(totals["tts"]["output_audio_tokens"])
    # This object is JSON-serialised into the settle request, and `json.dumps`
    # happily writes a bare NaN that no JSON parser will read back. A rejected
    # settle costs the session its whole reserved window, so a value that could
    # not survive the trip drops the cost line instead of the settle.
    if not all(math.isfinite(value) for value in units.values()):
        return None
    return units


def unmeasured_pinned_components(
    config: dict[str, Any], model_usage: Iterable[Any], *, tts_native_usage: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Pinned components whose zero in the report is an absence, not a metric.

    Settlement cannot tell the two apart -- both are zero native units -- but
    the worker can, and an operator reading a settled call for 0 STT seconds
    needs to know whether the far end was silent or whether the adapter never
    reported. Ordered so the audit line is stable.
    """
    expected = _expected_components(config)
    expected_s2s_live = "stt" not in expected
    kind_for_type = {"stt_usage": "stt", "llm_usage": "llm", "tts_usage": "tts"}
    seen: set[str] = set()
    for usage in model_usage:
        kind = kind_for_type.get(_text(getattr(usage, "type", "")))
        model = _text(getattr(usage, "model", ""))
        if expected_s2s_live:
            # The live model's duration and any backend tokens both arrive
            # as llm_usage, told apart by model. Without this the audit line
            # named the duration component unmeasured on every such call.
            # This is the audit line only: the normalizer below still owns
            # the one-summary rule and the money path.
            kind = _s2s_live_record_kind(kind, model, expected)
        tts = expected.get("tts")
        if (
            kind == "tts"
            and tts
            and tts[1] == GEMINI_LIVE_DURATION_MODEL
            and model in {tts[1], (expected.get("llm") or ("", ""))[1]}
        ):
            # Plugin may label the session google; the pin is gemini.
            seen.add("tts")
            continue
        if kind and (
            catalog_provider_id(getattr(usage, "provider", "")),
            model,
        ) == expected[kind]:
            seen.add(kind)
    # The native TTS meter is an explicit observation of every provider request
    # rather than a missing SDK metric, so a TTS it measured is not unmeasured.
    if tts_native_usage is not None:
        seen.add("tts")
    return tuple(sorted(set(expected) - seen))


def normalized_usage_from_model_usage(
    config: dict[str, Any],
    model_usage: Iterable[Any],
    *,
    tool_units: dict[str, int] | None = None,
    tts_native_usage: Mapping[str, Any] | None = None,
    soniox_usage: Mapping[str, Mapping[str, Any]] | None = None,
    allow_unmetered_components: bool = False,
) -> dict[str, Any]:
    """Return exact native usage for the pinned SDK adapter identities.

    LiveKit's ``LLMModelUsage.input_tokens`` includes cached input.  The
    control-plane contract therefore sends *non-cached* input separately from
    cached input, avoiding a double charge.  Agents 1.7.1 does not expose a
    separate Gemini thinking/reasoning count; output already includes billed
    thinking, so ``llm_reasoning_tokens`` remains zero rather than invented.

    A pinned component that emitted no metric at all reports zero units rather
    than failing the whole report; see the comment on the ``seen`` check below
    for why that is a measurement and not an estimate.

    A cascade TTS's billable quantity is ``tts_native_usage``: the snapshot of
    the worker's native unit meter, in the unit the TTS row's data names
    (characters, UTF-8 bytes or provider-reported tokens). The SDK's TTS
    events still prove the provider/model identity and carry synthesized
    audio seconds, but their character and token counts are never billed: they
    omit every request the caller interrupted.
    """
    expected = _expected_components(config)
    expected_s2s_live = "stt" not in expected
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seen: set[str] = set()
    kind_for_type = {"stt_usage": "stt", "llm_usage": "llm", "tts_usage": "tts"}
    for usage in model_usage:
        usage_type = _text(getattr(usage, "type", "") if not isinstance(usage, dict) else usage.get("type", ""))
        if expected_s2s_live and usage_type in {"llm_metrics", "realtime_model_metrics"}:
            raise MeteringContractError("live-engine billing requires collected model usage, not raw metrics")
        kind = kind_for_type.get(usage_type)
        if not kind:
            continue
        provider_raw = getattr(usage, "provider", "") if not isinstance(usage, dict) else usage.get("provider", "")
        model = _text(getattr(usage, "model", "") if not isinstance(usage, dict) else usage.get("model", ""))
        metadata = getattr(usage, "metadata", None) if not isinstance(usage, dict) else usage.get("metadata")
        if not model and metadata is not None:
            model = _text(
                getattr(metadata, "model_name", "")
                if not isinstance(metadata, dict)
                else metadata.get("model_name", "")
            )
        provider = catalog_provider_id(provider_raw)
        remapped = _s2s_live_record_kind(kind, model, expected) if expected_s2s_live else kind
        if expected_s2s_live and remapped == "tts" and kind == "llm":
            if usage_type != "llm_usage" or "tts" in seen:
                raise MeteringContractError("live-engine duration requires one collected model summary")
            kind = "tts"
            tts = expected.get("tts")
            if tts and tts[1] == GEMINI_LIVE_DURATION_MODEL:
                # The Google realtime plugin may name provider "google"; the
                # duration line is the pinned Gemini Live identity.
                provider, model = tts
        elif expected_s2s_live and remapped == "llm":
            kind = "llm"
        if expected_s2s_live and not provider:
            meta_provider = ""
            if metadata is not None:
                meta_provider = _text(
                    getattr(metadata, "model_provider", "")
                    if not isinstance(metadata, dict)
                    else metadata.get("model_provider", "")
                )
            tts = expected.get("tts")
            llm = expected.get("llm")
            if tts and model == tts[1]:
                provider = tts[0]
            elif llm and model == llm[1]:
                provider = llm[0]
            elif "openai" in meta_provider.lower() or model == GPT_LIVE_DURATION_MODEL or model in GPT_LIVE_BACKEND_MODELS:
                provider = "openai"
        if (provider, model) != expected[kind]:
            # Name BOTH identities. This refusal strands a reservation until an
            # operator releases it, and the previous message said only that
            # something did not match -- which cost one investigation a
            # production database read, a host log dive and an SDK teardown to
            # answer a question one log line should have answered. Provider and
            # model identifiers are catalog values, never credentials.
            observed = f"{_text(getattr(usage, 'provider', ''))!r}/{model!r}"
            wanted = f"{expected[kind][0]!r}/{expected[kind][1]!r}"
            raise MeteringContractError(
                f"{kind} SDK usage identity does not match the pinned component "
                f"contract: reported {observed}, pinned {wanted}"
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
            duration_field = "session_duration" if expected_s2s_live and kind == "tts" else "audio_duration"
            duration_value = (
                usage.get(duration_field, usage.get("audio_duration", 0))
                if isinstance(usage, dict)
                else getattr(usage, duration_field, getattr(usage, "audio_duration", 0))
            )
            totals[kind]["audio_seconds"] += _number(duration_value)
        else:  # Defensive future-proofing if kind_for_type gains a new value.
            raise MeteringContractError("SDK usage has an unsupported component kind")
    # Where the TTS quantity comes from is decided by which evidence the stack
    # carries, not by provider: Soniox's usage tracker for a Soniox voice, no
    # TTS quantity beyond duration for a voice-to-voice engine, and the native
    # unit meter for every other cascade voice -- required, because only it
    # sees a request the caller interrupted.
    # Soniox's own usage tracker is the one other TTS evidence source (its
    # contract is checked with the STT side below).
    soniox_tts = expected.get("tts", ("", ""))[0] == "soniox"
    if expected_s2s_live or soniox_tts:
        if tts_native_usage is not None:
            raise MeteringContractError("unexpected native TTS meter evidence for the selected TTS component")
        native_tts: dict[str, int] = {}
    elif tts_native_usage is None:
        raise MeteringContractError("TTS usage omitted the native meter evidence")
    else:
        native_tts = _native_tts_usage(tts_native_usage)
    # A session that produced no component metrics at all never ran: an agent
    # that joined a room nobody answered has nothing to meter, and reporting
    # an all-zero payload for it would only be rejected by the control plane's
    # "usage must contain native provider usage" gate. The terminal lifecycle
    # releases that hold, so keep failing closed here.
    #
    # ``allow_unmetered_components`` scopes that reasoning to a whole call.
    # One language SEGMENT of a switched call can legitimately meter nothing:
    # the recognizer is one shared call-wide identity billed to the opening
    # segment, so a segment where the caller spoke and the agent never did --
    # they hung up right after the switch, say -- has measured zeros for
    # every component it owns. The segmented caller enforces the never-ran
    # protection itself, across all segments, where it is actually decidable.
    if (
        not seen
        and not allow_unmetered_components
        and not any(native_tts.values())
    ):
        raise MeteringContractError("SDK usage omitted every pinned component metric")
    # A component that IS pinned, WAS constructed for this session, and emitted
    # no metric consumed no billable units. That is a measured zero, not an
    # estimate: the SDK's collector only materialises a record when a metric
    # fires, so absence and zero are the same observation on its side.
    #
    # This is what a one-way call looks like -- the agent speaks (llm + tts)
    # and the far end is never transcribed (no stt) -- and refusing it used to
    # hold the whole reservation indefinitely. The alternative already in
    # production is worse in exactly the same direction: the operator sweeper
    # eventually releases 100% of the hold, charging zero for the components
    # that *did* report as well. Zero-filling can only ever undercharge
    # relative to the truth, never overcharge, and it lets the call reach a
    # terminal money state on its own. ``unmeasured_pinned_components`` names
    # the filled components so the worker can log which zeros were measured
    # absences rather than measured zeros; ``totals`` is a defaultdict, so the
    # units below already read 0 for a component that emitted nothing.
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
    for field in ("tts_input_text_tokens", "tts_output_audio_tokens"):
        if field in native_tts:
            native_token_usage[field] = native_tts[field]
    return {
        "source": "livekit_agent_sdk",
        "stt_audio_seconds": totals["stt"]["audio_seconds"],
        "stt_input_audio_tokens": native_token_usage["stt_input_audio_tokens"],
        "stt_output_text_tokens": native_token_usage["stt_output_text_tokens"],
        "llm_input_tokens": int(totals["llm"]["input_tokens"]),
        "llm_cached_input_tokens": int(totals["llm"]["cached_input_tokens"]),
        # LiveKit Agents 1.7.1 exposes no separate reasoning count; Gemini's
        # provider-billed thinking is part of output_tokens above.
        "llm_reasoning_tokens": 0,
        "llm_output_tokens": int(totals["llm"]["output_tokens"]),
        "tts_characters": native_tts.get("tts_characters", 0),
        "tts_audio_seconds": totals["tts"]["audio_seconds"],
        "tts_utf8_bytes": native_tts.get("tts_utf8_bytes", 0),
        "tts_input_text_tokens": native_token_usage["tts_input_text_tokens"],
        "tts_output_audio_tokens": native_token_usage["tts_output_audio_tokens"],
        "tool_units": safe_tools,
    }


def component_usage_report_required(
    payload: Mapping[str, Any], *, confirmed_unanswered: bool
) -> bool:
    """Check an already-normalized whole-call report before staging its outbox.

    Zero metrics are a valid observation, but not a settlement report. The
    lifecycle owner decides whether that observation means an unanswered call
    or missing metering on a call that did run. Keep positive pre-answer usage.
    """
    metrics = (
        [segment["usage"] for segment in payload["segments"]]
        if "segments" in payload
        else [payload]
    )
    has_units = any(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for usage in metrics
        for value in usage.values()
    ) or any(value > 0 for value in payload.get("tool_units", {}).values())
    if not has_units and not confirmed_unanswered:
        raise MeteringContractError("answered or unknown call has empty component usage")
    return has_units


def normalized_usage_from_profile_segments(
    config: dict[str, Any],
    profile_usage_segments: Iterable[Mapping[str, Any]],
    *,
    tool_units: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Price immutable language segments without mixing provider identities.

    ``component_billing.profile_segments`` is created by admission from one
    frozen publication.  The worker supplies only language/publication/boundary
    identifiers plus native SDK usage objects.  A foreign provider/model is
    rejected by the existing single-stack normalizer *inside its own segment*;
    aggregating first would make a switched BYOK/provider identity impossible
    to distinguish or safely reconcile.
    """
    billing = config.get("component_billing")
    allowed = billing.get("profile_segments") if isinstance(billing, Mapping) else None
    if not isinstance(allowed, Mapping) or not allowed or len(allowed) > 32:
        raise MeteringContractError("component billing omitted bounded frozen profile segments")
    output: list[dict[str, Any]] = []
    previous_ended_boundary = 0
    for raw_segment in profile_usage_segments:
        # Check before inspecting/normalizing the 65th row. This bounds both
        # work and generator consumption without silently truncating billing
        # evidence from a buggy or hostile worker.
        if len(output) >= PROFILE_USAGE_SEGMENT_MAX:
            raise MeteringContractError("worker profile usage exceeds the bounded segment contract")
        if not isinstance(raw_segment, Mapping):
            raise MeteringContractError("worker profile usage segment is invalid")
        language = _text(raw_segment.get("language")).casefold()
        if not language or language not in allowed:
            raise MeteringContractError("worker profile usage segment is not an allowed frozen language")
        expected = allowed.get(language)
        if not isinstance(expected, Mapping):
            raise MeteringContractError("frozen profile segment contract is invalid")
        publication_version = raw_segment.get("publication_version")
        expected_version = expected.get("publication_version")
        if (
            isinstance(publication_version, bool)
            or isinstance(expected_version, bool)
            or publication_version != expected_version
        ):
            raise MeteringContractError("worker profile usage publication does not match frozen segment")
        started_boundary = raw_segment.get("started_boundary")
        ended_boundary = raw_segment.get("ended_boundary")
        if (
            isinstance(started_boundary, bool)
            or isinstance(ended_boundary, bool)
            or not isinstance(started_boundary, int)
            or not isinstance(ended_boundary, int)
            or started_boundary < 1
            or ended_boundary < started_boundary
            or started_boundary <= previous_ended_boundary
        ):
            raise MeteringContractError("worker profile usage boundary is invalid")
        previous_ended_boundary = ended_boundary
        components = expected.get("components")
        if not isinstance(components, list):
            raise MeteringContractError("frozen profile segment omitted component identities")
        segment_config = {
            "component_billing": {
                "usage_required": True,
                "components": components,
            }
        }
        usage = normalized_usage_from_model_usage(
            segment_config,
            raw_segment.get("model_usage") if isinstance(raw_segment.get("model_usage"), list) else (),
            # The controlled switch is never a customer app-tool call. Tool
            # units remain once per call below, not duplicated per segment.
            tool_units={},
            tts_native_usage=raw_segment.get("tts_native_usage"),
            soniox_usage=raw_segment.get("soniox_usage"),
            # One segment may legitimately meter nothing: the shared
            # recognizer bills to the opening segment, so a switch followed
            # by only caller speech -- or by a hangup -- leaves the new
            # segment with measured zeros. Refusing it stranded the whole
            # call's settlement at awaiting_usage after a successful switch.
            # The never-ran protection is re-imposed across ALL segments
            # below, where it is actually decidable.
            allow_unmetered_components=True,
        )
        # Tools are call-scoped and a language switch is not a customer tool;
        # carrying an empty tool map on every segment invites a future consumer
        # to sum it twice.  The one bounded call-level total is returned below.
        usage.pop("tool_units", None)
        # ``source`` is call-scoped for the same reason, and the settlement
        # contract REJECTS a segment whose metrics carry one -- deliberately,
        # because the validator re-stamps the adapter identity as
        # ``{"source": trusted, **segment_usage}`` and a supplied key would
        # win that splat. The single-stack normalizer above always emits it,
        # so leaving it nested here rejected the usage report of every
        # follow-the-caller call, switch or no switch, and stranded the
        # reservation at ``awaiting_usage``.
        usage.pop("source", None)
        output.append(
            {
                "language": language,
                "publication_version": publication_version,
                "started_boundary": started_boundary,
                "ended_boundary": ended_boundary,
                "usage": usage,
            }
        )
    if not output:
        raise MeteringContractError("worker profile usage omitted every frozen segment")
    # The single-stack never-ran protection, re-imposed at the only altitude
    # where a segmented call can decide it: a session that produced no native
    # component usage in ANY segment never carried media, and zero-filling it
    # would settle a call that should keep its reservation until the terminal
    # lifecycle releases the hold. Every normalized unit is a number, so "no
    # usage anywhere" is exactly "every value in every segment is zero".
    if not any(
        isinstance(value, (int, float)) and value
        for segment in output
        for value in segment["usage"].values()
    ):
        raise MeteringContractError("SDK usage omitted every pinned component metric")
    safe_tools: dict[str, int] = {}
    for name, count in (tool_units or {}).items():
        key = _tool_name(name)
        parsed = _number(count)
        if not parsed.is_integer() or int(parsed) > TOOL_USAGE_MAX_PER_NAME:
            raise MeteringContractError("tool usage exceeds the bounded per-tool limit")
        safe_tools[key] = int(parsed)
    if len(safe_tools) > TOOL_USAGE_MAX_NAMES or sum(safe_tools.values()) > TOOL_USAGE_MAX_TOTAL:
        raise MeteringContractError("tool usage exceeds the bounded billing contract")
    return {
        "source": "livekit_agent_sdk",
        "segments": output,
        "tool_units": safe_tools,
    }

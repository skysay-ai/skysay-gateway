# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Whole-call native counters from durable source ACKs, never SDK defaults."""
from __future__ import annotations

import math

try:
    from .capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS
    from .capabilities.workflow_sources import MAX_CALL_ATTEMPTS
    from .workflow_source_transport import source_fact
    from .workflow_speech import SpeechRefused
except ImportError:
    from capabilities.workflow_source_fields import SOURCE_USAGE_FIELDS
    from capabilities.workflow_sources import MAX_CALL_ATTEMPTS
    from workflow_source_transport import source_fact
    from workflow_speech import SpeechRefused


class WorkflowSourceMeter:
    def __init__(self, declarations, *, source_limit):
        if type(source_limit) is not int or not 1 <= source_limit <= MAX_CALL_ATTEMPTS:
            raise SpeechRefused("workflow meter requires its bounded source count")
        self._limit, self._sources, self._contracts = source_limit, {}, {}
        output_count, first_output = 0, None
        for declaration in declarations:
            identity = declaration.component, declaration.unit
            if declaration.component == "tts":
                output_count += 1
            # Bytes settle only for Fish Audio, the one provider whose hosted
            # usage contract requires ``tts_utf8_bytes``; the backend's source
            # unit contract refuses any other byte provider the same way.
            if (identity not in SOURCE_USAGE_FIELDS or output_count > 32
                    or (declaration.unit == "utf8_byte" and getattr(declaration, "provider", None) != "fishaudio")):
                raise SpeechRefused("workflow meter has no single declared owner per native component")
            if declaration.component == "tts" and first_output is not None:
                shared = ("role", "identity_kind", "metric_identity_kind", "max_metric_count", "provider_reference_format")
                if (getattr(declaration, "role", None) not in {"authored", "live"}
                        or any(getattr(declaration, key, None) != getattr(first_output, key, None) for key in shared)
                        or getattr(declaration, "source_count_limit", None) != source_limit
                        or getattr(first_output, "source_count_limit", None) != source_limit):
                    raise SpeechRefused("workflow meter has no single declared owner per native component")
            elif declaration.component == "tts":
                # The first output owns the role and the source count every
                # later one must share; the count is the meter's own limit.
                if (getattr(declaration, "role", None) not in {"authored", "live"}
                        or getattr(declaration, "source_count_limit", None) != source_limit):
                    raise SpeechRefused("workflow meter has no single declared owner per native component")
                first_output = declaration
            if declaration.component == "stt" and any(name == "stt" for name, _ in self._contracts):
                raise SpeechRefused("workflow meter has no single declared owner per native component")
            self._contracts[identity] = declaration
        if {name for name, _ in self._contracts} != {"stt", "tts"}:
            raise SpeechRefused("workflow meter requires its selected listener and output sources")

    def observe(self, fact):
        if type(fact) is not dict:
            raise SpeechRefused("workflow meter requires a durable source acknowledgement")
        fact = source_fact({"source": fact}, fact.get("source_attempt_id"))
        identity = fact["component"], fact["unit"]
        if identity not in self._contracts:
            raise SpeechRefused("workflow meter observed an unselected native component")
        source_id = fact["source_attempt_id"]
        prior = self._sources.get(source_id)
        if prior is not None and any(prior[key] != fact[key] for key in ("component", "unit", "binding_digest")):
            raise SpeechRefused("workflow meter source identity changed")
        if prior is None and len(self._sources) >= self._limit:
            raise SpeechRefused("workflow meter exceeded its source count")
        # Source responses contain no provider text, key or metric references.
        # Keep only exact bounded numeric facts needed at whole-call shutdown.
        self._sources[source_id] = {key: fact[key] for key in (
            "component", "unit", "binding_digest", "status", "absorbed")}
        self._sources[source_id]["total_units"] = (
            tuple(tuple(pair) for pair in fact["total_units"]) if fact["total_units"] is not None else None)

    def remaining_source_budget(self) -> int:
        return self._limit - len(self._sources)

    def report(self):
        # A source the registry settles by measurement seals ``exact`` with its
        # evidence; anything still unknown or absorbed leaves the verdict to Q2.
        if any(fact["absorbed"] or fact["status"] not in {"exact", "never_connected"} for fact in self._sources.values()):
            return None  # Q2 owns unresolved obligations; no guessed worker zero.
        values = {field: [] for identity in self._contracts for _, field in SOURCE_USAGE_FIELDS[identity]}
        for fact in self._sources.values():
            native = dict(fact["total_units"])
            for dimension, field in SOURCE_USAGE_FIELDS[fact["component"], fact["unit"]]:
                values[field].append(native[dimension])
        quantities = {field: math.fsum(parts) if field.endswith("audio_seconds") else sum(parts)
                      for field, parts in values.items()}
        # Every source is sealed, so all-zero is a measured zero:
        # the control plane releases such a call at its end.
        return {"source": "livekit_agent_sdk", **quantities, "tool_units": {}}

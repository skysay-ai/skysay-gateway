# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Native source dimensions and existing usage fields, shared across the wire."""
import math
from fractions import Fraction
from types import MappingProxyType

SOURCE_USAGE_FIELDS = MappingProxyType({
    ("stt", "audio_second"): (("audio_second", "stt_audio_seconds"),),
    ("stt", "token"): (("input_audio_tokens", "stt_input_audio_tokens"),
                       ("output_text_tokens", "stt_output_text_tokens")),
    ("tts", "audio_second"): (("audio_second", "tts_audio_seconds"),),
    ("tts", "character"): (("character", "tts_characters"),),
    ("tts", "utf8_byte"): (("utf8_byte", "tts_utf8_bytes"),),
    ("tts", "token"): (("input_text_tokens", "tts_input_text_tokens"),
                       ("output_audio_tokens", "tts_output_audio_tokens")),
})

#: The evidence kind a measured source seals.
MEASURED_EVIDENCE = "measured_partial"


def measured_source_units(rule, component, unit, seconds):
    """The tariff units of one measured duration under its registry rule.

    The worker and the control plane both call this, so a measured terminal
    has one conversion. Audio seconds settle unrounded; a token unit converts
    at the rule's per-minute usage reference, rounded down per dimension (the
    exact binary value of the measured float, so both sides agree to the
    token). Anything else has no measured settlement and refuses.
    """
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0
            or rule is None or rule["component"] != component):
        raise ValueError("measured source has no measured settlement")
    dimensions = tuple(dimension for dimension, _ in SOURCE_USAGE_FIELDS.get((component, unit), ()))
    per_minute = rule.get("token_units_per_minute")
    if unit == "audio_second" and per_minute is None:
        return {"audio_second": float(seconds)}
    if unit == "token" and per_minute is not None and set(per_minute) == set(dimensions):
        exact = Fraction(seconds)
        return {dimension: int(math.floor(exact * per_minute[dimension] / 60)) for dimension in dimensions}
    raise ValueError("measured source has no measured settlement")

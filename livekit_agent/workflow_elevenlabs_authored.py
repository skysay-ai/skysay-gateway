# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite ElevenLabs ``eleven_flash_v2_5`` authored source.

One HTTP ``/text-to-speech/{voice}/stream`` request through the metered
ElevenLabs adapter (``elevenlabs_tts``, the pinned plugin's ``ChunkedStream``)
at ``pcm_24000`` (raw PCM, no decoder). Its body's characters are handed to
the same native unit meter the Cascade path uses, and the line is charged by
its request's outcome under the capability's request billing: a completed or
worker-abandoned request its characters (ElevenLabs bills a response closed
early in full), a refused or failed one nothing.

The constructor's profile ``locale`` is the plugin's ``language``, sent as
``language_code`` exactly as the released Cascade path builds it.
"""
from __future__ import annotations

import re

try:
    from . import workflow_plugin_authored as plugin
    from .elevenlabs_tts import build_tts
except ImportError:
    import workflow_plugin_authored as plugin
    from elevenlabs_tts import build_tts


PROVIDER, MODEL = "elevenlabs", "eleven_flash_v2_5"
# Interpolated into the request URL path: alphanumeric only.
_VOICE = re.compile(r"[A-Za-z0-9]{1,64}")
AuthoredSourceRefused = plugin.AuthoredSourceRefused


def _build(secret, voice, locale, reference):
    session, meter = plugin.OwnedHttpSession(), plugin.native_unit_meter(PROVIDER, MODEL)
    tts = build_tts(native_unit_meter=meter, voice_id=voice, model=MODEL, encoding="pcm_24000", language=locale,
                    api_key=secret, http_session=session)
    if (tts.sample_rate != plugin.SAMPLE_RATE or tts._opts.encoding != "pcm_24000"
            or not isinstance(tts._opts.language, str) or not tts._opts.language):
        plugin.refuse()
    return plugin.PluginAuthoredSource(tts=tts, session=session, reference=reference,
        dimensions=("character",), sent=lambda: session.touched, settle=plugin.request_outcome_usage(meter))


def build_elevenlabs_authored_source(*, secret, constructor, client_reference_id):
    result = None
    try:
        voice, locale = plugin.require_inputs(constructor, PROVIDER, MODEL, _VOICE, secret, client_reference_id)
        result = _build(secret, voice, locale, client_reference_id)
    except Exception:
        result = None
    finally:
        del secret, constructor, client_reference_id
    if result is None:
        plugin.refuse()
    return result

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Fish Audio ``s2-pro`` authored source.

The repo's released ``fish_audio_tts.TTS`` (paid model, WAV framing, inline
WAV decoder) sends the text over the pinned plugin's live websocket, handing
the UTF-8 bytes of every ``event: text`` frame to the same native unit meter
the Cascade path uses; the line is charged by its request's outcome under the
capability's request billing. Fish takes no language parameter: the shared
constructor's ``locale`` is validated and not sent.
"""
from __future__ import annotations

import re

try:
    from . import workflow_plugin_authored as plugin
    from .fish_audio_tts import TTS
except ImportError:
    import workflow_plugin_authored as plugin
    from fish_audio_tts import TTS


PROVIDER, MODEL = "fishaudio", "s2-pro"
_VOICE = re.compile(r"[0-9a-f]{32}")
AuthoredSourceRefused = plugin.AuthoredSourceRefused


def _build(secret, voice, reference):
    session, meter = plugin.OwnedHttpSession(), plugin.native_unit_meter(PROVIDER, MODEL)
    tts = TTS(model=MODEL, voice_id=voice, output_format="wav", latency_mode="balanced",
              chunk_length=100, api_key=secret, native_unit_meter=meter, http_session=session)
    if tts.sample_rate != plugin.SAMPLE_RATE:
        plugin.refuse()
    # The socket opens through ``session.ws_connect``, which marks it touched.
    return plugin.PluginAuthoredSource(tts=tts, session=session, reference=reference,
        dimensions=("utf8_byte",), sent=lambda: session.touched, settle=plugin.request_outcome_usage(meter))


def build_fishaudio_authored_source(*, secret, constructor, client_reference_id):
    result = None
    try:
        voice, _locale = plugin.require_inputs(constructor, PROVIDER, MODEL, _VOICE, secret, client_reference_id)
        result = _build(secret, voice, client_reference_id)
    except Exception:
        result = None
    finally:
        del secret, constructor, client_reference_id
    if result is None:
        plugin.refuse()
    return result

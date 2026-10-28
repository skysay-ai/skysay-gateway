# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Azure ``neural`` authored source.

The released ``azure_speech.build_tts`` (SSML escaping, catalog identity and
Microsoft's billable-character rule) makes one REST request through the pinned
plugin's ``ChunkedStream`` at 24 kHz raw PCM, on the same native unit meter
the Cascade path uses: the escaped-SSML characters handed to the connection,
charged by the request's outcome under the capability's request billing.

Azure needs a region (the endpoint host) and an SSML ``xml:lang``. The region
is platform configuration copied beside the managed key at worker bootstrap
and handed to this factory because its declaration sets ``requires_region``;
it is never a constructor field. The ``xml:lang`` is the constructor's profile
``locale`` resolved exactly as the released Cascade path resolves it: the
capability artifact's locale table for ``azure_tts``/``neural``, then
``azure_speaking_locale`` for the region of that same language. A language
the table does not name is refused, never defaulted to ``en-US``.
"""
from __future__ import annotations

import re

try:
    from . import workflow_plugin_authored as plugin
    from . import azure_speech
    from .capabilities.registry import load_registry
except ImportError:
    import workflow_plugin_authored as plugin
    import azure_speech
    from capabilities.registry import load_registry


PROVIDER, MODEL, REGISTRY_NAME = "azure", "neural", "azure_tts"
_VOICE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z]{4})?-[A-Za-z]{2}-[A-Za-z0-9][A-Za-z0-9:-]{0,60}")
_LOCALE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z]{4})?-[A-Z]{2}")
AuthoredSourceRefused = plugin.AuthoredSourceRefused


def speaking_locale(voice, locale):
    """The SSML ``xml:lang`` for this voice and profile locale, or a refusal."""
    tag = load_registry().locale_for("tts", REGISTRY_NAME, MODEL, locale)
    if type(tag) is not str or _LOCALE.fullmatch(tag) is None:
        plugin.refuse()
    tag = azure_speech.azure_speaking_locale(voice, tag)
    if type(tag) is not str or _LOCALE.fullmatch(tag) is None:
        plugin.refuse()
    return tag


def _build(secret, voice, reference, region, locale):
    language = speaking_locale(voice, locale)
    session, meter = plugin.OwnedHttpSession(), plugin.native_unit_meter(PROVIDER, MODEL)
    tts = azure_speech.build_tts(voice=voice, language=language,
        sample_rate=plugin.SAMPLE_RATE, speech_key=secret, speech_region=azure_speech.azure_region(region),
        native_unit_meter=meter)
    # build_tts takes no session; the pinned plugin reads this one field.
    if tts.sample_rate != plugin.SAMPLE_RATE or getattr(tts, "_session", False) is not None:
        plugin.refuse()
    tts._session = session
    return plugin.PluginAuthoredSource(tts=tts, session=session, reference=reference,
        dimensions=("character",), sent=lambda: session.touched, settle=plugin.request_outcome_usage(meter))


def build_azure_authored_source(*, secret, constructor, client_reference_id, region):
    result = None
    try:
        voice, locale = plugin.require_inputs(constructor, PROVIDER, MODEL, _VOICE, secret, client_reference_id)
        result = _build(secret, voice, client_reference_id, region, locale)
    except Exception:
        result = None
    finally:
        del secret, constructor, client_reference_id, region
    if result is None:
        plugin.refuse()
    return result

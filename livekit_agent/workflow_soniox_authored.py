# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Inactive finite Soniox ``tts-rt-v2`` authored source.

The repo's released ``soniox_tts.TTS`` puts the opaque source attempt id in the
start frame's ``client_reference_id`` and marks the moment text is handed to
the websocket. Soniox's exact TTS tokens exist only in its usage log, which
this source does not read: a request whose text was sent is
``partial_unknown`` even when it completed, and one that sent no text is
``connect_unknown``. Reconciliation from the usage log belongs elsewhere.

The constructor's profile ``locale`` is Soniox's ``language``, exactly as the
released Cascade path builds it; the start frame never falls back to ``en``.
"""
from __future__ import annotations

import asyncio
import re

try:
    from . import workflow_plugin_authored as plugin
    from . import soniox_tts
    from .native_tts_usage import TTSRequestActivity
except ImportError:
    import workflow_plugin_authored as plugin
    import soniox_tts
    from native_tts_usage import TTSRequestActivity


PROVIDER, MODEL = "soniox", "tts-rt-v2"
_VOICE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
DIMENSIONS = ("input_text_tokens", "output_audio_tokens")
AuthoredSourceRefused = plugin.AuthoredSourceRefused


class _OwnedSonioxTTS(soniox_tts.TTS):
    """Close also joins the connection's own close task and the fatal-close tasks."""

    async def aclose(self):
        connection = self._TTS__current_connection  # pinned 1.8.8 name-mangled field
        await super().aclose()
        tasks = [task for task in (getattr(connection, "_close_task", None), *self._skysay_unowned_close_tasks)
                 if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _build(secret, voice, reference, language):
    session, activity = plugin.OwnedHttpSession(), TTSRequestActivity()
    tts = _OwnedSonioxTTS(model=MODEL, language=language, voice=voice, audio_format="pcm_s16le",
        sample_rate=plugin.SAMPLE_RATE, api_key=secret, client_reference_id=reference,
        on_request_started=activity.mark_started, http_session=session)
    return plugin.PluginAuthoredSource(tts=tts, session=session, reference=reference,
        dimensions=DIMENSIONS, sent=lambda: activity.started, exact=lambda _text: None)


def build_soniox_authored_source(*, secret, constructor, client_reference_id):
    result = None
    try:
        voice, locale = plugin.require_inputs(constructor, PROVIDER, MODEL, _VOICE, secret, client_reference_id)
        result = _build(secret, voice, client_reference_id, locale)
    except Exception:
        result = None
    finally:
        del secret, constructor, client_reference_id
    if result is None:
        plugin.refuse()
    return result

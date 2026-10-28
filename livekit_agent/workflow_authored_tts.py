# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Configured TTS input provenance, distinct from native acoustic verification.

A fresh public chunked stream is fully collected and closed before playback.
Only engines declaring text speech can authorize this source. No model caption
or synthetic acoustic approval stands in for native wording qualification.
"""
from __future__ import annotations

import time

try:
    from .workflow_speech import AuthoredSpeech, SpeechRefused
    from .workflow_source_owner import WorkflowSourceOwner
    from .workflow_speech_segments import source_text_fits
except ImportError:
    from workflow_speech import AuthoredSpeech, SpeechRefused
    from workflow_source_owner import WorkflowSourceOwner
    from workflow_speech_segments import source_text_fits


class AuthoredTtsPreparation:
    def __init__(self, *, declaration, constructor, journal, managed=None, resource=None, observe=None) -> None:
        bound = getattr(declaration, "max_input_utf8", None)
        if (declaration.role != "authored" or declaration.component != "tts"
                or type(bound) is not int or not 1 <= bound <= 64000):
            raise SpeechRefused("authored speech requires its declared TTS source")
        self.declaration, self.constructor, self.journal = declaration, constructor, journal
        self.managed = managed
        self.resource = resource
        # Latency evidence only: ``observe(stage, at)`` hears the line ledger's
        # output stages (claimed, granted, ask, first audio) on the monotonic clock.
        self._observe = observe
        self._used = False

    def _mark(self, stage):
        if self._observe is not None:
            try:
                self._observe(stage, time.monotonic())
            except Exception:  # noqa: BLE001 - a diagnostic cannot break a segment
                pass

    async def prepare(self, gate, binding, owner, *, operation) -> AuthoredSpeech:
        if self._used:
            raise SpeechRefused("authored TTS requires a fresh segment source")
        self._used = True
        self._mark("delivery_claimed")  # the owner hands over its granted claim
        text = gate.source_text(binding)
        if not source_text_fits(text, self.declaration):
            raise SpeechRefused("authored TTS text exceeds its declared source bound")
        await operation(self.journal.allow_authored_source(owner), timeout=5)
        speech = await operation(self._capture(gate, binding, owner, text))
        await operation(self.journal.approve_authored(owner, speech.candidate.pcm_digest), timeout=5)
        return speech

    async def _capture(self, gate, binding, owner, text):
        source = WorkflowSourceOwner(journal=self.journal.source_journal(), declaration=self.declaration,
                                     constructor=self.constructor, delivery=owner, managed=self.managed,
                                     resource=self.resource)
        stream = None
        final = False
        try:
            await source._open()
            self._mark("source_granted")
            from livekit.agents import APIConnectOptions
            stream = source.adapter.synthesize(text, conn_options=APIConnectOptions(max_retry=0, timeout=10))
            self._mark("ask_sent")
            heard = False
            async for event in stream:
                if final or type(event.is_final) is not bool:
                    raise SpeechRefused("authored TTS source crossed its complete segment boundary")
                gate.capture_frame(binding, event.frame)
                if not heard:
                    heard = True
                    self._mark("session_first_audio")
                final = event.is_final
            # __anext__ may end after a cancelled SDK task. Public completion,
            # its exact input, no producer error AND a final frame are required.
            if (not final or stream.done is not True or stream.exception is not None
                    or stream.input_text != text):
                raise SpeechRefused("authored TTS source did not complete its exact input")
            if await source.adapter.wait_terminal() is not True:
                raise SpeechRefused("authored TTS source has no finite terminal acknowledgement")
        finally:
            # The adapter owns this stream, its provider component and metrics.
            # Its single close joins all children before acknowledging usage.
            await source.aclose()
        return gate.seal_authored(binding)

    def completed(self, gate, speech: AuthoredSpeech) -> None:
        if not isinstance(speech, AuthoredSpeech):
            raise SpeechRefused("authored TTS completion has no owned source")
        gate.completed_authored(speech)

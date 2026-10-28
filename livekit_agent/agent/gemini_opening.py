# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Separate Gemini pickup captions from the first actual response generation.

The pinned plugin starts a generation even for an input-transcript-only event.
During callee-first pickup that creates a premature SpeechHandle; cancelling it
can discard the later audio while an explicit opening waits for another event.
Keep the plugin's input transcription/context machinery, but publish its first
response event only when output or a tool arrives. Restore ordinary SDK behavior
when the opening owner finishes, including a pending generation on handoff. No caller audio or transcript is buffered by this shim.
"""
import asyncio
import time


class GeminiPickupGeneration:
    def __init__(self, session):
        from livekit.plugins.google.realtime.realtime_api import RealtimeSession
        if not isinstance(session, RealtimeSession):
            raise TypeError("Gemini pickup shim requires the qualified realtime session")
        self.session=session
        self.pending=None
        self.closed=False
        self.start=session._start_new_generation
        self.content=session._handle_server_content
        self.tools=session._handle_tool_calls
        session._start_new_generation=self._start
        session._handle_server_content=self._content
        session._handle_tool_calls=self._tools
    def _start(self):
        s=self.session
        emit=s.emit
        future=s._pending_generation_fut
        # The pending explicit request must resolve at output, not at a
        # placeholder generation caused only by an incoming caption. The
        # pinned method is synchronous; nothing else can replace this future
        # while its two generation/start notifications are captured.
        s._pending_generation_fut=None
        self.pending=None
        def collect(name,*args):
            if name=='generation_created': self.pending=args[0]
            elif name!='input_speech_started': emit(name,*args)
        s.emit=collect
        try: self.start()
        finally:
            s.emit=emit
            s._pending_generation_fut=future
    def _publish(self):
        if self.pending is None: return
        s=self.session; event=self.pending; self.pending=None
        future=s._pending_generation_fut
        if future is not None and not future.done():
            event.user_initiated=True
            future.set_result(event)
            s._pending_generation_fut=None
        else:
            s._handle_input_speech_started()
        s.emit('generation_created',event)
    def _content(self,sc):
        parts=sc.model_turn.parts if sc.model_turn else []
        if (sc.output_transcription and sc.output_transcription.text) or any(
                not p.thought and (p.inline_data or p.text) for p in parts or []):
            self._publish()
        self.content(sc)
    def _tools(self,tools):
        self._publish()
        self.tools(tools)
    async def wait_for_response(self, *, ended, superseded, timeout=4.0):
        deadline=time.monotonic()+timeout
        while (self.pending is not None and self.session._current_generation is not None
               and not self.session._current_generation._done):
            if ended.is_set() or superseded(): return
            if time.monotonic()>=deadline:
                raise TimeoutError("Gemini pickup response did not settle")
            try:
                await asyncio.wait_for(ended.wait(), timeout=.02)
            except asyncio.TimeoutError:
                pass

    def close(self, *, publish_pending=False):
        if self.closed:return
        if publish_pending:
            self._publish()
        self.closed=True
        self.pending=None
        self.session._start_new_generation=self.start
        self.session._handle_server_content=self.content
        self.session._handle_tool_calls=self.tools

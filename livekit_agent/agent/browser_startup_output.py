# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Pinned SDK output seam for the browser startup witness."""
from typing import Any
from livekit.agents.voice import io


class BrowserStartupOutput(io.AudioOutput):
    def __init__(self, sink: io.AudioOutput, witness: Any) -> None:
        super().__init__(label="BrowserStartup", next_in_chain=sink,
                         capabilities=io.AudioOutputCapabilities(pause=sink.can_pause))
        self.witness = witness

    @property
    def sample_rate(self):
        return self.next_in_chain.sample_rate

    async def capture_frame(self, frame):
        if not self.witness.forward_frame():
            return
        await super().capture_frame(frame)
        await self.next_in_chain.capture_frame(frame)

    def flush(self):
        super().flush()
        self.next_in_chain.flush()

    def clear_buffer(self):
        self.next_in_chain.clear_buffer()

    def pause(self):
        self.next_in_chain.pause()

    def resume(self):
        self.next_in_chain.resume()

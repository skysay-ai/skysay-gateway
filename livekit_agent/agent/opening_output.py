# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Synchronous output evidence for deciding whether an opening was empty."""
from livekit.agents.voice import io
from livekit_evidence import _frame_rms, _PLAYOUT_TAIL_QUIET_RMS


class OpeningOutput(io.AudioOutput):
    """Count non-silent PCM accepted by the actual output sink.

    The independent room recorder can lag playout completion. Its clock is
    evidence of what arrived in the room, not a safe empty-response trigger.
    This wrapper forwards the sink's completion and interruption semantics.
    """

    def __init__(self, sink):
        super().__init__(label="opening-output", next_in_chain=sink,
                         sample_rate=sink.sample_rate,
                         capabilities=io.AudioOutputCapabilities(pause=sink.can_pause))
        self.audible_seconds = 0.0

    async def capture_frame(self, frame):
        await super().capture_frame(frame)
        await self.next_in_chain.capture_frame(frame)
        samples = frame.samples_per_channel * frame.num_channels
        if _frame_rms(frame.data, 0, samples) >= _PLAYOUT_TAIL_QUIET_RMS:
            self.audible_seconds += frame.samples_per_channel / frame.sample_rate

    def flush(self):
        super().flush()
        self.next_in_chain.flush()

    def clear_buffer(self):
        self.next_in_chain.clear_buffer()

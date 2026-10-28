# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Offline pinned-SDK session and detector contracts; no provider qualification."""
import unittest
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

from livekit import rtc
from livekit.agents import vad
from livekit.agents.version import __version__

try:
    from apps.voice.workflow_playback import create_workflow_playback_session
    from apps.voice import workflow_detector as detector_module
    from apps.voice.capabilities.workflow_listener import freeze_voice_input
except ImportError:
    from workflow_playback import create_workflow_playback_session
    import workflow_detector as detector_module
    from capabilities.workflow_listener import freeze_voice_input


class ModelLessPlaybackSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_default_inference_or_provider_is_constructed(self):
        self.assertEqual(__version__, "1.8.8")
        session = create_workflow_playback_session()
        try:
            for field in ("stt", "llm", "tts", "vad"):
                with self.subTest(field=field):
                    self.assertIsNone(getattr(session, field), f"the SDK added a default {field}")
        finally:
            await session.aclose()


    async def test_pinned_detector_bounds_event_storage_without_truncating_its_clock(self):
        policy = freeze_voice_input({"listener": {"registry_name": "azure", "model": "realtime",
            "languages": ["en"], "credential_source": "platform_managed"},
            "silence_hold_ms": 1500, "interruption_min_duration_ms": 150}, required_languages=["en"])
        with patch.object(detector_module, "SILERO_VAD_PINNED_PATH", Path("/app/models/silero_vad_pinned.onnx")):
            detector = detector_module.load_workflow_detector(policy)
        stream = detector.stream()
        decisions = 0
        def probability(_model, _frame):
            nonlocal decisions
            decisions += 1
            return .99 if decisions <= 800 else 0.0
        starts, ends, clocks = [], [], []
        async def consume():
            async for event in stream:
                if event.type == vad.VADEventType.START_OF_SPEECH:
                    starts.append(event)
                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    ends.append(event)
                elif event.type == vad.VADEventType.INFERENCE_DONE:
                    clocks.append(event.samples_index)
        try:
            with patch.object(type(stream._model), "__call__", probability):
                reader = asyncio.create_task(consume())
                for _ in range(1450):
                    stream.push_frame(rtc.AudioFrame(data=bytes(480 * 2), sample_rate=24000,
                        num_channels=1, samples_per_channel=480))
                stream.end_input()
                await asyncio.wait_for(reader, 10)
            assert len(starts) == len(ends) == 1, (len(starts), len(ends))
            assert clocks == sorted(clocks) and clocks[-1] > 27 * 16000
            retained = sum(frame.samples_per_channel for frame in ends[0].frames)
            assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in ends[0].frames)
            assert ends[0].samples_index - starts[0].samples_index > 24 * 16000
            assert retained <= int(20.5 * 24000), f"SDK kept {retained / 24000}s; capture accepts only its bounded event buffer"
            print(json.dumps({"passed": True, "detector": "real pinned Silero stream",
                "inference": "synthetic finite schedule", "endpoint_seconds": ends[0].samples_index / 16000,
                "retained_event_seconds": retained / 24000, "provider_or_acoustic_qualification": False}))
        finally:
            await stream.aclose()


class LosslessCallerIngressTest(unittest.IsolatedAsyncioTestCase):
    """The pinned SDK's own queue: a blocked loop loses no caller audio.

    A real local track feeds the real ``WorkflowRtcIngress``; the native SDK
    keeps delivering while the loop is blocked, as on a loaded host. The
    oracle is an extra FFI subscriber whose filter runs on the native
    callback thread after the stream's own callback was scheduled, so it
    sees what reached Python independently of the counter under test.
    """

    async def asyncSetUp(self):
        from livekit.rtc._ffi_client import FfiClient
        from livekit.rtc.version import __version__ as rtc_version
        # The frontier's two-turn drain rests on this SDK's AudioStream._run.
        self.assertEqual(rtc_version, "1.1.20")
        self.ffi = FfiClient.instance
        self.source = rtc.AudioSource(24000, 1, queue_size_ms=3000)
        self.track = rtc.LocalAudioTrack.create_audio_track("sky1349-ingress", self.source)

    async def ingress(self, **kwargs):
        try:
            from apps.voice.workflow_input_capture import WorkflowRtcIngress
        except ImportError:
            from workflow_input_capture import WorkflowRtcIngress
        ingress = WorkflowRtcIngress(self.track, sample_rate=24000, input_epoch=1, prefix_padding_samples=0, **kwargs)
        handle = ingress._stream._ffi_handle.handle
        arrivals = []
        def native(event):
            if (event.WhichOneof("message") == "audio_stream_event"
                    and event.audio_stream_event.stream_handle == handle
                    and event.audio_stream_event.HasField("frame_received")):
                arrivals.append(time.monotonic())
            return False
        oracle = self.ffi.queue.subscribe(asyncio.get_running_loop(), filter_fn=native)
        async def close():
            self.ffi.queue.unsubscribe(oracle)
            await ingress.aclose()
            await self.source.aclose()
        self.addAsyncCleanup(close)
        return ingress, arrivals

    async def capture(self, seconds):
        for _ in range(int(seconds * 50)):
            await self.source.capture_frame(rtc.AudioFrame(bytes(960), 24000, 1, 480))

    async def test_a_blocked_loop_keeps_every_frame_and_the_frontier_covers_what_arrived(self):
        ingress, arrivals = await self.ingress()
        consumed = []
        async def consume():
            async for frame in ingress:
                ingress.fence.receive_frame(samples=frame.samples_per_channel, sample_rate=24000)
                consumed.append(frame.samples_per_channel)
        consumer = asyncio.create_task(consume())
        self.addAsyncCleanup(asyncio.gather, consumer, return_exceptions=True)
        self.addCleanup(consumer.cancel)
        await self.capture(2.0)
        time.sleep(1.0)  # the loop is blocked; the native SDK keeps delivering
        presented = time.monotonic()
        arrived = 480 * sum(1 for at in arrivals if at < presented)
        stale = ingress._delivered + ingress.fence.queued_audio_samples
        consumed_formula = ingress.fence.received_samples + ingress.fence.queued_audio_samples
        frontier = await ingress.fence.frontier()
        self.assertGreater(arrived, 40 * 480, "the blocked loop let no backlog accumulate")
        self.assertGreaterEqual(frontier, arrived, "audio that reached the worker before P indexes past the boundary")
        # Negative controls: the counter read without the drain, and the
        # consumed-plus-slack formula, both put pre-P audio after the boundary.
        self.assertLess(stale, arrived)
        self.assertLess(consumed_formula, arrived)
        for _ in range(300):
            await asyncio.sleep(0.02)
            if len(arrivals) == len(consumed) and len(consumed) >= 100 and ingress._delivered == sum(consumed):
                break
        self.assertEqual(len(consumed), len(arrivals), "the ingress discarded caller audio")
        self.assertEqual(sum(consumed), ingress._delivered)
        print(json.dumps({"passed": True, "sdk": "livekit-rtc 1.1.20 local track", "blocked_seconds": 1.0,
            "frames_arrived_before_presentation": arrived // 480, "frontier_frames": frontier // 480,
            "stale_counter_frames": stale // 480, "consumed_formula_frames": consumed_formula // 480,
            "frames_consumed": len(consumed), "frames_arrived": len(arrivals)}))

    async def test_a_backlog_past_its_cap_latches_once_and_the_sdk_queue_stays_bounded(self):
        try:
            from apps.voice import workflow_input_capture as capture_module
        except ImportError:
            import workflow_input_capture as capture_module
        overflows = []
        with patch.object(capture_module, "MAX_CALLER_INGRESS_BACKLOG_SECONDS", 1):
            ingress, arrivals = await self.ingress(on_overflow=overflows.append)
        # Nothing consumes: the backlog passes the 1 s cap while the loop runs.
        await self.capture(2.0)
        for _ in range(300):
            await asyncio.sleep(0.02)
            if len(arrivals) >= 75:
                break
        queued = len(ingress._stream._queue._queue)
        self.assertGreaterEqual(len(arrivals), 75)
        self.assertEqual(len(overflows), 1)
        self.assertIs(ingress.overflow.result(), overflows[0])
        self.assertEqual(overflows[0].backlog_seconds, 1.02)
        self.assertLessEqual(queued, 50, "the SDK queue retained more than its cap")
        print(json.dumps({"passed": True, "cap_seconds": 1, "frames_arrived": len(arrivals),
            "queued_frames": queued, "latched_backlog_seconds": overflows[0].backlog_seconds}))


if __name__ == "__main__":
    unittest.main()

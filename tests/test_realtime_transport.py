from __future__ import annotations

import unittest

from openphonex_gateway.realtime import AudioFrame, CallMetrics, CancellationToken
from openphonex_gateway.realtime.transport import PacedRtpSender, RtpReceiveStats, RtpSendState


class RealtimeTransportTests(unittest.TestCase):
    def test_rtp_sender_closes_upstream_stream_when_cancelled_between_chunks(self) -> None:
        packets: list[bytes] = []
        cancellation = CancellationToken()
        state = {"closed": False, "requested_chunks": 0}

        def chunks():
            try:
                state["requested_chunks"] += 1
                yield AudioFrame.tone(20).pcm16
                state["requested_chunks"] += 1
                yield AudioFrame.tone(20, hz=660).pcm16
            finally:
                state["closed"] = True

        sender = PacedRtpSender(
            packets.append,
            clock_sleep=lambda _seconds: cancellation.cancel(),
        )
        metrics = CallMetrics()

        sent = sender.send_pcm16_stream(chunks(), cancellation=cancellation, metrics=metrics)

        self.assertEqual(sent, 1)
        self.assertEqual(len(packets), 1)
        self.assertEqual(state["requested_chunks"], 1)
        self.assertTrue(state["closed"])
        cancelled = metrics.first("rtp_stream_cancelled")
        self.assertIsNotNone(cancelled)
        assert cancelled is not None
        self.assertEqual(cancelled.attrs["packets"], 1)

    def test_rtp_sender_closes_upstream_stream_when_cancelled_during_buffer_flush(self) -> None:
        packets: list[bytes] = []
        cancellation = CancellationToken()
        state = {"closed": False}
        two_frames = AudioFrame.tone(20).pcm16 + AudioFrame.tone(20, hz=660).pcm16

        def chunks():
            try:
                yield two_frames
            finally:
                state["closed"] = True

        sender = PacedRtpSender(
            packets.append,
            clock_sleep=lambda _seconds: cancellation.cancel(),
        )

        sent = sender.send_pcm16_stream(chunks(), cancellation=cancellation)

        self.assertEqual(sent, 1)
        self.assertEqual(len(packets), 1)
        self.assertTrue(state["closed"])

    def test_rtp_sender_prebuffers_before_first_packet(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        state = {"requested_chunks": 0, "requested_at_first_send": 0}

        def chunks():
            for hz in (440, 660, 880):
                state["requested_chunks"] += 1
                yield AudioFrame.tone(20, hz=hz).pcm16

        def send_packet(packet: bytes) -> None:
            if not packets:
                state["requested_at_first_send"] = state["requested_chunks"]
            packets.append(packet)

        sender = PacedRtpSender(
            send_packet,
            prebuffer_ms=40,
            clock_sleep=lambda _seconds: None,
        )

        sent = sender.send_pcm16_stream(chunks(), metrics=metrics)

        self.assertEqual(sent, 3)
        self.assertEqual(len(packets), 3)
        self.assertGreaterEqual(state["requested_at_first_send"], 2)
        prebuffer = metrics.first("rtp_prebuffer_ready")
        self.assertIsNotNone(prebuffer)
        assert prebuffer is not None
        self.assertEqual(prebuffer.attrs["target_ms"], 40)
        self.assertGreaterEqual(prebuffer.attrs["actual_ms"], 40)

    def test_rtp_receive_stats_track_loss_jitter_duplicates_and_reordering(self) -> None:
        packets: list[bytes] = []
        sender = PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=10, timestamp=1000, ssrc=0xAABBCCDD),
            clock_sleep=lambda _seconds: None,
        )
        sender.send_pcm16_stream(
            [
                AudioFrame.tone(20).pcm16,
                AudioFrame.tone(20).pcm16,
                AudioFrame.tone(20).pcm16,
            ]
        )
        self.assertEqual(len(packets), 3)

        stats = RtpReceiveStats()
        stats.observe_packet(packets[0], peer=("10.0.0.5", 12000), arrival_time=1.00)
        stats.observe_packet(packets[2], peer=("10.0.0.5", 12000), arrival_time=1.05)
        stats.observe_packet(packets[2], peer=("10.0.0.5", 12000), arrival_time=1.06)
        stats.observe_packet(packets[1], peer=("10.0.0.5", 12000), arrival_time=1.07)

        payload = stats.as_dict()
        self.assertEqual(payload["packets_received"], 4)
        self.assertEqual(payload["packets_lost"], 1)
        self.assertEqual(payload["duplicate_packets"], 1)
        self.assertEqual(payload["out_of_order_packets"], 1)
        self.assertEqual(payload["first_sequence"], 10)
        self.assertEqual(payload["last_sequence"], 12)
        self.assertEqual(payload["first_timestamp"], 1000)
        self.assertEqual(payload["last_timestamp"], 1320)
        self.assertEqual(payload["ssrc"], 0xAABBCCDD)
        self.assertEqual(payload["payload_bytes"], 640)
        self.assertEqual(payload["peer"], {"host": "10.0.0.5", "port": 12000})
        self.assertGreater(payload["jitter_ms"], 0)
        self.assertGreaterEqual(payload["max_jitter_ms"], payload["jitter_ms"])

    def test_rtp_receive_stats_count_malformed_packets_without_advancing_sequence(self) -> None:
        packets: list[bytes] = []
        sender = PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=21, timestamp=4000, ssrc=7),
            clock_sleep=lambda _seconds: None,
        )
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])

        stats = RtpReceiveStats()
        stats.observe_packet(packets[0], peer=("127.0.0.1", 10000), arrival_time=3.0)
        stats.observe_packet(b"short", peer=("127.0.0.1", 10000), arrival_time=3.1)
        stats.observe_packet(b"\x40" + b"\x00" * 11, peer=("127.0.0.1", 10000), arrival_time=3.2)

        payload = stats.as_dict()
        self.assertEqual(payload["packets_received"], 1)
        self.assertEqual(payload["decode_errors"], 2)
        self.assertEqual(payload["last_sequence"], 21)
        self.assertEqual(payload["last_timestamp"], 4000)


if __name__ == "__main__":
    unittest.main()

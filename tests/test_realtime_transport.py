from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest.mock import patch

from openphonex_gateway.rtp import decode_rtp_ulaw_payload
from openphonex_gateway.realtime import AudioFrame, CallMetrics, CancellationToken
from openphonex_gateway.realtime.transport import PacedRtpSender, RtpReceiveStats, RtpSendState, UdpRtpTransport


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RealtimeTransportTests(unittest.TestCase):
    def test_rtp_receiver_start_failure_releases_bound_udp_port(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        with patch(
            "openphonex_gateway.realtime.transport.threading.Thread.start",
            side_effect=RuntimeError("synthetic thread start failure"),
        ), self.assertRaisesRegex(RuntimeError, "synthetic thread start failure"):
            UdpRtpTransport(("127.0.0.1", port))

        rebound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            rebound.bind(("127.0.0.1", port))
        finally:
            rebound.close()

    def test_rtp_receiver_keeps_network_timing_truthful_during_700ms_compute_pause(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=100, timestamp=8000, ssrc=0xAABBCCDD),
            clock_sleep=lambda _seconds: None,
        ).send_pcm16_stream([AudioFrame.tone(20, hz=440 + index).pcm16 for index in range(36)])
        # Receive and consume one packet, then stop consuming for the next 35
        # 20 ms arrivals: exactly 700 ms of logical compute pause. The ingress
        # thread must keep receipt timestamps truthful while compute is idle.
        clock = _ManualClock(start=1.0)
        transport = UdpRtpTransport(
            ("127.0.0.1", 0),
            receive_queue_frames=40,
            clock=clock,
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            destination = transport.sock.getsockname()
            sender.sendto(packets[0], destination)
            self.assertTrue(_wait_for_received_packets(transport, 1))
            first_frame = transport.recv_frame()
            for index, packet in enumerate(packets[1:], start=2):
                clock.advance(0.02)
                sender.sendto(packet, destination)
                self.assertTrue(_wait_for_received_packets(transport, index))

            frames = [first_frame, *(transport.recv_frame() for _packet in packets[1:])]
            stats = transport.stats.as_dict()

            self.assertTrue(all(frame is not None for frame in frames))
            self.assertEqual(stats["packets_received"], 36)
            self.assertEqual(stats["packets_lost"], 0)
            self.assertEqual(stats["max_jitter_ms"], 0.0)
            self.assertEqual(stats["receive_queue"]["overflows"], 0)
            self.assertEqual(stats["receive_queue"]["frames_dequeued"], 36)
            self.assertEqual(stats["receive_queue"]["depth_frames"], 0)
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_delivers_frames_in_network_order(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=20, timestamp=4000, ssrc=7),
            clock_sleep=lambda _seconds: None,
        ).send_pcm16_stream([AudioFrame.tone(20, hz=hz).pcm16 for hz in (330, 440, 550)])
        transport = UdpRtpTransport(
            ("127.0.0.1", 0),
            receive_queue_frames=3,
            clock=_SteppedClock(start=2.0, step=0.02),
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for packet in packets:
                sender.sendto(packet, transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, len(packets)))

            received = [transport.recv_frame(), transport.recv_frame(), transport.recv_frame()]

            self.assertEqual(
                [frame.pcm16 if frame is not None else None for frame in received],
                [decode_rtp_ulaw_payload(packet) for packet in packets],
            )
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_drops_oldest_frame_on_overflow_without_reporting_network_loss(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=40, timestamp=6000, ssrc=9),
            clock_sleep=lambda _seconds: None,
        ).send_pcm16_stream([AudioFrame.tone(20, hz=hz).pcm16 for hz in (300, 400, 500, 600)])
        transport = UdpRtpTransport(
            ("127.0.0.1", 0),
            receive_queue_frames=2,
            clock=_SteppedClock(start=3.0, step=0.02),
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for packet in packets:
                sender.sendto(packet, transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, len(packets)))

            received = [transport.recv_frame(), transport.recv_frame()]
            stats = transport.stats.as_dict()

            self.assertEqual(
                [frame.pcm16 if frame is not None else None for frame in received],
                [decode_rtp_ulaw_payload(packet) for packet in packets[-2:]],
            )
            self.assertEqual(stats["packets_received"], 4)
            self.assertEqual(stats["packets_lost"], 0)
            self.assertEqual(stats["max_jitter_ms"], 0.0)
            queue = stats["receive_queue"]
            self.assertEqual(queue["capacity_frames"], 2)
            self.assertEqual(queue["overflow_policy"], "drop_oldest")
            self.assertEqual(queue["frames_enqueued"], 4)
            self.assertEqual(queue["frames_dequeued"], 2)
            self.assertEqual(queue["frames_dropped"], 2)
            self.assertEqual(queue["frames_discarded_on_stop"], 0)
            self.assertEqual(queue["overflows"], 2)
            self.assertEqual(queue["max_depth_frames"], 2)
            self.assertEqual(queue["depth_frames"], 0)
            self.assertTrue(queue["receiver_running"])
            self.assertEqual(queue["receiver_errors"], 0)
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_rejects_unexpected_source_without_redirecting_send_peer(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16]
        )
        expected_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unexpected_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        expected_sender.bind(("127.0.0.1", 0))
        unexpected_sender.bind(("127.0.0.1", 0))
        transport = UdpRtpTransport(
            ("127.0.0.1", 0),
            peer=expected_sender.getsockname(),
            receive_queue_frames=2,
        )
        try:
            unexpected_sender.sendto(packets[0], transport.sock.getsockname())
            expected_sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, 1))

            self.assertIsNotNone(transport.recv_frame())
            stats = transport.stats.as_dict()
            self.assertEqual(transport.peer, expected_sender.getsockname())
            self.assertEqual(stats["packets_received"], 1)
            self.assertEqual(stats["unexpected_peer_packets"], 1)
        finally:
            expected_sender.close()
            unexpected_sender.close()
            transport.close()

    def test_rtp_receiver_learns_only_the_first_valid_source(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16]
        )
        malformed_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        valid_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        malformed_sender.bind(("127.0.0.1", 0))
        valid_sender.bind(("127.0.0.1", 0))
        transport = UdpRtpTransport(("127.0.0.1", 0))
        try:
            malformed_sender.sendto(b"short", transport.sock.getsockname())
            valid_sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, 1))

            self.assertIsNotNone(transport.recv_frame())
            stats = transport.stats.as_dict()
            self.assertEqual(transport.peer, valid_sender.getsockname())
            self.assertEqual(stats["decode_errors"], 1)
            self.assertEqual(stats["unexpected_peer_packets"], 0)
        finally:
            malformed_sender.close()
            valid_sender.close()
            transport.close()

    def test_rtp_receiver_does_not_learn_peer_from_header_only_datagram(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16]
        )
        header_only_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        valid_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        header_only_sender.bind(("127.0.0.1", 0))
        valid_sender.bind(("127.0.0.1", 0))
        transport = UdpRtpTransport(("127.0.0.1", 0))
        try:
            header_only_sender.sendto(packets[0][:12], transport.sock.getsockname())
            self.assertTrue(_wait_for_decode_errors(transport, 1))
            valid_sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, 1))

            self.assertIsNotNone(transport.recv_frame())
            self.assertEqual(transport.peer, valid_sender.getsockname())
            self.assertEqual(transport.stats.as_dict()["decode_errors"], 1)
        finally:
            header_only_sender.close()
            valid_sender.close()
            transport.close()

    def test_rtp_receiver_can_require_ari_authoritative_peer_before_accepting_media(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16]
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        transport = UdpRtpTransport(("127.0.0.1", 0), learn_peer_from_source=False)
        try:
            sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_unexpected_peer_packets(transport, 1))
            self.assertIsNone(transport.peer)
            self.assertEqual(transport.stats.as_dict()["packets_received"], 0)
            self.assertIsNone(transport.recv_frame())

            transport.peer = sender.getsockname()
            sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, 1))
            self.assertIsNotNone(transport.recv_frame())
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_matches_configured_hostname_to_source_ip(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16]
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        transport = UdpRtpTransport(
            ("127.0.0.1", 0),
            peer=("localhost", sender.getsockname()[1]),
        )
        try:
            sender.sendto(packets[0], transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, 1))

            self.assertIsNotNone(transport.recv_frame())
            self.assertEqual(transport.peer, ("localhost", sender.getsockname()[1]))
            self.assertEqual(transport.stats.as_dict()["unexpected_peer_packets"], 0)
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_close_stops_ingress_and_wakes_waiters(self) -> None:
        transport = UdpRtpTransport(("127.0.0.1", 0), recv_timeout_seconds=0.5)
        self.assertTrue(_wait_for_receiver_state(transport, running=True))
        received: list[AudioFrame | None] = []
        waiter = threading.Thread(target=lambda: received.append(transport.recv_frame()))
        waiter.start()
        time.sleep(0.01)

        started = time.monotonic()
        transport.close()
        waiter.join(timeout=0.1)

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(received, [None])
        self.assertIsNone(transport.recv_frame())
        self.assertFalse(transport.stats.as_dict()["receive_queue"]["receiver_running"])
        transport.close()

    def test_rtp_receiver_stop_freezes_final_queue_metrics_before_socket_close(self) -> None:
        packets: list[bytes] = []
        PacedRtpSender(packets.append, clock_sleep=lambda _seconds: None).send_pcm16_stream(
            [AudioFrame.tone(20).pcm16, AudioFrame.tone(20, hz=660).pcm16]
        )
        transport = UdpRtpTransport(("127.0.0.1", 0))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for packet in packets:
                sender.sendto(packet, transport.sock.getsockname())
            self.assertTrue(_wait_for_received_packets(transport, len(packets)))

            transport.stop_receiving()
            stats = transport.stats.as_dict()

            self.assertFalse(stats["receive_queue"]["receiver_running"])
            self.assertEqual(stats["receive_queue"]["frames_discarded_on_stop"], 2)
            self.assertEqual(stats["receive_queue"]["depth_frames"], 0)
            self.assertIsNone(transport.recv_frame())
            self.assertGreaterEqual(transport.sock.fileno(), 0)
        finally:
            sender.close()
            transport.close()

    def test_rtp_receiver_rejects_an_unbounded_zero_capacity_queue(self) -> None:
        with self.assertRaisesRegex(ValueError, "receive_queue_frames"):
            UdpRtpTransport(("127.0.0.1", 0), receive_queue_frames=0)

    def test_rtp_sender_marks_initial_packet_without_changing_timestamp(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        metrics = CallMetrics(clock=clock.now)
        sender = PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=10, timestamp=1000, ssrc=0xAABBCCDD),
            clock=clock.now,
            clock_sleep=clock.sleep,
        )

        sender.send_pcm16_stream(
            [AudioFrame.tone(20).pcm16, AudioFrame.tone(20).pcm16],
            metrics=metrics,
        )

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, False])
        self.assertEqual([packet[1] & 0x7F for packet in packets], [0, 0])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [1000, 1160])
        talkspurt = metrics.first("rtp_talkspurt_started")
        self.assertIsNotNone(talkspurt)
        assert talkspurt is not None
        self.assertEqual(
            talkspurt.attrs,
            {
                "reason": "initial",
                "marker": True,
                "inter_packet_gap_ms": 0.0,
                "timestamp_advance_samples": 0,
                "stream_packet_index": 0,
            },
        )

    def test_rtp_sender_marks_and_timestamps_stream_resumed_after_9640ms_silence(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        metrics = CallMetrics(clock=clock.now)
        sender = PacedRtpSender(
            packets.append,
            state=RtpSendState(sequence=10, timestamp=1000, ssrc=0xAABBCCDD),
            clock=clock.now,
            clock_sleep=clock.sleep,
        )

        sender.send_pcm16_stream(
            [AudioFrame.tone(20).pcm16, AudioFrame.tone(20).pcm16],
            metrics=metrics,
        )
        # The previous live failure resumed after a roughly 9.6s packet gap.
        # send_pcm16_stream already paced 20ms after the second packet, so this
        # adds 9.62s to make the exact last-packet-to-next-packet gap 9.64s.
        clock.advance(9.620)
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16], metrics=metrics)

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, False, True])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [1000, 1160, 78280])
        talkspurts = [event for event in metrics.events if event.name == "rtp_talkspurt_started"]
        self.assertEqual(len(talkspurts), 2)
        self.assertEqual(talkspurts[1].attrs["reason"], "gap")
        self.assertEqual(talkspurts[1].attrs["inter_packet_gap_ms"], 9640.0)
        self.assertEqual(talkspurts[1].attrs["timestamp_advance_samples"], 76960)
        self.assertEqual(talkspurts[1].attrs["stream_packet_index"], 0)

    def test_rtp_sender_keeps_back_to_back_streams_contiguous(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        sender = PacedRtpSender(packets.append, clock=clock.now, clock_sleep=clock.sleep)

        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, False])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [0, 160])

    def test_rtp_sender_does_not_mark_normal_scheduler_jitter(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        sender = PacedRtpSender(packets.append, clock=clock.now, clock_sleep=clock.sleep)

        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])
        clock.advance(0.019)
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, False])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [0, 160])

    def test_rtp_sender_marks_one_full_suppressed_frame(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        sender = PacedRtpSender(packets.append, clock=clock.now, clock_sleep=clock.sleep)

        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])
        clock.advance(0.020)
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16])

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, True])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [0, 320])

    def test_rtp_sender_marks_and_timestamps_midstream_source_stall(self) -> None:
        packets: list[bytes] = []
        clock = _FakeClock()
        metrics = CallMetrics(clock=clock.now)
        sender = PacedRtpSender(packets.append, clock=clock.now, clock_sleep=clock.sleep)

        def stalled_chunks():
            yield AudioFrame.tone(20).pcm16
            clock.advance(0.700)
            yield AudioFrame.tone(20, hz=660).pcm16

        sender.send_pcm16_stream(stalled_chunks(), metrics=metrics)

        self.assertEqual([bool(packet[1] & 0x80) for packet in packets], [True, True])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in packets], [0, 5760])
        talkspurts = [event for event in metrics.events if event.name == "rtp_talkspurt_started"]
        self.assertEqual(len(talkspurts), 2)
        self.assertEqual(talkspurts[1].attrs["reason"], "gap")
        self.assertEqual(talkspurts[1].attrs["inter_packet_gap_ms"], 720.0)
        self.assertEqual(talkspurts[1].attrs["timestamp_advance_samples"], 5600)
        self.assertEqual(talkspurts[1].attrs["stream_packet_index"], 1)

    def test_rtp_sender_failed_send_does_not_advance_wire_or_timing_state(self) -> None:
        attempted_packets: list[bytes] = []
        clock = _FakeClock()
        metrics = CallMetrics(clock=clock.now)
        state = RtpSendState(sequence=10, timestamp=1000, ssrc=0xAABBCCDD)

        def fail_once(packet: bytes) -> None:
            attempted_packets.append(packet)
            if len(attempted_packets) == 1:
                raise OSError("synthetic send failure")

        sender = PacedRtpSender(
            fail_once,
            state=state,
            clock=clock.now,
            clock_sleep=clock.sleep,
        )

        with self.assertRaises(OSError):
            sender.send_pcm16_stream([AudioFrame.tone(20).pcm16], metrics=metrics)

        self.assertEqual((state.sequence, state.timestamp), (10, 1000))
        self.assertIsNone(metrics.first("rtp_talkspurt_started"))
        clock.advance(1.0)
        sender.send_pcm16_stream([AudioFrame.tone(20).pcm16], metrics=metrics)

        self.assertEqual([bool(packet[1] & 0x80) for packet in attempted_packets], [True, True])
        self.assertEqual([int.from_bytes(packet[4:8], "big") for packet in attempted_packets], [1000, 1000])
        self.assertEqual((state.sequence, state.timestamp), (11, 1160))
        self.assertEqual(
            len([event for event in metrics.events if event.name == "rtp_talkspurt_started"]),
            1,
        )

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
        self.assertFalse(prebuffer.attrs["source_exhausted"])
        self.assertIsNotNone(metrics.first("first_rtp_packet_sent"))

    def test_rtp_sender_prebuffer_does_not_stall_streams_shorter_than_cushion(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()

        sender = PacedRtpSender(
            packets.append,
            prebuffer_ms=200,
            clock_sleep=lambda _seconds: None,
        )

        sent = sender.send_pcm16_stream(
            (AudioFrame.tone(20, hz=hz).pcm16 for hz in (440, 660, 880)),
            metrics=metrics,
        )

        self.assertEqual(sent, 3)
        self.assertEqual(len(packets), 3)
        prebuffer = metrics.first("rtp_prebuffer_ready")
        self.assertIsNotNone(prebuffer)
        assert prebuffer is not None
        self.assertEqual(prebuffer.attrs["target_ms"], 200)
        self.assertEqual(prebuffer.attrs["actual_ms"], 60)
        self.assertTrue(prebuffer.attrs["source_exhausted"])

    def test_rtp_sender_prebuffer_honors_cancellation_before_first_packet(self) -> None:
        packets: list[bytes] = []
        metrics = CallMetrics()
        cancellation = CancellationToken()
        state = {"closed": False}

        def chunks():
            try:
                yield AudioFrame.tone(20).pcm16
                cancellation.cancel()
                yield AudioFrame.tone(20, hz=660).pcm16
            finally:
                state["closed"] = True

        sender = PacedRtpSender(
            packets.append,
            prebuffer_ms=80,
            clock_sleep=lambda _seconds: None,
        )

        sent = sender.send_pcm16_stream(chunks(), cancellation=cancellation, metrics=metrics)

        self.assertEqual(sent, 0)
        self.assertEqual(packets, [])
        self.assertTrue(state["closed"])
        cancelled = metrics.first("rtp_stream_cancelled")
        self.assertIsNotNone(cancelled)
        assert cancelled is not None
        self.assertEqual(cancelled.attrs["packets"], 0)

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


class _SteppedClock:
    def __init__(self, *, start: float, step: float) -> None:
        self.current = start
        self.step = step

    def __call__(self) -> float:
        current = self.current
        self.current += self.step
        return current


class _ManualClock:
    def __init__(self, *, start: float) -> None:
        self.current = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.current

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.current += seconds


def _wait_for_received_packets(transport: UdpRtpTransport, expected: int, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if transport.stats.as_dict()["packets_received"] >= expected:
            return True
        time.sleep(0.001)
    return False


def _wait_for_receiver_state(transport: UdpRtpTransport, *, running: bool, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receive_queue = transport.stats.as_dict().get("receive_queue", {})
        if receive_queue.get("receiver_running") is running:
            return True
        time.sleep(0.001)
    return False


def _wait_for_decode_errors(transport: UdpRtpTransport, expected: int, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if transport.stats.as_dict()["decode_errors"] >= expected:
            return True
        time.sleep(0.001)
    return False


def _wait_for_unexpected_peer_packets(transport: UdpRtpTransport, expected: int, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if transport.stats.as_dict()["unexpected_peer_packets"] >= expected:
            return True
        time.sleep(0.001)
    return False


if __name__ == "__main__":
    unittest.main()

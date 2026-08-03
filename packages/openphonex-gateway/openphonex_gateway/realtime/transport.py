# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Callable, Iterable

try:  # monolith: media modules resolve under the telephony_ai.media shim name
    from telephony_ai.media.rtp import decode_rtp_ulaw_payload, linear_to_mulaw
except ModuleNotFoundError:  # standalone openphonex-gateway (OSS, no control plane)
    from ..rtp import decode_rtp_ulaw_payload, linear_to_mulaw

from .frames import AudioFrame
from .metrics import CallMetrics


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class RtpSendState:
    sequence: int = 1
    timestamp: int = 0
    ssrc: int = 0x12345678


@dataclass
class RtpReceiveStats:
    packets_received: int = 0
    packets_lost: int = 0
    out_of_order_packets: int = 0
    duplicate_packets: int = 0
    payload_bytes: int = 0
    decode_errors: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    ssrc: int | None = None
    jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0
    peer_host: str = ""
    peer_port: int | None = None
    receive_queue_capacity_frames: int = 0
    receive_queue_frames_enqueued: int = 0
    receive_queue_frames_dequeued: int = 0
    receive_queue_frames_dropped: int = 0
    receive_queue_frames_discarded_on_stop: int = 0
    receive_queue_overflows: int = 0
    receive_queue_max_depth_frames: int = 0
    receive_queue_depth_frames: int = 0
    receiver_running: bool = False
    receiver_errors: int = 0
    unexpected_peer_packets: int = 0
    _last_arrival_ms: float | None = None
    _last_rtp_timestamp: int | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def observe_packet(self, packet: bytes, *, peer: tuple[str, int], arrival_time: float, sample_rate: int = 8000) -> None:
        with self._lock:
            self.peer_host, self.peer_port = peer
            if len(packet) < 12:
                self.decode_errors += 1
                return
            try:
                sequence, timestamp, ssrc = _rtp_header_fields(packet)
            except ValueError:
                self.decode_errors += 1
                return
            payload_len = max(0, len(packet) - 12)
            if self.packets_received == 0:
                self.first_sequence = sequence
                self.first_timestamp = timestamp
                self.ssrc = ssrc
            advance_sequence = True
            if self.last_sequence is not None:
                gap = _rtp_sequence_delta(sequence, self.last_sequence)
                if gap == 0:
                    self.duplicate_packets += 1
                    advance_sequence = False
                elif gap > 1:
                    self.packets_lost += gap - 1
                elif gap < 0:
                    self.out_of_order_packets += 1
                    advance_sequence = False
            self.packets_received += 1
            self.payload_bytes += payload_len
            if advance_sequence:
                self.last_sequence = sequence
                self.last_timestamp = timestamp
                self._observe_jitter(timestamp, arrival_time, sample_rate=sample_rate)

    def _observe_jitter(self, timestamp: int, arrival_time: float, *, sample_rate: int) -> None:
        arrival_ms = arrival_time * 1000.0
        if self._last_arrival_ms is None or self._last_rtp_timestamp is None:
            self._last_arrival_ms = arrival_ms
            self._last_rtp_timestamp = timestamp
            return
        arrival_delta_ms = arrival_ms - self._last_arrival_ms
        rtp_delta_ms = max(0, _rtp_timestamp_delta(timestamp, self._last_rtp_timestamp)) / max(1, sample_rate) * 1000.0
        transit_delta = abs(arrival_delta_ms - rtp_delta_ms)
        self.jitter_ms += (transit_delta - self.jitter_ms) / 16.0
        self.max_jitter_ms = max(self.max_jitter_ms, transit_delta)
        self._last_arrival_ms = arrival_ms
        self._last_rtp_timestamp = timestamp

    def as_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "packets_received": self.packets_received,
                "packets_lost": self.packets_lost,
                "out_of_order_packets": self.out_of_order_packets,
                "duplicate_packets": self.duplicate_packets,
                "payload_bytes": self.payload_bytes,
                "decode_errors": self.decode_errors,
                "first_sequence": self.first_sequence,
                "last_sequence": self.last_sequence,
                "first_timestamp": self.first_timestamp,
                "last_timestamp": self.last_timestamp,
                "ssrc": self.ssrc,
                "jitter_ms": round(self.jitter_ms, 3),
                "max_jitter_ms": round(self.max_jitter_ms, 3),
                "peer": {"host": self.peer_host, "port": self.peer_port},
                "unexpected_peer_packets": self.unexpected_peer_packets,
                "receive_queue": {
                    "capacity_frames": self.receive_queue_capacity_frames,
                    "overflow_policy": "drop_oldest",
                    "frames_enqueued": self.receive_queue_frames_enqueued,
                    "frames_dequeued": self.receive_queue_frames_dequeued,
                    "frames_dropped": self.receive_queue_frames_dropped,
                    "frames_discarded_on_stop": self.receive_queue_frames_discarded_on_stop,
                    "overflows": self.receive_queue_overflows,
                    "max_depth_frames": self.receive_queue_max_depth_frames,
                    "depth_frames": self.receive_queue_depth_frames,
                    "receiver_running": self.receiver_running,
                    "receiver_errors": self.receiver_errors,
                },
            }

    def configure_receive_queue(self, capacity_frames: int) -> None:
        with self._lock:
            self.receive_queue_capacity_frames = capacity_frames

    def observe_receive_queue_enqueue(self, *, depth_frames: int, dropped_frames: int = 0) -> None:
        with self._lock:
            self.receive_queue_frames_enqueued += 1
            self.receive_queue_depth_frames = depth_frames
            self.receive_queue_max_depth_frames = max(self.receive_queue_max_depth_frames, depth_frames)
            if dropped_frames:
                self.receive_queue_overflows += 1
                self.receive_queue_frames_dropped += dropped_frames

    def observe_receive_queue_dequeue(self, *, depth_frames: int) -> None:
        with self._lock:
            self.receive_queue_frames_dequeued += 1
            self.receive_queue_depth_frames = depth_frames

    def observe_receive_queue_stop(self, *, discarded_frames: int) -> None:
        with self._lock:
            self.receive_queue_frames_discarded_on_stop += discarded_frames
            self.receive_queue_depth_frames = 0

    def set_receiver_running(self, running: bool) -> None:
        with self._lock:
            self.receiver_running = running

    def observe_receiver_error(self) -> None:
        with self._lock:
            self.receiver_errors += 1

    def observe_unexpected_peer(self) -> None:
        with self._lock:
            self.unexpected_peer_packets += 1

    def observe_decode_error(self) -> None:
        with self._lock:
            self.decode_errors += 1


class PacedRtpSender:
    """Streams PCM frames as 20ms ulaw RTP packets with cancellation support."""

    def __init__(
        self,
        send_packet: Callable[[bytes], None],
        *,
        state: RtpSendState | None = None,
        packet_ms: int = 20,
        prebuffer_ms: int = 0,
        clock: Callable[[], float] = time.monotonic,
        clock_sleep: Callable[[float], None] = time.sleep,
    ):
        self.send_packet = send_packet
        self.state = state or RtpSendState()
        self.packet_ms = packet_ms
        # Optional jitter cushion: accumulate this many ms of PCM before the
        # first packet of each stream so TTS-provider/decoder startup jitter
        # does not reach the caller as audio gaps. 0 (default) keeps the
        # historical send-immediately behavior. send_pcm16_stream() runs once
        # per TTS phrase, so the cushion applies per phrase start and shifts
        # first_rtp_packet_sent by up to prebuffer_ms (see rtp_prebuffer_ready
        # in the metrics for the actually buffered amount).
        self.prebuffer_ms = max(0, int(prebuffer_ms))
        self.clock = clock
        self.clock_sleep = clock_sleep
        self._last_packet_at: float | None = None

    def send_pcm16_stream(
        self,
        chunks: Iterable[bytes],
        *,
        sample_rate: int = 8000,
        cancellation: CancellationToken | None = None,
        metrics: CallMetrics | None = None,
    ) -> int:
        frame_samples = max(1, int(sample_rate * self.packet_ms / 1000))
        frame_bytes = frame_samples * 2
        buffered = b""
        packets = 0
        cancelled_by_token = False
        iterator = iter(chunks)
        source_exhausted = False
        try:
            # Phase 1 (optional): accumulate the prebuffer cushion before the
            # first packet. Short streams exit early via source_exhausted, so
            # phrases shorter than the cushion are not stalled.
            prebuffer_frames = 0
            if self.prebuffer_ms > 0:
                prebuffer_frames = max(1, (self.prebuffer_ms + self.packet_ms - 1) // self.packet_ms)
            prebuffer_target_bytes = prebuffer_frames * frame_bytes
            while prebuffer_target_bytes and len(buffered) < prebuffer_target_bytes:
                if cancellation and cancellation.cancelled:
                    cancelled_by_token = True
                    return packets
                try:
                    chunk = next(iterator)
                except StopIteration:
                    source_exhausted = True
                    break
                if not isinstance(chunk, bytes):
                    raise TypeError("RTP PCM chunks must be bytes")
                buffered += chunk
            if prebuffer_target_bytes and metrics:
                actual_ms = round((len(buffered) // 2) / max(1, sample_rate) * 1000, 3)
                metrics.mark(
                    "rtp_prebuffer_ready",
                    target_ms=self.prebuffer_ms,
                    actual_ms=actual_ms,
                    bytes=len(buffered),
                    source_exhausted=source_exhausted,
                )
            # Phase 2: refill to one frame, then send it paced at packet_ms.
            # Preserve the source's final samples by padding a trailing partial
            # frame with PCM silence instead of dropping the end of speech.
            while True:
                if cancellation and cancellation.cancelled:
                    cancelled_by_token = True
                    break
                while len(buffered) < frame_bytes and not source_exhausted:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        source_exhausted = True
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("RTP PCM chunks must be bytes")
                    buffered += chunk
                if len(buffered) < frame_bytes:
                    if source_exhausted and buffered:
                        buffered += b"\x00" * (frame_bytes - len(buffered))
                    else:
                        break
                if cancellation and cancellation.cancelled:
                    cancelled_by_token = True
                    return packets
                payload_pcm = buffered[:frame_bytes]
                buffered = buffered[frame_bytes:]
                marker, reason, gap_ms, timestamp_advance = self._prepare_talkspurt(
                    sample_rate=sample_rate,
                    frame_samples=frame_samples,
                )
                self._send_one(
                    payload_pcm,
                    marker=marker,
                    timestamp_advance=timestamp_advance,
                )
                if marker and metrics:
                    metrics.mark(
                        "rtp_talkspurt_started",
                        reason=reason,
                        marker=True,
                        inter_packet_gap_ms=gap_ms,
                        timestamp_advance_samples=timestamp_advance,
                        stream_packet_index=packets,
                    )
                packets += 1
                if packets == 1 and metrics:
                    first_rtp = metrics.mark("first_rtp_packet_sent", bytes=len(payload_pcm), packet_ms=self.packet_ms, sample_rate=sample_rate)
                    metrics.mark(
                        "caller_hears_first_audio_estimate",
                        at=first_rtp.at + (self.packet_ms / 1000.0),
                        basis="first_rtp_packet_sent",
                        packet_ms=self.packet_ms,
                    )
                self.clock_sleep(self.packet_ms / 1000)
            return packets
        finally:
            if cancelled_by_token and metrics:
                metrics.mark("rtp_stream_cancelled", packets=packets)
            close = getattr(iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    def _prepare_talkspurt(self, *, sample_rate: int, frame_samples: int) -> tuple[bool, str, float, int]:
        """Keep the RTP media clock continuous when silence is suppressed.

        The RTP timestamp already points one packet beyond the last payload.
        If at least one additional packet interval elapsed with no packet sent,
        advance over those suppressed samples and mark the resumed talkspurt.
        Checking every packet also covers a provider/decoder stall inside one
        PCM iterator, not only silence between separate TTS phrase streams.
        """

        if self._last_packet_at is None:
            return True, "initial", 0.0, 0

        elapsed_seconds = max(0.0, self.clock() - self._last_packet_at)
        elapsed_samples = round(elapsed_seconds * sample_rate)
        timestamp_advance = max(0, elapsed_samples - frame_samples)
        if timestamp_advance < frame_samples:
            return False, "continuous", round(elapsed_seconds * 1000.0, 3), 0

        return True, "gap", round(elapsed_seconds * 1000.0, 3), timestamp_advance

    def _send_one(self, pcm16: bytes, *, marker: bool = False, timestamp_advance: int = 0) -> None:
        samples = struct.unpack("<" + "h" * (len(pcm16) // 2), pcm16)
        payload = bytes(linear_to_mulaw(sample) for sample in samples)
        marker_and_payload_type = 0x80 if marker else 0x00
        packet_timestamp = self.state.timestamp + max(0, timestamp_advance)
        header = struct.pack(
            "!BBHII",
            0x80,
            marker_and_payload_type,
            self.state.sequence & 0xFFFF,
            packet_timestamp & 0xFFFFFFFF,
            self.state.ssrc,
        )
        self.send_packet(header + payload)
        self._last_packet_at = self.clock()
        self.state.sequence += 1
        self.state.timestamp = packet_timestamp + len(samples)


class UdpRtpTransport:
    """Receive RTP continuously and hand decoded caller frames to compute.

    Socket receipt, sequence, loss, and jitter accounting run on a dedicated
    thread. Slow STT/LLM/TTS work therefore cannot masquerade as network
    jitter. A bounded queue crosses into worker code; overflow drops the
    oldest audio frame and reports it separately from wire packet loss.
    """

    def __init__(
        self,
        bind: tuple[str, int],
        peer: tuple[str, int] | None = None,
        *,
        recv_timeout_seconds: float = 0.05,
        receive_queue_frames: int = 250,
        learn_peer_from_source: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        if receive_queue_frames < 1:
            raise ValueError("receive_queue_frames must be at least 1")
        self._peer_lock = threading.Lock()
        self._peer: tuple[str, int] | None = None
        self._peer_hosts: set[str] = set()
        self.peer = peer
        self._learn_peer_from_source = bool(learn_peer_from_source)
        self.recv_timeout_seconds = max(0.001, float(recv_timeout_seconds))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind)
        self.sock.settimeout(min(self.recv_timeout_seconds, 0.05))
        self.clock = clock
        self.stats = RtpReceiveStats()
        self.stats.configure_receive_queue(receive_queue_frames)
        self._receive_queue_frames = int(receive_queue_frames)
        self._receive_queue: deque[AudioFrame] = deque()
        self._receive_condition = threading.Condition()
        self._receiver_stop = threading.Event()
        self._receiver_stop_lock = threading.Lock()
        self._closed = threading.Event()
        self._receiver_error: Exception | None = None
        try:
            self._receiver_thread = threading.Thread(
                target=self._receive_loop,
                name="openphonex-rtp-receiver",
                daemon=True,
            )
            self._receiver_thread.start()
        except BaseException:
            # Construction failed after bind(), so no owner exists to call
            # close(). Release the port here before propagating the failure.
            self._receiver_stop.set()
            self._closed.set()
            with suppress(OSError):
                self.sock.close()
            receiver_thread = getattr(self, "_receiver_thread", None)
            if receiver_thread is not None and receiver_thread.is_alive():
                receiver_thread.join(timeout=1.0)
            raise

    def recv_frame(self) -> AudioFrame | None:
        deadline = time.monotonic() + self.recv_timeout_seconds
        with self._receive_condition:
            while not self._receive_queue and not self._receiver_stop.is_set() and self._receiver_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._receive_condition.wait(timeout=remaining)
            if self._receiver_stop.is_set():
                return None
            if self._receive_queue:
                frame = self._receive_queue.popleft()
                self.stats.observe_receive_queue_dequeue(depth_frames=len(self._receive_queue))
                return frame
            if self._receiver_error is not None:
                raise self._receiver_error
            return None

    def send_packet(self, packet: bytes) -> None:
        with self._peer_lock:
            peer = self._peer
        if not peer:
            raise RuntimeError("RTP peer is not known yet")
        self.sock.sendto(packet, peer)

    @property
    def peer(self) -> tuple[str, int] | None:
        with self._peer_lock:
            return self._peer

    @peer.setter
    def peer(self, peer: tuple[str, int] | None) -> None:
        # The media controller assigns the authoritative remote RTP endpoint after transport
        # construction. The receive loop never calls this setter: an inbound
        # datagram must not be able to redirect outbound agent audio.
        resolved_hosts = _resolve_ipv4_hosts(peer[0]) if peer is not None else set()
        with self._peer_lock:
            self._peer = peer
            self._peer_hosts = resolved_hosts

    def close(self) -> None:
        if self._closed.is_set():
            return
        try:
            self.stop_receiving()
        finally:
            self._closed.set()
            self.sock.close()

    def stop_receiving(self) -> None:
        """Stop ingress and freeze final receipt/queue metrics before close."""

        with self._receiver_stop_lock:
            if not self._receiver_stop.is_set():
                self._receiver_stop.set()
                with self._receive_condition:
                    discarded = len(self._receive_queue)
                    self._receive_queue.clear()
                    self.stats.observe_receive_queue_stop(discarded_frames=discarded)
                    self._receive_condition.notify_all()
            self._receiver_thread.join(timeout=1.0)
            if self._receiver_thread.is_alive():
                raise RuntimeError("RTP receiver did not stop within 1 second")

    def _receive_loop(self) -> None:
        self.stats.set_receiver_running(True)
        try:
            while not self._receiver_stop.is_set():
                try:
                    packet, addr = self.sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._receiver_stop.is_set() or self._closed.is_set():
                        break
                    self.stats.observe_receiver_error()
                    with self._receive_condition:
                        self._receiver_error = exc
                        self._receive_condition.notify_all()
                    break
                arrival_time = self.clock()
                try:
                    _rtp_header_fields(packet)
                except ValueError:
                    # A malformed datagram cannot claim the learned peer or
                    # terminate ingress for subsequent valid media.
                    self.stats.observe_decode_error()
                    continue
                if not self._accept_or_learn_peer(addr):
                    self.stats.observe_unexpected_peer()
                    continue
                self.stats.observe_packet(packet, peer=addr, arrival_time=arrival_time)
                try:
                    frame = AudioFrame(decode_rtp_ulaw_payload(packet), direction="caller")
                except Exception as exc:
                    self.stats.observe_decode_error()
                    self.stats.observe_receiver_error()
                    with self._receive_condition:
                        self._receiver_error = exc
                        self._receive_condition.notify_all()
                    break
                self._enqueue_received_frame(frame)
        finally:
            self.stats.set_receiver_running(False)
            with self._receive_condition:
                self._receive_condition.notify_all()

    def _enqueue_received_frame(self, frame: AudioFrame) -> None:
        with self._receive_condition:
            if self._receiver_stop.is_set():
                self.stats.observe_receive_queue_stop(discarded_frames=1)
                return
            dropped = 0
            if len(self._receive_queue) >= self._receive_queue_frames:
                self._receive_queue.popleft()
                dropped = 1
            self._receive_queue.append(frame)
            self.stats.observe_receive_queue_enqueue(
                depth_frames=len(self._receive_queue),
                dropped_frames=dropped,
            )
            self._receive_condition.notify()

    def _accept_or_learn_peer(self, addr: tuple[str, int]) -> bool:
        with self._peer_lock:
            if self._peer is None:
                if not self._learn_peer_from_source:
                    return False
                self._peer = addr
                self._peer_hosts = {addr[0]}
                return True
            return self._peer[1] == addr[1] and addr[0] in self._peer_hosts


def _rtp_header_fields(packet: bytes) -> tuple[int, int, int]:
    if len(packet) < 12:
        raise ValueError("RTP packet too short")
    if len(packet) == 12:
        raise ValueError("RTP packet has no audio payload")
    first, _payload_type, sequence, timestamp, ssrc = struct.unpack("!BBHII", packet[:12])
    version = first >> 6
    if version != 2:
        raise ValueError("unsupported RTP version")
    return sequence, timestamp, ssrc


def _resolve_ipv4_hosts(host: str) -> set[str]:
    hosts = {host}
    with suppress(OSError):
        hosts.update(
            str(sockaddr[0])
            for _family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(
                host,
                None,
                family=socket.AF_INET,
                type=socket.SOCK_DGRAM,
            )
        )
    return hosts


def _rtp_sequence_delta(sequence: int, previous: int) -> int:
    delta = (sequence - previous) & 0xFFFF
    if delta > 0x7FFF:
        return delta - 0x10000
    return delta


def _rtp_timestamp_delta(timestamp: int, previous: int) -> int:
    return (timestamp - previous) & 0xFFFFFFFF

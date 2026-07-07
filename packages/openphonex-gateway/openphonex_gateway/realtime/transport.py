# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import socket
import struct
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
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
    _last_arrival_ms: float | None = None
    _last_rtp_timestamp: int | None = None

    def observe_packet(self, packet: bytes, *, peer: tuple[str, int], arrival_time: float, sample_rate: int = 8000) -> None:
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
        }


class PacedRtpSender:
    """Streams PCM frames as 20ms ulaw RTP packets with cancellation support."""

    def __init__(
        self,
        send_packet: Callable[[bytes], None],
        *,
        state: RtpSendState | None = None,
        packet_ms: int = 20,
        clock_sleep: Callable[[float], None] = time.sleep,
    ):
        self.send_packet = send_packet
        self.state = state or RtpSendState()
        self.packet_ms = packet_ms
        self.clock_sleep = clock_sleep

    def send_pcm16_stream(
        self,
        chunks: Iterable[bytes],
        *,
        sample_rate: int = 8000,
        cancellation: CancellationToken | None = None,
        metrics: CallMetrics | None = None,
    ) -> int:
        frame_samples = max(1, int(sample_rate * self.packet_ms / 1000))
        buffered = b""
        packets = 0
        cancelled_by_token = False
        iterator = iter(chunks)
        try:
            while True:
                if cancellation and cancellation.cancelled:
                    cancelled_by_token = True
                    break
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                if not isinstance(chunk, bytes):
                    raise TypeError("RTP PCM chunks must be bytes")
                buffered += chunk
                usable = (len(buffered) // 2) * 2
                if usable != len(buffered):
                    buffered, remainder = buffered[:usable], buffered[usable:]
                else:
                    remainder = b""
                offset = 0
                frame_bytes = frame_samples * 2
                while offset + frame_bytes <= len(buffered):
                    if cancellation and cancellation.cancelled:
                        cancelled_by_token = True
                        return packets
                    payload_pcm = buffered[offset : offset + frame_bytes]
                    self._send_one(payload_pcm)
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
                    offset += frame_bytes
                buffered = buffered[offset:] + remainder
            return packets
        finally:
            if cancelled_by_token and metrics:
                metrics.mark("rtp_stream_cancelled", packets=packets)
            close = getattr(iterator, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    def _send_one(self, pcm16: bytes) -> None:
        samples = struct.unpack("<" + "h" * (len(pcm16) // 2), pcm16)
        payload = bytes(linear_to_mulaw(sample) for sample in samples)
        header = struct.pack("!BBHII", 0x80, 0x00, self.state.sequence & 0xFFFF, self.state.timestamp & 0xFFFFFFFF, self.state.ssrc)
        self.send_packet(header + payload)
        self.state.sequence += 1
        self.state.timestamp += len(samples)


class UdpRtpTransport:
    def __init__(
        self,
        bind: tuple[str, int],
        peer: tuple[str, int] | None = None,
        *,
        recv_timeout_seconds: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.peer = peer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind)
        self.sock.settimeout(recv_timeout_seconds)
        self.clock = clock
        self.stats = RtpReceiveStats()

    def recv_frame(self) -> AudioFrame | None:
        try:
            packet, addr = self.sock.recvfrom(2048)
        except socket.timeout:
            return None
        self.peer = addr
        self.stats.observe_packet(packet, peer=addr, arrival_time=self.clock())
        try:
            return AudioFrame(decode_rtp_ulaw_payload(packet), direction="caller")
        except Exception:
            self.stats.decode_errors += 1
            raise

    def send_packet(self, packet: bytes) -> None:
        if not self.peer:
            raise RuntimeError("RTP peer is not known yet")
        self.sock.sendto(packet, self.peer)

    def close(self) -> None:
        self.sock.close()


def _rtp_header_fields(packet: bytes) -> tuple[int, int, int]:
    first, _payload_type, sequence, timestamp, ssrc = struct.unpack("!BBHII", packet[:12])
    version = first >> 6
    if version != 2:
        raise ValueError("unsupported RTP version")
    return sequence, timestamp, ssrc


def _rtp_sequence_delta(sequence: int, previous: int) -> int:
    delta = (sequence - previous) & 0xFFFF
    if delta > 0x7FFF:
        return delta - 0x10000
    return delta


def _rtp_timestamp_delta(timestamp: int, previous: int) -> int:
    return (timestamp - previous) & 0xFFFFFFFF

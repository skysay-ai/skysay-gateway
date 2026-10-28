# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import struct

BIAS = 0x84
CLIP = 32635


def linear_to_mulaw(sample: int) -> int:
    sign = 0
    if sample < 0:
        sample = -sample
        sign = 0x80
    sample = min(sample, CLIP) + BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


MULAW_DECODE = []
for i in range(256):
    u_val = ~i & 0xFF
    t = ((u_val & 0x0F) << 3) + BIAS
    t <<= (u_val & 0x70) >> 4
    MULAW_DECODE.append(BIAS - t if (u_val & 0x80) else t - BIAS)


def mulaw_to_linear(value: int) -> int:
    return MULAW_DECODE[value & 0xFF]


def packetize_pcm16_as_rtp_ulaw(pcm16: bytes, *, ssrc: int = 0x12345678, start_sequence: int = 1, start_timestamp: int = 0) -> list[bytes]:
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    samples = struct.unpack("<" + "h" * (len(pcm16) // 2), pcm16)
    packets: list[bytes] = []
    seq = start_sequence
    ts = start_timestamp
    for idx in range(0, len(samples), 160):
        chunk = samples[idx : idx + 160]
        if len(chunk) < 160:
            break
        payload = bytes(linear_to_mulaw(s) for s in chunk)
        header = struct.pack("!BBHII", 0x80, 0x00, seq & 0xFFFF, ts & 0xFFFFFFFF, ssrc)
        packets.append(header + payload)
        seq += 1
        ts += 160
    return packets


def decode_rtp_ulaw_payload(packet: bytes) -> bytes:
    if len(packet) < 12:
        raise ValueError("RTP packet too short")
    pcm = bytearray()
    for value in packet[12:]:
        pcm.extend(struct.pack("<h", mulaw_to_linear(value)))
    return bytes(pcm)


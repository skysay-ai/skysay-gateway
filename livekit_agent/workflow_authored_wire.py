# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Bounded private framing for inactive authored speech sources.

Only the parent retains source identities. A child exchanges random lease IDs,
PCM and typed numeric facts; it never returns text, credentials or provider
error messages.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from enum import IntEnum


PROTOCOL = "skysay-authored-source-1"
PROVIDER, MODEL = "gemini", "gemini-3.1-flash-tts-preview"
SDK_VERSION = "2.31.0"
SAMPLE_RATE = 24000
MAX_PACKET = 256 * 1024
MAX_PCM_FRAME = 65536
MAX_PCM_BYTES = 2 * 1024 * 1024
MAX_PCM_FRAMES = 4096
MAX_TEXT_UTF8 = 16384
MAX_TOKENS = 100_000_000
MAX_SOURCES = 4096
HEADER = struct.Struct("!BI")
LEASE = re.compile(r"[0-9a-f]{32}")
REFERENCE = re.compile(r"[A-Za-z0-9_-]{16,96}")


class WireRefused(RuntimeError):
    """Fixed failure without untrusted payload or provider details."""


class Kind(IntEnum):
    PREFLIGHT = 1
    CONFIG = 2
    SEEN = 3
    PCM = 4
    END = 5
    RELEASE = 6
    CANCEL = 7
    CLOSED = 8
    ERROR = 9
    STOP = 10


def refuse():
    raise WireRefused("finite authored source refused")


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            refuse()
        result[key] = value
    return result


def _number(_value):
    refuse()


def unpack_json(payload, keys):
    result = value = None
    try:
        if type(payload) is not bytes or len(payload) > MAX_PACKET:
            refuse()
        value = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_number)
        if type(value) is not dict or set(value) != keys:
            refuse()
        result = value
    except Exception:
        pass
    finally:
        del payload, value
    if result is None:
        refuse()
    return result


def pack_json(value):
    result = None
    try:
        result = json.dumps(value, ensure_ascii=False, allow_nan=False,
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(result) > MAX_PACKET:
            result = None
    except Exception:
        pass
    finally:
        del value
    if result is None:
        refuse()
    return result


def encode(kind, payload):
    result = None
    try:
        if not isinstance(kind, Kind) or type(payload) is not bytes or len(payload) > MAX_PACKET:
            refuse()
        result = HEADER.pack(kind, len(payload)) + payload
    except Exception:
        pass
    finally:
        del kind, payload
    if result is None:
        refuse()
    return result


def header(value):
    result = None
    try:
        if type(value) is not bytes or len(value) != HEADER.size:
            refuse()
        raw_kind, length = HEADER.unpack(value)
        if length > MAX_PACKET:
            refuse()
        result = Kind(raw_kind), length
    except Exception:
        pass
    finally:
        del value
    if result is None:
        refuse()
    return result


def lease(value):
    return type(value) is str and LEASE.fullmatch(value) is not None


def reference(value):
    return type(value) is str and REFERENCE.fullmatch(value) is not None


def token_pair(value):
    return (type(value) in (list, tuple) and len(value) == 2
            and all(type(number) is int and 0 <= number <= MAX_TOKENS for number in value))


def control(payload):
    value = unpack_json(payload, {"lease"})
    if not lease(value["lease"]):
        refuse()
    return value["lease"]


def voice(value):
    return (type(value) is str and 1 <= len(value) <= 256 and value == value.strip()
            and all(character.isprintable() for character in value)
            and len(value.encode("utf-8")) <= 256)


def audio_format(value):
    """Synthetic contract forms; neither literal is provider qualification."""
    if type(value) is str:
        fields = value.split(";")
        if fields[0].lower() == "audio/pcm" and fields[1:] == ["rate=24000"]:
            return "pcm"
        if fields[0].lower() == "audio/l16" and fields[1:] == ["codec=pcm", "rate=24000"]:
            return "l16"
    refuse()


@dataclass(slots=True, weakref_slot=True, repr=False)
class Request:
    lease_id: str
    secret: str = field(repr=False)
    text: str = field(repr=False)
    voice_id: str

    def __repr__(self):
        return "<AuthoredSourceRequest redacted>"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("authored source requests cannot be serialized")

    def clear(self):
        self.secret = self.text = ""


def configuration(payload):
    result = value = None
    try:
        value = unpack_json(payload, {"lease", "provider", "model", "voice_id", "secret", "text"})
        if (not lease(value["lease"]) or value["provider"] != PROVIDER or value["model"] != MODEL
                or type(value["provider"]) is not str or type(value["model"]) is not str
                or not voice(value["voice_id"])
                or type(value["secret"]) is not str or not value["secret"].strip()
                or len(value["secret"]) > 16384
                or type(value["text"]) is not str or not value["text"].strip()
                or len(value["text"]) > MAX_TEXT_UTF8
                or len(value["text"].encode("utf-8")) > MAX_TEXT_UTF8):
            refuse()
        result = Request(value["lease"], value["secret"], value["text"], value["voice_id"])
    except Exception:
        pass
    finally:
        del payload, value
    if result is None:
        refuse()
    return result


def pcm(payload):
    if (type(payload) is not bytes or len(payload) <= 32 or len(payload) > 32 + MAX_PCM_FRAME
            or (len(payload) - 32) % 2):
        refuse()
    try:
        identity = payload[:32].decode("ascii")
    except UnicodeError:
        identity = ""
    if not lease(identity):
        refuse()
    return identity, payload[32:]


@dataclass(frozen=True, slots=True)
class End:
    lease_id: str
    response_seen: bool
    tokens: tuple[int, int] | None
    response_ended: bool
    functional: bool
    pcm_frames: int
    pcm_bytes: int


def ending(payload):
    value = unpack_json(payload, {"lease", "response_seen", "tokens", "response_ended", "functional", "frames", "bytes"})
    if (not lease(value["lease"])
            or any(type(value[key]) is not bool for key in ("response_seen", "response_ended", "functional"))
            or value["tokens"] is not None and not token_pair(value["tokens"])
            or type(value["frames"]) is not int or not 0 <= value["frames"] <= MAX_PCM_FRAMES
            or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= MAX_PCM_BYTES
            or value["bytes"] % 2 or bool(value["frames"]) != bool(value["bytes"])
            or value["functional"] and not (value["response_seen"] and value["response_ended"] and value["bytes"])
            or value["tokens"] is not None and not (value["response_seen"] and value["response_ended"])):
        refuse()
    return End(value["lease"], value["response_seen"],
               None if value["tokens"] is None else tuple(value["tokens"]),
               value["response_ended"], value["functional"], value["frames"], value["bytes"])


class IncomingLease:
    """Parent-side transcript of one lease, independent of provider assertions."""
    def __init__(self, identity):
        if not lease(identity):
            refuse()
        self.identity = identity
        self.seen = self.closed = False
        self.frames = self.bytes = 0
        self.end = None

    def accept(self, kind, payload):
        if self.closed:
            refuse()
        if kind == Kind.SEEN:
            if self.seen or self.end is not None or control(payload) != self.identity:
                refuse()
            self.seen = True
        elif kind == Kind.PCM:
            identity, data = pcm(payload)
            if not self.seen or self.end is not None or identity != self.identity:
                refuse()
            self.frames += 1
            self.bytes += len(data)
            if self.frames > MAX_PCM_FRAMES or self.bytes > MAX_PCM_BYTES:
                refuse()
            return data
        elif kind == Kind.END:
            end = ending(payload)
            if (self.end is not None or end.lease_id != self.identity or end.response_seen != self.seen
                    or (end.pcm_frames, end.pcm_bytes) != (self.frames, self.bytes)):
                refuse()
            self.end = end
        elif kind == Kind.CLOSED:
            if self.end is None or control(payload) != self.identity:
                refuse()
            self.closed = True
        else:
            refuse()
        return None

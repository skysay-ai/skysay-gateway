from __future__ import annotations

import base64
import os
import json
import re
import socket
import ssl
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Protocol

from .transport import UdpRtpTransport


class AriClientProtocol(Protocol):
    def post(self, path: str, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        ...

    def delete(self, path: str) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True)
class AsteriskExternalMediaConfig:
    ari_url: str = "http://127.0.0.1:8088/ari"
    ari_user: str = ""
    ari_password: str = ""
    app: str = "voicebot"
    external_host: str = "telephony-ai-realtime-worker"
    rtp_port: int = 10000
    asterisk_rtp_host: str = ""
    format: str = "ulaw"
    direction: str = "both"
    answer_inbound: bool = True

    @property
    def external_media_host(self) -> str:
        return f"{self.external_host}:{self.rtp_port}"

    def redacted(self) -> dict[str, Any]:
        return {
            "ari_url": self.ari_url,
            "app": self.app,
            "external_host": self.external_media_host,
            "asterisk_rtp_host": self.asterisk_rtp_host,
            "format": self.format,
            "direction": self.direction,
            "answer_inbound": self.answer_inbound,
            "secrets_redacted": True,
            "missing_live_env": [key for key, value in (("ASTERISK_ARI_USER", self.ari_user), ("ASTERISK_ARI_PASSWORD", self.ari_password)) if not value],
        }


class AsteriskRestClient:
    def __init__(self, config: AsteriskExternalMediaConfig):
        self.config = config
        token = base64.b64encode(f"{config.ari_user}:{config.ari_password}".encode()).decode()
        self.auth_header = f"Basic {token}"

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = self.config.ari_url.rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        payload = json.dumps(body or {}).encode() if body is not None else b""
        request = urllib.request.Request(url, data=payload if method != "GET" else None, method=method)
        request.add_header("Authorization", self.auth_header)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read()
        return json.loads(data.decode()) if data else None

    def post(self, path: str, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return self._request("POST", path, params, body)

    def delete(self, path: str) -> dict[str, Any] | None:
        return self._request("DELETE", path)


class AriWebSocketEventClient:
    """Small stdlib ARI event stream client for the realtime worker.

    Asterisk exposes ARI events over WebSocket. Keeping this here avoids adding a
    dependency before the live ExternalMedia path is proven.
    """

    def __init__(self, config: AsteriskExternalMediaConfig):
        self.config = config

    def events_url(self) -> str:
        parsed = urllib.parse.urlparse(self.config.ari_url.rstrip("/"))
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        query = urllib.parse.urlencode({"app": self.config.app, "api_key": f"{self.config.ari_user}:{self.config.ari_password}"})
        return urllib.parse.urlunparse((scheme, parsed.netloc, f"{base_path}/events", "", query, ""))

    def iter_events(self, stop: Event | None = None) -> Any:
        with _WebSocketConnection(self.events_url()) as ws:
            while not (stop and stop.is_set()):
                payload = ws.recv_text(timeout=1.0)
                if payload is None:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


class _WebSocketConnection:
    def __init__(self, url: str):
        self.url = url
        self.sock: socket.socket | ssl.SSLSocket | None = None

    def __enter__(self) -> "_WebSocketConnection":
        parsed = urllib.parse.urlparse(self.url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        raw = socket.create_connection((host, port), timeout=10)
        if parsed.scheme == "wss":
            self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        else:
            self.sock = raw
        self.sock.settimeout(10)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {parsed.netloc}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "\r\n",
        ]
        self.sock.sendall("\r\n".join(headers).encode("ascii"))
        response = self._read_http_response()
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"ARI WebSocket handshake failed: {response.splitlines()[0].decode(errors='replace')}")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.sock:
            try:
                self._send_frame(0x8, b"")
            except Exception:
                pass
            self.sock.close()
            self.sock = None

    def recv_text(self, *, timeout: float = 1.0) -> str | None:
        if not self.sock:
            raise RuntimeError("WebSocket is not connected")
        self.sock.settimeout(timeout)
        try:
            opcode, payload = self._read_frame()
        except socket.timeout:
            return None
        if opcode == 0x1:
            return payload.decode("utf-8", errors="replace")
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            self._send_frame(0xA, payload)
        return None

    def _read_http_response(self) -> bytes:
        assert self.sock is not None
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            data += chunk
        return data

    def _read_frame(self) -> tuple[int, bytes]:
        assert self.sock is not None
        header = self._read_exact(2)
        first, second = header[0], header[1]
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
        return opcode, payload

    def _read_exact(self, size: int) -> bytes:
        assert self.sock is not None
        data = b""
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise RuntimeError("WebSocket closed while reading frame")
            data += chunk
        return data

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        assert self.sock is not None
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | length)
        elif length <= 0xFFFF:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, length)
        masked = bytes(byte ^ mask[idx % 4] for idx, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)


@dataclass
class AsteriskExternalMediaState:
    primary_channel_id: str | None = None
    bridge_id: str | None = None
    external_channel_id: str | None = None
    call_direction: str = "inbound"
    call_direction_source: str = "default_inbound"
    primary_channel_state: str = ""
    openphonex: dict[str, str] = field(default_factory=dict)
    external_channel_added: bool = False
    primary_channel_added: bool = False
    rtp_peer: tuple[str, int] | None = None
    primary_ended: bool = False

    def primary_channel_is_up(self) -> bool:
        # Unknown state (ARI event without a channel state) is treated as up so
        # the outbound worker does not stall waiting for a ChannelStateChange
        # that will never arrive.
        return (self.primary_channel_state or "Up").strip().lower() == "up"


class AsteriskExternalMediaSession:
    """Owns only Asterisk bridge/ExternalMedia setup for the realtime worker."""

    def __init__(self, *, ari: AriClientProtocol, config: AsteriskExternalMediaConfig, transport: UdpRtpTransport | None = None):
        self.ari = ari
        self.config = config
        self.transport = transport
        self.state = AsteriskExternalMediaState()

    def handle_event(self, event: dict[str, Any]) -> str:
        event_type = str(event.get("type") or "")
        channel = event.get("channel") or {}
        channel_id = channel.get("id")
        channel_name = str(channel.get("name") or "")
        if event_type == "StasisStart":
            if channel_name.startswith("UnicastRTP"):
                self.state.external_channel_id = str(channel_id)
                self.ensure_external_added()
                return "external_started"
            if self.state.primary_channel_id and channel_id != self.state.primary_channel_id:
                return "secondary_ignored"
            self.state.primary_channel_id = str(channel_id)
            self.state.primary_channel_state = str(channel.get("state") or "")
            self.state.call_direction, self.state.call_direction_source = _call_direction_from_event(event)
            self.state.openphonex = _openphonex_metadata_from_event(event)
            if self.config.answer_inbound and channel.get("state") != "Up":
                self.ari.post(f"/channels/{self.state.primary_channel_id}/answer")
            self.setup_bridge_external_media()
            return "primary_started"
        if event_type == "ChannelStateChange" and channel_id == self.state.primary_channel_id:
            self.state.primary_channel_state = str(channel.get("state") or "")
            return "primary_state_changed"
        if event_type == "StasisEnd" and channel_id == self.state.primary_channel_id:
            self.state.primary_ended = True
            self.cleanup()
            return "primary_ended"
        return "ignored"

    def setup_bridge_external_media(self) -> None:
        if not self.state.primary_channel_id:
            return
        if not self.state.bridge_id:
            bridge = self.ari.post("/bridges", {"type": "mixing", "name": "openphonex-realtime"})
            self.state.bridge_id = str((bridge or {}).get("id") or "openphonex-realtime")
        if not self.state.primary_channel_added:
            self.ari.post(f"/bridges/{self.state.bridge_id}/addChannel", {"channel": self.state.primary_channel_id})
            self.state.primary_channel_added = True
        if not self.state.external_channel_id:
            external = self.ari.post(
                "/channels/externalMedia",
                {
                    "app": self.config.app,
                    "external_host": self.config.external_media_host,
                    "format": self.config.format,
                    "encapsulation": "rtp",
                    "transport": "udp",
                    "connection_type": "client",
                    "direction": self.config.direction,
                },
            )
            self.state.external_channel_id = str((external or {}).get("id") or "")
            self._capture_rtp_peer(external or {})
        self.ensure_external_added()

    def ensure_external_added(self) -> None:
        if self.state.bridge_id and self.state.external_channel_id and not self.state.external_channel_added:
            self.ari.post(f"/bridges/{self.state.bridge_id}/addChannel", {"channel": self.state.external_channel_id})
            self.state.external_channel_added = True

    def cleanup(self) -> None:
        if self.state.external_channel_id:
            try:
                self.ari.delete(f"/channels/{self.state.external_channel_id}")
            except Exception:
                pass

    def _capture_rtp_peer(self, external: dict[str, Any]) -> None:
        vars_ = external.get("channelvars") or {}
        address = vars_.get("UNICASTRTP_LOCAL_ADDRESS")
        port = vars_.get("UNICASTRTP_LOCAL_PORT")
        if address and port:
            peer = (self.config.asterisk_rtp_host or str(address), int(port))
            self.state.rtp_peer = peer
            if self.transport:
                self.transport.peer = peer


OPENPHONEX_METADATA_VARIABLES: dict[str, str] = {
    "OPENPHONEX_CALL_ID": "call_id",
    "OPENPHONEX_ORGANIZATION_ID": "organization_id",
    "OPENPHONEX_PROJECT_ID": "project_id",
    "OPENPHONEX_NUMBER_ID": "number_id",
    "OPENPHONEX_AGENT_ID": "agent_id",
    "OPENPHONEX_AGENT_ENDPOINT_ID": "agent_endpoint_id",
    "OPENPHONEX_PHONE_SERVER_ID": "phone_server_id",
    "OPENPHONEX_TRIAL_CALL_ID": "trial_call_id",
    "OPENPHONEX_MEDIA_PIPELINE": "media_pipeline",
    "OPENPHONEX_WORKER_JOB_ID": "worker_job_id",
}


def _channel_variables_from_event(event: dict[str, Any]) -> dict[str, Any]:
    channel = event.get("channel") or {}
    variables: dict[str, Any] = {}
    if isinstance(channel, dict):
        for key in ("variables", "channelvars", "dialplan"):
            value = channel.get(key)
            if isinstance(value, dict):
                variables.update(value)
    return variables


def _call_direction_from_event(event: dict[str, Any]) -> tuple[str, str]:
    variables = _channel_variables_from_event(event)
    for key in (
        "OPENPHONEX_CALL_DIRECTION",
        "OPENPHONEX_DIRECTION",
        "CALL_DIRECTION",
        "call_direction",
        "direction",
    ):
        normalized = _normalize_call_direction(variables.get(key))
        if normalized:
            return normalized, f"channel_variable:{key}"
    app_arg_directions = {
        direction
        for direction in (_normalize_call_direction(arg) for arg in _stasis_app_args(event))
        if direction
    }
    if len(app_arg_directions) == 1:
        return next(iter(app_arg_directions)), "stasis_app_args"
    return "inbound", "default_inbound"


def _stasis_app_args(event: dict[str, Any]) -> list[str]:
    """Stasis application arguments from the origination (e.g. appArgs=outbound)."""

    args = event.get("args")
    parts: list[str] = []
    if isinstance(args, list):
        for arg in args:
            parts.extend(str(arg).split(","))
    elif isinstance(args, str):
        parts.extend(args.split(","))
    return [part.strip() for part in parts if part.strip()]


def _openphonex_metadata_from_event(event: dict[str, Any]) -> dict[str, str]:
    variables = _channel_variables_from_event(event)
    metadata: dict[str, str] = {}
    for variable_name, metadata_key in OPENPHONEX_METADATA_VARIABLES.items():
        value = _safe_openphonex_metadata_value(variables.get(variable_name))
        if value:
            metadata[metadata_key] = value
    return metadata


def _safe_openphonex_metadata_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if (text.startswith("+") and len(digits) >= 6) or (len(digits) >= 7 and not re.search(r"[A-Za-z_]", text)):
        return ""
    if len(text) > 160:
        text = text[:160]
    return re.sub(r"[^A-Za-z0-9_.:@/-]", "_", text)


def _normalize_call_direction(value: Any) -> str:
    direction = str(value or "").strip().lower().replace("-", "_")
    if direction in {"inbound", "incoming", "in"}:
        return "inbound"
    if direction in {"outbound", "outgoing", "out"}:
        return "outbound"
    return ""

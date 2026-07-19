# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Portable OpenPhonex control-plane boundary for the LiveKit candidate.

The media runtime may be LiveKit, Asterisk, or another SIP stack.  This module
keeps the contracts which must remain outside model-generated input: call
authorization, resolved agent configuration, tool authorization, evidence
import, billing-duration provenance, and idempotent media cleanup.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from .evidence_contract import (
        valid_effective_voice_stack_identity,
        valid_semantic_response_audio_policy,
    )
except ImportError:  # copied into /app in the pinned sandbox image
    from evidence_contract import (
        valid_effective_voice_stack_identity,
        valid_semantic_response_audio_policy,
    )


class ControlPlaneRejected(RuntimeError):
    """The authoritative control plane rejected or malformed a request."""


class SessionStateError(RuntimeError):
    """A media action was attempted outside the permitted session state."""


class IncompleteArtifactEvidence(RuntimeError):
    """The caller-side PR #79 artifact set is incomplete."""


@dataclass(frozen=True)
class OutboundDispatch:
    call_id: str
    agent_id: str
    from_number_id: str
    from_e164: str
    to_e164: str
    participant_identity: str


def parse_outbound_dispatch(metadata: str) -> OutboundDispatch | None:
    """Parse non-secret LiveKit dispatch metadata for an outbound call."""

    if not metadata:
        return None
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("LiveKit dispatch metadata must be valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "openphonex.outbound-dispatch.v1":
        return None
    fields = {
        name: str(payload.get(name) or "").strip()
        for name in (
            "call_id",
            "agent_id",
            "from_number_id",
            "from_e164",
            "to_e164",
            "participant_identity",
        )
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ValueError(f"outbound dispatch omitted required fields: {', '.join(missing)}")
    if not fields["from_e164"].startswith("+") or not fields["to_e164"].startswith("+"):
        raise ValueError("outbound dispatch numbers must use E.164 format")
    return OutboundDispatch(**fields)


class JsonTransport(Protocol):
    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _origin(split: urllib.parse.SplitResult) -> tuple[str, str, int]:
    default_port = {"http": 80, "https": 443}.get(split.scheme, -1)
    port = split.port if split.port is not None else default_port
    return (split.scheme, (split.hostname or "").lower(), port)


def _require_confidential_base_url(base_url: str) -> None:
    """Bearer credentials only travel over https, or http to verified loopback.

    Production is unaffected (it uses https://api.openphonex.com or the
    host-local http://127.0.0.1 control plane); the guard exists for
    self-hosters who would otherwise leak the worker/gateway tokens in
    plaintext by pointing OPENPHONEX_CONTROL_PLANE_URL at a remote http URL.
    """

    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        return
    if split.scheme == "http" and _is_loopback_host(split.hostname):
        return
    raise ValueError(
        "control-plane base_url must be https, or http on verified loopback "
        f"(127.0.0.0/8, ::1, localhost); refusing to send bearer credentials to {base_url!r}"
    )


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that would replay the Authorization header off-origin.

    urllib's default handler copies request headers (including Authorization)
    onto the redirected request; a compromised or misconfigured control plane
    could use that to exfiltrate the bearer token to another host. Same-origin
    redirects (scheme, host, and effective port all unchanged) keep working.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        # urllib resolves Location against the request URL before calling this
        # hook; joining again is a no-op there but keeps this hook correct for
        # a relative newurl too.
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        target = urllib.parse.urlsplit(absolute)
        if _origin(target) != _origin(urllib.parse.urlsplit(req.full_url)):
            raise ControlPlaneRejected(
                f"control plane redirected across origins (HTTP {code} to "
                f"{target.scheme}://{target.netloc}); refusing to forward bearer credentials"
            )
        return super().redirect_request(req, fp, code, msg, headers, absolute)


class UrllibJsonTransport:
    """Small bearer-auth JSON client; credentials never enter room metadata.

    Transport hardening: the base URL must be https or verified
    loopback http, and cross-origin redirects are refused so the bearer token
    can never be replayed to another host.
    """

    def __init__(
        self, *, base_url: str, bearer_token: str, timeout_seconds: float = 10.0
    ) -> None:
        _require_confidential_base_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        handlers: list[urllib.request.BaseHandler] = [_SameOriginRedirectHandler()]
        # A loopback control plane must never be reached through a proxy: the
        # default ProxyHandler honours HTTP_PROXY/http_proxy, which would route
        # the loopback-http request (and its Authorization: Bearer header)
        # through an external proxy host. Pin an empty ProxyHandler (no proxy)
        # for loopback. For https we keep the default proxy behaviour so an
        # operator's HTTPS_PROXY still works, and TLS keeps the token
        # confidential end-to-end via CONNECT.
        if _is_loopback_host(urllib.parse.urlsplit(base_url).hostname):
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.bearer_token}")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise ControlPlaneRejected(f"control plane returned HTTP {exc.code}: {detail}") from exc
        if not isinstance(decoded, dict):
            raise ControlPlaneRejected("control plane returned a non-object response")
        return decoded


@dataclass
class Gate2CallSession:
    transport: JsonTransport
    authorization_transport: JsonTransport
    tool_transport: JsonTransport
    call_id: str
    gateway_id: str
    authorization_call_id: str
    call_authorization_token: str = field(repr=False)
    config: dict[str, Any]
    from_number_id: str = ""
    to_e164: str = ""
    media_admitted: bool = False
    audit_events: list[str] = field(
        default_factory=lambda: ["authorization_allowed", "realtime_config_loaded"]
    )
    _cleaned: bool = False
    _completed: bool = False
    _lifecycle_finalized: bool = False
    _tool_sequence: int = 0

    @property
    def allowed_tools(self) -> frozenset[str]:
        tools = self.config.get("tools")
        if not isinstance(tools, list) or any(
            not isinstance(item, str) or not item for item in tools
        ):
            raise ControlPlaneRejected("realtime config tools must be a list of non-empty names")
        return frozenset(tools)

    def admit_media(self) -> None:
        if self._cleaned:
            raise SessionStateError("cannot admit media after cleanup")
        if self.media_admitted:
            return
        # Construction is only possible after authorization and config fetch.
        if self.audit_events[:2] != ["authorization_allowed", "realtime_config_loaded"]:
            raise SessionStateError("authorization and realtime config must precede media")
        if self.call_authorization_token:
            consumed = self.authorization_transport.request(
                "POST",
                "/v1/gateway-call-authorizations/consume",
                {
                    "from_number_id": self.from_number_id,
                    "to": self.to_e164,
                },
            )
            if consumed.get("status") != "consumed" or consumed.get("call_id") != self.call_id:
                raise ControlPlaneRejected("single-call authorization was not consumed for this call")
            self.call_authorization_token = ""
            self.audit_events.append("authorization_consumed")
        else:
            self.audit_events.append("inbound_policy_admitted")
        self.media_admitted = True
        self.audit_events.append("media_admitted")

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("tool invocation requires admitted live media")
        if name not in self.allowed_tools:
            self.audit_events.append("tool_rejected_not_allowlisted")
            raise ControlPlaneRejected(f"tool is not allowlisted for this agent: {name}")
        self._tool_sequence += 1
        request_id = f"{self.call_id}:{self._tool_sequence}"
        result = self.tool_transport.request(
            "POST",
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        self.audit_events.append("tool_invoked_allowlisted")
        return result

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Fetch schemas, then expose only names allowed by realtime-config."""

        response = self.tool_transport.request(
            "POST",
            "/mcp",
            {"jsonrpc": "2.0", "id": f"{self.call_id}:tools", "method": "tools/list"},
        )
        result = response.get("result")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise ControlPlaneRejected("MCP tools/list returned no tool definitions")
        definitions = [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("name") in self.allowed_tools
        ]
        if {str(tool.get("name")) for tool in definitions} != set(self.allowed_tools):
            raise ControlPlaneRejected("realtime-config allowlist names a tool absent from MCP tools/list")
        self.audit_events.append("tool_schemas_filtered_to_allowlist")
        return definitions

    def report_lifecycle(
        self,
        *,
        event_id: str,
        event_type: str,
        source_component: str,
        observed_at: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("lifecycle reporting requires admitted live media")
        payload: dict[str, Any] = {
            "event_id": event_id,
            "call_id": self.call_id,
            "event_type": event_type,
            "source_component": source_component,
        }
        if observed_at:
            payload["observed_at"] = observed_at
        if details:
            payload["details"] = details
        result = self.transport.request(
            "POST",
            f"/v1/gateway-deployments/{self.gateway_id}/call-lifecycle-events",
            payload,
        )
        expected = {
            "sip.answered": "call.answered",
            "room.media_started": "media.started",
            "room.media_heartbeat": "media.heartbeat",
            "room.ended": "media.ended",
            "sip.ended": "call.ended",
            "sip.failed": "call.failed",
            "session.reconciled": "call.ended",
        }.get(event_type)
        if not expected or result.get("canonical_type") != expected:
            raise ControlPlaneRejected("control plane rejected the lifecycle event contract")
        self.audit_events.append(f"lifecycle_{event_type.replace('.', '_')}")
        if expected in {"call.ended", "call.failed"}:
            billing = result.get("billing")
            if not isinstance(billing, dict) or billing.get("status") not in {
                "settled",
                "released",
                "no_reservation",
            }:
                raise ControlPlaneRejected("terminal call event did not produce an authoritative billing verdict")
            self._lifecycle_finalized = True
            self.audit_events.append("billing_finalized_from_lifecycle")
        return result

    def complete(
        self, *, source_path: Path, duration_sources: dict[str, float]
    ) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("completion requires admitted media before cleanup")
        if self._completed:
            raise SessionStateError("call completion is already finalized")
        if not self._lifecycle_finalized:
            raise SessionStateError("artifact upload requires finalized gateway lifecycle billing")
        _validate_pr79_artifact_set(source_path)
        normalized_duration = _duration_provenance(duration_sources)
        provenance = json.dumps(normalized_duration, sort_keys=True).encode()
        artifact_paths = [
            source_path / "caller.wav",
            source_path / "agent.wav",
            source_path / "mixed.wav",
            source_path / "transcript.json",
            *sorted(source_path.glob("turn-*-caller.wav")),
            *sorted(source_path.glob("turn-*-agent.wav")),
        ]
        parts = [_upload_part(path.name, path.read_bytes()) for path in artifact_paths]
        parts.append(_upload_part("duration-provenance.json", provenance))
        payload = {
            "parts": parts,
        }
        imported = self.transport.request(
            "POST", f"/v1/calls/{self.call_id}/artifacts", payload
        )
        self._completed = True
        self.audit_events.extend(["billing_duration_provenance_recorded", "artifacts_imported"])
        return {**normalized_duration, "artifact_import": imported}

    def cleanup(
        self,
        *,
        remove_participant: Callable[[], None],
        delete_room: Callable[[], None],
    ) -> dict[str, Any]:
        if self._cleaned:
            return {"clean": True, "already_clean": True}
        errors: list[dict[str, str]] = []
        for resource, operation in (
            ("participant", remove_participant),
            ("room", delete_room),
        ):
            try:
                operation()
                self.audit_events.append(f"{resource}_removed")
            except Exception as exc:  # cleanup must continue to the second resource
                errors.append({"resource": resource, "error": type(exc).__name__})
                self.audit_events.append(f"{resource}_cleanup_failed")
        self._cleaned = not errors
        if self._cleaned:
            self.audit_events.append("cleanup_complete")
        return {"clean": self._cleaned, "errors": errors}


class Gate2ControlPlaneAdapter:
    def __init__(
        self,
        *,
        gateway_id: str,
        transport: JsonTransport,
        tool_transport: JsonTransport | None = None,
        call_transport: JsonTransport | None = None,
    ) -> None:
        if not gateway_id:
            raise ValueError("gateway_id is required")
        self.gateway_id = gateway_id
        self.transport = transport
        # Production should inject an organization-scoped tool credential;
        # narrow gateway tokens deliberately cannot invoke customer tools.
        self.tool_transport = tool_transport or transport
        # Inbound routing is an organization call mutation and deliberately is
        # not granted to a narrow gateway token.
        self.call_transport = call_transport or transport

    def report_failed_before_media(
        self, *, call_id: str, event_id: str, reason: str
    ) -> dict[str, Any]:
        """Close a reserved call which policy prevented from reaching media."""

        result = self.transport.request(
            "POST",
            f"/v1/gateway-deployments/{self.gateway_id}/call-lifecycle-events",
            {
                "event_id": event_id,
                "call_id": call_id,
                "event_type": "sip.failed",
                "source_component": "livekit_sip",
                "details": {"reason": reason},
            },
        )
        billing = result.get("billing") if isinstance(result, dict) else None
        if result.get("canonical_type") != "call.failed" or not isinstance(
            billing, dict
        ) or billing.get("status") not in {"released", "no_reservation"}:
            raise ControlPlaneRejected(
                "pre-media failure did not release the call reservation"
            )
        return result

    def prepare_outbound(
        self,
        *,
        call_id: str,
        agent_id: str,
        from_number_id: str,
        to_e164: str,
        metadata: dict[str, Any],
    ) -> Gate2CallSession:
        # This endpoint evaluates policy and only then mints a destination-bound,
        # single-use call token. Nothing media-related happens before it passes.
        authorization = self.transport.request(
            "POST",
            f"/v1/gateway-deployments/{self.gateway_id}/call-authorizations",
            {
                "call_id": call_id,
                "from_number_id": from_number_id,
                "to": to_e164,
                "metadata": metadata,
            },
        )
        if authorization.get("authorized") is not True:
            raise ControlPlaneRejected("outbound call authorization was denied")
        authorization_call_id = authorization.get("call_id")
        token = authorization.get("token")
        if not isinstance(call_id, str) or not call_id:
            raise ControlPlaneRejected("outbound orchestration omitted its existing call id")
        if (
            not isinstance(authorization_call_id, str)
            or not authorization_call_id
            or not isinstance(token, str)
            or not token
        ):
            raise ControlPlaneRejected(
                "allowed authorization omitted the call id or single-use token"
            )
        if authorization_call_id != call_id:
            raise ControlPlaneRejected("single-call authorization is not bound to the session call id")

        config = self.transport.request("GET", f"/v1/agents/{agent_id}/realtime-config")
        _validate_realtime_config(config, expected_agent_id=agent_id)
        consuming_transport = UrllibJsonTransport(
            base_url=self.transport.base_url,
            bearer_token=token,
            timeout_seconds=self.transport.timeout_seconds,
        ) if isinstance(self.transport, UrllibJsonTransport) else self.transport
        return Gate2CallSession(
            transport=self.transport,
            authorization_transport=consuming_transport,
            tool_transport=self.tool_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            authorization_call_id=authorization_call_id,
            call_authorization_token=token,
            config=config,
            from_number_id=from_number_id,
            to_e164=to_e164,
        )

    def resume_authorized_outbound(
        self,
        *,
        call_id: str,
        agent_id: str,
        from_number_id: str,
        to_e164: str,
    ) -> Gate2CallSession:
        """Resume a call whose single-use authorization was consumed by the dialer.

        The production agent receives only identifiers in LiveKit dispatch
        metadata. The control plane verifies that this gateway consumed the
        call-bound authorization before returning the effective per-call
        realtime config.
        """

        config = self.transport.request("GET", f"/v1/calls/{call_id}/realtime-config")
        _validate_realtime_config(config, expected_agent_id=agent_id)
        if config.get("call_id") != call_id or config.get("direction") != "outbound":
            raise ControlPlaneRejected("call realtime config is not bound to this outbound call")
        return Gate2CallSession(
            transport=self.transport,
            authorization_transport=self.transport,
            tool_transport=self.tool_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            authorization_call_id=call_id,
            call_authorization_token="",
            config=config,
            from_number_id=from_number_id,
            to_e164=to_e164,
            media_admitted=True,
            audit_events=[
                "authorization_consumed_by_originator",
                "realtime_config_loaded",
                "media_admitted",
            ],
        )

    def prepare_inbound(
        self,
        *,
        organization_id: str,
        to_e164: str,
        from_e164: str,
        provider_call_id: str,
    ) -> Gate2CallSession:
        """Resolve number policy and the authoritative agent before subscribing.

        LiveKit SIP may have created the room, but the agent must not connect or
        consume its audio until this route/policy call returns an allowed route.
        """

        routed = self.call_transport.request(
            "POST",
            "/v1/calls/inbound/route",
            {
                "organization_id": organization_id,
                "to_phone_number": to_e164,
                "from_phone_number": from_e164,
                "provider": "didww",
                "provider_call_id": provider_call_id,
                "live_media": True,
                "media_pipeline": "realtime",
            },
        )
        route = routed.get("route")
        if (
            not isinstance(route, dict)
            or route.get("routed") is not True
            or routed.get("status") == "live_blocked"
        ):
            raise ControlPlaneRejected("inbound number route or live-call policy was denied")
        call_id = routed.get("id")
        agent_id = route.get("agent_id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(agent_id, str)
            or not agent_id
        ):
            raise ControlPlaneRejected("inbound route omitted the call or agent identity")
        config = self.transport.request("GET", f"/v1/agents/{agent_id}/realtime-config")
        _validate_realtime_config(config, expected_agent_id=agent_id)
        return Gate2CallSession(
            transport=self.transport,
            authorization_transport=self.transport,
            tool_transport=self.tool_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            authorization_call_id="",
            call_authorization_token="",
            config=config,
        )


def _validate_realtime_config(config: dict[str, Any], *, expected_agent_id: str) -> None:
    required_strings = ("agent_id", "organization_id", "project_id", "language")
    missing = [
        name
        for name in required_strings
        if not isinstance(config.get(name), str) or not config.get(name)
    ]
    if missing:
        raise ControlPlaneRejected(f"realtime config omitted required fields: {', '.join(missing)}")
    if config["agent_id"] != expected_agent_id:
        raise ControlPlaneRejected("realtime config agent id does not match the requested agent")
    for component in ("transcriber", "model", "voice"):
        value = config.get(component)
        if (
            not isinstance(value, dict)
            or not value.get("registry_name")
            or not value.get("model")
        ):
            raise ControlPlaneRejected(f"realtime config has no runnable {component}")
    tools = config.get("tools")
    if not isinstance(tools, list) or any(
        not isinstance(item, str) or not item for item in tools
    ):
        raise ControlPlaneRejected("realtime config tools must be a list of non-empty names")
    if config.get("recording_enabled") is not True:
        raise ControlPlaneRejected("Gate 2 requires caller-side recording to be enabled")


def _validate_pr79_artifact_set(root: Path) -> None:
    required = ("caller.wav", "agent.wav", "mixed.wav", "transcript.json")
    missing = [
        name
        for name in required
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]
    caller_turns = {
        path.name.removesuffix("-caller.wav")
        for path in root.glob("turn-*-caller.wav")
        if path.stat().st_size
    }
    agent_turns = {
        path.name.removesuffix("-agent.wav")
        for path in root.glob("turn-*-agent.wav")
        if path.stat().st_size
    }
    if caller_turns != agent_turns or not caller_turns:
        missing.append("one-to-one turn-*-caller.wav and turn-*-agent.wav")
    transcript = root / "transcript.json"
    if transcript.is_file():
        try:
            value = json.loads(transcript.read_text())
            if not isinstance(value, dict) or not isinstance(value.get("turns"), list) or not value["turns"]:
                missing.append("transcript.json with at least one turn")
            elif not valid_semantic_response_audio_policy(
                value.get("semantic_response_audio_policy")
            ):
                missing.append("transcript.json with the PR #79 no-filler policy")
            elif value.get("openphonex") and not valid_effective_voice_stack_identity(
                value.get("effective_voice_stack")
            ):
                missing.append("transcript.json with a valid effective voice stack identity")
        except (OSError, ValueError):
            missing.append("valid transcript.json")
    for path in [root / name for name in required[:3]] + sorted(root.glob("turn-*.wav")):
        if path.is_file() and not _is_pcm_s16le_8000_mono(path):
            missing.append(f"{path.name} pcm_s16le_8000_mono")
    if missing:
        raise IncompleteArtifactEvidence("missing PR #79 evidence: " + ", ".join(sorted(set(missing))))


def _duration_provenance(sources: dict[str, float]) -> dict[str, Any]:
    allowed = {"sip_participant", "room_track", "recording"}
    normalized: dict[str, float] = {}
    for name, value in sources.items():
        if name not in allowed:
            raise ControlPlaneRejected(f"unsupported billing duration source: {name}")
        try:
            duration = float(value)
        except (TypeError, ValueError) as exc:
            raise ControlPlaneRejected(f"invalid billing duration source: {name}") from exc
        if duration < 0:
            raise ControlPlaneRejected(f"negative billing duration source: {name}")
        normalized[name] = round(duration, 3)
    if "sip_participant" not in normalized or "recording" not in normalized:
        raise ControlPlaneRejected("billing provenance requires SIP participant and recording durations")
    source = max(normalized, key=normalized.__getitem__)
    return {
        "billing_duration_seconds": normalized[source],
        "billing_duration_source": source,
        "duration_sources": normalized,
    }


def _is_pcm_s16le_8000_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as recording:
            return (
                recording.getnchannels() == 1
                and recording.getsampwidth() == 2
                and recording.getframerate() == 8_000
                and recording.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def _upload_part(filename: str, content: bytes) -> dict[str, str]:
    return {
        "filename": filename,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "sha256": hashlib.sha256(content).hexdigest(),
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""Portable Skysay control-plane boundary for the LiveKit candidate.

The media runtime is LiveKit SIP + LiveKit Agents.  This module
keeps the contracts which must remain outside model-generated input: call
authorization, resolved agent configuration, tool authorization, evidence
import, billing-duration provenance, and idempotent media cleanup.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import ipaddress
import json
import os
import re
import select
import threading
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
import wave
import time
import weakref
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

try:
    from .workflow_requests import WorkflowTurnRequests
except ImportError:  # the image imports required sources as flat modules
    from workflow_requests import WorkflowTurnRequests

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

if __package__:
    from .tool_declarations import (
        BUILTIN_TOOL_DECLARATIONS,
        KNOWLEDGE_TOOL_DECLARATION,
        builtin_tool_declaration,
        implied_tool_names,
    )
else:  # the image imports the same required source as a flat module
    from tool_declarations import (
        BUILTIN_TOOL_DECLARATIONS,
        KNOWLEDGE_TOOL_DECLARATION,
        builtin_tool_declaration,
        implied_tool_names,
    )


# The control plane mints this leg's correlation id and returns it in the
# release route response; the release host turns it into a ``<call_id>.json``
# filename on its session volume. Adopt it only when it is a safe filename
# component so a compromised or malformed control-plane response can never name
# a path separator or a traversal segment.
_RELEASE_CANARY_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


#: The ``purpose`` a release-canary session stamps into its session-state
#: file. Matches the canonical route purpose the platform admits the canary
#: under, so the two names cannot drift apart silently.
CANARY_SESSION_PURPOSE = "platform_release_canary"


class ControlPlaneRejected(RuntimeError):
    """The authoritative control plane rejected or malformed a request.

    ``status_code`` carries the HTTP status when the rejection came from an
    HTTP response, and stays ``None`` when it came from a malformed body or a
    client-side contract check. A caller that has to tell "this can never
    succeed" (404: the control plane has no such call) from "try again later"
    (502, a timeout) reads this attribute. Parsing it back out of the message
    text is what the reconciler used to have no way to avoid, and a sweep that
    cannot see the difference retries a permanently missing call forever.
    """

    def __init__(
        self,
        *args: Any,
        status_code: int | None = None,
        reason: str = "",
        route_missing: bool = False,
        error_code: str = "",
        request_path: str = "",
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        #: The API's machine-readable refusal code, when its body carried one.
        self.reason = reason
        #: True only for the router's own unknown-path 404, which is how an
        #: API that predates a route answers it -- never for a by-id 404.
        self.route_missing = route_missing
        #: The body's ``error`` field: the API's error class or code. Separate
        #: from ``reason``, which a body that names its own rule (a blocker, a
        #: call-variables code) fills with that rule instead.
        self.error_code = error_code
        #: The path of the request the control plane answered, query dropped.
        #: Where a refusal came FROM is a fact of the request, never of the
        #: worker's stage: one stage can make two requests.
        self.request_path = request_path


class BrowserSessionAlreadyOwned(ControlPlaneRejected):
    """Another media worker already owns this browser session."""


class SessionStateError(RuntimeError):
    """A media action was attempted outside the permitted session state."""


class KeypadValueUnavailable(RuntimeError):
    """The control plane did not release a keypad value.

    Carries the API's refusal code and nothing else: the message is fixed, the
    request's field and any response body are not kept, so no log or traceback
    that prints this can carry either. The controller ends the call
    ``ivr_timeout/data_not_authorised``; the model is never told.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__("keypad value not released")
        self.reason = reason if re.fullmatch(r"[A-Za-z0-9_]{1,64}", reason or "") else ""


class IncompleteArtifactEvidence(RuntimeError):
    """The caller-side PR #79 artifact set is incomplete."""


class ArtifactUploadFailed(ControlPlaneRejected):
    """The call's evidence bundle could not be stored, after bounded retries.

    Carries only fixed-vocabulary diagnosis -- the step, the API's reason code
    or the transport's classification, the HTTP status and the bundle size --
    so the worker log, the error report and the call's own timeline can all
    say WHY without repeating a response body.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        reason: str,
        status_code: int | None = None,
        total_bytes: int = 0,
        upload_id: str = "",
    ) -> None:
        # The failure record accepts snake_case codes only; an older API
        # names its refusal by exception class (``PayloadTooLargeError``).
        reason = re.sub(r"(?<!^)(?=[A-Z])", "_", reason).lower()
        super().__init__(message, status_code=status_code, reason=reason)
        self.stage = stage
        self.total_bytes = total_bytes
        self.upload_id = upload_id

    @property
    def retryable(self) -> bool:
        """Whether trying the same bundle again later can succeed.

        No HTTP status means the request never got an answer (the network, a
        timeout, an API that could not take it yet), and a 5xx/429 is the API
        saying "not now". Every other refusal -- a cap, a credential, a call
        the API does not have -- answers the same way next time.
        """
        if self.reason in _RETRYABLE_UPLOAD_REASONS:
            return True
        if self.status_code is None:
            return self.reason != "malformed_response"
        return self.status_code in _RETRYABLE_UPLOAD_STATUSES


# Every terminal-call-lifecycle ``billing.status`` the control plane can
# legitimately return once it has produced an AUTHORITATIVE money verdict for
# the call. ``no_reservation``/``settled``/``released`` are self-explanatory;
# ``payment_due`` has already captured what could be captured and left a
# collectible remainder (still authoritative, never re-released); and
# ``awaiting_usage`` is a deliberate hold, not a failure — sip.ended can race
# a hosted-AI-component worker's post-media native usage report, and the
# control plane retains the reservation rather than estimate a price (see
# ``hosted_ai_component_usage.py::_await_component_usage``). A caller that
# reports a terminal lifecycle event (this class, or the standalone
# reconciler in ``reconciliation.py``) must accept every one of these as
# "done", not just the subset that happens to be raised in a given call's
# history — the two callers drifting out of sync once let the reconciler
# treat an authoritative ``awaiting_usage`` hold as a rejection and retry
# the same crashed session forever.
AUTHORITATIVE_TERMINAL_BILLING_STATUSES = frozenset(
    {"settled", "released", "payment_due", "awaiting_usage", "no_reservation"}
)


_COMPONENT_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COMPONENT_TOOL_NAME_LIMIT = 32
#: A function name every engine's provider accepts; anything else would
#: refuse the whole model session, so such an entry is left out instead.
_MODEL_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def _model_tool_definition(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """A shared ``{name, description, parameters}`` declaration in the
    worker's ``{name, description, inputSchema}`` shape."""
    return {
        "name": declaration["name"],
        "description": declaration["description"],
        "inputSchema": declaration["parameters"],
    }


#: Deeper than any real tool schema; a deeper one is left out before anything
#: walks it recursively (a copy, a provider's serializer).
_MAX_TOOL_SCHEMA_DEPTH = 32


def _schema_too_deep(schema: Any) -> bool:
    """Whether ``schema`` nests objects or lists deeper than the bound,
    measured without recursion."""
    stack = [(schema, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_TOOL_SCHEMA_DEPTH:
            return True
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        stack.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return False


def _tool_label(name: Any) -> str:
    """A tool's name as an omission event may carry it: a valid function
    name, or nothing, never free text from the entry."""
    text = str(name or "").strip() if isinstance(name, str) else ""
    return text if _MODEL_TOOL_NAME_RE.fullmatch(text) else ""


def _safe_argument_character_count(arguments: Any) -> int:
    """Count model input without retaining or serializing its contents."""
    try:
        rendered = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return 0
    return min(16_384, len(rendered))


def _telemetry_outcome(result: Any) -> str:
    if not isinstance(result, dict):
        return "error"
    if result.get("ok") is False:
        return "error"
    outcome = str(result.get("outcome") or "").strip().lower()
    if outcome in {"empty", "timeout", "cancelled", "error", "fallback"}:
        return outcome
    return "success"


#: What a model on a ``model_safe_results`` call hears when the control plane
#: could not give a well-formed answer. No cause: an exception name, a status
#: code or a body is nothing an anonymous listener should be read.
MODEL_SAFE_UNAVAILABLE_ERROR = (
    "That tool is unavailable right now. Tell the visitor and help them another way."
)
_KNOWLEDGE_RESULT_KEYS = frozenset({"tool", "outcome", "passages", "source_titles", "duration_ms"})


def model_safe_unavailable(name: str) -> dict[str, Any]:
    return {"ok": False, "tool": name, "error": MODEL_SAFE_UNAVAILABLE_ERROR, "reason_code": "tool_unavailable"}


def is_model_safe_tool_result(result: Any, name: str) -> bool:
    """Exactly the shapes the control plane's model-safe surface returns.

    A tool: ``{ok: true, tool, result: object}`` or ``{ok: false, tool, error,
    reason_code}`` (non-empty strings), each with an optional string
    ``telemetry_correlation_id``. A knowledge search: every key of the
    retrieval answer (``outcome``, ``passages`` of string ``title``/``text``,
    string ``source_titles``, a non-negative integer ``duration_ms``) and the
    optional correlation. Anything else -- a missing key,
    another tool's answer, a mistyped field, an extra key -- is not
    a result handed to the model.
    """
    if not isinstance(result, dict) or result.get("tool") != name:
        return False
    if not isinstance(result.get("telemetry_correlation_id", ""), str):
        return False
    if name == Gate2CallSession._KNOWLEDGE_TOOL_NAME and "ok" not in result:
        passages = result.get("passages")
        return (
            set(result) - {"telemetry_correlation_id"} == _KNOWLEDGE_RESULT_KEYS
            and isinstance(result.get("outcome"), str)
            and bool(result["outcome"])
            and isinstance(passages, list)
            and all(
                isinstance(p, dict) and set(p) == {"title", "text"}
                and isinstance(p["title"], str) and isinstance(p["text"], str)
                for p in passages
            )
            and type(result["duration_ms"]) is int
            and result["duration_ms"] >= 0
            and isinstance(result["source_titles"], list)
            and all(isinstance(t, str) for t in result["source_titles"])
        )
    keys = set(result) - {"telemetry_correlation_id"}
    if result.get("ok") is True:
        return keys == {"ok", "tool", "result"} and isinstance(result["result"], dict)
    if result.get("ok") is False:
        return (
            keys == {"ok", "tool", "error", "reason_code"}
            and isinstance(result["error"], str)
            and bool(result["error"])
            and isinstance(result["reason_code"], str)
            and bool(result["reason_code"])
        )
    return False


@dataclass(frozen=True)
class OutboundDispatch:
    call_id: str
    agent_id: str
    from_number_id: str
    from_e164: str
    to_e164: str
    participant_identity: str


@dataclass(frozen=True)
class WorkspaceBrowserTestDispatch:
    call_id: str
    agent_id: str
    organization_id: str
    project_id: str
    participant_identity: str
    snapshot: str
    # Explicitly dispatched LiveKit agent workers do not join with a caller-
    # supplied participant identity. LiveKit assigns ``agent-<job id>`` and
    # authenticates their configured name in ``lk.agent.name`` instead. This is
    # empty for ordinary browser participants, which keep the exact identity
    # contract above.
    participant_agent_name: str = ""


def parse_workspace_browser_test_dispatch(metadata: str) -> WorkspaceBrowserTestDispatch | None:
    """Parse tenant-bound workspace browser-test dispatch metadata."""

    if not metadata:
        return None
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("LiveKit dispatch metadata must be valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "skysay.workspace-browser-test.v1":
        return None
    fields = {
        name: str(payload.get(name) or "").strip()
        for name in (
            "call_id",
            "agent_id",
            "organization_id",
            "project_id",
            "participant_identity",
        )
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ValueError(f"workspace browser-test dispatch omitted required fields: {', '.join(missing)}")
    snapshot = str(payload.get("snapshot") or "published").strip() or "published"
    if snapshot not in {"published", "draft"}:
        raise ValueError("workspace browser-test snapshot must be published or draft")
    participant_agent_name = str(payload.get("participant_agent_name") or "").strip()
    return WorkspaceBrowserTestDispatch(
        **fields,
        snapshot=snapshot,
        participant_agent_name=participant_agent_name,
    )


def linked_participant_kinds(
    browser_test: WorkspaceBrowserTestDispatch | None,
    *,
    agent_kind: Any,
    sdk_default: Any,
) -> Any:
    """``RoomOptions.participant_kinds`` for the participant a session links.

    The pinned ``RoomIO`` checks the linked participant's kind even when its
    identity is exact, and its default kinds (connector, SIP, standard) leave
    out AGENT. A synthetic caller is an agent job, so without this the session
    waits forever for a participant already in the room. Only a browser-test
    dispatch that names an agent caller -- whose ``lk.agent.name`` the worker
    has already matched -- takes the AGENT kind; the exact identity still
    decides which agent. Every other call keeps the SDK default. The SDK values
    are passed in so this module stays importable without livekit.
    """

    if browser_test is not None and browser_test.participant_agent_name:
        return [agent_kind]
    return sdk_default


def arms_late_caller_publications(
    browser_test: WorkspaceBrowserTestDispatch | None,
) -> bool:
    """Whether the worker subscribes caller audio published after its sweep.

    The worker connects with ``SUBSCRIBE_NONE``, so a caller track published
    after the worker-entry sweep stays unsubscribed unless this is true. A phone
    call's track can arrive after the sweep. So can a synthetic agent caller's:
    the worker matches it on ``lk.agent.name`` as soon as it joins, and it
    publishes its microphone only after connecting. A human browser test keeps
    the sweep alone, as before.
    """

    return browser_test is None or bool(browser_test.participant_agent_name)


def parse_outbound_dispatch(metadata: str) -> OutboundDispatch | None:
    """Parse non-secret LiveKit dispatch metadata for an outbound call."""

    if not metadata:
        return None
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("LiveKit dispatch metadata must be valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "skysay.outbound-dispatch.v1":
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

    Production is unaffected (it uses https://api.skysay.ai or the
    host-local http://127.0.0.1 control plane); the guard exists for
    self-hosters who would otherwise leak the worker/gateway tokens in
    plaintext by pointing SKYSAY_CONTROL_PLANE_URL at a remote http URL.
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


#: Idle keep-alive connections one transport keeps per origin, and how long
#: one may sit idle before it is closed instead of reused. Measured from the
#: voice host on 2026-10-26, api.skysay.ai (Cloudflare) still served a
#: connection idle for 120 s; 30 s stays far inside that.
KEEPALIVE_IDLE_CONNECTIONS = 4
KEEPALIVE_IDLE_SECONDS = 30.0


class _IdleConnections:
    """A transport's idle keep-alive connections, per origin. Thread-safe.

    A connection is returned only after a whole 2xx body was read on it and
    the server did not ask to close it. One is lent again only while it is
    younger than ``KEEPALIVE_IDLE_SECONDS`` and its socket has nothing to
    read: an idle connection whose close (or any write) has already reached
    this host is readable, so it is discarded before any request is written
    to it. A close still in flight when the request is written cannot be
    seen; that request fails like any connection failure and is never sent
    again. Idle connections are kept far shorter than the edge keeps them,
    so that race needs an abnormal close.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._idle: dict[tuple[Any, str], list[tuple[Any, float]]] = {}
        self._clock = clock

    def lend(self, key: tuple[Any, str]) -> Any | None:
        while True:
            with self._lock:
                entries = self._idle.get(key)
                if not entries:
                    return None
                connection, idle_since = entries.pop()
            if self._clock() - idle_since < KEEPALIVE_IDLE_SECONDS and self._quiet(connection):
                return connection
            connection.close()

    @staticmethod
    def _quiet(connection: Any) -> bool:
        sock = connection.sock
        if sock is None:
            return False
        try:
            pending = getattr(sock, "pending", None)
            if pending is not None and pending():
                return False
            return not select.select([sock], [], [], 0)[0]
        except (OSError, ValueError):
            return False

    def give_back(self, key: tuple[Any, str], connection: Any) -> None:
        with self._lock:
            entries = self._idle.setdefault(key, [])
            if len(entries) < KEEPALIVE_IDLE_CONNECTIONS:
                entries.append((connection, self._clock()))
                return
        connection.close()

    def close(self) -> None:
        with self._lock:
            entries = [connection for idle in self._idle.values() for connection, _ in idle]
            self._idle.clear()
        for connection in entries:
            connection.close()


class _KeepAliveMixin:
    """urllib's ``do_open`` without ``Connection: close`` for a direct 2xx.

    Everything a request carries is what urllib builds; only the connection
    outlives it. A proxied or tunnelled request, and any response that is not
    a 2xx (an error, whose body the caller reads bounded, or a redirect the
    same-origin handler judges), take urllib's own one-shot path or close the
    connection exactly as urllib does. A request is never retried: a write or
    read that fails on a reused connection fails the request, as it would on
    a fresh one, so no command can apply twice.
    """

    _connections: _IdleConnections

    def do_open(self, http_class, req, **http_conn_args):  # type: ignore[override]
        if req.has_proxy() or req._tunnel_host or not req.host:
            return super().do_open(http_class, req, **http_conn_args)
        key = (http_class, req.host)
        connection = self._connections.lend(key)
        if connection is None:
            connection = http_class(req.host, timeout=req.timeout, **http_conn_args)
        elif isinstance(req.timeout, (int, float)):
            connection.timeout = req.timeout
            connection.sock.settimeout(req.timeout)
        connection.set_debuglevel(self._debuglevel)
        headers = dict(req.unredirected_hdrs)
        headers.update({k: v for k, v in req.headers.items() if k not in headers})
        headers = {name.title(): value for name, value in headers.items()}
        try:
            try:
                connection.request(req.get_method(), req.selector, req.data, headers,
                                   encode_chunked=req.has_header("Transfer-encoding"))
            except OSError as err:
                raise urllib.error.URLError(err)
            response = connection.getresponse()
        except BaseException:
            connection.close()
            raise
        if not 200 <= response.status < 300:
            # urllib's own shape: the connection closes when this response does.
            if connection.sock:
                connection.sock.close()
                connection.sock = None
            response.url = req.get_full_url()
            response.msg = response.reason
            return response
        try:
            body = response.read()
        except BaseException:
            # A response that will close was detached from its connection by
            # getresponse(): close it too, or its socket outlives the error.
            response.close()
            connection.close()
            raise
        if response.will_close:
            connection.close()
        else:
            self._connections.give_back(key, connection)
        result = urllib.response.addinfourl(io.BytesIO(body), response.headers, req.get_full_url(), response.status)
        result.msg = response.reason
        return result


class _KeepAliveHTTPHandler(_KeepAliveMixin, urllib.request.HTTPHandler):
    def __init__(self, connections: _IdleConnections) -> None:
        super().__init__()
        self._connections = connections


class _KeepAliveHTTPSHandler(_KeepAliveMixin, urllib.request.HTTPSHandler):
    def __init__(self, connections: _IdleConnections) -> None:
        super().__init__()
        self._connections = connections


#: The worker contract this worker speaks, presented on every control-plane
#: request and on every queue claim. The control plane admits a call with a
#: schema-4 call-variables snapshot and protocol marker only to a worker
#: presenting this protocol, and
#: hands such a worker nothing else; there is no compatibility path. The
#: worker image ships this directory on its own, so this is a second copy of
#: the control plane's worker-contract constants, pinned equal by the
#: control plane's own suite rather than imported.
WORKER_CONTRACT = 7
WORKER_CONTRACT_HEADER = "X-Skysay-Worker-Contract"
#: Where this deployment keeps call content, read once at startup
#: (``adopt_deployment_content_storage``) and presented on every request, so
#: the control plane can refuse a worker whose mode is stale. Another copy of a
#: control-plane constant (services/gateway_content_storage.py), pinned equal by
#: tests/test_content_local_worker.py.
CONTENT_STORAGE_ENV = "SKYSAY_CONTENT_STORAGE"
CONTENT_STORAGE_HEADER = "X-Skysay-Content-Storage"
CONTENT_STORAGE_GATEWAY = "gateway"
CONTENT_STORAGE_MODES = frozenset({"skysay", CONTENT_STORAGE_GATEWAY})
ARTIFACT_RETAINED_ON_GATEWAY = "retained_on_gateway"


def content_kept_on_gateway(environ: Mapping[str, str] | None = None) -> bool:
    return str((os.environ if environ is None else environ).get(CONTENT_STORAGE_ENV) or "") == CONTENT_STORAGE_GATEWAY


def adopt_deployment_content_storage(environ: Any, *, transport: "JsonTransport | None" = None) -> str:
    """Read this self-hosted deployment's content storage once, at startup.

    The managed worker (it holds the platform media token) has no deployment of
    its own and reads nothing. A self-hosted worker records the mode in its
    environment, where every job process and every request inherits it. A
    content-local deployment must not run with an organization key: such a key
    cannot say which deployment it serves.
    """
    if str(environ.get("SKYSAY_PLATFORM_MEDIA_TOKEN") or "").strip():
        return ""
    gateway_id = str(environ.get("SKYSAY_GATEWAY_ID") or "").strip()
    token = str(environ.get("SKYSAY_GATEWAY_API_TOKEN") or "").strip()
    base_url = str(environ.get("SKYSAY_CONTROL_PLANE_URL") or "").strip()
    if not gateway_id or not (transport or (token and base_url)):
        return ""
    reader = transport or UrllibJsonTransport(base_url=base_url, bearer_token=token)
    deployment = reader.request("GET", f"/v1/gateway-deployments/{urllib.parse.quote(gateway_id, safe='')}/content-storage")
    mode = str(deployment.get("content_storage") or "skysay")
    if mode not in CONTENT_STORAGE_MODES:
        raise ControlPlaneRejected("the control plane named an unknown content storage mode")
    if mode == CONTENT_STORAGE_GATEWAY and str(environ.get("SKYSAY_WORKER_API_TOKEN") or "").strip():
        raise SystemExit(
            "this deployment keeps call content on the gateway: remove SKYSAY_WORKER_API_TOKEN from "
            ".env.livekit (content-local runs on the deployment's gateway token alone)"
        )
    environ[CONTENT_STORAGE_ENV] = mode
    return mode


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
        # Connections are reused across requests (keep-alive): a workflow line
        # makes several control-plane requests back to back, and each fresh
        # connection costs a TCP and TLS handshake (~20 ms from the voice host).
        self._connections = _IdleConnections()
        # Idle sockets close with the transport, not whenever they are collected.
        weakref.finalize(self, self._connections.close)
        handlers: list[urllib.request.BaseHandler] = [
            _SameOriginRedirectHandler(),
            _KeepAliveHTTPHandler(self._connections),
            _KeepAliveHTTPSHandler(self._connections),
        ]
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
        return self._send(
            method,
            path,
            body,
            content_type="application/json" if body is not None else "",
            timeout_seconds=self.timeout_seconds,
        )

    def request_bytes(
        self,
        method: str,
        path: str,
        body: bytes,
        *,
        content_type: str,
        timeout_seconds: float,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a raw body with a timeout sized by the caller to that body.

        The class-wide timeout suits small JSON calls; an artifact chunk or a
        finalize that imports a long call's recordings needs its own.
        """
        return self._send(
            method,
            path,
            body,
            content_type=content_type,
            timeout_seconds=timeout_seconds,
            headers=headers,
        )

    def _send(
        self,
        method: str,
        path: str,
        body: bytes | None,
        *,
        content_type: str,
        timeout_seconds: float,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, data=body, method=method)
        request.add_header("Accept", "application/json")
        for name, value in self._credential_headers().items():
            request.add_header(name, value)
        if body is not None and content_type:
            request.add_header("Content-Type", content_type)
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        # Every request the worker makes presents its contract; the control
        # plane reads it where it matters (config delivery) and ignores it
        # elsewhere. Added LAST: ``add_header`` keeps the latest value for a
        # name, so a caller-supplied header of the same name cannot lower it.
        request.add_header(WORKER_CONTRACT_HEADER, str(WORKER_CONTRACT))
        content_storage = str(os.environ.get(CONTENT_STORAGE_ENV) or "")
        if content_storage:
            request.add_header(CONTENT_STORAGE_HEADER, content_storage)
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            # The codes are read from the WHOLE body, bounded; only the
            # diagnostic text is cut. A not-runnable refusal carries its full
            # blocker list and a suspension its full sentence, and either runs
            # past 500 characters, which used to cost the code itself.
            body = exc.read(MAX_ERROR_BODY_BYTES).decode(errors="replace")
            reason, route_missing = _refusal_fields(exc.code, body)
            raise ControlPlaneRejected(
                f"control plane returned HTTP {exc.code}: {body[:500]}",
                status_code=exc.code,
                reason=reason,
                route_missing=route_missing,
                error_code=_refusal_error_code(body),
                request_path=urllib.parse.urlsplit(path).path,
            ) from exc
        if not isinstance(decoded, dict):
            raise ControlPlaneRejected("control plane returned a non-object response")
        return decoded


    def _credential_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"}


class InternalTokenJsonTransport(UrllibJsonTransport):
    """The homepage demo worker's client for its call's shared call routes.

    That worker holds the internal token and nothing else (``voice_role_guard``
    refuses every call credential on its host). The control plane accepts it
    on exactly the call-keyed routes a homepage demo call shares with a phone
    call -- tools, knowledge search, artifact upload -- and binds it there to a
    homepage demo call. The worker's host/epoch rides along for the placement
    fence, as on its profile fetch.
    """

    def __init__(
        self, *, base_url: str, internal_token: str, placement: str = "", timeout_seconds: float = 10.0
    ) -> None:
        super().__init__(base_url=base_url, bearer_token="", timeout_seconds=timeout_seconds)
        self._internal_token = internal_token
        self._placement = placement

    def _credential_headers(self) -> dict[str, str]:
        headers = {"X-Skysay-Internal-Token": self._internal_token}
        if self._placement:
            headers["X-Skysay-Voice-Placement"] = self._placement
        return headers


#: How much of an error body is read to find its codes. Far above any body the
#: API builds for a refusal; a bound so a misbehaving peer cannot make the
#: worker buffer an unbounded answer.
MAX_ERROR_BODY_BYTES = 64 * 1024

_REFUSAL_CODE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _refusal_error_code(detail: str) -> str:
    """The body's ``error`` field, bounded like ``reason``; ``""`` otherwise."""
    try:
        decoded = json.loads(detail)
    except ValueError:
        return ""
    code = str(decoded.get("error") or "") if isinstance(decoded, dict) else ""
    return code if _REFUSAL_CODE.fullmatch(code) else ""


#: The control plane's DESIGNED refusals of a call that it also RECORDS: the
#: workspace's or the platform's state decided against the call, the route
#: wrote the refused call's own terminal row, and nothing about the worker is
#: wrong. Keyed on the API's own wire form -- ``(HTTP status, body "error")`` --
#: and pinned to the control plane's classes by the worker-entry suite; this
#: image cannot import them.
#:
#: Each of these is raised by the inbound route only AFTER its carrier-key
#: insert (the replay gate), inside the admission transaction, so it can never
#: be the answer to a redelivery of a call that was already admitted; and
#: ``inbound_attempts.record_rejected_attempts`` records each as the call's
#: ``failed`` row. The maintenance window is the one exception to the first
#: half: it refuses before the gate, and was quiet before this table existed.
#:
#: Deliberately NOT here, so they still raise and page: no status (a check this
#: client made itself), 401; a plain 403 ``ForbiddenError`` (a credential's
#: scope looks the same on the wire as the plan's concurrency limit and as the
#: replay of an already-rejected attempt); 423 ``SuspensionError`` and 400
#: ``LanguageNotPublishedError`` (refused BEFORE the replay gate and not
#: recorded, so one can be the answer about a call admitted earlier); 404, 409
#: (a worker-contract refusal among them), 408, generic 429/5xx and any other 400.
#: ``KycProviderCreditsExhaustedError`` is a 402 too, but no route this worker
#: calls can return it.
DESIGNED_REFUSALS = frozenset(
    {
        (402, "InsufficientBalanceError"),  # the wallet could not cover the reservation
        (429, "realtime_capacity_unavailable"),  # recorded provider-capacity admission refusal
        (400, "realtime_config_not_runnable"),  # the published agent cannot run
        (503, "platform_maintenance"),  # an operator's maintenance window
    }
)

#: ``CallVariablesRefused`` is sent as a plain ``BadRequestError`` naming its
#: rule in ``reason``; only these reasons make that 400 a designed refusal.
#: A copy of ``call_variables_admission.REFUSAL_CODES``, pinned equal.
CALL_VARIABLES_REFUSAL_CODES = frozenset(
    {
        "agent_not_published",
        "call_variables_invalid",
        "call_variables_missing",
        "call_variables_over_budget",
        "keypad_value_in_text",
        "keypad_value_not_dialable",
        "per_call_text_on_workflow",
        "schema_pair_mismatch",
        "workflow_binding_failed",
    }
)

#: The release route's refusal of the reserved tuple outside a release window.
#: Designed on THAT leg only.
RELEASE_CANARY_NOT_ARMED = "release_canary_not_armed"

#: The routes that refuse a call BEFORE admitting one and record the refusal
#: themselves (the inbound route writes the failed call row and rolls the
#: reservation back; the release route creates no customer record at all). A
#: refusal from any other request was made after something was admitted.
ROUTES_THAT_REFUSE_BEFORE_ADMISSION = frozenset(
    {"/v1/calls/inbound/route", "/v1/platform-media/release-canary/route"}
)


def designed_refusal(error: BaseException, *, release_canary_leg: bool = False) -> bool:
    """Whether ``error`` is the control plane refusing the call by design."""
    if not isinstance(error, ControlPlaneRejected) or error.status_code is None:
        return False
    key = (error.status_code, error.error_code)
    if key in DESIGNED_REFUSALS:
        return True
    if key == (400, "BadRequestError"):
        return error.reason in CALL_VARIABLES_REFUSAL_CODES
    return release_canary_leg and key == (403, RELEASE_CANARY_NOT_ARMED)


def _refusal_fields(status_code: int, detail: str) -> tuple[str, bool]:
    """``(reason code, route missing)`` read from an error body, never raising."""
    try:
        decoded = json.loads(detail)
    except ValueError:
        return "", False
    if not isinstance(decoded, dict):
        return "", False
    reason = str(decoded.get("reason") or decoded.get("error") or "")
    reason = reason if re.fullmatch(r"[A-Za-z0-9_]{1,64}", reason) else ""
    route_missing = (
        status_code == 404 and decoded.get("error") == "not_found" and "path" in decoded
    )
    return reason, route_missing


def call_credential_token(environ: Mapping[str, str]) -> str:
    """Bearer token for the worker's call-bound control-plane requests.

    A deployment now needs ONE runtime credential. Its deployment-bound
    gateway token performs inbound routing, reads the resulting call's
    realtime config, and lists MCP tool schemas -- all bound server-side to
    the organization that owns the deployment, so a compromised host reaches
    that deployment's workspace and no wider surface.

    An installation that still sets an organization-scoped worker key keeps
    using it unchanged: this is additive, not a migration. The optional
    outbound-dial profile still needs that separate key, because claiming
    worker jobs is organization authority a gateway token does not carry.
    """
    worker_token = str(environ.get("SKYSAY_WORKER_API_TOKEN") or "").strip()
    if worker_token:
        return worker_token
    gateway_token = str(environ.get("SKYSAY_GATEWAY_API_TOKEN") or "").strip()
    if gateway_token:
        return gateway_token
    raise KeyError("SKYSAY_GATEWAY_API_TOKEN")


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
    # The transport the credential lease rides. It is NOT the call transport:
    # a deployment still carrying a legacy organization-scoped worker key
    # (queue outbound mode) sends every call-bound request with that key, and
    # the lease route admits only the managed media worker or a GATEWAY token.
    # Sending it the org key would 403 every call pre-media. None means "use
    # the ordinary transport", which is correct on cloud, where the platform
    # media token serves both.
    credential_transport: JsonTransport | None = None
    # True on Skysay's own managed media workers, which already hold the
    # platform provider keys. False on a customer's self-hosted box, which
    # holds none of them and therefore leases platform-managed components too.
    # Defaults to False so a new construction site fails toward asking, not
    # toward assuming keys are present that are not.
    platform_media_worker: bool = False
    browser_session_id: str = ""
    browser_outcome: dict[str, str] | None = None
    browser_outcome_finalized: bool = False
    media_admitted: bool = False
    audit_events: list[str] = field(
        default_factory=lambda: ["authorization_allowed", "realtime_config_loaded"]
    )
    _cleaned: bool = False
    _completed: bool = False
    _lifecycle_finalized: bool = False
    _tool_sequence: int = 0
    _workflow_requests: WorkflowTurnRequests = field(default_factory=WorkflowTurnRequests, repr=False)
    # Only populated after all bounded separate-report retries fail. It is
    # carried in the next terminal lifecycle POST so the control plane can
    # atomically persist native usage before it settles that terminal event.
    # This is deliberately a short-lived fallback, not a durable worker queue.
    _pending_component_usage: dict[str, Any] | None = field(default=None, repr=False)
    usage_report_outbox: Any = field(default=None, repr=False)
    # Assigned by the media worker after it creates its per-call bounded
    # collector.  The observer sees counts and opaque audit correlation only;
    # it never receives arguments, result bodies, passages, or secrets.
    telemetry_observer: Callable[..., None] | None = field(default=None, repr=False)
    # Where a workflow turn's round trip went, reported per turn by the same
    # collector from the timing the turn response carries; durations only.
    workflow_turn_observer: Callable[..., None] | None = field(default=None, repr=False)
    # The governed root installs one whole-call meter before creating any
    # source journal. Listener and output callers cannot opt individual ACKs out.
    workflow_source_observer: Callable[..., None] | None = field(default=None, repr=False)

    # The configured tools the last ``tool_definitions()`` left out, each as
    # ``{"tool": <valid name or "">, "reason": <code>}``; never the entry's
    # description, schema, URL or any other content.
    omitted_tools: list[dict[str, str]] = field(default_factory=list, repr=False)

    _KNOWLEDGE_TOOL_NAME = KNOWLEDGE_TOOL_DECLARATION["name"]
    # One declaration for the API, this worker and the homepage demo
    # (``tool_declarations``); this is that table in the worker's shape.
    _BUILTIN_TOOL_DEFINITIONS = {
        name: _model_tool_definition(declaration) for name, declaration in BUILTIN_TOOL_DECLARATIONS.items()
    }
    _KNOWLEDGE_TOOL_DEFINITION = _model_tool_definition(KNOWLEDGE_TOOL_DECLARATION)

    @property
    def allowed_tools(self) -> frozenset[str]:
        """The configured names plus the built-ins they imply, within the
        call's reserved tool contract when it has one: a name outside it
        would have its usage refused at settlement."""
        names = set(_realtime_tool_names(self.config.get("tools")))
        names |= implied_tool_names(names)
        pinned = _component_billing_tool_names(self.config)
        return frozenset(names if pinned is None else names & pinned)

    @property
    def knowledge_tool_enabled(self) -> bool:
        """Whether this exact, already-authorized call may search sources.

        The marker is emitted only by the control plane's call-config
        resolver.  It is deliberately independent of the generic MCP
        allowlist: knowledge search has no customer-controlled endpoint or
        arguments other than its query string.
        """
        knowledge = self.config.get("knowledge")
        if knowledge is None:
            return False
        if not isinstance(knowledge, dict) or knowledge.get("tool") != self._KNOWLEDGE_TOOL_NAME:
            raise ControlPlaneRejected("realtime config has an invalid call-scoped knowledge tool")
        return True

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

    def invoke_tool(self, name: str, arguments: dict[str, Any], *, caller_input: dict[str, int] | None = None) -> dict[str, Any]:
        """One tool or knowledge call, through the call-keyed control-plane route.

        On a call whose config says ``model_safe_results`` (an anonymous
        listener's surface), only the control plane's exact model-safe shape is
        returned; a malformed answer or any failure reaching it is the one
        fixed unavailable sentence, never an exception the model reads.
        """
        if self.config.get("model_safe_results") is not True:
            return self._invoke_tool(name, arguments, caller_input=caller_input)
        try:
            result = self._invoke_tool(name, arguments, caller_input=caller_input)
        except Exception:  # noqa: BLE001 - its telemetry is already emitted
            return model_safe_unavailable(name)
        if not is_model_safe_tool_result(result, name):
            self.audit_events.append("tool_result_malformed")
            return model_safe_unavailable(name)
        return result

    def _invoke_tool(self, name: str, arguments: dict[str, Any], *, caller_input: dict[str, int] | None = None) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("tool invocation requires admitted live media")
        started_at = time.monotonic()
        input_characters = _safe_argument_character_count(arguments)
        try:
            if name == self._KNOWLEDGE_TOOL_NAME:
                if not self.knowledge_tool_enabled:
                    self.audit_events.append("knowledge_tool_rejected_not_configured")
                    raise ControlPlaneRejected("knowledge search is not configured for this call")
                # Discard every model-supplied field except the short query.  In
                # particular, the model can never select an organization, project,
                # agent, document, or knowledge base; all of those are derived by
                # POST /v1/calls/{call_id}/knowledge/query from the canonical call.
                query = arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ControlPlaneRejected("knowledge search requires a non-empty query")
                self._tool_sequence += 1
                result = self.tool_transport.request(
                    "POST",
                    f"/v1/calls/{self.call_id}/knowledge/query",
                    {"query": query},
                )
                self.audit_events.append("knowledge_tool_invoked_call_scoped")
            else:
                if name not in self.allowed_tools:
                    self.audit_events.append("tool_rejected_not_allowlisted")
                    raise ControlPlaneRejected(f"tool is not allowlisted for this agent: {name}")
                self._tool_sequence += 1
                request_id = f"{self.call_id}:{self._tool_sequence}"
                result = self.tool_transport.request(
                    "POST",
                    f"/v1/calls/{self.call_id}/agent-tool-calls",
                    {
                        "tool_call_id": request_id,
                        "name": name,
                        "arguments": arguments,
                        "caller_input": caller_input,
                    },
                )
                self.audit_events.append("tool_invoked_allowlisted")
            self._emit_tool_telemetry(
                name=name,
                started_at=started_at,
                outcome=_telemetry_outcome(result),
                audit_correlation_id=str(
                    result.get("telemetry_correlation_id")
                    or result.get("audit_correlation_id")
                    or ""
                ),
                input_characters=input_characters,
                reason_code=str(result.get("reason_code") or ""),
            )
            return result
        except Exception as exc:
            self._emit_tool_telemetry(
                name=name,
                started_at=started_at,
                outcome="error",
                input_characters=input_characters,
                reason_code=type(exc).__name__,
            )
            raise

    def action_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("caller inputs require admitted live media")
        if command not in {"action-producer", "action-activity", "action-inputs", "action-question", "action-release", "action-status", "workflow-action-release", "workflow-action-cancel"}:
            raise ValueError("unknown caller input command")
        return self.tool_transport.request("POST", f"/v1/calls/{self.call_id}/{command}", payload)

    def report_transfer_outcome(self, outcome: str) -> dict[str, Any]:
        """Tell the control plane how this call's one transfer dial ended.

        The attempt row was written ``connecting`` when the tool authorized
        the dial; this moves it to the outcome. The call is named by the
        route, never by the body.
        """
        return self.tool_transport.request(
            "POST",
            f"/v1/calls/{self.call_id}/transfer-outcome",
            {"outcome": str(outcome)},
        )

    def claim_transfer_dial(self) -> dict[str, Any]:
        """Claim (``claimed``) this call's one recorded transfer attempt before dialing it.

        A workflow transfer step's waiting turn hands the worker the dial;
        the control plane lets exactly one claim through, so the same dial is
        never placed twice. The call is named by the route, never the body.
        """
        return self.tool_transport.request("POST", f"/v1/calls/{self.call_id}/transfer-dial", {})

    def advance_workflow_turn(self, voice_input: dict[str, Any]) -> dict[str, Any]:
        """Start one logical turn with one bounded same-submission recovery.

        The immutable request binds the exact text, observation kind, sample
        origin, answered question and last acknowledged turn. An unresolved response
        after transport recovery requires an explicit retry. A second
        utterance never implicitly retries the old request.
        """
        self._require_workflow_media()
        return self._workflow_requests.advance(voice_input, send=self._send_workflow_turn)

    def retry_workflow_turn(self) -> dict[str, Any]:
        """Retry only the unresolved request, without replacing its input."""
        self._require_workflow_media()
        return self._workflow_requests.retry(send=self._send_workflow_turn)

    def _require_workflow_media(self) -> None:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("workflow turn advance requires admitted live media")

    def workflow_delivery_journal(self, reply: dict[str, Any], *, step_index: int):
        """Bind the presentation port to this call's latest acknowledged turn.

        Final facts remain reportable after media ends. New source, approval
        and exposure commands require both live media and the same local turn.
        The backend independently enforces current durable authority.
        """
        try:
            from .workflow_delivery_transport import WorkflowDeliveryTransport
        except ImportError:
            from workflow_delivery_transport import WorkflowDeliveryTransport

        self._require_workflow_media()
        request_id = reply.get("request_id")
        submission = self._workflow_requests.acknowledged_submission(request_id)

        def send_workflow_delivery(payload):
            if payload.get("command") not in {"finish", "source-delta", "source-terminal", "source-close"}:
                self._require_workflow_media()
                self._workflow_requests.acknowledged_submission(request_id)
            return self._send_workflow_journal(payload)

        return WorkflowDeliveryTransport(
            call_id=self.call_id, run_id=self.config.get("workflow_run_id"), submission=submission,
            reply=reply, step_index=step_index, send=send_workflow_delivery, observe_sources=self.workflow_source_observer)

    def workflow_controller_journal(self, spec):
        """Controller sources bind to this live call, never a business reply."""
        try:
            from .workflow_delivery_transport import WorkflowControllerTransport
        except ImportError:
            from workflow_delivery_transport import WorkflowControllerTransport
        self._require_workflow_media()
        def send(payload):
            if payload.get("command") not in {"controller-finish", "source-delta", "source-terminal", "source-close"}:
                self._require_workflow_media()
            return self._send_workflow_journal(payload)
        return WorkflowControllerTransport(call_id=self.call_id, run_id=self.config.get("workflow_run_id"),
            spec=spec, send=send, observe_sources=self.workflow_source_observer)

    def confirmation_source_journal(self):
        """The additional finite listener belongs to this free-form call."""
        try:
            from .workflow_source_transport import WorkflowSourceTransport
        except ImportError:
            from workflow_source_transport import WorkflowSourceTransport
        def send(payload):
            if payload.get("command") not in {"source-delta", "source-terminal", "source-close"}:
                if not self.media_admitted or self._cleaned:
                    raise SessionStateError("confirmation listening requires admitted live media")
            return self._send_source_journal("action-sources", payload)
        return WorkflowSourceTransport(call_id=self.call_id, run_id=None, allocation="confirmation_listener", send=send)

    def workflow_source_journal(self):
        """Bind continuous caller sources before any spoken output turn exists."""
        try:
            from .workflow_source_transport import WorkflowSourceTransport
        except ImportError:
            from workflow_source_transport import WorkflowSourceTransport
        self._require_workflow_media()
        def send(payload):
            if payload.get("command") not in {"source-delta", "source-terminal", "source-close"}:
                self._require_workflow_media()
            return self._send_workflow_journal(payload)
        return WorkflowSourceTransport(call_id=self.call_id, run_id=self.config.get("workflow_run_id"), send=send,
                                       observe=self.workflow_source_observer)

    def _send_workflow_journal(self, payload):
        return self._send_source_journal("workflow-deliveries", payload)

    def _send_source_journal(self, route, payload):
        try:
            # Claim/start are deliberately single transmissions. Only the
            # controller retries an identical immutable terminal fact.
            return self.tool_transport.request(
                "POST", f"/v1/calls/{self.call_id}/{route}", payload)
        except ControlPlaneRejected as error:
            if type(error.status_code) is int and 500 <= error.status_code < 600:
                raise ConnectionError("source delivery response is unavailable") from None
            raise
        except (urllib.error.URLError, http.client.IncompleteRead):
            raise ConnectionError("source delivery response is unavailable") from None

    def _send_workflow_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.monotonic()
        submission = dict(payload)
        for attempt in range(2):
            self._require_workflow_media()
            try:
                response = self.tool_transport.request(
                    "POST", f"/v1/calls/{self.call_id}/workflow-turns", dict(submission)
                )
                break
            except ControlPlaneRejected as exc:
                if attempt or type(exc.status_code) is not int or not 500 <= exc.status_code < 600:
                    raise
            except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead):
                if attempt:
                    raise
        observer = self.workflow_turn_observer
        if observer is not None and isinstance(response, dict):
            try:
                observer(
                    started_at=started_at,
                    timing=response.get("timing"),
                    decision_provider=response.get("decision_provider") or "",
                    decision_model=response.get("decision_model") or "",
                )
            except Exception:
                # Observability must never interfere with a turn.
                pass
        return response

    def fetch_keypad_value(self, field_name: str) -> str:
        """The dialled string of one keypad field of this live call.

        Asked for at the moment of pressing and kept only by the value-mode
        redactor. A refusal, a transport error and a malformed answer are all
        :class:`KeypadValueUnavailable`, which carries a reason code and never
        the field, the body or the value.
        """
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("keypad value requires admitted live media")
        try:
            response = self.tool_transport.request(
                "POST", f"/v1/calls/{self.call_id}/keypad-values/{urllib.parse.quote(str(field_name), safe='')}", {}
            )
        except ControlPlaneRejected as error:
            raise KeypadValueUnavailable(error.reason) from None
        except Exception:  # noqa: BLE001 - a transport failure reads as unavailable, with nothing attached
            raise KeypadValueUnavailable("transport_error") from None
        value = response.get("value") if isinstance(response, dict) else None
        if not (isinstance(value, str) and value and len(value) <= 32 and set(value) <= set("0123456789*#")):
            raise KeypadValueUnavailable("malformed_answer")
        return value

    def _emit_tool_telemetry(
        self,
        *,
        name: str,
        started_at: float,
        outcome: str,
        audit_correlation_id: str = "",
        input_characters: int = 0,
        reason_code: str = "",
    ) -> None:
        observer = self.telemetry_observer
        if observer is None:
            return
        try:
            observer(
                name=name,
                started_at=started_at,
                outcome=outcome,
                audit_correlation_id=audit_correlation_id,
                input_characters=input_characters,
                reason_code=reason_code,
            )
        except Exception:
            # Observability must never interfere with a tool result or call.
            return

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Expose configured tools plus the one call-scoped retrieval tool.

        Built-in schemas come from the shared declarations, webhook and
        connection schemas from the authoritative structured agent config.
        Legacy string tools retain the existing MCP behavior. Knowledge search
        is intentionally not fetched from MCP: it is an in-process,
        server-authorized declaration whose only model field is ``query``.

        Every entry is declared on its own: one this worker cannot declare (an
        unknown type or built-in, a malformed or repeated entry, a legacy name
        the listing lacks, a name outside the reserved tool contract) is left
        out and recorded in ``omitted_tools`` by name and reason, and the call
        keeps every other tool. Nothing here stops a call from starting.
        """

        pinned_component_tools = _component_billing_tool_names(self.config)
        omitted: list[dict[str, str]] = []
        self.omitted_tools = omitted

        def omit(name: Any, reason: str) -> None:
            omitted.append({"tool": _tool_label(name), "reason": reason})

        def pinned(name: str) -> bool:
            # Agent configuration can change after a call reserves its
            # immutable rate contract. Do not expose a later tool to this
            # already-funded call; a subsequent call gets a fresh snapshot.
            if pinned_component_tools is None or name in pinned_component_tools:
                return True
            omit(name, "not_in_call_contract")
            return False

        definitions: list[dict[str, Any]] = []
        declared: set[str] = set()
        mcp_tool_names: list[str] = []
        for item in self.config.get("tools") or []:
            if isinstance(item, str):
                name, tool_type = item.strip(), ""
            elif isinstance(item, dict):
                name = item.get("name").strip() if isinstance(item.get("name"), str) else ""
                tool_type = str(item.get("type") or "").strip()
            else:
                omit("", "malformed_entry")
                continue
            if not _MODEL_TOOL_NAME_RE.fullmatch(name):
                omit("", "invalid_name")
                continue
            if name in declared or name in mcp_tool_names:
                omit(name, "duplicate_name")
                continue
            if name == self._KNOWLEDGE_TOOL_NAME:
                # Only the call-scoped knowledge config enables this tool.
                omit(name, "reserved_name")
                continue
            if not pinned(name):
                continue
            if isinstance(item, str):
                mcp_tool_names.append(name)
                continue
            if tool_type == "builtin":
                declaration = builtin_tool_declaration(item)
                if declaration is None:
                    omit(name, "no_sms_templates" if name == "send_sms" else "builtin_not_declared")
                    continue
                definitions.append(_model_tool_definition(declaration))
            elif tool_type in ("webhook", "connection"):
                # A connection tool reaches the worker hydrated to the same
                # declaration a webhook carries; its transport stays on the
                # control plane, which executes it by name.
                parameters = item.get("parameters")
                if not isinstance(parameters, dict):
                    omit(name, "missing_parameters")
                    continue
                if _schema_too_deep(parameters):
                    omit(name, "invalid_schema")
                    continue
                definitions.append(
                    {
                        "name": name,
                        "description": str(item.get("description") or f"Skysay tool {name}"),
                        "inputSchema": parameters,
                    }
                )
            else:
                omit(name, "unsupported_type")
                continue
            declared.add(name)
        for name in sorted(implied_tool_names(declared)):
            if pinned(name):
                definitions.append(dict(self._BUILTIN_TOOL_DEFINITIONS[name]))
                declared.add(name)
        if mcp_tool_names:
            try:
                response = self.tool_transport.request(
                    "POST",
                    "/mcp",
                    {"jsonrpc": "2.0", "id": f"{self.call_id}:tools", "method": "tools/list"},
                )
                result = response.get("result") if isinstance(response, dict) else None
                tools = result.get("tools") if isinstance(result, dict) else None
            except Exception:  # noqa: BLE001 - the listing costs its own tools, not the call
                tools = None
            if not isinstance(tools, list):
                for name in mcp_tool_names:
                    omit(name, "mcp_listing_unavailable")
            else:
                listed: set[str] = set()
                for tool in tools:
                    name = tool.get("name") if isinstance(tool, dict) else None
                    if (
                        name in mcp_tool_names and name not in listed
                        and isinstance(tool.get("inputSchema"), dict) and not _schema_too_deep(tool["inputSchema"])
                    ):
                        definitions.append(tool)
                        listed.add(name)
                for name in mcp_tool_names:
                    if name not in listed:
                        omit(name, "mcp_schema_missing")
                self.audit_events.append("tool_schemas_filtered_to_allowlist")
        try:
            knowledge_enabled = self.knowledge_tool_enabled
        except ControlPlaneRejected:
            omit(self._KNOWLEDGE_TOOL_NAME, "invalid_knowledge_config")
            knowledge_enabled = False
        if knowledge_enabled and pinned(self._KNOWLEDGE_TOOL_NAME):
            # A knowledge attachment that became ready after reserve is
            # available on the next call, not retroactively on this one.
            definitions.append(dict(self._KNOWLEDGE_TOOL_DEFINITION))
        return definitions

    def report_lifecycle(
        self,
        *,
        event_id: str,
        event_type: str,
        source_component: str,
        observed_at: str = "",
        details: dict[str, Any] | None = None,
        funding_generation: int | None = None,
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
        if funding_generation is not None:
            payload["funding_generation"] = funding_generation
        if details:
            payload["details"] = details
        if self.browser_outcome is not None and event_type in {"browser.outcome", "sip.ended"}:
            payload["browser_outcome"] = dict(self.browser_outcome)
        component_usage_fallback = (
            dict(self._pending_component_usage)
            if event_type == "sip.ended" and self._pending_component_usage is not None
            else None
        )
        if component_usage_fallback is not None:
            # The normal path is always POST /billing-usage. This only carries
            # the same normalized, non-secret numeric object when the bounded
            # network retry was exhausted, allowing the terminal server
            # transaction to persist usage before its settlement branch.
            payload["component_usage"] = component_usage_fallback
        route = f"/v1/gateway-deployments/{self.gateway_id}/call-lifecycle-events"
        forwarded_component_usage = component_usage_fallback is not None
        try:
            result = self.transport.request("POST", route, payload)
        except (OSError, ControlPlaneRejected) as exc:
            if component_usage_fallback is None or self._component_usage_error_is_retryable(exc):
                raise
            # The EVENT was refused because of the usage object it is
            # carrying, not because of the event.  That object has already
            # been rejected by the separate usage route, so forwarding it
            # again can only keep failing -- and it does not fail alone: a
            # terminal event that never lands leaves the call unsettled, and
            # an artifact upload requires a finalized lifecycle, so one
            # invalid usage payload was costing the call its settlement AND
            # its recording and transcript.
            #
            # Report the event once more without the passenger. Terminal
            # settlement with no native usage is the documented
            # ``awaiting_usage`` hold, which the audited operator sweeper
            # already owns -- a far smaller loss than the whole call. The
            # server transaction is atomic and the event is idempotent on
            # ``event_id``, so the refused attempt left nothing behind.
            #
            # Only for a CONTRACT rejection. A transport error or a 5xx may
            # still deliver on a later attempt, and throwing away billing
            # evidence for one of those would be the worse trade -- so the
            # same predicate the usage retry uses decides here too.
            #
            # The retry is also the DIAGNOSIS, which is why nothing is
            # discarded before it answers. A rejection this method cannot see
            # the cause of -- an expired token, an unrelated contract failure
            # -- is not the usage object's fault, and guessing from the
            # message text would make a wrong guess silently cost a customer's
            # billing evidence. So: send the same event without the passenger.
            # If it is accepted, the passenger was demonstrably the problem
            # and only then is it dropped. If it is refused too, that
            # exception propagates with this one as its context and the staged
            # usage is still there for whatever retries next.
            payload.pop("component_usage", None)
            result = self.transport.request("POST", route, payload)
            self._pending_component_usage = None
            forwarded_component_usage = False
            self.audit_events.append("component_usage_fallback_discarded")
        expected = {
            "browser.outcome": "browser.outcome",
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
            if (
                not isinstance(billing, dict)
                or billing.get("status") not in AUTHORITATIVE_TERMINAL_BILLING_STATUSES
            ):
                raise ControlPlaneRejected("terminal call event did not produce an authoritative billing verdict")
            self.browser_outcome_finalized = self.browser_outcome is not None and billing.get("status") in {"released", "settled", "payment_due", "no_reservation"}
            self._lifecycle_finalized = True
            self.audit_events.append("billing_finalized_from_lifecycle")
        if forwarded_component_usage:
            self._pending_component_usage = None
            self.audit_events.append("component_usage_fallback_forwarded")
        return result

    def acknowledge_hangup(
        self,
        *,
        attempt: int,
        outcome: str,
        reason: str = "",
    ) -> dict[str, Any] | None:
        """Report the outcome of an operator-requested teardown.

        Deliberately its own route rather than a passenger on the lifecycle
        event: acknowledging is not a lifecycle event, and folding it in would
        make a heartbeat capable of settling a request as a side effect.

        Gateway-bound, so it authenticates exactly as the lifecycle posts do
        and a gateway token can only acknowledge on its own gateway.

        ``None`` means NO ROW MATCHED -- another consumer won the
        compare-and-set, or this attempt number has been superseded. That is a
        lost race, not a failure: the caller does nothing and never retries the
        same attempt, because it can never match again. Two consumers
        legitimately hold the same pending request, so losing is the ORDINARY
        case rather than a fault.

        The loser is told so by a **200 whose ``hangup_acknowledgement`` is
        null**, not by a null body and not by an error status. Reading the
        envelope instead of that key would score every lost race as a success
        and leave the request unacknowledged while this worker believed it had
        settled it.

        This is NOT a report that the call ended. Terminal reporting stays on
        the ordinary lifecycle path so settlement runs exactly once.
        """

        route = f"/v1/gateway-deployments/{self.gateway_id}/call-hangup-acknowledgements"
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            # An integer, never a string: the other side answers a non-integer
            # attempt with a 400.
            "attempt": int(attempt),
            "outcome": outcome,
        }
        if reason:
            payload["reason"] = reason
        result = self.transport.request("POST", route, payload)
        if not isinstance(result, dict):
            return None
        settled = result.get("hangup_acknowledgement")
        return settled if isinstance(settled, dict) and settled else None

    def native_agent(self, operation: str, tool_set_hash: str, agent_id: str = "") -> dict[str, Any]:
        """Resolve, register or forget the shared Eleven agent for one tool set.

        Call-bound like every media route: the control plane scopes the
        registry to this call's organization; the worker sends only the hash.
        """
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("native agent registry requires admitted live media")
        result = self.transport.request("POST", f"/v1/calls/{self.call_id}/native-agent", {
            "operation": operation, "tool_set_hash": tool_set_hash, "agent_id": agent_id})
        if not isinstance(result, dict) or not isinstance(result.get("agent_name"), str):
            raise ControlPlaneRejected("native agent registry response is invalid")
        return result

    def report_component_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Post normalized SDK usage before ``sip.ended`` settles the wallet.

        This is intentionally a teardown-time control-plane action, never a
        LiveKit media callback.  The route is call-bound and accepts no
        organization/project input from the worker.
        """
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("component usage reporting requires admitted live media")
        if not isinstance(usage, dict):
            raise ControlPlaneRejected("component usage must be an object")
        if not self.platform_media_worker:
            # The usage route belongs to the managed media worker; a gateway
            # token is always refused there. A self-hosted call's usage rides
            # its terminal ``sip.ended`` event, as the fallback below does.
            self._pending_component_usage = dict(usage)
            self.audit_events.append("component_usage_deferred_to_lifecycle")
            return {"call_id": self.call_id}
        # Stage before the first request. A later ``sip.ended`` can carry this
        # same bounded numeric object if *any* separate POST fails; only a
        # successful call-bound response clears it.
        self._pending_component_usage = dict(usage)
        outbox_digest = self.usage_report_outbox.stage("usage", usage) if self.usage_report_outbox else ""
        last_error: Exception | None = None
        # A short bounded retry covers a control-plane transition without
        # moving I/O into a media callback. Do not retry contract failures:
        # they must retain the hold for explicit reconciliation instead.
        for _attempt in range(3):
            try:
                result = self.transport.request(
                    "POST", f"/v1/calls/{self.call_id}/billing-usage", usage
                )
            except (OSError, ControlPlaneRejected) as exc:
                last_error = exc
                if self._component_usage_error_is_retryable(exc) and _attempt < 2:
                    continue
                break
            if result.get("call_id") != self.call_id:
                raise ControlPlaneRejected("component usage response is not bound to the active call")
            self._pending_component_usage = None
            self.audit_events.append("component_usage_reported")
            if self.usage_report_outbox:
                self.usage_report_outbox.acknowledge("usage", outbox_digest)
            return result
        # The RAM fallback joins the terminal lifecycle request. Separately,
        # the host outbox retains this exact report across process loss; if
        # neither reaches the API, terminal recovery bounds the hold's age.
        self._pending_component_usage = dict(usage)
        self.audit_events.append("component_usage_retry_exhausted")
        assert last_error is not None
        raise last_error

    def report_refused_metering(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Persist the counts a refused metering report still held.

        Diagnostic only: this must never be mixed into ``sip.ended`` or the
        customer ``call.ended`` webhook. A single attempt is enough; failure is
        the caller's to swallow so teardown still finishes.
        """
        if not isinstance(evidence, dict):
            raise ControlPlaneRejected("refused metering evidence must be an object")
        outbox_digest = self.usage_report_outbox.stage("refused", evidence) if self.usage_report_outbox else ""
        route = f"/v1/calls/{self.call_id}/refused-metering"
        dropped_session_duration = False
        try:
            result = self.transport.request("POST", route, evidence)
        except ControlPlaneRejected as exc:
            # An API that predates ``session_duration`` in this snapshot
            # refuses the whole report as carrying an unsupported field. The
            # voice worker can ship before the API, so keep everything else
            # rather than lose the report: one retry without that field.
            without = _without_session_duration(evidence)
            if exc.status_code != 400 or without is None:
                raise
            result = self.transport.request("POST", route, without)
            dropped_session_duration = True
        if result.get("call_id") != self.call_id:
            raise ControlPlaneRejected("refused metering response is not bound to the active call")
        if self.usage_report_outbox and not dropped_session_duration:
            self.usage_report_outbox.acknowledge("refused", outbox_digest)
        self.audit_events.append("refused_metering_reported")
        if dropped_session_duration:
            self.audit_events.append("refused_metering_session_duration_dropped")
            result = {**result, "session_duration_dropped": True}
        return result

    def lease_component_credentials(self) -> dict[str, dict[str, str]]:
        """Fetch this call's provider keys, never as config data.

        On the MANAGED CLOUD worker, platform-managed components still make no
        request: that worker holds the platform's provider keys already, and
        asking for them would add blast radius and nothing else.

        On a SELF-HOSTED worker there are no platform keys on the box, so the
        request is made for every component. What comes back is whatever
        Skysay has issued for THIS customer; a platform-managed component
        the control plane cannot issue (Google, Fish Audio) is simply absent
        from the response and keeps using the deployment's own key, unchanged.

        Two different strictnesses, on purpose:

        * a selected BYOK component MUST be leased, with an exactly matching
          provider. A missing or malformed BYOK lease is a hard pre-media
          failure -- silently falling back to a platform key would bill the
          wrong account and break the promise the customer bought;
        * a platform-managed component MAY be leased. Its absence is the
          documented "keep your own key" case, so equality is not the rule
          there. Anything the response DOES carry is still checked against the
          snapshot, and a component that is not in the snapshot at all is
          refused outright.
        """
        billing = self.config.get("component_billing")
        if not isinstance(billing, dict):
            return {}
        selections = [
            item for item in billing.get("components") or [] if isinstance(item, dict)
        ]
        required = [
            item
            for item in selections
            if str(item.get("credential_source") or "") == "customer_byok"
        ]
        optional = (
            [
                item
                for item in selections
                if str(item.get("credential_source") or "") == "platform_managed"
            ]
            if not self.platform_media_worker
            else []
        )
        if not required and not optional:
            return {}
        result = (self.credential_transport or self.transport).request(
            "POST", f"/v1/calls/{self.call_id}/component-credentials/lease", {}
        )
        if str(result.get("call_id") or "") != self.call_id:
            raise ControlPlaneRejected("component credential lease is not bound to the active call")
        raw_components = result.get("components")
        if not isinstance(raw_components, dict):
            raise ControlPlaneRejected("component credential lease omitted components")
        leased: dict[str, dict[str, str]] = {}
        required_names = {str(item.get("component") or "") for item in required}
        optional_names = {str(item.get("component") or "") for item in optional}
        if not required_names <= set(raw_components):
            raise ControlPlaneRejected("component credential lease does not match the selected components")
        if not set(raw_components) <= (required_names | optional_names):
            raise ControlPlaneRejected("component credential lease carries an unselected component")
        for item in required + optional:
            component = str(item.get("component") or "")
            provider = str(item.get("provider") or "")
            raw = raw_components.get(component)
            if raw is None and component in optional_names:
                continue
            if not isinstance(raw, dict) or str(raw.get("provider") or "") != provider:
                raise ControlPlaneRejected("component credential lease provider does not match the selected component")
            secret = raw.get("secret")
            if not isinstance(secret, str) or not secret:
                raise ControlPlaneRejected("component credential lease omitted a selected key")
            leased[component] = {"provider": provider, "secret": secret}
        self.audit_events.append("component_credentials_leased")
        return leased

    @staticmethod
    def _component_usage_error_is_retryable(error: Exception) -> bool:
        """Retry transport failures and explicit transient HTTP responses only."""
        if isinstance(error, OSError):
            return True
        message = str(error)
        return any(marker in message for marker in ("HTTP 408", "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"))

    def complete(
        self,
        *,
        source_path: Path,
        duration_sources: dict[str, float],
        require_semantic_turns: bool = True,
        semantic_evidence_status: str | None = None,
    ) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("completion requires admitted media before cleanup")
        if self._completed:
            raise SessionStateError("call completion is already finalized")
        if not self._lifecycle_finalized:
            raise SessionStateError("artifact upload requires finalized gateway lifecycle billing")
        bundle = collect_artifact_bundle(
            source_path,
            duration_sources=duration_sources,
            require_semantic_turns=require_semantic_turns,
            semantic_evidence_status=semantic_evidence_status,
        )
        named_parts = bundle.parts
        evidence = bundle.evidence
        normalized_duration = bundle.duration
        gaps = evidence["missing"]
        if str(self.config.get("content_storage") or "") == CONTENT_STORAGE_GATEWAY:
            # Content-local: the bundle stays where it was written, in this
            # gateway's call-evidence directory. No request at all, and a
            # status the reconciler never retries.
            self._completed = True
            self.audit_events.append("artifacts_retained_on_gateway")
            return {
                **normalized_duration,
                "artifact_import": {"status": ARTIFACT_RETAINED_ON_GATEWAY},
                "evidence": evidence,
            }
        if callable(getattr(self.transport, "request_bytes", None)):
            # A failure propagates unreported: the caller decides whether it is
            # retried later or is final, and only a final one belongs on the
            # call's timeline (``report_artifact_upload_failure`` below).
            imported = upload_artifact_bundle(self.transport, self.call_id, named_parts)
        else:
            # A transport that can only speak small JSON (an in-process test
            # double) gets the single-request shape it always had.
            imported = self.transport.request(
                "POST",
                f"/v1/calls/{self.call_id}/artifacts",
                {
                    "parts": [_upload_part(name, content) for name, content in named_parts],
                    "evidence": evidence,
                },
            )
        self._completed = True
        self.audit_events.extend(["billing_duration_provenance_recorded", "artifacts_imported"])
        if gaps:
            self.audit_events.append("artifact_evidence_incomplete")
        return {
            **normalized_duration,
            "artifact_import": imported,
            "evidence": evidence,
        }

    def accept_live_demo_settlement(self, settled: Mapping[str, Any]) -> None:
        """The homepage demo's authoritative settlement as this call's billed end.

        A phone call's artifacts upload after the gateway's terminal billing
        event; a homepage demo call has no gateway and no wallet, and its
        billing ends with the demo's own settlement, which closes the call row.
        Accepted only when the control plane says it closed THIS call.
        """
        if (
            not isinstance(settled, Mapping)
            or settled.get("ok") is not True
            or str(settled.get("call_id") or "") != self.call_id
            or str(settled.get("call_status") or "") not in {"completed", "failed"}
        ):
            raise ControlPlaneRejected("the live demo settlement did not close this call")
        self._lifecycle_finalized = True
        self.audit_events.append("live_demo_settlement_accepted")

    def report_artifact_upload_failure(self, failure: ArtifactUploadFailed) -> bool:
        """Put a FINAL upload failure on the call's timeline (best effort)."""
        return report_artifact_upload_failure(self.transport, self.call_id, failure)

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


@dataclass(frozen=True)
class ArtifactBundle:
    """One call's evidence bundle, exactly as it is uploaded."""

    parts: list[tuple[str, bytes]]
    evidence: dict[str, Any]
    duration: dict[str, Any]


def collect_artifact_bundle(
    source_path: Path,
    *,
    duration_sources: dict[str, float],
    require_semantic_turns: bool = True,
    semantic_evidence_status: str | None = None,
) -> ArtifactBundle:
    """Read an evidence directory into the bundle ``complete`` uploads.

    Shared with the operator recovery command, so a re-upload of a retained
    evidence directory selects exactly the files the worker would have sent.
    """
    # The PR #79 contract classifies the upload, it does not gate it.
    # Refusing to store anything discarded the caller/agent audio,
    # transcript and telemetry that DID survive a mid-reply hangup, and
    # left the platform with no record of the call at all.  Upload every
    # part that exists and carry the gaps with it.
    gaps = _pr79_artifact_gaps(
        source_path,
        require_semantic_turns=require_semantic_turns,
    )
    normalized_duration = _duration_provenance(duration_sources)
    provenance = json.dumps(normalized_duration, sort_keys=True).encode()
    # The bounded canonical telemetry bundle is created only
    # after live media has ended.  It is optional here for backwards
    # compatibility with legacy evidence, but every current production
    # worker writes it before calling ``complete``.
    artifact_paths = [
        path
        for path in (
            source_path / "caller.wav",
            source_path / "agent.wav",
            source_path / "mixed.wav",
            source_path / "transcript.json",
            source_path / "waveform-sources.json",
            *sorted(source_path.glob("turn-*-caller.wav")),
            *sorted(source_path.glob("turn-*-agent.wav")),
            source_path / "opening-agent.wav",
            *sorted(source_path.glob("opening-*-agent.wav")),
            source_path / "terminal-agent.wav",
            *sorted(source_path.glob("terminal-*-agent.wav")),
            # A speech-to-speech agent's speech no caller turn paired.
            *sorted(source_path.glob("speech-*-agent.wav")),
            source_path / "call-telemetry.json",
            source_path / "sip-call-statistics.json",
        )
        if path.is_file()
    ]
    if not artifact_paths:
        # Nothing survived, so there is no partial record to preserve and
        # the caller must not read this as a stored call.
        raise IncompleteArtifactEvidence("missing PR #79 evidence: " + ", ".join(gaps))
    evidence = {
        "schema": "skysay.artifact_evidence_status.v1",
        "status": "complete" if not gaps else "incomplete",
        # The recorder is the only authority on WHY a semantic turn is
        # missing.  Without its verdict, say so rather than guessing one.
        "semantic_evidence_status": semantic_evidence_status
        or ("complete" if not gaps else "unknown"),
        "require_semantic_turns": require_semantic_turns,
        "missing": gaps,
    }
    named_parts = [(path.name, path.read_bytes()) for path in artifact_paths]
    named_parts.append(("duration-provenance.json", provenance))
    named_parts.append(
        (
            "evidence-status.json",
            json.dumps(evidence, indent=2, sort_keys=True).encode() + b"\n",
        )
    )
    return ArtifactBundle(parts=named_parts, evidence=evidence, duration=normalized_duration)


def _no_keypad_value_for_the_canary(self: Any, field_name: str) -> str:
    """The canary has no call variables: asking it for a keypad value is always a refusal."""

    del self, field_name
    raise KeypadValueUnavailable("platform_release_canary")


@dataclass
class PlatformReleaseCanarySession:
    """Local-only session facade for the fixed operational PSTN canary.

    The control plane resolves the runnable agent config from its canonical
    source DID, but this object deliberately performs no lifecycle, billing,
    artifact, credential-lease, tool, or dashboard request.  Its ID is local
    and non-customer-facing so the standard worker cleanup shape can remain
    unchanged without fabricating a customer call.
    """

    config: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"platform-release-canary-{uuid.uuid4().hex}")
    #: What this session is for, stamped into the worker's session-state file
    #: so the crash reconciler can tell a platform canary from a customer call
    #: without a call row to ask. A canary id is never a call row, so a sweep
    #: that treats it like one asks the control plane about a call that cannot
    #: exist. Ordinary session objects have no ``purpose`` at all, so a reader
    #: uses ``getattr(session, "purpose", "")``.
    browser_session_id: str = ""
    browser_outcome: dict[str, str] | None = None
    browser_outcome_finalized: bool = False
    purpose: str = CANARY_SESSION_PURPOSE
    media_admitted: bool = False
    telemetry_observer: Callable[..., None] | None = field(default=None, repr=False)
    workflow_turn_observer: Callable[..., None] | None = field(default=None, repr=False)
    suppress_lifecycle_reporting: bool = True
    # Read unconditionally by the shared pre-media path, so it must exist here
    # even though the value cannot change this session's behaviour: the route
    # below strips ``component_billing``, and ``lease_component_credentials``
    # returns an empty overlay without a request either way. False mirrors
    # ``Gate2CallSession``'s default, which fails toward asking rather than
    # toward assuming keys are present that are not.
    platform_media_worker: bool = False
    usage_report_outbox: Any = field(default=None, repr=False)
    #: The canary declares no tools, so it leaves none out.
    omitted_tools: list[dict[str, str]] = field(default_factory=list, repr=False)
    _cleaned: bool = False
    _completed: bool = False

    #: The shared worker path reads this off its session; the canary never navigates, so it only refuses.
    fetch_keypad_value = _no_keypad_value_for_the_canary

    def admit_media(self) -> None:
        if self._cleaned:
            raise SessionStateError("cannot admit media after cleanup")
        self.media_admitted = True

    def lease_component_credentials(self) -> dict[str, dict[str, str]]:
        return {}

    def tool_definitions(self) -> list[dict[str, Any]]:
        return []

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del name, arguments
        raise ControlPlaneRejected("release canary does not expose tools")

    def report_lifecycle(self, **_kwargs: Any) -> dict[str, str]:
        return {"status": "platform_release_canary"}

    def report_component_usage(self, _payload: dict[str, Any]) -> dict[str, str]:
        return {"status": "platform_release_canary"}

    def report_refused_metering(self, _payload: dict[str, Any]) -> dict[str, str]:
        return {"status": "platform_release_canary"}

    def report_transfer_outcome(self, _outcome: str) -> dict[str, str]:
        return {"status": "platform_release_canary"}

    def claim_transfer_dial(self) -> dict[str, Any]:
        return {"ok": False, "reason_code": "platform_release_canary"}

    def acknowledge_hangup(self, **_kwargs: Any) -> None:
        """Nothing to settle: no operator can have asked for this call to end.

        The canary's call id is local and never reaches the control plane, so
        no hangup request can exist against it and there is no row to
        compare-and-set.  ``None`` is exactly what the real session returns
        when no row matched, and the worker already treats that as "another
        consumer won it, do nothing" -- so the shared teardown behaves here
        without this facade inventing a settlement or issuing the request the
        canary is defined not to make.
        """

        return None

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        if not self.media_admitted or self._cleaned:
            raise SessionStateError("completion requires admitted media before cleanup")
        if self._completed:
            raise SessionStateError("call completion is already finalized")
        self._completed = True
        return {"artifact_import": {"status": "platform_release_canary"}}

    def report_artifact_upload_failure(self, _failure: Any) -> bool:
        """Nothing to report against: the canary uploads nothing."""
        return False

    def cleanup(
        self,
        *,
        remove_participant: Callable[[], None],
        delete_room: Callable[[], None],
    ) -> dict[str, Any]:
        if self._cleaned:
            return {"clean": True, "already_clean": True}
        remove_participant()
        delete_room()
        self._cleaned = True
        return {"clean": True, "errors": []}


class Gate2ControlPlaneAdapter:
    def __init__(
        self,
        *,
        gateway_id: str,
        organization_id: str = "",
        transport: JsonTransport,
        tool_transport: JsonTransport | None = None,
        call_transport: JsonTransport | None = None,
        platform_transport: JsonTransport | None = None,
    ) -> None:
        if not gateway_id:
            raise ValueError("gateway_id is required")
        self.gateway_id = gateway_id
        # The self-host worker is organization-bound. Its inbound-route call
        # must name that same tenant so the control plane can reject a stale
        # or mismatched worker key before it looks up a DID.
        self.organization_id = organization_id.strip()
        self.transport = transport
        # Production should inject an organization-scoped tool credential;
        # narrow gateway tokens deliberately cannot invoke customer tools.
        self.tool_transport = tool_transport or transport
        # Inbound routing is an organization call mutation and deliberately is
        # not granted to a narrow gateway token.
        self.call_transport = call_transport or transport
        # The managed LiveKit fleet discovers the tenant from the called DID.
        # A dedicated, narrowly scoped platform credential is required for that
        # cross-tenant lookup and all subsequent call-bound actions.
        self.platform_transport = platform_transport
        self.outbound_dial_plan: dict[str, Any] | None = None

    def _require_content_storage(self, config: dict[str, Any]) -> None:
        """A self-hosted worker runs only a config that says where content is kept.

        A control plane too old to say (a rollback below this release) is
        refused before media, rather than uploading content by default.
        """
        if self.platform_transport is None and str(config.get("content_storage") or "") not in CONTENT_STORAGE_MODES:
            raise ControlPlaneRejected(
                "the call config does not say where call content is kept; this gateway needs a newer control plane"
            )

    def report_failed_before_media(
        self,
        *,
        call_id: str,
        event_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Close a reserved call which policy prevented from reaching media.

        ``details`` is merged alongside ``reason`` rather than
        replacing it -- e.g. the SIP response code a ``create_sip_participant``
        failure actually carried, which the control plane uses to tell a
        ring timeout from a genuine carrier rejection.
        """

        result = (self.platform_transport or self.transport).request(
            "POST",
            f"/v1/gateway-deployments/{self.gateway_id}/call-lifecycle-events",
            {
                "event_id": event_id,
                "call_id": call_id,
                "event_type": "sip.failed",
                "source_component": "livekit_sip",
                # `reason` is spread LAST so it always wins: dict-literal
                # merge order means an earlier `{"reason": reason, **details}`
                # would let a `details["reason"]` key silently override the
                # canonical classification (Codex + DeepSeek review). Only
                # one caller passes `details` today (the SIP status code)
                # and never sets its own "reason", but the lifecycle service
                # keys billing/status off this field, so it must stay
                # authoritative regardless of what a future caller supplies.
                "details": {**(details or {}), "reason": reason},
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
        execution_lease: dict[str, str] | None = None,
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
                **(execution_lease or {}),
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
        config = None
        if execution_lease is not None:
            config = authorization.get("realtime_config")
            plan = authorization.get("dial_plan")
            if (authorization.get("worker_job_id") != execution_lease["worker_job_id"]
                or not isinstance(config, dict) or not isinstance(plan, dict)
                or plan.get("call_id") != call_id or plan.get("agent_id") != agent_id
                or plan.get("number_id") != from_number_id or plan.get("to_phone_number") != to_e164
                or not plan.get("organization_id") or not plan.get("project_id") or not plan.get("from_phone_number")):
                raise ControlPlaneRejected("hosted authorization omitted its canonical call plan")
            self.outbound_dial_plan = plan
        return self._session_for_call_authorization(
            call_id=call_id,
            agent_id=agent_id,
            from_number_id=from_number_id,
            to_e164=to_e164,
            token=token,
            config=config,
        )

    def _session_for_call_authorization(
        self,
        *,
        call_id: str,
        agent_id: str,
        from_number_id: str,
        to_e164: str,
        token: str,
        config: dict[str, Any] | None = None,
    ) -> Gate2CallSession:
        """Build the pre-media session an allowed single-call token unlocks.

        Both ways of obtaining that token end here, so the media-admission
        contract cannot drift between them: the config is fetched for the
        agent the control plane named, and the token is spent by
        ``admit_media`` against the destination it is bound to.
        """
        if config is None:
            config = self.transport.request("GET", f"/v1/agents/{agent_id}/realtime-config")
        _validate_realtime_config(config, expected_agent_id=agent_id)
        self._require_content_storage(config)
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
            platform_media_worker=self.platform_transport is not None,
            credential_transport=self.platform_transport or self.transport,
            authorization_call_id=call_id,
            call_authorization_token=token,
            config=config,
            from_number_id=from_number_id,
            to_e164=to_e164,
        )

    def admit_local_outbound(
        self,
        *,
        to_e164: str,
        from_number_id: str = "",
        from_phone_number: str = "",
        request: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], str]:
        """Ask the control plane to admit an outbound call this box will dial.

        The queue path exists because the dialer did not request the call. On a
        self-hosted box it did, so this asks for admission and origination
        authority in ONE call: the control plane runs the same policy decision,
        wallet reserve and call record it runs for a remote request, and
        answers with the call plus the single-use token that admits its media.

        Nothing about the workspace is sent: the organization and project are
        read from this deployment's own token server-side. What IS sent back is
        checked against what was asked for -- a dial plan naming another
        destination or another caller ID is refused rather than dialled.

        Returns the validated dial plan and the single-use token separately, so
        the caller knows the admitted call id before it does anything that can
        fail -- and can therefore report a pre-media failure for it.
        """
        if not from_number_id and not from_phone_number:
            raise ControlPlaneRejected("an outbound call needs a caller id")
        body = dict(request or {})
        body["to"] = to_e164
        if from_number_id:
            body["from_number_id"] = from_number_id
        if from_phone_number:
            body["from_phone_number"] = from_phone_number
        response = self.transport.request(
            "POST",
            f"/v1/gateway-deployments/{self.gateway_id}/outbound-calls",
            body,
        )
        if response.get("authorized") is not True:
            call = response.get("call") if isinstance(response.get("call"), dict) else {}
            detail = ", ".join(
                part
                for part in (
                    f"call {call.get('id')}" if call.get("id") else "",
                    f"status {call.get('status')}" if call.get("status") else "",
                )
                if part
            )
            raise ControlPlaneRejected(
                "outbound call admission did not authorize origination"
                + (f" ({detail})" if detail else "")
            )
        authorization = response.get("authorization")
        dial_plan = response.get("dial_plan")
        if not isinstance(authorization, dict) or not isinstance(dial_plan, dict):
            raise ControlPlaneRejected("authorized admission omitted its authorization or dial plan")
        token = authorization.get("token")
        fields = {
            name: str(dial_plan.get(name) or "").strip()
            for name in ("call_id", "agent_id", "number_id", "from_phone_number", "to_phone_number")
        }
        if not isinstance(token, str) or not token:
            raise ControlPlaneRejected("authorized admission omitted its single-use token")
        missing = [name for name, value in fields.items() if not value]
        if missing:
            raise ControlPlaneRejected(
                "authorized admission omitted required dial-plan fields: " + ", ".join(missing)
            )
        if str(authorization.get("call_id") or "") != fields["call_id"]:
            raise ControlPlaneRejected("single-call authorization is not bound to the admitted call")
        if fields["to_phone_number"] != to_e164:
            raise ControlPlaneRejected("admitted dial plan names a different destination")
        if from_number_id and fields["number_id"] != from_number_id:
            raise ControlPlaneRejected("admitted dial plan names a different caller id")
        if from_phone_number and fields["from_phone_number"] != from_phone_number:
            raise ControlPlaneRejected("admitted dial plan names a different caller id")
        fields["outbound_trunk_name"] = str(dial_plan.get("outbound_trunk_name") or "").strip()
        return fields, token

    def session_for_local_outbound(self, dial_plan: dict[str, str], token: str) -> Gate2CallSession:
        """Turn an admitted dial plan into the same pre-media session the queue path builds."""

        return self._session_for_call_authorization(
            call_id=dial_plan["call_id"],
            agent_id=dial_plan["agent_id"],
            from_number_id=dial_plan["number_id"],
            to_e164=dial_plan["to_phone_number"],
            token=token,
        )

    def prepare_workspace_browser_test(
        self,
        *,
        call_id: str,
        agent_id: str,
        organization_id: str,
        project_id: str,
        admission: str = "",
    ) -> Gate2CallSession:
        """Resume an authenticated workspace browser test from trusted call state.

        ``admission`` is ``"live_call_test"`` when this worker admitted the job
        on the live-call test branch (``phone_admission``); the control plane
        refuses the claim when that does not match the call.
        """

        call_bound_transport = self.platform_transport or self.call_transport
        claim = call_bound_transport.request(
            "POST",
            f"/v1/platform-media/browser-sessions/{call_id}/claim",
            {"admission": admission} if admission else {},
        )
        if (
            isinstance(claim, dict)
            and claim.get("ok") is False
            and claim.get("status") == "live"
            and claim.get("already_owned") is True
            and claim.get("call_id") == call_id
        ):
            raise BrowserSessionAlreadyOwned(
                "workspace browser-test session is already owned by a media worker"
            )
        if (
            not isinstance(claim, dict)
            or claim.get("ok") is not True
            or claim.get("status") != "live"
            or claim.get("call_id") != call_id
        ):
            raise ControlPlaneRejected("workspace browser-test claim was not accepted")
        config = call_bound_transport.request("GET", f"/v1/calls/{call_id}/realtime-config")
        _validate_realtime_config(
            config,
            expected_agent_id=agent_id,
            expected_organization_id=organization_id,
            expected_project_id=project_id,
        )
        if config.get("call_id") != call_id or config.get("direction") != "web_test":
            raise ControlPlaneRejected("call realtime config is not bound to this browser test")
        return Gate2CallSession(
            transport=call_bound_transport,
            authorization_transport=call_bound_transport,
            tool_transport=call_bound_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            platform_media_worker=self.platform_transport is not None,
            credential_transport=self.platform_transport or self.transport,
            authorization_call_id=call_id,
            call_authorization_token="",
            config=config,
            browser_session_id=str(claim.get("session_id") or ""),
            media_admitted=True,
            audit_events=[
                "authorization_allowed",
                "realtime_config_loaded",
                "media_admitted",
            ],
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

        call_bound_transport = self.platform_transport or self.call_transport
        config = call_bound_transport.request("GET", f"/v1/calls/{call_id}/realtime-config")
        _validate_realtime_config(config, expected_agent_id=agent_id)
        self._require_content_storage(config)
        if config.get("call_id") != call_id or config.get("direction") != "outbound":
            raise ControlPlaneRejected("call realtime config is not bound to this outbound call")
        return Gate2CallSession(
            transport=call_bound_transport,
            authorization_transport=call_bound_transport,
            tool_transport=call_bound_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            platform_media_worker=self.platform_transport is not None,
            credential_transport=self.platform_transport or self.transport,
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
        to_e164: str,
        from_e164: str,
        provider_call_id: str,
        inbound_trunk_id: str = "",
    ) -> Gate2CallSession:
        """Resolve number policy and the authoritative agent before subscribing.

        LiveKit SIP may have created the room, but the agent must not connect or
        consume its audio until this route/policy call returns an allowed route.
        """

        platform_transport = self.platform_transport or self.call_transport
        routed = platform_transport.request(
            "POST",
            "/v1/calls/inbound/route",
            {
                "to_phone_number": to_e164,
                "from_phone_number": from_e164,
                "provider": "didww",
                "provider_call_id": provider_call_id,
                "inbound_trunk_id": inbound_trunk_id,
                "live_media": True,
                "media_pipeline": "realtime",
                **({"organization_id": self.organization_id} if self.organization_id else {}),
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
        organization_id = route.get("organization_id")
        project_id = route.get("project_id")
        if not isinstance(organization_id, str) or not organization_id or not isinstance(project_id, str) or not project_id:
            raise ControlPlaneRejected("inbound route omitted the organization/project binding")
        config = platform_transport.request("GET", f"/v1/calls/{call_id}/realtime-config")
        _validate_realtime_config(
            config,
            expected_agent_id=agent_id,
            expected_organization_id=organization_id,
            expected_project_id=project_id,
        )
        self._require_content_storage(config)
        return Gate2CallSession(
            transport=platform_transport,
            authorization_transport=platform_transport,
            tool_transport=platform_transport,
            call_id=call_id,
            gateway_id=self.gateway_id,
            platform_media_worker=self.platform_transport is not None,
            credential_transport=self.platform_transport or self.transport,
            authorization_call_id="",
            call_authorization_token="",
            config=config,
        )

    def prepare_platform_release_canary(
        self, canonical_route: Mapping[str, str], provider_call_id: str = ""
    ) -> PlatformReleaseCanarySession:
        """Claim the platform-only release route for one inbound leg.

        The production-only fixed tuple lives in the private release contract,
        not in this portable LiveKit control-plane adapter.  The caller is the
        host-owned release path; this adapter only proves that the platform
        returned exactly the tuple that path supplied.

        The platform admits this route only while the release host holds an
        open single-use claim, so it can reject a leg that merely presents the
        fixed tuple.  ``provider_call_id`` binds the consumed claim to this
        carrier leg: LiveKit dispatch is at-least-once, and a retransmitted
        job for the same leg must replay the claim it already consumed rather
        than be refused as an impostor.
        """
        platform_transport = self.platform_transport
        if platform_transport is None:
            raise ControlPlaneRejected("release canary requires a platform media transport")
        routed = platform_transport.request(
            "POST",
            "/v1/platform-media/release-canary/route",
            {"provider_call_id": str(provider_call_id or "")},
        )
        expected = {name: str(canonical_route.get(name) or "") for name in (
            "purpose", "source_e164", "target_e164"
        )}
        if (
            expected["purpose"] != "platform_release_canary"
            or not expected["source_e164"].startswith("+")
            or not expected["source_e164"][1:].isdigit()
            or not expected["target_e164"].startswith("+")
            or not expected["target_e164"][1:].isdigit()
        ):
            raise ControlPlaneRejected("release canary caller omitted a valid canonical route")
        if any(routed.get(name) != value for name, value in expected.items()):
            raise ControlPlaneRejected("release canary route did not return the canonical fixed tuple")
        config = routed.get("config")
        if not isinstance(config, dict):
            raise ControlPlaneRejected("release canary route omitted a runtime config")
        try:
            from .gpt_live_runtime import is_eleven_native_config
        except ImportError:  # the worker image imports flat modules
            from gpt_live_runtime import is_eleven_native_config
        if is_eleven_native_config(config):
            # This facade has no durable call or wallet reserve. Native
            # qualification must use the ordinary admitted call.
            raise ControlPlaneRejected("native provider sessions are unsupported on the release canary")
        _validate_realtime_config(
            config,
            expected_agent_id=str(config.get("agent_id") or ""),
            # Caller-side recording is a customer obligation: it exists so a
            # tenant's call produces the consent-bearing artifact set the
            # platform promises to keep.  This route has no tenant, no call
            # record, and no artifact path, so the control plane switches
            # recording off on purpose — and the worker reads the flag nowhere
            # else.  Requiring it here does not protect a customer; it only
            # rejects the one config the platform route can ever return.
            require_recording=False,
        )
        if config.get("tools") or config.get("knowledge"):
            raise ControlPlaneRejected("release canary config must not expose customer tools or knowledge")
        if "component_billing" in config:
            raise ControlPlaneRejected("release canary config must omit component billing")
        # Presence alone flips the shared worker path onto the workflow-driven
        # turn loop, which drives the session through ``advance_workflow_turn``.
        # This facade implements no workflow turn, so the boundary refuses the
        # config rather than leaving the gap unreachable only because today's
        # agent-level resolver happens never to emit the key.
        if config.get("workflow_run_id"):
            raise ControlPlaneRejected("release canary config must not pin a workflow run")
        # Adopt the control plane's correlation id when it returned a safe one,
        # so the host reads back exactly this session's terminal state file. A
        # control plane predating this returns none, and the session keeps its
        # locally minted id -- the run then proves the call on its conversation
        # evidence, without the finished-call assertion the host cannot key.
        session_call_id = str(routed.get("call_id") or "")
        if session_call_id and _RELEASE_CANARY_CALL_ID.fullmatch(session_call_id):
            return PlatformReleaseCanarySession(config=config, call_id=session_call_id)
        return PlatformReleaseCanarySession(config=config)


def _validate_realtime_config(
    config: dict[str, Any],
    *,
    expected_agent_id: str,
    expected_organization_id: str | None = None,
    expected_project_id: str | None = None,
    require_recording: bool = True,
) -> None:
    """Reject a runtime config the worker must not run.

    ``require_recording`` defaults to on, because every customer route reaches
    this function and a customer call that answers without recording breaks
    the artifact contract.  Only the platform release canary passes it off:
    that route is not a customer call at all.
    """
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
    if expected_organization_id is not None and config["organization_id"] != expected_organization_id:
        raise ControlPlaneRejected("realtime config organization does not match the called number")
    if expected_project_id is not None and config["project_id"] != expected_project_id:
        raise ControlPlaneRejected("realtime config project does not match the called number")
    for component in ("transcriber", "model", "voice"):
        value = config.get(component)
        if (
            not isinstance(value, dict)
            or not value.get("registry_name")
            or not value.get("model")
        ):
            raise ControlPlaneRejected(f"realtime config has no runnable {component}")
    _realtime_tool_names(config.get("tools"))
    _component_billing_tool_names(config)
    if require_recording and config.get("recording_enabled") is not True:
        raise ControlPlaneRejected("Gate 2 requires caller-side recording to be enabled")


def _realtime_tool_names(tools: Any) -> list[str]:
    """Read names from the authoritative string or structured tool allowlist.

    An entry without a usable name allows nothing; ``tool_definitions()``
    leaves it out and records it, and the call keeps its other tools.
    """
    if not isinstance(tools, list):
        raise ControlPlaneRejected("realtime config tools must be a list")
    names: list[str] = []
    for item in tools:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _component_billing_tool_names(config: dict[str, Any]) -> frozenset[str] | None:
    """Return the bounded names frozen with a component-billed reserve.

    Legacy/non-component calls intentionally retain their existing broad tool
    behavior. A component-metered call instead has to prove that every model
    schema it exposes was selected before its wallet reserve, where any known
    paid rate was pinned.
    """
    billing = config.get("component_billing")
    if billing is None:
        return None
    if not isinstance(billing, dict) or billing.get("usage_required") is not True:
        raise ControlPlaneRejected("component billing config is invalid")
    contract = billing.get("tool_contract")
    if not isinstance(contract, dict):
        raise ControlPlaneRejected("component billing config omitted its pinned tool contract")
    names = contract.get("allowed_names")
    if not isinstance(names, list):
        raise ControlPlaneRejected("component billing tool contract allowlist is invalid")
    if len(names) > _COMPONENT_TOOL_NAME_LIMIT:
        raise ControlPlaneRejected("component billing tool contract exceeds the bounded name limit")
    normalized = [str(name or "").strip() for name in names]
    if (
        len(set(normalized)) != len(normalized)
        or any(not _COMPONENT_TOOL_NAME_RE.fullmatch(name) for name in normalized)
    ):
        raise ControlPlaneRejected("component billing tool contract has duplicate or unsafe names")
    return frozenset(normalized)


def _pr79_artifact_gaps(
    root: Path, *, require_semantic_turns: bool = True
) -> list[str]:
    """Describe every way ``root`` falls short of complete PR #79 evidence.

    This is the DEFINITION of complete evidence, not a precondition for
    storing anything.  ``_validate_pr79_artifact_set`` raises on any gap;
    ``Gate2CallSession.complete`` uploads what exists and records the gaps.
    """

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
    if caller_turns != agent_turns or (require_semantic_turns and not caller_turns):
        missing.append("one-to-one turn-*-caller.wav and turn-*-agent.wav")
    transcript = root / "transcript.json"
    if transcript.is_file():
        try:
            value = json.loads(transcript.read_text())
            if not isinstance(value, dict) or not isinstance(value.get("turns"), list):
                missing.append("transcript.json with a turns list")
            else:
                if require_semantic_turns and not value["turns"]:
                    missing.append("transcript.json with at least one turn")
                if len(value["turns"]) != len(caller_turns):
                    missing.append("transcript turns with one-to-one caller/agent audio clips")
                if value["turns"]:
                    declared_caller = {
                        str(turn.get("caller_audio"))
                        for turn in value["turns"]
                        if isinstance(turn, dict) and turn.get("caller_audio")
                    }
                    declared_agent = {
                        str(turn.get("agent_audio"))
                        for turn in value["turns"]
                        if isinstance(turn, dict) and turn.get("agent_audio")
                    }
                    actual_caller = {f"{stem}-caller.wav" for stem in caller_turns}
                    actual_agent = {f"{stem}-agent.wav" for stem in agent_turns}
                    if (declared_caller and declared_caller != actual_caller) or (
                        declared_agent and declared_agent != actual_agent
                    ):
                        missing.append("transcript audio filenames matching turn clips")
                if not valid_semantic_response_audio_policy(
                    value.get("semantic_response_audio_policy")
                ):
                    missing.append("transcript.json with the PR #79 no-filler policy")
                if value.get("skysay") and not valid_effective_voice_stack_identity(
                    value.get("effective_voice_stack")
                ):
                    missing.append("transcript.json with a valid effective voice stack identity")
                if any(
                    isinstance(turn, dict) and _turn_is_degraded(turn)
                    for turn in value["turns"]
                ):
                    # A turn recovered without its assistant text still writes
                    # both clips, so the one-to-one check above passes.
                    # Complete evidence requires the spoken response to have
                    # text, and this keeps a degraded turn from being mistaken
                    # for a complete one.
                    missing.append("transcript turns with recorded assistant text")
        except (OSError, ValueError):
            missing.append("valid transcript.json")
    for path in [root / name for name in required[:3]] + sorted(root.glob("turn-*.wav")):
        if path.is_file() and not _is_pcm_s16le_8000_mono(path):
            missing.append(f"{path.name} pcm_s16le_8000_mono")
    return sorted(set(missing))


def _turn_is_degraded(turn: dict[str, Any]) -> bool:
    """A turn that declares itself incomplete, or names an empty response.

    Only turns using the recorder's ``user``/``assistant`` shape are judged
    here, so an alternative legacy turn shape is not newly rejected.
    """

    if turn.get("degraded") is True:
        return True
    return "assistant" in turn and not str(turn.get("assistant") or "").strip()


def _validate_pr79_artifact_set(
    root: Path, *, require_semantic_turns: bool = True
) -> None:
    gaps = _pr79_artifact_gaps(root, require_semantic_turns=require_semantic_turns)
    if gaps:
        raise IncompleteArtifactEvidence("missing PR #79 evidence: " + ", ".join(gaps))


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


#: Raw bytes per artifact-upload request. Small enough that no plausible
#: ingress body limit or slow uplink turns one request into the failure, and
#: well under the API's own per-chunk cap (8 MiB by default).
ARTIFACT_CHUNK_BYTES = 4 * 1024 * 1024
#: Tries per request, including the first. Backoff doubles from one second, so
#: a request gives up after ~7 s of waiting plus its own sized timeouts.
ARTIFACT_UPLOAD_ATTEMPTS = 4
_RETRYABLE_UPLOAD_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
#: Refusals that a later attempt can outlive: chunks the API lost, evidence
#: the retrying process cannot see yet, and an API that has not shipped the
#: upload session yet (the rolling-release window).
_RETRYABLE_UPLOAD_REASONS = frozenset({"upload_incomplete", "evidence_unavailable", "legacy_body_too_large"})
#: ``SKYSAY_MAX_ARTIFACT_UPLOAD_BODY_BYTES``'s default on the API, which no
#: deployment overrides: the most an API without upload sessions accepts.
LEGACY_ARTIFACT_BODY_CAP_BYTES = 32 * 1024 * 1024
_UPLOAD_ID = re.compile(r"^aup_[0-9a-f]{48}$")


def artifact_upload_timeout(body_bytes: int, *, floor_seconds: float = 30.0) -> float:
    """A request timeout that grows with what the request moves.

    Sized for a slow 256 KiB/s path. A finalize passes the whole bundle size,
    because the API copies every part into object storage before answering.
    Capped at ten minutes: past that the request is not slow, it is lost.
    """
    return min(600.0, floor_seconds + body_bytes / (256 * 1024))


def _upload_step(
    send: Callable[[], dict[str, Any]],
    *,
    stage: str,
    total_bytes: int,
    upload_id: str,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """One upload request with bounded, backed-off retries of transient failures.

    Retried: a transport error (connection reset, timeout, truncated read) and
    the HTTP statuses that mean "not now". Everything else -- a 4xx refusal, a
    malformed answer -- is final at once, because repeating it cannot help.

    The failure is raised ``from None``: the underlying rejection's message
    carries up to 500 characters of the API's response body, and an error
    report walks the whole exception chain. Everything worth reporting is
    already on ``ArtifactUploadFailed`` as fixed-vocabulary fields.
    """
    import http.client

    for attempt in range(1, ARTIFACT_UPLOAD_ATTEMPTS + 1):
        try:
            return send()
        except ControlPlaneRejected as exc:
            if exc.route_missing and stage == "begin":
                raise
            retryable = exc.status_code in _RETRYABLE_UPLOAD_STATUSES
            if not retryable or attempt == ARTIFACT_UPLOAD_ATTEMPTS:
                raise ArtifactUploadFailed(
                    f"artifact upload {stage} failed: HTTP {exc.status_code} {exc.reason or 'rejected'}",
                    stage=stage,
                    reason=exc.reason or ("http_" + str(exc.status_code) if exc.status_code else "malformed_response"),
                    status_code=exc.status_code,
                    total_bytes=total_bytes,
                    upload_id=upload_id,
                ) from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # ValueError is an undecodable response body; it is final.
            if isinstance(exc, ValueError) or attempt == ARTIFACT_UPLOAD_ATTEMPTS:
                raise ArtifactUploadFailed(
                    f"artifact upload {stage} failed: {type(exc).__name__}",
                    stage=stage,
                    reason="malformed_response" if isinstance(exc, ValueError) else "transport_error",
                    total_bytes=total_bytes,
                    upload_id=upload_id,
                ) from None
        sleep(float(2 ** (attempt - 1)))
    raise AssertionError("unreachable")  # pragma: no cover


def upload_artifact_bundle(
    transport: Any,
    call_id: str,
    parts: list[tuple[str, bytes]],
    *,
    on_existing: str = "replace",
    chunk_bytes: int = ARTIFACT_CHUNK_BYTES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Store one call's evidence bundle through the chunked upload session.

    ``begin`` declares every part's size and sha256, each part then travels
    as ``chunk_bytes``-sized raw bodies, and ``finalize`` imports the whole
    bundle atomically. Every request is small and has a timeout sized to what
    it carries, so the length of the call no longer decides whether its
    evidence survives.

    An API that predates the session routes answers ``begin`` with the
    router's unknown-path 404. Only then does this fall back to the single
    ``POST /artifacts`` request, which is what that API understands; any other
    refusal is final and raises ``ArtifactUploadFailed``.
    """
    call_path = f"/v1/calls/{urllib.parse.quote(call_id, safe='')}"
    total_bytes = sum(len(content) for _name, content in parts)
    manifest = {
        "parts": [
            {"filename": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in parts
        ],
        "chunk_bytes": chunk_bytes,
        "on_existing": on_existing,
    }

    def send_json(path: str, payload: dict[str, Any], moves_bytes: int = 0) -> Callable[[], dict[str, Any]]:
        body = json.dumps(payload, sort_keys=True).encode()
        timeout = artifact_upload_timeout(len(body) + moves_bytes, floor_seconds=60.0 if moves_bytes else 30.0)
        return lambda: transport.request_bytes(
            "POST", path, body, content_type="application/json", timeout_seconds=timeout
        )

    try:
        begun = _upload_step(
            send_json(f"{call_path}/artifact-uploads", manifest),
            stage="begin",
            total_bytes=total_bytes,
            upload_id="",
            sleep=sleep,
        )
    except ControlPlaneRejected as exc:
        if not exc.route_missing:
            raise
        if on_existing == "refuse":
            # The single request cannot say "do not supersede", so falling back
            # would silently drop the only thing the caller asked to protect.
            raise ArtifactUploadFailed(
                "the API has no upload sessions, and only they can refuse to supersede a recording",
                stage="legacy",
                reason="refuse_needs_upload_sessions",
                total_bytes=total_bytes,
            ) from None
        return _legacy_single_request_upload(transport, call_path, parts, total_bytes=total_bytes, sleep=sleep)
    upload_id = str(begun.get("upload_id") or "")
    if not _UPLOAD_ID.match(upload_id):
        raise ArtifactUploadFailed(
            "artifact upload begin returned no upload id",
            stage="begin",
            reason="malformed_response",
            total_bytes=total_bytes,
        )
    session_path = f"{call_path}/artifact-uploads/{upload_id}"

    def send_chunks() -> None:
        for index, (_name, content) in enumerate(parts):
            for offset in range(0, len(content), chunk_bytes):
                chunk = content[offset : offset + chunk_bytes]
                digest = hashlib.sha256(chunk).hexdigest()
                _upload_step(
                    lambda chunk=chunk, digest=digest, index=index, offset=offset: transport.request_bytes(
                        "POST",
                        f"{session_path}/parts/{index}?offset={offset}",
                        chunk,
                        content_type="application/octet-stream",
                        timeout_seconds=artifact_upload_timeout(len(chunk)),
                        headers={"X-Content-SHA256": digest},
                    ),
                    stage="chunk",
                    total_bytes=total_bytes,
                    upload_id=upload_id,
                    sleep=sleep,
                )

    def finalize() -> dict[str, Any]:
        return _upload_step(
            send_json(f"{session_path}/finalize", {}, moves_bytes=total_bytes),
            stage="finalize",
            total_bytes=total_bytes,
            upload_id=upload_id,
            sleep=sleep,
        )

    if begun.get("status") != "finalized":
        try:
            send_chunks()
        except ArtifactUploadFailed as exc:
            # Another upload of this byte-identical bundle (an operator's
            # recovery, say) finalized while these chunks were in flight. The
            # bundle is stored; finalize answers from its receipt.
            if exc.reason != "upload_finalized":
                raise
    try:
        return finalize()
    except ArtifactUploadFailed as exc:
        # The API restarted or lost staged chunks between our acknowledged
        # sends and the finalize. Every chunk is idempotent, so send them all
        # again once; a second incomplete answer is final.
        if exc.reason != "upload_incomplete":
            raise
    try:
        send_chunks()
    except ArtifactUploadFailed as exc:
        # A concurrent finalize of this same upload won in between; its
        # receipt is the answer, and finalize reads it.
        if exc.reason != "upload_finalized":
            raise
    return finalize()


def _legacy_single_request_upload(
    transport: Any,
    call_path: str,
    parts: list[tuple[str, bytes]],
    *,
    total_bytes: int,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """The pre-session single request, for an API that has no session routes.

    Still bounded by that API's body cap -- a long call fails here exactly as
    it always did -- but with a timeout sized to the body and the same retry
    and failure classification as the session path.
    """
    body = json.dumps(
        {"parts": [_upload_part(name, content) for name, content in parts]}, sort_keys=True
    ).encode()
    if len(body) > LEGACY_ARTIFACT_BODY_CAP_BYTES:
        # That API refuses it by closing the connection mid-send, which reads
        # as a transport error and would be retried -- four times the whole
        # body -- for a verdict that is already known.
        raise ArtifactUploadFailed(
            "artifact bundle exceeds the single-request body cap of an API without upload sessions",
            stage="legacy",
            reason="legacy_body_too_large",
            total_bytes=total_bytes,
        )
    result = _upload_step(
        lambda: transport.request_bytes(
            "POST",
            f"{call_path}/artifacts",
            body,
            content_type="application/json",
            timeout_seconds=artifact_upload_timeout(len(body), floor_seconds=60.0),
        ),
        stage="legacy",
        total_bytes=total_bytes,
        upload_id="",
        sleep=sleep,
    )
    # The same top-level verdict the session's finalize answers with, so no
    # caller has to know which path carried the bundle.
    outcome = (result.get("calls") or [{}])[0]
    return {**result, "imported": bool(outcome.get("imported")), "reason": str(outcome.get("reason") or "")}


def report_artifact_upload_failure(transport: Any, call_id: str, failure: ArtifactUploadFailed) -> bool:
    """Best effort: put the final upload verdict on the call's own timeline."""
    try:
        transport.request(
            "POST",
            f"/v1/calls/{urllib.parse.quote(call_id, safe='')}/artifact-upload-failures",
            {
                "reason": failure.reason,
                "stage": failure.stage,
                "status_code": failure.status_code,
                "total_bytes": failure.total_bytes,
                "upload_id": failure.upload_id or None,
            },
        )
    except Exception:  # noqa: BLE001 - the report must never mask the failure it describes
        return False
    return True


def recover_call_artifacts(
    transport: Any,
    call_id: str,
    evidence_dir: Path,
    *,
    replace: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Re-upload a call's retained evidence directory (operator recovery).

    For a call whose worker could not store its bundle. Idempotent: running it
    again for the same directory reaches the same upload, which the API
    answers from its receipt without importing or notifying twice. A call
    that already HAS a recording is refused unless ``replace`` is set,
    because a recovered bundle is not byte-identical to one the worker
    stored (the recorder's semantic verdict is gone with the worker) and
    would otherwise supersede it.

    The duration sources all come from ``mixed.wav``, exactly what the
    worker itself reports. Billing was settled from the gateway lifecycle
    when the call ended; the settle the import runs is idempotent, so a
    recovered bundle does not bill the call a second time.
    """
    mixed = evidence_dir / "mixed.wav"
    try:
        with wave.open(str(mixed), "rb") as recording:
            duration = recording.getnframes() / recording.getframerate()
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        raise ArtifactUploadFailed(
            "the evidence directory's mixed.wav cannot be read",
            stage="begin",
            reason="evidence_unreadable",
        ) from None
    bundle = collect_artifact_bundle(
        evidence_dir,
        duration_sources={"sip_participant": duration, "room_track": duration, "recording": duration},
    )
    try:
        imported = upload_artifact_bundle(
            transport,
            call_id,
            bundle.parts,
            on_existing="replace" if replace else "refuse",
            sleep=sleep,
        )
    except ArtifactUploadFailed as exc:
        if exc.reason != "call_already_has_recording":
            report_artifact_upload_failure(transport, call_id, exc)
        raise
    return {
        "call_id": call_id,
        "imported": bool(imported.get("imported")),
        "reason": str(imported.get("reason") or ""),
        "upload_id": str(imported.get("upload_id") or ""),
        "part_count": len(bundle.parts),
        "total_bytes": sum(len(content) for _name, content in bundle.parts),
        "evidence_status": bundle.evidence["status"],
    }


#: How long a failed upload waits before each retry, after the worker's own
#: attempt: 1 min, 5 min, 15 min, 1 h, 4 h. About 5.3 hours end to end, which
#: outlasts a control-plane deploy or an outage of the object store, and ends
#: days before host retention deletes the evidence the retry reads.
ARTIFACT_RETRY_DELAYS_SECONDS = (60, 300, 900, 3600, 14400)

#: The ``artifact_status`` values a session-state file carries for an upload
#: that failed. ``upload_pending`` is the only one a sweep acts on.
ARTIFACT_UPLOAD_PENDING = "upload_pending"
ARTIFACT_UPLOAD_ABANDONED = "upload_abandoned"


def next_artifact_attempt_at(attempts_made: int, now: datetime) -> str | None:
    """When the next upload attempt is due, or ``None`` once the bound is spent."""
    index = attempts_made - 1
    if index < 0 or index >= len(ARTIFACT_RETRY_DELAYS_SECONDS):
        return None
    due = now + timedelta(seconds=ARTIFACT_RETRY_DELAYS_SECONDS[index])
    return due.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def artifact_retry_params(
    *,
    duration_sources: dict[str, float],
    require_semantic_turns: bool,
    semantic_evidence_status: str | None,
) -> dict[str, Any]:
    """What a later retry needs to rebuild the SAME bundle the worker built.

    Persisted in the session-state file. Byte-identical bundles reach the same
    upload id, so a retry reuses what an earlier attempt already staged and a
    bundle that was in fact imported answers from its receipt.
    """
    return {
        "duration_sources": {name: float(value) for name, value in duration_sources.items()},
        "require_semantic_turns": bool(require_semantic_turns),
        "semantic_evidence_status": semantic_evidence_status,
    }


def retry_call_artifact_upload(
    transport: Any,
    call_id: str,
    evidence_dir: Path,
    params: Mapping[str, Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """One scheduled retry of a worker's failed upload, from its evidence dir."""
    if not evidence_dir.is_dir():
        raise ArtifactUploadFailed(
            "the call's evidence directory is not visible to this process",
            stage="begin",
            reason="evidence_unavailable",
        )
    try:
        bundle = collect_artifact_bundle(
            evidence_dir,
            duration_sources=dict(params.get("duration_sources") or {}),
            require_semantic_turns=bool(params.get("require_semantic_turns", True)),
            semantic_evidence_status=params.get("semantic_evidence_status") or None,
        )
    except (OSError, IncompleteArtifactEvidence):
        # A local read that failed, or a directory that is momentarily empty:
        # the next attempt can see it, so it is retried, never abandoned here.
        raise ArtifactUploadFailed(
            "the call's evidence could not be read",
            stage="begin",
            reason="evidence_unavailable",
        ) from None
    return upload_artifact_bundle(transport, call_id, bundle.parts, sleep=sleep)


def _without_session_duration(evidence: dict[str, Any]) -> dict[str, Any] | None:
    """``evidence`` minus every record's ``session_duration``; None if it had none."""
    records = evidence.get("records")
    if not isinstance(records, list) or not any(
        isinstance(record, dict) and "session_duration" in record for record in records
    ):
        return None
    return {
        **evidence,
        "records": [
            {key: value for key, value in record.items() if key != "session_duration"}
            if isinstance(record, dict)
            else record
            for record in records
        ],
    }

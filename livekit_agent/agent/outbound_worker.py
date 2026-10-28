# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Mapping

try:
    from ..control_plane import WORKER_CONTRACT, ControlPlaneRejected, Gate2ControlPlaneAdapter, UrllibJsonTransport, adopt_deployment_content_storage, content_kept_on_gateway
    from ..error_tracking import capture_exception, install_error_tracking
    from ..production import terminate_pre_media_call
    from ..worker_drain import install_stop_handlers, sleep_unless_stopped
    from ..prewarm import PHONE_AGENT_CAPACITY_VAR, resolve_phone_agent_capacity
    from ..phone_admission import CAPACITY_REFUSED_KEY
    from ..agents_readiness import READINESS_DIR_ENV, UNAVAILABLE_AFTER_SECONDS, read_readiness
except ImportError:  # flat /app import in the worker image
    from control_plane import WORKER_CONTRACT, ControlPlaneRejected, Gate2ControlPlaneAdapter, UrllibJsonTransport, adopt_deployment_content_storage, content_kept_on_gateway
    from error_tracking import capture_exception, install_error_tracking
    from production import terminate_pre_media_call
    from worker_drain import install_stop_handlers, sleep_unless_stopped
    from prewarm import PHONE_AGENT_CAPACITY_VAR, resolve_phone_agent_capacity
    from phone_admission import CAPACITY_REFUSED_KEY
    from agents_readiness import READINESS_DIR_ENV, UNAVAILABLE_AFTER_SECONDS, read_readiness


AGENT_NAME = "skysay-voice-c3"
OUTBOUND_TRUNK_NAME = "skysay-selfhost-outbound"
DISPATCH_SCHEMA = "skysay.outbound-dispatch.v1"

# A create_sip_participant failure's SIP 487 is trusted as our OWN
# ringing_timeout elapsing (a no-answer outcome) only when the wall-clock
# time since the INVITE was sent is within this many seconds of the
# configured window -- an abort at any other point (room deleted by a
# customer-initiated cancel, or another server-side teardown) gets the same
# 487 response but is not a no-answer outcome (Codex review).
_RINGING_TIMEOUT_ELAPSED_SLACK_SECONDS = 5

# A private alias, not a bare ``time.monotonic()`` call: ``time`` is the real
# process-wide stdlib module asyncio's own event loop also reads its clock
# from, so a test patching ``time.monotonic`` directly to script the SIP
# dial's elapsed duration would starve the loop of a real clock and hang.
# Patching this module-level name instead leaves the real clock alone.
_monotonic = time.monotonic


class OutboundWorkerError(RuntimeError):
    pass


#: The control plane's two worker-contract refusals: this worker may not run
#: this call (it did not present contract 3, or the call carries no contract-3
#: snapshot). Neither changes with a retry, so the job fails once.
CONTRACT_REFUSALS = frozenset({"worker_contract_required", "call_contract_missing"})


def contract_refusal(exc: BaseException) -> str:
    """The contract refusal reason an exception carries, or ``""``."""
    for candidate in (exc, exc.__cause__):
        reason = str(getattr(candidate, "reason", "") or "")
        if isinstance(candidate, ControlPlaneRejected) and reason in CONTRACT_REFUSALS:
            return reason
    return ""


class OutboundPostAdmissionError(OutboundWorkerError):
    """A terminal outcome after the outbound authorization may have dialed."""

    outbound_retry_safe = False


def _required(environ: Mapping[str, str], name: str) -> str:
    value = str(environ.get(name) or "").strip()
    if not value:
        raise OutboundWorkerError(f"missing required setting {name}")
    return value


def _optional(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name) or "").strip()


def _enabled(environ: Mapping[str, str], name: str) -> bool:
    return str(environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_id(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in "-_")[:96]


def _redact_phone(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    return "..." + digits[-4:] if len(digits) >= 4 else "[redacted]"


# The text a refusal carries is written by the far end, not by us. Bound it
# hard: one hostile or malfunctioning carrier must not be able to write an
# unbounded string into the event row and job record of every call it refuses.
_SIP_REASON_MAX_CHARS = 120
_SIP_ERROR_CODE_MAX_CHARS = 40

# Seven or more digits, counted ACROSS the separators a carrier writes a
# number with, because "+94 77 123 4567" contains no run of seven on its own
# (Codex review). That many digits is an identifier -- a destination number, an
# account reference -- never part of a SIP reason phrase or an error vocabulary
# word. The call already records its own numbers in their own fields; a second,
# unnormalised copy inside free text is not worth carrying. A separator class
# this narrow leaves a port, a SIP status, and a hex trunk id alone: a letter
# ends the run.
#
# The dot is NOT a separator here, so an address survives: "dial tcp
# 203.0.113.10:5060: connect: connection refused" is exactly the shape a
# refusal that never became a SIP response carries, and WHICH endpoint refused
# is recorded nowhere else on a pre-media failure -- no span, no carrier call
# id (Kimi review). The cost is that a dot-separated number survives, and that
# trade is one-sided: that number is already in the call's own fields.
_PHONE_LIKE_DIGITS = re.compile(r"\d(?:[\s\-()]*\d){6,}")

# `user@host` / `user:password@host` inside a URI the far end echoed back. The
# host is the diagnostic half; the userinfo is the half that can be a trunk
# credential, so it is removed and the host kept.
#
# The halves stop at whitespace on purpose. Letting the password half span
# spaces (`(?::[^@]*)?`) would catch a malformed userinfo containing a raw
# space -- which RFC 3261 does not allow -- at the cost of eating everything
# between any `scheme:` and any later `@`: measured, it turns "sip: dial
# failed to user@sip.didlogic.net" into "sip.didlogic.net" and deletes the
# whole diagnosis. Losing a real message to protect against an invalid one is
# the worse trade; the credential-name rule below and the length bound are
# what cover the remainder.
_URI_USERINFO = re.compile(r"[^\s:@/]+(?::[^\s@/]*)?@")

# A credential announced by name outside any URI ("password hunter2"). The
# NAME is kept and EVERYTHING after it dropped -- not just the next token,
# because "password is hunter2" and "token set to abc" put the value one word
# further along and a rule that only ate the next token would leak exactly the
# thing it exists to catch (Kimi review). Once one of these words appears the
# rest of the string is suspect, and the word itself is the diagnosis worth
# keeping.
#
# The name is matched with NO leading word boundary, only a trailing one, so a
# prefixed or spaced spelling cannot slip past: `client_secret`, `api_key`,
# `api-key`, `apikey` and `api key` all land on one of these words, and
# `secretary` and `keyword` still do not because the boundary after must hold
# (Codex review found the leading `\b` let the first two through).
#
# `authorization` is in the list but no broader `auth` stem is: "401
# Unauthorized" and "407 Proxy Authentication Required" are real SIP reason
# phrases, and they are exactly the evidence a trunk-credential problem
# leaves -- blanking them would suppress the answer to one of the three
# questions this record exists to answer. Neither spells "authorization", so
# the header name can be caught without touching them.
#
# `bearer` is NOT here, and that is a correction rather than an omission: Q.850
# causes 57/58/65 are "Bearer capability not authorized / not presently
# available / not implemented", which SIP gateways copy into reason phrases,
# and which are exactly the trunk-capability evidence this record exists to
# keep (Kimi review). The scheme rule below covers it instead, by its value.
# `username` earns its place beside `password`: a trunk authenticates with an
# `auth_username`/`auth_password` pair, and a proxy that names the rejected
# username outside a URI would otherwise write the very half `_URI_USERINFO`
# above exists to strip (Kimi review). It costs no real phrase -- "User busy"
# and "No user responding" do not contain it, which bare `user` would have.
# The second branch catches a name carrying a SUFFIX -- `passwordHash=...`,
# `clientSecretValue=...`, `authTokenValue=...` -- which the trailing boundary
# alone rejected, because the suffix keeps the boundary from landing (Codex
# review). It is admitted only when an assignment follows, which is what
# separates a labelled credential from prose: `secretary said no` and `keyword
# lookup failed` have the same stems and no `=` or `:` after them, so they are
# still left alone.
_NAMED_CREDENTIAL = re.compile(
    r"(?i)(password|passwd|passphrase|username|secret|credential|cred|key|token"
    r"|authori[sz]ation)(?:s?\b|[a-z]*\s*(?=[:=])).*"
)

# An auth scheme followed by anything that is not an ordinary word. `Authori-
# zation: Basic <base64>` was surviving every other rule, because the header
# name was excluded and none of the names above appear in it (Codex review).
#
# What follows such a scheme is either telephony prose or a credential; there
# is no third case. So the value is redacted UNLESS it is recognisably a word,
# which is the only direction that fails closed against a credential shape
# nobody predicted. Two earlier attempts at the opposite direction both
# leaked: requiring the token to look like base64 (a digit or `+/=`) let
# `Basic dGVzdHVzZXI` and `Bearer abcdefghijklmnop` through, and a minimum
# length let `Basic YTpi` -- base64 of "a:b" -- through.
#
# "A word" is a lower-case run of ordinary length, or one of the longer
# telephony words these schemes really precede. Either must be FOLLOWED by
# whitespace or the end, so a token that merely opens with one of them
# (`accessKeyId=...`, `callH7x9K2mPqRsT`) is not mistaken for it -- a trailing
# word boundary alone let those through (Kimi review). Case matters, so the
# word test is not folded: `YTpi` is a credential, `capability` is prose.
#
# The residual is a token that IS an ordinary short lower-case word, such as
# `Bearer abcdef`. Closing it needs a complete allowlist of reason phrases,
# which carriers paraphrase, and would blank the Q.850 causes on every
# paraphrase it missed -- the diagnostic this record exists to keep. A
# six-character lower-case bearer token is also not a usable credential, and
# such a string can only reach the operator-only sink anyway: a SIP response's
# reason phrase, the one field a caller reads, does not carry auth headers.
# The rules stop here deliberately.
_AUTH_SCHEME_VALUE = re.compile(
    r"\b((?i:basic|digest|bearer|ntlm|negotiate))\s+"
    r"(?!(?:(?i:capabilit|servic|call|rate|channel|connect|authenticat|access)[A-Za-z]*"
    r"|[a-z]{1,12})(?:\s|$))"
    r"\S+.*"
)

# The dialled URI echoed back with its userinfo but with the `@` out of reach
# -- beyond the examined window, or absent because the far end truncated it.
# Anchored on the scheme instead, so it needs no `@` at all (Kimi review). Two
# things that are NOT userinfo are kept, because which host and port a dial was
# refused on is diagnostic: a purely numeric second part is a port, and a
# bracketed first part is an IPv6 literal host (RFC 3261 allows one, and
# `sip:[2001:db8::1]:5060` otherwise reads as a user/password pair).
_URI_SCHEME_USERINFO = re.compile(r"(?i)\b(sips?|tel):[^\s:@/\[]+:(?!\d+\b)[^\s@/]+")

# How much of an untrusted string is examined at all. Generous rather than
# tight: the substitutions below need to SEE a construct to strip it, and a
# `user:password@host` whose `@` fell beyond this window would keep its
# password (Kimi review). Everything surviving is truncated to the caller's
# much smaller limit afterwards, so this bounds regex work, not output.
_REFUSAL_INPUT_MAX_CHARS = 2_000


def _bounded_refusal_text(value: Any, limit: int) -> str:
    """Make one untrusted refusal string safe to persist, or return ''.

    The input slice is capped BEFORE the substitutions below so the regex work
    stays linear in a constant, whatever the far end sent.
    """

    if not isinstance(value, str):
        return ""
    text = "".join(
        character if character.isprintable() else " "
        for character in value[:_REFUSAL_INPUT_MAX_CHARS]
    )
    text = _URI_USERINFO.sub("", text)
    text = _URI_SCHEME_USERINFO.sub(r"\1: ...", text)
    text = _NAMED_CREDENTIAL.sub(r"\1 ...", text)
    text = _AUTH_SCHEME_VALUE.sub(r"\1 ...", text)
    text = _PHONE_LIKE_DIGITS.sub("...", text)
    return " ".join(text.split())[:limit]


def _declared(exc: BaseException, name: str) -> Any:
    """One declared attribute of a client error, or None if reading it raises.

    These are PROPERTIES computed from the server's own response metadata, so
    a malformed response can make reading one raise rather than return. This
    runs inside the failure path that reports the call and releases its
    reservation: an attribute that raises must cost the evidence, never the
    teardown, or a failed call becomes a stuck one holding a wallet reserve.
    """

    try:
        return getattr(exc, name, None)
    except Exception:
        return None


def _sip_refusal_evidence(exc: BaseException | None) -> dict[str, Any]:
    """What the SIP client itself said about a refusal, bounded and structured.

    Only the fields the SIP client DECLARES are read -- never ``str(exc)`` and
    never ``.args``, so an arbitrary exception's own prose can never reach a
    call record, and an exception that reached no SIP service at all yields
    nothing at all rather than something invented.

    Two shapes exist and both matter. When the far end answered the INVITE
    with a real SIP response, the client raises a SIP-specific error carrying
    ``sip_status_code`` and the response's reason phrase. When no SIP response
    was ever received -- no route, no free channel, the media plane refusing
    the request outright -- the client raises its ordinary server error, which
    carries no SIP fields at all: only its own error code and message. That
    second shape is the one an outbound failure used to record NOTHING about,
    which left "the SIP dial raised something" as the whole diagnosis for the
    class of failure that needs a diagnosis most.

    Attribute reads rather than an isinstance check on the client's classes:
    this must work identically against a test double that only shapes the
    attributes, and every field is type-guarded because an unrelated exception
    can carry a same-named attribute of a different type (an HTTP client whose
    ``.code`` is an int status, say) -- which is dropped, deliberately.
    """

    if exc is None:
        return {}
    evidence: dict[str, Any] = {}
    sip_status_code = _declared(exc, "sip_status_code")
    if isinstance(sip_status_code, int) and not isinstance(sip_status_code, bool):
        evidence["sip_status_code"] = sip_status_code
    # The SIP response's own reason phrase: "Busy Here", "Not Found", or
    # whatever text the carrier chose. A protocol element the far end sent
    # about this call, so it is safe for the caller to read.
    reason = _bounded_refusal_text(_declared(exc, "sip_status"), _SIP_REASON_MAX_CHARS)
    if reason:
        evidence["sip_reason"] = reason
    # The client's own message, which is where a refusal that never became a
    # SIP response says what actually happened. Kept under a SEPARATE key
    # because it is our media plane's prose about our own infrastructure --
    # "dial tcp 203.0.113.10:5060: connect: connection refused" names a host
    # the caller has no relationship with -- and only one of these two sinks
    # may carry that. See ``_tenant_readable`` below.
    detail = _bounded_refusal_text(_declared(exc, "message"), _SIP_REASON_MAX_CHARS)
    if detail:
        evidence["transport_detail"] = detail
    # Deliberately NOT named `sip_*`: this is the media plane client's own
    # transport-error vocabulary (`unavailable`, `not_found`,
    # `invalid_argument`, ...), not a SIP protocol element, and an operator
    # reading it beside `sip_status_code` must not confuse the two. It is a
    # small closed vocabulary, which makes it the one field a refusal without
    # any SIP response can still be classified by.
    error_code = _bounded_refusal_text(_declared(exc, "code"), _SIP_ERROR_CODE_MAX_CHARS)
    if error_code:
        evidence["livekit_error_code"] = error_code
    return evidence


# Evidence only an operator may read. The two sinks have different audiences:
# a `call.failed` event is readable by the tenant on their own call, while
# `worker_jobs.last_error` is not -- `worker_jobs:read` is deliberately kept
# out of the customer-grantable scope set, pinned by a tenant-isolation test.
# The repository's review contract keeps provider internals and raw backend
# failure text out of anything a customer can read (Codex review), so the
# media plane's own prose goes to the operator sink alone.
_OPERATOR_ONLY_EVIDENCE = frozenset({"transport_detail"})


def _tenant_readable(evidence: dict[str, Any]) -> dict[str, Any]:
    """The subset of a refusal's evidence the caller may read on their call.

    The SIP status code and reason phrase are protocol elements the far end
    sent about THIS call, and the error code is a closed vocabulary naming a
    class of failure -- all three describe the caller's own call and say
    nothing about how it was carried. The transport message is the exception:
    it is our media plane's internal failure text, naming our infrastructure.
    """

    return {
        key: value for key, value in evidence.items() if key not in _OPERATOR_ONLY_EVIDENCE
    }


def reported_failure_details(evidence: dict[str, Any], stage: str) -> dict[str, Any]:
    """The pre-media failure details this deployment reports to the control plane.

    A deployment that keeps call content on the gateway reports machine values
    only: the carrier's free-text reason phrase stays in the local log.
    """
    readable = _tenant_readable(evidence)
    if content_kept_on_gateway():
        readable = {key: value for key, value in readable.items() if key != "sip_reason"}
    return {**readable, "failure_evidence_version": 1, "failure_stage": stage}


#: How many claimed jobs one originator may be dialling at once: the phone
#: host's measured call capacity (``SKYSAY_PHONE_AGENT_CAPACITY``), the same
#: value the phone agents admit by (``phone_admission``) and keep at least that
#: many warm processes for. Admission is decided
#: when a job is ENQUEUED (plan capacity, the organization's execution slots,
#: per-number caps, a campaign's own max_concurrency), so this never widens
#: what may dial; it only stops one worker from serialising what was already
#: admitted. Absent, 1 keeps the original one-at-a-time loop exactly.
MAX_IN_FLIGHT_VAR = PHONE_AGENT_CAPACITY_VAR
#: A hosted claim's lease is 300 s server-side. A job renews it on this cadence
#: for as long as it is unsettled, so settlement never meets a lease that ran
#: out while the call rang or its report was retried.
LEASE_RENEW_SECONDS = 60.0
#: A dialled call's completion report is retried, never turned into a failure:
#: the SIP leg exists, so failing the job would fail a live call and release
#: its reservation. Delays between attempts, in seconds.
COMPLETE_RETRY_DELAYS = (2.0, 5.0, 10.0)
#: How long a dialled call waits for its own agent to JOIN the room before the
#: SIP leg is placed. The phone agent joins right after it accepts the dispatch
#: (``ctx.connect`` is its first step), so this is well under a second when a
#: worker took the job. When no worker takes it -- the gate refused it, no
#: worker is registered, the job crashed on start -- nothing is dialled: the
#: call fails before any leg exists, instead of being answered into a room
#: nobody is in. That was the worst outcome a customer could get.
AGENT_JOIN_TIMEOUT_SECONDS = 10.0
AGENT_JOIN_POLL_SECONDS = 0.2
#: ``livekit.protocol.models.ParticipantInfo.Kind.AGENT``.
_AGENT_PARTICIPANT_KIND = 4


#: Each LiveKit API call before the dial gets its own deadline. livekit-api
#: sends every request with no timeout of its own, so a request LiveKit accepts
#: and never answers would otherwise hold its call -- and in organization mode,
#: which has no overall originate deadline, the drain -- forever.
LIVEKIT_RPC_TIMEOUT_SECONDS = 10.0
#: The longest a concurrent originator waits for one job's agent before it
#: claims the next anyway: that job's control-plane calls (lease renewal,
#: admission pair, media admission; 10 s each), its three LiveKit calls, and
#: its join wait, with room to spare. Every one of those is bounded, so this
#: is a backstop: no single job can stop an originator claiming.
CLAIM_PACING_CAP_SECONDS = 4 * 10 + 3 * LIVEKIT_RPC_TIMEOUT_SECONDS + AGENT_JOIN_TIMEOUT_SECONDS + 10


#: Every bounded step of an originate before its dial: the admission and
#: media-admission control-plane calls (10 s each), the three LiveKit calls,
#: and the wait for the call's own agent to join.
PRE_DIAL_BOUND_SECONDS = 2 * 10 + 3 * LIVEKIT_RPC_TIMEOUT_SECONDS + AGENT_JOIN_TIMEOUT_SECONDS
#: The hosted originate's own deadline outlasts the dial's by this much, so a
#: callee who never answers is ended by the dial's own ring timer and recorded
#: as a ring timeout, never cut off by this deadline first as a rejection.
ORIGINATE_DEADLINE_MARGIN_SECONDS = 10


def _originate_deadline_seconds(settings: "OutboundWorkerSettings") -> float:
    """The pre-dial steps, then the dial with its participant timeout, then the margin.

    The join wait has its own term here: it never eats into the dial's time.
    """

    return PRE_DIAL_BOUND_SECONDS + settings.ringing_timeout_seconds + 10 + ORIGINATE_DEADLINE_MARGIN_SECONDS


async def _livekit_rpc(awaitable: Awaitable[Any]) -> Any:
    return await asyncio.wait_for(awaitable, timeout=LIVEKIT_RPC_TIMEOUT_SECONDS)


class AgentNeverJoinedError(OutboundWorkerError):
    """The call's agent did not join its room in time, so the leg was never placed."""


class PhoneCapacityRefusedError(OutboundWorkerError):
    """The phone agents refused this call's dispatch for capacity; nothing was dialled.

    Not a failure of the call: the job goes back to the queue.
    Raised only from the join wait, which is before the SIP participant is
    created -- the proof, for the control plane's ``defer``, that no leg
    exists.
    """

    outbound_retry_safe = True


def _capacity_refused(metadata: str) -> bool:
    try:
        parsed = json.loads(metadata) if metadata else {}
    except ValueError:
        return False
    return isinstance(parsed, dict) and bool(parsed.get(CAPACITY_REFUSED_KEY))


async def _wait_for_agent_join(client: Any, api_module: Any, room_name: str) -> None:
    """Return once an agent participant is in ``room_name``; raise if none joins in time.

    Also raises, at once, when the agents marked the room refused for
    capacity (``phone_admission.CAPACITY_REFUSED_KEY``).
    """

    async def poll() -> None:
        while True:
            response = await client.room.list_participants(api_module.ListParticipantsRequest(room=room_name))
            for participant in getattr(response, "participants", ()) or ():
                if getattr(participant, "kind", None) == _AGENT_PARTICIPANT_KIND or str(
                    getattr(participant, "identity", "") or ""
                ).startswith("agent-"):
                    return
            listed = await client.room.list_rooms(api_module.ListRoomsRequest(names=[room_name]))
            for room in getattr(listed, "rooms", ()) or ():
                if getattr(room, "name", "") == room_name and _capacity_refused(str(getattr(room, "metadata", "") or "")):
                    raise PhoneCapacityRefusedError("the phone agents refused this dispatch for capacity; nothing was dialled")
            await asyncio.sleep(AGENT_JOIN_POLL_SECONDS)

    # One deadline over the whole wait, the RPCs included: livekit-api sends
    # each request with no timeout of its own, so a LiveKit that accepts the
    # request and never answers would otherwise hold this call, and every
    # claim waiting for its join, forever. An answer after the deadline is
    # discarded with the request, so it cannot lead to a dial either.
    try:
        await asyncio.wait_for(poll(), timeout=AGENT_JOIN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise AgentNeverJoinedError("the call's agent never joined its room; nothing was dialled") from None


def _max_in_flight(environ: Mapping[str, str]) -> int:
    # The host's declared capacity -- the SAME number the phone agents admit
    # by (prewarm.phone_capacity) -- or one call at a time when none is
    # declared. It used to be clamped one below the agents' job gate, because
    # that gate counted a launching job twice; the agents now admit by an
    # exact count of their own, so the clamp is gone. A dispatch the agents
    # refuse anyway (their capacity is shared with inbound calls) is never
    # dialled: the originator waits for its agent to join.
    return resolve_phone_agent_capacity(environ) or 1


@dataclass(frozen=True)
class OutboundWorkerSettings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    control_plane_url: str
    gateway_id: str
    gateway_token: str
    worker_token: str
    organization_id: str
    worker_id: str
    trunk_name: str
    poll_seconds: float
    ringing_timeout_seconds: int
    max_call_duration_seconds: int
    execute: bool
    worker_mode: str = "organization"
    max_in_flight: int = 1
    #: This host's agents readiness directory (``agents_readiness``). Empty
    #: turns the claim barrier off and claims as before; the managed voice
    #: host sets it, see ``agents_readiness.READINESS_DIR_ENV``.
    agents_readiness_dir: str = ""

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "OutboundWorkerSettings":
        poll_seconds = float(environ.get("SKYSAY_OUTBOUND_POLL_SECONDS") or "2")
        ringing_timeout = int(environ.get("SKYSAY_OUTBOUND_RINGING_TIMEOUT_SECONDS") or "45")
        max_duration = int(environ.get("SKYSAY_OUTBOUND_MAX_CALL_DURATION_SECONDS") or "1800")
        worker_mode = str(environ.get("SKYSAY_OUTBOUND_WORKER_MODE") or "organization").strip()
        if worker_mode not in {"platform", "organization"}:
            raise OutboundWorkerError("outbound worker mode must be platform or organization")
        if poll_seconds < 0:
            raise OutboundWorkerError("SKYSAY_OUTBOUND_POLL_SECONDS must be non-negative")
        if ringing_timeout < 5 or ringing_timeout > 120:
            raise OutboundWorkerError("outbound ringing timeout must be between 5 and 120 seconds")
        if max_duration < 30 or max_duration > 14_400:
            raise OutboundWorkerError("outbound max call duration must be between 30 and 14400 seconds")
        return cls(
            livekit_url=_required(environ, "LIVEKIT_URL"),
            livekit_api_key=_required(environ, "LIVEKIT_API_KEY"),
            livekit_api_secret=_required(environ, "LIVEKIT_API_SECRET"),
            control_plane_url=_required(environ, "SKYSAY_CONTROL_PLANE_URL"),
            gateway_id=_required(environ, "SKYSAY_GATEWAY_ID"),
            gateway_token=_required(environ, "SKYSAY_GATEWAY_API_TOKEN"),
            # Hosted mode uses the dedicated platform_outbound role. Customer
            # queue mode retains its own organization key; local origination
            # needs no queue credential. The mode never grants authority: the
            # server verifies the credential role, target and lease.
            worker_token=_optional(environ, "SKYSAY_WORKER_API_TOKEN"),
            organization_id="" if worker_mode == "platform" else _required(environ, "SKYSAY_ORGANIZATION_ID"),
            worker_id=str(environ.get("SKYSAY_OUTBOUND_WORKER_ID") or "livekit-outbound-1").strip(),
            trunk_name=str(environ.get("SKYSAY_OUTBOUND_TRUNK_NAME") or OUTBOUND_TRUNK_NAME).strip(),
            poll_seconds=poll_seconds,
            ringing_timeout_seconds=ringing_timeout,
            max_call_duration_seconds=max_duration,
            execute=_enabled(environ, "SKYSAY_EXECUTE_LIVEKIT_OUTBOUND"),
            worker_mode=worker_mode,
            max_in_flight=_max_in_flight(environ),
            agents_readiness_dir=str(environ.get(READINESS_DIR_ENV) or "").strip(),
        )


def _gateway_transport(settings: OutboundWorkerSettings) -> UrllibJsonTransport:
    return UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=settings.gateway_token,
    )


def _require_worker_token(settings: OutboundWorkerSettings) -> str:
    if not settings.worker_token:
        raise OutboundWorkerError(
            "the outbound job queue needs SKYSAY_WORKER_API_TOKEN; a deployment that "
            "originates its own calls uses `originate` instead and needs no organization key"
        )
    return settings.worker_token


def _transports(settings: OutboundWorkerSettings) -> tuple[UrllibJsonTransport, UrllibJsonTransport]:
    worker = UrllibJsonTransport(
        base_url=settings.control_plane_url,
        bearer_token=_require_worker_token(settings),
    )
    return _gateway_transport(settings), worker


def _validate_job(settings: OutboundWorkerSettings, job: dict[str, Any]) -> dict[str, str]:
    if job.get("kind") != "call.outbound_originate" or job.get("status") != "running":
        raise OutboundWorkerError("claimed job is not a running outbound originate job")
    if settings.worker_mode == "platform":
        if job.get("claim_target") != "platform" or not job.get("organization_id"):
            raise OutboundWorkerError("hosted originator refuses non-platform work")
        _lease_seconds(settings, job)
    elif str(job.get("organization_id") or "") != settings.organization_id:
        raise OutboundWorkerError("claimed job belongs to a different organization")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise OutboundWorkerError("outbound job payload must be an object")
    fields = {
        "call_id": str(payload.get("call_id") or "").strip(),
        "agent_id": str(payload.get("agent_id") or "").strip(),
        "from_number_id": str(payload.get("number_id") or "").strip(),
        "from_e164": str(payload.get("from_phone_number") or "").strip(),
        "to_e164": str(payload.get("to_phone_number") or "").strip(),
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise OutboundWorkerError(f"outbound job omitted required fields: {', '.join(missing)}")
    if not fields["from_e164"].startswith("+") or not fields["to_e164"].startswith("+"):
        raise OutboundWorkerError("outbound job numbers must use E.164 format")
    gate = payload.get("live_call_gate")
    if not isinstance(gate, dict) or gate.get("enabled") is not True:
        raise OutboundWorkerError("outbound job live-call gate is not enabled")
    return fields


def _lease_seconds(settings: OutboundWorkerSettings, job: dict[str, Any]) -> float:
    if not job.get("lease_token") or job.get("locked_by") != settings.worker_id:
        raise OutboundWorkerError("job has no lease for this worker")
    try:
        deadline = datetime.fromisoformat(str(job.get("lease_expires_at") or "").replace("Z", "+00:00"))
        seconds = (deadline - datetime.now(UTC)).total_seconds()
    except (TypeError, ValueError) as exc:
        raise OutboundWorkerError("invalid worker lease deadline") from exc
    if seconds <= 10:
        raise OutboundWorkerError("worker lease is stale")
    return seconds


def _duration_limits(settings: OutboundWorkerSettings) -> dict[str, Any]:
    from google.protobuf.duration_pb2 import Duration

    return {
        "ringing_timeout": Duration(seconds=settings.ringing_timeout_seconds),
        "max_call_duration": Duration(seconds=settings.max_call_duration_seconds),
    }


async def _select_outbound_trunk(
    client: Any,
    api_module: Any,
    settings: OutboundWorkerSettings,
    from_e164: str,
    *,
    trunk_name: str = "",
) -> Any:
    # Per-number trunk dispatch: the control plane resolves the trunk name
    # from the NUMBER's provider account (BYO SIP trunk carriers record
    # theirs) and threads it through the job payload; the worker's
    # env-configured name stays the default for managed numbers. Caller-ID
    # membership remains a hard filter either way, so a job can never dial
    # out of a trunk that does not carry its number.
    wanted_name = str(trunk_name or "").strip() or settings.trunk_name
    response = await _livekit_rpc(client.sip.list_outbound_trunk(api_module.ListSIPOutboundTrunkRequest()))
    from_digits = from_e164.removeprefix("+")
    matches = [
        trunk
        for trunk in response.items
        if str(getattr(trunk, "name", "")) == wanted_name
        and from_digits in {str(number).removeprefix("+") for number in getattr(trunk, "numbers", ())}
    ]
    if len(matches) != 1:
        raise OutboundWorkerError(
            "expected exactly one configured outbound trunk containing the requested caller ID"
        )
    return matches[0]


async def _originate_admitted_call(
    settings: OutboundWorkerSettings,
    adapter: Any,
    admit: Any,
    *,
    known_call_id: str = "",
    api_module: Any | None = None,
    agent_joined: asyncio.Event | None = None,
    before_dial: Any | None = None,
    defer_capacity_refusal: bool = False,
) -> dict[str, Any]:
    """Dial one call whose ORIGINATION the control plane has already approved.

    ``admit`` is the one step that differs between the two originators: the
    queue path claimed somebody else's job and mints its token from the call it
    was handed, while a locally-originating deployment asks for admission and
    the token together. Everything after it -- spending that token on media
    admission, selecting the trunk that carries the caller ID, the room, the
    explicit agent dispatch, the SIP participant, and the pre-media teardown
    that releases a reservation -- is deliberately ONE body, so the two
    originators cannot drift into two different dialers.

    ``admit`` reports the admitted call id through ``observe_call_id`` as soon
    as it knows one, so a failure between admission and the session is still
    reported against the call that is holding the reservation.
    """
    fields: dict[str, str] = {}
    trunk_name = ""
    resolved_call_id = str(known_call_id or "")
    call: Any | None = None
    client: Any | None = None
    room_name = ""
    participant_identity = ""
    room_cleanup_required = False
    participant_cleanup_required = False
    may_have_originated = False
    stage = "outbound_authorization"
    sip_dial_started_at: float | None = None

    def observe_call_id(call_id: str) -> None:
        nonlocal resolved_call_id
        resolved_call_id = str(call_id or "") or resolved_call_id

    async def terminate_failure(exc: BaseException | None = None) -> None:
        async def remove_participant() -> None:
            if client is None or not participant_cleanup_required:
                return
            await client.room.remove_participant(
                api_module.RoomParticipantIdentity(
                    room=room_name,
                    identity=participant_identity,
                )
            )

        async def delete_room() -> None:
            if client is None or not room_cleanup_required:
                return
            await client.room.delete_room(api_module.DeleteRoomRequest(room=room_name))

        # LiveKit answers a create_sip_participant failure with a
        # SipCallError carrying the real SIP response (486 Busy Here, 480
        # Temporarily Unavailable, 603 Decline, ...) when the FAR END actively
        # rejected the call. When our OWN ``ringing_timeout`` (set on the
        # request in ``_duration_limits``) elapses first with nobody having
        # answered, LiveKit cancels the INVITE itself and the standard SIP
        # response to a canceled INVITE transaction is 487 Request Terminated
        # (RFC 3261) -- proven against the 2026-08-28 incident, where every
        # failure logged "SIP invite failed" at exactly the configured ring
        # window. That is a no-answer outcome, not a carrier rejection, so it
        # must not be reported (and therefore billed/labelled) the same way.
        # The status code is read from the same bounded evidence the report
        # below carries, and is absent (unclassified) for any exception that
        # never reached a SIP response at all -- a refusal the client reports
        # as a transport error, or this worker's own asyncio.CancelledError.
        evidence = _sip_refusal_evidence(exc)
        sip_status_code = evidence.get("sip_status_code")
        # A 487 only means the INVITE transaction was canceled -- not why
        # (Codex review). Our own ringing_timeout elapsing is one cause, but
        # the room/participant can also be torn down mid-dial by an
        # unrelated abort (e.g. a customer-initiated cancel of a still-
        # ringing call, which reaches LiveKit as a SIP CANCEL and gets the
        # exact same 487 response). Only the former ran for (approximately)
        # the full configured window, so require the elapsed wall-clock time
        # since the INVITE was sent to be within
        # ``_RINGING_TIMEOUT_ELAPSED_SLACK_SECONDS`` of it before trusting a
        # 487 as a ring timeout; an abort a few seconds into dialing keeps
        # this stage's ordinary rejected classification instead.
        elapsed_since_sip_dial = (
            _monotonic() - sip_dial_started_at if sip_dial_started_at is not None else None
        )
        # The far end does not always answer the cancelled INVITE.
        # The 487 above is what arrives when the CARRIER responds to our CANCEL;
        # when the client cancels on its own deadline and no response comes
        # back, it raises with no SIP status at all and only its own error code,
        # "canceled". Measured on the 2026-09-09 06:17Z batch: all 23 calls
        # classified as a SIP-participant rejection carried NO sip_status_code,
        # `livekit_error_code` "canceled", and a failure 47 to 48 seconds after
        # dialing against a 45-second ring window -- the elapsed half of this
        # test already matched, and only the 487 half failed. The SIP container
        # logged those endings as "no-answer". So a recipient who simply did not
        # pick up was recorded as a technical rejection, and on the previous
        # batch that put 37 of 50 attempts in the same bucket as real failures.
        #
        # Requiring the ABSENCE of a status code keeps the two shapes apart: a
        # carrier that answered with a real code is classified by that code and
        # never borrows this branch.
        cancelled_without_sip_response = (
            sip_status_code is None
            and str(evidence.get("livekit_error_code") or "") == "canceled"
        )
        is_ringing_timeout = (
            stage == "outbound_sip_participant"
            and (sip_status_code == 487 or cancelled_without_sip_response)
            and elapsed_since_sip_dial is not None
            and elapsed_since_sip_dial
            >= settings.ringing_timeout_seconds - _RINGING_TIMEOUT_ELAPSED_SLACK_SECONDS
        )
        reason = (
            "outbound_ringing_timeout"
            if is_ringing_timeout
            else {
                "outbound_authorization": "outbound_authorization_rejected",
                "outbound_media_admission": "outbound_media_admission_rejected",
                "outbound_trunk": "outbound_trunk_rejected",
                "outbound_room": "outbound_room_rejected",
                "outbound_dispatch": "outbound_dispatch_rejected",
                # The dispatch produced no agent: the same customer-facing
                # reason, reported before any leg was placed.
                "outbound_agent_join": "outbound_dispatch_rejected",
                "outbound_sip_participant": "outbound_sip_participant_rejected",
            }.get(stage, "outbound_pre_media_rejected")
        )
        await terminate_pre_media_call(
            call=call,
            adapter=adapter,
            call_id=resolved_call_id,
            reason=reason,
            # Every stage's refusal carries whatever the SIP client said about
            # it, not just the SIP-participant stage's: a room or dispatch call
            # refused by the same client leaves the same kind of evidence, and
            # withholding it there would only recreate the blind spot one stage
            # earlier. `None` rather than `{}` when there is nothing to say, so
            # a report about a failure the client never classified stays
            # visibly empty. This event is readable by the caller, so it
            # carries the tenant-readable subset; the operator's full copy goes
            # onto the worker job. The version/stage marker remains even when
            # the provider supplied no classified detail, so future readers
            # can distinguish an explicit unknown from historical absence.
            details=reported_failure_details(evidence, stage),
            remove_participant=remove_participant,
            delete_room=delete_room,
        )

    try:
        fields, trunk_name, call = await asyncio.to_thread(admit, observe_call_id)
        resolved_call_id = fields["call_id"]
        room_name = f"opx-out-{_safe_id(resolved_call_id)}"
        participant_identity = f"pstn-{_safe_id(resolved_call_id)}"
        dispatch = {
            "schema": DISPATCH_SCHEMA,
            **fields,
            "participant_identity": participant_identity,
        }
        # Once this starts, a failure may race real SIP origination. The
        # worker must settle the one claim terminally rather than redial.
        stage = "outbound_media_admission"
        may_have_originated = True
        await asyncio.to_thread(call.admit_media)
        if api_module is None:
            from livekit import api as api_module
        client = api_module.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        stage = "outbound_trunk"
        # Per-number trunk dispatch, read fresh from the job payload on every
        # attempt, so a retry dials the same BYO trunk the control plane
        # resolved for this number - never the env default by accident.
        trunk = await _select_outbound_trunk(
            client,
            api_module,
            settings,
            fields["from_e164"],
            trunk_name=trunk_name,
        )
        # Set before the RPC: a timeout can occur after LiveKit created it.
        room_cleanup_required = True
        stage = "outbound_room"
        await _livekit_rpc(
            client.room.create_room(
                api_module.CreateRoomRequest(
                    name=room_name,
                    empty_timeout=60,
                    departure_timeout=15,
                    max_participants=3,
                    metadata=json.dumps(dispatch, sort_keys=True) if fields.get("worker_job_id") else "",
                )
            )
        )
        stage = "outbound_dispatch"
        await _livekit_rpc(
            client.agent_dispatch.create_dispatch(
                api_module.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME,
                    room=room_name,
                    metadata=json.dumps(dispatch, sort_keys=True),
                )
            )
        )
        # Dial only once this call's own agent is in the room. If it never
        # comes, the failure below deletes the room and reports the call
        # before any leg exists.
        stage = "outbound_agent_join"
        await _wait_for_agent_join(client, api_module, room_name)
        if agent_joined is not None:
            # A concurrent originator claims its next job only now: two agent
            # jobs never launch at once (livekit-agents counts a launching job
            # twice, which could refuse the next dispatch), and no claimed job
            # ever queues behind another call's join.
            agent_joined.set()
        if before_dial is not None:
            # Recorded server-side before the SIP participant exists: from
            # here the job can never be deferred. If it cannot be
            # recorded, nothing is dialled.
            stage = "outbound_dial_marker"
            await asyncio.to_thread(before_dial)
        # A timeout can occur after the SIP participant exists; remove this
        # deterministic identity before deleting the call-scoped room.
        participant_cleanup_required = True
        stage = "outbound_sip_participant"
        sip_dial_started_at = _monotonic()
        participant = await client.sip.create_sip_participant(
            api_module.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=trunk.sip_trunk_id,
                sip_call_to=fields["to_e164"],
                sip_number=fields["from_e164"].removeprefix("+"),
                participant_identity=participant_identity,
                participant_name="Skysay outbound call",
                participant_attributes={
                    "skysay.call_id": fields["call_id"],
                    "skysay.direction": "outbound",
                    # The trunk this call was dialled on, so the agent can tell whether the
                    # line is proven to carry keys (navigation_dtmf). A name, never a secret.
                    "skysay.outbound_trunk": str(getattr(trunk, "name", "") or ""),
                },
                wait_until_answered=True,
                media=api_module.SIPMediaConfig(
                    only_listed_codecs=True,
                    codecs=[api_module.SIPCodec(name="PCMU", rate=8_000)],
                ),
                **_duration_limits(settings),
            ),
            timeout=settings.ringing_timeout_seconds + 10,
        )
        return {
            "executed": True,
            "provider": "livekit_sip",
            "call_id": fields["call_id"],
            "room_name": room_name,
            "participant_identity": participant_identity,
            "provider_call_id": str(getattr(participant, "sip_call_id", "") or ""),
            "call_status": "answered",
            "from": _redact_phone(fields["from_e164"]),
            "to": _redact_phone(fields["to_e164"]),
            "secrets_redacted": True,
            **({key: fields[key] for key in ("organization_id", "project_id", "worker_job_id")} if fields.get("worker_job_id") else {}),
        }
    except asyncio.CancelledError as exc:
        # Shutdown or the hosted lease deadline can cancel an in-flight RPC.
        # Cleanup is conservative: cancellation is not proof SIP never started.
        await terminate_failure(exc)
        raise
    except PhoneCapacityRefusedError as exc:
        if not defer_capacity_refusal:
            # Only a hosted queue job can be given back (``defer``). A
            # deployment-local or organization-mode call fails as a dispatch
            # that produced no agent did before (Codex, !220): its
            # reservation and slot are released by the terminal report.
            await terminate_failure(exc)
            raise OutboundPostAdmissionError("outbound execution ended after media authorization") from exc
        # Not a failure of the call, which nobody dialled (the SIP participant
        # is created only after the join wait): the room goes, nothing is
        # reported against the call, and the job is given back to the queue --
        # but only once the room's delete is CONFIRMED. The retry reuses the
        # room name, so a delete that failed or timed out and lands later would
        # hang the retried call up mid-conversation (Codex, adversarial review
        # of the defer, reproduced). Unconfirmed, the call fails as a dispatch
        # that produced no agent always did.
        deleted = False
        if room_cleanup_required and client is not None:
            try:
                await _livekit_rpc(client.room.delete_room(api_module.DeleteRoomRequest(room=room_name)))
                deleted = True
            except Exception:
                deleted = False
        if not deleted:
            await terminate_failure(exc)
            raise OutboundPostAdmissionError("outbound execution ended after media authorization") from exc
        raise
    except Exception as exc:
        await terminate_failure(exc)
        if may_have_originated:
            raise OutboundPostAdmissionError(
                "outbound execution ended after media authorization"
            ) from exc
        raise
    finally:
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                # A close failure must not hide the canonical SIP failure or
                # change the worker's retry decision.
                pass


async def originate_job(
    settings: OutboundWorkerSettings,
    job: dict[str, Any],
    *,
    api_module: Any | None = None,
    agent_joined: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Dial a call the control plane admitted for somebody else to place."""

    fields = _validate_job(settings, job)
    gateway_transport, worker_transport = _transports(settings)
    payload = job["payload"]
    auth_metadata = payload.get("authorization_metadata")
    if not isinstance(auth_metadata, dict):
        auth_metadata = {"channel": "voice"}
    adapter = Gate2ControlPlaneAdapter(
        gateway_id=settings.gateway_id,
        transport=gateway_transport,
        call_transport=gateway_transport if settings.worker_mode == "platform" else worker_transport,
        tool_transport=gateway_transport if settings.worker_mode == "platform" else worker_transport,
    )

    def admit(_observe_call_id: Any) -> tuple[dict[str, str], str, Any]:
        session = adapter.prepare_outbound(
            call_id=fields["call_id"],
            agent_id=fields["agent_id"],
            from_number_id=fields["from_number_id"],
            to_e164=fields["to_e164"],
            metadata=auth_metadata,
            **({"execution_lease": {"worker_job_id": job["id"], "worker_id": settings.worker_id, "lease_token": job["lease_token"]}}
               if settings.worker_mode == "platform" else {}),
        )
        if settings.worker_mode == "platform":
            plan = adapter.outbound_dial_plan
            if not plan or plan.get("organization_id") != job["organization_id"]:
                raise OutboundWorkerError("hosted dial plan differs from the claimed tenant")
            fields.update({"organization_id": plan["organization_id"], "project_id": plan["project_id"],
                           "worker_job_id": job["id"], "from_e164": plan["from_phone_number"]})
            return fields, str(plan.get("outbound_trunk_name") or ""), session
        # Per-number trunk dispatch, read fresh from the job payload on every
        # attempt, so a retry dials the same BYO trunk the control plane
        # resolved for this number - never the env default by accident.
        return fields, str(payload.get("outbound_trunk_name") or ""), session

    return await _originate_admitted_call(
        settings,
        adapter,
        admit,
        known_call_id="" if settings.worker_mode == "platform" else fields["call_id"],
        api_module=api_module,
        agent_joined=agent_joined,
        before_dial=(
            (lambda: worker_transport.request(
                "POST",
                f"/v1/worker-jobs/{job['id']}/dialing",
                {"worker_id": settings.worker_id, "lease_token": job.get("lease_token", "")},
            ))
            if settings.worker_mode == "platform"
            else None
        ),
        defer_capacity_refusal=settings.worker_mode == "platform",
    )


async def originate_local_call(
    settings: OutboundWorkerSettings,
    *,
    to_e164: str,
    from_number_id: str = "",
    from_phone_number: str = "",
    request: dict[str, Any] | None = None,
    api_module: Any | None = None,
) -> dict[str, Any]:
    """Place a call this deployment asked for, with only its own credential.

    The queue is not involved and neither is an organization-scoped key: the
    control plane admits the call against this deployment's token, applies the
    same policy decision and wallet reserve a remote request gets, and returns
    the single-use token that admits its media.
    """

    if not settings.execute:
        raise OutboundWorkerError(
            "outbound execution is disabled; set SKYSAY_EXECUTE_LIVEKIT_OUTBOUND=1 to place calls"
        )
    transport = _gateway_transport(settings)
    # The dial reads its agent's config with this deployment's mode, as the
    # agent process does: once, from the deployment, before any request.
    await asyncio.to_thread(adopt_deployment_content_storage, os.environ, transport=transport)
    adapter = Gate2ControlPlaneAdapter(
        gateway_id=settings.gateway_id,
        transport=transport,
    )

    def admit(observe_call_id: Any) -> tuple[dict[str, str], str, Any]:
        dial_plan, token = adapter.admit_local_outbound(
            to_e164=to_e164,
            from_number_id=from_number_id,
            from_phone_number=from_phone_number,
            request=request,
        )
        # Reported before the config fetch below: from here on the call exists
        # and holds a reservation, so a later failure must be attributable.
        observe_call_id(dial_plan["call_id"])
        session = adapter.session_for_local_outbound(dial_plan, token)
        return (
            {
                "call_id": dial_plan["call_id"],
                "agent_id": dial_plan["agent_id"],
                "from_number_id": dial_plan["number_id"],
                "from_e164": dial_plan["from_phone_number"],
                "to_e164": dial_plan["to_phone_number"],
            },
            dial_plan.get("outbound_trunk_name", ""),
            session,
        )

    return await _originate_admitted_call(settings, adapter, admit, api_module=api_module)


def _job_error_code(exc: BaseException) -> str:
    """Exception CLASS NAMES, plus the SIP client's own bounded evidence.

    Never the exception's message: that is unbounded and can carry provider
    detail. The structured fields appended here are the opposite -- declared
    by the SIP client, length-bounded, and stripped of URI credentials and
    identifiers before they are serialised.

    A post-admission wrapper is terminal for the job, so on its own it tells an
    operator nothing about WHY. Appending the wrapped class keeps a failure
    caused by a per-number trunk that does not exist or does not carry the
    caller ID (OutboundWorkerError) distinguishable from a transient SDK or
    transport error, without widening what the job record reveals.

    The class names alone stopped being enough once a whole class of outbound
    refusals turned out to reach the job record as one undifferentiated server
    error, collapsing an unallocated destination, a carrier declining, and a
    trunk with no free channel into a single word. The same evidence the
    call's own failure event now carries is appended here too, so an operator
    holding the job -- who may never look at the call -- can tell those apart.
    """

    name = type(exc).__name__
    cause = exc.__cause__
    code = name if cause is None or type(cause) is type(exc) else f"{name}:{type(cause).__name__}"
    refusal = contract_refusal(exc)
    if refusal:
        # A bounded, validated token from the control plane's own body, never
        # its message: the operator holding the job sees WHICH contract check
        # refused this worker.
        code = f"{code}:{refusal}"
    # The evidence rides on the WRAPPED exception when a post-admission
    # wrapper was applied, and on the exception itself when the failure was
    # raised before one could be.
    evidence = _sip_refusal_evidence(cause) or _sip_refusal_evidence(exc)
    if not evidence:
        return code
    return f"{code} {json.dumps(evidence, sort_keys=True)}"


class AgentsUnavailableError(OutboundWorkerError):
    """No phone agent on this host has been registered and admitting for too long.

    Raised only to be reported (error tracking), never out of the loop: the
    originator keeps waiting, claims nothing, and the queued calls stay
    queued, so an operator has to be told.
    """


class OutboundWorker:
    def __init__(self, settings: OutboundWorkerSettings) -> None:
        self.settings = settings
        self.gateway_transport, self.worker_transport = _transports(settings)
        # When this host's agents were first seen not ready, on the monotonic
        # clock; None while they are ready. And whether this not-ready spell
        # has already raised its alert.
        self._agents_unready_since: float | None = None
        self._agents_alerted = False
        # The barrier's two clocks: beacons carry wall time (written by
        # another process on this host), the wait is measured monotonically.
        self._wall_clock = time.time
        self._monotonic = time.monotonic

    def agents_barrier(self) -> dict[str, Any] | None:
        """None when this host's agents can take a dispatch; otherwise why not.

        The claim barrier. After a boot or restart the originator
        used to claim a job before any agents worker had registered with
        LiveKit; the dispatch found nobody, the join wait ran out, and the call
        failed although the host could serve it seconds later. So nothing is
        claimed until a fresh beacon says an agents worker here is registered
        and not draining. Nothing claimed is nothing attempted: the job stays
        queued with its attempts untouched.

        The heartbeat says so meanwhile: ``waiting_for_agents``, and after
        :data:`UNAVAILABLE_AFTER_SECONDS` of it ``agents_unavailable``, with one
        error-tracking alert per spell, so queued calls never wait silently.
        """

        directory = self.settings.agents_readiness_dir
        if not directory:
            return None
        readiness = read_readiness(Path(directory), now=self._wall_clock())
        now = self._monotonic()
        if readiness.ready:
            if self._agents_unready_since is not None:
                print(
                    "SKYSAY_OUTBOUND "
                    + json.dumps(
                        {"event": "agents_ready", "waited_seconds": int(now - self._agents_unready_since)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            self._agents_unready_since = None
            self._agents_alerted = False
            return None
        if self._agents_unready_since is None:
            self._agents_unready_since = now
        waited = now - self._agents_unready_since
        unavailable = waited >= UNAVAILABLE_AFTER_SECONDS
        if unavailable and not self._agents_alerted:
            self._agents_alerted = True
            capture_exception(
                AgentsUnavailableError(
                    f"no phone agent on this host has been registered and admitting for {int(waited)} s; "
                    "outbound calls are queued and not being dialled"
                ),
                agents_reason=readiness.reason,
            )
        status = "agents_unavailable" if unavailable else "waiting_for_agents"
        heartbeat_ok = self.heartbeat(status, {"agents_reason": readiness.reason, "agents_waited_seconds": int(waited)})
        return {
            "claimed": False,
            "dry_run": False,
            "agents_ready": False,
            "agents_reason": readiness.reason,
            "agents_waited_seconds": int(waited),
            "heartbeat_status": status,
            "heartbeat": heartbeat_ok,
        }

    def heartbeat(self, status: str, extra: Mapping[str, Any] | None = None) -> bool:
        """Report that this originator exists and in what mode.

        Returns False instead of raising: a control plane that is briefly
        unreachable, or a key missing the heartbeat scope, must not take down
        a worker that is otherwise able to dial. The caller records the result
        so a persistently failing heartbeat is visible in the worker's own
        output rather than silent.
        """

        try:
            response = self.worker_transport.request(
                "POST",
                "/v1/worker-heartbeats",
                {
                    "worker_id": self.settings.worker_id,
                    "worker_type": "livekit_outbound",
                    "status": status,
                    "metadata": {
                        "execute_gate_enabled": self.settings.execute,
                        "trunk_name": self.settings.trunk_name,
                        **dict(extra or {}),
                    },
                },
            )
        except Exception:
            return False
        return bool(response.get("id"))

    def preview(self) -> dict[str, Any]:
        authenticated = self.heartbeat("staging" if not self.settings.execute else "ready")
        return {
            "execute": self.settings.execute,
            "worker_id": self.settings.worker_id,
            "organization_id": self.settings.organization_id,
            "worker_mode": self.settings.worker_mode,
            "trunk_name": self.settings.trunk_name,
            "worker_token_authenticated": authenticated,
            "secrets_redacted": True,
        }

    def claim(self) -> dict[str, Any] | None:
        response = self.worker_transport.request(
            "POST",
            "/v1/worker-jobs/claim",
            {
                "worker_id": self.settings.worker_id,
                "kind": "call.outbound_originate",
                "worker_contract": WORKER_CONTRACT,
                **({"organization_id": self.settings.organization_id} if self.settings.worker_mode != "platform" else {}),
            },
        )
        job = response.get("job")
        return job if isinstance(job, dict) else None

    def settle(self, job: dict[str, Any], action: str, body: dict[str, Any]) -> dict[str, Any]:
        proof = {"worker_id": self.settings.worker_id, "lease_token": job.get("lease_token", "")} if self.settings.worker_mode == "platform" else {}
        return self.worker_transport.request("POST", f"/v1/worker-jobs/{job['id']}/{action}", {**body, **proof})

    async def run_once(self) -> dict[str, Any]:
        if not self.settings.execute:
            return {"claimed": False, "dry_run": True, **self.preview()}
        # The heartbeat has to live HERE, in the loop that can actually dial.
        # It used to be emitted only by `preview()`, which this branch never
        # reaches -- so the worker announced itself while the execute gate was
        # shut and went silent the moment it was opened. Anything watching
        # `livekit_outbound` freshness to decide whether an originator exists
        # was reading a signal that meant the opposite of what it looked like.
        #
        # And it is emitted AFTER a successful claim attempt, never before.
        # Customer queue claiming is organization-bound. A legacy worker
        # holding a token for one organization while configured for another
        # heartbeats happily and then fails every claim, so reporting readiness
        # first would refresh a green signal forever while placing no calls. An
        # idle claim still counts: it proves the worker may claim, which is
        # exactly what "ready" is supposed to mean.
        #
        # Unless this host's agents cannot take a dispatch: then nothing is
        # claimed at all, and the heartbeat says why.
        try:
            blocked = await asyncio.to_thread(self.agents_barrier)
        except Exception as exc:
            # The barrier reads a local directory; a failure to read it is
            # "not ready", never a reason to claim blind.
            blocked = {"claimed": False, "dry_run": False, "agents_ready": False, "event": "agents_barrier_failed", "error_type": type(exc).__name__}
        if blocked is not None:
            return blocked
        job = await asyncio.to_thread(self.claim)
        heartbeat_ok = await asyncio.to_thread(self.heartbeat, "ready")
        if job is None:
            return {"claimed": False, "dry_run": False, "idle": True, "heartbeat": heartbeat_ok}
        return await self.process_job(job, heartbeat_ok)

    async def process_job(
        self, job: dict[str, Any], heartbeat_ok: bool, *, agent_joined: asyncio.Event | None = None
    ) -> dict[str, Any]:
        """Dial one claimed job and settle it. The unit a supervisor runs in parallel.

        ``agent_joined`` is set once the call's agent is in its room: the
        supervisor waits for it before claiming the next job.
        """

        renewer: asyncio.Task[None] | None = None
        try:
            try:
                if self.settings.worker_mode == "platform":
                    _validate_job(self.settings, job)
                    renewed = await asyncio.to_thread(self.settle, job, "renew", {})
                    seconds = min(_originate_deadline_seconds(self.settings), _lease_seconds(self.settings, renewed) - 10)
                    if renewed.get("id") != job["id"] or renewed.get("lease_token") != job["lease_token"]:
                        raise OutboundWorkerError("renewal returned a different worker lease")
                    renewer = asyncio.create_task(self._renew_until_settled(job))
                    result = await asyncio.wait_for(
                        originate_job(self.settings, job, agent_joined=agent_joined), timeout=seconds
                    )
                else:
                    result = await originate_job(self.settings, job, agent_joined=agent_joined)
            except asyncio.CancelledError:
                if self.settings.worker_mode == "platform":
                    try:
                        await asyncio.wait_for(asyncio.to_thread(self.settle, job, "fail", {"error": "dispatch_cancelled", "retry": False, "max_attempts": 1}), timeout=5)
                    except Exception:
                        pass
                raise
            except PhoneCapacityRefusedError as exc:
                if self.settings.worker_mode != "platform":
                    return await self._settle_failure(job, exc)
                # Back to the queue: not an attempt, the call not failed, its
                # reservation kept. The control plane bounds how
                # often, and the next attempt is authorized afresh.
                try:
                    deferred = await asyncio.to_thread(self.settle, job, "defer", {"reason": "phone_capacity_refused"})
                except Exception:
                    deferred = {"status": "unconfirmed"}
                return {
                    "claimed": True,
                    "job_id": job.get("id", ""),
                    "deferred": True,
                    "error_type": type(exc).__name__,
                    "job_status": deferred.get("status", "queued"),
                    "secrets_redacted": True,
                }
            except Exception as exc:
                return await self._settle_failure(job, exc)
            # The call was dialled: from here the job is reported complete or
            # left for the control plane, NEVER failed. A generic ``fail`` here
            # used to fail a live call and release its reservation because its
            # completion report did not land.
            completed = await self._complete_dialled(job, result)
            if completed is None:
                return {
                    "claimed": True,
                    "job_id": job.get("id", ""),
                    "result": result,
                    "job_status": "unconfirmed",
                    "settlement": "complete_unconfirmed",
                    # Not a failure of the CALL, which is never failed here, but
                    # an unfinished settlement: a one-shot run exits non-zero.
                    "error_type": "CompletionUnconfirmed",
                    "heartbeat": heartbeat_ok,
                }
            return {
                "claimed": True,
                "job_id": job.get("id", ""),
                "result": result,
                "job_status": completed.get("status", "succeeded"),
                "heartbeat": heartbeat_ok,
            }
        finally:
            if renewer is not None:
                renewer.cancel()
                try:
                    await renewer
                except BaseException:
                    pass

    async def _settle_failure(self, job: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        retry = self.settings.worker_mode != "platform" and bool(getattr(exc, "outbound_retry_safe", True))
        if contract_refusal(exc):
            # The control plane refused this worker for this call on its
            # contract. Nothing about that changes on a redial: the job fails
            # once, for an operator to read, and stays out of the retry loop.
            retry = False
        try:
            failed = await asyncio.to_thread(
                self.settle, job, "fail",
                {
                    "error": _job_error_code(exc),
                    "retry": retry,
                    "delay_seconds": 5 if retry else 0,
                    "max_attempts": 3 if retry else 1,
                },
            )
        except Exception:
            if self.settings.worker_mode != "platform":
                raise
            failed = {"status": "unconfirmed"}
        return {
            "claimed": True,
            "job_id": job.get("id", ""),
            "error_type": type(exc).__name__,
            "job_status": failed.get("status", "failed"),
            "secrets_redacted": True,
        }

    async def _complete_dialled(self, job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        """Report a dialled call complete; retry, and never fall back to ``fail``.

        Returns the confirmed job, or None when no attempt was confirmed. An
        unconfirmed job stays ``running`` under this worker, and the control
        plane's terminal originate reaper settles it: the documented path a
        crash after the dial already takes.
        """

        error_type = ""
        for attempt, delay in enumerate((0.0, *COMPLETE_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            try:
                completed = await asyncio.to_thread(self.settle, job, "complete", {"result": result})
            except Exception as exc:
                error_type = type(exc).__name__
                continue
            if self.settings.worker_mode != "platform":
                return completed
            if completed.get("id") == job["id"] and completed.get("status") == "succeeded":
                return completed
            error_type = "completion_not_confirmed"
        print(
            "SKYSAY_OUTBOUND "
            + json.dumps(
                {"event": "complete_unconfirmed", "job_id": job.get("id", ""), "attempts": attempt + 1, "error_type": error_type},
                sort_keys=True,
            ),
            flush=True,
        )
        return None

    async def _renew_until_settled(self, job: dict[str, Any]) -> None:
        """Keep a hosted claim's lease alive until its job is settled. Cancelled by the job."""

        for _ in itertools.count():
            await asyncio.sleep(LEASE_RENEW_SECONDS)
            try:
                await asyncio.to_thread(self.settle, job, "renew", {})
            except Exception as exc:
                # A renewal that fails never aborts a live dial: settlement
                # reports the stale lease itself, and the reaper covers it.
                print(
                    "SKYSAY_OUTBOUND "
                    + json.dumps({"event": "lease_renew_failed", "job_id": job.get("id", ""), "error_type": type(exc).__name__}, sort_keys=True),
                    flush=True,
                )

    async def run(self, *, watch: bool) -> int:
        # Stop CLAIMING on SIGTERM, but finish whatever is already claimed.
        #
        # Without this the default disposition applies and the process dies
        # the instant a release recreates the container -- in the middle of
        # `originate_job`, after the claim and before the complete/fail
        # report. The row stays `status=running, locked_by=<this worker>`
        # forever, because nothing reclaims a claimed job whose owner is
        # gone, and it keeps holding one of the organization's bounded
        # outbound execution slots while the paired call row reads terminal.
        # One release left exactly one such row and one permanently burnt
        # slot; the leak was invisible from the calls view.
        #
        # This closes the release case, which is the common one. An
        # ungraceful death -- OOM, SIGKILL after the stop grace, host loss --
        # still orphans a claim, and only a control-plane lease expiry can
        # cover that; it is a separate change against this one.
        stop = asyncio.Event()
        release_handlers = install_stop_handlers(stop) if watch else (lambda: None)
        try:
            # Only a worker allowed to dial runs the supervisor: the execute
            # gate lives in run_once, and a shut gate keeps its staging loop.
            if (
                watch
                and getattr(self.settings, "execute", False) is True
                and int(getattr(self.settings, "max_in_flight", 1) or 1) > 1
            ):
                return await self._supervise(stop)
            while True:
                # Before the claim, not only after the poll: the signal can
                # arrive between installing the handler and the first claim,
                # and a claim made after the stop is a dial this worker may
                # not live to report -- the orphaned job the drain exists to
                # prevent.
                #
                # It stops the NEXT claim, and deliberately not one already in
                # flight. A claim that returns after the stop has already been
                # taken server-side, so the honest options are to finish it or
                # to requeue it -- and requeueing a job whose dial may already
                # have reached the carrier duplicates a call rather than
                # saving one. So this worker finishes it, and
                # ORIGINATOR_STOP_GRACE_SECONDS (300s) is sized for exactly
                # that path: the whole claim-to-settlement chain measures 272s
                # worst case, counted term by term in
                # test_the_originators_grace_outlasts_a_whole_claim_not_just_the_dial.
                # An earlier 130s grace covering only the dial is the bug that
                # test was written for.
                if watch and stop.is_set():
                    print(
                        "SKYSAY_OUTBOUND " + json.dumps({"drained": True, "worker_id": self.settings.worker_id}, sort_keys=True),
                        flush=True,
                    )
                    return 0
                result = await self.run_once()
                print("SKYSAY_OUTBOUND " + json.dumps(result, sort_keys=True), flush=True)
                if not watch:
                    return 0 if not result.get("error_type") else 1
                if not await sleep_unless_stopped(stop, self.settings.poll_seconds):
                    print(
                        "SKYSAY_OUTBOUND " + json.dumps({"drained": True, "worker_id": self.settings.worker_id}, sort_keys=True),
                        flush=True,
                    )
                    return 0
        finally:
            release_handlers()

    async def _supervise(self, stop: asyncio.Event) -> int:
        """Dial up to ``max_in_flight`` claimed jobs at once, and never cancel one.

        A stop ends CLAIMING; every job already claimed runs to its settlement,
        in parallel, inside the same ORIGINATOR_STOP_GRACE_SECONDS one job
        needs. A claim or heartbeat that raises is logged and retried after the
        poll interval: it must never unwind this method, because unwinding it
        is what would cancel the calls in flight.
        """

        cap = self.settings.max_in_flight
        in_flight: set[asyncio.Task[None]] = set()
        # Every control-plane report and lease renewal is a blocking HTTP call
        # run in a thread. The default pool is sized from the CPU count (six
        # threads on a two-core host), so N jobs each settling while their
        # renewals run could queue behind each other and stretch a report past
        # the stop grace. Room for every job's report and renewal, plus the
        # claim and heartbeat.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2 * cap + 2))
        try:
            while not stop.is_set():
                if len(in_flight) >= cap:
                    await self._wait_for(stop, in_flight, timeout=None)
                    continue
                try:
                    blocked = await asyncio.to_thread(self.agents_barrier)
                except Exception as exc:
                    # The barrier reads a local directory; a failure to read it
                    # is "not ready", never a reason to claim blind.
                    blocked = {"event": "agents_barrier_failed", "error_type": type(exc).__name__}
                if blocked is not None:
                    print("SKYSAY_OUTBOUND " + json.dumps(blocked, sort_keys=True), flush=True)
                    await self._wait_for(stop, in_flight, timeout=self.settings.poll_seconds)
                    continue
                try:
                    job = await asyncio.to_thread(self.claim)
                    heartbeat_ok = await asyncio.to_thread(self.heartbeat, "ready")
                except Exception as exc:
                    print(
                        "SKYSAY_OUTBOUND " + json.dumps({"event": "claim_failed", "error_type": type(exc).__name__}, sort_keys=True),
                        flush=True,
                    )
                    job, heartbeat_ok = None, False
                if job is not None:
                    agent_joined = asyncio.Event()
                    task = asyncio.create_task(self._run_claimed(job, heartbeat_ok, agent_joined))
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)
                    # Claim the next job once this one's agent has joined, or
                    # its originate has ended without one: the next dispatch
                    # never overlaps this agent's launch, and a claimed job
                    # never waits on another call. The legs still ring
                    # together, and the next claim does not wait out a poll
                    # interval behind one that is ringing.
                    await self._wait_for_join(stop, task, agent_joined)
                    continue
                await self._wait_for(stop, in_flight, timeout=self.settings.poll_seconds)
        finally:
            if in_flight:
                # Never cancelled: a claimed job may already have reached the
                # carrier, and only its own settlement can say what happened.
                await asyncio.wait(set(in_flight))
            print(
                "SKYSAY_OUTBOUND " + json.dumps({"drained": True, "worker_id": self.settings.worker_id}, sort_keys=True),
                flush=True,
            )
        return 0

    async def _run_claimed(self, job: dict[str, Any], heartbeat_ok: bool, agent_joined: asyncio.Event) -> None:
        try:
            result = await self.process_job(job, heartbeat_ok, agent_joined=agent_joined)
        except Exception as exc:
            result = {"claimed": True, "job_id": job.get("id", ""), "error_type": type(exc).__name__, "job_status": "unconfirmed"}
        print("SKYSAY_OUTBOUND " + json.dumps(result, sort_keys=True), flush=True)

    @staticmethod
    async def _wait_for_join(stop: asyncio.Event, task: asyncio.Task[None], agent_joined: asyncio.Event) -> None:
        """Until the job's agent joined, its task ended, a stop, or the pacing cap.

        Cancels only its own waiters, never the job.
        """

        waiters = [
            asyncio.ensure_future(stop.wait()),
            asyncio.ensure_future(agent_joined.wait()),
            asyncio.ensure_future(asyncio.sleep(CLAIM_PACING_CAP_SECONDS)),
        ]
        try:
            await asyncio.wait({*waiters, task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()

    @staticmethod
    async def _wait_for(stop: asyncio.Event, in_flight: set[asyncio.Task[None]], *, timeout: float | None) -> None:
        """Until a stop, a finished job, or ``timeout``. Cancels only its own waiters."""

        waiters: list[asyncio.Future[Any]] = [asyncio.ensure_future(stop.wait())]
        if timeout is not None:
            waiters.append(asyncio.ensure_future(asyncio.sleep(timeout)))
        try:
            await asyncio.wait({*waiters, *in_flight}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()


def _default_caller_number(environ: Mapping[str, str]) -> str:
    """The first DID this deployment serves, used when none is named.

    A single-DID box answers and dials on the same number, and that number is
    already configured once. Taking the first entry keeps `originate --to ...`
    a complete command there, while a box with several numbers names the one
    it wants.
    """
    configured = str(environ.get("SCH236_TEST_DID_E164") or "").strip()
    return configured.split(",")[0].strip() if configured else ""


def _originate_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="outbound_worker.py originate",
        description="Place one outbound call from this deployment, without the job queue",
    )
    parser.add_argument("--to", required=True, help="destination in E.164, e.g. +37255501234")
    parser.add_argument(
        "--from",
        dest="from_phone_number",
        default="",
        help="caller id in E.164; defaults to the first number in SCH236_TEST_DID_E164",
    )
    parser.add_argument(
        "--from-number-id", default="", help="caller id by control-plane number id, instead of --from"
    )
    parser.add_argument("--agent-id", default="", help="agent to speak; defaults to the number's own agent")
    # Required, not defaulted: outbound authorization is a declaration about
    # WHY this call is being placed, and inventing one here would put a
    # compliance claim in the record that nobody actually made. A call without
    # them is refused by policy anyway -- better to say so before dialling.
    parser.add_argument("--purpose", required=True, help="declared call purpose, evaluated by policy")
    parser.add_argument("--reason", required=True, help="why this call was asked for")
    parser.add_argument("--recipient-type", default="", help="declared recipient type, evaluated by policy")
    parser.add_argument("--relationship", default="", help="declared relationship, evaluated by policy")
    parser.add_argument("--authorized-by", default="", help="who asked for this call")
    parser.add_argument("--idempotency-key", default="", help="retry this exact request safely")
    args = parser.parse_args(argv)

    if not args.to.startswith("+"):
        raise OutboundWorkerError("--to must be an E.164 number starting with +")
    from_phone_number = args.from_phone_number.strip() or (
        "" if args.from_number_id else _default_caller_number(os.environ)
    )
    if from_phone_number and not from_phone_number.startswith("+"):
        raise OutboundWorkerError("--from must be an E.164 number starting with +")
    if not from_phone_number and not args.from_number_id:
        raise OutboundWorkerError(
            "no caller id: pass --from in E.164, or set SCH236_TEST_DID_E164"
        )
    request: dict[str, Any] = {}
    for key, value in (
        ("agent_id", args.agent_id),
        ("purpose", args.purpose),
        ("recipient_type", args.recipient_type),
        ("relationship", args.relationship),
        ("idempotency_key", args.idempotency_key),
    ):
        if value:
            request[key] = value
    request["user_authorization"] = {
        "authorized_by": args.authorized_by or "self_hosted_gateway",
        "reason": args.reason,
    }
    settings = OutboundWorkerSettings.from_environ(os.environ)
    result = asyncio.run(
        originate_local_call(
            settings,
            to_e164=args.to,
            from_number_id=args.from_number_id,
            from_phone_number=from_phone_number,
            request=request,
        )
    )
    print("SKYSAY_OUTBOUND " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    install_error_tracking(component="voice-outbound-worker")
    try:
        if arguments and arguments[0] == "originate":
            # A one-shot originator, not a second long-running service: it
            # opens no port, so nothing on this host can be talked into
            # dialling. `docker compose run` does NOT apply profiles, so the
            # execution gate this path enforces itself
            # (SKYSAY_EXECUTE_LIVEKIT_OUTBOUND) is the one that gates it;
            # `--profile outbound` still gates the queue watcher.
            return _originate_main(arguments[1:])
        parser = argparse.ArgumentParser(description="Skysay LiveKit outbound originate worker")
        parser.add_argument("--once", action="store_true", help="poll once and exit")
        parser.add_argument("--watch", action="store_true", help="poll continuously")
        parser.add_argument("--status", action="store_true", help="show a redacted queue preview and exit")
        args = parser.parse_args(arguments)
        worker = OutboundWorker(OutboundWorkerSettings.from_environ(os.environ))
        if args.status:
            print(json.dumps(worker.preview(), sort_keys=True))
            return 0
        return asyncio.run(worker.run(watch=args.watch and not args.once))
    except (SystemExit, KeyboardInterrupt):
        # An operator pressing Ctrl-C on a one-shot originator is not a crash,
        # and reporting it would spend an alert on somebody doing their job.
        raise
    except BaseException as exc:
        # This line deliberately carries only the exception CLASS -- an
        # outbound failure message can quote the number it was dialling. The
        # report carries the traceback, scrubbed by the same rule.
        capture_exception(exc, component="voice-outbound-worker")
        print(
            "SKYSAY_OUTBOUND_FAILED "
            + json.dumps({"error_type": type(exc).__name__}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

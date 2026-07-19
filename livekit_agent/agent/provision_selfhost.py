# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
"""One-shot LiveKit SIP bootstrap for the self-hosted OpenPhonex stack.

Generated into the public export as `livekit_agent/agent/provision_selfhost.py`
(template: scripts/oss-release/templates/provision_selfhost.py). This is the
provisioner the self-host compose runs:

    docker compose -f docker-compose.livekit.yml --profile provision run --rm provision

It creates the LiveKit SIP **inbound** trunk + dispatch rule for your DID so an
incoming call is routed to the agent. An **outbound** trunk is created ONLY if
you also supply the outbound carrier credentials (inbound answering does not
need them). It is idempotent: rerun it whenever the DID or the carrier
allowlist changes.

Environment (all documented in .env.livekit.example / INSTALL.md):

  Required
    LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET   (read by LiveKitAPI)
    SCH236_TEST_DID_E164     the DID this agent answers, E.164 with leading +
    DIDWW_INBOUND_ADDRESSES  comma-separated PUBLIC IPv4 CIDRs your carrier
                             signals FROM (inbound allowlist)

  Optional (create an outbound trunk too — needed only to place calls OUT)
    DIDWW_OUTBOUND_HOST / DIDWW_OUTBOUND_FROM_USER
    DIDWW_OUTBOUND_AUTH_USERNAME / DIDWW_OUTBOUND_AUTH_PASSWORD
    (supply all four, or none)

On success it prints `SELFHOST_PROVISIONED {...}`; on failure it prints
`SELFHOST_PROVISION_FAILED {...}` with the real error message and exits 1.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Any

# Must match production_agent.py's AGENT_NAME so the dispatch rule routes the
# inbound call to the worker that registered under this name.
AGENT_NAME = "sch236-livekit-demo"
INBOUND_TRUNK_NAME = "openphonex-selfhost-inbound"
OUTBOUND_TRUNK_NAME = "openphonex-selfhost-outbound"
RULE_NAME = "openphonex-selfhost-inbound-rule"
ROOM_PREFIX = "openphonex-selfhost-"

_E164_DIGITS = re.compile(r"[1-9][0-9]{6,14}")
_HOSTNAME = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)

_OUTBOUND_VARS = (
    "DIDWW_OUTBOUND_HOST",
    "DIDWW_OUTBOUND_FROM_USER",
    "DIDWW_OUTBOUND_AUTH_USERNAME",
    "DIDWW_OUTBOUND_AUTH_PASSWORD",
)

# CIDR ranges a carrier would NEVER signal from. A DIDWW_INBOUND_ADDRESSES entry
# is rejected if it OVERLAPS any of these (checking the whole network, not just
# its network address — so a mixed range like 172.0.0.0/8 that straddles the
# 172.16.0.0/12 private block is caught).
_NON_PUBLIC_NETS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",       # CGNAT
        "127.0.0.0/8",         # loopback
        "169.254.0.0/16",      # link-local
        "172.16.0.0/12",       # private
        "192.0.0.0/24",        # IETF protocol assignments
        "192.0.2.0/24",        # TEST-NET-1 (documentation)
        "192.88.99.0/24",      # 6to4 relay anycast
        "192.168.0.0/16",      # private
        "198.18.0.0/15",       # benchmarking
        "198.51.100.0/24",     # TEST-NET-2 (documentation)
        "203.0.113.0/24",      # TEST-NET-3 (documentation)
        "224.0.0.0/4",         # multicast
        "240.0.0.0/4",         # reserved
        "255.255.255.255/32",  # broadcast
    )
)


def _is_loopback_host(hostname: str | None) -> bool:
    """Exact loopback: `localhost`, an IPv4 in 127.0.0.0/8, or ::1.

    Mirrors the agent's control_plane._is_loopback_host so the LIVEKIT_URL guard
    matches the agent's control-plane transport guard exactly.
    """
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_livekit_url(url: str) -> None:
    """LIVEKIT_URL must be wss://, or ws:// on an EXACT loopback host.

    A remote ws:// URL makes LiveKit send the admin JWT (and, for outbound
    provisioning, the SIP auth credentials) over plaintext. Reject it.
    """
    url = (url or "").strip()
    if not url:
        raise ProvisionConfigError(
            "missing required setting LIVEKIT_URL (ws://127.0.0.1:7880 for the "
            "host-local stack, or wss://<host> for a remote LiveKit)"
        )
    split = urllib.parse.urlsplit(url)
    if split.scheme == "wss":
        return
    if split.scheme == "ws" and _is_loopback_host(split.hostname):
        return
    raise ProvisionConfigError(
        "LIVEKIT_URL must be wss://, or ws:// on an exact loopback host "
        "(127.0.0.0/8, ::1, localhost). A remote ws:// URL would send the "
        f"LiveKit admin JWT (and any SIP credentials) in plaintext: {url!r}"
    )


class ProvisionConfigError(RuntimeError):
    """A self-host provisioning setting is missing or malformed."""


@dataclass(frozen=True)
class OutboundSettings:
    host: str
    from_users: tuple[str, ...]
    auth_username: str
    auth_password: str

    @property
    def from_user(self) -> str:
        """Backward-compatible primary caller ID."""

        return self.from_users[0]


@dataclass(frozen=True)
class SelfhostSettings:
    did: str
    inbound_addresses: tuple[str, ...]
    outbound: OutboundSettings | None


def _required(name: str, environ: dict[str, str]) -> str:
    value = (environ.get(name) or "").strip()
    if not value:
        raise ProvisionConfigError(
            f"missing required setting {name} (see .env.livekit.example / INSTALL.md)"
        )
    return value


def load_settings(environ: dict[str, str] | None = None) -> SelfhostSettings:
    environ = dict(os.environ if environ is None else environ)

    # Refuse to provision over a plaintext remote LiveKit control connection.
    _validate_livekit_url(environ.get("LIVEKIT_URL", ""))

    did = _required("SCH236_TEST_DID_E164", environ).removeprefix("+")
    if _E164_DIGITS.fullmatch(did) is None:
        raise ProvisionConfigError(
            "SCH236_TEST_DID_E164 must be an E.164 number (leading + then digits), e.g. +15551230000"
        )

    addresses: list[str] = []
    for item in _required("DIDWW_INBOUND_ADDRESSES", environ).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError as exc:
            raise ProvisionConfigError(
                f"DIDWW_INBOUND_ADDRESSES entry {item!r} is not a valid IPv4 CIDR"
            ) from exc
        # Reject the WHOLE network if it overlaps any non-public range — a mixed
        # range like 172.0.0.0/8 (which straddles the 172.16/12 private block)
        # must not slip through a network-address-only `is_global` check.
        if network.version != 4 or any(network.overlaps(r) for r in _NON_PUBLIC_NETS):
            raise ProvisionConfigError(
                f"DIDWW_INBOUND_ADDRESSES entry {item!r} overlaps private/loopback/link-local/"
                "documentation/reserved space. Use YOUR carrier's real, fully-public signaling "
                "ranges (from its portal) — a real carrier never sends INVITEs from those ranges. "
                "Note a broad CIDR like 172.0.0.0/8 is rejected because it straddles private space"
            )
        addresses.append(str(network))
    if not addresses:
        raise ProvisionConfigError("DIDWW_INBOUND_ADDRESSES must list at least one public IPv4 CIDR")

    # Outbound is OPTIONAL: supply all four vars to also create an outbound
    # trunk, or none to provision inbound-only.
    present = [name for name in _OUTBOUND_VARS if (environ.get(name) or "").strip()]
    outbound: OutboundSettings | None = None
    if present:
        if len(present) != len(_OUTBOUND_VARS):
            missing = [name for name in _OUTBOUND_VARS if name not in present]
            raise ProvisionConfigError(
                "outbound trunk needs ALL of "
                + ", ".join(_OUTBOUND_VARS)
                + " (or none for inbound-only); missing: "
                + ", ".join(missing)
            )
        host = _required("DIDWW_OUTBOUND_HOST", environ).lower()
        if _HOSTNAME.fullmatch(host) is None:
            raise ProvisionConfigError("DIDWW_OUTBOUND_HOST must be a DNS hostname, e.g. sip.yourcarrier.com")
        from_users = tuple(
            item.strip().removeprefix("+")
            for item in _required("DIDWW_OUTBOUND_FROM_USER", environ).split(",")
            if item.strip()
        )
        if not from_users or any(_E164_DIGITS.fullmatch(value) is None for value in from_users):
            raise ProvisionConfigError(
                "DIDWW_OUTBOUND_FROM_USER must contain one or more comma-separated E.164 numbers"
            )
        if len(set(from_users)) != len(from_users):
            raise ProvisionConfigError("DIDWW_OUTBOUND_FROM_USER contains duplicate caller IDs")
        auth_username = _required("DIDWW_OUTBOUND_AUTH_USERNAME", environ)
        auth_password = _required("DIDWW_OUTBOUND_AUTH_PASSWORD", environ)
        for name, value in (
            ("DIDWW_OUTBOUND_AUTH_USERNAME", auth_username),
            ("DIDWW_OUTBOUND_AUTH_PASSWORD", auth_password),
        ):
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
                raise ProvisionConfigError(f"{name} contains control characters")
        outbound = OutboundSettings(
            host=host,
            from_users=from_users,
            auth_username=auth_username,
            auth_password=auth_password,
        )

    return SelfhostSettings(
        did=did,
        inbound_addresses=tuple(addresses),
        outbound=outbound,
    )


async def provision(*, api_module: Any | None = None, environ: dict[str, str] | None = None) -> dict[str, Any]:
    settings = load_settings(environ)
    if api_module is None:
        from livekit import api as api_module

    client = api_module.LiveKitAPI()
    new_trunk_ids: list[str] = []
    new_rule_ids: list[str] = []
    try:
        # Record the OLD objects we own; we delete them ONLY AFTER the new ones
        # exist, so inbound routing is never left down by a mid-run failure.
        rules = await client.sip.list_dispatch_rule(api_module.ListSIPDispatchRuleRequest())
        old_rule_ids = [r.sip_dispatch_rule_id for r in rules.items if r.name == RULE_NAME]
        inbound = await client.sip.list_inbound_trunk(api_module.ListSIPInboundTrunkRequest())
        outbound = await client.sip.list_outbound_trunk(api_module.ListSIPOutboundTrunkRequest())
        old_trunk_ids = [
            t.sip_trunk_id
            for t in (*inbound.items, *outbound.items)
            if t.name in {INBOUND_TRUNK_NAME, OUTBOUND_TRUNK_NAME}
        ]

        # CREATE the replacements first (the old objects keep routing calls).
        inbound_trunk = await client.sip.create_inbound_trunk(
            api_module.CreateSIPInboundTrunkRequest(
                trunk=api_module.SIPInboundTrunkInfo(
                    name=INBOUND_TRUNK_NAME,
                    numbers=[settings.did],
                    allowed_addresses=list(settings.inbound_addresses),
                    media=api_module.SIPMediaConfig(
                        only_listed_codecs=True,
                        codecs=[api_module.SIPCodec(name="PCMU", rate=8_000)],
                    ),
                )
            )
        )
        new_trunk_ids.append(inbound_trunk.sip_trunk_id)

        outbound_trunk_id = "not_created"
        if settings.outbound is not None:
            outbound_trunk = await client.sip.create_outbound_trunk(
                api_module.CreateSIPOutboundTrunkRequest(
                    trunk=api_module.SIPOutboundTrunkInfo(
                        name=OUTBOUND_TRUNK_NAME,
                        address=settings.outbound.host,
                        from_host=settings.outbound.host,
                        transport=api_module.SIPTransport.SIP_TRANSPORT_UDP,
                        numbers=list(settings.outbound.from_users),
                        auth_username=settings.outbound.auth_username,
                        auth_password=settings.outbound.auth_password,
                        media=api_module.SIPMediaConfig(
                            only_listed_codecs=True,
                            codecs=[api_module.SIPCodec(name="PCMU", rate=8_000)],
                        ),
                    )
                )
            )
            new_trunk_ids.append(outbound_trunk.sip_trunk_id)
            outbound_trunk_id = outbound_trunk.sip_trunk_id

        rule = await client.sip.create_dispatch_rule(
            api_module.CreateSIPDispatchRuleRequest(
                dispatch_rule=api_module.SIPDispatchRuleInfo(
                    name=RULE_NAME,
                    trunk_ids=[inbound_trunk.sip_trunk_id],
                    numbers=[settings.did],
                    rule=api_module.SIPDispatchRule(
                        dispatch_rule_individual=api_module.SIPDispatchRuleIndividual(room_prefix=ROOM_PREFIX)
                    ),
                    room_config=api_module.RoomConfiguration(
                        empty_timeout=30,
                        departure_timeout=10,
                        agents=[api_module.RoomAgentDispatch(agent_name=AGENT_NAME)],
                    ),
                )
            )
        )
        new_rule_ids.append(rule.sip_dispatch_rule_id)

        # New routing is live -> retire the OLD objects. A delete FAILURE here
        # is NOT swallowed: a lingering old trunk keeps its (possibly stale,
        # more-permissive) allowlist active, so we name the stale ids in the
        # result, warn prominently, and exit nonzero — the operator must delete
        # them by hand rather than believe provisioning fully succeeded.
        stale: list[str] = []
        for rid in old_rule_ids:
            try:
                await client.sip.delete_dispatch_rule(
                    api_module.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rid)
                )
            except Exception as exc:
                stale.append(f"dispatch_rule:{rid} ({type(exc).__name__})")
        for tid in old_trunk_ids:
            try:
                await client.sip.delete_trunk(api_module.DeleteSIPTrunkRequest(sip_trunk_id=tid))
            except Exception as exc:
                stale.append(f"trunk:{tid} ({type(exc).__name__})")

        result = {
            "agent_name": AGENT_NAME,
            "dispatch_rule_id": rule.sip_dispatch_rule_id,
            "inbound_trunk_id": inbound_trunk.sip_trunk_id,
            "outbound_trunk_id": outbound_trunk_id,
            "codec": "PCMU/8000",
            "inbound_allowlist_cidrs": len(settings.inbound_addresses),
            "stale_objects": stale,
        }
        print("SELFHOST_PROVISIONED " + json.dumps(result, sort_keys=True), flush=True)
        if stale:
            print(
                "WARNING: could not delete the previous SIP object(s): "
                + ", ".join(stale)
                + " — they may keep a stale, more-permissive allowlist active. "
                "Delete them manually (LiveKit SIP list/delete), then rerun.",
                file=sys.stderr,
                flush=True,
            )
        return result
    except Exception:
        # Roll back ONLY the partially-created NEW objects; leave the OLD ones
        # intact so a mid-run failure never leaves inbound routing down.
        for rid in new_rule_ids:
            try:
                await client.sip.delete_dispatch_rule(
                    api_module.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id=rid)
                )
            except Exception:
                pass
        for tid in reversed(new_trunk_ids):
            try:
                await client.sip.delete_trunk(api_module.DeleteSIPTrunkRequest(sip_trunk_id=tid))
            except Exception:
                pass
        raise
    finally:
        await client.aclose()


def _redact(message: str, settings_error: bool) -> str:
    # Our own ProvisionConfigError messages are static and safe to show. For any
    # other exception, still surface the message (LiveKit API errors are server
    # status strings, not request bodies) but defensively strip the outbound
    # auth password if it somehow appears, and bound the length.
    text = message
    if not settings_error:
        pw = (os.environ.get("DIDWW_OUTBOUND_AUTH_PASSWORD") or "").strip()
        if pw and pw in text:
            text = text.replace(pw, "***")
    return text[:500]


def main() -> int:
    try:
        result = asyncio.run(provision())
        # Provisioning of the NEW routing succeeded, but if we could not delete
        # the previous objects, exit nonzero so the operator does not treat a
        # stale, possibly more-permissive allowlist as a clean success.
        return 1 if result.get("stale_objects") else 0
    except ProvisionConfigError as exc:
        print(
            "SELFHOST_PROVISION_FAILED "
            + json.dumps(
                {"error_type": "ProvisionConfigError", "error": _redact(str(exc), True)},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 — surface the real error for the installer
        print(
            "SELFHOST_PROVISION_FAILED "
            + json.dumps(
                {"error_type": type(exc).__name__, "error": _redact(str(exc), False)},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

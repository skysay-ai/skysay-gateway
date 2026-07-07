#!/usr/bin/env python3
"""Originate an outbound call into a self-hosted Asterisk ARI Stasis app.

Standalone, control-plane-free CLI for OpenPhonex self-hosters (SCH-202 M4). It
POSTs to ARI ``/channels`` on the customer's OWN Asterisk to originate an
outbound call, passing an explicit ``callerId`` (a plain ``channel originate``
can present an anonymous caller that DIDWW rejects with 403) and the outbound
DIRECTION as the Stasis ``appArgs``.

Depends only on the Python standard library and ``openphonex_common.config``; it
must never import the private control plane (``telephony_ai.app`` / ``server`` /
``store`` / ``asterisk``). The queue-driven control-plane worker
(``telephony_ai.workers.outbound_originate``) is a separate, distinct path that
stays in the control plane.

Run as ``python -m openphonex_gateway.originate <target>`` or via the
``openphonex-originate`` console script.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from openphonex_common.config import merged_env


def normalize_phone_digits(value: str) -> str:
    return re.sub(r"\D+", "", value or "")


def normalize_destination_digits(value: str) -> str:
    digits = normalize_phone_digits(value)
    if value.strip().startswith("00"):
        digits = digits[2:]
    if not digits:
        raise SystemExit("Target phone number must contain digits")
    return digits


def require(env: dict[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise SystemExit(f"Missing required variable: {name}")
    return value


def caller_id_for_profile(env: dict[str, str], profile: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if profile == "us":
        return require(env, "DIDWW_CALLER_ID_US")
    if profile == "ee":
        return require(env, "DIDWW_CALLER_ID_EE")
    return require(env, "DIDWW_CALLER_ID")


def endpoint_for_caller_id(env: dict[str, str], caller_id: str, explicit_endpoint: str | None) -> str:
    if explicit_endpoint:
        return explicit_endpoint
    digits = normalize_phone_digits(caller_id)
    if digits and digits == normalize_phone_digits(env.get("DIDWW_CALLER_ID_EE", "")):
        return "didww-outbound-ee"
    if digits and digits == normalize_phone_digits(env.get("DIDWW_CALLER_ID_US", "")):
        return "didww-outbound-us"
    return "didww-outbound"


def ari_post(base_url: str, user: str, password: str, path: str, params: dict[str, str]):
    url = base_url.rstrip("/") + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST", data=b"")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", "Basic " + token)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    return json.loads(data.decode()) if data else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Originate an outbound call into an Asterisk ARI Stasis app")
    parser.add_argument("target", help="Phone number, e.g. +37253231038")
    parser.add_argument("--env-file", default=None, help="Path to an env file with ARI/DIDWW settings")
    parser.add_argument("--ari-url", default=None, help="ARI base URL (default http://127.0.0.1:8088/ari or ASTERISK_ARI_URL)")
    parser.add_argument("--app", default=None, help="Stasis application name (default voicebot or ASTERISK_ARI_APP)")
    parser.add_argument("--app-args", default=None, help="Stasis appArgs / call direction (default outbound)")
    parser.add_argument("--from-cli", choices=["default", "us", "ee"], default="default", help="DIDWW caller ID profile to present")
    parser.add_argument("--endpoint", default=None, help="Override the PJSIP endpoint profile")
    parser.add_argument("--outbound-host", default=None, help="Override DIDWW Voice OUT host")
    parser.add_argument("--caller-id", default=None, help="Override the presented caller ID")
    parser.add_argument("--timeout", default=None, help="Originate timeout in seconds (default 45)")
    parser.add_argument("--dry-run", action="store_true", help="Print the ARI originate payload without placing the call")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser() if args.env_file else None
    env = merged_env(env_path)

    caller_id = caller_id_for_profile(env, args.from_cli, args.caller_id)
    endpoint_profile = endpoint_for_caller_id(env, caller_id, args.endpoint)
    outbound_host = args.outbound_host or require(env, "DIDWW_OUTBOUND_HOST")
    target_digits = normalize_destination_digits(args.target)
    endpoint = f"PJSIP/{endpoint_profile}/sip:{target_digits}@{outbound_host}"

    app = args.app or env.get("ASTERISK_ARI_APP", "voicebot")
    app_args = args.app_args or env.get("ASTERISK_ARI_APP_ARGS", "outbound")
    ari_url = args.ari_url or env.get("ASTERISK_ARI_URL", "http://127.0.0.1:8088/ari")
    timeout = args.timeout or env.get("ASTERISK_ARI_TIMEOUT", "45")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "endpoint": endpoint,
                    "target_digits": target_digits,
                    "app": app,
                    "app_args": app_args,
                    "caller_id": caller_id,
                    "endpoint_profile": endpoint_profile,
                    "outbound_host": outbound_host,
                    "ari_url": ari_url,
                }
            )
        )
        return 0

    user = require(env, "ASTERISK_ARI_USER")
    password = require(env, "ASTERISK_ARI_PASSWORD")
    channel = ari_post(
        ari_url,
        user,
        password,
        "/channels",
        {
            "endpoint": endpoint,
            "app": app,
            "appArgs": app_args,
            "callerId": caller_id,
            "timeout": str(timeout),
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "channel_id": channel.get("id") if isinstance(channel, dict) else None,
                "channel_name": channel.get("name") if isinstance(channel, dict) else None,
                "state": channel.get("state") if isinstance(channel, dict) else None,
                "caller_id": caller_id,
                "endpoint_profile": endpoint_profile,
                "outbound_host": outbound_host,
                "target_digits": target_digits,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
#
# OpenPhonex LiveKit self-host smoke test.
#
# Run this AFTER `docker compose -f docker-compose.livekit.yml up -d --build`
# (and after the one-shot `--profile provision run --rm provision`) to confirm
# the stack is actually able to take a call:
#
#   bash selfhost-smoketest.sh
#
# It verifies, with a clear PASS/FAIL per check:
#   1. tooling (docker + compose v2 + curl)
#   2. the three filled-in config files exist and hold no template values
#   3. every required variable in .env.livekit is set
#   4. the control-plane URL shape (https, or http on loopback — the agent
#      refuses anything else at runtime)
#   5. all five services are running (redis, livekit, sip, agents, reconciler)
#  5b. registration topology only: the sidecar is configured, running, armed,
#      actually holding a registration, its NAT keepalive is under the 30s
#      floor, and 5060/udp is firewalled statefully (a stateless DROP would
#      also block replies to outbound calls)
#   6. redis answers PONG
#   7. livekit-server answers on http://127.0.0.1:7880
#   8. the agent registered with LiveKit and prewarmed (log evidence)
#   9. the control plane accepts the gateway token (number-rules read)
#  10. the same gateway token is accepted on the OTHER surface a call needs
#      (MCP tools/list), plus any legacy organization worker key still set
#  11. sessions/ + call-evidence/ are writable by the in-container agent uid
#
# Exit code 0 = every hard check passed -> place a test call to your DID.
#           2 = everything that COULD be checked passed, but something could
#               not be checked at all (e.g. firewall rules need root). Not a
#               pass: an unchecked security boundary must not read as ready.
# The script is read-only: it places no calls and changes nothing.

set -u

cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.livekit.yml"
COMPOSE="docker compose -f ${COMPOSE_FILE}"
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
UNVERIFIED_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); printf '[PASS] %s\n' "$1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); printf '[FAIL] %s\n       -> %s\n' "$1" "$2"; }
warn() { WARN_COUNT=$((WARN_COUNT + 1)); printf '[WARN] %s\n       -> %s\n' "$1" "$2"; }
# A check that could not be performed at all. Distinct from WARN because it
# must NOT be reportable as PASS: telling an operator the stack is call-ready
# while a security boundary went unchecked is worse than saying nothing.
unverified() { UNVERIFIED_COUNT=$((UNVERIFIED_COUNT + 1)); printf '[????] %s\n       -> %s\n' "$1" "$2"; }

# Run curl with a bearer token WITHOUT putting it in argv (which is visible in
# `ps`). The token is fed as a curl config on stdin (`-K -`), via the printf
# builtin (no separate process); extra curl args are passed through.
bearer_curl() {
  _tok="$1"; shift
  printf 'header = "Authorization: Bearer %s"\n' "$_tok" | curl -K - "$@"
}

# Classify the control-plane URL with the agent's OWN rule, via python3 — NOT
# fragile shell parsing. urllib.parse.urlsplit extracts the true host (so
# userinfo / query / fragment tricks like `http://remote?@localhost` cannot
# masquerade as loopback), and we reuse control_plane._is_loopback_host from the
# exported agent when importable (single source of truth), else an identical
# inline check. Prints "<host> <mode>" (mode: https|loopback|reject) and exits 0
# iff the URL is safe to send bearer tokens to (https anywhere, or http on an
# EXACT loopback host: 127.0.0.0/8, ::1, localhost).
cp_classify() {
  _agent_dir="$(cd "$(dirname "$0")/../../../livekit_agent" 2>/dev/null && pwd || true)"
  python3 - "$1" "$_agent_dir" <<'PY'
import sys, urllib.parse
url = sys.argv[1]
agent_dir = sys.argv[2] if len(sys.argv) > 2 else ""

def _inline_loopback(host):
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    import ipaddress
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

is_loopback = _inline_loopback
if agent_dir:
    sys.path.insert(0, agent_dir)
    try:
        import control_plane
        is_loopback = control_plane._is_loopback_host  # single source of truth
    except Exception:
        pass

sp = urllib.parse.urlsplit(url)
host = sp.hostname or ""
loopback = "loopback" if is_loopback(host) else "remote"
# Prints "<host> <mode> <loopback>". Proxy-bypass is decided from the LOOPBACK
# flag INDEPENDENT of scheme, so an https-on-loopback control plane also
# bypasses the proxy. Exit 0 iff safe to send bearer tokens.
if sp.scheme == "https":
    print(host, "https", loopback); sys.exit(0)
if sp.scheme == "http" and loopback == "loopback":
    print(host, "loopback", loopback); sys.exit(0)
print(host, "reject", loopback); sys.exit(1)
PY
}

# --- 1. tooling ---------------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  pass "docker is installed"
else
  fail "docker is installed" "Install Docker Engine + Compose v2 first."
  echo "Cannot continue without docker."; exit 1
fi
if docker compose version >/dev/null 2>&1; then
  pass "docker compose v2 is available"
else
  fail "docker compose v2 is available" "Install the Compose v2 plugin (docker compose, not docker-compose v1)."
  exit 1
fi
if command -v curl >/dev/null 2>&1; then
  pass "curl is installed"
else
  fail "curl is installed" "Install curl; the HTTP checks below need it."
fi
# python3 was always a hard dependency here -- the control-plane URL classifier,
# the service-account key check and the capability-manifest read all shell out
# to it -- but nothing said so, so its absence surfaced as whichever of those
# three ran first, blaming the wrong thing.
if command -v python3 >/dev/null 2>&1; then
  pass "python3 is installed"
else
  fail "python3 is installed" "Install python3; the control-plane URL classifier, the capability-manifest read and the service-account key check all use it."
fi

# --- 2. filled-in config files ------------------------------------------------
for f in livekit.yaml sip.yaml .env.livekit .env.registrar; do
  if [ ! -f "$f" ]; then
    if [ "$f" = .env.registrar ]; then
      # Every deployment creates this one; only a NAT'd host fills it in. The
      # provision step reads it to choose the topology, so a missing file fails
      # that step rather than silently defaulting.
      fail "config file $f exists" "cp ${f}.example ${f} — every deployment needs this file. On a host with a public IP, copy it and leave the core four values blank; that means 'static endpoint'. See SELFHOST.md '1. Configure'."
    else
      fail "config file $f exists" "cp ${f}.example ${f} (see SELFHOST.md '1. Configure') and fill it in."
    fi
  elif grep -q "changeme" "$f" 2>/dev/null; then
    fail "config file $f is filled in" "$f still contains template 'changeme' values."
  else
    pass "config file $f exists and is filled in"
  fi
done
[ -f .env.livekit ] || { echo "Cannot continue without .env.livekit."; exit 1; }

# --- 3. required variables ----------------------------------------------------
# Non-evaluating dotenv read: parse KEY=VALUE LITERALLY. NEVER `source` the file
# — a `$`, `` ` ``, or `$(...)` in a credential would be shell-expanded /
# executed. `read -r` + `export "$k=$v"` assigns the value verbatim (the RHS is
# not re-scanned for substitution). (The container entrypoint DOES source this
# file, so keep values free of shell metacharacters — see .env.livekit.example.)
# Which names each file assigned a NON-EMPTY value to. Two facts make this
# necessary rather than pedantic:
#
#   * the agents container is handed .env.livekit as carrier.env and nothing
#     else, so a value that happens to be exported in the shell running this
#     script is invisible to the worker -- and the sidecar likewise sees only
#     .env.registrar. One shared list would let a credential in the wrong file
#     satisfy the wrong container.
#   * the two files are read into one process environment, so a name in both
#     leaves the shell holding whichever was read last. Recording emptiness per
#     file, and asking membership rather than re-reading the value, keeps that
#     shadowing out of the answer.
#
# Named ENV_KEYS_* rather than LIVEKIT_*: the release guard reads a
# LIVEKIT_-prefixed name in this script as a setting an operator must find
# documented in an example file, and these two are bookkeeping.
ENV_KEYS_LIVEKIT=""
ENV_KEYS_REGISTRAR=""
_LAST_FILE_KEYS=""

_strip_leading_blanks() {
  _stripped="$1"
  while :; do
    case "$_stripped" in
      ' '*) _stripped="${_stripped# }" ;;
      "$(printf '\t')"*) _stripped="${_stripped#"$(printf '\t')"}" ;;
      *) break ;;
    esac
  done
  printf '%s' "$_stripped"
}

_assigned() {
  case " $1 " in
    *" $2 "*) return 0 ;;
  esac
  return 1
}

# Assigned a non-empty value in .env.livekit -- the agents container's world.
configured() { _assigned "$ENV_KEYS_LIVEKIT" "$1"; }
# Assigned a non-empty value in .env.registrar -- the sidecar's world.
configured_registrar() { _assigned "$ENV_KEYS_REGISTRAR" "$1"; }

read_env_file() {
  _LAST_FILE_KEYS=""
  [ -f "$1" ] || return 0
  while IFS= read -r _line || [ -n "$_line" ]; do
    # Leading whitespace and an `export ` prefix are both valid dotenv --
    # Compose accepts them, the container entrypoint `.`-sources this file
    # where `export NAME=value` is plain shell, and the SIP_REGISTRATION
    # contract below greps for exactly that prefix. Strip them rather than
    # skipping the line: skipping means a variable the worker receives and
    # this gate reports as missing.
    _line="$(_strip_leading_blanks "$_line")"
    case "$_line" in
      export' '*|export"$(printf '\t')"*)
        _line="$(_strip_leading_blanks "${_line#export}")"
        ;;
    esac
    case "$_line" in ''|'#'*) continue ;; esac
    _key="${_line%%=*}"
    case "$_key" in
      "$_line") continue ;;             # no '=' on the line
      *[!A-Za-z0-9_]*) continue ;;      # not a shell identifier -> skip
    esac
    _val="${_line#*=}"
    # Strip a trailing CR so the checks below inspect the intended value. This
    # is a courtesy, NOT a repair -- the container sources the file verbatim.
    # The CRLF check after the reads is what tells the operator.
    _val="${_val%$(printf '\r')}"
    case "$_val" in                     # strip one layer of surrounding quotes
      \"*\") _val="${_val#\"}"; _val="${_val%\"}" ;;
      \'*\') _val="${_val#\'}"; _val="${_val%\'}" ;;
    esac
    export "$_key=$_val"
    # LAST assignment in the file wins, emptiness included: a duplicated line
    # that blanks an earlier value is what the container ends up with, so a
    # key recorded from the first of the pair would report a credential the
    # worker does not have.
    if [ -n "$_val" ]; then
      _assigned "$_LAST_FILE_KEYS" "$_key" || _LAST_FILE_KEYS="$_LAST_FILE_KEYS $_key"
    else
      _rebuilt=""
      for _seen in $_LAST_FILE_KEYS; do
        [ "$_seen" = "$_key" ] || _rebuilt="$_rebuilt $_seen"
      done
      _LAST_FILE_KEYS="$_rebuilt"
    fi
  done < "$1"
}
# The sidecar's credentials live in their own file (see .env.registrar.example
# for why they are not merged). Read it with the SAME non-evaluating parser --
# and read it FIRST, so .env.livekit wins the shared process environment. That
# is the provisioner's own precedence (it receives .env.registrar as
# environment and then sources .env.livekit over the top), and it is what
# makes the VALUES this script goes on to inspect -- the control-plane URL,
# the service-account JSON, the DID list -- the ones the agents container is
# handed. The reverse order left a name present in both files validated from
# the wrong file. SIP_REGISTRATION_* cannot be shadowed the other way: the
# contract below hard-fails any of those names appearing in .env.livekit.
read_env_file .env.registrar
ENV_KEYS_REGISTRAR="$_LAST_FILE_KEYS"
read_env_file .env.livekit
ENV_KEYS_LIVEKIT="$_LAST_FILE_KEYS"

# CRLF is not cosmetic here. The container entrypoint SOURCES this file as
# shell (`. /run/sch172/carrier.env`) rather than parsing it as dotenv, so a
# carriage return stays inside every value: tokens 401, the control-plane URL
# fails to classify, the DID is not E.164. The reader above strips CR before
# the checks so they inspect what was meant -- which is exactly how a CRLF
# file could otherwise reach RESULT: PASS while no value in the container is
# the one on screen. Say so instead.
for _f in .env.livekit .env.registrar; do
  [ -f "$_f" ] || continue
  if LC_ALL=C grep -q "$(printf '\r')" "$_f" 2>/dev/null; then
    fail "$_f has Unix line endings" \
      "it contains CR line endings. The container sources this file as shell, so every value keeps a trailing carriage return — the checks below strip it and would pass, the worker cannot. Convert it: LC_ALL=C sed -i.bak 's/\r\$//' $_f (or dos2unix $_f), then re-run."
  else
    pass "$_f has Unix line endings"
  fi
done

# THE CONTRACT: SIP_REGISTRATION_* belongs in .env.registrar and nowhere else.
# This is not tidiness. The provision container receives .env.registrar as
# environment and then sources .env.livekit over the top, so a leftover line
# here -- including an EMPTY placeholder from an older template -- wins and
# blanks the real value. The provisioner then sees an empty block, builds the
# static-endpoint trunk with a carrier address allowlist, and every INVITE the
# sidecar forwards is rejected by it: total inbound outage, everything green.
# Read precedence differs between the provisioner and this script by
# construction, so the only safe rule is that the variables appear once.
#
# KNOWN EDGE, deliberately not guarded: this matches one assignment per line,
# so a contrived `FOO=1 SIP_REGISTRATION_REGISTRAR=` slips past. Every
# realistic accident -- a copied template, a leftover block, an `export` line
# -- is caught, and adding a parser for the contrived case would be the next
# guard on a hazard the plumbing creates rather than a fix for it. If this area
# produces another interaction defect, the answer is to remove the hazard: have
# the provision entrypoint drop SIP_REGISTRATION_* when it sources
# carrier.env, so a stray line cannot matter. Not another check.
STRAY_REG_VARS="$(grep -oE '^[[:space:]]*(export[[:space:]]+)?SIP_REGISTRATION_[A-Z0-9_]*[[:space:]]*=' .env.livekit 2>/dev/null | grep -oE 'SIP_REGISTRATION_[A-Z0-9_]*' | sort -u | tr '\n' ' ')"
if [ -n "$STRAY_REG_VARS" ]; then
  fail "SIP_REGISTRATION_* appears only in .env.registrar" \
    "found in .env.livekit: ${STRAY_REG_VARS}— delete those lines (even empty ones). The provision step sources .env.livekit AFTER .env.registrar, so these override the real values, and provisioning silently builds the wrong trunk. See SELFHOST.md 'Which file holds what, and who reads it'."
else
  pass "SIP_REGISTRATION_* is confined to .env.registrar"
fi

# The bootstrap key must never have reached this file. selfhost-bootstrap.sh
# takes it as an environment variable (or reads it with no echo) precisely so
# there is nothing on the box to delete afterwards; a line here would quietly
# turn "revoke it when you are done" into a secret nobody remembers to remove,
# and it is an ORGANISATION-scoped key, which is the blast radius the
# deployment-bound token exists to avoid.
STRAY_BOOTSTRAP_KEY="$(grep -cE '^[[:space:]]*(export[[:space:]]+)?OPENPHONEX_BOOTSTRAP_API_KEY[[:space:]]*=' .env.livekit 2>/dev/null || true)"
if [ "${STRAY_BOOTSTRAP_KEY:-0}" -gt 0 ] 2>/dev/null; then
  fail "the bootstrap key is not stored on this host" \
    "OPENPHONEX_BOOTSTRAP_API_KEY is assigned in .env.livekit. Nothing reads it from there -- it is an organization-scoped key that only selfhost-bootstrap.sh needs, once, from the environment. Delete the line and revoke that key in the dashboard."
else
  pass "the bootstrap key is not stored on this host"
fi

# gateway-state.json records which deployment this box registered. If it and
# .env.livekit name different gateways, one of them is a leftover from another
# install: the stack would run as one deployment while a re-run of the
# bootstrap repaired the other, and nothing else in this script would notice.
if [ -f gateway-state.json ]; then
  STATE_GATEWAY_ID="$(python3 - <<'"'"'PY'"'"' 2>/dev/null || true
import json
try:
    with open("gateway-state.json", encoding="utf-8") as handle:
        document = json.load(handle)
except (OSError, ValueError):
    document = {}
print(str((document or {}).get("gateway_id") or ""))
PY
)"
  if [ -z "$STATE_GATEWAY_ID" ]; then
    fail "gateway-state.json agrees with .env.livekit" \
      "gateway-state.json is unreadable or names no gateway_id. Delete it and re-run selfhost-bootstrap.sh, which will adopt this deployment from .env.livekit."
  elif [ -z "${OPENPHONEX_GATEWAY_ID:-}" ]; then
    fail "gateway-state.json agrees with .env.livekit" \
      "gateway-state.json names $STATE_GATEWAY_ID but .env.livekit sets no OPENPHONEX_GATEWAY_ID. Re-run selfhost-bootstrap.sh to fill it in."
  elif [ "$STATE_GATEWAY_ID" != "$OPENPHONEX_GATEWAY_ID" ]; then
    fail "gateway-state.json agrees with .env.livekit" \
      "gateway-state.json names $STATE_GATEWAY_ID but .env.livekit runs as $OPENPHONEX_GATEWAY_ID. One is a leftover from another install; delete gateway-state.json and re-run selfhost-bootstrap.sh to adopt the deployment this stack actually uses."
  else
    pass "gateway-state.json agrees with .env.livekit ($STATE_GATEWAY_ID)"
  fi
fi

# Variables the stack cannot start without, whatever the agent is configured to
# say. Provider keys are deliberately NOT in this list -- see below.
REQUIRED_VARS="LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET \
OPENPHONEX_CONTROL_PLANE_URL OPENPHONEX_GATEWAY_ID OPENPHONEX_GATEWAY_API_TOKEN \
OPENPHONEX_ORGANIZATION_ID SCH236_TEST_DID_E164"
MISSING=""
for v in $REQUIRED_VARS; do
  configured "$v" || MISSING="$MISSING $v"
done
if [ -z "$MISSING" ]; then
  pass "all required .env.livekit variables are set"
else
  fail "all required .env.livekit variables are set" "missing:$MISSING (each is documented in .env.livekit.example)"
fi

# WHICH provider keys you need is decided by the voice stack your agent
# publishes, and this script cannot see that stack. But it does not have to
# guess either: the worker builds a known set of providers, so what is
# *structurally* required follows from that set.
#
#   LLM  -- only gemini/google, so GEMINI_API_KEY is always needed.
#   STT  -- at least one of the STT providers this image can build.
#   TTS  -- never demanded on its own: gemini_tts reuses the Gemini key.
#
# The provider set is deliberately NOT written out below. Part of it lives in
# the capability manifest that ships beside the agent sources this compose file
# builds from, and the copy that used to be pinned here went stale the first
# time a provider was added: a correct Soniox-only install was hard-FAILed for
# having "no STT credential", and an install holding only the three older keys
# was told it covered every supported provider while a Tamil agent had nothing
# to run on. So read the manifest instead of restating it.
#
# The three older providers predate the manifest and declare no credential
# variable in it, so they are still named here. Every capability added since
# resolves from the artifact, and adding another one needs no edit to this file.
# ASK Compose where it builds the agent image from; do not imitate it. The
# context is `${LIVEKIT_AGENT_CONTEXT:-...}`, and resolving that the way Compose
# does means shell-versus-project-`.env` precedence, empty-value semantics,
# quoting and inline-comment stripping -- a second implementation of Compose's
# parser, which is exactly the thing that would drift from the first. `config`
# resolves and prints; it starts nothing.
AGENT_CONTEXT="$($COMPOSE config --format json 2>/dev/null | python3 -c '
import json, sys

try:
    document = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
service = (document.get("services") or {}).get("agents") or {}
context = str((service.get("build") or {}).get("context") or "").strip()
if not context:
    raise SystemExit(1)
print(context)
' 2>/dev/null)" || AGENT_CONTEXT=""
# Compose could not answer (older Compose, an invalid file, no python3). Fall
# back to the same default the compose file declares, and let the manifest
# check below report the path it actually tried.
[ -n "$AGENT_CONTEXT" ] || AGENT_CONTEXT="${LIVEKIT_AGENT_CONTEXT:-../../../livekit_agent}"
CAPABILITY_MANIFEST="${AGENT_CONTEXT}/capabilities/capabilities.json"

# component:VARIABLE pairs for every runnable capability that declares a
# managed credential variable. Same default path as the agents service's build
# context in docker-compose.livekit.yml, so the two cannot drift apart.
if REGISTRY_CREDENTIALS=$(python3 - "$CAPABILITY_MANIFEST" <<'PY' 2>/dev/null
import json, re, sys

# The names below are expanded through `eval` in the shell, so accept only
# shell identifiers. The manifest ships with the repo rather than arriving
# from anywhere, but a file-derived string reaching eval is not a property to
# leave resting on that.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

with open(sys.argv[1], encoding="utf-8") as handle:
    doc = json.load(handle)
rows = set()
for capability in doc.get("capabilities") or []:
    if capability.get("runnable") is not True:
        continue
    component = str(capability.get("component") or "").strip()
    variable = str((capability.get("credential") or {}).get("managed_env") or "").strip()
    if component in {"stt", "tts", "llm"} and IDENTIFIER.match(variable):
        rows.add(f"{component}:{variable}")
print(" ".join(sorted(rows)))
PY
); then
  MANIFEST_READ=1
else
  MANIFEST_READ=0
  REGISTRY_CREDENTIALS=""
  if [ ! -f "$CAPABILITY_MANIFEST" ]; then
    fail "the worker capability manifest is readable" \
      "no file at ${CAPABILITY_MANIFEST} — this checkout is incomplete. That file ships beside the agent sources docker-compose.livekit.yml builds the image from, so if it is missing, \`docker compose up -d --build\` has nothing to build either. Re-clone the repository, or set LIVEKIT_AGENT_CONTEXT to the agent sources."
  else
    fail "the worker capability manifest is readable" \
      "${CAPABILITY_MANIFEST} exists but could not be parsed. Either python3 is missing (see the tooling checks above) or the file is not valid JSON — restore it from the repository rather than editing it."
  fi
fi

# Bash 3.2 on stock macOS has no associative arrays: keep these as
# space-delimited lists and dedupe with a substring test.
STT_REGISTRY_VARS=""
ALL_REGISTRY_VARS=""
for _entry in $REGISTRY_CREDENTIALS; do
  _component=${_entry%%:*}
  _variable=${_entry#*:}
  case " $ALL_REGISTRY_VARS " in
    *" $_variable "*) ;;
    *) ALL_REGISTRY_VARS="$ALL_REGISTRY_VARS $_variable" ;;
  esac
  if [ "$_component" = "stt" ]; then
    case " $STT_REGISTRY_VARS " in
      *" $_variable "*) ;;
      *) STT_REGISTRY_VARS="$STT_REGISTRY_VARS $_variable" ;;
    esac
  fi
done

GEMINI_SET=0; configured GEMINI_API_KEY && GEMINI_SET=1
STT_SET=0
STT_PATH_UNCHECKED=0
configured DEEPGRAM_API_KEY && STT_SET=1
for _variable in $STT_REGISTRY_VARS; do
  configured "$_variable" && STT_SET=1
done
# A path is a CONTAINER path -- the key file is mounted read-only into the
# agents service -- so this script, running on the host, cannot open it. Say
# that rather than treating "non-empty" as "valid".
if configured GOOGLE_APPLICATION_CREDENTIALS; then
  STT_SET=1
  STT_PATH_UNCHECKED=1
fi
# The inline variant IS checkable here, and a malformed value is a deterministic
# first-call failure worth catching now: the agent needs real service-account
# fields, so an empty object or a truncated paste must not read as configured.
if configured GOOGLE_APPLICATION_CREDENTIALS_JSON; then
  if GOOGLE_APPLICATION_CREDENTIALS_JSON="$GOOGLE_APPLICATION_CREDENTIALS_JSON" python3 -c '
import json, os, sys
try:
    doc = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
except Exception:
    sys.exit(1)
if not isinstance(doc, dict):
    sys.exit(1)
missing = [k for k in ("type", "client_email", "private_key") if not doc.get(k)]
sys.exit(2 if missing else 0)
' 2>/dev/null; then
    STT_SET=1
  else
    fail "GOOGLE_APPLICATION_CREDENTIALS_JSON is a service-account key" \
      "it does not parse as JSON, or is missing type/client_email/private_key — re-paste the whole downloaded key on one line, single-quoted (it contains \")"
  fi
fi

STT_REGISTRY_HINT=""
[ -n "$STT_REGISTRY_VARS" ] && STT_REGISTRY_HINT=", or one of:$STT_REGISTRY_VARS"
if [ "$GEMINI_SET" -eq 0 ]; then
  fail "GEMINI_API_KEY is set" \
    "every supported stack resolves its model to Gemini, and Gemini TTS reuses the same key"
elif [ "$STT_SET" -eq 0 ]; then
  fail "an STT credential is set" \
    "set DEEPGRAM_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS/_JSON for a Google Cloud STT path${STT_REGISTRY_HINT}; no supported stack transcribes without one"
else
  # Deliberately the WEAKER claim, and it says so. This proves a runnable stack
  # is *possible* with the keys present; it cannot prove your agent's published
  # stack is covered, because that would mean reading the agent, and the
  # self-host worker key is scoped to calls:read + calls:write on purpose.
  #
  # The gap this leaves surfaces on the first call rather than here: an agent
  # publishing a provider whose key is absent -- ElevenLabs TTS with no
  # ELEVENLABS_API_KEY, Deepgram STT when only the Google credentials are set,
  # a Tamil stack with no Soniox or Fish key.
  MISSING_OPTIONAL=""
  for _variable in ELEVENLABS_API_KEY DEEPGRAM_API_KEY $ALL_REGISTRY_VARS; do
    case " $MISSING_OPTIONAL " in
      *" $_variable "*) continue ;;
    esac
    configured "$_variable" || MISSING_OPTIONAL="$MISSING_OPTIONAL $_variable"
  done
  # Google is the fifth provider this image builds, and it is the ONLY
  # transcriber for si-LK. Leaving it out of the count let an install with
  # every other key be told it covered everything, while a Sinhala agent had
  # no credential at all. It is a pair (path OR inline JSON), so one term.
  if ! configured GOOGLE_APPLICATION_CREDENTIALS && ! configured GOOGLE_APPLICATION_CREDENTIALS_JSON; then
    MISSING_OPTIONAL="$MISSING_OPTIONAL GOOGLE_APPLICATION_CREDENTIALS(_JSON)"
  fi
  STT_CAVEAT=""
  [ "$STT_PATH_UNCHECKED" -eq 1 ] && STT_CAVEAT=" GOOGLE_APPLICATION_CREDENTIALS names a path inside the agents container, so its contents are not checked here."
  # Without the manifest the enumeration above is incomplete, so the stronger
  # sentence would be a claim this run has no evidence for.
  MANIFEST_CAVEAT=""
  [ "$MANIFEST_READ" -eq 0 ] && MANIFEST_CAVEAT=" The capability manifest could not be read, so the providers it declares were not considered at all."
  if [ -n "$MISSING_OPTIONAL" ] || [ "$MANIFEST_READ" -eq 0 ]; then
    pass "provider credentials can run a supported stack (not set:$MISSING_OPTIONAL — this check cannot read your agent, so it proves a stack is possible, not that the one you published is covered.${STT_CAVEAT}${MANIFEST_CAVEAT})"
  else
    pass "provider credentials cover every provider this image can build.${STT_CAVEAT}"
  fi
fi

# One or more comma-separated E.164 numbers: one trunk serves every DID you
# point at this gateway, and re-running provisioning with a different single
# number replaces the trunk rather than adding to it.
DID_LIST_OK=1
DID_COUNT=0
if [ -n "${SCH236_TEST_DID_E164:-}" ]; then
  OLD_IFS="$IFS"; IFS=,
  for _did in ${SCH236_TEST_DID_E164}; do
    # Trim only the ENDS. `tr -d '[:space:]'` also deleted internal spaces, so
    # `+1 2193126551` passed here and was then refused by the provisioner, which
    # trims surrounding whitespace and nothing else.
    _did="${_did#"${_did%%[![:space:]]*}"}"
    _did="${_did%"${_did##*[![:space:]]}"}"
    [ -n "$_did" ] || continue
    DID_COUNT=$((DID_COUNT + 1))
    # `+[0-9]*` would accept `+1219x126551`: it checks one digit and then lets
    # `*` match anything. Require the leading +, at least one digit after it,
    # and NO non-digit anywhere -- the same set the provisioner accepts, so a
    # config cannot provision successfully and then fail this gate.
    case "$_did" in
      +*) ;;
      *) DID_LIST_OK=0; BAD_DID="$_did"; continue ;;
    esac
    # Mirror the provisioner's `[1-9][0-9]{6,14}` exactly: no non-digits, no
    # leading zero, and 7-15 digits. Anything looser lets this gate report a DID
    # as valid that provisioning will then refuse -- which is the same
    # disagreement as a missing `+`, only quieter, because it only shows up at
    # the boundaries.
    _digits="${_did#+}"
    case "$_digits" in
      ""|*[!0-9]*|0*) DID_LIST_OK=0; BAD_DID="$_did"; continue ;;
    esac
    if [ "${#_digits}" -lt 7 ] || [ "${#_digits}" -gt 15 ]; then
      DID_LIST_OK=0; BAD_DID="$_did"; continue
    fi
  done
  IFS="$OLD_IFS"
  if [ "$DID_COUNT" -eq 0 ]; then
    fail "SCH236_TEST_DID_E164 lists at least one number" "got '${SCH236_TEST_DID_E164}'; expected e.g. +15551230000"
  elif [ "$DID_LIST_OK" -eq 1 ]; then
    pass "SCH236_TEST_DID_E164 lists ${DID_COUNT} E.164 number(s) with leading + (${SCH236_TEST_DID_E164})"
  else
    fail "every SCH236_TEST_DID_E164 entry is E.164 with leading +" \
      "'${BAD_DID}' is not; expected e.g. +15551230000 or +15551230000,+3725551234"
  fi
fi

# --- 4. control-plane URL shape (mirrors the runtime guard EXACTLY) -----------
# The bearer tokens are sent to this URL in steps 9-10, so validate it with the
# SAME rule the agent enforces (via python3, robust against userinfo/query/
# fragment tricks): https anywhere, or http ONLY to an EXACT loopback host.
CPU="${OPENPHONEX_CONTROL_PLANE_URL:-}"
CPU_OK=0
CPU_MODE=""
CPU_HOST=""
CPU_LOOPBACK=""
if [ -n "$CPU" ]; then
  if CP_OUT="$(cp_classify "$CPU")"; then
    CPU_OK=1
    # CP_OUT is "<host> <mode> <loopback>"
    CPU_HOST="${CP_OUT%% *}"
    CPU_LOOPBACK="${CP_OUT##* }"
    CPU_MODE="$(printf '%s' "$CP_OUT" | awk '{print $2}')"
    if [ "$CPU_MODE" = https ]; then
      pass "control-plane URL uses https (host '$CPU_HOST')"
    else
      pass "control-plane URL is exact-loopback http (host '$CPU_HOST', allowed)"
    fi
  else
    CPU_HOST="${CP_OUT%% *}"
    fail "control-plane URL is https or exact-loopback" "the agent REFUSES to send bearer tokens over '$CPU' (resolved host '$CPU_HOST' is not loopback, or the scheme is unsupported); use https:// or http://127.0.0.1:PORT"
  fi
fi

# A LOOPBACK control plane (http OR https) must never route credentialed curls
# through a proxy — a proxy would receive the Authorization header. Decide the
# bypass from the LOOPBACK flag INDEPENDENT of scheme; a remote https control
# plane keeps default proxy behaviour (an operator HTTPS_PROXY still works).
CP_NOPROXY=()
if [ "$CPU_LOOPBACK" = loopback ]; then CP_NOPROXY=(--noproxy '*'); fi

# Decide what the kernel would actually do with a specific packet, by walking
# the INPUT chain in order the way netfilter does.
#
# Grepping the ruleset for "a DROP mentioning 5060" is not good enough, and
# every one of these has been a real false PASS: a TCP-only rule, a rule scoped
# to one source host, "--dport 50600" matching a "5060" substring, and a
# blanket ACCEPT sitting above the DROP so the DROP never fires.
#
# Usage: fw_verdict <NEW|ESTABLISHED> "<output of iptables -S INPUT>"
# Both are argv, not stdin: the heredoc below IS this python process's stdin.
# Prints one word: ACCEPT, DROP, REJECT or INDETERMINATE.
# INDETERMINATE means a rule might apply but cannot be evaluated statically
# (a jump to a user chain, an unmodelled match like -m recent, or a rule
# scoped so it neither clearly applies nor clearly does not) -- reported as
# unverified rather than guessed either way.
fw_verdict() {
  python3 - "$1" "$2" <<'PYFW'
import sys

state = sys.argv[1]          # NEW or ESTABLISHED
rules = sys.argv[2].splitlines()
PORT = 5060
DECIDING = {"ACCEPT", "DROP", "REJECT"}
# Targets that never terminate traversal, so scanning continues past them.
NON_TERMINAL = {"LOG", "AUDIT", "MARK", "CONNMARK", "TRACE", "NFLOG", "TEE", "TOS", "TTL"}

policy = "ACCEPT"
verdict = None
maybe = set()   # targets of rules that might apply but cannot be proven to

def port_matches(spec):
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            lo, _, hi = part.partition(":")
            lo = int(lo) if lo else 0
            hi = int(hi) if hi else 65535
            if lo <= PORT <= hi:
                return True
        elif part.isdigit() and int(part) == PORT:
            return True
    return False

for raw in rules:
    tok = raw.split()
    if not tok:
        continue
    if tok[0] == "-P" and len(tok) >= 3 and tok[1] == "INPUT":
        policy = tok[2]
        continue
    if tok[0] != "-A" or len(tok) < 2 or tok[1] != "INPUT":
        continue
    if verdict is not None:
        continue

    # certain=False means "this rule might apply to our packet but we cannot
    # prove it does" -- e.g. it is scoped to a source we are not simulating.
    applies, certain, target = True, True, None
    i = 2
    while i < len(tok):
        a = tok[i]
        neg = False
        if a == "!":
            neg = True
            i += 1
            a = tok[i] if i < len(tok) else ""
        nxt = tok[i + 1] if i + 1 < len(tok) else ""

        if a == "-i":
            on_lo = nxt == "lo"
            # Our probe arrives from off-host, i.e. NOT on loopback.
            if (on_lo and not neg) or (neg and not on_lo):
                # "-i lo" cannot match, and "! -i eth0" we cannot resolve.
                applies = applies and (False if on_lo and not neg else applies)
                if on_lo and not neg:
                    applies = False
                else:
                    certain = False
            elif not on_lo and not neg:
                certain = False       # a specific non-lo interface: unknown
            i += 2
        elif a in ("-s", "-d"):
            if nxt not in ("0.0.0.0/0", "::/0"):
                certain = False       # scoped rule: does not decide the general case
            i += 2
        elif a == "-p":
            proto = nxt.lower()
            is_udp = proto in ("udp", "17")
            # "-p udp" matches; "-p tcp" does not; "! -p tcp" DOES match a UDP
            # packet; "! -p udp" does not.
            if neg:
                if is_udp:
                    applies = False
            elif not (is_udp or proto == "all"):
                applies = False
            i += 2
        elif a in ("--dport", "--dports", "--destination-port", "--destination-ports"):
            hit = port_matches(nxt)
            if neg:
                hit = not hit
            if not hit:
                applies = False
            i += 2
        elif a in ("--sport", "--sports", "--source-port"):
            certain = False
            i += 2
        elif a in ("--ctstate", "--state"):
            states = {s.strip().upper() for s in nxt.split(",")}
            hit = state in states
            if neg:
                hit = not hit
            if not hit:
                applies = False
            i += 2
        elif a in ("-f", "--fragment"):
            # -f matches second-and-later fragments only. Our probe is a single
            # unfragmented datagram, so this is an exact NON-match, not an
            # unknown -- which matters: common anti-spoofing boilerplate like
            # "-A INPUT -f -j DROP" must neither be read as closing the port
            # (it does not) nor condemn an otherwise correct ruleset.
            if not neg:
                applies = False
            i += 1
        elif a == "-m":
            if nxt == "tcp":
                # iptables silently attaches `-m tcp` to any rule with --dport
                # and a tcp-ish protocol spec, including "! -p tcp --dport 5060".
                # Measured against iptables-nft, that stored form DOES block a
                # UDP packet -- the opposite of what the tcp port match implies,
                # and not something worth asserting across nft/legacy backends.
                # A positive "-p tcp" has already excluded the packet above; any
                # other combination is declined rather than guessed.
                certain = False
            elif nxt in ("conntrack", "state", "udp", "multiport", "comment"):
                pass                  # fully modelled below
            else:
                certain = False       # addrtype / limit / recent / hashlimit / u32 / ...
            i += 2
        elif a in ("--src-type", "--dst-type"):
            # -m addrtype: depends on how the kernel classifies the address,
            # which we are not simulating.
            certain = False
            i += 2
        elif a in ("-j", "-g"):
            # -g is a goto rather than a jump, but for deciding this packet the
            # relevant question is the same: where does traversal continue.
            target = nxt
            i += 2
        elif a in ("--comment", "--to-ports", "--reject-with", "--log-prefix"):
            i += 2
        else:
            i += 1

    if not applies or target is None:
        continue
    if target in NON_TERMINAL:
        continue
    if target not in DECIDING:
        # a jump to a user-defined chain: not modelled
        verdict = "INDETERMINATE"
        continue
    if certain:
        verdict = target
    else:
        # Might fire, might not. Remember what it would have done: netfilter
        # would have hit it BEFORE anything below, so if the eventual answer
        # differs from this, the answer is not knowable statically.
        maybe.add(target)

if verdict is None:
    verdict = policy if policy in DECIDING else "INDETERMINATE"
if verdict != "INDETERMINATE" and (maybe - {verdict}):
    verdict = "INDETERMINATE"
print(verdict)
PYFW
}

# --- 5. services running ------------------------------------------------------
RUNNING="$($COMPOSE ps --services --status running 2>/dev/null)"
for svc in redis livekit sip agents reconciler; do
  if printf '%s\n' "$RUNNING" | grep -qx "$svc"; then
    pass "service '$svc' is running"
  else
    fail "service '$svc' is running" "start the stack: $COMPOSE up -d --build; then: $COMPOSE logs -n 50 $svc"
  fi
done

# --- 5b. registering SIP sidecar (only when the registration topology is used) -
# Deliberately keyed on the variable rather than on the profile: a half-filled
# SIP_REGISTRATION_* block with the sidecar not started is the failure mode
# where a self-hoster believes they are reachable and is not.
# Registration mode is decided by what is DEPLOYED, not only by what is in the
# env right now. An operator who provisioned in registration mode and later
# emptied the block without re-provisioning still has a trunk with no address
# allowlist — exactly the case where these checks matter most — so also treat a
# registrar container that exists (running or stopped) as registration mode.
REG_MODE=0
[ -n "${SIP_REGISTRATION_REGISTRAR:-}" ] && REG_MODE=1
if [ "$REG_MODE" = 0 ] && $COMPOSE ps -a --services 2>/dev/null | grep -qx registrar; then
  REG_MODE=1
  warn "the registrar sidecar is deployed but no SIP_REGISTRATION_* settings were found" \
    "checking as registration mode anyway. Expected them in ./.env.registrar (see .env.registrar.example). If you meant to go back to a public-IP deployment, re-run the provision step so the trunk regains its address allowlist — removing the file alone does NOT undo it."
fi
if [ "$REG_MODE" = 1 ]; then
  REG_MISSING=""
  # Same invariant as .env.livekit, one file over: docker's `env_file` hands
  # the sidecar .env.registrar, so a value living only in this shell is one the
  # container never receives. Topology DETECTION above deliberately stays
  # broad -- treating a deployment as registration mode runs MORE checks, so
  # erring that way is the safe direction; erring here would skip them.
  for v in SIP_REGISTRATION_USERNAME SIP_REGISTRATION_PASSWORD SIP_REGISTRATION_ALLOWED_ADDRESSES; do
    configured_registrar "$v" || REG_MISSING="$REG_MISSING $v"
  done
  if [ -z "$REG_MISSING" ]; then
    pass "registration topology: all required SIP_REGISTRATION_* variables are set"
  else
    fail "registration topology: all required SIP_REGISTRATION_* variables are set" \
      "missing:$REG_MISSING (see .env.registrar.example)"
  fi

  if printf '%s\n' "$RUNNING" | grep -qx registrar; then
    pass "service 'registrar' is running"
  else
    fail "service 'registrar' is running" \
      "SIP_REGISTRATION_REGISTRAR is set but the sidecar is not started: $COMPOSE --profile registrar up -d --build registrar"
  fi

  if $COMPOSE logs registrar 2>/dev/null | grep -q REGISTRAR_REGISTRATION_ARMED; then
    pass "registrar armed its outbound registration"
  else
    fail "registrar armed its outbound registration" \
      "$COMPOSE logs -n 50 registrar — look for REGISTRAR_CONFIG_INVALID or REGISTRAR_BOOTSTRAP_FAILED"
  fi

  # Armed is not the same as registered, and for this deployment the
  # difference IS the product: a self-hoster behind NAT whose failure mode is
  # "everything looks up, calls don't arrive" reads this script's exit code as
  # the answer to "am I reachable".
  #
  # Call the SAME script the container healthcheck uses. Do not reimplement the
  # predicate here: the obvious "timer_expires is non-zero" test reports a
  # healthy registration throughout a total registration failure, because
  # Kamailio reuses that field as a retry deadline. That divergence was shipped
  # once already. See registrar/healthcheck.sh.
  if $COMPOSE exec -T registrar /usr/local/bin/openphonex-registrar-health >/dev/null 2>&1; then
    pass "registrar is holding a live registration with the carrier"
  else
    fail "registrar is holding a live registration with the carrier" \
      "give it ~60s after start. If it stays down the carrier is refusing the REGISTER — check the credentials and SIP_REGISTRATION_REGISTRAR, then $COMPOSE logs -n 100 registrar. Note a rejected registration retries forever and looks alive from the outside; this check does not."
  fi

  # In registration mode the LiveKit trunk carries no address allowlist, so
  # the host firewall is the boundary in front of the SIP listener. Both
  # families are checked: NAT covers IPv4 only, and a dual-stack host usually
  # has a global IPv6 address with nothing in front of it.
  #
  # Two questions, asked of the ruleset as the kernel would answer them:
  #   NEW packet to 5060 from off-host  -> must be dropped
  #   reply to our own outbound (ESTABLISHED) -> must NOT be dropped
  for fam in 4 6; do
    if [ "$fam" = 4 ]; then fwcmd=iptables; else fwcmd=ip6tables; fi

    if ! command -v "$fwcmd" >/dev/null 2>&1; then
      if [ "$fam" = 6 ]; then
        # Absent ip6tables is only harmless if IPv6 is not actually live.
        # Scope 00 in /proc/net/if_inet6 is a global address.
        if [ -r /proc/net/if_inet6 ] && awk '$4 == "00" { found = 1 } END { exit !found }' /proc/net/if_inet6; then
          unverified "IPv6: 5060/udp exposure could not be checked" \
            "this host has a GLOBAL IPv6 address but ip6tables is not installed, so the v6 path is unknown. IPv6 has no NAT in front of it — install ip6tables (or nftables) and apply the same rule, or verify by hand. See SELFHOST.md."
        else
          pass "IPv6: no global address on this host, so 5060/udp is not reachable over v6"
        fi
      else
        unverified "IPv4: 5060/udp exposure could not be checked" \
          "iptables is not installed; verify the equivalent rule by hand — see SELFHOST.md"
      fi
      continue
    fi

    FW_RULES="$($fwcmd -S INPUT 2>/dev/null)"
    if [ -z "$FW_RULES" ]; then
      unverified "IPv$fam: 5060/udp exposure could not be checked" \
        "could not read the $fwcmd rules — re-run with sudo. In registration mode this is the ONLY thing in front of the SIP listener, so the run cannot be reported as PASS without it."
      continue
    fi

    FW_NEW="$(fw_verdict NEW "$FW_RULES")"
    FW_EST="$(fw_verdict ESTABLISHED "$FW_RULES")"

    case "$FW_NEW" in
      DROP|REJECT)
        case "$FW_EST" in
          DROP|REJECT)
            fail "IPv$fam: 5060/udp DROP also blocks replies to outbound calls" \
              "unsolicited packets are dropped correctly, but so are replies to traffic this host sent from port 5060 — livekit-sip sends outbound signalling from its listening socket, so outbound calling breaks silently. Scope the DROP with -m conntrack --ctstate NEW. See SELFHOST.md." ;;
          INDETERMINATE)
            unverified "IPv$fam: cannot tell whether replies to outbound calls survive" \
              "the ruleset contains a rule this check cannot evaluate (a jump to a user chain, or a match like -m recent). Confirm by hand that ESTABLISHED traffic to 5060/udp is accepted." ;;
          *)
            pass "IPv$fam: 5060/udp drops unsolicited traffic and admits replies" ;;
        esac ;;
      INDETERMINATE)
        unverified "IPv$fam: cannot prove 5060/udp is closed to unsolicited traffic" \
          "the ruleset contains a rule this check cannot evaluate statically, or one scoped so narrowly it does not decide the general case. Confirm by hand — see SELFHOST.md." ;;
      *)
        fail "IPv$fam: 5060/udp is open to unsolicited traffic" \
          "an unsolicited UDP packet to port 5060 from off-host would be ACCEPTed. In registration mode the LiveKit trunk has no address allowlist, so anyone who reaches this port gets an answered call into your agent. Add: $fwcmd -A INPUT ! -i lo -p udp --dport 5060 -m conntrack --ctstate NEW -j DROP. See SELFHOST.md." ;;
    esac
  done

  # A keepalive at or above the shortest common NAT UDP timeout is the silent
  # killer: the carrier keeps showing you registered while calls stop arriving.
  KA="${SIP_REGISTRATION_KEEPALIVE_SECONDS:-25}"
  if [ "$KA" -ge 5 ] 2>/dev/null && [ "$KA" -le 28 ]; then
    pass "NAT keepalive is ${KA}s (under the 30s floor NAT bindings commonly use)"
  else
    fail "NAT keepalive is under 30s" \
      "SIP_REGISTRATION_KEEPALIVE_SECONDS='$KA'; use 5-28 or calls stop arriving between registration refreshes"
  fi
fi

# --- 6. redis -----------------------------------------------------------------
if [ -n "$RUNNING" ] && $COMPOSE exec -T redis redis-cli -h 127.0.0.1 -p 6380 ping 2>/dev/null | grep -q PONG; then
  pass "redis answers PONG on 127.0.0.1:6380"
else
  fail "redis answers PONG on 127.0.0.1:6380" "$COMPOSE logs -n 50 redis"
fi

# --- 7. livekit-server --------------------------------------------------------
# Loopback probe -> never via a proxy.
LK_CODE="$(curl --noproxy '*' -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:7880/ 2>/dev/null || true)"
if [ "$LK_CODE" != "000" ] && [ -n "$LK_CODE" ]; then
  pass "livekit-server answers on http://127.0.0.1:7880 (HTTP $LK_CODE)"
else
  fail "livekit-server answers on http://127.0.0.1:7880" "$COMPOSE logs -n 50 livekit (config error in livekit.yaml? redis down?)"
fi

# --- 8. agent registered + prewarmed ------------------------------------------
AGENT_LOGS="$($COMPOSE logs --no-color agents 2>/dev/null | tail -n 400)"
if printf '%s' "$AGENT_LOGS" | grep -q "registered worker"; then
  pass "agent registered with LiveKit ('registered worker' in logs)"
else
  fail "agent registered with LiveKit" "no 'registered worker' in: $COMPOSE logs agents — check LIVEKIT_API_KEY/SECRET match livekit.yaml, and $COMPOSE logs -n 100 agents for tracebacks"
fi
if printf '%s' "$AGENT_LOGS" | grep -q '"event": "process_warm"'; then
  pass "agent processes prewarmed (process_warm in logs)"
else
  warn "agent processes prewarmed" "no process_warm event yet; first warm-up can take ~1 min after start (model download-files ran at build time)"
fi

# Bearer tokens are sent ONLY when the control-plane URL passed the exact
# https/loopback validation above — never to an unvalidated (possibly remote
# plaintext) host, which the runtime would refuse anyway.
if [ "$CPU_OK" != 1 ]; then
  fail "control-plane URL validated before sending tokens" "steps 9-10 skipped: fix the control-plane URL above; the smoke test will not send bearer tokens to an unvalidated URL"
fi

# --- 9. control plane accepts the GATEWAY token -------------------------------
if [ "$CPU_OK" = 1 ] && [ -n "${OPENPHONEX_GATEWAY_ID:-}" ] && [ -n "${OPENPHONEX_GATEWAY_API_TOKEN:-}" ]; then
  GW_CODE="$(bearer_curl "${OPENPHONEX_GATEWAY_API_TOKEN}" ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} \
    -s -o /dev/null -m 10 -w '%{http_code}' \
    "${CPU%/}/v1/gateway-deployments/${OPENPHONEX_GATEWAY_ID}/number-rules" 2>/dev/null || true)"
  case "$GW_CODE" in
    200|304) pass "control plane accepts the gateway token (number-rules HTTP $GW_CODE)" ;;
    401|403) fail "control plane accepts the gateway token" "HTTP $GW_CODE — wrong, revoked, or expired tgw_ token; it may belong to another organization or the id/token pair belongs to a different deployment" ;;
    000|"") fail "control plane is reachable" "no HTTP response from ${CPU} — URL wrong, network/firewall, or control plane down" ;;
    *) fail "control plane accepts the gateway token" "unexpected HTTP $GW_CODE from ${CPU%/}/v1/gateway-deployments/${OPENPHONEX_GATEWAY_ID}/number-rules" ;;
  esac
elif [ "$CPU_OK" = 1 ]; then
  fail "control plane accepts the gateway token" "OPENPHONEX_GATEWAY_ID / OPENPHONEX_GATEWAY_API_TOKEN not both set"
fi

# --- 10. the gateway token is accepted on the SECOND surface a call needs -----
# A call needs this one token on two different control-plane surfaces: the
# number-rules sync proved in step 9, and the MCP tool-schema catalogue the
# worker hands its model. A token can pass one and fail the other (narrowed
# scopes, a deployment/organization mismatch), so proving both is what makes
# this gate mean "the credential this stack runs on works" rather than "some
# credential works".
#
# NOTE: neither check proves the INBOUND-ROUTE authority itself — the control
# plane exposes no non-mutating probe for it, so a 200 here means
# "authenticates", not "call-ready". The first real call is that check.
if [ "$CPU_OK" = 1 ] && [ -n "${OPENPHONEX_GATEWAY_API_TOKEN:-}" ]; then
  MCP_CODE="$(bearer_curl "${OPENPHONEX_GATEWAY_API_TOKEN}" ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} \
    -s -o /dev/null -m 10 -w '%{http_code}' -X POST \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"selfhost-smoketest","method":"tools/list"}' \
    "${CPU%/}/mcp" 2>/dev/null || true)"
  case "$MCP_CODE" in
    200) pass "control plane accepts the gateway token for tool schemas (MCP tools/list HTTP 200; call/route authority is exercised by the first real call)" ;;
    401|403) fail "control plane accepts the gateway token for tool schemas" "HTTP $MCP_CODE from ${CPU%/}/mcp — the gateway token was not accepted here even though step 9 may have passed. A control plane older than the one-credential release still refuses tgw_ on /mcp: either upgrade it, or set OPENPHONEX_WORKER_API_TOKEN to an organization-scoped tai_ key as before" ;;
    000|"") fail "MCP endpoint is reachable" "no HTTP response from ${CPU%/}/mcp" ;;
    *) fail "control plane accepts the gateway token for tool schemas" "unexpected HTTP $MCP_CODE from ${CPU%/}/mcp" ;;
  esac
elif [ "$CPU_OK" = 1 ]; then
  fail "control plane accepts the gateway token for tool schemas" "OPENPHONEX_GATEWAY_API_TOKEN not set"
fi

# --- 10b. an organization worker key, only if one is still set ---------------
# The baseline install no longer sets this. It is read ONLY by the legacy
# `queue` outbound mode's job-claiming worker, and an install may still carry
# it, so when it IS present it has to work: a stale or revoked tai_ key sitting
# in this file is how queue-mode outbound dialling fails after everything else
# reports green. A deployment in `local` outbound mode needs no such key --
# deleting the line is what narrows the box.
if [ "$CPU_OK" = 1 ] && [ -n "${OPENPHONEX_WORKER_API_TOKEN:-}" ]; then
  WORKER_MCP_CODE="$(bearer_curl "${OPENPHONEX_WORKER_API_TOKEN}" ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} \
    -s -o /dev/null -m 10 -w '%{http_code}' -X POST \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"selfhost-smoketest-worker","method":"tools/list"}' \
    "${CPU%/}/mcp" 2>/dev/null || true)"
  case "$WORKER_MCP_CODE" in
    200) pass "control plane authenticates the organization worker key still set here (MCP tools/list HTTP 200)" ;;
    401|403) fail "control plane authenticates the organization worker key still set here" "HTTP $WORKER_MCP_CODE — OPENPHONEX_WORKER_API_TOKEN is wrong, revoked or expired. Only the legacy queue outbound mode reads it, so either replace it, or switch this deployment to outbound_execution_mode=local and delete the line; leaving a dead key here breaks --profile outbound" ;;
    000|"") fail "MCP endpoint is reachable" "no HTTP response from ${CPU%/}/mcp" ;;
    *) fail "control plane authenticates the organization worker key still set here" "unexpected HTTP $WORKER_MCP_CODE from ${CPU%/}/mcp" ;;
  esac
fi

# --- 11. artifact directories (REAL write probe as the agent uid) -------------
# The agent runs as uid 10001 and writes per-call state under /sessions and
# recordings under /call-evidence. A root-owned 0755 host dir EACCESes the first
# call. Probe an ACTUAL write AS uid 10001 by exec'ing into the running agents
# container (its /sessions and /call-evidence are the bind-mounted host dirs).
# A failed write is a FAIL — it WILL break the first call.
AGENTS_UP=0
printf '%s\n' "$RUNNING" | grep -qx agents && AGENTS_UP=1
if [ "$AGENTS_UP" = 1 ]; then
  for cdir in /sessions /call-evidence; do
    hostdir="${cdir#/}"
    if $COMPOSE exec -T -u 10001 agents sh -c "p=$cdir/.smoketest-write-probe.\$\$; touch \"\$p\" && rm -f \"\$p\"" 2>/dev/null; then
      pass "agent uid 10001 can write $cdir (host ./$hostdir) — probed in-container"
    else
      fail "agent uid 10001 can write $cdir (host ./$hostdir)" "the agent will EACCES the first call — run: sudo chown -R 10001:10001 $hostdir (and ensure the ./$hostdir bind mount exists)"
    fi
  done
else
  # Agents container not running (already FAILed in step 5): fall back to a
  # host-side ownership check so a wrong owner is still surfaced as a FAIL.
  for d in sessions call-evidence; do
    if [ ! -d "$d" ]; then
      fail "directory $d/ exists and is writable by the agent uid" "mkdir -p $d && sudo chown -R 10001:10001 $d"
      continue
    fi
    OWNER_UID="$(stat -c %u "$d" 2>/dev/null || stat -f %u "$d" 2>/dev/null || echo '?')"
    if [ "$OWNER_UID" = "10001" ]; then
      pass "directory $d/ is owned by uid 10001 (agents container down; ownership check only)"
    else
      fail "directory $d/ is owned by uid 10001" "owner is uid $OWNER_UID; the agent writes as uid 10001 and will EACCES — run: sudo chown -R 10001:10001 $d"
    fi
  done
fi

# --- summary ------------------------------------------------------------------
echo
echo "================================================================"
echo "Smoke test: ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${WARN_COUNT} warning(s), ${UNVERIFIED_COUNT} unverified."
if [ "$FAIL_COUNT" -eq 0 ] && [ "$UNVERIFIED_COUNT" -gt 0 ]; then
  echo "RESULT: UNVERIFIED — every check that could run passed, but ${UNVERIFIED_COUNT} could not be performed."
  echo "The [????] items above are not warnings: each is a check whose failure would matter."
  echo "Resolve them (usually: re-run with sudo) and run again before treating this stack as ready."
  exit 2
fi
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "RESULT: PASS — the stack is up; both control-plane tokens authenticate."
  echo "Next: call your DID (${SCH236_TEST_DID_E164:-<your DID>}) from any phone."
  echo "Watch it live:  $COMPOSE logs -f agents"
  echo "Expect:         media_admitted -> turn_latency per turn -> call_complete"
  echo "(The first call also confirms the worker token's call/route scope.)"
  exit 0
fi
echo "RESULT: FAIL — fix the [FAIL] items above, then re-run: bash selfhost-smoketest.sh"
exit 1

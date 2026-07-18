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
#   6. redis answers PONG
#   7. livekit-server answers on http://127.0.0.1:7880
#   8. the agent registered with LiveKit and prewarmed (log evidence)
#   9. the control plane accepts the gateway token (number-rules read)
#  10. the control plane accepts the worker token (MCP tools/list)
#  11. sessions/ + call-evidence/ are writable by the in-container agent uid
#
# Exit code 0 = every hard check passed -> place a test call to your DID.
# The script is read-only: it places no calls and changes nothing.

set -u

cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.livekit.yml"
COMPOSE="docker compose -f ${COMPOSE_FILE}"
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); printf '[PASS] %s\n' "$1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); printf '[FAIL] %s\n       -> %s\n' "$1" "$2"; }
warn() { WARN_COUNT=$((WARN_COUNT + 1)); printf '[WARN] %s\n       -> %s\n' "$1" "$2"; }

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

# --- 2. filled-in config files ------------------------------------------------
for f in livekit.yaml sip.yaml .env.livekit; do
  if [ ! -f "$f" ]; then
    fail "config file $f exists" "cp ${f}.example ${f} (see SELFHOST.md '1. Configure') and fill it in."
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
while IFS= read -r _line || [ -n "$_line" ]; do
  case "$_line" in ''|'#'*) continue ;; esac
  _key="${_line%%=*}"
  case "$_key" in
    "$_line") continue ;;             # no '=' on the line
    *[!A-Za-z0-9_]*) continue ;;      # not a shell identifier -> skip
  esac
  _val="${_line#*=}"
  case "$_val" in                     # strip one layer of surrounding quotes
    \"*\") _val="${_val#\"}"; _val="${_val%\"}" ;;
    \'*\') _val="${_val#\'}"; _val="${_val%\'}" ;;
  esac
  export "$_key=$_val"
done < .env.livekit

REQUIRED_VARS="LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET \
OPENPHONEX_CONTROL_PLANE_URL OPENPHONEX_GATEWAY_ID OPENPHONEX_GATEWAY_API_TOKEN \
OPENPHONEX_WORKER_API_TOKEN OPENPHONEX_ORGANIZATION_ID SCH236_TEST_DID_E164 \
DEEPGRAM_API_KEY GEMINI_API_KEY ELEVENLABS_API_KEY"
MISSING=""
for v in $REQUIRED_VARS; do
  eval "val=\${$v:-}"
  [ -n "$val" ] || MISSING="$MISSING $v"
done
if [ -z "$MISSING" ]; then
  pass "all required .env.livekit variables are set"
else
  fail "all required .env.livekit variables are set" "missing:$MISSING (each is documented in .env.livekit.example)"
fi

case "${SCH236_TEST_DID_E164:-}" in
  +[0-9]*) pass "SCH236_TEST_DID_E164 is E.164 with leading + (${SCH236_TEST_DID_E164})" ;;
  "") : ;;  # already reported missing above
  *) fail "SCH236_TEST_DID_E164 is E.164 with leading +" "got '${SCH236_TEST_DID_E164}'; expected e.g. +15551230000" ;;
esac

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

# --- 5. services running ------------------------------------------------------
RUNNING="$($COMPOSE ps --services --status running 2>/dev/null)"
for svc in redis livekit sip agents reconciler; do
  if printf '%s\n' "$RUNNING" | grep -qx "$svc"; then
    pass "service '$svc' is running"
  else
    fail "service '$svc' is running" "start the stack: $COMPOSE up -d --build; then: $COMPOSE logs -n 50 $svc"
  fi
done

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
    401|403) fail "control plane accepts the gateway token" "HTTP $GW_CODE — wrong or revoked tgw_ token, or the id/token pair belongs to a different deployment" ;;
    000|"") fail "control plane is reachable" "no HTTP response from ${CPU} — URL wrong, network/firewall, or control plane down" ;;
    *) fail "control plane accepts the gateway token" "unexpected HTTP $GW_CODE from ${CPU%/}/v1/gateway-deployments/${OPENPHONEX_GATEWAY_ID}/number-rules" ;;
  esac
elif [ "$CPU_OK" = 1 ]; then
  fail "control plane accepts the gateway token" "OPENPHONEX_GATEWAY_ID / OPENPHONEX_GATEWAY_API_TOKEN not both set"
fi

# --- 10. control plane authenticates the WORKER token (MCP) -------------------
# NOTE: this proves the worker token AUTHENTICATES against the control plane
# (MCP tools/list). It does NOT prove the token carries the calls/route scope
# the first inbound call needs (POST /v1/calls/inbound/route) — the control
# plane exposes no non-mutating probe for that scope, so a 200 here means
# "authenticates", not "call-ready". The first real call is the scope check.
if [ "$CPU_OK" = 1 ] && [ -n "${OPENPHONEX_WORKER_API_TOKEN:-}" ]; then
  MCP_CODE="$(bearer_curl "${OPENPHONEX_WORKER_API_TOKEN}" ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} \
    -s -o /dev/null -m 10 -w '%{http_code}' -X POST \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":"selfhost-smoketest","method":"tools/list"}' \
    "${CPU%/}/mcp" 2>/dev/null || true)"
  case "$MCP_CODE" in
    200) pass "control plane authenticates the worker token (MCP tools/list HTTP 200; call/route scope is exercised by the first real call)" ;;
    401|403) fail "control plane authenticates the worker token" "HTTP $MCP_CODE — the worker token is wrong/revoked (this is the organization-scoped token, not the tgw_ gateway token)" ;;
    000|"") fail "MCP endpoint is reachable" "no HTTP response from ${CPU%/}/mcp" ;;
    *) fail "control plane authenticates the worker token" "unexpected HTTP $MCP_CODE from ${CPU%/}/mcp" ;;
  esac
elif [ "$CPU_OK" = 1 ]; then
  fail "control plane authenticates the worker token" "OPENPHONEX_WORKER_API_TOKEN not set"
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
echo "Smoke test: ${PASS_COUNT} passed, ${FAIL_COUNT} failed, ${WARN_COUNT} warning(s)."
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

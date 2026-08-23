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
#  10. the control plane accepts the worker token (MCP tools/list)
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
read_env_file() {
  [ -f "$1" ] || return 0
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in ''|'#'*) continue ;; esac
    _key="${_line%%=*}"
    case "$_key" in
      "$_line") continue ;;             # no '=' on the line
      *[!A-Za-z0-9_]*) continue ;;      # not a shell identifier -> skip
    esac
    _val="${_line#*=}"
    _val="${_val%$(printf '\r')}"      # CRLF files: Compose strips \r, so match it
    case "$_val" in                     # strip one layer of surrounding quotes
      \"*\") _val="${_val#\"}"; _val="${_val%\"}" ;;
      \'*\') _val="${_val#\'}"; _val="${_val%\'}" ;;
    esac
    export "$_key=$_val"
  done < "$1"
}
read_env_file .env.livekit
# The sidecar's credentials live in their own file (see .env.registrar.example
# for why they are not merged). Read it with the SAME non-evaluating parser.
read_env_file .env.registrar

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

# Variables the stack cannot start without, whatever the agent is configured to
# say. Provider keys are deliberately NOT in this list -- see below.
REQUIRED_VARS="LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET \
OPENPHONEX_CONTROL_PLANE_URL OPENPHONEX_GATEWAY_ID OPENPHONEX_GATEWAY_API_TOKEN \
OPENPHONEX_WORKER_API_TOKEN OPENPHONEX_ORGANIZATION_ID SCH236_TEST_DID_E164"
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

# WHICH provider keys you need is decided by the voice stack your agent
# publishes, and this script cannot see that stack. But it does not have to
# guess either: the worker accepts a fixed, small set of providers, so what is
# *structurally* required follows from that set.
#
#   LLM  -- only gemini/google, so GEMINI_API_KEY is always needed.
#   STT  -- deepgram (DEEPGRAM_API_KEY) or google chirp_2
#           (GOOGLE_APPLICATION_CREDENTIALS[_JSON]); at least one.
#   TTS  -- elevenlabs (ELEVENLABS_API_KEY) or gemini_tts, which reuses the
#           Gemini key. So ElevenLabs is genuinely optional.
#
# This check has been wrong in three different directions: demanding all three
# failed a correct Deepgram+Gemini install, warning let an ElevenLabs-only
# install report PASS with no STT and no LLM, and UNVERIFIED meant a documented
# valid stack could never reach the PASS the runbook requires. Deriving it from
# the provider set is the first version that is right for both.
GEMINI_SET=0; [ -n "${GEMINI_API_KEY:-}" ] && GEMINI_SET=1
STT_SET=0
STT_PATH_UNCHECKED=0
[ -n "${DEEPGRAM_API_KEY:-}" ] && STT_SET=1
# A path is a CONTAINER path -- the key file is mounted read-only into the
# agents service -- so this script, running on the host, cannot open it. Say
# that rather than treating "non-empty" as "valid".
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
  STT_SET=1
  STT_PATH_UNCHECKED=1
fi
# The inline variant IS checkable here, and a malformed value is a deterministic
# first-call failure worth catching now: the agent needs real service-account
# fields, so an empty object or a truncated paste must not read as configured.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ]; then
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

if [ "$GEMINI_SET" -eq 0 ]; then
  fail "GEMINI_API_KEY is set" \
    "every supported stack resolves its model to Gemini, and Gemini TTS reuses the same key"
elif [ "$STT_SET" -eq 0 ]; then
  fail "an STT credential is set" \
    "set DEEPGRAM_API_KEY, or GOOGLE_APPLICATION_CREDENTIALS/_JSON for the Google chirp_2 path; no supported stack transcribes without one"
else
  # Deliberately the WEAKER claim, and it says so. This proves a runnable stack
  # is *possible* with the keys present; it cannot prove your agent's published
  # stack is covered, because that would mean reading the agent, and the
  # self-host worker key is scoped to calls:read + calls:write on purpose.
  #
  # The two gaps this leaves, both surfacing on the first call rather than here:
  # an agent publishing ElevenLabs TTS with no ELEVENLABS_API_KEY, and an agent
  # publishing Deepgram STT when only the Google credentials are set.
  MISSING_OPTIONAL=""
  [ -n "${ELEVENLABS_API_KEY:-}" ] || MISSING_OPTIONAL="$MISSING_OPTIONAL ELEVENLABS_API_KEY"
  [ -n "${DEEPGRAM_API_KEY:-}" ] || MISSING_OPTIONAL="$MISSING_OPTIONAL DEEPGRAM_API_KEY"
  STT_CAVEAT=""
  [ "$STT_PATH_UNCHECKED" -eq 1 ] && STT_CAVEAT=" GOOGLE_APPLICATION_CREDENTIALS names a path inside the agents container, so its contents are not checked here."
  if [ -n "$MISSING_OPTIONAL" ]; then
    pass "provider credentials can run a supported stack (not set:$MISSING_OPTIONAL — this check cannot read your agent, so it proves a stack is possible, not that the one you published is covered.${STT_CAVEAT})"
  else
    pass "provider credentials cover every supported provider.${STT_CAVEAT}"
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
  for v in SIP_REGISTRATION_USERNAME SIP_REGISTRATION_PASSWORD SIP_REGISTRATION_ALLOWED_ADDRESSES; do
    eval "val=\${$v:-}"
    [ -n "$val" ] || REG_MISSING="$REG_MISSING $v"
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
    401|403) fail "control plane authenticates the worker token" "HTTP $MCP_CODE — the worker credential was not accepted (wrong, revoked, expired, or a tgw_ token pasted here); MCP tools/list proves authentication only, while the first inbound route checks OPENPHONEX_ORGANIZATION_ID against the tai_ key and DID" ;;
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

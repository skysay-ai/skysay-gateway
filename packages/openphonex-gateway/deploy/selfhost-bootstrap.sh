#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# OpenPhonex LiveKit self-host — one-time bootstrap.
#
# You set TWO things. Everything else is derived and written for you:
#
#   OPENPHONEX_CONTROL_PLANE_URL   the control plane (https, or http on loopback)
#   OPENPHONEX_BOOTSTRAP_API_KEY   an organization API key carrying `gateways:write`
#
# Run it BEFORE `docker compose up`, from this directory:
#
#   OPENPHONEX_CONTROL_PLANE_URL=https://api.openphonex.com \
#   OPENPHONEX_BOOTSTRAP_API_KEY=tai_... \
#     bash selfhost-bootstrap.sh
#
# Omit OPENPHONEX_BOOTSTRAP_API_KEY and it is read from the terminal without
# echo. Either way it is NEVER written to disk: there is nothing on this box to
# delete afterwards, which is why "revoke it when you are done" is a one-step
# instruction rather than a cleanup checklist.
#
# It derives and writes:
#   OPENPHONEX_ORGANIZATION_ID   from the key's own workspace
#   OPENPHONEX_GATEWAY_ID        by registering this deployment
#   OPENPHONEX_GATEWAY_API_TOKEN the deployment-bound tgw_ token (no expiry)
#   SCH236_TEST_DID_E164         the one agent-routed active number, if there is exactly one
#   LIVEKIT_API_KEY/SECRET       generated, and instantiated into livekit.yaml + sip.yaml
#
# It does NOT derive DIDWW_INBOUND_ADDRESSES. The control plane holds no carrier
# signalling ranges and cannot know them for a carrier you brought yourself; a
# guessed allowlist is the everything-green-but-no-calls failure this runbook
# fights hardest. Read them off your carrier's portal and set them by hand.
#
# Safe to re-run. Per value: a non-empty .env.livekit entry is left alone, a
# missing one is filled from gateway-state.json, and only a value that exists in
# neither is derived. It never edits a livekit.yaml/sip.yaml you already have.
#
# Why this is a host script and not a container: livekit-server reads
# ./livekit.yaml when it starts, .env.livekit is bind-mounted READ-ONLY into
# every service, and the `provision` profile needs the stack already up. All
# three mean this work has to finish before the first `docker compose up`.

set -u

cd "$(dirname "$0")" || exit 1

ENV_FILE=".env.livekit"
ENV_EXAMPLE=".env.livekit.example"
STATE_FILE="gateway-state.json"
LIVEKIT_YAML="livekit.yaml"
LIVEKIT_YAML_EXAMPLE="livekit.yaml.example"
SIP_YAML="sip.yaml"
SIP_YAML_EXAMPLE="sip.yaml.example"

# The variable the bootstrap key arrives in. Named once: the smoke test refuses
# to find this name assigned in .env.livekit, and both scripts must mean the
# same variable for that check to be worth anything.
BOOTSTRAP_KEY_VAR="OPENPHONEX_BOOTSTRAP_API_KEY"

# Set only for the window between "the mint request goes out" and "the token is
# on disk". An interrupt inside it leaves a live, non-expiring credential this
# box does not hold, and saying nothing there is the one outcome nothing can
# recover from.
MINT_IN_FLIGHT=0

say()  { printf '%s\n' "$1"; }
step() { printf '[..] %s\n' "$1"; }
ok()   { printf '[ok] %s\n' "$1"; }
die()  { printf '[!!] %s\n' "$1" >&2; [ "$#" -gt 1 ] && printf '     -> %s\n' "$2" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 is required" "it parses every control-plane response here; install python3 and re-run"
command -v openssl >/dev/null 2>&1 || OPENSSL_MISSING=1
command -v curl >/dev/null 2>&1 || die "curl is required" "install curl and re-run"

# --- inputs -------------------------------------------------------------------

CPU="${OPENPHONEX_CONTROL_PLANE_URL:-}"

# The URL may already be in .env.livekit from a previous run or by hand. Read it
# non-evaluating (never `source` a credential file) before asking for one.
env_value() {
  # $1 = variable name. Prints the assigned value, or nothing. Literal read:
  # the RHS is never re-scanned, so a `$` or backtick in a secret is inert.
  [ -f "$ENV_FILE" ] || return 0
  ENV_LOOKUP_NAME="$1" ENV_LOOKUP_FILE="$ENV_FILE" python3 - <<'PY'
import os, sys
name = os.environ["ENV_LOOKUP_NAME"]
try:
    with open(os.environ["ENV_LOOKUP_FILE"], encoding="utf-8") as handle:
        text = handle.read()
except OSError:
    sys.exit(0)


def _assignment(name: str):
    """Every line the CONTAINER would treat as an assignment to `name`.

    Leading whitespace is allowed because `source` allows it, and `export` may
    be followed by any run of blanks. This definition is duplicated verbatim in
    the writer below; a test pins the two copies byte-identical, because the
    two ways this file has already gone wrong were a reader and a writer that
    disagreed about which line is the assignment.
    """
    import re as _re

    return _re.compile(r"\A[ \t]*(?:export[ \t]+)?" + _re.escape(name) + r"=")


def scalar(raw: str) -> str:
    """What the CONTAINER gets when it sources this line.

    The file is `source`d, so shell rules decide what "set" means: a wholly
    quoted value is its quoted contents, and an UNQUOTED trailing ` #` comment
    is not part of the value. Deliberately a subset -- no escapes, no adjacent
    concatenation -- because .env.livekit.example tells you to single-quote
    anything exotic. This is the ONLY place the file is interpreted; the writer
    is told what it found rather than parsing it a second time, because two
    parsers that disagree are how a `FOO=""` gets a generated secret written
    into the YAMLs and skipped here.
    """
    if raw[:1] in (" ", "\t"):
        # `FOO= value` assigns NOTHING and then runs `value` as a command;
        # `FOO= # note` assigns nothing and comments. Either way the container
        # ends up with an empty FOO, so that is what this reports.
        return ""
    # strip(" \t"), not strip(): a bare strip() also removes a pasted
    # non-breaking space, which the shell keeps inside the value. The gate
    # refuses those lines, so this only has to match the shell for what it lets
    # through.
    value = raw.strip(" \t")
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        return value[1:end] if end > 0 else value[1:]
    # `FOO=#x` is the VALUE `#x`: the shell needs whitespace before a comment.
    for marker in (" #", "\t#"):
        cut = value.find(marker)
        if cut >= 0:
            value = value[:cut].rstrip()
    return value


pattern = _assignment(name)
found = None
for line in text.split("\n"):
    match = pattern.match(line)
    if match:
        # LAST assignment wins, and only one line is ever printed. Sourcing
        # this file keeps the last one, so anything else would disagree with
        # the container; printing every match would splice two values into one
        # shell variable, and a credential with an embedded newline in it is a
        # different and much worse problem.
        found = scalar(line[match.end():])
if found is not None:
    print(found)
PY
}

# Refuse a line this script cannot read the way the container will.
#
# `scalar` above implements the simple shapes .env.livekit.example asks for. It
# does NOT implement backslash escapes, adjacent quoted segments, or expansion
# -- and for those, guessing is worse than stopping: a value read one way here
# and another way by the entrypoint puts two different credentials into two
# files and reports success. So the ambiguity is a hard error with the fix in
# it, not a silent divergence.
assert_env_file_is_interpretable() {
  [ -f "$ENV_FILE" ] || return 0
  ENV_LOOKUP_FILE="$ENV_FILE" python3 - <<'PY' || die "$ENV_FILE has a line this script cannot read the way the container will" "each report above says which line and why. Single-quote the whole value (FOO=...) so it is taken literally, put one assignment at the start of its own line, and use LF line endings"
import os, re, sys


def _assignment(name: str):
    """Every line the CONTAINER would treat as an assignment to `name`.

    Leading whitespace is allowed because `source` allows it, and `export` may
    be followed by any run of blanks. This definition is duplicated verbatim in
    the writer below; a test pins the two copies byte-identical, because the
    two ways this file has already gone wrong were a reader and a writer that
    disagreed about which line is the assignment.
    """
    import re as _re

    return _re.compile(r"\A[ \t]*(?:export[ \t]+)?" + _re.escape(name) + r"=")

# The seven this script derives. Their MEANING has to match what the container
# gets: a value read one way here and another way by the entrypoint puts two
# different credentials into two files and reports success.
WATCHED = (
    "OPENPHONEX_CONTROL_PLANE_URL",
    "OPENPHONEX_GATEWAY_ID",
    "OPENPHONEX_GATEWAY_API_TOKEN",
    "OPENPHONEX_ORGANIZATION_ID",
    "SCH236_TEST_DID_E164",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
)

try:
    # newline="" so universal-newline translation does NOT hide a carriage
    # return. Finding one is part of the point of this pass.
    with open(os.environ["ENV_LOOKUP_FILE"], encoding="utf-8", newline="") as handle:
        text = handle.read()
except OSError:
    sys.exit(0)


def semantic_problems(name, raw):
    """Ways this value would mean one thing here and another to `source`."""
    if raw.endswith("\r"):
        return [name + " ends with a carriage return (Windows line endings)"]
    for char in raw:
        if char.isspace() and char not in " \t\r\n":
            return [
                name
                + " contains a non-ASCII space (usually pasted from a browser or a document);"
                + " retype the value"
            ]
    if raw[:1] in (" ", "\t"):
        rest = raw.strip()
        if rest and not rest.startswith("#"):
            # `FOO= value` sets FOO to NOTHING and runs `value` as a command.
            return [
                name
                + " has a space straight after the =, so it would be set to nothing"
                + " and the rest of the line would be run as a command"
            ]
        return []
    value = raw.strip()
    if value[:1] in ("'", '"'):
        quote = value[0]
        stop = value.find(quote, 1)
        if stop < 0:
            return [name + " opens a quote it never closes"]
        if "\\" in value[: stop + 1]:
            return [name + " uses a backslash escape, which this script does not interpret"]
        rest = value[stop + 1:]
        # The shell needs WHITESPACE before a comment, so text stuck straight
        # onto a closing quote is part of the value. Reading it as a comment
        # put a different secret in .env.livekit than in the YAML files.
        if rest.strip() and not (rest[:1] in (" ", "\t") and rest.strip().startswith("#")):
            return [name + " has more than one segment after its closing quote"]
        if quote == '"' and ("$" in value[1:stop] or "`" in value[1:stop]):
            return [name + " is double-quoted, so the entrypoint would expand it"]
        return []
    for marker in (" #", "\t#"):
        cut = value.find(marker)
        if cut >= 0:
            value = value[:cut].rstrip()
    # Only now, with any trailing comment removed: a note like
    # `KEY=abc # copied from C:\\Users\\me` is a perfectly good line, and the
    # shell never looks at the comment.
    if "$" in value or "`" in value:
        return [name + " would be expanded by the entrypoint"]
    if re.search(r"[ \t]", value):
        # `FOO=a b` sets FOO=a and then RUNS b.
        return [name + " has an unquoted space, so the shell would run the rest as a command"]
    if re.search(r"""[\\'"&;|<>()]""", value):
        # Adjacent quoting (`ab'c'd` is `abcd`), a background `&`, a redirect --
        # every one of these means something to the shell and nothing to the
        # reader above.
        return [
            name
            + " has an unquoted shell metacharacter; single-quote the whole value so it is literal"
        ]
    return []


def unterminated_quote(line):
    """The quote the shell would still be inside at end of line, or "".

    Tokenised rather than counted: `FOO=1.2.3.0/24 # don't guess` is a fine
    line, and counting apostrophes across the comment would refuse it.
    """
    quote = ""
    previous = ""
    index = 0
    while index < len(line):
        char = line[index]
        if quote != "'" and char == "\\":
            # A backslash escapes the next character everywhere except inside
            # single quotes, so `abc\\'def` is not an open quote.
            index += 2
            previous = ""
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or previous in (" ", "\t")):
            return ""
        previous = char
        index += 1
    return quote


problems = []
for number, line in enumerate(text.split("\n"), start=1):
    if line.lstrip().startswith("#"):
        continue
    # Tier 1, EVERY variable: an unbalanced quote stops the container sourcing
    # this file at all, so it is not only the seven that matter here.
    if "=" in line:
        open_quote = unterminated_quote(line)
        if open_quote:
            problems.append(
                "line %d: an unbalanced %s would break sourcing this file" % (number, open_quote)
            )
    # Tier 2, the seven this script reads and writes. Matched with the SAME
    # definition the reader and the writer use, so a line one of them acts on
    # can never be a line this pass skipped.
    for name in WATCHED:
        match = _assignment(name).match(line)
        if match:
            problems.extend(
                "line %d: %s" % (number, problem)
                for problem in semantic_problems(name, line[match.end():])
            )
        elif re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"=", line):
            # `DUMMY=1 LIVEKIT_API_KEY=x` IS an assignment to the shell and is
            # not one to this script, so the two would end up disagreeing.
            problems.append(
                "line %d: %s is assigned in a position this script does not read; "
                "put it at the start of its own line" % (number, name)
            )

for problem in problems:
    print("[!!] " + problem, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
}

assert_env_file_is_interpretable
ENV_CPU="$(env_value OPENPHONEX_CONTROL_PLANE_URL)"
if [ -n "$CPU" ] && [ -n "$ENV_CPU" ] && [ "$CPU" != "$ENV_CPU" ]; then
  die "OPENPHONEX_CONTROL_PLANE_URL disagrees with $ENV_FILE" "the environment says '$CPU' and the file says '$ENV_CPU'. Pick one -- this script will not overwrite the file, so it would validate one control plane while the stack talked to another"
fi
[ -n "$CPU" ] || CPU="$ENV_CPU"
if [ -z "$CPU" ]; then
  printf 'Control-plane URL (e.g. https://api.openphonex.com): '
  IFS= read -r CPU || true
fi
[ -n "$CPU" ] || die "no control-plane URL" "set OPENPHONEX_CONTROL_PLANE_URL, or put it in $ENV_FILE"

# Same rule the agent enforces at runtime, computed the same way (urlsplit, so
# `http://remote?@localhost` cannot masquerade as loopback). Refusing here means
# the bootstrap key never leaves the box over a channel the worker would reject.
CP_CLASS="$(OPX_CP_URL="$CPU" python3 - <<'PY'
import os, sys, urllib.parse
split = urllib.parse.urlsplit(os.environ["OPX_CP_URL"])
host = split.hostname or ""


def loopback(name: str) -> bool:
    if not name:
        return False
    if name.lower() == "localhost":
        return True
    import ipaddress
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


if split.scheme == "https":
    print("https", "loopback" if loopback(host) else "remote")
    sys.exit(0)
if split.scheme == "http" and loopback(host):
    print("loopback", "loopback")
    sys.exit(0)
print("reject", "remote")
sys.exit(1)
PY
)" || die "control-plane URL is not safe to send a credential to" "'$CPU' must be https://, or http:// on an exact loopback host (127.0.0.0/8, ::1, localhost)"

# A LOOPBACK control plane must never route a credentialed curl through a proxy
# -- the proxy would receive the Authorization header. An ARRAY, not a string:
# an unquoted "--noproxy *" would be split AND glob-expanded against this
# directory, and curl would silently treat the filenames as extra URLs.
CP_TRANSFER_OK=0
CP_NOPROXY=()
case "$CP_CLASS" in
  *loopback) CP_NOPROXY=(--noproxy '*') ;;
esac

# --- state --------------------------------------------------------------------

state_value() {
  [ -f "$STATE_FILE" ] || return 0
  STATE_LOOKUP_NAME="$1" STATE_LOOKUP_FILE="$STATE_FILE" python3 - <<'PY'
import json, os, sys
try:
    with open(os.environ["STATE_LOOKUP_FILE"], encoding="utf-8") as handle:
        document = json.load(handle)
except (OSError, ValueError):
    sys.exit(0)
if not isinstance(document, dict):
    sys.exit(0)
value = document.get(os.environ["STATE_LOOKUP_NAME"])
if isinstance(value, str):
    print(value)
PY
}

STATE_PRESENT=0
[ -f "$STATE_FILE" ] && STATE_PRESENT=1

# --- the bootstrap key --------------------------------------------------------

BOOTSTRAP_KEY="$(eval "printf '%s' \"\${${BOOTSTRAP_KEY_VAR}:-}\"")"

# --- HTTP ---------------------------------------------------------------------

# Bearer tokens go to curl as a config on stdin (`-K -`), never in argv, which
# `ps` shows to every user on the box. `printf` is a builtin, so the value never
# becomes another process's arguments either.
#
# $1 = the NAME of the shell variable holding the token, $2 = method,
# $3 = path, $4 = request body ("" for none). Prints "<status>\n<body>".
cp_call() {
  _cc_token="$(eval "printf '%s' \"\${$1:-}\"")"
  # A bearer value is fed to curl as a CONFIG line. A newline inside it would
  # not be a broken header, it would be an extra curl directive -- so refuse
  # one rather than execute it.
  # ANSI-C quoting, not $(printf '\n'): command substitution strips trailing
  # newlines, so that spelling yields an EMPTY pattern which matches every
  # token -- a guard that refuses everything is not a guard.
  case "$_cc_token" in
    *$'\n'*|*$'\r'*)
      die "a control-plane credential contains a line break" "check $ENV_FILE: a token must be one line"
      ;;
  esac
  _cc_method="$2"
  _cc_path="$3"
  _cc_body="${4:-}"
  # The response body NEVER touches the disk. A mint response carries a live,
  # non-expiring credential, and a temporary file survives an interrupt between
  # the write and the cleanup -- so curl appends the status code to stdout
  # instead of being handed a file to write the body into.
  # CP_TRANSFER_OK is curl's OWN exit code, captured next to the call and kept
  # separate from the HTTP status: a reset that truncates the body still prints
  # the status line, so a caller that looked only at "201" would report a clean
  # mint from a response it never fully received.
  if [ -n "$_cc_body" ]; then
    _cc_raw="$(printf 'header = "Authorization: Bearer %s"\n' "$_cc_token" | curl -K - \
      ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} -s -m 30 -w '\n%{http_code}' \
      -X "$_cc_method" -H 'Content-Type: application/json' --data-binary "$_cc_body" \
      "${CPU%/}$_cc_path" 2>/dev/null)"
    CP_TRANSFER_OK=$?
  else
    _cc_raw="$(printf 'header = "Authorization: Bearer %s"\n' "$_cc_token" | curl -K - \
      ${CP_NOPROXY[@]+"${CP_NOPROXY[@]}"} -s -m 30 -w '\n%{http_code}' \
      -X "$_cc_method" "${CPU%/}$_cc_path" 2>/dev/null)"
    CP_TRANSFER_OK=$?
  fi
  # The status is the last line curl printed; everything before it is the body.
  _cc_status="$(printf '%s' "$_cc_raw" | tail -n 1)"
  printf '%s\n' "${_cc_status:-000}"
  printf '%s' "$_cc_raw" | sed '$d'
}

# Said whenever a mint request was SENT and its outcome is not a token we hold.
# The control plane may have created one; only the operator can check.
on_interrupt() {
  [ "${MINT_IN_FLIGHT:-0}" = 1 ] && warn_possible_orphan
  exit 130
}
trap on_interrupt INT TERM

warn_possible_orphan() {
  say ""
  say "     NOTE: the mint request reached the control plane, so a gateway token"
  say "     may exist for $GATEWAY_ID even though this box did not receive one."
  say "     Check API Keys -> Gateway Tokens and revoke any token you do not hold."
}

# --- persisting what we derived ----------------------------------------------
#
# Defined before the steps that can fail, and called by them: a token minted a
# moment ago has to reach the disk even when the NEXT step gives up, or a
# re-run would mint a second one and orphan the first.

write_env_file() {
#
# Anchored: each value replaces the EMPTY template assignment for its name, and
# only if there is exactly one of them. A name that is already set is left
# alone, so re-running can never overwrite something an operator typed.
#
# Each OPX_SET_<NAME> is paired with an OPX_NOW_<NAME>: what the container
# would see for that name TODAY, read once by env_value. The writer never
# re-interprets the file, so it cannot disagree with the rest of the script
# about whether a value is set.
OPX_ENV_FILE="$ENV_FILE" \
OPX_SET_OPENPHONEX_CONTROL_PLANE_URL="$CPU" \
OPX_NOW_OPENPHONEX_CONTROL_PLANE_URL="$(env_value OPENPHONEX_CONTROL_PLANE_URL)" \
OPX_SET_OPENPHONEX_GATEWAY_ID="$GATEWAY_ID" \
OPX_NOW_OPENPHONEX_GATEWAY_ID="$(env_value OPENPHONEX_GATEWAY_ID)" \
OPX_SET_OPENPHONEX_GATEWAY_API_TOKEN="$GATEWAY_TOKEN" \
OPX_NOW_OPENPHONEX_GATEWAY_API_TOKEN="$(env_value OPENPHONEX_GATEWAY_API_TOKEN)" \
OPX_SET_OPENPHONEX_ORGANIZATION_ID="$ORGANIZATION_ID" \
OPX_NOW_OPENPHONEX_ORGANIZATION_ID="$(env_value OPENPHONEX_ORGANIZATION_ID)" \
OPX_SET_SCH236_TEST_DID_E164="$DID_E164" \
OPX_NOW_SCH236_TEST_DID_E164="$(env_value SCH236_TEST_DID_E164)" \
OPX_SET_LIVEKIT_API_KEY="$LIVEKIT_KEY" \
OPX_NOW_LIVEKIT_API_KEY="$(env_value LIVEKIT_API_KEY)" \
OPX_SET_LIVEKIT_API_SECRET="$LIVEKIT_SECRET" \
OPX_NOW_LIVEKIT_API_SECRET="$(env_value LIVEKIT_API_SECRET)" \
python3 - <<'PY' || return 1
import os, re, sys


def _assignment(name: str):
    """Every line the CONTAINER would treat as an assignment to `name`.

    Leading whitespace is allowed because `source` allows it, and `export` may
    be followed by any run of blanks. This definition is duplicated verbatim in
    the writer below; a test pins the two copies byte-identical, because the
    two ways this file has already gone wrong were a reader and a writer that
    disagreed about which line is the assignment.
    """
    import re as _re

    return _re.compile(r"\A[ \t]*(?:export[ \t]+)?" + _re.escape(name) + r"=")


path = os.environ["OPX_ENV_FILE"]
with open(path, encoding="utf-8") as handle:
    lines = handle.read().split("\n")

# Bare where it is unambiguous; single-quoted otherwise. This file is `source`d
# by the container entrypoint, so an unquoted `$`, backtick or space would be
# expanded or split.
SAFE = re.compile(r"\A[A-Za-z0-9_./:+@,=-]+\Z")


def render(name: str, value: str) -> str:
    if SAFE.match(value):
        return f"{name}={value}"
    return "{}='{}'".format(name, value.replace("'", "'\\''"))


written: list[str] = []
skipped: list[str] = []
for variable, value in sorted(
    (name[len("OPX_SET_"):], raw)
    for name, raw in os.environ.items()
    if name.startswith("OPX_SET_")
):
    if not value:
        continue
    if f"OPX_NOW_{variable}" not in os.environ:
        # A caller that forgot the pairing would silently fall back to
        # "unset" and overwrite an operator's value.
        print(f"[!!] internal: no OPX_NOW_{variable} was supplied", file=sys.stderr)
        sys.exit(1)
    assigned = _assignment(variable)
    # ONE assignment per name, whatever its value. A file holding both
    # `FOO=mine` and a later `FOO=` would otherwise have one of them filled
    # and the two would shadow each other when the container sources this file
    # -- silently, and only at runtime.
    assigned_at = [index for index, line in enumerate(lines) if assigned.match(line)]
    if not assigned_at:
        print(f"[!!] {path} has no '{variable}=' line to fill in", file=sys.stderr)
        sys.exit(1)
    if len(assigned_at) != 1:
        print(
            f"[!!] {path} has {len(assigned_at)} '{variable}=' assignments; refusing to guess "
            "which one the container will end up using",
            file=sys.stderr,
        )
        sys.exit(1)
    # "Already set" means what the CONTAINER would see, decided by the one
    # reader in this script. `FOO=""` is a line, but it is not a value.
    if os.environ.get(f"OPX_NOW_{variable}", ""):
        skipped.append(variable)
        continue
    rendered = render(variable, value)
    lines[assigned_at[0]] = rendered
    # Reverse assertion: the value landed exactly once, and no other line for
    # this name is still empty.
    if sum(1 for line in lines if line == rendered) != 1:
        print(f"[!!] refusing to write {path}: '{variable}' did not land exactly once", file=sys.stderr)
        sys.exit(1)
    if sum(1 for line in lines if assigned.match(line)) != 1:
        print(f"[!!] refusing to write {path}: '{variable}' is assigned more than once", file=sys.stderr)
        sys.exit(1)
    written.append(variable)

with open(path, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines))

for variable in written:
    print(f"[ok] set {variable}")
for variable in skipped:
    print(f"[ok] left {variable} as it was")
PY
  chmod 600 "$ENV_FILE" 2>/dev/null || true

}

write_state_file() {
#
# Identity and metadata only. The token secret lives in .env.livekit and
# nowhere else, so this file can be read without handing anyone a credential.
# No container mounts it; it exists so a re-run knows what it already did.
OPX_STATE_FILE="$STATE_FILE" \
OPX_STATE_CPU="$CPU" \
OPX_STATE_ORG="$ORGANIZATION_ID" \
OPX_STATE_PROJECT="$PROJECT_ID" \
OPX_STATE_GATEWAY="$GATEWAY_ID" \
OPX_STATE_TOKEN_ID="$GATEWAY_TOKEN_ID" \
OPX_STATE_TOKEN_PREFIX="$GATEWAY_TOKEN_PREFIX" \
OPX_STATE_DID="$DID_E164" \
OPX_STATE_ADOPTED="$ADOPTED" \
python3 - <<'PY' || return 1
import json, os
from datetime import datetime, timezone

path = os.environ["OPX_STATE_FILE"]
now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
previous = {}
try:
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict):
        previous = loaded
except (OSError, ValueError):
    previous = {}

document = {
    "schema_version": 1,
    "control_plane_url": os.environ["OPX_STATE_CPU"],
    "organization_id": os.environ["OPX_STATE_ORG"],
    "project_id": os.environ["OPX_STATE_PROJECT"] or str(previous.get("project_id") or ""),
    "gateway_id": os.environ["OPX_STATE_GATEWAY"],
    "gateway_token_id": os.environ["OPX_STATE_TOKEN_ID"] or str(previous.get("gateway_token_id") or ""),
    "gateway_token_prefix": os.environ["OPX_STATE_TOKEN_PREFIX"] or str(previous.get("gateway_token_prefix") or ""),
    "did_e164": os.environ["OPX_STATE_DID"],
    "adopted": os.environ["OPX_STATE_ADOPTED"] == "1",
    "created_at": str(previous.get("created_at") or now),
    "updated_at": now,
}
with open(path, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY
  chmod 600 "$STATE_FILE" 2>/dev/null || true

}

persist() {
  write_env_file || return 1
  write_state_file || return 1
}

# Blank one assignment so the anchored writer may fill it.
#
# The writer never overwrites a non-empty value -- that is what keeps a re-run
# from clobbering something an operator typed. A token we have just REPLACED is
# the one case where the old value must go, so it is cleared here, explicitly,
# by name, and only once the replacement exists.
blank_env_assignment() {
  OPX_ENV_FILE="$ENV_FILE" OPX_BLANK_NAME="$1" python3 - <<'PY' || return 1
import os, re, sys


def _assignment(name: str):
    """Every line the CONTAINER would treat as an assignment to `name`.

    Leading whitespace is allowed because `source` allows it, and `export` may
    be followed by any run of blanks. This definition is duplicated verbatim in
    the writer below; a test pins the two copies byte-identical, because the
    two ways this file has already gone wrong were a reader and a writer that
    disagreed about which line is the assignment.
    """
    import re as _re

    return _re.compile(r"\A[ \t]*(?:export[ \t]+)?" + _re.escape(name) + r"=")


path = os.environ["OPX_ENV_FILE"]
name = os.environ["OPX_BLANK_NAME"]
with open(path, encoding="utf-8") as handle:
    lines = handle.read().split("\n")
assigned = _assignment(name)
at = [index for index, line in enumerate(lines) if assigned.match(line)]
if len(at) != 1:
    print(f"[!!] {path} has {len(at)} '{name}=' lines; refusing to clear any of them", file=sys.stderr)
    sys.exit(1)
lines[at[0]] = f"{name}="
if sum(1 for line in lines if line == f"{name}=") != 1:
    print(f"[!!] refusing to write {path}: '{name}' did not clear exactly once", file=sys.stderr)
    sys.exit(1)
with open(path, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines))
PY
}

# --- what is already set ------------------------------------------------------

ENV_ORG="$(env_value OPENPHONEX_ORGANIZATION_ID)"
ENV_GATEWAY="$(env_value OPENPHONEX_GATEWAY_ID)"
ENV_TOKEN="$(env_value OPENPHONEX_GATEWAY_API_TOKEN)"
ENV_DID="$(env_value SCH236_TEST_DID_E164)"
ENV_LK_KEY="$(env_value LIVEKIT_API_KEY)"
ENV_LK_SECRET="$(env_value LIVEKIT_API_SECRET)"

STATE_ORG="$(state_value organization_id)"
STATE_PROJECT="$(state_value project_id)"
STATE_GATEWAY="$(state_value gateway_id)"
STATE_TOKEN_ID="$(state_value gateway_token_id)"
STATE_DID="$(state_value did_e164)"

# Seeded here, not in section 5: persist() runs before that section whenever a
# credential has just been minted, and `set -u` turns an unset name into a hard
# failure at exactly the moment the token must reach the disk.
LIVEKIT_KEY="$ENV_LK_KEY"
LIVEKIT_SECRET="$ENV_LK_SECRET"
YAML_WRITTEN=0

# An explicit override, for a control plane too old to default the organization
# from the key. It is checked against the file rather than silently winning: the
# writer never overwrites a non-empty value, so a process value that contradicts
# .env.livekit would leave this script talking about one workspace and the
# container running in another.
ORGANIZATION_ID="${OPENPHONEX_ORGANIZATION_ID:-}"
if [ -n "$ORGANIZATION_ID" ] && [ -n "$ENV_ORG" ] && [ "$ORGANIZATION_ID" != "$ENV_ORG" ]; then
  die "OPENPHONEX_ORGANIZATION_ID disagrees with $ENV_FILE" "the environment says '$ORGANIZATION_ID' and the file says '$ENV_ORG'. Pick one -- this script will not overwrite the file, so it would leave the two disagreeing"
fi
[ -n "$ORGANIZATION_ID" ] || ORGANIZATION_ID="${ENV_ORG:-$STATE_ORG}"
PROJECT_ID="$STATE_PROJECT"
GATEWAY_ID="${ENV_GATEWAY:-$STATE_GATEWAY}"
GATEWAY_TOKEN="$ENV_TOKEN"
DID_E164="${ENV_DID:-$STATE_DID}"
GATEWAY_TOKEN_ID=""
GATEWAY_TOKEN_PREFIX=""
REVOKED_TOKEN_ID=""
REGISTERED=0
MINTED=0
ADOPTED=0

if [ ! -f "$ENV_FILE" ]; then
  [ -f "$ENV_EXAMPLE" ] || die "$ENV_FILE is missing and so is $ENV_EXAMPLE" "run this from packages/openphonex-gateway/deploy/"
  cp "$ENV_EXAMPLE" "$ENV_FILE" || die "could not create $ENV_FILE"
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  ok "created $ENV_FILE from the template (chmod 600)"
  # The gate above ran before this file existed. Re-run it so "checked" always
  # means "checked the file we are about to read and write".
  assert_env_file_is_interpretable
fi

# --- adoption -----------------------------------------------------------------
#
# A deployment that already has every value only gains a state file. Anything
# else would mean re-registering a gateway that is already answering calls.
if [ -n "$ENV_GATEWAY" ] && [ -n "$ENV_TOKEN" ] && [ -n "$ENV_ORG" ] && [ -n "$ENV_DID" ] \
   && [ -n "$ENV_LK_KEY" ] && [ -n "$ENV_LK_SECRET" ] && [ "$STATE_PRESENT" = 0 ]; then
  ADOPTED=1
fi

if [ "$ADOPTED" = 0 ]; then
  # --- decide whether a control-plane credential is needed at all --------------
  NEED_REGISTER=0
  NEED_TOKEN=0
  NEED_DID=0
  [ -n "$GATEWAY_ID" ] || NEED_REGISTER=1
  [ -n "$GATEWAY_TOKEN" ] || NEED_TOKEN=1
  [ -n "$DID_E164" ] || NEED_DID=1

  if [ "$NEED_REGISTER" = 1 ] || [ "$NEED_TOKEN" = 1 ] || [ "$NEED_DID" = 1 ] || [ -z "$ORGANIZATION_ID" ]; then
    if [ -z "$BOOTSTRAP_KEY" ] && [ "$NEED_TOKEN" = 0 ] && [ "$NEED_REGISTER" = 0 ] && [ -n "$ORGANIZATION_ID" ]; then
      : # only the DID is missing and the gateway token can read it
    elif [ -z "$BOOTSTRAP_KEY" ]; then
      printf 'Bootstrap API key (organization key with gateways:write, not echoed): '
      stty -echo 2>/dev/null || true
      IFS= read -r BOOTSTRAP_KEY || true
      stty echo 2>/dev/null || true
      printf '\n'
    fi
  fi

  # --- 1. register the deployment ---------------------------------------------
  if [ "$NEED_REGISTER" = 1 ]; then
    [ -n "$BOOTSTRAP_KEY" ] || die "no bootstrap key" "set $BOOTSTRAP_KEY_VAR (an organization API key with gateways:write) and re-run"
    step "registering this deployment"
    _body='{"mode":"self_hosted"}'
    if [ -n "$ORGANIZATION_ID" ]; then
      _body="$(OPX_ORG="$ORGANIZATION_ID" python3 -c 'import json,os; print(json.dumps({"mode":"self_hosted","organization_id":os.environ["OPX_ORG"]}))')"
    fi
    _resp="$(cp_call BOOTSTRAP_KEY POST /v1/gateway-deployments "$_body")"
    _status="$(printf '%s' "$_resp" | head -n 1)"
    _payload="$(printf '%s' "$_resp" | tail -n +2)"
    case "$_status" in
      201|200) : ;;
      000) die "the control plane did not answer" "no HTTP response from ${CPU%/}/v1/gateway-deployments — wrong URL, network, or the control plane is down" ;;
      401) die "the bootstrap key was rejected (HTTP 401)" "it is wrong, revoked or expired; mint a fresh organization API key with gateways:write" ;;
      403) die "the bootstrap key lacks gateways:write (HTTP 403)" "mint an organization API key carrying gateways:write and re-run" ;;
      400)
        # A control plane older than the defaulting change still demands both
        # ids in the body. Say exactly which override fixes it.
        die "the control plane refused the registration (HTTP 400)" "$(printf '%s' "$_payload" | head -c 400). If it is asking for organization_id, this control plane predates principal-defaulted registration: re-run with OPENPHONEX_ORGANIZATION_ID=org_... in the environment, the same way you pass the control-plane URL (find it under dashboard -> organization settings)"
        ;;
      *) die "unexpected HTTP $_status registering the deployment" "$(printf '%s' "$_payload" | head -c 400)" ;;
    esac
    _fields="$(OPX_PAYLOAD="$_payload" python3 - <<'PY'
import json, os, sys
try:
    document = json.loads(os.environ["OPX_PAYLOAD"])
except ValueError:
    sys.exit(1)
if not isinstance(document, dict):
    sys.exit(1)
for name in ("id", "organization_id", "project_id"):
    print(str(document.get(name) or ""))
PY
)" || die "the control plane returned an unreadable deployment" "expected JSON with id/organization_id/project_id"
    GATEWAY_ID="$(printf '%s\n' "$_fields" | sed -n 1p)"
    ORGANIZATION_ID="$(printf '%s\n' "$_fields" | sed -n 2p)"
    PROJECT_ID="$(printf '%s\n' "$_fields" | sed -n 3p)"
    [ -n "$GATEWAY_ID" ] || die "the control plane returned no gateway id"
    REGISTERED=1
    ok "registered gateway $GATEWAY_ID in $ORGANIZATION_ID"
  fi

  # --- 2. the gateway token ----------------------------------------------------
  #
  # Probe an existing token before minting. Without the probe, every casual
  # re-run would leave one more live, orphaned credential in the workspace.
  TOKEN_OK=0
  if [ -n "$GATEWAY_TOKEN" ]; then
    step "checking the gateway token already in $ENV_FILE"
    _resp="$(cp_call GATEWAY_TOKEN GET "/v1/gateway-deployments/$GATEWAY_ID/number-rules" "")"
    _status="$(printf '%s' "$_resp" | head -n 1)"
    case "$_status" in
      200|304) TOKEN_OK=1; ok "the existing gateway token still works" ;;
      401) say "     the existing gateway token is no longer accepted (HTTP 401); a replacement will be minted" ;;
      403)
        # Authenticated, but refused for THIS gateway. Two causes look identical
        # from here: a token belonging to another deployment, and one minted
        # without number-rules:read. Neither justifies this script overwriting a
        # credential that still authenticates -- it could be another gateway's
        # only copy -- and both are fixed by the same one-line edit.
        die "the token in $ENV_FILE authenticates but is refused for $GATEWAY_ID (HTTP 403)" "either it belongs to another deployment, or it was minted without number-rules:read. Clear the OPENPHONEX_GATEWAY_API_TOKEN= line in $ENV_FILE and re-run: a replacement will be minted. This script will not clear it for you, because a token that still authenticates may be another deployment's only copy"
        ;;
      000) die "the control plane did not answer" "no HTTP response from ${CPU%/} — wrong URL, network, or the control plane is down" ;;
      *) die "unexpected HTTP $_status reading number rules" "cannot tell whether the existing token is usable; fix the control plane first" ;;
    esac
  fi

  if [ "$TOKEN_OK" = 1 ]; then
    # Carry the ids the state file already knows so re-persisting does not
    # blank them out.
    GATEWAY_TOKEN_ID="$STATE_TOKEN_ID"
    GATEWAY_TOKEN_PREFIX="$(state_value gateway_token_prefix)"
  fi

  if [ "$TOKEN_OK" = 0 ]; then
    [ -n "$BOOTSTRAP_KEY" ] || die "no bootstrap key" "set $BOOTSTRAP_KEY_VAR (an organization API key with gateways:write) and re-run"
    # Prove the file can RECEIVE the token before asking for one. A one-time
    # secret that is minted and then cannot be written is the worst outcome
    # here: it is live in the workspace, recoverable from nowhere, and the next
    # run mints another. Normalising the single assignment to an empty anchor
    # proves all three things the write needs -- the line exists, there is
    # exactly one of it, and this file is writable -- and can only ever discard
    # a token the control plane has already refused.
    blank_env_assignment OPENPHONEX_GATEWAY_API_TOKEN \
      || die "cannot write OPENPHONEX_GATEWAY_API_TOKEN into $ENV_FILE" "the file needs exactly one OPENPHONEX_GATEWAY_API_TOKEN= line and must be writable; fix it and re-run. Nothing has been minted"
    step "minting a gateway token for $GATEWAY_ID"
    MINT_IN_FLIGHT=1
    # No expires_in_days on purpose: an expiry without auto-rotation only
    # manufactures a lockout on a box nobody is watching.
    _resp="$(cp_call BOOTSTRAP_KEY POST "/v1/gateway-deployments/$GATEWAY_ID/tokens" '{"name":"selfhost-bootstrap"}')"
    _status="$(printf '%s' "$_resp" | head -n 1)"
    _payload="$(printf '%s' "$_resp" | tail -n +2)"
    case "$_status" in
      201|200)
        if [ "${CP_TRANSFER_OK:-0}" != 0 ]; then
          warn_possible_orphan
          die "the mint response was cut short (curl exit ${CP_TRANSFER_OK})" "re-run once the connection is stable"
        fi
        ;;
      401|403) die "the bootstrap key may not mint a token for $GATEWAY_ID (HTTP $_status)" "it needs gateways:write in the workspace that owns this deployment" ;;
      404) die "gateway $GATEWAY_ID does not exist in this workspace (HTTP 404)" "clear OPENPHONEX_GATEWAY_ID in $ENV_FILE and $STATE_FILE to register a new deployment" ;;
      000)
        # No status at all. The request may still have been committed before the
        # connection died, so this is exactly the case that needs the warning.
        warn_possible_orphan
        die "the control plane did not answer" "no HTTP response from ${CPU%/}"
        ;;
      *)
        warn_possible_orphan
        die "unexpected HTTP $_status minting the gateway token" "$(printf '%s' "$_payload" | head -c 400)"
        ;;
    esac
    _minted="$(OPX_PAYLOAD="$_payload" python3 - <<'PY'
import json, os, sys
try:
    document = json.loads(os.environ["OPX_PAYLOAD"])
except ValueError:
    sys.exit(1)
if not isinstance(document, dict) or not document.get("token"):
    sys.exit(1)
print(str(document.get("token")))
print(str(document.get("id") or ""))
print(str(document.get("token_prefix") or ""))
PY
)" || { warn_possible_orphan; die "the control plane returned no token" "expected a one-time \`token\` field in the mint response"; }
    GATEWAY_TOKEN="$(printf '%s\n' "$_minted" | sed -n 1p)"
    GATEWAY_TOKEN_ID="$(printf '%s\n' "$_minted" | sed -n 2p)"
    GATEWAY_TOKEN_PREFIX="$(printf '%s\n' "$_minted" | sed -n 3p)"
    MINTED=1
    ok "minted gateway token ${GATEWAY_TOKEN_PREFIX:-tgw_...} (no expiry)"

    # The token this one replaces is dead weight in the workspace. Retire it.
    if [ -n "$STATE_TOKEN_ID" ] && [ "$STATE_TOKEN_ID" != "$GATEWAY_TOKEN_ID" ]; then
      _resp="$(cp_call BOOTSTRAP_KEY POST "/v1/gateway-deployments/$GATEWAY_ID/tokens/$STATE_TOKEN_ID/revoke" '{}')"
      _status="$(printf '%s' "$_resp" | head -n 1)"
      case "$_status" in
        200|201|404) REVOKED_TOKEN_ID="$STATE_TOKEN_ID"; ok "revoked the replaced token $STATE_TOKEN_ID" ;;
        *) say "     could not revoke the replaced token $STATE_TOKEN_ID (HTTP $_status) — revoke it from API Keys -> Gateway Tokens" ;;
      esac
    fi
  fi

  # A token that exists must reach the disk before the next step is allowed to
  # fail. Otherwise a DID problem strands a live credential nothing records,
  # and the re-run mints another one.
  if [ "$MINTED" = 1 ] || [ "$REGISTERED" = 1 ]; then
    persist || { warn_possible_orphan; die "could not write $ENV_FILE / $STATE_FILE" "the gateway token has been minted; fix the file permissions and re-run"; }
    MINT_IN_FLIGHT=0
    ok "saved $ENV_FILE and $STATE_FILE"
  fi

  # --- 3. the DID --------------------------------------------------------------
  if [ -z "$DID_E164" ]; then
    step "looking for an agent-routed number on this deployment"
    _resp="$(cp_call GATEWAY_TOKEN GET "/v1/gateway-deployments/$GATEWAY_ID/number-rules" "")"
    _status="$(printf '%s' "$_resp" | head -n 1)"
    _payload="$(printf '%s' "$_resp" | tail -n +2)"
    case "$_status" in
      200) : ;;
      000) die "the control plane did not answer" "no HTTP response from ${CPU%/}" ;;
      *) die "unexpected HTTP $_status reading number rules" "$(printf '%s' "$_payload" | head -c 400)" ;;
    esac
    _candidates="$(OPX_PAYLOAD="$_payload" python3 - <<'PY'
import json, os, sys
try:
    document = json.loads(os.environ["OPX_PAYLOAD"])
except ValueError:
    sys.exit(2)
rules = document.get("number_rules") if isinstance(document, dict) else None
if not isinstance(rules, list):
    sys.exit(2)
saw_rule = False
found = []
for rule in rules:
    if not isinstance(rule, dict):
        continue
    saw_rule = True
    incoming = rule.get("incoming")
    if not isinstance(incoming, dict):
        continue
    if incoming.get("when_called") != "answer_with_openphonex_agent":
        continue
    e164 = str(rule.get("e164") or "").strip()
    if e164:
        found.append(e164)
# A control plane that predates per-rule e164 answers with rules and no number.
# That is a different failure from "you have no agent-routed number", and
# telling the operator the wrong one sends them to fix the wrong thing.
if saw_rule and not found and not any(str(r.get("e164") or "").strip() for r in rules if isinstance(r, dict)):
    sys.exit(3)
for e164 in sorted(set(found)):
    print(e164)
PY
)"
    _rc=$?
    case "$_rc" in
      0) : ;;
      3) die "this control plane does not report the number behind each rule" "it predates per-rule e164 in the gateway number-rules document; set SCH236_TEST_DID_E164 in $ENV_FILE by hand and re-run" ;;
      *) die "the number-rules document was unreadable" "expected JSON with a number_rules array" ;;
    esac
    _count="$(printf '%s' "$_candidates" | grep -c '^+' 2>/dev/null || true)"
    [ -n "$_count" ] || _count=0
    if [ "$_count" -eq 0 ]; then
      die "no agent-routed number on this deployment's workspace" "in the dashboard, point a number at a published agent (Numbers -> the number -> answer with an OpenPhonex agent), then re-run. Or set SCH236_TEST_DID_E164 by hand"
    elif [ "$_count" -gt 1 ]; then
      say ""
      say "     this workspace has more than one agent-routed number:"
      printf '%s\n' "$_candidates" | sed 's/^/       /'
      die "cannot choose a number for you" "provisioning builds ONE inbound trunk, and picking the wrong number would leave you with a green stack that never rings. Set SCH236_TEST_DID_E164 in $ENV_FILE (comma-separate to serve several) and re-run"
    fi
    DID_E164="$(printf '%s' "$_candidates" | sed -n 1p)"
    ok "using $DID_E164"
  fi

  # --- 4. the organization ------------------------------------------------------
  if [ -z "$ORGANIZATION_ID" ]; then
    _resp="$(cp_call GATEWAY_TOKEN GET "/v1/gateway-deployments/$GATEWAY_ID/number-rules" "")"
    _payload="$(printf '%s' "$_resp" | tail -n +2)"
    ORGANIZATION_ID="$(OPX_PAYLOAD="$_payload" python3 -c '
import json, os
try:
    document = json.loads(os.environ["OPX_PAYLOAD"])
except ValueError:
    document = {}
print(str((document or {}).get("organization_id") or ""))
' 2>/dev/null)"
    [ -n "$ORGANIZATION_ID" ] || die "could not determine the organization" "set OPENPHONEX_ORGANIZATION_ID in $ENV_FILE (dashboard -> organization settings) and re-run"
  fi
fi

# --- 5. the LiveKit pair ------------------------------------------------------
#
# The same pair has to appear in livekit.yaml, sip.yaml and .env.livekit. It is
# the most error-prone hand step in the whole install and a mismatch shows up
# only as a failed smoke-test step 8, so derive it once here.
if [ -f "$LIVEKIT_YAML" ]; then
  # An operator's existing config is authoritative. Read the pair OUT of it and
  # sync the env to match; never regenerate, never rewrite their file.
  _pair="$(OPX_YAML="$LIVEKIT_YAML" python3 - <<'PY'
import os, re, sys


def _yaml_scalar(raw: str) -> str:
    """A plain YAML scalar: quoted contents, or text up to an unquoted ` #`.

    Reading `key: secret # note` as `secret # note` would copy a comment into
    .env.livekit and leave the two files holding different credentials.
    """
    value = raw.strip()
    if "\\" in value:
        print("[!!] a LiveKit value uses a backslash escape", file=sys.stderr)
        sys.exit(2)
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            print("[!!] a LiveKit value opens a quote it never closes", file=sys.stderr)
            sys.exit(2)
        rest = value[end + 1:].strip()
        if rest and not rest.startswith("#"):
            # YAML doubled-quote escaping lands here, and so does anything
            # else this does not implement. Two servers running on two
            # different secrets is what a guess would buy.
            # (bash inside $(...) mis-parses an odd number of apostrophes in a
            # heredoc body, so this comment keeps them balanced.)
            print("[!!] a LiveKit value uses quoting this script cannot read", file=sys.stderr)
            sys.exit(2)
        return value[1:end]
    cut = value.find(" #")
    if cut >= 0:
        value = value[:cut]
    return value.strip()


try:
    with open(os.environ["OPX_YAML"], encoding="utf-8") as handle:
        lines = handle.read().splitlines()
except OSError:
    sys.exit(1)
inside = False
for line in lines:
    if re.match(r"^keys:\s*$", line):
        inside = True
        continue
    if inside:
        if line[:1] not in (" ", "\t"):
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            continue
        key = _yaml_scalar(key)
        value = _yaml_scalar(value)
        if key and value:
            print(key)
            print(value)
            sys.exit(0)
sys.exit(1)
PY
)" && [ -n "$_pair" ] || die "could not read the LiveKit key pair out of $LIVEKIT_YAML" "it has no filled-in entry under 'keys:', or its value uses quoting this script will not guess at. Fill it in plainly (or delete the file to have one generated) and re-run"
  _yaml_key="$(printf '%s\n' "$_pair" | sed -n 1p)"
  _yaml_secret="$(printf '%s\n' "$_pair" | sed -n 2p)"
  if [ -n "$LIVEKIT_KEY" ] && [ "$LIVEKIT_KEY" != "$_yaml_key" ]; then
    die "LIVEKIT_API_KEY in $ENV_FILE disagrees with $LIVEKIT_YAML" "the same pair must be in livekit.yaml, sip.yaml and $ENV_FILE; fix them by hand — this script will not choose which one is right"
  fi
  # The SECRET half matters just as much, and the writer will not overwrite a
  # non-empty value, so adopting a differing one here would leave the files
  # disagreeing while this script reported success.
  if [ -n "$LIVEKIT_SECRET" ] && [ "$LIVEKIT_SECRET" != "$_yaml_secret" ]; then
    die "LIVEKIT_API_SECRET in $ENV_FILE disagrees with $LIVEKIT_YAML" "the same pair must be in livekit.yaml, sip.yaml and $ENV_FILE; fix them by hand — this script will not choose which one is right"
  fi
  LIVEKIT_KEY="$_yaml_key"
  LIVEKIT_SECRET="$_yaml_secret"
  ok "read the LiveKit pair out of the existing $LIVEKIT_YAML"
elif [ -z "$LIVEKIT_KEY" ] || [ -z "$LIVEKIT_SECRET" ]; then
  # Only the MISSING half is generated. Replacing a key an operator already
  # chose would write one identity into livekit.yaml and leave the other in
  # .env.livekit -- the three-file mismatch this step exists to prevent.
  [ -n "$LIVEKIT_KEY" ] || LIVEKIT_KEY="opx-selfhost"
  if [ -z "$LIVEKIT_SECRET" ]; then
    [ "${OPENSSL_MISSING:-0}" = 1 ] && die "openssl is required to generate the LiveKit secret" "install openssl, or fill LIVEKIT_API_SECRET in $ENV_FILE by hand"
    LIVEKIT_SECRET="$(openssl rand -hex 32)"
    [ -n "$LIVEKIT_SECRET" ] || die "openssl produced no secret"
  fi
fi

# Instantiate the yamls only when they are absent. An existing file is the
# operator's, whatever is in it.
if [ ! -f "$LIVEKIT_YAML" ]; then
  [ -f "$LIVEKIT_YAML_EXAMPLE" ] || die "$LIVEKIT_YAML_EXAMPLE is missing" "run this from packages/openphonex-gateway/deploy/"
  OPX_SRC="$LIVEKIT_YAML_EXAMPLE" OPX_DEST="$LIVEKIT_YAML" OPX_KEY="$LIVEKIT_KEY" OPX_SECRET="$LIVEKIT_SECRET" python3 - <<'PY' || exit 1
import os, sys
src, dest = os.environ["OPX_SRC"], os.environ["OPX_DEST"]
key, secret = os.environ["OPX_KEY"], os.environ["OPX_SECRET"]
with open(src, encoding="utf-8") as handle:
    lines = handle.read().split("\n")
anchor = "  changeme-api-key: changeme-api-secret"
if lines.count(anchor) != 1:
    print(f"[!!] {src} has {lines.count(anchor)} key-pair anchors, expected exactly 1", file=sys.stderr)
    sys.exit(1)
lines[lines.index(anchor)] = f"  {key}: {secret}"
if sum(1 for line in lines if line == f"  {key}: {secret}") != 1:
    print(f"[!!] refusing to write {dest}: the generated pair did not land exactly once", file=sys.stderr)
    sys.exit(1)
with open(dest, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines))
PY
  chmod 600 "$LIVEKIT_YAML" 2>/dev/null || true
  YAML_WRITTEN=1
  ok "wrote $LIVEKIT_YAML"
fi

if [ -f "$SIP_YAML" ]; then
  # Never rewritten -- but a stale one holding a DIFFERENT pair is the exact
  # failure this whole step exists to prevent, and it surfaces only as a failed
  # smoke-test step 8 much later. Check it and say which files disagree.
  _sip_pair="$(OPX_YAML="$SIP_YAML" python3 - <<'PY'
import os, re, sys


def _yaml_scalar(raw: str) -> str:
    """A plain YAML scalar: quoted contents, or text up to an unquoted ` #`."""
    value = raw.strip()
    if "\\" in value:
        print("[!!] a LiveKit value uses a backslash escape", file=sys.stderr)
        sys.exit(2)
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            print("[!!] a LiveKit value opens a quote it never closes", file=sys.stderr)
            sys.exit(2)
        rest = value[end + 1:].strip()
        if rest and not rest.startswith("#"):
            # YAML doubled-quote escaping lands here, and so does anything
            # else this does not implement. Two servers running on two
            # different secrets is what a guess would buy.
            # (bash inside $(...) mis-parses an odd number of apostrophes in a
            # heredoc body, so this comment keeps them balanced.)
            print("[!!] a LiveKit value uses quoting this script cannot read", file=sys.stderr)
            sys.exit(2)
        return value[1:end]
    cut = value.find(" #")
    if cut >= 0:
        value = value[:cut]
    return value.strip()


try:
    with open(os.environ["OPX_YAML"], encoding="utf-8") as handle:
        text = handle.read()
except OSError:
    sys.exit(1)
found = {}
for line in text.splitlines():
    match = re.match(r"^(api_key|api_secret)\s*:\s*(.+?)\s*$", line)
    if match and match.group(1) not in found:
        found[match.group(1)] = _yaml_scalar(match.group(2))
if "api_key" not in found or "api_secret" not in found:
    sys.exit(1)
print(found["api_key"])
print(found["api_secret"])
PY
)" || die "could not read the LiveKit pair out of $SIP_YAML" "it has no api_key/api_secret, or a value uses quoting this script will not guess at. Fill them in plainly (or delete the file to have one written) and re-run"
  _sip_key="$(printf '%s\n' "$_sip_pair" | sed -n 1p)"
  _sip_secret="$(printf '%s\n' "$_sip_pair" | sed -n 2p)"
  if [ "$_sip_key" != "$LIVEKIT_KEY" ] || [ "$_sip_secret" != "$LIVEKIT_SECRET" ]; then
    die "$SIP_YAML holds a different LiveKit pair" "livekit.yaml/$ENV_FILE resolved to key '$LIVEKIT_KEY', $SIP_YAML says '$_sip_key'. One pair must appear in all three files; a mismatch shows up only as a failed smoke-test step 8. Fix them by hand — this script will not choose which one is right"
  fi
  ok "the existing $SIP_YAML already holds the same pair"
else
  [ -f "$SIP_YAML_EXAMPLE" ] || die "$SIP_YAML_EXAMPLE is missing" "run this from packages/openphonex-gateway/deploy/"
  OPX_SRC="$SIP_YAML_EXAMPLE" OPX_DEST="$SIP_YAML" OPX_KEY="$LIVEKIT_KEY" OPX_SECRET="$LIVEKIT_SECRET" python3 - <<'PY' || exit 1
import os, sys
src, dest = os.environ["OPX_SRC"], os.environ["OPX_DEST"]
key, secret = os.environ["OPX_KEY"], os.environ["OPX_SECRET"]
with open(src, encoding="utf-8") as handle:
    lines = handle.read().split("\n")
for anchor, replacement in (
    ("api_key: changeme-api-key", f"api_key: {key}"),
    ("api_secret: changeme-api-secret", f"api_secret: {secret}"),
):
    if lines.count(anchor) != 1:
        print(f"[!!] {src} has {lines.count(anchor)} '{anchor}' anchors, expected exactly 1", file=sys.stderr)
        sys.exit(1)
    lines[lines.index(anchor)] = replacement
    if sum(1 for line in lines if line == replacement) != 1:
        print(f"[!!] refusing to write {dest}: '{replacement.split(':')[0]}' did not land exactly once", file=sys.stderr)
        sys.exit(1)
with open(dest, "w", encoding="utf-8", newline="\n") as handle:
    handle.write("\n".join(lines))
PY
  chmod 600 "$SIP_YAML" 2>/dev/null || true
  YAML_WRITTEN=1
  ok "wrote $SIP_YAML"
fi

persist || die "could not write $ENV_FILE / $STATE_FILE"

# --- summary ------------------------------------------------------------------

say ""
if [ "$ADOPTED" = 1 ]; then
  say "Adopted existing deployment $GATEWAY_ID; nothing changed."
  say "Wrote $STATE_FILE (chmod 600) so a future re-run knows this box is already set up."
else
  say "Bootstrap complete."
  [ "$REGISTERED" = 1 ] && say "  registered   $GATEWAY_ID"
  [ "$MINTED" = 1 ] && say "  minted       gateway token ${GATEWAY_TOKEN_PREFIX:-tgw_...} (no expiry)"
  [ -n "$REVOKED_TOKEN_ID" ] && say "  revoked      replaced token $REVOKED_TOKEN_ID"
  [ "$YAML_WRITTEN" = 1 ] && say "  wrote        livekit.yaml / sip.yaml with one matching key pair"
  say "  wrote        $ENV_FILE and $STATE_FILE (both chmod 600)"
fi
say ""
say "STILL YOURS TO SET, by hand, in $ENV_FILE:"
say "  DIDWW_INBOUND_ADDRESSES  your carrier's PUBLIC SIP SIGNALLING ranges (not media)."
say "                           Nothing in the control plane knows them, and a wrong"
say "                           allowlist is a stack that looks perfect and never rings."
say "  provider API keys        whichever your agent's published voice stack resolves to."
say ""
if [ -n "$BOOTSTRAP_KEY" ]; then
  say "REVOKE THE BOOTSTRAP KEY: the organization key you passed in $BOOTSTRAP_KEY_VAR is"
  say "not needed again and was never written to this box. Revoke it now in the dashboard"
  say "under API Keys. From here on this deployment runs on its own tgw_ token alone."
  say ""
fi
say "Next:  docker compose -f docker-compose.livekit.yml up -d --build"
say "       docker compose -f docker-compose.livekit.yml --profile provision run --rm provision"
say "       bash selfhost-smoketest.sh"

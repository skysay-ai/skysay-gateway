#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
#
# Renders the registering SIP sidecar's runtime state from the environment and
# runs Kamailio. Nothing here is persisted and no credential is ever written
# into the image or the repository: the registration credentials arrive as env
# vars and land only in a runtime directory.
#
# Every input is validated up front and rejected loudly. The failure mode this
# guards against is a config that loads without complaint and then silently
# rejects (or never receives) every call -- which, on a NAT'd host with no
# inbound reachability to test against, is close to undiagnosable.
set -eu

RUN_DIR=/run/openphonex-registrar
DB_DIR="$RUN_DIR/db"

die() {
	echo "REGISTRAR_CONFIG_INVALID {\"error\": \"$1\"}" >&2
	exit 1
}

require() {
	eval "value=\${$1:-}"
	[ -n "$value" ] || die "$1 is required but empty"
}

# IPv4 dotted quad, each octet 0-255. POSIX shell has no regex, so this is done
# arithmetically -- deliberately, because a permissive check here is exactly
# what lets a malformed allowlist entry load as a row that never matches.
is_ipv4() {
	_ip=$1
	case "$_ip" in
		*[!0-9.]* | "" | *..* | .* | *.) return 1 ;;
	esac
	_n=0
	_rest=$_ip
	while [ -n "$_rest" ]; do
		_oct=${_rest%%.*}
		[ -n "$_oct" ] || return 1
		[ "${#_oct}" -le 3 ] || return 1
		# Reject leading zeros. Shell arithmetic reads 010 as OCTAL 8, so
		# "010.0.0.0/8" would sail past the reserved-space check as 8.0.0.0/8,
		# and "08" aborts with "Illegal number" and the wrong error. Python's
		# ipaddress rejects these too, so the two validation layers agree.
		case "$_oct" in
			0) : ;;
			0*) return 1 ;;
		esac
		[ "$_oct" -le 255 ] 2>/dev/null || return 1
		_n=$((_n + 1))
		case "$_rest" in
			*.*) _rest=${_rest#*.} ;;
			*) _rest="" ;;
		esac
	done
	[ "$_n" -eq 4 ]
}

# --- reserved IPv4 space --------------------------------------------------
# Mirrors _NON_PUBLIC_NETS in the self-host provisioner. An allowlist is a
# boundary: a range that is not a plausible carrier signalling source either
# does nothing or, in the 0.0.0.0/0 case, quietly disarms the boundary
# altogether by matching every source that reaches the socket.
_RESERVED_NETS="0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 \
172.16.0.0/12 192.0.0.0/24 192.0.2.0/24 192.88.99.0/24 192.168.0.0/16 \
198.18.0.0/15 198.51.100.0/24 203.0.113.0/24 224.0.0.0/4 240.0.0.0/4 \
255.255.255.255/32"

ip_to_int() {
	IFS=. read -r _a _b _c _d <<EOF
$1
EOF
	echo $(( (_a * 16777216) + (_b * 65536) + (_c * 256) + _d ))
}

prefix_mask() {
	if [ "$1" -eq 0 ]; then echo 0; else echo $(( 4294967295 - (1 << (32 - $1)) + 1 )); fi
}

# Do two CIDRs share any address? Compare both networks under the SHORTER
# prefix: if they agree there, one contains the other.
cidrs_overlap() {
	_an=$(ip_to_int "$1"); _ap=$2; _bn=$(ip_to_int "$3"); _bp=$4
	if [ "$_ap" -lt "$_bp" ]; then _m=$(prefix_mask "$_ap"); else _m=$(prefix_mask "$_bp"); fi
	[ $(( _an & _m )) -eq $(( _bn & _m )) ]
}

require SIP_REGISTRATION_REGISTRAR
require SIP_REGISTRATION_USERNAME
require SIP_REGISTRATION_PASSWORD
require SIP_REGISTRATION_ALLOWED_ADDRESSES

REGISTRAR="$SIP_REGISTRATION_REGISTRAR"
USERNAME="$SIP_REGISTRATION_USERNAME"
PASSWORD="$SIP_REGISTRATION_PASSWORD"
REALM="${SIP_REGISTRATION_REALM:-$REGISTRAR}"
EXPIRES="${SIP_REGISTRATION_EXPIRES:-600}"
KEEPALIVE="${SIP_REGISTRATION_KEEPALIVE_SECONDS:-25}"
LISTEN_PORT="${SIP_REGISTRATION_LISTEN_PORT:-5062}"
TARGET="${SIP_REGISTRATION_TARGET:-127.0.0.1:5060}"
CONTACT_ADDRESS="${SIP_REGISTRATION_CONTACT_ADDRESS:-}"
LOG_LEVEL="${SIP_REGISTRATION_LOG_LEVEL:-2}"

case "$REGISTRAR" in
	*:*|*/*) die "SIP_REGISTRATION_REGISTRAR must be a bare host name, no scheme and no port (got: $REGISTRAR)" ;;
esac
case "$TARGET" in
	*:*) : ;;
	*) die "SIP_REGISTRATION_TARGET must be host:port (got: $TARGET)" ;;
esac
TARGET_HOST=${TARGET%:*}
TARGET_PORT=${TARGET##*:}
[ "$TARGET_HOST" = "localhost" ] && TARGET_HOST=127.0.0.1
# The target address is not merely a forwarding destination: it becomes the
# local-leg entry in the source allowlist, so it has to be an address we can
# compare a packet's source against, not a name resolved at some later time.
is_ipv4 "$TARGET_HOST" || die "SIP_REGISTRATION_TARGET host must be an IPv4 address (got: $TARGET_HOST); the default 127.0.0.1:5060 is almost always right"
case "$TARGET_PORT" in
	*[!0-9]*|"") die "SIP_REGISTRATION_TARGET port must be numeric (got: $TARGET)" ;;
esac
# db_text uses ':' as its field separator and these values are written into a
# db_text row, so a colon corrupts the table rather than failing cleanly.
case "$USERNAME$PASSWORD$REALM" in
	*:*) die "registration credentials must not contain ':'" ;;
esac
# Config placeholders are __OPX_...__-delimited precisely so a value can never
# be mistaken for one. Refuse the marker outright rather than rely on that
# remaining true forever.
case "$USERNAME$PASSWORD$REALM$REGISTRAR$TARGET$CONTACT_ADDRESS" in
	*__OPX_*) die "values must not contain the reserved substitution marker '__OPX_'" ;;
esac
# Every value below is interpolated into a Kamailio directive of the form
#   #!substdef "!TOKEN!value!g"
# so a '!' in the value terminates the directive early and Kamailio refuses to
# start -- AFTER provisioning has already switched the trunk. '"' and '\' break
# the same string. These are legal in a SIP user part, so this is a real
# restriction rather than a theoretical one; carriers generate alphanumeric
# credentials, and failing here beats failing after the trunk moved.
case "$USERNAME$PASSWORD$REALM$REGISTRAR$TARGET$CONTACT_ADDRESS" in
	*'!'*|*'"'*|*'\'*) die "registration values must not contain '!', '\"' or backslash: they terminate the Kamailio config directive these are substituted into" ;;
esac
# Control characters and newlines end the directive's line outright. Compare
# against a stripped copy rather than globbing for "$(printf '\n')" -- command
# substitution eats the trailing newline, leaving an empty pattern that matches
# every value. (It did, and this check rejected valid credentials until the
# test below caught it.)
_ctl_in="$USERNAME$PASSWORD$REALM"
_ctl_out=$(printf '%s' "$_ctl_in" | tr -d '[:cntrl:]')
[ "$_ctl_in" = "$_ctl_out" ] || die "registration credentials must not contain control characters"
# A keepalive slower than the shortest common NAT UDP timeout (30s) silently
# loses inbound calls between registration refreshes, which is the exact
# failure this sidecar exists to prevent. Refuse rather than half-work.
[ "$KEEPALIVE" -ge 5 ] 2>/dev/null || die "SIP_REGISTRATION_KEEPALIVE_SECONDS must be an integer >= 5"
[ "$KEEPALIVE" -le 28 ] || die "SIP_REGISTRATION_KEEPALIVE_SECONDS must be <= 28 (NAT UDP bindings commonly expire at 30s)"
case "$LISTEN_PORT" in
	*[!0-9]*|"") die "SIP_REGISTRATION_LISTEN_PORT must be a port number (got: $LISTEN_PORT)" ;;
esac
{ [ "$LISTEN_PORT" -ge 1 ] && [ "$LISTEN_PORT" -le 65535 ]; } || die "SIP_REGISTRATION_LISTEN_PORT out of range (got: $LISTEN_PORT)"

# SIP_REGISTRATION_CONTACT_ADDRESS feeds Kamailio's reg_contact_addr, which
# wants a bare host:port. A `sip:` prefix or a missing port produces a Contact
# the carrier cannot use, with no error at startup.
if [ -n "$CONTACT_ADDRESS" ]; then
	case "$CONTACT_ADDRESS" in
		sip:*|sips:*) die "SIP_REGISTRATION_CONTACT_ADDRESS must be bare host:port, with no sip: prefix (got: $CONTACT_ADDRESS)" ;;
		*:*:*) die "SIP_REGISTRATION_CONTACT_ADDRESS must be IPv4 host:port; IPv6 literals are not supported (got: $CONTACT_ADDRESS)" ;;
		*:*) : ;;
		*) die "SIP_REGISTRATION_CONTACT_ADDRESS must include a port, e.g. 203.0.113.10:$LISTEN_PORT (got: $CONTACT_ADDRESS)" ;;
	esac
	_ca_port=${CONTACT_ADDRESS##*:}
	case "$_ca_port" in
		*[!0-9]*|"") die "SIP_REGISTRATION_CONTACT_ADDRESS port must be numeric (got: $CONTACT_ADDRESS)" ;;
	esac
	# This is advertised TO THE CARRIER as where to send calls, so unlike
	# SIP_REGISTRATION_TARGET it has to be publicly reachable. A private or
	# loopback address here registers perfectly happily -- Contact is just a
	# header, the carrier accepts it -- the health check reports ONLINE, and no
	# call ever arrives. That is the exact silent failure this whole feature
	# exists to remove, and 192.168.x.y is a plausible guess, not an exotic one.
	_ca_host=${CONTACT_ADDRESS%:*}
	is_ipv4 "$_ca_host" || die "SIP_REGISTRATION_CONTACT_ADDRESS host must be a public IPv4 address (got: $_ca_host); it is advertised to the carrier as where to send your calls"
	for _res in $_RESERVED_NETS; do
		_rn=${_res%%/*}; _rp=${_res#*/}
		if cidrs_overlap "$_ca_host" 32 "$_rn" "$_rp"; then
			die "SIP_REGISTRATION_CONTACT_ADDRESS host '$_ca_host' is in reserved/private space ($_res). It is advertised to the carrier as the address to send calls to, so it must be your PUBLIC address. Leave this empty unless you have a static port-forward"
		fi
	done
fi

# Both are interpolated into the config; a non-integer makes Kamailio refuse to
# start, which on the provisioning path happens after the trunk has moved.
case "$EXPIRES" in
	*[!0-9]*|"") die "SIP_REGISTRATION_EXPIRES must be a positive integer of seconds (got: $EXPIRES)" ;;
esac
{ [ "$EXPIRES" -ge 30 ] && [ "$EXPIRES" -le 86400 ]; } || die "SIP_REGISTRATION_EXPIRES must be between 30 and 86400 seconds (got: $EXPIRES); carriers enforce their own minimum on top -- DIDWW answers 423 below 600"
case "$LOG_LEVEL" in
	-[0-9]|[0-9]) : ;;
	*) die "SIP_REGISTRATION_LOG_LEVEL must be a Kamailio debug level between -3 and 4 (got: $LOG_LEVEL)" ;;
esac
{ [ "$LOG_LEVEL" -ge -3 ] && [ "$LOG_LEVEL" -le 4 ]; } || die "SIP_REGISTRATION_LOG_LEVEL must be between -3 and 4 (got: $LOG_LEVEL)"

# Resolve the carrier and the interface that reaches it BEFORE anything is
# rendered: the local address is not only what Kamailio binds, it is also the
# local-leg entry in the source allowlist below.
# Binding a real interface address rather than a wildcard also keeps 0.0.0.0
# out of our own Via (which no registrar can answer) and keeps the proxy off
# every other interface of the host.
# A carrier may publish only _sip._udp SRV records and no A record on the
# registration domain -- which is legal, and Kamailio resolves the SRV itself
# when it sends the REGISTER. Only the interface choice needs an address here,
# so fall back to the interface that reaches the internet rather than refusing
# to start on a perfectly valid carrier.
registrar_ip=$(getent ahostsv4 "$REGISTRAR" 2>/dev/null | awk 'NR==1{print $1}')
if [ -n "$registrar_ip" ]; then
	local_ip_source="registrar-a-record"
	route_probe="$registrar_ip"
else
	local_ip_source="default-route (no A record for $REGISTRAR; SRV is resolved by the SIP stack)"
	route_probe="198.51.100.1"
fi
local_ip=$(ip route get "$route_probe" 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1)
[ -n "$local_ip" ] || die "cannot determine which local address reaches the network (no route to $route_probe)"

rm -rf "$RUN_DIR"
mkdir -p "$DB_DIR"
chmod 700 "$RUN_DIR" "$DB_DIR"

# --- db_text: table versions Kamailio validates at startup -------------------
cat > "$DB_DIR/version" <<EOF
id(int,auto) table_name(string) table_version(int)
1:uacreg:5
2:address:6
3:trusted:6
EOF

# The permissions module always initialises its `trusted` backend as well;
# ship it empty so only the `address` allowlist below decides who may call us.
cat > "$DB_DIR/trusted" <<EOF
id(int,auto) src_ip(string) proto(string) from_pattern(string,null) ruri_pattern(string,null) tag(string,null) priority(int)
EOF

# --- db_text: the (empty) registrant table -----------------------------------
# Kamailio only initialises its registrant when a database is configured, so
# one is configured -- but it stays EMPTY. The single registration record is
# pushed over the control socket after startup (see below), which keeps the
# carrier password out of every file on disk.
cat > "$DB_DIR/uacreg" <<EOF
id(int,auto) l_uuid(string) l_username(string) l_domain(string) r_username(string) r_domain(string) realm(string) auth_username(string) auth_password(string) auth_ha1(string) auth_proxy(string) expires(int) flags(int) reg_delay(int) contact_addr(string) socket(string)
EOF

# --- db_text: the carrier allowlist ------------------------------------------
# Each entry MUST be a validated IPv4 CIDR. Kamailio happily loads an
# unparseable entry as a domain-name record that allow_source_address() never
# consults: no startup error, and every call answered 403. IPv6 entries are
# rejected explicitly -- their colons collide with db_text's field separator,
# so they would corrupt the table rather than merely fail to match.
ALLOWLIST_ROWS=""
ALLOWLIST_COUNT=0
ALLOWLIST_SHOWN=""
for cidr in $(printf '%s' "$SIP_REGISTRATION_ALLOWED_ADDRESSES" | tr ',' ' '); do
	[ -n "$cidr" ] || continue
	case "$cidr" in
		*:*) die "SIP_REGISTRATION_ALLOWED_ADDRESSES entry '$cidr' looks like IPv6; only IPv4 CIDRs are supported" ;;
	esac
	addr=${cidr%%/*}
	if [ "$addr" = "$cidr" ]; then
		mask=32
	else
		mask=${cidr#*/}
	fi
	is_ipv4 "$addr" || die "SIP_REGISTRATION_ALLOWED_ADDRESSES entry '$cidr' is not an IPv4 address or CIDR (a hostname will NOT work here)"
	case "$mask" in
		*[!0-9]*|"") die "SIP_REGISTRATION_ALLOWED_ADDRESSES entry '$cidr' has a non-numeric prefix length" ;;
	esac
	[ "$mask" -le 32 ] || die "SIP_REGISTRATION_ALLOWED_ADDRESSES entry '$cidr' has a prefix length above /32"
	# A reserved or private range here is not a narrower allowlist, it is a
	# broken one -- and 0.0.0.0/0 matches every source that can reach the
	# socket, which on an endpoint-independent NAT is the internet.
	for _res in $_RESERVED_NETS; do
		_rn=${_res%%/*}; _rp=${_res#*/}
		if cidrs_overlap "$addr" "$mask" "$_rn" "$_rp"; then
			die "SIP_REGISTRATION_ALLOWED_ADDRESSES entry '$cidr' overlaps reserved/private space ($_res). Use your carrier's real public signalling ranges; 0.0.0.0/0 in particular would accept calls from anyone who can reach this socket"
		fi
	done
	ALLOWLIST_COUNT=$((ALLOWLIST_COUNT + 1))
	ALLOWLIST_SHOWN="$ALLOWLIST_SHOWN${ALLOWLIST_SHOWN:+,}$addr/$mask"
	ALLOWLIST_ROWS="$ALLOWLIST_ROWS$ALLOWLIST_COUNT:1:$addr:$mask:0:
"
done
[ "$ALLOWLIST_COUNT" -gt 0 ] || die "SIP_REGISTRATION_ALLOWED_ADDRESSES contained no usable entries"

# Group 2 is the LOCAL leg: livekit-sip talking back to us for in-dialog
# requests. It is pinned to the target's port as well as its address, so only
# traffic genuinely from that service counts as local. Without this group the
# source gate in kamailio.cfg would reject livekit-sip's own BYE.
LOCAL_ROWS="$((ALLOWLIST_COUNT + 1)):2:127.0.0.1:32:$TARGET_PORT:
"
LOCAL_ROWS="$LOCAL_ROWS$((ALLOWLIST_COUNT + 2)):2:$local_ip:32:$TARGET_PORT:
"
if [ "$TARGET_HOST" != "127.0.0.1" ] && [ "$TARGET_HOST" != "$local_ip" ]; then
	LOCAL_ROWS="$LOCAL_ROWS$((ALLOWLIST_COUNT + 3)):2:$TARGET_HOST:32:$TARGET_PORT:
"
fi
{
	echo 'id(int,auto) grp(int) ip_addr(string) mask(int) port(int) tag(string,null)'
	printf '%s' "$ALLOWLIST_ROWS"
	printf '%s' "$LOCAL_ROWS"
} > "$DB_DIR/address"

# --- config substitutions -----------------------------------------------------
# Kamailio's registrant demands a Contact address. When the operator has not
# declared a reachable one, advertise the local socket: the carrier's registrar
# records the source it actually observed (rport/received) and a registrar tier
# built for dynamic endpoints routes calls there, not to the Contact literal.
if [ -n "$CONTACT_ADDRESS" ]; then
	contact_addr_param="$CONTACT_ADDRESS"
else
	contact_addr_param="$local_ip:$LISTEN_PORT"
fi

cat > "$RUN_DIR/defines.cfg" <<EOF
#!substdef "!__OPX_LOGLEVEL__!$LOG_LEVEL!g"
#!substdef "!__OPX_LISTEN_PORT__!$LISTEN_PORT!g"
#!substdef "!__OPX_LOCAL_IP__!$local_ip!g"
#!substdef "!__OPX_TARGET__!$TARGET!g"
#!substdef "!__OPX_KEEPALIVE_CALLID__!nat-keepalive-$(hostname)!g"
#!substdef "!__OPX_KEEPALIVE_SECS__!$KEEPALIVE!g"
#!substdef "!__OPX_REGISTRAR__!$REGISTRAR!g"
#!substdef "!__OPX_REG_AOR__!$USERNAME@$REGISTRAR!g"
#!substdef "!__OPX_CONTACT_ADDR__!$contact_addr_param!g"
EOF

# Kamailio drops to the unprivileged `kamailio` user once its socket is bound
# (user=/group= in kamailio.cfg), so hand it the state it must read and the
# directory it creates its control socket in. This has to come AFTER every
# runtime file above is written, or the new files stay root-owned and the
# workers cannot read them. 0700 keeps the credentials unreadable to anything
# else in the container.
chown -R kamailio:kamailio "$RUN_DIR"

echo "REGISTRAR_STARTING {\"registrar\": \"$REGISTRAR\", \"user\": \"$USERNAME\", \"listen\": \"$local_ip:$LISTEN_PORT\", \"listen_chosen_by\": \"$local_ip_source\", \"target\": \"$TARGET\", \"expires\": $EXPIRES, \"keepalive_seconds\": $KEEPALIVE, \"carrier_allowlist\": \"$ALLOWLIST_SHOWN\", \"contact_address\": \"$contact_addr_param\", \"contact_address_source\": \"${CONTACT_ADDRESS:+declared}${CONTACT_ADDRESS:-local-socket}\"}"

# Kamailio treats the exit of ANY of its own children as fatal, so the helpers
# below must be siblings, not children -- hence this shell stays PID 1 and
# forwards signals rather than exec'ing Kamailio.
kamailio -DD -E -f /etc/openphonex-registrar/kamailio.cfg &
kamailio_pid=$!

# Push the single registration record once the control socket is up. Commands
# go in on stdin, so the carrier password never appears in the process table
# and is never written to a file. `.` is Kamailio's "leave unset" marker.
(
	i=0
	while [ "$i" -lt 60 ]; do
		if printf 'core.version\n' | kamcmd -s "unixs:$RUN_DIR/kamailio_ctl" >/dev/null 2>&1; then
			if printf 'uac.reg_add %s %s %s %s %s %s %s %s . sip:%s %s 0 0 . .\n' \
				"$USERNAME" "$USERNAME" "$REGISTRAR" "$USERNAME" "$REGISTRAR" "$REALM" \
				"$USERNAME" "$PASSWORD" "$REGISTRAR" "$EXPIRES" \
				| kamcmd -s "unixs:$RUN_DIR/kamailio_ctl" >/dev/null 2>&1; then
				echo "REGISTRAR_REGISTRATION_ARMED {\"registrar\": \"$REGISTRAR\", \"contact\": \"sip:$USERNAME@$contact_addr_param\"}"
			else
				echo "REGISTRAR_BOOTSTRAP_FAILED {\"error\": \"uac.reg_add was rejected\"}" >&2
				kill -TERM "$kamailio_pid" 2>/dev/null
			fi
			exit 0
		fi
		i=$((i + 1))
		sleep 0.5
	done
	echo "REGISTRAR_BOOTSTRAP_FAILED {\"error\": \"control socket never became ready\"}" >&2
	kill -TERM "$kamailio_pid" 2>/dev/null
) &

# Kamailio binds one address for the life of the process. On a dynamic-IP host
# -- the entire point of this sidecar -- a DHCP renumber leaves it bound to an
# address that no longer exists: the registration lapses, calls stop arriving,
# and nothing crashes, so the container's restart policy never fires. Watch for
# it and exit so the container restarts onto the new address.
(
	while sleep 30; do
		kill -0 "$kamailio_pid" 2>/dev/null || exit 0
		now_ip=$(ip route get "$route_probe" 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1)
		if [ -n "$now_ip" ] && [ "$now_ip" != "$local_ip" ]; then
			echo "REGISTRAR_ADDRESS_CHANGED {\"was\": \"$local_ip\", \"now\": \"$now_ip\", \"action\": \"restarting to rebind\"}" >&2
			kill -TERM "$kamailio_pid" 2>/dev/null
			exit 0
		fi
	done
) &

trap 'kill -TERM "$kamailio_pid" 2>/dev/null' INT TERM
wait "$kamailio_pid"
status=$?
echo "REGISTRAR_EXITED {\"status\": $status}"
exit "$status"

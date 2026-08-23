#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
#
# Is the sidecar currently HOLDING a registration with the carrier?
#
# Exit 0 = yes. Anything else = no, and inbound calls are not arriving.
#
# The obvious check -- "timer_expires is non-zero" -- is wrong, and wrong in the
# dangerous direction. Kamailio reuses that field as the RETRY deadline: on a
# failed registration it sets `timer_expires = now + reg_retry_interval`
# (uac_reg.c, the `error:` path), so a box whose credentials the carrier is
# rejecting shows a healthy-looking future timestamp forever. Measured against a
# registrar that accepts and then starts rejecting:
#
#   never registered   flags: 0   timer_expires: 0
#   registered         flags: 20  timer_expires: <future>     <- ONLINE bit set
#   rejected/lapsed    flags: 16  timer_expires: <future>     <- ONLINE cleared
#
# Only the ONLINE bit distinguishes the last two. This uses Kamailio's own
# definition of a live registration (uac_reg.c: `(flags & UAC_REG_ONLINE) &&
# timer_expires > now`) -- both halves, because there is also a window where
# ONLINE is still set but the registration has already expired.
set -u

CTL="unixs:${REGISTRAR_RUN_DIR:-/run/openphonex-registrar}/kamailio_ctl"
UAC_REG_ONLINE=4   # 1<<2 in uac_reg.c

dump=$(kamcmd -s "$CTL" uac.reg_dump 2>/dev/null) || exit 1
[ -n "$dump" ] || exit 1

printf '%s\n' "$dump" | awk -v now="$(date +%s)" -v online="$UAC_REG_ONLINE" '
	$1 == "flags:"         { f = $2 + 0 }
	$1 == "timer_expires:" { t = $2 + 0 }
	END {
		if (f == "" || t == "") exit 1
		# bitwise AND without depending on a gawk-only and()
		if (int(f / online) % 2 != 1) exit 1
		if (t <= now) exit 1
		exit 0
	}
'

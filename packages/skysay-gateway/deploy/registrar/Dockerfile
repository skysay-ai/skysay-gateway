# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Unigox OÜ
#
# Skysay Gateway — registering SIP sidecar image.
#
# Kamailio is a SIP PROXY, not a back-to-back user agent: it forwards the
# carrier's INVITE to livekit-sip and stays out of the media path entirely, so
# RTP continues to flow directly between the carrier and livekit-sip exactly as
# it does in the static-endpoint topology. Nothing here terminates or relays
# audio.
#
# Ubuntu 24.04 is pinned because it ships Kamailio 5.7 — the version this
# sidecar is verified against. It needs three things that are not universally
# present in older builds: the `uac` registrant with its `uac.reg_add` control
# command, `add_local_rport()`, and the `rtimer` timer routes.
FROM ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

# Everything this sidecar loads — uac (outbound registration), rtimer (the NAT
# keepalive), permissions (the carrier source allowlist), db_text, ctl — ships
# in the main `kamailio` package on noble. The kamailio-extra-modules and
# kamailio-utils-modules packages carry unrelated things (acc_json, evapi,
# http_client, ...) and are deliberately NOT installed.
# iproute2 resolves which local interface reaches the carrier.
RUN set -eux; \
	export DEBIAN_FRONTEND=noninteractive; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		kamailio \
		iproute2 \
		ca-certificates; \
	apt-get purge -y --auto-remove; \
	rm -rf /var/lib/apt/lists/*; \
	systemctl disable kamailio 2>/dev/null || true; \
	rm -rf /etc/kamailio

COPY kamailio.cfg /etc/skysay-registrar/kamailio.cfg
COPY entrypoint.sh /usr/local/bin/skysay-registrar-entrypoint.sh
# Single source of truth for "is a registration actually held" -- the compose
# healthcheck and the smoke test both call this, so the two cannot drift.
COPY healthcheck.sh /usr/local/bin/skysay-registrar-health
RUN chmod 0755 /usr/local/bin/skysay-registrar-entrypoint.sh \
	/usr/local/bin/skysay-registrar-health

# The entrypoint starts as root because the sidecar may bind a privileged port
# on the host network, then Kamailio drops to the unprivileged `kamailio` user
# (user=/group= in kamailio.cfg) once the socket is bound. Only PID 1 -- the
# supervisor shell that forwards signals -- stays root; every worker that
# parses carrier-controlled SIP runs unprivileged. Verify with:
#   docker compose exec registrar ps -eo pid,user,comm
ENTRYPOINT ["/usr/local/bin/skysay-registrar-entrypoint.sh"]

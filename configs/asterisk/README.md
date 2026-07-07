# Asterisk configuration templates

> Supersedes the earlier "Asterisk Extraction Notes" placeholder. As of M6
> (SCH-202) the extraction is done: these are the templated, secret-free configs.

These files are **configuration data authored by OpenPhonex**, licensed
Apache-2.0. They are **not** Asterisk source code and do **not** link against
Asterisk. The OpenPhonex gateway worker communicates with Asterisk exclusively
over the ARI HTTP/WebSocket network API — an arms-length network boundary, not a
derivative work. Asterisk itself is GPLv2 and is obtained by the operator
directly from upstream (see `Dockerfile.asterisk`, which `apt-get install`s
Asterisk from the distribution repository on the operator's own machine). We
redistribute zero GPL-licensed code. See the repository `NOTICE` and
`THIRD_PARTY_LICENSES.md` for the full licensing posture.

These files are derived from the working prototype at
`state/didww/asterisk-duplex/etc-default/`, with **every** secret replaced by a
`${ENV}` reference: SIP trunk username/password, the public/NAT IP, the provider
host, and the ARI password. No live `pjsip.conf`, `ari.conf`, `.env`, or other
secret-bearing file is ever copied into this repo.

## Files

| File                   | Purpose                                                        |
|------------------------|----------------------------------------------------------------|
| `pjsip.conf.template`  | SIP trunk (inbound registration + endpoint + outbound). Rendered with `${SIP_TRUNK_USERNAME}`, `${SIP_TRUNK_PASSWORD}`, `${SIP_PROVIDER_HOST}`, `${PUBLIC_IP}`. |
| `ari.conf.template`    | ARI user for the gateway worker. Rendered with `${ARI_USER}`, `${ARI_PASSWORD}`. |
| `extensions.conf`      | Dialplan. Inbound → `Stasis(voicebot,inbound)`. No secrets.     |
| `http.conf`            | ARI HTTP transport (port 8088, private network only).          |
| `rtp.conf`             | RTP media port range (must match the compose udp range).       |

## No secrets, ever

Every credential and public address in the `*.template` files is a `${ENV}`
reference resolved at render/run time from `.env.gateway`. Nothing in this
directory contains a live SIP username, password, public IP, or ARI password.
The release secret-scanner (`scripts/oss-release/scan_secrets.py`) enforces this
before any export can be published.

## Rendering

The topology-B quickstart mounts this directory read-only into the `asterisk`
container. The `${...}` references are substituted from the environment. For a
one-shot local render you can use `envsubst`:

```bash
set -a; . ./.env.gateway; set +a
envsubst < configs/asterisk/pjsip.conf.template > /etc/asterisk/pjsip.conf
envsubst < configs/asterisk/ari.conf.template   > /etc/asterisk/ari.conf
```

`ASTERISK_ARI_URL` must always point at the **internal** compose address
(`http://asterisk:8088/ari`); the ARI port is never published to the host.

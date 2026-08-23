# INSTALL — self-hosted OpenPhonex LiveKit stack

This is the complete, linear install runbook for the **recommended** self-host
stack: the same media plane OpenPhonex production runs. It is written to be
executable step by step **by a person or an AI agent** — every step has a
command, an expected result, and a failure pointer. No step requires
information that is not defined here or in the referenced file.

Background reading (optional):
[`packages/openphonex-gateway/deploy/SELFHOST.md`](packages/openphonex-gateway/deploy/SELFHOST.md).

---

## Step 0 — Prerequisites (collect these before touching the host)

| # | You need | How you know you have it |
|---|---|---|
| 0.1 | A Linux host with root/sudo. On a Mac, Docker Desktop **cannot** run this stack — it has no host networking, so the containers bind their ports and stay unreachable even from `localhost`; run it inside a Colima VM instead (`colima start --network-address --network-mode bridged`) and treat that VM as the host for every step below. A **public IPv4** your carrier can send calls to is the simple case; if you do not have one, see "Choosing your inbound topology" below | `curl -4 ifconfig.me` prints an IP — and if that IP is not actually bound to this host, you need the registration topology |
| 0.2 | Docker Engine + **Compose v2** on that host | `docker compose version` prints a version |
| 0.3 | A **DID** (phone number) on a SIP trunk that sends calls to your host, plus your carrier's **SIP signaling IP ranges** AND its **RTP / media IP ranges** (from the carrier portal — these are often DIFFERENT range sets) | e.g. DID `+15551230000`, signaling `203.0.113.0/24`, media `203.0.113.128/25` |
| 0.4 | An **OpenPhonex control plane** URL | OpenPhonex Cloud: `https://api.openphonex.com`, or your own deployment |
| 0.5 | From that control plane: **organization id**, registered **gateway deployment id**, two **different** credentials (`tgw_` gateway token and `tai_` worker key), and an **agent + phone-number route** configured for your DID | **API Keys → Gateway Tokens** creates the deployment-bound `tgw_`; the separate regular **API Keys** card creates the organization-scoped `tai_` worker key |
| 0.6 | Provider API keys: **Deepgram**, **Gemini**, **ElevenLabs** | from console.deepgram.com / aistudio.google.com / elevenlabs.io |
| 0.7 | Host firewall able to open `5060/udp` (to the SIP **signaling** ranges) and `10000-10100/udp` (to the RTP **media** ranges) separately. In the registration topology you DROP inbound `5060/udp` instead and forward nothing — see step 4 | cloud-firewall or `ufw`/`nftables` access |

### Choosing your inbound topology

There are two ways a call reaches this stack, and you must pick one before you
start. They differ only in how the carrier's INVITE arrives; the media path,
the agent, and the control-plane contract are identical.

| | **Static endpoint** (default) | **Registration** (`--profile registrar`) |
|---|---|---|
| Your host | has a public IP the carrier can reach | has **no** reachable public address: home/office NAT, dynamic IP, CGNAT |
| Carrier trunk | points at your IP | set to SIP-registration / dynamic-endpoint mode |
| Who starts the conversation | the carrier, inbound | **you**, by REGISTERing outbound |
| Extra services | none | one: the `registrar` sidecar |
| Extra variables | none | the `SIP_REGISTRATION_*` block in step 3 |
| Media | carrier ⇄ livekit-sip directly | carrier ⇄ livekit-sip directly (**the sidecar carries signalling only**) |

Pick **static endpoint** if you have a public IP — it has fewer moving parts.
Pick **registration** only if you genuinely cannot receive an unsolicited
inbound packet. Registration mode has one hard requirement on the carrier side:
the trunk must latch its outbound RTP onto the address it actually receives
media from (carriers call this *symmetric RTP*), because the port this stack
advertises is rewritten by your NAT on the way out. If your carrier cannot do
that, registration mode will complete the call and then have no audio.

Everything below applies to both topologies unless a step says otherwise.

## Step 1 — Get the code

```bash
# Obtain the preview gateway source package from OpenPhonex first.
cd openphonex-gateway/packages/openphonex-gateway/deploy
```

**Expect:** you are in a directory containing `docker-compose.livekit.yml`,
`livekit.yaml.example`, `sip.yaml.example`, `.env.livekit.example`, and
`selfhost-smoketest.sh`.

## Step 2 — Create the four config files

```bash
cp livekit.yaml.example livekit.yaml
cp sip.yaml.example sip.yaml
cp .env.livekit.example .env.livekit
cp .env.registrar.example .env.registrar   # sidecar credentials; leave blank for a public-IP host
chmod 600 livekit.yaml sip.yaml .env.livekit
mkdir -p sessions call-evidence
sudo chown -R 10001:10001 sessions call-evidence
```

Generate the LiveKit API key pair (you invent it; it is local to this stack):

```bash
echo "key:    opx-selfhost"
echo "secret: $(openssl rand -hex 32)"
```

Put the **same pair** in all three files:

- `livekit.yaml` → under `keys:` as `opx-selfhost: <secret>` (replace the
  `changeme-*` line)
- `sip.yaml` → `api_key: opx-selfhost` and `api_secret: <secret>`
- `.env.livekit` → `LIVEKIT_API_KEY=opx-selfhost`,
  `LIVEKIT_API_SECRET=<secret>`

**Never commit the filled-in files** (the repo `.gitignore` already excludes
them).

## Step 3 — Fill in every variable in `.env.livekit`

Each variable is also documented inline in the file. Complete reference:

### Create the two OpenPhonex credentials first — separately

After registering the gateway deployment, open the workspace **API Keys**
page. These credentials are intentionally different and cannot stand in for
one another:

1. Use the **Gateway Tokens** card to create a deployment-bound `tgw_` token.
   Its one-time reveal includes the exact line below; replace the blank
   `OPENPHONEX_GATEWAY_API_TOKEN=` assignment in `.env.livekit` with the whole
   revealed line.

   ```dotenv
   OPENPHONEX_GATEWAY_API_TOKEN=tgw_...
   ```

2. Use the separate regular **API Keys** card and click **Create self-host
   worker key**. It creates the organization-scoped `tai_` value for
   `OPENPHONEX_WORKER_API_TOKEN` with the required inbound-routing scope
   `calls:read` + `calls:write`.

   ```dotenv
   OPENPHONEX_WORKER_API_TOKEN=tai_...
   ```

Both secrets are shown once only and are absent from list views. A `tgw_` token
is bound to one gateway and cannot authenticate the worker; a `tai_` worker
key must never be reused as a gateway token. The gateway smoke check fails
closed for a wrong, revoked, expired, cross-organization, or mismatched
gateway-id/token pair; mint a replacement from that registered gateway's
**Gateway Tokens** card.

`OPENPHONEX_ORGANIZATION_ID` must name the workspace that minted the `tai_`
key and owns the DID. A worker key from another workspace is rejected before
the control plane returns the inbound call's runtime configuration.

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `LIVEKIT_URL` | ws URL of this stack's LiveKit. Provisioning requires `wss://` or `ws://` on loopback (a remote `ws://` is refused — it would leak the admin JWT) | default for the host-local stack | `ws://127.0.0.1:7880` |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | the pair from step 2 | you generated it | `opx-selfhost` / 64 hex chars |
| `OPENPHONEX_CONTROL_PLANE_URL` | control-plane base URL. **Must be `https://`** (the agent refuses plaintext http except to loopback) | step 0.4 | `https://api.openphonex.com` |
| `OPENPHONEX_GATEWAY_ID` | your gateway deployment id | `POST /v1/gateway-deployments` (or dashboard → Gateways → new self-hosted gateway) | `gw_01HZX...` |
| `OPENPHONEX_GATEWAY_API_TOKEN` | deployment-scoped `tgw_` token for its matching gateway's number-rules, lifecycle, and artifact operations | **API Keys → Gateway Tokens** after registering that gateway — revealed **once** as `OPENPHONEX_GATEWAY_API_TOKEN=tgw_...` | `tgw_...` |
| `OPENPHONEX_WORKER_API_TOKEN` | organization-scoped `tai_` worker key (inbound routing, its call-bound runtime-config read, and MCP `tools/list` schema lookup; the self-host baseline is `calls:read` + `calls:write`) | separate regular **API Keys** card → **Create self-host worker key** — revealed **once** | `tai_...` |
| `OPENPHONEX_ORGANIZATION_ID` | the org that owns your DID | dashboard → organization settings | `org_01HZX...` |
| `SCH236_TEST_DID_E164` | the DID this agent answers, **with** `+` | your carrier + registered in OpenPhonex | `+15551230000` |
| `DEEPGRAM_API_KEY` | STT provider key | console.deepgram.com → API Keys | `...` |
| `GEMINI_API_KEY` | LLM provider key | aistudio.google.com → Get API key | `...` |
| `ELEVENLABS_API_KEY` | TTS provider key | elevenlabs.io → Profile → API Keys | `...` |
| `SCH236_NUM_IDLE_PROCESSES` | warm idle agent processes (optional) | default is fine | `2` |
| `DIDWW_INBOUND_ADDRESSES` | carrier signaling CIDRs (inbound allowlist for provisioning). **Public** ranges only — the example below is a documentation range and is rejected on purpose | carrier portal → signaling IPs | `203.0.113.0/24,198.51.100.0/28` (replace with your real ranges) |

The `provision` step uses `SCH236_TEST_DID_E164` (above) for the DID — you do
NOT set the DID twice.

**Optional — outbound trunk** (only to place calls OUT; inbound answering does
not need it). Supply **all four** or **none**:

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `DIDWW_OUTBOUND_HOST` | carrier outbound SIP host | carrier portal → SIP trunk | `sip.yourcarrier.com` |
| `DIDWW_OUTBOUND_FROM_USER` | one or more allowed outbound From-users, comma-separated, digits only | your caller-ID DID(s) | `15551230000,37255501234` |
| `DIDWW_OUTBOUND_AUTH_USERNAME` / `DIDWW_OUTBOUND_AUTH_PASSWORD` | outbound trunk auth pair | carrier portal → SIP trunk credentials | `...` |
| `OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND` | second fail-closed execution gate; leave `0` until provisioning + smoke tests pass | you | `0`, then `1` at cutover |
| `OPENPHONEX_OUTBOUND_WORKER_ID` | stable identity recorded when claiming jobs | choose a unique name per worker | `livekit-outbound-1` |
| `OPENPHONEX_OUTBOUND_TRUNK_NAME` | LiveKit outbound trunk the worker must match | provisioner default | `openphonex-selfhost-outbound` |
| `OPENPHONEX_OUTBOUND_POLL_SECONDS` | queue poll interval | default is fine | `2` |
| `OPENPHONEX_OUTBOUND_RINGING_TIMEOUT_SECONDS` | stop unanswered ringing after this many seconds (5–120) | your operating policy | `45` |
| `OPENPHONEX_OUTBOUND_MAX_CALL_DURATION_SECONDS` | hard LiveKit SIP call cap (30–14400) | your operating policy | `1800` |

**Optional — Sinhala (si-LK) voice stack** (only needed for an agent
configured with `language: si`; neither Deepgram nor ElevenLabs supports
Sinhala, so a `si` agent resolves to Google Chirp 2 → Gemini → Gemini-TTS
instead). Set exactly **one** of the two GCP credential variables:

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | path to a GCP service-account JSON key file (Cloud Speech-to-Text v2, role `roles/speech.client`) — **not** the Gemini API key above | GCP Console → IAM & Admin → Service Accounts → Keys → Add key (JSON) | `/run/secrets/gcp-speech-sa.json` |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | the same service-account JSON, inlined as one env var instead of a file | same service account as above | `{"type":"service_account",...}` |
| `GOOGLE_STT_LOCATION` | Cloud Speech-to-Text v2 recognizer region (optional; default `asia-southeast1`, empirically verified for si-LK Chirp 2 — Chirp 3 rejects si-LK with an HTTP 400 in both `eu` and `us`) | you | `asia-southeast1` |

**Optional — registering SIP sidecar** (registration topology only; skip this
entire section for a public-IP host). These live in their **own file**,
`.env.registrar`, created blank in step 2 — never in `.env.livekit`.

That separation is deliberate and is not worth undoing. Docker's `env_file`
passes the whole file to the container — it cannot pass part of one — and the
sidecar is the most exposed process in the stack: host-networked, parsing SIP
from whatever reaches the socket. Sharing `.env.livekit` with it would put your
control-plane tokens and provider API keys in that process's environment, so a
bug in a SIP parser would become a full credential compromise.

Start the stack with `--profile registrar` once they are set. The first four are
**all-or-none** —
provisioning refuses a partially filled block rather than guessing, because
half a registration block would retire the trunk's address allowlist while the
sidecar that replaces it still cannot start.

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `SIP_REGISTRATION_REGISTRAR` | carrier host to REGISTER to — bare hostname, no scheme, no port (SRV records are followed). Registration mode is selected when all four of these are present, not by this one alone | your carrier's registration / dynamic-endpoint docs | `sip.yourcarrier.com` |
| `SIP_REGISTRATION_USERNAME` | REGISTER user. **Not** your portal login and **not** the outbound trunk auth — carriers usually generate a dedicated pair when a trunk is switched to registration mode | the carrier's page for that trunk | `6d6CfMIqC4fatRtj` |
| `SIP_REGISTRATION_PASSWORD` | REGISTER password, issued with the username above | same page | (generated) |
| `SIP_REGISTRATION_REALM` | digest realm the carrier challenges with | defaults to `SIP_REGISTRATION_REGISTRAR`, which is right for most carriers | `sip.yourcarrier.com` |
| `SIP_REGISTRATION_ALLOWED_ADDRESSES` | public **IPv4** CIDRs the carrier may deliver calls FROM. The sidecar rejects INVITEs from anywhere else. In registration mode a carrier often delivers from its **registrar** tier rather than its published static-signalling nodes, so use a range covering both. Hostnames, IPv6 ranges, and anything overlapping private/CGNAT/documentation/reserved space are rejected at startup — `0.0.0.0/0` included, since it removes the boundary rather than narrowing it. This guards the sidecar's port only — it does **not** replace the loopback-only firewall rule on `5060/udp` in step 4 | carrier IP documentation | `203.0.113.0/24,198.51.100.0/28` |
| `SIP_REGISTRATION_KEEPALIVE_SECONDS` | how often to poke the carrier to keep the NAT binding alive. **Not** the registration interval — carriers enforce a long minimum registration expiry (often 600s) while NAT UDP bindings expire in 30–120s, so without this, calls silently stop arriving while the registration still looks healthy | default is fine; must be 5–28 | `25` |
| `SIP_REGISTRATION_EXPIRES` | requested registration lifetime in seconds; the carrier may raise it and answer `423` with its own minimum | default is fine | `600` |
| `SIP_REGISTRATION_LISTEN_PORT` | UDP port the sidecar listens on. The default keeps livekit-sip on its normal 5060 so `sip.yaml` needs no change, and a non-5060 port also side-steps most consumer-router SIP ALGs | default is fine | `5062` |
| `SIP_REGISTRATION_TARGET` | where the sidecar forwards calls — livekit-sip on loopback. Must be an IPv4 address and port, not a hostname: it doubles as the local-leg entry in the sidecar's source allowlist | default is fine | `127.0.0.1:5060` |
| `SIP_REGISTRATION_CONTACT_ADDRESS` | publicly reachable `host:port` to advertise, for the minority of setups where you know it (a static port-forward or 1:1 NAT). **Leave empty behind an ordinary NAT** — empty is the correct value, not a fallback, because only the carrier can observe the address it should use | you, only if you have a port-forward | *(empty)* |
| `SIP_REGISTRATION_LOG_LEVEL` | sidecar verbosity | default is fine; raise to `3` to trace individual SIP messages | `2` |

In registration mode `DIDWW_INBOUND_ADDRESSES` is **not required and not used**:
every INVITE reaches livekit-sip from this host via the sidecar, so a carrier
address allowlist on the trunk would reject every real call. The sidecar's
`SIP_REGISTRATION_ALLOWED_ADDRESSES` is the carrier filter instead, applied one
hop earlier.

**Verify** (all required variables non-empty):

```bash
bash selfhost-smoketest.sh || true   # the config/variable checks at the top must all PASS
```

## Step 4 — Open the firewall (to your carrier only)

Open the two port ranges to the **matching** carrier ranges — signaling and
media are often separate range sets, so scope each independently:

- `5060/udp` (SIP signaling) → to the carrier's **signaling** IP ranges.
- `10000-10100/udp` (RTP media) → to the carrier's **RTP / media** IP ranges.

Do not open them to the world; do not open anything else — LiveKit's API (7880)
and Redis (6380) bind loopback-only. Note `DIDWW_INBOUND_ADDRESSES` (used by
provisioning) is the **signaling** allowlist — the RTP ranges only need the
firewall opening, not a provisioning entry.

**In registration mode, replace the inbound `5060/udp` rule with a DROP — on
IPv4 AND IPv6.** Nothing needs to reach that port unsolicited: the sidecar's
own outbound REGISTER opens the path, and the return traffic is
already-established from a stateful firewall's point of view. The RTP range is
likewise admitted automatically once livekit-sip has sent its first packet.

```bash
sudo iptables  -A INPUT ! -i lo -p udp --dport 5060 -m conntrack --ctstate NEW -j DROP
sudo ip6tables -A INPUT ! -i lo -p udp --dport 5060 -m conntrack --ctstate NEW -j DROP
```

`--ctstate NEW` is required, not decoration: `livekit-sip` sends outbound
signalling from the same socket it listens on, so your carrier's replies also
arrive with destination port 5060. A bare `--dport 5060 -j DROP` would drop
them and break outbound calling silently. Scoping to `NEW` drops only
unsolicited traffic. (UDP is conntrack-tracked, so a reply to something you
sent is `ESTABLISHED`, not `NEW`.)

This is **required, not hardening**. In registration mode the LiveKit trunk
carries no address allowlist (it cannot — every INVITE now arrives from this
host), so the host firewall is what stands in for it. And your NAT only covers
IPv4: a dual-stack host usually also has a **global IPv6 address with no NAT in
front of it**, which would expose `5060/udp` to the internet directly, past the
sidecar and its carrier allowlist. Anyone who reaches it gets an answered call
into your agent — unauthenticated voice access to the LLM and metered spend on
your provider keys. Check with `ss -lunp | grep 5060`.

### Upgrading an existing deployment

`SIP_REGISTRATION_*` used to live in `.env.livekit`. Every deployment now needs
`.env.registrar`, and those variables must appear **only** there:

```bash
cd packages/openphonex-gateway/deploy
cp .env.registrar.example .env.registrar && chmod 600 .env.registrar
# move any SIP_REGISTRATION_* values across, then delete the WHOLE block from
# .env.livekit — including empty placeholder lines
docker compose -f docker-compose.livekit.yml --profile provision run --rm provision
```

**This applies to public-IP deployments too**, and they are the ones it catches
out. The old template shipped empty `SIP_REGISTRATION_REGISTRAR=` placeholders,
which sit in `.env.livekit` doing nothing — until the host moves behind NAT.
The provision container receives `.env.registrar` as environment and then
sources `.env.livekit` over the top, so those empty lines win, the provisioner
sees an empty block, and it quietly builds the static-endpoint trunk. Every
INVITE the sidecar forwards is then rejected by that trunk's carrier allowlist:
inbound is dead while everything reports healthy.

`bash selfhost-smoketest.sh` fails if it finds `SIP_REGISTRATION_*` in
`.env.livekit`, so run it after upgrading. And **re-run the provision step** as
above — that is the half with no error message of its own.

## Step 5 — Start the stack

```bash
docker compose -f docker-compose.livekit.yml up -d --build
# registration topology only — also start the sidecar:
docker compose -f docker-compose.livekit.yml --profile registrar up -d --build registrar
```

**Expect:** the first build takes a few minutes (pip install + model
pre-download), then five services are up:

```bash
docker compose -f docker-compose.livekit.yml ps --services --status running
# -> agents, livekit, reconciler, redis, sip (order may vary)
```

## Step 6 — Provision the SIP trunk + dispatch rule (one-shot)

```bash
docker compose -f docker-compose.livekit.yml --profile provision run --rm provision
```

**Expect:** a final `SELFHOST_PROVISIONED {...}` line (with the
`dispatch_rule_id`, `inbound_trunk_id`, and `outbound_trunk_id` —
`not_created` if you left the outbound vars blank).
**If it prints `SELFHOST_PROVISION_FAILED {...}`:** the JSON `error` field is
the exact reason (e.g. a missing/malformed variable, or a non-public
`DIDWW_INBOUND_ADDRESSES` entry). Fix that and rerun.

The guided self-host worker key is inbound-only. It deliberately omits the
global `worker_jobs:*` authority, and the compose disables its non-critical
global worker-heartbeat loop. Do not start the optional `outbound-worker` profile
with this baseline credential; keep `OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND=0`
unless OpenPhonex has issued a separately reviewed outbound-worker credential.

## Step 7 — Verify everything (the smoke test)

```bash
bash selfhost-smoketest.sh
```

`RESULT: UNVERIFIED` (exit 2) means every check that could run passed but at
least one could not be performed at all — usually the firewall check, which
needs root. Treat it as not-yet-ready rather than a pass.

**Expect:** `RESULT: PASS` and a per-check `[PASS]` list covering: services
running, redis PONG, livekit-server answering, **agent registered with
LiveKit**, the matching deployment-bound **gateway `tgw_` token** accepted by
the non-mutating number-rules read, the organization-scoped **worker `tai_`
key** accepted by MCP `tools/list`, artifact dirs writable. Every `[FAIL]`
line prints the exact fix; apply it and re-run until PASS.

## Step 8 — Place a test call

Call your DID from any phone.

**Expect:** the agent answers with its configured first message, and

```bash
docker compose -f docker-compose.livekit.yml logs -f agents
```

streams `media_admitted` → per-turn `turn_telemetry` → `call_complete`.
Artifacts land in `call-evidence/<call_id>/`, including the bounded,
versioned `call-telemetry.json` written after media ends; the control plane
receives the lifecycle/billing events and the uploaded recording, transcript,
and telemetry evidence.

**Done.** The stack self-recovers (`restart: unless-stopped`; the
`reconciler` closes sessions of a crashed agent so no dangling billed
minutes).

## Troubleshooting

The smoke test localizes almost everything. For the full symptom → fix table
(agent restart loops, provisioning interlocks, 401/403 token confusion,
one-way audio, slow first answer, EACCES on artifact dirs) see
[`SELFHOST.md` → Troubleshooting (LiveKit)](packages/openphonex-gateway/deploy/SELFHOST.md#troubleshooting-livekit).

## Updating

```bash
git pull
docker compose -f docker-compose.livekit.yml up -d --build
bash selfhost-smoketest.sh
```

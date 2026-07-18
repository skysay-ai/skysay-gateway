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
| 0.1 | A Linux host with a **public IPv4** and root/sudo | `curl -4 ifconfig.me` prints the IP |
| 0.2 | Docker Engine + **Compose v2** on that host | `docker compose version` prints a version |
| 0.3 | A **DID** (phone number) on a SIP trunk that sends calls to your host, plus your carrier's **SIP signaling IP ranges** AND its **RTP / media IP ranges** (from the carrier portal — these are often DIFFERENT range sets) | e.g. DID `+15551230000`, signaling `203.0.113.0/24`, media `203.0.113.128/25` |
| 0.4 | An **OpenPhonex control plane** URL | OpenPhonex Cloud: `https://api.openphonex.com`, or your own deployment |
| 0.5 | From that control plane: **organization id**, **gateway deployment id**, **per-gateway token**, **worker API token**, and an **agent + phone-number route** configured for your DID | see the variable table in step 3 for how to mint each |
| 0.6 | Provider API keys: **Deepgram**, **Gemini**, **ElevenLabs** | from console.deepgram.com / aistudio.google.com / elevenlabs.io |
| 0.7 | Host firewall able to open `5060/udp` (to the SIP **signaling** ranges) and `10000-10100/udp` (to the RTP **media** ranges) separately | cloud-firewall or `ufw`/`nftables` access |

## Step 1 — Get the code

```bash
# If you are installing from a fork, replace `skyzer` with your own org/user.
git clone https://github.com/skyzer/openphonex-gateway.git openphonex-gateway
cd openphonex-gateway/packages/openphonex-gateway/deploy
```

**Expect:** you are in a directory containing `docker-compose.livekit.yml`,
`livekit.yaml.example`, `sip.yaml.example`, `.env.livekit.example`, and
`selfhost-smoketest.sh`.

## Step 2 — Create the three config files

```bash
cp livekit.yaml.example livekit.yaml
cp sip.yaml.example sip.yaml
cp .env.livekit.example .env.livekit
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

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `LIVEKIT_URL` | ws URL of this stack's LiveKit. Provisioning requires `wss://` or `ws://` on loopback (a remote `ws://` is refused — it would leak the admin JWT) | default for the host-local stack | `ws://127.0.0.1:7880` |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | the pair from step 2 | you generated it | `opx-selfhost` / 64 hex chars |
| `OPENPHONEX_CONTROL_PLANE_URL` | control-plane base URL. **Must be `https://`** (the agent refuses plaintext http except to loopback) | step 0.4 | `https://api.openphonex.com` |
| `OPENPHONEX_GATEWAY_ID` | your gateway deployment id | `POST /v1/gateway-deployments` (or dashboard → Gateways → new self-hosted gateway) | `gw_01HZX...` |
| `OPENPHONEX_GATEWAY_API_TOKEN` | per-gateway scoped token | `POST /v1/gateway-deployments/{id}/tokens` — shown **once** at mint | `tgw_...` |
| `OPENPHONEX_WORKER_API_TOKEN` | **organization-scoped** worker token (inbound routing + MCP tools). Protect it; see `SECURITY.md` | dashboard → organization API keys | `tai_...` |
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
| `DIDWW_OUTBOUND_FROM_USER` | outbound From-user, digits only | usually your DID | `15551230000` |
| `DIDWW_OUTBOUND_AUTH_USERNAME` / `DIDWW_OUTBOUND_AUTH_PASSWORD` | outbound trunk auth pair | carrier portal → SIP trunk credentials | `...` |

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

## Step 5 — Start the stack

```bash
docker compose -f docker-compose.livekit.yml up -d --build
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

## Step 7 — Verify everything (the smoke test)

```bash
bash selfhost-smoketest.sh
```

**Expect:** `RESULT: PASS` and a per-check `[PASS]` list covering: services
running, redis PONG, livekit-server answering, **agent registered with
LiveKit**, **control plane accepts the gateway token**, **control plane
accepts the worker token**, artifact dirs writable. Every `[FAIL]` line
prints the exact fix; apply it and re-run until PASS.

## Step 8 — Place a test call

Call your DID from any phone.

**Expect:** the agent answers with its configured first message, and

```bash
docker compose -f docker-compose.livekit.yml logs -f agents
```

streams `media_admitted` → per-turn `turn_latency` → `call_complete`.
Artifacts land in `call-evidence/<call_id>/`; the control plane receives the
lifecycle/billing events and the uploaded recording + transcript.

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

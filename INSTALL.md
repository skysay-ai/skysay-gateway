# INSTALL — self-hosted Skysay LiveKit stack

This is the complete, linear install runbook for the **recommended** self-host
stack, the Skysay media plane. It is written to be
executable step by step **by a person or an AI agent** — every step has a
command, an expected result, and a failure pointer. No step requires
information that is not defined here or in the referenced file.

Background reading (optional):
[`packages/skysay-gateway/deploy/SELFHOST.md`](packages/skysay-gateway/deploy/SELFHOST.md).

---

## Step 0 — Prerequisites (collect these before touching the host)

| # | You need | How you know you have it |
|---|---|---|
| 0.1 | A Linux host with root/sudo. On a Mac, Docker Desktop **cannot** run this stack — it has no host networking, so the containers bind their ports and stay unreachable even from `localhost`; run it inside a Colima VM instead (`colima start --network-address --network-mode bridged`) and treat that VM as the host for every step below. A **public IPv4** your carrier can send calls to is the simple case; if you do not have one, see "Choosing your inbound topology" below | `curl -4 ifconfig.me` prints an IP — and if that IP is not actually bound to this host, you need the registration topology |
| 0.2 | Docker Engine + **Compose v2** on that host | `docker compose version` prints a version |
| 0.3 | A **DID** (phone number) on a SIP trunk that sends calls to your host, plus your carrier's **SIP signaling IP ranges** AND its **RTP / media IP ranges** (from the carrier portal — these are often DIFFERENT range sets) | e.g. DID `+15551230000`, signaling `203.0.113.0/24`, media `203.0.113.128/25` |
| 0.4 | An **Skysay control plane** URL | Skysay Cloud: `https://api.skysay.ai` |
| 0.5 | From that control plane: **organization id**, registered **gateway deployment id**, the long-lived credential you configure (the `skygw_` gateway token), and an **agent + phone-number route** configured for your DID | **API Keys → Gateway Tokens** creates the deployment-bound `skygw_`. An inbound-answering stack needs no second, organization-scoped key |
| 0.6 | Provider API keys for the components your agent's published voice stack actually resolves to — commonly **Deepgram** (STT) and **Gemini** (LLM), plus **ElevenLabs** only if your stack uses ElevenLabs TTS | from console.deepgram.com / aistudio.google.com / elevenlabs.io. An English agent on Gemini TTS needs no ElevenLabs key; Estonian requires one |
| 0.7 | Host firewall able to open `5060/udp` (to the SIP **signaling** ranges) and `10000-10100/udp` (to the RTP **media** ranges) separately. In the registration topology you DROP inbound `5060/udp` instead and forward nothing — see step 4 | cloud-firewall or `ufw`/`nftables` access |

### Getting 0.3–0.5, in order, as a new Skysay customer

Row 0.5 is one table cell describing several days of work. If you are starting
from nothing, this is the sequence — and the order matters, because carrier
registration is the one step you cannot hurry.

1. **Create a workspace.** Sign up at `skysay.ai` and sign in with the
   emailed code.
2. **Create an agent.** Name, language, first message and system prompt, then
   its STT / model / voice. **Publish it.** An unpublished or incomplete agent
   cannot answer a call, and the workspace will tell you exactly what is
   missing.
3. **Get a number.** Pick the country and number type you want.
   - If that combination needs **no carrier registration**, the number
     activates and you can move on.
   - If it **does**, you attach an end-user profile — a business or individual
     identity, with address and supporting documents — and submit it. The
     carrier reviews it. **This can take days, and the number cannot take calls
     until it is approved.** Start it before you need it.
4. **Route the number to the agent.** A number with no agent is a number that
   rings nowhere.
5. **Register this gateway.** Settings → Gateway → register a self-hosted
   gateway. That gives you `SKYSAY_GATEWAY_ID`.
6. **Mint the credential.** **API Keys → Gateway Tokens** → a `skygw_` token,
   bound to the one gateway you just registered. This is
   `SKYSAY_GATEWAY_API_TOKEN`, and for an inbound-answering stack it is the
   long-lived credential you configure (during a call the gateway also holds a call authorization and any provider keys leased to it): it routes the inbound call, reads
   that call's runtime config, and lists MCP tool schemas, each bound
   server-side to the workspace this deployment belongs to.

   It is shown **once**. Copy it straight into `.env.livekit`.
7. **Copy your organization id** into `SKYSAY_ORGANIZATION_ID`.

Steps 1, 2 and 5–7 are minutes. Step 3 is the one that can block you, so if you
are reading this to plan a deployment, start the number first.

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
# Obtain the preview gateway source package from Skysay first.
cd skysay-gateway/packages/skysay-gateway/deploy
```

**Expect:** you are in a directory containing `docker-compose.livekit.yml`,
`livekit.yaml.example`, `sip.yaml.example`, `.env.livekit.example`,
`selfhost-bootstrap.sh`, and `selfhost-smoketest.sh`.

## Step 2 (short way) — `selfhost-bootstrap.sh`: you set two values

Run this **before** `docker compose up`. Steps 2 and 3 below are what it does
for you; read them if you would rather do it by hand, or if it stops and asks
you for something.

```bash
SKYSAY_CONTROL_PLANE_URL=https://api.skysay.ai \
SKYSAY_BOOTSTRAP_API_KEY=sky_... \
  bash selfhost-bootstrap.sh
```

The key is an organization API key carrying `gateways:write`. Omit the variable
and it is read from the terminal without echo. **It is never written to this
box**, so retiring it afterwards is one action in the dashboard and nothing
else; the script reminds you by name when it finishes.

Derived and written: `SKYSAY_ORGANIZATION_ID`, `SKYSAY_GATEWAY_ID` (it
registers the deployment), the deployment-bound `SKYSAY_GATEWAY_API_TOKEN`
(minted with **no expiry** — an expiry with no auto-rotation only manufactures a
lockout), `SCH236_TEST_DID_E164`, and the `LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`
pair, instantiated identically into `livekit.yaml` and `sip.yaml`. It also
writes `gateway-state.json` (chmod 600) holding identity only — never the token
secret, and no container reads it.

**Still yours to set by hand:** `DIDWW_INBOUND_ADDRESSES` (the control plane
holds no carrier signalling ranges and cannot know them for a carrier you
brought yourself; a wrong allowlist is the everything-green-but-never-rings
failure) and your provider API keys.

Against an older control plane that cannot default your workspace from the key,
registration fails with a shaped error and you re-run with
`SKYSAY_ORGANIZATION_ID=org_...` in the environment, alongside the URL.

Re-running is safe and never guesses: a value already in `.env.livekit` is left
alone, an existing `livekit.yaml` is read rather than rewritten, and an existing
gateway token is probed before anything is minted (so a casual re-run does not
leave a live orphaned credential). Exactly one agent-routed number is used —
zero or several stops the run and tells you what to do, because provisioning
builds one trunk and the wrong number would leave you with a stack that never
rings. A fully configured host with no state file is adopted, unchanged.

## Step 2 (long way) — create the four config files

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

### Create the Skysay credential first

After registering the gateway deployment, open the workspace **API Keys**
page. (`selfhost-bootstrap.sh` does this step for you.)

1. Use the **Gateway Tokens** card to create a deployment-bound `skygw_` token.
   Its one-time reveal includes the exact line below; replace the blank
   `SKYSAY_GATEWAY_API_TOKEN=` assignment in `.env.livekit` with the whole
   revealed line.

   ```dotenv
   SKYSAY_GATEWAY_API_TOKEN=skygw_...
   ```

That is the long-lived credential you configure for an inbound stack; during a call the gateway also holds a call authorization and any provider keys leased to it. There is deliberately
no second, organization-scoped key on this box: the deployment-bound token
cannot be widened, and every organization it acts in is read out of the token
row server-side rather than from anything this host sends — so a compromise
here reaches this deployment's workspace and stops.

The secret is shown once only and is absent from list views. The gateway smoke
check fails closed for a wrong, revoked, expired, cross-organization, or
mismatched gateway-id/token pair; mint a replacement from that registered
gateway's **Gateway Tokens** card.

`SKYSAY_ORGANIZATION_ID` must name the workspace that registered this
gateway and owns the DID. The control plane compares it with the token's own
organization and refuses a mismatch before it returns the inbound call's
runtime configuration.

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `LIVEKIT_URL` | ws URL of this stack's LiveKit. Provisioning requires `wss://` or `ws://` on loopback (a remote `ws://` is refused — it would leak the admin JWT) | default for the host-local stack | `ws://127.0.0.1:7880` |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | the pair from step 2 | you generated it | `opx-selfhost` / 64 hex chars |
| `SKYSAY_CONTROL_PLANE_URL` | control-plane base URL. **Must be `https://`** (the agent refuses plaintext http except to loopback) | step 0.4 | `https://api.skysay.ai` |
| `SKYSAY_GATEWAY_ID` | your gateway deployment id | `POST /v1/gateway-deployments` (or dashboard → Gateways → new self-hosted gateway) | `gw_01HZX...` |
| `SKYSAY_GATEWAY_API_TOKEN` | deployment-scoped `skygw_` token — the long-lived credential you configure (during a call the gateway also holds a call authorization and any provider keys leased to it): number-rules sync, inbound routing, that call's runtime-config read, MCP `tools/list` schema lookup, lifecycle and artifact operations | **API Keys → Gateway Tokens** after registering that gateway — revealed **once** as `SKYSAY_GATEWAY_API_TOKEN=skygw_...` | `skygw_...` |
| `SKYSAY_ORGANIZATION_ID` | the org that owns your DID | dashboard → organization settings | `org_01HZX...` |
| `SCH236_TEST_DID_E164` | the DID(s) this gateway answers, **with** `+`. Comma-separate for more than one — they share one trunk and one dispatch rule | your carrier + registered in Skysay | `+15551230000` or `+15551230000,+3725551234` |
| `DEEPGRAM_API_KEY` | STT provider key | console.deepgram.com → API Keys | `...` |
| `GEMINI_API_KEY` | LLM provider key | aistudio.google.com → Get API key | `...` |
| `ELEVENLABS_API_KEY` | TTS provider key | elevenlabs.io → Profile → API Keys | `...` |
| `SCH236_NUM_IDLE_PROCESSES` | warm idle agent processes (optional) | default is fine | `2` |
| `SKYSAY_PHONE_AGENT_CAPACITY` | how many calls one worker takes at once (optional). The worker counts the calls it has accepted exactly; past this number a new incoming call hears a busy signal straight away, and an outgoing one is not dialled | set it to what your host measurably carries well; unset keeps the ceiling `SKYSAY_WORKER_MAX_JOBS` gave (`4` at its default) | `3` |
| `SKYSAY_WORKER_MAX_JOBS` | the most call processes one worker runs at once (optional). A call that has ended gives its place back at once, but its process keeps running for a few seconds while it writes the call's recording and reports, and it counts here until it exits | `5`; keep it about two above `SKYSAY_PHONE_AGENT_CAPACITY` | `5` |
| `SKYSAY_WORKER_JOB_MEMORY_WARN_MB` | when one job process's memory (PSS) is worth a log line (optional) | default is fine | `600` |
| `SKYSAY_WORKER_JOB_MEMORY_LIMIT_MB` | when one job process is killed for memory (optional). That process is a call, and a healthy 30-minute call peaks at 1.3-1.8 GB while it writes its recording after hangup, so a limit must sit above that; `0` never kills | default is fine | `0` |
| `DIDWW_INBOUND_ADDRESSES` | carrier signaling CIDRs (inbound allowlist for provisioning). **Public** ranges only — the example below is a documentation range and is rejected on purpose | carrier portal → signaling IPs | `203.0.113.0/24,198.51.100.0/28` (replace with your real ranges) |

The `provision` step uses `SCH236_TEST_DID_E164` (above) for every DID it lists
— re-running it with a different single number replaces the trunk rather than
adding to it, so put all your numbers in that one setting. You do
NOT set the DID twice.

**Optional — outbound trunk** (only to place calls OUT; inbound answering does
not need it). Supply **all four** or **none**:

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `DIDWW_OUTBOUND_HOST` | carrier outbound SIP host | carrier portal → SIP trunk | `sip.yourcarrier.com` |
| `DIDWW_OUTBOUND_FROM_USER` | one or more allowed outbound From-users, comma-separated, digits only | your caller-ID DID(s) | `15551230000,37255501234` |
| `DIDWW_OUTBOUND_AUTH_USERNAME` / `DIDWW_OUTBOUND_AUTH_PASSWORD` | outbound trunk auth pair | carrier portal → SIP trunk credentials | `...` |
| `SKYSAY_EXECUTE_LIVEKIT_OUTBOUND` | second fail-closed execution gate; leave `0` until provisioning + smoke tests pass | you | `0`, then `1` at cutover |
| `SKYSAY_OUTBOUND_WORKER_ID` | stable identity recorded when claiming jobs (legacy `queue` mode only) | choose a unique name per worker | `livekit-outbound-1` |
| `SKYSAY_OUTBOUND_TRUNK_NAME` | LiveKit outbound trunk the worker must match | provisioner default | `skysay-selfhost-outbound` |
| `SKYSAY_OUTBOUND_POLL_SECONDS` | queue poll interval (legacy `queue` mode only) | default is fine | `2` |
| `SKYSAY_WORKER_API_TOKEN` | organization-scoped `sky_` key, read ONLY by the legacy `queue` mode's job-claiming worker — leave blank on a `local`-mode deployment | workspace → API Keys | *(blank)* |
| `SKYSAY_OUTBOUND_RINGING_TIMEOUT_SECONDS` | stop unanswered ringing after this many seconds (5–120) | your operating policy | `45` |
| `SKYSAY_OUTBOUND_MAX_CALL_DURATION_SECONDS` | hard LiveKit SIP call cap (30–14400) | your operating policy | `1800` |

**Optional — Sinhala (si-LK) voice stack** (only needed for an agent
configured with `language: si`; neither Deepgram nor ElevenLabs supports
Sinhala, so a `si` agent resolves to Google Chirp 2 → Gemini → Gemini-TTS
— Chirp 2 is the only Google Sinhala transcriber). Set exactly **one** of
the two GCP credential variables:

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | path to a GCP service-account JSON key file (Cloud Speech-to-Text, role `roles/speech.client`) — **not** the Gemini API key above | GCP Console → IAM & Admin → Service Accounts → Keys → Add key (JSON) | `/run/secrets/gcp-speech-sa.json` |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | the same service-account JSON, inlined as one env var instead of a file | same service account as above | `{"type":"service_account",...}` |
| `GOOGLE_STT_LOCATION` | Cloud Speech-to-Text v2 recognizer region for Chirp 2 selections, Sinhala included. Leave it unset to take the worker default, `europe-west4` (measured 2026-09-05 from a European host: si-LK finals 531 ms sooner than `asia-southeast1`, byte-identical transcripts); set it only to pin another region. Chirp 2 serves si-LK in `europe-west4`, `asia-southeast1` and `us-central1`; a Sinhala Chirp 2 call is refused before media if this names any other region (`eu` and `global` included), so it fails visibly instead of transcribing nothing. Other Chirp 2 languages follow whatever region this names. Google STT must use chirp_2; leftover v1 identity is refused. | you | unset (`europe-west4`) |

**Optional — Tamil (ta-IN) voice stack** (only needed for an agent configured
with `language: ta`). The worker serves `en`, `et` and `si` from the pipelines
above; every language added since is declared in the capability manifest that
ships with the agent sources. Tamil transcription defaults to Deepgram
nova-3; Soniox and Azure remain alternatives. One Soniox key covers both its
STT and its TTS. Google Speech V1 is not a Tamil option.

| Variable | What it is | Where to get it | Example |
|---|---|---|---|
| `SONIOX_API_KEY` | Soniox account key — STT **and** TTS. Required for the default Tamil stack | soniox.com → console → API keys | `...` |
| `FISH_AUDIO_API_KEY` | Fish Audio account key. Only if the agent's voice is a Fish Audio voice | fish.audio → account → API keys | `...` |
| `OPENAI_API_KEY` | OpenAI account key for GPT-Live conversations. Workflows need Skysay's managed media and do not run on a self-hosted gateway | OpenAI console → API keys | `...` |
| `CARTESIA_API_KEY` | Cartesia account key. Only if the agent's voice is a Cartesia Sonic 3.6 voice selected from its provider-backed inventory | play.cartesia.ai → API keys | `...` |
| `AZURE_SPEECH_KEY` | Azure AI Speech subscription key. One key covers both the si-LK transcriber and the Azure neural Sinhala/Tamil voices | portal.azure.com → your Speech resource → Keys and Endpoint | `...` |
| `AZURE_SPEECH_REGION` | The region that Speech resource is in. Required whenever the key is set — it selects the endpoint host, so a key alone reaches nothing | the same portal page, "Location/Region" | `germanywestcentral` |

Google Speech V1 `command_and_search` is gone, and Tamil is not
declared on Chirp 2. Tamil transcription defaults to Deepgram nova-3; Soniox
and Azure remain alternatives. Gemini TTS remains selectable for Tamil. The
Sinhala section's GCP credential is only needed for a Google Chirp 2
transcriber, not for Tamil STT.

**Optional once Skysay supplies the credential.** Deepgram and ElevenLabs
components can run on a credential Skysay issues for your workspace — not
the platform's own key, which never leaves the platform, but a separate
per-workspace one leased to this deployment for a single live call. Soniox,
Google (Gemini, Cloud STT), Fish Audio, and Cartesia have no issuer yet, so those keys
stay in this file. This is opt-in per provider and per workspace: until
Skysay has issued you a credential, every component reads this file exactly
as before, and upgrading the worker changes nothing.

**Optional once your workspace holds the key instead.** A provider key in this
file whose provider supports workspace keys (not `AZURE_SPEECH_KEY` or
`OPENAI_API_KEY`, which have no workspace-key path) is per-component optional:
store the key in your workspace
(**Settings → Provider keys**) and bind it to the agent's transcriber, model or
voice, and the control plane leases it to this deployment for one call at a time
rather than expecting a value here. The deployment's `skygw_` token needs the
`component_credentials:lease` scope for that — tokens minted before the scope
existed do not carry it, and rotation preserves the old scope list, so issue a
new token instead. A component that is not bound to a stored key keeps reading
this file exactly as before; there is no fallback in either direction.

Do **not** put `SONIOX_STT_API_KEY`, `SONIOX_TTS_API_KEY` or
`FISH_AUDIO_TTS_API_KEY` in this file. Those are per-call lease names the
control plane injects for a workspace that brings its own key; the worker
treats one of them without its `SKYSAY_BYOK_*` marker as a corrupt lease
and refuses the call. A copy here is inert rather than an outage —
`carrier-entrypoint.sh` unsets every lease name after it sources this file —
but it is still not a key the worker will ever read.

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
| `SIP_REGISTRATION_REALM` | digest realm the carrier challenges with | defaults to `SIP_REGISTRATION_REGISTRAR`, which is right for most carriers — but a carrier that challenges with its bare domain while you register against a regional host needs it set explicitly (DIDWW: always `sip.didww.com`, even when registering via `ams.sip.didww.com`; the symptom of a wrong realm is `realms do not match` in the registrar log and a registration that retries forever) | `sip.yourcarrier.com` |
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
cd packages/skysay-gateway/deploy
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

### Outbound calling from this box

Outbound is optional; skip this if the deployment only answers calls.

There are two ways a deployment's outbound calls get dialed, and
`outbound_execution_mode` on the deployment says which one you are on. Read it
with your gateway token:

```bash
curl -sS -H "Authorization: Bearer $SKYSAY_GATEWAY_API_TOKEN" \
  "$SKYSAY_CONTROL_PLANE_URL/v1/gateway-deployments/$SKYSAY_GATEWAY_ID" \
  | grep -o '"outbound_execution_mode":"[a-z]*"'
```

**`local` (recommended for a self-hosted box).** This box asks the control
plane to admit the call it is about to place, and dials it with the single-use
token that request returns. It needs **no organization-scoped key** — the same
`skygw_` token does inbound and outbound. Ask the workspace owner to set the mode
(`PATCH /v1/gateway-deployments/{id}/outbound-execution-mode` with
`{"outbound_execution_mode": "local"}`, `gateways:write`), set
`SKYSAY_EXECUTE_LIVEKIT_OUTBOUND=1` in `.env.livekit`, and place a call:

```bash
docker compose -f docker-compose.livekit.yml run --rm outbound-worker \
  python /app/outbound_worker.py originate \
    --to +15551234567 --purpose appointment --reason "confirm tomorrow's booking"
```

`--from` defaults to the first number in `SCH236_TEST_DID_E164`, and the agent
defaults to that number's own agent. `--purpose` and `--reason` are required:
the control plane's outbound authorization is a decision about a stated
purpose, and this command will not invent one for you. It prints one
`SKYSAY_OUTBOUND {...}` line with the call id, or
`SKYSAY_OUTBOUND_FAILED {...}` with the reason.

Note that `docker compose run` ignores compose profiles, so
`SKYSAY_EXECUTE_LIVEKIT_OUTBOUND=1` is what gates this command — it refuses
while the gate is `0`. `--profile outbound` still gates the long-running
queue-claiming worker below.

Once the deployment is in `local` mode, a remote `POST /v1/calls` for this
workspace's numbers is **refused** rather than queued, with a message naming
the command above. That is deliberate: this box no longer polls the job queue,
so a queued call would reserve money and never ring.

**`queue` (the default, and every deployment created before local
origination).** Outbound calls are requested from the control plane with an
organization-scoped API key (`POST /v1/calls`), and this box's
`--profile outbound` worker claims and dials them. Claiming another requester's
`call.outbound_originate` job is `worker_jobs:*` authority over the whole
workspace queue, which a deployment-bound token deliberately does not carry —
so this mode is the one part of the stack that reads an organization-scoped
`sky_` key, in `SKYSAY_WORKER_API_TOKEN`. Leave that variable blank and
keep `SKYSAY_EXECUTE_LIVEKIT_OUTBOUND=0` unless you mean to take back the
organization-wide blast radius on purpose.

## Step 7 — Verify everything (the smoke test)

```bash
bash selfhost-smoketest.sh
```

`RESULT: UNVERIFIED` (exit 2) means every check that could run passed but at
least one could not be performed at all — usually the firewall check, which
needs root. Treat it as not-yet-ready rather than a pass.

**Expect:** `RESULT: PASS` and a per-check `[PASS]` list covering: services
running, redis PONG, livekit-server answering, **agent registered with
LiveKit**, the matching deployment-bound **gateway `skygw_` token** accepted by
the non-mutating number-rules read and by MCP `tools/list`, artifact dirs
writable (an organization-scoped `sky_` key is checked too, but only if an
install still sets one). Every `[FAIL]`
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
receives the lifecycle/billing events and, after the call, the recording and
transcript, with the telemetry evidence. During the call it also receives the
arguments of each tool call, any knowledge-base question the agent sends and,
when the agent has tools that write to your systems, what the caller says; and
audio and text go from the gateway
to your carrier and to the speech and model providers your agent uses.

**Done.** The stack self-recovers (`restart: unless-stopped`; the
`reconciler` closes sessions of a crashed agent so no dangling billed
minutes).

## Troubleshooting

The smoke test localizes almost everything. For the full symptom → fix table
(agent restart loops, provisioning interlocks, 401/403 token confusion,
one-way audio, slow first answer, EACCES on artifact dirs) see
[`SELFHOST.md` → Troubleshooting (LiveKit)](packages/skysay-gateway/deploy/SELFHOST.md#troubleshooting-livekit).

## Updating

```bash
git pull
docker compose -f docker-compose.livekit.yml up -d --build
bash selfhost-smoketest.sh
```

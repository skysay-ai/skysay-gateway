# Self-host quickstart

**This stack is the same media plane OpenPhonex production runs.** Follow it
top to bottom; every step has a copy-pasteable command and a verifiable
expected result. (An AI agent can execute this guide as written; the repo-root
`INSTALL.md` in the public export is the same runbook in checklist form.)

---

# LiveKit self-host (matches production)

Same agent code (shipped at [`livekit_agent/`](../../../livekit_agent/) in the
public repo — the runtime modules are byte-identical to the internal production
source, enforced by export tests), same pinned component versions, same
control-plane contract.

```
      PSTN / your carrier (SIP trunk, your DID)
                    |
        UDP 5060 + RTP 10000-10100
                    v
  [ livekit-sip v1.7.0 ] --- [ livekit-server v1.13.4 ] --- [ redis 8.8.0 ]
                    \                 |
                     \                | (agent dispatch, host-local)
                      v               v
                  [ agents: the production LiveKit agent ]
                  [ reconciler: stale-session cleanup    ]
                    |
                    |  HTTPS (bearer tokens; per-call config, tools,
                    v   lifecycle/billing events, artifact upload)
          [ OpenPhonex control plane (Cloud or self-hosted) ]
```

Pinned components (same tag **and** digest as production — enforced by an
export test):

| Component | Version |
|---|---|
| `livekit/livekit-server` | `v1.13.4` (digest-pinned) |
| `livekit/sip` | `v1.7.0` (digest-pinned) |
| `redis` | `8.8.0-alpine` (digest-pinned) |
| `livekit-agents` + Deepgram/Google/ElevenLabs/Silero/turn-detector plugins | `1.6.6` (pip-pinned in `livekit_agent/agent/requirements.txt`) |
| Voice pipeline (default) | Deepgram nova-3 → Gemini → ElevenLabs (resolved per call from realtime-config) |
| Voice pipeline (si-LK / Sinhala) | Google Chirp 2 → Gemini → Gemini-TTS — see [Language stacks: Sinhala (si-LK)](#language-stacks-sinhala-si-lk) below |

## 0. What you need first

| # | Requirement | Where to get it |
|---|---|---|
| 1 | A Linux host with a **public IP** and Docker Engine + Compose v2 | any cloud VM (2 vCPU / 4 GB is plenty to start) |
| 2 | A **SIP trunk + DID** (phone number) | your carrier (production uses DIDWW; any carrier that sends SIP/RTP to your host works). Note BOTH the carrier's **signaling IP ranges** and its **RTP / media IP ranges** from its portal — they are often different range sets |
| 3 | An **OpenPhonex control plane** you can reach | OpenPhonex Cloud (`https://api.openphonex.com`) or your own deployment |
| 4 | From that control plane: an **organization id**, a **gateway deployment id + `tgw_` token**, a **worker API token**, and an **agent + phone-number route** configured for your DID | dashboard, or `POST /v1/gateway-deployments` → `POST /v1/gateway-deployments/{id}/tokens` (token shown once) |
| 5 | Provider keys: `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` | console.deepgram.com / aistudio.google.com / elevenlabs.io |
| 5b | *(only for si-LK / Sinhala agents)* a GCP service account for Cloud Speech-to-Text v2 | see [Language stacks: Sinhala (si-LK)](#language-stacks-sinhala-si-lk) below |
| 6 | Firewall: open `5060/udp` to the carrier's **signaling** ranges and `10000-10100/udp` to its **RTP / media** ranges (scope each range set separately) | your host/cloud firewall. LiveKit's own API and Redis bind loopback-only. `50000-50020/udp` is only needed if you later add WebRTC (browser) participants |

## 1. Configure (three files)

```bash
cd packages/openphonex-gateway/deploy
cp livekit.yaml.example livekit.yaml   # LiveKit server config
cp sip.yaml.example sip.yaml           # LiveKit SIP bridge config
cp .env.livekit.example .env.livekit   # control-plane + provider credentials
chmod 600 livekit.yaml sip.yaml .env.livekit   # never commit the filled-in files
mkdir -p sessions call-evidence
# The agent drops to uid 10001 inside the container before writing these dirs,
# so give the host-side bind mounts to that uid or the first call EACCESes:
sudo chown -R 10001:10001 sessions call-evidence
```

Then fill them in:

1. **Generate one LiveKit key pair** (you invent it; nobody issues it):
   `openssl rand -hex 32` for the secret, any short id (e.g. `opx-selfhost`)
   for the key. Put the **same pair in all three files**: `livekit.yaml`
   (`keys:` map), `sip.yaml` (`api_key`/`api_secret`), `.env.livekit`
   (`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`).
2. **Fill in every remaining variable in `.env.livekit`.** Each one is
   documented inline — what it is, where to get it, and an example value.
   Two rules worth repeating here:
   - `OPENPHONEX_CONTROL_PLANE_URL` **must be `https://`** for a remote
     control plane. The agent refuses to send bearer tokens over plaintext
     http to anything but verified loopback (`127.0.0.0/8`, `::1`,
     `localhost`), and refuses cross-origin redirects.
   - `SCH236_TEST_DID_E164` takes the leading `+` (`+15551230000`) — it is the
     one DID you set; provisioning reuses it.
   - `DIDWW_INBOUND_ADDRESSES` must be your carrier's **real public** signaling
     ranges. The example value is a documentation range and is rejected on
     purpose (so you can't provision an allowlist no real call matches).
   - Outbound trunk vars (`DIDWW_OUTBOUND_*`) are **optional** — set all four
     to also create an outbound trunk, or leave all four blank for
     inbound-only. `DIDWW_OUTBOUND_FROM_USER` accepts comma-separated caller
     IDs when your carrier permits more than one From number.

## 2. Bring the stack up

```bash
docker compose -f docker-compose.livekit.yml up -d --build
```

The first build installs the pinned `livekit-agents` wheels and pre-downloads
the VAD/turn-detector model files into the image (a few minutes, once).

## 3. Provision the SIP trunk + dispatch rule (one-shot)

```bash
docker compose -f docker-compose.livekit.yml --profile provision run --rm provision
```

Expect a final line `SELFHOST_PROVISIONED {...}`. This registers your DID and
carrier allowlist with LiveKit SIP (and an outbound trunk if you set the
optional `DIDWW_OUTBOUND_*` vars); rerun it only when the DID or the carrier IP
ranges change. If it prints `SELFHOST_PROVISION_FAILED {...}`, the JSON `error`
field states the exact reason — see [Troubleshooting](#troubleshooting-livekit).

### Outbound calling

The `outbound-worker` compose profile claims only policy-approved
`call.outbound_originate` jobs. It consumes the call's short-lived,
destination-bound authorization before dialing and explicitly dispatches the
production agent into the LiveKit room. Per-call instructions, opening line,
and language are fetched from OpenPhonex by call ID; they are never trusted
from SIP or room metadata.

Keep `OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND=0` while staging. Run the worker
`--status` command from `INSTALL.md`; after it succeeds, set the gate to `1`
and start `docker compose --profile outbound up -d outbound-worker`.

## 4. Verify — run the smoke test

```bash
bash selfhost-smoketest.sh
```

It prints a `[PASS]`/`[FAIL]` line per check (services up, redis, LiveKit,
**agent registered**, **gateway token accepted**, **worker token
authenticates**, artifact dirs writable) and ends with `RESULT: PASS` when the
stack is up and both tokens authenticate. Every `[FAIL]` line includes the
exact command or fix to apply.

`RESULT: PASS` is necessary but **not sufficient** for a call to succeed: the
worker-token check only proves it *authenticates* (MCP `tools/list`), not that
it carries the `calls:write` / routing scope, and the smoke test does not
verify that your DID's number route + agent are configured in the control
plane. The first real call in step 5 is the end-to-end confirmation.

Manual spot-check, if you prefer:

```bash
docker compose -f docker-compose.livekit.yml logs -n 20 agents
```

Expect a `registered worker` line and one `SCH236_PRODUCTION {"event":
"process_warm", ...}` line per warm process.

## 5. Place a test call

Call your DID from any phone. The agent answers with the configured first
message, and `docker compose -f docker-compose.livekit.yml logs -f agents`
streams structured events:

```
media_admitted        (semantic_config_source: "realtime-config")
turn_latency          (one per conversational turn)
model_tools_executed  (when the agent invokes an allowlisted tool)
call_complete         (with the billing source)
```

Call artifacts (caller/agent/mixed WAV + transcript + per-turn clips) land in
`call-evidence/<call_id>/` and are uploaded to the control plane; per-session
state lives in `sessions/`.

## Security notes

- **Worker token is organization-scoped** (`OPENPHONEX_WORKER_API_TOKEN`).
  This mirrors production: inbound routing and MCP tool invocation are org
  operations, so whoever holds this token can route calls and invoke tools for
  the whole organization. Treat the box accordingly: `chmod 600 .env.livekit`,
  no shared shell access, and **rotate the token immediately** if the host is
  ever suspected compromised. (Narrowing this to per-call tokens is a planned
  control-plane change; the gateway `tgw_` token is already narrowly scoped.)
  See the repo-root `SECURITY.md`.
- **Transport is enforced, not just advised**: the agent refuses plaintext
  http to non-loopback control planes and refuses cross-origin redirects, so
  the bearer tokens cannot be tricked onto another host.
- **Firewall to your carrier only.** `5060/udp` and `10000-10100/udp` should
  admit your carrier's published ranges, not the whole internet; the provision
  step additionally enforces the carrier allowlist at the trunk level.

## How per-call config resolves (the production path)

The agent joins the room **unsubscribed** and admits no caller audio until
the control plane authorizes the call:

1. `POST /v1/calls/inbound/route` (worker token) — resolves your DID to the
   authoritative agent and creates the call. Endpoint-routed calls resolve
   tools the same way.
2. `GET /v1/agents/{agent_id}/realtime-config` (gateway token) — the resolved
   semantic config: system prompt, first message, language, transcriber/model/
   voice provider selection (Deepgram/Gemini/ElevenLabs by default, or Google
   Chirp 2/Gemini/Gemini-TTS for `language: si`), and the tool allowlist. The
   agent validates it and only then subscribes to media.
3. Tools: the agent fetches schemas via MCP `tools/list` on `POST /mcp`,
   exposes **only** allowlisted tools to the model, and every invocation
   crosses the control plane again (`tools/call`) with an audit trail.
4. Lifecycle: `sip.answered` / `room.media_started` / heartbeats /
   `room.ended` / `sip.ended` are reported to
   `POST /v1/gateway-deployments/{id}/call-lifecycle-events`; the terminal
   event produces the authoritative billing verdict.
5. Artifacts (caller/agent/mixed WAV + transcript + per-turn clips) upload to
   `POST /v1/calls/{call_id}/artifacts` with duration provenance.

The contract surface is [`contracts/gateway-v1.yaml`](../../../contracts/gateway-v1.yaml)
+ [`contracts/schemas/`](../../../contracts/schemas/).

The `reconciler` service closes sessions whose agent died mid-call
(`session.reconciled` lifecycle events), so a crash can never leave dangling
billed minutes.

## Language stacks: Sinhala (si-LK)

The default pipeline (Deepgram nova-3 → Gemini → ElevenLabs) does not cover
Sinhala — **neither Deepgram nor ElevenLabs supports si-LK at all**. An agent
configured with `language: si` instead resolves to:

| Component | Provider | Notes |
|---|---|---|
| STT (transcriber) | Google Cloud Speech-to-Text v2, **Chirp 2** | `asia-southeast1` recognizer location — empirically verified against the live API for si-LK. Chirp 3 rejects si-LK with an HTTP 400 in both `eu` and `us`; Chirp 2 in `asia-southeast1` transcribed a Sinhala test phrase correctly at 0.89 confidence, and `asia-southeast1` is also geographically closer to Sri Lanka than `eu` |
| LLM (model) | Gemini | same model catalog as every other language — Gemini handles Sinhala natively, no separate credential |
| TTS (voice) | **Gemini-TTS** (`livekit.plugins.google.beta.GeminiTTS`) | same `GEMINI_API_KEY` as the LLM above; non-streaming (chunked HTTP synthesis, not a realtime websocket) |

The catalog rejects any other combination for `si` up front — picking
Deepgram or ElevenLabs for a Sinhala agent fails validation at config-save
time with a clear error, not silently at call time.

### 1. GCP service account (Chirp 2 credentials)

Chirp 2 is a Cloud Speech-to-Text **v2** API call authenticated with a real
GCP service account — this is **not** the `GEMINI_API_KEY` above (that's a
Gemini API key, a completely different Google product surface).

1. In the [GCP Console](https://console.cloud.google.com), pick or create a
   project, then enable **Cloud Speech-to-Text API**
   (`speech.googleapis.com`).
2. IAM & Admin → Service Accounts → **Create service account**. Grant it the
   **Cloud Speech Client** role (`roles/speech.client`) — no broader role is
   needed.
3. Open the new service account → **Keys** → **Add key** → **Create new key**
   → JSON. Download the file.
4. Give the agent stack that key one of two ways:
   - **File mount (default):** copy the JSON to the host, bind-mount it into
     the `agents` container, and set `GOOGLE_APPLICATION_CREDENTIALS=<path
     inside the container>` in `.env.livekit`.
   - **Inline env var:** if your platform only supports secrets-as-env-vars,
     set `GOOGLE_APPLICATION_CREDENTIALS_JSON=<the file's contents, as a
     single-quoted single line>` instead — no file mount needed. The agent
     tries this first if both are set.

### 2. Env vars (new in `.env.livekit`)

```
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp-speech-sa.json   # OR:
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
GOOGLE_STT_LOCATION=asia-southeast1   # optional; asia-southeast1 is the default and the empirically verified si-LK region (Chirp 3 rejects si-LK with an HTTP 400 in eu/us)
```

`GEMINI_API_KEY` (already required by the default pipeline) covers both the
Gemini LLM and Gemini-TTS — no extra key for TTS.

### 3. Configure the agent

Set the agent's `language` to `si` and its transcriber/voice to the si-LK
stack (via the dashboard's agent config editor — the language picker and
provider dropdowns are catalog-driven and simply show the new options once
this deploy has the credentials above — or via the API/MCP `set_agent_voice_stack` /`create_agent` call):

```json
{
  "language": "si",
  "transcriber": {"provider": "google", "model": "chirp_2"},
  "model": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
  "voice": {"provider": "gemini", "model": "gemini-2.5-flash-preview-tts", "voice_id": "Kore"}
}
```

### 4. Turn-taking caveat

The pinned turn-detector plugin (`livekit-plugins-turn-detector` /
`inference.TurnDetector(version="v1-mini")`) does not have a calibrated model
for Sinhala (its local model only covers ar/de/en/es/fr/hi/id/it/ja/ko/nl/
pt/tr/zh — the same reason Estonian `et` calls already fall back to VAD-only
today). The worker detects this automatically per-call and falls back to
Silero VAD + provider end-of-utterance timing instead of running the
turn-detector uncalibrated. **Tradeoff:** turn-taking on si-LK calls is
slightly slower / less natural than on a turn-detector-supported language —
the agent waits out a fixed silence window (350–1500 ms) rather than acting on
a semantic end-of-turn signal. This is a deliberate, tested fallback (see
`TURN_DETECTOR_SUPPORTED_LANGUAGES` in [`livekit_agent/demo.py`](../../../livekit_agent/demo.py)),
not an oversight.

### 5. Validating before a real customer call

Do **not** place a live PSTN call to validate this — synthesize and transcribe
a short recorded Sinhala sample offline first. Once you have a short si-LK WAV
sample (a native speaker saying a few sentences), a founder-run validation
looks like:

**Empirically confirmed against the live Google Speech-to-Text v2 API:**
`chirp_2` + `asia-southeast1` transcribes si-LK correctly (0.89 confidence on
a Gemini-TTS-synthesized Sinhala test phrase); `chirp_3` rejects si-LK with an
HTTP 400 in both `eu` and `us`. Use the `chirp_2`/`asia-southeast1` pairing
below, not `chirp_3`.

```python
# validate_sinhala_stack.py — run with: python3 validate_sinhala_stack.py sample.wav
import sys
from livekit.plugins import google
from livekit.plugins.google.beta import GeminiTTS

stt = google.STT(model="chirp_2", languages="si-LK", location="asia-southeast1")   # picks up
                                                                                     # GOOGLE_APPLICATION_CREDENTIALS
tts = GeminiTTS(model="gemini-2.5-flash-preview-tts", voice_name="Kore")  # picks up GEMINI_API_KEY / GOOGLE_API_KEY

# 1. STT: confirm a recorded si-LK sample transcribes to readable Sinhala text.
#    (see livekit.plugins.google.STT.recognize / .stream for the exact call
#    shape used by build_demo_stack() in livekit_agent/demo.py)
#
# 2. TTS: confirm a short Sinhala sentence synthesizes to audible, correct speech.
#    tts.synthesize("ඔබට ණය ගෙවීමේ දිනය මතක් කිරීමට කතා කරමි.")
```

This is the same build-the-exact-pinned-plugin-objects-and-round-trip-a-short-
phrase pattern this stack already uses for provider interop smokes; use the
snippet above as a starting point once you have a recorded sample. **This
repo's automated validation of Gemini-TTS Sinhala output is a
plugin-source-code / vendor-documentation check only — nobody on this change
has heard the actual audio.** Listen to the synthesized sample yourself before
routing real customer calls to it.

## Updating

```bash
git pull
docker compose -f docker-compose.livekit.yml up -d --build
bash selfhost-smoketest.sh
```

Version pins (image digests, plugin versions) only change when production's
change — they are enforced equal by the release tests.

## Troubleshooting (LiveKit)

- **`bash selfhost-smoketest.sh` first.** It pinpoints the failing layer and
  prints the fix; everything below assumes you have its output.
- **`agents` restarts in a loop / exits immediately:** almost always a missing
  or malformed variable in `.env.livekit` — `docker compose -f
  docker-compose.livekit.yml logs -n 100 agents` shows the exact `KeyError` or
  validation message. The entrypoint also exits early (`carrier runtime
  environment is unavailable`) if `.env.livekit` is missing from the bind
  mount.
- **`ValueError: control-plane base_url must be https...`:** you pointed
  `OPENPHONEX_CONTROL_PLANE_URL` at plaintext http on a non-loopback host.
  Use `https://` (this is a deliberate guard — it protects your tokens).
- **Provision prints `SELFHOST_PROVISION_FAILED {...}`:** read the JSON
  `error` field — it is the exact reason. Common ones: a missing/blank
  required variable; a `DIDWW_INBOUND_ADDRESSES` entry that isn't a public
  IPv4 CIDR (use your carrier's real ranges, not the documentation example);
  or a partially-filled `DIDWW_OUTBOUND_*` set (supply all four or none).
- **Smoke test: `control plane accepts the gateway token` FAILS with 401/403:**
  the `tgw_` token is wrong, revoked, or belongs to another deployment id.
  Mint a fresh one: `POST /v1/gateway-deployments/{id}/tokens`. If the
  **worker** check fails instead, you pasted a `tgw_` token where the
  org-scoped worker token belongs (they are different tokens).
- **Call rings but nobody answers:** the dispatch rule or number route is off.
  Check (a) provision succeeded (`SELFHOST_PROVISIONED`), (b) the control
  plane has a phone-number route + agent for the DID, (c)
  `SCH236_TEST_DID_E164` has the leading `+`.
- **Call connects, instant hangup, or one-way/no audio:** firewall. Confirm
  `5060/udp` AND the RTP range `10000-10100/udp` are open to your carrier's
  ranges, and that `DIDWW_INBOUND_ADDRESSES` covers the address the INVITE
  actually came from (the `sip` service logs show the source IP).
- **First call is slow to answer:** the model prewarm removes the per-call
  cold start, but only after `process_warm` appears in the agents log — give a
  fresh stack a minute.
- **`EACCES`/permission errors on `sessions/` or `call-evidence/`:**
  `sudo chown -R 10001:10001 sessions call-evidence` (the agent drops to
  uid 10001 before writing).

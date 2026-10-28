# Skysay Gateway

**Self-host the Skysay media plane.** This repository is the LiveKit-based
voice stack behind [Skysay](https://skysay.ai): livekit-server + livekit-sip +
the media-plane agent in [`livekit_agent/`](livekit_agent/) + a session
reconciler + redis. Generated from the Skysay source, with documented omissions, substitutions and export transformations.

**You self-host the media plane** — call audio (SIP/RTP) terminates on your
servers, and the voice pipeline runs with the speech and model providers your
agent uses: on provider keys you put in the gateway's environment, or on a key
Skysay issues for your workspace or one you stored in Skysay, leased to the
gateway for each call.

**What leaves your servers.** The gateway runs on your servers and carries the call's audio. From there, audio and text go to your carrier and to the speech and model providers your agent uses. Skysay Cloud receives the call's metadata; during the call, the arguments of each tool call, any knowledge-base question the agent sends and, when the agent has tools that write to your systems, what the caller says; after the call, the recording and transcript.
**Skysay Cloud stays the control plane** — phone numbers (global carrier
coverage), end-user KYC/KYB compliance, Number Rules & routing, per-call
authorization, billing & wallet, and the dashboard · API · MCP.

```
      PSTN / your carrier (SIP trunk, your DID)
                    |
        UDP 5060 + RTP 10000-10100
                    v
  [ livekit-sip v1.13.0 ] --- [ livekit-server v1.13.6 ] --- [ redis 8.8.0 ]
                    \                 |
                     \                | (agent dispatch, host-local)
                      v               v
                  [ agents: the exported LiveKit worker  ]
                  [ reconciler: stale-session cleanup    ]
                    |
                    |  HTTPS (bearer tokens; per-call config, tools,
                    v   lifecycle/billing events, artifact upload)
          [ Skysay Cloud control plane ]
```

- License: **Apache-2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE))
- Brand: [`TRADEMARK.md`](TRADEMARK.md) — the code is open; "Skysay" is not licensed
- Contract: [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) (v1.0.0, stable)
- Agent/LLM guide: [`llms.txt`](llms.txt) · agent instructions: [`AGENTS.md`](AGENTS.md)

## What runs on a self-hosted gateway

| Engine | Self-hosted on your own keys? |
|---|---|
| Cascade (STT + LLM + TTS, every provider in this repository) | Yes |
| GPT-Live (free-form calls) | Yes (`OPENAI_API_KEY`) |
| Gemini Live | Yes (`GEMINI_API_KEY`, stock model) |
| Eleven Realtime | No, Skysay Cloud only: its agent registry and billing run on Skysay's managed media, and its modules are not in this repository |
| Workflows (any engine) | No, Skysay Cloud only: workflow turns and deliveries need Skysay's managed media |
| Phone-menu navigation | No, Skysay Cloud only (this repository ships a stand-in that reports it off) |

A call your gateway cannot run is refused by name before any media starts:
`engine_not_self_hostable` or `workflow_not_self_hostable`.

## Repository map

| Path | What |
|------|------|
| [`INSTALL.md`](INSTALL.md) | **Step-by-step self-host install runbook** — written to be executable by a person or an AI agent |
| [`AGENTS.md`](AGENTS.md) | Operating instructions for AI agents working in this repo |
| [`livekit_agent/`](livekit_agent/) | **The LiveKit media-plane agent** — generated from the Skysay source (see the provenance note above) + a minimal pinned Dockerfile |
| [`packages/skysay-gateway/deploy/`](packages/skysay-gateway/deploy/SELFHOST.md) | The self-host stack: compose file, config templates, `SELFHOST.md` runbook, `selfhost-smoketest.sh` |
| [`packages/skysay-gateway/`](packages/skysay-gateway/) | Caller-side evidence referee and portable RTP codecs; no call runtime or provider registry |
| [`packages/skysay-common/`](packages/skysay-common/) | Control-plane-free shared config + runtime helpers |
| [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) | The exact gateway↔Cloud API surface (stable v1.0.0) |
| [`contracts/schemas/`](contracts/schemas/) | JSON Schemas for the stable request/response bodies |

## Quickstart (self-host)

**Follow [`INSTALL.md`](INSTALL.md)** — a linear runbook (prerequisites →
configure → `docker compose up` → provision → smoke test → test call) with
every environment variable documented (what it is, where to get it, example).
The stack ships a self-check:

```bash
cd packages/skysay-gateway/deploy
bash selfhost-smoketest.sh   # PASS/FAIL per check; PASS = call-ready
```

Pinned components (tag **and** digest; livekit-server and livekit-sip carry
the versions this repository intends the production voice box to run —
treat this as the target pin, not a live parity claim, and re-verify against
the running containers; redis intentionally does not track production —
production is on 7.4.5-alpine): livekit-server v1.13.6, livekit-sip v1.13.0, redis 8.8.0,
livekit-agents 1.8.8 (+ Deepgram/Google/ElevenLabs/Silero/turn-detector/OpenAI
plugins, pip-pinned in
[`livekit_agent/agent/requirements.txt`](livekit_agent/agent/requirements.txt)).

Per-call agent config — system prompt, models, voice, and the tool
allowlist — resolves live from the control plane
(`POST /v1/calls/inbound/route` → `GET /v1/calls/{call_id}/realtime-config`);
the agent admits **no caller audio** until that authorization completes.
Background and troubleshooting:
[`packages/skysay-gateway/deploy/SELFHOST.md`](packages/skysay-gateway/deploy/SELFHOST.md).

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/skysay-common -e packages/skysay-gateway numpy==2.5.3

python -m unittest discover -s tests    # runs without any cloud access
```

## Safety

The agent is **fail-closed**: it joins each inbound room unsubscribed and
consumes no caller audio until the control plane resolves the number route and
returns a validated realtime-config. Tools are exposed to the model only if
allowlisted there, and every invocation crosses the control plane with an
audit trail. The control-plane transport requires **https** (or verified
loopback http) and refuses cross-origin redirects. Audio on the wire is
`pcm_s16le_8000_mono` (16-bit LE PCM, 8 kHz, mono; no negotiation). See
[`SECURITY.md`](SECURITY.md) for the token model (gateway `skygw_` token vs the
organization-scoped worker token).

## More

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`SECURITY.md`](SECURITY.md)
- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)

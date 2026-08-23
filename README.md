# OpenPhonex Gateway

**Self-host the media plane OpenPhonex production runs.** This repository is
the LiveKit-based voice stack behind [OpenPhonex](https://openphonex.com):
livekit-server + livekit-sip + the **production media-plane agent** (the
runtime modules in [`livekit_agent/`](livekit_agent/) are byte-identical to
what OpenPhonex production runs, enforced by release tests) + a session
reconciler + redis.

**You self-host the media plane** — call audio (SIP/RTP) terminates on your
servers, and the voice pipeline (Deepgram STT → Gemini LLM → ElevenLabs TTS)
runs on your own provider keys.
**OpenPhonex Cloud stays the control plane** — phone numbers (global carrier
coverage), end-user KYC/KYB compliance, Number Rules & routing, per-call
authorization, billing & wallet, and the dashboard · API · MCP.
The only thing connecting the two is a thin control link: bearer tokens ·
per-call config · lifecycle/billing events · artifact upload.

```
      PSTN / your carrier (SIP trunk, your DID)
                    |
        UDP 5060 + RTP 10000-10100
                    v
  [ livekit-sip v1.7.0 ] --- [ livekit-server v1.13.5 ] --- [ redis 8.8.0 ]
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

- License: **Apache-2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE))
- Brand: [`TRADEMARK.md`](TRADEMARK.md) — the code is open; "OpenPhonex" is not licensed
- Contract: [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) (v1.0.0, stable)
- Agent/LLM guide: [`llms.txt`](llms.txt) · agent instructions: [`AGENTS.md`](AGENTS.md)

## Repository map

| Path | What |
|------|------|
| [`INSTALL.md`](INSTALL.md) | **Step-by-step self-host install runbook** — written to be executable by a person or an AI agent |
| [`AGENTS.md`](AGENTS.md) | Operating instructions for AI agents working in this repo |
| [`livekit_agent/`](livekit_agent/) | **The production LiveKit media-plane agent** — runtime modules byte-identical to what OpenPhonex production runs (enforced by release tests) + a minimal pinned Dockerfile |
| [`packages/openphonex-gateway/deploy/`](packages/openphonex-gateway/deploy/SELFHOST.md) | The self-host stack: compose file, config templates, `SELFHOST.md` runbook, `selfhost-smoketest.sh` |
| [`packages/openphonex-gateway/`](packages/openphonex-gateway/) | Shared realtime pipeline / provider library (streaming STT·LLM·TTS primitives, audio policy, call evidence) used by the stack's tests |
| [`packages/openphonex-common/`](packages/openphonex-common/) | Control-plane-free shared config + runtime helpers |
| [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) | The exact gateway↔Cloud API surface (stable v1.0.0) |
| [`contracts/schemas/`](contracts/schemas/) | JSON Schemas for the stable request/response bodies |

## Quickstart (self-host — matches production)

**Follow [`INSTALL.md`](INSTALL.md)** — a linear runbook (prerequisites →
configure → `docker compose up` → provision → smoke test → test call) with
every environment variable documented (what it is, where to get it, example).
The stack ships a self-check:

```bash
cd packages/openphonex-gateway/deploy
bash selfhost-smoketest.sh   # PASS/FAIL per check; PASS = call-ready
```

Pinned components (tag **and** digest; livekit-server and livekit-sip match
the production voice box as of 2026-08-11, redis intentionally does not —
production is on 7.4.5-alpine): livekit-server v1.13.5, livekit-sip v1.7.0, redis 8.8.0,
livekit-agents 1.6.6 (+ Deepgram/Google/ElevenLabs/Silero/turn-detector
plugins, pip-pinned in
[`livekit_agent/agent/requirements.txt`](livekit_agent/agent/requirements.txt)).

Per-call agent config — system prompt, models, voice, and the tool
allowlist — resolves live from the control plane
(`POST /v1/calls/inbound/route` → `GET /v1/agents/{id}/realtime-config`);
the agent admits **no caller audio** until that authorization completes.
Background and troubleshooting:
[`packages/openphonex-gateway/deploy/SELFHOST.md`](packages/openphonex-gateway/deploy/SELFHOST.md).

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/openphonex-common -e packages/openphonex-gateway

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
[`SECURITY.md`](SECURITY.md) for the token model (gateway `tgw_` token vs the
organization-scoped worker token).

## More

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`SECURITY.md`](SECURITY.md)
- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)

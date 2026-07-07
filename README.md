# OpenPhonex Gateway

A self-hostable **media gateway** for [OpenPhonex](https://openphonex.com): RTP
transport + a realtime STT → LLM → TTS voice pipeline + a standalone Asterisk
ARI originate CLI. It registers to **OpenPhonex Cloud** on a scoped, per-gateway
token and runs your choice of providers — including fully **local models**
(faster-whisper + Ollama/vLLM + Piper/Kokoro) so no built-in provider API keys
are required.

This is **topology B**: numbers, KYB/compliance, billing, and abuse policy stay
in OpenPhonex Cloud; the media plane runs on your infrastructure. **No cloud
secrets ever live on the gateway** — it holds only a narrowly-scoped `tgw_`
token.

- License: **Apache-2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE))
- Brand: [`TRADEMARK.md`](TRADEMARK.md) — the code is open; "OpenPhonex" is not licensed
- Contract: [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) (v1.0.0, stable)
- Agent/LLM guide: [`llms.txt`](llms.txt)

## Repository map

| Path | What |
|------|------|
| [`packages/openphonex-gateway/`](packages/openphonex-gateway/) | Media server, RTP transport, realtime pipeline, provider registry (incl. local adapters), `openphonex-originate` CLI |
| [`packages/openphonex-common/`](packages/openphonex-common/) | Control-plane-free shared config + runtime helpers |
| [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) | The exact gateway↔Cloud API surface (stable v1.0.0) |
| [`contracts/schemas/`](contracts/schemas/) | JSON Schemas for the stable request/response bodies |
| [`configs/asterisk/`](configs/asterisk/) | Templated, secret-free Asterisk configs (Apache-2.0 config data) |
| [`docker-compose.gateway.yml`](docker-compose.gateway.yml) | Topology-B quickstart (gateway + Asterisk, self-contained) |
| [`packages/openphonex-gateway/deploy/`](packages/openphonex-gateway/deploy/SELFHOST.md) | Self-host quickstart: gateway-only container joining YOUR existing Asterisk (standalone, no Cloud) |

## Quickstart (self-host, existing Asterisk)

Already running your own Asterisk container? The gateway can join its docker
network as a single lean container and serve your Stasis app — fully
standalone, zero OpenPhonex Cloud calls. Follow
[`packages/openphonex-gateway/deploy/SELFHOST.md`](packages/openphonex-gateway/deploy/SELFHOST.md).

## Quickstart (topology B)

1. **In OpenPhonex Cloud**, create a self-hosted gateway deployment:
   `POST /v1/gateway-deployments` with
   `deployment_model_id=managed_numbers_self_hosted_gateway` → returns a `gw_...`
   id (status `awaiting_connection`).
2. **Issue a per-gateway token**:
   `POST /v1/gateway-deployments/{gw}/tokens` → returns a `tgw_...` token (shown
   once; default scopes `gateways:report, worker_jobs:enqueue, artifacts:upload,
   number-rules:read`). Paste it into `.env.gateway`.
3. `cp .env.gateway.example .env.gateway` and fill in `OPENPHONEX_GATEWAY_ID`,
   `OPENPHONEX_GATEWAY_API_TOKEN`, the SIP trunk vars, and the ARI password.
4. `docker compose -f docker-compose.gateway.yml up -d --build`
5. Pull models: `ollama pull llama3`; download a Piper voice into `./models`;
   faster-whisper self-downloads `base` on first use.
6. `curl http://127.0.0.1:18772/status`. The worker POSTs to
   `/v1/gateway-deployments/{gw}/health-report` with the `tgw_` token and the
   deployment flips `awaiting_connection` → `active` in the dashboard.
7. The gateway polls `GET /v1/gateway-deployments/{gw}/number-rules`
   (ETag / `If-None-Match`) for routing.
8. **Only after** the offline validation gate is green and an operator approves,
   set `REALTIME_LIVE_ENABLED=1` to place real calls.

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/openphonex-common -e packages/openphonex-gateway
pip install -e "packages/openphonex-gateway[local]"   # optional local models

python -m unittest discover -s tests                  # gateway tests, no cloud
openphonex-originate +15555550123 --dry-run           # print ARI payload, no call
python -m openphonex_gateway.realtime.worker --offline-synthetic-call --offline-output-dir /tmp/df
```

## Providers (the extension point)

Bring your own STT/LLM/TTS by advertising a `ProviderSpec` — via a Python entry
point (`openphonex.stt` / `openphonex.llm` / `openphonex.tts`), a dotted-path env
value (`REALTIME_STT_PROVIDER=myco.stt:Spec`), or a module `PROVIDER_SPEC`. See
[`llms.txt`](llms.txt) and
[`packages/openphonex-gateway/openphonex_gateway/realtime/registry.py`](packages/openphonex-gateway/openphonex_gateway/realtime/registry.py).

Shipped local adapters: **faster-whisper** (STT), **Ollama/vLLM** via an
OpenAI-compatible client (LLM), **Piper** and **Kokoro** (TTS). Optional extras:
`openphonex-gateway[local-stt|local-tts|local-llm|local]`.

## Safety

Live calls are **fail-closed** behind `REALTIME_LIVE_ENABLED` (and optional
`REALTIME_REQUIRE_OPERATOR_APPROVAL`). Asterisk ARI is never exposed publicly.
The gateway holds only the scoped `tgw_` token. Audio is `pcm_s16le_8000_mono`
(16-bit LE PCM, 8 kHz, mono; no negotiation).

## More

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`SECURITY.md`](SECURITY.md)
- [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)
- Asterisk (GPLv2) posture: see [`NOTICE`](NOTICE) — we distribute zero GPL code;
  Asterisk is obtained by you from upstream, and we talk to it only over ARI.

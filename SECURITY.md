# Security policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
suspected vulnerability.

- Use GitHub's private "Report a vulnerability" advisory flow on this repository, or
- email the maintainers (see the repository profile).

We aim to acknowledge reports within a few business days and will coordinate a
fix and disclosure timeline with you.

## Scope and threat model

The OpenPhonex Gateway is the **self-hosted LiveKit media plane**: call audio
terminates on the operator's host, while the OpenPhonex control plane stays
authoritative for routing, per-call config, tool allowlists, and billing. The
stack holds two bearer credentials with very different scopes — the narrow
per-gateway token (`tgw_...`; default scopes `gateways:report,
worker_jobs:enqueue, artifacts:upload, number-rules:read`) and the
organization-scoped worker token described next. No other cloud secret ever
lives on the gateway host.

## Worker token (self-host stack)

The LiveKit stack additionally mounts `OPENPHONEX_WORKER_API_TOKEN` — an
**organization-scoped** token used for inbound call routing
(`POST /v1/calls/inbound/route`) and MCP schema discovery (`POST /mcp`
`tools/list`; model calls are call-bound). This **mirrors production**:
routing and tools are organization operations, so the scope is deliberate, not
an oversight. The practical consequence:

The guided self-host baseline grants only `calls:read` + `calls:write`. It
deliberately omits the global `worker_jobs:*` authority; the self-host compose disables the
non-critical global worker-heartbeat loop and does not enable the optional
outbound-worker profile with this key.

- **A compromised gateway box holds organization-wide call/tool authority.**
  Treat the host as sensitive infrastructure: minimal shell access, disk
  encryption where applicable, `chmod 600` on `.env.livekit` (the quickstart
  does this), and no reuse of the box for unrelated workloads.
- **Rotate immediately on any suspicion of compromise** (mint a new worker
  token in the control plane, update `.env.livekit`, restart the `agents` and
  `reconciler` services, revoke the old token).
- The per-gateway `tgw_` token in the same file stays narrowly scoped; do not
  substitute one for the other.
- Narrowing the worker credential to per-call tokens is planned control-plane
  work; until then this section is the operative guidance.

## Control-plane transport

The agent **refuses** to send its bearer tokens over plaintext http to
anything but verified loopback (`127.0.0.0/8`, `::1`, `localhost`), and
refuses to follow cross-origin redirects while authenticated. Point
`OPENPHONEX_CONTROL_PLANE_URL` at an `https://` URL for any remote control
plane; this is enforced at runtime, not just documented.

Security-relevant boundaries:

- **Fail-closed media admission.** The agent joins each inbound room
  unsubscribed and consumes no caller audio until the control plane resolves
  the number route and returns a validated realtime-config.
- **Tools are allowlisted.** The model sees only tools named in the resolved
  config; every invocation crosses the control plane again with an audit trail.
- **Loopback-only internals.** LiveKit's API (7880) and Redis (6380) bind
  loopback; only SIP signaling + RTP are exposed, and the quickstart firewalls
  them to the carrier's published ranges.
- **Cloud-scoped policy stays in Cloud.** Numbers, KYB/compliance, billing, and
  abuse policy are NOT on the gateway.

## Credentials that DO live on the host

The self-host box holds several secrets in `.env.livekit` (and
`livekit.yaml` / `sip.yaml`). Keep the files `chmod 600`, do not reuse the box
for unrelated workloads, and rotate on any suspicion of compromise:

| Credential | Purpose | Rotate by |
|---|---|---|
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit admin API pair (also in `livekit.yaml` / `sip.yaml`) | generate a new pair (`openssl rand -hex 32`), update all three files, restart the stack |
| `OPENPHONEX_GATEWAY_API_TOKEN` (`tgw_...`) | per-gateway control-plane token (narrow scope) | `POST /v1/gateway-deployments/{id}/tokens/rotate` (old token revoked) |
| `OPENPHONEX_WORKER_API_TOKEN` (`tai_...`) | **organization-scoped** worker token (inbound routing, call-bound config, and MCP `tools/list` schema discovery) — highest-value secret here | mint a new org worker token, update `.env.livekit`, restart `agents`+`reconciler`, revoke the old one |
| `DEEPGRAM_API_KEY` / `GEMINI_API_KEY` / `ELEVENLABS_API_KEY` | AI-provider keys (STT/LLM/TTS) — billed to your provider accounts | rotate in each provider's console |
| `DIDWW_OUTBOUND_AUTH_USERNAME` / `DIDWW_OUTBOUND_AUTH_PASSWORD` (optional) | outbound SIP trunk auth, only if you provision outbound | rotate in your carrier portal |

The worker token is the one to guard most closely (org-wide authority); the
`tgw_` token is narrowly scoped by design. If the host is compromised, assume
ALL of the above are exposed and rotate every one.

## Do not commit secrets

This repository is published from a private monolith by an automated export that
runs a stdlib secret-scanner over the entire staged tree before any publish is
permitted. If you contribute, keep every credential and public address as an
`${ENV}` reference — never a literal.

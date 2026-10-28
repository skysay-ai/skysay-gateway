# Security policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
suspected vulnerability.

- Use GitHub's private "Report a vulnerability" advisory flow on this repository, or
- email the maintainers (see the repository profile).

We aim to acknowledge reports within a few business days and will coordinate a
fix and disclosure timeline with you.

## Scope and threat model

The Skysay Gateway is the **self-hosted LiveKit media plane**: call audio
terminates on the operator's host, while the Skysay control plane stays
authoritative for routing, per-call config, tool allowlists, and billing.
The gateway runs on your servers and carries the call's audio. From there, audio and text go to your carrier and to the speech and model providers your agent uses. Skysay Cloud receives the call's metadata; during the call, the arguments of each tool call, any knowledge-base question the agent sends and, when the agent has tools that write to your systems, what the caller says; after the call, the recording and transcript.

The only long-lived control-plane credential you configure is the deployment's `skygw_` gateway token
(plus the organization worker key if you keep the legacy `queue` outbound mode, described next). During a
call the gateway also holds a call authorization (the single-use `tca_` of an outbound call) and any provider
keys leased to it (one Skysay issues for your workspace, or one you stored in Skysay); a leased provider key
does not expire with the call.
The gateway token is narrow (`skygw_...`; default scopes `gateways:report,
worker_jobs:enqueue, artifacts:upload, number-rules:read, calls:authorize`) and
covers inbound AND outbound once the deployment is in `local` outbound mode.

## What the one credential can reach

Inbound call routing (`POST /v1/calls/inbound/route`), the resulting call's
runtime-config read, and MCP schema discovery (`POST /mcp` `tools/list`) all
run on the deployment-bound `skygw_` token. Each of them derives its organization
**server-side from the token row**; nothing this host sends can name a
workspace, so a stolen token acts in exactly the workspace its deployment
belongs to.

Two further limits are worth knowing, because they are what keeps "one
credential" from meaning "one skeleton key":

- `tools/list` is the **only** MCP method a gateway token may call. It returns
  the static tool catalogue — names, descriptions, input schemas — and reads no
  workspace row. `tools/call`, `resources/read` and the rest stay refused, so
  this token can never send an SMS or read a recording through MCP.
- The per-token `call.inbound_route` **kind allowlist** entry is a real switch.
  Mint a token without it and that deployment cannot route calls at all, which
  is how you quarantine a suspect box without touching the rest of the
  workspace.

Still treat the host as sensitive infrastructure: minimal shell access, disk
encryption where applicable, `chmod 600` on `.env.livekit` (the quickstart does
this), and no reuse of the box for unrelated workloads. **Rotate immediately on
any suspicion of compromise** —
`POST /v1/gateway-deployments/{id}/tokens/rotate`, update `.env.livekit`,
restart the `agents` and `reconciler` services.

## Outbound: local origination, and the legacy queue mode

A deployment is in one of two outbound modes
(`outbound_execution_mode` on `GET /v1/gateway-deployments/{id}`).

**`local` — no organization-scoped key at all.** The box asks the control plane
to admit the call it is about to place
(`POST /v1/gateway-deployments/{id}/outbound-calls`, on the same `skygw_` token)
and dials it with the single-use `tca_` that request returns. The organization
is read from the token row, the outbound authorization decision, caps, kill
switches and the wallet reserve all run server-side first, and the token is
bound to one (gateway, organization, from-number, to-number, call). A
compromise of this host therefore reaches outbound calling **for this
deployment's workspace only**, under the same policy engine as any other
request — and no farther.

**`queue` — the legacy mode, and the one organization-scoped key.**
`--profile outbound` reads `SKYSAY_WORKER_API_TOKEN`, an
**organization-scoped** `sky_` key, because claiming
`call.outbound_originate` jobs somebody else requested is `worker_jobs:*`
authority over the whole workspace queue, which a deployment-bound token
deliberately does not carry and must not be given. If you keep this mode, do so
knowingly: **a compromise of the box then reaches organization-wide outbound
calling**, and rotation means minting a new key in the control plane, updating
`.env.livekit`, restarting the `outbound-worker` service, and revoking the old
one.

Switching to `local` and deleting the `SKYSAY_WORKER_API_TOKEN` line is
what actually narrows the box.

## Control-plane transport

The agent **refuses** to send its bearer tokens over plaintext http to
anything but verified loopback (`127.0.0.0/8`, `::1`, `localhost`), and
refuses to follow cross-origin redirects while authenticated. Point
`SKYSAY_CONTROL_PLANE_URL` at an `https://` URL for any remote control
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
| `SKYSAY_GATEWAY_API_TOKEN` (`skygw_...`) | the one long-lived control-plane token you configure: deployment-bound, organization derived server-side (number-rules sync, inbound routing, call-bound config, MCP `tools/list`, lifecycle, artifacts) | `POST /v1/gateway-deployments/{id}/tokens/rotate` (old token revoked) |
| `SKYSAY_WORKER_API_TOKEN` (`sky_...`) (legacy `queue` outbound mode only) | **organization-scoped** key read only by the job-claiming `--profile outbound` worker — highest-value secret here when present; unset on a `local`-mode deployment | mint a new org worker key, update `.env.livekit`, restart `outbound-worker`, revoke the old one |
| `OPENAI_API_KEY` / `DEEPGRAM_API_KEY` / `GEMINI_API_KEY` / `ELEVENLABS_API_KEY` / `SONIOX_API_KEY` / `FISH_AUDIO_API_KEY` / `CARTESIA_API_KEY` / `AZURE_SPEECH_KEY` | AI-provider keys (STT/LLM/TTS) — billed to your provider accounts. Which of them the host holds depends on the languages your agents speak; every one present is billable | rotate in each provider's console |
| `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_APPLICATION_CREDENTIALS_JSON` (optional) | GCP **service-account private key** for Cloud Speech-to-Text, only if an agent uses Google Chirp 2 (Sinhala). A service-account key is not an API key: it can be used against whatever else that account is granted | GCP Console → IAM & Admin → Service Accounts → Keys: add a new key, update the file/variable, restart `agents`, then **delete** the old key |
| `DIDWW_OUTBOUND_AUTH_USERNAME` / `DIDWW_OUTBOUND_AUTH_PASSWORD` (optional) | outbound SIP trunk auth, only if you provision outbound | rotate in your carrier portal |

If this deployment still runs the legacy `queue` outbound mode, its `sky_` key
is the one to guard most closely (org-wide authority); the `skygw_` token is
deployment-bound by design, and in `local` outbound mode it is the long-lived
credential you configure; during a call the gateway also holds a call authorization and any provider keys leased to it.
If the host is compromised, assume ALL of the above are exposed and rotate
every one.

## Do not commit secrets

This repository is published from a private monolith by an automated export that
runs a stdlib secret-scanner over the entire staged tree before any publish is
permitted. If you contribute, keep every credential and public address as an
`${ENV}` reference — never a literal.

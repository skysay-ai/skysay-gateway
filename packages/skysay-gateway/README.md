# skysay-gateway

The Skysay gateway support package contains caller-side voice-evidence
utilities and portable RTP codecs, plus the **self-host deploy stack** under
[`deploy/`](deploy/). The production media plane is LiveKit Agents; this
package does not provide a second call runtime.

**Self-hosting the media plane?** Start with
[`deploy/SELFHOST.md`](deploy/SELFHOST.md) (step-by-step form: the repo-root
`INSTALL.md` in the public export). The stack
([`deploy/docker-compose.livekit.yml`](deploy/docker-compose.livekit.yml))
runs the Skysay agent code (shipped at `livekit_agent/` in the public repo
layout, generated from the Skysay source with documented omissions,
substitutions and export transformations), pinned livekit-server/livekit-sip
versions, and
per-call config resolved from the control plane
(`prepare_inbound` → realtime-config, `contracts/gateway-v1.yaml`). After
`docker compose up`, verify with
[`deploy/selfhost-smoketest.sh`](deploy/selfhost-smoketest.sh).

This package is deliberately import-light at its root and depends only on
`skysay-common` (control-plane-free config/runtime helpers) — never on the
private control plane. The evidence referee is explicitly available at
`skysay_gateway.evidence`, and the console command remains
`skysay-voice-evidence`.

## 0.2 migration

`skysay_gateway.realtime` was removed in 0.2. It contained an obsolete
pre-LiveKit pipeline, provider/plugin registry, and transports with no runtime
consumer. No compatibility namespace is retained for that deleted runtime:
migrate caller-side referee imports to `skysay_gateway.evidence`; run calls
through the LiveKit agent stack instead.

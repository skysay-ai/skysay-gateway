# openphonex-gateway

The OpenPhonex media-plane support package: the shared realtime voice-pipeline
library (streaming STT / LLM / TTS primitives, turn detection, semantic audio
policy, call evidence) and the home of the **self-host deploy stack** under
[`deploy/`](deploy/).

**Self-hosting the media plane?** Start with
[`deploy/SELFHOST.md`](deploy/SELFHOST.md) (step-by-step form: the repo-root
`INSTALL.md` in the public export). The stack
([`deploy/docker-compose.livekit.yml`](deploy/docker-compose.livekit.yml))
mirrors OpenPhonex production exactly: the same agent code (shipped at
`livekit_agent/` in the public repo layout, byte-identical to the internal
source of truth), the same pinned livekit-server/livekit-sip versions, and
per-call config resolved from the control plane
(`prepare_inbound` → realtime-config, `contracts/gateway-v1.yaml`). After
`docker compose up`, verify with
[`deploy/selfhost-smoketest.sh`](deploy/selfhost-smoketest.sh).

This package is deliberately import-light at its root and depends only on
`openphonex-common` (control-plane-free config/runtime helpers) — never on the
private control plane. `import openphonex_gateway` must not pull in the
`realtime` / `pipeline` submodules, so the package stays cheap to import and
free of heavy provider dependencies at module load.

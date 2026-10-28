# AGENTS.md — operating instructions for AI agents

You are working in the **Skysay Gateway** repository: the self-hostable
media plane for Skysay voice agents. This file tells you how to do the
common jobs here without guessing.

## If your task is: install / deploy the self-hosted stack

Follow **[`INSTALL.md`](INSTALL.md)** exactly, in order. It is written for
you: every step has a command, an expected result, and a failure pointer.
Rules:

1. **Do not skip verification steps.** After configuration and after `up`,
   run `bash packages/skysay-gateway/deploy/selfhost-smoketest.sh` and do
   not proceed past a `RESULT: FAIL` — each `[FAIL]` line contains its fix.
2. **Collect the step-0 prerequisites first** (DID, carrier IP ranges,
   control-plane ids/tokens, provider keys). If any is missing, stop and ask
   the operator for exactly that item — nothing else is needed.
3. **Secrets go only into** `packages/skysay-gateway/deploy/`
   `{.env.livekit, livekit.yaml, sip.yaml}` (chmod 600). Never commit them —
   `.gitignore` already excludes them; the export secret-scanner enforces it.
4. Every environment variable is documented (what / where to get it /
   example) in `INSTALL.md` step 3 and inline in `.env.livekit.example`.
   There are no undocumented variables.
5. The inbound success criterion is **step 8**: a real phone call to the DID is
   answered and `call_complete` appears in the agents log.

## If your task is: place an outbound call

First find out which originate path this deployment is on. Read
`outbound_execution_mode` from `GET /v1/gateway-deployments/{id}` with the
gateway token; it is one of two values and they need different things of you.

- `local` — this box dials its own calls, and it holds no organization-scoped
  key. Ask the box to dial:

  ```bash
  cd packages/skysay-gateway/deploy
  docker compose -f docker-compose.livekit.yml run --rm outbound-worker \
    python /app/outbound_worker.py originate \
      --to +<E164> --purpose <purpose> --reason "<why this call>"
  ```

  It prints one `SKYSAY_OUTBOUND {...}` line with the call id, or
  `SKYSAY_OUTBOUND_FAILED {...}`. `--from` defaults to the first number in
  `SCH236_TEST_DID_E164`; `--agent-id` defaults to that number's own agent.
  `--purpose` and `--reason` are required and are evaluated by the control
  plane's outbound policy — state what is true, do not invent one to get past
  a refusal. `SKYSAY_EXECUTE_LIVEKIT_OUTBOUND=1` must be set in
  `.env.livekit`; the command refuses otherwise. A remote `POST /v1/calls` for
  this workspace's numbers is refused with a message naming this command.

- `queue` — outbound calls are requested from the control plane with an
  organization-scoped API key (`POST /v1/calls`), and this box's
  `--profile outbound` worker claims and dials them. Check the worker with
  `python /app/outbound_worker.py --status` before enabling its execution gate.

Either way, the control plane runs policy, caps, kill switches and the wallet
reserve before anything dials, and this box cannot reach media without the
single-use `tca_` it mints. Do not try to route around a refusal.

## If your task is: understand or modify the code

- Repo map and architecture: [`README.md`](README.md), machine-oriented
  summary: [`llms.txt`](llms.txt).
- `livekit_agent/` is the media-plane agent, generated from the Skysay
  source with documented omissions, substitutions and export
  transformations. Treat changes there as production changes:
  conservative, tested, no new env vars without documenting them in the
  self-host templates.
- `packages/skysay-gateway/` is the caller-side evidence referee plus
  portable RTP codecs; `packages/skysay-common/` supplies control-plane-
  free config/runtime helpers. Do not add a call runtime or provider registry
  to the portable gateway package.
- The gateway ↔ control-plane API surface is
  [`contracts/gateway-v1.yaml`](contracts/gateway-v1.yaml) (stable v1.0.0) +
  [`contracts/schemas/`](contracts/schemas/). Do not invent endpoints.

Run the test suite before and after changes:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/skysay-common -e packages/skysay-gateway numpy==2.5.3
python -m unittest discover -s tests
```

## Security boundaries you must preserve

- The control-plane transport requires **https** (or verified loopback http)
  and refuses cross-origin redirects — never weaken this.
- `SKYSAY_GATEWAY_API_TOKEN` is the one long-lived credential you configure: never print it,
  never write it anywhere except `.env.livekit`, and recommend rotation if it
  may have leaked (see [`SECURITY.md`](SECURITY.md)). It is deployment-bound —
  every organization it reaches is derived server-side from the token — so do
  not "fix" an authorization failure by substituting an organization-scoped
  `sky_` key. `SKYSAY_WORKER_API_TOKEN` is organization-scoped and is read
  ONLY by the legacy `queue` outbound mode's job-claiming worker; the same
  handling rules apply, more strictly. On a `local`-mode deployment it should
  not be set at all — adding it back does not fix anything and widens what a
  compromise of this host reaches.
- Live-call behavior is fail-closed; do not bypass authorization steps
  (`prepare_inbound` → realtime-config) in `livekit_agent/`.

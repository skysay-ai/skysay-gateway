# Contributing to Skysay Gateway

Thanks for your interest in the Skysay Gateway.

## Licensing of contributions

By submitting a contribution you agree it is licensed under the project's
**Apache License, Version 2.0** (see `LICENSE`), per Apache-2.0 §5. Do not submit
code you do not have the right to license this way.

Please mark modified files as changed (Apache-2.0 §4(b)) and keep the `NOTICE`
attribution intact.

## Scope of this repository

This repository is the **media plane** only — the self-hostable LiveKit stack
(`livekit_agent/` + `packages/skysay-gateway/deploy/`), the portable
evidence/RTP package, and the gateway↔Cloud contract. The
Skysay Cloud control plane (numbers, KYB/compliance, billing, abuse
policy) is a separate, private system and is intentionally out of scope here.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/skysay-common -e packages/skysay-gateway numpy==2.5.3

# run the tests (no control plane, no network, no providers required):
python -m unittest discover -s tests
```

## Never commit secrets

Every credential and public address must be an `${ENV}` reference. The upstream
project publishes this repo through an automated export that runs a secret-scan
over the whole tree and **refuses to publish on any finding**. Keep it clean.

## The gateway↔Cloud contract

`contracts/gateway-v1.yaml` (v1.0.0, stable) plus `contracts/schemas/*.json` are
a **backwards-compatibility guarantee**. Changes to the stable surface must be
additive. Open an issue before proposing a contract change.

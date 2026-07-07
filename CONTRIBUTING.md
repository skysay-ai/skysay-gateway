# Contributing to OpenPhonex Gateway

Thanks for your interest in the OpenPhonex Gateway.

## Licensing of contributions

By submitting a contribution you agree it is licensed under the project's
**Apache License, Version 2.0** (see `LICENSE`), per Apache-2.0 §5. Do not submit
code you do not have the right to license this way.

Please mark modified files as changed (Apache-2.0 §4(b)) and keep the `NOTICE`
attribution intact.

## Scope of this repository

This repository is the **gateway** only — the RTP transport, the realtime
STT/LLM/TTS pipeline, the provider plugin registry (with local-model adapters),
the standalone ARI originate CLI, the gateway↔Cloud contract, and the Asterisk
config templates. The OpenPhonex Cloud control plane (numbers, KYB/compliance,
billing, abuse policy) is a separate, private system and is intentionally out of
scope here.

## Provider plugins (the extension point)

The cleanest way to extend the gateway is a **provider plugin**. Advertise a
`ProviderSpec` (see `packages/openphonex-gateway/openphonex_gateway/realtime/registry.py`)
in one of three ways:

1. a Python entry point in group `openphonex.stt` / `openphonex.llm` / `openphonex.tts`;
2. a dotted-path env value, e.g. `REALTIME_STT_PROVIDER=myco.stt:Spec`; or
3. a module-level `PROVIDER_SPEC` attribute.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/openphonex-common -e packages/openphonex-gateway
# optional local-model adapters:
pip install -e "packages/openphonex-gateway[local]"

# run the gateway tests (no control plane, no network, no providers required):
python -m unittest discover -s tests

# offline smokes (no calls, no network):
python -m openphonex_gateway.realtime.worker --offline-synthetic-call --offline-output-dir /tmp/df
openphonex-originate +15555550123 --dry-run
```

## Never commit secrets

Every credential and public address must be an `${ENV}` reference. The upstream
project publishes this repo through an automated export that runs a secret-scan
over the whole tree and **refuses to publish on any finding**. Keep it clean.

## The gateway↔Cloud contract

`contracts/gateway-v1.yaml` (v1.0.0, stable) plus `contracts/schemas/*.json` are
a **backwards-compatibility guarantee**. Changes to the stable surface must be
additive. Open an issue before proposing a contract change.

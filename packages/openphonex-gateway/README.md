# openphonex-gateway

The OpenPhonex media gateway: RTP transport and the realtime voice pipeline
(STT / LLM / TTS / turn-detection) plus a standalone, control-plane-free ARI
originate CLI for self-hosters.

This package is deliberately import-light at its root and depends only on
`openphonex-common` (control-plane-free config/runtime helpers) — never on the
private control plane (`telephony_ai.app` / `server` / `store`).

During the M4 extraction the modules physically live here while the legacy
import name `telephony_ai.media.*` keeps resolving to them via a thin
`__path__`-redirect shim in `src/telephony_ai/media/__init__.py`, so the
monolith and its tests are unchanged.

## Originate CLI

```bash
python -m openphonex_gateway.originate +37253231038 --env-file /etc/openphonex/asterisk.env --dry-run
# or, after `pip install -e packages/openphonex-gateway`:
openphonex-originate +37253231038 --env-file /etc/openphonex/asterisk.env --dry-run
```

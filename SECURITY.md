# Security policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
suspected vulnerability.

- Use GitHub's private "Report a vulnerability" advisory flow on this repository, or
- email the maintainers (see the repository profile).

We aim to acknowledge reports within a few business days and will coordinate a
fix and disclosure timeline with you.

## Scope and threat model

The OpenPhonex Gateway is a **self-hosted media gateway**. It holds only a
**scoped, per-gateway token** (`tgw_...`) — never an organization API key
(`tapi_...`) and never any cloud provider secret. The token's default scopes are
narrow: `gateways:report, worker_jobs:enqueue, artifacts:upload,
number-rules:read`.

Security-relevant boundaries:

- **Fail-closed by default.** Live calls are disabled until `REALTIME_LIVE_ENABLED=1`
  is explicitly set (optionally gated further by `REALTIME_REQUIRE_OPERATOR_APPROVAL`).
- **ARI is never public.** The Asterisk ARI HTTP/WebSocket port (8088) must be
  bound to the private compose network only, never to a public interface.
- **No cloud secrets on the gateway.** Numbers, KYB/compliance, billing, and
  abuse policy stay in OpenPhonex Cloud.
- **Rotate on suspicion.** If a `tgw_` token is exposed, rotate it via
  `POST /v1/gateway-deployments/{id}/tokens/rotate` and the old token is revoked.

## Do not commit secrets

This repository is published from a private monolith by an automated export that
runs a stdlib secret-scanner over the entire staged tree before any publish is
permitted. If you contribute, keep every credential and public address as an
`${ENV}` reference — never a literal.

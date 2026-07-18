# openphonex-common

Control-plane-free shared primitives for OpenPhonex: environment/config loading
(`config`) and runtime helpers such as structured logging, secret redaction and
readiness probes (`runtime`).

Pure standard library, no dependency on the OpenPhonex control plane. During the
extraction into this package the legacy import names `telephony_ai.config` and
`telephony_ai.runtime` alias to `openphonex_common.config` / `.runtime` via
`sys.modules` shims, preserving a single module identity so existing
`mock.patch("telephony_ai.config...")` call sites keep working unchanged.

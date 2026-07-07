# Third-party licenses

The OpenPhonex Gateway is licensed under Apache-2.0 (see `LICENSE`). It depends
on the third-party components listed below. **None of these are bundled, forked,
modified, or redistributed by this repository** — they are declared as
dependencies (or invoked as external binaries) and obtained by the operator from
their respective upstreams at install/build/run time.

## Runtime dependencies

| Component            | License        | How it is used                                                        | Bundled? |
|----------------------|----------------|-----------------------------------------------------------------------|----------|
| websocket-client     | Apache-2.0     | Deepgram live streaming STT transport (`openphonex-gateway` core dep)  | No (pip) |
| faster-whisper       | MIT            | Optional local STT adapter (`[local-stt]` extra)                      | No (pip) |
| numpy                | BSD-3-Clause   | Optional PCM resample helper for local models (`[local-*]` extras)    | No (pip) |
| Piper                | MIT            | Optional local TTS engine — invoked as an external `piper` binary     | No       |
| Kokoro               | Apache-2.0     | Optional local TTS engine (`[local-tts]` extra)                       | No (pip) |
| ffmpeg               | LGPL-2.1 / GPL | Optional audio transcode — invoked as an external binary if present   | No       |

The core gateway and its test suite never require the optional local-model
extras; each local adapter is lazily imported behind a `ProviderSpec` and only
loaded when explicitly selected via `REALTIME_STT_PROVIDER` /
`REALTIME_LLM_PROVIDER` / `REALTIME_TTS_PROVIDER`.

## External services (descriptive references only)

The built-in cloud adapters (`deepgram`, `gemini`, `elevenlabs`) call the
respective vendor HTTP/WebSocket APIs when the operator supplies their own API
keys. No vendor SDK or key is bundled. "DIDWW", "Deepgram", "Gemini",
"ElevenLabs", "Ollama", and "vLLM" are trademarks of their respective owners and
are referenced descriptively.

## Asterisk (GPLv2)

Asterisk is licensed under **GPLv2 with the OpenSSL linking exception**.
OpenPhonex does **not** fork, bundle, modify, statically or dynamically link, or
redistribute Asterisk or any Asterisk binary.

- `Dockerfile.asterisk` runs `apt-get install asterisk` from the operator's own
  Ubuntu package repository at the operator's build time. The operator therefore
  obtains Asterisk directly from upstream, under GPLv2. This repository ships no
  Asterisk code.
- The `configs/asterisk/*.conf` / `*.conf.template` files are **configuration
  data** authored by OpenPhonex (an interface), independently licensed
  Apache-2.0. Configuration files are not a derivative work of the program they
  configure.
- The Python gateway talks to Asterisk **only** over the ARI HTTP/WebSocket
  network API. A network protocol boundary is not "linking" and does not create
  a derivative work under the GPL.

Because OpenPhonex distributes no GPL code and interacts with Asterisk only
across a network boundary, GPLv2's copyleft does not extend to OpenPhonex code,
and Apache-2.0 is compatible with this deployment model.

## Full license texts

Apache-2.0 components are covered by the repository `LICENSE`. The MIT and
BSD-3-Clause texts for `faster-whisper`, `numpy`, and `Piper` are reproduced in
their respective upstream distributions; consult each package's `LICENSE` file
as installed in your environment. This project does not redistribute those
files because it does not redistribute those packages.

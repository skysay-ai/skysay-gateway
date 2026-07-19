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

The core packages and their test suite never require the optional local-model
extras; each local adapter is lazily imported behind a `ProviderSpec` and only
loaded when explicitly selected.

## Self-host stack components (obtained by the operator, not redistributed)

The compose file in `packages/openphonex-gateway/deploy/` runs the following
as separate, digest-pinned container images / pip packages that the operator
pulls directly from their upstreams:

| Component | License | How it is used | Bundled? |
|---|---|---|---|
| LiveKit server (`livekit/livekit-server`) | Apache-2.0 | WebRTC/media control plane of the stack | No (image) |
| LiveKit SIP (`livekit/sip`) | Apache-2.0 | SIP↔LiveKit bridge for carrier trunks | No (image) |
| livekit-agents + provider plugins | Apache-2.0 | The agent framework the media agent runs on | No (pip) |
| Redis (`redis:8.8`) | AGPLv3 / RSALv2 / SSPLv1 (upstream tri-license; AGPLv3 is OSI-approved as of Redis 8) | Host-local message bus for LiveKit | No (image) |

## External services (descriptive references only)

The production voice pipeline calls the Deepgram, Google (Gemini), and
ElevenLabs HTTP/WebSocket APIs with the operator's own API keys. No vendor SDK
or key is bundled. "LiveKit", "Redis", "DIDWW", "Deepgram", "Gemini",
"ElevenLabs", "Ollama", and "vLLM" are trademarks of their respective owners
and are referenced descriptively.

## Full license texts

Apache-2.0 components are covered by the repository `LICENSE`. The MIT and
BSD-3-Clause texts for `faster-whisper`, `numpy`, and `Piper` are reproduced in
their respective upstream distributions; consult each package's `LICENSE` file
as installed in your environment. This project does not redistribute those
files because it does not redistribute those packages.

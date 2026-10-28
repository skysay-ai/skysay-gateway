# Third-party licenses

The Skysay Gateway is licensed under Apache-2.0 (see `LICENSE`). It depends
on the third-party components listed below. **None of these are bundled, forked,
modified, or redistributed by this repository** — they are declared as
dependencies (or invoked as external binaries) and obtained by the operator from
their respective upstreams at install/build/run time.

## Gateway library dependencies

The portable `skysay-gateway` library declares only
`skysay-common`, another Skysay package included in this export. Its
caller-side evidence referee and RTP codec use the Python standard library; it
does not bundle provider adapters, local models, or a call runtime.

## Self-host stack components (obtained by the operator, not redistributed)

The compose file in `packages/skysay-gateway/deploy/` runs the following
as separate, digest-pinned container images / pip packages that the operator
pulls directly from their upstreams:

| Component | License | How it is used | Bundled? |
|---|---|---|---|
| LiveKit server (`livekit/livekit-server`) | Apache-2.0 | WebRTC/media control plane of the stack | No (image) |
| LiveKit SIP (`livekit/sip`) | Apache-2.0 | SIP↔LiveKit bridge for carrier trunks | No (image) |
| livekit-agents + provider plugins | Apache-2.0 | The agent framework the media agent runs on | No (pip) |
| Redis (`redis:8.8`) | AGPLv3 / RSALv2 / SSPLv1 (upstream tri-license; AGPLv3 is OSI-approved as of Redis 8) | Host-local message bus for LiveKit | No (image) |

## External services (descriptive references only)

The voice pipeline calls the speech and model providers the agent uses
(Deepgram, Google, OpenAI, ElevenLabs, Soniox, Fish Audio, Cartesia, Azure)
over their HTTP/WebSocket APIs, through the pip-installed livekit-agents
provider plugins and vendor SDKs listed in
`livekit_agent/agent/requirements.txt`. No API key is bundled. "LiveKit",
"Redis", "DIDWW" and the provider names are trademarks of their respective
owners and are referenced descriptively.

## Full license texts

Apache-2.0 components are covered by the repository `LICENSE`. This project
does not redistribute third-party Python packages or container images.

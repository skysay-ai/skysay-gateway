# Self-host quickstart

**This stack is the same media plane OpenPhonex production runs.** Follow it
top to bottom; every step has a copy-pasteable command and a verifiable
expected result. (An AI agent can execute this guide as written; the repo-root
`INSTALL.md` in the public export is the same runbook in checklist form.)

---

# LiveKit self-host (matches production)

Same agent code (shipped at [`livekit_agent/`](../../../livekit_agent/) in the
public repo — the runtime modules are byte-identical to the internal production
source, enforced by export tests), same pinned component versions, same
control-plane contract.

```
      PSTN / your carrier (SIP trunk, your DID)
                    |
        UDP 5060 + RTP 10000-10100
                    v
  [ livekit-sip v1.7.0 ] --- [ livekit-server v1.13.5 ] --- [ redis 8.8.0 ]
                    \                 |
                     \                | (agent dispatch, host-local)
                      v               v
                  [ agents: the production LiveKit agent ]
                  [ reconciler: stale-session cleanup    ]
                    |
                    |  HTTPS (bearer tokens; per-call config, tools,
                    v   lifecycle/billing events, artifact upload)
          [ OpenPhonex control plane (Cloud or self-hosted) ]
```

Every component below is digest-pinned. `livekit-server` and `livekit-sip`
match the production voice box tag-and-digest (verified against the running
containers on 2026-08-11); `redis` deliberately does not — production is still
on `7.4.5-alpine`. The export test enforces that this file and the internal
sandbox compose agree with **each other**; it cannot see the production host,
whose compose file is not tracked in this repository.

| Component | Version |
|---|---|
| `livekit/livekit-server` | `v1.13.5` (digest-pinned) |
| `livekit/sip` | `v1.7.0` (digest-pinned) |
| `redis` | `8.8.0-alpine` (digest-pinned) |
| `livekit-agents` + Deepgram/Google/ElevenLabs/Silero/turn-detector plugins | `1.6.6` (pip-pinned in `livekit_agent/agent/requirements.txt`) |
| Voice pipeline (default) | Deepgram nova-3 → Gemini → ElevenLabs (resolved per call from realtime-config) |
| Voice pipeline (si-LK / Sinhala) | Google Chirp 2 → Gemini → Gemini-TTS — see [Language stacks: Sinhala (si-LK)](#language-stacks-sinhala-si-lk) below |

## Choosing your inbound topology

Read this before step 0 — it decides two of the answers below.

A call reaches this stack in one of two ways. **The media path is identical in
both**: RTP flows directly between your carrier and `livekit-sip`. They differ
only in how the carrier's INVITE gets to you.

### Static endpoint (default)

Your host has a public IP. The carrier sends INVITEs to it, matched against the
trunk's address allowlist. Nothing extra runs. If you have a public IP, use
this — it has fewer moving parts and one less thing to keep alive.

### Registration (`--profile registrar`)

Your host has **no reachable public address** — home or office NAT, a dynamic
IP, or CGNAT. Instead of waiting to be called, a small sidecar REGISTERs
*outbound* to your carrier. That single outbound packet is what makes you
reachable: it opens a path through your NAT, and the carrier delivers the
INVITE back down it. LiveKit never learns it was behind NAT.

```
   your carrier  <--- REGISTER (outbound, refreshed + kept alive) ---.
        |                                                            |
        |  INVITE, back down that same path                          |
        v                                                            |
   [ your NAT ] ---> [ registrar sidecar :5062 ] --- proxied --.     |
                            (signalling only)                  |     |
                                                               v     |
                                          [ livekit-sip :5060 ] -----'
                                                   ^
                        RTP goes straight here, never through the sidecar
```

The sidecar is a SIP **proxy**, not a back-to-back user agent. It does not
touch, transcode, or relay audio; it does not appear in the SDP. The media path
is byte-for-byte the static-endpoint one.

**Two things decide whether registration mode works for you, and both are on
the carrier's side:**

1. **The carrier must send the INVITE to the address it observed you from**
   (the `received`/`rport` it recorded), not to the address literal in your
   registration's `Contact`. Behind NAT those differ and nothing on your side
   can know the former. Carriers that offer a registration/dynamic-endpoint
   product normally do this; it is the entire point of the product.
2. **The carrier must latch its outbound RTP onto the address it actually
   receives media from** — usually called *symmetric RTP*, sometimes a
   per-trunk switch you have to turn on. Your NAT rewrites the port this stack
   advertises in its SDP, so a carrier that trusts the advertised port will
   send audio into a black hole. Symptom: the call connects and stays silent.

If your carrier does neither, registration mode is not usable and you need a
public IP.

**What the sidecar does, in one paragraph.** It REGISTERs to your carrier and
re-registers before expiry. Separately — and this is the part that is easy to
miss — it sends a cheap keepalive every 25 seconds, because carriers enforce a
long minimum registration lifetime (600 seconds is common) while NAT UDP
bindings expire after 30–120 seconds. Without the keepalive the registration
stays green in the carrier's portal while calls quietly stop arriving. When an
INVITE does arrive it is accepted only from the carrier ranges you listed, then
proxied to `livekit-sip` on loopback with the SDP untouched.

### Which file holds what, and who reads it

There are two credential files and several readers. Getting this wrong has a
specific, nasty signature — everything reports healthy and no call arrives — so
the split is stated once, here, and enforced by the smoke test.

| | `.env.livekit` | `.env.registrar` |
|---|---|---|
| Holds | control-plane tokens, provider API keys, the DID | **only** `SIP_REGISTRATION_*` |
| Read by | `agents`, `reconciler`, `outbound-worker`, `provision`, the smoke test | `registrar`, `provision`, the smoke test |
| Created by | every deployment | every deployment (leave it blank if you have a public IP) |

**`SIP_REGISTRATION_*` must never appear in `.env.livekit`.** Not even as an
empty placeholder. The provision container receives `.env.registrar` as
environment and then sources `.env.livekit` over the top, so a leftover
`SIP_REGISTRATION_REGISTRAR=` line there wins and blanks the real value. The
provisioner then sees an empty block, quietly builds the *static-endpoint*
trunk with a carrier address allowlist, and every INVITE the sidecar forwards —
which arrives from this host, not from the carrier — is rejected by it. The
smoke test fails if it finds those variables in `.env.livekit`, so you will be
told rather than left to discover it on the first call.

If you set this stack up before the split, see "Upgrading" below.

### Carrier-side checklist for registration mode

Verified against a live carrier (DIDWW) with packet captures. Names differ per
carrier; the behaviours do not. The labels below are DIDWW's portal labels; for
the equivalent API attribute names see the API subsection that follows. In the
DIDWW portal the form is **Voice → Inbound Trunks → Create New → SIP Trunk**,
and the settings are spread across its **General**, **Authorization** and
**Media & DTMF** sections. An existing trunk is editable after creation
(**Actions → Edit**), so a static trunk does not have to be recreated.

| What to set | Why it matters |
|---|---|
| Trunk **Type: Dynamic Registration** (the other choice, **Static Endpoint**, is the default) | Switching it makes the carrier generate a dedicated REGISTER username/password. Those are the credentials this sidecar wants — not your portal login and not the outbound trunk auth. DIDWW generates them and displays them **immediately after the trunk is created**, so capture them then; they are your `SIP_REGISTRATION_USERNAME` and `SIP_REGISTRATION_PASSWORD`, and remain viewable later from the trunk's entry on the Inbound Trunks page. Your `SIP_REGISTRATION_REGISTRAR` is `sip.didww.com`, or a regional host such as `nyc.sip.didww.com` / `ams.sip.didww.com` to pin a location. DIDWW allows at most 10 simultaneous registrations per trunk. |
| Leave **Enable Authorization** off | It is off by default and registration mode does not need it. It configures a separate digest pair that *you* choose, and it is easy to mistake for the generated REGISTER credentials above. They are unrelated: filling this in does not give the sidecar anything, and leaving it off costs nothing. |
| **Force Symmetric RTP** on (default **off**) | The single most important setting. Your NAT rewrites the RTP port advertised in the SDP, so a carrier that trusts that port sends audio nowhere. Captured on a real call: the carrier sent 95 packets to the advertised port (all dropped by the NAT) before latching onto the observed source and delivering the next 564. Without this you get a call that connects and is silent. |
| **Symmetric RTP Ignore RTCP** on (default **off**; only takes effect while Force Symmetric RTP is on) | RTCP arrives from a different NAT-mapped port and can steer the latch to the wrong place. |
| **Use DID in R-URI** on (default **off**) | **Required for this stack.** It replaces the user part the carrier took from your registration's `Contact` with the called number, so the INVITE's Request-URI carries the DID. That matters because the sidecar relays the INVITE with its user part untouched — that user part *is* the called number as far as the rest of the stack is concerned — and provisioning binds both the LiveKit inbound trunk and its dispatch rule to your DID. Leave it off and the arriving call presents the registration username instead, matching nothing. |
| **Capacity on the number itself** | Easy to miss because it is not part of the trunk config at all. A DID with no included channels, no capacity pool and no shared capacity group is rejected by the carrier **at its own edge** — the caller hears "number not available" and nothing ever reaches you, so there is nothing in your logs to debug. Check the number's capacity as well as its trunk. |

Two more things that cost real debugging time and are worth knowing:

- **The INVITE may not come from the carrier's published signalling IPs.** In
  registration mode the call is delivered by the carrier's *registrar* tier,
  which is a different host set from the "static SIP" ranges its documentation
  lists. Allowlisting only the published static ranges silently drops every
  call. Use a range that covers both — this is why
  `SIP_REGISTRATION_ALLOWED_ADDRESSES` is separate from
  `DIDWW_INBOUND_ADDRESSES`.
- **Check a number's trunk with an `include=`.** DIDWW's JSON:API omits
  relationship data from a plain `GET /v3/dids/{id}`, which reads exactly like
  a number that has been detached from its trunk. Use
  `?include=voice_in_trunk`, the related endpoint, or a reverse filter before
  concluding anything is wrong.

### Building that trunk over DIDWW's API instead of the portal

If you hold a DIDWW API key, the table above has a reproducible form. Every
field name here comes from DIDWW's published v3 reference — the
[inbound trunk object](https://doc.didww.com/api3/2026-04-16/inventory-resources/voice-in-trunks/voice-in-trunk-object.html),
[create inbound trunk](https://doc.didww.com/api3/2026-04-16/inventory-resources/voice-in-trunks/create-voice-in-trunk.html)
and [update DID](https://doc.didww.com/api3/2026-04-16/inventory-resources/did/update-did.html)
pages at version `2026-04-16`. DIDWW versions its reference by date; check the
current one before copying field names.

The SIP settings are **not** top-level attributes. They live in a nested
`configuration` object typed `sip_configurations`, and a flat payload is not
what the reference documents. The four portal settings from the checklist above
are `enabled_sip_registration` (Dynamic Registration), `use_did_in_ruri` (Use
DID in R-URI — documented as working only when `enabled_sip_registration` is
`true`), `force_symmetric_rtp` and `symmetric_rtp_ignore_rtcp`.

```bash
curl -sS https://api.didww.com/v3/voice_in_trunks \
  -H 'Api-Key: <your DIDWW API key>' \
  -H 'Content-Type: application/vnd.api+json' \
  -H 'Accept: application/vnd.api+json' \
  -d '{
    "data": {
      "type": "voice_in_trunks",
      "attributes": {
        "name": "openphonex-selfhost",
        "priority": 1,
        "weight": 1,
        "capacity_limit": 2,
        "configuration": {
          "type": "sip_configurations",
          "attributes": {
            "username": "openphonex-selfhost",
            "host": "example.com",
            "enabled_sip_registration": true,
            "use_did_in_ruri": true,
            "force_symmetric_rtp": true,
            "symmetric_rtp_ignore_rtcp": true
          }
        }
      }
    }
  }'
```

Then point your number at the trunk it just returned:

```bash
curl -sS -X PATCH https://api.didww.com/v3/dids/<did-id> \
  -H 'Api-Key: <your DIDWW API key>' \
  -H 'Content-Type: application/vnd.api+json' \
  -H 'Accept: application/vnd.api+json' \
  -d '{
    "data": {
      "type": "dids",
      "id": "<did-id>",
      "relationships": {
        "voice_in_trunk": {
          "data": {"type": "voice_in_trunks", "id": "<trunk-id>"}
        }
      }
    }
  }'
```

`PATCH /v3/dids/{id}` is also where the capacity row above is set:
`dedicated_channels_count` assigns dedicated channels to the number and
`capacity_limit` caps its concurrent inbound calls (`null` for unlimited).

**Two things the published reference does not answer.** Stated rather than
guessed, because getting either wrong is silent:

- **What `username` and `host` should be once `enabled_sip_registration` is
  `true`.** Both are documented as required for `sip_configurations`, and both
  are described only in their static-endpoint meaning — the user and host parts
  of the R-URI DIDWW sends INVITEs to. In registration mode that destination is
  the registered binding instead, and the R-URI user part comes from your
  `Contact` (or from the called number, with `use_did_in_ruri`). Send values you
  will recognise, then open the trunk in the portal and confirm it shows as a
  Dynamic Registration trunk before trusting it. The reference does document a
  `"{DID}"` pattern for `username` — "replaced by called DID number in E164
  format" — but does not say how that interacts with `use_did_in_ruri`.
- **Where the generated REGISTER credentials come back.** No documented
  response attribute carries them; `auth_user`/`auth_password` on the trunk are
  the other pair, the one you set. Create the trunk however you like, then read
  the registration username and password from the portal — that is the only
  place DIDWW documents them appearing.

### Required: make `5060/udp` loopback-only

**Do this before you start the sidecar.** In registration mode the LiveKit
trunk is provisioned *without* an address allowlist — it has to be, because
every INVITE now reaches `livekit-sip` from your own host rather than from the
carrier, so a carrier allowlist would reject every real call. That removes a
boundary, and the host firewall has to replace it.

```bash
# IPv4 and IPv6 — both, on the host running the stack
sudo iptables  -A INPUT ! -i lo -p udp --dport 5060 -m conntrack --ctstate NEW -j DROP
sudo ip6tables -A INPUT ! -i lo -p udp --dport 5060 -m conntrack --ctstate NEW -j DROP
# persist them however your distro does (netfilter-persistent, firewalld, ufw)
```

**`--ctstate NEW` is load-bearing — do not "simplify" it away.** `livekit-sip`
sends its *outbound* signalling from the same socket it listens on, so replies
from your carrier arrive with **destination port 5060** as well. A bare
`--dport 5060 -j DROP` therefore also drops the answers to your own outbound
calls, and outbound calling breaks silently. Scoping the DROP to `NEW` drops
only unsolicited traffic and lets replies through. (Yes, this works for UDP:
netfilter's conntrack synthesises pseudo-connections for it, so a reply to a
packet you sent is `ESTABLISHED`, not `NEW`.)

Scoping the DROP is deliberate rather than putting an
`-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT` above it: that only
works if it really does land above, and `-A` appends to whatever chain you
already have. A self-contained rule has no ordering dependency. If your host
already has a stateful accept early in `INPUT` — ufw and firewalld both do —
either form works.

**Why both families.** Your NAT protects IPv4 only. A dual-stack host — common
in exactly the home/office/CGNAT situations this mode exists for — usually also
holds a **global IPv6 address with no NAT in front of it**, so an IPv6-reachable
`5060/udp` is reachable from the internet directly, bypassing the sidecar, the
pinhole and `SIP_REGISTRATION_ALLOWED_ADDRESSES` entirely. Verify with
`ss -lunp | grep 5060` and check whether the socket is bound `[::]` or a global
v6 address.

**What is at stake if you skip it.** Not nuisance calls. Anyone who reaches
`5060/udp` gets an answered call into your agent: unauthenticated voice access
to the LLM, prompt injection against whatever tools that agent is allowed to
invoke, and metered spend on your Deepgram / Gemini / ElevenLabs keys for as
long as they care to keep talking.

`SIP_REGISTRATION_ALLOWED_ADDRESSES` is defence in depth on top of this, not a
substitute for it: it protects the sidecar's own port, which is the only path
the carrier uses, but it cannot protect `livekit-sip`'s port from anything that
can reach the host directly.

If you would rather keep a trunk-level boundary as well, provision the LiveKit
inbound trunk with `allowed_addresses` set to the host's own LAN address (the
address the sidecar sends from — `REGISTRAR_STARTING` logs it as `listen`).
That restores a real check at the cost of reprovisioning whenever DHCP hands
the host a different address.

**Upgrading from an earlier setup.** The `SIP_REGISTRATION_*` settings used to
live in `.env.livekit`. Delete them from there — **including empty placeholder
lines**, which is the case that actually bites: a public-IP deployment that
copied the old template has `SIP_REGISTRATION_REGISTRAR=` sitting in
`.env.livekit` doing nothing, and the day it moves behind NAT those empty lines
silently override the real values (see "Which file holds what"). Copy any real
values into `.env.registrar`, delete the whole block from `.env.livekit`, and
**re-run the provision step** — the
provisioner reads them to choose the topology, and a run that cannot see them
builds the static-endpoint trunk whose address allowlist rejects every call
coming through the sidecar. Compose refuses to start the sidecar without
`.env.registrar`, so the file half is loud; the re-provision is the silent half.

**Known limits of registration mode**, stated plainly:

- **One media latch per call.** The carrier learns your media address once,
  from your first RTP packet. If your NAT rebinds mid-call, audio does not
  follow.
- **IPv4 only.** The sidecar registers over IPv4, binds one IPv4 address, and
  accepts only IPv4 CIDRs in `SIP_REGISTRATION_ALLOWED_ADDRESSES` (an IPv6
  entry is rejected at startup rather than silently ignored). A carrier that
  requires IPv6 signalling is not supported in this mode.
- **The sidecar is the source-filtering boundary** for the carrier's path, and
  the host firewall above is the boundary for everything else. Neither alone is
  sufficient.

**How the sidecar decides what to accept.** Every request is authorised before
any routing decision, in-dialog requests included — a proxy that gates only new
dialogs is an open relay to anyone who can present a `To` tag. Exactly two
sources are accepted: your carrier (the ranges in
`SIP_REGISTRATION_ALLOWED_ADDRESSES`) and the local `livekit-sip`, pinned to
both its address and its port. Everything else gets 403 and is logged. In-dialog
traffic arriving from the carrier is pinned to the local target rather than
routed by whatever `Route` set it carries, and the local side may only ever be
relayed back toward the carrier. Kamailio's workers drop to an unprivileged user
once the socket is bound; only the supervisor process stays root.
- **`livekit-sip` rate-limits inbound calls per source IP.** With a proxy in
  front, every call shares one source address, so that limiter applies to your
  whole trunk rather than per-caller. Fine for a single DID; worth knowing
  before you point heavy traffic at it.
- **SIP ALGs.** Consumer routers that "help" with SIP corrupt it. The sidecar
  defaults to port 5062 rather than 5060 partly to stay out of their way, but
  if you can disable SIP ALG on your router, do.

Fill in `.env.registrar` (created blank in step 1). It is deliberately separate
from `.env.livekit`, because `env_file` hands a container the whole file and
this is the one process on the box parsing SIP straight off the internet — see
"Which file holds what, and who reads it". Then:

```bash
docker compose -f docker-compose.livekit.yml --profile registrar up -d --build registrar
docker compose -f docker-compose.livekit.yml logs -f registrar
```

Expect `REGISTRAR_STARTING {...}` then `REGISTRAR_REGISTRATION_ARMED {...}`,
and thereafter silence — the sidecar is quiet when healthy. To confirm the
registration itself is live, watch for your carrier showing the endpoint as
registered, or raise `SIP_REGISTRATION_LOG_LEVEL=3` and look for the REGISTER
exchange.

## 0. What you need first

| # | Requirement | Where to get it |
|---|---|---|
| 1 | A Linux host with Docker Engine + Compose v2, and **either** a public IP (static-endpoint topology) **or** the registration topology above. On a Mac, see [Running on macOS](#running-on-macos) — Docker Desktop cannot run this stack | any cloud VM (2 vCPU / 4 GB is plenty to start); a NAT'd box at home/office works with `--profile registrar` |
| 2 | A **SIP trunk + DID** (phone number) | your carrier (production uses DIDWW; any carrier that sends SIP/RTP to your host works). Note BOTH the carrier's **signaling IP ranges** and its **RTP / media IP ranges** from its portal — they are often different range sets |
| 3 | An **OpenPhonex control plane** you can reach | OpenPhonex Cloud (`https://api.openphonex.com`) or your own deployment |
| 4 | From that control plane: an **organization id**, a registered **gateway deployment id**, two **different** credentials (`tgw_` for that gateway and `tai_` for its worker), and an **agent + phone-number route** configured for your DID | **API Keys → Gateway Tokens** creates the `tgw_`; the separate regular **API Keys** card creates the `tai_` worker key (both are shown once) |
| 5 | Provider keys: `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` | console.deepgram.com / aistudio.google.com / elevenlabs.io |
| 5b | *(only for si-LK / Sinhala agents)* a GCP service account for Cloud Speech-to-Text v2 | see [Language stacks: Sinhala (si-LK)](#language-stacks-sinhala-si-lk) below |
| 6 | Firewall: open `5060/udp` to the carrier's **signaling** ranges and `10000-10100/udp` to its **RTP / media** ranges (scope each range set separately). **Registration topology: instead DROP inbound `5060/udp` on both IPv4 and IPv6 except on loopback** — nothing needs to reach it unsolicited, and on a dual-stack host IPv6 has no NAT in front of it (see "Required: make `5060/udp` loopback-only") | your host/cloud firewall. LiveKit's own API and Redis bind loopback-only. `50000-50020/udp` is only needed if you later add WebRTC (browser) participants |

### Running on macOS

**Docker Desktop cannot run this compose file.** Every service here uses
`network_mode: host`, and macOS does not implement it: the containers start and
bind their ports, and nothing can reach them — not even `localhost` on the same
Mac. There is no flag or compose edit that fixes this, so Docker Desktop is not
a supported path.

Use [Colima](https://github.com/abiosoft/colima) in bridged mode instead:

```bash
colima start --network-address --network-mode bridged
```

The Linux VM then holds its own address on your LAN, `network_mode: host` means
what it means on Linux, and this compose file runs unmodified inside it. Every
command below — including `bash selfhost-smoketest.sh` — runs in that VM, not
on the Mac.

With `--profile registrar` that is all you need: **no public IP, no static IP,
and no port-forwards on your router.** The sidecar's outbound REGISTER is what
makes the VM reachable, and the router's NAT stays in front of it.

## 1. Configure (four files)

```bash
cd packages/openphonex-gateway/deploy
cp livekit.yaml.example livekit.yaml   # LiveKit server config
cp sip.yaml.example sip.yaml           # LiveKit SIP bridge config
cp .env.livekit.example .env.livekit   # control-plane + provider credentials
cp .env.registrar.example .env.registrar   # sidecar credentials; leave BLANK unless you need registration mode
chmod 600 livekit.yaml sip.yaml .env.livekit .env.registrar   # never commit the filled-in files
mkdir -p sessions call-evidence
# The agent drops to uid 10001 inside the container before writing these dirs,
# so give the host-side bind mounts to that uid or the first call EACCESes:
sudo chown -R 10001:10001 sessions call-evidence
```

Then fill them in:

### Create the two control-plane credentials — separately

Open your workspace's **API Keys** page after registering the gateway
deployment. These are different credentials with deliberately different trust
boundaries; do not substitute one for the other.

1. In the **Gateway Tokens** card, select the registered deployment and create
   its gateway token. It is a deployment-bound `tgw_` credential. The one-time
   reveal includes this exact assignment; replace the blank
   `OPENPHONEX_GATEWAY_API_TOKEN=` assignment in `.env.livekit` with the whole
   revealed line:

   ```dotenv
   OPENPHONEX_GATEWAY_API_TOKEN=tgw_...
   ```

   In the smoke test, the `tgw_` token is validated through that exact
   gateway's non-mutating number-rules read. It cannot authenticate the
   organization worker.
2. In the separate regular **API Keys** card, click **Create self-host worker
   key**. That creates the organization-scoped `tai_` key for
   `OPENPHONEX_WORKER_API_TOKEN`, with the required fixed baseline:
   `calls:read` + `calls:write` for inbound routing and its immediate
   call-bound runtime-config read. Copy its one-time value into:

   ```dotenv
   OPENPHONEX_WORKER_API_TOKEN=tai_...
   ```

Both plaintext values are revealed only at creation time and do not appear in
the API-key or gateway-token lists. Revoke and replace a lost or exposed token;
there is no later reveal.

Set `OPENPHONEX_ORGANIZATION_ID` to the same workspace that minted the `tai_`
key and owns the DID. A worker key from another workspace cannot route this
deployment's calls: the control plane rejects the tenant mismatch before it
returns a runtime configuration.

1. **Generate one LiveKit key pair** (you invent it; nobody issues it):
   `openssl rand -hex 32` for the secret, any short id (e.g. `opx-selfhost`)
   for the key. Put the **same pair in all three files**: `livekit.yaml`
   (`keys:` map), `sip.yaml` (`api_key`/`api_secret`), `.env.livekit`
   (`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET`).
2. **Fill in every remaining variable in `.env.livekit`.** Each one is
   documented inline — what it is, where to get it, and an example value.
   Two rules worth repeating here:
   - `OPENPHONEX_CONTROL_PLANE_URL` **must be `https://`** for a remote
     control plane. The agent refuses to send bearer tokens over plaintext
     http to anything but verified loopback (`127.0.0.0/8`, `::1`,
     `localhost`), and refuses cross-origin redirects.
   - `SCH236_TEST_DID_E164` takes the leading `+` (`+15551230000`), and accepts
     a comma-separated list (`+15551230000,+3725551234`) when this gateway
     should answer more than one number. They share one inbound trunk and one
     dispatch rule, and provisioning reuses the list. Note that re-running
     provisioning with a *different* single number does not add it: the flow
     creates a replacement trunk and retires the old one, so the second number
     would silently take the first one's place. List them together instead.
   - `DIDWW_INBOUND_ADDRESSES` must be your carrier's **real public** signaling
     ranges. The example value is a documentation range and is rejected on
     purpose (so you can't provision an allowlist no real call matches).
   - Outbound trunk vars (`DIDWW_OUTBOUND_*`) are **optional** — set all four
     to also create an outbound trunk, or leave all four blank for
     inbound-only. `DIDWW_OUTBOUND_FROM_USER` accepts comma-separated caller
     IDs when your carrier permits more than one From number.

## 2. Bring the stack up

```bash
docker compose -f docker-compose.livekit.yml up -d --build
# registration topology only:
docker compose -f docker-compose.livekit.yml --profile registrar up -d --build registrar
```

The first build installs the pinned `livekit-agents` wheels and pre-downloads
the VAD/turn-detector model files into the image (a few minutes, once).

## 3. Provision the SIP trunk + dispatch rule (one-shot)

```bash
docker compose -f docker-compose.livekit.yml --profile provision run --rm provision
```

Expect a final line `SELFHOST_PROVISIONED {...}`. This registers your DID and
carrier allowlist with LiveKit SIP (and an outbound trunk if you set the
optional `DIDWW_OUTBOUND_*` vars); rerun it only when the DID or the carrier IP
ranges change. In the registration topology the trunk is created **without** an
address allowlist on purpose (`inbound_source_policy: "sidecar"` in the output)
— see "Choosing your inbound topology". If it prints `SELFHOST_PROVISION_FAILED {...}`, the JSON `error`
field states the exact reason — see [Troubleshooting](#troubleshooting-livekit).

### Outbound calling

The `outbound-worker` compose profile claims only policy-approved
`call.outbound_originate` jobs. It consumes the call's short-lived,
destination-bound authorization before dialing and explicitly dispatches the
production agent into the LiveKit room. Per-call instructions, opening line,
and language are fetched from OpenPhonex by call ID; they are never trusted
from SIP or room metadata.

The guided self-host worker key is intentionally **inbound-only**: it excludes
the global `worker_jobs:*` authority, so do not start this optional profile
with that baseline credential. Keep `OPENPHONEX_EXECUTE_LIVEKIT_OUTBOUND=0`
unless OpenPhonex has issued a separately reviewed outbound-worker credential.

## 4. Verify — run the smoke test

```bash
bash selfhost-smoketest.sh
```

It prints a `[PASS]`/`[FAIL]` line per check (services up, redis, LiveKit,
**agent registered**, the deployment-bound **gateway `tgw_` token accepted**
by the non-mutating number-rules read, the organization-scoped **worker `tai_`
key authenticates** through MCP `tools/list`, artifact dirs writable) and ends
with `RESULT: PASS` when the stack is up and both credentials authenticate.
Every `[FAIL]` line includes the exact command or fix to apply.

`RESULT: PASS` is necessary but **not sufficient** for a call to succeed: the
worker-token check only proves it *authenticates* (MCP `tools/list`), not that
it carries the `calls:read` + `calls:write` routing/config scopes. Because
`tools/list` has no target DID, it also cannot establish that the `tai_` key
belongs to the configured `OPENPHONEX_ORGANIZATION_ID`; the inbound route
checks that tenant binding before it resolves the DID. The smoke test does not
verify that your DID's number route + agent are configured in the control
plane. The first real call in step 5 is the end-to-end confirmation.

Manual spot-check, if you prefer:

```bash
docker compose -f docker-compose.livekit.yml logs -n 20 agents
```

Expect a `registered worker` line and one `SCH236_PRODUCTION {"event":
"process_warm", ...}` line per warm process.

## 5. Place a test call

Call your DID from any phone. The agent answers with the configured first
message, and `docker compose -f docker-compose.livekit.yml logs -f agents`
streams structured events:

```
media_admitted        (semantic_config_source: "realtime-config")
turn_latency          (one per conversational turn)
model_tools_executed  (when the agent invokes an allowlisted tool)
call_complete         (with the billing source)
```

Call artifacts (caller/agent/mixed WAV + transcript + per-turn clips) land in
`call-evidence/<call_id>/` and are uploaded to the control plane; per-session
state lives in `sessions/`.

## Security notes

- **Worker token is organization-scoped** (`OPENPHONEX_WORKER_API_TOKEN`,
  prefix `tai_`). It comes from the regular **API Keys** card, not **Gateway
  Tokens**. The self-host worker preset includes only `calls:read` +
  `calls:write`, so it can route inbound calls and read their call-bound
  runtime configuration. The self-host compose explicitly disables its
  non-critical global worker-heartbeat loop; it does not grant `worker_jobs:*`
  or enable the optional outbound profile.
  This mirrors production: inbound routing and MCP `tools/list` schema
  discovery are org operations. Model tool calls use a call-bound endpoint, but
  whoever holds this token can still run the worker for the whole organization.
  Treat the box accordingly: `chmod 600 .env.livekit`, no shared shell access,
  and **rotate the token immediately** if the host is ever suspected
  compromised. (Narrowing this to per-call tokens is a planned control-plane
  change; the gateway `tgw_` token is already narrowly scoped.) See the
  repo-root `SECURITY.md`.
- **Gateway tokens are deployment-scoped** (`OPENPHONEX_GATEWAY_API_TOKEN`,
  prefix `tgw_`). Create them only from **API Keys → Gateway Tokens** after
  registering the deployment, and use one only with its matching gateway id.
  A valid token for another deployment or organization is rejected; it is not
  a worker credential.
- **Transport is enforced, not just advised**: the agent refuses plaintext
  http to non-loopback control planes and refuses cross-origin redirects, so
  the bearer tokens cannot be tricked onto another host.
- **Firewall to your carrier only.** `5060/udp` and `10000-10100/udp` should
  admit your carrier's published ranges, not the whole internet; the provision
  step additionally enforces the carrier allowlist at the trunk level.

## How per-call config resolves (the production path)

The agent joins the room **unsubscribed** and admits no caller audio until
the control plane authorizes the call:

1. `POST /v1/calls/inbound/route` (worker token) — resolves the **called DID**
   to the authoritative agent and creates the call, returning the
   organization/project binding that owns that number. Endpoint-routed calls
   resolve tools the same way.
2. `GET /v1/calls/{call_id}/realtime-config` (worker token) — the resolved
   semantic config: system prompt, first message, language, transcriber/model/
   voice provider selection (Deepgram/Gemini/ElevenLabs by default, or Google
   Chirp 2/Gemini/Gemini-TTS for `language: si`), and the tool allowlist. The
   agent validates it — including that its organization and project match the
   ones the called number resolved to — and only then subscribes to media.
3. Tools: the agent fetches schemas via MCP `tools/list` on `POST /mcp`,
   exposes **only** allowlisted tools to the model, and every invocation
   crosses the control plane again via
   `POST /v1/calls/{call_id}/agent-tool-calls` with an audit trail.
4. Lifecycle: `sip.answered` / `room.media_started` /
   `room.ended` / `sip.ended` are reported to
   `POST /v1/gateway-deployments/{id}/call-lifecycle-events`; the terminal
   event produces the authoritative billing verdict.
5. Artifacts (caller/agent/mixed WAV + transcript + per-turn clips) upload to
   `POST /v1/calls/{call_id}/artifacts` with duration provenance.

The contract surface is [`contracts/gateway-v1.yaml`](../../../contracts/gateway-v1.yaml)
+ [`contracts/schemas/`](../../../contracts/schemas/).

The `reconciler` service closes sessions whose agent died mid-call
(`session.reconciled` lifecycle events), so a crash can never leave dangling
billed minutes.

## Language stacks: Sinhala (si-LK)

The default pipeline (Deepgram nova-3 → Gemini → ElevenLabs) does not cover
Sinhala — **neither Deepgram nor ElevenLabs supports si-LK at all**. An agent
configured with `language: si` instead resolves to:

| Component | Provider | Notes |
|---|---|---|
| STT (transcriber) | Google Cloud Speech-to-Text v2, **Chirp 2** | `asia-southeast1` recognizer location — empirically verified against the live API for si-LK. Chirp 3 rejects si-LK with an HTTP 400 in both `eu` and `us`; Chirp 2 in `asia-southeast1` transcribed a Sinhala test phrase correctly at 0.89 confidence, and `asia-southeast1` is also geographically closer to Sri Lanka than `eu` |
| LLM (model) | Gemini | same model catalog as every other language — Gemini handles Sinhala natively, no separate credential |
| TTS (voice) | **Gemini-TTS** (`livekit.plugins.google.beta.GeminiTTS`) | same `GEMINI_API_KEY` as the LLM above; non-streaming (chunked HTTP synthesis, not a realtime websocket) |

The catalog rejects any other combination for `si` up front — picking
Deepgram or ElevenLabs for a Sinhala agent fails validation at config-save
time with a clear error, not silently at call time.

### 1. GCP service account (Chirp 2 credentials)

Chirp 2 is a Cloud Speech-to-Text **v2** API call authenticated with a real
GCP service account — this is **not** the `GEMINI_API_KEY` above (that's a
Gemini API key, a completely different Google product surface).

1. In the [GCP Console](https://console.cloud.google.com), pick or create a
   project, then enable **Cloud Speech-to-Text API**
   (`speech.googleapis.com`).
2. IAM & Admin → Service Accounts → **Create service account**. Grant it the
   **Cloud Speech Client** role (`roles/speech.client`) — no broader role is
   needed.
3. Open the new service account → **Keys** → **Add key** → **Create new key**
   → JSON. Download the file.
4. Give the agent stack that key one of two ways:
   - **File mount (default):** copy the JSON to the host, bind-mount it into
     the `agents` container, and set `GOOGLE_APPLICATION_CREDENTIALS=<path
     inside the container>` in `.env.livekit`.
   - **Inline env var:** if your platform only supports secrets-as-env-vars,
     set `GOOGLE_APPLICATION_CREDENTIALS_JSON=<the file's contents, as a
     single-quoted single line>` instead — no file mount needed. The agent
     tries this first if both are set.

### 2. Env vars (new in `.env.livekit`)

```
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp-speech-sa.json   # OR:
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
GOOGLE_STT_LOCATION=asia-southeast1   # optional; asia-southeast1 is the default and the empirically verified si-LK region (Chirp 3 rejects si-LK with an HTTP 400 in eu/us)
```

`GEMINI_API_KEY` (already required by the default pipeline) covers both the
Gemini LLM and Gemini-TTS — no extra key for TTS.

### 3. Configure the agent

Set the agent's `language` to `si` and its transcriber/voice to the si-LK
stack (via the dashboard's agent config editor — the language picker and
provider dropdowns are catalog-driven and simply show the new options once
this deploy has the credentials above — or via the API/MCP `set_agent_voice_stack` /`create_agent` call):

```json
{
  "language": "si",
  "transcriber": {"provider": "google", "model": "chirp_2"},
  "model": {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
  "voice": {"provider": "gemini", "model": "gemini-2.5-flash-preview-tts", "voice_id": "Kore"}
}
```

### 4. Turn-taking caveat

The pinned turn-detector plugin (`livekit-plugins-turn-detector` /
`inference.TurnDetector(version="v1-mini")`) does not have a calibrated model
for Sinhala (its local model only covers ar/de/en/es/fr/hi/id/it/ja/ko/nl/
pt/tr/zh — the same reason Estonian `et` calls already fall back to VAD-only
today). The worker detects this automatically per-call and falls back to
Silero VAD + provider end-of-utterance timing instead of running the
turn-detector uncalibrated. **Tradeoff:** turn-taking on si-LK calls is
slightly slower / less natural than on a turn-detector-supported language —
the agent waits out a fixed silence window (350–1500 ms) rather than acting on
a semantic end-of-turn signal. This is a deliberate, tested fallback (see
`TURN_DETECTOR_SUPPORTED_LANGUAGES` in [`livekit_agent/demo.py`](../../../livekit_agent/demo.py)),
not an oversight.

### 5. Validating before a real customer call

Do **not** place a live PSTN call to validate this — synthesize and transcribe
a short recorded Sinhala sample offline first. Once you have a short si-LK WAV
sample (a native speaker saying a few sentences), a founder-run validation
looks like:

**Empirically confirmed against the live Google Speech-to-Text v2 API:**
`chirp_2` + `asia-southeast1` transcribes si-LK correctly (0.89 confidence on
a Gemini-TTS-synthesized Sinhala test phrase); `chirp_3` rejects si-LK with an
HTTP 400 in both `eu` and `us`. Use the `chirp_2`/`asia-southeast1` pairing
below, not `chirp_3`.

```python
# validate_sinhala_stack.py — run with: python3 validate_sinhala_stack.py sample.wav
import sys
from livekit.plugins import google
from livekit.plugins.google.beta import GeminiTTS

stt = google.STT(model="chirp_2", languages="si-LK", location="asia-southeast1")   # picks up
                                                                                     # GOOGLE_APPLICATION_CREDENTIALS
tts = GeminiTTS(model="gemini-2.5-flash-preview-tts", voice_name="Kore")  # picks up GEMINI_API_KEY / GOOGLE_API_KEY

# 1. STT: confirm a recorded si-LK sample transcribes to readable Sinhala text.
#    (see livekit.plugins.google.STT.recognize / .stream for the exact call
#    shape used by build_demo_stack() in livekit_agent/demo.py)
#
# 2. TTS: confirm a short Sinhala sentence synthesizes to audible, correct speech.
#    tts.synthesize("ඔබට ණය ගෙවීමේ දිනය මතක් කිරීමට කතා කරමි.")
```

This is the same build-the-exact-pinned-plugin-objects-and-round-trip-a-short-
phrase pattern this stack already uses for provider interop smokes; use the
snippet above as a starting point once you have a recorded sample. **This
repo's automated validation of Gemini-TTS Sinhala output is a
plugin-source-code / vendor-documentation check only — nobody on this change
has heard the actual audio.** Listen to the synthesized sample yourself before
routing real customer calls to it.

## Updating

```bash
git pull
docker compose -f docker-compose.livekit.yml up -d --build
bash selfhost-smoketest.sh
```

Version pins (image digests, plugin versions) only change when production's
change — they are enforced equal by the release tests.

## Troubleshooting (LiveKit)

- **`bash selfhost-smoketest.sh` first.** It pinpoints the failing layer and
  prints the fix; everything below assumes you have its output.
- **`agents` restarts in a loop / exits immediately:** almost always a missing
  or malformed variable in `.env.livekit` — `docker compose -f
  docker-compose.livekit.yml logs -n 100 agents` shows the exact `KeyError` or
  validation message. The entrypoint also exits early (`carrier runtime
  environment is unavailable`) if `.env.livekit` is missing from the bind
  mount.
- **`ValueError: control-plane base_url must be https...`:** you pointed
  `OPENPHONEX_CONTROL_PLANE_URL` at plaintext http on a non-loopback host.
  Use `https://` (this is a deliberate guard — it protects your tokens).
- **Provision prints `SELFHOST_PROVISION_FAILED {...}`:** read the JSON
  `error` field — it is the exact reason. Common ones: a missing/blank
  required variable; a `DIDWW_INBOUND_ADDRESSES` entry that isn't a public
  IPv4 CIDR (use your carrier's real ranges, not the documentation example);
  or a partially-filled `DIDWW_OUTBOUND_*` set (supply all four or none).
- **Smoke test: `control plane accepts the gateway token` FAILS with 401/403:**
  the `tgw_` token is wrong, revoked, expired, from another organization, or
  does not match that gateway deployment id. Mint a fresh one from **API Keys
  → Gateway Tokens** for that registered deployment. If the **worker** check
  fails instead, create/replace its organization-scoped `tai_` key from the
  separate regular **API Keys** card (**Create self-host worker key**); a
  `tgw_` pasted where the worker key belongs is rejected (they are different
  tokens).
- **Call rings but nobody answers:** the dispatch rule or number route is off.
  Check (a) provision succeeded (`SELFHOST_PROVISIONED`), (b) the control
  plane has a phone-number route + agent for the DID, (c)
  `SCH236_TEST_DID_E164` has the leading `+`.
- **Call connects, instant hangup, or one-way/no audio:** firewall. Confirm
  `5060/udp` AND the RTP range `10000-10100/udp` are open to your carrier's
  ranges, and that `DIDWW_INBOUND_ADDRESSES` covers the address the INVITE
  actually came from (the `sip` service logs show the source IP).
- **Registration topology, the carrier shows you registered but no call ever
  arrives:** the keepalive is the first suspect — a NAT binding that expires
  between registration refreshes produces exactly this (healthy registration,
  no calls). Confirm `SIP_REGISTRATION_KEEPALIVE_SECONDS` is set and under 30.
  If that is fine, the carrier is probably delivering to the `Contact` literal
  rather than the address it observed; ask it which, and if it is the literal,
  you need `SIP_REGISTRATION_CONTACT_ADDRESS` and a static port-forward.
- **Registration topology, call connects but is silent:** the carrier is not
  latching onto the address your media actually comes from. Turn on symmetric
  RTP on the trunk. Your NAT rewrites the port advertised in the SDP, so a
  carrier that trusts that port cannot reach you. Expect the silent call to
  drop itself after about half a minute rather than hang on — the trunk's RTP
  Timeout (30 s by default on DIDWW) disconnects a call that receives no RTP.
- **Registration topology, `403 Forbidden` in the `registrar` logs:** the
  INVITE arrived from an address outside
  `SIP_REGISTRATION_ALLOWED_ADDRESSES`. The log line names the source; in
  registration mode carriers often deliver from a registrar tier that is not in
  their published static-signalling list.
- **First call is slow to answer:** the model prewarm removes the per-call
  cold start, but only after `process_warm` appears in the agents log — give a
  fresh stack a minute.
- **`EACCES`/permission errors on `sessions/` or `call-evidence/`:**
  `sudo chown -R 10001:10001 sessions call-evidence` (the agent drops to
  uid 10001 before writing).

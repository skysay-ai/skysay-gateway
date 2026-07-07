# Self-host quickstart — gateway against YOUR existing Asterisk

Run the OpenPhonex Gateway (streaming Deepgram STT → Gemini LLM → ElevenLabs
v3 TTS over Asterisk ExternalMedia) as a single container next to an Asterisk
container you already operate. **Standalone and cloud-free**: with all
`OPENPHONEX_GATEWAY_*` variables unset the worker makes **zero** calls to
OpenPhonex Cloud (see [Cloud/managed mode is optional](#cloudmanaged-mode-is-optional-and-off)).

The topology below is the one validated live on the OpenPhonex demo
deployment: the gateway container **joins the existing Asterisk docker
network** at a **static IP**, and ExternalMedia RTP flows
container-to-container on that network — no host hairpin, no published RTP
ports.

```
[ your Asterisk container ]  <--ARI ws/http-->  [ openphonex-gateway ]
   e.g. 172.18.0.2:8088                            static 172.18.0.10
        |                                                 |
        +----------- RTP (ExternalMedia, UDP) ------------+
              same docker network, container-to-container
```

## Prerequisites

- Docker with Compose v2 on the host that runs your Asterisk container.
- An **existing Asterisk container** (any distro image) with:
  - **ARI enabled** — `http.conf` serving on `0.0.0.0:8088` *inside the
    container network* (do not publish 8088 to the internet) and an
    `ari.conf` user/password;
  - a **Stasis app** in the dialplan for the calls you want the agent to
    handle, e.g. `same => n,Stasis(voicebot)` — the app name goes into
    `ASTERISK_ARI_APP`;
  - for outbound calls: a PJSIP endpoint that can dial out through your SIP
    trunk (see [Place an outbound call](#place-an-outbound-call)).
- Provider keys: `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`.

## 1. Find your Asterisk network and IPs

```bash
docker network ls                                   # find the network name
docker inspect <asterisk-container> \
  --format '{{json .NetworkSettings.Networks}}'     # network name + Asterisk IP
docker network inspect <network-name>               # subnet + taken addresses
```

You need three things:

| What | Example | Goes into |
|------|---------|-----------|
| Network name | `my-asterisk_default` | `ASTERISK_DOCKER_NETWORK` |
| Asterisk container IP | `172.18.0.2` | `ASTERISK_ARI_URL`, `ASTERISK_RTP_HOST` |
| A free IP on that subnet | `172.18.0.10` | `GATEWAY_STATIC_IP`, `ASTERISK_EXTERNAL_MEDIA_HOST` |

If your Asterisk container restarts and changes IP, prefer its network alias
(container name) in `ASTERISK_ARI_URL`/`ASTERISK_RTP_HOST` — the gateway's own
IP must stay static either way, because Asterisk sends RTP to it by address.

## 2. Configure

```bash
cd packages/openphonex-gateway/deploy
cp .env.selfhost.example .env.selfhost    # never commit the filled-in file
```

Fill in `.env.selfhost` (every variable is commented in the example): the
topology block from step 1, the ARI credentials, the provider keys, the
opening utterances, and — when you are ready to take real calls —
`REALTIME_LIVE_ENABLED=1`. The worker is **fail-closed**: with
`REALTIME_LIVE_ENABLED=0` it refuses to serve Asterisk and just logs
`ari_connect_error` with `Set REALTIME_LIVE_ENABLED=1 ...` on every retry.

Optional preflight (validates env/provider/live gates offline, opens no ARI
connection, places no calls) — run it **before** `up` (a one-off compose run
claims the static IP, so don't do this while the gateway is running):

```bash
docker compose --env-file .env.selfhost -f docker-compose.selfhost.yml \
  run --rm --no-deps gateway \
  python -u -m openphonex_gateway.realtime.worker --check-live-startup
```

## 3. Build and start

```bash
docker compose --env-file .env.selfhost -f docker-compose.selfhost.yml up -d --build
```

Notes:

- `--env-file .env.selfhost` is required: the network name and static IP are
  compose *interpolation* variables; the service-level `env_file:` alone does
  not feed interpolation. Missing it fails loudly with
  `set ASTERISK_DOCKER_NETWORK in .env.selfhost ...`.
- The image builds **lean** (`GATEWAY_EXTRAS=""` build arg): the cloud
  provider pipeline needs no optional extras. It is a `python:3.11-slim` base
  — no `ss`, `netstat`, or `curl` inside (see
  [Troubleshooting](#troubleshooting)).

## 4. Verify

**Logs** (structured JSON on stderr; `docker logs` shows it):

```bash
docker logs -n 40 openphonex-gateway
```

Expect, in order:

```
{"event":"health_server_listening","url":"http://0.0.0.0:18772",...}
{"event":"ari_connect_attempt","attempt":1,...}
{"event":"ari_ws_opening",...}
{"event":"ari_ws_connected","ari_url":"http://172.18.0.2:8088/ari","app":"voicebot",...}
```

`ari_ws_connected` fires only after the ARI WebSocket 101 handshake — the
worker is subscribed to your Stasis app and will receive calls. Repeated
`ari_connect_error` + `ari_connect_retry_wait` lines mean ARI is
unreachable/mis-authed (or `REALTIME_LIVE_ENABLED` is still 0); the worker
keeps retrying with bounded backoff instead of crashing (defaults:
`REALTIME_ARI_CONNECT_MAX_ATTEMPTS=0` = forever, backoff 2s linear capped at
30s).

**Health endpoint** (loopback-only on the host):

```bash
curl -s http://127.0.0.1:18772/health    # also: /status, /
```

Returns the full redacted worker status JSON (providers, ARI config, RTP
pool, last calls). Also available in-container:

```bash
docker exec openphonex-gateway python -m openphonex_gateway.realtime.worker --status
```

**Inbound smoke test:** dial a number your dialplan routes into
`Stasis(<ASTERISK_ARI_APP>)` — the agent greets with
`REALTIME_INBOUND_OPENING_UTTERANCE`, and `docker logs -f openphonex-gateway`
streams transcripts.

## Place an outbound call

The gateway ships a standalone, control-plane-free originate CLI:
`openphonex_gateway.originate` (also installed as the `openphonex-originate`
console script). It POSTs to your own Asterisk's ARI `/channels` with an
explicit `callerId` (plain `channel originate` can present an anonymous
caller that many trunks reject with 403) and puts the call into your Stasis
app with `appArgs=outbound`, so the same running worker answers it and speaks
`REALTIME_OPENING_UTTERANCE`.

Dry-run first (prints the ARI payload, places no call):

```bash
docker exec openphonex-gateway \
  python -m openphonex_gateway.originate +15555550123 --dry-run
```

Then the real call:

```bash
docker exec openphonex-gateway \
  python -m openphonex_gateway.originate +15555550123 --endpoint <your-pjsip-endpoint>
```

How the dial string is built: `PJSIP/<endpoint>/sip:<digits>@<DIDWW_OUTBOUND_HOST>`.

- `--endpoint` — your PJSIP endpoint section name. Default is
  `didww-outbound` (the reference deployment's endpoint; the shipped
  `configs/asterisk/pjsip.conf.template` uses `trunk-outbound`), so pass your
  own explicitly.
- `DIDWW_OUTBOUND_HOST` (env) or `--outbound-host` — the SIP host dialed
  through the endpoint.
- `DIDWW_CALLER_ID` (env) or `--caller-id` — the presented caller ID.
  (`--from-cli us|ee` selects `DIDWW_CALLER_ID_US`/`DIDWW_CALLER_ID_EE`
  profiles instead.)
- `--app` / `--app-args` / `--ari-url` / `--timeout` override
  `ASTERISK_ARI_APP` / `ASTERISK_ARI_APP_ARGS` / `ASTERISK_ARI_URL` /
  `ASTERISK_ARI_TIMEOUT`.
- A real (non-`--dry-run`) call additionally needs `ASTERISK_ARI_USER` +
  `ASTERISK_ARI_PASSWORD` — already present in the container env.
- `--env-file <path>` reads the settings from a dotenv file instead (useful
  when running the CLI outside the container: `pip install -e
  packages/openphonex-common -e packages/openphonex-gateway`, then
  `openphonex-originate +1555... --env-file deploy/.env.selfhost`).

## The ONE-CONSUMER rule

**Never run a second Stasis consumer on the same app name.** Two ARI
WebSocket subscribers on one app steal each other's events and both break. If
you are migrating off an existing bot that owns the app (e.g. a legacy
process on `voicebot`): stop the old consumer first, or run this gateway on a
different `ASTERISK_ARI_APP` and repoint your dialplan when ready. To see who
owns what:

```bash
curl -s -u <ari-user>:<ari-password> http://<asterisk-ip>:8088/ari/applications
```

To prove ARI reachability *without* touching a production app, start a
throwaway container with an override (`-e ASTERISK_ARI_APP=canary`, no
published ports, auto-assigned IP via plain `docker run --network ...`), watch
for `ari_ws_connected`, then remove it.

## Updating

```bash
git pull
docker compose --env-file .env.selfhost -f docker-compose.selfhost.yml up -d --build
```

`up -d --build` rebuilds the image and recreates the container only when
something changed. Your `.env.selfhost`, `calls/`, and `recordings/` are
untouched.

## Cloud/managed mode is OPTIONAL (and off)

This quickstart runs the gateway **standalone**. The `OPENPHONEX_GATEWAY_*` /
`OPENPHONEX_API_URL` variables and the `--report-health` flag exist for
topology B (managed numbers from OpenPhonex Cloud + a per-gateway `tgw_`
token) — see the repo-root `README.md` and `docker-compose.gateway.yml`.
Leave them unset here: the compose command in this directory does not even
pass `--report-health`, and even if it did, the reporter short-circuits to
`{"configured": false, "reason": "missing OPENPHONEX_API_URL, OPENPHONEX_GATEWAY_ID, or OPENPHONEX_GATEWAY_API_TOKEN"}`
without making any HTTP request.

## Troubleshooting

- **`docker logs` looks empty / where are the logs?** Structured JSON worker
  logs go to **stderr**; the final `--serve-asterisk` result JSON goes to
  **stdout**. `docker logs` shows both, but if you pipe, remember
  `2>&1`. The compose file sets `PYTHONUNBUFFERED=1` *and* runs `python -u`,
  so lines appear immediately.
- **STT dies instantly on every call / Deepgram WebSocket closes with 400:**
  `DEEPGRAM_UTTERANCE_END_MS` below **1000** is rejected by Deepgram (its
  documented minimum). The worker clamps lower values up to 1000 when
  building the URL, so a too-low override degrades gracefully — but keep the
  env at ≥ 1000 and tune `DEEPGRAM_ENDPOINTING_MS` instead if you want
  snappier end-of-turn detection.
- **No `ss`/`netstat` in the lean image.** To check the health listener from
  inside the container, read `/proc/net/tcp` — port 18772 is hex `4954`:

  ```bash
  docker exec openphonex-gateway grep -i ':4954' /proc/net/tcp   # state 0A = LISTEN
  ```

- **`ari_connect_error` forever:** check the message field. `Set
  REALTIME_LIVE_ENABLED=1 ...` → flip the gate. `Missing live Asterisk env`
  → `ASTERISK_ARI_USER`/`ASTERISK_ARI_PASSWORD` empty. Connection
  refused/timeouts → wrong `ASTERISK_ARI_URL`, ARI not enabled, or the
  gateway is not on the Asterisk network (`docker inspect
  openphonex-gateway --format '{{json .NetworkSettings.Networks}}'`).
- **One-way or no audio:** `ASTERISK_EXTERNAL_MEDIA_HOST` must be the
  gateway's static IP (where Asterisk sends RTP) and `ASTERISK_RTP_HOST` the
  Asterisk container's IP (where the gateway sends RTP). Both containers must
  share the network — never let compose create a new network with the same
  name (`external: true` in the compose file guards this).
- **Agent talks over the caller / gets cut off by echo:** tune
  `REALTIME_BARGE_IN_RMS`. 650 is sensitive; larger values (toward 32000)
  approach half-duplex.
- **Static IP already in use:** a leftover one-off container (e.g. an old
  `compose run`) may still hold it — `docker ps -a` and remove it, or pick
  another free IP (update `GATEWAY_STATIC_IP` *and*
  `ASTERISK_EXTERNAL_MEDIA_HOST` together).

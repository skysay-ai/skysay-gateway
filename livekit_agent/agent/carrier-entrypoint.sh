#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

umask 077
if [ ! -r /run/sch172/carrier.env ]; then
  echo "carrier runtime environment is unavailable" >&2
  exit 1
fi
# The container's stop deadline belongs to whoever set the matching
# `stop_grace_period`, which is Compose -- and this file is sourced AFTER
# Compose's environment, so a line in it would otherwise win. The worker
# derives its drain budget from this number, so a carrier.env value would be a
# budget Docker does not enforce: larger restores the mid-drain SIGKILL, and
# smaller ends sessions early. Saved here, restored below -- unconditionally,
# so that when Compose set nothing the restore is the empty string and any
# value the file supplied is discarded rather than trusted; the worker reads a
# blank as absent and falls back to its compiled-in default. Not exported, and
# assigned before `set -a`, so the helper itself never reaches the worker.
compose_stop_grace="${SKYSAY_VOICE_STOP_GRACE_SECONDS:-}"
# The worker's HTTP port belongs to Compose for the same reason, and the stakes
# are the pair itself. The release hook hands the transient colour `0` so the
# OS picks a free port; every other worker takes the fixed 8081. A value in
# this file would win over that and give BOTH colours the same port on a host
# whose network namespace is shared -- the second to bind dies there, silently,
# because the port serves only a health endpoint nothing reads. Saved and
# restored unconditionally, so a file that sets nothing restores the empty
# string and the worker falls back to its compiled-in default exactly as
# before.
compose_worker_port="${SKYSAY_VOICE_WORKER_PORT:-}"
set -a
# shellcheck disable=SC1091
. /run/sch172/carrier.env
set +a
SKYSAY_VOICE_STOP_GRACE_SECONDS="$compose_stop_grace"
export SKYSAY_VOICE_STOP_GRACE_SECONDS
SKYSAY_VOICE_WORKER_PORT="$compose_worker_port"
export SKYSAY_VOICE_WORKER_PORT
# The per-call BYOK lease names, dropped AFTER the file is sourced so neither a
# line in it nor a container environment variable can survive into the worker.
#
# A workspace that brings its own provider key gets it injected into one call's
# in-memory provider environment under a component-scoped name, beside an
# SKYSAY_BYOK_* marker. The worker refuses a call that carries one of these
# names without its marker instead of quietly billing the platform account key
# -- so an operator who reads the names in component_credentials.py and copies
# one into this file would otherwise refuse every call on that provider, with
# an error naming neither the file nor the variable. No deployment supplies
# these names legitimately; drop them here rather than asking each config
# surface not to set them.
#
# This list must stay equal to _SCOPED_COMPONENT_VARIABLES in
# component_credentials.py -- a test compares the two sets in both directions,
# so a new lease variable fails until it is listed here as well. Only lease
# names belong here: an account key (SONIOX_API_KEY, FISH_AUDIO_API_KEY, ...)
# added by mistake would disable managed calls on that provider.
unset \
  SONIOX_STT_API_KEY \
  SONIOX_TTS_API_KEY \
  SKYSAY_BYOK_SONIOX_STT \
  SKYSAY_BYOK_SONIOX_TTS \
  FISH_AUDIO_TTS_API_KEY \
  SKYSAY_BYOK_FISHAUDIO_TTS \
  CARTESIA_TTS_API_KEY \
  SKYSAY_BYOK_CARTESIA_TTS \
  GEMINI_TTS_API_KEY \
  SKYSAY_PLATFORM_GEMINI_TTS
# Resolved into a variable first, so `set -e` stops the container if the uid
# cannot be read. Inline, a failing `id` leaves the test comparing an empty
# string, which is an error the `if` reads as false -- and the worker would
# then start with the privileges the branch below exists to give up.
container_uid="$(id -u)"
if [ "$container_uid" -eq 0 ]; then
  # Production keeps the Google service-account bind mount root-owned and
  # mode 0400.  The worker deliberately runs as the unprivileged `agent`
  # account, so hand it a second, container-local copy that it can read but
  # cannot replace.  Do not put the JSON itself in the process environment:
  # ADC accepts the path and the secret stays out of logs and process listings.
  # Inline credentials take precedence in demo.py and need no file staging.
  google_credentials_source="${GOOGLE_APPLICATION_CREDENTIALS:-}"
  google_credentials_inline="${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}"
  if [ -n "$google_credentials_source" ] && [ -z "$google_credentials_inline" ]; then
    if [ ! -f "$google_credentials_source" ] || [ -L "$google_credentials_source" ] || [ ! -r "$google_credentials_source" ]; then
      echo "google speech credential file is unavailable" >&2
      exit 1
    fi
    google_credentials_directory=/run/skysay
    google_credentials_runtime="$google_credentials_directory/google-application-credentials.json"
    install -d -o root -g root -m 0711 "$google_credentials_directory"
    install -o root -g agent -m 0440 -- "$google_credentials_source" "$google_credentials_runtime"
    GOOGLE_APPLICATION_CREDENTIALS="$google_credentials_runtime"
    export GOOGLE_APPLICATION_CREDENTIALS
  fi
  # The agents write their readiness beacon here as the `agent`
  # account, and the release hook creates the bind-mounted directory as root.
  # Handed over only where it is writable: the originator mounts the same
  # directory read-only (it only reads beacons), and a read-only mount is left
  # exactly as it is.
  agents_readiness_dir="${SKYSAY_AGENTS_READINESS_DIR:-}"
  if [ -n "$agents_readiness_dir" ] && [ -d "$agents_readiness_dir" ] && [ -w "$agents_readiness_dir" ]; then
    chown agent:agent "$agents_readiness_dir"
    chmod 0755 "$agents_readiness_dir"
  fi
  # `setpriv`, not `runuser`, and the difference is the whole graceful stop.
  #
  # util-linux `runuser` does not replace itself with the worker: it forks,
  # supervises, and on SIGTERM forwards the signal, sleeps TWO SECONDS, then
  # SIGKILLs the child ("Session terminated, killing shell..." in
  # su-common.c). As the container's PID 1 that makes every stop grace
  # decorative -- measured in this exact base image, a worker asked to drain
  # for 25s was killed at 2s and the container then sat until Docker's own
  # deadline, so a promote both cut the call AND waited out the full grace it
  # was cutting the call to honour. `setpriv` execs, so the worker IS pid 1
  # and receives the signal itself.
  #
  # It gives up exactly what runuser gave up -- same uid, same gid, same
  # supplementary groups (the `agent` group is how the staged Google
  # credential above is readable) -- and nothing more. The environment
  # runuser used to set for the target user is set here instead: without it
  # the worker inherits root's HOME and cannot find the model files
  # `python -m livekit.agents download-files` staged into the agent's home at
  # build time.
  HOME=/home/agent
  USER=agent
  LOGNAME=agent
  export HOME USER LOGNAME
  exec setpriv --reuid=agent --regid=agent --init-groups --inh-caps=-all -- "$@"
fi

# A non-root self-hosted image cannot safely repair an unreadable mount. Fail
# before the provider starts instead of turning every call into a late,
# provider-shaped PermissionError.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -z "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ] && [ ! -r "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "google speech credential file is unavailable" >&2
  exit 1
fi
exec "$@"

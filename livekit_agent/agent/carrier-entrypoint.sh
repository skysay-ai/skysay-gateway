#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

umask 077
if [ ! -r /run/sch172/carrier.env ]; then
  echo "carrier runtime environment is unavailable" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. /run/sch172/carrier.env
set +a
# The per-call BYOK lease names, dropped AFTER the file is sourced so neither a
# line in it nor a container environment variable can survive into the worker.
#
# A workspace that brings its own provider key gets it injected into one call's
# in-memory provider environment under a component-scoped name, beside an
# OPENPHONEX_BYOK_* marker. The worker refuses a call that carries one of these
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
  OPENPHONEX_BYOK_SONIOX_STT \
  OPENPHONEX_BYOK_SONIOX_TTS \
  FISH_AUDIO_TTS_API_KEY \
  OPENPHONEX_BYOK_FISHAUDIO_TTS \
  GEMINI_TTS_API_KEY \
  OPENPHONEX_PLATFORM_GEMINI_TTS
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
    google_credentials_directory=/run/openphonex
    google_credentials_runtime="$google_credentials_directory/google-application-credentials.json"
    install -d -o root -g root -m 0711 "$google_credentials_directory"
    install -o root -g agent -m 0440 -- "$google_credentials_source" "$google_credentials_runtime"
    GOOGLE_APPLICATION_CREDENTIALS="$google_credentials_runtime"
    export GOOGLE_APPLICATION_CREDENTIALS
  fi
  exec runuser -u agent -- "$@"
fi

# A non-root self-hosted image cannot safely repair an unreadable mount. Fail
# before the provider starts instead of turning every call into a late,
# provider-shaped PermissionError.
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -z "${GOOGLE_APPLICATION_CREDENTIALS_JSON:-}" ] && [ ! -r "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "google speech credential file is unavailable" >&2
  exit 1
fi
exec "$@"

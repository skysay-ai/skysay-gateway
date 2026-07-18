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
if [ "$(id -u)" -eq 0 ]; then
  exec runuser -u agent -- "$@"
fi
exec "$@"

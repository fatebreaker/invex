#!/usr/bin/env bash
# Run the registered wide-book addition (scripts/run_iex_wide.sh) once the iex-v2 run has finished, either way.
#   setsid nohup scripts/wait_iex_wide.sh >> logs/iex_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*"; }
until grep -Eq "run_iex_ppo (done|FAILED)" logs/iex_wait.log 2>/dev/null; do sleep 1800; done
scripts/run_iex_wide.sh >> logs/iex_wide.log 2>&1 && log "run_iex_wide done" || log "run_iex_wide FAILED"

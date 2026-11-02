#!/usr/bin/env bash
# Run the registered PPO addition (scripts/run_iex_ppo.sh) once the iex-v1 test has run.
#   setsid nohup scripts/wait_iex_ppo.sh >> logs/iex_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "run_iex_test done" logs/iex_wait.log 2>/dev/null; do sleep 1800; done
scripts/run_iex_ppo.sh >> logs/iex_ppo.log 2>&1 && log "run_iex_ppo done" || log "run_iex_ppo FAILED"

#!/usr/bin/env bash
# iex-v8: run scripts/run_iex_v8.sh once, after every registered final analysis of the prospective IEX test has
# finished: the fitted-Q, PPO and wide-book runs (logs/iex_wait.log) and the iex-v5 and iex-v6 families.
#   setsid nohup scripts/wait_iex_v8.sh >> logs/iex_v8_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*"; }
until grep -Eq "run_iex_wide (done|FAILED)" logs/iex_wait.log 2> /dev/null \
    && grep -q "final done" logs/iex_v5_wait.log 2> /dev/null && grep -q "final done" logs/iex_v6_wait.log 2> /dev/null; do
    sleep 1800
done
scripts/run_iex_v8.sh >> logs/iex_v8.log 2>&1 && log "final done" || log "run_iex_v8 FAILED"

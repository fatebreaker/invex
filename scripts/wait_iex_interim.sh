#!/usr/bin/env bash
# iex-v4 part B: wait until five listed test days of the prospective test are built and pass the book checks (they are
# built by scripts/wait_iex_test.sh as HIST publishes them), then run the interim look once.
#   setsid nohup scripts/wait_iex_interim.sh >> logs/iex_interim_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date '+%F %T')] $*"; }
until python scripts/iex_days.py --interim > /dev/null 2>&1; do sleep 1800; done
log "five days: $(python scripts/iex_days.py --interim)"
scripts/run_iex_interim.sh >> logs/iex_interim.log 2>&1 && log "run_iex_interim done" || log "run_iex_interim FAILED"

#!/usr/bin/env bash
# iex-v6: run its secondary families once the registered interim evaluations have finished (interim spec, alpha
# 0.001), and once the registered final evaluations have finished (alpha 0.05). Each runs once.
#   setsid nohup scripts/wait_iex_v6.sh >> logs/iex_v6_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*"; }
done_interim=0
while true; do
    if [ $done_interim -eq 0 ] && grep -q "run_iex_interim done" logs/iex_interim_wait.log 2> /dev/null; then
        scripts/run_iex_v6.sh iex_interim iex_interim_ppo prereg/confirmatory_iex_interim_v6.json IexInterim
        done_interim=1
    fi
    if grep -q "run_iex_test done" logs/iex_wait.log 2> /dev/null && grep -q "run_iex_ppo done" logs/iex_wait.log 2> /dev/null; then
        scripts/run_iex_v6.sh iex_test iex_test_ppo prereg/confirmatory_iex_v6.json IexTest
        log "final done"
        exit 0
    fi
    sleep 1800
done

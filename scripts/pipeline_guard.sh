#!/usr/bin/env bash
# iex-v6/v7: record, when each registered IEX run starts, the SHA-256 of every pipeline file in the working tree and
# whether it matches the iex-v6 manifest (prereg/timestamps/manifest_iex_v6_*.txt). Each run is watched on its own,
# in parallel, so that runs in any order are each checked when they start; every 6 hours, until the iex-v6 families
# have run on the final analysis, a heartbeat check is logged too. It changes nothing and stops nothing; its log is the record that the prospective test ran on the registered
# code.
#   setsid nohup scripts/pipeline_guard.sh >> logs/pipeline_guard.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
M=$(ls prereg/timestamps/manifest_iex_v6_*.txt | head -n 1)
FILES="src/invex experiments/transfer.py experiments/confirm.py experiments/confirm_robust.py experiments/analyze_iex.py
experiments/ppo.py scripts/run_iex_test.sh scripts/run_iex_ppo.sh scripts/run_iex_wide.sh scripts/run_iex_interim.sh
scripts/run_iex_v5.sh scripts/run_iex_v6.sh scripts/build_iex.sh scripts/iex_days.py scripts/freeze_models.py scripts/env.sh"
log() { echo "[$(date '+%F %T')] $*"; }
check() {  # compare every pipeline file with the manifest's entry for tag iex-v6
    local bad=0 n=0
    for f in $(find $FILES -type f \( -name '*.py' -o -name '*.sh' \) | sort); do
        n=$((n + 1))
        want=$(awk -v f="$f" '$1 == "tag" {t = $2} t == "iex-v6" && $2 == f {print $1}' "$M")
        have=$(sha256sum "$f" | cut -d' ' -f1)
        [ "$want" = "$have" ] || { bad=$((bad + 1)); log "  differs from iex-v6: $f"; }
    done
    log "$1: $n pipeline files, $bad differ from the iex-v6 manifest; HEAD $(git rev-parse --short HEAD)"
}
watch() {  # wait for a run's start marker in its log, then check once
    local f=$1 m=$2
    until grep -q "$m" "$f" 2> /dev/null; do sleep 120; done
    check "$f started ($m)"
}
check "guard started"
watch logs/iex_interim.log "interim days" &
watch logs/iex_test.log "freeze episodes" &
watch logs/iex_ppo.log "evaluation" &
watch logs/iex_wide.log "evaluation" &
watch logs/iex_v5_wait.log "final done" &
( until grep -q "final done" logs/iex_v6_wait.log 2> /dev/null; do sleep 21600; check "heartbeat"; done ) &
wait
log "guard done"

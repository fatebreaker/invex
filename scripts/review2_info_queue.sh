#!/usr/bin/env bash
# Review round 2 (exploratory, post hoc): the equal-information arm R0info beyond Stage 1, as two reviewers asked:
# on Stage 2 (with PPO in MS+tick and raw units, never scored there), on the IEX test days' one-tick books and on all
# their books (the wide ones included). Each evaluation reads its sealed table once, logged.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*"; }
INFO="zero_R0info_s[0-2] zeronn_R0info_s[0-4] zeroppo_R0info_s[0-2]"
run() {  # out episodes reason models...
    out=$1 eps=$2 reason=$3
    shift 3
    [ -s "results/transfer/eval_${out}.parquet" ] && { log "skip $out (exists)"; return; }
    log "start $out"
    python experiments/transfer.py eval --episodes "$eps" --models "$@" --out "$out" --unseal --reason "$reason" \
        --workers 12 > "logs/eval_${out}.log" 2>&1 && log "done $out" || log "FAILED $out"
}
run stage2_info stage2_v0 "post hoc (review round 2): the equal-information arm, and PPO, on Stage 2 (exploratory)" \
    $INFO "zeroppo_R3tick_s[0-2]" "zeroppo_R0_s[0-2]"
run iex_test_info iex_test_v0 "post hoc (review round 2): the equal-information arm on the IEX test days (exploratory)" $INFO
run iex_test_wide_info iex_test_all_v0 \
    "post hoc (review round 2): the equal-information arm on every IEX test-day book (exploratory)" $INFO
log "queue done"

#!/usr/bin/env bash
# Tree fitted-Q parity check for the TMLR review, round 1 (exploratory, validation): raw units (R0, and R0r with the
# cost in spreads), the equal-information arm (R0info) and MS+tick (R3tick), each refitted with more capacity than
# the registered settings (300 rounds of 63-leaf trees): 1,000 rounds (tag tunea), 255 leaves (tuneb) and both
# (tunec), seeds 0-2. Every other setting as registered.
#   setsid nohup scripts/review_tree_queue.sh >> logs/review_tree_queue.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=16
log() { echo "[$(date '+%F %T')] $*"; }
fit() {  # tag rounds leaves
    local tag=$1 rounds=$2 leaves=$3
    log "start $tag ($rounds rounds, $leaves leaves)"
    python experiments/transfer.py fit --data train_v0 --tag "$tag" --reps R3tick R0 R0r R0info --seeds 0 1 2 \
        --threads 16 --rounds "$rounds" --params "{\"num_leaves\": $leaves}" > "logs/fit_${tag}.log" 2>&1 \
        && log "done $tag" || log "FAILED $tag"
}
fit tunea 1000 63
fit tuneb 300 255
fit tunec 1000 255
log "queue done"

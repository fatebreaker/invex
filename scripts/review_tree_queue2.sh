#!/usr/bin/env bash
# Tree fitted-Q parity check for the TMLR review, round 1 (exploratory, validation), one fit per model so that a
# restart skips the models already saved: R3tick, R0, R0r and R0info refitted with 1,000 rounds (tunea), 255 leaves
# (tuneb) or both (tunec), seeds 0-2; every other setting as registered.
#   setsid nohup nice -n 5 scripts/review_tree_queue2.sh >> logs/review_tree_queue2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=16
log() { echo "[$(date '+%F %T')] $*"; }
for spec in "tunea 1000 63" "tuneb 300 255" "tunec 1000 255"; do
    set -- $spec
    tag=$1 rounds=$2 leaves=$3
    for rep in R3tick R0 R0r R0info; do
        for seed in 0 1 2; do
            [ -s "results/transfer/models/${tag}_${rep}_s${seed}.pkl" ] && continue
            log "start ${tag}_${rep}_s${seed} ($rounds rounds, $leaves leaves)"
            python experiments/transfer.py fit --data train_v0 --tag "$tag" --reps "$rep" --seeds "$seed" --threads 16 \
                --rounds "$rounds" --params "{\"num_leaves\": $leaves}" >> "logs/fit_${tag}.log" 2>&1 \
                && log "done ${tag}_${rep}_s${seed}" || log "FAILED ${tag}_${rep}_s${seed}"
        done
    done
done
log "queue done"

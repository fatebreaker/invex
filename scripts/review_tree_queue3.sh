#!/usr/bin/env bash
# Tree capacity check, reduced (TMLR review round 1; exploratory, validation): after the 1,000-round MS+tick fits
# (tunea_R3tick_s0-2), one seed per remaining configuration: R0, R0r and R0info with 1,000 rounds (tunea); R3tick, R0,
# R0r and R0info with 255 leaves (tuneb) and with both (tunec). Waits for the running fit to finish.
#   setsid nohup nice -n 5 scripts/review_tree_queue3.sh >> logs/review_tree_queue3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=16
log() { echo "[$(date '+%F %T')] $*"; }
until [ -s results/transfer/models/tunea_R3tick_s2.pkl ]; do sleep 60; done
for spec in "tunea 1000 63 R0 R0r R0info" "tuneb 300 255 R3tick R0 R0r R0info" "tunec 1000 255 R3tick R0 R0r R0info"; do
    set -- $spec
    tag=$1 rounds=$2 leaves=$3
    shift 3
    for rep in "$@"; do
        [ -s "results/transfer/models/${tag}_${rep}_s0.pkl" ] && continue
        log "start ${tag}_${rep}_s0 ($rounds rounds, $leaves leaves)"
        python experiments/transfer.py fit --data train_v0 --tag "$tag" --reps "$rep" --seeds 0 --threads 16 \
            --rounds "$rounds" --params "{\"num_leaves\": $leaves}" >> "logs/fit_${tag}.log" 2>&1 \
            && log "done ${tag}_${rep}_s0" || log "FAILED ${tag}_${rep}_s0"
    done
done
log "queue done"

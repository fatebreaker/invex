#!/usr/bin/env bash
# Review round 4 (R5; exploratory, validation only): domain randomisation in time, price and size
# (experiments/time_dr.py): stretched training stock-days, neural fits on raw features (cost in spreads, R0r; in
# cents, R0), validation evaluation.
#   setsid nohup scripts/review4_tdr_queue.sh >> logs/review4_tdr.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
[ -s data/transitions/tdr_v0/data.npz ] || python experiments/time_dr.py gen --name tdr_v0 --workers 12 \
    > logs/tdr_gen.log 2>&1 && log "gen done" || log "gen FAILED"
CUDA_VISIBLE_DEVICES=6 python experiments/time_dr.py fit --reps R0r R0 --seeds 0 1 2 --device cuda \
    > logs/tdr_fit.log 2>&1 && log "fit done" || log "fit FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 \
    --models 'tdrnn_*' --out val_v0_tdr --workers 16 > logs/eval_val_tdr.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

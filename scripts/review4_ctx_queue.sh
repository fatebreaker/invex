#!/usr/bin/env bash
# Review round 4 (R5/R6; exploratory, validation): the learned context encoder (experiments/ctx_fqi.py): build the
# training set's trailing-window series, fit three seeds, evaluate on the validation days.
#   setsid nohup scripts/review4_ctx_queue.sh >> logs/review4_ctx.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
[ -s data/transitions/ctx_train_v0.npy ] || python experiments/ctx_fqi.py build --source train_v0 --workers 12 \
    > logs/ctx_build.log 2>&1 && log "build done" || log "build FAILED"
CUDA_VISIBLE_DEVICES=4 python experiments/ctx_fqi.py fit --seeds 0 1 2 --device cuda > logs/ctx_fit.log 2>&1 \
    && log "fit done" || log "fit FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/ctx_fqi.py eval --episodes val_v0 \
    --out val_v0_ctx --workers 16 > logs/ctx_eval.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

#!/usr/bin/env bash
# Review round 3 (R1; exploratory, validation only): the learning curve for the equal-information arm (R0info, trees)
# at the registered curve sizes and seeds, as for R3tick and R0 (experiments/learning_curve.py), then scored on the
# validation days.
#   setsid nohup scripts/review3_curve_info.sh >> logs/review3_curve_info.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=24 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
python experiments/learning_curve.py fit --reps R0info --seeds 0 1 2 --threads 24 > logs/curve_fit_info.log 2>&1 \
    && log "fit done" || log "fit FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes val_v0 --models 'curve_R0info*' --out val_v0_curve_info \
    --workers 20 > logs/curve_eval_info.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

#!/usr/bin/env bash
# Review round 3 (R6 N4; exploratory, validation only): the equal-information arm (R0info) refit on the same 20
# stock-bootstrap resamples of the training split as the registered refits (tag zboot, seeds 0-19: seed k resamples
# the stocks with default_rng(10000 + k)), every other setting as registered; then scored on the validation days.
#   setsid nohup scripts/review3_info_boot.sh >> logs/review3_info_boot.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=24 OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
fit() {  # log suffix, seeds...
    local part=$1; shift
    python experiments/transfer.py fit --data train_v0 --tag zboot --boot-stocks --reps R0info --seeds "$@" \
        --threads 24 > "logs/fit_zboot_info_${part}.log" 2>&1 && log "fit $part done" || log "fit $part FAILED"
}
log "start"
fit a 0 1 2 3 4 5 6 7 8 9 &
fit b 10 11 12 13 14 15 16 17 18 19 &
wait
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes val_v0 --models 'zboot_R0info_s*' --out val_v0_boot_info \
    --workers 20 > logs/eval_val_boot_info.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

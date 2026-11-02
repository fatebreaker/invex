#!/usr/bin/env bash
# Review round 4 (R6; exploratory, validation only): units from 15- and 60-minute windows. Rebuild the window
# statistics of the validation episodes and of the training transitions, retrain MS+tick (trees) with them, and
# evaluate the registered MS+tick policies and the retrained ones with the other window's units. experiments/window_units.py.
#   setsid nohup scripts/review4_window_queue.sh >> logs/review4_window.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
for W in 900 3600; do
    [ -s results/transfer/window_val_w$W.parquet ] || python experiments/window_units.py build-val --window $W \
        --workers 16 > logs/window_val_w$W.log 2>&1 && log "val w$W built" || log "val w$W FAILED"
    [ -s data/transitions/train_w$W/eps.parquet ] || python experiments/window_units.py build --window $W \
        --source train_v0 --out train_w$W --workers 16 > logs/window_train_w$W.log 2>&1 && log "train w$W built" \
        || log "train w$W FAILED"
    python experiments/transfer.py fit --data train_w$W --tag w$W --reps R3tick R0info --seeds 0 1 2 --threads 16 \
        > logs/fit_w$W.log 2>&1 && log "fit w$W done" || log "fit w$W FAILED"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/window_units.py eval --window $W \
        --models 'zero_R3tick_s[0-2]' 'zero_R0info_s[0-2]' "w${W}_R3tick_s[0-2]" "w${W}_R0info_s[0-2]" \
        --workers 16 > logs/eval_val_w$W.log 2>&1 && log "eval w$W done" || log "eval w$W FAILED"
done
log "queue done"

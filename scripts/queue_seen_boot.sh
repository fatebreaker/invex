#!/usr/bin/env bash
# The seen R3tick comparator refit on the same 20 resamples of the training stocks as the zero-shot refits (tag
# zboot), with the validation stocks' own training days kept whole and the total size-matched, then scored on
# validation (third review: training-data variation of the transfer gap, H1). Exploratory; no sealed table is read.
#   scripts/queue_seen_boot.sh >> logs/queue_seen_boot.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
fit() {
    python experiments/transfer.py fit --data train_v0 valseen_v0 --size-of train_v0 --tag sboot --boot-stocks \
        --reps R3tick --seeds "$@" --threads 16
}
fit 0 1 2 3 4 5 6 7 8 9 > logs/fit_sboot_a.log 2>&1 &
fit 10 11 12 13 14 15 16 17 18 19 > logs/fit_sboot_b.log 2>&1 &
wait
log "fits done"
python experiments/transfer.py eval --episodes val_v0 --models "sboot_*" --out val_v0_sboot > logs/eval_val_sboot.log 2>&1
log "validation done"

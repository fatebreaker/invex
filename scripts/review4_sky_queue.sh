#!/usr/bin/env bash
# Review round 4 (R5 "a non-anticipative bound"; exploratory, validation only): a same-day skyline, MS+tick trees
# trained on the validation stocks' own validation days (other start times, seed tag valsky), size-matched to
# train_v0, then scored on the validation table. It knows each day's market, not the episode's own future at decision
# time; it is optimistic (its training episodes overlap the evaluation's in time). Waits for the contemporaneous queue.
#   setsid nohup scripts/review4_sky_queue.sh >> logs/review4_sky.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=20 OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "queue done" logs/review4_contemp.log; do sleep 60; done
log "start"
[ -s data/transitions/valsky_v0/data.npz ] || python experiments/transfer.py gen --name valsky_v0 --split val \
    --days 2019-10-30 2019-12-30 --per-book 800 --workers 20 > logs/gen_valsky.log 2>&1 \
    && log "gen done" || log "gen FAILED"
python experiments/transfer.py fit --data valsky_v0 --size-of train_v0 --tag sky --reps R3tick R0 --seeds 0 1 2 \
    --threads 20 > logs/fit_sky.log 2>&1 && log "fit done" || log "fit FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 \
    --models 'sky_*' --out val_v0_sky --workers 20 > logs/eval_val_sky.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

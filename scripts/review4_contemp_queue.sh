#!/usr/bin/env bash
# Review round 4 (R5 "contemporaneous skylines"; exploratory, post hoc): tree policies trained on the 2019
# training-split stocks' own days in the test periods (Stage 1: 2020-21; Stage 2: 2025-26), size-matched to train_v0,
# then scored once on each sealed stage table (one logged read each). Zero-shot (2019) minus contemporaneous isolates
# the cost of the years; no training stock is in either sealed table.
#   setsid nohup scripts/review4_contemp_queue.sh >> logs/review4_contemp.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=20 OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
[ -s data/transitions/ctmp1_v0/data.npz ] || python experiments/transfer.py gen --name ctmp1_v0 --split train \
    --days 2020-01-30 2021-07-13 2021-08-13 --per-book 180 --workers 20 > logs/gen_ctmp1.log 2>&1 \
    && log "gen ctmp1 done" || log "gen ctmp1 FAILED"
[ -s data/transitions/ctmp2_v0/data.npz ] || python experiments/transfer.py gen --name ctmp2_v0 --split train \
    --days 2025-12-08 2025-12-09 2025-12-10 2025-12-11 2025-12-12 2026-05-15 2026-05-18 2026-06-12 --per-book 100 \
    --workers 20 > logs/gen_ctmp2.log 2>&1 && log "gen ctmp2 done" || log "gen ctmp2 FAILED"
python experiments/transfer.py fit --data ctmp1_v0 --size-of train_v0 --tag ctmp1 --reps R3tick R0 R0info \
    --seeds 0 1 2 --threads 20 > logs/fit_ctmp1.log 2>&1 && log "fit ctmp1 done" || log "fit ctmp1 FAILED"
python experiments/transfer.py fit --data ctmp2_v0 --size-of train_v0 --tag ctmp2 --reps R3tick R0 R0info \
    --seeds 0 1 2 --threads 20 > logs/fit_ctmp2.log 2>&1 && log "fit ctmp2 done" || log "fit ctmp2 FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes stage1_v0 --models 'ctmp1_*' --out stage1_ctmp --unseal \
    --reason "post hoc (review round 4): contemporaneous skyline, policies trained on the training stocks' 2020-21 days (exploratory)" \
    --workers 20 > logs/eval_stage1_ctmp.log 2>&1 && log "eval stage1 done" || log "eval stage1 FAILED"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes stage2_v0 --models 'ctmp2_*' --out stage2_ctmp --unseal \
    --reason "post hoc (review round 4): contemporaneous skyline, policies trained on the training stocks' 2025-26 days (exploratory)" \
    --workers 20 > logs/eval_stage2_ctmp.log 2>&1 && log "eval stage2 done" || log "eval stage2 FAILED"
log "queue done"

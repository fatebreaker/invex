#!/usr/bin/env bash
# Review round 4 (R6; exploratory, validation only): invariance penalties (V-REx, GroupDRO, IRMv1) with scale
# environments, neural fitted Q-iteration on raw features with the cost in spreads (R0r) and on RAW+INFO (R0info),
# two strengths each, seed 0; then the validation evaluation. experiments/inv_fqi.py.
#   setsid nohup scripts/review4_inv_queue.sh >> logs/review4_inv.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
CUDA_VISIBLE_DEVICES=6 python experiments/inv_fqi.py --method vrex --strength 1 10 --seeds 0 > logs/fit_inv_vrex.log 2>&1 \
    && log "vrex done" || log "vrex FAILED" &
( CUDA_VISIBLE_DEVICES=7 python experiments/inv_fqi.py --method gdro --strength 1 10 --seeds 0 > logs/fit_inv_gdro.log 2>&1 \
    && log "gdro done" || log "gdro FAILED"
  CUDA_VISIBLE_DEVICES=7 python experiments/inv_fqi.py --method irm --strength 1 10 --seeds 0 > logs/fit_inv_irm.log 2>&1 \
    && log "irm done" || log "irm FAILED" ) &
wait
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes val_v0 --models 'vrexnn*' 'gdronn*' 'irmnn*' --out val_v0_inv \
    --workers 16 > logs/eval_val_inv.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

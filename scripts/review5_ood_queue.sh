#!/usr/bin/env bash
# Review round 5 (R5 "score the learned alternatives where the units matter"; exploratory, validation days only):
# the learned alternatives to the units on IEX's validation days (iex_val_v0: 2026-09-16 and 10-14, validation-split
# stocks; 83% of its episodes are on wide books, where the units' division matters), with three seeds each.
#   1. seeds 1 and 2 of the invariance penalties at their better strengths (IRM 1, V-REx 1, GroupDRO 10), GPU 7;
#   2. the trailing-window series of the IEX validation episodes for the context encoder;
#   3. Nasdaq validation scores of the new penalty seeds (val_v0_inv2);
#   4. IEX validation scores of the penalties, time randomisation, RAW+INFO (neural) and the context encoder.
# No sealed table is read.
#   setsid nohup scripts/review5_ood_queue.sh >> logs/review5_ood.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
( for m in "irm 1" "vrex 1" "gdro 10"; do
    set -- $m
    CUDA_VISIBLE_DEVICES=7 python experiments/inv_fqi.py --method "$1" --strength "$2" --seeds 1 2 \
        > "logs/fit_inv_${1}_s12.log" 2>&1 && log "$1 seeds 1-2 done" || log "$1 seeds 1-2 FAILED"
  done ) &
FITS=$!
[ -s results/transfer/ctx_iex_val_v0.npy ] || python experiments/ctx_fqi.py build-val --episodes iex_val_v0 --workers 12 \
    > logs/ctx_build_iexval.log 2>&1 && log "ctx series done" || log "ctx series FAILED"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/ctx_fqi.py eval --episodes iex_val_v0 --out iex_val_v0_ctx --workers 16 \
    > logs/ctx_eval_iexval.log 2>&1 && log "ctx eval done" || log "ctx eval FAILED"
python experiments/transfer.py eval --episodes iex_val_v0 --models 'tdrnn_*' 'zeronn_R0info_*' \
    --out iex_val_v0_tdr --workers 16 > logs/eval_iexval_tdr.log 2>&1 && log "tdr eval done" || log "tdr eval FAILED"
wait $FITS
python experiments/transfer.py eval --episodes val_v0 --models 'vrexnn1_*' 'gdronn10_*' 'irmnn1_*' \
    --out val_v0_inv2 --workers 16 > logs/eval_val_inv2.log 2>&1 && log "inv val eval done" || log "inv val eval FAILED"
python experiments/transfer.py eval --episodes iex_val_v0 --models 'vrexnn*' 'gdronn*' 'irmnn*' \
    --out iex_val_v0_inv --workers 16 > logs/eval_iexval_inv.log 2>&1 && log "inv iex eval done" || log "inv iex eval FAILED"
log "queue done"

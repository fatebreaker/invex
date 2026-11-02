#!/usr/bin/env bash
# Main real-data run on the validation split, decisive evidence first (resumable by output files).
#   setsid nohup scripts/queue_transfer.sh >> logs/queue_transfer.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
REPS="R0 R0desc R1 R2 R3 R3s R3d R2tick R3tick Rpi Rdim"
M=results/transfer/models
step() { echo "[$(date +%H:%M:%S)] $1"; }
have_all() { for r in $REPS; do [ -s "$M/$1_${r}_s$2.pkl" ] || return 1; done; }

step "zero-shot models, seed 0"
have_all zero 0 || python experiments/transfer.py fit --data train_v0 --tag zero --reps $REPS --seeds 0 > logs/fit_zero_s0.log 2>&1
step "validation: zero-shot seed 0"
[ -s results/transfer/eval_val_v0_zero_s0.parquet ] || \
    python experiments/transfer.py eval --episodes val_v0 --tags zero --out val_v0_zero_s0 > logs/eval_val_zero_s0.log 2>&1
step "seen models, seed 0"
have_all seen 0 || python experiments/transfer.py fit --data train_v0 valseen_v0 --size-of train_v0 --tag seen --reps $REPS \
    --seeds 0 > logs/fit_seen_s0.log 2>&1
step "zero-shot models, seeds 1 and 2"
for s in 1 2; do
    have_all zero $s || python experiments/transfer.py fit --data train_v0 --tag zero --reps $REPS --seeds $s > logs/fit_zero_s$s.log 2>&1
done
step "validation: all models"
[ -s results/transfer/eval_val_v0.parquet ] || \
    python experiments/transfer.py eval --episodes val_v0 --tags zero seen --out val_v0 > logs/eval_val.log 2>&1
step "analysis"
python experiments/analyze_transfer.py --eval val_v0 --macro-prefix Val > logs/analysis_val.log 2>&1
step "done"

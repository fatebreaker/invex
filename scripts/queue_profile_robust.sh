#!/usr/bin/env bash
# The demeaned-markout world at s = 16 (indwa) with a profile whose between-stock variance is floored at the real
# profile's (stock_profile.py build --tau2-floor base -> profiles/indwarb*), so that the comparator is not handed a
# constant: the comparator retrained with it on the same episodes and seeds, scored against indwa's blind and oracle
# policies. Exploratory, validation only.
#   setsid nohup scripts/queue_profile_robust.sh >> logs/queue_profile_robust.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
log() { echo "[$(date +%H:%M:%S)] $*"; }
DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
w=indwa; n=indwarb
log "$n: comparator with the floored profile in $w"
[ -s data/transitions/train_${w}_v0/data.npz ] || python experiments/transfer.py gen --name train_${w}_v0 \
    --split train --days $DAYS --per-book 100 --seed-tag train_v0 --surcharge $w --workers 16 > logs/gen_${w}_rb.log 2>&1
INVEX_PRED_THREADS=8 python experiments/transfer.py fit --data train_${w}_v0 --tag $n --reps R3tickp --seeds 0 1 2 \
    --threads 8 --profile $n > logs/fit_$n.log 2>&1
$ONE python experiments/transfer.py eval --episodes val_v0 --models "${n}_*" --profile $n --planted $w \
    --out val_v0_$n --workers 16 > logs/eval_val_$n.log 2>&1
python experiments/stock_profile.py score --eval val_v0_$w val_v0_${w}x val_v0_$n --profile $n \
    --contrasts ${n}_R3tickp:${w}_R3tick ${w}x_R3tickp:${w}_R3tick ${w}_R3tickp:${w}_R3tick --out $n \
    > logs/profile_score_$n.log 2>&1
cat logs/profile_score_$n.log
rm -rf data/transitions/train_${w}_v0
log "done"

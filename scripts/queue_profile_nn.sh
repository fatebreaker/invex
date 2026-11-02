#!/usr/bin/env bash
# The stock-profile comparator with the neural learner (five seeds, GPUs 2, 3, 6, 7): on the real data against the
# neural MS+tick policies, and in one planted world (surcharge scale 1, draw 1) as its positive control.
# Exploratory, post hoc, validation only.
#   setsid nohup scripts/queue_profile_nn.sh >> logs/queue_profile_nn.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
GPUS=(2 3 6 7 2)
DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
fitnn() {  # data tag profile reps...
    local data=$1 tag=$2 prof=$3; shift 3
    for sd in 0 1 2 3 4; do
        [ -s results/transfer/models/${tag}_${1}_s$sd.pkl ] && continue
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/transfer.py fit --data $data --tag $tag --learner nn \
            --device cuda:${GPUS[$sd]} --reps "$@" --seeds $sd --profile $prof > logs/fit_${tag}_s$sd.log 2>&1 &
        [ $sd = 3 ] && wait
    done
    wait
}
until [ -s results/transfer/profiles/base_lodo.parquet ]; do sleep 30; done
fitnn train_v0 profnn base R3tickp
$ONE python experiments/transfer.py eval --episodes val_v0 --models 'profnn_R3tickp_s[0-4]' 'zeronn_R3tick_s[0-4]' \
    --profile base --out val_v0_profnn --workers 16 > logs/eval_val_profnn.log 2>&1
python experiments/stock_profile.py score --eval val_v0_profnn --contrasts profnn_R3tickp:zeronn_R3tick \
    --out val_base_nn > logs/profile_score_base_nn.log 2>&1
cat logs/profile_score_base_nn.log
log "real data done"
# the planted world injoa (built by scripts/queue_profile_inject.sh), with its own copy of the transitions
until [ -s results/transfer/profiles/injoa_oracle_lodo.parquet ]; do sleep 60; done
[ -s data/transitions/train_injoann_v0/data.npz ] || python experiments/transfer.py gen --name train_injoann_v0 \
    --split train --days $DAYS --per-book 100 --seed-tag train_v0 --surcharge injoa --workers 16 > logs/gen_injoann.log 2>&1
fitnn train_injoann_v0 injoann injoa R3tick R3tickp
fitnn train_injoann_v0 injoannx injoa_oracle R3tickp
$ONE python experiments/transfer.py eval --episodes val_v0 --models 'injoann_*' --profile injoa --out val_v0_injoann \
    --workers 16 > logs/eval_val_injoann.log 2>&1
$ONE python experiments/transfer.py eval --episodes val_v0 --models 'injoannx_*' --profile injoa_oracle \
    --out val_v0_injoannx --workers 16 > logs/eval_val_injoannx.log 2>&1
python experiments/stock_profile.py score --eval val_v0_injoann val_v0_injoannx --profile injoa \
    --contrasts injoann_R3tickp:injoann_R3tick injoannx_R3tickp:injoann_R3tick --out injoann > logs/profile_score_injoann.log 2>&1
cat logs/profile_score_injoann.log
rm -rf data/transitions/train_injoann_v0
log "done"

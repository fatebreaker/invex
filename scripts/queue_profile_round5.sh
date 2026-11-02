#!/usr/bin/env bash
# After the fifth review: more planted worlds for the stock-profile comparator (as scripts/queue_profile_inject.sh),
# with a second kind of planted structure. Specs "<name> <scale> <draw> <kind>": kind cross (a per-stock crossing
# surcharge phi) or passive (a per-stock markout psi on shares filled while resting). Exploratory, validation only.
#   setsid nohup scripts/queue_profile_round5.sh "inmoa 1 1 passive" "injsa 0.6 1 cross" ... \
#       >> logs/queue_profile_round5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
log() { echo "[$(date +%H:%M:%S)] $*"; }
DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
for spec in "$@"; do
    set -- $spec
    name=$1
    log "$name: scale $2, draw $3, $4"
    [ -s results/transfer/profiles/${name}_oracle_lodo.parquet ] || python experiments/stock_profile.py build \
        --name $name --scale $2 --draw $3 --kind $4 > logs/profile_build_$name.log 2>&1
    [ -s data/transitions/train_${name}_v0/data.npz ] || python experiments/transfer.py gen --name train_${name}_v0 \
        --split train --days $DAYS --per-book 100 --seed-tag train_v0 --surcharge $name --workers 16 \
        > logs/gen_$name.log 2>&1
    INVEX_PRED_THREADS=8 python experiments/transfer.py fit --data train_${name}_v0 --tag $name --reps R3tick R3tickp \
        --seeds 0 1 2 --threads 8 --profile $name > logs/fit_$name.log 2>&1 &
    INVEX_PRED_THREADS=8 python experiments/transfer.py fit --data train_${name}_v0 --tag ${name}x --reps R3tickp \
        --seeds 0 1 2 --threads 8 --profile ${name}_oracle > logs/fit_${name}x.log 2>&1 &
    wait
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${name}_*" --profile $name --out val_v0_$name \
        --workers 16 > logs/eval_val_$name.log 2>&1
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${name}x_*" --profile ${name}_oracle \
        --out val_v0_${name}x --workers 16 > logs/eval_val_${name}x.log 2>&1
    python experiments/stock_profile.py score --eval val_v0_$name val_v0_${name}x --profile $name \
        --contrasts ${name}_R3tickp:${name}_R3tick ${name}x_R3tickp:${name}_R3tick --out $name \
        > logs/profile_score_$name.log 2>&1
    cat logs/profile_score_$name.log
    rm -rf data/transitions/train_${name}_v0
    log "$name done"
done
log "all done"

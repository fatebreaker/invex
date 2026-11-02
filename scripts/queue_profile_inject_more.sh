#!/usr/bin/env bash
# More planted worlds for the stock-profile comparator, as in scripts/queue_profile_inject.sh: for each scale a hidden
# per-stock crossing surcharge is drawn, the training transitions are rebuilt with it in every cost (same episodes and
# behaviour path as train_v0), and three policies are fitted: MS+tick (blind), MS+tick plus the measured profile (the
# comparator) and MS+tick plus the surcharge itself (oracle). All are scored on the validation days with the surcharge
# counted. Exploratory, post hoc, validation only.
#   setsid nohup scripts/queue_profile_inject_more.sh "q 0.75 1" "q 0.75 2" >> logs/queue_profile_inject.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
# evaluation workers run one thread each; fits keep OpenMP, with the trees' prediction threads set explicitly (the
# backward induction predicts every branch at every stage)
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
log() { echo "[$(date +%H:%M:%S)] $*"; }
DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
# more planted worlds, after scripts/queue_profile_inject.sh: specs "<label> <scale> <draw>" as arguments
until grep -q "\] done" logs/queue_profile_inject.log; do sleep 60; done
for spec in "$@"; do
    set -- $spec
    name=inj$1$( [ "$3" = 1 ] && echo a || echo b )
    log "$name: scale $2, draw $3"
    python experiments/stock_profile.py build --name $name --scale $2 --draw $3 > logs/profile_build_$name.log 2>&1
    [ -s data/transitions/train_${name}_v0/data.npz ] || python experiments/transfer.py gen --name train_${name}_v0 \
        --split train --days $DAYS --per-book 100 --seed-tag train_v0 --surcharge $name --workers 24 \
        > logs/gen_$name.log 2>&1
    INVEX_PRED_THREADS=16 python experiments/transfer.py fit --data train_${name}_v0 --tag $name --reps R3tick R3tickp \
        --seeds 0 1 2 --threads 16 --profile $name > logs/fit_$name.log 2>&1 &
    INVEX_PRED_THREADS=16 python experiments/transfer.py fit --data train_${name}_v0 --tag ${name}x --reps R3tickp \
        --seeds 0 1 2 --threads 16 --profile ${name}_oracle > logs/fit_${name}x.log 2>&1 &
    wait
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${name}_*" --profile $name --out val_v0_$name \
        --workers 24 > logs/eval_val_$name.log 2>&1
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${name}x_*" --profile ${name}_oracle \
        --out val_v0_${name}x --workers 24 > logs/eval_val_${name}x.log 2>&1
    python experiments/stock_profile.py score --eval val_v0_$name val_v0_${name}x --profile $name \
        --contrasts ${name}_R3tickp:${name}_R3tick ${name}x_R3tickp:${name}_R3tick --out $name \
        > logs/profile_score_$name.log 2>&1
    cat logs/profile_score_$name.log
    rm -rf data/transitions/train_${name}_v0
done
log "done"

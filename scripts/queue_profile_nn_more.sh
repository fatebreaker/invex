#!/usr/bin/env bash
# The neural stock-profile comparator in more planted worlds (built by scripts/queue_profile_inject*.sh), five seeds
# on GPUs 2, 3, 6, 7: its power where the trees' comparator recovers little. Exploratory, post hoc, validation only.
#   setsid nohup scripts/queue_profile_nn_more.sh injqa injha >> logs/queue_profile_nn_more.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
GPUS=(${NN_GPUS:-2 3 6 7 2})  # five seeds; NN_GPUS="7 7 7 7 7" for one GPU
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
for w in "$@"; do
    log "$w: neural"
    nn=${w}nn
    # a world planted in the replay itself (hidden queue or markout) is evaluated with it too
    PLANTED=$(cd experiments && python -c "import transfer as TR; print('--planted $w' if TR.planted_rules('$w') else '')")
    [ -s data/transitions/train_${nn}_v0/data.npz ] || python experiments/transfer.py gen --name train_${nn}_v0 \
        --split train --days $DAYS --per-book 100 --seed-tag train_v0 --surcharge $w --workers 16 > logs/gen_${nn}.log 2>&1
    fitnn train_${nn}_v0 $nn $w R3tick R3tickp
    fitnn train_${nn}_v0 ${nn}x ${w}_oracle R3tickp
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${nn}_*" --profile $w $PLANTED --out val_v0_$nn \
        --workers 16 > logs/eval_val_$nn.log 2>&1
    $ONE python experiments/transfer.py eval --episodes val_v0 --models "${nn}x_*" --profile ${w}_oracle $PLANTED \
        --out val_v0_${nn}x --workers 16 > logs/eval_val_${nn}x.log 2>&1
    python experiments/stock_profile.py score --eval val_v0_$nn val_v0_${nn}x --profile $w \
        --contrasts ${nn}_R3tickp:${nn}_R3tick ${nn}x_R3tickp:${nn}_R3tick --out $nn > logs/profile_score_$nn.log 2>&1
    cat logs/profile_score_$nn.log
    rm -rf data/transitions/train_${nn}_v0
    log "$w done"
done
log "all done"

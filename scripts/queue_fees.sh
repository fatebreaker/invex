#!/usr/bin/env bash
# Fee-aware training (fourth review round): the training transitions rebuilt with every cost net of Nasdaq's taker
# fee and maker rebate, on the same episodes and behaviour path as train_v0 / valseen_v0 / testseen_v0 (--seed-tag);
# zero-shot policies in raw units, z-scores, MS and MS+tick, and seen MS+tick comparators, three seeds each; scored
# with the gross-trained policies on validation and, logged, on the sealed stages. Exploratory and post hoc.
#   scripts/queue_fees.sh >> logs/queue_fees.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
REASON="exploratory, post hoc (fourth review round): fee-aware training"
DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
for s in train valseen testseen; do
    split=${s%seen}  # train, val, test
    [ -s data/transitions/${s}_fee_v0/data.npz ] || python experiments/transfer.py gen --name ${s}_fee_v0 \
        --split $split --days $DAYS --per-book 100 --seed-tag ${s}_v0 --fees nasdaq --workers 24 \
        > logs/gen_${s}_fee.log 2>&1
done
log "transitions done"
python experiments/transfer.py fit --data train_fee_v0 --tag feezero --reps R0 R1 --seeds 0 1 2 --threads 12 \
    > logs/fit_feezero_a.log 2>&1 &
python experiments/transfer.py fit --data train_fee_v0 --tag feezero --reps R3 R3tick --seeds 0 1 2 --threads 12 \
    > logs/fit_feezero_b.log 2>&1 &
wait
python experiments/transfer.py fit --data train_fee_v0 valseen_fee_v0 --size-of train_fee_v0 --tag feeseen \
    --reps R3tick --seeds 0 1 2 --threads 12 > logs/fit_feeseen.log 2>&1 &
python experiments/transfer.py fit --data train_fee_v0 testseen_fee_v0 --size-of train_fee_v0 --tag feeseent \
    --reps R3tick --seeds 0 1 2 --threads 12 > logs/fit_feeseent.log 2>&1 &
wait
log "fits done"
GROSS=("zero_R0_s[0-2]" "zero_R1_s[0-2]" "zero_R3_s[0-2]" "zero_R3tick_s[0-2]")
python experiments/transfer.py eval --episodes val_v0 --models "feezero_*" "feeseen_*" "${GROSS[@]}" "seen_R3tick_s0" \
    --out val_v0_fee --workers 24 > logs/eval_val_fee.log 2>&1
log "validation done"
for st in stage1 stage2; do
    python experiments/transfer.py eval --episodes ${st}_v0 --unseal --reason "$REASON" --models "feezero_*" \
        "feeseent_*" "${GROSS[@]}" "seent_R3tick_s[0-2]" --out ${st}_fee --workers 24 > logs/eval_${st}_fee.log 2>&1
    log "$st done"
done

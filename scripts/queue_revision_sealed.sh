#!/usr/bin/env bash
# EXPLORATORY evaluations of the review-response models on the already-opened sealed tables. Each load is
# logged in prereg/access_log.jsonl with the reason below. Run after scripts/queue_revision.sh.
#   scripts/queue_revision_sealed.sh >> logs/queue_revision.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
REASON="exploratory, post hoc (review response): neural learner, learned normalisation, stock identity, seeds 3-7"
log() { echo "[$(date +%H:%M:%S)] $*"; }
NEW_STOCK=("zeronn_*" "zero_R0pop_*" "zero_R3tickdesc_*" "zero_*_s[3-7]")

# validation first (not sealed): the MS+tick+descriptors arm, trained after the first validation scoring
python experiments/transfer.py eval --episodes val_v0 --models "zero_R3tickdesc_*" "zeronn_R3tickdesc_*" \
    --out val_v0_rev2 > logs/eval_val_rev2.log 2>&1
log "validation (descriptors arm) done"

python experiments/transfer.py eval --episodes stage1_v0 --unseal --reason "$REASON; extrapolation seeds 1-2" \
    --models "${NEW_STOCK[@]}" "seentid_*" "seentidr_*" "seent_R3tick_s[3-7]" "x*_*_s[12]" --out stage1_rev \
    > logs/eval_stage1_rev.log 2>&1
log "stage 1 done"
python experiments/transfer.py eval --episodes stage2_v0 --unseal --reason "$REASON" \
    --models "${NEW_STOCK[@]}" "seentid_*" "seentidr_*" "seent_R3tick_s[3-7]" --out stage2_rev \
    > logs/eval_stage2_rev.log 2>&1
log "stage 2 done"
python experiments/transfer.py eval --episodes crypto_test_v0 --unseal --reason "$REASON" \
    --models "${NEW_STOCK[@]}" --out crypto_test_rev > logs/eval_crypto_rev.log 2>&1
log "sealed exploratory evaluations done"

#!/usr/bin/env bash
# Neural learner seeds 3-4 (five in all), scored on validation and, logged and exploratory, on the sealed stock stages
# and the crypto test (second review: more seeds for the neural arms).
#   scripts/queue_nn_seeds.sh >> logs/queue_nn_seeds.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
REASON="exploratory, post hoc (second review): neural learner seeds 3-4"
log() { echo "[$(date +%H:%M:%S)] $*"; }
NNREPS="R3tick R0 R0r R0desc R3 R1 R2 R0pop R0aug R3tickdesc"
for s in 3 4; do
    ( CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      python experiments/transfer.py fit --data train_v0 --tag zeronn --learner nn --device cuda \
          --reps $NNREPS --seeds $s > logs/fit_zeronn_s$s.log 2>&1 ) &
done
wait
log "fits done"
python experiments/transfer.py eval --episodes val_v0 --models "zeronn_*_s[34]" --out val_v0_rev5 \
    > logs/eval_val_rev5.log 2>&1
log "validation done"
python experiments/transfer.py eval --episodes stage1_v0 --unseal --reason "$REASON" --models "zeronn_*_s[34]" \
    --out stage1_rev2 > logs/eval_stage1_rev2.log 2>&1
log "stage 1 done"
python experiments/transfer.py eval --episodes stage2_v0 --unseal --reason "$REASON" --models "zeronn_*_s[34]" \
    --out stage2_rev2 > logs/eval_stage2_rev2.log 2>&1
log "stage 2 done"
python experiments/transfer.py eval --episodes crypto_test_v0 --unseal --reason "$REASON" --models "zeronn_*_s[34]" \
    --out crypto_test_rev2 > logs/eval_crypto_rev2.log 2>&1
log "crypto done"

#!/usr/bin/env bash
# Robustness arms requested by the review, all EXPLORATORY (after the pre-registered sealed tests):
#   GPU lane: neural FQI (zeronn_*) for the main representations and the learned-normalisation arms
#   CPU lane: per-stock learned target scale on trees (zero_R0pop), stock-identity comparators (seenid on
#             validation stocks, seentid on test stocks), five more seeds (3-7) of the main zero-shot and seen
#             policies, learning-curve seeds 1-2
# then evaluations on validation only. The sealed-table evaluations are in scripts/queue_revision_sealed.sh.
#   scripts/queue_revision.sh >> logs/queue_revision.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
NNREPS="R3tick R0 R0r R0desc R3 R1 R2 R0pop R0aug"
log() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_lane() {
    for s in 0 1 2; do
        ( CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
          python experiments/transfer.py fit --data train_v0 --tag zeronn --learner nn --device cuda \
              --reps $NNREPS --seeds $s > logs/fit_zeronn_s$s.log 2>&1 ) &
    done
    wait
    log "gpu lane done"
}

cpu_lane() {
    python experiments/transfer.py fit --data train_v0 --tag zero --reps R0pop --seeds 0 1 2 --threads 24 \
        > logs/fit_zero_R0pop.log 2>&1
    log "R0pop done"
    python experiments/transfer.py fit --data train_v0 valseen_v0 --size-of train_v0 --tag seenid \
        --reps R3tickid R0id --seeds 0 1 2 --threads 24 > logs/fit_seenid.log 2>&1
    python experiments/transfer.py fit --data train_v0 testseen_v0 --size-of train_v0 --tag seentid \
        --reps R3tickid R0id --seeds 0 1 2 --threads 24 > logs/fit_seentid.log 2>&1
    log "identity comparators done"
    for s in 3 4 5 6 7; do
        python experiments/transfer.py fit --data train_v0 --tag zero \
            --reps R3tick R0 R0r R3 R0desc R1 R2 Rdim --seeds $s --threads 24 > logs/fit_zero_s$s.log 2>&1
        python experiments/transfer.py fit --data train_v0 valseen_v0 --size-of train_v0 --tag seen \
            --reps R3tick --seeds $s --threads 24 > logs/fit_seen_s$s.log 2>&1
        python experiments/transfer.py fit --data train_v0 testseen_v0 --size-of train_v0 --tag seent \
            --reps R3tick --seeds $s --threads 24 > logs/fit_seent_s$s.log 2>&1
        log "seed $s done"
    done
    python experiments/learning_curve.py fit --seeds 1 2 --threads 24 > logs/curve_fit_s12.log 2>&1
    log "cpu lane done"
}

log "start"
gpu_lane > logs/queue_revision_gpu.log 2>&1 &
cpu_lane > logs/queue_revision_cpu.log 2>&1 &
wait
log "fits done"

python experiments/transfer.py eval --episodes val_v0 \
    --models "zeronn_*" "zero_R0pop_*" "zero_*_s[3-7]" "seenid_*" "seen_R3tick_s[3-7]" \
    --out val_v0_rev > logs/eval_val_rev.log 2>&1
python experiments/transfer.py eval --episodes val_v0 --tags curve --out val_v0_curve > logs/curve_eval.log 2>&1
log "fits and validation done"

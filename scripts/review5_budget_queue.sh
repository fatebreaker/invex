#!/usr/bin/env bash
# Review round 5 (R5, "equal tuning budget: seed 0 only"; exploratory, validation): seeds 1 and 2 of each learner's
# best configuration that has only seed 0 (fitted Q-iteration in raw units: tuneb_R0; PPO: ppovshi_R3tick, ppovs_R0;
# the registered settings already have three seeds), then the validation evaluation. experiments/equal_budget.py.
#   setsid nohup scripts/review5_budget_queue.sh >> logs/review5_budget.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=16
log() { echo "[$(date '+%F %T')] $*"; }
log "start"
for s in 1 2; do
    [ -s "results/transfer/models/tuneb_R0_s${s}.pkl" ] && continue
    python experiments/transfer.py fit --data train_v0 --tag tuneb --reps R0 --seeds "$s" --threads 16 --rounds 300 \
        --params '{"num_leaves": 255}' >> logs/fit_tuneb.log 2>&1 && log "done tuneb_R0_s$s" || log "FAILED tuneb_R0_s$s"
done
export NUMBA_NUM_THREADS=10
run() {  # gpu tag rep seed extra...
    local gpu=$1 tag=$2 rep=$3 seed=$4; shift 4
    [ -s "results/transfer/models/${tag}_${rep}_s${seed}.pkl" ] && { log "skip ${tag}_${rep}_s${seed}"; return; }
    log "start ${tag}_${rep}_s${seed} on GPU $gpu $*"
    CUDA_VISIBLE_DEVICES=$gpu python experiments/ppo.py --rep "$rep" --seed "$seed" --tag "$tag" --data train_v0 \
        --workers 10 --device cuda "$@" > "logs/ppo_${tag}_${rep}_s${seed}.log" 2>&1 \
        && log "done ${tag}_${rep}_s${seed}" || log "FAILED ${tag}_${rep}_s${seed}"
}
lane_a() { for s in 1 2; do run 6 ppovshi R3tick $s --vscale auto --lr 1e-3; done; }
lane_b() { for s in 1 2; do run 7 ppovs R0 $s --vscale auto; done; }
lane_a & lane_b & wait
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 \
    --models 'tuneb_R0_s[12]' 'ppovshi_R3tick_s[12]' 'ppovs_R0_s[12]' --out val_v0_budget2 --workers 16 \
    > logs/eval_val_budget2.log 2>&1 && log "eval done" || log "eval FAILED"
log "queue done"

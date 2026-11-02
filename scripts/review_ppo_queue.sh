#!/usr/bin/env bash
# PPO runs for the TMLR review, round 1 (exploratory, validation): the equal-information arm R0info with the
# registered PPO settings, and a parity check of PPO's settings for raw (R0) and MS+tick units: the value targets
# rescaled (--vscale auto) at three learning rates. Two lanes, one GPU each, run one after another within a lane.
#   setsid nohup scripts/review_ppo_queue.sh >> logs/review_ppo_queue.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=12
log() { echo "[$(date '+%F %T')] $*"; }
run() {  # gpu tag rep seed extra...
    local gpu=$1 tag=$2 rep=$3 seed=$4; shift 4
    [ -s "results/transfer/models/${tag}_${rep}_s${seed}.pkl" ] && { log "skip ${tag}_${rep}_s${seed}"; return; }
    log "start ${tag}_${rep}_s${seed} on GPU $gpu $*"
    CUDA_VISIBLE_DEVICES=$gpu python experiments/ppo.py --rep "$rep" --seed "$seed" --tag "$tag" --data train_v0 \
        --workers 12 --device cuda "$@" > "logs/ppo_${tag}_${rep}_s${seed}.log" 2>&1 \
        && log "done ${tag}_${rep}_s${seed}" || log "FAILED ${tag}_${rep}_s${seed}"
}
lane_a() {
    for s in 0 1 2; do run 6 zeroppo R0info $s; done
    run 6 ppovs R0 0 --vscale auto
    run 6 ppovs R3tick 0 --vscale auto
}
lane_b() {
    run 7 ppovslo R0 0 --vscale auto --lr 1e-4
    run 7 ppovslo R3tick 0 --vscale auto --lr 1e-4
    run 7 ppovshi R0 0 --vscale auto --lr 1e-3
    run 7 ppovshi R3tick 0 --vscale auto --lr 1e-3
}
lane_a & lane_b & wait
log "queue done"

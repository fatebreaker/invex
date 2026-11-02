#!/usr/bin/env bash
# After the fifth review (exploratory, validation only): PPO seen policies (trained on as many episodes that include
# the validation stocks' 2019 days, as fitted Q-iteration's seen comparator) in MS+tick and raw units, so that H1's
# transfer gap can be read with PPO; raw units with a per-stock reward scale (R0pop); and 450-iteration runs of
# MS+tick and raw (longppo) for the learning curves. Three runs at a time on GPUs 2, 3, 6; then validation.
#   setsid nohup scripts/queue_ppo_round5.sh >> logs/queue_ppo_round5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
GPUS=(2 3 6)
run() {  # tag rep seed gpu extra-args...
    local tag=$1 rep=$2 s=$3 gpu=$4; shift 4
    [ -s "results/transfer/models/${tag}_${rep}_s$s.pkl" ] && return
    CUDA_VISIBLE_DEVICES=$gpu python experiments/ppo.py --rep "$rep" --seed "$s" --tag "$tag" --workers 16 \
        --device cuda "$@" > "logs/ppo_${tag}_${rep}_s$s.log" 2>&1
    log "${tag}_${rep}_s$s: $(tail -n 1 logs/ppo_${tag}_${rep}_s$s.log)"
}
batch() {  # tag rep extra-args...: seeds 0-2 in parallel
    local tag=$1 rep=$2; shift 2
    for s in 0 1 2; do run "$tag" "$rep" "$s" "${GPUS[$s]}" "$@" & done
    wait
}
batch seenppo R3tick --data train_v0 valseen_v0 --size-of train_v0
batch seenppo R0 --data train_v0 valseen_v0 --size-of train_v0
batch zeroppo R0pop
batch longppo R3tick --iters 450
batch longppo R0 --iters 450
python experiments/transfer.py eval --episodes val_v0 --models 'seenppo_*_s[0-2]' 'zeroppo_R0pop_s[0-2]' \
    'longppo_*_s[0-2]' --out val_v0_ppo5 --workers 20 > logs/eval_val_v0_ppo5.log 2>&1
log "done"

#!/usr/bin/env bash
# PPO policies (experiments/ppo.py) on the Nasdaq training stock-days: MS+tick and raw units, three seeds, two runs
# at a time; then validation on Nasdaq (val_v0) and IEX (iex_val_v0).
#   setsid nohup scripts/queue_ppo.sh >> logs/queue_ppo.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
# one BLAS / OpenMP thread per rollout worker: the workers' numpy layers would otherwise each start a full set
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
run() {  # rep seed gpu
    [ -s "results/transfer/models/zeroppo_$1_s$2.pkl" ] && return
    CUDA_VISIBLE_DEVICES=$3 python experiments/ppo.py --rep "$1" --seed "$2" --tag zeroppo --workers 16 \
        --device cuda > "logs/ppo_$1_s$2.log" 2>&1
    log "zeroppo_$1_s$2: $(tail -n 1 logs/ppo_$1_s$2.log)"
}
for s in 0 1 2; do
    run R3tick "$s" 6 &
    run R0 "$s" 7 &
    wait
done
python experiments/transfer.py eval --episodes val_v0 --models 'zeroppo_*_s[0-2]' --out val_v0_ppo --workers 20 \
    > logs/eval_val_v0_ppo.log 2>&1
python experiments/transfer.py eval --episodes iex_val_v0 --models 'zeroppo_*_s[0-2]' --out iex_val_v0_ppo \
    --workers 20 > logs/eval_iex_val_v0_ppo.log 2>&1
log "done"

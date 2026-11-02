#!/usr/bin/env bash
# After scripts/queue_ppo.sh: PPO for per-feature z-scores (R1) and tick-free MS units (R3), three seeds, then
# validation of every PPO policy on Nasdaq and IEX.
#   setsid nohup scripts/queue_ppo2.sh >> logs/queue_ppo2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
for _ in $(seq 1 360); do grep -q "\] done" logs/queue_ppo.log 2>/dev/null && break; sleep 20; done
run() {  # rep seed gpu
    [ -s "results/transfer/models/zeroppo_$1_s$2.pkl" ] && return
    CUDA_VISIBLE_DEVICES=$3 python experiments/ppo.py --rep "$1" --seed "$2" --tag zeroppo --workers 16 \
        --device cuda > "logs/ppo_$1_s$2.log" 2>&1
    log "zeroppo_$1_s$2: $(tail -n 1 logs/ppo_$1_s$2.log)"
}
for s in 0 1 2; do
    run R1 "$s" 6 &
    run R3 "$s" 7 &
    wait
done
python experiments/transfer.py eval --episodes val_v0 --models 'zeroppo_R1_s[0-2]' 'zeroppo_R3_s[0-2]' \
    --out val_v0_ppo2 --workers 20 > logs/eval_val_v0_ppo2.log 2>&1
python experiments/transfer.py eval --episodes iex_val_v0 --models 'zeroppo_R1_s[0-2]' 'zeroppo_R3_s[0-2]' \
    --out iex_val_v0_ppo2 --workers 20 > logs/eval_iex_val_v0_ppo2.log 2>&1
log "done"

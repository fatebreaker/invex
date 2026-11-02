#!/usr/bin/env bash
# After the seventh review (exploratory, validation only): PPO with raw features and the cost in the stock's spread
# (R0r, H8's ablation with PPO), three seeds, so that PPO's raw-unit loss splits into features and cost unit as fitted
# Q-iteration's does (tag rawrppo: no registered IEX evaluation's model glob matches it); started once the PPO planted control (queue_ppo_planted.sh) has finished. Then validation.
#   setsid nohup scripts/queue_ppo_round8.sh >> logs/queue_ppo_round8.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
R=${INVEX_RESULTS:-results}
until grep -q "done" logs/queue_ppo_planted.log; do sleep 120; done
log "start"
GPUS=(4 5 6)
for s in 0 1 2; do
    [ -s "$R/transfer/models/rawrppo_R0r_s$s.pkl" ] && continue
    CUDA_VISIBLE_DEVICES=${GPUS[$s]} python experiments/ppo.py --rep R0r --seed $s --tag rawrppo --workers 16 \
        --device cuda > logs/ppo_rawrppo_R0r_s$s.log 2>&1 &
done
wait
log "trained: $(ls $R/transfer/models/rawrppo_R0r_s*.pkl | wc -l) models"
python experiments/transfer.py eval --episodes val_v0 --models 'rawrppo_R0r_s[0-2]' --out val_v0_ppo8 --workers 20 \
    > logs/eval_val_v0_ppo8.log 2>&1
log "done"

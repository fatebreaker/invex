#!/usr/bin/env bash
# After the sixth review: PPO given the stock's probe profile (R3tickp, tag profppo), the strong per-stock comparator
# with the stronger learner; then evaluation on the validation days with the stocks' profiles. Exploratory.
#   setsid nohup scripts/queue_ppo_round6.sh >> logs/queue_ppo_round6.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
# wait for the planted-world queues, which share the CPUs
until grep -q "all done" logs/queue_profile_hidden.log && grep -q "all done" logs/queue_profile_markout.log \
      && grep -q "all done" logs/queue_profile_markout_b.log; do sleep 60; done
GPUS=(${PPO_GPUS:-2 3 6})
for s in 0 1 2; do
    python experiments/ppo.py --rep R3tickp --profile base --seed $s --tag profppo --workers 16 \
        --device cuda:${GPUS[$s]} > logs/ppo_profppo_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do tail -n 1 logs/ppo_profppo_s$s.log | sed "s/^/[$(date +%H:%M:%S)] /"; done
env OMP_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 --models 'profppo_R3tickp_s[0-2]' \
    --profile base --out val_v0_ppo6 --workers 32 > logs/eval_val_ppo6.log 2>&1
log "done"

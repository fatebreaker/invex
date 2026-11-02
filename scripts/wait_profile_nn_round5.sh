#!/usr/bin/env bash
# After the PPO queue frees the GPUs: the neural profile comparator in the passive-markout worlds (s = 1 and 0.6).
#   setsid nohup scripts/wait_profile_nn_round5.sh >> logs/queue_profile_nn_round5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
until grep -q "\] done" logs/queue_ppo_round5.log; do sleep 60; done
until [ -s results/transfer/profiles/inmsa_oracle_lodo.parquet ]; do sleep 60; done
scripts/queue_profile_nn_more.sh inmoa inmsa

#!/usr/bin/env bash
# The fine-tuned reference of the TMLR review, round 1 (exploratory): the Nasdaq-trained PPO MS+tick policies
# (zeroppo_R3tick_s0-2) fine-tuned on the IEX training episodes with the budget of the IEX-trained PPO policies of
# iex-v8 (ixppo: iex_train_v1 subsampled to the size of train_v0, 150 iterations), tag ftppo. Starts when the PPO queue
# of this review has finished (its log says so).
#   setsid nohup nice -n 5 scripts/review_ft_queue.sh >> logs/review_ft_queue.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*"; }
# starts at once on GPU 4 (review round 1)
for s in 0 1 2; do
    [ -s "results/transfer/models/ftppo_R3tick_s${s}.pkl" ] && continue
    log "start ftppo_R3tick_s${s}"
    CUDA_VISIBLE_DEVICES=4 python experiments/ppo.py --rep R3tick --seed "$s" --tag ftppo --data iex_train_v1 \
        --size-of train_v0 --workers 12 --device cuda --init "zeroppo_R3tick_s${s}" > "logs/ppo_ftppo_R3tick_s${s}.log" 2>&1 \
        && log "done ftppo_R3tick_s${s}" || log "FAILED ftppo_R3tick_s${s}"
done
log "queue done"

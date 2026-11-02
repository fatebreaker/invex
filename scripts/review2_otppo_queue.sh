#!/usr/bin/env bash
# Review round 2 (exploratory, post hoc): what P4 measures. PPO trained by the zero-shot recipe on Nasdaq's one-tick
# books only (the 2019 training stocks and days, w_onetick >= 0.5, subsampled to the size of train_v0), tag otppo, then
# scored on the IEX test days (one logged read): it separates specialisation to one-tick books from the venue.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export NUMBA_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*"; }
GPUS=(5 6 7)
for s in 0 1 2; do
    [ -s "results/transfer/models/otppo_R3tick_s${s}.pkl" ] && continue
    log "start otppo_R3tick_s${s} on GPU ${GPUS[$s]}"
    CUDA_VISIBLE_DEVICES=${GPUS[$s]} python experiments/ppo.py --rep R3tick --seed "$s" --tag otppo --data nq_onetick_v1 \
        --size-of train_v0 --workers 8 --device cuda > "logs/ppo_otppo_R3tick_s${s}.log" 2>&1 &
done
wait
for s in 0 1 2; do
    [ -s "results/transfer/models/otppo_R3tick_s${s}.pkl" ] || { log "FAILED otppo_R3tick_s${s}"; exit 1; }
done
log "trained"
python experiments/transfer.py eval --episodes iex_test_v0 --models 'otppo_R3tick_s[0-2]' --out iex_test_ot --unseal \
    --reason "post hoc (review round 2): PPO trained on Nasdaq one-tick books, on the IEX test days (exploratory)" \
    --workers 12 > logs/eval_iex_test_ot.log 2>&1 && log "done iex_test_ot" || log "FAILED iex_test_ot"
log "queue done"

#!/usr/bin/env bash
# A positive control for PPO with the stock's probe profile: PPO trained and evaluated in a planted world (default
# inata, a per-stock markout at s = 2, worth 1.4 margins to the fitted-Q oracle): blind MS+tick, MS+tick with the
# world's profile, and MS+tick with the planted value itself (oracle), three seeds each. Exploratory, validation only.
#   setsid nohup scripts/queue_ppo_planted.sh inata >> logs/queue_ppo_planted.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
w=${1:-inata}
t=ppl$w
GPUS=(${PPO_GPUS:-1 2 3 4 5 6 7 1 2})
i=0
run() {  # tag rep profile seed
    python experiments/ppo.py --rep $2 --planted $w ${3:+--profile $3} --seed $4 --tag $1 --workers 12 \
        --device cuda:${GPUS[$i]} > logs/ppo_$1_$2_s$4.log 2>&1 &
    i=$((i + 1))
}
for s in 0 1 2; do
    run $t R3tick "" $s
    run $t R3tickp $w $s
    run ${t}x R3tickp ${w}_oracle $s
    wait
    log "seed $s trained"
done
OMP_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 --models "${t}_*" --profile $w --markout $w \
    --out val_v0_$t --workers 24 > logs/eval_val_$t.log 2>&1
OMP_NUM_THREADS=1 python experiments/transfer.py eval --episodes val_v0 --models "${t}x_*" --profile ${w}_oracle \
    --markout $w --out val_v0_${t}x --workers 24 > logs/eval_val_${t}x.log 2>&1
python experiments/stock_profile.py score --eval val_v0_$t val_v0_${t}x --profile $w \
    --contrasts ${t}_R3tickp:${t}_R3tick ${t}x_R3tickp:${t}_R3tick --out $t > logs/profile_score_$t.log 2>&1
cat logs/profile_score_$t.log
log "done"

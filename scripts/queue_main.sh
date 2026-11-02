#!/usr/bin/env bash
# Sequential queue (resumable: finished steps are skipped). Steps never wait on process names
# (pgrep -f also matches the shells that launched them).
#   setsid nohup scripts/queue_main.sh >> logs/queue_main.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
TRAIN_DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
step() { echo "[$(date +%H:%M:%S)] $1"; }

step "advantage pilot"
[ -s results/fqi_pilot_adv/summary.json ] || python experiments/fqi_pilot.py --algo adv --stocks 300 --fit-episodes 100 \
    --eval-episodes 100 --rounds 300 --workers 32 > logs/fqi_pilot_adv.log 2>&1

step "transitions train_v0"
[ -s "$INVEX_DATA/transitions/train_v0/data.npz" ] || python experiments/transfer.py gen --name train_v0 --split train \
    --days $TRAIN_DAYS --per-book 100 > logs/gen_train_v0.log 2>&1

step "transitions valseen_v0"
[ -s "$INVEX_DATA/transitions/valseen_v0/data.npz" ] || python experiments/transfer.py gen --name valseen_v0 --split val \
    --days $TRAIN_DAYS --per-book 100 > logs/gen_valseen_v0.log 2>&1

step "extrapolation control pc1x"
[ -s results/zi_controls/pc1x/summary.json ] || python experiments/zi_controls.py --control pc1x > logs/zi_pc1x.log 2>&1

step "done"

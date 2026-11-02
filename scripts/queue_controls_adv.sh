#!/usr/bin/env bash
# Synthetic controls with the advantage learner, at half the threads so the main run keeps going.
#   setsid nohup scripts/queue_controls_adv.sh >> logs/queue_controls_adv.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
export INVEX_FIT_THREADS=16
for c in pc1 pc1x pc2 pc3; do
    echo "[$(date +%H:%M:%S)] $c"
    [ -s results/zi_controls_adv/$c/summary.json ] || \
        python experiments/zi_controls.py --control $c --algo adv --workers 16 > logs/zi_adv_$c.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

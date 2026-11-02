#!/usr/bin/env bash
# The R0r arm (raw features, cost in spreads) in the synthetic controls, with MS units for the paired contrast
# (simulations are seeded: the MS arm reproduces the main run's). After the extrapolation R0r queue.
#   setsid nohup scripts/queue_controls_r0r.sh >> logs/queue_controls_r0r.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 1440); do grep -q "\] done" logs/queue_xtrap_r0r.log 2>/dev/null && break; sleep 60; done
export INVEX_FIT_THREADS=16
for c in pc1 pc1x pc2 pc3; do
    echo "[$(date +%H:%M:%S)] $c"
    [ -s results/zi_controls_adv_r0r/$c/summary.json ] || \
        python experiments/zi_controls.py --control $c --algo adv --reps R0r R3 --out-tag _r0r --workers 16 \
        > logs/zi_adv_r0r_$c.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

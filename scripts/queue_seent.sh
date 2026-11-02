#!/usr/bin/env bash
# Seen policies for the test-split stocks (their 2019 training days; no sealed data), the reference of the Stage-1
# transfer gap. Runs after D-hat, alongside the extrapolation queue, at 16 threads.
#   setsid nohup scripts/queue_seent.sh >> logs/queue_seent.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
TRAIN_DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
REPS="R0 R0desc R1 R2 R3 R3s R3d R2tick R3tick Rpi Rdim"
for _ in $(seq 1 2880); do grep -q "\] done" logs/queue_dhat.log 2>/dev/null && break; sleep 60; done
[ -s data/transitions/testseen_v0/data.npz ] || \
    python experiments/transfer.py gen --name testseen_v0 --split test --days $TRAIN_DAYS --workers 16 > logs/gen_testseen.log 2>&1
for s in 0 1 2; do
    echo "[$(date +%H:%M:%S)] seent seed $s"
    python experiments/transfer.py fit --data train_v0 testseen_v0 --size-of train_v0 --tag seent --reps $REPS \
        --seeds $s --threads 16 > logs/fit_seent_s$s.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

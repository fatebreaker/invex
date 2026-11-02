#!/usr/bin/env bash
# D-hat probes on the training days of training and validation stocks, after the main validation run.
#   setsid nohup scripts/queue_dhat.sh >> logs/queue_dhat.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
TRAIN_DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
for _ in $(seq 1 1440); do grep -q "\] done" logs/queue_transfer.log && break; sleep 60; done
[ -s results/transfer/dhat_train_days.parquet ] || \
    python experiments/transfer.py dhat --splits train val --days $TRAIN_DAYS --per-book 50 --out train_days --workers 32
echo "[$(date +%H:%M:%S)] done"

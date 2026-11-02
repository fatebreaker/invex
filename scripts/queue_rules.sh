#!/usr/bin/env bash
# Tuned queue-rule baselines on validation: per-asset tuning days, pooled tuning stocks, the frozen table.
#   setsid nohup scripts/queue_rules.sh >> logs/queue_rules.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
TRAIN_DAYS="2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18"
[ -s results/rules/grid_val_own.parquet ] || python experiments/rules.py grid --split val --days $TRAIN_DAYS --out val_own
[ -s results/rules/grid_train_pool.parquet ] || \
    python experiments/rules.py grid --split train --days $TRAIN_DAYS --max-stocks 300 --out train_pool
[ -s results/rules/grid_val_v0.parquet ] || python experiments/rules.py grid --episodes val_v0 --out val_v0
echo "[$(date +%H:%M:%S)] done"

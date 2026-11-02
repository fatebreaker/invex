#!/usr/bin/env bash
# After the main validation run: the shuffled-units placebo (all zero-shot seeds) and the analysis with it.
#   setsid nohup scripts/queue_val_extras.sh >> logs/queue_val_extras.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 1440); do grep -q "\] done" logs/queue_transfer.log && break; sleep 60; done
[ -s results/transfer/eval_val_v0_placebo.parquet ] || \
    python experiments/transfer.py eval --episodes val_v0 --tags zero --out val_v0_placebo --placebo shuffle --workers 16 \
    > logs/eval_val_placebo.log 2>&1
python experiments/analyze_transfer.py --eval val_v0 --placebo val_v0_placebo --macro-prefix Val > logs/analysis_val.log 2>&1
python paper/fig_transfer.py --eval val_v0 --out fig_transfer_val >> logs/analysis_val.log 2>&1
echo "[$(date +%H:%M:%S)] done"

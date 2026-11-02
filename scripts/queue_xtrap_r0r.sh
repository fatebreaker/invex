#!/usr/bin/env bash
# After the extrapolation queue: the R0r arm (raw features, cost in spreads) for every scale statistic, then the
# evaluations and analyses again with it.
#   setsid nohup scripts/queue_xtrap_r0r.sh >> logs/queue_xtrap_r0r.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 1440); do grep -q "\] done" logs/queue_xtrap.log 2>/dev/null && break; sleep 60; done
for st in activity tradesize spread; do
    echo "[$(date +%H:%M:%S)] $st"
    python experiments/extrapolation.py fit --stat $st --seeds 0 --reps R0r >> logs/xtrap_fit_$st.log 2>&1
    rm -f results/transfer/eval_val_v0_x$st.parquet results/transfer/eval_val_v0_x$st.json
    python experiments/transfer.py eval --episodes val_v0 --tags x$st xref$st --out val_v0_x$st --workers 32 \
        > logs/xtrap_eval_$st.log 2>&1
    python experiments/extrapolation.py analyze --stat $st > logs/xtrap_analysis_$st.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

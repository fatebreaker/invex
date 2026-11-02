#!/usr/bin/env bash
# Real-data extrapolation (the analogue of PC1x), after D-hat: restricted and size-matched reference models per
# scale statistic, the validation evaluation, the analysis.
#   setsid nohup scripts/queue_xtrap.sh >> logs/queue_xtrap.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
# (started after D-hat)
for st in activity tradesize spread; do
    echo "[$(date +%H:%M:%S)] $st"
    python experiments/extrapolation.py fit --stat $st --seeds 0 --reps R0 R0desc R1 R2 R3 R3tick Rpi Rdim \
        >> logs/xtrap_fit_$st.log 2>&1
    [ -s results/transfer/eval_val_v0_x$st.parquet ] || \
        python experiments/transfer.py eval --episodes val_v0 --tags x$st xref$st --out val_v0_x$st --workers 32 \
        > logs/xtrap_eval_$st.log 2>&1
    python experiments/extrapolation.py analyze --stat $st > logs/xtrap_analysis_$st.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

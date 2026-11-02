#!/usr/bin/env bash
# Robustness on validation, after the main analysis: the zero-shot policies under the other fill rules and
# latencies (same models, same episodes), each analysed like the main evaluation.
#   setsid nohup scripts/queue_robust.sh >> logs/queue_robust.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 1440); do grep -q "\] done" logs/queue_val_extras.log 2>/dev/null && break; sleep 60; done
run() {  # name, extra eval args
    [ -s results/transfer/eval_val_v0_$1.parquet ] || \
        python experiments/transfer.py eval --episodes val_v0 --tags zero --out val_v0_$1 --workers 16 "${@:2}" \
        > logs/eval_val_$1.log 2>&1
    python experiments/analyze_transfer.py --eval val_v0_$1 --macro-prefix Rob$(echo $1 | sed 's/_//g') \
        > logs/analysis_val_$1.log 2>&1
    echo "[$(date +%H:%M:%S)] $1"
}
run pessimistic --rule pessimistic
run optimistic --rule optimistic
run lat0 --latency-us 0
run lat1ms --latency-us 1000
run lat10ms --latency-us 10000
echo "[$(date +%H:%M:%S)] done"

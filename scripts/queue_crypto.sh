#!/usr/bin/env bash
# After every Coinbase day is built and validated: freeze the sealed crypto test table (2024-26; draws episodes
# only, runs no policy) and fit the remaining crypto-trained seeds.
#   setsid nohup scripts/queue_crypto.sh >> logs/queue_crypto.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 1440); do grep -q "\] done" logs/build_coinbase.log 2>/dev/null && break; sleep 60; done
TEST_DAYS="2024-01-01 2024-04-01 2024-07-01 2024-10-01 2025-01-01 2025-04-01 2025-07-01 2025-10-01 2026-01-01 2026-04-01 2026-07-01 2026-10-01"
[ -s prereg/episodes_crypto_test_v0.json ] || \
    python -m invex.eval.freeze --name crypto_test_v0 --venue coinbase --split all --days $TEST_DAYS --per-book 300 \
        --sealed --workers 8
for s in 1 2; do
    python experiments/transfer.py fit --data cb_train_v0 --tag cbseen --seeds $s --threads 8 > logs/fit_cbseen_s$s.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

#!/usr/bin/env bash
# Coinbase L3, quarterly first-of-month days 2020-07 .. 2026-10, 00:00-02:00 UTC, 16 USD pairs spanning
# relative tick sizes (pairs not yet listed on a day are skipped). Resumable.
#   setsid nohup scripts/download_coinbase.sh >> logs/download_coinbase.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
PAIRS="BTC-USD ETH-USD LTC-USD BCH-USD LINK-USD XLM-USD ETC-USD ATOM-USD ALGO-USD XTZ-USD SOL-USD DOGE-USD ADA-USD AVAX-USD DOT-USD SHIB-USD"
DAYS=""
for y in 2020 2021 2022 2023 2024 2025 2026; do
    for m in 01 04 07 10; do
        [[ "$y-$m" < "2020-07" ]] && continue
        DAYS="$DAYS $y-$m-01"
    done
done
python -m invex.coinbase.tardis --pairs $PAIRS --days $DAYS --minutes 120 --workers 8
echo "[$(date +%H:%M:%S)] download done"

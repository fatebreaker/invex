#!/usr/bin/env bash
# Normalise and validate every Coinbase day once the download has finished (resumable: built days are skipped,
# so it must not run on a day whose download is incomplete).
#   setsid nohup scripts/build_coinbase.sh >> logs/build_coinbase.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for _ in $(seq 1 720); do grep -q "download done" logs/download_coinbase.log && break; sleep 60; done
PAIRS="BTC-USD ETH-USD LTC-USD BCH-USD LINK-USD XLM-USD ETC-USD ATOM-USD ALGO-USD XTZ-USD SOL-USD DOGE-USD ADA-USD AVAX-USD DOT-USD SHIB-USD"
DAYS=$(ls "$INVEX_RAW/coinbase" | sort)
python -m invex.coinbase.normalize --days $DAYS --pairs $PAIRS --workers 8
for d in $DAYS; do
    [ -s data/coinbase/$d/book_qa.json ] || python -m invex.book.validate --venue coinbase --day $d --workers 8
done
echo "[$(date +%H:%M:%S)] done"

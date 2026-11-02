#!/usr/bin/env bash
# The descriptive crypto check registered in prereg/iex_prospective_v1.md: run the crypto family on one new
# Coinbase first-of-month day (2026-11-01 or 2026-12-01) with the replication's procedure, once.
#   scripts/run_crypto_prospective.sh 2026-11-01
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v1 > /dev/null || { echo "no iex-v1 tag: refusing"; exit 1; }
D="$1"
case "$D" in 2026-11-01|2026-12-01) ;; *) echo "not a registered day: $D"; exit 1 ;; esac
N="crypto_$(echo "$D" | tr -d '-')_v0"
PAIRS="BTC-USD ETH-USD LTC-USD BCH-USD LINK-USD XLM-USD ETC-USD ATOM-USD ALGO-USD XTZ-USD SOL-USD DOGE-USD ADA-USD AVAX-USD DOT-USD SHIB-USD"
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "download $D"
python -m invex.coinbase.tardis --pairs $PAIRS --days "$D" --minutes 120 --workers 8 >> "logs/download_$N.log" 2>&1
python -m invex.coinbase.normalize --days "$D" --pairs $PAIRS --workers 1 >> "logs/build_$N.log" 2>&1
[ -s "data/coinbase/$D/book_qa.json" ] || python -m invex.book.validate --venue coinbase --day "$D" --workers 8 \
    >> "logs/build_$N.log" 2>&1
[ -s "prereg/episodes_$N.json" ] || python -m invex.eval.freeze --name "$N" --venue coinbase --split all --days "$D" \
    --per-book 300 --sealed --workers 8 >> "logs/build_$N.log" 2>&1
python scripts/freeze_models.py --check
MODELS=$(python -c "import json; m = json.load(open('prereg/models_v1.json'))['models']; \
print(' '.join(sorted(k[:-4] for k in m if k.startswith(('zero_', 'cbseen_')))))")
[ -s "results/transfer/eval_$N.parquet" ] || \
    python experiments/transfer.py eval --episodes "$N" --models $MODELS --out "$N" --unseal \
        --reason "crypto prospective check $D: the single registered evaluation (iex-v1)" --workers 16 \
        > "logs/eval_$N.log" 2>&1
python experiments/confirm.py --spec prereg/confirmatory_v1.json --family crypto --eval "$N" \
    --macro-prefix "CryptoProsp$(date -d "$D" +%b)" > "logs/confirm_$N.log" 2>&1
log "done $D"

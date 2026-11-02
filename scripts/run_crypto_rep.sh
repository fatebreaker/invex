#!/usr/bin/env bash
# The registered crypto replication (prereg/crypto_replication_v1.md): download, build and validate the 22 fresh
# first-of-month Coinbase days, freeze the episode table, evaluate the 69 frozen prereg-v1 policies once, and run the
# registered crypto family. Refuses to run without the crypto-rep-v1 tag.
#   setsid nohup scripts/run_crypto_rep.sh >> logs/crypto_rep.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/crypto-rep-v1 > /dev/null || { echo "no crypto-rep-v1 tag: refusing"; exit 1; }
PAIRS="BTC-USD ETH-USD LTC-USD BCH-USD LINK-USD XLM-USD ETC-USD ATOM-USD ALGO-USD XTZ-USD SOL-USD DOGE-USD ADA-USD AVAX-USD DOT-USD SHIB-USD"
DAYS="2024-02-01 2024-03-01 2024-05-01 2024-06-01 2024-08-01 2024-09-01 2024-11-01 2024-12-01 2025-02-01 2025-03-01
2025-05-01 2025-06-01 2025-08-01 2025-09-01 2025-11-01 2025-12-01 2026-02-01 2026-03-01 2026-05-01 2026-06-01 2026-08-01
2026-09-01"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "download"
python -m invex.coinbase.tardis --pairs $PAIRS --days $DAYS --minutes 120 --workers 8 >> logs/download_coinbase_rep.log 2>&1
log "build"
python -m invex.coinbase.normalize --days $DAYS --pairs $PAIRS --workers 8 >> logs/build_coinbase_rep.log 2>&1
for d in $DAYS; do
    [ -s data/coinbase/$d/book_qa.json ] || python -m invex.book.validate --venue coinbase --day $d --workers 8 \
        >> logs/build_coinbase_rep.log 2>&1
done
log "freeze episodes"
[ -s prereg/episodes_crypto_rep_v0.json ] || \
    python -m invex.eval.freeze --name crypto_rep_v0 --venue coinbase --split all --days $DAYS --per-book 300 \
        --sealed --workers 8 >> logs/build_coinbase_rep.log 2>&1
python scripts/freeze_models.py --check
MODELS=$(python -c "import json; m = json.load(open('prereg/models_v1.json'))['models']; \
print(' '.join(sorted(k[:-4] for k in m if k.startswith(('zero_', 'cbseen_')))))")
log "evaluation ($(echo $MODELS | wc -w) frozen policies)"
[ -s results/transfer/eval_crypto_rep.parquet ] || \
    python experiments/transfer.py eval --episodes crypto_rep_v0 --models $MODELS --out crypto_rep --unseal \
        --reason "crypto replication: the single registered evaluation (crypto-rep-v1)" --workers 16 \
        > logs/eval_crypto_rep.log 2>&1
log "confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_v1.json --family crypto --eval crypto_rep \
    --macro-prefix CryptoRep > logs/confirm_crypto_rep.log 2>&1
log "exploratory"
python experiments/analyze_crypto.py --eval crypto_rep --macro-prefix CryptoRep > logs/analysis_crypto_rep.log 2>&1 || true
log "done"

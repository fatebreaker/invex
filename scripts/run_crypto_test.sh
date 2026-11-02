#!/usr/bin/env bash
# The sealed crypto test (Coinbase, 2024-26): stock-trained and crypto-trained policies, evaluated once; the
# pre-registered crypto family; the exploratory analysis and figure.
#   setsid nohup scripts/run_crypto_test.sh >> logs/crypto_test.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/prereg-v1 > /dev/null || { echo "no prereg-v1 tag: refusing"; exit 1; }
grep -q '"version": "v1"' prereg/confirmatory_v1.json || { echo "spec is still a draft: refusing"; exit 1; }
python scripts/freeze_models.py --check
echo "[$(date +%H:%M:%S)] evaluation"
[ -s results/transfer/eval_crypto_test.parquet ] || \
    python experiments/transfer.py eval --episodes crypto_test_v0 --tags zero cbseen --out crypto_test --unseal \
        --reason "crypto test: the single pre-registered evaluation (prereg-v1)" --workers 16 > logs/eval_crypto_test.log 2>&1
echo "[$(date +%H:%M:%S)] confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_v1.json --family crypto --eval crypto_test \
    --macro-prefix CryptoTest > logs/confirm_crypto_test.log 2>&1
echo "[$(date +%H:%M:%S)] exploratory"
python experiments/analyze_crypto.py --eval crypto_test --macro-prefix CryptoTest > logs/analysis_crypto_test.log 2>&1
python paper/fig_crypto.py --eval crypto_test --out fig_crypto >> logs/analysis_crypto_test.log 2>&1
echo "[$(date +%H:%M:%S)] done"

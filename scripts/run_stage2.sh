#!/usr/bin/env bash
# Stage 2 (2025-26, six years after training): the single evaluation of the sealed table with the same frozen
# policies, the pre-registered confirmatory family again, and the exploratory analyses.
#   setsid nohup scripts/run_stage2.sh >> logs/stage2.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/prereg-v1 > /dev/null || { echo "no prereg-v1 tag: refusing"; exit 1; }
grep -q '"version": "v1"' prereg/confirmatory_v1.json || { echo "spec is still a draft: refusing"; exit 1; }
python scripts/freeze_models.py --check
echo "[$(date +%H:%M:%S)] evaluation"
[ -s results/transfer/eval_stage2.parquet ] || \
    python experiments/transfer.py eval --episodes stage2_v0 --tags zero seent --out stage2 --unseal \
        --reason "Stage 2: the single pre-registered evaluation (prereg-v1)" --workers 32 > logs/eval_stage2.log 2>&1
echo "[$(date +%H:%M:%S)] confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_v1.json --family stocks --eval stage2 --seen-tag seent \
    --seen-data testseen_v0 --macro-prefix StageTwo > logs/confirm_stage2.log 2>&1
echo "[$(date +%H:%M:%S)] exploratory"
python experiments/analyze_transfer.py --eval stage2 --seen-tag seent --macro-prefix StageTwo > logs/analysis_stage2.log 2>&1
python paper/fig_transfer.py --eval stage2 --out fig_transfer_stage2 >> logs/analysis_stage2.log 2>&1
echo "[$(date +%H:%M:%S)] done"

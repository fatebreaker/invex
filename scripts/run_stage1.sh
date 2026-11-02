#!/usr/bin/env bash
# Stage 1: the single evaluation of the sealed table, then the pre-registered confirmatory analysis and the
# exploratory ones. Refuses to run without the prereg-v1 tag and unchanged models.
#   setsid nohup scripts/run_stage1.sh >> logs/stage1.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/prereg-v1 > /dev/null || { echo "no prereg-v1 tag: refusing"; exit 1; }
grep -q '"version": "v1"' prereg/confirmatory_v1.json || { echo "spec is still a draft: refusing"; exit 1; }
python scripts/freeze_models.py --check
TAGS="zero seent xactivity xrefactivity xtradesize xreftradesize xspread xrefspread"
echo "[$(date +%H:%M:%S)] evaluation"
[ -s results/transfer/eval_stage1.parquet ] || \
    python experiments/transfer.py eval --episodes stage1_v0 --tags $TAGS --out stage1 --unseal \
        --reason "Stage 1: the single pre-registered evaluation (prereg-v1)" --workers 32 > logs/eval_stage1.log 2>&1
echo "[$(date +%H:%M:%S)] confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_v1.json --family stocks --eval stage1 --seen-tag seent \
    --seen-data testseen_v0 --macro-prefix StageOne > logs/confirm_stage1.log 2>&1
echo "[$(date +%H:%M:%S)] exploratory"
python experiments/analyze_transfer.py --eval stage1 --seen-tag seent --macro-prefix StageOne > logs/analysis_stage1.log 2>&1
python paper/fig_transfer.py --eval stage1 --out fig_transfer_stage1 >> logs/analysis_stage1.log 2>&1
for st in activity tradesize spread; do
    python experiments/extrapolation.py analyze --stat $st --eval stage1 --macro-prefix StageOneXtrap \
        >> logs/analysis_stage1.log 2>&1
done
echo "[$(date +%H:%M:%S)] done"

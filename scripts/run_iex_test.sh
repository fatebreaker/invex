#!/usr/bin/env bash
# The registered IEX prospective test (prereg/iex_prospective_v1.md): download, build and validate the ten test
# days, freeze the sealed episode table, check the frozen policies, evaluate them once and run the registered
# family and the pre-specified secondary analyses. Refuses to run without the iex-v1 tag, or if a test day began
# before the tag.
#   setsid nohup scripts/run_iex_test.sh >> logs/iex_test.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v1 > /dev/null || { echo "no iex-v1 tag: refusing"; exit 1; }
LISTED=$(python -c "import json; print(' '.join(json.load(open('prereg/iex_test_days_v1.json'))['days']))")
RESERVE=$(python -c "import json; print(' '.join(json.load(open('prereg/iex_test_days_v1.json'))['reserve']))")
python scripts/iex_days.py --check  # every listed and reserve day opens after the tag
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "build"
scripts/build_iex.sh $LISTED
# a listed day that is missing or fails the book checks is replaced by the first reserve day that passes
DAYS=$(python scripts/iex_days.py --resolve 2> /dev/null) || { scripts/build_iex.sh $RESERVE; \
    DAYS=$(python scripts/iex_days.py --resolve); }
python scripts/iex_days.py --check --qa
log "freeze episodes"
# the registered population (competitive IEX book over the trailing window), and every episode for analysis 4a
[ -s prereg/episodes_iex_test_v0.json ] || \
    python -m invex.eval.freeze --name iex_test_v0 --venue iex --split test --days $DAYS --per-book 200 \
        --min-onetick 0.5 --sealed --workers 16
[ -s prereg/episodes_iex_test_all_v0.json ] || \
    python -m invex.eval.freeze --name iex_test_all_v0 --venue iex --split test --days $DAYS --per-book 100 --sealed \
        --workers 16
python scripts/freeze_models.py --check
python scripts/freeze_models.py --check --manifest prereg/models_iex_v1.json
log "evaluation"
[ -s results/transfer/eval_iex_test.parquet ] || \
    python experiments/transfer.py eval --episodes iex_test_v0 --models 'zero_*_s[0-2]' 'seent_*_s[0-2]' \
        'iexseen_R3tick_s[0-2]' 'zeronn_*_s[0-4]' --out iex_test --unseal \
        --reason "IEX prospective test: the single registered evaluation (iex-v1)" --workers 20 \
        > logs/eval_iex_test.log 2>&1
[ -s results/transfer/eval_iex_test_all.parquet ] || \
    python experiments/transfer.py eval --episodes iex_test_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' \
        'iexseen_R3tick_s[0-2]' --out iex_test_all --unseal \
        --reason "IEX prospective test: the single evaluation of the unrestricted table (iex-v1, analysis 4a)" \
        --workers 20 > logs/eval_iex_test_all.log 2>&1
log "confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_iex_v1.json --family iex --eval iex_test \
    --macro-prefix IexTest > logs/confirm_iex_test.log 2>&1
log "secondary"
python experiments/analyze_iex.py --eval iex_test --eval-all iex_test_all --macro-prefix IexTest \
    > logs/analysis_iex_test.log 2>&1 || true
log "done"

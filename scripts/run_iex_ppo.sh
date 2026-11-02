#!/usr/bin/env bash
# The registered PPO addition to the IEX prospective test (prereg/iex_ppo_v2.md): after scripts/run_iex_test.sh has
# run, evaluate the frozen PPO policies once on the same sealed population table and run the registered families.
# Refuses to run without the iex-v2 tag or before the iex-v1 evaluation exists.
#   setsid nohup scripts/run_iex_ppo.sh >> logs/iex_ppo.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v2 > /dev/null || { echo "no iex-v2 tag: refusing"; exit 1; }
[ -s results/transfer/eval_iex_test.parquet ] || { echo "the iex-v1 evaluation has not run: refusing"; exit 1; }
python scripts/iex_days.py --check --tag iex-v2
python scripts/freeze_models.py --check --manifest prereg/models_iex_v2.json
python scripts/freeze_models.py --check --manifest prereg/models_iex_v1.json
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "evaluation"
[ -s results/transfer/eval_iex_test_ppo.parquet ] || \
    python experiments/transfer.py eval --episodes iex_test_v0 --models 'zeroppo_*_s[0-2]' 'iexseen_R3tick_s[0-2]' \
        'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' --out iex_test_ppo --unseal \
        --reason "IEX prospective test: the single registered evaluation of the PPO policies (iex-v2)" --workers 20 \
        > logs/eval_iex_test_ppo.log 2>&1
log "confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_iex_v2.json --family iex --eval iex_test_ppo \
    --macro-prefix IexPpo > logs/confirm_iex_test_ppo.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_v2.json --family iexlearner --eval iex_test_ppo \
    --macro-prefix IexPpoLearner > logs/confirm_iex_test_ppo_learner.log 2>&1
log "done"

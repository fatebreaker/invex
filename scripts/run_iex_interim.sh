#!/usr/bin/env bash
# iex-v4 part B (prereg/iex_retro_interim_v4.md): a registered interim look at the prospective test. The families of
# iex-v1, v2 and v3 on the first five listed test days that pass the book checks, at a Haybittle-Peto level (alpha
# 0.001, prereg/confirmatory_iex_interim_v4_*.json); the final ten-day analysis (scripts/run_iex_test.sh) is unchanged.
#   started by scripts/wait_iex_interim.sh
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v4 > /dev/null || { echo "no iex-v4 tag: refusing"; exit 1; }
python scripts/iex_days.py --check  # every listed and reserve day opens after the iex-v1 tag
log() { echo "[$(date +%H:%M:%S)] $*"; }
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
DAYS=$(python scripts/iex_days.py --interim)
log "interim days: $DAYS"
log "freeze episodes"
[ -s prereg/episodes_iex_interim_v0.json ] || \
    python -m invex.eval.freeze --name iex_interim_v0 --venue iex --split test --days $DAYS --per-book 200 \
        --min-onetick 0.5 --sealed --workers 16
[ -s prereg/episodes_iex_interim_all_v0.json ] || \
    python -m invex.eval.freeze --name iex_interim_all_v0 --venue iex --split test --days $DAYS --per-book 100 --sealed \
        --workers 16
python scripts/freeze_models.py --check
python scripts/freeze_models.py --check --manifest prereg/models_iex_v1.json
python scripts/freeze_models.py --check --manifest prereg/models_iex_v2.json
log "evaluation (iex-v1 families)"
[ -s results/transfer/eval_iex_interim.parquet ] || \
    python experiments/transfer.py eval --episodes iex_interim_v0 --models 'zero_*_s[0-2]' 'seent_*_s[0-2]' \
        'iexseen_R3tick_s[0-2]' 'zeronn_*_s[0-4]' --out iex_interim --unseal \
        --reason "IEX prospective test, registered interim look: the single registered evaluation (iex-v4 part B, families of iex-v1)" \
        --workers 20 > logs/eval_iex_interim.log 2>&1
[ -s results/transfer/eval_iex_interim_all.parquet ] || \
    python experiments/transfer.py eval --episodes iex_interim_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' \
        'iexseen_R3tick_s[0-2]' --out iex_interim_all --unseal \
        --reason "IEX prospective test, registered interim look: the single evaluation of the unrestricted table (iex-v4 part B, 4a)" \
        --workers 20 > logs/eval_iex_interim_all.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v1.json --family iex --eval iex_interim \
    --macro-prefix IexInterim > logs/confirm_iex_interim.log 2>&1
python experiments/analyze_iex.py --eval iex_interim --eval-all iex_interim_all --macro-prefix IexInterim \
    > logs/analysis_iex_interim.log 2>&1 || true
log "evaluation (iex-v2 families)"
[ -s results/transfer/eval_iex_interim_ppo.parquet ] || \
    python experiments/transfer.py eval --episodes iex_interim_v0 --models 'zeroppo_*_s[0-2]' 'iexseen_R3tick_s[0-2]' \
        'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' --out iex_interim_ppo --unseal \
        --reason "IEX prospective test, registered interim look: the single registered evaluation of the PPO policies (iex-v4 part B)" \
        --workers 20 > logs/eval_iex_interim_ppo.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v2.json --family iex --eval iex_interim_ppo \
    --macro-prefix IexInterimPpo > logs/confirm_iex_interim_ppo.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v2.json --family iexlearner --eval iex_interim_ppo \
    --macro-prefix IexInterimPpoLearner > logs/confirm_iex_interim_ppo_learner.log 2>&1
log "evaluation (iex-v3 families)"
[ -s results/transfer/eval_iex_interim_wide.parquet ] || \
    $ONE python experiments/transfer.py eval --episodes iex_interim_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R3_s[0-2]' \
        'zero_R0_s[0-2]' 'zero_R0pop_s[0-2]' 'zero_R1_s[0-2]' 'zeronn_R3tick_s[0-4]' 'zeronn_R0_s[0-4]' \
        'zeronn_R0pop_s[0-4]' 'zeroppo_R3tick_s[0-2]' 'zeroppo_R3_s[0-2]' 'zeroppo_R0_s[0-2]' 'zeroppo_R1_s[0-2]' \
        --out iex_interim_wide --unseal \
        --reason "IEX prospective test, registered interim look: the single registered evaluation of the wide-book families (iex-v4 part B)" \
        --workers 20 > logs/eval_iex_interim_wide.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v3.json --family iexwide --eval iex_interim_wide \
    --macro-prefix IexInterimWide > logs/confirm_iex_interim_wide.log 2>&1
for f in iexwide_cap iexwide_bps iexwide_more; do
    python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v3.json --family $f --eval iex_interim_wide \
        --macro-prefix "IexInterimWide${f#iexwide_}" >> logs/confirm_iex_interim_wide.log 2>&1 || true
done
python experiments/confirm.py --spec prereg/confirmatory_iex_interim_v4_v3.json --family iexpop_popart --eval iex_interim \
    --macro-prefix IexInterimPopArt >> logs/confirm_iex_interim_wide.log 2>&1 || true
log "done"

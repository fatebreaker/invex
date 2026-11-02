#!/usr/bin/env bash
# iex-v4 part A (prereg/iex_retro_interim_v4.md): the registered IEX families of iex-v1, v2 and v3, unchanged, on ten
# earlier IEX days registered before they were downloaded. Checks once, before any download, that none of the days is
# on disk; then builds them, freezes the sealed tables, checks the frozen policies and evaluates each family once.
#   setsid nohup scripts/run_iex_retro.sh >> logs/iex_retro.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v4 > /dev/null || { echo "no iex-v4 tag: refusing"; exit 1; }
SPEC=prereg/iex_retro_days_v4.json
log() { echo "[$(date +%H:%M:%S)] $*"; }
ONE="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
if [ ! -s logs/iex_retro_absent.txt ]; then  # the first start only: nothing downloaded before the tag's run
    python scripts/iex_days.py --spec $SPEC --absent | tee logs/iex_retro_absent.txt
    date -u +"checked %F %T UTC" >> logs/iex_retro_absent.txt
fi
LISTED=$(python -c "import json; print(' '.join(json.load(open('$SPEC'))['days']))")
RESERVE=$(python -c "import json; print(' '.join(json.load(open('$SPEC'))['reserve']))")
log "build"
scripts/build_iex.sh $LISTED
DAYS=$(python scripts/iex_days.py --spec $SPEC --resolve 2> /dev/null) || { scripts/build_iex.sh $RESERVE; \
    DAYS=$(python scripts/iex_days.py --spec $SPEC --resolve); }
log "days in use: $DAYS"
log "freeze episodes"
[ -s prereg/episodes_iex_retro_v0.json ] || \
    python -m invex.eval.freeze --name iex_retro_v0 --venue iex --split test --days $DAYS --per-book 200 \
        --min-onetick 0.5 --sealed --workers 16
[ -s prereg/episodes_iex_retro_all_v0.json ] || \
    python -m invex.eval.freeze --name iex_retro_all_v0 --venue iex --split test --days $DAYS --per-book 100 --sealed \
        --workers 16
python scripts/freeze_models.py --check
python scripts/freeze_models.py --check --manifest prereg/models_iex_v1.json
python scripts/freeze_models.py --check --manifest prereg/models_iex_v2.json
log "evaluation (iex-v1 families)"
[ -s results/transfer/eval_iex_retro.parquet ] || \
    python experiments/transfer.py eval --episodes iex_retro_v0 --models 'zero_*_s[0-2]' 'seent_*_s[0-2]' \
        'iexseen_R3tick_s[0-2]' 'zeronn_*_s[0-4]' --out iex_retro --unseal \
        --reason "IEX retrospective test: the single registered evaluation (iex-v4 part A, families of iex-v1)" \
        --workers 20 > logs/eval_iex_retro.log 2>&1
[ -s results/transfer/eval_iex_retro_all.parquet ] || \
    python experiments/transfer.py eval --episodes iex_retro_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' \
        'iexseen_R3tick_s[0-2]' --out iex_retro_all --unseal \
        --reason "IEX retrospective test: the single evaluation of the unrestricted table (iex-v4 part A, 4a)" \
        --workers 20 > logs/eval_iex_retro_all.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_v1.json --family iex --eval iex_retro \
    --macro-prefix IexRetro > logs/confirm_iex_retro.log 2>&1
python experiments/analyze_iex.py --eval iex_retro --eval-all iex_retro_all --macro-prefix IexRetro \
    > logs/analysis_iex_retro.log 2>&1 || true
log "evaluation (iex-v2 families)"
[ -s results/transfer/eval_iex_retro_ppo.parquet ] || \
    python experiments/transfer.py eval --episodes iex_retro_v0 --models 'zeroppo_*_s[0-2]' 'iexseen_R3tick_s[0-2]' \
        'zero_R3tick_s[0-2]' 'zero_R0_s[0-2]' --out iex_retro_ppo --unseal \
        --reason "IEX retrospective test: the single registered evaluation of the PPO policies (iex-v4 part A)" \
        --workers 20 > logs/eval_iex_retro_ppo.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_v2.json --family iex --eval iex_retro_ppo \
    --macro-prefix IexRetroPpo > logs/confirm_iex_retro_ppo.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_v2.json --family iexlearner --eval iex_retro_ppo \
    --macro-prefix IexRetroPpoLearner > logs/confirm_iex_retro_ppo_learner.log 2>&1
log "evaluation (iex-v3 families)"
[ -s results/transfer/eval_iex_retro_wide.parquet ] || \
    $ONE python experiments/transfer.py eval --episodes iex_retro_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R3_s[0-2]' \
        'zero_R0_s[0-2]' 'zero_R0pop_s[0-2]' 'zero_R1_s[0-2]' 'zeronn_R3tick_s[0-4]' 'zeronn_R0_s[0-4]' \
        'zeronn_R0pop_s[0-4]' 'zeroppo_R3tick_s[0-2]' 'zeroppo_R3_s[0-2]' 'zeroppo_R0_s[0-2]' 'zeroppo_R1_s[0-2]' \
        --out iex_retro_wide --unseal \
        --reason "IEX retrospective test: the single registered evaluation of the wide-book families (iex-v4 part A)" \
        --workers 20 > logs/eval_iex_retro_wide.log 2>&1
python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family iexwide --eval iex_retro_wide \
    --macro-prefix IexRetroWide > logs/confirm_iex_retro_wide.log 2>&1
for f in iexwide_cap iexwide_bps iexwide_more; do
    python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family $f --eval iex_retro_wide \
        --macro-prefix "IexRetroWide${f#iexwide_}" >> logs/confirm_iex_retro_wide.log 2>&1 || true
done
python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family iexpop_popart --eval iex_retro \
    --macro-prefix IexRetroPopArt >> logs/confirm_iex_retro_wide.log 2>&1 || true
log "done"

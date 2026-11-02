#!/usr/bin/env bash
# The registered wide-book addition to the IEX prospective test (prereg/iex_wide_v3.md): after the iex-v1 and iex-v2
# evaluations, evaluate the frozen policies once on the sealed table of every episode and run the registered family
# and the pre-specified secondary families. Refuses to run without the iex-v3 tag or before the iex-v1 table exists.
#   setsid nohup scripts/run_iex_wide.sh >> logs/iex_wide.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v3 > /dev/null || { echo "no iex-v3 tag: refusing"; exit 1; }
[ -s prereg/episodes_iex_test_all_v0.json ] || { echo "iex_test_all_v0 has not been frozen: refusing"; exit 1; }
[ -s results/transfer/eval_iex_test.parquet ] || { echo "the iex-v1 evaluation has not run: refusing"; exit 1; }
python scripts/iex_days.py --check --tag iex-v3
python scripts/freeze_models.py --check --manifest prereg/models_iex_v1.json
python scripts/freeze_models.py --check --manifest prereg/models_iex_v2.json
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "evaluation"
[ -s results/transfer/eval_iex_test_wide.parquet ] || \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python experiments/transfer.py eval --episodes iex_test_all_v0 --models 'zero_R3tick_s[0-2]' 'zero_R3_s[0-2]' \
        'zero_R0_s[0-2]' 'zero_R0pop_s[0-2]' 'zero_R1_s[0-2]' 'zeronn_R3tick_s[0-4]' 'zeronn_R0_s[0-4]' \
        'zeronn_R0pop_s[0-4]' 'zeroppo_R3tick_s[0-2]' 'zeroppo_R3_s[0-2]' 'zeroppo_R0_s[0-2]' 'zeroppo_R1_s[0-2]' \
        --out iex_test_wide --unseal \
        --reason "IEX prospective test: the single registered evaluation of the wide-book families (iex-v3)" \
        --workers 20 > logs/eval_iex_test_wide.log 2>&1
log "confirmatory"
python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family iexwide --eval iex_test_wide \
    --macro-prefix IexWide > logs/confirm_iex_test_wide.log 2>&1
log "secondary"
for f in iexwide_cap iexwide_bps iexwide_more; do
    python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family $f --eval iex_test_wide \
        --macro-prefix "IexWide${f#iexwide_}" >> logs/confirm_iex_test_wide.log 2>&1 || true
done
python experiments/confirm.py --spec prereg/confirmatory_iex_v3.json --family iexpop_popart --eval iex_test \
    --macro-prefix IexPopArt >> logs/confirm_iex_test_wide.log 2>&1 || true
log "done"

#!/usr/bin/env bash
# iex-v5 (prereg/iex_capped_v5.md): the capped secondary families on the stored outputs of the registered IEX
# evaluations. No policy is evaluated. Usage: run_iex_v5.sh <eval> <ppo eval> <spec> <macro prefix>, e.g.
#   scripts/run_iex_v5.sh iex_test iex_test_ppo prereg/confirmatory_iex_v5.json IexTestCap
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v5 > /dev/null || { echo "no iex-v5 tag: refusing"; exit 1; }
ev=$1; evp=$2; spec=$3; pre=$4
log() { echo "[$(date +%H:%M:%S)] $*"; }
python experiments/confirm.py --spec $spec --family iex_cap --eval $ev --macro-prefix ${pre} \
    > logs/confirm_${ev}_v5.log 2>&1 || log "iex_cap on $ev FAILED"
python experiments/confirm.py --spec $spec --family iexppo_cap --eval $evp --macro-prefix ${pre}Ppo \
    > logs/confirm_${evp}_v5.log 2>&1 || log "iexppo_cap on $evp FAILED"
python experiments/confirm.py --spec $spec --family iexlearner_cap --eval $evp --macro-prefix ${pre}PpoLearner \
    >> logs/confirm_${evp}_v5.log 2>&1 || log "iexlearner_cap on $evp FAILED"
log "v5 done: $ev $evp"

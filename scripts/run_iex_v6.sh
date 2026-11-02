#!/usr/bin/env bash
# iex-v6 (prereg/iex_robust_v6.md): the secondary families of iex-v6 on the stored outputs of the registered IEX
# evaluations (no policy is evaluated). Usage: run_iex_v6.sh <eval> <ppo eval> <spec> <macro prefix>, e.g.
#   scripts/run_iex_v6.sh iex_test iex_test_ppo prereg/confirmatory_iex_v6.json IexTest
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v6 > /dev/null || { echo "no iex-v6 tag: refusing"; exit 1; }
ev=$1; evp=$2; spec=$3; pre=$4
log() { echo "[$(date +%H:%M:%S)] $*"; }
python experiments/confirm.py --spec $spec --family iex_cap133 --eval $ev --macro-prefix ${pre}CapRule \
    > logs/confirm_${ev}_v6.log 2>&1 || log "iex_cap133 on $ev FAILED"
python experiments/confirm.py --spec $spec --family iexppo_cap133 --eval $evp --macro-prefix ${pre}PpoCapRule \
    >> logs/confirm_${ev}_v6.log 2>&1 || log "iexppo_cap133 on $evp FAILED"
for est in episode median; do
    E=$(echo $est | sed 's/^./\U&/')
    python experiments/confirm_robust.py --spec $spec --family iex_$est --eval $ev --macro-prefix ${pre}$E \
        >> logs/confirm_${ev}_v6.log 2>&1 || log "iex_$est on $ev FAILED"
    python experiments/confirm_robust.py --spec $spec --family iexppo_$est --eval $evp --macro-prefix ${pre}Ppo$E \
        >> logs/confirm_${ev}_v6.log 2>&1 || log "iexppo_$est on $evp FAILED"
done
log "v6 done: $ev $evp"

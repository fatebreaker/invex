#!/usr/bin/env bash
# A crypto-trained reference from the evaluation period itself, cross-fitted by day (third review, question 2): the
# 34 days of the crypto test and its replication (2024-26) in two folds of alternate dates; an MS+tick policy fitted
# on each fold (size-matched to the 2020-23 reference cb_train_v0), and each day scored with the fold that did not
# train on it. Exploratory and post hoc; the sealed crypto tables are read once more, logged.
#   scripts/queue_crypto_crossfit.sh >> logs/queue_crypto_crossfit.log 2>&1
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
REASON="exploratory, post hoc (third review): crypto reference trained on 2024-26, cross-fitted by day"
DAYS=$(python - <<'EOF'
import json
d = sorted(json.load(open("prereg/episodes_crypto_test_v0.json"))["days"] + json.load(open("prereg/episodes_crypto_rep_v0.json"))["days"])
print(" ".join(d[0::2]) + "|" + " ".join(d[1::2]))
EOF
)
FOLD_A=${DAYS%|*}
FOLD_B=${DAYS#*|}
log "fold a: $FOLD_A"
log "fold b: $FOLD_B"
[ -s data/transitions/cbx_a_v0/data.npz ] || python experiments/transfer.py gen --name cbx_a_v0 --venue coinbase \
    --split all --days $FOLD_A --per-book 1000 --workers 8 > logs/gen_cbx_a.log 2>&1
[ -s data/transitions/cbx_b_v0/data.npz ] || python experiments/transfer.py gen --name cbx_b_v0 --venue coinbase \
    --split all --days $FOLD_B --per-book 1000 --workers 8 > logs/gen_cbx_b.log 2>&1
log "transitions done"
while pgrep -f "^python experiments/transfer.py fit .*--tag sboot" > /dev/null; do sleep 30; done  # thread budget
python experiments/transfer.py fit --data cbx_a_v0 --size-of cb_train_v0 --tag cbxa --reps R3tick --seeds 0 1 2 \
    --threads 12 > logs/fit_cbxa.log 2>&1 &
python experiments/transfer.py fit --data cbx_b_v0 --size-of cb_train_v0 --tag cbxb --reps R3tick --seeds 0 1 2 \
    --threads 12 > logs/fit_cbxb.log 2>&1 &
wait
log "fits done"
for ev in crypto_test crypto_rep; do
    python experiments/transfer.py eval --episodes ${ev}_v0 --unseal --reason "$REASON" --models "cbx[ab]_R3tick_s*" \
        --out ${ev}_cbx > logs/eval_${ev}_cbx.log 2>&1
    log "$ev scored"
done

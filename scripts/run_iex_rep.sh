#!/usr/bin/env bash
# The second IEX prospective test (iex-rep-v1, prereg/iex_replication_v1.md): build and check the ten test days,
# freeze the two sealed episode tables, check the frozen policies and the pipeline code, evaluate every policy once
# per table, and run the registered and the secondary families. Refuses to run without the iex-rep-v1 tag, if a test
# day opened before the tag, if a policy changed, or if a pipeline file differs from the tag. Started once by
# scripts/wait_iex_rep.sh.
#   scripts/run_iex_rep.sh          # the registered run
#   scripts/run_iex_rep.sh --dry    # the same steps on the IEX validation days (iex_val_v0, not sealed), outputs
#                                   # named iexrepdry*, to test the script before the tag
set -euf  # -f: the policy patterns below reach transfer.py unexpanded
cd "$(dirname "$0")/.."
source scripts/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date '+%F %T')] $*"; }
TAG=iex-rep-v1
SPEC=prereg/confirmatory_iex_rep_v1.json
DAYS_SPEC=prereg/iex_rep_days_v1.json
PIPELINE="src/invex experiments/transfer.py experiments/confirm.py experiments/confirm_robust.py experiments/confirm_rep.py
experiments/ctx_fqi.py experiments/iex_rep_secondary.py experiments/dist_compare.py experiments/ppo.py scripts/run_iex_rep.sh
scripts/wait_iex_rep.sh scripts/build_iex.sh scripts/iex_days.py scripts/freeze_models_rep.py scripts/merge_evals.py
scripts/env.sh $SPEC $DAYS_SPEC prereg/models_iex_rep_v1.json prereg/iex_replication_v1.md"
MODELS1="zero_R3tick_s[0-2] zero_R0_s[0-2] zero_R0info_s[0-2] zero_R0pop_s[0-2] iexseen_R3tick_s[0-2]
zeronn_R3tick_s[0-4] zeronn_R0_s[0-4] zeronn_R0info_s[0-4] zeronn_R0pop_s[0-4]
zeroppo_R3tick_s[0-2] zeroppo_R0_s[0-2] zeroppo_R0info_s[0-2] zeroppo_R0pop_s[0-2] ixppo_R3tick_s[0-2]"
MODELS2="zero_R3tick_s[0-2] zero_R0_s[0-2] zero_R0info_s[0-2] zero_R0pop_s[0-2] zero_R1_s[0-2]
zeronn_R3tick_s[0-4] zeronn_R0_s[0-4] zeronn_R0r_s[0-4] zeronn_R0info_s[0-4] zeronn_R0pop_s[0-4]
zeroppo_R3tick_s[0-2] zeroppo_R0_s[0-2] zeroppo_R0info_s[0-2] tdrnn_R0r_s[0-2]
irmnn1_R0r_s[0-2] irmnn1_R0info_s[0-2] vrexnn1_R0r_s[0-2] vrexnn1_R0info_s[0-2] gdronn10_R0r_s[0-2] gdronn10_R0info_s[0-2]"

if [ "${1:-}" = "--dry" ]; then
    POP=iex_val_v0; ALL=iex_val_v0; O=iexrepdry; UNSEAL=(); P=IexRepDry
else
    git rev-parse -q --verify refs/tags/$TAG > /dev/null || { log "no $TAG tag: refusing"; exit 1; }
    python scripts/iex_days.py --spec $DAYS_SPEC --check --tag $TAG
    git diff --quiet $TAG -- $PIPELINE || { log "a pipeline file differs from $TAG: refusing"; git diff --stat $TAG -- $PIPELINE; exit 1; }
    extra=$(git ls-files --others --exclude-standard -- src/invex)
    [ -z "$extra" ] || { log "untracked files in src/invex: $extra; refusing"; exit 1; }
    log "pipeline: every file identical to $TAG ($(git rev-parse --short $TAG^{commit})); HEAD $(git rev-parse --short HEAD)"
    log "build"
    LISTED=$(python -c "import json; print(' '.join(json.load(open('$DAYS_SPEC'))['days']))")
    RESERVE=$(python -c "import json; print(' '.join(json.load(open('$DAYS_SPEC'))['reserve']))")
    scripts/build_iex.sh $LISTED
    # a listed day that is missing or fails the book checks is replaced by the first reserve day that passes
    DAYS=$(python scripts/iex_days.py --spec $DAYS_SPEC --resolve 2> /dev/null) || { scripts/build_iex.sh $RESERVE; \
        DAYS=$(python scripts/iex_days.py --spec $DAYS_SPEC --resolve); }
    python scripts/iex_days.py --spec $DAYS_SPEC --check --qa --tag $TAG
    log "freeze episodes: $DAYS"
    [ -s prereg/episodes_iex_rep_v0.json ] || \
        python -m invex.eval.freeze --name iex_rep_v0 --venue iex --split test --days $DAYS --per-book 200 \
            --min-onetick 0.5 --sealed --workers 16
    [ -s prereg/episodes_iex_rep_all_v0.json ] || \
        python -m invex.eval.freeze --name iex_rep_all_v0 --venue iex --split test --days $DAYS --per-book 100 --sealed \
            --workers 16
    POP=iex_rep_v0; ALL=iex_rep_all_v0; O=iex_rep; UNSEAL=(--unseal); P=IexRep
fi
python scripts/freeze_models_rep.py --check || { log "a policy changed: refusing"; exit 1; }
R() { [ ${#UNSEAL[@]} -eq 0 ] && echo "" || echo "$1"; }

log "evaluation: competitive books ($POP)"
[ -s results/transfer/eval_${O}.parquet ] || \
    python experiments/transfer.py eval --episodes $POP --models $MODELS1 --out $O ${UNSEAL[@]+"${UNSEAL[@]}"} \
        --reason "$(R "IEX replication: the single evaluation of the competitive-book table ($TAG)")" --workers 20 \
        > logs/eval_${O}.log 2>&1
log "evaluation: every episode ($ALL)"
[ -s results/transfer/eval_${O}_all.parquet ] || \
    python experiments/transfer.py eval --episodes $ALL --models $MODELS2 --out ${O}_all ${UNSEAL[@]+"${UNSEAL[@]}"} \
        --reason "$(R "IEX replication: the single evaluation of the table of every episode ($TAG)")" --workers 20 \
        > logs/eval_${O}_all.log 2>&1
log "evaluation: the context encoder ($ALL)"
[ -s results/transfer/ctx_${ALL}.npy ] || \
    python experiments/ctx_fqi.py build-val --episodes $ALL ${UNSEAL[@]+"${UNSEAL[@]}"} \
        --reason "$(R "IEX replication: the context series of the table of every episode ($TAG)")" --workers 16 \
        > logs/ctx_${O}_all.log 2>&1
[ -s results/transfer/eval_${O}_all_ctx.parquet ] || \
    python experiments/ctx_fqi.py eval --episodes $ALL --out ${O}_all_ctx ${UNSEAL[@]+"${UNSEAL[@]}"} \
        --reason "$(R "IEX replication: the single evaluation of the context encoder ($TAG)")" --workers 16 \
        >> logs/ctx_${O}_all.log 2>&1
python scripts/merge_evals.py --out ${O}_wide ${O}_all ${O}_all_ctx

log "families"
fams=$(python -c "import json; print(' '.join(json.load(open('$SPEC'))['families']))")
for f in $fams; do
    ev=$(python -c "import json; print(json.load(open('$SPEC'))['families']['$f']['eval'])" | sed "s/^iex_rep/$O/")
    pre=$P$(echo ${f#rep_} | sed -E 's/(^|_)([a-z])/\U\2/g')
    python experiments/confirm_rep.py --spec $SPEC --family $f --eval $ev --macro-prefix $pre \
        >> logs/confirm_${O}.log 2>&1 && log "  $f on $ev" || log "  $f on $ev FAILED"
done
log "secondary"
python experiments/iex_rep_secondary.py --pop $O --wide ${O}_wide --prefix ${P}Sec --out ${O}_secondary \
    > logs/secondary_${O}.log 2>&1 && log "  secondary done" || log "  secondary FAILED"
set +f
grep -h "^| [A-Z][0-9]" results/transfer/confirm_${O}*_rep_{fqi,ppo,learner,lm,wide,info,infowide,alt}.md 2> /dev/null || true
log "done"

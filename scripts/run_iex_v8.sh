#!/usr/bin/env bash
# iex-v8 (prereg/iex_amend_v8.md): once every registered final analysis of the prospective IEX test has finished,
# (1) check the pipeline files against the iex-v6 manifest, as scripts/pipeline_guard.sh does, and the policies
#     against their recorded hashes;
# (2) evaluate the zero-shot and the IEX-trained PPO MS+tick policies on the population table (one more logged read);
# (3) run the iex-v8 families: P4 under both estimands, and the episode-weighted L and W families. The
#     episode-weighted I and P families are those of iex-v6, already run by scripts/run_iex_v6.sh.
# Started by scripts/wait_iex_v8.sh; runs once.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
git rev-parse -q --verify refs/tags/iex-v8 > /dev/null || { echo "no iex-v8 tag: refusing"; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }
python scripts/freeze_models_v8.py --check || { log "an iex-v8 policy changed: refusing"; exit 1; }
python scripts/freeze_models.py --manifest prereg/models_iex_v2.json --check || { log "a PPO policy changed: refusing"; exit 1; }
M=$(ls prereg/timestamps/manifest_iex_v6_*.txt | head -n 1)
bad=0; n=0
for f in $(find src/invex experiments/transfer.py experiments/confirm.py experiments/confirm_robust.py \
           experiments/analyze_iex.py experiments/ppo.py scripts/run_iex_test.sh scripts/run_iex_ppo.sh \
           scripts/run_iex_wide.sh scripts/run_iex_interim.sh scripts/run_iex_v5.sh scripts/run_iex_v6.sh \
           scripts/build_iex.sh scripts/iex_days.py scripts/freeze_models.py scripts/env.sh -type f \
           \( -name '*.py' -o -name '*.sh' \) | sort); do
    n=$((n + 1))
    want=$(awk -v f="$f" '$1 == "tag" {t = $2} t == "iex-v6" && $2 == f {print $1}' "$M")
    [ "$want" = "$(sha256sum "$f" | cut -d' ' -f1)" ] || { bad=$((bad + 1)); log "  differs from iex-v6: $f"; }
done
log "iex-v8 run started: $n pipeline files, $bad differ from the iex-v6 manifest; HEAD $(git rev-parse --short HEAD)"
spec=prereg/confirmatory_iex_v8.json
python experiments/transfer.py eval --episodes iex_test_v0 --models 'zeroppo_R3tick_s[0-2]' 'ixppo_R3tick_s[0-2]' \
    --out iex_test_v8 --unseal --reason "IEX prospective test: the learner-matched PPO comparison (iex-v8)" \
    --workers 20 > logs/eval_iex_test_v8.log 2>&1 || { log "evaluation FAILED"; exit 1; }
python experiments/confirm.py --spec $spec --family iexlm --eval iex_test_v8 --macro-prefix IexTestLm \
    > logs/confirm_iex_test_v8.log 2>&1 || log "iexlm FAILED"
python experiments/confirm_robust.py --spec $spec --family iexlm_episode --eval iex_test_v8 \
    --macro-prefix IexTestLmEpisode >> logs/confirm_iex_test_v8.log 2>&1 || log "iexlm_episode FAILED"
python experiments/confirm_robust.py --spec $spec --family iexlearner_episode --eval iex_test_ppo \
    --macro-prefix IexTestLearnerEpisode >> logs/confirm_iex_test_v8.log 2>&1 || log "iexlearner_episode FAILED"
python experiments/confirm_robust.py --spec $spec --family iexwide_episode --eval iex_test_wide \
    --macro-prefix IexTestWideEpisode >> logs/confirm_iex_test_v8.log 2>&1 || log "iexwide_episode FAILED"
log "v8 done"

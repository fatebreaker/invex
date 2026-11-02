#!/usr/bin/env bash
# Stage-2 episode table (sealed): wait for the last ITCH day to validate, compute the 2025-26 day statistics,
# freeze. Freezing draws episodes only; no policy is run on a sealed table here.
#   setsid nohup scripts/queue_stage2.sh >> logs/queue_stage2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
DAYS="2025-11-28 2025-12-08 2025-12-09 2025-12-10 2025-12-11 2025-12-12 2026-05-15 2026-05-18 2026-06-12"
STAGE2="2025-12-08 2025-12-09 2025-12-10 2025-12-11 2025-12-12 2026-05-15 2026-05-18 2026-06-12"
for _ in $(seq 1 720); do [ -s data/itch/2026-06-12/book_qa.json ] && break; sleep 60; done
for d in $DAYS; do
    [ -s data/itch/$d/daystats.parquet ] || python -m invex.eval.universe stats --day $d --workers 16
    echo "[$(date +%H:%M:%S)] stats $d"
done
[ -s prereg/episodes_stage2_v0.json ] || \
    python -m invex.eval.freeze --name stage2_v0 --split test --days $STAGE2 --sealed --workers 16
echo "[$(date +%H:%M:%S)] done"

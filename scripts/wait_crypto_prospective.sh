#!/usr/bin/env bash
# Run scripts/run_crypto_prospective.sh for 2026-11-01 and 2026-12-01, each once, two days after the day (the free
# first-of-month data are complete by then).
#   setsid nohup scripts/wait_crypto_prospective.sh >> logs/crypto_prospective.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
for D in 2026-11-01 2026-12-01; do
    ready=$(date -d "$D + 2 days" +%s)
    while [ "$(date +%s)" -lt "$ready" ]; do sleep 3600; done
    scripts/run_crypto_prospective.sh "$D" || echo "[$(date '+%F %T')] $D FAILED"
done

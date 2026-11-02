#!/usr/bin/env bash
# Complete pending OpenTimestamps proofs once a Bitcoin block includes them (tries hourly, gives up after 3 days).
#   setsid nohup scripts/ots_upgrade.sh prereg/timestamps/*.ots >> logs/ots.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
for i in $(seq 72); do
    left=0
    for f in "$@"; do
        ots upgrade "$f" 2>&1 | grep -q "Timestamp complete" && echo "[$(date '+%F %T')] complete: $f" || left=1
    done
    [ $left -eq 0 ] && exit 0
    sleep 3600
done
echo "[$(date '+%F %T')] still pending after 72 tries"

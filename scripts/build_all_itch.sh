#!/usr/bin/env bash
# Build and validate every ITCH day as soon as its download is complete (resumable: built days are skipped).
#   setsid nohup scripts/build_all_itch.sh >> logs/build_all_itch.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
days=$(python -c "from invex.itch import ITCH_DAYS; print(' '.join(ITCH_DAYS))")
for day in $days; do
    gz="$INVEX_RAW/itch/gz/$day.itch50.gz"
    # wait for the downloader's atomic rename (at most 12 h)
    for _ in $(seq 1 1440); do [ -s "$gz" ] && break; sleep 30; done
    [ -s "$gz" ] || { echo "$day: no download, skipped"; continue; }
    parts=1
    case "$day" in 2025-*|2026-*) parts=2 ;; esac
    if [ ! -s "$INVEX_DATA/itch/$day/qa.json" ]; then
        python -m invex.itch.build --day "$day" --parts "$parts" || { echo "$day: BUILD FAILED"; continue; }
    fi
    if [ ! -s "$INVEX_DATA/itch/$day/book_qa.json" ]; then
        python -m invex.book.validate --venue itch --day "$day" --workers 32 || echo "$day: VALIDATE FAILED"
    fi
done
echo "all days done $(date)"

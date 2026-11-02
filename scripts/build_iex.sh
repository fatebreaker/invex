#!/usr/bin/env bash
# Download, normalise, validate and describe IEX DEEP+ days (resumable: built days are skipped). A day whose
# download or build fails is logged and skipped; the others go on.
#   setsid nohup scripts/build_iex.sh 2026-09-16 2025-11-12 ... >> logs/build_iex.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date +%H:%M:%S)] $*"; }
for d in "$@"; do
    if (
        set -e
        python -m invex.iex.hist --days "$d"
        [ -s "data/iex/$d/qa.json" ] || python -m invex.iex.normalize --days "$d"
        [ -s "data/iex/$d/book_qa.json" ] || python -m invex.book.validate --venue iex --day "$d" --workers 16
        [ -s "data/iex/$d/daystats.parquet" ] || python -m invex.eval.universe stats --venue iex --day "$d" --workers 16
    ); then
        log "$d built"
    else
        log "$d FAILED"
    fi
done
log "done"

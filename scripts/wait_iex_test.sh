#!/usr/bin/env bash
# Build each IEX test day once HIST publishes it (T+1), and run the registered test (scripts/run_iex_test.sh) once
# the listed days are built and the registered rule yields ten usable days. Building a day computes its book checks
# and stock statistics only; no policy is run before run_iex_test.sh.
#   setsid nohup scripts/wait_iex_test.sh >> logs/iex_wait.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[$(date '+%F %T')] $*"; }
LISTED=$(python -c "import json; print(' '.join(json.load(open('prereg/iex_test_days_v1.json'))['days']))")
RESERVE=$(python -c "import json; print(' '.join(json.load(open('prereg/iex_test_days_v1.json'))['reserve']))")
while true; do
    today=$(TZ=America/New_York date +%F)
    for d in $LISTED $RESERVE; do
        [[ "$d" < "$today" ]] || continue
        [ -s "data/iex/$d/book_qa.json" ] && continue
        log "try $d"
        scripts/build_iex.sh "$d"
    done
    pending=0
    for d in $LISTED; do [ -s "data/iex/$d/book_qa.json" ] || pending=1; done
    last=$(echo $RESERVE | awk '{print $NF}')
    # a reserve day may replace a listed day only once every listed day is built (or the reserves have all passed)
    if { [ $pending -eq 0 ] || [[ "$last" < "$today" ]]; } && python scripts/iex_days.py --resolve > /dev/null 2>&1; then
        log "ten usable days: $(python scripts/iex_days.py --resolve)"
        scripts/run_iex_test.sh >> logs/iex_test.log 2>&1 && log "run_iex_test done" || log "run_iex_test FAILED"
        exit 0
    fi
    sleep 1800
done

#!/usr/bin/env bash
# Hold 20 GB on the shared /data disk for this project's remaining registered IEX runs (other users' jobs can fill it).
# The space is held as 2 GB files in .disk_reserve/. One file is released whenever /data has under 10 GB free while a
# build or registered run of ours is going; everything is released once the iex-v8 run (the last IEX run) has ended.
# Not part of the pipeline: it reads no data and touches none of the pipeline's files.
#   setsid nohup scripts/disk_reserve.sh >> logs/disk_reserve.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
R=.disk_reserve
log() { echo "[$(date '+%F %T')] $*"; }
avail() { df -B1G --output=avail /data | tail -1 | tr -d ' '; }
ours() { pgrep -f 'scripts/build_iex.sh|scripts/run_iex_|experiments/transfer.py|experiments/confirm' > /dev/null; }
finished() { grep -Eq "final done|FAILED" logs/iex_v8_wait.log 2> /dev/null; }
mkdir -p $R
if [ -z "$(ls $R)" ]; then
    for i in $(seq -w 1 10); do fallocate -l 2G $R/chunk_$i || { log "fallocate failed at chunk $i"; break; }; done
fi
log "holding $(ls $R | wc -l) chunks of 2 GB; /data free $(avail) GB"
while [ -n "$(ls $R)" ]; do
    if finished; then
        rm -f $R/chunk_*
        log "the iex-v8 run has ended: released everything; /data free $(avail) GB"
        break
    fi
    if [ "$(avail)" -lt 10 ] && ours; then
        f=$(ls $R | head -n 1)
        rm -f "$R/$f"
        sleep 2
        log "released $f: /data free $(avail) GB, $(ls $R | wc -l) chunks left"
        continue
    fi
    sleep 30
done
rmdir $R 2> /dev/null
log "done"

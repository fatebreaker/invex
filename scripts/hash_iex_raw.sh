#!/usr/bin/env bash
# SHA-256 of every downloaded IEX DEEP+ file (the raw pcaps live on /tmp, which is wiped at boot, and IEX serves them
# for twelve months), with IEX's listing (URL generation, size), written to supplement/iex_raw_files.tsv. Read-only.
#   setsid nohup scripts/hash_iex_raw.sh >> logs/hash_iex_raw.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
out=supplement/iex_raw_files.tsv
tmp=$(mktemp)
printf "day\tfile\tbytes\tsha256\tiex_generation\n" > "$tmp"
for d in $(ls /tmp/invex_raw/iex | sort); do
    f=/tmp/invex_raw/iex/$d/DPLS.pcap.gz
    [ -s "$f" ] || continue
    gen=$(python3 -c "import json,re; d=json.load(open('/tmp/invex_raw/iex/$d/listing.json')); print(re.search(r'generation=(\d+)', d['link']).group(1))")
    h=$(nice -n 15 ionice -c3 sha256sum "$f" | cut -d' ' -f1)
    printf "%s\t%s\t%s\t%s\t%s\n" "$d" "$(basename "$f")" "$(stat -c %s "$f")" "$h" "$gen" >> "$tmp"
    echo "[$(date '+%F %T')] $d $h"
done
mv "$tmp" "$out"
echo "[$(date '+%F %T')] done: $(($(wc -l < "$out") - 1)) files"

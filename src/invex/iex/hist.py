"""Download IEX DEEP+ days from HIST, IEX's free historical data (T+1, the last twelve months).

    python -m invex.iex.hist --days 2026-10-14 2026-09-16

The listing `https://iextrading.com/api/1.0/hist?date=YYYYMMDD` names one gzipped pcap-ng capture per feed and
day; DEEP+ is the feed 'DPLS'. Files go to $INVEX_RAW/iex/<day>/DPLS.pcap.gz, written to a .part file, resumed if
interrupted and checked against the listed size before the rename (one curl per file: two writers to one .part
would corrupt it). Terms: "Data provided for free by IEX. By accessing or using IEX Historical Data, you agree to
the IEX Historical Data Terms of Use" (https://iextrading.com/iex-historical-data-terms/).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from ..paths import RAW

LISTING = "https://iextrading.com/api/1.0/hist?date={}"
FEED = "DPLS"


def local_path(day: str) -> Path:
    return RAW / "iex" / day / f"{FEED}.pcap.gz"


def listing(day: str) -> dict:
    """The DEEP+ entry of the day's HIST listing ({} on a day without one, e.g. a holiday)."""
    with urllib.request.urlopen(LISTING.format(day.replace("-", "")), timeout=60) as r:
        rows = json.loads(r.read() or b"[]")
    hit = [x for x in rows if x.get("feed") == FEED]
    return hit[0] if hit else {}


def download(day: str, retries: int = 5) -> Path | None:
    out = local_path(day)
    entry = listing(day)
    if not entry:
        print(f"{day}: no {FEED} file listed", flush=True)
        return None
    size = int(entry["size"])
    if out.exists() and out.stat().st_size == size:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = open(out.parent / ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)  # one writer per day: wait for another process's download to end
    if out.exists() and out.stat().st_size == size:
        return out
    (out.parent / "listing.json").write_text(json.dumps(entry, indent=1))
    part = out.with_name(out.name + ".part")
    t0 = time.time()
    for attempt in range(retries):
        subprocess.run(["curl", "-s", "-L", "--retry", "5", "-C", "-", "-o", str(part), entry["link"]], check=False)
        if part.exists() and part.stat().st_size == size:
            part.replace(out)
            print(f"{day}: {size / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
            return out
        if part.exists() and part.stat().st_size > size:
            part.unlink()  # longer than listed: start over
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{day}: download incomplete after {retries} attempts")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True)
    a = ap.parse_args(argv)
    for d in a.days:
        download(d)
    print("download done", flush=True)


if __name__ == "__main__":
    main()

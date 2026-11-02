"""Coinbase order-by-order (L3) data from the Tardis.dev HTTP API, first day of each month (free, no key).

    python -m invex.coinbase.tardis --pairs BTC-USD ETH-USD --days 2020-07-01 2026-09-01 --minutes 120

The API returns one minute per request, so a pair-day is fetched minute by minute. Each minute's response is
stored as received (gzip) in $INVEX_RAW/coinbase/<day>/<pair>/<HHMM>.ndjson.gz, written atomically, so a
restart skips what is already there. Channels: full_snapshot (the book at subscription, with its sequence
number), received, open, done, match and change. `received` adds nothing to the book, but sequence numbers run
across all channels, so it is needed to check the stream for gaps.
Requests are retried with exponential back-off on errors and 429s, and stay well under the rate limit.
Only first-of-month days are free; other days are refused locally. A pair-day whose first minute is empty
(the pair was not listed) is skipped after that one request.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import time
import urllib.parse
from pathlib import Path

import requests

from ..paths import RAW, ensure_space

URL = "https://api.tardis.dev/v1/data-feeds/coinbase"
CHANNELS = ("full_snapshot", "received", "open", "done", "match", "change")


def minute_path(day: str, pair: str, minute: int) -> Path:
    return RAW / "coinbase" / day / pair / f"{minute // 60:02d}{minute % 60:02d}.ndjson.gz"


def _url(day: str, pair: str, minute: int) -> str:
    start = f"{day}T{minute // 60:02d}:{minute % 60:02d}:00"
    filters = json.dumps([{"channel": c, "symbols": [pair]} for c in CHANNELS], separators=(",", ":"))
    return f"{URL}?from={start}&to={start}&filters={urllib.parse.quote(filters)}"


def fetch_minute(day: str, pair: str, minute: int, session: requests.Session, retries: int = 8) -> Path:
    if not day.endswith("-01"):
        raise ValueError(f"{day}: only the first day of a month is free")
    dest = minute_path(day, pair, minute)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    for attempt in range(retries):
        try:
            with session.get(_url(day, pair, minute), headers={"Accept-Encoding": "gzip"}, stream=True,
                             timeout=120) as r:
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.raw.stream(1 << 20, decode_content=False):
                        fh.write(chunk)
                enc = r.headers.get("Content-Encoding", "")
            if enc != "gzip":  # keep the store uniform: gzip it ourselves
                raw = tmp.read_bytes()
                tmp.write_bytes(gzip.compress(raw))
            os.replace(tmp, dest)
            return dest
        except (requests.RequestException, RuntimeError) as e:
            wait = min(120, 2 * 2**attempt)
            print(f"  {pair} {day} minute {minute}: {e!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{pair} {day} minute {minute}: failed after {retries} attempts")


def is_empty(path: Path) -> bool:
    with gzip.open(path, "rb") as fh:
        return fh.read(1) == b""


def fetch_day(day: str, pair: str, minutes: int, workers: int = 6) -> int:
    """Fetch minutes 0..minutes-1; returns how many were fetched (0 if the pair did not trade that day)."""
    ensure_space(RAW / "coinbase", 2.0)
    session = requests.Session()
    if is_empty(fetch_minute(day, pair, 0, session)):  # not listed (yet, or any more)
        return 0
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda m: fetch_minute(day, pair, m, session), range(1, minutes)))
    return minutes


def first_days(start: str = "2020-07", end: str = "2026-10") -> list[str]:
    y, m = map(int, start.split("-"))
    y2, m2 = map(int, end.split("-"))
    out = []
    while (y, m) <= (y2, m2):
        out.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--days", nargs="+", default=["all"], help="YYYY-MM-01 ... or 'all' (2020-07 to 2026-10)")
    ap.add_argument("--minutes", type=int, default=120, help="minutes from 00:00 UTC")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    days = first_days() if a.days == ["all"] else a.days
    for d in days:
        for p in a.pairs:
            t0 = time.time()
            try:
                n = fetch_day(d, p, a.minutes, a.workers)
                print(f"{p} {d}: {n} minutes in {time.time() - t0:.0f}s" + ("" if n else " (not listed)"), flush=True)
            except Exception as e:  # keep going
                print(f"{p} {d}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()

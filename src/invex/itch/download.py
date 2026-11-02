"""Download the public Nasdaq TotalView-ITCH 5.0 sample days (~149 GB gzipped).

    python -m invex.itch.download --days all --conns 4
    python -m invex.itch.download --days 2019-01-30 2019-03-27

Files go to $INVEX_RAW/itch/gz/<YYYY-MM-DD>.itch50.gz. Each file is fetched as
`conns` byte ranges in parallel (the server honours Range requests); a part
that was cut off resumes from where it stopped. The assembled file must match
the server's Content-Length and start with the gzip magic number before it is
renamed into place, and a 404 page served with status 200 (the 2018 files
are like that) is rejected by its size. Days already present and complete are
skipped, so the command can be rerun at any time ($INVEX_RAW is on /tmp,
which is wiped at boot). A manifest records the source, size and
Last-Modified header of every file.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import requests

from ..paths import RAW, ensure_space
from . import BASE_URL, ITCH_DAYS

MIN_BYTES = 100 * 1024**2  # every real day is > 3 GB; the 404 pages are 1,245 bytes
CHUNK = 4 * 1024**2
GZIP_MAGIC = b"\x1f\x8b"


def gz_dir() -> Path:
    return RAW / "itch" / "gz"


def local_path(day: str) -> Path:
    return gz_dir() / f"{day}.itch50.gz"


def _manifest_path() -> Path:
    return gz_dir() / "manifest.json"


def load_manifest() -> dict:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _save_manifest(m: dict) -> None:
    p = _manifest_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True))
    os.replace(tmp, p)


def head(url: str, session: requests.Session) -> dict:
    r = session.head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    size = int(r.headers.get("Content-Length", "0"))
    if size < MIN_BYTES:
        raise RuntimeError(f"{url}: {size} bytes (a 404 page?)")
    if r.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise RuntimeError(f"{url}: server does not accept byte ranges")
    return {"size": size, "last_modified": r.headers.get("Last-Modified", ""), "etag": r.headers.get("ETag", "")}


def _fetch_range(url: str, part: Path, start: int, stop: int, retries: int = 8) -> None:
    """Download bytes [start, stop) of `url` into `part`, resuming a partial part."""
    want = stop - start
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            have = 0
        try:
            hdr = {"Range": f"bytes={start + have}-{stop - 1}"}
            with requests.get(url, headers=hdr, stream=True, timeout=120) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206 Partial Content, got {r.status_code}")
                with open(part, "ab") as fh:
                    for buf in r.iter_content(CHUNK):
                        fh.write(buf)
        except (requests.RequestException, RuntimeError) as e:
            wait = min(300, 5 * 2**attempt)
            print(f"  {part.name}: {e!r}; retry in {wait}s", flush=True)
            time.sleep(wait)
    have = part.stat().st_size if part.exists() else 0
    if have != want:
        raise RuntimeError(f"{part}: {have} of {want} bytes after {retries} attempts")


def download_day(day: str, conns: int = 4, session: requests.Session | None = None) -> Path:
    dest = local_path(day)
    url = BASE_URL + ITCH_DAYS[day]
    session = session or requests.Session()
    info = head(url, session)
    if dest.exists() and dest.stat().st_size == info["size"]:
        return dest
    ensure_space(gz_dir(), info["size"] / 1024**3 * 2)  # parts + assembled copy
    bounds = [info["size"] * i // conns for i in range(conns + 1)]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(conns)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(conns) as ex:
        futs = [ex.submit(_fetch_range, url, parts[i], bounds[i], bounds[i + 1]) for i in range(conns)]
        for f in futs:
            f.result()
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                while buf := fh.read(64 * CHUNK):
                    out.write(buf)
    with open(tmp, "rb") as fh:
        magic = fh.read(2)
    if tmp.stat().st_size != info["size"] or magic != GZIP_MAGIC:
        raise RuntimeError(f"{day}: assembled {tmp.stat().st_size} bytes, magic {magic!r}")
    os.replace(tmp, dest)
    for p in parts:
        p.unlink()
    secs = time.time() - t0
    m = load_manifest()
    m[day] = {"url": url, **info, "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(secs)}
    _save_manifest(m)
    print(f"{day}: {info['size'] / 1e9:.2f} GB in {secs / 60:.1f} min ({info['size'] / secs / 1e6:.1f} MB/s)", flush=True)
    return dest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", default=["all"], help="YYYY-MM-DD ... or 'all'")
    ap.add_argument("--conns", type=int, default=4, help="parallel byte ranges per file")
    a = ap.parse_args(argv)
    days = list(ITCH_DAYS) if a.days == ["all"] else a.days
    unknown = [d for d in days if d not in ITCH_DAYS]
    if unknown:
        raise SystemExit(f"unknown days {unknown}; known: {list(ITCH_DAYS)}")
    gz_dir().mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    failed = []
    for d in days:
        try:
            download_day(d, a.conns, session)
        except Exception as e:  # keep going: one bad day must not block the rest
            print(f"{d}: FAILED {e!r}", flush=True)
            failed.append(d)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()

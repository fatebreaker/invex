"""Build the event tables of one ITCH day.

    python -m invex.itch.build --day 2019-01-30
    python -m invex.itch.build --day 2025-12-08 --parts 2   # large days: half the stocks per pass

Steps: decompress the download with pigz to $INVEX_RAW/itch/bin/<day>.bin
(kept for the next build unless --drop-bin), index it, read the stock
directory and system events, turn every stock with a directory entry into its
event table (invex.itch.annotate) and write $INVEX_DATA/itch/<day>/
{events.parquet, symbols.parquet, qa.json}. The tables start with a snapshot
of the book at 09:25 and end 5 minutes after the close.

The build stops before writing anything if the framing is broken (a message
length that does not match its type, an unknown type, a partial last
message), timestamps go backwards, or the close in the file differs from the
calendar; unresolved order references are counted in qa.json (expected 0).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from ..io.store import EventWriter, day_dir
from ..paths import BUDGET_GB, DATA, RAW, ensure_space
from ..provenance import provenance, write_json
from . import ITCH_DAYS
from .annotate import annotate
from .directory import stock_directory, system_events, timestamp_decreases
from .download import local_path
from .index import index_file, order_by_locate

HALF_DAYS = {"2025-11-28"}
NS = 1_000_000_000


def hms(h: int, m: int = 0, s: int = 0) -> int:
    return ((h * 60 + m) * 60 + s) * NS


def session(day: str) -> dict:
    close = hms(13) if day in HALF_DAYS else hms(16)
    return {"open": hms(9, 30), "close": close, "t_snap": hms(9, 25), "t_end": close + hms(0, 5)}


def bin_path(day: str) -> Path:
    return RAW / "itch" / "bin" / f"{day}.bin"


def decompress(day: str) -> Path:
    gz, out = local_path(day), bin_path(day)
    if out.exists():
        return out
    if not gz.exists():
        raise FileNotFoundError(f"{gz} (run python -m invex.itch.download --days {day})")
    ensure_space(out.parent, gz.stat().st_size * 4 / 1024**3)
    tmp = out.with_name(out.name + ".tmp")
    t0 = time.time()
    with open(tmp, "wb") as fh:
        subprocess.run(["pigz", "-dc", str(gz)], stdout=fh, check=True)
    tmp.replace(out)
    print(f"  decompressed {gz.stat().st_size / 1e9:.1f} -> {out.stat().st_size / 1e9:.1f} GB in {time.time() - t0:.0f}s", flush=True)
    return out


def build(day: str, parts: int = 1, drop_bin: bool = False) -> dict:
    if day not in ITCH_DAYS:
        raise SystemExit(f"unknown day {day}")
    t_start = time.time()
    sess = session(day)
    path = decompress(day)
    buf = np.memmap(path, dtype=np.uint8, mode="r")

    t0 = time.time()
    idx = index_file(buf, sess["t_snap"], sess["t_end"])
    t_index = time.time() - t0
    events = system_events(buf, idx)
    decreases = timestamp_decreases(buf, idx)
    problems = []
    if idx.bad_len or idx.unknown or idx.tail:
        problems.append(f"framing: bad_len={idx.bad_len} unknown={idx.unknown} tail={idx.tail}")
    if decreases:
        problems.append(f"{decreases} timestamp decreases")
    if "M" in events and abs(events["M"] - sess["close"]) > 60 * NS:
        problems.append(f"end of market hours at {events['M'] / 3600e9:.3f}h, calendar close {sess['close'] / 3600e9:.3f}h")
    if problems:
        raise RuntimeError(f"{day}: " + "; ".join(problems))

    directory = stock_directory(buf, idx)
    directory = directory[idx.nrows[directory["locate"].to_numpy()] > 0].reset_index(drop=True)
    locs = directory["locate"].to_numpy(dtype=np.int64)
    bound = idx.nrows[locs]
    perm, start = order_by_locate(idx.loc, idx.counts)

    out_dir = day_dir("itch", day)
    est_gb = bound.sum() * 12 / 1024**3  # ~12 bytes per row compressed (upper bound: rows <= bound)
    ensure_space(out_dir, est_gb, BUDGET_GB, DATA)
    writer = EventWriter(out_dir / "events.parquet")
    stats = []
    t0 = time.time()
    # split the stocks into parts of similar size; stocks keep their directory order across parts
    cuts = np.searchsorted(np.cumsum(bound), np.linspace(0, bound.sum(), parts + 1)[1:-1])
    for part in np.split(np.arange(len(locs)), cuts):
        if part.size == 0:
            continue
        out = annotate(buf, idx.off, idx.typ, perm, start, locs[part], bound[part], idx.max_ref,
                       sess["t_snap"], sess["t_end"])
        for j, k in enumerate(part):
            a, n = out["row_start"][j], out["n_rows"][j]
            rg = writer.write({c: out[c][a: a + n] for c in S.COLUMNS})
            stats.append({"locate": int(locs[k]), "rg": rg, "n_rows": int(n), "n_snap": int(out["n_snap"][j]),
                          "n_err": int(out["n_err"][j]), "n_over": int(out["n_over"][j])})
        del out
    writer.close()
    t_annotate = time.time() - t0

    counts = pd.DataFrame({f"n_{c}": idx.counts[locs, ord(c)] for c in "AFECXDUPQ"})
    sym = pd.concat([directory, counts], axis=1).merge(pd.DataFrame(stats), on="locate")
    sym.to_parquet(out_dir / "symbols.parquet", index=False)

    qa = {
        "day": day,
        "session": sess,
        "system_events": events,
        "messages": idx.n,
        "max_ref": idx.max_ref,
        "bad_len": idx.bad_len,
        "unknown_types": idx.unknown,
        "tail_bytes": idx.tail,
        "timestamp_decreases": decreases,
        "symbols": len(sym),
        "rows": int(sym["n_rows"].sum()),
        "unresolved_refs": int(sym["n_err"].sum()),
        "overexecutions": int(sym["n_over"].sum()),
        "events_parquet_gb": (out_dir / "events.parquet").stat().st_size / 1e9,
        "seconds": {"index": round(t_index), "annotate_write": round(t_annotate), "total": round(time.time() - t_start)},
        "provenance": provenance(source=str(local_path(day)), bin_bytes=int(buf.shape[0])),
    }
    write_json(out_dir / "qa.json", qa)
    del buf
    if drop_bin:
        path.unlink()
    print(f"{day}: {qa['messages']:,} messages, {len(sym):,} symbols, {qa['rows']:,} rows, "
          f"{qa['events_parquet_gb']:.1f} GB, unresolved {qa['unresolved_refs']}, over {qa['overexecutions']}, "
          f"{qa['seconds']}", flush=True)
    return qa


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", nargs="+", required=True)
    ap.add_argument("--parts", type=int, default=1, help="annotate the stocks in this many passes (saves RAM)")
    ap.add_argument("--drop-bin", action="store_true", help="delete the decompressed file afterwards")
    ap.add_argument("--force", action="store_true", help="rebuild even if qa.json exists")
    a = ap.parse_args(argv)
    for day in a.day:
        if (day_dir("itch", day) / "qa.json").exists() and not a.force:
            print(f"{day}: already built", flush=True)
            continue
        build(day, a.parts, a.drop_bin)


if __name__ == "__main__":
    main()

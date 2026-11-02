"""Frozen episode tables for validation and the sealed test stages, and the sealed loader.

    python -m invex.eval.freeze --name val_v0    --split val  --days 2019-10-30 2019-12-30
    python -m invex.eval.freeze --name stage1_v0 --split test --days 2020-01-30 2021-07-13 2021-08-13 --sealed
    python -m invex.eval.freeze --name stage2_v0 --split test --days 2025-12-08 ... --sealed
    python -m invex.eval.freeze --name crypto_test_v0 --venue coinbase --split all --days 2024-01-01 ... --sealed

Freezing only draws episodes (start times, task cells, window statistics):
no policy is run. The table is written to $INVEX_DATA/episodes/<name>.parquet
and its hash, counts and days to prereg/episodes_<name>.json. `load(name)`
returns a table and refuses a sealed one unless `unseal=True`; every load of
a sealed table is appended to prereg/access_log.jsonl (who, when, why), so
"evaluated once" can be audited.

Stocks first seen after the split was frozen (on later days) count as test
assets, since no policy can have seen them.
"""

from __future__ import annotations

import argparse
import getpass
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ..env.episodes import book_episodes, table_hash
from ..io.store import day_dir, read_group, symbols
from ..paths import DATA, REPO
from ..provenance import provenance, write_json
from .universe import eligible, load_splits

N_PER_BOOK = 100


def _task(args):
    venue, day, stock, n, ko, tag, min_onetick = (*args, 0.0)[:7]
    book, open_ns, close_ns, tick = book_of(venue, day, stock)
    df = book_episodes(book, venue, day, stock, open_ns, close_ns, n, ko, seed_tag=tag, tick=tick,
                       **scales_of(venue, day, stock))
    if min_onetick > 0 and len(df):  # only episodes whose book was at one tick for this share of the window
        df = df[df["w_onetick"] >= min_onetick].reset_index(drop=True)
    return df


CRYPTO_MIN_MINUTES = 75  # a units window (30 min) plus room for start times and horizons


def book_of(venue: str, day: str, stock: str):
    """(replayed book, session open, session close, tick) of one symbol-day. ITCH: the day's continuous
    session and the penny tick (the universe excludes sub-dollar prices); Coinbase: the pair's own stream
    (snapshot to last message) and its tick."""
    from ..sim.tracker import prepare

    sym = symbols(venue, day).set_index("stock")
    if venue == "coinbase":
        o, c, tick = int(sym.loc[stock, "t_snap"]), int(sym.loc[stock, "t_end"]), int(sym.loc[stock, "tick"])
    else:
        qa = json.loads((day_dir(venue, day) / "qa.json").read_text())
        o, c, tick = qa["session"]["open"], qa["session"]["close"], 100
    return prepare(read_group(venue, day, int(sym.loc[stock, "rg"])), o, c), o, c, tick


def scales_of(venue: str, day: str, stock: str) -> dict:
    """{'px_scale', 'qty_scale'} of one symbol-day: per pair-day on Coinbase (prices in its ticks), the
    venue's otherwise."""
    if venue == "coinbase":
        r = symbols(venue, day).set_index("stock").loc[stock]
        return {"px_scale": float(r["px_scale"]), "qty_scale": float(r["qty_scale"])}
    return {}


CRYPTO_MAX_MISMATCH = 1e-4  # feed inconsistencies tolerated per message (a done whose size disagrees with its fills)


def crypto_pairs(day: str) -> list[str]:
    """Coinbase pairs whose stream normalised cleanly and is long enough for episodes. A few done or change
    messages whose sizes disagree with the order's own fills (the feed's, not ours: the sequence is complete) are
    tolerated; the replay deletes whatever the fills left."""
    sym = symbols("coinbase", day)
    bad = (sym["done_mismatch"] + sym["change_mismatch"]) / sym["messages"].clip(lower=1)
    ok = ((sym["unresolved"] == 0) & (bad <= CRYPTO_MAX_MISMATCH)
          & ((sym["t_end"] - sym["t_snap"]) >= CRYPTO_MIN_MINUTES * 60 * 10**9))
    return sorted(sym.loc[ok, "stock"])


def stocks_for(split: str, day: str, venue: str = "itch") -> list[str]:
    if venue == "coinbase":
        if split != "all":
            raise ValueError("crypto pairs are split by day, not by asset: use split 'all'")
        return crypto_pairs(day)
    sp = load_splits()["split"]
    st = pd.read_parquet(day_dir(venue, day) / "daystats.parquet") if (day_dir(venue, day) / "daystats.parquet").exists() else None
    if st is None:
        raise FileNotFoundError(f"run `python -m invex.eval.universe stats --venue {venue} --day {day}` first")
    ok = st[eligible(st)]["stock"]
    if split == "all":  # every eligible stock (the IEX-trained reference)
        return sorted(ok)
    if split == "test":
        return sorted(s for s in ok if sp.get(s, "test") == "test")
    return sorted(s for s in ok if sp.get(s) == split)


def freeze(name: str, split: str, days: list[str], sealed: bool, n: int = N_PER_BOOK, workers: int = 32,
           venue: str = "itch", min_onetick: float = 0.0) -> dict:
    path = DATA / "episodes" / f"{name}.parquet"
    if path.exists():
        raise SystemExit(f"{path} exists: frozen tables are never rewritten")
    ko = load_splits()["ko_constants"]
    jobs = [(venue, d, s, n, ko, name, min_onetick) for d in days for s in stocks_for(split, d, venue)]
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            parts = [p for p in ex.map(_task, jobs, chunksize=2) if len(p)]
    finally:
        os.environ.clear()
        os.environ.update(env)
    df = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {"name": name, "venue": venue, "split": split, "days": days, "sealed": sealed, "episodes": len(df),
            "stocks": int(df["stock"].nunique()), "books": len(parts), "per_book": n, "min_onetick": min_onetick,
            "sha256": table_hash(df),
            "families": df["family"].value_counts().to_dict(), "provenance": provenance()}
    write_json(REPO / "prereg" / f"episodes_{name}.json", meta)
    print(f"{name}: {len(df):,} episodes, {meta['stocks']} stocks, {len(parts)} books, sha256 {meta['sha256'][:12]}")
    return meta


def load(name: str, unseal: bool = False, reason: str = "") -> pd.DataFrame:
    meta = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())
    if meta["sealed"]:
        if not unseal:
            raise PermissionError(f"{name} is sealed: pass unseal=True with a reason (logged)")
        with open(REPO / "prereg" / "access_log.jsonl", "a") as fh:
            fh.write(json.dumps({"table": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "user": getpass.getuser(), "reason": reason}) + "\n")
    df = pd.read_parquet(DATA / "episodes" / f"{name}.parquet")
    if table_hash(df) != meta["sha256"]:
        raise RuntimeError(f"{name}: table hash differs from the frozen one")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    ap.add_argument("--venue", default="itch", choices=["itch", "coinbase", "iex"])
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--per-book", type=int, default=N_PER_BOOK)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-onetick", type=float, default=0.0,
                    help="keep episodes whose book was at one tick for at least this share of the trailing window")
    a = ap.parse_args(argv)
    freeze(a.name, a.split, a.days, a.sealed, a.per_book, a.workers, a.venue, a.min_onetick)


if __name__ == "__main__":
    main()

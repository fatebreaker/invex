"""Day statistics per symbol, the eligible universe, and the frozen asset split.

    python -m invex.eval.universe stats --day 2019-01-30 ...     # data/itch/<day>/daystats.parquet
    python -m invex.eval.universe split --days <training days>    # prereg/splits_v0.json (+ sha256)

Eligible common stocks (per training day): authentic issue (not a test
symbol), not an ETP, issue class common or ordinary share, round lot 100,
VWAP >= $2, >= 500 trades in the continuous session; a symbol is in the
universe if it is eligible on at least 3 of the training days. Strata:
listing market (Nasdaq vs other) x eta_sigma tercile x activity decile
(median over training days); within each stratum symbols are assigned 50%
train / 15% validation / 35% test by a seeded permutation of their sorted
tickers. Symbols first seen on later days are test assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..io.store import day_dir, read_group, symbols
from ..paths import REPO
from ..provenance import provenance, write_json

NASDAQ_LISTED = {"Q", "G", "S"}
SPLIT_SEED = 20261014
FRACTIONS = {"train": 0.50, "val": 0.15, "test": 0.35}


def _session(day: str, venue: str = "itch") -> tuple[int, int]:
    qa = json.loads((day_dir(venue, day) / "qa.json").read_text())
    return qa["session"]["open"], qa["session"]["close"]


def _stats_one(args) -> dict:
    day, rg, stock, open_ns, close_ns, venue = (*args, "itch")[:6]
    from ..book.calib import calibrate
    from ..env.units import relative_tick
    from ..sim.tracker import prepare

    ev = read_group(venue, day, rg)
    book = prepare(ev, open_ns, close_ns)
    cal = calibrate(book, open_ns, close_ns)
    w = cal.window(close_ns, int((close_ns - open_ns) // 1_000_000_000))
    eta_s_sigma, eta_s = relative_tick(w) if w.trades > 0 else (np.nan, np.nan)
    return {"stock": stock, "trades": w.trades, "volume": w.volume, "vwap": w.vwap / 1e4, "spread": w.spread / 1e4,
            "depth": w.depth, "onetick": w.onetick, "sigma_1min": w.sigma_1min, "sigma_trade": w.sigma_trade / 1e4,
            "eta_sigma": eta_s_sigma, "eta_s": eta_s, "W": w.kyle_obizhaeva_w() if w.trades > 0 else np.nan,
            "fifo_ok": book.qa["fifo_ok"], "fifo_bad": book.qa["fifo_bad"]}


def day_stats(day: str, workers: int = 32, venue: str = "itch") -> pd.DataFrame:
    """IEX tables hold common stocks only (invex.iex.normalize), with the attributes of the newest ITCH directory."""
    sym = symbols(venue, day)
    cand = sym[(sym["authenticity"] == "P") & (sym["etp_flag"] != "Y") & sym["issue_class"].isin(["C", "O"])]
    open_ns, close_ns = _session(day, venue)
    jobs = [(day, int(r.rg), r.stock, open_ns, close_ns, venue)
            for r in cand.sort_values("n_rows", ascending=False).itertuples()]
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_stats_one, jobs, chunksize=4))
    finally:
        os.environ.clear()
        os.environ.update(env)
    df = pd.DataFrame(res).merge(cand[["stock", "market_category", "round_lot", "issue_class"]], on="stock")
    df.insert(0, "day", day)
    df.to_parquet(day_dir(venue, day) / "daystats.parquet", index=False)
    return df


def eligible(df: pd.DataFrame) -> pd.Series:
    return (df["round_lot"] == 100) & (df["vwap"] >= 2.0) & (df["trades"] >= 500) & np.isfinite(df["eta_sigma"])


def ko_constants(st: pd.DataFrame, session_s: float = 23_400.0) -> dict:
    """Kyle-Obizhaeva unit constants: median R2 unit = median R3 unit over full-session symbol-days.

    Raw KO units (x = W / W*): time x^(-2/3), size V x^(-2/3), price P sigma_day x^(-1/3) (price units);
    R3 units: session seconds / trades, volume / trades, mean spread (price units)."""
    from ..env.units import W_STAR

    x = st["W"].to_numpy() / W_STAR
    raw_t = x ** (-2 / 3)
    raw_q = st["volume"].to_numpy() * x ** (-2 / 3)
    raw_p = st["vwap"].to_numpy() * 1e4 * st["sigma_1min"].to_numpy() * np.sqrt(390.0) * x ** (-1 / 3)
    r3_t = session_s / st["trades"].to_numpy()
    r3_q = st["volume"].to_numpy() / st["trades"].to_numpy()
    r3_p = st["spread"].to_numpy() * 1e4
    med = lambda a: float(np.nanmedian(a))  # noqa: E731
    return {"c_t": med(r3_t) / med(raw_t), "c_q": med(r3_q) / med(raw_q), "c_p": med(r3_p) / med(raw_p)}


def make_splits(days: list[str]) -> dict:
    stats = pd.concat([pd.read_parquet(day_dir("itch", d) / "daystats.parquet") for d in days])
    stats["ok"] = eligible(stats)
    per = stats.groupby("stock").agg(n_ok=("ok", "sum"), market=("market_category", "first"),
                                     eta=("eta_sigma", "median"), trades=("trades", "median"))
    uni = per[per["n_ok"] >= min(3, len(days))].copy()
    uni["listing"] = np.where(uni["market"].isin(NASDAQ_LISTED), "nasdaq", "other")
    uni["eta_t"] = pd.qcut(np.log(uni["eta"]), 3, labels=False)
    uni["act_d"] = pd.qcut(np.log(uni["trades"]), 10, labels=False)
    rng = np.random.default_rng(SPLIT_SEED)
    split = {}
    for _key, g in uni.groupby(["listing", "eta_t", "act_d"]):
        names = sorted(g.index)
        rng.shuffle(names)
        n = len(names)
        n_tr = int(round(n * FRACTIONS["train"]))
        n_va = int(round(n * FRACTIONS["val"]))
        for i, s in enumerate(names):
            split[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    eta_cuts = np.exp(np.quantile(np.log(uni["eta"]), [1 / 3, 2 / 3])).tolist()
    ko = ko_constants(stats[stats["ok"] & stats["stock"].isin(uni.index)])
    body = {"version": "v0", "days": days, "seed": SPLIT_SEED, "fractions": FRACTIONS, "eta_sigma_tercile_cuts": eta_cuts,
            "ko_constants": ko,
            "n": {k: sum(v == k for v in split.values()) for k in FRACTIONS}, "split": dict(sorted(split.items())),
            "strata": {s: [r.listing, int(r.eta_t), int(r.act_d)] for s, r in uni.sort_index().iterrows()}}
    blob = json.dumps(body, sort_keys=True).encode()
    out = {**body, "sha256": hashlib.sha256(blob).hexdigest(), "provenance": provenance()}
    write_json(REPO / "prereg" / "splits_v0.json", out)
    print(f"universe {len(uni)} symbols: {out['n']}; eta_sigma tercile cuts {eta_cuts}; sha256 {out['sha256'][:12]}")
    return out


def load_splits(version: str = "v0") -> dict:
    return json.loads((REPO / "prereg" / f"splits_{version}.json").read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("stats")
    a1.add_argument("--day", nargs="+", required=True)
    a1.add_argument("--workers", type=int, default=32)
    a1.add_argument("--venue", default="itch", choices=["itch", "iex"])
    a2 = sub.add_parser("split")
    a2.add_argument("--days", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        for d in a.day:
            df = day_stats(d, a.workers, a.venue)
            print(f"{d}: {len(df)} common stocks, eligible {int(eligible(df).sum())}", flush=True)
    else:
        if (REPO / "prereg" / "splits_v0.json").exists():
            raise SystemExit("prereg/splits_v0.json exists: splits are frozen (write a new version instead)")
        make_splits(a.days)


if __name__ == "__main__":
    main()

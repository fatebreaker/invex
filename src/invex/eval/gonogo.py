"""Go/no-go test before any RL: do passive-order dynamics collapse in invariant units?

    python -m invex.eval.gonogo measure --days 2019-01-30 2019-03-27 ... [--n-stocks 300]
    python -m invex.eval.gonogo analyze --days ...

measure: for each pilot stock-day (training split only, stratified), 200
virtual one-lot orders join the near touch at random times between 10:00 and
15:20 (100 us latency, exact fill rule) and are tracked for up to 30 minutes
(not past 15:50). Each probe records the trailing 30-minute window statistics
(so every unit system can be computed afterwards), the shares ahead at
arrival, the times of its first and completing fills, and the mid at a
log-spaced grid of delays after the completing fill (markouts) and at the end
of tracking (opportunity cost). Results: results/gonogo/probes_<day>.parquet.
Each probe time also runs part (C)'s episodes (one round lot, three horizon
families: 30 mean inter-trade times, 30 Kyle-Obizhaeva time units, 60 s):
the paths of invex.sim.probe.rest_then_cross_paths, in episodes_<day>.parquet.

analyze: computes, per unit system, how much the per-asset curves disagree
(between-asset variance net of sampling noise, tau^2) for fill probabilities
at horizons of 1..100 time units and markouts at 1/10/30 time units, against
a shuffled-units placebo, plus the slope left on log activity; see
prereg/gonogo_rule.md for the decision rule fixed before running it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..io.store import day_dir, read_group, symbols
from ..paths import RESULTS
from ..provenance import provenance, write_json

NS = 1_000_000_000
LATENCY_NS = 100_000
N_PROBES = 200
WINDOW_S = 1800
MAX_TRACK_S = 1800
MARKOUT_DELAYS_S = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


def hms(h, m=0, s=0) -> int:
    return ((h * 60 + m) * 60 + s) * NS


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest(), "little")


K = 10
FAMILY_UNITS = 30.0  # F_MS / F_KO horizons in time units
FAMILY_P_S = 60.0  # F_P horizon in seconds


def probe_symbol(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    day, rg, stock, open_ns, close_ns, c_t = args
    from ..book.calib import calibrate
    from ..env.units import W_STAR
    from ..sim import probe
    from ..sim.tracker import prepare

    book = prepare(read_group("itch", day, rg), open_ns, close_ns)
    cal = calibrate(book, open_ns, close_ns)
    rng = np.random.default_rng(_seed(day, stock, "gonogo-v0"))
    t_lo, t_hi, t_end = open_ns + hms(0, 30), close_ns - hms(0, 40), close_ns - hms(0, 10)
    t0 = np.sort(rng.integers(t_lo, t_hi, N_PROBES))
    sides = np.where(np.arange(N_PROBES) % 2 == 0, 1, -1)
    t_arr = t0 + LATENCY_NS
    bid, ask = probe.touch_at(book, t_arr)
    ok = (bid > 0) & (ask > 0)
    prices = np.where(sides > 0, bid, ask)
    t_max = np.minimum(t0 + MAX_TRACK_S * NS, t_end)
    res = probe.passive_probes(book, sides[ok], prices[ok], np.full(ok.sum(), 100), t_arr[ok], t_max[ok])
    rows = []
    t0k, sk, pk, tmk = t0[ok], sides[ok], prices[ok], t_max[ok]
    mid0 = probe.mid_at(book, t_arr[ok])
    mid_end = probe.mid_at(book, tmk)
    done = res["t_done"]
    delays = np.array(MARKOUT_DELAYS_S)
    q_times = np.where(done[:, None] > 0, done[:, None] + (delays[None, :] * NS).astype(np.int64), 0)
    mids_after = probe.mid_at(book, q_times.ravel()).reshape(q_times.shape)
    mids_after[done <= 0] = np.nan
    horizons = np.zeros((len(t0k), 3), dtype=np.int64)
    for j in range(len(t0k)):
        w = cal.window(int(t0k[j]), WINDOW_S)
        x = w.kyle_obizhaeva_w() / W_STAR if w.trades > 0 else np.nan
        t_ko = c_t * x ** (-2 / 3) if np.isfinite(x) and x > 0 else np.nan
        horizons[j] = [int(FAMILY_UNITS * w.inter_trade * NS), int(FAMILY_UNITS * t_ko * NS) if np.isfinite(t_ko) else 0,
                       int(FAMILY_P_S * NS)]
        rows.append({
            "day": day, "stock": stock, "probe": j, "t0": int(t0k[j]), "side": int(sk[j]), "price": int(pk[j]),
            "ahead0": int(res["ahead0"][j]), "status": int(res["status"][j]), "t_first": int(res["t_first"][j]),
            "t_done": int(done[j]), "t_max": int(tmk[j]), "mid0": mid0[j], "mid_end": mid_end[j],
            "spread0": float(ask[ok][j] - bid[ok][j]),
            "w_seconds": w.seconds, "w_trades": w.trades, "w_volume": w.volume, "w_vwap": w.vwap, "w_mid": w.mid,
            "w_spread": w.spread, "w_depth": w.depth, "w_onetick": w.onetick, "w_sigma_1min": w.sigma_1min,
            "tick": w.tick, **{f"mid_after_{d:g}": mids_after[j, i] for i, d in enumerate(MARKOUT_DELAYS_S)},
        })
    probes = pd.DataFrame(rows)
    # part (C): rest-then-cross paths, three horizon families per probe time, one round lot
    eps = []
    for f, fam in enumerate(("MS", "KO", "P")):
        hz = horizons[:, f]
        use = (hz > 0) & (t0k + hz + LATENCY_NS < t_end)
        if not use.any():
            continue
        pth = probe.rest_then_cross_paths(book, sk[use], np.full(use.sum(), 100), t0k[use], hz[use], K, LATENCY_NS)
        idx = np.flatnonzero(use)
        for r, j in enumerate(idx):
            eps.append({"day": day, "stock": stock, "probe": int(j), "family": fam, "T_ns": int(hz[j]),
                        "joined": int(pth["joined"][r]),
                        **{c: pth[c][r].tolist() for c in ("filled", "cash", "ahead", "xfill", "xcash", "xshort")}})
    return probes, pd.DataFrame(eps)


def pilot_stocks(day_list: list[str], n: int, seed: int = 7) -> list[str]:
    """n training-split stocks, stratified by eta tercile x activity decile (fixed seed)."""
    from .universe import load_splits

    sp = load_splits()
    train = [s for s, v in sp["split"].items() if v == "train"]
    strata = pd.DataFrame({"stock": train, "key": [tuple(sp["strata"][s][1:]) for s in train]})
    rng = np.random.default_rng(seed)
    per = max(1, n // strata["key"].nunique())
    pick = []
    for _k, g in strata.groupby("key"):
        names = sorted(g["stock"])
        pick += list(rng.choice(names, size=min(per, len(names)), replace=False))
    return sorted(pick)[:n] if len(pick) >= n else sorted(pick)


def measure(day: str, stocks: list[str], workers: int = 32) -> pd.DataFrame:
    qa = json.loads((day_dir("itch", day) / "qa.json").read_text())
    open_ns, close_ns = qa["session"]["open"], qa["session"]["close"]
    sym = symbols("itch", day).set_index("stock")
    from .universe import load_splits

    c_t = load_splits()["ko_constants"]["c_t"]
    jobs = [(day, int(sym.loc[s, "rg"]), s, open_ns, close_ns, c_t) for s in stocks if s in sym.index]
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(probe_symbol, jobs, chunksize=2))
    finally:
        os.environ.clear()
        os.environ.update(env)
    df = pd.concat([p for p, _e in parts], ignore_index=True)
    eps = pd.concat([e for _p, e in parts if len(e)], ignore_index=True)
    out = RESULTS / "gonogo"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"probes_{day}.parquet", index=False)
    eps.to_parquet(out / f"episodes_{day}.parquet", index=False)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("measure")
    a1.add_argument("--days", nargs="+", required=True)
    a1.add_argument("--n-stocks", type=int, default=300)
    a1.add_argument("--workers", type=int, default=32)
    a2 = sub.add_parser("analyze")
    a2.add_argument("--days", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        stocks = pilot_stocks(a.days, a.n_stocks)
        meta = {"days": a.days, "stocks": stocks, "n_probes": N_PROBES, "latency_ns": LATENCY_NS,
                "window_s": WINDOW_S, "max_track_s": MAX_TRACK_S, "provenance": provenance()}
        write_json(RESULTS / "gonogo" / "measure_meta.json", meta)
        for d in a.days:
            t0 = time.time()
            df = measure(d, stocks, a.workers)
            filled = (df["t_done"] > 0).mean()
            print(f"{d}: {df['stock'].nunique()} stocks, {len(df):,} probes, filled within tracking {filled:.1%}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    else:
        from .gonogo_analysis import analyze

        analyze(a.days)


if __name__ == "__main__":
    main()

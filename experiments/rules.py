"""Tuned queue-rule baselines: a grid of (theta, iota) rules on episodes, for per-asset and pooled tuning.

    # the grid on each validation stock's own training days (per-asset tuning) and on training stocks (pooled)
    python experiments/rules.py grid --split val   --days D1 ... D5 --out val_own
    python experiments/rules.py grid --split train --days D1 D2 --max-stocks 300 --out train_pool
    # the same grid on a frozen table (selection happens in the analysis, never on the table itself)
    python experiments/rules.py grid --episodes val_v0 --out val_v0

QueueRule (invex.eval.policies) joins the near touch and crosses once the queue ahead, in mean trade sizes, exceeds
theta times the time left, in mean inter-trade times, or the imbalance against the agent exceeds iota. The
per-asset baseline picks, for each stock, the rule with the lowest mean cost on that stock's own other days; the
pooled baseline picks one rule for all stocks from the training stocks. Writes results/rules/grid_<out>.parquet:
one row per (episode, rule) with the cost in spread units.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from invex.env.episodes import book_episodes
from invex.env.units import unit_frame, valid_units
from invex.eval import freeze
from invex.eval import policies as P
from invex.eval.freeze import book_of, scales_of, stocks_for
from invex.eval.universe import load_splits
from invex.paths import RESULTS
from invex.provenance import provenance, write_json
from invex.venues import venue as venue_of

K = 10
OUT = RESULTS / "rules"
THETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)
IOTAS = (0.2, 0.4, 0.6, 0.8, 1.01)  # 1.01: never cross on imbalance


def grid_policies() -> list:
    return [P.QueueRule(th, io, "R3", name=f"rule_{th:g}_{io:g}") for th in THETAS for io in IOTAS]


def grid_task(args):
    venue, day, stock, eps, n, ko, tag = args
    book, o, c, tick = book_of(venue, day, stock)
    if eps is None:
        eps = book_episodes(book, venue, day, stock, o, c, n, ko, seed_tag=tag, tick=tick, **scales_of(venue, day, stock))
        if eps.empty:
            return None
        eps = eps[valid_units(unit_frame(eps, ko))]
        if eps.empty:
            return None
    eps = eps.reset_index(drop=True)
    uf = unit_frame(eps, ko)
    res = P.evaluate_book(book, eps, uf, grid_policies(), K, venue_of(venue).latency_ns, tick=tick)
    res = res.merge(eps[["key", "Q", "w_spread"]], on="key")
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res["day"], res["stock"] = day, stock
    return res[["key", "day", "stock", "policy", "c_spread", "filled"]]


def main(argv=None):
    from transfer import pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--split", choices=["train", "val", "test", "all"])
    g.add_argument("--days", nargs="+", default=[])
    g.add_argument("--episodes", default="", help="a frozen table instead of drawn episodes")
    g.add_argument("--venue", default="itch", choices=["itch", "coinbase"])
    g.add_argument("--max-stocks", type=int, default=0, help="a seeded sample of stocks per day")
    g.add_argument("--per-book", type=int, default=100)
    g.add_argument("--out", required=True)
    g.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    ko = load_splits()["ko_constants"]
    t0 = time.time()
    if a.episodes:
        eps = freeze.load(a.episodes)  # sealed tables are refused: tuning never touches them
        jobs = [(v, d, s, g_, 0, ko, "") for (v, d, s), g_ in eps.groupby(["venue", "day", "stock"])]
    else:
        jobs = []
        for d in a.days:
            stocks = stocks_for(a.split, d, a.venue)
            if a.max_stocks and len(stocks) > a.max_stocks:
                stocks = sorted(np.random.default_rng(len(d)).choice(stocks, a.max_stocks, replace=False))
            jobs += [(a.venue, d, s, None, a.per_book, ko, f"rules-{a.out}") for s in stocks]
    res = pd.concat(pool(grid_task, jobs, a.workers), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / f"grid_{a.out}.parquet", index=False)
    write_json(OUT / f"grid_{a.out}.json", {"jobs": len(jobs), "rows": len(res), "seconds": round(time.time() - t0),
                                            "rules": len(grid_policies()), "provenance": provenance(args=vars(a))})
    best = res.groupby("policy")["c_spread"].mean().sort_values()
    print(f"grid {a.out}: {len(jobs)} books, {res['key'].nunique():,} episodes, {time.time() - t0:.0f}s; best pooled:")
    print(best.head(5).round(4).to_string())


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    main()

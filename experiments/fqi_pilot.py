"""First end-to-end FQI pilot, inside the training split only (validation and test stay untouched).

    python experiments/fqi_pilot.py --fit-days 2019-01-30 2019-03-27 --eval-day 2019-08-30 --stocks 300

The training-split stocks are halved by a hash of the ticker: FQI is fitted on
the first half (fit days) and evaluated zero-shot on the second half on a day
it never saw, against the baselines (immediate cross, post-then-cross, the
pooled queue/imbalance rule tuned on the fit half). Writes
results/fqi_pilot/{models/*.pkl, eval.parquet, summary.json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from invex.agents import fqi, fqi_adv
from invex.env import execution as ex
from invex.env.episodes import book_episodes, seed_of
from invex.env.units import unit_frame, valid_units
from invex.eval import policies as P
from invex.eval.stats import cluster_bootstrap
from invex.eval.universe import load_splits
from invex.io.store import day_dir, read_group, symbols
from invex.paths import RESULTS
from invex.provenance import provenance, write_json
from invex.sim.tracker import EXACT, prepare

K = 10
LATENCY = 100_000
OUT = RESULTS / "fqi_pilot"
ALGO = {"plain": fqi, "adv": fqi_adv}


def _session(day):
    qa = json.loads((day_dir("itch", day) / "qa.json").read_text())
    return qa["session"]["open"], qa["session"]["close"]


def half(stock: str) -> int:
    return hashlib.blake2b(stock.encode(), digest_size=2).digest()[0] % 2


def _book(day, stock):
    sym = symbols("itch", day).set_index("stock")
    open_ns, close_ns = _session(day)
    return prepare(read_group("itch", day, int(sym.loc[stock, "rg"])), open_ns, close_ns), open_ns, close_ns


def gen_task(args):
    day, stock, n, ko, algo = args
    book, o, c = _book(day, stock)
    eps = book_episodes(book, "itch", day, stock, o, c, n, ko, seed_tag="fqi-pilot-fit")
    if eps.empty:
        return None
    data = ALGO[algo].generate_book(book, eps, K, LATENCY, EXACT, 100, seed_of(day, stock, "behaviour"))
    return eps, data


def eval_task(args):
    day, stock, n, ko, model_paths, rules = args
    book, o, c = _book(day, stock)
    eps = book_episodes(book, "itch", day, stock, o, c, n, ko, seed_tag="fqi-pilot-eval")
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    ok = valid_units(uf)
    eps = eps[ok].reset_index(drop=True)
    uf = unit_frame(eps, ko)
    pols = [P.Immediate(), P.PostThenCross()] + [P.QueueRule(*r) for r in rules]
    pols += [P.FQIPolicy(fqi_adv.load_model(p)) for p in model_paths]
    res = P.evaluate_book(book, eps, uf, pols, K, LATENCY)
    return res.merge(eps[["key", "stock", "day", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]], on="key")


def pool(fn, jobs, workers):
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["INVEX_PRED_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex_:
            return [r for r in ex_.map(fn, jobs, chunksize=1) if r is not None]
    finally:
        os.environ.clear()
        os.environ.update(env)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-days", nargs="+", default=["2019-01-30", "2019-03-27"])
    ap.add_argument("--eval-day", default="2019-08-30")
    ap.add_argument("--stocks", type=int, default=300, help="training-split stocks per half (at most)")
    ap.add_argument("--fit-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--reps", nargs="+", default=["R0", "R2", "R3", "R3tick"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--algo", default="plain", choices=list(ALGO))
    ap.add_argument("--skip-rule", action="store_true", help="reuse no queue-rule tuning (baselines only)")
    a = ap.parse_args(argv)
    global OUT
    OUT = RESULTS / ("fqi_pilot" if a.algo == "plain" else f"fqi_pilot_{a.algo}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    sp = load_splits()
    ko = sp["ko_constants"]
    train = sorted(s for s, v in sp["split"].items() if v == "train")
    rng = np.random.default_rng(0)
    fit_stocks = sorted(rng.permutation([s for s in train if half(s) == 0])[: a.stocks])
    eval_stocks = sorted(rng.permutation([s for s in train if half(s) == 1])[: a.stocks])
    t0 = time.time()

    # 1. transitions on the fit half
    jobs = []
    for d in a.fit_days:
        have = set(symbols("itch", d)["stock"])
        jobs += [(d, s, a.fit_episodes, ko, a.algo) for s in fit_stocks if s in have]
    parts = pool(gen_task, jobs, a.workers)
    eps = pd.concat([p[0] for p in parts], ignore_index=True)
    data = ALGO[a.algo].concat([p[1] for p in parts])
    uf = unit_frame(eps, ko)
    ok = valid_units(uf)
    keep = np.flatnonzero(ok)
    eps = eps.iloc[keep].reset_index(drop=True)
    data = fqi_adv.subset(data, keep) if a.algo == "adv" else \
        {k: (v[:, keep] if k in ("raw", "masks", "costs", "done_next") else v[:, :, keep]) for k, v in data.items()}
    uf = unit_frame(eps, ko)
    c_r0 = float(np.median(uf["R3"]["t"]))
    print(f"generated {len(eps):,} fit episodes on {len(parts)} books in {time.time() - t0:.0f}s", flush=True)

    # 2. FQI per representation
    paths = []
    for rep in a.reps:
        t1 = time.time()
        m = ALGO[a.algo].fit(data, eps, uf, rep, K, c_r0 if rep.startswith("R0") else 1.0, n_rounds=a.rounds,
                             threads=32, log=lambda s: print(s, flush=True))
        path = OUT / "models" / f"fqi_{rep}.pkl"
        m.save(path)
        paths.append(str(path))
        print(f"fitted {rep} in {time.time() - t1:.0f}s", flush=True)

    # 3. pooled queue rule (R3 units) tuned on the fit half: one evaluation pass over a small grid
    grid = [(th, io, "R3") for th in (0.3, 1.0, 3.0, np.inf) for io in (0.3, 0.6, 2.0)]
    if a.skip_rule:
        grid = [(3.0, 2.0, "R3")]
    tune_jobs = [(d, s, 30, ko, [], grid) for d in a.fit_days[:1] for s in fit_stocks
                 if s in set(symbols("itch", d)["stock"])]
    tune = pd.concat(pool(eval_task, tune_jobs, a.workers), ignore_index=True)
    tune["c"] = tune["cost"] / (tune["Q"] * tune["w_spread"])
    rule_costs = tune[tune.policy.str.startswith("queue")].groupby("policy")["c"].mean()
    best = rule_costs.idxmin()
    best_rule = next(r for r in grid if P.QueueRule(*r).name == best)
    print("queue rule grid (spread units):", rule_costs.round(4).to_dict(), "-> best", best, flush=True)

    # 4. zero-shot evaluation on the other half, unseen day
    have = set(symbols("itch", a.eval_day)["stock"])
    ev_jobs = [(a.eval_day, s, a.eval_episodes, ko, paths, [best_rule]) for s in eval_stocks if s in have]
    res = pd.concat(pool(eval_task, ev_jobs, a.workers), ignore_index=True)
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
    res.to_parquet(OUT / "eval.parquet", index=False)

    # 5. paired summary vs post-then-cross, clustered by stock
    piv = res.pivot_table(index=["key", "stock"], columns="policy", values="c_spread").reset_index()
    base = "post_then_cross"
    summary = {"fit_episodes": len(eps), "eval_episodes": int(piv.shape[0]), "fit_stocks": len(fit_stocks),
               "eval_stocks": int(res.stock.nunique()), "best_rule": best, "policies": {}}
    for pol in [c for c in piv.columns if c not in ("key", "stock")]:
        d = (piv[pol] - piv[base]).groupby(piv["stock"]).mean()
        ci = cluster_bootstrap(d.to_numpy(), n_boot=4000)
        summary["policies"][pol] = {"mean_cost_spread": float(piv[pol].mean()),
                                    "diff_vs_post_then_cross": ci,
                                    "mean_cost_bps": float(res.loc[res.policy == pol, "c_bps"].mean())}
        print(f"{pol:28s} cost {piv[pol].mean():.4f} spreads; vs post-then-cross {ci['mean']:+.4f} "
              f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]", flush=True)
    summary["provenance"] = provenance(args=vars(a))
    write_json(OUT / "summary.json", summary)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Per-stock policies: the same learner trained on one stock's own training days only.

    python experiments/per_asset.py [--reps R3tick R0] [--days 1 2 5] [--workers 16]

For every validation stock, fits an advantage-FQI policy (the pooled policies' hyperparameters, and a small-data
setting with smaller leaves and fewer rounds) on the
stock's own episodes in valseen_v0 (its 2019 training days), using its first d days for each d in --days, and
scores it on the stock's frozen validation episodes. Paired with the pooled policies' costs on the same episodes
(results/transfer/eval_val_v0.parquet), this measures what a stock's own data buys against what the pool and the
units buy. Writes results/transfer/eval_val_v0_per_asset.parquet (same layout as transfer.py eval) and prints the
paired summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import K, OUT, load_transitions, pool  # noqa: E402

from invex.agents import fqi_adv  # noqa: E402
from invex.env.features import FlowBook, r1_stats  # noqa: E402
from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval import policies as P  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402


SMALL = {"min_child_samples": 20, "num_leaves": 15}  # a small-data setting, so the comparison does not rest on
SMALL_ROUNDS = 150  # trees that cannot split on a few hundred episodes


FINETUNE_FROM = OUT / "models" / "zero_R3tick_s0.pkl"  # the pooled zero-shot policy that fine-tuning starts from


def task(args):
    stock, tr_eps, tr_data, ev_eps, reps, days, ko, finetune, ft = args
    tr_days = sorted(tr_eps["day"].unique())
    models = []
    base = fqi_adv.load_model(FINETUNE_FROM) if finetune else None
    for d in days:
        sel = np.flatnonzero(tr_eps["day"].isin(tr_days[:d]).to_numpy())
        if len(sel) < 50:
            continue
        e = tr_eps.iloc[sel].reset_index(drop=True)
        dd = fqi_adv.subset(tr_data, sel)
        uf = unit_frame(e, ko)
        for rep in ([] if ft.get("no_scratch") else reps):
            c_r0 = float(np.median(uf["R3"]["t"]))
            c = c_r0 if rep.startswith("R0") else 1.0
            m = fqi_adv.fit(dd, e, uf, rep, K, c, threads=1, seed=0, log=lambda s: None)
            models.append(P.FQIPolicy(m, name=f"asset_{rep}_d{d}"))
            m = fqi_adv.fit(dd, e, uf, rep, K, c, n_rounds=SMALL_ROUNDS, threads=1, seed=0, params=SMALL,
                            log=lambda s: None)
            models.append(P.FQIPolicy(m, name=f"assetsmall_{rep}_d{d}"))
        if base is not None:  # the pooled policy, adapted to the stock's own days
            m = fqi_adv.finetune(base, dd, e, uf, n_rounds=ft["rounds"], threads=1, seed=0,
                                 params={"min_child_samples": ft["min_child"], "num_leaves": 15})
            models.append(P.FQIPolicy(m, name=f"finetune{ft['label']}_{base.rep}_d{d}"))
    out = []
    for day, g in ev_eps.groupby("day"):
        book, o, c, tick = book_of("itch", day, stock)
        g = g.reset_index(drop=True).copy()
        for col, v in r1_stats(FlowBook.build(book), g, tick=tick).items():
            g[col] = v
        uf = unit_frame(g, ko)
        out.append(P.evaluate_book(book, g, uf, models, K, venue_of("itch").latency_ns, tick=tick))
    return pd.concat(out, ignore_index=True) if out else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", nargs="+", default=["R3tick", "R0"])
    ap.add_argument("--days", nargs="+", type=int, default=[1, 2, 5])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--finetune", action="store_true", help="also adapt the pooled zero-shot MS+tick policy per stock")
    ap.add_argument("--out", default="val_v0_per_asset")
    ap.add_argument("--ft-rounds", type=int, default=50)
    ap.add_argument("--ft-min-child", type=int, default=20)
    ap.add_argument("--ft-label", default="")
    ap.add_argument("--no-scratch", action="store_true", help="skip the policies trained from scratch")
    a = ap.parse_args(argv)
    ko = load_splits()["ko_constants"]
    tr_eps, tr_data = load_transitions(["valseen_v0"])
    ev = freeze.load("val_v0")
    jobs = []
    for stock, idx in tr_eps.groupby("stock").indices.items():
        g = ev[ev["stock"] == stock]
        if len(g):
            jobs.append((stock, tr_eps.iloc[idx].reset_index(drop=True), fqi_adv.subset(tr_data, idx), g, a.reps,
                         a.days, ko, a.finetune, {"rounds": a.ft_rounds, "min_child": a.ft_min_child,
                                                  "label": a.ft_label, "no_scratch": a.no_scratch}))
    t0 = time.time()
    res = pd.concat(pool(task, jobs, a.workers), ignore_index=True)
    res = res.merge(ev[["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]],
                    on="key")
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res.to_parquet(OUT / f"eval_{a.out}.parquet", index=False)
    write_json(OUT / f"eval_{a.out}.json", {"stocks": len(jobs), "rows": len(res),
                                                    "seconds": round(time.time() - t0),
                                                    "provenance": provenance(args=vars(a))})
    print(f"per-asset: {len(jobs)} stocks, {len(res):,} rows, {time.time() - t0:.0f}s")
    pooled = pd.read_parquet(OUT / "eval_val_v0.parquet") if (OUT / "eval_val_v0.parquet").exists() else \
        pd.read_parquet(OUT / "eval_val_v0_s0.parquet")
    ref = pooled[pooled["policy"].str.match(r"^zero_R3tick_s\d+$")].groupby("key")["c_spread"].mean().rename("ref")
    both = res.join(ref, on="key")
    for pol, g in both.groupby("policy"):
        by = (g["c_spread"] - g["ref"]).groupby(g["stock"]).mean().dropna()
        ci = cluster_bootstrap(by.to_numpy(), n_boot=4000)
        print(f"  {pol:18s} minus pooled zero-shot MS+tick: {ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] "
              f"(mean cost {g['c_spread'].mean():.4f})")


if __name__ == "__main__":
    main()

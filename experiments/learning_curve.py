"""Learning curve of pooled zero-shot policies: how much data from other stocks equals a stock's own days.

    python experiments/learning_curve.py fit  [--sizes 1000 3000 10000 30000 100000] [--reps R3tick R0]
    python experiments/transfer.py eval --episodes val_v0 --tags curve --out val_v0_curve
    python experiments/learning_curve.py analyze

Pooled zero-shot policies are trained on random subsets of the training transitions (train_v0) of the given
sizes and scored on the frozen validation episodes, next to the full-size policies (eval_val_v0) and the
per-stock policies trained on each stock's own training days (eval_val_v0_per_asset). Models are named
curve_<rep>n<size>_s<seed>, so that the representation field carries the size.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import K, OUT, load_transitions  # noqa: E402

from invex.agents import fqi_adv  # noqa: E402
from invex.env.units import unit_frame  # noqa: E402
from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

SIZES = (1000, 3000, 10_000, 30_000, 100_000)


def cmd_fit(a):
    ko = load_splits()["ko_constants"]
    eps, data = load_transitions(["train_v0"])
    for seed in a.seeds:
        for n in a.sizes:
            # seed 0 keeps the original subsets; other seeds draw new subsets and fit with their own seed
            rs = 2000 + n + 100_000 * seed
            keep = np.sort(np.random.default_rng(rs).choice(len(eps), size=min(n, len(eps)), replace=False))
            e = eps.iloc[keep].reset_index(drop=True)
            d = fqi_adv.subset(data, keep)
            uf = unit_frame(e, ko)
            c_r0 = float(np.median(uf["R3"]["t"]))
            for rep in a.reps:
                path = OUT / "models" / f"curve_{rep}n{n}_s{seed}.pkl"
                if path.exists():
                    continue
                t0 = time.time()
                m = fqi_adv.fit(d, e, uf, rep, K, c_r0 if rep.startswith("R0") else 1.0, threads=a.threads,
                                seed=seed, log=lambda s: None)
                m.save(path)
                print(f"curve {rep} n={n:,} seed {seed}: {time.time() - t0:.0f}s", flush=True)


def cmd_analyze(a):
    cur = pd.read_parquet(OUT / "eval_val_v0_curve.parquet")
    full = pd.read_parquet(OUT / "eval_val_v0.parquet")
    own = pd.read_parquet(OUT / "eval_val_v0_per_asset.parquet")
    ref = full[full["policy"].str.match(r"^zero_R3tick_s\d+$")].groupby("key")["c_spread"].mean().rename("ref")
    rows = []
    # every (representation, size): costs averaged over the seeds available (paired on the same episodes)
    pat = re.compile(r"^curve_(?P<rep>[A-Za-z0-9]+?)n(?P<n>\d+)_s(?P<seed>\d+)$")
    m_ = cur["policy"].str.extract(pat)
    cur = cur.assign(rep=m_["rep"], n=pd.to_numeric(m_["n"]), seed=pd.to_numeric(m_["seed"])).dropna(subset=["rep"])
    for (rep, n), g in cur.groupby(["rep", "n"]):
        seeds = sorted(g["seed"].unique().astype(int))
        per_seed = {int(s): float(gs["c_spread"].mean()) for s, gs in g.groupby("seed")}
        mean = g.groupby("key").agg(c_spread=("c_spread", "mean"), stock=("stock", "first")).reset_index()
        rows.append({"rep": rep, "n": int(n), "policy": f"curve_{rep}n{int(n)}", "g": mean, "seeds": seeds,
                     "per_seed": per_seed})
    for rep in ("R3tick", "R0"):
        g = full[full["policy"].str.match(rf"^zero_{rep}_s\d+$")]
        per_seed = {int(p.rsplit("_s", 1)[1]): float(gp["c_spread"].mean()) for p, gp in g.groupby("policy")}
        mean = g.groupby("key").agg(c_spread=("c_spread", "mean"), stock=("stock", "first")).reset_index()
        rows.append({"rep": rep, "n": 376_610, "policy": f"zero_{rep}", "g": mean, "seeds": sorted(per_seed),
                     "per_seed": per_seed})
    for pol, g in own[own["policy"].str.match(r"^assetsmall_R3tick_d\d$")].groupby("policy"):
        rows.append({"rep": "own " + pol.split("_")[-1], "n": 0, "policy": pol, "g": g})
    for pol in ("post_then_cross", "immediate"):
        rows.append({"rep": pol, "n": 0, "policy": pol, "g": full[full["policy"] == pol]})
    out = []
    for r in rows:
        g = r["g"].join(ref, on="key")
        by = (g["c_spread"] - g["ref"]).groupby(g["stock"]).mean().dropna()
        ci = cluster_bootstrap(by.to_numpy(), n_boot=a.n_boot)
        out.append({"rep": r["rep"], "n": r["n"], "cost": float(r["g"]["c_spread"].mean()), "minus_full_ms_tick": ci,
                    "seeds": r.get("seeds", []), "per_seed_cost": r.get("per_seed", {})})
    write_json(OUT / "learning_curve.json", {"points": out, "provenance": provenance(args=vars(a))})
    for o in sorted(out, key=lambda o: (o["rep"], o["n"])):
        c = o["minus_full_ms_tick"]
        print(f"  {o['rep']:10s} n={o['n']:>8,}  cost {o['cost']:.4f}  minus full MS+tick {c['mean']:+.4f} "
              f"[{c['lo']:+.4f}, {c['hi']:+.4f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    f.add_argument("--reps", nargs="+", default=["R3tick", "R0"])
    f.add_argument("--threads", type=int, default=16)
    f.add_argument("--seeds", nargs="+", type=int, default=[0])
    g = sub.add_parser("analyze")
    g.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()

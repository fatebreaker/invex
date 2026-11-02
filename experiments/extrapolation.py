"""Extrapolation on real stocks: train in the middle of one scale's range, test outside it (the analogue of PC1x).

    python experiments/extrapolation.py fit     --stat activity [--reps ...] [--seeds 0]
    python experiments/transfer.py eval --episodes val_v0 --tags xactivity xrefactivity --out val_v0_xactivity
    python experiments/extrapolation.py analyze --stat activity

A scale statistic is computed per stock as the median over its episodes' trailing windows:

    activity    trades per second            (the time scale)
    tradesize   shares per trade             (the size scale)
    spread      mean spread in dollars       (the price scale)

Two policies per representation, trained on the training transitions (train_v0):

    restricted  "x<stat>"     only training stocks whose statistic lies between the 25th and 75th percentile of
                              the training stocks
    reference   "xref<stat>"  a random sample of all training stocks' episodes, of the same size

Both are scored on the frozen validation episodes. The extrapolation gap of a validation stock is the restricted
policy's cost minus the reference policy's, in spreads per share; units that absorb the scale should show no
gap for stocks outside the training range, raw units should. Writes results/transfer/xtrap_<stat>.json and
paper/numbers_xtrap_<stat>.tex.
"""

from __future__ import annotations

import argparse
import json
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
from invex.eval import freeze  # noqa: E402
from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.paths import DATA, REPO  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

STATS = ("activity", "tradesize", "spread")
REPS = ("R0", "R0r", "R0desc", "R1", "R2", "R3", "R3s", "R3d", "R2tick", "R3tick", "Rpi", "Rdim")
NAME = re.compile(r"^(?P<tag>x[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def stat_of(eps: pd.DataFrame, stat: str) -> pd.Series:
    """The statistic per stock (median over the stock's episodes)."""
    if stat == "activity":
        x = eps["w_trades"] / eps["w_seconds"]
    elif stat == "tradesize":
        x = eps["w_volume"] / eps["w_trades"].clip(lower=1)
    elif stat == "spread":
        x = eps["w_spread"] / (eps["px_scale"] if "px_scale" in eps else 1e4)
    else:
        raise ValueError(stat)
    return x.groupby(eps["stock"]).median()


def cmd_fit(a):
    ko = load_splits()["ko_constants"]
    eps, data = load_transitions(["train_v0"])
    s = stat_of(eps, a.stat)
    lo, hi = s.quantile(0.25), s.quantile(0.75)
    inside = set(s[(s >= lo) & (s <= hi)].index)
    keep_r = np.flatnonzero(eps["stock"].isin(inside).to_numpy())
    meta = {"stat": a.stat, "q25": float(lo), "q75": float(hi), "stocks_inside": len(inside),
            "episodes": int(len(keep_r)), "provenance": provenance(args=vars(a))}
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"xtrap_{a.stat}_fit.json", meta)
    print(f"{a.stat}: training range [{lo:.4g}, {hi:.4g}], {len(inside)} stocks, {len(keep_r):,} episodes", flush=True)
    for seed in a.seeds:
        keep_f = np.sort(np.random.default_rng(1000 + seed).choice(len(eps), size=len(keep_r), replace=False))
        for tag, keep in ((f"x{a.stat}", keep_r), (f"xref{a.stat}", keep_f)):
            e = eps.iloc[keep].reset_index(drop=True)
            d = fqi_adv.subset(data, keep)
            uf = unit_frame(e, ko)
            c_r0 = float(np.median(uf["R3"]["t"]))
            for rep in a.reps:
                path = OUT / "models" / f"{tag}_{rep}_s{seed}.pkl"
                if path.exists():
                    continue
                t0 = time.time()
                m = fqi_adv.fit(d, e, uf, rep, K, c_r0 if rep.startswith("R0") else 1.0, n_rounds=a.rounds,
                                threads=a.threads, seed=seed, log=lambda s_: None)
                m.save(path)
                print(f"{tag} {rep} seed {seed}: {len(e):,} episodes, {time.time() - t0:.0f}s", flush=True)


def mname(*parts) -> str:
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def cmd_analyze(a):
    fit = json.loads((OUT / f"xtrap_{a.stat}_fit.json").read_text())
    ev = a.eval or f"val_v0_x{a.stat}"
    res = pd.read_parquet(OUT / f"eval_{ev}.parquet")
    if "excluded" in res:
        res = res[~res["excluded"]]
    m = res["policy"].str.extract(NAME)
    res = res[m["tag"].notna()].assign(tag=m["tag"], rep=m["rep"])
    per = res.groupby(["key", "stock", "tag", "rep"])["c_spread"].mean().unstack(["tag", "rep"])
    name = json.loads((OUT / f"eval_{ev}.json").read_text())["episodes"]
    sealed = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"]
    eps = freeze.load(name, unseal=sealed, reason=f"extrapolation analysis of {ev}" if sealed else "")
    s = stat_of(eps, a.stat)
    where = pd.Series(np.where(s < fit["q25"], "below", np.where(s > fit["q75"], "above", "inside")), index=s.index)
    stocks = per.index.get_level_values("stock")
    out = {"stat": a.stat, "training_range": [fit["q25"], fit["q75"]], "stocks": where.value_counts().to_dict(),
           "gap": {}, "provenance": provenance(args=vars(a))}
    macros = {}
    for rep in REPS:
        r, f = (f"x{a.stat}", rep), (f"xref{a.stat}", rep)
        if r not in per.columns or f not in per.columns:
            continue
        by = (per[r] - per[f]).groupby(stocks).mean().dropna()
        w = where.reindex(by.index)
        row = {}
        for g, sel in (("outside", w != "inside"), ("inside", w == "inside"), ("below", w == "below"),
                       ("above", w == "above")):
            if sel.sum() >= 5:
                row[g] = cluster_bootstrap(by[sel].to_numpy(), n_boot=a.n_boot)
                stem = mname(a.macro_prefix, a.stat, rep, g)
                macros[stem] = f"{row[g]['mean']:+.3f}"
                macros[stem + "Lo"], macros[stem + "Hi"] = f"{row[g]['lo']:+.3f}", f"{row[g]['hi']:+.3f}"
        out["gap"][rep] = row
    # D-hat against the restricted training stocks (probes on training days): does it predict the extrapolation gap?
    dh_path = OUT / "dhat_train_days.parquet"
    if dh_path.exists():
        from scipy.stats import spearmanr

        from invex.env.features import BASE_SYSTEM
        from invex.eval.universe import load_splits

        split = load_splits()["split"]
        dh = pd.read_parquet(dh_path)
        perj = dh.groupby(["system", "stock", "policy"])[["sum", "count"]].sum()
        perj = (perj["sum"] / perj["count"]).rename("j").reset_index()
        tr_inside = {st for st in perj["stock"].unique() if split.get(st) == "train"}
        tr_eps_stat = stat_of(pd.read_parquet(DATA / "transitions" / "train_v0" / "eps.parquet",
                                              columns=["stock", "w_trades", "w_seconds", "w_volume", "w_spread"]), a.stat)
        tr_inside &= set(tr_eps_stat[(tr_eps_stat >= fit["q25"]) & (tr_eps_stat <= fit["q75"])].index)
        ref = perj[perj["stock"].isin(tr_inside)].groupby(["system", "policy"])["j"].mean().rename("ref")
        perj = perj.join(ref, on=["system", "policy"])
        dhat = (perj["j"] - perj["ref"]).abs().groupby([perj["system"], perj["stock"]]).max()
        out["dhat_spearman"] = {}
        for rep_, row in out["gap"].items():
            sys_ = BASE_SYSTEM.get(rep_)
            if sys_ not in dhat.index.get_level_values(0):
                continue
            r_, f_ = (f"x{a.stat}", rep_), (f"xref{a.stat}", rep_)
            g = (per[r_] - per[f_]).groupby(stocks).mean().dropna()
            x = dhat.loc[sys_].reindex(g.index)
            ok = x.notna()
            rho = spearmanr(x[ok], g[ok]).correlation
            out["dhat_spearman"][rep_] = {"rho": float(rho), "n": int(ok.sum())}
            macros[mname(a.macro_prefix, a.stat, rep_, "dhat rho")] = f"{rho:+.2f}"
    tag = "" if not a.eval else f"_{a.eval}"
    write_json(OUT / f"xtrap_{a.stat}{tag}.json", out)
    with open(REPO / "paper" / f"numbers_xtrap_{a.stat}{tag}.tex", "w") as fh:
        fh.write(f"% generated by experiments/extrapolation.py analyze --stat {a.stat} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{a.stat}: training range [{fit['q25']:.4g}, {fit['q75']:.4g}]; validation stocks {out['stocks']}")
    for rep, row in out["gap"].items():
        print(f"  {rep:7s} " + "; ".join(f"{g} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for g, v in row.items()))
    for rep, r in out.get("dhat_spearman", {}).items():
        print(f"  D-hat vs gap, {rep:7s} Spearman {r['rho']:+.3f} (n={r['n']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--stat", required=True, choices=STATS)
    f.add_argument("--reps", nargs="+", default=list(REPS))
    f.add_argument("--seeds", nargs="+", type=int, default=[0])
    f.add_argument("--rounds", type=int, default=300)
    f.add_argument("--threads", type=int, default=32)
    g = sub.add_parser("analyze")
    g.add_argument("--stat", required=True, choices=STATS)
    g.add_argument("--n-boot", type=int, default=10_000)
    g.add_argument("--eval", default="", help="evaluation to analyse (default: val_v0_x<stat>)")
    g.add_argument("--macro-prefix", default="Xtrap")
    a = ap.parse_args(argv)
    {"fit": cmd_fit, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()

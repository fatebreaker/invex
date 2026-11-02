"""Analysis of a transfer evaluation (experiments/transfer.py eval): contrasts, transfer gaps, macros.

    python experiments/analyze_transfer.py --eval val_v0 [--ref R3] [--macro-prefix Val]

Policies are named <tag>_<rep>_s<seed> (tags: zero, seen) plus the baselines. Per episode, each
representation's cost is averaged over seeds; per stock, paired differences are averaged over its episodes;
confidence intervals come from a cluster bootstrap over stocks stratified by relative-tick tercile (frozen
cuts). Reports, overall and by tercile: every zero-shot representation against the reference, the transfer
gap (zero-shot minus seen, same representation), the gain over post-then-cross and, with --placebo, the cost of
giving each episode another stock's units (power check). Episodes the evaluation
flagged as excluded (they start in or span a halt, pause, auction guard or feed outage) are dropped. Writes
results/transfer/analysis_<eval>.json and paper/numbers_<eval>.tex.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from invex.env.units import unit_frame
from invex.eval import freeze
from invex.eval.stats import cluster_bootstrap
from invex.eval.universe import load_splits
from invex.paths import REPO, RESULTS
from invex.provenance import provenance, write_json

TERCILES = ("small", "mid", "large")
NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")


def macro_name(*parts) -> str:
    """LaTeX command names may only contain letters: digits are spelled out."""
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
             "8": "Eight", "9": "Nine"}
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(words.get(c, c) for c in s if c.isalnum())


def fmt(x: float, d: int = 3) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", required=True, help="evaluation name (results/transfer/eval_<name>.parquet)")
    ap.add_argument("--episodes", default="", help="episode table (defaults to the eval's)")
    ap.add_argument("--ref", default="R3")
    ap.add_argument("--macro-prefix", default="")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--placebo", default="", help="a shuffled-units evaluation of the same episodes (eval name)")
    ap.add_argument("--seen-tag", default="seen", help="tag of the policies the transfer gap compares with")
    a = ap.parse_args(argv)
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    n_excluded = int(res.loc[res["excluded"], "key"].nunique()) if "excluded" in res else 0
    if n_excluded:  # halts, pauses, auction guards, feed outages: the same episodes leave every policy
        res = res[~res["excluded"]]
    sp = load_splits()

    # seed-averaged cost (spread units) per (episode, tag, representation)
    m = res["policy"].str.extract(NAME)
    rl = res[m["tag"].notna()].assign(tag=m["tag"], rep=m["rep"])
    per = rl.groupby(["key", "stock", "tag", "rep"])["c_spread"].mean().unstack(["tag", "rep"])
    base = res[m["tag"].isna()].pivot_table(index=["key", "stock"], columns="policy", values="c_spread")
    base.columns = pd.MultiIndex.from_tuples([("base", c) for c in base.columns], names=per.columns.names)
    per = per.join(base)

    # relative-tick tercile of each stock (median over its episodes' windows, frozen cuts)
    eps = freeze.load(a.episodes or meta["episodes"], unseal=True, reason=f"analysis of {a.eval}") \
        if (a.episodes or meta["episodes"]).startswith("stage") else freeze.load(a.episodes or meta["episodes"])
    uf = unit_frame(eps, sp["ko_constants"])
    eta = pd.Series(uf["_eta_sigma"].to_numpy(), index=eps["stock"]).groupby(level=0).median()
    terc = pd.Series(np.digitize(np.log(eta.to_numpy()), np.log(sp["eta_sigma_tercile_cuts"])), index=eta.index)

    stocks = per.index.get_level_values("stock")
    reps = sorted({r for t, r in per.columns if t == "zero"})
    out = {"eval": a.eval, "ref": a.ref, "episodes": int(len(per)), "excluded": n_excluded,
           "stocks": int(stocks.nunique()), "contrasts": {},
           "transfer_gap": {}, "vs_post_then_cross": {}, "placebo": {}, "mean_cost": {}}
    macros = {}

    def summarize(diff: pd.Series, key: str, group: str):
        by_stock = diff.groupby(stocks).mean().dropna()
        st = terc.reindex(by_stock.index).to_numpy()
        res_ = {"all": cluster_bootstrap(by_stock.to_numpy(), st, n_boot=a.n_boot)}
        for t in range(3):
            sel = st == t
            if sel.sum() >= 5:
                res_[TERCILES[t]] = cluster_bootstrap(by_stock.to_numpy()[sel], n_boot=a.n_boot)
        out[group][key] = res_
        for k, ci in res_.items():
            stem = macro_name(a.macro_prefix, group.replace("_", " ").title().replace(" ", ""), key, k)
            macros[stem] = fmt(ci["mean"])
            macros[stem + "Lo"] = fmt(ci["lo"])
            macros[stem + "Hi"] = fmt(ci["hi"])

    for rep in reps:
        z = per[("zero", rep)]
        out["mean_cost"][rep] = float(z.mean())
        if rep != a.ref and ("zero", a.ref) in per.columns:
            summarize(z - per[("zero", a.ref)], f"{rep}-{a.ref}", "contrasts")
        if (a.seen_tag, rep) in per.columns:
            summarize(z - per[(a.seen_tag, rep)], rep, "transfer_gap")
        if ("base", "post_then_cross") in per.columns:
            summarize(z - per[("base", "post_then_cross")], rep, "vs_post_then_cross")
    if a.placebo:  # cost with another stock's units minus cost with the stock's own, same models and episodes
        pl = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.placebo}.parquet")
        pl = pl[pl["key"].isin(res["key"])]
        both = res.merge(pl[["key", "policy", "c_spread"]], on=["key", "policy"], suffixes=("", "_pl"))
        mp = both["policy"].str.extract(NAME)
        both = both[mp["tag"] == "zero"].assign(rep=mp["rep"])
        d = (both["c_spread_pl"] - both["c_spread"]).groupby([both["key"], both["stock"], both["rep"]]).mean().unstack("rep")
        stocks_pl = d.index.get_level_values("stock")
        for rep in d.columns:
            by_stock = d[rep].groupby(stocks_pl).mean().dropna()
            st = terc.reindex(by_stock.index).to_numpy()
            ci = cluster_bootstrap(by_stock.to_numpy(), st, n_boot=a.n_boot)
            out["placebo"][rep] = {"all": ci}
            stem = macro_name(a.macro_prefix, "Placebo", rep, "all")
            macros[stem], macros[stem + "Lo"], macros[stem + "Hi"] = fmt(ci["mean"]), fmt(ci["lo"]), fmt(ci["hi"])
    for b in ("immediate", "post_then_cross"):
        if ("base", b) in per.columns:
            out["mean_cost"][b] = float(per[("base", b)].mean())
    out["provenance"] = provenance(args=vars(a))
    write_json(RESULTS / "transfer" / f"analysis_{a.eval}.json", out)
    with open(REPO / "paper" / f"numbers_{a.eval}.tex", "w") as fh:
        fh.write(f"% generated by experiments/analyze_transfer.py --eval {a.eval} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{a.eval}: {out['episodes']:,} episodes, {out['stocks']} stocks")
    print("mean cost (spreads):", {k: round(v, 4) for k, v in out["mean_cost"].items()})
    for group in ("contrasts", "transfer_gap", "vs_post_then_cross", "placebo"):
        for key, r in out[group].items():
            line = "; ".join(f"{k} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for k, v in r.items())
            print(f"{group:20s} {key:14s} {line}")


if __name__ == "__main__":
    main()

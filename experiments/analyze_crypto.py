"""Equities to crypto: equity-trained policies, zero-shot on Coinbase pairs, against crypto-trained ones.

    python experiments/analyze_crypto.py --eval crypto_dev [--seen-tag cbseen] [--macro-prefix CryptoDev]

Policies: baselines, equity-trained zero-shot policies (tag zero) and policies trained on earlier Coinbase days
(tag cbseen), all named <tag>_<rep>_s<seed>. Costs are paired per episode and averaged over seeds, in two units:
the pair's mean spread (as for stocks) and its volatility per trade. The second matters on crypto: by 2023 the
largest pairs trade at a one-tick spread that is tiny next to their volatility, so spread units let them dominate
any average. Pairs are few, so intervals come from a two-way bootstrap over pairs and days. Reports, for each
representation: the zero-shot contrast with microstructure units, the cross-venue transfer gap (equity-trained
minus crypto-trained, same representation) and the gain over each baseline. Writes
results/transfer/crypto_<eval>.json and paper/numbers_crypto_<eval>.tex.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from invex.env.units import unit_frame
from invex.eval import freeze
from invex.eval.stats import pigeonhole_bootstrap
from invex.eval.universe import load_splits
from invex.paths import REPO, RESULTS
from invex.provenance import provenance, write_json

NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def mname(*parts) -> str:
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def two_way(diff: pd.Series, pair: pd.Series, day: pd.Series, n_boot: int) -> dict:
    """Mean of diff with every pair weighted equally (and each of its days equally), with a pairs x days bootstrap
    of the same weighted mean: cell (pair, day) enters with its mean and weight 1 / (the pair's number of days)."""
    cm = diff.groupby([pair, day]).mean().unstack()
    w = cm.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    ci = pigeonhole_bootstrap((cm.fillna(0.0) * w).to_numpy(), w.to_numpy(), n_boot=n_boot)
    by_pair = (cm.fillna(0.0) * w).sum(axis=1)
    return {"mean": ci["mean"], "median_pair": float(by_pair.median()), "lo": ci["lo"], "hi": ci["hi"],
            "wins": int((by_pair < 0).sum()), "pairs": int(by_pair.shape[0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--seen-tag", default="cbseen")
    ap.add_argument("--ref", default="R3")
    ap.add_argument("--macro-prefix", default="Crypto")
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args(argv)
    import json

    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    name = a.episodes or meta["episodes"]
    eps = freeze.load(name, unseal=True, reason=f"analysis of {a.eval}") if json.loads(
        (REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"] else freeze.load(name)
    uf = unit_frame(eps, load_splits()["ko_constants"])
    eps = eps.assign(sigma_trade=uf["_sigma_trade"].to_numpy())
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    if "excluded" in res:
        res = res[~res["excluded"]]
    # evaluations written after prereg-v1 also store sigma_trade; use the episode table's (the same values)
    res = res.drop(columns=["sigma_trade"], errors="ignore").merge(eps[["key", "sigma_trade"]], on="key")
    res["c_sigma"] = res["cost"] / (res["Q"] * res["sigma_trade"])
    m = res["policy"].str.extract(NAME)
    res["tag"] = m["tag"].fillna("base")
    res["rep"] = m["rep"].fillna(res["policy"])
    out = {"eval": a.eval, "episodes": int(res["key"].nunique()), "pairs": int(res["stock"].nunique()),
           "days": int(res["day"].nunique()), "units": {}, "provenance": provenance(args=vars(a))}
    macros = {}
    for unit in ("c_spread", "c_sigma"):
        per = res.groupby(["key", "stock", "day", "tag", "rep"])[unit].mean().unstack(["tag", "rep"])
        pair = pd.Series(per.index.get_level_values("stock"), index=per.index)
        day = pd.Series(per.index.get_level_values("day"), index=per.index)
        u = {"mean_cost": {}, "contrast": {}, "venue_gap": {}, "vs_immediate": {}, "vs_post_then_cross": {}}
        for (tag, rep) in per.columns:
            u["mean_cost"][f"{tag}_{rep}"] = float(per[(tag, rep)].groupby(pair).mean().mean())
        reps = sorted(r for t, r in per.columns if t == "zero")
        for rep in reps:
            z = per[("zero", rep)]
            blocks = {"vs_immediate": per[("base", "immediate")], "vs_post_then_cross": per[("base", "post_then_cross")]}
            if rep != a.ref and ("zero", a.ref) in per.columns:
                blocks["contrast"] = per[("zero", a.ref)]
            if (a.seen_tag, rep) in per.columns:
                blocks["venue_gap"] = per[(a.seen_tag, rep)]
            for k, other in blocks.items():
                r = two_way(z - other, pair, day, a.n_boot)
                u[k][rep] = r
                stem = mname(a.macro_prefix, "sigma" if unit == "c_sigma" else "spread", k.replace("_", " "), rep)
                macros[stem] = f"{r['mean']:+.3f}"
                macros[stem + "Lo"], macros[stem + "Hi"] = f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
                macros[stem + "Wins"] = str(r["wins"])
        out["units"][unit] = u
    macros[mname(a.macro_prefix, "pairs")] = str(out["pairs"])
    macros[mname(a.macro_prefix, "days")] = str(out["days"])
    macros[mname(a.macro_prefix, "episodes")] = f"{out['episodes']:,}".replace(",", "{,}")
    write_json(RESULTS / "transfer" / f"crypto_{a.eval}.json", out)
    with open(REPO / "paper" / f"numbers_crypto_{a.eval}.tex", "w") as fh:
        fh.write(f"% generated by experiments/analyze_crypto.py --eval {a.eval} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{a.eval}: {out['episodes']:,} episodes, {out['pairs']} pairs, {out['days']} days")
    for unit, u in out["units"].items():
        print(f"[{unit}] mean cost (pairs equally weighted):",
              {k: round(v, 3) for k, v in sorted(u["mean_cost"].items(), key=lambda kv: kv[1])})
        for k in ("contrast", "venue_gap", "vs_immediate"):
            for rep, r in u[k].items():
                print(f"  {k:12s} {rep:7s} {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] median-pair {r['median_pair']:+.3f}"
                      f" wins {r['wins']}/{r['pairs']}")


if __name__ == "__main__":
    main()

"""D-hat (the RL-free transfer diagnostic) against realised transfer: H6 on validation.

    python experiments/analyze_dhat.py --dhat train_days --eval val_v0 [--macro-prefix Dhat]

D-hat_u(b) = max over the probe rules of |mean normalised cost on stock b - mean over training stocks| in unit
system u, from the probes on the training days only (experiments/transfer.py dhat; invex.eval.dhat). It is set
against each validation stock's realised zero-shot loss for the representations built on u:

    transfer gap   zero-shot minus seen (same representation)
    contrast       zero-shot minus zero-shot in microstructure units (how much worse u's policy does than MS's)

Reports Spearman correlations across validation stocks with stock-bootstrap CIs, and D-hat's level per system
(median over validation stocks, and over training stocks as an in-sample reference), also on a common scale: each
system's D-hat times the stock's own median ratio of that system's price unit to its mean spread (the macros). Writes
results/transfer/dhat_analysis_<eval>.json and paper/numbers_dhat.tex.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from invex.eval.universe import load_splits
from invex.paths import DATA, REPO, RESULTS
from invex.provenance import provenance, write_json

NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
SYSTEM_OF = {"R0": "R0", "R2": "R2", "R2tick": "R2", "R3": "R3", "R3tick": "R3", "R3s": "R3s", "R3d": "R3d"}
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def mname(*parts) -> str:
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def dhat_table(name: str, split: dict) -> pd.DataFrame:
    """D-hat per (system, stock), with each stock's split."""
    d = pd.read_parquet(RESULTS / "transfer" / f"dhat_{name}.parquet")
    per = d.groupby(["system", "stock", "policy"])[["sum", "count"]].sum()
    per = (per["sum"] / per["count"]).rename("j").reset_index()
    per["split"] = per["stock"].map(split).fillna("test")
    ref = per[per["split"] == "train"].groupby(["system", "policy"])["j"].mean().rename("ref")
    per = per.join(ref, on=["system", "policy"])
    per["dev"] = (per["j"] - per["ref"]).abs()
    out = per.groupby(["system", "stock"]).agg(dhat=("dev", "max"), split=("split", "first")).reset_index()
    return out


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int = 0) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i]).correlation)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return {"rho": rho, "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dhat", default="train_days")
    ap.add_argument("--eval", default="val_v0")
    ap.add_argument("--macro-prefix", default="Dhat")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    split = load_splits()["split"]
    dh = dhat_table(a.dhat, split)
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    if "excluded" in res:
        res = res[~res["excluded"]]
    m = res["policy"].str.extract(NAME)
    res = res[m["tag"].notna()].assign(tag=m["tag"], rep=m["rep"])
    per = res.groupby(["stock", "tag", "rep"])["c_spread"].mean().unstack(["tag", "rep"])
    out = {"dhat": a.dhat, "eval": a.eval, "level": {}, "tg": {}, "contrast": {}, "provenance": provenance(args=vars(a))}
    macros: dict = {}
    # a common scale: each system's D-hat times the stock's own (median) ratio of that system's price unit to its spread
    from invex.env.units import unit_frame

    ko = load_splits()["ko_constants"]
    ratios = []
    for name in ("train_v0", "valseen_v0"):
        e = pd.read_parquet(DATA / "transitions" / name / "eps.parquet")
        uf = unit_frame(e, ko)
        r = pd.DataFrame({s_: uf[s_]["p"].to_numpy() / uf["R3"]["p"].to_numpy() for s_ in ("R0", "R2", "R3", "R3s", "R3d")})
        ratios.append(r.groupby(e["stock"].to_numpy()).median())
    ratio = pd.concat(ratios).groupby(level=0).median()
    dh["dhat_spreads"] = [d * ratio.at[st, sy] if st in ratio.index else np.nan
                          for d, st, sy in zip(dh["dhat"], dh["stock"], dh["system"])]
    for system, g in dh.groupby("system"):
        out["level"][system] = {"val_median": float(g.loc[g["split"] == "val", "dhat"].median()),
                                "train_median": float(g.loc[g["split"] == "train", "dhat"].median()),
                                "val_median_spreads": float(g.loc[g["split"] == "val", "dhat_spreads"].median())}
        macros[mname(a.macro_prefix, "level", system)] = f"{out['level'][system]['val_median_spreads']:.2f}"
    for rep, system in SYSTEM_OF.items():
        d = dh[(dh["system"] == system)].set_index("stock")["dhat"]
        for k, target in (("tg", ("seen", rep)), ("contrast", ("zero", "R3"))):
            if ("zero", rep) not in per.columns or target not in per.columns or (k == "contrast" and rep == "R3"):
                continue
            y = (per[("zero", rep)] - per[target]).dropna()
            x = d.reindex(y.index)
            ci = spearman_ci(x.to_numpy(float), y.to_numpy(float), a.n_boot)
            out[k][rep] = ci
            stem = mname(a.macro_prefix, k, rep)
            macros[stem] = f"{ci['rho']:+.2f}"
            macros[stem + "Lo"], macros[stem + "Hi"] = f"{ci['lo']:+.2f}", f"{ci['hi']:+.2f}"
    write_json(RESULTS / "transfer" / f"dhat_analysis_{a.eval}.json", out)
    with open(REPO / "paper" / "numbers_dhat.tex", "w") as fh:
        fh.write(f"% generated by experiments/analyze_dhat.py --eval {a.eval} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("D-hat level (median):", {s: {k: round(v, 3) for k, v in r.items()} for s, r in out["level"].items()})
    for k in ("tg", "contrast"):
        for rep, ci in out[k].items():
            print(f"  Spearman(D-hat, {k:8s}) {rep:7s} {ci['rho']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] (n={ci['n']})")


if __name__ == "__main__":
    main()

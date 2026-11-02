"""The equal-information arm and the parity checks of the TMLR review (exploratory): validation, and the sealed
stages from their stored evaluations.

    python experiments/info_arm.py                 # validation
    python experiments/info_arm.py --stage stage1  # a sealed stage (stage1, stage2), post hoc, from its stored evaluations

Paired cost differences on the validation table val_v0 (spreads per share, seeds averaged within an episode, every
stock weighted equally as in the registered stock analysis, cluster bootstrap over stocks stratified by relative-tick
tercile, 95% intervals), between policies stored in several evaluations of that table:
  * R0info (raw features plus the window statistics of the MS, MS-sigma and KO units, cost in spreads) against MS+tick (R3tick), raw features with
    the cost in spreads (R0r) and raw units (R0), for trees, the neural learner and PPO;
  * capacity: trees with 1,000 rounds (tunea), 255 leaves (tuneb) or both (tunec), for R3tick, R0, R0r and R0info;
  * PPO with its value targets rescaled (ppovs*), at three learning rates, for R0 and R3tick.
Writes results/transfer/info_arm.json and paper/numbers_info_arm.tex (macros prefixed Info).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402

EVALS = ["val_v0", "val_v0_rev", "val_v0_ppo", "val_v0_info", "val_v0_tune", "val_v0_tunec", "val_v0_ppotune", "val_v0_peg"]
NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
# (label, a, b): a - b, each "<tag>:<rep>"
CONTRASTS = [
    ("TreesInfoVsMstick", "zero:R0info", "zero:R3tick"), ("TreesInfoVsRawr", "zero:R0info", "zero:R0r"),
    ("TreesInfoVsRaw", "zero:R0info", "zero:R0"), ("TreesRawrVsMstick", "zero:R0r", "zero:R3tick"),
    ("TreesRawVsMstick", "zero:R0", "zero:R3tick"),
    ("NnInfoVsMstick", "zeronn:R0info", "zeronn:R3tick"), ("NnInfoVsRawr", "zeronn:R0info", "zeronn:R0r"),
    ("NnInfoVsRaw", "zeronn:R0info", "zeronn:R0"), ("NnRawrVsMstick", "zeronn:R0r", "zeronn:R3tick"),
    ("NnRawVsMstick", "zeronn:R0", "zeronn:R3tick"),
    ("PpoInfoVsMstick", "zeroppo:R0info", "zeroppo:R3tick"), ("PpoInfoVsRaw", "zeroppo:R0info", "zeroppo:R0"),
    ("PpoRawVsMstick", "zeroppo:R0", "zeroppo:R3tick"),
]
# each arm at its best capacity on validation: raw units with 255 leaves, MS+tick as registered (the larger settings
# made it worse)
CONTRASTS += [("TuneBestRawVsMstick", "tuneb:R0", "zero:R3tick")]
CONTRASTS += [("PegVsMstick", "base:peg_then_cross", "zero:R3tick"),
              ("PegVsPost", "base:peg_then_cross", "base:post_then_cross"),
              ("MstickVsPost", "zero:R3tick", "base:post_then_cross"),
              ("PegVsImmediate", "base:peg_then_cross", "base:immediate")]
for tag, label in (("tunea", "Rounds"), ("tuneb", "Leaves"), ("tunec", "Both")):
    CONTRASTS += [(f"Tune{label}RawVsMstick", f"{tag}:R0", f"{tag}:R3tick"),
                  (f"Tune{label}RawrVsMstick", f"{tag}:R0r", f"{tag}:R3tick"),
                  (f"Tune{label}InfoVsMstick", f"{tag}:R0info", f"{tag}:R3tick"),
                  (f"Tune{label}MstickVsBase", f"{tag}:R3tick", "zero:R3tick"),
                  (f"Tune{label}RawVsBase", f"{tag}:R0", "zero:R0"),
                  (f"Tune{label}RawrVsBase", f"{tag}:R0r", "zero:R0r"),
                  (f"Tune{label}InfoVsBase", f"{tag}:R0info", "zero:R0info")]
for tag, label in (("ppovs", "Mid"), ("ppovslo", "Low"), ("ppovshi", "High")):
    CONTRASTS += [(f"PpoVs{label}RawVsMstick", f"{tag}:R0", f"{tag}:R3tick"),
                  (f"PpoVs{label}MstickVsBase", f"{tag}:R3tick", "zeroppo:R3tick"),
                  (f"PpoVs{label}RawVsBase", f"{tag}:R0", "zeroppo:R0")]


def costs(evals) -> pd.DataFrame:
    """Per-episode cost (spreads), seeds averaged, one column per '<tag>:<rep>' found in the evaluations."""
    cols = {}
    stock = None
    for ev in evals:
        p = RESULTS / "transfer" / f"eval_{ev}.parquet"
        if not p.exists():
            continue
        res = pd.read_parquet(p, columns=["key", "policy", "stock", "c_spread", "excluded"])
        res = res[~res["excluded"]]
        m = res["policy"].str.extract(NAME)
        res = res.assign(col=(m["tag"] + ":" + m["rep"]).fillna("base:" + res["policy"]))
        for col, g in res.groupby("col"):
            if col not in cols:
                cols[col] = g.groupby("key")["c_spread"].mean()
        s = res.drop_duplicates("key").set_index("key")["stock"]
        stock = s if stock is None else stock.combine_first(s)
    df = pd.DataFrame(cols)
    df["stock"] = stock.reindex(df.index)
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="", help="a sealed stage (stage1, stage2): its stored evaluations, post hoc")
    args = ap.parse_args()
    evals = EVALS if not args.stage else [args.stage, args.stage + "_rev", args.stage + "_info"]
    df = costs(evals)
    sp = load_splits()
    cuts = np.log(sp["eta_sigma_tercile_cuts"])
    # the stock's relative-tick tercile, as in the registered analysis (median over its episodes of eta_sigma)
    from invex.env.units import unit_frame  # noqa: E402
    from invex.eval import freeze  # noqa: E402
    if args.stage:  # the relative tick as stored by the evaluation (identical to the episode table's; checked on
        # validation), so the sealed table is not loaded again
        r = pd.read_parquet(RESULTS / "transfer" / f"eval_{args.stage}_info.parquet", columns=["key", "stock", "eta"])
        r = r.drop_duplicates("key")
        eta = r.groupby("stock")["eta"].median()
    else:
        full = freeze.load("val_v0")
        uf = unit_frame(full, sp["ko_constants"])
        eta = pd.Series(uf["_eta_sigma"].to_numpy(), index=full["stock"].to_numpy()).groupby(level=0).median()
    out, m = {}, {}
    for label, a, b in CONTRASTS:
        if a not in df or b not in df:
            continue
        d = (df[a] - df[b]).dropna()
        by_stock = d.groupby(df.loc[d.index, "stock"]).mean()
        strata = np.digitize(np.log(eta.reindex(by_stock.index).to_numpy()), cuts)
        r = cluster_bootstrap(by_stock.to_numpy(), strata, n_boot=10_000, level=0.95)
        out[label] = {"a": a, "b": b, **{k: float(v) for k, v in r.items() if k in ("mean", "lo", "hi", "se")},
                      "stocks": int(len(by_stock)), "episodes": int(len(d)), "episode_mean": float(d.mean())}
        pre = "Info" + {"stage1": "StageOne", "stage2": "StageTwo"}.get(args.stage, "")
        m[f"{pre}{label}"] = f"{r['mean']:+.3f}"
        m[f"{pre}{label}Lo"] = f"{r['lo']:+.3f}"
        m[f"{pre}{label}Hi"] = f"{r['hi']:+.3f}"
        print(f"{label:28s} {a:>16s} - {b:<16s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"({len(by_stock)} stocks; by episode {d.mean():+.4f})")
    for col in sorted(c for c in df.columns if c != "stock"):  # mean cost of each arm, stocks weighted equally
        v = df[col].dropna()
        out.setdefault("costs", {})[col] = float(v.groupby(df.loc[v.index, "stock"]).mean().mean())
    sfx = f"_{args.stage}" if args.stage else ""
    (RESULTS / "transfer" / f"info_arm{sfx}.json").write_text(json.dumps(out, indent=1))
    lines = [f"% generated by experiments/info_arm.py {'--stage ' + args.stage if args.stage else ''} (exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / f"numbers_info_arm{sfx}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

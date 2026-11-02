"""Baselines on validation: tuned queue rules, per-stock policies and the learning curve, against pooled zero-shot.

    python experiments/analyze_baselines.py

Reads the rule grids (results/rules/grid_{val_own,train_pool,val_v0}.parquet), the per-stock policies
(eval_val_v0_per_asset) and the learning curve (results/transfer/learning_curve.json), pairs everything with the
pooled zero-shot MS+tick policy (eval_val_v0, seeds averaged) on the same validation episodes, and writes
paper/numbers_baselines.tex (and results/transfer/baselines.json).

  pooled rule      the (theta, iota) with the lowest mean cost over the training stocks' episodes
  per-stock rule   for each validation stock, the rule with the lowest mean cost on its own training days
"""

from __future__ import annotations

import json

import pandas as pd

from invex.eval.stats import cluster_bootstrap
from invex.paths import REPO, RESULTS
from invex.provenance import provenance, write_json

R = RESULTS / "rules"
T = RESULTS / "transfer"


def ci_of(diff: pd.Series, stock: pd.Series) -> dict:
    return cluster_bootstrap(diff.groupby(stock).mean().dropna().to_numpy(), n_boot=10_000)


def main() -> None:
    full = pd.read_parquet(T / "eval_val_v0.parquet")
    full = full[~full["excluded"]] if "excluded" in full else full
    ref = full[full["policy"].str.match(r"^zero_R3tick_s\d+$")].groupby("key")["c_spread"].mean().rename("ref")
    ptc = full[full["policy"] == "post_then_cross"].set_index("key")["c_spread"].rename("ptc")
    m, out = {}, {}

    own = pd.read_parquet(R / "grid_val_own.parquet")
    pool = pd.read_parquet(R / "grid_train_pool.parquet")
    v = pd.read_parquet(R / "grid_val_v0.parquet")
    best_pool = pool.groupby("policy")["c_spread"].mean().idxmin()
    best_own = own.groupby(["stock", "policy"])["c_spread"].mean().groupby(level=0).idxmin().map(lambda t: t[1])
    v = v.join(ref, on="key")
    pooled = v[v["policy"] == best_pool].set_index("key")
    per = v[v["policy"] == v["stock"].map(best_own).fillna(best_pool)].set_index("key")
    out["rules"] = {"pooled_rule": best_pool, "pooled_cost": float(pooled["c_spread"].mean()),
                    "per_stock_cost": float(per["c_spread"].mean()),
                    "per_stock_minus_pooled": ci_of(per["c_spread"] - pooled["c_spread"].reindex(per.index),
                                                     per["stock"]),
                    "zero_mstick_minus_per_stock": ci_of(per["ref"] - per["c_spread"], per["stock"])}
    r = out["rules"]
    m["RulesPooledCost"] = f"{r['pooled_cost']:.3f}"
    m["RulesPerStockCost"] = f"{r['per_stock_cost']:.3f}"
    for k, stem in (("per_stock_minus_pooled", "RulesPerStockMinusPooled"),
                    ("zero_mstick_minus_per_stock", "RulesZeroMinusPerStock")):
        m[stem], m[stem + "Lo"], m[stem + "Hi"] = (f"{r[k][x]:+.3f}" for x in ("mean", "lo", "hi"))

    pa = pd.read_parquet(T / "eval_val_v0_per_asset.parquet").join(ref, on="key")
    out["per_stock_fqi"] = {}
    for pol, g in pa.groupby("policy"):
        ci = ci_of(g["c_spread"] - g["ref"], g["stock"])
        out["per_stock_fqi"][pol] = {"cost": float(g["c_spread"].mean()), "minus_zero_mstick": ci}
    for pol, stem in (("assetsmall_R3tick_d5", "PerStockFive"), ("assetsmall_R3tick_d1", "PerStockOne")):
        r = out["per_stock_fqi"][pol]
        m[stem + "Cost"] = f"{r['cost']:.3f}"
        m[stem + "Minus"], m[stem + "MinusLo"], m[stem + "MinusHi"] = (f"{r['minus_zero_mstick'][x]:+.3f}"
                                                                       for x in ("mean", "lo", "hi"))
    # fine-tuning the pooled policy on the stock's own days, against the very model it starts from (seed 0)
    base0 = full[full["policy"] == "zero_R3tick_s0"].groupby("key")["c_spread"].mean().rename("base0")
    for name, label in (("val_v0_finetune", "Strong"), ("val_v0_finetune_light", "Light")):
        if not (T / f"eval_{name}.parquet").exists():
            continue
        ft = pd.read_parquet(T / f"eval_{name}.parquet")
        ft = ft[ft["policy"].str.startswith("finetune")].join(base0, on="key")
        for pol, g in ft.groupby("policy"):
            d = pol.rsplit("_d", 1)[1]
            ci = ci_of(g["c_spread"] - g["base0"], g["stock"])
            out.setdefault("finetune", {})[pol] = ci
            stem = f"Finetune{label}D{'One' if d == '1' else 'Five'}"
            m[stem], m[stem + "Lo"], m[stem + "Hi"] = (f"{ci[x]:+.3f}" for x in ("mean", "lo", "hi"))
    # economic size: the gain of the pooled MS+tick policy in basis points of the price (per-stock means)
    zb = full[full["policy"].str.match(r"^zero_R3tick_s\d+$")].groupby("key")["c_bps"].mean().rename("z")
    for b, stem in (("post_then_cross", "ValGainBpsPost"), ("immediate", "ValGainBpsImmediate")):
        bb = full[full["policy"] == b].set_index("key")[["c_bps", "stock"]].join(zb)
        ci = ci_of(bb["c_bps"] - bb["z"], bb["stock"])
        m[stem], m[stem + "Lo"], m[stem + "Hi"] = (f"{ci[x]:.2f}" for x in ("mean", "lo", "hi"))
    hs = full[full["policy"] == "post_then_cross"].groupby("stock")["c_bps"].mean()
    m["ValPostCostBps"] = f"{hs.mean():.2f}"
    m["ValPostThenCrossCost"] = f"{ptc.mean():.3f}"
    m["ValZeroMsTickCost"] = f"{ref.mean():.3f}"

    lc = json.loads((T / "learning_curve.json").read_text())["points"]
    words = {1000: "OneK", 3000: "ThreeK", 10_000: "TenK", 30_000: "ThirtyK", 100_000: "HundredK",
             max(q["n"] for q in lc): "Full"}
    m["CurveFullEpisodes"] = f"{max(q['n'] for q in lc):,}".replace(",", "{,}")
    m["CurveSeeds"] = str(len(lc[0].get("seeds", ["0"])))
    for q in lc:
        if q["n"] in words and q["rep"] in ("R3tick", "R0"):
            m[f"Curve{'MsTick' if q['rep'] == 'R3tick' else 'Raw'}{words[q['n']]}"] = f"{q['cost']:.3f}"
    out["provenance"] = provenance()
    write_json(T / "baselines.json", out)
    with open(REPO / "paper" / "numbers_baselines.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_baselines.py -- do not edit\n")
        for k, val in sorted(m.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    for k, val in sorted(m.items()):
        print(f"{k:32s} {val}")


if __name__ == "__main__":
    main()

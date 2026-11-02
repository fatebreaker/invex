"""How the main Nasdaq contrasts vary across stocks (review round 2; exploratory, from the supplement's test inputs).

    python experiments/per_stock_tails.py [--dir supplement/registered_tests]

For H1 (zero-shot MS+tick minus the seen policy), H2 (raw minus MS units) and H3 (z-scores minus MS+tick) at each
sealed stage (the paper reports the between-stock standard deviations and the tercile means; the shrunken quantiles
and shares depend on the noise model and are kept for comparison): the quantiles of the per-stock mean differences
and the shares of stocks beyond the margin, raw and after
normal-normal empirical-Bayes shrinkage (each stock's sampling variance from the spread of its day means; a stock with
one day takes the median spread of one day mean), the mean in each relative-tick tercile, and two estimates of the
between-stock standard deviation that rest on different models of the noise (see between_sd). Reads no sealed table.
Writes results/transfer/per_stock_tails.json and paper/numbers_per_stock_tails.tex.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from invex.paths import REPO, RESULTS

STAGES = {"StageOne": "confirm_stage1_stocks", "StageTwo": "confirm_stage2_stocks"}
HYPS = ("H1", "H2", "H3")
MARGIN = 0.015
N_BOOT = 2_000


def shrink(cells: pd.DataFrame, means: pd.Series) -> pd.Series:
    """Empirical-Bayes means: each stock's mean pulled towards the grand mean by its share of signal."""
    g = cells.groupby("stock")["mean"]
    days = g.count().reindex(means.index)
    s2 = g.var(ddof=1).reindex(means.index)  # spread of the stock's day means
    v = s2.fillna(s2.median()) / days  # a stock with one day takes the median spread of one day mean
    mu = means.mean()
    tau2 = max(float(means.var(ddof=1) - v.mean()), 0.0)
    return mu + tau2 / (tau2 + v) * (means - mu)


def between_sd(cells: pd.DataFrame, means: pd.Series) -> dict:
    """Two estimates of the between-stock standard deviation: the covariance across stocks of the means over alternate
    days (stocks with two days or more; needs no model of each stock's noise), and the moment estimate with one pooled
    variance of a day mean (negative estimates of the variance are reported as zero). The first has a 95% interval
    from resampling stocks (a negative covariance is reported as minus the root of its magnitude)."""
    c = cells.sort_values(["stock", "day"]).assign(half=lambda d: d.groupby("stock").cumcount() % 2)
    t = c.groupby(["stock", "half"])[["sum", "n"]].sum()
    hm = (t["sum"] / t["n"]).unstack().dropna()
    cov = float(np.cov(hm[0], hm[1])[0, 1])
    x, y = hm[0].to_numpy(), hm[1].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(x), size=(N_BOOT, len(x)))  # stocks resampled
    xb, yb = x[idx], y[idx]
    covb = ((xb - xb.mean(1, keepdims=True)) * (yb - yb.mean(1, keepdims=True))).sum(1) / (len(x) - 1)
    sdb = np.sign(covb) * np.sqrt(np.abs(covb))
    g = cells.groupby("stock")["mean"]
    days = g.count()
    pooled = float((g.var(ddof=1) * (days - 1)).sum() / (days - 1).sum())
    mom = float(means.var(ddof=1) - (pooled / days.reindex(means.index)).mean())
    return {"cross_half": float(np.sign(cov) * np.sqrt(abs(cov))), "cross_half_stocks": int(len(hm)),
            "cross_half_corr": float(np.corrcoef(x, y)[0, 1]),
            "cross_half_ci": [float(v) for v in np.quantile(sdb, [0.025, 0.975])], "pooled": float(np.sqrt(max(mom, 0.0)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "supplement" / "registered_tests"))
    a = ap.parse_args()
    cells = pd.read_parquet(f"{a.dir}/cells.parquet")
    stocks = pd.read_parquet(f"{a.dir}/stocks.parquet")
    out, m = {}, {}
    for tag, stem in STAGES.items():
        for h in HYPS:
            s = stocks[(stocks["analysis"] == stem) & (stocks["hypothesis"] == h)].set_index("stock")
            c = cells[(cells["analysis"] == stem) & (cells["hypothesis"] == h)]
            raw = s["mean"]
            eb = shrink(c, raw)
            r = {"stocks": int(len(raw))}
            for name, x in (("raw", raw), ("eb", eb)):
                q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
                r[name] = {"q": [float(v) for v in q], "above": float((x > MARGIN).mean()),
                           "below": float((x < -MARGIN).mean())}
            terc = s["tercile"].map({0: "small", 1: "mid", 2: "large"})
            r["terciles"] = {k: float(v) for k, v in raw.groupby(terc).mean().items()}
            r["between_sd"] = between_sd(c, raw)
            out[f"{tag}{h}"] = r
            k = f"Tails{tag}{h.replace('H1', 'HOne').replace('H2', 'HTwo').replace('H3', 'HThree')}"
            m[k + "EbFive"], m[k + "EbNinetyfive"] = f"{r['eb']['q'][0]:+.3f}", f"{r['eb']['q'][4]:+.3f}"
            m[k + "RawFive"], m[k + "RawNinetyfive"] = f"{r['raw']['q'][0]:+.3f}", f"{r['raw']['q'][4]:+.3f}"
            m[k + "EbAbove"] = f"{100 * r['eb']['above']:.0f}\\%"
            m[k + "EbBelow"] = f"{100 * r['eb']['below']:.0f}\\%"
            m[k + "RawAbove"] = f"{100 * r['raw']['above']:.0f}\\%"
            for t, v in r["terciles"].items():
                m[k + t.capitalize()] = f"{v:+.3f}"
            bsd = r["between_sd"]
            m[k + "XhSd"], m[k + "PooledSd"] = f"{bsd['cross_half']:.3f}", f"{bsd['pooled']:.3f}"
            m[k + "XhSdLo"], m[k + "XhSdHi"] = (f"{v:.3f}" for v in bsd["cross_half_ci"])
            m[k + "XhCorr"] = f"{bsd['cross_half_corr']:.2f}"
            print(f"{tag} {h}: {len(raw)} stocks; raw 5-95% [{r['raw']['q'][0]:+.3f}, {r['raw']['q'][4]:+.3f}], "
                  f"EB [{r['eb']['q'][0]:+.3f}, {r['eb']['q'][4]:+.3f}]; EB beyond +margin {r['eb']['above']:.1%}, "
                  f"below -margin {r['eb']['below']:.1%}; terciles {r['terciles']}; between-stock SD {r['between_sd']}")
    (RESULTS / "transfer" / "per_stock_tails.json").write_text(json.dumps(out, indent=1))
    lines = ["% generated by experiments/per_stock_tails.py (exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_per_stock_tails.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

"""PPO, the on-policy learner (invex.agents.ppo, experiments/ppo.py), against fitted Q-iteration on the validation
days -> paper/numbers_ppo.tex.

    python experiments/analyze_ppo.py

Paired cost differences on the same validation episodes (val_v0), per share in spreads (and basis points of the
arrival mid), stocks weighted equally, seeds averaged (three per learner), 90% intervals resampling stocks and,
within each arm, seeds. Exploratory, post hoc, validation only: no PPO policy has been scored on a sealed table.

After the fifth review (val_v0_ppo5, scripts/queue_ppo_round5.sh): PPO seen policies (H1's comparator with PPO), raw
units with a per-stock reward scale (R0pop) and 450-iteration runs (longppo), with tab_ppo5.tex and the learning
curves' last ten iterations. After the sixth review: the fitted-Q seen policy with all six seeds evaluated on these
episodes (s0 in val_v0, s3-s7 in val_v0_rev), stock-only intervals (H1's statistic, seeds averaged) for the transfer
gaps, the difference between the PPO and fitted-Q transfer gaps, and the 450-iteration training cost compared with the
end of the 150-iteration runs (whose entropy schedule ends at the same value). After the seventh review: where PPO's
transfer gap lies across stocks (`coverage`): by the distance of each validation stock's window statistics from the
training stocks' (mean distance to its ten nearest training stocks, statistics standardised over the training stocks),
and how much of its spread across stocks is signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_profile import boot, boot_seeds  # noqa: E402

from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

OUT = RESULTS / "transfer"
EVALS = ("val_v0", "val_v0_ppo", "val_v0_ppo2", "val_v0_ppo5", "val_v0_ppo6", "val_v0_ppo8")
# (macro, a, b): arms are model names without the seed; base:<name> for a baseline
CONTRASTS = [
    ("MsTickVsTrees", "zeroppo_R3tick", "zero_R3tick"),
    ("RawVsMsTick", "zeroppo_R0", "zeroppo_R3tick"),
    ("ZVsMsTick", "zeroppo_R1", "zeroppo_R3tick"),
    ("MsVsMsTick", "zeroppo_R3", "zeroppo_R3tick"),
    ("RawVsTrees", "zeroppo_R0", "zero_R0"),
    ("TreesRawVsMsTick", "zero_R0", "zero_R3tick"),
    ("TreesZVsMsTick", "zero_R1", "zero_R3tick"),
    ("TreesMsVsMsTick", "zero_R3", "zero_R3tick"),
    ("MsTickVsPost", "zeroppo_R3tick", "post_then_cross"),
    ("MsTickVsImmediate", "zeroppo_R3tick", "immediate"),
    # fifth review
    ("RawVsTreesMsTick", "zeroppo_R0", "zero_R3tick"),
    ("TgMsTick", "zeroppo_R3tick", "seenppo_R3tick"),
    ("TgRaw", "zeroppo_R0", "seenppo_R0"),
    ("TreesTgMsTick", "zero_R3tick", "seen_R3tick"),
    ("PopVsMsTick", "zeroppo_R0pop", "zeroppo_R3tick"),
    ("PopVsRaw", "zeroppo_R0pop", "zeroppo_R0"),
    ("LongMsTickVsMsTick", "longppo_R3tick", "zeroppo_R3tick"),
    ("LongRawVsLongMsTick", "longppo_R0", "longppo_R3tick"),
    ("LongMsTickVsTrees", "longppo_R3tick", "zero_R3tick"),
    ("LongRawVsTreesMsTick", "longppo_R0", "zero_R3tick"),
    # sixth review: PPO given the stock's probe profile (val_v0_ppo6, scripts/queue_ppo_round6.sh)
    ("ProfVsMsTick", "profppo_R3tickp", "zeroppo_R3tick"),
    ("ProfVsSeen", "profppo_R3tickp", "seenppo_R3tick"),
    # seventh review: raw features with the cost in the stock's spread (H8's ablation; val_v0_ppo8)
    ("RawrVsMsTick", "rawrppo_R0r", "zeroppo_R3tick"),
    ("RawrVsRaw", "rawrppo_R0r", "zeroppo_R0"),
]
TABLE = [  # (label, macro) for tab_ppo5.tex
    (r"PPO \Rmst{} $-$ fitted-Q \Rmst", "MsTickVsTrees"),
    (r"PPO raw $-$ PPO \Rmst", "RawVsMsTick"),
    (r"PPO raw $-$ fitted-Q \Rmst", "RawVsTreesMsTick"),
    (r"PPO raw, per-stock reward scale $-$ PPO \Rmst", "PopVsMsTick"),
    (r"PPO raw, per-stock reward scale $-$ PPO raw", "PopVsRaw"),
    (r"PPO raw, cost in spreads $-$ PPO \Rmst", "RawrVsMsTick"),
    (r"PPO raw, cost in spreads $-$ PPO raw", "RawrVsRaw"),
    (r"transfer gap, PPO \Rmst{} (zero-shot $-$ seen)", "TgMsTick"),
    (r"transfer gap, PPO raw (zero-shot $-$ seen)", "TgRaw"),
    (r"transfer gap, fitted-Q \Rmst{} (zero-shot $-$ seen)", "TreesTgMsTick"),
    (r"PPO \Rmst, 450 $-$ 150 iterations", "LongMsTickVsMsTick"),
    (r"PPO raw $-$ PPO \Rmst, 450 iterations", "LongRawVsLongMsTick"),
    (r"PPO \Rmst{} (450 iterations) $-$ fitted-Q \Rmst", "LongMsTickVsTrees"),
    (r"PPO raw (450 iterations) $-$ fitted-Q \Rmst", "LongRawVsTreesMsTick"),
    (r"PPO \Rmst{} with the stock's probe profile $-$ PPO \Rmst", "ProfVsMsTick"),
]


def f3(x: float) -> str:
    return f"{x:+.4f}" if abs(x) < 0.0005 else f"{x:+.3f}"  # one more digit, so that a sign near zero is not lost


def coverage(per: pd.DataFrame, rng_seed: int = 0) -> dict:
    """PPO's and fitted Q's transfer gaps (zero-shot minus seen, MS+tick) per validation stock against the stock's
    distance from the training stocks in window statistics (analyze_profile.stock_covariates): Spearman correlation,
    means by distance tercile (stocks resampled), far minus near tercile (terciles recut in each replicate), the far
    tercile's share of the total, and the share of the gap's variance across stocks that is signal rather than
    episode noise."""
    from scipy.stats import spearmanr
    from analyze_profile import stock_covariates
    from invex.eval.universe import load_splits
    X = stock_covariates()
    sp = pd.Series(load_splits()["split"])
    tr = X[X.index.isin(sp.index[sp == "train"])]
    mu, sd = tr.mean(), tr.std()
    z, out = ((tr - mu) / sd).to_numpy(), {}
    rng = np.random.default_rng(rng_seed)
    for lab, a, b in (("Ppo", "zeroppo_R3tick", "seenppo_R3tick"), ("Trees", "zero_R3tick", "seen_R3tick")):
        k = (per[a] - per[b]).dropna()
        g = k.groupby(level="stock")
        tg = g.mean()
        signal = 1 - float((g.var() / g.count()).mean() / tg.var())
        zv = ((X.reindex(tg.index).dropna() - mu) / sd)
        dist = pd.Series(np.sort(np.sqrt(((zv.to_numpy()[:, None, :] - z[None]) ** 2).sum(-1)), 1)[:, :10].mean(1),
                         index=zv.index)
        v, q = tg.reindex(dist.index).to_numpy(), dist.to_numpy()
        cut = np.quantile(q, [1 / 3, 2 / 3])
        ter = {"Near": q <= cut[0], "Far": q > cut[1]}
        idx = rng.integers(0, len(v), (4000, len(v)))
        diff = []
        for ii in idx:
            vv, qq = v[ii], q[ii]
            c = np.quantile(qq, [1 / 3, 2 / 3])
            diff.append(vv[qq > c[1]].mean() - vv[qq <= c[0]].mean())
        r = {"stocks": int(len(v)), "rho": float(spearmanr(q, v)[0]), "rho_p": float(spearmanr(q, v)[1]),
             "signal": signal, "far_share": float(v[ter["Far"]].sum() / v.sum()),
             "far_minus_near": [float(v[ter["Far"]].mean() - v[ter["Near"]].mean()), *map(float, np.percentile(diff, [5, 95]))]}
        for t, m in ter.items():
            b_ = v[m][rng.integers(0, m.sum(), (4000, m.sum()))].mean(1)
            r[t.lower()] = [float(v[m].mean()), *map(float, np.percentile(b_, [5, 95]))]
        out[lab] = r
    return out


def main() -> None:
    res = pd.concat([pd.read_parquet(OUT / f"eval_{e}.parquet") for e in EVALS if (OUT / f"eval_{e}.parquet").exists()],
                    ignore_index=True)
    rev = OUT / "eval_val_v0_rev.parquet"  # seen_R3tick seeds 3-7 on the same validation episodes
    if rev.exists():
        r = pd.read_parquet(rev)
        r = r[r["policy"].str.startswith("seen_R3tick_s") & r["key"].isin(res["key"])]
        res = pd.concat([res, r], ignore_index=True)
    res = res[~res["excluded"]] if "excluded" in res else res
    res = res.drop_duplicates(["key", "policy"])  # the baselines appear in every evaluation, identically
    res["arm"] = res["policy"].str.replace(r"_s\d+$", "", regex=True)
    out, macros = {"evals": EVALS, "contrasts": {}}, {}
    for unit, suf in (("c_spread", ""), ("c_bps", "Bps")):
        per = res.groupby(["arm", "stock", "key"])[unit].mean().unstack(0)
        pol = res.groupby(["policy", "stock"])[unit].mean().unstack(0)
        for name, x, y in CONTRASTS:
            if x not in per or y not in per:
                continue
            d = (per[x] - per[y]).dropna()
            m, lo, hi = boot(d.groupby(level="stock").mean())
            px = pol[[c for c in pol if c.rsplit("_s", 1)[0] == x]] if x in per else None
            py = pol[[c for c in pol if c == y or c.rsplit("_s", 1)[0] == y]]
            slo, shi = boot_seeds(px.to_numpy(), py.to_numpy())
            out["contrasts"][name + suf] = {"a": x, "b": y, "unit": unit, "mean": m, "lo": lo, "hi": hi,
                                           "lo_seed": slo, "hi_seed": shi}
            key = "Ppo" + name + suf
            macros[key], macros[key + "Lo"], macros[key + "Hi"] = f3(m), f3(slo), f3(shi)
            macros[key + "StockLo"], macros[key + "StockHi"] = f3(lo), f3(hi)
            out["contrasts"][name + suf]["seeds"] = [int(px.shape[1]), int(py.shape[1])]
            print(f"{unit:8s} {x:16s} - {y:16s} {m:+.4f} [{slo:+.4f}, {shi:+.4f}]")
        # PPO's transfer gap minus fitted-Q's (stocks resampled, seeds averaged)
        tg = ["zeroppo_R3tick", "seenppo_R3tick", "zero_R3tick", "seen_R3tick"]
        if all(a in per for a in tg):
            d = (per[tg[0]] - per[tg[1]] - per[tg[2]] + per[tg[3]]).dropna()
            m, lo, hi = boot(d.groupby(level="stock").mean())
            out["contrasts"]["TgDiffMsTick" + suf] = {"mean": m, "lo": lo, "hi": hi}
            k = "PpoTgDiffMsTick" + suf
            macros[k], macros[k + "Lo"], macros[k + "Hi"] = f3(m), f3(lo), f3(hi)
        if unit == "c_spread" and "RawrVsRaw" in out["contrasts"]:  # the cost unit's share of PPO's raw-unit loss
            share = -out["contrasts"]["RawrVsRaw"]["mean"] / out["contrasts"]["RawVsMsTick"]["mean"]
            macros["PpoCostUnitShare"] = f"{100 * share:.0f}\\%"
        if unit == "c_spread":
            for arm in ("zeroppo_R3tick", "zero_R3tick"):
                cost = per[arm].groupby(level="stock").mean().mean()  # stocks weighted equally
                macros["PpoCost" + ("Ppo" if arm.startswith("zeroppo") else "Trees")] = f"{cost:.3f}"
            macros["PpoStocks"] = str(per.index.get_level_values("stock").nunique())
            if all(a in per for a in ("zeroppo_R3tick", "seenppo_R3tick", "zero_R3tick", "seen_R3tick")):
                cv = coverage(per)
                out["coverage"] = cv
                for lab, r in cv.items():
                    p = f"PpoCover{lab}"
                    macros.update({p + "Rho": f"{r['rho']:+.2f}", p + "Signal": f"{100 * max(r['signal'], 0):.0f}\\%",
                                   p + "FarShare": f"{100 * r['far_share']:.0f}\\%",
                                   p + "Near": f3(r["near"][0]), p + "Far": f3(r["far"][0]),
                                   p + "FarLo": f3(r["far"][1]), p + "FarHi": f3(r["far"][2]),
                                   p + "Diff": f3(r["far_minus_near"][0]), p + "DiffLo": f3(r["far_minus_near"][1]),
                                   p + "DiffHi": f3(r["far_minus_near"][2])})
    curves = {}  # training cost (the stochastic policy's, in its own cost unit), mean of ten iterations
    for f in sorted((OUT / "ppo").glob("*ppo_*.json")):
        c = pd.DataFrame(json.loads(f.read_text())["curve"])
        curves[f.stem] = {"iters": len(c), "at150": float(c["cost"].iloc[140:150].mean()),
                          "last": float(c["cost"].iloc[-10:].mean())}
    out["curves"] = curves
    for arm, key in (("longppo_R3tick", "MsTick"), ("longppo_R0", "Raw")):
        cs = [v for k, v in curves.items() if k.rsplit("_s", 1)[0] == arm]
        # the 150-iteration runs of the same representation: their entropy bonus has fallen to the same final value
        short = [v for k, v in curves.items() if k.rsplit("_s", 1)[0] == arm.replace("longppo", "zeroppo")]
        if cs:
            a150, last = np.mean([v["at150"] for v in cs]), np.mean([v["last"] for v in cs])
            macros[f"PpoCurve{key}At"], macros[f"PpoCurve{key}Last"] = f"{a150:.3f}", f"{last:.3f}"
            macros[f"PpoCurve{key}Drop"] = f"{100 * (1 - last / a150):.1f}\\%"
            if short:
                end = np.mean([v["last"] for v in short])
                macros[f"PpoCurve{key}ShortEnd"] = f"{end:.3f}"
                macros[f"PpoCurve{key}DropVsShort"] = f"{100 * (1 - last / end):.1f}\\%"
    # seventh review: PPO's positive control, in the planted markout world inata (scripts/queue_ppo_planted.sh)
    pl = OUT / "profile_score_pplinata.json"
    if pl.exists() and (OUT / "profile_score_pplinata_pol.parquet").exists():
        from analyze_profile import MARGIN, pct, share_ci
        d = json.loads(pl.read_text())["contrasts"]
        for key, c in (("Oracle", d["pplinatax_R3tickp-pplinata_R3tick"]), ("Cmp", d["pplinata_R3tickp-pplinata_R3tick"])):
            macros.update({f"PpoPl{key}": f3(c["mean"]), f"PpoPl{key}Lo": f3(c["lo_seed"]),
                           f"PpoPl{key}Hi": f3(c["hi_seed"]), f"PpoPl{key}Margins": f"{-c['mean'] / MARGIN:.1f}"})
        sh, slo, shi = share_ci(["pplinata"])
        macros.update({"PpoPlShare": pct(sh), "PpoPlShareLo": pct(slo), "PpoPlShareHi": pct(shi)})
        out["planted"] = {"world": "inata", "contrasts": d, "share": [sh, slo, shi]}
        TABLE.extend([(r"PPO, planted markout world ($s = 2$): planted value $-$ blind", "PlOracle"),
                      (r"PPO, planted markout world ($s = 2$): profile $-$ blind", "PlCmp")])
    rows = [rf"{lab} & ${macros['Ppo' + k]}$ & $[{macros['Ppo' + k + 'Lo']}, {macros['Ppo' + k + 'Hi']}]$ \\"
            for lab, k in TABLE if "Ppo" + k in macros]
    (REPO / "paper" / "tab_ppo5.tex").write_text("\n".join(
        [r"\begin{tabular}{lcc}", r"\toprule", r"contrast (validation) & spreads per share & 90\% CI \\", r"\midrule",
         *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")
    write_json(OUT / "ppo_val_v0.json", {**out, "provenance": provenance(args={})})
    with open(REPO / "paper" / "numbers_ppo.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_ppo.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()

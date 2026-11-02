"""The stock-profile comparator and its positive control (experiments/stock_profile.py) -> paper/numbers_profile.tex
and paper/tab_profile.tex.

    python experiments/analyze_profile.py

Reads results/transfer/profile_score_{val_base,inj*,inm*}.json (and the per-stock, per-policy tables beside them) and
profiles/*.json. Planted worlds are named <kind><scale><draw>[nn]: kind inj (a per-stock crossing surcharge), inm (a
per-stock charge on passive fills: the crossing surcharge's mirror, since every order ends filled), ina (a per-stock
multiple of each passive fill's markout) or inh (a per-stock hidden queue), scale h/s/q/o/t, draw a-d, nn for the
neural learner. Recovered shares are
the comparator's gain over the blind policy as a share of the oracle's, with 90% intervals that resample stocks and,
within each arm, seeds (and, for a scale's average, every draw of it at once). Exploratory, post hoc, validation only.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from invex.paths import DATA, REPO, RESULTS

OUT = RESULTS / "transfer"
SCALES = {"h": ("Half", 0.5), "s": ("PointSix", 0.6), "q": ("ThreeQuarter", 0.75), "o": ("One", 1.0),
          "t": ("Two", 2.0), "f": ("Four", 4.0), "e": ("Eight", 8.0), "w": ("Sixteen", 16.0)}
DRAWS = {"a": 1, "b": 2, "c": 3, "d": 4}
KINDS = {"inj": ("Inj", "crossing"), "inm": ("Mark", "passive charge"), "ina": ("Mo", "markout"),
         "ind": ("Dmo", "demeaned markout"), "inq": ("Hab", "hidden queue"), "inh": ("Hid", "hidden queue (prop.)")}
KIND_ORDER = {"inj": 0, "inm": 1, "ina": 2, "ind": 3, "inq": 4, "inh": 5}
MARGIN = 0.015  # the registered equivalence margin of H1, spreads per share
WORLD = re.compile(r"^(inj|inm|ina|ind|inq|inh)([hsqotfew])([a-d])(nn)?$")


def f3(x: float) -> str:
    return f"{x:+.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%".replace("-", "$-$")


def arms(name: str, blind: str = "", cmp_: str = "", oracle: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per stock (rows) and seed (columns): the blind, comparator and oracle policies' costs in a planted world (arm
    names default to the world's own)."""
    pol = pd.read_parquet(OUT / f"profile_score_{name}_pol.parquet")
    pol = pol.dropna()

    def arm(a):
        return pol[[c for c in pol if c.rsplit("_s", 1)[0] == a]].to_numpy()

    return (arm(blind or f"{name}_R3tick"), arm(cmp_ or f"{name}_R3tickp"), arm(oracle or f"{name}x_R3tickp"))


def share_ci(names: list[str], n: int = 2000, seed: int = 0, **arm_names) -> tuple[float, float, float]:
    """Comparator gain over oracle gain, summed over the worlds, and its 90% interval (stocks and seeds resampled)."""
    rng = np.random.default_rng(seed)
    W = [arms(m, **arm_names) for m in names]
    point = sum(c.mean() - b.mean() for b, c, _ in W) / sum(o.mean() - b.mean() for b, _, o in W)
    reps = np.empty(n)
    for i in range(n):
        gc = go = 0.0
        for b, c, o in W:
            s = rng.integers(0, b.shape[0], b.shape[0])
            pick = [x[s][:, rng.integers(0, x.shape[1], x.shape[1])].mean() for x in (b, c, o)]
            gc += pick[1] - pick[0]
            go += pick[2] - pick[0]
        reps[i] = gc / go
    return float(point), float(np.percentile(reps, 5)), float(np.percentile(reps, 95))


def stock_covariates(days: tuple | None = None) -> pd.DataFrame:
    """Per stock, medians over its 2019 training-day episodes (or those of `days`): the window statistics (log trade
    rate, log trade size, log spread in ticks, log tick over price, share of the window at one tick, log depth in
    trade sizes, log one-minute volatility in spreads)."""
    cols = ["stock", "day", "w_seconds", "w_trades", "w_volume", "w_spread", "w_mid", "w_onetick", "w_depth",
            "w_sigma_1min", "tick"]
    e = pd.concat([pd.read_parquet(DATA / "transitions" / n / "eps.parquet", columns=cols)
                   for n in ("train_v0", "valseen_v0")], ignore_index=True)
    if days is not None:
        e = e[e["day"].isin(days)]
    tr = e["w_trades"].clip(lower=1)
    tsize = (e["w_volume"] / tr).clip(lower=1)
    x = pd.DataFrame({"stock": e["stock"], "rate": np.log(tr / e["w_seconds"]), "size": np.log(tsize),
                      "spread": np.log(e["w_spread"] / e["tick"]), "tick": np.log(e["tick"] / e["w_mid"]),
                      "onetick": e["w_onetick"], "depth": np.log(e["w_depth"].clip(lower=1) / tsize),
                      "vol": np.log(e["w_sigma_1min"].clip(lower=1e-9) / e["w_spread"])})
    return x.replace([np.inf, -np.inf], np.nan).groupby("stock").median().dropna()


def residual_signal(name: str = "base") -> dict:
    """How much of the profiles' variance across stocks the units and tick leave, and how much of that is signal:
    OLS of each column's unshrunk estimate on the stock covariates; signal = residual variance less mean sampling
    variance."""
    pr = pd.read_parquet(OUT / "profiles" / f"{name}.parquet").set_index("stock")
    X = stock_covariates()
    out = {}
    for c in ("p_post", "p_wait", "p_improve", "p_queue"):
        d = pr[[c + "_raw", c + "_se"]].join(X, how="inner").dropna()
        y = d[c + "_raw"].to_numpy()
        A = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        v_res = float(r @ r / (len(y) - A.shape[1]))
        se2 = float((d[c + "_se"] ** 2).mean())
        out[c] = signal_split(y, se2, A)
    return out


def signal_split(y: np.ndarray, se2: float, A: np.ndarray) -> dict:
    """OLS of y on A: R^2, reliability, residual reliability, signal SDs and the share of the signal explained."""
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    v_res = float(r @ r / (len(y) - A.shape[1]))
    v_tot = float(y.var(ddof=1))
    sig, sig_res = max(v_tot - se2, 0.0), max(v_res - se2, 0.0)
    return {"stocks": len(y), "rsq": float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))),
            "reliability": float(1 - se2 / v_tot), "resid_reliability": max(0.0, 1 - se2 / v_res),
            "resid_signal_sd": float(np.sqrt(sig_res)), "signal_sd": float(np.sqrt(sig)),
            "explained": float(1 - sig_res / sig) if sig > 0 else float("nan")}


def residual_signal_crossfit() -> dict:
    """As residual_signal, cross-fitted: the window statistics from some of a stock's 2019 days and its profile (the
    unshrunk mean of the probes' day means, with the sampling variance from their spread) from the others, in both
    directions (two days / three days), averaged. Day-level shocks then cannot enter both sides."""
    pr = pd.read_parquet(OUT / "profile_probes.parquet")
    w = pr.pivot_table(index=["stock", "day", "key"], columns="policy", values="j")
    day = w.groupby(level=["stock", "day"]).mean()
    n = w.groupby(level=["stock", "day"]).size().rename("n")
    days = sorted(pr["day"].unique())
    folds = [(tuple(days[:2]), tuple(days[2:])), (tuple(days[2:]), tuple(days[:2]))]
    probes = {"p_post": "post_then_cross", "p_wait": "cross_at_8", "p_improve": "improve_then_cross",
              "p_queue": "q_1_0.4"}
    out = {c: [] for c in probes}
    for x_days, y_days in folds:
        X = stock_covariates(x_days)
        for c, pol in probes.items():
            d = pd.DataFrame({"d": day[pol] - day["cross_at_0"]}).join(n).reset_index()
            d = d[d["day"].isin(y_days)]
            g = d.groupby("stock")
            y = g.apply(lambda t: np.average(t["d"], weights=t["n"])).rename("y")
            se2 = (g["d"].var(ddof=1) / g["d"].count()).rename("se2")
            t = pd.concat([y, se2], axis=1).join(X, how="inner").dropna()
            A = np.column_stack([np.ones(len(t)), t[X.columns].to_numpy()])
            out[c].append(signal_split(t["y"].to_numpy(), float(t["se2"].mean()), A))
    return {c: {k: float(np.mean([v[k] for v in r])) for k in r[0]} for c, r in out.items()}


def markout_decomposition(name: str = "inata", scale: float = 2.0, draw: int = 1) -> dict:
    """How much of a markout world's planted shift in the probes is a per-stock price of resting. The probes run on the
    same episodes with and without the markout and fill identically, so per stock and probe the shift in the mean cost
    is omega x (mean markout per passive share) x (passive share). Fitted with one global mean markout, and with one
    per stock (least squares through the origin): R^2 of each, over every stock and posting probe."""
    import sys
    sys.path.insert(0, str(REPO / "experiments"))
    pw = pd.read_parquet(OUT / f"profile_probes_{name}.parquet")
    pb = pd.read_parquet(OUT / "profile_probes.parquet")
    m = pw.merge(pb, on=["key", "policy", "stock", "day"], suffixes=("", "_b"))
    m = m[m["policy"] != "cross_at_0"]
    om = pd.read_parquet(OUT / "profiles" / f"{name}.parquet").set_index("stock")["omega"]  # the world's own draw
    m = m.assign(shift=m["j"] - m["j_b"], x=m["stock"].map(om).to_numpy() * (1.0 - m["tf"]))
    g = m.groupby(["stock", "policy"])[["shift", "x"]].mean()
    g = g[np.abs(g["x"]) > 0]
    tss = float(((g["shift"] - g["shift"].mean()) ** 2).sum())
    mbar = float((g["shift"] * g["x"]).sum() / (g["x"] ** 2).sum())
    r2_global = 1 - float(((g["shift"] - mbar * g["x"]) ** 2).sum()) / tss
    by = g.groupby(level="stock")
    ms = by.apply(lambda t: (t["shift"] * t["x"]).sum() / (t["x"] ** 2).sum())
    fit = g["x"] * ms.reindex(g.index.get_level_values("stock")).to_numpy()
    r2_stock = 1 - float(((g["shift"] - fit) ** 2).sum()) / tss
    q = ms.quantile([0.25, 0.75])
    return {"world": name, "mean_markout": mbar, "r2_global": r2_global, "r2_stock": r2_stock,
            "stock_markout_iqr": [float(q.iloc[0]), float(q.iloc[1])], "stocks": int(ms.size)}


def planted_shift(name: str) -> dict:
    """What a planted world does to posting and waiting on the validation days: the mean and the standard deviation of
    the change in its per-episode cost (planted world minus the real one, same episodes), and its cost's SD in both."""
    cols = ["key", "policy", "c_spread"]
    b = pd.read_parquet(OUT / "eval_val_v0.parquet", columns=cols)
    w = pd.read_parquet(OUT / f"eval_val_v0_{name}.parquet", columns=cols)
    m = w[w["policy"] == "post_then_cross"].merge(b[b["policy"] == "post_then_cross"], on=["key", "policy"],
                                                  suffixes=("", "_0"))
    d = m["c_spread"] - m["c_spread_0"]
    return {"shift": float(d.mean()), "charge_sd": float(d.std()), "sd": float(m["c_spread"].std()),
            "sd_base": float(m["c_spread_0"].std()), "episodes": int(len(m))}


def main() -> None:
    macros, rows = {}, []
    base = json.loads((OUT / "profile_score_val_base.json").read_text())
    c = base["contrasts"]["prof_R3tickp-zero_R3tick"]
    macros.update({"ProfReal": f3(c["mean"]), "ProfRealLo": f3(c["lo"]), "ProfRealHi": f3(c["hi"]),
                   "ProfRealSeedLo": f3(c["lo_seed"]), "ProfRealSeedHi": f3(c["hi_seed"]),
                   "ProfRealStocks": str(base["stocks"])})
    rel = json.loads((OUT / "profiles" / "base.json").read_text())["columns"]
    r = [v["reliability"] for v in rel.values()]
    macros.update({"ProfReliabilityMin": f"{min(r):.2f}", "ProfReliabilityMax": f"{max(r):.2f}",
                   "ProfSdMin": f"{min(v['sd'] for v in rel.values()):.3f}",
                   "ProfSdMax": f"{max(v['sd'] for v in rel.values()):.3f}"})
    rs = residual_signal()
    macros.update({"ProfUnitsRsqMin": f"{min(v['rsq'] for v in rs.values()):.2f}",
                   "ProfUnitsRsqMax": f"{max(v['rsq'] for v in rs.values()):.2f}",
                   "ProfResidRelMin": f"{min(v['resid_reliability'] for v in rs.values()):.2f}",
                   "ProfResidRelMax": f"{max(v['resid_reliability'] for v in rs.values()):.2f}",
                   "ProfResidSdMin": f"{min(v['resid_signal_sd'] for v in rs.values()):.3f}",
                   "ProfResidSdMax": f"{max(v['resid_signal_sd'] for v in rs.values()):.3f}",
                   "ProfExplainedMin": f"{100 * min(v['explained'] for v in rs.values()):.0f}\\%",
                   "ProfExplainedMax": f"{100 * max(v['explained'] for v in rs.values()):.0f}\\%",
                   "ProfQueueResidSd": f"{rs['p_queue']['resid_signal_sd']:.3f}"})
    cf = residual_signal_crossfit()
    if (OUT / "profile_probes_inata.parquet").exists():
        md = markout_decomposition()
        for w in ("indfa", "indea", "indwa"):  # the demeaned markout: how much is still a per-stock price of resting
            if (OUT / "profiles" / f"{w}.parquet").exists():
                d = markout_decomposition(w)
                lab = next(v[0] for k, v in SCALES.items() if k == w[3])
                macros.update({f"ProfDmo{lab}RsqGlobal": f"{100 * d['r2_global']:.0f}\\%",
                               f"ProfDmo{lab}RsqStock": f"{100 * d['r2_stock']:.0f}\\%",
                               f"ProfDmo{lab}MeanMarkout": f"{d['mean_markout']:+.3f}"})
                md[w] = d
        macros.update({"ProfMoMeanMarkout": f"{md['mean_markout']:.2f}",
                       "ProfMoMeanMarkoutLo": f"{md['stock_markout_iqr'][0]:.2f}",
                       "ProfMoMeanMarkoutHi": f"{md['stock_markout_iqr'][1]:.2f}",
                       "ProfMoRsqGlobal": f"{100 * md['r2_global']:.0f}\\%",
                       "ProfMoRsqStock": f"{100 * md['r2_stock']:.1f}\\%"})
    else:
        md = None
    macros.update({"ProfCfExplainedMin": f"{100 * min(v['explained'] for v in cf.values()):.0f}\\%",
                   "ProfCfExplainedMax": f"{100 * max(v['explained'] for v in cf.values()):.0f}\\%",
                   "ProfCfReliabilityMin": f"{min(v['reliability'] for v in cf.values()):.2f}",
                   "ProfCfReliabilityMax": f"{max(v['reliability'] for v in cf.values()):.2f}",
                   "ProfCfResidSdMax": f"{max(v['resid_signal_sd'] for v in cf.values()):.3f}"})
    rows.append(("real data", "", "", "trees", "", "", f"{c['mean']:+.4f} [{c['lo_seed']:+.4f}, {c['hi_seed']:+.4f}]", ""))
    f_nn = OUT / "profile_score_val_base_nn.json"
    if f_nn.exists():
        c = json.loads(f_nn.read_text())["contrasts"]["profnn_R3tickp-zeronn_R3tick"]
        macros.update({"ProfRealNn": f3(c["mean"]), "ProfRealNnSeedLo": f3(c["lo_seed"]),
                       "ProfRealNnSeedHi": f3(c["hi_seed"])})
        rows.append(("", "", "", "neural", "", "", f"{c['mean']:+.4f} [{c['lo_seed']:+.4f}, {c['hi_seed']:+.4f}]", ""))
    worlds = []
    for f in OUT.glob("profile_score_in*.json"):
        name = f.stem.removeprefix("profile_score_")
        if (m := WORLD.match(name)) and (OUT / f"profile_score_{name}_pol.parquet").exists():
            worlds.append((m.group(1), SCALES[m.group(2)][1], m.group(3), bool(m.group(4)), name))
    worlds.sort(key=lambda w: (KIND_ORDER[w[0]], *w[1:]))
    groups: dict = {}
    summary = {}
    for kind, s, draw, nn, name in worlds:
        klab, kdesc = KINDS[kind]
        lab = next(v[0] for v in SCALES.values() if v[1] == s)
        d = json.loads((OUT / f"profile_score_{name}.json").read_text())["contrasts"]
        orc, cmp_ = d[f"{name}x_R3tickp-{name}_R3tick"], d[f"{name}_R3tickp-{name}_R3tick"]
        world = name.removesuffix("nn")  # the neural runs use the same planted world and profile
        cols_ = json.loads((OUT / "profiles" / f"{world}.json").read_text())["columns"].values()
        corr = np.mean([v["corr_phi"] for v in cols_])
        flat = max(v["sd"] for v in cols_) < 1e-4  # the shrunk profile is constant across stocks: no correlation
        sh, slo, shi = share_ci([name])
        summary[name] = {"oracle": orc, "comparator": cmp_, "share": sh, "share_lo": slo, "share_hi": shi, "corr": corr}
        pre = f"Prof{klab}{lab}"
        if nn:
            macros.update({f"{pre}NnOracle": f3(orc["mean"]), f"{pre}NnCmp": f3(cmp_["mean"]), f"{pre}NnShare": pct(sh),
                           f"{pre}NnShareLo": pct(slo), f"{pre}NnShareHi": pct(shi)})
        else:
            groups.setdefault((klab, lab), []).append((name, orc["mean"], cmp_["mean"]))
            if draw == "a":
                macros[f"{pre}Corr"] = f"{corr:+.2f}".replace("-", "$-$")
                macros.update({f"{pre}Oracle": f3(orc["mean"]), f"{pre}Cmp": f3(cmp_["mean"]),
                               f"{pre}OracleLo": f3(orc["lo_seed"]), f"{pre}OracleHi": f3(orc["hi_seed"]),
                               f"{pre}CmpLo": f3(cmp_["lo_seed"]), f"{pre}CmpHi": f3(cmp_["hi_seed"])})
        rows.append((kdesc, f"{s:g}", str(DRAWS[draw]), "neural" if nn else "trees", "const." if flat else f"{corr:+.2f}",
                     f"{orc['mean']:+.4f} [{orc['lo_seed']:+.4f}, {orc['hi_seed']:+.4f}]",
                     f"{cmp_['mean']:+.4f} [{cmp_['lo_seed']:+.4f}, {cmp_['hi_seed']:+.4f}]",
                     f"{pct(sh)} [{pct(slo)}, {pct(shi)}]" if orc["hi_seed"] < 0 else "--"))  # no oracle gain: no share
    f_rb = OUT / "profile_score_indwarb.json"
    if f_rb.exists():  # the demeaned markout at s = 16 with the floored profile: after its own world's row
        c_rb = json.loads(f_rb.read_text())["contrasts"]
        o_rb, m_rb = c_rb["indwax_R3tickp-indwa_R3tick"], c_rb["indwarb_R3tickp-indwa_R3tick"]
        corr_rb = np.mean([v["corr_phi"] for v in json.loads((OUT / "profiles" / "indwarb.json").read_text())["columns"].values()])
        sh, slo, shi = share_ci(["indwarb"], blind="indwa_R3tick", cmp_="indwarb_R3tickp", oracle="indwax_R3tickp")
        at = next((i for i, r in enumerate(rows) if r[0] == "demeaned markout" and r[1] == "16"), len(rows) - 1) + 1
        rows.insert(at, ("\\quad floored profile", "16", "1", "trees", f"{corr_rb:+.2f}",
                         f"{o_rb['mean']:+.4f} [{o_rb['lo_seed']:+.4f}, {o_rb['hi_seed']:+.4f}]",
                         f"{m_rb['mean']:+.4f} [{m_rb['lo_seed']:+.4f}, {m_rb['hi_seed']:+.4f}]",
                         f"{pct(sh)} [{pct(slo)}, {pct(shi)}]"))
    for (klab, lab), v in groups.items():  # averaged over the draws of a scale
        names, o, cm = zip(*v)
        o, cm = np.array(o), np.array(cm)
        sh, slo, shi = share_ci(list(names))
        pre = f"Prof{klab}{lab}"
        macros.update({f"{pre}OracleAvg": f3(o.mean()), f"{pre}CmpAvg": f3(cm.mean()), f"{pre}Share": pct(sh),
                       f"{pre}ShareLo": pct(slo), f"{pre}ShareHi": pct(shi), f"{pre}Draws": str(len(v)),
                       f"{pre}OracleMargins": f"{-o.mean() / MARGIN:.1f}", f"{pre}CmpMargins": f"{-cm.mean() / MARGIN:.1f}"})
        summary[f"{klab}{lab}"] = {"draws": list(names), "oracle": float(o.mean()), "comparator": float(cm.mean()),
                                   "share": sh, "share_lo": slo, "share_hi": shi}
    shifts = {}
    for name in ("indea", "indwa", "inqfa", "inqwa"):  # the worlds that keep a stock's average price of resting
        if (OUT / f"eval_val_v0_{name}.parquet").exists():
            sh_ = shifts[name] = planted_shift(name)
            klab, lab = KINDS[name[:3]][0], SCALES[name[3]][0]
            macros.update({f"Prof{klab}{lab}PostShift": f"{sh_['shift']:+.2f}", f"Prof{klab}{lab}PostSd": f"{sh_['sd']:.2f}",
                           f"Prof{klab}{lab}ChargeSd": f"{sh_['charge_sd']:.2f}", "ProfPostSdBase": f"{sh_['sd_base']:.2f}"})
    summary["shifts"] = shifts
    f = OUT / "profile_score_indwarb.json"  # the demeaned markout at s = 16 with a floored profile (queue_profile_robust)
    if f.exists():
        c = json.loads(f.read_text())["contrasts"]["indwarb_R3tickp-indwa_R3tick"]
        sh, slo, shi = share_ci(["indwarb"], blind="indwa_R3tick", cmp_="indwarb_R3tickp", oracle="indwax_R3tickp")
        macros.update({"ProfDmoRobustCmp": f3(c["mean"]), "ProfDmoRobustCmpLo": f3(c["lo_seed"]),
                       "ProfDmoRobustCmpHi": f3(c["hi_seed"]), "ProfDmoRobustCmpMargins": f"{-c['mean'] / MARGIN:.1f}",
                       "ProfDmoRobustShare": pct(sh), "ProfDmoRobustShareLo": pct(slo), "ProfDmoRobustShareHi": pct(shi)})
        rb = json.loads((OUT / "profiles" / "indwarb.json").read_text())["columns"]
        macros.update({"ProfDmoRobustSdMin": f"{min(v['sd'] for v in rb.values()):.3f}",
                       "ProfDmoRobustSdMax": f"{max(v['sd'] for v in rb.values()):.3f}",
                       "ProfDmoRobustCorrMin": f"{min(v['corr_phi'] for v in rb.values()):+.2f}".replace("-", "$-$"),
                       "ProfDmoRobustCorrMax": f"{max(v['corr_phi'] for v in rb.values()):+.2f}".replace("-", "$-$")})
        summary["DmoRobust"] = {"comparator": c, "share": [sh, slo, shi]}
    base_pr = pd.read_parquet(OUT / "profiles" / "base.parquet").set_index("stock")
    for name in ("indea", "indwa"):  # what the comparator was given in the demeaned-markout worlds
        f = OUT / "profiles" / f"{name}.parquet"
        if not f.exists():
            continue
        pr = pd.read_parquet(f).set_index("stock")
        cols = ["p_post", "p_wait", "p_improve", "p_queue"]
        rel = [1 - (pr[c + "_se"] ** 2).mean() / pr[c + "_raw"].var() for c in cols]
        raw = [np.corrcoef(pr[c + "_raw"], pr["omega"])[0, 1] for c in cols]
        j = pr.join(base_pr[[c + "_raw" for c in cols]], rsuffix="_b")
        shift = [np.corrcoef(j[c + "_raw"] - j[c + "_raw_b"], j["omega"])[0, 1] for c in cols]
        lab = SCALES[name[3]][0]
        neg = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
        macros.update({f"ProfDmo{lab}RelMin": neg(min(rel)), f"ProfDmo{lab}RelMax": neg(max(rel)),
                       f"ProfDmo{lab}ShrunkSd": f"{max(pr[c].std() for c in cols):.0e}",
                       f"ProfDmo{lab}RawCorrMin": neg(min(raw)), f"ProfDmo{lab}RawCorrMax": neg(max(raw)),
                       f"ProfDmo{lab}ShiftCorrMin": neg(min(shift)), f"ProfDmo{lab}ShiftCorrMax": neg(max(shift))})
        sc = summary.get(f"Dmo{lab}")
        if sc:
            o = json.loads((OUT / f"profile_score_{name}.json").read_text())["contrasts"][f"{name}x_R3tickp-{name}_R3tick"]
            macros.update({f"ProfDmo{lab}OracleMarginsLo": f"{-o['hi_seed'] / MARGIN:.1f}",
                           f"ProfDmo{lab}OracleMarginsHi": f"{-o['lo_seed'] / MARGIN:.1f}"})
    with open(REPO / "paper" / "numbers_profile.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_profile.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    (OUT / "profile_summary.json").write_text(json.dumps({"worlds": summary, "residual": rs, "residual_crossfit": cf, "markout": md},
                                                          indent=1))
    lines = [r"\begin{tabular}{lcclcccc}", r"\toprule",
             r"planted cost & $s$ & draw & learner & corr.\ & oracle $-$ blind & comparator $-$ blind & recovered \\",
             r"\midrule"]
    num = re.compile(r"[+-]\d")
    lines += [" & ".join(f"${x}$" if num.match(x) else x for x in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (REPO / "paper" / "tab_profile.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(f"{k} = {v}" for k, v in sorted(macros.items())))
    print(json.dumps(rs, indent=1))


if __name__ == "__main__":
    main()

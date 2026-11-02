"""Analysis of the go/no-go probes, as fixed in prereg/gonogo_rule.md.

`analyze(days)` reads results/gonogo/{probes,episodes}_<day>.parquet and
writes results/gonogo/analysis.json plus a readable summary (analysis.md).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..env.units import unit_frame
from ..paths import RESULTS
from ..provenance import provenance, write_json
from .gonogo import FAMILY_P_S, K, LATENCY_NS, MARKOUT_DELAYS_S, NS
from .universe import load_splits

SYSTEMS = ("R0", "R2", "R3", "R3s", "R3d")
HORIZONS = (1, 3, 10, 30, 100)
DELTAS = (1, 10, 30)
THETAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)
N_PLACEBO = 5
MIN_N = 50


# --- units per probe ---------------------------------------------------------------------------

def unit_table(p: pd.DataFrame, ko: dict) -> dict[str, pd.DataFrame]:
    """Per probe: time (s), size (shares), price (price units) in each system (invex.env.units.unit_frame)."""
    return unit_frame(p, ko)


def _interp_mid(p: pd.DataFrame, delay_s: np.ndarray) -> np.ndarray:
    """Mid at t_done + delay (log-time interpolation on the measured grid)."""
    grid = np.array(MARKOUT_DELAYS_S)
    mids = p[[f"mid_after_{d:g}" for d in MARKOUT_DELAYS_S]].to_numpy(float)
    lg = np.log(grid + 0.05)
    x = np.log(np.clip(delay_s, 0, grid[-1]) + 0.05)
    k = np.clip(np.searchsorted(lg, x) - 1, 0, len(grid) - 2)
    w = (x - lg[k]) / (lg[k + 1] - lg[k])
    r = np.arange(len(p))
    out = mids[r, k] * (1 - w) + mids[r, k + 1] * w
    out[delay_s > grid[-1]] = np.nan
    return out


# --- per asset-day summaries ---------------------------------------------------------------------

def _fill_summary(p, t_unit, h):
    t_arr = p["t0"].to_numpy() + LATENCY_NS
    H = (h * t_unit * NS).astype(np.int64)
    done = p["t_done"].to_numpy()
    filled = (done > 0) & (done - t_arr <= H)
    usable = filled | (t_arr + H <= p["t_max"].to_numpy())
    d = pd.DataFrame({"g": p["g"].to_numpy()[usable], "y": filled[usable].astype(float)})
    a = d.groupby("g")["y"].agg(["sum", "count"])
    a = a[a["count"] >= MIN_N]
    ph = (a["sum"] + 0.5) / (a["count"] + 1.0)
    est = np.log(ph / (1 - ph))
    var = 1.0 / (a["count"] * ph * (1 - ph))
    return est, var


def _markout_summary(p, t_unit, p_unit, delta):
    ok = p["t_done"].to_numpy() > 0
    mid = _interp_mid(p, delta * t_unit)
    m = p["side"].to_numpy() * (mid - p["price"].to_numpy()) / p_unit
    ok &= np.isfinite(m)
    d = pd.DataFrame({"g": p["g"].to_numpy()[ok], "y": m[ok]})
    a = d.groupby("g")["y"].agg(["mean", "var", "count"])
    a = a[a["count"] >= MIN_N]
    return a["mean"], a["var"] / a["count"]


def tau2(est: pd.Series, var: pd.Series) -> float:
    return float(max(0.0, np.var(est.to_numpy(), ddof=1) - np.mean(var.to_numpy())))


def wls_slope(y, x, w) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    resid = y - X @ beta
    cov = np.linalg.inv(X.T @ W @ X) * float(resid @ W @ resid / max(len(y) - 2, 1))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def eta_share(est, var, log_eta) -> float:
    """Share of the noise-free between-asset variance explained by log eta (method of moments on residuals)."""
    t_all = tau2(est, var)
    if t_all <= 0:
        return float("nan")
    b = np.polyfit(log_eta, est.to_numpy(), 1)
    resid = pd.Series(est.to_numpy() - np.polyval(b, log_eta), index=est.index)
    return float(1 - tau2(resid, var) / t_all)


def summaries(p: pd.DataFrame, units: dict, system: str, c_u: float, unit_override=None) -> dict:
    t_u = units[system]["t"].to_numpy() * c_u
    p_u = units[system]["p"].to_numpy()
    if unit_override is not None:
        t_u, p_u = unit_override
    out = {}
    out["S1"] = _fill_summary(p, t_u, 10)
    out["S2"] = _fill_summary(p, t_u, 100)
    out["S3"] = _markout_summary(p, t_u, p_u, 10)
    out["S4"] = _markout_summary(p, t_u, p_u, 30)
    for h in HORIZONS:
        out[f"fill_h{h}"] = _fill_summary(p, t_u, h)
    return out


# --- part (C) ----------------------------------------------------------------------------------------

def rule_costs(ep: pd.DataFrame, pr: pd.DataFrame, units: dict, system: str) -> np.ndarray:
    """Cost (spread units) of every episode under every theta: array (n_episodes, len(THETAS)); plus
    the immediate-cross cost in column -1 of the returned (n, len(THETAS)+1) array."""
    j = ep["row"].to_numpy()
    t_u = units[system]["t"].to_numpy()[j]
    q_u = units[system]["q"].to_numpy()[j]
    side = pr["side"].to_numpy()[j]
    mid0 = pr["mid0"].to_numpy()[j]
    spread0 = pr["spread0"].to_numpy()[j]
    Q = 100.0
    cash = np.stack(ep["cash"].to_numpy())
    xcash = np.stack(ep["xcash"].to_numpy())
    ahead = np.stack(ep["ahead"].to_numpy()).astype(float)
    filled = np.stack(ep["filled"].to_numpy())
    T = ep["T_ns"].to_numpy() / NS
    total = cash + xcash  # cost of stopping (crossing the rest) at decision k
    cost_k = side[:, None] * (total / Q - mid0[:, None]) / spread0[:, None]
    ks = np.arange(K + 1)
    t_left = T[:, None] * (K - ks[None, :]) / K
    lhs = ahead / q_u[:, None]
    rhs_base = t_left / t_u[:, None]
    out = np.empty((len(ep), len(THETAS) + 1))
    active = filled < Q
    for i, th in enumerate(THETAS):
        if np.isinf(th):
            cross = np.zeros_like(lhs, dtype=bool)
        else:
            cross = (lhs > th * rhs_base) & active
        cross[:, 0] = False
        cross[:, K] = True
        k_star = np.argmax(cross[:, 1:], axis=1) + 1
        out[:, i] = cost_k[np.arange(len(ep)), k_star]
    out[:, -1] = cost_k[:, 0]
    return out


def part_c(ep: pd.DataFrame, pr: pd.DataFrame, units: dict, eta_cuts) -> dict:
    res = {}
    first = (pr["probe"].to_numpy()[ep["row"].to_numpy()] < 100)
    g = ep["g"].to_numpy()
    eta = units["_eta_sigma"].to_numpy()[ep["row"].to_numpy()]
    for u in SYSTEMS:
        if u in ("R3s",):  # same time and size units as R3: identical rule
            continue
        C = rule_costs(ep, pr, units, u)
        ok = np.isfinite(C).all(axis=1)
        df = pd.DataFrame(C[:, :-1], columns=[str(t) for t in THETAS])
        df["imm"] = C[:, -1]
        df["g"] = g
        df["first"] = first
        df["eta"] = eta
        df = df[ok]
        th_cols = [str(t) for t in THETAS]
        fh = df[df["first"]].groupby("g")[th_cols].mean()
        sh = df[~df["first"]].groupby("g")[th_cols + ["imm", "eta"]].mean()
        common = fh.index.intersection(sh.index)
        fh, sh = fh.loc[common], sh.loc[common]
        th_a = fh.idxmin(axis=1)  # per asset-day
        cost_a = np.array([sh.loc[i, th_a[i]] for i in common])
        th_pool = fh.mean().idxmin()
        cost_pool = sh[th_pool].to_numpy()
        terc = np.digitize(np.log(sh["eta"].to_numpy()), np.log(eta_cuts))
        # theta pooled within eta terciles (the GO-with-tick variant)
        cost_pool_t = np.empty(len(common))
        th_pool_t = {}
        for t in range(3):
            m = terc == t
            if m.any():
                th_t = fh[m].mean().idxmin()
                th_pool_t[t] = th_t
                cost_pool_t[m] = sh.loc[common[m], th_t].to_numpy()
        rows = {}
        for t in range(3):
            m = terc == t
            imp = float(np.mean(sh["imm"].to_numpy()[m] - cost_a[m])) if m.any() else np.nan
            rows[t] = {"n": int(m.sum()), "improvement": imp,
                       "regret_pooled": float(np.mean(cost_pool[m] - cost_a[m])) if m.any() else np.nan,
                       "regret_pooled_within_tercile": float(np.mean(cost_pool_t[m] - cost_a[m])) if m.any() else np.nan}
            rows[t]["ratio"] = rows[t]["regret_pooled"] / imp if imp else np.nan
            rows[t]["ratio_within_tercile"] = rows[t]["regret_pooled_within_tercile"] / imp if imp else np.nan
        res[u] = {"theta_pooled": th_pool, "theta_pooled_by_tercile": th_pool_t, "terciles": rows,
                  "theta_a_counts": th_a.value_counts().to_dict()}
    return res


# --- driver ------------------------------------------------------------------------------------------

def analyze(days: list[str]) -> dict:
    sp = load_splits()
    ko = sp["ko_constants"]
    pr = pd.concat([pd.read_parquet(RESULTS / "gonogo" / f"probes_{d}.parquet") for d in days], ignore_index=True)
    pr = pr[pr["status"] == 0].reset_index(drop=True)
    pr["g"] = pr["day"] + "|" + pr["stock"]
    with np.errstate(divide="ignore", invalid="ignore"):
        units = unit_table(pr, ko)
    # amendment 1: every system on the same probes, all units finite and positive
    valid = np.isfinite(units["_eta_sigma"].to_numpy())
    for u in SYSTEMS:
        v = units[u].to_numpy()
        valid &= np.isfinite(v).all(axis=1) & (v > 0).all(axis=1)
    n_dropped = int((~valid).sum())
    pr = pr[valid].reset_index(drop=True)
    units = unit_table(pr, ko)
    med_t3 = float(np.nanmedian(units["R3"]["t"]))
    c = {u: med_t3 / float(np.nanmedian(units[u]["t"])) for u in SYSTEMS}
    out = {"days": days, "n_probes": len(pr), "n_dropped_undefined_units": n_dropped,
           "n_asset_days": int(pr["g"].nunique()), "c_u": c, "systems": {}}
    rng = np.random.default_rng(0)
    gday = pr.groupby("g")["day"].first()
    eta_g = units["_eta_sigma"].groupby(pr["g"]).median()
    act_g = units["_activity"].groupby(pr["g"]).mean()
    for u in SYSTEMS:
        real = summaries(pr, units, u, c[u])
        # placebo: every asset-day takes another asset-day's median units (same day)
        med = pd.DataFrame({"t": units[u]["t"] * c[u], "p": units[u]["p"]}).groupby(pr["g"]).median()
        plac = {k: [] for k in real}
        for _ in range(N_PLACEBO):
            perm = {}
            for d, gs in gday.groupby(gday):
                names = gs.index.to_numpy()
                perm.update(dict(zip(names, rng.permutation(names))))
            src = pr["g"].map(perm)
            t_pl = med.loc[src, "t"].to_numpy()
            p_pl = med.loc[src, "p"].to_numpy()
            s_pl = summaries(pr, units, u, 1.0, unit_override=(t_pl, p_pl))
            for k, (e, v) in s_pl.items():
                plac[k].append(tau2(e, v))
        row = {}
        for k, (e, v) in real.items():
            t_real = tau2(e, v)
            t_plac = float(np.mean(plac[k]))
            row[k] = {"tau2": t_real, "tau2_placebo": t_plac, "reduction": 1 - t_real / t_plac if t_plac > 0 else np.nan,
                      "n": int(len(e)), "mean": float(np.mean(e))}
            if k in ("S1", "S2"):
                x = np.log(act_g.loc[e.index].to_numpy())
                row[k]["beta_activity"], row[k]["beta_se"] = wls_slope(e.to_numpy(), x, 1.0 / v.to_numpy())
            if k in ("S1", "S2", "S3", "S4"):
                row[k]["eta_share"] = eta_share(e, v, np.log(eta_g.loc[e.index].to_numpy()))
        out["systems"][u] = row
    # part (C)
    eps = pd.concat([pd.read_parquet(RESULTS / "gonogo" / f"episodes_{d}.parquet") for d in days], ignore_index=True)
    key = pr.reset_index().set_index(["day", "stock", "probe"])["index"]
    eps["row"] = key.reindex(pd.MultiIndex.from_frame(eps[["day", "stock", "probe"]])).to_numpy()
    eps = eps[np.isfinite(eps["row"])].copy()
    eps["row"] = eps["row"].astype(int)
    eps["g"] = pr["g"].to_numpy()[eps["row"].to_numpy()]
    out["part_c"] = part_c(eps, pr, units, sp["eta_sigma_tercile_cuts"])
    out["decision"] = decide(out)
    out["provenance"] = provenance()
    write_json(RESULTS / "gonogo" / "analysis.json", out)
    (RESULTS / "gonogo" / "analysis.md").write_text(render(out))
    print(render(out))
    return out


def decide(out: dict) -> dict:
    sysr = out["systems"]
    pc = out["part_c"]

    def crit(u):
        s = sysr[u]
        i = sum(s[k]["reduction"] >= 0.75 for k in ("S1", "S2", "S3", "S4"))
        ii = all(abs(s[k]["beta_activity"]) <= 0.1 for k in ("S1", "S2"))
        iii = all(r["ratio"] <= 0.25 for r in pc[u]["terciles"].values() if r["n"])
        iii_t = all(r["ratio_within_tercile"] <= 0.25 for r in pc[u]["terciles"].values() if r["n"])
        eta_ok = np.nanmean([s[k]["eta_share"] for k in ("S1", "S2", "S3", "S4")]) >= 0.5
        nogo = sum(r["ratio_within_tercile"] > 0.5 for r in pc[u]["terciles"].values() if r["n"]) >= 2
        return {"i_reductions>=0.75": int(i), "i": i >= 3, "ii": bool(ii), "iii": bool(iii), "iii_within_tercile": bool(iii_t),
                "eta_share>=0.5": bool(eta_ok), "nogo_condition": bool(nogo)}

    c = {u: crit(u) for u in ("R0", "R2", "R3", "R3d")}
    if any(c[u]["i"] and c[u]["ii"] and c[u]["iii"] for u in ("R2", "R3")):
        verdict = "GO-universal"
    elif any(c[u]["eta_share>=0.5"] and c[u]["iii_within_tercile"] for u in ("R2", "R3")):
        verdict = "GO-with-tick"
    elif all(c[u]["nogo_condition"] for u in c):
        verdict = "NO-GO"
    else:
        verdict = "other (decide at G2 with the full evidence)"
    return {"verdict": verdict, "criteria": c}


def render(out: dict) -> str:
    lines = [f"# Go/no-go analysis: {out['n_asset_days']} asset-days, {out['n_probes']:,} probes ({', '.join(out['days'])})", ""]
    lines.append("| system | " + " | ".join(f"{k} red." for k in ("S1", "S2", "S3", "S4")) + " | beta S1 | beta S2 | eta share S1 |")
    lines.append("|---" * 8 + "|")
    for u, s in out["systems"].items():
        lines.append(f"| {u} | " + " | ".join(f"{s[k]['reduction']:.2f}" for k in ("S1", "S2", "S3", "S4"))
                     + f" | {s['S1']['beta_activity']:.3f} | {s['S2']['beta_activity']:.3f} | {s['S1']['eta_share']:.2f} |")
    lines += ["", "Part (C): pooled-rule regret / improvement by eta tercile (pooled across all; pooled within tercile)", ""]
    for u, r in out["part_c"].items():
        t = r["terciles"]
        lines.append(f"- {u}: theta*={r['theta_pooled']}; " + "; ".join(
            f"T{k}: {v['ratio']:.2f} / {v['ratio_within_tercile']:.2f} (imp {v['improvement']:.3f}, n={v['n']})" for k, v in t.items()))
    lines += ["", f"**Verdict: {out['decision']['verdict']}**", "", "```", json.dumps(out["decision"]["criteria"], indent=1), "```"]
    return "\n".join(lines)

"""Analyses added after the sealed tests in response to review (all exploratory): training-seed variation, the
stock-by-day bootstrap, the margin, costs in basis points and net of fees, Benjamini-Hochberg over the
exploratory families, the learner and normalisation arms, the stock-identity comparator, and absolute contrasts
outside the training range.

    python experiments/analyze_revision.py --eval stage1 [--n-boot 4000]

Reads results/transfer/eval_<eval>.parquet and, when present, eval_<eval>_rev.parquet (the same episodes scored
by more policies: seeds 3-7, the neural learner, learned normalisation, stock identity). No episode table is
loaded: the relative tick and the window statistics of each episode come from the _rev evaluation, which stores
them. Costs are per share, in the stock's mean spread unless stated. Writes results/transfer/revision_<eval>.json
and paper/numbers_revision_<eval>.tex.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from invex.eval.stats import bh, cluster_bootstrap, pigeonhole_bootstrap
from invex.eval.universe import load_splits
from invex.paths import DATA, REPO, RESULTS
from invex.provenance import provenance, write_json

OUT = RESULTS / "transfer"
NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}
DELTA = 0.015  # the registered margin (spreads per share); 0.10 volatility units for crypto, set in main
STRATIFY = True
# Nasdaq fees in the reference price unit (1e-4 dollars) per share: taker 0.30 cents, maker rebate 0.20 cents
TAKER, REBATE = 30.0, 20.0
SEEN = {"val_v0": ("seen", "seenid", "valseen_v0"), "stage1": ("seent", "seentid", "testseen_v0"),
        "stage2": ("seent", "seentid", "testseen_v0")}
# the identity comparator with strongly regularised categorical splits (tag + "r")
PREFIX = {"val_v0": "RevVal", "stage1": "RevStageOne", "stage2": "RevStageTwo", "crypto_test": "RevCrypto",
          "crypto_rep": "RevCryptoRep"}
# learner, normalisation and identity arms: (name, a, b, stocks), costs of a minus b
ARMS = [
    ("nn raw vs ms", "zeronn:R0", "zeronn:R3", "all"),
    ("nn zscore vs ms+tick", "zeronn:R1", "zeronn:R3tick", "all"),
    ("nn desc vs ms", "zeronn:R0desc", "zeronn:R3", "all"),
    ("nn ko vs ms", "zeronn:R2", "zeronn:R3", "all"),
    ("nn tick vs ms", "zeronn:R3tick", "zeronn:R3", "all"),
    ("nn raws vs ms", "zeronn:R0r", "zeronn:R3", "all"),
    ("nn raw vs raws", "zeronn:R0", "zeronn:R0r", "all"),
    ("nn pop vs ms+tick", "zeronn:R0pop", "zeronn:R3tick", "all"),
    ("nn aug vs ms+tick", "zeronn:R0aug", "zeronn:R3tick", "all"),
    ("nn aug vs raw", "zeronn:R0aug", "zeronn:R0", "all"),
    ("nn ms+tick vs trees ms+tick", "zeronn:R3tick", "zero:R3tick", "all"),
    ("nn raw vs trees raw", "zeronn:R0", "zero:R0", "all"),
    ("trees ms+tick+desc vs ms+tick", "zero:R3tickdesc", "zero:R3tick", "all"),
    ("nn ms+tick+desc vs ms+tick", "zeronn:R3tickdesc", "zeronn:R3tick", "all"),
    ("trees pop vs ms", "zero:R0pop", "zero:R3", "all"),
    ("trees pop vs raws", "zero:R0pop", "zero:R0r", "all"),
    ("trees pop vs raw", "zero:R0pop", "zero:R0", "all"),
    ("zero ms+tick vs id ms+tick", "zero:R3tick", "id:R3tickid", "seen"),
    ("zero raw vs id raw", "zero:R0", "id:R0id", "seen"),
    ("id ms+tick vs seen ms+tick", "id:R3tickid", "seen:R3tick", "seen"),
    ("zero ms+tick vs idr ms+tick", "zero:R3tick", "idr:R3tickid", "seen"),
    ("idr ms+tick vs seen ms+tick", "idr:R3tickid", "seen:R3tick", "seen"),
]
ECON = [("raw vs ms", "zero:R0", "zero:R3"), ("raws vs ms", "zero:R0r", "zero:R3"),
        ("raw vs raws", "zero:R0", "zero:R0r"), ("tick vs ms", "zero:R3tick", "zero:R3"),
        ("ms+tick vs post", "zero:R3tick", "base:post_then_cross"),
        ("ms+tick vs immediate", "zero:R3tick", "base:immediate")]


def mname(*parts) -> str:
    """LaTeX macro name: every word capitalised, digits spelled out ('zero ms+tick vs post' -> ZeroMsTickVsPost)."""
    words = [w for p in parts for w in str(p).replace("_", " ").replace("+", " ").split()]
    s = "".join(w[:1].upper() + w[1:] for w in words)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def load(ev: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-episode rows of every policy (original evaluation plus _rev), excluded episodes dropped, and one row
    of episode information per key."""
    parts = [pd.read_parquet(OUT / f"eval_{ev}.parquet")]
    revs = sorted(OUT.glob(f"eval_{ev}_rev*.parquet"))
    if revs:
        # a policy scored in several files keeps the rows of the first _rev file that has it (the _rev files also
        # count passive fills); the replay is deterministic, so its costs are the same everywhere
        rev = []
        for f in revs:
            r = pd.read_parquet(f)
            seen = set().union(*(set(x["policy"].unique()) for x in rev)) if rev else set()
            rev.append(r[~r["policy"].isin(seen)])
        rev = pd.concat(rev, ignore_index=True)
        parts[0] = parts[0][~parts[0]["policy"].isin(rev["policy"].unique())]
        parts.append(rev)
    res = pd.concat(parts, ignore_index=True)
    res = res[~res["excluded"]]
    info_cols = [c for c in ("stock", "day", "Q", "w_mid", "w_spread", "tick", "eta", "sigma_trade", "w_trades",
                             "w_seconds", "w_volume", "px_scale") if c in res]
    info = res.groupby("key")[info_cols].first()
    if len(parts) > 1:  # columns only the _rev evaluation stores
        extra = [c for c in ("eta", "sigma_trade", "w_trades", "w_seconds", "w_volume") if c in parts[1]]
        info[extra] = parts[1].groupby("key")[extra].first().reindex(info.index)
    return res, info


def column_of(policy: pd.Series, seen: str, seen_id: str) -> pd.DataFrame:
    """'<tag>:<rep>' (seen and identity tags renamed 'seen' and 'id'; baselines 'base:<name>') and the seed."""
    u = pd.Series(policy.unique())
    m = u.str.extract(NAME)
    tag = m["tag"].fillna("base").replace({seen: "seen", seen_id: "id", seen_id + "r": "idr"})
    col = dict(zip(u, tag + ":" + m["rep"].fillna(u)))
    seed = dict(zip(u, pd.to_numeric(m["seed"]).fillna(-1).astype(int)))
    return pd.DataFrame({"col": policy.map(col), "seed": policy.map(seed)})


def per_stock_seed(res: pd.DataFrame, col: str, value: str = "unit_cost") -> pd.DataFrame:
    """Stocks x seeds table of mean cost of policy column `col` (in the evaluation's unit: spreads for stocks,
    volatility per trade for crypto)."""
    g = res[res["col"] == col]
    return g.groupby(["stock", "seed"])[value].mean().unstack("seed")


def strata_of(info: pd.DataFrame, stocks: pd.Index) -> np.ndarray | None:
    """Relative-tick tercile of each stock (frozen cuts), when the episodes' relative ticks are known; none for
    crypto pairs, whose analysis is not stratified."""
    if not STRATIFY or "eta" not in info or info["eta"].isna().all():
        return None
    cuts = load_splits()["eta_sigma_tercile_cuts"]
    eta = info.groupby("stock")["eta"].median().reindex(stocks)
    return np.digitize(np.log(eta.to_numpy()), np.log(cuts))


def contrast(res, info, a, b, stocks_sel, n_boot, level=0.90, seed=0) -> dict | None:
    """a minus b, seeds averaged: stock-cluster bootstrap (tercile strata when known), per-seed estimates,
    a stock-and-seed bootstrap (training variance) and a stock-by-day bootstrap."""
    A, B = per_stock_seed(res, a), per_stock_seed(res, b)
    idx = A.index.intersection(B.index)
    if stocks_sel is not None:
        idx = idx[idx.isin(stocks_sel)]
    if len(idx) < 10 or A.empty or B.empty:
        return None
    A, B = A.loc[idx], B.loc[idx]
    by_stock = A.mean(axis=1) - B.mean(axis=1)
    out = {"stocks": int(len(idx)), "seeds_a": [int(s) for s in A.columns], "seeds_b": [int(s) for s in B.columns]}
    out["stock"] = cluster_bootstrap(by_stock.to_numpy(), strata_of(info, idx), n_boot=n_boot, level=level, seed=seed,
                                     return_boots=True)
    boots = out["stock"].pop("boots")
    out["p_two_sided"] = float(min(1.0, 2 * min((1 + np.sum(boots <= 0)), (1 + np.sum(boots >= 0))) / (1 + n_boot)))
    common = [s for s in A.columns if s in B.columns and s >= 0]
    out["per_seed"] = {int(s): float((A[s] - B[s]).mean()) for s in common}
    if len(common) > 1:
        out["seed_sd"] = float(np.std(list(out["per_seed"].values()), ddof=1))
    # stock-and-seed bootstrap: stocks with replacement, and each policy's seeds with replacement
    rng = np.random.default_rng(seed + 1)
    a_, b_ = A.to_numpy(), B.to_numpy()
    est = np.empty(n_boot)
    for i in range(n_boot):
        s_ = rng.integers(0, len(idx), len(idx))
        ca = rng.integers(0, a_.shape[1], a_.shape[1])
        cb = rng.integers(0, b_.shape[1], b_.shape[1])
        est[i] = np.nanmean(np.nanmean(a_[s_][:, ca], axis=1) - np.nanmean(b_[s_][:, cb], axis=1))
    q = (1 - level) / 2
    out["stock_seed"] = {"lo": float(np.quantile(est, q)), "hi": float(np.quantile(est, 1 - q))}
    # stock-by-day bootstrap of the seed-averaged per-episode difference (stocks weighted equally)
    ea = res[res["col"] == a].groupby("key")["unit_cost"].mean()
    eb = res[res["col"] == b].groupby("key")["unit_cost"].mean()
    d = (ea - eb).dropna()
    d = d[info.loc[d.index, "stock"].isin(idx).to_numpy()]
    cm = d.groupby([info.loc[d.index, "stock"].to_numpy(), info.loc[d.index, "day"].to_numpy()]).mean().unstack()
    w = cm.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    tw = pigeonhole_bootstrap((cm.fillna(0.0) * w).to_numpy(), w.to_numpy(), n_boot=n_boot, level=level, seed=seed)
    out["stock_day"] = {"lo": tw["lo"], "hi": tw["hi"], "days": int(cm.shape[1])}
    lo, hi = out["stock"]["lo"], out["stock"]["hi"]
    out["margin"] = ("inside" if (lo > -DELTA and hi < DELTA) else
                     "beyond" if (lo > DELTA or hi < -DELTA) else "straddles")
    return out


def economics(res, info, a, b, n_boot) -> dict | None:
    """a minus b in basis points: per-stock means (stocks weighted equally), net of fees, and notional-weighted
    (total cost over total notional), each with a stock-cluster bootstrap 95% interval. Only rows that count
    passive fills (the _rev evaluation) are used, so every figure here comes from the same seeds."""
    if "passive" not in res:
        return None
    ok = res["passive"].notna()
    ea = res[ok & (res["col"] == a)].groupby("key")[["cost", "passive"]].mean()
    eb = res[ok & (res["col"] == b)].groupby("key")[["cost", "passive"]].mean()
    k = ea.index.intersection(eb.index)
    if len(k) == 0:
        return None
    I = info.loc[k]
    notional = I["Q"] * I["w_mid"]
    net = lambda e: e["cost"] + TAKER * (I["Q"] - e["passive"]) - REBATE * e["passive"]  # noqa: E731
    d_gross = 1e4 * (ea.loc[k, "cost"] - eb.loc[k, "cost"]) / notional
    d_net = 1e4 * (net(ea.loc[k]) - net(eb.loc[k])) / notional
    st = I["stock"]
    out = {}
    for name, d in (("bps", d_gross), ("bps_net", d_net)):
        out[name] = cluster_bootstrap(d.groupby(st).mean().to_numpy(), n_boot=n_boot)
    # notional-weighted: ratio of sums over stocks
    num = (1e4 * (ea.loc[k, "cost"] - eb.loc[k, "cost"])).groupby(st).sum()
    den = notional.groupby(st).sum()
    rng = np.random.default_rng(3)
    nb, db = num.to_numpy(), den.to_numpy()
    boots = np.array([(lambda i: nb[i].sum() / db[i].sum())(rng.integers(0, len(nb), len(nb))) for _ in range(n_boot)])
    out["bps_notional"] = {"mean": float(nb.sum() / db.sum()), "lo": float(np.quantile(boots, 0.025)),
                           "hi": float(np.quantile(boots, 0.975))}
    out["passive_share_a"] = float((ea.loc[k, "passive"] / I["Q"]).mean())
    out["passive_share_b"] = float((eb.loc[k, "passive"] / I["Q"]).mean())
    out["seeds_used"] = sorted(int(s) for s in res.loc[ok & (res["col"] == a), "seed"].unique())
    return out


CRYPTO_BPS = [("ms+tick vs immediate", "zero:R3tick", "base:immediate"),
              ("ms+tick vs post", "zero:R3tick", "base:post_then_cross"),
              ("raw vs ms+tick", "zero:R0", "zero:R3tick"),
              ("ms+tick vs crypto-trained", "zero:R3tick", "cbseen:R3tick")]


def crypto_bps(res, info, n_boot) -> dict:
    """Crypto contrasts in basis points of the price (gross), seeds averaged, pairs weighted equally, with a
    pair-cluster bootstrap 95% interval."""
    out = {}
    for name, a, b in CRYPTO_BPS:
        ea = res[res["col"] == a].groupby("key")["c_bps"].mean()
        eb = res[res["col"] == b].groupby("key")["c_bps"].mean()
        d = (ea - eb).dropna()
        if d.empty:
            continue
        by = d.groupby(info.loc[d.index, "stock"].to_numpy()).mean()
        out[name] = cluster_bootstrap(by.to_numpy(), n_boot=n_boot)
    return out


def bh_family(res, info, cols, seen_stocks, n_boot) -> dict:
    """Every zero-shot representation against MS units, and every transfer gap, overall and by tercile:
    two-sided bootstrap p-values, Benjamini-Hochberg at q = 0.10."""
    tests = []
    reps = sorted({c.split(":")[1] for c in cols if c.startswith("zero:")} - {"R3"})
    pairs = [(f"contrast {r}", f"zero:{r}", "zero:R3", None) for r in reps]
    pairs += [(f"gap {r}", f"zero:{r}", f"seen:{r}", seen_stocks) for r in sorted({c.split(":")[1] for c in cols
                                                                                   if c.startswith("seen:")})]
    for name, a, b, sel in pairs:
        A, B = per_stock_seed(res, a).mean(axis=1), per_stock_seed(res, b).mean(axis=1)
        d = (A - B).dropna()
        if sel is not None:
            d = d[d.index.isin(sel)]
        if len(d) < 10:
            continue
        st = strata_of(info, d.index)
        groups = [("all", np.ones(len(d), bool))]
        if st is not None:
            groups += [(t, st == i) for i, t in enumerate(("small", "mid", "large"))]
        for g, m in groups:
            if m.sum() < 10:
                continue
            r = cluster_bootstrap(d.to_numpy()[m], n_boot=n_boot, return_boots=True)
            bt = r.pop("boots")
            p = min(1.0, 2 * min(1 + np.sum(bt <= 0), 1 + np.sum(bt >= 0)) / (1 + n_boot))
            tests.append({"test": f"{name} {g}", "mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "p": float(p)})
    if tests:
        q = bh([t["p"] for t in tests])
        for t, qq in zip(tests, q):
            t["q"] = float(qq)
    return {"tests": tests, "n": len(tests), "discoveries": int(sum(t["q"] < 0.10 for t in tests))}


def extrapolation_abs(res, info, n_boot) -> dict:
    """Absolute contrasts on Stage 1 stocks outside each training range: restricted raw minus restricted
    MS+tick (and the same for the reference policies trained on all training stocks)."""
    out = {}
    for stat in ("activity", "tradesize", "spread"):
        fitf = OUT / f"xtrap_{stat}_fit.json"
        if not fitf.exists():
            continue
        fit = json.loads(fitf.read_text())
        if stat == "activity":
            x = info["w_trades"] / info["w_seconds"]
        elif stat == "tradesize":
            x = info["w_volume"] / info["w_trades"].clip(lower=1)
        else:
            x = info["w_spread"] / (info["px_scale"] if "px_scale" in info else 1e4)
        s = x.groupby(info["stock"]).median()
        where = pd.Series(np.where(s < fit["q25"], "below", np.where(s > fit["q75"], "above", "inside")),
                          index=s.index)
        row = {}
        # the extrapolation gap of each representation (restricted minus reference), seeds averaged
        for rep in ("R0", "R0r", "R3", "R3tick"):
            A = per_stock_seed(res, f"x{stat}:{rep}").mean(axis=1)
            B = per_stock_seed(res, f"xref{stat}:{rep}").mean(axis=1)
            d = (A - B).dropna()
            for g in ("outside", "below", "above"):
                sel = where.reindex(d.index)
                m = (sel != "inside") if g == "outside" else (sel == g)
                if m.sum() >= 10:
                    row[f"gap {rep} {g}"] = cluster_bootstrap(d[m.to_numpy()].to_numpy(), n_boot=n_boot)
        for kind, tag in (("restricted", f"x{stat}"), ("reference", f"xref{stat}")):
            for rep in ("R0", "R3"):
                A = per_stock_seed(res, f"{tag}:{rep}").mean(axis=1)
                B = per_stock_seed(res, f"{tag}:R3tick").mean(axis=1)
                d = (A - B).dropna()
                for g in ("outside", "below", "above", "inside"):
                    sel = where.reindex(d.index)
                    m = (sel != "inside") if g == "outside" else (sel == g)
                    if m.sum() >= 10:
                        row[f"{kind} {rep} {g}"] = cluster_bootstrap(d[m.to_numpy()].to_numpy(), n_boot=n_boot)
        out[stat] = {"stocks": where.value_counts().to_dict(), "contrasts_vs_ms_tick": row}
    return out


def tg_seen(res, info, seen_stocks, n_boot) -> dict:
    """Transfer gap of every representation (zero-shot minus seen, seeds 0-2 as in the registered analysis) on the
    stocks whose own 2019 days the seen policy saw, overall (stratified by relative-tick tercile) and per tercile."""
    out = {}
    if seen_stocks is None:
        return out
    reps = sorted({c.split(":", 1)[1] for c in res["col"].unique() if c.startswith("zero:")})
    for rep in reps:
        if f"seen:{rep}" not in set(res["col"].unique()):
            continue
        A = per_stock_seed(res, f"zero:{rep}")
        B = per_stock_seed(res, f"seen:{rep}")
        A, B = A[[c for c in A.columns if c <= 2]], B[[c for c in B.columns if c <= 2]]
        d = (A.mean(axis=1) - B.mean(axis=1)).dropna()
        d = d[d.index.isin(seen_stocks)]
        if len(d) < 10:
            continue
        st = strata_of(info, d.index)
        r = {"all": cluster_bootstrap(d.to_numpy(), st, n_boot=n_boot), "stocks": int(len(d))}
        if st is not None:
            for t, name in enumerate(("small", "mid", "large")):
                if (st == t).sum() >= 5:
                    r[name] = cluster_bootstrap(d.to_numpy()[st == t], n_boot=n_boot)
        out[rep] = r
    return out


def h1_testsplit(res, info, seen_stocks, n_boot) -> dict | None:
    """H1 (zero-shot minus seen MS+tick, 90% interval) on the test-split stocks only, the stock set the registration
    names in parentheses; the registered analysis uses every stock whose 2019 days the seen policy saw."""
    if seen_stocks is None:
        return None
    split = load_splits()["split"]
    test = pd.Index([s for s in seen_stocks if split.get(s) == "test"])
    return contrast(res, info, "zero:R3tick", "seen:R3tick", test, n_boot)


def macros_of(out: dict, P: str) -> dict:
    """The paper's macros, all from the stored results (so they can be rewritten without re-running the analysis)."""
    macros = {}
    for hid, r in out["hypotheses"].items():
        H = mname(hid)
        macros[P + H + "Est"] = f"{r['stock']['mean']:+.3f}"
        for tag, key in (("Stock", "stock"), ("Seed", "stock_seed"), ("Day", "stock_day")):
            macros[P + H + tag + "Lo"], macros[P + H + tag + "Hi"] = (f"{r[key][x]:+.3f}" for x in ("lo", "hi"))
        macros[P + H + "Seeds"] = str(min(len(r["seeds_a"]), len(r["seeds_b"])))
        if "seed_sd" in r:
            macros[P + H + "SeedSd"] = f"{r['seed_sd']:.4f}"
        macros[P + H + "Margin"] = r["margin"]
    if "cost_unit_share" in out:
        macros[P + "CostUnitShare"] = f"{100 * out['cost_unit_share']:.0f}\\%"
    arms = out["arms"]
    for name, r in arms.items():  # intervals resample stocks and training seeds together
        M = P + mname("arm", name)
        iv = r.get("stock_seed", r["stock"])
        macros[M] = f"{r['stock']['mean']:+.3f}"
        macros[M + "Lo"], macros[M + "Hi"] = (f"{iv[k]:+.3f}" for k in ("lo", "hi"))
    if "nn raw vs raws" in arms and "nn raw vs ms" in arms:  # the cost unit's share of the neural learner's loss
        share = arms["nn raw vs raws"]["stock"]["mean"] / arms["nn raw vs ms"]["stock"]["mean"]
        macros[P + "NnCostUnitShare"] = f"{100 * share:.0f}\\%" if share < 0.995 else "about 100\\%"
    for name, r in out["economics"].items():
        M = P + mname("econ", name)
        macros[M + "PassiveA"], macros[M + "PassiveB"] = (f"{100 * r[k]:.0f}\\%" for k in
                                                          ("passive_share_a", "passive_share_b"))
        for k in ("bps", "bps_net", "bps_notional"):
            macros[M + mname(k)] = f"{r[k]['mean']:+.2f}"
            macros[M + mname(k) + "Lo"], macros[M + mname(k) + "Hi"] = (f"{r[k][z]:+.2f}" for z in ("lo", "hi"))
    for name, r in out.get("crypto_bps", {}).items():
        M = P + mname("bps", name)
        macros[M], macros[M + "Lo"], macros[M + "Hi"] = (f"{r[z]:+.2f}" for z in ("mean", "lo", "hi"))
    if "bh" in out:
        macros[P + "BhTests"], macros[P + "BhDiscoveries"] = str(out["bh"]["n"]), str(out["bh"]["discoveries"])
    if out.get("h1_testsplit"):
        r = out["h1_testsplit"]
        macros[P + "HOneTestSplit"] = f"{r['stock']['mean']:+.3f}"
        macros[P + "HOneTestSplitLo"], macros[P + "HOneTestSplitHi"] = (f"{r['stock'][x]:+.3f}" for x in ("lo", "hi"))
        macros[P + "HOneTestSplitStocks"] = f"{r['stocks']:,}".replace(",", "{,}")
    for rep, r in out.get("tg_seen", {}).items():
        macros[P + mname("tg seen", rep) + "Stocks"] = f"{r['stocks']:,}".replace(",", "{,}")
        for g in ("all", "small", "mid", "large"):
            if g in r:
                M = P + mname("tg seen", rep, g)
                macros[M], macros[M + "Lo"], macros[M + "Hi"] = (f"{r[g][z]:+.3f}" for z in ("mean", "lo", "hi"))
    for stat, v in out.get("extrapolation_abs", {}).items():
        for k, r in v["contrasts_vs_ms_tick"].items():
            M = P + mname("xabs", stat, k)
            macros[M], macros[M + "Lo"], macros[M + "Hi"] = (f"{r[z]:+.3f}" for z in ("mean", "lo", "hi"))
    return macros


def analyse(a) -> dict:
    global DELTA, STRATIFY
    crypto = a.eval.startswith("crypto")
    seen_tag, id_tag, seen_data = SEEN.get(a.eval, ("seen", "seenid", ""))
    res, info = load(a.eval)
    res = pd.concat([res.reset_index(drop=True), column_of(res["policy"].reset_index(drop=True), seen_tag, id_tag)],
                    axis=1)
    if crypto:  # the registered crypto unit: volatility per trade (known per episode from the _rev evaluation)
        DELTA, STRATIFY = 0.10, False
        sig = res["key"].map(info["sigma_trade"]) if "sigma_trade" in info else np.nan
        res["unit_cost"] = res["cost"] / (res["Q"] * sig)
    else:
        res["unit_cost"] = res["c_spread"]
    cols = set(res["col"].unique())
    seen_stocks = None
    if seen_data and (DATA / "transitions" / seen_data / "eps.parquet").exists():
        seen_stocks = pd.Index(pd.read_parquet(DATA / "transitions" / seen_data / "eps.parquet",
                                               columns=["stock"])["stock"].unique())
    if getattr(a, "only_arms", False):  # recompute the learner, normalisation and identity arms (e.g. new seeds)
        out = json.loads((OUT / f"revision_{a.eval}.json").read_text())
        out["arms"] = {}
        for name, x, y, sel in ARMS:
            if x in cols and y in cols:
                r = contrast(res, info, x, y, seen_stocks if sel == "seen" else None, a.n_boot, level=0.95)
                if r is not None:
                    out["arms"][name] = r
        write_json(OUT / f"revision_{a.eval}.json", out)
        return out
    if getattr(a, "only_tg", False):  # add the seen-stock transfer gaps (and H1 on test-split stocks) to the results
        out = json.loads((OUT / f"revision_{a.eval}.json").read_text())
        out["tg_seen"] = tg_seen(res, info, seen_stocks, a.n_boot)
        out["h1_testsplit"] = h1_testsplit(res, info, seen_stocks, a.n_boot)
        write_json(OUT / f"revision_{a.eval}.json", out)
        return out
    spec = json.loads((REPO / "prereg" / "confirmatory_v1.json").read_text())
    out = {"eval": a.eval, "episodes": int(info.shape[0]), "stocks": int(info["stock"].nunique()),
           "hypotheses": {}, "arms": {}, "economics": {}, "provenance": provenance(args=vars(a))}
    for h in spec["families"]["crypto" if crypto else "stocks"]["hypotheses"]:
        if h["a"] not in cols or h["b"] not in cols:
            continue
        r = contrast(res, info, h["a"], h["b"], seen_stocks if h.get("stocks") == "seen" else None, a.n_boot)
        if r is not None:
            out["hypotheses"][h["id"]] = r
    if "H2" in out["hypotheses"] and "H9" in out["hypotheses"]:
        out["cost_unit_share"] = out["hypotheses"]["H9"]["stock"]["mean"] / out["hypotheses"]["H2"]["stock"]["mean"]
    for name, x, y, sel in ARMS:
        if x not in cols or y not in cols:
            continue
        r = contrast(res, info, x, y, seen_stocks if sel == "seen" else None, a.n_boot, level=0.95)
        if r is not None:
            out["arms"][name] = r
    if "Q" in info and "cost" in res and not crypto:
        for name, x, y in ECON:
            if x not in cols or y not in cols:
                continue
            r = economics(res, info, x, y, a.n_boot)
            if r is not None:
                out["economics"][name] = r
    if crypto:
        out["crypto_bps"] = crypto_bps(res, info, a.n_boot)
    if not crypto:
        out["bh"] = bh_family(res, info, cols, seen_stocks, a.n_boot)
    if a.eval == "stage1" and "w_trades" in info:
        out["extrapolation_abs"] = extrapolation_abs(res, info, a.n_boot)
    out["tg_seen"] = tg_seen(res, info, seen_stocks, a.n_boot)
    out["h1_testsplit"] = h1_testsplit(res, info, seen_stocks, a.n_boot)
    write_json(OUT / f"revision_{a.eval}.json", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--only-arms", action="store_true",
                    help="recompute only the learner, normalisation and identity arms in the stored results")
    ap.add_argument("--only-tg", action="store_true",
                    help="compute only the seen-stock transfer gaps and add them to the stored results")
    ap.add_argument("--from-json", action="store_true",
                    help="rewrite the macros from results/transfer/revision_<eval>.json without re-running the analysis")
    a = ap.parse_args(argv)
    if a.from_json:
        out = json.loads((OUT / f"revision_{a.eval}.json").read_text())
    else:
        out = analyse(a)
    P = PREFIX.get(a.eval, mname("rev", a.eval))
    macros = macros_of(out, P)
    with open(REPO / "paper" / f"numbers_revision_{a.eval}.tex", "w") as fh:
        fh.write(f"% generated by experiments/analyze_revision.py --eval {a.eval} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for hid, r in out["hypotheses"].items():
        print(f"{hid}: {r['stock']['mean']:+.4f} stock [{r['stock']['lo']:+.4f}, {r['stock']['hi']:+.4f}] "
              f"stock+seed [{r['stock_seed']['lo']:+.4f}, {r['stock_seed']['hi']:+.4f}] "
              f"stock x day [{r['stock_day']['lo']:+.4f}, {r['stock_day']['hi']:+.4f}] "
              f"seeds {len(r['seeds_a'])}/{len(r['seeds_b'])} sd {r.get('seed_sd', float('nan')):.4f} {r['margin']}")
    for name, r in out["arms"].items():
        print(f"  arm {name}: {r['stock']['mean']:+.4f} [{r['stock']['lo']:+.4f}, {r['stock']['hi']:+.4f}] "
              f"({r['stocks']} stocks)")
    for name, r in out["economics"].items():
        print(f"  econ {name}: {r['bps']['mean']:+.2f} bp, net {r['bps_net']['mean']:+.2f}, "
              f"notional-weighted {r['bps_notional']['mean']:+.2f}; passive {r['passive_share_a']:.2f}/"
              f"{r['passive_share_b']:.2f}")
    for name, r in out.get("crypto_bps", {}).items():
        print(f"  crypto bps {name}: {r['mean']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")
    if "bh" in out:
        print(f"  BH: {out['bh']['discoveries']} of {out['bh']['n']} exploratory tests at q = 0.10")


if __name__ == "__main__":
    main()

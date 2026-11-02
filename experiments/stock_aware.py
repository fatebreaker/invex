"""A stock-aware comparator (fourth review round): a pooled policy plus per-stock corrections learned from the
stock's own days, kept only as far as they predict the stock's held-out days.

The pooled model's advantage A_k(s, a) gets a per-stock offset b[stock, cell, a] for each non-crossing action, in
four cells (first or second half of the decisions x an order resting or not):

    A~_k(s, a) = A_k(s, a) + b[stock, cell(k, s), a]

The offsets are shrunken means of the stock's fitted-Q residuals y - A on its own training days, with the targets
y of the fitted model (one-step costs plus the corrected continuation), b = sum(residual) / (count + lambda). The
shrinkage lambda is chosen by leave-one-day-out cross-validation over the stocks' five 2019 days: offsets fitted on
four days, the corrected policy replayed on the fifth, and the lambda with the lowest replayed cost kept, from a grid
that includes infinity (no correction). The comparator thus uses a stock's data only as far as that lowers its cost
on other days (--cv residual: by the squared residual instead). Then two passes recompute the targets with the
corrected continuation.

    # offsets for the validation stocks (their 2019 training days), checked on the held-out validation days
    python experiments/stock_aware.py fit  --models zero_R3tick_s0 zero_R0_s0 --data valseen_v0
    python experiments/stock_aware.py eval --episodes val_v0 --models zero_R3tick_s0 zero_R0_s0 --data valseen_v0 \
        --out val_v0_aware
    # the sealed stages, with offsets from the test stocks' 2019 days (logged)
    python experiments/stock_aware.py eval --episodes stage1_v0 --unseal --reason "..." --data testseen_v0 ...

Offsets: results/transfer/aware/<model>__<data>.npz. Evaluation: results/transfer/eval_<out>.parquet with the
pooled policy (<model>) and the corrected one (aware_<model>) on the same episodes. Exploratory, post hoc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer as TR  # noqa: E402

from invex.agents import fqi_adv  # noqa: E402
from invex.agents.fqi import rep_inputs  # noqa: E402
from invex.env import execution as ex  # noqa: E402
from invex.env.episodes import excluded  # noqa: E402
from invex.env.features import F_R_BEHIND, F_R_IN, F_R_NEAR, REWARD_SYSTEM, FlowBook, r1_stats  # noqa: E402
from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval import policies as P  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402

K = TR.K
ACTS = np.array(fqi_adv.NON_CROSS)
N_CELL = 4
LAMBDAS = (0.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, np.inf)
AWARE = TR.OUT / "aware"


def cells(k: int, raw: np.ndarray) -> np.ndarray:
    resting = (raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND]) > 0
    return 2 * int(k >= K // 2) + resting.astype(np.int64)


class AwarePolicy:
    """Greedy on the pooled advantages plus the stock's offsets (zero for a stock without offsets). Offsets are in
    the model's reward unit, or (in_spreads) in the stock's mean spread, converted per episode."""

    def __init__(self, model, stocks: np.ndarray, offsets: np.ndarray, name: str, in_spreads: bool = False):
        self.model, self.name, self.in_spreads = model, name, in_spreads
        self.index = {s: i for i, s in enumerate(stocks)}
        self.offsets = offsets  # (stocks, N_CELL, len(ACTS))

    def act(self, k, raw, masks, eps, uf):
        X = rep_inputs(raw, eps, uf, self.model.rep, self.model.c_lookback)
        q = self.model.q_values(k, X)
        i = eps["stock"].map(self.index).fillna(-1).to_numpy(np.int64)
        c = cells(k, raw)
        b = np.where((i >= 0)[:, None], self.offsets[np.maximum(i, 0), c], 0.0)
        if self.in_spreads:
            b = b * (uf["R3"]["p"].to_numpy() / uf[REWARD_SYSTEM[self.model.rep]]["p"].to_numpy())[:, None]
        q[:, ACTS] += b
        q[masks == 0] = np.inf
        return np.argmin(q, axis=1)


# --- fit ------------------------------------------------------------------------------------------

def base_arrays(model, data: dict, eps: pd.DataFrame, uf: dict):
    """Pooled advantages at the visited states A[k] (n, N_ACT) and at the next states An[k][a], with cells."""
    rep, c_lb = model.rep, model.c_lookback
    A = np.stack([model.q_values(k, rep_inputs(data["raw"][k], eps, uf, rep, c_lb)) for k in range(K)])
    cell = np.stack([cells(k, data["raw"][k]) for k in range(K)])
    An = np.zeros((K - 1, len(ACTS), len(eps), ex.N_ACT))
    cell_n = np.zeros((K - 1, len(ACTS), len(eps)), np.int64)
    for k in range(K - 1):
        for j, a in enumerate(ACTS):
            An[k, j] = model.q_values(k + 1, rep_inputs(data["raw_next"][k, a], eps, uf, rep, c_lb))
            cell_n[k, j] = cells(k + 1, data["raw_next"][k, a])
    return A, cell, An, cell_n


def residuals(data, norm, A, An, cell_n, si, b):
    """Residuals y - A (K, n, len(ACTS)) of the fitted targets with the offset-corrected continuation; NaN where
    the action was not valid."""
    n = A.shape[1]
    r = np.full((K, n, len(ACTS)), np.nan)
    for k in range(K):
        c_now = data["costs"][k][:, ex.CROSS]
        for j, a in enumerate(ACTS):
            y = (data["costs"][k][:, a] - c_now) / norm
            if k < K - 1:
                q = An[k, j].copy()
                q[:, ACTS] += b[si, cell_n[k, j]]
                mn = data["masks_next"][k, a].astype(bool)
                q[~mn] = np.inf
                q[:, ex.CROSS] = 0.0
                v = data["cross_next"][k, a] / norm + np.minimum(0.0, q.min(axis=1))
                y = y + np.where(data["done_next"][k][:, a] == 1, 0.0, v)
            ok = data["masks"][k][:, a] == 1
            r[k, ok, j] = (y - A[k][:, a])[ok]
    return r


def sums(r, cell, si, di, n_stock, n_day):
    """Residual sums and counts by (stock, day, cell, action)."""
    S = np.zeros((n_stock, n_day, N_CELL, len(ACTS)))
    N = np.zeros_like(S)
    for k in range(K):
        for j in range(len(ACTS)):
            ok = ~np.isnan(r[k, :, j])
            np.add.at(S, (si[ok], di[ok], cell[k][ok], j), r[k, ok, j])
            np.add.at(N, (si[ok], di[ok], cell[k][ok], j), 1.0)
    return S, N


def shrunk(S, N, lam):
    if np.isinf(lam):
        return np.zeros_like(S)
    return S / (N + lam + 1e-12)


def cv_lambda(r, cell, si, di, n_stock, n_day):
    """Leave-one-day-out squared error of the residuals around the offsets, per lambda (summed over everything)."""
    S, N = sums(r, cell, si, di, n_stock, n_day)
    S_all, N_all = S.sum(axis=1), N.sum(axis=1)
    # squared error on day d with offsets b from the other days: sum (r - b)^2 = sum r^2 - 2 b S_d + b^2 N_d
    R2 = np.zeros((n_stock, n_day, N_CELL, len(ACTS)))
    for k in range(K):
        for j in range(len(ACTS)):
            ok = ~np.isnan(r[k, :, j])
            np.add.at(R2, (si[ok], di[ok], cell[k][ok], j), r[k, ok, j] ** 2)
    out = {}
    for lam in LAMBDAS:
        b = shrunk(S_all[:, None] - S, N_all[:, None] - N, lam)
        out[float(lam)] = float((R2 - 2 * b * S + b ** 2 * N).sum())
    return out


def cv_task(args):
    """Replay one stock-day of the training transitions under the offsets fitted without that day, one policy per
    lambda; returns the cost per lambda in spreads per share (summed) and the episode count."""
    day, stock, eps, name, offsets, ko = args
    venue = str(eps["venue"].iloc[0])
    book, _o, _c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True).copy()
    uf = unit_frame(eps, ko)
    model = TR._model(str(TR.OUT / "models" / f"{name}.pkl"))
    pols = [AwarePolicy(model, np.array([stock]), offsets[i][None], name=str(i)) for i in range(len(offsets))]
    res = P.evaluate_book(book, eps, uf, pols, K, venue_of(venue).latency_ns, tick=tick)
    ex_ = excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy())
    res = res.merge(pd.DataFrame({"key": eps["key"], "norm": eps["Q"] * eps["w_spread"], "ex": ex_}), on="key")
    res = res[~res["ex"]]
    return (res["cost"] / res["norm"]).groupby(res["policy"].astype(int)).sum().reindex(range(len(offsets)),
                                                                                       fill_value=0.0).to_numpy()


def cv_policy(name, eps, S, N, stocks, days, ko, workers):
    """Leave-one-day-out cost of the corrected policy per lambda: offsets from four days, replayed on the fifth."""
    S_all, N_all = S.sum(axis=1), N.sum(axis=1)
    si = {s: i for i, s in enumerate(stocks)}
    jobs = []
    for (_v, d, s), g in eps.groupby(["venue", "day", "stock"]):
        i, j = si[s], int(np.searchsorted(days, str(d)))
        offs = np.stack([shrunk(S_all[i] - S[i, j], N_all[i] - N[i, j], lam) for lam in LAMBDAS])
        jobs.append((d, s, g, name, offs, ko))
    tot = np.sum(TR.pool(cv_task, jobs, workers), axis=0)
    return {float(lam): float(c) for lam, c in zip(LAMBDAS, tot)}


def cmd_fit(a):
    ko = load_splits()["ko_constants"]
    eps, data = TR.load_transitions([a.data])
    uf = unit_frame(eps, ko)
    stocks = np.array(sorted(eps["stock"].unique()))
    days = np.array(sorted(eps["day"].astype(str).unique()))
    si = pd.Series(np.arange(len(stocks)), index=stocks)[eps["stock"]].to_numpy()
    di = pd.Series(np.arange(len(days)), index=days)[eps["day"].astype(str)].to_numpy()
    AWARE.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        t0 = time.time()
        model = fqi_adv.load_model(TR.OUT / "models" / f"{name}.pkl")
        norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[model.rep]]["p"].to_numpy()
        A, cell, An, cell_n = base_arrays(model, data, eps, uf)
        b = np.zeros((len(stocks), N_CELL, len(ACTS)))
        r = residuals(data, norm, A, An, cell_n, si, b)
        if a.cv == "policy":
            S, N = sums(r, cell, si, di, len(stocks), len(days))
            cv = cv_policy(name, eps, S, N, stocks, days, ko, a.workers)
        else:
            cv = cv_lambda(r, cell, si, di, len(stocks), len(days))
        lam = min(cv, key=cv.get)
        for _ in range(a.passes if np.isfinite(lam) else 0):
            S, N = sums(r, cell, si, di, len(stocks), len(days))
            b = shrunk(S.sum(axis=1), N.sum(axis=1), lam)
            r = residuals(data, norm, A, An, cell_n, si, b)
        np.savez(AWARE / f"{name}__{a.data}.npz", stocks=stocks, offsets=b, lam=lam, cv=a.cv,
                 cv_lambda=np.array(list(cv)), cv_sse=np.array(list(cv.values())))
        rel = {k: v / cv[float(np.inf)] - 1 for k, v in cv.items()}
        print(f"{name} on {a.data}: lambda {lam:g}; CV error relative to no correction "
              + ", ".join(f"{k:g}: {v:+.4%}" for k, v in rel.items())
              + f"; |b| mean {np.abs(b).mean():.4f}; {time.time() - t0:.0f}s", flush=True)


# --- per-stock patience ---------------------------------------------------------------------------

def episodes_task(args):
    venue, day, stock, n, ko, tag = args
    from invex.env.episodes import book_episodes
    from invex.env.units import valid_units
    from invex.eval.freeze import scales_of

    book, o, c, tick = book_of(venue, day, stock)
    eps = book_episodes(book, venue, day, stock, o, c, n, ko, seed_tag=tag, tick=tick, **scales_of(venue, day, stock))
    if eps.empty:
        return None
    return eps[valid_units(unit_frame(eps, ko))].reset_index(drop=True)


def cmd_episodes(a):
    """A dense episode table (no transitions) on some stocks' training days, for tuning per-stock patience."""
    from invex.eval.freeze import stocks_for

    ko = load_splits()["ko_constants"]
    jobs = [("itch", d, s, a.per_book, ko, a.name) for d in a.days for s in stocks_for(a.split, d, "itch")]
    eps = pd.concat(TR.pool(episodes_task, jobs, a.workers), ignore_index=True)
    (TR.TRANS / a.name).mkdir(parents=True, exist_ok=True)
    eps.to_parquet(TR.TRANS / a.name / "eps.parquet", index=False)
    write_json(TR.TRANS / a.name / "meta.json", {"name": a.name, "split": a.split, "days": a.days,
                                                 "per_book": a.per_book, "episodes": len(eps), "transitions": False,
                                                 "provenance": provenance(args=vars(a))})
    print(f"{a.name}: {len(eps):,} episodes", flush=True)


SHIFTS = (-0.3, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.3)


def shift_task(args):
    """One stock-day of the training transitions replayed under every uniform shift (in spreads) of the pooled
    advantages; per shift: summed cost in spreads per share and the episode count."""
    day, stock, eps, name, ko = args
    venue = str(eps["venue"].iloc[0])
    book, _o, _c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True).copy()
    uf = unit_frame(eps, ko)
    model = TR._model(str(TR.OUT / "models" / f"{name}.pkl"))
    pols = [AwarePolicy(model, np.array([stock]), np.full((1, N_CELL, len(ACTS)), d), name=str(i), in_spreads=True)
            for i, d in enumerate(SHIFTS)]
    res = P.evaluate_book(book, eps, uf, pols, K, venue_of(venue).latency_ns, tick=tick)
    ex_ = excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy())
    res = res.merge(pd.DataFrame({"key": eps["key"], "norm": eps["Q"] * eps["w_spread"], "Q": eps["Q"], "ex": ex_}),
                    on="key")
    res = res[~res["ex"]]
    g = res.assign(c=res["cost"] / res["norm"], taken=1.0 - res["passive"] / res["Q"]).groupby(res["policy"].astype(int))
    s = g["c"].sum().reindex(range(len(SHIFTS)), fill_value=0.0)
    tk_ = g["taken"].sum().reindex(range(len(SHIFTS)), fill_value=0.0)  # share of each order taken (not rested)
    n = g.size().reindex(range(len(SHIFTS)), fill_value=0)
    return stock, str(day), s.to_numpy(), n.to_numpy(), tk_.to_numpy()


def cmd_fit_shift(a):
    """Per stock, the shift with the lowest cost on its own training days; its leave-one-day-out cost (the shift
    chosen on four days, scored on the fifth) against the pooled policy's, summed over stocks."""
    ko = load_splits()["ko_constants"]
    eps = pd.read_parquet(TR.TRANS / a.data / "eps.parquet")
    AWARE.mkdir(parents=True, exist_ok=True)
    i0 = SHIFTS.index(0.0)
    for name in a.models:
        t0 = time.time()
        jobs = [(d, s, g, name, ko) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
        out = TR.pool(shift_task, jobs, a.workers)
        stocks = np.array(sorted({o[0] for o in out}))
        days = sorted({o[1] for o in out})
        S = np.zeros((len(stocks), len(days), len(SHIFTS)))
        N, TK = np.zeros_like(S), np.zeros_like(S)
        si, di = {s: i for i, s in enumerate(stocks)}, {d: i for i, d in enumerate(days)}
        for s, d, ss, nn, tt in out:
            S[si[s], di[d]], N[si[s], di[d]], TK[si[s], di[d]] = ss, nn, tt
        mean = lambda s_, n_: np.where(n_ > 0, s_ / np.maximum(n_, 1), np.inf)  # noqa: E731
        cv, base, n_cv = 0.0, 0.0, 0.0
        for j in range(len(days)):
            pick = np.argmin(mean(S.sum(axis=1) - S[:, j], N.sum(axis=1) - N[:, j]), axis=1)
            cv += S[np.arange(len(stocks)), j, pick].sum()
            base += S[:, j, i0].sum()
            n_cv += N[:, j, i0].sum()
        best = np.array(SHIFTS)[np.argmin(mean(S.sum(axis=1), N.sum(axis=1)), axis=1)]
        offsets = np.repeat(np.repeat(best[:, None, None], N_CELL, axis=1), len(ACTS), axis=2)
        np.savez(AWARE / f"{name}__{a.data}__shift.npz", stocks=stocks, offsets=offsets, in_spreads=True,
                 shifts=np.array(SHIFTS), sums=S, counts=N, taken=TK, cv_cost=cv / n_cv, base_cost=base / n_cv)
        print(f"{name} on {a.data}: per-stock shift, leave-one-day-out cost {cv / n_cv:.4f} vs pooled "
              f"{base / n_cv:.4f} ({(cv - base) / n_cv:+.4f} spreads); shifts chosen: "
              + ", ".join(f"{d:g}: {int((best == d).sum())}" for d in SHIFTS) + f"; {time.time() - t0:.0f}s", flush=True)


def cmd_eval_shifts(a):
    """Every shift replayed on the held-out episodes of the stocks with offsets: per (stock, day, shift) summed cost,
    share taken and count, for the injected-structure check (`inject`)."""
    ko = load_splits()["ko_constants"]
    eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
    for name in a.models:
        z = np.load(AWARE / f"{name}__{a.data}__shift.npz")
        e = eps[eps["stock"].isin(set(z["stocks"]))]
        out = TR.pool(shift_task, [(d, s, g, name, ko) for (_v, d, s), g in e.groupby(["venue", "day", "stock"])],
                      a.workers)
        stocks, days = np.array(sorted({o[0] for o in out})), sorted({o[1] for o in out})
        S = np.zeros((len(stocks), len(days), len(SHIFTS)))
        N, TK = np.zeros_like(S), np.zeros_like(S)
        si, di = {s: i for i, s in enumerate(stocks)}, {d: i for i, d in enumerate(days)}
        for s, d, ss, nn, tt in out:
            S[si[s], di[d]], N[si[s], di[d]], TK[si[s], di[d]] = ss, nn, tt
        np.savez(AWARE / f"{name}__{a.data}__heldout_{a.episodes}.npz", stocks=stocks, sums=S, counts=N, taken=TK,
                 shifts=np.array(SHIFTS))
        print(f"{name}: {len(stocks)} stocks, {len(days)} held-out days", flush=True)


SCALES = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)


def cmd_inject(a):
    """Injected stock-specific structure with known truth: each stock gets a crossing surcharge phi (spreads per
    share taken, uniform on [-s/2, s/2], fixed across days) that no unit or feature reveals. The pooled policy
    ignores it; the per-stock shift, chosen on the stock's training days with the surcharge counted, can adapt.

    recovered(s): the comparator's held-out gain over the pooled policy (cost with the surcharge, stocks weighted
    equally), less its gain at s = 0, which is the comparator on the real data.
    achievable(s): what the best shift per stock could gain, estimated in sample on all of the stock's days
    (training and held-out pooled), less the same at s = 0, which removes most of the selection noise of picking the
    best of the shifts in sample. Both average over draws of phi."""
    from invex.eval.stats import cluster_bootstrap

    tr = np.load(AWARE / f"{a.model}__{a.data}__shift.npz")
    ho = np.load(AWARE / f"{a.model}__{a.data}__heldout_{a.episodes}.npz")
    common = np.intersect1d(tr["stocks"], ho["stocks"])
    it = pd.Series(np.arange(len(tr["stocks"])), index=tr["stocks"])[common].to_numpy()
    ih = pd.Series(np.arange(len(ho["stocks"])), index=ho["stocks"])[common].to_numpy()
    S, N, TK = (tr[k][it].sum(axis=1) for k in ("sums", "counts", "taken"))
    Sh, Nh, TKh = (ho[k][ih].sum(axis=1) for k in ("sums", "counts", "taken"))
    ok = (N.min(axis=1) > 0) & (Nh.min(axis=1) > 0)
    S, N, TK, Sh, Nh, TKh = (x[ok] for x in (S, N, TK, Sh, Nh, TKh))
    i0 = SHIFTS.index(0.0)
    rows = np.arange(len(S))

    def one(phi):
        pick = np.argmin((S + phi * TK) / N, axis=1)
        ch = (Sh + phi * TKh) / Nh
        cp = (S + Sh + phi * (TK + TKh)) / (N + Nh)
        return ch[rows, pick] - ch[:, i0], cp.min(axis=1) - cp[:, i0]

    g0, a0 = one(np.zeros((len(S), 1)))
    out = {"model": a.model, "stocks": int(ok.sum()), "draws": a.draws, "scales": {}}
    for s in SCALES:
        gains, ach = [], []
        for r in range(a.draws if s > 0 else 1):
            g, ac = one(np.random.default_rng(r).uniform(-s / 2, s / 2, len(S))[:, None])
            gains.append(g)
            ach.append(ac)
        g, ac = np.mean(gains, axis=0), np.mean(ach, axis=0)
        out["scales"][str(s)] = {"gain": cluster_bootstrap(g, n_boot=4000, level=0.90),
                                 "recovered": cluster_bootstrap(g - g0, n_boot=4000, level=0.90),
                                 "achievable": cluster_bootstrap(ac - a0, n_boot=4000, level=0.90)}
        r_, rc, av = (out["scales"][str(s)][k] for k in ("gain", "recovered", "achievable"))
        print(f"s = {s:.2f}: stock-aware minus pooled {r_['mean']:+.4f} [{r_['lo']:+.4f}, {r_['hi']:+.4f}]; recovered "
              f"{rc['mean']:+.4f} [{rc['lo']:+.4f}, {rc['hi']:+.4f}]; achievable {av['mean']:+.4f}", flush=True)
    write_json(TR.OUT / f"aware_inject_{a.model}.json", {**out, "provenance": provenance(args=vars(a))})


# --- eval -----------------------------------------------------------------------------------------

def eval_task(args):
    day, stock, eps, names, data_name, ko, kind = (*args, "")[:7]
    venue = str(eps["venue"].iloc[0])
    book, _o, _c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True).copy()
    for col, v in r1_stats(FlowBook.build(book), eps, tick=tick).items():
        eps[col] = v
    uf = unit_frame(eps, ko)
    pols = []
    for n in names:
        model = TR._model(str(TR.OUT / "models" / f"{n}.pkl"))
        z = np.load(AWARE / f"{n}__{data_name}{kind}.npz")
        pols += [P.FQIPolicy(model, name=n), AwarePolicy(model, z["stocks"], z["offsets"], name=f"aware{kind[2:]}_{n}",
                                                        in_spreads=bool(z["in_spreads"]) if "in_spreads" in z else False)]
    res = P.evaluate_book(book, eps, uf, pols, K, venue_of(venue).latency_ns, tick=tick)
    ex_ = pd.Series(excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy()), index=eps["key"])
    res["excluded"] = res["key"].map(ex_).to_numpy(bool)
    return res


def cmd_eval(a):
    ko = load_splits()["ko_constants"]
    eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
    seen = set(pd.read_parquet(TR.TRANS / a.data / "eps.parquet", columns=["stock"])["stock"])
    if a.seen_only:
        eps = eps[eps["stock"].isin(seen)]
    kind = "__shift" if a.kind == "shift" else ""
    jobs = [(d, s, g, a.models, a.data, ko, kind) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
    t0 = time.time()
    res = pd.concat(TR.pool(eval_task, jobs, a.workers), ignore_index=True)
    keep = ["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]
    res = res.merge(eps[keep], on="key")
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
    res["seen"] = res["stock"].isin(seen)
    res.to_parquet(TR.OUT / f"eval_{a.out}.parquet", index=False)
    write_json(TR.OUT / f"eval_{a.out}.json", {"episodes": a.episodes, "data": a.data, "models": a.models,
                                               "rows": len(res), "seconds": round(time.time() - t0),
                                               "provenance": provenance(args=vars(a))})
    r = res[~res["excluded"] & res["seen"]]
    print(f"{a.out}: {len(res):,} rows, {time.time() - t0:.0f}s", flush=True)
    print(r.groupby("policy")["c_spread"].mean().round(4).to_string(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--models", nargs="+", required=True)
    f.add_argument("--data", required=True)
    f.add_argument("--passes", type=int, default=2)
    f.add_argument("--cv", default="policy", choices=["policy", "residual"],
                   help="choose lambda by the replayed cost on the held-out day (policy) or by the squared residual")
    f.add_argument("--workers", type=int, default=24)
    d = sub.add_parser("episodes")
    d.add_argument("--name", required=True)
    d.add_argument("--split", required=True, choices=["train", "val", "test"])
    d.add_argument("--days", nargs="+", required=True)
    d.add_argument("--per-book", type=int, default=1000)
    d.add_argument("--workers", type=int, default=24)
    s = sub.add_parser("fit-shift")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--workers", type=int, default=24)
    h = sub.add_parser("eval-shifts")
    h.add_argument("--episodes", required=True)
    h.add_argument("--models", nargs="+", required=True)
    h.add_argument("--data", required=True)
    h.add_argument("--unseal", action="store_true")
    h.add_argument("--reason", default="")
    h.add_argument("--workers", type=int, default=24)
    j = sub.add_parser("inject")
    j.add_argument("--model", required=True)
    j.add_argument("--data", required=True)
    j.add_argument("--episodes", required=True)
    j.add_argument("--draws", type=int, default=50)
    e = sub.add_parser("eval")
    e.add_argument("--kind", default="offsets", choices=["offsets", "shift"])
    e.add_argument("--episodes", required=True)
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--unseal", action="store_true")
    e.add_argument("--reason", default="")
    e.add_argument("--workers", type=int, default=24)
    e.add_argument("--seen-only", action="store_true", help="only the stocks with offsets (default: all)")
    a = ap.parse_args(argv)
    os.environ.setdefault("INVEX_PRED_THREADS", "16")
    {"fit": cmd_fit, "fit-shift": cmd_fit_shift, "episodes": cmd_episodes, "eval-shifts": cmd_eval_shifts,
     "inject": cmd_inject, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

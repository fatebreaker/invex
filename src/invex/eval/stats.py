"""Inference for paired policy comparisons on frozen episodes.

Policies are compared on identical episodes, so the unit of analysis is the
per-asset (or per asset-day) mean of a paired difference. Uncertainty comes
from resampling assets (stratified by eta tercile), or assets and days
crossed (pigeonhole bootstrap, Owen 2007); universality is claimed only
through equivalence (two one-sided tests against a margin fixed in advance).

* summarize(values)                 mean, median, normal 95% half-width
* cluster_bootstrap(d, strata)      CI of the mean over clusters, resampled within strata
* pigeonhole_bootstrap(cells, ...)  CI of a mean over a clusters x days table, both resampled
* tost(lo90, hi90, delta)           equivalence from a 90% interval
* holm(p), bh(p)                    multiple-comparison adjustments
* cluster_ols(y, X, g)              OLS with cluster-robust standard errors (G/(G-1) small-sample factor)
"""

from __future__ import annotations

import math

import numpy as np


def summarize(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci95": float("nan"), "n": 0}
    se = a.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(a.mean()), "median": float(np.median(a)), "ci95": float(1.96 * se), "n": n}


def cluster_bootstrap(d, strata=None, n_boot: int = 10_000, level: float = 0.95, seed: int = 0,
                      return_boots: bool = False) -> dict:
    """Bootstrap CI for the mean of per-cluster values `d`, resampling clusters within `strata`.

    The estimate is the equally weighted mean over clusters (per stratum sizes are preserved). With
    return_boots, the replicates are returned too (key 'boots'), for bootstrap tests."""
    d = np.asarray(d, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    s = np.zeros(len(d), dtype=int) if strata is None else np.asarray(strata)[ok]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == k) for k in np.unique(s)]
    total = np.zeros(n_boot)
    for g in groups:
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        total += d[g][idx].sum(axis=1)
    boots = total / len(d)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    out = {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "se": float(boots.std(ddof=1)), "n": int(len(d))}
    if return_boots:
        out["boots"] = boots
    return out


def boot_p(boots: np.ndarray, kind: str, delta: float = 0.0) -> float:
    """Percentile-bootstrap p-value (the test that rejects at level a exactly when the matching percentile
    interval excludes the null): 'greater' (mean > 0), 'less' (mean < 0), 'tost' (|mean| < delta: the larger of
    the two one-sided p-values), 'noninferior' (mean < delta)."""
    b = np.asarray(boots, dtype=float)
    n = len(b)
    if kind == "greater":
        return float((1 + np.sum(b <= 0)) / (1 + n))
    if kind == "less":
        return float((1 + np.sum(b >= 0)) / (1 + n))
    if kind == "noninferior":
        return float((1 + np.sum(b >= delta)) / (1 + n))
    if kind == "tost":
        return float(max((1 + np.sum(b >= delta)) / (1 + n), (1 + np.sum(b <= -delta)) / (1 + n)))
    raise ValueError(kind)


def pigeonhole_bootstrap(sums: np.ndarray, counts: np.ndarray, n_boot: int = 2000, level: float = 0.95,
                         seed: int = 0, return_boots: bool = False) -> dict:
    """Mean over observations in a clusters x days table (cell sums and counts), resampling clusters
    and days independently with replacement (a two-way crossed bootstrap)."""
    sums = np.nan_to_num(np.asarray(sums, dtype=float))
    counts = np.asarray(counts, dtype=float)
    A, D = sums.shape
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wa = np.bincount(rng.integers(0, A, A), minlength=A).astype(float)
        wd = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        boots[b] = (wa @ sums @ wd) / max((wa @ counts @ wd), 1e-12)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    est = float(sums.sum() / max(counts.sum(), 1e-12))
    out = {"mean": est, "lo": float(lo), "hi": float(hi), "se": float(boots.std(ddof=1))}
    if return_boots:
        out["boots"] = boots
    return out


def tost(lo90: float, hi90: float, delta: float) -> bool:
    """Equivalence within +-delta at the 5% level: the 90% interval lies inside (-delta, delta)."""
    return bool(-delta < lo90 and hi90 < delta)


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        running = min(running, p[i] * m / (r + 1))
        adj[i] = running
    return adj


def cluster_ols(y, X, g) -> dict:
    """OLS of y on X (add a constant column yourself) with standard errors clustered by g.

    Uses the G/(G-1) * (N-1)/(N-K) small-sample correction; t-tests should use G-1 degrees of freedom."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    g = np.asarray(g)
    N, Kx = X.shape
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ b
    groups = np.unique(g)
    G = len(groups)
    meat = np.zeros((Kx, Kx))
    for c in groups:
        m = g == c
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    bread = np.linalg.inv(XtX)
    V = bread @ meat @ bread * (G / max(G - 1, 1)) * ((N - 1) / max(N - Kx, 1))
    return {"b": b, "se": np.sqrt(np.diag(V)), "n": N, "clusters": G, "df": G - 1}

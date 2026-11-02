"""Finite-horizon fitted Q-iteration on exogenous replay paths.

Data (`generate_book`): episodes run with a randomised behaviour policy; at
every decision the state is branched over all valid actions on the same
market path (invex.env.execution.branch_stage), which gives each action's exact
one-step cost and next state. Nothing about the market depends on the agent,
so there is no exploration problem, only a state-coverage one.

Fit (`fit`): for k = K-1 .. 0, one LightGBM regressor per stage on (features
in representation `rep`, action) with target

    cost_k(s, a) / (Q * price unit)  +  [not done] * min_{a' valid} Q_{k+1}(s'_a, a')

(the last stage's cost already includes the forced cross). Costs are divided
by the representation's own price unit: a representation fixes the units of
the state and of the reward.

`FQIPolicy` acts greedily (lowest predicted cost among valid actions).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..env import execution as ex
from ..env.features import BASE_SYSTEM, N_RAW, PROFILE_COLS, REWARD_SYSTEM, FlowBook, raw_features, represent
from ..sim import tracker as tk
from ..venues import to_ref

BEHAVIOUR = np.array([0.40, 0.25, 0.10, 0.10, 0.15])  # HOLD JOIN IMPROVE CANCEL CROSS, renormalised over valid

LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  deterministic=True, force_row_wise=True, max_cat_to_onehot=8)


def generate_book(book: tk.ReplayBook, eps: pd.DataFrame, K: int, latency: int, rule: int, tick: int,
                  seed: int) -> dict:
    """Branched transitions for the episodes `eps` (one symbol-day). Arrays are float32 / int8."""
    fb = FlowBook.build(book)
    n = len(eps)
    sg = eps["side"].to_numpy(np.int64)
    t0 = eps["t0"].to_numpy(np.int64)
    T = eps["T_ns"].to_numpy(np.int64)
    Q = eps["Q"].to_numpy(np.int64)
    mid0 = eps["mid0"].to_numpy(float)
    st = ex.new_states(Q)
    rng = np.random.default_rng(seed)
    raw = np.zeros((K, n, N_RAW), np.float32)
    masks = np.zeros((K, n, ex.N_ACT), np.int8)
    costs = np.zeros((K, n, ex.N_ACT), np.float32)
    raw_next = np.zeros((K - 1, ex.N_ACT, n, N_RAW), np.float32)
    masks_next = np.zeros((K - 1, ex.N_ACT, n, ex.N_ACT), np.int8)
    done_next = np.zeros((K, n, ex.N_ACT), np.int8)
    arrs = book.arrays()
    for k in range(K):
        raw[k] = raw_features(fb, st, sg, mid0, t0, T, Q, k, K, tick)
        m = np.zeros((n, ex.N_ACT), np.int64)
        nxt = np.zeros((n, ex.N_ACT, ex.AGENT_SIZE), np.int64)
        c = np.zeros((n, ex.N_ACT))
        ex.branch_stage(*arrs, st, sg, mid0, t0, T, k, K, latency, tick, rule, m, nxt, c)
        masks[k] = m
        costs[k] = np.nan_to_num(c, nan=0.0)
        done_next[k] = nxt[:, :, ex.S_DONE]
        if k < K - 1:
            for a in range(ex.N_ACT):
                sa = np.ascontiguousarray(nxt[:, a])
                raw_next[k, a] = raw_features(fb, sa, sg, mid0, t0, T, Q, k + 1, K, tick)
                mn = np.zeros((n, ex.N_ACT), np.int64)
                ex.masks_stage(*arrs, sa, sg, t0, T, k + 1, K, tick, mn)
                masks_next[k, a] = mn
        p = m * BEHAVIOUR[None, :]
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(n)[:, None]
        a_b = (np.cumsum(p, axis=1) < u).sum(axis=1)
        st = np.ascontiguousarray(nxt[np.arange(n), a_b])
    return {"raw": raw, "masks": masks, "costs": costs, "raw_next": raw_next, "masks_next": masks_next,
            "done_next": done_next}


def concat(parts: list[dict]) -> dict:
    """Concatenate generate_book outputs along the episode axis."""
    out = {}
    for key, ax in (("raw", 1), ("masks", 1), ("costs", 1), ("raw_next", 2), ("masks_next", 2), ("done_next", 1)):
        out[key] = np.concatenate([p[key] for p in parts], axis=ax)
    return out


def rep_inputs(raw: np.ndarray, eps: pd.DataFrame, uf: dict, rep: str, c_lookback: float) -> np.ndarray:
    """Features of representation rep for raw rows aligned with eps (one row per episode).

    Raw units (R0) measure time in c_lookback seconds, the same for every asset (a constant rescaling, so
    trees are unaffected); it only places R0's flow lookback at 10 * c_lookback seconds, the median
    lookback of the other systems.

    A name ending in "id" (e.g. R3tickid) appends the stock's identity as a last, categorical column: its index
    in the frozen 2019 universe, -1 (missing) for stocks outside it. That is an asset-aware comparator, not a
    representation."""
    if rep.endswith("id"):
        X = rep_inputs(raw, eps, uf, rep[:-2], c_lookback)
        return np.column_stack([X, stock_codes(eps["stock"]).astype(float)])
    sys_ = BASE_SYSTEM[rep]
    u = uf[sys_]
    ctx = {"tick": eps["tick"].to_numpy(float), "sigma_trade": uf["_sigma_trade"].to_numpy(),
           "price": eps["w_mid"].to_numpy(float), "volume_rate": (eps["w_volume"] / eps["w_seconds"]).to_numpy(float),
           "sigma_1min": eps["w_sigma_1min"].to_numpy(float)}
    if "h_inter" in eps:
        ctx.update({k: eps[k].to_numpy(float) for k in ("h_inter", "h_tsize", "h_spread")})
    if "p_post" in eps:
        ctx.update({k: eps[k].to_numpy(float) for k in PROFILE_COLS})
    if rep.endswith("info"):  # the MS units themselves (seconds, shares, price units) as inputs
        ctx.update({"info_inter": uf["R3"]["t"].to_numpy(float), "info_tsize": uf["R3"]["q"].to_numpy(float),
                    "info_spread": uf["R3"]["p"].to_numpy(float)})
        if "px_scale" in eps:  # in the reference (ITCH) frame, as for the descriptors
            f_px, f_q = to_ref(eps["px_scale"].to_numpy(float), eps["qty_scale"].to_numpy(float))
            for c in ("tick", "price", "info_spread", "sigma_trade"):
                ctx[c] = ctx[c] * f_px
            for c in ("volume_rate", "info_tsize"):
                ctx[c] = ctx[c] * f_q
    if rep.endswith("desc") and "px_scale" in eps:  # descriptors in the reference (ITCH) frame: 1e-4 $, shares
        f_px, f_q = to_ref(eps["px_scale"].to_numpy(float), eps["qty_scale"].to_numpy(float))
        ctx["tick"], ctx["price"], ctx["volume_rate"] = ctx["tick"] * f_px, ctx["price"] * f_px, ctx["volume_rate"] * f_q
    if rep == "Rpi":
        from ..env.features import represent_pi

        return represent_pi(raw.astype(float), eps["tick"].to_numpy(float))
    if rep == "Rdim":
        from ..env.features import represent_dim

        return represent_dim(raw.astype(float))
    if rep == "R1":
        from ..env.features import represent_r1

        r1 = {c: eps[c].to_numpy(float) for c in eps.columns if c.startswith("r1")}
        return represent_r1(raw.astype(float), r1, np.full(len(eps), 10.0 * c_lookback))
    X, _names = represent(raw.astype(float), u["t"].to_numpy() * c_lookback if sys_ == "R0" else u["t"].to_numpy(),
                          u["q"].to_numpy(), u["p"].to_numpy(), rep, ctx=ctx)
    return X


_CODES: dict | None = None


def stock_codes(stocks: pd.Series) -> np.ndarray:
    """Index of each stock in the frozen 2019 universe (prereg/splits_v0.json), -1 if outside it."""
    global _CODES
    if _CODES is None:
        from ..eval.universe import load_splits

        _CODES = {s: i for i, s in enumerate(sorted(load_splits()["split"]))}
    return stocks.map(_CODES).fillna(-1).to_numpy(np.int64)


def _with_action(X: np.ndarray, a: int) -> np.ndarray:
    return np.column_stack([X, np.full(X.shape[0], a, dtype=float)])


def _q_all(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    big = np.vstack([_with_action(X, a) for a in range(ex.N_ACT)])
    threads = int(os.environ.get("INVEX_PRED_THREADS", "0"))  # 1 inside process pools
    return model.predict(big, num_threads=threads).reshape(ex.N_ACT, n).T


@dataclass
class FQIModel:
    rep: str
    K: int
    models: list = field(default_factory=list)  # one Booster per stage
    c_lookback: float = 1.0  # R0: time unit (s) used to place the flow lookback (10 units)

    def q_values(self, k: int, X: np.ndarray) -> np.ndarray:
        return _q_all(self.models[k], X)

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"rep": self.rep, "K": self.K, "c_lookback": self.c_lookback,
                         "models": [m.model_to_string() for m in self.models]}, fh)

    @classmethod
    def load(cls, path) -> "FQIModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["rep"], d["K"], [lgb.Booster(model_str=s) for s in d["models"]], d["c_lookback"])


def fit(data: dict, eps: pd.DataFrame, uf: dict, rep: str, K: int, c_lookback: float, n_rounds: int = 300,
        threads: int = 16, seed: int = 0, log=print) -> FQIModel:
    """Backward fitted Q-iteration (see module docstring)."""
    norm = (eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy())  # cost -> per share, reward units
    model = FQIModel(rep, K, [None] * K, c_lookback)
    params = dict(LGB_PARAMS, num_threads=threads, seed=seed)
    for k in range(K - 1, -1, -1):
        X = rep_inputs(data["raw"][k], eps, uf, rep, c_lookback)
        m = data["masks"][k].astype(bool)
        y = data["costs"][k] / norm[:, None]
        if k < K - 1:
            nxt = model.models[k + 1]
            for a in range(ex.N_ACT):
                Xn = rep_inputs(data["raw_next"][k, a], eps, uf, rep, c_lookback)
                qn = _q_all(nxt, Xn)
                mn = data["masks_next"][k, a].astype(bool)
                qn[~mn] = np.inf
                v = qn.min(axis=1)
                v[~np.isfinite(v)] = 0.0
                y[:, a] += np.where(data["done_next"][k][:, a] == 1, 0.0, v)
        rows, targ = [], []
        for a in range(ex.N_ACT):
            sel = m[:, a]
            rows.append(_with_action(X[sel], a))
            targ.append(y[sel, a])
        Xa, ya = np.vstack(rows), np.concatenate(targ)  # NaN features are fine: LightGBM routes them
        ds = lgb.Dataset(Xa, ya, categorical_feature=[Xa.shape[1] - 1], free_raw_data=True)
        model.models[k] = lgb.train(params, ds, num_boost_round=n_rounds)
        log(f"  stage {k}: {len(ya):,} rows, target mean {np.mean(ya):.4f}")
    return model

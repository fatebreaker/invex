"""Fitted Q-iteration on advantages over crossing now (the exact baseline the replay gives for free).

The value of a state is dominated by the cost already locked in (how far the
price has moved since arrival, times the shares left), which no action
changes. Plain FQI spends its capacity on that level (see invex.agents.fqi).
Here every stage learns

    A_k(s, a) = Q_k(s, a) - C_k(s),   C_k(s) = exact cost of crossing everything now

for a != CROSS (A_k(s, CROSS) = 0 by definition). C_k(s) comes from branching
(the CROSS branch) at the visited state, and, for next states, from one extra
CROSS step at stage k + 1. The targets are

    y = [cost_k(s, a) - C_k(s)] / norm + [not done] * (C_{k+1}(s') / norm + min(0, min_{a' valid} A_{k+1}(s', a')))

with norm = Q * price unit of the representation, as in invex.agents.fqi.
The greedy policy picks the lowest advantage among valid actions (0 = cross).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..env import execution as ex
from ..env.features import N_RAW, REWARD_SYSTEM, FlowBook, raw_features
from ..sim import tracker as tk
from .fqi import BEHAVIOUR, LGB_PARAMS, rep_inputs

NON_CROSS = (ex.HOLD, ex.JOIN, ex.IMPROVE, ex.CANCEL)


def fee_cost(before: np.ndarray, after: np.ndarray, taker: float, rebate: float) -> np.ndarray:
    """Exchange fees between two agent states (rows of ex.AGENT_SIZE), in price units x shares: the taker fee on
    shares filled by crossing, less the maker rebate on shares filled while resting. The fee and rebate may be
    per-episode arrays (the first axis of the states)."""
    passive = after[..., ex.S_PASSIVE] - before[..., ex.S_PASSIVE]
    taken = after[..., ex.S_FILLED] - before[..., ex.S_FILLED] - passive
    tk, rb = (np.asarray(x, float) for x in (taker, rebate))
    tk, rb = (x.reshape(x.shape + (1,) * (taken.ndim - x.ndim)) if x.ndim else x for x in (tk, rb))
    return tk * taken - rb * passive


def generate_book(book: tk.ReplayBook, eps: pd.DataFrame, K: int, latency: int, rule: int, tick: int,
                  seed: int, taker: float = 0.0, rebate: float = 0.0) -> dict:
    """As invex.agents.fqi.generate_book, plus cross_next[k, a, e]: the exact cost of crossing at stage k+1
    from the state reached by action a at stage k. With a taker fee or maker rebate (price units per share, or
    one per episode), every cost is net of fees; the states, and so the behaviour policy's path, do not depend on them."""
    fees = bool(np.any(np.asarray(taker) != 0.0) or np.any(np.asarray(rebate) != 0.0))
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
    cross_next = np.zeros((K - 1, ex.N_ACT, n), np.float32)
    done_next = np.zeros((K, n, ex.N_ACT), np.int8)
    arrs = book.arrays()
    cross = np.full(n, ex.CROSS, np.int64)
    for k in range(K):
        raw[k] = raw_features(fb, st, sg, mid0, t0, T, Q, k, K, tick)
        m = np.zeros((n, ex.N_ACT), np.int64)
        nxt = np.zeros((n, ex.N_ACT, ex.AGENT_SIZE), np.int64)
        c = np.zeros((n, ex.N_ACT))
        ex.branch_stage(*arrs, st, sg, mid0, t0, T, k, K, latency, tick, rule, m, nxt, c)
        masks[k] = m
        if fees:
            c = c + fee_cost(st[:, None, :], nxt, taker, rebate)
        costs[k] = np.nan_to_num(c, nan=0.0)
        done_next[k] = nxt[:, :, ex.S_DONE]
        if k < K - 1:
            for a in range(ex.N_ACT):
                sa = np.ascontiguousarray(nxt[:, a])
                raw_next[k, a] = raw_features(fb, sa, sg, mid0, t0, T, Q, k + 1, K, tick)
                mn = np.zeros((n, ex.N_ACT), np.int64)
                ex.masks_stage(*arrs, sa, sg, t0, T, k + 1, K, tick, mn)
                masks_next[k, a] = mn
                sc = sa.copy()
                cc = np.zeros(n)
                ex.rollout_stage(*arrs, sc, sg, mid0, t0, T, k + 1, K, cross, latency, tick, rule, cc)
                cross_next[k, a] = cc + fee_cost(sa, sc, taker, rebate) if fees else cc
        p = m * BEHAVIOUR[None, :]
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(n)[:, None]
        a_b = (np.cumsum(p, axis=1) < u).sum(axis=1)
        st = np.ascontiguousarray(nxt[np.arange(n), a_b])
    return {"raw": raw, "masks": masks, "costs": costs, "raw_next": raw_next, "masks_next": masks_next,
            "cross_next": cross_next, "done_next": done_next}


EPISODE_AXIS = {"raw": 1, "masks": 1, "costs": 1, "done_next": 1, "raw_next": 2, "masks_next": 2, "cross_next": 2}


def concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts], axis=ax) for k, ax in EPISODE_AXIS.items()}


def subset(data: dict, keep: np.ndarray) -> dict:
    return {k: np.take(v, keep, axis=EPISODE_AXIS[k]) for k, v in data.items()}


def _with_action(X: np.ndarray, a: int) -> np.ndarray:
    return np.column_stack([X, np.full(X.shape[0], a, dtype=float)])


@dataclass
class AdvModel:
    rep: str
    K: int
    models: list = field(default_factory=list)
    c_lookback: float = 1.0

    def q_values(self, k: int, X: np.ndarray) -> np.ndarray:
        """Advantages over crossing now, (n, N_ACT); the CROSS column is 0."""
        n = X.shape[0]
        out = np.zeros((n, ex.N_ACT))
        big = np.vstack([_with_action(X, a) for a in NON_CROSS])
        threads = int(os.environ.get("INVEX_PRED_THREADS", "0"))
        pred = self.models[k].predict(big, num_threads=threads).reshape(len(NON_CROSS), n).T
        for i, a in enumerate(NON_CROSS):
            out[:, a] = pred[:, i]
        return out

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"rep": self.rep, "K": self.K, "c_lookback": self.c_lookback, "kind": "advantage",
                         "models": [m.model_to_string() for m in self.models]}, fh)

    @classmethod
    def load(cls, path) -> "AdvModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["rep"], d["K"], [lgb.Booster(model_str=s) for s in d["models"]], d["c_lookback"])


def fit(data: dict, eps: pd.DataFrame, uf: dict, rep: str, K: int, c_lookback: float, n_rounds: int = 300,
        threads: int = 16, seed: int = 0, params: dict | None = None, log=print,
        norm: np.ndarray | None = None) -> AdvModel:
    """norm (optional, per episode) replaces Q * the representation's price unit as the unit of the targets,
    e.g. a per-stock scale learned from the training data (invex.agents.fqi_nn.stock_scale)."""
    if norm is None:
        norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
    model = AdvModel(rep, K, [None] * K, c_lookback)
    prm = dict(LGB_PARAMS, num_threads=threads, seed=seed, **(params or {}))
    for k in range(K - 1, -1, -1):
        X = rep_inputs(data["raw"][k], eps, uf, rep, c_lookback)
        m = data["masks"][k].astype(bool)
        c_now = data["costs"][k][:, ex.CROSS]
        y = (data["costs"][k] - c_now[:, None]) / norm[:, None]
        if k < K - 1:
            nxt = model.models[k + 1]
            for a in NON_CROSS:
                Xn = rep_inputs(data["raw_next"][k, a], eps, uf, rep, c_lookback)
                An = model.q_values(k + 1, Xn)
                mn = data["masks_next"][k, a].astype(bool)
                An[~mn] = np.inf
                An[:, ex.CROSS] = 0.0  # crossing is always valid
                v = data["cross_next"][k, a] / norm + np.minimum(0.0, An.min(axis=1))
                y[:, a] += np.where(data["done_next"][k][:, a] == 1, 0.0, v)
        rows, targ = [], []
        for a in NON_CROSS:
            sel = m[:, a]
            rows.append(_with_action(X[sel], a))
            targ.append(y[sel, a])
        Xa, ya = np.vstack(rows), np.concatenate(targ)
        cats = [Xa.shape[1] - 2, Xa.shape[1] - 1] if rep.endswith("id") else [Xa.shape[1] - 1]
        ds = lgb.Dataset(Xa, ya, categorical_feature=cats, free_raw_data=True)
        model.models[k] = lgb.train(prm, ds, num_boost_round=n_rounds)
        log(f"  stage {k}: {len(ya):,} rows, advantage target mean {np.mean(ya):+.4f} sd {np.std(ya):.4f}")
    return model


def finetune(base: AdvModel, data: dict, eps: pd.DataFrame, uf: dict, n_rounds: int = 50, threads: int = 1,
             seed: int = 0, params: dict | None = None) -> AdvModel:
    """Adapt a fitted model to new data (e.g. one stock's own days): the same backward recursion as `fit`, each
    stage's booster continued for n_rounds more trees from the base model's predictions (LightGBM init_model)."""
    rep, K = base.rep, base.K
    norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
    model = AdvModel(rep, K, [None] * K, base.c_lookback)
    prm = dict(LGB_PARAMS, num_threads=threads, seed=seed, **(params or {}))
    for k in range(K - 1, -1, -1):
        X = rep_inputs(data["raw"][k], eps, uf, rep, base.c_lookback)
        m = data["masks"][k].astype(bool)
        c_now = data["costs"][k][:, ex.CROSS]
        y = (data["costs"][k] - c_now[:, None]) / norm[:, None]
        if k < K - 1:
            for a in NON_CROSS:
                Xn = rep_inputs(data["raw_next"][k, a], eps, uf, rep, base.c_lookback)
                An = model.q_values(k + 1, Xn)
                mn = data["masks_next"][k, a].astype(bool)
                An[~mn] = np.inf
                An[:, ex.CROSS] = 0.0
                v = data["cross_next"][k, a] / norm + np.minimum(0.0, An.min(axis=1))
                y[:, a] += np.where(data["done_next"][k][:, a] == 1, 0.0, v)
        rows, targ = [], []
        for a in NON_CROSS:
            sel = m[:, a]
            rows.append(_with_action(X[sel], a))
            targ.append(y[sel, a])
        Xa, ya = np.vstack(rows), np.concatenate(targ)
        ds = lgb.Dataset(Xa, ya, categorical_feature=[Xa.shape[1] - 1], free_raw_data=False)
        model.models[k] = lgb.train(prm, ds, num_boost_round=n_rounds, init_model=base.models[k])
    return model


def load_model(path):
    """Either kind of fitted model (plain FQIModel or AdvModel), by the tag stored in the file."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if d.get("kind") == "nn_advantage":
        from .fqi_nn import NNAdvModel

        return NNAdvModel.from_dict(d)
    if d.get("kind") == "ppo":
        from .ppo import PPOModel

        return PPOModel.from_dict(d)
    if d.get("kind") == "advantage":
        return AdvModel(d["rep"], d["K"], [lgb.Booster(model_str=s) for s in d["models"]], d["c_lookback"])
    from .fqi import FQIModel

    return FQIModel(d["rep"], d["K"], [lgb.Booster(model_str=s) for s in d["models"]], d["c_lookback"])

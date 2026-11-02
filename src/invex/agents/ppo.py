"""Proximal policy optimisation in the replay: the on-policy learner, outside fitted Q-iteration.

The policy is one network for all K decisions: the representation's features (invex.agents.fqi.rep_inputs) through
sign(x) log(1 + |x|), standardised with statistics of the training transitions (as invex.agents.fqi_nn), plus a
one-hot decision index, give logits over the five actions; invalid actions are masked. A separate value network
predicts the remaining cost. Rewards are the step costs in the representation's own cost unit (Q times its price
unit, as for fitted Q-iteration: spreads for MS units, cents for raw units), negated. There is no discounting
(the horizon is K decisions), advantages are GAE(lambda) with lambda 0.95, and updates use the clipped surrogate
objective with an entropy bonus.

Rollouts run in persistent worker processes, each holding a shard of the training stock-days in memory and
replaying the current stochastic policy on the frozen training episodes (the episodes fitted Q-iteration was
trained on). The fitted model's greedy policy picks the most likely valid action; `q_values` returns negated
logits so the evaluation code (invex.eval.policies.FQIPolicy) can use it unchanged.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

from ..env import execution as ex
from .fqi import rep_inputs

CLIP = 8.0


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def mlp_forward(layers: list, Z: np.ndarray) -> np.ndarray:
    """layers: [(W (d_in, d_out), b)], SiLU between layers, linear output."""
    h = Z
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = _silu(h)
    return h


@dataclass
class PPOModel:
    rep: str
    K: int
    c_lookback: float
    mu: np.ndarray
    sd: np.ndarray
    pi: list = field(default_factory=list)  # [(W, b)] numpy, float32
    v: list = field(default_factory=list)

    def inputs(self, k: int, X: np.ndarray) -> np.ndarray:
        Z = (slog(X.astype(np.float64)) - self.mu) / self.sd
        Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
        oh = np.zeros((Z.shape[0], self.K))
        oh[:, k] = 1.0
        return np.hstack([Z, oh]).astype(np.float32)

    def logits(self, k: int, X: np.ndarray) -> np.ndarray:
        return mlp_forward(self.pi, self.inputs(k, X))

    def q_values(self, k: int, X: np.ndarray) -> np.ndarray:
        """Negated logits: the evaluation code takes the valid action with the lowest value."""
        return -self.logits(k, X).astype(np.float64)

    def to_dict(self) -> dict:
        return {"kind": "ppo", "rep": self.rep, "K": self.K, "c_lookback": self.c_lookback, "mu": self.mu,
                "sd": self.sd, "pi": self.pi, "v": self.v}

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.to_dict(), fh)

    @classmethod
    def from_dict(cls, d: dict) -> "PPOModel":
        return cls(d["rep"], d["K"], d["c_lookback"], d["mu"], d["sd"], d["pi"], d["v"])


def input_stats(data: dict, eps, uf: dict, rep: str, c_lookback: float, n: int = 200_000, seed: int = 0):
    """Mean and sd of slog(features) over the training transitions (every stage, a random subset of episodes)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(eps), size=min(n, len(eps)), replace=False))
    e = eps.iloc[idx].reset_index(drop=True)
    u = {k: (v.iloc[idx].reset_index(drop=True) if hasattr(v, "iloc") else v[idx]) for k, v in uf.items()}
    L = np.vstack([slog(rep_inputs(np.asarray(data["raw"][k])[idx], e, u, rep, c_lookback).astype(np.float64))
                   for k in range(data["raw"].shape[0])])
    L = np.where(np.isfinite(L), L, np.nan)
    mu, sd = np.nan_to_num(np.nanmean(L, axis=0)), np.nan_to_num(np.nanstd(L, axis=0))
    return mu, np.where(sd > 1e-8, sd, 1.0)


def rollout_book(model: PPOModel, book, fb, arrs, eps, uf, norm: np.ndarray, K: int, latency: int, tick: int,
                 rng: np.random.Generator, rule: int, greedy: bool = False) -> dict:
    """One stock-day's episodes under the model's stochastic (or greedy) policy: per episode and decision, the
    network input, action, its log-probability, the valid-action mask, the reward (minus the step cost in the
    representation's cost unit) and whether the episode was still open."""
    from ..env.features import raw_features

    n = len(eps)
    sg = eps["side"].to_numpy(np.int64)
    t0 = eps["t0"].to_numpy(np.int64)
    T = eps["T_ns"].to_numpy(np.int64)
    Q = eps["Q"].to_numpy(np.int64)
    mid0 = eps["mid0"].to_numpy(float)
    st = ex.new_states(Q)
    d_in = len(model.mu) + K
    Z = np.zeros((n, K, d_in), np.float32)
    A = np.zeros((n, K), np.int64)
    LP = np.zeros((n, K), np.float32)
    M = np.zeros((n, K, ex.N_ACT), np.bool_)
    R = np.zeros((n, K), np.float32)
    alive = np.zeros((n, K), np.bool_)
    for k in range(K):
        alive[:, k] = st[:, ex.S_DONE] == 0
        raw = raw_features(fb, st, sg, mid0, t0, T, Q, k, K, tick)
        X = rep_inputs(raw, eps, uf, model.rep, model.c_lookback)
        zk = model.inputs(k, X)
        m = np.zeros((n, ex.N_ACT), np.int64)
        ex.masks_stage(*arrs, st, sg, t0, T, k, K, tick, m)
        mb = m == 1
        lg = mlp_forward(model.pi, zk).astype(np.float64)
        lg = np.where(mb, lg, -np.inf)
        lg -= lg.max(axis=1, keepdims=True)
        p = np.exp(lg)
        p /= p.sum(axis=1, keepdims=True)
        if greedy:
            a = p.argmax(axis=1)
        else:
            a = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        a = np.where(mb[np.arange(n), a], a, ex.HOLD)
        c = np.zeros(n)
        ex.rollout_stage(*arrs, st, sg, mid0, t0, T, k, K, a.astype(np.int64), latency, tick, rule, c)
        Z[:, k], A[:, k], M[:, k] = zk, a, mb
        LP[:, k] = np.log(np.maximum(p[np.arange(n), a], 1e-12))
        R[:, k] = -c / norm
    return {"Z": Z, "A": A, "LP": LP, "M": M, "R": R, "alive": alive}


def gae(R, V, alive, lam):
    """Advantages and return targets per (episode, decision), no discounting; V is 0 where not alive."""
    n, k_ = R.shape
    adv = np.zeros_like(R)
    nxt_v = np.zeros(n, np.float32)
    nxt_a = np.zeros(n, np.float32)
    for k in range(k_ - 1, -1, -1):
        cont = alive[:, k + 1] if k + 1 < k_ else np.zeros(n, bool)
        nv = np.where(cont, nxt_v, 0.0)
        delta = R[:, k] + nv - V[:, k]
        adv[:, k] = np.where(alive[:, k], delta + lam * np.where(cont, nxt_a, 0.0), 0.0)
        nxt_v, nxt_a = V[:, k], adv[:, k]
    return adv, adv + V

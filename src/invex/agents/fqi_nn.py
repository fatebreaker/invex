"""Neural fitted Q-iteration on advantages: the learner-robustness arm.

Same data, targets, stages and greedy policy as invex.agents.fqi_adv, with a multilayer perceptron per stage
in place of boosted trees. Trees split one feature at a time and cannot form ratios of features; the network
sees every feature through sign(x) log(1 + |x|), standardised, so a ratio of two features is a difference of
two inputs, which a linear layer represents exactly. If raw units lose only because trees cannot normalise,
they should stop losing here.

Each stage's network has one output per non-crossing action (the advantage over crossing now; CROSS is 0)
and is trained on the valid actions only (masked squared error), warm-started from the next stage's network.

Two learned alternatives to given units, for the raw representation:
  * stock_scale(): a per-stock target scale estimated from the training data (the root mean square of the
    one-step advantages), the PopArt idea of normalising each task's targets by a learned scale. It needs the
    stock's identity during training only; at test time the greedy action does not depend on the scale.
  * augment_scales(): random rescaling of every episode's prices and sizes (domain randomisation), applied to
    the features and the targets consistently. Time is not rescaled: the flow windows of the raw features are
    fixed in seconds, so a rescaled clock would not be consistent with them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..env import execution as ex
from ..env.features import (F_ADV, F_AHEAD, F_FAR, F_FLOW0, F_NEAR, F_Q, F_SPREAD, NL, REWARD_SYSTEM)
from .fqi import rep_inputs
from .fqi_adv import NON_CROSS

CLIP = 8.0


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _torch():
    import torch

    return torch


def _net(d_in: int, hidden: int, depth: int):
    torch = _torch()
    nn = torch.nn
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, len(NON_CROSS)))
    return nn.Sequential(*layers)


@dataclass
class NNAdvModel:
    rep: str
    K: int
    c_lookback: float = 1.0
    hidden: int = 256
    depth: int = 3
    states: list = field(default_factory=list)  # per stage: state_dict (CPU tensors)
    mu: list = field(default_factory=list)  # per stage: input mean (after slog)
    sd: list = field(default_factory=list)
    t_scale: list = field(default_factory=list)  # per stage: target scale
    _nets: dict = field(default_factory=dict, repr=False)

    def _stage_net(self, k: int, device: str = "cpu"):
        key = (k, device)
        if key not in self._nets:
            torch = _torch()
            net = _net(len(self.mu[k]), self.hidden, self.depth)
            net.load_state_dict(self.states[k])
            net.eval()
            self._nets[key] = net.to(device)
        return self._nets[key]

    def inputs(self, k: int, X: np.ndarray) -> np.ndarray:
        Z = (slog(X.astype(np.float64)) - self.mu[k]) / self.sd[k]
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP).astype(np.float32)

    def predict(self, k: int, X: np.ndarray, device: str = "cpu", batch: int = 1 << 16) -> np.ndarray:
        """Advantages of the non-crossing actions, (n, len(NON_CROSS)), in target units."""
        torch = _torch()
        threads = int(os.environ.get("INVEX_PRED_THREADS", "0"))
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        net = self._stage_net(k, device)
        Z = self.inputs(k, X)
        out = np.empty((Z.shape[0], len(NON_CROSS)), np.float64)
        with torch.no_grad():
            for i in range(0, Z.shape[0], batch):
                out[i:i + batch] = net(torch.from_numpy(Z[i:i + batch]).to(device)).double().cpu().numpy()
        return out * self.t_scale[k]

    def q_values(self, k: int, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Advantages over crossing now, (n, N_ACT); the CROSS column is 0 (as invex.agents.fqi_adv.AdvModel)."""
        out = np.zeros((X.shape[0], ex.N_ACT))
        pred = self.predict(k, X, device)
        for i, a in enumerate(NON_CROSS):
            out[:, a] = pred[:, i]
        return out

    def to_dict(self) -> dict:
        return {"kind": "nn_advantage", "rep": self.rep, "K": self.K, "c_lookback": self.c_lookback,
                "hidden": self.hidden, "depth": self.depth, "states": self.states, "mu": self.mu, "sd": self.sd,
                "t_scale": self.t_scale}

    def save(self, path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self.to_dict(), fh)

    @classmethod
    def from_dict(cls, d: dict) -> "NNAdvModel":
        return cls(d["rep"], d["K"], d["c_lookback"], d["hidden"], d["depth"], d["states"], d["mu"], d["sd"],
                   d["t_scale"])


# --- learned alternatives to given units -----------------------------------------------------------

def stock_scale(data: dict, eps: pd.DataFrame) -> np.ndarray:
    """Per-episode target unit Q * s_stock, where s_stock is the root mean square of the stock's one-step
    advantages per share (valid non-crossing actions, all stages) in its training episodes (PopArt-style)."""
    c = data["costs"]  # (K, n, N_ACT)
    m = data["masks"].astype(bool)
    Q = eps["Q"].to_numpy(float)
    adv = (c - c[:, :, [ex.CROSS]]) / Q[None, :, None]
    sel = m.copy()
    sel[:, :, ex.CROSS] = False
    sq = np.where(sel, adv ** 2, 0.0).sum(axis=(0, 2))
    cnt = sel.sum(axis=(0, 2))
    key = eps["venue"].astype(str) + ":" + eps["stock"].astype(str) if "venue" in eps else eps["stock"].astype(str)
    g = pd.DataFrame({"k": key.to_numpy(), "sq": sq, "cnt": cnt}).groupby("k")[["sq", "cnt"]].sum()
    s = np.sqrt(g["sq"] / g["cnt"].clip(lower=1))
    s = s.where(s > 0, s[s > 0].median())
    return Q * key.map(s).to_numpy(float)


SIZE_COLS = [F_NEAR, F_FAR, F_AHEAD, F_Q] + list(range(F_FLOW0, F_FLOW0 + NL))
PRICE_COLS = [F_SPREAD, F_ADV]


def augment_scales(data: dict, eps: pd.DataFrame, seed: int, span: float = 10.0) -> tuple[dict, pd.DataFrame]:
    """The data plus one rescaled copy: every episode's prices times p and sizes times s, p and s log-uniform
    in [1/span, span]. Features, costs (price x size) and the order size Q are rescaled consistently, so the
    raw-unit target cost / Q scales by p."""
    rng = np.random.default_rng(seed + 7919)
    n = len(eps)
    p = np.exp(rng.uniform(-np.log(span), np.log(span), n))
    s = np.exp(rng.uniform(-np.log(span), np.log(span), n))
    aug = {}
    for name in ("raw", "raw_next"):
        x = data[name].copy()
        x[..., PRICE_COLS] *= p.reshape((1,) * (x.ndim - 2) + (n, 1)).astype(x.dtype)
        x[..., SIZE_COLS] *= s.reshape((1,) * (x.ndim - 2) + (n, 1)).astype(x.dtype)
        aug[name] = x
    aug["costs"] = data["costs"] * (p * s)[None, :, None].astype(data["costs"].dtype)
    aug["cross_next"] = data["cross_next"] * (p * s)[None, None, :].astype(data["cross_next"].dtype)
    for name in ("masks", "masks_next", "done_next"):
        aug[name] = data[name]
    e2 = eps.copy()
    e2["Q"] = eps["Q"].to_numpy(float) * s
    from .fqi_adv import EPISODE_AXIS

    both = {k: np.concatenate([data[k], aug[k]], axis=EPISODE_AXIS[k]) for k in EPISODE_AXIS}
    return both, pd.concat([eps, e2], ignore_index=True)


# --- fit ------------------------------------------------------------------------------------------

def fit(data: dict, eps: pd.DataFrame, uf: dict, rep: str, K: int, c_lookback: float, seed: int = 0,
        device: str = "cpu", norm: np.ndarray | None = None, hidden: int = 256, depth: int = 3,
        epochs: int = 40, patience: int = 4, batch: int = 4096, lr: float = 1e-3, log=print) -> NNAdvModel:
    """Backward over stages as invex.agents.fqi_adv.fit. 5% of episodes are held out for early stopping."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if norm is None:
        norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
    n = len(eps)
    hold = rng.random(n) < 0.05
    model = NNAdvModel(rep, K, c_lookback, hidden, depth, [None] * K, [None] * K, [None] * K, [None] * K)
    prev_state = None
    for k in range(K - 1, -1, -1):
        X = rep_inputs(data["raw"][k], eps, uf, rep, c_lookback)
        m = data["masks"][k].astype(bool)
        c_now = data["costs"][k][:, ex.CROSS]
        y = (data["costs"][k] - c_now[:, None]) / norm[:, None]
        if k < K - 1:
            for a in NON_CROSS:
                Xn = rep_inputs(data["raw_next"][k, a], eps, uf, rep, c_lookback)
                An = model.q_values(k + 1, Xn, device)
                mn = data["masks_next"][k, a].astype(bool)
                An[~mn] = np.inf
                An[:, ex.CROSS] = 0.0
                v = data["cross_next"][k, a] / norm + np.minimum(0.0, An.min(axis=1))
                y[:, a] += np.where(data["done_next"][k][:, a] == 1, 0.0, v)
        Y = y[:, list(NON_CROSS)]
        M = m[:, list(NON_CROSS)] & np.isfinite(Y)
        Y = np.where(M, Y, 0.0)
        L = slog(X.astype(np.float64))
        mu = np.nanmean(np.where(np.isfinite(L), L, np.nan), axis=0)
        sd = np.nanstd(np.where(np.isfinite(L), L, np.nan), axis=0)
        mu, sd = np.nan_to_num(mu), np.where(np.nan_to_num(sd) > 1e-8, np.nan_to_num(sd), 1.0)
        model.mu[k], model.sd[k] = mu, sd
        t_scale = float(np.sqrt((Y[M] ** 2).mean())) or 1.0
        model.t_scale[k] = t_scale
        Z = model.inputs(k, X)
        net = _net(Z.shape[1], hidden, depth)
        if prev_state is not None:
            net.load_state_dict(prev_state)
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        tr, va = np.flatnonzero(~hold), np.flatnonzero(hold)
        Zt = torch.from_numpy(Z).to(device)
        Yt = torch.from_numpy((Y / t_scale).astype(np.float32)).to(device)
        Mt = torch.from_numpy(M.astype(np.float32)).to(device)

        def loss_on(idx):
            out = net(Zt[idx])
            return ((out - Yt[idx]) ** 2 * Mt[idx]).sum() / Mt[idx].sum().clamp(min=1.0)

        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + k)
        va_t = torch.from_numpy(va).to(device)
        for ep in range(epochs):
            net.train()
            perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)].to(device)
            for i in range(0, len(perm), batch):
                opt.zero_grad(set_to_none=True)
                loss = loss_on(perm[i:i + batch])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(sum(loss_on(va_t[i:i + 65536]) * len(va_t[i:i + 65536])
                               for i in range(0, len(va_t), 65536)) / max(len(va_t), 1))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        model.states[k] = best_state
        model._nets.clear()
        prev_state = best_state
        log(f"  stage {k}: {int(M.sum()):,} targets, held-out mse {best * t_scale ** 2:.5f} "
            f"(target rms {t_scale:.4f}), {ep + 1} epochs")
    model._nets.clear()
    return model

"""Invariance penalties with stocks grouped into environments (review round 4, R6; exploratory, validation only).

    python experiments/inv_fqi.py --method vrex --strength 1 10 --reps R0r R0info --seeds 0 --device cuda
    python experiments/transfer.py eval --episodes val_v0 --models 'vrexnn_*' 'gdronn_*' 'irmnn_*' --out val_v0_inv

Neural fitted Q-iteration exactly as invex.agents.fqi_nn.fit (same network, inputs, targets, warm starts and early
stopping), with the stage regression's loss replaced by an environment-aware one. Environments are the nine cells of
the training episodes' price level (trailing-window mid) by trade rate (trades per second in the window), in terciles
of the training set, so they differ in exactly the scales the units remove. Per stage and minibatch, with R_e the
masked squared error of environment e:
  * vrex   (Krueger et al. 2021): mean_e R_e + beta Var_e R_e;
  * gdro   (Sagawa et al. 2020): sum_e q_e R_e, with q_e proportional to exp(eta * running R_e) (eta = strength/100);
  * irm    (Arjovsky et al. 2019, IRMv1): sum_e R_e + lambda sum_e ||d R_e(w f) / dw at w = 1||^2, the penalty switched on
    after the first epoch.
The model is saved as an ordinary neural fitted-Q policy (kind nn_advantage), tag <method>nn, with the strength in the
representation field's suffix only through the tag (e.g. vrexnn10), so transfer.py evaluates it unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import K, OUT, load_transitions  # noqa: E402

from invex.agents import fqi_nn  # noqa: E402
from invex.agents.fqi import rep_inputs  # noqa: E402
from invex.agents.fqi_adv import NON_CROSS  # noqa: E402
from invex.env import execution as ex  # noqa: E402
from invex.env.features import REWARD_SYSTEM  # noqa: E402
from invex.env.units import unit_frame  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402


def environments(eps: pd.DataFrame) -> np.ndarray:
    """Nine environments: price-level tercile x trade-rate tercile of the trailing window (training-set cuts)."""
    price = np.log(eps["w_mid"].to_numpy(float))
    rate = np.log(eps["w_trades"].to_numpy(float).clip(1) / eps["w_seconds"].to_numpy(float).clip(1))
    cp = np.quantile(price, [1 / 3, 2 / 3])
    cr = np.quantile(rate, [1 / 3, 2 / 3])
    return (np.digitize(price, cp) * 3 + np.digitize(rate, cr)).astype(np.int64)


def fit(data, eps, uf, rep, c_lookback, method, strength, seed=0, device="cpu", epochs=40, patience=4, batch=4096,
        lr=1e-3, hidden=256, depth=3, log=print) -> fqi_nn.NNAdvModel:
    torch = fqi_nn._torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
    n = len(eps)
    hold = rng.random(n) < 0.05
    env = environments(eps)
    n_env = int(env.max()) + 1
    model = fqi_nn.NNAdvModel(rep, K, c_lookback, hidden, depth, [None] * K, [None] * K, [None] * K, [None] * K)
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
        L = fqi_nn.slog(X.astype(np.float64))
        mu = np.nanmean(np.where(np.isfinite(L), L, np.nan), axis=0)
        sd = np.nanstd(np.where(np.isfinite(L), L, np.nan), axis=0)
        mu, sd = np.nan_to_num(mu), np.where(np.nan_to_num(sd) > 1e-8, np.nan_to_num(sd), 1.0)
        model.mu[k], model.sd[k] = mu, sd
        t_scale = float(np.sqrt((Y[M] ** 2).mean())) or 1.0
        model.t_scale[k] = t_scale
        Z = model.inputs(k, X)
        net = fqi_nn._net(Z.shape[1], hidden, depth)
        if prev_state is not None:
            net.load_state_dict(prev_state)
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        tr, va = np.flatnonzero(~hold), np.flatnonzero(hold)
        Zt = torch.from_numpy(Z).to(device)
        Yt = torch.from_numpy((Y / t_scale).astype(np.float32)).to(device)
        Mt = torch.from_numpy(M.astype(np.float32)).to(device)
        Et = torch.from_numpy(env).to(device)
        q = torch.ones(n_env, device=device) / n_env  # GroupDRO weights

        def env_risks(idx, w=None):
            out = net(Zt[idx])
            if w is not None:
                out = out * w
            se = ((out - Yt[idx]) ** 2 * Mt[idx]).sum(dim=1)
            cnt = Mt[idx].sum(dim=1)
            e = Et[idx]
            num = torch.zeros(n_env, device=device).index_add_(0, e, se)
            den = torch.zeros(n_env, device=device).index_add_(0, e, cnt)
            present = den > 0
            return num[present] / den[present], present

        def plain(idx):
            out = net(Zt[idx])
            return ((out - Yt[idx]) ** 2 * Mt[idx]).sum() / Mt[idx].sum().clamp(min=1.0)

        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + k)
        va_t = torch.from_numpy(va).to(device)
        for ep in range(epochs):
            net.train()
            perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)].to(device)
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                opt.zero_grad(set_to_none=True)
                if method == "vrex":
                    r, _ = env_risks(idx)
                    loss = r.mean() + strength * (r.var(unbiased=False) if len(r) > 1 else 0.0)
                elif method == "gdro":
                    r, present = env_risks(idx)
                    with torch.no_grad():
                        q[present] = q[present] * torch.exp(strength / 100.0 * r.detach())
                        q /= q.sum()
                    loss = (q[present] / q[present].sum() * r).sum()
                elif method == "irm":
                    w = torch.tensor(1.0, device=device, requires_grad=True)
                    r, _ = env_risks(idx, w)
                    pen = 0.0
                    if ep >= 1:
                        for re_ in r:
                            gw = torch.autograd.grad(re_, w, create_graph=True)[0]
                            pen = pen + gw ** 2
                    loss = r.sum() + strength * pen
                    loss = loss / max(1.0, float(strength))  # keep the step size comparable (Gulrajani and Lopez-Paz)
                else:
                    raise ValueError(method)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():  # early stopping on the plain held-out error, as fqi_nn
                vl = float(sum(plain(va_t[i:i + 65536]) * len(va_t[i:i + 65536])
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
        log(f"  stage {k}: held-out mse {best * t_scale ** 2:.5f}, {ep + 1} epochs")
    model._nets.clear()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["vrex", "gdro", "irm"])
    ap.add_argument("--strength", nargs="+", type=float, required=True)
    ap.add_argument("--reps", nargs="+", default=["R0r", "R0info"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ko = load_splits()["ko_constants"]
    eps, data = load_transitions(["train_v0"])
    uf = unit_frame(eps, ko)
    c_r0 = float(np.median(uf["R3"]["t"]))
    for s in a.strength:
        tag = f"{a.method}nn{int(s) if s == int(s) else str(s).replace('.', 'p')}"
        for seed in a.seeds:
            for rep in a.reps:
                path = OUT / "models" / f"{tag}_{rep}_s{seed}.pkl"
                if path.exists():
                    continue
                t0 = time.time()
                m = fit(data, eps, uf, rep, c_r0 if rep.startswith("R0") else 1.0, a.method, s, seed=seed,
                        device=a.device, log=lambda x: None)
                m.save(path)
                print(f"{tag} {rep} seed {seed}: {time.time() - t0:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main()

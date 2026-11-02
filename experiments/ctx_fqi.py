"""A learned context encoder in place of hand-designed units (review round 4, R5/R6; exploratory).

    python experiments/ctx_fqi.py build --source train_v0          # per-episode context series for the training set
    python experiments/ctx_fqi.py build-val --episodes val_v0       # and for an evaluation table
    python experiments/ctx_fqi.py fit --seeds 0 1 2 --device cuda
    python experiments/ctx_fqi.py eval --episodes val_v0 --out val_v0_ctx

The policy sees the raw features (physical units) and, as context, the stock's trailing 30 minutes as a time series:
for each of the 30 minutes before the episode starts, the number of trades, the volume, the time-averaged spread and
touch depth, and the log mid return, from the same per-second calibration as the units (invex.book.calib), in physical
units. A small encoder (two layers over the 150 numbers) maps the series to a 16-dimensional context that enters the
advantage network with the raw features; encoder and network are trained jointly, stage by stage, exactly as neural
fitted Q-iteration (invex.agents.fqi_nn: inputs through sign(x) log(1 + |x|), standardised with training means; masked
squared error; warm starts; early stopping on 5% of episodes). The cost is in spreads, as for RAW+INFO, so the encoder
must learn from the series what the units and the window statistics supply by hand. Models go to
results/transfer/models_ctx/ (a kind transfer.py does not know), and 'eval' scores them with transfer.py's replay and
baselines. Reads no sealed table unless --unseal is given (logged).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import K, OUT, TRANS, load_transitions, pool  # noqa: E402

from invex.agents import fqi_nn  # noqa: E402
from invex.agents.fqi import rep_inputs  # noqa: E402
from invex.agents.fqi_adv import NON_CROSS  # noqa: E402
from invex.book.calib import calibrate  # noqa: E402
from invex.env import execution as ex  # noqa: E402
from invex.env.episodes import excluded  # noqa: E402
from invex.env.features import FlowBook, r1_stats  # noqa: E402
from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval import policies as P  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402

MINUTES, N_SERIES, D_CTX = 30, 5, 16
MODELS = OUT / "models_ctx"
BASE_REP = "R0r"  # raw features, cost in spreads


def series_task(args):
    venue, day, stock, t0s = args
    book, o, c, tick = book_of(venue, day, stock)
    v = venue_of(venue)
    cal = calibrate(book, o, c, tick, v.px_scale, v.qty_scale)
    out = np.zeros((len(t0s), MINUTES, N_SERIES), np.float32)
    for j, t in enumerate(t0s):
        b = cal._s(int(t))
        edges = [max(0, b - 60 * (MINUTES - i)) for i in range(MINUTES + 1)]
        for i in range(MINUTES):
            a_, b_ = edges[i], edges[i + 1]
            secs = max(b_ - a_, 1)
            m0, m1 = cal.mid[a_], cal.mid[b_]
            out[j, i] = (cal.trades[b_] - cal.trades[a_], cal.volume[b_] - cal.volume[a_],
                         (cal.spread[b_] - cal.spread[a_]) / secs, (cal.depth[b_] - cal.depth[a_]) / secs,
                         np.log(m1 / m0) * 1e4 if m0 > 0 and m1 > 0 else 0.0)
    return venue, day, stock, out


def build_series(eps: pd.DataFrame, workers: int) -> np.ndarray:
    eps = eps.reset_index(drop=True)
    groups = list(eps.groupby(["venue", "day", "stock"]).groups.items())
    parts = pool(series_task, [(v, d, s, eps.loc[idx, "t0"].to_numpy()) for (v, d, s), idx in groups], workers)
    out = np.zeros((len(eps), MINUTES, N_SERIES), np.float32)
    for ((_v, _d, _s), idx), (_a, _b, _c, arr) in zip(groups, parts):
        out[np.asarray(idx)] = arr
    return out


# --- the model ------------------------------------------------------------------------------------

def _nets(d_raw: int, hidden: int = 256, depth: int = 3):
    torch = fqi_nn._torch()
    nn = torch.nn

    class CtxNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(MINUTES * N_SERIES, 128), nn.SiLU(), nn.Linear(128, D_CTX))
            layers, d = [], d_raw + D_CTX
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            layers.append(nn.Linear(d, len(NON_CROSS)))
            self.head = nn.Sequential(*layers)

        def forward(self, x, s):
            return self.head(torch.cat([x, self.enc(s)], dim=1))

    return CtxNet()


class CtxModel:
    """Per stage: the network's state, the standardisation of the raw inputs and of the series, the target scale."""

    def __init__(self, K: int, c_lookback: float):
        self.K, self.c_lookback, self.rep = K, c_lookback, BASE_REP
        self.states, self.mu, self.sd, self.smu, self.ssd, self.t_scale = ([None] * K for _ in range(6))
        self._nets = {}

    def inputs(self, k, X, S):
        Z = (fqi_nn.slog(X.astype(np.float64)) - self.mu[k]) / self.sd[k]
        Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=fqi_nn.CLIP, neginf=-fqi_nn.CLIP), -fqi_nn.CLIP, fqi_nn.CLIP)
        W = (fqi_nn.slog(S.reshape(len(S), -1).astype(np.float64)) - self.smu[k]) / self.ssd[k]
        W = np.clip(np.nan_to_num(W, nan=0.0), -fqi_nn.CLIP, fqi_nn.CLIP)
        return Z.astype(np.float32), W.astype(np.float32)

    def net(self, k, device="cpu"):
        if (k, device) not in self._nets:
            n = _nets(len(self.mu[k]))
            n.load_state_dict(self.states[k])
            n.eval()
            self._nets[(k, device)] = n.to(device)
        return self._nets[(k, device)]

    def q_values(self, k, X, S, device="cpu"):
        torch = fqi_nn._torch()
        Z, W = self.inputs(k, X, S)
        out = np.zeros((len(Z), ex.N_ACT))
        with torch.no_grad():
            pred = self.net(k, device)(torch.from_numpy(Z).to(device), torch.from_numpy(W).to(device))
        pred = pred.double().cpu().numpy() * self.t_scale[k]
        for i, a in enumerate(NON_CROSS):
            out[:, a] = pred[:, i]
        return out

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if k != "_nets"}
        with open(path, "wb") as fh:
            pickle.dump(d, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = cls(d["K"], d["c_lookback"])
        m.__dict__.update(d)
        m._nets = {}
        return m


def fit(data, eps, S, uf, c_lookback, seed=0, device="cpu", epochs=40, patience=4, batch=4096, lr=1e-3, log=print):
    torch = fqi_nn._torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from invex.env.features import REWARD_SYSTEM

    norm = eps["Q"].to_numpy(float) * uf[REWARD_SYSTEM[BASE_REP]]["p"].to_numpy()
    n = len(eps)
    hold = rng.random(n) < 0.05
    model = CtxModel(K, c_lookback)
    Sflat = fqi_nn.slog(S.reshape(n, -1).astype(np.float64))
    prev = None
    for k in range(K - 1, -1, -1):
        X = rep_inputs(data["raw"][k], eps, uf, BASE_REP, c_lookback)
        m = data["masks"][k].astype(bool)
        c_now = data["costs"][k][:, ex.CROSS]
        y = (data["costs"][k] - c_now[:, None]) / norm[:, None]
        if k < K - 1:
            for a in NON_CROSS:
                Xn = rep_inputs(data["raw_next"][k, a], eps, uf, BASE_REP, c_lookback)
                An = model.q_values(k + 1, Xn, S, device)
                mn = data["masks_next"][k, a].astype(bool)
                An[~mn] = np.inf
                An[:, ex.CROSS] = 0.0
                v = data["cross_next"][k, a] / norm + np.minimum(0.0, An.min(axis=1))
                y[:, a] += np.where(data["done_next"][k][:, a] == 1, 0.0, v)
        Y = y[:, list(NON_CROSS)]
        M = m[:, list(NON_CROSS)] & np.isfinite(Y)
        Y = np.where(M, Y, 0.0)
        L = fqi_nn.slog(X.astype(np.float64))
        mu, sd = np.nanmean(np.where(np.isfinite(L), L, np.nan), 0), np.nanstd(np.where(np.isfinite(L), L, np.nan), 0)
        model.mu[k], model.sd[k] = np.nan_to_num(mu), np.where(np.nan_to_num(sd) > 1e-8, np.nan_to_num(sd), 1.0)
        model.smu[k], s_sd = Sflat.mean(0), Sflat.std(0)
        model.ssd[k] = np.where(s_sd > 1e-8, s_sd, 1.0)
        t_scale = float(np.sqrt((Y[M] ** 2).mean())) or 1.0
        model.t_scale[k] = t_scale
        Z, W = model.inputs(k, X, S)
        net = _nets(Z.shape[1])
        if prev is not None:
            net.load_state_dict(prev)
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
        tr, va = np.flatnonzero(~hold), np.flatnonzero(hold)
        Zt, Wt = torch.from_numpy(Z).to(device), torch.from_numpy(W).to(device)
        Yt = torch.from_numpy((Y / t_scale).astype(np.float32)).to(device)
        Mt = torch.from_numpy(M.astype(np.float32)).to(device)

        def loss_on(idx):
            out = net(Zt[idx], Wt[idx])
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
        prev = best_state
        log(f"  stage {k}: held-out mse {best * t_scale ** 2:.5f}, {ep + 1} epochs")
    model._nets.clear()
    return model


class CtxPolicy:
    def __init__(self, model: CtxModel, series: np.ndarray, name: str):
        self.model, self.series, self.name = model, series, name

    def act(self, k, raw, masks, eps, uf):
        X = rep_inputs(raw, eps, uf, self.model.rep, self.model.c_lookback)
        q = self.model.q_values(k, X, self.series)
        q[masks == 0] = np.inf
        return np.argmin(q, axis=1)


def eval_task(args):
    day, stock, eps, series, paths, ko = args
    venue = str(eps["venue"].iloc[0])
    book, o, c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True).copy()
    for col, v in r1_stats(FlowBook.build(book), eps, tick=tick).items():
        eps[col] = v
    uf = unit_frame(eps, ko)
    pols = [P.Immediate(), P.PostThenCross()]
    pols += [CtxPolicy(CtxModel.load(p), series, os.path.basename(p)[:-4]) for p in paths]
    res = P.evaluate_book(book, eps, uf, pols, K, venue_of(venue).latency_ns, tick=tick)  # the exact fill rule
    ex_ = pd.Series(excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy()), index=eps["key"])
    res["excluded"] = res["key"].map(ex_).to_numpy(bool)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", default="train_v0")
    b.add_argument("--workers", type=int, default=16)
    bv = sub.add_parser("build-val")
    bv.add_argument("--episodes", default="val_v0")
    bv.add_argument("--workers", type=int, default=16)
    bv.add_argument("--unseal", action="store_true")
    bv.add_argument("--reason", default="")
    f = sub.add_parser("fit")
    f.add_argument("--seeds", nargs="+", type=int, default=[0])
    f.add_argument("--device", default="cuda")
    e = sub.add_parser("eval")
    e.add_argument("--episodes", default="val_v0")
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--unseal", action="store_true")
    e.add_argument("--reason", default="")
    a = ap.parse_args()
    t0 = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        eps = pd.read_parquet(TRANS / a.source / "eps.parquet")
        np.save(TRANS / f"ctx_{a.source}.npy", build_series(eps, a.workers))
        print(f"ctx_{a.source}: {len(eps):,} episodes, {time.time() - t0:.0f}s")
    elif a.cmd == "build-val":
        eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
        np.save(OUT / f"ctx_{a.episodes}.npy", build_series(eps, a.workers))
        print(f"ctx_{a.episodes}: {len(eps):,} episodes, {time.time() - t0:.0f}s")
    elif a.cmd == "fit":
        ko = load_splits()["ko_constants"]
        eps, data = load_transitions(["train_v0"])
        S = np.load(TRANS / "ctx_train_v0.npy")
        uf = unit_frame(eps, ko)
        c_r0 = float(np.median(uf["R3"]["t"]))
        for seed in a.seeds:
            path = MODELS / f"ctxnn_R0r_s{seed}.pkl"
            if path.exists():
                continue
            m = fit(data, eps, S, uf, c_r0, seed=seed, device=a.device, log=lambda s: None)
            m.save(path)
            print(f"ctxnn seed {seed}: {time.time() - t0:.0f}s -> {path.name}", flush=True)
    else:
        ko = load_splits()["ko_constants"]
        eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason).reset_index(drop=True)
        S = np.load(OUT / f"ctx_{a.episodes}.npy")
        paths = sorted(str(p) for p in MODELS.glob("ctxnn_*.pkl"))
        jobs = [(d, s, g, S[g.index.to_numpy()], paths, ko) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
        res = pd.concat(pool(eval_task, jobs, a.workers), ignore_index=True)
        keep = ["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]
        res = res.merge(eps[keep], on="key")
        res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
        res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
        res.to_parquet(OUT / f"eval_{a.out}.parquet", index=False)
        write_json(OUT / f"eval_{a.out}.json", {"episodes": a.episodes, "models": [os.path.basename(p) for p in paths],
                                                "rows": len(res), "seconds": round(time.time() - t0),
                                                "provenance": provenance(args=vars(a))})
        print(res.groupby("policy")["c_spread"].mean().round(4).to_string())


if __name__ == "__main__":
    main()

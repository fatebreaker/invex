"""Train PPO policies in the replay on the Nasdaq training stock-days (invex.agents.ppo).

    python experiments/ppo.py --rep R3tick --seed 0 --tag zeroppo --workers 24 --device cuda:0

The training episodes are those of the training transitions (`--data`, default train_v0: the 2019 training-split
stock-days fitted Q-iteration learned from). Each iteration replays the current stochastic policy on a random
`--frac` of the stock-days, held by persistent workers, and takes `--epochs` passes of clipped-surrogate updates.
The final policy is saved as results/transfer/models/<tag>_<rep>_s<seed>.pkl, the learning curve as
results/transfer/ppo/<tag>_<rep>_s<seed>.json.

A seen policy, as transfer.py fit: `--data train_v0 valseen_v0 --size-of train_v0` (the same random episodes per seed).
R0pop: raw units with each episode's reward divided by a per-stock scale learned from the training transitions
(invex.agents.fqi_nn.stock_scale, as fitted Q-iteration's R0pop), so that the critic's targets and the advantages
have a common scale across stocks (PopArt-style). R3tickp with `--profile NAME`: MS+tick plus the stock's probe
profile (experiments/stock_profile.py), leave-one-day-out in training as for fitted Q-iteration. `--planted NAME`
replays a world planted in the replay itself (a per-stock markout or hidden queue of stock_profile.py) with each
stock's rule, as transfer.py gen and eval do: a positive control for PPO with the profile.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from invex.paths import DATA, RESULTS
from invex.provenance import provenance, write_json

TRANS = DATA / "transitions"
OUT = RESULTS / "transfer"
K = 10


def worker_main(conn, shard, rep, seed, c_lookback, rules=None):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from invex.agents.ppo import PPOModel, rollout_book
    from invex.env.features import REWARD_SYSTEM, FlowBook
    from invex.env.units import unit_frame
    from invex.eval.freeze import book_of
    from invex.eval.universe import load_splits
    from invex.sim.tracker import EXACT
    from invex.venues import venue as venue_of

    ko = load_splits()["ko_constants"]
    books = []
    for (venue, day, stock), e in shard:
        book, _o, _c, tick = book_of(venue, day, stock)
        e = e.reset_index(drop=True)
        uf = unit_frame(e, ko)
        norm = e["Q"].to_numpy(float) * uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
        if "_norm" in e:  # R0pop: the per-stock scale learned from the training transitions
            norm = e["_norm"].to_numpy(float)
        rule = (rules or {}).get(stock, EXACT)  # a planted world's per-stock rule, else the exact replay
        books.append((book, FlowBook.build(book), book.arrays(), e, uf, norm, tick, venue_of(venue).latency_ns, rule))
    conn.send(("ready", len(books)))
    rng = np.random.default_rng(seed)
    while True:
        cmd, payload = conn.recv()
        if cmd == "stop":
            break
        params, frac, greedy = payload
        model = PPOModel.from_dict(params)
        outs = []
        for (book, fb, arrs, e, uf, norm, tick, lat, rule) in books:
            if rng.random() >= frac:
                continue
            outs.append(rollout_book(model, book, fb, arrs, e, uf, norm, K, lat, tick, rng, rule, greedy))
        if outs:
            res = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        else:
            res = {}
        conn.send(("data", res))


def build_nets(d_in, hidden, device):
    import torch

    def net(d_out):
        return torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                   torch.nn.SiLU(), torch.nn.Linear(hidden, d_out)).to(device)

    pi, v = net(5), net(1)
    with torch.no_grad():  # start near uniform over the valid actions
        pi[-1].weight.mul_(0.01)
        pi[-1].bias.zero_()
    return pi, v


def layers_of(net) -> list:
    lin = [m for m in net if hasattr(m, "weight")]
    return [(m.weight.detach().cpu().numpy().T.astype(np.float32).copy(),
             m.bias.detach().cpu().numpy().astype(np.float32).copy()) for m in lin]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="zeroppo")
    ap.add_argument("--data", nargs="+", default=["train_v0"])
    ap.add_argument("--size-of", default="", help="subsample the episodes to this set's size (a seen policy)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--frac", type=float, default=0.35, help="share of training stock-days replayed per iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, nargs=2, default=[0.01, 0.001], help="entropy bonus, start and end")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--init", default="", help="a saved PPO model (name in results/transfer/models) to start from: its "
                    "weights and input statistics (fine-tuning on --data; review round 1)")
    ap.add_argument("--vscale", default="none", choices=["none", "auto"],
                    help="value targets as they are (registered), or divided by the standard deviation of the first "
                         "iteration's returns, so the critic's Huber loss sits in the same regime whatever the reward "
                         "unit (a parity check, review round 1)")
    ap.add_argument("--books", type=int, default=0, help="use only this many stock-days (smoke tests)")
    ap.add_argument("--profile", default="", help="probe profile table for R3tickp (leave-one-day-out in training)")
    ap.add_argument("--planted", default="", help="a planted world (profile table with kappa or omega) to train in")
    a = ap.parse_args(argv)

    import torch

    from invex.agents.ppo import PPOModel, gae, input_stats
    from invex.env.units import unit_frame
    from invex.eval.universe import load_splits

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    eps = pd.concat([pd.read_parquet(TRANS / d / "eps.parquet") for d in a.data], ignore_index=True)
    zs = [np.load(TRANS / d / "data.npz", mmap_mode="r") for d in a.data]
    keep = np.arange(len(eps))
    if a.size_of:  # as transfer.py fit
        n_target = len(pd.read_parquet(TRANS / a.size_of / "eps.parquet", columns=["key"]))
        keep = np.sort(np.random.default_rng(a.seed).choice(len(eps), size=min(n_target, len(eps)), replace=False))
        eps = eps.iloc[keep].reset_index(drop=True)
    if a.profile:  # as transfer.py fit --profile: each training episode sees the profile of its stock's other days
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from transfer import attach_profile
        eps = attach_profile(eps, a.profile, lodo=True)
    base = "R0" if a.rep == "R0pop" else a.rep
    if a.rep == "R0pop":
        from invex.agents.fqi_nn import stock_scale
        cat = {k: np.concatenate([np.asarray(z[k]) for z in zs], axis=1)[:, keep] for k in ("costs", "masks")}
        eps["_norm"] = stock_scale(cat, eps)
        del cat
    c_lookback = 1.0
    if a.rep in ("R0", "R0desc", "R0r", "R0pop", "R0aug"):  # as invex.agents.fqi: R0's lookback at the others' median
        c_lookback = float(np.median(unit_frame(eps, load_splits()["ko_constants"])["R3"]["t"]))

    class _Raw:  # the training transitions' raw features, stage by stage, for the kept episodes
        shape = (zs[0]["raw"].shape[0],)

        def __getitem__(self, k):
            return np.concatenate([np.asarray(z["raw"][k]) for z in zs])[keep]

    mu, sd = input_stats({"raw": _Raw()}, eps, unit_frame(eps, load_splits()["ko_constants"]), base, c_lookback,
                         seed=a.seed)
    init = None
    if a.init:  # fine-tuning: the starting policy's own input statistics, so its inputs mean what they meant
        import pickle
        with open(OUT / "models" / f"{a.init}.pkl", "rb") as fh:
            init = pickle.load(fh)
        if init.get("kind") != "ppo" or init["rep"] != base or float(init["c_lookback"]) != c_lookback:
            raise SystemExit(f"{a.init} is not a PPO model of {base} with the same lookback")
        mu, sd = np.asarray(init["mu"]), np.asarray(init["sd"])
    groups = [(k, g) for k, g in eps.groupby(["venue", "day", "stock"])]
    if a.books:
        groups = [groups[i] for i in np.sort(rng.choice(len(groups), a.books, replace=False))]
    order = np.argsort([-len(g) for _, g in groups])
    shards = [[] for _ in range(a.workers)]
    for i, j in enumerate(order):  # round-robin by size: similar work per worker
        shards[i % a.workers].append(groups[j])
    rules = {}
    if a.planted:  # per-stock tracker rules of a planted world, as transfer.py gen
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from transfer import planted_rules
        rules = planted_rules(a.planted)
        if not rules:
            raise SystemExit(f"{a.planted} plants nothing in the replay")
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    t0 = time.time()
    for w in range(a.workers):
        p_, c_ = ctx.Pipe()
        pr = ctx.Process(target=worker_main, args=(c_, shards[w], base, a.seed * 1000 + w, c_lookback, rules),
                         daemon=True)
        pr.start()
        conns.append(p_)
        procs.append(pr)
    n_books = sum(c.recv()[1] for c in conns)
    print(f"{a.tag} {a.rep} seed {a.seed}: {n_books} stock-days loaded by {a.workers} workers in "
          f"{time.time() - t0:.0f}s", flush=True)

    d_in = len(mu) + K
    pi, v = build_nets(d_in, a.hidden, a.device)
    if init is not None:
        with torch.no_grad():
            for net, layers in ((pi, init["pi"]), (v, init["v"])):
                lin = [m for m in net if hasattr(m, "weight")]
                if len(lin) != len(layers):
                    raise SystemExit(f"{a.init}: {len(layers)} layers, the network has {len(lin)}")
                for m, (W, b) in zip(lin, layers):
                    m.weight.copy_(torch.from_numpy(np.asarray(W).T.copy()))
                    m.bias.copy_(torch.from_numpy(np.asarray(b).copy()))
        print(f"  started from {a.init}", flush=True)
    opt = torch.optim.Adam(list(pi.parameters()) + list(v.parameters()), lr=a.lr)
    curve = []
    vs = None  # the scale of the value targets (1 unless --vscale auto)
    for it in range(a.iters):
        t_it = time.time()
        model = PPOModel(base, K, c_lookback, mu, sd, layers_of(pi), layers_of(v))
        for c in conns:
            c.send(("rollout", (model.to_dict(), a.frac, False)))
        parts = [c.recv()[1] for c in conns]
        parts = [p for p in parts if p]
        B = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        n, k_ = B["R"].shape
        ep_cost = float(-B["R"].sum(axis=1).mean())
        Zf = torch.from_numpy(B["Z"].reshape(n * k_, -1)).to(a.device)
        with torch.no_grad():
            V = torch.cat([v(Zf[i:i + (1 << 17)]) for i in range(0, Zf.shape[0], 1 << 17)]).squeeze(1)
        if vs is None:
            vs = 1.0
            if a.vscale == "auto":  # fixed after the first iteration: returns of the initial policy
                vs = float(np.std(gae(B["R"], np.zeros_like(B["R"], dtype=np.float32), B["alive"], a.lam)[1]
                                  [B["alive"]])) or 1.0
                print(f"  value targets divided by {vs:.4f}", flush=True)
        V = V.cpu().numpy().reshape(n, k_) * vs * B["alive"]
        adv, ret = gae(B["R"], V.astype(np.float32), B["alive"], a.lam)
        # a decision exists where the episode is open and more than one action is valid
        use = (B["alive"] & (B["M"].sum(axis=2) > 1)).reshape(-1)
        val_use = B["alive"].reshape(-1)
        idx = np.flatnonzero(use)
        A_t = torch.from_numpy(adv.reshape(-1)).to(a.device)
        R_t = torch.from_numpy(ret.reshape(-1)).to(a.device)
        act = torch.from_numpy(B["A"].reshape(-1)).to(a.device)
        lp0 = torch.from_numpy(B["LP"].reshape(-1)).to(a.device)
        msk = torch.from_numpy(B["M"].reshape(n * k_, -1)).to(a.device)
        a_mu, a_sd = A_t[idx].mean(), A_t[idx].std().clamp(min=1e-8)
        ent_coef = a.ent[0] + (a.ent[1] - a.ent[0]) * it / max(a.iters - 1, 1)
        vidx = np.flatnonzero(val_use)
        stats = []
        for _ in range(a.epochs):
            perm = idx[rng.permutation(len(idx))]
            vperm = vidx[rng.permutation(len(vidx))]
            n_mb = max(1, len(perm) // a.minibatch)
            for b in range(n_mb):
                j = torch.from_numpy(perm[b::n_mb]).to(a.device)
                jv = torch.from_numpy(vperm[b::n_mb]).to(a.device)
                lg = pi(Zf[j]).masked_fill(~msk[j], -1e9)
                logp_all = torch.log_softmax(lg, dim=1)
                lp = logp_all.gather(1, act[j][:, None]).squeeze(1)
                ratio = torch.exp(lp - lp0[j])
                an = (A_t[j] - a_mu) / a_sd
                s1, s2 = ratio * an, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * an
                p_all = logp_all.exp()
                ent = -(p_all * torch.where(msk[j], logp_all, torch.zeros_like(logp_all))).sum(1).mean()
                v_loss = torch.nn.functional.smooth_l1_loss(v(Zf[jv]).squeeze(1), R_t[jv] / vs)
                loss = -torch.min(s1, s2).mean() - ent_coef * ent + 0.5 * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(v.parameters()), 1.0)
                opt.step()
                with torch.no_grad():
                    stats.append((float(ent), float((lp0[j] - lp).mean()), float(v_loss)))
        e_, kl, vl = np.mean(stats, axis=0)
        curve.append({"iter": it, "episodes": int(n), "cost": ep_cost, "entropy": e_, "kl": kl, "v_loss": vl,
                      "seconds": round(time.time() - t_it, 1)})
        if it % 5 == 0 or it == a.iters - 1:
            print(f"  iter {it}: {n:,} episodes, cost {ep_cost:.4f}, entropy {e_:.3f}, kl {kl:.4f}, v_loss {vl:.4f}, "
                  f"{time.time() - t_it:.1f}s", flush=True)
    model = PPOModel(base, K, c_lookback, mu, sd, layers_of(pi), layers_of(v))
    for c in conns:
        c.send(("stop", None))
    for pr in procs:
        pr.join(timeout=30)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}_{a.rep}_s{a.seed}"
    model.save(OUT / "models" / f"{name}.pkl")
    (OUT / "ppo").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "ppo" / f"{name}.json", {"curve": curve, "books": n_books, "provenance": provenance(args=vars(a))})
    print(f"{name}: saved, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

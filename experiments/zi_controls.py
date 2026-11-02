"""Synthetic positive and negative controls in the zero-intelligence market (invex.sim.zi).

    python experiments/zi_controls.py --control pc1    # exact similarity: random time/size/price scalings
    python experiments/zi_controls.py --control pc1x   # exact similarity, test scales outside the training range
    python experiments/zi_controls.py --control pc2    # relative tick varies (limit-order density alpha)
    python experiments/zi_controls.py --control pc3    # granularity varies (cancellation rate delta)

Every asset gets random scalings (time x 1/c, shares x s, prices x p), so raw
units (R0) never line up across assets. Test assets are simulated on two
"days" (independent paths, same law). Two FQI policies per representation:

  zero-shot  trained on the training assets only;
  seen       trained on the training assets plus the test assets' day 1, subsampled to the same size.

Both are scored on the test assets' day 2, on identical episodes.
Transfer gap TG = cost(zero-shot) - cost(seen), in spreads, per test asset.

What each control must show:
- PC1: TG ~ 0 for R3-family units and a clear TG for R0 (otherwise the pipeline is broken).
- PC2: tick-free units lose more as the test asset's tick moves away from the training mix; R3tick does not.
- PC3: a granularity change that R3d's depth units hide.

Writes results/zi_controls/<control>/{eval.parquet, summary.json}.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from invex.agents import fqi, fqi_adv
from invex.env.episodes import book_episodes, seed_of
from invex.env.units import unit_frame, valid_units
from invex.eval import policies as P
from invex.eval.stats import cluster_bootstrap
from invex.paths import RESULTS
from invex.provenance import provenance, write_json
from invex.sim.tracker import EXACT, prepare
from invex.sim.zi import ZIParams, simulate

K = 10
LATENCY = 100_000
SESSION_S = 6.5 * 3600
KO = {"c_t": 4.68, "c_q": 1.7e-4, "c_p": 0.028}  # any constants: they rescale all assets alike
THREADS = int(os.environ.get("INVEX_FIT_THREADS", "32"))


def make_assets(control: str, n: int, seed: int, tag: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        s = int(rng.choice([1, 3, 10, 30]))
        p = int(rng.choice([1, 2, 5, 10]))
        if control == "pc1x":  # extrapolation: training and test scales do not overlap
            if tag == "tr":
                c, s, p = float(np.exp(rng.uniform(np.log(0.3), 0.0))), int(rng.choice([1, 3])), int(rng.choice([1, 2]))
            else:
                c = float(np.exp(rng.uniform(np.log(1.5), np.log(4.0))))
                s, p = int(rng.choice([10, 30])), int(rng.choice([5, 10]))
        if control in ("pc1", "pc1x"):
            base = ZIParams(alpha=0.05, delta=0.01)
        elif control == "pc2":
            base = ZIParams(alpha=float(np.exp(rng.uniform(np.log(0.02), np.log(1.0)))), delta=0.01)
        elif control == "pc3":
            base = ZIParams(alpha=0.2, delta=float(np.exp(rng.uniform(np.log(0.002), np.log(0.05)))))
        else:
            raise ValueError(control)
        out.append({"name": f"{tag}{i:03d}", "params": base.scaled(c, s, p), "alpha": base.alpha,
                    "delta": base.delta, "c": c, "s": s, "p": p})
    return out


def _book(asset: dict, day: int):
    ev, o, c = simulate(asset["params"], SESSION_S, warmup_s=1800, seed=seed_of(asset["name"], day) % (2**31))
    return prepare(ev, o, c), o, c


ALGO = {"plain": fqi, "adv": fqi_adv}


def gen_task(args):
    asset, day, n, algo = args
    book, o, c = _book(asset, day)
    eps = book_episodes(book, "zi", f"d{day}", asset["name"], o, c, n, KO, seed_tag="zi-fit")
    if eps.empty:
        return None
    eps["tick"] = asset["params"].tick
    data = ALGO[algo].generate_book(book, eps, K, LATENCY, EXACT, asset["params"].tick,
                                    seed_of(asset["name"], day, "beh"))
    return eps, data


def eval_task(args):
    asset, day, n, model_paths = args
    book, o, c = _book(asset, day)
    eps = book_episodes(book, "zi", f"d{day}", asset["name"], o, c, n, KO, seed_tag="zi-eval")
    eps["tick"] = asset["params"].tick
    uf = unit_frame(eps, KO)
    eps = eps[valid_units(uf)].reset_index(drop=True)
    uf = unit_frame(eps, KO)
    pols = [P.Immediate(), P.PostThenCross()]
    pols += [P.FQIPolicy(fqi_adv.load_model(path), name=name) for name, path in model_paths]
    res = P.evaluate_book(book, eps, uf, pols, K, LATENCY, tick=asset["params"].tick)
    res = res.merge(eps[["key", "Q", "w_spread"]], on="key")
    res["asset"] = asset["name"]
    res["alpha"], res["delta"] = asset["alpha"], asset["delta"]
    return res


def pool(fn, jobs, workers):
    env = os.environ.copy()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["INVEX_PRED_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex_:
            return [r for r in ex_.map(fn, jobs, chunksize=1) if r is not None]
    finally:
        os.environ.clear()
        os.environ.update(env)


def fit_models(parts, reps, tag, out_dir, rounds, keep_n=None, seed=0, algo="plain"):
    eps = pd.concat([p[0] for p in parts], ignore_index=True)
    data = ALGO[algo].concat([p[1] for p in parts])
    uf = unit_frame(eps, KO)
    keep = np.flatnonzero(valid_units(uf))
    if keep_n is not None and len(keep) > keep_n:
        keep = np.sort(np.random.default_rng(seed).choice(keep, keep_n, replace=False))
    eps = eps.iloc[keep].reset_index(drop=True)
    data = fqi_adv.subset(data, keep) if algo == "adv" else \
        {k: (v[:, keep] if k in ("raw", "masks", "costs", "done_next") else v[:, :, keep]) for k, v in data.items()}
    uf = unit_frame(eps, KO)
    c_r0 = float(np.median(uf["R3"]["t"]))
    paths = []
    for rep in reps:
        m = ALGO[algo].fit(data, eps, uf, rep, K, c_r0 if rep.startswith("R0") else 1.0, n_rounds=rounds,
                           threads=THREADS, log=lambda s: None)
        path = out_dir / f"{tag}_{rep}.pkl"
        m.save(path)
        paths.append((f"{tag}_{rep}", str(path)))
    return paths, len(eps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True, choices=["pc1", "pc1x", "pc2", "pc3"])
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--reps", nargs="+", default=["R0", "R2", "R3", "R3d", "R3tick", "Rpi", "Rdim"])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--algo", default="plain", choices=list(ALGO))
    ap.add_argument("--out-tag", default="", help="suffix of the output directory (e.g. a run with other --reps)")
    a = ap.parse_args(argv)
    out = RESULTS / (("zi_controls" if a.algo == "plain" else f"zi_controls_{a.algo}") + a.out_tag) / a.control
    (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train = make_assets(a.control, a.n_train, 1, "tr")
    test = make_assets(a.control, a.n_test, 2, "te")
    tr_parts = pool(gen_task, [(x, 0, a.episodes, a.algo) for x in train] + [(x, 1, a.episodes, a.algo) for x in train],
                    a.workers)
    te_parts = pool(gen_task, [(x, 1, a.episodes, a.algo) for x in test], a.workers)
    zero_paths, n_fit = fit_models(tr_parts, a.reps, "zero", out / "models", a.rounds, algo=a.algo)
    # seen: training assets' day 0 plus the test assets' day 1, subsampled to the zero-shot size
    seen_parts = [p for p in tr_parts if p[0]["day"].iloc[0] == "d0"] + te_parts
    seen_paths, n_seen = fit_models(seen_parts, a.reps, "seen", out / "models", a.rounds, keep_n=n_fit, algo=a.algo)
    print(f"{a.control}: fitted on {n_fit:,} (zero-shot) and {n_seen:,} (seen) episodes in {time.time() - t0:.0f}s",
          flush=True)
    res = pd.concat(pool(eval_task, [(x, 2, a.episodes, zero_paths + seen_paths) for x in test], a.workers),
                    ignore_index=True)
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res.to_parquet(out / "eval.parquet", index=False)
    per = res.pivot_table(index=["asset", "key"], columns="policy", values="c_spread").reset_index()
    asset_info = res.groupby("asset")[["alpha", "delta"]].first()
    summary = {"control": a.control, "n_fit": n_fit, "n_seen": n_seen, "n_eval": int(per.shape[0]), "reps": {}}
    for rep in a.reps:
        z, s = f"zero_{rep}", f"seen_{rep}"
        tg = (per[z] - per[s]).groupby(per["asset"]).mean()
        ci = cluster_bootstrap(tg.to_numpy(), n_boot=4000)
        row = {"tg": ci, "cost_zero": float(per[z].mean()), "cost_seen": float(per[s].mean())}
        if a.control in ("pc2", "pc3"):
            knob = np.log(asset_info.loc[tg.index, "alpha" if a.control == "pc2" else "delta"].to_numpy())
            row["tg_slope_on_abs_log_knob"] = float(np.polyfit(np.abs(knob - np.median(knob)), tg.to_numpy(), 1)[0])
        summary["reps"][rep] = row
        print(f"{rep:7s} TG {ci['mean']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}] spreads; "
              f"zero-shot {row['cost_zero']:.4f}, seen {row['cost_seen']:.4f}", flush=True)
    for b in ("immediate", "post_then_cross"):
        summary[b] = float(per[b].mean())
    print("baselines:", {b: round(summary[b], 4) for b in ("immediate", "post_then_cross")}, flush=True)
    summary["provenance"] = provenance(args=vars(a))
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()

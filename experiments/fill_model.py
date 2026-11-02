"""A one-step fill-probability rule (fourth review round): an empirical fill model and a one-step decision.

While its order rests at the near touch (or is about to be posted there), the policy compares the cost of crossing
now, C, with the model's expected cost of waiting until the end,

    W = f * c_pass + (1 - f) * c_miss,

where f is the expected fraction of the remaining shares filled passively by the end, c_pass the price of those
fills and c_miss the per-share cost of the forced cross at the end (both relative to the arrival mid, so c_miss
carries the adverse selection of the orders that do not fill). It crosses if C < W and otherwise joins or holds.
Each of f, c_pass, c_miss and C is a LightGBM regression on the MS+tick state (costs per share in mean spreads; or,
with --rep R0, on raw features with costs in the raw price unit),
fitted on post-and-wait rollouts on the training stocks' training days: before it crosses, the policy visits exactly
the states of post-and-wait, so the data are its own. This is fill-probability execution in the spirit of Cartea et
al. (2015) and Moallemi and Yuan (2016), with the probabilities and prices estimated rather than modelled. With
--fees the decision also counts Nasdaq's taker fee and maker rebate. Exploratory, post hoc.

    python experiments/fill_model.py gen  --data train_v0 --per-book 40
    python experiments/fill_model.py fit  --seeds 0 1 2 [--rep R0]
    python experiments/fill_model.py eval --episodes val_v0 --out val_v0_fill [--models fill fillr0]
    python experiments/fill_model.py eval --episodes stage1_v0 --unseal --reason "..." --out stage1_fill

Rollouts: $INVEX_DATA/fillmodel/rollouts.parquet; models: results/transfer/models/fill_s<seed>.pkl (MS+tick) and
fillr0_s<seed>.pkl (raw units, --rep R0); evaluation: results/transfer/eval_<out>.parquet (policies <stem>_s<seed> and,
deciding net of fees, fillfee..._s<seed>). With --exact-cross (validation, after the sixth review) also
fillx..._s<seed>: the cost of crossing now read from the visible book wherever the far touch covers the remaining shares
(the far price against the arrival mid; the regression only where it does not).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer as TR  # noqa: E402

from invex.agents.fqi import LGB_PARAMS, rep_inputs  # noqa: E402
from invex.env import execution as ex  # noqa: E402
from invex.env.episodes import excluded  # noqa: E402
from invex.env.features import (F_ADV, F_FAR, F_Q, F_R_BEHIND, F_R_IN, F_R_NEAR, F_REM, F_SPREAD, N_RAW,  # noqa: E402
                                 REWARD_SYSTEM, FlowBook, raw_features)
from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval import policies as P  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.paths import DATA  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.sim import tracker as tk  # noqa: E402
from invex.sim.tracker import EXACT  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402

K = TR.K
TARGETS = ("f", "c_pass", "c_miss", "c_cross")
ROLL = DATA / "fillmodel"


# --- gen ------------------------------------------------------------------------------------------

def gen_task(args):
    """Post-and-wait on one stock-day; per episode and decision: the state's raw features, the cost of crossing
    now, and what waiting to the end brings (passive shares and their price, the forced cross)."""
    day, stock, eps = args
    book, _o, _c, tick = book_of("itch", day, stock)
    eps = eps.reset_index(drop=True)
    fb = FlowBook.build(book)
    arrs = book.arrays()
    sg, t0, T = eps["side"].to_numpy(np.int64), eps["t0"].to_numpy(np.int64), eps["T_ns"].to_numpy(np.int64)
    Q, mid0 = eps["Q"].to_numpy(np.int64), eps["mid0"].to_numpy(float)
    lat = venue_of("itch").latency_ns
    n = len(eps)
    st = ex.new_states(Q)
    raw = np.zeros((K, n, N_RAW))
    rem, live = np.zeros((K, n)), np.zeros((K, n), bool)
    c_cross, c_step = np.zeros((K, n)), np.zeros((K, n))
    passive, pcash = np.zeros((K, n)), np.zeros((K, n))
    for k in range(K):
        raw[k] = raw_features(fb, st, sg, mid0, t0, T, Q, k, K, tick)
        rem[k], live[k] = st[:, ex.S_REM], st[:, ex.S_DONE] == 0
        m = np.zeros((n, ex.N_ACT), np.int64)
        ex.masks_stage(*arrs, st, sg, t0, T, k, K, tick, m)
        sc, cc = st.copy(), np.zeros(n)
        ex.rollout_stage(*arrs, sc, sg, mid0, t0, T, k, K, np.full(n, ex.CROSS, np.int64), lat, tick, EXACT, cc)
        c_cross[k] = cc
        resting = st[:, tk.ST_ACTIVE] == 1
        a = np.where(~resting & (m[:, ex.JOIN] == 1), ex.JOIN, ex.HOLD).astype(np.int64)
        before, c = st.copy(), np.zeros(n)
        ex.rollout_stage(*arrs, st, sg, mid0, t0, T, k, K, a, lat, tick, EXACT, c)
        c_step[k] = c
        passive[k] = st[:, ex.S_PASSIVE] - before[:, ex.S_PASSIVE]
        pcash[k] = passive[k] * sg * (st[:, tk.ST_PX] - mid0)  # passive fills of this step, at the order's price
    # from decision k to the end
    W = np.cumsum(c_step[::-1], axis=0)[::-1]
    Pk = np.cumsum(passive[::-1], axis=0)[::-1]
    Ck = np.cumsum(pcash[::-1], axis=0)[::-1]
    rows = []
    for k in range(K):
        ok = live[k] & (rem[k] > 0)
        df = pd.DataFrame(raw[k][ok], columns=[f"x{i}" for i in range(N_RAW)])
        df["ep"] = np.flatnonzero(ok)
        df["k"] = k
        df["rem"] = rem[k][ok]
        df["passive"] = Pk[k][ok]
        df["pass_cost"] = Ck[k][ok]
        df["miss_cost"] = (W[k] - Ck[k])[ok]
        df["cross_cost"] = c_cross[k][ok]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["key"] = eps["key"].to_numpy()[out["ep"].to_numpy()]
    return out.drop(columns="ep")


def cmd_gen(a):
    eps = pd.read_parquet(TR.TRANS / a.data / "eps.parquet")
    eps = eps.groupby(["day", "stock"], group_keys=False).head(a.per_book)
    jobs = [(d, s, g) for (d, s), g in eps.groupby(["day", "stock"])]
    t0 = time.time()
    res = pd.concat(TR.pool(gen_task, jobs, a.workers), ignore_index=True)
    ROLL.mkdir(parents=True, exist_ok=True)
    res.to_parquet(ROLL / "rollouts.parquet", index=False)
    eps[eps["key"].isin(res["key"])].to_parquet(ROLL / "eps.parquet", index=False)
    print(f"rollouts: {len(res):,} decisions of {res['key'].nunique():,} episodes, {time.time() - t0:.0f}s", flush=True)


# --- fit ------------------------------------------------------------------------------------------

def lookback(rep: str, uf: dict) -> float:
    """As experiments/transfer.py fit: raw units measure time in the median MS time unit of the training data."""
    return float(np.median(uf["R3"]["t"])) if rep.startswith("R0") else 1.0


def design(roll: pd.DataFrame, eps: pd.DataFrame, ko, rep: str) -> tuple[np.ndarray, dict, float]:
    """Inputs of representation rep and the four targets (per share, in its reward unit: the mean spread for MS+tick,
    the price unit for raw units)."""
    e = eps.set_index("key").loc[roll["key"]].reset_index()
    uf = unit_frame(e, ko)
    raw = roll[[f"x{i}" for i in range(N_RAW)]].to_numpy()
    c_lb = lookback(rep, unit_frame(eps, ko))
    X = rep_inputs(raw, e, uf, rep, c_lb)
    up = uf[REWARD_SYSTEM[rep]]["p"].to_numpy()
    rem, pas = roll["rem"].to_numpy(float), roll["passive"].to_numpy(float)
    miss = rem - pas
    y = {"f": pas / rem,
         "c_pass": np.where(pas > 0, roll["pass_cost"].to_numpy() / np.maximum(pas, 1) / up, np.nan),
         "c_miss": np.where(miss > 0, roll["miss_cost"].to_numpy() / np.maximum(miss, 1) / up, np.nan),
         "c_cross": roll["cross_cost"].to_numpy() / rem / up}
    return X, y, c_lb


def cmd_fit(a):
    ko = load_splits()["ko_constants"]
    roll = pd.read_parquet(ROLL / "rollouts.parquet")
    eps = pd.read_parquet(ROLL / "eps.parquet")
    X, y, c_lb = design(roll, eps, ko, a.rep)
    stem = "fill" if a.rep == "R3tick" else f"fill{a.rep.lower()}"
    for seed in a.seeds:
        t0 = time.time()
        prm = dict(LGB_PARAMS, num_threads=a.threads, seed=seed)
        boosters = {}
        for name in TARGETS:
            ok = np.isfinite(y[name])
            boosters[name] = lgb.train(prm, lgb.Dataset(X[ok], y[name][ok]), num_boost_round=a.rounds).model_to_string()
        with open(TR.OUT / "models" / f"{stem}_s{seed}.pkl", "wb") as fh:
            pickle.dump({"kind": "fill_model", "rep": a.rep, "c_lookback": c_lb, "boosters": boosters}, fh)
        print(f"{stem} seed {seed}: {len(X):,} decisions, {time.time() - t0:.0f}s", flush=True)


# --- policy and eval ------------------------------------------------------------------------------

class FillModelPolicy:
    def __init__(self, path, fees: tuple[float, float] = (0.0, 0.0), name: str = "fill", exact_cross: bool = False):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self.models = {k: lgb.Booster(model_str=s) for k, s in d["boosters"].items()}
        self.rep, self.c_lb = d.get("rep", "R3tick"), d.get("c_lookback", 1.0)
        self.taker, self.rebate = fees
        self.name = name
        self.exact_cross = exact_cross

    def act(self, k, raw, masks, eps, uf):
        X = rep_inputs(raw, eps, uf, self.rep, self.c_lb)
        th = int(os.environ.get("INVEX_PRED_THREADS", "0"))
        p = {n: m.predict(X, num_threads=th) for n, m in self.models.items()}
        up = uf[REWARD_SYSTEM[self.rep]]["p"].to_numpy()
        f = np.clip(p["f"], 0.0, 1.0)
        wait_cost = f * (p["c_pass"] - self.rebate / up) + (1 - f) * (p["c_miss"] + self.taker / up)
        c_cross = p["c_cross"]
        if self.exact_cross:  # the visible book: the far touch against the arrival mid, where it covers the order
            rem = raw[:, F_REM] * raw[:, F_Q]
            book = (raw[:, F_SPREAD] > 0) & (rem <= raw[:, F_FAR])
            c_cross = np.where(book, (raw[:, F_ADV] + 0.5 * raw[:, F_SPREAD]) / up, c_cross)
        cross_cost = c_cross + self.taker / up
        resting = (raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND]) > 0
        wait = np.where(resting | (masks[:, ex.JOIN] == 0), ex.HOLD, ex.JOIN)
        return np.where(cross_cost < wait_cost, ex.CROSS, wait)


def eval_task(args):
    day, stock, eps, stems, exact = (*args, False)[:5]
    venue = str(eps["venue"].iloc[0])
    book, _o, _c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True)
    uf = unit_frame(eps, load_splits()["ko_constants"])
    pols = []
    for stem in stems:  # fill_s0 -> fill_s0 (gross decision) and fillfee_s0 (net); fillr0_s0 -> fillfeer0_s0
        path = TR.OUT / "models" / f"{stem}.pkl"
        pols += [FillModelPolicy(path, name=stem),
                 FillModelPolicy(path, TR.NASDAQ_FEES, name=stem.replace("fill", "fillfee", 1))]
        if exact:
            pols.append(FillModelPolicy(path, name=stem.replace("fill", "fillx", 1), exact_cross=True))
    res = P.evaluate_book(book, eps, uf, pols, K, venue_of(venue).latency_ns, tick=tick)
    ex_ = pd.Series(excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy()), index=eps["key"])
    res["excluded"] = res["key"].map(ex_).to_numpy(bool)
    return res


def cmd_eval(a):
    eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
    stems = [f"{m}_s{s}" for m in a.models for s in a.seeds]
    jobs = [(d, s, g, stems, a.exact_cross) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
    t0 = time.time()
    res = pd.concat(TR.pool(eval_task, jobs, a.workers), ignore_index=True)
    keep = ["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]
    res = res.merge(eps[keep], on="key")
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
    res.to_parquet(TR.OUT / f"eval_{a.out}.parquet", index=False)
    write_json(TR.OUT / f"eval_{a.out}.json", {"episodes": a.episodes, "seeds": a.seeds, "rows": len(res),
                                               "seconds": round(time.time() - t0),
                                               "provenance": provenance(args=vars(a))})
    print(f"{a.out}: {len(res):,} rows, {time.time() - t0:.0f}s", flush=True)
    print(res[~res["excluded"]].groupby("policy")["c_spread"].mean().round(4).to_string(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--data", default="train_v0")
    g.add_argument("--per-book", type=int, default=40)
    g.add_argument("--workers", type=int, default=24)
    f = sub.add_parser("fit")
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    f.add_argument("--rep", default="R3tick", choices=["R3tick", "R0"], help="state representation of the four models")
    f.add_argument("--rounds", type=int, default=300)
    f.add_argument("--threads", type=int, default=12)
    e = sub.add_parser("eval")
    e.add_argument("--episodes", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    e.add_argument("--models", nargs="+", default=["fill"], help="model stems: fill (MS+tick), fillr0 (raw units)")
    e.add_argument("--unseal", action="store_true")
    e.add_argument("--reason", default="")
    e.add_argument("--workers", type=int, default=24)
    e.add_argument("--exact-cross", action="store_true", help="also the rule with the cost of crossing read from the book")
    a = ap.parse_args(argv)
    {"gen": cmd_gen, "fit": cmd_fit, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

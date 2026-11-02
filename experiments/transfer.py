"""Zero-shot transfer on real data: generate transitions, fit FQI per representation, evaluate on frozen episodes.

    # transitions (branched, behaviour-policy states) for a split on some days
    python experiments/transfer.py gen  --name train_v0   --split train --days D1 ... D5
    python experiments/transfer.py gen  --name valseen_v0 --split val   --days D1 ... D5
    # models: zero-shot (training stocks only) and seen (+ the evaluation stocks' other days), size-matched
    python experiments/transfer.py fit  --data train_v0 --tag zero --reps R0 R2 R3 R3tick --seeds 0 1 2
    python experiments/transfer.py fit  --data train_v0 valseen_v0 --size-of train_v0 --tag seen ...
    # robustness arms (after the sealed tests): neural learner, learned normalisation, stock identity
    python experiments/transfer.py fit  --data train_v0 --tag zeronn --learner nn --device cuda --reps R0 R3tick R0aug
    python experiments/transfer.py fit  --data train_v0 --tag zero --reps R0pop
    python experiments/transfer.py fit  --data train_v0 valseen_v0 --size-of train_v0 --tag seenid --reps R3tickid
    # every policy on a frozen episode table (invex.eval.freeze)
    python experiments/transfer.py eval --episodes val_v0 --tags zero seen --out val_v0

Transitions live in $INVEX_DATA/transitions/<name>/ (eps.parquet, data.npz),
models in results/transfer/models/<tag>_<rep>_s<seed>.pkl, evaluations in
results/transfer/eval_<out>.parquet (one row per episode and policy, raw cost
plus spread and bps normalisations). Sealed tables need --unseal --reason.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from invex.agents import fqi_adv, fqi_nn
from invex.env.episodes import book_episodes, excluded, seed_of
from invex.env.features import PROFILE_COLS, FlowBook, r1_stats
from invex.env.units import unit_frame, valid_units
from invex.eval import freeze
from invex.eval import policies as P
from invex.eval.freeze import book_of, scales_of, stocks_for
from invex.eval.universe import load_splits
from invex.paths import DATA, RESULTS, ensure_space
from invex.provenance import provenance, write_json
from invex.sim.tracker import (DMARK_BASE, EXACT, HABS_BASE, HIDDEN_BASE, MARKOUT_BASE, MARKOUT_ZERO, OPTIMISTIC,
                               PESSIMISTIC)
from invex.venues import venue as venue_of

K = 10
OUT = RESULTS / "transfer"
# Nasdaq fees in the ITCH price unit (1e-4 dollars) per share: taker 0.30 cents, maker rebate 0.20 cents
# (as experiments/analyze_revision.py)
NASDAQ_FEES = (30.0, 20.0)
TRANS = DATA / "transitions"


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


# --- gen ------------------------------------------------------------------------------------------

def gen_task(args):
    venue, day, stock, n, ko, tag, taker, rebate, min_onetick, phi, psi, rule = (*args, 0.0, 0.0, 0.0, 0.0, 0.0, EXACT)[:12]
    book, o, c, tick = book_of(venue, day, stock)
    eps = book_episodes(book, venue, day, stock, o, c, n, ko, seed_tag=tag, tick=tick, **scales_of(venue, day, stock))
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    # --min-onetick: only episodes whose book was at one tick for at least this share of the trailing window
    eps = eps[valid_units(uf) & (eps["w_onetick"].to_numpy() >= min_onetick)].reset_index(drop=True)
    if eps.empty:
        return None
    for col, v in r1_stats(FlowBook.build(book), eps, tick=tick).items():  # R1's per-asset feature statistics
        eps[col] = v
    if phi:  # an injected per-stock crossing surcharge, phi spreads per share taken (stock_profile.py inject)
        taker = taker + phi * eps["w_spread"].to_numpy(float)
    if psi:  # an injected per-stock markout on passive fills, psi spreads per share filled while resting
        rebate = rebate - psi * eps["w_spread"].to_numpy(float)
    # rule: a planted per-stock structure in the replay itself (stock_profile.py, planted_rules), else the exact replay
    rule = int(rule)
    return eps, fqi_adv.generate_book(book, eps, K, venue_of(venue).latency_ns, rule, tick,
                                      seed_of(day, stock, tag, "behaviour"), taker=taker, rebate=rebate)


def hidden_rule(kappa: float) -> int:
    """The tracker rule for a planted hidden queue of kappa hidden shares per displayed share ahead (0: EXACT)."""
    return HIDDEN_BASE + int(round(100 * kappa)) if kappa else EXACT


def markout_rule(omega: float) -> int:
    """The tracker rule for a planted markout: passive fills pay omega times their markout one decision interval
    later (0: EXACT)."""
    return MARKOUT_BASE + MARKOUT_ZERO + int(round(1000 * omega)) if omega else EXACT


def dmarkout_rule(omega: float, mbar: float) -> int:
    """A planted demeaned markout: passive fills pay omega times (their markout less mbar spreads at the fill)."""
    return DMARK_BASE + 10_000 * (1000 + int(round(1000 * mbar))) + 5000 + int(round(100 * omega)) if omega else EXACT


def habs_rule(shares: float) -> int:
    """A planted hidden queue of this many shares at every resting price (0: EXACT)."""
    return HABS_BASE + int(round(shares)) if shares >= 1 else EXACT


def planted_rules(name: str) -> dict:
    """Per stock, the tracker rule of a world planted in the replay itself (stock_profile.py: kinds hidden, markout,
    dmarkout, habs); empty for a world whose planted cost is added to the costs (cross, passive)."""
    pr = load_profile(name)
    if "hshares" in pr:  # before kappa: a habs world also carries kappa, in trade sizes
        return {s: habs_rule(float(v)) for s, v in pr["hshares"].items()}
    if "kappa" in pr:
        return {s: hidden_rule(float(v)) for s, v in pr["kappa"].items()}
    if "omega" in pr and "mbar" in pr:
        return {s: dmarkout_rule(float(o), float(m)) for s, o, m in zip(pr.index, pr["omega"], pr["mbar"])}
    if "omega" in pr:
        return {s: markout_rule(float(v)) for s, v in pr["omega"].items()}
    return {}


def cmd_gen(a):
    out = TRANS / a.name
    if (out / "data.npz").exists():
        raise SystemExit(f"{out} exists")
    ko = load_splits()["ko_constants"]
    if a.fees != "none" and a.venue != "itch":
        raise SystemExit("--fees nasdaq is for the itch venue")
    taker, rebate = NASDAQ_FEES if a.fees == "nasdaq" else (0.0, 0.0)
    # --seed-tag: draw the same episodes and behaviour path as an earlier set (e.g. a net-of-fee copy of train_v0)
    tag = a.seed_tag or a.name
    inj = load_profile(a.surcharge) if a.surcharge else pd.DataFrame(columns=["phi", "psi"])
    phi, psi = inj["phi"], inj["psi"] if "psi" in inj else pd.Series(dtype=float)
    rules = planted_rules(a.surcharge) if a.surcharge else {}
    jobs = [(a.venue, d, s, a.per_book, ko, tag, taker, rebate, a.min_onetick, float(phi.get(s, 0.0)),
             float(psi.get(s, 0.0)), rules.get(s, EXACT))
            for d in a.days for s in stocks_for(a.split, d, a.venue)]
    ensure_space(out, len(jobs) * a.per_book * 7e3 / 1024**3)
    t0 = time.time()
    parts = pool(gen_task, jobs, a.workers)
    eps = pd.concat([p[0] for p in parts], ignore_index=True)
    data = fqi_adv.concat([p[1] for p in parts])
    out.mkdir(parents=True, exist_ok=True)
    eps.to_parquet(out / "eps.parquet", index=False)
    np.savez(out / "data.npz", **data)
    meta = {"name": a.name, "venue": a.venue, "split": a.split, "days": a.days, "per_book": a.per_book,
            "seed_tag": tag, "fees": {"taker": taker, "rebate": rebate}, "min_onetick": a.min_onetick,
            "surcharge": a.surcharge,
            "episodes": len(eps),
            "books": len(parts), "seconds": round(time.time() - t0), "provenance": provenance(args=vars(a))}
    write_json(out / "meta.json", meta)
    print(f"{a.name}: {len(eps):,} episodes from {len(parts)} books in {meta['seconds']}s", flush=True)


HISTORY_SETS = ("train_v0", "valseen_v0", "testseen_v0")  # every 2019-universe stock's five 2019 training days
HISTORY_REPS = ("R3h", "R3tickh")


def history_units() -> pd.DataFrame:
    """Per-stock constants from its own 2019 training days (the episodes of the training transitions): median inter-trade
    time, trade size and spread of the trailing windows. For R3h and R3tickh (invex.env.features)."""
    path = OUT / "history_units.parquet"
    if not path.exists():
        e = pd.concat([pd.read_parquet(TRANS / n / "eps.parquet", columns=["stock", "w_seconds", "w_trades", "w_volume",
                                                                            "w_spread"]) for n in HISTORY_SETS])
        tr = e["w_trades"].clip(lower=1)
        e = e.assign(h_inter=e["w_seconds"] / tr, h_tsize=e["w_volume"] / tr, h_spread=e["w_spread"])
        e.groupby("stock")[["h_inter", "h_tsize", "h_spread"]].median().reset_index().to_parquet(path, index=False)
    return pd.read_parquet(path)


def load_profile(name: str) -> pd.DataFrame:
    """A per-stock table of experiments/stock_profile.py, indexed by stock: the probe profile (PROFILE_COLS) and,
    for an injected world, the stock's surcharge phi."""
    return pd.read_parquet(OUT / "profiles" / f"{name}.parquet").set_index("stock")


def attach_profile(eps: pd.DataFrame, name: str, lodo: bool = False) -> pd.DataFrame:
    """eps with the stock's probe profile (NaN for stocks without one), index kept. For R3tickp. With lodo (training),
    the profile of the stock's other days for each episode's day, so that it never includes the day trained on."""
    if lodo:
        pr = pd.read_parquet(OUT / "profiles" / f"{name}_lodo.parquet").set_index(["stock", "day"])
        idx = pd.MultiIndex.from_arrays([eps["stock"], eps["day"]])
        return eps.assign(**{c: pr[c].reindex(idx).to_numpy() for c in PROFILE_COLS})
    pr = load_profile(name)
    return eps.assign(**{c: eps["stock"].map(pr[c]).to_numpy() for c in PROFILE_COLS})


def attach_history(eps: pd.DataFrame) -> pd.DataFrame:
    """eps with the stock's history constants (NaN for stocks outside the 2019 universe), index kept."""
    h = history_units().set_index("stock")
    return eps.assign(**{c: eps["stock"].map(h[c]).to_numpy() for c in h.columns})


def load_transitions(names: list[str]) -> tuple[pd.DataFrame, dict]:
    eps, data = [], []
    for n in names:
        eps.append(pd.read_parquet(TRANS / n / "eps.parquet"))
        with np.load(TRANS / n / "data.npz") as z:
            data.append({k: z[k] for k in z.files})
    return pd.concat(eps, ignore_index=True), fqi_adv.concat(data)


# --- fit ------------------------------------------------------------------------------------------

def cmd_fit(a):
    ko = load_splits()["ko_constants"]
    eps, data = load_transitions(a.data)
    if a.size_of:
        n_target = len(pd.read_parquet(TRANS / a.size_of / "eps.parquet"))
    else:
        n_target = len(eps)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(eps), size=min(n_target, len(eps)), replace=False))
        if a.boot_stocks:  # training-data variation: resample the training stocks with replacement
            # the first data set's stocks only, with the same draw as a refit on it alone; the episodes of any
            # further data set (a seen policy's evaluation stocks) are kept whole, then the total is size-matched
            n_first = len(pd.read_parquet(TRANS / a.data[0] / "eps.parquet", columns=["stock"]))
            stocks = eps["stock"].to_numpy()[:n_first]
            idx = pd.Series(np.arange(n_first)).groupby(stocks).apply(np.asarray)
            draw = np.random.default_rng(10_000 + seed).choice(len(idx), size=len(idx), replace=True)
            keep = np.concatenate([idx.iloc[i] for i in draw] + [np.arange(n_first, len(eps))])
            if a.size_of and len(keep) > n_target:
                keep = np.sort(rng.choice(keep, size=n_target, replace=False))
        e = eps.iloc[keep].reset_index(drop=True)
        if any(r in HISTORY_REPS for r in a.reps):
            e = attach_history(e)
        if any(r.endswith("tickp") for r in a.reps):
            e = attach_profile(e, a.profile, lodo=True)
        d = fqi_adv.subset(data, keep)
        uf = unit_frame(e, ko)
        c_r0 = float(np.median(uf["R3"]["t"]))
        for rep in a.reps:
            t0 = time.time()
            # learned alternatives to given units, on raw features: R0pop (per-stock target scale learned from
            # the training data) and R0aug (random rescaling of prices and sizes; neural learner only)
            base = {"R0pop": "R0", "R0aug": "R0"}.get(rep, rep)
            c_lb = c_r0 if base.startswith("R0") else 1.0
            norm = fqi_nn.stock_scale(d, e) if rep == "R0pop" else None
            if a.learner == "nn":
                dd, ee = fqi_nn.augment_scales(d, e, seed) if rep == "R0aug" else (d, e)
                m = fqi_nn.fit(dd, ee, unit_frame(ee, ko) if rep == "R0aug" else uf, base, K, c_lb, seed=seed,
                               device=a.device, norm=norm, epochs=a.epochs, log=print)
            else:
                if rep == "R0aug":
                    raise SystemExit("R0aug needs --learner nn")
                m = fqi_adv.fit(d, e, uf, base, K, c_lb, n_rounds=a.rounds, threads=a.threads, seed=seed,
                                log=lambda s: None, norm=norm, params=json.loads(a.params) if a.params else None)
            path = OUT / "models" / f"{a.tag}_{rep}_s{seed}.pkl"
            m.save(path)
            print(f"{a.tag} {rep} seed {seed}: {len(e):,} episodes, {time.time() - t0:.0f}s -> {path.name}", flush=True)


# --- dhat -----------------------------------------------------------------------------------------

DHAT_SYSTEMS = ("R0", "R2", "R3", "R3s", "R3d")


def dhat_task(args):
    from invex.eval.dhat import probe_policies

    venue, day, stock, n, ko = args
    book, o, c, tick = book_of(venue, day, stock)
    eps = book_episodes(book, venue, day, stock, o, c, n, ko, seed_tag="dhat-v0", tick=tick,
                        **scales_of(venue, day, stock))
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    eps = eps[valid_units(uf)].reset_index(drop=True)
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    out = []
    for system in DHAT_SYSTEMS:
        res = P.evaluate_book(book, eps, uf, probe_policies(system), K, venue_of(venue).latency_ns, tick=tick)
        res = res.merge(pd.DataFrame({"key": eps["key"], "norm": eps["Q"].to_numpy() * uf[system]["p"].to_numpy()}),
                        on="key")
        res["j"] = res["cost"] / res["norm"]
        out.append(res.groupby("policy")["j"].agg(["sum", "count"]).reset_index().assign(system=system))
    df = pd.concat(out, ignore_index=True)
    df["stock"], df["day"] = stock, day
    return df


def cmd_dhat(a):
    ko = load_splits()["ko_constants"]
    jobs = [(a.venue, d, s, a.per_book, ko) for d in a.days for sp in a.splits for s in stocks_for(sp, d, a.venue)]
    t0 = time.time()
    res = pd.concat(pool(dhat_task, jobs, a.workers), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / f"dhat_{a.out}.parquet", index=False)
    print(f"dhat {a.out}: {len(jobs)} books, {time.time() - t0:.0f}s", flush=True)


# --- eval -----------------------------------------------------------------------------------------

_MODELS: dict = {}  # per worker process: each model is loaded once, not once per stock-day


def _model(path):
    if path not in _MODELS:
        _MODELS[path] = fqi_adv.load_model(path)
    return _MODELS[path]


RULES = {"exact": EXACT, "pessimistic": PESSIMISTIC, "optimistic": OPTIMISTIC}


def eval_task(args):
    day, stock, eps, model_paths, ko, rule, latency = (*args, "exact", None)[:7] if len(args) == 5 else args
    venue = str(eps["venue"].iloc[0])
    book, o, c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True).copy()
    for col, v in r1_stats(FlowBook.build(book), eps, tick=tick).items():
        eps[col] = v
    uf = unit_frame(eps, ko)
    pols = [P.Immediate(), P.PostThenCross()]
    if "peg" in os.environ.get("INVEX_EXTRA_BASELINES", "").split(","):  # off unless asked (review round 1)
        pols.append(P.PegThenCross())
    pols += [P.FQIPolicy(_model(p), name=os.path.basename(p)[:-4]) for p in model_paths]
    lat = venue_of(venue).latency_ns if latency is None else latency
    res = P.evaluate_book(book, eps, uf, pols, K, lat, rule=RULES[rule] if isinstance(rule, str) else int(rule),
                          tick=tick)
    # episodes that start in or span a halt, pause, auction guard or feed outage (dropped by the analysis)
    ex_ = pd.Series(excluded(book, eps["t0"].to_numpy(), eps["T_ns"].to_numpy()), index=eps["key"])
    res["excluded"] = res["key"].map(ex_).to_numpy(bool)
    return res


def shuffle_units(eps: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    """Placebo: every episode gets the trailing-window statistics (hence the units) of a random episode of another
    stock on the same day (stocks are paired by a random cyclic shift). Task, path and tick are untouched."""
    from invex.env.units import WINDOW_COLUMNS

    rng = np.random.default_rng(seed)
    cols = [c for c in WINDOW_COLUMNS if c != "tick"]
    eps = eps.copy()
    for _day, g in eps.groupby("day"):
        order = rng.permutation(g["stock"].unique())
        donor = dict(zip(order, np.roll(order, 1)))
        by_stock = {s: gi.index.to_numpy() for s, gi in g.groupby("stock")}
        for s, idx in by_stock.items():
            pick = rng.choice(by_stock[donor[s]], size=len(idx), replace=True)
            eps.loc[idx, cols] = eps.loc[pick, cols].to_numpy()
    return eps


def cmd_eval(a):
    ko = load_splits()["ko_constants"]
    eps = true_eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
    if a.history:  # history constants for R3h / R3tickh models (stocks without a 2019 history get none)
        eps = true_eps = attach_history(eps)
    if a.profile:  # the stocks' probe profiles, for R3tickp models
        eps = true_eps = attach_profile(eps, a.profile)
    if a.placebo == "shuffle":  # policies see the wrong units; costs are still normalised by the true ones
        eps = shuffle_units(eps)
    paths = sorted(str(p) for t in a.tags for p in (OUT / "models").glob(f"{t}_*.pkl"))
    if a.models:  # explicit model names (glob patterns, without .pkl), added to the tags' models
        paths = sorted(set(paths) | {str(p) for pat in a.models for p in (OUT / "models").glob(f"{pat}.pkl")})
    if a.seeds:
        paths = [p for p in paths if any(p.endswith(f"_s{s}.pkl") for s in a.seeds)]
    if a.reps:
        paths = [p for p in paths if os.path.basename(p).split("_")[1] in a.reps]
    lat = None if a.latency_us is None else int(a.latency_us * 1000)
    planted = a.planted or a.hidden or a.markout  # --hidden and --markout: the older names of --planted
    if planted and a.rule != "exact":
        raise SystemExit("--planted needs --rule exact")
    if planted:
        rules = planted_rules(planted)
        rule_of = lambda s: rules.get(s, EXACT)  # noqa: E731
    else:
        rule_of = lambda s: a.rule  # noqa: E731
    jobs = [(d, s, g, paths, ko, rule_of(s), lat) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
    t0 = time.time()
    res = pd.concat(pool(eval_task, jobs, a.workers), ignore_index=True)
    # per-episode per-trade volatility and relative tick (tick / sigma_trade) of the true units, so analyses of
    # the evaluation need not reload the episode table
    uf_true = unit_frame(true_eps, ko)
    keep = ["key", "venue", "day", "stock", "Q", "side", "family", "cell", "w_spread", "w_mid", "tick"]
    keep += [c for c in ("w_trades", "w_seconds", "w_volume", "px_scale", "qty_scale") if c in true_eps]
    cols = true_eps[keep].assign(sigma_trade=uf_true["_sigma_trade"].to_numpy(), eta=uf_true["_eta_sigma"].to_numpy())
    res = res.merge(cols, on="key")
    res["c_spread"] = res["cost"] / (res["Q"] * res["w_spread"])
    res["c_bps"] = 1e4 * res["cost"] / (res["Q"] * res["w_mid"])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / f"eval_{a.out}.parquet", index=False)
    write_json(OUT / f"eval_{a.out}.json", {"episodes": a.episodes, "placebo": a.placebo, "rule": a.rule, "planted": a.planted or a.hidden or a.markout,
                                            "latency_us": a.latency_us,
                                            "models": [os.path.basename(p) for p in paths],
                                            "rows": len(res), "seconds": round(time.time() - t0),
                                            "provenance": provenance(args=vars(a))})
    print(f"{a.out}: {len(res):,} rows, {len(paths)} models, {time.time() - t0:.0f}s", flush=True)
    print(res.groupby("policy")["c_spread"].mean().round(4).to_string(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--name", required=True)
    g.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    g.add_argument("--venue", default="itch", choices=["itch", "coinbase", "iex"])
    g.add_argument("--days", nargs="+", required=True)
    g.add_argument("--per-book", type=int, default=100)
    g.add_argument("--workers", type=int, default=32)
    g.add_argument("--fees", default="none", choices=["none", "nasdaq"],
                   help="nasdaq: every cost net of Nasdaq's taker fee and maker rebate (fee-aware training)")
    g.add_argument("--min-onetick", type=float, default=0.0,
                   help="keep episodes whose book was at one tick for at least this share of the trailing window")
    g.add_argument("--seed-tag", default="", help="episode and behaviour seeds of this set name (default: --name)")
    g.add_argument("--surcharge", default="", help="a profile table with a per-stock crossing surcharge phi (spreads "
                   "per share taken) added to every cost (experiments/stock_profile.py inject)")
    f = sub.add_parser("fit")
    f.add_argument("--data", nargs="+", required=True)
    f.add_argument("--size-of", default="")
    f.add_argument("--tag", required=True)
    f.add_argument("--reps", nargs="+", default=["R0", "R0desc", "R1", "R2", "R3", "R3s", "R3d", "R2tick", "R3tick",
                                                 "Rpi", "Rdim"])
    f.add_argument("--seeds", nargs="+", type=int, default=[0])
    f.add_argument("--rounds", type=int, default=300)
    f.add_argument("--threads", type=int, default=32)
    f.add_argument("--learner", default="gbt", choices=["gbt", "nn"],
                   help="gbt: boosted trees (the pre-registered learner); nn: multilayer perceptron per stage")
    f.add_argument("--device", default="cpu", help="torch device for --learner nn")
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--params", default="", help="LightGBM parameters overriding the defaults, as JSON")
    f.add_argument("--profile", default="base", help="probe-profile table for R3tickp (experiments/stock_profile.py)")
    f.add_argument("--boot-stocks", action="store_true",
                   help="fit on a resample of the training stocks drawn with replacement (seeded by the seed)")
    e = sub.add_parser("eval")
    e.add_argument("--episodes", required=True)
    e.add_argument("--tags", nargs="*", default=[])
    e.add_argument("--models", nargs="*", default=[], help="model name patterns, e.g. 'zero_*_s[3-7]' 'zeronn_*'")
    e.add_argument("--out", required=True)
    e.add_argument("--unseal", action="store_true")
    e.add_argument("--reason", default="")
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--placebo", default="none", choices=["none", "shuffle"],
                   help="shuffle: each episode gets the units of a random other stock that day")
    e.add_argument("--seeds", nargs="+", type=int, default=[], help="only models of these seeds")
    e.add_argument("--reps", nargs="+", default=[], help="only models of these representations")
    e.add_argument("--rule", default="exact", choices=list(RULES), help="fill rule of the replay (robustness)")
    e.add_argument("--hidden", default="", help="a profile table with a planted per-stock hidden queue kappa "
                                                "(stock_profile.py build --kind hidden): replay with it")
    e.add_argument("--markout", default="", help="a profile table with a planted per-stock markout omega "
                                                 "(stock_profile.py build --kind markout): replay with it")
    e.add_argument("--planted", default="", help="a profile table of any world planted in the replay: replay with it")
    e.add_argument("--latency-us", type=float, default=None, help="order latency in microseconds (default: venue's)")
    e.add_argument("--history", action="store_true", help="attach the stocks' 2019 history constants (R3h, R3tickh)")
    e.add_argument("--profile", default="", help="attach this probe-profile table (R3tickp models)")
    h = sub.add_parser("dhat")
    h.add_argument("--splits", nargs="+", default=["train", "val"])
    h.add_argument("--venue", default="itch", choices=["itch", "coinbase", "iex"])
    h.add_argument("--days", nargs="+", required=True)
    h.add_argument("--per-book", type=int, default=50)
    h.add_argument("--out", required=True)
    h.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    {"gen": cmd_gen, "fit": cmd_fit, "eval": cmd_eval, "dhat": cmd_dhat}[a.cmd](a)


if __name__ == "__main__":
    main()

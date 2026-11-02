"""A stock-aware comparator with a powered positive control (after the fourth review): MS+tick plus the stock's probe
profile, what four simple rules saved over crossing at once on the stock's own 2019 training days, in spreads.

The profile carries what the units leave out if stocks differ in fills or adverse selection: in exactly invariant
markets every stock's profile is the same up to noise. Profile columns (invex.env.features.PROFILE_COLS), each the
rule's mean cost minus that of crossing at once, per share in spreads of the trailing window:
  p_post     post at the touch, cross at the end (post_then_cross)
  p_wait     wait without trading, cross at the eighth decision (cross_at_8)
  p_improve  one tick inside, cross at the end (improve_then_cross)
  p_queue    the queue rule theta = 1, iota = 0.4 in MS units
Each is shrunk toward the training stocks' mean by empirical Bayes (sampling variance from the day-to-day spread of
the stock's five days). The probes run on the D-hat episodes (seed tag dhat-v0, 200 drawn per stock-day, a median
of 172 after the units filter), never on evaluation days.

Positive control (`inject`): every training and validation stock gets a hidden crossing surcharge phi, uniform on
[-s/2, s/2] spreads per share taken and fixed across days. No unit reveals it; the probe profile does, because the
probes pay it on the stock's 2019 days. Policies are trained with it in every cost (transfer.py gen --surcharge):
MS+tick (blind to phi), MS+tick plus the measured profile (the comparator), and MS+tick plus phi itself (an oracle
for the achievable gain). On the held-out validation days the comparator should recover a gain of margin size
when the oracle shows one is there. A second kind of planted structure (`build --kind passive`): a per-stock markout
psi on shares filled while resting (adverse selection after a passive fill), which the probes that post pay in
proportion to how much of their order fills passively. Because every order ends fully filled (the rest is crossed at
the end), psi per passive share is a per-stock constant plus a crossing rebate psi: the passive world is the crossing
world with phi = -psi, not a second structure. A genuinely different, state-dependent structure (`--kind hidden`, after
the sixth review): a per-stock hidden queue, kappa hidden shares ahead of every resting order per displayed share
ahead at its arrival, uniform on [0, s] and fixed across days (tracker rule HIDDEN_BASE + 100 kappa). It changes when
resting orders fill, not the price of any action; the policy's queue position shows only the displayed shares. The
probes are rerun with it (`probe --hidden-scale s --draw d --out NAME`), training transitions are generated with it
(`transfer.py gen --surcharge NAME`), and policies are evaluated with it (`transfer.py eval --hidden NAME`). The
hidden queue's effect saturates once the hidden shares block most fills at the order's price, so a second
state-dependent structure (`--kind markout`): every passive fill pays omega times its markout, the move of the mid
against it over one decision interval after the fill, with omega uniform on [-s/2, s/2] per stock (tracker rule
MARKOUT_BASE + MARKOUT_ZERO + 1000 omega). It makes fills that precede adverse moves dearer for some stocks and cheaper
for others, so what a policy should do depends on the state as well as the stock. To first order, though, it is the
crossing family again: the mean markout (about a third of a spread) does almost all the work. So, after the seventh
review, two structures that do not change a stock's average price of resting:
  dmarkout  each passive fill pays omega x (its markout - mbar x the spread at the fill), mbar the stock's own mean
            markout per passive share on its 2019 days (`constants`); omega uniform on [-s/2, s/2]
  habs      kappa x (the stock's median trade size) hidden shares ahead of every resting order at its price, improving
            or not; kappa uniform on [0, s]
    python experiments/stock_profile.py constants                         # profiles/planted_constants.parquet
    python experiments/stock_profile.py probe --dmarkout-scale 8 --draw 1 --out indea
    python experiments/stock_profile.py build --name indea --scale 8 --draw 1 --kind dmarkout

    python experiments/stock_profile.py probe --per-book 200 --workers 32   # profile_probes.parquet
    python experiments/stock_profile.py build --name base                   # profiles/base.parquet
    python experiments/stock_profile.py build --name injob --scale 1 --draw 1   # + profiles/injob_oracle.parquet
    python experiments/stock_profile.py score --eval val_v0_injob val_v0_injobx --profile injob --out injob
    python experiments/stock_profile.py probe --hidden-scale 2 --draw 1 --out inqta   # profile_probes_inqta.parquet
    python experiments/stock_profile.py build --name inqta --scale 2 --draw 1 --kind hidden

Exploratory, post hoc, validation only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer as TR  # noqa: E402

from invex.env.features import PROFILE_COLS  # noqa: E402
from invex.env.units import unit_frame, valid_units  # noqa: E402
from invex.eval import policies as P  # noqa: E402
from invex.eval.dhat import CrossAt, ImproveThenCross  # noqa: E402
from invex.eval.freeze import book_of, scales_of, stocks_for  # noqa: E402
from invex.env.episodes import book_episodes  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402

TRAIN_DAYS = ("2019-01-30", "2019-03-27", "2019-07-30", "2019-08-30", "2019-10-18")
PROBES = {"cross_at_0": "base", "post_then_cross": "p_post", "cross_at_8": "p_wait",
          "improve_then_cross": "p_improve", "q_1_0.4": "p_queue"}
PROFILES = TR.OUT / "profiles"


def probe_policies():
    return [CrossAt(0), P.PostThenCross(), CrossAt(8), ImproveThenCross(), P.QueueRule(1.0, 0.4, "R3", name="q_1_0.4")]


def probe_task(args):
    venue, day, stock, n, ko, rule, marks = (*args, TR.EXACT, False)[:7]
    book, o, c, tick = book_of(venue, day, stock)
    eps = book_episodes(book, venue, day, stock, o, c, n, ko, seed_tag="dhat-v0", tick=tick,
                        **scales_of(venue, day, stock))
    if eps.empty:
        return None
    eps = eps[valid_units(unit_frame(eps, ko))].reset_index(drop=True)
    if eps.empty:
        return None
    uf = unit_frame(eps, ko)
    res = P.evaluate_book(book, eps, uf, probe_policies(), TR.K, venue_of(venue).latency_ns, rule=rule, tick=tick,
                          markouts=marks)
    res = res.merge(pd.DataFrame({"key": eps["key"], "Q": eps["Q"].to_numpy(),
                                  "norm": eps["Q"].to_numpy() * uf["R3"]["p"].to_numpy()}), on="key")
    res["j"] = res["cost"] / res["norm"]
    res["tf"] = (res["filled"] - res["passive"]) / res["Q"]  # share of the order taken by crossing
    out = res[["key", "policy", "j", "tf"] + (["mark", "mspr"] if marks else [])].copy()
    out["stock"], out["day"] = stock, day
    return out


def probe_stocks() -> dict:
    """The probes' books: split -> day -> stocks."""
    return {sp: {d: stocks_for(sp, d, "itch") for d in TRAIN_DAYS} for sp in ("train", "val")}


def planted_stocks() -> np.ndarray:
    """Every stock with a probe book: the stocks over which a planted draw is made (probe and build alike)."""
    return np.unique([s for v in probe_stocks().values() for ss in v.values() for s in ss])


def cmd_probe(a):
    ko = load_splits()["ko_constants"]
    stocks = probe_stocks()
    rules = {}
    every = planted_stocks()
    if a.hidden_scale:  # the planted hidden queue, per stock (the same draw over every stock of both splits)
        rules = {s: TR.hidden_rule(k) for s, k in hidden_ratios(every, a.hidden_scale, a.draw).items()}
    elif a.markout_scale:  # the planted markout, per stock
        rules = {s: TR.markout_rule(m) for s, m in markouts(every, a.markout_scale, a.draw).items()}
    elif a.dmarkout_scale or a.habs_scale:
        pv = planted_values("dmarkout" if a.dmarkout_scale else "habs", a.dmarkout_scale or a.habs_scale, a.draw)
        rules = {s: (TR.dmarkout_rule(r["omega"], r["mbar"]) if a.dmarkout_scale else TR.habs_rule(r["hshares"]))
                 for s, r in pv.iterrows()}
    jobs = [("itch", d, s, a.per_book, ko, rules.get(s, TR.EXACT)) for d in TRAIN_DAYS for sp in ("train", "val")
            for s in stocks[sp][d]]
    res = pd.concat(TR.pool(probe_task, jobs, a.workers), ignore_index=True)
    res.to_parquet(TR.OUT / (f"profile_probes_{a.out}.parquet" if a.out else "profile_probes.parquet"), index=False)
    print(f"probes: {len(jobs)} books, {res['key'].nunique():,} episodes", flush=True)


def cmd_constants(a):
    """Per stock, the constants the demeaned markout and the hidden queue need, from its 2019 training days: mbar, the
    mean markout per passive share of the four posting probes in units of the spread at the fill (a markout rule with
    omega = 0, which books markouts and charges nothing), and tsize, the median trade size of its trailing windows."""
    ko = load_splits()["ko_constants"]
    stocks = probe_stocks()
    rule = TR.MARKOUT_BASE + TR.MARKOUT_ZERO  # omega = 0
    jobs = [("itch", d, s, a.per_book, ko, rule, True) for d in TRAIN_DAYS for sp in ("train", "val")
            for s in stocks[sp][d]]
    res = pd.concat(TR.pool(probe_task, jobs, a.workers), ignore_index=True)
    res = res[res["policy"] != "cross_at_0"]
    g = res.groupby("stock")[["mark", "mspr"]].sum()
    mbar = (g["mark"] / 2.0 / g["mspr"].where(g["mspr"] > 0)).rename("mbar")
    tsize = TR.history_units().set_index("stock")["h_tsize"].rename("tsize")
    out = pd.concat([mbar, tsize], axis=1).reindex(planted_stocks())
    out["mbar"] = out["mbar"].fillna(out["mbar"].median())
    out["tsize"] = out["tsize"].fillna(out["tsize"].median())
    PROFILES.mkdir(parents=True, exist_ok=True)
    out.rename_axis("stock").reset_index().to_parquet(PROFILES / "planted_constants.parquet", index=False)
    print(out.describe().to_string(), flush=True)


def planted_values(kind: str, scale: float, draw: int) -> pd.DataFrame:
    """Per stock (every planted stock), a dmarkout world's omega and mbar, or a habs world's kappa and hshares."""
    c = pd.read_parquet(PROFILES / "planted_constants.parquet").set_index("stock").reindex(planted_stocks())
    if kind == "dmarkout":
        return pd.DataFrame({"omega": markouts(planted_stocks(), scale, draw), "mbar": c["mbar"].round(3)})
    kappa = hidden_ratios(planted_stocks(), scale, draw)
    return pd.DataFrame({"kappa": kappa, "hshares": (kappa * c["tsize"]).round()})


def surcharges(stocks: np.ndarray, scale: float, draw: int) -> pd.Series:
    """phi per stock, uniform on [-scale/2, scale/2], a fixed draw (stocks sorted, so the draw is reproducible)."""
    stocks = np.sort(stocks)
    rng = np.random.default_rng(1_000 + draw)
    return pd.Series(scale * (rng.random(len(stocks)) - 0.5), index=stocks, name="phi")


def hidden_ratios(stocks: np.ndarray, scale: float, draw: int) -> pd.Series:
    """kappa per stock, hidden shares per displayed share ahead, uniform on [0, scale] (rounded to hundredths, as the
    tracker rule carries it), a fixed draw independent of the surcharge draws."""
    stocks = np.sort(stocks)
    rng = np.random.default_rng(2_000 + draw)
    return pd.Series(np.round(scale * rng.random(len(stocks)), 2), index=stocks, name="kappa")


def markouts(stocks: np.ndarray, scale: float, draw: int) -> pd.Series:
    """omega per stock, the multiple of a passive fill's markout it pays, uniform on [-scale/2, scale/2] (rounded to
    thousandths, as the tracker rule carries it), a fixed draw independent of the other planted draws."""
    stocks = np.sort(stocks)
    rng = np.random.default_rng(3_000 + draw)
    return pd.Series(np.round(scale * (rng.random(len(stocks)) - 0.5), 3), index=stocks, name="omega")


TAU2: dict = {}  # the between-stock variance each column's last shrinkage used (profile)


def profile(probes: pd.DataFrame, phi: pd.Series, train_stocks: set,
            psi: pd.Series | None = None, tau2_floor: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrunk profile per stock with the planted costs counted in every probe's cost (phi per share taken, psi per
    share filled while resting: the probes' orders always fill completely, so that share is 1 - tf), and the same
    leaving out each day in turn (per stock and day: what a training stock's profile would be without that day, as
    at evaluation, where the profile never includes the day evaluated). The prior (the training stocks' mean and
    between-stock variance) is computed once from all days."""
    psi = pd.Series(dtype=float) if psi is None else psi
    p = probes.assign(j=probes["j"] + probes["stock"].map(phi).fillna(0.0).to_numpy() * probes["tf"]
                      + probes["stock"].map(psi).fillna(0.0).to_numpy() * (1.0 - probes["tf"]))
    w = p.pivot_table(index=["stock", "day", "key"], columns="policy", values="j")
    day = w.groupby(level=["stock", "day"]).mean()
    n = w.groupby(level=["stock", "day"]).size()
    out, lodo = {}, {}
    for pol, col in PROBES.items():
        if col == "base":
            continue
        d = day[pol] - day["cross_at_0"]
        g = pd.DataFrame({"d": d, "n": n}).groupby(level="stock")
        est = g.apply(lambda x: np.average(x["d"], weights=x["n"]))
        se2 = g["d"].var(ddof=1) / g["d"].count()
        se2 = se2.fillna(se2.median())
        tr = est.index.isin(list(train_stocks))
        m = est[tr].mean()
        tau2 = max(est[tr].var(ddof=1) - se2[tr].mean(), 1e-8, (tau2_floor or {}).get(col, 0.0))
        TAU2[col] = tau2
        out[col] = m + tau2 / (tau2 + se2) * (est - m)
        out[col + "_raw"], out[col + "_se"] = est, np.sqrt(se2)
        # leave one day out: the weighted mean of the other days, shrunk with its own (larger) sampling variance
        x = pd.DataFrame({"d": d, "n": n})
        tot = (x["d"] * x["n"]).groupby(level="stock").transform("sum")
        cnt = x["n"].groupby(level="stock").transform("sum")
        nd = x["d"].groupby(level="stock").transform("count")
        loo = (tot - x["d"] * x["n"]) / (cnt - x["n"])
        # its sampling variance from the other days alone, so that the left-out day enters neither the stock's
        # estimate nor its sampling variance (it enters the prior's mean and variance with a weight of about 1/950)
        s1 = x["d"].groupby(level="stock").transform("sum") - x["d"]
        s2 = (x["d"] ** 2).groupby(level="stock").transform("sum") - x["d"] ** 2
        k = nd - 1
        var_loo = (s2 - s1 ** 2 / k.clip(lower=1)) / (k - 1).clip(lower=1)
        se2_loo = (var_loo / k.clip(lower=1)).where(k >= 2, se2.median())
        lodo[col] = m + tau2 / (tau2 + se2_loo) * (loo - m)
    return pd.DataFrame(out), pd.DataFrame(lodo)


def cmd_build(a):
    hidden = a.kind in ("hidden", "markout", "dmarkout", "habs")  # planted in the replay itself: the probes ran with it
    pname = a.probes or a.name  # a world's probes can be reused under another name (e.g. another shrinkage)
    probes = pd.read_parquet(TR.OUT / (f"profile_probes_{pname}.parquet" if hidden else "profile_probes.parquet"))
    split = pd.Series(load_splits()["split"])
    stocks = probes["stock"].unique()
    zero = pd.Series(0.0, index=np.sort(stocks))
    if hidden:  # the probes already ran with each stock's hidden queue or markout; no cost is added
        # drawn over the same stocks as in `probe` (not only those with probe episodes), so that the values match
        if a.kind in ("dmarkout", "habs"):
            pv = planted_values(a.kind, a.scale, a.draw)
            planted = pv["omega" if a.kind == "dmarkout" else "kappa"]
        else:
            planted = (hidden_ratios if a.kind == "hidden" else markouts)(planted_stocks(), a.scale, a.draw)
        phi, psi = zero, zero
    else:
        planted = surcharges(stocks, a.scale, a.draw) if a.scale else zero
        phi, psi = (planted, zero) if a.kind == "cross" else (zero, planted)
    floor = None
    if a.tau2_floor == "base":  # floor each column's between-stock variance at the real profile's
        base = pd.read_parquet(TR.OUT / "profile_probes.parquet")
        z = pd.Series(0.0, index=np.sort(base["stock"].unique()))
        profile(base, z, set(split.index[split == "train"]), z)
        floor = dict(TAU2)
        print("tau2 floor (real profile):", {k: round(v, 6) for k, v in floor.items()}, flush=True)
    pr, lodo = profile(probes, phi, set(split.index[split == "train"]), psi, tau2_floor=floor)
    print("tau2 used:", {k: round(v, 6) for k, v in TAU2.items()}, flush=True)
    pr["phi"], pr["psi"] = phi.reindex(pr.index).to_numpy(), psi.reindex(pr.index).to_numpy()
    pl = {"cross": "phi", "passive": "psi", "hidden": "kappa", "markout": "omega", "dmarkout": "omega",
          "habs": "kappa"}[a.kind]
    extra = {"dmarkout": "mbar", "habs": "hshares"}.get(a.kind)  # the rule's second constant, carried along
    if hidden:
        pr[pl] = planted.reindex(pr.index).to_numpy()
        if extra:
            pr[extra] = pv[extra].reindex(pr.index).to_numpy()
    PROFILES.mkdir(parents=True, exist_ok=True)
    pr.rename_axis("stock").reset_index().to_parquet(PROFILES / f"{a.name}.parquet", index=False)
    lodo = lodo.reset_index()
    lodo["phi"], lodo["psi"] = lodo["stock"].map(phi).to_numpy(), lodo["stock"].map(psi).to_numpy()
    if hidden:
        lodo[pl] = lodo["stock"].map(planted).to_numpy()
        if extra:
            lodo[extra] = lodo["stock"].map(pv[extra]).to_numpy()
    lodo.to_parquet(PROFILES / f"{a.name}_lodo.parquet", index=False)
    if a.scale:  # the oracle: the true planted value in every profile column
        orc = pd.DataFrame({c: pr[pl] for c in PROFILE_COLS}).assign(phi=pr["phi"], psi=pr["psi"])
        lo = lodo[["stock", "day"]].assign(**{c: lodo[pl] for c in PROFILE_COLS}, phi=lodo["phi"], psi=lodo["psi"])
        if hidden:
            orc[pl], lo[pl] = pr[pl], lodo[pl]
            if extra:
                orc[extra], lo[extra] = pr[extra], lodo[extra]
        orc.rename_axis("stock").reset_index().to_parquet(PROFILES / f"{a.name}_oracle.parquet", index=False)
        lo.to_parquet(PROFILES / f"{a.name}_oracle_lodo.parquet", index=False)
    rows = {c: {"mean": float(pr[c].mean()), "sd": float(pr[c].std()), "corr_phi": float(np.corrcoef(pr[c], pr[pl])[0, 1])
                if a.scale else None, "reliability": float(1 - (pr[c + "_se"] ** 2).mean() / pr[c + "_raw"].var())}
            for c in PROFILE_COLS}
    write_json(PROFILES / f"{a.name}.json", {"scale": a.scale, "draw": a.draw, "kind": a.kind, "stocks": len(pr),
                                             "columns": rows, "provenance": provenance(args=vars(a))})
    for c, r in rows.items():
        print(c, {k: (round(v, 4) if v is not None else None) for k, v in r.items()})


def boot(by_stock: pd.Series, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = by_stock.to_numpy()
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 5)), float(np.percentile(b, 95))


def boot_seeds(X: np.ndarray, Y: np.ndarray, n: int = 4000, seed: int = 0):
    """90% interval of mean(X) - mean(Y) (stocks x seeds), resampling stocks and, within each arm, seeds."""
    rng = np.random.default_rng(seed)
    b = np.empty(n)
    for i in range(n):
        s = rng.integers(0, X.shape[0], X.shape[0])
        sx, sy = rng.integers(0, X.shape[1], X.shape[1]), rng.integers(0, Y.shape[1], Y.shape[1])
        b[i] = X[s][:, sx].mean() - Y[s][:, sy].mean()
    return float(np.percentile(b, 5)), float(np.percentile(b, 95))


def cmd_score(a):
    res = pd.concat([pd.read_parquet(TR.OUT / f"eval_{e}.parquet") for e in a.eval], ignore_index=True)
    res = res[~res["excluded"]] if "excluded" in res else res
    inj = TR.load_profile(a.profile) if a.profile else pd.DataFrame(columns=["phi", "psi"])
    phi, psi = inj["phi"], inj["psi"] if "psi" in inj else pd.Series(dtype=float)
    res["c_inj"] = (res["c_spread"] + res["stock"].map(phi).fillna(0.0).to_numpy() * (res["filled"] - res["passive"]) / res["Q"]
                    + res["stock"].map(psi).fillna(0.0).to_numpy() * res["passive"] / res["Q"])
    res["arm"] = res["policy"].str.replace(r"_s\d+$", "", regex=True)
    per = res.groupby(["arm", "stock", "key"])["c_inj"].mean().unstack(0).dropna()
    arms = list(per.columns)
    out = {"eval": a.eval, "profile": a.profile, "episodes": int(len(per)), "stocks": int(per.index.get_level_values(0).nunique()),
           "mean": {k: float(per[k].mean()) for k in arms}, "contrasts": {}}
    # per stock and seed, on the episodes every policy has: for intervals that resample seeds as well as stocks
    pol = res[res["key"].isin(per.index.get_level_values("key"))].groupby(["policy", "stock"])["c_inj"].mean().unstack(0)
    for x, y in [tuple(c.split(":")) for c in a.contrasts]:
        if x in per and y in per:
            m, lo, hi = boot((per[x] - per[y]).groupby(level="stock").mean())
            slo, shi = boot_seeds(pol[[c for c in pol if c.rsplit("_s", 1)[0] == x]].to_numpy(),
                                  pol[[c for c in pol if c.rsplit("_s", 1)[0] == y]].to_numpy())
            out["contrasts"][f"{x}-{y}"] = {"mean": m, "lo": lo, "hi": hi, "lo_seed": slo, "hi_seed": shi}
            print(f"{x:24s} - {y:24s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}] stocks+seeds [{slo:+.4f}, {shi:+.4f}]")
    write_json(TR.OUT / f"profile_score_{a.out}.json", {**out, "provenance": provenance(args=vars(a))})
    pol.to_parquet(TR.OUT / f"profile_score_{a.out}_pol.parquet")  # per stock and policy: intervals for shares


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--per-book", type=int, default=200)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--hidden-scale", type=float, default=0.0, help="plant a hidden queue, kappa uniform on [0, s]")
    p.add_argument("--markout-scale", type=float, default=0.0, help="plant a markout, omega uniform on [-s/2, s/2]")
    p.add_argument("--dmarkout-scale", type=float, default=0.0, help="plant a demeaned markout, omega on [-s/2, s/2]")
    p.add_argument("--habs-scale", type=float, default=0.0, help="plant a hidden queue of kappa trade sizes, [0, s]")
    c = sub.add_parser("constants")
    c.add_argument("--per-book", type=int, default=200)
    c.add_argument("--workers", type=int, default=32)
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--out", default="", help="write profile_probes_<out>.parquet")
    b = sub.add_parser("build")
    b.add_argument("--name", required=True)
    b.add_argument("--scale", type=float, default=0.0)
    b.add_argument("--draw", type=int, default=0)
    b.add_argument("--probes", default="", help="read profile_probes_<probes>.parquet (default: the --name's)")
    b.add_argument("--tau2-floor", choices=("", "base"), default="",
                   help="base: floor each column's between-stock variance at the real profile's")
    b.add_argument("--kind", choices=("cross", "passive", "hidden", "markout", "dmarkout", "habs"), default="cross",
                   help="plant a crossing surcharge phi, a charge psi on passive fills, a hidden queue kappa or a "
                        "markout omega (the last two read profile_probes_<name>.parquet)")
    s = sub.add_parser("score")
    s.add_argument("--eval", nargs="+", required=True)
    s.add_argument("--profile", default="")
    s.add_argument("--contrasts", nargs="+", required=True, help="arm pairs a:b (arms are model names without _s<seed>)")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"probe": cmd_probe, "build": cmd_build, "score": cmd_score, "constants": cmd_constants}[a.cmd](a)


if __name__ == "__main__":
    main()

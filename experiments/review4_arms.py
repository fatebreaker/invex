"""The round-4 exploratory arms against their baselines (review round 4; post hoc, from stored evaluations).

    python experiments/review4_arms.py [inv ctx window contemp]

Paired cost differences in spreads per share, seeds averaged within an episode, every stock weighted equally, 95%
intervals from the stock bootstrap stratified by relative-tick tercile (10,000 replicates):
  * inv: invariance penalties (experiments/inv_fqi.py; seed 0) against the plain neural learner on the same inputs and
    against neural MS+tick (seeds 0-4), validation;
  * ctx: the learned context encoder (experiments/ctx_fqi.py; seeds 0-2) against neural RAW+INFO, raw features with the
    cost in spreads, and MS+tick, validation;
  * window: units from 15- and 60-minute windows (experiments/window_units.py), the registered policies evaluated with
    them and policies retrained with them, against the registered 30-minute evaluation, validation;
  * tdr: domain randomisation in time, price and size (experiments/time_dr.py; seeds 0-2) against the plain neural
    learner, its price-and-size augmentation (R0aug) and neural MS+tick, validation;
  * sky: a same-day skyline (trees trained on the validation stocks' own validation days, other start times) against
    the zero-shot policy, and the share of the gap to the clairvoyant hindsight bound each closes, validation;
  * iexood: the learned alternatives (penalties, encoder, time randomisation; three seeds each) against neural MS+tick
    and the plain learner on IEX's validation days, wide books, also without shortfall episodes (round 5);
  * contemp: policies trained on the training stocks' own days of each sealed stage (scripts/review4_contemp_queue.sh)
    against the registered zero-shot policies on that stage: the cost of the years, per representation.
Writes results/transfer/review4_arms.json and paper/numbers_review4_arms.tex. Reads no sealed table.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from invex.eval.stats import cluster_bootstrap
from invex.eval.universe import load_splits
from invex.paths import REPO, RESULTS

OUT = RESULTS / "transfer"
CUTS = None


def load(evals: list[str], arms: dict[str, list[int]]) -> pd.DataFrame:
    """Per-episode cost of each arm ('<tag>_<rep>'), averaged over its seeds; one column per arm."""
    pols = [f"{arm}_s{s}" for arm, seeds in arms.items() for s in seeds]
    parts = []
    for ev in evals:
        f = OUT / f"eval_{ev}.parquet"
        if not f.exists():
            continue
        t = ds.dataset(f).to_table(columns=["key", "policy", "stock", "c_spread", "excluded"],
                                   filter=ds.field("policy").isin(pols)).to_pandas()
        parts.append(t[~t["excluded"]])
    res = pd.concat(parts, ignore_index=True).drop_duplicates(["key", "policy"])
    res["arm"] = res["policy"].str.replace(r"_s\d+$", "", regex=True)
    missing = sorted(set(arms) - set(res["arm"]))
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    return res.pivot_table(index=["key", "stock"], columns="arm", values="c_spread", aggfunc="mean")


def eta_of(evals: list[str]) -> pd.Series:
    for ev in evals:
        f = OUT / f"eval_{ev}.parquet"
        if f.exists() and "eta" in ds.dataset(f).schema.names:
            return ds.dataset(f).to_table(columns=["stock", "eta"]).to_pandas().groupby("stock")["eta"].median()
    raise SystemExit("no evaluation stores the relative tick")


def contrast(per: pd.DataFrame, a: str, b: str, eta: pd.Series) -> dict:
    d = (per[a] - per[b]).dropna().groupby(level="stock").mean()
    st = np.digitize(np.log(eta.reindex(d.index).to_numpy()), CUTS)
    r = cluster_bootstrap(d.to_numpy(), st, n_boot=10_000, level=0.95)
    return {"mean": float(r["mean"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "stocks": int(len(d))}


def put(m: dict, out: dict, name: str, r: dict) -> None:
    out[name] = r
    m[name], m[name + "Lo"], m[name + "Hi"] = f"{r['mean']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
    print(f"  {name:40s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] ({r['stocks']} stocks)", flush=True)


VAL_EVALS = ["val_v0", "val_v0_rev", "val_v0_rev5", "val_v0_profnn", "val_v0_info", "val_v0_r0r"]
NN = list(range(5))
SEEDS3 = {"vrexnn1": [0, 1, 2], "gdronn10": [0, 1, 2], "irmnn1": [0, 1, 2]}  # the better strengths, three seeds (round 5)


def part_inv(m, out):
    methods = ["vrexnn1", "vrexnn10", "gdronn1", "gdronn10", "irmnn1", "irmnn10"]
    arms = {f"{x}_{r}": SEEDS3.get(x, [0]) for x in methods for r in ("R0r", "R0info")}
    arms |= {"zeronn_R0r": NN, "zeronn_R0info": NN, "zeronn_R3tick": NN}
    per = load(VAL_EVALS + ["val_v0_inv", "val_v0_inv2"], arms)
    eta = eta_of(["val_v0_info"])
    for x in methods:
        for r, R in (("R0r", "Rawr"), ("R0info", "Info")):
            tag = {"vrexnn1": "VrexOne", "vrexnn10": "VrexTen", "gdronn1": "GdroOne", "gdronn10": "GdroTen",
                   "irmnn1": "IrmOne", "irmnn10": "IrmTen"}[x]  # macro names cannot hold digits
            put(m, out, f"Inv{tag}{R}VsPlain", contrast(per, f"{x}_{r}", f"zeronn_{r}", eta))
            put(m, out, f"Inv{tag}{R}VsMstick", contrast(per, f"{x}_{r}", "zeronn_R3tick", eta))


def part_ctx(m, out):
    arms = {"ctxnn_R0r": [0, 1, 2], "zeronn_R0r": NN, "zeronn_R0info": NN, "zeronn_R3tick": NN}
    per = load(VAL_EVALS + ["val_v0_ctx"], arms)
    eta = eta_of(["val_v0_info"])
    for b, B in (("zeronn_R0info", "Info"), ("zeronn_R0r", "Rawr"), ("zeronn_R3tick", "Mstick")):
        put(m, out, f"CtxVs{B}", contrast(per, "ctxnn_R0r", b, eta))


def part_tdr(m, out):
    arms = {"tdrnn_R0r": [0, 1, 2], "tdrnn_R0": [0, 1, 2], "zeronn_R0r": NN, "zeronn_R0": NN, "zeronn_R0aug": NN,
            "zeronn_R3tick": NN}
    per = load(VAL_EVALS + ["val_v0_tdr"], arms)
    eta = eta_of(["val_v0_info"])
    put(m, out, "TdrRawrVsPlain", contrast(per, "tdrnn_R0r", "zeronn_R0r", eta))
    put(m, out, "TdrRawrVsMstick", contrast(per, "tdrnn_R0r", "zeronn_R3tick", eta))
    put(m, out, "TdrRawVsPlain", contrast(per, "tdrnn_R0", "zeronn_R0", eta))
    put(m, out, "TdrRawVsAug", contrast(per, "tdrnn_R0", "zeronn_R0aug", eta))
    put(m, out, "TdrRawVsMstick", contrast(per, "tdrnn_R0", "zeronn_R3tick", eta))


def part_sky(m, out):
    """The same-day skyline against the zero-shot policy, and the share of the gap from crossing at once to the
    clairvoyant hindsight bound that each closes (as experiments/analyze_round4.py, stock means, 95% bootstrap)."""
    arms = {"sky_R3tick": [0, 1, 2], "sky_R0": [0, 1, 2], "zero_R3tick": [0, 1, 2], "zero_R0": [0, 1, 2],
            "zeroppo_R3tick": [0, 1, 2]}
    per = load(VAL_EVALS + ["val_v0_ppo", "val_v0_sky"], arms)
    eta = eta_of(["val_v0_info"])
    put(m, out, "SkyMstickVsZero", contrast(per, "sky_R3tick", "zero_R3tick", eta))
    put(m, out, "SkyRawVsZero", contrast(per, "sky_R0", "zero_R0", eta))
    put(m, out, "SkyRawVsMstick", contrast(per, "sky_R0", "sky_R3tick", eta))
    base = pd.read_parquet(OUT / "eval_val_v0.parquet", columns=["key", "policy", "stock", "c_spread", "Q", "w_spread",
                                                                 "excluded"])
    base = base[~base["excluded"]]
    imm = base[base["policy"] == "immediate"].set_index("key")
    h = pd.read_parquet(OUT / "hindsight_val_v0.parquet")
    h = h[~h["overflow"]].set_index("key")
    keys = imm.index.intersection(h.index)
    st = imm.loc[keys, "stock"]
    lv = {"hindsight": (h.loc[keys, "hs_gross"] / (imm.loc[keys, "Q"] * imm.loc[keys, "w_spread"])).groupby(st).mean(),
          "immediate": imm.loc[keys, "c_spread"].groupby(st).mean()}
    flat = per.reset_index().set_index("key")
    for c in arms:
        lv[c] = flat.loc[flat.index.intersection(keys), c].groupby(flat.loc[flat.index.intersection(keys), "stock"]).mean()
    L = pd.DataFrame(lv).dropna()
    gap = L["immediate"] - L["hindsight"]
    rng = np.random.default_rng(7)
    for c, name in (("zero_R3tick", "Zero"), ("sky_R3tick", "Sky"), ("zeroppo_R3tick", "Ppo")):
        gain = L["immediate"] - L[c]
        boots = [gain.iloc[i].mean() / gap.iloc[i].mean() for i in (rng.integers(0, len(L), len(L)) for _ in range(2000))]
        r = {"mean": float(gain.mean() / gap.mean()), "lo": float(np.quantile(boots, 0.025)),
             "hi": float(np.quantile(boots, 0.975)), "stocks": int(len(L))}
        out[f"SkyShare{name}"] = r
        m[f"SkyShare{name}"] = f"{100 * r['mean']:.0f}\\%"
        m[f"SkyShare{name}Lo"], m[f"SkyShare{name}Hi"] = f"{100 * r['lo']:.0f}\\%", f"{100 * r['hi']:.0f}\\%"
        print(f"  share of the hindsight gap closed by {c}: {r['mean']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]")


def part_iexood(m, out):
    """Review round 5 (R5): the learned alternatives where the units matter, IEX's validation days (iex_val_v0,
    validation-split stocks), on its wide books (the trailing window at one tick less than half the time) and,
    separately, without the episodes in which some evaluated policy's cross ran beyond the visible book."""
    arms = {"zeronn_R3tick": NN, "zeronn_R0r": NN, "zeronn_R0info": NN, "zeronn_R0": NN, "zeronn_R0aug": NN,
            "tdrnn_R0r": [0, 1, 2], "tdrnn_R0": [0, 1, 2], "ctxnn_R0r": [0, 1, 2]}
    arms |= {f"{x}_{r}": [0, 1, 2] for x in SEEDS3 for r in ("R0r", "R0info")}
    evals = ["iex_val_v0_nn", "iex_val_v0_tdr", "iex_val_v0_ctx", "iex_val_v0_inv"]
    pols = [f"{a}_s{s}" for a, ss in arms.items() for s in ss]
    parts = []
    for ev in evals:
        tb = ds.dataset(OUT / f"eval_{ev}.parquet").to_table(
            columns=["key", "policy", "stock", "c_spread", "shortfall", "excluded"],
            filter=ds.field("policy").isin(pols)).to_pandas()
        parts.append(tb[~tb["excluded"]])
    res = pd.concat(parts, ignore_index=True).drop_duplicates(["key", "policy"])
    res["arm"] = res["policy"].str.replace(r"_s\d+$", "", regex=True)
    missing = sorted(set(arms) - set(res["arm"]))
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    eps = pd.read_parquet(REPO / "data" / "episodes" / "iex_val_v0.parquet", columns=["key", "w_onetick"])
    wide = set(eps.loc[eps["w_onetick"] < 0.5, "key"])
    short = set(res.loc[res["shortfall"] > 0, "key"])
    per_all = res.pivot_table(index=["key", "stock"], columns="arm", values="c_spread", aggfunc="mean")
    keys = per_all.index.get_level_values("key")
    eta = eta_of(["iex_val_v0_nn"])
    out["IexOodEpisodes"] = {"wide": int(keys.isin(wide).sum()), "shortfall_wide": int((keys.isin(wide) & keys.isin(short)).sum())}
    m["IexOodWideEpisodes"] = f"{int(keys.isin(wide).sum()):,}".replace(",", "{,}")
    m["IexOodWideStocks"] = str(per_all[keys.isin(wide)].index.get_level_values("stock").nunique())
    m["IexOodShortfallEpisodes"] = f"{out['IexOodEpisodes']['shortfall_wide']:,}".replace(",", "{,}")
    tags = {"irmnn1": "Irm", "vrexnn1": "Vrex", "gdronn10": "Gdro"}
    for sub, S in (("wide", ""), ("clean", "Clean")):
        sel = keys.isin(wide) & (~keys.isin(short) if sub == "clean" else True)
        per = per_all[sel]
        P = f"IexOod{S}"
        put(m, out, f"{P}RawrVsMstick", contrast(per, "zeronn_R0r", "zeronn_R3tick", eta))
        put(m, out, f"{P}InfoVsMstick", contrast(per, "zeronn_R0info", "zeronn_R3tick", eta))
        put(m, out, f"{P}CtxVsMstick", contrast(per, "ctxnn_R0r", "zeronn_R3tick", eta))
        put(m, out, f"{P}CtxVsRawr", contrast(per, "ctxnn_R0r", "zeronn_R0r", eta))
        put(m, out, f"{P}CtxVsInfo", contrast(per, "ctxnn_R0r", "zeronn_R0info", eta))
        put(m, out, f"{P}TdrRawrVsMstick", contrast(per, "tdrnn_R0r", "zeronn_R3tick", eta))
        put(m, out, f"{P}TdrRawrVsPlain", contrast(per, "tdrnn_R0r", "zeronn_R0r", eta))
        put(m, out, f"{P}TdrRawVsPlain", contrast(per, "tdrnn_R0", "zeronn_R0", eta))
        put(m, out, f"{P}TdrRawVsAug", contrast(per, "tdrnn_R0", "zeronn_R0aug", eta))
        for x, X in tags.items():
            for r, R in (("R0r", "Rawr"), ("R0info", "Info")):
                put(m, out, f"{P}{X}{R}VsPlain", contrast(per, f"{x}_{r}", f"zeronn_{r}", eta))
                put(m, out, f"{P}{X}{R}VsMstick", contrast(per, f"{x}_{r}", "zeronn_R3tick", eta))


def part_window(m, out):
    eta = eta_of(["val_v0_info"])
    for W, tagW in ((900, "Fifteen"), (3600, "Sixty")):
        ev = f"val_v0_w{W}"
        if not (OUT / f"eval_{ev}.parquet").exists():
            print(f"  (no {ev})")
            continue
        # the same models under the other window (their own evaluation) against the registered 30-minute one
        for rep, R in (("R3tick", "Mstick"), ("R0info", "Info")):
            other = load([ev], {f"zero_{rep}": [0, 1, 2], f"w{W}_{rep}": [0, 1, 2]})
            base = load(["val_v0", "val_v0_info"], {f"zero_{rep}": [0, 1, 2], "zero_R0": [0, 1, 2]})
            j = other.join(base, rsuffix="_reg", how="inner")
            put(m, out, f"Win{tagW}{R}ZeroVsReg", contrast(j, f"zero_{rep}", f"zero_{rep}_reg", eta))
            put(m, out, f"Win{tagW}{R}RetrainVsReg", contrast(j, f"w{W}_{rep}", f"zero_{rep}_reg", eta))
            if rep == "R3tick":
                put(m, out, f"Win{tagW}RawVsRetrainMstick", contrast(j, "zero_R0", f"w{W}_{rep}", eta))


def part_contemp(m, out):
    for stage, tag, info in (("stage1", "StageOne", "stage1_info"), ("stage2", "StageTwo", "stage2_info")):
        ev = f"{stage}_ctmp"
        if not (OUT / f"eval_{ev}.parquet").exists():
            print(f"  (no {ev})")
            continue
        c = "ctmp1" if stage == "stage1" else "ctmp2"
        arms = {f"{c}_{r}": [0, 1, 2] for r in ("R3tick", "R0", "R0info")}
        arms |= {f"zero_{r}": [0, 1, 2] for r in ("R3tick", "R0", "R0info")}
        per = load([stage, info, ev], arms)
        eta = eta_of([ev])
        for r, R in (("R3tick", "Mstick"), ("R0", "Raw"), ("R0info", "Info")):
            put(m, out, f"Ctmp{tag}{R}Years", contrast(per, f"zero_{r}", f"{c}_{r}", eta))
        put(m, out, f"Ctmp{tag}RawVsMstick", contrast(per, f"{c}_R0", f"{c}_R3tick", eta))
        put(m, out, f"Ctmp{tag}InfoVsMstick", contrast(per, f"{c}_R0info", f"{c}_R3tick", eta))
        per = per.assign(dd=(per["zero_R0"] - per[f"{c}_R0"]) - (per["zero_R3tick"] - per[f"{c}_R3tick"]),
                         z=0.0)
        put(m, out, f"Ctmp{tag}YearsRawMinusMstick", contrast(per, "dd", "z", eta))


def main() -> None:
    global CUTS
    CUTS = np.log(load_splits()["eta_sigma_tercile_cuts"])
    parts = sys.argv[1:] or ["inv", "ctx", "window", "contemp", "tdr", "sky"]
    f = OUT / "review4_arms.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    mf = REPO / "paper" / "numbers_review4_arms.tex"
    m = {}
    if mf.exists():
        import re
        m = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", mf.read_text()))
    for p in parts:
        print(p, flush=True)
        {"inv": part_inv, "ctx": part_ctx, "window": part_window, "contemp": part_contemp, "tdr": part_tdr, "sky": part_sky,
         "iexood": part_iexood}[p](m, out)
    f.write_text(json.dumps(out, indent=1))
    lines = ["% generated by experiments/review4_arms.py (post hoc, exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    mf.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

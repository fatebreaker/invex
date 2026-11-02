"""Fourth review round (all exploratory, post hoc): fee-aware training, the best cost attainable in hindsight, a
model-based (fill-probability) baseline and a stock-aware comparator, on validation and the two sealed stages.

    python experiments/analyze_round4.py --eval stage1 [--n-boot 4000]

Reads results/transfer/eval_<eval>_fee.parquet (fee-aware and gross-trained policies, with passive fills),
hindsight_<episodes>.parquet and eval_<eval>_{fill,hist,aware,offsets,idr3}.parquet, whichever exist. Costs are per
share, in the stock's mean spread, seeds averaged, stocks weighted equally; net costs add Nasdaq's taker fee on
shares taken and subtract its rebate on shares filled resting. Intervals: stock-cluster bootstrap with the
stock-and-seed interval beside it, 95% (90% for the transfer gaps, as H1). Writes results/transfer/round4_<eval>.json
and paper/numbers_round4_<eval>.tex (prefix RFourVal, RFourStageOne, RFourStageTwo; RFourShift, RFourInject).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_revision as AR  # noqa: E402

from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.paths import DATA, REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

OUT = RESULTS / "transfer"
TAKER, REBATE = AR.TAKER, AR.REBATE
EPISODES = {"val_v0": "val_v0", "stage1": "stage1_v0", "stage2": "stage2_v0"}
SEEN_DATA = {"val_v0": "valseen_v0", "stage1": "testseen_v0", "stage2": "testseen_v0"}
PREFIX = {"val_v0": "RFourVal", "stage1": "RFourStageOne", "stage2": "RFourStageTwo"}
TAGS = {"zero": "gross", "feezero": "fee", "seen": "grossseen", "seent": "grossseen", "feeseen": "feeseen",
        "feeseent": "feeseen", "fill": "fill", "fillfee": "fillfee", "fillr0": "fillraw", "fillfeer0": "fillfeeraw",
        "seenidr": "idr", "hist": "hist", "fillx": "fillx", "fillxr0": "fillxraw"}
# evaluations read, and which of their policies (the pooled ones repeat those of the fee evaluation)
SOURCES = {"fee": lambda p: p == p, "fill": lambda p: p == p, "aware": lambda p: p.str.startswith("awareshift"),
           "offsets": lambda p: p.str.startswith("aware_"),
           "idr3": lambda p: p.str.startswith("seenidr") | (p == "seen_R3_s0"),
           "hist": lambda p: p.str.startswith("hist_"),
           "fillx": lambda p: p.str.startswith("fillx")}  # the fill rule reading the cost of crossing from the book
NAME = re.compile(r"^(?P<tag>[a-z][a-z0-9]*)_(?:(?P<rep>[A-Za-z0-9]+)_)?s(?P<seed>\d+)$")

FEE_CONTRASTS = [  # (name, a, b, seen stocks only); costs net of fees
    ("fee ms+tick vs fee raw", "fee:R3tick", "fee:R0", False),
    ("fee ms+tick vs fee zscore", "fee:R3tick", "fee:R1", False),
    ("fee ms+tick vs fee ms", "fee:R3tick", "fee:R3", False),
    ("fee raw vs fee ms", "fee:R0", "fee:R3", False),
    ("gross ms+tick vs gross raw", "gross:R3tick", "gross:R0", False),
    ("fee ms+tick vs gross ms+tick", "fee:R3tick", "gross:R3tick", False),
    ("fee raw vs gross raw", "fee:R0", "gross:R0", False),
    ("fee ms+tick vs post", "fee:R3tick", "base:post_then_cross", False),
    ("fee gap ms+tick", "fee:R3tick", "feeseen:R3tick", True),
    ("fillfee vs fee ms+tick", "fillfee:model", "fee:R3tick", False),
]
GROSS_CONTRASTS = [  # gross costs
    ("gross ms+tick vs gross raw", "gross:R3tick", "gross:R0", False),
    ("fee ms+tick vs gross ms+tick", "fee:R3tick", "gross:R3tick", False),
    ("fee ms+tick vs fee raw", "fee:R3tick", "fee:R0", False),
    ("fill vs ms+tick", "fill:model", "gross:R3tick", False),
    ("fill vs raw", "fill:model", "gross:R0", False),
    ("fill vs post", "fill:model", "base:post_then_cross", False),
    ("fill raw vs fill ms+tick", "fillraw:model", "fill:model", False),
    ("fillx vs fill", "fillx:model", "fill:model", False),
    ("fillx raw vs fill raw", "fillxraw:model", "fillraw:model", False),
    ("offsets ms+tick vs ms+tick", "offsets:R3tick", "gross:R3tick", True),
    ("offsets ms vs ms", "offsets:R3", "gross:R3", True),
    ("offsets raw vs raw", "offsets:R0", "gross:R0", True),
    ("idr ms vs seen ms", "idr:R3id", "grossseen:R3", True),
    ("idr ms+tick vs seen ms+tick", "idr:R3tickid", "grossseen:R3tick", True),
    ("hist ms vs raw", "hist:R3h", "gross:R0", True),
    ("hist ms vs ms", "hist:R3h", "gross:R3", True),
    ("ms+tick vs hist ms+tick", "gross:R3tick", "hist:R3tickh", True),
    ("aware ms+tick vs ms+tick", "aware:R3tick", "gross:R3tick", True),
    ("aware ms vs ms", "aware:R3", "gross:R3", True),
    ("aware raw vs raw", "aware:R0", "gross:R0", True),
    ("ms+tick vs aware ms+tick", "gross:R3tick", "aware:R3tick", True),
]
BPS_CONTRASTS = [("fee ms+tick vs fee raw", "fee:R3tick", "fee:R0"),
                 ("gross ms+tick vs gross raw", "gross:R3tick", "gross:R0"),
                 ("fee ms+tick vs gross ms+tick", "fee:R3tick", "gross:R3tick"),
                 ("fee ms+tick vs post", "fee:R3tick", "base:post_then_cross")]
LEVELS = ["base:immediate", "base:post_then_cross", "gross:R0", "gross:R3tick", "fill:model", "fee:R3tick",
          "fee:R0", "fillfee:model"]


def columns(policy: pd.Series) -> pd.DataFrame:
    u = pd.Series(policy.unique())
    m = u.str.extract(NAME)
    aware = u.str.startswith("aware")
    rep = m["rep"].fillna("model")
    rep = rep.where(~aware, u.str.extract(r"_([A-Za-z0-9]+)_s\d+$")[0])
    tag = m["tag"].map(TAGS)
    # awareshift_<model>: per-stock patience; aware_<model>: per-stock offsets
    tag = tag.where(~aware, np.where(u.str.startswith("awareshift"), "aware", "offsets"))
    col = (tag + ":" + rep).where(tag.notna(), "base:" + u)
    seed = pd.to_numeric(u.str.extract(r"_s(\d+)$")[0]).fillna(-1).astype(int)
    return pd.DataFrame({"col": policy.map(dict(zip(u, col))), "seed": policy.map(dict(zip(u, seed)))})


def load(ev: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for suffix, keep in SOURCES.items():
        f = OUT / f"eval_{ev}_{suffix}.parquet"
        if f.exists():
            r = pd.read_parquet(f)
            parts.append(r[keep(r["policy"])])
    res = pd.concat(parts, ignore_index=True)
    res = res[~res["excluded"]].reset_index(drop=True)
    res = pd.concat([res, columns(res["policy"])], axis=1)
    info = res.groupby("key")[["stock", "day", "Q", "w_mid", "w_spread"] + (["eta"] if "eta" in res else [])].first()
    res["net_cost"] = res["cost"] + TAKER * (res["Q"] - res["passive"]) - REBATE * res["passive"]
    res["net_spread"] = res["net_cost"] / (res["Q"] * res["w_spread"])
    res["net_bps"] = 1e4 * res["net_cost"] / (res["Q"] * res["w_mid"])
    return res, info


def contrasts(res, info, specs, unit, seen, n_boot) -> dict:
    r = res.assign(unit_cost=res[unit])
    cols = set(r["col"].unique())
    out = {}
    for name, a, b, seen_only in specs:
        if a not in cols or b not in cols:
            continue
        # 95% intervals, as the arms of experiments/analyze_revision.py; 90% for the transfer gaps, as H1
        level = 0.90 if seen_only and name.endswith(("gap ms+tick", "vs aware ms+tick", "vs hist ms+tick")) else 0.95
        rr = r
        if a.split(":")[0] in ("offsets", "aware") or b.split(":")[0] in ("offsets", "aware"):
            # a corrected policy against its own base: the same seeds
            sa = set(r.loc[r["col"].isin([a, b]) & r["col"].str.match("offsets|aware"), "seed"])
            rr = r[~r["col"].isin([a, b]) | r["seed"].isin(sa)]
        c = AR.contrast(rr, info, a, b, seen if seen_only else None, n_boot, level=level)
        if c is not None:
            c["level"] = level
            out[name] = c
    return out


def bps(res, info, n_boot) -> dict:
    """Basis-point contrasts, gross and net, with the intervals of the spread contrasts (stocks, and stocks with
    training seeds, resampled; 95%)."""
    out = {}
    cols = set(res["col"].unique())
    for unit in ("c_bps", "net_bps"):
        r = res.assign(unit_cost=res[unit])
        for name, a, b in BPS_CONTRASTS:
            if a in cols and b in cols:
                c = AR.contrast(r, info, a, b, None, n_boot, level=0.95)
                if c is not None:
                    out[f"{name} {unit}"] = c
    return out


def hindsight(res, info, ev, n_boot) -> dict | None:
    """Mean cost of each policy and of the hindsight bound (stocks weighted equally), gross and net; the share of
    the gap from crossing immediately to the bound that each policy closes, with a stock bootstrap 95% interval."""
    f = OUT / f"hindsight_{EPISODES[ev]}.parquet"
    if not f.exists():
        return None
    h = pd.read_parquet(f)
    h = h[~h["overflow"]].set_index("key")
    keys = info.index.intersection(h.index)
    norm = info.loc[keys, "Q"] * info.loc[keys, "w_spread"]
    st = info.loc[keys, "stock"]
    meta = json.loads(f.with_suffix(".json").read_text())
    out = {"episodes": int(len(keys)), "dropped_overflow": int((pd.read_parquet(f)["overflow"]).sum()),
           "states_median": meta["states_median"], "states_max": meta["states_max"], "max_states": meta["max_states"]}
    for kind, unit, hcol in (("gross", "c_spread", "hs_gross"), ("net", "net_spread", "hs_net")):
        lv = {"hindsight": (h.loc[keys, hcol] / norm).groupby(st).mean()}
        for c in LEVELS:
            e = res[res["col"] == c].groupby("key")[unit].mean().reindex(keys)
            if e.notna().any():
                lv[c] = e.groupby(st).mean()
        L = pd.DataFrame(lv).dropna()
        o = {"levels": {k: float(v) for k, v in L.mean().items()}, "stocks": int(len(L))}
        rng = np.random.default_rng(7)
        gap = L["base:immediate"] - L["hindsight"]
        o["headroom"] = float(gap.mean())
        shares = {}
        for c in L.columns:
            if c in ("hindsight", "base:immediate"):
                continue
            gain = L["base:immediate"] - L[c]
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(L), len(L))
                boots.append(gain.iloc[i].mean() / gap.iloc[i].mean())
            shares[c] = {"mean": float(gain.mean() / gap.mean()), "lo": float(np.quantile(boots, 0.025)),
                         "hi": float(np.quantile(boots, 0.975))}
        o["share_closed"] = shares
        o["margin_share"] = float(AR.DELTA / gap.mean())
        # the margin against what learning actually gains (MS+tick relative to crossing now and to post-and-wait)
        ms = "gross:R3tick" if kind == "gross" else "fee:R3tick"
        if ms in L:
            o["margin_vs_gain_immediate"] = float(AR.DELTA / (L["base:immediate"] - L[ms]).mean())
            o["margin_vs_gain_post"] = float(AR.DELTA / (L["base:post_then_cross"] - L[ms]).mean())
        out[kind] = o
    return out


def passive(res) -> dict:
    p = res.assign(ps=res["passive"] / res["Q"]).groupby(["col", "stock"])["ps"].mean().groupby("col").mean()
    return {c: float(v) for c, v in p.items()}


def macros_of(out: dict, P: str) -> dict:
    m = {}
    for group in ("net", "gross"):
        for name, r in out.get(group, {}).items():
            M = P + AR.mname(group, name)
            iv = r.get("stock_seed", r["stock"])
            m[M] = f"{r['stock']['mean']:+.3f}"
            m[M + "Lo"], m[M + "Hi"] = (f"{r['stock'][k]:+.3f}" for k in ("lo", "hi"))
            m[M + "SeedLo"], m[M + "SeedHi"] = (f"{iv[k]:+.3f}" for k in ("lo", "hi"))
            m[M + "Margin"] = r["margin"]
    g, f = out.get("gross", {}).get("gross ms+tick vs gross raw"), out.get("net", {}).get("fee ms+tick vs fee raw")
    if g and f:  # the unit contrast net of fees under fee-aware training, as a share of its gross size
        m[P + "FeeShrink"] = f"{100 * f['stock']['mean'] / g['stock']['mean']:.0f}\\%"
    for name, r in out.get("bps", {}).items():
        M = P + AR.mname("bps", name)
        m[M] = f"{r['stock']['mean']:+.2f}"
        m[M + "Lo"], m[M + "Hi"] = (f"{r['stock_seed'][k]:+.2f}" for k in ("lo", "hi"))
    for c, v in out.get("passive", {}).items():
        m[P + AR.mname("passive", c.replace(":", " "))] = f"{100 * v:.0f}\\%"
    hs = out.get("hindsight")
    if hs:
        m[P + "HsEpisodes"] = f"{hs['episodes']:,}".replace(",", "{,}")
        m[P + "HsDropped"] = str(hs["dropped_overflow"])
        m[P + "HsStatesMedian"] = f"{hs['states_median']:.0f}"
        m[P + "HsStatesMax"] = f"{hs['states_max']:,}".replace(",", "{,}")
        for kind in ("gross", "net"):
            K_ = P + "Hs" + kind.capitalize()
            for c, v in hs[kind]["levels"].items():
                m[K_ + AR.mname("level", c.replace(":", " "))] = f"{v:+.3f}"
            m[K_ + "Headroom"] = f"{hs[kind]['headroom']:.2f}"
            m[K_ + "MarginShare"] = f"{100 * hs[kind]['margin_share']:.1f}\\%"
            for k in ("immediate", "post"):
                if f"margin_vs_gain_{k}" in hs[kind]:
                    m[K_ + AR.mname("margin vs gain", k)] = f"{100 * hs[kind][f'margin_vs_gain_{k}']:.0f}\\%"
            for c, r in hs[kind]["share_closed"].items():
                M = K_ + AR.mname("share", c.replace(":", " "))
                m[M] = f"{100 * r['mean']:.0f}\\%"
                m[M + "Lo"], m[M + "Hi"] = (f"{100 * r[k]:.0f}" for k in ("lo", "hi"))
    return m


INJECT = {"zero_R3tick_s0": "MsTick", "zero_R0_s0": "Raw", "zero_R3_s0": "Ms"}
SCALE_NAMES = {"0.0": "Real", "0.1": "Ten", "0.2": "Twenty", "0.5": "Fifty", "1.0": "Hundred", "2.0": "TwoHundred"}


def stock_aware(data: str = "valseen_dense_v0") -> tuple[dict, dict]:
    """The per-stock patience comparator on validation (experiments/stock_aware.py): its leave-one-day-out cost on
    the stocks' own 2019 days against the pooled policy's, and the injected-structure check on the held-out days."""
    out, m = {}, {}
    for model, M in INJECT.items():
        f = OUT / "aware" / f"{model}__{data}__shift.npz"
        if f.exists():
            z = np.load(f)
            d = float(z["cv_cost"] - z["base_cost"])
            out[model] = {"cv_minus_pooled": d}
            m["RFourShiftCv" + M] = f"{d:+.4f}"
        g = OUT / f"aware_inject_{model}.json"
        if g.exists():
            j = json.loads(g.read_text())
            out.setdefault(model, {})["inject"] = j["scales"]
            m["RFourInjectStocks"] = str(j["stocks"])
            for s, r in j["scales"].items():
                K_ = "RFourInject" + M + SCALE_NAMES.get(s, s.replace(".", ""))
                for part, key in (("", "gain"), ("Recovered", "recovered"), ("Achievable", "achievable")):
                    m[K_ + part], m[K_ + part + "Lo"], m[K_ + part + "Hi"] = (f"{r[key][k]:+.3f}"
                                                                          for k in ("mean", "lo", "hi"))
    return out, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=list(EPISODES))
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    res, info = load(a.eval)
    seen = pd.Index(pd.read_parquet(DATA / "transitions" / SEEN_DATA[a.eval] / "eps.parquet",
                                    columns=["stock"])["stock"].unique())
    out = {"eval": a.eval, "policies": sorted(res["col"].unique()), "stocks": int(info["stock"].nunique()),
           "seen_stocks": int(info["stock"].isin(seen).groupby(info["stock"]).first().sum()),
           "net": contrasts(res, info, FEE_CONTRASTS, "net_spread", seen, a.n_boot),
           "gross": contrasts(res, info, GROSS_CONTRASTS, "c_spread", seen, a.n_boot),
           "bps": bps(res, info, a.n_boot), "passive": passive(res),
           "hindsight": hindsight(res, info, a.eval, a.n_boot), "provenance": provenance(args=vars(a))}
    macros = macros_of(out, PREFIX[a.eval])
    if a.eval == "val_v0":
        out["stock_aware"], m = stock_aware()
        macros.update(m)
    write_json(OUT / f"round4_{a.eval}.json", out)
    with open(REPO / "paper" / f"numbers_round4_{a.eval}.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_round4.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for group in ("net", "gross"):
        for name, r in out[group].items():
            s = r["stock"]
            print(f"{group:5s} {name:32s} {s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] ({r['stocks']} stocks, "
                  f"{r['margin']})")
    for name, r in out["bps"].items():
        s, ss = r["stock"], r["stock_seed"]
        print(f"bps   {name:40s} {s['mean']:+.3f} [{ss['lo']:+.3f}, {ss['hi']:+.3f}] (stocks and seeds)")
    if out["hindsight"]:
        for kind in ("gross", "net"):
            h = out["hindsight"][kind]
            print(kind, "levels", {k: round(v, 3) for k, v in h["levels"].items()}, "headroom", round(h["headroom"], 3))
            print(kind, "share closed", {k: round(v["mean"], 3) for k, v in h["share_closed"].items()})


if __name__ == "__main__":
    main()

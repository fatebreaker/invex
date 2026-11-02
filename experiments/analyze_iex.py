"""The pre-specified secondary analyses of the IEX prospective test (prereg/iex_prospective_v1.md, "Secondary").

    python experiments/analyze_iex.py --eval iex_test --eval-all iex_test_all --macro-prefix IexTest
    python experiments/analyze_iex.py --eval iex_val_v0 --macro-prefix IexVal      # validation (not sealed)

Costs are per share, in spreads (c_spread) unless stated, paired per episode with seeds averaged; stocks are weighted
equally and each stock's days equally, and intervals come from the two-way bootstrap over stocks and days (90%,
10,000 replicates), as in the registered family. The population is the registered one (episodes whose IEX book was at
one tick for at least half of the trailing window) except in analysis 4a. Analyses:

  1. I2 with the registered equity margin (0.015) instead of the IEX margin;
  2. H1 on IEX: zero-shot MS+tick against the policy that also trained on the stock's 2019 Nasdaq days (seent), on
     the stocks it saw; equivalence within the IEX margin and within 0.015;
  3. the other stocks-family contrasts of prereg/confirmatory_v1.json (H2-H9), equivalence tests at the IEX margin,
     Benjamini-Hochberg over the eight;
  4. I1-I3 (a) on every episode, competitive book or not (the unrestricted replay: the table of --eval-all, or the
     main table before the population filter), and (b) on the episodes of the population whose immediate cross at
     the start fits in the visible book (no depth shortfall);
  5. I1-I3 in basis points of the arrival mid;
  6. I2 and I3 day by day, and the number of days that agree in sign with the pooled estimate;
  7. I2 and I3 by relative-tick tercile (frozen cuts);
  8. the neural learner (zeronn, five seeds): its MS+tick against crossing immediately and against the IEX-trained
     policy, and its raw units against its MS+tick.
Writes results/transfer/iex_secondary_<eval>.{json,md} and paper/numbers_iex_<eval>.tex.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from invex.env.units import unit_frame
from invex.eval import freeze
from invex.eval.stats import bh, boot_p
from invex.eval.universe import load_splits
from invex.paths import DATA, REPO, RESULTS
from invex.provenance import provenance, write_json

from confirm import mname, policy_costs, two_way_boots  # noqa: E402  (experiments/ is on the path when run)

I_FAMILY = (("I1", "zero:R3tick", "base:immediate", "less"), ("I2", "zero:R3tick", "iexseen:R3tick", "noninferior"),
            ("I3", "zero:R0", "zero:R3tick", "greater"))
NN_FAMILY = (("N1", "zeronn:R3tick", "base:immediate", "less"), ("N2", "zeronn:R3tick", "iexseen:R3tick", "noninferior"),
             ("N3", "zeronn:R0", "zeronn:R3tick", "greater"))
TERCILES = ("small", "mid", "large")


def test(diff: pd.Series, kind: str, delta: float, n_boot: int) -> dict:
    diff = diff.dropna()
    if diff.empty:
        return {"n_stocks": 0}
    r = two_way_boots(diff, n_boot, 0.90)
    boots = r.pop("boots")
    kinds = [kind] if kind else []
    out = {**r, "n_stocks": int(diff.index.get_level_values("stock").nunique()), "n_eps": int(len(diff))}
    for k in kinds:
        out["p"] = boot_p(boots, k, delta)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--spec", default="prereg/confirmatory_iex_v1.json")
    ap.add_argument("--eval-all", default="", help="evaluation of the unrestricted episode table (analysis 4a)")
    ap.add_argument("--macro-prefix", default="")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    spec = json.loads(open(REPO / a.spec).read()) if (REPO / a.spec).exists() else None
    fam = spec["families"]["iex"]
    d_iex = next(h["delta"] for h in fam["hypotheses"] if h["id"] == "I2")
    min_onetick = fam.get("min_onetick", 0.0)
    stocks_fam = json.loads((REPO / "prereg" / "confirmatory_v1.json").read_text())["families"]["stocks"]
    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    name = meta["episodes"]
    sealed = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"]
    eps = freeze.load(name, unseal=sealed, reason=f"secondary analysis of {a.eval} (iex-v1)" if sealed else "")
    sp = load_splits()
    uf = unit_frame(eps, sp["ko_constants"])
    eta = pd.Series(uf["_eta_sigma"].to_numpy(), index=eps["stock"]).groupby(level=0).median()
    terc = pd.Series(np.digitize(np.log(eta.to_numpy()), np.log(sp["eta_sigma_tercile_cuts"])), index=eta.index)
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    res = res[~res["excluded"]]
    if a.eval_all:  # analysis 4a: every episode, competitive book or not
        m_all = json.loads((RESULTS / "transfer" / f"eval_{a.eval_all}.json").read_text())["episodes"]
        if json.loads((REPO / "prereg" / f"episodes_{m_all}.json").read_text())["sealed"]:
            freeze.load(m_all, unseal=True, reason=f"secondary analysis 4a of {a.eval} (iex-v1)")
        r_all = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval_all}.parquet")
        every = policy_costs(r_all[~r_all["excluded"]], "c_spread", "seent")
    else:
        every = policy_costs(res, "c_spread", "seent")
    pop = set(eps.loc[eps["w_onetick"] >= min_onetick, "key"])
    res = res[res["key"].isin(pop)]
    per = policy_costs(res, "c_spread", "seent")
    per_bps = policy_costs(res, "c_bps", "seent")
    imm = res[res["policy"] == "immediate"].set_index("key")["shortfall"]
    absorbable = set(imm.index[imm == 0])
    keys = per.index.get_level_values("key")
    out = {"eval": a.eval, "delta_iex": d_iex, "min_onetick": min_onetick, "episodes": int(len(per)),
           "episodes_all": int(len(every)),
           "stocks": int(per.index.get_level_values("stock").nunique()), "analyses": {},
           "provenance": provenance(args=vars(a))}
    A = out["analyses"]

    def diff(tab, x, y, sel=None):
        if x not in tab or y not in tab:
            return pd.Series(dtype=float)
        d_ = tab[x] - tab[y]
        return d_ if sel is None else d_[sel]

    # 1. I2 at the equity margin
    A["I2_delta015"] = test(diff(per, "zero:R3tick", "iexseen:R3tick"), "noninferior", 0.015, a.n_boot)
    # 2. H1 on IEX
    seen = set(pd.read_parquet(DATA / "transitions" / "testseen_v0" / "eps.parquet", columns=["stock"])["stock"])
    on_seen = per.index.get_level_values("stock").isin(seen)
    h1 = diff(per, "zero:R3tick", "seen:R3tick", on_seen)
    A["H1_iex"] = test(h1, "tost", d_iex, a.n_boot)
    A["H1_iex"]["p_delta015"] = test(h1, "tost", 0.015, a.n_boot).get("p")
    # 3. H2-H9
    hs = []
    for h in stocks_fam["hypotheses"]:
        if h["id"] == "H1":
            continue
        r = test(diff(per, h["a"], h["b"]), h["test"], d_iex, a.n_boot)
        A[h["id"] + "_iex"] = {**r, "a": h["a"], "b": h["b"], "test": h["test"]}
        hs.append(h["id"] + "_iex")
    ps = [A[k].get("p", np.nan) for k in hs]
    ok = [k for k, p in zip(hs, ps) if np.isfinite(p)]
    for k, q in zip(ok, bh([A[k]["p"] for k in ok])):
        A[k]["p_bh"] = float(q)
    # 4. subsets, 5. bp, 6. by day, 7. by tercile
    for hid, x, y, kind in I_FAMILY:
        dlt = d_iex if kind == "noninferior" else 0.0
        A[f"{hid}_all"] = test(diff(every, x, y), kind, dlt, a.n_boot)
        A[f"{hid}_absorbable"] = test(diff(per, x, y, keys.isin(absorbable)), kind, dlt, a.n_boot)
        A[f"{hid}_bps"] = test(diff(per_bps, x, y), "", 0.0, a.n_boot)
        dd = diff(per, x, y)
        if dd.empty:
            continue
        if hid in ("I2", "I3"):
            by_day = dd.groupby(level=["stock", "day"]).mean().groupby(level="day").mean()
            pooled = float(dd.groupby(level=["stock", "day"]).mean().groupby(level="stock").mean().mean())
            A[f"{hid}_by_day"] = {"days": {k: float(v) for k, v in by_day.items()},
                                 "agree": int((np.sign(by_day) == np.sign(pooled)).sum()), "n_days": int(len(by_day))}
            st = dd.index.get_level_values("stock")
            for t in range(3):
                A[f"{hid}_{TERCILES[t]}"] = test(dd[terc.reindex(st).to_numpy() == t], kind, dlt, a.n_boot)
    # 8. the neural learner
    for hid, x, y, kind in NN_FAMILY:
        A[hid] = test(diff(per, x, y), kind, d_iex if kind == "noninferior" else 0.0, a.n_boot)
    write_json(RESULTS / "transfer" / f"iex_secondary_{a.eval}.json", out)
    lines = [f"# IEX secondary analyses: {a.eval}", "", f"{out['episodes']:,} episodes, {out['stocks']} stocks; IEX "
             f"margin {d_iex}", "", "| analysis | estimate [90% interval] | p | stocks |", "|---|---|---|---|"]
    macros = {}
    for k, v in A.items():
        if "mean" in v:
            p = v.get("p_bh", v.get("p"))
            lines.append(f"| {k} | {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] | "
                         f"{'' if p is None else format(p, '.4f')} | {v['n_stocks']} |")
            s_ = mname(a.macro_prefix, k.replace("_", ""))
            nd = 2 if k.endswith("_bps") else 3
            macros[s_], macros[s_ + "Lo"], macros[s_ + "Hi"] = (f"{v['mean']:+.{nd}f}", f"{v['lo']:+.{nd}f}",
                                                                 f"{v['hi']:+.{nd}f}")
            macros[s_ + "Stocks"] = f"{v['n_stocks']:,}"
            if p is not None:
                macros[s_ + "P"] = f"{p:.3f}" if p >= 0.001 else "<0.001"
        elif "days" in v:
            lines.append(f"| {k} | " + ", ".join(f"{d}: {x:+.3f}" for d, x in v["days"].items())
                         + f" | agree {v['agree']}/{v['n_days']} | |")
            macros[mname(a.macro_prefix, k.replace("_", ""), "agree")] = str(v["agree"])
            macros[mname(a.macro_prefix, k.replace("_", ""), "ndays")] = str(v["n_days"])
    (RESULTS / "transfer" / f"iex_secondary_{a.eval}.md").write_text("\n".join(lines) + "\n")
    with open(REPO / "paper" / f"numbers_iex_{a.eval}.tex", "w") as fh:
        fh.write(f"% generated by experiments/analyze_iex.py --eval {a.eval} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Post hoc re-analysis of the prospective IEX test (TMLR review, round 1). Decides nothing; every number is
labelled post hoc in the paper.

    python experiments/iex_test_reanalysis.py [--n-boot 10000]

Recomputes every registered IEX test from the stored registered evaluations with the registered estimators and
bootstrap (experiments/confirm.py, experiments/confirm_robust.py: same population, units, seeds and bootstrap seed,
so the registered estimates and p-values are reproduced exactly; the script checks this), under both estimands of
iex-v8: (a) every stock weighted equally and each of its days equally, (b) every episode weighted equally. It adds:
  1. the 95% intervals, which invert the iex-v8 decisions at 0.025 (the registered tables print 90% intervals);
  2. P4 at the equity margin of 0.015 as well as the registered 0.020;
  3. a sensitivity that drops every episode in which some evaluated policy's cross ran beyond the venue's visible
     book (the shortfall rule of src/invex/env/execution.py), something a routed order would not do;
  4. a third weighting: each stock-day's mean weighted by the stock's traded notional on IEX that day;
  5. Holm over all 28 estimand-specific IEX tests (14 hypotheses x 2 estimands) at 0.05.
Loads each sealed episode table once through the logged loader (reason "post hoc ..."). Writes
results/transfer/iex_test_reanalysis.json and paper/numbers_iex_reanalysis.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import policy_costs  # noqa: E402

from invex.eval import freeze  # noqa: E402
from invex.eval.stats import boot_p, holm, pigeonhole_bootstrap  # noqa: E402
from invex.paths import DATA, REPO, RESULTS  # noqa: E402
from invex.provenance import write_json  # noqa: E402

# (eval, registered spec, family, registered result stem (a), stored episode-family result stem (b))
FAMILIES = [
    ("iex_test", "confirmatory_iex_v1.json", "iex", "confirm_iex_test_iex", "confirm_iex_test_iex_episode"),
    ("iex_test_ppo", "confirmatory_iex_v2.json", "iex", "confirm_iex_test_ppo_iex", "confirm_iex_test_ppo_iexppo_episode"),
    ("iex_test_ppo", "confirmatory_iex_v2.json", "iexlearner", "confirm_iex_test_ppo_iexlearner",
     "confirm_iex_test_ppo_iexlearner_episode"),
    ("iex_test_wide", "confirmatory_iex_v3.json", "iexwide", "confirm_iex_test_wide_iexwide",
     "confirm_iex_test_wide_iexwide_episode"),
    ("iex_test_v8", "confirmatory_iex_v8.json", "iexlm", "confirm_iex_test_v8_iexlm", "confirm_iex_test_v8_iexlm_episode"),
]
LETTERS = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}


def by_stock_table(diff: pd.Series):
    cm = diff.groupby(level=["stock", "day"]).mean().unstack()
    w = cm.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    return (cm.fillna(0.0) * w).to_numpy(), w.to_numpy()


def by_episode_table(diff: pd.Series):
    g = diff.groupby(level=["stock", "day"])
    return g.sum().unstack().fillna(0.0).to_numpy(), g.count().unstack().fillna(0.0).to_numpy()


def by_notional_table(diff: pd.Series, notional: pd.Series):
    cm = diff.groupby(level=["stock", "day"]).mean()
    w = notional.reindex(cm.index).fillna(0.0)
    return (cm * w).unstack().fillna(0.0).to_numpy(), w.unstack().fillna(0.0).to_numpy()


def test(sums, counts, kind, delta, n_boot, deltas=()):
    r = pigeonhole_bootstrap(sums, counts, n_boot=n_boot, level=0.90, return_boots=True)
    b = r.pop("boots")
    lo95, hi95 = np.quantile(b, [0.025, 0.975])
    out = {**r, "lo95": float(lo95), "hi95": float(hi95), "p": boot_p(b, kind, delta)}
    for d in deltas:
        out[f"p_delta_{d}"] = boot_p(b, kind, d)
    return out


def notional_of(days) -> pd.Series:
    parts = []
    for d in days:
        st = pd.read_parquet(DATA / "iex" / d / "daystats.parquet", columns=["stock", "volume", "vwap"])
        parts.append(st.assign(day=d, notional=st["volume"] * st["vwap"]))
    s = pd.concat(parts).set_index(["stock", "day"])["notional"]
    return s[~s.index.duplicated()]


def mac(s: str) -> str:
    return "".join(LETTERS.get(c, c) for c in s)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    tables, out, checks = {}, {"families": [], "holm_all": {}}, []
    for ev, spec_name, fam_name, stem_a, stem_b in FAMILIES:
        meta = json.loads((RESULTS / "transfer" / f"eval_{ev}.json").read_text())
        name = meta["episodes"]
        if name not in tables:
            tables[name] = freeze.load(name, unseal=True,
                                       reason="post hoc (review round 1): 95% intervals, margin 0.015 for P4, "
                                              "shortfall-free and notional-weighted sensitivities, Holm over all "
                                              "IEX tests, from the stored registered evaluations")
        eps = tables[name]
        spec = json.loads((REPO / "prereg" / spec_name).read_text())
        fam = spec["families"][fam_name]
        w = eps["w_onetick"]
        keep = set(eps.loc[(w >= fam.get("min_onetick", -np.inf)) & (w < fam.get("max_onetick", np.inf)), "key"])
        res = pd.read_parquet(RESULTS / "transfer" / f"eval_{ev}.parquet",
                              columns=["key", "policy", "excluded", "shortfall", "stock", "day", fam["unit"]])
        res = res[~res["excluded"] & res["key"].isin(keep)]
        short_keys = set(res.loc[res["shortfall"] > 0, "key"])
        per = policy_costs(res, fam["unit"], "seen")
        notional = notional_of(sorted(res["day"].unique()))
        reg_a = {h["id"]: h for h in json.loads((RESULTS / "transfer" / f"{stem_a}.json").read_text())["hypotheses"]}
        reg_b = {h["id"]: h for h in json.loads((RESULTS / "transfer" / f"{stem_b}.json").read_text())["hypotheses"]}
        rows = []
        for h in fam["hypotheses"]:
            diff = (per[h["a"]] - per[h["b"]]).dropna()
            delta = h.get("delta", spec.get("delta", 0.0))
            deltas = (0.015,) if h["id"] == "P4" else ()
            keys = diff.index.get_level_values("key")
            clean = diff[~keys.isin(short_keys)]
            r = {"id": h["id"], "test": h["test"], "delta": delta, "episodes": int(len(diff)),
                 "shortfall_episodes": int(keys.isin(short_keys).sum()),
                 "stocks": int(diff.index.get_level_values("stock").nunique())}
            r["a"] = test(*by_stock_table(diff), h["test"], delta, a.n_boot, deltas)
            r["b"] = test(*by_episode_table(diff), h["test"], delta, a.n_boot, deltas)
            r["a_clean"] = test(*by_stock_table(clean), h["test"], delta, a.n_boot)
            r["b_clean"] = test(*by_episode_table(clean), h["test"], delta, a.n_boot)
            r["notional"] = test(*by_notional_table(diff, notional), h["test"], delta, a.n_boot)
            for k, reg in (("a", reg_a), ("b", reg_b)):  # the registered numbers must come out again
                g = reg[h["id"]]
                same = (abs(g["estimates"]["all"]["mean"] - r[k]["mean"]) < 1e-9 and abs(g["p"] - r[k]["p"]) < 1e-12)
                checks.append((ev, fam_name, h["id"], k, same))
            rows.append(r)
        for est in ("a", "b", "a_clean", "b_clean", "notional"):  # Holm within the family, as registered
            for r, padj in zip(rows, holm([r[est]["p"] for r in rows])):
                r[est]["p_holm"] = float(padj)
        out["families"].append({"eval": ev, "spec": spec_name, "family": fam_name, "hypotheses": rows})
    # Holm over all 28 estimand-specific tests at 0.05
    allp = [(f["family"], r["id"], est, r[est]["p"]) for f in out["families"] for r in f["hypotheses"] for est in ("a", "b")]
    for (fam_name, hid, est, _p), padj in zip(allp, holm([x[3] for x in allp])):
        out["holm_all"][f"{hid}:{est}"] = float(padj)
    out["reproduced"] = all(c[-1] for c in checks)
    out["checks_failed"] = [c[:4] for c in checks if not c[-1]]
    write_json(RESULTS / "transfer" / "iex_test_reanalysis.json", out)

    m = {}
    for f in out["families"]:
        for r in f["hypotheses"]:
            p = "IexRe" + mac(r["id"])
            for est, tag in (("a", "Stock"), ("b", "Episode"), ("a_clean", "StockClean"), ("b_clean", "EpisodeClean"),
                             ("notional", "Notional")):
                e = r[est]
                m[f"{p}{tag}"] = f"{e['mean']:+.3f}"
                m[f"{p}{tag}LoNinetyfive"] = f"{e['lo95']:+.3f}"
                m[f"{p}{tag}HiNinetyfive"] = f"{e['hi95']:+.3f}"
                m[f"{p}{tag}Pholm"] = "<0.001" if e["p_holm"] < 0.001 else f"{e['p_holm']:.3f}"
            m[f"{p}ShortfallEpisodes"] = f"{r['shortfall_episodes']:,}".replace(",", "{,}")
            if "p_delta_0.015" in r["b"]:
                m[f"{p}EpisodePMarginFifteen"] = f"{r['b']['p_delta_0.015']:.3f}"
                m[f"{p}StockPMarginFifteen"] = f"{r['a']['p_delta_0.015']:.3f}"
    changed = [k for k, v in out["holm_all"].items()
               if (v < 0.05) != (next(r[k.split(':')[1]]["p_holm"] for f in out["families"] for r in f["hypotheses"]
                                      if r["id"] == k.split(":")[0]) < 0.025)]
    m["IexReHolmAllChanged"] = ", ".join(changed) if changed else "none"
    lines = ["% generated by experiments/iex_test_reanalysis.py (post hoc) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_iex_reanalysis.tex").write_text("\n".join(lines) + "\n")
    print(f"registered numbers reproduced: {out['reproduced']} {out['checks_failed']}")
    for f in out["families"]:
        for r in f["hypotheses"]:
            s = f"{r['id']:3s} ({r['test']}, delta {r['delta']}):"
            for est in ("a", "b", "a_clean", "b_clean", "notional"):
                e = r[est]
                s += f"  {est} {e['mean']:+.3f} [{e['lo95']:+.3f},{e['hi95']:+.3f}] pH {e['p_holm']:.4f}"
            if "p_delta_0.015" in r["b"]:
                s += f"  | P4 at 0.015: a p {r['a']['p_delta_0.015']:.3f}, b p {r['b']['p_delta_0.015']:.3f}"
            print(s + f"  ({r['shortfall_episodes']} shortfall episodes)")
    print("Holm over all 28 tests at 0.05 changes:", m["IexReHolmAllChanged"])


if __name__ == "__main__":
    main()

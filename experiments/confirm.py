"""Confirmatory analysis of a sealed stage: the pre-registered hypotheses, exactly as specified.

    python experiments/confirm.py --spec prereg/confirmatory_v1.json --family stocks --eval stage1 \\
        --seen-tag seent --seen-data testseen_v0 --macro-prefix StageOne
    python experiments/confirm.py --spec prereg/confirmatory_v1.json --family crypto --eval crypto_test --macro-prefix CryptoTest

The spec (frozen with the pre-registration) lists, per family, hypotheses about the paired cost difference
a - b between two policies on the same episodes ("<tag>:<representation>" averaged over the seeds present, or
"base:<baseline>"; tag "seen" stands for --seen-tag). Per hypothesis:

  * stocks: per-episode differences are averaged within stock; the stock-cluster bootstrap is stratified by
    relative-tick tercile (frozen cuts); costs in spreads per share;
  * crypto: every pair is weighted equally and each of its days equally; the bootstrap resamples pairs and days
    (two-way); costs in volatility per trade;
  * a family with "bootstrap": "two_way" (iex) weights and resamples its stocks and days the same way, in the
    family's unit; with "min_onetick", only episodes whose book was at one tick for at least that share of the
    trailing window are analysed, and with "max_onetick", only those below that share (the wide books); with
    "cap", every policy's cost is capped at that value in the family's unit;
  * the test is 'greater' (a costs more), 'less', 'tost' (equivalence within +-delta) or 'noninferior'
    (a costs less than b + delta), by percentile-bootstrap p-values; a hypothesis with "strata": "all+terciles"
    must hold overall and in every tercile (intersection-union: its p-value is the largest of the four);
  * Holm's adjustment runs over the hypotheses of the family; a hypothesis is supported if its adjusted p-value
    is below alpha.

Episodes flagged as excluded by the evaluation are dropped. Writes results/transfer/confirm_<eval>_<family>.{json,md}
and paper/numbers_confirm_<eval>_<family>.tex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

from invex.env.units import unit_frame
from invex.eval import freeze
from invex.eval.stats import boot_p, cluster_bootstrap, holm, pigeonhole_bootstrap
from invex.eval.universe import load_splits
from invex.paths import DATA, REPO, RESULTS
from invex.provenance import provenance, write_json

NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
TERCILES = ("small", "mid", "large")
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def mname(*parts) -> str:
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def policy_costs(res: pd.DataFrame, unit: str, seen_tag: str) -> pd.DataFrame:
    """Per-episode cost of every policy, seeds averaged: columns '<tag>:<rep>' and 'base:<name>'."""
    m = res["policy"].str.extract(NAME)
    tag = m["tag"].fillna("base").replace({seen_tag: "seen"})
    rep = m["rep"].fillna(res["policy"])
    col = tag + ":" + rep
    return res.assign(col=col).pivot_table(index=["key", "stock", "day"], columns="col", values=unit, aggfunc="mean")


def population(eps: pd.DataFrame, fam: dict) -> set | None:
    """Keys of the family's population: share of the trailing window the book sat at one tick in
    [min_onetick, max_onetick); None if the family names neither bound."""
    if "min_onetick" not in fam and "max_onetick" not in fam:
        return None
    w = eps["w_onetick"]
    return set(eps.loc[(w >= fam.get("min_onetick", -np.inf)) & (w < fam.get("max_onetick", np.inf)), "key"])


def two_way_boots(diff: pd.Series, n_boot: int, level: float) -> dict:
    cm = diff.groupby(level=["stock", "day"]).mean().unstack()
    w = cm.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    return pigeonhole_bootstrap((cm.fillna(0.0) * w).to_numpy(), w.to_numpy(), n_boot=n_boot, level=level,
                                return_boots=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--family", required=True, help="a family of the spec (stocks, crypto, iex, ...)")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--seen-tag", default="seen")
    ap.add_argument("--seen-data", default="", help="transitions the seen policy saw beyond the training set (its "
                    "stocks are the only ones a 'stocks: seen' hypothesis is tested on)")
    ap.add_argument("--macro-prefix", default="")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    spec_text = open(a.spec).read()
    spec = json.loads(spec_text)
    if a.family not in spec["families"]:
        raise SystemExit(f"family {a.family} is not in {a.spec}")
    fam = spec["families"][a.family]
    alpha = spec["alpha"]
    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    name = meta["episodes"]
    sealed = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"]
    eps = freeze.load(name, unseal=sealed, reason=f"confirmatory analysis of {a.eval}" if sealed else "")
    sp = load_splits()
    uf = unit_frame(eps, sp["ko_constants"])
    eps = eps.assign(sigma_trade=uf["_sigma_trade"].to_numpy(), eta=uf["_eta_sigma"].to_numpy())
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    n_excl = int(res.loc[res["excluded"], "key"].nunique()) if "excluded" in res else 0
    if n_excl:
        res = res[~res["excluded"]]
    n_pop = None
    keep = population(eps, fam)  # the family's population, by how competitive the book was before the start
    if keep is not None:
        n_pop = int(res["key"].nunique())
        res = res[res["key"].isin(keep)]
    unit = fam["unit"]
    if unit == "c_sigma":
        # evaluations written after prereg-v1 also store sigma_trade; the registered unit is the episode table's
        # value, and the stored copy must agree with it
        stored = res[["key", "sigma_trade"]].drop_duplicates("key") if "sigma_trade" in res else None
        res = res.drop(columns=["sigma_trade"], errors="ignore").merge(eps[["key", "sigma_trade"]], on="key")
        if stored is not None:
            both = stored.merge(eps[["key", "sigma_trade"]], on="key", suffixes=("_eval", "_eps"))
            if not np.allclose(both["sigma_trade_eval"], both["sigma_trade_eps"], rtol=1e-6, equal_nan=True):
                raise SystemExit("sigma_trade stored by the evaluation differs from the episode table")
        res["c_sigma"] = res["cost"] / (res["Q"] * res["sigma_trade"])
    if "cap" in fam:
        res = res.assign(**{unit: res[unit].clip(upper=fam["cap"])})
    per = policy_costs(res, unit, a.seen_tag)
    eta = eps.groupby("stock")["eta"].median()
    terc = pd.Series(np.digitize(np.log(eta.to_numpy()), np.log(sp["eta_sigma_tercile_cuts"])), index=eta.index)
    out = {"spec": a.spec, "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(), "family": a.family,
           "eval": a.eval, "episodes": int(per.shape[0]), "excluded": n_excl, "episodes_before_population": n_pop,
           "stocks": int(per.index.get_level_values("stock").nunique()), "hypotheses": [],
           "provenance": provenance(args=vars(a))}
    for h in fam["hypotheses"]:
        if h["a"] not in per or h["b"] not in per:
            out["hypotheses"].append({**h, "missing": [c for c in (h["a"], h["b"]) if c not in per]})
            continue
        diff = (per[h["a"]] - per[h["b"]]).dropna()
        if h.get("stocks") == "seen":  # a transfer gap is defined only for stocks the seen policy saw
            if not a.seen_data:
                raise SystemExit(f"{h['id']}: --seen-data is required")
            seen_stocks = set(pd.read_parquet(DATA / "transitions" / a.seen_data / "eps.parquet", columns=["stock"])["stock"])
            diff = diff[diff.index.get_level_values("stock").isin(seen_stocks)]
        level = 0.90 if h["test"] in ("tost", "noninferior") else 1 - 2 * alpha  # the interval the test inverts
        delta = h.get("delta", spec.get("delta", 0.0))
        parts = {}
        if a.family == "crypto" or fam.get("bootstrap") == "two_way":
            parts["all"] = two_way_boots(diff, a.n_boot, level)
        else:
            by_stock = diff.groupby(level="stock").mean()
            st = terc.reindex(by_stock.index).to_numpy()
            parts["all"] = cluster_bootstrap(by_stock.to_numpy(), st, n_boot=a.n_boot, level=level, return_boots=True)
            if h.get("strata") == "all+terciles":
                for t in range(3):
                    sel = st == t
                    parts[TERCILES[t]] = cluster_bootstrap(by_stock.to_numpy()[sel], n_boot=a.n_boot, level=level,
                                                           return_boots=True)
        ps = {k: boot_p(v.pop("boots"), h["test"], delta) for k, v in parts.items()}
        out["hypotheses"].append({**h, "delta": delta, "level": level, "estimates": parts, "p_parts": ps,
                                  "p": max(ps.values()), "assets": int(diff.index.get_level_values("stock").nunique())})
    tested = [x for x in out["hypotheses"] if "p" in x]
    for x, padj in zip(tested, holm([x["p"] for x in tested])):
        x["p_holm"] = float(padj)
        x["supported"] = bool(padj < alpha)
    stem = f"confirm_{a.eval}_{a.family}"
    write_json(RESULTS / "transfer" / f"{stem}.json", out)
    lines = [f"# Confirmatory analysis: {a.eval}, family {a.family}", "",
             f"Spec {a.spec} (sha256 {out['spec_sha256'][:12]}), {out['episodes']:,} episodes, {out['stocks']} assets, "
             f"{n_excl} excluded; alpha {alpha}, Holm over {len(tested)} hypotheses.", "",
             "| id | claim | a | b | test | estimate (interval) | p | Holm p | supported |", "|---|---|---|---|---|---|---|---|---|"]
    macros = {}
    for x in out["hypotheses"]:
        if "p" not in x:
            lines.append(f"| {x['id']} | {x.get('claim', '')} | {x['a']} | {x['b']} | {x['test']} | missing {x['missing']} | | | |")
            continue
        e = x["estimates"]["all"]
        lines.append(f"| {x['id']} | {x.get('claim', '')} | {x['a']} | {x['b']} | {x['test']}"
                     f"{' ±' + format(x['delta'], 'g') if x['test'] in ('tost', 'noninferior') else ''} | "
                     f"{e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] | {x['p']:.4f} | {x['p_holm']:.4f} | "
                     f"{'yes' if x['supported'] else 'no'} |")
        for k, v in x["estimates"].items():
            s_ = mname(a.macro_prefix, x["id"], k)
            macros[s_], macros[s_ + "Lo"], macros[s_ + "Hi"] = f"{v['mean']:+.3f}", f"{v['lo']:+.3f}", f"{v['hi']:+.3f}"
        macros[mname(a.macro_prefix, x["id"], "pholm")] = f"{x['p_holm']:.3f}" if x["p_holm"] >= 0.001 else "<0.001"
        macros[mname(a.macro_prefix, x["id"], "supported")] = "supported" if x["supported"] else "not supported"
    (RESULTS / "transfer" / f"{stem}.md").write_text("\n".join(lines) + "\n")
    with open(REPO / "paper" / f"numbers_{stem}.tex", "w") as fh:
        fh.write(f"% generated by experiments/confirm.py --eval {a.eval} --family {a.family} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Secondary IEX families with estimators that no cost cap drives (iex-v6, prereg/iex_robust_v6.md).

    python experiments/confirm_robust.py --spec prereg/confirmatory_iex_v6.json --family iex_episode --eval iex_test \\
        --macro-prefix IexTestEpisode

A family of the spec names one of two estimators of the paired cost difference a - b, otherwise exactly as in
experiments/confirm.py (population, units, exclusions, seeds averaged per episode, tests, Holm within the family):
  * "estimator": "episode": every episode weighted equally (instead of every stock, and each of its days, equally);
    two-way bootstrap over stocks and days on the cell sums and counts;
  * "estimator": "median": the median over stocks of each stock's mean difference (its days weighted equally); the
    two-way bootstrap resamples stocks and days, recomputes each stock's mean over its resampled days and takes the
    weighted median over the resampled stocks.
Reads the episode table only for the population and the units, as confirm.py does (each read logged). Writes
results/transfer/confirm_<eval>_<family>.{json,md} and paper/numbers_confirm_<eval>_<family>.tex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import mname, policy_costs, population  # noqa: E402

from invex.eval import freeze  # noqa: E402
from invex.eval.stats import boot_p, holm, pigeonhole_bootstrap  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def median_boots(diff: pd.Series, n_boot: int, level: float, seed: int = 0) -> dict:
    """Median over stocks of each stock's mean (days weighted equally), two-way bootstrap over stocks and days."""
    cm = diff.groupby(level=["stock", "day"]).mean().unstack()
    v = np.nan_to_num(cm.to_numpy())
    m = cm.notna().to_numpy().astype(float)
    A, D = v.shape
    rng = np.random.default_rng(seed)

    def stat(wa, wd):
        den = m @ wd
        per = (v * m) @ wd / np.where(den > 0, den, 1.0)
        return weighted_median(per, wa * (den > 0))

    est = stat(np.ones(A), np.ones(D))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wa = np.bincount(rng.integers(0, A, A), minlength=A).astype(float)
        wd = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        boots[b] = stat(wa, wd)
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return {"mean": est, "lo": float(lo), "hi": float(hi), "se": float(boots.std(ddof=1)), "boots": boots}


def episode_boots(diff: pd.Series, n_boot: int, level: float) -> dict:
    """Every episode weighted equally, two-way bootstrap over stocks and days."""
    g = diff.groupby(level=["stock", "day"])
    sums, counts = g.sum().unstack(), g.count().unstack()
    return pigeonhole_bootstrap(sums.fillna(0.0).to_numpy(), counts.fillna(0.0).to_numpy(), n_boot=n_boot,
                                level=level, return_boots=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--macro-prefix", default="")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    spec_text = open(a.spec).read()
    spec = json.loads(spec_text)
    fam = spec["families"][a.family]
    alpha, est_kind = spec["alpha"], fam["estimator"]
    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    name = meta["episodes"]
    sealed = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"]
    eps = freeze.load(name, unseal=sealed, reason=f"secondary analysis ({est_kind}) of {a.eval} (iex-v6)" if sealed else "")
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a.eval}.parquet")
    n_excl = int(res.loc[res["excluded"], "key"].nunique()) if "excluded" in res else 0
    if n_excl:
        res = res[~res["excluded"]]
    keep = population(eps, fam)
    if keep is not None:
        res = res[res["key"].isin(keep)]
    unit = fam["unit"]
    if "cap" in fam:
        res = res.assign(**{unit: res[unit].clip(upper=fam["cap"])})
    per = policy_costs(res, unit, "seen")
    out = {"spec": a.spec, "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(), "family": a.family,
           "estimator": est_kind, "eval": a.eval, "episodes": int(per.shape[0]), "excluded": n_excl,
           "stocks": int(per.index.get_level_values("stock").nunique()), "hypotheses": [],
           "provenance": provenance(args=vars(a))}
    for h in fam["hypotheses"]:
        if h["a"] not in per or h["b"] not in per:
            out["hypotheses"].append({**h, "missing": [c for c in (h["a"], h["b"]) if c not in per]})
            continue
        diff = (per[h["a"]] - per[h["b"]]).dropna()
        level = 0.90 if h["test"] in ("tost", "noninferior") else 1 - 2 * alpha
        delta = h.get("delta", spec.get("delta", 0.0))
        e = (episode_boots if est_kind == "episode" else median_boots)(diff, a.n_boot, level)
        p = boot_p(e.pop("boots"), h["test"], delta)
        out["hypotheses"].append({**h, "delta": delta, "level": level, "estimates": {"all": e}, "p": p})
    tested = [x for x in out["hypotheses"] if "p" in x]
    for x, padj in zip(tested, holm([x["p"] for x in tested])):
        x["p_holm"], x["supported"] = float(padj), bool(padj < alpha)
    stem = f"confirm_{a.eval}_{a.family}"
    write_json(RESULTS / "transfer" / f"{stem}.json", out)
    lines = [f"# Secondary analysis ({est_kind}): {a.eval}, family {a.family}", "",
             f"Spec {a.spec} (sha256 {out['spec_sha256'][:12]}), {out['episodes']:,} episodes, {out['stocks']} stocks, "
             f"{n_excl} excluded; alpha {alpha}, Holm over {len(tested)} hypotheses.", "",
             "| id | a | b | test | estimate (interval) | p | Holm p | supported |", "|---|---|---|---|---|---|---|---|"]
    macros = {}
    for x in tested:
        e = x["estimates"]["all"]
        lines.append(f"| {x['id']} | {x['a']} | {x['b']} | {x['test']} | {e['mean']:+.4f} [{e['lo']:+.4f}, "
                     f"{e['hi']:+.4f}] | {x['p']:.4f} | {x['p_holm']:.4f} | {'yes' if x['supported'] else 'no'} |")
        s_ = mname(a.macro_prefix, x["id"], "all")
        macros[s_], macros[s_ + "Lo"], macros[s_ + "Hi"] = f"{e['mean']:+.3f}", f"{e['lo']:+.3f}", f"{e['hi']:+.3f}"
        macros[mname(a.macro_prefix, x["id"], "pholm")] = f"{x['p_holm']:.3f}" if x["p_holm"] >= 0.001 else "<0.001"
        macros[mname(a.macro_prefix, x["id"], "supported")] = "supported" if x["supported"] else "not supported"
    (RESULTS / "transfer" / f"{stem}.md").write_text("\n".join(lines) + "\n")
    with open(REPO / "paper" / f"numbers_{stem}.tex", "w") as fh:
        fh.write(f"% generated by experiments/confirm_robust.py --eval {a.eval} --family {a.family} -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

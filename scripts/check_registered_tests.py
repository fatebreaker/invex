"""Recompute every registered test on sealed data from the supplement's test inputs, with the registered code.

    python scripts/check_registered_tests.py [--dir supplement/registered_tests] [--only STEM ...]

Needs no data. For each stored analysis in DIR/analyses (written by experiments/confirm.py or
experiments/confirm_robust.py and exported by experiments/export_test_inputs.py), rebuilds what its estimator reads
from DIR/cells.parquet (per hypothesis, stock and day: the count, sum and mean of the paired differences) and
DIR/stocks.parquet (per hypothesis and stock: the mean over its episodes and its relative-tick tercile), reruns the
registered estimator and bootstrap (the same functions, seeds and number of replicates), the p-values and Holm's
adjustment over the family, and compares every estimate, interval end, p-value and decision with the stored analysis.
Prints one line per analysis and exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from confirm import TERCILES, two_way_boots  # noqa: E402
from confirm_robust import median_boots  # noqa: E402

from invex.eval.stats import boot_p, cluster_bootstrap, holm, pigeonhole_bootstrap  # noqa: E402
from invex.paths import REPO  # noqa: E402


def rerun(script: str, args: dict, spec: dict, fam: dict, h: dict, cells: pd.DataFrame, stocks: pd.DataFrame) -> dict:
    """Estimates and p-values of one hypothesis, as its script computes them from the paired differences."""
    level = 0.90 if h["test"] in ("tost", "noninferior") else 1 - 2 * spec["alpha"]
    delta = h.get("delta", spec.get("delta", 0.0))
    n_boot = args["n_boot"]
    cell = cells.set_index(["stock", "day"])
    parts = {}
    if script == "confirm_robust.py":
        if fam["estimator"] == "episode":
            parts["all"] = pigeonhole_bootstrap(cell["sum"].unstack().fillna(0.0).to_numpy(),
                                                cell["n"].unstack().fillna(0.0).to_numpy(), n_boot=n_boot,
                                                level=level, return_boots=True)
        else:  # one value per cell, so the script's own cell means are these
            parts["all"] = median_boots(cell["mean"], n_boot, level)
    elif args["family"] == "crypto" or fam.get("bootstrap") == "two_way":
        parts["all"] = two_way_boots(cell["mean"], n_boot, level)
    else:
        x, st = stocks["mean"].to_numpy(), stocks["tercile"].to_numpy()
        parts["all"] = cluster_bootstrap(x, st, n_boot=n_boot, level=level, return_boots=True)
        if h.get("strata") == "all+terciles":
            for t in range(3):
                parts[TERCILES[t]] = cluster_bootstrap(x[st == t], n_boot=n_boot, level=level, return_boots=True)
    ps = {k: boot_p(v.pop("boots"), h["test"], delta) for k, v in parts.items()}
    return {"estimates": parts, "p_parts": ps, "p": max(ps.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO / "supplement" / "registered_tests")
    ap.add_argument("--only", nargs="*", default=None, help="analysis stems to check (default: all)")
    a = ap.parse_args()
    cells = pd.read_parquet(a.dir / "cells.parquet")
    stocks = pd.read_parquet(a.dir / "stocks.parquet")
    cells = {k: g for k, g in cells.groupby(["analysis", "hypothesis"], sort=False)}
    stocks = {k: g for k, g in stocks.groupby(["analysis", "hypothesis"], sort=False)}
    bad, n_tests, n_supported = [], 0, 0
    for f in sorted((a.dir / "analyses").glob("*.json")):
        if a.only and f.stem not in a.only:
            continue
        d = json.loads(f.read_text())
        script, args = Path(d["provenance"]["argv"][0]).name, d["provenance"]["args"]
        spec = json.loads((REPO / args["spec"]).read_text())
        fam = spec["families"][args["family"]]
        tested = [x for x in d["hypotheses"] if "p" in x]
        new = [rerun(script, args, spec, fam, x, cells[(f.stem, x["id"])], stocks[(f.stem, x["id"])]) for x in tested]
        for x, y, padj in zip(tested, new, holm([y["p"] for y in new])):
            for k, e in x["estimates"].items():
                for q in ("mean", "lo", "hi"):
                    if abs(e[q] - y["estimates"][k][q]) > 1e-12:
                        bad.append(f"{f.stem} {x['id']} {k} {q}: stored {e[q]!r}, recomputed {y['estimates'][k][q]!r}")
            checks = [("p", y["p"]), ("p_holm", float(padj)), ("supported", bool(padj < spec["alpha"]))]
            checks += [("p_parts", y["p_parts"])] if "p_parts" in x else []
            for q, v in checks:
                if x[q] != v:
                    bad.append(f"{f.stem} {x['id']} {q}: stored {x[q]!r}, recomputed {v!r}")
        n_tests += len(tested)
        n_supported += sum(x["supported"] for x in tested)
        decisions = ", ".join(f"{x['id']} {'yes' if x['supported'] else 'no'}" for x in tested)
        print(f"{f.stem:44s} {script:18s} {decisions}", flush=True)
    print(f"\n{n_tests} tests ({n_supported} supported) recomputed;", "no difference" if not bad else f"{len(bad)} differences:")
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

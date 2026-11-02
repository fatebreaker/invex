"""The registered stock-weighted IEX test-day tests rerun without the one-episode stock NHP (post hoc; no sealed read).

    python experiments/iex_without_nhp.py [--dir supplement/registered_tests]
    python experiments/iex_without_nhp.py --from-evals    # the same from the stored evaluations (no sealed read)

Uses the supplement's inputs of the registered tests (experiments/export_test_inputs.py): for the families decided with
stocks weighted equally (I1-I3, P1-P3, L1-L2, P4, and the one-tick learned-scale family Y1-Y2), it drops NHP's
stock-day cells and reruns the registered estimator (confirm.two_way_boots: two-way bootstrap over stocks and days,
seed 0, 10,000 replicates), the percentile p-value and Holm's adjustment within the family, and reports the estimate,
its 95% interval and the decision at 2.5% (iex-v8) and at 5% (as first registered). With NHP kept it reproduces the
stored analyses. With --from-evals it prepares the cells itself, as confirm.py does, from the stored evaluations: every
episode of the IEX test table is in the one-tick population, which each stored analysis's episode count confirms, so
no sealed table is read. Writes
results/transfer/iex_without_nhp.json and paper/numbers_iex_without_nhp.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import mname, policy_costs, two_way_boots  # noqa: E402

from invex.eval.stats import boot_p, holm  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402

ANALYSES = ("confirm_iex_test_iex", "confirm_iex_test_ppo_iex", "confirm_iex_test_ppo_iexlearner",
            "confirm_iex_test_v8_iexlm", "confirm_iex_test_iexpop_popart")
DROP = "NHP"


def family(d: dict, cells: pd.DataFrame, drop: str | None) -> list[dict]:
    spec = json.loads((REPO / d["provenance"]["args"]["spec"]).read_text())
    out = []
    for h in [x for x in d["hypotheses"] if "p" in x]:
        c = cells[cells["hypothesis"] == h["id"]]
        if drop:
            c = c[c["stock"] != drop]
        r = two_way_boots(c.set_index(["stock", "day"])["mean"], d["provenance"]["args"]["n_boot"], 0.95)
        delta = h.get("delta", spec.get("delta", 0.0))
        out.append({"id": h["id"], "mean": r["mean"], "lo95": r["lo"], "hi95": r["hi"],
                    "p": boot_p(r.pop("boots"), h["test"], delta), "stored_p": h["p"], "stored_mean": h["estimates"]["all"]["mean"]})
    for x, padj in zip(out, holm([x["p"] for x in out])):
        x["p_holm"] = float(padj)
        x["supported_025"], x["supported_05"] = bool(padj < 0.025), bool(padj < 0.05)
    return out


_PER: dict = {}


def cells_from_evals(d: dict) -> pd.DataFrame:
    """Per hypothesis, stock and day, the mean paired difference, prepared as confirm.py does from the stored evaluation
    (exclusions, unit, seeds averaged per episode); the one-tick population is the whole table (checked)."""
    a = d["provenance"]["args"]
    fam = json.loads((REPO / a["spec"]).read_text())["families"][a["family"]]
    assert "cap" not in fam and fam.get("max_onetick") is None and fam.get("min_onetick") == 0.5, fam
    key = (a["eval"], fam["unit"], a.get("seen_tag", "seen"))
    if key not in _PER:
        res = pd.read_parquet(RESULTS / "transfer" / f"eval_{a['eval']}.parquet",
                              columns=["key", "policy", "stock", "day", fam["unit"], "excluded"])
        if res["excluded"].any():
            res = res[~res["excluded"]]
        _PER[key] = policy_costs(res, fam["unit"], key[2])
    per = _PER[key]
    assert per.shape[0] == d["episodes"], (per.shape[0], d["episodes"])
    out = []
    for h in [x for x in d["hypotheses"] if "p" in x]:
        g = (per[h["a"]] - per[h["b"]]).dropna().groupby(level=["stock", "day"])
        out.append(g.mean().rename("mean").reset_index().assign(hypothesis=h["id"]))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO / "supplement" / "registered_tests")
    ap.add_argument("--from-evals", action="store_true")
    a = ap.parse_args()
    cells = None if a.from_evals else pd.read_parquet(a.dir / "cells.parquet")
    res, m = {}, {}
    for stem in ANALYSES:
        if a.from_evals:
            d = json.loads((RESULTS / "transfer" / f"{stem}.json").read_text())
            c = cells_from_evals(d)
        else:
            d = json.loads((a.dir / "analyses" / f"{stem}.json").read_text())
            c = cells[cells["analysis"] == stem]
        keep = family(d, c, None)  # must reproduce the stored analysis
        for x in keep:
            assert abs(x["mean"] - x["stored_mean"]) < 1e-12 and x["p"] == x["stored_p"], (stem, x)
        res[stem] = family(d, c, DROP)
        for x in res[stem]:
            k = mname("IexNoNhp", x["id"])
            m[k], m[k + "Lo"], m[k + "Hi"] = f"{x['mean']:+.3f}", f"{x['lo95']:+.3f}", f"{x['hi95']:+.3f}"
            m[k + "Pholm"] = f"{x['p_holm']:.3f}" if x["p_holm"] >= 0.001 else "<0.001"
            m[k + "Held"] = ("holds at 2.5\\%" if x["supported_025"] else "holds at 5\\% only" if x["supported_05"]
                             else "does not hold")
            print(f"{x['id']:3s} {x['mean']:+.4f} [{x['lo95']:+.4f}, {x['hi95']:+.4f}] Holm p {x['p_holm']:.4f} "
                  f"2.5%: {x['supported_025']} 5%: {x['supported_05']}")
    (RESULTS / "transfer" / "iex_without_nhp.json").write_text(json.dumps(res, indent=1))
    lines = ["% generated by experiments/iex_without_nhp.py (post hoc) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_iex_without_nhp.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

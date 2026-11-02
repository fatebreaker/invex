"""Join evaluations of the same episode table into one, for a family whose policies were scored by different scripts
(iex-rep-v1: the context encoder is scored by experiments/ctx_fqi.py, every other policy by experiments/transfer.py).

    python scripts/merge_evals.py --out iex_rep_wide iex_rep_all iex_rep_all_ctx

Every part must name the same episode table. The baselines both scripts score (crossing immediately, posting and
waiting) must agree to 1e-9 in cost on every episode; the first part's rows are kept. Writes
results/transfer/eval_<out>.{parquet,json}.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from invex.paths import RESULTS
from invex.provenance import provenance, write_json

COLS = ["key", "policy", "stock", "day", "cost", "c_spread", "c_bps", "shortfall", "excluded"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args()
    T = RESULTS / "transfer"
    names = {json.loads((T / f"eval_{p}.json").read_text())["episodes"] for p in a.parts}
    if len(names) != 1:
        raise SystemExit(f"the parts score different episode tables: {sorted(names)}")
    frames = [pd.read_parquet(T / f"eval_{p}.parquet", columns=COLS) for p in a.parts]
    first = frames[0]
    for p, f in zip(a.parts[1:], frames[1:]):
        both = first.merge(f, on=["key", "policy"], suffixes=("", "_other"))
        if not np.allclose(both["cost"], both["cost_other"], rtol=0, atol=1e-9, equal_nan=True):
            raise SystemExit(f"{p}: the policies it shares with {a.parts[0]} differ in cost")
        print(f"{p}: {both['policy'].nunique()} shared policies agree on {len(both):,} rows")
    res = pd.concat(frames, ignore_index=True).drop_duplicates(["key", "policy"], keep="first")
    res.to_parquet(T / f"eval_{a.out}.parquet", index=False)
    write_json(T / f"eval_{a.out}.json", {"episodes": names.pop(), "parts": a.parts, "rows": len(res),
                                          "policies": sorted(res["policy"].unique()), "provenance": provenance(args=vars(a))})
    print(f"eval_{a.out}: {len(res):,} rows, {res['policy'].nunique()} policies")


if __name__ == "__main__":
    main()

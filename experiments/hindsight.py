"""The best cost attainable in hindsight on every frozen episode: the cheapest sequence of actions (HOLD, JOIN,
IMPROVE, CANCEL, CROSS at the K decision times), found by exact dynamic programming on the episode's own replayed
path (invex.env.execution.hindsight). No policy with these actions and decision times can do better, so the gap
between a policy and this bound is an upper bound on what better learning could still save. Gross of fees, and,
on Nasdaq, separately optimised net of the taker fee and maker rebate. Exploratory, post hoc (fourth review round).

    python experiments/hindsight.py --episodes val_v0
    python experiments/hindsight.py --episodes stage1_v0 --unseal --reason "..."

Writes results/transfer/hindsight_<episodes>.parquet (key, hs_gross, hs_net, states, overflow) and a .json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer as TR  # noqa: E402

from invex.env import execution as ex  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval.freeze import book_of  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402
from invex.sim.tracker import EXACT  # noqa: E402
from invex.venues import venue as venue_of  # noqa: E402


def task(args):
    day, stock, eps, max_states = args
    venue = str(eps["venue"].iloc[0])
    book, _o, _c, tick = book_of(venue, day, stock)
    eps = eps.reset_index(drop=True)
    arrs = book.arrays()
    sg, t0, T = eps["side"].to_numpy(np.int64), eps["t0"].to_numpy(np.int64), eps["T_ns"].to_numpy(np.int64)
    Q, mid0 = eps["Q"].to_numpy(np.int64), eps["mid0"].to_numpy(float)
    lat = venue_of(venue).latency_ns
    n = len(eps)
    out = {"key": eps["key"].to_numpy()}
    fees = {"gross": (0.0, 0.0), "net": TR.NASDAQ_FEES if venue == "itch" else None}
    states, overflow = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for name, f in fees.items():
        best = np.full(n, np.nan)
        if f is not None:
            ns, ov = np.zeros(n, np.int64), np.zeros(n, np.int64)
            ex.hindsight(*arrs, sg, mid0, t0, T, Q, TR.K, lat, tick, EXACT, f[0], f[1], max_states, best, ns, ov)
            states, overflow = np.maximum(states, ns), np.maximum(overflow, ov)
        out["hs_" + name] = best
    out["states"], out["overflow"] = states, overflow.astype(bool)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--unseal", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-states", type=int, default=1024)
    a = ap.parse_args()
    eps = freeze.load(a.episodes, unseal=a.unseal, reason=a.reason)
    jobs = [(d, s, g, a.max_states) for (_v, d, s), g in eps.groupby(["venue", "day", "stock"])]
    t0 = time.time()
    res = pd.concat(TR.pool(task, jobs, a.workers), ignore_index=True)
    out = TR.OUT / f"hindsight_{a.episodes}.parquet"
    res.to_parquet(out, index=False)
    write_json(out.with_suffix(".json"), {"episodes": a.episodes, "rows": len(res), "max_states": a.max_states,
                                          "overflow": int(res["overflow"].sum()), "states_max": int(res["states"].max()),
                                          "states_median": float(res["states"].median()),
                                          "seconds": round(time.time() - t0), "provenance": provenance(args=vars(a))})
    print(f"{out.name}: {len(res):,} episodes, overflow {int(res['overflow'].sum())}, states median "
          f"{res['states'].median():.0f} max {res['states'].max()}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

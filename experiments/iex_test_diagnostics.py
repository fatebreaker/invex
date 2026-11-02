"""Post hoc diagnostics of the prospective IEX test (TMLR review, round 1): why the stock-weighted and the
episode-weighted estimates of the fitted-Q family (I1-I3) differ on the test days.

    python experiments/iex_test_diagnostics.py [--eval iex_test] [--top 15]

Reads the stored registered evaluation and loads the sealed episode table once through the logged loader (reason
"post hoc ..."), so the read appears in prereg/access_log.jsonl. Population and estimators are those of the
registered analysis (experiments/confirm.py, experiments/confirm_robust.py): one-tick books (w_onetick >= 0.5),
spreads per share, seeds averaged; (a) every stock weighted equally and each stock's days equally, (b) every episode
weighted equally. Reports, for I1-I3 (and P1-P3 for comparison): each estimand by day, leave one day out, the
stock-days with the largest influence on (a), and the episodes behind them (each policy's cost, how many shares
the crossing rule priced beyond the visible depth, the spread, the order size). Writes
results/transfer/iex_test_diagnostics.json and prints a report. Decides nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import policy_costs  # noqa: E402  (the registered per-episode cost table)

from invex.eval import freeze  # noqa: E402
from invex.provenance import write_json  # noqa: E402
from invex.paths import RESULTS  # noqa: E402

TESTS = {"I1": ("zero:R3tick", "base:immediate"), "I2": ("zero:R3tick", "iexseen:R3tick"),
         "I3": ("zero:R0", "zero:R3tick")}
TESTS_PPO = {"P1": ("zeroppo:R3tick", "base:immediate"), "P2": ("zeroppo:R3tick", "iexseen:R3tick"),
             "P3": ("zeroppo:R0", "zeroppo:R3tick")}


def by_stock(diff: pd.Series) -> float:
    """Estimand (a): mean over stocks of the mean over the stock's days of its day means."""
    cm = diff.groupby(level=["stock", "day"]).mean()
    return float(cm.groupby(level="stock").mean().mean())


def by_episode(diff: pd.Series) -> float:
    return float(diff.mean())


def influence(diff: pd.Series) -> pd.DataFrame:
    """Each stock-day's share of estimand (a): day mean / (days of the stock) / (stocks)."""
    cm = diff.groupby(level=["stock", "day"]).agg(["mean", "size", "min", "max"])
    days = cm.groupby(level="stock").size()
    n = len(days)
    cm["days_of_stock"] = days.reindex(cm.index.get_level_values("stock")).to_numpy()
    cm["contribution"] = cm["mean"] / cm["days_of_stock"] / n
    cm["episode_contribution"] = cm["mean"] * cm["size"] / len(diff)
    return cm.sort_values("contribution", key=np.abs, ascending=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="iex_test")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--ppo-suffix", default="_ppo", help="the PPO evaluation's suffix ('' to skip it)")
    a = ap.parse_args(argv)
    meta = json.loads((RESULTS / "transfer" / f"eval_{a.eval}.json").read_text())
    sealed = json.loads((freeze.REPO / "prereg" / f"episodes_{meta['episodes']}.json").read_text())["sealed"]
    eps = freeze.load(meta["episodes"], unseal=sealed,
                      reason=f"post hoc (review round 1): day, leave-one-day-out and stock-day influence diagnostics "
                             f"of the stored {a.eval} evaluation")
    keep = set(eps.loc[eps["w_onetick"] >= 0.5, "key"])
    out = {"eval": a.eval, "population": "w_onetick >= 0.5", "tests": {}}
    for ev, tests in ((a.eval, TESTS),) + (((a.eval + a.ppo_suffix, TESTS_PPO),) if a.ppo_suffix else ()):
        res = pd.read_parquet(RESULTS / "transfer" / f"eval_{ev}.parquet")
        res = res[~res["excluded"] & res["key"].isin(keep)]
        per = policy_costs(res, "c_spread", "seen")
        detail = res.set_index("key")
        days = sorted(per.index.get_level_values("day").unique())
        for hid, (pa, pb) in tests.items():
            diff = (per[pa] - per[pb]).dropna()
            r = {"a": pa, "b": pb, "by_stock": by_stock(diff), "by_episode": by_episode(diff),
                 "stocks": int(diff.index.get_level_values("stock").nunique()), "episodes": int(len(diff))}
            dl = diff.index.get_level_values("day")
            r["by_day"] = {d: {"by_stock": by_stock(diff[dl == d]), "by_episode": by_episode(diff[dl == d]),
                               "stocks": int(diff[dl == d].index.get_level_values("stock").nunique()),
                               "episodes": int((dl == d).sum())} for d in days}
            r["leave_one_day_out"] = {d: {"by_stock": by_stock(diff[dl != d]), "by_episode": by_episode(diff[dl != d])}
                                      for d in days}
            inf = influence(diff)
            top = inf.head(a.top)
            r["top_stock_days"] = []
            for (stock, day), row in top.iterrows():
                d = diff.xs((stock, day), level=("stock", "day"))
                worst = d.abs().sort_values(ascending=False).head(3).index.get_level_values("key")
                eps_rows = []
                for k in worst:
                    rows = detail.loc[[k]]
                    pol = {}
                    for p in (pa, pb):
                        tag, rep = p.split(":")
                        sel = rows["policy"] == rep if tag == "base" else rows["policy"].str.startswith(f"{tag}_{rep}_s")
                        pol[p] = rows.loc[sel, ["policy", "c_spread", "shortfall", "passive", "filled"]].to_dict("records")
                    first = rows.iloc[0]
                    eps_rows.append({"key": k, "diff": float(d.loc[k]),
                                     "w_spread_ticks": float(first["w_spread"] / first["tick"]),
                                     "Q": int(first["Q"]), "family": first["family"], "side": int(first["side"]),
                                     "w_trades": float(first["w_trades"]), "policies": pol})
                r["top_stock_days"].append({"stock": stock, "day": day, "mean": float(row["mean"]),
                                            "episodes": int(row["size"]), "days_of_stock": int(row["days_of_stock"]),
                                            "contribution": float(row["contribution"]),
                                            "episode_contribution": float(row["episode_contribution"]),
                                            "min": float(row["min"]), "max": float(row["max"]), "worst": eps_rows})
            r["top_share_of_by_stock"] = float(top["contribution"].sum() / r["by_stock"]) if r["by_stock"] else None
            out["tests"][hid] = r
        # how often a cross ran out of visible depth, by policy (the shortfall rule of src/invex/env/execution.py)
        sf = res.assign(any_short=res["shortfall"] > 0).groupby("policy")["any_short"].mean()
        out.setdefault("shortfall_share", {}).update({p: float(v) for p, v in sf.items()})
    write_json(RESULTS / "transfer" / f"{a.eval}_diagnostics.json", out)
    for hid, r in out["tests"].items():
        print(f"\n## {hid}: {r['a']} - {r['b']}: by stock {r['by_stock']:+.4f}, by episode {r['by_episode']:+.4f} "
              f"({r['stocks']} stocks, {r['episodes']:,} episodes)")
        print("day        by stock  by episode  stocks  | leave out: by stock  by episode")
        for d, v in r["by_day"].items():
            lo = r["leave_one_day_out"][d]
            print(f"{d}  {v['by_stock']:+.4f}   {v['by_episode']:+.4f}   {v['stocks']:5d}   |   "
                  f"{lo['by_stock']:+.4f}   {lo['by_episode']:+.4f}")
        print(f"top {a.top} stock-days carry {r['top_share_of_by_stock']:.0%} of the stock-weighted estimate:")
        for t in r["top_stock_days"][:8]:
            print(f"  {t['stock']:6s} {t['day']} mean {t['mean']:+9.3f} over {t['episodes']:3d} episodes, stock on "
                  f"{t['days_of_stock']} day(s): {t['contribution']:+.4f} of (a), {t['episode_contribution']:+.5f} "
                  f"of (b); worst diff {t['worst'][0]['diff']:+.2f} (spread {t['worst'][0]['w_spread_ticks']:.2f} "
                  f"ticks, Q {t['worst'][0]['Q']}, {t['worst'][0]['family']})")
    print("\nshare of episodes whose cross ran out of visible depth, by policy:")
    for p, v in sorted(out["shortfall_share"].items(), key=lambda x: -x[1])[:12]:
        print(f"  {p:24s} {v:.4%}")


if __name__ == "__main__":
    main()

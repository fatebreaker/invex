"""The equal-information arm on the IEX test days (review round 2; exploratory, post hoc).

    python experiments/info_iex.py

Paired cost differences in spreads per share, seeds averaged within an episode, between Nasdaq-trained policies of the
same learner: raw features plus the window statistics with the cost in spreads (R0info) against MS+tick (R3tick), and
raw units (R0) against MS+tick, for trees, the neural learner and PPO. Two populations, as registered for the IEX
test: the one-tick books (the whole table iex_test_v0, kept because w_onetick >= 0.5; checked by its size) and the
wide books of iex_test_all_v0 (w_onetick < 0.5; its sealed table is loaded once, logged, for that share). Two
estimands, as in iex-v8: every stock and each of its days weighted equally, and every episode weighted equally; the
registered two-way bootstrap over stocks and days (seed 0, 10,000 replicates), 95% intervals.
Writes results/transfer/info_iex.json and paper/numbers_info_iex.tex.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import two_way_boots  # noqa: E402

from invex.eval import freeze  # noqa: E402
from invex.eval.stats import pigeonhole_bootstrap  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402

NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<rep>[A-Za-z0-9]+)_s(?P<seed>\d+)$")
POPS = {"OneTick": ("iex_test", "iex_test_ppo", "iex_test_info"), "Wide": ("iex_test_wide", "iex_test_wide_info")}
CONTRASTS = [(f"{lab}{kind}", f"{tag}:{a}", f"{tag}:R3tick") for lab, tag in (("Trees", "zero"), ("Nn", "zeronn"),
             ("Ppo", "zeroppo")) for kind, a in (("InfoVsMstick", "R0info"), ("RawVsMstick", "R0"))]
ONE_TICK_EPISODES = 688_404  # the registered one-tick population (AnalysedIexTest)


def costs(evals, keep: set | None) -> pd.DataFrame:
    """Per-episode cost (spreads), seeds averaged, one column per '<tag>:<rep>', with the episode's stock and day."""
    cols, idx = {}, None
    for ev in evals:
        res = pd.read_parquet(RESULTS / "transfer" / f"eval_{ev}.parquet",
                              columns=["key", "policy", "stock", "day", "c_spread", "excluded"])
        res = res[~res["excluded"]]
        if keep is not None:
            res = res[res["key"].isin(keep)]
        m = res["policy"].str.extract(NAME)
        res = res.assign(col=(m["tag"] + ":" + m["rep"]).fillna("base:" + res["policy"]))
        for col, g in res.groupby("col"):
            if col not in cols:
                cols[col] = g.groupby("key")["c_spread"].mean()
        s = res.drop_duplicates("key").set_index("key")[["stock", "day"]]
        idx = s if idx is None else idx.combine_first(s)
    df = pd.DataFrame(cols)
    return df.join(idx)


def main() -> None:
    out, m = {}, {}
    for pop, evals in POPS.items():  # one-tick books first: they need no sealed read
        keep = None
        if pop == "Wide":
            eps = freeze.load("iex_test_all_v0", unseal=True,
                              reason="post hoc (review round 2): the wide-book population of the equal-information arm "
                                     "on the IEX test days (exploratory)")
            keep = set(eps.loc[eps["w_onetick"] < 0.5, "key"])
        df = costs(evals, keep)
        if pop == "OneTick":
            assert len(df) == ONE_TICK_EPISODES, len(df)
        out[pop] = {"episodes": int(len(df)), "stocks": int(df["stock"].nunique())}
        for label, a, b in CONTRASTS:
            if a not in df or b not in df:
                continue
            d = (df[a] - df[b]).dropna()
            diff = pd.Series(d.to_numpy(), index=pd.MultiIndex.from_arrays([df.loc[d.index, "stock"], df.loc[d.index, "day"]],
                                                                           names=["stock", "day"]))
            ra = two_way_boots(diff, 10_000, 0.95)
            g = diff.groupby(level=["stock", "day"])
            rb = pigeonhole_bootstrap(g.sum().unstack().fillna(0.0).to_numpy(), g.count().unstack().fillna(0.0).to_numpy(),
                                      n_boot=10_000, level=0.95)
            for tag, r in (("Stock", ra), ("Episode", rb)):
                out[pop][f"{label}{tag}"] = {k: float(r[k]) for k in ("mean", "lo", "hi")}
                k = f"InfoIex{pop}{label}{tag}"
                m[k], m[k + "Lo"], m[k + "Hi"] = f"{r['mean']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
                print(f"{pop:8s} {label:18s} {tag:8s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        for lab in ("Trees", "Nn", "Ppo"):  # share of the raw-unit loss the arm closes, episodes weighted equally
            i, r = out[pop].get(f"{lab}InfoVsMstickEpisode"), out[pop].get(f"{lab}RawVsMstickEpisode")
            if i and r and r["mean"] > 0:
                m[f"InfoIex{pop}{lab}ShareEpisode"] = f"{100 * (1 - i['mean'] / r['mean']):.0f}\\%"
    (RESULTS / "transfer" / "info_iex.json").write_text(json.dumps(out, indent=1))
    lines = ["% generated by experiments/info_iex.py (post hoc, exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_info_iex.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

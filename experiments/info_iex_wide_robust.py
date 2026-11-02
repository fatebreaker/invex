"""The equal-information arm on IEX's wide books, without the replay's least realistic costs (review round 4, R5 N3;
exploratory, post hoc).

    python experiments/info_iex_wide_robust.py

As experiments/info_iex.py for the wide books of iex_test_all_v0 (one more logged load, for the population), in two
readings that the registered wide-book families also report: (i) without every episode in which some evaluated policy's
cross ran beyond the visible book (shortfall > 0 for any policy of either wide-book evaluation), and (ii) with every
policy's cost capped at 31 spreads per share before seeds are averaged (as the registered iexwide_cap family).
RAW+INFO (R0info) and raw units (R0) against MS+tick (R3tick) for trees, the neural learner and PPO, under both
estimands (stocks and their days weighted equally; episodes weighted equally), 95% intervals from the registered
bootstraps. Writes results/transfer/info_iex_wide_robust.json and paper/numbers_info_iex_wide_robust.tex.
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
EVALS = ("iex_test_wide", "iex_test_wide_info")
CONTRASTS = [(f"{lab}{kind}", f"{tag}:{a}", f"{tag}:R3tick") for lab, tag in (("Trees", "zero"), ("Nn", "zeronn"),
             ("Ppo", "zeroppo")) for kind, a in (("InfoVsMstick", "R0info"), ("RawVsMstick", "R0"))]
CAP = 31.0
WIDE_EPISODES = 831_443  # the wide-book population of info_iex.py (InfoIexWide), checked below


def load(keep: set) -> pd.DataFrame:
    parts = []
    for ev in EVALS:
        r = pd.read_parquet(RESULTS / "transfer" / f"eval_{ev}.parquet",
                            columns=["key", "policy", "stock", "day", "c_spread", "shortfall", "excluded"])
        parts.append(r[~r["excluded"] & r["key"].isin(keep)])
    return pd.concat(parts, ignore_index=True)


def per_episode(res: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Per-episode cost, seeds averaged, one column per '<tag>:<rep>', with the episode's stock and day."""
    m = res["policy"].str.extract(NAME)
    res = res.assign(col=(m["tag"] + ":" + m["rep"]).fillna("base:" + res["policy"]))
    c = res["c_spread"].clip(upper=cap) if cap is not None else res["c_spread"]
    df = res.assign(c=c).groupby(["key", "col"])["c"].mean().unstack()
    idx = res.drop_duplicates("key").set_index("key")[["stock", "day"]]
    return df.join(idx)


def main() -> None:
    eps = freeze.load("iex_test_all_v0", unseal=True,
                      reason="post hoc (review round 4): the wide-book population of the equal-information arm, "
                             "without shortfall episodes and capped (R5 N3; exploratory)")
    keep = set(eps.loc[eps["w_onetick"] < 0.5, "key"])
    res = load(keep)
    short = set(res.loc[res["shortfall"] > 0, "key"])
    out, m = {"shortfall_episodes": len(short)}, {}
    for reading, (cap, drop) in {"Clean": (None, short), "Cap": (CAP, set())}.items():
        df = per_episode(res[~res["key"].isin(drop)], cap)
        if reading == "Cap":
            assert len(df) == WIDE_EPISODES, len(df)
        out[reading] = {"episodes": int(len(df)), "stocks": int(df["stock"].nunique())}
        for label, a, b in CONTRASTS:
            d = (df[a] - df[b]).dropna()
            diff = pd.Series(d.to_numpy(), index=pd.MultiIndex.from_arrays(
                [df.loc[d.index, "stock"], df.loc[d.index, "day"]], names=["stock", "day"]))
            ra = two_way_boots(diff, 10_000, 0.95)
            g = diff.groupby(level=["stock", "day"])
            rb = pigeonhole_bootstrap(g.sum().unstack().fillna(0.0).to_numpy(), g.count().unstack().fillna(0.0).to_numpy(),
                                      n_boot=10_000, level=0.95)
            for tag, r in (("Stock", ra), ("Episode", rb)):
                out[reading][f"{label}{tag}"] = {k: float(r[k]) for k in ("mean", "lo", "hi")}
                k = f"InfoWide{reading}{label}{tag}"
                m[k], m[k + "Lo"], m[k + "Hi"] = f"{r['mean']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
                print(f"{reading:5s} {label:18s} {tag:8s} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        for lab in ("Trees", "Nn", "Ppo"):
            i, r = out[reading].get(f"{lab}InfoVsMstickEpisode"), out[reading].get(f"{lab}RawVsMstickEpisode")
            if i and r and r["mean"] > 0:
                m[f"InfoWide{reading}{lab}ShareEpisode"] = f"{100 * (1 - i['mean'] / r['mean']):.0f}\\%"
    m["InfoWideShortfallEpisodes"] = f"{len(short):,}".replace(",", "{,}")
    m["InfoWideCleanEpisodes"] = f"{out['Clean']['episodes']:,}".replace(",", "{,}")
    (RESULTS / "transfer" / "info_iex_wide_robust.json").write_text(json.dumps(out, indent=1))
    lines = ["% generated by experiments/info_iex_wide_robust.py (post hoc, exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_info_iex_wide_robust.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

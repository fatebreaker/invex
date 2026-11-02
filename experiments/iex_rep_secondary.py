"""Pre-specified secondary analyses of the second IEX prospective test (iex-rep-v1, prereg/iex_replication_v1.md,
section 6). They are reported whatever they show and decide nothing.

    python experiments/iex_rep_secondary.py [--pop iex_rep] [--wide iex_rep_wide] [--prefix IexRepSec]

1. The cost distribution (as experiments/dist_compare.py for the Nasdaq stages): raw units against MS+tick for each
   learner, and the Nasdaq-trained against the IEX-trained MS+tick policy (fitted Q, trees), on the competitive and
   the wide books; differences in the median, the 1st and 99th percentiles, the mean of the worst 5% of episodes and
   the standard deviation, episodes weighted equally, each policy's seeds averaged; 95% intervals from a stock
   bootstrap (1,000 replicates).
2. Day by day: every registered hypothesis's episode-weighted estimate on each test day, and the number of days whose
   sign agrees with the pooled estimate.
3. The wide-book families D and A without the episodes in which some evaluated policy's cross ran beyond IEX's
   visible book (the replay's least realistic episodes), episode-weighted, the registered two-way bootstrap.
Each sealed table is loaded once (logged). Writes results/transfer/iex_rep_secondary.json and
paper/numbers_iex_rep_secondary.tex.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm  # noqa: E402
from confirm import mname, population  # noqa: E402
from confirm_rep import NAME, TAG  # noqa: E402
from confirm_robust import episode_boots  # noqa: E402
from dist_compare import stats_of  # noqa: E402

from invex.eval import freeze  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402

SPEC = REPO / "prereg" / "confirmatory_iex_rep_v1.json"
PAIRS = {"TreesRawMst": ("zero:R0", "zero:R3tick"), "NnRawMst": ("zeronn:R0", "zeronn:R3tick"),
         "PpoRawMst": ("zeroppo:R0", "zeroppo:R3tick"), "TreesZeroIex": ("zero:R3tick", "iexseen:R3tick")}
N_BOOT = 1000


def per_episode(ev: str, fam: dict, eps_cache: dict, keep_shortfall: bool = True) -> pd.DataFrame:
    meta = json.loads((RESULTS / "transfer" / f"eval_{ev}.json").read_text())
    name = meta["episodes"]
    if name not in eps_cache:
        sealed = json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"]
        eps_cache[name] = freeze.load(name, unseal=sealed, reason=f"secondary analysis of {ev} ({TAG})" if sealed else "")
    res = pd.read_parquet(RESULTS / "transfer" / f"eval_{ev}.parquet",
                          columns=["key", "policy", "stock", "day", "c_spread", "shortfall", "excluded"])
    res = res[~res["excluded"]]
    keep = population(eps_cache[name], fam)
    if keep is not None:
        res = res[res["key"].isin(keep)]
    if not keep_shortfall:
        res = res[~res["key"].isin(set(res.loc[res["shortfall"] > 0, "key"]))]
    return confirm.policy_costs(res, "c_spread", "seen")


def distribution(per: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> dict:
    df = per[[a, b]].dropna().reset_index()
    codes, uniq = pd.factorize(df["stock"])
    xa, xb = df[a].to_numpy(), df[b].to_numpy()
    point = stats_of(xa, xb, np.ones(len(df)))
    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        for k, v in stats_of(xa, xb, cnt[codes]).items():
            boots[k].append(v)
    return {k: {"est": point[k], "lo": float(np.quantile(boots[k], 0.025)), "hi": float(np.quantile(boots[k], 0.975))}
            for k in ("Med", "One", "Ninetynine", "Cvar", "Sd", "Mean")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", default="iex_rep")
    ap.add_argument("--wide", default="iex_rep_wide")
    ap.add_argument("--prefix", default="IexRepSec")
    ap.add_argument("--out", default="iex_rep_secondary")
    a = ap.parse_args()
    confirm.NAME = NAME
    spec = json.loads(SPEC.read_text())
    fams = spec["families"]
    evs = {"iex_rep": a.pop, "iex_rep_wide": a.wide}
    cache, out, m = {}, {"distribution": {}, "days": {}, "no_shortfall": {}}, {}
    pers = {"OneTick": per_episode(a.pop, fams["rep_fqi"], cache), "Wide": per_episode(a.wide, fams["rep_wide"], cache)}
    for i, (pop, per) in enumerate(pers.items()):  # 1. the cost distribution
        for j, (lab, (x, y)) in enumerate(PAIRS.items()):
            if x not in per or y not in per:
                continue
            r = distribution(per, x, y, np.random.default_rng([0, i, j]))
            out["distribution"][f"{pop}{lab}"] = r
            for k, v in r.items():
                s = mname(a.prefix, "Dist", pop, lab, k)
                m[s], m[s + "Lo"], m[s + "Hi"] = f"{v['est']:+.3f}", f"{v['lo']:+.3f}", f"{v['hi']:+.3f}"
            print(pop, lab, {k: f"{v['est']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for k, v in r.items()}, flush=True)
    for fname, fam in fams.items():  # 2. day by day, every registered hypothesis
        if fam.get("secondary"):
            continue
        per = pers["Wide" if "max_onetick" in fam else "OneTick"]
        for h in fam["hypotheses"]:
            d = (per[h["a"]] - per[h["b"]]).dropna()
            pooled = float(d.mean())
            by_day = d.groupby(level="day").mean()
            agree = int((np.sign(by_day) == np.sign(pooled)).sum())
            out["days"][h["id"]] = {"pooled": pooled, "by_day": {str(k): float(v) for k, v in by_day.items()},
                                    "agree": agree, "days": int(len(by_day))}
            m[mname(a.prefix, "Days", h["id"], "agree")] = f"{agree} of {len(by_day)}"
            print(f"{h['id']}: pooled {pooled:+.4f}, {agree} of {len(by_day)} days agree in sign", flush=True)
    per = per_episode(a.wide, fams["rep_wide"], cache, keep_shortfall=False)  # 3. without shortfall episodes
    out["no_shortfall"]["episodes"] = int(per.shape[0])
    m[mname(a.prefix, "Clean", "episodes")] = f"{per.shape[0]:,}".replace(",", "{,}")
    for fname in ("rep_infowide", "rep_alt"):
        for h in fams[fname]["hypotheses"]:
            e = episode_boots((per[h["a"]] - per[h["b"]]).dropna(), 10_000, 0.95)
            e.pop("boots")
            out["no_shortfall"][h["id"]] = e
            s = mname(a.prefix, "Clean", h["id"])
            m[s], m[s + "Lo"], m[s + "Hi"] = f"{e['mean']:+.3f}", f"{e['lo']:+.3f}", f"{e['hi']:+.3f}"
            print(f"{h['id']} without shortfall episodes: {e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]", flush=True)
    (RESULTS / "transfer" / f"{a.out}.json").write_text(json.dumps(out, indent=1))
    lines = [f"% generated by experiments/iex_rep_secondary.py ({TAG}, pre-specified secondary) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / f"numbers_{a.out}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

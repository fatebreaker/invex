"""What every registered test on sealed data reads of its paired differences, for the review supplement (post hoc;
no new analysis).

    python experiments/export_test_inputs.py              # every stored analysis of a sealed episode table
    python experiments/export_test_inputs.py --dry-run    # the analyses of unsealed (validation) tables only

For every stored analysis (results/transfer/confirm_<eval>_<family>.json, written by experiments/confirm.py or
experiments/confirm_robust.py), redoes the analysis's preparation as that script does (exclusions, population, unit,
cap, seeds averaged per episode, the seen policy's stocks) and keeps, per hypothesis, the paired differences reduced
to what the estimators read: per stock and day their count, sum and mean; per stock their mean over its episodes and
its relative-tick tercile. scripts/check_registered_tests.py recomputes every estimate, interval, p-value and Holm
decision from these alone, with the registered code, and compares them with the stored analyses. Each sealed episode
table is read once (logged). Writes DIR/cells.parquet, DIR/stocks.parquet and DIR/analyses/<stem>.json (the stored
analyses, without the working directory and host in their provenance); DIR is supplement/registered_tests, or a
scratch directory with --dry-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import policy_costs, population  # noqa: E402

from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.paths import DATA, REPO, RESULTS  # noqa: E402

REASON = "post hoc (review supplement): the inputs of the registered tests, for scripts/check_registered_tests.py"


def stored(sealed: bool) -> list[tuple[str, str, dict, dict, str]]:
    """(stem, script, args, analysis, episode table) of every stored analysis of a sealed (or unsealed) table."""
    out = []
    for f in sorted((RESULTS / "transfer").glob("confirm_*.json")):
        d = json.loads(f.read_text())
        args = d["provenance"]["args"]
        name = json.loads((RESULTS / "transfer" / f"eval_{args['eval']}.json").read_text())["episodes"]
        if json.loads((REPO / "prereg" / f"episodes_{name}.json").read_text())["sealed"] == sealed:
            out.append((f.stem, Path(d["provenance"]["argv"][0]).name, args, d, name))
    return out


def episodes(name: str, sealed: bool) -> pd.DataFrame:
    """The episode table with the units confirm.py adds (the per-trade volatility and the relative tick)."""
    eps = freeze.load(name, unseal=sealed, reason=REASON if sealed else "")
    uf = unit_frame(eps, load_splits()["ko_constants"])
    return eps.assign(sigma_trade=uf["_sigma_trade"].to_numpy(), eta=uf["_eta_sigma"].to_numpy())


def differences(script: str, args: dict, eps: pd.DataFrame, res: pd.DataFrame):
    """Per tested hypothesis of the analysis, the per-episode paired differences, prepared as its script does."""
    spec = json.loads((REPO / args["spec"]).read_text())
    fam = spec["families"][args["family"]]
    if "excluded" in res and res["excluded"].any():
        res = res[~res["excluded"]]
    keep = population(eps, fam)
    if keep is not None:
        res = res[res["key"].isin(keep)]
    unit = fam["unit"]
    if script == "confirm.py" and unit == "c_sigma":
        res = res.drop(columns=["sigma_trade"], errors="ignore").merge(eps[["key", "sigma_trade"]], on="key")
        res["c_sigma"] = res["cost"] / (res["Q"] * res["sigma_trade"])
    if "cap" in fam:
        res = res.assign(**{unit: res[unit].clip(upper=fam["cap"])})
    per = policy_costs(res, unit, args.get("seen_tag", "seen"))
    for h in fam["hypotheses"]:
        if h["a"] not in per or h["b"] not in per:
            continue
        diff = (per[h["a"]] - per[h["b"]]).dropna()
        if script == "confirm.py" and h.get("stocks") == "seen":
            seen = set(pd.read_parquet(DATA / "transitions" / args["seen_data"] / "eps.parquet", columns=["stock"])["stock"])
            diff = diff[diff.index.get_level_values("stock").isin(seen)]
        yield h["id"], diff


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(tempfile.mkdtemp(prefix="test_inputs_")) if a.dry_run else REPO / "supplement" / "registered_tests"
    (out / "analyses").mkdir(parents=True, exist_ok=True)
    todo = stored(sealed=not a.dry_run)
    cells, stocks, cache = [], [], {}
    for stem, script, args, d, name in sorted(todo, key=lambda t: (t[4], t[2]["eval"])):
        if name not in cache:
            cache = {name: episodes(name, sealed=not a.dry_run)}  # one table in memory at a time
        eps = cache[name]
        if cache.get("eval") != args["eval"]:
            cache["eval"] = args["eval"]
            cache["res"] = pd.read_parquet(RESULTS / "transfer" / f"eval_{args['eval']}.parquet")
        eta = eps.groupby("stock")["eta"].median()
        terc = pd.Series(np.digitize(np.log(eta.to_numpy()), np.log(load_splits()["eta_sigma_tercile_cuts"])),
                         index=eta.index)
        try:
            diffs = list(differences(script, args, eps, cache["res"]))
        except KeyError as e:  # an analysis written by an earlier version of its script, without that argument
            print(f"{stem}: skipped (no {e} in its arguments)", flush=True)
            continue
        for hid, diff in diffs:
            g = diff.groupby(level=["stock", "day"])
            c = pd.DataFrame({"n": g.count(), "sum": g.sum(), "mean": g.mean()}).reset_index()
            cells.append(c.assign(analysis=stem, hypothesis=hid))
            s = diff.groupby(level="stock").mean().rename("mean").reset_index()
            stocks.append(s.assign(tercile=terc.reindex(s["stock"]).to_numpy(), analysis=stem, hypothesis=hid))
        n_h = len(diffs)
        prov = {k: v for k, v in d["provenance"].items() if k not in ("cwd", "host")}
        (out / "analyses" / f"{stem}.json").write_text(json.dumps({**d, "provenance": prov}, indent=1))
        print(f"{stem}: {n_h} hypotheses", flush=True)
    lead = ["analysis", "hypothesis", "stock"]
    c = pd.concat(cells, ignore_index=True)
    c = c[lead + ["day", "n", "sum", "mean"]].astype({"n": "int32"})
    s = pd.concat(stocks, ignore_index=True)[lead + ["mean", "tercile"]].astype({"tercile": "int8"})
    c.to_parquet(out / "cells.parquet", index=False, compression="zstd")
    s.to_parquet(out / "stocks.parquet", index=False, compression="zstd")
    print(f"{out}: {len(todo)} analyses, {len(c):,} stock-day cells, {len(s):,} stock rows")


if __name__ == "__main__":
    main()

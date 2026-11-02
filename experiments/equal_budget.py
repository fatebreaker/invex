"""PPO against fitted Q-iteration with equal tuning budgets (review round 4, R5; exploratory, validation only).

    python experiments/equal_budget.py

Each learner has four configurations on the validation days: fitted Q-iteration with trees at its registered setting
and with 1,000 rounds, 255 leaves or both (tags zero, tunea, tuneb, tunec); PPO at its registered setting and with its
value targets scaled three ways (tags zeroppo, ppovslo, ppovs, ppovshi). Seed 0 of every configuration. For MS+tick
(R3tick) and raw units (R0), the best configuration of each learner is chosen by its mean validation cost (stocks
weighted equally), on the same days that score it, which flatters both learners alike. Reported: PPO's best minus
fitted Q-iteration's best in each representation, and raw minus MS+tick within each learner at its best, with 95%
intervals from the stock bootstrap stratified by relative-tick tercile. Reads only stored evaluations.
Review round 5 adds two checks (macros Budget*Seeds and Budget*Heldout): the chosen configurations scored with all
their seeds (three each, scripts/review5_budget_queue.sh), and the choice made on one validation day and scored on the
other (each direction, seeds averaged; a stock's contrast is its mean over the held-out days it trades on).
Writes results/transfer/equal_budget.json and paper/numbers_equal_budget.tex.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from invex.eval.stats import cluster_bootstrap
from invex.eval.universe import load_splits
from invex.paths import REPO, RESULTS

OUT = RESULTS / "transfer"
EVALS = ["val_v0", "val_v0_ppo", "val_v0_tune", "val_v0_tunec", "val_v0_ppotune", "val_v0_budget2"]
CONFIGS = {"fqi": ("zero", "tunea", "tuneb", "tunec"), "ppo": ("zeroppo", "ppovslo", "ppovs", "ppovshi")}
REPS = ("R3tick", "R0")


def main() -> None:
    want = {f"{t}_{r}_s0" for ts in CONFIGS.values() for t in ts for r in REPS}
    seeded = {f"{t}_{r}_s{s}" for ts in CONFIGS.values() for t in ts for r in REPS for s in (0, 1, 2)}
    parts = []
    for ev in EVALS:
        f = OUT / f"eval_{ev}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f, columns=["key", "policy", "stock", "day", "c_spread", "excluded"])
        parts.append(r[r["policy"].isin(seeded) & ~r["excluded"]])
    res_all = pd.concat(parts, ignore_index=True).drop_duplicates(["key", "policy"])
    res = res_all[res_all["policy"].isin(want)]
    per = res.pivot_table(index="stock", columns="policy", values="c_spread", aggfunc="mean")
    missing = sorted(want - set(per.columns))
    if missing:
        raise SystemExit(f"missing evaluations: {missing}")
    per = per.dropna()
    eta = pd.read_parquet(OUT / "eval_val_v0_tune.parquet", columns=["stock"]).drop_duplicates("stock")  # stocks only
    del eta
    cuts = np.log(load_splits()["eta_sigma_tercile_cuts"])
    # the relative tick per stock, as the validation evaluations store it
    info = pd.read_parquet(OUT / "eval_val_v0_info.parquet", columns=["stock", "eta"]).groupby("stock")["eta"].median()
    strata = np.digitize(np.log(info.reindex(per.index).to_numpy()), cuts)
    best, out, m = {}, {"configs": {}}, {}
    for lrn, tags in CONFIGS.items():
        for rep in REPS:
            costs = {t: float(per[f"{t}_{rep}_s0"].mean()) for t in tags}
            b = min(costs, key=costs.get)
            best[(lrn, rep)] = f"{b}_{rep}_s0"
            out["configs"][f"{lrn}:{rep}"] = {"costs": costs, "best": b}
            m[f"Budget{lrn.title()}{'MsTick' if rep == 'R3tick' else 'Raw'}Best"] = b
            print(f"{lrn} {rep}: " + ", ".join(f"{t} {c:.4f}" for t, c in costs.items()) + f" -> best {b}")

    def contrast(name: str, a: str, b: str) -> None:
        d = (per[a] - per[b]).to_numpy()
        r = cluster_bootstrap(d, strata, n_boot=10_000, level=0.95)
        out[name] = {k: float(r[k]) for k in ("mean", "lo", "hi")}
        m[name], m[name + "Lo"], m[name + "Hi"] = f"{r['mean']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
        print(f"{name}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")

    contrast("BudgetPpoVsFqiMsTick", best[("ppo", "R3tick")], best[("fqi", "R3tick")])
    contrast("BudgetPpoVsFqiRaw", best[("ppo", "R0")], best[("fqi", "R0")])
    contrast("BudgetFqiRawVsMsTick", best[("fqi", "R0")], best[("fqi", "R3tick")])
    contrast("BudgetPpoRawVsMsTick", best[("ppo", "R0")], best[("ppo", "R3tick")])
    contrast("BudgetPpoRawVsFqiMsTick", best[("ppo", "R0")], best[("fqi", "R3tick")])

    # round 5: the chosen configurations with all their seeds (seeds averaged within an episode)
    res_all = res_all.assign(arm=res_all["policy"].str.replace(r"_s\d+$", "", regex=True))
    seeds = res_all.groupby("arm")["policy"].nunique()
    by_ep = res_all.groupby(["key", "stock", "day", "arm"])["c_spread"].mean().unstack("arm")
    arm = {k: v.rsplit("_s", 1)[0] for k, v in best.items()}
    out["seeds"] = {a: int(seeds.get(a, 0)) for a in arm.values()}
    print("seeds per chosen configuration:", out["seeds"])
    per_s = by_ep.groupby(level="stock").mean()

    def contrast_on(name, frame, a, b):
        d = (frame[a] - frame[b]).dropna()
        st = np.digitize(np.log(info.reindex(d.index).to_numpy()), cuts)
        r = cluster_bootstrap(d.to_numpy(), st, n_boot=10_000, level=0.95)
        out[name] = {k: float(r[k]) for k in ("mean", "lo", "hi")} | {"stocks": int(len(d))}
        m[name], m[name + "Lo"], m[name + "Hi"] = f"{r['mean']:+.3f}", f"{r['lo']:+.3f}", f"{r['hi']:+.3f}"
        print(f"{name}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] ({len(d)} stocks)")

    pairs = [("PpoVsFqiMsTick", ("ppo", "R3tick"), ("fqi", "R3tick")), ("PpoVsFqiRaw", ("ppo", "R0"), ("fqi", "R0")),
             ("FqiRawVsMsTick", ("fqi", "R0"), ("fqi", "R3tick")), ("PpoRawVsMsTick", ("ppo", "R0"), ("ppo", "R3tick"))]
    for nm, a, b in pairs:
        contrast_on(f"Budget{nm}Seeds", per_s, arm[a], arm[b])

    # round 5: choose on one validation day (seed 0), score on the other (seeds averaged)
    s0 = res_all[res_all["policy"].str.endswith("_s0")]
    day_cost = s0.groupby(["day", "stock", "arm"])["c_spread"].mean().groupby(["day", "arm"]).mean()
    days = sorted(by_ep.index.get_level_values("day").unique())
    held, out["heldout_choice"] = [], {}
    for sel, score in ((days[0], days[1]), (days[1], days[0])):
        pick = {}
        for lrn, tags in CONFIGS.items():
            for rep in REPS:
                pick[(lrn, rep)] = min((f"{t}_{rep}" for t in tags), key=lambda a: day_cost.get((sel, a), np.inf))
        out["heldout_choice"][f"{sel}->{score}"] = {f"{k[0]}:{k[1]}": v for k, v in pick.items()}
        print(f"chosen on {sel}, scored on {score}:", pick)
        sc = by_ep[by_ep.index.get_level_values("day") == score].groupby(level="stock").mean()
        held.append(pd.DataFrame({nm: sc[pick[a]] - sc[pick[b]] for nm, a, b in pairs}))
    hd = pd.concat(held).groupby(level=0).mean()
    for nm, _a, _b in pairs:
        contrast_on(f"Budget{nm}Heldout", hd.assign(z=0.0), nm, "z")
    (OUT / "equal_budget.json").write_text(json.dumps(out, indent=1))
    lines = ["% generated by experiments/equal_budget.py (post hoc, exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (REPO / "paper" / "numbers_equal_budget.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

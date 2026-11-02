"""Training-data variation on validation: policies refit on stock-bootstrap resamples of the training split
(transfer.py fit --boot-stocks, tag zboot) against training seeds on the fixed training set (tag zero).

    python experiments/analyze_boot.py

For each registered stock contrast that the refits cover, the standard deviation of the contrast across the bootstrap
refits, across seeds, and the stock-bootstrap standard error of the seed-averaged contrast. Then the sealed Stage 1 and
Stage 2 hypothesis intervals widened by training-data variation (normal approximation), and whether each decision
survives, two ways: by the variance across refits in excess of the variance across seeds plus the seed variance of the
seed-averaged sealed estimate (a refit varies the seed as well as the sample), and, as an upper bound, by the full
variance across refits. Exploratory; writes results/transfer/boot_val_v0.json and paper/numbers_boot.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_revision as AR  # noqa: E402

from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

OUT = RESULTS / "transfer"
# (hypothesis, a, b, comparator): the registered stock contrasts the refits (R0, R0r, R1, R3, R3tick) cover. H1's
# comparator is the seen policy (tags seen and, refit on the same resamples of the training stocks, sboot)
CONTRASTS = [("H1", "R3tick", "R3tick", "seen"), ("H2", "R0", "R3", "zero"), ("H3", "R1", "R3tick", "zero"),
             ("H6", "R3tick", "R3", "zero"), ("H8", "R0r", "R3", "zero"), ("H9", "R0", "R0r", "zero")]
BOOT_TAG = {"zero": "zboot", "seen": "sboot"}
DELTA = 0.015


def per_stock(res: pd.DataFrame, policy: str) -> pd.Series:
    g = res[res["policy"] == policy]
    return g.groupby("stock")["c_spread"].mean()


def main() -> None:
    res, _ = AR.load("val_v0")
    files = sorted(OUT.glob("eval_val_v0_boot*.parquet")) + sorted(OUT.glob("eval_val_v0_sboot.parquet"))
    boot = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    boot = boot[boot["policy"].str.match(r"^[zs]boot_")]
    boot = boot[~boot["excluded"]]
    bseeds = sorted({int(p.rsplit("_s", 1)[1]) for p in boot["policy"].unique() if p.startswith("zboot_")})
    zseeds = sorted({int(p.rsplit("_s", 1)[1]) for p in res["policy"].unique() if p.startswith("zero_R3tick_s")})
    out = {"boot_seeds": bseeds, "zero_seeds": zseeds, "contrasts": {}, "provenance": provenance()}
    macros = {}
    have_b, have_z = set(boot["policy"]), set(res["policy"])
    for hid, a, b, tb in CONTRASTS:
        bt = BOOT_TAG[tb]
        bs = [s for s in bseeds if f"zboot_{a}_s{s}" in have_b and f"{bt}_{b}_s{s}" in have_b]
        if len(bs) < 2:
            continue
        vb = [float((per_stock(boot, f"zboot_{a}_s{s}") - per_stock(boot, f"{bt}_{b}_s{s}")).mean()) for s in bs]
        zs = [s for s in zseeds if f"zero_{a}_s{s}" in have_z and f"{tb}_{b}_s{s}" in have_z]
        vz = [float((per_stock(res, f"zero_{a}_s{s}") - per_stock(res, f"{tb}_{b}_s{s}")).mean()) for s in zs]
        d = pd.concat([per_stock(res, f"zero_{a}_s{s}") - per_stock(res, f"{tb}_{b}_s{s}") for s in zs], axis=1).mean(axis=1)
        se = cluster_bootstrap(d.dropna().to_numpy(), n_boot=4000)["se"]
        r = {"boot_seeds": bs, "seed_seeds": zs, "boot_values": vb, "boot_mean": float(np.mean(vb)), "boot_sd": float(np.std(vb, ddof=1)),
             "seed_values": vz, "seed_sd": float(np.std(vz, ddof=1)), "stock_se": float(se)}
        out["contrasts"][hid] = r
        H = AR.mname(hid)
        macros[f"BootVal{H}Sd"] = f"{r['boot_sd']:.4f}"
        macros[f"BootVal{H}SeedSd"] = f"{r['seed_sd']:.4f}"
        macros[f"BootVal{H}Se"] = f"{r['stock_se']:.4f}"
        macros[f"BootVal{H}Mean"] = f"{r['boot_mean']:+.3f}"
        macros[f"BootVal{H}SeedMean"] = f"{np.mean(vz):+.3f}"
        print(f"{hid} {a}-{b}: boot mean {r['boot_mean']:+.4f} sd {r['boot_sd']:.4f} | seed sd {r['seed_sd']:.4f} | "
              f"stock se {r['stock_se']:.4f}")
    # the sealed intervals widened by the validation training-data SD (normal approximation, 90%)
    spec = {h["id"]: h for h in json.loads((REPO / "prereg" / "confirmatory_v1.json").read_text())["families"]["stocks"]["hypotheses"]}
    for hid, r in out["contrasts"].items():
        r["train_sd"] = float(np.sqrt(max(0.0, r["boot_sd"] ** 2 - r["seed_sd"] ** 2)))
        macros[f"BootVal{AR.mname(hid)}TrainSd"] = f"{r['train_sd']:.4f}"
    out["widened"] = {}
    kept, kept_full, kept_scaled = [], [], []
    for stage, P in (("stage1", "StageOne"), ("stage2", "StageTwo")):
        rv = json.loads((OUT / f"revision_{stage}.json").read_text())["hypotheses"]
        for hid, r in out["contrasts"].items():
            h = rv.get(hid)
            if h is None:
                continue
            se, m, test = h["stock"]["se"], h["stock"]["mean"], spec[hid]["test"]
            seed_var = h.get("seed_sd", r["seed_sd"]) ** 2 / len(h["seeds_a"])
            # policy-by-stock variation averages over more stocks on a sealed stage: the training-sample variance
            # scaled by the ratio of the stage's seed variance to validation's (an assumption, not a measurement)
            ratio2 = (h.get("seed_sd", r["seed_sd"]) / r["seed_sd"]) ** 2
            for name, w, acc in (("", np.sqrt(se ** 2 + r["train_sd"] ** 2 + seed_var), kept),
                                 ("Full", np.sqrt(se ** 2 + r["boot_sd"] ** 2), kept_full),
                                 ("Scaled", np.sqrt(se ** 2 + ratio2 * r["train_sd"] ** 2 + seed_var), kept_scaled)):
                lo, hi = m - 1.645 * float(w), m + 1.645 * float(w)
                ok = {"greater": lo > 0, "less": hi < 0, "tost": -DELTA < lo and hi < DELTA}.get(test, None)
                out["widened"][f"{stage}:{hid}{':' + name.lower() if name else ''}"] = {"lo": lo, "hi": hi,
                                                                                      "test": test, "supported": ok}
                acc.append(bool(ok))
                print(f"  {stage} {hid} ({test}) {name or 'excess':6s}: {m:+.4f} widened [{lo:+.4f}, {hi:+.4f}] -> "
                      f"{'kept' if ok else 'LOST'}")
                H = f"Boot{P}{AR.mname(hid)}{name}Wide"
                macros[H + "Lo"], macros[H + "Hi"] = (f"{x:+.4f}" if abs(x) < 0.0005 else f"{x:+.3f}" for x in (lo, hi))
    macros["BootKept"], macros["BootKeptFull"], macros["BootChecked"] = str(sum(kept)), str(sum(kept_full)), str(len(kept))
    macros["BootKeptScaled"] = str(sum(kept_scaled))
    c = out["contrasts"]
    if "H2" in c and "H9" in c:  # the trees' cost-unit share of the raw-unit loss, refits against the fixed sample
        macros["BootValCostUnitShare"] = f"{100 * c['H9']['boot_mean'] / c['H2']['boot_mean']:.0f}\\%"
        macros["BootValCostUnitShareSeed"] = (f"{100 * np.mean(c['H9']['seed_values']) / np.mean(c['H2']['seed_values']):.0f}"
                                              "\\%")
    macros["BootRefits"] = str(len(bseeds))
    write_json(OUT / "boot_val_v0.json", out)
    with open(REPO / "paper" / "numbers_boot.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_boot.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()

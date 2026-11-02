"""Synthetic controls: transfer gaps and contrasts per representation (both learners), macros and figure.

    python experiments/analyze_controls.py

For each control (results/zi_controls{,_adv}/<control>/eval.parquet) and representation, per test asset:
the transfer gap TG (zero-shot minus seen, same representation) and the contrast with microstructure units
(zero-shot minus zero-shot R3), with cluster-bootstrap CIs over the test assets. For PC2 and PC3 also over the
third of test assets whose knob (log alpha, log delta) lies farthest from the middle of its range ("extremes").
Writes results/zi_controls_summary.json, paper/numbers_controls.tex and paper/figures/fig_controls.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from invex.eval.stats import cluster_bootstrap  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

CONTROLS = ("pc1", "pc1x", "pc2", "pc3")
LEARNERS = {"adv": "zi_controls_adv", "plain": "zi_controls"}
REPS = ("R0", "R0r", "R2", "R3", "R3d", "R3tick", "Rpi", "Rdim")
LABEL = {"R0": "raw", "R0r": "raw$^{s}$", "R2": "KO", "R3": "MS", "R3d": "MS$_d$", "R3tick": "MS+tick", "Rpi": "ratio", "Rdim": "dimless"}
KNOB = {"pc2": "alpha", "pc3": "delta"}
TITLE = {"pc1": "PC1\nexact similarity", "pc1x": "PC1x\nextrapolation", "pc2": "PC2\nrelative tick varies",
         "pc3": "PC3\ngranularity varies"}
WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
         "8": "Eight", "9": "Nine"}


def mname(*parts) -> str:
    s = "".join(str(p)[:1].upper() + str(p)[1:] for p in parts)
    return "".join(WORDS.get(c, c) for c in s if c.isalnum())


def analyse(path: Path, control: str, n_boot: int = 10_000) -> dict:
    res = pd.read_parquet(path)
    extra = Path(str(path).replace(f"{path.parent.parent.name}/", f"{path.parent.parent.name}_r0r/"))
    if extra.exists():  # the R0r arm, run separately on the same seeded assets and episodes
        x = pd.read_parquet(extra)
        res = pd.concat([res, x[x["policy"].str.endswith("_R0r")]], ignore_index=True)
    per = res.pivot_table(index=["asset", "key"], columns="policy", values="c_spread")
    asset = per.index.get_level_values("asset")
    info = res.groupby("asset")[["alpha", "delta"]].first()
    ext = None
    if control in KNOB:
        x = np.log(info[KNOB[control]])
        dist = (x - 0.5 * (x.min() + x.max())).abs()
        ext = set(dist[dist >= dist.quantile(2 / 3)].index)
    out = {"baselines": {b: float(per[b].mean()) for b in ("immediate", "post_then_cross") if b in per}, "reps": {}}
    for rep in REPS:
        z, s = f"zero_{rep}", f"seen_{rep}"
        if z not in per:
            continue
        row = {"cost_zero": float(per[z].mean())}
        measures = {"tg": per[z] - per[s]}
        if rep != "R3":
            measures["contrast"] = per[z] - per["zero_R3"]
        for k, d in measures.items():
            by = d.groupby(asset).mean()
            row[k] = cluster_bootstrap(by.to_numpy(), n_boot=n_boot)
            if ext is not None:
                row[k + "_ext"] = cluster_bootstrap(by[by.index.isin(ext)].to_numpy(), n_boot=n_boot)
        out["reps"][rep] = row
    return out


def figure(summary: dict, learner: str, stem: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(fs.FULL, 2.2), sharey=False)
    present = [rep for rep in REPS if all(rep in summary[learner][c]["reps"] for c in CONTROLS)]
    for ax, (letter, c) in zip(axes, zip("abcd", CONTROLS)):
        r = summary[learner][c]["reps"]
        xs = np.arange(len(present))
        for j, (k, lab, col, off, mk) in enumerate((("tg", "transfer gap: zero-shot $-$ seen", fs.CATEGORICAL[0], -0.17, "o"),
                                                    ("contrast", "zero-shot: representation $-$ MS", fs.CATEGORICAL[1], 0.17, "D"))):
            for i, rep in enumerate(present):
                if rep not in r or k not in r[rep]:
                    continue
                ci = r[rep][k]
                ax.plot([i + off, i + off], [ci["lo"], ci["hi"]], color=col, lw=1.2, solid_capstyle="round")
                ax.plot(i + off, ci["mean"], mk, color=col, ms=3.6 if mk == "o" else 3.0, mec=fs.SURFACE, mew=0.6,
                        label=lab if i == 0 else None)
        ax.axhline(0, color=fs.INK2, lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[x] for x in present], rotation=60, ha="right")
        ax.grid(axis="x", visible=False)
        fs.panel_title(ax, letter, TITLE[c])
        if letter == "a":
            ax.set_ylabel("cost difference (spreads)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=0.6)
    fs.save(fig, stem)
    plt.close(fig)


def main() -> None:
    summary = {}
    for learner, d in LEARNERS.items():
        summary[learner] = {}
        for c in CONTROLS:
            p = RESULTS / d / c / "eval.parquet"
            if p.exists():
                summary[learner][c] = analyse(p, c)
    summary["provenance"] = provenance()
    write_json(RESULTS / "zi_controls_summary.json", summary)
    macros = {}
    for learner in LEARNERS:
        for c, s in summary[learner].items():
            for rep, row in s["reps"].items():
                for k in ("tg", "contrast", "tg_ext", "contrast_ext"):
                    if k in row:
                        stem = mname("Zi", learner, c.replace("pc", "Pc").replace("1x", "Onex"), k.replace("_ext", "Ext"), rep)
                        macros[stem] = f"{row[k]['mean']:+.3f}"
                        macros[stem + "Lo"] = f"{row[k]['lo']:+.3f}"
                        macros[stem + "Hi"] = f"{row[k]['hi']:+.3f}"
                macros[mname("Zi", learner, c.replace("pc", "Pc").replace("1x", "Onex"), "cost", rep)] = f"{row['cost_zero']:.3f}"
    with open(REPO / "paper" / "numbers_controls.tex", "w") as fh:
        fh.write("% generated by experiments/analyze_controls.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    (REPO / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    for learner in LEARNERS:
        if len(summary[learner]) == len(CONTROLS):
            figure(summary, learner, REPO / "paper" / "figures" / ("fig_controls" if learner == "adv" else "fig_controls_plain"))
    for learner in LEARNERS:
        for c, s in summary[learner].items():
            print(f"{learner:5s} {c:5s} " + "  ".join(
                f"{rep}: TG {row['tg']['mean']:+.3f}" + (f" C {row['contrast']['mean']:+.3f}" if 'contrast' in row else "")
                for rep, row in s["reps"].items()))


if __name__ == "__main__":
    main()

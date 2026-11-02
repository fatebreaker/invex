"""A crypto-trained reference from the evaluation period, cross-fitted by day (third review, question 2;
scripts/queue_crypto_crossfit.sh). The 34 days of crypto_test_v0 and crypto_rep_v0 fall in two folds of alternate
dates; MS+tick policies fitted on one fold (tags cbxa, cbxb) score only the other fold's days, so every day is scored
by a reference trained on the same period but not on that day. Exploratory, post hoc.

    python experiments/crypto_crossfit.py

Contrasts in volatility units per trade, with the two-way (pairs x days) bootstrap of experiments/confirm.py over the
34 days pooled: the stock-trained MS+tick policy against the new reference (non-inferiority at the registered C2
margin of 0.10), the new reference against the registered 2020-23 one (cbseen), and the new reference against
crossing immediately. The same contrasts in spreads per share (no registered margin; prefix CryptoXSpread). Writes
results/transfer/crypto_crossfit.json and paper/numbers_crypto_crossfit.tex (CryptoX).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirm as C  # noqa: E402

from invex.env.units import unit_frame  # noqa: E402
from invex.eval import freeze  # noqa: E402
from invex.eval.stats import boot_p  # noqa: E402
from invex.eval.universe import load_splits  # noqa: E402
from invex.paths import REPO, RESULTS  # noqa: E402
from invex.provenance import provenance, write_json  # noqa: E402

EVALS = ("crypto_test", "crypto_rep")
REASON = "exploratory, post hoc (third review): analysis of the cross-fitted 2024-26 crypto reference"
OUT = RESULTS / "transfer"
# (macro name, a, b, test, margin)
CONTRASTS = [("ZeroVsRef", "zero:R3tick", "cbx:R3tick", "noninferior", 0.10),
             ("RefVsOld", "cbx:R3tick", "cbseen:R3tick", "two-sided", 0.0),
             ("RefVsImmediate", "cbx:R3tick", "base:immediate", "less", 0.0),
             ("ZeroVsOld", "zero:R3tick", "cbseen:R3tick", "noninferior", 0.10)]


def folds() -> tuple[set, set]:
    days = sorted(d for ev in EVALS for d in json.loads((REPO / "prereg" / f"episodes_{ev}_v0.json").read_text())["days"])
    return set(days[0::2]), set(days[1::2])


def costs(ev: str, fold_a: set, fold_b: set, unit: str) -> pd.DataFrame:
    meta = json.loads((OUT / f"eval_{ev}.json").read_text())
    eps = freeze.load(meta["episodes"], unseal=True, reason=REASON)
    uf = unit_frame(eps, load_splits()["ko_constants"])
    eps = eps.assign(sigma_trade=uf["_sigma_trade"].to_numpy())
    res = pd.concat([pd.read_parquet(OUT / f"eval_{ev}.parquet"), pd.read_parquet(OUT / f"eval_{ev}_cbx.parquet")],
                    ignore_index=True)
    if "excluded" in res:
        res = res[~res["excluded"]]
    res = res.drop(columns=["sigma_trade"], errors="ignore").merge(eps[["key", "sigma_trade"]], on="key")
    res["c_sigma"] = res["cost"] / (res["Q"] * res["sigma_trade"])
    res = res[~res["policy"].str.match(r"^zero_.*_s[3-9]$")]  # the frozen prereg-v1 policies, as registered
    day = res["day"].astype(str)
    # each fold's reference scores only the other fold's days
    res = res[~((res["policy"].str.startswith("cbxa_") & day.isin(fold_a))
                | (res["policy"].str.startswith("cbxb_") & day.isin(fold_b)))]
    res["policy"] = res["policy"].str.replace(r"^cbx[ab]_", "cbx_", regex=True)
    return C.policy_costs(res, unit, "seen")


def main() -> None:
    fa, fb = folds()
    pers = {u: pd.concat([costs(ev, fa, fb, u) for ev in EVALS]) for u in ("c_sigma", "c_spread")}
    per = pers["c_sigma"]
    out = {"fold_a": sorted(fa), "fold_b": sorted(fb), "episodes": int(per.shape[0]),
           "days": int(per.index.get_level_values("day").nunique()), "contrasts": {}, "provenance": provenance()}
    macros = {"CryptoXDays": str(out["days"])}
    for (u, pre), (name, a, b, test, delta) in ((x, c) for x in (("c_sigma", ""), ("c_spread", "Spread"))
                                                 for c in CONTRASTS):
        per = pers[u]
        if u == "c_spread":
            test, delta = "two-sided", 0.0
        diff = (per[a] - per[b]).dropna()
        level = 0.90
        r = C.two_way_boots(diff, 10_000, level)
        boots = r.pop("boots")
        p = boot_p(boots, test, delta) if test != "two-sided" else None
        out["contrasts"][pre + name] = {"unit": u, "a": a, "b": b, "test": test, "delta": delta, "estimate": r, "p": p,
                                  "episodes": int(diff.shape[0])}
        H = f"CryptoX{pre}{name}"
        macros[H], macros[H + "Lo"], macros[H + "Hi"] = (f"{r[k]:+.3f}" for k in ("mean", "lo", "hi"))
        if p is not None:
            macros[H + "P"] = f"{p:.3f}"
        print(f"{u:8s} {name:15s} {a} - {b}: {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
              + (f" p({test}, {delta}) {p:.4f}" if p is not None else ""))
    write_json(OUT / "crypto_crossfit.json", out)
    with open(REPO / "paper" / "numbers_crypto_crossfit.tex", "w") as fh:
        fh.write("% generated by experiments/crypto_crossfit.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()

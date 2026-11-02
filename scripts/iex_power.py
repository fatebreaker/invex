"""The power of every registered IEX test on new days, at the retrospective variance (prereg/iex_outcomes_v7.md):
one-sided normal approximation with the retrospective estimate as the true effect and the registered two-way
bootstrap's standard error, at alpha 0.05 and at Holm's first step (alpha / family size).

    python scripts/iex_power.py   # -> results/transfer/iex_power_from_retro.json
"""

from __future__ import annotations

import json

from scipy.stats import norm

from invex.paths import RESULTS

FAMILIES = ("confirm_iex_retro_iex", "confirm_iex_retro_ppo_iex", "confirm_iex_retro_ppo_iexlearner",
            "confirm_iex_retro_wide_iexwide")


def main(alpha: float = 0.05) -> None:
    rows = []
    for f in FAMILIES:
        d = json.loads((RESULTS / "transfer" / f"{f}.json").read_text())
        hs = [h for h in d["hypotheses"] if "p" in h]
        for h in hs:
            e = h["estimates"]["all"]
            est, se, t, dl = e["mean"], e["se"], h["test"], h.get("delta", 0.0)
            x = {"greater": est / se, "less": -est / se, "noninferior": (dl - est) / se}[t]
            p = float(norm.cdf(x - norm.ppf(1 - alpha)))
            p1 = float(norm.cdf(x - norm.ppf(1 - alpha / len(hs))))
            rows.append({"family": f, "id": h["id"], "test": t, "est": est, "se": se, "power": p, "power_holm1": p1})
            print(f"{h['id']:3s} {t:12s} est {est:+.3f} se {se:.3f} power {p:.2f} ({p1:.2f})")
    (RESULTS / "transfer" / "iex_power_from_retro.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()

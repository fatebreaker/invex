"""Exact similarity end to end: an asset and its exact rescaling (shares x 10, prices x 5, same seed) must give
identical rule costs in microstructure units, so D-hat is 0; in raw units the costs scale with the price."""

import numpy as np
import pandas as pd

from invex.env.episodes import book_episodes
from invex.env.units import unit_frame
from invex.eval import policies as P
from invex.eval.dhat import dhat, probe_policies
from invex.sim.tracker import prepare
from invex.sim.zi import ZIParams, simulate

KO = {"c_t": 4.68, "c_q": 1.7e-4, "c_p": 0.028}


def run(prm, s_factor):
    ev, o, c = simulate(prm, 3 * 3600, warmup_s=1800, seed=13)
    book = prepare(ev, o, c)
    eps = book_episodes(book, "zi", "d0", "Z", o, c, 120, KO)
    eps = eps[eps["family"] == "MS"].reset_index(drop=True)  # MS sizes scale with the trade size
    eps["tick"] = prm.tick
    uf = unit_frame(eps, KO)
    res = P.evaluate_book(book, eps, uf, probe_policies("R3"), 10, 100_000, tick=prm.tick)
    res = res.merge(eps[["key", "Q", "w_spread", "t0", "cell"]], on="key")
    res["j"] = res["cost"] / (res["Q"] * res["w_spread"])  # R3 price unit = mean spread
    res["j_raw"] = res["cost"] / (res["Q"] * 100.0)
    return res


def test_exact_rescaling_gives_identical_rule_costs_and_zero_dhat():
    base = ZIParams(alpha=0.2)
    a = run(base, 1)
    b = run(base.scaled(c=1.0, s=10, p=5), 10)
    m = a.merge(b, on=["t0", "cell", "policy"], suffixes=("_a", "_b"))
    assert len(m) > 200
    assert np.allclose(m["j_a"], m["j_b"], rtol=1e-9, atol=1e-9)
    assert np.allclose(m["j_raw_b"], 5 * m["j_raw_a"], rtol=1e-6, atol=1e-6)  # raw units see the price scale
    costs = pd.concat([a.assign(stock="A"), b.assign(stock="B")])
    d = dhat(costs, {"A"})
    assert d["B"] < 1e-9
    d_raw = dhat(costs.assign(j=costs["j_raw"]), {"A"})
    assert d_raw["B"] > 0.1

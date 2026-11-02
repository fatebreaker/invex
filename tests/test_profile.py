"""The stock-profile comparator: per-episode fees, the R3tickp features and the leave-one-day-out profile."""

import importlib.util

import numpy as np
import pandas as pd

from invex.agents.fqi_adv import fee_cost
from invex.env import execution as ex
from invex.env.features import N_RAW, PROFILE_COLS, represent
from invex.paths import REPO

_spec = importlib.util.spec_from_file_location("stock_profile", REPO / "experiments" / "stock_profile.py")
SP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SP)


def test_fee_cost_takes_one_fee_per_episode():
    rng = np.random.default_rng(0)
    n = 6
    before = np.zeros((n, ex.AGENT_SIZE), np.int64)
    after = np.zeros((n, ex.N_ACT, ex.AGENT_SIZE), np.int64)
    after[..., ex.S_FILLED] = rng.integers(0, 100, (n, ex.N_ACT))
    after[..., ex.S_PASSIVE] = after[..., ex.S_FILLED] // 3
    taker = rng.random(n)
    got = fee_cost(before[:, None, :], after, taker, 0.2)
    for i in range(n):  # the same as charging each episode its own scalar fee
        assert np.allclose(got[i], fee_cost(before[i][None, :], after[i], float(taker[i]), 0.2))
    # one fee per episode on single states (the continuation's crossing cost)
    one = fee_cost(before, after[:, 0], taker, 0.0)
    assert np.allclose(one, taker * (after[:, 0, ex.S_FILLED] - after[:, 0, ex.S_PASSIVE]))


def test_r3tickp_appends_tick_context_and_profile():
    n = 4
    raw = np.ones((n, N_RAW))
    ctx = {"tick": np.full(n, 100.0), "sigma_trade": np.full(n, 50.0)}
    ctx.update({c: np.arange(n) + i for i, c in enumerate(PROFILE_COLS)})
    X, names = represent(raw, np.ones(n), np.ones(n), np.full(n, 200.0), "R3tickp", ctx=ctx)
    Xt, names_t = represent(raw, np.ones(n), np.ones(n), np.full(n, 200.0), "R3tick", ctx=ctx)
    assert names == names_t + list(PROFILE_COLS)
    assert np.allclose(X[:, : Xt.shape[1]], Xt)
    for i, c in enumerate(PROFILE_COLS):
        assert np.allclose(X[:, Xt.shape[1] + i], ctx[c])


def test_leave_one_day_out_profile_excludes_its_day():
    rng = np.random.default_rng(1)
    days = [f"2019-0{d}-01" for d in range(1, 6)]
    rows = []
    effect = rng.normal(0, 1.0, 40)  # stocks differ by far more than the day noise, so shrinkage is negligible
    for s in range(40):
        for d, day in enumerate(days):
            noise = rng.normal(0, 0.01) + (5.0 if (s == 0 and d == 2) else 0.0)  # one outlying day for stock S0
            for e in range(4):
                key = f"S{s}-{day}-{e}"
                rows.append((key, "cross_at_0", 1.0, 1.0, f"S{s}", day))
                for pol in ("post_then_cross", "cross_at_8", "improve_then_cross", "q_1_0.4"):
                    rows.append((key, pol, 1.0 + effect[s] + noise, 0.5, f"S{s}", day))
    probes = pd.DataFrame(rows, columns=["key", "policy", "j", "tf", "stock", "day"])
    stocks = {f"S{s}" for s in range(1, 40)}  # S0 is not a training stock, so its outlier does not move the prior
    full, lodo = SP.profile(probes, pd.Series(0.0, index=sorted(probes["stock"].unique())), stocks)
    lo = lodo.set_index(["stock", "day"]) if "stock" in lodo else lodo
    # the outlying day's own profile leaves it out (estimate and sampling variance); the other days' profiles include it
    # (pulled up, though shrunk hard, since the outlier also makes them noisy)
    assert abs(lo.loc[("S0", days[2]), "p_post"] - effect[0]) < 0.05
    assert lo.loc[("S0", days[0]), "p_post"] > lo.loc[("S0", days[2]), "p_post"] + 0.1
    # a surcharge phi moves every profile by -phi times the share not taken by crossing (here 0.5)
    full2, _ = SP.profile(probes, pd.Series(0.4, index=sorted(probes["stock"].unique())), stocks)
    assert np.allclose(full2["p_post_raw"] - full["p_post_raw"], -0.4 * 0.5)
    # a markout psi on passive fills moves every profile by +psi times the share filled while resting (here 0.5;
    # crossing at once fills nothing passively)
    zero = pd.Series(0.0, index=sorted(probes["stock"].unique()))
    full3, _ = SP.profile(probes, zero, stocks, zero + 0.4)
    assert np.allclose(full3["p_post_raw"] - full["p_post_raw"], 0.4 * 0.5)

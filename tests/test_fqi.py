"""Plumbing of the fitted Q learners: with a model that can memorise, the greedy action at the last
stage must be the action whose realised cost is lowest (actions, masks and columns line up)."""

import numpy as np
import pytest

from invex.agents import fqi, fqi_adv
from invex.env import execution as ex
from invex.env.episodes import book_episodes
from invex.env.units import unit_frame, valid_units
from invex.sim.tracker import EXACT, prepare
from invex.sim.zi import ZIParams, simulate

KO = {"c_t": 4.68, "c_q": 1.7e-4, "c_p": 0.028}
MEMORISE = dict(num_leaves=255, min_child_samples=1, min_data_in_bin=1, learning_rate=0.3, lambda_l2=0.0,
                bagging_fraction=1.0, bagging_freq=0, feature_fraction=1.0, min_data_per_group=1, cat_smooth=0.0,
                cat_l2=0.0, max_cat_to_onehot=8)


@pytest.fixture(scope="module")
def zi_data():
    ev, o, c = simulate(ZIParams(alpha=0.2), 4 * 3600, warmup_s=1800, seed=5)
    book = prepare(ev, o, c)
    eps = book_episodes(book, "zi", "d0", "Z", o, c, 80, KO)
    uf = unit_frame(eps, KO)
    eps = eps[valid_units(uf)].reset_index(drop=True)
    uf = unit_frame(eps, KO)
    return book, eps, uf


@pytest.mark.parametrize("mod", [fqi, fqi_adv])
def test_last_stage_greedy_matches_realised_best(zi_data, mod, monkeypatch):
    book, eps, uf = zi_data
    K = 4
    data = mod.generate_book(book, eps, K, 100_000, EXACT, 100, seed=1)
    monkeypatch.setattr(mod, "LGB_PARAMS", dict(mod.LGB_PARAMS, **MEMORISE))
    model = mod.fit(data, eps, uf, "R3", K, 1.0, n_rounds=400, threads=4, log=lambda s: None)
    k = K - 1
    X = fqi.rep_inputs(data["raw"][k], eps, uf, "R3", 1.0)
    q = model.q_values(k, X)
    m = data["masks"][k].astype(bool)
    q[~m] = np.inf
    realised = np.where(m, data["costs"][k], np.inf)
    distinct = np.sort(realised, axis=1)
    # a clear best action: better than the runner-up by 5% of a spread per share
    clear = (distinct[:, 1] - distinct[:, 0]) > 0.05 * (eps["Q"] * eps["w_spread"]).to_numpy()
    agree = np.argmin(q, axis=1) == np.argmin(realised, axis=1)
    assert clear.sum() > 20
    assert agree[clear].mean() > 0.9


def test_neural_learner_last_stage_and_roundtrip(zi_data, tmp_path):
    from invex.agents import fqi_nn

    book, eps, uf = zi_data
    K = 4
    data = fqi_adv.generate_book(book, eps, K, 100_000, EXACT, 100, seed=1)
    model = fqi_nn.fit(data, eps, uf, "R3", K, 1.0, seed=0, epochs=400, patience=400, batch=64, lr=3e-3,
                       log=lambda s: None)
    k = K - 1
    X = fqi.rep_inputs(data["raw"][k], eps, uf, "R3", 1.0)
    q = model.q_values(k, X)
    m = data["masks"][k].astype(bool)
    q[~m] = np.inf
    realised = np.where(m, data["costs"][k], np.inf)
    distinct = np.sort(realised, axis=1)
    clear = (distinct[:, 1] - distinct[:, 0]) > 0.05 * (eps["Q"] * eps["w_spread"]).to_numpy()
    agree = np.argmin(q, axis=1) == np.argmin(realised, axis=1)
    assert clear.sum() > 20
    assert agree[clear].mean() > 0.8
    path = tmp_path / "nn.pkl"
    model.save(path)
    again = fqi_adv.load_model(path)
    np.testing.assert_allclose(again.q_values(k, X), model.q_values(k, X), rtol=1e-6, atol=1e-9)


def test_learned_scales_and_identity(zi_data):
    from invex.agents import fqi_nn

    book, eps, uf = zi_data
    K = 3
    data = fqi_adv.generate_book(book, eps, K, 100_000, EXACT, 100, seed=1)
    # one stock: the PopArt-style unit is Q times one positive constant
    norm = fqi_nn.stock_scale(data, eps)
    ratio = norm / eps["Q"].to_numpy(float)
    assert np.all(ratio > 0) and np.allclose(ratio, ratio[0])
    # rescaled copy: raw-unit target cost / Q is the original times the price factor p
    both, e2 = fqi_nn.augment_scales(data, eps, seed=3)
    n = len(eps)
    assert len(e2) == 2 * n and both["raw"].shape[1] == 2 * n and both["raw_next"].shape[2] == 2 * n
    p = both["raw"][0, n:, 0] / np.where(data["raw"][0, :, 0] != 0, data["raw"][0, :, 0], np.nan)
    ok = np.isfinite(p)
    y0 = data["costs"][K - 1] / eps["Q"].to_numpy(float)[:, None]
    y1 = both["costs"][K - 1][n:] / e2["Q"].to_numpy(float)[n:, None]
    np.testing.assert_allclose(y1[ok], (y0 * p[:, None])[ok], rtol=1e-4, atol=1e-6)
    # identity column: the stock's index in the frozen universe ("Z" is a listed ticker), -1 outside it
    X = fqi.rep_inputs(data["raw"][0], eps, uf, "R3tickid", 1.0)
    assert X.shape[1] == fqi.rep_inputs(data["raw"][0], eps, uf, "R3tick", 1.0).shape[1] + 1
    assert np.all(X[:, -1] >= 0) and len(np.unique(X[:, -1])) == 1
    X = fqi.rep_inputs(data["raw"][0], eps.assign(stock="NOT A STOCK"), uf, "R3tickid", 1.0)
    assert np.all(X[:, -1] == -1)
    # MS+tick plus descriptors: the MS+tick columns, then the four raw descriptors
    Xd = fqi.rep_inputs(data["raw"][0], eps, uf, "R3tickdesc", 1.0)
    Xt = fqi.rep_inputs(data["raw"][0], eps, uf, "R3tick", 1.0)
    assert Xd.shape[1] == Xt.shape[1] + 4
    np.testing.assert_array_equal(Xd[:, :Xt.shape[1]], Xt)


def test_passive_fills_are_counted(zi_data):
    from invex.eval import policies as P

    book, eps, uf = zi_data
    res = P.evaluate_book(book, eps, uf, [P.Immediate(), P.PostThenCross()], 4, 100_000, tick=100)
    imm, post = res[res["policy"] == "immediate"], res[res["policy"] == "post_then_cross"]
    assert (imm["passive"] == 0).all()
    assert ((post["passive"] >= 0) & (post["passive"] <= post["filled"])).all()
    assert (post["passive"] > 0).mean() > 0.2


def test_fee_aware_transitions(zi_data):
    """Net-of-fee transitions: the same states and path; crossing everything pays the taker fee on every share, and
    any cost changes by at most the fee on the shares the step could fill."""
    book, eps, uf = zi_data
    K, taker, rebate = 4, 30.0, 20.0
    gross = fqi_adv.generate_book(book, eps, K, 100_000, EXACT, 100, seed=1)
    net = fqi_adv.generate_book(book, eps, K, 100_000, EXACT, 100, seed=1, taker=taker, rebate=rebate)
    for key in ("raw", "masks", "masks_next", "done_next"):
        np.testing.assert_array_equal(np.nan_to_num(gross[key]), np.nan_to_num(net[key]))
    Q = eps["Q"].to_numpy(float)
    # stage 0, CROSS from the initial state: every share is taken
    np.testing.assert_allclose(net["costs"][0][:, ex.CROSS] - gross["costs"][0][:, ex.CROSS], taker * Q, rtol=1e-5)
    d = net["costs"] - gross["costs"]
    assert (d <= taker * Q[None, :, None] + 1e-3).all() and (d >= -rebate * Q[None, :, None] - 1e-3).all()
    # resting orders that fill earn the rebate somewhere
    assert (d < -1e-6).any()

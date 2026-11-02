"""The make-or-take environment (invex.env.execution) and its features, on the toy market."""

import numpy as np
import pytest

from invex.env import execution as ex
from invex.env.features import F_IMB, F_R_BEHIND, F_R_IN, F_R_NEAR, F_SPREAD, FlowBook, raw_features
from invex.sim import tracker as tk
from invex.sim.probe import mid_at, rest_then_cross_paths
from invex.sim.toy import TICK, toy_stream

from test_tracker import FAR, build_table

K = 10


@pytest.fixture(scope="module")
def market():
    msgs = toy_stream(21, n=3000)
    book = tk.prepare(build_table(msgs, msgs[0][1]), 0, FAR)
    rng = np.random.default_rng(3)
    n = 60
    t0 = np.sort(rng.integers(book.ts[100], book.ts[-400], n))
    sides = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int64)
    Q = np.full(n, 300, np.int64)
    T = np.full(n, int(3e8), np.int64)
    return book, t0, sides, Q, T


def run_policy(book, t0, sides, Q, T, actions_fn, latency):
    st = ex.new_states(Q)
    mid0 = mid_at(book, t0)
    total = np.zeros(len(t0))
    for k in range(K):
        acts = np.array([actions_fn(e, k) for e in range(len(t0))], dtype=np.int64)
        cost = np.zeros(len(t0))
        ex.rollout_stage(*book.arrays(), st, sides, mid0, t0, T, k, K, acts, latency, TICK, tk.EXACT, cost)
        total += cost
    return st, total, mid0


def test_join_then_hold_equals_post_then_cross_probe(market):
    book, t0, sides, Q, T = market
    st, total, mid0 = run_policy(book, t0, sides, Q, T, lambda e, k: ex.JOIN if k == 0 else ex.HOLD, latency=0)
    pth = rest_then_cross_paths(book, sides, Q, t0, T, K, latency=0)
    want = sides * ((pth["cash"][:, K] + pth["xcash"][:, K]) - mid0 * Q)
    assert np.allclose(total, want)
    assert (st[:, ex.S_FILLED] == Q).all() and (st[:, ex.S_REM] == 0).all() and (st[:, ex.S_DONE] == 1).all()
    # the books balance: summed step costs = side * (cash - mid0 * Q)
    assert np.allclose(total, sides * (st[:, ex.S_CASH] - mid0 * Q))


def test_immediate_cross_equals_walk(market):
    book, t0, sides, Q, T = market
    lat = 100_000
    st, total, mid0 = run_policy(book, t0, sides, Q, T, lambda e, k: ex.CROSS, latency=lat)
    for e in range(len(t0)):
        j = int(np.searchsorted(book.ts, t0[e] + lat, side="right")) - 1
        f, c, _u, short = tk.walk(*book.arrays(), int(sides[e]), int(Q[e]), j, ex.MAX_LEVELS, ex.PENALTY_TICKS, TICK)
        if short == 0:
            assert np.isclose(total[e], sides[e] * (c - mid0[e] * Q[e]))


def test_branch_matches_rollout(market):
    book, t0, sides, Q, T = market
    lat = 100_000
    rng = np.random.default_rng(0)
    st = ex.new_states(Q)
    mid0 = mid_at(book, t0)
    for k in range(K):
        n = len(t0)
        masks = np.zeros((n, ex.N_ACT), np.int64)
        nxt = np.zeros((n, ex.N_ACT, ex.AGENT_SIZE), np.int64)
        costs = np.zeros((n, ex.N_ACT))
        ex.branch_stage(*book.arrays(), st, sides, mid0, t0, T, k, K, lat, TICK, tk.EXACT, masks, nxt, costs)
        acts = np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int64)
        cost = np.zeros(n)
        st_copy = st.copy()
        ex.rollout_stage(*book.arrays(), st_copy, sides, mid0, t0, T, k, K, acts, lat, TICK, tk.EXACT, cost)
        assert np.array_equal(st_copy, nxt[np.arange(n), acts])
        assert np.allclose(cost, costs[np.arange(n), acts])
        # invalid actions are NaN, valid ones finite; CANCEL only when resting; HOLD always
        assert (masks[:, ex.HOLD] == 1).all()
        resting = (st[:, tk.ST_ACTIVE] == 1) & (st[:, ex.S_DONE] == 0)
        assert np.array_equal(masks[:, ex.CANCEL] == 1, resting)
        assert np.isnan(costs[masks == 0]).all() and np.isfinite(costs[masks == 1]).all()
        st = st_copy
    assert (st[:, ex.S_REM] == 0).all()


def test_features_are_consistent_with_state(market):
    book, t0, sides, Q, T = market
    fb = FlowBook.build(book)
    st = ex.new_states(Q)
    mid0 = mid_at(book, t0)
    cost = np.zeros(len(t0))
    ex.rollout_stage(*book.arrays(), st, sides, mid0, t0, T, 0, K, np.full(len(t0), ex.JOIN, np.int64), 0, TICK,
                     tk.EXACT, cost)
    raw = raw_features(fb, st, sides, mid0, t0, T, Q, 1, K)
    assert np.nanmin(raw[:, F_SPREAD]) >= TICK
    assert np.nanmin(raw[:, F_IMB]) >= -1 and np.nanmax(raw[:, F_IMB]) <= 1
    resting = st[:, tk.ST_ACTIVE] == 1
    flags = raw[:, [F_R_NEAR, F_R_IN, F_R_BEHIND]].sum(axis=1)
    assert np.array_equal(flags == 1, resting)


def test_hindsight_is_exact_on_small_trees():
    """The hindsight bound equals brute force over every action sequence (K = 3), and no policy beats it."""
    import itertools

    from invex.env.episodes import book_episodes
    from invex.env.units import unit_frame, valid_units
    from invex.eval import policies as P
    from invex.sim.tracker import EXACT, prepare
    from invex.sim.zi import ZIParams, simulate

    ko = {"c_t": 4.68, "c_q": 1.7e-4, "c_p": 0.028}
    ev, o, c = simulate(ZIParams(alpha=0.2), 2 * 3600, warmup_s=1800, seed=11)
    book = prepare(ev, o, c)
    eps = book_episodes(book, "zi", "d0", "Z", o, c, 30, ko)
    eps = eps[valid_units(unit_frame(eps, ko))].reset_index(drop=True)
    uf = unit_frame(eps, ko)
    K, lat = 3, 100_000
    arrs = book.arrays()
    sg, t0, T = eps["side"].to_numpy(np.int64), eps["t0"].to_numpy(np.int64), eps["T_ns"].to_numpy(np.int64)
    Q, mid0 = eps["Q"].to_numpy(np.int64), eps["mid0"].to_numpy(float)
    n = len(eps)
    for taker, rebate in ((0.0, 0.0), (30.0, 20.0)):
        best, ns, ov = np.zeros(n), np.zeros(n, np.int64), np.zeros(n, np.int64)
        ex.hindsight(*arrs, sg, mid0, t0, T, Q, K, lat, 100, EXACT, taker, rebate, 256, best, ns, ov)
        assert ov.sum() == 0
        brute = np.full(n, np.inf)
        for seq in itertools.product(range(ex.N_ACT), repeat=K):
            st = ex.new_states(Q)
            tot = np.zeros(n)
            for k, a in enumerate(seq):
                m = np.zeros((n, ex.N_ACT), np.int64)
                ex.masks_stage(*arrs, st, sg, t0, T, k, K, 100, m)
                acts = np.where(m[:, a] == 1, a, ex.HOLD).astype(np.int64)
                before = st.copy()
                cc = np.zeros(n)
                ex.rollout_stage(*arrs, st, sg, mid0, t0, T, k, K, acts, lat, 100, EXACT, cc)
                passive = st[:, ex.S_PASSIVE] - before[:, ex.S_PASSIVE]
                tot += cc + taker * (st[:, ex.S_FILLED] - before[:, ex.S_FILLED] - passive) - rebate * passive
            brute = np.minimum(brute, tot)
        np.testing.assert_allclose(best, brute, rtol=1e-9, atol=1e-6)
    res = P.evaluate_book(book, eps, uf, [P.Immediate(), P.PostThenCross()], K, lat, tick=100)
    best0 = np.zeros(n)
    ex.hindsight(*arrs, sg, mid0, t0, T, Q, K, lat, 100, EXACT, 0.0, 0.0, 256, best0, ns, ov)
    for _, g in res.groupby("policy"):
        assert (g["cost"].to_numpy() >= best0 - 1e-6).all()


def test_hidden_queue_rule(market):
    """The planted hidden queue (rule HIDDEN_BASE + h): h = 0 is the exact replay; more hidden shares ahead never
    add passive fills, remove some, and never change the queue position the policy sees at placement."""
    book, t0, sides, Q, T = market
    lat = 100_000

    def join_hold(rule):
        st = ex.new_states(Q)
        mid0 = mid_at(book, t0)
        total = np.zeros(len(t0))
        for k in range(K):
            acts = np.full(len(t0), ex.JOIN if k == 0 else ex.HOLD, np.int64)
            cost = np.zeros(len(t0))
            ex.rollout_stage(*book.arrays(), st, sides, mid0, t0, T, k, K, acts, lat, TICK, rule, cost)
            total += cost
        return st, total

    st0, c0 = join_hold(tk.EXACT)
    st1, c1 = join_hold(tk.HIDDEN_BASE)
    assert np.array_equal(st0, st1) and np.array_equal(c0, c1)
    st2, c2 = join_hold(tk.HIDDEN_BASE + 150)
    # at placement: the displayed queue is unchanged and the hidden shares are 1.5 per displayed share
    n_hidden = 0
    for e in range(len(t0)):
        j = int(np.searchsorted(book.ts, t0[e], side="right")) - 1
        price = int(book.bid_px[j] if sides[e] > 0 else book.ask_px[j])
        a, b = ex.new_states(Q[e:e + 1])[0], ex.new_states(Q[e:e + 1])[0]
        ex._post(book.arrays(), a, int(sides[e]), price, int(t0[e] + lat), tk.EXACT)
        ex._post(book.arrays(), b, int(sides[e]), price, int(t0[e] + lat), tk.HIDDEN_BASE + 150)
        assert a[tk.ST_AHEAD] == b[tk.ST_AHEAD] and a[ex.S_HIDDEN] == 0
        assert b[ex.S_HIDDEN] == (b[tk.ST_AHEAD] * 150 + 50) // 100
        n_hidden += b[ex.S_HIDDEN] > 0
    assert n_hidden > 0
    assert (st2[:, ex.S_PASSIVE] <= st0[:, ex.S_PASSIVE]).all()
    assert st2[:, ex.S_PASSIVE].sum() < st0[:, ex.S_PASSIVE].sum()
    assert (st2[:, ex.S_FILLED] == Q).all()


def test_markout_rule(market):
    """The planted markout (rule MARKOUT_BASE + MARKOUT_ZERO + m): fills are unchanged, m = 0 is the exact replay, and
    the extra cost is linear in m: m/1000 times the booked markouts of the passive fills, each marked one decision
    interval after it against the mid."""
    book, t0, sides, Q, T = market
    lat = 100_000
    base = tk.MARKOUT_BASE + tk.MARKOUT_ZERO

    def join_hold(rule):
        st = ex.new_states(Q)
        mid0 = mid_at(book, t0)
        total = np.zeros(len(t0))
        for k in range(K):
            acts = np.full(len(t0), ex.JOIN if k == 0 else ex.HOLD, np.int64)
            cost = np.zeros(len(t0))
            ex.rollout_stage(*book.arrays(), st, sides, mid0, t0, T, k, K, acts, lat, TICK, rule, cost)
            total += cost
        return st, total

    st0, c0 = join_hold(tk.EXACT)
    stz, cz = join_hold(base)
    stp, cp = join_hold(base + 1000)
    stm, cm = join_hold(base - 500)
    keep = [i for i in range(ex.AGENT_SIZE) if i not in (ex.S_TAU, ex.S_MARK, ex.S_MSPR)]
    assert np.array_equal(st0[:, keep], stz[:, keep]) and np.allclose(c0, cz)
    assert np.array_equal(st0[:, ex.S_PASSIVE], stp[:, ex.S_PASSIVE])
    assert np.allclose(cp - c0, stp[:, ex.S_MARK] / 2.0)
    assert np.allclose(cm - c0, -0.5 * stp[:, ex.S_MARK] / 2.0)
    assert (stp[:, ex.S_MARK] != 0).any()
    # marked by hand: replay the resting order row by row and mark each fill one interval later
    checked = 0
    for e in np.flatnonzero(stp[:, ex.S_MARK] != 0)[:10]:
        st = ex.new_states(Q[e:e + 1])[0]
        j = int(np.searchsorted(book.ts, t0[e], side="right")) - 1
        price = int(book.bid_px[j] if sides[e] > 0 else book.ask_px[j])
        ex._post(book.arrays(), st, int(sides[e]), price, int(t0[e] + lat), tk.EXACT)
        tau = int(T[e] // K)
        end = int(np.searchsorted(book.ts, t0[e] + T[e] + lat, side="right"))
        want = 0
        for i in range(int(st[tk.ST_NEXT]), end):
            f, _ = tk.advance(*book.arrays(), st, i + 1, tk.EXACT)
            if f:
                jm = int(np.searchsorted(book.ts, book.ts[i] + tau, side="right")) - 1
                if book.bid_px[jm] >= 0 and book.ask_px[jm] >= 0:
                    want += int(sides[e]) * f * (2 * price - int(book.bid_px[jm]) - int(book.ask_px[jm]))
            if not st[tk.ST_ACTIVE]:
                break
        assert stp[e, ex.S_MARK] == want
        checked += 1
    assert checked > 0


def test_demeaned_markout_and_absolute_hidden_rules(market):
    """The demeaned markout with mbar = 0 charges what the plain markout charges on the fills with a quoted spread;
    measured mbar makes the mean charge per passive share about zero; the absolute hidden queue binds an improving
    order too."""
    book, t0, sides, Q, T = market
    lat = 100_000

    def run(rule, act0=ex.JOIN):
        st = ex.new_states(Q)
        mid0 = mid_at(book, t0)
        total = np.zeros(len(t0))
        for k in range(K):
            acts = np.full(len(t0), act0 if k == 0 else ex.HOLD, np.int64)
            cost = np.zeros(len(t0))
            ex.rollout_stage(*book.arrays(), st, sides, mid0, t0, T, k, K, acts, lat, TICK, rule, cost)
            total += cost
        return st, total

    st0, c0 = run(tk.EXACT)
    plain, cp = run(tk.MARKOUT_BASE + tk.MARKOUT_ZERO + 1000)  # omega = 1
    dm0, cd0 = run(tk.DMARK_BASE + 10_000 * 1000 + 5000 + 100)  # omega = 1, mbar = 0
    assert np.array_equal(plain[:, ex.S_PASSIVE], st0[:, ex.S_PASSIVE])
    # with mbar = 0 the demeaned rule books 1000 x the plain markout of a fill with a quoted spread and 1 x that of a
    # fill without one, so the two agree exactly on every order all of whose fills had a quoted spread
    same = dm0[:, ex.S_MARK] == 1000 * plain[:, ex.S_MARK]
    assert same.mean() > 0.9 and (plain[same, ex.S_MARK] != 0).any()  # 95% of this book's orders
    assert np.allclose(cd0 - c0, dm0[:, ex.S_MARK] / 2000.0)
    mbar = (plain[:, ex.S_MARK].sum() / 2.0) / plain[:, ex.S_MSPR].sum()  # mean markout per share, in fill spreads
    code = int(round(1000 * mbar)) + 1000
    dm, cd = run(tk.DMARK_BASE + 10_000 * code + 5000 + 100)
    per_share = (cd - c0).sum() / max(plain[:, ex.S_PASSIVE].sum(), 1)
    raw_per_share = (cp - c0).sum() / max(plain[:, ex.S_PASSIVE].sum(), 1)
    assert abs(per_share) < 0.05 * abs(raw_per_share)
    # the absolute hidden queue: 5,000 hidden shares delay an improving order's fills, the proportional one does not
    _, ci = run(tk.EXACT, ex.IMPROVE)
    sh, _ = run(tk.HABS_BASE + 5000, ex.IMPROVE)
    sp, _ = run(tk.HIDDEN_BASE + 300, ex.IMPROVE)
    si, _ = run(tk.EXACT, ex.IMPROVE)
    assert np.array_equal(sp[:, ex.S_PASSIVE], si[:, ex.S_PASSIVE])
    assert sh[:, ex.S_PASSIVE].sum() < si[:, ex.S_PASSIVE].sum()

"""Fast tracker (event tables, oid trick) vs reference tracker (raw messages, explicit queues)."""

import numpy as np
import pytest

from invex import schema as S
from invex.itch.annotate import annotate
from invex.itch.index import index_file, order_by_locate
from invex.sim import tracker as tk
from invex.sim.toy import TICK, encode, toy_stream
from invex.sim.tracker_ref import reference_fills

FAR = 1 << 60


def build_table(msgs, t_snap, t_end=FAR):
    buf = np.frombuffer(encode(msgs), dtype=np.uint8)
    idx = index_file(buf, t_snap, t_end)
    perm, start = order_by_locate(idx.loc, idx.counts)
    sel = np.array([1])
    out = annotate(buf, idx.off, idx.typ, perm, start, sel, idx.nrows[sel], idx.max_ref, t_snap, t_end)
    assert out["n_err"][0] == 0 and out["n_over"][0] == 0
    n = out["n_rows"][0]
    ev = np.empty(n, dtype=S.EVENT_DTYPE)
    for c in S.COLUMNS:
        ev[c] = out[c][:n]
    return ev


def fast_fills(book, sgn, price, qty, t_arr, rule=tk.EXACT):
    st = tk.new_state()
    a = book.arrays()
    if tk.place(*a, st, sgn, price, qty, t_arr) != tk.PLACED:
        return None
    fills = []
    for i in range(int(st[tk.ST_NEXT]), book.n):
        f, _cash = tk.advance(*a, st, i + 1, rule)
        if f:
            fills.append((int(book.ts[i]), int(f)))
        if not st[tk.ST_ACTIVE]:
            break
    return fills


def random_orders(book, rng, t_lo, t_hi, k):
    for _ in range(k):
        t_arr = int(rng.integers(t_lo, t_hi))
        j = int(np.searchsorted(book.ts, t_arr, side="right")) - 1
        sgn = 1 if rng.random() < 0.5 else -1
        touch = book.bid_px[j] if sgn > 0 else book.ask_px[j]
        if touch < 0:
            continue
        # at the touch, one tick inside (alone), or one or two ticks behind it
        price = int(touch + sgn * TICK * rng.choice([0, 0, 1, -1, -2]))
        yield sgn, price, 100 * int(rng.integers(1, 6)), t_arr


@pytest.mark.parametrize("seed", range(30))
def test_fast_tracker_matches_reference(seed):
    msgs = toy_stream(seed, n=1500)
    t_first, t_last = msgs[0][1], msgs[-1][1]
    t_snap = t_first + (t_last - t_first) // 3  # orders resting at t_snap become SNAP rows
    book = tk.prepare(build_table(msgs, t_snap), 0, FAR)
    rng = np.random.default_rng(1000 + seed)
    n_checked = n_filled = 0
    for sgn, price, qty, t_arr in random_orders(book, rng, t_snap, t_last, 40):
        ref = reference_fills(msgs, sgn, price, qty, t_arr)
        fast = fast_fills(book, sgn, price, qty, t_arr)
        assert fast == ref, (sgn, price, qty, t_arr)
        n_checked += 1
        n_filled += bool(ref)
    assert n_checked >= 30 and n_filled >= 5  # the test exercises fills, not just empty lists


@pytest.mark.parametrize("seed", range(10))
def test_fill_rules_are_ordered(seed):
    msgs = toy_stream(seed, n=1500)
    book = tk.prepare(build_table(msgs, msgs[0][1]), 0, FAR)
    rng = np.random.default_rng(seed)
    for sgn, price, qty, t_arr in random_orders(book, rng, msgs[0][1], msgs[-1][1], 30):
        tot = {}
        for rule in (tk.PESSIMISTIC, tk.EXACT, tk.OPTIMISTIC):
            f = fast_fills(book, sgn, price, qty, t_arr, rule)
            tot[rule] = None if f is None else sum(q for _, q in f)
        if tot[tk.EXACT] is None:
            continue
        assert tot[tk.PESSIMISTIC] <= tot[tk.EXACT] <= tot[tk.OPTIMISTIC]


def test_marketable_order_is_rejected():
    msgs = toy_stream(3, n=500)
    book = tk.prepare(build_table(msgs, msgs[0][1]), 0, FAR)
    j = book.n // 2
    t = int(book.ts[j])
    j_arr = int(np.searchsorted(book.ts, t, side="right")) - 1
    st = tk.new_state()
    assert tk.place(*book.arrays(), st, 1, int(book.ask_px[j_arr]), 100, t) == tk.REJECTED_MARKETABLE
    assert tk.place(*book.arrays(), st, -1, int(book.bid_px[j_arr]), 100, t) == tk.REJECTED_MARKETABLE


@pytest.mark.parametrize("seed", range(5))
def test_walk_matches_book_depth(seed):
    from invex.sim.tracker_ref import AGENT  # noqa: F401  (same module family)

    msgs = toy_stream(seed, n=800)
    book = tk.prepare(build_table(msgs, msgs[0][1]), 0, FAR)
    # rebuild the book at a random row with plain Python and compare a sweep's cost
    rng = np.random.default_rng(seed)
    for _ in range(10):
        j = int(rng.integers(10, book.n - 1))
        depth = {1: {}, -1: {}}
        live = {}
        for i in range(j + 1):
            k, o = book.kind[i], int(book.oid[i])
            if k in (S.ADD, S.SNAP, S.MOD_ADD):
                live[i] = [int(book.side[i]), int(book.px[i]), int(book.qty[i])]
            elif k in (S.EXEC, S.CANCEL):
                live[o][2] -= int(book.qty[i])
            elif k in (S.DELETE, S.MOD_DEL):
                live[o][2] = 0
        for s, p, q in live.values():
            if q > 0:
                depth[s][p] = depth[s].get(p, 0) + q
        for sgn in (1, -1):
            q = 100 * int(rng.integers(1, 10))
            got, cash, used, short = tk.walk(*book.arrays(), sgn, q, j, 50, 5, TICK)
            want, wcash, left = 0, 0, q
            for p in sorted(depth[-sgn], reverse=(sgn < 0)):
                take = min(left, depth[-sgn][p])
                left -= take
                want += take
                wcash += take * p
                if left == 0:
                    break
            assert got == want and short == q - want
            if short == 0:
                assert cash == wcash


def test_rest_then_cross_paths_match_stepwise_simulation():
    from invex.sim.probe import rest_then_cross_paths

    msgs = toy_stream(11, n=1500)
    book = tk.prepare(build_table(msgs, msgs[0][1]), 0, FAR)
    rng = np.random.default_rng(5)
    n, K, lat = 40, 10, 100_000
    t0 = np.sort(rng.integers(book.ts[50], book.ts[-200], n))
    sides = np.where(np.arange(n) % 2 == 0, 1, -1)
    qtys = np.full(n, 300)
    T = np.full(n, int(2e8))  # 0.2 s: ~100 messages of the toy market
    out = rest_then_cross_paths(book, sides, qtys, t0, T, K, lat)
    a = book.arrays()
    for j in range(n):
        ia = int(np.searchsorted(book.ts, t0[j] + lat, side="right")) - 1
        price = int(book.bid_px[ia] if sides[j] > 0 else book.ask_px[ia])
        st = tk.new_state()
        ok = tk.place(*a, st, int(sides[j]), price, int(qtys[j]), int(t0[j] + lat)) == tk.PLACED
        assert bool(out["joined"][j]) == ok
        for k in range(1, K + 1):
            t_k = t0[j] + (T[j] * k) // K
            if ok:
                tk.advance(*a, st, int(np.searchsorted(book.ts, t_k, side="right")), tk.EXACT)
            assert out["ahead"][j, k] == (st[tk.ST_AHEAD] if st[tk.ST_ACTIVE] else 0)
            i_end = int(np.searchsorted(book.ts, t_k + lat, side="right"))
            if ok:
                tk.advance(*a, st, i_end, tk.EXACT)
            assert out["filled"][j, k] == st[tk.ST_FILLED] and out["cash"][j, k] == st[tk.ST_CASH]
            rem = int(qtys[j] - st[tk.ST_FILLED])
            if rem > 0:
                fx, cx, _u, sx = tk.walk(*a, int(sides[j]), rem, i_end - 1, 50, 5, TICK)
                assert (out["xfill"][j, k], out["xcash"][j, k], out["xshort"][j, k]) == (fx, cx, sx)

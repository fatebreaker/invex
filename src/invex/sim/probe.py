"""Batched probes on one replayed symbol-day: many virtual orders at once.

`passive_probes(book, sides, prices, qtys, t_arr, t_max, rule)` rests each
order (post-only) and tracks it until it is completely filled or `t_max`;
it returns the shares ahead at arrival, the times of the first and of the
completing fill (-1 if none) and a status (0 placed, 1 rejected as
marketable). `mid_at(book, times)` returns the mid after the last row at or
before each time (NaN when one side of the book is empty). Orders are
independent, so the loops run in parallel.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from . import tracker as tk


@nb.njit(parallel=True, cache=True)
def _passive(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
             sides, prices, qtys, t_arr, t_max, rule, ahead0, t_first, t_done, status):
    for j in nb.prange(sides.shape[0]):
        st = np.zeros(tk.ST_SIZE, dtype=np.int64)
        s = tk.place(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off,
                     lv_rows, st, sides[j], prices[j], qtys[j], t_arr[j])
        status[j] = s
        t_first[j] = -1
        t_done[j] = -1
        if s != tk.PLACED:
            ahead0[j] = -1
            continue
        ahead0[j] = st[tk.ST_AHEAD]
        i_max = tk.row_after(ts, t_max[j])
        tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                   st, i_max, rule)
        if st[tk.ST_FIRST] >= 0:
            t_first[j] = ts[st[tk.ST_FIRST]]
        if st[tk.ST_ACTIVE] == 0:
            t_done[j] = ts[st[tk.ST_NEXT] - 1]  # advance stops right after the completing row


def passive_probes(book: tk.ReplayBook, sides, prices, qtys, t_arr, t_max, rule: int = tk.EXACT) -> dict:
    n = len(sides)
    out = {k: np.zeros(n, dtype=np.int64) for k in ("ahead0", "t_first", "t_done", "status")}
    _passive(*book.arrays(), np.asarray(sides, np.int64), np.asarray(prices, np.int64), np.asarray(qtys, np.int64),
             np.asarray(t_arr, np.int64), np.asarray(t_max, np.int64), rule,
             out["ahead0"], out["t_first"], out["t_done"], out["status"])
    return out


@nb.njit(parallel=True, cache=True)
def _rest_paths(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                sides, qtys, t0, horizon, K, latency, rule, tick,
                filled, cash, ahead, xfill, xcash, xshort, joined):
    """For each episode: join the near touch at t0 + latency, then at decision k = 1..K (t_k = t0 + k T / K)
    record the shares ahead at t_k (what a rule may look at), the shares filled and cash once a cancel
    sent at t_k has arrived (t_k + latency), and what crossing the rest then costs (walk the book after
    t_k + latency). Column 0 is the immediate cross at t0 (nothing resting)."""
    for j in nb.prange(sides.shape[0]):
        sgn = sides[j]
        q = qtys[j]
        ta = t0[j] + latency
        # immediate cross at t0
        ia = tk.row_after(ts, ta) - 1
        f0, c0, _u, s0 = tk.walk(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels,
                                 lv_off, lv_rows, sgn, q, ia, 50, 5, tick)
        filled[j, 0] = 0
        cash[j, 0] = 0
        ahead[j, 0] = -1
        xfill[j, 0] = f0
        xcash[j, 0] = c0
        xshort[j, 0] = s0
        st = np.zeros(tk.ST_SIZE, dtype=np.int64)
        price = bid_px[ia] if sgn > 0 else ask_px[ia]
        ok = 0
        if ia >= 0 and price >= 0:
            if tk.place(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off,
                        lv_rows, st, sgn, price, q, ta) == tk.PLACED:
                ok = 1
        joined[j] = ok
        for k in range(1, K + 1):
            tk_ = t0[j] + (horizon[j] * k) // K
            # the decision sees the queue at t_k; a cancel sent then arrives after the latency,
            # and fills in between still count
            if ok:
                tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off,
                           lv_rows, st, tk.row_after(ts, tk_), rule)
            ahead[j, k] = st[tk.ST_AHEAD] if st[tk.ST_ACTIVE] == 1 else 0
            i_end = tk.row_after(ts, tk_ + latency)
            if ok:
                tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off,
                           lv_rows, st, i_end, rule)
            filled[j, k] = st[tk.ST_FILLED]
            cash[j, k] = st[tk.ST_CASH]
            rem = q - st[tk.ST_FILLED]
            if rem > 0:
                fx, cx, _u, sx = tk.walk(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz,
                                         levels, lv_off, lv_rows, sgn, rem, i_end - 1, 50, 5, tick)
            else:
                fx, cx, sx = 0, 0, 0
            xfill[j, k] = fx
            xcash[j, k] = cx
            xshort[j, k] = sx


def rest_then_cross_paths(book: tk.ReplayBook, sides, qtys, t0, horizon, K: int = 10, latency: int = 100_000,
                          rule: int = tk.EXACT, tick: int = 100) -> dict:
    """Arrays (n_episodes, K + 1): filled, cash, ahead, xfill, xcash, xshort; plus joined (n,).

    A threshold rule "cross at the first decision k where ahead_k / size_unit > theta * time_left_k /
    time_unit" then costs cash_k + xcash_k for k* (k = K if never), the same for every theta and unit
    system, so rules can be compared without re-simulating."""
    n = len(sides)
    out = {k: np.zeros((n, K + 1), dtype=np.int64) for k in ("filled", "cash", "ahead", "xfill", "xcash", "xshort")}
    out["joined"] = np.zeros(n, dtype=np.int64)
    _rest_paths(*book.arrays(), np.asarray(sides, np.int64), np.asarray(qtys, np.int64), np.asarray(t0, np.int64),
                np.asarray(horizon, np.int64), K, latency, rule, tick,
                out["filled"], out["cash"], out["ahead"], out["xfill"], out["xcash"], out["xshort"], out["joined"])
    return out


@nb.njit(parallel=True, cache=True)
def _mids(ts, bid_px, ask_px, times, out):
    for j in nb.prange(times.shape[0]):
        i = tk.row_after(ts, times[j]) - 1
        if i >= 0 and bid_px[i] >= 0 and ask_px[i] >= 0:
            out[j] = 0.5 * (bid_px[i] + ask_px[i])
        else:
            out[j] = np.nan


def mid_at(book: tk.ReplayBook, times) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64)
    out = np.empty(times.shape[0])
    _mids(book.ts, book.bid_px, book.ask_px, times, out)
    return out


def touch_at(book: tk.ReplayBook, times) -> tuple[np.ndarray, np.ndarray]:
    """Best bid and ask (price units, -1 if empty) after the last row at or before each time."""
    i = np.searchsorted(book.ts, np.asarray(times, dtype=np.int64), side="right") - 1
    ok = i >= 0
    bid = np.where(ok, book.bid_px[np.maximum(i, 0)], -1)
    ask = np.where(ok, book.ask_px[np.maximum(i, 0)], -1)
    return bid, ask


@nb.njit(parallel=True, cache=True)
def _twins(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
           rows, offset, t_max, rule, first_row):
    """A virtual order with the side, price and size of the real order created at rows[j], inserted at row
    rows[j] + offset (0: the real order's own queue position; 1: right behind it); first_row[j] = row of
    its first fill (-1 if none by t_max)."""
    n_lev = levels.shape[0]
    for j in nb.prange(rows.shape[0]):
        r = rows[j]
        ins = r + offset
        st = np.zeros(tk.ST_SIZE, dtype=np.int64)
        L = tk.level_index(levels, px[r])
        sd = 0 if side[r] > 0 else 1
        st[tk.ST_ACTIVE] = 1
        st[tk.ST_SIDE] = side[r]
        st[tk.ST_PX] = px[r]
        st[tk.ST_REM] = qty[r]
        st[tk.ST_AHEAD] = tk.size_at(lvl, lv_off, lv_rows, n_lev, sd, L, ins - 1)
        st[tk.ST_INS] = ins
        st[tk.ST_NEXT] = ins
        st[tk.ST_LEV] = L
        st[tk.ST_FIRST] = -1
        if offset == 0:
            st[tk.ST_AHEAD] = tk.size_at(lvl, lv_off, lv_rows, n_lev, sd, L, r - 1)
        tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                   st, tk.row_after(ts, t_max[j]), rule)
        first_row[j] = st[tk.ST_FIRST]


def twin_probes(book: tk.ReplayBook, rows, offset: int, t_max, rule: int = tk.EXACT) -> np.ndarray:
    rows = np.asarray(rows, np.int64)
    out = np.full(rows.shape[0], -1, np.int64)
    _twins(*book.arrays(), rows, offset, np.asarray(t_max, np.int64), rule, out)
    return out

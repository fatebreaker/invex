"""The atomic make-or-take task in non-intrusive replay.

An episode buys (sgn = +1) or sells (-1) Q shares within T, starting at t0,
with K decisions at t_k = t0 + k T / K (k = 0..K-1) and a forced cross of
whatever is left at t0 + T. At a decision the agent sees the book after every
event at or before t_k; its action reaches the exchange `latency` later, and a
resting order keeps receiving fills until its cancel arrives. Actions:

    HOLD     keep the current state (a resting order keeps its queue priority)
    JOIN     rest the remaining shares at the near touch (a resting order elsewhere is cancelled first)
    IMPROVE  rest them one tick inside the spread (spread >= 2 ticks)
    CANCEL   cancel the resting order
    CROSS    cancel and take the remaining shares from the visible book

Posts are post-only: an order that would execute on arrival is not placed.
The market path is exogenous, so `branch` can evaluate every action from the
same state on the same path, which is what fitted Q-iteration needs, and
every policy can be compared on identical episodes.

Cost of a step: side * (cash paid - mid0 * shares) over the step, in price
units x shares (side = +1 buy: paying more than the arrival mid is a cost).
Divide by Q and a price unit for a normalised cost.

Agent state: the tracker state (invex.sim.tracker, ST_*) of the resting
order, then remaining shares, total cash, total filled, done flag, number of
rejected posts and shortfall shares (crosses beyond visible depth).
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ..sim import tracker as tk

HOLD, JOIN, IMPROVE, CANCEL, CROSS = range(5)
N_ACT = 5
ACTION_NAMES = ("HOLD", "JOIN", "IMPROVE", "CANCEL", "CROSS")

# S_FILLED counts every filled share; S_PASSIVE only those filled while resting (maker fills, for fees)
# S_HIDDEN: hidden shares ahead of the resting order; S_TAU, S_MARK: markout horizon and booked markouts of passive
# fills (planted positive controls only, rules from tk.HIDDEN_BASE and tk.MARKOUT_BASE on)
S_REM, S_CASH, S_FILLED, S_DONE, S_REJ, S_SHORT, S_PASSIVE, S_HIDDEN, S_TAU, S_MARK, S_MSPR = (
    tk.ST_SIZE + i for i in range(11))
AGENT_SIZE = tk.ST_SIZE + 11
assert S_HIDDEN == tk.ST_HIDDEN and S_TAU == tk.ST_TAU and S_MARK == tk.ST_MARK and S_MSPR == tk.ST_MSPR

MAX_LEVELS = 50
PENALTY_TICKS = 5


@nb.njit(inline="always")
def _touch(bid_px, ask_px, j, sgn):
    """(near, far) touch prices after row j for an agent on side sgn (-1 if empty)."""
    if j < 0:
        return -1, -1
    if sgn > 0:
        return bid_px[j], ask_px[j]
    return ask_px[j], bid_px[j]


@nb.njit
def init_state(st, q):
    for i in range(AGENT_SIZE):
        st[i] = 0
    st[tk.ST_FIRST] = -1
    st[S_REM] = q


@nb.njit
def action_mask(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                st, sgn, t_k, tick, mask):
    """Valid actions at decision time t_k (mask[a] = 1)."""
    for a in range(N_ACT):
        mask[a] = 0
    if st[S_DONE] == 1:
        mask[HOLD] = 1
        return
    j = tk.row_after(ts, t_k) - 1
    near, far = _touch(bid_px, ask_px, j, sgn)
    resting = st[tk.ST_ACTIVE] == 1
    mask[HOLD] = 1
    mask[CROSS] = 1
    if resting:
        mask[CANCEL] = 1
    if near >= 0 and far >= 0:
        if not (resting and st[tk.ST_PX] == near):
            mask[JOIN] = 1
        if abs(far - near) >= 2 * tick:
            inside = near + sgn * tick
            if not (resting and st[tk.ST_PX] == inside):
                mask[IMPROVE] = 1


@nb.njit
def _cancel(arrs, st, i_arr, rule):
    """Advance the resting order to its cancel's arrival row, then remove it."""
    ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows = arrs
    if st[tk.ST_ACTIVE] == 1:
        f0 = st[tk.ST_FILLED]
        c0 = st[tk.ST_CASH]
        tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                   st, i_arr, rule)
        df = st[tk.ST_FILLED] - f0
        st[S_REM] -= df
        st[S_FILLED] += df
        st[S_PASSIVE] += df
        st[S_CASH] += st[tk.ST_CASH] - c0
        st[tk.ST_ACTIVE] = 0


@nb.njit
def _rest(arrs, st, i_end, rule):
    """Advance a resting order to row i_end (exclusive), booking its fills."""
    ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows = arrs
    if st[tk.ST_ACTIVE] == 1:
        f0 = st[tk.ST_FILLED]
        c0 = st[tk.ST_CASH]
        tk.advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                   st, i_end, rule)
        df = st[tk.ST_FILLED] - f0
        st[S_REM] -= df
        st[S_FILLED] += df
        st[S_PASSIVE] += df
        st[S_CASH] += st[tk.ST_CASH] - c0


@nb.njit
def _cross(arrs, st, sgn, j, tick):
    ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows = arrs
    rem = st[S_REM]
    if rem <= 0:
        return
    f, c, _u, short = tk.walk(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels,
                              lv_off, lv_rows, sgn, rem, j, MAX_LEVELS, PENALTY_TICKS, tick)
    if f == 0 and short == rem:
        # nothing visible on the other side: price the whole remainder off the near touch
        near_px = bid_px[j] if sgn > 0 else ask_px[j]
        if j >= 0 and near_px >= 0:
            c = rem * (near_px + sgn * (PENALTY_TICKS + 1) * tick)
    st[S_CASH] += c
    st[S_FILLED] += rem
    st[S_SHORT] += short
    st[S_REM] = 0


@nb.njit
def _post(arrs, st, sgn, price, t_arr, rule):
    ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows = arrs
    # reuse the tracker slots for the new order (its own filled / cash counters start at 0)
    st[tk.ST_FILLED] = 0
    st[tk.ST_CASH] = 0
    r = tk.place(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                 st, sgn, price, st[S_REM], t_arr)
    hid = rule >= tk.HIDDEN_BASE and rule < tk.MARKOUT_BASE
    habs = rule >= tk.HABS_BASE and rule < tk.HABS_BASE + tk.RULE_SPAN
    if hid:
        st[S_HIDDEN] = (st[tk.ST_AHEAD] * (rule - tk.HIDDEN_BASE) + 50) // 100
    elif habs:  # the same hidden shares wherever the order rests, improving or not
        st[S_HIDDEN] = rule - tk.HABS_BASE
    else:
        st[S_HIDDEN] = 0
    if r != tk.PLACED:
        st[S_REJ] += 1
        st[tk.ST_ACTIVE] = 0


@nb.njit
def step(arrs, st, sgn, mid0, t_k, t_next, last, a, latency, tick, rule):
    """Apply action a at t_k, then let the market run to t_next (exclusive of later events).
    If `last`, cross whatever is left at t_next (the episode end). Returns the step cost
    side * (cash paid - mid0 * shares filled) in price units x shares."""
    ts = arrs[0]
    bid_px = arrs[8]
    ask_px = arrs[10]
    if st[S_DONE] == 1:
        return 0.0
    cash0 = st[S_CASH]
    filled0 = st[S_FILLED]
    mark0 = st[S_MARK]
    plain = rule >= tk.MARKOUT_BASE and rule < tk.MARKOUT_END
    dmark = rule >= tk.DMARK_BASE and rule < tk.DMARK_BASE + tk.RULE_SPAN
    if plain or dmark:
        st[S_TAU] = t_next - t_k  # passive fills are marked one decision interval later
    j_dec = tk.row_after(ts, t_k) - 1
    i_arr = tk.row_after(ts, t_k + latency)
    near, far = _touch(bid_px, ask_px, j_dec, sgn)
    if a == JOIN or a == IMPROVE or a == CANCEL or a == CROSS:
        _cancel(arrs, st, i_arr, rule)
        if a == CROSS:
            _cross(arrs, st, sgn, i_arr - 1, tick)
        elif (a == JOIN or a == IMPROVE) and st[S_REM] > 0 and near >= 0:
            price = near if a == JOIN else near + sgn * tick
            _post(arrs, st, sgn, price, t_k + latency, rule)
    i_next = tk.row_after(ts, t_next)
    if st[S_REM] > 0:
        _rest(arrs, st, i_next, rule)
    if last and st[S_REM] > 0:
        i_end = tk.row_after(ts, t_next + latency)
        _cancel(arrs, st, i_end, rule)
        _cross(arrs, st, sgn, i_end - 1, tick)
    if st[S_REM] <= 0:
        st[S_DONE] = 1
        st[tk.ST_ACTIVE] = 0
    c = sgn * ((st[S_CASH] - cash0) - mid0 * (st[S_FILLED] - filled0))
    if plain:
        c += (rule - tk.MARKOUT_BASE - tk.MARKOUT_ZERO) / 1000.0 * (st[S_MARK] - mark0) / 2.0
    elif dmark:
        c += ((rule - tk.DMARK_BASE) % 10_000 - 5000) / 100.0 * (st[S_MARK] - mark0) / 2000.0
    return c


@nb.njit(parallel=True, cache=True)
def rollout_stage(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                  states, sgns, mid0, t0, T, k, K, actions, latency, tick, rule, cost):
    """Stage k for every episode (all on one book): apply actions[e], record cost[e]."""
    arrs = (ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows)
    for e in nb.prange(states.shape[0]):
        t_k = t0[e] + (T[e] * k) // K
        t_n = t0[e] + (T[e] * (k + 1)) // K
        cost[e] = step(arrs, states[e], sgns[e], mid0[e], t_k, t_n, k == K - 1, actions[e], latency, tick, rule)


@nb.njit(parallel=True, cache=True)
def branch_stage(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                 states, sgns, mid0, t0, T, k, K, latency, tick, rule, masks, next_states, costs):
    """Every valid action from every episode's current state at stage k (states are not modified):
    next_states[e, a], costs[e, a]; masks[e, a] = 1 where the action was valid."""
    arrs = (ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows)
    for e in nb.prange(states.shape[0]):
        t_k = t0[e] + (T[e] * k) // K
        t_n = t0[e] + (T[e] * (k + 1)) // K
        m = np.zeros(N_ACT, dtype=np.int64)
        action_mask(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                    states[e], sgns[e], t_k, tick, m)
        for a in range(N_ACT):
            masks[e, a] = m[a]
            for i in range(AGENT_SIZE):
                next_states[e, a, i] = states[e, i]
            if m[a] == 1:
                costs[e, a] = step(arrs, next_states[e, a], sgns[e], mid0[e], t_k, t_n, k == K - 1, a, latency, tick,
                                   rule)
            else:
                costs[e, a] = np.nan


@nb.njit(parallel=True, cache=True)
def masks_stage(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                states, sgns, t0, T, k, K, tick, masks):
    for e in nb.prange(states.shape[0]):
        t_k = t0[e] + (T[e] * k) // K
        m = np.zeros(N_ACT, dtype=np.int64)
        action_mask(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
                    states[e], sgns[e], t_k, tick, m)
        for a in range(N_ACT):
            masks[e, a] = m[a]


@nb.njit(inline="always")
def _equivalent(a, b):
    """Two agent states with the same future: the same remaining shares and, if an order rests, the same order
    (price, size, queue ahead, insertion and next row). Cash and fill counters only book the past."""
    if a[S_DONE] != b[S_DONE]:
        return False
    if a[S_DONE] == 1:
        return True
    if a[S_REM] != b[S_REM] or a[tk.ST_ACTIVE] != b[tk.ST_ACTIVE]:
        return False
    if a[tk.ST_ACTIVE] == 0:
        return True  # an order is placed afresh (tracker fields overwritten) or the rest is crossed
    return (a[tk.ST_PX] == b[tk.ST_PX] and a[tk.ST_REM] == b[tk.ST_REM] and a[tk.ST_AHEAD] == b[tk.ST_AHEAD]
            and a[tk.ST_INS] == b[tk.ST_INS] and a[tk.ST_NEXT] == b[tk.ST_NEXT] and a[tk.ST_LEV] == b[tk.ST_LEV]
            and a[S_HIDDEN] == b[S_HIDDEN])


@nb.njit(parallel=True, cache=True)
def hindsight(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
              sgns, mid0, t0, T, Q, K, latency, tick, rule, taker, rebate, max_states, best, n_states, overflow):
    """The lowest cost any sequence of actions attains on each episode's path, known in advance (a bound for every
    policy with these actions and decision times). Exact dynamic programming forward over stages: every valid
    action from every reachable state, merging states with the same future (`_equivalent`) and keeping the
    cheapest past of each. Costs as `step`, plus taker * shares taken - rebate * shares filled resting.
    overflow[e] = 1 if more than max_states distinct states were reachable (best[e] is then not exact)."""
    arrs = (ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows)
    for e in nb.prange(sgns.shape[0]):
        cur = np.zeros((max_states, AGENT_SIZE), np.int64)
        nxt = np.zeros((max_states, AGENT_SIZE), np.int64)
        cur_c = np.zeros(max_states)
        nxt_c = np.zeros(max_states)
        tmp = np.zeros(AGENT_SIZE, np.int64)
        m = np.zeros(N_ACT, np.int64)
        init_state(cur[0], Q[e])
        n_cur = 1
        peak = 1
        for k in range(K):
            t_k = t0[e] + (T[e] * k) // K
            t_n = t0[e] + (T[e] * (k + 1)) // K
            n_nxt = 0
            for i in range(n_cur):
                action_mask(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off,
                            lv_rows, cur[i], sgns[e], t_k, tick, m)
                for a in range(N_ACT):
                    if m[a] == 0:
                        continue
                    for f in range(AGENT_SIZE):
                        tmp[f] = cur[i, f]
                    c = step(arrs, tmp, sgns[e], mid0[e], t_k, t_n, k == K - 1, a, latency, tick, rule)
                    passive = tmp[S_PASSIVE] - cur[i, S_PASSIVE]
                    c += taker * (tmp[S_FILLED] - cur[i, S_FILLED] - passive) - rebate * passive
                    total = cur_c[i] + c
                    j = -1
                    for jj in range(n_nxt):
                        if _equivalent(nxt[jj], tmp):
                            j = jj
                            break
                    if j >= 0:
                        if total < nxt_c[j]:
                            nxt_c[j] = total
                    elif n_nxt < max_states:
                        for f in range(AGENT_SIZE):
                            nxt[n_nxt, f] = tmp[f]
                        nxt_c[n_nxt] = total
                        n_nxt += 1
                    else:
                        overflow[e] = 1
            for i in range(n_nxt):
                for f in range(AGENT_SIZE):
                    cur[i, f] = nxt[i, f]
                cur_c[i] = nxt_c[i]
            n_cur = n_nxt
            if n_cur > peak:
                peak = n_cur
        b = np.inf
        for i in range(n_cur):
            if cur_c[i] < b:
                b = cur_c[i]
        best[e] = b
        n_states[e] = peak


def new_states(qtys) -> np.ndarray:
    qtys = np.asarray(qtys, dtype=np.int64)
    st = np.zeros((len(qtys), AGENT_SIZE), dtype=np.int64)
    st[:, tk.ST_FIRST] = -1
    st[:, S_REM] = qtys
    return st

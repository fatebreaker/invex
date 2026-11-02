"""Santa Fe zero-intelligence order book (Daniels et al. 2003; Farmer, Patelli & Zovko 2005),
with price-time priority, emitting invex event tables. Used for the controls.

Per side, limit orders of `size` shares arrive at rate `alpha` per tick at the L
prices behind the opposite best quote (a buy at ask - k ticks, k = 1..L, so
limits never cross); market orders of `size` shares arrive at rate `mu` and
sweep the other side in price-time priority; every resting order is
cancelled at rate `delta`. Prices live on a grid of `tick` price units.

Exact symmetries: multiplying alpha, mu and delta by c rescales time by 1/c;
multiplying `size` by s rescales every share count by s; multiplying `tick`
(and the price level) by p rescales every price by p. Assets related by these
maps are exactly similar: in units of (inter-trade time, trade size, spread)
their dynamics are identical. What is left are dimensionless parameters
(L, alpha / mu, delta / mu): they set the spread in ticks, queue lengths in
orders and so on. The controls vary them on purpose (e.g. the relative tick).

`simulate(params, seconds, warmup_s, seed)` returns (events, open_ns, close_ns):
the table starts empty and fills during the warm-up; the session is
[warmup, warmup + seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .. import schema as S

NS = 1_000_000_000


@dataclass(frozen=True)
class ZIParams:
    alpha: float = 0.05  # limit orders per second per tick per side
    mu: float = 0.5  # market orders per second per side
    delta: float = 0.01  # cancellation rate per resting order (1/s)
    L: int = 30  # limit-order placement range (ticks behind the opposite best)
    size: int = 100  # shares per order
    tick: int = 100  # price units per tick (100 = $0.01)
    p0_ticks: int = 5000  # initial mid, in ticks (price level = p0_ticks * tick)

    def scaled(self, c: float = 1.0, s: int = 1, p: int = 1) -> "ZIParams":
        """The exactly similar asset: time / c, shares * s, prices * p."""
        return ZIParams(self.alpha * c, self.mu * c, self.delta * c, self.L, self.size * s, self.tick * p,
                        self.p0_ticks)


@nb.njit(cache=True)
def _sim(alpha, mu, delta, L, size, tick, p0, t_end_ns, seed, cap_rows, cap_orders):
    np.random.seed(seed)
    G = 2 * p0  # price grid in ticks [0, G)
    head = np.full((2, G), -1, np.int64)  # [0] bids, [1] asks
    tail = np.full((2, G), -1, np.int64)
    lvl_n = np.zeros((2, G), np.int64)  # resting orders per level
    o_side = np.zeros(cap_orders, np.int64)
    o_lvl = np.zeros(cap_orders, np.int64)
    o_row = np.zeros(cap_orders, np.int64)
    o_rem = np.zeros(cap_orders, np.int64)
    o_next = np.full(cap_orders, -1, np.int64)
    o_prev = np.full(cap_orders, -1, np.int64)
    alive = np.zeros(cap_orders, np.int64)  # list of live order ids (swap-remove)
    pos = np.full(cap_orders, -1, np.int64)
    n_alive = 0
    n_ord = 0
    ts = np.zeros(cap_rows, np.int64)
    kind = np.zeros(cap_rows, np.uint8)
    side = np.zeros(cap_rows, np.int8)
    px = np.zeros(cap_rows, np.int64)
    qty = np.zeros(cap_rows, np.int64)
    oid = np.zeros(cap_rows, np.int32)
    r = 0
    bb = -1  # best bid level
    ba = -1  # best ask level
    t = 0.0
    last_mid = p0
    while True:
        rate = 2.0 * alpha * L + 2.0 * mu + delta * n_alive
        t += np.random.exponential(1.0 / rate)
        t_ns = np.int64(t * 1e9)
        if t_ns >= t_end_ns or r >= cap_rows - 64 or n_ord >= cap_orders - 2:
            break
        u = np.random.random() * rate
        if u < 2.0 * alpha * L:  # limit order
            sd = 0 if np.random.random() < 0.5 else 1
            k = 1 + np.random.randint(L)
            if sd == 0:
                ref = ba if ba >= 0 else last_mid + 1
                lv = ref - k
            else:
                ref = bb if bb >= 0 else last_mid - 1
                lv = ref + k
            if lv < 1 or lv >= G - 1:
                continue
            o = n_ord
            n_ord += 1
            o_side[o] = sd
            o_lvl[o] = lv
            o_rem[o] = size
            o_row[o] = r
            if tail[sd, lv] < 0:
                head[sd, lv] = o
                o_prev[o] = -1
            else:
                o_next[tail[sd, lv]] = o
                o_prev[o] = tail[sd, lv]
            o_next[o] = -1
            tail[sd, lv] = o
            lvl_n[sd, lv] += 1
            alive[n_alive] = o
            pos[o] = n_alive
            n_alive += 1
            if sd == 0 and (bb < 0 or lv > bb):
                bb = lv
            if sd == 1 and (ba < 0 or lv < ba):
                ba = lv
            ts[r] = t_ns
            kind[r] = S.ADD
            side[r] = 1 if sd == 0 else -1
            px[r] = lv * tick
            qty[r] = size
            oid[r] = r
            r += 1
        elif u < 2.0 * alpha * L + 2.0 * mu:  # market order: sweep the other side
            buy = np.random.random() < 0.5
            sd = 1 if buy else 0  # side being hit
            want = size
            while want > 0:
                lv = ba if sd == 1 else bb
                if lv < 0:
                    break
                o = head[sd, lv]
                take = o_rem[o] if o_rem[o] < want else want
                o_rem[o] -= take
                want -= take
                ts[r] = t_ns
                kind[r] = S.EXEC
                side[r] = 1 if sd == 0 else -1
                px[r] = lv * tick
                qty[r] = take
                oid[r] = o_row[o]
                r += 1
                if o_rem[o] == 0:
                    # unlink the head
                    nx = o_next[o]
                    head[sd, lv] = nx
                    if nx >= 0:
                        o_prev[nx] = -1
                    else:
                        tail[sd, lv] = -1
                    lvl_n[sd, lv] -= 1
                    p_ = pos[o]
                    last = alive[n_alive - 1]
                    alive[p_] = last
                    pos[last] = p_
                    n_alive -= 1
                    pos[o] = -1
                    if lvl_n[sd, lv] == 0:
                        if sd == 1:
                            while ba >= 0 and ba < G and lvl_n[1, ba] == 0:
                                ba += 1
                            if ba >= G:
                                ba = -1
                        else:
                            while bb >= 0 and lvl_n[0, bb] == 0:
                                bb -= 1
        else:  # cancel a random resting order
            if n_alive == 0:
                continue
            o = alive[np.random.randint(n_alive)]
            sd = o_side[o]
            lv = o_lvl[o]
            p_, n_ = o_prev[o], o_next[o]
            if p_ >= 0:
                o_next[p_] = n_
            else:
                head[sd, lv] = n_
            if n_ >= 0:
                o_prev[n_] = p_
            else:
                tail[sd, lv] = p_
            lvl_n[sd, lv] -= 1
            q = pos[o]
            last = alive[n_alive - 1]
            alive[q] = last
            pos[last] = q
            n_alive -= 1
            pos[o] = -1
            ts[r] = t_ns
            kind[r] = S.DELETE
            side[r] = 1 if sd == 0 else -1
            px[r] = lv * tick
            qty[r] = o_rem[o]
            oid[r] = o_row[o]
            r += 1
            if lvl_n[sd, lv] == 0:
                if sd == 1 and lv == ba:
                    while ba >= 0 and ba < G and lvl_n[1, ba] == 0:
                        ba += 1
                    if ba >= G:
                        ba = -1
                elif sd == 0 and lv == bb:
                    while bb >= 0 and lvl_n[0, bb] == 0:
                        bb -= 1
        if bb >= 0 and ba >= 0:
            last_mid = (bb + ba) // 2
    return ts[:r], kind[:r], side[:r], px[:r], qty[:r], oid[:r]


def simulate(prm: ZIParams, seconds: float, warmup_s: float = 1800.0, seed: int = 0):
    """Event table (invex.schema.EVENT_DTYPE) and the session bounds (open_ns, close_ns)."""
    t_end = int((warmup_s + seconds) * NS)
    est = int(t_end / NS * (2 * prm.alpha * prm.L + 2 * prm.mu) * 3 + 10_000)
    cols = _sim(prm.alpha, prm.mu, prm.delta, prm.L, prm.size, prm.tick, prm.p0_ticks, t_end, seed,
                est, est)
    ev = np.zeros(cols[0].shape[0], dtype=S.EVENT_DTYPE)
    for name, c in zip(("ts", "kind", "side", "px", "qty", "oid"), cols):
        ev[name] = c
    ev["aux"] = np.where(ev["kind"] == S.EXEC, ev["px"], 0)
    ev["flags"] = np.where(ev["kind"] == S.EXEC, S.F_PRINTABLE, 0).astype(np.uint8)
    return ev, int(warmup_s * NS), t_end

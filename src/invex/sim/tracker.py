"""Virtual orders in non-intrusive L3 replay.

The historical event stream does not react to the agent. A virtual limit
order arrives at `t_dec + latency` (behind every event with a timestamp at or
before its arrival), joins the back of its price level, and is filled by the
same aggressors that hit the real queue. `prepare(ev, open, close)` turns one
symbol's event table into a `ReplayBook` (event columns, the replayed best
quotes and level sizes, and a per-level index of rows); the compiled
functions below work on its arrays and can be inlined into batched
environments.

Order state is a small int64 vector (see the ST_* indices). Fill rules:

* EXACT (default): the shares ahead at arrival are the displayed size of the
  level; cancels, deletes, replaces and executions of orders added before the
  arrival (`oid < ins`) reduce them. The order fills when an execution at its
  level hits an order added after it (that aggressor would have hit the
  virtual order first), or any execution at its level once nothing is ahead;
  when an execution on its side happens at a worse price (trade-through, judged
  by the execution price); when
  an order on the other side is added at a price that crosses it (only
  possible when it is alone inside the spread); and when a hidden execution
  prints at its price on its side once nothing is ahead (displayed orders have
  priority over non-displayed ones at the same price), which includes an
  execution of another order away from its displayed price at exactly the
  virtual order's price. An execution of an order at the virtual order's level
  but away from its displayed price (discretion) never fills it; it only
  removes shares ahead.
* PESSIMISTIC: fills only on trade-throughs and crossing adds (always last in
  the queue, hidden prints ignored): a lower bound.
* OPTIMISTIC: any execution at its price fills it, queue ignored: an upper bound.
* HIDDEN_BASE + h (h = 1, 2, ...; a planted positive control, never the replay itself): EXACT, plus h/100 hidden
  shares ahead per displayed share ahead at arrival. The hidden shares are not in the queue position the policy sees;
  whatever would fill the order under EXACT (an execution at its price or through it, an opposite order crossing it)
  first consumes them, since an aggressor reaching the order meets them first. They live in slot ST_HIDDEN of an agent
  state (invex.env.execution.S_HIDDEN) and are set when the order is placed there.
* MARKOUT_BASE + MARKOUT_ZERO + m (another planted control): EXACT, and every passive fill also books its markout,
  side x shares x (fill price - mid one decision interval after the fill), in slot ST_MARK (twice the price units, so
  that it stays an integer); the environment charges m/1000 times it (invex.env.execution.step). The interval is set
  by the environment in slot ST_TAU. Fills whose spread at the fill and mid one interval later are both quoted also
  add shares x spread at the fill to slot ST_MSPR, from which a stock's mean markout per passive share is measured.
* DMARK_BASE + 10^4 (1000 + 1000 mbar) + (5000 + 100 m) (a third planted control): as the markout rule, but each fill
  pays m/100 times its markout less mbar spreads at the fill, the stock's own mean markout per passive share, so that
  the stock's average price of resting is unchanged and only which fills cost more varies. Fills without a quoted
  spread at the fill or mid one interval later pay nothing.
* HABS_BASE + h (a fourth planted control): EXACT, plus h hidden shares ahead of every resting order at its price,
  wherever it rests (an improving order too); consumed as for HIDDEN_BASE.

All passive fills are at the order's limit price. Market orders walk the
visible book at arrival (`walk`); depth that is not there is charged at the
last level plus `penalty_ticks` and reported as a shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .. import schema as S
from ..book.derive import BOOK_KINDS, derive

EXACT, PESSIMISTIC, OPTIMISTIC = 0, 1, 2
HIDDEN_BASE = 100  # rule HIDDEN_BASE + h: EXACT with h/100 hidden shares ahead per displayed share (planted control)
MARKOUT_BASE, MARKOUT_ZERO = 10_000, 5_000  # rule MARKOUT_BASE + MARKOUT_ZERO + m: EXACT, fills charged m/1000 markouts
MARKOUT_END = 20_000
DMARK_BASE = 100_000_000  # rule DMARK_BASE + 10^4 (1000 + 1000 mbar) + 5000 + 100 m: demeaned markout (planted control)
HABS_BASE = 200_000_000  # rule HABS_BASE + h: h hidden shares ahead at every resting price (planted control)
RULE_SPAN = 100_000_000

# order state vector
ST_ACTIVE, ST_SIDE, ST_PX, ST_REM, ST_AHEAD, ST_INS, ST_NEXT, ST_LEV, ST_FILLED, ST_CASH, ST_FIRST = range(11)
ST_SIZE = 11  # ST_FIRST: row of the first fill (-1 before any)
ST_HIDDEN = ST_SIZE + 7  # hidden shares ahead, in an agent state only (= invex.env.execution.S_HIDDEN)
ST_TAU, ST_MARK = ST_SIZE + 8, ST_SIZE + 9  # markout horizon (ns) and booked markouts (agent state only)
ST_MSPR = ST_SIZE + 10  # shares x spread at the fill of the marked fills (agent state only)

PLACED, REJECTED_MARKETABLE = 0, 1


@dataclass
class ReplayBook:
    ts: np.ndarray
    kind: np.ndarray
    side: np.ndarray
    px: np.ndarray
    qty: np.ndarray
    oid: np.ndarray
    aux: np.ndarray
    flags: np.ndarray
    lvl: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    levels: np.ndarray  # distinct book prices, ascending
    lv_off: np.ndarray  # rows of level (sd, L) are lv_rows[lv_off[g]:lv_off[g+1]], g = sd * n_lev + L
    lv_rows: np.ndarray
    qa: dict

    @property
    def n(self) -> int:
        return self.ts.shape[0]

    def arrays(self) -> tuple:
        """Positional arguments for the compiled functions (in this order)."""
        return (self.ts, self.kind, self.side, self.px, self.qty, self.oid, self.aux, self.lvl, self.bid_px,
                self.bid_sz, self.ask_px, self.ask_sz, self.levels, self.lv_off, self.lv_rows)


def prepare(ev: np.ndarray, open_ns: int, close_ns: int) -> ReplayBook:
    ev = ev.copy()
    d = derive(ev, open_ns, close_ns)
    n_lev = max(1, d.levels.shape[0])
    book = np.flatnonzero(np.isin(ev["kind"], BOOK_KINDS))
    sd = (ev["side"][book] < 0).astype(np.int64)
    key = sd * n_lev + d.lev[book]
    order = np.argsort(key, kind="stable")
    lv_rows = book[order].astype(np.int64)
    lv_off = np.zeros(2 * n_lev + 1, dtype=np.int64)
    np.cumsum(np.bincount(key, minlength=2 * n_lev), out=lv_off[1:])
    levels = d.levels if d.levels.shape[0] else np.zeros(1, np.int64)
    c = lambda a: np.ascontiguousarray(a)  # noqa: E731
    return ReplayBook(c(ev["ts"]), c(ev["kind"]), c(ev["side"]), c(ev["px"]), c(ev["qty"]), c(ev["oid"]),
                      c(ev["aux"]), c(ev["flags"]), d.lvl, d.bid_px, d.bid_sz, d.ask_px, d.ask_sz, levels, lv_off, lv_rows, d.qa)


@nb.njit(inline="always")
def row_after(ts, t):
    """First row with ts > t (rows at or before t happened before an arrival at t)."""
    lo, hi = 0, ts.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(inline="always")
def level_index(levels, p):
    lo, hi = 0, levels.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if levels[mid] < p:
            lo = mid + 1
        else:
            hi = mid
    if lo < levels.shape[0] and levels[lo] == p:
        return lo
    return -1


@nb.njit(inline="always")
def size_at(lvl, lv_off, lv_rows, n_lev, sd, L, j):
    """Displayed shares at level (sd, L) after row j (0 if the level never had rows by then)."""
    if L < 0 or j < 0:
        return 0
    g = sd * n_lev + L
    lo, hi = lv_off[g], lv_off[g + 1]
    # last row <= j
    while lo < hi:
        mid = (lo + hi) // 2
        if lv_rows[mid] <= j:
            lo = mid + 1
        else:
            hi = mid
    if lo == lv_off[g]:
        return 0
    return lvl[lv_rows[lo - 1]]


@nb.njit
def place(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
          st, sgn, price, q, t_arr):
    """Rest a virtual limit order (sgn = +1 buy / -1 sell) arriving at t_arr. Post-only:
    an order that would execute on arrival is rejected."""
    i_ins = row_after(ts, t_arr)
    j = i_ins - 1
    if j >= 0:
        if sgn > 0 and ask_px[j] >= 0 and price >= ask_px[j]:
            return REJECTED_MARKETABLE
        if sgn < 0 and bid_px[j] >= 0 and price <= bid_px[j]:
            return REJECTED_MARKETABLE
    L = level_index(levels, price)
    sd = 0 if sgn > 0 else 1
    st[ST_ACTIVE] = 1
    st[ST_SIDE] = sgn
    st[ST_PX] = price
    st[ST_REM] = q
    st[ST_AHEAD] = size_at(lvl, lv_off, lv_rows, levels.shape[0], sd, L, j)
    st[ST_INS] = i_ins
    st[ST_NEXT] = i_ins
    st[ST_LEV] = L
    st[ST_FIRST] = -1
    return PLACED


@nb.njit
def advance(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
            st, i_end, rule):
    """Process rows [st[ST_NEXT], i_end) for a resting order; returns (shares filled, cash = sum px * shares)."""
    if st[ST_ACTIVE] == 0:
        return 0, 0
    sgn = st[ST_SIDE]
    p = st[ST_PX]
    rem = st[ST_REM]
    ahead = st[ST_AHEAD]
    ins = st[ST_INS]
    habs_rule = rule >= HABS_BASE and rule < HABS_BASE + RULE_SPAN
    hidden_rule = (rule >= HIDDEN_BASE and rule < MARKOUT_BASE) or habs_rule
    dmark_rule = rule >= DMARK_BASE and rule < DMARK_BASE + RULE_SPAN
    markout_rule = (rule >= MARKOUT_BASE and rule < MARKOUT_END) or dmark_rule
    m_milli = (rule - DMARK_BASE) // 10_000 - 1000 if dmark_rule else 0
    mspr = 0
    exact = rule == EXACT or rule >= HIDDEN_BASE
    hidden = st[ST_HIDDEN] if hidden_rule else 0
    mark = 0
    filled = 0
    i = st[ST_NEXT]
    while i < i_end and rem > 0:
        k = kind[i]
        f = 0
        if k == S.EXEC:
            if side[i] == sgn:
                xp = aux[i]  # execution price (ITCH 'C' may differ from the displayed price px)
                if px[i] == p and xp == p:
                    if rule == OPTIMISTIC:
                        f = qty[i]
                    elif exact:
                        if oid[i] >= ins or ahead <= 0:
                            f = qty[i]
                        else:
                            ahead -= qty[i]
                            if ahead < 0:
                                ahead = 0
                else:
                    if px[i] == p and exact and oid[i] < ins:
                        # executed away from its displayed price: it leaves the queue, the aggressor
                        # did not want the displayed price, so no fill for the virtual order
                        ahead -= qty[i]
                        if ahead < 0:
                            ahead = 0
                    if (sgn > 0 and xp < p) or (sgn < 0 and xp > p):
                        f = qty[i]  # trade-through
                    elif xp == p and rule != PESSIMISTIC and (ahead <= 0 or rule == OPTIMISTIC):
                        f = qty[i]  # a print at its price against non-displayed interest: displayed goes first
        elif k == S.CANCEL or k == S.DELETE or k == S.MOD_DEL:
            if exact and side[i] == sgn and px[i] == p and oid[i] < ins:
                ahead -= qty[i]
                if ahead < 0:
                    ahead = 0
        elif k == S.ADD or k == S.MOD_ADD:
            if side[i] == -sgn and ((sgn > 0 and px[i] <= p) or (sgn < 0 and px[i] >= p)):
                f = qty[i]  # an order on the other side crosses the virtual order: it would have traded
        elif k == S.HIDDEN:
            if rule != PESSIMISTIC and px[i] == p and (ahead <= 0 or rule == OPTIMISTIC) and i > 0:
                # side of a hidden print: at or below the best bid it was a sale into hidden bids
                if (sgn > 0 and bid_px[i - 1] >= 0 and p <= bid_px[i - 1]) or (sgn < 0 and ask_px[i - 1] >= 0 and p >= ask_px[i - 1]):
                    f = qty[i]
        if f > 0 and hidden > 0:  # planted hidden shares at its price trade before it, whatever fills it
            u = f if f < hidden else hidden
            hidden -= u
            f -= u
        if f > 0:
            if f > rem:
                f = rem
            if st[ST_FIRST] < 0:
                st[ST_FIRST] = i
            rem -= f
            filled += f
            if markout_rule:
                jm = row_after(ts, ts[i] + st[ST_TAU]) - 1
                if bid_px[jm] >= 0 and ask_px[jm] >= 0:
                    mk = sgn * f * (2 * p - bid_px[jm] - ask_px[jm])
                    quoted = i > 0 and bid_px[i - 1] >= 0 and ask_px[i - 1] >= 0
                    spr = ask_px[i - 1] - bid_px[i - 1] if quoted else 0
                    if dmark_rule:
                        if quoted:  # in thousandths: the markout less the stock's mean at this fill's spread
                            mark += 1000 * mk - 2 * m_milli * f * spr
                    else:
                        mark += mk
                    if quoted:
                        mspr += f * spr
        i += 1
    st[ST_NEXT] = i
    if markout_rule:
        st[ST_MARK] += mark
        st[ST_MSPR] += mspr
    st[ST_REM] = rem
    st[ST_AHEAD] = ahead
    if hidden_rule:
        st[ST_HIDDEN] = hidden
    st[ST_FILLED] += filled
    st[ST_CASH] += filled * p
    if rem == 0:
        st[ST_ACTIVE] = 0
    return filled, filled * p


@nb.njit
def walk(ts, kind, side, px, qty, oid, aux, lvl, bid_px, bid_sz, ask_px, ask_sz, levels, lv_off, lv_rows,
         sgn, q, j, max_levels, penalty_ticks, tick):
    """Market order for q shares (sgn = +1 buy) against the visible book after row j.

    Returns (shares filled from displayed depth, cash, levels used, shortfall);
    the shortfall is priced by the caller (e.g. last level +/- penalty_ticks * tick)."""
    n_lev = levels.shape[0]
    rem = q
    cash = 0
    used = 0
    last = -1
    if sgn > 0:
        if j < 0 or ask_px[j] < 0:
            return 0, 0, 0, q
        L = level_index(levels, ask_px[j])
        while rem > 0 and L < n_lev and used < max_levels:
            s = size_at(lvl, lv_off, lv_rows, n_lev, 1, L, j)
            if s > 0:
                take = s if s < rem else rem
                rem -= take
                cash += take * levels[L]
                used += 1
                last = levels[L]
            L += 1
        if rem > 0 and last >= 0:
            cash += rem * (last + penalty_ticks * tick)
    else:
        if j < 0 or bid_px[j] < 0:
            return 0, 0, 0, q
        L = level_index(levels, bid_px[j])
        while rem > 0 and L >= 0 and used < max_levels:
            s = size_at(lvl, lv_off, lv_rows, n_lev, 0, L, j)
            if s > 0:
                take = s if s < rem else rem
                rem -= take
                cash += take * levels[L]
                used += 1
                last = levels[L]
            L -= 1
        if rem > 0 and last >= 0:
            cash += rem * (last - penalty_ticks * tick)
    return q - rem, cash, used, rem


def new_state() -> np.ndarray:
    return np.zeros(ST_SIZE, dtype=np.int64)

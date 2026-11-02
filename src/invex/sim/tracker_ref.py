"""Reference virtual-order tracker: plain Python, explicit queues, raw messages.

It reads the toy/ITCH message tuples of invex.sim.toy (not the event tables),
keeps every price level as a FIFO list of real orders, and inserts the virtual
order into that list as a marker at its arrival. Fills follow the EXACT rule of
invex.sim.tracker, stated in terms of queue positions instead of row numbers:
an execution at the marker's level fills it when the order hit is behind the
marker or nothing real is in front of it; trade-throughs, crossing adds and
prints at its price against non-displayed interest as in the fast tracker.
Agreement with the fast tracker checks the annotate pass, the `oid` trick and
the share bookkeeping together.
"""

from __future__ import annotations

AGENT = "AGENT"


def reference_fills(msgs: list[tuple], sgn: int, price: int, qty: int, t_arr: int) -> list[tuple[int, int]] | None:
    """Fills (ts, shares) of a virtual limit order arriving at t_arr; None if rejected (marketable)."""
    levels: dict[tuple[int, int], list] = {}
    orders: dict[int, tuple[int, int]] = {}
    rem = {}
    placed = False
    left = qty
    fills: list[tuple[int, int]] = []

    def best(side):
        ps = [p for (s, p), q in levels.items() if s == side and any(e != AGENT for e in q)]
        if not ps:
            return None
        return max(ps) if side > 0 else min(ps)

    def at_front():
        q = levels[(sgn, price)]
        return q.index(AGENT) == 0

    def behind(ref):
        q = levels[(sgn, price)]
        return q.index(ref) > q.index(AGENT)

    def remove(ref, shares):
        side, p = orders[ref]
        rem[ref] -= shares
        if rem[ref] <= 0:
            levels[(side, p)].remove(ref)
            del orders[ref], rem[ref]

    def fill(ts, shares):
        nonlocal left
        f = min(left, shares)
        if f > 0:
            left -= f
            fills.append((ts, f))

    for m in msgs:
        t, ts = m[0], m[1]
        if not placed and ts > t_arr:
            bb, ba = best(1), best(-1)
            if (sgn > 0 and ba is not None and price >= ba) or (sgn < 0 and bb is not None and price <= bb):
                return None
            levels.setdefault((sgn, price), []).append(AGENT)
            placed = True
        if t == "A":
            ref, side, sh, p = m[2], 1 if m[3] == "B" else -1, m[4], m[5]
            if placed and left > 0 and side == -sgn and ((sgn > 0 and p <= price) or (sgn < 0 and p >= price)):
                fill(ts, sh)
            levels.setdefault((side, p), []).append(ref)
            orders[ref] = (side, p)
            rem[ref] = sh
        elif t in ("E", "C"):
            ref, sh = m[2], m[3]
            side, p = orders[ref]
            xp = p if t == "E" else m[6]
            if placed and left > 0 and side == sgn:
                if p == price and xp == price:
                    if behind(ref) or at_front():
                        fill(ts, sh)
                elif (sgn > 0 and xp < price) or (sgn < 0 and xp > price):
                    fill(ts, sh)
                elif xp == price and p != price and at_front():
                    fill(ts, sh)
            remove(ref, sh)
        elif t == "X":
            remove(m[2], m[3])
        elif t == "D":
            remove(m[2], rem[m[2]])
        elif t == "U":
            old, new, sh, p = m[2], m[3], m[4], m[5]
            side = orders[old][0]
            remove(old, rem[old])
            if placed and left > 0 and side == -sgn and ((sgn > 0 and p <= price) or (sgn < 0 and p >= price)):
                fill(ts, sh)
            levels.setdefault((side, p), []).append(new)
            orders[new] = (side, p)
            rem[new] = sh
        elif t == "P":
            p = m[4]
            if placed and left > 0 and p == price and at_front():
                bb, ba = best(1), best(-1)
                if (sgn > 0 and bb is not None and price <= bb) or (sgn < 0 and ba is not None and price >= ba):
                    fill(ts, m[3])
    return fills

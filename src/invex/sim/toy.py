"""A small FIFO matching engine that emits ITCH-style messages (for tests and debugging).

`toy_stream(seed, n)` returns a list of message tuples:

    ("A", ts, ref, side, shares, price)          side 'B' / 'S'
    ("E", ts, ref, shares, match)                 execution at the displayed price
    ("C", ts, ref, shares, match, printable, price)
    ("X", ts, ref, shares)  ("D", ts, ref)  ("U", ts, old, new, shares, price)
    ("P", ts, side, shares, price, match)         hidden execution (side always 'B', as Nasdaq reports it)

Limit orders rest without crossing; market orders sweep the other side in
price-time priority (one E per resting order hit, same timestamp); some
executions are reported with a price away from the displayed one (as
discretionary orders are); hidden prints happen at the touch. `encode`
turns the messages into an ITCH 5.0 binary stream for one stock.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..itch import spec as M

TICK = 100  # $0.01 in ITCH units


class _Book:
    def __init__(self):
        self.levels = {1: {}, -1: {}}  # side -> price -> deque([ref, shares])
        self.orders = {}  # ref -> (side, price)

    def best(self, side: int):
        lv = [p for p, q in self.levels[side].items() if q]
        if not lv:
            return None
        return max(lv) if side > 0 else min(lv)

    def add(self, ref, side, price, shares):
        self.levels[side].setdefault(price, deque()).append([ref, shares])
        self.orders[ref] = (side, price)

    def take(self, ref, shares):
        side, price = self.orders[ref]
        q = self.levels[side][price]
        for e in q:
            if e[0] == ref:
                e[1] -= shares
                if e[1] <= 0:
                    q.remove(e)
                    del self.orders[ref]
                return
        raise KeyError(ref)

    def rem(self, ref):
        side, price = self.orders[ref]
        return next(e[1] for e in self.levels[side][price] if e[0] == ref)


def toy_stream(seed: int, n: int = 2000, t0: int = 9 * 3600 * 10**9, mid: int = 1_000_000) -> list[tuple]:
    rng = np.random.default_rng(seed)
    book = _Book()
    out: list[tuple] = []
    ref, match, ts = 0, 0, t0
    for _ in range(n):
        if rng.random() > 0.2:  # otherwise same timestamp as the previous message
            ts += int(rng.exponential(2e6)) + 1
        u = rng.random()
        bb, ba = book.best(1), book.best(-1)
        if u < 0.45 or not book.orders:  # limit order
            side = 1 if rng.random() < 0.5 else -1
            ref_px = (bb if side > 0 else ba) or (mid - TICK if side > 0 else mid + TICK)
            p = ref_px + side * TICK * int(rng.integers(-3, 2))
            if side > 0 and ba is not None and p >= ba:
                p = ba - TICK
            if side < 0 and bb is not None and p <= bb:
                p = bb + TICK
            ref += 1
            sh = 100 * int(rng.integers(1, 5))
            book.add(ref, side, p, sh)
            out.append(("A", ts, ref, "B" if side > 0 else "S", sh, p))
        elif u < 0.65:  # cancel part or all of a random order
            r = int(rng.choice(list(book.orders)))
            left = book.rem(r)
            if left > 100 and rng.random() < 0.5:
                c = 100 * int(rng.integers(1, left // 100))
                book.take(r, c)
                out.append(("X", ts, r, c))
            else:
                book.take(r, left)
                out.append(("D", ts, r))
        elif u < 0.80:  # market order sweeping the other side
            side = 1 if rng.random() < 0.5 else -1  # aggressor buys (1) or sells (-1)
            want = 100 * int(rng.integers(1, 8))
            while want > 0:
                best = book.best(-side)
                if best is None:
                    break
                r, left = book.levels[-side][best][0]
                q = min(left, want)
                match += 1
                book.take(r, q)
                out.append(("E", ts, r, q, match))
                want -= q
        elif u < 0.87:  # replace a random order: new reference, new price, back of the queue
            r = int(rng.choice(list(book.orders)))
            side, p = book.orders[r]
            left = book.rem(r)
            book.take(r, left)
            p2 = p + side * TICK * int(rng.integers(-2, 2))
            bb, ba = book.best(1), book.best(-1)
            if side > 0 and ba is not None and p2 >= ba:
                p2 = ba - TICK
            if side < 0 and bb is not None and p2 <= bb:
                p2 = bb + TICK
            ref += 1
            sh = 100 * int(rng.integers(1, 5))
            book.add(ref, side, p2, sh)
            out.append(("U", ts, r, ref, sh, p2))
        elif u < 0.92:  # execution away from the displayed price (discretion), any queue position
            r = int(rng.choice(list(book.orders)))
            side, p = book.orders[r]
            q = min(book.rem(r), 100)
            match += 1
            book.take(r, q)
            out.append(("C", ts, r, q, match, True, p + side * TICK))
        elif u < 0.95:  # hidden print at the touch
            if bb is not None and ba is not None:
                match += 1
                out.append(("P", ts, "B", 100, bb if rng.random() < 0.5 else ba, match))
        else:  # execution of the head order reported with its own price
            side = 1 if rng.random() < 0.5 else -1
            best = book.best(side)
            if best is not None:
                r, left = book.levels[side][best][0]
                q = min(left, 100)
                match += 1
                book.take(r, q)
                out.append(("C", ts, r, q, match, True, best))
    return out


def encode(msgs: list[tuple], locate: int = 1, stock: str = "TST") -> bytes:
    f = M.frame
    parts = [f(M.msg_S(0, "O")), f(M.msg_R(locate, 1, stock))]
    for m in msgs:
        t, ts = m[0], m[1]
        if t == "A":
            parts.append(f(M.msg_A(locate, ts, m[2], m[3], m[4], stock, m[5])))
        elif t == "E":
            parts.append(f(M.msg_E(locate, ts, m[2], m[3], m[4])))
        elif t == "C":
            parts.append(f(M.msg_C(locate, ts, m[2], m[3], m[4], m[5], m[6])))
        elif t == "X":
            parts.append(f(M.msg_X(locate, ts, m[2], m[3])))
        elif t == "D":
            parts.append(f(M.msg_D(locate, ts, m[2])))
        elif t == "U":
            parts.append(f(M.msg_U(locate, ts, m[2], m[3], m[4], m[5])))
        elif t == "P":
            parts.append(f(M.msg_P(locate, ts, m[2], m[3], stock, m[4], m[5])))
    return b"".join(parts)

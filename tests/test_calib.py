"""Calibration windows (invex.book.calib) and unit systems (invex.env.units) on a hand-built day."""

import math

import numpy as np
import pytest

from invex import schema as S
from invex.book.calib import calibrate
from invex.env.units import fit_ko_constants, relative_tick, units
from invex.sim.tracker import prepare

NS = 10**9
OPEN = 9 * 3600 * NS + 30 * 60 * NS
CLOSE = OPEN + 600 * NS  # a 10-minute session keeps the test small


def day():
    rows = [
        (OPEN - 300 * NS, S.SNAP, 1, 100_000, 300, 0, 0, 0),  # bid 10.00 x 300
        (OPEN - 300 * NS, S.SNAP, -1, 100_200, 200, 1, 0, 0),  # ask 10.02 x 200: spread 2 ticks
        (OPEN + 100 * NS, S.ADD, 1, 100_100, 100, 2, 0, 0),  # bid 10.01: spread 1 tick from t = 100 s
        (OPEN + 200 * NS, S.EXEC, -1, 100_200, 50, 1, 100_200, S.F_PRINTABLE),  # buyer lifts 50 + 50 (one trade)
        (OPEN + 200 * NS, S.EXEC, -1, 100_200, 50, 1, 100_200, S.F_PRINTABLE),
        (OPEN + 300 * NS, S.HIDDEN, 1, 100_150, 100, -1, 100_150, S.F_PRINTABLE),  # hidden print: a trade too
        (OPEN + 400 * NS, S.DELETE, 1, 100_100, 100, 2, 0, 0),  # back to 2 ticks
    ]
    ev = np.zeros(len(rows), dtype=S.EVENT_DTYPE)
    for i, r in enumerate(rows):
        ev[i] = r
    return ev


def test_window_integrals():
    book = prepare(day(), OPEN, CLOSE)
    cal = calibrate(book, OPEN, CLOSE)
    w = cal.window(CLOSE, 600)
    assert w.seconds == 600
    assert w.trades == 2  # one aggressor run + one hidden print
    assert w.volume == 200
    assert w.vwap == pytest.approx((100 * 100_200 + 100 * 100_150) / 200)
    # spread: 200 for [0,100) s, 100 for [100,400), 200 for [400,600)
    assert w.spread == pytest.approx((200 * 100 + 100 * 300 + 200 * 200) / 600)
    assert w.onetick == pytest.approx(300 / 600)
    # touch depth (bid + ask) / 2: (300+200)/2 for 100 s, (100+200)/2 for 100 s, (100+100)/2 for 200 s, (300+100)/2 after
    want = (250 * 100 + 150 * 100 + 100 * 200 + 200 * 200) / 600
    assert w.depth == pytest.approx(want)
    # a window that ends at 150 s sees only the first 150 s and no trades
    w2 = cal.window(OPEN + 150 * NS, 150)
    assert w2.trades == 0 and w2.spread == pytest.approx((200 * 100 + 100 * 50) / 150)
    assert w.mid == pytest.approx((100_000 + 100_200) / 2)


def test_units_scale_like_the_hypotheses():
    book = prepare(day(), OPEN, CLOSE)
    w = calibrate(book, OPEN, CLOSE).window(CLOSE, 600)
    w.sigma_1min = 0.001  # the toy day is too short for 1-minute returns
    u3 = units(w, "R3")
    assert (u3.time, u3.size, u3.price) == pytest.approx((300.0, 100.0, w.spread))
    u0 = units(w, "R0")
    assert (u0.time, u0.size, u0.price) == (1.0, 100.0, 100.0)
    # R2: doubling W divides the time and size units by 2^(2/3) and the price unit... recomputed from W
    import copy
    w2 = copy.copy(w)
    w2.volume *= 8  # W x 8
    a, b = units(w, "R2"), units(w2, "R2")
    assert b.time / a.time == pytest.approx(8 ** (-2 / 3))
    assert b.size / a.size == pytest.approx(8 * 8 ** (-2 / 3))
    assert b.price / a.price == pytest.approx(8 ** (-1 / 3))
    eta_sig, eta_s = relative_tick(w)
    assert eta_s == pytest.approx(100 / w.spread)
    assert math.isfinite(eta_sig)
    c = fit_ko_constants([w, w2])
    assert c.c_t > 0 and c.c_q > 0 and c.c_p > 0

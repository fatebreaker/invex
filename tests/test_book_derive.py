"""Book replay (invex.book.derive): levels, best quotes, flags and the QA counters."""

import numpy as np

from invex import schema as S
from invex.book.derive import derive

from test_itch_parse import T, T_END, T_SNAP, run, rows_of, day_stream

OPEN, CLOSE = T(9, 30), T(16)


def table(rows):
    ev = np.zeros(len(rows), dtype=S.EVENT_DTYPE)
    for i, r in enumerate(rows):
        ev[i] = r
    return ev


def test_handcrafted_day_replays_cleanly():
    _, out = run(day_stream())
    aaa = table(rows_of(out, 0))
    d = derive(aaa, OPEN, CLOSE)
    assert d.qa["unresolved"] == 0 and d.qa["crossed"] == 0 and d.qa["del_mismatch"] == 0
    # E ref1 and C ref2 (at its displayed price) are queue fills; C ref7 at 10.01 vs displayed 10.00 is not
    assert d.qa["fifo_bad"] == 0 and d.qa["fifo_ok"] == 2 and d.qa["exec_away"] == 1 and d.qa["ts_back"] == 0
    # right after the snapshot: bid 10.00 x (60 + nothing else at 10.00), ask 10.05 x 150
    i_state = 3
    assert (d.bid_px[i_state], d.bid_sz[i_state], d.ask_px[i_state], d.ask_sz[i_state]) == (100_000, 60, 100_500, 150)
    # the last row is the closing cross: book at that point holds the ask (140 left) only
    assert d.ask_px[-1] == 100_500 and d.ask_sz[-1] == 140 and d.bid_px[-1] == 100_000 and d.bid_sz[-1] == 50
    # continuous flag: not before the open, not in the 10 s before the closing cross, not while halted
    cont = (aaa["flags"] & S.F_CONTINUOUS) > 0
    assert not cont[aaa["ts"] < OPEN].any()
    assert not cont[(aaa["ts"] >= T(11)) & (aaa["ts"] < T(11, 5))].any()
    assert cont[(aaa["ts"] >= T(9, 31)) & (aaa["ts"] < T(9, 40))].all()
    bbb = table(rows_of(out, 1))
    d = derive(bbb, OPEN, CLOSE)
    assert d.qa["negative"] == 1  # the over-execution of the only order


def test_fifo_violation_and_crossed_book_are_counted():
    t = T(10)
    rows = [
        (T_SNAP, S.SNAP, 1, 100_000, 100, 0, 0, 0),
        (T_SNAP, S.SNAP, 1, 100_000, 100, 1, 0, 0),
        (T_SNAP, S.SNAP, -1, 100_100, 100, 2, 0, 0),
        (t, S.EXEC, 1, 100_000, 100, 1, 100_000, S.F_PRINTABLE),  # hits the second order first
        (t + 1, S.ADD, 1, 100_200, 100, 4, 0, 0),  # bid above the ask: crossed
        (t + 2, S.DELETE, 1, 100_200, 50, 4, 0, 0),  # wrong share count (100 were left)
    ]
    d = derive(table(rows), OPEN, CLOSE)
    assert d.qa["fifo_bad"] == 1 and d.qa["fifo_ok"] == 0
    assert d.qa["crossed"] == 1
    assert d.qa["del_mismatch"] == 1
    # after the delete the best bid falls back to the remaining 10.00 order
    assert d.bid_px[-1] == 100_000 and d.bid_sz[-1] == 100


def test_trade_start_marks_first_execution_of_each_sweep():
    t = T(10)
    rows = [
        (T_SNAP, S.SNAP, -1, 100_100, 100, 0, 0, 0),
        (T_SNAP, S.SNAP, -1, 100_100, 100, 1, 0, 0),
        (T_SNAP, S.SNAP, -1, 100_200, 100, 2, 0, 0),
        (t, S.EXEC, -1, 100_100, 100, 0, 100_100, S.F_PRINTABLE),
        (t, S.EXEC, -1, 100_100, 100, 1, 100_100, S.F_PRINTABLE),
        (t, S.EXEC, -1, 100_200, 50, 2, 100_200, S.F_PRINTABLE),
        (t + 5, S.EXEC, -1, 100_200, 50, 2, 100_200, S.F_PRINTABLE),
    ]
    ev = table(rows)
    d = derive(ev, OPEN, CLOSE)
    starts = (ev["flags"] & S.F_TRADE_START) > 0
    assert starts.tolist() == [False, False, False, True, False, False, True]
    assert d.ask_px[-1] == -1 and d.qa["fifo_ok"] == 4
    assert d.lvl[3] == 100 and d.lvl[4] == 0  # level 10.01 empties after the second execution


def test_episodes_overlapping_a_halt_are_flagged():
    from invex.env.episodes import excluded
    from invex.sim.tracker import prepare

    _, out = run(day_stream())
    book = prepare(table(rows_of(out, 0)), OPEN, CLOSE)
    t0 = np.array([T(9, 35), T(10, 58), T(11, 2), T(11, 6)], np.int64)
    T_ns = np.array([60, 180, 60, 60], np.int64) * 10**9
    # quiet minute; spans the 11:00 halt; inside the halt; after the resumption (cross guard passed at 11:06)
    assert excluded(book, t0, T_ns).tolist() == [False, True, True, False]

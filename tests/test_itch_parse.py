"""ITCH 5.0 index + annotate on handcrafted messages, checked against the plain-Python reference."""

import numpy as np

from invex import schema as S
from invex.itch import spec as M
from invex.itch.annotate import annotate
from invex.itch.index import index_file, order_by_locate
from invex.itch.pydecode import iter_messages, reference_tables


def T(h, m=0, s=0, ns=0):
    return ((h * 60 + m) * 60 + s) * 1_000_000_000 + ns


T_SNAP, T_END = T(9, 25), T(16, 5)


def day_stream() -> bytes:
    f = M.frame
    msgs = [
        M.msg_S(T(3), "O"),
        M.msg_R(1, T(3, 0, 1), "AAA", "Q"),
        M.msg_R(2, T(3, 0, 1), "BBB", "G"),
        M.msg_R(3, T(3, 0, 1), "CCC", "Q", etp="Y"),
        # pre-market
        M.msg_A(1, T(4), 1, "B", 100, "AAA", 100_000),
        M.msg_A(1, T(4, 1), 2, "S", 200, "AAA", 100_500),
        M.msg_A(1, T(5), 3, "B", 300, "AAA", 100_000, mpid="ABCD"),
        M.msg_A(3, T(5), 11, "B", 10, "CCC", 50_000),
        M.msg_E(1, T(6), 1, 40, 1001),
        M.msg_X(1, T(6, 30), 2, 50),
        M.msg_U(1, T(7), 3, 4, 250, 99_900),
        M.msg_H(1, T(8), "AAA", "T"),
        M.msg_A(1, T(8), 5, "B", 100, "AAA", 100_000),
        M.msg_D(1, T(8, 30), 5),
        M.msg_A(2, T(9), 6, "S", 500, "BBB", 200_000),
        M.msg_h(2, T(9, 10), "BBB", "H", "Q"),
        # continuous session
        M.msg_h(2, T(9, 30), "BBB", "T", "Q"),
        M.msg_A(1, T(9, 31), 7, "B", 100, "AAA", 100_000),
        M.msg_h(2, T(9, 31), "BBB", "H", "B"),  # BX operational halt: not Nasdaq, ignored
        M.msg_E(1, T(9, 32), 1, 60, 1002),
        M.msg_C(1, T(9, 33), 7, 50, 1003, True, 100_100),
        M.msg_C(1, T(9, 34), 2, 10, 1004, False, 100_500),
        M.msg_X(1, T(9, 35), 4, 50),
        M.msg_U(1, T(9, 36), 4, 8, 300, 99_800),
        M.msg_U(1, T(9, 37), 8, 9, 100, 99_700),
        M.msg_D(1, T(9, 38), 9),
        M.msg_P(1, T(9, 39), "B", 70, "AAA", 100_200, 1005),
        M.msg_B(1, T(10), 1003),
        M.msg_E(2, T(10), 6, 600, 1006),  # more shares than are left: counted
        M.msg_H(1, T(11), "AAA", "H"),
        M.msg_J(1, T(11, 1), "AAA", 100_000, 105_000, 95_000),
        M.msg_H(1, T(11, 5), "AAA", "T"),
        M.msg_Y(1, T(11, 6), "AAA"),
        M.msg_E(1, T(12), 999, 10, 1007),  # unknown order: an error, skipped
        M.msg_Q(1, T(16), 1000, "AAA", 100_000, 1008, "C"),
        M.msg_A(1, T(16, 10), 10, "B", 100, "AAA", 100_000),  # after t_end: dropped
        M.msg_S(T(20), "C"),
    ]
    return b"".join(f(m) for m in msgs)


def run(data: bytes, sel=(1, 2, 3)):
    buf = np.frombuffer(data, dtype=np.uint8)
    idx = index_file(buf, T_SNAP, T_END)
    perm, start = order_by_locate(idx.loc, idx.counts)
    sel = np.array(sel)
    out = annotate(buf, idx.off, idx.typ, perm, start, sel, idx.nrows[sel], idx.max_ref, T_SNAP, T_END)
    return idx, out


def rows_of(out, j):
    a, n = out["row_start"][j], out["n_rows"][j]
    cols = [out[c][a:a + n] for c in S.COLUMNS]
    return [tuple(int(c[i]) for c in cols) for i in range(n)]


def test_index_counts_and_framing():
    data = day_stream()
    idx, _ = run(data)
    msgs = list(iter_messages(data))
    assert idx.n == len(msgs)
    assert idx.bad_len == 0 and idx.unknown == 0 and idx.tail == 0
    assert idx.max_ref == 11
    assert idx.counts[1, ord("A")] == 5 and idx.counts[1, ord("F")] == 1 and idx.counts[1, ord("U")] == 3
    assert idx.counts[0, ord("S")] == 2
    # offsets point at type bytes
    assert all(data[o] == ord(t) for o, (t, _d) in zip(idx.off, msgs))


def test_annotate_matches_reference():
    data = day_stream()
    idx, out = run(data)
    ref = reference_tables(data, T_SNAP, T_END)
    for j, loc in enumerate((1, 2, 3)):
        assert rows_of(out, j) == ref[loc], f"locate {loc}"
        assert out["n_rows"][j] <= idx.nrows[loc]


def test_snapshot_priority_and_semantics():
    data = day_stream()
    _, out = run(data)
    aaa = rows_of(out, 0)
    snaps = [r for r in aaa if r[1] == S.SNAP]
    # resting at 09:25: ref1 (60 left), ref2 (150 left), ref4 (replaced at 07:00, 250) in that priority
    assert [(r[2], r[3], r[4]) for r in snaps] == [(1, 100_000, 60), (-1, 100_500, 150), (1, 99_900, 250)]
    assert [r[5] for r in snaps] == [0, 1, 2]  # SNAP rows point to themselves
    # the trading state before the snapshot follows the SNAP rows
    assert aaa[3] == (T_SNAP, S.STATE, 0, 0, 0, -1, ord("T"), 0)
    # the replace chain: MOD_DEL of row 2's order, MOD_ADD that points to itself, then the next replace
    mods = [(i, r) for i, r in enumerate(aaa) if r[1] in (S.MOD_DEL, S.MOD_ADD)]
    (i1, d1), (i2, a1), (i3, d2), (i4, a2) = mods
    assert d1[5] == 2 and d1[4] == 200 and a1[5] == i2 and a1[2] == 1 and a1[3] == 99_800
    assert d2[5] == i2 and d2[4] == 300 and a2[5] == i4 and a2[3] == 99_700
    # non-printable C: execution flagged with price, not printable; printable C carries its price in aux
    cs = [r for r in aaa if r[1] == S.EXEC and r[7] & S.F_EXEC_PRICE]
    assert [(r[6], bool(r[7] & S.F_PRINTABLE)) for r in cs] == [(100_100, True), (100_500, False)]
    # nothing at or after t_end, and the closing cross is kept (16:00 < t_end)
    assert max(r[0] for r in aaa) < T_END
    assert any(r[1] == S.CROSS and r[6] == ord("C") and r[4] == 1000 for r in aaa)


def test_errors_and_overexecution_are_counted():
    _, out = run(day_stream())
    assert out["n_err"].tolist() == [1, 0, 0]
    assert out["n_over"].tolist() == [0, 1, 0]
    assert out["n_snap"].tolist() == [3, 1, 1]


def test_stock_with_only_premarket_messages_gets_a_snapshot():
    _, out = run(day_stream())
    ccc = rows_of(out, 2)
    assert ccc == [(T_SNAP, S.SNAP, 1, 50_000, 10, 0, 0, 0)]


def test_operational_halt_state_carried_into_snapshot():
    _, out = run(day_stream())
    bbb = rows_of(out, 1)
    states = [r[6] for r in bbb if r[1] == S.STATE]
    # halted before the snapshot (h 'H' on Nasdaq), resumed at 09:30; the BX halt is ignored
    assert states == [S.STATE_OPHALT + ord("H"), S.STATE_OPHALT + ord("T")]


def test_framing_errors_detected():
    data = day_stream()
    bad = data + M.frame(M.msg_D(1, T(20), 1) + b"\x00")  # 'D' with one byte too many
    bad += M.frame(b"Z" + b"\x00" * 10)  # unknown type
    bad += b"\x00\x23" + b"A" * 5  # truncated tail
    buf = np.frombuffer(bad, dtype=np.uint8)
    idx = index_file(buf, T_SNAP, T_END)
    assert idx.bad_len == 1 and idx.unknown == 1 and idx.tail == 7

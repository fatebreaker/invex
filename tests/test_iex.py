"""IEX DEEP+ capture reader and event mapping, on handcrafted pcap-ng captures."""

import struct

import numpy as np

from invex import schema as S
from invex.book.derive import derive
from invex.iex import pcap as P
from invex.iex.normalize import build_stock, hms, session

SYM = b"ZIEXT   "


def shb() -> bytes:
    body = struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)
    n = 12 + len(body)
    return struct.pack("<II", 0x0A0D0D0A, n) + body + struct.pack("<I", n)


def idb() -> bytes:
    body = struct.pack("<HHI", 1, 0, 65535)
    n = 12 + len(body)
    return struct.pack("<II", 1, n) + body + struct.pack("<I", n)


def epb(payload: bytes, vlan: bool = False) -> bytes:
    udp = struct.pack(">HHHH", 1000, 2000, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, b"\x17\xe2\x9b\x80",
                     b"\xe9\xd7\x15\x01") + udp
    eth = b"\x01" * 12 + ((b"\x81\x00\x00\x01") if vlan else b"") + b"\x08\x00" + ip
    pad = (-len(eth)) % 4
    body = struct.pack("<IIIII", 0, 0, 0, len(eth), len(eth)) + eth + b"\x00" * pad
    n = 12 + len(body)
    return struct.pack("<II", 6, n) + body + struct.pack("<I", n)


def segment(first_seq: int, msgs: list[bytes], session_id: int = 7, protocol: int = 0x8005) -> bytes:
    payload = b"".join(struct.pack("<H", len(m)) + m for m in msgs)
    hdr = struct.pack("<BBHIIHHqqq", 1, 0, protocol, 1, session_id, len(payload), len(msgs), 0, first_seq, 0)
    return hdr + payload


def ts(h, m=0, s=0, ns=0):
    return hms(h, m, s) + ns


def add(t, oid, side, size, px):
    return struct.pack("<cBq8sqIq", b"a", ord("8") if side > 0 else ord("5"), t, SYM, oid, size, px)


def modify(t, oid, size, px, keep):
    return struct.pack("<cBq8sqIq", b"M", 1 if keep else 0, t, SYM, oid, size, px)


def delete(t, oid):
    return struct.pack("<cBq8sq", b"R", 0, t, SYM, oid)


def execd(t, oid, size, px, tid, flags=0):
    return struct.pack("<cBq8sqIqq", b"L", flags, t, SYM, oid, size, px, tid)


def trade(t, size, px, tid, flags=0):
    return struct.pack("<cBq8sIqq", b"T", flags, t, SYM, size, px, tid)


def brk(t, size, px, tid):
    return struct.pack("<cBq8sIqq", b"B", 0, t, SYM, size, px, tid)


def status(t, code):
    return struct.pack("<cBq8s4s", b"H", ord(code), t, SYM, b"    ")


def system(t, code):
    return struct.pack("<cBq", b"S", ord(code), t)


def capture(segments: list[bytes]) -> np.ndarray:
    raw = shb() + idb() + b"".join(epb(s, vlan=(i % 2 == 1)) for i, s in enumerate(segments))
    return np.frombuffer(raw, np.uint8)


def test_scan_drops_duplicates_and_counts_gaps():
    m = [system(ts(9, 30), "R"), add(ts(9, 30, 1), 1, 1, 100, 990000), add(ts(9, 30, 2), 2, -1, 200, 1000000),
         delete(ts(9, 30, 3), 1)]
    segs = [segment(1, m[:2]), segment(1, m[:2]),  # the same segment twice (A/B feeds)
            segment(2, m[1:3]),  # overlaps by one message
            segment(5, m[3:4]),  # sequence 4 never arrives
            segment(9, [], protocol=0x8004), segment(6, [])]  # another protocol; a heartbeat
    sc = P.scan(capture(segs))
    assert sc.stats["messages"] == 4
    assert sc.stats["duplicates"] == 3
    assert sc.stats["gap_messages"] == 1
    assert sc.stats["other_packets"] == 1 and sc.stats["heartbeats"] == 1
    msg = P.decode(capture(segs), sc)
    assert list(msg["typ"]) == [ord("S"), ord("a"), ord("a"), ord("R")]
    assert msg["sym"][1] == P.symbol_code("ZIEXT") and P.symbol_name(msg["sym"][1]) == "ZIEXT"
    assert msg["oid"][2] == 2 and msg["size"][2] == 200 and msg["price"][2] == 1000000 and msg["code"][2] == ord("5")
    assert msg["ts"][3] == ts(9, 30, 3)


def stock_events(msgs: list[bytes]):
    buf = capture([segment(1, msgs)])
    msg = P.decode(buf, P.scan(buf))
    sess = session("2026-10-14")
    cols, qa = build_stock({k: msg[k] for k in ("typ", "code", "ts", "oid", "size", "price", "tid")},
                           sess["t_snap"], sess["t_end"])
    return cols, qa


def test_mapping_snapshot_modify_exec_hidden_break_clear():
    msgs = [
        add(ts(9, 0), 11, 1, 300, 990000),  # pre-market: collapsed into the 09:25 snapshot
        add(ts(9, 1), 12, 1, 100, 990000),
        add(ts(9, 2), 13, -1, 100, 1000000),
        modify(ts(9, 3), 11, 300, 990000, keep=False),  # priority reset before the snapshot: now behind 12
        add(ts(9, 30, 1), 14, -1, 200, 1000000),
        modify(ts(9, 30, 2), 14, 150, 1000000, keep=True),  # size cut, keeps priority -> CANCEL 50
        modify(ts(9, 30, 3), 13, 100, 1010000, keep=False),  # repriced -> MOD_DEL + MOD_ADD
        execd(ts(9, 30, 4), 14, 50, 1000000, tid=501),  # partial
        execd(ts(9, 30, 5), 14, 100, 999900, tid=502),  # rest, price improved
        trade(ts(9, 30, 6), 40, 995000, tid=503),  # non-displayed
        brk(ts(9, 30, 7), 40, 995000, tid=503),
        delete(ts(9, 30, 8), 12),
        status(ts(9, 31), "P"), status(ts(9, 36), "T"),
        delete(ts(9, 37), 999),  # unknown order: counted
        struct.pack("<cBq8s", b"C", 0, ts(9, 38), SYM),  # clear book: 11 and 13 leave
        add(ts(16, 10), 15, 1, 100, 990000),  # after the table's end
    ]
    cols, qa = stock_events(msgs)
    k = list(cols["kind"])
    assert k[:3] == [S.SNAP] * 3
    # snapshot in time priority: 12 (09:01), 13 (09:02), then 11 (reset at 09:03)
    assert list(cols["qty"][:3]) == [100, 100, 300] and list(cols["side"][:3]) == [1, -1, 1]
    assert (cols["ts"][:3] == ts(9, 25)).all() and list(cols["oid"][:3]) == [0, 1, 2]
    assert k[3:] == [S.ADD, S.CANCEL, S.MOD_DEL, S.MOD_ADD, S.EXEC, S.EXEC, S.HIDDEN, S.BREAK, S.DELETE, S.STATE,
                     S.STATE, S.DELETE, S.DELETE]
    assert cols["qty"][4] == 50 and cols["oid"][4] == 3  # CANCEL of 14's cut
    assert cols["oid"][5] == 1 and cols["oid"][6] == 6 and cols["px"][6] == 1010000  # 13 moved, new row
    assert cols["qty"][7] == 50 and cols["flags"][7] == S.F_PRINTABLE
    assert cols["aux"][8] == 999900 and cols["flags"][8] == S.F_PRINTABLE | S.F_EXEC_PRICE
    assert cols["flags"][9] & S.F_BROKEN and cols["px"][9] == cols["aux"][9] == 995000
    assert cols["aux"][10] == 503
    assert cols["oid"][11] == 0  # order 12, created by the first SNAP row
    assert list(cols["aux"][12:14]) == [ord("P"), ord("T")]
    assert list(cols["oid"][14:16]) == [2, 6] and list(cols["qty"][14:16]) == [300, 100]  # the clear: 11 and 13
    assert qa["unresolved"] == 1 and qa["clears"] == 1 and qa["breaks"] == 1 and qa["break_unmatched"] == 0
    # the replayed book is consistent: nothing unresolved or negative, no rows out of time order
    ev = np.empty(len(k), S.EVENT_DTYPE)
    for c in S.COLUMNS:
        ev[c] = cols[c]
    d = derive(ev, hms(9, 30), hms(16))
    assert d.qa["unresolved"] == 0 and d.qa["negative"] == 0 and d.qa["del_mismatch"] == 0 and d.qa["ts_back"] == 0


def test_cross_rows_and_clamped_times():
    msgs = [
        add(ts(9, 26), 1, -1, 100, 1000000),
        execd(ts(9, 30), 1, 60, 1000000, tid=1, flags=0x08),  # opening cross
        trade(ts(9, 30), 10, 1000000, tid=2, flags=0x08),  # same cross: no second CROSS row
        add(ts(9, 30, 5), 2, 1, 100, 990000),
        add(ts(9, 30, 4), 3, 1, 100, 980000),  # time earlier than the previous row: clamped
    ]
    cols, qa = stock_events(msgs)
    assert list(cols["kind"]) == [S.ADD, S.CROSS, S.EXEC, S.HIDDEN, S.ADD, S.ADD]
    assert qa["ts_clamped"] == 1 and (np.diff(cols["ts"]) >= 0).all()

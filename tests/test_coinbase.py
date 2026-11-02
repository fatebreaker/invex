"""Coinbase L3 messages -> event table (invex.coinbase.normalize): exact parsing, message semantics, book QA."""

import numpy as np

from invex import schema as S
from invex.book.derive import derive
from invex.coinbase.normalize import dec, normalize, ns_of_day
from datetime import datetime, timezone

DAY = "2026-09-01"
TICK = 1_000_000  # $0.01 in 1e-8 $: the tables store prices in ticks
LOT = 25_000_000  # 0.25 coins in 1e-8 coins (the gcd of the sizes below): ... and sizes in lots


def px(x: str) -> int:
    return dec(x) // TICK


def lots(x: str) -> int:
    return dec(x) // LOT


def test_exact_decimals_and_times():
    assert dec("0.00000001") == 1 and dec("5") == 500_000_000 and dec("78588.83") == 7_858_883_000_000
    assert dec(".5") == 50_000_000 and dec("0.123456789") == 12_345_678  # beyond 1e-8 is cut, not rounded
    d0 = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
    assert ns_of_day("2026-09-01T00:03:36.308096Z", d0) == 216_308_096_000
    assert ns_of_day("2026-09-01T00:00:00.000000123Z", d0) == 123
    assert ns_of_day("2026-08-31T23:59:59.5Z", d0) == -500_000_000


def t(us):
    return f"{DAY}T00:00:{us // 1_000_000:02d}.{us % 1_000_000:06d}Z"


def stream():
    snap = {"type": "full_snapshot", "sequence": 100, "time": t(0),
            "bids": [["99.99", "1", "b1"], ["99.99", "2", "b2"], ["99.98", "3", "b3"]],
            "asks": [["100.01", "1.5", "a1"], ["100.02", "2", "a2"]]}
    m = [
        {"type": "open", "sequence": 99, "time": t(1), "order_id": "old", "side": "buy", "price": "99.99",
         "remaining_size": "9"},  # before the snapshot: ignored
        {"type": "received", "sequence": 101, "time": t(10), "order_id": "x"},
        {"type": "open", "sequence": 102, "time": t(10), "order_id": "b4", "side": "buy", "price": "99.99",
         "remaining_size": "0.5"},
        # a market buy takes a1 (1.5) and 0.5 of a2
        {"type": "match", "sequence": 103, "time": t(20), "maker_order_id": "a1", "taker_order_id": "tk",
         "side": "sell", "price": "100.01", "size": "1.5"},
        {"type": "done", "sequence": 104, "time": t(20), "order_id": "a1", "side": "sell", "price": "100.01",
         "remaining_size": "0", "reason": "filled"},
        {"type": "match", "sequence": 105, "time": t(20), "maker_order_id": "a2", "taker_order_id": "tk",
         "side": "sell", "price": "100.02", "size": "0.5"},
        # b2 shrinks in place (keeps priority), b1 is repriced away (back of a new level)
        {"type": "change", "sequence": 106, "time": t(30), "order_id": "b2", "side": "buy", "reason": "modify_order",
         "old_price": "99.99", "new_price": "99.99", "old_size": "2", "new_size": "1.25"},
        {"type": "change", "sequence": 107, "time": t(31), "order_id": "b1", "side": "buy", "reason": "modify_order",
         "old_price": "99.99", "new_price": "99.97", "old_size": "1", "new_size": "1"},
        # b3 is modified through the market to buy 3 at 100.02: takes a2's 1.5 left, 1.5 rests at 100.02
        {"type": "match", "sequence": 108, "time": t(40), "maker_order_id": "a2", "taker_order_id": "b3",
         "side": "sell", "price": "100.02", "size": "1.5"},
        {"type": "done", "sequence": 109, "time": t(40), "order_id": "a2", "side": "sell", "price": "100.02",
         "remaining_size": "0", "reason": "filled"},
        {"type": "change", "sequence": 110, "time": t(40), "order_id": "b3", "side": "buy",
         "reason": "remainder_after_modification", "old_price": "99.98", "new_price": "100.02", "old_size": "3",
         "new_size": "1.5"},
        {"type": "open", "sequence": 111, "time": t(50), "order_id": "a3", "side": "sell", "price": "100.05",
         "remaining_size": "1"},
        # a market sell hits the moved b3 as a maker, then b2 cancels
        {"type": "match", "sequence": 112, "time": t(60), "maker_order_id": "b3", "taker_order_id": "tk2",
         "side": "buy", "price": "100.02", "size": "0.5"},
        {"type": "done", "sequence": 113, "time": t(70), "order_id": "b2", "side": "buy", "price": "99.99",
         "remaining_size": "1.25", "reason": "canceled"},
        {"type": "done", "sequence": 114, "time": t(71), "order_id": "tk2", "side": "sell", "reason": "filled"},
        # a gap of more than MAX_HOLE sequence numbers ends the stream
        {"type": "open", "sequence": 200, "time": t(80), "order_id": "late", "side": "buy", "price": "99.99",
         "remaining_size": "1"},
    ]
    return snap, m


def test_message_semantics_and_book():
    snap, msgs = stream()
    cols, qa = normalize(DAY, "TEST-USD", [(snap, t(0))], list(reversed(msgs)))  # applied in sequence order
    assert qa["unresolved"] == 0 and qa["done_mismatch"] == 0 and qa["change_mismatch"] == 0
    assert qa["gap_at_sequence"] == 200 and qa["holes"] == 0 and qa["taker_fills"] == 1 and qa["reseeds"] == 0
    k = cols["kind"]
    want = [S.SNAP] * 5 + [S.ADD, S.EXEC, S.EXEC, S.CANCEL, S.MOD_DEL, S.MOD_ADD,
                           S.MOD_DEL, S.EXEC, S.MOD_ADD, S.ADD, S.EXEC, S.DELETE]
    assert k.tolist() == want
    assert cols["qty"][8] == lots("0.75") and cols["px"][10] == px("99.97")  # shrink in place; repriced away
    assert cols["qty"][11] == lots("3") and cols["px"][13] == px("100.02") and cols["qty"][13] == lots("1.5")
    assert cols["oid"][15] == 13  # the fill of the moved order refers to its new row
    assert (cols["oid"][[0, 1, 2, 3, 4, 5, 10, 13, 14]] == [0, 1, 2, 3, 4, 5, 10, 13, 14]).all()
    ev = np.zeros(len(k), dtype=S.EVENT_DTYPE)
    for c in S.COLUMNS:
        ev[c] = cols[c]
    d = derive(ev, 0, 10**12, tick=qa["tick"], tick_floor=0)
    assert d.qa["unresolved"] == 0 and d.qa["negative"] == 0 and d.qa["del_mismatch"] == 0
    assert d.qa["crossed"] == 0 and d.qa["fifo_bad"] == 0 and d.qa["fifo_ok"] == 4
    # after everything: best bid 1.0 left of b3 at 100.02 vs ask 100.05
    assert d.bid_px[-1] == px("100.02") and d.bid_sz[-1] == lots("1") and d.ask_px[-1] == px("100.05")
    assert qa["tick"] == 1 and qa["tick_raw"] == TICK and qa["px_scale"] == 100.0  # 100 ticks per dollar
    assert qa["lot_raw"] == LOT and qa["qty_scale"] == 4.0  # 4 lots per coin
    assert (cols["aux"][cols["kind"] == S.EXEC] == [px("100.01"), px("100.02"), px("100.02"), px("100.02")]).all()


def test_no_snapshot():
    cols, qa = normalize(DAY, "TEST-USD", [], [])
    assert cols is None and qa["error"] == "no snapshot"


def test_reconnect_reseeds_from_the_next_snapshot():
    from invex.env.episodes import outages
    from invex.sim.tracker import prepare

    snap, msgs = stream()
    msgs = msgs[:-1]  # up to seq 114, then a reconnect: 115..149 are lost
    # during the outage: b4 (0.5 at 99.99) was cancelled, b1 (1 at 99.97) cut to 0.4, the moved b3 (1 left at
    # 100.02) repriced to 100.00, a3 untouched, and n1 (2 at 99.96) is new
    snap2 = {"type": "full_snapshot", "sequence": 150, "generated": True,
             "bids": [["100.00", "1", "b3"], ["99.97", "0.4", "b1"], ["99.96", "2", "n1"]],
             "asks": [["100.05", "1", "a3"]]}
    msgs += [
        {"type": "open", "sequence": 148, "time": t(90), "order_id": "zz", "side": "buy", "price": "99.90",
         "remaining_size": "1"},  # reflected in snap2 (already done by then): skipped
        {"type": "done", "sequence": 149, "time": t(91), "order_id": "zz", "side": "buy", "reason": "canceled",
         "remaining_size": "1"},
        {"type": "open", "sequence": 151, "time": t(95), "order_id": "a4", "side": "sell", "price": "100.04",
         "remaining_size": "0.3"},
        {"type": "match", "sequence": 152, "time": t(96), "maker_order_id": "b3", "taker_order_id": "tk3",
         "side": "buy", "price": "100.00", "size": "0.25"},
    ]
    cols, qa = normalize(DAY, "TEST-USD", [(snap2, t(99)), (snap, t(0))], msgs)
    assert qa["reseeds"] == 1 and qa["gap_at_sequence"] == 0 and qa["unresolved"] == 0
    assert (qa["reseed_deleted"], qa["reseed_cut"], qa["reseed_moved"], qa["reseed_added"]) == (1, 1, 1, 1)
    k = cols["kind"]
    i_out = int(np.flatnonzero((k == S.STATE) & (cols["aux"] == S.STATE_OUTAGE))[0])
    i_res = int(np.flatnonzero((k == S.STATE) & (cols["aux"] == ord("T")))[0])
    assert cols["ts"][i_out] == ns_of_day(t(71), datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc))
    assert cols["ts"][i_res] == ns_of_day(t(91), datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc))
    assert sorted(k[i_out + 1:i_res].tolist()) == sorted([S.DELETE, S.CANCEL, S.MOD_DEL, S.MOD_ADD, S.ADD])
    ev = np.zeros(len(k), dtype=S.EVENT_DTYPE)
    for c in S.COLUMNS:
        ev[c] = cols[c]
    d = derive(ev.copy(), 0, 10**12, tick=qa["tick"], tick_floor=0)
    assert d.qa["unresolved"] == 0 and d.qa["negative"] == 0 and d.qa["del_mismatch"] == 0 and d.qa["crossed"] == 0
    lot = qa["lot_raw"]
    assert d.bid_px[-1] == px("100.00") and d.bid_sz[-1] == dec("0.75") // lot and d.ask_px[-1] == px("100.04")
    book = prepare(ev, 0, 10**12)
    cont = (book.flags & S.F_CONTINUOUS) > 0
    assert not cont[i_out:i_res].any() and cont[i_res + 1:].all() and cont[:i_out].all()
    lo, hi = outages(book)
    assert lo.tolist() == [cols["ts"][i_out]] and hi.tolist() == [cols["ts"][i_res]]


def test_units_do_not_depend_on_the_venue_scales():
    """One market in ITCH units (1e-4 $, shares) and in Coinbase units (ticks of $0.01, 1e-8 coins): every
    unit system gives the same physical units, and the dimensionless quantities agree."""
    import pandas as pd

    from invex.env.units import SYSTEMS, unit_frame

    ko = {"c_t": 4.68, "c_q": 1.72e-4, "c_p": 0.028}
    itch = {"w_seconds": 1800.0, "w_trades": 900.0, "w_volume": 2e5, "w_vwap": 1_000_500.0, "w_mid": 1_000_000.0,
            "w_spread": 250.0, "w_depth": 3e3, "w_onetick": 0.3, "w_sigma_1min": 8e-4, "tick": 100}
    f_px, f_q = 1 / 100, 1e8  # ITCH price units per tick; coin units per share
    cb = {**itch, "w_volume": itch["w_volume"] * f_q, "w_vwap": itch["w_vwap"] * f_px, "w_mid": itch["w_mid"] * f_px,
          "w_spread": itch["w_spread"] * f_px, "w_depth": itch["w_depth"] * f_q, "tick": 1, "px_scale": 100.0,
          "qty_scale": 1e8}
    a = unit_frame(pd.DataFrame([itch]), ko)
    b = unit_frame(pd.DataFrame([cb]), ko)
    for u in SYSTEMS:
        ua, ub = a[u].iloc[0], b[u].iloc[0]
        assert np.isclose(ua.t, ub.t) and np.isclose(ua.q, ub.q / f_q) and np.isclose(ua.p, ub.p / f_px), u
    assert np.isclose(a["_eta_sigma"].iloc[0], b["_eta_sigma"].iloc[0])
    assert np.isclose(a["_activity"].iloc[0], b["_activity"].iloc[0])


def test_a_single_lost_message_is_bridged_by_the_next_snapshot():
    snap, msgs = stream()
    msgs = msgs[:-1]
    # seq 115 is lost (b3's cancel); 116 arrives, then a snapshot without b3
    snap2 = {"type": "full_snapshot", "sequence": 117, "generated": True,
             "bids": [["99.97", "1", "b1"]], "asks": [["100.05", "1", "a3"]]}
    msgs += [{"type": "received", "sequence": 116, "time": t(80), "order_id": "q"},
             {"type": "open", "sequence": 118, "time": t(85), "order_id": "a5", "side": "sell", "price": "100.03",
              "remaining_size": "1"}]
    cols, qa = normalize(DAY, "TEST-USD", [(snap, t(0)), (snap2, t(82))], msgs)
    assert qa["reseeds"] == 1 and qa["holes"] == 0 and qa["reseed_deleted"] == 2  # b3 and b4 are gone
    ev = np.zeros(len(cols["kind"]), dtype=S.EVENT_DTYPE)
    for c in S.COLUMNS:
        ev[c] = cols[c]
    d = derive(ev, 0, 10**12, tick=1, tick_floor=0)
    assert d.qa["crossed"] == 0 and d.bid_px[-1] == px("99.97") and d.ask_px[-1] == px("100.03")

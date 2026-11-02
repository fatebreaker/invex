"""Coinbase L3 messages -> invex event tables (same schema as ITCH).

    python -m invex.coinbase.normalize --days 2026-09-01 --pairs BTC-USD ETH-USD [--minutes 120]

Per pair-day: the book is seeded from the first full_snapshot (SNAP rows in its order: price levels best first,
orders in time priority within a level); only messages with a sequence number above the snapshot's are applied,
in sequence order. Mapping:

    open                        -> ADD (the resting remainder of a new order)
    match                       -> EXEC of the maker order (aux = trade price)
    done                        -> DELETE of what is left (nothing after fills)
    change, same price, smaller -> CANCEL (keeps priority; modify_order or self-trade prevention)
    change, otherwise           -> MOD_DEL + MOD_ADD (repriced or enlarged: back of the queue)

An order modified through the market arrives as its taker matches first, then a change with reason
remainder_after_modification (new_size = what is left to rest) or a done: the old order leaves at the first
match (MOD_DEL, whole) and the remainder joins its new level at the change (MOD_ADD). Each change's old_size
must equal the size we track (change_mismatch counts the exceptions), as must each done's remaining_size.

Messages about orders that never rested (takers, rejected posts) are ignored. A gap in sequence numbers (a
websocket reconnect; even one lost message can leave a stale order) is bridged by the reconnect's snapshot: the outage is marked (STATE_OUTAGE ... 'T': not
continuous, and no episode may overlap it) and the book is reconciled with the snapshot (see `normalize`).
Prices and sizes are parsed exactly into integers of 1e-8 dollars and 1e-8 coins, then prices are divided by the
pair-day's tick (the gcd of all its prices) and sizes by its lot (the gcd of all its sizes): the table's price unit
is one tick (tick = 1, px_scale = ticks per dollar, tick_raw = the tick in 1e-8 dollars) and its size unit one lot
(qty_scale = lots per coin, lot_raw = the lot in 1e-8 coins), so sizes, level sums and price x size stay far below
int64 limits. Times are ns since 00:00 UTC.
Output: $INVEX_DATA/coinbase/<day>/{events.parquet (one row group per pair), symbols.parquet, qa.json}.
"""

from __future__ import annotations

import argparse
import gzip
import math
import time
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from .. import schema as S
from ..io.store import EventWriter, day_dir
from ..paths import BUDGET_GB, DATA, ensure_space
from ..provenance import provenance, write_json
from .tardis import minute_path

PX_SCALE = 100_000_000  # price units per dollar
MAX_HOLE = 5  # missing sequence numbers tolerated (untracked message types); more means a reconnect
QTY_SCALE = 100_000_000  # size units per coin


def dec(s: str) -> int:
    """Exact decimal string -> integer in 1e-8 units."""
    if "." in s:
        w, f = s.split(".", 1)
        return int(w or "0") * 100_000_000 + int((f + "00000000")[:8])
    return int(s) * 100_000_000


def ns_of_day(iso: str, day0: datetime) -> int:
    """'2026-09-01T00:00:00.014347Z' (micro- or nanoseconds, Z optional) -> ns since 00:00 UTC of day0;
    times on the previous day come out negative."""
    ns = ((int(iso[11:13]) * 60 + int(iso[14:16])) * 60 + int(iso[17:19])) * 1_000_000_000
    if len(iso) > 20 and iso[19] == ".":
        frac = iso[20:].rstrip("Z")
        ns += int((frac + "000000000")[:9])
    if iso[:10] != day0.strftime("%Y-%m-%d"):
        d = datetime.fromisoformat(iso[:10]).replace(tzinfo=timezone.utc)
        ns += int((d - day0).total_seconds()) * 1_000_000_000
    return ns


def messages(day: str, pair: str, minutes: int):
    """([(snapshot dict, local time)], messages) from the stored minute files, in file order. Tardis marks a
    websocket reconnect with an empty line; the reconnect's own snapshot follows a few seconds later."""
    snaps, out = [], []
    for m in range(minutes):
        p = minute_path(day, pair, m)
        if not p.exists():
            break
        with gzip.open(p, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                sp = line.index(b" ")
                msg = orjson.loads(line[sp + 1:])
                if msg.get("type") == "full_snapshot":
                    snaps.append((msg, line[:sp].decode()))
                    continue
                out.append(msg)
    return snaps, out


def build_pair(day: str, pair: str, minutes: int) -> tuple[dict | None, dict]:
    snaps, msgs = messages(day, pair, minutes)
    return normalize(day, pair, snaps, msgs)


def normalize(day: str, pair: str, snaps: list, msgs: list[dict]) -> tuple[dict | None, dict]:
    """One pair-day's messages -> (event columns, QA); (None, QA with an error) without a snapshot.

    `snaps` is [(snapshot, local receive time)]. A gap of more than MAX_HOLE sequence numbers (a reconnect)
    opens an outage (STATE row, code STATE_OUTAGE): messages are skipped up to the next snapshot, the book
    is reconciled with it (DELETE what is gone, CANCEL size cuts, MOD_DEL + MOD_ADD for repriced or enlarged
    orders, ADD what is new, in the snapshot's priority order) and a 'T' row ends the outage. Without a
    later snapshot the stream ends at the gap."""
    day0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    qa = {"pair": pair, "messages": len(msgs)}
    if not snaps:
        qa["error"] = "no snapshot"
        return None, qa
    seq_of = lambda m: int(m["sequence"])  # noqa: E731
    snaps = sorted(((int(sn["sequence"]), sn, loc) for sn, loc in snaps), key=lambda x: x[0])
    s0, snap, snap_local = snaps[0]
    msgs = sorted(msgs, key=seq_of)
    rows = {k: [] for k in S.COLUMNS}

    def emit(ts, kind, side, px, qty, oid, aux=0, flags=0):
        rows["ts"].append(ts)
        rows["kind"].append(kind)
        rows["side"].append(side)
        rows["px"].append(px)
        rows["qty"].append(qty)
        rows["oid"].append(oid)
        rows["aux"].append(aux)
        rows["flags"].append(flags)
        return len(rows["ts"]) - 1

    def seed_time(i_next: int, sn_seq: int, local: str, floor: int | None) -> tuple[int, int]:
        """(time of the snapshot's state, index of the first message after it): the exchange time of the last
        message the snapshot already reflects (its own time is only Tardis's receive time), kept no later than
        the next message and no earlier than `floor`."""
        i, t_last = i_next, None
        while i < len(msgs) and seq_of(msgs[i]) <= sn_seq:
            t_last = msgs[i]["time"]
            i += 1
        t = ns_of_day(t_last, day0) if t_last else ns_of_day(local, day0)
        if i < len(msgs):
            t = min(t, ns_of_day(msgs[i]["time"], day0))
        return (t if floor is None else max(t, floor)), i

    book: dict[str, list] = {}  # order id -> [creation row, side, px, remaining]
    t_snap, i = seed_time(0, s0, snap_local, None)
    for sd, levels in ((1, snap["bids"]), (-1, snap["asks"])):
        for price, size, oid_ in levels:
            r = len(rows["ts"])
            emit(t_snap, S.SNAP, sd, dec(price), dec(size), r)
            book[oid_] = [r, sd, dec(price), dec(size)]
    n_snap = len(rows["ts"])
    unresolved = mismatch = change_mismatch = gap_at = holes = taker_fills = ts_clamped = 0
    rs = {"reseeds": 0, "outage_ns": 0, "reseed_deleted": 0, "reseed_cut": 0, "reseed_moved": 0, "reseed_added": 0}
    moving: set[str] = set()  # orders between their taker matches and the change that re-rests them
    prev = s0
    end_ts = t_snap

    def reseed(sn, ts):
        want, order = {}, []
        for sd, levels in ((1, sn["bids"]), (-1, sn["asks"])):
            for price, size, oid_ in levels:
                want[oid_] = (sd, dec(price), dec(size))
                order.append(oid_)
        moved = set()
        for oid_, o in list(book.items()):
            w = want.get(oid_)
            if w is None:
                emit(ts, S.DELETE, o[1], o[2], o[3], o[0])
                del book[oid_]
                rs["reseed_deleted"] += 1
            elif w[1] != o[2] or w[2] > o[3]:
                emit(ts, S.MOD_DEL, o[1], o[2], o[3], o[0])
                del book[oid_]
                moved.add(oid_)
            elif w[2] < o[3]:
                emit(ts, S.CANCEL, o[1], o[2], o[3] - w[2], o[0])
                o[3] = w[2]
                rs["reseed_cut"] += 1
        for oid_ in order:  # new and moved orders join the back of their levels, in priority order
            if oid_ not in book:
                sd, px, q = want[oid_]
                r = len(rows["ts"])
                emit(ts, S.MOD_ADD if oid_ in moved else S.ADD, sd, px, q, r)
                book[oid_] = [r, sd, px, q]
                rs["reseed_moved" if oid_ in moved else "reseed_added"] += 1
        moving.clear()

    while i < len(msgs):
        m = msgs[i]
        seq = seq_of(m)
        if seq <= prev:
            i += 1
            continue
        if seq > prev + 1:
            # lost messages (a reconnect; even one lost cancel leaves a stale order): reseed from the next snapshot
            nxt = next(((q, sn, loc) for q, sn, loc in snaps if q > prev), None)
            if nxt is None and seq - prev - 1 > MAX_HOLE:
                gap_at = seq
                break
            if nxt is not None:
                q1, sn1, loc1 = nxt
                t_out = end_ts
                emit(t_out, S.STATE, 0, 0, 0, -1, S.STATE_OUTAGE)
                t_rs, i = seed_time(i, q1, loc1, t_out)
                reseed(sn1, t_rs)
                emit(t_rs, S.STATE, 0, 0, 0, -1, ord("T"))
                rs["reseeds"] += 1
                rs["outage_ns"] += t_rs - t_out
                prev, end_ts = q1, t_rs
                continue
            holes += 1  # no snapshot left to reseed from: carry on through a small hole, and count it
        prev = seq
        i += 1
        t = m["type"]
        if t == "received":
            continue
        ts = ns_of_day(m["time"], day0)
        if ts < end_ts:  # exchange times out of sequence order: keep the table monotone, count it
            ts_clamped += 1
            ts = end_ts
        end_ts = ts
        if t == "open":
            sd = 1 if m["side"] == "buy" else -1
            px, q = dec(m["price"]), dec(m["remaining_size"])
            r = len(rows["ts"])
            emit(ts, S.ADD, sd, px, q, r)
            book[m["order_id"]] = [r, sd, px, q]
        elif t == "match":
            q, xp = dec(m["size"]), dec(m["price"])
            tk_ = book.pop(m["taker_order_id"], None)
            if tk_ is not None:
                # a resting order modified through the market: the matches come first, then a change
                # (remainder_after_modification) or a done; the old order leaves the book now, whole
                emit(ts, S.MOD_DEL, tk_[1], tk_[2], tk_[3], tk_[0])
                moving.add(m["taker_order_id"])
                taker_fills += 1
            o = book.get(m["maker_order_id"])
            if o is None:
                unresolved += 1
                continue
            emit(ts, S.EXEC, o[1], o[2], q, o[0], xp, S.F_PRINTABLE | (S.F_EXEC_PRICE if xp != o[2] else 0))
            o[3] -= q
            if o[3] <= 0:
                del book[m["maker_order_id"]]
        elif t == "done":
            moving.discard(m["order_id"])  # modified through the market and filled: nothing rests
            o = book.pop(m["order_id"], None)
            if o is None:
                continue
            left = dec(m.get("remaining_size") or "0")
            if left != o[3]:
                mismatch += 1
            if o[3] > 0:
                emit(ts, S.DELETE, o[1], o[2], o[3], o[0])
        elif t == "change":
            oid_ = m["order_id"]
            if oid_ in moving:  # the remainder of an order modified through the market joins its new level
                moving.discard(oid_)
                sd, new_px, new_q = (1 if m["side"] == "buy" else -1), dec(m["new_price"]), dec(m["new_size"])
                if new_q > 0:
                    r = len(rows["ts"])
                    emit(ts, S.MOD_ADD, sd, new_px, new_q, r)
                    book[oid_] = [r, sd, new_px, new_q]
                continue
            o = book.get(oid_)
            if o is None:
                continue
            if m.get("old_size") and dec(m["old_size"]) != o[3]:
                change_mismatch += 1
            new_px = dec(m["new_price"]) if m.get("new_price") else o[2]
            new_q = dec(m["new_size"])
            if new_px == o[2] and new_q < o[3]:
                emit(ts, S.CANCEL, o[1], o[2], o[3] - new_q, o[0])
                o[3] = new_q
            elif new_px != o[2] or new_q > o[3]:
                emit(ts, S.MOD_DEL, o[1], o[2], o[3], o[0])
                r = len(rows["ts"])
                emit(ts, S.MOD_ADD, o[1], new_px, new_q, r)
                book[oid_] = [r, o[1], new_px, new_q]
    n = len(rows["ts"])
    # sizes in lots (the gcd of every size), computed on Python integers: a large-supply coin's order can exceed
    # int64 in units of 1e-8 coins before the division
    lot = reduce(math.gcd, {q for q, k in zip(rows["qty"], rows["kind"]) if k != S.STATE and q > 0}, 0) or 1
    rows["qty"] = [q // lot for q in rows["qty"]]
    cols = {
        "ts": np.asarray(rows["ts"], np.int64), "kind": np.asarray(rows["kind"], np.uint8),
        "side": np.asarray(rows["side"], np.int8), "px": np.asarray(rows["px"], np.int64),
        "qty": np.asarray(rows["qty"], np.int64), "oid": np.asarray(rows["oid"], np.int32),
        "aux": np.asarray(rows["aux"], np.int64), "flags": np.asarray(rows["flags"], np.uint8),
    }
    # prices in ticks (the gcd of every price, book and trade): price x size products stay far from int64 limits
    book_rows = cols["kind"] != S.STATE
    ex_rows = cols["kind"] == S.EXEC
    prices = np.unique(np.concatenate([cols["px"][book_rows], cols["aux"][ex_rows]]))
    tick = int(np.gcd.reduce(prices[prices > 0])) if (prices > 0).any() else 1
    cols["px"][book_rows] //= tick
    cols["aux"][ex_rows] //= tick
    qa.update(rows=n, snap_rows=n_snap, unresolved=unresolved, done_mismatch=mismatch, change_mismatch=change_mismatch,
              gap_at_sequence=gap_at, holes=holes, taker_fills=taker_fills, ts_clamped=ts_clamped, **rs, t_snap=t_snap, t_end=end_ts,
              tick=1, tick_raw=tick, px_scale=PX_SCALE / tick, lot_raw=lot, qty_scale=QTY_SCALE / lot)
    return cols, qa


def build_day(day: str, pairs: list[str], minutes: int) -> dict:
    out_dir = day_dir("coinbase", day)
    ensure_space(out_dir, 2.0, BUDGET_GB, DATA)
    writer = EventWriter(out_dir / "events.parquet")
    syms, t0 = [], time.time()
    t_open, t_close = None, None
    for p in pairs:
        cols, qa = build_pair(day, p, minutes)
        if cols is None:
            print(f"{day} {p}: {qa.get('error')}", flush=True)
            continue
        rg = writer.write(cols)
        syms.append({"stock": p, "rg": rg, "n_rows": qa["rows"], **{k: v for k, v in qa.items() if k not in ("pair",)}})
        t_open = qa["t_snap"] if t_open is None else max(t_open, qa["t_snap"])
        t_close = qa["t_end"] if t_close is None else min(t_close, qa["t_end"])
    writer.close()
    pd.DataFrame(syms).to_parquet(out_dir / "symbols.parquet", index=False)
    session = {"open": int(t_open or 0), "close": int(t_close or 0), "t_snap": int(t_open or 0),
               "t_end": int(t_close or 0)}
    q = {"day": day, "venue": "coinbase", "session": session, "pairs": syms, "minutes": minutes,
         "seconds": round(time.time() - t0), "provenance": provenance()}
    write_json(out_dir / "qa.json", q)
    print(f"{day}: {len(syms)} pairs, {sum(s['n_rows'] for s in syms):,} rows, "
          f"unresolved {sum(s['unresolved'] for s in syms)}, gaps {sum(bool(s['gap_at_sequence']) for s in syms)}, "
          f"{q['seconds']}s", flush=True)
    return q


def _build(args):
    return build_day(*args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--minutes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=1, help="days built in parallel")
    ap.add_argument("--force", action="store_true", help="rebuild days that are already built")
    a = ap.parse_args(argv)
    days = [d for d in a.days if a.force or not (day_dir("coinbase", d) / "qa.json").exists()]
    if a.workers <= 1:
        for d in days:
            build_day(d, a.pairs, a.minutes)
        return
    import concurrent.futures as cf
    import multiprocessing as mp

    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_build, [(d, a.pairs, a.minutes) for d in days]))


if __name__ == "__main__":
    main()

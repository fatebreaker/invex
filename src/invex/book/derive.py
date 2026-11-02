"""Replay one symbol's event table through a price-time-priority book.

`derive(ev, open_ns, close_ns)` returns, for every row:

* `lvl`: displayed shares at the row's (side, price) level after the row
  (book rows only; 0 otherwise);
* `bid_px, bid_sz, ask_px, ask_sz`: best quotes after the row (-1 / 0 if empty);

sets the F_TRADE_START flag on the first execution of each aggressor run
(executions with the same timestamp against the same side) and the
F_CONTINUOUS flag inside continuous trading (between the open and the close,
trading state 'T', no operational halt, no LULD collar, no feed outage, outside
the 10 s around a cross), and counts the checks used by invex.book.validate:

* `fifo_ok / fifo_bad`: executions that hit the oldest resting order at their
  level, or not (price-time priority says: always the oldest);
* `crossed`: rows inside continuous trading after which the best bid is at or
  above the best ask (a single venue's own book should never lock or cross);
* `unresolved`: rows that refer to an order that is not resting;
* `negative`: rows after which an order or a level would hold negative shares;
* `off_tick`: displayed prices off the penny grid (sub-dollar prices excepted);
* `del_mismatch`: DELETE / MOD_DEL rows whose shares differ from what the
  replay says was left (checks the annotate pass's bookkeeping);
* `exec_away`: executions at a price other than the order's displayed price
  (ITCH 'C' with a different price: discretion, price improvement). They are
  not queue fills and are left out of the FIFO count;
* `ts_back`: rows whose timestamp is earlier than the previous row's (0).

FIFO is counted inside continuous trading only (crosses allocate differently).

Levels are indexed by the distinct prices of the day (not by ticks), so stub
quotes far from the market cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .. import schema as S

CROSS_GUARD_NS = 10 * 1_000_000_000
PENNY = 100  # $0.01 in ITCH price units (1e-4 $)
DOLLAR = 10_000


@nb.njit(cache=True)
def _replay(ts, kind, side, px, qty, oid, aux, flags, lev, n_lev, open_ns, close_ns, tick, tick_floor,
            lvl, bid_px, bid_sz, ask_px, ask_sz, levels, qa):
    n = ts.shape[0]
    size = np.zeros((2, n_lev), dtype=np.int64)  # [0] bids, [1] asks
    head = np.full((2, n_lev), -1, dtype=np.int64)
    tail = np.full((2, n_lev), -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    prv = np.full(n, -1, dtype=np.int64)
    rem = np.zeros(n, dtype=np.int64)
    live = np.zeros(n, dtype=np.bool_)
    bb, ba = -1, -1  # best bid / best ask level index
    state = ord("T")
    ophalt = 0
    collar = False
    outage = False
    cross_until = np.int64(-1)
    # cross windows: mark the 10 s before every cross as well (known only in hindsight)
    cross_from = np.full(n, np.int64(1) << 62, dtype=np.int64)
    nxt_cross = np.int64(1) << 62
    for i in range(n - 1, -1, -1):
        if kind[i] == S.CROSS:
            nxt_cross = ts[i]
        cross_from[i] = nxt_cross
    fifo_ok = fifo_bad = crossed = unresolved = negative = off_tick = del_mismatch = exec_away = ts_back = 0
    last_exec_ts = np.int64(-1)
    last_exec_side = 0
    for i in range(n):
        k = kind[i]
        if i > 0 and ts[i] < ts[i - 1]:
            ts_back += 1
        if k == S.STATE:
            a = aux[i]
            if a == S.STATE_COLLAR:
                collar = True
            elif a == S.STATE_OUTAGE:
                outage = True
            elif a >= S.STATE_OPHALT:
                ophalt = a - S.STATE_OPHALT
            else:
                state = a
                if a == ord("T"):
                    collar = False
                    outage = False
        elif k == S.CROSS:
            cross_until = ts[i] + CROSS_GUARD_NS
        cont = (open_ns <= ts[i] < close_ns and state == ord("T") and ophalt != ord("H") and not collar
                and not outage and ts[i] >= cross_until and ts[i] < cross_from[i] - CROSS_GUARD_NS)
        if cont:
            flags[i] |= S.F_CONTINUOUS
        if k == S.ADD or k == S.SNAP or k == S.MOD_ADD or k == S.EXEC or k == S.CANCEL or k == S.DELETE or k == S.MOD_DEL:
            sd = 0 if side[i] > 0 else 1
            L = lev[i]
            o = oid[i]
            if k == S.ADD or k == S.SNAP or k == S.MOD_ADD:
                if px[i] >= tick_floor and px[i] % tick != 0:
                    off_tick += 1
                rem[i] = qty[i]
                live[i] = True
                size[sd, L] += qty[i]
                if tail[sd, L] < 0:
                    head[sd, L] = i
                else:
                    nxt[tail[sd, L]] = i
                    prv[i] = tail[sd, L]
                tail[sd, L] = i
                if sd == 0 and (bb < 0 or L > bb):
                    bb = L
                elif sd == 1 and (ba < 0 or L < ba):
                    ba = L
            else:
                if o < 0 or not live[o]:
                    unresolved += 1
                    lvl[i] = size[sd, L]
                else:
                    if k == S.EXEC:
                        if cont:
                            if (flags[i] & S.F_EXEC_PRICE) and aux[i] != px[i]:
                                exec_away += 1  # executed away from its displayed price: not a queue fill
                            elif head[sd, L] == o:
                                fifo_ok += 1
                            else:
                                fifo_bad += 1
                        if not (ts[i] == last_exec_ts and sd == last_exec_side):
                            flags[i] |= S.F_TRADE_START
                        last_exec_ts = ts[i]
                        last_exec_side = sd
                    d = qty[i]
                    if k == S.DELETE or k == S.MOD_DEL:
                        if qty[i] != rem[o]:
                            del_mismatch += 1
                        d = rem[o]
                    rem[o] -= d
                    size[sd, L] -= d
                    if rem[o] < 0 or size[sd, L] < 0:
                        negative += 1
                    if rem[o] <= 0:
                        live[o] = False
                        p_, n_ = prv[o], nxt[o]
                        if p_ >= 0:
                            nxt[p_] = n_
                        else:
                            head[sd, L] = n_
                        if n_ >= 0:
                            prv[n_] = p_
                        else:
                            tail[sd, L] = p_
                    if size[sd, L] <= 0:
                        if sd == 0 and L == bb:
                            while bb >= 0 and size[0, bb] <= 0:
                                bb -= 1
                        elif sd == 1 and L == ba:
                            while ba >= 0 and ba < n_lev and size[1, ba] <= 0:
                                ba += 1
                            if ba >= n_lev:
                                ba = -1
            lvl[i] = size[sd, L]
        if bb >= 0:
            bid_px[i] = levels[bb]
            bid_sz[i] = size[0, bb]
        else:
            bid_px[i] = -1
            bid_sz[i] = 0
        if ba >= 0:
            ask_px[i] = levels[ba]
            ask_sz[i] = size[1, ba]
        else:
            ask_px[i] = -1
            ask_sz[i] = 0
        if cont and bb >= 0 and ba >= 0 and levels[bb] >= levels[ba]:
            crossed += 1
    qa[0] = fifo_ok
    qa[1] = fifo_bad
    qa[2] = crossed
    qa[3] = unresolved
    qa[4] = negative
    qa[5] = off_tick
    qa[6] = del_mismatch
    qa[7] = exec_away
    qa[8] = ts_back


QA_KEYS = ("fifo_ok", "fifo_bad", "crossed", "unresolved", "negative", "off_tick", "del_mismatch", "exec_away",
           "ts_back")
BOOK_KINDS = (S.ADD, S.SNAP, S.MOD_ADD, S.EXEC, S.CANCEL, S.DELETE, S.MOD_DEL)


@dataclass
class Derived:
    lvl: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    levels: np.ndarray  # distinct book prices, ascending
    lev: np.ndarray  # level index of each row (book rows; 0 otherwise)
    qa: dict


def derive(ev: np.ndarray, open_ns: int, close_ns: int, tick: int = PENNY, tick_floor: int = DOLLAR) -> Derived:
    """Replay `ev` (structured array, invex.schema.EVENT_DTYPE); sets flags in place."""
    n = ev.shape[0]
    book = np.isin(ev["kind"], BOOK_KINDS)
    levels, inv = np.unique(ev["px"][book], return_inverse=True)
    lev = np.zeros(n, dtype=np.int64)
    lev[book] = inv
    out = [np.zeros(n, dtype=np.int64) for _ in range(5)]
    qa = np.zeros(len(QA_KEYS), dtype=np.int64)
    flags = np.ascontiguousarray(ev["flags"])
    _replay(np.ascontiguousarray(ev["ts"]), np.ascontiguousarray(ev["kind"]), np.ascontiguousarray(ev["side"]),
            np.ascontiguousarray(ev["px"]), np.ascontiguousarray(ev["qty"]), np.ascontiguousarray(ev["oid"]),
            np.ascontiguousarray(ev["aux"]), flags, lev, max(1, levels.shape[0]), np.int64(open_ns),
            np.int64(close_ns), np.int64(tick), np.int64(tick_floor), *out,
            levels if levels.shape[0] else np.zeros(1, np.int64), qa)
    ev["flags"] = flags
    return Derived(*out, levels=levels, lev=lev, qa=dict(zip(QA_KEYS, qa.tolist())))

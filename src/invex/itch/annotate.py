"""Pass 2: turn each stock's ITCH messages into its event table (invex.schema).

Stocks are processed in parallel; each order reference number belongs to one
stock, so the shared per-reference arrays (side, price, shares left, row of
creation) are written by one thread only. For each stock:

* messages before `t_snap` (pre-market) only update the order state; at the
  first message at or after `t_snap` the orders still resting are written as
  SNAP rows in time priority (the order of their ADD or replace messages),
  followed by a STATE row with the last trading state, if there was one;
* from then on every book, trade and state message becomes one row (a
  replace two: MOD_DEL of the old order, MOD_ADD of the new one, which
  inherits the side and joins the back of its level);
* messages at or after `t_end` are dropped.

Message mapping: A/F -> ADD; E -> EXEC (printable, at the order's price);
C -> EXEC with `aux` = execution price (printable or not); X -> CANCEL;
D -> DELETE (qty = shares left); U -> MOD_DEL + MOD_ADD; P -> HIDDEN;
Q -> CROSS (aux = cross type); B -> BREAK (aux = match number); H -> STATE
(aux = state letter); h (Nasdaq market only) -> STATE (aux = 1000 + action
letter); J -> STATE (aux = 2000, px / qty = lower / upper collar). A message
about an order reference that was never added is counted as an error and
skipped; an execution or cancel of more shares than are left is counted too.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from .. import schema as S
from .index import u16, u32, u48, u64  # noqa: F401  (u16 used via u48)

_A, _F, _E, _C, _X, _D, _U = (ord(c) for c in "AFECXDU")
_P, _Q, _B, _H, _h, _J = (ord(c) for c in "PQBHhJ")
_BUY, _YES = ord("B"), ord("Y")
_MKT_NASDAQ = ord("Q")


@nb.njit(inline="always")
def _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, ts, kind, side, px, qty, oid, aux, flags):
    o_ts[r] = ts
    o_kind[r] = kind
    o_side[r] = side
    o_px[r] = px
    o_qty[r] = qty
    o_oid[r] = oid
    o_aux[r] = aux
    o_flags[r] = flags


@nb.njit(parallel=True, cache=True)
def _annotate(buf, off, typ, perm, loc_start, sel, row_start, t_snap, t_end,
              ref_row, ref_rem, ref_px, ref_side,
              o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags,
              n_rows, n_err, n_over, n_snap, overflow):
    max_ref = ref_side.shape[0] - 1
    for j in nb.prange(sel.shape[0]):
        l = sel[j]
        base = row_start[j]
        limit = row_start[j + 1] - 2  # keep room for the (at most two) emits of one message
        r = base
        snapped = False
        err = 0
        over = 0
        state = 0  # last trading-state letter seen before the snapshot (0 = none)
        ophalt = 0  # last operational-halt action before the snapshot (0 = none)
        a = loc_start[l]
        b = loc_start[l + 1]
        k = a
        while k <= b:
            if k < b:
                i = perm[k]
                p = off[i]
                t = np.int64(typ[i])
                ts = u48(buf, p + 5)
            else:
                ts = t_end  # past the last message: make sure the snapshot is taken
                t = 0
                p = 0
            if not snapped and ts >= t_snap:
                # resting orders in time priority = order of the messages that created them
                ns = 0
                for k2 in range(a, k):
                    i2 = perm[k2]
                    t2 = np.int64(typ[i2])
                    if t2 == _A or t2 == _F:
                        ref = u64(buf, off[i2] + 11)
                    elif t2 == _U:
                        ref = u64(buf, off[i2] + 19)
                    else:
                        continue
                    if ref_rem[ref] > 0 and ref_row[ref] < 0:
                        _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                              t_snap, S.SNAP, ref_side[ref], np.int64(ref_px[ref]), np.int64(ref_rem[ref]),
                              r - base, 0, 0)
                        ref_row[ref] = r - base
                        r += 1
                        ns += 1
                if state != 0:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          t_snap, S.STATE, 0, 0, 0, -1, state, 0)
                    r += 1
                if ophalt != 0:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          t_snap, S.STATE, 0, 0, 0, -1, S.STATE_OPHALT + ophalt, 0)
                    r += 1
                n_snap[j] = ns
                snapped = True
            if k == b or ts >= t_end:
                break
            k += 1
            if r > limit:
                # the row bound was wrong: stop rather than write into the next stock's rows
                overflow[j] = 1
                break

            if t == _A or t == _F:
                ref = u64(buf, p + 11)
                side = 1 if buf[p + 19] == _BUY else -1
                sh = u32(buf, p + 20)
                px = u32(buf, p + 32)
                ref_side[ref] = side
                ref_px[ref] = px
                ref_rem[ref] = sh
                if snapped:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          ts, S.ADD, side, px, sh, r - base, 0, 0)
                    ref_row[ref] = r - base
                    r += 1
                else:
                    ref_row[ref] = -1
            elif t == _E or t == _C or t == _X:
                ref = u64(buf, p + 11)
                sh = u32(buf, p + 19)
                if ref <= 0 or ref > max_ref or ref_side[ref] == 0:
                    err += 1
                    continue
                if sh > ref_rem[ref]:
                    over += 1
                ref_rem[ref] -= sh
                if not snapped:
                    continue
                if t == _E:
                    kind, aux, fl = S.EXEC, np.int64(ref_px[ref]), S.F_PRINTABLE
                elif t == _C:
                    kind, aux = S.EXEC, u32(buf, p + 32)
                    fl = S.F_EXEC_PRICE | (S.F_PRINTABLE if buf[p + 31] == _YES else 0)
                else:
                    kind, aux, fl = S.CANCEL, 0, 0
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, kind, ref_side[ref], np.int64(ref_px[ref]), sh, ref_row[ref], aux, fl)
                r += 1
            elif t == _D:
                ref = u64(buf, p + 11)
                if ref <= 0 or ref > max_ref or ref_side[ref] == 0:
                    err += 1
                    continue
                left = np.int64(ref_rem[ref])
                ref_rem[ref] = 0
                if snapped:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          ts, S.DELETE, ref_side[ref], np.int64(ref_px[ref]), left, ref_row[ref], 0, 0)
                    r += 1
            elif t == _U:
                old = u64(buf, p + 11)
                new = u64(buf, p + 19)
                sh = u32(buf, p + 27)
                px = u32(buf, p + 31)
                if old <= 0 or old > max_ref or ref_side[old] == 0 or new <= 0 or new > max_ref:
                    err += 1
                    continue
                side = ref_side[old]
                left = np.int64(ref_rem[old])
                ref_rem[old] = 0
                ref_side[new] = side
                ref_px[new] = px
                ref_rem[new] = sh
                if snapped:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          ts, S.MOD_DEL, side, np.int64(ref_px[old]), left, ref_row[old], 0, 0)
                    r += 1
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          ts, S.MOD_ADD, side, px, sh, r - base, 0, 0)
                    ref_row[new] = r - base
                    r += 1
                else:
                    ref_row[new] = -1
            elif not snapped:
                if t == _H:
                    state = np.int64(buf[p + 19])
                elif t == _h and buf[p + 19] == _MKT_NASDAQ:
                    ophalt = np.int64(buf[p + 20])
            elif t == _P:
                side = 1 if buf[p + 19] == _BUY else -1
                px = u32(buf, p + 32)
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, S.HIDDEN, side, px, u32(buf, p + 20), -1, px, S.F_PRINTABLE)
                r += 1
            elif t == _Q:
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, S.CROSS, 0, u32(buf, p + 27), u64(buf, p + 11), -1, np.int64(buf[p + 39]), 0)
                r += 1
            elif t == _B:
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, S.BREAK, 0, 0, 0, -1, u64(buf, p + 11), 0)
                r += 1
            elif t == _H:
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, S.STATE, 0, 0, 0, -1, np.int64(buf[p + 19]), 0)
                r += 1
            elif t == _h:
                if buf[p + 19] == _MKT_NASDAQ:
                    _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                          ts, S.STATE, 0, 0, 0, -1, S.STATE_OPHALT + np.int64(buf[p + 20]), 0)
                    r += 1
            elif t == _J:
                _emit(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r,
                      ts, S.STATE, 0, u32(buf, p + 27), u32(buf, p + 23), -1, S.STATE_COLLAR, 0)
                r += 1
        n_rows[j] = r - base
        n_err[j] = err
        n_over[j] = over


def annotate(buf: np.ndarray, off: np.ndarray, typ: np.ndarray, perm: np.ndarray, loc_start: np.ndarray,
             sel: np.ndarray, bound: np.ndarray, max_ref: int, t_snap: int, t_end: int) -> dict:
    """Event tables for the stock locates `sel` (rows of stock j: out[row_start[j]: row_start[j] + n_rows[j]]).

    `bound[j]` is Index.nrows[sel[j]]: orders added before t_snap plus the rows of the messages in
    [t_snap, t_end). The snapshot may add two STATE rows and the overflow guard keeps two rows of
    slack, so each stock gets bound + 4 rows. Raises if any stock still overflows.
    Returns the concatenated columns plus row_start, n_rows, n_err, n_over, n_snap.
    """
    sel = np.ascontiguousarray(sel, dtype=np.int64)
    row_start = np.zeros(sel.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.asarray(bound, dtype=np.int64) + 4, out=row_start[1:])
    total = int(row_start[-1])
    out = {
        "ts": np.empty(total, np.int64), "kind": np.empty(total, np.uint8), "side": np.empty(total, np.int8),
        "px": np.empty(total, np.int64), "qty": np.empty(total, np.int64), "oid": np.empty(total, np.int32),
        "aux": np.empty(total, np.int64), "flags": np.empty(total, np.uint8),
    }
    ref_row = np.full(max_ref + 1, -1, dtype=np.int32)
    ref_rem = np.zeros(max_ref + 1, dtype=np.int32)
    ref_px = np.zeros(max_ref + 1, dtype=np.uint32)
    ref_side = np.zeros(max_ref + 1, dtype=np.int8)
    n = sel.shape[0]
    n_rows, n_err, n_over, n_snap, overflow = (np.zeros(n, np.int64) for _ in range(5))
    old_chunk = nb.get_parallel_chunksize()
    nb.set_parallel_chunksize(1)  # stocks differ by orders of magnitude in size: schedule dynamically
    try:
        _annotate(buf, off, typ, perm, loc_start, sel, row_start, np.int64(t_snap), np.int64(t_end),
                  ref_row, ref_rem, ref_px, ref_side,
                  out["ts"], out["kind"], out["side"], out["px"], out["qty"], out["oid"], out["aux"], out["flags"],
                  n_rows, n_err, n_over, n_snap, overflow)
    finally:
        nb.set_parallel_chunksize(old_chunk)
    if overflow.any():
        raise RuntimeError(f"row bound exceeded for locates {sel[overflow > 0][:10].tolist()}")
    out.update(row_start=row_start, n_rows=n_rows, n_err=n_err, n_over=n_over, n_snap=n_snap)
    return out

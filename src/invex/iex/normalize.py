"""IEX DEEP+ days -> invex event tables (same schema as ITCH).

    python -m invex.iex.normalize --days 2026-10-14

Per day: decompress the HIST capture with pigz to $INVEX_RAW/iex/<day>/DPLS.pcapng (removed after the build unless
--keep-pcap), read every DEEP+ message in sequence order (`invex.iex.pcap`), and build one table per common stock.
Common stocks are taken from the newest ITCH stock directory (REF_ITCH_DAY: authentic issue, not an ETP, issue class
common or ordinary share), since IEX's own directory covers only IEX-listed securities. Mapping:

    a  Add Order                              -> ADD
    M  Order Modify, maintain priority        -> CANCEL of the size cut (same price, smaller size)
    M  Order Modify, reset priority           -> MOD_DEL + MOD_ADD (back of the queue at the new price and size)
    R  Order Delete                           -> DELETE of what is left
    L  Order Executed                         -> EXEC (aux = trade price; F_EXEC_PRICE when it differs from the
                                                 order's price: IEX's price improvement or sliding)
    T  Trade (non-displayed liquidity)        -> HIDDEN (px = aux = trade price)
    B  Trade Break                            -> BREAK (aux = trade id); the broken execution gets F_BROKEN
    C  Clear Book                             -> DELETE of every resting order
    H  Trading Status                         -> STATE ('H' halted, 'P' paused, 'O' order acceptance as 'Q', 'T')
    O  Operational Halt                       -> STATE (STATE_OPHALT + 'H' / 'T')
    single-price cross executions (L/T flag 0x08) are preceded by a CROSS row (the 10 s guard of invex.book.derive)

As for ITCH, a table starts with a snapshot of the book at 09:25 ET (SNAP rows in time priority, then the trading
state if it is not 'T') and ends 5 minutes after the close; times are ns since local (New York) midnight. A Modify
that keeps priority while raising the size or moving the price is applied as a reset and counted (odd_modify);
references to orders that are not resting are counted (unresolved, expected 0).
Output: $INVEX_DATA/iex/<day>/{events.parquet (one row group per stock), symbols.parquet, qa.json}.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numba as nb
import numpy as np
import pandas as pd

from .. import schema as S
from ..io.store import EventWriter, day_dir, symbols
from ..paths import BUDGET_GB, DATA, ensure_space
from ..provenance import provenance, write_json
from . import pcap as P
from .hist import local_path

REF_ITCH_DAY = "2026-06-12"
NS = 1_000_000_000
NY = ZoneInfo("America/New_York")
HALF_DAYS = {"2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"}

# per-stock QA counters
Q_UNRES, Q_OVER, Q_CLAMP, Q_ODD_MOD, Q_NOOP_MOD, Q_REUSE, Q_CLEARS, Q_BREAKS, Q_BREAK_MISS, Q_SNAP = range(10)
C_T, C_Q, C_O, C_H = ord("T"), ord("Q"), ord("O"), ord("H")  # state codes, as ints for numba
Q_NAMES = ("unresolved", "overexec", "ts_clamped", "odd_modify", "noop_modify", "id_reuse", "clears", "breaks",
           "break_unmatched", "n_snap")


def hms(h: int, m: int = 0, s: int = 0) -> int:
    return ((h * 60 + m) * 60 + s) * NS


def session(day: str) -> dict:
    close = hms(13) if day in HALF_DAYS else hms(16)
    return {"open": hms(9, 30), "close": close, "t_snap": hms(9, 25), "t_end": close + hms(0, 5)}


def midnight_ns(day: str) -> int:
    """Epoch ns of local midnight in New York on `day`."""
    return int(datetime.fromisoformat(day).replace(tzinfo=NY).timestamp()) * NS


@nb.njit(inline="always")
def _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, kind, side, px, qty, oid, aux, flags):
    o_ts[r] = t
    o_kind[r] = kind
    o_side[r] = side
    o_px[r] = px
    o_qty[r] = qty
    o_oid[r] = oid
    o_aux[r] = aux
    o_flags[r] = flags
    return r + 1


@nb.njit(cache=True)
def _build(typ, code, ts, od, size, price, tid, n_dense, t_snap, t_end,
           o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, o_tid, qa):
    live = np.zeros(n_dense, np.bool_)
    oside = np.zeros(n_dense, np.int8)
    opx = np.zeros(n_dense, np.int64)
    orem = np.zeros(n_dense, np.int64)
    orow = np.full(n_dense, -1, np.int64)
    oprio = np.zeros(n_dense, np.int64)
    r = 0
    state = C_T
    ophalt = False
    snapped = False
    last = np.int64(0)
    last_cross = np.int64(-1)
    n = typ.shape[0]
    for j in range(n + 1):
        t = ts[j] if j < n else t_end + 1
        if not snapped and t >= t_snap:
            idx = np.flatnonzero(live)
            idx = idx[np.argsort(oprio[idx], kind="mergesort")]
            for d in idx:
                orow[d] = r
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t_snap, S.SNAP, oside[d], opx[d],
                         orem[d], r, 0, 0)
            qa[Q_SNAP] = idx.shape[0]
            if state != C_T:
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t_snap, S.STATE, 0, 0, 0, -1,
                         state, 0)
            if ophalt:
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t_snap, S.STATE, 0, 0, 0, -1,
                         S.STATE_OPHALT + C_H, 0)
            snapped = True
            last = t_snap
        if j == n or t > t_end:
            break
        emit = snapped
        if emit:
            if t < last:
                qa[Q_CLAMP] += 1
                t = last
            last = t
        m = typ[j]
        d = od[j]
        if m == 0x61:  # a: Add Order
            if live[d]:
                qa[Q_REUSE] += 1
            live[d] = True
            oside[d] = 1 if code[j] == 0x38 else -1
            opx[d] = price[j]
            orem[d] = size[j]
            oprio[d] = j
            if emit:
                orow[d] = r
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.ADD, oside[d], opx[d],
                         orem[d], r, 0, 0)
        elif m == 0x4D:  # M: Order Modify
            if not live[d]:
                qa[Q_UNRES] += 1
                continue
            keep = (code[j] & 1) == 1
            if size[j] <= 0:
                if emit:
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.DELETE, oside[d],
                             opx[d], orem[d], orow[d], 0, 0)
                live[d] = False
            elif keep and price[j] == opx[d] and size[j] < orem[d]:
                if emit:
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.CANCEL, oside[d],
                             opx[d], orem[d] - size[j], orow[d], 0, 0)
                orem[d] = size[j]
            elif keep and price[j] == opx[d] and size[j] == orem[d]:
                qa[Q_NOOP_MOD] += 1
            else:
                if keep:
                    qa[Q_ODD_MOD] += 1
                if emit:
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.MOD_DEL, oside[d],
                             opx[d], orem[d], orow[d], 0, 0)
                    orow[d] = r
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.MOD_ADD, oside[d],
                             price[j], size[j], r, 0, 0)
                opx[d] = price[j]
                orem[d] = size[j]
                oprio[d] = j
        elif m == 0x52:  # R: Order Delete
            if not live[d]:
                qa[Q_UNRES] += 1
                continue
            if emit:
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.DELETE, oside[d], opx[d],
                         orem[d], orow[d], 0, 0)
            live[d] = False
        elif m == 0x4C:  # L: Order Executed
            if not live[d]:
                qa[Q_UNRES] += 1
                continue
            if emit:
                if (code[j] & 0x08) != 0 and t != last_cross:
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.CROSS, 0, price[j], 0,
                             -1, price[j], 0)
                    last_cross = t
                fl = S.F_PRINTABLE | (S.F_EXEC_PRICE if price[j] != opx[d] else 0)
                o_tid[r] = tid[j]
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.EXEC, oside[d], opx[d],
                         size[j], orow[d], price[j], fl)
            orem[d] -= size[j]
            if orem[d] < 0:
                qa[Q_OVER] += 1
            if orem[d] <= 0:
                live[d] = False
        elif m == 0x54:  # T: Trade against non-displayed liquidity
            if emit:
                if (code[j] & 0x08) != 0 and t != last_cross:
                    r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.CROSS, 0, price[j], 0,
                             -1, price[j], 0)
                    last_cross = t
                o_tid[r] = tid[j]
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.HIDDEN, 0, price[j],
                         size[j], -1, price[j], S.F_PRINTABLE)
        elif m == 0x42:  # B: Trade Break
            if emit:
                qa[Q_BREAKS] += 1
                hit = False
                for q in range(r - 1, -1, -1):
                    if o_tid[q] == tid[j] and (o_kind[q] == S.EXEC or o_kind[q] == S.HIDDEN):
                        o_flags[q] |= S.F_BROKEN
                        hit = True
                        break
                if not hit:
                    qa[Q_BREAK_MISS] += 1
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.BREAK, 0, price[j],
                         size[j], -1, tid[j], 0)
        elif m == 0x43:  # C: Clear Book
            qa[Q_CLEARS] += 1
            for dd in range(n_dense):
                if live[dd]:
                    if emit:
                        r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.DELETE, oside[dd],
                                 opx[dd], orem[dd], orow[dd], 0, 0)
                    live[dd] = False
        elif m == 0x48:  # H: Trading Status
            st = np.int64(code[j])
            if st == C_O:
                st = C_Q
            state = st
            if emit:
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.STATE, 0, 0, 0, -1, st, 0)
        elif m == 0x4F:  # O: Operational Halt Status
            ophalt = code[j] == C_O
            if emit:
                r = _put(o_ts, o_kind, o_side, o_px, o_qty, o_oid, o_aux, o_flags, r, t, S.STATE, 0, 0, 0, -1,
                         S.STATE_OPHALT + (C_H if ophalt else C_T), 0)
    return r


def build_stock(msg: dict[str, np.ndarray], t_snap: int, t_end: int) -> tuple[dict, dict]:
    """One stock's messages (sequence order; ts already in local ns) -> (event columns, QA counters)."""
    typ = msg["typ"]
    ref = np.isin(typ, np.frombuffer(b"aMRL", np.uint8))
    od = np.full(typ.shape[0], -1, np.int64)
    n_dense = 0
    if ref.any():
        u, inv = np.unique(msg["oid"][ref], return_inverse=True)
        od[ref] = inv
        n_dense = u.shape[0]
    n_clear = int((typ == 0x43).sum())
    cap = 2 * typ.shape[0] + n_dense * (1 + n_clear) + 16
    out = {c: np.zeros(cap, S.EVENT_DTYPE[c]) for c in S.COLUMNS}
    o_tid = np.full(cap, -1, np.int64)
    qa = np.zeros(len(Q_NAMES), np.int64)
    r = _build(typ, msg["code"], msg["ts"], od, msg["size"], msg["price"], msg["tid"], n_dense, t_snap, t_end,
               out["ts"], out["kind"], out["side"], out["px"], out["qty"], out["oid"], out["aux"], out["flags"], o_tid,
               qa)
    return {c: out[c][:r] for c in S.COLUMNS}, dict(zip(Q_NAMES, (int(x) for x in qa)))


@nb.njit(cache=True, parallel=True)
def _lookup(sym, keys, out):
    for i in nb.prange(sym.shape[0]):
        s = sym[i]
        lo, hi = 0, keys.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < s:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo if (s != 0 and lo < keys.shape[0] and keys[lo] == s) else -1


@nb.njit(cache=True)
def _group(sid, n_groups):
    """Stable counting sort of message indices by stock id (-1 dropped): (order, start)."""
    cnt = np.zeros(n_groups + 1, np.int64)
    for i in range(sid.shape[0]):
        if sid[i] >= 0:
            cnt[sid[i] + 1] += 1
    start = np.cumsum(cnt)
    pos = start[:-1].copy()
    order = np.empty(start[-1], np.int64)
    for i in range(sid.shape[0]):
        g = sid[i]
        if g >= 0:
            order[pos[g]] = i
            pos[g] += 1
    return order, start


def common_stocks(ref_day: str = REF_ITCH_DAY) -> pd.DataFrame:
    sym = symbols("itch", ref_day)
    keep = (sym["authenticity"] == "P") & (sym["etp_flag"] != "Y") & sym["issue_class"].isin(["C", "O"])
    return sym.loc[keep, ["stock", "market_category", "round_lot", "issue_class", "authenticity", "etp_flag"]]


def system_events(msg: dict, midnight: int) -> dict:
    hit = np.flatnonzero(msg["typ"] == 0x53)
    return {chr(int(msg["code"][i])): int(msg["ts"][i] - midnight) for i in hit}


def decompress(day: str):
    gz = local_path(day)
    out = gz.with_name("DPLS.pcapng")
    if out.exists():
        return out
    if not gz.exists():
        raise FileNotFoundError(f"{gz} (run python -m invex.iex.hist --days {day})")
    ensure_space(out.parent, gz.stat().st_size * 4 / 1024**3)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        subprocess.run(["pigz", "-dc", str(gz)], stdout=fh, check=True)
    tmp.replace(out)
    return out


def build_day(day: str, keep_pcap: bool = False) -> dict:
    t0 = time.time()
    sess = session(day)
    path = decompress(day)
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    sc = P.scan(buf)
    msg = P.decode(buf, sc)
    del buf
    t_read = time.time() - t0
    midnight = midnight_ns(day)
    msg["ts"] -= midnight
    events = system_events(msg, 0)
    problems = []
    if sc.stats["bad_blocks"] or sc.stats["truncated"]:
        problems.append(f"capture: bad_blocks={sc.stats['bad_blocks']} truncated={sc.stats['truncated']}")
    for code, want in (("R", sess["open"]), ("M", sess["close"])):
        if code not in events or abs(events[code] - want) > 60 * NS:
            problems.append(f"system event {code} at {events.get(code, float('nan')) / 3600e9:.3f}h, calendar "
                            f"{want / 3600e9:.3f}h")
    if problems:
        raise RuntimeError(f"{day}: " + "; ".join(problems))

    cand = common_stocks().reset_index(drop=True)
    codes = np.array([P.symbol_code(s) for s in cand["stock"]], np.uint64)
    order_c = np.argsort(codes)
    keys = codes[order_c]
    sid = np.empty(msg["sym"].shape[0], np.int64)
    _lookup(msg["sym"], keys, sid)
    order, start = _group(sid, keys.shape[0])

    out_dir = day_dir("iex", day)
    ensure_space(out_dir, order.shape[0] * 14 / 1024**3, BUDGET_GB, DATA)
    writer = EventWriter(out_dir / "events.parquet")
    rows = []
    for g in range(keys.shape[0]):
        a, b = start[g], start[g + 1]
        if b == a:
            continue
        sel = order[a:b]
        cols, q = build_stock({k: msg[k][sel] for k in ("typ", "code", "ts", "oid", "size", "price", "tid")},
                              sess["t_snap"], sess["t_end"])
        if cols["ts"].shape[0] == 0:
            continue
        rg = writer.write(cols)
        typ = msg["typ"][sel]
        rows.append({"stock": cand["stock"].iloc[order_c[g]], "rg": rg, "n_rows": int(cols["ts"].shape[0]),
                     **{f"n_{c}": int((typ == ord(c)).sum()) for c in "aMRLTBC"}, **q})
    writer.close()
    sym = pd.DataFrame(rows).merge(cand, on="stock", how="left")
    sym.to_parquet(out_dir / "symbols.parquet", index=False)
    qa = {"day": day, "venue": "iex", "session": sess, "system_events": events, "ref_itch_day": REF_ITCH_DAY,
          **sc.stats, "stocks": len(sym), "rows": int(sym["n_rows"].sum()),
          **{k: int(sym[k].sum()) for k in Q_NAMES if k != "n_snap"},
          "seconds": {"read": round(t_read), "total": round(time.time() - t0)}, "provenance": provenance()}
    write_json(out_dir / "qa.json", qa)
    if not keep_pcap:
        path.unlink()
    print(f"{day}: {sc.stats['messages']:,} messages ({sc.stats['duplicates']:,} duplicates, {sc.stats['gap_messages']} "
          f"missing), {len(sym)} stocks, {qa['rows']:,} rows, unresolved {qa['unresolved']}, overexec "
          f"{qa['overexec']}, {qa['seconds']['total']}s", flush=True)
    return qa


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--keep-pcap", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild days that are already built")
    a = ap.parse_args(argv)
    for d in a.days:
        if a.force or not (day_dir("iex", d) / "qa.json").exists():
            build_day(d, a.keep_pcap)


if __name__ == "__main__":
    main()

"""Plain-Python ITCH 5.0 decoding: slow, simple, and used to check the compiled passes.

`iter_messages(data)` yields (type letter, fields dict) for each framed message.
`reference_tables(data, t_snap, t_end)` builds the event tables with a dict-based
order book, following the same rules as invex.itch.annotate, so the two can be
compared row by row (tests and `python -m invex.itch.pydecode <file> <n>`).
"""

from __future__ import annotations

import struct
from collections import defaultdict

from .. import schema as S
from .spec import LENGTH


def _ts(m: bytes) -> int:
    return int.from_bytes(m[5:11], "big")


def decode(m: bytes) -> tuple[str, dict]:
    t = chr(m[0])
    d = {"locate": struct.unpack_from(">H", m, 1)[0], "ts": _ts(m)}
    if t in "AF":
        ref, side, sh, stock, px = struct.unpack_from(">Qc I8sI".replace(" ", ""), m, 11)
        d.update(ref=ref, side=side.decode(), shares=sh, stock=stock.decode().strip(), price=px)
    elif t == "E":
        d.update(zip(("ref", "shares", "match"), struct.unpack_from(">QIQ", m, 11)))
    elif t == "C":
        ref, sh, match, pr, px = struct.unpack_from(">QIQcI", m, 11)
        d.update(ref=ref, shares=sh, match=match, printable=pr == b"Y", price=px)
    elif t == "X":
        d.update(zip(("ref", "shares"), struct.unpack_from(">QI", m, 11)))
    elif t == "D":
        d.update(ref=struct.unpack_from(">Q", m, 11)[0])
    elif t == "U":
        d.update(zip(("old", "new", "shares", "price"), struct.unpack_from(">QQII", m, 11)))
    elif t == "P":
        ref, side, sh, stock, px, match = struct.unpack_from(">QcI8sIQ", m, 11)
        d.update(ref=ref, side=side.decode(), shares=sh, stock=stock.decode().strip(), price=px, match=match)
    elif t == "Q":
        sh, stock, px, match, ct = struct.unpack_from(">Q8sIQc", m, 11)
        d.update(shares=sh, stock=stock.decode().strip(), price=px, match=match, cross_type=ct.decode())
    elif t == "B":
        d.update(match=struct.unpack_from(">Q", m, 11)[0])
    elif t == "H":
        stock, state, _res, reason = struct.unpack_from(">8scc4s", m, 11)
        d.update(stock=stock.decode().strip(), state=state.decode(), reason=reason.decode().strip())
    elif t == "h":
        stock, mkt, act = struct.unpack_from(">8scc", m, 11)
        d.update(stock=stock.decode().strip(), market=mkt.decode(), action=act.decode())
    elif t == "J":
        stock, ref_px, up, lo, ext = struct.unpack_from(">8sIIII", m, 11)
        d.update(stock=stock.decode().strip(), ref_price=ref_px, upper=up, lower=lo, extension=ext)
    elif t == "S":
        d.update(event=chr(m[11]))
    elif t == "R":
        d.update(stock=m[11:19].decode().strip(), market_category=chr(m[19]), round_lot=struct.unpack_from(">I", m, 21)[0],
                 issue_class=chr(m[26]), authenticity=chr(m[29]), etp_flag=chr(m[33]))
    return t, d


def iter_messages(data: bytes, limit: int | None = None):
    pos, n, k = 0, len(data), 0
    while pos + 2 <= n and (limit is None or k < limit):
        L = struct.unpack_from(">H", data, pos)[0]
        if L == 0 or pos + 2 + L > n:
            return
        m = data[pos + 2: pos + 2 + L]
        t = chr(m[0])
        if LENGTH.get(t) != L:
            raise ValueError(f"message {k} type {t!r} has length {L}, expected {LENGTH.get(t)}")
        yield decode(m)
        pos += 2 + L
        k += 1


def reference_tables(data: bytes, t_snap: int, t_end: int) -> dict[int, list[tuple]]:
    """locate -> list of event rows (ts, kind, side, px, qty, oid, aux, flags), same rules as annotate."""
    orders: dict[int, list] = {}  # ref -> [locate, side, px, rem, row or None, creation_seq]
    rows: dict[int, list] = defaultdict(list)
    snapped: set[int] = set()
    state: dict[int, int] = {}
    ophalt: dict[int, int] = {}
    created: dict[int, list] = defaultdict(list)  # locate -> refs in creation order
    seen_locs: list[int] = []
    done: set[int] = set()

    def snapshot(loc: int) -> None:
        out = rows[loc]
        for ref in created[loc]:
            o = orders[ref]
            if o[3] > 0 and o[4] is None:
                o[4] = len(out)
                out.append((t_snap, S.SNAP, o[1], o[2], o[3], o[4], 0, 0))
        if loc in state:
            out.append((t_snap, S.STATE, 0, 0, 0, -1, state[loc], 0))
        if loc in ophalt:
            out.append((t_snap, S.STATE, 0, 0, 0, -1, S.STATE_OPHALT + ophalt[loc], 0))
        snapped.add(loc)

    for t, d in iter_messages(data):
        loc, ts = d["locate"], d["ts"]
        if t not in "AFECXDUPQBHhJ":
            continue
        if loc not in seen_locs:
            seen_locs.append(loc)
        if loc in done:
            continue
        if loc not in snapped and ts >= t_snap:
            snapshot(loc)
        if ts >= t_end:
            done.add(loc)
            continue
        out = rows[loc]
        on = loc in snapped
        if t in "AF":
            side = 1 if d["side"] == "B" else -1
            o = [loc, side, d["price"], d["shares"], None]
            orders[d["ref"]] = o
            created[loc].append(d["ref"])
            if on:
                o[4] = len(out)
                out.append((ts, S.ADD, side, d["price"], d["shares"], o[4], 0, 0))
        elif t in "ECX":
            o = orders.get(d["ref"])
            if o is None:
                continue
            o[3] -= d["shares"]
            if on:
                if t == "E":
                    kind, aux, fl = S.EXEC, o[2], S.F_PRINTABLE
                elif t == "C":
                    kind, aux, fl = S.EXEC, d["price"], S.F_EXEC_PRICE | (S.F_PRINTABLE if d["printable"] else 0)
                else:
                    kind, aux, fl = S.CANCEL, 0, 0
                out.append((ts, kind, o[1], o[2], d["shares"], o[4], aux, fl))
        elif t == "D":
            o = orders.get(d["ref"])
            if o is None:
                continue
            left, o[3] = o[3], 0
            if on:
                out.append((ts, S.DELETE, o[1], o[2], left, o[4], 0, 0))
        elif t == "U":
            o = orders.get(d["old"])
            if o is None:
                continue
            left, o[3] = o[3], 0
            n = [loc, o[1], d["price"], d["shares"], None]
            orders[d["new"]] = n
            created[loc].append(d["new"])
            if on:
                out.append((ts, S.MOD_DEL, o[1], o[2], left, o[4], 0, 0))
                n[4] = len(out)
                out.append((ts, S.MOD_ADD, n[1], n[2], n[3], n[4], 0, 0))
        elif not on:
            if t == "H":
                state[loc] = ord(d["state"])
            elif t == "h" and d["market"] == "Q":
                ophalt[loc] = ord(d["action"])
        elif t == "P":
            side = 1 if d["side"] == "B" else -1
            out.append((ts, S.HIDDEN, side, d["price"], d["shares"], -1, d["price"], S.F_PRINTABLE))
        elif t == "Q":
            out.append((ts, S.CROSS, 0, d["price"], d["shares"], -1, ord(d["cross_type"]), 0))
        elif t == "B":
            out.append((ts, S.BREAK, 0, 0, 0, -1, d["match"], 0))
        elif t == "H":
            out.append((ts, S.STATE, 0, 0, 0, -1, ord(d["state"]), 0))
        elif t == "h":
            if d["market"] == "Q":
                out.append((ts, S.STATE, 0, 0, 0, -1, S.STATE_OPHALT + ord(d["action"]), 0))
        elif t == "J":
            out.append((ts, S.STATE, 0, d["lower"], d["upper"], -1, S.STATE_COLLAR, 0))
    for loc in seen_locs:  # stocks whose messages all came before t_snap
        if loc not in snapped:
            snapshot(loc)
    return dict(rows)

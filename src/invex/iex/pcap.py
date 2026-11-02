"""Read IEX DEEP+ messages out of a HIST capture (pcap-ng of Ethernet / IPv4 / UDP / IEX-TP segments).

`scan(buf)` walks the capture's blocks (Enhanced and Simple Packet Blocks; other blocks are skipped), keeps the UDP
payloads that are IEX-TP v1 segments of DEEP+ (message protocol 0x8005), and returns the offset and length of every
message in sequence order: a message whose sequence number was already seen (a duplicate segment) is dropped and
skipped sequence numbers are counted as a gap. `decode(buf, off)` reads the fields the event tables need. All
IEX-TP and DEEP+ fields are little endian; Ethernet, IP and UDP headers are big endian.

DEEP+ v1.06 layouts used (offsets in the message):

    a  Add Order       side 1, ts 2, symbol 10, order id 18, size 26 (u32), price 30
    M  Order Modify    flags 1 (bit 0: maintain priority), ts 2, symbol 10, order id 18, size 26, price 30
    R  Order Delete    ts 2, symbol 10, order id 18
    L  Order Executed  sale flags 1, ts 2, symbol 10, order id 18, size 26, price 30, trade id 38
    T  Trade           sale flags 1, ts 2, symbol 10, size 18, price 22, trade id 30   (non-displayed)
    B  Trade Break     sale flags 1, ts 2, symbol 10, size 18, price 22, trade id 30
    C  Clear Book      ts 2, symbol 10
    H  Trading Status  status 1, ts 2, symbol 10
    O  Operational Halt status 1, ts 2, symbol 10
    S  System Event    event 1, ts 2                                                   (no symbol)

Prices are fixed point with 4 decimals (the ITCH price unit, 1e-4 $), timestamps ns since the epoch (UTC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

PROTOCOL_DEEP_PLUS = 0x8005
SYMBOLIC = b"DHIOPEaMRLTBC"  # message types that carry a symbol at offset 10

# scan statistics
N_MSGS, N_DUP, N_GAP, N_PACKETS, N_OTHER, N_BAD, N_SESSIONS, N_HEARTBEAT, N_TRUNC = range(9)
STAT_NAMES = ("messages", "duplicates", "gap_messages", "packets", "other_packets", "bad_blocks", "session_changes",
              "heartbeats", "truncated")


@nb.njit(inline="always")
def _u16(b, i):
    return np.int64(b[i]) | (np.int64(b[i + 1]) << 8)


@nb.njit(inline="always")
def _u16be(b, i):
    return (np.int64(b[i]) << 8) | np.int64(b[i + 1])


@nb.njit(inline="always")
def _u32(b, i):
    return np.int64(b[i]) | (np.int64(b[i + 1]) << 8) | (np.int64(b[i + 2]) << 16) | (np.int64(b[i + 3]) << 24)


@nb.njit(inline="always")
def _u64(b, i):
    v = np.uint64(0)
    for k in range(8):
        v |= np.uint64(b[i + k]) << np.uint64(8 * k)
    return v


@nb.njit(inline="always")
def _i64(b, i):
    lo = np.int64(_u32(b, i))
    hi = np.int64(_u32(b, i + 4))
    return lo | (hi << 32)


@nb.njit(cache=True)
def _scan(b, fill, off, ln, stats):
    n = b.shape[0]
    p = 0
    k = 0
    next_seq = np.int64(-1)
    session = np.int64(-1)
    while p + 12 <= n:
        btype = _u32(b, p)
        blen = _u32(b, p + 4)
        if blen < 12 or p + blen > n:
            stats[N_BAD] += 1
            break
        d = np.int64(-1)
        cap = np.int64(0)
        if btype == 6:  # Enhanced Packet Block
            cap = _u32(b, p + 20)
            d = p + 28
        elif btype == 3:  # Simple Packet Block
            cap = min(_u32(b, p + 8), blen - 16)
            d = p + 12
        if d >= 0:
            stats[N_PACKETS] += 1
            ok = cap >= 42
            l3 = d + 14
            s = np.int64(0)
            plen = np.int64(0)
            if ok:
                et = _u16be(b, d + 12)
                if et == 0x8100:
                    et = _u16be(b, d + 16)
                    l3 = d + 18
                ok = et == 0x0800 and b[l3 + 9] == 17
            if ok:
                l4 = l3 + (b[l3] & 0x0F) * 4
                s = l4 + 8
                plen = _u16be(b, l4 + 4) - 8
                ok = plen >= 40 and s + plen <= d + cap and b[s] == 1 and _u16(b, s + 2) == 0x8005
            if not ok:
                stats[N_OTHER] += 1
            else:
                sess = _u32(b, s + 8)
                cnt = _u16(b, s + 14)
                first = _i64(b, s + 24)
                if sess != session:
                    if session >= 0:
                        stats[N_SESSIONS] += 1
                    session = sess
                    next_seq = -1
                if cnt == 0:
                    stats[N_HEARTBEAT] += 1
                q = s + 40
                end = s + plen
                for j in range(cnt):
                    if q + 2 > end:
                        stats[N_TRUNC] += 1
                        break
                    ml = _u16(b, q)
                    if q + 2 + ml > end:
                        stats[N_TRUNC] += 1
                        break
                    seq = first + j
                    if next_seq < 0 or seq >= next_seq:
                        if next_seq >= 0 and seq > next_seq:
                            stats[N_GAP] += seq - next_seq
                        if fill:
                            off[k] = q + 2
                            ln[k] = ml
                        k += 1
                        next_seq = seq + 1
                    else:
                        stats[N_DUP] += 1
                    q += 2 + ml
        p += blen
    stats[N_MSGS] = k
    return k


@dataclass
class Scan:
    off: np.ndarray  # int64: offset of each message's data (after its length field)
    ln: np.ndarray  # int32: message length
    stats: dict


def scan(buf: np.ndarray) -> Scan:
    if buf.shape[0] < 12 or _u32(buf, 0) != 0x0A0D0D0A or _u32(buf, 8) != 0x1A2B3C4D:
        raise ValueError("not a little-endian pcap-ng capture")
    stats = np.zeros(len(STAT_NAMES), np.int64)
    dummy = np.zeros(1, np.int64)
    n = _scan(buf, False, dummy, dummy.astype(np.int32), stats)
    off = np.empty(n, np.int64)
    ln = np.empty(n, np.int32)
    stats[:] = 0
    _scan(buf, True, off, ln, stats)
    return Scan(off, ln, dict(zip(STAT_NAMES, (int(x) for x in stats))))


@nb.njit(cache=True, parallel=True)
def _decode(b, off, ln, typ, code, ts, sym, oid, size, price, tid, symbolic):
    for i in nb.prange(off.shape[0]):
        o = off[i]
        t = b[o]
        typ[i] = t
        code[i] = b[o + 1] if ln[i] > 1 else 0
        ts[i] = _i64(b, o + 2) if ln[i] >= 10 else 0
        if symbolic[t] and ln[i] >= 18:
            sym[i] = _u64(b, o + 10)
        if (t == 0x61 or t == 0x4D or t == 0x4C) and ln[i] >= 38:  # a, M, L
            oid[i] = _i64(b, o + 18)
            size[i] = _u32(b, o + 26)
            price[i] = _i64(b, o + 30)
            if t == 0x4C and ln[i] >= 46:
                tid[i] = _i64(b, o + 38)
        elif t == 0x52 and ln[i] >= 26:  # R
            oid[i] = _i64(b, o + 18)
        elif (t == 0x54 or t == 0x42) and ln[i] >= 38:  # T, B
            size[i] = _u32(b, o + 18)
            price[i] = _i64(b, o + 22)
            tid[i] = _i64(b, o + 30)


def decode(buf: np.ndarray, sc: Scan) -> dict[str, np.ndarray]:
    n = sc.off.shape[0]
    out = {"typ": np.empty(n, np.uint8), "code": np.empty(n, np.uint8), "ts": np.empty(n, np.int64),
           "sym": np.zeros(n, np.uint64), "oid": np.zeros(n, np.int64), "size": np.zeros(n, np.int64),
           "price": np.zeros(n, np.int64), "tid": np.zeros(n, np.int64)}
    symbolic = np.zeros(256, np.bool_)
    symbolic[np.frombuffer(SYMBOLIC, np.uint8)] = True
    _decode(buf, sc.off, sc.ln, out["typ"], out["code"], out["ts"], out["sym"], out["oid"], out["size"], out["price"],
            out["tid"], symbolic)
    return out


def symbol_code(s: str) -> np.uint64:
    """The 8-byte, space-padded ASCII symbol as the little-endian integer `decode` stores."""
    raw = s.encode("ascii")[:8].ljust(8, b" ")
    return np.frombuffer(raw, "<u8")[0]


def symbol_name(code) -> str:
    return np.asarray(code, "<u8").tobytes().decode("ascii", "replace").rstrip()

"""Pass 1 over a decompressed ITCH 5.0 file: find, type and count every message.

`index_file(buf, t_snap, t_end)` walks the 2-byte length framing once to count
messages, then once more to record each message's offset (of its type byte),
type and stock locate, and per stock locate:

* `nrows[l]`: an upper bound on the rows the stock's event table will need
  (orders added before `t_snap`, which may become SNAP rows, plus every
  book, trade or state message in [t_snap, t_end); a replace counts twice);
* `counts[l, c]`: messages of type code `c` (ASCII) over the whole day.

It also checks the framing: every message length must match the
specification for its type, and the lengths must add up to the file size.
`order_by_locate` then groups the message numbers by stock (a stable
counting sort, so each stock's messages stay in exchange order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .spec import LEN_BY_CODE

_LEN = np.array(LEN_BY_CODE, dtype=np.int64)
N_LOC = 65536

# message types that produce rows in the event tables
_ROW_W = np.zeros(256, dtype=np.int64)
for _c in "AFECXDPQBHhJ":
    _ROW_W[ord(_c)] = 1
_ROW_W[ord("U")] = 2


@nb.njit(inline="always")
def u16(b, p):
    return (np.int64(b[p]) << 8) | np.int64(b[p + 1])


@nb.njit(inline="always")
def u32(b, p):
    return (np.int64(b[p]) << 24) | (np.int64(b[p + 1]) << 16) | (np.int64(b[p + 2]) << 8) | np.int64(b[p + 3])


@nb.njit(inline="always")
def u48(b, p):
    return (u16(b, p) << 32) | u32(b, p + 2)


@nb.njit(inline="always")
def u64(b, p):
    return (u32(b, p) << 32) | u32(b, p + 4)


@nb.njit(cache=True)
def _count(buf):
    n = buf.shape[0]
    pos = 0
    cnt = 0
    while pos + 2 <= n:
        L = u16(buf, pos)
        if L == 0 or pos + 2 + L > n:
            break
        pos += 2 + L
        cnt += 1
    return cnt, pos


@nb.njit(cache=True)
def _fill(buf, n_msgs, off, typ, loc, counts, nrows, length, row_w, t_snap, t_end):
    pos = 0
    max_ref = 0
    bad_len = 0
    unknown = 0
    fA, fF, fU = ord("A"), ord("F"), ord("U")
    for i in range(n_msgs):
        L = u16(buf, pos)
        p = pos + 2
        t = np.int64(buf[p])
        l = u16(buf, p + 1)
        off[i] = p
        typ[i] = t
        loc[i] = l
        counts[l, t] += 1
        exp = length[t]
        if exp == 0:
            unknown += 1
        elif exp != L:
            bad_len += 1
        w = row_w[t]
        if w > 0:
            ts = u48(buf, p + 5)
            if ts < t_snap:
                if t == fA or t == fF or t == fU:
                    nrows[l] += 1  # may still be resting at t_snap -> a SNAP row
            elif ts < t_end:
                nrows[l] += w
        if t == fA or t == fF:
            r = u64(buf, p + 11)
            if r > max_ref:
                max_ref = r
        elif t == fU:
            r = u64(buf, p + 19)
            if r > max_ref:
                max_ref = r
        pos = p + L
    return max_ref, bad_len, unknown


@dataclass
class Index:
    off: np.ndarray  # int64[n]  offset of each message's type byte
    typ: np.ndarray  # uint8[n]
    loc: np.ndarray  # uint16[n]
    counts: np.ndarray  # int64[65536, 256]
    nrows: np.ndarray  # int64[65536] upper bound on event-table rows per locate
    max_ref: int
    bad_len: int  # messages whose length differs from the specification
    unknown: int  # messages of a type not in the specification
    tail: int  # bytes left after the last complete message

    @property
    def n(self) -> int:
        return self.off.shape[0]


def index_file(buf: np.ndarray, t_snap: int, t_end: int) -> Index:
    n_msgs, end = _count(buf)
    off = np.empty(n_msgs, dtype=np.int64)
    typ = np.empty(n_msgs, dtype=np.uint8)
    loc = np.empty(n_msgs, dtype=np.uint16)
    counts = np.zeros((N_LOC, 256), dtype=np.int64)
    nrows = np.zeros(N_LOC, dtype=np.int64)
    max_ref, bad_len, unknown = _fill(buf, n_msgs, off, typ, loc, counts, nrows, _LEN, _ROW_W, t_snap, t_end)
    return Index(off, typ, loc, counts, nrows, int(max_ref), int(bad_len), int(unknown), int(buf.shape[0] - end))


@nb.njit(cache=True)
def _group(loc, start, perm):
    nxt = start[:-1].copy()
    for i in range(loc.shape[0]):
        l = loc[i]
        perm[nxt[l]] = i
        nxt[l] += 1


def order_by_locate(loc: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(perm, start): the messages of locate l are perm[start[l]:start[l+1]], in file order."""
    per_loc = counts.sum(axis=1)
    start = np.zeros(N_LOC + 1, dtype=np.int64)
    np.cumsum(per_loc, out=start[1:])
    perm = np.empty(loc.shape[0], dtype=np.int64)
    _group(loc, start, perm)
    return perm, start

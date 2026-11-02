"""Stock directory, system events and timestamp checks from an indexed ITCH file."""

from __future__ import annotations

import struct

import numba as nb
import numpy as np
import pandas as pd

from .index import Index, u48
from .spec import R_FIELDS


def stock_directory(buf: np.ndarray, idx: Index) -> pd.DataFrame:
    """One row per stock locate from the 'R' messages (the first one if repeated)."""
    rows, seen = [], set()
    for i in np.flatnonzero(idx.typ == ord("R")):
        p = int(idx.off[i])
        m = bytes(buf[p: p + 39])
        loc = struct.unpack_from(">H", m, 1)[0]
        if loc in seen:
            continue
        seen.add(loc)
        d = {"locate": loc}
        for name, o, fmt in R_FIELDS:
            v = struct.unpack_from(fmt if fmt.startswith(">") else ">" + fmt, m, o)[0]
            d[name] = v.decode().strip() if isinstance(v, bytes) else v
        rows.append(d)
    return pd.DataFrame(rows).sort_values("locate").reset_index(drop=True)


def system_events(buf: np.ndarray, idx: Index) -> dict[str, int]:
    """System event code -> timestamp (ns since midnight), e.g. 'Q' market open, 'M' market close."""
    ev = {}
    for i in np.flatnonzero(idx.typ == ord("S")):
        p = int(idx.off[i])
        code = chr(buf[p + 11])
        ev.setdefault(code, int.from_bytes(bytes(buf[p + 5: p + 11]), "big"))
    return ev


@nb.njit(cache=True)
def _ts_decreases(buf, off):
    n = 0
    prev = 0
    for i in range(off.shape[0]):
        ts = u48(buf, off[i] + 5)
        if ts < prev:
            n += 1
        prev = ts
    return n


def timestamp_decreases(buf: np.ndarray, idx: Index) -> int:
    """Messages whose timestamp is earlier than the previous message's (should be 0)."""
    return int(_ts_decreases(buf, idx.off))

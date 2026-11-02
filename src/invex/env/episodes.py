"""Episode tables: the frozen tasks every policy is scored on.

An episode is (venue, day, stock, t0, side, family, Q, T). Each start time
draws one task family and one cell:

    MS  Q = q * mean trade size,     T = h * mean inter-trade time   (microstructure units)
    KO  Q = q * Kyle-Obizhaeva size,  T = h * Kyle-Obizhaeva time   (frozen constants)
    P   Q in {100, 300} shares,       T in {10, 60, 300} s          (physical units; equities only)

with q in {0.5, 1, 2} and h in {10, 30, 100}; units come from the trailing
30-minute window ending at t0, whose statistics are stored with the episode
(invex.env.units.WINDOW_COLUMNS) so any unit system can be computed later.
Episodes that overlap a feed outage (Coinbase reconnects) are dropped.
Start times are uniform in [10:00, 15:30]; episodes longer than 30 minutes or
ending after 15:50 are dropped, as are those whose units are undefined. Every
episode has a key (blake2b of its definition) and a table hash is recorded
when a table is frozen.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .. import schema as S
from ..book.calib import calibrate
from ..sim import probe
from ..sim.tracker import ReplayBook
from ..venues import venue as venue_of
from .units import W_STAR, window_columns

NS = 1_000_000_000
FAMILIES = ("MS", "KO", "P")
Q_CELLS = (0.5, 1.0, 2.0)
H_CELLS = (10.0, 30.0, 100.0)
P_Q = (100, 300)
P_T = (10.0, 60.0, 300.0)
WINDOW_S = 1800
MAX_T_S = 1800


def _blake(*parts) -> str:
    return hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=10).hexdigest()


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def outages(book: ReplayBook) -> tuple[np.ndarray, np.ndarray]:
    """[start, end] of each feed outage (an OUTAGE state row up to the next 'T' row); empty for ITCH."""
    st = np.flatnonzero(book.kind == S.STATE)
    lo, hi, start = [], [], None
    for i in st:
        a = int(book.aux[i])
        if a == S.STATE_OUTAGE and start is None:
            start = int(book.ts[i])
        elif a == ord("T") and start is not None:
            lo.append(start)
            hi.append(int(book.ts[i]))
            start = None
    if start is not None:
        lo.append(start)
        hi.append(int(book.ts[-1]))
    return np.asarray(lo, np.int64), np.asarray(hi, np.int64)


def excluded(book: ReplayBook, t0: np.ndarray, T_ns: np.ndarray) -> np.ndarray:
    """Episodes that start outside continuous trading or whose span [t0, t0 + T] contains a row outside it (a
    halt, a pause, a collar, the guard around a cross, a feed outage): the replay is not a fair test there.
    The state at t0 is the state after the last row at or before t0. Bool per episode."""
    nc = np.concatenate([[0], np.cumsum((book.flags & S.F_CONTINUOUS) == 0)])
    t0 = np.asarray(t0, np.int64)
    lo = np.maximum(np.searchsorted(book.ts, t0, side="right") - 1, 0)
    hi = np.searchsorted(book.ts, t0 + np.asarray(T_ns, np.int64), side="right")
    return nc[hi] - nc[lo] > 0


def book_episodes(book: ReplayBook, venue: str, day: str, stock: str, open_ns: int, close_ns: int, n: int,
                  ko: dict, seed_tag: str = "episodes-v0", tick: int = 100, px_scale: float | None = None,
                  qty_scale: float | None = None) -> pd.DataFrame:
    """Up to n episodes on one replayed symbol-day (deterministic given the arguments). The venue
    (invex.venues) sets the task families and, unless given (per symbol-day), the price and size scales;
    `tick` is the symbol's tick in table price units."""
    v = venue_of(venue)
    families = v.families
    px_scale = v.px_scale if px_scale is None else px_scale
    qty_scale = v.qty_scale if qty_scale is None else qty_scale
    if px_scale is None or qty_scale is None:
        raise ValueError(f"{venue}: price and size scales are per symbol-day; pass them")
    cal = calibrate(book, open_ns, close_ns, tick, px_scale, qty_scale)
    rng = np.random.default_rng(seed_of(venue, day, stock, seed_tag))
    lo, hi, end = open_ns + 30 * 60 * NS, close_ns - 30 * 60 * NS, close_ns - 10 * 60 * NS
    if hi <= lo:
        return pd.DataFrame()
    t0s = np.sort(rng.integers(lo, hi, n))
    fam = rng.integers(0, len(families), n)
    qi = rng.integers(0, 3, n)
    hi_ = rng.integers(0, 3, n)
    pq = rng.integers(0, 2, n)
    sides = np.where(rng.random(n) < 0.5, 1, -1)
    mid0 = probe.mid_at(book, t0s)
    out_lo, out_hi = outages(book)
    rows = []
    for j in range(n):
        w = cal.window(int(t0s[j]), WINDOW_S)
        if w.trades <= 0 or not np.isfinite(w.sigma_1min) or w.sigma_1min <= 0 or not np.isfinite(mid0[j]):
            continue
        f = families[fam[j]]
        if f == "MS":
            Q = Q_CELLS[qi[j]] * w.trade_size
            T = H_CELLS[hi_[j]] * w.inter_trade
        elif f == "KO":
            x = w.kyle_obizhaeva_w() / W_STAR
            Q = Q_CELLS[qi[j]] * ko["c_q"] * (w.volume * 390.0 / (w.seconds / 60.0)) * x ** (-2 / 3)
            T = H_CELLS[hi_[j]] * ko["c_t"] * x ** (-2 / 3)
        else:
            Q = P_Q[pq[j]]
            T = P_T[hi_[j]]
        Q = int(max(1, round(Q)))
        T_ns = int(T * NS)
        if not (0 < T <= MAX_T_S) or t0s[j] + T_ns > end:
            continue
        if out_lo.size and ((out_lo <= t0s[j] + T_ns) & (out_hi >= t0s[j])).any():  # the feed was down
            continue
        cell = f"{f}:{Q_CELLS[qi[j]] if f != 'P' else P_Q[pq[j]]}:{H_CELLS[hi_[j]] if f != 'P' else P_T[hi_[j]]}"
        rows.append({"venue": venue, "day": day, "stock": stock, "t0": int(t0s[j]), "side": int(sides[j]),
                     "family": f, "cell": cell, "Q": Q, "T_ns": T_ns, "mid0": float(mid0[j]),
                     **window_columns(w), "px_scale": px_scale, "qty_scale": qty_scale})
    df = pd.DataFrame(rows)
    if len(df):
        df["key"] = [_blake(r.venue, r.day, r.stock, r.t0, r.side, r.cell, r.Q, r.T_ns) for r in df.itertuples()]
    return df


def table_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(df["key"])).encode()).hexdigest()

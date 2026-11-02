"""Trailing-window statistics of one symbol-day, for the unit systems (invex.env.units).

`calibrate(book, open_ns, close_ns)` integrates the replayed book on a
1-second grid of the continuous session: trades (aggressor runs plus hidden
prints), volume, notional, time-weighted spread, touch depth and one-tick
time, and the mid at every second. `Calib.window(t0, length)` then returns the
statistics of any window [t0 - length, t0) in O(1) (differences of cumulative
sums), so units can be recomputed at every episode start without lookahead.

A "trade" is one aggressor run (executions at one timestamp against one
side, flagged F_TRADE_START by the replay) or one hidden print; volume counts
printable executions and hidden prints, never auction crosses. Only rows
flagged F_CONTINUOUS count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba as nb
import numpy as np

from .. import schema as S
from ..sim.tracker import ReplayBook

NS = 1_000_000_000


@nb.njit(cache=True)
def _integrate(ts, kind, qty, px, aux, flags, bid_px, ask_px, bid_sz, ask_sz, open_ns, n_sec, tick,
               c_trades, c_vol, c_notional, c_spread, c_depth, c_onetick, mid):
    """Cumulative sums at second boundaries open_ns + s * 1e9 (s = 0..n_sec): value at s covers [open, t_s)."""
    n = ts.shape[0]
    trades = 0.0
    vol = 0.0
    notional = 0.0
    sp_int = 0.0
    dp_int = 0.0
    ot_int = 0.0
    # state between rows: spread / depth after the last processed row, valid from its time on
    cur_sp = -1.0
    cur_dp = 0.0
    cur_mid = -1.0
    t_prev = open_ns
    i = 0
    # skip rows before the open but keep their book state
    while i < n and ts[i] < open_ns:
        if bid_px[i] >= 0 and ask_px[i] >= 0:
            cur_sp = ask_px[i] - bid_px[i]
            cur_mid = 0.5 * (ask_px[i] + bid_px[i])
            cur_dp = 0.5 * (bid_sz[i] + ask_sz[i])
        else:
            cur_sp = -1.0
        i += 1
    for s in range(n_sec + 1):
        t_s = open_ns + s * 1_000_000_000
        while i < n and ts[i] < t_s:
            dt = ts[i] - t_prev
            if cur_sp > 0:
                sp_int += cur_sp * dt
                dp_int += cur_dp * dt
                if cur_sp <= tick:
                    ot_int += dt
            t_prev = ts[i]
            k = kind[i]
            cont = (flags[i] & S.F_CONTINUOUS) != 0
            if cont:
                if k == S.EXEC and (flags[i] & S.F_PRINTABLE) != 0:
                    if (flags[i] & S.F_TRADE_START) != 0:
                        trades += 1
                    vol += qty[i]
                    notional += qty[i] * aux[i]
                elif k == S.HIDDEN:
                    trades += 1
                    vol += qty[i]
                    notional += qty[i] * aux[i]
            if bid_px[i] >= 0 and ask_px[i] >= 0:
                cur_sp = ask_px[i] - bid_px[i]
                cur_mid = 0.5 * (ask_px[i] + bid_px[i])
                cur_dp = 0.5 * (bid_sz[i] + ask_sz[i])
            else:
                cur_sp = -1.0
            i += 1
        dt = t_s - t_prev
        if cur_sp > 0 and dt > 0:
            sp_int += cur_sp * dt
            dp_int += cur_dp * dt
            if cur_sp <= tick:
                ot_int += dt
        t_prev = t_s
        c_trades[s] = trades
        c_vol[s] = vol
        c_notional[s] = notional
        c_spread[s] = sp_int / 1e9
        c_depth[s] = dp_int / 1e9
        c_onetick[s] = ot_int / 1e9
        mid[s] = cur_mid


@dataclass
class Window:
    seconds: float
    trades: float
    volume: float
    vwap: float  # price units (ITCH: 1e-4 $)
    mid: float  # mid at the end of the window
    spread: float  # time-weighted mean spread, price units
    depth: float  # time-weighted mean touch depth (bid + ask) / 2, shares
    onetick: float  # share of time the spread was one tick
    sigma_1min: float  # std of 1-minute log mid returns inside the window
    tick: int
    px_scale: int = 10_000  # price units per dollar (invex.venues)
    qty_scale: int = 1  # size units per share or coin

    @property
    def inter_trade(self) -> float:  # seconds
        return self.seconds / max(self.trades, 1.0)

    @property
    def trade_size(self) -> float:  # shares
        return self.volume / max(self.trades, 1.0)

    @property
    def sigma_trade(self) -> float:
        """Volatility per trade, price units: P * sigma_window / sqrt(trades)."""
        minutes = self.seconds / 60.0
        return self.mid * self.sigma_1min * math.sqrt(minutes / max(self.trades, 1.0))

    def kyle_obizhaeva_w(self, session_minutes: float = 390.0) -> float:
        """Trading activity W = P * V * sigma per day, from the window scaled to a full session ($ units).
        The session length is a convention (W per 390 minutes at the window's rates) shared by all venues, so
        the KO units are the same function of per-minute activity everywhere."""
        minutes = self.seconds / 60.0
        v_day = self.volume * session_minutes / minutes
        sigma_day = self.sigma_1min * math.sqrt(session_minutes)
        return (self.vwap / self.px_scale) * (v_day / self.qty_scale) * sigma_day


@dataclass
class Calib:
    open_ns: int
    tick: int
    px_scale: int
    qty_scale: int
    trades: np.ndarray
    volume: np.ndarray
    notional: np.ndarray
    spread: np.ndarray
    depth: np.ndarray
    onetick: np.ndarray
    mid: np.ndarray

    def _s(self, t: int) -> int:
        return int(min(max((t - self.open_ns) // NS, 0), self.trades.shape[0] - 1))

    def window(self, t0: int, length_s: int) -> Window:
        b = self._s(t0)
        a = max(0, b - length_s)
        secs = float(b - a)
        d = lambda x: float(x[b] - x[a])  # noqa: E731
        vol, notional = d(self.volume), d(self.notional)
        # 1-minute log mid returns ending inside the window
        m = self.mid[a: b + 1: 60]
        m = m[m > 0]
        r = np.diff(np.log(m)) if m.shape[0] > 1 else np.zeros(0)
        return Window(
            seconds=secs, trades=d(self.trades), volume=vol,
            vwap=notional / vol if vol > 0 else float(self.mid[b]),
            mid=float(self.mid[b]), spread=d(self.spread) / max(secs, 1.0), depth=d(self.depth) / max(secs, 1.0),
            onetick=d(self.onetick) / max(secs, 1.0),
            sigma_1min=float(np.sqrt(np.mean(r * r))) if r.shape[0] >= 5 else float("nan"),
            tick=self.tick, px_scale=self.px_scale, qty_scale=self.qty_scale,
        )


def calibrate(book: ReplayBook, open_ns: int, close_ns: int, tick: int = 100, px_scale: int = 10_000,
              qty_scale: int = 1) -> Calib:
    n_sec = int((close_ns - open_ns) // NS)
    arrs = [np.zeros(n_sec + 1) for _ in range(7)]
    _integrate(book.ts, book.kind, book.qty, book.px, book.aux, book.flags, book.bid_px, book.ask_px,
               book.bid_sz, book.ask_sz, np.int64(open_ns), n_sec, float(tick), *arrs)
    return Calib(open_ns, tick, px_scale, qty_scale, *arrs)

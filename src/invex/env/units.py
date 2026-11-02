"""Unit systems: how time, size and price are measured for one asset at one moment.

Each system turns the trailing window before an episode (invex.book.calib.Window)
into a time unit (seconds), a size unit (shares) and a price unit (venue price
units). A policy or a statistic expressed in a system's units is "the same"
across assets exactly when that system's invariance hypothesis holds.

* R0 raw: 1 second, 100 shares (or coins), 1 cent (the same for every asset).
* R2 Kyle-Obizhaeva: W = P * V * sigma per day (window scaled to a session);
  time ~ (W/W*)^(-2/3) (one bet every 1/gamma), size ~ V * (W/W*)^(-2/3)
  (bet size), price ~ P * sigma * (W/W*)^(-1/3) (cost of a bet), W* = 0.02 * 40 * 1e6.
* R3 microstructure (Benzaquen-Donier-Bouchaud spread-cost units): time = mean
  inter-trade time, size = mean trade size, price = mean spread.
* R3s: as R3 with price = volatility per trade.
* R3d: as R3 with size = mean touch depth (predicted to alias assets whose
  queues differ in trade sizes).

The free constants of R2 (c_t, c_q, c_p) only rescale every asset alike; they
are set once so that, over the training asset-days, the median R2 unit equals
the median R3 unit (`fit_ko_constants`), which makes horizons and sizes in the
two systems comparable. Relative tick: eta_sigma = tick / sigma_trade (primary)
and eta_s = tick / spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..book.calib import Window

W_STAR = 0.02 * 40 * 1e6
SYSTEMS = ("R0", "R2", "R3", "R3s", "R3d")


@dataclass(frozen=True)
class Units:
    time: float  # seconds
    size: float  # shares
    price: float  # venue price units


@dataclass(frozen=True)
class KOConstants:
    c_t: float = 1.0
    c_q: float = 1.0
    c_p: float = 1.0


def units(w: Window, system: str, ko: KOConstants = KOConstants()) -> Units:
    if system == "R0":  # 1 s, 100 shares (coins), $0.01, in the window's venue units
        return Units(1.0, 100.0 * w.qty_scale, w.px_scale / 100.0)
    if system == "R2":
        x = w.kyle_obizhaeva_w() / W_STAR
        minutes = w.seconds / 60.0
        v_day = w.volume * 390.0 / max(minutes, 1e-9)
        sigma_day = w.sigma_1min * math.sqrt(390.0)
        return Units(ko.c_t * x ** (-2 / 3), ko.c_q * v_day * x ** (-2 / 3), ko.c_p * w.mid * sigma_day * x ** (-1 / 3))
    if system == "R3":
        return Units(w.inter_trade, w.trade_size, w.spread)
    if system == "R3s":
        return Units(w.inter_trade, w.trade_size, w.sigma_trade)
    if system == "R3d":
        return Units(w.inter_trade, w.depth, w.spread)
    raise ValueError(system)


def relative_tick(w: Window) -> tuple[float, float]:
    """(eta_sigma, eta_s) = (tick / volatility per trade, tick / mean spread)."""
    return w.tick / max(w.sigma_trade, 1e-12), w.tick / max(w.spread, 1e-12)


def fit_ko_constants(windows: list[Window]) -> KOConstants:
    """Constants that make the median R2 units equal the median R3 units over `windows`."""
    raw = np.array([[*units(w, "R2").__dict__.values()] for w in windows if np.isfinite(w.sigma_1min) and w.trades > 0])
    r3 = np.array([[*units(w, "R3").__dict__.values()] for w in windows if np.isfinite(w.sigma_1min) and w.trades > 0])
    c = np.median(r3, axis=0) / np.median(raw, axis=0)
    return KOConstants(float(c[0]), float(c[1]), float(c[2]))


WINDOW_COLUMNS = ("w_seconds", "w_trades", "w_volume", "w_vwap", "w_mid", "w_spread", "w_depth", "w_onetick",
                  "w_sigma_1min", "tick")


def window_columns(w: Window) -> dict:
    """The trailing-window statistics stored with every probe or episode (see unit_frame)."""
    return {"w_seconds": w.seconds, "w_trades": w.trades, "w_volume": w.volume, "w_vwap": w.vwap, "w_mid": w.mid,
            "w_spread": w.spread, "w_depth": w.depth, "w_onetick": w.onetick, "w_sigma_1min": w.sigma_1min,
            "tick": w.tick}


def unit_frame(df, ko: dict) -> dict:
    """Vectorised `units` for a table with WINDOW_COLUMNS: {system: DataFrame(t, q, p)} plus
    '_eta_sigma' (tick / volatility per trade) and '_activity' (trades per second)."""
    import pandas as pd

    secs = df["w_seconds"].to_numpy(float)
    trades = np.maximum(df["w_trades"].to_numpy(float), 1.0)
    vol = df["w_volume"].to_numpy(float)
    mid = df["w_mid"].to_numpy(float)
    sig = df["w_sigma_1min"].to_numpy(float)
    minutes = secs / 60.0
    v_day = vol * 390.0 / np.maximum(minutes, 1e-9)
    # venue scales (invex.venues); tables without them are ITCH-scaled
    pxs = df["px_scale"].to_numpy(float) if "px_scale" in df else np.full(len(df), 1e4)
    qts = df["qty_scale"].to_numpy(float) if "qty_scale" in df else np.ones(len(df))
    with np.errstate(divide="ignore", invalid="ignore"):
        W = (df["w_vwap"].to_numpy(float) / pxs) * (v_day / qts) * sig * np.sqrt(390.0)
        x = W / W_STAR
        inter = secs / trades
        tsize = vol / trades
        sig_tr = mid * sig * np.sqrt(minutes / trades)
        n = len(df)
        one = np.ones(n)
        tbl = {
            "R0": (one, 100 * qts, pxs / 100),
            "R2": (ko["c_t"] * x ** (-2 / 3), ko["c_q"] * v_day * x ** (-2 / 3),
                   ko["c_p"] * mid * sig * np.sqrt(390.0) * x ** (-1 / 3)),
            "R3": (inter, tsize, df["w_spread"].to_numpy(float)),
            "R3s": (inter, tsize, sig_tr),
            "R3d": (inter, df["w_depth"].to_numpy(float), df["w_spread"].to_numpy(float)),
        }
        if "h_inter" in df:  # R3h: microstructure units from the stock's own history (constants per stock)
            tbl["R3h"] = (df["h_inter"].to_numpy(float), df["h_tsize"].to_numpy(float), df["h_spread"].to_numpy(float))
        out = {u: pd.DataFrame({"t": t, "q": q, "p": pr}, index=df.index) for u, (t, q, pr) in tbl.items()}
        out["_eta_sigma"] = pd.Series(df["tick"].to_numpy(float) / sig_tr, index=df.index)
        out["_sigma_trade"] = pd.Series(sig_tr, index=df.index)
    out["_activity"] = pd.Series(trades / np.maximum(secs, 1.0), index=df.index)
    return out


def valid_units(uf: dict) -> np.ndarray:
    """Rows whose units are finite and positive in every system and whose eta_sigma is finite."""
    ok = np.isfinite(uf["_eta_sigma"].to_numpy())
    for u in SYSTEMS:
        v = uf[u].to_numpy()
        ok &= np.isfinite(v).all(axis=1) & (v > 0).all(axis=1)
    return ok

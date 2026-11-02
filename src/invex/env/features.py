"""Decision features, raw and in each unit system.

`FlowBook(book)` adds per-row cumulative signed aggressor volume, trade
counts and the time of the last trade to a ReplayBook. `raw_features` then
computes, for many episodes at their decision times, the RAW_NAMES quantities
in venue units (shares, ns -> seconds, price units), with order flow over
every lookback in LOOKBACKS_S. `represent(raw, units, rep)` divides each
quantity by the unit of its category (time, size or price) in representation
`rep` and keeps the flow at the lookback closest (in log time) to 10 time
units; tree models are invariant to monotone transforms of single features,
so the division is all a representation does for them.

Signs: flow and the mid move are oriented so that positive means adverse for
the agent (buy pressure / a rising price for a buyer).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .. import schema as S
from ..sim import tracker as tk
from .execution import S_REM

LOOKBACKS_S = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
NL = len(LOOKBACKS_S)

# raw feature layout
F_SPREAD, F_NEAR, F_FAR, F_IMB, F_ADV = 0, 1, 2, 3, 4
F_FLOW0 = 5  # NL columns
F_TRADES0 = F_FLOW0 + NL  # NL columns
F_SINCE = F_TRADES0 + NL
F_TLEFT, F_REM, F_R_NEAR, F_R_IN, F_R_BEHIND, F_AHEAD, F_Q, F_T = range(F_SINCE + 1, F_SINCE + 9)
N_RAW = F_T + 1
RAW_NAMES = (["spread", "near_size", "far_size", "imbalance", "adverse_move"]
             + [f"flow_{L:g}s" for L in LOOKBACKS_S] + [f"trades_{L:g}s" for L in LOOKBACKS_S]
             + ["since_trade", "time_left", "rem_frac", "rest_near", "rest_inside", "rest_behind", "ahead", "Q", "T"])
assert len(RAW_NAMES) == N_RAW


@nb.njit(cache=True)
def _flows(ts, kind, side, qty, aux, flags, bid_px, ask_px, cum_sv, cum_tr, last_tr):
    sv = 0
    tr = 0
    last = np.int64(-1)
    for i in range(ts.shape[0]):
        if flags[i] & S.F_CONTINUOUS:
            k = kind[i]
            if k == S.EXEC and (flags[i] & S.F_PRINTABLE):
                sv -= side[i] * qty[i]  # the aggressor traded against the resting side
                if flags[i] & S.F_TRADE_START:
                    tr += 1
                    last = ts[i]
            elif k == S.HIDDEN and i > 0:
                sg = 0
                if ask_px[i - 1] >= 0 and aux[i] >= ask_px[i - 1]:
                    sg = 1
                elif bid_px[i - 1] >= 0 and aux[i] <= bid_px[i - 1]:
                    sg = -1
                sv += sg * qty[i]
                tr += 1
                last = ts[i]
        cum_sv[i] = sv
        cum_tr[i] = tr
        last_tr[i] = last


@dataclass
class FlowBook:
    book: tk.ReplayBook
    cum_sv: np.ndarray
    cum_tr: np.ndarray
    last_tr: np.ndarray

    @classmethod
    def build(cls, book: tk.ReplayBook) -> "FlowBook":
        n = book.n
        cum_sv = np.zeros(n, np.int64)
        cum_tr = np.zeros(n, np.int64)
        last_tr = np.zeros(n, np.int64)
        _flows(book.ts, book.kind, book.side, book.qty, book.aux, book.flags, book.bid_px, book.ask_px,
               cum_sv, cum_tr, last_tr)
        return cls(book, cum_sv, cum_tr, last_tr)


@nb.njit(parallel=True, cache=True)
def _raw(ts, bid_px, bid_sz, ask_px, ask_sz, cum_sv, cum_tr, last_tr, states, sgns, mid0, t0, T, Q, k, K,
         tick, lookbacks_ns, out):
    nl = lookbacks_ns.shape[0]
    for e in nb.prange(states.shape[0]):
        sgn = sgns[e]
        t_k = t0[e] + (T[e] * k) // K
        j = tk.row_after(ts, t_k) - 1
        for f in range(out.shape[1]):
            out[e, f] = np.nan
        if j < 0:
            continue
        b, a = bid_px[j], ask_px[j]
        bs, as_ = bid_sz[j], ask_sz[j]
        if b >= 0 and a >= 0:
            out[e, F_SPREAD] = a - b
            mid = 0.5 * (a + b)
            out[e, F_ADV] = sgn * (mid - mid0[e])
        near_sz = bs if sgn > 0 else as_
        far_sz = as_ if sgn > 0 else bs
        near_px = b if sgn > 0 else a
        out[e, F_NEAR] = near_sz
        out[e, F_FAR] = far_sz
        tot = near_sz + far_sz
        out[e, F_IMB] = (near_sz - far_sz) / tot if tot > 0 else 0.0
        for li in range(nl):
            j0 = tk.row_after(ts, t_k - lookbacks_ns[li]) - 1
            sv0 = cum_sv[j0] if j0 >= 0 else 0
            tr0 = cum_tr[j0] if j0 >= 0 else 0
            out[e, F_FLOW0 + li] = sgn * (cum_sv[j] - sv0)
            out[e, F_TRADES0 + li] = cum_tr[j] - tr0
        lt = last_tr[j]
        out[e, F_SINCE] = (t_k - lt) / 1e9 if lt >= 0 else (t_k - t0[e]) / 1e9 + 3600.0
        out[e, F_TLEFT] = (K - k) / K
        st = states[e]
        out[e, F_REM] = st[S_REM] / Q[e]
        resting = st[tk.ST_ACTIVE] == 1
        out[e, F_R_NEAR] = 0.0
        out[e, F_R_IN] = 0.0
        out[e, F_R_BEHIND] = 0.0
        out[e, F_AHEAD] = 0.0
        if resting:
            p = st[tk.ST_PX]
            if near_px >= 0 and p == near_px:
                out[e, F_R_NEAR] = 1.0
            elif near_px >= 0 and sgn * (p - near_px) > 0:
                out[e, F_R_IN] = 1.0
            else:
                out[e, F_R_BEHIND] = 1.0
            out[e, F_AHEAD] = st[tk.ST_AHEAD]
        out[e, F_Q] = Q[e]
        out[e, F_T] = T[e] / 1e9


def raw_features(fb: FlowBook, states, sgns, mid0, t0, T, Q, k: int, K: int, tick: int = 100) -> np.ndarray:
    out = np.empty((len(sgns), N_RAW))
    b = fb.book
    _raw(b.ts, b.bid_px, b.bid_sz, b.ask_px, b.ask_sz, fb.cum_sv, fb.cum_tr, fb.last_tr, states,
         np.asarray(sgns, np.int64), np.asarray(mid0, float), np.asarray(t0, np.int64), np.asarray(T, np.int64),
         np.asarray(Q, np.int64), k, K, tick, (np.array(LOOKBACKS_S) * 1e9).astype(np.int64), out)
    return out


# --- representations -----------------------------------------------------------------------------

REPS = ("R0", "R0desc", "R1", "R2", "R3", "R3s", "R3d", "R2tick", "R3tick", "Rpi", "Rdim", "R0r")
# the unit system of each representation's features (and of its rule probes);
# R1 standardises features by the asset's own statistics and measures cost in its own mean spread
BASE_SYSTEM = {"R0": "R0", "R0desc": "R0", "R1": "R3", "R2": "R2", "R3": "R3", "R3s": "R3s", "R3d": "R3d",
               "R2tick": "R2", "R3tick": "R3", "Rpi": "R3", "Rdim": "R3", "R0r": "R0"}
# the unit system of its reward: the feature system, except for the ablation R0r (raw features, cost in the
# stock's mean spread), which separates what raw features cost from what a reward in cents costs
REWARD_SYSTEM = {**BASE_SYSTEM, "R0r": "R3"}
# asset-aware comparators (the stock's identity as a categorical feature; not a representation of the market)
# (R3id, tick-free: a positive control, since the tick is a fixed property of the stock that its identity carries)
for _r in ("R0", "R3", "R3tick"):
    BASE_SYSTEM[_r + "id"] = BASE_SYSTEM[_r]
    REWARD_SYSTEM[_r + "id"] = REWARD_SYSTEM[_r]
# MS+tick plus raw asset descriptors: does anything the units leave out help?
BASE_SYSTEM["R3tickdesc"] = REWARD_SYSTEM["R3tickdesc"] = "R3"
# the stock's own history (per-stock constants from its 2019 training days): R3h, microstructure units taken from it
# (a stock-aware positive control: its scale comes only from the stock's past), and R3tickh, MS+tick plus how the
# current window and the tick compare with that history (a stock-aware comparator)
BASE_SYSTEM["R3h"] = REWARD_SYSTEM["R3h"] = "R3h"
BASE_SYSTEM["R3tickh"] = REWARD_SYSTEM["R3tickh"] = "R3"
# MS+tick plus the stock's probe profile (per-stock constants from its 2019 training days): what simple rules saved
# over crossing at once, in spreads, which carries fill and adverse-selection differences that the units do not
# (a stock-aware comparator; experiments/stock_profile.py)
BASE_SYSTEM["R3tickp"] = REWARD_SYSTEM["R3tickp"] = "R3"
PROFILE_COLS = ("p_post", "p_wait", "p_improve", "p_queue")
# equal information without the normalisation (review round 1): raw features plus every window statistic any unit
# system is built from (the MS units' mean inter-trade time, trade size and spread; price, volume rate, one-minute
# volatility, per-trade volatility and tick), with the cost in the stock's mean spread as for MS. A learner given
# this can form any unit system's ratios itself (the neural learners see inputs as signed logs, so a ratio is a
# difference); if it matches MS+tick, the units are a parametrisation rather than information the policy needs
BASE_SYSTEM["R0info"] = "R0"
REWARD_SYSTEM["R0info"] = "R3"
INFO_COLS = ("info_inter", "info_tsize", "info_spread", "price", "volume_rate", "sigma_1min", "sigma_trade", "tick")


def _pick_lookback(raw: np.ndarray, first: int, lookback_s: np.ndarray) -> np.ndarray:
    """Column first..first+NL-1 interpolated (log time) at each row's lookback."""
    lg = np.log(np.array(LOOKBACKS_S))
    x = np.log(np.clip(lookback_s, LOOKBACKS_S[0], LOOKBACKS_S[-1]))
    i = np.clip(np.searchsorted(lg, x) - 1, 0, NL - 2)
    w = (x - lg[i]) / (lg[i + 1] - lg[i])
    r = np.arange(raw.shape[0])
    return raw[r, first + i] * (1 - w) + raw[r, first + i + 1] * w


def represent(raw: np.ndarray, u_t, u_q, u_p, rep: str, lookback_units: float = 10.0, ctx: dict | None = None):
    """Features of representation `rep` from raw features and per-row units (seconds, shares, price units).

    ctx (per-row arrays) carries 'tick', 'sigma_trade', and for R0desc 'price', 'volume_rate', 'sigma_1min'."""
    u_t, u_q, u_p = (np.asarray(x, float) for x in (u_t, u_q, u_p))
    flow = _pick_lookback(raw, F_FLOW0, lookback_units * u_t)
    trades = _pick_lookback(raw, F_TRADES0, lookback_units * u_t)
    cols = [raw[:, F_SPREAD] / u_p, raw[:, F_NEAR] / u_q, raw[:, F_FAR] / u_q, raw[:, F_IMB],
            raw[:, F_ADV] / u_p, flow / u_q, trades, raw[:, F_SINCE] / u_t, raw[:, F_TLEFT], raw[:, F_REM],
            raw[:, F_R_NEAR], raw[:, F_R_IN], raw[:, F_R_BEHIND], raw[:, F_AHEAD] / u_q, raw[:, F_Q] / u_q,
            raw[:, F_T] / u_t]
    names = ["spread", "near_size", "far_size", "imbalance", "adverse_move", "flow", "trades", "since_trade",
             "time_left", "rem_frac", "rest_near", "rest_inside", "rest_behind", "ahead", "Q", "T"]
    if rep.endswith("tick") or rep.endswith("tickdesc") or rep.endswith("tickh") or rep.endswith("tickp"):
        cols += [np.log(ctx["tick"] / u_p), np.log(ctx["tick"] / ctx["sigma_trade"])]
        names += ["log_tick_price_unit", "log_tick_sigma_trade"]
    if rep.endswith("tickh"):  # the current units and the tick against the stock's own history
        cols += [np.log(u_t / ctx["h_inter"]), np.log(u_q / ctx["h_tsize"]), np.log(u_p / ctx["h_spread"]),
                 np.log(ctx["tick"] / ctx["h_spread"])]
        names += ["log_inter_vs_history", "log_size_vs_history", "log_spread_vs_history", "log_tick_history_spread"]
    if rep.endswith("tickp"):  # the stock's probe profile, as is (already in spreads)
        cols += [ctx[c] for c in PROFILE_COLS]
        names += list(PROFILE_COLS)
    if rep.endswith("info"):  # R0info: every window statistic, in physical units
        cols += [ctx[c] for c in INFO_COLS]
        names += list(INFO_COLS)
    if rep.endswith("desc"):  # R0desc; R3tickdesc (MS+tick plus the same descriptors, a check on what units omit)
        cols += [ctx["price"], ctx["volume_rate"], ctx["sigma_1min"], ctx["tick"]]
        names += ["price", "volume_rate", "sigma_1min", "tick"]
    return np.column_stack(cols), names


# --- R1: per-feature z-scores from the asset's own trailing window -----------------------------------

R1_FEATURES = (F_SPREAD, F_NEAR, F_FAR, F_IMB, F_SINCE) + tuple(range(F_FLOW0, F_FLOW0 + NL)) + \
    tuple(range(F_TRADES0, F_TRADES0 + NL))


def r1_stats(fb: FlowBook, eps, window_s: float = 1800.0, m: int = 60, tick: int = 100):
    """Mean and sd of each market feature over m evenly spaced times in the trailing window before each
    episode (episode's own side, no resting order), plus the sd of the mid move over the episode's
    horizon inside the window. Returns {column name: array} to attach to the episode table."""
    from ..sim.probe import mid_at
    from .execution import new_states

    n = len(eps)
    t0 = eps["t0"].to_numpy(np.int64)
    T = eps["T_ns"].to_numpy(np.int64)
    side = eps["side"].to_numpy(np.int64)
    Q = eps["Q"].to_numpy(np.int64)
    W = int(window_s * 1e9)
    frac = (np.arange(m) + 0.5) / m
    times = (t0[:, None] - W + (frac[None, :] * W)).astype(np.int64)  # (n, m)
    flat_t = times.ravel()
    rep_side = np.repeat(side, m)
    rep_T = np.repeat(T, m)
    rep_Q = np.repeat(Q, m)
    mid = mid_at(fb.book, flat_t)
    st = new_states(rep_Q)
    raw = raw_features(fb, st, rep_side, np.nan_to_num(mid), flat_t, rep_T, rep_Q, 0, 1, tick).reshape(n, m, N_RAW)
    out = {}
    with np.errstate(all="ignore"):
        for f in R1_FEATURES:
            x = raw[:, :, f]
            out[f"r1mu_{f}"] = np.nanmean(x, axis=1)
            out[f"r1sd_{f}"] = np.nanstd(x, axis=1)
        # price move over the horizon, from pairs (t, t + T) inside the window
        later = np.minimum(times + T[:, None], t0[:, None])
        move = (mid_at(fb.book, later.ravel()) - mid).reshape(n, m)
        ok = (times + T[:, None]) <= t0[:, None]
        move = np.where(ok, move, np.nan)
        out["r1_move_sd"] = np.nanstd(move, axis=1)
    return out


def represent_r1(raw: np.ndarray, ctx: dict, lookback_s: np.ndarray):
    """R1 features: z-scores of market features by the asset's own window statistics; queue ahead and order
    size per mean touch size, the horizon per mean time since the last trade, the mid move per its own sd."""
    def z(f, x=None):
        x = raw[:, f] if x is None else x
        sd = ctx[f"r1sd_{f}"]
        return (x - ctx[f"r1mu_{f}"]) / np.where(sd > 0, sd, np.nan)

    # flow and trade counts at the lookback (fixed seconds, like raw units), z-scored at the nearest lookback
    li = np.clip(np.searchsorted(np.log(np.array(LOOKBACKS_S)), np.log(lookback_s)), 0, NL - 1)
    r = np.arange(raw.shape[0])

    def z_block(first):
        mu = np.column_stack([ctx[f"r1mu_{first + i}"] for i in range(NL)])[r, li]
        sd = np.column_stack([ctx[f"r1sd_{first + i}"] for i in range(NL)])[r, li]
        return (raw[r, first + li] - mu) / np.where(sd > 0, sd, np.nan)

    flow = z_block(F_FLOW0)
    trades = z_block(F_TRADES0)
    near_mu = np.where(ctx[f"r1mu_{F_NEAR}"] > 0, ctx[f"r1mu_{F_NEAR}"], np.nan)
    since_mu = np.where(ctx[f"r1mu_{F_SINCE}"] > 0, ctx[f"r1mu_{F_SINCE}"], np.nan)
    move_sd = np.where(ctx["r1_move_sd"] > 0, ctx["r1_move_sd"], np.nan)
    cols = [z(F_SPREAD), z(F_NEAR), z(F_FAR), z(F_IMB), raw[:, F_ADV] / move_sd, flow, trades, z(F_SINCE),
            raw[:, F_TLEFT], raw[:, F_REM], raw[:, F_R_NEAR], raw[:, F_R_IN], raw[:, F_R_BEHIND],
            raw[:, F_AHEAD] / near_mu, raw[:, F_Q] / near_mu, raw[:, F_T] / since_mu]
    return np.column_stack(cols)


def represent_pi(raw: np.ndarray, tick: np.ndarray, K: int = 10):
    """Rpi: ratios inside the current state, no per-asset calibration. Price unit = current spread, size unit =
    current near-touch size, time unit = the horizon T; flow and trade counts over the last decision interval T/K."""
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(raw[:, F_SPREAD] > 0, raw[:, F_SPREAD], np.nan)
        near = np.where(raw[:, F_NEAR] > 0, raw[:, F_NEAR], np.nan)
        touch = np.where(raw[:, F_NEAR] + raw[:, F_FAR] > 0, raw[:, F_NEAR] + raw[:, F_FAR], np.nan)
        T = raw[:, F_T]
        lb = np.maximum(T / K, LOOKBACKS_S[0])
        flow = _pick_lookback(raw, F_FLOW0, lb)
        trades = _pick_lookback(raw, F_TRADES0, lb)
        cols = [tick / spread, raw[:, F_FAR] / near, raw[:, F_IMB], raw[:, F_ADV] / spread, flow / touch, trades,
                raw[:, F_SINCE] / T, raw[:, F_TLEFT], raw[:, F_REM], raw[:, F_R_NEAR], raw[:, F_R_IN], raw[:, F_R_BEHIND],
                raw[:, F_AHEAD] / near, raw[:, F_Q] / near]
    return np.column_stack(cols)


def represent_dim(raw: np.ndarray):
    """Rdim: the dimensionless core only (imbalance, time and shares left, order status)."""
    return np.column_stack([raw[:, F_IMB], raw[:, F_TLEFT], raw[:, F_REM], raw[:, F_R_NEAR], raw[:, F_R_IN],
                            raw[:, F_R_BEHIND]])

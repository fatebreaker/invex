"""Policies for the make-or-take task, and paired evaluation on frozen episodes.

A policy maps (stage k, raw features, valid-action masks, episodes, unit frame) to actions. Baselines:

    Immediate        cross everything at the first decision
    PostThenCross    join the near touch and wait; the environment crosses the rest at the end
    PegThenCross     join the near touch and re-join it whenever the quote has moved away from the order (losing
                     priority), the usual simple passive tactic; the environment crosses the rest at the end
    QueueRule        join, then cross once the queue ahead (size units) exceeds theta * time left (time units)
                     or the imbalance against the agent exceeds iota; theta, iota in a unit system
    FQIPolicy        greedy on a fitted Q model (invex.agents.fqi)

`evaluate_book` replays every policy on the same episodes of one symbol-day and
returns one row per (episode, policy): raw cost (price units x shares),
shares filled (all), shares filled passively (resting), shortfall and rejected posts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..env import execution as ex
from ..env.features import F_AHEAD, F_IMB, F_R_BEHIND, F_R_IN, F_R_NEAR, F_T, F_TLEFT, FlowBook, raw_features
from ..sim import tracker as tk


class Immediate:
    name = "immediate"

    def act(self, k, raw, masks, eps, uf):
        return np.where(masks[:, ex.CROSS] == 1, ex.CROSS, ex.HOLD)


class PostThenCross:
    name = "post_then_cross"

    def act(self, k, raw, masks, eps, uf):
        resting = raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND] > 0
        return np.where(~resting & (masks[:, ex.JOIN] == 1), ex.JOIN, ex.HOLD)


class PegThenCross:
    name = "peg_then_cross"

    def act(self, k, raw, masks, eps, uf):
        at_touch = raw[:, F_R_NEAR] + raw[:, F_R_IN] > 0  # resting at the near touch or inside the spread
        a = np.where(at_touch, ex.HOLD, ex.JOIN)  # not resting, or left behind by the quote: (re)join the touch
        return np.where(np.take_along_axis(masks, a[:, None], 1)[:, 0] == 1, a, ex.HOLD)


class QueueRule:
    def __init__(self, theta: float, iota: float, system: str = "R3", name: str | None = None):
        self.theta, self.iota, self.system = theta, iota, system
        self.name = name or f"queue_{system}_{theta:g}_{iota:g}"

    def act(self, k, raw, masks, eps, uf):
        u = uf[self.system]
        resting = raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND] > 0
        t_left_units = raw[:, F_TLEFT] * raw[:, F_T] / u["t"].to_numpy()
        ahead_units = raw[:, F_AHEAD] / u["q"].to_numpy()
        cross = (ahead_units > self.theta * t_left_units) | (raw[:, F_IMB] > self.iota)
        a = np.where(resting, np.where(cross & (k > 0), ex.CROSS, ex.HOLD), ex.JOIN)
        return np.where(np.take_along_axis(masks, a[:, None], 1)[:, 0] == 1, a, ex.HOLD)


class FQIPolicy:
    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or f"fqi_{model.rep}"

    def act(self, k, raw, masks, eps, uf):
        from ..agents.fqi import rep_inputs

        X = rep_inputs(raw, eps, uf, self.model.rep, self.model.c_lookback)
        q = self.model.q_values(k, X)
        q[masks == 0] = np.inf
        return np.argmin(q, axis=1)


def evaluate_book(book: tk.ReplayBook, eps: pd.DataFrame, uf: dict, policies: list, K: int, latency: int,
                  rule: int = tk.EXACT, tick: int = 100, markouts: bool = False) -> pd.DataFrame:
    fb = FlowBook.build(book)
    arrs = book.arrays()
    sg = eps["side"].to_numpy(np.int64)
    t0 = eps["t0"].to_numpy(np.int64)
    T = eps["T_ns"].to_numpy(np.int64)
    Q = eps["Q"].to_numpy(np.int64)
    mid0 = eps["mid0"].to_numpy(float)
    n = len(eps)
    out = []
    for pol in policies:
        st = ex.new_states(Q)
        total = np.zeros(n)
        for k in range(K):
            raw = raw_features(fb, st, sg, mid0, t0, T, Q, k, K, tick)
            m = np.zeros((n, ex.N_ACT), np.int64)
            ex.masks_stage(*arrs, st, sg, t0, T, k, K, tick, m)
            a = np.asarray(pol.act(k, raw, m, eps, uf), dtype=np.int64)
            a = np.where(np.take_along_axis(m, a[:, None], 1)[:, 0] == 1, a, ex.HOLD)
            c = np.zeros(n)
            ex.rollout_stage(*arrs, st, sg, mid0, t0, T, k, K, a, latency, tick, rule, c)
            total += c
        out.append(pd.DataFrame({"key": eps["key"].to_numpy(), "policy": pol.name, "cost": total,
                                 "filled": st[:, ex.S_FILLED], "passive": st[:, ex.S_PASSIVE],
                                 "shortfall": st[:, ex.S_SHORT],
                                 "rejected": st[:, ex.S_REJ],
                                 # under a markout rule: the booked markouts and shares x spread at the fill
                                 **({"mark": st[:, ex.S_MARK], "mspr": st[:, ex.S_MSPR]} if markouts else {})}))
    return pd.concat(out, ignore_index=True)

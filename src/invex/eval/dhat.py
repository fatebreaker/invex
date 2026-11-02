"""D-hat: an RL-free distance between an asset's dynamics and the training mixture, per unit system.

The zero-shot regret of a policy class on asset b is bounded by twice the
largest difference, over the class, between a policy's normalised value on b
and on the training mixture S (the simulation-lemma argument; the paths are
exogenous, so no horizon factor). D-hat estimates that sup with a fixed probe
family of simple rules expressed in the representation's own units:

    D_u(b) = max_m | mean_b J_u(pi_m) - mean_S J_u(pi_m) |

where J_u is the cost per share divided by u's price unit. Probes run on the
asset's training-day episodes only (never on evaluation days), so D-hat is
available before deployment. Under exact similarity in u it is 0 up to noise.

probe_policies(system) returns the family: cross at decision j (j = 0, 2, 5,
8), post-then-cross, improve-then-cross, and the queue/imbalance rules
QueueRule(theta, iota, system) on a 4 x 4 grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..env import execution as ex
from ..env.features import F_R_BEHIND, F_R_IN, F_R_NEAR
from . import policies as P


class CrossAt:
    def __init__(self, j: int):
        self.j = j
        self.name = f"cross_at_{j}"

    def act(self, k, raw, masks, eps, uf):
        if k >= self.j:
            return np.full(raw.shape[0], ex.CROSS)
        resting = raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND] > 0
        return np.where(~resting & (masks[:, ex.JOIN] == 1), ex.JOIN, ex.HOLD)


class ImproveThenCross:
    name = "improve_then_cross"

    def act(self, k, raw, masks, eps, uf):
        resting = raw[:, F_R_NEAR] + raw[:, F_R_IN] + raw[:, F_R_BEHIND] > 0
        a = np.where(masks[:, ex.IMPROVE] == 1, ex.IMPROVE, ex.JOIN)
        return np.where(~resting, a, ex.HOLD)


def probe_policies(system: str) -> list:
    pols = [CrossAt(j) for j in (0, 2, 5, 8)] + [P.PostThenCross(), ImproveThenCross()]
    for th in (0.3, 1.0, 3.0, 10.0):
        for io in (0.2, 0.4, 0.6, 0.8):
            pols.append(P.QueueRule(th, io, system, name=f"q_{th:g}_{io:g}"))
    return pols


def dhat(costs: pd.DataFrame, train_assets: set, by: str = "stock") -> pd.Series:
    """costs: one row per (episode, probe) with columns [by, 'policy', 'j'] (normalised cost).

    Returns D-hat per asset: the largest absolute difference between the asset's mean probe cost and
    the training assets' (each training asset weighted equally)."""
    per = costs.groupby([by, "policy"])["j"].mean().unstack("policy")
    ref = per.loc[per.index.isin(train_assets)].mean()
    return (per - ref).abs().max(axis=1)

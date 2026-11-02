"""Zero-shot transfer across a family of torque-limited pendulums: a second domain outside finance (exploratory).

    source scripts/env.sh
    python experiments/pendulum/pendulum.py check                        # similarity check, grid value iteration
    python experiments/pendulum/pendulum.py pilot --seeds 0 1 2          # natural units only: does FQI swing up?
    python experiments/pendulum/pendulum.py run --workers 16 --threads 1  # every fit of (a)-(c); resumable
    python experiments/pendulum/pendulum.py report                       # JSON, macros and figure

Environment. A point-mass pendulum, upright at theta = 0:
    theta'' = (g/l) sin(theta) - b/(m l^2) theta' + u/(m l^2),
integrated with semi-implicit Euler at step dt, theta wrapped to (-pi, pi]. Natural units: time tau0 = sqrt(l/g),
angular velocity 1/tau0, torque m g l. The dimensionless groups are the same for every system: u_max/(m g l) = 0.5,
b tau0/(m l^2) = 0.1, dt/tau0 = 0.1, horizon H = 200 steps, discount 0.98. In natural units the update is
    Omega' = Omega + (dt/tau0) (sin theta - 0.1 Omega + u/(m g l)),  theta' = theta + (dt/tau0) Omega',
the same map for every (m, l, g), so a policy in natural units is exactly invariant (up to rounding).
Systems draw (m, l, g) log-uniformly in [0.3, 3] x [0.3, 3] x [3, 30], so tau0 lies in [0.1, 1]. Out-of-range
test systems draw (m, l, g) the same way and then move l and g at fixed l*g (so m g l keeps its training
distribution) to tau0 log-uniform in [3, 9] (slow) or [1/90, 1/30] (fast), at least 3x outside [0.1, 1].
Actions: torques {-1, -0.5, 0, 0.5, 1} x u_max. Cost per step (natural): c~ = (1 - cos theta) + 0.1 (omega tau0)^2
+ 0.01 (u/(m g l))^2; in raw units c_raw = m g l c~.

Representations (features; training cost): nat (cos, sin, omega tau0; c~), raw (cos, sin, omega in rad/s; c_raw),
rawr = raw^s (raw features; c~), rawi = raw+info (cos, sin, omega, tau0; c~), rawd = raw+desc (cos, sin, omega, m,
l, g; c_raw). Every policy is scored by the mean c~ per step over H steps from the hanging position
(theta0 = pi + 0.05 xi, omega0 tau0 = 0.05 xi', 16 episodes per test system, the same draws for every policy).

Learner. Discounted fitted Q-iteration (gamma 0.98, 50 iterations, 200 boosting rounds per iteration) with the
registered tree settings of the paper (invex.agents.fqi: LightGBM, 63 leaves, learning rate 0.05, >= 200 samples
per leaf, feature fraction 0.9, bagging 0.8, L2 penalty 1, deterministic; the action a categorical input), on
branched data as in the paper: every sampled state carries all five actions' one-step costs and next states.
States are uniform: theta uniform, omega tau0 uniform in [-8, 8]. Each iteration sets
y_a(s) = c(s, a) + gamma min_a' Q(s'_a, a') with the previous iteration's Q and, as the registered learner regresses
advantages rather than the cost-to-go, splits it per state into the mean over actions, V(s) = mean_a y_a(s) (one
regressor on s), and the deviations y_a(s) - V(s) (one regressor A on (s, a)); Q = V + A, and the greedy policy is
argmin_a A(s, a). (The baseline is learned here: the pendulum has no action with an exactly known cost-to-go, which
"crossing now" is in execution.)

Two changes to the specification, both forced by pilots (run `check` and `pilot --plain` to reproduce):
  * dt/tau0 = 0.1, not 0.05: at 0.05, 50 iterations of gamma = 0.98 look ahead only 2.5 tau0, and even exact grid
    value iteration truncated at 50 iterations does not swing the pendulum up within 200 steps;
  * advantage targets: plain FQI (one regressor on (s, a) fitting the cost-to-go itself) never lifted the pendulum,
    since the action moves the cost-to-go far less than the state does.
Pilots in natural units (`pilot`, one system's data, which in natural units is the same as any mix of systems):
with 100 rounds, 13,107 states were unreliable (one seed in three never balanced) and 50,688 states left one
episode in 32 hanging on one seed in three (costs 0.649, 0.805, 0.659); 150 rounds on 50,688 states (0.621,
0.603) and 200 rounds on 36,864 states (0.607, 0.607, 0.604) swung up every episode. Hence STATES = 576 per
system (36,864 states, 184,320 branched transitions per fit) and 200 rounds, the most consistent setting.

Design (3 seeds; a seed draws the systems, the data and the learner's randomness). 64 training systems, 32 in-range
and 16 out-of-range test systems. All fits have the same number of training states, 64 x STATES.
  (a) zero-shot: STATES states from each training system; seen: one comparator per group of 8 in-range test systems,
      trained on the 64 training systems plus those 8, each with 64 STATES / 72 states (a test system's share 1/72).
  (b) the same zero-shot policies on the out-of-range systems; seen comparators per group of 8 (4 slow, 4 fast).
  (c) eps sweep (nat and raw) on 4 in-range test systems per seed, spread over their range of tau0 (chosen from the
      draws before any fit): comparators in which each of G test systems has share eps of the data and the
      training systems share 1 - G eps, for (eps, G) = (1/64, 4), (1/16, 4), (1/4, 2), (1/2, 1).
  (d) added after (b) showed out-of-range seen policies failing as well: a pooled seen comparator per seed and
      representation (nat, raw, rawr, rawi), with all 16 out-of-range systems at half the data (1/32 each) and the
      training systems the other half, as in PC1x where the out-of-range test assets were a large part of the
      seen data.
Transfer gap TG = zero-shot cost - seen cost, per test system. Intervals: 95% percentile bootstrap over test systems,
stratified by seed (10,000 resamples), of the mean over seeds and systems. These intervals hold the trained policies
fixed; in natural units, where zero-shot and seen policies learn the same problem, TG measures training noise alone.

Follow-ups (review round 4, R5; additive: the fits and macros above are unchanged, and `report` analyses the original
design on the original fits only):
  * a neural learner (`fqi_nn`): the same branched data, advantage split and gamma, 50 iterations; two MLPs (2 x 256
    ReLU; V with one output, A with one per action) on the paper's neural-learner inputs (sign(x) log(1+|x|),
    standardised), targets scaled to unit RMS, Adam 1e-3, batches of 1,024, 10 epochs per iteration, warm-started.
    Settings fixed in natural-unit pilots (`pilot --learner nn`, recorded in results/transfer/pendulum_nn_pilots.jsonl):
    ReLU/10 epochs swung up every episode on 3 seeds (0.683, 0.742, 0.736); tanh did too (0.740, 0.751, 0.724); 20
    epochs left one seed unbalanced. Fits (`run --plan nn --device cuda:0`, keys nn_*): zero-shot for every
    representation; in-range seen (1/72) and pooled out-of-range comparators for nat, raw, rawi. GPU 4, 6 or 7.
  * trees with 4x data per system (2,304 states; `run --plan x4`, keys x4_*): zero-shot nat and rawi.
  * the eps sweep on seeds 3..8 as well (`run --plan sweep`): nat and raw zero-shot and sweep fits; the sweep is then
    summarised over all 9 seeds with a two-level bootstrap (seeds, then test systems within a seed), which includes
    the noise of training, with the raw-minus-nat difference of transfer gaps, its slope in eps and the 1/2 - 1/64
    contrast. Macros PendNn*, PendXfour*, PendSweep*.

Outputs: results/transfer/pendulum.json, paper/numbers_pendulum.tex, paper/figures/fig_pendulum.pdf. Each fit's
evaluation is cached as one JSON file (--cache, default results/transfer/pendulum_cache), so `run` resumes and
`report` reads only stored evaluations (`report --from-json` rebuilds the macros and figure from the JSON alone).
The run of 31 October 2026: 165 fits, 16 single-thread workers, about 14 minutes per fit, 2.9 hours in all.

CARL's pendulum (review round 5, R4; exploratory, post hoc): gym's Pendulum-v1 with (g, l, m) as context and dt,
max_speed, max_torque fixed, so natural units are only approximately invariant. Subcommands carl-check, carl-pilot,
carl-oracle, carl-run and carl-report; see CARL_DOC below. It writes only results/transfer/carl_cache/,
results/transfer/carl_*.json(l), paper/numbers_carl.tex and paper/figures/fig_carl.pdf.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THREADS = "4"
if "--threads" in sys.argv:
    _THREADS = sys.argv[sys.argv.index("--threads") + 1]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THREADS

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "results" / "transfer" / "pendulum.json"
OUT_TEX = REPO / "paper" / "numbers_pendulum.tex"
OUT_FIG = REPO / "paper" / "figures" / "fig_pendulum.pdf"
CACHE = REPO / "results" / "transfer" / "pendulum_cache"  # one JSON per fit (the evaluations), plus config.json

ACTIONS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # fractions of u_max: dimensionless by construction
N_ACT = len(ACTIONS)
OMEGA_MAX = 8.0  # sampled states: omega tau0 uniform in [-8, 8]
REPS = ("nat", "raw", "rawr", "rawi", "rawd")
CONTRASTS = (("raw", "rawr"), ("rawi", "rawr"), ("rawi", "raw"), ("rawd", "raw"))  # zero-shot, a - b
EPS_REPS = ("nat", "raw")
POOL_REPS = ("nat", "raw", "rawr", "rawi")  # the out-of-range systems pooled at half the seen data (as in PC1x)
TAU_TRAIN = (0.1, 1.0)  # the range of tau0 = sqrt(l/g) implied by l in [0.3, 3] and g in [3, 30]

CFG = dict(seeds=[0, 1, 2], n_train=64, n_test=32, n_out=16, group=8, n_sweep=4,
           eps=[(1 / 64, 4), (1 / 16, 4), (1 / 4, 2), (1 / 2, 1)],
           states=576, iters=50, rounds=200, gamma=0.98, u_ratio=0.5, damping=0.1, dt_ratio=0.1, horizon=200,
           episodes=16, init_noise=0.05, reps=list(REPS), eps_reps=list(EPS_REPS), pool_reps=list(POOL_REPS),
           # follow-ups (review round 4, R5); additive, the fits above are unchanged:
           sweep_seeds=[3, 4, 5, 6, 7, 8],  # extra seeds for the eps sweep only (nat and raw zero-shot + sweep fits)
           x4_states=2304, x4_reps=["nat", "rawi"],  # trees with 4x data per system, zero-shot
           nn_reps=list(REPS), nn_seen_reps=["nat", "raw", "rawi"],  # the neural learner (see fqi_nn)
           nn=dict(hidden=256, layers=2, act="relu", lr=1e-3, batch=1024, epochs=10, weight_decay=0.0))

# uids: training 0..63, in-range test 100..131, out-of-range test 200..215 (200..207 slow, 208..215 fast)
TRAIN0, TEST0, OUT0 = 0, 100, 200


# ----------------------------------------------------------------------------------------------- environment
class Family:
    """The dimensionless groups, fixed across systems."""

    def __init__(self, u_ratio=0.5, damping=0.1, dt_ratio=0.1, horizon=200, gamma=0.98):
        self.u_ratio, self.damping, self.dt_ratio, self.horizon, self.gamma = u_ratio, damping, dt_ratio, horizon, gamma

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg["u_ratio"], cfg["damping"], cfg["dt_ratio"], cfg["horizon"], cfg["gamma"])


class Systems:
    """Physical parameters (m, l, g) of a batch of systems (one per row), with the derived u_max, b and dt."""

    def __init__(self, m, l, g, fam: Family):
        self.m, self.l, self.g = (np.asarray(x, float) for x in (m, l, g))
        self.fam = fam
        self.tau0 = np.sqrt(self.l / self.g)
        self.mgl = self.m * self.g * self.l
        self.u_max = fam.u_ratio * self.mgl
        self.b = fam.damping * self.m * self.l ** 2 / self.tau0
        self.dt = fam.dt_ratio * self.tau0

    def __len__(self):
        return len(self.m)


def wrap(x):
    """Wrap an angle to (-pi, pi]."""
    return np.pi - np.mod(np.pi - x, 2 * np.pi)


def step(S: Systems, th, om, u):
    """One semi-implicit Euler step in raw units."""
    ml2 = S.m * S.l ** 2
    om2 = om + S.dt * (S.g / S.l * np.sin(th) - S.b / ml2 * om + u / ml2)
    return wrap(th + S.dt * om2), om2


def cost_tilde(S: Systems, th, om, u):
    """The natural (dimensionless) cost of one step."""
    return (1.0 - np.cos(th)) + 0.1 * (om * S.tau0) ** 2 + 0.01 * (u / S.mgl) ** 2


def features(rep: str, S: Systems, th, om) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    if rep == "nat":
        return np.column_stack([c, s, om * S.tau0])
    if rep in ("raw", "rawr"):
        return np.column_stack([c, s, om])
    if rep == "rawi":
        return np.column_stack([c, s, om, np.broadcast_to(S.tau0, c.shape)])
    if rep == "rawd":
        return np.column_stack([c, s, om] + [np.broadcast_to(x, c.shape) for x in (S.m, S.l, S.g)])
    raise ValueError(rep)


def train_cost(rep: str, S: Systems, ct):
    """The cost a representation is trained on: natural, or raw (energy-like) m g l c~."""
    return S.mgl * ct if rep in ("raw", "rawd") else ct


def draw_systems(seed: int, cfg) -> dict:
    """uid -> (m, l, g, set) for one seed."""
    rng = np.random.default_rng([seed, 1001])

    def logu(lo, hi, n):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    out = {}
    for uid0, n, name in ((TRAIN0, cfg["n_train"], "train"), (TEST0, cfg["n_test"], "test")):
        m, l, g = logu(0.3, 3, n), logu(0.3, 3, n), logu(3, 30, n)
        for i in range(n):
            out[uid0 + i] = (m[i], l[i], g[i], name)
    n = cfg["n_out"]
    m, l, g = logu(0.3, 3, n), logu(0.3, 3, n), logu(3, 30, n)
    half = n // 2
    tau = np.concatenate([logu(3 * TAU_TRAIN[1], 9 * TAU_TRAIN[1], half),
                          logu(TAU_TRAIN[0] / 9, TAU_TRAIN[0] / 3, n - half)])
    s = np.sqrt(l * g)  # l g fixed, so m g l keeps its training distribution
    for i in range(n):
        out[OUT0 + i] = (m[i], s[i] * tau[i], s[i] / tau[i], "slow" if i < half else "fast")
    return out


def systems_of(table: dict, uids, fam: Family) -> Systems:
    p = np.array([table[u][:3] for u in uids], float).reshape(-1, 3)
    return Systems(p[:, 0], p[:, 1], p[:, 2], fam)


def transitions(S1: Systems, n_states: int, seed: int, uid: int) -> dict:
    """Branched transitions of one system: n_states uniform random states, each with all five actions. Rows are
    drawn row-major from one stream, so the first k states do not depend on n_states (nested subsamples).
    Shapes: th, om (n,); th2, om2, ct (N_ACT, n)."""
    rng = np.random.default_rng([seed, uid, 11])
    U = rng.random((n_states, 2))
    th = np.pi * (2 * U[:, 0] - 1)
    om = OMEGA_MAX * (2 * U[:, 1] - 1) / S1.tau0
    u = ACTIONS[:, None] * S1.u_max
    th2, om2 = step(S1, th[None, :], om[None, :], u)
    return {"th": th, "om": om, "th2": th2, "om2": om2, "ct": cost_tilde(S1, th[None, :], om[None, :], u)}


def initial_states(seed: int, uid: int, episodes: int, noise: float):
    """Hanging position plus a small noise, in natural units; the same draws for every policy."""
    rng = np.random.default_rng([seed, uid, 23])
    z = rng.standard_normal((episodes, 2))
    return wrap(np.pi + noise * z[:, 0]), noise * z[:, 1]


# ----------------------------------------------------------------------------------------------- learner
LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  deterministic=True, force_row_wise=True, max_cat_to_onehot=8)


def _with_action(X, a):
    return np.column_stack([X, np.broadcast_to(np.asarray(a, float), (X.shape[0],))])


def _stack_actions(X):
    """(N_ACT * n, F + 1): X with each action index in turn (action-major)."""
    return np.vstack([_with_action(X, b) for b in range(N_ACT)])


class QModel:
    """Q(s, a) = V(s) + A(s, a) (V is None for plain targets); the greedy policy needs A only."""

    def __init__(self, bV, bA):
        self.bV, self.bA = bV, bA

    def adv(self, X, threads=1):
        n = X.shape[0]
        return self.bA.predict(_stack_actions(X), num_threads=threads).reshape(N_ACT, n).T

    def q(self, X, threads=1):
        A = self.adv(X, threads)
        return A if self.bV is None else A + self.bV.predict(X, num_threads=threads)[:, None]


def fqi(X, Xn, c, gamma, iters, rounds, threads, seed, plain=False, log=None) -> tuple[QModel, list]:
    """Discounted fitted Q-iteration on branched data (see the module docstring). X (n, F) states; Xn (N_ACT, n, F)
    next states; c (N_ACT, n) one-step costs. plain=True regresses the targets themselves with one regressor on
    (s, a) (used only by `pilot --plain`). Returns the model and, per iteration, the mean absolute change of the
    targets relative to their mean absolute value."""
    import lightgbm as lgb

    n, F = X.shape
    params = dict(LGB_PARAMS, num_threads=threads, seed=seed)
    dsA = lgb.Dataset(_stack_actions(X), label=c.ravel(), categorical_feature=[F], free_raw_data=False,
                      params=params).construct()
    dsV = None if plain else lgb.Dataset(X, label=c.mean(0), free_raw_data=False, params=params).construct()
    Xn_flat = Xn.reshape(N_ACT * n, F)
    y = c.copy()
    model, change = None, []
    for k in range(iters):
        if plain:
            dsA.set_label(y.ravel())
            model = QModel(None, lgb.train(params, dsA, num_boost_round=rounds))
        else:
            base = y.mean(0)
            dsV.set_label(base)
            dsA.set_label((y - base[None, :]).ravel())
            model = QModel(lgb.train(params, dsV, num_boost_round=rounds),
                           lgb.train(params, dsA, num_boost_round=rounds))
        if k < iters - 1:
            y_new = c + gamma * model.q(Xn_flat, threads).min(axis=1).reshape(N_ACT, n)
            change.append(float(np.abs(y_new - y).mean() / max(np.abs(y_new).mean(), 1e-12)))
            if log:
                log(f"    iter {k + 1}: mean target {y_new.mean():.3f}, relative change {change[-1]:.4f}")
            y = y_new
    return model, change


# ----------------------------------------------------------------------------------------------- neural learner
def _nn_input(X):
    """The input transform of the paper's neural learner: sign(x) log(1 + |x|) (then standardised)."""
    return np.sign(X) * np.log1p(np.abs(X))


class NNModel:
    """Q(s, a) = V(s) + A(s, a): two MLPs on standardised sign-log inputs; V has one output, A one per action.
    Targets were scaled to unit RMS, so outputs are rescaled here. The greedy policy needs A only."""

    def __init__(self, netV, netA, mu, sd, mV, sV, sA, device):
        self.netV, self.netA, self.mu, self.sd = netV, netA, mu, sd
        self.mV, self.sV, self.sA, self.device = mV, sV, sA, device

    def _t(self, X):
        import torch
        return torch.as_tensor((_nn_input(X) - self.mu) / self.sd, dtype=torch.float32, device=self.device)

    def adv(self, X, threads=1):
        import torch
        with torch.no_grad():
            return self.netA(self._t(X)).double().cpu().numpy() * self.sA

    def q(self, X, threads=1):
        import torch
        with torch.no_grad():
            x = self._t(X)
            return (self.netA(x).double().cpu().numpy() * self.sA
                    + (self.netV(x).double().cpu().numpy() * self.sV + self.mV))


def _mlp(n_in, n_out, hidden, layers, act):
    import torch.nn as nn
    A = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[act]
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), A()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


def fqi_nn(X, Xn, c, gamma, iters, hp: dict, seed: int, device="cpu", log=None) -> tuple[NNModel, list]:
    """Discounted fitted Q-iteration with MLPs, on the same branched data and with the same advantage split as `fqi`:
    each iteration's targets y_a(s) are split into V(s) = mean_a y_a(s) and the deviations y_a(s) - V(s); one MLP
    fits V (one output), another the deviations (one output per action), each target scaled to unit RMS. Both are
    warm-started from the previous iteration and trained for hp['epochs'] epochs of Adam on minibatches."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n, F = X.shape
    Z = _nn_input(X)
    mu, sd = Z.mean(0), np.maximum(Z.std(0), 1e-6)
    x = torch.as_tensor((Z - mu) / sd, dtype=torch.float32, device=device)
    xn = torch.as_tensor((_nn_input(Xn.reshape(N_ACT * n, F)) - mu) / sd, dtype=torch.float32, device=device)
    ct = torch.as_tensor(c, dtype=torch.float32, device=device)  # (N_ACT, n)
    netV = _mlp(F, 1, hp["hidden"], hp["layers"], hp["act"]).to(device)
    netA = _mlp(F, N_ACT, hp["hidden"], hp["layers"], hp["act"]).to(device)
    opt = torch.optim.Adam(list(netV.parameters()) + list(netA.parameters()), lr=hp["lr"],
                           weight_decay=hp.get("weight_decay", 0.0))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    y = ct.clone()
    model, change = None, []
    for k in range(iters):
        base = y.mean(0)
        dev = (y - base[None, :]).T  # (n, N_ACT)
        mV, sV = float(base.mean()), float(base.std()) + 1e-8
        sA = float(dev.pow(2).mean().sqrt()) + 1e-8
        tV, tA = ((base - mV) / sV)[:, None], dev / sA
        for _ in range(hp["epochs"]):
            perm = torch.randperm(n, generator=gen).to(device)
            for i in range(0, n, hp["batch"]):
                b = perm[i:i + hp["batch"]]
                loss = ((netV(x[b]) - tV[b]) ** 2).mean() + ((netA(x[b]) - tA[b]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model = NNModel(netV, netA, mu, sd, mV, sV, sA, device)
        if k < iters - 1:
            with torch.no_grad():
                qn = netA(xn) * sA + (netV(xn) * sV + mV)
                y_new = ct + gamma * qn.min(dim=1).values.reshape(N_ACT, n)
            change.append(float((y_new - y).abs().mean() / y_new.abs().mean().clamp_min(1e-12)))
            if log:
                log(f"    iter {k + 1}: loss {float(loss):.4f}, mean target {float(y_new.mean()):.3f}, "
                    f"relative change {change[-1]:.4f}")
            y = y_new
    return model, change


def rollout(model: QModel, rep: str, S: Systems, th0, Om0, horizon: int, threads=1):
    """Greedy rollouts, one per row of S. Returns the mean c~ per step and the share of the last quarter of the
    horizon spent upright (|theta| < 0.3)."""
    th, om = th0.copy(), Om0 / S.tau0
    tot, up = np.zeros(len(th)), np.zeros(len(th))
    last = max(horizon // 4, 1)
    for t in range(horizon):
        u = ACTIONS[model.adv(features(rep, S, th, om), threads).argmin(axis=1)] * S.u_max
        tot += cost_tilde(S, th, om, u)
        th, om = step(S, th, om, u)
        if t >= horizon - last:
            up += np.abs(th) < 0.3
    return tot / horizon, up / last


# ----------------------------------------------------------------------------------------------- design
def groups(cfg, seed=None):
    """In-range test uids, their groups of `group`, out-of-range uids (slow, then fast), their groups (half slow,
    half fast), and, given a seed, the eps-sweep systems: n_sweep in-range test systems spread over their range of
    tau0 (sorted by tau0, taken at evenly spaced ranks, so the smallest and the largest are included), fixed by the
    draws alone, before any fit."""
    I = [TEST0 + i for i in range(cfg["n_test"])]
    G = cfg["group"]
    gin = [I[i:i + G] for i in range(0, len(I), G)]
    half = cfg["n_out"] // 2
    slow = [OUT0 + i for i in range(half)]
    fast = [OUT0 + half + i for i in range(cfg["n_out"] - half)]
    h = G // 2
    gout = [slow[i:i + h] + fast[i:i + h] for i in range(0, half, h)]
    sweep = []
    if seed is not None:
        table = draw_systems(seed, cfg)
        by_tau = sorted(I, key=lambda u: table[u][1] / table[u][2])
        ranks = np.round(np.linspace(0, len(I) - 1, cfg["n_sweep"])).astype(int)
        sweep = [by_tau[r] for r in ranks]
    return I, gin, slow + fast, gout, sweep


def eps_label(e: float) -> str:
    return {1 / 64: "1/64", 1 / 16: "1/16", 1 / 4: "1/4", 1 / 2: "1/2"}.get(e, f"{e:g}")


def plan(cfg) -> list[dict]:
    """Every fit: its training data (uid, states) and the test systems it is scored on."""
    n, nt, G = cfg["states"], cfg["n_train"], cfg["group"]
    T = [TRAIN0 + i for i in range(nt)]
    I, gin, O, gout, _ = groups(cfg)
    k_seen = n * nt // (nt + G)
    if k_seen * (nt + G) != n * nt:
        raise SystemExit(f"states={n} is not a multiple of {(nt + G) // np.gcd(nt, nt + G)}: seen fits would not be "
                         "size-matched")
    fits = []
    for seed in cfg["seeds"]:
        for rep in cfg["reps"]:
            fits.append(dict(key=f"s{seed}_{rep}_zero", seed=seed, rep=rep, kind="zero", group=0, eps=0.0,
                             parts=[(u, n) for u in T], evals=I + O))
            for g, grp in enumerate(gin):
                fits.append(dict(key=f"s{seed}_{rep}_seenin{g}", seed=seed, rep=rep, kind="seen_in", group=g,
                                 eps=1 / (nt + G), parts=[(u, k_seen) for u in T + grp], evals=grp))
            for g, grp in enumerate(gout):
                fits.append(dict(key=f"s{seed}_{rep}_seenout{g}", seed=seed, rep=rep, kind="seen_out", group=g,
                                 eps=1 / (nt + G), parts=[(u, k_seen) for u in T + grp], evals=grp))
        # all out-of-range systems together at half the data (each 1/32), the training systems the other half
        for rep in cfg.get("pool_reps", []):
            fits.append(dict(key=f"s{seed}_{rep}_seenoutpool", seed=seed, rep=rep, kind="seen_outpool", group=0,
                             eps=1 / (2 * len(O)), parts=[(u, n // 2) for u in T] + [(u, n * nt // (2 * len(O))) for u in O],
                             evals=O))
            assert nt * (n // 2) + len(O) * (n * nt // (2 * len(O))) == nt * n
        sweep = groups(cfg, seed)[4]  # sorted by tau0
        for rep in cfg["eps_reps"]:
            for e, Ge in cfg["eps"]:
                k_te, k_tr = e * nt * n, (1 - Ge * e) * n
                if abs(k_te - round(k_te)) > 1e-9 or abs(k_tr - round(k_tr)) > 1e-9:
                    raise SystemExit(f"states={n}: eps={e} does not give whole state counts")
                k_te, k_tr = int(round(k_te)), int(round(k_tr))
                assert Ge * k_te + nt * k_tr == nt * n
                ng = len(sweep) // Ge
                for g in range(ng):
                    grp = sweep[g::ng]  # interleaved in tau0, so a group does not pool similar systems
                    fits.append(dict(key=f"s{seed}_{rep}_eps{round(1 / e)}_{g}", seed=seed, rep=rep, kind="eps",
                                     group=g, eps=e, parts=[(u, k_tr) for u in T] + [(u, k_te) for u in grp],
                                     evals=grp))
    order = {"zero": 0, "seen_in": 1, "seen_out": 2, "eps": 3, "seen_outpool": 4}
    # the optional raw+desc arm's seen fits last, so an interrupted run loses them first
    return sorted(fits, key=lambda f: (f["rep"] == "rawd" and f["kind"] != "zero", order[f["kind"]], f["seed"],
                                       f["rep"], f["eps"], f["group"]))


def plan_extra(cfg, which: str) -> list[dict]:
    """Follow-up fits (review round 4). Keys never collide with the main plan's: the sweep's extra seeds use the main
    naming (seeds 3..8 appear nowhere else), trees with 4x data are tagged x4_, the neural learner nn_."""
    if which == "sweep":
        c2 = dict(cfg, seeds=list(cfg["sweep_seeds"]), reps=list(cfg["eps_reps"]), pool_reps=[])
        return [f for f in plan(c2) if f["kind"] in ("zero", "eps")]
    if which == "x4":
        n, nt = cfg["x4_states"], cfg["n_train"]
        T = [TRAIN0 + i for i in range(nt)]
        I, _, O, _, _ = groups(cfg)
        return [dict(key=f"x4_s{seed}_{rep}_zero", seed=seed, rep=rep, kind="zero", group=0, eps=0.0, tag="x4",
                     parts=[(u, n) for u in T], evals=I + O) for seed in cfg["seeds"] for rep in cfg["x4_reps"]]
    if which == "nn":
        out = []
        for f in plan(cfg):
            keep = (f["kind"] == "zero" and f["rep"] in cfg["nn_reps"]) or \
                   (f["kind"] in ("seen_in", "seen_outpool") and f["rep"] in cfg["nn_seen_reps"])
            if keep:
                out.append(dict(f, key="nn_" + f["key"], tag="nn", learner="nn"))
        return out
    raise ValueError(which)


def assemble(fit, cfg, fam: Family):
    """Training arrays of one fit in its representation: states X, next states Xn, costs c."""
    table = draw_systems(fit["seed"], cfg)
    th, om, th2, om2, ct, rows = [], [], [], [], [], []
    for uid, k in fit["parts"]:
        if k == 0:
            continue
        d = transitions(systems_of(table, [uid], fam), k, fit["seed"], uid)
        th.append(d["th"]), om.append(d["om"]), th2.append(d["th2"]), om2.append(d["om2"]), ct.append(d["ct"])
        rows.append(np.full(k, uid))
    th, om = np.concatenate(th), np.concatenate(om)
    th2, om2, ct = np.concatenate(th2, axis=1), np.concatenate(om2, axis=1), np.concatenate(ct, axis=1)
    S = systems_of(table, np.concatenate(rows), fam)
    rep = fit["rep"]
    X = features(rep, S, th, om)
    Xn = np.stack([features(rep, S, th2[b], om2[b]) for b in range(N_ACT)])
    return X, Xn, train_cost(rep, S, ct)


def evaluate(model: QModel, fit, cfg, fam: Family, threads: int) -> dict:
    table = draw_systems(fit["seed"], cfg)
    E = cfg["episodes"]
    uids = fit["evals"]
    S = systems_of(table, np.repeat(uids, E), fam)
    init = [initial_states(fit["seed"], u, E, cfg["init_noise"]) for u in uids]
    th0 = np.concatenate([a for a, _ in init])
    Om0 = np.concatenate([b for _, b in init])
    cost, up = rollout(model, fit["rep"], S, th0, Om0, fam.horizon, threads)
    return {str(u): {"cost": cost[i * E:(i + 1) * E].round(6).tolist(), "upright": up[i * E:(i + 1) * E].tolist()}
            for i, u in enumerate(uids)}


def run_fit(job) -> str:
    fit, cfg, cache, threads = job[:4]
    path = Path(cache) / f"{fit['key']}.json"
    if path.exists():
        return f"{fit['key']}: cached"
    t0 = time.time()
    fam = Family.from_cfg(cfg)
    X, Xn, c = assemble(fit, cfg, fam)
    if fit.get("learner") == "nn":
        model, change = fqi_nn(X, Xn, c, fam.gamma, cfg["iters"], cfg["nn"], seed=fit["seed"], device=job[4])
    else:
        model, change = fqi(X, Xn, c, fam.gamma, cfg["iters"], cfg["rounds"], threads, seed=fit["seed"])
    t1 = time.time()
    ev = evaluate(model, fit, cfg, fam, threads)
    rec = dict(fit, parts=None, n_states=int(X.shape[0]), test_states=int(sum(k for u, k in fit["parts"] if u >= TEST0)),
               fit_seconds=round(t1 - t0, 1), eval_seconds=round(time.time() - t1, 1), last_change=change[-1] if change else None,
               eval=ev)
    rec.pop("parts")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(path)
    m = np.mean([np.mean(v["cost"]) for v in ev.values()])
    upr = np.mean([np.mean(v["upright"]) for v in ev.values()])
    return f"{fit['key']}: cost {m:.4f}, upright {upr:.3f}, fit {t1 - t0:.0f}s"


def cmd_run(args, cfg):
    import multiprocessing as mp

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "config.json").write_text(json.dumps(cfg, indent=1))
    todo = plan(cfg) if args.plan == "main" else []
    for which in ("x4", "sweep", "nn"):  # the follow-ups, x4 first (the longest fits)
        if args.plan in (which, "extra" if which != "nn" else "-"):
            todo += plan_extra(cfg, which)
    fits = [f for f in todo if not (cache / f"{f['key']}.json").exists()]
    if args.only:
        fits = [f for f in fits if any(s in f["key"] for s in args.only)]
    print(f"{len(fits)} fits to run, {args.workers} workers x {args.threads} threads", flush=True)
    t0 = time.time()
    jobs = [(f, cfg, str(cache), int(args.threads), args.device) for f in fits]
    if args.workers == 1:
        for i, j in enumerate(jobs):
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {run_fit(j)}", flush=True)
        return
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_fit, jobs)):
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- checks
def similarity_check(cfg) -> float:
    """Largest difference, over a 200-step random dimensionless action sequence, between the natural-unit states of
    four systems whose tau0 spans 0.1 to 10."""
    fam = Family.from_cfg(cfg)
    S = Systems([0.3, 1.0, 3.0, 1.0], [0.3, 1.0, 3.0, 50.0], [30.0, 9.81, 3.0, 0.5], fam)
    th, om = np.full(4, np.pi - 0.1), np.zeros(4)
    rng = np.random.default_rng(0)
    for _ in range(fam.horizon):
        th, om = step(S, th, om, ACTIONS[rng.integers(0, N_ACT)] * S.u_max)
    return float(max(np.ptp(th), np.ptp(om * S.tau0)))


def cmd_check(args, cfg):
    """(1) Exact similarity: the same dimensionless action sequence gives the same natural-unit trajectory on
    systems with very different (m, l, g). (2) Grid value iteration on the dimensionless problem (the exact
    Bellman operator on a 361 x 321 grid with bilinear interpolation), truncated at `iters` iterations as FQI is:
    the greedy policy's cost from the hanging position, for dt/tau0 = 0.05 and 0.1."""
    fam = Family.from_cfg(cfg)
    S = Systems([0.3, 1.0, 3.0, 1.0], [0.3, 1.0, 3.0, 50.0], [30.0, 9.81, 3.0, 0.5], fam)
    th, om = np.full(4, np.pi - 0.1), np.zeros(4)
    rng = np.random.default_rng(0)
    for _ in range(fam.horizon):
        th, om = step(S, th, om, ACTIONS[rng.integers(0, N_ACT)] * S.u_max)
    print("tau0:", np.round(S.tau0, 4))
    print("theta after", fam.horizon, "random steps:", th, " omega tau0:", om * S.tau0)
    print("max spread:", np.ptp(th), np.ptp(om * S.tau0))
    for dt in (0.05, 0.1):
        for iters in (cfg["iters"], 200):
            c, first, up = grid_vi(dt, cfg["u_ratio"], cfg["damping"], cfg["gamma"], iters, cfg["horizon"])
            print(f"grid VI dt/tau0={dt} iterations={iters}: cost/step {c:.3f}, first upright step (median) {first}, "
                  f"upright share of last quarter {up:.3f}")


def grid_vi(dt, u_ratio, damping, gamma, iters, horizon, nth=361, nom=321):
    acts = ACTIONS * u_ratio

    def stp(th, om, u):
        om2 = om + dt * (np.sin(th) - damping * om + u)
        return wrap(th + dt * om2), om2

    def cst(th, om, u):
        return (1 - np.cos(th)) + 0.1 * om ** 2 + 0.01 * u ** 2

    ths = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    oms = np.linspace(-OMEGA_MAX, OMEGA_MAX, nom)
    TH, OM = np.meshgrid(ths, oms, indexing="ij")

    def interp(V, th, om):
        i = (th + np.pi) / (2 * np.pi) * nth
        j = (np.clip(om, -OMEGA_MAX, OMEGA_MAX) + OMEGA_MAX) / (2 * OMEGA_MAX) * (nom - 1)
        i0 = np.floor(i).astype(int)
        j0 = np.minimum(np.floor(j).astype(int), nom - 2)
        fi, fj = i - i0, j - j0
        i0, i1 = i0 % nth, (i0 + 1) % nth
        return ((1 - fi) * (1 - fj) * V[i0, j0] + fi * (1 - fj) * V[i1, j0] + (1 - fi) * fj * V[i0, j0 + 1]
                + fi * fj * V[i1, j0 + 1])

    nxt = [stp(TH, OM, u) for u in acts]
    cs = [cst(TH, OM, u) for u in acts]
    V = np.zeros_like(TH)
    for _ in range(iters):
        V = np.min([cs[a] + gamma * interp(V, *nxt[a]) for a in range(N_ACT)], axis=0)
    rng = np.random.default_rng(0)
    E = 16
    th, om = wrap(np.pi + 0.05 * rng.standard_normal(E)), 0.05 * rng.standard_normal(E)
    tot, first, up = np.zeros(E), np.full(E, horizon), np.zeros(E)
    for t in range(horizon):
        q = np.array([cst(th, om, u) + gamma * interp(V, *stp(th, om, u)) for u in acts])
        u = acts[q.argmin(0)]
        tot += cst(th, om, u)
        th, om = stp(th, om, u)
        first = np.where((first == horizon) & (np.abs(th) < 0.3), t + 1, first)
        if t >= horizon - horizon // 4:
            up += np.abs(th) < 0.3
    return tot.mean() / horizon, float(np.median(first)), up.mean() / (horizon // 4)


def cmd_pilot(args, cfg):
    """Natural units on pooled data of one system (equivalent to any mix of systems in natural units)."""
    fam = Family.from_cfg(cfg)
    S1 = Systems([1.0], [1.0], [9.81], fam)
    n = cfg["states"] * cfg["n_train"]
    for seed in args.seeds:
        t0 = time.time()
        d = transitions(S1, n, seed, 0)
        X = features("nat", S1, d["th"], d["om"])
        Xn = np.stack([features("nat", S1, d["th2"][b], d["om2"][b]) for b in range(N_ACT)])
        if args.learner == "nn":
            model, change = fqi_nn(X, Xn, d["ct"], fam.gamma, cfg["iters"], cfg["nn"], seed, device=args.device,
                                   log=print if args.verbose else None)
        else:
            model, change = fqi(X, Xn, d["ct"], fam.gamma, cfg["iters"], cfg["rounds"], int(args.threads), seed,
                                plain=args.plain, log=print if args.verbose else None)
        t1 = time.time()
        E = 32
        th0, Om0 = initial_states(seed, 0, E, cfg["init_noise"])
        Sx = Systems(np.ones(E), np.ones(E), np.full(E, 9.81), fam)
        c, up = rollout(model, "nat", Sx, th0, Om0, fam.horizon, int(args.threads))
        if args.learner == "nn":
            rec = dict(learner="nn", seed=seed, states=n, iters=cfg["iters"], nn=cfg["nn"], dt_ratio=cfg["dt_ratio"],
                       u_ratio=cfg["u_ratio"], cost=float(c.mean()), cost_max=float(c.max()), upright=float(up.mean()),
                       upright_min=float(up.min()), fit_seconds=round(t1 - t0, 1), episodes=E)
            with open(REPO / "results" / "transfer" / "pendulum_nn_pilots.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"[{args.learner if args.learner == 'nn' else ('plain' if args.plain else 'advantage')} states {n} "
              f"rounds {cfg['rounds']} iters {cfg['iters']} dt {cfg['dt_ratio']} u {cfg['u_ratio']}"
              f"{' nn ' + json.dumps(cfg['nn']) if args.learner == 'nn' else ''}] "
              f"seed {seed}: cost/step {c.mean():.4f} (min {c.min():.3f}, max {c.max():.3f}), upright share of the "
              f"last quarter {up.mean():.3f} (min {up.min():.2f}), last change {change[-1]:.4f}, fit {t1 - t0:.0f}s",
              flush=True)


# ----------------------------------------------------------------------------------------------- report
def load_fits(cache: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(cache.glob("*_*.json"))]


def boot(vals: dict, B=10_000, seed=0) -> dict:
    """Mean over seeds of the mean over test systems; 95% percentile bootstrap over test systems, stratified by
    seed. vals: (seed, uid) -> value."""
    seeds = sorted({s for s, _ in vals})
    arrs = [np.array([vals[k] for k in sorted(vals) if k[0] == s]) for s in seeds]
    rng = np.random.default_rng(seed)
    bs = np.zeros(B)
    for a in arrs:
        bs += a[rng.integers(0, len(a), (B, len(a)))].mean(axis=1)
    bs /= len(arrs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(np.mean([a.mean() for a in arrs])), "lo": float(lo), "hi": float(hi),
            "n": int(sum(len(a) for a in arrs)), "per_seed": [float(a.mean()) for a in arrs]}


def tables(fits: list[dict]):
    """Per (seed, uid): mean cost and upright share of each fit kind."""
    Z, U, S_in, S_out, S_eps, Uin, Uout = {}, {}, {}, {}, {}, {}, {}
    for f in fits:
        for u, v in f["eval"].items():
            k = (f["seed"], int(u))
            c, up = float(np.mean(v["cost"])), float(np.mean(v["upright"]))
            if f["kind"] == "zero":
                Z.setdefault(f["rep"], {})[k] = c
                U.setdefault(f["rep"], {})[k] = up
            elif f["kind"] == "seen_in":
                S_in.setdefault(f["rep"], {})[k] = c
                Uin.setdefault(f["rep"], {})[k] = up
            elif f["kind"] == "seen_out":
                S_out.setdefault(f["rep"], {})[k] = c
                Uout.setdefault(f["rep"], {})[k] = up
            elif f["kind"] == "seen_outpool":
                S_eps.setdefault("_pool", {}).setdefault(f["rep"], {})[k] = c
            else:
                S_eps.setdefault(f["rep"], {}).setdefault(eps_label(f["eps"]), {})[k] = c
    return Z, U, S_in, S_out, S_eps, Uin, Uout


def summarise(fits: list[dict], cfg) -> dict:
    Z, U, S_in, S_out, S_eps, Uin, Uout = tables(fits)
    reps = [r for r in REPS if r in Z]
    O = groups(cfg)[2]
    slow = set(O[:cfg["n_out"] // 2])

    def sub(d, keys):
        return {k: d[k] for k in keys if k in d}

    res = {"in_range": {}, "out_range": {}, "eps_sweep": {}}
    for name, S_seen, Us in (("in_range", S_in, Uin), ("out_range", S_out, Uout)):
        for rep in reps:
            keys = sorted(S_seen.get(rep, {}))
            if not keys:
                continue
            z = sub(Z[rep], keys)
            r = {"zero": boot(z), "seen": boot(sub(S_seen[rep], keys)),
                 "tg": boot({k: Z[rep][k] - S_seen[rep][k] for k in keys}),
                 "upright_zero": boot(sub(U[rep], keys)), "upright_seen": boot(sub(Us[rep], keys))}
            if "nat" in Z:
                r["vs_nat"] = boot({k: Z[rep][k] - Z["nat"][k] for k in keys})
            if name == "out_range":
                for part, ks in (("slow", [k for k in keys if k[1] in slow]), ("fast", [k for k in keys if k[1] not in slow])):
                    r[part] = {"zero": boot(sub(Z[rep], ks)), "tg": boot({k: Z[rep][k] - S_seen[rep][k] for k in ks}),
                               "upright_zero": boot(sub(U[rep], ks))}
                    if "nat" in Z:
                        r[part]["vs_nat"] = boot({k: Z[rep][k] - Z["nat"][k] for k in ks})
            res[name][rep] = r
        if name == "in_range" and "nat" in Z:
            # zero-shot cost relative to natural units by tercile of tau0 among each seed's in-range test systems
            terc = {}
            for s in sorted({k[0] for k in Z["nat"]}):
                table = draw_systems(s, cfg)
                us = sorted((k[1] for k in Z["nat"] if k[0] == s and k[1] < OUT0),
                            key=lambda u: table[u][1] / table[u][2])
                for i, part in enumerate(np.array_split(np.array(us), 3)):
                    for u in part:
                        terc[(s, int(u))] = ("tau_low", "tau_mid", "tau_high")[i]
            for rep in reps:
                if rep == "nat" or rep not in res[name]:
                    continue
                res[name][rep]["by_tau"] = {t: boot({k: Z[rep][k] - Z["nat"][k] for k, v in terc.items() if v == t})
                                            for t in ("tau_low", "tau_mid", "tau_high")}
        # each seen fit's transfer gap, averaged over its test systems: the spread across fits shows the noise of
        # training, which the bootstrap over test systems holds fixed
        kind = "seen_in" if name == "in_range" else "seen_out"
        for rep in reps:
            if rep not in res[name]:
                continue
            per_fit = [float(np.mean([Z[rep][(f["seed"], int(u))] - np.mean(v["cost"]) for u, v in f["eval"].items()]))
                       for f in fits if f["kind"] == kind and f["rep"] == rep]
            res[name][rep]["tg_by_fit"] = {"values": per_fit, "min": float(np.min(per_fit)), "max": float(np.max(per_fit)),
                                           "sd": float(np.std(per_fit, ddof=1)) if len(per_fit) > 1 else None}
        # zero-shot contrasts between representations (paired by test system)
        keys = sorted(k for k in S_seen.get("nat", {}))
        res[name]["contrasts"] = {f"{a}_vs_{b}": boot({k: Z[a][k] - Z[b][k] for k in keys})
                                  for a, b in CONTRASTS if a in Z and b in Z and keys}
    pool = S_eps.pop("_pool", {})
    res["out_range_pooled"] = {}
    for rep, d in pool.items():
        keys = sorted(d)
        r = {"seen": boot(d), "tg": boot({k: Z[rep][k] - d[k] for k in keys}), "zero": boot(sub(Z[rep], keys))}
        for part, ks in (("slow", [k for k in keys if k[1] in slow]), ("fast", [k for k in keys if k[1] not in slow])):
            r[part] = {"tg": boot({k: Z[rep][k] - d[k] for k in ks}), "seen": boot(sub(d, ks))}
        res["out_range_pooled"][rep] = r
    for rep, by_eps in S_eps.items():
        out = {}
        for lab, d in by_eps.items():
            keys = sorted(d)
            num, den = lab.split("/") if "/" in lab else (lab, "1")
            out[lab] = {"eps": float(num) / float(den), "tg": boot({k: Z[rep][k] - d[k] for k in keys}), "seen": boot(d),
                        "zero": boot(sub(Z[rep], keys))}
        keys = sorted(next(iter(by_eps.values())))
        if "nat" in Z:
            out["excess_vs_nat"] = boot({k: Z[rep][k] - Z["nat"][k] for k in keys})
        # least-squares slope of TG on eps through the origin, per (seed, system), then averaged
        e = np.array([out[lab]["eps"] for lab in by_eps])
        slope = {k: float(np.dot(e, [Z[rep][k] - by_eps[lab][k] for lab in by_eps]) / np.dot(e, e)) for k in keys}
        out["slope"] = boot(slope)
        res["eps_sweep"][rep] = out
    return res


# ----------------------------------------------------------------------------------------------- follow-ups
def boot2(vals: dict, B=10_000, seed=0) -> dict:
    """Two-level bootstrap: resample seeds (fits) with replacement, then test systems within each drawn seed; the
    statistic is the mean over seeds of the mean over systems. vals: (seed, uid) -> value. Unlike `boot`, the
    interval includes the noise of training, since every seed has its own fits."""
    seeds = sorted({s for s, _ in vals})
    arrs = [np.array([vals[k] for k in sorted(vals) if k[0] == s]) for s in seeds]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(arrs), (B, len(arrs)))
    bs = np.zeros(B)
    for j in range(len(arrs)):
        for i, a in enumerate(arrs):
            rows = np.nonzero(pick[:, j] == i)[0]
            if len(rows):
                bs[rows] += a[rng.integers(0, len(a), (len(rows), len(a)))].mean(axis=1)
    bs /= len(arrs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(np.mean([a.mean() for a in arrs])), "lo": float(lo), "hi": float(hi),
            "n": int(sum(len(a) for a in arrs)), "seeds": len(arrs), "per_seed": [float(a.mean()) for a in arrs]}


def tau_terciles(cfg, seeds) -> dict:
    """(seed, uid) -> tau0 tercile among that seed's in-range test systems."""
    terc = {}
    for s in seeds:
        table = draw_systems(s, cfg)
        us = sorted((u for u, v in table.items() if v[3] == "test"), key=lambda u: table[u][1] / table[u][2])
        for i, part in enumerate(np.array_split(np.array(us), 3)):
            for u in part:
                terc[(s, int(u))] = ("tau_low", "tau_mid", "tau_high")[i]
    return terc


def zero_table(fits: list[dict], cfg) -> dict:
    """Zero-shot costs of one family of fits, in and out of range, and relative to its own natural-units fits."""
    Z, U = {}, {}
    for f in fits:
        if f["kind"] != "zero":
            continue
        for u, v in f["eval"].items():
            Z.setdefault(f["rep"], {})[(f["seed"], int(u))] = float(np.mean(v["cost"]))
            U.setdefault(f["rep"], {})[(f["seed"], int(u))] = float(np.mean(v["upright"]))
    seeds = sorted({k[0] for d in Z.values() for k in d})
    terc = tau_terciles(cfg, seeds)
    out = {}
    for rep, d in Z.items():
        r = {}
        for part, sel in (("in", lambda u: TEST0 <= u < OUT0), ("out", lambda u: u >= OUT0)):
            ks = sorted(k for k in d if sel(k[1]))
            r[part] = {"zero": boot({k: d[k] for k in ks}), "upright": boot({k: U[rep][k] for k in ks})}
            if rep != "nat" and "nat" in Z:
                r[part]["vs_nat"] = boot({k: d[k] - Z["nat"][k] for k in ks})
        if rep != "nat" and "nat" in Z:
            r["in"]["by_tau"] = {t: boot({k: d[k] - Z["nat"][k] for k, v in terc.items() if v == t and k in d})
                                 for t in ("tau_low", "tau_mid", "tau_high")}
        out[rep] = r
    return out, Z


def summarise_followups(main_fits: list[dict], extra: list[dict], cfg) -> dict:
    res = {}
    # the neural learner: zero-shot table, in-range seen (1/72) and pooled out-of-range comparators
    nn = [f for f in extra if f.get("tag") == "nn"]
    if nn:
        zt, _ = zero_table(nn, cfg)
        s = summarise([dict(f, tag=None) for f in nn], cfg)
        res["neural"] = {"zero": zt, "in_range": s["in_range"], "out_range_pooled": s["out_range_pooled"],
                         "fits": len(nn), "fit_minutes_median": float(np.median([f["fit_seconds"] for f in nn]) / 60)}
    # trees with 4x data per system: zero-shot nat and rawi, and the change of rawi's gap from 1x to 4x
    x4 = [f for f in extra if f.get("tag") == "x4"]
    if x4:
        zt4, Z4 = zero_table(x4, cfg)
        _, Z1 = zero_table([f for f in main_fits if f["rep"] in ("nat", "rawi")], cfg)
        r = {"zero": zt4}
        if "rawi" in Z4 and "nat" in Z4:
            for part, sel in (("in", lambda u: TEST0 <= u < OUT0), ("out", lambda u: u >= OUT0)):
                ks = sorted(k for k in Z4["rawi"] if sel(k[1]) and k in Z1.get("rawi", {}))
                r[f"gap_change_{part}"] = boot({k: (Z4["rawi"][k] - Z4["nat"][k]) - (Z1["rawi"][k] - Z1["nat"][k])
                                                for k in ks})
                r[f"nat_change_{part}"] = boot({k: Z4["nat"][k] - Z1["nat"][k] for k in ks})
        res["x4"] = dict(r, fits=len(x4), fit_minutes_median=float(np.median([f["fit_seconds"] for f in x4]) / 60))
    # the eps sweep over all seeds (main seeds plus sweep_seeds), with intervals that resample fits
    tree = [f for f in main_fits + extra if not f.get("tag") and f["rep"] in cfg["eps_reps"] and f["kind"] in ("zero", "eps")]
    Z, S = {}, {}
    for f in tree:
        for u, v in f["eval"].items():
            k = (f["seed"], int(u))
            if f["kind"] == "zero":
                Z.setdefault(f["rep"], {})[k] = float(np.mean(v["cost"]))
            else:
                S.setdefault(f["rep"], {}).setdefault(f["eps"], {})[k] = float(np.mean(v["cost"]))
    if "raw" in S and "nat" in S:
        eps = sorted(S["raw"])
        keys = sorted(set.intersection(*[set(S[r][e]) for r in ("raw", "nat") for e in eps]))
        tg = {r: {e: {k: Z[r][k] - S[r][e][k] for k in keys} for e in eps} for r in ("raw", "nat")}
        d = {e: {k: tg["raw"][e][k] - tg["nat"][e][k] for k in keys} for e in eps}
        ev = np.array(eps)
        X = np.column_stack([np.ones_like(ev), ev])
        coef = {k: np.linalg.lstsq(X, np.array([d[e][k] for e in eps]), rcond=None)[0][1] for k in keys}
        res["sweep"] = {"seeds": sorted({k[0] for k in keys}), "systems": len(keys),
                        "eps": {eps_label(e): {"eps": e, "tg_raw": boot2(tg["raw"][e]), "tg_nat": boot2(tg["nat"][e]),
                                               "diff": boot2(d[e]), "tg_raw_onelevel": boot(tg["raw"][e]),
                                               "tg_nat_onelevel": boot(tg["nat"][e])} for e in eps},
                        "slope_diff": boot2(coef),
                        "half_minus_64th_diff": boot2({k: d[max(eps)][k] - d[min(eps)][k] for k in keys}),
                        "excess_raw": boot2({k: Z["raw"][k] - Z["nat"][k] for k in keys})}
    return res


def fmt(x, signed):
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def macros(res: dict, cfg, fits) -> dict:
    name = {"nat": "Nat", "raw": "Raw", "rawr": "Rawr", "rawi": "Rawi", "rawd": "Rawd"}
    m = {}

    def put(key, st, signed):
        m[key] = fmt(st["mean"], signed)
        m[key + "Lo"] = fmt(st["lo"], signed)
        m[key + "Hi"] = fmt(st["hi"], signed)

    for part, P in (("in_range", "InRange"), ("out_range", "OutRange")):
        for key, st in res[part].get("contrasts", {}).items():
            a, b = key.split("_vs_")
            put(f"Pend{P}{name[a]}Vs{name[b]}", st, True)
        for rep, r in res[part].items():
            if rep == "contrasts":
                continue
            R = name[rep]
            put(f"Pend{P}Cost{R}", r["zero"], False)
            put(f"Pend{P}Seen{R}", r["seen"], False)
            put(f"Pend{P}Tg{R}", r["tg"], True)
            m[f"Pend{P}Up{R}"] = f"{r['upright_zero']['mean']:.3f}"
            if "tg_by_fit" in r:
                m[f"Pend{P}Tg{R}FitMin"] = fmt(r["tg_by_fit"]["min"], True)
                m[f"Pend{P}Tg{R}FitMax"] = fmt(r["tg_by_fit"]["max"], True)
            if rep != "nat" and "vs_nat" in r:
                put(f"Pend{P}{R}VsNat", r["vs_nat"], True)
            for t, T in (("tau_low", "TauLow"), ("tau_mid", "TauMid"), ("tau_high", "TauHigh")):
                if t in r.get("by_tau", {}):
                    put(f"Pend{P}{T}{R}VsNat", r["by_tau"][t], True)
            for sp, SP in (("slow", "Slow"), ("fast", "Fast")):
                if sp in r:
                    put(f"Pend{P}{SP}Cost{R}", r[sp]["zero"], False)
                    put(f"Pend{P}{SP}Tg{R}", r[sp]["tg"], True)
                    if rep != "nat" and "vs_nat" in r[sp]:
                        put(f"Pend{P}{SP}{R}VsNat", r[sp]["vs_nat"], True)
    for rep, r in res.get("out_range_pooled", {}).items():
        R = name[rep]
        put(f"PendOutPoolTg{R}", r["tg"], True)
        put(f"PendOutPoolSeen{R}", r["seen"], False)
        for sp, SP in (("slow", "Slow"), ("fast", "Fast")):
            put(f"PendOutPool{SP}Tg{R}", r[sp]["tg"], True)
    lab = {"1/64": "SixtyFourth", "1/16": "Sixteenth", "1/4": "Quarter", "1/2": "Half"}
    for rep, r in res["eps_sweep"].items():
        R = name[rep]
        for k, L in lab.items():
            if k in r:
                put(f"PendEps{L}Tg{R}", r[k]["tg"], True)
        if "excess_vs_nat" in r and rep != "nat":
            put(f"PendEpsExcess{R}", r["excess_vs_nat"], True)
            if "1/2" in r:
                m[f"PendEpsHalfShareOfExcess{R}"] = f"{r['1/2']['tg']['mean'] / r['excess_vs_nat']['mean']:.3f}"
        put(f"PendEpsSlope{R}", r["slope"], True)
    if "test_outside_sample" in res:
        m["PendInRangeOutsideSample"] = str(res["test_outside_sample"]["count"])
        m["PendInRangeSystemsTotal"] = str(res["test_outside_sample"]["total"])
    for key, G in (("0.05", "Fine"), (f"{cfg['dt_ratio']:g}", "Coarse")):
        if key in res.get("grid_vi", {}):
            m[f"PendGrid{G}Cost"] = f"{res['grid_vi'][key]['cost']:.3f}"
            m[f"PendGrid{G}Up"] = f"{res['grid_vi'][key]['upright_last_quarter']:.3f}"
    # settings
    n = cfg["states"]
    m.update({"PendTrainSystems": str(cfg["n_train"]), "PendTestSystems": str(cfg["n_test"]),
              "PendOutSystems": str(cfg["n_out"]), "PendGroup": str(cfg["group"]), "PendSweepSystems": str(cfg["n_sweep"]),
              "PendSeeds": str(len(cfg["seeds"])), "PendStates": f"{n:,}".replace(",", "{,}"),
              "PendStatesTotal": f"{n * cfg['n_train']:,}".replace(",", "{,}"),
              "PendTransitionsTotal": f"{n * cfg['n_train'] * N_ACT:,}".replace(",", "{,}"),
              "PendIters": str(cfg["iters"]), "PendRounds": str(cfg["rounds"]), "PendGamma": f"{cfg['gamma']}",
              "PendHorizon": str(cfg["horizon"]), "PendDtRatio": f"{cfg['dt_ratio']}", "PendURatio": f"{cfg['u_ratio']}",
              "PendDamping": f"{cfg['damping']}", "PendEpisodes": str(cfg["episodes"]),
              "PendFits": str(len(fits)),
              "PendFitMinutes": f"{np.median([f['fit_seconds'] for f in fits]) / 60:.1f}"})
    followup_macros(res.get("followups", {}), cfg, m, put, name)
    return m


def followup_macros(fu: dict, cfg, m: dict, put, name: dict) -> None:
    """Macros of the review-round-4 follow-ups: PendNn* (neural learner), PendXfour* (trees, 4x data), PendSweep*
    (the eps sweep over all seeds, two-level intervals)."""
    taus = (("tau_low", "TauLow"), ("tau_mid", "TauMid"), ("tau_high", "TauHigh"))
    lab = {"1/64": "SixtyFourth", "1/16": "Sixteenth", "1/4": "Quarter", "1/2": "Half"}
    if "neural" in fu:
        nnr = fu["neural"]
        for rep, r in nnr["zero"].items():
            R = name[rep]
            put(f"PendNnInRangeCost{R}", r["in"]["zero"], False)
            put(f"PendNnOutRangeCost{R}", r["out"]["zero"], False)
            m[f"PendNnInRangeUp{R}"] = f"{r['in']['upright']['mean']:.3f}"
            m[f"PendNnOutRangeUp{R}"] = f"{r['out']['upright']['mean']:.3f}"
            if "vs_nat" in r["in"]:
                put(f"PendNnInRange{R}VsNat", r["in"]["vs_nat"], True)
                put(f"PendNnOutRange{R}VsNat", r["out"]["vs_nat"], True)
                for t, T in taus:
                    put(f"PendNnInRange{T}{R}VsNat", r["in"]["by_tau"][t], True)
        for rep, r in nnr["in_range"].items():
            if rep == "contrasts":
                for key, st in r.items():
                    a, b = key.split("_vs_")
                    put(f"PendNnInRange{name[a]}Vs{name[b]}", st, True)
                continue
            put(f"PendNnInRangeTg{name[rep]}", r["tg"], True)
            m[f"PendNnInRangeTg{name[rep]}FitMin"] = fmt(r["tg_by_fit"]["min"], True)
            m[f"PendNnInRangeTg{name[rep]}FitMax"] = fmt(r["tg_by_fit"]["max"], True)
        for rep, r in nnr["out_range_pooled"].items():
            put(f"PendNnOutPoolTg{name[rep]}", r["tg"], True)
        hp = cfg["nn"]
        m.update({"PendNnFits": str(nnr["fits"]), "PendNnFitMinutes": f"{nnr['fit_minutes_median']:.1f}",
                  "PendNnHidden": str(hp["hidden"]), "PendNnLayers": str(hp["layers"]), "PendNnEpochs": str(hp["epochs"]),
                  "PendNnBatch": f"{hp['batch']:,}".replace(",", "{,}"), "PendNnLr": f"{hp['lr']:g}",
                  "PendNnAct": {"relu": "ReLU", "tanh": "tanh", "silu": "SiLU"}[hp["act"]]})
    if "x4" in fu:
        x = fu["x4"]
        for rep, r in x["zero"].items():
            R = name[rep]
            put(f"PendXfourInRangeCost{R}", r["in"]["zero"], False)
            put(f"PendXfourOutRangeCost{R}", r["out"]["zero"], False)
            if "vs_nat" in r["in"]:
                put(f"PendXfourInRange{R}VsNat", r["in"]["vs_nat"], True)
                put(f"PendXfourOutRange{R}VsNat", r["out"]["vs_nat"], True)
                for t, T in taus:
                    put(f"PendXfourInRange{T}{R}VsNat", r["in"]["by_tau"][t], True)
        for part, P in (("in", "InRange"), ("out", "OutRange")):
            if f"gap_change_{part}" in x:
                put(f"PendXfour{P}RawiGapChange", x[f"gap_change_{part}"], True)
                put(f"PendXfour{P}NatChange", x[f"nat_change_{part}"], True)
        m["PendXfourStates"] = f"{cfg['x4_states']:,}".replace(",", "{,}")
        m["PendXfourFitMinutes"] = f"{x['fit_minutes_median']:.1f}"
    if "sweep" in fu:
        sw = fu["sweep"]
        m["PendSweepSeeds"] = str(len(sw["seeds"]))
        m["PendSweepPairs"] = str(sw["systems"])  # (seed, system) pairs; PendSweepSystems is per seed
        excl2 = excl1 = 0
        for k, L in lab.items():
            if k not in sw["eps"]:
                continue
            r = sw["eps"][k]
            put(f"PendSweepTgRaw{L}", r["tg_raw"], True)
            put(f"PendSweepTgNat{L}", r["tg_nat"], True)
            put(f"PendSweepDiff{L}", r["diff"], True)
            excl2 += r["tg_nat"]["lo"] > 0 or r["tg_nat"]["hi"] < 0
            excl1 += r["tg_nat_onelevel"]["lo"] > 0 or r["tg_nat_onelevel"]["hi"] < 0
        m["PendSweepNatExclZero"] = str(excl2)
        m["PendSweepNatExclZeroOneLevel"] = str(excl1)
        put("PendSweepSlopeDiff", sw["slope_diff"], True)
        put("PendSweepHalfMinusSixtyFourthDiff", sw["half_minus_64th_diff"], True)
        put("PendSweepExcessRaw", sw["excess_raw"], True)


def cmd_report(args, cfg):
    global OUT_JSON, OUT_TEX, OUT_FIG
    stored_json = OUT_JSON  # read by --from-json, before any --outdir redirect
    if args.outdir:  # e.g. a smoke test: keep the deliverables untouched
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        OUT_JSON, OUT_TEX, OUT_FIG = d / OUT_JSON.name, d / OUT_TEX.name, d / OUT_FIG.name
    cache = Path(args.cache)
    if args.from_json:
        stored = json.loads(stored_json.read_text())
        fits, cfg = stored["fits"] + stored.get("fits_extra", []), {**CFG, **stored["settings"]}
    else:
        fits = load_fits(cache)
        cfg = {**CFG, **json.loads((cache / "config.json").read_text())} if (cache / "config.json").exists() else cfg
    # the original design: tree fits of the main seeds; the follow-ups (extra sweep seeds, x4_, nn_) are analysed apart
    extra = [f for f in fits if f.get("tag") or f["seed"] not in cfg["seeds"]]
    fits = [f for f in fits if not f.get("tag") and f["seed"] in cfg["seeds"]]
    planned = {f["key"] for f in plan(cfg)}
    have = {f["key"] for f in fits}
    missing = sorted(planned - have)
    if missing:
        print(f"warning: {len(missing)} planned fits missing, e.g. {missing[:5]}")
    res = summarise(fits, cfg)
    res["followups"] = summarise_followups(fits, extra, cfg)
    pilots = REPO / "results" / "transfer" / "pendulum_nn_pilots.jsonl"
    if pilots.exists():
        res["followups"]["nn_pilots"] = [json.loads(x) for x in pilots.read_text().splitlines() if x.strip()]
    # exact (grid) value iteration on the dimensionless problem, truncated as FQI is: why dt/tau0 = 0.1
    res["grid_vi"] = {}
    for dt in (0.05, cfg["dt_ratio"]):
        c, first, up = grid_vi(dt, cfg["u_ratio"], cfg["damping"], cfg["gamma"], cfg["iters"], cfg["horizon"])
        res["grid_vi"][f"{dt:g}"] = {"dt_ratio": dt, "iters": cfg["iters"], "cost": c, "first_upright_median": first,
                                     "upright_last_quarter": up}
    res["similarity_check_max_abs_diff"] = similarity_check(cfg)
    # in-range test systems whose tau0 lies outside the range of that seed's 64 sampled training systems
    cnt, tot = 0, 0
    for seed in cfg["seeds"]:
        t = draw_systems(seed, cfg)
        tr = [np.sqrt(v[1] / v[2]) for v in t.values() if v[3] == "train"]
        te = [np.sqrt(v[1] / v[2]) for v in t.values() if v[3] == "test"]
        cnt += sum(1 for x in te if x < min(tr) or x > max(tr))
        tot += len(te)
    res["test_outside_sample"] = {"count": cnt, "total": tot}
    fam = Family.from_cfg(cfg)
    systems = []
    for seed in cfg["seeds"]:
        for uid, (m_, l_, g_, set_) in draw_systems(seed, cfg).items():
            S = Systems([m_], [l_], [g_], fam)
            systems.append({"seed": seed, "uid": uid, "set": set_, "m": m_, "l": l_, "g": g_, "tau0": float(S.tau0[0]),
                            "mgl": float(S.mgl[0])})
    out = {"description": "Pendulum family: zero-shot transfer across physical scales (exploratory, outside finance). "
                          "Costs are mean natural cost per step (c~) over the horizon from the hanging position. "
                          "tg = zero-shot - seen per test system; vs_nat = zero-shot - natural units zero-shot. "
                          "Intervals: 95% percentile bootstrap over test systems, stratified by seed (10,000 resamples). "
                          "Generated by experiments/pendulum/pendulum.py.",
           "settings": cfg, "missing_fits": missing, **res, "systems": systems, "fits": fits, "fits_extra": extra}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    m = macros(res, cfg, fits)
    lines = ["% generated by experiments/pendulum/pendulum.py report (exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print_summary(res)
    figure(res, Path(args.png) if args.png else None)
    print(f"wrote {OUT_JSON}, {OUT_TEX}, {OUT_FIG}")


def print_summary(res):
    for part in ("in_range", "out_range"):
        print(f"== {part}")
        for rep, r in res[part].items():
            if rep == "contrasts":
                for k, st in r.items():
                    print(f"  {k}: {st['mean']:+.4f} [{st['lo']:+.4f}, {st['hi']:+.4f}]")
                continue
            vs = r.get("vs_nat", {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")})
            print(f"  {rep:5s} zero {r['zero']['mean']:.4f} [{r['zero']['lo']:.4f}, {r['zero']['hi']:.4f}]  "
                  f"seen {r['seen']['mean']:.4f}  TG {r['tg']['mean']:+.4f} [{r['tg']['lo']:+.4f}, {r['tg']['hi']:+.4f}]  "
                  f"vs nat {vs['mean']:+.4f} [{vs['lo']:+.4f}, {vs['hi']:+.4f}]  upright {r['upright_zero']['mean']:.3f}"
                  f"/{r['upright_seen']['mean']:.3f}")
            if "tg_by_fit" in r:
                print(f"        TG per seen fit: min {r['tg_by_fit']['min']:+.4f} max {r['tg_by_fit']['max']:+.4f} "
                      f"sd {r['tg_by_fit']['sd'] if r['tg_by_fit']['sd'] is not None else float('nan'):.4f} "
                      f"(n={len(r['tg_by_fit']['values'])})")
            if "by_tau" in r:
                print("        vs nat by tau0 tercile: " + "  ".join(f"{t} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
                                                              for t, v in r["by_tau"].items()))
            for sp in ("slow", "fast"):
                if sp in r:
                    print(f"        {sp}: zero {r[sp]['zero']['mean']:.4f} TG {r[sp]['tg']['mean']:+.4f} "
                          f"[{r[sp]['tg']['lo']:+.4f}, {r[sp]['tg']['hi']:+.4f}] upright {r[sp]['upright_zero']['mean']:.3f}")
    print("== out of range, pooled at half the seen data")
    for rep, r in res.get("out_range_pooled", {}).items():
        print(f"  {rep:5s} seen {r['seen']['mean']:.4f}  TG {r['tg']['mean']:+.4f} [{r['tg']['lo']:+.4f}, {r['tg']['hi']:+.4f}]  "
              f"slow {r['slow']['tg']['mean']:+.4f}  fast {r['fast']['tg']['mean']:+.4f}")
    print("== eps sweep")
    for rep, r in res["eps_sweep"].items():
        row = "  ".join(f"{k}: {v['tg']['mean']:+.4f} [{v['tg']['lo']:+.4f}, {v['tg']['hi']:+.4f}]"
                        for k, v in sorted(r.items(), key=lambda kv: kv[1].get("eps", 9)) if "tg" in v)
        ex = r.get("excess_vs_nat", {}).get("mean", float("nan"))
        print(f"  {rep:5s} {row}  excess vs nat {ex:+.4f}  slope {r['slope']['mean']:+.4f}")


LABEL = {"nat": "natural", "raw": "raw", "rawr": "raw$^{s}$", "rawi": "raw+info", "rawd": "raw+desc"}


def figure(res, png: Path | None):
    """(a, b) transfer gap and zero-shot cost relative to natural units, by representation, in and out of the
    training range of tau0 (in (b) also the seen policies with the out-of-range systems pooled at half their data);
    (c) the transfer gap against the test system's share eps of the seen policy's data (log axis; with the follow-up
    seeds and two-level intervals when present); (d) raw+info minus natural units, zero-shot, for each learner. Blue
    is always a transfer gap, orange always a zero-shot cost relative to natural units (as in fig_controls)."""
    sys.path.insert(0, str(REPO / "paper"))
    import figstyle as fs  # noqa: E402  (sets the Agg backend)
    import matplotlib.pyplot as plt  # noqa: E402

    TG_COL, VS_COL = fs.CATEGORICAL[0], fs.CATEGORICAL[1]
    fu = res.get("followups", {})
    reps = [r for r in REPS if r in res["in_range"] or r in res["out_range"]]
    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 4.6), gridspec_kw=dict(height_ratios=[1, 0.92]))
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    pooled = res.get("out_range_pooled", {})
    for ax, letter, part, title in ((axes[0], "a", "in_range", "trees, in range"),
                                    (axes[1], "b", "out_range", "trees, out of range")):
        block = dict(res[part])
        series = [("tg", "transfer gap (each test system 1/72 of the seen data)", TG_COL, "o", 3.8, True),
                  ("vs_nat", "zero-shot cost $-$ natural units", VS_COL, "D", 3.2, True)]
        if part == "out_range" and pooled:
            series.insert(1, ("tg_pool", "transfer gap (out-of-range systems pooled at 1/2 of the seen data)", TG_COL,
                              "o", 3.8, False))
            for rep in pooled:
                if rep in block:
                    block[rep] = dict(block[rep], tg_pool=pooled[rep]["tg"])
        offs = np.linspace(-0.24, 0.24, len(series)) if len(series) == 3 else (-0.15, 0.15)
        for (key, lab, col, mk, ms, filled), off in zip(series, offs):
            first = True
            for i, rep in enumerate(reps):
                if rep not in block or key not in block[rep] or (key == "vs_nat" and rep == "nat"):
                    continue
                ci = block[rep][key]
                ax.plot([i + off, i + off], [ci["lo"], ci["hi"]], color=col, lw=1.2, solid_capstyle="round")
                ax.plot(i + off, ci["mean"], mk, color=col, ms=ms, mfc=col if filled else fs.SURFACE,
                        mec=fs.SURFACE if filled else col, mew=0.6 if filled else 0.9, label=lab if first else None)
                first = False
        ax.axhline(0, color=fs.INK2, lw=0.6)
        ax.set_xticks(np.arange(len(reps)))
        ax.set_xticklabels([LABEL[r] for r in reps], rotation=35, ha="right")
        ax.set_xlim(-0.5, len(reps) - 0.5)
        ax.grid(axis="x", visible=False)
        fs.panel_title(ax, letter, title)
    axes[0].set_ylabel(r"cost per step, $\tilde c$")
    # (c) the sweep
    ax = axes[2]
    sw = fu.get("sweep")
    if sw:
        pts = {r: sorted((v["eps"], v[f"tg_{r}"]) for v in sw["eps"].values()) for r in ("raw", "nat")}
        ex = sw["excess_raw"]
        title = f"transfer gap against $\\varepsilon$ ({len(sw['seeds'])} seeds)"
    else:
        old = res["eps_sweep"]
        pts = {r: sorted((v["eps"], v["tg"]) for v in old[r].values() if isinstance(v, dict) and "eps" in v)
               for r in ("raw", "nat") if r in old}
        ex = old.get("raw", {}).get("excess_vs_nat")
        title = r"transfer gap against $\varepsilon$"
    if ex:
        ax.axhspan(ex["lo"], ex["hi"], color=VS_COL, alpha=0.10, lw=0, zorder=0)
        ax.axhline(ex["mean"], color=VS_COL, lw=1.0, ls=(0, (4, 2)))
        ax.text(2 ** -6.3, ex["hi"] - 0.03, "raw zero-shot cost $-$ natural units", color=fs.INK2, fontsize=6.5,
                va="top", ha="left")
    for rep, mk, ls, face, lab in (("raw", "o", "-", TG_COL, "raw"), ("nat", "s", (0, (2, 1.5)), fs.SURFACE,
                                                                       "natural units")):
        if rep not in pts:
            continue
        e = np.array([p[0] for p in pts[rep]]) * (1.06 if rep == "nat" else 1 / 1.06)
        ax.plot(e, [p[1]["mean"] for p in pts[rep]], ls=ls, color=TG_COL, lw=1.0)
        for xx, (_, ci) in zip(e, pts[rep]):
            ax.plot([xx, xx], [ci["lo"], ci["hi"]], color=TG_COL, lw=1.2, solid_capstyle="round")
        ax.plot(e, [p[1]["mean"] for p in pts[rep]], mk, ls="none", color=TG_COL, mfc=face,
                mec=TG_COL if rep == "nat" else fs.SURFACE, mew=0.9 if rep == "nat" else 0.6, ms=3.8, label=lab)
    ax.axhline(0, color=fs.INK2, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlim(2 ** -6.5, 2 ** -0.6)
    ax.set_xticks([1 / 64, 1 / 16, 1 / 4, 1 / 2])
    ax.set_xticklabels(["1/64", "1/16", "1/4", "1/2"])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(r"share $\varepsilon$ of the seen data")
    ax.set_ylabel(r"cost per step, $\tilde c$")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=2.2, borderaxespad=0.2)
    fs.panel_title(ax, "c", title)
    # (d) information versus division, by learner
    ax = axes[3]
    rows = [("trees", res["in_range"].get("rawi", {}).get("vs_nat"), res["out_range"].get("rawi", {}).get("vs_nat"))]
    if "x4" in fu and "rawi" in fu["x4"]["zero"]:
        z = fu["x4"]["zero"]["rawi"]
        rows.append(("trees, 4$\\times$ data", z["in"].get("vs_nat"), z["out"].get("vs_nat")))
    if "neural" in fu and "rawi" in fu["neural"]["zero"]:
        z = fu["neural"]["zero"]["rawi"]
        rows.append(("neural", z["in"].get("vs_nat"), z["out"].get("vs_nat")))
    for j, (lab, cin, cout) in enumerate(rows):
        for ci, off, filled, leg in ((cin, -0.12, True, "in range"), (cout, 0.12, False, "out of range")):
            if ci is None:
                continue
            ax.plot([j + off, j + off], [ci["lo"], ci["hi"]], color=VS_COL, lw=1.2, solid_capstyle="round")
            ax.plot(j + off, ci["mean"], "D", color=VS_COL, ms=3.4, mfc=VS_COL if filled else fs.SURFACE,
                    mec=fs.SURFACE if filled else VS_COL, mew=0.6 if filled else 0.9, label=leg if j == 0 else None)
    ax.axhline(0, color=fs.INK2, lw=0.6)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", borderaxespad=0.2)
    fs.panel_title(ax, "d", "raw+info $-$ natural units, zero-shot")
    handles = {}
    for a_ in axes[:2]:
        for h_, l_ in zip(*a_.get_legend_handles_labels()):
            handles.setdefault(l_, h_)
    order = [k for k in handles if "1/72" in k] + [k for k in handles if "pooled" in k] + \
            [k for k in handles if "natural units" in k and "transfer" not in k]
    fig.legend([handles[k] for k in order], order, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.5)
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    if png:
        fig.savefig(png, dpi=200)
    plt.close(fig)


# ======================================================================================= CARL's pendulum
# Review round 5 (R4, exploratory, post hoc): CARL's contextual Pendulum (Benjamins et al., TMLR 2023), i.e. gym's
# Pendulum-v1 with (g, l, m) as context and dt, max_speed, max_torque fixed in SI units, so that natural units are
# only approximately right. Re-implemented here (no gym/CARL install); see CARL_DOC. Everything below is additive: it
# writes only to results/transfer/carl_cache/ (keys carl_*), results/transfer/carl_pendulum.json,
# paper/numbers_carl.tex and paper/figures/fig_carl.pdf.
CARL_DOC = """CARL's pendulum (review round 5, R4; exploratory, post hoc).

    python experiments/pendulum/pendulum.py carl-check  --workers 8     # grid value iteration: which contexts are solvable
    python experiments/pendulum/pendulum.py carl-pilot  ...             # nat+ctx only: settings and ranges
    python experiments/pendulum/pendulum.py carl-run    --learner trees --workers 12 --threads 1
    python experiments/pendulum/pendulum.py carl-run    --learner nn --workers 2 --device cuda:0
    python experiments/pendulum/pendulum.py carl-report

Environment: gym Pendulum-v1 (gymnasium classic_control/pendulum.py), theta = 0 upright:
    u = clip(u, -2, 2); w' = clip(w + (3 g/(2 l) sin(theta) + 3/(m l^2) u) dt, -8, 8); theta' = theta + w' dt;
    cost = angle_normalize(theta)^2 + 0.1 w^2 + 0.001 u^2 (state before the step), 200 steps,
    start theta ~ U[-pi, pi], w ~ U[-1, 1]; defaults g = 10, m = 1, l = 1, dt = 0.05 s, max_speed = 8 rad/s,
    max_torque = 2 N m. CARL's context features are g, l, m, dt, max_speed and the two start bounds (max_torque is not
    one); we vary g, l, m and keep the others at gym's defaults, as CARL's experiments do.
The dynamics depend on a = 3 g/(2 l) and b = 3/(m l^2) only. Natural units: time tau0 = 1/sqrt(a) = sqrt(2 l/(3 g)),
torque m g l/2 (the largest gravity torque on the rod). In them the problem depends on the groups delta = dt/tau0,
ubar = max_torque/(m g l/2) and wmax = max_speed tau0 = max_speed dt/delta (a function of delta, as dt and max_speed
are fixed); the cost's weight on the natural angular velocity, 0.1/tau0^2 = 0.1 (delta/dt)^2, also follows delta, and
the torque cost of each action fraction is the same everywhere. Two contexts with the same (delta, ubar) are the same
problem; across the family they vary, so natural units are only approximately invariant (as the one-cent tick
breaks it for stocks). (g, l, m) -> (k g, k l, m/k^2) leaves a, b, and so the raw problem itself, unchanged.
Actions: torques {-1, -1/2, 0, 1/2, 1} x max_torque (gym's torque is continuous; discretised as in the family above).
Representations (all trained on gym's cost; all scored on it): raw (cos, sin, w): gym's observation; rawc =
raw+ctx (raw + g, l, m): CARL's concatenation of the varying context ("concat non-static"); rawi = raw+info (raw +
tau0); nat (cos, sin, w tau0); natc = nat+ctx (nat + log delta, log ubar, log wmax), the analogue of MS+tick.
raw+info has the information of nat, raw+ctx that of nat+ctx.

Contexts (carl_contexts): per seed 64 training and 32 in-range test contexts, g, l, m each log-uniform in [2/3, 3/2] x
the default; 8 "sim" contexts (a draw moved by (k g, k l, m/k^2) until each feature is >= 1.25x outside the box: the
context is out of range, the problem is in range) and 8 "ext" contexts (g, m below and l above the box by 1.25-2x:
slower, lighter pendulums; every feature and delta out of range). Evaluation: 16 episodes per test context from gym's
start distribution, the same draws for every policy; the oracle is grid value iteration per context (`carl-oracle`).
Designs, as in the family above: zero-shot; a seen comparator per group of 8 in-range test contexts (1/72 each); one
pooled comparator with all 16 out-of-range contexts at half the data (1/32 each). 3 seeds, 2 learners, 180 fits.

Pilots (nat+ctx only; records in results/transfer/carl_pilots.jsonl and carl_check.json):
  * `carl-check --map`: with gym's limits fixed, fast pendulums with little torque cannot be swung up at all (the
    speed limit caps the energy at the bottom); gamma 0.98 truncated at 50 iterations is within 0.07 per step of
    gamma 0.99 with 300 iterations on solvable contexts in the training range, so FQI keeps the family's settings.
  * `carl-check --boxes`: grid value iteration solves (>= 90% of episodes) 0.81 of contexts drawn from [0.5, 2]^3, the
    range first planned, 0.98 from [2/3, 3/2]^3.
  * `carl-pilot` (trees, seeds 0, 1): nat+ctx solved 0.72-0.82 of in-range test episodes on [0.5, 2]^3 (oracle
    0.88-0.91), 0.97-0.99 on [2/3, 3/2]^3 (oracle 0.98-1.00). Hence the box [2/3, 3/2]. (The pilot also printed
    nat+ctx's out-of-range costs.)
  * `carl-pilot --learner nn`: the settings of the family's neural follow-up, and four variants (20 epochs; 20
    epochs at learning rate 3e-4; tanh; 512 units); none lowered nat+ctx's mean in-range cost by more than 0.012, so
    the family's settings were kept.
Run of 1 November 2026: trees 6 workers x 2 threads, about 7 minutes per fit, 1.7 hours; neural 2 GPU workers, about
1.4 minutes per fit, 1.0 hour.
"""
CARL_CACHE = REPO / "results" / "transfer" / "carl_cache"  # one JSON per fit, keys carl_*; plus carl_config.json
CARL_JSON = REPO / "results" / "transfer" / "carl_pendulum.json"
CARL_TEX = REPO / "paper" / "numbers_carl.tex"
CARL_FIG = REPO / "paper" / "figures" / "fig_carl.pdf"
CARL_PILOTS = REPO / "results" / "transfer" / "carl_pilots.jsonl"
CARL_REPS = ("natc", "nat", "rawc", "rawi", "raw")
CARL_LABEL = {"natc": "nat+ctx", "nat": "nat", "rawc": "raw+ctx", "rawi": "raw+info", "raw": "raw"}
GYM = dict(g=10.0, l=1.0, m=1.0, dt=0.05, max_speed=8.0, max_torque=2.0)
CARL_CFG = dict(seeds=[0, 1, 2], n_train=64, n_test=32, n_sim=8, n_ext=8, group=8,
                box=[2 / 3, 1.5],  # training factors of the default, log-uniform, for each of g, l, m (set in pilots)
                mgl_box=None,  # optional [lo, hi] on the factor of m g l (set in pilots, if needed)
                out_factor=[1.25, 2.0],  # ext contexts: each feature this many times beyond the box's edge
                states=576, iters=50, rounds=200, gamma=0.98, horizon=200, episodes=16,
                reps=list(CARL_REPS), nn=dict(CFG["nn"]))
# uids: training 0..63, in-range test 100..131, context out of range with the problem in range (sim) 200..207, every
# feature and the groups out of range (ext) 300..307
C_TRAIN0, C_TEST0, C_SIM0, C_EXT0 = 0, 100, 200, 300


class Ctx:
    """A batch of CARL pendulum contexts (one per row); gym's fixed dt, max_speed, max_torque."""

    def __init__(self, g, l, m):
        self.g, self.l, self.m = (np.asarray(x, float) for x in (g, l, m))
        self.dt, self.max_speed, self.max_torque = GYM["dt"], GYM["max_speed"], GYM["max_torque"]
        self.a = 3 * self.g / (2 * self.l)
        self.b = 3 / (self.m * self.l ** 2)
        self.tau0 = 1 / np.sqrt(self.a)
        self.unat = self.m * self.g * self.l / 2
        self.delta = self.dt / self.tau0
        self.ubar = self.max_torque / self.unat
        self.wmax = self.max_speed * self.tau0

    def __len__(self):
        return len(self.g)


def carl_step(C: Ctx, th, om, u):
    """gym Pendulum-v1's step (theta wrapped to (-pi, pi], which changes neither the cost nor the observation)."""
    u = np.clip(u, -C.max_torque, C.max_torque)
    om2 = np.clip(om + (C.a * np.sin(th) + C.b * u) * C.dt, -C.max_speed, C.max_speed)
    return wrap(th + om2 * C.dt), om2


def carl_cost(th, om, u):
    return wrap(th) ** 2 + 0.1 * om ** 2 + 0.001 * u ** 2


def carl_features(rep: str, C: Ctx, th, om) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)

    def b(x):
        return np.broadcast_to(x, c.shape)

    if rep == "raw":
        return np.column_stack([c, s, om])
    if rep == "rawc":
        return np.column_stack([c, s, om, b(C.g), b(C.l), b(C.m)])
    if rep == "rawi":
        return np.column_stack([c, s, om, b(C.tau0)])
    if rep == "nat":
        return np.column_stack([c, s, om * C.tau0])
    if rep == "natc":
        return np.column_stack([c, s, om * C.tau0, b(np.log(C.delta)), b(np.log(C.ubar)), b(np.log(C.wmax))])
    raise ValueError(rep)


def carl_in_box(fg, fl, fm, cfg) -> np.ndarray:
    lo, hi = cfg["box"]
    ok = (fg >= lo) & (fg <= hi) & (fl >= lo) & (fl <= hi) & (fm >= lo) & (fm <= hi)
    if cfg.get("mgl_box"):
        ok &= (fg * fl * fm >= cfg["mgl_box"][0]) & (fg * fl * fm <= cfg["mgl_box"][1])
    return ok


def carl_draw_box(rng, n, cfg):
    """n factor triples (g, l, m) log-uniform in the box (rejection sampling if a box on m g l is set)."""
    lo, hi = np.log(cfg["box"][0]), np.log(cfg["box"][1])
    out = np.zeros((0, 3))
    while len(out) < n:
        f = np.exp(rng.uniform(lo, hi, (4 * n, 3)))
        out = np.vstack([out, f[carl_in_box(f[:, 0], f[:, 1], f[:, 2], cfg)]])
    return out[:n]


def carl_contexts(seed: int, cfg) -> dict:
    """uid -> (g, l, m, set) for one seed. train/test: factors (of gym's default) drawn from the training distribution.
    sim: a draw from it moved by (g, l, m) -> (k g, k l, m/k^2), which leaves the problem itself unchanged, with k the
    smallest factor that puts each of g, l and m at least 1.25 times outside the training box (k > 1 for half, k < 1
    for the other half): the context is out of range, the groups (and the raw problem) are in range. ext: every
    feature outside the box (extrapolation in the sense of Kirk et al. and CARL), each drawn log-uniformly between
    out_factor[0] and out_factor[1] times beyond the box's edge, in the direction that stays solvable under gym's fixed
    torque and speed limits: weaker gravity, a longer and lighter rod (so tau0 lies above its training range, delta
    below it and wmax above it; ubar above or in its range)."""
    rng = np.random.default_rng([seed, 2027])
    lo, hi = cfg["box"]
    out = {}
    for uid0, n, name in ((C_TRAIN0, cfg["n_train"], "train"), (C_TEST0, cfg["n_test"], "test")):
        f = carl_draw_box(rng, n, cfg)
        for i in range(n):
            out[uid0 + i] = (GYM["g"] * f[i, 0], GYM["l"] * f[i, 1], GYM["m"] * f[i, 2], name)
    n = cfg["n_sim"]
    f = carl_draw_box(rng, n, cfg)
    margin = 1.25
    for i in range(n):
        fg, fl, fm = f[i]
        if i < n // 2:  # up: g, l above the box, m below
            k = margin * max(hi / min(fg, fl), np.sqrt(fm / lo))
        else:
            k = min(lo / max(fg, fl), np.sqrt(fm / hi)) / margin
        out[C_SIM0 + i] = (GYM["g"] * fg * k, GYM["l"] * fl * k, GYM["m"] * fm / k ** 2,
                           "sim_up" if i < n // 2 else "sim_down")
    n = cfg["n_ext"]
    a, b = np.log(cfg["out_factor"][0]), np.log(cfg["out_factor"][1])
    e = np.exp(rng.uniform(a, b, (n, 3)))
    for i in range(n):
        out[C_EXT0 + i] = (GYM["g"] * lo / e[i, 0], GYM["l"] * hi * e[i, 1], GYM["m"] * lo / e[i, 2], "ext")
    return out


def carl_ctx_of(table: dict, uids) -> Ctx:
    p = np.array([table[u][:3] for u in uids], float).reshape(-1, 3)
    return Ctx(p[:, 0], p[:, 1], p[:, 2])


def carl_transitions(C1: Ctx, n_states: int, seed: int, uid: int) -> dict:
    """Branched transitions of one context: n_states states uniform on the reachable set (theta in [-pi, pi], w in
    [-max_speed, max_speed] rad/s), each with all five torques; nested in n_states, as `transitions`."""
    rng = np.random.default_rng([seed, uid, 31])
    U = rng.random((n_states, 2))
    th = np.pi * (2 * U[:, 0] - 1)
    om = C1.max_speed * (2 * U[:, 1] - 1)
    u = ACTIONS[:, None] * C1.max_torque
    th2, om2 = carl_step(C1, th[None, :], om[None, :], u)
    return {"th": th, "om": om, "th2": th2, "om2": om2, "c": carl_cost(th[None, :], om[None, :], u)}


def carl_starts(seed: int, uid: int, episodes: int):
    """gym's start distribution, theta ~ U[-pi, pi], w ~ U[-1, 1]; the same draws for every policy."""
    rng = np.random.default_rng([seed, uid, 37])
    return np.pi * rng.uniform(-1, 1, episodes), rng.uniform(-1, 1, episodes)


def carl_rollout(policy, C: Ctx, th0, om0, horizon: int):
    """Greedy rollouts, one per row of C; policy(th, om) -> action indices. Returns the mean gym cost per step and
    the share of the last quarter of the horizon spent upright (|theta| < 0.3)."""
    th, om = th0.copy(), om0.copy()
    tot, up = np.zeros(len(th)), np.zeros(len(th))
    last = max(horizon // 4, 1)
    for t in range(horizon):
        u = ACTIONS[policy(th, om)] * C.max_torque
        tot += carl_cost(th, om, u)
        th, om = carl_step(C, th, om, u)
        if t >= horizon - last:
            up += np.abs(th) < 0.3
    return tot / horizon, up / last


def carl_grid_vi(g, l, m, gamma, iters, horizon=200, episodes=64, nth=361, nom=321, seed=0):
    """Value iteration on a (theta, w) grid with bilinear interpolation, for one context (raw units; w in
    [-max_speed, max_speed], the reachable set), truncated at `iters` iterations as FQI is; the greedy policy's mean
    gym cost per step and upright share of the last quarter from gym's start distribution."""
    C = Ctx([g], [l], [m])
    ths = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    W = C.max_speed
    oms = np.linspace(-W, W, nom)
    TH, OM = np.meshgrid(ths, oms, indexing="ij")

    def interp(V, th, om):
        i = (wrap(th) + np.pi) / (2 * np.pi) * nth
        j = (np.clip(om, -W, W) + W) / (2 * W) * (nom - 1)
        i0 = np.floor(i).astype(int)
        j0 = np.minimum(np.floor(j).astype(int), nom - 2)
        fi, fj = i - i0, j - j0
        i0, i1 = i0 % nth, (i0 + 1) % nth
        return ((1 - fi) * (1 - fj) * V[i0, j0] + fi * (1 - fj) * V[i1, j0] + (1 - fi) * fj * V[i0, j0 + 1]
                + fi * fj * V[i1, j0 + 1])

    acts = ACTIONS * C.max_torque
    nxt = [carl_step(C, TH, OM, u) for u in acts]
    cs = [carl_cost(TH, OM, u) for u in acts]
    V = np.zeros_like(TH)
    for _ in range(iters):
        V = np.min([cs[a] + gamma * interp(V, *nxt[a]) for a in range(N_ACT)], axis=0)

    def policy(th, om):
        return np.argmin([carl_cost(th, om, u) + gamma * interp(V, *carl_step(C, th, om, u)) for u in acts], axis=0)

    th0, om0 = carl_starts(seed, 999, episodes)
    Cx = Ctx(np.full(episodes, g), np.full(episodes, l), np.full(episodes, m))
    c, up = carl_rollout(policy, Cx, th0, om0, horizon)
    return float(c.mean()), float(up.mean()), float((up > 0.9).mean())


def _carl_vi_job(job):
    g, l, m, gamma, iters = job[:5]
    return job, carl_grid_vi(g, l, m, gamma, iters, episodes=job[5] if len(job) > 5 else 64)


CARL_BOXES = {"wide": [0.5, 2.0], "chosen": [2 / 3, 1.5], "mid": [0.625, 1.6]}


def cmd_carl_check(args, cfg):
    """(1) The transformation (g, l, m) -> (k g, k l, m/k^2) leaves gym's trajectories unchanged. (2) --map: grid
    value iteration on a map of (tau0 factor, ubar) for each (gamma, iterations): mean cost per step, upright share,
    share of episodes upright for > 90% of the last quarter. (3) --boxes: the same on 42 contexts drawn log-uniformly
    from each candidate box of factors (g, l, m each in [lo, hi]), gamma and iterations as FQI. Stored in
    results/transfer/carl_check.json (map, boxes, and the macros they give)."""
    import multiprocessing as mp

    C = Ctx([10.0, 20.0, 5.0], [1.0, 2.0, 0.5], [1.0, 0.25, 4.0])
    th, om = np.full(3, np.pi - 0.1), np.zeros(3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        th, om = carl_step(C, th, om, ACTIONS[rng.integers(0, N_ACT)] * C.max_torque)
    spread = float(max(np.ptp(th), np.ptp(om)))
    print("similarity transformation, max spread after 200 random steps:", spread)
    path = REPO / "results" / "transfer" / "carl_check.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    if isinstance(store, list):  # the first version stored the map alone
        store = {"map": store}
    store["transformation_max_spread"] = spread
    if args.map:
        taus = [float(x) for x in args.taus]
        ubars = [float(x) for x in args.ubars]
        jobs = []
        for gam, it in zip(args.gammas, args.vi_iters):
            for t in taus:
                for ub in ubars:  # realise (tau0 factor t, ubar) with g = 10: l = t^2, m from ubar
                    l = t ** 2
                    jobs.append((10.0, l, GYM["max_torque"] / (ub * 10.0 * l / 2), float(gam), int(it)))
        with mp.get_context("spawn").Pool(args.workers) as pool:
            res = pool.map(_carl_vi_job, jobs)
        new = []
        for (g, l, m, gam, it), (c, up, solved) in res:
            Cx = Ctx([g], [l], [m])
            new.append(dict(g=g, l=l, m=m, gamma=gam, iters=it, tau_factor=float(np.sqrt(l)), ubar=float(Cx.ubar[0]),
                            delta=float(Cx.delta[0]), wmax=float(Cx.wmax[0]), cost=c, upright=up, solved=solved))
        done = {(r["gamma"], r["iters"]) for r in new}
        store["map"] = [r for r in store.get("map", []) if (r["gamma"], r["iters"]) not in done] + new
    if args.boxes:
        rng = np.random.default_rng(5)
        jobs = []
        for name, (lo, hi) in CARL_BOXES.items():
            f = np.exp(rng.uniform(np.log(lo), np.log(hi), (42, 3)))
            jobs += [(10 * a, b, c, cfg["gamma"], cfg["iters"], 32, name) for a, b, c in f]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            res = pool.map(_carl_vi_job, jobs)
        store["boxes"] = {name: {"box": CARL_BOXES[name], "contexts": [dict(g=j[0], l=j[1], m=j[2], cost=r[0], upright=r[1],
                                                                            solved=r[2]) for j, r in res if j[6] == name]}
                          for name in CARL_BOXES}
    for gam, it in sorted({(r["gamma"], r["iters"]) for r in store.get("map", [])}):
        rows = [r for r in store["map"] if r["gamma"] == gam and r["iters"] == it]
        ubars = sorted({round(r["ubar"], 6) for r in rows})
        print(f"== gamma {gam}, {it} iterations: cost/step (share of episodes upright)")
        print("tau\\ubar " + " ".join(f"{u:>11.2f}" for u in ubars))
        for t in sorted({r["tau_factor"] for r in rows}):
            row = sorted([r for r in rows if abs(r["tau_factor"] - t) < 1e-9], key=lambda r: r["ubar"])
            print(f"{t:8.2f} " + " ".join(f"{r['cost']:6.3f}({r['solved']:.2f})" for r in row))
    mac = {}
    for name, b in store.get("boxes", {}).items():
        c = np.array([x["cost"] for x in b["contexts"]])
        sv = np.array([x["solved"] for x in b["contexts"]])
        N = {"wide": "Wide", "chosen": "Box", "mid": "Mid"}[name]
        mac[f"CarlCheck{N}Solvable"] = f"{np.mean(sv >= 0.9):.2f}"
        mac[f"CarlCheck{N}CostMean"] = f"{c.mean():.3f}"
        mac[f"CarlCheck{N}CostMedian"] = f"{np.median(c):.3f}"
        print(f"box {name} {b['box']}: contexts solvable (>= 90% of episodes) {np.mean(sv >= 0.9):.2f}, cost mean "
              f"{c.mean():.3f}, median {np.median(c):.3f}, max {c.max():.2f}")
    mp_ = store.get("map", [])
    short = {(round(r["tau_factor"], 4), round(r["ubar"], 4)): r for r in mp_ if r["gamma"] == cfg["gamma"] and r["iters"] == cfg["iters"]}
    long_ = {(round(r["tau_factor"], 4), round(r["ubar"], 4)): r for r in mp_ if r["gamma"] == 0.99 and r["iters"] == 300}
    if short and long_:
        lo, hi = cfg["box"]
        inside = [k for k in short if k in long_ and np.sqrt(lo / hi) - 1e-9 <= k[0] <= np.sqrt(hi / lo) + 1e-9
                  and 0.4 / hi ** 3 - 1e-9 <= k[1] <= 0.4 / lo ** 3 + 1e-9 and short[k]["solved"] >= 0.9]
        gain = max(short[k]["cost"] - long_[k]["cost"] for k in inside)
        mac["CarlCheckLongGainMax"] = f"{gain:.3f}"
        d = short.get((1.0, 0.4))
        if d:
            mac["CarlCheckDefaultCost"] = f"{d['cost']:.3f}"
            mac["CarlCheckDefaultCostLong"] = f"{long_[(1.0, 0.4)]['cost']:.3f}"
        print(f"largest gain of gamma 0.99 / 300 iterations over {cfg['gamma']} / {cfg['iters']} on solvable cells "
              f"inside the training ranges: {gain:.3f} ({len(inside)} cells)")
    store["macros"] = mac
    path.write_text(json.dumps(store, indent=1))


def carl_groups(cfg):
    """In-range test uids and their groups (for the seen comparators), and the out-of-range uids (sim, then ext)."""
    I = [C_TEST0 + i for i in range(cfg["n_test"])]
    G = cfg["group"]
    gin = [I[i:i + G] for i in range(0, len(I), G)]
    O = [C_SIM0 + i for i in range(cfg["n_sim"])] + [C_EXT0 + i for i in range(cfg["n_ext"])]
    return I, gin, O


def carl_plan(cfg, learner: str, seeds=None, reps=None, kinds=("zero", "seen_in", "seen_outpool")) -> list[dict]:
    """Every fit of one learner: zero-shot on the training contexts; seen comparators for each group of in-range test
    contexts (each 1/72 of the data, as in design (a) of the family above); one pooled comparator with all
    out-of-range contexts at half the data (each 1/32, as in design (d)). All fits have 64 x states states."""
    n, nt, G = cfg["states"], cfg["n_train"], cfg["group"]
    T = [C_TRAIN0 + i for i in range(nt)]
    I, gin, O = carl_groups(cfg)
    k_seen = n * nt // (nt + G)
    assert k_seen * (nt + G) == n * nt, "states must make the seen fits size-matched"
    k_out = n * nt // (2 * len(O))
    assert nt * (n // 2) + len(O) * k_out == nt * n
    lt = {"trees": "tr", "nn": "nn"}[learner]
    fits = []
    for seed in (seeds if seeds is not None else cfg["seeds"]):
        for rep in (reps if reps is not None else cfg["reps"]):
            base = dict(seed=seed, rep=rep, learner=learner, family="carl")
            if "zero" in kinds:
                fits.append(dict(base, key=f"carl_{lt}_s{seed}_{rep}_zero", kind="zero", group=0, eps=0.0,
                                 parts=[(u, n) for u in T], evals=I + O))
            if "seen_in" in kinds:
                for g, grp in enumerate(gin):
                    fits.append(dict(base, key=f"carl_{lt}_s{seed}_{rep}_seenin{g}", kind="seen_in", group=g,
                                     eps=1 / (nt + G), parts=[(u, k_seen) for u in T + grp], evals=grp))
            if "seen_outpool" in kinds:
                fits.append(dict(base, key=f"carl_{lt}_s{seed}_{rep}_seenoutpool", kind="seen_outpool", group=0,
                                 eps=1 / (2 * len(O)), parts=[(u, n // 2) for u in T] + [(u, k_out) for u in O], evals=O))
    order = {"zero": 0, "seen_in": 1, "seen_outpool": 2}
    return sorted(fits, key=lambda f: (order[f["kind"]], f["seed"], CARL_REPS.index(f["rep"]), f["group"]))


def carl_assemble(fit, cfg):
    table = carl_contexts(fit["seed"], cfg)
    th, om, th2, om2, c, rows = [], [], [], [], [], []
    for uid, k in fit["parts"]:
        d = carl_transitions(carl_ctx_of(table, [uid]), k, fit["seed"], uid)
        th.append(d["th"]), om.append(d["om"]), th2.append(d["th2"]), om2.append(d["om2"]), c.append(d["c"])
        rows.append(np.full(k, uid))
    th, om = np.concatenate(th), np.concatenate(om)
    th2, om2, c = np.concatenate(th2, axis=1), np.concatenate(om2, axis=1), np.concatenate(c, axis=1)
    C = carl_ctx_of(table, np.concatenate(rows))
    rep = fit["rep"]
    X = carl_features(rep, C, th, om)
    Xn = np.stack([carl_features(rep, C, th2[b], om2[b]) for b in range(N_ACT)])
    return X, Xn, c


def carl_evaluate(model, fit, cfg, threads: int) -> dict:
    table = carl_contexts(fit["seed"], cfg)
    E = cfg["episodes"]
    uids = fit["evals"]
    C = carl_ctx_of(table, np.repeat(uids, E))
    st = [carl_starts(fit["seed"], u, E) for u in uids]
    th0, om0 = np.concatenate([a for a, _ in st]), np.concatenate([b for _, b in st])
    rep = fit["rep"]

    def policy(th, om):
        return model.adv(carl_features(rep, C, th, om), threads).argmin(axis=1)

    cost, up = carl_rollout(policy, C, th0, om0, cfg["horizon"])
    return {str(u): {"cost": cost[i * E:(i + 1) * E].round(6).tolist(), "upright": up[i * E:(i + 1) * E].tolist()}
            for i, u in enumerate(uids)}


def carl_run_fit(job) -> str:
    fit, cfg, cache, threads, device = job
    path = Path(cache) / f"{fit['key']}.json"
    if path.exists():
        return f"{fit['key']}: cached"
    t0 = time.time()
    X, Xn, c = carl_assemble(fit, cfg)
    if fit["learner"] == "nn":
        model, change = fqi_nn(X, Xn, c, cfg["gamma"], cfg["iters"], cfg["nn"], seed=fit["seed"], device=device)
    else:
        model, change = fqi(X, Xn, c, cfg["gamma"], cfg["iters"], cfg["rounds"], threads, seed=fit["seed"])
    t1 = time.time()
    ev = carl_evaluate(model, fit, cfg, threads)
    rec = dict(fit, n_states=int(X.shape[0]), test_states=int(sum(k for u, k in fit["parts"] if u >= C_TEST0)),
               fit_seconds=round(t1 - t0, 1), eval_seconds=round(time.time() - t1, 1),
               last_change=change[-1] if change else None, eval=ev)
    rec.pop("parts")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(path)
    m = np.mean([np.mean(v["cost"]) for v in ev.values()])
    upr = np.mean([np.mean(v["upright"]) for v in ev.values()])
    return f"{fit['key']}: cost {m:.4f}, upright {upr:.3f}, fit {t1 - t0:.0f}s"


def _carl_oracle_job(job):
    seed, uid, g, l, m, gamma, iters, E = job
    C = Ctx([g], [l], [m])
    # grid value iteration as in carl_grid_vi, scored on the same start draws as the policies
    ths = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    W = C.max_speed
    oms = np.linspace(-W, W, 321)
    TH, OM = np.meshgrid(ths, oms, indexing="ij")
    nth, nom = 361, 321

    def interp(V, th, om):
        i = (wrap(th) + np.pi) / (2 * np.pi) * nth
        j = (np.clip(om, -W, W) + W) / (2 * W) * (nom - 1)
        i0 = np.floor(i).astype(int)
        j0 = np.minimum(np.floor(j).astype(int), nom - 2)
        fi, fj = i - i0, j - j0
        i0, i1 = i0 % nth, (i0 + 1) % nth
        return ((1 - fi) * (1 - fj) * V[i0, j0] + fi * (1 - fj) * V[i1, j0] + (1 - fi) * fj * V[i0, j0 + 1]
                + fi * fj * V[i1, j0 + 1])

    acts = ACTIONS * C.max_torque
    nxt = [carl_step(C, TH, OM, u) for u in acts]
    cs = [carl_cost(TH, OM, u) for u in acts]
    V = np.zeros_like(TH)
    for _ in range(iters):
        V = np.min([cs[a] + gamma * interp(V, *nxt[a]) for a in range(N_ACT)], axis=0)
    th0, om0 = carl_starts(seed, uid, E)
    Cx = Ctx(np.full(E, g), np.full(E, l), np.full(E, m))

    def policy(th, om):
        return np.argmin([carl_cost(th, om, u) + gamma * interp(V, *carl_step(Cx, th, om, u)) for u in acts], axis=0)

    c, up = carl_rollout(policy, Cx, th0, om0, 200)
    return seed, uid, c.round(6).tolist(), up.tolist()


def cmd_carl_oracle(args, cfg):
    """Grid value iteration per test context (the context known, exact dynamics; truncated at the FQI iterations
    and discount), on the policies' start draws: a reference for what the task allows. One file per seed."""
    import multiprocessing as mp

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    I, _, O = carl_groups(cfg)
    jobs = []
    for seed in cfg["seeds"]:
        if (cache / f"carl_oracle_s{seed}.json").exists():
            continue
        t = carl_contexts(seed, cfg)
        jobs += [(seed, u, *t[u][:3], cfg["gamma"], cfg["iters"], cfg["episodes"]) for u in I + O]
    if not jobs:
        print("oracle: cached")
        return
    with mp.get_context("spawn").Pool(int(args.workers)) as pool:
        res = pool.map(_carl_oracle_job, jobs)
    for seed in sorted({r[0] for r in res}):
        ev = {str(u): {"cost": c, "upright": up} for s, u, c, up in res if s == seed}
        (cache / f"carl_oracle_s{seed}.json").write_text(json.dumps(dict(seed=seed, kind="oracle", eval=ev)))
    print(f"oracle: {len(jobs)} contexts")


def carl_cfg_from_args(args) -> dict:
    cfg = json.loads(json.dumps(CARL_CFG))
    for k in ("seeds", "states", "iters", "rounds", "gamma", "episodes", "n_train", "n_test", "n_sim", "n_ext"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if getattr(args, "box", None):
        cfg["box"] = [float(x) for x in args.box]
    if getattr(args, "out_factor", None):
        cfg["out_factor"] = [float(x) for x in args.out_factor]
    for k in ("hidden", "layers", "act", "lr", "batch", "epochs"):
        v = getattr(args, f"nn_{k}", None)
        if v is not None:
            cfg["nn"][k] = v
    return cfg


def cmd_carl_run(args, cfg):
    import multiprocessing as mp

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    cpath = cache / "carl_config.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        if old != cfg:
            raise SystemExit(f"{cpath} differs from this configuration; use another --cache")
    else:
        cpath.write_text(json.dumps(cfg, indent=1))
    kinds = tuple(args.kinds) if args.kinds else ("zero", "seen_in", "seen_outpool")
    todo = carl_plan(cfg, args.learner, reps=args.reps, kinds=kinds)
    fits = [f for f in todo if not (cache / f"{f['key']}.json").exists()]
    if args.only:
        fits = [f for f in fits if any(s in f["key"] for s in args.only)]
    print(f"{len(fits)} {args.learner} fits to run, {args.workers} workers x {args.threads} threads, device "
          f"{args.device}", flush=True)
    t0 = time.time()
    devices = args.device.split(",")
    jobs = [(f, cfg, str(cache), int(args.threads), devices[i % len(devices)]) for i, f in enumerate(fits)]
    if args.workers == 1:
        for i, j in enumerate(jobs):
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {carl_run_fit(j)}", flush=True)
        return
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
        for i, msg in enumerate(pool.imap_unordered(carl_run_fit, jobs)):
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] {msg}", flush=True)


def cmd_carl_pilot(args, cfg):
    """nat+ctx only: zero-shot fits on the training contexts of the given seeds into a pilot cache, then each test
    context's cost against grid value iteration (the oracle). Appends a record to results/transfer/carl_pilots.jsonl."""
    cache = Path(args.cache)
    args.kinds, args.only = ["zero"], None
    args.reps = list(args.reps) if args.reps else ["natc"]
    cmd_carl_run(args, cfg)
    cmd_carl_oracle(args, cfg)
    I, _, O = carl_groups(cfg)
    orc = {}
    for seed in cfg["seeds"]:
        o = json.loads((cache / f"carl_oracle_s{seed}.json").read_text())
        for u, v in o["eval"].items():
            orc[(seed, int(u))] = (float(np.mean(v["cost"])), float(np.mean(np.array(v["upright"]) > 0.9)))
    lt = {"trees": "tr", "nn": "nn"}[args.learner]
    for rep in args.reps:
        for seed in cfg["seeds"]:
            f = json.loads((cache / f"carl_{lt}_s{seed}_{rep}_zero.json").read_text())
            rows = {}
            for part, sel in (("in", lambda u: C_TEST0 <= u < C_SIM0), ("sim", lambda u: C_SIM0 <= u < C_EXT0),
                              ("ext", lambda u: u >= C_EXT0)):
                us = [int(u) for u in f["eval"] if sel(int(u))]
                c = np.array([np.mean(f["eval"][str(u)]["cost"]) for u in us])
                s = np.array([np.mean(np.array(f["eval"][str(u)]["upright"]) > 0.9) for u in us])
                oc = np.array([orc[(seed, u)][0] for u in us])
                osv = np.array([orc[(seed, u)][1] for u in us])
                rows[part] = dict(cost=float(c.mean()), oracle=float(oc.mean()), solved=float(s.mean()),
                                  oracle_solved=float(osv.mean()), worst_excess=float((c - oc).max()),
                                  contexts_behind=int(((s < osv - 0.25)).sum()), n=len(us))
            rec = dict(learner=args.learner, rep=rep, seed=seed, box=cfg["box"], out_factor=cfg["out_factor"],
                       states=cfg["states"], iters=cfg["iters"], rounds=cfg["rounds"], gamma=cfg["gamma"],
                       nn=cfg["nn"] if args.learner == "nn" else None, fit_seconds=f["fit_seconds"], **rows,
                       cache=str(cache))
            with open(CARL_PILOTS, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[pilot {args.learner} {rep} box {cfg['box']} seed {seed}] " + "  ".join(
                f"{p}: cost {r['cost']:.3f} (oracle {r['oracle']:.3f}), solved {r['solved']:.2f} (oracle "
                f"{r['oracle_solved']:.2f}), worst excess {r['worst_excess']:.2f}, behind {r['contexts_behind']}/{r['n']}"
                for p, r in rows.items()) + f", fit {f['fit_seconds']:.0f}s", flush=True)


def carl_parts(u: int) -> str:
    return "in" if C_TEST0 <= u < C_SIM0 else ("sim" if C_SIM0 <= u < C_EXT0 else ("ext" if u >= C_EXT0 else "train"))


CARL_CONTRASTS = (("natc", "nat"), ("rawc", "natc"), ("rawi", "nat"), ("raw", "nat"), ("rawi", "natc"), ("raw", "natc"))


def carl_summarise(fits: list[dict], oracle: dict, cfg) -> dict:
    """Per learner and part (in, sim, ext): zero-shot cost, upright share, share of episodes solved (upright for more
    than 90% of the last quarter), cost relative to nat+ctx, transfer gap against the seen comparators (in range: the
    groups at 1/72; out of range: the pooled comparator at 1/32 each), and paired contrasts; two-level bootstrap
    (seeds, then contexts within a seed). oracle: (seed, uid) -> cost of grid value iteration."""
    res = {}
    for learner in ("trees", "nn"):
        F = [f for f in fits if f["learner"] == learner]
        if not F:
            continue
        Z, U, V, S = {}, {}, {}, {}
        for f in F:
            for u, v in f["eval"].items():
                k = (f["seed"], int(u))
                c = float(np.mean(v["cost"]))
                if f["kind"] == "zero":
                    Z.setdefault(f["rep"], {})[k] = c
                    U.setdefault(f["rep"], {})[k] = float(np.mean(v["upright"]))
                    V.setdefault(f["rep"], {})[k] = float(np.mean(np.array(v["upright"]) > 0.9))
                else:
                    S.setdefault(f["rep"], {})[k] = c
        reps = [r for r in CARL_REPS if r in Z]
        out = {"fits": len(F), "fit_minutes_median": float(np.median([f["fit_seconds"] for f in F]) / 60)}
        for part in ("in", "sim", "ext"):
            r_part = {}
            for rep in reps:
                ks = sorted(k for k in Z[rep] if carl_parts(k[1]) == part)
                if not ks:
                    continue
                r = {"zero": boot2({k: Z[rep][k] for k in ks}), "upright": boot2({k: U[rep][k] for k in ks}),
                     "solved": boot2({k: V[rep][k] for k in ks})}
                ko = [k for k in ks if k in oracle]
                if ko:
                    r["vs_oracle"] = boot2({k: Z[rep][k] - oracle[k] for k in ko})
                if rep != "natc" and "natc" in Z:
                    kk = [k for k in ks if k in Z["natc"]]
                    r["vs_natc"] = boot2({k: Z[rep][k] - Z["natc"][k] for k in kk})
                kt = [k for k in ks if k in S.get(rep, {})]
                if kt:
                    r["tg"] = boot2({k: Z[rep][k] - S[rep][k] for k in kt})
                    r["seen"] = boot2({k: S[rep][k] for k in kt})
                    r["tg_contexts"] = len(kt)
                    kind = "seen_in" if part == "in" else "seen_outpool"
                    per_fit = [float(np.mean([Z[rep][(f["seed"], int(u))] - np.mean(v["cost"]) for u, v in f["eval"].items()
                                              if carl_parts(int(u)) == part]))
                               for f in F if f["kind"] == kind and f["rep"] == rep]
                    r["tg_by_fit"] = {"values": per_fit, "min": float(np.min(per_fit)), "max": float(np.max(per_fit))}
                r_part[rep] = r
            con = {}
            for a, b in CARL_CONTRASTS:
                if a in Z and b in Z:
                    ks = sorted(k for k in Z[a] if carl_parts(k[1]) == part and k in Z[b])
                    if ks:
                        con[f"{a}_vs_{b}"] = boot2({k: Z[a][k] - Z[b][k] for k in ks})
            r_part["contrasts"] = con
            out[part] = r_part
        res[learner] = out
    res["oracle"] = {}
    for part in ("in", "sim", "ext"):
        ks = sorted(k for k in oracle if carl_parts(k[1]) == part)
        if ks:
            res["oracle"][part] = {"cost": boot2({k: oracle[k] for k in ks})}
    return res


def carl_macros(res: dict, cfg, extra: dict) -> dict:
    name = {"natc": "Natc", "nat": "Nat", "rawc": "Rawc", "rawi": "Rawi", "raw": "Raw"}
    L = {"trees": "Tr", "nn": "Nn"}
    P = {"in": "In", "sim": "Sim", "ext": "Ext"}
    m = {}

    def put(key, st, signed):
        m[key] = fmt(st["mean"], signed)
        m[key + "Lo"] = fmt(st["lo"], signed)
        m[key + "Hi"] = fmt(st["hi"], signed)

    for learner, lt in L.items():
        if learner not in res:
            continue
        r_l = res[learner]
        for part, pt in P.items():
            for rep, r in r_l.get(part, {}).items():
                if rep == "contrasts":
                    for key, st in r.items():
                        a, b = key.split("_vs_")
                        put(f"Carl{lt}{pt}{name[a]}Vs{name[b]}", st, True)
                        m[f"Carl{lt}{pt}{name[a]}Vs{name[b]}SeedMin"] = fmt(min(st["per_seed"]), True)
                        m[f"Carl{lt}{pt}{name[a]}Vs{name[b]}SeedMax"] = fmt(max(st["per_seed"]), True)
                    continue
                R = name[rep]
                put(f"Carl{lt}{pt}Cost{R}", r["zero"], False)
                m[f"Carl{lt}{pt}Up{R}"] = f"{r['upright']['mean']:.3f}"
                m[f"Carl{lt}{pt}Solved{R}"] = f"{r['solved']['mean']:.3f}"
                if "vs_oracle" in r:
                    put(f"Carl{lt}{pt}{R}VsOracle", r["vs_oracle"], True)
                if "vs_natc" in r:  # equal to the contrast of the same name where both exist
                    put(f"Carl{lt}{pt}{R}VsNatc", r["vs_natc"], True)
                if "tg" in r:
                    put(f"Carl{lt}{pt}Tg{R}", r["tg"], True)
                    put(f"Carl{lt}{pt}Seen{R}", r["seen"], False)
                    m[f"Carl{lt}{pt}Tg{R}FitMin"] = fmt(r["tg_by_fit"]["min"], True)
                    m[f"Carl{lt}{pt}Tg{R}FitMax"] = fmt(r["tg_by_fit"]["max"], True)
                    # the gap's share of the excess over nat+ctx, per seed (review round 5, R6), where every
                    # seed's excess is clearly positive
                    if "vs_natc" in r and min(r["vs_natc"]["per_seed"]) > 0.05:
                        sh = [t / e for t, e in zip(r["tg"]["per_seed"], r["vs_natc"]["per_seed"])]
                        m[f"Carl{lt}{pt}TgShare{R}SeedMin"] = f"{min(sh):.2f}"
                        m[f"Carl{lt}{pt}TgShare{R}SeedMax"] = f"{max(sh):.2f}"
        m[f"CarlFits{lt}"] = str(r_l["fits"])
        m[f"CarlFitMinutes{lt}"] = f"{r_l['fit_minutes_median']:.1f}"
    for part, pt in P.items():
        if part in res.get("oracle", {}):
            m[f"CarlOracle{pt}Cost"] = fmt(res["oracle"][part]["cost"]["mean"], False)
    for k, v in extra.items():
        m[k] = v
    n = cfg["states"]
    from fractions import Fraction

    def frac(x):  # 2/3 rather than 0.667
        f = Fraction(x).limit_denominator(10)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

    m.update({"CarlBoxLo": frac(cfg["box"][0]), "CarlBoxHi": frac(cfg["box"][1]),
              "CarlTrainContexts": str(cfg["n_train"]), "CarlTestContexts": str(cfg["n_test"]),
              "CarlSimContexts": str(cfg["n_sim"]), "CarlExtContexts": str(cfg["n_ext"]),
              "CarlOutContexts": str(cfg["n_sim"] + cfg["n_ext"]), "CarlGroup": str(cfg["group"]),
              "CarlSeeds": str(len(cfg["seeds"])), "CarlStates": f"{n:,}".replace(",", "{,}"),
              "CarlStatesTotal": f"{n * cfg['n_train']:,}".replace(",", "{,}"),
              "CarlIters": str(cfg["iters"]), "CarlRounds": str(cfg["rounds"]), "CarlGamma": f"{cfg['gamma']}",
              "CarlHorizon": str(cfg["horizon"]), "CarlEpisodes": str(cfg["episodes"]),
              "CarlOutFactorLo": f"{cfg['out_factor'][0]:g}", "CarlOutFactorHi": f"{cfg['out_factor'][1]:g}"})
    return m


def carl_figure(res: dict, out: Path, png: Path | None):
    """Rows: trees, neural. Columns: in range; context out of range with the problem in range (sim); every feature
    and the groups out of range (ext). Per representation: transfer gap (blue; in range against seen comparators
    with each test context 1/72 of the data, out of range against the pooled comparator, 1/32 each) and zero-shot
    cost minus nat+ctx (orange)."""
    sys.path.insert(0, str(REPO / "paper"))
    import figstyle as fs  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402

    TG_COL, VS_COL = fs.CATEGORICAL[0], fs.CATEGORICAL[1]
    learners = [l for l in ("trees", "nn") if l in res]
    fig, axs = plt.subplots(len(learners), 3, figsize=(fs.FULL, 2.15 * len(learners) + 0.35), squeeze=False)
    titles = {"in": "in range", "sim": "context out, problem in range", "ext": "every feature out"}
    letters = iter("abcdef")
    for i, learner in enumerate(learners):
        for j, part in enumerate(("in", "sim", "ext")):
            ax = axs[i, j]
            block = res[learner].get(part, {})
            reps = [r for r in CARL_REPS if r in block]
            for key, col, mk, off, lab in (("tg", TG_COL, "o", -0.14, "transfer gap (zero-shot $-$ seen)"),
                                           ("vs_natc", VS_COL, "D", 0.14, "zero-shot cost $-$ nat+ctx")):
                first = True
                for x, rep in enumerate(reps):
                    if key not in block[rep]:
                        continue
                    ci = block[rep][key]
                    ax.plot([x + off, x + off], [ci["lo"], ci["hi"]], color=col, lw=1.2, solid_capstyle="round")
                    ax.plot(x + off, ci["mean"], mk, color=col, ms=3.6, mec=fs.SURFACE, mew=0.6,
                            label=lab if first and i == 0 and j == 0 else None)
                    first = False
            ax.axhline(0, color=fs.INK2, lw=0.6)
            ax.set_xticks(np.arange(len(reps)))
            ax.set_xticklabels([CARL_LABEL[r] for r in reps], rotation=35, ha="right")
            ax.set_xlim(-0.5, len(reps) - 0.5)
            ax.grid(axis="x", visible=False)
            fs.panel_title(ax, next(letters), titles[part])
            if j == 0:
                ax.set_ylabel(f"{'trees' if learner == 'trees' else 'neural'}: cost per step")
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), columnspacing=1.5)
    fig.tight_layout(w_pad=0.8, h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if png:
        fig.savefig(png, dpi=200)
    plt.close(fig)


def carl_extra(cfg, check: dict, pilots: list) -> dict:
    """Macros of the settings that pilots and checks fixed (see CARL_DOC): the groups' ranges, the default context's
    groups, the grid-value-iteration checks and the nat+ctx pilots."""
    m = {}
    C0 = Ctx([GYM["g"]], [GYM["l"]], [GYM["m"]])
    m.update({"CarlDefaultTau": f"{C0.tau0[0]:.3f}", "CarlDefaultDelta": f"{C0.delta[0]:.3f}",
              "CarlDefaultUbar": f"{C0.ubar[0]:.2f}", "CarlDefaultWmax": f"{C0.wmax[0]:.2f}"})
    rng_ = {}
    for seed in cfg["seeds"]:
        for u, (g, l, mm, st) in carl_contexts(seed, cfg).items():
            C = Ctx([g], [l], [mm])
            part = "Train" if st in ("train", "test") else ("Sim" if st.startswith("sim") else "Ext")
            for q, v in (("Tau", C.tau0[0]), ("Delta", C.delta[0]), ("Ubar", C.ubar[0]), ("Wmax", C.wmax[0]),
                         ("G", g), ("L", l), ("M", mm)):
                rng_.setdefault((part, q), []).append(float(v))
    for (part, q), v in rng_.items():
        m[f"Carl{part}{q}Min"] = f"{min(v):.3g}"
        m[f"Carl{part}{q}Max"] = f"{max(v):.3g}"
    for k, v in check.get("macros", {}).items():
        m[k] = v
    for k, v in carl_pilot_macros(pilots, cfg).items():
        m[k] = v
    return m


def carl_pilot_macros(pilots: list, cfg) -> dict:
    """nat+ctx pilots on the chosen box and on [0.5, 2]: share of in-range test episodes solved, against the oracle."""
    m = {}
    for learner, lt in (("trees", "Tr"), ("nn", "Nn")):
        for box, B in ((cfg["box"], ""), ([0.5, 2.0], "Wide")):
            rows = [p for p in pilots if p["learner"] == learner and p["rep"] == "natc"
                    and np.allclose(p["box"], box) and p["states"] == cfg["states"]
                    and (learner == "trees" or p["nn"] == cfg["nn"])]
            if not rows:
                continue
            m[f"CarlPilot{lt}{B}Solved"] = f"{np.mean([p['in']['solved'] for p in rows]):.2f}"
            m[f"CarlPilot{lt}{B}OracleSolved"] = f"{np.mean([p['in']['oracle_solved'] for p in rows]):.2f}"
            m[f"CarlPilot{lt}{B}Cost"] = f"{np.mean([p['in']['cost'] for p in rows]):.3f}"
            m[f"CarlPilot{lt}{B}OracleCost"] = f"{np.mean([p['in']['oracle'] for p in rows]):.3f}"
            m[f"CarlPilot{lt}{B}Seeds"] = str(len(rows))
    # neural variants on the chosen box (nat+ctx, same seeds): the largest reduction of the mean in-range cost
    base = [p for p in pilots if p["learner"] == "nn" and p["rep"] == "natc" and np.allclose(p["box"], cfg["box"])
            and p["nn"] == cfg["nn"]]
    var = {}
    for p in pilots:
        if p["learner"] == "nn" and p["rep"] == "natc" and np.allclose(p["box"], cfg["box"]) and p["nn"] != cfg["nn"]:
            var.setdefault(json.dumps(p["nn"], sort_keys=True), []).append(p["in"]["cost"])
    if base and var:
        b = np.mean([p["in"]["cost"] for p in base])
        m["CarlPilotNnVariants"] = str(len(var))
        m["CarlPilotNnBestGain"] = f"{max(b - np.mean(v) for v in var.values()):.3f}"
    return m


def cmd_carl_report(args, cfg):
    out_json, out_tex, out_fig = CARL_JSON, CARL_TEX, CARL_FIG
    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        out_json, out_tex, out_fig = d / CARL_JSON.name, d / CARL_TEX.name, d / CARL_FIG.name
    cache = Path(args.cache)
    if args.from_json:
        stored = json.loads(CARL_JSON.read_text())
        fits, cfg, oracle_list = stored["fits"], stored["settings"], stored["oracle_evals"]
        check, pilots = stored.get("check", {}), stored.get("pilots", [])
    else:
        fits = [json.loads(p.read_text()) for p in sorted(cache.glob("carl_*.json"))
                if not p.name.startswith(("carl_oracle", "carl_config"))]
        oracle_list = [json.loads(p.read_text()) for p in sorted(cache.glob("carl_oracle_s*.json"))]
        cp = REPO / "results" / "transfer" / "carl_check.json"
        check = json.loads(cp.read_text()) if cp.exists() else {}
        pilots = [json.loads(x) for x in CARL_PILOTS.read_text().splitlines() if x.strip()] if CARL_PILOTS.exists() else []
    oracle = {(o["seed"], int(u)): float(np.mean(v["cost"])) for o in oracle_list for u, v in o["eval"].items()}
    planned = {f["key"] for learner in ("trees", "nn") for f in carl_plan(cfg, learner)}
    missing = sorted(planned - {f["key"] for f in fits})
    if missing:
        print(f"warning: {len(missing)} planned fits missing, e.g. {missing[:6]}")
    res = carl_summarise(fits, oracle, cfg)
    # oracle solved share per part
    for part in ("in", "sim", "ext"):
        v = [float(np.mean(np.array(e["upright"]) > 0.9)) for o in oracle_list for u, e in o["eval"].items()
             if carl_parts(int(u)) == part]
        if v and part in res["oracle"]:
            res["oracle"][part]["solved"] = float(np.mean(v))
    extra = carl_extra(cfg, check, pilots)
    for part, pt in (("in", "In"), ("sim", "Sim"), ("ext", "Ext")):
        if "solved" in res["oracle"].get(part, {}):
            extra[f"CarlOracle{pt}Solved"] = f"{res['oracle'][part]['solved']:.3f}"
    contexts = []
    for seed in cfg["seeds"]:
        for u, (g, l, mm, st) in carl_contexts(seed, cfg).items():
            C = Ctx([g], [l], [mm])
            contexts.append(dict(seed=seed, uid=u, set=st, g=g, l=l, m=mm, tau0=float(C.tau0[0]), delta=float(C.delta[0]),
                                 ubar=float(C.ubar[0]), wmax=float(C.wmax[0])))
    doc = {"description": "CARL's pendulum (gym Pendulum-v1 with g, l, m as context; dt, max_speed, max_torque fixed): "
                          "zero-shot transfer across contexts (review round 5, R4; exploratory, post hoc). Costs are "
                          "gym's cost per step (mean over 200 steps from gym's start distribution). tg = zero-shot - "
                          "seen per test context; vs_natc = zero-shot - nat+ctx zero-shot; vs_oracle = zero-shot - "
                          "grid value iteration. Intervals: two-level bootstrap (seeds, then contexts), 10,000 "
                          "resamples. Generated by experiments/pendulum/pendulum.py carl-report.",
           "doc": CARL_DOC, "settings": cfg, "missing_fits": missing, **res, "macros_extra": extra,
           "check": check, "pilots": pilots, "contexts": contexts, "oracle_evals": oracle_list, "fits": fits}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=1))
    m = carl_macros(res, cfg, extra)
    lines = ["% generated by experiments/pendulum/pendulum.py carl-report (exploratory) -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    out_tex.write_text("\n".join(lines) + "\n")
    carl_print(res)
    carl_figure(res, out_fig, Path(args.png) if args.png else None)
    print(f"wrote {out_json}, {out_tex}, {out_fig}")


def carl_print(res):
    for learner in ("trees", "nn"):
        if learner not in res:
            continue
        for part in ("in", "sim", "ext"):
            print(f"== {learner}, {part} (oracle {res['oracle'].get(part, {}).get('cost', {}).get('mean', float('nan')):.3f})")
            for rep, r in res[learner].get(part, {}).items():
                if rep == "contrasts":
                    for k, st in r.items():
                        print(f"  {k}: {st['mean']:+.3f} [{st['lo']:+.3f}, {st['hi']:+.3f}]")
                    continue
                vs = r.get("vs_natc", {"mean": 0.0, "lo": 0.0, "hi": 0.0})
                tg = r.get("tg", {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")})
                print(f"  {rep:5s} zero {r['zero']['mean']:.3f} [{r['zero']['lo']:.3f}, {r['zero']['hi']:.3f}]  vs natc "
                      f"{vs['mean']:+.3f} [{vs['lo']:+.3f}, {vs['hi']:+.3f}]  TG {tg['mean']:+.3f} [{tg['lo']:+.3f}, "
                      f"{tg['hi']:+.3f}]  upright {r['upright']['mean']:.3f} solved {r['solved']['mean']:.3f}")


def _carl_main(argv):
    ap = argparse.ArgumentParser(description="CARL's pendulum (review round 5, R4; exploratory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("carl-check")
    p.add_argument("--threads", default=_THREADS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--taus", nargs="+", default=[0.35, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83])
    p.add_argument("--ubars", nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2.0])
    p.add_argument("--gammas", nargs="+", type=float, default=[0.98])
    p.add_argument("--vi-iters", dest="vi_iters", nargs="+", type=int, default=[50])
    p.add_argument("--map", action="store_true")
    p.add_argument("--boxes", action="store_true")
    for name in ("carl-run", "carl-pilot", "carl-oracle", "carl-report"):
        p = sub.add_parser(name)
        p.add_argument("--threads", default=_THREADS)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--cache", default=str(CARL_CACHE))
        p.add_argument("--seeds", type=int, nargs="+", default=None)
        for k in ("states", "iters", "rounds", "episodes", "n_train", "n_test", "n_sim", "n_ext"):
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int, default=None)
        p.add_argument("--gamma", type=float, default=None)
        p.add_argument("--box", nargs=2, default=None, help="training factors of the default for g, l, m")
        p.add_argument("--out-factor", dest="out_factor", nargs=2, default=None)
        p.add_argument("--learner", default="trees", choices=["trees", "nn"])
        p.add_argument("--device", default="cpu", help="torch device(s) for the neural learner, e.g. cuda:0,cuda:1")
        p.add_argument("--reps", nargs="+", default=None, choices=CARL_REPS)
        p.add_argument("--kinds", nargs="+", default=None, choices=["zero", "seen_in", "seen_outpool"])
        p.add_argument("--only", nargs="+", default=None)
        for k, t in (("hidden", int), ("layers", int), ("act", str), ("lr", float), ("batch", int), ("epochs", int)):
            p.add_argument(f"--nn-{k}", dest=f"nn_{k}", type=t, default=None)
        if name == "carl-report":
            p.add_argument("--from-json", action="store_true")
            p.add_argument("--png", default=None)
            p.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)
    if int(args.workers) * int(args.threads) > 16:
        raise SystemExit("workers x threads must be at most 16")
    cfg = carl_cfg_from_args(args) if args.cmd != "carl-check" else json.loads(json.dumps(CARL_CFG))
    if args.cmd == "carl-report" and not args.from_json and (Path(args.cache) / "carl_config.json").exists():
        cfg = json.loads((Path(args.cache) / "carl_config.json").read_text())
    {"carl-check": cmd_carl_check, "carl-run": cmd_carl_run, "carl-pilot": cmd_carl_pilot,
     "carl-oracle": cmd_carl_oracle, "carl-report": cmd_carl_report}[args.cmd](args, cfg)


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("carl-"):
        return _carl_main(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "pilot", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--threads", default=_THREADS, help="threads per worker (LightGBM, OpenMP, BLAS)")
        p.add_argument("--cache", default=str(CACHE), help="directory of per-fit evaluations")
        for k in ("states", "iters", "rounds", "episodes", "horizon", "n_train", "n_test", "n_out", "n_sweep"):
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int, default=None)
        for k in ("dt_ratio", "u_ratio", "gamma"):
            p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=float, default=None)
        p.add_argument("--seeds", type=int, nargs="+", default=None)
        p.add_argument("--reps", nargs="+", default=None, choices=REPS)
        p.add_argument("--eps-reps", dest="eps_reps", nargs="+", default=None, choices=REPS)
        if name == "run":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--only", nargs="+", default=None, help="run only fits whose key contains one of these")
            p.add_argument("--plan", default="main", choices=["main", "x4", "sweep", "extra", "nn"],
                           help="main: the original design; x4, sweep (extra = both): tree follow-ups; nn: neural")
            p.add_argument("--device", default="cpu", help="torch device for --plan nn, e.g. cuda:0")
        if name == "pilot":
            p.add_argument("--verbose", action="store_true")
            p.add_argument("--plain", action="store_true", help="regress the cost-to-go itself (no advantage split)")
            p.add_argument("--learner", default="trees", choices=["trees", "nn"])
            p.add_argument("--device", default="cpu", help="torch device for --learner nn, e.g. cuda:0")
            for k, t in (("hidden", int), ("layers", int), ("act", str), ("lr", float), ("batch", int),
                         ("epochs", int), ("weight_decay", float)):
                p.add_argument(f"--nn-{k.replace('_', '-')}", dest=f"nn_{k}", type=t, default=None)
        if name == "report":
            p.add_argument("--from-json", action="store_true", help="rebuild macros and figure from the stored JSON")
            p.add_argument("--png", default=None, help="also write a PNG preview here")
            p.add_argument("--outdir", default=None, help="write the JSON, macros and figure here instead")
    args = ap.parse_args()
    cfg = dict(CFG)
    for k in list(cfg):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg["nn"] = dict(cfg["nn"], **{k[3:]: getattr(args, k) for k in vars(args) if k.startswith("nn_") and
                                   k not in ("nn_reps", "nn_seen_reps") and getattr(args, k) is not None})
    if args.cmd == "run" and int(args.workers) * int(args.threads) > 16:
        raise SystemExit("workers x threads must be at most 16")
    {"check": cmd_check, "pilot": cmd_pilot, "run": cmd_run, "report": cmd_report}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()

"""PPO pieces: advantage estimates and the saved model's greedy action."""

import numpy as np

from invex.agents.fqi_adv import load_model
from invex.agents.ppo import PPOModel, gae, mlp_forward


def test_gae_matches_hand_computation():
    R = np.array([[-1.0, -2.0, -3.0], [-1.0, -5.0, 0.0]], np.float32)
    V = np.array([[-6.0, -4.0, -2.0], [-3.0, -4.0, 0.0]], np.float32)
    alive = np.array([[1, 1, 1], [1, 1, 0]], bool)  # episode 2 is done after its second decision
    adv, ret = gae(R, V, alive, lam=0.5)
    # episode 1: deltas -1+(-4)-(-6)=1, -2+(-2)-(-4)=0, -3-(-2)=-1; A2=-1, A1=0+0.5*-1=-0.5, A0=1+0.5*-0.5=0.75
    assert np.allclose(adv[0], [0.75, -0.5, -1.0])
    # episode 2: deltas -1+(-4)-(-3)=-2, -5-(-4)=-1 (no value after it is done); A1=-1, A0=-2+0.5*-1=-2.5
    assert np.allclose(adv[1], [-2.5, -1.0, 0.0])
    assert np.allclose(ret, adv + V)
    # with lambda = 1 and no discounting, the return target is the realised cost to go
    adv1, ret1 = gae(R, V, alive, lam=1.0)
    assert np.allclose(ret1[0], [-6.0, -5.0, -3.0]) and np.allclose(ret1[1, :2], [-6.0, -5.0])


def test_saved_model_picks_most_likely_action(tmp_path):
    rng = np.random.default_rng(0)
    d, K = 4, 3
    pi = [(rng.normal(size=(d + K, 8)).astype(np.float32), np.zeros(8, np.float32)),
          (rng.normal(size=(8, 5)).astype(np.float32), np.zeros(5, np.float32))]
    m = PPOModel("R3tick", K, 1.0, np.zeros(d), np.ones(d), pi, pi)
    path = tmp_path / "m.pkl"
    m.save(path)
    m2 = load_model(path)
    X = np.expm1(np.abs(rng.normal(size=(6, d))))  # inputs pass through slog, so use positive values
    lg = mlp_forward(pi, m2.inputs(1, X))
    assert np.allclose(m2.q_values(1, X), -lg, atol=1e-5)
    assert (np.argmin(m2.q_values(1, X), axis=1) == lg.argmax(axis=1)).all()

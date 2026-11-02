"""Coverage and correctness of the inference helpers (invex.eval.stats)."""

import numpy as np

from invex.eval import stats as st


def test_cluster_bootstrap_covers_true_mean():
    rng = np.random.default_rng(0)
    hits = 0
    for r in range(200):
        d = rng.normal(0.3, 1.0, 120)
        ci = st.cluster_bootstrap(d, strata=np.arange(120) % 3, n_boot=1000, seed=r)
        hits += ci["lo"] < 0.3 < ci["hi"]
    assert 0.90 <= hits / 200 <= 0.99


def test_pigeonhole_bootstrap_widens_with_day_effects():
    rng = np.random.default_rng(1)
    A, D = 200, 6
    asset = rng.normal(0, 1, A)[:, None]
    day = rng.normal(0, 1, D)[None, :]
    counts = np.full((A, D), 10.0)
    sums = (0.5 + asset + day) * counts
    two_way = st.pigeonhole_bootstrap(sums, counts, n_boot=1000)
    one_way = st.cluster_bootstrap((sums / counts).mean(axis=1), n_boot=1000)
    # crossed day effects add uncertainty that resampling assets alone misses
    assert (two_way["hi"] - two_way["lo"]) > 1.5 * (one_way["hi"] - one_way["lo"])


def test_tost_and_corrections():
    assert st.tost(-0.01, 0.02, 0.05) and not st.tost(-0.06, 0.02, 0.05)
    p = np.array([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(st.holm(p), [0.04, 0.09, 0.09, 0.2])
    assert np.allclose(st.bh(p), [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def test_cluster_ols_recovers_slope_and_clusters_inflate_se():
    rng = np.random.default_rng(2)
    G, n = 50, 40
    g = np.repeat(np.arange(G), n)
    x = rng.normal(size=G * n) + np.repeat(rng.normal(size=G), n)
    y = 1.0 + 2.0 * x + np.repeat(rng.normal(size=G), n) + rng.normal(size=G * n)
    X = np.column_stack([np.ones_like(x), x])
    r = st.cluster_ols(y, X, g)
    assert abs(r["b"][1] - 2.0) < 0.15
    naive = st.cluster_ols(y, X, np.arange(len(y)))
    assert r["se"][0] > 2 * naive["se"][0]  # the intercept carries the cluster effect


def test_boot_p_matches_interval_inversion():
    from invex.eval.stats import boot_p, cluster_bootstrap

    rng = np.random.default_rng(3)
    d = rng.normal(0.02, 0.05, 300)  # per-cluster effects, mean clearly > 0
    r = cluster_bootstrap(d, n_boot=4000, level=0.90, return_boots=True)
    assert boot_p(r["boots"], "greater") < 0.05 and r["lo"] > 0
    assert boot_p(r["boots"], "less") > 0.95
    # equivalence: within 0.04 but not within 0.02 (the 90% interval decides, as in TOST)
    assert (boot_p(r["boots"], "tost", 0.04) < 0.05) == (-0.04 < r["lo"] and r["hi"] < 0.04)
    assert boot_p(r["boots"], "tost", 0.02) > 0.05

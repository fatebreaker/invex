"""Zero-intelligence market (invex.sim.zi): exact similarity, a clean book, and the full stack on top."""

import numpy as np

from invex import schema as S
from invex.book.derive import derive
from invex.sim.zi import ZIParams, simulate

BASE = ZIParams()


def test_rescaled_asset_gives_rescaled_path():
    ev1, o1, c1 = simulate(BASE, 3600, warmup_s=600, seed=7)
    ev2, o2, c2 = simulate(BASE.scaled(c=2.0, s=10, p=10), 1800, warmup_s=300, seed=7)
    n = min(len(ev1), len(ev2))
    assert n > 10_000
    # same events in the same order: kinds, sides, oids; shares x 10, prices x 10, time / 2
    a, b = ev1[: n - 50], ev2[: n - 50]
    assert np.array_equal(a["kind"], b["kind"]) and np.array_equal(a["side"], b["side"])
    assert np.array_equal(a["oid"], b["oid"])
    assert np.array_equal(b["qty"], a["qty"] * 10) and np.array_equal(b["px"], a["px"] * 10)
    assert np.max(np.abs(b["ts"] * 2 - a["ts"])) <= 2


def test_book_is_clean_and_fifo():
    ev, o, c = simulate(BASE, 3600, warmup_s=600, seed=3)
    d = derive(ev, o, c)
    assert d.qa["unresolved"] == 0 and d.qa["negative"] == 0 and d.qa["crossed"] == 0
    assert d.qa["fifo_bad"] == 0 and d.qa["fifo_ok"] > 1000
    cont = (ev["flags"] & S.F_CONTINUOUS) > 0
    spread = (d.ask_px - d.bid_px)[cont & (d.ask_px > 0) & (d.bid_px > 0)]
    assert spread.min() >= BASE.tick  # never locked or crossed


def test_episodes_and_environment_run_on_zi():
    from invex.env import execution as ex
    from invex.env.episodes import book_episodes
    from invex.eval import policies as P
    from invex.env.units import unit_frame, valid_units
    from invex.sim.tracker import prepare

    ko = {"c_t": 4.68, "c_q": 1.7e-4, "c_p": 0.028}
    ev, o, c = simulate(BASE, 4 * 3600, warmup_s=1800, seed=11)
    book = prepare(ev, o, c)
    eps = book_episodes(book, "zi", "d0", "Z0", o, c, 40, ko)
    assert len(eps) > 10
    uf = unit_frame(eps, ko)
    eps = eps[valid_units(uf)].reset_index(drop=True)
    uf = unit_frame(eps, ko)
    res = P.evaluate_book(book, eps, uf, [P.Immediate(), P.PostThenCross()], 10, 100_000)
    assert (res["filled"] == np.repeat(eps["Q"].to_numpy()[None, :], 2, axis=0).ravel()).all()
    imm = res[res.policy == "immediate"]["cost"].to_numpy() / (eps["Q"] * eps["w_spread"]).to_numpy()
    assert 0.3 < np.mean(imm) < 3.0  # crossing costs about half a spread or more

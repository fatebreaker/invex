"""paper/make_iex_outcomes.py: the registered outcome sentences (prereg/iex_outcomes_v7.md, section 2)."""

import importlib.util

import pytest

from invex.paths import REPO

spec = importlib.util.spec_from_file_location("make_iex_outcomes", REPO / "paper" / "make_iex_outcomes.py")
mio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mio)


def hyp(hid, mean, lo, hi, supported, delta=0.0):
    return {"id": hid, "delta": delta, "supported": supported, "p_holm": 0.01,
            "estimates": {"all": {"mean": mean, "lo": lo, "hi": hi}}}


@pytest.mark.parametrize("lo, hi, supported, branch, words", [
    (-0.05, 0.01, True, "supported", "is not worse than one trained on IEX"),
    (-0.20, 0.03, False, "undecided", "is undecided"),
    (0.021, 0.08, False, "worse", "by more than the margin"),
])
def test_i2_branches(lo, hi, supported, branch, words):
    s, b = mio.i2_sentence("On ten later IEX days, registered before they occurred",
                           hyp("I2", (lo + hi) / 2, lo, hi, supported, delta=0.02))
    assert b == branch and words in s and s.startswith("On ten later IEX days")
    assert ("margin 0.020" in s) == (branch != "worse")


def test_family_counts_and_grammar():
    claims = mio.FAMILIES["ppo"][2]
    fam = {"P1": hyp("P1", -0.5, -0.6, -0.4, True), "P2": hyp("P2", -0.1, -0.2, 0.03, False, 0.02),
           "P3": hyp("P3", 0.09, 0.01, 0.2, True)}
    s = mio.family_sentence("With PPO", fam, claims)
    assert s.startswith("With PPO, two of three registered tests hold: beating crossing immediately (P1")
    assert "(P2: $-0.100$) is not shown." in s
    fam = {k: dict(v, supported=True) for k, v in fam.items()}
    assert mio.family_sentence("With PPO", fam, claims).startswith("With PPO, all three registered tests hold")
    fam = {k: dict(v, supported=False) for k, v in fam.items()}
    assert "none of the three registered tests holds" in mio.family_sentence("With PPO", fam, claims)
    one = {"P1": dict(fam["P1"], supported=True), "P2": fam["P2"], "P3": fam["P3"]}
    assert "one of three registered tests holds" in mio.family_sentence("With PPO", one, claims)
    learner = {"L1": hyp("L1", -0.07, -0.13, -0.01, False), "L2": hyp("L2", -0.03, -0.1, 0.05, False)}
    assert "neither registered test holds" in mio.family_sentence("Of the learner comparisons", learner,
                                                                  mio.FAMILIES["learner"][2])


def test_dry_run_refuses_paper_dir():
    with pytest.raises(SystemExit):
        mio.main(["--eval", "iex_retro", "--prefix", "IexTest", "--out-dir", str(REPO / "paper")])


def test_v8_either_estimand_at_0025(monkeypatch):
    """iex-v8: supported if the Holm p is below 0.025 under either estimand; the estimand is named."""
    fams = {"fq": {"I2": dict(hyp("I2", -0.07, -0.22, 0.024, False, delta=0.02), p_holm=0.06),
                   "I1": dict(hyp("I1", -0.4, -0.5, -0.3, True), p_holm=0.03)}}
    epi = {"I2": dict(hyp("I2", -0.003, -0.007, 0.001, True, delta=0.02), p_holm=0.001),
           "I1": dict(hyp("I1", -0.18, -0.19, -0.17, True), p_holm=0.03)}
    monkeypatch.setattr(mio, "load", lambda ev, fam: epi)
    mio.apply_v8(fams, "iex_test")
    i2, i1 = fams["fq"]["I2"], fams["fq"]["I1"]
    assert i2["supported"] and i2["by"] == ["b"] and i2["registered_supported"] is False
    assert not i1["supported"] and i1["registered_supported"] is True  # 0.03 holds at 0.05 but not at 0.025
    s, branch = mio.i2_sentence("On ten later IEX days", i2)
    assert branch == "supported" and s.startswith("On ten later IEX days, with episodes weighted equally, the")
    assert "with stocks weighted equally, as first registered, this is undecided ($-0.070$" in s


def test_v8_adverse_estimate_sign_note_and_first_registered_count():
    """A supporting estimate above zero is reported as costing more (by less than the margin); opposite signs of the
    two estimands are stated; a family names its first-registered count when it differs from the iex-v8 decision."""
    p4 = dict(hyp("P4", -0.127, -0.554, 0.033, False, delta=0.02), by=["b"], supported=True,
              episode=hyp("P4", 0.011, 0.008, 0.015, True, delta=0.02))
    s, branch = mio.i2_sentence("Against PPO trained on IEX by the same recipe", p4, hid="P4",
                                who="the Nasdaq-trained PPO policy", ref="that policy", first_registered=False)
    assert branch == "supported" and "costs more than that policy, by less than the margin" in s
    assert "not worse" not in s and "as first registered" not in s and "this is undecided" in s
    fq = {"I1": dict(hyp("I1", -0.139, -0.33, 0.31, True), by=["b"], registered_supported=False,
                     episode=hyp("I1", -0.187, -0.196, -0.179, True)),
          "I3": dict(hyp("I3", -0.115, -0.54, 0.05, True), by=["b"], registered_supported=False,
                     episode=hyp("I3", 0.042, 0.037, 0.048, True))}
    claims = {"I1": "beating crossing immediately", "I3": "raw units losing"}
    s = mio.family_sentence("With fitted Q-iteration", fq, claims)
    assert "(I3: $-0.115$ by stock, $+0.042$ by episode; by episode only; the two estimates differ in sign)" in s
    assert "(I1: $-0.139$ by stock, $-0.187$ by episode; by episode only)" in s
    assert s.endswith("; as first registered (stocks weighted equally, level 0.05), neither of the two tests holds.")

"""The confirmatory script's population rule: competitive books, wide books, or every episode."""

import importlib.util

import numpy as np
import pandas as pd

from invex.paths import REPO

_spec = importlib.util.spec_from_file_location("confirm", REPO / "experiments" / "confirm.py")
confirm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(confirm)


def test_population_splits_episodes_by_onetick_share():
    eps = pd.DataFrame({"key": list("abcde"), "w_onetick": [0.0, 0.49, 0.5, 1.0, np.nan]})
    assert confirm.population(eps, {}) is None
    assert confirm.population(eps, {"min_onetick": 0.5}) == {"c", "d"}  # the iex-v1 population, as before
    assert confirm.population(eps, {"max_onetick": 0.5}) == {"a", "b"}  # its complement among defined shares
    assert confirm.population(eps, {"min_onetick": 0.4, "max_onetick": 0.6}) == {"b", "c"}

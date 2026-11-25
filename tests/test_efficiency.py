import numpy as np

from efficiency import models_to_target


def test_models_to_target_counts_from_one_and_reports_misses():
    best = np.array([0.5, 0.6, 0.6, 0.7])
    assert models_to_target(best, 0.6) == 2
    assert models_to_target(best, 0.45) == 1
    assert models_to_target(best, 0.71) is None

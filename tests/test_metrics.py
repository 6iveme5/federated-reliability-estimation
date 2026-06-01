import numpy as np

from fedrel_journal.metrics import minmax01, risk_at_fraction


def test_minmax01_constant_returns_midpoint():
    values = minmax01(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(values, 0.5)


def test_risk_at_fraction_selects_lowest_reliability():
    errors = np.array([0, 1, 1, 0])
    reliability = np.array([0.9, 0.1, 0.2, 0.8])
    assert risk_at_fraction(errors, reliability, fraction=0.5) == 1.0


def test_risk_at_fraction_selects_at_least_one_sample():
    errors = np.array([0, 1, 0])
    reliability = np.array([0.2, 0.1, 0.3])
    assert risk_at_fraction(errors, reliability, fraction=0.01) == 1.0

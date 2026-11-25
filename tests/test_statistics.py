import numpy as np

from pm_forecasting.statistics import diebold_mariano


def test_dm_detects_persistently_lower_loss():
    rng = np.random.default_rng(42)
    baseline = rng.normal(4.0, 0.4, size=400)
    candidate = baseline - 0.5 + rng.normal(0, 0.05, size=400)
    statistic, p_value, difference = diebold_mariano(candidate, baseline, horizon=24)
    assert statistic < 0
    assert p_value < 0.01
    assert difference < 0


def test_dm_returns_finite_result_for_equal_mean_noisy_losses():
    rng = np.random.default_rng(123)
    baseline = rng.normal(2.0, 0.5, size=300)
    candidate = baseline + rng.normal(0, 0.2, size=300)
    statistic, p_value, difference = diebold_mariano(candidate, baseline, horizon=6)
    assert np.isfinite([statistic, p_value, difference]).all()
    assert 0 <= p_value <= 1


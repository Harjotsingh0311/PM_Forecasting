import numpy as np

from pm_forecasting.analysis import _nested_f_test, _residualize


def test_weather_residualization_removes_linear_control_signal():
    rng = np.random.default_rng(7)
    controls = rng.normal(size=(300, 3))
    values = controls @ np.array([2.0, -1.5, 0.5]) + rng.normal(0, 0.05, 300)
    valid = np.ones(300, dtype=bool)
    residuals = _residualize(values, controls, valid)
    before = np.abs(np.corrcoef(values, controls[:, 0])[0, 1])
    after = np.abs(np.corrcoef(residuals, controls[:, 0])[0, 1])
    assert after < before / 4
    assert np.var(residuals) < np.var(values) / 100


def test_nested_predictive_test_detects_added_history():
    rng = np.random.default_rng(9)
    n = 800
    other = rng.normal(size=n)
    response = np.zeros(n)
    for index in range(1, n):
        response[index] = 0.25 * response[index - 1] + 0.8 * other[index - 1] + rng.normal(0, 0.3)
    statistic, p_value, observations, partial_r2 = _nested_f_test(
        response, response, other, controls=None, max_lag=4
    )
    assert statistic > 1
    assert p_value < 0.01
    assert observations > 700
    assert partial_r2 > 0

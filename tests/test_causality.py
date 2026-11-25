import numpy as np
import pandas as pd

from pm_forecasting.data import _causal_quality_flags


def test_quality_flags_do_not_change_when_future_is_replaced():
    rng = np.random.default_rng(42)
    values = pd.Series(rng.lognormal(2.0, 0.4, size=400))
    cutoff = 250
    altered = values.copy()
    altered.iloc[cutoff + 1 :] = rng.normal(1000, 100, size=len(values) - cutoff - 1)
    original_flags = _causal_quality_flags(values)
    altered_flags = _causal_quality_flags(altered)
    for original, changed in zip(original_flags, altered_flags, strict=True):
        np.testing.assert_array_equal(original.iloc[: cutoff + 1], changed.iloc[: cutoff + 1])


def test_short_forward_fill_is_past_only():
    values = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 7.0])
    result = values.ffill(limit=3)
    np.testing.assert_allclose(result.iloc[:4], [1, 1, 1, 1])
    assert np.isnan(result.iloc[4])


from types import SimpleNamespace

import numpy as np

from pm_forecasting.classical import build_classical_features


def test_classical_feature_shape_and_pair_order():
    rng = np.random.default_rng(7)
    data = SimpleNamespace(
        features=rng.normal(size=(240, 3, 17)).astype(np.float32),
        stations=["A", "B", "C"],
    )
    features, origins, stations = build_classical_features(data, np.array([168, 170]))
    # 6 pollutant lags x 2 + 2 rolling windows x (mean/std) x 2
    # + 13 weather/calendar values + 3 station indicators.
    assert features.shape == (6, 36)
    np.testing.assert_array_equal(origins, [168, 168, 168, 170, 170, 170])
    np.testing.assert_array_equal(stations, [0, 1, 2, 0, 1, 2])
    assert np.isfinite(features).all()


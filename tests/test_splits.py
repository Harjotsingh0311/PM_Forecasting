import numpy as np
import pandas as pd

from pm_forecasting.config import load_config
from pm_forecasting.splits import assert_no_boundary_leakage, build_origins


def test_split_boundaries_match_protocol():
    config = load_config()
    timestamps = pd.date_range("2024-05-02 20:00:00", "2025-12-11 07:00:00", freq="h")
    origins = build_origins(timestamps, config["splits"], 168, 24)
    assert len(timestamps) == 14_100
    assert len(origins.test) == 1_689
    assert len(origins.val_w) == 985
    assert len(origins.val_r) == 697
    assert np.all(origins.train_a + 23 < origins.val_w[0])
    assert np.all(origins.train_b + 23 < origins.val_r[0])
    assert_no_boundary_leakage(origins, timestamps, config["splits"], 168, 24)


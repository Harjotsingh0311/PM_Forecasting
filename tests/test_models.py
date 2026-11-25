from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pm_forecasting.models import make_model


@pytest.mark.parametrize(
    "name",
    ["cnn", "lstm", "tcn", "ebamtn", "cpmtcn", "cpmtcn_no_exchange", "cpmtcn_soft"],
)
def test_model_output_shapes(name):
    data = SimpleNamespace(
        features=np.zeros((200, 3, 17), dtype=np.float32),
        stations=["A", "B", "C"],
        target_log_mean=np.zeros((3, 2), dtype=np.float32),
        target_log_std=np.ones((3, 2), dtype=np.float32),
        target_raw_std=np.ones((3, 2), dtype=np.float32),
    )
    config = {
        "data": {"horizon": 24},
        "model": {"channels": 8, "hidden_size": 16, "station_embedding": 4, "dropout": 0.0},
    }
    model = make_model(name, data, config)
    x = torch.randn(2, 168, 17)
    cp = torch.randn(2, 168, 2)
    station = torch.tensor([0, 2])
    prediction = model(x, station, cp)
    assert prediction.shape == (2, 24, 2)
    assert torch.isfinite(prediction).all()
    if name == "cpmtcn":
        assert len(model.last_gates) == 6
    if name == "cpmtcn_soft":
        assert model.constraint_penalty is not None
        assert model.constraint_penalty.ndim == 0
        assert model.constraint_penalty >= 0


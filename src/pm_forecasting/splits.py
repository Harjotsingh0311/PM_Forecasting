from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OriginSets:
    train_a: np.ndarray
    val_w: np.ndarray
    train_b: np.ndarray
    val_r: np.ndarray
    test: np.ndarray

    @property
    def train(self) -> np.ndarray:
        return np.concatenate([self.train_a, self.train_b])


def _inside(values: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (values >= pd.Timestamp(start)) & (values <= pd.Timestamp(end))


def build_origins(
    timestamps: pd.DatetimeIndex,
    split_config: dict,
    input_length: int = 168,
    horizon: int = 24,
) -> OriginSets:
    """Return target-start indices, enforcing the proposal's boundary rules."""

    def for_block(name: str, require_history_inside: bool) -> np.ndarray:
        start, end = split_config[name]
        candidates = np.flatnonzero(_inside(timestamps, start, end))
        valid: list[int] = []
        for origin in candidates:
            target_end = origin + horizon - 1
            history_start = origin - input_length
            if history_start < 0 or target_end >= len(timestamps):
                continue
            if not _inside(timestamps[[origin, target_end]], start, end).all():
                continue
            if require_history_inside and not _inside(
                timestamps[[history_start, origin - 1]], start, end
            ).all():
                continue
            valid.append(origin)
        return np.asarray(valid, dtype=np.int64)

    return OriginSets(
        train_a=for_block("train_a", True),
        val_w=for_block("val_w", False),
        train_b=for_block("train_b", True),
        val_r=for_block("val_r", False),
        test=for_block("test", False),
    )


def training_time_mask(timestamps: pd.DatetimeIndex, split_config: dict) -> np.ndarray:
    a = _inside(timestamps, *split_config["train_a"])
    b = _inside(timestamps, *split_config["train_b"])
    return a | b


def assert_no_boundary_leakage(
    origins: OriginSets,
    timestamps: pd.DatetimeIndex,
    split_config: dict,
    input_length: int,
    horizon: int,
) -> None:
    for name in ("train_a", "train_b"):
        start, end = split_config[name]
        for origin in getattr(origins, name):
            indices = [origin - input_length, origin + horizon - 1]
            if not _inside(timestamps[indices], start, end).all():
                raise AssertionError(f"{name} window at {origin} crosses a split boundary")


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import OriginSets, build_origins, training_time_mask

BASE_FEATURE_NAMES = [
    "pm25_log_z",
    "pm10_log_z",
    "pm25_unreliable",
    "pm10_unreliable",
    "temperature_z",
    "pressure_z",
    "precipitation_z",
    "rain_z",
    "wind_speed_z",
    "wind_east_z",
    "wind_north_z",
    "wind_unknown",
    "cloud_level_z",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

WIND_ANGLE = {"S": 0, "SV": 45, "V": 90, "JV": 135, "J": 180, "JZ": 225, "Z": 270, "SZ": 315}
CLOUD_LEVEL = {"jasno": 0, "delno oblačno": 1, "pretežno oblačno": 2, "oblačno": 3}


@dataclass
class PreparedData:
    timestamps: np.ndarray
    stations: list[str]
    features: np.ndarray
    cp_features: np.ndarray
    targets_raw: np.ndarray
    targets_norm: np.ndarray
    target_mask: np.ndarray
    target_log_mean: np.ndarray
    target_log_std: np.ndarray
    target_raw_std: np.ndarray
    target_q90: np.ndarray
    coarse_log_mean: np.ndarray
    coarse_log_std: np.ndarray
    origins: OriginSets


def load_station_frames(raw_dir: str | Path) -> tuple[pd.DatetimeIndex, list[str], list[pd.DataFrame]]:
    paths = sorted(Path(raw_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    frames: list[pd.DataFrame] = []
    reference: pd.DatetimeIndex | None = None
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8")
        missing = {"datetime", "PM10", "PM2.5"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame = frame.sort_values("datetime").reset_index(drop=True)
        index = pd.DatetimeIndex(frame["datetime"])
        if index.has_duplicates:
            raise ValueError(f"Duplicate timestamps in {path.name}")
        if reference is None:
            reference = index
        elif not reference.equals(index):
            raise ValueError(f"Timestamp grid in {path.name} differs from the first station")
        frames.append(frame)
    assert reference is not None
    return reference, [path.stem for path in paths], frames


def _causal_quality_flags(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    valid = values.where(values >= 0)
    rolling = valid.rolling(24, min_periods=24)
    stuck = (rolling.max() - rolling.min()).eq(0)
    history = valid.shift(1).rolling(24, min_periods=12)
    median = history.median()
    mad = valid.shift(1).rolling(24, min_periods=12).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    spike = (valid - median).abs().gt(6 * mad.where(mad > 0))
    return stuck.fillna(False), spike.fillna(False)


def _zscore(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[train_mask]
    mean = np.nanmean(selected, axis=0)
    std = np.nanstd(selected, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    result = (values - mean) / std
    return np.nan_to_num(result, nan=0.0).astype(np.float32), mean, std


def _calendar(index: pd.DatetimeIndex, timezone: str, timestamps_are_utc: bool) -> np.ndarray:
    if timestamps_are_utc:
        local = index.tz_localize("UTC").tz_convert(timezone)
    else:
        # The source has a complete naive hourly grid across DST transitions.  When the
        # temperature diagnostic identifies local civil time, preserve the recorded wall
        # clock instead of inventing or dropping an ambiguous transition hour.
        local = index
    hour = local.hour.to_numpy()
    weekday = local.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
        ]
    ).astype(np.float32)


def prepare_data(config: dict) -> PreparedData:
    data_cfg = config["data"]
    timestamps, stations, frames = load_station_frames(data_cfg["raw_dir"])
    cutoff = float(data_cfg.get("max_station_missing_pct", 100.0)) / 100.0
    keep = [
        max(frame["PM2.5"].isna().mean(), frame["PM10"].isna().mean()) <= cutoff
        for frame in frames
    ]
    stations = [station for station, accepted in zip(stations, keep, strict=True) if accepted]
    frames = [frame for frame, accepted in zip(frames, keep, strict=True) if accepted]
    if not stations:
        raise ValueError("Every station exceeded the configured missingness cutoff")
    train_mask = training_time_mask(timestamps, config["splits"])
    calendar = _calendar(timestamps, data_cfg["timezone"], data_cfg["timestamps_are_utc"])
    t_count, s_count = len(timestamps), len(stations)
    features = np.zeros((t_count, s_count, len(BASE_FEATURE_NAMES)), dtype=np.float32)
    cp_features = np.zeros((t_count, s_count, 2), dtype=np.float32)
    targets_raw = np.full((t_count, s_count, 2), np.nan, dtype=np.float32)
    targets_norm = np.zeros_like(targets_raw)
    target_mask = np.zeros_like(targets_raw, dtype=bool)
    target_log_mean = np.zeros((s_count, 2), dtype=np.float32)
    target_log_std = np.ones((s_count, 2), dtype=np.float32)
    target_raw_std = np.ones((s_count, 2), dtype=np.float32)
    target_q90 = np.zeros((s_count, 2), dtype=np.float32)
    coarse_log_mean = np.zeros(s_count, dtype=np.float32)
    coarse_log_std = np.ones(s_count, dtype=np.float32)

    for station_idx, frame in enumerate(frames):
        pm10 = pd.to_numeric(frame["PM10"], errors="coerce").where(lambda x: x >= 0)
        pm25 = pd.to_numeric(frame["PM2.5"], errors="coerce").where(lambda x: x >= 0)
        pm_series = [pm25, pm10]
        unreliable: list[np.ndarray] = []
        for series in pm_series:
            stuck, spike = _causal_quality_flags(series)
            unreliable.append((series.isna() | stuck | spike).to_numpy(dtype=np.float32))

        raw_pm = np.column_stack([pm25.to_numpy(), pm10.to_numpy()])
        targets_raw[:, station_idx] = raw_pm
        target_mask[:, station_idx] = np.isfinite(raw_pm)
        log_pm = np.log1p(raw_pm)
        log_z, log_mean, log_std = _zscore(log_pm, train_mask)
        targets_norm[:, station_idx] = log_z
        target_log_mean[station_idx] = log_mean
        target_log_std[station_idx] = log_std
        target_raw_std[station_idx] = np.nanstd(raw_pm[train_mask], axis=0)
        target_q90[station_idx] = np.nanquantile(raw_pm[train_mask], 0.90, axis=0)

        input_pm = np.column_stack(
            [series.ffill(limit=3).to_numpy(dtype=np.float64) for series in pm_series]
        )
        input_pm_z = (np.log1p(input_pm) - log_mean) / log_std

        wind_speed = pd.to_numeric(frame["wind_speed"], errors="coerce").where(lambda x: x >= 0)
        direction = frame["wind_direction"].map(WIND_ANGLE)
        radians = np.deg2rad(direction)
        wind_unknown = direction.isna().to_numpy(dtype=np.float64)
        weather = np.column_stack(
            [
                pd.to_numeric(frame["temperature"], errors="coerce"),
                pd.to_numeric(frame["pressure"], errors="coerce").where(lambda x: x >= 0),
                pd.to_numeric(frame["precipitation"], errors="coerce").where(lambda x: x >= 0),
                pd.to_numeric(frame["rain"], errors="coerce").where(lambda x: x >= 0),
                wind_speed,
                wind_speed * np.sin(radians),
                wind_speed * np.cos(radians),
                frame["clouds"].map(CLOUD_LEVEL),
            ]
        ).astype(np.float64)
        weather = pd.DataFrame(weather).ffill(limit=3).to_numpy()
        weather_z, _, _ = _zscore(weather, train_mask)

        features[:, station_idx, 0:2] = np.nan_to_num(input_pm_z, nan=0.0)
        features[:, station_idx, 2] = unreliable[0]
        features[:, station_idx, 3] = unreliable[1]
        features[:, station_idx, 4:11] = weather_z[:, :7]
        features[:, station_idx, 11] = wind_unknown
        features[:, station_idx, 12] = weather_z[:, 7]
        features[:, station_idx, 13:17] = calendar

        coarse = np.maximum(pm10.to_numpy() - pm25.to_numpy(), 0)
        coarse_log = np.log1p(coarse)
        coarse_z, c_mean, c_std = _zscore(coarse_log[:, None], train_mask)
        coarse_log_mean[station_idx] = c_mean[0]
        coarse_log_std[station_idx] = c_std[0]
        cp_features[:, station_idx, 0] = coarse_z[:, 0]
        cp_features[:, station_idx, 1] = np.maximum(unreliable[0], unreliable[1])

    origins = build_origins(
        timestamps,
        config["splits"],
        data_cfg["input_length"],
        data_cfg["horizon"],
    )
    return PreparedData(
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        stations=stations,
        features=features,
        cp_features=cp_features,
        targets_raw=targets_raw,
        targets_norm=targets_norm,
        target_mask=target_mask,
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
        target_raw_std=target_raw_std,
        target_q90=target_q90,
        coarse_log_mean=coarse_log_mean,
        coarse_log_std=coarse_log_std,
        origins=origins,
    )


def save_prepared(data: PreparedData, cache_path: str | Path, metadata_path: str | Path) -> None:
    cache_path, metadata_path = Path(cache_path), Path(metadata_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        timestamps=data.timestamps,
        features=data.features,
        cp_features=data.cp_features,
        targets_raw=data.targets_raw,
        targets_norm=data.targets_norm,
        target_mask=data.target_mask,
        target_log_mean=data.target_log_mean,
        target_log_std=data.target_log_std,
        target_raw_std=data.target_raw_std,
        target_q90=data.target_q90,
        coarse_log_mean=data.coarse_log_mean,
        coarse_log_std=data.coarse_log_std,
        train_a=data.origins.train_a,
        val_w=data.origins.val_w,
        train_b=data.origins.train_b,
        val_r=data.origins.val_r,
        test=data.origins.test,
    )
    metadata_path.write_text(
        json.dumps({"stations": data.stations, "feature_names": BASE_FEATURE_NAMES}, indent=2),
        encoding="utf-8",
    )


def load_prepared(cache_path: str | Path, metadata_path: str | Path) -> PreparedData:
    archive = np.load(cache_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    origins = OriginSets(**{name: archive[name] for name in ("train_a", "val_w", "train_b", "val_r", "test")})
    return PreparedData(
        timestamps=archive["timestamps"],
        stations=metadata["stations"],
        features=archive["features"],
        cp_features=archive["cp_features"],
        targets_raw=archive["targets_raw"],
        targets_norm=archive["targets_norm"],
        target_mask=archive["target_mask"],
        target_log_mean=archive["target_log_mean"],
        target_log_std=archive["target_log_std"],
        target_raw_std=archive["target_raw_std"],
        target_q90=archive["target_q90"],
        coarse_log_mean=archive["coarse_log_mean"],
        coarse_log_std=archive["coarse_log_std"],
        origins=origins,
    )


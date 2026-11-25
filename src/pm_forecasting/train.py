from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .losses import composite_loss, normalized_to_raw
from .models import make_model, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GPUWindowStore:
    def __init__(self, data, device: torch.device, input_length: int, horizon: int):
        self.features = torch.as_tensor(data.features, device=device)
        self.cp_features = torch.as_tensor(data.cp_features, device=device)
        self.targets_norm = torch.as_tensor(data.targets_norm, device=device)
        self.targets_raw = torch.as_tensor(data.targets_raw, device=device)
        self.target_mask = torch.as_tensor(data.target_mask, device=device)
        self.input_length, self.horizon = input_length, horizon
        self.n_stations = len(data.stations)
        self.history_offsets = torch.arange(-input_length, 0, device=device)
        self.target_offsets = torch.arange(horizon, device=device)

    def batch(self, origins: torch.Tensor, stations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        history_index = origins[:, None] + self.history_offsets[None, :]
        target_index = origins[:, None] + self.target_offsets[None, :]
        x = self.features[history_index, stations[:, None]]
        cp = self.cp_features[history_index, stations[:, None]]
        target_norm = self.targets_norm[target_index, stations[:, None]]
        target_raw = self.targets_raw[target_index, stations[:, None]]
        mask = self.target_mask[target_index, stations[:, None]]
        return x, cp, target_norm, target_raw, mask


def _pairs(origins: np.ndarray, n_stations: int, stride: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    selected = origins[offset::stride]
    return np.repeat(selected, n_stations), np.tile(np.arange(n_stations), len(selected))


@torch.no_grad()
def validation_loss(model, store, origins, data, config, device) -> float:
    model.eval()
    batch_size = config["training"]["batch_size"]
    origin_values, station_values = _pairs(origins, store.n_stations, config["data"]["eval_stride"])
    losses = []
    tensors = [
        torch.as_tensor(data.target_log_mean, device=device),
        torch.as_tensor(data.target_log_std, device=device),
        torch.as_tensor(data.target_raw_std, device=device),
        torch.as_tensor(data.target_q90, device=device),
    ]
    for start in range(0, len(origin_values), batch_size):
        batch_origins = torch.as_tensor(origin_values[start : start + batch_size], device=device)
        batch_stations = torch.as_tensor(station_values[start : start + batch_size], device=device)
        x, cp, target_norm, target_raw, mask = store.batch(batch_origins, batch_stations)
        prediction = model(x, batch_stations, cp)
        loss = composite_loss(prediction, target_norm, target_raw, mask, batch_stations, *tensors)
        penalty = getattr(model, "constraint_penalty", None)
        if penalty is not None:
            loss = loss + config["model"].get("physical_hinge_weight", 0.1) * penalty
        losses.append(float(loss))
    return float(np.mean(losses))


def train_model(model_name: str, data, config: dict, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the configured training workflow")
    torch.set_float32_matmul_precision("high")
    model = make_model(model_name, data, config).to(device)
    store = GPUWindowStore(data, device, config["data"]["input_length"], config["data"]["horizon"])
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
    stats = [
        torch.as_tensor(data.target_log_mean, device=device),
        torch.as_tensor(data.target_log_std, device=device),
        torch.as_tensor(data.target_raw_std, device=device),
        torch.as_tensor(data.target_q90, device=device),
    ]
    best_loss, best_state, stale_epochs = float("inf"), None, 0
    history, started = [], time.perf_counter()
    rng = np.random.default_rng(seed)
    train_stride = config["data"]["train_stride"]
    for epoch in range(training["max_epochs"]):
        model.train()
        offset = int(rng.integers(train_stride))
        origin_values, station_values = _pairs(data.origins.train, len(data.stations), train_stride, offset)
        order = rng.permutation(len(origin_values))
        epoch_losses = []
        for start in range(0, len(order), training["batch_size"]):
            batch_indices = order[start : start + training["batch_size"]]
            batch_origins = torch.as_tensor(origin_values[batch_indices], device=device)
            batch_stations = torch.as_tensor(station_values[batch_indices], device=device)
            x, cp, target_norm, target_raw, mask = store.batch(batch_origins, batch_stations)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=training["amp"]):
                prediction = model(x, batch_stations, cp)
                loss = composite_loss(prediction, target_norm, target_raw, mask, batch_stations, *stats)
                penalty = getattr(model, "constraint_penalty", None)
                if penalty is not None:
                    loss = loss + config["model"].get("physical_hinge_weight", 0.1) * penalty
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        val_w = validation_loss(model, store, data.origins.val_w, data, config, device)
        val_r = validation_loss(model, store, data.origins.val_r, data, config, device)
        validation = (val_w + val_r) / 2
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "val_w": val_w, "val_r": val_r, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation < best_loss - 1e-5:
            best_loss = validation
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training["patience"]:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    checkpoint_dir = Path(config["project"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_seed{seed}.pt"
    torch.save({"model": best_state, "history": history, "config": config, "seed": seed}, checkpoint_path)
    return {
        "model": model,
        "history": history,
        "best_validation": best_loss,
        "training_seconds": elapsed,
        "parameters": parameter_count(model),
        "checkpoint": str(checkpoint_path),
    }


@torch.no_grad()
def forecast_test(model, data, config: dict) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    store = GPUWindowStore(data, device, config["data"]["input_length"], config["data"]["horizon"])
    origins = data.origins.test[:: config["data"]["eval_stride"]]
    origin_values, station_values = _pairs(origins, len(data.stations), 1)
    predictions, truths, masks, gate_batches = [], [], [], []
    gate_names: list[str] = []
    log_mean = torch.as_tensor(data.target_log_mean, device=device)
    log_std = torch.as_tensor(data.target_log_std, device=device)
    model.eval()
    for start in range(0, len(origin_values), config["training"]["batch_size"]):
        batch_origins = torch.as_tensor(origin_values[start : start + config["training"]["batch_size"]], device=device)
        batch_stations = torch.as_tensor(station_values[start : start + config["training"]["batch_size"]], device=device)
        x, cp, _, target_raw, mask = store.batch(batch_origins, batch_stations)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config["training"]["amp"]):
            prediction_norm = model(x, batch_stations, cp)
        prediction = normalized_to_raw(prediction_norm.float(), batch_stations, log_mean, log_std)
        if getattr(model, "last_gates", None):
            if not gate_names:
                gate_names = sorted(model.last_gates)
            gate_batches.append(
                torch.stack([model.last_gates[name].mean(dim=1) for name in gate_names], dim=1)
                .cpu()
                .numpy()
            )
        predictions.append(prediction.cpu().numpy())
        truths.append(target_raw.cpu().numpy())
        masks.append(mask.cpu().numpy())
    result = {
        "origins": origins,
        "station_indices": station_values,
        "prediction": np.concatenate(predictions),
        "truth": np.concatenate(truths),
        "mask": np.concatenate(masks),
    }
    if gate_batches:
        result["gate_names"] = np.asarray(gate_names)
        result["gate_values"] = np.concatenate(gate_batches)
    return result



from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else 3)
        dilation = kwargs.get("dilation", 1)
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(*args, padding=self.left_padding, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return result[..., : -self.left_padding] if self.left_padding else result


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return x + residual


class MultiScaleStem(nn.Module):
    def __init__(self, in_features: int, channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [CausalConv1d(in_features, channels, kernel_size=kernel) for kernel in (3, 5, 7)]
        )
        self.scale_logits = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.scale_logits.softmax(dim=0)
        return sum(weight * branch(x) for weight, branch in zip(weights, self.branches, strict=True))


class ForecastModel(nn.Module):
    def __init__(self, n_stations: int, station_embedding: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.station_embedding = nn.Embedding(n_stations, station_embedding)

    def station_context(self, station: torch.Tensor, steps: int) -> torch.Tensor:
        return self.station_embedding(station).unsqueeze(1).expand(-1, steps, -1)


class CNN1D(ForecastModel):
    def __init__(self, n_features: int, n_stations: int, horizon: int, channels: int = 64, station_embedding: int = 8, dropout: float = 0.1):
        super().__init__(n_stations, station_embedding, horizon)
        in_channels = n_features + station_embedding
        self.network = nn.Sequential(
            CausalConv1d(in_channels, channels, kernel_size=7), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=5, dilation=2), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=3, dilation=4), nn.GELU(),
        )
        self.head = nn.Linear(channels, horizon * 2)

    def forward(self, x: torch.Tensor, station: torch.Tensor, cp: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([x, self.station_context(station, x.shape[1])], dim=-1).transpose(1, 2)
        return self.head(self.network(x)[..., -1]).view(-1, self.horizon, 2)


class LSTM(ForecastModel):
    def __init__(self, n_features: int, n_stations: int, horizon: int, hidden_size: int = 64, station_embedding: int = 8, dropout: float = 0.1):
        super().__init__(n_stations, station_embedding, horizon)
        self.lstm = nn.LSTM(n_features + station_embedding, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, horizon * 2)

    def forward(self, x: torch.Tensor, station: torch.Tensor, cp: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([x, self.station_context(station, x.shape[1])], dim=-1)
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).view(-1, self.horizon, 2)


class TCN(ForecastModel):
    def __init__(self, n_features: int, n_stations: int, horizon: int, channels: int = 32, station_embedding: int = 8, dropout: float = 0.1):
        super().__init__(n_stations, station_embedding, horizon)
        self.input = CausalConv1d(n_features + station_embedding, channels, kernel_size=3)
        self.blocks = nn.Sequential(*[ResidualTCNBlock(channels, 2**level, dropout) for level in range(6)])
        self.head = nn.Linear(channels, horizon * 2)

    def forward(self, x: torch.Tensor, station: torch.Tensor, cp: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([x, self.station_context(station, x.shape[1])], dim=-1).transpose(1, 2)
        encoded = self.blocks(self.input(x))
        return self.head(encoded[..., -1]).view(-1, self.horizon, 2)


class EBAMTN(ForecastModel):
    """Lean EBAMTN re-implementation with a 24-hour output head."""

    def __init__(self, n_features: int, n_stations: int, horizon: int, channels: int = 32, hidden_size: int = 64, station_embedding: int = 8, dropout: float = 0.1):
        super().__init__(n_stations, station_embedding, horizon)
        total_features = n_features + station_embedding
        self.stem = MultiScaleStem(total_features, channels)
        self.tcn = nn.Sequential(*[ResidualTCNBlock(channels, 2**i, dropout) for i in range(5)])
        self.lstm = nn.LSTM(total_features, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(channels + hidden_size * 2, channels + hidden_size * 2), nn.Sigmoid())
        self.head = nn.Linear(channels + hidden_size * 2, horizon * 2)

    def forward(self, x: torch.Tensor, station: torch.Tensor, cp: torch.Tensor | None = None) -> torch.Tensor:
        sequence = torch.cat([x, self.station_context(station, x.shape[1])], dim=-1)
        conv = self.tcn(self.stem(sequence.transpose(1, 2)))[..., -1]
        recurrent, _ = self.lstm(sequence)
        recurrent, _ = self.attention(recurrent, recurrent, recurrent, need_weights=False)
        fused = torch.cat([conv, recurrent[:, -1]], dim=-1)
        fused = fused * self.gate(fused)
        return self.head(fused).view(-1, self.horizon, 2)


class CPGE(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.a_gate = nn.Conv1d(channels * 2, channels, 1)
        self.b_gate = nn.Conv1d(channels * 2, channels, 1)
        self.a_from_b = nn.Conv1d(channels, channels, 1)
        self.b_from_a = nn.Conv1d(channels, channels, 1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joined = torch.cat([a, b], dim=1)
        gate_a, gate_b = torch.sigmoid(self.a_gate(joined)), torch.sigmoid(self.b_gate(joined))
        return a + gate_a * self.a_from_b(b), b + gate_b * self.b_from_a(a), gate_a, gate_b


class CPMTCNLite(ForecastModel):
    """Input-restricted two-tower TCN with gated cross-pollutant exchange."""

    common_indices = list(range(4, 17))

    def __init__(
        self,
        n_stations: int,
        horizon: int,
        target_log_mean: torch.Tensor,
        target_log_std: torch.Tensor,
        channels: int = 32,
        station_embedding: int = 8,
        dropout: float = 0.1,
        use_cpge: bool = True,
    ):
        super().__init__(n_stations, station_embedding, horizon)
        tower_features = 2 + len(self.common_indices) + station_embedding
        self.fine_stem = MultiScaleStem(tower_features, channels)
        self.coarse_stem = MultiScaleStem(tower_features, channels)
        self.fine_blocks = nn.ModuleList([ResidualTCNBlock(channels, 2**i, dropout) for i in range(6)])
        self.coarse_blocks = nn.ModuleList([ResidualTCNBlock(channels, 2**i, dropout) for i in range(6)])
        self.exchanges = nn.ModuleDict({str(i): CPGE(channels) for i in (1, 3, 5)})
        self.use_cpge = use_cpge
        self.fine_head = nn.Linear(channels, horizon)
        self.coarse_head = nn.Linear(channels, horizon)
        self.register_buffer("target_log_mean", target_log_mean.float())
        self.register_buffer("target_log_std", target_log_std.float())
        self.last_gates: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, station: torch.Tensor, cp: torch.Tensor | None = None) -> torch.Tensor:
        if cp is None:
            raise ValueError("CP-MTCN-lite requires coarse-part features")
        context = self.station_context(station, x.shape[1])
        common = x[..., self.common_indices]
        fine = torch.cat([x[..., [0, 2]], common, context], dim=-1).transpose(1, 2)
        coarse = torch.cat([cp, common, context], dim=-1).transpose(1, 2)
        fine, coarse = self.fine_stem(fine), self.coarse_stem(coarse)
        gate_readout: dict[str, torch.Tensor] = {}
        for level, (fine_block, coarse_block) in enumerate(zip(self.fine_blocks, self.coarse_blocks, strict=True)):
            fine, coarse = fine_block(fine), coarse_block(coarse)
            key = str(level)
            if self.use_cpge and key in self.exchanges:
                fine, coarse, gate_fine, gate_coarse = self.exchanges[key](fine, coarse)
                gate_readout[f"fine_{level}"] = gate_fine.mean(dim=-1).detach()
                gate_readout[f"coarse_{level}"] = gate_coarse.mean(dim=-1).detach()
        self.last_gates = gate_readout

        pm25_norm = self.fine_head(fine[..., -1])
        mean25 = self.target_log_mean[station, 0].unsqueeze(1)
        std25 = self.target_log_std[station, 0].unsqueeze(1)
        pm25_raw = torch.expm1(pm25_norm * std25 + mean25).clamp_min(0)
        coarse_raw = F.softplus(self.coarse_head(coarse[..., -1]))
        pm10_raw = pm25_raw + coarse_raw
        mean10 = self.target_log_mean[station, 1].unsqueeze(1)
        std10 = self.target_log_std[station, 1].unsqueeze(1)
        pm10_norm = (torch.log1p(pm10_raw) - mean10) / std10
        return torch.stack([pm25_norm, pm10_norm], dim=-1)


def make_model(name: str, data, config: dict) -> nn.Module:
    options = config["model"]
    common = dict(
        n_features=data.features.shape[-1],
        n_stations=len(data.stations),
        horizon=config["data"]["horizon"],
        station_embedding=options["station_embedding"],
        dropout=options["dropout"],
    )
    name = name.lower()
    if name == "cnn":
        return CNN1D(**common, channels=options.get("hidden_size", 64))
    if name == "lstm":
        return LSTM(**common, hidden_size=options.get("hidden_size", 64))
    if name == "tcn":
        return TCN(**common, channels=options.get("channels", 32))
    if name == "ebamtn":
        return EBAMTN(**common, channels=options.get("channels", 32), hidden_size=options.get("hidden_size", 64))
    if name in {"cpmtcn", "cpmtcn_no_exchange"}:
        common.pop("n_features")
        return CPMTCNLite(
            **common,
            target_log_mean=torch.as_tensor(data.target_log_mean),
            target_log_std=torch.as_tensor(data.target_log_std),
            channels=options.get("channels", 32),
            use_cpge=name == "cpmtcn",
        )
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


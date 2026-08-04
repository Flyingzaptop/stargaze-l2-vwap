from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(self.norm2(self.conv2(x)))
        return F.gelu(x + residual)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(int(input_size), int(hidden_size), kernel_size=1)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(
                int(hidden_size),
                int(kernel_size),
                dilation=2**layer,
                dropout=float(dropout),
            )
            for layer in range(int(layers))
        )
        self.output_norm = nn.LayerNorm(int(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TCN input must be [batch, time, features]")
        hidden = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden[:, :, -1])


@dataclass(frozen=True)
class ModelShape:
    input_size: int
    horizons: int
    hidden_size: int = 96
    layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.10
    embedding_size: int = 48


class DirectLineForecaster(nn.Module):
    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.shape = shape
        self.encoder = TCNEncoder(
            shape.input_size,
            shape.hidden_size,
            shape.layers,
            shape.kernel_size,
            shape.dropout,
        )
        self.mean_head = nn.Linear(shape.hidden_size, shape.horizons)
        self.sigma_head = nn.Linear(shape.hidden_size, shape.horizons)
        self.quality_head = nn.Linear(shape.hidden_size, shape.horizons)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        return {
            "mean": self.mean_head(hidden),
            "sigma": F.softplus(self.sigma_head(hidden)) + 0.05,
            "quality_logit": self.quality_head(hidden),
        }


class DirectSlopeForecaster(nn.Module):
    """Predict one anchored price-line slope per horizon.

    The output is a robustly-normalized slope. There is deliberately no
    intercept, uncertainty head, or auxiliary classifier: the only learned
    answer is the angle of the line through the current price.
    """

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.shape = shape
        self.encoder = TCNEncoder(
            shape.input_size,
            shape.hidden_size,
            shape.layers,
            shape.kernel_size,
            shape.dropout,
        )
        self.slope_head = nn.Linear(shape.hidden_size, shape.horizons)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"slope": self.slope_head(self.encoder(x))}


class DirectAngleForecaster(nn.Module):
    """Predict bounded volatility-normalized angles for several horizons."""

    def __init__(self, shape: ModelShape, *, max_angle_radians: float = 1.483529864) -> None:
        super().__init__()
        self.shape = shape
        self.max_angle_radians = float(max_angle_radians)
        self.encoder = TCNEncoder(
            shape.input_size,
            shape.hidden_size,
            shape.layers,
            shape.kernel_size,
            shape.dropout,
        )
        self.angle_head = nn.Linear(shape.hidden_size, shape.horizons)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.angle_head(self.encoder(x))
        return {"angle": self.max_angle_radians * torch.tanh(raw)}


class DirectRegimeForecaster(nn.Module):
    def __init__(self, shape: ModelShape, *, classes: int = 3) -> None:
        super().__init__()
        self.shape = shape
        self.classes = int(classes)
        self.encoder = TCNEncoder(
            shape.input_size,
            shape.hidden_size,
            shape.layers,
            shape.kernel_size,
            shape.dropout,
        )
        self.head = nn.Linear(shape.hidden_size, shape.horizons * self.classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        logits = self.head(hidden).reshape(len(x), self.shape.horizons, self.classes)
        return {"regime_logits": logits}


class RetrievalForecaster(nn.Module):
    def __init__(self, shape: ModelShape, *, task: str, classes: int = 3) -> None:
        super().__init__()
        if task not in {"line", "regime"}:
            raise ValueError("retrieval task must be 'line' or 'regime'")
        self.shape = shape
        self.task = task
        self.classes = int(classes)
        self.encoder = TCNEncoder(
            shape.input_size,
            shape.hidden_size,
            shape.layers,
            shape.kernel_size,
            shape.dropout,
        )
        self.embedding_head = nn.Linear(shape.hidden_size, shape.embedding_size)
        if task == "line":
            self.proxy_mean = nn.Linear(shape.embedding_size, shape.horizons)
            self.proxy_sigma = nn.Linear(shape.embedding_size, shape.horizons)
            self.proxy_quality = nn.Linear(shape.embedding_size, shape.horizons)
        else:
            self.proxy_regime = nn.Linear(shape.embedding_size, shape.horizons * self.classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        embedding = F.normalize(self.embedding_head(hidden), dim=-1)
        result = {"embedding": embedding}
        if self.task == "line":
            result.update(
                {
                    "mean": self.proxy_mean(embedding),
                    "sigma": F.softplus(self.proxy_sigma(embedding)) + 0.05,
                    "quality_logit": self.proxy_quality(embedding),
                }
            )
        else:
            result["regime_logits"] = self.proxy_regime(embedding).reshape(
                len(x),
                self.shape.horizons,
                self.classes,
            )
        return result

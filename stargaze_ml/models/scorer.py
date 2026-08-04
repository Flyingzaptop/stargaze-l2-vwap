from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .policy import VenueEncoder


@dataclass(frozen=True)
class CurveModelConfig:
    input_dim: int
    venue_feature_dim: int
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.10
    layer_norm_eps: float = 1e-5
    num_venues: int = 9
    use_venue_embeddings: bool = False
    num_aux_horizons: int = 0
    auxiliary_output_dim: int = 0
    separate_task_towers: bool = False

    def __post_init__(self) -> None:
        if min(self.input_dim, self.venue_feature_dim, self.d_model, self.nhead, self.num_layers, self.num_venues) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.num_aux_horizons < 0:
            raise ValueError("num_aux_horizons cannot be negative")
        if self.auxiliary_output_dim < 0:
            raise ValueError("auxiliary_output_dim cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CurveModelOutput:
    logits: Tensor
    scores: Tensor
    future_edges: Tensor | None = None


class FourCurveCausalTransformer(nn.Module):
    """Causal multi-venue Transformer returning exactly four score curves."""

    def __init__(self, config: CurveModelConfig) -> None:
        super().__init__()
        self.config = config
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps),
            nn.Linear(config.input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.venue_encoder = VenueEncoder(config)  # type: ignore[arg-type]
        self.input_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        def temporal_tower() -> nn.TransformerEncoder:
            tower_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                tower_layer,
                num_layers=config.num_layers,
                norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
                enable_nested_tensor=False,
            )

        def score_head(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, width),
            )

        if config.separate_task_towers:
            self.temporal = None
            self.task_towers = nn.ModuleDict(
                {"backward_task": temporal_tower(), "forward_task": temporal_tower()}
            )
            self.score_head = None
            self.task_heads = nn.ModuleDict(
                {"backward_task": score_head(2), "forward_task": score_head(2)}
            )
        else:
            self.temporal = nn.TransformerEncoder(
                layer,
                num_layers=config.num_layers,
                norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
                enable_nested_tensor=False,
            )
            self.task_towers = None
            self.score_head = score_head(4)
            self.task_heads = None
        auxiliary_width = (
            config.auxiliary_output_dim
            if config.auxiliary_output_dim > 0
            else 2 * config.num_aux_horizons
        )
        self.future_edge_head = nn.Linear(config.d_model, auxiliary_width) if auxiliary_width > 0 else None

    def forward(
        self,
        base_features: Tensor,
        venue_features: Tensor,
        venue_mask: Tensor | None = None,
    ) -> CurveModelOutput:
        if base_features.ndim != 3 or base_features.shape[-1] != self.config.input_dim:
            raise ValueError("base_features must have shape [B, T, input_dim]")
        tokens = self.base_encoder(base_features)
        encoding = self._sinusoidal_encoding(
            base_features.shape[1], self.config.d_model, tokens.device, tokens.dtype
        )
        tokens = self.input_norm(tokens + encoding.unsqueeze(0))
        tokens = self.venue_encoder(tokens, venue_features, venue_mask)
        length = tokens.shape[1]
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        if self.task_towers is not None and self.task_heads is not None:
            backward_hidden = self.task_towers["backward_task"](tokens, mask=causal_mask, is_causal=True)
            forward_hidden = self.task_towers["forward_task"](tokens, mask=causal_mask, is_causal=True)
            backward_logits = self.task_heads["backward_task"](backward_hidden)
            forward_logits = self.task_heads["forward_task"](forward_hidden)
            logits = torch.stack(
                (
                    backward_logits[..., 0],
                    forward_logits[..., 0],
                    backward_logits[..., 1],
                    forward_logits[..., 1],
                ),
                dim=-1,
            )
            auxiliary_hidden = forward_hidden
        else:
            if self.temporal is None or self.score_head is None:
                raise RuntimeError("shared temporal model is not initialized")
            auxiliary_hidden = self.temporal(tokens, mask=causal_mask, is_causal=True)
            logits = self.score_head(auxiliary_hidden)
        future_edges = self.future_edge_head(auxiliary_hidden) if self.future_edge_head is not None else None
        return CurveModelOutput(logits=logits, scores=torch.sigmoid(logits), future_edges=future_edges)

    @staticmethod
    def _sinusoidal_encoding(length: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        angles = positions * frequencies.unsqueeze(0)
        result = torch.zeros(length, width, device=device, dtype=torch.float32)
        result[:, 0::2] = torch.sin(angles)
        result[:, 1::2] = torch.cos(angles[:, : result[:, 1::2].shape[1]])
        return result.to(dtype=dtype)

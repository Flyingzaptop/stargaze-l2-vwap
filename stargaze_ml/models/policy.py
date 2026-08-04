"""Hierarchical causal Transformer policy for multi-venue market sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..contracts import ACTION_NAMES, Action, PositionSide


# A concise alias for model call sites; PositionSide remains exported too.
Position = PositionSide
NUM_ACTIONS = len(ACTION_NAMES)

_VALID_ACTIONS = (
    (True, True, True, False, False, False, False),
    (False, False, False, True, True, False, False),
    (False, False, False, False, False, True, True),
)


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture and input dimensions for the policy.

    ``input_dim`` describes the base ``[B, T, F]`` input. Set
    ``venue_feature_dim`` when ``[B, T, V, Fv]`` venue features will be used.
    ``num_horizons`` determines the auxiliary-head output dimensions.
    """

    input_dim: int
    venue_feature_dim: int | None = None
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_horizons: int = 4
    layer_norm_eps: float = 1e-5
    num_venues: int = 9
    use_venue_embeddings: bool = False

    def __post_init__(self) -> None:
        positive_ints = {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "num_horizons": self.num_horizons,
            "num_venues": self.num_venues,
        }
        if self.venue_feature_dim is not None:
            positive_ints["venue_feature_dim"] = self.venue_feature_dim
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be positive")


@dataclass
class ModelOutput:
    """Policy and auxiliary predictions for every sequence timestamp."""

    action_logits: Tensor
    valid_action_mask: Tensor
    actions: Tensor
    forward_long: Tensor
    forward_short: Tensor
    horizon_logits: Tensor
    future_flow: Tensor
    future_liquidity: Tensor

    @property
    def forward_predictions(self) -> Tensor:
        """Return long/short predictions as ``[B, T, H, 2]``."""

        return torch.stack((self.forward_long, self.forward_short), dim=-1)

    @property
    def selected_actions(self) -> Tensor:
        """Alias that makes the deterministic decision field explicit."""

        return self.actions

    @property
    def horizon_class_logits(self) -> Tensor:
        """Alias matching the horizon-class auxiliary-head terminology."""

        return self.horizon_logits


def _coerce_position_indices(position_state: Tensor) -> Tensor:
    if position_state.is_floating_point():
        rounded = position_state.round()
        if not torch.equal(position_state, rounded):
            raise ValueError("position_state values must be integer position IDs")
        position_state = rounded
    positions = position_state.to(dtype=torch.long)
    if positions.numel() == 0:
        raise ValueError("position_state cannot be empty")
    if torch.any((positions < PositionSide.FLAT) | (positions > PositionSide.SHORT)):
        raise ValueError("position_state values must be FLAT (0), LONG (1), or SHORT (2)")
    return positions


def build_valid_action_mask(position_state: Tensor) -> Tensor:
    """Build the structural action mask for flat, long, and short states.

    The returned boolean tensor has ``position_state.shape + (7,)``. ``True``
    means that an action is structurally possible in that position state.
    """

    positions = _coerce_position_indices(position_state)
    templates = torch.tensor(_VALID_ACTIONS, dtype=torch.bool, device=positions.device)
    return templates[positions]


def deterministic_argmax(action_logits: Tensor, valid_action_mask: Tensor | None = None) -> Tensor:
    """Select the first maximum logit, optionally enforcing a validity mask."""

    if action_logits.ndim < 1 or action_logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"action_logits must end in {NUM_ACTIONS} actions")
    if valid_action_mask is not None:
        if valid_action_mask.shape != action_logits.shape:
            raise ValueError("valid_action_mask must have the same shape as action_logits")
        if valid_action_mask.dtype is not torch.bool:
            raise TypeError("valid_action_mask must be a boolean tensor")
        if torch.any(~valid_action_mask.any(dim=-1)):
            raise ValueError("every row must contain at least one valid action")
        action_logits = action_logits.masked_fill(~valid_action_mask, -torch.inf)
    return torch.argmax(action_logits, dim=-1)


class VenueEncoder(nn.Module):
    """Encode venues, exchange information across them, then fuse into base tokens."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        if config.venue_feature_dim is None:
            raise ValueError("venue_feature_dim is required for VenueEncoder")

        d_model = config.d_model
        self.feature_dim = config.venue_feature_dim
        self.num_venues = config.num_venues
        self.venue_embedding = (
            nn.Embedding(config.num_venues, d_model) if config.use_venue_embeddings else None
        )
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim, eps=config.layer_norm_eps),
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model, eps=config.layer_norm_eps),
        )
        self.cross_venue_attention = nn.MultiheadAttention(
            d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.venue_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.base_to_venue_attention = nn.MultiheadAttention(
            d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.fusion_projection = nn.Linear(d_model, d_model, bias=False)
        self.fusion_gate = nn.Linear(2 * d_model, d_model)
        self.fusion_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        base_tokens: Tensor,
        venue_features: Tensor,
        venue_mask: Tensor | None = None,
    ) -> Tensor:
        if venue_features.ndim != 4:
            raise ValueError("venue_features must have shape [B, T, V, Fv]")
        batch_size, sequence_length, venue_count, feature_dim = venue_features.shape
        if base_tokens.shape[:2] != (batch_size, sequence_length):
            raise ValueError("base and venue batch/sequence dimensions must match")
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"expected venue feature dimension {self.feature_dim}, got {feature_dim}"
            )
        if venue_count == 0:
            raise ValueError("venue_features must contain at least one venue")

        encoded = self.input_encoder(venue_features)
        if self.venue_embedding is not None:
            if venue_count != self.num_venues:
                raise ValueError(f"expected {self.num_venues} venues, got {venue_count}")
            venue_ids = torch.arange(venue_count, device=venue_features.device)
            encoded = encoded + self.venue_embedding(venue_ids).view(1, 1, venue_count, -1)
        flattened = encoded.reshape(
            batch_size * sequence_length, venue_count, -1
        )
        valid_venues, safe_valid_venues, all_missing = self._prepare_mask(
            venue_mask,
            batch_size,
            sequence_length,
            venue_count,
            venue_features.device,
        )
        if valid_venues is not None:
            flattened = flattened.masked_fill(~valid_venues.unsqueeze(-1), 0.0)

        venue_context, _ = self.cross_venue_attention(
            flattened,
            flattened,
            flattened,
            key_padding_mask=None if safe_valid_venues is None else ~safe_valid_venues,
            need_weights=False,
        )
        encoded_venues = self.venue_norm(flattened + venue_context)
        if valid_venues is not None:
            encoded_venues = encoded_venues.masked_fill(~valid_venues.unsqueeze(-1), 0.0)

        query = base_tokens.reshape(batch_size * sequence_length, 1, -1)
        fused_context, _ = self.base_to_venue_attention(
            query,
            encoded_venues,
            encoded_venues,
            key_padding_mask=None if safe_valid_venues is None else ~safe_valid_venues,
            need_weights=False,
        )
        fused_context = fused_context.squeeze(1)
        if all_missing is not None:
            fused_context = fused_context.masked_fill(all_missing.unsqueeze(-1), 0.0)
        fused_context = fused_context.reshape(batch_size, sequence_length, -1)

        gate = torch.sigmoid(self.fusion_gate(torch.cat((base_tokens, fused_context), dim=-1)))
        update = gate * self.fusion_projection(fused_context)
        return self.fusion_norm(base_tokens + update)

    @staticmethod
    def _prepare_mask(
        venue_mask: Tensor | None,
        batch_size: int,
        sequence_length: int,
        venue_count: int,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if venue_mask is None:
            return None, None, None
        expected_shape = (batch_size, sequence_length, venue_count)
        if venue_mask.shape != expected_shape:
            raise ValueError(f"venue_mask must have shape {expected_shape}")
        if venue_mask.dtype is not torch.bool:
            raise TypeError("venue_mask must be a boolean tensor")
        if venue_mask.device != device:
            raise ValueError("venue_mask and venue_features must be on the same device")

        valid = venue_mask.reshape(batch_size * sequence_length, venue_count)
        all_missing = ~valid.any(dim=-1)
        safe_valid = valid.clone()
        safe_valid[all_missing, 0] = True
        return valid, safe_valid, all_missing


class HierarchicalCausalTransformerPolicy(nn.Module):
    """Causal sequence policy with per-timestamp hierarchical venue fusion."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.base_encoder = nn.Sequential(
            nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps),
            nn.Linear(config.input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.position_embedding = nn.Embedding(len(PositionSide), config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.venue_encoder = (
            VenueEncoder(config) if config.venue_feature_dim is not None else None
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
            enable_nested_tensor=False,
        )

        self.flat_action_head = nn.Linear(config.d_model, 3)
        self.long_action_head = nn.Linear(config.d_model, 2)
        self.short_action_head = nn.Linear(config.d_model, 2)
        self.forward_long_head = nn.Linear(config.d_model, config.num_horizons)
        self.forward_short_head = nn.Linear(config.d_model, config.num_horizons)
        self.horizon_class_head = nn.Linear(config.d_model, config.num_horizons)
        self.future_flow_head = nn.Linear(config.d_model, config.num_horizons)
        self.future_liquidity_head = nn.Linear(config.d_model, config.num_horizons)
        self.register_buffer(
            "_valid_action_templates",
            torch.tensor(_VALID_ACTIONS, dtype=torch.bool),
            persistent=False,
        )

    def forward(
        self,
        base_features: Tensor,
        position_state: Tensor,
        venue_features: Tensor | None = None,
        venue_mask: Tensor | None = None,
    ) -> ModelOutput:
        """Run the policy over a sequence.

        Args:
            base_features: Base market features shaped ``[B, T, F]``.
            position_state: FLAT/LONG/SHORT IDs shaped ``[B, T]``. A ``[B]``
                current-state tensor is also accepted and broadcast over time.
            venue_features: Optional venue features shaped ``[B, T, V, Fv]``.
            venue_mask: Optional boolean availability mask shaped ``[B, T, V]``.
        """

        batch_size, sequence_length = self._validate_base_features(base_features)
        positions = self._normalize_position_state(
            position_state, batch_size, sequence_length, base_features.device
        )

        tokens = self.base_encoder(base_features)
        tokens = tokens + self.position_embedding(positions)
        positions_encoding = self._sinusoidal_encoding(
            sequence_length, self.config.d_model, tokens.device, tokens.dtype
        )
        tokens = self.input_norm(tokens + positions_encoding.unsqueeze(0))

        if venue_features is not None:
            if self.venue_encoder is None:
                raise ValueError(
                    "venue_features were provided but config.venue_feature_dim is None"
                )
            if venue_features.device != base_features.device:
                raise ValueError("base_features and venue_features must be on the same device")
            if venue_features.dtype != base_features.dtype:
                raise TypeError("base_features and venue_features must have the same dtype")
            tokens = self.venue_encoder(tokens, venue_features, venue_mask)
        elif venue_mask is not None:
            raise ValueError("venue_mask cannot be provided without venue_features")

        causal_mask = self.causal_attention_mask(sequence_length, tokens.device)
        hidden = self.temporal_transformer(tokens, mask=causal_mask, is_causal=True)

        valid_actions = self._valid_action_templates[positions]
        action_logits = torch.cat(
            (self.flat_action_head(hidden), self.long_action_head(hidden), self.short_action_head(hidden)),
            dim=-1,
        ).masked_fill(~valid_actions, -torch.inf)
        actions = deterministic_argmax(action_logits)

        return ModelOutput(
            action_logits=action_logits,
            valid_action_mask=valid_actions,
            actions=actions,
            forward_long=self.forward_long_head(hidden),
            forward_short=self.forward_short_head(hidden),
            horizon_logits=self.horizon_class_head(hidden),
            future_flow=self.future_flow_head(hidden),
            future_liquidity=self.future_liquidity_head(hidden),
        )

    @staticmethod
    def causal_attention_mask(sequence_length: int, device: torch.device) -> Tensor:
        """Return an explicit mask where ``True`` blocks attention to the future."""

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ).triu(diagonal=1)

    @staticmethod
    def deterministic_actions(
        action_logits: Tensor, valid_action_mask: Tensor | None = None
    ) -> Tensor:
        return deterministic_argmax(action_logits, valid_action_mask)

    def _validate_base_features(self, base_features: Tensor) -> tuple[int, int]:
        if base_features.ndim != 3:
            raise ValueError("base_features must have shape [B, T, F]")
        batch_size, sequence_length, feature_dim = base_features.shape
        if batch_size == 0 or sequence_length == 0:
            raise ValueError("base_features batch and sequence dimensions must be non-empty")
        if feature_dim != self.config.input_dim:
            raise ValueError(
                f"expected base feature dimension {self.config.input_dim}, got {feature_dim}"
            )
        if not base_features.is_floating_point():
            raise TypeError("base_features must be a floating-point tensor")
        return batch_size, sequence_length

    @staticmethod
    def _normalize_position_state(
        position_state: Tensor,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        if position_state.device != device:
            raise ValueError("base_features and position_state must be on the same device")
        if position_state.shape == (batch_size,):
            position_state = position_state.unsqueeze(1).expand(-1, sequence_length)
        elif position_state.shape != (batch_size, sequence_length):
            raise ValueError(
                "position_state must have shape [B, T], or [B] for a broadcast state"
            )
        return _coerce_position_indices(position_state)

    @staticmethod
    def _sinusoidal_encoding(
        sequence_length: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        work_dtype = torch.float32
        positions = torch.arange(sequence_length, device=device, dtype=work_dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=work_dtype)
            * (-math.log(10_000.0) / d_model)
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(sequence_length, d_model, device=device, dtype=work_dtype)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)


# Short aliases keep the public API convenient without obscuring the full model name.
CausalTransformerPolicy = HierarchicalCausalTransformerPolicy
ModelConfig = PolicyConfig

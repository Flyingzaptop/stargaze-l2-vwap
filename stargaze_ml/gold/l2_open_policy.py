"""Single-output causal LSTM: the model decides only when to enter."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class L2OpenPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 96, initial_bias: float = -3.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.open_head = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.open_head.weight, gain=0.1)
        nn.init.constant_(self.open_head.bias, float(initial_bias))

    def forward(self, x: Tensor) -> Tensor:
        encoded, _ = self.lstm(x)
        return self.open_head(encoded).squeeze(-1)


def exploration_probability(
    logits: Tensor, *, temperature: float, random_action_floor: float
) -> Tensor:
    if temperature <= 0 or not 0 <= random_action_floor <= 1:
        raise ValueError("invalid exploration schedule")
    learned = torch.sigmoid(logits / float(temperature))
    return learned * (1.0 - float(random_action_floor)) + 0.5 * float(random_action_floor)


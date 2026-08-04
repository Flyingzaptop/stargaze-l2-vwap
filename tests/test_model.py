import torch

from stargaze_ml.models import (
    Action,
    HierarchicalCausalTransformerPolicy,
    PolicyConfig,
    Position,
    build_valid_action_mask,
)


def _config(dropout: float = 0.1) -> PolicyConfig:
    return PolicyConfig(
        input_dim=6,
        venue_feature_dim=4,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=dropout,
        num_horizons=3,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    base = torch.randn(2, 6, 6)
    venues = torch.randn(2, 6, 3, 4)
    positions = torch.tensor(
        [
            [Position.FLAT, Position.FLAT, Position.LONG, Position.LONG, Position.FLAT, Position.SHORT],
            [Position.SHORT, Position.SHORT, Position.FLAT, Position.LONG, Position.LONG, Position.FLAT],
        ]
    )
    return base, venues, positions


def test_output_shapes_with_and_without_venues() -> None:
    base, venues, positions = _inputs()
    model = HierarchicalCausalTransformerPolicy(_config()).eval()

    with torch.no_grad():
        output = model(base, positions, venues)
        base_only_output = model(base, positions)

    assert output.action_logits.shape == (2, 6, 7)
    assert output.valid_action_mask.shape == (2, 6, 7)
    assert output.actions.shape == (2, 6)
    assert output.actions.dtype == torch.long
    assert output.forward_long.shape == (2, 6, 3)
    assert output.forward_short.shape == (2, 6, 3)
    assert output.forward_predictions.shape == (2, 6, 3, 2)
    assert output.horizon_logits.shape == (2, 6, 3)
    assert output.future_flow.shape == (2, 6, 3)
    assert output.future_liquidity.shape == (2, 6, 3)
    assert base_only_output.action_logits.shape == (2, 6, 7)


def test_structural_action_masks_are_exact() -> None:
    positions = torch.tensor([Position.FLAT, Position.LONG, Position.SHORT])

    mask = build_valid_action_mask(positions)

    expected = torch.tensor(
        [
            [True, True, True, False, False, False, False],
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_impossible_actions_cannot_be_selected() -> None:
    model = HierarchicalCausalTransformerPolicy(_config()).eval()
    with torch.no_grad():
        model.flat_action_head.weight.zero_()
        model.long_action_head.weight.zero_()
        model.short_action_head.weight.zero_()
        model.flat_action_head.bias.copy_(torch.tensor([0.0, 100.0, 90.0]))
        model.long_action_head.bias.copy_(torch.tensor([80.0, 70.0]))
        model.short_action_head.bias.copy_(torch.tensor([60.0, 50.0]))

    base = torch.zeros(1, 3, 6)
    venues = torch.zeros(1, 3, 2, 4)
    positions = torch.tensor([[Position.FLAT, Position.LONG, Position.SHORT]])
    with torch.no_grad():
        output = model(base, positions, venues)

    assert output.actions.tolist() == [
        [Action.OPEN_LONG, Action.HOLD_LONG, Action.HOLD_SHORT]
    ]
    assert torch.isneginf(output.action_logits[~output.valid_action_mask]).all()
    assert output.valid_action_mask.gather(-1, output.actions.unsqueeze(-1)).all()


def test_eval_mode_is_deterministic() -> None:
    base, venues, positions = _inputs()
    model = HierarchicalCausalTransformerPolicy(_config(dropout=0.45)).eval()

    with torch.no_grad():
        first = model(base, positions, venues)
        second = model(base, positions, venues)

    for first_tensor, second_tensor in (
        (first.action_logits, second.action_logits),
        (first.actions, second.actions),
        (first.forward_long, second.forward_long),
        (first.forward_short, second.forward_short),
        (first.horizon_logits, second.horizon_logits),
        (first.future_flow, second.future_flow),
        (first.future_liquidity, second.future_liquidity),
    ):
        assert torch.equal(first_tensor, second_tensor)


def test_future_perturbation_has_strict_prefix_invariance() -> None:
    base, venues, positions = _inputs()
    model = HierarchicalCausalTransformerPolicy(_config()).eval()
    prefix_length = 4

    perturbed_base = base.clone()
    perturbed_venues = venues.clone()
    perturbed_positions = positions.clone()
    perturbed_base[:, prefix_length:] = torch.randn_like(perturbed_base[:, prefix_length:]) * 1000
    perturbed_venues[:, prefix_length:] = (
        torch.randn_like(perturbed_venues[:, prefix_length:]) * 1000
    )
    perturbed_positions[:, prefix_length:] = torch.tensor(
        [[Position.SHORT, Position.LONG], [Position.LONG, Position.SHORT]]
    )

    with torch.no_grad():
        original = model(base, positions, venues)
        perturbed = model(perturbed_base, perturbed_positions, perturbed_venues)

    for original_tensor, perturbed_tensor in (
        (original.action_logits, perturbed.action_logits),
        (original.actions, perturbed.actions),
        (original.valid_action_mask, perturbed.valid_action_mask),
        (original.forward_long, perturbed.forward_long),
        (original.forward_short, perturbed.forward_short),
        (original.horizon_logits, perturbed.horizon_logits),
        (original.future_flow, perturbed.future_flow),
        (original.future_liquidity, perturbed.future_liquidity),
    ):
        assert torch.equal(
            original_tensor[:, :prefix_length], perturbed_tensor[:, :prefix_length]
        )

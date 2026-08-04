"""Magnitude-aware direction learning for open-only VWAP excursion trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stargaze_ml.training.data import RobustNormalizer
from .l2_dominance_swap import _eligible_single_cross_events, _iter_batches
from .l2_multivwap_side import _open_entries
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices


@dataclass(frozen=True)
class ProfitDirectionConfig:
    epochs: int = 15
    head_only_epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 3e-4
    pnl_scale_ticks: float = 100.0
    min_side_weight: float = 0.25
    max_side_weight: float = 10.0
    regression_weight: float = 0.5
    distillation_weight: float = 0.25
    seed: int = 20260807


class L2ProfitDirectionPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.open_head = nn.Linear(hidden_size, 1)
        self.side_head = nn.Linear(hidden_size, 1)
        self.long_value_head = nn.Linear(hidden_size, 1)
        self.short_value_head = nn.Linear(hidden_size, 1)
        for head in (self.side_head, self.long_value_head, self.short_value_head):
            nn.init.zeros_(head.bias)
            nn.init.orthogonal_(head.weight, gain=0.1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.lstm(x)
        return (
            self.open_head(encoded).squeeze(-1),
            self.side_head(encoded).squeeze(-1),
            self.long_value_head(encoded).squeeze(-1),
            self.short_value_head(encoded).squeeze(-1),
        )


def executable_side_pnls(
    data: PreparedOpenData,
    start: int,
    crossing: int,
    config: OpenReinforceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    decisions = np.arange(int(start), int(crossing), dtype=np.int64)
    entry = decisions + 1
    exit_index = int(crossing) + 1
    round_trip_cost = 2.0 * (
        config.commission_per_fill_ticks + config.slippage_per_fill_ticks
    )
    long_pnl = (
        data.first_bid[exit_index] - data.first_ask[entry]
    ) / config.tick_size - round_trip_cost
    short_pnl = (
        data.first_bid[entry] - data.first_ask[exit_index]
    ) / config.tick_size - round_trip_cost
    return long_pnl.astype(np.float32), short_pnl.astype(np.float32)


def _make_batch(
    data: PreparedOpenData,
    events: np.ndarray,
    normalizer: RobustNormalizer,
    market_config: OpenReinforceConfig,
    config: ProfitDirectionConfig,
) -> tuple[np.ndarray, ...]:
    lengths = (data.event_crossing_1[events] - data.event_start[events]).astype(np.int64)
    maximum = int(lengths.max())
    x = np.zeros((len(events), maximum, data.x.shape[1]), dtype=np.float32)
    side = np.zeros((len(events), maximum), dtype=np.float32)
    long_value = np.zeros_like(side)
    short_value = np.zeros_like(side)
    weight = np.zeros_like(side)
    mask = np.zeros((len(events), maximum), dtype=bool)
    for row, event in enumerate(events):
        start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event])
        length = int(lengths[row])
        long_pnl, short_pnl = executable_side_pnls(data, start, crossing, market_config)
        x[row, :length] = normalizer.transform(data.x[start:crossing])
        side[row, :length] = long_pnl > short_pnl
        long_value[row, :length] = np.arcsinh(long_pnl / config.pnl_scale_ticks)
        short_value[row, :length] = np.arcsinh(short_pnl / config.pnl_scale_ticks)
        weight[row, :length] = np.clip(
            np.abs(long_pnl - short_pnl) / config.pnl_scale_ticks,
            config.min_side_weight,
            config.max_side_weight,
        )
        mask[row, :length] = (
            data.gate_open[start:crossing]
            & data.valid_feature[start:crossing]
            & data.observed[start + 1 : crossing + 1]
            & bool(data.observed[crossing + 1])
        )
    return x, side, long_value, short_value, weight, mask, lengths


def _packed_heads(
    model: L2ProfitDirectionPolicy, x: torch.Tensor, lengths: np.ndarray
) -> tuple[torch.Tensor, ...]:
    packed = pack_padded_sequence(
        x, torch.from_numpy(lengths), batch_first=True, enforce_sorted=False
    )
    encoded, _ = model.lstm(packed)
    encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
    return (
        model.open_head(encoded).squeeze(-1),
        model.side_head(encoded).squeeze(-1),
        model.long_value_head(encoded).squeeze(-1),
        model.short_value_head(encoded).squeeze(-1),
    )


def _trade_predictions(
    model: L2ProfitDirectionPolicy,
    teacher: L2OpenPolicy,
    data: PreparedOpenData,
    normalizer: RobustNormalizer,
    events: np.ndarray,
    open_threshold: float,
    device: torch.device,
    market_config: OpenReinforceConfig,
    config: ProfitDirectionConfig,
) -> list[dict[str, float]]:
    entries = _open_entries(teacher, data, normalizer, events, open_threshold, device)
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for event, entry in entries.items():
            start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event])
            x = torch.from_numpy(normalizer.transform(data.x[start:crossing]))[None].to(device)
            _, side_logit, long_value, short_value = model(x)
            offset = int(entry) - start
            long_pnl, short_pnl = executable_side_pnls(
                data, int(entry), crossing, market_config
            )
            actual_long = float(long_pnl[0]); actual_short = float(short_pnl[0])
            probability = float(torch.sigmoid(side_logit[0, offset]).cpu())
            predicted_long = float(
                np.sinh(float(long_value[0, offset].cpu())) * config.pnl_scale_ticks
            )
            predicted_short = float(
                np.sinh(float(short_value[0, offset].cpu())) * config.pnl_scale_ticks
            )
            rows.append({
                "event": float(event), "entry": float(entry),
                "long_pnl": actual_long, "short_pnl": actual_short,
                "oracle_pnl": max(actual_long, actual_short),
                "side_probability": probability,
                "side_confidence": abs(probability - 0.5) * 2.0,
                "predicted_long_pnl": predicted_long,
                "predicted_short_pnl": predicted_short,
                "predicted_edge": max(predicted_long, predicted_short),
                "predicted_advantage": abs(predicted_long - predicted_short),
            })
    return rows


def _summary(rows: list[dict[str, float]], mode: str, cutoff: float = 0.0) -> dict[str, float]:
    selected = [row for row in rows if row["side_confidence"] >= cutoff]
    if not selected:
        return {"trades": 0, "mean_pnl_ticks": 0.0, "median_pnl_ticks": 0.0,
                "win_rate": 0.0, "oracle_win_rate": 0.0}
    if mode == "classifier":
        pnl = np.asarray([
            row["long_pnl"] if row["side_probability"] >= 0.5 else row["short_pnl"]
            for row in selected
        ])
    elif mode == "value":
        pnl = np.asarray([
            row["long_pnl"]
            if row["predicted_long_pnl"] >= row["predicted_short_pnl"]
            else row["short_pnl"]
            for row in selected
        ])
    else:
        raise ValueError("mode must be classifier or value")
    oracle = np.asarray([row["oracle_pnl"] for row in selected])
    return {
        "trades": int(len(pnl)),
        "mean_pnl_ticks": float(pnl.mean()),
        "median_pnl_ticks": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()),
        "p05_pnl_ticks": float(np.quantile(pnl, 0.05)),
        "total_pnl_ticks": float(pnl.sum()),
        "oracle_win_rate": float((oracle > 0).mean()),
        "side_accuracy": float(np.mean([
            (row["side_probability"] >= 0.5) == (row["long_pnl"] > row["short_pnl"])
            for row in selected
        ])),
    }


def train_profit_direction(
    prepared_path: str | Path,
    open_checkpoint_path: str | Path,
    output_dir: str | Path,
    config: ProfitDirectionConfig,
    *,
    device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    data = PreparedOpenData(prepared_path)
    checkpoint = torch.load(Path(open_checkpoint_path).resolve(strict=True), map_location=device, weights_only=False)
    market_config = OpenReinforceConfig(**checkpoint["config"])
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), market_config.hidden_size).to(device)
    teacher.load_state_dict(checkpoint["model_state"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    model = L2ProfitDirectionPolicy(len(data.feature_names), market_config.hidden_size).to(device)
    model.lstm.load_state_dict(teacher.lstm.state_dict())
    model.open_head.load_state_dict(teacher.open_head.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_events = _eligible_single_cross_events(data, 0, data.train_end)
    validation_events = _eligible_single_cross_events(data, data.train_end, data.validation_end)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int | bool]] = []
    best_score = -np.inf; best_state = None
    for epoch in range(config.epochs):
        head_only = epoch < config.head_only_epochs
        for parameter in model.lstm.parameters(): parameter.requires_grad_(not head_only)
        for parameter in model.open_head.parameters(): parameter.requires_grad_(not head_only)
        model.train(); losses: list[float] = []
        for events in _iter_batches(train_events, config.batch_size, rng):
            batch = _make_batch(data, events, normalizer, market_config, config)
            x, side, long_value, short_value, weight, mask, lengths = batch
            if not np.any(mask):
                continue
            xt = torch.from_numpy(x).to(device); mt = torch.from_numpy(mask).to(device)
            open_logits, side_logits, long_pred, short_pred = _packed_heads(model, xt, lengths)
            with torch.no_grad(): teacher_open = teacher(xt)
            target = torch.from_numpy(side).to(device); wt = torch.from_numpy(weight).to(device)
            lv = torch.from_numpy(long_value).to(device); sv = torch.from_numpy(short_value).to(device)
            side_raw = nn.functional.binary_cross_entropy_with_logits(
                side_logits[mt], target[mt], reduction="none"
            )
            side_loss = (side_raw * wt[mt]).sum() / wt[mt].sum().clamp_min(1e-6)
            value_loss = 0.5 * (
                nn.functional.smooth_l1_loss(long_pred[mt], lv[mt])
                + nn.functional.smooth_l1_loss(short_pred[mt], sv[mt])
            )
            distillation = nn.functional.mse_loss(open_logits[mt], teacher_open[mt])
            loss = side_loss + config.regression_weight * value_loss + config.distillation_weight * distillation
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); logits_all=[]; targets_all=[]; weights_all=[]
        with torch.no_grad():
            for begin in range(0, len(validation_events), config.batch_size):
                events = validation_events[begin:begin + config.batch_size]
                batch = _make_batch(data, events, normalizer, market_config, config)
                x, side, _, _, weight, mask, lengths = batch
                _, logits, _, _ = _packed_heads(
                    model, torch.from_numpy(x).to(device), lengths
                )
                logits_all.append(logits.cpu().numpy()[mask]); targets_all.append(side[mask]); weights_all.append(weight[mask])
        logits_np=np.concatenate(logits_all); targets_np=np.concatenate(targets_all); weights_np=np.concatenate(weights_all)
        auc=float(roc_auc_score(targets_np, logits_np))
        weighted_auc=float(roc_auc_score(targets_np, logits_np, sample_weight=weights_np))
        row={"epoch":epoch+1,"loss":float(np.mean(losses)),"validation_auc":auc,
             "validation_weighted_auc":weighted_auc,"head_only":head_only}
        history.append(row); print(json.dumps(row), flush=True)
        if weighted_auc > best_score:
            best_score=weighted_auc
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state); model.to(device); model.eval()
    open_threshold = float(checkpoint["validation"]["best"]["threshold"])
    split_rows: dict[str, list[dict[str, float]]] = {}
    for split, left, right in (
        ("validation", data.train_end, data.validation_end),
        ("test", data.validation_end, len(data.x)),
    ):
        events = _event_indices(data, left, right, good_only=False)
        split_rows[split] = _trade_predictions(
            model, teacher, data, normalizer, events, open_threshold, device,
            market_config, config,
        )
    validation = split_rows["validation"]; test = split_rows["test"]
    candidate_cutoffs = sorted(set(
        [0.0] + [float(np.quantile([row["side_confidence"] for row in validation], q))
                 for q in (0.50, 0.75, 0.90, 0.95)]
    ))
    grid=[]
    for mode in ("classifier", "value"):
        for cutoff in candidate_cutoffs:
            row={"mode":mode,"confidence_cutoff":cutoff,**_summary(validation,mode,cutoff)}
            grid.append(row)
    eligible=[row for row in grid if row["trades"] >= 60]
    selected=max(eligible or grid,key=lambda row:row["mean_pnl_ticks"])
    fixed_test={"mode":selected["mode"],"confidence_cutoff":selected["confidence_cutoff"],
                **_summary(test,str(selected["mode"]),float(selected["confidence_cutoff"]))}
    report={"device":str(device),"best_validation_weighted_auc":best_score,
            "open_threshold":open_threshold,"history":history,"validation_grid":grid,
            "selected_on_validation":selected,"fixed_test":fixed_test,
            "test_unfiltered_classifier":_summary(test,"classifier",0.0),
            "test_unfiltered_value":_summary(test,"value",0.0)}
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":asdict(config),
                "market_config":asdict(market_config),"feature_names":data.feature_names,
                "normalizer":normalizer.to_dict(),"open_threshold":open_threshold,
                "evaluation":report},out/"final.pt")
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

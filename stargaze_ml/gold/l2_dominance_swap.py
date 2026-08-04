"""Dynamic price/VWAP dominance head and evidence-gated position swaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stargaze_ml.training.data import RobustNormalizer
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices


@dataclass(frozen=True)
class DominanceConfig:
    epochs: int = 15
    head_only_epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 3e-4
    distillation_weight: float = 0.5
    seed: int = 20260806
    open_threshold: float | None = None
    confirmation_seconds: int = 3
    swap_confidences: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)


class L2DominanceSwapPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.open_head = nn.Linear(hidden_size, 1)
        self.dominance_head = nn.Linear(hidden_size, 1)
        self.side_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.dominance_head.bias)
        nn.init.orthogonal_(self.dominance_head.weight, gain=0.1)
        nn.init.zeros_(self.side_head.bias)
        nn.init.orthogonal_(self.side_head.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.lstm(x)
        return (
            self.open_head(encoded).squeeze(-1),
            self.dominance_head(encoded).squeeze(-1),
            self.side_head(encoded).squeeze(-1),
        )


def dynamic_dominance_target(
    mid: np.ndarray, vwap: np.ndarray, start: int, crossing: int, side: int
) -> np.ndarray:
    """One means VWAP dominates: price closes more of the remaining gap."""
    rows = np.arange(int(start), int(crossing), dtype=np.int64)
    price_closure = -int(side) * (mid[int(crossing)] - mid[rows])
    vwap_closure = int(side) * (vwap[int(crossing)] - vwap[rows])
    return (price_closure > vwap_closure).astype(np.float32)


def dynamic_long_side_target(
    data: PreparedOpenData, start: int, crossing: int
) -> np.ndarray:
    """One means a long entered next second beats a short at the crossing."""
    decisions = np.arange(int(start), int(crossing), dtype=np.int64)
    entry = decisions + 1; exit_index = int(crossing) + 1
    long_pnl = data.first_bid[exit_index] - data.first_ask[entry]
    short_pnl = data.first_bid[entry] - data.first_ask[exit_index]
    return (long_pnl > short_pnl).astype(np.float32)


def dynamic_reversion_target(
    data: PreparedOpenData, start: int, crossing: int, event_side: int
) -> np.ndarray:
    """One means the mean-reversion side beats continuation at the crossing."""
    long_wins = dynamic_long_side_target(data, start, crossing)
    return long_wins if int(event_side) < 0 else 1.0 - long_wins


def _eligible_single_cross_events(data: PreparedOpenData, left: int, right: int) -> np.ndarray:
    return np.flatnonzero(
        (data.event_start >= int(left))
        & (data.event_crossing_1 + 1 < int(right))
        & (data.event_crossing_1 > data.event_start)
    )


def _iter_batches(events: np.ndarray, batch_size: int, rng: np.random.Generator):
    order = events.copy(); rng.shuffle(order)
    for left in range(0, len(order), int(batch_size)):
        yield order[left : left + int(batch_size)]


def _make_batch(
    data: PreparedOpenData, events: np.ndarray, normalizer: RobustNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lengths = (data.event_crossing_1[events] - data.event_start[events]).astype(np.int64)
    maximum = int(lengths.max())
    x = np.zeros((len(events), maximum, data.x.shape[1]), dtype=np.float32)
    target = np.zeros((len(events), maximum), dtype=np.float32)
    side_target = np.zeros((len(events), maximum), dtype=np.float32)
    mask = np.zeros((len(events), maximum), dtype=bool)
    for row, event in enumerate(events):
        start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event]); length = int(lengths[row])
        x[row, :length] = normalizer.transform(data.x[start:crossing])
        target[row, :length] = dynamic_dominance_target(
            data.mid, data.primary_vwap, start, crossing, int(data.event_side[event])
        )
        side_target[row, :length] = dynamic_reversion_target(
            data, start, crossing, int(data.event_side[event])
        )
        mask[row, :length] = (
            data.valid_feature[start:crossing]
            & data.observed[start + 1 : crossing + 1]
            & bool(data.observed[crossing + 1])
        )
    return x, target, side_target, mask, lengths


def _packed_heads(
    model: L2DominanceSwapPolicy, x: torch.Tensor, lengths: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = pack_padded_sequence(
        x, torch.from_numpy(lengths), batch_first=True, enforce_sorted=False
    )
    encoded, _ = model.lstm(packed)
    encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
    return (
        model.open_head(encoded).squeeze(-1),
        model.dominance_head(encoded).squeeze(-1),
        model.side_head(encoded).squeeze(-1),
    )


def _fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    temperatures = np.linspace(0.35, 3.0, 107)
    best_temperature = 1.0; best_loss = float("inf")
    for temperature in temperatures:
        scaled = np.clip(logits / temperature, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-scaled))
        loss = -np.mean(target * np.log(probability + 1e-9) + (1.0 - target) * np.log(1.0 - probability + 1e-9))
        if loss < best_loss:
            best_loss = float(loss); best_temperature = float(temperature)
    return best_temperature


def _predict_event(
    model: L2DominanceSwapPolicy, data: PreparedOpenData, normalizer: RobustNormalizer,
    start: int, crossing: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = normalizer.transform(data.x[int(start):int(crossing)])[None]
    with torch.no_grad():
        open_logits, dominance_logits, side_logits = model(torch.from_numpy(x).to(device))
    return (
        open_logits[0].cpu().numpy(), dominance_logits[0].cpu().numpy(),
        side_logits[0].cpu().numpy(),
    )


def _split_predictions(
    model: L2DominanceSwapPolicy, data: PreparedOpenData, normalizer: RobustNormalizer,
    events: np.ndarray, device: torch.device,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sequences: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    output: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    model.eval()
    for event in events:
        pieces: list[tuple[np.ndarray, np.ndarray]] = []
        for start, crossing in (
            (int(data.event_start[event]), int(data.event_crossing_1[event])),
            (int(data.event_crossing_1[event]), int(data.event_crossing_2[event])),
        ):
            key = (start, crossing)
            if key not in sequences:
                sequences[key] = _predict_event(model, data, normalizer, start, crossing, device)
            pieces.append(sequences[key])
        output[int(event)] = (
            np.concatenate((pieces[0][0], pieces[1][0])),
            np.concatenate((pieces[0][1], pieces[1][1])),
            np.concatenate((pieces[0][2], pieces[1][2])),
        )
    return output


def _position_from_belief(event_side: int, q_vwap: float) -> int:
    return -int(event_side) if q_vwap >= 0.5 else int(event_side)


def _trade_episode(
    data: PreparedOpenData, event: int, open_logits: np.ndarray, dominance_logits: np.ndarray,
    side_logits: np.ndarray, *, dominance_temperature: float, side_temperature: float,
    open_threshold: float, exit_crossing: int, entry_mode: str,
    swap_confidence: float | None, confirmation_seconds: int,
    commission_ticks: float, slippage_ticks: float, tick_size: float,
) -> tuple[float, int, int, float, float] | None:
    start = int(data.event_start[event]); crossing1 = int(data.event_crossing_1[event]); crossing2 = int(data.event_crossing_2[event])
    gate = int(data.event_gate_index[event])
    if gate < start:
        return None
    first_length = crossing1 - start
    open_probability = 1.0 / (1.0 + np.exp(-np.clip(open_logits[:first_length], -30.0, 30.0)))
    allowed = (
        data.gate_open[start:crossing1]
        & data.valid_feature[start:crossing1]
        & data.observed[start + 1 : crossing1 + 1]
    )
    hit = np.flatnonzero(allowed & (open_probability >= float(open_threshold)))
    if hit.size == 0:
        return None
    entry_decision = start + int(hit[0]); entry_execution = entry_decision + 1
    q = 1.0 / (1.0 + np.exp(-np.clip(dominance_logits / float(dominance_temperature), -30.0, 30.0)))
    side_probability = 1.0 / (1.0 + np.exp(-np.clip(side_logits / float(side_temperature), -30.0, 30.0)))
    entry_offset = entry_decision - start
    if entry_mode == "mean_reversion":
        position = -int(data.side[entry_decision])
    elif entry_mode == "continuation":
        position = int(data.side[entry_decision])
    elif entry_mode == "side_model":
        position = (
            -int(data.side[entry_decision])
            if float(side_probability[entry_offset]) >= 0.5
            else int(data.side[entry_decision])
        )
    else:
        raise ValueError("unknown entry_mode")
    cost = float(commission_ticks) + float(slippage_ticks)
    if position > 0:
        cash = -(data.first_ask[entry_execution] / tick_size + cost)
    else:
        cash = data.first_bid[entry_execution] / tick_size - cost
    swaps = 0; evidence_side = 0; evidence_count = 0
    final_crossing = crossing1 if int(exit_crossing) == 1 else crossing2
    for decision in range(entry_decision + 1, final_crossing):
        offset = decision - start
        belief = float(side_probability[offset])
        desired = 0
        if swap_confidence is not None:
            if belief >= float(swap_confidence):
                desired = -int(data.side[decision])
            elif belief <= 1.0 - float(swap_confidence):
                desired = int(data.side[decision])
        if desired != 0 and desired != position and data.observed[decision + 1]:
            if desired == evidence_side:
                evidence_count += 1
            else:
                evidence_side = desired; evidence_count = 1
            if evidence_count >= int(confirmation_seconds):
                execution = decision + 1
                if position > 0:
                    cash += 2.0 * (data.first_bid[execution] / tick_size - cost)
                else:
                    cash -= 2.0 * (data.first_ask[execution] / tick_size + cost)
                position = desired; swaps += 1; evidence_side = 0; evidence_count = 0
        else:
            evidence_side = 0; evidence_count = 0
    exit_execution = final_crossing + 1
    if position > 0:
        cash += data.first_bid[exit_execution] / tick_size - cost
    else:
        cash -= data.first_ask[exit_execution] / tick_size + cost
    target_at_entry = float(dynamic_dominance_target(
        data.mid, data.primary_vwap, entry_decision, crossing1, int(data.side[entry_decision])
    )[0])
    side_target = float(dynamic_reversion_target(
        data, entry_decision, crossing1, int(data.side[entry_decision])
    )[0])
    return float(cash), int(swaps), int(entry_decision), target_at_entry, side_target


def evaluate_swap_grid(
    model: L2DominanceSwapPolicy, data: PreparedOpenData, normalizer: RobustNormalizer,
    config: DominanceConfig, dominance_temperature: float, side_temperature: float,
    device: torch.device,
) -> dict[str, object]:
    cost_config = OpenReinforceConfig()
    output: dict[str, object] = {}
    for split, left, right in (
        ("validation", data.train_end, data.validation_end),
        ("test", data.validation_end, len(data.x)),
    ):
        events = _event_indices(data, left, right, good_only=False)
        predictions = _split_predictions(model, data, normalizer, events, device)
        split_rows: list[dict[str, object]] = []
        for exit_crossing in (1, 2):
            settings = [
                ("mean_reversion", None), ("continuation", None),
                ("side_model", None),
                *(("side_model", confidence) for confidence in config.swap_confidences),
            ]
            for entry_mode, confidence in settings:
                pnls: list[float] = []; swaps: list[int] = []; labels: list[int] = []; beliefs: list[float] = []
                side_labels: list[int] = []; side_beliefs: list[float] = []
                for event in events:
                    open_logits, dominance_logits, side_logits = predictions[int(event)]
                    trade = _trade_episode(
                        data, int(event), open_logits, dominance_logits, side_logits,
                        dominance_temperature=dominance_temperature,
                        side_temperature=side_temperature,
                        open_threshold=config.open_threshold, exit_crossing=exit_crossing,
                        entry_mode=entry_mode,
                        swap_confidence=confidence, confirmation_seconds=config.confirmation_seconds,
                        commission_ticks=cost_config.commission_per_fill_ticks,
                        slippage_ticks=cost_config.slippage_per_fill_ticks,
                        tick_size=cost_config.tick_size,
                    )
                    if trade is None:
                        continue
                    pnl, swap_count, entry, label, side_label = trade
                    pnls.append(pnl); swaps.append(swap_count); labels.append(int(label))
                    offset = entry - int(data.event_start[event])
                    beliefs.append(float(1.0 / (1.0 + np.exp(-dominance_logits[offset] / dominance_temperature))))
                    side_labels.append(int(side_label))
                    side_beliefs.append(float(1.0 / (1.0 + np.exp(-side_logits[offset] / side_temperature))))
                pnl_array = np.asarray(pnls); swap_array = np.asarray(swaps)
                row = {
                    "exit_crossing": exit_crossing,
                    "entry_mode": entry_mode,
                    "swap_confidence": confidence,
                    "trades": int(len(pnl_array)),
                    "mean_pnl_ticks": float(pnl_array.mean()) if len(pnl_array) else 0.0,
                    "median_pnl_ticks": float(np.median(pnl_array)) if len(pnl_array) else 0.0,
                    "win_rate": float((pnl_array > 0).mean()) if len(pnl_array) else 0.0,
                    "p05_pnl_ticks": float(np.quantile(pnl_array, 0.05)) if len(pnl_array) else 0.0,
                    "total_pnl_ticks": float(pnl_array.sum()),
                    "mean_swaps": float(swap_array.mean()) if len(swap_array) else 0.0,
                    "swapped_trade_fraction": float((swap_array > 0).mean()) if len(swap_array) else 0.0,
                    "entry_dominance_accuracy": float(((np.asarray(beliefs) >= 0.5) == np.asarray(labels)).mean()) if labels else 0.0,
                    "entry_dominance_auc": float(roc_auc_score(labels, beliefs)) if len(set(labels)) > 1 else 0.5,
                    "entry_side_accuracy": float(((np.asarray(side_beliefs) >= 0.5) == np.asarray(side_labels)).mean()) if side_labels else 0.0,
                    "entry_side_auc": float(roc_auc_score(side_labels, side_beliefs)) if len(set(side_labels)) > 1 else 0.5,
                }
                split_rows.append(row)
        output[split] = split_rows
    validation_rows = output["validation"]
    eligible = [row for row in validation_rows if int(row["trades"]) >= 20 and row["entry_mode"] == "side_model"]
    best = max(eligible, key=lambda row: float(row["mean_pnl_ticks"]))
    test_match = next(
        row for row in output["test"]
        if row["exit_crossing"] == best["exit_crossing"]
        and row["entry_mode"] == best["entry_mode"]
        and row["swap_confidence"] == best["swap_confidence"]
    )
    output["selected_on_validation"] = best
    output["fixed_test"] = test_match
    return output


def train_dominance_swap(
    prepared_path: str | Path, open_checkpoint_path: str | Path,
    output_dir: str | Path, config: DominanceConfig,
    *, device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    data = PreparedOpenData(prepared_path)
    checkpoint = torch.load(Path(open_checkpoint_path).resolve(strict=True), map_location=device, weights_only=False)
    if config.open_threshold is None:
        config = replace(
            config,
            open_threshold=float(checkpoint["validation"]["best"]["threshold"]),
        )
    open_config = OpenReinforceConfig(**checkpoint["config"])
    normalizer = RobustNormalizer.from_dict(checkpoint["normalizer"])
    teacher = L2OpenPolicy(len(data.feature_names), open_config.hidden_size).to(device)
    teacher.load_state_dict(checkpoint["model_state"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    model = L2DominanceSwapPolicy(len(data.feature_names), open_config.hidden_size).to(device)
    model.lstm.load_state_dict(teacher.lstm.state_dict()); model.open_head.load_state_dict(teacher.open_head.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_events = _eligible_single_cross_events(data, 0, data.train_end)
    validation_events = _eligible_single_cross_events(data, data.train_end, data.validation_end)
    rng = np.random.default_rng(config.seed); history: list[dict[str, float]] = []; best_auc = -1.0; best_state = None
    for epoch in range(config.epochs):
        head_only = epoch < config.head_only_epochs
        for parameter in model.lstm.parameters(): parameter.requires_grad_(not head_only)
        for parameter in model.open_head.parameters(): parameter.requires_grad_(not head_only)
        model.train(); losses=[]
        for events in _iter_batches(train_events, config.batch_size, rng):
            x, target, side_target, mask, lengths = _make_batch(data, events, normalizer)
            xt = torch.from_numpy(x).to(device); yt = torch.from_numpy(target).to(device); syt = torch.from_numpy(side_target).to(device); mt = torch.from_numpy(mask).to(device)
            open_logits, dominance_logits, side_logits = _packed_heads(model, xt, lengths)
            with torch.no_grad():
                teacher_open = teacher(xt)
            dominance_loss = nn.functional.binary_cross_entropy_with_logits(dominance_logits[mt], yt[mt])
            side_loss = nn.functional.binary_cross_entropy_with_logits(side_logits[mt], syt[mt])
            distillation = nn.functional.mse_loss(open_logits[mt], teacher_open[mt])
            loss = dominance_loss + side_loss + float(config.distillation_weight) * distillation
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_logits=[]; val_target=[]; val_side_logits=[]; val_side_target=[]
        with torch.no_grad():
            for begin in range(0, len(validation_events), config.batch_size):
                events = validation_events[begin:begin+config.batch_size]
                x, target, side_target, mask, lengths = _make_batch(data, events, normalizer)
                _, logits, side_logits = _packed_heads(model, torch.from_numpy(x).to(device), lengths)
                val_logits.append(logits.cpu().numpy()[mask]); val_target.append(target[mask])
                val_side_logits.append(side_logits.cpu().numpy()[mask]); val_side_target.append(side_target[mask])
        logits_np=np.concatenate(val_logits); target_np=np.concatenate(val_target)
        auc=float(roc_auc_score(target_np, logits_np)); side_auc=float(roc_auc_score(np.concatenate(val_side_target),np.concatenate(val_side_logits))); row={"epoch":epoch+1,"loss":float(np.mean(losses)),"validation_dominance_auc":auc,"validation_side_auc":side_auc,"head_only":head_only}; history.append(row); print(json.dumps(row),flush=True)
        if side_auc > best_auc:
            best_auc=side_auc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert best_state is not None; model.load_state_dict(best_state); model.to(device); model.eval()
    val_logits=[]; val_target=[]; val_side_logits=[]; val_side_target=[]
    with torch.no_grad():
        for begin in range(0,len(validation_events),config.batch_size):
            events=validation_events[begin:begin+config.batch_size]; x,target,side_target,mask,lengths=_make_batch(data,events,normalizer)
            _,logits,side_logits=_packed_heads(model,torch.from_numpy(x).to(device),lengths); val_logits.append(logits.cpu().numpy()[mask]); val_target.append(target[mask]); val_side_logits.append(side_logits.cpu().numpy()[mask]); val_side_target.append(side_target[mask])
    temperature=_fit_temperature(np.concatenate(val_logits),np.concatenate(val_target))
    side_temperature=_fit_temperature(np.concatenate(val_side_logits),np.concatenate(val_side_target))
    evaluation=evaluate_swap_grid(model,data,normalizer,config,temperature,side_temperature,device)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":asdict(config),"open_config":asdict(open_config),"feature_names":data.feature_names,"normalizer":normalizer.to_dict(),"dominance_temperature":temperature,"side_temperature":side_temperature,"evaluation":evaluation},out/"final.pt")
    report={"device":str(device),"best_validation_side_auc":best_auc,"dominance_temperature":temperature,"side_temperature":side_temperature,"history":history,"evaluation":evaluation}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

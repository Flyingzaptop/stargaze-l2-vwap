"""Sequence model for symmetric PRICE-vs-VWAP dominance."""

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
from .l2_causal_rate import CausalRateConfig, causal_rate_select, direction_and_score, robust_validation_score, summarize_selected
from .l2_dominance_model import apply_price_dominance_veto, dominance_target
from .l2_dominance_swap import _eligible_single_cross_events, _iter_batches
from .l2_open_policy import L2OpenPolicy
from .l2_open_reinforce import OpenReinforceConfig, PreparedOpenData, _event_indices
from .l2_profit_direction import ProfitDirectionConfig, _make_batch
from .l2_risk_direction import L2RiskDirectionPolicy, RiskDirectionConfig, _trade_rows


@dataclass(frozen=True)
class DominanceLSTMConfig:
    epochs: int = 15
    head_only_epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 3e-4
    positive_class_weight: float = 2.0
    distillation_weight: float = 0.25
    seed: int = 20260816


class L2DominancePolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.open_head = nn.Linear(hidden_size, 1)
        self.dominance_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.dominance_head.bias)
        nn.init.orthogonal_(self.dominance_head.weight, gain=0.1)


def _packed(model: L2DominancePolicy, x: torch.Tensor, lengths: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_padded_sequence(x, torch.from_numpy(lengths), batch_first=True, enforce_sorted=False)
    encoded, _ = model.lstm(packed)
    encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
    return model.open_head(encoded).squeeze(-1), model.dominance_head(encoded).squeeze(-1)


def _raw_local_delta(data: PreparedOpenData, events: np.ndarray, maximum: int, local_index: int) -> np.ndarray:
    result = np.zeros((len(events), maximum), dtype=np.float32)
    for row, event in enumerate(events):
        start = int(data.event_start[event]); crossing = int(data.event_crossing_1[event])
        result[row, :crossing-start] = data.x[start:crossing, local_index]
    return result


def _entry_probabilities(
    model: L2DominancePolicy,
    rows: list[dict[str, float | int]],
    data: PreparedOpenData,
    normalizer: RobustNormalizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability=[]; delta=[]; label=[]; model.eval()
    local_index=data.feature_names.index("mid_vwap_60s_minus_mid_ticks")
    with torch.no_grad():
        for row in rows:
            event=int(row["event_index"]); entry=int(row["entry_index"])
            start=int(data.event_start[event]); crossing=int(data.event_crossing_1[event])
            x=torch.from_numpy(normalizer.transform(data.x[start:crossing]))[None].to(device)
            _,logit=_packed(model,x,np.asarray([crossing-start],dtype=np.int64))
            probability.append(float(torch.sigmoid(logit[0,entry-start]).cpu()))
            delta.append(float(data.x[entry,local_index]))
            label.append(int(dominance_target(
                np.asarray([delta[-1]]),np.asarray([row["long_pnl"]]),np.asarray([row["short_pnl"]])
            )[0]))
    return np.asarray(probability),np.asarray(delta),np.asarray(label)


def train_dominance_lstm(
    prepared_path: str|Path, open_checkpoint_path: str|Path,
    risk_checkpoint_path: str|Path, rate_report_path: str|Path,
    output_dir: str|Path, config: DominanceLSTMConfig, *, device_name: str="auto",
) -> dict[str,object]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else device_name if device_name!="auto" else "cpu")
    data=PreparedOpenData(prepared_path)
    open_state=torch.load(open_checkpoint_path,map_location=device,weights_only=False)
    risk_state=torch.load(risk_checkpoint_path,map_location=device,weights_only=False)
    rate_report=json.loads(Path(rate_report_path).read_text(encoding="utf-8"))
    market=OpenReinforceConfig(**open_state["config"])
    normalizer=RobustNormalizer.from_dict(open_state["normalizer"])
    teacher=L2OpenPolicy(len(data.feature_names),market.hidden_size).to(device)
    teacher.load_state_dict(open_state["model_state"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    model=L2DominancePolicy(len(data.feature_names),market.hidden_size).to(device)
    model.lstm.load_state_dict(teacher.lstm.state_dict()); model.open_head.load_state_dict(teacher.open_head.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    profit_config=ProfitDirectionConfig()
    train_events=_eligible_single_cross_events(data,0,data.train_end)
    val_events=_eligible_single_cross_events(data,data.train_end,data.validation_end)
    local_index=data.feature_names.index("mid_vwap_60s_minus_mid_ticks")
    rng=np.random.default_rng(config.seed); history=[]; best=-np.inf; best_state=None
    for epoch in range(config.epochs):
        head_only=epoch<config.head_only_epochs
        for parameter in model.lstm.parameters(): parameter.requires_grad_(not head_only)
        for parameter in model.open_head.parameters(): parameter.requires_grad_(not head_only)
        model.train(); losses=[]
        for events in _iter_batches(train_events,config.batch_size,rng):
            x,side,_,_,weight,mask,lengths=_make_batch(data,events,normalizer,market,profit_config)
            delta=_raw_local_delta(data,events,x.shape[1],local_index)
            relation=np.sign(delta); oracle=np.where(side>0.5,1,-1)
            target=(oracle==-relation).astype(np.float32); mask &= relation!=0
            xt=torch.from_numpy(x).to(device); mt=torch.from_numpy(mask).to(device)
            open_logits,dominance_logits=_packed(model,xt,lengths)
            with torch.no_grad(): teacher_open=teacher(xt)
            yt=torch.from_numpy(target).to(device); wt=torch.from_numpy(weight).to(device)
            class_weight=torch.where(yt>0.5,torch.full_like(yt,config.positive_class_weight),torch.ones_like(yt))
            raw=nn.functional.binary_cross_entropy_with_logits(dominance_logits[mt],yt[mt],reduction="none")
            effective=wt[mt]*class_weight[mt]
            dominance_loss=(raw*effective).sum()/effective.sum().clamp_min(1e-6)
            distill=nn.functional.mse_loss(open_logits[mt],teacher_open[mt])
            loss=dominance_loss+config.distillation_weight*distill
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scores=[];targets=[];weights=[]; model.eval()
        with torch.no_grad():
            for begin in range(0,len(val_events),config.batch_size):
                events=val_events[begin:begin+config.batch_size]
                x,side,_,_,weight,mask,lengths=_make_batch(data,events,normalizer,market,profit_config)
                delta=_raw_local_delta(data,events,x.shape[1],local_index); relation=np.sign(delta)
                target=(np.where(side>0.5,1,-1)==-relation).astype(np.float32); mask &= relation!=0
                _,logit=_packed(model,torch.from_numpy(x).to(device),lengths)
                scores.append(logit.cpu().numpy()[mask]);targets.append(target[mask]);weights.append(weight[mask])
        auc=float(roc_auc_score(np.concatenate(targets),np.concatenate(scores),sample_weight=np.concatenate(weights)))
        row={"epoch":epoch+1,"loss":float(np.mean(losses)),"weighted_dominance_auc":auc,"head_only":head_only}
        history.append(row);print(json.dumps(row),flush=True)
        if auc>best:
            best=auc;best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    assert best_state is not None;model.load_state_dict(best_state);model.to(device);model.eval()

    risk_config=RiskDirectionConfig(**risk_state["config"])
    risk=L2RiskDirectionPolicy(len(data.feature_names),market.hidden_size).to(device)
    risk.load_state_dict(risk_state["model_state"]);risk.eval()
    policy=risk_state["evaluation"]["selected_on_validation"]
    mode=str(policy["mode"]);penalty=float(policy["penalty"]);field=str(policy["filter_field"]);fallback=float(policy["cutoff"])
    target_rate=int(rate_report["selected_on_validation"]["target_trades_per_day"])
    open_threshold=float(risk_state["open_threshold"])
    val_rows=_trade_rows(risk,teacher,data,normalizer,_event_indices(data,data.train_end,data.validation_end,good_only=False),open_threshold,device,market,risk_config)
    test_rows=_trade_rows(risk,teacher,data,normalizer,_event_indices(data,data.validation_end,len(data.x),good_only=False),open_threshold,device,market,risk_config)
    day_ns=86_400_000_000_000;val_days=max(len({int(row["entry_ts_ns"])//day_ns for row in val_rows}),1);expected=len(val_rows)/val_days
    rate_config=CausalRateConfig(target_trades_per_day=target_rate)
    val_selected=causal_rate_select(val_rows,mode=mode,penalty=penalty,filter_field=field,expected_candidates_per_day=expected,fallback_cutoff=fallback,config=rate_config)
    initial=[direction_and_score(row,mode=mode,penalty=penalty,filter_field=field)[1] for row in val_rows]
    test_selected=causal_rate_select(test_rows,mode=mode,penalty=penalty,filter_field=field,expected_candidates_per_day=expected,fallback_cutoff=fallback,config=rate_config,initial_scores=initial)
    vp,vd,vy=_entry_probabilities(model,val_selected,data,normalizer,device);tp,td,ty=_entry_probabilities(model,test_selected,data,normalizer,device)
    baseline=summarize_selected(val_selected);grid=[{"threshold":None,"selection_score":robust_validation_score(baseline),**baseline}]
    for threshold in np.linspace(0.5,0.95,19):
        changed=apply_price_dominance_veto(val_selected,vp,vd,float(threshold));metrics=summarize_selected(changed)
        grid.append({"threshold":float(threshold),"selection_score":robust_validation_score(metrics),**metrics})
    selected=max(grid,key=lambda row:float(row["selection_score"]))
    fixed_rows=test_selected if selected["threshold"] is None else apply_price_dominance_veto(test_selected,tp,td,float(selected["threshold"]))
    report={"device":str(device),"config":asdict(config),"history":history,"best_weighted_dominance_auc":best,
            "entry_validation_auc":float(roc_auc_score(vy,vp)),"entry_test_auc_diagnostic":float(roc_auc_score(ty,tp)),
            "selected_on_validation":selected,"fixed_test":summarize_selected(fixed_rows),"fixed_test_trades":fixed_rows}
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":asdict(config),"normalizer":normalizer.to_dict(),"evaluation":report},output/"final.pt")
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

"""Tail-risk-aware direction policy for VWAP excursion entries."""

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
from .l2_profit_direction import ProfitDirectionConfig, _make_batch, executable_side_pnls


@dataclass(frozen=True)
class RiskDirectionConfig:
    epochs: int = 15
    head_only_epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 3e-4
    pnl_scale_ticks: float = 100.0
    tail_threshold_ticks: float = 300.0
    min_side_weight: float = 0.25
    max_side_weight: float = 10.0
    value_weight: float = 0.25
    tail_weight: float = 0.5
    opportunity_weight: float = 0.25
    distillation_weight: float = 0.25
    seed: int = 20260808


class L2RiskDirectionPolicy(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.open_head = nn.Linear(hidden_size, 1)
        self.side_head = nn.Linear(hidden_size, 1)
        self.long_value_head = nn.Linear(hidden_size, 1)
        self.short_value_head = nn.Linear(hidden_size, 1)
        self.long_tail_head = nn.Linear(hidden_size, 1)
        self.short_tail_head = nn.Linear(hidden_size, 1)
        self.opportunity_head = nn.Linear(hidden_size, 1)
        for head in (
            self.side_head, self.long_value_head, self.short_value_head,
            self.long_tail_head, self.short_tail_head, self.opportunity_head,
        ):
            nn.init.zeros_(head.bias)
            nn.init.orthogonal_(head.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded, _ = self.lstm(x)
        return tuple(head(encoded).squeeze(-1) for head in (
            self.open_head, self.side_head, self.long_value_head,
            self.short_value_head, self.long_tail_head,
            self.short_tail_head, self.opportunity_head,
        ))


def _packed_heads(
    model: L2RiskDirectionPolicy, x: torch.Tensor, lengths: np.ndarray
) -> tuple[torch.Tensor, ...]:
    packed = pack_padded_sequence(
        x, torch.from_numpy(lengths), batch_first=True, enforce_sorted=False
    )
    encoded, _ = model.lstm(packed)
    encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
    return tuple(head(encoded).squeeze(-1) for head in (
        model.open_head, model.side_head, model.long_value_head,
        model.short_value_head, model.long_tail_head,
        model.short_tail_head, model.opportunity_head,
    ))


def _auc(target: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    if len(np.unique(target)) < 2:
        return 0.5
    return float(roc_auc_score(target, score, sample_weight=weight))


def _trade_rows(
    model: L2RiskDirectionPolicy, teacher: L2OpenPolicy,
    data: PreparedOpenData, normalizer: RobustNormalizer, events: np.ndarray,
    open_threshold: float, device: torch.device,
    market_config: OpenReinforceConfig, config: RiskDirectionConfig,
) -> list[dict[str, float]]:
    entries = _open_entries(teacher, data, normalizer, events, open_threshold, device)
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for event, entry in entries.items():
            start=int(data.event_start[event]); crossing=int(data.event_crossing_1[event])
            x=torch.from_numpy(normalizer.transform(data.x[start:crossing]))[None].to(device)
            outputs=model(x); offset=int(entry)-start
            side=float(torch.sigmoid(outputs[1][0,offset]).cpu())
            pred_long=float(np.sinh(float(outputs[2][0,offset].cpu()))*config.pnl_scale_ticks)
            pred_short=float(np.sinh(float(outputs[3][0,offset].cpu()))*config.pnl_scale_ticks)
            long_tail=float(torch.sigmoid(outputs[4][0,offset]).cpu())
            short_tail=float(torch.sigmoid(outputs[5][0,offset]).cpu())
            opportunity=float(torch.sigmoid(outputs[6][0,offset]).cpu())
            long_pnl,short_pnl=executable_side_pnls(data,int(entry),crossing,market_config)
            rows.append({
                "long_pnl":float(long_pnl[0]),"short_pnl":float(short_pnl[0]),
                "side_probability":side,"predicted_long_pnl":pred_long,
                "predicted_short_pnl":pred_short,"long_tail_probability":long_tail,
                "short_tail_probability":short_tail,
                "opportunity_probability":opportunity,
            })
    return rows


def _summarize(
    rows: list[dict[str,float]], *, mode: str, penalty: float,
    filter_field: str, cutoff: float,
) -> dict[str,float]:
    chosen=[]
    for row in rows:
        if mode=="classifier": side=1 if row["side_probability"]>=0.5 else -1
        elif mode=="value": side=1 if row["predicted_long_pnl"]>=row["predicted_short_pnl"] else -1
        elif mode=="risk":
            long_score=row["predicted_long_pnl"]-penalty*row["long_tail_probability"]
            short_score=row["predicted_short_pnl"]-penalty*row["short_tail_probability"]
            side=1 if long_score>=short_score else -1
        else: raise ValueError("unknown mode")
        tail=row["long_tail_probability"] if side>0 else row["short_tail_probability"]
        predicted=row["predicted_long_pnl"] if side>0 else row["predicted_short_pnl"]
        risk_edge=predicted-penalty*tail
        score={"opportunity_probability":row["opportunity_probability"],
               "negative_tail_probability":-tail,"risk_edge":risk_edge}[filter_field]
        if score>=cutoff:
            pnl=row["long_pnl"] if side>0 else row["short_pnl"]
            chosen.append((pnl,max(row["long_pnl"],row["short_pnl"]),tail))
    if not chosen:
        return {"trades":0,"mean_pnl_ticks":0.0,"median_pnl_ticks":0.0,
                "win_rate":0.0,"p05_pnl_ticks":0.0,"oracle_win_rate":0.0,
                "mean_tail_probability":0.0}
    values=np.asarray(chosen)
    return {"trades":int(len(values)),"mean_pnl_ticks":float(values[:,0].mean()),
            "median_pnl_ticks":float(np.median(values[:,0])),
            "win_rate":float((values[:,0]>0).mean()),
            "p05_pnl_ticks":float(np.quantile(values[:,0],0.05)),
            "total_pnl_ticks":float(values[:,0].sum()),
            "oracle_win_rate":float((values[:,1]>0).mean()),
            "mean_tail_probability":float(values[:,2].mean())}


def train_risk_direction(
    prepared_path: str|Path, open_checkpoint_path: str|Path,
    output_dir: str|Path, config: RiskDirectionConfig, *, device_name: str="auto",
) -> dict[str,object]:
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else device_name if device_name!="auto" else "cpu")
    data=PreparedOpenData(prepared_path)
    checkpoint=torch.load(Path(open_checkpoint_path).resolve(strict=True),map_location=device,weights_only=False)
    market=OpenReinforceConfig(**checkpoint["config"]); normalizer=RobustNormalizer.from_dict(checkpoint["normalizer"])
    teacher=L2OpenPolicy(len(data.feature_names),market.hidden_size).to(device); teacher.load_state_dict(checkpoint["model_state"]); teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    model=L2RiskDirectionPolicy(len(data.feature_names),market.hidden_size).to(device)
    model.lstm.load_state_dict(teacher.lstm.state_dict()); model.open_head.load_state_dict(teacher.open_head.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    profit_config=ProfitDirectionConfig(pnl_scale_ticks=config.pnl_scale_ticks,
        min_side_weight=config.min_side_weight,max_side_weight=config.max_side_weight)
    train_events=_eligible_single_cross_events(data,0,data.train_end)
    val_events=_eligible_single_cross_events(data,data.train_end,data.validation_end)
    rng=np.random.default_rng(config.seed); history=[]; best=-np.inf; best_state=None
    for epoch in range(config.epochs):
        head_only=epoch<config.head_only_epochs
        for p in model.lstm.parameters(): p.requires_grad_(not head_only)
        for p in model.open_head.parameters(): p.requires_grad_(not head_only)
        model.train(); losses=[]
        for events in _iter_batches(train_events,config.batch_size,rng):
            x,side,lv,sv,weight,mask,lengths=_make_batch(data,events,normalizer,market,profit_config)
            if not np.any(mask): continue
            xt=torch.from_numpy(x).to(device); mt=torch.from_numpy(mask).to(device)
            out=_packed_heads(model,xt,lengths)
            with torch.no_grad(): teacher_open=teacher(xt)
            yt=torch.from_numpy(side).to(device); wt=torch.from_numpy(weight).to(device)
            lvt=torch.from_numpy(lv).to(device); svt=torch.from_numpy(sv).to(device)
            long_pnl=torch.sinh(lvt)*config.pnl_scale_ticks; short_pnl=torch.sinh(svt)*config.pnl_scale_ticks
            long_tail=(long_pnl<=-config.tail_threshold_ticks).float(); short_tail=(short_pnl<=-config.tail_threshold_ticks).float()
            opportunity=(torch.maximum(long_pnl,short_pnl)>0).float()
            side_raw=nn.functional.binary_cross_entropy_with_logits(out[1][mt],yt[mt],reduction="none")
            side_loss=(side_raw*wt[mt]).sum()/wt[mt].sum().clamp_min(1e-6)
            value_loss=0.5*(nn.functional.mse_loss(out[2][mt],lvt[mt])+nn.functional.mse_loss(out[3][mt],svt[mt]))
            long_severity=1.0+long_tail*torch.clamp((-long_pnl-config.tail_threshold_ticks)/config.tail_threshold_ticks,0,5)
            short_severity=1.0+short_tail*torch.clamp((-short_pnl-config.tail_threshold_ticks)/config.tail_threshold_ticks,0,5)
            tail_loss=0.5*((nn.functional.binary_cross_entropy_with_logits(out[4][mt],long_tail[mt],reduction="none")*long_severity[mt]).mean()+(nn.functional.binary_cross_entropy_with_logits(out[5][mt],short_tail[mt],reduction="none")*short_severity[mt]).mean())
            opp_weight=torch.where(opportunity>0,torch.ones_like(opportunity),torch.full_like(opportunity,3.0))
            opp_loss=(nn.functional.binary_cross_entropy_with_logits(out[6][mt],opportunity[mt],reduction="none")*opp_weight[mt]).mean()
            distill=nn.functional.mse_loss(out[0][mt],teacher_open[mt])
            loss=side_loss+config.value_weight*value_loss+config.tail_weight*tail_loss+config.opportunity_weight*opp_loss+config.distillation_weight*distill
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); arrays=[[] for _ in range(7)]
        with torch.no_grad():
            for begin in range(0,len(val_events),config.batch_size):
                events=val_events[begin:begin+config.batch_size]
                x,side,lv,sv,weight,mask,lengths=_make_batch(data,events,normalizer,market,profit_config)
                out=_packed_heads(model,torch.from_numpy(x).to(device),lengths)
                lp=np.sinh(lv)*config.pnl_scale_ticks; sp=np.sinh(sv)*config.pnl_scale_ticks
                targets=(side,(lp<=-config.tail_threshold_ticks).astype(np.float32),(sp<=-config.tail_threshold_ticks).astype(np.float32),(np.maximum(lp,sp)>0).astype(np.float32),weight)
                arrays[0].append(out[1].cpu().numpy()[mask]); arrays[1].append(out[4].cpu().numpy()[mask]); arrays[2].append(out[5].cpu().numpy()[mask]); arrays[3].append(out[6].cpu().numpy()[mask])
                arrays[4].append(targets[0][mask]); arrays[5].append(targets[1][mask]); arrays[6].append(targets[2][mask])
                if begin==0: opp_target=[]; side_weight=[]
                opp_target.append(targets[3][mask]); side_weight.append(targets[4][mask])
        side_auc=_auc(np.concatenate(arrays[4]),np.concatenate(arrays[0]),np.concatenate(side_weight))
        long_tail_auc=_auc(np.concatenate(arrays[5]),np.concatenate(arrays[1])); short_tail_auc=_auc(np.concatenate(arrays[6]),np.concatenate(arrays[2])); opp_auc=_auc(np.concatenate(opp_target),np.concatenate(arrays[3]))
        score=side_auc+0.25*(long_tail_auc+short_tail_auc)+0.1*opp_auc
        row={"epoch":epoch+1,"loss":float(np.mean(losses)),"weighted_side_auc":side_auc,"long_tail_auc":long_tail_auc,"short_tail_auc":short_tail_auc,"opportunity_auc":opp_auc,"selection_score":score,"head_only":head_only}
        history.append(row); print(json.dumps(row),flush=True)
        if score>best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert best_state is not None; model.load_state_dict(best_state); model.to(device); model.eval()
    threshold=float(checkpoint["validation"]["best"]["threshold"]); splits={}
    for name,left,right in (("validation",data.train_end,data.validation_end),("test",data.validation_end,len(data.x))):
        splits[name]=_trade_rows(model,teacher,data,normalizer,_event_indices(data,left,right,good_only=False),threshold,device,market,config)
    validation=splits["validation"]; test=splits["test"]; grid=[]
    for mode in ("classifier","value","risk"):
        for penalty in ((0.0,) if mode!="risk" else (100.0,300.0,600.0,1000.0)):
            for field in ("opportunity_probability","negative_tail_probability","risk_edge"):
                scored=[]
                for row in validation:
                    if mode=="classifier": side=1 if row["side_probability"]>=0.5 else -1
                    elif mode=="value": side=1 if row["predicted_long_pnl"]>=row["predicted_short_pnl"] else -1
                    else:
                        side=1 if row["predicted_long_pnl"]-penalty*row["long_tail_probability"]>=row["predicted_short_pnl"]-penalty*row["short_tail_probability"] else -1
                    tail=row["long_tail_probability"] if side>0 else row["short_tail_probability"]
                    pred=row["predicted_long_pnl"] if side>0 else row["predicted_short_pnl"]
                    scored.append({"opportunity_probability":row["opportunity_probability"],"negative_tail_probability":-tail,"risk_edge":pred-penalty*tail}[field])
                for q in (0.0,0.5,0.75,0.9,0.95):
                    cutoff=-np.inf if q==0 else float(np.quantile(scored,q))
                    grid.append({"mode":mode,"penalty":penalty,"filter_field":field,"cutoff":cutoff,**_summarize(validation,mode=mode,penalty=penalty,filter_field=field,cutoff=cutoff)})
    eligible=[r for r in grid if int(r["trades"])>=60]; selected=max(eligible or grid,key=lambda r:float(r["mean_pnl_ticks"]))
    fixed={"mode":selected["mode"],"penalty":selected["penalty"],"filter_field":selected["filter_field"],"cutoff":selected["cutoff"],**_summarize(test,mode=str(selected["mode"]),penalty=float(selected["penalty"]),filter_field=str(selected["filter_field"]),cutoff=float(selected["cutoff"]))}
    report={
        "device":str(device),
        "config":asdict(config),
        "best_selection_score":best,
        "history":history,
        "validation_grid":grid,
        "selected_on_validation":selected,
        "fixed_test":fixed,
    }
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); torch.save({"model_state":model.state_dict(),"config":asdict(config),"market_config":asdict(market),"normalizer":normalizer.to_dict(),"open_threshold":threshold,"evaluation":report},out/"final.pt"); (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

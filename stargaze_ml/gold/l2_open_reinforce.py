"""Event-level REINFORCE for the single-output open-only policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from stargaze_ml.training.data import RobustNormalizer
from .l2_open_policy import L2OpenPolicy, exploration_probability


@dataclass(frozen=True)
class OpenReinforceConfig:
    hidden_size: int = 96
    epochs: int = 30
    warmup_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 3e-4
    entropy_start: float = 0.002
    entropy_peak: float = 0.02
    entropy_end: float = 0.001
    temperature_start: float = 1.0
    temperature_peak: float = 1.5
    temperature_end: float = 0.8
    floor_start: float = 0.02
    floor_peak: float = 0.25
    floor_end: float = 0.005
    tick_size: float = 0.01
    commission_per_fill_ticks: float = 15.0
    slippage_per_fill_ticks: float = 1.0
    seed: int = 20260804
    reward_mode: str = "event_side"


def schedule(start: float, peak: float, end: float, epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return start + (peak - start) * ((epoch + 1) / warmup)
    span = max(total - warmup, 1)
    return peak + (end - peak) * ((epoch - warmup + 1) / span)


class PreparedOpenData:
    def __init__(self, path: str | Path) -> None:
        with np.load(Path(path).resolve(strict=True), allow_pickle=False) as p:
            for name in p.files:
                setattr(self, name, np.ascontiguousarray(p[name]))
        self.train_end = int(np.asarray(self.train_end).item())
        self.validation_end = int(np.asarray(self.validation_end).item())
        self.feature_names = tuple(str(x) for x in self.feature_names.tolist())


def _event_indices(data: PreparedOpenData, left: int, right: int, *, good_only: bool) -> np.ndarray:
    mask = (
        (data.event_start >= left) & (data.event_crossing_2 + 1 < right)
        & data.event_gated & (data.event_gate_index >= data.event_start)
    )
    if good_only:
        mask &= data.event_good
    return np.flatnonzero(mask)


def _pnl_ticks(data: PreparedOpenData, events: np.ndarray, entry_decision: np.ndarray, crossing: int, config: OpenReinforceConfig) -> np.ndarray:
    result = np.zeros(len(events), dtype=np.float32)
    active = entry_decision >= 0
    if not np.any(active):
        return result
    ev = events[active]; entry_exec = entry_decision[active] + 1
    exit_exec = (data.event_crossing_1[ev] if crossing == 1 else data.event_crossing_2[ev]) + 1
    side = data.event_side[ev]
    long_pnl = data.first_bid[exit_exec] - data.first_ask[entry_exec]
    short_pnl = data.first_bid[entry_exec] - data.first_ask[exit_exec]
    if config.reward_mode == "event_side":
        gross = np.where(side < 0, long_pnl, short_pnl) / config.tick_size
    elif config.reward_mode == "oracle_best":
        gross = np.maximum(long_pnl, short_pnl) / config.tick_size
    else:
        raise ValueError("reward_mode must be 'event_side' or 'oracle_best'")
    result[active] = gross.astype(np.float32) - 2.0 * (
        config.commission_per_fill_ticks + config.slippage_per_fill_ticks
    )
    return result


def evaluate_thresholds(
    model: L2OpenPolicy, data: PreparedOpenData, events: np.ndarray,
    normalizer: RobustNormalizer, device: torch.device, config: OpenReinforceConfig,
) -> dict[str, object]:
    model.eval(); logits_by_event: list[np.ndarray] = []
    with torch.no_grad():
        for ev in events:
            left=int(data.event_start[ev]); right=int(data.event_crossing_1[ev])
            x=normalizer.transform(data.x[left:right])[None]
            logits_by_event.append(model(torch.from_numpy(x).to(device)).cpu().numpy()[0])
    candidates = np.unique(np.r_[np.geomspace(0.001, 0.05, 9), np.linspace(0.1, 0.9, 9)])
    rows=[]
    for threshold in candidates:
        entries=np.full(len(events),-1,dtype=np.int64)
        for j,ev in enumerate(events):
            left=int(data.event_start[ev]); gate=int(data.event_gate_index[ev]); offset=gate-left
            p=1/(1+np.exp(-np.clip(logits_by_event[j][offset:],-30,30)))
            hit=np.flatnonzero(p>=threshold)
            if hit.size: entries[j]=gate+int(hit[0])
        p1=_pnl_ticks(data,events,entries,1,config); p2=_pnl_ticks(data,events,entries,2,config)
        traded=entries>=0
        rows.append({"threshold":float(threshold),"trades":int(traded.sum()),
                     "mean_pnl_cross1":float(p1[traded].mean()) if traded.any() else 0.0,
                     "mean_pnl_cross2":float(p2[traded].mean()) if traded.any() else 0.0,
                     "total_pnl_cross1":float(p1.sum()),"total_pnl_cross2":float(p2.sum())})
    eligible=[r for r in rows if r["trades"]>=max(20,int(len(events)*0.05))]
    best=max(eligible or rows,key=lambda r:(r["mean_pnl_cross1"]+r["mean_pnl_cross2"])/2)
    return {"best":best,"grid":rows}


def train_open_policy(
    prepared: str | Path, output_dir: str | Path, config: OpenReinforceConfig,
    *, device_name: str = "auto",
) -> dict[str, object]:
    if config.reward_mode not in {"event_side", "oracle_best"}:
        raise ValueError("reward_mode must be 'event_side' or 'oracle_best'")
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else device_name if device_name!="auto" else "cpu")
    data=PreparedOpenData(prepared)
    train_mask=data.valid_feature & (np.arange(len(data.x))<data.train_end)
    normalizer=RobustNormalizer.fit(data.x,train_mask,clip=12.0)
    model=L2OpenPolicy(len(data.feature_names),config.hidden_size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    rng=np.random.default_rng(config.seed); generator=torch.Generator(device=device).manual_seed(config.seed)
    history=[]
    for epoch in range(config.epochs):
        good_only=epoch<config.warmup_epochs
        events=_event_indices(data,0,data.train_end,good_only=good_only)
        rng.shuffle(events)
        entropy_coef=schedule(config.entropy_start,config.entropy_peak,config.entropy_end,epoch,config.warmup_epochs,config.epochs)
        temperature=schedule(config.temperature_start,config.temperature_peak,config.temperature_end,epoch,config.warmup_epochs,config.epochs)
        floor=schedule(config.floor_start,config.floor_peak,config.floor_end,epoch,config.warmup_epochs,config.epochs)
        rewards=[]; opens=0
        model.train()
        for begin in range(0,len(events),config.batch_size):
            ev=events[begin:begin+config.batch_size]
            lengths=(data.event_crossing_1[ev]-data.event_start[ev]).astype(np.int64)
            max_len=int(lengths.max()); bx=np.zeros((len(ev),max_len,data.x.shape[1]),np.float32)
            allowed=np.zeros((len(ev),max_len),bool)
            for j,e in enumerate(ev):
                left=int(data.event_start[e]); length=int(lengths[j]); gate=int(data.event_gate_index[e])-left
                bx[j,:length]=normalizer.transform(data.x[left:left+length])
                allowed[j,gate:length]=data.valid_feature[left+gate:left+length] & data.observed[left+gate+1:left+length+1]
            xt=torch.from_numpy(bx).to(device)
            packed=pack_padded_sequence(xt,torch.from_numpy(lengths),batch_first=True,enforce_sorted=False)
            encoded,_=model.lstm(packed); encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=max_len)
            logits=model.open_head(encoded).squeeze(-1)
            probs=exploration_probability(logits,temperature=temperature,random_action_floor=floor)
            allowed_t=torch.from_numpy(allowed).to(device); active=torch.ones(len(ev),dtype=torch.bool,device=device)
            entry=np.full(len(ev),-1,dtype=np.int64); logsum=torch.zeros(len(ev),device=device); entsum=torch.zeros(len(ev),device=device); decisions=torch.zeros(len(ev),device=device)
            for t in range(max_len):
                decision=active & allowed_t[:,t]
                if not decision.any(): continue
                p=probs[:,t].clamp(1e-6,1-1e-6); draw=torch.rand(len(ev),device=device,generator=generator)<p; opened=decision&draw
                logsum += torch.where(decision,torch.where(draw,p.log(),torch.log1p(-p)),torch.zeros_like(p))
                entsum += torch.where(decision,-p*p.log()-(1-p)*torch.log1p(-p),torch.zeros_like(p)); decisions += decision
                for j in torch.nonzero(opened, as_tuple=False).flatten().cpu().tolist():
                    entry[j] = int(data.event_start[ev[j]]) + t
                active &= ~opened
            r1=_pnl_ticks(data,ev,entry,1,config); r2=_pnl_ticks(data,ev,entry,2,config); reward=(r1+r2)*0.5
            rt=torch.from_numpy(reward).to(device); advantage=(rt-rt.mean())/(rt.std(unbiased=False)+1e-6)
            entropy=(entsum/decisions.clamp_min(1)).mean(); loss=-(advantage.detach()*logsum).mean()-entropy_coef*entropy
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            rewards.extend(reward.tolist()); opens += int((entry>=0).sum())
        row={"epoch":epoch+1,"good_only":good_only,"events":len(events),"opens":opens,
             "mean_reward_ticks":float(np.mean(rewards)),"entropy":entropy_coef,"temperature":temperature,"floor":floor}
        history.append(row); print(json.dumps(row),flush=True)
    val_events=_event_indices(data,data.train_end,data.validation_end,good_only=False)
    evaluation=evaluate_thresholds(model,data,val_events,normalizer,device,config)
    test_events=_event_indices(data,data.validation_end,len(data.x),good_only=False)
    test_grid=evaluate_thresholds(model,data,test_events,normalizer,device,config)
    selected=float(evaluation["best"]["threshold"])
    fixed_test=min(test_grid["grid"],key=lambda row:abs(float(row["threshold"])-selected))
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":asdict(config),"feature_names":data.feature_names,
                "normalizer":normalizer.to_dict(),"validation":evaluation,
                "test_fixed_validation_threshold":fixed_test},out/"final.pt")
    report={"device":str(device),"history":history,"validation":evaluation,
            "test_fixed_validation_threshold":fixed_test}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

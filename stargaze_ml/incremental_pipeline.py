from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from .artifacts import load_frames, write_json
from .config import DataConfig
from .contracts import CausalFrames, VENUES
from .curve_pipeline import _load_execution_quotes, _save_execution_quotes
from .data import (
    CausalReplayBuilder,
    DatasetCatalog,
    build_execution_quote_scenarios,
    build_record_log_extension,
    load_market_state,
    rebuild_market_state,
    save_market_state,
)
from .features.state import GLOBAL_FEATURE_NAMES, MarketState


def _concatenate_frames(left: CausalFrames, right: CausalFrames) -> CausalFrames:
    if left.feature_names != right.feature_names or left.venue_feature_names != right.venue_feature_names or left.venues != right.venues:
        raise ValueError("frame schemas do not match")
    if int(right.ts_ns[0]) <= int(left.ts_ns[-1]):
        raise ValueError("extension frames overlap the base frame cache")
    return CausalFrames(
        ts_ns=np.concatenate((left.ts_ns, right.ts_ns)),
        x=np.concatenate((left.x, right.x), axis=0),
        venue_x=np.concatenate((left.venue_x, right.venue_x), axis=0),
        bid=np.concatenate((left.bid, right.bid), axis=0),
        ask=np.concatenate((left.ask, right.ask), axis=0),
        valid=np.concatenate((left.valid, right.valid)),
        segment_id=np.concatenate((left.segment_id, right.segment_id)),
        feature_names=left.feature_names,
        venue_feature_names=left.venue_feature_names,
        venues=left.venues,
    )


def _audit_checkpoint_bbo(frames: CausalFrames, state: MarketState) -> dict[str, object]:
    rows: dict[str, object] = {}
    all_match = True
    for index, venue in enumerate(VENUES):
        frame_bid, frame_ask = float(frames.bid[-1, index]), float(frames.ask[-1, index])
        state_bid, state_ask = state.books[venue].bbo()
        match = bool(
            np.isclose(frame_bid, state_bid, rtol=0.0, atol=1e-9, equal_nan=True)
            and np.isclose(frame_ask, state_ask, rtol=0.0, atol=1e-9, equal_nan=True)
        )
        all_match &= match
        rows[venue] = {
            "frame_bid": frame_bid,
            "frame_ask": frame_ask,
            "state_bid": float(state_bid),
            "state_ask": float(state_ask),
            "match": match,
        }
    result = {"cutoff_ns": int(frames.ts_ns[-1]), "all_match": all_match, "venues": rows}
    if not all_match:
        raise RuntimeError("market-state checkpoint BBO does not match the base frame cutoff")
    return result


def prepare_incremental_data(args: object) -> dict[str, object]:
    started = perf_counter()
    old_root = Path(args.old_raw_dir)
    live_root = Path(args.live_raw_dir)
    destination = Path(args.out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_frames = load_frames(Path(args.base_frames))
    cutoff_ns = int(base_frames.ts_ns[-1])
    extension_root = destination / "raw_extension"

    extension_manifest_path = destination / "extension_manifest.json"
    if extension_manifest_path.exists() and extension_root.exists():
        extension_manifest = json.loads(extension_manifest_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps({"stage": "extract_extension", "cutoff_ns": cutoff_ns}), flush=True)
        extension_manifest = build_record_log_extension(
            old_root,
            live_root,
            extension_root,
            after_ts_ns=cutoff_ns,
        )
        write_json(extension_manifest_path, extension_manifest)
    old_catalog = DatasetCatalog.discover(old_root)
    extension_catalog = DatasetCatalog.discover(extension_root)

    state_path = destination / "market_state_at_cutoff.pkl"
    if state_path.exists():
        state = load_market_state(state_path)
    else:
        print(json.dumps({"stage": "rebuild_market_state", "workers": int(args.workers)}), flush=True)
        venue_width = base_frames.venue_x.shape[1] * base_frames.venue_x.shape[2]
        consensus_column = venue_width + GLOBAL_FEATURE_NAMES.index("consensus_mid")
        history = base_frames.x[-10_000:, consensus_column].astype(float).tolist()
        state = rebuild_market_state(
            old_catalog,
            end_ts_ns=cutoff_ns,
            cadence_ms=int(args.cadence_ms),
            segment_id=int(base_frames.segment_id[-1]),
            mid_history=history,
            workers=int(args.workers),
        )
        save_market_state(state_path, state)
    write_json(destination / "checkpoint_bbo_audit.json", _audit_checkpoint_bbo(base_frames, state))

    cadence_ns = int(args.cadence_ms) * 1_000_000
    start_ns = cutoff_ns + cadence_ns
    end_ns = int(extension_catalog.common_end_ns // cadence_ns * cadence_ns)
    extension_frames_path = destination / "extension_frames.npz"
    if extension_frames_path.exists():
        frames = load_frames(extension_frames_path)
        if int(frames.ts_ns[0]) != start_ns or int(frames.ts_ns[-1]) != end_ns:
            raise ValueError("cached extension frames do not match the immutable extension interval")
    else:
        print(json.dumps({"stage": "extension_frames", "start_ns": start_ns, "end_ns": end_ns}), flush=True)
        last_percent = [-1]

        def progress(done: int, total: int) -> None:
            percent = int(100 * done / max(total, 1))
            if percent != last_percent[0] and (percent % 10 == 0 or done == total):
                print(json.dumps({"stage": "extension_frames", "done": done, "total": total, "percent": percent}), flush=True)
                last_percent[0] = percent

        frames = CausalReplayBuilder(
            extension_catalog,
            DataConfig(raw_dir=extension_root, cadence_ms=int(args.cadence_ms), max_stale_ms=int(args.max_stale_ms)),
        ).build(
            start_ts_ns=start_ns,
            end_ts_ns=end_ns,
            initial_state=load_market_state(state_path),
            progress=progress,
        )
        frames.save(extension_frames_path, metadata={"cutoff_ns": cutoff_ns, "catalog": extension_catalog.manifest()})
    extended_frames_path = destination / "extended_frames.npz"
    if extended_frames_path.exists():
        extended = load_frames(extended_frames_path)
        if len(extended.ts_ns) != len(base_frames.ts_ns) + len(frames.ts_ns) or int(extended.ts_ns[-1]) != end_ns:
            raise ValueError("cached extended frames do not align with base and extension caches")
    else:
        extended = _concatenate_frames(base_frames, frames)
        extended.save(extended_frames_path, metadata={"base_frames": str(Path(args.base_frames).resolve()), "cutoff_ns": cutoff_ns})

    print(json.dumps({"stage": "extension_execution_quotes"}), flush=True)
    extension_execution = build_execution_quote_scenarios(
        extension_catalog,
        frames.ts_ns,
        latencies_ms=(100.0, 250.0, 500.0),
        notional_usd=float(args.notional_usd),
        max_stale_ms=float(args.max_stale_ms),
        initial_state=load_market_state(state_path),
    )
    _save_execution_quotes(destination / "extension_execution_quotes.npz", extension_execution)
    base_execution = _load_execution_quotes(Path(args.base_execution))
    combined = {
        latency: type(quotes)(
            latency,
            np.concatenate((base_execution[latency].bid, quotes.bid)),
            np.concatenate((base_execution[latency].ask, quotes.ask)),
            np.concatenate((base_execution[latency].valid, quotes.valid)),
        )
        for latency, quotes in extension_execution.items()
    }
    _save_execution_quotes(destination / "extended_execution_quotes.npz", combined)
    output = {
        "cutoff_ns": cutoff_ns,
        "extension_ticks": len(frames.ts_ns),
        "extension_hours": float(frames.ts_ns[-1] - frames.ts_ns[0]) / 3.6e12,
        "extended_ticks": len(extended.ts_ns),
        "end_ns": int(extended.ts_ns[-1]),
        "elapsed_seconds": perf_counter() - started,
        "config": asdict(DataConfig(raw_dir=extension_root, cadence_ms=int(args.cadence_ms), max_stale_ms=int(args.max_stale_ms))) | {"raw_dir": str(extension_root)},
    }
    write_json(destination / "incremental_summary.json", output)
    print(json.dumps({"stage": "complete", **output}), flush=True)
    return output


__all__ = ["prepare_incremental_data"]

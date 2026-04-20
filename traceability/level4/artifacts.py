"""Artifact writers for Level-4 SHARP/SMOOTH full-timeline outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .types import Level4Result, Level4SegmentSummary


def write_level4_artifacts(
    result: Level4Result,
    output_dir: Path,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_csv = output_dir / "level4_segments.csv"
    segments_json = output_dir / "level4_segments.json"
    metadata_json = output_dir / "level4_run_metadata.json"

    report_segments, trim_ctx = _build_report_segments(
        segments=result.segments,
        fs_hz=float(result.fs_hz),
        trim_start_sec=float(trim_start_sec),
        trim_end_sec=float(trim_end_sec),
    )

    _write_segments_csv(report_segments, segments_csv)
    _write_segments_json(report_segments, segments_json)
    _write_metadata_json(result, report_segments, trim_ctx, metadata_json)

    return {
        "segments_csv": segments_csv,
        "segments_json": segments_json,
        "metadata_json": metadata_json,
    }


def _write_segments_csv(report_segments: List[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["segment_id", "timesteps", "duration", "starting_time", "ending_time", "label_vector"]
        )
        for seg in report_segments:
            writer.writerow(
                [
                    seg["segment_id"],
                    seg["timesteps"],
                    seg["duration"],
                    seg["starting_time"],
                    seg["ending_time"],
                    "[" + "".join(seg["label_vector"]) + "]",
                ]
            )


def _write_segments_json(report_segments: List[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(report_segments, indent=2), encoding="utf-8")


def _write_metadata_json(
    result: Level4Result,
    report_segments: List[dict],
    trim_ctx: dict,
    output_path: Path,
) -> None:
    fs = float(result.fs_hz)
    keep_start = int(trim_ctx["keep_start_idx"])
    keep_end = int(trim_ctx["keep_end_idx"])
    effective_steps = int(keep_end - keep_start)

    counts = _count_states_from_report_segments(report_segments)
    turn_preds = list(result.event_predictions)
    sharp_vote_hist: dict[str, int] = {}
    for pred in turn_preds:
        key = str(int(pred.rule_diagnostics.get("sharp_votes_total", 0)))
        sharp_vote_hist[key] = sharp_vote_hist.get(key, 0) + 1

    payload = {
        "session_id": result.session_id,
        "num_steps": effective_steps,
        "num_segments": int(len(report_segments)),
        "source_num_steps": int(sum(seg.timesteps for seg in result.segments)),
        "source_duration_sec": float(sum(seg.timesteps for seg in result.segments) / fs),
        "effective_duration_sec": float(effective_steps / fs),
        "coverage": {
            "trimmed_start_idx": keep_start,
            "trimmed_end_idx": keep_end,
            "expected_total_steps": effective_steps,
        },
        "level1_state_counts": counts["level1_state_counts"],
        "level2_state_counts": counts["level2_state_counts"],
        "level3_state_counts": counts["level3_state_counts"],
        "level4_turn_subtype_counts_by_direction": counts["level4_turn_subtype_counts_by_direction"],
        "level4_turn_subtype_counts_total": counts["level4_turn_subtype_counts_total"],
        "turn_event_stats": {
            "num_turn_events": int(len(turn_preds)),
            "sharp_turn_events": int(sum(1 for p in turn_preds if p.turn_subtype == "SHARP")),
            "smooth_turn_events": int(sum(1 for p in turn_preds if p.turn_subtype == "SMOOTH")),
            "sharp_vote_histogram_total": {k: int(v) for k, v in sorted(sharp_vote_hist.items(), key=lambda kv: int(kv[0]))},
            "mean_duration_sec": None if not turn_preds else float(np.mean([p.duration_sec for p in turn_preds])),
            "mean_abs_yaw_mean": None
            if not turn_preds
            else float(np.mean([float(p.event_features.get("mean_abs_yaw", 0.0)) for p in turn_preds])),
        },
        "model_info": {
            "subtype_mode": "rules",
            "rule_version": result.config.rule_version,
            "threshold_source": result.config.threshold_source,
        },
        "time_alignment": {
            "trim_start_sec_input": trim_ctx["trim_start_sec_input"],
            "trim_end_sec_input": trim_ctx["trim_end_sec_input"],
            "trim_start_steps_used": trim_ctx["trim_start_steps"],
            "trim_end_steps_used": trim_ctx["trim_end_steps"],
            "trim_start_sec_used": trim_ctx["trim_start_sec_used"],
            "trim_end_sec_used": trim_ctx["trim_end_sec_used"],
            "timeline_origin": "trimmed_window_start",
        },
        "config": result.config.to_dict(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _count_states_from_report_segments(report_segments: List[dict]) -> dict:
    l1 = {"STOPPED": 0, "MOVING": 0}
    l2 = {"STOPPED": 0, "STRAIGHT": 0, "TURNING": 0}
    l3 = {
        "STOPPED": 0,
        "MOVING_STRAIGHT_ACCELERATING": 0,
        "MOVING_STRAIGHT_DECELERATING": 0,
        "MOVING_STRAIGHT_CRUISING": 0,
        "MOVING_TURNING_LEFT": 0,
        "MOVING_TURNING_RIGHT": 0,
    }
    l4_by_dir = {
        "LEFT": {"SHARP": 0, "SMOOTH": 0},
        "RIGHT": {"SHARP": 0, "SMOOTH": 0},
    }
    l4_total = {"SHARP": 0, "SMOOTH": 0}

    for seg in report_segments:
        steps = int(seg["timesteps"])
        labels = list(seg["label_vector"])
        label_set = set(labels)

        if labels == ["(STOPPED)"]:
            l1["STOPPED"] += steps
            l2["STOPPED"] += steps
            l3["STOPPED"] += steps
            continue

        l1["MOVING"] += steps

        if "(STRAIGHT)" in label_set:
            l2["STRAIGHT"] += steps
            if "(ACCELERATING)" in label_set:
                l3["MOVING_STRAIGHT_ACCELERATING"] += steps
            elif "(DECELERATING)" in label_set:
                l3["MOVING_STRAIGHT_DECELERATING"] += steps
            elif "(CRUISING)" in label_set:
                l3["MOVING_STRAIGHT_CRUISING"] += steps
            continue

        if "(TURNING)" in label_set:
            l2["TURNING"] += steps
            direction = "LEFT" if "(LEFT)" in label_set else "RIGHT"
            l3[f"MOVING_TURNING_{direction}"] += steps
            subtype = None
            for token in labels:
                if token == "(SHARP LEFT)" or token == "(SHARP RIGHT)":
                    subtype = "SHARP"
                elif token == "(SMOOTH LEFT)" or token == "(SMOOTH RIGHT)":
                    subtype = "SMOOTH"
            if subtype is not None:
                l4_by_dir[direction][subtype] += steps
                l4_total[subtype] += steps

    return {
        "level1_state_counts": l1,
        "level2_state_counts": l2,
        "level3_state_counts": l3,
        "level4_turn_subtype_counts_by_direction": l4_by_dir,
        "level4_turn_subtype_counts_total": l4_total,
    }


def _build_report_segments(
    segments: List[Level4SegmentSummary],
    fs_hz: float,
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[List[dict], dict]:
    if fs_hz <= 0:
        raise ValueError("`fs_hz` must be > 0.")
    if trim_start_sec < 0 or trim_end_sec < 0:
        raise ValueError("Trim seconds must be >= 0.")

    total_steps = int(sum(seg.timesteps for seg in segments))
    trim_start_steps = int(round(trim_start_sec * fs_hz))
    trim_end_steps = int(round(trim_end_sec * fs_hz))
    keep_start = trim_start_steps
    keep_end = total_steps - trim_end_steps
    if keep_start >= keep_end:
        raise ValueError("Invalid trim window leaves no timesteps.")

    report_segments: List[dict] = []
    for seg in segments:
        overlap_start = max(seg.start_idx, keep_start)
        overlap_end = min(seg.end_idx, keep_end)
        if overlap_end <= overlap_start:
            continue

        rebased_start = overlap_start - keep_start
        rebased_end = overlap_end - keep_start
        steps = int(rebased_end - rebased_start)
        report_segments.append(
            {
                "segment_id": int(len(report_segments)),
                "timesteps": steps,
                "duration": _format_duration_verbose(steps / fs_hz),
                "starting_time": _format_mmss_ss(rebased_start / fs_hz),
                "ending_time": _format_mmss_ss(rebased_end / fs_hz),
                "label_vector": list(seg.label_vector),
            }
        )

    trim_ctx = {
        "trim_start_sec_input": float(trim_start_sec),
        "trim_end_sec_input": float(trim_end_sec),
        "trim_start_steps": int(trim_start_steps),
        "trim_end_steps": int(trim_end_steps),
        "trim_start_sec_used": float(trim_start_steps / fs_hz),
        "trim_end_sec_used": float(trim_end_steps / fs_hz),
        "keep_start_idx": int(keep_start),
        "keep_end_idx": int(keep_end),
    }
    return report_segments, trim_ctx


def _format_duration_verbose(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d} mins {seconds:05.2f} seconds"


def _format_mmss_ss(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d}:{seconds:05.2f}"

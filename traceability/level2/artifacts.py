"""Artifact writers for Level-2 outputs."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .pipeline import L2_STOPPED, STRAIGHT, TURNING, Level2Result


def write_level2_artifacts(
    result: Level2Result,
    output_dir: Path,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
) -> Dict[str, Path]:
    """
    Write Level-2 outputs for one session.

    Files:
    - `level2_segments.csv`
    - `level2_segments.json`
    - `level2_run_metadata.json`
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_csv = output_dir / "level2_segments.csv"
    segments_json = output_dir / "level2_segments.json"
    metadata_json = output_dir / "level2_run_metadata.json"

    # Clean stale file from older schema versions so each run yields only 3 artifacts.
    legacy_timestep_csv = output_dir / "level2_timestep_labels.csv"
    if legacy_timestep_csv.exists():
        try:
            legacy_timestep_csv.unlink()
        except PermissionError:
            logging.warning(
                "Could not remove legacy artifact (permission denied): %s",
                legacy_timestep_csv.as_posix(),
            )

    report_segments, trim_context = _build_report_segments(
        result=result,
        trim_start_sec=trim_start_sec,
        trim_end_sec=trim_end_sec,
    )
    _write_segments_csv(report_segments, segments_csv)
    _write_segments_json(report_segments, segments_json)
    _write_metadata_json(result, report_segments, trim_context, metadata_json)

    return {
        "metadata_json": metadata_json,
        "segments_csv": segments_csv,
        "segments_json": segments_json,
    }


def _write_segments_csv(report_segments: List[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "segment_id",
                "timesteps",
                "duration",
                "starting_time",
                "ending_time",
                "label_vector",
            ]
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
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(report_segments, fp, indent=2)


def _write_metadata_json(
    result: Level2Result,
    report_segments: List[dict],
    trim_context: dict,
    output_path: Path,
) -> None:
    keep_slice = slice(trim_context["keep_start_idx"], trim_context["keep_end_idx"])
    effective_turn_state = result.turn_state[keep_slice]
    effective_moving_state = result.moving_state[keep_slice]
    effective_steps = int(len(effective_turn_state))

    level1_stopped_steps = int((effective_moving_state == 0).sum())
    level1_moving_steps = int((effective_moving_state == 1).sum())

    level2_stopped_steps = int((effective_turn_state == L2_STOPPED).sum())
    level2_straight_steps = int((effective_turn_state == STRAIGHT).sum())
    level2_turning_steps = int((effective_turn_state == TURNING).sum())

    moving_mask = effective_turn_state != L2_STOPPED
    moving_only_states = effective_turn_state[moving_mask]
    straight_moving_steps = int((moving_only_states == STRAIGHT).sum())
    turning_moving_steps = int((moving_only_states == TURNING).sum())

    if len(effective_turn_state) >= 2:
        prev_state = effective_turn_state[:-1]
        next_state = effective_turn_state[1:]
        straight_to_turning = int(
            ((prev_state == STRAIGHT) & (next_state == TURNING)).sum()
        )
        turning_to_straight = int(
            ((prev_state == TURNING) & (next_state == STRAIGHT)).sum()
        )
    else:
        straight_to_turning = 0
        turning_to_straight = 0

    payload = {
        "session_id": result.session_id,
        "num_steps": effective_steps,
        "num_segments": int(len(report_segments)),
        "source_num_steps": int(len(result.turn_state)),
        "source_duration_sec": float(len(result.turn_state) / result.config.fs_hz),
        "effective_duration_sec": float(effective_steps / result.config.fs_hz),
        "coverage": {
            "trimmed_start_idx": trim_context["keep_start_idx"],
            "trimmed_end_idx": trim_context["keep_end_idx"],
            "first_start_idx": 0 if report_segments else None,
            "last_end_idx": effective_steps if report_segments else None,
            "expected_total_steps": effective_steps,
        },
        "level1_state_counts": {
            "STOPPED": level1_stopped_steps,
            "MOVING": level1_moving_steps,
        },
        "level2_state_counts_total": {
            "STOPPED": level2_stopped_steps,
            "STRAIGHT": level2_straight_steps,
            "TURNING": level2_turning_steps,
        },
        "level2_state_counts_moving_only": {
            "STRAIGHT": straight_moving_steps,
            "TURNING": turning_moving_steps,
        },
        "transition_counts": {
            "STRAIGHT_TO_TURNING": straight_to_turning,
            "TURNING_TO_STRAIGHT": turning_to_straight,
        },
        "time_alignment": {
            "trim_start_sec_input": trim_context["trim_start_sec_input"],
            "trim_end_sec_input": trim_context["trim_end_sec_input"],
            "trim_start_steps_used": trim_context["trim_start_steps"],
            "trim_end_steps_used": trim_context["trim_end_steps"],
            "trim_start_sec_used": trim_context["trim_start_sec_used"],
            "trim_end_sec_used": trim_context["trim_end_sec_used"],
            "timeline_origin": "trimmed_window_start",
        },
        "config": {
            "fs_hz": result.config.fs_hz,
            "yaw_col_index": result.config.yaw_col_index,
            "steer_angle_col_index": result.config.steer_angle_col_index,
            "steer_speed_col_index": result.config.steer_speed_col_index,
            "speed_col_index": result.config.speed_col_index,
            "yaw_median_window": result.config.yaw_median_window,
            "yaw_on": result.config.yaw_on,
            "yaw_off": result.config.yaw_off,
            "steer_on_deg": result.config.steer_on_deg,
            "steer_speed_on_dps": result.config.steer_speed_on_dps,
            "assist_speed_min": result.config.assist_speed_min,
            "straight_steer_max_deg": result.config.straight_steer_max_deg,
            "persistence_on_steps": result.config.persistence_on_steps,
            "persistence_off_steps": result.config.persistence_off_steps,
            "initial_turn_state": result.config.initial_turn_state,
        },
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _build_report_segments(
    result: Level2Result,
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[List[dict], dict]:
    fs = float(result.config.fs_hz)
    total_steps = int(len(result.turn_state))

    if trim_start_sec < 0:
        raise ValueError(f"`trim_start_sec` must be >= 0, got {trim_start_sec}.")
    if trim_end_sec < 0:
        raise ValueError(f"`trim_end_sec` must be >= 0, got {trim_end_sec}.")

    trim_start_steps = _seconds_to_steps(trim_start_sec, fs)
    trim_end_steps = _seconds_to_steps(trim_end_sec, fs)
    keep_start_idx = trim_start_steps
    keep_end_idx = total_steps - trim_end_steps

    if keep_start_idx >= keep_end_idx:
        raise ValueError(
            "Invalid trim window: after applying start/end trims there are no timesteps left. "
            f"total_steps={total_steps}, trim_start_steps={trim_start_steps}, trim_end_steps={trim_end_steps}."
        )

    report_segments: List[dict] = []
    for seg in result.segments:
        overlap_start = max(seg.start_idx, keep_start_idx)
        overlap_end = min(seg.end_idx, keep_end_idx)
        if overlap_end <= overlap_start:
            continue

        rebased_start_idx = overlap_start - keep_start_idx
        rebased_end_idx = overlap_end - keep_start_idx
        timesteps = rebased_end_idx - rebased_start_idx
        duration_sec = timesteps / fs

        report_segments.append(
            {
                "segment_id": len(report_segments),
                "timesteps": int(timesteps),
                "duration": _format_duration_verbose(duration_sec),
                "starting_time": _format_mmss_ss(rebased_start_idx / fs),
                "ending_time": _format_mmss_ss(rebased_end_idx / fs),
                "label_vector": list(seg.label_vector),
            }
        )

    if not report_segments:
        raise ValueError("No segments remain after trim window projection.")

    trim_context = {
        "trim_start_sec_input": float(trim_start_sec),
        "trim_end_sec_input": float(trim_end_sec),
        "trim_start_steps": int(trim_start_steps),
        "trim_end_steps": int(trim_end_steps),
        "trim_start_sec_used": float(trim_start_steps / fs),
        "trim_end_sec_used": float(trim_end_steps / fs),
        "keep_start_idx": int(keep_start_idx),
        "keep_end_idx": int(keep_end_idx),
    }
    return report_segments, trim_context


def _seconds_to_steps(seconds: float, fs_hz: float) -> int:
    # Snap trims to nearest sample so coverage fields stay integral.
    return int(round(seconds * fs_hz))


def _format_duration_verbose(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d} mins {seconds:05.2f} seconds"


def _format_mmss_ss(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d}:{seconds:05.2f}"

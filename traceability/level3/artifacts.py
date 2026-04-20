"""Artifact writers for Level-3 outputs."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from traceability.level2 import L2_STOPPED, STRAIGHT as L2_STRAIGHT, TURNING as L2_TURNING

from .pipeline import (
    ACCELERATING,
    CRUISING,
    DECELERATING,
    LEFT,
    LON_NA,
    L3_MOVING_STRAIGHT_ACCEL,
    L3_MOVING_STRAIGHT_CRUISE,
    L3_MOVING_STRAIGHT_DECEL,
    L3_MOVING_TURNING_LEFT,
    L3_MOVING_TURNING_RIGHT,
    L3_STOPPED,
    LAT_NA,
    RIGHT,
    Level3Result,
)


def write_level3_artifacts(
    result: Level3Result,
    output_dir: Path,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
) -> Dict[str, Path]:
    """
    Write Level-3 outputs for one session.

    Files:
    - `level3_segments.csv`
    - `level3_segments.json`
    - `level3_run_metadata.json`
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_csv = output_dir / "level3_segments.csv"
    segments_json = output_dir / "level3_segments.json"
    metadata_json = output_dir / "level3_run_metadata.json"

    # Clean stale file from older schema versions so each run yields only 3 artifacts.
    legacy_timestep_csv = output_dir / "level3_timestep_labels.csv"
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
    result: Level3Result,
    report_segments: List[dict],
    trim_context: dict,
    output_path: Path,
) -> None:
    keep_slice = slice(trim_context["keep_start_idx"], trim_context["keep_end_idx"])
    effective_moving_state = result.moving_state[keep_slice]
    effective_turn_state = result.turn_state[keep_slice]
    effective_lon_state = result.lon_state[keep_slice]
    effective_lat_state = result.lat_state[keep_slice]
    effective_composite_state = result.composite_state[keep_slice]
    effective_steps = int(len(effective_composite_state))

    level1_stopped_steps = int((effective_moving_state == 0).sum())
    level1_moving_steps = int((effective_moving_state == 1).sum())

    level2_stopped_steps = int((effective_turn_state == L2_STOPPED).sum())
    level2_straight_steps = int((effective_turn_state == L2_STRAIGHT).sum())
    level2_turning_steps = int((effective_turn_state == L2_TURNING).sum())

    level3_total_counts = {
        "STOPPED": int((effective_composite_state == L3_STOPPED).sum()),
        "MOVING_STRAIGHT_ACCELERATING": int(
            (effective_composite_state == L3_MOVING_STRAIGHT_ACCEL).sum()
        ),
        "MOVING_STRAIGHT_DECELERATING": int(
            (effective_composite_state == L3_MOVING_STRAIGHT_DECEL).sum()
        ),
        "MOVING_STRAIGHT_CRUISING": int(
            (effective_composite_state == L3_MOVING_STRAIGHT_CRUISE).sum()
        ),
        "MOVING_TURNING_LEFT": int(
            (effective_composite_state == L3_MOVING_TURNING_LEFT).sum()
        ),
        "MOVING_TURNING_RIGHT": int(
            (effective_composite_state == L3_MOVING_TURNING_RIGHT).sum()
        ),
    }

    straight_mask = effective_turn_state == L2_STRAIGHT
    lon_straight = effective_lon_state[straight_mask]
    level3_moving_straight_counts = {
        "ACCELERATING": int((lon_straight == ACCELERATING).sum()),
        "DECELERATING": int((lon_straight == DECELERATING).sum()),
        "CRUISING": int((lon_straight == CRUISING).sum()),
    }

    turning_mask = effective_turn_state == L2_TURNING
    lat_turning = effective_lat_state[turning_mask]
    level3_moving_turning_counts = {
        "LEFT": int((lat_turning == LEFT).sum()),
        "RIGHT": int((lat_turning == RIGHT).sum()),
    }

    transition_counts = _compute_transition_counts(
        lon_state=effective_lon_state,
        lat_state=effective_lat_state,
    )

    payload = {
        "session_id": result.session_id,
        "num_steps": effective_steps,
        "num_segments": int(len(report_segments)),
        "source_num_steps": int(len(result.composite_state)),
        "source_duration_sec": float(len(result.composite_state) / result.config.fs_hz),
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
        "level2_state_counts": {
            "STOPPED": level2_stopped_steps,
            "STRAIGHT": level2_straight_steps,
            "TURNING": level2_turning_steps,
        },
        "level3_state_counts_total": level3_total_counts,
        "level3_state_counts_moving_straight": level3_moving_straight_counts,
        "level3_state_counts_moving_turning": level3_moving_turning_counts,
        "transition_counts": transition_counts,
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
            "pedalangle_col_index": result.config.pedalangle_col_index,
            "pedalpressure_col_index": result.config.pedalpressure_col_index,
            "steer_angle_col_index": result.config.steer_angle_col_index,
            "steer_speed_col_index": result.config.steer_speed_col_index,
            "speed_col_index": result.config.speed_col_index,
            "yaw_col_index": result.config.yaw_col_index,
            "speed_median_window": result.config.speed_median_window,
            "accel_median_window": result.config.accel_median_window,
            "yaw_median_window": result.config.yaw_median_window,
            "a_on": result.config.a_on,
            "a_off": result.config.a_off,
            "pedal_on": result.config.pedal_on,
            "brake_on": result.config.brake_on,
            "pedal_cruise_max": result.config.pedal_cruise_max,
            "brake_cruise_max": result.config.brake_cruise_max,
            "lon_persistence_steps": result.config.lon_persistence_steps,
            "yaw_deadband": result.config.yaw_deadband,
            "direction_persistence_steps": result.config.direction_persistence_steps,
            "steer_angle_bootstrap_min": result.config.steer_angle_bootstrap_min,
            "steer_speed_bootstrap_min": result.config.steer_speed_bootstrap_min,
            "initial_lon_state": result.config.initial_lon_state,
            "default_turn_direction": result.config.default_turn_direction,
        },
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _compute_transition_counts(lon_state: object, lat_state: object) -> dict:
    lon = lon_state
    lat = lat_state

    if len(lon) < 2:
        return {
            "ACCEL_TO_CRUISE": 0,
            "CRUISE_TO_ACCEL": 0,
            "DECEL_TO_CRUISE": 0,
            "CRUISE_TO_DECEL": 0,
            "LEFT_TO_RIGHT": 0,
            "RIGHT_TO_LEFT": 0,
        }

    lon_prev = lon[:-1]
    lon_next = lon[1:]
    lon_valid = (lon_prev != LON_NA) & (lon_next != LON_NA)

    lat_prev = lat[:-1]
    lat_next = lat[1:]
    lat_valid = (lat_prev != LAT_NA) & (lat_next != LAT_NA)

    return {
        "ACCEL_TO_CRUISE": int(
            (lon_valid & (lon_prev == ACCELERATING) & (lon_next == CRUISING)).sum()
        ),
        "CRUISE_TO_ACCEL": int(
            (lon_valid & (lon_prev == CRUISING) & (lon_next == ACCELERATING)).sum()
        ),
        "DECEL_TO_CRUISE": int(
            (lon_valid & (lon_prev == DECELERATING) & (lon_next == CRUISING)).sum()
        ),
        "CRUISE_TO_DECEL": int(
            (lon_valid & (lon_prev == CRUISING) & (lon_next == DECELERATING)).sum()
        ),
        "LEFT_TO_RIGHT": int(
            (lat_valid & (lat_prev == LEFT) & (lat_next == RIGHT)).sum()
        ),
        "RIGHT_TO_LEFT": int(
            (lat_valid & (lat_prev == RIGHT) & (lat_next == LEFT)).sum()
        ),
    }


def _build_report_segments(
    result: Level3Result,
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[List[dict], dict]:
    fs = float(result.config.fs_hz)
    total_steps = int(len(result.composite_state))

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

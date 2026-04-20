"""Build final per-segment outputs with metrics from Level-4 segments + CAN signals."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from traceability.level3.pipeline import centered_median, central_difference

from .types import FinalMetricsConfig


_ACCEL_LABELS = {"(MOVING)", "(STRAIGHT)", "(ACCELERATING)"}
_DECEL_LABELS = {"(MOVING)", "(STRAIGHT)", "(DECELERATING)"}
_CRUISE_LABELS = {"(MOVING)", "(STRAIGHT)", "(CRUISING)"}
_TURN_BASE_LABELS = {"(MOVING)", "(TURNING)"}
_TURN_SUBTYPE_TOKENS = {
    "(SHARP LEFT)",
    "(SHARP RIGHT)",
    "(SMOOTH LEFT)",
    "(SMOOTH RIGHT)",
}


def _to_float(value: Any) -> float:
    return float(value)


def _to_int(value: Any) -> int:
    return int(value)


def _round_metric(value: float) -> float:
    """Round metric values to 2 decimal places for final artifact readability."""
    return round(float(value), 2)


def _build_signal_views(
    session_matrix: np.ndarray,
    config: FinalMetricsConfig,
) -> dict[str, np.ndarray]:
    speed = np.asarray(session_matrix[:, config.speed_col_index], dtype=np.float64)
    yaw_abs = np.abs(np.asarray(session_matrix[:, config.yaw_col_index], dtype=np.float64))
    steer_abs = np.abs(
        np.asarray(session_matrix[:, config.steer_angle_col_index], dtype=np.float64)
    )

    # Keep acceleration definition consistent with Level-3.
    speed_smooth = centered_median(speed, window=config.speed_median_window)
    accel_raw = central_difference(speed_smooth, fs_hz=config.fs_hz)
    accel = centered_median(accel_raw, window=config.accel_median_window)

    return {
        "speed": speed,
        "yaw_abs": yaw_abs,
        "steer_abs": steer_abs,
        "accel": accel,
    }


def _compute_metrics_for_segment(
    label_vector: list[str],
    start_idx: int,
    end_idx: int,
    signals: Mapping[str, np.ndarray],
) -> dict[str, float]:
    if end_idx <= start_idx:
        return {}

    labels = set(label_vector)

    if _ACCEL_LABELS.issubset(labels):
        accel_seg = signals["accel"][start_idx:end_idx]
        return {
            "mean_accel": _round_metric(_to_float(np.mean(accel_seg))),
            "peak_accel": _round_metric(_to_float(np.max(accel_seg))),
        }

    if _DECEL_LABELS.issubset(labels):
        accel_seg = signals["accel"][start_idx:end_idx]
        return {
            "mean_decel": _round_metric(_to_float(np.mean(accel_seg))),
            "peak_decel": _round_metric(_to_float(np.min(accel_seg))),
        }

    if _CRUISE_LABELS.issubset(labels):
        speed_seg = signals["speed"][start_idx:end_idx]
        return {
            "mean_speed": _round_metric(_to_float(np.mean(speed_seg))),
        }

    is_turning = _TURN_BASE_LABELS.issubset(labels)
    has_sharp_or_smooth = any(token in labels for token in _TURN_SUBTYPE_TOKENS)
    if is_turning and has_sharp_or_smooth:
        speed_seg = signals["speed"][start_idx:end_idx]
        yaw_seg = signals["yaw_abs"][start_idx:end_idx]
        steer_seg = signals["steer_abs"][start_idx:end_idx]
        return {
            "mean_speed": _round_metric(_to_float(np.mean(speed_seg))),
            "peak_abs_yaw": _round_metric(_to_float(np.max(yaw_seg))),
            "peak_abs_steer_angle": _round_metric(_to_float(np.max(steer_seg))),
        }

    return {}


def build_final_session_outputs(
    *,
    session_id: str,
    session_matrix: np.ndarray,
    level4_segments: list[dict],
    config: FinalMetricsConfig,
    level4_timeline_start_idx: int = 0,
    source_paths: Mapping[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """
    Build final segments (with `metrics`) and run metadata for one session.

    `level4_segments` is expected to come from:
    artifacts/level4_25hz/<session_id>/level4_segments.json
    """
    config.validate()
    if session_matrix.ndim != 2:
        raise ValueError(f"`session_matrix` must be 2D, got shape={session_matrix.shape}.")
    if session_matrix.shape[1] < 6:
        raise ValueError(
            f"`session_matrix` must have >= 6 columns, got width={session_matrix.shape[1]}."
        )
    if level4_timeline_start_idx < 0:
        raise ValueError(
            "`level4_timeline_start_idx` must be >= 0, "
            f"got {level4_timeline_start_idx}."
        )
    if not isinstance(level4_segments, list):
        raise ValueError("`level4_segments` must be a list of segment rows.")

    signals = _build_signal_views(session_matrix=session_matrix, config=config)

    report_segments: list[dict] = []
    cursor = 0
    n_steps = int(session_matrix.shape[0])

    accelerating_segments = 0
    decelerating_segments = 0
    cruising_segments = 0
    turning_segments = 0
    segments_with_metrics = 0

    for idx, row in enumerate(level4_segments):
        if not isinstance(row, dict):
            raise ValueError(f"Segment row at index {idx} must be an object, got {type(row)}.")

        timesteps = _to_int(row.get("timesteps", 0))
        if timesteps <= 0:
            raise ValueError(f"Invalid `timesteps` for segment row index {idx}: {timesteps}.")

        start_idx = level4_timeline_start_idx + cursor
        end_idx = start_idx + timesteps
        cursor += timesteps

        if end_idx > n_steps:
            raise ValueError(
                "Segment coverage exceeds session length: "
                f"segment_idx={idx}, end_idx={end_idx}, session_steps={n_steps}."
            )

        label_vector = list(row.get("label_vector", []))
        metrics = _compute_metrics_for_segment(
            label_vector=label_vector,
            start_idx=start_idx,
            end_idx=end_idx,
            signals=signals,
        )

        label_set = set(label_vector)
        if _ACCEL_LABELS.issubset(label_set):
            accelerating_segments += 1
        elif _DECEL_LABELS.issubset(label_set):
            decelerating_segments += 1
        elif _CRUISE_LABELS.issubset(label_set):
            cruising_segments += 1
        elif _TURN_BASE_LABELS.issubset(label_set):
            turning_segments += 1
        if metrics:
            segments_with_metrics += 1

        report_segments.append(
            {
                "segment_id": _to_int(row.get("segment_id", idx)),
                "timesteps": timesteps,
                "duration": str(row.get("duration", "")),
                "starting_time": str(row.get("starting_time", "")),
                "ending_time": str(row.get("ending_time", "")),
                "label_vector": label_vector,
                "metrics": metrics,
            }
        )

    timeline_end_idx = level4_timeline_start_idx + cursor
    if timeline_end_idx > n_steps:
        raise ValueError(
            "Level-4 timeline window exceeds session length: "
            f"end_idx={timeline_end_idx}, session_steps={n_steps}."
        )

    metadata = {
        "session_id": str(session_id),
        "num_segments": _to_int(len(report_segments)),
        "num_steps_in_level4_timeline": _to_int(cursor),
        "session_num_steps": _to_int(n_steps),
        "coverage": {
            "timeline_start_idx": _to_int(level4_timeline_start_idx),
            "timeline_end_idx": _to_int(timeline_end_idx),
            "timeline_duration_sec": _to_float(cursor / config.fs_hz),
        },
        "metrics_summary": {
            "segments_with_metrics": _to_int(segments_with_metrics),
            "accelerating_segments": _to_int(accelerating_segments),
            "decelerating_segments": _to_int(decelerating_segments),
            "cruising_segments": _to_int(cruising_segments),
            "turning_segments": _to_int(turning_segments),
        },
        "config": config.to_dict(),
        "source_paths": dict(source_paths or {}),
    }
    return report_segments, metadata

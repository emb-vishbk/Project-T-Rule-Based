"""Rule-based Level-4 SHARP vs SMOOTH classification on Level-3 TURNING segments."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .types import (
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_VALUES,
    TURN_SUBTYPE_SHARP,
    TURN_SUBTYPE_SMOOTH,
    Level4EventPrediction,
    Level4Result,
    Level4RuleConfig,
    Level4SegmentSummary,
)


_L3_TURN_LEFT_VECTOR = ["(MOVING)", "(TURNING)", "(LEFT)"]
_L3_TURN_RIGHT_VECTOR = ["(MOVING)", "(TURNING)", "(RIGHT)"]


def infer_direction_from_level3_label_vector(label_vector: list[str]) -> str:
    if list(label_vector) == _L3_TURN_LEFT_VECTOR:
        return DIRECTION_LEFT
    if list(label_vector) == _L3_TURN_RIGHT_VECTOR:
        return DIRECTION_RIGHT
    raise ValueError(f"Not a Level-3 TURNING label vector: {label_vector!r}")


def build_level4_turn_token(direction: str, turn_subtype: str) -> str:
    if direction not in DIRECTION_VALUES:
        raise ValueError(f"Invalid direction {direction!r}.")
    if turn_subtype not in (TURN_SUBTYPE_SHARP, TURN_SUBTYPE_SMOOTH):
        raise ValueError(f"Invalid turn subtype {turn_subtype!r}.")
    return f"({turn_subtype} {direction})"


def compute_turn_event_features(
    session_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    fs_hz: float,
) -> Dict[str, float]:
    """Compute event-local turning aggressiveness features on the core segment only."""
    core = np.asarray(session_matrix[start_idx:end_idx, :6], dtype=np.float64)
    if core.ndim != 2 or core.shape[0] == 0 or core.shape[1] < 6:
        raise ValueError("Invalid event slice for Level-4 feature extraction.")

    yaw = core[:, 5]
    steer_angle = core[:, 2]
    steer_speed = core[:, 3]
    speed = core[:, 4]
    yaw_abs = np.abs(yaw)

    return {
        "duration_sec": float((end_idx - start_idx) / fs_hz),
        "peak_abs_yaw": float(np.max(yaw_abs)),
        "mean_abs_yaw": float(np.mean(yaw_abs)),
        "integrated_abs_yaw": float(np.sum(yaw_abs) / fs_hz),
        "peak_abs_steer_angle": float(np.max(np.abs(steer_angle))),
        "peak_abs_steer_speed": float(np.max(np.abs(steer_speed))),
        "speed_mean": float(np.mean(speed)),
    }


def classify_turn_subtype_rule_based(
    event_features: Mapping[str, float],
    config: Level4RuleConfig | None = None,
) -> Tuple[str, Dict[str, float | int | bool]]:
    """
    Conservative SHARP vs SMOOTH rule.

    Design intent:
    - classify SHARP only for clearly aggressive corner-like turns
    - classify all other turns (including short turns and smooth highway curves) as SMOOTH
    """
    cfg = config or Level4RuleConfig()
    cfg.validate()

    def _f(name: str) -> float:
        return float(event_features.get(name, 0.0))

    duration_sec = max(0.0, _f("duration_sec"))
    mean_abs_yaw = max(0.0, _f("mean_abs_yaw"))
    peak_abs_yaw = max(0.0, _f("peak_abs_yaw"))
    peak_abs_steer_angle = max(0.0, _f("peak_abs_steer_angle"))
    peak_abs_steer_speed = max(0.0, _f("peak_abs_steer_speed"))

    if duration_sec < cfg.min_sharp_duration_sec:
        return (
            TURN_SUBTYPE_SMOOTH,
            {
                "sharp_votes_total": 0,
                "sharp_votes_yaw": 0,
                "sharp_votes_steer": 0,
                "blocked_by_min_duration": True,
                "blocked_by_long_duration_guard": False,
                "duration_sec": duration_sec,
            },
        )

    yaw_votes = int(mean_abs_yaw >= cfg.sharp_mean_abs_yaw_min) + int(peak_abs_yaw >= cfg.sharp_peak_abs_yaw_min)
    steer_votes = int(peak_abs_steer_angle >= cfg.sharp_peak_abs_steer_angle_min) + int(
        peak_abs_steer_speed >= cfg.sharp_peak_abs_steer_speed_min
    )
    total_votes = yaw_votes + steer_votes

    long_duration_block = bool(
        duration_sec >= cfg.long_duration_guard_sec and mean_abs_yaw < cfg.long_duration_mean_abs_yaw_min
    )

    sharp_ok = total_votes >= int(cfg.sharp_vote_threshold)
    if cfg.require_yaw_vote:
        sharp_ok = sharp_ok and yaw_votes >= 1
    if cfg.require_steer_vote:
        sharp_ok = sharp_ok and steer_votes >= 1
    if long_duration_block:
        sharp_ok = False

    return (
        TURN_SUBTYPE_SHARP if sharp_ok else TURN_SUBTYPE_SMOOTH,
        {
            "sharp_votes_total": int(total_votes),
            "sharp_votes_yaw": int(yaw_votes),
            "sharp_votes_steer": int(steer_votes),
            "blocked_by_min_duration": False,
            "blocked_by_long_duration_guard": bool(long_duration_block),
            "duration_sec": float(duration_sec),
        },
    )


def run_level4(
    session_id: str,
    session_matrix: np.ndarray,
    level3_segments: Sequence[dict],
    config: Level4RuleConfig | None = None,
) -> Level4Result:
    """
    Upgrade Level-3 full-timeline segments to Level-4 by appending SHARP/SMOOTH tokens on TURNING segments.
    """
    cfg = config or Level4RuleConfig()
    cfg.validate()

    matrix = np.asarray(session_matrix)
    if matrix.ndim != 2:
        raise ValueError(f"`session_matrix` must be 2D, got shape={matrix.shape}.")
    if matrix.shape[1] < 6:
        raise ValueError(f"`session_matrix` must have >= 6 columns, got width={matrix.shape[1]}.")

    preds: List[Level4EventPrediction] = []
    upgraded_segments: List[Level4SegmentSummary] = []

    cursor = 0
    for row in level3_segments:
        seg_id = int(row["segment_id"])
        timesteps = int(row["timesteps"])
        start_idx = cursor
        end_idx = cursor + timesteps
        cursor = end_idx

        label_vector = list(row["label_vector"])
        if label_vector in (_L3_TURN_LEFT_VECTOR, _L3_TURN_RIGHT_VECTOR):
            direction = infer_direction_from_level3_label_vector(label_vector)
            features = compute_turn_event_features(
                session_matrix=matrix,
                start_idx=start_idx,
                end_idx=end_idx,
                fs_hz=cfg.fs_hz,
            )
            subtype, diagnostics = classify_turn_subtype_rule_based(features, config=cfg)
            pred = Level4EventPrediction(
                session_id=str(session_id),
                level3_segment_id=seg_id,
                direction=direction,
                turn_subtype=subtype,
                start_idx=start_idx,
                end_idx=end_idx,
                timesteps=timesteps,
                duration_sec=float(timesteps / cfg.fs_hz),
                event_features={k: float(v) for k, v in features.items()},
                rule_diagnostics=dict(diagnostics),
            )
            pred.validate()
            preds.append(pred)
            label_vector = label_vector + [build_level4_turn_token(direction=direction, turn_subtype=subtype)]

        upgraded_segments.append(
            Level4SegmentSummary(
                segment_id=seg_id,
                start_idx=start_idx,
                end_idx=end_idx,
                timesteps=timesteps,
                duration=str(row["duration"]),
                starting_time=str(row["starting_time"]),
                ending_time=str(row["ending_time"]),
                label_vector=label_vector,
            )
        )

    if cursor != matrix.shape[0]:
        raise ValueError(
            "Level-3 segments do not cover session length exactly: "
            f"coverage={cursor}, session_steps={matrix.shape[0]}."
        )

    return Level4Result(
        session_id=str(session_id),
        fs_hz=float(cfg.fs_hz),
        source_level3_segments=[dict(x) for x in level3_segments],
        segments=upgraded_segments,
        event_predictions=preds,
        config=cfg,
    )

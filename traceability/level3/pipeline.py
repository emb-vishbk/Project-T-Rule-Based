"""Pure Level-3 logic: longitudinal and lateral maneuver labeling with hierarchical composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from traceability.level2 import L2_STOPPED, STRAIGHT as L2_STRAIGHT, TURNING as L2_TURNING

# Longitudinal sub-states (for MOVING & STRAIGHT only)
LON_NA = 0
ACCELERATING = 1
DECELERATING = 2
CRUISING = 3

# Lateral direction sub-states (for MOVING & TURNING only)
LAT_NA = 0
LEFT = 1
RIGHT = 2

# Full timeline Level-3 composite states
L3_STOPPED = 0
L3_MOVING_STRAIGHT_ACCEL = 1
L3_MOVING_STRAIGHT_DECEL = 2
L3_MOVING_STRAIGHT_CRUISE = 3
L3_MOVING_TURNING_LEFT = 4
L3_MOVING_TURNING_RIGHT = 5

_TOKEN_BY_COMPOSITE_STATE = {
    L3_STOPPED: ["(STOPPED)"],
    L3_MOVING_STRAIGHT_ACCEL: ["(MOVING)", "(STRAIGHT)", "(ACCELERATING)"],
    L3_MOVING_STRAIGHT_DECEL: ["(MOVING)", "(STRAIGHT)", "(DECELERATING)"],
    L3_MOVING_STRAIGHT_CRUISE: ["(MOVING)", "(STRAIGHT)", "(CRUISING)"],
    L3_MOVING_TURNING_LEFT: ["(MOVING)", "(TURNING)", "(LEFT)"],
    L3_MOVING_TURNING_RIGHT: ["(MOVING)", "(TURNING)", "(RIGHT)"],
}

_NAME_BY_COMPOSITE_STATE = {
    L3_STOPPED: "STOPPED",
    L3_MOVING_STRAIGHT_ACCEL: "MOVING_STRAIGHT_ACCELERATING",
    L3_MOVING_STRAIGHT_DECEL: "MOVING_STRAIGHT_DECELERATING",
    L3_MOVING_STRAIGHT_CRUISE: "MOVING_STRAIGHT_CRUISING",
    L3_MOVING_TURNING_LEFT: "MOVING_TURNING_LEFT",
    L3_MOVING_TURNING_RIGHT: "MOVING_TURNING_RIGHT",
}


@dataclass(frozen=True)
class Level3Config:
    """Parameters for Level-3 longitudinal and lateral maneuver labeling."""

    fs_hz: float = 10.0
    pedalangle_col_index: int = 0
    pedalpressure_col_index: int = 1
    steer_angle_col_index: int = 2
    steer_speed_col_index: int = 3
    speed_col_index: int = 4
    yaw_col_index: int = 5

    speed_median_window: int = 7
    accel_median_window: int = 5
    yaw_median_window: int = 5

    a_on: float = 1.2
    a_off: float = 0.6
    pedal_on: float = 12.0
    brake_on: float = 150.0
    pedal_cruise_max: float = 8.0
    brake_cruise_max: float = 100.0
    lon_persistence_steps: int = 6

    yaw_deadband: float = 0.8
    direction_persistence_steps: int = 5
    steer_angle_bootstrap_min: float = 5.0
    steer_speed_bootstrap_min: float = 20.0

    initial_lon_state: int = CRUISING
    default_turn_direction: int = RIGHT

    def validate(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError(f"`fs_hz` must be > 0, got {self.fs_hz}.")

        for name, value in (
            ("pedalangle_col_index", self.pedalangle_col_index),
            ("pedalpressure_col_index", self.pedalpressure_col_index),
            ("steer_angle_col_index", self.steer_angle_col_index),
            ("steer_speed_col_index", self.steer_speed_col_index),
            ("speed_col_index", self.speed_col_index),
            ("yaw_col_index", self.yaw_col_index),
        ):
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0, got {value}.")

        for name, value in (
            ("speed_median_window", self.speed_median_window),
            ("accel_median_window", self.accel_median_window),
            ("yaw_median_window", self.yaw_median_window),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(
                    f"`{name}` must be a positive odd integer, got {value}."
                )

        if self.a_off > self.a_on:
            raise ValueError(
                f"`a_off` ({self.a_off}) must be <= `a_on` ({self.a_on})."
            )
        if self.lon_persistence_steps <= 0:
            raise ValueError(
                f"`lon_persistence_steps` must be >= 1, got {self.lon_persistence_steps}."
            )
        if self.direction_persistence_steps <= 0:
            raise ValueError(
                "`direction_persistence_steps` must be >= 1, "
                f"got {self.direction_persistence_steps}."
            )
        if self.yaw_deadband < 0:
            raise ValueError(
                f"`yaw_deadband` must be >= 0, got {self.yaw_deadband}."
            )

        if self.initial_lon_state not in (ACCELERATING, DECELERATING, CRUISING):
            raise ValueError(
                "`initial_lon_state` must be ACCELERATING(1), DECELERATING(2), "
                f"or CRUISING(3), got {self.initial_lon_state}."
            )
        if self.default_turn_direction not in (LEFT, RIGHT):
            raise ValueError(
                "`default_turn_direction` must be LEFT(1) or RIGHT(2), "
                f"got {self.default_turn_direction}."
            )


@dataclass(frozen=True)
class Level3SegmentSummary:
    """Run-length encoded segment summary for Level-3 labels."""

    segment_id: int
    start_idx: int
    end_idx: int
    length_steps: int
    fs_hz: float
    duration_sec: float
    duration_mmss: str
    label_vector: List[str]

    @property
    def label_vector_compact(self) -> str:
        return "[" + "".join(self.label_vector) + "]"

    @property
    def timesteps(self) -> int:
        return self.length_steps

    @property
    def duration(self) -> str:
        minutes, seconds = _split_minutes_seconds(self.duration_sec)
        return f"{minutes:02d} mins {seconds:05.2f} seconds"

    @property
    def starting_time(self) -> str:
        return _format_mmss_ss(self.start_idx / self.fs_hz)

    @property
    def ending_time(self) -> str:
        return _format_mmss_ss(self.end_idx / self.fs_hz)


@dataclass(frozen=True)
class Level3Result:
    """Level-3 outputs for one session."""

    session_id: str
    moving_state: np.ndarray
    turn_state: np.ndarray
    lon_state: np.ndarray
    lat_state: np.ndarray
    composite_state: np.ndarray
    speed_smooth: np.ndarray
    accel: np.ndarray
    yaw_smooth: np.ndarray
    segments: List[Level3SegmentSummary]
    config: Level3Config


def run_level3(
    session_id: str,
    session_matrix: np.ndarray,
    moving_state: np.ndarray,
    turn_state: np.ndarray,
    config: Level3Config,
) -> Level3Result:
    """Run Level-3 hierarchical labeling and segment summarization for one session."""
    config.validate()

    if session_matrix.ndim != 2:
        raise ValueError(f"Expected `session_matrix` to be 2D, got {session_matrix.ndim}D.")

    for name, col_idx in (
        ("pedalangle_col_index", config.pedalangle_col_index),
        ("pedalpressure_col_index", config.pedalpressure_col_index),
        ("steer_angle_col_index", config.steer_angle_col_index),
        ("steer_speed_col_index", config.steer_speed_col_index),
        ("speed_col_index", config.speed_col_index),
        ("yaw_col_index", config.yaw_col_index),
    ):
        if col_idx >= session_matrix.shape[1]:
            raise ValueError(
                f"`{name}`={col_idx} is out of bounds for matrix width {session_matrix.shape[1]}."
            )

    moving_state_arr = np.asarray(moving_state)
    if moving_state_arr.ndim != 1:
        raise ValueError(
            f"`moving_state` must be 1D, got shape={moving_state_arr.shape}."
        )

    turn_state_arr = np.asarray(turn_state)
    if turn_state_arr.ndim != 1:
        raise ValueError(
            f"`turn_state` must be 1D, got shape={turn_state_arr.shape}."
        )

    total_steps = len(session_matrix)
    if len(moving_state_arr) != total_steps:
        raise ValueError(
            "`moving_state` length must match `session_matrix` rows, "
            f"got len(moving_state)={len(moving_state_arr)}, T={total_steps}."
        )
    if len(turn_state_arr) != total_steps:
        raise ValueError(
            "`turn_state` length must match `session_matrix` rows, "
            f"got len(turn_state)={len(turn_state_arr)}, T={total_steps}."
        )

    if not np.isin(moving_state_arr, [0, 1]).all():
        raise ValueError("`moving_state` must contain only binary values {0, 1}.")
    if not np.isin(turn_state_arr, [L2_STOPPED, L2_STRAIGHT, L2_TURNING]).all():
        raise ValueError(
            "`turn_state` must contain only Level-2 values {STOPPED, STRAIGHT, TURNING}."
        )

    moving_state_bin = moving_state_arr.astype(np.int8, copy=True)
    turn_state_l2 = turn_state_arr.astype(np.int8, copy=True)
    _validate_input_gate_consistency(moving_state=moving_state_bin, turn_state=turn_state_l2)

    pedalangle = np.asarray(session_matrix[:, config.pedalangle_col_index], dtype=np.float64)
    pedalpressure = np.asarray(session_matrix[:, config.pedalpressure_col_index], dtype=np.float64)
    steer_angle = np.asarray(session_matrix[:, config.steer_angle_col_index], dtype=np.float64)
    steer_speed = np.asarray(session_matrix[:, config.steer_speed_col_index], dtype=np.float64)
    speed_raw = np.asarray(session_matrix[:, config.speed_col_index], dtype=np.float64)
    yaw_raw = np.asarray(session_matrix[:, config.yaw_col_index], dtype=np.float64)

    speed_smooth = centered_median(speed_raw, window=config.speed_median_window)
    accel_raw = central_difference(speed_smooth, fs_hz=config.fs_hz)
    accel_smooth = centered_median(accel_raw, window=config.accel_median_window)
    yaw_smooth = centered_median(yaw_raw, window=config.yaw_median_window)

    lon_state, lat_state, composite_state = compute_level3_states(
        moving_state=moving_state_bin,
        turn_state=turn_state_l2,
        pedalangle=pedalangle,
        pedalpressure=pedalpressure,
        steer_angle=steer_angle,
        steer_speed=steer_speed,
        yaw_smooth=yaw_smooth,
        accel=accel_smooth,
        config=config,
    )

    _validate_output_consistency(
        moving_state=moving_state_bin,
        turn_state=turn_state_l2,
        lon_state=lon_state,
        lat_state=lat_state,
        composite_state=composite_state,
    )

    segments = rle_level3_segments(composite_state=composite_state, fs_hz=config.fs_hz)
    _validate_segments_cover_all(segments, total_steps=len(composite_state))

    return Level3Result(
        session_id=session_id,
        moving_state=moving_state_bin,
        turn_state=turn_state_l2,
        lon_state=lon_state,
        lat_state=lat_state,
        composite_state=composite_state,
        speed_smooth=speed_smooth,
        accel=accel_smooth,
        yaw_smooth=yaw_smooth,
        segments=segments,
        config=config,
    )


def centered_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median (zero-phase for offline use)."""
    if values.ndim != 1:
        raise ValueError(f"`values` must be 1D, got shape={values.shape}.")
    if window <= 0 or window % 2 == 0:
        raise ValueError(f"`window` must be a positive odd integer, got {window}.")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    if window == 1:
        return values.astype(np.float64, copy=True)

    half_window = window // 2
    med = np.empty(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start_idx = max(0, idx - half_window)
        end_idx = min(len(values), idx + half_window + 1)
        med[idx] = float(np.median(values[start_idx:end_idx]))
    return med


def central_difference(values: np.ndarray, fs_hz: float) -> np.ndarray:
    """Central-difference derivative with one-sided edges."""
    if values.ndim != 1:
        raise ValueError(f"`values` must be 1D, got shape={values.shape}.")
    if fs_hz <= 0:
        raise ValueError(f"`fs_hz` must be > 0, got {fs_hz}.")

    n_steps = len(values)
    if n_steps == 0:
        return np.empty(0, dtype=np.float64)

    dt = 1.0 / fs_hz
    out = np.empty(n_steps, dtype=np.float64)

    if n_steps == 1:
        out[0] = 0.0
        return out
    if n_steps == 2:
        slope = float((values[1] - values[0]) / dt)
        out[0] = slope
        out[1] = slope
        return out

    out[0] = float((values[1] - values[0]) / dt)
    out[-1] = float((values[-1] - values[-2]) / dt)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return out


def compute_level3_states(
    moving_state: np.ndarray,
    turn_state: np.ndarray,
    pedalangle: np.ndarray,
    pedalpressure: np.ndarray,
    steer_angle: np.ndarray,
    steer_speed: np.ndarray,
    yaw_smooth: np.ndarray,
    accel: np.ndarray,
    config: Level3Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute longitudinal/lateral sub-states and final Level-3 composite state."""
    n_steps = len(moving_state)
    lon_state = np.empty(n_steps, dtype=np.int8)
    lat_state = np.empty(n_steps, dtype=np.int8)
    composite_state = np.empty(n_steps, dtype=np.int8)

    current_lon_state = config.initial_lon_state
    pending_lon_state: int | None = None
    pending_lon_count = 0

    current_turn_direction: int | None = None
    pending_dir_state: int | None = None
    pending_dir_count = 0

    for idx in range(n_steps):
        if moving_state[idx] == 0:
            lon_state[idx] = LON_NA
            lat_state[idx] = LAT_NA
            composite_state[idx] = L3_STOPPED

            current_lon_state = config.initial_lon_state
            pending_lon_state = None
            pending_lon_count = 0

            current_turn_direction = None
            pending_dir_state = None
            pending_dir_count = 0
            continue

        if turn_state[idx] == L2_STRAIGHT:
            # Level-3 longitudinal mode applies only in MOVING+STRAIGHT.
            lat_state[idx] = LAT_NA
            current_turn_direction = None
            pending_dir_state = None
            pending_dir_count = 0

            lon_candidate = _longitudinal_candidate(
                accel_value=float(accel[idx]),
                pedalangle_value=float(pedalangle[idx]),
                pedalpressure_value=float(pedalpressure[idx]),
                config=config,
            )

            if lon_candidate is None or lon_candidate == current_lon_state:
                pending_lon_state = None
                pending_lon_count = 0
            else:
                if pending_lon_state == lon_candidate:
                    pending_lon_count += 1
                else:
                    pending_lon_state = lon_candidate
                    pending_lon_count = 1

                if pending_lon_count >= config.lon_persistence_steps:
                    current_lon_state = lon_candidate
                    pending_lon_state = None
                    pending_lon_count = 0

            lon_state[idx] = current_lon_state
            composite_state[idx] = _composite_from_straight_lon_state(current_lon_state)
            continue

        if turn_state[idx] == L2_TURNING:
            # Level-3 turn direction applies only in MOVING+TURNING.
            lon_state[idx] = LON_NA
            current_lon_state = config.initial_lon_state
            pending_lon_state = None
            pending_lon_count = 0

            if current_turn_direction is None:
                current_turn_direction = _bootstrap_turn_direction(
                    steer_angle_value=float(steer_angle[idx]),
                    steer_speed_value=float(steer_speed[idx]),
                    yaw_value=float(yaw_smooth[idx]),
                    config=config,
                )

            dir_candidate = _direction_candidate(
                yaw_value=float(yaw_smooth[idx]),
                yaw_deadband=config.yaw_deadband,
            )

            if dir_candidate is None or dir_candidate == current_turn_direction:
                pending_dir_state = None
                pending_dir_count = 0
            else:
                if pending_dir_state == dir_candidate:
                    pending_dir_count += 1
                else:
                    pending_dir_state = dir_candidate
                    pending_dir_count = 1

                if pending_dir_count >= config.direction_persistence_steps:
                    current_turn_direction = dir_candidate
                    pending_dir_state = None
                    pending_dir_count = 0

            lat_state[idx] = current_turn_direction
            composite_state[idx] = _composite_from_turn_direction(current_turn_direction)
            continue

        raise ValueError(
            f"Invalid Level-2 state at index {idx}: {turn_state[idx]}."
        )

    return lon_state, lat_state, composite_state


def rle_level3_segments(
    composite_state: np.ndarray,
    fs_hz: float,
) -> List[Level3SegmentSummary]:
    """Run-length encode per-step Level-3 composite labels into segment summaries."""
    if composite_state.ndim != 1:
        raise ValueError(
            f"`composite_state` must be 1D, got shape={composite_state.shape}."
        )
    if len(composite_state) == 0:
        return []
    if fs_hz <= 0:
        raise ValueError(f"`fs_hz` must be > 0, got {fs_hz}.")

    segments: List[Level3SegmentSummary] = []
    start_idx = 0
    current_state = int(composite_state[0])
    segment_id = 0

    for idx in range(1, len(composite_state)):
        state_value = int(composite_state[idx])
        if state_value != current_state:
            segments.append(
                _make_segment(
                    segment_id=segment_id,
                    start_idx=start_idx,
                    end_idx=idx,
                    composite_state=current_state,
                    fs_hz=fs_hz,
                )
            )
            segment_id += 1
            start_idx = idx
            current_state = state_value

    segments.append(
        _make_segment(
            segment_id=segment_id,
            start_idx=start_idx,
            end_idx=len(composite_state),
            composite_state=current_state,
            fs_hz=fs_hz,
        )
    )
    return segments


def _longitudinal_candidate(
    accel_value: float,
    pedalangle_value: float,
    pedalpressure_value: float,
    config: Level3Config,
) -> int | None:
    if accel_value <= -config.a_on or pedalpressure_value >= config.brake_on:
        return DECELERATING

    if accel_value >= config.a_on or (
        pedalangle_value >= config.pedal_on and pedalpressure_value < config.brake_on
    ):
        return ACCELERATING

    if (
        abs(accel_value) <= config.a_off
        and pedalangle_value <= config.pedal_cruise_max
        and pedalpressure_value <= config.brake_cruise_max
    ):
        return CRUISING

    return None


def _direction_candidate(yaw_value: float, yaw_deadband: float) -> int | None:
    if yaw_value >= yaw_deadband:
        return RIGHT
    if yaw_value <= -yaw_deadband:
        return LEFT
    return None


def _bootstrap_turn_direction(
    steer_angle_value: float,
    steer_speed_value: float,
    yaw_value: float,
    config: Level3Config,
) -> int:
    if abs(steer_angle_value) >= config.steer_angle_bootstrap_min and steer_angle_value != 0:
        return _direction_from_signed_value(steer_angle_value)

    if abs(steer_speed_value) >= config.steer_speed_bootstrap_min and steer_speed_value != 0:
        return _direction_from_signed_value(steer_speed_value)

    if yaw_value != 0:
        return _direction_from_signed_value(yaw_value)

    return config.default_turn_direction


def _direction_from_signed_value(value: float) -> int:
    return RIGHT if value > 0 else LEFT


def _composite_from_straight_lon_state(lon_state: int) -> int:
    if lon_state == ACCELERATING:
        return L3_MOVING_STRAIGHT_ACCEL
    if lon_state == DECELERATING:
        return L3_MOVING_STRAIGHT_DECEL
    if lon_state == CRUISING:
        return L3_MOVING_STRAIGHT_CRUISE
    raise ValueError(f"Invalid longitudinal state for MOVING+STRAIGHT: {lon_state}.")


def _composite_from_turn_direction(direction: int) -> int:
    if direction == LEFT:
        return L3_MOVING_TURNING_LEFT
    if direction == RIGHT:
        return L3_MOVING_TURNING_RIGHT
    raise ValueError(f"Invalid turn direction state for MOVING+TURNING: {direction}.")


def _make_segment(
    segment_id: int,
    start_idx: int,
    end_idx: int,
    composite_state: int,
    fs_hz: float,
) -> Level3SegmentSummary:
    tokens = _TOKEN_BY_COMPOSITE_STATE.get(composite_state)
    if tokens is None:
        raise ValueError(f"Unknown Level-3 composite state value: {composite_state}.")

    length_steps = end_idx - start_idx
    duration_sec = float(length_steps / fs_hz)
    return Level3SegmentSummary(
        segment_id=segment_id,
        start_idx=start_idx,
        end_idx=end_idx,
        length_steps=length_steps,
        fs_hz=fs_hz,
        duration_sec=duration_sec,
        duration_mmss=_format_duration_mmss(duration_sec),
        label_vector=list(tokens),
    )


def _validate_input_gate_consistency(
    moving_state: np.ndarray,
    turn_state: np.ndarray,
) -> None:
    if moving_state.shape != turn_state.shape:
        raise ValueError(
            "`moving_state` and `turn_state` must have same shape, "
            f"got {moving_state.shape} and {turn_state.shape}."
        )

    if not np.array_equal((turn_state != L2_STOPPED).astype(np.int8), moving_state.astype(np.int8)):
        raise ValueError(
            "Input gate consistency violated: `(turn_state != STOPPED)` must match `moving_state`."
        )


def _validate_output_consistency(
    moving_state: np.ndarray,
    turn_state: np.ndarray,
    lon_state: np.ndarray,
    lat_state: np.ndarray,
    composite_state: np.ndarray,
) -> None:
    if not (
        moving_state.shape
        == turn_state.shape
        == lon_state.shape
        == lat_state.shape
        == composite_state.shape
    ):
        raise ValueError("All Level-3 output arrays must have identical shapes.")

    if not np.array_equal(
        (composite_state != L3_STOPPED).astype(np.int8),
        moving_state.astype(np.int8),
    ):
        raise ValueError(
            "Output gate consistency violated: `(composite_state != STOPPED)` must match `moving_state`."
        )

    stopped_mask = moving_state == 0
    if stopped_mask.any():
        if not np.all(composite_state[stopped_mask] == L3_STOPPED):
            raise ValueError("STOPPED timesteps must map to Level-3 STOPPED composite state.")
        if not np.all(lon_state[stopped_mask] == LON_NA):
            raise ValueError("STOPPED timesteps must not have longitudinal sub-states.")
        if not np.all(lat_state[stopped_mask] == LAT_NA):
            raise ValueError("STOPPED timesteps must not have lateral direction sub-states.")

    straight_mask = turn_state == L2_STRAIGHT
    if straight_mask.any():
        if not np.isin(lon_state[straight_mask], [ACCELERATING, DECELERATING, CRUISING]).all():
            raise ValueError("MOVING+STRAIGHT timesteps must have valid longitudinal states.")
        if not np.all(lat_state[straight_mask] == LAT_NA):
            raise ValueError("MOVING+STRAIGHT timesteps must not have LEFT/RIGHT direction values.")
        if not np.isin(
            composite_state[straight_mask],
            [
                L3_MOVING_STRAIGHT_ACCEL,
                L3_MOVING_STRAIGHT_DECEL,
                L3_MOVING_STRAIGHT_CRUISE,
            ],
        ).all():
            raise ValueError("MOVING+STRAIGHT timesteps must map to straight Level-3 composite states.")

    turning_mask = turn_state == L2_TURNING
    if turning_mask.any():
        if not np.all(lon_state[turning_mask] == LON_NA):
            raise ValueError("MOVING+TURNING timesteps must not have longitudinal values.")
        if not np.isin(lat_state[turning_mask], [LEFT, RIGHT]).all():
            raise ValueError("MOVING+TURNING timesteps must have LEFT/RIGHT direction values.")
        if not np.isin(
            composite_state[turning_mask],
            [L3_MOVING_TURNING_LEFT, L3_MOVING_TURNING_RIGHT],
        ).all():
            raise ValueError("MOVING+TURNING timesteps must map to turning Level-3 composite states.")


def _format_duration_mmss(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d}:{seconds:04.1f}"


def _split_minutes_seconds(duration_sec: float) -> tuple[int, float]:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return minutes, seconds


def _format_mmss_ss(duration_sec: float) -> str:
    minutes, seconds = _split_minutes_seconds(duration_sec)
    return f"{minutes:02d}:{seconds:05.2f}"


def _validate_segments_cover_all(
    segments: Sequence[Level3SegmentSummary],
    total_steps: int,
) -> None:
    if total_steps == 0 and len(segments) == 0:
        return
    if total_steps <= 0:
        raise ValueError(f"`total_steps` must be > 0 when segments exist, got {total_steps}.")
    if len(segments) == 0:
        raise ValueError("No segments produced for a non-empty session.")

    if segments[0].start_idx != 0:
        raise ValueError(f"Coverage gap: first segment starts at {segments[0].start_idx}, expected 0.")
    if segments[-1].end_idx != total_steps:
        raise ValueError(
            f"Coverage gap: last segment ends at {segments[-1].end_idx}, expected {total_steps}."
        )

    previous_end = 0
    for seg in segments:
        if seg.start_idx != previous_end:
            raise ValueError(
                "Segments are not contiguous/disjoint: "
                f"segment_id={seg.segment_id}, start_idx={seg.start_idx}, expected={previous_end}."
            )
        if seg.end_idx <= seg.start_idx:
            raise ValueError(
                f"Invalid segment length for segment_id={seg.segment_id}: "
                f"start_idx={seg.start_idx}, end_idx={seg.end_idx}."
            )
        if not any(seg.label_vector == tokens for tokens in _TOKEN_BY_COMPOSITE_STATE.values()):
            raise ValueError(
                "Invalid label token combination in Level-3 segment "
                f"segment_id={seg.segment_id}: {seg.label_vector}."
            )
        previous_end = seg.end_idx


def state_from_tokens(label_vector: List[str]) -> int:
    """Map Level-3 label tokens back to composite state ids for validation/reporting."""
    for state, tokens in _TOKEN_BY_COMPOSITE_STATE.items():
        if label_vector == tokens:
            return state
    raise ValueError(f"Unknown Level-3 label vector: {label_vector}.")


def state_name(state: int) -> str:
    """Human-readable Level-3 composite state name for reporting."""
    if state not in _NAME_BY_COMPOSITE_STATE:
        raise ValueError(f"Unknown Level-3 composite state value: {state}.")
    return _NAME_BY_COMPOSITE_STATE[state]

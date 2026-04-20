"""Pure Level-2 logic: MOVING-gated STRAIGHT vs TURNING labeling and segment RLE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

L2_STOPPED = 0
STRAIGHT = 1
TURNING = 2

_TOKEN_BY_STATE = {
    L2_STOPPED: ["(STOPPED)"],
    STRAIGHT: ["(MOVING)", "(STRAIGHT)"],
    TURNING: ["(MOVING)", "(TURNING)"],
}
_NAME_BY_STATE = {
    L2_STOPPED: "STOPPED",
    STRAIGHT: "STRAIGHT",
    TURNING: "TURNING",
}


@dataclass(frozen=True)
class Level2Config:
    """Parameters for Level-2 straight/turn segmentation on MOVING timesteps."""

    fs_hz: float = 10.0
    yaw_col_index: int = 5
    steer_angle_col_index: int = 2
    steer_speed_col_index: int = 3
    speed_col_index: int = 4
    yaw_median_window: int = 5
    yaw_on: float = 2.2
    yaw_off: float = 1.0
    steer_on_deg: float = 35.0
    steer_speed_on_dps: float = 80.0
    assist_speed_min: float = 2.0
    straight_steer_max_deg: float = 10.0
    persistence_on_steps: int = 8
    persistence_off_steps: int = 10
    initial_turn_state: int = STRAIGHT

    def validate(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError(f"`fs_hz` must be > 0, got {self.fs_hz}.")

        for name, value in (
            ("yaw_col_index", self.yaw_col_index),
            ("steer_angle_col_index", self.steer_angle_col_index),
            ("steer_speed_col_index", self.steer_speed_col_index),
            ("speed_col_index", self.speed_col_index),
        ):
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0, got {value}.")

        if self.yaw_median_window <= 0 or self.yaw_median_window % 2 == 0:
            raise ValueError(
                "`yaw_median_window` must be a positive odd integer, "
                f"got {self.yaw_median_window}."
            )
        if self.yaw_off > self.yaw_on:
            raise ValueError(
                f"`yaw_off` ({self.yaw_off}) must be <= `yaw_on` ({self.yaw_on})."
            )
        if self.persistence_on_steps <= 0:
            raise ValueError(
                "`persistence_on_steps` must be >= 1, "
                f"got {self.persistence_on_steps}."
            )
        if self.persistence_off_steps <= 0:
            raise ValueError(
                "`persistence_off_steps` must be >= 1, "
                f"got {self.persistence_off_steps}."
            )
        if self.initial_turn_state not in (STRAIGHT, TURNING):
            raise ValueError(
                "`initial_turn_state` must be STRAIGHT(1) or TURNING(2), "
                f"got {self.initial_turn_state}."
            )


@dataclass(frozen=True)
class Level2SegmentSummary:
    """Run-length encoded segment summary for Level-2 labels."""

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
class Level2Result:
    """Level-2 outputs for one session."""

    session_id: str
    moving_state: np.ndarray
    yaw_abs_raw: np.ndarray
    yaw_abs_smooth: np.ndarray
    turn_state: np.ndarray
    segments: List[Level2SegmentSummary]
    config: Level2Config


def run_level2(
    session_id: str,
    session_matrix: np.ndarray,
    moving_state: np.ndarray,
    config: Level2Config,
) -> Level2Result:
    """Run Level-2 straight/turn labeling and segment summarization for one session."""
    config.validate()

    if session_matrix.ndim != 2:
        raise ValueError(f"Expected `session_matrix` to be 2D, got {session_matrix.ndim}D.")

    for name, col_idx in (
        ("yaw_col_index", config.yaw_col_index),
        ("steer_angle_col_index", config.steer_angle_col_index),
        ("steer_speed_col_index", config.steer_speed_col_index),
        ("speed_col_index", config.speed_col_index),
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
    if len(moving_state_arr) != len(session_matrix):
        raise ValueError(
            "`moving_state` length must match `session_matrix` rows, "
            f"got len(moving_state)={len(moving_state_arr)}, T={len(session_matrix)}."
        )
    if not np.isin(moving_state_arr, [0, 1]).all():
        raise ValueError("`moving_state` must contain only binary values {0, 1}.")

    moving_state_binary = moving_state_arr.astype(np.int8, copy=True)

    speed = np.asarray(session_matrix[:, config.speed_col_index], dtype=np.float64)
    yaw_abs_raw = np.abs(np.asarray(session_matrix[:, config.yaw_col_index], dtype=np.float64))
    yaw_abs_smooth = centered_median(yaw_abs_raw, window=config.yaw_median_window)
    steer_abs = np.abs(
        np.asarray(session_matrix[:, config.steer_angle_col_index], dtype=np.float64)
    )
    steer_speed_abs = np.abs(
        np.asarray(session_matrix[:, config.steer_speed_col_index], dtype=np.float64)
    )

    turn_state = moving_gated_turn_state(
        moving_state=moving_state_binary,
        speed=speed,
        yaw_abs_smooth=yaw_abs_smooth,
        steer_abs=steer_abs,
        steer_speed_abs=steer_speed_abs,
        config=config,
    )

    _validate_turn_state_gate_consistency(
        turn_state=turn_state,
        moving_state=moving_state_binary,
    )

    segments = rle_level2_segments(turn_state=turn_state, fs_hz=config.fs_hz)
    _validate_segments_cover_all(segments, total_steps=len(turn_state))

    return Level2Result(
        session_id=session_id,
        moving_state=moving_state_binary,
        yaw_abs_raw=yaw_abs_raw,
        yaw_abs_smooth=yaw_abs_smooth,
        turn_state=turn_state,
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


def moving_gated_turn_state(
    moving_state: np.ndarray,
    speed: np.ndarray,
    yaw_abs_smooth: np.ndarray,
    steer_abs: np.ndarray,
    steer_speed_abs: np.ndarray,
    config: Level2Config,
) -> np.ndarray:
    """Apply MOVING-gated hysteresis + persistence for STRAIGHT vs TURNING."""
    n_steps = len(moving_state)
    turn_state = np.empty(n_steps, dtype=np.int8)

    current_turn_state = config.initial_turn_state
    pending_state = None
    pending_count = 0

    for idx in range(n_steps):
        if moving_state[idx] == 0:
            turn_state[idx] = L2_STOPPED
            current_turn_state = config.initial_turn_state
            pending_state = None
            pending_count = 0
            continue

        yaw_value = float(yaw_abs_smooth[idx])
        steer_value = float(steer_abs[idx])
        steer_speed_value = float(steer_speed_abs[idx])
        speed_value = float(speed[idx])

        candidate = None
        if yaw_value >= config.yaw_on:
            candidate = TURNING
        elif yaw_value <= config.yaw_off and steer_value <= config.straight_steer_max_deg:
            candidate = STRAIGHT
        elif (
            steer_value >= config.steer_on_deg
            and steer_speed_value >= config.steer_speed_on_dps
            and speed_value >= config.assist_speed_min
        ):
            candidate = TURNING

        if candidate is None or candidate == current_turn_state:
            pending_state = None
            pending_count = 0
        else:
            if pending_state == candidate:
                pending_count += 1
            else:
                pending_state = candidate
                pending_count = 1

            required_steps = (
                config.persistence_on_steps
                if candidate == TURNING
                else config.persistence_off_steps
            )
            if pending_count >= required_steps:
                current_turn_state = candidate
                pending_state = None
                pending_count = 0

        turn_state[idx] = current_turn_state

    return turn_state


def rle_level2_segments(turn_state: np.ndarray, fs_hz: float) -> List[Level2SegmentSummary]:
    """Run-length encode per-step Level-2 labels into segment summaries."""
    if turn_state.ndim != 1:
        raise ValueError(f"`turn_state` must be 1D, got shape={turn_state.shape}.")
    if len(turn_state) == 0:
        return []
    if fs_hz <= 0:
        raise ValueError(f"`fs_hz` must be > 0, got {fs_hz}.")

    segments: List[Level2SegmentSummary] = []
    start_idx = 0
    current_state = int(turn_state[0])
    segment_id = 0

    for idx in range(1, len(turn_state)):
        state_value = int(turn_state[idx])
        if state_value != current_state:
            segments.append(
                _make_segment(
                    segment_id=segment_id,
                    start_idx=start_idx,
                    end_idx=idx,
                    state=current_state,
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
            end_idx=len(turn_state),
            state=current_state,
            fs_hz=fs_hz,
        )
    )
    return segments


def _make_segment(
    segment_id: int,
    start_idx: int,
    end_idx: int,
    state: int,
    fs_hz: float,
) -> Level2SegmentSummary:
    if state not in _TOKEN_BY_STATE:
        raise ValueError(f"Unknown Level-2 state value: {state}.")
    length_steps = end_idx - start_idx
    duration_sec = float(length_steps / fs_hz)
    return Level2SegmentSummary(
        segment_id=segment_id,
        start_idx=start_idx,
        end_idx=end_idx,
        length_steps=length_steps,
        fs_hz=fs_hz,
        duration_sec=duration_sec,
        duration_mmss=_format_duration_mmss(duration_sec),
        label_vector=list(_TOKEN_BY_STATE[state]),
    )


def _validate_turn_state_gate_consistency(
    turn_state: np.ndarray,
    moving_state: np.ndarray,
) -> None:
    if turn_state.shape != moving_state.shape:
        raise ValueError(
            "`turn_state` and `moving_state` must have the same shape, "
            f"got {turn_state.shape} and {moving_state.shape}."
        )

    is_moving_from_turn = (turn_state != L2_STOPPED).astype(np.int8)
    if not np.array_equal(is_moving_from_turn, moving_state.astype(np.int8)):
        raise ValueError(
            "Gate consistency violated: `(turn_state != L2_STOPPED)` must match `moving_state`."
        )

    moving_mask = moving_state == 1
    if moving_mask.any():
        moving_values = np.unique(turn_state[moving_mask])
        if not np.isin(moving_values, [STRAIGHT, TURNING]).all():
            raise ValueError(
                "Level-2 moving timesteps must be only STRAIGHT/TURNING, "
                f"got values {moving_values.tolist()}."
            )


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
    segments: Sequence[Level2SegmentSummary],
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
        if not any(seg.label_vector == tokens for tokens in _TOKEN_BY_STATE.values()):
            raise ValueError(
                "Invalid label token combination in Level-2 segment "
                f"segment_id={seg.segment_id}: {seg.label_vector}."
            )
        previous_end = seg.end_idx


def state_from_tokens(label_vector: List[str]) -> int:
    """Map Level-2 label tokens back to state ids for validation/reporting."""
    for state, tokens in _TOKEN_BY_STATE.items():
        if label_vector == tokens:
            return state
    raise ValueError(f"Unknown Level-2 label vector: {label_vector}.")


def state_name(state: int) -> str:
    """Human-readable Level-2 state name for reporting."""
    if state not in _NAME_BY_STATE:
        raise ValueError(f"Unknown Level-2 state value: {state}.")
    return _NAME_BY_STATE[state]

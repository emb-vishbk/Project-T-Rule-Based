"""Pure Level-1 logic: smoothing, hysteresis labeling, and segment RLE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from traceability.data.can_session import SPEED_COLUMN_INDEX

STOPPED = 0
MOVING = 1

_TOKEN_BY_STATE = {
    STOPPED: "(STOPPED)",
    MOVING: "(MOVING)",
}
_NAME_BY_STATE = {
    STOPPED: "STOPPED",
    MOVING: "MOVING",
}


@dataclass(frozen=True)
class Level1Config:
    """Parameters for Level-1 stop/move segmentation."""

    fs_hz: float = 10.0
    speed_col_index: int = SPEED_COLUMN_INDEX
    median_window: int = 5
    v_stopped: float = 0.5
    v_moving: float = 1.0
    persistence_steps: int = 3
    initial_state: int = STOPPED

    def validate(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError(f"`fs_hz` must be > 0, got {self.fs_hz}.")
        if self.speed_col_index < 0:
            raise ValueError(f"`speed_col_index` must be >= 0, got {self.speed_col_index}.")
        if self.median_window <= 0 or self.median_window % 2 == 0:
            raise ValueError(
                f"`median_window` must be a positive odd integer, got {self.median_window}."
            )
        if self.v_stopped > self.v_moving:
            raise ValueError(
                f"`v_stopped` ({self.v_stopped}) must be <= `v_moving` ({self.v_moving})."
            )
        if self.persistence_steps <= 0:
            raise ValueError(
                f"`persistence_steps` must be >= 1, got {self.persistence_steps}."
            )
        if self.initial_state not in (STOPPED, MOVING):
            raise ValueError(
                f"`initial_state` must be STOPPED({STOPPED}) or MOVING({MOVING}), got {self.initial_state}."
            )


@dataclass(frozen=True)
class SegmentSummary:
    """Run-length encoded segment summary for Level-1 labels."""

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
class Level1Result:
    """Level-1 outputs for one session."""

    session_id: str
    speed_raw: np.ndarray
    speed_smooth: np.ndarray
    moving_state: np.ndarray
    segments: List[SegmentSummary]
    config: Level1Config


def run_level1(session_id: str, session_matrix: np.ndarray, config: Level1Config) -> Level1Result:
    """Run Level-1 stop/moving labeling and segment summarization for one session."""
    config.validate()

    if session_matrix.ndim != 2:
        raise ValueError(f"Expected `session_matrix` to be 2D, got {session_matrix.ndim}D.")
    if config.speed_col_index >= session_matrix.shape[1]:
        raise ValueError(
            f"`speed_col_index`={config.speed_col_index} is out of bounds for "
            f"matrix width {session_matrix.shape[1]}."
        )

    speed_raw = np.asarray(session_matrix[:, config.speed_col_index], dtype=np.float64)
    speed_smooth = rolling_median_centered(speed_raw, window=config.median_window)
    moving_state = hysteresis_stop_move(
        speed=speed_smooth,
        v_stopped=config.v_stopped,
        v_moving=config.v_moving,
        persistence_steps=config.persistence_steps,
        initial_state=config.initial_state,
    )
    segments = rle_level1_segments(moving_state=moving_state, fs_hz=config.fs_hz)
    _validate_segments_cover_all(segments, total_steps=len(moving_state))

    return Level1Result(
        session_id=session_id,
        speed_raw=speed_raw,
        speed_smooth=speed_smooth,
        moving_state=moving_state,
        segments=segments,
        config=config,
    )


def rolling_median_centered(values: np.ndarray, window: int) -> np.ndarray:
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


def rolling_median_causal(values: np.ndarray, window: int) -> np.ndarray:
    """
    Backward-compatible alias.

    Level-1 now uses centered median smoothing to avoid causal lag.
    """
    return rolling_median_centered(values=values, window=window)


def hysteresis_stop_move(
    speed: np.ndarray,
    v_stopped: float,
    v_moving: float,
    persistence_steps: int,
    initial_state: int = STOPPED,
) -> np.ndarray:
    """Two-state stop/move hysteresis with optional persistence."""
    if speed.ndim != 1:
        raise ValueError(f"`speed` must be 1D, got shape={speed.shape}.")
    if len(speed) == 0:
        return np.empty(0, dtype=np.int8)
    if v_stopped > v_moving:
        raise ValueError(f"`v_stopped` ({v_stopped}) must be <= `v_moving` ({v_moving}).")
    if persistence_steps <= 0:
        raise ValueError(f"`persistence_steps` must be >= 1, got {persistence_steps}.")
    if initial_state not in (STOPPED, MOVING):
        raise ValueError(
            f"`initial_state` must be STOPPED({STOPPED}) or MOVING({MOVING}), got {initial_state}."
        )

    state = np.empty(len(speed), dtype=np.int8)
    current_state = _resolve_initial_state(
        first_speed=float(speed[0]),
        v_stopped=v_stopped,
        v_moving=v_moving,
        default_state=initial_state,
    )
    state[0] = current_state

    pending_state = None
    pending_count = 0

    for idx in range(1, len(speed)):
        s = float(speed[idx])
        candidate = _candidate_state(s, v_stopped=v_stopped, v_moving=v_moving)

        if candidate is None or candidate == current_state:
            pending_state = None
            pending_count = 0
        else:
            if candidate == pending_state:
                pending_count += 1
            else:
                pending_state = candidate
                pending_count = 1

            if pending_count >= persistence_steps:
                current_state = candidate
                pending_state = None
                pending_count = 0

        state[idx] = current_state

    return state


def rle_level1_segments(moving_state: np.ndarray, fs_hz: float) -> List[SegmentSummary]:
    """Run-length encode per-step Level-1 labels into segment summaries."""
    if moving_state.ndim != 1:
        raise ValueError(f"`moving_state` must be 1D, got shape={moving_state.shape}.")
    if len(moving_state) == 0:
        return []
    if fs_hz <= 0:
        raise ValueError(f"`fs_hz` must be > 0, got {fs_hz}.")

    segments: List[SegmentSummary] = []
    start_idx = 0
    current_state = int(moving_state[0])
    segment_id = 0

    for idx in range(1, len(moving_state)):
        this_state = int(moving_state[idx])
        if this_state != current_state:
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
            current_state = this_state

    segments.append(
        _make_segment(
            segment_id=segment_id,
            start_idx=start_idx,
            end_idx=len(moving_state),
            state=current_state,
            fs_hz=fs_hz,
        )
    )
    return segments


def _make_segment(segment_id: int, start_idx: int, end_idx: int, state: int, fs_hz: float) -> SegmentSummary:
    if state not in _TOKEN_BY_STATE:
        raise ValueError(f"Unknown Level-1 state value: {state}.")
    length_steps = end_idx - start_idx
    duration_sec = float(length_steps / fs_hz)
    return SegmentSummary(
        segment_id=segment_id,
        start_idx=start_idx,
        end_idx=end_idx,
        length_steps=length_steps,
        fs_hz=fs_hz,
        duration_sec=duration_sec,
        duration_mmss=_format_duration_mmss(duration_sec),
        label_vector=[_TOKEN_BY_STATE[state]],
    )


def _candidate_state(speed_value: float, v_stopped: float, v_moving: float) -> int | None:
    if speed_value <= v_stopped:
        return STOPPED
    if speed_value >= v_moving:
        return MOVING
    return None


def _resolve_initial_state(first_speed: float, v_stopped: float, v_moving: float, default_state: int) -> int:
    candidate = _candidate_state(first_speed, v_stopped=v_stopped, v_moving=v_moving)
    if candidate is None:
        return default_state
    return candidate


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


def _validate_segments_cover_all(segments: Sequence[SegmentSummary], total_steps: int) -> None:
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
        if len(seg.label_vector) != 1:
            raise ValueError(
                f"Level-1 segment_id={seg.segment_id} must have one label token, got {seg.label_vector}."
            )
        token = seg.label_vector[0]
        if token not in _TOKEN_BY_STATE.values():
            raise ValueError(
                f"Invalid label token in segment_id={seg.segment_id}: {token}. "
                f"Expected one of {set(_TOKEN_BY_STATE.values())}."
            )
        previous_end = seg.end_idx


def state_name(state: int) -> str:
    """Human-readable state name for reporting."""
    if state not in _NAME_BY_STATE:
        raise ValueError(f"Unknown Level-1 state value: {state}.")
    return _NAME_BY_STATE[state]

"""Level-4 dataclasses and constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


DIRECTION_LEFT = "LEFT"
DIRECTION_RIGHT = "RIGHT"
DIRECTION_VALUES = (DIRECTION_LEFT, DIRECTION_RIGHT)

TURN_SUBTYPE_SHARP = "SHARP"
TURN_SUBTYPE_SMOOTH = "SMOOTH"
TURN_SUBTYPE_VALUES = (TURN_SUBTYPE_SHARP, TURN_SUBTYPE_SMOOTH)


@dataclass(frozen=True)
class Level4RuleConfig:
    """
    Rule thresholds for SHARP vs SMOOTH classification on Level-3 TURNING segments.

    Defaults are derived from `can_data_final_v` turn-event distribution analysis
    (`artifacts_tmp_final_v/level4_rule_analysis_can_data_final_v.json`) and are
    intentionally conservative (smooth-by-default).
    """

    fs_hz: float = 10.0

    # Conservative SHARP gating to avoid overclassifying short/smooth highway turns.
    min_sharp_duration_sec: float = 2.0
    long_duration_guard_sec: float = 12.0
    long_duration_mean_abs_yaw_min: float = 12.0

    # Aggressiveness thresholds (event-local only; no context).
    sharp_mean_abs_yaw_min: float = 10.0
    sharp_peak_abs_yaw_min: float = 20.0
    sharp_peak_abs_steer_angle_min: float = 180.0
    sharp_peak_abs_steer_speed_min: float = 240.0

    # SHARP requires 3/4 votes (conservative), and at least one yaw + one steer vote.
    sharp_vote_threshold: int = 3
    require_yaw_vote: bool = True
    require_steer_vote: bool = True

    # Metadata / provenance
    rule_version: str = "rule_based_v1_sharp_smooth"
    threshold_source: str = "can_data_final_v_turn_event_distribution"

    def validate(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError("`fs_hz` must be > 0.")
        if self.min_sharp_duration_sec < 0:
            raise ValueError("`min_sharp_duration_sec` must be >= 0.")
        if self.long_duration_guard_sec <= 0:
            raise ValueError("`long_duration_guard_sec` must be > 0.")
        if self.long_duration_mean_abs_yaw_min <= 0:
            raise ValueError("`long_duration_mean_abs_yaw_min` must be > 0.")
        if self.sharp_vote_threshold <= 0:
            raise ValueError("`sharp_vote_threshold` must be >= 1.")
        for name in (
            "sharp_mean_abs_yaw_min",
            "sharp_peak_abs_yaw_min",
            "sharp_peak_abs_steer_angle_min",
            "sharp_peak_abs_steer_speed_min",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"`{name}` must be > 0.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Level4EventPrediction:
    session_id: str
    level3_segment_id: int
    direction: str
    turn_subtype: str
    start_idx: int
    end_idx: int
    timesteps: int
    duration_sec: float
    event_features: Dict[str, float]
    rule_diagnostics: Dict[str, float | int | bool]

    def validate(self) -> None:
        if self.direction not in DIRECTION_VALUES:
            raise ValueError(f"Invalid direction {self.direction!r}.")
        if self.turn_subtype not in TURN_SUBTYPE_VALUES:
            raise ValueError(f"Invalid subtype {self.turn_subtype!r}.")
        if self.end_idx <= self.start_idx:
            raise ValueError("Invalid [start_idx, end_idx) for Level-4 event.")
        if self.timesteps != self.end_idx - self.start_idx:
            raise ValueError("`timesteps` mismatch with [start_idx,end_idx).")


@dataclass(frozen=True)
class Level4SegmentSummary:
    segment_id: int
    start_idx: int
    end_idx: int
    timesteps: int
    duration: str
    starting_time: str
    ending_time: str
    label_vector: List[str]

    def to_dict(self) -> dict:
        return {
            "segment_id": int(self.segment_id),
            "timesteps": int(self.timesteps),
            "duration": str(self.duration),
            "starting_time": str(self.starting_time),
            "ending_time": str(self.ending_time),
            "label_vector": list(self.label_vector),
        }


@dataclass(frozen=True)
class Level4Result:
    session_id: str
    fs_hz: float
    source_level3_segments: List[dict]
    segments: List[Level4SegmentSummary]
    event_predictions: List[Level4EventPrediction]
    config: Level4RuleConfig

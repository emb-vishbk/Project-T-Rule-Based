"""Dataclasses for final 25 Hz result generation with segment metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FinalMetricsConfig:
    """Signal indices and preprocessing settings for final metric computation."""

    fs_hz: float = 25.0
    speed_col_index: int = 4
    yaw_col_index: int = 5
    steer_angle_col_index: int = 2

    # Use the same smoothing setup as current 25 Hz Level-3 profile.
    speed_median_window: int = 17
    accel_median_window: int = 11

    def validate(self) -> None:
        if self.fs_hz <= 0:
            raise ValueError(f"`fs_hz` must be > 0, got {self.fs_hz}.")
        for name, value in (
            ("speed_col_index", self.speed_col_index),
            ("yaw_col_index", self.yaw_col_index),
            ("steer_angle_col_index", self.steer_angle_col_index),
        ):
            if value < 0:
                raise ValueError(f"`{name}` must be >= 0, got {value}.")
        for name, value in (
            ("speed_median_window", self.speed_median_window),
            ("accel_median_window", self.accel_median_window),
        ):
            if value <= 0 or value % 2 == 0:
                raise ValueError(
                    f"`{name}` must be a positive odd integer, got {value}."
                )

    def to_dict(self) -> dict:
        return asdict(self)

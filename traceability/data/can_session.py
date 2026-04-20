"""CAN session loading utilities with strict shape/schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

CAN_COLUMNS: Tuple[str, ...] = (
    "pedalangle",
    "pedalpressure",
    "steer_angle",
    "steer_speed",
    "speed",
    "yaw",
)
SPEED_COLUMN_INDEX = 4


@dataclass(frozen=True)
class CanSession:
    """In-memory representation of one HDD CAN session."""

    session_id: str
    values: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])


def resolve_session_path(session_id: str, data_root: Path) -> Path:
    """Resolve a session id to its expected `.npy` location under `data_root`."""
    session_path = Path(data_root) / f"{session_id}.npy"
    if not session_path.exists():
        raise FileNotFoundError(
            f"Session file not found: {session_path}. "
            f"Expected location is '<data_root>/{session_id}.npy'."
        )
    return session_path


def load_can_session(session_path: Path, session_id: str) -> CanSession:
    """Load and validate one HDD CAN session matrix (T,6) float64."""
    matrix = np.load(Path(session_path), allow_pickle=False)

    if matrix.ndim != 2:
        raise ValueError(
            f"Invalid CAN matrix rank for session '{session_id}': expected 2D, got {matrix.ndim}D."
        )
    if matrix.shape[1] != len(CAN_COLUMNS):
        raise ValueError(
            f"Invalid CAN matrix width for session '{session_id}': "
            f"expected {len(CAN_COLUMNS)} columns {CAN_COLUMNS}, got {matrix.shape[1]}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"Session '{session_id}' is empty (T=0).")
    if matrix.dtype != np.float64:
        raise ValueError(
            f"Invalid dtype for session '{session_id}': expected float64, got {matrix.dtype}."
        )

    return CanSession(session_id=session_id, values=matrix)


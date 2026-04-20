"""3 Hz CAN session loading utilities for Level-1 processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAN3HZ_NUM_COLUMNS = 8


@dataclass(frozen=True)
class CanSession3Hz:
    """In-memory representation of one 3 Hz CAN session."""

    session_id: str
    values: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])


def resolve_session_path_3hz(session_id: str, data_root: Path) -> Path:
    """Resolve a session id to `<data_root>/<session_id>.npy` for 3 Hz data."""
    session_path = Path(data_root) / f"{session_id}.npy"
    if not session_path.exists():
        raise FileNotFoundError(
            f"3 Hz session file not found: {session_path}. "
            f"Expected location is '<data_root>/{session_id}.npy'."
        )
    return session_path


def load_can_session_3hz(session_path: Path, session_id: str) -> CanSession3Hz:
    """Load and validate one 3 Hz CAN session matrix (T,8) float64."""
    matrix = np.load(Path(session_path), allow_pickle=False)

    if matrix.ndim != 2:
        raise ValueError(
            f"Invalid 3 Hz CAN matrix rank for session '{session_id}': expected 2D, got {matrix.ndim}D."
        )
    if matrix.shape[1] != CAN3HZ_NUM_COLUMNS:
        raise ValueError(
            f"Invalid 3 Hz CAN matrix width for session '{session_id}': "
            f"expected {CAN3HZ_NUM_COLUMNS} columns, got {matrix.shape[1]}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"3 Hz session '{session_id}' is empty (T=0).")
    if matrix.dtype != np.float64:
        raise ValueError(
            f"Invalid dtype for 3 Hz session '{session_id}': expected float64, got {matrix.dtype}."
        )

    return CanSession3Hz(session_id=session_id, values=matrix)


"""Batch-generate Level-3 artifacts for all 3 Hz CAN sessions."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_SUBDIR = "level3_3hz"
SEGMENTS_FILENAME = "level3_segments.json"


def _load_session_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def discover_session_ids(data_root: Path) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    session_ids = sorted(p.stem for p in data_root.glob("*.npy") if p.is_file())
    if not session_ids:
        raise RuntimeError(f"No .npy session files found under: {data_root}")
    return session_ids


def _resolve_sessions(args: argparse.Namespace) -> list[str]:
    session_ids = _load_session_ids_file(args.session_ids_file) or discover_session_ids(Path(args.data_root))
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if args.start_index >= len(session_ids):
        return []
    session_ids = session_ids[args.start_index :]
    if args.max_sessions is not None:
        if args.max_sessions <= 0:
            raise ValueError("--max-sessions must be > 0 when provided.")
        session_ids = session_ids[: args.max_sessions]
    return session_ids


def _segments_path(artifacts_root: Path, session_id: str) -> Path:
    return artifacts_root / ARTIFACT_SUBDIR / session_id / SEGMENTS_FILENAME


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Level-3 artifact generation for all HDD 3 Hz CAN sessions. "
            "Raw 3 Hz (T,8) matrices are mapped to the canonical Level-1/2/3 schema before processing."
        )
    )
    parser.add_argument("--data-root", default="data/can_data_3hz", help="Directory with 3 Hz <session_id>.npy files.")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root output directory (writes artifacts/level3_3hz/<session_id>/...).")
    parser.add_argument("--session-ids-file", default=None, help="Optional file with one session_id per line.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based start index in resolved session list.")
    parser.add_argument("--trim-start-sec", type=float, default=0.0, help="Trim seconds from start when writing artifacts.")
    parser.add_argument("--trim-end-sec", type=float, default=0.0, help="Trim seconds from end when writing artifacts.")
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable median smoothing in Level-1/2/3 preprocessing (uses all median windows=1).",
    )
    parser.add_argument("--raw3hz-pedalangle-col-index", type=int, default=0, help="3 Hz raw column index for pedal angle.")
    parser.add_argument("--raw3hz-steer-angle-col-index", type=int, default=1, help="3 Hz raw column index for steering angle.")
    parser.add_argument("--raw3hz-steer-speed-col-index", type=int, default=2, help="3 Hz raw column index for steering speed.")
    parser.add_argument("--raw3hz-speed-col-index", type=int, default=3, help="3 Hz raw column index for vehicle speed.")
    parser.add_argument("--raw3hz-pedalpressure-col-index", type=int, default=4, help="3 Hz raw column index for pedal/brake pressure.")
    parser.add_argument("--raw3hz-yaw-col-index", type=int, default=7, help="3 Hz raw column index for yaw.")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Compatibility only (ignored). This wrapper runs in-process and does not spawn subprocesses.",
    )
    parser.add_argument("--show-traceback", action="store_true", help="Print traceback on per-session failures.")
    return parser.parse_args(argv)


def _raw3hz_mapping(args: argparse.Namespace) -> dict[str, int]:
    return {
        "pedalangle": int(args.raw3hz_pedalangle_col_index),
        "pedalpressure": int(args.raw3hz_pedalpressure_col_index),
        "steer_angle": int(args.raw3hz_steer_angle_col_index),
        "steer_speed": int(args.raw3hz_steer_speed_col_index),
        "speed": int(args.raw3hz_speed_col_index),
        "yaw": int(args.raw3hz_yaw_col_index),
    }


def _load_canonical_3hz_matrix(session_id: str, data_root: Path, mapping: dict[str, int]):
    import numpy as np
    from traceability.data import load_can_session_3hz, resolve_session_path_3hz

    session = load_can_session_3hz(
        resolve_session_path_3hz(session_id=session_id, data_root=data_root),
        session_id=session_id,
    )
    raw = session.values
    width = int(raw.shape[1])
    for name, col in mapping.items():
        if col < 0 or col >= width:
            raise ValueError(f"Invalid raw 3 Hz column index for {name}: {col} (width={width})")

    return np.column_stack(
        [
            raw[:, mapping["pedalangle"]],
            raw[:, mapping["pedalpressure"]],
            raw[:, mapping["steer_angle"]],
            raw[:, mapping["steer_speed"]],
            raw[:, mapping["speed"]],
            raw[:, mapping["yaw"]],
        ]
    )


def _process_session(session_id: str, args: argparse.Namespace) -> None:
    from traceability.level1 import Level1Config, run_level1
    from traceability.level2 import Level2Config, run_level2
    from traceability.level3 import Level3Config, run_level3, write_level3_artifacts

    artifacts_root = Path(args.artifacts_root)
    data_root = Path(args.data_root)
    matrix = _load_canonical_3hz_matrix(session_id=session_id, data_root=data_root, mapping=_raw3hz_mapping(args))

    level1_median_window = 1 if bool(args.no_smoothing) else 3
    level2_yaw_median_window = 1 if bool(args.no_smoothing) else 3
    level3_median_window = 1 if bool(args.no_smoothing) else 3
    level1_config = Level1Config(fs_hz=3.0, median_window=level1_median_window, persistence_steps=1)
    level2_config = Level2Config(
        fs_hz=3.0,
        yaw_median_window=level2_yaw_median_window,
        persistence_on_steps=2,
        persistence_off_steps=3,
    )
    level3_config = Level3Config(
        fs_hz=3.0,
        speed_median_window=level3_median_window,
        accel_median_window=level3_median_window,
        yaw_median_window=level3_median_window,
        lon_persistence_steps=2,
        direction_persistence_steps=2,
    )

    level1_result = run_level1(session_id=session_id, session_matrix=matrix, config=level1_config)
    level2_result = run_level2(
        session_id=session_id,
        session_matrix=matrix,
        moving_state=level1_result.moving_state,
        config=level2_config,
    )
    level3_result = run_level3(
        session_id=session_id,
        session_matrix=matrix,
        moving_state=level1_result.moving_state,
        turn_state=level2_result.turn_state,
        config=level3_config,
    )
    write_level3_artifacts(
        result=level3_result,
        output_dir=artifacts_root / ARTIFACT_SUBDIR / session_id,
        trim_start_sec=float(args.trim_start_sec),
        trim_end_sec=float(args.trim_end_sec),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trim_start_sec < 0 or args.trim_end_sec < 0:
        raise ValueError("Trim values must be >= 0.")

    session_ids = _resolve_sessions(args)
    if not session_ids:
        print("No sessions to process (empty resolved session list).")
        return 0

    artifacts_root = Path(args.artifacts_root)
    existing_count = sum(_segments_path(artifacts_root, sid).exists() for sid in session_ids)

    print("Level-3 3Hz batch run configuration:")
    print(f"- data_root: {Path(args.data_root).as_posix()}")
    print(f"- artifacts_root: {artifacts_root.as_posix()}")
    print(f"- total_sessions_resolved: {len(session_ids)}")
    print(f"- sessions_with_existing_level3_3hz_segments: {existing_count}")
    print("- mode: overwrite-existing")
    print(f"- no_smoothing: {bool(args.no_smoothing)}")
    print(f"- raw_3hz_to_level123_mapping: {_raw3hz_mapping(args)}")
    print("- execution_mode: in-process (ignoring --python-exe)")

    successes = 0
    failures = 0
    failed_ids: list[str] = []
    start_ts = time.time()

    for idx, session_id in enumerate(session_ids, start=1):
        print(f"[{idx}/{len(session_ids)}] RUN  {session_id}")
        t0 = time.time()
        try:
            _process_session(session_id, args)
        except Exception as exc:  # pragma: no cover
            failures += 1
            failed_ids.append(session_id)
            elapsed = time.time() - t0
            print(f"[{idx}/{len(session_ids)}] FAIL {session_id} ({elapsed:.1f}s) - {exc.__class__.__name__}: {exc}")
            if args.show_traceback:
                traceback.print_exc()
            if args.fail_fast:
                total_elapsed = time.time() - start_ts
                print("Stopping due to --fail-fast.")
                print(f"Summary: success={successes} failed={failures} total_elapsed_sec={total_elapsed:.1f}")
                return 1
            continue

        successes += 1
        elapsed = time.time() - t0
        print(f"[{idx}/{len(session_ids)}] OK   {session_id} ({elapsed:.1f}s)")

    total_elapsed = time.time() - start_ts
    print("Batch run complete.")
    print(f"Summary: success={successes} failed={failures} total_elapsed_sec={total_elapsed:.1f}")
    if failed_ids:
        print(f"Failed session ids ({len(failed_ids)}): {', '.join(failed_ids)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Batch-generate Level-4 (SHARP/SMOOTH) artifacts for all 25 Hz CAN sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _resolve_level3_artifacts_root(args: argparse.Namespace) -> Path:
    if args.level3_artifacts_root:
        return Path(args.level3_artifacts_root)
    return Path(args.artifacts_root) / "level3_25hz"


def _resolve_level4_output_root(args: argparse.Namespace) -> Path:
    if args.level4_output_root:
        return Path(args.level4_output_root)
    return Path(args.artifacts_root) / "level4_25hz"


def _existing_level4_segments_path(args: argparse.Namespace, session_id: str) -> Path:
    return _resolve_level4_output_root(args) / session_id / "level4_segments.json"


def _existing_level3_segments_path(args: argparse.Namespace, session_id: str) -> Path:
    return _resolve_level3_artifacts_root(args) / session_id / "level3_segments.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Level-4 SHARP/SMOOTH inference for all HDD 25 Hz CAN sessions. "
            "Direction is inherited from Level-3 TURNING segments."
        )
    )
    parser.add_argument("--data-root", default="data/can_data_25hz", help="Directory with <session_id>.npy files.")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root artifacts directory (used to derive Level-3 and Level-4 paths by default).")
    parser.add_argument("--level3-artifacts-root", default=None, help="Override Level-3 artifacts root (default: <artifacts-root>/level3_25hz).")
    parser.add_argument("--level4-output-root", default=None, help="Override Level-4 output root (default: <artifacts-root>/level4_25hz).")
    parser.add_argument("--session-ids-file", default=None, help="Optional file with one session_id per line.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based start index in resolved session list.")
    parser.add_argument("--trim-start-sec", type=float, default=0.0, help="Trim seconds from start when writing artifacts.")
    parser.add_argument("--trim-end-sec", type=float, default=0.0, help="Trim seconds from end when writing artifacts.")
    parser.add_argument("--fs-hz", type=float, default=25.0, help="Sampling rate in Hz (default: 25.0).")

    # SHARP rule tuning (conservative defaults derived from can_data_final_v analysis).
    parser.add_argument("--min-sharp-duration-sec", type=float, default=2.0, help="Minimum turn duration eligible for SHARP.")
    parser.add_argument("--long-duration-guard-sec", type=float, default=12.0, help="If duration >= this, require stronger yaw via long-duration guard.")
    parser.add_argument("--long-duration-mean-abs-yaw-min", type=float, default=12.0, help="Mean |yaw| required for long-duration SHARP eligibility.")
    parser.add_argument("--sharp-mean-abs-yaw-min", type=float, default=10.0, help="SHARP vote threshold for mean |yaw|.")
    parser.add_argument("--sharp-peak-abs-yaw-min", type=float, default=20.0, help="SHARP vote threshold for peak |yaw|.")
    parser.add_argument("--sharp-peak-abs-steer-angle-min", type=float, default=180.0, help="SHARP vote threshold for peak |steer_angle|.")
    parser.add_argument("--sharp-peak-abs-steer-speed-min", type=float, default=240.0, help="SHARP vote threshold for peak |steer_speed|.")
    parser.add_argument("--sharp-vote-threshold", type=int, default=3, help="Minimum total aggressiveness votes to classify SHARP.")
    parser.add_argument("--allow-yawless-sharp", action="store_true", help="Disable the default requirement for at least one yaw-based SHARP vote.")
    parser.add_argument("--allow-steerless-sharp", action="store_true", help="Disable the default requirement for at least one steer-based SHARP vote.")

    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Compatibility only (ignored). This wrapper runs in-process and does not spawn subprocesses.",
    )
    parser.add_argument("--show-traceback", action="store_true", help="Print traceback on per-session failures.")
    return parser.parse_args(argv)


def _build_rule_config(args: argparse.Namespace):
    from traceability.level4 import Level4RuleConfig

    return Level4RuleConfig(
        fs_hz=float(args.fs_hz),
        min_sharp_duration_sec=float(args.min_sharp_duration_sec),
        long_duration_guard_sec=float(args.long_duration_guard_sec),
        long_duration_mean_abs_yaw_min=float(args.long_duration_mean_abs_yaw_min),
        sharp_mean_abs_yaw_min=float(args.sharp_mean_abs_yaw_min),
        sharp_peak_abs_yaw_min=float(args.sharp_peak_abs_yaw_min),
        sharp_peak_abs_steer_angle_min=float(args.sharp_peak_abs_steer_angle_min),
        sharp_peak_abs_steer_speed_min=float(args.sharp_peak_abs_steer_speed_min),
        sharp_vote_threshold=int(args.sharp_vote_threshold),
        require_yaw_vote=not bool(args.allow_yawless_sharp),
        require_steer_vote=not bool(args.allow_steerless_sharp),
    )


def _process_session(session_id: str, args: argparse.Namespace) -> None:
    from traceability.data import load_can_session, resolve_session_path
    from traceability.level4 import run_level4, write_level4_artifacts

    data_root = Path(args.data_root)
    l3_root = _resolve_level3_artifacts_root(args)
    l4_root = _resolve_level4_output_root(args)
    session = load_can_session(resolve_session_path(session_id=session_id, data_root=data_root), session_id=session_id)

    level3_segments_path = l3_root / session_id / "level3_segments.json"
    level3_segments = json.loads(level3_segments_path.read_text(encoding="utf-8"))
    if not isinstance(level3_segments, list):
        raise ValueError(f"Invalid Level-3 segments JSON (expected list): {level3_segments_path}")

    result = run_level4(
        session_id=session_id,
        session_matrix=session.values,
        level3_segments=level3_segments,
        config=_build_rule_config(args),
    )
    write_level4_artifacts(
        result=result,
        output_dir=l4_root / session_id,
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

    l3_root = _resolve_level3_artifacts_root(args)
    l4_root = _resolve_level4_output_root(args)
    existing_l4_count = sum(_existing_level4_segments_path(args, sid).exists() for sid in session_ids)
    existing_l3_count = sum(_existing_level3_segments_path(args, sid).exists() for sid in session_ids)
    cfg = _build_rule_config(args)
    cfg.validate()

    print("Level-4 25Hz batch run configuration:")
    print(f"- data_root: {Path(args.data_root).as_posix()}")
    print(f"- artifacts_root: {Path(args.artifacts_root).as_posix()}")
    print(f"- level3_artifacts_root: {l3_root.as_posix()}")
    print(f"- level4_output_root: {l4_root.as_posix()}")
    print(f"- total_sessions_resolved: {len(session_ids)}")
    print(f"- sessions_with_existing_level3_segments: {existing_l3_count}")
    print(f"- sessions_with_existing_level4_segments: {existing_l4_count}")
    print("- mode: overwrite-existing")
    print(f"- execution_mode: in-process (ignoring --python-exe)")
    print(f"- rule_config: {cfg.to_dict()}")

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

"""Batch-generate final segment outputs with metrics for all 25 Hz sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from traceability.data import load_can_session, resolve_session_path
from traceability.final.artifacts import write_final_artifacts
from traceability.final.pipeline import build_final_session_outputs
from traceability.final.types import FinalMetricsConfig


def _load_session_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _discover_from_level4_root(level4_root: Path) -> list[str]:
    if not level4_root.exists():
        raise FileNotFoundError(f"Level-4 root not found: {level4_root}")
    if not level4_root.is_dir():
        raise NotADirectoryError(f"Level-4 root is not a directory: {level4_root}")

    session_ids = sorted(
        p.name
        for p in level4_root.iterdir()
        if p.is_dir() and (p / "level4_segments.json").exists()
    )
    if not session_ids:
        raise RuntimeError(
            "No session folders with level4_segments.json found under "
            f"{level4_root}"
        )
    return session_ids


def _resolve_sessions(args: argparse.Namespace) -> list[str]:
    session_ids = _load_session_ids_file(args.session_ids_file) or _discover_from_level4_root(
        Path(args.level4_root)
    )
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final 25 Hz segment outputs with metrics by combining "
            "Level-4 segments and CAN signal values."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/can_data_25hz",
        help="Directory containing <session_id>.npy CAN files.",
    )
    parser.add_argument(
        "--level4-root",
        default="artifacts/level4_25hz",
        help="Directory containing level4_25hz/<session_id>/level4_segments.json.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/final",
        help="Output directory root for final results.",
    )
    parser.add_argument(
        "--session-ids-file",
        default=None,
        help="Optional file with one session id per line.",
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based start index in resolved session list.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    parser.add_argument("--show-traceback", action="store_true", help="Print traceback on per-session failures.")

    parser.add_argument("--fs-hz", type=float, default=25.0, help="Sampling rate in Hz.")
    parser.add_argument(
        "--speed-median-window",
        type=int,
        default=17,
        help="Centered median window for speed before acceleration derivation.",
    )
    parser.add_argument(
        "--accel-median-window",
        type=int,
        default=11,
        help="Centered median window for acceleration smoothing.",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> FinalMetricsConfig:
    cfg = FinalMetricsConfig(
        fs_hz=float(args.fs_hz),
        speed_median_window=int(args.speed_median_window),
        accel_median_window=int(args.accel_median_window),
    )
    cfg.validate()
    return cfg


def _load_level4_inputs(level4_root: Path, session_id: str) -> tuple[list[dict], int, dict[str, str]]:
    session_dir = level4_root / session_id
    segments_path = session_dir / "level4_segments.json"
    metadata_path = session_dir / "level4_run_metadata.json"

    if not segments_path.exists():
        raise FileNotFoundError(f"Missing Level-4 segments: {segments_path}")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list):
        raise ValueError(f"Invalid Level-4 segments JSON (expected list): {segments_path}")

    timeline_start_idx = 0
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coverage = metadata.get("coverage", {})
        timeline_start_idx = int(coverage.get("trimmed_start_idx", 0))

    source_paths = {
        "level4_segments_json": str(segments_path),
        "level4_run_metadata_json": str(metadata_path) if metadata_path.exists() else "",
    }
    return segments, timeline_start_idx, source_paths


def _process_session(
    session_id: str,
    *,
    args: argparse.Namespace,
    config: FinalMetricsConfig,
) -> None:
    data_root = Path(args.data_root)
    level4_root = Path(args.level4_root)
    output_root = Path(args.output_root)

    session_path = resolve_session_path(session_id=session_id, data_root=data_root)
    can_session = load_can_session(session_path=session_path, session_id=session_id)

    level4_segments, timeline_start_idx, source_paths = _load_level4_inputs(
        level4_root=level4_root,
        session_id=session_id,
    )
    source_paths["can_session_npy"] = str(session_path)

    final_segments, run_metadata = build_final_session_outputs(
        session_id=session_id,
        session_matrix=can_session.values,
        level4_segments=level4_segments,
        config=config,
        level4_timeline_start_idx=timeline_start_idx,
        source_paths=source_paths,
    )
    write_final_artifacts(
        segments=final_segments,
        metadata=run_metadata,
        output_dir=output_root / session_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _build_config(args)
    session_ids = _resolve_sessions(args)
    if not session_ids:
        print("No sessions to process (empty resolved session list).")
        return 0

    print("Final results 25Hz batch run configuration:")
    print(f"- data_root: {Path(args.data_root).as_posix()}")
    print(f"- level4_root: {Path(args.level4_root).as_posix()}")
    print(f"- output_root: {Path(args.output_root).as_posix()}")
    print(f"- total_sessions_resolved: {len(session_ids)}")
    print(f"- metrics_config: {config.to_dict()}")

    successes = 0
    failures = 0
    failed_ids: list[str] = []
    start_ts = time.time()

    for idx, session_id in enumerate(session_ids, start=1):
        print(f"[{idx}/{len(session_ids)}] RUN  {session_id}")
        t0 = time.time()
        try:
            _process_session(session_id, args=args, config=config)
        except Exception as exc:  # pragma: no cover
            failures += 1
            failed_ids.append(session_id)
            elapsed = time.time() - t0
            print(
                f"[{idx}/{len(session_ids)}] FAIL {session_id} ({elapsed:.1f}s) - "
                f"{exc.__class__.__name__}: {exc}"
            )
            if args.show_traceback:
                traceback.print_exc()
            if args.fail_fast:
                total_elapsed = time.time() - start_ts
                print("Stopping due to --fail-fast.")
                print(
                    f"Summary: success={successes} failed={failures} "
                    f"total_elapsed_sec={total_elapsed:.1f}"
                )
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

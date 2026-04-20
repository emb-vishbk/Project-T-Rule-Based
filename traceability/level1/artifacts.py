"""Artifact writers for Level-1 outputs."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .pipeline import Level1Result


def write_level1_artifacts(
    result: Level1Result,
    output_dir: Path,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
) -> Dict[str, Path]:
    """
    Write Level-1 outputs for one session.

    Files:
    - `level1_segments.csv`
    - `level1_segments.json`
    - `level1_run_metadata.json`
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_csv = output_dir / "level1_segments.csv"
    segments_json = output_dir / "level1_segments.json"
    metadata_json = output_dir / "level1_run_metadata.json"

    # Clean stale file from older schema versions so each run yields only 3 artifacts.
    legacy_timestep_csv = output_dir / "level1_timestep_labels.csv"
    if legacy_timestep_csv.exists():
        try:
            legacy_timestep_csv.unlink()
        except PermissionError:
            logging.warning(
                "Could not remove legacy artifact (permission denied): %s",
                legacy_timestep_csv.as_posix(),
            )

    report_segments, trim_context = _build_report_segments(
        result=result,
        trim_start_sec=trim_start_sec,
        trim_end_sec=trim_end_sec,
    )
    _write_segments_csv(report_segments, segments_csv)
    _write_segments_json(report_segments, segments_json)
    _write_metadata_json(result, report_segments, trim_context, metadata_json)

    return {
        "metadata_json": metadata_json,
        "segments_csv": segments_csv,
        "segments_json": segments_json,
    }


def _write_segments_csv(report_segments: List[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "segment_id",
                "timesteps",
                "duration",
                "starting_time",
                "ending_time",
                "label_vector",
            ]
        )
        for seg in report_segments:
            writer.writerow(
                [
                    seg["segment_id"],
                    seg["timesteps"],
                    seg["duration"],
                    seg["starting_time"],
                    seg["ending_time"],
                    "[" + "".join(seg["label_vector"]) + "]",
                ]
            )


def _write_segments_json(report_segments: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(report_segments, fp, indent=2)


def _write_metadata_json(
    result: Level1Result,
    report_segments: List[dict],
    trim_context: dict,
    output_path: Path,
) -> None:
    effective_state = result.moving_state[trim_context["keep_start_idx"] : trim_context["keep_end_idx"]]
    stopped_steps = int((effective_state == 0).sum())
    moving_steps = int((effective_state == 1).sum())
    effective_steps = int(len(effective_state))

    payload = {
        "session_id": result.session_id,
        "num_steps": effective_steps,
        "num_segments": int(len(report_segments)),
        "source_num_steps": int(len(result.moving_state)),
        "source_duration_sec": float(len(result.moving_state) / result.config.fs_hz),
        "effective_duration_sec": float(effective_steps / result.config.fs_hz),
        "coverage": {
            "trimmed_start_idx": trim_context["keep_start_idx"],
            "trimmed_end_idx": trim_context["keep_end_idx"],
            "first_start_idx": 0 if report_segments else None,
            "last_end_idx": effective_steps if report_segments else None,
            "expected_total_steps": effective_steps,
        },
        "level1_state_counts": {
            "STOPPED": stopped_steps,
            "MOVING": moving_steps,
        },
        "time_alignment": {
            "trim_start_sec_input": trim_context["trim_start_sec_input"],
            "trim_end_sec_input": trim_context["trim_end_sec_input"],
            "trim_start_steps_used": trim_context["trim_start_steps"],
            "trim_end_steps_used": trim_context["trim_end_steps"],
            "trim_start_sec_used": trim_context["trim_start_sec_used"],
            "trim_end_sec_used": trim_context["trim_end_sec_used"],
            "timeline_origin": "trimmed_window_start",
        },
        "config": {
            "fs_hz": result.config.fs_hz,
            "speed_col_index": result.config.speed_col_index,
            "median_window": result.config.median_window,
            "v_stopped": result.config.v_stopped,
            "v_moving": result.config.v_moving,
            "persistence_steps": result.config.persistence_steps,
            "initial_state": result.config.initial_state,
        },
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def _build_report_segments(
    result: Level1Result,
    trim_start_sec: float,
    trim_end_sec: float,
) -> Tuple[List[dict], dict]:
    fs = float(result.config.fs_hz)
    total_steps = int(len(result.moving_state))

    if trim_start_sec < 0:
        raise ValueError(f"`trim_start_sec` must be >= 0, got {trim_start_sec}.")
    if trim_end_sec < 0:
        raise ValueError(f"`trim_end_sec` must be >= 0, got {trim_end_sec}.")

    trim_start_steps = _seconds_to_steps(trim_start_sec, fs)
    trim_end_steps = _seconds_to_steps(trim_end_sec, fs)
    keep_start_idx = trim_start_steps
    keep_end_idx = total_steps - trim_end_steps

    if keep_start_idx >= keep_end_idx:
        raise ValueError(
            "Invalid trim window: after applying start/end trims there are no timesteps left. "
            f"total_steps={total_steps}, trim_start_steps={trim_start_steps}, trim_end_steps={trim_end_steps}."
        )

    report_segments: List[dict] = []
    for seg in result.segments:
        overlap_start = max(seg.start_idx, keep_start_idx)
        overlap_end = min(seg.end_idx, keep_end_idx)
        if overlap_end <= overlap_start:
            continue

        rebased_start_idx = overlap_start - keep_start_idx
        rebased_end_idx = overlap_end - keep_start_idx
        timesteps = rebased_end_idx - rebased_start_idx
        duration_sec = timesteps / fs

        report_segments.append(
            {
                "segment_id": len(report_segments),
                "timesteps": int(timesteps),
                "duration": _format_duration_verbose(duration_sec),
                "starting_time": _format_mmss_ss(rebased_start_idx / fs),
                "ending_time": _format_mmss_ss(rebased_end_idx / fs),
                "label_vector": list(seg.label_vector),
            }
        )

    if not report_segments:
        raise ValueError("No segments remain after trim window projection.")

    trim_context = {
        "trim_start_sec_input": float(trim_start_sec),
        "trim_end_sec_input": float(trim_end_sec),
        "trim_start_steps": int(trim_start_steps),
        "trim_end_steps": int(trim_end_steps),
        "trim_start_sec_used": float(trim_start_steps / fs),
        "trim_end_sec_used": float(trim_end_steps / fs),
        "keep_start_idx": int(keep_start_idx),
        "keep_end_idx": int(keep_end_idx),
    }
    return report_segments, trim_context


def _seconds_to_steps(seconds: float, fs_hz: float) -> int:
    # Snap trims to the nearest 10 Hz sample to keep step counts integral.
    return int(round(seconds * fs_hz))


def _format_duration_verbose(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d} mins {seconds:05.2f} seconds"


def _format_mmss_ss(duration_sec: float) -> str:
    minutes = int(duration_sec // 60.0)
    seconds = duration_sec - minutes * 60.0
    return f"{minutes:02d}:{seconds:05.2f}"

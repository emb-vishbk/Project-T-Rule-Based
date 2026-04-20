"""Artifact writers for final per-segment metric outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def write_final_artifacts(
    *,
    segments: List[dict],
    metadata: dict,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments_csv = output_dir / "segments.csv"
    segments_json = output_dir / "segments.json"
    metadata_json = output_dir / "run_metadata.json"

    _write_segments_csv(segments, segments_csv)
    _write_segments_json(segments, segments_json)
    _write_metadata_json(metadata, metadata_json)

    return {
        "segments_csv": segments_csv,
        "segments_json": segments_json,
        "metadata_json": metadata_json,
    }


def _write_segments_csv(segments: List[dict], output_path: Path) -> None:
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
                "metrics",
            ]
        )
        for seg in segments:
            writer.writerow(
                [
                    int(seg["segment_id"]),
                    int(seg["timesteps"]),
                    str(seg["duration"]),
                    str(seg["starting_time"]),
                    str(seg["ending_time"]),
                    "[" + "".join(list(seg["label_vector"])) + "]",
                    json.dumps(seg.get("metrics", {}), ensure_ascii=True, separators=(",", ":")),
                ]
            )


def _write_segments_json(segments: List[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")


def _write_metadata_json(metadata: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

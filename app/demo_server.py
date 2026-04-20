#!/usr/bin/env python3
"""Local demo app server for synchronized driving video and Level-4 segments."""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


SESSION_ID_PATTERN = re.compile(r"(20\d{10})")
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".webm", ".mkv")
CHUNK_SIZE = 1024 * 1024
CLIENT_DISCONNECT_WINERRORS = {10053, 10054}
CLIENT_DISCONNECT_ERRNOS = {errno.EPIPE, errno.ECONNABORTED, errno.ECONNRESET}


@dataclass(frozen=True)
class SessionRecord:
    """Session-level metadata and artifact paths."""

    session_id: str
    metadata_path: Path
    segments_path: Path
    duration_sec: float
    num_segments: int
    num_turn_events: int
    sharp_turn_events: int


@dataclass
class AppState:
    """In-memory server state."""

    static_root: Path
    sessions: dict[str, SessionRecord]
    video_paths: dict[str, Path]
    featured_sessions: list[str]


def parse_time_to_seconds(value: str) -> float:
    """Convert time strings like MM:SS.xx or HH:MM:SS.xx into seconds."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Unsupported time format: {value!r}")


def find_session_id(text: str) -> str | None:
    """Return first 12-digit session id found in text."""
    match = SESSION_ID_PATTERN.search(text)
    return match.group(1) if match else None


def load_sessions(artifacts_root: Path) -> dict[str, SessionRecord]:
    """Load all available final-result sessions from artifacts root."""
    sessions: dict[str, SessionRecord] = {}
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")

    for session_dir in sorted(artifacts_root.iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        metadata_path = session_dir / "run_metadata.json"
        segments_path = session_dir / "segments.json"
        if not metadata_path.exists() or not segments_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            turn_stats = metadata.get("turn_event_stats", {})
            if not isinstance(turn_stats, dict):
                turn_stats = {}
            metrics_summary = metadata.get("metrics_summary", {})
            if not isinstance(metrics_summary, dict):
                metrics_summary = {}
            coverage = metadata.get("coverage", {})
            if not isinstance(coverage, dict):
                coverage = {}
            record = SessionRecord(
                session_id=session_id,
                metadata_path=metadata_path,
                segments_path=segments_path,
                duration_sec=float(
                    metadata.get(
                        "source_duration_sec",
                        coverage.get("timeline_duration_sec", 0.0),
                    )
                ),
                num_segments=int(metadata.get("num_segments", 0)),
                num_turn_events=int(
                    turn_stats.get(
                        "num_turn_events",
                        metrics_summary.get("turning_segments", 0),
                    )
                ),
                sharp_turn_events=int(turn_stats.get("sharp_turn_events", 0)),
            )
            sessions[session_id] = record
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            print(f"[WARN] Skipping bad metadata for {session_id}: {exc}", file=sys.stderr)

    if not sessions:
        raise RuntimeError(f"No session artifacts found under: {artifacts_root}")
    return sessions


def load_video_map(map_path: Path | None, valid_sessions: set[str]) -> dict[str, Path]:
    """Load optional session->video mapping from a JSON file."""
    if map_path is None:
        return {}

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError(f"Video map must be JSON object, got: {type(mapping).__name__}")

    resolved: dict[str, Path] = {}
    for session_id, raw_path in mapping.items():
        if session_id not in valid_sessions:
            continue
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (map_path.parent / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            resolved[session_id] = candidate.resolve()
    return resolved


def is_better_video_candidate(
    candidate: Path,
    current: Path,
    ext_priority: dict[str, int],
) -> bool:
    """Choose preferred video when multiple files match one session id."""

    def score(path: Path) -> tuple[int, int, int, str]:
        return (
            ext_priority.get(path.suffix.lower(), 99),
            len(path.parts),
            len(path.name),
            str(path).lower(),
        )

    return score(candidate) < score(current)


def scan_video_root(
    video_root: Path | None,
    valid_sessions: set[str],
    extensions: set[str],
    existing: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Scan video root recursively and map files to session id."""
    resolved = dict(existing or {})
    if video_root is None:
        return resolved
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    ext_priority = {ext: i for i, ext in enumerate(sorted(extensions))}

    for file_path in sorted(video_root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions:
            continue

        session_id = find_session_id(file_path.name)
        if session_id is None or session_id not in valid_sessions:
            continue

        current = resolved.get(session_id)
        if current is None or is_better_video_candidate(file_path, current, ext_priority):
            resolved[session_id] = file_path.resolve()

    return resolved


def choose_featured_sessions(
    sessions: dict[str, SessionRecord],
    video_paths: dict[str, Path],
    explicit: list[str],
    limit: int = 4,
) -> list[str]:
    """Pick featured sessions (explicit list or automatic ranking)."""
    if explicit:
        return [session_id for session_id in explicit if session_id in sessions][:limit]

    candidate_ids = [sid for sid in sessions if sid in video_paths]
    if not candidate_ids:
        candidate_ids = list(sessions.keys())

    def rank(session_id: str) -> tuple[float, float]:
        item = sessions[session_id]
        minutes = max(item.duration_sec / 60.0, 1e-6)
        turn_density = item.num_turn_events / minutes
        score = (
            turn_density * 8.0
            + item.sharp_turn_events * 0.35
            + item.num_turn_events * 0.05
        )
        return (score, -item.duration_sec)

    ranked = sorted(candidate_ids, key=rank, reverse=True)
    return ranked[:limit]


def load_segments_for_session(record: SessionRecord) -> list[dict[str, Any]]:
    """Load session segments JSON and add numeric start/end seconds."""
    raw_segments = json.loads(record.segments_path.read_text(encoding="utf-8"))
    if not isinstance(raw_segments, list):
        raise ValueError(f"Segments for {record.session_id} are not a list")

    normalized: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        start_time = str(item.get("starting_time", "0:00.00"))
        end_time = str(item.get("ending_time", "0:00.00"))
        copy = dict(item)
        copy["starting_sec"] = parse_time_to_seconds(start_time)
        copy["ending_sec"] = parse_time_to_seconds(end_time)
        normalized.append(copy)
    return normalized


def build_handler(state: AppState) -> type[BaseHTTPRequestHandler]:
    """Create request handler class bound to app state."""

    class DemoRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self.serve_static_file(state.static_root / "index.html")
                return
            if path.startswith("/static/"):
                relative = path.removeprefix("/static/")
                target = (state.static_root / relative).resolve()
                if state.static_root not in target.parents:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self.serve_static_file(target)
                return
            if path == "/api/sessions":
                self.handle_sessions_list()
                return
            if path.startswith("/api/session/"):
                session_id = unquote(path.removeprefix("/api/session/")).strip()
                self.handle_session_details(session_id)
                return
            if path.startswith("/video/"):
                session_id = unquote(path.removeprefix("/video/")).strip()
                self.serve_video_for_session(session_id)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} - {fmt % args}", file=sys.stderr)

        def send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def is_client_disconnect(self, exc: BaseException) -> bool:
            """Return True when exception indicates the client closed the connection."""
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                return True
            if not isinstance(exc, OSError):
                return False

            if exc.errno in CLIENT_DISCONNECT_ERRNOS:
                return True
            winerror = getattr(exc, "winerror", None)
            if winerror in CLIENT_DISCONNECT_WINERRORS:
                return True
            return False

        def safe_send_error(self, status: HTTPStatus, message: str) -> None:
            """Try sending an HTTP error without crashing on disconnected clients."""
            try:
                self.send_error(status, message)
            except OSError as exc:
                if not self.is_client_disconnect(exc):
                    raise

        def serve_static_file(self, target: Path) -> None:
            if not target.exists() or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
                return

            mime_type, _ = mimetypes.guess_type(str(target))
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def handle_sessions_list(self) -> None:
            featured = set(state.featured_sessions)
            payload = {
                "featured_sessions": state.featured_sessions,
                "total_sessions": len(state.sessions),
                "video_mapped_sessions": len(state.video_paths),
                "sessions": [
                    {
                        "session_id": item.session_id,
                        "duration_sec": item.duration_sec,
                        "num_segments": item.num_segments,
                        "num_turn_events": item.num_turn_events,
                        "sharp_turn_events": item.sharp_turn_events,
                        "has_video": item.session_id in state.video_paths,
                        "featured": item.session_id in featured,
                    }
                    for item in sorted(state.sessions.values(), key=lambda x: x.session_id)
                ],
            }
            self.send_json(payload)

        def handle_session_details(self, session_id: str) -> None:
            record = state.sessions.get(session_id)
            if record is None:
                self.send_json(
                    {"error": f"Session '{session_id}' not found in level4 artifacts."},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            try:
                metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
                segments = load_segments_for_session(record)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(
                    {"error": f"Failed to load session '{session_id}': {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

            video_path = state.video_paths.get(session_id)
            payload = {
                "session_id": record.session_id,
                "metadata": metadata,
                "segments": segments,
                "video_url": f"/video/{record.session_id}" if video_path else None,
                "video_filename": video_path.name if video_path else None,
                "video_size_bytes": video_path.stat().st_size if video_path else None,
            }
            self.send_json(payload)

        def serve_video_for_session(self, session_id: str) -> None:
            video_path = state.video_paths.get(session_id)
            if video_path is None:
                self.safe_send_error(
                    HTTPStatus.NOT_FOUND,
                    f"No mapped video file found for session {session_id}",
                )
                return
            if not video_path.exists() or not video_path.is_file():
                self.safe_send_error(HTTPStatus.NOT_FOUND, "Mapped video file is missing")
                return

            file_size = video_path.stat().st_size
            content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
            range_header = self.headers.get("Range")

            try:
                with video_path.open("rb") as handle:
                    if range_header:
                        start, end = self.parse_range_header(range_header, file_size)
                        if start is None or end is None:
                            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                            self.send_header("Content-Range", f"bytes */{file_size}")
                            self.send_header("Accept-Ranges", "bytes")
                            self.end_headers()
                            return

                        content_length = end - start + 1
                        self.send_response(HTTPStatus.PARTIAL_CONTENT)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                        self.send_header("Content-Length", str(content_length))
                        self.end_headers()

                        handle.seek(start)
                        self.stream_file(handle, content_length)
                        return

                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(file_size))
                    self.end_headers()
                    self.stream_file(handle, file_size)
            except OSError as exc:
                if self.is_client_disconnect(exc):
                    return
                self.safe_send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Video read error: {exc}")

        def parse_range_header(
            self,
            range_header: str,
            file_size: int,
        ) -> tuple[int | None, int | None]:
            """Parse HTTP Range header for a single bytes range."""
            value = range_header.strip()
            if not value.startswith("bytes="):
                return (None, None)

            first_range = value[6:].split(",", 1)[0].strip()
            match = re.fullmatch(r"(\d*)-(\d*)", first_range)
            if not match:
                return (None, None)

            start_group, end_group = match.groups()
            if not start_group and not end_group:
                return (None, None)

            if not start_group:
                suffix_length = int(end_group)
                if suffix_length <= 0:
                    return (None, None)
                start = max(file_size - suffix_length, 0)
                end = file_size - 1
            else:
                start = int(start_group)
                end = int(end_group) if end_group else file_size - 1
                if start >= file_size:
                    return (None, None)
                end = min(end, file_size - 1)

            if start > end:
                return (None, None)
            return (start, end)

        def stream_file(self, handle: Any, remaining: int) -> None:
            while remaining > 0:
                chunk = handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except OSError as exc:
                    if self.is_client_disconnect(exc):
                        break
                    raise
                remaining -= len(chunk)

    return DemoRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve synchronized session video + Level-4 segment demo UI "
            "on a local web server."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/final/results"),
        help="Path to final results root (default: artifacts/final/results).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Optional directory that contains driving session video files. "
            "Files are matched by 12-digit session id in filename."
        ),
    )
    parser.add_argument(
        "--video-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping session_id to absolute/relative video file path. "
            "Relative paths are resolved from the map file location."
        ),
    )
    parser.add_argument(
        "--featured-sessions",
        type=str,
        default="",
        help="Comma-separated session ids to show as featured suggestions.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_VIDEO_EXTENSIONS),
        help="Comma-separated video extensions to scan (e.g. .mp4,.mov,.mkv).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app_root = Path(__file__).resolve().parent
    static_root = (app_root / "static").resolve()
    if not static_root.exists():
        raise FileNotFoundError(f"Missing static folder: {static_root}")

    artifacts_root = args.artifacts_root.resolve()
    sessions = load_sessions(artifacts_root)
    valid_session_ids = set(sessions.keys())

    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    }
    if not extensions:
        raise ValueError("At least one video extension is required.")

    video_paths = load_video_map(args.video_map.resolve() if args.video_map else None, valid_session_ids)
    video_paths = scan_video_root(
        args.video_root.resolve() if args.video_root else None,
        valid_session_ids,
        extensions,
        existing=video_paths,
    )

    explicit_featured = [
        value.strip()
        for value in args.featured_sessions.split(",")
        if value.strip()
    ]
    featured_sessions = choose_featured_sessions(
        sessions=sessions,
        video_paths=video_paths,
        explicit=explicit_featured,
        limit=4,
    )

    state = AppState(
        static_root=static_root,
        sessions=sessions,
        video_paths=video_paths,
        featured_sessions=featured_sessions,
    )

    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"[INFO] Sessions loaded: {len(sessions)} from {artifacts_root}")
    print(f"[INFO] Videos mapped: {len(video_paths)}")
    if featured_sessions:
        print(f"[INFO] Featured sessions: {', '.join(featured_sessions)}")
    else:
        print("[INFO] Featured sessions: none")
    print(f"[INFO] Open demo at: http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

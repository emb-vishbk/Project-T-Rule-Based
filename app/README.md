# Driving Session Demo App

This folder contains a local web demo that synchronizes:

- session video playback (left side)
- Level-4 segment JSON objects from `artifacts/level4_25hz/<session_id>/level4_segments.json` (right side)

As the video time changes, the active segment row is highlighted and auto-scrolled.

## Run

From project root:

```bash
python3 app/demo_server.py \
  --video-root "/absolute/path/to/driving_session_videos"
```

Then open:

`http://127.0.0.1:8080`

## Video Mapping Options

1. Automatic scan (`--video-root`)
- The server scans recursively for video files (`.mp4,.m4v,.mov,.webm,.mkv` by default).
- A file is mapped when its filename contains the 12-digit session id (for example `201702271017.mp4`).

2. Explicit map (`--video-map`)
- Use a JSON map file for exact paths:

```bash
python3 app/demo_server.py \
  --video-map app/video_map.sample.json
```

Relative paths in map files are resolved from the map file location.

## Useful Flags

```bash
python3 app/demo_server.py \
  --video-root "/path/to/videos" \
  --featured-sessions "201706141720,201706141033,201703081055,201702271017" \
  --host 127.0.0.1 \
  --port 8080
```

## Notes for Large Videos (800 MB to 1.5 GB)

- This setup is suitable for local demos because streaming uses HTTP byte-range requests.
- Browser seek performance depends on codec/container. `H.264 + AAC in MP4` usually gives best compatibility.
- Keep videos on SSD for smoother scrubbing.

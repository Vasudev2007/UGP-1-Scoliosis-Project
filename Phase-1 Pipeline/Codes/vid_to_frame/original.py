#!/usr/bin/env python3
"""
Headless video to frame extractor.

Defaults:
  - Input directory:   <repo>/Scoliosis/input
  - Output directory:  <repo>/Scoliosis/working/Common/frames
  - Frames per video:  45

Usage:
    python vid_to_frame.py
    python vid_to_frame.py --config /path/to/config.json
    python vid_to_frame.py --input-dir ... --output-dir ... --frames 60

Config JSON keys:
    {
        "input_dir": "...",
        "output_dir": "...",
        "frames": 45,
        "videos": ["p_env_far.mp4", "p_env_close.mp4"],   # optional whitelist
        "source_dir": "/path/to/original/videos"          # optional sync source
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

LOGGER = logging.getLogger("vid_to_frame")

DEFAULT_FRAMES = 45
VALID_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_dir() -> Path:
    return repo_root() / "Scoliosis" / "input"


def default_output_dir() -> Path:
    return repo_root() / "Scoliosis" / "working" / "Common" / "frames"


def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def list_videos(input_dir: Path, whitelist: Optional[Iterable[str]] = None) -> List[Path]:
    if whitelist:
        return [input_dir / name for name in whitelist if (input_dir / name).exists()]
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS)


def compute_frame_indices(total_frames: int, num_samples: int) -> List[int]:
    if total_frames <= 0:
        return []
    if num_samples >= total_frames:
        return list(range(total_frames))
    step = total_frames / float(num_samples)
    return sorted({min(total_frames - 1, int(i * step)) for i in range(num_samples)})


def extract_frames(video_path: Path, output_dir: Path, num_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = compute_frame_indices(total_frames, num_frames)
    if not indices:
        LOGGER.warning("Skipping %s: unable to compute frame indices", video_path.name)
        cap.release()
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = output_dir / f"frame_{saved:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        saved += 1
    cap.release()
    if saved:
        LOGGER.info("Saved %d frame(s) for %s to %s", saved, video_path.name, output_dir)
    else:
        LOGGER.warning("No frames extracted for %s (check source video)", video_path.name)
    return saved


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless video to frame extractor.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--input-dir", help="Input directory containing videos.")
    parser.add_argument("--output-dir", help="Directory to place extracted frames.")
    parser.add_argument("--frames", type=int, help="Frames to sample per video.")
    parser.add_argument("--videos", nargs="*", help="Optional list of video filenames to process.")
    parser.add_argument("--source-dir", help="Optional fallback directory containing source videos.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    cfg = load_config(args.config)

    input_dir = Path(args.input_dir or cfg.get("input_dir", default_input_dir())).expanduser()
    output_dir = Path(args.output_dir or cfg.get("output_dir", default_output_dir())).expanduser()
    num_frames = int(args.frames or cfg.get("frames", DEFAULT_FRAMES))
    whitelist = args.videos or cfg.get("videos")
    source_dir_val = args.source_dir or cfg.get("source_dir")
    source_dir = Path(source_dir_val).expanduser() if source_dir_val else None

    if num_frames <= 0:
        raise ValueError("frames must be positive")

    if source_dir and source_dir.exists():
        videos_to_copy = list_videos(source_dir, whitelist)
        if videos_to_copy:
            input_dir.mkdir(parents=True, exist_ok=True)
            for src in videos_to_copy:
                dest = input_dir / src.name
                try:
                    if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                        shutil.copy2(src, dest)
                        LOGGER.info("Synced %s → %s", src, dest)
                except Exception as exc:
                    LOGGER.warning("Failed to copy %s → %s: %s", src, dest, exc)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(input_dir, whitelist)
    if not videos:
        LOGGER.warning("No videos found in %s", input_dir)
        return 0

    LOGGER.info(
        "Processing %d video(s) from %s → %s (frames=%d)",
        len(videos),
        input_dir,
        output_dir,
        num_frames,
    )
    total_saved = 0
    for video in videos:
        target_dir = output_dir / video.stem
        saved = extract_frames(video, target_dir, num_frames)
        total_saved += saved
    LOGGER.info("Extraction complete. Saved %d frames across %d video(s).", total_saved, len(videos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

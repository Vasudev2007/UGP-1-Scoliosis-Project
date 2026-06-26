#!/usr/bin/env python3
"""
Headless video to frame extractor.

Behavior change: If --output-root or env SCOL_OUTPUT_ROOT is provided,
the script will prefer <OUTPUT_ROOT>/Scoliosis/input as the input directory
unless --input-dir or the config JSON explicitly specifies input_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
    # Backwards-compatible repo-root default
    return repo_root() / "Scoliosis" / "input"


def default_output_frames_dir() -> Path:
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


def _extract_by_indices_from_capture(cap: cv2.VideoCapture, indices: List[int], output_dir: Path) -> int:
    saved = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            LOGGER.debug("Failed to read frame %d", idx)
            continue
        frame_path = output_dir / f"frame_{saved:04d}.png"
        try:
            cv2.imwrite(str(frame_path), frame)
            saved += 1
        except Exception as exc:
            LOGGER.warning("Failed to write %s: %s", frame_path, exc)
    return saved


def _read_all_frames(cap: cv2.VideoCapture) -> List:
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        return frames


def extract_frames(video_path: Path, output_dir: Path, num_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Unable to open video %s — skipping", video_path.name)
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        indices = compute_frame_indices(total_frames, num_frames)
        if not indices:
            LOGGER.warning("Skipping %s: unable to compute frame indices (count=%s)", video_path.name, total_frames)
            cap.release()
            return 0
        saved = _extract_by_indices_from_capture(cap, indices, output_dir)
        cap.release()
    else:
        LOGGER.info("CAP_PROP_FRAME_COUNT unavailable for %s — reading frames to compute indices", video_path.name)
        frames = _read_all_frames(cap)
        cap.release()
        total_frames = len(frames)
        if total_frames == 0:
            LOGGER.warning("Skipping %s: no frames read from source", video_path.name)
            return 0
        indices = compute_frame_indices(total_frames, num_frames)
        saved = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            frame = frames[idx]
            frame_path = output_dir / f"frame_{saved:04d}.png"
            try:
                cv2.imwrite(str(frame_path), frame)
                saved += 1
            except Exception as exc:
                LOGGER.warning("Failed to write %s: %s", frame_path, exc)

    if saved:
        LOGGER.info("Saved %d frame(s) for %s to %s", saved, video_path.name, output_dir)
    else:
        LOGGER.warning("No frames extracted for %s (check source video)", video_path.name)
    return saved


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    LOGGER.setLevel(level)


def resolve_input_dir(
    args_input: Optional[str], cfg: dict, output_root_arg: Optional[str], env_output_root: Optional[str]
) -> Path:
    """
    Resolution order (highest precedence -> lowest):
    1. CLI --input-dir
    2. config['input_dir'] from --config
    3. CLI --output-root -> <output_root>/Scoliosis/input
    4. env SCOL_OUTPUT_ROOT -> <env>/Scoliosis/input
    5. fallback repo-root Scoliosis/input (existing behavior)
    """
    if args_input:
        LOGGER.debug("Using input-dir from CLI: %s", args_input)
        return Path(args_input).expanduser()
    if cfg.get("input_dir"):
        LOGGER.debug("Using input_dir from config: %s", cfg.get("input_dir"))
        return Path(cfg.get("input_dir")).expanduser()
    if output_root_arg:
        candidate = Path(output_root_arg).expanduser() / "Scoliosis" / "input"
        LOGGER.debug("Using input-dir derived from --output-root: %s", candidate)
        return candidate
    if env_output_root:
        candidate = Path(env_output_root).expanduser() / "Scoliosis" / "input"
        LOGGER.debug("Using input-dir derived from SCOL_OUTPUT_ROOT env: %s", candidate)
        return candidate
    # final fallback (old behavior)
    fallback = default_input_dir()
    LOGGER.debug("Falling back to repo-root default input-dir: %s", fallback)
    return fallback


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless video to frame extractor.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--input-dir", help="Input directory containing videos.")
    parser.add_argument("--output-dir", help="Directory to place extracted frames.")
    parser.add_argument("--frames", type=int, help="Frames to sample per video.")
    parser.add_argument("--videos", nargs="*", help="Optional list of video filenames to process.")
    parser.add_argument("--source-dir", help="Optional fallback directory containing source videos.")
    parser.add_argument("--output-root", help="If provided, use <output-root>/Scoliosis/input as the input dir.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    cfg = load_config(args.config)

    # Prefer resolution order implemented in resolve_input_dir
    env_output_root = os.environ.get("SCOL_OUTPUT_ROOT")
    input_dir = resolve_input_dir(args.input_dir, cfg, args.output_root, env_output_root)
    output_dir = Path(args.output_dir or cfg.get("output_dir", default_output_frames_dir())).expanduser()
    num_frames = int(args.frames or cfg.get("frames", DEFAULT_FRAMES))
    whitelist = args.videos or cfg.get("videos")
    source_dir_val = args.source_dir or cfg.get("source_dir")
    source_dir = Path(source_dir_val).expanduser() if source_dir_val else None

    LOGGER.info("Final resolved input_dir: %s", input_dir)
    LOGGER.info("Final resolved output_dir: %s", output_dir)

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

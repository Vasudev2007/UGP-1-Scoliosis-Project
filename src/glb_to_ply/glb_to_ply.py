#!/usr/bin/env python3
"""
Headless GLB → PLY conversion utility.

Defaults:
  input_dir  : <repo>/Scoliosis/working/Common/vggt_reconstruction
  output_dir : <repo>/Scoliosis/working/Common/vggt_reconstruction_ply
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pymeshlab

LOGGER = logging.getLogger("glb_to_ply")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_dir() -> Path:
    return repo_root() / "Scoliosis" / "working" / "Common" / "vggt_reconstruction"


def default_output_dir() -> Path:
    return repo_root() / "Scoliosis" / "working" / "Common" / "vggt_reconstruction_ply"


def load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def list_glb_files(input_dir: Path, whitelist: Optional[Iterable[str]]) -> List[Path]:
    if whitelist:
        return [input_dir / name for name in whitelist if (input_dir / name).exists()]
    return sorted(p for p in input_dir.glob("*.glb"))


def convert_glb(input_path: Path, output_path: Path, binary: bool) -> None:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(
        str(output_path),
        binary=binary,
        save_vertex_color=True,
        save_face_color=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GLB files to PLY.")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--input-dir", dest="input_dir", help="Directory containing GLB files.")
    parser.add_argument("--output-dir", help="Directory to write PLY files.")
    parser.add_argument("--binary", action="store_true", help="Export binary PLY files.")
    parser.add_argument("--ascii", action="store_true", help="Export ASCII PLY files.")
    parser.add_argument("--files", nargs="*", help="Optional list of GLB filenames to convert.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    cfg = load_config(args.config)
    input_dir = Path(args.input_dir or cfg.get("input_dir", default_input_dir())).expanduser()
    output_dir = Path(args.output_dir or cfg.get("output_dir", default_output_dir())).expanduser()
    whitelist = args.files or cfg.get("files")

    binary = cfg.get("binary", True)
    if args.binary:
        binary = True
    if args.ascii:
        binary = False

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = list_glb_files(input_dir, whitelist)
    if not glb_files:
        LOGGER.warning("No GLB files found in %s", input_dir)
        return 0

    LOGGER.info(
        "Converting %d GLB file(s) from %s → %s (binary=%s)",
        len(glb_files),
        input_dir,
        output_dir,
        binary,
    )
    for glb_path in glb_files:
        output_path = output_dir / f"{glb_path.stem}.ply"
        LOGGER.info("Converting %s → %s", glb_path.name, output_path.name)
        convert_glb(glb_path, output_path, binary)
    LOGGER.info("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

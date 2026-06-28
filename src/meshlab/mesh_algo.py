#!/usr/bin/env python3
"""
Headless point-cloud to mesh reconstruction using PyMeshLab.

Processing steps (matching GUI defaults):
  1. Compute normals        (k = 500, smooth_iter = 5)
  2. Screened Poisson       (depth = 8)
  3. Laplacian smoothing    (iterations = 2)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pymeshlab

LOGGER = logging.getLogger("mesh_reconstruction")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # avoid reconfiguring if already configured by caller
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        LOGGER.setLevel(level)


def load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless mesh reconstruction pipeline.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--input", dest="input_path", help="Input point cloud (PLY).")
    parser.add_argument("--output", dest="output_path", help="Output mesh file (PLY).")
    parser.add_argument("--normal-k", type=int, help="Normal computation neighbour count (default 500).")
    parser.add_argument("--normal-smooth", type=int, help="Normal smoothing iterations (default 5).")
    parser.add_argument("--poisson-depth", type=int, help="Screened Poisson octree depth (default 8).")
    parser.add_argument("--smooth-iter", type=int, help="Laplacian smoothing iterations (default 2).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    try:
        cfg = load_config(args.config)
    except Exception as e:
        LOGGER.error("Failed to load config: %s", e)
        return 2

    params = cfg.get("parameters")
    if isinstance(params, dict):
        params_input = params.get("input")
        params_output = params.get("output")
    else:
        params_input = params_output = None

    input_value = args.input_path or cfg.get("input") or params_input
    output_value = args.output_path or cfg.get("output") or params_output

    if not input_value or not output_value:
        LOGGER.error("Both input and output paths must be provided via arguments or config.")
        return 3

    input_path = Path(input_value).expanduser()
    output_path = Path(output_value).expanduser()

    normal_k = int(args.normal_k or cfg.get("normal_k", 500))
    normal_smooth = int(args.normal_smooth or cfg.get("normal_smooth", 5))
    poisson_depth = int(args.poisson_depth or cfg.get("poisson_depth", 8))
    smooth_iter = int(args.smooth_iter or cfg.get("smooth_iter", 2))

    # Basic sanity checks
    if normal_k <= 0:
        LOGGER.error("normal-k must be > 0")
        return 4
    if poisson_depth < 1 or poisson_depth > 14:
        LOGGER.warning("poisson-depth %d is unusual; large values can consume lots of memory.", poisson_depth)

    if not input_path.exists():
        LOGGER.error("Input point cloud not found: %s", input_path)
        return 5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading point cloud: %s", input_path)
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(str(input_path))
    except Exception as e:
        LOGGER.exception("Failed to load mesh (%s): %s", input_path, e)
        return 6

    mesh = ms.current_mesh()
    if mesh is None:
        LOGGER.error("No mesh loaded from %s (ms.current_mesh() is None)", input_path)
        return 7

    LOGGER.info("Loaded %d points", mesh.vertex_number())

    # Defensive API checks (helpful if different pymeshlab versions are installed)
    if not hasattr(ms, "compute_normal_for_point_clouds"):
        # try alternative name commonly used in examples
        if hasattr(ms, "compute_normals_for_point_clouds"):
            compute_normals_fn = ms.compute_normals_for_point_clouds
        else:
            LOGGER.error("PyMeshLab API does not expose a normals computation function on MeshSet.")
            return 8
    else:
        compute_normals_fn = ms.compute_normal_for_point_clouds

    LOGGER.info("Computing normals (k=%d, smooth=%d)", normal_k, normal_smooth)
    try:
        # view position can be a tuple; using numpy array is acceptable too
        compute_normals_fn(
            k=normal_k,
            smoothiter=normal_smooth,
            flipflag=False,
            viewpos=(0.0, 0.0, 0.0),
        )
    except TypeError:
        # Some pymeshlab versions expect different kwarg names; try a fallback
        try:
            compute_normals_fn(normal_k, normal_smooth, False, (0.0, 0.0, 0.0))
        except Exception as e:
            LOGGER.exception("Failed to compute normals: %s", e)
            return 9
    except Exception as e:
        LOGGER.exception("Failed to compute normals: %s", e)
        return 10

    LOGGER.info("Running screened Poisson reconstruction (depth=%d)", poisson_depth)
    poisson_fn = None
    if hasattr(ms, "generate_surface_reconstruction_screened_poisson"):
        poisson_fn = ms.generate_surface_reconstruction_screened_poisson
    elif hasattr(ms, "apply_filter"):
        # Some versions use apply_filter with a filter name dict; we won't attempt complex reflection here.
        poisson_fn = None

    if poisson_fn is None:
        LOGGER.error("Poisson reconstruction function not found in PyMeshLab MeshSet API.")
        return 11

    try:
        poisson_fn(
            depth=poisson_depth,
            fulldepth=5,
            cgdepth=0,
            scale=1.1,
            samplespernode=1.5,
            pointweight=4.0,
            iters=8,
            confidence=False,
            preclean=False,
        )
    except TypeError:
        # fallback in case different parameter names/order expected
        try:
            poisson_fn(poisson_depth)
        except Exception as e:
            LOGGER.exception("Poisson reconstruction failed: %s", e)
            return 12
    except Exception as e:
        LOGGER.exception("Poisson reconstruction failed: %s", e)
        return 13

    LOGGER.info("Applying Laplacian smoothing (iterations=%d)", smooth_iter)
    try:
        ms.apply_coord_laplacian_smoothing(
            stepsmoothnum=smooth_iter,
            cotangentweight=False,
            selected=False,
        )
    except Exception as e:
        LOGGER.exception("Laplacian smoothing failed: %s", e)
        return 14

    LOGGER.info("Saving mesh to %s", output_path)
    try:
        ms.save_current_mesh(str(output_path))
    except Exception as e:
        LOGGER.exception("Failed to save mesh to %s: %s", output_path, e)
        return 15

    mesh = ms.current_mesh()
    if mesh is None:
        LOGGER.warning("After processing, current_mesh() is None.")
    else:
        LOGGER.info(
            "Mesh reconstruction complete: %d vertices, %d faces",
            mesh.vertex_number(),
            mesh.face_number() if hasattr(mesh, "face_number") else -1,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

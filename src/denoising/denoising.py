#!/usr/bin/env python3
"""
Headless point-cloud denoising using Open3D.

Defaults:
  input_path  : <repo>/Scoliosis/working/Common/vggt_reconstruction_ply/p_env_far.ply
  output_path : <repo>/Scoliosis/working/far/denoising/p_env_far_denoised.ply

The filter chain matches the original GUI defaults:
  - Statistical outlier removal (k=30, std=2.0)
  - Radius outlier removal (nb_points=16, radius=0.05)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d

LOGGER = logging.getLogger("denoising")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_input_path() -> Path:
    return repo_root() / "Scoliosis" / "working" / "Common" / "vggt_reconstruction_ply" / "p_env_far.ply"


def default_output_path() -> Path:
    return repo_root() / "Scoliosis" / "working" / "far" / "denoising" / "p_env_far_denoised.ply"


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


def apply_statistical(pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    _, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(indices)


def apply_radius(pcd: o3d.geometry.PointCloud, nb_points: int, radius: float) -> o3d.geometry.PointCloud:
    _, indices = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(indices)


def compute_nn_stats(
    pcd: o3d.geometry.PointCloud,
    k: int = 6,
    sample_size: int = 20000,
) -> Dict[str, float]:
    n = len(pcd.points)
    if n == 0:
        return {"median_nn": 0.0, "mean_nn": 0.0}
    tree = o3d.geometry.KDTreeFlann(pcd)
    sample_count = min(n, sample_size)
    indices = np.linspace(0, n - 1, num=sample_count, dtype=int)
    collected = []
    for idx in indices:
        count, _, squared = tree.search_knn_vector_3d(pcd.points[idx], k + 1)
        if count < k + 1:
            continue
        d = np.sqrt(np.asarray(squared[1:]))  # skip the point itself
        collected.append(d)
    if not collected:
        return {"median_nn": 0.0, "mean_nn": 0.0}
    stacked = np.vstack(collected)
    return {
        "median_nn": float(np.median(stacked)),
        "mean_nn": float(np.mean(stacked)),
    }


def autosuggest_parameters(pcd: o3d.geometry.PointCloud) -> Dict[str, float]:
    pts = np.asarray(pcd.points)
    n = pts.shape[0]
    if n == 0:
        return {"sor_k": 30, "sor_std": 2.0, "ror_k": 16, "ror_radius": 0.05, "median_nn": 0.0}
    stats = compute_nn_stats(pcd, k=6)
    median_nn = max(1e-5, stats["median_nn"])
    sor_k = int(max(8, min(70, int(np.sqrt(n) / 2))))
    ror_k = int(max(6, min(40, int(np.sqrt(n) / 10))))
    ror_radius = float(max(1e-5, median_nn * 2.0))
    return {
        "sor_k": sor_k,
        "sor_std": 2.0,
        "ror_k": ror_k,
        "ror_radius": ror_radius,
        "median_nn": median_nn,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Headless point-cloud denoising.")
    parser.add_argument("--config", help="Optional JSON config.")
    parser.add_argument("--input", dest="input_path", help="Input PLY path.")
    parser.add_argument("--output", dest="output_path", help="Output PLY path.")
    parser.add_argument("--sor-k", type=int, help="SOR: neighbour count (default 30).")
    parser.add_argument("--sor-std", type=float, help="SOR: std ratio (default 2.0).")
    parser.add_argument("--ror-k", type=int, help="ROR: neighbour count (default 16).")
    parser.add_argument("--ror-radius", type=float, help="ROR: radius (default 0.05).")
    parser.add_argument("--disable-sor", action="store_true", help="Skip statistical outlier removal.")
    parser.add_argument("--disable-ror", action="store_true", help="Skip radius outlier removal.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    cfg = load_config(args.config)
    input_path = Path(args.input_path or cfg.get("input", default_input_path())).expanduser()
    output_path = Path(args.output_path or cfg.get("output", default_output_path())).expanduser()

    use_sor = not (cfg.get("disable_sor", False) or args.disable_sor)
    use_ror = not (cfg.get("disable_ror", False) or args.disable_ror)

    cfg_sor_k = cfg.get("sor_k")
    cfg_sor_std = cfg.get("sor_std")
    cfg_ror_k = cfg.get("ror_k")
    cfg_ror_radius = cfg.get("ror_radius")

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading point cloud from %s", input_path)
    pcd = o3d.io.read_point_cloud(str(input_path))
    LOGGER.info("Initial points: %d", len(pcd.points))

    suggestions = autosuggest_parameters(pcd)
    LOGGER.info(
        "Auto-suggested parameters — SOR k=%d, std=%.2f; ROR k=%d, radius=%.5f (median_nn=%.5f)",
        suggestions["sor_k"],
        suggestions["sor_std"],
        suggestions["ror_k"],
        suggestions["ror_radius"],
        suggestions["median_nn"],
    )

    if args.sor_k is not None:
        sor_k = int(args.sor_k)
    elif cfg_sor_k is not None:
        sor_k = int(cfg_sor_k)
    else:
        sor_k = suggestions["sor_k"]

    if args.sor_std is not None:
        sor_std = float(args.sor_std)
    elif cfg_sor_std is not None:
        sor_std = float(cfg_sor_std)
    else:
        sor_std = suggestions["sor_std"]

    if args.ror_k is not None:
        ror_k = int(args.ror_k)
    elif cfg_ror_k is not None:
        ror_k = int(cfg_ror_k)
    else:
        ror_k = suggestions["ror_k"]

    if args.ror_radius is not None:
        ror_radius = float(args.ror_radius)
    elif cfg_ror_radius is not None:
        ror_radius = float(cfg_ror_radius)
    else:
        ror_radius = suggestions["ror_radius"]

    processed = pcd
    if use_sor:
        LOGGER.info("Applying statistical outlier removal (k=%d, std=%.2f)", sor_k, sor_std)
        processed = apply_statistical(processed, sor_k, sor_std)
        LOGGER.info("After SOR: %d points", len(processed.points))
    if use_ror:
        LOGGER.info("Applying radius outlier removal (k=%d, radius=%.4f)", ror_k, ror_radius)
        processed = apply_radius(processed, ror_k, ror_radius)
        LOGGER.info("After ROR: %d points", len(processed.points))

    LOGGER.info("Saving cleaned point cloud to %s", output_path)
    o3d.io.write_point_cloud(str(output_path), processed)
    LOGGER.info("Denoising complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

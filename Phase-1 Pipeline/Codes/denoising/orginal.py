#!/usr/bin/env python3
"""
Headless point-cloud denoising using Open3D.

Defaults:
  input_path  : <repo>/Scoliosis/working/Common/vggt_reconstruction_ply/p_env_far.ply
  output_path : <repo>/Scoliosis/working/far/denoising/p_env_far_denoised.ply

The filter chain matches the original GUI defaults:
  - Statistical outlier removal (k=30, std=2.0)
  - Radius outlier removal (nb_points=16, radius=0.05)
  - DBSCAN clustering (eps=0.05, min_points=10, keep largest cluster)
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


def apply_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
    keep_largest: bool,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False))
    if labels.size == 0:
        return pcd
    mask = labels >= 0
    if not np.any(mask):
        return o3d.geometry.PointCloud() if keep_largest else pcd
    if keep_largest:
        unique, counts = np.unique(labels[mask], return_counts=True)
        largest = unique[np.argmax(counts)]
        keep_idx = np.where(labels == largest)[0]
    else:
        keep_idx = np.where(mask)[0]
    return pcd.select_by_index(keep_idx)


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
    parser.add_argument("--dbscan", dest="enable_dbscan", action="store_true", help="Enable DBSCAN cluster filtering.")
    parser.add_argument("--no-dbscan", dest="disable_dbscan", action="store_true", help="Disable DBSCAN cluster filtering.")
    parser.add_argument("--db-eps", type=float, help="DBSCAN: epsilon radius (default 0.05).")
    parser.add_argument("--db-min-points", type=int, help="DBSCAN: minimum points per cluster (default 10).")
    parser.add_argument("--db-keep-largest", dest="db_keep_largest", action="store_true", help="Keep only largest DBSCAN cluster (default).")
    parser.add_argument("--db-keep-all", dest="db_keep_all", action="store_true", help="Keep all DBSCAN clusters (no filtering).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    cfg = load_config(args.config)
    input_path = Path(args.input_path or cfg.get("input", default_input_path())).expanduser()
    output_path = Path(args.output_path or cfg.get("output", default_output_path())).expanduser()

    use_sor = not (cfg.get("disable_sor", False) or args.disable_sor)
    use_ror = not (cfg.get("disable_ror", False) or args.disable_ror)

    cfg_dbscan = cfg.get("use_dbscan")
    if args.enable_dbscan:
        use_dbscan = True
    elif args.disable_dbscan:
        use_dbscan = False
    elif cfg_dbscan is not None:
        use_dbscan = bool(cfg_dbscan)
    else:
        use_dbscan = True

    sor_k = int(args.sor_k or cfg.get("sor_k", 30))
    sor_std = float(args.sor_std or cfg.get("sor_std", 2.0))
    ror_k = int(args.ror_k or cfg.get("ror_k", 16))
    ror_radius = float(args.ror_radius or cfg.get("ror_radius", 0.05))

    db_eps = float(args.db_eps or cfg.get("db_eps", 0.05))
    db_min = int(args.db_min_points or cfg.get("db_min_points", 10))

    if args.db_keep_largest:
        keep_largest = True
    elif args.db_keep_all:
        keep_largest = False
    else:
        keep_largest = bool(cfg.get("dr › Ground plane estimation (manual)
☐10. Near › Alignment (manual)b_keep_largest", True))

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading point cloud from %s", input_path)
    pcd = o3d.io.read_point_cloud(str(input_path))
    LOGGER.info("Initial points: %d", len(pcd.points))

    processed = pcd
    if use_sor:
        LOGGER.info("Applying statistical outlier removal (k=%d, std=%.2f)", sor_k, sor_std)
        processed = apply_statistical(processed, sor_k, sor_std)
        LOGGER.info("After SOR: %d points", len(processed.points))
    if use_ror:
        LOGGER.info("Applying radius outlier removal (k=%d, radius=%.4f)", ror_k, ror_radius)
        processed = apply_radius(processed, ror_k, ror_radius)
        LOGGER.info("After ROR: %d points", len(processed.points))
    if use_dbscan:
        LOGGER.info(
            "Applying DBSCAN filtering (eps=%.4f, min_points=%d, keep_largest=%s)",
            db_eps,
            db_min,
            keep_largest,
        )
        processed = apply_dbscan(processed, db_eps, db_min, keep_largest)
        LOGGER.info("After DBSCAN: %d points", len(processed.points))

    LOGGER.info("Saving cleaned point cloud to %s", output_path)
    o3d.io.write_point_cloud(str(output_path), processed)
    LOGGER.info("Denoising complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

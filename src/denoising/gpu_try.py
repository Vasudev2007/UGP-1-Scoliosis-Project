#!/usr/bin/env python3
"""
Headless point-cloud denoising using Open3D with FAISS GPU acceleration (preferred).

Defaults:
  input_path  : <repo>/Scoliosis/working/Common/vggt_reconstruction_ply/p_env_far.ply
  output_path : <repo>/Scoliosis/working/far/denoising/p_env_far_denoised.ply

Filter chain (preserved semantics):
  - Statistical outlier removal (SOR):
      nb_neighbors default = 30
      std_ratio default     = 2.0
      Behavior: per-point mean distance to k neighbors (skip self), global mean/std,
                keep points with mean <= mean + std_ratio * std_all
  - Radius outlier removal (ROR):
      nb_points default = 16
      radius default    = 0.05
      Behavior: keep points that have at least nb_points neighbors within radius.
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


# --- Optional FAISS import (GPU) ---
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


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


#
# FAISS helper: build IndexFlatL2 and move to GPU if possible.
#
def _make_faiss_gpu_index(points_np: np.ndarray):
    """
    Create an IndexFlatL2 (exact L2) index for points_np (shape: (N,3)).
    Attempts to move the CPU index to GPU (device 0). Returns a tuple:
    (index_for_search, cpu_index, gpu_resources_or_None)
    where index_for_search is either GPU index or cpu_index (if GPU move fails).
    """
    # points_np must be float32
    pts32 = points_np.astype("float32")
    d = pts32.shape[1]
    cpu_index = faiss.IndexFlatL2(d)  # exact search (brute-force)
    cpu_index.add(pts32)
    try:
        # Try to allocate GPU resources and move the index to GPU device 0
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index, cpu_index, res
    except Exception:
        # If GPU move fails (no GPU, mismatch, etc.), return CPU index and None
        return cpu_index, cpu_index, None


def apply_statistical(pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    """
    Statistical Outlier Removal (SOR) using FAISS (GPU) when available.
    Semantics:
    - query k = nb_neighbors + 1 (to include self)
    - compute mean distance to neighbours excluding self
      - compute global mean and std, threshold = mean + std_ratio * std
    - keep points with mean <= threshold
    Falls back to Open3D's remove_statistical_outlier on failure or if FAISS unavailable.
    """
    points_np = np.asarray(pcd.points)
    n_points = points_np.shape[0]
    LOGGER.debug("apply_statistical (FAISS if available): n_points=%d, nb_neighbors=%d, std_ratio=%.3f", n_points, nb_neighbors, std_ratio)

    if FAISS_AVAILABLE:
        try:
            idx, _, res = _make_faiss_gpu_index(points_np)
            k = min(nb_neighbors + 1, n_points)  # +1 for self
            # D: squared distances (N,k), I: indices (N,k)
            D, I = idx.search(points_np.astype("float32"), k)

            # ignore self-distance (first column) if k > 1
            if k > 1:
                knn_sq = D[:, 1:k]  # shape (N, k-1)
            else:
                knn_sq = D[:, :1]

            # Guard negative numerical noise and sqrt to Euclidean distances
            knn_sq = np.maximum(knn_sq, 0.0)
            knn_dists = np.sqrt(knn_sq)  # (N, k-1)

            mean_d = knn_dists.mean(axis=1)  # per-point mean distance
            mean_all = float(mean_d.mean())
            std_all = float(mean_d.std(ddof=0))  # population std
            thresh = mean_all + float(std_ratio) * std_all

            keep_mask = mean_d <= thresh
            keep_idx = np.nonzero(keep_mask)[0].tolist()
            LOGGER.debug("SOR FAISS result: mean_all=%.6f std_all=%.6f thresh=%.6f kept=%d", mean_all, std_all, thresh, len(keep_idx))
            return pcd.select_by_index(keep_idx)
        except Exception as e:
            LOGGER.warning("FAISS-based SOR failed; falling back to Open3D CPU SOR. Error: %s", e)

    # Fallback: Open3D CPU SOR (original behavior)
    LOGGER.debug("SOR falling back to Open3D CPU implementation")
    _, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(indices)


def apply_radius(pcd: o3d.geometry.PointCloud, nb_points: int, radius: float) -> o3d.geometry.PointCloud:
    """
    Radius Outlier Removal (ROR) using FAISS (GPU) when available.
    Semantics:
    - use faiss.range_search with radius^2 (FAISS uses squared L2 distances)
    - count neighbors for each query (includes self if returned by FAISS)
    - keep points with count >= nb_points
    Falls back to Open3D's remove_radius_outlier on failure or if FAISS unavailable.
    """
    points_np = np.asarray(pcd.points)
    n_points = points_np.shape[0]
    LOGGER.debug("apply_radius (FAISS if available): n_points=%d, nb_points=%d, radius=%.6f", n_points, nb_points, radius)

    if FAISS_AVAILABLE:
        try:
            idx, cpu_idx, res = _make_faiss_gpu_index(points_np)
            radius_sq = float(radius) ** 2

            # range_search returns (lims, D, I)
            lims, D, I = idx.range_search(points_np.astype("float32"), radius_sq)
            # lims is length n_points+1; neighbors for i are in I[lims[i]:lims[i+1]]
            keep_indices = []
            for i in range(n_points):
                start = lims[i]
                end = lims[i + 1]
                count = int(end - start)  # includes self if present
                if count >= nb_points:
                    keep_indices.append(i)

            LOGGER.debug("ROR FAISS result: kept=%d out of %d", len(keep_indices), n_points)
            return pcd.select_by_index(keep_indices)
        except Exception as e:
            LOGGER.warning("FAISS-based ROR failed; falling back to Open3D CPU ROR. Error: %s", e)

    # Fallback: Open3D CPU ROR (original behavior)
    LOGGER.debug("ROR falling back to Open3D CPU implementation")
    _, indices = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(indices)


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

    sor_k = int(args.sor_k or cfg.get("sor_k", 30))
    sor_std = float(args.sor_std or cfg.get("sor_std", 2.0))
    ror_k = int(args.ror_k or cfg.get("ror_k", 16))
    ror_radius = float(args.ror_radius or cfg.get("ror_radius", 0.05))

    if not input_path.exists():
        raise FileNotFoundError(f"Input PLY not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading point cloud from %s", input_path)
    pcd = o3d.io.read_point_cloud(str(input_path))
    LOGGER.info("Initial points: %d", len(pcd.points))

    # Inform about FAISS / GPU availability
    if FAISS_AVAILABLE:
        LOGGER.info("FAISS available. Will attempt FAISS GPU acceleration if environment supports it.")
    else:
        LOGGER.info("FAISS not available. Using Open3D CPU implementations for SOR/ROR.")

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
    import argparse  # re-import safe (already imported at top) to satisfy some linters
    raise SystemExit(main())

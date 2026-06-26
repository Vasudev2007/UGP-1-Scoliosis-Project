#!/usr/bin/env python3
"""
Manual Alignment Tool V2 - Automated inputs + default correspondences

Changes:
 - Removes tkinter-based file dialogs for selecting source/target.
 - Default source: <repo_root>/Scoliosis/working/near/mesh_reconstruction/p_env_close_mesh.ply
 - Default target: <repo_root>/Scoliosis/working/far/mesh_reconstruction/p_env_far_mesh.ply
 - Default number of correspondence points: 4 (no terminal prompt)
 - Still uses EnhancedPointPicker to interactively pick points on each mesh.
 - Accepts optional CLI overrides: --source, --target, --num-points, --config
"""

from __future__ import annotations

import argparse
import json
import os
import copy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# EnhancedPointPicker must be importable in PYTHONPATH
from enhanced_point_picker import EnhancedPointPicker


def load_json_config(cfg_path: str) -> dict:
    """Load control/config JSON and return dict (safe)."""
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


class ManualAlignmentV2:
    def __init__(
        self,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
        num_points: int = 4,
        output_path: Optional[str] = None,
        secondary_output: Optional[str] = None,
    ):
        self.source_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.target_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.source_points: List[np.ndarray] = []
        self.target_points: List[np.ndarray] = []
        self.transformation: Optional[np.ndarray] = None
        self.aligned_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.scale: float = 1.0
        self.rmse: float = 0.0
        self.errors: np.ndarray = np.zeros((0,))
        self.max_error: float = 0.0
        self.source_path = source_path
        self.target_path = target_path
        self.num_points = max(3, min(10, int(num_points)))  # clamp between 3 and 10
        self.output_path = output_path
        self.secondary_output = secondary_output

    # ----------------- IO / mesh loading ----------------- #
    def load_mesh(self, path: str, name: str = "mesh") -> Optional[o3d.geometry.TriangleMesh]:
        """Load mesh from disk with diagnostics."""
        try:
            p = Path(path)
            print(f"Loading {name}: {path}")
            if not p.exists():
                print(f"Error: {name} file not found: {path}")
                return None
            mesh = o3d.io.read_triangle_mesh(str(path))
            if mesh is None or len(mesh.vertices) == 0:
                # try reading as point cloud for helpful message
                try:
                    pc = o3d.io.read_point_cloud(str(path))
                    pts = np.asarray(pc.points)
                    print(f"  Read as point cloud: {len(pts)} points (no triangles).")
                    if len(pts) > 0:
                        print("  Hint: run mesh reconstruction to produce a triangle mesh.")
                except Exception:
                    pass
                print(f"Error: {name} has no vertices or couldn't be read as triangle mesh.")
                return None
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            print(f"✓ {name} loaded: vertices={len(mesh.vertices):,} triangles={len(mesh.triangles):,}")
            return mesh
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None

    # ----------------- Point picking ----------------- #
    def pick_correspondence_points(self, mesh: o3d.geometry.TriangleMesh, num_points: int, mesh_name: str) -> Optional[List[np.ndarray]]:
        """
        Launch EnhancedPointPicker to allow interactive selection of points on 'mesh'.
        Requires user to pick `num_points` points in the same order for source/target.
        """
        print("\n" + "=" * 70)
        print(f"SELECT {num_points} LANDMARKS ON {mesh_name}")
        print("=" * 70)
        print("Tips: pick anatomical landmarks in the same order on both meshes.")
        print("Controls: SHIFT+LEFT-CLICK to pick; DELETE to undo; Q/ESC to finish.\n")

        picker = EnhancedPointPicker(mesh, num_points_needed=num_points, mesh_name=mesh_name)
        pts = picker.pick_points_interactive()

        if pts is None or len(pts) < 3:
            print(f"❌ Failed to pick enough points on {mesh_name} (need ≥3).")
            return None

        print(f"✓ Picked {len(pts)} points on {mesh_name}.")
        return pts

    # ----------------- Transformation computation ----------------- #
    def compute_rigid_transformation(self, source_pts: List[np.ndarray], target_pts: List[np.ndarray], allow_scale: bool = True) -> Tuple[np.ndarray, float]:
        """Compute similarity transformation (scale + rotation + translation) using Umeyama-like method."""
        S = np.asarray(source_pts, dtype=float)
        T = np.asarray(target_pts, dtype=float)
        src_centroid = S.mean(axis=0)
        tgt_centroid = T.mean(axis=0)
        S_centered = S - src_centroid
        T_centered = T - tgt_centroid

        if allow_scale:
            src_rms = np.sqrt(np.mean(np.sum(S_centered**2, axis=1)))
            tgt_rms = np.sqrt(np.mean(np.sum(T_centered**2, axis=1)))
            scale = tgt_rms / src_rms if src_rms > 0 else 1.0
        else:
            scale = 1.0

        S_scaled = S_centered * scale
        H = S_scaled.T @ T_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # reflection fix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - scale * R @ src_centroid
        Tmat = np.eye(4)
        Tmat[:3, :3] = scale * R
        Tmat[:3, 3] = t
        return Tmat, float(scale)

    def compute_alignment_error(self, source_pts: List[np.ndarray], target_pts: List[np.ndarray], T: np.ndarray) -> Tuple[float, np.ndarray]:
        S = np.asarray(source_pts, dtype=float)
        Tgt = np.asarray(target_pts, dtype=float)
        S_h = np.hstack([S, np.ones((S.shape[0], 1))])
        S_trans = (T @ S_h.T).T[:, :3]
        dists = np.linalg.norm(S_trans - Tgt, axis=1)
        rmse = float(np.sqrt(np.mean(dists**2)))
        return rmse, dists

    # ----------------- Visualization helpers ----------------- #
    def visualize_correspondences(self):
        """Show source (blue) and target (orange) and green lines connecting correspondences."""
        print("\nVisualizing correspondences (before alignment)...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Correspondences Preview", width=1400, height=900)
        src_vis = copy.deepcopy(self.source_mesh)
        src_vis.paint_uniform_color([0.2, 0.6, 1.0])
        vis.add_geometry(src_vis)
        tgt_vis = copy.deepcopy(self.target_mesh)
        tgt_vis.paint_uniform_color([1.0, 0.6, 0.2])
        vis.add_geometry(tgt_vis)

        # lines
        line_points = []
        line_lines = []
        for i in range(len(self.source_points)):
            line_points.append(self.source_points[i])
            line_points.append(self.target_points[i])
            line_lines.append([i * 2, i * 2 + 1])
        if line_points:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(line_points)
            ls.lines = o3d.utility.Vector2iVector(line_lines)
            ls.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(ls)

        # spheres
        for i, (spt, tpt) in enumerate(zip(self.source_points, self.target_points)):
            sph_s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sph_s.compute_vertex_normals()
            sph_s.paint_uniform_color([0, 0, 1])
            sph_s.translate(spt)
            vis.add_geometry(sph_s)

            sph_t = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sph_t.compute_vertex_normals()
            sph_t.paint_uniform_color([1, 0, 0])
            sph_t.translate(tpt)
            vis.add_geometry(sph_t)

        vis.run()
        vis.destroy_window()

    def visualize_alignment(self):
        """Show aligned source (blue) and target (orange) with residual lines."""
        print("\nVisualizing alignment result...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Alignment Result", width=1400, height=900)
        aligned_vis = copy.deepcopy(self.aligned_mesh)
        aligned_vis.paint_uniform_color([0.2, 0.6, 1.0])
        vis.add_geometry(aligned_vis)
        tgt_vis = copy.deepcopy(self.target_mesh)
        tgt_vis.paint_uniform_color([1.0, 0.6, 0.2])
        vis.add_geometry(tgt_vis)

        # transformed source points
        S = np.asarray(self.source_points)
        S_h = np.hstack([S, np.ones((S.shape[0], 1))])
        transformed = (self.transformation @ S_h.T).T[:, :3]

        for i, (pt_al, tgt_pt) in enumerate(zip(transformed, self.target_points)):
            sph_s = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sph_s.compute_vertex_normals()
            sph_s.paint_uniform_color([0, 1, 0])
            sph_s.translate(pt_al)
            vis.add_geometry(sph_s)

            sph_t = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sph_t.compute_vertex_normals()
            sph_t.paint_uniform_color([1, 0, 0])
            sph_t.translate(tgt_pt)
            vis.add_geometry(sph_t)

            # error line
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([pt_al, tgt_pt])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([1, 1, 0])
            vis.add_geometry(line_set)

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)
        vis.run()
        vis.destroy_window()

    # ----------------- Save / output ----------------- #
    def save_results(self, output_prefix: str = "manual_aligned_v2"):
        if self.aligned_mesh is None or self.transformation is None:
            print("No alignment to save.")
            return False

        secondary_mesh_path = None
        secondary_json_path = None

        if self.output_path:
            mesh_path = Path(self.output_path).expanduser()
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            base = mesh_path.with_suffix("")
            json_path = base.with_name(base.name + "_transform.json")
            if self.secondary_output:
                secondary_mesh_path = Path(self.secondary_output).expanduser()
                secondary_mesh_path.parent.mkdir(parents=True, exist_ok=True)
                sec_base = secondary_mesh_path.with_suffix("")
                secondary_json_path = sec_base.with_name(sec_base.name + "_transform.json")
        else:
            mesh_path = Path(f"{output_prefix}.ply")
            json_path = Path(f"{output_prefix}_transform.json")

        o3d.io.write_triangle_mesh(str(mesh_path), self.aligned_mesh)
        print(f"✓ Saved aligned mesh: {mesh_path}")

        data = {
            "transformation_matrix": self.transformation.tolist(),
            "scale": float(self.scale),
            "source_points": [pt.tolist() for pt in self.source_points],
            "target_points": [pt.tolist() for pt in self.target_points],
            "num_correspondences": len(self.source_points),
            "rmse": float(self.rmse),
            "max_error": float(self.max_error),
            "individual_errors": self.errors.tolist(),
            "method": "manual_correspondence_v2_with_scale",
            "version": "v2",
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved transformation: {json_path}")

        if secondary_mesh_path:
            o3d.io.write_triangle_mesh(str(secondary_mesh_path), self.aligned_mesh)
            print(f"✓ Saved aligned mesh copy: {secondary_mesh_path}")
            if secondary_json_path:
                with open(secondary_json_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Saved transformation copy: {secondary_json_path}")

        if self.scale != 1.0:
            print("\nUniform sizing applied:")
            print(f" Scale factor: {self.scale:.6f}")
        return True

    # ----------------- Main workflow ----------------- #
    def run(self):
        print("\n=== MANUAL CORRESPONDENCE ALIGNMENT V2 ===")

        # Load source & target (paths already set by caller / defaults)
        if not self.source_path or not self.target_path:
            print("Source or target path not specified.")
            return

        self.source_mesh = self.load_mesh(self.source_path, "Source")
        if self.source_mesh is None:
            return

        self.target_mesh = self.load_mesh(self.target_path, "Target")
        if self.target_mesh is None:
            return

        # Number of points is fixed (default 4); still allowed override via CLI arg
        num_points = self.num_points
        print(f"\nUsing {num_points} correspondence points (fixed).")

        # Pick points on source
        src_pts = self.pick_correspondence_points(self.source_mesh, num_points, "SOURCE")
        if src_pts is None:
            return
        self.source_points = src_pts

        # Pick points on target
        tgt_pts = self.pick_correspondence_points(self.target_mesh, num_points, "TARGET")
        if tgt_pts is None:
            return
        self.target_points = tgt_pts

        # Safety: ensure same count
        if len(self.source_points) != len(self.target_points):
            print("Point count mismatch between source and target. Aborting.")
            return

        # Preview correspondences
        self.visualize_correspondences()

        # Compute transformation (allow uniform scaling)
        Tmat, scale = self.compute_rigid_transformation(self.source_points, self.target_points, allow_scale=True)
        self.transformation = Tmat
        self.scale = scale
        print("\nTransformation matrix:")
        print(self.transformation)
        if self.scale != 1.0:
            print(f"Scale applied: {self.scale:.6f}")

        # Compute errors
        self.rmse, self.errors = self.compute_alignment_error(self.source_points, self.target_points, self.transformation)
        self.max_error = float(self.errors.max()) if self.errors.size else 0.0
        print(f"\nAlignment RMSE: {self.rmse:.6f} m, Max error: {self.max_error:.6f} m")
        for i, e in enumerate(self.errors):
            print(f" Point {i+1}: {e:.6f} m")

        # Apply transform to source mesh
        self.aligned_mesh = copy.deepcopy(self.source_mesh)
        self.aligned_mesh.transform(self.transformation)

        # Visualize final alignment
        self.visualize_alignment()

        # Save results (default prefix; change if desired)
        self.save_results()

        print("\n=== COMPLETED SUCCESSFULLY ===")


# ----------------- CLI / defaults ----------------- #
def default_paths_from_repo_root() -> Tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    default_source = str(repo_root / "Scoliosis" / "working" / "near" / "mesh_reconstruction" / "p_env_close_mesh.ply")
    default_target = str(repo_root / "Scoliosis" / "working" / "far" / "mesh_reconstruction" / "p_env_far_mesh.ply")
    return default_source, default_target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Alignment V2 (automated inputs, default 4 points)")
    parser.add_argument("--source", help="Source mesh path (overrides default automated location)")
    parser.add_argument("--target", help="Target mesh path (overrides default automated location)")
    parser.add_argument("--num-points", type=int, default=4, help="Number of correspondence points (default=4)")
    parser.add_argument("--output", help="Aligned mesh output path (PLY).")
    parser.add_argument("--secondary-output", help="Optional secondary path to save aligned mesh (PLY).")
    parser.add_argument("--config", help="Optional launcher JSON (can contain input/output etc.)")
    args = parser.parse_args()

    src_default, tgt_default = default_paths_from_repo_root()
    source_path = args.source or src_default
    target_path = args.target or tgt_default
    output_path = args.output
    secondary_output = args.secondary_output

    # If config passed, prefer any input keys inside it
    if args.config:
        cfg = load_json_config(args.config)
        # support top-level or nested parameters.input/parameters.target
        if isinstance(cfg, dict):
            params = cfg.get("parameters", cfg)
            # common keys in pipeline JSON may be 'input' or 'mesh' etc; try several possibilities
            source_path = source_path  # keep current if not overridden
            target_path = target_path
            if isinstance(params, dict):
                source_path = (
                    params.get("source")
                    or params.get("input")
                    or params.get("mesh_source")
                    or params.get("mesh")
                    or source_path
                )
                target_path = (
                    params.get("target")
                    or params.get("reference")
                    or params.get("mesh_target")
                    or target_path
                )
                output_path = params.get("output", output_path)
                secondary_output = params.get("secondary_output", secondary_output)

    print("Source mesh ->", source_path)
    print("Target mesh ->", target_path)
    print("Correspondence points ->", args.num_points)
    if output_path:
        print("Aligned mesh output ->", output_path)
    if secondary_output:
        print("Aligned mesh secondary output ->", secondary_output)

    tool = ManualAlignmentV2(
        source_path=source_path,
        target_path=target_path,
        num_points=args.num_points,
        output_path=output_path,
        secondary_output=secondary_output,
    )
    tool.run()

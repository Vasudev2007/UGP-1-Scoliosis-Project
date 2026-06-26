#!/usr/bin/env python3
"""
Ground Plane Estimator V2 - Pipeline-aware dual-output support

Changes for Fix B:
 - Reads --input or --config (launcher JSON).
 - Accepts --output (primary) and --working-output (secondary working copy).
 - If neither working-output nor working_dir provided, saves working copy to:
       <repo_root>/Scoliosis/working/far/gnd_estimate/<mesh_basename>_ground_plane.json
 - GUI (picker) opens automatically; no terminal Enter required.
 - Falls back to tkinter file dialog only if nothing else found.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import open3d as o3d

# Lazy tkinter import (avoid errors if unavailable)
try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog  # type: ignore
except Exception:
    tk = None
    filedialog = None

from enhanced_point_picker import EnhancedPointPicker  # ensure this is importable

DEFAULT_MESH_NAME = "p_env_far_mesh.ply"


def find_default_mesh(repo_root: Path) -> Optional[Path]:
    mesh_dir = repo_root / "Scoliosis" / "working" / "far" / "mesh_reconstruction"
    exact = mesh_dir / DEFAULT_MESH_NAME
    if exact.exists():
        return exact
    if mesh_dir.is_dir():
        all_ply = list(mesh_dir.glob("*.ply"))
        if all_ply:
            all_ply.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return all_ply[0]
    cwd_candidate = Path.cwd() / DEFAULT_MESH_NAME
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def load_json_config_paths(cfg_path: str) -> Dict[str, str]:
    """
    Read launcher JSON and extract keys:
      - input
      - output
      - working_output
      - working_dir
    Accepts top-level keys or nested under 'parameters'.
    """
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    out: Dict[str, str] = {}
    if isinstance(cfg, dict):
        # top-level
        for k in ("input", "output", "working_output", "working_dir"):
            if k in cfg and cfg[k]:
                out[k] = cfg[k]
        # nested under parameters
        params = cfg.get("parameters")
        if isinstance(params, dict):
            for k in ("input", "output", "working_output", "working_dir"):
                if k in params and params[k]:
                    out.setdefault(k, params[k])
    return out


class GroundPlaneEstimatorV2:
    def __init__(self, mesh_path: Optional[str] = None):
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.mesh_path: Optional[str] = mesh_path
        self.picked_points: list = []
        self.plane_params = None

    def select_mesh_file_dialog(self) -> bool:
        if filedialog is None:
            print("Tkinter not available; cannot open file dialog.")
            return False
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Mesh File",
            filetypes=[("PLY files", "*.ply"), ("OBJ files", "*.obj"), ("STL files", "*.stl"), ("All files", "*.*")],
            initialdir=os.getcwd(),
        )
        root.destroy()
        if file_path:
            self.mesh_path = file_path
            return True
        return False

    def load_mesh(self) -> bool:
        if not self.mesh_path:
            print("No mesh file provided!")
            return False
        try:
            p = Path(self.mesh_path)
            print(f"\nTrying to load mesh: {self.mesh_path}")
            if not p.exists():
                print("Error: file does not exist:", self.mesh_path)
                return False
            print("File extension:", p.suffix, "Size (bytes):", p.stat().st_size)
            mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            if mesh is None or len(mesh.vertices) == 0:
                # Helpful hint: maybe it's a point cloud
                try:
                    pc = o3d.io.read_point_cloud(self.mesh_path)
                    pts = np.asarray(pc.points)
                    print("Read as point cloud: points =", len(pts))
                    if len(pts) > 0:
                        print("Note: file contains points but no triangles. Run mesh reconstruction to create a triangle mesh.")
                except Exception:
                    pass
                print("Error: Mesh has no vertices.")
                return False
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            self.mesh = mesh
            print(f"\n✓ Mesh loaded: {os.path.basename(self.mesh_path)}")
            print(f"  Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")
            return True
        except Exception as e:
            print("Error loading mesh:", e)
            return False

    def pick_ground_points(self) -> bool:
        if self.mesh is None:
            print("No mesh loaded!")
            return False

        print("\n" + "=" * 70)
        print("GROUND POINT SELECTION (GUI will open automatically)")
        print("=" * 70)

        picker = EnhancedPointPicker(self.mesh, num_points_needed=5, mesh_name="Ground/Floor")
        self.picked_points = picker.pick_points_interactive()

        if self.picked_points is None or len(self.picked_points) < 3:
            print(f"\n❌ Need ≥3 points, got {len(self.picked_points) if self.picked_points else 0}.")
            return False
        print(f"\n✓ {len(self.picked_points)} points selected.")
        return True

    def estimate_plane_ransac(self, distance_threshold=0.01):
        pts = np.array(self.picked_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
        return plane_model, inliers

    def estimate_plane_least_squares(self):
        pts = np.array(self.picked_points)
        centroid = np.mean(pts, axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        return [float(normal[0]), float(normal[1]), float(normal[2]), d]

    def estimate_ground_plane(self) -> bool:
        print("\nEstimating ground plane...")
        if len(self.picked_points) >= 3:
            self.plane_params, inliers = self.estimate_plane_ransac()
            print(f"  RANSAC inliers: {len(inliers)}/{len(self.picked_points)}")
        else:
            self.plane_params = self.estimate_plane_least_squares()
        a, b, c, d = self.plane_params
        print(f"✓ Plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        return True

    def create_plane_mesh(self, size: float = 2.5, offset: float = 0.002):
        if self.plane_params is None or not self.picked_points:
            return None
        a, b, c, d = self.plane_params
        normal = np.array([a, b, c], dtype=float)
        n_norm = np.linalg.norm(normal)
        if n_norm == 0:
            return None
        normal = normal / n_norm
        centroid = np.mean(np.array(self.picked_points), axis=0).astype(float)
        arbitrary = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(arbitrary, normal)) > 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        v1 = np.cross(normal, arbitrary)
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        corners = [
            centroid - size * v1 - size * v2,
            centroid + size * v1 - size * v2,
            centroid + size * v1 + size * v2,
            centroid - size * v1 + size * v2,
        ]
        corners = [c + normal * offset for c in corners]
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(corners))
        plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        plane_mesh.compute_vertex_normals()
        plane_mesh.paint_uniform_color([0.0, 0.8, 0.2])
        return plane_mesh

    def visualize_result(self, plane_size: float = 2.5):
        print("\nOpening visualization window...")
        vis = o3d.visualization.Visualizer()
        vis.create_window("Ground Plane Estimation", width=1400, height=900)
        try:
            self.mesh.paint_uniform_color([0.75, 0.75, 0.75])
        except Exception:
            pass
        vis.add_geometry(self.mesh)
        pts = np.array(self.picked_points) if self.picked_points else np.zeros((0, 3))
        if pts.size > 0:
            bbox = self.mesh.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound()))
            sphere_r = max(0.002, diag * 0.0025)
        else:
            sphere_r = 0.01
        for pt in self.picked_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([1.0, 0.1, 0.1])
            sphere.translate(pt)
            vis.add_geometry(sphere)
        plane_mesh = self.create_plane_mesh(size=plane_size, offset=max(0.0005, sphere_r * 0.3))
        if plane_mesh is not None:
            vis.add_geometry(plane_mesh)
        else:
            print("Warning: plane mesh not available for visualization.")
        try:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.05, plane_size * 0.2))
            vis.add_geometry(frame)
        except Exception:
            pass
        try:
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.95, 0.95, 0.95])
            opt.mesh_show_back_face = True
            opt.point_size = 5.0
        except Exception:
            pass
        vis.run()
        vis.destroy_window()

    def save_plane_parameters(self, output_path: Optional[str] = None) -> bool:
        if self.plane_params is None:
            print("No plane to save!")
            return False
        if output_path is None:
            base_name = Path(self.mesh_path).stem if self.mesh_path else "ground_plane"
            output_path = f"{base_name}_ground_plane.json"
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        a, b, c, d = self.plane_params
        data = {
            "mesh_file": self.mesh_path,
            "plane_equation": {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "d": float(d),
                "equation": f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0",
            },
            "normal_vector": [float(a), float(b), float(c)],
            "picked_points": [pt.tolist() for pt in self.picked_points],
            "num_points": len(self.picked_points),
            "version": "v2_enhanced_picker",
        }
        with open(outp, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Saved plane parameters to {outp}")
        return True

    def run(self):
        print("\n=== GROUND PLANE ESTIMATOR V2 ===")
        if not self.mesh_path:
            repo_root = Path(__file__).resolve().parent.parent
            found = find_default_mesh(repo_root)
            if found:
                self.mesh_path = str(found)
                print(f"Auto-selected mesh: {self.mesh_path}")
            else:
                print("No default mesh found. Opening file dialog...")
                if not self.select_mesh_file_dialog():
                    print("No file chosen. Exiting.")
                    return
        if not self.load_mesh():
            print("Failed to load mesh. Exiting.")
            return
        if not self.pick_ground_points():
            print("No points selected. Exiting.")
            return
        self.estimate_ground_plane()
        self.visualize_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Plane Estimator V2 (Enhanced Picker)")
    parser.add_argument("--input", "-i", help="Input mesh path")
    parser.add_argument("--output", "-o", help="Primary output JSON path for plane parameters")
    parser.add_argument("--working-output", "-w", help="Secondary working-copy JSON path")
    parser.add_argument("--config", "-c", help="Launcher config JSON containing input/output/working_output/working_dir")
    args = parser.parse_args()

    mesh_path_arg: Optional[str] = None
    output_path_arg: Optional[str] = None
    working_output_arg: Optional[str] = None

    if args.input:
        mesh_path_arg = args.input

    if args.config:
        cfg_paths = load_json_config_paths(args.config)
        mesh_path_arg = mesh_path_arg or cfg_paths.get("input")
        output_path_arg = args.output or cfg_paths.get("output")
        working_output_arg = args.working_output or cfg_paths.get("working_output") or cfg_paths.get("working_dir")
    else:
        output_path_arg = args.output
        working_output_arg = args.working_output

    estimator = GroundPlaneEstimatorV2(mesh_path=mesh_path_arg)
    estimator.run()

    # After run, save primary and working copy as requested (best-effort)
    if estimator.plane_params is not None:
        repo_root = Path(__file__).resolve().parent.parent
        # Default working dir if nothing provided
        default_working_dir = repo_root / "Scoliosis" / "working" / "far" / "gnd_estimate"
        default_working_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(estimator.mesh_path).stem if estimator.mesh_path else "ground_plane"
        default_working_path = default_working_dir / f"{base_name}_ground_plane.json"

        # Determine final paths
        primary_path = Path(output_path_arg) if output_path_arg else None
        # if working_output_arg is a directory, convert into a file path inside it
        working_path = None
        if working_output_arg:
            wp = Path(working_output_arg)
            # if path ends with separator or is existing dir, write file inside it
            if working_output_arg.endswith(os.sep) or wp.exists() and wp.is_dir():
                working_path = wp / f"{base_name}_ground_plane.json"
            else:
                working_path = wp
        else:
            working_path = default_working_path

        # Save primary first (if present)
        try:
            if primary_path:
                estimator.save_plane_parameters(output_path=str(primary_path))
            # Save working copy (don't duplicate if same path)
            try:
                if primary_path and primary_path.resolve() == working_path.resolve():
                    print("Primary and working paths are identical; saved once.")
                else:
                    estimator.save_plane_parameters(output_path=str(working_path))
            except Exception:
                # If path resolution failed (e.g., primary_path is None), just attempt working save
                estimator.save_plane_parameters(output_path=str(working_path))
        except Exception as e:
            print("Error while saving outputs:", e)

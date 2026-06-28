#!/usr/bin/env python3
"""
Final visualization helper for the scoliosis pipeline.

This tool loads the aligned patient mesh together with the previously
estimated ground plane parameters and displays them for review.  It can also
optionally re-fit the plane (interactive point picking) if ``--adjust`` is
passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import open3d as o3d


HEADER_DEFAULT = "Final Visualization for Doctor's Review"


def load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_param(cfg: Dict[str, object], *candidates: str) -> Optional[object]:
    for key in candidates:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return None


class FinalVisualization:
    def __init__(
        self,
        mesh_path: Optional[str] = None,
        plane_json: Optional[str] = None,
        output_json: Optional[str] = None,
        secondary_output_json: Optional[str] = None,
        allow_adjust: bool = False,
        header_text: str = HEADER_DEFAULT,
    ) -> None:
        self.mesh_path = Path(mesh_path).expanduser() if mesh_path else None
        self.plane_json_path = Path(plane_json).expanduser() if plane_json else None
        self.output_json_path = Path(output_json).expanduser() if output_json else None
        self.secondary_output_path = (
            Path(secondary_output_json).expanduser() if secondary_output_json else None
        )
        self.allow_adjust = allow_adjust
        self.header_text = header_text

        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.picked_points: List[np.ndarray] = []
        self.point_cloud: Optional[o3d.geometry.PointCloud] = None
        self.plane_params: Optional[Sequence[float]] = None

    # ------------------------------------------------------------------ utilities
    def load_mesh(self) -> bool:
        if not self.mesh_path:
            print("No mesh path provided.")
            return False
        if not self.mesh_path.exists():
            print(f"Mesh path not found: {self.mesh_path}")
            return False
        try:
            self.mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        except Exception as exc:
            print(f"Failed to read mesh {self.mesh_path}: {exc}")
            return False
        if self.mesh is None or not self.mesh.has_vertices():
            print(f"Mesh {self.mesh_path} contains no vertices.")
            return False
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        print(f"Loaded mesh: {self.mesh_path} ({len(self.mesh.vertices):,} vertices)")
        return True

    def load_plane_from_json(self) -> bool:
        if not self.plane_json_path:
            return False
        if not self.plane_json_path.exists():
            print(f"Plane JSON not found: {self.plane_json_path}")
            return False
        try:
            with self.plane_json_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:
            print(f"Failed to read plane JSON: {exc}")
            return False

        plane = data.get("plane_equation", {})
        a = plane.get("a")
        b = plane.get("b")
        c = plane.get("c")
        d = plane.get("d")
        if None in (a, b, c, d):
            print(f"Plane JSON missing coefficients a/b/c/d: {self.plane_json_path}")
            return False
        self.plane_params = [float(a), float(b), float(c), float(d)]
        picked = data.get("picked_points", [])
        if isinstance(picked, list):
            self.picked_points = [np.asarray(pt, dtype=float) for pt in picked if isinstance(pt, list)]
        print(f"Loaded plane parameters from {self.plane_json_path}")
        return True

    # ------------------------------------------------------------------ interactive helpers
    def sample_point_cloud(self) -> None:
        assert self.mesh is not None
        num_points = max(100000, len(self.mesh.vertices) * 3)
        self.point_cloud = self.mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(self.point_cloud.points)
        z = points[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        colors = np.zeros((len(points), 3))
        if z_max > z_min:
            zn = (z - z_min) / (z_max - z_min)
            colors[:, 0] = zn
            colors[:, 2] = 1 - zn
            colors[:, 1] = 0.3
        else:
            colors[:, :] = [0.5, 0.5, 0.8]
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)

    def pick_points_interactive(self, num_points: int = 4) -> bool:
        assert self.mesh is not None
        print("\n" + "=" * 70)
        print("SELECT LANDMARKS ON THE GROUND PLANE")
        print("=" * 70)
        print("Controls: SHIFT + LEFT CLICK to add points, DELETE to undo, Q to finish.")
        self.sample_point_cloud()
        vis_edit = o3d.visualization.VisualizerWithEditing()
        vis_edit.create_window(
            window_name="Select points (SHIFT + CLICK), press Q when done",
            width=1400,
            height=800,
        )
        vis_edit.add_geometry(self.point_cloud)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis_edit.add_geometry(coord)
        vis_edit.run()
        indices = vis_edit.get_picked_points()
        vis_edit.destroy_window()
        if not indices:
            print("No points selected.")
            return False
        points = np.asarray(self.point_cloud.points)
        self.picked_points = [points[idx] for idx in indices]
        print(f"Selected {len(self.picked_points)} points.")
        return len(self.picked_points) >= 3

    # ------------------------------------------------------------------ plane estimation
    def estimate_plane_from_points(self, points: Iterable[Sequence[float]]) -> Optional[List[float]]:
        pts = np.asarray(list(points), dtype=float)
        if pts.shape[0] < 3:
            return None
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        d = -float(normal @ centroid)
        plane = [float(normal[0]), float(normal[1]), float(normal[2]), d]
        print(f"Estimated plane: {plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0")
        return plane

    def ensure_plane(self) -> bool:
        if self.plane_params is not None:
            return True
        if self.allow_adjust:
            if not self.pick_points_interactive():
                return False
            self.plane_params = self.estimate_plane_from_points(self.picked_points)
            return self.plane_params is not None
        print("No plane parameters provided and adjustments disabled.")
        return False

    # ------------------------------------------------------------------ visualization
    def create_plane_mesh(self, plane: Sequence[float], scale: float = 1.5) -> o3d.geometry.TriangleMesh:
        assert self.mesh is not None
        a, b, c, d = plane
        normal = np.asarray([a, b, c], dtype=float)
        normal = normal / np.linalg.norm(normal)
        if self.picked_points:
            center = np.mean(np.asarray(self.picked_points), axis=0)
        else:
            center = self.mesh.get_axis_aligned_bounding_box().center
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            v1 = np.cross(normal, np.array([1, 0, 0]))
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        diag = float(self.mesh.get_axis_aligned_bounding_box().get_max_bound() -
                      self.mesh.get_axis_aligned_bounding_box().get_min_bound())
        size = max(0.5, diag * scale * 0.25)

        vertices = [
            center - size * v1 - size * v2,
            center + size * v1 - size * v2,
            center + size * v1 + size * v2,
            center - size * v1 + size * v2,
        ]
        triangles = [[0, 1, 2], [0, 2, 3]]
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        plane_mesh.paint_uniform_color([0.0, 0.8, 0.0])
        plane_mesh.compute_vertex_normals()
        return plane_mesh

    def visualize(self) -> None:
        assert self.mesh is not None and self.plane_params is not None
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.header_text, width=1400, height=800)
        mesh_copy = o3d.geometry.TriangleMesh(self.mesh)
        mesh_copy.paint_uniform_color([0.85, 0.85, 0.85])
        vis.add_geometry(mesh_copy)
        plane_mesh = self.create_plane_mesh(self.plane_params)
        vis.add_geometry(plane_mesh)
        if self.picked_points:
            for pt in self.picked_points:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                sphere.translate(pt)
                vis.add_geometry(sphere)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.96, 0.96, 0.98])
        print("Opening visualization window. Close the window when finished reviewing.")
        vis.run()
        vis.destroy_window()

    # ------------------------------------------------------------------ output
    def save_plane(self) -> None:
        if self.plane_params is None:
            return
        payload = {
            "mesh_file": str(self.mesh_path) if self.mesh_path else None,
            "plane_equation": {
                "a": float(self.plane_params[0]),
                "b": float(self.plane_params[1]),
                "c": float(self.plane_params[2]),
                "d": float(self.plane_params[3]),
                "equation": f"{self.plane_params[0]:.6f}x + {self.plane_params[1]:.6f}y + "
                            f"{self.plane_params[2]:.6f}z + {self.plane_params[3]:.6f} = 0",
            },
            "picked_points": [pt.tolist() for pt in self.picked_points],
            "num_points": len(self.picked_points),
            "allow_adjust": self.allow_adjust,
        }
        targets = []
        if self.output_json_path:
            targets.append(self.output_json_path)
        if self.secondary_output_path:
            targets.append(self.secondary_output_path)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
            print(f"Saved plane parameters to {target}")

    # ------------------------------------------------------------------ main entry
    def run(self) -> None:
        print("\n" + "=" * 70)
        print(self.header_text.upper())
        print("=" * 70)

        if not self.load_mesh():
            return
        plane_loaded = self.load_plane_from_json()
        if not plane_loaded and not self.allow_adjust:
            print("No plane JSON provided; enable --adjust to re-fit the plane manually.")
            return
        if not self.ensure_plane():
            print("Unable to obtain plane parameters.")
            return
        self.visualize()
        if self.allow_adjust and (self.output_json_path or self.secondary_output_path):
            self.save_plane()
        print("\nReview complete.")


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final visualization for doctor's review.")
    parser.add_argument("--mesh", help="Aligned mesh path (PLY).")
    parser.add_argument("--ground-json", help="Ground plane JSON path.")
    parser.add_argument("--output-json", help="Optional path to write adjusted plane JSON.")
    parser.add_argument("--secondary-output-json", help="Optional second JSON output path.")
    parser.add_argument("--config", help="Launcher config JSON.")
    parser.add_argument("--adjust", action="store_true", help="Enable manual adjustment (point picking).")
    parser.add_argument("--header", help="Custom header / window title.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_cli(argv)
    mesh_path = args.mesh
    ground_json = args.ground_json
    output_json = args.output_json
    secondary_output = args.secondary_output_json
    header_text = args.header or HEADER_DEFAULT

    if args.config:
        cfg = load_config(args.config)
        params = cfg.get("parameters", cfg) if isinstance(cfg, dict) else {}
        mesh_path = mesh_path or resolve_param(params, "mesh", "mesh_path", "input", "aligned_mesh")
        ground_json = ground_json or resolve_param(params, "ground_json", "plane_json", "ground")
        output_json = output_json or resolve_param(params, "output_json", "output")
        secondary_output = secondary_output or resolve_param(params, "secondary_output", "copy_output")
        header_text = (
            resolve_param(params, "header", "header_text") or header_text
        )

    vis = FinalVisualization(
        mesh_path=mesh_path,
        plane_json=ground_json,
        output_json=output_json,
        secondary_output_json=secondary_output,
        allow_adjust=args.adjust,
        header_text=header_text,
    )
    vis.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

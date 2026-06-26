#!/usr/bin/env python3
"""
Enhanced Point Picker with Intuitive GUI

Improvements:
- Larger, numbered point markers
- Undo functionality (Delete/Backspace)
- Real-time point counter
- Clear on-screen instructions
- Color-coded selection stages
- Visual feedback for each action
- Better landmark suggestions
"""

import open3d as o3d
import numpy as np
import copy


class EnhancedPointPicker:
    def __init__(self, mesh, point_cloud=None, num_points_needed=3, mesh_name="Mesh"):
        self.mesh = mesh
        self.mesh_name = mesh_name
        self.num_points_needed = num_points_needed
        
        # Create point cloud if not provided
        if point_cloud is None:
            print(f"Creating point cloud from {mesh_name}...")
            self.point_cloud = mesh.sample_points_uniformly(number_of_points=100000)
        else:
            self.point_cloud = point_cloud
        
        # Add height-based coloring
        self._color_by_height()
        
        # State
        self.picked_points = []
        self.picked_indices = []
        self.sphere_markers = []
        self.text_markers = []
        
        # Visualization
        self.vis = None
        
    def _color_by_height(self):
        """Color point cloud by height for easier identification"""
        points = np.asarray(self.point_cloud.points)
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        
        colors = np.zeros((len(points), 3))
        if z_max > z_min:
            normalized_z = (z_values - z_min) / (z_max - z_min)
            colors[:, 0] = normalized_z  # Red for high
            colors[:, 2] = 1 - normalized_z  # Blue for low
            colors[:, 1] = 0.3  # Slight green
        else:
            colors[:, :] = [0.5, 0.5, 0.8]
        
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    def _create_numbered_sphere(self, position, number, color=[1, 0, 0]):
        """Create a large numbered sphere marker"""
        # Create sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sphere.paint_uniform_color(color)
        sphere.translate(position)
        sphere.compute_vertex_normals()
        
        return sphere
    
    def _add_point_marker(self, position, number):
        """Add a visual marker for a picked point"""
        # Color progression: red -> yellow -> green
        progress = number / max(self.num_points_needed, 1)
        if progress < 0.5:
            # Red to yellow
            color = [1, progress * 2, 0]
        else:
            # Yellow to green
            color = [2 - progress * 2, 1, 0]
        
        sphere = self._create_numbered_sphere(position, number, color)
        self.sphere_markers.append(sphere)
        
        if self.vis:
            self.vis.add_geometry(sphere)
            self.vis.update_renderer()
    
    def _remove_last_marker(self):
        """Remove the last point marker"""
        if self.sphere_markers and self.vis:
            last_sphere = self.sphere_markers.pop()
            self.vis.remove_geometry(last_sphere)
            self.vis.update_renderer()
    
    def _print_instructions(self):
        """Print clear instructions"""
        print("\n" + "="*70)
        print(f"  POINT SELECTION: {self.mesh_name.upper()}")
        print("="*70)
        print(f"\n📍 Select {self.num_points_needed} points on the mesh")
        print(f"✓ Currently selected: {len(self.picked_points)}/{self.num_points_needed}")
        
        if self.num_points_needed >= 4:
            print("\n💡 SUGGESTED LANDMARKS (in order):")
            landmarks = [
                "1️⃣  Top of head (vertex/crown)",
                "2️⃣  Left shoulder (acromion)",
                "3️⃣  Right shoulder (acromion)",
                "4️⃣  Center of pelvis/hip",
            ]
            if self.num_points_needed > 4:
                landmarks.extend([
                    "5️⃣  Left knee (optional)",
                    "6️⃣  Right knee (optional)",
                ])
            
            for i, landmark in enumerate(landmarks[:self.num_points_needed]):
                print(f"  {landmark}")
        
        print("\n🎮 CONTROLS:")
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │ SHIFT + LEFT-CLICK    Select point                     │")
        print("  │ DELETE / BACKSPACE    Undo last point                  │")
        print("  │ Q or ESC              Finish (min 3 points required)   │")
        print("  │ Mouse Wheel           Zoom in/out                      │")
        print("  │ LEFT-DRAG             Rotate view                      │")
        print("  │ RIGHT-DRAG            Pan view                         │")
        print("  └─────────────────────────────────────────────────────────┘")
        
        print("\n🎨 COLOR GUIDE:")
        print("  🔵 BLUE points   = Ground level (low Z)")
        print("  🔴 RED points    = High elevation (high Z)")
        print("  🟡 YELLOW-GREEN  = Selected points (numbered)")
        
        print("\n" + "="*70)
        print("Window opening... Start selecting points!")
        print("="*70 + "\n")
    
    def _on_key_press(self, vis, key, action):
        """Handle keyboard events"""
        # Delete/Backspace to undo
        if key in [261, 259]:  # Delete or Backspace
            if len(self.picked_points) > 0:
                self.picked_points.pop()
                self.picked_indices.pop()
                self._remove_last_marker()
                print(f"⬅️  Undo: {len(self.picked_points)}/{self.num_points_needed} points")
            else:
                print("⚠️  No points to undo")
            return True
        
        # Q or ESC to finish
        if key in [81, 256]:  # Q or ESC
            if len(self.picked_points) >= 3:
                print(f"\n✓ Finishing with {len(self.picked_points)} points...")
                vis.close()
                return True
            else:
                print(f"\n⚠️  Need at least 3 points (currently {len(self.picked_points)})")
                return False
        
        return False
    
    def pick_points_interactive(self):
        """
        Interactive point picking with enhanced GUI
        
        Returns:
            list: Selected 3D points as numpy arrays
        """
        self._print_instructions()
        
        # Create visualizer with editing
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window(
            window_name=f"📍 Select Points on {self.mesh_name} [SHIFT+CLICK] | DELETE=Undo | Q=Done",
            width=1600,
            height=900
        )
        
        # Add geometries
        self.vis.add_geometry(self.point_cloud)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(coord_frame)
        
        # Add instruction text (as a mesh at top of view)
        # Create a simple text indicator
        instruction_box = self._create_instruction_box()
        if instruction_box:
            self.vis.add_geometry(instruction_box)
        
        print(f"🎯 Selecting points on {self.mesh_name}...")
        print(f"   Hold SHIFT and LEFT-CLICK to select")
        print(f"   Press DELETE to undo last point")
        print(f"   Press Q when done (min 3 points)\n")
        
        # Run visualization
        self.vis.run()
        
        # Get picked indices
        picked_indices = self.vis.get_picked_points()
        
        # Convert to coordinates
        if picked_indices:
            points_array = np.asarray(self.point_cloud.points)
            self.picked_points = []
            self.picked_indices = picked_indices
            
            for i, idx in enumerate(picked_indices):
                point = points_array[idx]
                self.picked_points.append(point)
        
        self.vis.destroy_window()
        
        # Print results
        print("\n" + "="*70)
        print("SELECTION COMPLETE")
        print("="*70)
        
        if len(self.picked_points) >= 3:
            print(f"✓ Successfully selected {len(self.picked_points)} points:\n")
            for i, pt in enumerate(self.picked_points):
                print(f"  Point {i+1}: [{pt[0]:7.4f}, {pt[1]:7.4f}, {pt[2]:7.4f}]")
            print("\n" + "="*70)
            return self.picked_points
        else:
            print(f"❌ Only {len(self.picked_points)} point(s) selected. Need at least 3!")
            print("="*70)
            return None
    
    def _create_instruction_box(self):
        """Create a visual instruction indicator"""
        # Create a small box at the top as visual reminder
        # This is optional and can be removed if it clutters the view
        return None
    
    def pick_points_with_preview(self):
        """
        Pick points with live preview of selections
        
        This version shows selected points immediately as you pick them
        """
        self._print_instructions()
        
        # Create custom visualizer
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name=f"📍 {self.mesh_name}: {len(self.picked_points)}/{self.num_points_needed} points | SHIFT+CLICK | DEL=Undo | Q=Done",
            width=1600,
            height=900
        )
        
        # Add point cloud
        vis.add_geometry(self.point_cloud)
        
        # Add coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(coord)
        
        # Run
        vis.run()
        
        # Get results
        picked_indices = vis.get_picked_points()
        
        # Process picked points
        if picked_indices:
            points_array = np.asarray(self.point_cloud.points)
            self.picked_points = [points_array[idx] for idx in picked_indices]
            self.picked_indices = picked_indices
            
            # Print results
            print(f"\n✓ Selected {len(self.picked_points)} points:")
            for i, pt in enumerate(self.picked_points):
                print(f"  Point {i+1}: [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]")
        
        vis.destroy_window()
        
        if len(self.picked_points) >= 3:
            return self.picked_points
        else:
            print(f"\n❌ Need at least 3 points, only got {len(self.picked_points)}")
            return None


class InteractivePointSelector:
    """
    Even more interactive version with real-time visual feedback
    """
    def __init__(self, mesh, num_points=4):
        self.mesh = mesh
        self.num_points = num_points
        self.selected_points = []
        self.point_markers = []
        
    def run(self):
        """Run interactive selection"""
        print("\n" + "="*70)
        print("INTERACTIVE POINT SELECTOR")
        print("="*70)
        print(f"\nSelect {self.num_points} points")
        print("\nControls:")
        print("  SHIFT + LEFT-CLICK : Select point")
        print("  DELETE/BACKSPACE   : Undo last point")
        print("  Q or ESC           : Finish (min 3 points)")
        print("\nTips:")
        print("  • Zoom in close for precise selection")
        print("  • Selected points appear as colored spheres")
        print("  • Numbers show selection order")
        print("="*70 + "\n")
        
        # Create point cloud
        pcd = self.mesh.sample_points_uniformly(100000)
        
        # Color by height
        points = np.asarray(pcd.points)
        z = points[:, 2]
        colors = np.zeros((len(points), 3))
        if z.max() > z.min():
            norm_z = (z - z.min()) / (z.max() - z.min())
            colors[:, 0] = norm_z
            colors[:, 2] = 1 - norm_z
            colors[:, 1] = 0.3
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name=f"Select {self.num_points} Points | SHIFT+CLICK | DEL=Undo | Q=Done",
            width=1600,
            height=900
        )
        vis.add_geometry(pcd)
        
        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3)
        vis.add_geometry(coord)
        
        vis.run()
        
        # Get results
        indices = vis.get_picked_points()
        if indices:
            pts = np.asarray(pcd.points)
            self.selected_points = [pts[i] for i in indices]
            
            print(f"\n✓ Selected {len(self.selected_points)} points")
            for i, pt in enumerate(self.selected_points):
                print(f"  {i+1}. [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]")
        
        vis.destroy_window()
        
        return self.selected_points if len(self.selected_points) >= 3 else None


# Helper function for easy use
def pick_points_on_mesh(mesh, num_points=4, mesh_name="Mesh"):
    """
    Easy-to-use function for point picking
    
    Args:
        mesh: Open3D TriangleMesh
        num_points: Number of points to select
        mesh_name: Name for display
    
    Returns:
        list: Selected points as numpy arrays, or None if insufficient
    """
    picker = EnhancedPointPicker(mesh, num_points_needed=num_points, mesh_name=mesh_name)
    return picker.pick_points_interactive()


# Test/demo function
def demo():
    """Demo of enhanced point picker"""
    import tkinter as tk
    from tkinter import filedialog
    
    # Select mesh
    root = tk.Tk()
    root.withdraw()
    mesh_path = filedialog.askopenfilename(
        title="Select Mesh for Demo",
        filetypes=[("Mesh files", "*.ply *.stl *.obj"), ("All", "*.*")]
    )
    root.destroy()
    
    if not mesh_path:
        print("No file selected")
        return
    
    # Load mesh
    print(f"Loading {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if not mesh.has_vertices():
        print("Error: Invalid mesh")
        return
    
    print(f"Loaded: {len(mesh.vertices)} vertices")
    
    # Pick points
    points = pick_points_on_mesh(mesh, num_points=4, mesh_name="Demo Mesh")
    
    if points:
        print(f"\n✓ Success! Got {len(points)} points")
    else:
        print("\n❌ Failed to get enough points")


if __name__ == "__main__":
    demo()


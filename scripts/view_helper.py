"""
Quick viewer helper - converts PLY to other formats for easier viewing
"""
import open3d as o3d
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "../data/raw_scene.ply")

# Load the point cloud
pcd = o3d.io.read_point_cloud(INPUT_PATH)

print(f"Loaded: {len(pcd.points):,} points")
print(f"Has colors: {pcd.has_colors()}")

# Save as different formats for easier viewing
base_path = os.path.join(os.path.dirname(__file__), "../data/raw_scene")

# XYZ format (simple text)
o3d.io.write_point_cloud(f"{base_path}.xyz", pcd)
print(f"Saved XYZ: {base_path}.xyz")

# PCD format  
o3d.io.write_point_cloud(f"{base_path}.pcd", pcd)
print(f"Saved PCD: {base_path}.pcd")

# Also create a basic mesh for OBJ export
print("Creating mesh approximation...")
pcd.estimate_normals()
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
o3d.io.write_triangle_mesh(f"{base_path}_mesh.obj", mesh)
print(f"Saved mesh: {base_path}_mesh.obj")

print("\nFile locations:")
print(f"  PLY: {INPUT_PATH}")
print(f"  XYZ: {base_path}.xyz") 
print(f"  OBJ: {base_path}_mesh.obj")
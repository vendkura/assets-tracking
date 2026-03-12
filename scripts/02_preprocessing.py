"""
Script 02 — Preprocessing
==========================
Takes the raw synthetic scene and applies standard point cloud
preprocessing steps used in industrial pipelines:

  1. Voxel downsampling      — reduces density, speeds up processing
  2. Statistical outlier removal — strips background noise / clutter
  3. Normal estimation       — computes surface normals (needed for segmentation)

Input  : data/raw_scene.ply
Output : data/preprocessed.ply
"""

import numpy as np
import open3d as o3d
import os

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "../data/raw_scene.ply")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../data/preprocessed.ply")

print("=" * 60)
print("  Preprocessing Pipeline")
print("=" * 60)

# ─── Load ─────────────────────────────────────────────────────────────────────
pcd = o3d.io.read_point_cloud(INPUT_PATH)
print(f"\n  [LOAD] Raw scene: {len(pcd.points):,} points")

# ─── Step 1: Voxel Downsampling ───────────────────────────────────────────────
# Replaces dense clusters with one point per voxel cube
VOXEL_SIZE = 0.015   # 1.5 cm voxels — realistic for industrial LiDAR

pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
n_after_down = len(pcd_down.points)
reduction = (1 - n_after_down / len(pcd.points)) * 100

print(f"\n  [STEP 1] Voxel Downsampling (voxel_size={VOXEL_SIZE}m)")
print(f"    Before : {len(pcd.points):,} pts")
print(f"    After  : {n_after_down:,} pts")
print(f"    Reduced: {reduction:.1f}%")

# ─── Step 2: Statistical Outlier Removal ─────────────────────────────────────
# For each point, computes mean distance to its k nearest neighbours.
# Points with distance > mean + (std_ratio * std) are removed.
NB_NEIGHBORS = 20
STD_RATIO    = 1.5

pcd_clean, inlier_idx = pcd_down.remove_statistical_outlier(
    nb_neighbors=NB_NEIGHBORS,
    std_ratio=STD_RATIO
)
n_removed = n_after_down - len(pcd_clean.points)
outlier_pct = (n_removed / n_after_down) * 100

print(f"\n  [STEP 2] Statistical Outlier Removal")
print(f"    nb_neighbors : {NB_NEIGHBORS}")
print(f"    std_ratio    : {STD_RATIO}")
print(f"    Removed      : {n_removed:,} outlier pts ({outlier_pct:.1f}%)")
print(f"    Remaining    : {len(pcd_clean.points):,} pts")

# ─── Step 3: Normal Estimation ────────────────────────────────────────────────
# Computes surface normals using local neighborhood (KD-tree).
# Normals are essential for cylindrical shape detection downstream.
NORMAL_RADIUS    = 0.05   # 5 cm search radius
NORMAL_MAX_NN    = 30     # max neighbours considered

pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
    )
)
# Orient normals consistently (towards the camera at origin)
pcd_clean.orient_normals_towards_camera_location(camera_location=[0, -2, 2])

print(f"\n  [STEP 3] Normal Estimation")
print(f"    Search radius : {NORMAL_RADIUS}m")
print(f"    Max neighbours: {NORMAL_MAX_NN}")
print(f"    Normals computed: {len(pcd_clean.normals):,}")

# ─── Save ─────────────────────────────────────────────────────────────────────
o3d.io.write_point_cloud(OUTPUT_PATH, pcd_clean)

print()
print("─" * 60)
print(f"  Final clean cloud : {len(pcd_clean.points):,} points")
print(f"  Has normals       : {pcd_clean.has_normals()}")
print(f"  Has colors        : {pcd_clean.has_colors()}")
print(f"  Saved → {OUTPUT_PATH}")
print("─" * 60)

# ─── Visualize: side by side comparison ──────────────────────────────────────
print("\n  Launching viewer — LEFT: raw  |  RIGHT: clean + normals")
print("  (press Q to close)\n")

# Shift the clean cloud to the right for side-by-side
pcd_shifted = o3d.geometry.PointCloud(pcd_clean)
pts = np.asarray(pcd_shifted.points)
pts[:, 0] += 3.5
pcd_shifted.points = o3d.utility.Vector3dVector(pts)

# Color raw cloud uniformly gray for comparison
pcd_gray = o3d.geometry.PointCloud(pcd)
pcd_gray.paint_uniform_color([0.5, 0.5, 0.5])

# Draw normals on clean cloud
# o3d.visualization.draw_geometries(
#     [pcd_gray, pcd_shifted],
#     window_name="Script 02 — Preprocessing  |  Left: Raw   Right: Clean",
#     width=1400, height=700,
#     point_show_normal=True
# )

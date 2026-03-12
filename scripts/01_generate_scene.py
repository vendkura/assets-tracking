"""
Script 01 — Generate Synthetic Industrial Point Cloud
======================================================
Simulates a LiDAR scan of an industrial plant section containing:
  - Straight pipes (horizontal + vertical)
  - Elbow connectors
  - Valves (modeled as thick cylinders / flanges)
  - Structural noise (clutter, background floor/wall)

Output: data/raw_scene.ply
"""

import numpy as np
import open3d as o3d
import os

# ─── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../data/raw_scene.ply")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# ─── Helper: Sample points on a cylinder surface ──────────────────────────────
def cylinder_points(center, axis, radius, length, n_points, noise_std=0.005):
    """
    Generate surface points on a cylinder.
    center : (3,) midpoint
    axis   : (3,) unit direction vector
    radius : float
    length : float
    """
    axis = axis / np.linalg.norm(axis)

    # Build orthonormal basis perpendicular to axis
    perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(axis, perp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    t      = np.random.uniform(-length / 2, length / 2, n_points)
    theta  = np.random.uniform(0, 2 * np.pi, n_points)

    pts = (center
           + t[:, None] * axis
           + radius * np.cos(theta)[:, None] * u
           + radius * np.sin(theta)[:, None] * v)

    pts += np.random.normal(0, noise_std, pts.shape)
    return pts


# ─── Helper: Torus arc (elbow) ────────────────────────────────────────────────
def elbow_points(center, R, r, angle_start, angle_end, plane_normal, n_points, noise_std=0.005):
    """
    Torus section (elbow bend).
    R : major radius (bend radius)
    r : pipe radius (tube radius)
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    perp = np.array([1, 0, 0]) if abs(plane_normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(plane_normal, perp); u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)

    phi   = np.random.uniform(angle_start, angle_end, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    # Torus parametric surface
    cx = (R + r * np.cos(theta)) * np.cos(phi)
    cy = (R + r * np.cos(theta)) * np.sin(phi)
    cz = r * np.sin(theta)

    pts = center + cx[:, None] * u + cy[:, None] * v + cz[:, None] * plane_normal
    pts += np.random.normal(0, noise_std, pts.shape)
    return pts


# ─── Helper: Valve (flanged thick disk) ───────────────────────────────────────
def valve_points(center, axis, body_radius, body_length, flange_radius, n_points, noise_std=0.007):
    """
    Valve = thick body cylinder + two flanges (disks at each end).
    """
    axis = axis / np.linalg.norm(axis)
    perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(axis, perp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    all_pts = []

    # Body
    body = cylinder_points(center, axis, body_radius, body_length, int(n_points * 0.5), noise_std)
    all_pts.append(body)

    # Two flanges (disks)
    for sign in [-1, 1]:
        fc = center + sign * (body_length / 2) * axis
        t  = np.random.uniform(0, flange_radius, int(n_points * 0.25))
        theta = np.random.uniform(0, 2 * np.pi, int(n_points * 0.25))
        disk = fc + t[:, None] * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
        disk += np.random.normal(0, noise_std, disk.shape)
        all_pts.append(disk)

    return np.vstack(all_pts)


# ─── Helper: Background clutter ───────────────────────────────────────────────
def background_noise(bounds, n_points):
    """Random scattered points simulating LiDAR reflections off walls/floor."""
    pts = np.random.uniform(bounds[0], bounds[1], (n_points, 3))
    return pts


# ─── Build the scene ──────────────────────────────────────────────────────────
print("=" * 60)
print("  Industrial Point Cloud Generator")
print("  Simulating plant section: pipes + valves + noise")
print("=" * 60)

all_points = []
all_labels = []   # 0=background, 1=pipe, 2=elbow, 3=valve

PIPE_RADIUS  = 0.05   # 5 cm radius pipes
PIPE_NOISE   = 0.004
VALVE_NOISE  = 0.007
BG_NOISE     = 0.01

# ── Pipe network layout ───────────────────────────────────────────────────────

# Pipe A: horizontal, X direction — stops before valve (gap at x=0.35 and x=0.65)
pA1 = cylinder_points([-0.8, 0.0, 0.5], [1, 0, 0], PIPE_RADIUS, 1.1, 900, PIPE_NOISE)
pA2 = cylinder_points([0.75, 0.0, 0.5], [1, 0, 0], PIPE_RADIUS, 0.45, 400, PIPE_NOISE)
all_points.extend([pA1, pA2])
all_labels += [1] * (len(pA1) + len(pA2))
print(f"  [+] Pipe A (horizontal X)   : {len(pA1)+len(pA2):,} pts")

# Pipe B: continues after elbow, going in Y direction (gap from elbow)
pB = cylinder_points([1.1, 0.65, 0.5], [0, 1, 0], PIPE_RADIUS, 0.7, 700, PIPE_NOISE)
all_points.append(pB); all_labels += [1] * len(pB)
print(f"  [+] Pipe B (horizontal Y)   : {len(pB):,} pts")

# Pipe C: vertical riser (gap from elbow)
pC = cylinder_points([1.1, 1.1, 1.05], [0, 0, 1], PIPE_RADIUS, 1.0, 900, PIPE_NOISE)
all_points.append(pC); all_labels += [1] * len(pC)
print(f"  [+] Pipe C (vertical Z)     : {len(pC):,} pts")

# Pipe D: another horizontal run at different height — stops before valve
pD1 = cylinder_points([-1.1, 0.8, 1.2], [1, 0, 0], PIPE_RADIUS * 1.3, 0.85, 700, PIPE_NOISE)
pD2 = cylinder_points([0.15, 0.8, 1.2], [1, 0, 0], PIPE_RADIUS * 1.3, 0.85, 700, PIPE_NOISE)
all_points.extend([pD1, pD2])
all_labels += [1] * (len(pD1) + len(pD2))
print(f"  [+] Pipe D (wide horizontal): {len(pD1)+len(pD2):,} pts")

# Pipe E: partially occluded behind pipe A
pE = cylinder_points([0.1, 0.09, 0.5], [1, 0, 0], PIPE_RADIUS * 0.8, 0.8, 350, PIPE_NOISE)
all_points.append(pE); all_labels += [1] * len(pE)
print(f"  [+] Pipe E (occluded)       : {len(pE):,} pts  ← partial occlusion")

# ── Elbows ────────────────────────────────────────────────────────────────────
eA = elbow_points([1.05, 0.12, 0.5], R=0.10, r=PIPE_RADIUS,
                  angle_start=0, angle_end=np.pi/2,
                  plane_normal=[0, 0, 1], n_points=500, noise_std=PIPE_NOISE)
all_points.append(eA); all_labels += [2] * len(eA)
print(f"  [+] Elbow A (A→B bend)      : {len(eA):,} pts")

eB = elbow_points([1.1, 1.05, 0.62], R=0.10, r=PIPE_RADIUS,
                  angle_start=0, angle_end=np.pi/2,
                  plane_normal=[1, 0, 0], n_points=500, noise_std=PIPE_NOISE)
all_points.append(eB); all_labels += [2] * len(eB)
print(f"  [+] Elbow B (B→C bend)      : {len(eB):,} pts")

# ── Valves ────────────────────────────────────────────────────────────────────
vA = valve_points([0.5, 0.0, 0.5], [1, 0, 0],
                  body_radius=0.08, body_length=0.16, flange_radius=0.12,
                  n_points=700, noise_std=VALVE_NOISE)
all_points.append(vA); all_labels += [3] * len(vA)
print(f"  [+] Valve A (on Pipe A)     : {len(vA):,} pts")

vB = valve_points([-0.22, 0.8, 1.2], [1, 0, 0],
                  body_radius=0.10, body_length=0.20, flange_radius=0.15,
                  n_points=700, noise_std=VALVE_NOISE)
all_points.append(vB); all_labels += [3] * len(vB)
print(f"  [+] Valve B (on Pipe D)     : {len(vB):,} pts")

# ── Background / clutter ──────────────────────────────────────────────────────
bg = background_noise([[-1.2, -0.5, -0.1], [2.0, 2.0, 2.5]], 1800)
all_points.append(bg); all_labels += [0] * len(bg)
print(f"  [+] Background clutter      : {len(bg):,} pts")

# ─── Assemble ─────────────────────────────────────────────────────────────────
points = np.vstack(all_points)
labels = np.array(all_labels)

# Assign colors per class for visualization
COLOR_MAP = {
    0: [0.35, 0.35, 0.35],   # background → gray
    1: [0.2,  0.6,  1.0],    # pipe       → blue
    2: [0.2,  0.9,  0.5],    # elbow      → green
    3: [1.0,  0.45, 0.1],    # valve      → orange
}
colors = np.array([COLOR_MAP[l] for l in labels])

# ─── Create Open3D point cloud ────────────────────────────────────────────────
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save
o3d.io.write_point_cloud(OUTPUT_PATH, pcd)

# ─── Summary ──────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print(f"  Total points generated : {len(points):,}")
print(f"    Background           : {np.sum(labels == 0):,}")
print(f"    Pipes                : {np.sum(labels == 1):,}")
print(f"    Elbows               : {np.sum(labels == 2):,}")
print(f"    Valves               : {np.sum(labels == 3):,}")
print(f"  Saved → {OUTPUT_PATH}")
print("─" * 60)
print()
print("  Launching viewer  (press Q to close) ...")
print()

# ─── Visualize ────────────────────────────────────────────────────────────────
# o3d.visualization.draw_geometries(
#     [pcd],
#     window_name="Script 01 — Raw Industrial Scene",
#     width=1200, height=700,
#     point_show_normal=False
# )
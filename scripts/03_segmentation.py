"""
Script 03 — Segmentation (v2 — Two-Pass Strategy)
===================================================
The core challenge: a single EPS in DBSCAN cannot simultaneously
  - KEEP a long pipe as one cluster (needs large EPS)
  - SEPARATE a pipe from an adjacent valve (needs small EPS)

Solution: Two-pass approach used in real industrial pipelines:

  PASS 1 — Large-EPS DBSCAN + cylinder geometry check (PCA)
    → Clusters at large EPS keep pipes whole. Then PCA verifies
      which clusters are truly cylindrical (pipes).

  PASS 2 — Tighter DBSCAN on remaining points
    → Only compact objects remain (valves, elbows). Works well now
      because long pipes are already removed.

Input  : data/preprocessed.ply
Output : data/segmented.ply  /  data/segments/  /  data/segments_meta.json
"""

import numpy as np
import open3d as o3d
import os
import json

INPUT_PATH   = os.path.join(os.path.dirname(__file__), "../data/preprocessed.ply")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "../data/segmented.ply")
SEGMENTS_DIR = os.path.join(os.path.dirname(__file__), "../data/segments")
os.makedirs(SEGMENTS_DIR, exist_ok=True)

for f in os.listdir(SEGMENTS_DIR):
    os.remove(os.path.join(SEGMENTS_DIR, f))

print("=" * 60)
print("  Segmentation Pipeline  (Two-Pass Strategy)")
print("=" * 60)

CLASS_COLORS = {
    "pipe"    : [0.2,  0.6,  1.0],
    "valve"   : [1.0,  0.45, 0.1],
    "elbow"   : [0.2,  0.9,  0.5],
    "unknown" : [0.7,  0.2,  0.7],
    "noise"   : [0.25, 0.25, 0.25],
}

# ── Load ──────────────────────────────────────────────────────────────────────
pcd     = o3d.io.read_point_cloud(INPUT_PATH)
all_pts = np.asarray(pcd.points).copy()
colors  = np.full((len(all_pts), 3), 0.25)
print(f"\n  [LOAD] {len(all_pts):,} points")

cluster_info = []
seg_id       = 0

# ── Step 1: Remove floor/wall ─────────────────────────────────────────────────
print(f"\n  [STEP 1] Floor/Wall Removal (RANSAC Plane)")
remaining = o3d.geometry.PointCloud(pcd)
_, plane_idx = remaining.segment_plane(
    distance_threshold=0.025, ransac_n=3, num_iterations=1000
)
remaining = remaining.select_by_index(plane_idx, invert=True)
rem_pts   = np.asarray(remaining.points).copy()
print(f"    Removed {len(plane_idx):,} plane pts  →  {len(rem_pts):,} remaining")

# ── Helper: PCA cylinder check ────────────────────────────────────────────────
def pca_cylinder_check(pts):
    """Return (is_pipe, elongation, linearity, radius_est)"""
    if len(pts) < 30:
        return False, 0.0, 0.0, 0.0
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order   = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis    = eigvecs[:, order[0]]

    elongation = float(eigvals[0] / (eigvals[1] + 1e-9))
    linearity  = float(eigvals[0] / (eigvals.sum() + 1e-9))

    proj     = (pts - centroid) @ axis
    residual = pts - centroid - np.outer(proj, axis)
    radii    = np.linalg.norm(residual, axis=1)
    r_med    = float(np.median(radii))
    r_std    = float(np.std(radii))
    radial_ok = (r_std / (r_med + 1e-9)) < 0.6

    # Relaxed thresholds — pipes in a synthetic scene with moderate noise
    is_pipe = elongation > 3.5 and linearity > 0.55 and radial_ok
    return is_pipe, elongation, linearity, r_med

# ── PASS 1: Large-EPS cluster → PCA verify → extract pipes ───────────────────
print(f"\n  [PASS 1] Pipe Detection  (large EPS + PCA cylinder check)")

labels1 = np.array(o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(rem_pts)
).cluster_dbscan(eps=0.06, min_points=25, print_progress=False))

n1 = int(labels1.max()) + 1
print(f"    Clusters (EPS=0.06, min_pts=25): {n1}")
print(f"    {'CID':>4}  {'PTS':>6}  {'ELONG':>7}  {'LIN':>6}  {'RAD_OK':>7}  {'RESULT'}")
print(f"    {'-'*50}")

pipe_point_mask = np.zeros(len(rem_pts), dtype=bool)

for cid in range(n1):
    mask = labels1 == cid
    pts  = rem_pts[mask]
    is_pipe, elong, lin, rad = pca_cylinder_check(pts)

    # Compute radial_ok separately for display
    if len(pts) >= 30:
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals_d, eigvecs_d = np.linalg.eigh(cov)
        axis_d = eigvecs_d[:, np.argsort(eigvals_d)[::-1][0]]
        proj_d = (pts - centroid) @ axis_d
        res_d  = pts - centroid - np.outer(proj_d, axis_d)
        radii_d = np.linalg.norm(res_d, axis=1)
        rok = (np.std(radii_d) / (np.median(radii_d) + 1e-9)) < 0.6
    else:
        rok = False

    result = "✓ PIPE" if is_pipe else "✗ skip"
    print(f"    {cid:>4}  {len(pts):>6}  {elong:>7.2f}  {lin:>6.3f}  {str(rok):>7}  {result}")

pipe_point_mask = np.zeros(len(rem_pts), dtype=bool)

for cid in range(n1):
    mask = labels1 == cid
    pts  = rem_pts[mask]
    is_pipe, elong, lin, rad = pca_cylinder_check(pts)

    if not is_pipe:
        continue
    for i, p in enumerate(pts):
        idx = np.argmin(np.linalg.norm(all_pts - p, axis=1))
        colors[idx] = CLASS_COLORS["pipe"]

    pipe_point_mask[mask] = True
    bbox     = pts.max(axis=0) - pts.min(axis=0)
    centroid = pts.mean(axis=0)

    cluster_info.append({
        "cluster_id"  : seg_id,
        "label"       : "pipe",
        "n_points"    : int(len(pts)),
        "elongation"  : round(elong, 3),
        "linearity"   : round(lin, 3),
        "bbox_xyz"    : [round(float(b), 4) for b in bbox],
        "radius_est_m": round(rad, 4),
        "centroid"    : [round(float(c), 4) for c in centroid],
    })

    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points = o3d.utility.Vector3dVector(pts)
    seg_pcd.paint_uniform_color(CLASS_COLORS["pipe"])
    o3d.io.write_point_cloud(
        os.path.join(SEGMENTS_DIR, f"segment_{seg_id:02d}_pipe.ply"), seg_pcd
    )
    print(f"    Cluster {cid:02d} → PIPE  | pts={len(pts):4d}  "
          f"elong={elong:.1f}  lin={lin:.2f}  r={rad:.3f}m")
    seg_id += 1

pipe_n = sum(1 for c in cluster_info if c["label"] == "pipe")
print(f"    → {pipe_n} pipe(s) extracted")

# ── PASS 2: Compact objects from remainder ────────────────────────────────────
print(f"\n  [PASS 2] Compact Component Detection  (valves / elbows)")

# Remaining = non-pipe clusters + noise from pass 1
non_pipe_pts = rem_pts[~pipe_point_mask]
print(f"    Points remaining after pipe removal: {len(non_pipe_pts):,}")

if len(non_pipe_pts) > 10:
    rem2     = o3d.geometry.PointCloud()
    rem2.points = o3d.utility.Vector3dVector(non_pipe_pts)
    labels2  = np.array(rem2.cluster_dbscan(
        eps=0.055, min_points=12, print_progress=False
    ))
    n2 = int(labels2.max()) + 1
    print(f"    Clusters (EPS=0.055): {n2}")

    for cid2 in range(n2):
        mask2 = labels2 == cid2
        pts2  = non_pipe_pts[mask2]
        n     = len(pts2)
        if n < 12:
            continue

        bbox      = pts2.max(axis=0) - pts2.min(axis=0)
        extent    = np.sort(bbox)
        elongation = float(extent[2] / (extent[1] + 1e-6))
        centroid  = pts2.mean(axis=0)
        cov       = np.cov((pts2 - centroid).T)
        eigvals   = np.sort(np.abs(np.linalg.eigvalsh(cov)))
        linearity = float((eigvals[2] - eigvals[1]) / (eigvals[2] + 1e-6))

        # Catch residual pipes that DBSCAN merged with compact objects in pass 1
        if elongation > 4.5 and linearity > 0.85:
            label = "pipe"
        elif 2.5 <= elongation <= 5.5 and linearity < 0.70:
            label = "elbow"
        elif elongation < 2.5 and n >= 40:
            # Valves need enough points to be credible — avoid noise blobs
            label = "valve"
        else:
            label = "unknown"

        for p in pts2:
            idx = np.argmin(np.linalg.norm(all_pts - p, axis=1))
            colors[idx] = CLASS_COLORS[label]

        cluster_info.append({
            "cluster_id"  : seg_id,
            "label"       : label,
            "n_points"    : int(n),
            "elongation"  : round(elongation, 3),
            "linearity"   : round(linearity, 3),
            "bbox_xyz"    : [round(float(b), 4) for b in bbox],
            "radius_est_m": round(float(extent[0] / 2), 4),
            "centroid"    : [round(float(c), 4) for c in centroid],
        })

        seg_pcd2 = o3d.geometry.PointCloud()
        seg_pcd2.points = o3d.utility.Vector3dVector(pts2)
        seg_pcd2.paint_uniform_color(CLASS_COLORS[label])
        o3d.io.write_point_cloud(
            os.path.join(SEGMENTS_DIR, f"segment_{seg_id:02d}_{label}.ply"), seg_pcd2
        )
        print(f"    Cluster {cid2:02d} → {label:8s}  | pts={n:4d}  "
              f"elong={elongation:.1f}  lin={linearity:.2f}")
        seg_id += 1

# ── Save ──────────────────────────────────────────────────────────────────────
result = o3d.geometry.PointCloud()
result.points = o3d.utility.Vector3dVector(all_pts)
result.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(OUTPUT_PATH, result)

meta_path = os.path.join(os.path.dirname(__file__), "../data/segments_meta.json")
with open(meta_path, "w") as f:
    json.dump(cluster_info, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
label_counts = {}
for info in cluster_info:
    label_counts[info["label"]] = label_counts.get(info["label"], 0) + 1

print()
print("─" * 60)
print(f"  Total components: {len(cluster_info)}")
for lbl, cnt in sorted(label_counts.items()):
    tag = {"pipe":"blue","valve":"orange","elbow":"green","unknown":"purple"}.get(lbl,"")
    print(f"    {lbl:10s}: {cnt}  [{tag}]")
print(f"  Saved → {OUTPUT_PATH}")
print("─" * 60)
print("\n  Viewer launching...  (Q to close)")
print("  BLUE=pipe  ORANGE=valve  GREEN=elbow  PURPLE=unknown")

# o3d.visualization.draw_geometries(
#     [result],
#     window_name="Script 03 v2 — Two-Pass  |  blue=pipe  orange=valve  green=elbow",
#     width=1200, height=700
# )
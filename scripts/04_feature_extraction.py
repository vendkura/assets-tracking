"""
Script 04 — Feature Extraction
================================
For each detected component (from segmentation), extracts:
  - Component type, estimated radius, length
  - Endpoints (for pipes) or center position (for valves)
  - Bounding box
  - Confidence score

This structured output is what an asset management system (like Antea)
would consume to build a digital twin.

Input  : data/segments_meta.json  +  data/segments/*.ply
Output : data/asset_register.json
         outputs/feature_extraction_report.txt
"""

import numpy as np
import open3d as o3d
import os
import json

SEGMENTS_DIR  = os.path.join(os.path.dirname(__file__), "../data/segments")
META_PATH     = os.path.join(os.path.dirname(__file__), "../data/segments_meta.json")
OUTPUT_JSON   = os.path.join(os.path.dirname(__file__), "../data/asset_register.json")
OUTPUT_REPORT = os.path.join(os.path.dirname(__file__), "../outputs/feature_extraction_report.txt")
os.makedirs(os.path.join(os.path.dirname(__file__), "../outputs"), exist_ok=True)

print("=" * 60)
print("  Feature Extraction — Asset Register Builder")
print("=" * 60)

# ─── Load segment metadata ────────────────────────────────────────────────────
with open(META_PATH) as f:
    segments = json.load(f)

print(f"\n  [LOAD] {len(segments)} segments from metadata\n")

# ─── Feature extraction per component ────────────────────────────────────────

def fit_pipe_axis(pts):
    """
    PCA-based axis fitting for a pipe.
    Returns: axis direction (unit vec), endpoints, length.
    """
    centroid = pts.mean(axis=0)
    cov      = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]   # principal axis = direction of max variance

    # Project points onto axis to find endpoints
    proj  = (pts - centroid) @ axis
    p_min = centroid + proj.min() * axis
    p_max = centroid + proj.max() * axis
    length = float(np.linalg.norm(p_max - p_min))

    return axis, p_min, p_max, length


def estimate_radius(pts, axis, centroid):
    """
    Estimate cylinder radius by computing mean distance from axis.
    """
    axis = axis / np.linalg.norm(axis)
    vecs = pts - centroid
    proj = np.outer(vecs @ axis, axis)
    perp_dist = np.linalg.norm(vecs - proj, axis=1)
    return float(np.median(perp_dist))


def confidence_score(label, n_points, elongation, linearity):
    """
    Simple heuristic confidence based on cluster quality.
    """
    base = {
        "pipe"   : 0.75,
        "valve"  : 0.65,
        "elbow"  : 0.60,
        "unknown": 0.30,
    }.get(label, 0.3)

    # Bonus for large clean clusters
    size_bonus = min(0.15, n_points / 3000)

    # Bonus for clear geometric signature
    if label == "pipe":
        geom_bonus = min(0.10, (linearity - 0.5) * 0.2) if linearity > 0.5 else 0
    else:
        geom_bonus = 0.05

    return round(min(0.99, base + size_bonus + geom_bonus), 2)


# ─── Process each segment ─────────────────────────────────────────────────────
asset_register = []

print(f"  {'ID':>3}  {'TYPE':8}  {'RADIUS(m)':>10}  {'LENGTH(m)':>10}  "
      f"{'CONF':>6}  NOTES")
print("  " + "-" * 60)

for seg in segments:
    cid   = seg["cluster_id"]
    label = seg["label"]

    # Load the individual segment point cloud
    ply_files = [f for f in os.listdir(SEGMENTS_DIR)
                 if f.startswith(f"segment_{cid:02d}_")]
    if not ply_files:
        continue

    seg_pcd = o3d.io.read_point_cloud(os.path.join(SEGMENTS_DIR, ply_files[0]))
    pts     = np.asarray(seg_pcd.points)

    centroid = pts.mean(axis=0)
    asset    = {
        "asset_id"  : f"ASSET-{cid:03d}",
        "type"      : label.upper(),
        "cluster_id": cid,
        "n_points"  : seg["n_points"],
        "centroid"  : [round(float(c), 4) for c in centroid],
        "bbox_m"    : seg["bbox_xyz"],
    }

    notes = ""

    if label in ("pipe", "elbow"):
        axis, ep1, ep2, length = fit_pipe_axis(pts)
        radius = estimate_radius(pts, axis, centroid)

        asset.update({
            "axis_direction": [round(float(a), 4) for a in axis],
            "endpoint_1_m"  : [round(float(e), 4) for e in ep1],
            "endpoint_2_m"  : [round(float(e), 4) for e in ep2],
            "length_m"      : round(length, 4),
            "radius_m"      : round(radius, 4),
            "diameter_mm"   : round(radius * 2 * 1000, 1),
        })

        # Nominal pipe size classification
        diameter_mm = radius * 2 * 1000
        if diameter_mm < 40:
            notes = "DN25 (1\")"
        elif diameter_mm < 70:
            notes = "DN50 (2\")"
        elif diameter_mm < 110:
            notes = "DN80 (3\")"
        else:
            notes = "DN100 (4\"+)"

        if seg.get("elongation", 0) < 2.0:
            notes += "  ⚠ possible occlusion"

    elif label == "valve":
        bbox = np.array(seg["bbox_xyz"])
        body_length = float(bbox.max())
        radius      = estimate_radius(pts,
                                      np.array([1, 0, 0]),   # assume inline
                                      centroid)
        asset.update({
            "body_length_m" : round(body_length, 4),
            "body_radius_m" : round(radius, 4),
            "valve_type"    : "Gate/Globe (estimated)",
        })
        notes = "Inline valve — verify orientation"

    else:
        notes = "Unclassified — manual review needed"

    conf = confidence_score(label, seg["n_points"],
                            seg.get("elongation", 1),
                            seg.get("linearity", 0.5))
    asset["confidence"] = conf

    # Display
    r   = asset.get("radius_m", asset.get("body_radius_m", 0))
    ln  = asset.get("length_m", asset.get("body_length_m", 0))
    print(f"  {cid:>3}  {label:8}  {r:>10.4f}  {ln:>10.4f}  {conf:>6.2f}  {notes}")

    asset_register.append(asset)

# ─── Save asset register ──────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump(asset_register, f, indent=2)

# ─── Generate text report ─────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "  INDUSTRIAL ASSET DETECTION REPORT",
    "  Generated by: 3D Point Cloud Processing Pipeline",
    "  Project: LiDAR-based Digital Twin — Industrial Plant Demo",
    "=" * 65,
    "",
    f"  Total assets detected : {len(asset_register)}",
]

type_counts = {}
for a in asset_register:
    t = a["type"]
    type_counts[t] = type_counts.get(t, 0) + 1

for t, c in sorted(type_counts.items()):
    report_lines.append(f"    {t:10s} : {c}")

report_lines += ["", "-" * 65, "  ASSET DETAILS", "-" * 65, ""]

for asset in asset_register:
    report_lines.append(f"  [{asset['asset_id']}]  Type: {asset['type']}")
    report_lines.append(f"    Position (centroid) : {asset['centroid']}")
    report_lines.append(f"    Confidence          : {asset['confidence']}")

    if "length_m" in asset:
        report_lines.append(f"    Length              : {asset['length_m']} m")
        report_lines.append(f"    Diameter            : {asset.get('diameter_mm', '?')} mm")
        report_lines.append(f"    Endpoints           : {asset['endpoint_1_m']}")
        report_lines.append(f"                          {asset['endpoint_2_m']}")
    elif "body_length_m" in asset:
        report_lines.append(f"    Body length         : {asset['body_length_m']} m")
        report_lines.append(f"    Body radius         : {asset['body_radius_m']} m")
        report_lines.append(f"    Type estimate       : {asset.get('valve_type', '?')}")

    report_lines.append("")

report_lines += [
    "-" * 65,
    "  END OF REPORT",
    "=" * 65,
]

report_text = "\n".join(report_lines)

with open(OUTPUT_REPORT, "w") as f:
    f.write(report_text)

# ─── Final summary ────────────────────────────────────────────────────────────
print()
print("─" * 60)
print(report_text)
print()
print(f"  Asset register → {OUTPUT_JSON}")
print(f"  Report         → {OUTPUT_REPORT}")
print("─" * 60)

# ─── Visualize: annotated final result ───────────────────────────────────────
print("\n  Loading segmented cloud for final visualization ...")

seg_cloud_path = os.path.join(os.path.dirname(__file__), "../data/segmented.ply")
final_pcd = o3d.io.read_point_cloud(seg_cloud_path)

print("  Launching viewer  (press Q to close) ...")
# o3d.visualization.draw_geometries(
#     [final_pcd],
#     window_name="Script 04 — Detected Assets  |  blue=pipe  orange=valve  green=elbow",
#     width=1200, height=700
# )

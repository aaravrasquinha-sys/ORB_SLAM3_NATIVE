"""
convert_trajectory.py — Convert ORB-SLAM3 output into CUT3R's artifact layout.

Reads mono_video outputs (TUM trajectories, map_points.ply, tracking_log.csv)
and produces the exact file layout that CUT3R's analyze_run.py / visualize_map.py
expect, so every existing tool works unchanged.

Pose convention (NOT YET independently verified against the cloned ORB_SLAM3
source — orbslam3_build/ hasn't finished cloning as of writing this. This is
the well-documented, widely-relied-upon ORB-SLAM2/ORB-SLAM3 behavior: System's
SaveTrajectoryTUM/SaveKeyFrameTrajectoryTUM write Twc — translation = camera
centre in world, quaternion = rotation camera-frame -> world-frame
(camera-to-world). Treat this as a strong prior, not a confirmed fact):
  We read the TUM poses directly as 4x4 camera-to-world matrices — no
  inversion applied here. BEFORE trusting this on a real run, main() runs
  check_motion_direction() below: it compares the trajectory's net XZ
  displacement heading against... itself (there's no independent ground-truth
  heading yet), so what it actually catches is degenerate cases (near-zero
  motion, NaN/Inf poses) and prints the heading for a human to eyeball against
  the known real-world walking path. If mono_video.cc / real ORB_SLAM3 source
  is later found to write Twc instead (or some other convention), this
  function and this comment must be revisited — do not assume silently.

Axis convention: assumed OpenCV-style (+X right, +Y down, +Z forward), same
as CUT3R. NOT independently re-derived here — ORB-SLAM3 is built on OpenCV
throughout (cv::Mat camera matrices, standard pinhole projection), so this is
almost certainly correct, but if verification against source/real output ever
disagrees, apply an explicit documented change-of-basis matrix here rather
than silently transposing or renegotiating signs elsewhere.

Gravity alignment modes (config gravity.mode):
  camera_plane     — SVD-fit a plane to camera centres; rotate so its normal
                     maps to world up (-Y in OpenCV convention). Best for a
                     fixed-height rig.
  map_points_ransac — RANSAC a floor plane from sparse map points, mirroring
                     CUT3R's ground_plane fitting approach.
  none             — Skip alignment.

Scale anchoring modes (config scale.mode):
  camera_height    — After gravity alignment, fit a floor plane from map points,
                     measure camera height above it in map units, then apply
                     factor = camera.height_m / measured_height.
  ground_truth_path_length / net_displacement / segment — Scale to a known
                     physical measurement from config groundtruth.<video>.
  manual_factor    — Apply scale.manual_factor directly.

Outputs written to <run_dir>/:
  pred_poses.npy   (N,4,4) camera-to-world, gravity-aligned, metric-scaled
  orb_run_log.json per-frame tracking data + global metadata
  map/merged.ply   Gravity-aligned, scaled map points (matches visualize_map.py)

USAGE:
    python convert_trajectory.py --run_dir results/IMG_1205_run1
    python convert_trajectory.py --run_dir results/IMG_1205_run1 \
        --gravity_mode camera_plane --scale_mode camera_height
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

import pipeline_config as pcfg
from trajectory_coverage import compute_coverage

# ── Keys this script reads from config ──────────────────────────────────────
pcfg.register_keys(
    "shared.random_seed",
    "camera.height_m",
    "extraction.target_fps",
    "gravity.mode",
    "gravity.n_iters",
    "gravity.dist_thresh_frac",
    "gravity.min_inlier_frac",
    "gravity.min_horizontal_dot",
    "scale.mode",
    "scale.manual_factor",
    "scale.min_trajectory_coverage",
    "pointcloud.voxel_size_m",
    "pointcloud.outlier_removal_method",
    "pointcloud.outlier_statistical_nb_neighbors",
    "pointcloud.outlier_statistical_std_ratio",
    "pointcloud.outlier_radius_nb_points",
    "pointcloud.outlier_radius_m",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert ORB-SLAM3 output to CUT3R-compatible artifact layout.")
    ap.add_argument("--run_dir", required=True,
                    help="Run output dir containing CameraTrajectory.txt etc.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--video_name", default=None,
                    help="Video name key for groundtruth lookup in config.")
    ap.add_argument("--gravity_mode", default=None,
                    help="camera_plane | map_points_ransac | none")
    ap.add_argument("--scale_mode", default=None,
                    help="camera_height | ground_truth_path_length | "
                         "ground_truth_net_displacement | ground_truth_segment | manual_factor")
    ap.add_argument("--camera_height_m", type=float, default=None)
    ap.add_argument("--manual_factor", type=float, default=None)
    ap.add_argument("--random_seed", type=int, default=None)
    return ap.parse_args()


def load_config_and_resolve(args):
    cfg = pcfg.load_config(args.config)
    resolved = {}
    resolved["random_seed"]   = pcfg.resolve(args.random_seed,       "shared.random_seed",         0,               cfg, validate=pcfg.non_negative)
    resolved["camera_height_m"] = pcfg.resolve(args.camera_height_m, "camera.height_m",            0.57,            cfg, validate=pcfg.positive)
    resolved["gravity_mode"]  = pcfg.resolve(args.gravity_mode,      "gravity.mode",               "camera_plane",  cfg)
    resolved["n_iters"]       = pcfg.resolve(None,                   "gravity.n_iters",            200,             cfg, validate=pcfg.positive)
    resolved["dist_thresh_frac"] = pcfg.resolve(None,                "gravity.dist_thresh_frac",   0.01,            cfg, validate=pcfg.positive)
    resolved["min_inlier_frac"]  = pcfg.resolve(None,               "gravity.min_inlier_frac",    0.35,            cfg, validate=pcfg.in_range(0,1))
    resolved["min_horizontal_dot"] = pcfg.resolve(None,             "gravity.min_horizontal_dot", 0.7,             cfg, validate=pcfg.in_range(0,1))
    resolved["scale_mode"]    = pcfg.resolve(args.scale_mode,        "scale.mode",                 "camera_height", cfg)
    resolved["manual_factor"] = pcfg.resolve(args.manual_factor,     "scale.manual_factor",        1.0,             cfg, validate=pcfg.positive)
    resolved["min_trajectory_coverage"] = pcfg.resolve(None,         "scale.min_trajectory_coverage", 0.95,         cfg, validate=pcfg.in_range(0,1))
    resolved["target_fps"]    = pcfg.resolve(None,                   "extraction.target_fps",     30.0,            cfg, validate=pcfg.positive)
    resolved["voxel_size_m"]  = pcfg.resolve(None,                   "pointcloud.voxel_size_m",    0.02,            cfg, validate=pcfg.non_negative)
    resolved["outlier_method"] = pcfg.resolve(None,                  "pointcloud.outlier_removal_method", "none",  cfg)
    resolved["outlier_stat_k"] = pcfg.resolve(None,                  "pointcloud.outlier_statistical_nb_neighbors", 20, cfg, validate=pcfg.positive)
    resolved["outlier_stat_std"] = pcfg.resolve(None,                "pointcloud.outlier_statistical_std_ratio",    2.0, cfg, validate=pcfg.positive)
    resolved["outlier_rad_n"]  = pcfg.resolve(None,                  "pointcloud.outlier_radius_nb_points",         16,  cfg, validate=pcfg.positive)
    resolved["outlier_rad_m"]  = pcfg.resolve(None,                  "pointcloud.outlier_radius_m",                 0.05,cfg, validate=pcfg.positive)
    return cfg, resolved


# ── TUM trajectory loader ─────────────────────────────────────────────────────
def load_tum_trajectory(path: Path):
    """Parse TUM format: timestamp tx ty tz qx qy qz qw.
    Returns (timestamps, poses_4x4) where poses_4x4 is (N,4,4) camera-to-world."""
    from scipy.spatial.transform import Rotation

    timestamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            timestamps.append(ts)
            poses.append(T)

    return np.array(timestamps), np.array(poses, dtype=np.float64)


# ── PLY loader ────────────────────────────────────────────────────────────────
def load_ply(path: Path):
    """Load ASCII or binary PLY, returning (N,3) float32 array.
    Falls back to simple ASCII parser to avoid open3d dependency here."""
    lines = path.read_bytes().decode("latin-1").split("\n")
    # Find end_header
    header_end = 0
    n_vertices = 0
    for i, l in enumerate(lines):
        if l.startswith("element vertex"):
            n_vertices = int(l.split()[-1])
        if l.strip() == "end_header":
            header_end = i + 1
            break
    pts = []
    for l in lines[header_end:header_end + n_vertices]:
        parts = l.split()
        if len(parts) >= 3:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float32)


# ── Save PLY ──────────────────────────────────────────────────────────────────
def save_ply(path: Path, pts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"  Saved {path} ({len(pts):,} points)")


# ── Geometry helpers ──────────────────────────────────────────────────────────
def fit_plane_svd(points: np.ndarray):
    """Fit a plane to (N,3) points via SVD. Returns (normal, centroid)."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]          # smallest singular value → normal direction
    return normal / np.linalg.norm(normal), centroid


def ransac_floor_plane(pts: np.ndarray, n_iters: int, dist_thresh: float,
                        min_inlier_frac: float, min_horiz_dot: float, rng):
    """RANSAC plane fit from map points. Returns (normal, d, inlier_mask) or None."""
    N = len(pts)
    if N < 10:
        return None
    best_inliers = None
    best_n = 0
    for _ in range(n_iters):
        idx = rng.sample(range(N), 3)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal /= nrm
        d = -np.dot(normal, p0)
        dists = np.abs(pts @ normal + d)
        inliers = dists < dist_thresh
        n_in = inliers.sum()
        if n_in > best_n:
            best_n = n_in
            best_inliers = inliers
    if best_inliers is None or best_n < min_inlier_frac * N:
        return None
    # Refit on inliers
    normal, centroid = fit_plane_svd(pts[best_inliers])
    d = -np.dot(normal, centroid)
    dists = np.abs(pts @ normal + d)
    inliers = dists < dist_thresh
    if abs(normal[1]) < min_horiz_dot:
        return None   # not horizontal enough
    return normal, d, inliers


def find_horizontal_plane_candidates(pts: np.ndarray, n_iters: int, dist_thresh: float,
                                     min_inlier_frac: float, min_horiz_dot: float, rng,
                                     max_planes: int = 8):
    """Iteratively RANSAC-fit horizontal planes from map points, removing each
    plane's inliers before searching for the next. Returns a list of dicts
    (one per accepted plane): {normal, d, inlier_mask, n_inliers}, each of
    which passed min_horiz_dot and min_inlier_frac (against the ORIGINAL
    point count, so later/smaller planes are held to the same bar as the
    first). Ordered by discovery (roughly inlier count descending), NOT by
    height — the caller picks among these by whatever criterion it wants
    (e.g. lowest vs. floor camera_height, or strongest for gravity alignment).
    """
    N = len(pts)
    if N < 10:
        return []
    remaining_idx = np.arange(N)
    remaining_pts = pts.copy()
    candidates = []
    min_inliers = max(3, int(min_inlier_frac * N))

    for _ in range(max_planes):
        if len(remaining_pts) < 10:
            break
        best_inliers = None
        best_n = 0
        M = len(remaining_pts)
        for _ in range(n_iters):
            idx = rng.sample(range(M), 3)
            p0, p1, p2 = remaining_pts[idx[0]], remaining_pts[idx[1]], remaining_pts[idx[2]]
            v1, v2 = p1 - p0, p2 - p0
            normal = np.cross(v1, v2)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-9:
                continue
            normal /= nrm
            d = -np.dot(normal, p0)
            dists = np.abs(remaining_pts @ normal + d)
            inliers = dists < dist_thresh
            n_in = inliers.sum()
            if n_in > best_n:
                best_n = n_in
                best_inliers = inliers
        if best_inliers is None or best_n < min_inliers:
            break   # no more planes with enough support

        # Refit on inliers for a cleaner normal, then re-test against full
        # remaining set (inliers may grow slightly after refit).
        normal, centroid = fit_plane_svd(remaining_pts[best_inliers])
        d = -np.dot(normal, centroid)
        dists = np.abs(remaining_pts @ normal + d)
        inliers = dists < dist_thresh
        n_in = int(inliers.sum())

        # Consume these points (by original index) before next iteration,
        # regardless of horizontality, so a rejected wall doesn't block
        # discovery of a floor plane hiding behind it.
        global_idx = remaining_idx[inliers]
        remaining_idx = np.delete(remaining_idx, np.where(inliers)[0])
        remaining_pts = pts[remaining_idx]

        if n_in < min_inliers:
            continue
        if abs(normal[1]) < min_horiz_dot:
            continue   # not horizontal enough — discard, keep searching

        full_dists = np.abs(pts @ normal + d)
        full_inlier_mask = full_dists < dist_thresh
        candidates.append({
            "normal": normal, "d": float(d),
            "inlier_mask": full_inlier_mask,
            "n_inliers": int(full_inlier_mask.sum()),
        })

    return candidates


def rotation_to_align(src_vec: np.ndarray, tgt_vec: np.ndarray):
    """Rotation matrix R such that R @ src_vec = tgt_vec (both unit vectors)."""
    src = src_vec / np.linalg.norm(src_vec)
    tgt = tgt_vec / np.linalg.norm(tgt_vec)
    v = np.cross(src, tgt)
    c = np.dot(src, tgt)
    if abs(c + 1.0) < 1e-9:    # 180° rotation edge case
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues 180°
        R = 2 * np.outer(axis, axis) - np.eye(3)
        return R
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-12))
    return R


def apply_rigid(poses_4x4: np.ndarray, R_align: np.ndarray):
    """Apply a 3×3 rotation to all camera-to-world poses."""
    R4 = np.eye(4)
    R4[:3, :3] = R_align
    return R4[None] @ poses_4x4


def apply_rigid_to_points(pts: np.ndarray, R_align: np.ndarray):
    return (R_align @ pts.T).T


# ── Gravity alignment ─────────────────────────────────────────────────────────
def gravity_align(poses_4x4, pts, gravity_mode, n_iters, dist_thresh_frac,
                  min_inlier_frac, min_horiz_dot, rng):
    """Returns (aligned_poses, aligned_pts, R_align, residual_m, floor_info)."""
    world_up = np.array([0.0, -1.0, 0.0])   # -Y in OpenCV convention

    R_align = np.eye(3)
    residual = 0.0
    floor_info = {"mode": gravity_mode, "found": False}

    if gravity_mode == "none":
        return poses_4x4, pts, R_align, residual, floor_info

    cam_centers = poses_4x4[:, :3, 3]   # (N,3)

    if gravity_mode == "camera_plane":
        normal, centroid = fit_plane_svd(cam_centers)
        # Sign: normal should point "upward" = toward -Y. If normal.dot([0,-1,0]) < 0, flip.
        if np.dot(normal, world_up) < 0:
            normal = -normal
        R_align = rotation_to_align(normal, world_up)
        # Residual: RMS distance of cam centres from their mean plane
        centred = cam_centers - centroid
        dists = centred @ normal
        residual = float(np.sqrt(np.mean(dists ** 2)))
        floor_info = {"mode": "camera_plane", "found": True,
                      "plane_normal": normal.tolist(),
                      "plane_centroid": centroid.tolist(),
                      "residual_m": residual}

    elif gravity_mode == "map_points_ransac":
        if pts is None or len(pts) < 10:
            print("[WARN] gravity.mode=map_points_ransac but map_points.ply has < 10 points. "
                  "Falling back to none.")
            return poses_4x4, pts, R_align, residual, floor_info
        # dist_thresh as fraction of max XYZ range
        rng_scale = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        dist_thresh = dist_thresh_frac * rng_scale
        result = ransac_floor_plane(pts, n_iters, dist_thresh, min_inlier_frac,
                                    min_horiz_dot, rng)
        if result is None:
            print("[WARN] RANSAC floor plane not found. Falling back to camera_plane.")
            return gravity_align(poses_4x4, pts, "camera_plane", n_iters,
                                 dist_thresh_frac, min_inlier_frac, min_horiz_dot, rng)
        normal, d, inliers = result
        if np.dot(normal, world_up) < 0:
            normal = -normal; d = -d
        R_align = rotation_to_align(normal, world_up)
        floor_info = {"mode": "map_points_ransac", "found": True,
                      "plane_normal": normal.tolist(), "plane_d": float(d),
                      "n_inliers": int(inliers.sum()), "n_total": len(pts)}
        residual = 0.0

    # Apply rotation to poses and points
    aligned_poses = apply_rigid(poses_4x4, R_align)
    aligned_pts   = apply_rigid_to_points(pts, R_align) if pts is not None else pts
    return aligned_poses, aligned_pts, R_align, residual, floor_info


# ── Scale anchoring ───────────────────────────────────────────────────────────
def compute_scale(poses_4x4, pts, scale_mode, camera_height_m, manual_factor,
                  gt_config, scale_dist_thresh_frac, rng,
                  n_iters=None, min_inlier_frac=None, min_horiz_dot=None,
                  tracking_rows=None):
    """Returns (scale_factor, is_metric, scale_info)."""
    cam_centers = poses_4x4[:, :3, 3]
    info = {"mode": scale_mode, "is_metric": False, "factor": 1.0}

    if scale_mode == "manual_factor":
        info.update({"factor": manual_factor, "is_metric": False})
        return manual_factor, False, info

    if scale_mode == "camera_height":
        # Fit floor plane from map points (even in camera_plane gravity mode)
        if pts is None or len(pts) < 10:
            print("[WARN] scale.mode=camera_height but fewer than 10 map points. "
                  "Cannot measure floor. Falling back to factor=1.0 (non-metric).")
            info["note"] = "insufficient map points"
            return 1.0, False, info

        rng_scale = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        dist_thresh = scale_dist_thresh_frac * rng_scale
        candidates = find_horizontal_plane_candidates(
            pts, n_iters, dist_thresh, min_inlier_frac, min_horiz_dot, rng)

        if not candidates:
            raise RuntimeError(
                "scale.mode=camera_height: no horizontal plane with enough "
                f"inliers (>= {min_inlier_frac:.0%} of {len(pts)} points) and "
                f"normal within min_horizontal_dot={min_horiz_dot} of vertical "
                "was found in map_points.ply. This usually means the floor "
                "is texture-poor and never made it into the sparse map. "
                "Switch scale.mode to ground_truth_path_length (or another "
                "ground_truth_* mode) and fill in groundtruth.<video> in "
                "config.yaml instead."
            )

        # Among accepted horizontal planes, the floor is the one furthest
        # BELOW the camera path (largest positive Y in this gravity-aligned,
        # OpenCV +Y-down convention) — not the one with the most inliers.
        # The strongest plane is very often furniture (a tabletop, a seat)
        # since those tend to be flatter/more textured than a real floor.
        y_cam_mean = float(cam_centers[:, 1].mean())
        best = None
        for c in candidates:
            centroid_y = float(pts[c["inlier_mask"]][:, 1].mean())
            offset = centroid_y - y_cam_mean   # positive: plane below camera
            c["centroid_y"] = centroid_y
            c["offset_map_units"] = offset
            if offset > 1e-6 and (best is None or offset > best["offset_map_units"]):
                best = c

        if best is None:
            raise RuntimeError(
                "scale.mode=camera_height: found horizontal plane candidates "
                f"({[c['n_inliers'] for c in candidates]} inliers each) but none "
                "sit below the camera path (all offsets <= 0). Gravity alignment "
                "may be wrong, or the sparse map has no floor. Switch scale.mode "
                "to a ground_truth_* mode instead."
            )

        measured_height = best["offset_map_units"]
        factor = camera_height_m / measured_height

        plane_summary = {
            "n_candidates": len(candidates),
            "selected_n_inliers": best["n_inliers"],
            "selected_normal": best["normal"].tolist(),
            "selected_offset_map_units": measured_height,
            "candidate_n_inliers": [c["n_inliers"] for c in candidates],
            "candidate_offsets_map_units": [c["offset_map_units"] for c in candidates],
        }

        # ── Plausibility check ──────────────────────────────────────────
        # A wrong floor plane (furniture) silently produced a ~1.7x-2.4x
        # scale error in the past — this must refuse rather than proceed.
        path_len_map_units = float(np.sum(np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)))
        implied_path_length_m = path_len_map_units * factor
        gt = gt_config or {}
        gt_path_len_m = gt.get("path_length_m")

        plausibility = {"checked_against": None, "passed": None}

        if gt_path_len_m is not None:
            rel_err = abs(implied_path_length_m - gt_path_len_m) / gt_path_len_m
            plausibility.update({
                "checked_against": "groundtruth.path_length_m",
                "gt_path_length_m": gt_path_len_m,
                "implied_path_length_m": implied_path_length_m,
                "relative_error": rel_err,
                "passed": rel_err <= 0.25,
            })
            if rel_err > 0.25:
                implied_plane_height_above_floor_m = (
                    camera_height_m - measured_height * (gt_path_len_m / path_len_map_units)
                )
                info.update({"plane_selection": plane_summary, "plausibility": plausibility})
                raise RuntimeError(
                    "scale.mode=camera_height FAILED plausibility check: "
                    f"applying factor={factor:.4f} (from camera_height_m="
                    f"{camera_height_m:.3f} / measured_height={measured_height:.4f} "
                    f"map units) gives path_length={implied_path_length_m:.3f} m, "
                    f"but groundtruth.path_length_m={gt_path_len_m:.3f} m "
                    f"({rel_err:.0%} relative error, threshold 25%). "
                    f"Selected plane is {measured_height:.4f} map units "
                    f"({best['offset_map_units'] * factor:.3f} m at this factor) "
                    "below the camera path. If the ground-truth path length is "
                    "correct, the selected plane sits approximately "
                    f"{implied_plane_height_above_floor_m:.3f} m above the true "
                    "floor — almost certainly furniture, not the floor. Switch "
                    "scale.mode to ground_truth_path_length instead of trying to "
                    "fix camera_height's plane selection further."
                )
        else:
            # No ground truth to check against — nothing to compute a
            # relative error from, so just flag the ambiguity loudly and
            # let the human verify (e.g. render map_points.ply) before
            # trusting the run as metric.
            plausibility.update({
                "checked_against": "camera_height_m sanity bound (no ground truth)",
                "passed": None,
            })
            print(f"\n[WARN] scale.mode=camera_height: no groundtruth.path_length_m to "
                  f"cross-check against for this video. Selected plane offset = "
                  f"{measured_height:.4f} map units, factor={factor:.4f}. This plane "
                  "may be furniture rather than the floor — verify manually (e.g. "
                  "render map_points.ply) before trusting this run as metric.\n")

        info.update({"factor": factor, "is_metric": True,
                     "camera_height_m": camera_height_m,
                     "measured_height_map_units": float(measured_height),
                     "y_cam_mean": y_cam_mean, "y_floor_mean": best["centroid_y"],
                     "plane_selection": plane_summary,
                     "plausibility": plausibility})
        return factor, True, info

    # Ground-truth based modes
    gt = gt_config or {}

    if scale_mode == "ground_truth_path_length":
        gt_val = gt.get("path_length_m")
        if gt_val is None:
            _warn_no_gt(scale_mode, "path_length_m")
            return 1.0, False, info
        measured = float(np.sum(np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)))
        if measured < 1e-9:
            print("[WARN] Measured path length = 0. factor=1.0")
            return 1.0, False, info
        factor = gt_val / measured
        info.update({"factor": factor, "is_metric": True,
                     "gt_path_length_m": gt_val, "measured_path_length": measured})
        return factor, True, info

    if scale_mode == "ground_truth_net_displacement":
        gt_val = gt.get("net_displacement_m")
        if gt_val is None:
            _warn_no_gt(scale_mode, "net_displacement_m")
            return 1.0, False, info
        measured = float(np.linalg.norm(cam_centers[-1] - cam_centers[0]))
        if measured < 1e-9:
            print("[WARN] Measured net displacement = 0. factor=1.0")
            return 1.0, False, info
        factor = gt_val / measured
        info.update({"factor": factor, "is_metric": True,
                     "gt_net_displacement_m": gt_val, "measured_net_displacement": measured})
        return factor, True, info

    if scale_mode == "ground_truth_segment":
        seg = gt.get("segment")
        if seg is None:
            _warn_no_gt(scale_mode, "segment")
            return 1.0, False, info
        f0, f1 = seg["from_frame"], seg["to_frame"]
        gt_dist = seg["distance_m"]

        # cam_centers is indexed by position in CameraTrajectory.txt, which
        # only contains OK/OK_KLT frames — NOT by frame_index (frames that
        # failed to track, e.g. during initialization, are absent). Map
        # frame_index -> pose-array position via tracking_rows so from_frame/
        # to_frame mean what the user wrote in config.yaml.
        if not tracking_rows:
            raise RuntimeError(
                "scale.mode=ground_truth_segment requires tracking_log.csv "
                "(to map frame_index -> pose index) but it was not loaded. "
                "Ensure tracking_log.csv exists in the run directory."
            )
        ok_frame_indices = [int(row["frame_index"]) for row in tracking_rows
                             if row.get("tracking_state") in ("OK", "OK_KLT")]
        if len(ok_frame_indices) != len(cam_centers):
            raise RuntimeError(
                f"scale.mode=ground_truth_segment: tracking_log.csv reports "
                f"{len(ok_frame_indices)} OK/OK_KLT frames but CameraTrajectory.txt "
                f"has {len(cam_centers)} poses — they should match 1:1. Refusing "
                f"to guess a frame_index -> pose mapping."
            )

        if f0 not in ok_frame_indices or f1 not in ok_frame_indices:
            available_lo, available_hi = ok_frame_indices[0], ok_frame_indices[-1]
            raise RuntimeError(
                f"scale.mode=ground_truth_segment: segment bounds "
                f"[from_frame={f0}, to_frame={f1}] are not both tracked "
                f"(OK/OK_KLT) frames. Tracked frames span "
                f"[{available_lo}, {available_hi}] "
                f"({len(ok_frame_indices)} frames total, with gaps for any "
                f"NOT_INITIALIZED/LOST frames in between). Pick from_frame/"
                f"to_frame from frames actually present in tracking_log.csv "
                f"with state OK or OK_KLT."
            )

        i0 = ok_frame_indices.index(f0)
        i1 = ok_frame_indices.index(f1)
        if i1 <= i0:
            raise RuntimeError(
                f"scale.mode=ground_truth_segment: to_frame={f1} does not come "
                f"after from_frame={f0} in the tracked pose sequence."
            )

        n_poses_in_window = i1 - i0 + 1
        if n_poses_in_window < 20:
            raise RuntimeError(
                f"scale.mode=ground_truth_segment: segment [from_frame={f0}, "
                f"to_frame={f1}] spans only {n_poses_in_window} tracked poses "
                f"(minimum 20 required). Too few poses to measure a reliable "
                f"path length. Widen the segment or pick different bounds."
            )

        # Path length (sum of consecutive segment lengths), NOT net
        # displacement — a segment can curve, and path length is what
        # distance_m was hand-measured against.
        window = cam_centers[i0:i1 + 1]
        measured = float(np.sum(np.linalg.norm(np.diff(window, axis=0), axis=1)))
        if measured < 1e-9:
            raise RuntimeError(
                f"scale.mode=ground_truth_segment: measured path length over "
                f"[{f0},{f1}] is ~0 — camera did not move within this window."
            )
        factor = gt_dist / measured
        info.update({"factor": factor, "is_metric": True, "segment": seg,
                     "measured_segment_path_length": measured,
                     "n_poses_in_segment": n_poses_in_window,
                     "is_estimate": True})
        return factor, True, info

    print(f"[WARN] Unknown scale.mode '{scale_mode}'. factor=1.0")
    return 1.0, False, info


def check_whole_route_coverage(scale_mode, scale_info, factor, tracking_rows,
                                target_fps, cam_traj_path, min_coverage):
    """For whole-route GT scale modes (ground_truth_path_length,
    ground_truth_net_displacement), refuse to proceed if tracked coverage of
    the video is below min_coverage. These modes divide a whole-route GT
    distance by the trajectory's measured length; untracked frames mean the
    measured length under-represents the real route, inflating factor by
    roughly 1/coverage. ground_truth_segment is exempt — it is frame-bounded
    by construction, so untracked frames outside [from_frame, to_frame] don't
    bias it.

    Does not raise for other scale modes. Returns the coverage dict (or None
    if not computable) so callers can log it regardless of outcome.
    """
    if scale_mode not in ("ground_truth_path_length", "ground_truth_net_displacement"):
        return None
    if not scale_info.get("is_metric"):
        return None

    cov = compute_coverage(tracking_rows, target_fps, cam_traj_path)
    coverage = cov["coverage"]
    if coverage is None:
        return cov

    cov["passed"] = coverage >= min_coverage
    if cov["passed"]:
        return cov

    missing = [row for row in tracking_rows
               if row.get("tracking_state") not in ("OK", "OK_KLT")]
    missing_states = {}
    for row in missing:
        s = row.get("tracking_state", "")
        missing_states[s] = missing_states.get(s, 0) + 1
    missing_frames_desc = ", ".join(f"{v}x '{k}'" for k, v in sorted(missing_states.items()))

    approx_corrected = factor * coverage
    raise RuntimeError(
        f"scale.mode={scale_mode} FAILED coverage guard: only "
        f"{coverage:.1%} of the video is tracked "
        f"({cov['tracked_duration_s']:.2f}s tracked / {cov['video_duration_s']:.2f}s "
        f"total, {len(cov['ok_indices'])}/{cov['total_frames']} frames OK), "
        f"below scale.min_trajectory_coverage={min_coverage:.0%}. "
        f"Missing/non-OK frames: {len(missing)} ({missing_frames_desc}). "
        f"The factor that would have been applied is {factor:.4f}; the "
        f"roughly corrected value would be factor * coverage ≈ "
        f"{approx_corrected:.4f} — NOT applied, because constant-speed "
        f"correction is an assumption, not a measurement (the user may have "
        f"paused, or accelerated out of the start). "
        f"The ground-truth value for scale.mode={scale_mode} describes the "
        f"whole route, but this trajectory does not cover the whole route. "
        f"Fix by either improving tracking/initialization so coverage rises "
        f"above the threshold, or switching scale.mode to ground_truth_segment "
        f"with GT measured only between the frames that were actually tracked."
    )


def _warn_no_gt(mode, key):
    print(f"\n[WARN] scale.mode={mode} but groundtruth.{key} is null in config.yaml.")
    print(f"  This run will be flagged NON-METRIC (factor=1.0).")
    print(f"  Fill in the groundtruth entry in config.yaml to get metric output.\n")


# ── Voxel downsampling (pure numpy, no open3d required) ──────────────────────
def voxel_downsample(pts: np.ndarray, voxel_size: float):
    if voxel_size <= 0 or len(pts) == 0:
        return pts
    mins = pts.min(axis=0)
    keys = ((pts - mins) / voxel_size).astype(np.int64)
    # Use a dict keyed by voxel tuple
    vox = {}
    for i, k in enumerate(map(tuple, keys)):
        if k not in vox:
            vox[k] = pts[i]
    result = np.array(list(vox.values()), dtype=np.float32)
    return result


# ── Empirical pose-convention sanity check ────────────────────────────────────
def check_motion_direction(cam_poses: np.ndarray, timestamps: np.ndarray):
    """Cheap sanity check on the camera-to-world assumption: print net XZ
    displacement, path length, and start/end positions so a human can eyeball
    whether the trajectory shape/direction matches how the camera actually
    moved during capture. Also catches outright-degenerate output (near-zero
    motion, NaN/Inf) that would silently corrupt everything downstream.

    This is NOT a substitute for verifying the Twc vs Tcw convention against
    source or an independent ground truth — it can't prove correctness, only
    flag obvious breakage.
    """
    if len(cam_poses) < 2:
        print("  [WARN] < 2 poses — cannot check motion direction.")
        return

    centers = cam_poses[:, :3, 3]
    if not np.all(np.isfinite(centers)):
        n_bad = int((~np.isfinite(centers).all(axis=1)).sum())
        print(f"  [WARN] {n_bad}/{len(centers)} camera centres contain NaN/Inf. "
              f"Pose convention or upstream tracking is broken.")

    net_disp = centers[-1] - centers[0]
    net_dist = float(np.linalg.norm(net_disp))
    path_len = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
    heading_deg = float(np.degrees(np.arctan2(net_disp[0], net_disp[2])))  # atan2(X, Z)

    print(f"  Motion check: start={centers[0]}  end={centers[-1]}")
    print(f"  Net XZ displacement: {net_dist:.4f} map units "
          f"(heading {heading_deg:.1f} deg from +Z toward +X)")
    print(f"  Total path length: {path_len:.4f} map units "
          f"(straightness = net/path = {(net_dist / path_len) if path_len > 1e-9 else float('nan'):.3f})")
    if net_dist < 1e-6 and path_len < 1e-6:
        print("  [WARN] Camera barely moved in world frame. If the capture "
              "involved real motion, this strongly suggests poses are NOT "
              "camera-to-world (or tracking failed) — inspect before trusting "
              "downstream metric output.")
    print("  ^ Compare this heading/shape by eye against the known real-world "
          "walking path for this video before trusting the run as metric.")


# ── Load tracking log CSV ─────────────────────────────────────────────────────
def load_tracking_log(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Compute per-frame camera height above fitted floor ────────────────────────
def per_frame_height_above_floor(poses_4x4, pts):
    """Returns (heights, y_floor_mean) where heights[i] is camera i's height above floor."""
    cam_y = poses_4x4[:, 1, 3]
    if pts is None or len(pts) < 4:
        return np.full(len(poses_4x4), np.nan), np.nan
    y_vals = pts[:, 1]
    y_floor = float(pts[y_vals > np.percentile(y_vals, 60)][:, 1].mean())
    heights = y_floor - cam_y   # positive = camera above floor
    return heights.astype(np.float32), y_floor


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg, resolved = load_config_and_resolve(args)

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.exists():
        sys.exit(f"[ERROR] run_dir not found: {run_dir}")

    seed          = resolved["random_seed"][0]
    height_m      = resolved["camera_height_m"][0]
    grav_mode     = resolved["gravity_mode"][0]
    n_iters       = resolved["n_iters"][0]
    dist_frac     = resolved["dist_thresh_frac"][0]
    min_inl       = resolved["min_inlier_frac"][0]
    min_hdot      = resolved["min_horizontal_dot"][0]
    scale_mode    = resolved["scale_mode"][0]
    manual_factor = resolved["manual_factor"][0]
    min_coverage  = resolved["min_trajectory_coverage"][0]
    target_fps    = resolved["target_fps"][0]
    voxel_size    = resolved["voxel_size_m"][0]

    rng = random.Random(seed)

    # Ground-truth config for this video
    gt_cfg = None
    if args.video_name:
        gt_cfg = cfg.get(f"groundtruth.{args.video_name}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    cam_traj_path = run_dir / "CameraTrajectory.txt"
    kf_traj_path  = run_dir / "KeyFrameTrajectory.txt"
    ply_path      = run_dir / "map_points.ply"
    log_path      = run_dir / "tracking_log.csv"

    if not cam_traj_path.exists():
        sys.exit(f"[ERROR] CameraTrajectory.txt not found in {run_dir}. "
                 f"Run mono_video first.")

    print(f"\nLoading trajectories from {run_dir} ...")
    cam_ts, cam_poses = load_tum_trajectory(cam_traj_path)
    kf_ts,  kf_poses  = load_tum_trajectory(kf_traj_path) if kf_traj_path.exists() \
                        else (np.array([]), np.zeros((0, 4, 4)))
    N = len(cam_poses)
    print(f"  CameraTrajectory: {N} poses")
    print(f"  KeyFrameTrajectory: {len(kf_poses)} keyframes")

    pts = None
    if ply_path.exists():
        pts = load_ply(ply_path)
        print(f"  Map points: {len(pts):,}")
    else:
        print(f"  [WARN] map_points.ply not found — scale anchoring limited.")

    print(f"\nPose convention sanity check (raw, pre-alignment) — "
          f"see module docstring; NOT yet verified against ORB_SLAM3 source:")
    check_motion_direction(cam_poses, cam_ts)

    tracking_rows = []
    if log_path.exists():
        tracking_rows = load_tracking_log(log_path)
        print(f"  Tracking log: {len(tracking_rows)} entries")

    # ── Gravity alignment ─────────────────────────────────────────────────────
    print(f"\nGravity alignment: mode={grav_mode}")
    cam_poses_aligned, pts_aligned, R_align, residual, floor_info = gravity_align(
        cam_poses, pts, grav_mode, n_iters, dist_frac, min_inl, min_hdot, rng)
    print(f"  R_align:\n{R_align}")
    print(f"  Camera-centre coplanarity residual: {residual:.4f} map units")

    # Also align keyframe poses
    if len(kf_poses):
        kf_poses_aligned = apply_rigid(kf_poses, R_align)
    else:
        kf_poses_aligned = kf_poses

    # ── Scale anchoring ───────────────────────────────────────────────────────
    print(f"\nScale anchoring: mode={scale_mode}")
    scale_factor, is_metric, scale_info = compute_scale(
        cam_poses_aligned, pts_aligned, scale_mode, height_m, manual_factor,
        gt_cfg, dist_frac, rng, n_iters, min_inl, min_hdot,
        tracking_rows=tracking_rows)

    print(f"  Scale factor: {scale_factor:.6f}  (is_metric={is_metric})")
    if not is_metric:
        print("  [NON-METRIC] Output is in arbitrary units. "
              "Do not compare distances against physical measurements.")

    coverage_info = check_whole_route_coverage(
        scale_mode, scale_info, scale_factor, tracking_rows, target_fps,
        cam_traj_path, min_coverage)
    if coverage_info is not None:
        print(f"  Coverage: {coverage_info['coverage']:.1%} "
              f"({len(coverage_info['ok_indices'])}/{coverage_info['total_frames']} frames OK) "
              f"— guard {'passed' if coverage_info.get('passed') else 'N/A'}")

    # Apply scale to translations only
    cam_poses_scaled = cam_poses_aligned.copy()
    cam_poses_scaled[:, :3, 3] *= scale_factor
    pts_scaled = pts_aligned * scale_factor if pts_aligned is not None else None

    # ── Per-frame height above floor (after alignment + scale) ────────────────
    heights, y_floor = per_frame_height_above_floor(cam_poses_scaled, pts_scaled)

    # ── Save pred_poses.npy ───────────────────────────────────────────────────
    # CUT3R expects (N,4,4) camera-to-world, matching what analyze_run.py loads.
    # Our cam_poses_scaled is exactly that.
    out_poses = run_dir / "pred_poses.npy"
    np.save(out_poses, cam_poses_scaled.astype(np.float32))
    print(f"\n  pred_poses.npy -> {out_poses}  shape={cam_poses_scaled.shape}")

    # ── Save map/merged.ply ───────────────────────────────────────────────────
    if pts_scaled is not None:
        # Optional voxel downsample
        pts_final = voxel_downsample(pts_scaled, voxel_size)
        # Optional outlier removal (open3d if available)
        out_method = resolved["outlier_method"][0]
        if out_method != "none":
            pts_final = _outlier_removal(pts_final, out_method, resolved)
        map_path = run_dir / "map" / "merged.ply"
        save_ply(map_path, pts_final)

    # ── Build orb_run_log.json ────────────────────────────────────────────────
    per_frame = []
    for i, row in enumerate(tracking_rows):
        h = float(heights[i]) if i < len(heights) and not np.isnan(heights[i]) else None
        per_frame.append({
            "frame":           int(row.get("frame_index", i)),
            "timestamp":       float(row.get("timestamp", 0)),
            "tracking_state":  row.get("tracking_state", ""),
            "n_keypoints":     int(row.get("n_keypoints", 0)),
            "n_map_matches":   int(row.get("n_map_matches", 0)),
            "is_keyframe":     row.get("is_keyframe", "0") == "1",
            "height_above_floor_m": h,
        })

    coverage_log = None
    if coverage_info is not None:
        coverage_log = {
            "coverage": coverage_info["coverage"],
            "tracked_frames": len(coverage_info["ok_indices"]),
            "total_frames": coverage_info["total_frames"],
            "min_trajectory_coverage": min_coverage,
            "guard_passed": coverage_info.get("passed"),
        }

    log_payload = {
        "gravity": {**floor_info, "R_align": R_align.tolist(), "residual_m": residual},
        "scale": {**scale_info, "factor": scale_factor, "is_metric": is_metric},
        "coverage": coverage_log,
        "y_floor_scaled": float(y_floor) if not np.isnan(y_floor) else None,
        "camera_height_m": height_m,
        "n_frames": N,
        "n_keyframes": len(kf_poses),
        "per_frame": per_frame,
        "note_no_dense_depth": (
            "ORB-SLAM3 produces no per-pixel depth (pred_pts3d.npy) or "
            "confidence (pred_conf.npy). Downstream tools degrade gracefully "
            "when those files are absent — see analyze_run.py's load_run()."),
        "note_pose_convention": (
            "pred_poses.npy assumes the TUM trajectory files store Twc "
            "(camera-to-world), per ORB-SLAM2/3's documented "
            "SaveTrajectoryTUM behavior. NOT independently verified against "
            "this project's cloned ORB_SLAM3 source as of this run — see "
            "convert_trajectory.py module docstring and check_motion_direction()."),
    }

    log_out = run_dir / "orb_run_log.json"
    log_out.write_text(json.dumps(log_payload, indent=2))
    print(f"  orb_run_log.json -> {log_out}")

    pcfg.snapshot(resolved, run_dir)
    print(f"\nConversion complete. Run dir: {run_dir}")
    if not is_metric:
        print("  *** NON-METRIC RUN — fill in groundtruth in config.yaml ***")


def _outlier_removal(pts, method, resolved):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if method == "statistical":
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=resolved["outlier_stat_k"][0],
                std_ratio=resolved["outlier_stat_std"][0])
        elif method == "radius":
            cl, _ = pcd.remove_radius_outlier(
                nb_points=resolved["outlier_rad_n"][0],
                radius=resolved["outlier_rad_m"][0])
        else:
            return pts
        return np.asarray(cl.points, dtype=np.float32)
    except ImportError:
        print("[WARN] open3d not installed — skipping outlier removal.")
        return pts


if __name__ == "__main__":
    main()

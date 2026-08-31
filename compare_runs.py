"""
compare_runs.py — Compare a CUT3R run against one or more ORB-SLAM3 runs
(post-convert_trajectory.py) on trajectory shape, scale, and tracking health.

Reads ONLY already-saved artifacts:
  CUT3R run_dir:  pred_poses.npy (N,4,4 cam-to-world), map/merged.ply
  ORB run_dir:    pred_poses.npy (N,4,4 cam-to-world, gravity-aligned +
                   scaled by convert_trajectory.py), map/merged.ply,
                   orb_run_log.json (per-frame tracking state/keypoints/
                   keyframe flag, gravity/scale metadata)

Both pipelines write pred_poses.npy in the same convention (camera-to-world,
OpenCV axes, +X right / +Y down / +Z forward — see visualize_map.py and
convert_trajectory.py docstrings), so positions are directly comparable
without re-deriving axes here.

We do NOT resample either trajectory frame-for-frame: ORB-SLAM3 keyframes
are spaced irregularly (dropped on tracking loss, dense during slow/careful
motion), so a frame-index join would silently compare unrelated moments.
Instead:
  - Shape/scale/health metrics (path length, net displacement, Y span,
    heading-vs-yaw, tracking gaps) are computed independently per run.
  - The ONLY place trajectories are brought into a shared frame is the
    Umeyama/Procrustes alignment of ORB's camera-center point set onto
    CUT3R's (whole-trajectory rigid+scale fit, not per-frame correspondence
    — this is a shape-fitting problem, not a resampling problem). Umeyama
    needs point correspondence, which we approximate by resampling one
    trajectory's arc-length parameterization onto the other's — this is the
    one place interpolation is used, and only for the alignment fit itself,
    not for any reported per-frame metric.

USAGE:
    python compare_runs.py --cut3r_run_dir results/IMG_1205_cut3r \
        --orb_run_dir results/IMG_1205_orb
    python compare_runs.py --cut3r_run_dir results/IMG_1205_cut3r \
        --orb_run_dir results/IMG_1205_orb_run1 results/IMG_1205_orb_run2
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline_config as pcfg

# ── Keys this script reads from config ──────────────────────────────────────
pcfg.register_keys(
    "compare.umeyama_scale_free",
    "compare.axis_limits_shared",
    "diagnostics.heading_min_displacement_m",
    "diagnostics.heading_smooth_window",
    "diagnostics.dpi",
    "visualization.bin_size_m",
    "visualization.scatter_point_size",
    "visualization.scatter_alpha",
    "visualization.height_cmap",
    "visualization.dpi",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compare a CUT3R run against one or more ORB-SLAM3 runs.")
    ap.add_argument("--cut3r_run_dir", required=True, type=Path)
    ap.add_argument("--orb_run_dir", required=True, type=Path, nargs="+",
                     help="One or more ORB-SLAM3 run dirs (post convert_trajectory.py).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output_dir", type=Path, default=None,
                     help="Default: comparisons/<cut3r_run_dir.name>_vs_<orb names>/")
    ap.add_argument("--video_name", default=None,
                     help="Key into config groundtruth.<video_name> for error-vs-GT metrics.")
    return ap.parse_args()


def load_config_and_resolve(args):
    cfg = pcfg.load_config(args.config)
    resolved = {}
    resolved["umeyama_scale_free"] = pcfg.resolve(
        None, "compare.umeyama_scale_free", True, cfg)
    resolved["axis_limits_shared"] = pcfg.resolve(
        None, "compare.axis_limits_shared", True, cfg)
    resolved["heading_min_displacement_m"] = pcfg.resolve(
        None, "diagnostics.heading_min_displacement_m", 0.02, cfg, validate=pcfg.non_negative)
    resolved["heading_smooth_window"] = pcfg.resolve(
        None, "diagnostics.heading_smooth_window", 5, cfg, validate=pcfg.positive)
    resolved["dpi"] = pcfg.resolve(None, "diagnostics.dpi", 120, cfg, validate=pcfg.positive)
    resolved["bin_size_m"] = pcfg.resolve(None, "visualization.bin_size_m", 0.05, cfg, validate=pcfg.positive)
    resolved["scatter_point_size"] = pcfg.resolve(None, "visualization.scatter_point_size", 0.2, cfg, validate=pcfg.positive)
    resolved["scatter_alpha"] = pcfg.resolve(None, "visualization.scatter_alpha", 0.5, cfg, validate=pcfg.in_range(0.0, 1.0))
    resolved["height_cmap"] = pcfg.resolve(None, "visualization.height_cmap", "viridis", cfg)
    resolved["viz_dpi"] = pcfg.resolve(None, "visualization.dpi", 150, cfg, validate=pcfg.positive)
    return cfg, resolved


# ── Loading (mirrors analyze_run.py's load_run / CUT3R conventions) ─────────
def load_poses(run_dir: Path):
    p = run_dir / "pred_poses.npy"
    if not p.exists():
        sys.exit(f"[ERROR] {p} not found. Run the pipeline through pose export first.")
    return np.load(p)  # (N,4,4) camera-to-world


def load_ply(path: Path):
    """ASCII PLY loader, same minimal parser as convert_trajectory.py's — no
    open3d dependency here."""
    if not path.exists():
        return None
    lines = path.read_bytes().decode("latin-1").split("\n")
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
    return np.array(pts, dtype=np.float32) if pts else None


def load_orb_run_log(run_dir: Path):
    p = run_dir / "orb_run_log.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rotation_to_euler(R):
    """Identical convention to CUT3R's analyze_run.py — both pipelines share
    the same axis convention so this is directly reusable."""
    yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
    pitch = np.degrees(np.arcsin(-np.clip(R[1, 2], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[1, 0], R[1, 1]))
    return roll, pitch, yaw


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


# ── Per-run shape/health metrics (no cross-run resampling) ──────────────────
def trajectory_metrics(poses, min_disp, smooth_window):
    n = len(poses)
    centers = poses[:, :3, 3]
    path_length = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if n > 1 else 0.0
    net_disp = float(np.linalg.norm(centers[-1] - centers[0])) if n > 1 else 0.0
    y_span = float(centers[:, 1].max() - centers[:, 1].min()) if n > 0 else 0.0

    yaw = np.array([rotation_to_euler(poses[i][:3, :3])[2] for i in range(n)])

    # Smoothed heading vs yaw disagreement — same logic as CUT3R's plot_heading_vs_yaw
    if n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        sm_x = np.convolve(centers[:, 0], kernel, mode="same")
        sm_z = np.convolve(centers[:, 2], kernel, mode="same")
    else:
        sm_x, sm_z = centers[:, 0], centers[:, 2]

    heading = np.full(n, np.nan)
    for i in range(1, n):
        dx = sm_x[i] - sm_x[i - 1]
        dz = sm_z[i] - sm_z[i - 1]
        if np.hypot(dx, dz) >= min_disp:
            heading[i] = np.degrees(np.arctan2(dx, dz))

    valid = ~np.isnan(heading)
    diff = np.full(n, np.nan)
    diff[valid] = np.abs(wrap180(heading[valid] - yaw[valid]))
    mean_disagreement = float(np.nanmean(diff)) if valid.any() else float("nan")
    max_disagreement = float(np.nanmax(diff)) if valid.any() else float("nan")

    return {
        "n_frames": n,
        "path_length_m": path_length,
        "net_displacement_m": net_disp,
        "y_span_m": y_span,
        "mean_heading_yaw_disagreement_deg": mean_disagreement,
        "max_heading_yaw_disagreement_deg": max_disagreement,
        "start": centers[0].tolist() if n else None,
        "end": centers[-1].tolist() if n else None,
    }


def orb_tracking_health(run_log):
    if run_log is None:
        return {"n_keyframes": None, "longest_tracking_gap_frames": None,
                "n_relocalizations": None, "n_loop_closures": None,
                "scale_mode": None, "scale_factor": None, "is_metric": None}

    per_frame = run_log.get("per_frame", [])
    states = [f.get("tracking_state", "") for f in per_frame]

    # Longest contiguous run of non-OK tracking states
    longest_gap = 0
    cur = 0
    for s in states:
        if s != "OK":
            cur += 1
            longest_gap = max(longest_gap, cur)
        else:
            cur = 0

    return {
        "n_keyframes": run_log.get("n_keyframes"),
        "longest_tracking_gap_frames": longest_gap,
        # ORB-SLAM3's tracking_log.csv (see orbslam_ext/mono_video.cc) does not
        # currently emit discrete relocalization/loop-closure event counts —
        # left None rather than fabricated. Fill in if/when mono_video logs them.
        "n_relocalizations": None,
        "n_loop_closures": None,
        "scale_mode": run_log.get("scale", {}).get("mode"),
        "scale_factor": run_log.get("scale", {}).get("factor"),
        "is_metric": run_log.get("scale", {}).get("is_metric"),
    }


# ── Umeyama / Procrustes alignment ───────────────────────────────────────────
def resample_by_arclength(points, n_samples):
    """Resample a (N,3) polyline to n_samples points evenly spaced by
    cumulative arc length. Used ONLY to build point correspondence for the
    whole-trajectory Umeyama fit below — not used for any reported metric."""
    if len(points) < 2:
        return np.repeat(points, n_samples, axis=0) if len(points) else points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(points[:1], n_samples, axis=0)
    targets = np.linspace(0.0, cum[-1], n_samples)
    out = np.empty((n_samples, 3), dtype=points.dtype)
    for k in range(3):
        out[:, k] = np.interp(targets, cum, points[:, k])
    return out


def umeyama(src, tgt, with_scale=True):
    """Umeyama (1991) closed-form similarity transform: finds R, t, c minimizing
    sum ||tgt_i - (c*R@src_i + t)||^2. Returns (R (3,3), t (3,), c (float), rms).
    with_scale=False fixes c=1 (rigid-only fit)."""
    assert src.shape == tgt.shape
    n, dim = src.shape
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    cov = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        c = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    else:
        c = 1.0

    t = mu_tgt - c * R @ mu_src

    aligned = (c * (R @ src.T).T) + t
    rms = float(np.sqrt(np.mean(np.sum((aligned - tgt) ** 2, axis=1))))
    rot_angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return R, t, c, rms, rot_angle_deg


def align_orb_to_cut3r(cut3r_centers, orb_centers, with_scale):
    n_samples = min(len(cut3r_centers), len(orb_centers))
    n_samples = max(n_samples, 4)
    src = resample_by_arclength(orb_centers, n_samples)
    tgt = resample_by_arclength(cut3r_centers, n_samples)
    R, t, c, rms, rot_deg = umeyama(src, tgt, with_scale=with_scale)
    orb_aligned = (c * (R @ orb_centers.T).T) + t
    return orb_aligned, {"R": R.tolist(), "t": t.tolist(), "scale": c,
                          "rms_residual_m": rms, "rotation_angle_deg": rot_deg}


# ── Ground-truth error ───────────────────────────────────────────────────────
def ground_truth_error(metrics, gt_cfg):
    if not gt_cfg:
        return {}
    out = {}
    for key, metric_key, label in [
        ("path_length_m", "path_length_m", "path_length"),
        ("net_displacement_m", "net_displacement_m", "net_displacement"),
    ]:
        gt_val = gt_cfg.get(key)
        if gt_val is None:
            continue
        measured = metrics[metric_key]
        err_abs = measured - gt_val
        err_pct = (err_abs / gt_val * 100.0) if abs(gt_val) > 1e-9 else float("nan")
        out[f"{label}_gt_m"] = gt_val
        out[f"{label}_error_m"] = err_abs
        out[f"{label}_error_pct"] = err_pct
    return out


# ── Plot: overlaid trajectories (bird's-eye / side / front) ─────────────────
def plot_comparison_trajectories(out_dir, runs, dpi):
    """runs: list of (label, color, poses) — first is CUT3R."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], 0, 2, "X (m)", "Z (m)", "Bird's-eye (X, Z)"),
        (axes[1], 2, 1, "Z (m)", "Y (m)", "Side (Z, Y)"),
        (axes[2], 0, 1, "X (m)", "Y (m)", "Front (X, Y)"),
    ]
    for ax, ia, ib, xlabel, ylabel, title in panels:
        for label, color, poses in runs:
            centers = poses[:, :3, 3]
            ax.plot(centers[:, ia], centers[:, ib], color=color, label=label, linewidth=1.5, alpha=0.85)
            ax.scatter([centers[0, ia]], [centers[0, ib]], color=color, marker="o", s=60,
                       edgecolors="black", zorder=5)
            ax.scatter([centers[-1, ia]], [centers[-1, ib]], color=color, marker="s", s=60,
                       edgecolors="black", zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        if ib == 1:  # Y axis present — invert since +Y is down
            ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Trajectory comparison (circle=start, square=end)")
    fig.tight_layout()
    path = out_dir / "comparison_trajectories.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison_aligned(out_dir, cut3r_centers, orb_runs_aligned, dpi):
    """orb_runs_aligned: list of (label, color, aligned_centers, umeyama_info_scale_free, umeyama_info_scale_fixed)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, variant_idx, title in [(axes[0], 0, "Scale-free (shape only)"),
                                    (axes[1], 1, "Scale-fixed (rigid only)")]:
        ax.plot(cut3r_centers[:, 0], cut3r_centers[:, 2], color="tab:blue", label="CUT3R", linewidth=1.5)
        for label, color, aligned_variants, info_variants in orb_runs_aligned:
            aligned = aligned_variants[variant_idx]
            info = info_variants[variant_idx]
            ax.plot(aligned[:, 0], aligned[:, 2], color=color, linewidth=1.5, alpha=0.85,
                    label=f"{label} (s={info['scale']:.3f}, rms={info['rms_residual_m']:.3f}m)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Umeyama-aligned bird's-eye — {title}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "comparison_aligned.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison_floorplans(out_dir, cut3r_pts, orb_runs_pts, bin_size, point_size,
                                alpha, cmap, shared_limits, dpi):
    """orb_runs_pts: list of (label, pts)."""
    n_panels = 1 + len(orb_runs_pts)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    all_pts = [("CUT3R (dense per-pixel map)", cut3r_pts)] + \
              [(f"{label} (sparse ORB feature map)", pts) for label, pts in orb_runs_pts]

    if shared_limits:
        valid_pts = [p for _, p in all_pts if p is not None and len(p)]
        if valid_pts:
            all_x = np.concatenate([p[:, 0] for p in valid_pts])
            all_z = np.concatenate([p[:, 2] for p in valid_pts])
            xlim = (all_x.min(), all_x.max())
            zlim = (all_z.min(), all_z.max())
        else:
            xlim = zlim = None
    else:
        xlim = zlim = None

    for ax, (label, pts) in zip(axes, all_pts):
        if pts is None or len(pts) == 0:
            ax.set_title(f"{label}\n[no map points]")
            continue
        x, z, y = pts[:, 0], pts[:, 2], pts[:, 1]
        ax.scatter(x, z, s=point_size, c=-y, cmap=cmap, alpha=alpha)
        ax.set_title(f"{label}\nN={len(pts):,} points")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        if xlim:
            ax.set_xlim(xlim)
            ax.set_ylim(zlim)
        ax.invert_yaxis()

    fig.suptitle("Point-count and density are NOT comparable across pipelines: "
                  "CUT3R emits a dense per-pixel point cloud; ORB-SLAM3 emits sparse "
                  "tracked feature points only. ORB's map is not densified here.",
                  fontsize=10, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "comparison_floorplans.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Report writers ────────────────────────────────────────────────────────────
def write_metrics_csv(out_dir, rows):
    path = out_dir / "comparison_metrics.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()}, key=lambda k: (k != "run", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  Saved {path}")


def write_report(out_dir, cut3r_metrics, orb_entries):
    lines = []
    lines.append("=" * 70)
    lines.append("CUT3R vs ORB-SLAM3 comparison report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("CUT3R reference run:")
    lines.append(f"  frames: {cut3r_metrics['n_frames']}")
    lines.append(f"  path length: {cut3r_metrics['path_length_m']:.3f} m")
    lines.append(f"  net displacement: {cut3r_metrics['net_displacement_m']:.3f} m")
    lines.append(f"  Y span: {cut3r_metrics['y_span_m']:.3f} m")
    lines.append(f"  mean heading-vs-yaw disagreement: "
                 f"{cut3r_metrics['mean_heading_yaw_disagreement_deg']:.2f} deg")
    lines.append("")

    for entry in orb_entries:
        lines.append("-" * 70)
        lines.append(f"ORB-SLAM3 run: {entry['label']} ({entry['run_dir']})")
        m = entry["metrics"]
        h = entry["health"]
        u_free = entry["umeyama_scale_free"]
        u_fixed = entry["umeyama_scale_fixed"]
        gt_err = entry["gt_error"]

        lines.append(f"  frames: {m['n_frames']}   keyframes: {h['n_keyframes']}")
        lines.append(f"  path length: {m['path_length_m']:.3f} m")
        lines.append(f"  net displacement: {m['net_displacement_m']:.3f} m")
        lines.append(f"  Y span: {m['y_span_m']:.3f} m")
        lines.append(f"  start->end gap: "
                     f"{np.linalg.norm(np.array(m['end']) - np.array(m['start'])):.3f} m")
        lines.append(f"  mean/max heading-vs-yaw disagreement: "
                     f"{m['mean_heading_yaw_disagreement_deg']:.2f} / "
                     f"{m['max_heading_yaw_disagreement_deg']:.2f} deg")
        lines.append(f"  longest tracking-loss gap: {h['longest_tracking_gap_frames']} frames")
        lines.append(f"  relocalizations: {h['n_relocalizations']}   "
                     f"loop closures: {h['n_loop_closures']}")
        lines.append(f"  scale mode: {h['scale_mode']}   factor: {h['scale_factor']}   "
                     f"metric: {h['is_metric']}")
        if gt_err:
            for k, v in gt_err.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append(f"  Umeyama (scale-free):  scale={u_free['scale']:.4f}  "
                     f"rotation={u_free['rotation_angle_deg']:.2f} deg  "
                     f"rms={u_free['rms_residual_m']:.4f} m")
        lines.append(f"  Umeyama (scale-fixed): rotation={u_fixed['rotation_angle_deg']:.2f} deg  "
                     f"rms={u_fixed['rms_residual_m']:.4f} m")
        lines.append("")

    path = out_dir / "comparison_report.txt"
    path.write_text("\n".join(lines))
    print(f"  Saved {path}")


def main():
    args = parse_args()
    cfg, resolved = load_config_and_resolve(args)

    umeyama_scale_free_default = resolved["umeyama_scale_free"][0]
    axis_limits_shared = resolved["axis_limits_shared"][0]
    min_disp = resolved["heading_min_displacement_m"][0]
    smooth_window = resolved["heading_smooth_window"][0]
    dpi = resolved["dpi"][0]
    bin_size = resolved["bin_size_m"][0]
    point_size = resolved["scatter_point_size"][0]
    alpha = resolved["scatter_alpha"][0]
    cmap = resolved["height_cmap"][0]
    viz_dpi = resolved["viz_dpi"][0]

    cut3r_dir = args.cut3r_run_dir
    orb_dirs = args.orb_run_dir

    if args.output_dir:
        out_dir = args.output_dir
    else:
        orb_names = "_".join(d.name for d in orb_dirs)
        out_dir = Path("comparisons") / f"{cut3r_dir.name}_vs_{orb_names}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_cfg = None
    if args.video_name:
        gt_cfg = cfg.get(f"groundtruth.{args.video_name}")

    print(f"\nLoading CUT3R run: {cut3r_dir}")
    cut3r_poses = load_poses(cut3r_dir)
    cut3r_pts = load_ply(cut3r_dir / "map" / "merged.ply")
    cut3r_metrics = trajectory_metrics(cut3r_poses, min_disp, smooth_window)
    print(f"  {cut3r_metrics['n_frames']} frames, "
          f"path={cut3r_metrics['path_length_m']:.3f}m, "
          f"map points={0 if cut3r_pts is None else len(cut3r_pts):,}")

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    orb_entries = []
    plot_runs = [("CUT3R", "tab:blue", cut3r_poses)]
    aligned_runs_for_plot = []
    floorplan_orb_pts = []

    for idx, orb_dir in enumerate(orb_dirs):
        color = colors[idx % len(colors)]
        label = orb_dir.name
        print(f"\nLoading ORB-SLAM3 run: {orb_dir}")
        orb_poses = load_poses(orb_dir)
        orb_pts = load_ply(orb_dir / "map" / "merged.ply")
        run_log = load_orb_run_log(orb_dir)

        metrics = trajectory_metrics(orb_poses, min_disp, smooth_window)
        health = orb_tracking_health(run_log)
        gt_err = ground_truth_error(metrics, gt_cfg)

        print(f"  {metrics['n_frames']} frames, keyframes={health['n_keyframes']}, "
              f"path={metrics['path_length_m']:.3f}m, "
              f"map points={0 if orb_pts is None else len(orb_pts):,}")

        cut3r_centers = cut3r_poses[:, :3, 3]
        orb_centers = orb_poses[:, :3, 3]
        aligned_free, info_free = align_orb_to_cut3r(cut3r_centers, orb_centers, with_scale=True)
        aligned_fixed, info_fixed = align_orb_to_cut3r(cut3r_centers, orb_centers, with_scale=False)
        print(f"  Umeyama scale-free:  scale={info_free['scale']:.4f}  "
              f"rot={info_free['rotation_angle_deg']:.2f}deg  rms={info_free['rms_residual_m']:.4f}m")
        print(f"  Umeyama scale-fixed: rot={info_fixed['rotation_angle_deg']:.2f}deg  "
              f"rms={info_fixed['rms_residual_m']:.4f}m")

        orb_entries.append({
            "label": label, "run_dir": str(orb_dir), "metrics": metrics, "health": health,
            "gt_error": gt_err, "umeyama_scale_free": info_free, "umeyama_scale_fixed": info_fixed,
        })
        plot_runs.append((label, color, orb_poses))
        aligned_runs_for_plot.append((label, color, (aligned_free, aligned_fixed), (info_free, info_fixed)))
        floorplan_orb_pts.append((label, orb_pts))

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison_trajectories(out_dir, plot_runs, dpi=viz_dpi)
    plot_comparison_aligned(out_dir, cut3r_poses[:, :3, 3], aligned_runs_for_plot, dpi=viz_dpi)
    plot_comparison_floorplans(out_dir, cut3r_pts, floorplan_orb_pts, bin_size, point_size,
                                alpha, cmap, axis_limits_shared, dpi=viz_dpi)

    # ── CSV + report ─────────────────────────────────────────────────────────
    csv_rows = [{"run": "CUT3R", **cut3r_metrics}]
    for entry in orb_entries:
        row = {"run": entry["label"], **entry["metrics"], **entry["health"], **entry["gt_error"],
               "umeyama_scale_free_scale": entry["umeyama_scale_free"]["scale"],
               "umeyama_scale_free_rms_m": entry["umeyama_scale_free"]["rms_residual_m"],
               "umeyama_scale_free_rotation_deg": entry["umeyama_scale_free"]["rotation_angle_deg"],
               "umeyama_scale_fixed_rms_m": entry["umeyama_scale_fixed"]["rms_residual_m"],
               "umeyama_scale_fixed_rotation_deg": entry["umeyama_scale_fixed"]["rotation_angle_deg"]}
        csv_rows.append(row)
    write_metrics_csv(out_dir, csv_rows)
    write_report(out_dir, cut3r_metrics, orb_entries)

    pcfg.snapshot(resolved, out_dir)
    print(f"\nComparison complete. Output dir: {out_dir}")


if __name__ == "__main__":
    main()

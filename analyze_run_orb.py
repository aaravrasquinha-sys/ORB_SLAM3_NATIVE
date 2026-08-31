"""
analyze_run_orb.py — Standalone diagnostics for a completed+converted ORB-SLAM3
run, adapted from CUT3R's analyze_run.py. Keeps identical filenames, figure
layouts, axis labels, and DPI conventions where a CUT3R analogue exists, so
comparison across pipelines stays visually apples-to-apples.

Reads ONLY already-saved output files from a --run_dir:
  pred_poses.npy      (N,4,4) camera-to-world (written by convert_trajectory.py)
  orb_run_log.json    per-frame tracking state/keypoints/keyframe flag,
                       gravity/scale metadata (written by convert_trajectory.py)
  tracking_log.csv    optional raw mono_video output (frame_index, timestamp,
                       tracking_state, n_keypoints, n_map_matches, is_keyframe,
                       n_map_points_total) — orb_run_log.json's per_frame list
                       is preferred when both exist since it also carries
                       height_above_floor_m.

Does NOT import torch, does NOT touch the GPU — same constraint as CUT3R's
analyze_run.py, so this is safe to run on any machine with the run outputs.

ORB-SLAM3 has no dense per-pixel depth/confidence (no pred_pts3d.npy /
pred_conf.npy analogue) — CUT3R's confidence_over_time.png and inspect_frame
have no equivalent here and are intentionally NOT reproduced. Everything else
degrades gracefully to blank/NaN columns rather than fabricating values.

USAGE:
    python analyze_run_orb.py --run_dir results/IMG_1205_orb_run1
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline_config as pcfg
from trajectory_coverage import compute_coverage

pcfg.register_keys(
    "diagnostics.heading_min_displacement_m",
    "diagnostics.heading_smooth_window",
    "diagnostics.dpi",
    "diagnostics.window_disagreement_size",
    "camera.height_m",
    "extraction.target_fps",
)

# Tracking states considered a "loss" band for shading (matches
# state_name() in orbslam_ext/mono_video.cc).
LOSS_STATES = {"NOT_INITIALIZED", "RECENTLY_LOST", "LOST", "SYSTEM_NOT_READY", "NO_IMAGES_YET"}


# ─────────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path):
    poses_path = run_dir / "pred_poses.npy"
    if not poses_path.exists():
        raise SystemExit(f"[ERROR] {poses_path} not found. Run convert_trajectory.py first.")
    poses = np.load(poses_path)  # (N,4,4) camera-to-world

    run_log = None
    log_path = run_dir / "orb_run_log.json"
    if log_path.exists():
        run_log = json.loads(log_path.read_text())

    tracking_rows = None
    csv_path = run_dir / "tracking_log.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            tracking_rows = list(csv.DictReader(f))

    return poses, run_log, tracking_rows


def per_frame_records(run_log, tracking_rows, n):
    """Unify orb_run_log.json's per_frame list (preferred) and/or
    tracking_log.csv into one list of dicts indexed by frame, filling gaps
    with blanks rather than fabricating data."""
    records = [{"tracking_state": "", "n_keypoints": None, "n_map_matches": None,
                "is_keyframe": False, "height_above_floor_m": None} for _ in range(n)]

    if tracking_rows:
        for row in tracking_rows:
            i = int(row.get("frame_index", -1))
            if 0 <= i < n:
                records[i]["tracking_state"] = row.get("tracking_state", "")
                records[i]["n_keypoints"] = int(row.get("n_keypoints", 0))
                records[i]["n_map_matches"] = int(row.get("n_map_matches", 0))
                records[i]["is_keyframe"] = row.get("is_keyframe", "0") in ("1", "True", "true")

    if run_log and "per_frame" in run_log:
        for entry in run_log["per_frame"]:
            i = entry.get("frame")
            if i is None or not (0 <= i < n):
                continue
            records[i]["tracking_state"] = entry.get("tracking_state", records[i]["tracking_state"])
            if entry.get("n_keypoints") is not None:
                records[i]["n_keypoints"] = entry["n_keypoints"]
            if entry.get("n_map_matches") is not None:
                records[i]["n_map_matches"] = entry["n_map_matches"]
            if entry.get("is_keyframe") is not None:
                records[i]["is_keyframe"] = entry["is_keyframe"]
            records[i]["height_above_floor_m"] = entry.get("height_above_floor_m")

    return records


def rotation_to_euler(R):
    """Identical to CUT3R's analyze_run.py — both pipelines share axis convention."""
    yaw = np.degrees(np.arctan2(R[0, 2], R[2, 2]))
    pitch = np.degrees(np.arcsin(-np.clip(R[1, 2], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[1, 0], R[1, 1]))
    return roll, pitch, yaw


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


# ─────────────────────────────────────────────────────────────────────────
def write_trajectory_table(out_dir, poses, records):
    n = len(poses)
    path = out_dir / "trajectory_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z", "roll", "pitch", "yaw",
                    "tracking_state", "n_keypoints", "n_map_matches", "is_keyframe"])
        for i in range(n):
            t = poses[i][:3, 3]
            roll, pitch, yaw = rotation_to_euler(poses[i][:3, :3])
            r = records[i]
            w.writerow([i, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}",
                        f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}",
                        r["tracking_state"],
                        "" if r["n_keypoints"] is None else r["n_keypoints"],
                        "" if r["n_map_matches"] is None else r["n_map_matches"],
                        int(bool(r["is_keyframe"]))])
    print(f"  Saved {path}")


def plot_camera_trajectory(out_dir, poses, dpi=120):
    """Same three-panel bird's-eye/side/front layout used elsewhere in this
    project (compare_runs.py's comparison_trajectories.png) for visual
    consistency, single-run version."""
    centers = poses[:, :3, 3]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (axes[0], 0, 2, "X (m)", "Z (m)", "Bird's-eye (X, Z)"),
        (axes[1], 2, 1, "Z (m)", "Y (m)", "Side (Z, Y)"),
        (axes[2], 0, 1, "X (m)", "Y (m)", "Front (X, Y)"),
    ]
    for ax, ia, ib, xlabel, ylabel, title in panels:
        ax.plot(centers[:, ia], centers[:, ib], color="tab:orange", linewidth=1.5)
        ax.scatter([centers[0, ia]], [centers[0, ib]], color="green", marker="o", s=60,
                   edgecolors="black", zorder=5, label="start")
        ax.scatter([centers[-1, ia]], [centers[-1, ib]], color="red", marker="s", s=60,
                   edgecolors="black", zorder=5, label="end")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        if ib == 1:
            ax.invert_yaxis()
        ax.legend(fontsize=8)
    fig.suptitle("ORB-SLAM3 camera trajectory")
    fig.tight_layout()
    path = out_dir / "camera_trajectory.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_heading_vs_yaw(out_dir, poses, min_disp=0.02, smooth_window=5, dpi=120, target_fps=None):
    """Reused essentially unchanged from CUT3R's analyze_run.py — needs only
    poses, which both pipelines produce in the same convention."""
    n = len(poses)
    centers = np.stack([poses[i][:3, 3] for i in range(n)], axis=0)
    yaw = np.array([rotation_to_euler(poses[i][:3, :3])[2] for i in range(n)])

    if n >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        sm_x = np.convolve(centers[:, 0], kernel, mode="same")
        sm_z = np.convolve(centers[:, 2], kernel, mode="same")
    else:
        sm_x, sm_z = centers[:, 0], centers[:, 2]

    heading = np.full(n, np.nan)
    for i in range(1, n):
        dx, dz = sm_x[i] - sm_x[i - 1], sm_z[i] - sm_z[i - 1]
        v = np.array([dx, dz])
        if np.linalg.norm(v) >= min_disp:
            heading[i] = np.degrees(np.arctan2(v[0], v[1]))  # +X, +Z convention

    valid = ~np.isnan(heading)
    idx = np.arange(n)
    diff = np.full(n, np.nan)
    diff[valid] = wrap180(heading[valid] - yaw[valid])
    excluded = int((~valid).sum())

    starved_warning = None
    if n > 0 and excluded / n > 0.5:
        fps_note = f"extraction.target_fps={target_fps}" if target_fps is not None else "extraction.target_fps unknown"
        starved_warning = (
            f"[WARNING] heading_vs_yaw.png is starved of data: {excluded}/{n} frames "
            f"({100.0 * excluded / n:.1f}%) excluded by diagnostics.heading_min_displacement_m="
            f"{min_disp} m ({fps_note}). Do not trust heading_vs_yaw.png with this few valid points."
        )
        print(starved_warning)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(idx, yaw, label="Reported yaw", color="tab:blue")
    axes[0].plot(idx[valid], heading[valid], label="Motion heading (smoothed)", color="tab:orange")
    axes[0].set_ylabel("degrees")
    axes[0].set_title("Motion heading vs reported yaw")
    axes[0].legend()

    axes[1].plot(idx[valid], diff[valid], color="tab:red")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("heading - yaw (deg, wrapped to [-180,180])")

    fig.tight_layout()
    path = out_dir / "heading_vs_yaw.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")
    return diff, valid, excluded, starved_warning


def plot_tracking_health(out_dir, records, dpi=120):
    """Analogue of CUT3R's confidence_over_time.png: n_keypoints and
    n_map_matches per frame, with LOST/RECENTLY_LOST/NOT_INITIALIZED bands
    shaded (structural equivalent of confidence dropping to zero)."""
    n = len(records)
    idx = np.arange(n)
    keypoints = np.array([r["n_keypoints"] if r["n_keypoints"] is not None else np.nan for r in records])
    matches = np.array([r["n_map_matches"] if r["n_map_matches"] is not None else np.nan for r in records])
    states = [r["tracking_state"] for r in records]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, keypoints, label="n_keypoints", color="tab:blue")
    ax.plot(idx, matches, label="n_map_matches", color="tab:green")

    # Shade contiguous loss-state bands
    in_band = False
    band_start = None
    for i in range(n + 1):
        bad = i < n and states[i] in LOSS_STATES
        if bad and not in_band:
            in_band, band_start = True, i
        elif not bad and in_band:
            ax.axvspan(band_start - 0.5, i - 0.5, color="red", alpha=0.15)
            in_band = False

    ax.set_xlabel("frame")
    ax.set_ylabel("count")
    ax.set_title("Tracking health over time (red = NOT_INITIALIZED/RECENTLY_LOST/LOST)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "tracking_health.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_scale_stability(out_dir, records, poses, camera_height_m, dpi=120):
    """Structural analogue of CUT3R's scale_stability.png / model_height plot:
    camera height above the fitted floor per frame vs the configured
    camera.height_m, with non-OK tracking frames shaded."""
    n = len(records)
    idx = np.arange(n)
    heights = np.array([r["height_above_floor_m"] if r["height_above_floor_m"] is not None
                        else np.nan for r in records])
    states = [r["tracking_state"] for r in records]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, heights, color="tab:purple", label="camera height above floor")
    if camera_height_m is not None:
        ax.axhline(camera_height_m, color="gray", linestyle="--",
                   label=f"camera.height_m = {camera_height_m:.3f}")

    in_band = False
    band_start = None
    for i in range(n + 1):
        bad = i < n and states[i] in LOSS_STATES
        if bad and not in_band:
            in_band, band_start = True, i
        elif not bad and in_band:
            ax.axvspan(band_start - 0.5, i - 0.5, color="red", alpha=0.15)
            in_band = False

    ax.set_xlabel("frame")
    ax.set_ylabel("height (m)")
    ax.set_title("Scale stability: camera height above fitted floor")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "scale_stability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


def longest_state_gap(records, states_of_interest):
    n = len(records)
    best = None
    cur_start = None
    for i in range(n + 1):
        bad = i < n and records[i]["tracking_state"] in states_of_interest
        if bad and cur_start is None:
            cur_start = i
        elif not bad and cur_start is not None:
            length = i - cur_start
            if best is None or length > best[2]:
                best = (cur_start, i - 1, length)
            cur_start = None
    return best


def windowed_heading_disagreement(diff, valid, window):
    """Rolling-window mean |heading - yaw| disagreement, in addition to
    CUT3R's single 'largest yaw jump' stat. A single-frame jump metric can
    miss a failure that is smeared across many frames (e.g. a slow drift
    building up over ~100 frames rather than one sharp jump) — this
    surfaces the worst sustained window instead of only the worst instant.
    Returns (worst_window_start, worst_window_mean_abs_deg) or None."""
    n = len(diff)
    if n < window:
        return None
    abs_diff = np.abs(diff)
    best_start, best_mean = None, -1.0
    for start in range(0, n - window + 1):
        seg_valid = valid[start:start + window]
        if seg_valid.sum() < max(2, window // 2):
            continue  # too few defined headings in this window to trust it
        seg_mean = float(np.nanmean(abs_diff[start:start + window][seg_valid]))
        if seg_mean > best_mean:
            best_mean, best_start = seg_mean, start
    if best_start is None:
        return None
    return best_start, best_mean


def write_debug_summary(out_dir, poses, records, diff, valid, excluded, run_log,
                        min_disp, window, starved_warning=None, tracking_rows=None,
                        target_fps=None):
    n = len(poses)
    lines = []
    lines.append("=" * 65)
    lines.append("  ANALYZE_RUN_ORB - AUTOMATED DEBUG SUMMARY")
    lines.append("=" * 65)
    lines.append("")

    if tracking_rows is not None:
        lines.append("-" * 65)
        lines.append("TRAJECTORY COVERAGE")
        lines.append("-" * 65)
        traj_path = out_dir.parent / "CameraTrajectory.txt"
        cov = compute_coverage(tracking_rows, target_fps, traj_path)

        lines.append(f"Frames in tracking_log.csv: {cov['total_frames']}"
                      + (f"  ({cov['video_duration_s']:.2f} s @ {target_fps} fps)"
                         if cov['video_duration_s'] else ""))
        for state, count in sorted(cov["state_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {state or '(blank)'}: {count}")
        if cov["ok_indices"]:
            lines.append(f"First OK frame: {cov['first_ok_frame']}   Last OK frame: {cov['last_ok_frame']}")
        else:
            lines.append("No OK frames found.")

        lines.append(f"Poses in CameraTrajectory.txt (pred_poses.npy): {n}")

        if cov["first_pose_ts"] is not None:
            lines.append(f"First pose timestamp: {cov['first_pose_ts']:.3f} s   "
                         f"Last pose timestamp: {cov['last_pose_ts']:.3f} s")

        coverage = cov["coverage"]
        if coverage is not None:
            lines.append(f"Coverage fraction (tracked duration / total video duration): "
                         f"{coverage * 100:.1f}%")
            if coverage < 0.95:
                lines.append(f"  *** COVERAGE FLAG: only {coverage * 100:.1f}% of the video is "
                             f"covered by the trajectory. Any whole-route ground-truth value "
                             f"anchored onto this trajectory will be inflated by roughly "
                             f"1/coverage = {1.0 / coverage:.3f}x. ***")
        lines.append("")

    if starved_warning:
        lines.append("*** " + starved_warning + " ***")
        lines.append("")

    gap = longest_state_gap(records, LOSS_STATES)
    if gap:
        lines.append(f"Longest tracking-loss gap: frames [{gap[0]}, {gap[1]}] ({gap[2]} frames)")
    else:
        lines.append("No tracking-loss gaps found (always OK/OK_KLT).")

    yaws = np.array([rotation_to_euler(poses[i][:3, :3])[2] for i in range(n)])
    yaw_jumps = np.abs(np.diff(yaws))
    if len(yaw_jumps):
        j = int(np.argmax(yaw_jumps))
        lines.append(f"Largest single-frame yaw jump: frame {j} -> {j+1}  "
                     f"({yaws[j]:.2f} deg -> {yaws[j+1]:.2f} deg, "
                     f"delta={yaws[j+1]-yaws[j]:+.2f} deg)")

    lines.append(f"Motion-heading: excluded {excluded}/{n} frames with smoothed "
                 f"displacement < {min_disp} m (heading undefined there).")

    if valid.any():
        d = diff[valid]
        peak_idx = np.where(valid)[0][int(np.argmax(np.abs(d)))]
        lines.append(f"Motion-heading vs yaw disagreement (all valid frames): "
                     f"mean |diff| = {np.mean(np.abs(d)):.2f} deg, "
                     f"max |diff| = {np.max(np.abs(d)):.2f} deg at frame {peak_idx}")

        ok_mask = np.array([r["tracking_state"] in ("OK", "OK_KLT") for r in records])
        trust = valid & ok_mask
        if trust.any():
            dt = diff[trust]
            peak_idx_t = np.where(trust)[0][int(np.argmax(np.abs(dt)))]
            lines.append(f"Motion-heading vs yaw disagreement (tracking-OK subset): "
                         f"mean |diff| = {np.mean(np.abs(dt)):.2f} deg, "
                         f"max |diff| = {np.max(np.abs(dt)):.2f} deg at frame {peak_idx_t} "
                         f"({int(trust.sum())}/{n} frames)")

    lines.append("")
    lines.append(f"Windowed heading-vs-yaw degradation (window={window} frames):")
    lines.append("  Rationale: a single largest-jump stat can undersell a failure that is")
    lines.append("  spread across many frames (e.g. a slow drift smeared over ~100+ frames")
    lines.append("  rather than one sharp jump). This reports the worst sustained window's")
    lines.append("  mean |heading - yaw| instead of only the worst single instant.")
    wd = windowed_heading_disagreement(diff, valid, window)
    if wd:
        start, mean_abs = wd
        lines.append(f"  Worst window: frames [{start}, {start + window - 1}]  "
                     f"mean |diff| = {mean_abs:.2f} deg")
    else:
        lines.append("  Not enough valid heading frames to compute a windowed metric.")

    lines.append("")
    d3 = poses[:, :3, 3]
    path_len = float(np.sum(np.linalg.norm(np.diff(d3, axis=0), axis=1)))
    net_disp = float(np.linalg.norm(d3[-1] - d3[0]))
    y_span = float(d3[:, 1].max() - d3[:, 1].min())
    lines.append(f"Total path length (recomputed from pred_poses.npy): {path_len:.4f} m")
    lines.append(f"Net displacement  (recomputed from pred_poses.npy): {net_disp:.4f} m")
    lines.append(f"Y span            (recomputed from pred_poses.npy): {y_span:.4f} m")

    n_keyframes = sum(1 for r in records if r["is_keyframe"])
    lines.append(f"Keyframe count (from per-frame log): {n_keyframes}")
    if run_log is not None:
        lines.append(f"Keyframe count (from orb_run_log.json n_keyframes): {run_log.get('n_keyframes')}")
        scale = run_log.get("scale", {})
        lines.append(f"Scale mode: {scale.get('mode')}   factor: {scale.get('factor')}   "
                     f"is_metric: {scale.get('is_metric')}")
        if not scale.get("is_metric", False):
            lines.append("  *** NON-METRIC RUN — distances are in arbitrary units ***")
        if scale.get("is_estimate", False):
            lines.append("  *** METRIC RESULT IS ESTIMATE-DERIVED, NOT A GROUND MEASUREMENT ***")
            lines.append("  (scale.mode=ground_truth_segment: the groundtruth distance_m for "
                         "this segment was itself an estimate — see groundtruth.<video>.notes "
                         "in config.yaml. Treat distances as approximate, not measured.)")
        gravity = run_log.get("gravity", {})
        lines.append(f"Gravity alignment: mode={gravity.get('mode')}  "
                     f"residual_m={gravity.get('residual_m')}")
    else:
        lines.append("[WARN] orb_run_log.json not found — scale/gravity metadata unavailable.")

    # Map point count: only available if map/merged.ply was produced by convert_trajectory.py
    map_ply = out_dir.parent / "map" / "merged.ply"
    if map_ply.exists():
        n_pts = sum(1 for _ in map_ply.read_text(errors="ignore").splitlines())
        lines.append(f"map/merged.ply present (line count incl. header: {n_pts})")
    else:
        lines.append("map/merged.ply not found — map point count unavailable.")

    lines.append("")
    lines.append("Note: ORB-SLAM3 has no dense per-pixel depth/confidence output "
                 "(no pred_pts3d.npy/pred_conf.npy analogue). Downstream tools "
                 "degrade gracefully when those files are absent, matching how "
                 "CUT3R's own analyze_run.py already handles missing optional inputs.")
    lines.append("")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print("\n" + report)
    path = out_dir / "debug_summary.txt"
    path.write_text(report)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Standalone ORB-SLAM3 run diagnostics (no GPU needed).")
    ap.add_argument("--run_dir", required=True, help="Path to a run_dir already processed by convert_trajectory.py")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = pcfg.load_config(args.config)
    resolved = {}
    resolved["heading_min_displacement_m"] = pcfg.resolve(
        None, "diagnostics.heading_min_displacement_m", 0.02, cfg, validate=pcfg.non_negative)
    resolved["heading_smooth_window"] = pcfg.resolve(
        None, "diagnostics.heading_smooth_window", 5, cfg, validate=pcfg.positive)
    resolved["dpi"] = pcfg.resolve(None, "diagnostics.dpi", 120, cfg, validate=pcfg.positive)
    resolved["window_disagreement_size"] = pcfg.resolve(
        None, "diagnostics.window_disagreement_size", 20, cfg, validate=pcfg.positive)
    resolved["camera_height_m"] = pcfg.resolve(None, "camera.height_m", 0.57, cfg, validate=pcfg.positive)
    resolved["target_fps"] = pcfg.resolve(None, "extraction.target_fps", 30.0, cfg, validate=pcfg.positive)

    min_disp = resolved["heading_min_displacement_m"][0]
    smooth_window = resolved["heading_smooth_window"][0]
    dpi = resolved["dpi"][0]
    health_window = resolved["window_disagreement_size"][0]
    camera_height_m = resolved["camera_height_m"][0]
    target_fps = resolved["target_fps"][0]

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pcfg.snapshot(resolved, out_dir)

    print(f"Loading run from {run_dir} ...")
    poses, run_log, tracking_rows = load_run(run_dir)
    n = len(poses)
    print(f"  {n} frames loaded.")

    records = per_frame_records(run_log, tracking_rows, n)

    write_trajectory_table(out_dir, poses, records)
    plot_camera_trajectory(out_dir, poses, dpi=dpi)
    diff, valid, excluded, starved_warning = plot_heading_vs_yaw(
        out_dir, poses, min_disp=min_disp, smooth_window=smooth_window, dpi=dpi,
        target_fps=target_fps)
    plot_tracking_health(out_dir, records, dpi=dpi)
    plot_scale_stability(out_dir, records, poses, camera_height_m, dpi=dpi)
    write_debug_summary(out_dir, poses, records, diff, valid, excluded, run_log,
                        min_disp=min_disp, window=health_window,
                        starved_warning=starved_warning, tracking_rows=tracking_rows,
                        target_fps=target_fps)

    print(f"\nDone. All outputs saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

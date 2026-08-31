"""
trajectory_coverage.py — Shared helper for computing how much of a video's
duration a tracked trajectory actually covers.

Used by analyze_run_orb.py (reporting) and convert_trajectory.py (the
whole-route ground-truth scale guard) so the coverage arithmetic is defined
in exactly one place.
"""

from pathlib import Path


def compute_coverage(tracking_rows, target_fps, cam_traj_path):
    """Compute trajectory coverage from tracking_log.csv rows and the saved
    camera trajectory file.

    Args:
        tracking_rows: list of dict-like rows from tracking_log.csv (must have
            'tracking_state' and 'frame_index' keys).
        target_fps: extraction.target_fps used to derive total video duration
            from frame count. May be None (video_duration_s / coverage will
            then be None).
        cam_traj_path: Path to CameraTrajectory.txt (timestamp-first TUM
            format lines) used to derive tracked duration.

    Returns a dict with:
        total_frames, video_duration_s, state_counts (dict),
        ok_indices (list[int]), first_ok_frame, last_ok_frame,
        first_pose_ts, last_pose_ts, tracked_duration_s,
        coverage (float in [0,1] or None if not computable).
    """
    total_frames = len(tracking_rows)
    video_duration_s = total_frames / target_fps if target_fps else None

    state_counts = {}
    for row in tracking_rows:
        s = row.get("tracking_state", "")
        state_counts[s] = state_counts.get(s, 0) + 1

    ok_indices = [int(row["frame_index"]) for row in tracking_rows
                  if row.get("tracking_state") in ("OK", "OK_KLT")]
    first_ok_frame = ok_indices[0] if ok_indices else None
    last_ok_frame = ok_indices[-1] if ok_indices else None

    first_pose_ts = last_pose_ts = None
    cam_traj_path = Path(cam_traj_path)
    if cam_traj_path.exists():
        timestamps = []
        for line in cam_traj_path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            timestamps.append(float(line.split()[0]))
        if timestamps:
            first_pose_ts, last_pose_ts = timestamps[0], timestamps[-1]

    tracked_duration_s = None
    coverage = None
    if first_pose_ts is not None and last_pose_ts is not None:
        tracked_duration_s = last_pose_ts - first_pose_ts
        if video_duration_s:
            coverage = tracked_duration_s / video_duration_s

    return {
        "total_frames": total_frames,
        "video_duration_s": video_duration_s,
        "state_counts": state_counts,
        "ok_indices": ok_indices,
        "first_ok_frame": first_ok_frame,
        "last_ok_frame": last_ok_frame,
        "first_pose_ts": first_pose_ts,
        "last_pose_ts": last_pose_ts,
        "tracked_duration_s": tracked_duration_s,
        "coverage": coverage,
    }

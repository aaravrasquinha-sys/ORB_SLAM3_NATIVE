"""
calibrate_camera.py — Checkerboard camera calibration from a video.

Samples N well-spread frames from a calibration video, detects checkerboard
corners, and runs cv2.calibrateCamera to recover intrinsics + distortion.

USAGE:
    python calibrate_camera.py --video calib_iphone14pro_4k.mov
    python calibrate_camera.py --video calib_iphone14pro_4k.mov --report
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import json

import pipeline_config as pcfg

# ── Keys this script reads from config ──────────────────────────────────────
pcfg.register_keys(
    "calibration.n_frames",
    "calibration.board_cols",
    "calibration.board_rows",
    "calibration.square_size_mm",
    "calibration.max_rms_px",
    "calibration.min_frames",
    "calibration.output_json",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Checkerboard camera calibration from a video.")
    ap.add_argument("--video", required=True,
                    help="Path to calibration video file.")
    ap.add_argument("--config", default=None,
                    help="Path to config.yaml (default: repo root config.yaml)")
    ap.add_argument("--output_json", default=None,
                    help="Path to write calibration JSON (overrides config).")
    ap.add_argument("--n_frames", type=int, default=None,
                    help="Number of frames to sample from the video (overrides config).")
    ap.add_argument("--board_cols", type=int, default=None,
                    help="Inner corners along the board's long axis (overrides config).")
    ap.add_argument("--board_rows", type=int, default=None,
                    help="Inner corners along the board's short axis (overrides config).")
    ap.add_argument("--square_size_mm", type=float, default=None,
                    help="Physical checkerboard square size in mm (overrides config).")
    ap.add_argument("--max_rms_px", type=float, default=None,
                    help="Maximum acceptable reprojection RMS error in px (overrides config).")
    ap.add_argument("--min_frames", type=int, default=None,
                    help="Minimum number of boards that must be detected (overrides config).")
    ap.add_argument("--report", action="store_true",
                    help="Save an undistortion before/after PNG next to the output JSON.")
    return ap.parse_args()


def load_config_and_resolve(args):
    cfg = pcfg.load_config(args.config)
    resolved = {}

    resolved["n_frames"] = pcfg.resolve(
        args.n_frames, "calibration.n_frames", 40, cfg, validate=pcfg.positive)
    resolved["board_cols"] = pcfg.resolve(
        args.board_cols, "calibration.board_cols", 9, cfg, validate=pcfg.positive)
    resolved["board_rows"] = pcfg.resolve(
        args.board_rows, "calibration.board_rows", 6, cfg, validate=pcfg.positive)
    resolved["square_size_mm"] = pcfg.resolve(
        args.square_size_mm, "calibration.square_size_mm", 25.0, cfg, validate=pcfg.positive)
    resolved["max_rms_px"] = pcfg.resolve(
        args.max_rms_px, "calibration.max_rms_px", 1.0, cfg, validate=pcfg.positive)
    resolved["min_frames"] = pcfg.resolve(
        args.min_frames, "calibration.min_frames", 15, cfg, validate=pcfg.positive)

    default_output = str(pcfg.REPO_ROOT / "calibration" / "realsense_d435i.json")
    resolved["output_json"] = pcfg.resolve(
        args.output_json, "calibration.output_json", default_output, cfg)

    return cfg, resolved


def sample_frame_indices(total_frames: int, n_frames: int):
    """Evenly-spaced frame indices spanning the whole video."""
    if total_frames <= n_frames:
        return list(range(total_frames))
    return sorted(set(np.linspace(0, total_frames - 1, n_frames).astype(int).tolist()))


def _find_board(frame_gray: np.ndarray, pattern_size: tuple):
    """Try SB detector first, fall back to the classic detector. Returns
    (found, corners)."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(frame_gray, pattern_size)
        if found:
            return found, corners

    return cv2.findChessboardCorners(
        frame_gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)


def detect_corners(frame_gray: np.ndarray, board_cols: int, board_rows: int):
    """Detect checkerboard inner corners, refine to sub-pixel accuracy.

    Downscales large (e.g. 4K) frames before detection since the classic/SB
    detectors are unreliable at full resolution, then rescales the corner
    coordinates back up before sub-pixel refinement on the full-res frame.
    Tries both (cols, rows) and (rows, cols) board orientations.

    Returns (corners, (cols_used, rows_used)), or (None, None) if the board
    wasn't found in either orientation.
    """
    h, w = frame_gray.shape[:2]
    max_dim = max(h, w)
    scale = 1600.0 / max_dim if max_dim > 1600 else 1.0

    if scale < 1.0:
        small = cv2.resize(frame_gray, (int(round(w * scale)), int(round(h * scale))),
                            interpolation=cv2.INTER_AREA)
    else:
        small = frame_gray

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for cols_try, rows_try in ((board_cols, board_rows), (board_rows, board_cols)):
        found, corners = _find_board(small, (cols_try, rows_try))
        if not found:
            continue

        if scale < 1.0:
            corners = corners / scale

        corners = corners.astype(np.float32)
        corners = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
        return corners, (cols_try, rows_try)

    return None, None


def build_object_points(board_cols: int, board_rows: int, square_size_mm: float):
    """Object points for one board view, in the board's own coordinate frame
    (Z=0 plane), scaled by the physical square size (meters)."""
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= (square_size_mm / 1000.0)  # mm -> m
    return objp


def calibrate(obj_points, img_points, image_size):
    """Run cv2.calibrateCamera and return (rms, camera_matrix, dist_coeffs)."""
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    return rms, camera_matrix, dist_coeffs


def main():
    args = parse_args()
    cfg, resolved = load_config_and_resolve(args)

    n_frames       = resolved["n_frames"][0]
    board_cols     = resolved["board_cols"][0]
    board_rows     = resolved["board_rows"][0]
    square_size_mm = resolved["square_size_mm"][0]
    max_rms_px     = resolved["max_rms_px"][0]
    min_frames     = resolved["min_frames"][0]
    output_json    = Path(resolved["output_json"][0]).expanduser()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path.name}  {vid_w}x{vid_h}  {total_frames} frames")

    indices = sample_frame_indices(total_frames, n_frames)
    print(f"Sampling {len(indices)} frames, detecting {board_cols}x{board_rows} "
          f"inner-corner checkerboard...")

    locked_orientation = None
    objp_template = None

    obj_points = []
    img_points = []
    detected_frames = []
    sample_gray = None
    sample_corners = None

    debug_dir = output_json.parent / "debug_frames"
    n_debug_dumped = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  frame {idx:6d}: [WARN] could not read")
            continue

        if n_debug_dumped < 5:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(debug_path), frame)
            n_debug_dumped += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, orientation = detect_corners(gray, board_cols, board_rows)

        if corners is None:
            print(f"  frame {idx:6d}: NOT detected")
            continue

        if locked_orientation is None:
            locked_orientation = orientation
            objp_template = build_object_points(
                locked_orientation[0], locked_orientation[1], square_size_mm)
            if locked_orientation != (board_cols, board_rows):
                print(f"  [NOTE] locking in swapped board orientation "
                      f"{locked_orientation[0]}x{locked_orientation[1]} "
                      f"(config specifies {board_cols}x{board_rows})")

        if orientation != locked_orientation:
            print(f"  frame {idx:6d}: SKIPPED (orientation "
                  f"{orientation[0]}x{orientation[1]} != locked "
                  f"{locked_orientation[0]}x{locked_orientation[1]})")
            continue

        print(f"  frame {idx:6d}: detected")
        obj_points.append(objp_template)
        img_points.append(corners)
        detected_frames.append(idx)

        if sample_gray is None:
            sample_gray = gray
            sample_corners = corners

    cap.release()

    n_detected = len(detected_frames)
    print(f"\nDetected board in {n_detected}/{len(indices)} sampled frames.")

    if n_detected < min_frames:
        sys.exit(f"[ERROR] Only {n_detected} boards detected, need at least "
                  f"{min_frames} (calibration.min_frames). Aborting.")

    image_size = (vid_w, vid_h)
    rms, camera_matrix, dist_coeffs = calibrate(obj_points, img_points, image_size)

    print(f"\nRMS reprojection error: {rms:.4f} px "
          f"(max allowed: {max_rms_px} px)")

    if rms > max_rms_px:
        sys.exit(f"[ERROR] RMS reprojection error {rms:.4f} px exceeds "
                  f"calibration.max_rms_px ({max_rms_px} px). Aborting — "
                  f"recalibrate with better board coverage/lighting.")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    d = dist_coeffs.ravel().tolist()
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    k3 = d[4] if len(d) > 4 else 0.0

    result = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
        "width": vid_w, "height": vid_h,
        "rms_reprojection_error_px": rms,
        "n_frames_used": n_detected,
        "frames_used": detected_frames,
        "board_cols": board_cols,
        "board_rows": board_rows,
        "square_size_mm": square_size_mm,
        "source_video": str(video_path),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    print(f"\nCalibration written -> {output_json}")
    print(f"  fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
    print(f"  k1={k1:.6f} k2={k2:.6f} p1={p1:.6f} p2={p2:.6f} k3={k3:.6f}")

    if args.report:
        write_undistort_report(sample_gray, camera_matrix, dist_coeffs,
                               output_json.parent)

    pcfg.snapshot(resolved, output_json.parent)


def write_undistort_report(gray: np.ndarray, camera_matrix, dist_coeffs, out_dir: Path):
    """Save a side-by-side before/after undistortion PNG for a sanity check."""
    if gray is None:
        print("  [WARN] --report requested but no sample frame available; skipping.")
        return

    undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)
    combined = np.hstack([gray, undistorted])
    report_path = out_dir / "undistort_report.png"
    cv2.imwrite(str(report_path), combined)
    print(f"  Undistortion report (before | after) -> {report_path}")


if __name__ == "__main__":
    main()

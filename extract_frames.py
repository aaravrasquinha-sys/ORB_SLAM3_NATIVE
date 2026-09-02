"""
extract_frames.py — Extract frames from an iPhone video for ORB-SLAM3.

Writes numbered image files + timestamps.txt into the Linux scratch dir
(fast I/O), not /mnt/c. Scales calibration intrinsics to match the output
resolution — this is the single most common source of silent scale errors.

NOTE on frame rate:
  ORB-SLAM3 needs 30 fps (default here), NOT CUT3R's 10 fps. ORB-SLAM3's
  feature tracker requires small inter-frame motion; at 10 fps the camera
  baseline between frames is too large and tracking fails immediately.
  See CONFIG_GUIDE.md for details.

USAGE:
    python extract_frames.py --video IMG_1205.mov
    python extract_frames.py --video IMG_1205.mov --output_dir ~/orb_scratch/IMG_1205
    python extract_frames.py --video IMG_1205.mov --target_fps 15 --resize_longest_side 640
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import pipeline_config as pcfg

# ── Keys this script reads from config ──────────────────────────────────────
pcfg.register_keys(
    "shared.scratch_dir",
    "extraction.target_fps",
    "extraction.resize_longest_side",
    "extraction.output_format",
    "extraction.jpeg_quality",
    "calibration.output_json",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract frames from video for ORB-SLAM3 (30 fps, resized, "
                    "with timestamp file).")
    ap.add_argument("--video", required=True,
                    help="Path to input video file.")
    ap.add_argument("--config", default=None,
                    help="Path to config.yaml (default: repo root config.yaml)")
    ap.add_argument("--output_dir", default=None,
                    help="Directory to write frames + timestamps.txt. "
                         "Default: <scratch_dir>/<video_stem>/")
    ap.add_argument("--target_fps", type=float, default=None,
                    help="Target extraction frame rate (overrides config)")
    ap.add_argument("--resize_longest_side", type=int, default=None,
                    help="Resize longest side to this many pixels (overrides config)")
    ap.add_argument("--output_format", default=None,
                    help="'png' or 'jpg' (overrides config)")
    ap.add_argument("--jpeg_quality", type=int, default=None,
                    help="JPEG quality 0-100 (overrides config)")
    ap.add_argument("--calib_json", default=None,
                    help="Path to calibration JSON (overrides config). If provided, "
                         "scaled intrinsics are written alongside the frames.")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract even if output_dir already has frames.")
    return ap.parse_args()


def load_config_and_resolve(args):
    cfg = pcfg.load_config(args.config)
    resolved = {}

    resolved["target_fps"] = pcfg.resolve(
        args.target_fps, "extraction.target_fps", 30.0, cfg, validate=pcfg.non_negative)
    resolved["resize_longest_side"] = pcfg.resolve(
        args.resize_longest_side, "extraction.resize_longest_side", 960, cfg,
        validate=pcfg.positive)
    resolved["output_format"] = pcfg.resolve(
        args.output_format, "extraction.output_format", "png", cfg)
    resolved["jpeg_quality"] = pcfg.resolve(
        args.jpeg_quality, "extraction.jpeg_quality", 95, cfg,
        validate=pcfg.in_range(0, 100))

    default_calib = str(pcfg.REPO_ROOT / "calibration" / "realsense_d435i.json")
    resolved["calib_json"] = pcfg.resolve(
        args.calib_json, "calibration.output_json", default_calib, cfg)

    scratch = pcfg.resolve(None, "shared.scratch_dir", "~/orb_scratch", cfg)[0]
    resolved["scratch_dir"] = (None, "default")  # store raw for reference
    resolved["scratch_dir"] = pcfg.resolve(None, "shared.scratch_dir", "~/orb_scratch", cfg)

    return cfg, resolved, scratch


def scale_intrinsics(calib: dict, orig_w: int, orig_h: int,
                     new_w: int, new_h: int) -> dict:
    """Scale camera intrinsics from calibration resolution to extraction resolution.

    fx, fy, cx, cy scale linearly with the resize factor.
    Distortion coefficients (k1, k2, p1, p2, k3) do NOT change — they are
    defined in the normalised image plane and are resolution-independent.

    Getting this wrong silently corrupts every downstream result.
    """
    # Compute per-axis scale factors
    sx = new_w / orig_w
    sy = new_h / orig_h

    # Sanity check: aspect ratios must match (we only do uniform resize)
    if abs(sx - sy) > 0.01:
        print(f"  [WARN] Non-uniform resize detected (sx={sx:.4f}, sy={sy:.4f}). "
              f"Intrinsics scaling assumes uniform resize — results may be slightly off.")

    scaled = dict(calib)  # copy all fields (including distortion which stays unchanged)
    scaled["fx"] = calib["fx"] * sx
    scaled["fy"] = calib["fy"] * sy
    scaled["cx"] = calib["cx"] * sx
    scaled["cy"] = calib["cy"] * sy
    scaled["width"]  = new_w
    scaled["height"] = new_h
    scaled["calib_source_width"]  = orig_w
    scaled["calib_source_height"] = orig_h
    scaled["scale_factor_x"] = sx
    scaled["scale_factor_y"] = sy
    return scaled


def resize_frame(frame: np.ndarray, longest_side: int):
    """Resize frame so longest side = longest_side, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) == longest_side:
        return frame, w, h
    if w >= h:
        new_w = longest_side
        new_h = int(round(h * longest_side / w))
    else:
        new_h = longest_side
        new_w = int(round(w * longest_side / h))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w, new_h


def main():
    args = parse_args()
    cfg, resolved, scratch_base = load_config_and_resolve(args)

    target_fps     = resolved["target_fps"][0]
    resize_ls      = resolved["resize_longest_side"][0]
    out_fmt        = resolved["output_format"][0].lower().strip(".")
    jpeg_quality   = resolved["jpeg_quality"][0]
    calib_json_path = Path(resolved["calib_json"][0]).expanduser()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    # ── Determine output directory ───────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
    else:
        scratch = Path(scratch_base).expanduser()
        out_dir = scratch / video_path.stem

    # ── Check if already done ────────────────────────────────────────────────
    ts_file = out_dir / "timestamps.txt"
    if ts_file.exists() and not args.force:
        existing = list(out_dir.glob(f"*.{out_fmt}"))
        if existing:
            print(f"[SKIP] {out_dir} already contains {len(existing)} .{out_fmt} frames "
                  f"and timestamps.txt. Use --force to re-extract.")
            # Still print scaled intrinsics info if calib exists
            _maybe_print_scaled_calib(calib_json_path, out_dir)
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_fps <= 0:
        print("[WARN] Video FPS unknown; assuming 30.0")
        video_fps = 30.0

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution:   {vid_w}x{vid_h}")
    print(f"  FPS (native): {video_fps:.3f}")
    print(f"  Frames:       {total_frames}")

    # ── Compute frame step ───────────────────────────────────────────────────
    if target_fps <= 0 or target_fps >= video_fps:
        frame_step = 1
        effective_fps = video_fps
    else:
        frame_step = int(round(video_fps / target_fps))
        effective_fps = video_fps / frame_step

    print(f"  Target FPS:   {target_fps}")
    print(f"  Frame step:   {frame_step} (effective {effective_fps:.3f} fps)")
    print(f"  Resize to:    longest side = {resize_ls} px")
    print(f"  Output:       {out_dir}")

    # ── Extract frames ───────────────────────────────────────────────────────
    timestamps = []
    frame_paths = []
    actual_w, actual_h = None, None

    frame_idx = 0
    out_idx   = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            timestamp = frame_idx / video_fps  # seconds from start

            resized, new_w, new_h = resize_frame(frame, resize_ls)
            if actual_w is None:
                actual_w, actual_h = new_w, new_h
                print(f"  Actual output size: {actual_w}x{actual_h}")

            # Write frame
            fname = f"{out_idx:06d}.{out_fmt}"
            fpath = out_dir / fname
            if out_fmt in ("jpg", "jpeg"):
                cv2.imwrite(str(fpath), resized,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(fpath), resized)

            timestamps.append(timestamp)
            frame_paths.append(fname)
            out_idx += 1

            if out_idx % 100 == 0:
                print(f"  Extracted {out_idx} frames ...")

        frame_idx += 1

    cap.release()
    print(f"  Extracted {out_idx} frames total.")

    # ── Write timestamps.txt (TUM format: one float per line) ────────────────
    with open(ts_file, "w") as f:
        for ts in timestamps:
            f.write(f"{ts:.9f}\n")
    print(f"  timestamps.txt written ({len(timestamps)} entries).")

    # ── Write frame_list.txt (relative paths, for ORB-SLAM3 loader) ──────────
    fl_file = out_dir / "frame_list.txt"
    with open(fl_file, "w") as f:
        for fname in frame_paths:
            f.write(fname + "\n")
    print(f"  frame_list.txt written.")

    # ── Scale and write calibration ──────────────────────────────────────────
    if calib_json_path.exists():
        calib = json.loads(calib_json_path.read_text())
        orig_w = calib.get("width", vid_w)
        orig_h = calib.get("height", vid_h)
        scaled = scale_intrinsics(calib, orig_w, orig_h, actual_w, actual_h)

        scaled_out = out_dir / "scaled_intrinsics.json"
        scaled_out.write_text(json.dumps(scaled, indent=2))
        print(f"\n  Intrinsics scaled from {orig_w}x{orig_h} -> {actual_w}x{actual_h}:")
        print(f"    sx={scaled['scale_factor_x']:.6f}  "
              f"sy={scaled['scale_factor_y']:.6f}")
        print(f"    fx:  {calib['fx']:.4f} -> {scaled['fx']:.4f}")
        print(f"    fy:  {calib['fy']:.4f} -> {scaled['fy']:.4f}")
        print(f"    cx:  {calib['cx']:.4f} -> {scaled['cx']:.4f}")
        print(f"    cy:  {calib['cy']:.4f} -> {scaled['cy']:.4f}")
        print(f"    k1:  {scaled['k1']:.6f}  (unchanged)")
        print(f"    k2:  {scaled['k2']:.6f}  (unchanged)")
        print(f"    p1:  {scaled['p1']:.6f}  (unchanged)")
        print(f"    p2:  {scaled['p2']:.6f}  (unchanged)")
        print(f"  Scaled intrinsics -> {scaled_out}")
    else:
        print(f"\n  [WARN] Calibration JSON not found at {calib_json_path}. "
              f"Run calibrate_camera.py first. No scaled_intrinsics.json written.")
        scaled = None

    # ── Write extraction metadata ─────────────────────────────────────────────
    meta = {
        "video": str(video_path),
        "video_fps": video_fps,
        "video_width": vid_w,
        "video_height": vid_h,
        "frame_step": frame_step,
        "effective_fps": effective_fps,
        "target_fps": target_fps,
        "n_frames": out_idx,
        "output_width": actual_w,
        "output_height": actual_h,
        "output_format": out_fmt,
        "output_dir": str(out_dir),
    }
    meta_path = out_dir / "extraction_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    pcfg.snapshot(resolved, out_dir)
    print(f"\nExtraction complete. Output directory: {out_dir}")


def _maybe_print_scaled_calib(calib_json_path: Path, out_dir: Path):
    scaled_out = out_dir / "scaled_intrinsics.json"
    if scaled_out.exists():
        s = json.loads(scaled_out.read_text())
        print(f"  Scaled intrinsics already present ({s['width']}x{s['height']}): "
              f"fx={s['fx']:.2f} fy={s['fy']:.2f} cx={s['cx']:.2f} cy={s['cy']:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
capture_realsense_frames.py — Live-capture frames from an Intel RealSense
D435I for ORB-SLAM3, with the stereo (L/R infrared) module and motion module
(IMU) also enabled on the same device/pipeline.

This REPLACES extract_frames.py's job when the frame source is a live camera
instead of a pre-recorded video: the COLOR output layout it writes (numbered
frames, timestamps.txt, frame_list.txt, scaled_intrinsics.json,
extraction_meta.json) is byte-for-byte what extract_frames.py used to
produce, so every downstream stage (make_orbslam_settings.py, mono_video,
convert_trajectory.py, ...) works completely unchanged — this project is
still running plain monocular ORB-SLAM3 off the color stream only.

WHY STEREO + MOTION ARE ALSO ENABLED HERE (even though mono ORB-SLAM3 only
consumes color):
  - They're on at the hardware/USB level regardless, so the color pipeline
    needs to declare them in the SAME rs.config it starts with — opening a
    second, separate pipeline for them from another process is what actually
    causes USB bandwidth/resource conflicts, not enabling them together in
    one config.
  - Capturing them now means a future move to ORB-SLAM3 Stereo-Inertial mode
    doesn't require re-recording. If you don't want that data, pass
    --no_stereo / --no_motion, or leave --save_infrared off (default) to
    skip the (large) image writes while still keeping the stream declared.

SYNC NOTE: color + left/right infrared arrive together in one hardware-synced
frameset (frame.is_frameset()). The IMU does NOT — gyro/accel free-run at a
much higher rate than 30fps video (BMI085: gyro up to ~200Hz, accel up to
~250Hz). Polling wait_for_frames() only gives you the IMU sample nearest each
video frame and silently drops the rest. To log every IMU sample, this script
uses a frame CALLBACK (pipeline.start(config, callback)) instead of polling.

USAGE:
    python capture_realsense_frames.py --duration_sec 30
    python capture_realsense_frames.py --duration_sec 20 --output_dir results_run1 --preview
    python capture_realsense_frames.py --duration_sec 20 --write_calib
    python capture_realsense_frames.py --duration_sec 20 --save_infrared --emitter off
    python capture_realsense_frames.py --duration_sec 20 --no_stereo --no_motion   # color only
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline_config as pcfg
from extract_frames import scale_intrinsics, resize_frame

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit(
        "[ERROR] pyrealsense2 not importable. Install with:\n"
        "    pip install pyrealsense2 --break-system-packages"
    )

pcfg.register_keys(
    "shared.scratch_dir",
    "extraction.resize_longest_side",
    "extraction.output_format",
    "extraction.jpeg_quality",
    "calibration.output_json",
    "realsense.width",
    "realsense.height",
    "realsense.fps",
    "realsense.serial_number",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Live-capture color (+ optional stereo/IMU) from a RealSense D435I "
                    "for ORB-SLAM3.")
    ap.add_argument("--config", default=None,
                    help="Path to config.yaml (default: repo root config.yaml)")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write frames + timestamps.txt "
                         "(e.g. <scratch_dir>/<run_name>/).")
    ap.add_argument("--duration_sec", type=float, default=None,
                    help="Stop capturing after this many seconds. Omit + Ctrl-C "
                         "to stop manually.")
    ap.add_argument("--max_frames", type=int, default=None,
                    help="Stop capturing after this many COLOR frames (alternative to "
                         "--duration_sec; whichever limit hits first wins).")

    # Color stream (this is what the mono ORB-SLAM3 pipeline actually consumes)
    ap.add_argument("--width", type=int, default=None,
                    help="Color stream capture width (overrides config; default 1280)")
    ap.add_argument("--height", type=int, default=None,
                    help="Color stream capture height (overrides config; default 720)")
    ap.add_argument("--fps", type=int, default=None,
                    help="Color stream capture fps (overrides config; default 30). "
                         "Set orbslam.camera_fps in config.yaml to match.")
    ap.add_argument("--serial", default=None,
                    help="Device serial, if multiple RealSense cameras are connected "
                         "(overrides config).")
    ap.add_argument("--resize_longest_side", type=int, default=None,
                    help="Resize so the longest side equals this (overrides config)")
    ap.add_argument("--output_format", default=None,
                    help="'png' or 'jpg' (overrides config)")
    ap.add_argument("--jpeg_quality", type=int, default=None,
                    help="JPEG quality 0-100 (overrides config)")

    # Stereo (L/R infrared) module
    ap.add_argument("--no_stereo", action="store_true",
                    help="Don't enable the infrared/stereo module at all.")
    ap.add_argument("--stereo_width", type=int, default=None,
                    help="Infrared stream width (default: same as --width)")
    ap.add_argument("--stereo_height", type=int, default=None,
                    help="Infrared stream height (default: same as --height)")
    ap.add_argument("--stereo_fps", type=int, default=None,
                    help="Infrared stream fps (default: same as --fps)")
    ap.add_argument("--save_infrared", action="store_true",
                    help="Write left/right infrared frames to disk "
                         "(<output_dir>/infrared_left/, infrared_right/). Off by "
                         "default — this roughly triples per-run disk usage. Stream "
                         "stays enabled either way so the sync/bandwidth story is real.")
    ap.add_argument("--emitter", choices=["on", "off"], default="off",
                    help="IR projector state (default: off). Leave off for natural, "
                         "unpatterned stereo images (what a future ORB-SLAM3 Stereo "
                         "mode wants); 'on' gives denser/better depth but a visible "
                         "dot pattern baked into the IR images.")

    # Motion module (IMU)
    ap.add_argument("--no_motion", action="store_true",
                    help="Don't enable the motion module (accel/gyro) at all.")
    ap.add_argument("--save_imu", dest="save_imu", action="store_true", default=True,
                    help="Write every accel/gyro sample to <output_dir>/imu.csv "
                         "(default: on — this is cheap, just numbers).")
    ap.add_argument("--no_save_imu", dest="save_imu", action="store_false",
                    help="Enable the motion module but don't log samples.")

    ap.add_argument("--calib_json", default=None,
                    help="Path to an existing calibration JSON to scale + carry forward "
                         "for the color stream (overrides config). If omitted, "
                         "--write_calib pulls factory intrinsics from the device instead.")
    ap.add_argument("--write_calib", action="store_true",
                    help="Also read factory intrinsics (color + left-infrared + IMU, "
                         "and the extrinsics between them) from the device and write "
                         "them to --calib_json's path.")
    ap.add_argument("--preview", action="store_true",
                    help="Show a live OpenCV preview window of the color stream while "
                         "capturing (requires a display; skip when running headless).")
    ap.add_argument("--force", action="store_true",
                    help="Re-capture even if output_dir already has frames.")
    return ap.parse_args()


def load_config_and_resolve(args):
    cfg = pcfg.load_config(args.config)
    resolved = {}

    resolved["width"]  = pcfg.resolve(args.width,  "realsense.width",  1280, cfg, validate=pcfg.positive)
    resolved["height"] = pcfg.resolve(args.height, "realsense.height", 720,  cfg, validate=pcfg.positive)
    resolved["fps"]    = pcfg.resolve(args.fps,    "realsense.fps",    30,   cfg, validate=pcfg.positive)
    resolved["serial"] = pcfg.resolve(args.serial, "realsense.serial_number", None, cfg)

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

    return cfg, resolved


def read_full_calibration(profile, include_stereo, include_motion):
    """Factory intrinsics/extrinsics for whichever streams are active.
    Color block matches calibrate_camera.py's schema exactly (so
    make_orbslam_settings.py needs no changes); the rest are extra keys
    downstream mono consumers simply ignore, kept for a future
    stereo-inertial settings.yaml."""
    color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    c = color_sp.get_intrinsics()
    result = {
        "fx": c.fx, "fy": c.fy, "cx": c.ppx, "cy": c.ppy,
        "k1": c.coeffs[0], "k2": c.coeffs[1],
        "p1": c.coeffs[2], "p2": c.coeffs[3], "k3": c.coeffs[4],
        "width": c.width, "height": c.height,
        "distortion_model": str(c.model),
        "source": "realsense_factory_intrinsics",
        "rms_reprojection_error_px": None,
        "n_frames_used": None,
        "frames_used": [],
    }

    if include_stereo:
        try:
            left_sp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            li = left_sp.get_intrinsics()
            result["left_infrared"] = {
                "fx": li.fx, "fy": li.fy, "cx": li.ppx, "cy": li.ppy,
                "k1": li.coeffs[0], "k2": li.coeffs[1],
                "p1": li.coeffs[2], "p2": li.coeffs[3], "k3": li.coeffs[4],
                "width": li.width, "height": li.height,
                "distortion_model": str(li.model),
            }
            right_sp = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            ri = right_sp.get_intrinsics()
            result["right_infrared"] = {
                "fx": ri.fx, "fy": ri.fy, "cx": ri.ppx, "cy": ri.ppy,
                "k1": ri.coeffs[0], "k2": ri.coeffs[1],
                "p1": ri.coeffs[2], "p2": ri.coeffs[3], "k3": ri.coeffs[4],
                "width": ri.width, "height": ri.height,
                "distortion_model": str(ri.model),
            }
            ext_color_to_left = color_sp.get_extrinsics_to(left_sp)
            result["extrinsics_color_to_left_infrared"] = {
                "rotation": list(ext_color_to_left.rotation),
                "translation": list(ext_color_to_left.translation),
            }
            ext_left_to_right = left_sp.get_extrinsics_to(right_sp)
            result["stereo_baseline_m"] = ext_left_to_right.translation[0]
        except RuntimeError as e:
            print(f"  [WARN] Could not read stereo calibration: {e}")

    if include_motion:
        try:
            accel_sp = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
            gyro_sp = profile.get_stream(rs.stream.gyro).as_motion_stream_profile()
            ai = accel_sp.get_motion_intrinsics()
            gi = gyro_sp.get_motion_intrinsics()
            result["imu"] = {
                "accel_bias": list(ai.bias),
                "accel_noise_variances": list(ai.noise_variances),
                "gyro_bias": list(gi.bias),
                "gyro_noise_variances": list(gi.noise_variances),
            }
            ext_color_to_accel = color_sp.get_extrinsics_to(accel_sp)
            result["extrinsics_color_to_imu"] = {
                "rotation": list(ext_color_to_accel.rotation),
                "translation": list(ext_color_to_accel.translation),
            }
        except RuntimeError as e:
            print(f"  [WARN] Could not read IMU calibration: {e}")

    return result


def main():
    args = parse_args()
    cfg, resolved = load_config_and_resolve(args)

    width          = resolved["width"][0]
    height         = resolved["height"][0]
    fps            = resolved["fps"][0]
    serial         = resolved["serial"][0]
    resize_ls      = resolved["resize_longest_side"][0]
    out_fmt        = resolved["output_format"][0].lower().strip(".")
    jpeg_quality   = resolved["jpeg_quality"][0]
    calib_json_path = Path(resolved["calib_json"][0]).expanduser()

    enable_stereo = not args.no_stereo
    enable_motion = not args.no_motion
    stereo_w = args.stereo_width or width
    stereo_h = args.stereo_height or height
    stereo_fps = args.stereo_fps or fps

    if args.duration_sec is None and args.max_frames is None:
        sys.exit("[ERROR] Pass --duration_sec and/or --max_frames, or the capture "
                  "will run until Ctrl-C with no logged stop condition.")

    out_dir = Path(args.output_dir).expanduser()
    ts_file = out_dir / "timestamps.txt"
    if ts_file.exists() and not args.force:
        existing = list(out_dir.glob(f"*.{out_fmt}"))
        if existing:
            sys.exit(f"[SKIP] {out_dir} already contains {len(existing)} .{out_fmt} frames "
                      f"and timestamps.txt. Use --force to re-capture.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if enable_stereo and args.save_infrared:
        (out_dir / "infrared_left").mkdir(parents=True, exist_ok=True)
        (out_dir / "infrared_right").mkdir(parents=True, exist_ok=True)

    # ── Build the RealSense config: everything that will actually be on,
    #    declared in ONE config so the pipeline accounts for real USB
    #    bandwidth rather than fighting a second hidden consumer. ────────────
    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(str(serial))
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if enable_stereo:
        rs_config.enable_stream(rs.stream.infrared, 1, stereo_w, stereo_h, rs.format.y8, stereo_fps)
        rs_config.enable_stream(rs.stream.infrared, 2, stereo_w, stereo_h, rs.format.y8, stereo_fps)
    if enable_motion:
        rs_config.enable_stream(rs.stream.accel)
        rs_config.enable_stream(rs.stream.gyro)

    print(f"Starting RealSense D435I:")
    print(f"  color:  {width}x{height} @ {fps}fps")
    if enable_stereo:
        print(f"  stereo: {stereo_w}x{stereo_h} @ {stereo_fps}fps  "
              f"(save_infrared={args.save_infrared}, emitter={args.emitter})")
    if enable_motion:
        print(f"  motion: accel + gyro (save_imu={args.save_imu})")
    if serial:
        print(f"  serial: {serial}")

    # ── Frame callback: color/infrared arrive synced in one frameset; motion
    #    (accel/gyro) arrives asynchronously at a much higher rate. Both are
    #    handled here so no IMU sample is dropped waiting on wait_for_frames(). ──
    color_q = queue.Queue()
    infra_q = queue.Queue()
    imu_rows = []
    imu_lock = threading.Lock()

    def on_frame(frame):
        if frame.is_frameset():
            fs = frame.as_frameset()
            cf = fs.get_color_frame()
            if cf:
                img = np.asanyarray(cf.get_data()).copy()
                color_q.put((cf.get_timestamp(), img))
            if enable_stereo and args.save_infrared:
                ir1 = fs.get_infrared_frame(1)
                ir2 = fs.get_infrared_frame(2)
                if ir1 and ir2:
                    l = np.asanyarray(ir1.get_data()).copy()
                    r = np.asanyarray(ir2.get_data()).copy()
                    infra_q.put((ir1.get_timestamp(), l, r))
        elif frame.is_motion_frame() and args.save_imu:
            mf = frame.as_motion_frame()
            st = mf.get_profile().stream_type()
            d = mf.get_motion_data()
            sensor = "accel" if st == rs.stream.accel else "gyro"
            with imu_lock:
                imu_rows.append((mf.get_timestamp(), sensor, d.x, d.y, d.z))

    try:
        profile = pipeline.start(rs_config, on_frame)
    except RuntimeError as e:
        sys.exit(f"[ERROR] Could not start RealSense pipeline: {e}\n"
                  f"  Check the camera is on a USB 3.x port, not held open by "
                  f"realsense-viewer/another process, and that the requested "
                  f"color/stereo resolution+fps combination is supported together.")

    if enable_stereo:
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
        except RuntimeError as e:
            print(f"  [WARN] Could not set emitter state: {e}")

    if args.write_calib:
        full_calib = read_full_calibration(profile, enable_stereo, enable_motion)
        calib_json_path.parent.mkdir(parents=True, exist_ok=True)
        calib_json_path.write_text(json.dumps(full_calib, indent=2))
        print(f"  Calibration written -> {calib_json_path}"
              + (" (color + stereo + IMU)" if enable_stereo or enable_motion else " (color only)"))

    print(f"  Output: {out_dir}")
    print(f"  Resize to: longest side = {resize_ls} px")
    if args.duration_sec:
        print(f"  Stopping after {args.duration_sec}s")
    if args.max_frames:
        print(f"  Stopping after {args.max_frames} color frames")
    print("  Press Ctrl-C to stop early.\n")

    timestamps_ms = []
    frame_paths = []
    infra_timestamps_ms = []
    actual_w, actual_h = None, None
    out_idx = 0
    infra_idx = 0
    t_start_wall = time.time()

    try:
        while True:
            if args.duration_sec is not None and (time.time() - t_start_wall) >= args.duration_sec:
                break
            if args.max_frames is not None and out_idx >= args.max_frames:
                break

            try:
                ts_ms, img = color_q.get(timeout=0.5)
            except queue.Empty:
                continue

            resized, new_w, new_h = resize_frame(img, resize_ls)
            if actual_w is None:
                actual_w, actual_h = new_w, new_h
                print(f"  Actual output size: {actual_w}x{actual_h}")

            fname = f"{out_idx:06d}.{out_fmt}"
            fpath = out_dir / fname
            if out_fmt in ("jpg", "jpeg"):
                cv2.imwrite(str(fpath), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(fpath), resized)

            timestamps_ms.append(ts_ms)
            frame_paths.append(fname)
            out_idx += 1

            while not infra_q.empty():
                ir_ts, l, r = infra_q.get()
                cv2.imwrite(str(out_dir / "infrared_left" / f"{infra_idx:06d}.png"), l)
                cv2.imwrite(str(out_dir / "infrared_right" / f"{infra_idx:06d}.png"), r)
                infra_timestamps_ms.append(ir_ts)
                infra_idx += 1

            if args.preview:
                cv2.imshow("capture_realsense_frames (q to stop)", resized)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if out_idx % 100 == 0:
                print(f"  Captured {out_idx} color frames "
                      f"({len(imu_rows)} IMU samples so far)...")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by Ctrl-C.")
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"  Captured {out_idx} color frames, {infra_idx} infrared pairs, "
          f"{len(imu_rows)} IMU samples total.")
    if out_idx == 0:
        sys.exit("[ERROR] No color frames captured — nothing written.")

    # ── Write timestamps.txt (TUM format: seconds from start, device clock) ──
    t0 = timestamps_ms[0]
    with open(ts_file, "w") as f:
        for ts in timestamps_ms:
            f.write(f"{(ts - t0) / 1000.0:.9f}\n")
    print(f"  timestamps.txt written ({len(timestamps_ms)} entries, device-clock based).")

    fl_file = out_dir / "frame_list.txt"
    with open(fl_file, "w") as f:
        for fname in frame_paths:
            f.write(fname + "\n")
    print(f"  frame_list.txt written.")

    if enable_stereo and args.save_infrared and infra_idx:
        with open(out_dir / "infrared_timestamps.txt", "w") as f:
            for ts in infra_timestamps_ms:
                f.write(f"{(ts - t0) / 1000.0:.9f}\n")
        print(f"  infrared_timestamps.txt written ({infra_idx} pairs, "
              f"infrared_left/, infrared_right/).")

    if args.save_imu and imu_rows:
        imu_path = out_dir / "imu.csv"
        with open(imu_path, "w") as f:
            f.write("timestamp_s,sensor,x,y,z\n")
            for ts, sensor, x, y, z in sorted(imu_rows, key=lambda r: r[0]):
                f.write(f"{(ts - t0) / 1000.0:.9f},{sensor},{x:.8f},{y:.8f},{z:.8f}\n")
        print(f"  imu.csv written ({len(imu_rows)} accel+gyro samples, device-clock "
              f"timestamps shared with timestamps.txt's t0).")

    # ── Scale and write color calibration (same schema/behavior as extract_frames.py) ──
    if calib_json_path.exists():
        calib = json.loads(calib_json_path.read_text())
        calib_orig_w = calib.get("width", width)
        calib_orig_h = calib.get("height", height)
        scaled = scale_intrinsics(calib, calib_orig_w, calib_orig_h, actual_w, actual_h)

        scaled_out = out_dir / "scaled_intrinsics.json"
        scaled_out.write_text(json.dumps(scaled, indent=2))
        print(f"\n  Intrinsics scaled from {calib_orig_w}x{calib_orig_h} -> {actual_w}x{actual_h}:")
        print(f"    fx: {calib['fx']:.4f} -> {scaled['fx']:.4f}")
        print(f"    fy: {calib['fy']:.4f} -> {scaled['fy']:.4f}")
        print(f"    cx: {calib['cx']:.4f} -> {scaled['cx']:.4f}")
        print(f"    cy: {calib['cy']:.4f} -> {scaled['cy']:.4f}")
        print(f"  Scaled intrinsics -> {scaled_out}")
    else:
        print(f"\n  [WARN] Calibration JSON not found at {calib_json_path}. "
              f"Run get_realsense_calibration.py, or re-run this script with "
              f"--write_calib, before make_orbslam_settings.py.")

    meta = {
        "source": "realsense_d435i_live",
        "requested_color": {"width": width, "height": height, "fps": fps},
        "stereo_enabled": enable_stereo,
        "stereo": {"width": stereo_w, "height": stereo_h, "fps": stereo_fps,
                   "saved_to_disk": bool(args.save_infrared), "emitter": args.emitter}
                  if enable_stereo else None,
        "motion_enabled": enable_motion,
        "motion": {"saved_to_disk": bool(args.save_imu), "n_samples": len(imu_rows)}
                  if enable_motion else None,
        "device_serial": serial,
        "n_color_frames": out_idx,
        "output_width": actual_w,
        "output_height": actual_h,
        "output_format": out_fmt,
        "output_dir": str(out_dir),
    }
    (out_dir / "extraction_meta.json").write_text(json.dumps(meta, indent=2))

    pcfg.snapshot(resolved, out_dir)
    print(f"\nCapture complete. Output directory: {out_dir}")
    print("(mono ORB-SLAM3 pipeline below this point reads only the color frames, "
          "timestamps.txt, and scaled_intrinsics.json — unchanged.)")


if __name__ == "__main__":
    main()

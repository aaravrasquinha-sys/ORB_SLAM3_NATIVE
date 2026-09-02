#!/usr/bin/env python3
"""
get_realsense_calibration.py — Pull factory color-camera intrinsics directly
from a connected Intel RealSense D435I via pyrealsense2.

D400-series cameras ship per-unit factory calibration burned into the
device's EEPROM (queryable in-SDK, no checkerboard needed). This is normally
at least as accurate as an ad-hoc checkerboard run and removes calibration as
a manual step entirely. Writes the SAME JSON schema calibrate_camera.py
produces (fx, fy, cx, cy, k1-k3, p1, p2, width, height, ...), so
make_orbslam_settings.py, extract_frames.py, and capture_realsense_frames.py
all consume it unchanged — nothing downstream needs to know the difference.

Use calibrate_camera.py instead of / in addition to this only if you want an
independent checkerboard-based cross-check (e.g. after a physical bump to
the camera, or to sanity-check the factory numbers).

USAGE:
    python get_realsense_calibration.py
    python get_realsense_calibration.py --width 1280 --height 720 --fps 30
    python get_realsense_calibration.py --serial 138422072842 --output_json calibration/realsense_d435i.json
"""

import argparse
import json
import sys
from pathlib import Path

import pipeline_config as pcfg

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit(
        "[ERROR] pyrealsense2 not importable. Install with:\n"
        "    pip install pyrealsense2 --break-system-packages\n"
        "(already present in this project's tracked environment; if this "
        "fires, the venv/interpreter running this script differs from the "
        "one it was installed into)."
    )

pcfg.register_keys(
    "calibration.output_json",
    "realsense.width",
    "realsense.height",
    "realsense.fps",
    "realsense.serial_number",
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Read factory color-stream intrinsics from a connected RealSense D435I.")
    ap.add_argument("--config", default=None,
                     help="Path to config.yaml (default: repo root config.yaml)")
    ap.add_argument("--width", type=int, default=None,
                     help="Color stream width (overrides config; default 1280)")
    ap.add_argument("--height", type=int, default=None,
                     help="Color stream height (overrides config; default 720)")
    ap.add_argument("--fps", type=int, default=None,
                     help="Color stream fps (overrides config; default 30)")
    ap.add_argument("--serial", default=None,
                     help="Device serial number, if multiple RealSense cameras are "
                          "connected (overrides config).")
    ap.add_argument("--output_json", default=None,
                     help="Path to write calibration JSON (overrides config).")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = pcfg.load_config(args.config)

    width  = pcfg.resolve(args.width,  "realsense.width",  1280, cfg, validate=pcfg.positive)[0]
    height = pcfg.resolve(args.height, "realsense.height", 720,  cfg, validate=pcfg.positive)[0]
    fps    = pcfg.resolve(args.fps,    "realsense.fps",    30,   cfg, validate=pcfg.positive)[0]
    serial = pcfg.resolve(args.serial, "realsense.serial_number", None, cfg)[0]

    default_output = str(pcfg.REPO_ROOT / "calibration" / "realsense_d435i.json")
    output_json = Path(pcfg.resolve(
        args.output_json, "calibration.output_json", default_output, cfg)[0]).expanduser()

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(str(serial))
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print(f"Starting RealSense pipeline: {width}x{height} @ {fps}fps"
          f"{f' (serial {serial})' if serial else ''} ...")
    try:
        profile = pipeline.start(rs_config)
    except RuntimeError as e:
        sys.exit(f"[ERROR] Could not start RealSense pipeline: {e}\n"
                  f"  Check the camera is plugged in (USB 3.x port — this is a USB3.2 "
                  f"device) and not held open by realsense-viewer or another process.")

    try:
        dev = profile.get_device()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()

        name = dev.get_info(rs.camera_info.name)
        found_serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)

        print(f"  Device: {name}  serial={found_serial}  firmware={fw}")
        print(f"  Distortion model: {intr.model}")
        print(f"  fx={intr.fx:.4f} fy={intr.fy:.4f} cx={intr.ppx:.4f} cy={intr.ppy:.4f}")
        print(f"  coeffs (k1,k2,p1,p2,k3): {list(intr.coeffs)}")
    finally:
        pipeline.stop()

    result = {
        "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
        "k1": intr.coeffs[0], "k2": intr.coeffs[1],
        "p1": intr.coeffs[2], "p2": intr.coeffs[3], "k3": intr.coeffs[4],
        "width": intr.width, "height": intr.height,
        "distortion_model": str(intr.model),
        "source": "realsense_factory_intrinsics",
        "device_name": name,
        "device_serial": found_serial,
        "device_firmware": fw,
        # Kept for schema parity with calibrate_camera.py's output (consumers
        # like extract_frames.py don't require these, but some print/debug
        # paths reference them):
        "rms_reprojection_error_px": None,
        "n_frames_used": None,
        "frames_used": [],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    print(f"\nCalibration written -> {output_json}")


if __name__ == "__main__":
    main()

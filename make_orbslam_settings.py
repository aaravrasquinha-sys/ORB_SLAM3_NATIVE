#!/usr/bin/env python3
"""
make_orbslam_settings.py — Generate ORB-SLAM3 settings.yaml for a specific run.

Reads:
  - calibration/iphone14pro_4k_1x.json (or path from config)
  - config.yaml (orbslam.* section)

Writes:
  - <output_dir>/settings.yaml (ORB-SLAM3 format)

Usage:
  python3 make_orbslam_settings.py --calib_json calibration/iphone14pro_4k_1x.json \
    --output_dir results/IMG_1205_orb [--config config.yaml]

CRITICAL: Camera.fps MUST be an integer. ORB-SLAM3 rejects floats.
"""

import json
import argparse
import sys
from pathlib import Path

import pipeline_config as pcfg


def load_calibration(path):
    """Load calibration JSON (fx, fy, cx, cy, k1-k3, width, height)."""
    with open(path, 'r') as f:
        calib = json.load(f)
    required = ['fx', 'fy', 'cx', 'cy', 'width', 'height']
    missing = [k for k in required if k not in calib]
    if missing:
        raise ValueError(f"calibration JSON missing required keys: {missing}")
    return calib


def write_settings_yaml(output_path, camera, orb, viewer):
    """Write ORB-SLAM3 settings in the format its cv::FileStorage parser
    actually requires: a literal '%YAML:1.0' header (not a comment — OpenCV's
    persistence.cpp checks for this exact line) and FLAT dotted keys
    (Camera1.fx: 1400.0), NOT nested YAML mappings. A normal nested YAML file
    parses fine with PyYAML but is rejected by OpenCV's FileStorage with
    'Input file is invalid' — that was the earlier bug. fps is written as a
    bare integer (no decimal point) since ORB-SLAM3 hard-rejects a float fps.
    """
    def fnum(v):
        return f"{v:.8f}" if isinstance(v, float) else str(v)

    lines = [
        "%YAML:1.0",
        "",
        "#--------------------------------------------------------------------",
        "# Camera Parameters — Pinhole model, OpenCV distortion",
        "#--------------------------------------------------------------------",
        "File.version: \"1.0\"",
        "",
        "Camera.type: \"PinHole\"",
        "",
        f"Camera1.fx: {fnum(camera['fx'])}",
        f"Camera1.fy: {fnum(camera['fy'])}",
        f"Camera1.cx: {fnum(camera['cx'])}",
        f"Camera1.cy: {fnum(camera['cy'])}",
        "",
        f"Camera1.k1: {fnum(camera['k1'])}",
        f"Camera1.k2: {fnum(camera['k2'])}",
        f"Camera1.p1: {fnum(camera['p1'])}",
        f"Camera1.p2: {fnum(camera['p2'])}",
        f"Camera1.k3: {fnum(camera['k3'])}",
        "",
        f"Camera.width: {int(camera['width'])}",
        f"Camera.height: {int(camera['height'])}",
        "",
        f"Camera.fps: {int(camera['fps'])}",
        "",
        "# Color order: 0=BGR, 1=RGB",
        f"Camera.RGB: {int(camera['RGB'])}",
        "",
        "#--------------------------------------------------------------------",
        "# ORB Extractor",
        "#--------------------------------------------------------------------",
        f"ORBextractor.nFeatures: {int(orb['nFeatures'])}",
        f"ORBextractor.scaleFactor: {fnum(float(orb['scaleFactor']))}",
        f"ORBextractor.nLevels: {int(orb['nLevels'])}",
        f"ORBextractor.iniThFAST: {int(orb['iniThFAST'])}",
        f"ORBextractor.minThFAST: {int(orb['minThFAST'])}",
        "",
        "#--------------------------------------------------------------------",
        "# Viewer",
        "#--------------------------------------------------------------------",
        f"Viewer.KeyFrameSize: {fnum(float(viewer['KeyFrameSize']))}",
        f"Viewer.KeyFrameLineWidth: {fnum(float(viewer['KeyFrameLineWidth']))}",
        f"Viewer.GraphLineWidth: {fnum(float(viewer['GraphLineWidth']))}",
        f"Viewer.PointSize: {fnum(float(viewer['PointSize']))}",
        f"Viewer.CameraSize: {fnum(float(viewer['CameraSize']))}",
        f"Viewer.CameraLineWidth: {fnum(float(viewer['CameraLineWidth']))}",
        f"Viewer.ViewpointX: {fnum(float(viewer['ViewpointX']))}",
        f"Viewer.ViewpointY: {fnum(float(viewer['ViewpointY']))}",
        f"Viewer.ViewpointZ: {fnum(float(viewer['ViewpointZ']))}",
        f"Viewer.ViewpointF: {fnum(float(viewer['ViewpointF']))}",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate ORB-SLAM3 settings.yaml from calibration + config")
    parser.add_argument("--calib_json", required=True,
                        help="Path to calibration JSON (fx, fy, cx, cy, ...)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (settings.yaml will be written here)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing settings.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    settings_path = output_dir / "settings.yaml"

    if settings_path.exists() and not args.force:
        print(f"[SKIP] {settings_path} already exists (use --force to overwrite)")
        return 0

    # ── Load calibration ────────────────────────────────────────────────────
    print(f"[load] calibration from {args.calib_json}")
    calib = load_calibration(args.calib_json)

    fx = float(calib['fx'])
    fy = float(calib['fy'])
    cx = float(calib['cx'])
    cy = float(calib['cy'])
    width = int(calib['width'])
    height = int(calib['height'])
    k1 = float(calib.get('k1', 0.0))
    k2 = float(calib.get('k2', 0.0))
    k3 = float(calib.get('k3', 0.0))
    p1 = float(calib.get('p1', 0.0))
    p2 = float(calib.get('p2', 0.0))

    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  res={width}x{height}, distortion k1={k1:.4f} k2={k2:.4f} p1={p1:.4f} p2={p2:.4f} k3={k3:.4f}")

    # ── Load config ─────────────────────────────────────────────────────────
    print(f"[load] config from {args.config}")
    cfg = pcfg.load_config(args.config)

    # ORB extractor params
    n_features = cfg.get("orbslam.n_features", 1200)
    scale_factor = cfg.get("orbslam.scale_factor", 1.2)
    n_levels = cfg.get("orbslam.n_levels", 8)
    ini_th_fast = cfg.get("orbslam.ini_th_fast", 20)
    min_th_fast = cfg.get("orbslam.min_th_fast", 7)

    # Camera FPS — MUST BE INTEGER for ORB-SLAM3
    camera_fps_raw = cfg.get("orbslam.camera_fps", 30.0)
    camera_fps = int(round(camera_fps_raw))
    if camera_fps_raw != camera_fps:
        print(f"  [note] camera_fps rounded {camera_fps_raw} → {camera_fps} (ORB-SLAM3 requires integer)")

    rgb_order = cfg.get("orbslam.rgb_order", True)

    # Viewer params (cosmetic)
    kf_size = cfg.get("orbslam.keyframe_size", 0.05)
    kf_line_width = cfg.get("orbslam.keyframe_line_width", 1)
    graph_line_width = cfg.get("orbslam.graph_line_width", 0.9)
    point_size = cfg.get("orbslam.point_size", 2)
    camera_size = cfg.get("orbslam.camera_size", 0.08)
    camera_line_width = cfg.get("orbslam.camera_line_width", 3)

    # ── Build settings ───────────────────────────────────────────────────────
    camera = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
        "width": width, "height": height,
        "fps": camera_fps,  # INTEGER, not float
        "RGB": 1 if rgb_order else 0,
    }
    orb = {
        "nFeatures": n_features,
        "scaleFactor": scale_factor,
        "nLevels": n_levels,
        "iniThFAST": ini_th_fast,
        "minThFAST": min_th_fast,
    }
    viewer = {
        "KeyFrameSize": kf_size,
        "KeyFrameLineWidth": kf_line_width,
        "GraphLineWidth": graph_line_width,
        "PointSize": point_size,
        "CameraSize": camera_size,
        "CameraLineWidth": camera_line_width,
        "ViewpointX": cfg.get("orbslam.viewpoint_x", 0),
        "ViewpointY": cfg.get("orbslam.viewpoint_y", -0.7),
        "ViewpointZ": cfg.get("orbslam.viewpoint_z", -1.8),
        "ViewpointF": cfg.get("orbslam.viewpoint_f", 500),
    }

    # ── Write output ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    write_settings_yaml(settings_path, camera, orb, viewer)

    print(f"[write] {settings_path}")
    print("[OK] Settings generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
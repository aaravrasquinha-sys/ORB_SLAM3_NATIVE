#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end ORB-SLAM3 pipeline orchestrator.

Stages:
  1. extract       — video → numbered frames + timestamps
  2. settings      — calibration JSON + config → ORB-SLAM3 settings.yaml
  3. orbslam       — frames → ORB-SLAM3 output (trajectories, map, tracking log)
  4. convert       — ORB output → CUT3R-compatible (pred_poses.npy, etc.)
  5. analyze       — diagnostics (CSVs, PNGs)

Resumable: skips completed stages unless --force or --force_from is used.

Usage:
  python3 run_pipeline.py --video IMG_1112 --videos_dir /path/to/iphone_videos \
    --output_dir results/IMG_1112_run [--force] [--force_from STAGE]
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json
import time

import pipeline_config as pcfg


class PipelineError(Exception):
    pass


def run_cmd(cmd, description="", timeout_sec=None):
    """Run a shell command, die on error."""
    print(f"\n  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, timeout=timeout_sec, check=True)
        return result.returncode
    except subprocess.TimeoutExpired:
        raise PipelineError(f"{description} timed out after {timeout_sec}s")
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"{description} failed (exit {e.returncode})")


def stage_complete(marker_file):
    """Check if a stage has already completed (by existence of a marker file)."""
    return marker_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Run the full ORB-SLAM3 pipeline (extract → settings → orbslam → convert → analyze)")
    parser.add_argument("--video", required=True,
                        help="Run name. With --source file: looked up as <video>.mov/.mp4 "
                             "in --videos_dir. With --source realsense: just a label used "
                             "for the scratch/output directory name.")
    parser.add_argument("--source", default="file", choices=["file", "realsense"],
                        help="'file' (default): pre-recorded video via extract_frames.py, "
                             "as before. 'realsense': live-capture from a connected D435I "
                             "via capture_realsense_frames.py instead.")
    parser.add_argument("--videos_dir", default=None,
                        help="Directory containing video files. Required with --source file.")
    parser.add_argument("--duration_sec", type=float, default=None,
                        help="With --source realsense: how long to capture. Required with "
                             "--source realsense unless --max_frames is given.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="With --source realsense: stop after this many frames.")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for results (will be created)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--calib_json", default="calibration/realsense_d435i.json",
                        help="Path to calibration JSON (default: calibration/realsense_d435i.json)")
    parser.add_argument("--write_calib", action="store_true",
                        help="With --source realsense: pull factory intrinsics from the "
                             "device during capture instead of requiring --calib_json to "
                             "already exist.")
    parser.add_argument("--save_infrared", action="store_true",
                        help="With --source realsense: also write left/right infrared "
                             "frames to disk (off by default — extra disk usage).")
    parser.add_argument("--no_stereo", action="store_true",
                        help="With --source realsense: don't enable the stereo/infrared "
                             "module at all (by default it's enabled, matching the "
                             "hardware's stereo+RGB+motion-on config).")
    parser.add_argument("--no_motion", action="store_true",
                        help="With --source realsense: don't enable the motion module "
                             "(accel/gyro) at all.")
    parser.add_argument("--force", action="store_true",
                        help="Redo all stages from scratch")
    parser.add_argument("--force_from", default=None,
                        help="Redo from this stage onward (extract|settings|orbslam|convert|analyze)")
    args = parser.parse_args()

    video_name = args.video
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    calib_json_path = Path(args.calib_json).resolve()

    # ── Resolve paths ───────────────────────────────────────────────────────
    video_file = None
    if args.source == "file":
        if not args.videos_dir:
            print("[ERROR] --videos_dir is required with --source file")
            return 1
        videos_dir = Path(args.videos_dir).resolve()
        video_file = videos_dir / f"{video_name}.mov"
        if not video_file.exists():
            print(f"[ERROR] Video not found: {video_file}")
            return 1
    else:  # realsense
        if args.duration_sec is None and args.max_frames is None and not args.force:
            print("[ERROR] --source realsense needs --duration_sec and/or --max_frames "
                  "so the capture stage has a stop condition.")
            return 1
        if not args.write_calib and not calib_json_path.exists():
            print(f"[ERROR] {calib_json_path} not found. Either run "
                  f"get_realsense_calibration.py first, or pass --write_calib to pull "
                  f"factory intrinsics from the device during capture.")
            return 1

    # Load config
    cfg = pcfg.load_config(str(config_path))
    build_dir = Path(cfg.get("shared.orbslam_build_dir", "~/orbslam3_build")).expanduser()
    scratch_dir = Path(cfg.get("shared.scratch_dir", "~/orb_scratch")).expanduser()
    frames_dir = scratch_dir / video_name

    # ORB-SLAM3 paths
    vocab_path = build_dir / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
    mono_video_bin = build_dir / "orbslam_ext_build" / "mono_video"

    print("=" * 70)
    print("  ORB-SLAM3 Pipeline")
    print("=" * 70)
    print(f"Source:     {args.source}" + (f" ({video_file})" if video_file else " (RealSense D435I live capture)"))
    print(f"Output dir: {output_dir}")
    print(f"Frames dir: {frames_dir}")
    print(f"Build dir:  {build_dir}")

    # ── Sanity checks ───────────────────────────────────────────────────────
    if not config_path.exists():
        print(f"\n[WARN] {config_path} not found — using hardcoded defaults")
    if not calib_json_path.exists() and not (args.source == "realsense" and args.write_calib):
        print(f"[ERROR] {calib_json_path} not found. Run calibrate_camera.py / "
              f"get_realsense_calibration.py first, or provide --calib_json "
              f"(or --write_calib with --source realsense).")
        return 1
    if not mono_video_bin.exists():
        print(f"[ERROR] {mono_video_bin} not found. Build ORB-SLAM3 first: bash setup/build_orbslam3.sh")
        return 1
    if not vocab_path.exists():
        print(f"[ERROR] {vocab_path} not found. Did ORB-SLAM3 build succeed?")
        return 1
    camera_height_m = cfg.get("camera.height_m")
    if camera_height_m is None:
        print(f"[ERROR] camera.height_m not set in {config_path}. "
              "A missing camera height must never be guessed — set it explicitly.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    stages = ["extract", "settings", "orbslam", "convert", "analyze"]
    # force_from_idx = the lowest stage index that MUST be redone.
    # A stage at index i must run if i >= force_from_idx (forced) OR its
    # marker file doesn't exist yet. It's only skipped when i < force_from_idx
    # AND the marker already exists. Default (no --force/--force_from) means
    # force_from_idx = len(stages), i.e. nothing is force-redone — every
    # stage's own "does the marker exist" check decides.
    force_from_idx = len(stages)
    if args.force_from:
        if args.force_from in stages:
            force_from_idx = stages.index(args.force_from)
        else:
            print(f"[ERROR] --force_from must be one of: {', '.join(stages)}")
            return 1
    if args.force:
        force_from_idx = 0

    def must_run(stage_name, marker_file):
        """True if this stage needs to (re)run: either it's within the
        forced range, or its output doesn't exist yet."""
        idx = stages.index(stage_name)
        if idx >= force_from_idx:
            return True
        return not stage_complete(marker_file)

    t_start = time.time()

    try:
        # ──────────────────────────────────────────────────────────────────
        # STAGE 1: extract
        # ──────────────────────────────────────────────────────────────────
        ts_file = frames_dir / "timestamps.txt"
        if not must_run("extract", ts_file):
            print(f"\n[SKIP] extract — {ts_file} already exists. Use --force or --force_from extract to redo.")
        elif args.source == "file":
            print("\n" + "=" * 70)
            print("  STAGE: extract (video → frames + timestamps)")
            print("=" * 70)
            run_cmd([
                "python3", "extract_frames.py",
                "--video", str(video_file),
                "--output_dir", str(frames_dir),
                "--config", str(config_path),
            ], description="Frame extraction", timeout_sec=3600)
            if not ts_file.exists():
                raise PipelineError(f"extract did not produce {ts_file}")
        else:  # realsense
            print("\n" + "=" * 70)
            print("  STAGE: extract (RealSense D435I live capture → frames + timestamps)")
            print("=" * 70)
            capture_cmd = [
                "python3", "capture_realsense_frames.py",
                "--output_dir", str(frames_dir),
                "--config", str(config_path),
                "--calib_json", str(calib_json_path),
            ]
            if args.duration_sec is not None:
                capture_cmd += ["--duration_sec", str(args.duration_sec)]
            if args.max_frames is not None:
                capture_cmd += ["--max_frames", str(args.max_frames)]
            if args.write_calib:
                capture_cmd += ["--write_calib"]
            if args.save_infrared:
                capture_cmd += ["--save_infrared"]
            if args.no_stereo:
                capture_cmd += ["--no_stereo"]
            if args.no_motion:
                capture_cmd += ["--no_motion"]
            if args.force:
                capture_cmd += ["--force"]
            run_cmd(capture_cmd, description="RealSense capture",
                     timeout_sec=(args.duration_sec or 0) + 120 if args.duration_sec else 3600)
            if not ts_file.exists():
                raise PipelineError(f"extract did not produce {ts_file}")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 2: settings
        # ──────────────────────────────────────────────────────────────────
        settings_file = output_dir / "settings.yaml"
        if not must_run("settings", settings_file):
            print(f"\n[SKIP] settings — {settings_file} already exists.")
        else:
            print("\n" + "=" * 70)
            print("  STAGE: settings (calibration + config → ORB-SLAM3 yaml)")
            print("=" * 70)
            run_cmd([
                "python3", "make_orbslam_settings.py",
                "--calib_json", str(calib_json_path),
                "--output_dir", str(output_dir),
                "--config", str(config_path),
                "--force",
            ], description="Settings generation", timeout_sec=60)
            if not settings_file.exists():
                raise PipelineError(f"settings did not produce {settings_file}")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 3: orbslam
        # ──────────────────────────────────────────────────────────────────
        traj_file = output_dir / "CameraTrajectory.txt"
        if not must_run("orbslam", traj_file):
            print(f"\n[SKIP] orbslam — {traj_file} already exists.")
        else:
            print("\n" + "=" * 70)
            print("  STAGE: orbslam (mono_video)")
            print("=" * 70)
            run_cmd([
                str(mono_video_bin),
                str(vocab_path),
                str(settings_file),
                str(frames_dir),
                str(ts_file),
                str(output_dir),
            ], description="ORB-SLAM3 tracking", timeout_sec=7200)
            if not traj_file.exists():
                raise PipelineError(f"orbslam did not produce {traj_file}")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 4: convert
        # ──────────────────────────────────────────────────────────────────
        pred_poses_file = output_dir / "pred_poses.npy"
        if not must_run("convert", pred_poses_file):
            print(f"\n[SKIP] convert — {pred_poses_file} already exists.")
        else:
            print("\n" + "=" * 70)
            print("  STAGE: convert (ORB output → CUT3R format)")
            print("=" * 70)
            run_cmd([
                "python3", "convert_trajectory.py",
                "--run_dir", str(output_dir),
                "--video_name", video_name,
                "--config", str(config_path),
                "--camera_height_m", str(camera_height_m),
            ], description="Trajectory conversion", timeout_sec=300)
            if not pred_poses_file.exists():
                raise PipelineError(f"convert did not produce {pred_poses_file}")

        # ──────────────────────────────────────────────────────────────────
        # STAGE 5: analyze
        # ──────────────────────────────────────────────────────────────────
        analysis_dir = output_dir / "analysis"
        summary_file = analysis_dir / "debug_summary.txt"
        if not must_run("analyze", summary_file):
            print(f"\n[SKIP] analyze — {summary_file} already exists.")
        else:
            print("\n" + "=" * 70)
            print("  STAGE: analyze (diagnostics)")
            print("=" * 70)
            run_cmd([
                "python3", "analyze_run_orb.py",
                "--run_dir", str(output_dir),
                "--config", str(config_path),
            ], description="Analysis", timeout_sec=300)
            if not summary_file.exists():
                raise PipelineError(f"analyze did not produce {summary_file}")

    except PipelineError as e:
        t_elapsed = time.time() - t_start
        print(f"\n[ERROR] Stage '{e}' failed (exit 255) after {t_elapsed:.1f}s.")
        return 1

    # ── Success ──────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Wall time: {t_elapsed:.1f}s")
    print(f"Outputs saved in: {output_dir}")
    print("\nKey files:")
    for f in ["CameraTrajectory.txt", "pred_poses.npy", "tracking_log.csv", "run_meta.json"]:
        p = output_dir / f
        if p.exists():
            size = p.stat().st_size
            print(f"  {f:30s} ({size:>10,} bytes)")
    print("\nAnalysis:")
    for f in ["trajectory_table.csv", "camera_trajectory.png", "debug_summary.txt"]:
        p = analysis_dir / f
        if p.exists():
            size = p.stat().st_size
            print(f"  {f:30s} ({size:>10,} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
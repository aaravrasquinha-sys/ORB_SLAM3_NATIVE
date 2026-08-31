# ORB-SLAM3 Validation Pipeline

A self-contained monocular ORB-SLAM3 pipeline for validating iPhone video
captures, structured to be directly comparable against the sibling CUT3R
pipeline (shared `pipeline_config.py` config pattern, matching output
filenames and figure conventions where an analogue exists).

Project layout:

```
ORB-SLAM3/                      (Windows path, this repo — code/config/results)
  config.yaml                   central tuning config
  pipeline_config.py            shared config loader (verbatim from CUT3R)
  calibrate_camera.py           checkerboard camera calibration
  extract_frames.py             video -> numbered frames + timestamps.txt
  make_orbslam_settings.py      calibration + config -> ORB-SLAM3 settings.yaml
  orbslam_ext/mono_video.cc     custom ORB-SLAM3 runner (source)
  convert_trajectory.py         ORB-SLAM3 output -> CUT3R-compatible artifacts
  analyze_run_orb.py            standalone diagnostics (no GPU)
  compare_runs.py               CUT3R vs ORB-SLAM3 comparison
  run_pipeline.py               end-to-end, one video
  run_all.py                    batch over config's video list
  setup/build_orbslam3.sh       builds Pangolin + ORB-SLAM3 (Linux side)
  calibration/, results/, comparisons/

~/orbslam3_build/                (WSL2 Linux home — build artifacts only)
  Pangolin/, ORB_SLAM3/, orbslam_ext_build/mono_video

~/orb_scratch/                   (WSL2 Linux home — extracted frames)
  <video_name>/000000.png, ...
```

**Why the split**: CMake and heavy I/O both suffer badly on `/mnt/c/...`
(spaces in the Windows path break some builds outright, and NTFS-via-9P I/O
is slow). Build artifacts and extracted frames live entirely on the Linux
filesystem; only this repo's code, config, and final results live under the
Windows-visible path so they show up in VS Code.

## 1. WSL2 setup

Requires WSL2 with an Ubuntu distro installed and some free disk (build +
frame extraction easily use several GB per video).

```powershell
wsl --install -d Ubuntu   # if not already installed
wsl -d Ubuntu
```

Inside WSL2, `sudo apt-get update` once so package lists are fresh. The build
script installs its own apt dependencies but needs `sudo`, which requires an
interactive password — if you're driving this through an automated tool that
can't type a password, run the apt-get install line from `setup/build_orbslam3.sh`
manually first.

## 2. Build Pangolin + ORB-SLAM3

```bash
cd "/mnt/c/Users/<you>/Documents/Projects Bundle/DRDO ML-based VO/ORB-SLAM3"
bash setup/build_orbslam3.sh
```

Idempotent — safe to re-run; it skips stages whose artifacts already exist.
Every C++ compile-error patch applied along the way is recorded in
`setup/BUILD_NOTES.md` (file, exact error, exact fix) so the build is
reproducible on a fresh machine. Logs are piped through `tee` to `build.log`
files under `~/orbslam3_build/` so a dropped connection still leaves partial
output to inspect.

If it fails, read the error `setup/build_orbslam3.sh` prints (it fails loudly
rather than silently producing a broken build), fix it, add a patch to
`setup/orbslam3_patches.sh` if the fix should be automated for next time, and
re-run.

## 3. Shoot and run calibration

Shoot a calibration video: a 9x6-inner-corner checkerboard (25mm squares by
default — see `calibration.board_cols/rows/square_size_mm` in config.yaml),
filmed slowly from many angles/distances/tilts, in focus, well lit, filling a
good fraction of the frame. Aim for 30+ seconds so `calibrate_camera.py` has
enough candidate frames to hit `calibration.min_frames`.

```bash
python calibrate_camera.py --video path/to/calibration_video.mov --report
```

Writes `calibration/iphone14pro_4k_1x.json` (fx, fy, cx, cy, k1-k3, RMS
error, frames used) and, with `--report`, a before/after undistortion PNG.
**Fails loudly** if RMS reprojection error exceeds `calibration.max_rms_px`
(default 1.0px) or fewer than `calibration.min_frames` boards were detected —
don't proceed past a rejected calibration.

## 4. Run one video

```bash
python run_pipeline.py --video IMG_1205 --videos_dir /path/to/iphone_videos
```

Stages: extract frames -> generate ORB-SLAM3 settings -> run mono_video ->
convert trajectory to CUT3R-compatible artifacts -> analyze -> visualize.
Resumable — each stage is skipped if its output already exists. Force a
specific stage and everything after it:

```bash
python run_pipeline.py --video IMG_1205 --force_from orbslam
python run_pipeline.py --video IMG_1205 --force     # redo everything
```

Output lands in `results/IMG_1205_orb/` by default.

## 5. Run all videos

```bash
python run_all.py --videos_dir /path/to/iphone_videos
```

Iterates config.yaml's `videos:` list, calling `run_pipeline.py` per video.
**Continues past individual failures** (a crash on one clip doesn't abort the
batch) and prints a final summary table of successes/failures with per-video
timing. Exits non-zero if any video failed, so it's CI-friendly.

## 6. Comparing against a CUT3R run

```bash
python compare_runs.py --cut3r_run_dir /path/to/CUT3R/results/IMG_1205_run1 \
    --orb_run_dir results/IMG_1205_orb --video_name IMG_1205
```

Produces `comparison_trajectories.png` (overlaid raw trajectories),
`comparison_aligned.png` (Umeyama-aligned, scale-free and scale-fixed
variants), `comparison_metrics.csv` + `comparison_report.txt` (path length,
net displacement, tracking health, ground-truth error where available,
Umeyama residual/scale), and `comparison_floorplans.png` (side-by-side maps,
explicitly labeled sparse-vs-dense — see Known Caveats below).

## Output file reference

| File | Written by | Contents |
|---|---|---|
| `timestamps.txt`, `frame_list.txt`, `NNNNNN.png` | extract_frames.py | Extracted frames + TUM-format timestamps |
| `scaled_intrinsics.json` | extract_frames.py | Calibration intrinsics rescaled to extraction resolution |
| `settings.yaml` | make_orbslam_settings.py | ORB-SLAM3 `%YAML:1.0` settings file |
| `KeyFrameTrajectory.txt`, `CameraTrajectory.txt` | mono_video | TUM-format poses (keyframes / every tracked frame) |
| `map_points.ply` | mono_video | Raw sparse map points, world frame, pre-alignment |
| `tracking_log.csv` | mono_video | Per-frame: state, keypoints, map matches, keyframe flag |
| `run_meta.json` | mono_video | Wall-clock time, frame/keyframe counts, loss events |
| `pred_poses.npy` | convert_trajectory.py | (N,4,4) camera-to-world, gravity-aligned, scaled |
| `orb_run_log.json` | convert_trajectory.py | Per-frame tracking data + gravity/scale metadata |
| `map/merged.ply` | convert_trajectory.py | Gravity-aligned, scaled map points |
| `analysis/trajectory_table.csv`, `heading_vs_yaw.png`, `tracking_health.png`, `scale_stability.png`, `debug_summary.txt`, `camera_trajectory.png` | analyze_run_orb.py | Standalone diagnostics |
| `map/floorplan.png` | visualize_map.py | Top-down floor plan (scatter + density) |
| `comparisons/<name>/comparison_*.png/.csv/.txt` | compare_runs.py | Cross-pipeline comparison |
| `resolved_config.yaml` | pipeline_config.py (every script) | Snapshot of every resolved config value + its source tier |

## KNOWN CAVEATS

- **Video stabilization warping frames.** If the source video has in-camera
  or software stabilization applied, frames are geometrically warped in a
  way that doesn't match a real pinhole camera — this silently violates
  ORB-SLAM3's calibration model and can corrupt tracking or scale in ways
  that are hard to detect after the fact. Shoot with stabilization off where
  possible, or treat stabilized-video runs as qualitative only.

- **Rolling shutter vs. ORB-SLAM3's global-shutter assumption.** iPhone
  cameras use a rolling shutter (rows exposed sequentially, not
  simultaneously); ORB-SLAM3's projection model assumes global shutter (one
  instant per frame). Fast pans/rotations will show shear-like distortion
  that isn't modeled, degrading feature tracking and pose accuracy exactly
  during the motions most likely to already be difficult for monocular SLAM.

- **Monocular scale ambiguity and our anchoring approach.** ORB-SLAM3
  monocular output is scale-free by construction. We recover metric scale
  via `scale.mode` in config.yaml — `camera_height` (measure camera height
  above the fitted floor plane, anchor to `camera.height_m`) is preferred
  since it needs no external ground truth; ground-truth-based modes are more
  accurate when available but require manual measurement. **Any run without
  a valid scale anchor is flagged non-metric in its outputs** — don't compare
  non-metric run distances against physical measurements.

- **Sparse-vs-dense map comparison.** ORB-SLAM3 produces a sparse set of
  tracked/triangulated feature points; CUT3R produces a dense per-pixel point
  cloud. `compare_runs.py`'s `comparison_floorplans.png` labels this
  explicitly on the figure and never densifies ORB's map to make the visual
  comparison look closer than it is — point count and density are not
  meaningful metrics to compare directly between the two.

- **30 fps (ORB-SLAM3) vs. CUT3R's 10 fps.** ORB-SLAM3's frame-to-frame
  feature tracker needs small inter-frame motion baselines and fails at
  CUT3R's 10 fps; we extract at 30 fps by default (`extraction.target_fps`).
  This means the two pipelines see different amounts of information per
  second of video — factor this in before drawing conclusions from timing-
  or frame-count-based comparisons.

# ORB-SLAM3 Pipeline Config Guide

`config.yaml` (repo root) tunes the ORB-SLAM3 validation pipeline scripts without
touching code. Loaded by `pipeline_config.py` (verbatim copy of CUT3R's — same file,
same behavior). Precedence per value: **explicit CLI flag > config.yaml > hardcoded
fallback in code**. Missing keys or a missing config file are never fatal — they just
fall back to the in-code default, with a warning. Unrecognized keys (typos) warn and
are ignored. An invalid *value* for a recognized key (e.g. a negative voxel size, or a
mode string not in the allowed set) raises a `ConfigError` and stops the run —
deliberate fail-fast so a bad number doesn't silently produce a bad trajectory.

Every run also writes a `resolved_config.yaml` snapshot into its output directory,
recording the actual value used for every key and which tier it came from
(`cli` / `config` / `default`), so a run is always self-documenting.

## Sections

### `shared`
Read by multiple scripts; must stay consistent across them.

- `random_seed` (int) — RNG seed for gravity-alignment RANSAC (`convert_trajectory.py`)
  and frame sampling (`calibrate_camera.py`). Fixed seed = reproducible runs.
- `scratch_dir` (Linux path) — where extracted frames go. **Must be a Linux-side path**
  (e.g. `~/orb_scratch`), never `/mnt/c/...` — NTFS-via-9P I/O is slow and this dir
  gets read frame-by-frame during ORB-SLAM3 tracking.
- `orbslam_build_dir` (Linux path) — where `setup/build_orbslam3.sh` clones and builds
  Pangolin + ORB-SLAM3. Same "must be Linux-side" rule — CMake breaks on paths with
  spaces, and the Windows project path has them.
- `videos_dir` (path) — folder containing source iPhone video files. Must be set (CLI
  `--videos_dir` or edit here) before `run_pipeline.py`/`run_all.py` can find videos.

### `camera`
- `height_m` (meters) — floor-to-lens height, physical measurement. Metric output
  scales linearly with this — measure it carefully, the same way CUT3R does.

### `calibration` (`calibrate_camera.py`)
- `n_frames` (int) — frames sampled from the calibration video for checkerboard
  detection. More = better coverage, slower.
- `board_cols` / `board_rows` (int) — INNER corners of the checkerboard (squares - 1
  in each direction), not the square count.
- `square_size_mm` (float) — physical square size. Affects only extrinsics scale, not
  intrinsics.
- `max_rms_px` (float) — calibration is **rejected** if reprojection RMS exceeds this.
- `min_frames` (int) — calibration is **rejected** if fewer boards were detected.
- `output_json` (path) — where the calibration result is written; read back by
  `extract_frames.py` (to scale intrinsics) and `make_orbslam_settings.py`.

### `extraction` (`extract_frames.py`)
- `target_fps` (Hz) — **30, not CUT3R's 10.** ORB-SLAM3's frame-to-frame feature
  tracker needs small inter-frame motion; at 10 fps the baseline between frames is too
  large and tracking fails immediately. Don't lower this without understanding why it's
  30 in the first place.
- `resize_longest_side` (px) — 4K is far more than ORB-SLAM3 needs and slows feature
  detection; intrinsics are rescaled automatically to match.
- `output_format` (`png`|`jpg`), `jpeg_quality` (int 0-100, if jpg).

### `orbslam` (`make_orbslam_settings.py`)
All key names below map directly to ORB-SLAM3's own settings-file field names —
verified against the live `UZ-SLAMlab/ORB_SLAM3` `Examples/Monocular/TUM1.yaml`.

- `n_features` (int) — `ORBextractor.nFeatures`. Raise if tracking is lost frequently
  in texture-rich scenes; lower if slow on weak hardware.
- `scale_factor` (float) — `ORBextractor.scaleFactor`, pyramid level scale.
- `n_levels` (int) — `ORBextractor.nLevels`, pyramid level count.
- `ini_th_fast` / `min_th_fast` (int) — `ORBextractor.iniThFAST`/`minThFAST`. Lower
  both if initialization fails in low-texture scenes (see Quick Reference below).
- `camera_fps` (float) — `Camera.fps`. **Must match `extraction.target_fps`** or
  timing (and therefore velocity estimates) will be wrong.
- `rgb_order` (int, 0 or 1) — `Camera.RGB` color order.
- `use_viewer` (bool) — whether `run_pipeline.py` passes `--viewer` to `mono_video`.
  Keep `false` for batch runs (headless); `true` needs WSLg/X11.
- `keyframe_size`, `keyframe_line_width`, `graph_line_width`, `point_size`,
  `camera_size`, `camera_line_width`, `viewpoint_x/y/z/f` — Pangolin viewer geometry,
  only matter when `use_viewer` is true.

### `gravity` (`convert_trajectory.py`)
- `mode` (`camera_plane` | `map_points_ransac` | `none`) — `camera_plane` (default)
  SVD-fits a plane to camera centers and rotates it to world-up; best for a
  fixed-height rig where the camera path itself is roughly planar.
  `map_points_ransac` instead RANSACs a floor plane from the sparse map points,
  mirroring CUT3R's ground-plane approach — use this if the camera path itself isn't
  planar (e.g. handheld with height variation) but the room has a real floor.
- `n_iters`, `dist_thresh_frac`, `min_inlier_frac`, `min_horizontal_dot` — RANSAC
  parameters for `map_points_ransac`, same names/semantics as CUT3R's `ground_plane.*`.

### `scale` (`convert_trajectory.py`)
Monocular ORB-SLAM3 output is scale-free by construction — one of these modes MUST
resolve to a real anchor or the run is flagged **non-metric**.

- `mode` (`camera_height` | `ground_truth_path_length` | `ground_truth_net_displacement`
  | `ground_truth_segment` | `manual_factor`) — `camera_height` (preferred, default)
  measures camera height above the fitted floor in map units and anchors to
  `camera.height_m`, needing no external ground truth. The `ground_truth_*` modes read
  from that video's `groundtruth.<video>` entry; missing/null ground truth for the
  chosen mode falls back to `factor=1.0` **non-metric**, loudly, never silently.
- `manual_factor` (float) — used directly when `mode: manual_factor`.

### `groundtruth.<video_name>`
One entry per video in the `videos:` list. `path_length_m`, `net_displacement_m`
(both meters, null=unknown), `segment` (`{from_frame, to_frame, distance_m}` for a
partial-route measurement, null=unset), `notes` (free text). Fill in whichever fields
match your chosen `scale.mode` — leave the rest null.

### `videos`
Ordered list of video names (no extension) processed by `run_all.py`. Each name must
match a file in `videos_dir`; extensions tried in order `.mov, .mp4, .MP4, .MOV`.

### `pointcloud` (`convert_trajectory.py`)
- `voxel_size_m` (meters, 0=disabled) — downsample voxel size for `map/merged.ply`.
- `outlier_removal_method` (`none`|`statistical`|`radius`) — same semantics as CUT3R's
  `pointcloud.outlier_removal_method`; `statistical`/`radius` use the matching
  `outlier_statistical_*`/`outlier_radius_*` parameters below them.

### `visualization` (`visualize_map.py`, `analyze_run_orb.py`)
Mirrors CUT3R's keys exactly so `visualize_map.py` works unchanged against ORB output:
`bin_size_m`, `scatter_point_size`, `scatter_alpha`, `height_cmap`, `density_cmap`,
`dpi`.

### `diagnostics` (`analyze_run_orb.py`)
- `heading_min_displacement_m`, `heading_smooth_window` — same semantics as CUT3R's
  motion-heading-vs-yaw check.
- `depth_vis_percentile` — kept for config-schema compatibility with CUT3R; unused
  here since ORB-SLAM3 has no dense depth to visualize.
- `dpi` — diagnostic plot resolution.
- `window_disagreement_size` (int, frames) — rolling-window size for the
  windowed/cumulative heading-vs-yaw degradation metric in `debug_summary.txt`. Exists
  because a single largest-jump statistic can undersell a failure that's smeared across
  many frames (slow drift) rather than one sharp discontinuity — same rationale as
  CUT3R's own note about a ~110-frame gradual confidence collapse that a single-jump
  stat alone wouldn't have flagged clearly.

### `compare` (`compare_runs.py`)
- `umeyama_scale_free` (bool) — if true, the Umeyama alignment used to overlay ORB's
  trajectory onto CUT3R's also solves for scale (pure shape comparison); if false,
  scale is fixed at 1 (rigid-only fit). Both variants are always computed and reported
  side by side in `comparison_aligned.png` regardless of this default.
- `axis_limits_shared` (bool) — if true, `comparison_floorplans.png` uses identical
  axis limits across both pipelines' panels so visual density/extent comparisons aren't
  distorted by different auto-scaling.

## Quick Reference — failure scenarios

| Symptom | Likely cause / try |
|---|---|
| Tracking lost immediately (`tracking_state` never leaves `NOT_INITIALIZED`/goes straight to `LOST`) | Usually insufficient parallax in the first few frames (camera not moving enough at the start) or too few features. Lower `orbslam.ini_th_fast`/`min_th_fast` (e.g. 10/5) if the scene is low-texture; check `extraction.resize_longest_side` isn't so small that features are lost; confirm `settings.yaml`'s `Camera.width/height` actually matches the extracted frame resolution (a mismatch here silently breaks the projection model). |
| Map/trajectory is metric-wrong (implausible scale) but tracking looks healthy | Check `orb_run_log.json`'s `scale.is_metric` — if `false`, the run had no valid scale anchor and fell back to `factor=1.0`; fill in `groundtruth.<video>` for your chosen `scale.mode`, or switch to `scale.mode: camera_height` and confirm `camera.height_m` is measured correctly. If `is_metric` is `true` but still wrong, check `gravity.mode`'s residual in `orb_run_log.json`'s `gravity.residual_m` — a bad gravity fit throws off the floor-height measurement `camera_height` mode depends on. |
| Initialization never happens (`NOT_INITIALIZED` for the whole run) | Almost always insufficient camera translation in the opening frames — ORB-SLAM3's monocular initializer needs real parallax, not just rotation. Re-shoot with clear forward/lateral motion in the first 1-2 seconds, or trim the video to start there. |
| `heading_vs_yaw.png` disagreement is small per-frame but `debug_summary.txt`'s windowed metric flags a bad window | This is the point of `diagnostics.window_disagreement_size` — a slow drift smeared over many frames. Cross-check against `tracking_health.png` for the same frame range; sustained low `n_map_matches` there usually explains it. |
| Comparison floorplan looks "worse" for ORB than CUT3R just because it's sparser | Expected — see README's Known Caveats on sparse-vs-dense maps. `comparison_floorplans.png` labels this explicitly; don't read point density as a quality signal between the two pipelines. |
| Config key doesn't seem to be taking effect | Check console output for a `[pipeline_config] WARNING: ... unrecognized key(s)` line (typo) or a `ConfigError` traceback (invalid value for a real key) — both are reported at startup, before any processing begins. |
| Need to reproduce a run exactly | Confirm `shared.random_seed` is unchanged, and diff `resolved_config.yaml` between runs — every value + its source (cli/config/default) is recorded there. |

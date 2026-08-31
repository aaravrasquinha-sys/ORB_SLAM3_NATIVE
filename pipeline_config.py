"""
pipeline_config.py — shared config loader for the ORB-SLAM3 / CUT3R pipelines.

Stdlib + PyYAML only (falls back to a config.json of identical structure if
PyYAML isn't installed). Does NOT import torch, open3d, or matplotlib, so it
stays safe to import from the GPU-free post-processing scripts.

Precedence (highest wins), implemented by resolve():
  1. Explicit CLI argument (user typed it — argparse dest is not None)
  2. config.yaml value (dotted path)
  3. Hardcoded fallback in the calling script

USAGE (per script):
    cfg = load_config(args.config)
    voxel_size = resolve(args.voxel_size, "pointcloud.voxel_size_m", 0.02, cfg,
                          key="voxel_size_m", validate=positive)
    ...
    snapshot(resolved, out_dir)
"""

import json
import sys
import warnings
from pathlib import Path

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

REPO_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_NAME = "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Project-wide key registry.
#
# WHY THIS EXISTS: each script calls register_keys() for only the keys IT reads.
# If the unknown-key warning compared config.yaml against just that per-script
# set, then running extract_frames.py would flag every orbslam.*, gravity.*,
# scale.* and groundtruth.* key as "unrecognized" — dozens of false positives
# per run, which trains you to ignore the warning entirely and hides real typos.
#
# So we seed the registry with every key this project legitimately uses. A key
# in config.yaml that is NOT in this set is a genuine typo or a leftover from a
# different version, and warning about it is meaningful.
#
# When you add a new config key: add it to config.yaml, to the reading script's
# register_keys() call, AND here.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_KEYS = {
    # shared
    "shared.random_seed", "shared.scratch_dir", "shared.orbslam_build_dir",
    "shared.videos_dir",
    # camera
    "camera.height_m",
    # calibration
    "calibration.n_frames", "calibration.board_cols", "calibration.board_rows",
    "calibration.square_size_mm", "calibration.max_rms_px",
    "calibration.min_frames", "calibration.output_json",
    # extraction
    "extraction.target_fps", "extraction.resize_longest_side",
    "extraction.output_format", "extraction.jpeg_quality",
    # orbslam
    "orbslam.n_features", "orbslam.scale_factor", "orbslam.n_levels",
    "orbslam.ini_th_fast", "orbslam.min_th_fast", "orbslam.camera_fps",
    "orbslam.rgb_order", "orbslam.use_viewer",
    "orbslam.keyframe_size", "orbslam.keyframe_line_width",
    "orbslam.graph_line_width", "orbslam.point_size", "orbslam.camera_size",
    "orbslam.camera_line_width", "orbslam.viewpoint_x", "orbslam.viewpoint_y",
    "orbslam.viewpoint_z", "orbslam.viewpoint_f",
    # gravity
    "gravity.mode", "gravity.n_iters", "gravity.dist_thresh_frac",
    "gravity.min_inlier_frac", "gravity.min_horizontal_dot",
    # scale
    "scale.mode", "scale.manual_factor", "scale.min_trajectory_coverage",
    # pointcloud
    "pointcloud.voxel_size_m", "pointcloud.outlier_removal_method",
    "pointcloud.outlier_statistical_nb_neighbors",
    "pointcloud.outlier_statistical_std_ratio",
    "pointcloud.outlier_radius_nb_points", "pointcloud.outlier_radius_m",
    # visualization
    "visualization.bin_size_m", "visualization.density_cmap",
    "visualization.height_cmap", "visualization.scatter_alpha",
    "visualization.scatter_point_size", "visualization.dpi",
    # diagnostics
    "diagnostics.heading_min_displacement_m", "diagnostics.heading_smooth_window",
    "diagnostics.dpi", "diagnostics.window_disagreement_size",
    "diagnostics.depth_vis_percentile",
    # compare
    "compare.axis_limits_shared", "compare.umeyama_scale_free",
    # video list
    "videos",
}

# Per-run registry: seeded with PROJECT_KEYS, extended by register_keys().
_KNOWN_KEYS = set(PROJECT_KEYS)

# groundtruth.<video>.<field> is a user-populated mapping — the video names are
# arbitrary, so we can't enumerate them. Any key under one of these prefixes is
# accepted without warning.
_KNOWN_PREFIXES = ("groundtruth.",)


def register_keys(*dotted_keys):
    """Register dotted-path keys as 'known' so load_config()'s unknown-key
    warning doesn't flag them. Call once per script with every key it reads.

    Keys already in PROJECT_KEYS don't strictly need this, but calling it keeps
    each script self-documenting about what it actually consumes.
    """
    _KNOWN_KEYS.update(dotted_keys)


class ConfigError(Exception):
    """Raised for a nonsensical config value (e.g. negative voxel_size)."""


class Config:
    """Thin wrapper around a nested dict with dotted-path access."""

    def __init__(self, data=None, source=None):
        self._data = data or {}
        self.source = source  # Path the config was loaded from, or None

    def get(self, dotted_key, fallback=None):
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return fallback
            node = node[part]
        return fallback if node is None else node

    def flatten_keys(self):
        """All dotted-path leaf keys actually present in the loaded config."""
        out = []

        def _walk(node, prefix):
            if isinstance(node, dict):
                for k, v in node.items():
                    path = f"{prefix}.{k}" if prefix else k
                    _walk(v, path)
            else:
                out.append(prefix)

        _walk(self._data, "")
        return out


def load_config(path=None):
    """Load config.yaml (or config.json fallback) from `path`, or REPO_ROOT by
    default. A missing file is never fatal — returns an empty Config with one
    warning. Returns a Config instance."""
    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = REPO_ROOT / DEFAULT_CONFIG_NAME

    if not cfg_path.exists():
        print(f"[pipeline_config] WARNING: config file not found at {cfg_path} — "
              f"all values fall back to hardcoded in-code defaults.")
        return Config({}, source=None)

    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json" or not _HAVE_YAML:
        if not _HAVE_YAML and cfg_path.suffix.lower() != ".json":
            print(f"[pipeline_config] WARNING: PyYAML not installed — attempting to "
                  f"parse {cfg_path} as JSON instead. Run 'pip install pyyaml' to use "
                  f"a real config.yaml, or provide a config.json with identical structure.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[pipeline_config] WARNING: failed to parse {cfg_path} as JSON "
                  f"({e}) — falling back to hardcoded in-code defaults.")
            return Config({}, source=None)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            print(f"[pipeline_config] WARNING: failed to parse {cfg_path} as YAML "
                  f"({e}) — falling back to hardcoded in-code defaults.")
            return Config({}, source=None)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        print(f"[pipeline_config] WARNING: {cfg_path} did not parse to a mapping — "
              f"falling back to hardcoded in-code defaults.")
        return Config({}, source=None)

    cfg = Config(data, source=cfg_path)
    _warn_unknown_keys(cfg)
    return cfg


def _is_known(key):
    if key in _KNOWN_KEYS:
        return True
    return any(key.startswith(p) for p in _KNOWN_PREFIXES)


def _warn_unknown_keys(cfg):
    """Warn only about keys that belong to no script in this project — i.e.
    genuine typos. Keys read by *other* scripts are still legitimate, so they
    are not flagged (see PROJECT_KEYS rationale above)."""
    present = set(cfg.flatten_keys())
    unknown = sorted(k for k in present if not _is_known(k))
    if unknown:
        print(f"[pipeline_config] WARNING: {cfg.source} has unrecognized key(s), "
              f"ignoring (typo, or from a different script/version?): "
              f"{', '.join(unknown)}")


def _coerce(value, code_default, key):
    """Coerce `value` to the type of `code_default` where sensible."""
    if code_default is None or value is None:
        return value
    target_type = type(code_default)
    if isinstance(value, target_type):
        return value
    try:
        if target_type is bool:
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "yes", "1", "on"):
                    return True
                if low in ("false", "no", "0", "off"):
                    return False
                raise ValueError(value)
            return bool(value)
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        if target_type is str:
            return str(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"config key '{key}': cannot coerce value {value!r} to {target_type.__name__}")
    return value


def resolve(cli_value, config_key, code_default, cfg, key=None, validate=None):
    """Implements the CLI > config > code-default precedence in one call.

    cli_value:    args.<field> — must be None if the user didn't pass the flag
                  (achieved by setting argparse default=None for config-backed args)
    config_key:   dotted path into config.yaml, e.g. "pointcloud.voxel_size_m"
    code_default: the hardcoded fallback (also defines the expected type)
    cfg:          a Config instance from load_config()
    key:          label used in error messages / snapshot (defaults to config_key)
    validate:     optional callable(value) -> None, raises ConfigError on bad value

    Returns (value, tier) where tier is one of "cli" / "config" / "default".
    """
    label = key or config_key
    if cli_value is not None:
        value, tier = cli_value, "cli"
    else:
        raw = cfg.get(config_key, None)
        if raw is None:
            value, tier = code_default, "default"
        else:
            value, tier = _coerce(raw, code_default, label), "config"

    if validate is not None:
        try:
            validate(value)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"config key '{label}': invalid value {value!r} ({e})")

    return value, tier


def snapshot(resolved_dict, out_dir):
    """Write the fully-resolved parameter set (value + tier per key) to
    out_dir/resolved_config.yaml (or .json if PyYAML isn't available), so a
    completed run is self-documenting.

    resolved_dict: {key: (value, tier)}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: {"value": v, "source": tier} for k, (v, tier) in resolved_dict.items()}

    if _HAVE_YAML:
        out_path = out_dir / "resolved_config.yaml"
        out_path.write_text(yaml.safe_dump(payload, sort_keys=True, default_flow_style=False),
                             encoding="utf-8")
    else:
        out_path = out_dir / "resolved_config.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  Resolved config snapshot saved -> {out_path.name}")
    return out_path


# ── Common validators ────────────────────────────────────────────────────────
def positive(value):
    if value is None:
        return
    if value <= 0:
        raise ConfigError(f"expected a positive value, got {value}")


def non_negative(value):
    if value is None:
        return
    if value < 0:
        raise ConfigError(f"expected a non-negative value, got {value}")


def in_range(lo, hi):
    def _check(value):
        if value is None:
            return
        if not (lo <= value <= hi):
            raise ConfigError(f"expected a value in [{lo}, {hi}], got {value}")
    return _check
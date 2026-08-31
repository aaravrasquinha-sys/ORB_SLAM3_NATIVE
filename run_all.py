"""
run_all.py — Batch-run run_pipeline.py over every video in config.yaml's
`videos:` list, continuing past individual failures so one bad clip doesn't
block the rest of the batch. Prints a final summary table.

USAGE:
    python run_all.py
    python run_all.py --videos_dir /mnt/c/.../iphone_videos
    python run_all.py --force
    python run_all.py --only IMG_1205 IMG_1206
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pipeline_config as pcfg

pcfg.register_keys("shared.videos_dir")


def parse_args():
    ap = argparse.ArgumentParser(description="Batch-run the ORB-SLAM3 pipeline over config's video list.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--videos_dir", default=None, help="Overrides config shared.videos_dir")
    ap.add_argument("--only", nargs="+", default=None,
                     help="Restrict to these video names (must be in config's videos: list).")
    ap.add_argument("--force", action="store_true", help="Passed through to each run_pipeline.py call.")
    ap.add_argument("--force_from", default=None, help="Passed through to each run_pipeline.py call.")
    ap.add_argument("--python", default=sys.executable)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = pcfg.load_config(args.config)

    videos = cfg.get("videos", [])
    if not videos:
        sys.exit("[ERROR] config.yaml has no 'videos:' list (or it's empty). Nothing to run.")

    if args.only:
        unknown = set(args.only) - set(videos)
        if unknown:
            print(f"[WARN] --only names not in config's videos list, running anyway: {sorted(unknown)}")
        videos = args.only

    print(f"Batch run: {len(videos)} video(s): {videos}\n")

    results = []
    for i, video in enumerate(videos, 1):
        print(f"\n{'#' * 70}\n#  [{i}/{len(videos)}] {video}\n{'#' * 70}")
        cmd = [args.python, "run_pipeline.py", "--video", video]
        if args.config:
            cmd += ["--config", args.config]
        if args.videos_dir:
            cmd += ["--videos_dir", args.videos_dir]
        if args.force:
            cmd.append("--force")
        if args.force_from:
            cmd += ["--force_from", args.force_from]

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode == 0:
            results.append((video, "OK", elapsed, None))
            print(f"\n[{video}] OK ({elapsed:.1f}s)")
        else:
            results.append((video, "FAILED", elapsed, f"exit code {result.returncode}"))
            print(f"\n[{video}] FAILED (exit {result.returncode}, {elapsed:.1f}s) — continuing to next video.")

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  BATCH SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Video':<24} {'Status':<10} {'Time (s)':<10} {'Detail'}")
    print("-" * 70)
    n_ok = 0
    for video, status, elapsed, detail in results:
        print(f"{video:<24} {status:<10} {elapsed:<10.1f} {detail or ''}")
        if status == "OK":
            n_ok += 1
    print("-" * 70)
    print(f"{n_ok}/{len(results)} succeeded.")
    print(f"{'=' * 70}\n")

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

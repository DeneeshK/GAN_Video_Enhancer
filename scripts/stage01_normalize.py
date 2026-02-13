#!/usr/bin/env python3
"""
MODULE 01 — Decode + Normalize

Input  : Raw video
Output : Normalized video

Purpose:
    - Decode compressed formats
    - Normalize FPS, pixel format, colorspace
    - Prepare video for ML pipeline

Author: Video Enhancer Pipeline
"""

import argparse
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input" / "raw_videos"
OUTPUT_DIR = PROJECT_ROOT / "output" / "stage_01_normalized"

def check_ffmpeg():
    """Ensure ffmpeg is installed and accessible"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print(" FFmpeg not found. Please install FFmpeg before continuing.")
        sys.exit(1)


def normalize_video(input_video: Path, output_video: Path, fps: int = 30, crf: int = 16):
    """Run FFmpeg normalization pipeline"""

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-map_metadata", "-1",
        "-vf", f"fps={fps},format=rgb24",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video)
    ]

    print("\n Running FFmpeg Normalize:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"\n Module 01 Complete → {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 01 — Decode + Normalize")

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw input video"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to normalized output video"
    )

    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="Target FPS (default: 30)"
    )

    parser.add_argument(
        "--crf",
        default=16,
        type=int,
        help="CRF quality factor (default: 16, lower = higher quality)"
    )

    return parser.parse_args()


#def main():
#    args = parse_args()
#
#   if not args.input.exists():
#        print(f" Input file not found: {args.input}")
#        sys.exit(1)
#
#    check_ffmpeg()
#   normalize_video(args.input, args.output, args.fps, args.crf)

def main():

    check_ffmpeg()

    videos = list(INPUT_DIR.glob("*"))

    if not videos:
        print(f"No input videos found in {INPUT_DIR}")
        sys.exit(1)

    for input_video in videos:
        output_video = OUTPUT_DIR / input_video.name

        print(f"\n Normalizing: {input_video.name}")
        normalize_video(input_video, output_video)



if __name__ == "__main__":
    main()

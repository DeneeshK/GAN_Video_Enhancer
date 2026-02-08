"""
MODULE 03 — Video Stabilization (FFmpeg VidStab)

Input  : Deinterlaced video
Output : Stabilized video

Process:
    1) Motion detection → generates transforms.trf
    2) Motion compensation → stabilized output

"""

import argparse
import subprocess
from pathlib import Path
import sys


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        print("FFmpeg not found. Please install FFmpeg.")
        sys.exit(1)


def detect_motion(input_video: Path, transform_file: Path, shakiness=10, accuracy=15):
    transform_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={transform_file}",
        "-f", "null", "-"
    ]

    print("\n Running Motion Analysis:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"\n Motion analysis complete → {transform_file}")


def apply_stabilization(input_video: Path, transform_file: Path, output_video: Path, smoothing=30, crf=16):
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", f"vidstabtransform=smoothing={smoothing}:input={transform_file},unsharp=5:5:0.8:3:3:0.4",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    print("\n🚀 Applying Stabilization:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"\n Module 03 Complete → {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 03 — Video Stabilization (VidStab)")

    parser.add_argument("--input", required=True, type=Path, help="Input deinterlaced video")
    parser.add_argument("--output", required=True, type=Path, help="Output stabilized video")
    parser.add_argument("--transform", required=True, type=Path, help="Transform file (.trf)")
    parser.add_argument("--shakiness", default=10, type=int, help="Detection shakiness (default: 10)")
    parser.add_argument("--accuracy", default=15, type=int, help="Detection accuracy (default: 15)")
    parser.add_argument("--smoothing", default=30, type=int, help="Motion smoothing (default: 30)")
    parser.add_argument("--crf", default=16, type=int, help="Output quality (default: 16)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f" Input file not found: {args.input}")
        sys.exit(1)

    check_ffmpeg()

    detect_motion(args.input, args.transform, args.shakiness, args.accuracy)

    apply_stabilization(args.input, args.transform, args.output, args.smoothing, args.crf)


if __name__ == "__main__":
    main()

"""
MODULE 02 — Deinterlacing (BWDIF)

Input  : Normalized video
Output : Progressive deinterlaced video

Purpose:
    - Remove interlacing artifacts
    - Fix combing and horizontal tearing
    - Generate clean progressive frames

"""

import argparse
import subprocess
from pathlib import Path
import sys


def check_ffmpeg():
    """Ensure ffmpeg is installed"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        print(" FFmpeg not found. Please install FFmpeg.")
        sys.exit(1)


def deinterlace_video(input_video: Path, output_video: Path, crf: int = 16):
    """Apply high-quality BWDIF deinterlacing"""

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", "bwdif=mode=send_field:parity=auto:deint=all",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    print("\n Running BWDIF Deinterlacing:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"\n Module 02 Complete → {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 02 — Deinterlacing (BWDIF)")

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to normalized input video"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to deinterlaced output video"
    )

    parser.add_argument(
        "--crf",
        default=16,
        type=int,
        help="CRF quality factor (default: 16)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        print(f" Input file not found: {args.input}")
        sys.exit(1)

    check_ffmpeg()
    deinterlace_video(args.input, args.output, args.crf)


if __name__ == "__main__":
    main()

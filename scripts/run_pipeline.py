#!/usr/bin/env python3

import argparse
from core.pipeline import VideoEnhancementPipeline


def parse_args():
    parser = argparse.ArgumentParser("Video Enhancement Pipeline")

    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="Resume pipeline from this stage index (default: 0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = VideoEnhancementPipeline()
    pipeline.run(start_stage=args.start_stage)


if __name__ == "__main__":
    main()

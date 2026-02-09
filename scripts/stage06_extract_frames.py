#!/usr/bin/env python3
import os
import cv2
from tqdm import tqdm

# ================================
# PATH CONFIG
# ================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_VIDEO = os.path.join(
    ROOT_DIR,
    "output/stage_05_presharpen/presharpened.mp4"
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "output/stage_06_frames"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# UTILS
# ================================

def log(msg):
    print(f"[STAGE 06] {msg}")

# ================================
# MAIN
# ================================

def main():

    log("Starting Frame Extraction")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Total frames: {total_frames}")

    idx = 0

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{idx:06d}.png"
        )

        cv2.imwrite(out_path, frame)
        idx += 1

    cap.release()

    log(f"Extraction complete: {idx} frames written.")

# ================================
# ENTRY
# ================================

if __name__ == "__main__":
    main()

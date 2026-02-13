import cv2
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# PATH CONFIG
# ----------------------------

INPUT_DIR  = Path("output/stage_11_temporal")
OUTPUT_DIR = Path("output/stage_12_final")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "final_restored.mp4"

FPS = 59   # Change if your source is different

# ----------------------------
# COLOR GRADING
# ----------------------------

def filmic_curve(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def cinematic_grade(img):
    img = img.astype(np.float32) / 255.0

    # Mild contrast curve (avoids whitening)
    img = np.power(img, 0.92)

    # Black point anchoring (deeper blacks)
    img = (img - 0.02) / (1.0 - 0.02)
    img = np.clip(img, 0, 1)

    # Convert to HSV for tonal tuning
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    # Slight saturation reduction (prevents color burn + whiteness)
    hsv[...,1] *= 0.97

    # Gentle warmth bias
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    img[...,2] *= 1.02   # R channel slight lift
    img[...,1] *= 1.01   # G channel tiny lift

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img



# ----------------------------
# MAIN
# ----------------------------

def main():

    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(" No frames found in stage_11_temporal")

    print("\n STAGE 12 — FINAL COLOR GRADING + VIDEO RECONSTRUCTION")
    print("=" * 60)
    print(f" Input : {INPUT_DIR}")
    print(f"  Output: {OUTPUT_VIDEO}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {FPS}")
    print("=" * 60)

    temp_dir = OUTPUT_DIR / "graded_frames"
    temp_dir.mkdir(exist_ok=True)

    # Apply grading
    for frame_path in tqdm(frames, desc="Color grading"):
        img = cv2.imread(str(frame_path))
        graded = cinematic_grade(img)
        out_path = temp_dir / frame_path.name
        cv2.imwrite(str(out_path), graded)

    # Build video using ffmpeg
    #cmd = [
    #    "ffmpeg", "-y",
    #   "-framerate", str(FPS),
    #   "-i", str(temp_dir / "%06d.png"),
    #    "-c:v", "libx264",
    #    "-preset", "slow",
    #    "-crf", "16",
    #   "-pix_fmt", "yuv420p",
    #    str(OUTPUT_VIDEO)
    #]

    cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-pattern_type", "glob",
    "-i", str(temp_dir / "*.png"),
    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "16",
    "-pix_fmt", "yuv420p",
    str(OUTPUT_VIDEO)
]

    print("\n Reconstructing final video...")
    subprocess.run(cmd, check=True)

    print("\n FINAL VIDEO COMPLETE")
    print(f" Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

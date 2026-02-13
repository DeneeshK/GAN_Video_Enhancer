import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

INPUT_DIR  = Path("output/stage_08_superres")
OUTPUT_DIR = Path("output/stage_09_detail_refine")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sharpening Parameters
UNSHARP_RADIUS = 1.0
UNSHARP_AMOUNT = 0.6

DETAIL_RADIUS  = 0.8
DETAIL_AMOUNT  = 0.35

# ----------------------------
# UTILS
# ----------------------------

def unsharp_mask(img, radius=1.0, amount=0.6):
    blur = cv2.GaussianBlur(img, (0,0), radius)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def micro_detail_boost(img, radius=0.8, amount=0.35):
    blur = cv2.GaussianBlur(img, (0,0), radius)
    high_pass = cv2.subtract(img, blur)
    return cv2.addWeighted(img, 1.0, high_pass, amount, 0)


# ----------------------------
# PIPELINE
# ----------------------------

def process_image(img_path, out_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read: {img_path.name}")
        return

    # Step 1 — Gentle unsharp mask
    sharpened = unsharp_mask(img, UNSHARP_RADIUS, UNSHARP_AMOUNT)

    # Step 2 — Micro-detail recovery
    refined = micro_detail_boost(sharpened, DETAIL_RADIUS, DETAIL_AMOUNT)

    cv2.imwrite(str(out_path), refined)


# ----------------------------
# MAIN
# ----------------------------

def main():
    images = sorted(INPUT_DIR.glob("*.png"))

    if not images:
        raise RuntimeError(" No frames found in stage_08_superres_exp")

    print("\n STAGE 09 — DETAIL REFINEMENT")
    print("=" * 55)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Frames: {len(images)}\n")

    for img_path in tqdm(images, desc="Refining"):
        out_path = OUTPUT_DIR / img_path.name
        process_image(img_path, out_path)

    print("\n DETAIL REFINEMENT COMPLETE")


if __name__ == "__main__":
    main()

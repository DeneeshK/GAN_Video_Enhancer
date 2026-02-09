import cv2
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from gfpgan import GFPGANer

# ----------------------------
# PATH CONFIG
# ----------------------------

INPUT_DIR  = Path("output/stage_09_detail_refine")
OUTPUT_DIR = Path("output/stage_10_faces")
WEIGHTS    = Path("models/weights/GFPGANv1.4.pth")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# PARAMETERS
# ----------------------------

FACE_STRENGTH = 0.55
UPSCALE = 1
BG_UPSAMPLER = None

# ----------------------------
# UTILS
# ----------------------------

def detail_reconstruction(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.08, blur, -0.08, 0)

# ----------------------------
# MAIN
# ----------------------------

def main():

    print("\n👤 STAGE 10 — GFPGAN FACE ENHANCEMENT")
    print("=" * 60)

    if not WEIGHTS.exists():
        raise FileNotFoundError(f" Missing GFPGAN weights: {WEIGHTS}\nRun tools/download_models.py")

    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(" No frames found for face enhancement")

    print(f" Input : {INPUT_DIR}")
    print(f" Output: {OUTPUT_DIR}")
    print(f" Frames: {len(frames)}")
    print(f"  Strength: {FACE_STRENGTH}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")

    restorer = GFPGANer(
        model_path=str(WEIGHTS),
        upscale=UPSCALE,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=BG_UPSAMPLER,
        device=device
    )

    for idx, frame_path in enumerate(tqdm(frames, desc="Enhancing faces")):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        _, _, restored = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=FACE_STRENGTH
        )

        restored = detail_reconstruction(restored)

        out_path = OUTPUT_DIR / frame_path.name
        cv2.imwrite(str(out_path), restored)

        if idx % 40 == 0 and device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    print("\n STAGE 10 COMPLETE")

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

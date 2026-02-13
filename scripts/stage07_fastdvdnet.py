# ============================================
# MODULE 07 — VIDEO DENOISING USING FastDVDNet
# ============================================

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# -------------------------------
# PATH SETUP
# -------------------------------

ROOT_DIR = Path("/home/dk/Projects/video_enhancer_v2/video_enhancer")

FASTDVDNET_PATH = ROOT_DIR / "models" / "fastdvdnet"
sys.path.insert(0, str(FASTDVDNET_PATH))

from models import FastDVDnet


# -------------------------------
# NOISE MAP FUNCTION
# -------------------------------

def generate_noise_map(h, w, sigma=25/255.0, device="cuda"):
    return torch.full((1, 1, h, w), sigma, device=device)


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------

def run(
    input_dir: Path = ROOT_DIR / "output/stage_06_frames",
    output_dir: Path = ROOT_DIR / "output/stage_07_denoised",
    sigma: float = 25/255.0
):
    """
    FastDVDNet video denoising pipeline stage.

    This function can be called from:
        - CLI
        - run_pipeline.py
        - FastAPI
        - Flask
        - Django
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # MODEL LOAD
    # -------------------------------

    MODEL_PATH = FASTDVDNET_PATH / "model.pth"

    model = FastDVDnet(num_input_frames=5)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    new_state_dict = {}
    for k, v in checkpoint.items():
        new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(" FastDVDNet loaded successfully")

    # -------------------------------
    # FRAME LOADING
    # -------------------------------

    frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    if len(frames) < 5:
        raise RuntimeError("Need at least 5 frames for FastDVDNet")

    print(f"Total frames: {len(frames)}")

    pad = 2

    # -------------------------------
    # DENOISING LOOP
    # -------------------------------

    for i in tqdm(range(len(frames)), desc="FastDVDNet Denoising"):
        
        idxs = [min(max(j, 0), len(frames)-1) for j in range(i-pad, i+pad+1)]

        stack = []
        for j in idxs:
            img = cv2.imread(str(input_dir / frames[j]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            stack.append(img)

        stack = np.stack(stack)
        stack = torch.from_numpy(stack).permute(0,3,1,2).contiguous()
        stack = stack.unsqueeze(0).to(device)

        N, F, C, H, W = stack.shape
        stack = stack.view(N, F*C, H, W)

        noise_map = generate_noise_map(H, W, sigma, device)

        with torch.no_grad():
            out = model(stack, noise_map)[0]

        out = out.permute(1,2,0).cpu().numpy()
        out = (out * 255).clip(0,255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_dir / frames[i]), out)

    print(" MODULE 07 COMPLETE — All frames denoised")


# -------------------------------
# CLI ENTRYPOINT (optional)
# -------------------------------

if __name__ == "__main__":
    run()

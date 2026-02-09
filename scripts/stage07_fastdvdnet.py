# ============================================
# MODULE 07 — VIDEO DENOISING USING FastDVDNet
# ============================================

import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# -------------------------------
# PATH SETUP
# -------------------------------

ROOT_DIR = "/home/dk/Projects/video_enhancer_v2/video_enhancer"

# FastDVDnet repo path
FASTDVDNET_PATH = os.path.join(ROOT_DIR, "models", "fastdvdnet")
sys.path.insert(0, FASTDVDNET_PATH)  # Add repo to python path


from models import FastDVDnet

# -------------------------------
# DEVICE
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# -------------------------------
# INPUT / OUTPUT DIRS
# -------------------------------

INPUT_DIR = os.path.join(ROOT_DIR, "output/stage_06_frames")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/stage_07_denoised")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# MODEL LOAD
# -------------------------------

MODEL_PATH = os.path.join(FASTDVDNET_PATH, "model.pth")
model = FastDVDnet(num_input_frames=5)

checkpoint = torch.load(MODEL_PATH, map_location=device)

# Fix for multi-GPU trained models
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()
print(" FastDVDNet loaded successfully")

# -------------------------------
# FRAME LOADING
# -------------------------------

frames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
if len(frames) < 5:
    raise RuntimeError("Need at least 5 frames for FastDVDNet")
print(f"Total frames: {len(frames)}")

# -------------------------------
# NOISE MAP FUNCTION
# -------------------------------

def generate_noise_map(h, w, sigma=25/255.0):
    """
    FastDVDNet expects noise map shape: [1,1,H,W]
    """
    return torch.full((1, 1, h, w), sigma, device=device)

# -------------------------------
# DENOISING LOOP
# -------------------------------

pad = 2  # 5-frame temporal window (2 previous + current + 2 next)

for i in tqdm(range(len(frames))):
    
    # Compute 5-frame indices with boundary handling
    idxs = [min(max(j, 0), len(frames)-1) for j in range(i-pad, i+pad+1)]
    
    stack = []
    for j in idxs:
        img_path = os.path.join(INPUT_DIR, frames[j])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        stack.append(img)
    
    # Stack into numpy array (5, H, W, 3)
    stack = np.stack(stack)
    
    # Convert to tensor: (5, 3, H, W)
    stack = torch.from_numpy(stack).permute(0, 3, 1, 2).contiguous()
    
    # Add batch dim: (1, 5, 3, H, W)
    stack = stack.unsqueeze(0).to(device)
    
    # Combine frames to match FastDVDNet expected input: (1, 15, H, W)
    N, F, C, H, W = stack.shape
    stack = stack.view(N, F*C, H, W)
    
    # Generate noise map
    noise_map = generate_noise_map(H, W, sigma=25/255.0)
    
    # Denoise
    with torch.no_grad():
        out = model(stack, noise_map)[0]  # FastDVDNet returns tuple
    
    # Convert back to image
    out = out.permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(os.path.join(OUTPUT_DIR, frames[i]), out)

print(" MODULE 07 COMPLETE — All frames denoised")

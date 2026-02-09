import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ----------------------------
# PATH CONFIG
# ----------------------------

INPUT_DIR  = Path("output/stage_10_faces")
OUTPUT_DIR = Path("output/stage_11_temporal")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# TEMPORAL PARAMETERS
# ----------------------------

BASE_BLEND = 0.85     # Base current-frame dominance
MOTION_SCALE = 15.0   # Motion sensitivity
FLOW_SCALE = 1.0

# ----------------------------
# UTILS
# ----------------------------

def warp_frame(prev, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def compute_motion_mask(flow):
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    mask = np.exp(-mag / MOTION_SCALE)
    return np.clip(mask, 0.0, 1.0)


# ----------------------------
# MAIN
# ----------------------------

def main():

    frames = sorted(INPUT_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(" No frames found in stage_10_faces")

    print("\n STAGE 11 — TEMPORAL CONSISTENCY (GHOST-FREE)")
    print("=" * 60)
    print(f" Input : {INPUT_DIR}")
    print(f" Output: {OUTPUT_DIR}")
    print(f"  Frames: {len(frames)}")
    print("=" * 60)

    prev_frame = cv2.imread(str(frames[0]))
    cv2.imwrite(str(OUTPUT_DIR / frames[0].name), prev_frame)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for frame_path in tqdm(frames[1:], desc="Temporal refinement"):
        curr = cv2.imread(str(frame_path))
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,
            levels=4,
            winsize=15,
            iterations=4,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )

        warped_prev = warp_frame(prev_frame, flow * FLOW_SCALE)
        motion_mask = compute_motion_mask(flow)

        adaptive_alpha = BASE_BLEND + (1 - motion_mask) * (1 - BASE_BLEND)
        adaptive_alpha = adaptive_alpha[..., None]

        blended = curr * adaptive_alpha + warped_prev * (1 - adaptive_alpha)
        blended = blended.astype(np.uint8)

        out_path = OUTPUT_DIR / frame_path.name
        cv2.imwrite(str(out_path), blended)

        prev_frame = blended
        prev_gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

    print("\n STAGE 11 COMPLETE — GHOSTING SUPPRESSED")


if __name__ == "__main__":
    main()

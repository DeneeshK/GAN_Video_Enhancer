from pathlib import Path
import sys
import time

# Import stage modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts import stage01_normalize
from scripts import stage02_deinterlace
from scripts import stage03_stabilize
from scripts import stage04_deflicker, stage05_presharpen, stage06_extract_frames, stage07_fastdvdnet, stage08_superres_simple, stage09_detail_refine, stage10_gfpgan, stage11_temporal_refine, stage12_reconstruct_video

class VideoEnhancementPipeline:
    """
    Professional Pipeline Orchestrator
    Can be called from:
        - CLI
        - FastAPI
        - Flask
        - Django
        - Background workers
    """

    def __init__(self, logger=print):
        self.logger = logger

        self.stages = [
            ("Stage 01  Normalize", stage01_normalize.main),
            ("Stage 02  Deinterlace", stage02_deinterlace.main),
            ("Stage 03  Stabilize", stage03_stabilize.main),

            ("Stage 04  Deflicker", stage04_deflicker.main),
            ("Stage 05  Presharpen", stage05_presharpen.main),
            ("Stage 06  Extract Frames", stage06_extract_frames.main),
            ("Stage 07  FastDVDnet", stage07_fastdvdnet.run),
            
            ("Stage 08  Super Res", stage08_superres_simple.main),
            ("Stage 09  Detail Refine", stage09_detail_refine.main),
            ("stage 10  GFP_GAN", stage10_gfpgan.main),
            ("stage 11  Temporal Refine", stage11_temporal_refine.main),
            ("stage 12  Reconstructing Video", stage12_reconstruct_video.main)
            
        ]

    def run(self, start_stage: int = 0):
        """
        Run full pipeline.

        start_stage:
            Allows resume from any stage.
            Example:
                start_stage=1 → skips stage 0
        """

        self.logger("\n================ PIPELINE START ================\n")

        for i, (name, func) in enumerate(self.stages):

            if i < start_stage:
                self.logger(f" Skipping {name}")
                continue

            self.logger(f" Running {name}")
            t0 = time.time()

            func()

            self.logger(f" {name} completed in {time.time() - t0:.2f}s\n")

        self.logger("================ PIPELINE COMPLETE ================\n")

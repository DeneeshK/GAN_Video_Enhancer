#!/usr/bin/env python3
"""

Input:  output/stage_07_denoised/*.png
Output: output/stage_08_superres/*.png (4x upscaled)

Pipeline:
1. SwinIR: Preserves structure, clean upscale
2. RealESRGAN: Adds natural sharpness and texture detail

This hybrid approach prevents hallucinations while maintaining sharpness.
"""

import os
import sys
import glob
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class SwinIRUpscaler:
    """SwinIR model wrapper for structural upscaling"""
    
    def __init__(self, model_path, scale=4, device='cuda'):
        self.device = device
        self.scale = scale
        
        print(f"Loading SwinIR model from {model_path}...")
        
        # Import SwinIR architecture
        try:
            from models.swinir.models.network_swinir import SwinIR as net
        except ImportError:
            raise ImportError("SwinIR model not found. Please download first.")
        
        # Define model parameters
        self.model = net(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='1conv'
        )
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint, strict=True)
        self.model.eval()
        self.model = self.model.to(device)
        
        print("✓ SwinIR loaded successfully")
    
    def upscale(self, img):
        """
        Upscale image using SwinIR
        
        Args:
            img: numpy array (H, W, C) in RGB, range [0, 255]
        
        Returns:
            upscaled numpy array (H*scale, W*scale, C)
        """
        # Convert to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img)
        
        # Convert back to numpy
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output


class RealESRGANUpscaler:
    """RealESRGAN model wrapper for texture enhancement"""
    
    def __init__(self, model_path, scale=4, device='cuda'):
        self.device = device
        self.scale = scale
        
        print(f"Loading RealESRGAN model from {model_path}...")
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError("RealESRGAN not installed. Run: pip install realesrgan")
        
        # Define model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale
        )
        
        # Create upsampler
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if 'cuda' in device else False,
            device=device
        )
        
        print("✓ RealESRGAN loaded successfully")
    
    def enhance(self, img):
        """
        Enhance image using RealESRGAN
        
        Args:
            img: numpy array (H, W, C) in BGR or RGB, range [0, 255]
        
        Returns:
            enhanced numpy array
        """
        output, _ = self.upsampler.enhance(img, outscale=self.scale)
        return output


class HybridSuperResolution:
    """
    Hybrid pipeline: SwinIR → RealESRGAN
    
    Strategy:
    - SwinIR provides clean, artifact-free structural upscale
    - RealESRGAN adds natural texture and sharpness
    - No hallucinations, maximum quality
    """
    
    def __init__(self, swinir_path, esrgan_path, scale=4, device='cuda'):
        self.scale = scale
        self.device = device
        
        # Load models
        self.swinir = SwinIRUpscaler(swinir_path, scale=scale, device=device)
        self.esrgan = RealESRGANUpscaler(esrgan_path, scale=scale, device=device)
        
        print("✓ Hybrid Super Resolution pipeline ready")
    
    def process(self, img):
        """
        Process image through hybrid pipeline
        
        Args:
            img: numpy array (H, W, C) in RGB, range [0, 255]
        
        Returns:
            super-resolved image
        """
        # Stage 1: SwinIR (structure preservation)
        img_swinir = self.swinir.upscale(img)
        
        # Stage 2: RealESRGAN (texture enhancement)
        # Note: RealESRGAN expects BGR, so convert
        img_bgr = cv2.cvtColor(img_swinir, cv2.COLOR_RGB2BGR)
        img_enhanced = self.esrgan.enhance(img_bgr)
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        
        return img_rgb


def process_frames(input_dir, output_dir, swinir_path, esrgan_path, 
                   scale=4, device='cuda', start_frame=None, end_frame=None):
    """
    Process all frames through super resolution pipeline
    
    Args:
        input_dir: Directory with denoised frames
        output_dir: Directory to save super-resolved frames
        swinir_path: Path to SwinIR model weights
        esrgan_path: Path to RealESRGAN model weights
        scale: Upscaling factor (default: 4)
        device: 'cuda' or 'cpu'
        start_frame: Optional starting frame index
        end_frame: Optional ending frame index
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input frames
    frame_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not frame_paths:
        raise ValueError(f"No PNG frames found in {input_dir}")
    
    # Apply frame range if specified
    if start_frame is not None:
        frame_paths = frame_paths[start_frame:]
    if end_frame is not None:
        frame_paths = frame_paths[:end_frame]
    
    print(f"Found {len(frame_paths)} frames to process")
    
    # Initialize hybrid pipeline
    pipeline = HybridSuperResolution(
        swinir_path=swinir_path,
        esrgan_path=esrgan_path,
        scale=scale,
        device=device
    )
    
    # Process frames
    print("\n🎬 Processing frames through SwinIR → RealESRGAN pipeline...")
    
    for frame_path in tqdm(frame_paths, desc="Super Resolution"):
        # Load frame
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        output = pipeline.process(frame_rgb)
        
        # Save
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, frame_name)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
    
    print(f"\n✓ Super resolution complete!")
    print(f"  Output: {output_dir}")
    print(f"  Frames processed: {len(frame_paths)}")


def main():
    parser = argparse.ArgumentParser(description="Stage 08: Hybrid Super Resolution")
    parser.add_argument(
        '--input_dir',
        default='output/stage_07_denoised',
        help='Input directory with denoised frames'
    )
    parser.add_argument(
        '--output_dir',
        default='output/stage_08_superres',
        help='Output directory for super-resolved frames'
    )
    parser.add_argument(
        '--swinir_model',
        default='models/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
        help='Path to SwinIR model weights'
    )
    parser.add_argument(
        '--esrgan_model',
        default='models/realesrgan/RealESRGAN_x4plus.pth',
        help='Path to RealESRGAN model weights'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=4,
        help='Upscaling factor (default: 4)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for processing'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        help='Starting frame index (optional)'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        help='Ending frame index (optional)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("STAGE 08: HYBRID SUPER RESOLUTION")
    print("="*60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Scale:  {args.scale}x")
    print(f"Device: {args.device}")
    print(f"Pipeline: SwinIR → RealESRGAN")
    print("="*60)
    
    # Process frames
    process_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        swinir_path=args.swinir_model,
        esrgan_path=args.esrgan_model,
        scale=args.scale,
        device=args.device,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
    
    print("\n✅ Stage 08 complete!")


if __name__ == '__main__':
    main()
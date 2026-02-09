#!/usr/bin/env python3
"""
Stage 08: Super Resolution (RealESRGAN - Optimized)
====================================================

Input:  output/stage_07_denoised/*.png
Output: output/stage_08_superres/*.png

Improvements:
- 2x and 4x scaling support
- Sharpening to reduce blur
- Better edge preservation
- Configurable quality settings
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


class RealESRGANUpscaler:
    """RealESRGAN model wrapper with quality enhancements"""
    
    def __init__(self, model_path, scale=4, device='cuda', sharpen=True, 
                 sharpen_strength=0.3, edge_enhance=True):
        self.device = device
        self.target_scale = scale  # What user wants (2x or 4x)
        self.sharpen = sharpen
        self.sharpen_strength = sharpen_strength
        self.edge_enhance = edge_enhance
        
        print(f"Loading RealESRGAN model...")
        print(f"  Target scale: {scale}x")
        print(f"  Sharpening: {sharpen} (strength: {sharpen_strength})")
        print(f"  Edge enhance: {edge_enhance}")
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            print("✗ RealESRGAN not installed!")
            print("\nPlease run: pip install realesrgan basicsr --break-system-packages")
            sys.exit(1)
        
        # RealESRGAN_x4plus only does 4x, so we'll always load 4x model
        # and downscale to 2x if needed
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4  # Always use 4x model
        )
        
        # Create upsampler with optimized settings
        self.upsampler = RealESRGANer(
            scale=4,  # Model native scale
            model_path=model_path,
            model=model,
            tile=256,      # Smaller tiles = better quality
            tile_pad=10,   # More padding = less tile artifacts
            pre_pad=0,
            half=True if 'cuda' in device else False,
            device=device
        )
        
        print("✓ RealESRGAN loaded successfully")
    
    def apply_sharpening(self, img, strength=0.3):
        """
        Apply unsharp mask sharpening
        
        Args:
            img: BGR image
            strength: Sharpening strength (0.0-1.0)
        
        Returns:
            Sharpened image
        """
        # Create gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        
        return sharpened
    
    def enhance_edges(self, img):
        """
        Enhance edges using bilateral filter + detail enhancement
        
        Args:
            img: BGR image
        
        Returns:
            Edge-enhanced image
        """
        # Bilateral filter preserves edges while smoothing
        filtered = cv2.bilateralFilter(img, 5, 50, 50)
        
        # Extract edge details (convert to float for math operations)
        details = cv2.subtract(img, filtered).astype(np.float32)
        
        # Enhance details
        enhanced_details = (details * 1.3).clip(0, 255).astype(np.uint8)
        
        # Add back enhanced details
        enhanced = cv2.add(filtered, enhanced_details)
        
        return enhanced
    
    def enhance(self, img):
        """
        Enhance image using RealESRGAN with quality improvements
        
        Args:
            img: BGR image (H, W, 3)
        
        Returns:
            Enhanced BGR image at target scale
        """
        h, w = img.shape[:2]
        
        # Step 1: Apply RealESRGAN (always 4x)
        output, _ = self.upsampler.enhance(img, outscale=4)
        
        # Step 2: If target is 2x, downscale intelligently
        #if self.target_scale == 2:
         
         #   target_h = h * 2
          #  target_w = w * 2
            
            # Use INTER_AREA for downscaling (best quality)
           # output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Step 3: Edge enhancement (reduces blur)
        if self.edge_enhance:
            output = self.enhance_edges(output)
        
        # Step 4: Sharpening (reduces smudging)
        if self.sharpen:
            output = self.apply_sharpening(output, self.sharpen_strength)
        
        return output


def process_frames(input_dir, output_dir, model_path, scale=2, device='cuda', 
                   sharpen=True, sharpen_strength=0.3, edge_enhance=True,
                   start_frame=None, end_frame=None):
    """Process all frames through super resolution"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input frames
    frame_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not frame_paths:
        print(f"✗ No PNG frames found in {input_dir}")
        print("\nPlease ensure Stage 07 (denoising) has been completed.")
        sys.exit(1)
    
    # Apply frame range if specified
    if start_frame is not None:
        frame_paths = frame_paths[start_frame:]
    if end_frame is not None:
        frame_paths = frame_paths[:end_frame]
    
    print(f"Found {len(frame_paths)} frames to process")
    
    # Initialize upscaler
    print("\n" + "="*60)
    print("INITIALIZING SUPER RESOLUTION")
    print("="*60)
    
    upscaler = RealESRGANUpscaler(
        model_path=model_path,
        scale=scale,
        device=device,
        sharpen=sharpen,
        sharpen_strength=sharpen_strength,
        edge_enhance=edge_enhance
    )
    
    print("="*60)
    print("✓ Pipeline ready")
    print("="*60 + "\n")
    
    # Process frames
    print("🎬 Processing frames through RealESRGAN...\n")
    
    for frame_path in tqdm(frame_paths, desc="Super Resolution", ncols=80):
        # Load frame
        frame = cv2.imread(frame_path)
        
        # Process
        output = upscaler.enhance(frame)
        
        # Save with high quality
        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    print(f"\n✓ Super resolution complete!")
    print(f"  Output: {output_dir}")
    print(f"  Frames processed: {len(frame_paths)}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 08: Super Resolution (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Presets:
  --preset soft       : Minimal sharpening (sharpen=0.1)
  --preset balanced   : Moderate sharpening (sharpen=0.3) [DEFAULT]
  --preset sharp      : Strong sharpening (sharpen=0.5)
  --preset ultra      : Maximum sharpening (sharpen=0.7)

Examples:
  # 2x upscale with balanced quality
  python scripts/stage08_superres.py --scale 2 --preset balanced
  
  # 2x upscale with strong sharpening
  python scripts/stage08_superres.py --scale 2 --preset sharp
  
  # 4x upscale with ultra sharpening
  python scripts/stage08_superres.py --scale 4 --preset ultra
  
  # No sharpening (softer but no artifacts)
  python scripts/stage08_superres.py --scale 2 --no_sharpen
  
  # Test on 10 frames first
  python scripts/stage08_superres.py --scale 2 --start_frame 0 --end_frame 10
        """
    )
    
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
        '--model',
        default='models/realesrgan/RealESRGAN_x4plus.pth',
        help='Path to RealESRGAN model weights'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=2,
        choices=[2, 4],
        help='Upscaling factor: 2x or 4x (default: 2)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for processing'
    )
    parser.add_argument(
        '--preset',
        choices=['soft', 'balanced', 'sharp', 'ultra'],
        default='balanced',
        help='Quality preset (default: balanced)'
    )
    parser.add_argument(
        '--sharpen_strength',
        type=float,
        help='Custom sharpening strength (0.0-1.0, overrides preset)'
    )
    parser.add_argument(
        '--no_sharpen',
        action='store_true',
        help='Disable sharpening'
    )
    parser.add_argument(
        '--no_edge_enhance',
        action='store_true',
        help='Disable edge enhancement'
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
    
    # Determine sharpening strength
    if args.sharpen_strength is not None:
        sharpen_strength = args.sharpen_strength
    else:
        preset_map = {
            'soft': 0.1,      # Very gentle
            'balanced': 0.3,  # Good middle ground
            'sharp': 0.5,     # Strong sharpening
            'ultra': 0.7      # Maximum sharpening
        }
        sharpen_strength = preset_map[args.preset]
    
    # Apply sharpening flag
    sharpen = not args.no_sharpen
    edge_enhance = not args.no_edge_enhance
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("STAGE 08: SUPER RESOLUTION (OPTIMIZED)")
    print("="*60)
    print(f"Input:             {args.input_dir}")
    print(f"Output:            {args.output_dir}")
    print(f"Scale:             {args.scale}x")
    print(f"Device:            {args.device}")
    print(f"Preset:            {args.preset}")
    print(f"Sharpen:           {sharpen}")
    print(f"Sharpen Strength:  {sharpen_strength}")
    print(f"Edge Enhance:      {edge_enhance}")
    print("="*60 + "\n")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Model not found: {args.model}")
        print("\nPlease run: python tools/download_stage08_models.py")
        sys.exit(1)
    
    # Process frames
    process_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        scale=args.scale,
        device=args.device,
        sharpen=sharpen,
        sharpen_strength=sharpen_strength,
        edge_enhance=edge_enhance,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
    
    print("\n" + "="*60)
    print("✅ STAGE 08 COMPLETE!")
    print("="*60)
    print("\nQuality Tips:")
    print("  - Too sharp/artifacts? Try: --preset soft")
    print("  - Too blurry? Try: --preset sharp or --preset ultra")
    print("  - Faces blurred? Edge enhancement should help")
    print("  - Still issues? Try 4x scale: --scale 4")


if __name__ == '__main__':
    main()
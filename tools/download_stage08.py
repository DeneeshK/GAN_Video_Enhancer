#!/usr/bin/env python3
"""
Download Models for Stage 08: Super Resolution
===============================================

Downloads:
1. SwinIR (structure preservation)
2. RealESRGAN (texture enhancement)
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_swinir_model(output_dir):
    """Download SwinIR pretrained model"""
    
    print("\n📥 Downloading SwinIR model...")
    
    model_url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    model_path = os.path.join(output_dir, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    
    if os.path.exists(model_path):
        print(f"✓ SwinIR model already exists: {model_path}")
        return model_path
    
    download_file(model_url, model_path)
    print(f"✓ SwinIR model downloaded: {model_path}")
    
    return model_path


def download_realesrgan_model(output_dir):
    """Download RealESRGAN pretrained model"""
    
    print("\n📥 Downloading RealESRGAN model...")
    
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = os.path.join(output_dir, "RealESRGAN_x4plus.pth")
    
    if os.path.exists(model_path):
        print(f"✓ RealESRGAN model already exists: {model_path}")
        return model_path
    
    download_file(model_url, model_path)
    print(f"✓ RealESRGAN model downloaded: {model_path}")
    
    return model_path


def clone_swinir_repo(repo_dir):
    """Clone SwinIR repository for model architecture"""
    
    print("\n📦 Setting up SwinIR repository...")
    
    if os.path.exists(os.path.join(repo_dir, "models", "network_swinir.py")):
        print(f"✓ SwinIR repo already exists: {repo_dir}")
        return
    
    import subprocess
    
    os.makedirs(repo_dir, exist_ok=True)
    
    print("Cloning SwinIR repository...")
    subprocess.run([
        "git", "clone", 
        "https://github.com/JingyunLiang/SwinIR.git",
        repo_dir
    ], check=True)
    
    print(f"✓ SwinIR repo cloned: {repo_dir}")


def install_dependencies():
    """Install required Python packages"""
    
    print("\n📦 Installing dependencies...")
    
    import subprocess
    
    packages = [
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "tqdm",
        "pillow",
        "basicsr",
        "facexlib",
        "gfpgan",
        "realesrgan"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            package, "--break-system-packages"
        ], check=True)
    
    print("✓ All dependencies installed")


def main():
    """Download all models for Stage 08"""
    
    # Define directories
    project_root = Path(__file__).parent.parent
    swinir_dir = project_root / "models" / "swinir"
    realesrgan_dir = project_root / "models" / "realesrgan"
    
    print("="*60)
    print("STAGE 08: MODEL DOWNLOAD")
    print("="*60)
    print(f"Project root: {project_root}")
    print("="*60)
    
    # Install dependencies
    install_dependencies()
    
    # Clone SwinIR repository
    clone_swinir_repo(str(swinir_dir))
    
    # Download SwinIR model
    swinir_model = download_swinir_model(str(swinir_dir))
    
    # Download RealESRGAN model
    esrgan_model = download_realesrgan_model(str(realesrgan_dir))
    
    print("\n" + "="*60)
    print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("="*60)
    print(f"SwinIR model:    {swinir_model}")
    print(f"RealESRGAN model: {esrgan_model}")
    print("="*60)
    print("\nYou can now run: python scripts/stage08_superres.py")


if __name__ == '__main__':
    main()
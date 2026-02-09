import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
WEIGHTS_DIR = Path("models/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = WEIGHTS_DIR / "GFPGANv1.4.pth"

print("=" * 60)
print("⬇️  GFPGAN MODEL DOWNLOAD")
print("=" * 60)

if MODEL_PATH.exists():
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Model already exists: {MODEL_PATH}")
    print(f"📊 Size: {size_mb:.2f} MB")
    exit(0)

print(f"📦 Downloading GFPGAN model...")
print(f"➡️  Destination: {MODEL_PATH}")
print("⏳ Please wait...")

def progress(block, block_size, total):
    downloaded = block * block_size
    percent = min(downloaded * 100 / total, 100)
    print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)

print("\n✅ Download completed")

size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
print(f"📊 Final size: {size_mb:.2f} MB")

if size_mb < 300:
    print("⚠️ WARNING: File size seems small. Check integrity.")
else:
    print("🎯 Model ready for inference.")

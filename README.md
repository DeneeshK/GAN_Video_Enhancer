# 🎬 GAN Video Enhancer — Modular AI Video Restoration Pipeline

A **fully modular, stage-based AI video enhancement pipeline** built using **classical computer vision + deep learning models**.
This project focuses on **realistic restoration, temporal stability, controlled enhancement, and cinematic output quality**.

> Designed for **research, experimentation, and production deployment**.

---

## 📌 Project Repository

```bash
gh repo clone DeneeshK/GAN_Video_Enhancer
```

or

```bash
git clone https://github.com/DeneeshK/GAN_Video_Enhancer.git
```

---

# 📖 Project Overview

This project implements a **12-stage professional video enhancement pipeline** that processes raw compressed video into a **high-quality, temporally stable, cinematic output**.

Each processing step is **fully modular**, allowing:

* Independent execution of any stage
* Full pipeline execution using a single command
* Easy integration into **FastAPI / Flask / Django / backend systems**

The architecture is designed to **avoid common GAN artifacts** such as:

* Flicker
* Temporal instability
* Over-sharpening
* Plastic-looking faces
* Hallucinated textures

---

# 🏗️ Pipeline Design Philosophy

This project uses a **hybrid processing model**:

> **Classical Computer Vision → Deep Learning → Classical Refinement**

This ensures:

* Natural sharpness
* Stable temporal consistency
* Minimal hallucination
* Cinematic color rendering

---

# 🧩 Pipeline Modules (Stage-wise Design)

| Stage | Module                  | Description                                        |
| ----- | ----------------------- | -------------------------------------------------- |
| 01    | Decode + Normalize      | Standardize FPS, colorspace, format                |
| 02    | Deinterlacing (BWDIF)   | Remove interlacing artifacts                       |
| 03    | Stabilization (VidStab) | Motion correction + jitter removal                 |
| 04    | Temporal Deflicker      | Remove brightness & exposure flicker               |
| 05    | Pre-sharpen (OpenCV)    | Edge-preserving classical sharpening               |
| 06    | Frame Extraction        | Lossless PNG frame extraction                      |
| 07    | FastDVDNet Denoising    | AI-based video denoising                           |
| 08    | Super Resolution        | Real-ESRGAN (SwinIR optional - currently bypassed) |
| 09    | Detail Refinement       | Controlled micro-detail sharpening                 |
| 10    | Face Enhancement        | GFPGAN face restoration                            |
| 11    | Temporal Refinement     | Frame-to-frame stability correction                |
| 12    | Final Reconstruction    | Video rebuild + color grading                      |

---

# 🧠 Why This Architecture Works Better

GANs are extremely powerful — but **they amplify noise, flicker, and instability**.

This pipeline carefully **preconditions video before AI** and **refines after AI**, producing:

* Natural textures
* Stable motion
* Sharp edges without halos
* Organic faces
* Cinematic color output

---

# 📂 Project Structure

```
video_enhancer/
│
├── input/raw_videos/        # Raw input videos
├── output/                 # Outputs of all pipeline stages
├── models/                 # AI models + weights
├── scripts/                # Stage-wise execution scripts
├── core/                   # Pipeline orchestration logic
├── utils/                  # Shared helper utilities
├── configs/                # YAML configuration files
├── tools/                  # Extra pipeline tools
├── requirements.txt
├── requirements_stage08.txt
├── environment.yml
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/DeneeshK/GAN_Video_Enhancer.git
cd GAN_Video_Enhancer
```

---

## 2️⃣ Create Python Environment (Recommended)

Using Conda:

```bash
conda env create -f environment.yml
conda activate video_enhancer
```

or using pip:

```bash
pip install -r requirements.txt
pip install -r requirements_stage08.txt
```

---

## 3️⃣ Clone Required Model Repositories

```bash
cd models

git clone https://github.com/m-tassano/fastdvdnet.git fastdvdnet
git clone https://github.com/xinntao/Real-ESRGAN.git realesrgan
git clone https://github.com/TencentARC/GFPGAN.git gfpgan
```

> **Note:**
> SwinIR is included in pipeline design but currently **bypassed due to technical constraints**.

---

## 4️⃣ Download Model Weights

Place all pretrained weights inside:

```
models/weights/
```

Required:

* FastDVDNet → `model.pth`
* RealESRGAN → `RealESRGAN_x4plus.pth`
* GFPGAN → Official pretrained model

---

# 🚀 Running the Full Pipeline

Place your input video into:

```
input/raw_videos/
```

Then run:

```bash
python -m scripts.run_pipeline
```

This executes **all 12 stages automatically**.

---

# 🔧 Running Individual Stages

Each stage is independently runnable.

Example:

```bash
python scripts/stage01_normalize.py --help
```

This allows **fine-grained tuning** of:

* Sharpen strength
* Temporal smoothing
* Noise reduction
* Color parameters
* Face enhancement intensity

---

# 🔌 API / Backend Integration

The pipeline is designed for **direct backend usage**.

```python
from core.pipeline import VideoEnhancementPipeline

pipeline = VideoEnhancementPipeline()
pipeline.run()
```

This enables seamless deployment in:

* FastAPI
* Flask
* Django
* Rust backends (via subprocess calls)
* Desktop applications

---



# 🧪 Current Limitations

* **SwinIR stage is currently bypassed** due to stability constraints
* Heavy GPU memory usage during super-resolution
* High compute cost for full pipeline

---

# 🏆 Credits & Acknowledgements

This project builds upon the excellent work of:

* **FastDVDNet** — https://github.com/m-tassano/fastdvdnet
* **Real-ESRGAN** — https://github.com/xinntao/Real-ESRGAN
* **GFPGAN** — https://github.com/TencentARC/GFPGAN
* **FFmpeg** — https://ffmpeg.org
* **OpenCV** — https://opencv.org
* **PyTorch** — https://pytorch.org

All credit goes to the original authors of these models and frameworks.

---

# 🧠 Project Goal

> **High-quality video restoration with realism, stability, and cinematic rendering — without hallucination.**

---



---

# 🤝 Contributions

Pull requests, bug reports, performance improvements, and feature ideas are welcome.

---

# ⭐ If you find this project useful, consider giving it a star!

---


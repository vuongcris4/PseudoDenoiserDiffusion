---
name: Setup environment denoiser
description: Conda environment 'denoiser' đã được cài đặt cho project D3PM
type: reference
---

# Environment Setup

## Conda Environment: denoiser

Environment đã được cài đặt với:
- Python 3.10
- PyTorch 2.10.0 + CUDA 12.8
- mmcv-full 1.7.2
- opencv-python-headless
- diffusion_denoiser (dev mode)

## Kích hoạt environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate denoiser
```

## Lệnh train

```bash
# Single GPU
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py

# Multi-GPU
torchrun --nproc_per_node=4 tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py --launcher pytorch
```

## Data

Data OEM_v2_aDanh đã được symlink vào `data/OEM_v2_aDanh`

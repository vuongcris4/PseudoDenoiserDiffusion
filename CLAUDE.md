# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install package in development mode
pip install -e .

# Train (single GPU)
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py

# Train (multi-GPU)
torchrun --nproc_per_node=4 tools/train.py configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py --launcher pytorch

# Resume training
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py --resume-from work_dirs/d3pm_concat_uniform/latest.pth

# Inference (denoise pseudo-labels)
python tools/inference.py <config> <checkpoint> --img-dir <images> --pseudo-dir <pseudo_labels> --out-dir <output> --num-classes 7

# Evaluation (compute mIoU)
python tools/test.py <config> <checkpoint> --num-steps 50
```

## Architecture Overview

This is a **D3PM (Discrete Denoising Diffusion Probabilistic Model)** implementation for denoising pseudo-labels in satellite image segmentation.

### Core Components (`diffusion_denoiser/`)

1. **DiffusionDenoiserModel** (`models/diffusion_denoiser.py`): Top-level model that combines:
   - `ConditionalUNet`: Denoising network with timestep embedding
   - `DiscreteNoiseSchedule`: Precomputed transition matrices Q_t and Q_bar_t
   - `D3PM`: Training (forward diffusion + loss) and sampling (reverse diffusion)

2. **D3PM** (`diffusion/d3pm.py`): Implements the discrete diffusion pipeline:
   - Forward: `q_sample(x_0, t)` → noisy x_t using cumulative transition matrices
   - Training: hybrid loss = KL divergence + λ * cross-entropy on x_0 prediction
   - Sampling: reverse diffusion from x_T → x_0 using predicted posteriors

3. **ConditionalUNet** (`models/conditional_unet.py`): UNet with three conditioning modes:
   - `concat`: Satellite features concatenated with one-hot label at input
   - `crossattn`: Cross-attention injection at bottleneck and decoder
   - `hybrid`: Both concat and cross-attention
   - Supports `PretrainedConditionEncoder` (SegFormer-B2, ResNet50/101) with frozen early stages

4. **Noise Schedule** (`diffusion/noise_schedule.py`): Two transition types:
   - `uniform`: Any class → any class with probability β_t
   - `absorbing`: Non-mask classes → mask (absorbing state) with probability β_t

### Config System (`configs/`)

Configs follow MMSegmentation style with `_base_` inheritance:
- `configs/_base_/models/`: Model architecture (cond_type, transition_type, channels)
- `configs/_base_/datasets/`: Data paths, crop size, normalization
- `configs/_base_/schedules/`: Optimizer, LR schedule, EMA settings
- `configs/denoiser/`: Full configs combining base components

### Data Flow

Training: clean_label → q_sample → x_t → model(x_t, t, satellite) → loss
Inference: satellite + noisy_pseudo → reverse diffusion → denoised_label

Dataset (`datasets/pseudo_label_dataset.py`): Returns triplet (satellite_img, pseudo_label, clean_label) with random crop/flip augmentation.

### Key Design Patterns

- **x_0 parameterization**: Model predicts clean label logits directly, not noise
- **EMA**: Exponential moving average of weights for inference (decay=0.9999)
- **Hybrid loss**: KL divergence for diffusion objective + CE for sharper predictions
- **Straight-through estimator**: Soft x_0 probabilities for gradient flow in posterior computation

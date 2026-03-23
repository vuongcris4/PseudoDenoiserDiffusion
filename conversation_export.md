# DifusionDenoiser — Development Session Export

**Date:** March 23, 2026
**Participants:** Hung (Senior Researcher, PhD) & Claude

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Deep Dive: DifusionDenoiser Architecture](#2-deep-dive-difusiondenoiser-architecture)
3. [Running Experiments via Configuration Files](#3-running-experiments-via-configuration-files)
4. [Ablation Study Script](#4-ablation-study-script)
5. [ConditionEncoder Analysis](#5-conditionencoder-analysis)
6. [PretrainedConditionEncoder Implementation](#6-pretrainedconditionencoder-implementation)
7. [ConditionalUNet Decoder Analysis](#7-conditionalunet-decoder-analysis)
8. [Configuration Summary](#8-configuration-summary)
9. [File Manifest](#9-file-manifest)

---

## 1. Project Overview

Two complementary codebases for **pseudo-label denoising** in semantic segmentation:

### MMsegDenoiser (Discriminative Approach)

A standard MMSegmentation-based framework that uses a discriminative neural network to refine noisy pseudo-labels. Built on the OpenMMLab ecosystem with the familiar config-driven workflow: backbone → decoder head → loss.

### DifusionDenoiser (Generative Approach)

A novel approach leveraging **D3PM (Discrete Denoising Diffusion Probabilistic Models)** (Austin et al., NeurIPS 2021) to iteratively denoise pseudo-labels through a learned reverse diffusion process over discrete label distributions.

Key insight: pseudo-labels are categorical (discrete class assignments), making D3PM a natural fit — unlike standard continuous diffusion models.

---

## 2. Deep Dive: DifusionDenoiser Architecture

### 2.1 D3PM Forward Process

The forward process corrupts a clean label map `x₀` over `T` timesteps using a **transition matrix** `Q_t`:

```
q(x_t | x_{t-1}) = Cat(x_t; p = x_{t-1} Q_t)
```

Two transition matrix types are implemented:

**Uniform Transition:**
Each class can transition to any other class with equal probability. At `t → T`, the distribution approaches a uniform categorical distribution.

```python
Q_t = (1 - β_t) I + β_t / K · 1·1ᵀ
```

**Absorbing Transition:**
Each class can only transition to a dedicated "mask" state (class index K). At `t → T`, all labels are masked.

```python
Q_t[i,i] = 1 - β_t,  Q_t[i,K] = β_t  (for i ≠ K)
Q_t[K,K] = 1                           (absorbing state)
```

### 2.2 Beta Schedules

- **Linear:** `β_t` increases linearly from `β_start` to `β_end`
- **Cosine:** (Nichol & Dhariwal, 2021) smoother schedule with `ᾱ_t = cos²((t/T + s)/(1+s) · π/2)`

### 2.3 Reverse Process & Training

The model learns `p_θ(x_{t-1} | x_t, condition)` using a Conditional UNet. Training minimizes a **hybrid loss**:

```
L = L_KL + λ · L_CE
```

where `L_KL` is the variational bound (KL divergence between true and predicted posteriors) and `L_CE` is a direct cross-entropy term (λ = 0.01).

### 2.4 Conditional UNet

The UNet processes concatenated input `[x_t_onehot, condition_features]` with:

- **Timestep modulation:** Sinusoidal embedding → MLP → scale-and-shift FiLM conditioning on each ResBlock
- **Self-attention** at specified resolutions (e.g., 2×, 4× downsampling)
- **Cross-attention** to multi-scale condition features (from ConditionEncoder)
- **Skip connections** from encoder to decoder

Output: logits over K classes at original spatial resolution.

---

## 3. Running Experiments via Configuration Files

The project uses MMSegmentation's hierarchical config system (`mmcv.Config`). Configs compose via `_base_` inheritance:

```
configs/
├── _base_/
│   ├── models/           # Architecture definitions
│   ├── datasets/         # Data pipeline (pseudo_label_diffusion.py)
│   ├── schedules/        # LR, optimizer, runner settings
│   └── default_runtime.py
└── denoiser/
    └── d3pm_*_512x512_100k.py   # Concrete experiment configs
```

### Example Concrete Config

```python
_base_ = [
    '../_base_/models/d3pm_concat_uniform.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]
data = dict(samples_per_gpu=4, workers_per_gpu=4)
```

### Running Training

```bash
# Single GPU
python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
    --work-dir work_dirs/d3pm_concat_uniform

# Multi-GPU
torchrun --nproc_per_node=4 tools/train.py <config> \
    --work-dir <work_dir> --launcher pytorch
```

### Running Evaluation

```bash
python tools/test.py configs/denoiser/<config>.py <checkpoint> \
    --num-steps 50 --temperature 1.0
```

---

## 4. Ablation Study Script

Created `tools/run_ablation.sh` — a three-phase script for systematic ablation across all configuration variants.

### Phases

1. **Training:** Iterates over all 6 config variants, trains each sequentially
2. **Evaluation:** Runs inference with configurable denoising steps on each checkpoint
3. **Aggregation:** Parses evaluation logs and produces a summary table (`.txt` and `.csv`)

### Usage

```bash
bash tools/run_ablation.sh                  # Full study, single GPU
bash tools/run_ablation.sh --gpus 4         # Multi-GPU
bash tools/run_ablation.sh --eval-only      # Skip training
bash tools/run_ablation.sh --resume         # Resume incomplete runs
bash tools/run_ablation.sh --eval-steps 100 # Custom denoising steps
```

### Experiment Matrix (Original 6)

| Conditioning | Noise Type |
|:---|:---|
| Concat | Uniform |
| Concat | Absorbing |
| CrossAttn | Uniform |
| CrossAttn | Absorbing |
| Hybrid | Uniform |
| Hybrid | Absorbing |

---

## 5. ConditionEncoder Analysis

The original `ConditionEncoder` is a lightweight custom CNN that converts a raw RGB condition image into a multi-scale feature pyramid for cross-attention conditioning.

### Architecture

```
Input (B, 3, H, W)
  ↓ Conv2d(3 → base_ch, 3×3, pad=1) + GroupNorm + SiLU
  ↓ Conv2d(base_ch → base_ch, 3×3, stride=2, pad=1) + GN + SiLU    → Scale 0
  ↓ Conv2d(base_ch → 2·base_ch, 3×3, stride=2, pad=1) + GN + SiLU  → Scale 1
  ↓ Conv2d(2·base_ch → 4·base_ch, 3×3, stride=2, pad=1) + GN + SiLU → Scale 2
  ↓ Conv2d(4·base_ch → 8·base_ch, 3×3, stride=2, pad=1) + GN + SiLU → Scale 3
```

With `base_ch=64`, output channels are `[64, 128, 256, 512]` at resolutions `[H/2, H/4, H/8, H/16]`.

### Interface

- `forward(x) → List[Tensor]`: Returns multi-scale features aligned with UNet encoder levels
- `out_channels: List[int]`: Channel counts at each scale

---

## 6. PretrainedConditionEncoder Implementation

### Motivation

Replace the lightweight custom CNN with pretrained backbones (SegFormer-B2, ResNet-50, ResNet-101) to leverage ImageNet-learned representations while preventing catastrophic forgetting through selective layer freezing.

### Design Decisions

1. **Same interface** as `ConditionEncoder`: `forward() → List[Tensor]`, `out_channels` attribute → drop-in replacement
2. **Dynamic import** from mmseg via `importlib.import_module` to avoid hard dependency
3. **1×1 projection heads** map backbone channels to UNet cross-attention dimensions
4. **Backbone-specific freeze logic** for SegFormer (`patch_embed{i}`, `block{i}`, `norm{i}`) and ResNet (`conv1`, `norm1`, `layer{i}`)
5. **`train()` override** keeps frozen BatchNorm in eval mode during training

### Supported Backbones

| Backbone | Pretrained Source | Feature Channels | Default Out Channels |
|:---|:---|:---|:---|
| SegFormer-B2 | `pretrain/mit_b2.pth` | [64, 128, 320, 512] | [64, 128, 256, 512] |
| ResNet-50 | `open-mmlab://resnet50_v1c` | [256, 512, 1024, 2048] | [64, 128, 256, 512] |
| ResNet-101 | `open-mmlab://resnet101_v1c` | [256, 512, 1024, 2048] | [64, 128, 256, 512] |

### Class Structure

```python
class PretrainedConditionEncoder(nn.Module):
    _BACKBONE_DEFAULTS = {
        'segformer_b2': dict(
            module_path='mmseg.models.backbones',
            class_name='MixVisionTransformer',
            init_cfg=dict(embed_dims=64, num_heads=[1,2,5,8], ...),
            feature_channels=[64, 128, 320, 512],
        ),
        'resnet50': dict(..., feature_channels=[256, 512, 1024, 2048]),
        'resnet101': dict(..., feature_channels=[256, 512, 1024, 2048]),
    }

    def __init__(self, backbone_type, pretrained, freeze_stages=2,
                 out_channels=None, num_scales=4, backbone_cfg=None):
        # Build backbone, load weights, freeze stages, add projection heads
        ...

    def forward(self, x) -> List[Tensor]:
        feats = self.backbone(x)           # Multi-scale features
        return [proj(f) for proj, f in zip(self.projections, feats)]
```

### Integration into ConditionalUNet

```python
class ConditionalUNet(nn.Module):
    def __init__(self, ..., cond_encoder_type='lightweight', pretrained_cond_cfg=None):
        if cond_type in ('crossattn', 'hybrid'):
            if cond_encoder_type == 'pretrained':
                self.cond_encoder = PretrainedConditionEncoder(**pretrained_cond_cfg)
            else:
                self.cond_encoder = ConditionEncoder(...)
```

### New Config Files Created

**Base model configs (`configs/_base_/models/`):**

- `d3pm_crossattn_uniform_segformer.py` — SegFormer-B2, cross-attention, uniform noise, freeze_stages=2
- `d3pm_crossattn_absorbing_resnet50.py` — ResNet-50, cross-attention, absorbing noise, freeze_stages=1
- `d3pm_hybrid_uniform_resnet101.py` — ResNet-101, hybrid conditioning, uniform noise, freeze_stages=2

**Experiment configs (`configs/denoiser/`):**

- `d3pm_crossattn_uniform_segformer_512x512_100k.py`
- `d3pm_crossattn_absorbing_resnet50_512x512_100k.py`
- `d3pm_hybrid_uniform_resnet101_512x512_100k.py`

All use `optimizer = dict(lr=5e-5)` — lower learning rate to prevent catastrophic forgetting of pretrained weights.

---

## 7. ConditionalUNet Decoder Analysis

### Decoder Construction (`__init__`, lines ~737–773)

The decoder mirrors the encoder in reverse. For each encoder level (from deepest to shallowest):

```python
for level in reversed(range(len(channel_mult))):
    out_ch = base_channels * channel_mult[level]

    for i in range(num_res_blocks + 1):  # +1 block vs encoder
        skip_ch = enc_channels.pop()      # Match skip connection
        in_ch = ch + skip_ch              # Concatenated channels
        layers.append(ResBlock(in_ch, out_ch, time_emb_dim))

        if resolution in attn_resolutions:
            layers.append(SelfAttention(out_ch))
            if cond_type in ('crossattn', 'hybrid'):
                layers.append(CrossAttention(out_ch, cond_ch))

    if level > 0:
        layers.append(Upsample(out_ch))
        resolution *= 2
```

### Key Design Details

- **+1 ResBlock rule:** Decoder has `num_res_blocks + 1` blocks per level (vs `num_res_blocks` in encoder), because the first decoder block at each level must also process the skip connection from the corresponding encoder downsampling operation
- **Skip connection accounting:** `enc_channels` is a stack built during encoder construction; the decoder pops in reverse order to determine concatenation channel counts
- **Channel flow example** (base=128, mult=[1,2,4,8]):

| Level | Encoder Out | Skip Ch | Decoder In | Decoder Out |
|:---|:---|:---|:---|:---|
| 3 (deepest) | 1024 | 1024 | 2048 | 1024 |
| 2 | 512 | 512 | 1536→512 | 512 |
| 1 | 256 | 256 | 768→256 | 256 |
| 0 (shallowest) | 128 | 128 | 384→128 | 128 |

### Decoder Forward Pass (lines ~831–853)

```python
for module in self.decoder:
    if isinstance(module, ResBlock):
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = module(x, t_emb)
    elif isinstance(module, SelfAttention):
        x = module(x)
    elif isinstance(module, CrossAttention):
        cond_feat = cond_features[cond_idx]
        x = module(x, cond_feat)
    elif isinstance(module, Upsample):
        x = module(x)
```

Final output head: `GroupNorm → SiLU → Conv2d(base_ch, num_classes, 1×1)` → logits over K classes.

---

## 8. Configuration Summary

### Total: 9 Training Configurations

**Original 6 (lightweight ConditionEncoder):**

| # | Config | Conditioning | Noise |
|---|:---|:---|:---|
| 1 | `d3pm_concat_uniform_512x512_100k` | Concat | Uniform |
| 2 | `d3pm_concat_absorbing_512x512_100k` | Concat | Absorbing |
| 3 | `d3pm_crossattn_uniform_512x512_100k` | CrossAttn | Uniform |
| 4 | `d3pm_crossattn_absorbing_512x512_100k` | CrossAttn | Absorbing |
| 5 | `d3pm_hybrid_uniform_512x512_100k` | Hybrid | Uniform |
| 6 | `d3pm_hybrid_absorbing_512x512_100k` | Hybrid | Absorbing |

**New 3 (PretrainedConditionEncoder):**

| # | Config | Backbone | Conditioning | Noise |
|---|:---|:---|:---|:---|
| 7 | `d3pm_crossattn_uniform_segformer_512x512_100k` | SegFormer-B2 | CrossAttn | Uniform |
| 8 | `d3pm_crossattn_absorbing_resnet50_512x512_100k` | ResNet-50 | CrossAttn | Absorbing |
| 9 | `d3pm_hybrid_uniform_resnet101_512x512_100k` | ResNet-101 | Hybrid | Uniform |

---

## 9. File Manifest

### Modified Files

| File | Description |
|:---|:---|
| `diffusion_denoiser/models/conditional_unet.py` | Added `PretrainedConditionEncoder`; updated `ConditionalUNet.__init__` |
| `diffusion_denoiser/models/diffusion_denoiser.py` | Pass-through for `cond_encoder_type` and `pretrained_cond_cfg` |
| `diffusion_denoiser/models/__init__.py` | Export `PretrainedConditionEncoder` |

### New Files

| File | Description |
|:---|:---|
| `configs/_base_/models/d3pm_crossattn_uniform_segformer.py` | Base: SegFormer-B2 crossattn uniform |
| `configs/_base_/models/d3pm_crossattn_absorbing_resnet50.py` | Base: ResNet-50 crossattn absorbing |
| `configs/_base_/models/d3pm_hybrid_uniform_resnet101.py` | Base: ResNet-101 hybrid uniform |
| `configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py` | Experiment config |
| `configs/denoiser/d3pm_crossattn_absorbing_resnet50_512x512_100k.py` | Experiment config |
| `configs/denoiser/d3pm_hybrid_uniform_resnet101_512x512_100k.py` | Experiment config |
| `tools/run_ablation.sh` | Ablation study automation script |

---

## References

- Austin, J., Johnson, D.D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces. *NeurIPS 2021*.
- Nichol, A.Q. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS 2021*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

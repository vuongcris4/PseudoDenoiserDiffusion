# PseudoDenoiserDiffusion Codebase Documentation

## Overview

This is a **D3PM (Discrete Denoising Diffusion Probabilistic Model)** implementation for **denoising pseudo-labels in satellite image segmentation**. The core idea is to use diffusion models to refine noisy pseudo-labels (from a poorly trained segmentation model) into cleaner labels using the satellite image as conditioning.

---

## 1. Problem & Approach

### The Problem

- Satellite image segmentation models often produce noisy predictions (pseudo-labels)
- These pseudo-labels contain errors but also useful signal
- **Goal**: Refine pseudo-labels to be closer to ground truth

### The Solution: Discrete Diffusion

Unlike continuous diffusion (DDPM) that works with Gaussian noise, this uses **D3PM** which operates on discrete categorical states (class labels).

```
┌─────────────────────────────────────────────────────────────────┐
│                    D3PM Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training (Forward Diffusion):                                  │
│  pseudo_label ──[q(x_t|x_0)]──► x_t ──[model]──► predict x_0   │
│                                    │                            │
│  Inference (Reverse Diffusion):   │                            │
│  x_T ──[p_θ(x_{t-1}|x_t,cond)]──► ... ──► x_0 (denoised)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Process**:
1. **Forward diffusion**: Gradually corrupt a pseudo-label `x_pseudo → x_t` over T timesteps
2. **Reverse diffusion**: Learn to denoise `x_T → x_0` conditioned on the satellite image

---

## 2. Core Architecture

### A. D3PM (`diffusion/d3pm.py`)

**Location**: `diffusion_denoiser/diffusion/d3pm.py`

The heart of the diffusion process.

#### Forward Process (Training)

```python
# Sample random timestep t
t = randint(0, T, (B,))

# Corrupt pseudo-label using cumulative transition matrix
x_t = noise_schedule.q_sample(pseudo_label, t)
```

**Key Equations**:

| Equation | Code Location | Description |
|----------|---------------|-------------|
| `q(x_t | x_0) = x_0_onehot @ Q_bar_t^T` | Line 164 | Transition probability |
| `q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) × q(x_{t-1} | x_0)` | Lines 173-224 | Posterior |

#### Loss Function (Hybrid)

```python
# Cross-entropy on x_0 prediction
loss_ce = cross_entropy(x_0_logits, clean_label)

# KL divergence on posterior
loss_kl = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))

# Hybrid: loss_total = loss_kl + λ * loss_ce  (λ=1.0 default)
```

#### Reverse Process (Sampling/Inference)

```python
@torch.no_grad()
def sample(self, condition, noisy_label, num_steps=None):
    x_t = noisy_label  # Start from noisy pseudo-label

    for t in reversed(range(T)):
        # Predict clean label
        x_0_probs = softmax(model(x_t, t, condition))

        # Compute posterior and sample
        posterior = soft_posterior(x_0_probs, x_t, t)
        x_{t-1} = multinomial(posterior)

    return x_0  # Final denoised output
```

---

### B. Noise Schedule (`diffusion/noise_schedule.py`)

**Location**: `diffusion_denoiser/diffusion/noise_schedule.py`

Precomputes all transition matrices for efficiency.

#### Two Transition Types

| Type | Formula | Behavior |
|------|---------|----------|
| **Uniform** | `Q_t = (1 - β_t) * I + β_t / K * ones(K, K)` | Any class → any class with probability β_t |
| **Absorbing** | `Q_t[:K-1, K-1] += β_t`<br>`Q_t[K-1, K-1] = 1.0` | Non-mask classes → mask (absorbing state) |

#### Beta Schedules

```python
# Linear
β_t = β_start + t * (β_end - β_start) / T

# Cosine (Nichol & Dhariwal 2021)
α_bar = cos((t/T + 0.008) / 1.008 * π/2)²
β_t = 1 - α_bar[t+1] / α_bar[t]
```

---

### C. ConditionalUNet (`models/conditional_unet.py`)

**Location**: `diffusion_denoiser/models/conditional_unet.py`

The denoising network that predicts `x_0` from `x_t` and satellite image.

#### Three Conditioning Modes

| Mode | Mechanism | Description |
|------|-----------|-------------|
| `concat` | Input concatenation | Satellite img concatenated with one-hot `x_t` |
| `crossattn` | Cross-attention | Satellite features injected via cross-attn at bottleneck/decoder |
| `hybrid` | Both | Strongest conditioning, most compute |

#### Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         CONDITIONAL UNet            │
┌───────────────────┴─────────────────────────────────────┴───────────────────┐
│                                                                             │
│  Input: x_t (B, K, H, W)  [+ satellite concat if needed]                   │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────┐                              │
│  │         Input Conv                      │ → (B, C, H, W)               │
│  └─────────────────────────────────────────┘                              │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────┐                              │
│  │         Encoder Block 0                 │ ────┐  Skip                 │
│  │  [ResBlock × 2] + [SelfAttn] + [Cross]  │     │                       │
│  └─────────────────────────────────────────┘     │                       │
│       │  ↓                                        │                       │
│  ┌─────────────────────────────────────────┐      │                       │
│  │         Encoder Block 1                 │ ──────┤  Skip                 │
│  │  [ResBlock × 2] + [SelfAttn] + [Cross]  │      │                       │
│  └─────────────────────────────────────────┘      │                       │
│       │  ↓                                        │                       │
│  ┌─────────────────────────────────────────┐      │                       │
│  │         Encoder Block 2                 │ ──────┤  Skip                 │
│  │  [ResBlock × 2] + [SelfAttn] + [Cross]  │      │                       │
│  └─────────────────────────────────────────┘      │                       │
│       │  ↓                                        │                       │
│  ┌─────────────────────────────────────────┐      │                       │
│  │         Encoder Block 3                 │ ──────┘  Skip                 │
│  │  [ResBlock × 2] + [SelfAttn] + [Cross]  │                              │
│  └─────────────────────────────────────────┘                              │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────┐                              │
│  │           Bottleneck                    │                              │
│  │  ResBlock → SelfAttn → CrossAttn → ResBlock                            │
│  └─────────────────────────────────────────┘                              │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────┐                              │
│  │         Decoder Block 3                 │ ←───┘  Concat Skip           │
│  │  [ResBlock × 3] + [SelfAttn] + [Cross]  │                              │
│       │  ↑                                                                  │
│  ┌─────────────────────────────────────────┐                              │
│  │         Decoder Block 2                 │ ←───┘  Concat Skip           │
│  │  [ResBlock × 3] + [SelfAttn] + [Cross]  │                              │
│       │  ↑                                                                  │
│  ┌─────────────────────────────────────────┐                              │
│  │         Decoder Block 1                 │ ←───┘  Concat Skip           │
│  │  [ResBlock × 3] + [SelfAttn] + [Cross]  │                              │
│       │  ↑                                                                  │
│  ┌─────────────────────────────────────────┐                              │
│  │         Decoder Block 0                 │ ←───┘  Concat Skip           │
│  │  [ResBlock × 3] + [SelfAttn] + [Cross]  │                              │
│  └─────────────────────────────────────────┘                              │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────────────────────────────┐                              │
│  │  GroupNorm → SiLU → Output Conv         │ → (B, K, H, W) logits       │
│  └─────────────────────────────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Building Blocks

| Block | Lines | Description |
|-------|-------|-------------|
| `SinusoidalTimestepEmbedding` | 32-49 | Standard positional encoding for timestep |
| `TimestepMLP` | 52-64 | Projects timestep embedding to model dimension |
| `ResBlock` | 67-101 | Residual block with scale-shift modulation from timestep |
| `SelfAttention` | 104-125 | Multi-head attention on spatial features |
| `CrossAttention` | 128-179 | UNet features attend to satellite condition features |
| `ConditionEncoder` | 205-257 | Lightweight CNN for satellite features |
| `PretrainedConditionEncoder` | 264-600 | SegFormer/ResNet backbone with frozen early layers |

#### Condition Encoder Options

| Option | Class | Description |
|--------|-------|-------------|
| **Lightweight** | `ConditionEncoder` | Simple conv stack, trained from scratch |
| **Pretrained** | `PretrainedConditionEncoder` | SegFormer-B2 or ResNet50/101 with frozen early stages |

---

### D. DiffusionDenoiserModel (`models/diffusion_denoiser.py`)

**Location**: `diffusion_denoiser/models/diffusion_denoiser.py`

Top-level wrapper combining all components.

```python
class DiffusionDenoiserModel(nn.Module):
    """Full diffusion denoiser model."""

    def __init__(self, num_classes=7, num_timesteps=100, ...):
        super().__init__()

        # Build components
        self.unet = ConditionalUNet(...)
        self.noise_schedule = DiscreteNoiseSchedule(...)

        # D3PM wraps everything for training/sampling
        self.d3pm = D3PM(
            denoise_model=self.unet,
            noise_schedule=self.noise_schedule,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            loss_type='hybrid',
            hybrid_lambda=1.0,
        )

    def forward(self, clean_label, satellite_img, pseudo_label=None):
        """Training forward pass."""
        return self.d3pm(clean_label, satellite_img, x_init=pseudo_label)

    @torch.no_grad()
    def denoise(self, satellite_img, noisy_label, num_steps=None):
        """Reverse diffusion for inference."""
        return self.d3pm.sample(satellite_img, noisy_label, num_steps)
```

---

## 3. Data Pipeline

### Dataset (`datasets/pseudo_label_dataset.py`)

**Location**: `diffusion_denoiser/datasets/pseudo_label_dataset.py`

Returns triplets: `(satellite_img, pseudo_label, clean_label)`

| Field | Shape | Description |
|-------|-------|-------------|
| `satellite_img` | `(3, H, W)` | RGB satellite image, normalized |
| `pseudo_label` | `(H, W)` | Noisy label from external model (class indices) |
| `clean_label` | `(H, W)` | Ground truth for training (class indices) |

#### Augmentation

- Random crop to 512×512
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Batch                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  satellite_img  ──────────────────┐                             │
│  (B, 3, H, W)                    │                             │
│                                   ▼                            │
│  pseudo_label  ──►  q_sample(t)  ──►  x_t  ──►  model(x_t, t, sat)  │
│  (B, H, W)                                    │                │
│                                               ▼                │
│  clean_label  ──────────────────────►  loss(x_0_pred, clean)   │
│  (B, H, W)                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Training (`tools/train.py`)

### Training Loop

```python
for iteration in range(max_iters):
    batch = next(data_iter)
    satellite = batch['satellite_img']      # (B, 3, H, W)
    clean_label = batch['clean_label']      # (B, H, W)
    pseudo_label = batch['pseudo_label']    # (B, H, W)

    # Forward: corrupt pseudo_label, target is clean_label
    losses = model(clean_label, satellite, pseudo_label)
    loss = losses['loss_total']

    # Backward
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # EMA update
    ema.update(model)

    # Logging, checkpointing, evaluation...
```

### Key Training Features

#### 1. EMA (Exponential Moving Average)

```python
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.named_parameters()}

    def update(self, model):
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.data, alpha=1 - self.decay)
```

- **Decay**: 0.9999 (default)
- **Usage**: EMA weights used for inference only

#### 2. Optimizer & Scheduler

| Component | Configuration |
|-----------|---------------|
| Optimizer | AdamW (lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01) |
| Warmup | Linear warmup for 5000 iterations |
| Schedule | Cosine annealing (T_max = max_iters - warmup_iters) |

#### 3. Evaluation Metrics

```python
def evaluate(model, val_loader, device, num_steps=10):
    """Compute mIoU of denoised output vs raw pseudo-labels."""

    for batch in val_loader:
        # Denoise with reduced steps for speed
        pred = model.denoise(satellite, pseudo, num_steps=num_steps)

        # Compute per-class IoU
        for c in range(num_classes):
            inter[c] += (pred == c & clean == c).sum()
            union[c] += (pred == c | clean == c).sum()

    miou_pred = (inter / union).mean()
    miou_delta = miou_pred - miou_pseudo  # Improvement over baseline
```

#### 4. W&B Integration

- Full experiment config logging (model, data, training, diffusion params)
- Git commit tracking
- Loss curves, evaluation metrics
- Visual inference tables with overlays

---

## 5. Inference (`tools/inference.py`)

```python
# Load model with EMA weights
model = DiffusionDenoiserModel(...)
ckpt = torch.load(checkpoint)
load_ema_weights(ckpt['ema'])  # Use EMA for inference

for each image:
    # Load and normalize satellite image
    satellite = load_and_normalize(image_path)

    # Load pseudo-label
    pseudo = load_pseudo_label(pseudo_path)

    # Reverse diffusion
    denoised = model.denoise(
        satellite,
        pseudo,
        num_steps=50,  # Can use fewer steps than training T
        temperature=1.0
    )

    save(denoised)
```

---

## 6. Configuration System

Uses MMSegmentation-style `_base_` inheritance.

### Example Config

```python
# configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py
_base_ = [
    '../_base_/models/d3pm_hybrid_uniform.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(samples_per_gpu=4, workers_per_gpu=4)
```

### Available Configs

| Config | Conditioning | Noise Type | Backbone |
|--------|-------------|------------|----------|
| `d3pm_concat_uniform_512x512_100k.py` | Concat | Uniform | CNN |
| `d3pm_crossattn_uniform_512x512_100k.py` | CrossAttn | Uniform | CNN |
| `d3pm_hybrid_uniform_512x512_100k.py` | Hybrid | Uniform | CNN |
| `d3pm_hybrid_uniform_resnet101_512x512_100k.py` | Hybrid | Uniform | ResNet101 |
| `d3pm_crossattn_uniform_segformer_512x512_100k.py` | CrossAttn | Uniform | SegFormer-B2 |
| `d3pm_concat_absorbing_512x512_100k.py` | Concat | Absorbing | CNN |
| `d3pm_hybrid_absorbing_512x512_100k.py` | Hybrid | Absorbing | CNN |

---

## 7. Command Reference

### Training

```bash
# Single GPU
python tools/train.py configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py \
    configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py --launcher pytorch

# Resume training
python tools/train.py configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py \
    --resume-from work_dirs/d3pm_hybrid_uniform/latest.pth
```

### Inference

```bash
python tools/inference.py \
    configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py \
    work_dirs/d3pm_hybrid_uniform/latest.pth \
    --img-dir data/test/images \
    --pseudo-dir data/test/pseudo_labels \
    --out-dir data/test/refined_labels \
    --num-classes 7 \
    --num-steps 50
```

### Evaluation

```bash
python tools/test.py \
    configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py \
    work_dirs/d3pm_hybrid_uniform/latest.pth \
    --num-steps 50
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **x_0 Parameterization** | Model predicts clean label directly (not noise) - more intuitive for segmentation |
| **Hybrid Loss** | KL divergence for diffusion objective + CE for sharper predictions |
| **Straight-Through Estimator** | Uses soft `x_0` probabilities for gradient flow in posterior computation |
| **Pretrained Condition Encoder** | Frozen early layers of SegFormer/ResNet, finetuning later layers for transfer learning |
| **EMA for Inference** | Standard practice in diffusion models for better generation quality |

---

## 9. Class Reference

### Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `DiffusionDenoiserModel` | `models/diffusion_denoiser.py` | Top-level model wrapper |
| `D3PM` | `diffusion/d3pm.py` | D3PM training and sampling logic |
| `DiscreteNoiseSchedule` | `diffusion/noise_schedule.py` | Precomputed transition matrices |
| `ConditionalUNet` | `models/conditional_unet.py` | UNet with conditioning |
| `PretrainedConditionEncoder` | `models/conditional_unet.py` | SegFormer/ResNet backbone wrapper |
| `PseudoLabelDiffusionDataset` | `datasets/pseudo_label_dataset.py` | Dataset for training/inference |

### Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `q_sample()` | `DiscreteNoiseSchedule` | Sample `x_t` from `q(x_t | x_0)` |
| `q_posterior()` | `DiscreteNoiseSchedule` | Compute `q(x_{t-1} | x_t, x_0)` |
| `_predict_x0()` | `D3PM` | Run denoising model to predict clean label |
| `_compute_loss()` | `D3PM` | Compute hybrid KL + CE loss |
| `sample()` | `D3PM` | Reverse diffusion sampling |
| `denoise()` | `DiffusionDenoiserModel` | High-level denoising interface |

---

## 10. Reference

**D3PM Paper**: Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & Van Den Berg, R. (2021). "Structured Denoising Diffusion Models in Discrete State-Spaces". *NeurIPS 2021*.

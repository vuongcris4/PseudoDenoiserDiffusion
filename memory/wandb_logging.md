---
name: W&B Experiment Logging
description: Training script log đầy đủ thông số experiment vào W&B với config chi tiết
type: feedback
---

**W&B logging được cấu hình trong `tools/train.py`:**

Khi khởi tạo `wandb.init()`, toàn bộ thông số experiment được log bao gồm:

1. **Model architecture**: `cfg.model` (num_classes, num_timesteps, base_channels, etc.)
2. **Dataset settings**: type, data_root, num_classes, img_size, samples_per_gpu
3. **Training settings**: max_iters, optimizer, lr_scheduler, ema decay
4. **Runtime settings**: seed, checkpoint_interval, eval_interval, log_interval
5. **Diffusion specific**: num_timesteps, transition_type, beta_schedule, loss_type, hybrid_lambda
6. **UNet architecture**: base_channels, channel_mult, num_res_blocks, attn_resolutions, cond_type, dropout

**Run tags** tự động được set: [cond_type, transition_type, batch_size]

**Cách sử dụng:**
```bash
wandb login  # Lần đầu tiên
python tools/train.py configs/denoiser/d3pm_concat_uniform_oem_ciscr_512x512_100k.py
```

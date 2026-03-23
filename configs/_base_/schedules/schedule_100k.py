# Training schedule for diffusion model.
# Diffusion models typically require longer training than discriminative models.

optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
lr_scheduler = dict(
    type='cosine',
    warmup_iters=5000,
    warmup_ratio=1e-6,
    min_lr=1e-6)

max_iters = 100000
# checkpoint_interval and log_interval are defined in default_runtime.py
# checkpoint_interval = 10000
eval_interval = 10000
# log_interval = 100

# EMA for stable training
use_ema = True
ema_decay = 0.9999

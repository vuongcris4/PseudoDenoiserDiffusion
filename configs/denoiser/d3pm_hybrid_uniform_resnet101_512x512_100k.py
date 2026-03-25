_base_ = [
    '../_base_/models/d3pm_hybrid_uniform_resnet101.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(samples_per_gpu=2, workers_per_gpu=4)

# Lower LR for pretrained condition encoder to prevent catastrophic forgetting
optimizer = dict(lr=5e-5)

_base_ = [
    '../_base_/models/d3pm_crossattn_uniform_segformer.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

# Pretrained backbone uses BN; smaller batch may require adjustment
# Updated for 1 GPU with batch_size=2
data = dict(samples_per_gpu=2, workers_per_gpu=4)

# Lower LR for pretrained condition encoder to prevent catastrophic forgetting
optimizer = dict(lr=5e-5)

# Epoch-based training (1 epoch = 876 iterations with batch_size=2)
max_epochs = 100
ckpt_epoch_interval = 1      # checkpoint every epoch
eval_epoch_interval = 1      # evaluate every epoch

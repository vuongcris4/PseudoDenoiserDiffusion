_base_ = [
    '../_base_/models/d3pm_concat_uniform.py',
    '../_base_/datasets/pseudo_label_diffusion.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_100k.py',
]

data = dict(samples_per_gpu=4, workers_per_gpu=4)

# W&B logging config
wandb = dict(
    project='pseudo-denoiser-d3pm',
    name='d3pm_concat_uniform_512x512_100k'
)

# Default runtime settings.
log_interval = 50
checkpoint_interval = 5000  # iterations (fallback if epoch-based not set)
eval_interval = 10000       # iterations (fallback if epoch-based not set)

# Epoch-based intervals (preferred, overrides iteration-based if set)
ckpt_epoch_interval = 1     # checkpoint every N epochs
eval_epoch_interval = 1     # evaluate every N epochs

seed = 42
cudnn_benchmark = True
log_dir = 'work_dirs'

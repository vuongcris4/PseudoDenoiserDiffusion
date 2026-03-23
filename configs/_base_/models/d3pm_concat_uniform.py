# D3PM with concatenation conditioning and uniform noise.
#
# Satellite image is concatenated with one-hot noisy label at input level.
# Forward process: uniform transition matrix (any class → any class).

# num_classes is defined in dataset config
# num_classes = 7

model = dict(
    type='DiffusionDenoiserModel',
    num_classes=7,
    num_timesteps=100,
    base_channels=64,
    channel_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(8,),
    dropout=0.1,
    cond_type='concat',
    cond_channels=3,
    cond_base_channels=64,
    transition_type='uniform',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=0.01)

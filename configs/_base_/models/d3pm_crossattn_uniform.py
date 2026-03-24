# D3PM with cross-attention conditioning and uniform noise.
#
# Satellite features are injected via cross-attention in the UNet's
# bottleneck and at attention resolutions. Input is only the noisy label.

num_classes = 8

model = dict(
    type='DiffusionDenoiserModel',
    num_classes=num_classes,
    num_timesteps=100,
    base_channels=64,
    channel_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(8,),
    dropout=0.1,
    cond_type='crossattn',
    cond_channels=3,
    cond_base_channels=64,
    transition_type='uniform',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=1.0)

# D3PM with concatenation conditioning and absorbing noise.
#
# Forward process: absorbing transition (classes → mask state).
# The last class index (K-1) acts as the absorbing "mask" state.
# Useful when the model should learn to "unmask" labels.

num_classes = 7

model = dict(
    type='DiffusionDenoiserModel',
    num_classes=num_classes,
    num_timesteps=100,
    base_channels=64,
    channel_mult=(1, 2, 4, 8),
    num_res_blocks=2,
    attn_resolutions=(8,),
    dropout=0.1,
    cond_type='concat',
    cond_channels=3,
    cond_base_channels=64,
    transition_type='absorbing',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=0.01)

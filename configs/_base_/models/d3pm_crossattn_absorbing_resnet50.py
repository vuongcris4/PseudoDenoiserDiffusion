# D3PM with cross-attention conditioning, absorbing noise,
# and PRETRAINED ResNet-50 condition encoder.
#
# Lighter-weight variant using ResNetV1c-50.  Stem + layer1 frozen,
# layer2-4 finetuned.  Absorbing noise provides mask-state corruption.

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
    cond_encoder_type='pretrained',
    pretrained_cond_cfg=dict(
        backbone_type='resnet50',
        pretrained='open-mmlab://resnet50_v1c',
        freeze_stages=1,                      # Only freeze stem + layer1
        out_channels=[64, 128, 256, 512],
    ),
    transition_type='absorbing',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=0.01)

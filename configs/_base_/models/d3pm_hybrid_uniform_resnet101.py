# D3PM with hybrid conditioning (concat + cross-attention), uniform noise,
# and PRETRAINED ResNet-101 condition encoder.
#
# The satellite image is both concatenated at input and encoded by a
# ResNetV1c-101 for cross-attention injection.  The ResNet stem + layer1-2
# are frozen; layer3-4 are finetuned.  Features are projected from
# [256, 512, 1024, 2048] to [64, 128, 256, 512].

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
    cond_type='hybrid',
    cond_channels=3,
    cond_base_channels=64,                    # Ignored when pretrained
    cond_encoder_type='pretrained',
    pretrained_cond_cfg=dict(
        backbone_type='resnet101',
        pretrained='open-mmlab://resnet101_v1c',  # MMSeg model zoo
        freeze_stages=2,                          # Freeze stem + layer1-2
        out_channels=[64, 128, 256, 512],
    ),
    transition_type='uniform',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=0.1)

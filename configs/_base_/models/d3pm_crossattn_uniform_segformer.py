# D3PM with cross-attention conditioning, uniform noise,
# and PRETRAINED SegFormer-B2 condition encoder.
#
# The satellite image is encoded by a MixVisionTransformer (MiT-B2) whose
# first two stages are frozen (pretrained on ImageNet) and whose last two
# stages are finetuned jointly with the UNet.  Multi-scale features are
# projected to [64, 128, 256, 512] before cross-attention injection.

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
    cond_type='crossattn',
    cond_channels=3,
    cond_base_channels=64,                    # Ignored when pretrained
    cond_encoder_type='pretrained',
    pretrained_cond_cfg=dict(
        backbone_type='segformer_b2',
        pretrained='pretrain/mit_b2.pth',     # SegFormer-B2 checkpoint
        freeze_stages=2,                      # Freeze stages 0-1
        out_channels=[64, 128, 256, 512],     # Projection targets
    ),
    transition_type='uniform',
    beta_schedule='cosine',
    loss_type='hybrid',
    hybrid_lambda=0.01)

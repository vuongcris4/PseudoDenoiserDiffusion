"""Conditional UNet for discrete diffusion-based pseudo-label denoising.

This UNet takes as input:
    - x_t: one-hot encoded noisy label (B, K, H, W)
    - t: diffusion timestep embedding
    - condition: satellite image features

Conditioning is injected via two mechanisms:
    - 'concat': satellite features concatenated with x_t at input
    - 'crossattn': satellite features injected via cross-attention in
      the bottleneck and decoder blocks
    - 'hybrid': both concat and cross-attention

The architecture follows the standard UNet with residual blocks,
timestep modulation, and optional attention at specified resolutions.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Building blocks
# ============================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al.)."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepMLP(nn.Module):
    """Project sinusoidal embedding to model dimension."""

    def __init__(self, t_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)


class ResBlock(nn.Module):
    """Residual block with timestep modulation.

    Features are modulated by the timestep embedding via scale-shift:
        h = GroupNorm(h)
        h = h * (1 + scale) + shift
    where scale, shift are projected from the timestep embedding.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)  # scale and shift
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Timestep modulation: scale-shift
        t_out = self.act(self.t_proj(t_emb))[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial features."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: spatial features attend to condition features.

    Query: spatial features from the UNet (B, C, H, W)
    Key/Value: condition features (B, C_cond, H_c, W_c)
    """

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.GroupNorm(min(32, dim), dim)
        self.norm_kv = nn.GroupNorm(min(32, cond_dim), cond_dim)
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(cond_dim, dim, 1)
        self.v_proj = nn.Conv2d(cond_dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): UNet features (B, C, H, W).
            cond (Tensor): Condition features (B, C_cond, H_c, W_c).

        Returns:
            Tensor: Attended features (B, C, H, W).
        """
        B, C, H, W = x.shape
        nh = self.num_heads

        q = self.q_proj(self.norm_q(x))
        k = self.k_proj(self.norm_kv(cond))
        v = self.v_proj(self.norm_kv(cond))

        # Resize cond spatial dims to match x if needed
        if k.shape[2:] != x.shape[2:]:
            k = F.interpolate(k, size=(H, W), mode='bilinear', align_corners=False)
            v = F.interpolate(v, size=(H, W), mode='bilinear', align_corners=False)

        N = H * W
        q = q.view(B, nh, self.head_dim, N)
        k = k.view(B, nh, self.head_dim, N)
        v = v.view(B, nh, self.head_dim, N)

        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.out_proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ============================================================================
# Condition encoder (lightweight)
# ============================================================================

class ConditionEncoder(nn.Module):
    """Lightweight encoder for satellite image condition.

    Produces multi-scale features for cross-attention injection.
    Uses a simple convolutional stack (no pretrained weights needed,
    but can be replaced with a pretrained backbone).

    Args:
        in_channels (int): Input channels (3 for RGB). Default: 3.
        base_channels (int): Base channel count. Default: 64.
        num_scales (int): Number of output scales. Default: 4.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales
        channels = [base_channels * (2 ** i) for i in range(num_scales)]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, padding=3, stride=2),
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
        )

        self.stages = nn.ModuleList()
        for i in range(1, num_scales):
            self.stages.append(nn.Sequential(
                nn.Conv2d(channels[i - 1], channels[i], 3, stride=2, padding=1),
                nn.GroupNorm(min(32, channels[i]), channels[i]),
                nn.SiLU(),
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.GroupNorm(min(32, channels[i]), channels[i]),
                nn.SiLU(),
            ))

        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (Tensor): Satellite image (B, 3, H, W).

        Returns:
            list[Tensor]: Multi-scale features, from fine to coarse.
        """
        feats = []
        h = self.stem(x)
        feats.append(h)
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


# ============================================================================
# Pretrained condition encoder
# ============================================================================

class PretrainedConditionEncoder(nn.Module):
    """Condition encoder backed by a pretrained backbone (SegFormer or ResNet).

    Extracts multi-scale features from the satellite image using a backbone
    whose early layers are frozen and later layers are finetuned.  A 1x1
    projection head at each scale maps the backbone channel dimensions to a
    uniform target dimension expected by the UNet cross-attention layers.

    Supported backbones:
        - 'segformer_b2': MixVisionTransformer (MiT-B2) from mmseg.
          Produces features at strides 4, 8, 16, 32 with channels
          [64, 128, 320, 512].
        - 'resnet50' / 'resnet101': ResNetV1c from mmseg.
          Produces features at strides 4, 8, 16, 32 with channels
          [256, 512, 1024, 2048].

    Args:
        backbone_type (str): One of 'segformer_b2', 'resnet50', 'resnet101'.
        pretrained (str | None): Path to pretrained checkpoint, or an
            ``open-mmlab://`` model zoo URL for ResNet variants.
            Example: ``'pretrain/mit_b2.pth'``, ``'open-mmlab://resnet101_v1c'``.
        freeze_stages (int): Number of early backbone stages to freeze
            (parameters set to ``requires_grad=False``).  Stage indexing
            starts at 0.  For SegFormer: stages 0-3 correspond to
            ``num_layers`` groups.  For ResNet: stages 0-3 correspond to
            ``layer1``-``layer4``.  Default: 2 (freeze stages 0-1, finetune
            stages 2-3).
        out_channels (list[int] | None): Target channel dimensions for each
            scale after 1x1 projection.  Must have ``num_scales`` elements.
            If ``None``, defaults to ``[64, 128, 256, 512]``.
        num_scales (int): Number of output feature scales.  Default: 4.
        backbone_cfg (dict | None): Extra keyword arguments forwarded to the
            backbone constructor, allowing full customisation from config
            files.  When provided, these override internal defaults.
    """

    # -- Default backbone configurations ------------------------------------
    _BACKBONE_DEFAULTS: Dict[str, dict] = {
        'segformer_b2': dict(
            module_path='mmseg.models.backbones',
            class_name='MixVisionTransformer',
            init_cfg=dict(
                in_channels=3,
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 4, 6, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
            ),
            feature_channels=[64, 128, 320, 512],
        ),
        'resnet50': dict(
            module_path='mmseg.models.backbones',
            class_name='ResNetV1c',
            init_cfg=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            feature_channels=[256, 512, 1024, 2048],
        ),
        'resnet101': dict(
            module_path='mmseg.models.backbones',
            class_name='ResNetV1c',
            init_cfg=dict(
                depth=101,
                in_channels=3,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 1, 1),
                strides=(1, 2, 2, 2),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            feature_channels=[256, 512, 1024, 2048],
        ),
    }

    def __init__(
        self,
        backbone_type: str = 'segformer_b2',
        pretrained: Optional[str] = None,
        freeze_stages: int = 2,
        out_channels: Optional[List[int]] = None,
        num_scales: int = 4,
        backbone_cfg: Optional[dict] = None,
    ):
        super().__init__()

        if backbone_type not in self._BACKBONE_DEFAULTS:
            raise ValueError(
                f'Unsupported backbone_type={backbone_type!r}. '
                f'Choose from {list(self._BACKBONE_DEFAULTS.keys())}.')

        defaults = self._BACKBONE_DEFAULTS[backbone_type]
        self.backbone_type = backbone_type
        self.num_scales = num_scales

        if out_channels is None:
            out_channels = [64, 128, 256, 512]
        assert len(out_channels) == num_scales, (
            f'out_channels length ({len(out_channels)}) must equal '
            f'num_scales ({num_scales}).')
        self.out_channels = out_channels

        feature_channels = defaults['feature_channels'][:num_scales]

        # ---- Build backbone ------------------------------------------------
        self.backbone = self._build_backbone(defaults, backbone_cfg)

        # ---- Load pretrained weights ---------------------------------------
        if pretrained is not None:
            self._load_pretrained(pretrained)

        # ---- Freeze early stages -------------------------------------------
        self._freeze_stages(freeze_stages)

        # ---- 1x1 projection heads ------------------------------------------
        # Map backbone feature channels -> target out_channels for each scale.
        self.projections = nn.ModuleList()
        for i in range(num_scales):
            if feature_channels[i] == out_channels[i]:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(nn.Sequential(
                    nn.Conv2d(feature_channels[i], out_channels[i], 1, bias=False),
                    nn.GroupNorm(min(32, out_channels[i]), out_channels[i]),
                    nn.SiLU(),
                ))

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _build_backbone(defaults: dict,
                        backbone_cfg: Optional[dict]) -> nn.Module:
        """Instantiate the backbone via dynamic import."""
        module_path = defaults['module_path']
        class_name = defaults['class_name']

        try:
            import importlib
            mod = importlib.import_module(module_path)
            BackboneClass = getattr(mod, class_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f'Cannot import {class_name} from {module_path}. '
                f'Make sure mmseg is installed: pip install mmsegmentation. '
                f'Original error: {exc}')

        # Merge default config with user overrides
        cfg = dict(defaults['init_cfg'])
        if backbone_cfg is not None:
            cfg.update(backbone_cfg)

        logger.info(f'Building backbone: {class_name} with config: '
                    f'{list(cfg.keys())}')
        return BackboneClass(**cfg)

    def _load_pretrained(self, pretrained: str) -> None:
        """Load pretrained weights into the backbone.

        Supports:
            - Local file paths (e.g. 'pretrain/mit_b2.pth')
            - MMSeg model zoo URLs (e.g. 'open-mmlab://resnet101_v1c')
        """
        import os
        if pretrained.startswith('open-mmlab://'):
            # Delegate to mmcv's checkpoint loader which resolves model zoo
            try:
                from mmcv.runner import load_checkpoint
                load_checkpoint(self.backbone, pretrained, strict=False,
                                logger=logger)
                logger.info(f'Loaded pretrained weights from {pretrained}')
                return
            except ImportError:
                logger.warning(
                    'mmcv.runner not available; cannot resolve '
                    f'model zoo URL {pretrained}. Skipping.')
                return

        if not os.path.isfile(pretrained):
            logger.warning(f'Pretrained file not found: {pretrained}. '
                           f'Training condition encoder from scratch.')
            return

        ckpt = torch.load(pretrained, map_location='cpu')
        # Handle checkpoints that wrap state_dict in a 'state_dict' key
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # Filter keys that do not belong to the backbone
        backbone_state = self.backbone.state_dict()
        matched, skipped = {}, []
        for k, v in ckpt.items():
            # Strip common prefixes (e.g. 'backbone.' from MMSeg checkpoints)
            clean_key = k
            for prefix in ('backbone.', 'encoder.', 'module.'):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            if clean_key in backbone_state and \
                    v.shape == backbone_state[clean_key].shape:
                matched[clean_key] = v
            else:
                skipped.append(k)

        backbone_state.update(matched)
        self.backbone.load_state_dict(backbone_state, strict=False)
        logger.info(
            f'Loaded pretrained weights from {pretrained}: '
            f'{len(matched)} params loaded, {len(skipped)} skipped.')

    def _freeze_stages(self, freeze_stages: int) -> None:
        """Freeze backbone parameters up to (and including) *freeze_stages*.

        Stage 0 encompasses the stem / patch-embed layers.
        """
        if freeze_stages < 0:
            return  # Nothing to freeze

        if self.backbone_type.startswith('segformer'):
            self._freeze_segformer_stages(freeze_stages)
        elif self.backbone_type.startswith('resnet'):
            self._freeze_resnet_stages(freeze_stages)
        else:
            logger.warning(
                f'Freeze logic not implemented for {self.backbone_type}; '
                f'skipping.')

    def _freeze_segformer_stages(self, freeze_stages: int) -> None:
        """Freeze MixVisionTransformer stages.

        MiT organises parameters as:
            - patch_embed{i+1}  (patch embedding for stage i)
            - block{i+1}        (transformer blocks for stage i)
            - norm{i+1}         (layer norm for stage i)
        """
        frozen = 0
        for stage_idx in range(freeze_stages):
            for attr_prefix in (f'patch_embed{stage_idx + 1}',
                                f'block{stage_idx + 1}',
                                f'norm{stage_idx + 1}'):
                module = getattr(self.backbone, attr_prefix, None)
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                frozen += sum(1 for _ in module.parameters())

        logger.info(f'SegFormer: froze stages 0..{freeze_stages - 1} '
                    f'({frozen} parameters).')

    def _freeze_resnet_stages(self, freeze_stages: int) -> None:
        """Freeze ResNet stages.

        ResNetV1c organises as:
            Stage 0: stem (conv1, bn1) + layer1
            Stage 1: layer2
            Stage 2: layer3
            Stage 3: layer4
        We freeze the stem unconditionally when freeze_stages >= 0,
        then freeze layer{i+1} for each stage i < freeze_stages.
        """
        # Freeze stem
        if freeze_stages >= 0:
            if hasattr(self.backbone, 'conv1'):
                self.backbone.conv1.eval()
                for p in self.backbone.conv1.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, 'norm1'):
                self.backbone.norm1.eval()
                for p in self.backbone.norm1.parameters():
                    p.requires_grad = False

        frozen = 0
        for stage_idx in range(freeze_stages):
            layer_name = f'layer{stage_idx + 1}'
            layer = getattr(self.backbone, layer_name, None)
            if layer is None:
                continue
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
            frozen += sum(1 for _ in layer.parameters())

        logger.info(f'ResNet: froze stem + layers 1..{freeze_stages} '
                    f'({frozen} parameters).')

    # -- Forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features and project to target channels.

        Args:
            x (Tensor): Satellite image (B, 3, H, W).

        Returns:
            list[Tensor]: Projected features at each scale, fine -> coarse.
                Shapes: [(B, out_channels[i], H_i, W_i) for i in num_scales].
        """
        raw_feats = self.backbone(x)  # tuple/list of multi-scale tensors

        projected = []
        for i in range(self.num_scales):
            projected.append(self.projections[i](raw_feats[i]))

        return projected

    def train(self, mode: bool = True):
        """Override train to keep frozen stages in eval mode."""
        super().train(mode)
        # Re-freeze BN/LN in frozen stages
        if mode:
            for module in self.modules():
                # Frozen params already have requires_grad=False.
                # For BN layers with frozen params, force eval mode
                # to use running stats instead of batch stats.
                if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self


# ============================================================================
# Main UNet
# ============================================================================

class ConditionalUNet(nn.Module):
    """Conditional UNet for D3PM pseudo-label denoising.

    Args:
        num_classes (int): Number of segmentation classes K. Default: 7.
        base_channels (int): Base channel count. Default: 128.
        channel_mult (tuple): Channel multiplier at each resolution.
            Default: (1, 2, 4, 8).
        num_res_blocks (int): Residual blocks per resolution. Default: 2.
        attn_resolutions (tuple): Resolutions (as downsampling factors) at
            which to apply self-attention. Default: (2, 4).
        dropout (float): Dropout rate. Default: 0.1.
        cond_type (str): Conditioning type: 'concat', 'crossattn', 'hybrid'.
            Default: 'concat'.
        cond_channels (int): Channels of satellite image. Default: 3.
        cond_base_channels (int): Base channels for condition encoder
            (cross-attention mode). Default: 64.
        cond_encoder_type (str): Condition encoder variant.
            'lightweight': Original ConditionEncoder (custom CNN, no
                pretrained weights).
            'pretrained': PretrainedConditionEncoder backed by a frozen +
                finetuned SegFormer or ResNet backbone.
            Default: 'lightweight'.
        pretrained_cond_cfg (dict | None): Configuration forwarded to
            PretrainedConditionEncoder when cond_encoder_type='pretrained'.
            Expected keys: backbone_type, pretrained, freeze_stages,
            out_channels, backbone_cfg.  Ignored when
            cond_encoder_type='lightweight'.
        t_dim (int): Timestep embedding dimension. Default: 256.
    """

    def __init__(self,
                 num_classes: int = 7,
                 base_channels: int = 128,
                 channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 attn_resolutions: Tuple[int, ...] = (2, 4),
                 dropout: float = 0.1,
                 cond_type: str = 'concat',
                 cond_channels: int = 3,
                 cond_base_channels: int = 64,
                 cond_encoder_type: str = 'lightweight',
                 pretrained_cond_cfg: Optional[dict] = None,
                 t_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.cond_type = cond_type
        self.num_resolutions = len(channel_mult)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(t_dim)
        self.time_mlp = TimestepMLP(t_dim, t_dim)

        # Input channels: K (one-hot) + optional concat condition
        in_ch = num_classes
        if cond_type in ('concat', 'hybrid'):
            in_ch += cond_channels

        # Condition encoder for cross-attention
        if cond_type in ('crossattn', 'hybrid'):
            if cond_encoder_type == 'pretrained':
                pcfg = pretrained_cond_cfg or {}
                self.cond_encoder = PretrainedConditionEncoder(
                    backbone_type=pcfg.get('backbone_type', 'segformer_b2'),
                    pretrained=pcfg.get('pretrained', None),
                    freeze_stages=pcfg.get('freeze_stages', 2),
                    out_channels=pcfg.get('out_channels', None),
                    num_scales=len(channel_mult),
                    backbone_cfg=pcfg.get('backbone_cfg', None),
                )
            else:
                # Default: lightweight custom CNN encoder
                self.cond_encoder = ConditionEncoder(
                    cond_channels, cond_base_channels, len(channel_mult))
            cond_enc_channels = self.cond_encoder.out_channels
        else:
            self.cond_encoder = None
            cond_enc_channels = [0] * len(channel_mult)

        # Compute channel dims at each resolution
        channels = [base_channels * m for m in channel_mult]

        # Input projection
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ---- Encoder ----
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        enc_channels = []  # Track skip connection channels

        for level in range(self.num_resolutions):
            ch = channels[level]

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                # First block of each level receives input from:
                # - level 0: input_conv output (channels[0])
                # - level > 0: downsample from previous level (channels[level-1])
                # Subsequent blocks receive output from previous block (ch)
                if i == 0:
                    block_in = channels[0] if level == 0 else channels[level - 1]
                else:
                    block_in = ch
                level_blocks.append(ResBlock(block_in, ch, t_dim, dropout))

                # Self-attention at specified resolutions
                ds_factor = 2 ** level
                if ds_factor in attn_resolutions:
                    level_blocks.append(SelfAttention(ch))

                # Cross-attention with condition
                if cond_type in ('crossattn', 'hybrid') and ds_factor in attn_resolutions:
                    level_blocks.append(
                        CrossAttention(ch, cond_enc_channels[level]))

            self.encoder_blocks.append(level_blocks)
            enc_channels.append(ch)  # Push once per level for skip connection

            # Downsample (except last level)
            if level < self.num_resolutions - 1:
                self.encoder_downsamples.append(Downsample(ch))
            else:
                self.encoder_downsamples.append(nn.Identity())

        # ---- Bottleneck ----
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, t_dim, dropout)
        self.mid_attn = SelfAttention(mid_ch)
        if cond_type in ('crossattn', 'hybrid'):
            self.mid_cross_attn = CrossAttention(
                mid_ch, cond_enc_channels[-1])
        else:
            self.mid_cross_attn = None
        self.mid_block2 = ResBlock(mid_ch, mid_ch, t_dim, dropout)

        # ---- Decoder ----
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_projections = nn.ModuleList()

        # Decoder forward iterates: for level_idx, level in enumerate(reversed(range(num_resolutions)))
        # So we store decoder items in the same order as forward will iterate
        # decoder_items[0] = highest level (level N-1), decoder_items[N-1] = lowest level (level 0)

        for level in reversed(range(self.num_resolutions)):
            ch = channels[level]

            level_blocks = nn.ModuleList()

            # Projection
            prev_ch = mid_ch if level == self.num_resolutions - 1 else channels[level + 1]
            proj = nn.Conv2d(prev_ch, ch, 1) if prev_ch != ch else nn.Identity()
            self.decoder_projections.append(proj)

            skip_ch = enc_channels.pop()
            level_blocks.append(ResBlock(ch + skip_ch, ch, t_dim, dropout))

            # Additional res blocks
            for i in range(num_res_blocks - 1):
                level_blocks.append(ResBlock(ch, ch, t_dim, dropout))

            # Attention
            ds_factor = 2 ** level
            if ds_factor in attn_resolutions:
                level_blocks.append(SelfAttention(ch))

            if cond_type in ('crossattn', 'hybrid') and ds_factor in attn_resolutions:
                level_blocks.append(
                    CrossAttention(ch, cond_enc_channels[level]))

            self.decoder_blocks.append(level_blocks)

            upsample = Upsample(ch) if level > 0 else nn.Identity()
            self.decoder_upsamples.append(upsample)

        # Output projection
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], num_classes, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t (Tensor): One-hot noisy label (B, K, H, W).
            t (Tensor): Timestep indices (B,).
            condition (Tensor): Satellite image (B, 3, H, W).

        Returns:
            Tensor: Predicted x_0 logits (B, K, H, W).
        """
        # Timestep embedding
        t_emb = self.time_mlp(self.time_embed(t))

        # Condition features for cross-attention
        cond_feats = None
        if self.cond_encoder is not None:
            cond_feats = self.cond_encoder(condition)

        # Input: concatenate condition if needed
        if self.cond_type in ('concat', 'hybrid'):
            # Resize condition to match x_t spatial dims
            cond_resized = F.interpolate(
                condition, size=x_t.shape[2:],
                mode='bilinear', align_corners=False)
            h = torch.cat([x_t, cond_resized], dim=1)
        else:
            h = x_t

        h = self.input_conv(h)

        # ---- Encoder ----
        skips = []
        for level in range(self.num_resolutions):
            blocks = self.encoder_blocks[level]
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emb)
                elif isinstance(block, CrossAttention):
                    h = block(h, cond_feats[level])
                else:
                    h = block(h)
            skips.append(h)

            if level < self.num_resolutions - 1:
                h = self.encoder_downsamples[level](h)

        # ---- Bottleneck ----
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        if self.mid_cross_attn is not None:
            h = self.mid_cross_attn(h, cond_feats[-1])
        h = self.mid_block2(h, t_emb)

        # ---- Decoder ----
        for level_idx, level in enumerate(
                reversed(range(self.num_resolutions))):
            # Project h to match current level channels
            h = self.decoder_projections[level_idx](h)

            blocks = self.decoder_blocks[level_idx]
            for i, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    if i == 0:
                        skip = skips.pop()
                        h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                elif isinstance(block, CrossAttention):
                    h = block(h, cond_feats[level])
                else:
                    h = block(h)

            if level > 0:
                h = self.decoder_upsamples[level_idx](h)

        # Output
        h = F.silu(self.out_norm(h))
        logits = self.out_conv(h)

        return logits

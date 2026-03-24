"""DiffusionDenoiserModel: top-level wrapper combining UNet + D3PM.

This is the main model class that is instantiated from config. It builds
the conditional UNet, the discrete noise schedule, and the D3PM diffusion
process, then exposes train_step() and sample() interfaces.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..diffusion.d3pm import D3PM
from ..diffusion.noise_schedule import DiscreteNoiseSchedule
from .conditional_unet import ConditionalUNet


class DiffusionDenoiserModel(nn.Module):
    """Full diffusion denoiser model.

    Builds and connects:
        1. ConditionalUNet (denoising network)
        2. DiscreteNoiseSchedule (transition matrices)
        3. D3PM (training and sampling logic)

    Args:
        num_classes (int): Number of segmentation classes.
        num_timesteps (int): Diffusion timesteps T.
        base_channels (int): UNet base channel count.
        channel_mult (tuple): UNet channel multipliers.
        num_res_blocks (int): Residual blocks per resolution.
        attn_resolutions (tuple): Resolutions for attention.
        dropout (float): Dropout rate.
        cond_type (str): 'concat', 'crossattn', or 'hybrid'.
        cond_channels (int): Satellite image channels.
        cond_encoder_type (str): 'lightweight' (default CNN) or
            'pretrained' (SegFormer / ResNet with frozen early stages).
        pretrained_cond_cfg (dict | None): Config for
            PretrainedConditionEncoder.  Only used when
            cond_encoder_type='pretrained'.
        transition_type (str): D3PM noise type: 'uniform' or 'absorbing'.
        beta_schedule (str): 'linear' or 'cosine'.
        loss_type (str): 'kl', 'ce', or 'hybrid'.
        hybrid_lambda (float): CE weight in hybrid loss.
    """

    def __init__(self,
                 num_classes: int = 7,
                 num_timesteps: int = 100,
                 base_channels: int = 128,
                 channel_mult: tuple = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 attn_resolutions: tuple = (2, 4),
                 dropout: float = 0.1,
                 cond_type: str = 'concat',
                 cond_channels: int = 3,
                 cond_base_channels: int = 64,
                 cond_encoder_type: str = 'lightweight',
                 pretrained_cond_cfg: dict = None,
                 transition_type: str = 'uniform',
                 beta_schedule: str = 'cosine',
                 loss_type: str = 'hybrid',
                 hybrid_lambda: float = 0.01):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Build components
        unet = ConditionalUNet(
            num_classes=num_classes,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            cond_type=cond_type,
            cond_channels=cond_channels,
            cond_base_channels=cond_base_channels,
            cond_encoder_type=cond_encoder_type,
            pretrained_cond_cfg=pretrained_cond_cfg,
        )

        noise_schedule = DiscreteNoiseSchedule(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            transition_type=transition_type,
            beta_schedule=beta_schedule,
        )

        self.d3pm = D3PM(
            denoise_model=unet,
            noise_schedule=noise_schedule,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            loss_type=loss_type,
            hybrid_lambda=hybrid_lambda,
        )

    def forward(self, clean_label: torch.Tensor,
                satellite_img: torch.Tensor,
                pseudo_label: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward.

        Args:
            clean_label (Tensor): Clean GT labels (B, H, W).
                Used as target for loss computation.
            satellite_img (Tensor): Satellite image (B, 3, H, W).
                Used as condition for denoising.
            pseudo_label (Tensor, optional): Noisy pseudo-labels (B, H, W).
                If provided, forward diffusion starts from pseudo-label.
                If None, defaults to clean_label (self-training mode).

        Returns:
            dict: Loss dictionary.
        """
        return self.d3pm(clean_label, satellite_img, x_init=pseudo_label)

    @torch.no_grad()
    def denoise(self, satellite_img: torch.Tensor,
                noisy_label: Optional[torch.Tensor] = None,
                num_steps: Optional[int] = None,
                temperature: float = 1.0) -> torch.Tensor:
        """Denoise pseudo-labels via reverse diffusion.

        Args:
            satellite_img (Tensor): Satellite image (B, 3, H, W).
            noisy_label (Tensor, optional): Initial noisy label (B, H, W).
            num_steps (int, optional): Override number of denoising steps.
            temperature (float): Sampling temperature.

        Returns:
            Tensor: Denoised labels (B, H, W).
        """
        return self.d3pm.sample(
            satellite_img, noisy_label, num_steps, temperature)

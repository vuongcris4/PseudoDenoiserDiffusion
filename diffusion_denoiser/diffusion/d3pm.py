"""D3PM: Discrete Denoising Diffusion for Pseudo-label Refinement.

Reference:
    Austin et al., "Structured Denoising Diffusion Models in Discrete
    State-Spaces", NeurIPS 2021.

This module implements the full D3PM pipeline for conditional pseudo-label
denoising. Given a noisy pseudo-label x_T (from a poorly trained model)
and a satellite image as condition, the model iteratively denoises x_T
through T reverse steps to produce a clean label x_0.

Training objective:
    L_t = KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t, cond))

The model p_θ predicts the clean label x_0 (x0-parameterization),
from which the reverse posterior is computed analytically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .noise_schedule import DiscreteNoiseSchedule


class D3PM(nn.Module):
    """Discrete Denoising Diffusion Probabilistic Model.

    The model predicts x_0 from x_t and condition, then uses the analytical
    posterior q(x_{t-1} | x_t, x_0_pred) to sample x_{t-1}.

    Args:
        denoise_model (nn.Module): Neural network that takes
            (x_t_onehot, t_emb, condition) and outputs x_0 logits
            of shape (B, num_classes, H, W).
        noise_schedule (DiscreteNoiseSchedule): Precomputed transition matrices.
        num_classes (int): Number of segmentation classes K.
        num_timesteps (int): Total diffusion steps T.
        loss_type (str): Loss function type.
            'kl': KL divergence between true and predicted posterior.
            'ce': Cross-entropy on x_0 prediction.
            'hybrid': Weighted combination of KL and CE.
            Default: 'hybrid'.
        hybrid_lambda (float): Weight for CE term in hybrid loss. Default: 0.01.
        parameterization (str): What the model predicts.
            'x0': predict clean x_0 directly.
            Default: 'x0'.
    """

    def __init__(self,
                 denoise_model: nn.Module,
                 noise_schedule: DiscreteNoiseSchedule,
                 num_classes: int = 8,
                 num_timesteps: int = 100,
                 loss_type: str = 'hybrid',
                 hybrid_lambda: float = 0.01,
                 parameterization: str = 'x0',
                 ignore_index: int = 255):
        super().__init__()
        self.denoise_model = denoise_model
        self.noise_schedule = noise_schedule
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.hybrid_lambda = hybrid_lambda
        self.parameterization = parameterization
        self.ignore_index = ignore_index

    def forward(self, x_0: torch.Tensor, condition: torch.Tensor,
                x_init: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x_0 (Tensor): Clean labels / ground truth (B, H, W), class indices.
                Used as target for loss computation.
            condition (Tensor): Satellite image (B, 3, H, W).
            x_init (Tensor, optional): Initial noisy labels (pseudo-labels)
                (B, H, W), class indices. If None, defaults to x_0.
                Used as starting point for forward diffusion.

        Returns:
            dict: Loss dictionary with 'loss_total' and component losses.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Forward diffusion: sample x_t ~ q(x_t | x_init)
        # If x_init (pseudo_label) is provided, noise from pseudo-label
        # Otherwise, noise from ground truth (self-training mode)
        # Clamp to valid range (ignore pixels may have value 255)
        source = x_init if x_init is not None else x_0
        source = source.clamp(0, self.num_classes - 1)
        x_t = self.noise_schedule.q_sample(source, t)

        # Model predicts x_0 logits from x_t and condition
        x_0_logits = self._predict_x0(x_t, t, condition)  # (B, K, H, W)

        # Compute loss (always against ground truth x_0)
        losses = self._compute_loss(x_0, x_t, x_0_logits, t)

        return losses

    def _predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                    condition: torch.Tensor) -> torch.Tensor:
        """Run the denoising model to predict x_0 logits.

        Args:
            x_t (Tensor): Noisy labels (B, H, W), class indices.
            t (Tensor): Timestep indices (B,).
            condition (Tensor): Satellite image (B, 3, H, W).

        Returns:
            Tensor: Predicted x_0 logits (B, K, H, W).
        """
        # Clamp x_t to valid range (ignore pixels may have value 255)
        x_t_safe = x_t.long().clamp(0, self.num_classes - 1)
        # Convert x_t to one-hot: (B, K, H, W)
        x_t_onehot = F.one_hot(
            x_t_safe, self.num_classes).permute(0, 3, 1, 2).float()

        # Forward through denoising network
        x_0_logits = self.denoise_model(x_t_onehot, t, condition)

        return x_0_logits

    def _compute_loss(self, x_0: torch.Tensor, x_t: torch.Tensor,
                      x_0_logits: torch.Tensor,
                      t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the training loss.

        Args:
            x_0 (Tensor): Clean labels (B, H, W).
            x_t (Tensor): Noisy labels (B, H, W).
            x_0_logits (Tensor): Predicted x_0 logits (B, K, H, W).
            t (Tensor): Timestep indices (B,).

        Returns:
            dict: Loss components.
        """
        losses = {}

        # Cross-entropy loss on x_0 prediction (ignoring nodata pixels)
        ce_loss = F.cross_entropy(
            x_0_logits, x_0.long(), reduction='mean',
            ignore_index=self.ignore_index)
        losses['loss_ce'] = ce_loss

        if self.loss_type == 'ce':
            losses['loss_total'] = ce_loss
            return losses

        # KL divergence loss on reverse process
        kl_loss = self._kl_loss(x_0, x_t, x_0_logits, t)
        losses['loss_kl'] = kl_loss

        if self.loss_type == 'kl':
            losses['loss_total'] = kl_loss
        elif self.loss_type == 'hybrid':
            losses['loss_total'] = kl_loss + self.hybrid_lambda * ce_loss
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        return losses

    def _kl_loss(self, x_0: torch.Tensor, x_t: torch.Tensor,
                 x_0_logits: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """Compute KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)).

        Args:
            x_0 (Tensor): Clean labels (B, H, W).
            x_t (Tensor): Noisy labels (B, H, W).
            x_0_logits (Tensor): Predicted x_0 logits (B, K, H, W).
            t (Tensor): Timestep indices (B,).

        Returns:
            Tensor: Scalar KL loss.
        """
        # True posterior: q(x_{t-1} | x_t, x_0)
        true_posterior = self.noise_schedule.q_posterior(
            x_0, x_t, t)  # (B, H, W, K)

        # Predicted posterior: use predicted x_0 to compute posterior
        x_0_pred = x_0_logits.argmax(dim=1)  # (B, H, W)
        # Straight-through: use soft probs for gradient flow
        x_0_probs = F.softmax(x_0_logits, dim=1)  # (B, K, H, W)

        # Compute predicted posterior using soft x_0 probabilities
        pred_posterior = self._soft_posterior(
            x_0_probs, x_t, t)  # (B, H, W, K)

        # KL divergence
        true_posterior = true_posterior.clamp(min=1e-10)
        pred_posterior = pred_posterior.clamp(min=1e-10)

        kl = true_posterior * (
            torch.log(true_posterior) - torch.log(pred_posterior))
        kl = kl.sum(dim=-1)  # Sum over classes → (B, H, W)

        # Mask out ignore pixels from KL loss
        valid_mask = (x_0 != self.ignore_index)
        if valid_mask.any():
            kl = kl[valid_mask].mean()
        else:
            kl = kl.mean() * 0.0  # no valid pixels, return 0

        return kl

    def _soft_posterior(self, x_0_probs: torch.Tensor, x_t: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        """Compute predicted posterior using soft x_0 probabilities.

        p_θ(x_{t-1} | x_t) = Σ_{x_0} q(x_{t-1} | x_t, x_0) * p_θ(x_0 | x_t)

        Args:
            x_0_probs (Tensor): Predicted x_0 probs (B, K, H, W).
            x_t (Tensor): Noisy labels (B, H, W).
            t (Tensor): Timestep indices (B,).

        Returns:
            Tensor: Predicted posterior (B, H, W, K).
        """
        B, K, H, W = x_0_probs.shape

        # Get transition matrices
        Q_t_cur = self.noise_schedule.Q_t[t]  # (B, K, K)
        t_minus_1 = (t - 1).clamp(min=0)
        Q_bar_prev = self.noise_schedule.Q_bar[t_minus_1]  # (B, K, K)
        is_t0 = (t == 0).float().view(B, 1, 1)
        Q_bar_prev = is_t0 * torch.eye(K, device=x_t.device).unsqueeze(0) + \
                     (1 - is_t0) * Q_bar_prev

        # x_t one-hot: (B, H*W, K)
        x_t_oh = F.one_hot(x_t.long(), K).float().view(B, H * W, K)

        # For each possible x_0 value, compute q(x_{t-1} | x_t, x_0=k)
        # Then weight by p_θ(x_0=k)
        # Efficient: marginalize over x_0

        # q(x_t | x_{t-1}=j): Q_t[j, x_t] → (B, H*W, K) indexed by j
        prob_xt_given_xtm1 = torch.bmm(
            Q_t_cur, x_t_oh.transpose(1, 2)).transpose(1, 2)  # (B, H*W, K)

        # q(x_{t-1}=j | x_0=k): Q_bar_prev[k, j] → need to sum over k
        # weighted by p_θ(x_0=k)
        # p_θ(x_0): (B, K, H*W) → (B, H*W, K)
        x_0_probs_flat = x_0_probs.view(B, K, H * W).permute(0, 2, 1)  # (B, H*W, K)

        # Σ_k p(x_0=k) * Q_bar_prev[k, j] = x_0_probs @ Q_bar_prev → (B, H*W, K)
        prob_xtm1 = torch.bmm(x_0_probs_flat, Q_bar_prev.transpose(1, 2))

        # Posterior: ∝ q(x_t | x_{t-1}=j) * Σ_k p(x_0=k) * q(x_{t-1}=j | x_0=k)
        unnorm = prob_xt_given_xtm1 * prob_xtm1
        posterior = unnorm / (unnorm.sum(dim=-1, keepdim=True) + 1e-10)

        return posterior.view(B, H, W, K)

    @torch.no_grad()
    def sample(self, condition: torch.Tensor,
               noisy_label: Optional[torch.Tensor] = None,
               num_steps: Optional[int] = None,
               temperature: float = 1.0) -> torch.Tensor:
        """Reverse diffusion sampling.

        Args:
            condition (Tensor): Satellite image (B, 3, H, W).
            noisy_label (Tensor, optional): Initial noisy pseudo-label (B, H, W).
                If None, start from uniform random or absorbing state.
            num_steps (int, optional): Number of denoising steps.
                If None, use self.num_timesteps.
            temperature (float): Sampling temperature. Default: 1.0.

        Returns:
            Tensor: Denoised labels (B, H, W).
        """
        B, _, H, W = condition.shape
        device = condition.device
        T = num_steps or self.num_timesteps

        # Initialize x_T
        if noisy_label is not None:
            x_t = noisy_label.long().clamp(0, self.num_classes - 1)
        else:
            # Start from uniform random
            x_t = torch.randint(0, self.num_classes, (B, H, W), device=device)

        # Reverse process: t = T-1, T-2, ..., 0
        for t_val in reversed(range(T)):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Predict x_0
            x_0_logits = self._predict_x0(x_t, t, condition)
            x_0_probs = F.softmax(x_0_logits / temperature, dim=1)

            if t_val == 0:
                # At t=0, directly take the argmax prediction
                x_t = x_0_logits.argmax(dim=1)
            else:
                # Compute predicted posterior and sample
                posterior = self._soft_posterior(x_0_probs, x_t, t)  # (B, H, W, K)
                x_t = torch.multinomial(
                    posterior.view(-1, self.num_classes),
                    num_samples=1).view(B, H, W)

        return x_t

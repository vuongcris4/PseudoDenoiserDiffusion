"""Utility to log model parameter counts (total / trainable / frozen)."""

import torch.nn as nn
from typing import Callable, Optional


def _fmt(n: int) -> str:
    """Format a large number with M/K suffix."""
    if n >= 1_000_000:
        return f'{n / 1e6:.2f}M'
    elif n >= 1_000:
        return f'{n / 1e3:.1f}K'
    return str(n)


def log_model_params(
    model: nn.Module,
    logger_fn: Callable[[str], None] = print,
    wandb_run=None,
):
    """Print a per-component table of parameter counts and optionally log to W&B.

    Args:
        model: PyTorch model.
        logger_fn: Callable to print lines (default: print).
        wandb_run: Active wandb.run object to log config. Pass None to skip.
    """
    # ── Per top-level component breakdown ────────────────────────────────
    components = {}
    for name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        frozen = total - trainable
        components[name] = (total, trainable, frozen)

        # Second-level sub-modules (e.g. d3pm.denoise_model, d3pm.noise_schedule)
        for sub_name, sub_child in child.named_children():
            sub_total = sum(p.numel() for p in sub_child.parameters())
            sub_train = sum(p.numel() for p in sub_child.parameters() if p.requires_grad)
            sub_frozen = sub_total - sub_train
            components[f'  {name}.{sub_name}'] = (sub_total, sub_train, sub_frozen)

            # Third-level (e.g. d3pm.denoise_model.cond_encoder)
            for sub2_name, sub2_child in sub_child.named_children():
                s2_total = sum(p.numel() for p in sub2_child.parameters())
                s2_train = sum(p.numel() for p in sub2_child.parameters() if p.requires_grad)
                s2_frozen = s2_total - s2_train
                if s2_total > 0:
                    components[f'    {name}.{sub_name}.{sub2_name}'] = (s2_total, s2_train, s2_frozen)

    # ── Grand totals ────────────────────────────────────────────────────
    grand_total = sum(p.numel() for p in model.parameters())
    grand_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grand_frozen = grand_total - grand_train

    # ── Print table ─────────────────────────────────────────────────────
    w = 65
    logger_fn('')
    logger_fn('═' * w)
    logger_fn(f'{"Component":<40} {"Total":>8} {"Train":>8} {"Frozen":>8}')
    logger_fn('─' * w)
    for comp, (t, tr, fr) in components.items():
        if t > 0:
            logger_fn(f'{comp:<40} {_fmt(t):>8} {_fmt(tr):>8} {_fmt(fr):>8}')
    logger_fn('─' * w)
    logger_fn(f'{"TOTAL":<40} {_fmt(grand_total):>8} {_fmt(grand_train):>8} {_fmt(grand_frozen):>8}')
    logger_fn('═' * w)
    logger_fn('')

    # ── W&B config ──────────────────────────────────────────────────────
    if wandb_run is not None:
        wandb_run.config.update({
            'params/total': grand_total,
            'params/trainable': grand_train,
            'params/frozen': grand_frozen,
        }, allow_val_change=True)

    return dict(total=grand_total, trainable=grand_train, frozen=grand_frozen)

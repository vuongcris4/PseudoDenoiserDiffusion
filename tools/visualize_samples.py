"""Visualize dataset samples: satellite image, pseudo-label, and ground-truth label.

Usage:
    python tools/visualize_samples.py configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py
    python tools/visualize_samples.py configs/denoiser/d3pm_crossattn_uniform_segformer_512x512_100k.py --num 8 --split val
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from diffusion_denoiser.utils.config import Config
from diffusion_denoiser.datasets.oem_ciscr_dataset import OEMCISCRCrossEntropyDataset

from diffusion_denoiser.datasets.oem_classes import (
    CLASS_NAMES, CLASS_COLORS, NODATA_COLOR, colorize_mask,
)


def compute_diff(pseudo: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Create a diff map: green = agree, red = disagree."""
    h, w = pseudo.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    agree = pseudo == clean
    diff[agree] = [76, 175, 80]      # green
    diff[~agree] = [244, 67, 54]     # red
    return diff


def parse_args():
    p = argparse.ArgumentParser(description='Visualize dataset samples')
    p.add_argument('config', help='Config file path')
    p.add_argument('--num', type=int, default=4, help='Number of samples to show')
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--save', default=None, help='Save figure to path (e.g. vis.png)')
    p.add_argument('--no-show', action='store_true', help='Do not call plt.show()')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Build dataset (no augmentation for visualization)
    split_cfg = cfg.data.get(args.split, cfg.data.train)
    ds = OEMCISCRCrossEntropyDataset(
        data_root=split_cfg.data_root,
        split=args.split,
        img_size=split_cfg.get('img_size', 512),
        augment=False,
        num_classes=split_cfg.get('num_classes', 7),
    )

    num = min(args.num, len(ds))
    np.random.seed(args.seed)
    indices = np.random.choice(len(ds), num, replace=False)

    # ── Plot grid: 4 columns per sample ─────────────────────────────────
    # Columns: Image | Pseudo-label | Ground-truth | Diff (pseudo vs gt)
    n_cols = 4
    fig, axes = plt.subplots(num, n_cols, figsize=(n_cols * 4, num * 4))
    if num == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Satellite Image', 'Pseudo Label', 'Ground Truth', 'Diff (Pseudo vs GT)']

    for row, idx in enumerate(indices):
        sample = ds[idx]
        img = sample['satellite_img'].numpy().transpose(1, 2, 0)   # (H,W,3) [0,1]
        pseudo = sample['pseudo_label'].numpy()                     # (H,W) int
        clean = sample['clean_label'].numpy()                       # (H,W) int
        fname = sample['filename']

        pseudo_rgb = colorize_mask(pseudo)
        clean_rgb = colorize_mask(clean)
        diff_rgb = compute_diff(pseudo, clean)

        # Compute agreement stats
        total_px = pseudo.size
        agree_px = (pseudo == clean).sum()
        agree_pct = 100.0 * agree_px / total_px

        panels = [img, pseudo_rgb, clean_rgb, diff_rgb]
        for col, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(f'{fname}\n(idx {idx})', fontsize=9, rotation=0,
                              labelpad=80, va='center')
            if col == 3:
                ax.text(0.98, 0.02, f'{agree_pct:.1f}% agree',
                        transform=ax.transAxes, fontsize=10,
                        ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Legend ───────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=CLASS_COLORS[i] / 255.0, label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=len(CLASS_NAMES),
               fontsize=10, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f'Dataset Samples — {args.split} split ({num} samples)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if args.save:
        os.makedirs(osp.dirname(args.save) or '.', exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved to {args.save}')

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()

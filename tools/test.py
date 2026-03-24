"""Evaluation script for D3PM pseudo-label denoiser.

Computes mIoU, per-class IoU, and also reports the baseline mIoU
of the raw pseudo-labels (before denoising) for comparison.
Logs visual inference results to W&B as an image table.

When called from train.py (auto-eval), it reuses the active W&B run.
When called standalone, it resumes the same W&B run saved in the checkpoint
so that training + evaluation stay within a single experiment.

Usage:
    python tools/test.py \\
        configs/denoiser/d3pm_concat_uniform_512x512_100k.py \\
        work_dirs/d3pm_concat_uniform/latest.pth \\
        --num-steps 50

    # Without W&B logging
    python tools/test.py ... --no-wandb

    # Limit number of visualized samples (default: all)
    python tools/test.py ... --max-vis 20
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from diffusion_denoiser.utils.config import Config
from diffusion_denoiser.utils.param_utils import log_model_params
from diffusion_denoiser.models.diffusion_denoiser import DiffusionDenoiserModel
from diffusion_denoiser.datasets.pseudo_label_dataset import PseudoLabelDiffusionDataset
from diffusion_denoiser.datasets.oem_ciscr_dataset import OEMCISCRCrossEntropyDataset

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Registry for dataset classes (mirroring train.py)
DATASET_REGISTRY = {
    'PseudoLabelDiffusionDataset': PseudoLabelDiffusionDataset,
    'OEMCISCRCrossEntropyDataset': OEMCISCRCrossEntropyDataset,
}

# ── Colormap for 7 OEM classes ──────────────────────────────────────────────
# 0=Bareland, 1=Rangeland, 2=Developed, 3=Road, 4=Tree, 5=Water, 6=Agriculture
CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture',
]

CLASS_COLORS = np.array([
    [128, 128, 128],   # 0 Bareland     - gray
    [0, 255, 0],       # 1 Rangeland    - green
    [255, 0, 0],       # 2 Developed    - red
    [255, 255, 0],     # 3 Road         - yellow
    [0, 128, 0],       # 4 Tree         - dark green
    [0, 0, 255],       # 5 Water        - blue
    [255, 165, 0],     # 6 Agriculture  - orange
], dtype=np.uint8)


# ── Utility functions ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate D3PM denoiser')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--max-vis', type=int, default=None,
                        help='Max samples to visualize in W&B table '
                             '(default: all)')
    return parser.parse_args()


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class-index mask (H, W) to RGB (H, W, 3) using CLASS_COLORS."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(CLASS_COLORS))):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


def compute_miou(pred, gt, num_classes, ignore_index=255):
    """Compute per-class IoU."""
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for c in range(num_classes):
        valid = gt != ignore_index
        pred_c = (pred == c) & valid
        gt_c = (gt == c) & valid
        intersection[c] = (pred_c & gt_c).sum()
        union[c] = (pred_c | gt_c).sum()
    iou = intersection / (union + 1e-10)
    return iou


def denormalize_satellite(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert satellite image tensor back to uint8 RGB for visualization.

    Handles two normalization conventions:
      - [0, 1] range (OEMCISCRCrossEntropyDataset)
      - ImageNet mean/std (PseudoLabelDiffusionDataset)
    Autodetects based on value range.
    """
    img = img_tensor.cpu().numpy()  # (3, H, W)
    img = img.transpose(1, 2, 0)   # (H, W, 3)

    if img.max() <= 1.5:
        # Already in [0,1] range
        img = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        # ImageNet normalization: undo mean/std
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = (img * std + mean).clip(0, 255).astype(np.uint8)

    return img


def create_overlay(satellite_rgb: np.ndarray, mask_rgb: np.ndarray,
                   alpha: float = 0.45) -> np.ndarray:
    """Blend satellite image with colorized mask."""
    return cv2.addWeighted(satellite_rgb, 1 - alpha, mask_rgb, alpha, 0)


def create_diff_map(pseudo: np.ndarray, pred: np.ndarray,
                    clean: np.ndarray) -> np.ndarray:
    """Create a diff image showing where prediction differs from pseudo-label.

    Green = prediction fixed an error (pseudo wrong, pred correct)
    Red   = prediction introduced an error (pseudo correct, pred wrong)
    Gray  = both same or both wrong
    """
    h, w = pseudo.shape
    diff = np.full((h, w, 3), 180, dtype=np.uint8)  # gray background

    pseudo_correct = (pseudo == clean)
    pred_correct = (pred == clean)

    # Green: pseudo wrong → pred correct (improvement)
    improved = (~pseudo_correct) & pred_correct
    diff[improved] = [0, 200, 0]

    # Red: pseudo correct → pred wrong (regression)
    regressed = pseudo_correct & (~pred_correct)
    diff[regressed] = [200, 0, 0]

    # White: both correct
    both_correct = pseudo_correct & pred_correct
    diff[both_correct] = [255, 255, 255]

    return diff


# ── Core evaluation function (callable from train.py or standalone) ─────────

def run_evaluation(model, cfg, device, checkpoint_path=None,
                   use_wandb=False, num_steps=None, temperature=1.0,
                   max_vis=None):
    """Run full evaluation with visualization and optional W&B logging.

    This function can be called:
      - From train.py (auto-eval): model already loaded, W&B already active
      - Standalone via main(): model built from checkpoint

    Args:
        model: Loaded DiffusionDenoiserModel (already on device, eval mode).
        cfg: Config object.
        device: torch.device.
        checkpoint_path: Path to checkpoint (for metadata).
        use_wandb: Whether to log to W&B (assumes run is active).
        num_steps: Override denoising steps.
        temperature: Sampling temperature.
        max_vis: Max samples to visualize in W&B table.
    """
    model.eval()
    num_classes = cfg.model.num_classes

    # Load checkpoint metadata
    ckpt_iter = 'unknown'
    if checkpoint_path and osp.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_iter = ckpt.get('iter', 'unknown')

    # ── Build test dataset ──────────────────────────────────────────────
    test_cfg = cfg.data.test.copy()
    dataset_type = test_cfg.pop('type', 'OEMCISCRCrossEntropyDataset')
    if dataset_type in DATASET_REGISTRY:
        test_dataset = DATASET_REGISTRY[dataset_type](**test_cfg)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2)

    if max_vis is None:
        max_vis = len(test_dataset)

    # ── Evaluate ────────────────────────────────────────────────────────
    all_iou_pred = []
    all_iou_pseudo = []

    # W&B table: columns for visual comparison
    vis_table = None
    if use_wandb and HAS_WANDB:
        vis_table = wandb.Table(columns=[
            'Filename',
            'Satellite',
            'Pseudo-Label',
            'Clean GT',
            'Denoised Output',
            'Diff Map',
            'Pseudo Overlay',
            'Denoised Overlay',
            'mIoU_Pseudo',
            'mIoU_Denoised',
            'Δ mIoU',
        ])

    for i, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
        satellite = batch['satellite_img'].to(device)
        pseudo = batch['pseudo_label'].to(device)
        clean = batch['clean_label'].numpy()[0]
        filename = batch.get('filename', [f'sample_{i}'])[0]

        # Denoise
        with torch.no_grad():
            pred = model.denoise(
                satellite, pseudo,
                num_steps=num_steps,
                temperature=temperature)
            pred_np = pred.squeeze(0).cpu().numpy()

        pseudo_np = pseudo.squeeze(0).cpu().numpy()

        # Compute IoU
        iou_pred = compute_miou(pred_np, clean, num_classes)
        all_iou_pred.append(iou_pred)

        iou_pseudo = compute_miou(pseudo_np, clean, num_classes)
        all_iou_pseudo.append(iou_pseudo)

        # ── Log visual to W&B table ─────────────────────────────────
        if vis_table is not None and i < max_vis:
            sat_rgb = denormalize_satellite(satellite[0])
            pseudo_rgb = colorize_mask(pseudo_np, num_classes)
            clean_rgb = colorize_mask(clean, num_classes)
            pred_rgb = colorize_mask(pred_np, num_classes)
            diff_rgb = create_diff_map(pseudo_np, pred_np, clean)

            pseudo_overlay = create_overlay(sat_rgb, pseudo_rgb)
            pred_overlay = create_overlay(sat_rgb, pred_rgb)

            miou_p = iou_pseudo.mean()
            miou_d = iou_pred.mean()
            delta_i = miou_d - miou_p

            vis_table.add_data(
                filename,
                wandb.Image(sat_rgb, caption='Satellite'),
                wandb.Image(pseudo_rgb, caption=f'Pseudo mIoU={miou_p:.3f}'),
                wandb.Image(clean_rgb, caption='Clean GT'),
                wandb.Image(pred_rgb, caption=f'Denoised mIoU={miou_d:.3f}'),
                wandb.Image(diff_rgb,
                            caption='Green=fixed Red=regressed'),
                wandb.Image(pseudo_overlay, caption='Pseudo overlay'),
                wandb.Image(pred_overlay, caption='Denoised overlay'),
                round(float(miou_p), 4),
                round(float(miou_d), 4),
                round(float(delta_i), 4),
            )

    # ── Aggregate metrics ───────────────────────────────────────────────
    mean_iou_pred = np.mean(all_iou_pred, axis=0)
    mean_iou_pseudo = np.mean(all_iou_pseudo, axis=0)

    exp_name = osp.splitext(osp.basename(
        cfg._filename if hasattr(cfg, '_filename') else 'experiment'))[0]

    print('\n' + '=' * 70)
    print(f'Evaluation Results — {exp_name} (iter {ckpt_iter})')
    print('=' * 70)
    print(f'\n{"Class":<15} {"Pseudo (baseline)":<20} {"Denoised (ours)":<20} {"Δ":<10}')
    print('-' * 70)
    for c in range(num_classes):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
        delta = mean_iou_pred[c] - mean_iou_pseudo[c]
        sign = '+' if delta >= 0 else ''
        print(f'{name:<15} {mean_iou_pseudo[c]:<20.4f} {mean_iou_pred[c]:<20.4f} '
              f'{sign}{delta:<10.4f}')
    print('-' * 70)
    miou_pseudo = mean_iou_pseudo.mean()
    miou_pred = mean_iou_pred.mean()
    delta = miou_pred - miou_pseudo
    sign = '+' if delta >= 0 else ''
    print(f'{"mIoU":<15} {miou_pseudo:<20.4f} {miou_pred:<20.4f} '
          f'{sign}{delta:<10.4f}')
    print('=' * 70)

    # ── Log to W&B ──────────────────────────────────────────────────────
    if use_wandb and HAS_WANDB and wandb.run is not None:
        # Summary metrics
        wandb.summary['eval/mIoU_pseudo'] = round(miou_pseudo, 4)
        wandb.summary['eval/mIoU_output'] = round(miou_pred, 4)
        wandb.summary['eval/mIoU_improved'] = round(delta, 4)
        wandb.summary['eval/checkpoint_iter'] = ckpt_iter

        # Per-class IoU as individual summary metrics
        for c in range(num_classes):
            cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
            wandb.summary[f'eval/iou_pseudo/{cname}'] = round(float(mean_iou_pseudo[c]), 4)
            wandb.summary[f'eval/iou_denoised/{cname}'] = round(float(mean_iou_pred[c]), 4)
            wandb.summary[f'eval/iou_delta/{cname}'] = round(float(mean_iou_pred[c] - mean_iou_pseudo[c]), 4)

        # Per-class IoU table
        class_table = wandb.Table(
            columns=['Class', 'IoU_Pseudo', 'IoU_Denoised', 'Δ'],
            data=[
                [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}',
                 round(float(mean_iou_pseudo[c]), 4),
                 round(float(mean_iou_pred[c]), 4),
                 round(float(mean_iou_pred[c] - mean_iou_pseudo[c]), 4)]
                for c in range(num_classes)
            ])
        wandb.log({'eval/per_class_iou': class_table})

        # Visual comparison table
        if vis_table is not None:
            wandb.log({'eval/inference_samples': vis_table})

        # Bar charts
        wandb.log({
            'eval/per_class_pseudo': wandb.plot.bar(
                class_table, 'Class', 'IoU_Pseudo',
                title='Per-Class IoU — Pseudo-Label Baseline'),
            'eval/per_class_denoised': wandb.plot.bar(
                class_table, 'Class', 'IoU_Denoised',
                title='Per-Class IoU — Denoised Output'),
        })

        print('Results logged to W&B ✓')

    return dict(
        miou_pseudo=miou_pseudo,
        miou_pred=miou_pred,
        miou_delta=delta,
        per_class_pseudo=mean_iou_pseudo,
        per_class_pred=mean_iou_pred,
    )


# ── Standalone entry point ──────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    exp_name = osp.splitext(osp.basename(args.config))[0]

    # ── Load checkpoint ─────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # ── W&B init: resume SAME run as training ───────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb_run_id = ckpt.get('wandb_run_id', None)
        wandb_project = ckpt.get('wandb_project', None)
        wandb_entity = ckpt.get('wandb_entity', None)

        # Fallback to config if checkpoint doesn't have W&B info
        wandb_cfg = cfg.get('wandb', dict(project='pseudo-denoiser-d3pm'))
        if wandb_project is None:
            wandb_project = wandb_cfg.get('project', 'pseudo-denoiser-d3pm')

        try:
            if wandb_run_id:
                # Resume the SAME training run → all data in one experiment
                print(f'Resuming W&B run: {wandb_run_id} '
                      f'({wandb_entity}/{wandb_project})')
                wandb.init(
                    id=wandb_run_id,
                    project=wandb_project,
                    entity=wandb_entity,
                    resume='must',
                )
            else:
                # No run ID in checkpoint → create eval run linked to training
                print('No W&B run ID in checkpoint. '
                      'Creating new eval run (same project).')
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f'{exp_name}_eval',
                    job_type='evaluation',
                    config={
                        'config': args.config,
                        'checkpoint': args.checkpoint,
                        'experiment': exp_name,
                    })
            print(f'W&B run: {wandb.run.url}')
        except Exception as e:
            print(f'Warning: W&B init failed ({e}). Continuing without W&B.')
            use_wandb = False

    # ── Build model ─────────────────────────────────────────────────────
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model = DiffusionDenoiserModel(**model_cfg)

    if 'ema' in ckpt:
        state_dict = model.state_dict()
        for k, v in ckpt['ema'].items():
            if k in state_dict:
                state_dict[k] = v
        model.load_state_dict(state_dict)
        print('Loaded EMA weights.')
    else:
        model.load_state_dict(ckpt['model'])
        print('Loaded model weights (no EMA).')

    model = model.to(device)
    model.eval()
    log_model_params(model)
    print(f'Checkpoint iteration: {ckpt.get("iter", "unknown")}')

    # ── Run evaluation ──────────────────────────────────────────────────
    run_evaluation(
        model=model,
        cfg=cfg,
        device=device,
        checkpoint_path=args.checkpoint,
        use_wandb=use_wandb,
        num_steps=args.num_steps,
        temperature=args.temperature,
        max_vis=args.max_vis,
    )

    # ── Finish W&B ──────────────────────────────────────────────────────
    if use_wandb and HAS_WANDB:
        wandb.finish()
        print('W&B run finished ✓')


if __name__ == '__main__':
    main()

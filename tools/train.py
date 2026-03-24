"""Training script for D3PM discrete diffusion pseudo-label denoiser.

Usage:
    # Single GPU (epoch-based)
    python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        --cfg-options max_epochs=30

    # Multi-GPU
    torchrun --nproc_per_node=4 tools/train.py \
        configs/denoiser/d3pm_hybrid_uniform_512x512_100k.py --launcher pytorch

    # Resume
    python tools/train.py configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        --resume-from work_dirs/d3pm_concat_uniform/epoch_10.pth
"""

import argparse
import copy
import os
import os.path as osp
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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

# Registry for dataset classes
DATASET_REGISTRY = {
    'PseudoLabelDiffusionDataset': PseudoLabelDiffusionDataset,
    'OEMCISCRCrossEntropyDataset': OEMCISCRCrossEntropyDataset,
}


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.named_parameters() if v.requires_grad}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.named_parameters():
                if v.requires_grad and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        v.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA params."""
        for k, v in model.named_parameters():
            if v.requires_grad and k in self.shadow:
                v.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module, backup: dict):
        """Restore model params from backup."""
        for k, v in model.named_parameters():
            if k in backup:
                v.data.copy_(backup[k])


def parse_args():
    parser = argparse.ArgumentParser(description='Train D3PM denoiser')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', help='Working directory')
    parser.add_argument('--resume-from', help='Checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-options', nargs='+', action='store', default=[],
                        help='Override config settings. key=value pairs')
    parser.add_argument('--launcher', choices=['none', 'pytorch'],
                        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def build_model(cfg) -> DiffusionDenoiserModel:
    """Build model from config dict."""
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    return DiffusionDenoiserModel(**model_cfg)


def build_dataset(data_cfg, is_train=True):
    """Build dataset from config dict, dispatching to correct dataset class."""
    dataset_cfg = data_cfg.copy()
    dataset_type = dataset_cfg.pop('type', 'OEMCISCRCrossEntropyDataset')

    if dataset_type in DATASET_REGISTRY:
        cls = DATASET_REGISTRY[dataset_type]
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}. '
                         f'Available: {list(DATASET_REGISTRY.keys())}')
    return cls(**dataset_cfg)


def evaluate(model, val_loader, device, num_steps=10):
    """Evaluate by denoising validation pseudo-labels and computing mIoU.

    Returns:
        dict with keys:
            miou_pred: mIoU of denoised output vs clean label
            miou_pseudo: mIoU of raw pseudo-label vs clean label (baseline)
            miou_delta: improvement (miou_pred - miou_pseudo)
            per_class_iou_pred: per-class IoU of denoised output
            per_class_iou_pseudo: per-class IoU of pseudo-labels
    """
    model.eval()
    num_classes = model.num_classes

    # Accumulators for denoised prediction
    inter_pred = torch.zeros(num_classes, device=device)
    union_pred = torch.zeros(num_classes, device=device)
    # Accumulators for pseudo-label baseline
    inter_pseudo = torch.zeros(num_classes, device=device)
    union_pseudo = torch.zeros(num_classes, device=device)

    for batch in val_loader:
        satellite = batch['satellite_img'].to(device)
        pseudo = batch['pseudo_label'].to(device)
        clean = batch['clean_label'].to(device)

        # Denoise with reduced steps for speed
        pred = model.denoise(satellite, pseudo, num_steps=num_steps)

        # Build valid mask (exclude ignore pixels = 255)
        valid = (clean != 255)

        # Compute per-class IoU for denoised output
        for c in range(num_classes):
            pred_c = (pred == c) & valid
            pseudo_c = (pseudo == c) & valid
            gt_c = (clean == c) & valid

            inter_pred[c] += (pred_c & gt_c).sum()
            union_pred[c] += (pred_c | gt_c).sum()

            inter_pseudo[c] += (pseudo_c & gt_c).sum()
            union_pseudo[c] += (pseudo_c | gt_c).sum()

    iou_pred = inter_pred / (union_pred + 1e-10)
    iou_pseudo = inter_pseudo / (union_pseudo + 1e-10)

    miou_pred = iou_pred.mean().item()
    miou_pseudo = iou_pseudo.mean().item()
    miou_delta = miou_pred - miou_pseudo

    model.train()
    return dict(
        miou_pred=miou_pred,
        miou_pseudo=miou_pseudo,
        miou_delta=miou_delta,
        per_class_iou_pred=iou_pred.cpu().numpy(),
        per_class_iou_pseudo=iou_pseudo.cpu().numpy(),
    )


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Apply --cfg-options overrides (key=value pairs)
    if args.cfg_options:
        for opt in args.cfg_options:
            if '=' in opt:
                key, val = opt.split('=', 1)
                # Try to evaluate the value (int, float, etc.)
                try:
                    val = eval(val)
                except:
                    pass
                cfg.merge_from_dict({key: val})

    # Setup distributed
    distributed = args.launcher != 'none'
    if distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    rank = args.local_rank if distributed else 0

    # Work dir
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = osp.join(
            cfg.get('log_dir', 'work_dirs'),
            osp.splitext(osp.basename(args.config))[0])
    if rank == 0:
        os.makedirs(work_dir, exist_ok=True)

    # W&B init with full experiment metadata
    global HAS_WANDB  # may be reassigned in except block below
    if rank == 0 and HAS_WANDB:
        wandb_cfg = cfg.get('wandb', dict(project='pseudo-denoiser-d3pm'))

        # Build comprehensive experiment config
        full_config = {
            # Model architecture
            'model': cfg.get('model', {}),
            # Dataset settings
            'dataset': {
                'type': cfg.data.get('type', 'OEMCISCRCrossEntropyDataset'),
                'data_root': cfg.data.get('data_root', 'data/OEM_v2_aDanh'),
                'num_classes': cfg.get('num_classes', 7),
                'img_size': cfg.get('img_size', 512),
                'samples_per_gpu': cfg.data.get('samples_per_gpu', 1),
                'workers_per_gpu': cfg.data.get('workers_per_gpu', 4),
            },
            # Training settings
            'training': {
                'max_epochs': cfg.get('max_epochs', 100),
                'max_iters': cfg.get('max_iters', None),
                'optimizer': cfg.get('optimizer', {}),
                'lr_scheduler': cfg.get('lr_scheduler', {}),
                'ema': {
                    'use_ema': cfg.get('use_ema', True),
                    'ema_decay': cfg.get('ema_decay', 0.9999),
                },
            },
            # Runtime settings
            'runtime': {
                'seed': args.seed,
                'ckpt_epoch_interval': cfg.get('ckpt_epoch_interval', 1),
                'eval_epoch_interval': cfg.get('eval_epoch_interval', 1),
                'log_interval': cfg.get('log_interval', 50),
            },
            # Diffusion specific
            'diffusion': {
                'num_timesteps': cfg.model.get('num_timesteps', 100),
                'transition_type': cfg.model.get('transition_type', 'uniform'),
                'beta_schedule': cfg.model.get('beta_schedule', 'cosine'),
                'loss_type': cfg.model.get('loss_type', 'hybrid'),
                'hybrid_lambda': cfg.model.get('hybrid_lambda', 0.01),
            },
            # UNet architecture
            'unet': {
                'base_channels': cfg.model.get('base_channels', 128),
                'channel_mult': cfg.model.get('channel_mult', (1, 2, 4, 8)),
                'num_res_blocks': cfg.model.get('num_res_blocks', 2),
                'attn_resolutions': cfg.model.get('attn_resolutions', (2, 4)),
                'cond_type': cfg.model.get('cond_type', 'concat'),
                'dropout': cfg.model.get('dropout', 0.1),
            },
        }

        try:
            wandb.init(
                project=wandb_cfg.get('project', 'pseudo-denoiser-d3pm'),
                name=wandb_cfg.get('name', osp.splitext(osp.basename(args.config))[0]),
                config=full_config,
                dir=work_dir)

            # Log git commit info
            try:
                git_commit = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
                git_branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
                git_diff = subprocess.check_output(
                    ['git', 'diff', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()

                wandb.run.tags = list(wandb.run.tags or []) + [git_commit[:7]]
                wandb.run.log_code(root='.')
                wandb.config.update({
                    'git_commit': git_commit,
                    'git_branch': git_branch,
                    'git_dirty': bool(git_diff)
                })
            except Exception as git_err:
                print(f'Warning: Could not log git info: {git_err}')

            # Log additional run metadata
            wandb.run.tags = [
                cfg.model.get('cond_type', 'concat'),
                cfg.model.get('transition_type', 'uniform'),
                f"bs{cfg.data.get('samples_per_gpu', 1)}",
            ] + list(wandb.run.tags or [])
        except Exception as e:
            print(f'Warning: W&B init failed ({e}). Training without W&B logging.')
            HAS_WANDB = False

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build model
    model = build_model(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])
    raw_model = model.module if distributed else model

    # Log parameter counts (trainable vs frozen)
    if rank == 0:
        wb_run = wandb.run if HAS_WANDB else None
        log_model_params(raw_model, wandb_run=wb_run)

    # EMA
    use_ema = cfg.get('use_ema', True)
    ema = EMA(raw_model, cfg.get('ema_decay', 0.9999)) if use_ema else None

    # Optimizer
    opt_cfg = cfg.get('optimizer', dict(type='AdamW', lr=1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        betas=opt_cfg.get('betas', (0.9, 0.999)),
        weight_decay=opt_cfg.get('weight_decay', 0.01))

    # Dataset
    train_dataset = build_dataset(cfg.data.train, is_train=True)
    val_dataset = build_dataset(cfg.data.val, is_train=False)

    # LR scheduler
    max_epochs = cfg.get('max_epochs', None)
    max_iters = cfg.get('max_iters', None)

    # Compute training length
    import math
    iters_per_epoch = math.ceil(len(train_dataset) / cfg.data.samples_per_gpu)

    if max_epochs is not None:
        max_iters = max_epochs * iters_per_epoch
        print(f'Training for {max_epochs} epochs ({max_iters} iterations, {iters_per_epoch} iters/epoch)')
    elif max_iters is not None:
        max_epochs = math.ceil(max_iters / iters_per_epoch)
        print(f'Training for {max_iters} iterations (~{max_epochs} epochs, {iters_per_epoch} iters/epoch)')
    else:
        max_epochs = 100
        max_iters = max_epochs * iters_per_epoch
        print(f'Using default: {max_epochs} epochs ({max_iters} iterations)')

    warmup_iters = cfg.get('lr_scheduler', {}).get('warmup_iters', 5000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters - warmup_iters, eta_min=1e-6)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # Resume
    start_iter = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt.get('iter', 0)
        if ema and 'ema' in ckpt:
            ema.shadow = ckpt['ema']
        if rank == 0:
            print(f'Resumed from iter {start_iter}')

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    log_interval = cfg.get('log_interval', 50)

    # Epoch-based intervals (preferred)
    ckpt_epoch_interval = cfg.get('ckpt_epoch_interval', 1)
    eval_epoch_interval = cfg.get('eval_epoch_interval', 1)

    # Fallback to iteration-based if epoch-based not set
    ckpt_interval = cfg.get('checkpoint_interval', None)
    eval_interval = cfg.get('eval_interval', None)

    if rank == 0:
        print(f'\nCheckpoint: every {ckpt_epoch_interval} epoch(s)')
        print(f'Evaluation: every {eval_epoch_interval} epoch(s)')
        print(f'Logging: every {log_interval} iterations\n')

    # Epoch tracking
    start_epoch = start_iter // iters_per_epoch
    current_epoch = start_epoch

    if rank == 0:
        print(f'Starting training from epoch {current_epoch + 1}...')
        print(f'Model: {cfg.model.type}, cond: {cfg.model.cond_type}, '
              f'noise: {cfg.model.transition_type}')

    for iteration in range(start_iter, max_iters):
        # Detect epoch boundary
        new_epoch = iteration // iters_per_epoch
        is_epoch_end = ((iteration + 1) % iters_per_epoch == 0) or ((iteration + 1) == max_iters)
        if new_epoch != current_epoch:
            current_epoch = new_epoch
            if rank == 0:
                print(f'\n=== Epoch {current_epoch}/{max_epochs} complete ===')

        # Get batch (with cycling)
        try:
            batch = next(data_iter)
        except StopIteration:
            if train_sampler:
                train_sampler.set_epoch(current_epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        satellite = batch['satellite_img'].to(device)
        clean_label = batch['clean_label'].to(device)
        pseudo_label = batch['pseudo_label'].to(device)

        # Forward: noise from pseudo-label, target is clean_label
        losses = model(clean_label, satellite, pseudo_label)
        loss = losses['loss_total']

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # LR warmup
        if iteration < warmup_iters:
            lr_scale = min(1.0, (iteration + 1) / warmup_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.optimizer.lr * lr_scale
        else:
            scheduler.step()

        # EMA update
        if ema:
            ema.update(raw_model)

        # Logging
        if rank == 0 and (iteration + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            epoch_num = (iteration + 1) // iters_per_epoch + 1
            loss_str = ' | '.join(
                f'{k}: {v.item():.4f}' for k, v in losses.items())
            progress_pct = 100 * (iteration + 1) / max_iters
            print(f'[Epoch {epoch_num} | Iter {iteration + 1}/{max_iters} ({progress_pct:.1f}%)] '
                  f'{loss_str} | lr: {lr:.2e}')

            # W&B log
            if HAS_WANDB:
                log_dict = {k: v.item() for k, v in losses.items()}
                log_dict['learning_rate'] = lr
                log_dict['iteration'] = iteration + 1
                log_dict['epoch'] = epoch_num
                wandb.log(log_dict)

        # Epoch-based checkpoint & evaluation
        if rank == 0 and is_epoch_end:
            epoch_num = (iteration + 1) // iters_per_epoch
            if epoch_num == 0:
                epoch_num = 1  # handle edge case

            # Checkpoint every N epochs
            save_ckpt = (epoch_num % ckpt_epoch_interval == 0) or ((iteration + 1) == max_iters)
            # Also support fallback to iteration-based
            if ckpt_interval is not None and not save_ckpt:
                save_ckpt = ((iteration + 1) % ckpt_interval == 0)

            if save_ckpt:
                ckpt_path = osp.join(work_dir, f'epoch_{epoch_num}.pth')
                save_dict = dict(
                    model=raw_model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iter=iteration + 1,
                    epoch=epoch_num,
                    config_path=args.config)
                if ema:
                    save_dict['ema'] = ema.shadow
                # Save W&B run info so test.py can resume the same run
                if HAS_WANDB and wandb.run is not None:
                    save_dict['wandb_run_id'] = wandb.run.id
                    save_dict['wandb_project'] = wandb.run.project
                    save_dict['wandb_entity'] = wandb.run.entity
                torch.save(save_dict, ckpt_path)
                # Symlink latest
                latest = osp.join(work_dir, 'latest.pth')
                if osp.exists(latest):
                    os.remove(latest)
                os.symlink(osp.basename(ckpt_path), latest)
                print(f'Saved checkpoint: {ckpt_path}')

            # Evaluation every N epochs
            do_eval = (epoch_num % eval_epoch_interval == 0) or ((iteration + 1) == max_iters)
            # Also support fallback to iteration-based
            if eval_interval is not None and not do_eval:
                do_eval = ((iteration + 1) % eval_interval == 0)

            if do_eval:
                print(f'\n[Eval @ Epoch {epoch_num}, Iter {iteration + 1}]')
                # Apply EMA for evaluation
                if ema:
                    backup = {k: v.data.clone() for k, v in raw_model.named_parameters()}
                    ema.apply(raw_model)

                eval_results = evaluate(raw_model, val_loader, device)
                miou_pred = eval_results['miou_pred']
                miou_pseudo = eval_results['miou_pseudo']
                miou_delta = eval_results['miou_delta']
                per_class_pred = eval_results['per_class_iou_pred']
                per_class_pseudo = eval_results['per_class_iou_pseudo']

                sign = '+' if miou_delta >= 0 else ''
                # Per-class IoU table
                CLASS_NAMES = ['Bareland', 'Rangeland', 'Developed', 'Road',
                               'Tree', 'Water', 'Agriculture', 'Building']
                num_c = len(per_class_pred)

                print(f'  {"Class":<15} {"Pseudo":>10} {"Denoised":>10} {"Δ":>10}')
                print(f'  {"-"*45}')
                for c in range(num_c):
                    cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
                    d = per_class_pred[c] - per_class_pseudo[c]
                    s = '+' if d >= 0 else ''
                    print(f'  {cname:<15} {per_class_pseudo[c]:>10.4f} '
                          f'{per_class_pred[c]:>10.4f} {s}{d:>9.4f}')
                print(f'  {"-"*45}')
                print(f'  {"mIOU":<15} {miou_pseudo:>10.4f} '
                      f'{miou_pred:>10.4f} {sign}{miou_delta:>9.4f}')

                # W&B log evaluation metrics
                if HAS_WANDB:
                    log_dict = {
                        'val/mIoU_pseudo': miou_pseudo,
                        'val/mIoU_output': miou_pred,
                        'val/mIoU_improved': miou_delta,
                        'epoch': epoch_num,
                        'iteration': iteration + 1,
                    }
                    # Per-class IoU as individual scalar metrics
                    for c in range(num_c):
                        cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class_{c}'
                        log_dict[f'val/iou_pseudo/{cname}'] = float(per_class_pseudo[c])
                        log_dict[f'val/iou_denoised/{cname}'] = float(per_class_pred[c])
                        log_dict[f'val/iou_delta/{cname}'] = float(per_class_pred[c] - per_class_pseudo[c])
                    wandb.log(log_dict)

                if ema:
                    ema.restore(raw_model, backup)

    if rank == 0:
        print('Training complete.')

        # ── Auto-run final evaluation with visualization ────────────────
        print('\n' + '=' * 60)
        print('Running final evaluation with visualization...')
        print('=' * 60)
        latest_ckpt = osp.join(work_dir, 'latest.pth')
        if osp.exists(latest_ckpt):
            try:
                from tools.test import (
                    run_evaluation, colorize_mask, denormalize_satellite,
                    create_overlay, create_diff_map, CLASS_NAMES
                )
                # Apply EMA for final eval
                if ema:
                    backup = {k: v.data.clone()
                              for k, v in raw_model.named_parameters()}
                    ema.apply(raw_model)

                run_evaluation(
                    model=raw_model,
                    cfg=cfg,
                    device=device,
                    checkpoint_path=latest_ckpt,
                    use_wandb=(HAS_WANDB and wandb.run is not None),
                )

                if ema:
                    ema.restore(raw_model, backup)
            except Exception as e:
                print(f'Warning: Auto-evaluation failed: {e}')
                import traceback
                traceback.print_exc()
        else:
            print(f'No checkpoint found at {latest_ckpt}, skipping eval.')

        if HAS_WANDB:
            wandb.finish()


if __name__ == '__main__':
    main()

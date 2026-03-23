"""Inference script: denoise pseudo-labels via D3PM reverse diffusion.

Usage:
    python tools/inference.py \
        configs/denoiser/d3pm_concat_uniform_512x512_100k.py \
        work_dirs/d3pm_concat_uniform/latest.pth \
        --img-dir data/test/images \
        --pseudo-dir data/test/pseudo_labels \
        --out-dir data/test/refined_labels \
        --num-classes 7 \
        --num-steps 50
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from diffusion_denoiser.utils.config import Config
from diffusion_denoiser.models.diffusion_denoiser import DiffusionDenoiserModel


def parse_args():
    parser = argparse.ArgumentParser(description='D3PM inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--pseudo-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Override denoising steps (default: use T from config)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--img-suffix', default='.tif')
    parser.add_argument('--pseudo-suffix', default='.png')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Build model
    model_cfg = cfg.model.copy()
    model_cfg.pop('type', None)
    model = DiffusionDenoiserModel(**model_cfg)

    # Load checkpoint (handle EMA)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'ema' in ckpt:
        # Use EMA weights for inference
        state_dict = model.state_dict()
        for k, v in ckpt['ema'].items():
            if k in state_dict:
                state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt['model'])

    model = model.to(args.device)
    model.eval()

    # Normalization
    img_norm_cfg = cfg.data.get('img_norm_cfg', cfg.data.train.get(
        'img_norm_cfg', dict(mean=[123.675, 116.28, 103.53],
                             std=[58.395, 57.12, 57.375])))
    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)

    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.endswith(args.img_suffix)])

    print(f'Found {len(img_files)} images. Denoising with '
          f'{args.num_steps or cfg.model.num_timesteps} steps...')

    for img_file in tqdm(img_files, desc='Denoising'):
        # Load satellite image
        img = cv2.imread(osp.join(args.img_dir, img_file)).astype(np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # (3, H, W)
        img_t = torch.from_numpy(img).unsqueeze(0).float().to(args.device)

        # Load pseudo-label
        pseudo_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        pseudo_path = osp.join(args.pseudo_dir, pseudo_file)
        if not osp.exists(pseudo_path):
            print(f'Warning: {pseudo_file} not found, skipping.')
            continue
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)
        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]
        pseudo_t = torch.from_numpy(pseudo.astype(np.int64)).unsqueeze(0).to(args.device)

        # Denoise
        with torch.no_grad():
            pred = model.denoise(
                img_t, pseudo_t,
                num_steps=args.num_steps,
                temperature=args.temperature)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

        # Save
        out_file = img_file.replace(args.img_suffix, args.pseudo_suffix)
        Image.fromarray(pred).save(osp.join(args.out_dir, out_file))

    print(f'Done. Refined labels saved to {args.out_dir}')


if __name__ == '__main__':
    main()

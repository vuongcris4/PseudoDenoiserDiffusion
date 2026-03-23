"""Dataset for diffusion-based pseudo-label denoising.

Loads triplets of (satellite image, noisy pseudo-label, clean label).
Unlike the MMsegDenoiser dataset, this does NOT one-hot encode the
pseudo-label at the dataset level — the D3PM handles that internally.

The satellite image is normalized and returned as (3, H, W).
Labels are returned as class index maps (H, W).
"""

import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PseudoLabelDiffusionDataset(Dataset):
    """Dataset for D3PM pseudo-label denoising.

    Args:
        data_root (str): Root directory.
        img_dir (str): Satellite image directory relative to data_root.
        pseudo_label_dir (str): Pseudo-label directory.
        ann_dir (str): Clean label directory.
        num_classes (int): Number of segmentation classes.
        crop_size (tuple): Random crop size (H, W). Default: (512, 512).
        img_suffix (str): Image file suffix. Default: '.tif'.
        label_suffix (str): Label file suffix. Default: '.png'.
        img_norm_cfg (dict): Normalization config.
        is_train (bool): Whether in training mode.
        ignore_index (int): Ignore index. Default: 255.
    """

    def __init__(self,
                 data_root: str,
                 img_dir: str,
                 pseudo_label_dir: str,
                 ann_dir: str,
                 num_classes: int,
                 crop_size: Tuple[int, int] = (512, 512),
                 img_suffix: str = '.tif',
                 label_suffix: str = '.png',
                 img_norm_cfg: Optional[dict] = None,
                 is_train: bool = True,
                 ignore_index: int = 255):
        super().__init__()
        self.data_root = data_root
        self.img_dir = osp.join(data_root, img_dir)
        self.pseudo_label_dir = osp.join(data_root, pseudo_label_dir)
        self.ann_dir = osp.join(data_root, ann_dir)
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.is_train = is_train
        self.ignore_index = ignore_index

        if img_norm_cfg is None:
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375])
        self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.std = np.array(img_norm_cfg['std'], dtype=np.float32)

        # Collect filenames
        self.filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(self.img_suffix)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]
        label_fname = fname.replace(self.img_suffix, self.label_suffix)

        # Load satellite image (BGR → float32)
        img = cv2.imread(osp.join(self.img_dir, fname)).astype(np.float32)

        # Load pseudo-label and clean label
        pseudo = cv2.imread(
            osp.join(self.pseudo_label_dir, label_fname),
            cv2.IMREAD_UNCHANGED)
        clean = cv2.imread(
            osp.join(self.ann_dir, label_fname),
            cv2.IMREAD_UNCHANGED)

        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]
        if clean.ndim == 3:
            clean = clean[:, :, 0]

        pseudo = pseudo.astype(np.int64)
        clean = clean.astype(np.int64)

        # Data augmentation (training only)
        if self.is_train:
            img, pseudo, clean = self._random_crop(img, pseudo, clean)
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
                pseudo = pseudo[:, ::-1].copy()
                clean = clean[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[::-1, :, :].copy()
                pseudo = pseudo[::-1, :].copy()
                clean = clean[::-1, :].copy()
        else:
            # Center crop or pad for validation
            img, pseudo, clean = self._center_crop_or_pad(img, pseudo, clean)

        # Normalize image
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # (3, H, W)

        return dict(
            satellite_img=torch.from_numpy(img).float(),
            pseudo_label=torch.from_numpy(pseudo).long(),
            clean_label=torch.from_numpy(clean).long(),
            filename=fname,
        )

    def _random_crop(self, img, pseudo, clean):
        h, w = img.shape[:2]
        ch, cw = self.crop_size

        if h < ch or w < cw:
            img, pseudo, clean = self._pad(img, pseudo, clean, ch, cw)
            h, w = img.shape[:2]

        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
        img = img[top:top + ch, left:left + cw]
        pseudo = pseudo[top:top + ch, left:left + cw]
        clean = clean[top:top + ch, left:left + cw]
        return img, pseudo, clean

    def _center_crop_or_pad(self, img, pseudo, clean):
        h, w = img.shape[:2]
        ch, cw = self.crop_size

        if h < ch or w < cw:
            img, pseudo, clean = self._pad(img, pseudo, clean, ch, cw)
            h, w = img.shape[:2]

        top = (h - ch) // 2
        left = (w - cw) // 2
        img = img[top:top + ch, left:left + cw]
        pseudo = pseudo[top:top + ch, left:left + cw]
        clean = clean[top:top + ch, left:left + cw]
        return img, pseudo, clean

    def _pad(self, img, pseudo, clean, target_h, target_w):
        h, w = img.shape[:2]
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pseudo = np.pad(pseudo, ((0, pad_h), (0, pad_w)),
                        constant_values=self.ignore_index)
        clean = np.pad(clean, ((0, pad_h), (0, pad_w)),
                       constant_values=self.ignore_index)
        return img, pseudo, clean

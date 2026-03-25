"""Dataset for D3PM training with OEM dataset and CISC-R pseudo-labels.

This dataset reads from the OEM_v2_aDanh structure:
    data_root/
    ├── images/
    ├── pseudolabels/
    ├── labels/
    └── train.txt, val.txt, test.txt

Output format is compatible with D3PM training loop:
    {
        'satellite_img': tensor(3, H, W),
        'pseudo_label': tensor(H, W),
        'clean_label': tensor(H, W),
        'filename': str
    }
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

# 8 classes for segmentation (OEM raw labels 1-8 → mapped to 0-7)
# Raw 0 = nodata/background → mapped to IGNORE_INDEX (255)
NUM_CLASSES = 8
IGNORE_INDEX = 255


def get_split_file(data_root: str, split: str = 'train') -> str:
    """Get split file path.

    Args:
        data_root: Dataset root directory
        split: 'train', 'val', or 'test'

    Returns:
        Path to split file
    """
    split_file = os.path.join(data_root, f'{split}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f'Split file not found: {split_file}')
    return split_file


def find_pseudo_pairs(data_root: str, split_file: str) -> List[Tuple[str, str, str]]:
    """Find triplets (image, pseudo-label, ground-truth) from split file.

    Args:
        data_root: Dataset root containing images/, pseudolabels/, labels/
        split_file: Path to .txt file with filenames

    Returns:
        List of (img_path, pseudo_path, gt_path) tuples
    """
    pairs = []
    with open(split_file) as f:
        filenames = [l.strip() for l in f if l.strip()]

    for fn in filenames:
        img_path = os.path.join(data_root, 'images', fn)
        pseudo_path = os.path.join(data_root, 'pseudolabels', fn)
        gt_path = os.path.join(data_root, 'labels', fn)

        if os.path.exists(img_path) and os.path.exists(pseudo_path) and os.path.exists(gt_path):
            pairs.append((img_path, pseudo_path, gt_path))

    return pairs


class OEMCISCRCrossEntropyDataset(Dataset):
    """Dataset for D3PM training with OEM + CISC-R pseudo-labels.

    Output format:
        {
            'satellite_img': tensor(3, H, W) float32, normalized [0, 1]
            'pseudo_label': tensor(H, W) long, class indices 0-6
            'clean_label': tensor(H, W) long, class indices 0-6
            'filename': str
        }

    Note: Pseudo-labels are NOT one-hot encoded. D3PM handles discrete labels.
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 img_size: int = 512,
                 augment: bool = True,
                 num_classes: int = NUM_CLASSES):
        """
        Args:
            data_root: Dataset root directory
            split: 'train', 'val', or 'test'
            img_size: Output image size (default 512)
            augment: Enable augmentation (only for train split)
            num_classes: Number of segmentation classes (default 7)
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.num_classes = num_classes

        split_file = get_split_file(data_root, split)
        self.pairs = find_pseudo_pairs(data_root, split_file)
        print(f'OEMCISCRCrossEntropyDataset {split}: {len(self.pairs)} samples')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, any]:
        img_path, pseudo_path, gt_path = self.pairs[idx]

        # Read RGB image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] > 3:
                img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read pseudo-label (from CISC-R)
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_UNCHANGED)
        if pseudo is None:
            try:
                import tifffile
                pseudo = tifffile.imread(pseudo_path)
            except:
                pseudo = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if pseudo.ndim == 3:
            pseudo = pseudo[:, :, 0]

        # Read ground truth
        clean = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if clean is None:
            try:
                import tifffile
                clean = tifffile.imread(gt_path)
            except:
                clean = pseudo.copy()
        if clean.ndim == 3:
            clean = clean[:, :, 0]

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        pseudo = cv2.resize(pseudo, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)
        clean = cv2.resize(clean, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)

        # Augmentation (synchronous for all)
        if self.augment:
            if np.random.random() > 0.5:  # Horizontal flip
                img = img[:, ::-1].copy()
                pseudo = pseudo[:, ::-1].copy()
                clean = clean[:, ::-1].copy()
            if np.random.random() > 0.5:  # Vertical flip
                img = img[::-1, :].copy()
                pseudo = pseudo[::-1, :].copy()
                clean = clean[::-1, :].copy()

        # Convert to tensor
        # Image: [H, W, 3] -> [3, H, W], normalize to [0, 1]
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0

        # Remap OEM labels: raw 1-8 → class 0-7, raw 0 → IGNORE_INDEX (255)
        pseudo_np = pseudo.astype(np.int32)
        pseudo_np = np.where(pseudo_np == 0, IGNORE_INDEX, pseudo_np - 1)
        pseudo_np = np.clip(pseudo_np, 0, self.num_classes - 1)
        pseudo_np[pseudo == 0] = IGNORE_INDEX  # re-apply after clip
        pseudo_t = torch.from_numpy(pseudo_np.copy()).long()

        clean_np = clean.astype(np.int32)
        clean_np = np.where(clean_np == 0, IGNORE_INDEX, clean_np - 1)
        clean_np = np.clip(clean_np, 0, self.num_classes - 1)
        clean_np[clean == 0] = IGNORE_INDEX  # re-apply after clip
        clean_t = torch.from_numpy(clean_np.copy()).long()

        return {
            'satellite_img': img_t,
            'pseudo_label': pseudo_t,
            'clean_label': clean_t,
            'filename': os.path.basename(img_path)
        }
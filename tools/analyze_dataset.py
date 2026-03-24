"""Comprehensive dataset analysis: pseudo-label vs ground-truth.

Outputs:
  1. per_image_analysis.csv   – one row per image with per-class metrics
  2. overall_summary.csv      – global & per-split statistics
  3. confusion_matrix.csv     – class-level confusion matrix (pseudo→GT)
  4. class_error_analysis.csv  – which classes are most confused
  5. dataset_report.md        – human-readable markdown report
"""

import os, sys, csv, json
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'OEM_v2_aDanh')
DATA_ROOT = os.path.normpath(DATA_ROOT)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'analysis_output')

NUM_CLASSES = 8
IGNORE_INDEX = 255

# OEM class names (raw 1-8 → mapped 0-7)
CLASS_NAMES = {
    0: 'Barren',
    1: 'Rangeland',
    2: 'Developed',
    3: 'Road',
    4: 'Tree',
    5: 'Water',
    6: 'Agriculture',
    7: 'Building',
}


def remap_label(raw: np.ndarray) -> np.ndarray:
    """Remap OEM raw labels (1-8) → class indices (0-7), raw 0 → 255."""
    out = raw.astype(np.int32)
    out = np.where(out == 0, IGNORE_INDEX, out - 1)
    out = np.clip(out, 0, NUM_CLASSES - 1)
    out[raw == 0] = IGNORE_INDEX  # re-apply after clip
    return out


def compute_per_image_metrics(pseudo: np.ndarray, gt: np.ndarray, num_classes: int):
    """Compute per-image mIoU, precision, recall, per-class IoU/prec/rec, confusion."""
    valid_mask = (gt != IGNORE_INDEX) & (pseudo != IGNORE_INDEX)
    pseudo_v = pseudo[valid_mask]
    gt_v = gt[valid_mask]
    total_pixels = int(valid_mask.sum())

    if total_pixels == 0:
        return None

    # Confusion matrix: conf[gt_class, pred_class]
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c_gt in range(num_classes):
        for c_pred in range(num_classes):
            conf[c_gt, c_pred] = int(np.sum((gt_v == c_gt) & (pseudo_v == c_pred)))

    # Per-class metrics
    per_class = {}
    ious, precs, recs = [], [], []
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp  # other classes predicted as c
        fn = conf[c, :].sum() - tp  # c predicted as other classes
        gt_pixels = conf[c, :].sum()
        pred_pixels = conf[:, c].sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        prec = tp / pred_pixels if pred_pixels > 0 else float('nan')
        rec = tp / gt_pixels if gt_pixels > 0 else float('nan')

        per_class[c] = {
            'gt_pixels': int(gt_pixels),
            'pred_pixels': int(pred_pixels),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'iou': iou,
            'precision': prec,
            'recall': rec,
        }
        if not np.isnan(iou):
            ious.append(iou)
        if not np.isnan(prec):
            precs.append(prec)
        if not np.isnan(rec):
            recs.append(rec)

    # Accuracy
    correct = int((pseudo_v == gt_v).sum())
    accuracy = correct / total_pixels if total_pixels > 0 else 0
    mIoU = np.mean(ious) if ious else 0
    mean_prec = np.mean(precs) if precs else 0
    mean_rec = np.mean(recs) if recs else 0

    # Noisy pixel count
    noisy_pixels = int((pseudo_v != gt_v).sum())
    noise_ratio = noisy_pixels / total_pixels if total_pixels > 0 else 0

    # Dominant error: which class has highest FN
    max_fn_class = max(range(num_classes), key=lambda c: per_class[c]['fn'])

    return {
        'total_pixels': total_pixels,
        'correct_pixels': correct,
        'noisy_pixels': noisy_pixels,
        'accuracy': accuracy,
        'noise_ratio': noise_ratio,
        'mIoU': mIoU,
        'mean_precision': mean_prec,
        'mean_recall': mean_rec,
        'per_class': per_class,
        'confusion': conf,
        'most_misclassified_class': max_fn_class,
    }


def load_split(split: str) -> List[str]:
    """Load filenames from split file."""
    path = os.path.join(DATA_ROOT, f'{split}.txt')
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def read_label(path: str) -> np.ndarray:
    """Read a label .tif file."""
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        try:
            import tifffile
            label = tifffile.imread(path)
        except Exception:
            return None
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Data root: {DATA_ROOT}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")

    # ── 1. Per-image analysis ───────────────────────────────────────────
    all_rows = []
    global_conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    split_confs = {s: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for s in ['train', 'val', 'test']}
    split_stats = {s: {'count': 0, 'total_pixels': 0, 'correct': 0, 'noisy': 0} for s in ['train', 'val', 'test']}

    # Per-class global accumulators for micro-averaging
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fn = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Additional stats
    per_image_noise_ratios = []
    per_image_mious = []
    gt_class_dist = np.zeros(NUM_CLASSES, dtype=np.int64)
    pseudo_class_dist = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_noise_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_class_noise_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    # Edge/boundary analysis
    edge_error_pixels = 0
    non_edge_error_pixels = 0
    total_edge_pixels = 0
    total_non_edge_pixels = 0

    for split in ['train', 'val', 'test']:
        filenames = load_split(split)
        print(f"\n[INFO] Processing {split}: {len(filenames)} samples")

        for i, fn in enumerate(filenames):
            pseudo_path = os.path.join(DATA_ROOT, 'pseudolabels', fn)
            gt_path = os.path.join(DATA_ROOT, 'labels', fn)

            pseudo_raw = read_label(pseudo_path)
            gt_raw = read_label(gt_path)

            if pseudo_raw is None or gt_raw is None:
                print(f"  [WARN] Skipping {fn}: cannot read labels")
                continue

            pseudo = remap_label(pseudo_raw)
            gt = remap_label(gt_raw)

            metrics = compute_per_image_metrics(pseudo, gt, NUM_CLASSES)
            if metrics is None:
                continue

            # Edge analysis for this image
            kernel = np.ones((3, 3), np.uint8)
            gt_for_edge = gt.copy().astype(np.uint8)
            gt_for_edge[gt == IGNORE_INDEX] = 255
            eroded = cv2.erode(gt_for_edge, kernel, iterations=1)
            dilated = cv2.dilate(gt_for_edge, kernel, iterations=1)
            edge_mask = (eroded != dilated) & (gt != IGNORE_INDEX) & (pseudo != IGNORE_INDEX)
            non_edge_mask = (~edge_mask) & (gt != IGNORE_INDEX) & (pseudo != IGNORE_INDEX)
            error_mask = (pseudo != gt) & (gt != IGNORE_INDEX) & (pseudo != IGNORE_INDEX)

            edge_err = int((edge_mask & error_mask).sum())
            non_edge_err = int((non_edge_mask & error_mask).sum())
            edge_error_pixels += edge_err
            non_edge_error_pixels += non_edge_err
            total_edge_pixels += int(edge_mask.sum())
            total_non_edge_pixels += int(non_edge_mask.sum())

            # Accumulate
            global_conf += metrics['confusion']
            split_confs[split] += metrics['confusion']
            split_stats[split]['count'] += 1
            split_stats[split]['total_pixels'] += metrics['total_pixels']
            split_stats[split]['correct'] += metrics['correct_pixels']
            split_stats[split]['noisy'] += metrics['noisy_pixels']

            for c in range(NUM_CLASSES):
                pc = metrics['per_class'][c]
                class_tp[c] += pc['tp']
                class_fp[c] += pc['fp']
                class_fn[c] += pc['fn']
                gt_class_dist[c] += pc['gt_pixels']
                pseudo_class_dist[c] += pc['pred_pixels']
                per_class_noise_count[c] += pc['gt_pixels']
                per_class_noise_total[c] += pc['fn']

            per_image_noise_ratios.append(metrics['noise_ratio'])
            per_image_mious.append(metrics['mIoU'])

            # Build row for CSV
            row = {
                'filename': fn,
                'split': split,
                'img_height': gt_raw.shape[0],
                'img_width': gt_raw.shape[1],
                'total_valid_pixels': metrics['total_pixels'],
                'correct_pixels': metrics['correct_pixels'],
                'noisy_pixels': metrics['noisy_pixels'],
                'accuracy': f"{metrics['accuracy']:.4f}",
                'noise_ratio': f"{metrics['noise_ratio']:.4f}",
                'mIoU': f"{metrics['mIoU']:.4f}",
                'mean_precision': f"{metrics['mean_precision']:.4f}",
                'mean_recall': f"{metrics['mean_recall']:.4f}",
                'most_misclassified_class_id': metrics['most_misclassified_class'],
                'most_misclassified_class_name': CLASS_NAMES[metrics['most_misclassified_class']],
                'edge_error_pixels': edge_err,
                'non_edge_error_pixels': non_edge_err,
                'edge_total_pixels': int(edge_mask.sum()),
                'edge_error_rate': f"{edge_err / max(int(edge_mask.sum()), 1):.4f}",
                'non_edge_error_rate': f"{non_edge_err / max(int(non_edge_mask.sum()), 1):.4f}",
            }

            # Add per-class columns
            for c in range(NUM_CLASSES):
                cn = CLASS_NAMES[c]
                pc = metrics['per_class'][c]
                row[f'{cn}_gt_pixels'] = pc['gt_pixels']
                row[f'{cn}_pred_pixels'] = pc['pred_pixels']
                row[f'{cn}_iou'] = f"{pc['iou']:.4f}" if not np.isnan(pc['iou']) else 'N/A'
                row[f'{cn}_precision'] = f"{pc['precision']:.4f}" if not np.isnan(pc['precision']) else 'N/A'
                row[f'{cn}_recall'] = f"{pc['recall']:.4f}" if not np.isnan(pc['recall']) else 'N/A'
                row[f'{cn}_tp'] = pc['tp']
                row[f'{cn}_fp'] = pc['fp']
                row[f'{cn}_fn'] = pc['fn']

            # Add top-3 confused pairs for this image
            conf = metrics['confusion'].copy()
            np.fill_diagonal(conf, 0)
            flat = conf.flatten()
            top3_idx = np.argsort(flat)[-3:][::-1]
            for rank, idx in enumerate(top3_idx):
                gt_c = idx // NUM_CLASSES
                pred_c = idx % NUM_CLASSES
                row[f'confusion_rank{rank+1}_gt'] = CLASS_NAMES[gt_c]
                row[f'confusion_rank{rank+1}_pred'] = CLASS_NAMES[pred_c]
                row[f'confusion_rank{rank+1}_pixels'] = int(flat[idx])

            all_rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(filenames)}] processed...")

    # ── 2. Write per-image CSV ──────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'per_image_analysis.csv')
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
    print(f"\n[OK] Per-image CSV → {csv_path} ({len(all_rows)} rows)")

    # ── 3. Confusion matrix CSV ─────────────────────────────────────────
    conf_csv = os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')
    with open(conf_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['GT \\ Pred'] + [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
        w.writerow(header)
        for c_gt in range(NUM_CLASSES):
            row = [CLASS_NAMES[c_gt]] + [int(global_conf[c_gt, c_pred]) for c_pred in range(NUM_CLASSES)]
            w.writerow(row)
    print(f"[OK] Confusion matrix CSV → {conf_csv}")

    # ── 4. Class error analysis CSV ─────────────────────────────────────
    error_csv = os.path.join(OUTPUT_DIR, 'class_error_analysis.csv')
    with open(error_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'class_id', 'class_name', 'gt_total_pixels', 'pseudo_total_pixels',
            'gt_ratio', 'pseudo_ratio',
            'tp', 'fp', 'fn', 'iou', 'precision', 'recall', 'f1',
            'noise_rate',
            'top1_confused_with', 'top1_confused_pixels',
            'top2_confused_with', 'top2_confused_pixels',
            'top3_confused_with', 'top3_confused_pixels',
        ])
        total_gt_pixels = int(gt_class_dist.sum())
        total_pseudo_pixels = int(pseudo_class_dist.sum())
        for c in range(NUM_CLASSES):
            tp = int(class_tp[c])
            fp = int(class_fp[c])
            fn = int(class_fn[c])
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            noise_rate = fn / int(gt_class_dist[c]) if gt_class_dist[c] > 0 else 0

            # Top confused: for this GT class, which pred class gets the most FN?
            off_diag = global_conf[c, :].copy()
            off_diag[c] = 0
            top3 = np.argsort(off_diag)[-3:][::-1]

            row = [
                c, CLASS_NAMES[c], int(gt_class_dist[c]), int(pseudo_class_dist[c]),
                f"{gt_class_dist[c] / total_gt_pixels:.4f}" if total_gt_pixels else '0',
                f"{pseudo_class_dist[c] / total_pseudo_pixels:.4f}" if total_pseudo_pixels else '0',
                tp, fp, fn,
                f"{iou:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                f"{noise_rate:.4f}",
            ]
            for t in top3:
                row.append(CLASS_NAMES[t])
                row.append(int(off_diag[t]))
            w.writerow(row)
    print(f"[OK] Class error analysis CSV → {error_csv}")

    # ── 5. Overall summary CSV ──────────────────────────────────────────
    summary_csv = os.path.join(OUTPUT_DIR, 'overall_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'train', 'val', 'test', 'overall'])
        for metric_name, fn_val in [
            ('num_images', lambda s: split_stats[s]['count']),
            ('total_pixels', lambda s: split_stats[s]['total_pixels']),
            ('correct_pixels', lambda s: split_stats[s]['correct']),
            ('noisy_pixels', lambda s: split_stats[s]['noisy']),
            ('pixel_accuracy', lambda s: f"{split_stats[s]['correct'] / max(split_stats[s]['total_pixels'], 1):.4f}"),
            ('noise_ratio', lambda s: f"{split_stats[s]['noisy'] / max(split_stats[s]['total_pixels'], 1):.4f}"),
        ]:
            vals = [fn_val(s) for s in ['train', 'val', 'test']]
            # Overall
            if metric_name in ['pixel_accuracy', 'noise_ratio']:
                tot_pix = sum(split_stats[s]['total_pixels'] for s in ['train', 'val', 'test'])
                if metric_name == 'pixel_accuracy':
                    overall = f"{sum(split_stats[s]['correct'] for s in ['train','val','test']) / max(tot_pix,1):.4f}"
                else:
                    overall = f"{sum(split_stats[s]['noisy'] for s in ['train','val','test']) / max(tot_pix,1):.4f}"
            else:
                overall = sum(int(v) for v in vals)
            w.writerow([metric_name] + vals + [overall])

        # Global mIoU
        global_iou_per_class = []
        for c in range(NUM_CLASSES):
            tp = int(class_tp[c])
            fp = int(class_fp[c])
            fn = int(class_fn[c])
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            if not np.isnan(iou):
                global_iou_per_class.append(iou)
        global_miou = np.mean(global_iou_per_class) if global_iou_per_class else 0

        # Per-split mIoU
        split_mious = {}
        for s in ['train', 'val', 'test']:
            sc = split_confs[s]
            ious = []
            for c in range(NUM_CLASSES):
                tp = sc[c, c]
                fp = sc[:, c].sum() - tp
                fn = sc[c, :].sum() - tp
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
                if not np.isnan(iou):
                    ious.append(iou)
            split_mious[s] = np.mean(ious) if ious else 0

        w.writerow(['mIoU'] + [f"{split_mious[s]:.4f}" for s in ['train', 'val', 'test']] + [f"{global_miou:.4f}"])

        # Mean precision/recall
        for metric_key, arr_tp, arr_div in [
            ('mean_precision', class_tp, class_tp + class_fp),
            ('mean_recall', class_tp, class_tp + class_fn),
        ]:
            vals_per_class = []
            for c in range(NUM_CLASSES):
                if arr_div[c] > 0:
                    vals_per_class.append(float(arr_tp[c]) / float(arr_div[c]))
            overall_val = np.mean(vals_per_class) if vals_per_class else 0

            # Per-split
            split_vals = []
            for s in ['train', 'val', 'test']:
                sc = split_confs[s]
                v = []
                for c in range(NUM_CLASSES):
                    tp = sc[c, c]
                    if metric_key == 'mean_precision':
                        d = sc[:, c].sum()
                    else:
                        d = sc[c, :].sum()
                    if d > 0:
                        v.append(tp / d)
                split_vals.append(f"{np.mean(v):.4f}" if v else '0')
            w.writerow([metric_key] + split_vals + [f"{overall_val:.4f}"])

        # Edge error analysis
        edge_err_rate = edge_error_pixels / max(total_edge_pixels, 1)
        non_edge_err_rate = non_edge_error_pixels / max(total_non_edge_pixels, 1)
        w.writerow(['edge_error_rate', '', '', '', f"{edge_err_rate:.4f}"])
        w.writerow(['non_edge_error_rate', '', '', '', f"{non_edge_err_rate:.4f}"])
        w.writerow(['edge_total_pixels', '', '', '', total_edge_pixels])
        w.writerow(['non_edge_total_pixels', '', '', '', total_non_edge_pixels])

        # Noise distribution stats
        noise_arr = np.array(per_image_noise_ratios)
        w.writerow(['noise_ratio_mean', '', '', '', f"{noise_arr.mean():.4f}"])
        w.writerow(['noise_ratio_std', '', '', '', f"{noise_arr.std():.4f}"])
        w.writerow(['noise_ratio_min', '', '', '', f"{noise_arr.min():.4f}"])
        w.writerow(['noise_ratio_max', '', '', '', f"{noise_arr.max():.4f}"])
        w.writerow(['noise_ratio_median', '', '', '', f"{np.median(noise_arr):.4f}"])
        mIoU_arr = np.array(per_image_mious)
        w.writerow(['mIoU_mean', '', '', '', f"{mIoU_arr.mean():.4f}"])
        w.writerow(['mIoU_std', '', '', '', f"{mIoU_arr.std():.4f}"])
        w.writerow(['mIoU_min', '', '', '', f"{mIoU_arr.min():.4f}"])
        w.writerow(['mIoU_max', '', '', '', f"{mIoU_arr.max():.4f}"])

    print(f"[OK] Overall summary CSV → {summary_csv}")

    # ── 6. Markdown report ──────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, 'dataset_report.md')
    with open(report_path, 'w') as f:
        f.write("# Dataset Analysis Report: Pseudo-label vs Ground Truth\n\n")
        f.write(f"**Date:** Auto-generated\n")
        f.write(f"**Dataset:** OEM_v2_aDanh ({len(all_rows)} images)\n")
        f.write(f"**Classes:** {NUM_CLASSES} ({', '.join(CLASS_NAMES.values())})\n\n")

        f.write("## 1. Dataset Overview\n\n")
        f.write("| Split | Images | Total Pixels | Correct | Noisy | Pixel Acc | Noise Ratio |\n")
        f.write("|-------|--------|-------------|---------|-------|-----------|-------------|\n")
        for s in ['train', 'val', 'test']:
            st = split_stats[s]
            acc = st['correct'] / max(st['total_pixels'], 1)
            nr = st['noisy'] / max(st['total_pixels'], 1)
            f.write(f"| {s} | {st['count']} | {st['total_pixels']:,} | {st['correct']:,} | {st['noisy']:,} | {acc:.4f} | {nr:.4f} |\n")
        tot_p = sum(split_stats[s]['total_pixels'] for s in ['train', 'val', 'test'])
        tot_c = sum(split_stats[s]['correct'] for s in ['train', 'val', 'test'])
        tot_n = sum(split_stats[s]['noisy'] for s in ['train', 'val', 'test'])
        f.write(f"| **Total** | **{len(all_rows)}** | **{tot_p:,}** | **{tot_c:,}** | **{tot_n:,}** | **{tot_c/max(tot_p,1):.4f}** | **{tot_n/max(tot_p,1):.4f}** |\n\n")

        f.write("## 2. Global Metrics\n\n")
        f.write(f"- **mIoU:** {global_miou:.4f}\n")
        gprec_vals = [float(class_tp[c]) / float(class_tp[c] + class_fp[c]) for c in range(NUM_CLASSES) if (class_tp[c] + class_fp[c]) > 0]
        grec_vals = [float(class_tp[c]) / float(class_tp[c] + class_fn[c]) for c in range(NUM_CLASSES) if (class_tp[c] + class_fn[c]) > 0]
        f.write(f"- **Mean Precision:** {np.mean(gprec_vals):.4f}\n")
        f.write(f"- **Mean Recall:** {np.mean(grec_vals):.4f}\n")
        f.write(f"- **Pixel Accuracy:** {tot_c/max(tot_p,1):.4f}\n")
        f.write(f"- **Edge Error Rate:** {edge_err_rate:.4f} ({edge_error_pixels:,} / {total_edge_pixels:,} edge pixels)\n")
        f.write(f"- **Non-edge Error Rate:** {non_edge_err_rate:.4f} ({non_edge_error_pixels:,} / {total_non_edge_pixels:,} non-edge pixels)\n\n")

        f.write("## 3. Per-class Performance\n\n")
        f.write("| Class | GT Pixels | Pseudo Pixels | GT Ratio | IoU | Precision | Recall | F1 | Noise Rate |\n")
        f.write("|-------|-----------|---------------|----------|-----|-----------|--------|----|------------|\n")
        for c in range(NUM_CLASSES):
            tp = int(class_tp[c])
            fp = int(class_fp[c])
            fn = int(class_fn[c])
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            noise_r = fn / int(gt_class_dist[c]) if gt_class_dist[c] > 0 else 0
            gt_r = gt_class_dist[c] / total_gt_pixels if total_gt_pixels > 0 else 0
            f.write(f"| {CLASS_NAMES[c]} | {int(gt_class_dist[c]):,} | {int(pseudo_class_dist[c]):,} | {gt_r:.4f} | {iou:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {noise_r:.4f} |\n")
        f.write("\n")

        f.write("## 4. Confusion Matrix (GT rows → Pseudo columns)\n\n")
        f.write("| GT \\\\ Pred | " + " | ".join(CLASS_NAMES[c] for c in range(NUM_CLASSES)) + " |\n")
        f.write("|" + "---|" * (NUM_CLASSES + 1) + "\n")
        for c_gt in range(NUM_CLASSES):
            row_sum = global_conf[c_gt, :].sum()
            cells = []
            for c_pred in range(NUM_CLASSES):
                val = int(global_conf[c_gt, c_pred])
                pct = val / row_sum * 100 if row_sum > 0 else 0
                if c_gt == c_pred:
                    cells.append(f"**{val:,}** ({pct:.1f}%)")
                else:
                    cells.append(f"{val:,} ({pct:.1f}%)")
            f.write(f"| {CLASS_NAMES[c_gt]} | " + " | ".join(cells) + " |\n")
        f.write("\n")

        f.write("## 5. Top Confusion Pairs (GT class → wrongly predicted as)\n\n")
        # Find top-10 confused pairs
        conf_copy = global_conf.copy()
        np.fill_diagonal(conf_copy, 0)
        flat = conf_copy.flatten()
        top10 = np.argsort(flat)[-10:][::-1]
        f.write("| Rank | GT Class | Predicted As | Pixels | % of GT Class |\n")
        f.write("|------|----------|-------------|--------|---------------|\n")
        for rank, idx in enumerate(top10):
            gt_c = idx // NUM_CLASSES
            pred_c = idx % NUM_CLASSES
            pix = int(flat[idx])
            pct = pix / int(gt_class_dist[gt_c]) * 100 if gt_class_dist[gt_c] > 0 else 0
            f.write(f"| {rank+1} | {CLASS_NAMES[gt_c]} | {CLASS_NAMES[pred_c]} | {pix:,} | {pct:.2f}% |\n")
        f.write("\n")

        f.write("## 6. Noise Distribution Statistics\n\n")
        f.write(f"- **Mean noise ratio:** {noise_arr.mean():.4f} ± {noise_arr.std():.4f}\n")
        f.write(f"- **Median noise ratio:** {np.median(noise_arr):.4f}\n")
        f.write(f"- **Min / Max noise ratio:** {noise_arr.min():.4f} / {noise_arr.max():.4f}\n")
        f.write(f"- **Mean mIoU per image:** {mIoU_arr.mean():.4f} ± {mIoU_arr.std():.4f}\n")
        f.write(f"- **Min / Max mIoU:** {mIoU_arr.min():.4f} / {mIoU_arr.max():.4f}\n\n")

        # Histogram of noise ratios
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
        hist, _ = np.histogram(noise_arr, bins=bins)
        f.write("### Noise Ratio Distribution\n\n")
        f.write("| Noise Ratio Range | Count | Percentage |\n")
        f.write("|-------------------|-------|------------|\n")
        for i in range(len(bins) - 1):
            f.write(f"| {bins[i]:.2f} - {bins[i+1]:.2f} | {hist[i]} | {hist[i]/len(noise_arr)*100:.1f}% |\n")
        f.write("\n")

        f.write("## 7. Suggested Additional Metrics for Denoising Research\n\n")
        f.write("Các chỉ số gợi ý thêm để phân tích toàn diện cho bài toán denoising pseudo-label:\n\n")
        f.write("1. **Boundary Error Ratio** (đã tính ở trên): Tỷ lệ lỗi tại biên vs. không phải biên → xác định noise chủ yếu ở boundary hay interior\n")
        f.write("2. **Spatial Noise Clustering**: Dùng connected-component analysis trên error mask → đo kích thước vùng noise trung bình\n")
        f.write("3. **Per-Region Noise**: Tên khu vực (city/region trong filename) → xem region nào CISC-R predict kém nhất\n")
        f.write("4. **Class-conditional Transition Matrix**: Ma trận chuyển đổi chuẩn hóa → dùng làm prior cho D3PM transition\n")
        f.write("5. **Noise Spatial Heatmap**: Aggregate error positions → xem noise tập trung ở góc, biên hay giữa ảnh\n")
        f.write("6. **Small Object Error Rate**: Tính riêng cho các vùng GT nhỏ (< 100px) → pseudo-label thường nhầm object nhỏ\n")
        f.write("7. **Label Smoothness Score**: Entropy(pseudo) vs Entropy(GT) → đo mức \"noisy\" của pseudo-label\n")
        f.write("8. **Confidence Map Analysis**: Nếu CISC-R có confidence output → correlation giữa confidence thấp và error\n")
        f.write("9. **Temporal/Geographic Bias**: Group theo city → phát hiện bias theo geographic region\n")
        f.write("10. **Class Imbalance Impact**: Correlation giữa class frequency và noise rate → minority class bị ảnh hưởng nhiều hơn?\n\n")

        # Geographic analysis
        f.write("## 8. Geographic Analysis (Per-Region)\n\n")
        region_stats = defaultdict(lambda: {'count': 0, 'total': 0, 'noisy': 0})
        for row in all_rows:
            fn = row['filename']
            # Extract region: everything before the last underscore+number
            parts = fn.rsplit('_', 1)
            region = parts[0] if len(parts) > 1 else fn.replace('.tif', '')
            region_stats[region]['count'] += 1
            region_stats[region]['total'] += int(row['total_valid_pixels'])
            region_stats[region]['noisy'] += int(row['noisy_pixels'])

        # Sort by noise ratio descending
        region_list = sorted(region_stats.items(), key=lambda x: x[1]['noisy'] / max(x[1]['total'], 1), reverse=True)

        f.write("| Region | Images | Noise Ratio | Total Pixels |\n")
        f.write("|--------|--------|-------------|-------------|\n")
        for region, st in region_list[:20]:
            nr = st['noisy'] / max(st['total'], 1)
            f.write(f"| {region} | {st['count']} | {nr:.4f} | {st['total']:,} |\n")
        if len(region_list) > 20:
            f.write(f"| ... ({len(region_list) - 20} more regions) | | | |\n")
        f.write("\n")

    print(f"[OK] Report → {report_path}")
    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS IN: {OUTPUT_DIR}")
    print(f"  1. per_image_analysis.csv  ({len(all_rows)} rows)")
    print(f"  2. confusion_matrix.csv")
    print(f"  3. class_error_analysis.csv")
    print(f"  4. overall_summary.csv")
    print(f"  5. dataset_report.md")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

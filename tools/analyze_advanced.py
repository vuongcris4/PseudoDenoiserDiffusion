"""Advanced dataset analysis — MEMORY-EFFICIENT (streaming, no bulk load).

Processes images ONE AT A TIME, accumulating only lightweight stats.

  A. Per-class Difference Maps — spatial heatmaps (FN/FP per class)
  B. Region-level analysis — connected-component size vs accuracy
  C. Dimensionality reduction — PCA + t-SNE on 51-dim feature vectors

Outputs → analysis_output/advanced/
"""

import os, sys, csv, gc
import numpy as np
import cv2
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'OEM_v2_aDanh'))
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'analysis_output', 'advanced'))

NUM_CLASSES = 8
IGNORE_INDEX = 255
H, W = 1024, 1024  # standard size

CLASS_NAMES = ['Bareland', 'Rangeland', 'Developed', 'Road', 'Tree', 'Water', 'Agriculture', 'Building']
CLASS_COLORS_RGB = np.array([
    [128,0,0],[0,255,36],[148,148,148],[255,255,255],
    [34,97,38],[0,69,255],[75,181,73],[222,31,7]], dtype=np.uint8)


def remap_label(raw):
    out = raw.astype(np.int32)
    out = np.where(out == 0, IGNORE_INDEX, out - 1)
    out = np.clip(out, 0, NUM_CLASSES - 1)
    out[raw == 0] = IGNORE_INDEX
    return out


def read_label(path):
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


def load_split(split):
    with open(os.path.join(DATA_ROOT, f'{split}.txt')) as f:
        return [l.strip() for l in f if l.strip()]


def iterate_samples():
    """Yield (filename, split, pseudo, gt) one at a time — no bulk storage."""
    for split in ['train', 'val', 'test']:
        filenames = load_split(split)
        for fn in filenames:
            pseudo_raw = read_label(os.path.join(DATA_ROOT, 'pseudolabels', fn))
            gt_raw = read_label(os.path.join(DATA_ROOT, 'labels', fn))
            if pseudo_raw is None or gt_raw is None:
                continue
            yield fn, split, remap_label(pseudo_raw), remap_label(gt_raw)


# ═══════════════════════════════════════════════════════════════════════
# SINGLE-PASS: collect all accumulators in one sweep
# ═══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    diff_dir = os.path.join(OUTPUT_DIR, 'difference_maps')
    region_dir = os.path.join(OUTPUT_DIR, 'region_analysis')
    dr_dir = os.path.join(OUTPUT_DIR, 'dim_reduction')
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(region_dir, exist_ok=True)
    os.makedirs(dr_dir, exist_ok=True)

    print(f"[INFO] Data root: {DATA_ROOT}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")
    print(f"[INFO] Processing images one-by-one (memory-efficient)...\n")

    # ── Accumulators (lightweight: only floats/ints, no raw images) ──

    # A. Difference maps: spatial heatmaps (float64, 8×1024×1024 × 4 arrays ≈ 256 MB)
    fn_heatmaps = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)  # float32 to save RAM
    fp_heatmaps = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    gt_presence = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    pred_presence = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    # A. Class transition totals
    transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    transition_image_count = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # B. Region-level accumulators
    gt_region_sizes_all = defaultdict(list)     # class → list of sizes (ints, ~few MB total)
    region_accuracy_all = defaultdict(list)     # class → list of accuracy floats
    size_bins = [(0,100),(100,500),(500,2000),(2000,10000),(10000,50000),(50000,float('inf'))]
    size_bin_names = ['tiny(<100)','small(100-500)','medium(500-2k)','large(2k-10k)','xlarge(10k-50k)','huge(>50k)']
    size_bin_correct = {bn: 0 for bn in size_bin_names}
    size_bin_total = {bn: 0 for bn in size_bin_names}
    size_bin_count = {bn: 0 for bn in size_bin_names}
    fragmentation_ratios = []

    # C. Feature vectors for dim reduction (51 floats × 2189 ≈ 900 KB)
    features_list = []
    metadata_list = []

    # CSV writers for per-image outputs (stream to disk immediately)
    diff_csv_path = os.path.join(diff_dir, 'per_image_difference.csv')
    region_csv_path = os.path.join(region_dir, 'per_image_regions.csv')

    diff_csv_file = open(diff_csv_path, 'w', newline='')
    region_csv_file = open(region_csv_path, 'w', newline='')
    diff_writer = None
    region_writer = None

    img_count = 0

    for fn, split, pseudo, gt in iterate_samples():
        h, w = gt.shape
        if h != H or w != W:
            pseudo = cv2.resize(pseudo.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        valid = (gt != IGNORE_INDEX) & (pseudo != IGNORE_INDEX)
        total_valid = int(valid.sum())
        if total_valid == 0:
            continue

        img_count += 1
        diff_mask = (pseudo != gt) & valid

        # Region name
        parts = fn.rsplit('_', 1)
        region_name = parts[0] if len(parts) > 1 else fn.replace('.tif', '')

        # ── Per-image difference row ──
        diff_row = {'filename': fn, 'split': split, 'total_diff_pixels': int(diff_mask.sum())}

        # ── Per-image region row ──
        region_row = {'filename': fn, 'split': split}
        total_gt_regions = 0
        total_pseudo_regions = 0

        # ── Feature vector for dim reduction ──
        feat = []

        for c in range(NUM_CLASSES):
            gt_c = (gt == c) & valid
            pred_c = (pseudo == c) & valid
            fn_c = gt_c & (pseudo != c)
            fp_c = pred_c & (gt != c)

            # A. Heatmaps
            fn_heatmaps[c] += fn_c.astype(np.float32)
            fp_heatmaps[c] += fp_c.astype(np.float32)
            gt_presence[c] += gt_c.astype(np.float32)
            pred_presence[c] += pred_c.astype(np.float32)

            fn_count = int(fn_c.sum())
            fp_count = int(fp_c.sum())
            gt_c_count = int(gt_c.sum())
            pred_c_count = int(pred_c.sum())
            tp = gt_c_count - fn_count

            diff_row[f'{CLASS_NAMES[c]}_fn'] = fn_count
            diff_row[f'{CLASS_NAMES[c]}_fp'] = fp_count

            # A. Transitions
            for c2 in range(NUM_CLASSES):
                if c2 != c and fn_count > 0:
                    trans = int(((gt == c) & (pseudo == c2) & valid).sum())
                    if trans > 0:
                        transition_matrix[c, c2] += trans
                        transition_image_count[c, c2] += 1

            # B. Connected components
            cc_gt_c = gt_c.astype(np.uint8)
            cc_pred_c = pred_c.astype(np.uint8)
            n_gt_r, gt_labels = cv2.connectedComponents(cc_gt_c, connectivity=8)
            n_gt_r -= 1
            n_ps_r, _ = cv2.connectedComponents(cc_pred_c, connectivity=8)
            n_ps_r -= 1
            total_gt_regions += n_gt_r
            total_pseudo_regions += n_ps_r
            region_row[f'{CLASS_NAMES[c]}_gt_reg'] = n_gt_r
            region_row[f'{CLASS_NAMES[c]}_ps_reg'] = n_ps_r

            for r in range(1, n_gt_r + 1):
                rmask = gt_labels == r
                rsize = int(rmask.sum())
                gt_region_sizes_all[c].append(rsize)
                correct_in_r = int((rmask & (pseudo == c)).sum())
                acc = correct_in_r / rsize if rsize > 0 else 0
                region_accuracy_all[c].append(acc)
                for bi, (lo, hi) in enumerate(size_bins):
                    if lo <= rsize < hi:
                        size_bin_correct[size_bin_names[bi]] += correct_in_r
                        size_bin_total[size_bin_names[bi]] += rsize
                        size_bin_count[size_bin_names[bi]] += 1
                        break

            # C. Features
            noise_rate = fn_count / max(gt_c_count, 1) if gt_c_count > 0 else 0
            iou = tp / max(tp + fp_count + fn_count, 1)
            prec = tp / max(pred_c_count, 1) if pred_c_count > 0 else 0
            rec = tp / max(gt_c_count, 1) if gt_c_count > 0 else 0
            feat.extend([noise_rate, iou, prec, rec, gt_c_count / total_valid, pred_c_count / total_valid])

        # Overall features
        noisy = int(diff_mask.sum())
        noise_ratio = noisy / total_valid
        feat.append(noise_ratio)

        kernel = np.ones((3, 3), np.uint8)
        gt_u8 = gt.copy().astype(np.uint8)
        gt_u8[gt == IGNORE_INDEX] = 255
        eroded = cv2.erode(gt_u8, kernel, iterations=1)
        dilated = cv2.dilate(gt_u8, kernel, iterations=1)
        edge_mask = (eroded != dilated) & valid
        edge_err = int(((pseudo != gt) & edge_mask).sum())
        feat.append(edge_err / max(int(edge_mask.sum()), 1))
        feat.append(len(np.unique(gt[valid])) / NUM_CLASSES)

        features_list.append(feat)
        metadata_list.append({'filename': fn, 'split': split, 'region': region_name, 'noise_ratio': noise_ratio})

        # Region totals
        region_row['gt_regions'] = total_gt_regions
        region_row['ps_regions'] = total_pseudo_regions
        region_row['frag_ratio'] = f"{total_pseudo_regions / max(total_gt_regions, 1):.2f}"
        fragmentation_ratios.append(total_pseudo_regions / max(total_gt_regions, 1))

        # Write rows to CSV immediately (no accumulation in memory)
        if diff_writer is None:
            diff_writer = csv.DictWriter(diff_csv_file, fieldnames=list(diff_row.keys()))
            diff_writer.writeheader()
        diff_writer.writerow(diff_row)

        if region_writer is None:
            region_writer = csv.DictWriter(region_csv_file, fieldnames=list(region_row.keys()))
            region_writer.writeheader()
        region_writer.writerow(region_row)

        if img_count % 100 == 0:
            print(f"  [{img_count}] processed... (mem-efficient)")

    diff_csv_file.close()
    region_csv_file.close()
    print(f"\n[OK] Processed {img_count} images total")
    print(f"[OK] Per-image difference CSV → {diff_csv_path}")
    print(f"[OK] Per-image region CSV → {region_csv_path}")

    # ═══════════════════════════════════════════════════════════════════
    # POST-PROCESSING: generate outputs from accumulators
    # ═══════════════════════════════════════════════════════════════════

    # ── A. Difference maps ──────────────────────────────────────────────
    print("\n[PART A] Generating difference map images...")

    # Class transition CSV
    trans_csv = os.path.join(diff_dir, 'class_transitions.csv')
    with open(trans_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gt_class', 'pred_class', 'total_pixels', 'num_images', 'avg_per_image'])
        for c_gt in range(NUM_CLASSES):
            for c_pred in range(NUM_CLASSES):
                if c_gt == c_pred:
                    continue
                cnt = int(transition_matrix[c_gt, c_pred])
                nimgs = int(transition_image_count[c_gt, c_pred])
                if cnt > 0:
                    w.writerow([CLASS_NAMES[c_gt], CLASS_NAMES[c_pred], cnt, nimgs,
                                f"{cnt / max(nimgs, 1):.1f}"])
    print(f"  [OK] Class transitions → {trans_csv}")

    # Per-class heatmap images
    for c in range(NUM_CLASSES):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fn_rate = np.divide(fn_heatmaps[c], gt_presence[c],
                            out=np.zeros((H, W), dtype=np.float32), where=gt_presence[c] > 0)
        im0 = axes[0].imshow(fn_rate, cmap='Reds', vmin=0, vmax=1)
        axes[0].set_title(f'{CLASS_NAMES[c]} — False Negative Rate')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        fp_rate = np.divide(fp_heatmaps[c], pred_presence[c],
                            out=np.zeros((H, W), dtype=np.float32), where=pred_presence[c] > 0)
        im1 = axes[1].imshow(fp_rate, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title(f'{CLASS_NAMES[c]} — False Positive Rate')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(fn_heatmaps[c] + fp_heatmaps[c], cmap='hot')
        axes[2].set_title(f'{CLASS_NAMES[c]} — Total Error Count')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(diff_dir, f'diffmap_{CLASS_NAMES[c].lower()}.png'), dpi=120, bbox_inches='tight')
        plt.close()

    # Overview
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for c in range(NUM_CLASSES):
        ax = axes[c // 4, c % 4]
        fn_rate = np.divide(fn_heatmaps[c], gt_presence[c],
                            out=np.zeros((H, W), dtype=np.float32), where=gt_presence[c] > 0)
        ax.imshow(fn_rate, cmap='Reds', vmin=0, vmax=1)
        fn_total = int(fn_heatmaps[c].sum())
        gt_total = int(gt_presence[c].sum())
        rate = fn_total / max(gt_total, 1)
        ax.set_title(f'{CLASS_NAMES[c]}\nFN rate={rate:.2%}', fontsize=10)
        ax.axis('off')
    plt.suptitle('Per-class False Negative Rate Heatmaps', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(diff_dir, 'diffmap_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Heatmap images → {diff_dir}/diffmap_*.png")

    # Free heatmap memory
    del fn_heatmaps, fp_heatmaps, gt_presence, pred_presence
    gc.collect()

    # ── B. Region analysis plots ────────────────────────────────────────
    print("\n[PART B] Generating region analysis outputs...")

    # Region summary CSV
    region_summary_path = os.path.join(region_dir, 'region_summary.csv')
    with open(region_summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'gt_total_regions', 'mean_size', 'median_size',
                     'mean_region_accuracy', 'median_region_accuracy'])
        for c in range(NUM_CLASSES):
            sizes = np.array(gt_region_sizes_all[c]) if gt_region_sizes_all[c] else np.array([0])
            accs = np.array(region_accuracy_all[c]) if region_accuracy_all[c] else np.array([0])
            w.writerow([CLASS_NAMES[c], len(gt_region_sizes_all[c]),
                        f"{sizes.mean():.1f}", f"{np.median(sizes):.1f}",
                        f"{accs.mean():.4f}", f"{np.median(accs):.4f}"])
    print(f"  [OK] Region summary → {region_summary_path}")

    # Size-binned accuracy
    size_bin_csv = os.path.join(region_dir, 'size_binned_accuracy.csv')
    with open(size_bin_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['size_bin', 'num_regions', 'total_pixels', 'correct_pixels', 'accuracy'])
        for bn in size_bin_names:
            acc = size_bin_correct[bn] / max(size_bin_total[bn], 1)
            w.writerow([bn, size_bin_count[bn], size_bin_total[bn], size_bin_correct[bn], f"{acc:.4f}"])
    print(f"  [OK] Size-binned accuracy → {size_bin_csv}")

    # Region accuracy distribution plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for c in range(NUM_CLASSES):
        ax = axes[c // 4, c % 4]
        if region_accuracy_all[c]:
            accs = np.array(region_accuracy_all[c])
            ax.hist(accs, bins=20, range=(0, 1), color=CLASS_COLORS_RGB[c] / 255.0,
                    edgecolor='black', alpha=0.8)
            ax.axvline(accs.mean(), color='red', linestyle='--', label=f'mean={accs.mean():.2f}')
            ax.legend(fontsize=8)
        ax.set_title(f'{CLASS_NAMES[c]} ({len(region_accuracy_all[c])} regions)', fontsize=10)
        ax.set_xlabel('Region Accuracy')
    plt.suptitle('Per-Region Accuracy Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, 'region_accuracy_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Size vs accuracy scatter (sampled)
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in range(NUM_CLASSES):
        if gt_region_sizes_all[c]:
            sizes = np.array(gt_region_sizes_all[c])
            accs = np.array(region_accuracy_all[c])
            if len(sizes) > 3000:
                idx = np.random.default_rng(42).choice(len(sizes), 3000, replace=False)
                sizes, accs = sizes[idx], accs[idx]
            ax.scatter(sizes, accs, s=2, alpha=0.3, label=CLASS_NAMES[c],
                       color=CLASS_COLORS_RGB[c] / 255.0)
    ax.set_xscale('log')
    ax.set_xlabel('Region Size (pixels, log scale)')
    ax.set_ylabel('Region Accuracy')
    ax.set_title('Region Size vs Accuracy')
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, 'size_vs_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Size-binned bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    accs_binned = [size_bin_correct[bn] / max(size_bin_total[bn], 1) for bn in size_bin_names]
    counts = [size_bin_count[bn] for bn in size_bin_names]
    bars = ax.bar(range(len(size_bin_names)), accs_binned, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(size_bin_names)))
    ax.set_xticklabels(size_bin_names, rotation=30, ha='right')
    ax.set_ylabel('Pixel Accuracy')
    ax.set_title('Accuracy by Region Size')
    for i, (b, cnt) in enumerate(zip(bars, counts)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f'n={cnt:,}', ha='center', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, 'size_binned_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Fragmentation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fragmentation_ratios, bins=50, color='coral', edgecolor='black', alpha=0.8)
    ax.axvline(np.mean(fragmentation_ratios), color='red', linestyle='--',
               label=f'mean={np.mean(fragmentation_ratios):.2f}')
    ax.set_xlabel('Fragmentation Ratio (pseudo_regions / gt_regions)')
    ax.set_ylabel('Count')
    ax.set_title('Pseudo-label Fragmentation')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, 'fragmentation_ratio.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Region plots → {region_dir}/*.png")

    # Free region memory
    del gt_region_sizes_all, region_accuracy_all
    gc.collect()

    # ── C. Dimensionality reduction ─────────────────────────────────────
    print("\n[PART C] Dimensionality reduction...")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    X = np.array(features_list, dtype=np.float32)
    print(f"  Feature matrix: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    n_comp = min(10, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # Feature names
    feat_names = []
    for c in range(NUM_CLASSES):
        cn = CLASS_NAMES[c]
        feat_names.extend([f'{cn}_noise', f'{cn}_iou', f'{cn}_prec', f'{cn}_rec',
                           f'{cn}_gt_ratio', f'{cn}_pred_ratio'])
    feat_names.extend(['overall_noise', 'edge_error', 'class_diversity'])

    # PCA loadings CSV
    loadings_csv = os.path.join(dr_dir, 'pca_loadings.csv')
    with open(loadings_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature'] + [f'PC{i+1}' for i in range(n_comp)])
        for i, name in enumerate(feat_names):
            w.writerow([name] + [f"{pca.components_[j, i]:.4f}" for j in range(n_comp)])

    # PCA results CSV
    pca_csv = os.path.join(dr_dir, 'pca_results.csv')
    with open(pca_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'split', 'region', 'noise_ratio'] + [f'PC{i+1}' for i in range(n_comp)])
        for i, m in enumerate(metadata_list):
            w.writerow([m['filename'], m['split'], m['region'], f"{m['noise_ratio']:.4f}"] +
                       [f"{X_pca[i, j]:.4f}" for j in range(n_comp)])

    # PCA plots
    noise_arr = np.array([m['noise_ratio'] for m in metadata_list])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(1, len(explained)+1), explained, color='steelblue', edgecolor='black')
    ax1.plot(range(1, len(explained)+1), np.cumsum(explained), 'ro-', label='Cumulative')
    ax1.set_xlabel('PC')
    ax1.set_ylabel('Explained Variance')
    ax1.set_title(f'PCA Variance (PC1+PC2={explained[0]+explained[1]:.1%})')
    ax1.legend()

    sc = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=noise_arr, s=5, alpha=0.6, cmap='RdYlGn_r')
    plt.colorbar(sc, ax=ax2, label='Noise Ratio')
    ax2.set_xlabel(f'PC1 ({explained[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({explained[1]:.1%})')
    ax2.set_title('PCA: Noise Feature Space')
    plt.tight_layout()
    plt.savefig(os.path.join(dr_dir, 'pca_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # PCA by split
    fig, ax = plt.subplots(figsize=(8, 6))
    for s, color in [('train','blue'),('val','orange'),('test','green')]:
        mask = np.array([m['split'] == s for m in metadata_list])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5, alpha=0.5, color=color, label=s)
    ax.set_xlabel(f'PC1 ({explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%})')
    ax.set_title('PCA by Split')
    ax.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(dr_dir, 'pca_by_split.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # PCA loadings bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for pc_idx, ax in enumerate(axes):
        loadings = pca.components_[pc_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1][:15]
        names = [feat_names[i] for i in sorted_idx]
        vals = [loadings[i] for i in sorted_idx]
        colors = ['coral' if v < 0 else 'steelblue' for v in vals]
        ax.barh(range(len(names)), vals, color=colors, edgecolor='black')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'PC{pc_idx+1} Top Loadings ({explained[pc_idx]:.1%})')
    plt.tight_layout()
    plt.savefig(os.path.join(dr_dir, 'pca_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sc = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=noise_arr, s=5, alpha=0.6, cmap='RdYlGn_r')
    plt.colorbar(sc, ax=axes[0], label='Noise Ratio')
    axes[0].set_title('t-SNE by Noise Ratio')

    for s, color in [('train','blue'),('val','orange'),('test','green')]:
        mask = np.array([m['split'] == s for m in metadata_list])
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.5, color=color, label=s)
    axes[1].set_title('t-SNE by Split')
    axes[1].legend(markerscale=5)

    regions = [m['region'] for m in metadata_list]
    rcounts = defaultdict(int)
    for r in regions:
        rcounts[r] += 1
    top10 = sorted(rcounts, key=rcounts.get, reverse=True)[:10]
    cmap_tab = plt.cm.tab10
    for i, r in enumerate(top10):
        mask = np.array([m['region'] == r for m in metadata_list])
        axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=8, alpha=0.6,
                        color=cmap_tab(i/10), label=f'{r}({rcounts[r]})')
    other = np.array([m['region'] not in top10 for m in metadata_list])
    axes[2].scatter(X_tsne[other, 0], X_tsne[other, 1], s=3, alpha=0.2, color='gray', label='other')
    axes[2].set_title('t-SNE by Region')
    axes[2].legend(fontsize=7, markerscale=3, ncol=2)

    for ax in axes:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(dr_dir, 'tsne_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # t-SNE by dominant error class
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, m in enumerate(metadata_list):
        feat = features_list[i]
        # noise rates are at indices 0, 6, 12, 18, 24, 30, 36, 42
        noise_rates = [feat[c * 6] for c in range(NUM_CLASSES)]
        dom_class = np.argmax(noise_rates)
        m['dom_error'] = dom_class
    for c in range(NUM_CLASSES):
        mask = np.array([m['dom_error'] == c for m in metadata_list])
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=8, alpha=0.6,
                   color=CLASS_COLORS_RGB[c] / 255.0, label=CLASS_NAMES[c])
    ax.set_title('t-SNE by Dominant Error Class')
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(dr_dir, 'tsne_dominant_error.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save t-SNE CSV
    tsne_csv = os.path.join(dr_dir, 'tsne_results.csv')
    with open(tsne_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'split', 'region', 'noise_ratio', 'dominant_error', 'tsne_x', 'tsne_y'])
        for i, m in enumerate(metadata_list):
            w.writerow([m['filename'], m['split'], m['region'], f"{m['noise_ratio']:.4f}",
                        CLASS_NAMES[m['dom_error']], f"{X_tsne[i, 0]:.4f}", f"{X_tsne[i, 1]:.4f}"])

    print(f"  [OK] All dim reduction outputs → {dr_dir}/")

    print(f"\n{'='*60}")
    print(f"ALL ADVANCED OUTPUTS IN: {OUTPUT_DIR}")
    print(f"  A. difference_maps/   — heatmaps, transition CSV")
    print(f"  B. region_analysis/   — connected-component stats, size vs accuracy")
    print(f"  C. dim_reduction/     — PCA, t-SNE scatter, loadings")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
